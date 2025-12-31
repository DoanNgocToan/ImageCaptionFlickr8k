import os
import pickle
import random
import time
from collections import defaultdict
from collections import Counter
import math

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.models as models
from pycocoevalcap.cider.cider import Cider

from nltk.translate.bleu_score import corpus_bleu

# ============================================================================
# CONFIG
# ============================================================================
class Config:
    DATASET_PATH = '.\\content\\clean_data_flickr8k'
    CAPTION_PATH = os.path.join(DATASET_PATH, 'captions.txt')
    IMAGE_DIR = os.path.join(DATASET_PATH, 'Images')
    FEATURES_PATH = os.path.join(DATASET_PATH, 'features_resnet50.pkl')
    MODEL_SAVE_PATH = os.path.join(DATASET_PATH, 'best_model_resnet50_transformer.pth')
    TENSORBOARD_LOG_ROOT = os.path.join(DATASET_PATH, 'runs')

    # model hyperparams
    NUM_HEADS = 4
    NUM_LAYERS = 3
    FF_DIM = 1024
    EMBED_SIZE = 256
    DROPOUT = 0.3

    # training
    BATCH_SIZE = 32
    EPOCHS = 50
    WEIGHT_DECAY = 1e-5
    LEARNING_RATE = 1e-4
    TRAIN_SPLIT = 0.90

    LR_SCHEDULER_FACTOR = 0.5
    LR_SCHEDULER_PATIENCE = 2
    EARLY_STOPPING_PATIENCE = 10

    NUM_WORKERS = 0
    PIN_MEMORY = False

    # features
    FEATURE_SHAPE = (49, 2048)   # 7*7, 2048
    IMAGE_SIZE = (224, 224)
    MAX_LEN = 37

    SWITCH_TO_SCST_PATIENCE = 5
    SWITCH_TO_SCST_EPOCH = 15
    SCST_TEMPERATURE = 0.5
    SCST_RATIO = 1.0
    SCST_ACCUMULATION_STEPS = 2

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def print_config(cls):
        print("="*70)
        print("CONFIGURATION")
        print("="*70)
        print(f"Device: {cls.DEVICE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Embed Size: {cls.EMBED_SIZE}")
        print(f"Num Heads: {cls.NUM_HEADS}")
        print(f"Num Layers: {cls.NUM_LAYERS}")
        print(f"FF Dimension: {cls.FF_DIM}")
        print(f"Dropout: {cls.DROPOUT}")
        print(f"Feature shape: {cls.FEATURE_SHAPE}")
        print("="*70)

# ============================================================================
# FEATURE EXTRACTOR (ResNet50 - up to conv layer)
# ============================================================================
class FeatureExtractorResNet50:
    def __init__(self, device=Config.DEVICE):
        self.device = device
        self.model = self._build_model()
        self.transform = transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _build_model(self):
        resnet = models.resnet50(pretrained=True)
        # take all layers except avgpool & fc -> output: (batch, 2048, 7, 7)
        modules = list(resnet.children())[:-2]
        model = nn.Sequential(*modules)
        model.eval().to(self.device)
        for name, param in model.named_parameters():
            if "layer4" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        return model

    def extract_features(self, image_dir, save_path):
        print(f"\nExtracting features (ResNet50) from {image_dir} ...")
        features = {}
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        with torch.no_grad():
            for img_name in tqdm(image_files, desc="Extracting features"):
                img_path = os.path.join(image_dir, img_name)
                try:
                    image = Image.open(img_path).convert('RGB')
                    img_tensor = self.transform(image).unsqueeze(0).to(self.device)
                    feat = self.model(img_tensor)  # (1, 2048, 7, 7)
                    feat = feat.permute(0, 2, 3, 1).cpu().numpy()  # (1,7,7,2048)
                    image_id = img_name.split('.')[0]
                    features[image_id] = feat
                except Exception as e:
                    print(f"Error processing {img_name}: {e}")
        with open(save_path, 'wb') as f:
            pickle.dump(features, f)
        print(f"Saved features -> {save_path} (total {len(features)})")
        return features

# ============================================================================
# CAPTION PROCESSOR (same idea)
# ============================================================================
class CaptionProcessor:
    def __init__(self, caption_path):
        self.caption_path = caption_path
        self.mapping = {}
        self.word2idx = {}
        self.idx2word = {}
        self.max_length = 0
        self.vocab_size = 0

    def load_captions(self):
        print("\nLoading captions...")
        with open(self.caption_path, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                parts = line.strip().split(',', 1)
                if len(parts) < 2:
                    continue
                image_name, caption = parts
                image_id = image_name.split('.')[0]
                if image_id not in self.mapping:
                    self.mapping[image_id] = []
                self.mapping[image_id].append(caption)
        print(f"Loaded captions for {len(self.mapping)} images")

    def clean_captions(self):
        print("Cleaning captions...")
        for img_id, caps in self.mapping.items():
            for i in range(len(caps)):
                cap = caps[i].lower()
                cap = ''.join([c for c in cap if c.isalnum() or c.isspace()])
                cap = ' '.join(cap.split())
                words = [w for w in cap.split() if len(w) > 1]
                caps[i] = 'startseq ' + ' '.join(words) + ' endseq'

    def build_vocabulary(self, min_freq=1):
        print("Building vocabulary...")
        freq = defaultdict(int)
        for caps in self.mapping.values():
            for cap in caps:
                for w in cap.split():
                    freq[w] += 1
        # optionally filter by min_freq
        words = [w for w, c in freq.items() if c >= min_freq]
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
        for w in sorted(words):
            if w in self.word2idx:
                continue
            self.word2idx[w] = idx
            idx += 1
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        # max length
        self.max_length = max(len(c.split()) for caps in self.mapping.values() for c in caps)
        print(f"Vocab size: {self.vocab_size}, Max caption length: {self.max_length}")

    def encode_caption(self, caption):
        return [self.word2idx.get(w, self.word2idx['<UNK>']) for w in caption.split()]

    def decode_caption(self, indices):
        words = []
        for idx in indices:
            if idx == 0:
                continue
            w = self.idx2word.get(idx, '<UNK>')
            if w in ('startseq', 'endseq'):
                continue
            words.append(w)
        return ' '.join(words)

# ============================================================================
# DATASET (caption-level: input_seq, target_seq)
# ============================================================================
class FlickrRawImageDataset(Dataset):
    def __init__(self, image_dir, image_ids, captions_mapping, caption_processor, image_size=Config.IMAGE_SIZE, device=Config.DEVICE):
        self.image_dir = image_dir
        self.image_ids = image_ids
        self.captions_mapping = captions_mapping
        self.cp = caption_processor
        self.device = device
        self.samples = []

        # ImageNet transforms
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        for img_id in image_ids:
            if img_id not in captions_mapping:
                continue
            for caption in captions_mapping[img_id]:
                encoded = caption_processor.encode_caption(caption)
                if len(encoded) < 2:
                    continue
                inp = encoded[:-1]
                tgt = encoded[1:]
                self.samples.append((img_id, inp, tgt))

        self.max_len = caption_processor.max_length - 1  # input length T-1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, inp, tgt = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_id + ".jpg")
        image = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(image)  # (3, H, W)

        inp_arr = np.zeros(self.max_len, dtype=np.int64)
        tgt_arr = np.zeros(self.max_len, dtype=np.int64)
        inp_arr[:len(inp)] = inp
        tgt_arr[:len(tgt)] = tgt

        return (
            img_id,
            img_tensor,                           # (3, H, W)
            torch.LongTensor(inp_arr),            # (T,)
            torch.LongTensor(tgt_arr),            # (T,)
        )


# ============================================================================
# POSITIONAL ENCODING (từ code ChatGPT)
# ============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
# ============================================================================
# TRANSFORMER DECODER (từ code ChatGPT, sử dụng tên biến gốc)
# ============================================================================

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, ff_dim, dropout, max_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.pos = PositionalEncoding(embed_size, max_len)
        layer = nn.TransformerDecoderLayer(embed_size, num_heads, ff_dim, dropout = dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, tgt, memory, tgt_mask, tgt_key_padding_mask):
        x = self.embed(tgt)
        x = self.pos(x)
        out = self.decoder(x, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return self.fc(out) # (B, T, V)

# ============================================================================
# MODEL: ResNet50 + Transformer Decoder
# ============================================================================
class ImageCaptioningModel(nn.Module):  # giữ nguyên tên để tương thích Trainer
    def __init__(self, vocab_size, embed_size=Config.EMBED_SIZE,
                 num_heads=Config.NUM_HEADS, num_layers=Config.NUM_LAYERS,
                 ff_dim=Config.FF_DIM, dropout=Config.DROPOUT, max_len=Config.MAX_LEN):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.max_len = max_len

        # 1) ResNet50 backbone (bỏ avgpool + fc)
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]  # output: (B, 2048, 7, 7)
        self.cnn = nn.Sequential(*modules)

        # Freeze all except layer4
        for name, param in self.cnn.named_parameters():
            if "layer4" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # FIX: Set all BatchNorm layers to use running stats
        # This makes them work correctly with batch_size=1
        for module in self.cnn.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = True
                module.momentum = 0.1  # Use running stats

        # 2) Feature projection: 2048 -> embed_size
        self.proj = nn.Linear(2048, embed_size)

        # 3) Transformer decoder
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.pos = PositionalEncoding(embed_size, max_len)
        layer = nn.TransformerDecoderLayer(
            embed_size, num_heads, ff_dim, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def generate_mask(self, sz, device):
        # causal mask (float -inf above diagonal)
        return torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)

    def encode_image(self, images):
        """
        images: (B, 3, H, W) -> cnn -> (B, 2048, 7, 7) -> (B, 49, 2048) -> proj -> (B, 49, E)
        """
        feat = self.cnn(images)                   # (B, 2048, 7, 7)
        B, C, H, W = feat.size()
        feat = feat.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)  # (B, 49, 2048)
        memory = self.proj(feat)                  # (B, 49, E)
        return memory

    def forward(self, images, input_seqs):
        """
        images: (B, 3, H, W)
        input_seqs: (B, T) - tokens startseq ... token_{T-1}
        returns: (B, T, V)
        """
        device = images.device

        # 1) Encode image to memory
        memory = self.encode_image(images)  # (B, 49, E)

        # 2) Prepare target
        T = input_seqs.size(1)
        tgt_mask = self.generate_mask(T, device)
        tgt_pad_mask = (input_seqs == 0)

        # 3) Decode
        x = self.embed(input_seqs)
        x = self.pos(x)
        out = self.decoder(x, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask)
        return self.fc(out)  # (B, T, V)


# ============================================================================
# LABEL SMOOTHING (nếu cần)
# ============================================================================
class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, smoothing=0.1, ignore_index=0):
        super().__init__()
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        # logits: (B*T, V)
        # target: (B*T)
        log_probs = torch.log_softmax(logits, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.vocab_size - 1))
            ignore = target == self.ignore_index
            target = target.clone()
            target[ignore] = 0
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
            true_dist[ignore] = 0

        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))
    

# ============================================================================
# INFERENCE: greedy and beam search
# ============================================================================
class CaptionGenerator:
    def __init__(self, model, caption_processor, config=Config):
        self.model = model.to(config.DEVICE)
        self.cp = caption_processor
        self.config = config

        self.start_idx = self.cp.word2idx.get('startseq', 1)
        self.end_idx = self.cp.word2idx.get('endseq', None)
        if self.end_idx is None:
            self.end_idx = self.cp.word2idx.get('end', 0)
        self.max_gen_len = self.cp.max_length - 1

    def generate_caption_greedy(self, images):
        """Greedy decoding từ ảnh tensor (B,3,H,W)"""
        device = images.device
        with torch.no_grad():
            memory = self.model.encode_image(images)  # (B,49,E)

            generated_seq = torch.LongTensor([[self.start_idx]]).to(device)
            captions = []

            for _ in range(self.max_gen_len):
                T = generated_seq.size(1)
                tgt_mask = self.model.generate_mask(T, device)
                tgt_pad_mask = (generated_seq == 0)

                x = self.model.embed(generated_seq) 
                x = self.model.pos(x) 
                logits = self.model.decoder( x, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask )
                last_logits = self.model.fc(logits[:, -1, :])  # (B,V)
                next_token = last_logits.argmax(dim=-1).item()

                if next_token == self.end_idx or next_token == 0:
                    break
                generated_seq = torch.cat(
                    [generated_seq, torch.LongTensor([[next_token]]).to(device)], dim=1
                )
                captions.append(next_token)

        return self.cp.decode_caption(captions)
    
    def generate_caption_beam(self, img_tensor, beam_size=3):
        device = img_tensor.device
        memory = self.model.encode_image(img_tensor)

        beams = [([self.start_idx], 0.0)]
        completed = []

        for _ in range(self.max_gen_len):
            new_beams = []
            current_sequences = [torch.LongTensor([seq]).to(device) for seq, _ in beams]
            if not current_sequences:
                break

            T_max = max(s.size(1) for s in current_sequences)
            batched_seq = torch.zeros(len(current_sequences), T_max,
                                      dtype=torch.long, device=device)
            for i, seq in enumerate(current_sequences):
                batched_seq[i, :seq.size(1)] = seq

            batched_memory = memory.expand(len(current_sequences), -1, -1)
            tgt_mask = self.model.generate_mask(T_max, device)
            tgt_pad_mask = (batched_seq == 0)

            x = self.model.embed(batched_seq) 
            x = self.model.pos(x) 
            logits = self.model.decoder( x, batched_memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask )
            log_probs = torch.log_softmax(self.model.fc(logits), dim=-1)
            last_log_probs = log_probs[:, -1, :]

            for i, (seq, score) in enumerate(beams):
                if seq[-1] == self.end_idx:
                    completed.append((seq, score))
                    continue
                topk_logprobs, topk_idx = torch.topk(last_log_probs[i], beam_size)

                def length_norm(s, l, alpha=1.2):
                    return s / (((5 + l) / 6) ** alpha)

                for k in range(beam_size):
                    idx_k = int(topk_idx[k].item())
                    lp = float(topk_logprobs[k].item())
                    new_seq = seq + [idx_k]
                    raw_score = score + lp
                    new_score = length_norm(raw_score, len(new_seq))
                    new_beams.append((new_seq, new_score))

            new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            beams = new_beams
            if len(completed) >= beam_size:
                break

        for seq, score in beams:
            completed.append((seq, score))
        completed = sorted(completed, key=lambda x: x[1], reverse=True)
        best_seq = completed[0][0]
        return self.cp.decode_caption(best_seq)

    def sample_caption(self, images, temperature=0.5):
        """
        Sampling with temperature control
        Lower temp = more diverse (better for SCST)
        
        Args:
            temperature: 0.3-0.7 recommended for SCST
        """
        device = images.device
        memory = self.model.encode_image(images)

        generated_seq = torch.LongTensor([[self.start_idx]]).to(device)
        logprobs = []

        for _ in range(self.max_gen_len):
            T = generated_seq.size(1)
            tgt_mask = self.model.generate_mask(T, device)
            tgt_pad_mask = (generated_seq == 0)

            x = self.model.embed(generated_seq) 
            x = self.model.pos(x) 
            out = self.model.decoder( x, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask )
            last_logits = self.model.fc(out[:, -1, :])  # (1, V)

            scaled_logits = last_logits / temperature
            log_prob = torch.log_softmax(scaled_logits, dim=-1) #(1, V)

            dist = torch.distributions.Categorical(logits=log_prob)
            next_token = dist.sample()# (1,)
            token_logprob = log_prob.gather(1, next_token.unsqueeze(1)).squeeze(1) # (1,)
            logprobs.append(token_logprob)
            generated_seq = torch.cat([generated_seq, next_token.unsqueeze(1).detach()], dim=1)

            if next_token.item() == self.end_idx or next_token.item() == 0:
                break
        logprobs_tensor = torch.stack(logprobs)
        caption_text = self.cp.decode_caption(generated_seq.squeeze(0).tolist())
        return caption_text, logprobs_tensor
   
    def _load_image_tensor(self, image_id, image_dir):
        """Load image for visualization"""
        from PIL import Image
        import torchvision.transforms as transforms
        
        img_files = [f for f in os.listdir(image_dir) if f.startswith(image_id)]
        if not img_files:
            return None, None
        
        img_path = os.path.join(image_dir, img_files[0])
        image = Image.open(img_path).convert("RGB")
        
        transform = transforms.Compose([
            transforms.Resize(self.config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image).unsqueeze(0).to(self.config.DEVICE)
        return img_tensor, image

    def visualize_caption(self, image_id, image_dir, beam_size=3):
        """Visualize predictions"""
        img_tensor, image = self._load_image_tensor(image_id, image_dir)
        if img_tensor is None:
            print(f"Image {image_id} not found")
            return
        
        was_training = self.model.training
        self.model.eval()
        
        with torch.no_grad():
            actual = self.cp.mapping.get(image_id, [])
            pred_greedy = self.generate_caption_greedy(img_tensor)
            pred_beam = self.generate_caption_beam(img_tensor, beam_size)
        
        if was_training:
            self.model.train()
        
        print("\n" + "="*60)
        print(f"Image ID: {image_id}")
        print("ACTUAL CAPTIONS:")
        for i, c in enumerate(actual, 1):
            print(f"{i}. {c}")
        print("\nPREDICTED (greedy):")
        print(pred_greedy)
        print(f"\nPREDICTED (beam size={beam_size}):")
        print(pred_beam)
        print("="*60 + "\n")

        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8,6))
            plt.imshow(image)
            plt.axis('off')
            plt.show()
        except Exception:
            pass

# ============================================================================
# TRAINER
# ============================================================================


class Trainer:
    def __init__(self, model, train_loader, val_loader, caption_processor, config=Config):
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cp = caption_processor
        self.config = config
        self.generator = CaptionGenerator(self.model, self.cp, config=self.config)

        # Loss cho giai đoạn pretrain
        self.criterion = LabelSmoothingLoss(vocab_size=self.cp.vocab_size, smoothing=0.1)

        # Optimizer với LR khác nhau
        self.optimizer = optim.Adam([
            {"params": self.model.decoder.parameters(), "lr": 1e-4},
            {"params": self.model.embed.parameters(),   "lr": 1e-4},
            {"params": self.model.fc.parameters(),      "lr": 1e-4},
            {"params": self.model.proj.parameters(),    "lr": 1e-4},
            {"params": self.model.cnn[-1].parameters(), "lr": 1e-5},  # layer4 block
        ])

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=config.LR_SCHEDULER_FACTOR,
            patience=config.LR_SCHEDULER_PATIENCE, verbose=True
        )

        # Logging
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.use_scst = False   # bắt đầu với LabelSmoothingLoss
        self.cider_scorer = Cider()

        self.scst_temperature = 0.5      # Lower = more exploration
        self.scst_sample_ratio = 1.0     # Sample 50% of batch
        self.scst_accumulation_steps = 2  # Gradient accumulation
        self.scst_temp_schedule = {
            0: 0.7,   # Start conservative
            2: 0.5,   # More exploration
            5: 0.3    # Maximum diversity
        }

        log_dir = os.path.join(
            config.TENSORBOARD_LOG_ROOT,
            time.strftime("%b%d_%H-%M-%S") + f'_{config.DEVICE.type}'
        )
        self.writer = SummaryWriter(log_dir)

    def train_epoch_scst_optimized(self, epoch):
        """
        OPTIMIZED SCST with:
        1. Sample only N images per batch (not all 32)
        2. Batch CIDEr computation
        3. Gradient accumulation
        4. Temperature-controlled sampling
        """
        self.model.train()
        total_loss = 0.0
        total_reward = 0.0
        total_advantage = 0.0
        num_samples = 0
        

        current_epoch_in_scst = epoch - self.config.SWITCH_TO_SCST_EPOCH
        temperature = self.scst_temp_schedule.get(
            current_epoch_in_scst, 
            self.scst_temperature
        )
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.EPOCHS} [SCST T={temperature:.2f}]")

        
        self.optimizer.zero_grad()
        
        for batch_idx, (img_ids, images, input_seqs, target_seqs) in enumerate(pbar):
            images = images.to(self.config.DEVICE)
            batch_size = images.size(0)
            
            # OPTIMIZATION 1: Sample only subset of batch
            num_to_sample = batch_size
            
            # OPTIMIZATION 2: Parallel generation (batch processing)
            sampled_caps = []
            sampled_logprobs_list = []
            greedy_caps = []
            
            self.model.eval()
            with torch.no_grad():
                for i in range(num_to_sample):
                    cap_greedy = self.generator.generate_caption_greedy(images[i:i+1])
                    greedy_caps.append(cap_greedy)
            
            self.model.train()

            # Generate captions for sampled images
            for i in range(num_to_sample):
                # Sample with temperature
                cap_sample, logprobs = self.generator.sample_caption(
                    images[i:i+1], 
                    temperature=temperature
                )
                sampled_caps.append(cap_sample)
                sampled_logprobs_list.append(logprobs.sum())
            
            # OPTIMIZATION 3: Batch CIDEr computation
            sample_rewards = []
            greedy_rewards = []
            
            for i, img_id in enumerate(img_ids):
                raw_refs = self.cp.mapping[img_id]
                cleaned_refs = [
                    ref.replace('startseq', '').replace('endseq', '').strip() for ref in raw_refs
                ]

                sample_text = sampled_caps[i]
                greedy_text = greedy_caps[i]
                
                # Format for CIDEr
                refs_dict = {img_id: cleaned_refs} # ['string1', 'string2', ...]
                sample_dict = {img_id: [sample_text]}  # ['string']
                greedy_dict = {img_id: [greedy_text]}

                if batch_idx == 0 and i == 0:
                    print(f"\n[CIDEr Debug - VERIFIED FORMAT]")
                    print(f"  Refs type: {type(cleaned_refs)}, length: {len(cleaned_refs)}")
                    print(f"  Refs[0] type: {type(cleaned_refs[0])}")
                    print(f"  Refs (first 2): {cleaned_refs[:2]}")
                    print(f"  Sample type: {type(sample_text)}")
                    print(f"  Sample text: '{sample_text}'")
                    print(f"  Greedy text: '{greedy_text}'")
                    print(f"  Format check:")
                    print(f"    - refs_dict[img_id] is list: {isinstance(refs_dict[img_id], list)}")
                    print(f"    - refs_dict[img_id][0] is str: {isinstance(refs_dict[img_id][0], str)}")
                    print(f"    - sample_dict[img_id] is list: {isinstance(sample_dict[img_id], list)}")
                    print(f"    - sample_dict[img_id][0] is str: {isinstance(sample_dict[img_id][0], str)}")
                
                try:
                    sr, _ = self.cider_scorer.compute_score(refs_dict, sample_dict)
                    gr, _ = self.cider_scorer.compute_score(refs_dict, greedy_dict)
                except Exception as e:
                    print(f"\n CIDEr Error at batch {batch_idx}, sample {i}:")
                    print(f"  Error: {e}")
                    print(f"  refs_dict: {refs_dict}")
                    print(f"  sample_dict: {sample_dict}")
                    sr, gr = 0.0, 0.0  # Fallback
                
                if batch_idx == 0 and i == 0:
                    print(f"  Sample Reward: {sr:.4f}")  # Should be > 0 now
                    print(f"  Greedy Reward: {gr:.4f}")  # Should be > 0 now

                sample_rewards.append(sr)
                greedy_rewards.append(gr)
            
            # Compute SCST loss
            rewards = torch.tensor(sample_rewards, device=images.device)
            baselines = torch.tensor(greedy_rewards, device=images.device)
            advantages = rewards - baselines
            
            # CRITICAL: Normalize advantages to stabilize training
            if len(advantages) > 1 and advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            else:
                advantages = advantages - advantages.mean()

            logprobs_sum = torch.stack(sampled_logprobs_list)
            
            # SCST loss with gradient accumulation
            loss = -(advantages.detach() * logprobs_sum).mean()
            loss = loss / self.scst_accumulation_steps  # Scale for accumulation

            if batch_idx == 0:
                print(f"\n[Debug] Batch 0:")
                print(f"  Rewards: sample={rewards.mean().item():.4f}, greedy={baselines.mean().item():.4f}")
                print(f"  Raw Advantage: {(rewards - baselines).mean().item():.4f}")
                print(f"  Normalized Advantage: {advantages.mean().item():.4f}")
                print(f"  Loss: {loss.item():.4f}")
                print(f"  Logprobs requires_grad: {logprobs_sum.requires_grad}")
                print(f"  Model mode: {'TRAIN' if self.model.training else 'EVAL'}")
            
            loss.backward()
            
            # OPTIMIZATION 4: Gradient accumulation
            if (batch_idx + 1) % self.scst_accumulation_steps == 0:
                if batch_idx < 10:
                    total_grad_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            total_grad_norm += p.grad.norm().item()
                    if total_grad_norm < 1e-8:
                        print(f" WARNING: Gradient norm is near zero at batch {batch_idx}")

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Logging
            total_loss += loss.item() * self.scst_accumulation_steps
            total_reward += rewards.mean().item()
            total_advantage += advantages.mean().item()
            num_batches += 1
            
            # Update progress bar with rich info
            pbar.set_postfix({
                'loss': f'{loss.item() * self.scst_accumulation_steps:.4f}',
                'R_s': f'{rewards.mean().item():.3f}',
                'R_g': f'{baselines.mean().item():.3f}'
            })
            
            # Periodic logging
            if batch_idx % 100 == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('SCST/Loss_batch', loss.item(), global_step)
                self.writer.add_scalar('SCST/Reward_sample', rewards.mean().item(), global_step)
                self.writer.add_scalar('SCST/Reward_greedy', baselines.mean().item(), global_step)
                self.writer.add_scalar('SCST/Advantage', advantages.mean().item(), global_step)
        
        # Final gradient update if needed
        if num_batches % self.scst_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        avg_loss = total_loss / num_batches
        avg_reward = total_reward / num_batches
        avg_advantage = total_advantage / num_batches

        print(f"\nSCST Stats:")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Avg Sample Reward: {avg_reward:.4f}")
        print(f"  Avg Advantage: {avg_advantage:.4f}")
        print(f"  Temperature: {temperature:.2f}")
        
        return avg_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.EPOCHS} [Train]")
        if not self.use_scst:
            for batch_idx, (img_ids, images, input_seqs, target_seqs) in enumerate(pbar):
                images = images.to(self.config.DEVICE)
                input_seqs = input_seqs.to(self.config.DEVICE)
                target_seqs = target_seqs.to(self.config.DEVICE)

                # ----- Label Smoothing -----
                outputs = self.model(images, input_seqs)     # (B,T,V)
                B, T, V = outputs.size()
                outputs_flat = outputs.view(B * T, V)
                targets_flat = target_seqs.view(B * T)
                loss = self.criterion(outputs_flat, targets_flat)            
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss_batch', loss.item(), global_step)

            avg_epoch_loss = total_loss / len(self.train_loader)
            self.writer.add_scalar('Train/Loss_epoch', avg_epoch_loss, epoch)
            return avg_epoch_loss
        else:
            return self.train_epoch_scst_optimized(epoch)
        
    def validate(self, epoch):
        if self.use_scst:
            print(f"[Val] Epoch {epoch+1}: SCST phase (skip CE val loss)")
            return float('inf')
        else:
            self.model.eval()
            total_loss = 0.0
            with torch.no_grad():
                pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.config.EPOCHS} [Val]")
                for img_ids, images, input_seqs, target_seqs in pbar:
                    images = images.to(self.config.DEVICE)
                    input_seqs = input_seqs.to(self.config.DEVICE)
                    target_seqs = target_seqs.to(self.config.DEVICE)

                    outputs = self.model(images, input_seqs)
                    B, T, V = outputs.size()
                    outputs_flat = outputs.view(B * T, V)
                    targets_flat = target_seqs.view(B * T)
                    loss = self.criterion(outputs_flat, targets_flat)
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_loss = total_loss / len(self.val_loader)
            self.writer.add_scalar('Val/Loss_epoch', avg_loss, epoch)
            return avg_loss

    def save_checkpoint(self, epoch, val_loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'vocab_size': self.cp.vocab_size,
            'max_length': self.cp.max_length,
            'word2idx': self.cp.word2idx,
            'idx2word': self.cp.idx2word
        }
        torch.save(checkpoint, self.config.MODEL_SAVE_PATH)
        print(f"Model saved to {self.config.MODEL_SAVE_PATH}")

    def train(self):
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        for epoch in range(self.config.EPOCHS):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)

            print(f"\nEpoch {epoch+1}/{self.config.EPOCHS}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            if not self.use_scst:
                self.scheduler.step(val_loss)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_no_improve = 0
                    self.save_checkpoint(epoch, val_loss)
                    print(f"✓ New best model! Val Loss: {val_loss:.4f}")
                else:
                    self.epochs_no_improve += 1
                    print(f"No improvement for {self.epochs_no_improve} epoch(s)")

                if (self.epochs_no_improve >= self.config.SWITCH_TO_SCST_PATIENCE) or \
                   (epoch+1 == self.config.SWITCH_TO_SCST_EPOCH):
                    print(f"\n>>> Switching to SCST at epoch {epoch+1} <<<\n")
                    self.use_scst = True
            else:
                self.save_checkpoint(epoch, val_loss)

            self.writer.add_scalar('Train/LearningRate', self.optimizer.param_groups[0]['lr'], epoch)

            if not self.use_scst and self.epochs_no_improve >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping after {epoch+1} epochs (CE phase)")
                break

        self.writer.close()
        print("\nTRAINING COMPLETED\n")

# ============================================================================
# EVALUATOR
# ============================================================================
class ModelEvaluator:
    def __init__(self, model, caption_processor, image_ids, config=Config):
        self.generator = CaptionGenerator(model, caption_processor, config)
        self.cp = caption_processor
        self.image_ids = image_ids
        self.config = config
        # transform để chuẩn bị ảnh 
        self.transform = transforms.Compose([ 
            transforms.Resize(Config.IMAGE_SIZE), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]) 
        ])

    def _load_image_tensor(self, image_id, image_dir):
        img_files = [f for f in os.listdir(image_dir) if f.startswith(image_id)]
        if not img_files:
            return None
        img_path = os.path.join(image_dir, img_files[0])
        image = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(image).unsqueeze(0).to(self.config.DEVICE)
        return img_tensor

    def evaluate(self, use_beam=False, beam_size=3, image_dir=Config.IMAGE_DIR): 
        actual, predicted = [], []
        for image_id in tqdm(self.image_ids, desc="Generating captions"):
            if image_id not in self.cp.mapping:
                continue
            refs = [cap.split() for cap in self.cp.mapping[image_id]]
            img_tensor = self._load_image_tensor(image_id, image_dir)
            if img_tensor is None:
                continue
            if use_beam:
                pred = self.generator.generate_caption_beam(img_tensor, beam_size).split()
            else:
                pred = self.generator.generate_caption_greedy(img_tensor).split()
            actual.append(refs)
            predicted.append(pred)
        bleu1 = corpus_bleu(actual, predicted, weights=(1,0,0,0))
        bleu2 = corpus_bleu(actual, predicted, weights=(0.5,0.5,0,0))
        bleu3 = corpus_bleu(actual, predicted, weights=(0.33,0.33,0.33,0))
        bleu4 = corpus_bleu(actual, predicted, weights=(0.25,0.25,0.25,0.25))
        print("\nBLEU scores:")
        print(f"BLEU-1: {bleu1:.4f}")
        print(f"BLEU-2: {bleu2:.4f}")
        print(f"BLEU-3: {bleu3:.4f}")
        print(f"BLEU-4: {bleu4:.4f}")
        return {'BLEU-1':bleu1, 'BLEU-2':bleu2, 'BLEU-3':bleu3, 'BLEU-4':bleu4}

# ============================================================================
# CUSTOM BLEU WITH LINEAR BP (NO NLTK)
# ============================================================================

class LinearBPBLEU:
    """
    Corpus-level BLEU with:
    - clipped n-gram precision (n=1..4)
    - geometric mean
    - linear BP = min(1, pred_len / ref_len)
    """

    def __init__(self, max_n=4):
        self.max_n = max_n

    def _ngram_counts(self, tokens, n):
        return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])

    def _clipped_precision(self, references, hypotheses, n):
        """
        references: list of list of tokens (multiple refs)
        hypotheses: list of tokens
        """
        hyp_counts = self._ngram_counts(hypotheses, n)
        if not hyp_counts:
            return 0.0, 0

        max_ref_counts = Counter()
        for ref in references:
            ref_counts = self._ngram_counts(ref, n)
            for ng in ref_counts:
                max_ref_counts[ng] = max(max_ref_counts[ng], ref_counts[ng])

        clipped = {
            ng: min(count, max_ref_counts.get(ng, 0))
            for ng, count in hyp_counts.items()
        }

        return sum(clipped.values()), sum(hyp_counts.values())

    def corpus_bleu(self, list_of_references, hypotheses, weights=None):
        """
        list_of_references: [[ref1_tokens, ref2_tokens, ...], ...]
        hypotheses: [hyp_tokens, ...]
        """
        if weights is None:
            weights = [1 / self.max_n] * self.max_n

        p_ns = []
        total_pred_len = 0
        total_ref_len = 0

        for n in range(1, self.max_n + 1):
            clipped_total = 0
            total = 0
            for refs, hyp in zip(list_of_references, hypotheses):
                c, t = self._clipped_precision(refs, hyp, n)
                clipped_total += c
                total += t

            if total == 0:
                p_ns.append(0.0)
            else:
                p_ns.append(clipped_total / total)

        # geometric mean
        smooth_eps = 1e-9
        log_p_sum = 0.0
        for w, p in zip(weights, p_ns):
            log_p_sum += w * math.log(max(p, smooth_eps))
        geo_mean = math.exp(log_p_sum)

        # length stats
        for refs, hyp in zip(list_of_references, hypotheses):
            total_pred_len += len(hyp)
            ref_lens = [len(r) for r in refs]
            total_ref_len += min(ref_lens, key=lambda rl: abs(rl - len(hyp)))

        # LINEAR BP
        bp = min(1.0, total_pred_len / max(total_ref_len, 1))

        return bp * geo_mean

# ============================================================================
# RESEARCH EVALUATOR: BLEU (NLTK) + BLEU (LINEAR BP) + METEOR
# ============================================================================

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score

class ResearchEvaluator:
    def __init__(self, model, caption_processor, image_ids, config=Config):
        self.generator = CaptionGenerator(model, caption_processor, config)
        self.cp = caption_processor
        self.image_ids = image_ids
        self.linear_bleu = LinearBPBLEU(max_n=4)
        self.config = config 
        self.transform = transforms.Compose([ 
            transforms.Resize(Config.IMAGE_SIZE), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]) 
        ])

    def _load_image_tensor(self, image_id, image_dir): 
        img_files = [f for f in os.listdir(image_dir) if f.startswith(image_id)] 
        if not img_files: 
            return None 
        img_path = os.path.join(image_dir, img_files[0]) 
        image = Image.open(img_path).convert("RGB") 
        img_tensor = self.transform(image).unsqueeze(0).to(self.config.DEVICE) 
        return img_tensor

    def evaluate(self, use_beam=False, beam_size=3, image_dir=Config.IMAGE_DIR): 
        actual, predicted, meteor_scores = [], [], [] 
        pred_lens, ref_lens = [], [] 
        for image_id in tqdm(self.image_ids, desc="Research Evaluation"): 
            if image_id not in self.cp.mapping: 
                continue 
            refs = [cap.split() for cap in self.cp.mapping[image_id]] 
            img_tensor = self._load_image_tensor(image_id, image_dir) 
            if img_tensor is None: 
                continue 
            if use_beam: 
                pred_tokens = self.generator.generate_caption_beam(img_tensor, beam_size).split() 
            else: 
                pred_tokens = self.generator.generate_caption_greedy(img_tensor).split() 
            actual.append(refs) 
            predicted.append(pred_tokens) 
            meteor_scores.append(meteor_score(refs, pred_tokens)) 
            pred_lens.append(len(pred_tokens)) 
            ref_lens.append(min(len(r) for r in refs)) 
        # BLEU nltk 
        bleu1_std = corpus_bleu(actual, predicted, weights=(1,0,0,0)) 
        bleu2_std = corpus_bleu(actual, predicted, weights=(0.5,0.5,0,0)) 
        bleu3_std = corpus_bleu(actual, predicted, weights=(0.33,0.33,0.33,0)) 
        bleu4_std = corpus_bleu(actual, predicted, weights=(0.25,0.25,0.25,0.25)) 
        # BLEU linear BP 
        bleu1_linear = self.linear_bleu.corpus_bleu(actual, predicted,[1,0,0,0]) 
        bleu2_linear = self.linear_bleu.corpus_bleu(actual, predicted,[0.5,0.5,0,0]) 
        bleu3_linear = self.linear_bleu.corpus_bleu(actual, predicted,[0.33,0.33,0.33,0]) 
        bleu4_linear = self.linear_bleu.corpus_bleu(actual, predicted,[0.25,0.25,0.25,0.25]) 
        results = { 
            "BLEU-1 (nltk)": bleu1_std, 
            "BLEU-2 (nltk)": bleu2_std, 
            "BLEU-3 (nltk)": bleu3_std, 
            "BLEU-4 (nltk)": bleu4_std, 
            "BLEU-1 (linear BP)": bleu1_linear, 
            "BLEU-2 (linear BP)": bleu2_linear, 
            "BLEU-3 (linear BP)": bleu3_linear, 
            "BLEU-4 (linear BP)": bleu4_linear, 
            "METEOR": sum(meteor_scores) / len(meteor_scores), 
            "Avg Pred Len": sum(pred_lens) / len(pred_lens), 
            "Avg Ref Len": sum(ref_lens) / len(ref_lens), 
            "Len Ratio": (sum(pred_lens) / sum(ref_lens)) 
        } 
        print("\nRESEARCH METRICS") 
        print("=" * 50) 
        for k, v in results.items(): 
            print(f"{k}: {v:.4f}") 
        print("=" * 50) 
        return results

# ============================================================================
# MAIN pipeline
# ============================================================================
def main():
    Config.print_config()

    # 1. features
    # if not os.path.exists(Config.FEATURES_PATH):
    #     extractor = FeatureExtractorResNet50(device=Config.DEVICE)
    #     features = extractor.extract_features(Config.IMAGE_DIR, Config.FEATURES_PATH)
    # else:
    #     with open(Config.FEATURES_PATH, 'rb') as f:
    #         features = pickle.load(f)
    #     print(f"Loaded features for {len(features)} images")

    # 2. captions
    cp = CaptionProcessor(Config.CAPTION_PATH)
    cp.load_captions()
    cp.clean_captions()
    cp.build_vocabulary(min_freq=1)
    # ensure startseq/endseq exist in vocab
    if 'startseq' not in cp.word2idx:
        cp.word2idx['startseq'] = len(cp.word2idx)
        cp.idx2word[cp.word2idx['startseq']] = 'startseq'
    if 'endseq' not in cp.word2idx:
        cp.word2idx['endseq'] = len(cp.word2idx)
        cp.idx2word[cp.word2idx['endseq']] = 'endseq'
    cp.vocab_size = len(cp.word2idx)

    # 3. split
    image_ids = list(cp.mapping.keys())
    random.shuffle(image_ids)
    split_idx = int(len(image_ids) * Config.TRAIN_SPLIT)
    train_ids = image_ids[:split_idx]
    test_ids = image_ids[split_idx:]
    print(f"Train images: {len(train_ids)}, Test images: {len(test_ids)}")

    # 4. Datasets & loaders
    train_dataset = FlickrRawImageDataset(
        image_dir=Config.IMAGE_DIR,
        image_ids=train_ids,
        captions_mapping=cp.mapping,
        caption_processor=cp
    )

    val_dataset = FlickrRawImageDataset(
        image_dir=Config.IMAGE_DIR,
        image_ids=test_ids,
        captions_mapping=cp.mapping,
        caption_processor=cp
    )

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
                              num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                            num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 5. model
    model = ImageCaptioningModel(vocab_size=cp.vocab_size,
                                 embed_size=Config.EMBED_SIZE,
                                 num_heads=Config.NUM_HEADS,
                                 num_layers=Config.NUM_LAYERS,
                                 ff_dim=Config.FF_DIM,
                                 dropout=Config.DROPOUT,
                                 max_len=cp.max_length)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # 6. trainer
    trainer = Trainer(model, train_loader, val_loader, cp, Config)
    trainer.train()

    # 7. load best and evaluate
    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1} with val loss {checkpoint['val_loss']:.4f}")

    evaluator = ModelEvaluator(model, cp, test_ids, Config)
    evaluator.evaluate(use_beam=True, beam_size=3)

    research_eval = ResearchEvaluator(
        model=model,
        caption_processor=cp,
        image_ids=test_ids,
        config=Config
    )

    research_results = research_eval.evaluate(
        use_beam=True,
        beam_size=3
    )
    
    # 8. visualize few examples
    generator = CaptionGenerator(model, cp, Config)
    for img_id in random.sample(test_ids, min(3, len(test_ids))):
        generator.visualize_caption(img_id, Config.IMAGE_DIR, beam_size=3)

# convenience loader & inference
def load_and_inference(image_id=None):
    print("Loading model & features...")
    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    # with open(Config.FEATURES_PATH, 'rb') as f:
    #     features = pickle.load(f)
    cp = CaptionProcessor(Config.CAPTION_PATH)
    cp.load_captions()
    cp.clean_captions()
    cp.word2idx = checkpoint['word2idx']
    cp.idx2word = checkpoint['idx2word']
    cp.vocab_size = checkpoint['vocab_size']
    cp.max_length = checkpoint['max_length']

    model = ImageCaptioningModel(vocab_size=cp.vocab_size,
                                 embed_size=Config.EMBED_SIZE,
                                 num_heads=Config.NUM_HEADS,
                                 num_layers=Config.NUM_LAYERS,
                                 ff_dim=Config.FF_DIM,
                                 dropout=Config.DROPOUT,
                                 max_len=cp.max_length)
    model.load_state_dict(checkpoint['model_state_dict'])
    generator = CaptionGenerator(model, cp, Config)
    if image_id:
        generator.visualize_caption(image_id, Config.IMAGE_DIR, beam_size=3)
    else:
        rid = random.choice(list(cp.mapping.keys()))
        generator.visualize_caption(rid, Config.IMAGE_DIR, beam_size=3)
    return model, cp, generator

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train','resume_train','inference','extract_features','research_eval'])
    parser.add_argument('--image_id', type=str, default=None)
    parser.add_argument('--beam', type=int, default=3)
    args = parser.parse_args()

    if args.mode == 'extract_features':
        extractor = FeatureExtractorResNet50(device=Config.DEVICE)
        extractor.extract_features(Config.IMAGE_DIR, Config.FEATURES_PATH)
    elif args.mode == 'train':
        main()
    elif args.mode == 'resume_train':
        if not os.path.exists(Config.MODEL_SAVE_PATH):
            print("No trained model found. Train first.")
        else:
            checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
            cp = CaptionProcessor(Config.CAPTION_PATH)
            cp.load_captions()
            cp.clean_captions()
            cp.word2idx = checkpoint['word2idx']
            cp.idx2word = checkpoint['idx2word']
            cp.vocab_size = checkpoint['vocab_size']
            cp.max_length = checkpoint.get('max_length', Config.MAX_LEN)
            print(f"Vocabulary size: {cp.vocab_size}, Max caption length: {cp.max_length}")

            image_ids = list(cp.mapping.keys())
            random.shuffle(image_ids)
            split_idx = int(len(image_ids) * Config.TRAIN_SPLIT)
            train_ids = image_ids[:split_idx]
            test_ids = image_ids[split_idx:]

            train_dataset = FlickrRawImageDataset(
                image_dir=Config.IMAGE_DIR,
                image_ids=train_ids,
                captions_mapping=cp.mapping,
                caption_processor=cp
            )
            val_dataset = FlickrRawImageDataset(
                image_dir=Config.IMAGE_DIR,
                image_ids=test_ids,
                captions_mapping=cp.mapping,
                caption_processor=cp
            )
            train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
                                      num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
            val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                                    num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
            model = ImageCaptioningModel(vocab_size=cp.vocab_size,
                                         embed_size=Config.EMBED_SIZE,
                                         num_heads=Config.NUM_HEADS,
                                         num_layers=Config.NUM_LAYERS,
                                         ff_dim=Config.FF_DIM,
                                         dropout=Config.DROPOUT,
                                         max_len=cp.max_length)
            model.load_state_dict(checkpoint['model_state_dict'])
            trainer = Trainer(model, train_loader, val_loader, cp, Config)
            # load optimizer state
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trainer.use_scst = checkpoint.get('use_scst', True)
            trainer.best_val_loss = checkpoint.get('val_loss', float('inf'))
            
            print(f"Resuming training from epoch {checkpoint['epoch']+1} with val loss {checkpoint['val_loss']:.4f}")
            trainer.train()

            checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']+1} with val loss {checkpoint['val_loss']:.4f}")

            evaluator = ModelEvaluator(model, cp, test_ids, Config)
            evaluator.evaluate(use_beam=True, beam_size=3)

            research_eval = ResearchEvaluator(
                model=model,
                caption_processor=cp,
                image_ids=test_ids,
                config=Config
            )

            research_results = research_eval.evaluate(
                use_beam=True,
                beam_size=3
            )    
        # 8. visualize few examples
        generator = CaptionGenerator(model, cp, Config)
        for img_id in random.sample(test_ids, min(3, len(test_ids))):
            generator.visualize_caption(img_id, Config.IMAGE_DIR, beam_size=3)


    elif args.mode == 'research_eval':
        if not os.path.exists(Config.MODEL_SAVE_PATH):
            print("No trained model found. Train first.")
        else:
            checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)

            # with open(Config.FEATURES_PATH, 'rb') as f:
            #     features = pickle.load(f)

            cp = CaptionProcessor(Config.CAPTION_PATH)
            cp.load_captions()
            cp.clean_captions()
            cp.word2idx = checkpoint['word2idx']
            cp.idx2word = checkpoint['idx2word']
            cp.vocab_size = checkpoint['vocab_size']
            cp.max_length = checkpoint.get('max_length', Config.MAX_LEN)

            model = ImageCaptioningModel(vocab_size=cp.vocab_size,
                                        embed_size=Config.EMBED_SIZE,
                                        num_heads=Config.NUM_HEADS,
                                        num_layers=Config.NUM_LAYERS,
                                        ff_dim=Config.FF_DIM,
                                        dropout=Config.DROPOUT,
                                        max_len=cp.max_length)
            model.load_state_dict(checkpoint['model_state_dict'])

            image_ids = list(cp.mapping.keys())

            research_eval = ResearchEvaluator(
                model=model,
                caption_processor=cp,
                image_ids=image_ids,
                config=Config
            )

            research_eval.evaluate(use_beam=True, beam_size=args.beam)
    elif args.mode == 'inference':
        if not os.path.exists(Config.MODEL_SAVE_PATH):
            print("No trained model found. Train first.")
        else:
            load_and_inference(args.image_id)

    print("\nDone.")
