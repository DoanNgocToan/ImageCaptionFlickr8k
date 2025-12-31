"""
Image Captioning with Attention (timestep attention) using PyTorch
Dataset: Flickr30k
Changes vs original:
- ResNet50 feature extractor -> feature maps (1, 7, 7, 2048) saved as numpy
- Decoder uses attention at every timestep (LSTMCell-based)
- Dataset returns full input_seq and target_seq (caption-level)
- Beam Search implemented for inference
Author: Refactor for user request
"""

import os
import re
import pickle
import random
import time
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.models as models

from nltk.translate.bleu_score import corpus_bleu

# ============================================================================
# CONFIG
# ============================================================================
class Config:
    DATASET_PATH = '.\\flickr30k_images'  # adjust to your dataset path
    CAPTION_PATH = os.path.join(DATASET_PATH, 'results.csv')
    IMAGE_DIR = os.path.join(DATASET_PATH, 'flickr30k_images')
    FEATURES_PATH = os.path.join(DATASET_PATH, 'features_resnet50_flickr30k.pkl')
    MODEL_SAVE_PATH = os.path.join(DATASET_PATH, 'best_model_resnet50_attention_h512_30k.pth')
    TENSORBOARD_LOG_ROOT = os.path.join(DATASET_PATH, 'runs')

    # model hyperparams
    EMBED_SIZE = 512
    HIDDEN_SIZE = 512
    ATTENTION_DIM = 512

    EMBED_DROPOUT = 0.4
    LSTM_DROPOUT = 0.3
    DECODER_DROPOUT = 0.5

    # training
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 3e-4
    TRAIN_SPLIT = 0.90

    WEIGHT_DECAY = 1e-5
    LABEL_SMOOTHING = 0.1

    LR_SCHEDULER_FACTOR = 0.7
    LR_SCHEDULER_PATIENCE = 1
    EARLY_STOPPING_PATIENCE = 5

    GRAD_CLIP = 5.0

    NUM_WORKERS = 0
    PIN_MEMORY = False

    MAX_CAPTION_LENGTH = 35 
    MIN_WORD_FREQ = 3

    # features
    FEATURE_SHAPE = (49, 2048)   # 7*7, 2048
    IMAGE_SIZE = (224, 224)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def print_config(cls):
        print("="*70)
        print("OPTIMIZED CONFIGURATION FOR FLICKR8K")
        print("="*70)
        print(f"Device: {cls.DEVICE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Learning Rate: {cls.LEARNING_RATE:.0e}")
        print(f"Weight Decay: {cls.WEIGHT_DECAY:.0e}")
        print(f"Label Smoothing: {cls.LABEL_SMOOTHING}")
        print("-"*70)
        print(f"Embed Size: {cls.EMBED_SIZE}")
        print(f"Hidden Size: {cls.HIDDEN_SIZE} (Reduced from 512)")
        print(f"Attention Dim: {cls.ATTENTION_DIM}")
        print("-"*70)
        print(f"Embed Dropout: {cls.EMBED_DROPOUT}")
        print(f"LSTM Dropout: {cls.LSTM_DROPOUT}")
        print(f"Decoder Dropout: {cls.DECODER_DROPOUT}")
        print("-"*70)
        print(f"Feature shape: {cls.FEATURE_SHAPE}")
        print(f"Gradient Clip: {cls.GRAD_CLIP}")
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
        for p in model.parameters():
            p.requires_grad = False
        return model.to(self.device)

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
    URL_PATTERN = re.compile(r"(http|https|www\.)\S+")
    NON_ALNUM = re.compile(r"[^a-z0-9\s'’-]")
    CONTRACTIONS = {
        "n't": " not",
        "'re": " are",
        "'s": " is",
        "'d": " would",
        "'ll": " will",
        "'ve": " have",
        "'m": " am"
    }
    def __init__(self, caption_path):
        self.caption_path = caption_path
        self.mapping = {}
        self.word2idx = {}
        self.idx2word = {}
        self.max_length = 0
        self.vocab_size = 0

    def load_captions(self):
        print("\nLoading captions...")
        df = pd.read_csv(self.caption_path, sep="|", engine="python",skiprows=1, names=["image_name", "comment_number", "comment"])
        df.columns = [c.strip() for c in df.columns]

        for _, row in df.iterrows():
            image_name = row["image_name"].strip()
            caption = str(row["comment"]).strip()
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
                cap = self.URL_PATTERN.sub("", cap)
                cap = cap.replace("’", "'")
                for k, v in self.CONTRACTIONS.items():
                    cap = cap.replace(k, v)
                cap = self.NON_ALNUM.sub(" ", cap)
                cap = re.sub(r"\s+", " ", cap).strip()
                words = [w for w in cap.split() if len(w) > 1][:Config.MAX_CAPTION_LENGTH - 2]
                caps[i] = 'startseq ' + ' '.join(words) + ' endseq'

    def build_vocabulary(self, min_freq=Config.MIN_WORD_FREQ):
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
class FlickrDataset(Dataset):
    def __init__(self, image_ids, captions_mapping, features, caption_processor):
        self.image_ids = image_ids
        self.captions_mapping = captions_mapping
        self.features = features
        self.cp = caption_processor
        self.samples = []
        for img_id in image_ids:
            if img_id not in features:
                continue
            for caption in captions_mapping[img_id]:
                encoded = caption_processor.encode_caption(caption)
                # input = startseq ... token_{T-1}; target = token_1 ... endseq
                if len(encoded) < 2:
                    continue
                inp = encoded[:-1]
                tgt = encoded[1:]
                self.samples.append((img_id, inp, tgt))
        # compute max length from processor
        self.max_len = caption_processor.max_length - 1  # because input length is caption-1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, inp, tgt = self.samples[idx]
        # image features: (1,7,7,2048) -> reshape to (49,2048)
        feat = self.features[img_id].reshape(-1, Config.FEATURE_SHAPE[1])  # (49,2048)
        # pad input and target to self.max_len
        inp_arr = np.zeros(self.max_len, dtype=np.int64)
        tgt_arr = np.zeros(self.max_len, dtype=np.int64)
        inp_arr[:len(inp)] = inp
        tgt_arr[:len(tgt)] = tgt
        length = min(len(inp), self.max_len)
        return (
            torch.FloatTensor(feat),                 # (49,2048)
            torch.LongTensor(inp_arr),              # (max_len,)
            torch.LongTensor(tgt_arr),              # (max_len,)
            length                                 # effective length of input (before padding)
        )

# ============================================================================
# ATTENTION (Bahdanau) - per timestep
# ============================================================================
class Attention(nn.Module):
    def __init__(self, feature_dim=2048, hidden_dim=Config.HIDDEN_SIZE, attention_dim=Config.ATTENTION_DIM):
        super(Attention, self).__init__()
        self.W_feat = nn.Linear(feature_dim, attention_dim)      # projects image features
        self.W_hidden = nn.Linear(hidden_dim, attention_dim)     # projects hidden state
        self.V = nn.Linear(attention_dim, 1)                     # gives attention score

    def forward(self, features, hidden):
        """
        features: (batch, num_regions, feature_dim)  e.g. (B,49,2048)
        hidden: (batch, hidden_dim)
        returns: context (batch, feature_dim), alpha (batch, num_regions)
        """
        # projects
        feat_proj = self.W_feat(features)          # (B,49,attn)
        hid_proj = self.W_hidden(hidden).unsqueeze(1)  # (B,1,attn)
        e = torch.tanh(feat_proj + hid_proj)       # (B,49,attn)
        e = self.V(e).squeeze(2)                   # (B,49)
        alpha = torch.softmax(e, dim=1)            # (B,49)
        context = torch.bmm(alpha.unsqueeze(1), features).squeeze(1)  # (B, feature_dim)
        return context, alpha

# ============================================================================
# MODEL: Encoder already precomputed; Decoder with LSTMCell + attention per timestep
# ============================================================================
class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_size=Config.EMBED_SIZE, hidden_size=Config.HIDDEN_SIZE,
                 attention_dim=Config.ATTENTION_DIM, feature_dim=2048, embed_dropout=Config.EMBED_DROPOUT,
                 lstm_dropout=Config.LSTM_DROPOUT, decoder_dropout=Config.DECODER_DROPOUT):
        super(ImageCaptioningModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.feature_dim = feature_dim

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.embed_dropout = nn.Dropout(embed_dropout)

        # Attention
        self.attention = Attention(feature_dim, hidden_size, attention_dim)

        # LSTMCell: input will be [embed_t, context]
        self.lstm_cell = nn.LSTMCell(embed_size + feature_dim, hidden_size)
        self.lstm_dropout = nn.Dropout(lstm_dropout)

        # initialize h,c from mean image feature
        self.init_h = nn.Linear(feature_dim, hidden_size)
        self.init_c = nn.Linear(feature_dim, hidden_size)

        # decoder to vocab (use hidden + context)
        self.fc1 = nn.Linear(hidden_size + feature_dim, hidden_size)
        self.relu = nn.ReLU()
        self.decoder_dropout = nn.Dropout(decoder_dropout)
        self.fc2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, image_features, input_seqs, lengths):
        """
        image_features: (batch, 49, feature_dim)
        input_seqs: (batch, max_len)
        lengths: list/ tensor of actual lengths (number of tokens in input)
        returns: outputs: (batch, max_len, vocab_size)
        """
        batch_size = image_features.size(0)
        max_len = input_seqs.size(1)

        embeddings = self.embedding(input_seqs)  # (B, max_len, embed_size)
        embeddings = self.embed_dropout(embeddings)

        # init hidden/cell
        mean_feats = image_features.mean(dim=1)                # (B, feature_dim)
        h = torch.tanh(self.init_h(mean_feats))                # (B, hidden)
        c = torch.tanh(self.init_c(mean_feats))                # (B, hidden)

        outputs = torch.zeros(batch_size, max_len, self.vocab_size, device=image_features.device)

        # iterate timesteps
        for t in range(max_len):
            # compute attention using previous hidden state h
            context, alpha = self.attention(image_features, h)   # (B, feature_dim), (B,49)
            # input to LSTMCell: concat(embed_t, context)
            emb_t = embeddings[:, t, :]                         # (B, embed_size)
            lstm_input = torch.cat([emb_t, context], dim=1)     # (B, embed+feature)
            h, c = self.lstm_cell(lstm_input, (h, c))           # (B, hidden)

            h_drop = self.lstm_dropout(h)
            # compute output
            concat_h = torch.cat([h, context], dim=1)          # (B, hidden+feature)
            out = self.fc1(concat_h)
            out = self.relu(out)
            out = self.decoder_dropout(out)
            logits = self.fc2(out)                             # (B, vocab)
            outputs[:, t, :] = logits

        return outputs  # (B, max_len, vocab)

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

        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=config.LABEL_SMOOTHING)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=config.LR_SCHEDULER_FACTOR,
            patience=config.LR_SCHEDULER_PATIENCE, verbose=True
        )

        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.start_epoch = 0
        log_dir = os.path.join(
            config.TENSORBOARD_LOG_ROOT,
            time.strftime("%b%d_%H-%M-%S") + f'_h{config.HIDDEN_SIZE}_d{int(config.DECODER_DROPOUT*10)}'
        )
        # Sử dụng log_dir mới này cho SummaryWriter
        self.writer = SummaryWriter(log_dir)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.EPOCHS} [Train]")
        for batch_idx, (img_feats, input_seqs, target_seqs, lengths) in enumerate(pbar):
            img_feats = img_feats.to(self.config.DEVICE)             # (B,49,2048)
            input_seqs = input_seqs.to(self.config.DEVICE)           # (B,max_len)
            target_seqs = target_seqs.to(self.config.DEVICE)         # (B,max_len)
            lengths = lengths.to(self.config.DEVICE)

            outputs = self.model(img_feats, input_seqs, lengths)     # (B,max_len,vocab)
            # compute loss: flatten
            B, T, V = outputs.size()
            outputs_flat = outputs.view(B * T, V)
            targets_flat = target_seqs.view(B * T)

            loss = self.criterion(outputs_flat, targets_flat)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/Loss_batch', loss.item(), global_step)
        avg_epoch_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar('Train/Loss_epoch', avg_epoch_loss, epoch)
        return avg_epoch_loss

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.config.EPOCHS} [Val]")
            for img_feats, input_seqs, target_seqs, lengths in pbar:
                img_feats = img_feats.to(self.config.DEVICE)
                input_seqs = input_seqs.to(self.config.DEVICE)
                target_seqs = target_seqs.to(self.config.DEVICE)
                lengths = lengths.to(self.config.DEVICE)

                outputs = self.model(img_feats, input_seqs, lengths)
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
            'scheduler_state_dict': self.scheduler.state_dict(), 
            'best_val_loss': self.best_val_loss,
            'epochs_no_improve': self.epochs_no_improve,
            'val_loss': val_loss,
            'vocab_size': self.cp.vocab_size,
            'max_length': self.cp.max_length,
            'word2idx': self.cp.word2idx,
            'idx2word': self.cp.idx2word,
            'config': {
                'embed_size': self.config.EMBED_SIZE,
                'hidden_size': self.config.HIDDEN_SIZE,
                'attention_dim': self.config.ATTENTION_DIM,
                'embed_dropout': self.config.EMBED_DROPOUT,
                'lstm_dropout': self.config.LSTM_DROPOUT,
                'decoder_dropout': self.config.DECODER_DROPOUT
            }
        }
        torch.save(checkpoint, self.config.MODEL_SAVE_PATH)
        print(f"Model saved to {self.config.MODEL_SAVE_PATH}")

    def load_checkpoint(self, checkpoint_path):
        """
         Load checkpoint for resuming training
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.config.DEVICE)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded model weights from epoch {checkpoint['epoch'] + 1}")
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✓ Loaded optimizer state")
        
        # Load scheduler state (if exists)
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"✓ Loaded scheduler state")
        
        # Load training state
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf')))
        self.epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        
        print(f"✓ Resuming from epoch {self.start_epoch}")
        print(f"✓ Best val loss so far: {self.best_val_loss:.4f}")
        print(f"✓ Epochs without improvement: {self.epochs_no_improve}")
        
        return checkpoint

    def train(self, resume=False):
        if resume:
            print("\n" + "="*70)
            print("RESUMING TRAINING")
            print("="*70)
        else:
            print("\n" + "="*70)
            print("STARTING TRAINING")
            print("="*70)
        for epoch in range(self.config.EPOCHS):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)

            print(f"\nEpoch {epoch+1}/{self.config.EPOCHS}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            self.scheduler.step(val_loss)
        
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self.save_checkpoint(epoch, val_loss)
                print(f"✓ New best model! Val Loss: {val_loss:.4f}")
            else:
                self.epochs_no_improve += 1
                print(f"No improvement for {self.epochs_no_improve} epoch(s)")
            self.writer.add_scalar('Train/LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
        
            if self.epochs_no_improve >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping after {epoch+1} epochs")
                break

        self.writer.close()
        print("\nTRAINING COMPLETED\n")

# ============================================================================
# INFERENCE: greedy and beam search
# ============================================================================
class CaptionGenerator:
    def __init__(self, model, caption_processor, features, config=Config):
        self.model = model.to(config.DEVICE)
        self.model.eval()
        self.cp = caption_processor
        self.features = features
        self.config = config

        # shortcuts
        self.start_idx = self.cp.word2idx.get('startseq', 1)
        self.end_idx = self.cp.word2idx.get('endseq', None)
        if self.end_idx is None:
            # try fallback
            self.end_idx = self.cp.word2idx.get('end', 0)

    def _init_hidden(self, mean_feats):
        # mean_feats: (1, feature_dim)
        h = torch.tanh(self.model.init_h(mean_feats))
        c = torch.tanh(self.model.init_c(mean_feats))
        return h, c

    def generate_caption_greedy(self, image_id, max_length=None):
        if max_length is None:
            max_length = self.cp.max_length - 1
        feat = self.features[image_id]  # (1,7,7,2048)
        feat_t = torch.FloatTensor(feat).to(self.config.DEVICE)
        img_feats = feat_t.reshape(1, -1, Config.FEATURE_SHAPE[1])  # (1,49,2048)

        mean_feats = img_feats.mean(dim=1)  # (1,2048)
        h, c = self._init_hidden(mean_feats)

        generated = [self.start_idx]
        for t in range(max_length):
            # attention using previous hidden
            context, alpha = self.model.attention(img_feats, h)
            # prepare embedding for last predicted token
            last_token = torch.LongTensor([generated[-1]]).to(self.config.DEVICE)
            emb = self.model.embedding(last_token).squeeze(0)  # (embed,)
            inp = torch.cat([emb, context.squeeze(0)], dim=0).unsqueeze(0)  # (1, embed+feat)
            h, c = self.model.lstm_cell(inp, (h, c))
            concat_h = torch.cat([h, context], dim=1)
            out = self.model.fc1(concat_h)
            out = self.model.relu(out)
            out = self.model.decoder_dropout(out)
            logits = self.model.fc2(out)  # (1, vocab)
            probs = torch.softmax(logits, dim=1)
            next_idx = probs.argmax(dim=1).item()
            if next_idx == self.end_idx or next_idx == 0:
                break
            generated.append(next_idx)
        return self.cp.decode_caption(generated)

    def generate_caption_beam(self, image_id, beam_size=3, max_length=None):
        """
        Beam search implementation.
        Each beam: (sequence_of_indices, cumulative_logprob, hidden, cell)
        """
        if max_length is None:
            max_length = self.cp.max_length - 1

        feat = self.features[image_id]
        feat_t = torch.FloatTensor(feat).to(self.config.DEVICE)
        img_feats = feat_t.reshape(1, -1, Config.FEATURE_SHAPE[1])  # (1,49,2048)
        mean_feats = img_feats.mean(dim=1)
        h0, c0 = self._init_hidden(mean_feats)

        # initial beam
        beams = [([self.start_idx], 0.0, h0, c0)]
        completed = []

        for _ in range(max_length):
            new_beams = []
            for seq, score, h, c in beams:
                last = seq[-1]
                if last == self.end_idx:
                    # already finished
                    completed.append((seq, score))
                    continue
                # compute attention and next logits
                context, alpha = self.model.attention(img_feats, h)  # (1,feat), (1,49)
                last_token = torch.LongTensor([last]).to(self.config.DEVICE)
                emb = self.model.embedding(last_token).squeeze(0)
                inp = torch.cat([emb, context.squeeze(0)], dim=0).unsqueeze(0)
                h_new, c_new = self.model.lstm_cell(inp, (h, c))
                concat_h = torch.cat([h_new, context], dim=1)
                out = self.model.fc1(concat_h)
                out = self.model.relu(out)
                out = self.model.decoder_dropout(out)
                logits = self.model.fc2(out)  # (1,vocab)
                log_probs = torch.log_softmax(logits, dim=1).squeeze(0)  # (vocab,)

                topk_logprobs, topk_idx = torch.topk(log_probs, beam_size)
                for k in range(len(topk_idx)):
                    idx_k = int(topk_idx[k].item())
                    lp = float(topk_logprobs[k].item())
                    new_seq = seq + [idx_k]
                    new_score = score + lp
                    new_beams.append((new_seq, new_score, h_new, c_new))
            # select top beam_size beams
            new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            beams = new_beams

            # if we have enough completed sequences, can break (optional)
            if len(completed) >= beam_size:
                break

        # add remaining beams to completed
        for seq, score, _, _ in beams:
            completed.append((seq, score))
        # choose best completed by score
        completed = sorted(completed, key=lambda x: x[1], reverse=True)
        best_seq = completed[0][0]
        return self.cp.decode_caption(best_seq)

    def visualize_caption(self, image_id, image_dir, beam_size=3):
        # show image + print actual & predicted
        img_files = [f for f in os.listdir(image_dir) if f.startswith(image_id)]
        if not img_files:
            print(f"Image {image_id} not found in {image_dir}")
            return
        img_path = os.path.join(image_dir, img_files[0])
        image = Image.open(img_path)
        actual = self.cp.mapping.get(image_id, [])
        pred_greedy = self.generate_caption_greedy(image_id)
        pred_beam = self.generate_caption_beam(image_id, beam_size=beam_size)
        print("\n" + "="*60)
        print(f"Image ID: {image_id}")
        print("ACTUAL CAPTIONS:")
        for i, c in enumerate(actual, 1):
            print(f"{i}. {c}")
        print("\nPREDICTED (greedy):")
        print(pred_greedy)
        print("\nPREDICTED (beam size={}):".format(beam_size))
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
# EVALUATOR
# ============================================================================
class ModelEvaluator:
    def __init__(self, model, caption_processor, features, image_ids, config=Config):
        self.generator = CaptionGenerator(model, caption_processor, features, config)
        self.cp = caption_processor
        self.image_ids = image_ids

    def evaluate(self, use_beam=False, beam_size=3):
        actual = []
        predicted = []
        for image_id in tqdm(self.image_ids, desc="Generating captions"):
            if image_id not in self.cp.mapping:
                continue
            refs = [cap.split() for cap in self.cp.mapping[image_id]]
            if use_beam:
                pred = self.generator.generate_caption_beam(image_id, beam_size).split()
            else:
                pred = self.generator.generate_caption_greedy(image_id).split()
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

from collections import Counter
import math

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
    def __init__(self, model, caption_processor, features, image_ids, config=Config):
        self.generator = CaptionGenerator(model, caption_processor, features, config)
        self.cp = caption_processor
        self.image_ids = image_ids
        self.linear_bleu = LinearBPBLEU(max_n=4)

    def evaluate(self, use_beam=False, beam_size=3):
        actual = []
        predicted = []
        meteor_scores = []
        pred_lens = []
        ref_lens = []

        for image_id in tqdm(self.image_ids, desc="Research Evaluation"):
            if image_id not in self.cp.mapping:
                continue

            refs = [cap.split() for cap in self.cp.mapping[image_id]]

            if use_beam:
                pred_tokens = self.generator.generate_caption_beam(
                    image_id, beam_size
                ).split()
            else:
                pred_tokens = self.generator.generate_caption_greedy(image_id).split()

            actual.append(refs)
            predicted.append(pred_tokens)

            # METEOR (sentence-level)
            meteor_scores.append(
                meteor_score(refs, pred_tokens)
            )

            pred_lens.append(len(pred_tokens))
            ref_lens.append(min(len(r) for r in refs))

        # BLEU nltk
        bleu1_std = corpus_bleu(
            actual, predicted,
            weights=(1, 0, 0, 0)
        )
        bleu2_std = corpus_bleu(
            actual, predicted,
            weights=(0.5, 0.5, 0, 0)
        )
        bleu3_std = corpus_bleu(
            actual, predicted,
            weights=(0.33, 0.33, 0.33, 0)
        )
        bleu4_std = corpus_bleu(
            actual, predicted,
            weights=(0.25, 0.25, 0.25, 0.25)
        )

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
    if not os.path.exists(Config.FEATURES_PATH):
        extractor = FeatureExtractorResNet50(device=Config.DEVICE)
        features = extractor.extract_features(Config.IMAGE_DIR, Config.FEATURES_PATH)
    else:
        with open(Config.FEATURES_PATH, 'rb') as f:
            features = pickle.load(f)
        print(f"Loaded features for {len(features)} images")

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
    train_dataset = FlickrDataset(train_ids, cp.mapping, features, cp)
    val_dataset = FlickrDataset(test_ids, cp.mapping, features, cp)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
                              num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                            num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 5. model
    model = ImageCaptioningModel(vocab_size=cp.vocab_size,
                                 embed_size=Config.EMBED_SIZE,
                                 hidden_size=Config.HIDDEN_SIZE,
                                 attention_dim=Config.ATTENTION_DIM,
                                 feature_dim=Config.FEATURE_SHAPE[1],
                                 embed_dropout=Config.EMBED_DROPOUT,
                                 lstm_dropout=Config.LSTM_DROPOUT, 
                                 decoder_dropout=Config.DECODER_DROPOUT)
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

    evaluator = ModelEvaluator(model, cp, features, test_ids, Config)
    evaluator.evaluate(use_beam=True, beam_size=3)

    research_eval = ResearchEvaluator(
        model=model,
        caption_processor=cp,
        features=features,
        image_ids=test_ids,
        config=Config
    )

    research_results = research_eval.evaluate(
        use_beam=True,
        beam_size=3
    )
    print(f"Research evaluation results: {research_results}")
    # 8. visualize few examples
    generator = CaptionGenerator(model, cp, features, Config)
    for img_id in random.sample(test_ids, min(3, len(test_ids))):
        generator.visualize_caption(img_id, Config.IMAGE_DIR, beam_size=3)

def resume_training():
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print("No checkpoint found. Please train from scratch first.")
        return
    
    print("="*70)
    print("LOADING CHECKPOINT FOR RESUME TRAINING")
    print("="*70)
    
    # 1. Load checkpoint to get vocab info
    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    print(f"Found checkpoint from epoch {checkpoint['epoch'] + 1}")
    print(f"Previous best val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    # 2. Load features
    with open(Config.FEATURES_PATH, 'rb') as f:
        features = pickle.load(f)
    print(f"Loaded features for {len(features)} images")
    
    # 3. Load captions with SAME vocab from checkpoint
    cp = CaptionProcessor(Config.CAPTION_PATH)
    cp.load_captions()
    cp.clean_captions()
    cp.word2idx = checkpoint['word2idx']
    cp.idx2word = checkpoint['idx2word']
    cp.vocab_size = checkpoint['vocab_size']
    cp.max_length = checkpoint['max_length']
    print(f"Loaded vocabulary: {cp.vocab_size} words, max length: {cp.max_length}")
    
    image_ids = list(cp.mapping.keys())
    random.seed(42)  # Fixed seed!
    random.shuffle(image_ids)
    random.seed()  # Reset to random seed after split
    
    split_idx = int(len(image_ids) * Config.TRAIN_SPLIT)
    train_ids = image_ids[:split_idx]
    test_ids = image_ids[split_idx:]
    print(f"Train images: {len(train_ids)}, Test images: {len(test_ids)}")
    
    # 5. Create datasets & loaders
    train_dataset = FlickrDataset(train_ids, cp.mapping, features, cp)
    val_dataset = FlickrDataset(test_ids, cp.mapping, features, cp)
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    model = ImageCaptioningModel(
        vocab_size=cp.vocab_size,
        embed_size=Config.EMBED_SIZE,
        hidden_size=Config.HIDDEN_SIZE,
        attention_dim=Config.ATTENTION_DIM,
        feature_dim=Config.FEATURE_SHAPE[1],
        embed_dropout=Config.EMBED_DROPOUT,
        lstm_dropout=Config.LSTM_DROPOUT,
        decoder_dropout=Config.DECODER_DROPOUT
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")
    
    # 7. Create trainer and LOAD checkpoint
    trainer = Trainer(model, train_loader, val_loader, cp, Config)
    trainer.load_checkpoint(Config.MODEL_SAVE_PATH)
    
    # 8. Resume training from checkpoint epoch
    trainer.train(resume=True)
    
    # 9. Evaluation
    print("\n" + "="*70)
    print("EVALUATING BEST MODEL")
    print("="*70)
    
    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1} with val loss {checkpoint['val_loss']:.4f}")
    
    evaluator = ModelEvaluator(model, cp, features, test_ids, Config)
    evaluator.evaluate(use_beam=True, beam_size=3)
    
    research_eval = ResearchEvaluator(model, cp, features, test_ids, Config)
    research_results = research_eval.evaluate(use_beam=True, beam_size=3)
    print(f"Research evaluation results: {research_results}")
    
    # Visualize examples
    generator = CaptionGenerator(model, cp, features, Config)
    for img_id in random.sample(test_ids, min(3, len(test_ids))):
        generator.visualize_caption(img_id, Config.IMAGE_DIR, beam_size=3)

# convenience loader & inference
def load_and_inference(image_id=None):
    print("Loading model & features...")
    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    with open(Config.FEATURES_PATH, 'rb') as f:
        features = pickle.load(f)
    cp = CaptionProcessor(Config.CAPTION_PATH)
    cp.load_captions()
    cp.clean_captions()
    cp.word2idx = checkpoint['word2idx']
    cp.idx2word = checkpoint['idx2word']
    cp.vocab_size = checkpoint['vocab_size']
    cp.max_length = checkpoint['max_length']

    model = ImageCaptioningModel(vocab_size=cp.vocab_size,
                                 embed_size=Config.EMBED_SIZE,
                                 hidden_size=Config.HIDDEN_SIZE,
                                 attention_dim=Config.ATTENTION_DIM,
                                 feature_dim=Config.FEATURE_SHAPE[1],
                                 embed_dropout=Config.EMBED_DROPOUT,
                                 lstm_dropout=Config.LSTM_DROPOUT, 
                                 decoder_dropout=Config.DECODER_DROPOUT)
    model.load_state_dict(checkpoint['model_state_dict'])
    generator = CaptionGenerator(model, cp, features, Config)
    if image_id:
        generator.visualize_caption(image_id, Config.IMAGE_DIR, beam_size=3)
    else:
        rid = random.choice(list(cp.mapping.keys()))
        generator.visualize_caption(rid, Config.IMAGE_DIR, beam_size=3)
    return model, cp, features, generator

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
        resume_training()

    elif args.mode == 'research_eval':
        if not os.path.exists(Config.MODEL_SAVE_PATH):
            print("No trained model found. Train first.")
        else:
            checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)

            with open(Config.FEATURES_PATH, 'rb') as f:
                features = pickle.load(f)

            cp = CaptionProcessor(Config.CAPTION_PATH)
            cp.load_captions()
            cp.clean_captions()
            cp.word2idx = checkpoint['word2idx']
            cp.idx2word = checkpoint['idx2word']
            cp.vocab_size = checkpoint['vocab_size']
            cp.max_length = checkpoint['max_length']

            model = ImageCaptioningModel(
                vocab_size=cp.vocab_size,
                embed_size=Config.EMBED_SIZE,
                hidden_size=Config.HIDDEN_SIZE,
                attention_dim=Config.ATTENTION_DIM,
                feature_dim=Config.FEATURE_SHAPE[1],
                embed_dropout=Config.EMBED_DROPOUT,
                lstm_dropout=Config.LSTM_DROPOUT, 
                decoder_dropout=Config.DECODER_DROPOUT
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            

            image_ids = list(cp.mapping.keys())
            print(f"Evaluating on {len(image_ids)} images...")
            research_eval = ResearchEvaluator(
                model=model,
                caption_processor=cp,
                features=features,
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
