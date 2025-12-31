"""
Unified Image Captioning with EfficientNet (B0-B4) + LSTM + Attention
Dataset: Flickr8k
Architecture: Configurable EfficientNet variants
Author: Unified config-based approach
"""

import os
import re
import pickle
import random
import time
from collections import defaultdict, Counter
from dataclasses import dataclass

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b1, EfficientNet_B1_Weights,
    efficientnet_b2, EfficientNet_B2_Weights,
    efficientnet_b3, EfficientNet_B3_Weights,
    efficientnet_b4, EfficientNet_B4_Weights,
)

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score

# ============================================================================
# EFFICIENTNET SPECIFICATIONS
# ============================================================================
@dataclass
class EfficientNetSpec:
    """Specifications for each EfficientNet variant"""
    name: str
    input_size: int
    feature_height: int
    feature_width: int
    feature_dim: int
    model_fn: callable
    weights: object
    
    @property
    def num_regions(self):
        return self.feature_height * self.feature_width
    
    @property
    def feature_shape(self):
        return (self.num_regions, self.feature_dim)

# EfficientNet specifications mapping
EFFICIENTNET_SPECS = {
    'b0': EfficientNetSpec(
        name='b0',
        input_size=224,
        feature_height=7,
        feature_width=7,
        feature_dim=1280,
        model_fn=efficientnet_b0,
        weights=EfficientNet_B0_Weights.IMAGENET1K_V1
    ),
    'b1': EfficientNetSpec(
        name='b1',
        input_size=240,
        feature_height=8,
        feature_width=8,
        feature_dim=1280,
        model_fn=efficientnet_b1,
        weights=EfficientNet_B1_Weights.IMAGENET1K_V1
    ),
    'b2': EfficientNetSpec(
        name='b2',
        input_size=260,
        feature_height=9,
        feature_width=9,
        feature_dim=1408,
        model_fn=efficientnet_b2,
        weights=EfficientNet_B2_Weights.IMAGENET1K_V1
    ),
    'b3': EfficientNetSpec(
        name='b3',
        input_size=300,
        feature_height=10,
        feature_width=10,
        feature_dim=1536,
        model_fn=efficientnet_b3,
        weights=EfficientNet_B3_Weights.IMAGENET1K_V1
    ),
    'b4': EfficientNetSpec(
        name='b4',
        input_size=380,
        feature_height=12,
        feature_width=12,
        feature_dim=1792,
        model_fn=efficientnet_b4,
        weights=EfficientNet_B4_Weights.IMAGENET1K_V1
    ),
}

# ============================================================================
# CONFIG - Now accepts efficientnet_variant
# ============================================================================
class Config:
    # Dataset paths
    DATASET_PATH = '.\\content\\clean_data_flickr8k'
    BIG_DATASET_PATH = '.\\flickr30k_images'
    CAPTION_PATH = os.path.join(DATASET_PATH, 'captions.txt')
    IMAGE_DIR = os.path.join(DATASET_PATH, 'Images')
    
    # Model selection
    EFFICIENTNET_VARIANT = 'b2'  # Change this: 'b0', 'b1', 'b2', 'b3', 'b4'
    
    @classmethod
    def get_spec(cls):
        """Get EfficientNet specification"""
        return EFFICIENTNET_SPECS[cls.EFFICIENTNET_VARIANT]
    
    @classmethod
    def get_paths(cls):
        """Generate variant-specific paths"""
        variant = cls.EFFICIENTNET_VARIANT
        spec = cls.get_spec()
        
        paths = {
            'features': os.path.join(cls.DATASET_PATH, f'features_efficientnet_{variant}.pkl'),
            'model': os.path.join(cls.DATASET_PATH, f'best_model_efficientnet_{variant}_transformer.pth'),
            'tensorboard': os.path.join(cls.DATASET_PATH, 'runs', f'efficientnet_{variant}_transformer')
        }
        return paths
    
    # Model hyperparams - AUTOMATICALLY ADAPT to feature_dim
    EMBED_SIZE = 512
    HIDDEN_SIZE = 512
    ATTENTION_DIM = 512
    NUM_HEADS = 8
    NUM_LAYERS = 4
    FF_DIM = 2048
    DROPOUT = 0.5
    
    EMBED_DROPOUT = 0.4
    LSTM_DROPOUT = 0.3
    DECODER_DROPOUT = 0.5
    
    # Training
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    TRAIN_SPLIT = 0.80
    
    WEIGHT_DECAY = 1e-5
    LABEL_SMOOTHING = 0.1
    
    LR_SCHEDULER_FACTOR = 0.7
    LR_SCHEDULER_PATIENCE = 1
    EARLY_STOPPING_PATIENCE = 10
    
    GRAD_CLIP = 5.0
    
    SCHEDULED_SAMPLING = False
    SS_START_EPOCH = 50
    SS_MAX_RATIO = 0.5
    SS_STRATEGY = 'sigmoid'
    
    NUM_WORKERS = 0
    PIN_MEMORY = False
    
    MIN_WORD_FREQ = 1  # Rare word filtering
    MAX_LENGTH = 35  # Max caption length
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @classmethod
    def print_config(cls):
        spec = cls.get_spec()
        paths = cls.get_paths()
        
        print("="*70)
        print(f"EFFICIENTNET-{spec.name.upper()} + TRANSFORMER CONFIGURATION")
        print("="*70)
        print(f"Device: {cls.DEVICE}")
        print(f"CNN: EfficientNet-{spec.name.upper()}")
        print(f"Input Size: {spec.input_size}×{spec.input_size}")
        print(f"Feature Map: {spec.feature_height}×{spec.feature_width}×{spec.feature_dim}")
        print(f"Num Regions: {spec.num_regions}")
        print("-"*70)
        print(f"Decoder: Transformer")
        print(f"Embed Size: {cls.EMBED_SIZE}")
        print(f"Num Heads: {cls.NUM_HEADS}")
        print(f"Num Layers: {cls.NUM_LAYERS}")
        print(f"FF Dimension: {cls.FF_DIM}")
        print(f"Dropout: {cls.DROPOUT}")
        print("-"*70)
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Learning Rate: {cls.LEARNING_RATE:.0e}")
        print("-"*70)
        print(f"Features: {paths['features']}")
        print(f"Model: {paths['model']}")
        print("="*70)

# ============================================================================
# FEATURE EXTRACTOR - Unified for all EfficientNet variants
# ============================================================================
class FeatureExtractorEfficientNet:
    def __init__(self, variant='b0', device=Config.DEVICE):
        self.variant = variant
        self.spec = EFFICIENTNET_SPECS[variant]
        self.device = device
        self.model = self._build_model()
        self.transform = transforms.Compose([
            transforms.Resize((self.spec.input_size, self.spec.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def _build_model(self):
        """Build EfficientNet model dynamically"""
        print(f"Loading EfficientNet-{self.spec.name.upper()}...")
        model = self.spec.model_fn(weights=self.spec.weights)
        # Extract only features (before avgpool and classifier)
        model = model.features
        model.eval().to(self.device)
        for p in model.parameters():
            p.requires_grad = False
        return model
    
    def extract_features(self, image_dir, save_path):
        print(f"\n{'='*70}")
        print(f"EXTRACTING EFFICIENTNET-{self.spec.name.upper()} FEATURES")
        print(f"Input Size: {self.spec.input_size}×{self.spec.input_size}")
        print(f"Output Shape: {self.spec.feature_height}×{self.spec.feature_width}×{self.spec.feature_dim}")
        print(f"Num Regions: {self.spec.num_regions}")
        print(f"{'='*70}\n")
        
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
        
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"\n{'='*70}")
        print(f"Saved features to: {save_path}")
        print(f"Total images: {len(features)}")
        print(f"File size: {file_size_mb:.1f} MB")
        print(f"Per-image: {file_size_mb/len(features):.2f} MB")
        print(f"{'='*70}\n")
        return features

# ============================================================================
# CAPTION PROCESSOR (unchanged)
# ============================================================================
class CaptionProcessor:
    
    def __init__(self, caption_path, min_freq=Config.MIN_WORD_FREQ):
        self.caption_path = caption_path
        self.min_freq = min_freq
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
    
    def build_vocabulary(self):
        print(f"Building vocabulary (min_freq={self.min_freq})...")
        freq = defaultdict(int)
        for caps in self.mapping.values():
            for cap in caps:
                for w in cap.split():
                    freq[w] += 1
        
        # Filter by frequency
        words = [w for w, c in freq.items() if c >= self.min_freq]
        
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
        for w in sorted(words):
            if w in self.word2idx:
                continue
            self.word2idx[w] = idx
            idx += 1
        
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        self.max_length = max(len(c.split()) for caps in self.mapping.values() for c in caps)
        
        # Print statistics
        total_words = sum(freq.values())
        filtered_words = sum(c for w, c in freq.items() if c < self.min_freq)
        print(f"Vocab size: {self.vocab_size}")
        print(f"Max caption length: {self.max_length}")
        print(f"Rare words filtered: {len(freq) - len(words)} ({filtered_words/total_words*100:.1f}% of tokens)")
    
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
# DATASET
# ============================================================================
class FlickrDataset(Dataset):
    def __init__(self, image_ids, captions_mapping, features, caption_processor, feature_shape):
        self.image_ids = image_ids
        self.captions_mapping = captions_mapping
        self.features = features
        self.cp = caption_processor
        self.feature_shape = feature_shape
        self.samples = []
        
        for img_id in image_ids:
            if img_id not in features:
                continue
            for caption in captions_mapping[img_id]:
                encoded = caption_processor.encode_caption(caption)
                if len(encoded) < 2:
                    continue
                inp = encoded[:-1]
                tgt = encoded[1:]
                self.samples.append((img_id, inp, tgt))
        
        self.max_len = caption_processor.max_length - 1
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_id, inp, tgt = self.samples[idx]
        # Reshape to (num_regions, feature_dim)
        feat = self.features[img_id].reshape(self.feature_shape)
        
        inp_arr = np.zeros(self.max_len, dtype=np.int64)
        tgt_arr = np.zeros(self.max_len, dtype=np.int64)
        inp_arr[:len(inp)] = inp
        tgt_arr[:len(tgt)] = tgt
        length = min(len(inp), self.max_len)
        
        return (
            torch.FloatTensor(feat),
            torch.LongTensor(inp_arr),
            torch.LongTensor(tgt_arr),
            length
        )

# ============================================================================
# SCHEDULED SAMPLING
# ============================================================================
def get_sampling_probability(epoch, total_epochs, start_epoch, max_ratio, strategy='linear'):
    if epoch < start_epoch:
        return 0.0
    
    progress = (epoch - start_epoch) / max(1, total_epochs - start_epoch)
    progress = min(1.0, progress)
    
    if strategy == 'linear':
        ratio = max_ratio * progress
    elif strategy == 'exponential':
        ratio = max_ratio * (progress ** 2)
    elif strategy == 'sigmoid':
        import math
        x = 10 * (progress - 0.5)
        sigmoid = 1 / (1 + math.exp(-x))
        ratio = max_ratio * sigmoid
    else:
        ratio = 0.0
    
    return ratio
# ============================================================================
# POSITIONAL ENCODING - TỪ FILE RESNET
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
# TRANSFORMER DECODER - TỪ FILE RESNET
# ============================================================================  
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, ff_dim, dropout, max_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.pos = PositionalEncoding(embed_size, max_len)
        layer = nn.TransformerDecoderLayer(embed_size, num_heads, ff_dim, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, tgt, memory, tgt_mask, tgt_key_padding_mask):
        x = self.embed(tgt)
        x = self.pos(x)
        out = self.decoder(x, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return self.fc(out)


# ============================================================================
# MODEL - Automatically adapts to feature_dim
# ============================================================================
class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, feature_dim, embed_size=Config.EMBED_SIZE,
                 num_heads=Config.NUM_HEADS, num_layers=Config.NUM_LAYERS,
                 ff_dim=Config.FF_DIM, dropout=Config.DROPOUT, max_len=Config.MAX_LENGTH):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.feature_dim = feature_dim
        self.max_len = max_len

        # Feature projection (feature_dim -> embed_size)
        self.proj = nn.Linear(feature_dim, embed_size)

        # Transformer Decoder
        self.decoder = TransformerDecoder(
            vocab_size, embed_size, num_heads, num_layers, ff_dim, dropout, max_len
        )
    
    def generate_mask(self, sz, device):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(device)

    def forward(self, image_features, input_seqs):
        """
        image_features: (B, num_regions, feature_dim)
        input_seqs: (B, max_len)
        returns: (B, max_len, vocab_size)
        """
        device = image_features.device
        
        # Project image features to embed_size
        memory = self.proj(image_features)  # (B, num_regions, embed_size)

        # Prepare masks
        T = input_seqs.size(1)
        tgt_mask = self.generate_mask(T, device)
        tgt_pad_mask = (input_seqs == 0)

        # Decode
        outputs = self.decoder(input_seqs, memory, tgt_mask, tgt_pad_mask)
        
        return outputs

# Rest of the code (Trainer, CaptionGenerator, Evaluators, main) remains the same
# Just use Config.get_spec().feature_dim wherever needed

# ============================================================================
# TRAINER (with scheduled sampling)
# ============================================================================
class Trainer:
    def __init__(self, model, train_loader, val_loader, caption_processor, config=Config):
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cp = caption_processor
        self.config = config

        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=config.LABEL_SMOOTHING)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE, 
                                    weight_decay=config.WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=config.LR_SCHEDULER_FACTOR,
            patience=config.LR_SCHEDULER_PATIENCE, verbose=True
        )

        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        
        paths = config.get_paths()
        log_dir = os.path.join(paths['tensorboard'], time.strftime("%b%d_%H-%M-%S"))
        self.writer = SummaryWriter(log_dir)
        self.model_save_path = paths['model']

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.EPOCHS} [Train]")
        for batch_idx, (img_feats, input_seqs, target_seqs, length) in enumerate(pbar):
            img_feats = img_feats.to(self.config.DEVICE)
            input_seqs = input_seqs.to(self.config.DEVICE)
            target_seqs = target_seqs.to(self.config.DEVICE)

            outputs = self.model(img_feats, input_seqs)

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
            for img_feats, input_seqs, target_seqs, length in pbar:
                img_feats = img_feats.to(self.config.DEVICE)
                input_seqs = input_seqs.to(self.config.DEVICE)
                target_seqs = target_seqs.to(self.config.DEVICE)

                outputs = self.model(img_feats, input_seqs)
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
        spec = self.config.get_spec()
        checkpoint = {
            'epoch': epoch,
            'efficientnet_variant': self.config.EFFICIENTNET_VARIANT,
            'feature_dim': spec.feature_dim,
            'num_regions': spec.num_regions,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'val_loss': val_loss,
            'vocab_size': self.cp.vocab_size,
            'max_length': self.cp.max_length,
            'word2idx': self.cp.word2idx,
            'idx2word': self.cp.idx2word,
            'config': {
                'embed_size': self.config.EMBED_SIZE,
                'num_heads': self.config.NUM_HEADS,
                'num_layers': self.config.NUM_LAYERS,
                'ff_dim': self.config.FF_DIM,
                'dropout': self.config.DROPOUT,
                'min_word_freq': self.config.MIN_WORD_FREQ
            }
        }
        torch.save(checkpoint, self.model_save_path)
        print(f"Model saved to {self.model_save_path}")

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

            self.scheduler.step(val_loss)
        
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self.save_checkpoint(epoch, val_loss)
                print(f"New best model! Val Loss: {val_loss:.4f}")
            else:
                self.epochs_no_improve += 1
                print(f"No improvement for {self.epochs_no_improve} epoch(s)")
            
            self.writer.add_scalar('Train/LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
        
            if self.epochs_no_improve >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping after {epoch+1} epochs")
                break

        self.writer.close()
        print("\nTRAINING COMPLETED\n")

# CaptionGenerator and Evaluators remain similar - just use Config.get_spec()

# ============================================================================
# INFERENCE (unchanged)
# ============================================================================
class CaptionGenerator:
    def __init__(self, model, caption_processor, features, config=Config):
        self.model = model.to(config.DEVICE)
        self.model.eval()
        self.cp = caption_processor
        self.features = features
        self.config = config
        self.spec = config.get_spec()

        self.start_idx = self.cp.word2idx.get('startseq', 1)
        self.end_idx = self.cp.word2idx.get('endseq', None)
        if self.end_idx is None:
            self.end_idx = self.cp.word2idx.get('end', 0)
        self.max_gen_len = self.cp.max_length - 1

    def _prepare_features(self, image_id):
        feat = self.features[image_id]
        feat_t = torch.FloatTensor(feat).to(self.config.DEVICE)
        img_feats = feat_t.reshape(1, -1, self.spec.feature_dim)
        memory = self.model.proj(img_feats)
        return memory

    def generate_caption_greedy(self, image_id):
        memory = self._prepare_features(image_id)
        device = memory.device
        
        generated_seq = torch.LongTensor([[self.start_idx]]).to(device)

        for t in range(self.max_gen_len):
            T = generated_seq.size(1)
            tgt_mask = self.model.generate_mask(T, device)
            tgt_pad_mask = (generated_seq == 0)

            logits = self.model.decoder(generated_seq, memory, tgt_mask, tgt_pad_mask)
            
            last_logits = logits[:, -1, :]
            next_token = last_logits.argmax(dim=-1).item()

            if next_token == self.end_idx or next_token == 0:
                break
            
            next_token_tensor = torch.LongTensor([[next_token]]).to(device)
            generated_seq = torch.cat([generated_seq, next_token_tensor], dim=1)

        final_indices = generated_seq.squeeze(0).tolist()
        return self.cp.decode_caption(final_indices)

    def generate_caption_beam(self, image_id, beam_size=3):
        memory = self._prepare_features(image_id)
        device = memory.device
        
        beams = [([self.start_idx], 0.0)]
        completed = []

        for t in range(self.max_gen_len):
            new_beams = []
            
            current_sequences = [torch.LongTensor([seq]).to(device) for seq, _ in beams]
            
            if not current_sequences:
                break

            T_max = max(s.size(1) for s in current_sequences)
            batched_seq = torch.zeros(len(current_sequences), T_max, dtype=torch.long, device=device)
            for i, seq in enumerate(current_sequences):
                batched_seq[i, :seq.size(1)] = seq
            
            batched_memory = memory.expand(len(current_sequences), -1, -1)
            
            T_batch = batched_seq.size(1)
            tgt_mask = self.model.generate_mask(T_batch, device)
            tgt_pad_mask = (batched_seq == 0)

            logits = self.model.decoder(batched_seq, batched_memory, tgt_mask, tgt_pad_mask)
            log_probs = torch.log_softmax(logits, dim=-1)
            
            last_log_probs = log_probs[:, -1, :]
            
            for i, (seq, score) in enumerate(beams):
                if seq[-1] == self.end_idx:
                    completed.append((seq, score))
                    continue
                
                topk_logprobs, topk_idx = torch.topk(last_log_probs[i], beam_size)
                
                for k in range(beam_size):
                    idx_k = int(topk_idx[k].item())
                    lp = float(topk_logprobs[k].item())
                    new_seq = seq + [idx_k]
                    new_score = score + lp
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

    def visualize_caption(self, image_id, image_dir, beam_size=3):
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
# EVALUATORS (unchanged)
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

from collections import Counter
import math

class LinearBPBLEU:
    def __init__(self, max_n=4):
        self.max_n = max_n

    def _ngram_counts(self, tokens, n):
        return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])

    def _clipped_precision(self, references, hypotheses, n):
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

        smooth_eps = 1e-9
        log_p_sum = 0.0
        for w, p in zip(weights, p_ns):
            log_p_sum += w * math.log(max(p, smooth_eps))
        geo_mean = math.exp(log_p_sum)

        for refs, hyp in zip(list_of_references, hypotheses):
            total_pred_len += len(hyp)
            ref_lens = [len(r) for r in refs]
            total_ref_len += min(ref_lens, key=lambda rl: abs(rl - len(hyp)))

        bp = min(1.0, total_pred_len / max(total_ref_len, 1))

        return bp * geo_mean

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

            meteor_scores.append(meteor_score(refs, pred_tokens))
            pred_lens.append(len(pred_tokens))
            ref_lens.append(min(len(r) for r in refs))

        bleu1_std = corpus_bleu(actual, predicted, weights=(1, 0, 0, 0))
        bleu2_std = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
        bleu3_std = corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0))
        bleu4_std = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))

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
# VISUALIZATION
# ============================================================================
def visualize_sampling_schedules():
    import matplotlib.pyplot as plt
    
    epochs = list(range(30))
    strategies = ['linear', 'exponential', 'sigmoid']
    
    plt.figure(figsize=(12, 4))
    
    for i, strategy in enumerate(strategies, 1):
        probs = [
            get_sampling_probability(e, 30, 5, 0.5, strategy) 
            for e in epochs
        ]
        
        plt.subplot(1, 3, i)
        plt.plot(epochs, probs, linewidth=2)
        plt.axvline(x=Config.SS_START_EPOCH, color='r', linestyle='--', alpha=0.5, label='Start epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Sampling Probability')
        plt.title(f'{strategy.capitalize()} Schedule')
        plt.ylim([0, 0.6])
        plt.grid(alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('scheduled_sampling_strategies_efficientnet.png', dpi=150)
    print("Saved visualization to 'scheduled_sampling_strategies_efficientnet.png'")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================
def main():
    Config.print_config()
    spec = Config.get_spec()
    paths = Config.get_paths()
    
    # 1. Extract features
    if not os.path.exists(paths['features']):
        extractor = FeatureExtractorEfficientNet(variant=Config.EFFICIENTNET_VARIANT, device=Config.DEVICE)
        features = extractor.extract_features(Config.IMAGE_DIR, paths['features'])
    else:
        with open(paths['features'], 'rb') as f:
            features = pickle.load(f)
        file_size_mb = os.path.getsize(paths['features']) / (1024 * 1024)
        print(f"Loaded features for {len(features)} images ({file_size_mb:.1f} MB)")
    
    # 2. Process captions
    cp = CaptionProcessor(Config.CAPTION_PATH, min_freq=Config.MIN_WORD_FREQ)
    cp.load_captions()
    cp.clean_captions()
    cp.build_vocabulary()
    
    if 'startseq' not in cp.word2idx:
        cp.word2idx['startseq'] = len(cp.word2idx)
        cp.idx2word[cp.word2idx['startseq']] = 'startseq'
    if 'endseq' not in cp.word2idx:
        cp.word2idx['endseq'] = len(cp.word2idx)
        cp.idx2word[cp.word2idx['endseq']] = 'endseq'
    cp.vocab_size = len(cp.word2idx)
    
    # 3. Split dataset
    image_ids = list(cp.mapping.keys())
    random.shuffle(image_ids)
    split_idx = int(len(image_ids) * Config.TRAIN_SPLIT)
    train_ids = image_ids[:split_idx]
    test_ids = image_ids[split_idx:]
    print(f"Train images: {len(train_ids)}, Test images: {len(test_ids)}")
    
    # 4. Create datasets
    train_dataset = FlickrDataset(train_ids, cp.mapping, features, cp, spec.feature_shape)
    val_dataset = FlickrDataset(test_ids, cp.mapping, features, cp, spec.feature_shape)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
                              num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                            num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # 5. Initialize model
    model = ImageCaptioningModel(
        vocab_size=cp.vocab_size,
        feature_dim=specs.feature_dim,
        embed_size=Config.EMBED_SIZE,
        num_heads=Config.NUM_HEADS,
        num_layers=Config.NUM_LAYERS,
        ff_dim=Config.FF_DIM,
        dropout=Config.DROPOUT,
        max_len=cp.max_length
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    # 6. Train
    trainer = Trainer(model, train_loader, val_loader, cp, Config)
    trainer.train()
    
    print("\nTraining completed!")
    print(f"Best model saved at: {paths['model']}")

    checkpoint = torch.load(paths['model'], map_location=Config.DEVICE)
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
    
    
    # visualize_sampling_schedules()
    
    # Example: Get sampling probability for epoch 10
    # prob = get_sampling_probability(
    #     epoch=10,
    #     total_epochs=30,
    #     start_epoch=5,
    #     max_ratio=0.5,
    #     strategy='linear'
    # )
    # print(f"\nEpoch 10, Linear strategy: {prob:.3f}")
    # print(f"  → {prob*100:.1f}% of tokens will use predictions")
    # print(f"  → {(1-prob)*100:.1f}% of tokens will use ground truth")

    # 8. visualize few examples
    # generator = CaptionGenerator(model, cp, features, Config)
    # for img_id in random.sample(test_ids, min(3, len(test_ids))):
    #     generator.visualize_caption(img_id, Config.IMAGE_DIR, beam_size=3)

def resume_training():
    paths = Config.get_paths()
    specs = Config.get_spec()
    if not os.path.exists(paths['model']):
        print("No checkpoint found. Please train from scratch first.")
        return
    
    print("="*70)
    print("LOADING CHECKPOINT FOR RESUME TRAINING")
    print("="*70)
    
    # 1. Load checkpoint to get vocab info
    checkpoint = torch.load(paths['model'], map_location=Config.DEVICE)
    print(f"Found checkpoint from epoch {checkpoint['epoch'] + 1}")
    print(f"Previous best val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    # 2. Load features
    with open(paths['features'], 'rb') as f:
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
        feature_dim=specs.feature_dim,
        embed_size=Config.EMBED_SIZE,
        num_heads=Config.NUM_HEADS,
        num_layers=Config.NUM_LAYERS,
        ff_dim=Config.FF_DIM,
        dropout=Config.DROPOUT,
        max_len=cp.max_length
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")
    
    # 7. Create trainer and LOAD checkpoint
    trainer = Trainer(model, train_loader, val_loader, cp, Config)
    trainer.load_checkpoint(paths['model'])
    
    # 8. Resume training from checkpoint epoch
    trainer.train(resume=True)
    
    # 9. Evaluation
    print("\n" + "="*70)
    print("EVALUATING BEST MODEL")
    print("="*70)
    
    checkpoint = torch.load(paths['model'], map_location=Config.DEVICE)
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
    paths = Config.get_paths()
    specs = Config.get_spec()
    print("Loading model & features...")
    checkpoint = torch.load(paths['model'], map_location=Config.DEVICE)
    with open(paths['features'], 'rb') as f:
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
        feature_dim=specs.feature_dim,
        embed_size=Config.EMBED_SIZE,
        num_heads=Config.NUM_HEADS,
        num_layers=Config.NUM_LAYERS,
        ff_dim=Config.FF_DIM,
        dropout=Config.DROPOUT,
        max_len=cp.max_length
    )

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
    parser.add_argument('--variant', type=str, default='b2', 
                       choices=['b0', 'b1', 'b2', 'b3', 'b4'],
                       help='EfficientNet variant to use')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train','resume_train','inference','extract_features','research_eval'])
    parser.add_argument('--image_id', type=str, default=None)
    parser.add_argument('--beam', type=int, default=3)
    args = parser.parse_args()
    
    
    # Override config with CLI argument
    Config.EFFICIENTNET_VARIANT = args.variant
    specs = Config.get_spec()
    paths = Config.get_paths()

    if args.mode == 'train':
        main()
    elif args.mode == 'resume_train':
        resume_training()
    elif args.mode == 'extract_features':
        extractor = FeatureExtractorEfficientNet(variant=args.variant, device=Config.DEVICE)
        extractor.extract_features(Config.IMAGE_DIR, paths['features'])
    elif args.mode == 'research_eval':
        if not os.path.exists(paths['model']):
            print("No trained model found. Train first.")
        else:
            checkpoint = torch.load(paths['model'], map_location=Config.DEVICE)

            with open(paths['features'], 'rb') as f:
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
                feature_dim=specs.feature_dim,
                embed_size=Config.EMBED_SIZE,
                num_heads=Config.NUM_HEADS,
                num_layers=Config.NUM_LAYERS,
                ff_dim=Config.FF_DIM,
                dropout=Config.DROPOUT,
                max_len=cp.max_length
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
        if not os.path.exists(paths['model']):
            print("No trained model found. Train first.")
        else:
            load_and_inference(args.image_id)
    print("\nDone.")