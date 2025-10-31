#!/usr/bin/env python3
"""
Train Speech Emotion Recognition (SER) Module

This script trains the emotion recognition module on audio datasets
for emotion-sensitive facial expression generation.

Supported datasets:
- RAVDESS: Ryerson Audio-Visual Database of Emotional Speech and Song
- CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset
- IEMOCAP: Interactive Emotional Dyadic Motion Capture database
- Custom dataset

Usage:
    python scripts/train_emotion_recognition.py --dataset ravdess --data_path /path/to/data
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
from tqdm import tqdm

from nerf_triplane.emotion_module import EmotionRecognitionModule


class AudioEmotionDataset(Dataset):
    """Dataset for audio emotion recognition."""
    
    def __init__(self, data_path, split='train', sample_rate=16000, duration=3.0):
        self.data_path = data_path
        self.split = split
        self.sample_rate = sample_rate
        self.duration = duration
        self.max_length = int(sample_rate * duration)
        
        # Load dataset
        self.samples = self._load_dataset()
        
        # Emotion mapping
        self.emotion_map = {
            'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3,
            'fearful': 4, 'disgusted': 5, 'surprised': 6
        }
    
    def _load_dataset(self):
        """Load dataset file list and labels."""
        samples = []
        
        # Example: RAVDESS format
        # File naming: 03-01-01-01-01-01-01.wav
        # 03 = modality (audio)
        # 01 = vocal channel
        # 01 = emotion (01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised)
        
        data_dir = Path(self.data_path) / self.split
        if not data_dir.exists():
            print(f"[WARN] Data directory not found: {data_dir}")
            return samples
        
        for audio_file in data_dir.glob('**/*.wav'):
            try:
                # Parse emotion from filename (RAVDESS format)
                parts = audio_file.stem.split('-')
                emotion_code = int(parts[2])
                
                # Map emotion code to emotion name
                emotion_names = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']
                emotion = emotion_names[emotion_code - 1]
                
                # Map calm to neutral
                if emotion == 'calm':
                    emotion = 'neutral'
                
                if emotion in self.emotion_map:
                    samples.append((str(audio_file), self.emotion_map[emotion]))
            
            except (IndexError, ValueError):
                # Skip files that don't match expected format
                continue
        
        print(f"[INFO] Loaded {len(samples)} samples for {self.split} split")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_path, emotion_label = self.samples[idx]
        
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Pad or truncate to fixed length
        if waveform.shape[1] < self.max_length:
            padding = self.max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, :self.max_length]
        
        return waveform.squeeze(0), emotion_label


def train_emotion_model(args):
    """Train the emotion recognition model."""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    # Create datasets
    train_dataset = AudioEmotionDataset(args.data_path, split='train')
    val_dataset = AudioEmotionDataset(args.data_path, split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = EmotionRecognitionModule(
        emotion_model=args.emotion_model,
        num_emotions=7,
        emotion_dim=args.emotion_dim,
        use_pretrained=args.use_pretrained
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
        for audio, labels in pbar:
            audio = audio.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(audio)
            loss = criterion(outputs['emotion_logits'], labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            predictions = outputs['emotion_class']
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*train_correct/train_total:.2f}%'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for audio, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Val]'):
                audio = audio.to(device)
                labels = labels.to(device)
                
                outputs = model(audio)
                loss = criterion(outputs['emotion_logits'], labels)
                
                val_loss += loss.item()
                predictions = outputs['emotion_class']
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # Update learning rate
        scheduler.step()
        
        # Print statistics
        print(f'\nEpoch {epoch+1}/{args.epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        print(f'  LR: {scheduler.get_last_lr()[0]:.6f}\n')
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            save_path = os.path.join(args.output_dir, 'best_emotion_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'emotion_model': args.emotion_model,
                'emotion_dim': args.emotion_dim,
            }, save_path)
            print(f'[INFO] Saved best model with val_acc={val_accuracy:.2f}% to {save_path}')
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f'emotion_model_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
            }, save_path)
    
    print(f'\n[INFO] Training completed. Best val accuracy: {best_val_acc:.2f}%')


def main():
    parser = argparse.ArgumentParser(description='Train Speech Emotion Recognition')
    
    # Dataset options
    parser.add_argument('--dataset', type=str, default='ravdess', 
                       choices=['ravdess', 'cremad', 'iemocap', 'custom'],
                       help='Dataset name')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset directory')
    
    # Model options
    parser.add_argument('--emotion_model', type=str, default='wav2vec2',
                       choices=['wav2vec2', 'cnn', 'prosody'],
                       help='Emotion recognition model type')
    parser.add_argument('--emotion_dim', type=int, default=64,
                       help='Emotion embedding dimension')
    parser.add_argument('--use_pretrained', action='store_true',
                       help='Use pretrained emotion model')
    
    # Training options
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='output/emotion_training',
                       help='Output directory for checkpoints')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train model
    train_emotion_model(args)


if __name__ == '__main__':
    main()

