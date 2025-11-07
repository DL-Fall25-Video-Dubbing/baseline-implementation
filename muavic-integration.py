"""
MuAViC Data Loading - Complete Integration Example
===================================================

This script demonstrates how to:
1. Load preprocessed MuAViC data
2. Create a PyTorch Dataset
3. Extract audio features
4. Set up training pipeline

Usage:
    python muavic_integration_example.py --processed_dir /path/to/processed_data

Author: Based on arXiv:2011.03530v1
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import librosa
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse


# ============================================================================
# 1. AUDIO FEATURE EXTRACTION
# ============================================================================

class AudioFeatureExtractor:
    """
    Extract mel-spectrogram features for lip-sync training
    """
    def __init__(
        self,
        sr: int = 16000,
        n_mels: int = 64,
        hop_length: int = 512,
        n_fft: int = 2048,
        window_duration: float = 0.5
    ):
        """
        Args:
            sr: Sample rate
            n_mels: Number of mel-frequency bins
            hop_length: Hop length for STFT
            n_fft: FFT window size
            window_duration: Audio window duration in seconds
        """
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.window_duration = window_duration
        
        # Cache loaded audio files
        self._audio_cache = {}
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """Load audio file with caching"""
        audio_path = str(audio_path)
        
        if audio_path not in self._audio_cache:
            audio, _ = librosa.load(audio_path, sr=self.sr)
            self._audio_cache[audio_path] = audio
        
        return self._audio_cache[audio_path]
    
    def extract_features(
        self,
        audio_path: str,
        frame_idx: int,
        fps: float = 25.0
    ) -> np.ndarray:
        """
        Extract mel-spectrogram features for a specific frame
        
        Args:
            audio_path: Path to audio file
            frame_idx: Frame index
            fps: Video frames per second
        
        Returns:
            mel_features: [n_mels, time_steps] array
        """
        # Load audio
        audio = self.load_audio(audio_path)
        
        # Calculate time window for this frame
        frame_time = frame_idx / fps
        
        # Extract audio segment centered on frame time
        start_sample = int((frame_time - self.window_duration/2) * self.sr)
        end_sample = int((frame_time + self.window_duration/2) * self.sr)
        
        # Clip to valid range
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)
        
        # Pad if necessary
        audio_segment = audio[start_sample:end_sample]
        target_length = int(self.window_duration * self.sr)
        
        if len(audio_segment) < target_length:
            audio_segment = np.pad(
                audio_segment,
                (0, target_length - len(audio_segment)),
                mode='constant'
            )
        
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_segment,
            sr=self.sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / \
                       (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        return mel_spec_norm
    
    def clear_cache(self):
        """Clear audio cache to free memory"""
        self._audio_cache.clear()


# ============================================================================
# 2. PYTORCH DATASET
# ============================================================================

class MuAViCLipsyncDataset(Dataset):
    """
    PyTorch Dataset for MuAViC lip-sync training
    
    Each sample contains:
    - masked_input: Lower-face masked frame [3, 256, 256]
    - ground_truth: Original face frame [3, 256, 256]
    - audio_features: Mel-spectrogram [64, time_steps]
    - landmarks: Facial landmarks [478, 3]
    """
    
    def __init__(
        self,
        processed_dirs: List[str],
        audio_extractor: Optional[AudioFeatureExtractor] = None,
        transform=None,
        max_frames_per_video: Optional[int] = None
    ):
        """
        Args:
            processed_dirs: List of paths to preprocessed video directories
            audio_extractor: AudioFeatureExtractor instance
            transform: Optional transforms for data augmentation
            max_frames_per_video: Limit frames per video (for memory)
        """
        self.transform = transform
        self.max_frames_per_video = max_frames_per_video
        
        # Initialize audio extractor
        self.audio_extractor = audio_extractor or AudioFeatureExtractor()
        
        # Collect all frame samples
        self.samples = []
        self._collect_samples(processed_dirs)
        
        print(f"✓ Dataset initialized with {len(self.samples)} frames")
    
    def _collect_samples(self, processed_dirs: List[str]):
        """Collect all valid frame samples from processed directories"""
        for proc_dir in processed_dirs:
            proc_path = Path(proc_dir)
            
            # Load metadata
            meta_files = list(proc_path.glob("*_meta.json"))
            if not meta_files:
                print(f"Warning: No metadata in {proc_path}")
                continue
            
            with open(meta_files[0], 'r') as f:
                metadata = json.load(f)
            
            # Load NPZ arrays
            npz_file = proc_path / f"{meta_files[0].stem.replace('_meta', '_data')}.npz"
            if not npz_file.exists():
                print(f"Warning: No NPZ file in {proc_path}")
                continue
            
            npz_data = np.load(npz_file)
            
            # Get valid frames
            landmarks_valid = npz_data['landmarks_valid']
            num_frames = metadata['num_frames']
            
            # Extract audio path from video path
            video_path = Path(metadata['video_path'])
            audio_path = video_path  # Assuming audio is in video file
            
            # Check which frames have valid data
            cropped_dir = proc_path / 'cropped_faces'
            masked_dir = proc_path / 'masked_inputs'
            
            if not cropped_dir.exists() or not masked_dir.exists():
                print(f"Warning: Missing image directories in {proc_path}")
                continue
            
            # Limit frames if specified
            frame_indices = range(num_frames)
            if self.max_frames_per_video:
                # Sample evenly
                step = max(1, num_frames // self.max_frames_per_video)
                frame_indices = range(0, num_frames, step)
            
            # Add samples for valid frames
            for frame_idx in frame_indices:
                if frame_idx >= num_frames:
                    break
                
                if not landmarks_valid[frame_idx]:
                    continue
                
                # Check if frame files exist
                frame_name = f"frame_{frame_idx:06d}.jpg"
                cropped_path = cropped_dir / frame_name
                masked_path = masked_dir / frame_name
                
                if not cropped_path.exists() or not masked_path.exists():
                    continue
                
                self.samples.append({
                    'video_id': proc_path.name,
                    'frame_idx': frame_idx,
                    'fps': metadata['fps'],
                    'audio_path': str(audio_path),
                    'cropped_path': str(cropped_path),
                    'masked_path': str(masked_path),
                    'npz_path': str(npz_file),
                    'landmarks_idx': frame_idx  # Index in NPZ array
                })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load images
        masked_input = cv2.imread(sample['masked_path'])
        masked_input = cv2.cvtColor(masked_input, cv2.COLOR_BGR2RGB)
        
        ground_truth = cv2.imread(sample['cropped_path'])
        ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2RGB)
        
        # Load landmarks
        npz_data = np.load(sample['npz_path'])
        landmarks = npz_data['landmarks_smoothed'][sample['landmarks_idx']]
        
        # Extract audio features
        audio_features = self.audio_extractor.extract_features(
            audio_path=sample['audio_path'],
            frame_idx=sample['frame_idx'],
            fps=sample['fps']
        )
        
        # Convert to tensors
        masked_input = torch.from_numpy(masked_input).permute(2, 0, 1).float() / 255.0
        ground_truth = torch.from_numpy(ground_truth).permute(2, 0, 1).float() / 255.0
        landmarks = torch.from_numpy(landmarks).float()
        audio_features = torch.from_numpy(audio_features).float()
        
        # Apply transforms if specified
        if self.transform:
            masked_input = self.transform(masked_input)
            ground_truth = self.transform(ground_truth)
        
        return {
            'masked_input': masked_input,      # [3, 256, 256]
            'ground_truth': ground_truth,      # [3, 256, 256]
            'audio_features': audio_features,  # [64, time_steps]
            'landmarks': landmarks,            # [478, 3]
            'video_id': sample['video_id'],
            'frame_idx': sample['frame_idx']
        }


# ============================================================================
# 3. TRAINING UTILITIES
# ============================================================================

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle variable-length audio features
    """
    # Find max time steps in audio features
    max_time_steps = max(item['audio_features'].shape[1] for item in batch)
    
    # Pad audio features to same length
    padded_audio = []
    for item in batch:
        audio = item['audio_features']
        if audio.shape[1] < max_time_steps:
            padding = torch.zeros(audio.shape[0], max_time_steps - audio.shape[1])
            audio = torch.cat([audio, padding], dim=1)
        padded_audio.append(audio)
    
    return {
        'masked_input': torch.stack([item['masked_input'] for item in batch]),
        'ground_truth': torch.stack([item['ground_truth'] for item in batch]),
        'audio_features': torch.stack(padded_audio),
        'landmarks': torch.stack([item['landmarks'] for item in batch]),
        'video_id': [item['video_id'] for item in batch],
        'frame_idx': [item['frame_idx'] for item in batch]
    }


def create_dataloaders(
    train_dirs: List[str],
    val_dirs: List[str],
    batch_size: int = 16,
    num_workers: int = 4,
    max_frames_per_video: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders
    """
    # Create audio extractor (shared between datasets)
    audio_extractor = AudioFeatureExtractor()
    
    # Create datasets
    train_dataset = MuAViCLipsyncDataset(
        processed_dirs=train_dirs,
        audio_extractor=audio_extractor,
        max_frames_per_video=max_frames_per_video
    )
    
    val_dataset = MuAViCLipsyncDataset(
        processed_dirs=val_dirs,
        audio_extractor=audio_extractor,
        max_frames_per_video=max_frames_per_video
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


# ============================================================================
# 4. EXAMPLE USAGE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='MuAViC Data Loading Example')
    parser.add_argument('--processed_dir', type=str, required=True,
                       help='Path to processed data directory')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Max frames per video (for memory constraints)')
    
    args = parser.parse_args()
    
    # Find all processed video directories
    processed_root = Path(args.processed_dir)
    all_processed = [d for d in processed_root.iterdir() if d.is_dir()]
    
    print(f"Found {len(all_processed)} processed videos")
    
    # Split into train/val (80/20)
    split_idx = int(0.8 * len(all_processed))
    train_dirs = [str(d) for d in all_processed[:split_idx]]
    val_dirs = [str(d) for d in all_processed[split_idx:]]
    
    print(f"Train videos: {len(train_dirs)}")
    print(f"Val videos: {len(val_dirs)}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_dirs=train_dirs,
        val_dirs=val_dirs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_frames_per_video=args.max_frames
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test loading a batch
    print("\nTesting data loading...")
    batch = next(iter(train_loader))
    
    print("\nBatch contents:")
    print(f"  masked_input: {batch['masked_input'].shape}")
    print(f"  ground_truth: {batch['ground_truth'].shape}")
    print(f"  audio_features: {batch['audio_features'].shape}")
    print(f"  landmarks: {batch['landmarks'].shape}")
    print(f"  video_ids: {len(batch['video_id'])} items")
    
    print("\n✓ Data loading successful!")
    
    # Example: Iterate through a few batches
    print("\nIterating through 5 batches...")
    for i, batch in enumerate(train_loader):
        if i >= 5:
            break
        print(f"  Batch {i+1}: "
              f"masked={batch['masked_input'].shape}, "
              f"audio={batch['audio_features'].shape}")
    
    print("\n✓ All tests passed!")


if __name__ == '__main__':
    main()


# ============================================================================
# 5. QUICK START EXAMPLE (Without Arguments)
# ============================================================================

def quick_start_example():
    """
    Quick start example - no command line arguments needed
    Modify paths below to match your setup
    """
    print("="*60)
    print("MuAViC Data Loading - Quick Start Example")
    print("="*60)
    
    # MODIFY THESE PATHS
    PROCESSED_DATA_DIR = "/kaggle/working/processed_data"
    
    # Find processed videos
    processed_root = Path(PROCESSED_DATA_DIR)
    
    if not processed_root.exists():
        print(f"Error: {PROCESSED_DATA_DIR} not found!")
        print("Please run preprocessing first (Section 2.3 in notebook)")
        return
    
    all_processed = [d for d in processed_root.iterdir() if d.is_dir()]
    
    if len(all_processed) == 0:
        print(f"Error: No processed videos found in {PROCESSED_DATA_DIR}")
        print("Please run preprocessing first (Section 2.3 in notebook)")
        return
    
    print(f"\n✓ Found {len(all_processed)} processed videos")
    
    # Create simple dataset with one video
    print("\nCreating dataset with first video...")
    dataset = MuAViCLipsyncDataset(
        processed_dirs=[str(all_processed[0])],
        max_frames_per_video=100  # Limit to 100 frames for testing
    )
    
    print(f"✓ Dataset has {len(dataset)} frames")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )
    
    # Load one batch
    print("\nLoading first batch...")
    batch = next(iter(dataloader))
    
    print("\n✓ Batch loaded successfully!")
    print(f"  Masked input shape: {batch['masked_input'].shape}")
    print(f"  Ground truth shape: {batch['ground_truth'].shape}")
    print(f"  Audio features shape: {batch['audio_features'].shape}")
    print(f"  Landmarks shape: {batch['landmarks'].shape}")
    
    print("\n" + "="*60)
    print("✓ Quick start successful!")
    print("="*60)


# Uncomment to run quick start
# quick_start_example()