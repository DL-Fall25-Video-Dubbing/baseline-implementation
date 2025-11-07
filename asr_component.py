# ============================================================================
# ASR COMPONENT - BASELINE IMPLEMENTATION
#
# Project: Large-scale Multilingual Audio-Visual Dubbing (Baseline)
#
# Description:
# Implements the ASR (Automatic Speech Recognition) component using
# OpenAI's Whisper Medium model (pre-trained, no fine-tuning).
#
# Purpose:
# - Extract audio from input video
# - Detect source language automatically
# - Transcribe spoken content with timestamps
# - Support 5 languages: English, Spanish, French, German, Russian
#
# Input:  Video file (any format supported by ffmpeg)
# Output: Transcription with word-level timestamps
#
# Dependencies:
# pip install openai-whisper torch numpy
# ffmpeg (system installation)
# ============================================================================

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import whisper
import numpy as np

# ============================================================================
# ASR COMPONENT
# ============================================================================

class ASRComponent:
    """
    Automatic Speech Recognition using Whisper Medium (Pre-trained)
    
    Purpose: First stage of video dubbing pipeline
    - Extracts and transcribes audio from input video
    - Detects source language automatically  
    - Provides timestamped transcription for downstream processing
    
    Supported Languages (Baseline): en, es, fr, de, ru
    Model: OpenAI Whisper Medium (no training, fully pre-trained)
    
    Pipeline Role:
    Video → [ASR] → Transcription → MT → TTS → Lipsync → Dubbed Video
    """

    SUPPORTED_LANGUAGES = {
        'en': 'english',
        'fr': 'french',
        'es': 'spanish',
        'de': 'german',
        'ru': 'russian'
    }

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "auto"
    ):
        """
        Initialize ASR component with Whisper model

        Args:
            model_size: Whisper model size
                        ('tiny', 'base', 'small', 'medium', 'large', 'large-v3')
            device: Device to use ('cuda', 'cpu', 'auto')
        """
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Handle MPS (Apple Silicon) case
        if self.device == "cuda" and not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                print("CUDA not found, falling back to MPS.")
                self.device = "mps"
            else:
                print("CUDA not found, falling back to CPU.")
                self.device = "cpu"

        print(f"Loading Whisper model: {model_size} on {self.device}")
        
        # Load the Whisper model
        self.model = whisper.load_model(model_size, device=self.device)
        self.model_size = model_size

        print(f"✓ ASR Component initialized")
        print(f"  Model: Whisper-{model_size}")
        print(f"  Device: {self.device}")
        print(f"  Supported languages: {list(self.SUPPORTED_LANGUAGES.keys())}")

    def transcribe(
        self,
        audio_path: str,
        language: str = None,
        task: str = "transcribe",
        return_timestamps: bool = True,
        verbose: bool = False
    ) -> Dict:
        """
        [CORE METHOD] Transcribe audio/video to text with timestamps
        
        This is the main ASR method for the dubbing pipeline.
        Used to extract source language text from input video.

        Args:
            audio_path: Path to audio or video file
            language: Language code ('en', 'es', 'fr', 'de', 'ru')
                      If None, auto-detect source language
            task: 'transcribe' (keep same language, default for pipeline)
            return_timestamps: Always True for lipsync alignment
            verbose: Print detailed progress

        Returns:
            Dictionary with:
                - text: Full transcription text
                - segments: List of timestamped segments (for alignment)
                - language: Detected/specified language code
                - duration: Total audio duration in seconds
                - num_segments: Number of segments
        """
        # Validate language
        if language and language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Language '{language}' not supported. "
                f"Choose from: {list(self.SUPPORTED_LANGUAGES.keys())}"
            )

        # Validate file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"\n{'='*60}")
        print(f"Transcribing: {Path(audio_path).name}")
        print(f"Language: {language if language else 'auto-detect'}")
        print(f"{'='*60}")

        # Transcribe

        # segment-level is sufficient and matches `result['segments']`.
        result = self.model.transcribe(
            audio_path,
            language=language,
            task=task,
            verbose=verbose
            # word_timestamps=return_timestamps # Standard whisper returns segment timestamps
        )

        # Extract metadata
        duration = result['segments'][-1]['end'] if result['segments'] else 0
        
        transcription = {
            'text': result['text'].strip(),
            'segments': self._format_segments(result['segments']),
            'language': result['language'],
            'duration': duration,
            'num_segments': len(result['segments'])
        }

        print(f"\n✓ Transcription complete")
        print(f"  Language: {transcription['language']}")
        print(f"  Duration: {transcription['duration']:.2f}s")
        print(f"  Segments: {transcription['num_segments']}")
        print(f"  Text length: {len(transcription['text'])} chars")

        return transcription

    def _format_segments(self, segments: List[Dict]) -> List[Dict]:
        """Format segments for consistent output"""
        formatted = []
        for seg in segments:
            start = round(seg['start'], 2)
            end = round(seg['end'], 2)
            formatted.append({
                'id': seg['id'],
                'start': start,
                'end': end,
                'text': seg['text'].strip(),
                'duration': round(end - start, 2)
            })
        return formatted

    def detect_language(self, audio_path: str) -> Tuple[str, float]:
        """
        [CORE METHOD] Detect language of audio/video file
        
        Used to identify source language before transcription.

        Args:
            audio_path: Path to audio or video file

        Returns:
            (language_code, confidence): e.g., ('en', 0.99)
        """
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.device)
        
        _, probs = self.model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        confidence = probs[detected_lang]

        print(f"\nLanguage Detection:")
        print(f"  Detected: {detected_lang} ({confidence:.2%})")

        return detected_lang, confidence

    def get_model_info(self) -> Dict:
        """Get information about loaded model"""
        return {
            'model_name': f"Whisper-{self.model_size}",
            'device': self.device,
            'parameters': sum(p.numel() for p in self.model.parameters()) / 1e6,
            'supported_languages': self.SUPPORTED_LANGUAGES
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_audio_from_video(
    video_path: str,
    output_path: Optional[str] = None,
    sample_rate: int = 16000
) -> str:
    """
    [CORE UTILITY] Extract audio from video using ffmpeg
    
    Essential for video dubbing pipeline.

    Args:
        video_path: Path to input video file
        output_path: Output audio path (auto-generated if None)
        sample_rate: Audio sample rate (16000 Hz for Whisper)

    Returns:
        Path to extracted audio file (.wav format)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_path is None:
        output_path = str(Path(video_path).with_suffix('.wav'))

    cmd = [
        'ffmpeg', '-i', video_path,
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # PCM 16-bit
        '-ar', str(sample_rate),  # Sample rate
        '-ac', '1',  # Mono
        '-y',  # Overwrite
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"✓ Audio extracted: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"✗ FFMPEG Error extracting audio from {video_path}:")
        print(e.stderr.decode())
        raise
    except FileNotFoundError:
        print("✗ FFMPEG Error: 'ffmpeg' command not found.")
        print("  Please ensure ffmpeg is installed and in the system's PATH.")
        raise


def print_transcription(transcription: Dict, max_segments: int = 10):
    """
    Pretty print transcription results

    Args:
        transcription: Result from ASRComponent.transcribe()
        max_segments: Maximum segments to display
    """
    print(f"\n{'='*60}")
    print(f"TRANSCRIPTION RESULT")
    print(f"{'='*60}")
    print(f"Language: {transcription['language']}")
    print(f"Duration: {transcription['duration']:.2f}s")
    print(f"Segments: {transcription['num_segments']}")
    print(f"\n{'='*60}")
    print(f"FULL TEXT:")
    print(f"{'='*60}")
    print(transcription['text'])
    print(f"\n{'='*60}")
    print(f"SEGMENTS (showing first {max_segments}):")
    print(f"{'='*60}")

    for i, seg in enumerate(transcription['segments'][:max_segments], 1):
        start = f"{seg['start']:.2f}s"
        end = f"{seg['end']:.2f}s"
        print(f"{i:02d}. [{start} --> {end}]")
        print(f"    {seg['text']}")

    if transcription['num_segments'] > max_segments:
        print(f"\n... ({transcription['num_segments'] - max_segments} more segments)")


def export_to_srt(transcription: Dict, output_path: str):
    """
    Export transcription to SRT subtitle format
    
    Useful for visualization and debugging.

    Args:
        transcription: Result from ASRComponent.transcribe()
        output_path: Path to save SRT file
    """
    def format_timestamp(seconds: float) -> str:
        """Format seconds to SRT format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(transcription['segments'], 1):
            f.write(f"{i}\n")
            start = format_timestamp(segment['start'])
            end = format_timestamp(segment['end'])
            f.write(f"{start} --> {end}\n")
            f.write(f"{segment['text'].strip()}\n\n")

    print(f"✓ SRT exported: {output_path}")