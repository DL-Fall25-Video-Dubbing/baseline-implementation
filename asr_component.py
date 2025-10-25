# ============================================================================
# ASR COMPONENT - BASELINE IMPLEMENTATION
#
#
# Description:
# Implements the ASR (Automatic Speech Recognition) component for the
# "Large-scale multilingual audio visual dubbing" baseline project.
#
# This component uses OpenAI's Whisper model as a robust, multilingual
# ASR system, as discussed in the project plan.
#
# Dependencies:
# pip install openai-whisper torch numpy
# (And ffmpeg)
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
    Automatic Speech Recognition using Whisper (Baseline)
    Supports: English, French, Spanish, German, Russian

    Based on DeepMind baseline approach: Use pretrained ASR per language
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
        Transcribe audio file to text with timestamps

        Args:
            audio_path: Path to audio/video file
            language: Language code ('en', 'fr', 'es', 'de', 'ru')
                      If None, auto-detect
            task: 'transcribe' or 'translate' (to English)
            return_timestamps: Include word-level timestamps
            verbose: Print progress

        Returns:
            Dictionary with:
                - text: Full transcription
                - segments: List of segments with timestamps
                - language: Detected/specified language
                - duration: Audio duration in seconds
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
        """
        Format segments for consistent output

        Args:
            segments: Raw segments from Whisper

        Returns:
            Formatted segments with start, end, text
        """
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

    def transcribe_with_speaker_diarization(
        self,
        audio_path: str,
        language: str = None
    ) -> Dict:
        """
        Transcribe with simple speaker diarization
        (For videos with single speaker - baseline approach)

        Args:
            audio_path: Path to audio file
            language: Language code

        Returns:
            Transcription with speaker info
        """
        result = self.transcribe(audio_path, language=language)

        # Baseline: Assume single speaker
        # (DeepMind paper mentions "single speaker videos")
        for segment in result['segments']:
            segment['speaker'] = 'SPEAKER_01'

        return result

    def batch_transcribe(
        self,
        audio_paths: List[str],
        language: str = None,
        save_dir: Optional[str] = None
    ) -> List[Dict]:
        """
        Batch transcribe multiple files

        Args:
            audio_paths: List of audio file paths
            language: Language code (same for all)
            save_dir: Directory to save transcriptions (optional)

        Returns:
            List of transcription results
        """
        results = []

        for i, audio_path in enumerate(audio_paths, 1):
            print(f"\n[{i}/{len(audio_paths)}] Processing: {Path(audio_path).name}")

            try:
                result = self.transcribe(audio_path, language=language)
                results.append({
                    'file': audio_path,
                    'success': True,
                    'transcription': result
                })

                # Save individual result
                if save_dir:
                    self._save_transcription(result, audio_path, save_dir)

            except Exception as e:
                print(f"✗ Error: {str(e)}")
                results.append({
                    'file': audio_path,
                    'success': False,
                    'error': str(e)
                })

        # Summary
        successful = sum(1 for r in results if r['success'])
        print(f"\n{'='*60}")
        print(f"Batch complete: {successful}/{len(audio_paths)} successful")
        print(f"{'='*60}")

        return results

    def _save_transcription(
        self,
        transcription: Dict,
        audio_path: str,
        save_dir: str
    ):
        """Save transcription to JSON file"""
        os.makedirs(save_dir, exist_ok=True)

        filename = Path(audio_path).stem + '_transcript.json'
        save_path = os.path.join(save_dir, filename)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(transcription, f, ensure_ascii=False, indent=2)

        print(f"  Saved: {save_path}")

    def detect_language(self, audio_path: str) -> Tuple[str, float]:
        """
        Detect language of audio file

        Args:
            audio_path: Path to audio file

        Returns:
            (language_code, confidence)
        """
        # Load audio
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)

        # Make log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(self.device)

        # Detect language
        _, probs = self.model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        confidence = probs[detected_lang]

        print(f"\nLanguage Detection:")
        print(f"  Detected: {detected_lang} ({confidence:.2%})")

        return detected_lang, confidence

    def extract_utterances(
        self,
        transcription: Dict,
        max_duration: float = 12.0,
        min_duration: float = 1.0
    ) -> List[Dict]:
        """
        Split transcription into utterances (for baseline training)
        This mimics the paper's data processing pipeline, where long
        tracks are split into smaller chunks.

        Args:
            transcription: Result from transcribe()
            max_duration: Maximum utterance duration (seconds)
            min_duration: Minimum utterance duration (seconds)

        Returns:
            List of utterances with start, end, text
        """
        utterances = []
        current_utterance = {
            'start': 0.0,
            'end': 0.0,
            'text': ''
        }

        for segment in transcription['segments']:
            segment_duration = segment['duration']
            text = segment['text']

            # Start new utterance if current is empty
            if not current_utterance['text']:
                current_utterance['start'] = segment['start']
                current_utterance['end'] = segment['end']
                current_utterance['text'] = text
                continue

            # Check if adding this segment exceeds max duration
            potential_duration = segment['end'] - current_utterance['start']

            if potential_duration > max_duration:
                # Finalize current utterance if it's valid
                if current_utterance['end'] - current_utterance['start'] >= min_duration:
                    current_utterance['text'] = current_utterance['text'].strip()
                    utterances.append(current_utterance.copy())

                # Start new utterance
                current_utterance = {
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': text
                }
            else:
                # Add to current utterance
                current_utterance['end'] = segment['end']
                current_utterance['text'] += ' ' + text

        # Add final utterance
        if current_utterance['text'] and \
           current_utterance['end'] - current_utterance['start'] >= min_duration:
            current_utterance['text'] = current_utterance['text'].strip()
            utterances.append(current_utterance)

        print(f"\nExtracted {len(utterances)} utterances")
        if utterances:
            print(f"  Avg duration: {np.mean([u['end']-u['start'] for u in utterances]):.2f}s")

        return utterances

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
    Extract audio from video file using ffmpeg.

    Args:
        video_path: Path to video file
        output_path: Output audio path (auto-generated if None)
        sample_rate: Audio sample rate (Whisper requires 16000)

    Returns:
        Path to extracted audio file
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_path is None:
        # Save audio in a predictable way, e.g., 'video_name.wav'
        output_path = str(Path(video_path).with_suffix('.wav'))

    # Use ffmpeg to extract audio
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
        # Using DEVNULL for stdout, but capturing stderr for errors
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"✓ Audio extracted: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"✗ FFMPEG Error extracting audio from {video_path}:")
        print(e.stderr.decode())
        raise
    except FileNotFoundError:
        print("✗ FFMPEG Error: 'ffmpeg' command not found.")
        print("  Please ensure ffmpeg is installed and in your system's PATH.")
        raise


def format_timestamp(seconds: float, use_comma: bool = True) -> str:
    """
    Format seconds to HH:MM:SS,mmm (SRT format) or HH:MM:SS.mmm

    Args:
        seconds: Time in seconds
        use_comma: True for SRT format (HH:MM:SS,mmm), False (HH:MM:SS.mmm)

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    delimiter = ',' if use_comma else '.'
    
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', delimiter)


def export_to_srt(transcription: Dict, output_path: str):
    """
    Export transcription to SRT subtitle format

    Args:
        transcription: Result from ASRComponent.transcribe()
        output_path: Path to save SRT file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(transcription['segments'], 1):
            # Subtitle index
            f.write(f"{i}\n")

            # Timestamps
            start = format_timestamp(segment['start'], use_comma=True)
            end = format_timestamp(segment['end'], use_comma=True)
            f.write(f"{start} --> {end}\n")

            # Text
            f.write(f"{segment['text'].strip()}\n\n")

    print(f"✓ SRT exported: {output_path}")


def export_to_json(transcription: Dict, output_path: str):
    """
    Export transcription to JSON format

    Args:
        transcription: Result from ASRComponent.transcribe()
        output_path: Path to save JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transcription, f, ensure_ascii=False, indent=2)

    print(f"✓ JSON exported: {output_path}")


def print_transcription(transcription: Dict, max_segments: int = 10):
    """
    Pretty print transcription

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
        timestamp = f"[{format_timestamp(seg['start'], use_comma=False)} --> {format_timestamp(seg['end'], use_comma=False)}]"
        print(f"{i:02d}. {timestamp}")
        print(f"   {seg['text']}")
        print() # Add a newline for readability

    if transcription['num_segments'] > max_segments:
        print(f"... ({transcription['num_segments'] - max_segments} more segments)")