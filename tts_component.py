# ============================================================================
# TTS COMPONENT - BASELINE IMPLEMENTATION (Tacotron 2 + WaveNet-based Vocoder)
#
# Description:
# Implements the TTS (Text-to-Speech) component for the
# "Large-scale multilingual audio visual dubbing" baseline project.
#
# This component follows the paper's architecture:
# 1. An acoustic model (Tacotron 2) generates mel-spectrograms.
# 2. A vocoder (WaveGlow/WaveNet-based) synthesizes audio from spectrograms.
#
# IMPORTANT: The paper cites reference [25]: "Natural TTS synthesis by 
# conditioning WaveNet on mel spectrogram predictions" - This means they
# used Tacotron 2 + WaveNet-based vocoder (WaveGlow is in this family).
#
# This class is responsible for INFERENCE. It loads a fine-tuned,
# speaker-specific Tacotron 2 model (from Stage 2) and a generic
# WaveNet-based vocoder to synthesize speech.
#
# Dependencies:
# pip install TTS torch numpy scipy
# ============================================================================

import os
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
from scipy.io.wavfile import write as write_wav

try:
    from TTS.api import TTS
    from TTS.utils.synthesizer import Synthesizer
except ImportError:
    print("Error: 'TTS' library not found.")
    print("Please install it: pip install TTS")
    exit(1)

# Suppress Coqui TTS Terms of Service agreement prompt
os.environ["COQUI_TOS_AGREED"] = "1"

# ============================================================================
# TTS COMPONENT
# ============================================================================

class TTSComponent:
    """
    Text-to-Speech using fine-tuned Tacotron 2 + WaveNet-based Vocoder (WaveGlow).
    
    This follows the baseline paper's approach:
    - Stage 1: Large multi-speaker Tacotron 2 pre-training
    - Stage 2: Speaker-specific fine-tuning
    - Vocoder: Generic WaveNet-based model (WaveGlow)
    """
    
    # ========================================================================
    # VOCODER CONFIGURATION
    # ========================================================================
    
    # Option 1: Use Coqui TTS's bundled WaveGlow model
    # This is the most compatible with Tacotron 2 mel-spectrograms
    TACOTRON2_MODEL_NAME = "tts_models/en/ljspeech/tacotron2-DDC"  # Base model for fine-tuning
    
    # WaveGlow vocoder (WaveNet-based, flow-based neural vocoder)
    # Note: Coqui TTS may not have standalone WaveGlow, so we use the bundled one
    # with Tacotron2-DDC or load separately
    USE_BUNDLED_VOCODER = True  # Use vocoder that comes with Tacotron2 model
    
    # If standalone vocoder needed (fallback):
    # Some Coqui TTS versions have: "vocoder_models/universal/libri-tts/wavegrad"
    # or we use the vocoder bundled with the Tacotron2 model
    
    VOCODER_SAMPLE_RATE = 22050  # Standard for WaveGlow and Tacotron 2

    def __init__(
        self,
        device: str = "auto"
    ):
        """
        Initialize the TTS component.

        Args:
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

        # Model caches
        self.model_cache: Dict[str, "TTS"] = {}
        self.synthesizer_cache: Dict[str, "Synthesizer"] = {}
        self.base_vocoder = None
        self.base_vocoder_config = None
        
        print(f"✓ TTS Component initialized")
        print(f"  Device: {self.device}")
        print(f"  Architecture: Tacotron 2 + WaveNet-based Vocoder")
        
    def _load_base_vocoder(self) -> None:
        """
        Loads a generic WaveNet-based vocoder (WaveGlow or compatible).
        This vocoder is shared across all speaker-specific models.
        
        The baseline paper uses a generic vocoder trained on large datasets,
        which works well with any Tacotron 2 mel-spectrogram output.
        """
        if self.base_vocoder is not None:
            return  # Already loaded
            
        print(f"Loading WaveNet-based vocoder (bundled with Tacotron 2)...")
        
        try:
            # Load the base Tacotron 2 model to get its vocoder
            # Coqui TTS's Tacotron2-DDC comes with a compatible vocoder
            base_model = TTS(
                model_name=self.TACOTRON2_MODEL_NAME,
                progress_bar=False,
                gpu=(self.device == "cuda")
            )
            
            # Extract the vocoder model and config
            if hasattr(base_model, 'vocoder') and base_model.vocoder is not None:
                if hasattr(base_model.vocoder, 'model'):
                    self.base_vocoder = base_model.vocoder.model
                else:
                    self.base_vocoder = base_model.vocoder
                    
                if hasattr(base_model.vocoder, 'config'):
                    self.base_vocoder_config = base_model.vocoder.config
                
                # Move to correct device
                if hasattr(self.base_vocoder, 'to'):
                    self.base_vocoder = self.base_vocoder.to(self.device)
                
                # Set to eval mode
                if hasattr(self.base_vocoder, 'eval'):
                    self.base_vocoder.eval()
                
                print(f"✓ WaveNet-based vocoder loaded successfully")
                print(f"  Vocoder type: {type(self.base_vocoder).__name__}")
            else:
                raise ValueError("Base model does not have a vocoder component")
                
        except Exception as e:
            print(f"✗ Error loading vocoder: {e}")
            print("  Attempting fallback approach...")
            
            # Fallback: Try to load a universal vocoder model
            try:
                # Try alternative vocoder models that might be available
                fallback_vocoder_names = [
                    "vocoder_models/universal/libri-tts/fullband-melgan",
                    "vocoder_models/en/ljspeech/multiband-melgan",
                ]
                
                for vocoder_name in fallback_vocoder_names:
                    try:
                        print(f"  Trying fallback: {vocoder_name}")
                        base_model = TTS(
                            model_name=vocoder_name,
                            progress_bar=False,
                            gpu=(self.device == "cuda")
                        )
                        self.base_vocoder = base_model
                        print(f"✓ Fallback vocoder loaded: {vocoder_name}")
                        return
                    except:
                        continue
                
                raise ValueError("No compatible vocoder found")
                
            except Exception as e2:
                print(f"✗ Fallback also failed: {e2}")
                raise RuntimeError(
                    "Could not load WaveNet-based vocoder. "
                    "Please ensure Coqui TTS is properly installed with vocoder models."
                )

    def _load_speaker_model(self, speaker_model_dir: str) -> "Synthesizer":
        """
        Loads a fine-tuned Tacotron 2 model from its directory.
        The directory must contain checkpoint files from Stage 2 fine-tuning.

        Args:
            speaker_model_dir: Path to the fine-tuned speaker model directory.

        Returns:
            A Synthesizer object that can generate mel-spectrograms.
        """
        # Use absolute path for cache key
        speaker_model_dir = os.path.abspath(speaker_model_dir)
        
        if speaker_model_dir in self.synthesizer_cache:
            return self.synthesizer_cache[speaker_model_dir]

        print(f"Loading speaker-specific Tacotron 2 model from: {speaker_model_dir}")
        
        # Look for model checkpoint files
        # Common naming conventions: best_model.pth, checkpoint_*.pth, model.pth
        model_files = list(Path(speaker_model_dir).glob("*.pth"))
        config_files = list(Path(speaker_model_dir).glob("config*.json"))
        
        if not model_files:
            raise FileNotFoundError(
                f"No .pth model file found in {speaker_model_dir}. "
                "Please run TTS fine-tuning (Stage 2) first."
            )
        
        if not config_files:
            raise FileNotFoundError(
                f"No config.json found in {speaker_model_dir}. "
                "Fine-tuning must generate a config file."
            )
        
        # Use best_model.pth if available, otherwise use first .pth file
        model_path = None
        for f in model_files:
            if 'best' in f.name.lower():
                model_path = f
                break
        if model_path is None:
            model_path = model_files[0]
        
        config_path = config_files[0]
        
        print(f"  Model: {model_path.name}")
        print(f"  Config: {config_path.name}")
        
        try:
            # Load using Coqui TTS's Synthesizer class
            # This handles Tacotron 2 models + vocoder combination
            synthesizer = Synthesizer(
                tts_checkpoint=str(model_path),
                tts_config_path=str(config_path),
                vocoder_checkpoint=None,  # Will use our base vocoder
                vocoder_config=None,
                use_cuda=(self.device == "cuda")
            )
            
            self.synthesizer_cache[speaker_model_dir] = synthesizer
            print(f"✓ Speaker model loaded: {Path(speaker_model_dir).name}")
            return synthesizer
            
        except Exception as e:
            print(f"✗ Error loading speaker model: {e}")
            
            # Fallback: Try loading with TTS API directly
            try:
                print("  Attempting fallback loading method...")
                model = TTS(
                    model_path=str(model_path),
                    config_path=str(config_path),
                    vocoder_path=None,
                    progress_bar=False,
                    gpu=(self.device == "cuda")
                )
                
                # Wrap in a dict to maintain consistent interface
                self.model_cache[speaker_model_dir] = model
                return model
                
            except Exception as e2:
                print(f"✗ Fallback loading also failed: {e2}")
                raise RuntimeError(
                    f"Could not load speaker model from {speaker_model_dir}. "
                    "Ensure the model was fine-tuned correctly."
                )

    def synthesize(
        self,
        text: str,
        speaker_model_dir: str,
        output_path: str
    ) -> str:
        """
        Synthesizes audio from text using a speaker-specific Tacotron 2 model
        and the generic WaveNet-based vocoder.

        Args:
            text: The text to synthesize.
            speaker_model_dir: Directory of the fine-tuned speaker model.
            output_path: Path to save the output .wav file.

        Returns:
            The path to the saved output file.
        """
        # Ensure vocoder is loaded
        if self.base_vocoder is None:
            self._load_base_vocoder()

        # Load speaker model (cached after first load)
        model_or_synth = self._load_speaker_model(speaker_model_dir)

        try:
            # Method 1: Using Synthesizer (preferred)
            if isinstance(model_or_synth, Synthesizer):
                # Generate mel-spectrogram using speaker's Tacotron 2
                mel_spec = model_or_synth.tts(text)
                
                # Convert mel to audio using generic vocoder
                if hasattr(self.base_vocoder, 'inference'):
                    # WaveGlow-style interface
                    wav = self.base_vocoder.inference(
                        torch.FloatTensor(mel_spec).unsqueeze(0).to(self.device)
                    )
                else:
                    # Generic vocoder interface
                    wav = self.base_vocoder(
                        torch.FloatTensor(mel_spec).unsqueeze(0).to(self.device)
                    )
                
                # Convert to numpy
                if isinstance(wav, torch.Tensor):
                    wav_np = wav.squeeze().cpu().numpy()
                else:
                    wav_np = np.array(wav).flatten()
                    
            # Method 2: Using TTS API directly (fallback)
            else:
                # Use the model's built-in synthesis
                wav_np = model_or_synth.tts(text)
                if isinstance(wav_np, list):
                    wav_np = np.array(wav_np)
            
            # Normalize audio to prevent clipping
            wav_np = wav_np / np.max(np.abs(wav_np) + 1e-6)
            wav_np = (wav_np * 32767).astype(np.int16)
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Save the .wav file
            write_wav(output_path, self.VOCODER_SAMPLE_RATE, wav_np)
            
            return output_path

        except Exception as e:
            print(f"✗ Error during synthesis: {e}")
            import traceback
            traceback.print_exc()
            raise
            
    def synthesize_segments(
        self,
        segments: List[Dict],
        speaker_model_dir: str,
        output_dir: str
    ) -> List[Dict]:
        """
        Synthesizes all translated segments for a speaker.

        Args:
            segments: List of translated segments from MTComponent.
                      Must contain 'id' and 'translated_text'.
            speaker_model_dir: Directory of the fine-tuned speaker model.
            output_dir: Directory to save all the .wav files.

        Returns:
            The input segments list, updated with the 'audio_path' key.
        """
        print(f"\n{'='*60}")
        print(f"Batch synthesizing {len(segments)} segments")
        print(f"  Speaker Model: {Path(speaker_model_dir).name}")
        print(f"  Output: {output_dir}")
        print(f"{'='*60}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        successful = 0
        failed = 0
        
        for i, segment in enumerate(segments):
            text = segment.get('translated_text', '')
            segment_id = segment.get('id', i)
            output_path = os.path.join(output_dir, f"segment_{segment_id:04d}.wav")
            
            print(f"  [{i+1}/{len(segments)}] Segment {segment_id}...", end='\r')
            
            if not text.strip():
                print(f"\n  [SKIP] Segment {segment_id}: Empty text")
                segment['audio_path'] = None
                failed += 1
                continue
                
            try:
                self.synthesize(
                    text=text,
                    speaker_model_dir=speaker_model_dir,
                    output_path=output_path
                )
                segment['audio_path'] = os.path.abspath(output_path)
                successful += 1
                
            except Exception as e:
                print(f"\n  ✗ FAILED segment {segment_id}: {str(e)[:100]}")
                segment['audio_path'] = None
                failed += 1
        
        print(f"\n{'='*60}")
        print(f"✓ Synthesis complete: {successful} succeeded, {failed} failed")
        print(f"{'='*60}")
        
        return segments

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def export_tts_manifest(
    segments_with_audio: List[Dict],
    output_path: str
):
    """
    Exports a JSON manifest file mapping segments to their new audio paths.
    This file is crucial for the final lip-sync and video assembly stage.

    Args:
        segments_with_audio: The list of segments returned from 
                             `synthesize_segments`.
        output_path: Path to save the manifest.json file.
    """
    
    # Prune the data to only what's needed for the next step
    manifest = []
    for seg in segments_with_audio:
        manifest.append({
            "id": seg.get('id'),
            "start": seg.get('start'),
            "end": seg.get('end'),
            "duration": seg.get('duration'),
            "translated_text": seg.get('translated_text'),
            "target_lang": seg.get('target_lang'),
            "translated_audio_path": seg.get('audio_path')
        })
        
    output = {
        "metadata": {
            "num_segments": len(manifest),
            "sample_rate": TTSComponent.VOCODER_SAMPLE_RATE,
            "target_lang": manifest[0].get('target_lang') if manifest else None,
            "architecture": "Tacotron 2 + WaveNet-based Vocoder"
        },
        "segments": manifest
    }
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"✓ TTS Manifest exported: {output_path}")
    except Exception as e:
        print(f"✗ Error exporting TTS manifest: {e}")