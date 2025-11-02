# ============================================================================
# TTS COMPONENT - BASELINE IMPLEMENTATION (WaveNet-based Vocoder)
#
# Description:
# Implements the TTS (Text-to-Speech) component for the
# "Large-scale multilingual audio visual dubbing" baseline project.
#
# This component follows the paper's architecture:
# 1. An acoustic model (Tacotron 2) generates spectrograms.
# 2. A vocoder (WaveGlow) synthesizes audio from spectrograms.
#
# This implementation uses a WaveGlow vocoder, which is a flow-based
# model in the same family as WaveNet, to align with the paper's
# citation [25] (which points to a WaveNet-based vocoder).
#
# This class is responsible for INFERENCE. It loads a fine-tuned,
# speaker-specific Tacotron 2 model (from Stage 2) and a generic
# vocoder to synthesize speech.
#
# The actual Stage 2 fine-tuning is handled in a separate script.
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
    Text-to-Speech using a fine-tuned Tacotron 2 + Vocoder.
    This class is responsible for inference, loading the models
    created by the paper's Stage 2 fine-tuning process.
    """
    
    # --- MODEL UPDATE ---
    # Changed from HiFi-GAN to a WaveNet-based (WaveGlow) vocoder
    # to match the paper's citation [25] and your implementation plan.
    # This model was also trained on LJSpeech and is compatible with Tacotron 2.
    GENERIC_VOCODER_NAME = "vocoder_models/en/ljspeech/waveglow-librosa"
    VOCODER_SAMPLE_RATE = 22050  # This model also uses 22050Hz

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

        # Caches to hold loaded models in memory
        self.model_cache: Dict[str, "TTS"] = {}
        self.vocoder: Optional[torch.nn.Module] = None
        
        print(f"✓ TTS Component initialized")
        print(f"  Device: {self.device}")
        
    def _load_vocoder(self) -> None:
        """
        Loads the generic vocoder into memory.
        This is done once and shared by all speaker models.
        """
        if self.vocoder is None:
            print(f"Loading generic WaveNet-based vocoder: {self.GENERIC_VOCODER_NAME}...")
            # We load the vocoder by loading a full TTS model
            # and then just grabbing its vocoder component.
            try:
                tts_instance_for_vocoder = TTS(self.GENERIC_VOCODER_NAME, progress_bar=False).to(self.device)
                
                # We only need the vocoder model, not the full TTS class
                self.vocoder = tts_instance_for_vocoder.vocoder.model
                self.vocoder.remove_pre_net()  # Prepare for stand-alone use
                self.vocoder.eval()
                
                print(f"✓ Generic WaveNet-based vocoder loaded to {self.device}")
            except Exception as e:
                print(f"✗ Error loading generic vocoder: {e}")
                print("  Please ensure you have an internet connection to download the model.")
                raise

    def _load_speaker_model(self, speaker_model_dir: str) -> "TTS":
        """
        Loads a fine-tuned Tacotron 2 model from its directory.
        The directory must contain 'best_model.pth' and 'config.json'.

        Args:
            speaker_model_dir: Path to the fine-tuned speaker model directory.

        Returns:
            A loaded TTS (Tacotron 2) model instance.
        """
        # Use absolute path for cache key
        speaker_model_dir = os.path.abspath(speaker_model_dir)
        
        if speaker_model_dir in self.model_cache:
            return self.model_cache[speaker_model_dir]

        print(f"Loading speaker-specific model from: {speaker_model_dir}")
        
        model_path = os.path.join(speaker_model_dir, "best_model.pth")
        config_path = os.path.join(speaker_model_dir, "config.json")

        if not os.path.exists(model_path) or not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Model files not found in {speaker_model_dir}. "
                "Expected 'best_model.pth' and 'config.json'. "
                "Please run the fine-tuning script first."
            )
            
        try:
            # Load the Tacotron 2 model *without* a vocoder
            # (we will use our generic one)
            model = TTS(
                model_path=model_path,
                config_path=config_path,
                vocoder_path=None,  # We provide the vocoder manually
                progress_bar=False
            ).to(self.device)
            
            # Set to evaluation mode
            model.eval()
            
            self.model_cache[speaker_model_dir] = model
            print(f"✓ Speaker model loaded for: {os.path.basename(speaker_model_dir)}")
            return model
            
        except Exception as e:
            print(f"✗ Error loading speaker model from {speaker_model_dir}: {e}")
            raise

    def synthesize(
        self,
        text: str,
        speaker_model_dir: str,
        output_path: str
    ) -> str:
        """
        Synthesizes audio from text using a speaker-specific model.

        Args:
            text: The text to synthesize.
            speaker_model_dir: Directory of the fine-tuned speaker model.
            output_path: Path to save the output .wav file.

        Returns:
            The path to the saved output file.
        """
        # Ensure vocoder is loaded
        if self.vocoder is None:
            self._load_vocoder()

        # Ensure speaker model is loaded
        model = self._load_speaker_model(speaker_model_dir)

        # print(f"Synthesizing text for speaker: {os.path.basename(speaker_model_dir)}")

        try:
            # 1. Generate Spectrogram (Tacotron 2)
            # We use the internal `tts` method
            outputs = model.tts(text, speaker=None, language=None)
            
            # The 'outputs' from model.tts is the spectrogram
            spectrogram = outputs
                
            if not isinstance(spectrogram, torch.Tensor):
                spectrogram = torch.tensor(spectrogram, device=self.device)
            
            # Ensure tensor is on the correct device for the vocoder
            spectrogram = spectrogram.to(self.device)
            
            # Add batch dimension if missing
            if spectrogram.dim() == 2:
                spectrogram = spectrogram.unsqueeze(0)

            # 2. Synthesize Audio (Vocoder - WaveGlow)
            # Use the generic vocoder's `inference` method
            with torch.no_grad():
                wav = self.vocoder.inference(spectrogram)
            
            # 3. Format and Save
            wav_np = wav.view(-1).cpu().numpy()
            
            # Ensure directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Save the .wav file
            write_wav(output_path, self.VOCODER_SAMPLE_RATE, wav_np)
            
            # print(f"✓ Audio saved: {output_path}")
            return output_path

        except Exception as e:
            print(f"✗ Error during synthesis for text '{text[:50]}...': {e}")
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
        print(f"Batch synthesizing {len(segments)} segments for speaker...")
        print(f"  Speaker Model: {speaker_model_dir}")
        print(f"  Output Directory: {output_dir}")
        print(f"{'='*66}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i, segment in enumerate(segments):
            text = segment['translated_text']
            # Use segment ID for a stable filename
            segment_id = segment.get('id', i) 
            output_path = os.path.join(output_dir, f"segment_{segment_id:04d}.wav")
            
            print(f"  [{i+1}/{len(segments)}] Synthesizing segment {segment_id}...", end='\r')
            
            if not text.strip():
                print(f"  [SKIP] Segment {segment_id} is empty.")
                segment['audio_path'] = None
                continue
                
            try:
                self.synthesize(
                    text=text,
                    speaker_model_dir=speaker_model_dir,
                    output_path=output_path
                )
                segment['audio_path'] = os.path.abspath(output_path)
            except Exception as e:
                print(f"  ✗ FAILED to synthesize segment {segment_id}: {e}")
                segment['audio_path'] = None
        
        print(f"\n✓ Batch synthesis complete. {len(segments)} segments processed.")
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
            "translated_audio_path": seg.get('audio_path') # The new audio path
        })
        
    output = {
        "metadata": {
            "num_segments": len(manifest),
            "sample_rate": TTSComponent.VOCODER_SAMPLE_RATE,
            "target_lang": manifest[0]['target_lang'] if manifest else None
        },
        "segments": manifest
    }
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"✓ TTS Manifest exported: {output_path}")
    except Exception as e:
        print(f"✗ Error exporting TTS manifest: {e}")