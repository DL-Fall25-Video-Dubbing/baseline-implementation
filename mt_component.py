# ============================================================================
# MT COMPONENT - BASELINE IMPLEMENTATION
#
# Description:
# Implements the MT (Machine Translation) component for the
# "Large-scale multilingual audio visual dubbing" baseline project.
#
# This component uses Facebook's NLLB (No Language Left Behind) model,
# a powerful many-to-many multilingual translation model.
#
# Dependencies:
# pip install transformers torch numpy sacrebleu
# ============================================================================

import os
import json
from typing import Dict, List, Optional, Union

import torch
import numpy as np

# Suppress Hugging Face warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError:
    print("Error: 'transformers' library not found.")
    print("Please install it: pip install transformers")
    exit(1)

try:
    from sacrebleu.metrics import BLEU
except ImportError:
    print("Warning: 'sacrebleu' not found. BLEU score calculation will not work.")
    print("         Install with: pip install sacrebleu")
    BLEU = None # Define as None if not found

# ============================================================================
# MT COMPONENT
# ============================================================================

class MTComponent:
    """
    Machine Translation using NLLB (No Language Left Behind)
    Supports: English, French, Spanish, German, Russian (bidirectional)

    Based on DeepMind baseline approach: Use pretrained MT model
    """

    # NLLB language codes (different from Whisper codes!)
    LANGUAGE_CODES = {
        'en': 'eng_Latn',  # English
        'fr': 'fra_Latn',  # French
        'es': 'spa_Latn',  # Spanish
        'de': 'deu_Latn',  # German
        'ru': 'rus_Cyrl',  # Russian (Cyrillic)
    }

    # Reverse mapping
    CODE_TO_LANG = {v: k for k, v in LANGUAGE_CODES.items()}

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: str = "auto",
        max_length: int = 512
    ):
        """
        Initialize MT component with NLLB model

        Args:
            model_name: NLLB model name
                        - 'facebook/nllb-200-distilled-600M' (600M params, faster)
                        - 'facebook/nllb-200-1.3B' (1.3B params, better quality)
                        - 'facebook/nllb-200-3.3B' (3.3B params, best quality)
            device: Device to use ('cuda', 'cpu', 'auto')
            max_length: Maximum sequence length for translation
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

        self.model_name = model_name
        self.max_length = max_length

        print(f"Loading NLLB model: {model_name}")
        print(f"Device: {self.device}")
        print(f"This may take a few minutes on first run...")

        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
        except Exception as e:
            print(f"Error loading model {model_name}. Check model name and internet connection.")
            raise e

        print(f"✓ MT Component initialized")
        print(f"  Model: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  Supported languages: {list(self.LANGUAGE_CODES.keys())}")

    def translate(
        self,
        text: Union[str, List[str]],
        source_lang: str,
        target_lang: str,
        max_length: Optional[int] = None,
        num_beams: int = 5,
        temperature: float = 1.0
    ) -> Union[str, List[str]]:
        """
        Translate text from source to target language

        Args:
            text: Text string or list of strings to translate
            source_lang: Source language code ('en', 'fr', 'es', 'de', 'ru')
            target_lang: Target language code
            max_length: Maximum length of translation (None = auto)
            num_beams: Number of beams for beam search
            temperature: Sampling temperature (1.0 = greedy)

        Returns:
            Translated text (string or list of strings)
        """
        # Validate languages
        if source_lang not in self.LANGUAGE_CODES:
            raise ValueError(
                f"Source language '{source_lang}' not supported. "
                f"Choose from: {list(self.LANGUAGE_CODES.keys())}"
            )
        if target_lang not in self.LANGUAGE_CODES:
            raise ValueError(
                f"Target language '{target_lang}' not supported. "
                f"Choose from: {list(self.LANGUAGE_CODES.keys())}"
            )

        # Handle single string vs list
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        if not texts or (len(texts) == 1 and not texts[0]):
            return "" if is_single else []

        # Get NLLB language codes
        src_code = self.LANGUAGE_CODES[source_lang]
        tgt_code = self.LANGUAGE_CODES[target_lang]

        # Set source language for tokenizer
        self.tokenizer.src_lang = src_code

        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        # Generate translation
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_code],
                max_length=max_length or self.max_length,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=False  # Greedy decoding for baseline
            )

        # Decode
        translations = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )

        # Return single string or list
        return translations[0] if is_single else translations

    def translate_segments(
        self,
        segments: List[Dict],
        source_lang: str,
        target_lang: str,
        preserve_timing: bool = True,
        adjust_length: bool = True,
        verbose: bool = True,
        batch_size: int = 16
    ) -> List[Dict]:
        """
        Translate segments with timestamps (from ASR output) efficiently in batches.

        Args:
            segments: List of segments from ASR
                      Each segment: {'id', 'text', 'start', 'end'}
            source_lang: Source language code
            target_lang: Target language code
            preserve_timing: Keep original timestamps
            adjust_length: Warn if translation much longer/shorter
            verbose: Print progress
            batch_size: How many segments to translate in one GPU pass

        Returns:
            List of translated segments with metadata
        """
        if not segments:
            return []

        if verbose:
            print(f"\n{'='*60}")
            print(f"Translating {len(segments)} segments")
            print(f"Direction: {source_lang.upper()} → {target_lang.upper()} (Batch size: {batch_size})")
            print(f"{'='*60}")

        translated_segments = []
        total_length_ratio = []
        original_texts = [seg['text'].strip() for seg in segments]

        # Process in batches for efficiency
        for i in range(0, len(original_texts), batch_size):
            if verbose:
                print(f"  Processing segments {i+1}-{min(i+batch_size, len(segments))}/{len(segments)}...", end='\r')
            
            batch_texts = original_texts[i:i+batch_size]
            batch_segments = segments[i:i+batch_size]
            
            # Translate the batch
            translated_texts = self.translate(
                text=batch_texts,
                source_lang=source_lang,
                target_lang=target_lang
            )

            # Process batch results
            for j, translated_text in enumerate(translated_texts):
                segment = batch_segments[j]
                original_text = batch_texts[j]
                
                # Calculate length ratio (for dubbing alignment)
                length_ratio = len(translated_text) / max(len(original_text), 1)
                total_length_ratio.append(length_ratio)

                # Create translated segment
                translated_seg = {
                    'id': segment['id'],
                    'start': segment['start'],
                    'end': segment['end'],
                    'duration': segment.get('duration', segment['end'] - segment['start']),
                    'original_text': original_text,
                    'translated_text': translated_text,
                    'source_lang': source_lang,
                    'target_lang': target_lang,
                    'length_ratio': round(length_ratio, 2)
                }

                # Warning for large length differences (baseline paper mentions this issue)
                if adjust_length and (length_ratio < 0.5 or length_ratio > 1.5):
                    translated_seg['length_warning'] = True
                    if verbose:
                        print(f"\n  ⚠ Warning: Segment {segment['id']} length ratio {length_ratio:.2f}")

                translated_segments.append(translated_seg)

        if verbose:
            avg_ratio = np.mean(total_length_ratio) if total_length_ratio else 0
            print(f"\n\n✓ Translation complete")
            print(f"  Segments: {len(translated_segments)}")
            print(f"  Avg length ratio: {avg_ratio:.2f}")
            print(f"  (1.0 = same length, <1.0 = shorter, >1.0 = longer)")

            # Count warnings
            warnings = sum(1 for s in translated_segments if s.get('length_warning', False))
            if warnings > 0:
                print(f"  ⚠ {warnings} segments with length mismatch (>50%)")
                print(f"    (May need manual editing for dubbing)")

        return translated_segments

    def get_model_info(self) -> Dict:
        """Get information about loaded model"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'parameters': sum(p.numel() for p in self.model.parameters()) / 1e6,
            'supported_languages': list(self.LANGUAGE_CODES.keys()),
            'language_codes': self.LANGUAGE_CODES,
            'max_length': self.max_length
        }


