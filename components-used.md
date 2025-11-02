# I. Baseline Implementation.
1. ASR: using OpenAI's Whisper medium
2. TTS: Tacotron 2 + WaveNet-based vocoder [reference 25]. (Need to change HiFi-GAN, it's mistake). The paper does not mention using HiFi-GAN as the vocoder. The citation [25] that they use for their Tacotron 2 model is the original Google paper, which is titled "Natural tts synthesis by conditioning **wavenet** on mel spectrogram predictions". This strongly implies that their TTS system was a Tacotron 2 (for mel-spectrogram generation) paired with a **WaveNet-based vocoder** (for waveform generation), not HiFi-GAN.
3. MT: The paper's main focus was on the audiovisual (lipsync) component, so it does not describe the specific architecture of the MT model it used. The baseline paper handled the machine translation (MT) component as a distinct step in their pipeline, but they relied heavily on `human editors` to correct the automated output and solve key challenges. the paper was published in Nov 2020, to implement the baseline, the Facebook (Meta now) NLLB (No Language Left Behind, published in July 2022) is considered a state-of-the-art model for machine translation, especially for low-resource languages.

4. Lip-sync & background preservation: The system used custom approach of multi-speaker multilingual model (residual U-net) and speaker-specific finetuning. The system was explicitly designed to preserve the background and all other parts of the speaker's face and body. The core lip-sync model's job was not to generate a whole new video frame. Instead, it was trained to synthesize only the missing mouth region. The rest of the original frame (background, hair, eyes, nose, etc.) was kept intact and then combined with the newly generated mouth.


PROBLEMS:
1. NOTEBOOK
- Missing Step: Data Curation for 120-Hour Multilingual Set. This is the most significant gap. The plan is to pre-train on 120 hours of data from 5 specific languages from VoxCeleb2.The Problem: The VoxCeleb2 dataset is not labeled by language, only by speaker nationality. The current notebook pipeline does not have a step to select this specific 120-hour, 5-language subset.
Current Behavior: Cell 3 (data_preprocessing.py) processes one target video.Cell 9 (lipsync_component.py) defines a LipsyncDataset that scans all subdirectories in PROCESSED_DATA_DIR .Action Required: You must add a new data curation step before The current Cell 3. This new step would need to:Scan the entire extracted VoxCeleb2 dataset (from Cell 2.1).For each video file, use The ASRComponent.detect_language method  to identify its language.Copy the paths of files that match 'en', 'fr', 'es', 'de', or 'ru' to a new list.Keep adding files until you have 120 hours of audio.Then, run the DataPreprocessor (Cell 3) and LipsyncDataset (Cell 9) only on this new, filtered 120-hour list of files.




# II. Our Custom approach.

1. ASR: using openai's whisper-medium pre-trained model.
2. TTS: using zero-shot voice cloning (XTTS-v2 (Coqui))
3. MT: Using Meta Llama 3.1
4. LipSync and background preservation: Wav2Lip
the same 120-hr from voxceleb2 will be used using subset of 5 languages (english, russian, french, spanish and german).
Testing: the web-interface will be developed. the interface having multiple options: a user can upload a video in one of our five languages, choose the target language he wants to get the output video. He can also have an option of pasting the youtube link in one of five languages and be able to download the translated video. In all cases, the duration of video must nearly match.


=> python convert_notebook_to_pdf.py
