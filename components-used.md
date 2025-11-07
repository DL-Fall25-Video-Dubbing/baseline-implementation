# I. Baseline Implementation.
1. ASR: using OpenAI's Whisper medium
2. TTS: Tacotron 2 + WaveNet-based vocoder [reference 25]. (Need to change HiFi-GAN, it's mistake). The paper does not mention using HiFi-GAN as the vocoder. The citation [25] that they use for their Tacotron 2 model is the original Google paper, which is titled "Natural tts synthesis by conditioning **wavenet** on mel spectrogram predictions". This strongly implies that their TTS system was a Tacotron 2 (for mel-spectrogram generation) paired with a **WaveNet-based vocoder** (for waveform generation), not HiFi-GAN.
3. MT: The paper's main focus was on the audiovisual (lipsync) component, so it does not describe the specific architecture of the MT model it used. The baseline paper handled the machine translation (MT) component as a distinct step in their pipeline, but they relied heavily on `human editors` to correct the automated output and solve key challenges. the paper was published in Nov 2020, to implement the baseline, the Facebook (Meta now) NLLB (No Language Left Behind, published in July 2022) is considered a state-of-the-art model for machine translation, especially for low-resource languages.

4. Lip-sync & background preservation: The system used custom approach of multi-speaker multilingual model (residual U-net) and speaker-specific finetuning. The system was explicitly designed to preserve the background and all other parts of the speaker's face and body. The core lip-sync model's job was not to generate a whole new video frame. Instead, it was trained to synthesize only the missing mouth region. The rest of the original frame (background, hair, eyes, nose, etc.) was kept intact and then combined with the newly generated mouth.





# II. Our Custom approach.

1. ASR: using openai's whisper-medium pre-trained model.
2. TTS: using zero-shot voice cloning (XTTS-v2 (Coqui))
3. MT: Using Meta Llama 3.1
4. LipSync and background preservation: Wav2Lip
the same 120-hr from voxceleb2 will be used using subset of 5 languages (english, russian, french, spanish and german).
Testing: the web-interface will be developed. the interface having multiple options: a user can upload a video in one of our five languages, choose the target language he wants to get the output video. He can also have an option of pasting the youtube link in one of five languages and be able to download the translated video. In all cases, the duration of video must nearly match.


## ======= our approach ===========


This is an excellent, modern stack that cleverly swaps the baseline's "speaker-specific fine-tuning" for a more flexible "zero-shot" approach. This new pipeline is much faster at *inference time* because you don't need to re-train models for every new speaker.

Here is a step-by-step breakdown of how this custom approach would be implemented, from initial training to final deployment.

## Phase 1: Training the Wav2Lip Model (Offline Preparation)

Your new stack significantly reduces training time. The ASR (Whisper), MT (Llama 3.1), and TTS (XTTS-v2) models are all pre-trained and can be used directly.

The **only component you must train** is the Wav2Lip model. It needs to learn the general relationship between audio and mouth movements from your 120-hour dataset.

### Step 1: Data Curation (120-Hour, 5-Language Set)
This step is identical to the one we discussed for the baseline. You must first create your training dataset:
1.  **Scan:** Scan the raw, extracted VoxCeleb2 dataset.
2.  **Detect:** Use your `asr_component.py`'s `detect_language` function on a sample of each speaker's audio.
3.  **Filter:** Create a master list of all video files that belong to your 5 target languages (`en`, `fr`, `es`, `de`, `ru`).
4.  **Collect:** Stop once you have ~120 hours of video.
5.  **Pre-process:** Run your `data_preprocessing.py` on *all 120 hours* of video. This will generate the cropped faces and corresponding audio clips needed for training.

### Step 2: Wav2Lip Model Training
With your 120-hour dataset ready, you will train the Wav2Lip generator and its "expert" lip-sync discriminator.
* **Input:** The cropped faces and audio clips from Step 1.
* **Goal:** The model learns a speaker-agnostic mapping between audio features (from any of your 5 languages) and the corresponding visual mouth shapes.
* **Output:** A single, powerful, pre-trained model (e.g., `wav2lip_multilingual.pth`). This model is now ready for deployment and can lip-sync *any* speaker, even ones it has never seen.

---

## Phase 2: The Deployment & Inference Pipeline (The Web App)

This is the live pipeline your web-interface will use. It's a completely zero-shot pipeline that requires no fine-tuning per user.

Here is the flow of data from the user's click to the final download:

### Step 1: User Input (Upload or YouTube Link)
This is the entry point for your web application (which you'd build in React and FastAPI, as you're learning).

* **Can YouTube links be pasted freely?** **Yes.** Your FastAPI backend would handle this by using a library like `yt-dlp` (a fork of `youtube-dl`). When a user pastes a link, the backend simply downloads the video file to the server first.
* **Upload:** If the user uploads a file, it's saved to the server.
* In both cases, you now have a source file: `original_video.mp4`.

### Step 2: ASR (Whisper)
Your backend uses the `asr_component.py` to get a timed transcript.
* **Input:** `original_video.mp4`
* **Action:**
    1.  Extract the audio (`original_audio.wav`).
    2.  Run `asr.transcribe(original_audio.wav)`.
* **Output:** A JSON object with timed text segments: `transcript.json`.

### Step 3: MT (Llama 3.1)
Your backend uses the `mt_component.py` (now modified to call Llama 3.1) to translate the text.
* **Input:** The text segments from `transcript.json`.
* **Action:** Run `mt.translate_segments(...)` on the text.
* **Output:** A new JSON object with the *translated* text segments: `translated_transcript.json`.

### Step 4: TTS (Zero-Shot Voice Cloning)
This is the core of your new approach. Your backend uses the `tts_component.py` (now based on XTTS-v2).
* **Input 1 (Text):** The translated text from `translated_transcript.json`.
* **Input 2 (Voice):** The *original* audio file (`original_audio.wav`) from Step 2. This is used as the **voice reference clip**.
* **Action:** `tts.synthesize_segments(...)` clones the voice from `original_audio.wav` and uses it to speak the translated text.
* **Output:** The final, translated audio in the original speaker's voice: `translated_audio.wav`.

### Step 5: Duration Matching (Crucial Step)
This is how you solve the "nearly match" constraint.The baseline paper noted this was a major problem.
* **Problem:** `original_video.mp4` is 30.0 seconds, but `translated_audio.wav` might be 32.5 seconds.
* **Action:** Your backend must use an audio time-stretching library (like `pyrubberband` or a SoX command). It will "stretch" or "shrink" `translated_audio.wav` to match the *exact* duration of `original_video.mp4`.
* **Output:** A new file, `translated_audio_synced.wav`, that has the same duration as the original video.

### Step 6: Lip-Sync (Wav2Lip)
Your backend now has all the pieces to create the final video.
* **Input 1 (Video):** The `original_video.mp4` (provides the face and background).
* **Input 2 (Audio):** The time-stretched `translated_audio_synced.wav` (provides the new audio).
* **Input 3 (Model):** Your pre-trained `wav2lip_multilingual.pth` (from Phase 1).
* **Action:** Wav2Lip processes the video frame-by-frame. It takes the face from the original frame, generates a new mouth based on the *new* audio, and **blends that mouth back onto the original frame**, keeping the background and upper face 100% intact.
* **Output:** The final dubbed video: `final_dubbed_video.mp4`.

### Step 7: Deployment (Web Interface)
The frontend and backend tie this all together.
* **Frontend (React):** A simple UI with an "Upload" button, a "YouTube Link" text field, a language selection dropdown, and a "Translate" button. When the user clicks "Translate," it shows a "Processing..." spinner.
* **Backend (FastAPI):** An API endpoint (e.g., `/dub-video/`) that receives the request and runs Steps 1-6. This will be a long-running task, so it should run asynchronously.
* **Result:** When the pipeline is finished, the backend sends a success message, and the React frontend displays a "Download Your Video" link pointing to `final_dubbed_video.mp4`.

=> python convert_notebook_to_pdf.py
=> data: https://github.com/facebookresearch/muavic
