Video → Multi-speaker Lipsync → Single-speaker Lipsync → Translated Video
Audio → Multi-speaker TTS → Single-speaker TTS → Translated Audio
                ↑                      ↑
        Large-scale training    Speaker-specific fine-tuning
```

---

## 1. LIPSYNC MODEL - Multi-speaker Multilingual

### **Training Status**: ❌ **TRAINED FROM SCRATCH**

**Explicit confirmation** (Page 24, Appendix A):
> *"All network parameters are initialized from scratch with random weights"*

### Model Architecture

#### **Base Architecture**: Residual U-Net with 3 Encoders + 1 Decoder

---

### **A. Input Frame Encoder**

**Architecture** (Table 5, Page 24):
```
Input: 256×256×3 masked frames
├─ Residual Block 1: 16 filters, 3×3 kernel, stride 2
├─ Residual Block 2: 32 filters, 3×3 kernel, stride 2
├─ Residual Block 3: 64 filters, 3×3 kernel, stride 2
├─ Residual Block 4: 128 filters, 3×3 kernel, stride 2
├─ Residual Block 5: 256 filters, 3×3 kernel, stride 2
├─ Residual Block 6: 256 filters, 3×3 kernel, stride 2
├─ Residual Block 7: 512 filters, 3×3 kernel, stride 2
├─ Residual Block 8: 512 filters, 3×3 kernel, stride 2
└─ Residual Block 9: 512 filters, 3×3 kernel, stride 2
Output: 512×1×1 embedding
```

**Each Residual Block**: 2 Norm-Conv-ReLU layers per block

---

### **B. Audio Encoder**

**Architecture** (Table 6, Page 24):
```
Input: Mel-spectrogram (80 filter banks)
├─ Residual Conv1D Block 1: 128 filters, kernel 5, stride 2
│  └─ Input: 64 channels, length 24
├─ Residual Conv1D Block 2: 512 filters, kernel 5, stride 2
│  └─ Input: 128 channels, length 12
├─ Global Average Pooling
└─ Output: 512×1 embedding
```

**Processing**:
- 2 temporal residual blocks with 512 filters
- Multi-layer perceptron with residual structure
- Produces features for skip connections to decoder

---

### **C. Reference Frame Encoder**

**Architecture**: Same as Input Frame Encoder (Table 5)
```
Input: 10 reference frames (256×256×5 - inverse masked)
├─ Same 9 Residual Blocks as Input Encoder
├─ Produces N×512 embeddings (N=10 reference frames)
└─ Soft Attention Mechanism:
    αn = exp(-σ(A·Rn)) / Σ exp(-σ(A·Ri))
    R = Σ αn·Rn
Output: 512 dimensional attended embedding
```

**Reference Frame Selection**: K-means clustering (K=10) on facial landmarks

---

### **D. Temporal Network**

**Architecture**:
```
Input: Concatenated embeddings (512 from each encoder = 1536 total)
├─ 1D Conv Layer 1: 512 channels, kernel 3
├─ 1D Conv Layer 2: 512 channels, kernel 3
└─ Output: 512×9 (for 9 frame sequence)
```

---

### **E. Image Decoder**

**Architecture** (Table 7, Page 24):
```
Input: 512×1×1 embedding
├─ Residual Block 1: 512 filters, 3×3, upsample 2× → 2×2
├─ Residual Block 2: 512 filters, 3×3, upsample 2× → 4×4
├─ Residual Block 3: 256 filters, 3×3, upsample 2× → 8×8
├─ Residual Block 4: 256 filters, 3×3, upsample 2× → 16×16
├─ Residual Block 5: 128 filters, 3×3, upsample 2× → 32×32
├─ Residual Block 6: 64 filters, 3×3, upsample 2× → 64×64
├─ Residual Block 7: 32 filters, 3×3, upsample 2× → 128×128
├─ Residual Block 8: 16 filters, 3×3, upsample 2× → 256×256
├─ Residual Block 9: 3 filters, 3×3, upsample 2× → 256×256
└─ Sigmoid activation
Output: 256×256×3 RGB image
```

**Skip Connections**: U-Net style connections from each encoder layer to corresponding decoder layer

---

### **F. Landmark Decoder** (Auxiliary)

**Architecture**:
```
Input: 512 embedding
├─ 1 Residual Layer
└─ Output: 13×2 (26 values) - jaw and mouth landmark coordinates
```

---

### **G. Dual Discriminators**

#### **High-Resolution Spatiotemporal Discriminator**

**Architecture**:
```
Input: 3 sequential frames at full resolution (256×256×3×3)
├─ 3D Convolutional layers
├─ Tests: Image quality + short-term motion coherence
└─ Output: Real/Fake classification
```

#### **Low-Resolution Spatiotemporal Discriminator**

**Architecture**:
```
Input: Full sequence (9 frames) downsampled 1/4 + audio
├─ Processes: 64×64 resolution video
├─ Tests: Audio-visual synchronization + long-term motion
└─ Output: Real/Fake classification
```

**Loss**: Hinge loss (following Lim & Ye, Brock et al.)

---

### Training Configuration

**Dataset**:
- **Size**: 3,700 hours
- **Languages**: 20 languages (75% English)
- **Utterances**: ~3 million
- **Speakers**: ~464K (estimated)

**Training Details** (Page 13, 25):
```
Hardware: 32 GPUs
Batch size: 64 (2 per GPU)
Optimizer: Adam
Learning rates:
  - Generator: 5×10⁻⁴
  - Discriminators: 1×10⁻⁴
Gradient clipping: Global norm = 10
Iterations: 200,000
Duration: ~several days
```

**Loss Weights**:
- αRec = 1.0 (reconstruction)
- αLand = 100 (landmark)
- αGAN = 1×10⁻⁴ (adversarial)

**Loss Functions**:
```
1. MS-SSIM + L1: LRec = 0.86·LMS-SSIM + 0.14·LL1
2. Landmark L2: LLand = Σ ||(x̂l - xl) + (ŷl - yl)||²
3. Hinge GAN: LD = E[max(0, 1-D(x))] + E[max(0, 1+D(G(z)))]
4. Total: L = αRec·LRec + αLand·LLand + αGAN·LGAN
```

---

## 2. LIPSYNC MODEL - Single-speaker Fine-tuning

### **Training Status**: ❌ **FINE-TUNED FROM MULTI-SPEAKER MODEL**

**Architecture**: Same as multi-speaker model (no changes)

**Fine-tuning Details** (Page 15):
```
Initialization: Multi-speaker model weights
Data: Target speaker video(s) only
Iterations: 10,000
Batch size: 64
Learning rate: Same as multi-speaker
GPU: 32 GPUs
Duration: Several minutes of speaker data needed
```

**Options explored** (but full fine-tuning works best):
- ❌ Fix encoder, train decoder only
- ❌ Train decoder + temporal network only
- ✅ **Fine-tune all parameters** (best results)

---

## 3. TEXT-TO-SPEECH (TTS) - Multi-speaker Multilingual

### **Training Status**: ❌ **TRAINED FROM SCRATCH**

**Architecture**: Similar to Chen et al. [7] (Tacotron 2-based)

**Quote** (Page 2):
> *"Our system makes use of a speaker-adaptive text to speech model similar to Chen et al. [7]"*

**Model Type**: Speaker-adaptive multi-speaker multilingual TTS

**Training Details**:
- **Dataset**: "Thousands of hours" of multilingual speech
- **Languages**: Same 20 languages as lipsync
- **Approach**: Large-scale multi-speaker training with speaker embeddings

**Details not explicitly provided**, but based on Chen et al.:
```
Architecture: Tacotron 2 variant
├─ Text encoder
├─ Speaker embedding layer
├─ Attention mechanism
├─ Decoder (mel-spectrogram prediction)
└─ Vocoder (WaveNet or similar)
```

---

## 4. TEXT-TO-SPEECH - Single-speaker Fine-tuning

### **Training Status**: ❌ **FINE-TUNED FROM MULTI-SPEAKER TTS**

**Architecture**: Same as multi-speaker TTS

**Fine-tuning** (Page 15):
> *"adapted on a per-speaker basis using a small amount of additional reference speech (which can again be drawn from the target video, or from additional recordings of the same speaker)"*

**Data Requirements**: Several minutes of target speaker audio

---

## 5. ASR (Automatic Speech Recognition)

### **Training Status**: ✅ **PRE-TRAINED MODEL**

**Model**: One ASR model per language (Page 5)

**Quote**:
> *"We employ a speech recognition model to transcribe the video"*

**Details**: Not specified, but likely uses existing ASR systems per language

---

## 6. MACHINE TRANSLATION

### **Training Status**: Not explicitly discussed (assumed existing system)

**Quote** (Page 15):
> *"translate the transcripts into the target language using a machine translation model"*

**Note**: Human editors used to improve quality

---

## 7. DATA PROCESSING COMPONENTS

### **A. Face Detection & Tracking** ✅ **PRE-TRAINED TOOLS**

**Tools used** (Page 5):
- Face tracker
- Facial landmarker

**Processing Pipeline** (Figure 3, Page 5):
```
Raw Videos
├─ Shot Boundary Detection
├─ Face Detector/Tracker
├─ Face Landmark Smoothing (Gaussian kernel)
├─ View Canonicalization (Procrustes transform)
│   └─ Removes skew, keeps rotation + scaling
│   └─ Eye-nose alignment only (not lip/chin)
├─ Clip Quality Filter
│   └─ Blur detection model
│   └─ Image Laplacian variance
│   └─ Eye distance ≥80 pixels
│   └─ Frame rate 23-30 fps
└─ Speaking Filter (high precision)
    └─ Active speaker detection (Roth et al.)
```

---

### **B. Active Speaker Detection** ✅ **PRE-TRAINED MODEL**

**Model**: Roth et al. [19] - AVA Active Speaker

**Purpose**: Filter out non-speaking faces and voice-overs

---

## 8. RENDERING & POST-PROCESSING

### **Status**: ❌ **CUSTOM IMPLEMENTATION**

**Pipeline** (Pages 15-16):

#### **Face Cropping**:
```
1. Detect landmarks
2. Compute affine transform (Procrustes)
3. Extract 256×256 crop
4. Apply mask (lower face region)
```

#### **Blending**:
```
1. Generate polygonal mask from landmarks
2. Apply Gaussian blur to mask edges
3. Inverse affine transform to map back
4. Alpha blend: (1-blur(mask))·T(crop) + mask·original
```

**Mask Definition** (Page 12):
```
Polygonal mask vertices:
├─ Convex hull of: ears, nose (halfway), nose tip, chin points
├─ Chin landmarks shifted down for open mouth
└─ Filled polygon → binary mask → Gaussian blur