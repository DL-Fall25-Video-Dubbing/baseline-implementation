# ============================================================================
# LIPSYNC COMPONENT - BASELINE IMPLEMENTATION
#
#
# Description:
# Implements the Lipsync Generator and Discriminator models from the
# "Large-scale multilingual audio visual dubbing" paper (arXiv:2011.03530v1).
#
# This file defines the nn.Module architectures in PyTorch, based on
# Section 3.1.3 and Appendix A (Tables 5, 6, 7) of the paper.
#
# Dependencies:
# pip install torch
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# ============================================================================
# === 1. BUILDING BLOCKS (ResBlocks) ===
# ============================================================================

class DownResBlock(nn.Module):
    """
    Residual Block for downsampling, used in the encoders.
    Based on "2 Norm-Conv-ReLU layers per block"
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        
        # Shortcut path
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True)
        )
        
        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut(x)
        
        out = self.relu1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        
        out = out + shortcut
        return self.relu_out(out)

class UpResBlock(nn.Module):
    """
    Residual Block for upsampling, used in the decoder.
    Uses "Upsample 2x" (TransposeConv) + 2 Norm-Conv-ReLU layers
    """
    def __init__(self, in_channels, out_channels, skip_channels, kernel_size=3, padding=1):
        super().__init__()
        
        # Upsampling layer
        # Output channels = in_channels, so it can be added to skip connection
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0)
        
        # Total channels after concat(upsampled, skip)
        conv_in_channels = in_channels + skip_channels
        
        # Main path
        self.conv1 = nn.Conv2d(conv_in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        
        # Shortcut path
        self.shortcut = nn.Sequential(
            nn.Conv2d(conv_in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True)
        )
        
        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x, skip_connection):
        x = self.upsample(x)
        x = torch.cat([x, skip_connection], dim=1) # Concat along channel dim
        
        shortcut = self.shortcut(x)
        
        out = self.relu1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        
        out = out + shortcut
        return self.relu_out(out)


class AudioResBlock1D(nn.Module):
    """
    1D Residual Block for the Audio Encoder
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2):
        super().__init__()
        padding = (kernel_size - 1) // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm1 = nn.InstanceNorm1d(out_channels, affine=True)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.norm2 = nn.InstanceNorm1d(out_channels, affine=True)
        
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.InstanceNorm1d(out_channels, affine=True)
        )
        
        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.relu1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = out + shortcut
        return self.relu_out(out)

# ============================================================================
# === 2. ENCODER ARCHITECTURES ===
# ============================================================================

class InputFrameEncoder(nn.Module):
    """
    Encodes the masked input frames.
    Architecture from Table 5.
    Returns: List of skip connections and final embedding.
    """
    def __init__(self):
        super().__init__()
        # Initial convolution
        self.in_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(inplace=True)
        )
        
        # --- ResBlocks from Table 5 ---
        # Note: Table 5 starts from ResBlock 2, assuming 256x256 input
        # We add an initial block to go from 3 -> 16 channels @ 256x256
        # Let's adjust to match Table 5 exactly.
        
        # Table 5 seems to have a typo. It starts with ResBlock 1 (16->16, stride 1)
        # then ResBlock 2 (16->32, stride 2). Let's follow the filter list.
        # This implementation follows the *filter sizes* and *strides* in Table 5.
        
        self.block1 = DownResBlock(3, 16, stride=1, padding=1) # Stride 1 to keep 256x256
        self.block2 = DownResBlock(16, 32, stride=2)   # 256->128
        self.block3 = DownResBlock(32, 64, stride=2)   # 128->64
        self.block4 = DownResBlock(64, 128, stride=2)  # 64->32
        self.block5 = DownResBlock(128, 256, stride=2) # 32->16
        self.block6 = DownResBlock(256, 256, stride=2) # 16->8
        self.block7 = DownResBlock(256, 512, stride=2) # 8->4
        self.block8 = DownResBlock(512, 512, stride=2) # 4->2
        self.block9 = DownResBlock(512, 512, stride=2) # 2->1

    def forward(self, x):
        # x shape: [B, 3, 256, 256]
        skips = []
        s = self.block1(x); skips.append(s)    # s: [B, 16, 256, 256]
        s = self.block2(s); skips.append(s)    # s: [B, 32, 128, 128]
        s = self.block3(s); skips.append(s)    # s: [B, 64, 64, 64]
        s = self.block4(s); skips.append(s)    # s: [B, 128, 32, 32]
        s = self.block5(s); skips.append(s)    # s: [B, 256, 16, 16]
        s = self.block6(s); skips.append(s)    # s: [B, 256, 8, 8]
        s = self.block7(s); skips.append(s)    # s: [B, 512, 4, 4]
        s = self.block8(s); skips.append(s)    # s: [B, 512, 2, 2]
        s = self.block9(s)                     # s: [B, 512, 1, 1] (Embedding)
        
        # Return embedding and skips in reverse order (for decoder)
        return s, skips[::-1]


class AudioEncoder(nn.Module):
    """
    Encodes the audio mel-spectrogram.
    Architecture from Table 6.
    Returns: Final embedding and list of skip connections.
    """
    def __init__(self, in_channels=64, audio_len=24):
        super().__init__()
        # Following user's note: Input: 64 channels, length 24
        
        self.block1 = AudioResBlock1D(in_channels, 128, kernel_size=5, stride=2) # 24 -> 12
        self.block2 = AudioResBlock1D(128, 512, kernel_size=5, stride=2) # 12 -> 6
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # MLPs to create features for skip connections (as per paper text)
        # These must match the channel counts of the ImageDecoder's UpResBlocks
        # (512, 512, 256, 256, 128, 64, 32, 16, 3)
        # Create skips that match the *spatial dimensions* of the ImageEncoder skips.
        self.skip_mlps = nn.ModuleList([
            nn.Linear(512, 16 * 256 * 256), # Skip 1 (matches block1 out)
            nn.Linear(512, 32 * 128 * 128), # Skip 2
            nn.Linear(512, 64 * 64 * 64),   # ...
            nn.Linear(512, 128 * 32 * 32),
            nn.Linear(512, 256 * 16 * 16),
            nn.Linear(512, 256 * 8 * 8),
            nn.Linear(512, 512 * 4 * 4),
            nn.Linear(512, 512 * 2 * 2),
        ])
        
        # Simpler approach: return intermediate blocks
        self.mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512)
        )
        
    def forward(self, x):
        # x: [B, 64, 24]
        s1 = self.block1(x) # [B, 128, 12]
        s2 = self.block2(s1) # [B, 512, 6]
        
        embedding = self.global_pool(s2) # [B, 512, 1]
        embedding = embedding.squeeze(-1) # [B, 512]
        
        embedding_mlp = self.mlp(embedding) # [B, 512]
        
        # (This is an interpretation of "list of features with same sizes")
        skips = [
            embedding_mlp.view(-1, 512, 1, 1).expand(-1, -1, 2, 2),
            embedding_mlp.view(-1, 512, 1, 1).expand(-1, -1, 4, 4),
            embedding_mlp.view(-1, 512, 1, 1).expand(-1, -1, 8, 8),
            embedding_mlp.view(-1, 512, 1, 1).expand(-1, -1, 16, 16),
            embedding_mlp.view(-1, 512, 1, 1).expand(-1, -1, 32, 32),
            embedding_mlp.view(-1, 512, 1, 1).expand(-1, -1, 64, 64),
            embedding_mlp.view(-1, 512, 1, 1).expand(-1, -1, 128, 128),
            embedding_mlp.view(-1, 512, 1, 1).expand(-1, -1, 256, 256),
        ]

        return embedding_mlp, skips[::-1] # [B, 512] and list of skips


class ReferenceEncoder(nn.Module):
    """
    Encodes the N=10 reference frames.
    Architecture is identical to InputFrameEncoder (Table 5).
    """
    def __init__(self):
        super().__init__()
        # The encoder is the same as the InputFrameEncoder
        self.encoder = InputFrameEncoder()

    def forward(self, x):
        # x: [B, N, 3, 256, 256] where N=10
        B, N, C, H, W = x.shape
        
        # Reshape to run all N frames through the encoder in one batch
        x = x.view(B * N, C, H, W)
        
        # embedding: [B*N, 512, 1, 1], skips: List of [B*N, C, H, W]
        embedding, skips = self.encoder(x)
        
        # Reshape embedding back
        embedding = embedding.view(B, N, 512) # [B, N, 512]
        
        # We also need to process the skip connections
        # Reshape skips: List of [B, N, C, H, W]
        skips_reshaped = []
        for s in skips:
            _, C_s, H_s, W_s = s.shape
            skips_reshaped.append(s.view(B, N, C_s, H_s, W_s))
            
        return embedding, skips_reshaped

class SoftAttention(nn.Module):
    """
    Soft attention mechanism to combine Reference embeddings
    using the Audio embedding as the key.
    (Based on user's note: αn = exp(-σ(A·Rn)) / Σ exp(-σ(A·Ri)))
    """
    def __init__(self, dim=512):
        super().__init__()
        # We need a way to score A against Rn. Dot product is one way.
        # The paper's formula is unclear. Let's use standard dot-product attention.
        # Key = Audio (B, 1, 512), Value = Ref (B, N, 512)
        self.scale = dim ** -0.5

    def forward(self, audio_emb, ref_emb):
        # audio_emb: [B, 512] -> [B, 1, 512] (Query)
        # ref_emb:   [B, N, 512] (Key/Value)
        query = audio_emb.unsqueeze(1)
        
        # Attention scores
        # (B, 1, 512) @ (B, 512, N) -> (B, 1, N)
        scores = torch.bmm(query, ref_emb.transpose(1, 2)) * self.scale
        attn_weights = F.softmax(scores, dim=-1) # [B, 1, N]
        
        # Weighted sum of reference embeddings (values)
        # (B, 1, N) @ (B, N, 512) -> (B, 1, 512)
        context = torch.bmm(attn_weights, ref_emb)
        
        return context.squeeze(1) # [B, 512]

# ============================================================================
# === 3. DECODER AND TEMPORAL NET ===
# ============================================================================

class TemporalNet(nn.Module):
    """
    2-layer 1D CNN to aggregate temporal information
    from the combined 512-dim embedding.
    """
    def __init__(self, in_channels=512, out_channels=512, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [B, 512, T] (T=sequence length, e.g., 9)
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        return out # [B, 512, T]

class ImageDecoder(nn.Module):
    """
    Reconstructs the image from the bottleneck embedding and skip connections.
    Architecture from Table 7.
    """
    def __init__(self, audio_skip_dim=512, ref_skip_dim=512):
        super().__init__()
        
        # --- ResBlocks from Table 7 ---
        # Note: Skip channels = input_skip + ref_skip + audio_skip
        
        # Bottleneck: [B, 512, 1, 1]
        self.block1 = UpResBlock(512, 512, skip_channels=(512 + 512 + 512)) # 1->2
        self.block2 = UpResBlock(512, 512, skip_channels=(512 + 512 + 512)) # 2->4
        self.block3 = UpResBlock(512, 256, skip_channels=(256 + 512 + 512)) # 4->8
        self.block4 = UpResBlock(256, 256, skip_channels=(256 + 256 + 512)) # 8->16
        self.block5 = UpResBlock(256, 128, skip_channels=(128 + 256 + 512)) # 16->32
        self.block6 = UpResBlock(128, 64,  skip_channels=(64 + 128 + 512))  # 32->64
        self.block7 = UpResBlock(64, 32,   skip_channels=(32 + 64 + 512))   # 64->128
        self.block8 = UpResBlock(32, 16,   skip_channels=(16 + 32 + 512))   # 128->256
        
        # Final convolution to get 3 channels
        self.out_conv = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid() # As per user note
        )

    def forward(self, x, skips):
        # x: [B, 512, 1, 1] (Bottleneck embedding)
        # skips: List of 3 skip lists [input_skips, ref_skips, audio_skips]
        input_skips, ref_skips, audio_skips = skips
        
        # Apply attention to reference skips (avg over N)
        # This is an interpretation of how to combine N skip connections
        ref_skips_attn = [torch.mean(s, dim=1) for s in ref_skips]
        
        s = self.block1(x, torch.cat([input_skips[0], ref_skips_attn[0], audio_skips[0]], dim=1))
        s = self.block2(s, torch.cat([input_skips[1], ref_skips_attn[1], audio_skips[1]], dim=1))
        s = self.block3(s, torch.cat([input_skips[2], ref_skips_attn[2], audio_skips[2]], dim=1))
        s = self.block4(s, torch.cat([input_skips[3], ref_skips_attn[3], audio_skips[3]], dim=1))
        s = self.block5(s, torch.cat([input_skips[4], ref_skips_attn[4], audio_skips[4]], dim=1))
        s = self.block6(s, torch.cat([input_skips[5], ref_skips_attn[5], audio_skips[5]], dim=1))
        s = self.block7(s, torch.cat([input_skips[6], ref_skips_attn[6], audio_skips[6]], dim=1))
        s = self.block8(s, torch.cat([input_skips[7], ref_skips_attn[7], audio_skips[7]], dim=1))
        
        out = self.out_conv(s) # [B, 3, 256, 256]
        return out


class LandmarkDecoder(nn.Module):
    """
    Auxiliary head to predict 13 2D landmarks from the 512-dim embedding.
    """
    def __init__(self, in_channels=512, num_landmarks=13):
        super().__init__()
        self.layer = nn.Sequential(
            # "1 Residual Layer"
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, num_landmarks * 2) # 13 points * 2 coords (x,y)
        )
        self.num_landmarks = num_landmarks

    def forward(self, x):
        # x: [B, 512]
        out = self.layer(x)
        return out.view(-1, self.num_landmarks, 2) # [B, 13, 2]

# ============================================================================
# === 4. MAIN GENERATOR (U-Net) ===
# ============================================================================

class LipsyncGenerator(nn.Module):
    """
    The main Lipsync Generator U-Net model.
    Combines all the encoder and decoder components.
    """
    def __init__(self, audio_in_channels=64, audio_len=24, num_ref_frames=10):
        super().__init__()
        self.num_ref_frames = num_ref_frames
        
        # --- Encoders ---
        self.input_frame_encoder = InputFrameEncoder()
        self.audio_encoder = AudioEncoder(audio_in_channels, audio_len)
        # Reference encoder shares weights with input encoder
        self.reference_encoder = ReferenceEncoder()
        
        # --- Attention ---
        self.ref_attention = SoftAttention(dim=512)
        
        # --- Bottleneck Combination ---
        # Combines (Input + Audio + Ref) embeddings -> 512 dim
        self.embedding_combiner = nn.Sequential(
            nn.Linear(512 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
        )
        
        # --- Temporal Network ---
        self.temporal_net = TemporalNet(in_channels=512)
        
        # --- Decoders ---
        self.image_decoder = ImageDecoder()
        self.landmark_decoder = LandmarkDecoder(in_channels=512, num_landmarks=13)

    def forward(self, masked_frames, audio_mels, ref_frames):
        """
        Args:
            masked_frames (torch.Tensor): [B, T, 3, 256, 256]
            audio_mels (torch.Tensor):    [B, T, 64, 24]
            ref_frames (torch.Tensor):    [B, N, 3, 256, 256] (N=10)
        """
        B, T, C_img, H, W = masked_frames.shape
        _, _, C_aud, A_len = audio_mels.shape
        N = self.num_ref_frames
        
        # --- Process Time Sequence ---
        # We process the T-length sequence frame by frame
        # (A more optimized way is to use 3D convs, but let's
        # stick to the 2D+TemporalNet described)
        
        # Reshape to process all frames in one batch
        # [B*T, 3, 256, 256]
        masked_frames_flat = masked_frames.view(B * T, C_img, H, W)
        # [B*T, 64, 24]
        audio_mels_flat = audio_mels.view(B * T, C_aud, A_len)
        
        # --- Encoders ---
        # emb: [B*T, 512, 1, 1], skips: List of [B*T, C, H, W]
        input_emb_flat, input_skips_flat = self.input_frame_encoder(masked_frames_flat)
        input_emb_flat = input_emb_flat.squeeze() # [B*T, 512]
        
        # a_emb: [B*T, 512], a_skips: List of [B*T, C, H, W]
        audio_emb_flat, audio_skips_flat = self.audio_encoder(audio_mels_flat)
        
        # ref_emb: [B, N, 512], ref_skips: List of [B, N, C, H, W]
        ref_emb, ref_skips = self.reference_encoder(ref_frames)
        
        # --- Attention & Combination ---
        # We need to apply attention for each of the B*T frames
        # Use the audio embedding for each frame as the key
        
        # attended_ref_emb: [B*T, 512]
        attended_ref_emb_flat = self.ref_attention(
            audio_emb_flat,               # Query: [B*T, 512]
            ref_emb.unsqueeze(1).repeat(1, T, 1, 1).view(B*T, N, 512) # K/V: [B*T, N, 512]
        )
        
        # Combine embeddings [B*T, 512*3]
        combined_emb_flat = torch.cat(
            [input_emb_flat, audio_emb_flat, attended_ref_emb_flat], 
            dim=1
        )
        
        # [B*T, 512]
        bottleneck_emb_flat = self.embedding_combiner(combined_emb_flat)
        
        # --- Temporal Network ---
        # Reshape for 1D Conv: [B, C, T]
        bottleneck_emb_temporal = bottleneck_emb_flat.view(B, T, 512).transpose(1, 2)
        # [B, 512, T]
        temporal_out = self.temporal_net(bottleneck_emb_temporal)
        
        # Reshape back to flat batch for decoders
        # [B*T, 512]
        temporal_emb_flat = temporal_out.transpose(1, 2).reshape(B * T, 512)
        
        # --- Decoders ---
        
        # 1. Landmark Decoder
        # [B*T, 13, 2]
        pred_landmarks_flat = self.landmark_decoder(temporal_emb_flat)
        
        # 2. Image Decoder
        # We need to process ref_skips for the flat batch
        # [B, N, C, H, W] -> [B*T, N, C, H, W]
        ref_skips_flat = [
            s.unsqueeze(1).repeat(1, T, 1, 1, 1, 1).view(B*T, N, *s.shape[2:])
            for s in ref_skips
        ]
        
        all_skips = [input_skips_flat, ref_skips_flat, audio_skips_flat]
        
        # [B*T, 512] -> [B*T, 512, 1, 1]
        decoder_input = temporal_emb_flat.view(B * T, 512, 1, 1)
        
        # [B*T, 3, 256, 256]
        pred_images_flat = self.image_decoder(decoder_input, all_skips)
        
        # --- Reshape outputs ---
        pred_images = pred_images_flat.view(B, T, C_img, H, W)
        pred_landmarks = pred_landmarks_flat.view(B, T, 13, 2)
        
        return pred_images, pred_landmarks


# ============================================================================
# === 5. DISCRIMINATOR ARCHITECTURES ===
# ============================================================================

class HighResSpatioTemporalDiscriminator(nn.Module):
    """
    Discriminator for 3 full-res sequential frames.
    Checks for image quality and short-term motion.
    (Architecture is not specified, so this is a standard PatchGAN-style 3D-Conv)
    """
    def __init__(self, in_channels=3):
        super().__init__()
        
        def conv3d_block(in_c, out_c, stride=(1, 2, 2)):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=(3, 4, 4), stride=stride, padding=(1, 1, 1), bias=False),
                nn.InstanceNorm3d(out_c, affine=True),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.layers = nn.Sequential(
            # Input: [B, 3, 3, 256, 256] (C, T, H, W)
            conv3d_block(in_channels, 32),              # [B, 32, 3, 128, 128]
            conv3d_block(32, 64),                       # [B, 64, 3, 64, 64]
            conv3d_block(64, 128),                      # [B, 128, 3, 32, 32]
            conv3d_block(128, 256),                     # [B, 256, 3, 16, 16]
            nn.Conv3d(256, 1, kernel_size=(3, 4, 4), stride=1, padding=(1, 0, 0)) # [B, 1, 3, 13, 13]
        )

    def forward(self, x):
        # x: [B, T, 3, H, W] (T=3)
        x = x.transpose(1, 2) # [B, 3, T, H, W]
        return self.layers(x) # Output is a patch-grid of scores


class LowResAudioVisualDiscriminator(nn.Module):
    """
    Discriminator for downsampled full sequence + audio.
    Checks for audio-visual synchronization.
    (Architecture is not specified, this is a common approach)
    """
    def __init__(self, audio_channels=64, audio_len=24):
        super().__init__()
        
        # --- Video Tower (3D Conv) ---
        self.video_tower = nn.Sequential(
            # Input: [B, 3, 9, 64, 64] (C, T, H, W)
            nn.Conv3d(3, 32, kernel_size=3, stride=(1, 2, 2), padding=1), # [B, 32, 9, 32, 32]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=(1, 2, 2), padding=1), # [B, 64, 9, 16, 16]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=(1, 2, 2), padding=1), # [B, 128, 9, 8, 8]
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool3d((None, 1, 1)) # Pool H, W -> [B, 128, 9, 1, 1]
        )
        
        # --- Audio Tower (1D Conv) ---
        self.audio_tower = nn.Sequential(
            # Input: [B, 9, 64, 24] -> [B*9, 64, 24]
            # We need to process audio per frame and align with video
            # Let's use the audio encoder we already built
            AudioEncoder(audio_channels, audio_len),
            # Output is [B*T, 512]
        )
        self.audio_mlp = nn.Linear(512, 128)
        
        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128, 64), # video_feat + audio_feat
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, video_frames, audio_mels):
        # video_frames: [B, T, 3, H, W] (T=9, H=256)
        # audio_mels:   [B, T, 64, 24] (T=9)
        B, T, C_aud, A_len = audio_mels.shape
        
        # Downsample video frames
        video_lowres = F.interpolate(video_frames.view(B*T, 3, 256, 256), 
                                     size=(64, 64), 
                                     mode='bilinear', 
                                     align_corners=False)
        video_lowres = video_lowres.view(B, T, 3, 64, 64).transpose(1, 2) # [B, 3, T, 64, 64]
        
        # Process video
        video_feat = self.video_tower(video_lowres).squeeze() # [B, 128, 9]
        video_feat = video_feat.transpose(1, 2) # [B, 9, 128]
        
        # Process audio
        audio_mels_flat = audio_mels.view(B * T, C_aud, A_len)
        audio_emb_flat, _ = self.audio_encoder(audio_mels_flat) # [B*T, 512]
        audio_feat = self.audio_mlp(audio_emb_flat).view(B, T, 128) # [B, 9, 128]
        
        # Concatenate and classify
        combined_feat = torch.cat([video_feat, audio_feat], dim=2) # [B, 9, 256]
        
        # Get score per timestep
        scores = self.classifier(combined_feat) # [B, 9, 1]
        
        return scores # Return scores for each timestep