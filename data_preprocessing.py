# ============================================================================
# DATA PREPROCESSING COMPONENT - BASELINE IMPLEMENTATION
#
# Description:
# Implements the video data preprocessing pipeline based on the
# "Large-scale multilingual audio visual dubbing" paper (arXiv:2011.03530v1).
#
# Tasks:
# 1. Read video frames.
# 2. Detect faces and extract facial landmarks using MediaPipe.
# 3. Smooth landmarks over time using a Gaussian filter.
# 4. Perform view canonicalization (alignment and cropping) based on
#    eye and nose landmarks only, using Procrustes analysis.
# 5. Generate masked input frames and inverse-masked reference frames.
# 6. Store processed data and transformation matrices for rendering.
#
# Dependencies:
# pip install opencv-python mediapipe numpy scipy
# ============================================================================

import cv2
import mediapipe as mp
import numpy as np
import os
import json
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from typing import List, Tuple, Dict, Optional

# ============================================================================
# === MediaPipe Initialization ===
# ============================================================================

mp_face_mesh = mp.solutions.face_mesh

# Example key points for alignment (more robust):
# Left eye corners, Right eye corners, Nose tip, Midpoint between eyes
LEFT_EYE_INNER_CORNER = 133
LEFT_EYE_OUTER_CORNER = 33
RIGHT_EYE_INNER_CORNER = 362
RIGHT_EYE_OUTER_CORNER = 263
NOSE_TIP = 1
# Midpoint between eyes calculation needed, using bridge points like 6, 168
NOSE_BRIDGE_MID = 6 # Approx. top of nose bridge

# We'll use inner/outer corners, nose tip, and nose bridge mid
ALIGNMENT_INDICES = [
    LEFT_EYE_INNER_CORNER, LEFT_EYE_OUTER_CORNER,
    RIGHT_EYE_INNER_CORNER, RIGHT_EYE_OUTER_CORNER,
    NOSE_TIP, NOSE_BRIDGE_MID
]

# Indices for the mouth region (useful for validation/debugging)
MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, # Outer lips
                 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308] # Inner lips

# ============================================================================
# === Data Preprocessor Class ===
# ============================================================================

class DataPreprocessor:
    """
    Handles video preprocessing: face detection, alignment, cropping, masking.
    """
    def __init__(self, output_size: int = 256, smoothing_sigma: float = 2.0):
        """
        Initialize the preprocessor.

        Args:
            output_size: The size of the cropped face images (e.g., 256x256).
            smoothing_sigma: Sigma for Gaussian smoothing of landmarks over time.
        """
        self.output_size = output_size
        self.smoothing_sigma = smoothing_sigma
        
        # Initialize MediaPipe Face Mesh
        # Using static_image_mode=False enables tracking between frames
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1, # Assume single speaker per frame for baseline
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define the canonical face template based on mean landmark positions
        # These are *pixel* coordinates for a centered face in the output_size image
        self.canonical_template = self._create_canonical_template(output_size)

        # [cite_start]Define the mouth mask region in *normalized crop coordinates* [cite: 580]
        # (x1, y1, x2, y2) -> (left, top, right, bottom)
        self.norm_mask_rect = (0.08, 0.28, 0.92, 0.95)

    def _create_canonical_template(self, size: int) -> np.ndarray:
        """
        Creates the target alignment template based on key landmarks.
        Coordinates are in pixel space for the output image size.
        This represents the 'average' or desired pose in the output crop.
        Adjust these values if needed for better centering/scaling.
        """
        # Template points must correspond *exactly* to ALIGNMENT_INDICES
        template = np.array([
            # LEFT_EYE_INNER_CORNER, LEFT_EYE_OUTER_CORNER
            [0.35 * size, 0.4 * size], [0.20 * size, 0.4 * size],
            # RIGHT_EYE_INNER_CORNER, RIGHT_EYE_OUTER_CORNER
            [0.65 * size, 0.4 * size], [0.80 * size, 0.4 * size],
            # NOSE_TIP
            [0.50 * size, 0.6 * size],
            # NOSE_BRIDGE_MID (Top of nose bridge)
            [0.50 * size, 0.3 * size],
        ], dtype=np.float32)

        print(f"Using {len(ALIGNMENT_INDICES)} landmarks for alignment.")
        return template

    def process_video(self, video_path: str) -> Optional[Dict]:
        """
        Processes a video file to extract aligned face crops and metadata.

        Args:
            video_path: Path to the input video file.

        Returns:
            A dictionary containing processed frames, masks, transforms, etc.,
            or None if processing fails. Keys include:
            'video_path', 'landmarks_raw', 'landmarks_smoothed', 'transforms',
            'inverse_transforms', 'cropped_faces_paths', 'masked_inputs_paths',
            'reference_masks_paths', 'fps', 'frame_count'
            (Note: Images are not returned directly to save memory, paths are returned)
        """
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            print(f"Error: Video file has no frames or is corrupted: {video_path}")
            cap.release()
            return None
            
        print(f"Processing video: {os.path.basename(video_path)} ({frame_count} frames, {fps:.2f} FPS)")

        all_landmarks_raw = [] # Store raw landmarks (x,y,z scaled by image dims)

        # --- 1. Frame Reading and Landmark Detection ---
        for frame_idx in range(frame_count):
            success, frame = cap.read()
            if not success:
                print(f"Warning: Failed to read frame {frame_idx}. Skipping remaining frames.")
                # Pad landmarks_raw if needed to match expected frame count
                while len(all_landmarks_raw) < frame_count:
                    all_landmarks_raw.append(None)
                break # Exit loop

            img_h, img_w = frame.shape[:2]

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False # Performance optimization
            results = self.face_mesh.process(rgb_frame)
            rgb_frame.flags.writeable = True

            if results.multi_face_landmarks:
                # Get landmarks for the first (and assumed only) face
                face_landmarks = results.multi_face_landmarks[0]
                # Store landmarks scaled to image pixel coordinates
                landmarks = np.array([[lm.x * img_w, lm.y * img_h, lm.z]
                                      for lm in face_landmarks.landmark])
                all_landmarks_raw.append(landmarks)
            else:
                all_landmarks_raw.append(None) # Append None if no face detected
            
            print(f"  Detecting landmarks: Frame {frame_idx + 1}/{frame_count}", end='\r')

        cap.release()
        print("\n✓ Landmark detection complete.")

        if not any(lm is not None for lm in all_landmarks_raw):
            print("Error: No faces detected in any frame.")
            # Explicitly close MediaPipe resources
            self.face_mesh.close()
            return None

        # --- 2. Landmark Smoothing ---
        landmarks_smoothed = self._smooth_landmarks(all_landmarks_raw)
        print("✓ Landmark smoothing complete.")

        # --- 3. Alignment, Cropping, Masking (Process and Save Frame by Frame) ---
        # Instead of storing all images in memory, process and save them iteratively
        
        output_data = {
            'video_path': video_path,
            'fps': fps,
            'frame_count': frame_count,
            'output_size': self.output_size,
            'landmarks_raw': all_landmarks_raw, # Keep raw for potential debugging
            'landmarks_smoothed': [], # Will store the smoothed landmarks used
            'transforms': [], # Forward transforms
            'inverse_transforms': [], # Inverse transforms for rendering
            # Store paths instead of image arrays
            'cropped_faces_paths': [],
            'masked_inputs_paths': [],
            'reference_masks_paths': []
        }
        
        # Prepare output directories (example structure)
        base_output_dir = f"processed_data/{Path(video_path).stem}"
        cropped_dir = os.path.join(base_output_dir, "cropped_faces")
        masked_dir = os.path.join(base_output_dir, "masked_inputs")
        ref_mask_dir = os.path.join(base_output_dir, "reference_masks")
        os.makedirs(cropped_dir, exist_ok=True)
        os.makedirs(masked_dir, exist_ok=True)
        os.makedirs(ref_mask_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path) # Reopen video to read frames again
        
        for i in range(frame_count):
            success, frame = cap.read()
            if not success:
                print(f"Warning: Failed to read frame {i} on second pass.")
                # Append None placeholders for consistency
                output_data['landmarks_smoothed'].append(None)
                output_data['transforms'].append(None)
                output_data['inverse_transforms'].append(None)
                output_data['cropped_faces_paths'].append(None)
                output_data['masked_inputs_paths'].append(None)
                output_data['reference_masks_paths'].append(None)
                continue
                
            landmarks = landmarks_smoothed[i] # Use smoothed 2D landmarks

            if landmarks is None:
                # Handle frames where face wasn't detected/smoothed
                output_data['landmarks_smoothed'].append(None)
                output_data['transforms'].append(None)
                output_data['inverse_transforms'].append(None)
                output_data['cropped_faces_paths'].append(None)
                output_data['masked_inputs_paths'].append(None)
                output_data['reference_masks_paths'].append(None)
                continue
                
            output_data['landmarks_smoothed'].append(landmarks) # Store the landmarks used

            # --- 3a. [cite_start]Calculate Alignment Transform (Procrustes) --- [cite: 138]
            # Use only the specified eye/nose landmarks
            [cite_start]source_points = landmarks[ALIGNMENT_INDICES, :2].astype(np.float32) # Get x,y [cite: 203]
            target_points = self.canonical_template.astype(np.float32)

            transform_matrix, _ = cv2.estimateAffinePartial2D(source_points, target_points, method=cv2.LMEDS)

            if transform_matrix is None:
                 # Handle cases where transform estimation fails
                print(f"Warning: Failed to estimate transform for frame {i}. Skipping.")
                output_data['transforms'].append(None)
                output_data['inverse_transforms'].append(None)
                output_data['cropped_faces_paths'].append(None)
                output_data['masked_inputs_paths'].append(None)
                output_data['reference_masks_paths'].append(None)
                continue
                
            output_data['transforms'].append(transform_matrix) # 2x3 matrix

            # --- 3b. Warp and Crop ---
            cropped_face = cv2.warpAffine(
                frame,
                transform_matrix,
                (self.output_size, self.output_size),
                [cite_start]flags=cv2.INTER_CUBIC # Paper mentions bicubic [cite: 139]
            )
            cropped_path = os.path.join(cropped_dir, f"frame_{i:06d}.png")
            cv2.imwrite(cropped_path, cropped_face)
            output_data['cropped_faces_paths'].append(cropped_path)

            # --- 3c. [cite_start]Generate Masks --- [cite: 163, 285]
            mask_coords_px = ( # Convert normalized rect to pixel coords
                int(self.norm_mask_rect[0] * self.output_size),
                int(self.norm_mask_rect[1] * self.output_size),
                int(self.norm_mask_rect[2] * self.output_size),
                int(self.norm_mask_rect[3] * self.output_size)
            )
            
            # [cite_start]Input mask (zeros in mouth region) [cite: 165]
            masked_input = cropped_face.copy()
            masked_input[mask_coords_px[1]:mask_coords_px[3], mask_coords_px[0]:mask_coords_px[2]] = 0
            masked_path = os.path.join(masked_dir, f"frame_{i:06d}.png")
            cv2.imwrite(masked_path, masked_input)
            output_data['masked_inputs_paths'].append(masked_path)

        
            # So, we create a mask that keeps *only* the mouth rectangle.
            reference_mask_img = np.zeros_like(cropped_face)
            reference_mask_img[mask_coords_px[1]:mask_coords_px[3], mask_coords_px[0]:mask_coords_px[2]] = \
                cropped_face[mask_coords_px[1]:mask_coords_px[3], mask_coords_px[0]:mask_coords_px[2]]
            ref_mask_path = os.path.join(ref_mask_dir, f"frame_{i:06d}.png")
            cv2.imwrite(ref_mask_path, reference_mask_img)
            output_data['reference_masks_paths'].append(ref_mask_path)
            
            # --- 3d. [cite_start]Calculate Inverse Transform (for rendering) --- [cite: 407]
            inverse_transform = cv2.invertAffineTransform(transform_matrix)
            output_data['inverse_transforms'].append(inverse_transform)

            print(f"  Processing & Saving frames: {i+1}/{frame_count}", end='\r')

        cap.release()
        # Explicitly close MediaPipe resources
        self.face_mesh.close()
        
        print("\n✓ Alignment, cropping, masking, and saving complete.")
        
        # Save metadata and numpy arrays at the end
        self._save_metadata_and_arrays(output_data, base_output_dir)

        return output_data # Return dict with paths and arrays

    def _smooth_landmarks(self, all_landmarks_raw: List[Optional[np.ndarray]]) -> List[Optional[np.ndarray]]:
        """
        [cite_start]Smooth landmarks over time using a Gaussian filter[cite: 123].
        Handles missing frames by interpolation.
        Returns smoothed 2D landmarks (x, y).
        """
        num_frames = len(all_landmarks_raw)
        if num_frames == 0:
            return []
            
        first_valid_idx = -1
        num_landmarks = 0
        for i, lms in enumerate(all_landmarks_raw):
            if lms is not None:
                first_valid_idx = i
                num_landmarks = lms.shape[0] # Get count from first valid frame
                break
        
        if first_valid_idx == -1: # No landmarks found at all
            return [None] * num_frames
            
        landmark_tensor = np.zeros((num_frames, num_landmarks, 2), dtype=np.float32)
        valid_frames = np.zeros(num_frames, dtype=bool)

        for i, lms in enumerate(all_landmarks_raw):
            if lms is not None and lms.shape[0] == num_landmarks:
                landmark_tensor[i] = lms[:, :2] # Use only x, y
                valid_frames[i] = True
            else:
                valid_frames[i] = False

        if not np.all(valid_frames):
            # print("Interpolating missing landmarks...") # Can be verbose
            frame_indices = np.arange(num_frames)
            for j in range(num_landmarks):
                for k in range(2): # x and y
                    valid_coords = landmark_tensor[valid_frames, j, k]
                    valid_indices = frame_indices[valid_frames]
                    
                    if len(valid_indices) < 2:
                        # Fallback: fill missing with nearest valid or default (e.g., 0)
                        if len(valid_indices) == 1:
                            fill_value = valid_coords[0]
                        else:
                            fill_value = 0 # Or estimate default position
                        landmark_tensor[~valid_frames, j, k] = fill_value
                        landmark_tensor[:, j, k] = np.interp(frame_indices, valid_indices, valid_coords)
                    else:
                         landmark_tensor[:, j, k] = np.interp(frame_indices, valid_indices, valid_coords)

        if self.smoothing_sigma > 0 and num_frames > int(3 * self.smoothing_sigma):
            smoothed_tensor = gaussian_filter1d(landmark_tensor, sigma=self.smoothing_sigma, axis=0, mode='nearest')
        else:
            smoothed_tensor = landmark_tensor
            
        smoothed_landmarks_list = []
        for i in range(num_frames):
            if all_landmarks_raw[i] is not None:
                 smoothed_landmarks_list.append(smoothed_tensor[i])
            else:
                 smoothed_landmarks_list.append(None) # Keep None for frames initially missed
                 
        return smoothed_landmarks_list

    def _save_metadata_and_arrays(self, data: Dict, output_dir: str):
        """Saves metadata and numpy arrays related to processing."""
        base_name = Path(data['video_path']).stem
        
        # --- Save Metadata ---
        metadata = {
            'video_path': data['video_path'],
            'num_frames': data['frame_count'],
            'fps': data['fps'],
            'output_size': data['output_size'],
            'alignment_indices_used': ALIGNMENT_INDICES,
            'mask_rect_norm': self.norm_mask_rect
        }
        meta_path = os.path.join(output_dir, f"{base_name}_meta.json")
        try:
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Saved metadata to {meta_path}")
        except Exception as e:
            print(f"Error saving metadata: {e}")

        # --- Save Numpy Arrays ---
        npz_path = os.path.join(output_dir, f"{base_name}_data.npz")
        try:
            # Need to handle None values properly for saving
            landmarks_to_save = [lm if lm is not None else np.array([]) for lm in data['landmarks_smoothed']]
            transforms_to_save = [t if t is not None else np.array([]) for t in data['transforms']]
            inv_transforms_to_save = [it if it is not None else np.array([]) for it in data['inverse_transforms']]
            
            np.savez_compressed(
                npz_path,
                landmarks_smoothed=np.array(landmarks_to_save, dtype=object),
                transforms=np.array(transforms_to_save, dtype=object),
                inverse_transforms=np.array(inv_transforms_to_save, dtype=object)
            )
            print(f"Saved landmarks and transforms to {npz_path}")
        except Exception as e:
            print(f"Error saving numpy data: {e}")


# ============================================================================
# === Utility Functions ===
# ============================================================================

# (load_processed_data function remains useful for loading the saved .npz/json)

def load_processed_data(data_dir: str) -> Optional[Dict]:
    """
    Loads the processed data back from disk (metadata and numpy arrays).
    Image paths can be reconstructed or loaded separately.
    """
    base_name = None
    meta_path = None
    
    # Find the metadata file to determine the base name
    for fname in os.listdir(data_dir):
        if fname.endswith("_meta.json"):
            meta_path = os.path.join(data_dir, fname)
            base_name = fname.replace("_meta.json", "")
            break
            
    if not meta_path or not base_name:
        print(f"Error: Could not find '_meta.json' file in {data_dir}")
        return None
        
    data_npz_path = os.path.join(data_dir, f"{base_name}_data.npz")
    if not os.path.exists(data_npz_path):
        print(f"Error: Could not find '{base_name}_data.npz' file in {data_dir}")
        return None

    print(f"Loading processed data for '{base_name}' from {data_dir}")

    # Load metadata
    with open(meta_path, 'r') as f:
        data = json.load(f)

    # Load numpy data
    try:
        np_data = np.load(data_npz_path, allow_pickle=True)
        # Convert back from object arrays, handling potential empty arrays for Nones
        data['landmarks_smoothed'] = [lm if lm.size > 0 else None for lm in np_data['landmarks_smoothed']]
        data['transforms'] = [t if t.size > 0 else None for t in np_data['transforms']]
        data['inverse_transforms'] = [it if it.size > 0 else None for it in np_data['inverse_transforms']]
    except Exception as e:
        print(f"Error loading numpy data: {e}")
        return None
        
    # Add paths to image sequences (assuming standard directory structure)
    data['cropped_faces_dir'] = os.path.join(data_dir, 'cropped_faces')
    data['masked_inputs_dir'] = os.path.join(data_dir, 'masked_inputs')
    data['reference_masks_dir'] = os.path.join(data_dir, 'reference_masks')
    
    print("✓ Data loaded (metadata and arrays). Image paths constructed.")
    return data

# (load_image_sequence remains useful if needed later)
def load_image_sequence(dir_path: str) -> List[Optional[np.ndarray]]:
    images = []
    if not os.path.isdir(dir_path):
        print(f"Warning: Image directory not found: {dir_path}")
        return []
        
    fnames = sorted([f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not fnames:
        print(f"Warning: No images found in {dir_path}")
        return []
        
    print(f"Loading image sequence from {dir_path} ({len(fnames)} frames)...")
    for i, fname in enumerate(fnames):
        img_path = os.path.join(dir_path, fname)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
        else:
            print(f"Warning: Failed to load image {img_path}")
            images.append(None) # Keep placeholder if loading failed
        print(f"  Loading frame {i+1}/{len(fnames)}", end='\r')
    print("\n✓ Image sequence loaded.")
    return images