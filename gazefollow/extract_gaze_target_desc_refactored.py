import os
import sys
import cv2
import json
import torch
import numpy as np
import pandas as pd
from sympy.codegen.ast import continue_
from tqdm import tqdm
from pathlib import Path
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from huggingface_hub import snapshot_download
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2_point_segmentation import segment_with_points # Assuming this function is in a file named sam2_point_segmentation.py

# --- Configuration ---
MODEL_ID = 'microsoft/Florence-2-large'
SAM2_CHECKPOINT_PATH = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG_PATH = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ANNOT_PATH = r"D:\Projects\data\gazefollow\test_annotations_release.txt"
ANNOT_PATH = r"D:\Projects\data\gazefollow\train_annotations_release.txt"
SPLIT_TYPE = "train" if "train" in ANNOT_PATH else "test"
BASE_DATA_DIR_PATH = r"D:\Projects\data\gazefollow"
LLAVA_RESULTS_DIR = r"D:\Projects\LLaVA-NeXT\llava_attention_sweep\20250503_001255_You_are_an_expert_vision_assis" # Note: year seems off

# --- Helper Functions ---
def fix_wsl_paths(path: str) -> str:
    """Convert Windows paths to WSL paths if necessary."""
    if path.startswith('/mnt/'):
        return path
    path = path.replace("\\", os.sep) # Use raw string or escape backslashes properly
    drive_parts = path.split(os.sep)
    if len(drive_parts) > 0 and len(drive_parts[0]) > 1 and drive_parts[0][1] == ':':
        drive_letter = drive_parts[0][0].lower()
        wsl_path = f'/mnt/{drive_letter}/' + os.sep.join(drive_parts[1:])
        return wsl_path
    return path

# --- Initialize Models ---
print(f"Current Working Directory: {os.getcwd()}")
print(f"Python Path: {sys.path}")

# Apply WSL path fix to relevant paths
annot_path_fixed = fix_wsl_paths(ANNOT_PATH)
base_data_dir_path_fixed = Path(fix_wsl_paths(BASE_DATA_DIR_PATH))
llava_results_dir_fixed = Path(fix_wsl_paths(LLAVA_RESULTS_DIR))
gaze_segmentations_dir_fixed = Path(annot_path_fixed).parent / f"{SPLIT_TYPE}_gaze_segmentations"
gaze_segmentations_dir_fixed.mkdir(parents=True, exist_ok=True)

# Download Florence-2 model
local_dir = snapshot_download(
    MODEL_ID,
    # cache_dir="/mnt/ssd/hf_cache" # Optional: point at your fastest disk
)

# Load Florence-2 model and processor
florence_model = AutoModelForCausalLM.from_pretrained(
    local_dir,
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
    low_cpu_mem_usage=True,
).eval().to(DEVICE) # Explicitly move to device
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# Build SAM2 model and predictor
sam2_model = build_sam2(SAM2_CONFIG_PATH, SAM2_CHECKPOINT_PATH, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

print("Models loaded successfully.")

# --- Data Loading and Preprocessing ---
def load_gt_data(annot_path_str: str):
    df = pd.read_csv(annot_path_str, sep="\t", header=None)
    df = df[0].str.split(",", expand=True)
    if len(df.columns) == 17:
        df.columns = ['image_path', 'id', 'body_bbox_x', 'body_bbox_y', 'body_bbox_width', 'body_bbox_height',
                      'eye_x', 'eye_y', 'gaze_x', 'gaze_y', 'head_bbox_x_min', 'head_bbox_y_min',
                      'head_bbox_x_max', 'head_bbox_y_max', 'in_or_out', 'meta', 'original_path']
    else:
        df.columns = ['image_path', 'id', 'body_bbox_x', 'body_bbox_y', 'body_bbox_width', 'body_bbox_height',
                      'eye_x', 'eye_y', 'gaze_x', 'gaze_y', 'head_bbox_x_min', 'head_bbox_y_min',
                      'head_bbox_x_max', 'head_bbox_y_max', 'meta', 'original_path']

    numeric_columns = ['id', 'body_bbox_x', 'body_bbox_y', 'body_bbox_width', 'body_bbox_height',
                       'eye_x', 'eye_y', 'gaze_x', 'gaze_y',
                       'head_bbox_x_min', 'head_bbox_y_min', 'head_bbox_x_max', 'head_bbox_y_max']

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def group_by_image(df):
    compact_df = df.groupby('image_path').agg({
        'eye_x': 'mean',
        'eye_y': 'mean',
        'gaze_x': 'mean',
        'gaze_y': 'mean',
        'body_bbox_x': 'mean',
        'body_bbox_y': 'mean',
        'body_bbox_width': 'mean',
        'body_bbox_height': 'mean',
    }).reset_index()
    return compact_df

# --- Bounding Box Utilities ---
def get_intersection(bbox1, bbox2):
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    return intersection_area

def get_iou(bbox1, bbox2):
    intersection_area = get_intersection(bbox1, bbox2)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    total_area = bbox1_area + bbox2_area - intersection_area
    if total_area == 0:
        return 0.0
    iou = intersection_area / total_area
    return iou

# --- Florence Model Interaction ---
def run_florence_example(task_prompt, image, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE, torch.float16)
    generated_ids = florence_model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=8,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    return parsed_answer

# --- Main Processing Logic ---
def process_image_row(row_data, original_df, config):
    img_path = config["base_data_dir"] / row_data['image_path']
    if not img_path.exists():
        return row_data['image_path'], "image file not found"

    relevant_rows_mask = original_df['image_path'] == row_data['image_path']
    in_or_out = original_df.loc[relevant_rows_mask, 'in_or_out'].values[0]

    if in_or_out == '0':
        return row_data['image_path'], 'outside of the frame'

    img = Image.open(img_path).convert("RGB")
    h, w = img.height, img.width

    gaze_points_relative = original_df.loc[relevant_rows_mask, ['gaze_x', 'gaze_y']].values
    gaze_points_absolute = gaze_points_relative.copy()
    gaze_points_absolute[:, 0] = (w * gaze_points_absolute[:, 0]).astype(int)
    gaze_points_absolute[:, 1] = (h * gaze_points_absolute[:, 1]).astype(int)

    # Create a small bounding box around the gaze points
    offset = 0.00
    min_x_rel = (np.min(gaze_points_relative[:, 0]) - offset)
    max_x_rel = (np.max(gaze_points_relative[:, 0]) + offset)
    min_y_rel = (np.min(gaze_points_relative[:, 1]) - offset)
    max_y_rel = (np.max(gaze_points_relative[:, 1]) + offset)
    # Bbox in relative coordinates [0,1]
    gaze_bbox_relative = np.array([min_x_rel, min_y_rel, max_x_rel, max_y_rel]).reshape(-1, 4)
    gaze_bbox_relative_torch = torch.from_numpy(gaze_bbox_relative)

    segmentation_args = {
        "image_path": str(img_path),
        "sam2_predictor": sam2_predictor, # Passed from global scope
        "output_dir": str(config["gaze_segmentations_dir"]),
        "prefix": "gaze_",
        "save_masks": True,
        "plot_all_masks": False,
        "use_intersection": False,
    }

    if config["segment_w_bb"]:
        _, person_results = segment_with_points(
            **segmentation_args,
            point_coords=gaze_points_absolute, # SAM2 expects absolute coords
            boxes=gaze_bbox_relative_torch, # SAM2 expects relative coords for boxes
            box_labels=[[1]],
            box_confidences=np.array([1]),
            increase_box_offset=35,
        )
    else:
        _, person_results = segment_with_points(
            **segmentation_args,
            point_coords=gaze_points_absolute, # SAM2 expects absolute coords
        )

    gaze_target_box_absolute = person_results.get('boxes', None)
    if gaze_target_box_absolute is None:
        return row_data['image_path'], 'missing gaze target after SAM2 segmentation'

    body_bbox_relative = row_data[['body_bbox_x', 'body_bbox_y', 'body_bbox_width', 'body_bbox_height']].values
    body_bbox_absolute = [
        body_bbox_relative[0] * w, 
        body_bbox_relative[1] * h, 
        (body_bbox_relative[0] + body_bbox_relative[2]) * w, 
        (body_bbox_relative[1] + body_bbox_relative[3]) * h
    ]
    total_body_area = (body_bbox_absolute[2] - body_bbox_absolute[0]) * (body_bbox_absolute[3] - body_bbox_absolute[1])
    
    # Check if the segmented gaze target significantly overlaps with the body
    if total_body_area > 0: # Avoid division by zero if body_bbox is invalid
        intersection_with_body = get_intersection(gaze_target_box_absolute[0], body_bbox_absolute)
        if intersection_with_body / total_body_area > 0.5:
            print(f"Using static boxes for {row_data['image_path']} due to high overlap with body.")
            gaze_target_box_absolute = person_results.get('static_boxes', None)
            if gaze_target_box_absolute is None:
                return row_data['image_path'], 'missing gaze target (static boxes failed)'
            gaze_target_box_absolute = gaze_target_box_absolute.numpy() # Ensure it's a numpy array

    final_gaze_target_box = gaze_target_box_absolute[0]

    caption_found = False
    target_bbox_for_florence = final_gaze_target_box # Default to SAM2 output
    desc = ""
    detailed_caption_text = ""

    if config["dense_caption"]:
        detailed_caption_prompt = '<MORE_DETAILED_CAPTION>'
        detailed_caption_results = run_florence_example(detailed_caption_prompt, img)
        detailed_caption_text = detailed_caption_results.get(detailed_caption_prompt, "")

        if detailed_caption_text:
            grounding_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
            grounding_results = run_florence_example(grounding_prompt, img, detailed_caption_text)
            
            bboxes_from_grounding = grounding_results.get('<CAPTION_TO_PHRASE_GROUNDING>', {}).get('bboxes', [])
            labels_from_grounding = grounding_results.get('<CAPTION_TO_PHRASE_GROUNDING>', {}).get('labels', [])
            
            filtered_bboxes = []
            filtered_labels = []
            if total_body_area > 0: # ensure total_body_area is not zero
                for box, label in zip(bboxes_from_grounding, labels_from_grounding):
                    if get_intersection(box, body_bbox_absolute) / total_body_area <= 0.5:
                        filtered_bboxes.append(box)
                        filtered_labels.append(label)
            else: # if total_body_area is zero, no filtering based on body bbox can be done
                filtered_bboxes = bboxes_from_grounding
                filtered_labels = labels_from_grounding

            if filtered_bboxes:
                iou_scores = [get_iou(final_gaze_target_box, box) for box in filtered_bboxes]
                if iou_scores: # ensure iou_scores is not empty
                    max_iou_score = max(iou_scores)
                    if max_iou_score > 0.1:
                        max_iou_index = iou_scores.index(max_iou_score)
                        desc = filtered_labels[max_iou_index]
                        target_bbox_for_florence = filtered_bboxes[max_iou_index]
                        caption_found = True
    
    if not caption_found:
        # doesnt seem to produce good results, skip the image if previous steps unsuccessful
        pass

        # Convert absolute gaze target box to <loc_x><loc_y><loc_x_end><loc_y_end> format for Florence
    #     gaze_target_loc_str = (
    #         f"<loc_{int(1000 * final_gaze_target_box[0] / w)}><loc_{int(1000 * final_gaze_target_box[1] / h)}>"
    #         f"<loc_{int(1000 * final_gaze_target_box[2] / w)}><loc_{int(1000 * final_gaze_target_box[3] / h)}>"
    #     )
    #     region_desc_prompt = '<REGION_TO_DESCRIPTION>'
    #     region_desc_results = run_florence_example(region_desc_prompt, img, text_input=gaze_target_loc_str)
    #     desc = region_desc_results.get(region_desc_prompt, '').split('<')[0].strip()
    #     target_bbox_for_florence = final_gaze_target_box # Use the SAM2 box if dense captioning failed or wasn't used
    #
    return row_data['image_path'], {'gaze target description': desc, 'bbox': [int(v) for v in target_bbox_for_florence],
                                    'full_img_caption': detailed_caption_text}

def main():
    df_gt = load_gt_data(annot_path_fixed) # Use fixed path
    compact_df = group_by_image(df_gt)

    # Configuration for processing
    # Note: SPLIT_TYPE, BASE_DATA_DIR_PATH, ANNOT_PATH are now global constants
    # and their fixed versions are used for paths.
    # gaze_segmentations_dir_fixed is also globally defined.
    processing_config = {
        "base_data_dir": base_data_dir_path_fixed,
        "gaze_segmentations_dir": gaze_segmentations_dir_fixed,
        "dense_caption": True, # Set to False to disable dense captioning part
        "segment_w_bb": True,  # Set to False to segment with points only
        "save_interval": 100, # How often to save intermediate results
    }

    results_dd = {}
    # nb_images = 10 # For testing: process a small number of images
    # compact_df_subset = compact_df.sample(nb_images) if nb_images else compact_df
    compact_df_subset = compact_df # Process all
    
    # Start processing from a specific index if needed, e.g., for resuming
    start_index = 54
    # compact_df_subset = compact_df.iloc[start_index:] 

    progress_bar = tqdm(compact_df_subset.iterrows(), total=len(compact_df_subset))
    for index, row in progress_bar:
        if index < start_index: # if using iloc above, this check is not needed.
            continue

        image_id, result = process_image_row(row, df_gt, processing_config)
        results_dd[image_id] = result
        
        if isinstance(result, dict) and result.get('gaze target description'):
            progress_bar.set_description(f"Desc: {result['gaze target description']}")
        elif isinstance(result, str): # Handle cases where a string message is returned
            progress_bar.set_description(f"Skipped: {result}")
        else:
             progress_bar.set_description("Processing...")

        if (index + 1) % processing_config["save_interval"] == 0 or (index + 1) == len(compact_df_subset):
            results_file = Path(annot_path_fixed).parent / f"{SPLIT_TYPE}_gazetarget_results_refactored.json"
            # Basic error handling for saving, without try-except
            temp_save_path = results_file.with_suffix(".json.tmp")
            save_successful = False
            with open(temp_save_path, 'w') as f:
                json.dump(results_dd, f, indent=4)
                save_successful = True # Assume dump is successful if no error before this
            
            if save_successful:
                 os.replace(temp_save_path, results_file) # Atomic rename if possible
                 print(f"Results saved to {results_file}")
            else:
                 print(f"Failed to save results to {temp_save_path}. Previous results (if any) at {results_file} are preserved.")
                 if temp_save_path.exists():
                     os.remove(temp_save_path) # Clean up temp file if save failed

    print("Processing complete.")
    final_results_file = Path(annot_path_fixed).parent / f"{SPLIT_TYPE}_gazetarget_results_refactored.json"
    with open(final_results_file, 'w') as f:
        json.dump(results_dd, f, indent=4)
    print(f"Final results saved to {final_results_file}")

if __name__ == "__main__":
    main() 