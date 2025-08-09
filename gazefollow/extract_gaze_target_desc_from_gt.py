import os
import sys
print(f"Current Working Directory: {os.getcwd()}")
print(f"Python Path: {sys.path}")
import cv2
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
# from docs.research_utils import fix_wsl_paths
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from huggingface_hub import snapshot_download
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2_point_segmentation import load_sam2_model, segment_with_points
import requests
import copy
import torch
import os

def fix_wsl_paths(path: str) -> str:
    """Convert Windows paths to WSL paths if necessary."""
    # If path already starts with /mnt/, assume it's correct WSL format
    if path.startswith('/mnt/'):
        return path
    # Otherwise, attempt conversion from Windows format
    path = path.replace("\\", os.sep)
    drive_parts = path.split(os.sep)
    if len(drive_parts) > 0 and len(drive_parts[0]) > 1 and drive_parts[0][1] == ':':
        drive_letter = drive_parts[0][0].lower()
        # Reconstruct path starting from /mnt/<drive_letter>/...
        wsl_path = f'/mnt/{drive_letter}/' + os.sep.join(drive_parts[1:])
        return wsl_path
    else:
        # If it doesn't look like a Windows path either, return original
        return path

model_id = 'microsoft/Florence-2-large'
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

local_dir = snapshot_download(
    "microsoft/Florence-2-large",
 #   cache_dir="/mnt/ssd/hf_cache"     # point at your fastest disk
)
# model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().cuda()
# model = AutoModelForCausalLM.from_pretrained(local_dir, local_files_only=True, trust_remote_code=True, torch_dtype='auto').eval().cuda()
florence_model = AutoModelForCausalLM.from_pretrained(local_dir, local_files_only=True,  # skip all remote I/O
                                                      trust_remote_code=True,
                                                      torch_dtype="auto",
                                                      device_map="auto",
                                                      low_cpu_mem_usage=True, ).eval().cuda()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
# build sam 2
sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
sam2_predictor = SAM2ImagePredictor(sam2_model)


task_prompt = '<REGION_TO_DESCRIPTION>'
# annot_path = r"D:\Projects\data\gazefollow\test_annotations_release.txt"
annot_path = r"D:\Projects\data\gazefollow\train_annotations_release.txt"
split_type = "train" if "train" in annot_path else "test2"
gaze_segmentations_dir = Path(annot_path).parent / f"{split_type}_gaze_segmentations"
base_data_dir_path = fr"D:\Projects\data\gazefollow"
llava_results_dir = r"D:\Projects\LLaVA-NeXT\llava_attention_sweep\20250503_001255_You_are_an_expert_vision_assis"
gaze_segmentations_dir.mkdir(parents=True, exist_ok=True)
llava_results_dir = Path(fix_wsl_paths(llava_results_dir))
base_data_dir_path = Path(fix_wsl_paths(base_data_dir_path))
annot_path = fix_wsl_paths(annot_path)
gaze_segmentations_dir = Path(fix_wsl_paths(str(gaze_segmentations_dir)))
#%%

def load_gt_data(annot_path):
    df = pd.read_csv(annot_path, sep="\t", header=None)
    # split the columns with ',' delimeter
    df = df[0].str.split(",", expand=True)
    # add the columns names:
    # [image_path,id,body_bbox_x,body_bbox_y,body_bbox_width,body_bbox_height,eye_x,eye_y,gaze_x,gaze_y,head_bbox_x_min,head_bbox_y_min,head_bbox_x_max,head_bbox_y_max,in_or_out,meta]
    if len(df.columns) == 17:
        df.columns = ['image_path', 'id', 'body_bbox_x', 'body_bbox_y', 'body_bbox_width', 'body_bbox_height',
                      'eye_x', 'eye_y', 'gaze_x', 'gaze_y', 'head_bbox_x_min', 'head_bbox_y_min',
                      'head_bbox_x_max', 'head_bbox_y_max', 'in_or_out', 'meta', 'original_path']
    else:
        df.columns = ['image_path', 'id', 'body_bbox_x', 'body_bbox_y', 'body_bbox_width', 'body_bbox_height',
                      'eye_x', 'eye_y', 'gaze_x', 'gaze_y', 'head_bbox_x_min', 'head_bbox_y_min',
                      'head_bbox_x_max', 'head_bbox_y_max', 'meta', 'original_path']

    # Convert all the numerical columns to numeric types
    numeric_columns = ['id', 'body_bbox_x', 'body_bbox_y', 'body_bbox_width', 'body_bbox_height',
                       'eye_x', 'eye_y', 'gaze_x', 'gaze_y',
                       'head_bbox_x_min', 'head_bbox_y_min', 'head_bbox_x_max', 'head_bbox_y_max']

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def get_intersection(bbox1, bbox2):
    """
    Calculate the intersection of two bounding boxes.

    Parameters
    ----------
    bbox1 : list or array-like
        Coordinates of the first bounding box in the format [x_min, y_min, x_max, y_max].
    bbox2 : list or array-like
        Coordinates of the second bounding box in the format [x_min, y_min, x_max, y_max].

    Returns
    -------
    float
        The intersection area.
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    # If there is no overlap, the intersection area is 0, so IoU is 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate the area of intersection
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    return intersection_area

def get_iou(bbox1, bbox2):
    intersection_area = get_intersection(bbox1, bbox2)
    total_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]) + (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]) - intersection_area
    if total_area == 0:
        return 0.0
    iou = intersection_area / total_area
    return iou

def run_example(task_prompt, image, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
    generated_ids = florence_model.generate(
      input_ids=inputs["input_ids"].cuda(),
      pixel_values=inputs["pixel_values"].cuda(),
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

def group_by_image(df):
    # group by image_path
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

# main
# if __name__ == "__main__":
df = load_gt_data(annot_path)
compact_df = group_by_image(df)

df['person_desc'] = ''
save_interval = 5
results_dd = {}
#%%
dense_caption = True
segment_w_bb = True
nb_images = 1000
# randomly choose nb_images from dataframe
# compact_df = compact_df.sample(nb_images)
# iterate and extract person descriptions for each image
progr = tqdm(compact_df.iterrows(), total=compact_df.shape[0])
for index, row in progr:
    caption_found = False
    # get the relevant rows in the original dataframe
    relevant_rows = df['image_path'] == row['image_path']
    in_or_out = df.loc[relevant_rows, 'in_or_out'].values[0]

    # gather all the gaze points from relevant rows
    gaze_points = df.loc[relevant_rows, ['gaze_x', 'gaze_y']].values

    # add random noise gaze points
    # noise = np.random.uniform(-0.01, 0.01, size=gaze_points.shape)
   # load the image
    img_path = base_data_dir_path / row['image_path']
    if not img_path.exists():
       continue
    if in_or_out == '0':
        results_dd[row['image_path']] = 'outside of the frame'
        continue
    img = Image.open(img_path).convert("RGB")
    h = img.height
    w = img.width

    # lets create a small bounding box around the gaze points
    # get the min and max of the gaze points
    offset = 0.00
    min_x = (np.min(gaze_points[:, 0]) - offset)
    max_x = (np.max(gaze_points[:, 0]) + offset)
    min_y = (np.min(gaze_points[:, 1]) - offset)
    max_y = (np.max(gaze_points[:, 1]) + offset)
    # create a small bounding box around the gaze points
    bounding_box = np.array([min_x, min_y, max_x, max_y]).reshape(-1, 4)
    bounding_box = torch.from_numpy(bounding_box)
    box_labels = [[1]]
    box_confidences = np.array([1])
    # correct gaze points to their absolute values
    gaze_points[:, 0] = (w * gaze_points[:, 0]).astype(int)
    gaze_points[:, 1] = (h * gaze_points[:, 1]).astype(int)
    # segment with bounding box
    if segment_w_bb:
        result_path, person_results = segment_with_points(
            image_path=str(img_path),
            point_coords=gaze_points,
            boxes=bounding_box,
            sam2_predictor=sam2_predictor,
            box_labels=box_labels,
            box_confidences=box_confidences,
            increase_box_offset=35,
            plot_all_masks=False,
            use_intersection=False,
            save_masks=True
        )
    else:
        gaze_points[:, 0] = (w * gaze_points[:, 0]).astype(int)
        gaze_points[:, 1] = (w * gaze_points[:, 1]).astype(int)
        # segment with gaze points
        result_path, person_results = segment_with_points(
            image_path=str(img_path),
            point_coords=gaze_points,
            sam2_predictor=sam2_predictor,
            output_dir=str(gaze_segmentations_dir),
            prefix=f"gaze_",
            save_masks=True,
        )

    # get the gaze target
    gaze_target_box = person_results.get('boxes', None)
    if gaze_target_box is None:
        results_dd[row['image_path']] = 'missing gaze target'
        continue
    # in case the resulted gaze_target_box also includes a large part of the body bbox, use the static_boxes
    # get body bbox
    body_bbox = row[['body_bbox_x', 'body_bbox_y', 'body_bbox_width', 'body_bbox_height']].values
    body_bbox = [body_bbox[0], body_bbox[1], body_bbox[0] + body_bbox[2], body_bbox[1] + body_bbox[3]]
    # normalize the bbox to the image size
    body_bbox = [body_bbox[0] * w, body_bbox[1] * h, body_bbox[2] * w, body_bbox[3] * h]
    total_body_area = (body_bbox[2] - body_bbox[0]) * (body_bbox[3] - body_bbox[1])

    # check the iou between the gaze_target_box and the body_bbox
    intersection_area = get_intersection(gaze_target_box[0], body_bbox)
    if intersection_area / total_body_area > 0.5:
        print(f"Using static boxes for {row['image_path']}")
        gaze_target_box = person_results.get('static_boxes', None)
        if gaze_target_box is None:
            results_dd[row['image_path']] = 'missing gaze target'
            continue
        gaze_target_box = gaze_target_box.numpy()
    gaze_target_box = gaze_target_box[0]
    # lets normalize the bboxes to the [0-1]*1000 range
    gaze_target_x = int(1000 * gaze_target_box[0] / w)
    gaze_target_y = int(1000 * gaze_target_box[1] / h)
    gaze_target_x_end = int(1000 * gaze_target_box[2] / w)
    gaze_target_y_end = int(1000 * gaze_target_box[3] / h)

    if dense_caption:
        task_prompt = '<MORE_DETAILED_CAPTION>'
        results = run_example(task_prompt, img)
        text_input = results[task_prompt]
        task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
        results = run_example(task_prompt, img, text_input)
        results['<MORE_DETAILED_CAPTION>'] = text_input
        # lets remove boxes that have high intersection with the body bbox
        bad_boxes = []
        bad_labels = []
        for box, label in zip(results['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes'], results['<CAPTION_TO_PHRASE_GROUNDING>']['labels']):
            if get_intersection(box, body_bbox) / total_body_area > 0.5:
                # lets remove the boxes that have large intersection with our origin person
                bad_boxes.append(box)
                bad_labels.append(label)
        for box, label in zip(bad_boxes, bad_labels):
            results['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes'].remove(box)
            results['<CAPTION_TO_PHRASE_GROUNDING>']['labels'].remove(label)
            
        # get the iou between the gaze_target_box and all the boxes in the results
        iou_scores = []
        for box in results['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes']:
            iou_scores.append(get_iou(gaze_target_box, box))
        # get the max iou score
        max_iou_score = max(iou_scores)
        if max_iou_score > 0.7:
            # get the index of the max iou score
            max_iou_index = iou_scores.index(max_iou_score)
            # get the text of the max iou score
            desc = results['<CAPTION_TO_PHRASE_GROUNDING>']['labels'][max_iou_index]
            target_bbox = results['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes'][max_iou_index]
            # convert to ints
            target_bbox = [int(val) for val in target_bbox]
            caption_found = True

    if not caption_found:
        target_bbox = gaze_target_box.tolist()
        task_prompt = '<REGION_TO_DESCRIPTION>'
        results = run_example(task_prompt, img, text_input=f"<loc_{gaze_target_x}><loc_{gaze_target_y}>"
                                                           f"<loc_{gaze_target_x_end}><loc_{gaze_target_y_end}>")
        desc = results.get(task_prompt, '').split('<')[0]
    progr.set_description(f"Person description:{desc}")
    # save the person description in the dataframe
    # df.loc[relevant_rows, 'person_desc'] = desc

    results_dd[row['image_path']] = {'gaze target description': desc,
                                     'bbox': target_bbox}

    # save the results
    # if index % save_interval == 0:
        # save the results to a json file
    results_file = Path(annot_path).parent / f"{split_type}_gazetarget_results.json"
    try:
        with open(results_file, 'w') as f:
            json.dump(results_dd, f, indent=4)
    except:
        pass

