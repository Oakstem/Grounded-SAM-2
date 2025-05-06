#!/usr/bin/env python
import os
import sys
import torch
import traceback
import numpy as np
from pathlib import Path
import argparse
import re
import json
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from sam2_point_segmentation import load_sam2_model, segment_with_points

def windows_to_wsl_path(windows_path):
    """Convert a Windows path to a WSL path."""
    if not windows_path:
        return None
        
    # Remove drive letter and convert backslashes to forward slashes
    if ':' in windows_path:
        drive, path = windows_path.split(':', 1)
        path = path.replace('\\', '/')
        wsl_path = f"/mnt/{drive.lower()}{path}"
        return wsl_path
    return windows_path.replace('\\', '/')

def load_points_from_file(file_path):
    """Load point coordinates from a file (supports .pt files with 'centers' key)."""
    # Convert PosixPath to string if needed
    file_path_str = str(file_path)
    if file_path_str.endswith('.pt'):
        points_data = torch.load(file_path, weights_only=False)
        if isinstance(points_data, dict) and 'centers' in points_data:
            return points_data['centers']
        else:
            raise ValueError(f"The .pt file does not contain the 'centers' key")
    else:
        raise ValueError(f"Unsupported file format for points: {file_path}")

def detect_image_ids(attn_base_dir):
    """Automatically detect image IDs from directory names."""
    if os.name != 'nt' and '\\' in attn_base_dir:
        attn_base_dir = windows_to_wsl_path(attn_base_dir)
    
    base_path = Path(attn_base_dir)
    image_ids = []
    
    # Pattern to match directories like: 00000001_attn
    attn_pattern = re.compile(r'(\d+)_attn')
    
    for item in base_path.iterdir():
        if item.is_dir():
            match = attn_pattern.match(item.name)
            if match:
                image_id = match.group(1)
                image_ids.append(image_id)
    
    return sorted(image_ids)

def find_person_dirs(base_dir, target_image_id):
    """Find all person directories for the specified image ID."""
    if os.name != 'nt' and '\\' in base_dir:
        base_dir = windows_to_wsl_path(base_dir)
    
    base_path = Path(base_dir)
    person_dirs = []
    
    # Look for the specific image ID directory
    attn_dir = base_path / f"{target_image_id}_attn"
    if attn_dir.exists():
        person_maps_path = attn_dir / "layer_23" / "each_person_attn_maps"
        if person_maps_path.exists():
            # Find all person_X directories
            for person_dir in person_maps_path.iterdir():
                if person_dir.is_dir() and person_dir.name.startswith("person_"):
                    person_dirs.append({
                        "image_id": target_image_id,
                        "person_dir": person_dir,
                        "attn_dir": attn_dir,
                        "layer_dir": attn_dir / "layer_23"
                    })
    
    return person_dirs

def find_image_file(image_id, images_base_dir):
    """Find the original image file for the given image ID."""
    if os.name != 'nt' and '\\' in images_base_dir:
        images_base_dir = windows_to_wsl_path(images_base_dir)
    
    base_path = Path(images_base_dir)
    
    # First look for a direct match with the image ID
    direct_match = list(base_path.glob(f"*/{image_id}.jpg"))
    if direct_match:
        return direct_match[0]
    
    # If not found, try to find the image in a subdirectory matching the image ID
    id_subdir = base_path / image_id
    if id_subdir.exists() and id_subdir.is_dir():
        # Look for images in this directory
        image_files = list(id_subdir.glob("*.jpg"))
        if image_files:
            # Return the matching ID file if exists, otherwise the first one
            for img in image_files:
                if img.stem == image_id:
                    return img
            # If no exact match, return the first image
            return image_files[0]
    
    # Try to find by searching all subdirectories (could be slow for large directories)
    all_matches = list(base_path.glob(f"**/{image_id}.jpg"))
    if all_matches:
        return all_matches[0]
    
    return None

def process_points_for_image(args, image_path, person_dirs, sam2_predictor, save_masks=True, save_format='json',
                            text_prompt=None, grounding_model=None, box_threshold=0.3, text_threshold=0.25):
    """Process all points for a given image."""
    image_results = {}
    
    # Process text prompts with Grounding DINO if available
    boxes = None
    box_confidences = None
    labels = None
    if text_prompt and grounding_model:
        from PIL import Image
        import numpy as np
        from grounding_dino.groundingdino.util.inference import predict
        
        # Load image for Grounding DINO
        image_pil = Image.open(image_path).convert("RGB")
        image_tens = torch.from_numpy(np.array(image_pil)).permute(2, 0, 1).float()
        
        # Get predictions from Grounding DINO
        print(f"Running Grounding DINO with prompt: '{text_prompt}'")
        try:
            boxes, box_confidences, labels = predict(
                model=grounding_model,
                image=image_tens,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )
        except:
            from grounding_dino.groundingdino.util.inference import load_model
            grounding_model = load_model(
                model_config_path=args.gdino_config,
                model_checkpoint_path=args.gdino_checkpoint,
                device=args.device
            )
            traceback.print_exc()
        
        if boxes is not None and len(boxes) > 0:
            print(f"Found {len(boxes)} objects matching '{text_prompt}'")
            # TODO: Process the detected objects with SAM2
            # For now, just store the detection results
            # image_results["text_prompt_detections"] = {
            #     "boxes": boxes.tolist() if isinstance(boxes, np.ndarray) else boxes,
            #     "scores": box_confidences.tolist() if isinstance(box_confidences, np.ndarray) else box_confidences,
            #     "phrases": labels
            # }
        else:
            print(f"No objects found matching '{text_prompt}'")
    
    for person_dir_info in person_dirs:
        person_dir = person_dir_info["person_dir"]
        layer_dir = person_dir_info["layer_dir"]
        person_id = person_dir.name
        
        # Create segmentation output directory in the original layer dir
        segmentation_dir = layer_dir / "segmentation_results"
        segmentation_dir.mkdir(exist_ok=True, parents=True)
        
        # Create person-specific output directory
        person_segmentation_dir = segmentation_dir / person_id
        person_segmentation_dir.mkdir(exist_ok=True, parents=True)
        
        # Process all point files in the person directory
        results = {}
        
        # Find person points file
        person_points_file = list(person_dir.glob(f"{person_id}_attn_map_smooth_centers.pt"))
        if person_points_file:
            try:
                person_points = load_points_from_file(person_points_file[0])
                if len(person_points) == 0:
                    print(f"No points found in {person_points_file[0]}")
                    continue
                # Run segmentation for person points
                result_path, person_results = segment_with_points(
                    image_path=str(image_path),
                    point_coords=person_points,
                    sam2_predictor=sam2_predictor,
                    output_dir=str(person_segmentation_dir),
                    prefix=f"{person_id}_person",
                    save_masks=save_masks,
                    boxes=boxes,
                    box_confidences=box_confidences,
                    box_labels=labels,
                )
                # Store only the essential information, not the full masks
                results["person"] = {
                    "scores": person_results["scores"].tolist(),
                    "boxes": person_results["boxes"].tolist() if isinstance(person_results["boxes"], np.ndarray) else person_results["boxes"],
                    # We don't store the masks to save space
                }
            except Exception as e:
               print(f"Error processing person points for {person_id}: {str(e)}")
               traceback.print_exc()
        
        # Find gaze target points file
        gaze_points_files = list(person_dir.glob("gaze_target_*_attn_map_smooth_centers.pt"))
        for gaze_file in gaze_points_files:
            gaze_id = gaze_file.stem.split("_attn_map")[0]
            try:
                gaze_points = load_points_from_file(gaze_file)
                # Run segmentation for gaze target points
                result_path, gaze_results = segment_with_points(
                    image_path=str(image_path),
                    point_coords=gaze_points,
                    sam2_predictor=sam2_predictor,
                    output_dir=str(person_segmentation_dir),
                    prefix=f"{person_id}_{gaze_id}",
                    save_masks=save_masks
                )
                if "gaze_targets" not in results:
                    results["gaze_targets"] = {}
                # Store only the essential information, not the full masks
                results["gaze_targets"][gaze_id] = {
                    "scores": gaze_results["scores"].tolist(),
                    "boxes": gaze_results["boxes"].tolist() if isinstance(gaze_results["boxes"], np.ndarray) else gaze_results["boxes"],
                    # We don't store the masks to save space
                }
            except Exception as e:
                print(f"Error processing gaze points for {gaze_id}: {str(e)}")
        
        # Store results for this person
        if results:
            image_results[person_id] = results
            
            # Save results for this specific person and image
            image_name = Path(image_path).stem
            results_path = person_segmentation_dir / f"{image_name}_segmentation_results"
            save_results(results, results_path, save_format=save_format)
    
    return image_results

def save_results(results, output_path, save_format='json'):
    """Save the results dictionary to a file in the specified format."""
    if save_format == 'json':
        # Save as JSON format (more portable but larger)
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(results, f)
    else:
        # Save as NPY format (smaller but less portable)
        np.save(output_path.with_suffix('.npy'), results)

def process_image_id(args, image_id, sam2_predictor=None, grounding_model=None):
    """Process a single image ID."""
    # Load SAM2 model for this process if not provided
    if sam2_predictor is None:
        sam2_predictor = load_sam2_model(args.checkpoint, args.config, args.device)
    
    # Load Grounding DINO model if running in parallel and text prompt is specified
    if args.text_prompt and grounding_model is None:
        from grounding_dino.groundingdino.util.inference import load_model
        print(f"Loading Grounding DINO model for image ID {image_id} (parallel worker)")
        grounding_model = load_model(
            model_config_path=args.gdino_config,
            model_checkpoint_path=args.gdino_checkpoint,
            device=args.device
        )
    
    # Find person directories for the specified image ID
    print(f"Finding person directories for image ID {image_id}")
    person_dirs = find_person_dirs(args.attn_base_dir, image_id)
    print(f"Found {len(person_dirs)} person directories for {image_id}")
    
    if not person_dirs:
        print(f"No person directories found for image ID {image_id}")
        return {
            "image_id": image_id,
            "status": "skipped",
            "reason": "no_person_dirs"
        }
    
    # Find image file for this ID
    image_file = find_image_file(image_id, args.images_base_dir)
    
    if not image_file:
        print(f"Warning: No image file found for ID {image_id}")
        return {
            "image_id": image_id,
            "status": "skipped",
            "reason": "no_image_file"
        }
    
    print(f"Found image file: {image_file}")
    
    # Process the image
    print(f"Processing image: {image_file.stem}")
    
    # Process points for this image
    image_results = process_points_for_image(
        args=args,
        image_path=image_file,
        person_dirs=person_dirs,
        sam2_predictor=sam2_predictor,
        save_masks=args.save_masks,
        save_format=args.save_format,
        text_prompt=args.text_prompt,
        grounding_model=grounding_model,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold
    )
    
    # Save aggregated results for all persons in this image
    if image_results:
        # Save in layer directory
        layer_dir = person_dirs[0]["layer_dir"]
        segmentation_dir = layer_dir / "segmentation_results"
        segmentation_dir.mkdir(exist_ok=True, parents=True)
        
        all_results_path = segmentation_dir / f"{image_file.stem}_all_segmentation_results"
        save_results(image_results, all_results_path, save_format=args.save_format)
    
    # Save a summary of processed files
    summary = {
        "image_id": image_id,
        "processed_file": str(image_file),
        "processed_file_stem": image_file.stem
    }
    
    # Save summary in the layer directory
    layer_dir = person_dirs[0]["layer_dir"]
    segmentation_dir = layer_dir / "segmentation_results"
    with open(segmentation_dir / "processing_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    return {
        "image_id": image_id,
        "status": "completed",
        "processed_file": str(image_file)
    }

def main():
    parser = argparse.ArgumentParser(description="Auto-detect and process all attention directories")
    parser.add_argument("--attn-base-dir", required=True, 
                        help="Base directory containing attention directories")
    parser.add_argument("--images-base-dir", required=True,
                        help="Base directory containing images")
    parser.add_argument("--checkpoint", default="./checkpoints/sam2.1_hiera_large.pt", 
                        help="Path to SAM2 checkpoint")
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_l.yaml",
                        help="Path to SAM2 model config")
    parser.add_argument("--save-masks", action="store_true", 
                        help="Save individual masks as separate files")
    parser.add_argument("--save-format", choices=['json', 'npy'], default='json',
                        help="Format to save results (json or npy)")
    parser.add_argument("--device", default="cpu",
                        help="Device to run the model on")
    parser.add_argument("--test-ids", type=str, default=None,
                        help="Comma-separated list of specific image IDs to process (for testing)")
    parser.add_argument("--parallel", action="store_true",
                        help="Process image IDs in parallel (one process per ID)")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Maximum number of parallel workers when using --parallel")
    parser.add_argument("--text-prompt", type=str, default=None,
                        help="Text prompt for Grounding DINO model to identify objects")
    parser.add_argument("--gdino-checkpoint", default="./gdino_checkpoints/groundingdino_swint_ogc.pth",
                        help="Path to Grounding DINO checkpoint")
    parser.add_argument("--gdino-config", default="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                        help="Path to Grounding DINO model config")
    parser.add_argument("--box-threshold", type=float, default=0.3,
                        help="Box threshold for Grounding DINO predictions")
    parser.add_argument("--text-threshold", type=float, default=0.25,
                        help="Text threshold for Grounding DINO predictions")
    args = parser.parse_args()
    
    # Convert Windows paths if running in WSL
    if 'wsl' in os.uname().release.lower():
        args.attn_base_dir = windows_to_wsl_path(args.attn_base_dir)
        args.images_base_dir = windows_to_wsl_path(args.images_base_dir)
        args.checkpoint = windows_to_wsl_path(args.checkpoint)
        args.config = windows_to_wsl_path(args.config)
    
    start_time = time.time()
    
    # Determine which image IDs to process
    if args.test_ids:
        image_ids = args.test_ids.split(',')
        print(f"Processing {len(image_ids)} specified test image IDs")
    else:
        # Auto-detect all image IDs from directory names
        image_ids = detect_image_ids(args.attn_base_dir)
        print(f"Auto-detected {len(image_ids)} image IDs to process")
    
    if not image_ids:
        print(f"No image IDs detected in {args.attn_base_dir}")
        return

    # Load SAM2 model for sequential processing
    print(f"Loading SAM2 model from {args.checkpoint}")
    sam2_predictor = load_sam2_model(args.checkpoint, args.config, args.device)
    print(f"SAM2 model loaded successfully")

    # Log text prompt configuration if provided
    if args.text_prompt:
        print(f"Using text prompt: '{args.text_prompt}'")
        print(f"Grounding DINO settings - Box threshold: {args.box_threshold}, Text threshold: {args.text_threshold}")
        
        # Load Grounding DINO model
        from grounding_dino.groundingdino.util.inference import load_model, predict
        print(f"Loading Grounding DINO model from {args.gdino_checkpoint}")
        grounding_model = load_model(
            model_config_path=args.gdino_config,
            model_checkpoint_path=args.gdino_checkpoint,
            device=args.device
        )
        print(f"Grounding DINO model loaded successfully")
    else:
        grounding_model = None

    # Process image IDs
    results = []
    
    if args.parallel and len(image_ids) > 1:
        print(f"Processing {len(image_ids)} image IDs in parallel with {args.max_workers} workers")
        
        # When running in parallel, we need a special handling for the models
        if args.text_prompt:
            print(f"WARNING: In parallel mode, the text prompt feature will attempt to load the")
            print(f"Grounding DINO model in each worker process. This may increase memory usage.")
            print(f"If you experience out-of-memory errors, try running sequentially instead.")
        
        print(f"NOTE: In parallel mode, the SAM2 model will be loaded in each worker process.")
        
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_id = {executor.submit(process_image_id, args, image_id): image_id for image_id in image_ids}
            for future in tqdm(future_to_id, desc="Processing image IDs", total=len(image_ids)):
                image_id = future_to_id[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"Completed processing for image ID {image_id}")
                except Exception as e:
                    print(f"Error processing image ID {image_id}: {str(e)}")
                    traceback.print_exc()
                    results.append({
                        "image_id": image_id,
                        "status": "error",
                        "error": str(e)
                    })
    else:
        print(f"Processing {len(image_ids)} image IDs sequentially")
        for image_id in tqdm(image_ids, desc="Processing image IDs"):
            try:
                result = process_image_id(args, image_id, sam2_predictor, grounding_model)
                results.append(result)
                print(f"Completed processing for image ID {image_id}")
            except Exception as e:
                print(f"Error processing image ID {image_id}: {str(e)}")
                traceback.print_exc()
                results.append({
                    "image_id": image_id,
                    "status": "error",
                    "error": str(e)
                })
    
    # Save overall processing summary
    end_time = time.time()
    processing_time = end_time - start_time
    
    summary = {
        "total_image_ids": len(image_ids),
        "completed": sum(1 for r in results if r["status"] == "completed"),
        "skipped": sum(1 for r in results if r["status"] == "skipped"),
        "errors": sum(1 for r in results if r["status"] == "error"),
        "processing_time_seconds": processing_time,
        "processing_time_formatted": f"{processing_time // 3600}h {(processing_time % 3600) // 60}m {int(processing_time % 60)}s",
        "results": results
    }
    
    # Save summary in the base directory
    base_dir = Path(args.attn_base_dir)
    overall_summary_path = base_dir / "segmentation_processing_summary.json"
    with open(overall_summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Processing completed in {summary['processing_time_formatted']}")
    print(f"Completed: {summary['completed']}, Skipped: {summary['skipped']}, Errors: {summary['errors']}")
    print(f"Overall summary saved to {overall_summary_path}")

if __name__ == "__main__":
    main() 