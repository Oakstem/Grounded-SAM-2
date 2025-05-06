#!/usr/bin/env python
import os
import cv2
import torch
import numpy as np
import supervision as sv
from pathlib import Path
from sam2.build_sam import build_sam2
from torchvision.ops import box_convert
from sam2.sam2_image_predictor import SAM2ImagePredictor
from alons_utils import windows_to_wsl_path
import argparse


def load_sam2_model(checkpoint_path, model_config, device="cuda"):
    """Load SAM2 model from checkpoint."""
    sam2_model = build_sam2(model_config, checkpoint_path, device=device)
    return SAM2ImagePredictor(sam2_model)


def get_filename_with_prefix(image_path, prefix):
    """Generate output filename with prefix."""
    path = Path(image_path)
    return f"{prefix}_{path.name}"


def load_points_from_file(file_path):
    """Load point coordinates from a file (supports .pt files with 'centers' key)."""
    if file_path.endswith('.pt'):
        points_data = torch.load(file_path, weights_only=False)
        if isinstance(points_data, dict) and 'centers' in points_data:
            return points_data['centers']
        else:
            raise ValueError(f"The .pt file does not contain the 'centers' key")
    else:
        raise ValueError(f"Unsupported file format for points: {file_path}")


def segment_with_points(
    image_path, 
    point_coords, 
    sam2_predictor,
    output_dir="outputs", 
    prefix="sam2", 
    save_masks=False,
    text_prompt="person.",
    grounding_model=None,
    box_threshold=0.35,
    text_threshold=0.25,
    use_intersection=True,
    boxes=None,
    box_confidences=None,
    box_labels=None,
):
    """
    Perform point-based and/or text-based segmentation on an image using SAM2.
    
    Args:
        image_path: Path to the input image
        point_coords: Numpy array of point coordinates [[x1, y1], [x2, y2], ...]
        sam2_predictor: SAM2 image predictor instance
        output_dir: Directory to save outputs
        prefix: Prefix for output filenames
        save_masks: Whether to save masks as .npy files
        text_prompt: Optional text description for object detection
        grounding_model: Optional Grounding DINO model for text-based detection
        box_threshold: Confidence threshold for box detections
        text_threshold: Confidence threshold for text predictions
        use_intersection: If True, use only the intersection of point and text masks
        
    Returns:
        output_path: Path to the saved visualization
    """
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and process image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Get image dimensions
    h, w, _ = image.shape
    
    # Set image for SAM2 predictor
    sam2_predictor.set_image(image)
    
    # Variables to store detection results
    # boxes = None
    # box_confidences = None
    # box_labels = None
    text_masks = None
    
    # Perform text-based detection if requested
    if text_prompt and grounding_model:
        from grounding_dino.groundingdino.util.inference import load_image, predict
        from torchvision.ops import box_convert
        
        # Load image for Grounding DINO
        _, dino_image = load_image(image_path)
        
        print(f"Running detection with prompt: '{text_prompt}'")
        # Run detection
        boxes, box_confidences, box_labels = predict(
            model=grounding_model,
            image=dino_image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        
        print(f"Found {len(boxes)} boxes with labels: {box_labels}")

    # Prepare point labels (assuming all points are foreground)
    point_labels = np.ones(len(point_coords))

    # Process boxes for SAM2
    if boxes is not None and len(boxes) > 0:
        from torchvision.ops import box_convert
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        # Get text-based masks
        text_masks, text_scores, text_logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
    else:
        input_boxes = None


        # # Adjust dimensions if needed
        # if text_masks.ndim == 4:
        #     text_masks = text_masks.squeeze(1)
    

    
    # Get point-based segmentation masks
    point_masks, point_scores, point_logits = sam2_predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=None,
        multimask_output=False,
    )
    
    # Convert masks to the right format
    if point_masks.ndim == 4:
        point_masks = point_masks.squeeze(1)
    
    # Determine which masks to use for final output
    if text_masks is not None and use_intersection:
        # Compute intersection of text and point masks
        final_masks = np.zeros_like(point_masks)
        intersection_found = False
        
        for i in range(len(point_masks)):
            for j in range(len(text_masks)):
                intersection = np.logical_and(point_masks[i], text_masks[j])
                if np.any(intersection):  # Only add if intersection is not empty
                    # final_masks[i] = np.logical_or(final_masks[i], intersection)
                    final_masks[i] = np.logical_or(final_masks[i], text_masks[j])
                    intersection_found = True
        
        # Check if any intersection was found
        if not intersection_found:
            print("No intersection found between point and text masks. Using point masks as fallback.")
            final_masks = point_masks
        else:
            print(f"Found intersection between point and text masks")
        
        # Use scores from point-based segmentation
        final_scores = point_scores
    else:
        # Use only point-based masks
        final_masks = point_masks
        final_scores = point_scores

    # get bounding boxes
    pseudo_boxes = get_bbs(final_masks)

    results_dd = {
        "masks": final_masks,
        "scores": final_scores,
        "boxes": pseudo_boxes,
    }

    # Save masks if requested
    if save_masks:
        image_name = Path(image_path).stem
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(exist_ok=True, parents=True)

        # Save masks in numpy format
        mask_file = masks_dir / f"{prefix}_{image_name}_masks.npy"
        np.save(mask_file, results_dd)
        
        # Save scores (confidence values)
        scores_file = masks_dir / f"{prefix}_{image_name}_scores.npy"
        np.save(scores_file, final_scores)
        
        print(f"Saved masks to {mask_file} and scores to {scores_file}")
    
    # Create a copy of the original image for visualization
    annotated_frame = image.copy()
    
    # Add point visualization
    for x, y in point_coords:
        cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 0, 255), -1)  # Red circle for points
    
    # Add bounding box visualization if available
    if boxes is not None and len(boxes) > 0:
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        
        # Create detections for boxes
        box_detections = sv.Detections(
            xyxy=input_boxes,
            class_id=np.arange(len(box_labels)),
            confidence=box_confidences.numpy() if isinstance(box_confidences, torch.Tensor) else box_confidences
        )
        
        # Draw boxes
        box_annotator = sv.BoxAnnotator(color=sv.Color.GREEN)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=box_detections)
        
        # Add labels
        if box_labels:
            labels = [
                f"{label} {conf:.2f}" 
                for label, conf in zip(box_labels, box_confidences.numpy() if isinstance(box_confidences, torch.Tensor) else box_confidences)
            ]
            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=box_detections, labels=labels)
    
    # Add mask visualization
    if len(final_masks) > 0:
        # Create dummy boxes based on mask dimensions for supervision visualization
        h, w = final_masks.shape[1:]
        # dummy_boxes = np.array([[0, 0, w, h]] * len(final_masks))
        
    mask_detections = sv.Detections(
        xyxy=pseudo_boxes,
        mask=final_masks.astype(bool),
    )

    # Annotate pseudo_boxes as bounding boxes (in blue)
    pseudo_box_annotator = sv.BoxAnnotator(color=sv.Color.BLUE, color_lookup=sv.ColorLookup.INDEX)
    annotated_frame = pseudo_box_annotator.annotate(scene=annotated_frame, detections=mask_detections)

    mask_annotator = sv.MaskAnnotator(color=sv.Color.YELLOW, color_lookup=sv.ColorLookup.INDEX, opacity=0.5)
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=mask_detections)
    
    # Save the result
    output_path = output_dir / get_filename_with_prefix(image_path, prefix)
    cv2.imwrite(str(output_path), annotated_frame)
    print(f"Saved segmentation result to {output_path}")
    
    return output_path, results_dd


def get_bbs(final_masks):
    bounding_boxes = []
    if len(final_masks) > 0:
        # Create dummy boxes based on mask dimensions for supervision visualization
        h, w = final_masks.shape[1:]
        for mask in final_masks:
            # Find nonzero mask area and get bounding box [x_min, y_min, x_max, y_max]
            ys, xs = np.where(mask)
            if len(xs) > 0 and len(ys) > 0:
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                bounding_boxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])
            else:
                bounding_boxes.append([0, 0, w, h])
        dummy_boxes = np.array(bounding_boxes)
    return dummy_boxes


def main():
    parser = argparse.ArgumentParser(description="SAM2 point-based segmentation")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--points", help="Point coordinates in format 'x1,y1;x2,y2;...'")
    parser.add_argument("--points-file", help="Path to file containing point coordinates")
    parser.add_argument("--prefix", default="sam2", help="Prefix for output filename")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save output")
    parser.add_argument("--checkpoint", default="./checkpoints/sam2.1_hiera_large.pt", 
                        help="Path to SAM2 checkpoint")
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_l.yaml",
                        help="Path to SAM2 model config")
    parser.add_argument("--save-masks", action="store_true", 
                        help="Save masks as separate .npy files")
    # Add new arguments for text-based detection
    parser.add_argument("--text-prompt", help="Text prompt for Grounding DINO detection")
    parser.add_argument("--gdino-checkpoint", 
                        default="gdino_checkpoints/groundingdino_swint_ogc.pth",
                        help="Path to Grounding DINO checkpoint")
    parser.add_argument("--gdino-config", 
                        default="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                        help="Path to Grounding DINO config")
    parser.add_argument("--box-threshold", type=float, default=0.35,
                        help="Confidence threshold for box detection")
    parser.add_argument("--text-threshold", type=float, default=0.25,
                        help="Confidence threshold for text detection")
    parser.add_argument("--use-intersection", action="store_true",
                        help="Use intersection of point and text masks")
    args = parser.parse_args()

    # fix wsl paths
    if 'wsl' in os.uname().release.lower():
        args.image = windows_to_wsl_path(args.image)
        args.points_file = windows_to_wsl_path(args.points_file) if args.points_file else None
        if args.gdino_checkpoint:
            args.gdino_checkpoint = windows_to_wsl_path(args.gdino_checkpoint)
        if args.gdino_config:
            args.gdino_config = windows_to_wsl_path(args.gdino_config)

    # Check that either points or points-file is provided if no text prompt
    if args.points is None and args.points_file is None and args.text_prompt is None:
        parser.error("Either --points, --points-file, or --text-prompt must be provided")
    
    # Parse point coordinates if provided
    point_coords = None
    if args.points_file:
        point_coords = load_points_from_file(args.points_file)
    elif args.points:
        # Parse point coordinates from input string
        point_coords = []
        for point_str in args.points.split(';'):
            x, y = map(int, point_str.split(','))
            point_coords.append([x, y])
        point_coords = np.array(point_coords)
    else:
        # If only text prompt is provided, use empty point coordinates
        point_coords = np.array([])
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load SAM2 model
    sam2_predictor = load_sam2_model(args.checkpoint, args.config, device)
    
    # Load Grounding DINO model if text prompt is provided
    grounding_model = None
    if args.text_prompt:
        from grounding_dino.groundingdino.util.inference import load_model
        print(f"Loading Grounding DINO model from {args.gdino_checkpoint}")
        grounding_model = load_model(
            model_config_path=args.gdino_config,
            model_checkpoint_path=args.gdino_checkpoint,
            device=device
        )
    
    # Perform segmentation
    result_path, results_dd = segment_with_points(
        image_path=args.image,
        point_coords=point_coords,
        sam2_predictor=sam2_predictor,
        output_dir=args.output_dir,
        prefix=args.prefix,
        save_masks=args.save_masks,
        text_prompt=args.text_prompt,
        grounding_model=grounding_model,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        use_intersection=args.use_intersection
    )


if __name__ == "__main__":
    main() 