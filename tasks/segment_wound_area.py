from PIL import Image
import numpy as np
from helpers import get_hf_model, save_image, compare_vis
import wandb
import glob
import os

from utils.get_wound_bbox import get_wound_bbox_vqa
from utils.get_segmentation_mask import get_segmentation_mask_sam

from metrics.hausdorff_distance import hausdorff_distance

def segment_wound_area(config: dict):
    """
    Description: Task to extract wound area as a bitmask
    Input: 
        Config: Dict
    Output: 
        Masked Wound Area: Numpy Array
    """
    print("------------------Segment Wound Task-----------------------")
    wandb.init(project=config['wandb']['project_id'])
    input_dir = config['paths']['input_directory']
    image_paths = glob.glob(os.path.join(input_dir, "*.*"))
    gt_dir = config['paths']['test_directory']
    gt_paths = glob.glob(os.path.join(gt_dir, "*.*"))
    if len(gt_paths) != len(image_paths):
        print(len(gt_paths), len(image_paths))
        return
    for image_path in image_paths:
        print(f"------------------Processing Image: {image_path}-----------------------")
        gt_path = gt_dir + image_path[-8:]
        image = np.array(Image.open(image_path))
        gt = np.array(Image.open(gt_path).convert("L"))
        wound_bbox = get_wound_bbox_vqa(config, image)
        mask = get_segmentation_mask_sam(config, image, wound_bbox)
        compare = compare_vis(config, gt, mask)
        wandb.log({
            "segmentation_mask": wandb.Image(mask, caption="Predicted Wound Mask"),
            "ground truth": wandb.Image(gt, caption="Ground Truth"),
            "comparision": wandb.Image(compare, caption="Comparision"),
            "hausdorff distance": hausdorff_distance(gt, mask)
        })
    wandb.finish()
    return mask