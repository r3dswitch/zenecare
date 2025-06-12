from PIL import Image
import numpy as np
from helpers import get_hf_model, save_image

from utils.get_wound_bbox import get_wound_bbox_vqa
from utils.get_segmentation_mask import get_segmentation_mask_sam

def segment_wound_area(config: dict):
    """
    Description: Task to extract wound area as a bitmask
    Input: 
        Config: Dict
    Output: 
        Masked Wound Area: Numpy Array
    """
    # wandb.init(project=config.get("wandb_project", "wound_segmentation"), config=config)
    image_path = config['paths']['input_path']
    image = np.array(Image.open(image_path))
    wound_bbox = get_wound_bbox_vqa(config, image)
    mask = get_segmentation_mask_sam(config, image, wound_bbox)
    return mask