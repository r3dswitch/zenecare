from PIL import Image
from helpers import get_hf_model, save_image
import numpy as np

from utils.get_wound_bbox import get_wound_bbox_vqa
from utils.get_segmentation_mask import get_segmentation_mask_sam
from utils.get_wound_edge import get_wound_edge

def segment_wound_edge(config: dict):
    """
    Description: Task to extract wound edges and return final image with contours
    Input: 
        Config: Dict
    Output: 
        Edges Image: Numpy Array
    """
    image_path = config['paths']['input_path']
    image = np.array(Image.open(image_path))
    wound_bbox = get_wound_bbox_vqa(config, image)
    mask = get_segmentation_mask_sam(config, image, wound_bbox)
    edges = get_wound_edge(config, mask)
    return edges