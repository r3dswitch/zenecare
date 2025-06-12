from PIL import Image
from helpers import get_hf_model

from utils.detect_wound_bbox import detect_wound_bbox
from utils.get_segmentation_mask import get_segmentation_mask
from utils.get_wound_spline import get_wound_spline
    
def segment_wound_edge(config):
    image_path = config['paths']['input_path']
    image = Image.open(image_path)
    processor, detector = get_hf_model(config, "bbox_detection")
    wound_bbox = detect_wound_bbox(config, processor, detector, image)
    mask = get_segmentation_mask(config, wound_bbox)
    edges = get_wound_spline(config, image, mask)
    
    return edges