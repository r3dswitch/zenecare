from PIL import Image
from utils.detect_wound_bbox import detect_wound_bbox
from helpers import load_config, get_hf_model, visualise
from utils.get_segmentation_mask import get_segmentation_mask

config = load_config("config.yaml")
    
def segment_wound_area():
    image_path = config['paths']['input_path']
    save_path = config['paths']['output_path']

    image = Image.open(image_path)

    processor, detector = get_hf_model(config, "bbox_detection")
    wound_bbox = detect_wound_bbox(config, processor, detector, image_path)
    mask = get_segmentation_mask(config, wound_bbox)
    
    visualise(config, image, wound_bbox, mask, save_path)
    
    return mask