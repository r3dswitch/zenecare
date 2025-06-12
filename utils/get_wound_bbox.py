import torch
import numpy as np
from PIL import Image
from helpers import get_hf_model

def get_wound_bbox_vqa(config: dict, image: np.ndarray):
    """
    Description: Finds the bounding box for the wound if it exists using VQA model and then returns aabb without padding
    Input: 
        Config: dict,
        Image: Numpy Array
    Output: 
        Bounding Box: List[x1, y1, x2, y2]
    """
    print("------------------Getting Bounding Box-----------------------")
    image = Image.fromarray(np.uint8(image))
    prompts = config['tasks']['bbox_detection']['prompts']
    threshold = config['tasks']['bbox_detection']['threshold']
    processor, model = get_hf_model(config, "bbox_detection")
    inputs = processor(text=prompts, images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs=outputs, threshold=threshold, target_sizes=[image.size]
    )[0]

    for box in results["boxes"]:
        if box is not None:
            return tuple(map(int, box.tolist()))
    return None