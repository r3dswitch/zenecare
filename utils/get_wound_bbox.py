import torch
import numpy as np
from PIL import Image
from transformers import PreTrainedModel, ProcessorMixin

def get_wound_bbox_vqa(config: dict, processor: ProcessorMixin, model: PreTrainedModel, image: np.ndarray):
    """
    Description: Finds the bounding box for the wound if it exists using VQA model and then returns aabb without padding
    Input: 
        Config: dict, 
        Processor: HF Transformers Processor, 
        Model: HF Transformers Model, 
        Image: Numpy Array
    Output: 
        Bounding Box: List[x1, y1, x2, y2]
    """
    image = Image.fromarray(np.uint8(image))
    inputs = processor(text=config['prompts'], images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs=outputs, threshold=config['threshold'], target_sizes=[image.size]
    )[0]

    for box in results["boxes"]:
        if box is not None:
            return tuple(map(int, box.tolist()))
    return None