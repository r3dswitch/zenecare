import torch
import numpy as np
from PIL import Image
from typing import Tuple
from helpers import get_hf_model

def get_segmentation_mask_sam(config: dict, image: np.ndarray, bbox: Tuple[int, int, int, int]):
    """
    Description: Takes an image and a bounding box and finds wound segmentation mask using open vocabulary segmentation
    Input: 
        Config: dict, 
        Image: Numpy Array, 
        Bounding Box: List[x1, y1, x2, y2]
    Output: 
        Mask: Numpy Array
    """
    print("------------------Getting Segmentation Map-----------------------")
    image = Image.fromarray(np.uint8(image))
    device = config['envs']['device']
    processor, model = get_hf_model(config, "segmentation")
    inputs = processor(
        image,
        input_boxes=[[list(map(float, bbox))]],
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )

    return masks[0][0][0].numpy()