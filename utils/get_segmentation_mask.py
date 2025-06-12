import torch
import numpy as np
from PIL import Image
from typing import Tuple
from transformers import ProcessorMixin, PreTrainedModel

def get_segmentation_mask_sam(config: dict, image: np.ndarray, bbox: Tuple[int, int, int, int], processor: ProcessorMixin, model: PreTrainedModel):
    """
    Description: Takes an image and a bounding box and finds wound segmentation mask using open vocabulary segmentation
    Input: 
        Config: dict, 
        Image: Numpy Array, 
        Bounding Box: List[x1, y1, x2, y2]
        Processor: HF Transformers Processor
        Model: HF Transformers Pretrained Model
    Output: 
        Mask: Numpy Array
    """
    image = Image.fromarray(np.uint8(image))
    
    inputs = processor(
        image,
        input_boxes=[[list(map(float, bbox))]],
        return_tensors="pt"
    ).to(config['device'])
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )

    return masks[0][0][0].numpy()