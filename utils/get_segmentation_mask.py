from helpers import load_config, get_hf_model
import torch
from PIL import Image

def get_segmentation_mask(config: dict, bbox):
    processor, model = get_hf_model(config, "segmentation")
    device = torch.device(config['device'])
    image_path = config['paths']['input_path']
    image = Image.open(image_path).convert("RGB")
    
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