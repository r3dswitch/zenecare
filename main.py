import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import SamModel, SamProcessor, Owlv2Processor, Owlv2ForObjectDetection
import cv2

class WoundSegmentationSAM:
    def __init__(self, model_name="facebook/sam-vit-base"):
        self.device = torch.device("cpu")
        self.model = SamModel.from_pretrained(model_name).to(self.device)
        self.processor = SamProcessor.from_pretrained(model_name)
        
    def segment_with_bbox(self, bbox, image_path="/teamspace/studios/this_studio/data/images/0011.png"):
        image = Image.open(image_path).convert("RGB")
        
        inputs = self.processor(
            image,
            input_boxes=[[list(map(float, bbox))]],
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )
        
        return masks[0][0][0].numpy()
    
def segment_wound_with_bbox_example():
    segmenter = WoundSegmentationSAM()
    
    # Bounding box around wound [x1, y1, x2, y2]
    image_path = "/teamspace/studios/this_studio/data/images/0011.png"
    wound_bbox = detect_wound_bbox(image_path)
    mask = segmenter.segment_with_bbox(wound_bbox, image_path)
    
    image = Image.open(image_path)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original with Bbox')
    
    plt.gca().add_patch(plt.Rectangle(
        (wound_bbox[0], wound_bbox[1]), 
        wound_bbox[2]-wound_bbox[0], 
        wound_bbox[3]-wound_bbox[1],
        fill=False, color='red', linewidth=2
    ))
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Wound Segmentation')
    
    plt.show()
    plt.savefig("out.png")
    
    return mask

def detect_wound_bbox(image_path: str, threshold=0.2):
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
    image = Image.open(image_path).convert("RGB")

    # Query for zero-shot detection
    texts = [["a wound"]]  # Can expand this with more prompts if needed
    inputs = processor(text=texts, images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process the output
    results = processor.post_process_grounded_object_detection(
        outputs=outputs, threshold=threshold, target_sizes=[image.size]
    )[0]

    # Return first wound bbox if any
    for box in results["boxes"]:
        if box is not None:
            print(box)
            return tuple(map(int, box.tolist()))
    return None

# Interactive usage
if __name__ == "__main__":
    # Initialize segmenter
    segmenter = WoundSegmentationSAM()
    mask = segment_wound_with_bbox_example()