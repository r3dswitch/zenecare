import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import SamModel, SamProcessor, Owlv2Processor, Owlv2ForObjectDetection

def load_config(path: str) -> dict:
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_hf_model(config: dict, task_name: str):
    if task_name == "bbox_detection":
        processor = Owlv2Processor.from_pretrained(config[task_name]['processor'])
        model = Owlv2ForObjectDetection.from_pretrained(config[task_name]['model'])
    elif task_name == "segmentation":
        processor = SamProcessor.from_pretrained(config[task_name]['processor'])
        model = SamModel.from_pretrained(config[task_name]['model']).to(torch.device(config['device']))
    else:
        return None
    return processor, model

def visualise(config: dict, image, wound_bbox, mask, save_path: str):
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
        
        plt.savefig(save_path)