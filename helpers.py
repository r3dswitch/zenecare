import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple
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

def visualise_bbox(config: dict, image: np.ndarray, wound_bbox: Tuple[int, int, int, int]):
        height, width = image.shape[:2]
        dpi = 100
        plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        plt.axis('off')
        plt.imshow(image)
        plt.gca().add_patch(plt.Rectangle(
            (wound_bbox[0], wound_bbox[1]), 
            wound_bbox[2]-wound_bbox[0], 
            wound_bbox[3]-wound_bbox[1],
            fill=False, color='red', linewidth=2
        ))
        plt.savefig(config['paths']['bbox_path'], bbox_inches='tight', pad_inches=0)

def visualise_area(config: dict, image: np.ndarray, mask: np.ndarray):
        height, width = image.shape[:2]
        dpi = 100
        plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        plt.axis('off')
        plt.imshow(image)
        plt.imshow(mask, cmap='gray')
        plt.savefig(config['paths']['area_path'], bbox_inches='tight', pad_inches=0)

def visualise_edges(config: dict, image: np.ndarray, edges: np.ndarray):
        height, width = image.shape[:2]
        dpi = 100
        plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        plt.axis('off')
        plt.imshow(image)
        plt.imshow(edges)
        plt.savefig(config['paths']['edge_path'], bbox_inches='tight', pad_inches=0)

def save_image(image: np.ndarray, save_path: str):
    image = Image.fromarray(image)
    image.save(save_path)