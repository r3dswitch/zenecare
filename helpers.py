import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple
from transformers import SamModel, SamProcessor, Owlv2Processor, Owlv2ForObjectDetection
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def load_config(path: str) -> dict:
    print(f"------------------Loading Config: {path}-----------------------")
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_hf_model(config: dict, task_name: str):
    print(f"------------------Getting Model for: {task_name}-----------------------")
    processor_id = config['tasks'][task_name]['processor']
    model_id = config['tasks'][task_name]['model']
    
    if task_name == "bbox_detection":
        processor = Owlv2Processor.from_pretrained(processor_id)
        model = Owlv2ForObjectDetection.from_pretrained(model_id)   
    elif task_name == "segmentation":
        processor = SamProcessor.from_pretrained(processor_id)
        model = SamModel.from_pretrained(model_id)
    else:
        return None
    return processor, model

def visualise_bbox(config: dict, image: np.ndarray, wound_bbox: Tuple[int, int, int, int]):
        print("------------------Visualising Bbox-----------------------")
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
        print("------------------Visualising Area-----------------------")
        height, width = image.shape[:2]
        dpi = 100
        plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        plt.axis('off')
        plt.imshow(image)
        plt.imshow(mask, cmap='gray')
        plt.savefig(config['paths']['area_path'], bbox_inches='tight', pad_inches=0)

def visualise_edges(config: dict, image: np.ndarray, edges: np.ndarray):
        print("------------------Visualising Edges-----------------------")
        height, width = image.shape[:2]
        dpi = 100
        plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        plt.axis('off')
        plt.imshow(image)
        plt.imshow(edges)
        plt.savefig(config['paths']['edge_path'], bbox_inches='tight', pad_inches=0)

def compare_vis(config: dict, mask1: np.ndarray, mask2: np.ndarray):
    print("------------------Comparing Visualisations-----------------------")
    if mask1.shape != mask2.shape:
        raise ValueError("Both masks must have the same shape")

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(np.zeros_like(mask1), cmap='gray')  # background

    plt.imshow(mask1, cmap='Reds', alpha=0.5, interpolation='none')   # mask1 in red
    plt.imshow(mask2, cmap='Blues', alpha=0.5, interpolation='none')  # mask2 in blue

    plt.axis('off')
    plt.show()
    plt.savefig(config['paths']['compare_path'], bbox_inches='tight', pad_inches=0)
    
    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = canvas.get_width_height()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)

    return img

def save_image(image: np.ndarray, save_path: str):
    print("------------------Saving Image-----------------------")
    image = Image.fromarray(image)
    image.save(save_path)