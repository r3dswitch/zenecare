import torch

def detect_wound_bbox(config: dict, processor, model, image):
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