import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def get_wound_edge(config: dict, mask: np.ndarray):
    """
    Description: Find wound contours from a segmentation mask, add it to a base image and return the base image with edges
    Input: 
        Config: Dict, 
        Mask: Numpy Array
    Output: 
        Image with Edges: Numpy Array
    """
    image_path = config['paths']['input_path']
    base_image = np.array(Image.open(image_path))
    
    # Step 1: Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contours found in mask")

    # Step 2: Get the largest contour
    contour = max(contours, key=cv2.contourArea).squeeze()
    if contour.ndim != 2:
        raise ValueError("Contour not in expected shape")

    # Close the contour
    contour = np.vstack([contour, contour[0]])

    # Step 3: Fit a spline
    tck, u = splprep([contour[:, 0], contour[:, 1]], s=5.0, per=True)
    x_spline, y_spline = splev(np.linspace(0, 1, 1000), tck)

    # Step 4: Prepare the background
    if base_image is not None:
        if len(base_image.shape) == 2:  # grayscale
            background = cv2.cvtColor(base_image, cv2.COLOR_GRAY2RGB)
        else:
            background = base_image.copy()

    # Step 5: Plot and save
    plt.figure(figsize=(6, 6))
    plt.imshow(background)
    plt.plot(x_spline, y_spline, 'r-', linewidth=2)
    plt.axis('off')
    plt.tight_layout()

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    canvas = FigureCanvas(fig)

    ax.imshow(background)
    ax.plot(x_spline, y_spline, 'r-', linewidth=2)
    ax.axis('off')
    fig.tight_layout(pad=0)

    # Draw the canvas and extract the RGB image
    canvas.draw()
    width, height = canvas.get_width_height()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)

    plt.close(fig)
    return img

