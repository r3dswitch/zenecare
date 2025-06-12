import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def get_wound_spline(config, base_image, mask):
    """
    Generates a spline from a binary mask and saves it superimposed on an optional base image.

    Parameters:
    - mask: Binary mask (2D numpy array).
    - output_file: File path to save the output image (e.g., "spline.png").
    - base_image: Optional RGB or grayscale image (numpy array) to draw the spline on.
    """
    base_image = np.array(base_image)
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
    else:
        # Default to grayscale mask background
        background = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Step 5: Plot and save
    plt.figure(figsize=(6, 6))
    plt.imshow(background)
    plt.plot(x_spline, y_spline, 'r-', linewidth=2)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(config['paths']['edge_path'], bbox_inches='tight', pad_inches=0)

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

