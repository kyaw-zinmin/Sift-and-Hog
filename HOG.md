# Histogram of Oriented Gradients (HOG) Visualization

This project demonstrates the application of Histogram of Oriented Gradients (HOG) feature extraction using scikit-image. It supports processing a static sample image or real-time video from a webcam, visualizing the HOG output alongside the input.

## Features
- Computes HOG features on images with customizable parameters (orientations, pixels_per_cell, cells_per_block).
- Visualizes HOG for a sample image using Matplotlib.
- Applies HOG in real-time to webcam video frames using OpenCV for display.
- User-interactive menu to select between image or video mode.

## Requirements
- Python 3.x
- scikit-image (`skimage`)
- Matplotlib
- OpenCV (`opencv-python`)
- NumPy (implicitly used via skimage)

## Installation
Install the required packages via pip:

```
pip install scikit-image matplotlib opencv-python
```

## Usage
1. Run the script:

   ```
   python your_script_name.py
   ```

2. At the prompt, type '1' for static image processing or '2' for live video from webcam.

3. For image mode:
   - Displays a figure with the input sample image (astronaut) and its rescaled HOG visualization.

4. For video mode:
   - Opens webcam feed, applies HOG to each frame, and shows the HOG output in a window.
   - Press 'q' to exit.

### Example Output
- Image mode: Side-by-side Matplotlib plot of original and HOG image.
- Video mode: Real-time OpenCV window showing HOG-transformed frames.

## Code Structure
- `hog_img()`: Loads a sample image, computes HOG with visualization, and displays using Matplotlib.
- `hog_video()`: Captures webcam video, resizes frames, computes HOG, and displays in a loop until 'q' is pressed.
- `main()`: Entry point with user input to select and run either image or video mode.

## Customization
- Adjust HOG parameters (e.g., `orientations=8`, `pixels_per_cell=(16, 16)`, `cells_per_block=(1, 1)` for image or `(4, 4)` for video) to fine-tune feature extraction.
- In video mode, change frame resize dimensions (e.g., `(640, 480)`) for performance.
- Replace the sample image in `hog_img()` with your own via `cv2.imread` or other loaders.
- For better HOG visibility, modify the rescale intensity range in `exposure.rescale_intensity`.

## Limitations
- Video mode requires a webcam; no fallback for missing hardware.
- Performance in video mode may vary based on hardware—larger cells_per_block can speed it up but reduce detail.
- No error handling for invalid user input or package imports.
- Sample image is hardcoded from skimage.data; for custom images, modify the code accordingly.

## License
This project is licensed under the MIT License. Feel free to use and modify as needed.