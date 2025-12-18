# Custom SIFT Implementation with Harris Corners

This project implements a modified Scale-Invariant Feature Transform (SIFT) algorithm where Harris Corner Detection is used to identify keypoints instead of the standard Difference of Gaussians (DoG) approach. The SIFT descriptors are then computed at these Harris keypoints, and brute-force matching is performed between a reference and target image.

## Features
- Detects corners using Harris Corner Detection with customizable parameters.
- Converts Harris response matrix to SIFT keypoints.
- Computes SIFT descriptors at the detected keypoints.
- Matches descriptors using BFMatcher with L1 norm and cross-checking.
- Visualizes Harris corners, SIFT keypoints/descriptors, and matched features using OpenCV windows.

## Requirements
- Python 3.x
- OpenCV (`opencv-python`)
- NumPy

## Installation
Install the required packages via pip:

```
pip install opencv-python numpy
```

## Usage
1. Place your reference and target images in the `./my-sift-images/` directory as `reference.jpg` and `target.jpg`.
2. Run the script:

   ```
   python your_script_name.py
   ```

3. The script will display multiple windows:
   - Harris Corners on reference and target images (red circles for visibility).
   - SIFT descriptors visualized as larger circles on keypoints.
   - Matched features between the two images (top 50 matches shown).

4. Press 'q' to close the windows.

### Example Output
The script processes images and shows real-time visualizations. Adjust parameters like block size, ksize, or thresholds in the code for optimization.

## Code Structure
- `hr_corners(original_image, gray_image, get_dst=False)`: Computes Harris corners, draws larger red circles for visibility, and optionally returns the response matrix.
- `hr_dst_to_sift_kp(hr_dst)`: Converts Harris response to SIFT keypoints with adjustable size (e.g., 50 for larger descriptor circles).
- `sift_at_hr(original_image, gray_image, hr_kp, get_sift=False)`: Computes and draws SIFT descriptors at Harris keypoints.
- `main()`: Main function handling image loading, processing, matching, and visualization.

## Customization
- To make Harris points even larger, increase the `radius` in `cv2.circle` within `hr_corners`.
- For bigger descriptor circles, raise the `size` parameter in `cv2.KeyPoint` creation.
- Reduce computation time by lowering `blockSize` and `ksize` in `cv2.cornerHarris` (e.g., to 2 and 3).
- Adjust the threshold (e.g., `0.01 * dst.max()`) to control the number of detected corners.

## Limitations
- Performance may be slow on high-resolution images due to large Harris parameters; optimize as needed.
- Assumes images are in BGR format via `cv2.imread`.
- No error handling for missing images or invalid paths.

## License
This project is licensed under the MIT License. Feel free to use and modify as needed.