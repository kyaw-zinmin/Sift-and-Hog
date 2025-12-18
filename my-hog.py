import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
import cv2


def hog_img():
    image = data.astronaut()

    fd, hog_image = hog(
        image,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        visualize=True,
        channel_axis=-1
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Input image')
    plt.show()

def hog_video():

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        fd, hog_frame = hog(
            frame,
            orientations=8,
            pixels_per_cell=(16, 16),
            cells_per_block=(4, 4),
            visualize=True,
            channel_axis=-1
        )
        
        cv2.imshow("Live HOG", hog_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    choice = input("Type 1 for image\nType 2 for video\n")
    if choice == '1':
        hog_img()
    else:
        hog_video()

main()