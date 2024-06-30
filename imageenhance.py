import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def read_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_COLOR)

def save_image(image, output_path):
    cv2.imwrite(output_path, image)

def display_image(image, title="Image"):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def histogram_equalization(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(img_gray)
    return cv2.cvtColor(equ, cv2.COLOR_GRAY2BGR)

def clahe_equalization(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img_gray)
    return cv2.cvtColor(cl1, cv2.COLOR_GRAY2BGR)

def sharpen_image(image):
    kernel = np.array([[0, -1, 0], 
                       [-1, 5,-1], 
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def blur_image(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def edge_detection(image):
    edges = cv2.Canny(image, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def adjust_brightness_contrast(image, alpha=1.5, beta=50):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def denoise_image(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 30, 30, 7, 21)

def enhance_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def main():
    input_dir = "images/"
    output_dir = "results/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_path = os.path.join(input_dir, "sample.jpg")
    image = read_image(image_path)

    enhanced_images = {
        "histogram_equalization.jpg": histogram_equalization(image),
        "clahe_equalization.jpg": clahe_equalization(image),
        "sharpen.jpg": sharpen_image(image),
        "blur.jpg": blur_image(image),
        "edges.jpg": edge_detection(image),
        "brightness_contrast.jpg": adjust_brightness_contrast(image),
        "denoise.jpg": denoise_image(image),
        "color_enhance.jpg": enhance_color(image)
    }

    for filename, enhanced_image in enhanced_images.items():
        output_path = os.path.join(output_dir, filename)
        save_image(enhanced_image, output_path)
        display_image(enhanced_image, title=filename)

if __name__ == "__main__":
    main()
