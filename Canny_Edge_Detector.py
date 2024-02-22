import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_blur(image, kernel_size):
  blur = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
  return blur

def sobel_filters(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return sobel_x, sobel_y

def gradient_magnitude(grad_x, grad_y):
    return np.sqrt(grad_x**2 + grad_y**2)

def gradient_direction(grad_x, grad_y):
    return np.arctan2(grad_y, grad_x)

def non_max_suppression(magnitude, direction):
    rows, cols = magnitude.shape
    suppressed = np.zeros((rows, cols), dtype=np.uint8)
    angle = direction * (180.0 / np.pi)
    angle[angle < 0] += 180  # Convert negative angles to positive

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            q, r = 255, 255  # Intensity values of neighboring pixels

            # Determine gradient directions
            if 0 <= angle[i, j] < 22.5 or 157.5 <= angle[i, j] <= 180:
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]

            # Perform non-maximal suppression
            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                suppressed[i, j] = magnitude[i, j]
            else:
                suppressed[i, j] = 0

    return suppressed

def Double_thresholding(image, low_threshold, high_threshold):
    # Find indices of pixels greater than the high threshold
    strong_i, strong_j = np.where(image >= high_threshold)

    # Find indices of pixels between low and high thresholds
    weak_i, weak_j = np.where((image <= high_threshold) & (image >= low_threshold))

    # Set strong edges to 255, weak edges to 50
    image[strong_i, strong_j] = 255
    image[weak_i, weak_j] = 50

    return image

def plot_histogram(image, title):
    plt.figure(figsize=(8, 6))
    plt.hist(image.ravel(), 256, [0, 256])
    plt.title(title)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()

def canny_edge_detection(image, kernel_size, low_threshold, high_threshold):
    # Step 1: Gaussian blur
    blurred = gaussian_blur(image, kernel_size)

    # Step 2: Sobel operators for gradient calculation
    grad_x, grad_y = sobel_filters(blurred)

    # Step 3: Gradient magnitude and direction
    grad_mag = gradient_magnitude(grad_x, grad_y)
    grad_dir = gradient_direction(grad_x, grad_y)

    # Step 4: Non-maximum suppression
    suppressed = non_max_suppression(grad_mag, grad_dir)

    # Step 5: Hysteresis thresholding
    edges = Double_thresholding(suppressed, low_threshold, high_threshold)

    return edges

# Read the input image
image = cv2.imread('house.jpg', cv2.IMREAD_GRAYSCALE)

# Define parameters
kernel_size = 5
low_threshold = 30
high_threshold = 100

# Apply Canny edge detection
edges = canny_edge_detection(image, kernel_size, low_threshold, high_threshold)

# Display the original and processed images
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.show()

# Plot input image and histogram
plot_histogram(image, 'Histogram: INput Image')

blurred = gaussian_blur(image, kernel_size)
plt.imshow(blurred, cmap='gray')
plt.title('Gaussian Blur :L=30 H=100')
plt.show()

 # Step 2: Sobel operators for gradient calculation
grad_x, grad_y = sobel_filters(blurred)


# Step 3: Gradient magnitude and direction
grad_mag = gradient_magnitude(grad_x, grad_y)
grad_dir = gradient_direction(grad_x, grad_y)
plt.imshow(grad_mag, cmap='gray')
plt.title('Gradient Magnitude : :L=30 H=100')
plt.show()

# Step 4: Non-maximum suppression
suppressed = non_max_suppression(grad_mag, grad_dir)
plt.imshow(suppressed, cmap='gray')
plt.title('Non-Maximum Suppression : :L=30 H=100')
plt.show()

# Plot processed image and histogram
plt.figure(figsize=(12, 6))
    

plt.imshow(edges, cmap='gray')
plt.title('Processed: Image (Canny) : :L=30 H=100')
plt.axis('off')
    

plot_histogram(edges, 'Histogram: processed Image (Canny)')

    # Step 5: double thresholding

plt.imshow(edges, cmap='gray')
plt.title('Final Double Thresholding : :L=30 H=100')
plt.axis('off')

plt.show()
