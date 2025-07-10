import cv2
import numpy as np

# --- Configuration ---
# Your original input image (without the black border)
INPUT_IMAGE_PATH = 'usa.png'
# The desired output filename for your main application
OUTPUT_IMAGE_PATH = 'usa_with_border.jpg'
# How thick the black border should be
BORDER_THICKNESS = 15
# How many times to dilate the image to widen state borders.
# A value of 1 is usually sufficient.
DILATION_ITERATIONS = 1

# --- Script ---

# Load the original image from disk
image = cv2.imread(INPUT_IMAGE_PATH)

if image is None:
    print(f"Error: Could not load the image from '{INPUT_IMAGE_PATH}'. Check the path and file name.")
    exit()

# --- Step 1: Add the black border ---
print(f"Adding a {BORDER_THICKNESS}px black border...")
image_with_border = cv2.copyMakeBorder(
    image,
    top=BORDER_THICKNESS,
    bottom=BORDER_THICKNESS,
    left=BORDER_THICKNESS,
    right=BORDER_THICKNESS,
    borderType=cv2.BORDER_CONSTANT,
    value=[0, 0, 0]  # Black color
)
print("Border added.")

# --- Step 2: Apply Dilation to widen the state lines ---
print(f"Applying dilation with {DILATION_ITERATIONS} iteration(s)...")
# A 3x3 kernel is standard for this operation
kernel = np.ones((3, 3), np.uint8)
# Dilate the image with border. This expands the white lines.
final_processed_image = cv2.dilate(image_with_border, kernel, iterations=DILATION_ITERATIONS)
print("Dilation complete.")

# --- Display for verification ---
# You can uncomment these lines if you want to see the results before saving
# cv2.imshow("Original Image", image)
# cv2.imshow("With Border (Before Dilation)", image_with_border)
# cv2.imshow("Final Processed Image (With Dilation)", final_processed_image)
# print("\nDisplaying images. Press any key to save and close.")
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# --- Step 3: Save the final processed image ---
cv2.imwrite(OUTPUT_IMAGE_PATH, final_processed_image)
print(f"\nSuccess! The final processed image has been saved as '{OUTPUT_IMAGE_PATH}'")