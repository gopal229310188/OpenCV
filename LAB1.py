import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

# 1. Reading, displaying, and writing an image using OpenCV
def basic_image_operations():
    image = cv2.imread("SampleImage.jpeg")  # Replace "SampleImage.jpg" with your image file
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)  # Wait indefinitely for a key press
    cv2.destroyAllWindows()

    # Writing the image to disk in PNG format
    cv2.imwrite("output_image.png", image)
    print("Image saved as output_image.png")

# 2. Convert the image to another format using OpenCV
def convert_image_format():
    image = cv2.imread("SampleImage.jpeg")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale Image", gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("gray_image.jpg", gray_image)
    print("Converted image saved as gray_image.jpg")

# 3. Perform image resizing using OpenCV
def resize_image():
    image = cv2.imread("SampleImage.jpeg")
    resized_image = cv2.resize(image, (300, 300))  # Resize to 300x300 pixels
    cv2.imshow("Resized Image", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("resized_image.jpg", resized_image)
    print("Resized image saved as resized_image.jpg")

# 4. Convert a colored image into a grayscale image using OpenCV
def convert_to_grayscale():
    image = cv2.imread("SampleImage.jpeg")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale Image", gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("grayscale_image.jpg", gray_image)
    print("Grayscale image saved as grayscale_image.jpg")

# 5. Scaling, rotation, and shifting operation on the image
def transform_image():
    image = cv2.imread("SampleImage.jpeg")
    rows, cols = image.shape[:2]

    # Scaling
    scaled_image = cv2.resize(image, None, fx=1.5, fy=1.5)
    cv2.imshow("Scaled Image", scaled_image)

    # Rotation
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)  # Rotate by 45 degrees
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    cv2.imshow("Rotated Image", rotated_image)

    # Shifting
    translation_matrix = np.float32([[1, 0, 50], [0, 1, 50]])  # Shift right by 50 pixels and down by 50 pixels
    shifted_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
    cv2.imshow("Shifted Image", shifted_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 6. Play a video using OpenCV
def play_video():
    cap = cv2.VideoCapture("test_video.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Video", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 7. Draw an image using the formula f(i, j) = sin(2πf(i + j))
def generate_image_with_frequency():
    rows, cols = 300, 300  # Image dimensions
    frequency = float(input("Enter the frequency for the sinusoidal function: "))
    image = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            image[i, j] = int(127.5 * (1 + math.sin(2 * math.pi * frequency * (i + j))))

    cv2.imshow("Generated Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("generated_image.jpg", image)
    print("Generated image saved as generated_image.jpg")

def function_selector():
    print("\nSelect a function to execute:")
    print("1. Basic Image Operations")
    print("2. Convert Image Format")
    print("3. Resize Image")
    print("4. Convert Image to Grayscale")
    print("5. Transform Image (Scaling, Rotation, Shifting)")
    print("6. Play Video")
    print("7. Generate Image with Frequency")

    choice = input("Enter the number corresponding to your choice: ")

    if choice == "1":
        basic_image_operations()
    elif choice == "2":
        convert_image_format()
    elif choice == "3":
        resize_image()
    elif choice == "4":
        convert_to_grayscale()
    elif choice == "5":
        transform_image()
    elif choice == "6":
        play_video()
    elif choice == "7":
        generate_image_with_frequency()
    else:
        print("Invalid choice. Please try again.")

if __name__ == "__main__":
    function_selector()
