# Required imports
import cv2
import os
import copy

# CONST
HAR_FACE_FILE_PATH = './haarcascades_files/haarcascade_frontalface_default.xml'
INPUT_DATA_PATH = './Images'


def detect_faces(input_image, face_classifier):
    try:
        # Convert input image into gray scale
        gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        # Find faces from image
        faces = face_classifier.detectMultiScale(gray_image, 1.3, 5)

        # Check if face list is empty or not
        if faces is ():
            return None

        # Iterate over each face from list and get bounding rect
        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(input_image, (x, y), (x+w, y+h), (255, 0, 255), 2)
        return input_image
    except Exception as identifier:
        print('[ ERROR ]', identifier)

# Entry point of code


def main():
    try:
        # Creating face classifier objet
        face_classifier = cv2.CascadeClassifier(HAR_FACE_FILE_PATH)

        # Iterating over all images from input dir
        for imageName in os.listdir(INPUT_DATA_PATH):
            # Creating input image file path
            input_image_path = os.path.join(INPUT_DATA_PATH, imageName)

            # Reading input image
            input_image = cv2.imread(input_image_path)

            # Detecting faces
            output_image = detect_faces(input_image, face_classifier)

            # If input image has faces display output
            if output_image is not None:
                cv2.imshow('output', output_image)
                cv2.waitKey(0)

        # Close all windows
        cv2.destroyAllWindows()
    except Exception as identifier:
        print('[ ERROR ]', identifier)


if __name__ == "__main__":
    main()
