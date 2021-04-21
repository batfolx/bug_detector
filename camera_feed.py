import detect_images
import cv2


def detect_webcam():
    capture = cv2.VideoCapture(0)
    while True:

        ret, frame = capture.read()
        if not ret:
            continue

        frame_boxes = detect_images.detect_image(frame)
        cv2.imshow('Feed', frame_boxes)
        if cv2.waitKey(1) & ord('q') == 0xFF:
            break

    cv2.destroyAllWindows()
    capture.release()

