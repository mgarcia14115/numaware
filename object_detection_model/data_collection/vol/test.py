import cv2 as cv

for idx in range(10):

    cap = cv.VideoCapture(idx)

    if cap.isOpened():
        print(f"Camera {idx} is good")
    else:
        print(f"Camera {idx} is bad")
