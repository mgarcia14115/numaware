import cv2 as cv

cam = cv.VideoCapture(0)

if not cam.isOpened():
    raise IOError("Cannot open webcam")

ret, frame = cam.read()

if not ret:
    raise IOError("Cannot read from webcam")


cv.imwrite("/vol/img.jpg",frame)

cam.release()

print("Image captured and saved")
