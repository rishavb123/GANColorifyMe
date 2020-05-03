import cv2
import time

capture = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

res = (28, 28)

while True:
    _, img = capture.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    if len(faces) > 0:
        faces = list(faces)
        faces.sort(key=lambda rect: -rect[2] * rect[3])
        x, y, w, h = faces[0]
        faceImg = img[y: y + h, x: x + h]
        s = str(int(time.time() * 10))
        faceImg = cv2.resize(faceImg, res)
        cv2.imwrite('./images/color/image-' + s + '.png', faceImg)
        cv2.imwrite('./images/grayscale/image-' + s + '.png', cv2.cvtColor(faceImg, cv2.COLOR_BGR2GRAY))
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
capture.release()
