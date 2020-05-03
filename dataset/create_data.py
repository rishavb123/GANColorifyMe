import sys

import cv2
import time

capture = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

res = (28, 28)
frames = -1 if len(sys.argv) < 2 else int(sys.argv[1])

count = 0

while True:
    cur_time = time.time()
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
        s = str(int(time.time() * 1000))
        faceImg = cv2.resize(faceImg, res)
        cv2.imwrite('./images/color/image-' + s + '.png', faceImg)
        cv2.imwrite('./images/grayscale/image-' + s + '.png', cv2.cvtColor(faceImg, cv2.COLOR_BGR2GRAY))
        count += 1
        if count == frames:
            print(count, '/', frames, 'fps:', int(1 / (time.time() - cur_time)), '               ', end='\r')
            break
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
    print(count, '/', frames, 'fps:', int(1 / (time.time() - cur_time)), '               ', end='\r')
print()
capture.release()
