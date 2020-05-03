import cv2

capture = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

resolutions = [(30, 30), (40, 40), (50, 50)]
display_res = (500, 500)

while True:
    _, img = capture.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray)
    if len(faces) > 0:
        faces = list(faces)
        faces.sort(key=lambda face: -face[2] * face[3])
        x, y, w, h = faces[0]
        faceImg = img[y: y + h, x: x + h]
        for res in resolutions:
            cv2.imshow('Face - Res: ' + str(res), cv2.resize(cv2.resize(faceImg, res), display_res))
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
capture.release()
