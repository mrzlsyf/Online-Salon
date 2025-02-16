# import library
import random

import cv2

# face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# read hair_filter
angka = random.randrange(0, 7)
hair_filter = cv2.imread('hair_filter'+str(angka)+'.png')
hair_h, hair_w, hair_c = hair_filter.shape
filter_gray = cv2.cvtColor(hair_filter, cv2.COLOR_BGR2GRAY)
ret, ori_hair = cv2.threshold(filter_gray, 0, 255, cv2.THRESH_BINARY_INV)
ori_hair_inv = cv2.bitwise_not(ori_hair)

# read video
cap = cv2.VideoCapture(0)
ret, img = cap.read()
img_h, img_w = img.shape[:2]

# keep running until the user stops the loop
while True:
    # find the face
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # for each face found
    for (x, y, w, h) in faces:
        # face coordinate
        face_w = w
        face_h = h
        face_x1 = x
        face_x2 = face_x1+face_w
        face_y1 = y
        face_y2 = face_y1+face_h

        # size hair_filter
        hair_width = int(1.5 * face_w)
        hair_height = int(hair_width * hair_h/hair_w)

        # hair filter coordinate location settings
        hair_x1 = face_x2 - int(face_w/2) - int(hair_width/2)
        hair_x2 = hair_x1 + hair_width
        hair_y1 = face_y1 - int(face_h*0.9)
        hair_y2 = hair_y1 + hair_height

        # check to see if it is out of frame or not
        if hair_x1 < 0:
            hair_x1 = 0
        if hair_y1 < 0:
            hair_y1 = 0
        if hair_x2 > img_w:
            hair_x2 = img_w
        if hair_y2 > img_h:
            hair_y2 = img_h

        # take into account any changes outside the frame
        hair_width = hair_x2 - hair_x1
        hair_height = hair_y2 - hair_y1

        # change the size of the hair to fit the face
        hair_filter = cv2.resize(hair_filter, (hair_width, hair_height), interpolation=cv2.INTER_AREA)
        hair = cv2.resize(ori_hair, (hair_width, hair_height), interpolation=cv2.INTER_AREA)
        hair_inv = cv2.resize(ori_hair_inv, (hair_width, hair_height), interpolation=cv2.INTER_AREA)

        # take ROI for hair_filter
        roi = img[hair_y1:hair_y2, hair_x1:hair_x2]
        roi_bg = cv2.bitwise_and(roi, roi, mask=hair)
        roi_fg = cv2.bitwise_and(hair_filter, hair_filter, mask=hair_inv)
        dst = cv2.add(roi_bg, roi_fg)

        img[hair_y1:hair_y2, hair_x1:hair_x2] = dst

        break

    # show image
    cv2.imshow('Online Salon', img)

    # if press 'q'  then out
    if cv2.waitKey(1) == ord('q'):
        break

# shutdown camera
cap.release()
cv2.destroyAllWindows()