__author__='yan9yu'
import sys
import numpy as np
import cv2

im = cv2.imread('../data/handwritten.png')
im3 = im.copy()

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
thresh = cv2.adaptiveThreshold(blur, 250, 1, 1, 11, 2)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APROX_SIMPLE)

samples = np.empty((0,100), np.float32)
responses=[]
keys = [i for i in range(65,90)]

for cnt in contours:

    if cv2.contourArea(cnt)>70:
        [x,y,w,h]=cv2.boundingRect(cnt)

        if h>38:
            cv2.rectangle(im, (x,y), (x+w,y+h), (255,0,0), 2)
            roi=thresh[y:y+h, x:x+w]
            roismall=cv2.resize(roi,(20,20))
            cv2.imshow('my',im)
            key=cv2.waitKey(0)

            if key==27:
                sys.exit()
            elif key in keys:
                responses.append(int(chr(key)))
                sample = roismall.reshape((1,100))
                samples=np.append(samples, sample, 0)

responses=np.array(responses, np.float)
responses = responses.reshape((responses.size, 1))
print("training complete")

samples = np.float32(samples)
responses = np.float32(responses)

cv2.imwrite("../data/train_result.png", im)
np.savetxt('../data/generalsamples.data', samples)
np.savetxt('../data/generalresponses.data', responses)

