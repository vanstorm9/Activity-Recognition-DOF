import numpy as np
import math
import cv2
import os
import os.path
from time import time


# Libraries to preform machine learning
import sys
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,accuracy_score, confusion_matrix

from sklearn.decomposition import PCA, RandomizedPCA

# from mpl_toolkits.mplot3d import Axes3D
from sklearn.externals import joblib

from sklearn import cross_validation
from sklearn.linear_model import Ridge
from sklearn.learning_curve import validation_curve, learning_curve
from sklearn.externals import joblib

max_frame = 40
resize_x= 0.3
resize_y = 0.3
main_counter = 0


def draw_flow(im,flow,step=16):
    h,w = im.shape[:2]
    y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1)
    fx,fy = flow[y,x].T

    # create line endpoints
    lines = np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
    lines = np.int32(lines)

    # create image and draw
    vis = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    for (x1,y1),(x2,y2) in lines:
        cv2.line(vis,(x1,y1),(x2,y2),(0,255,0),1)
        cv2.circle(vis,(x1,y1),1,(0,255,0), -1)
    return vis

def prev_frame_setup():
    
    ret, frame_f = capf.read()
    prev_gray = cv2.cvtColor(frame_f,cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.resize(prev_gray, (0,0), fx=resize_x, fy=resize_y)

    capf.release()
    return prev_gray

def find_face():
    
    detected = False
    prev_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.resize(prev_gray, (0,0), fx=resize_x, fy=resize_y)
    face = face_classifier.detectMultiScale(prev_gray, 1.2, 4)

    if len(face) != 0:
        detected = True

    return (detected, prev_gray)

def sensitive_override_check(prob_s, pred):
    if pred == 'nothing':
        override_arr = [prob_s[0,3], prob_s[0,2], prob_s[0,0]]
        max_comp = max(override_arr)

        max_ind = [i for i, j in enumerate(override_arr) if j == max_comp][0]

        qualified_override = False
        if max_comp > 16:
            qualified_override = True

        if qualified_override:
            if max_ind == 0:
                pred = 'waving'
            elif max_ind == 1:
                pred = 'giving'
            else:
                pred = 'Angry'

        #print 'Sensitive Override triggered. . .'
    return pred

def emotion_to_speech(pred):
    engine = pyttsx.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate)
    if pred == 'Neutral':
        speech = 'Hello, you seem fine today'
    elif pred == 'Smiling':
        speech = 'You seem happy. I am very happy that you are happy!'
    elif pred == 'Shocked':
        speech = 'What is wrong? You look like you seen a ghost.'
    elif pred == 'Angry':
        speech = 'Why are you angry? Did something annoy or frustrate you?'
    print speech 
    engine.say(speech)
    engine.runAndWait()

    
slash = '/'

folder_trans = np.array([])
target = np.array([])
label_trans = np.array([])
folder = ''
choice = ''

face_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

t0 = time()

print 'Now loading trained model. . .'
model = joblib.load('Optical-Model/optical-model-diverse.pkl')
t1 = time()
    
print 'Loading time: ', round(time()-t0, 3), 's'

# ---------------------------------
path = 'test.avi'
cap3 = cv2.VideoCapture(path)  
ret, prev_gray = cap3.read()
#prev_gray = cv2.cvtColor(prev_gray,cv2.COLOR_BGR2GRAY)
#prev_gray = cv2.resize(prev_gray, (0,0), fx=resize_x, fy=resize_y)
cap3.release()


capf = cv2.VideoCapture(0)
sensitive_out = 's'

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(path,fourcc, 20.0, (640,480))

print 'Looking for someone to detect. . .'
main_counter = 0
while True:
    ret, frame = capf.read()
    
    detected = False

    if frame == None:
        print 'frame is NONE'
        continue
    
    cv2.imshow('Looking for face. . .', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print 'You have quit'
        break

    detected, prev_gray = find_face()
    
    if detected == True or main_counter!= 0:
        if main_counter == 0:
            print 'Face Detected'
            
            main_prev_gray = frame
            main_prev_gray = cv2.cvtColor(main_prev_gray, cv2.COLOR_BGR2GRAY)
            main_prev_gray = cv2.resize(main_prev_gray, (0,0), fx=resize_x, fy=resize_y)

        else:
            print 'i: ', main_counter
        if frame == None:
            print 'frame is NONE'
            continue
        out.write(frame)
        main_counter = main_counter + 1
        if main_counter > max_frame:
            main_counter = 0
            
            break

prev_gray = main_prev_gray
out.release()
capf.release()
cap4 = cv2.VideoCapture(path)

j = 0
while(cap4.isOpened()):
    ret, frame = cap4.read()

    if frame == None:
        print 'Frame failure, trying again. . .'
        continue

    if j > max_frame + 1:
        break
    frame = cv2.resize(frame, (0,0), fx=resize_x, fy=resize_y)
    #frame = frame[y: y+h, x: x+w]
    #cv2.imshow('To test with', frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray,gray,None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Working with the flow matrix
    flow_mat = flow.flatten()
    if j == 1:
        sub_main = flow_mat
    elif j != 0:
        sub_main = np.concatenate((sub_main, flow_mat))
        prev_gray = gray
        # To show us visually each video
        cv2.imshow('Optical flow',draw_flow(gray,flow))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    j = j + 1


cv2.destroyAllWindows()
cap4.release()

model.predict(sub_main)
            
prob = model.predict_proba(sub_main)
prob_s = np.around(prob, decimals=5)
prob_s = prob_s* 100
# Determine amount of time to predict
t1 = time()
pred = model.predict(sub_main)


print 'predicting time: ', round(time()-t1, 3), 's'

print ''
print 'Prediction: '
print pred

print 'Probability: '
print 'Nothing: ', prob_s[0,1]
#print 'Smiling: ', prob_s[0,3]
print 'Waving: ', prob_s[0,2]
print 'Giving: ', prob_s[0,0]
detected = 0
print ''
print ''








