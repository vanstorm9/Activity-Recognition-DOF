import cv2


i = 0
cam = cv2.VideoCapture(0)
while True:
    
    print cam.isOpened()
    print i 
    while True:
        ret, img = cam.read()

        if img == None:
            print 'Frame is NONE'
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('video',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                print 'You have quit'
                break
    i = i + 1
cam.release()
