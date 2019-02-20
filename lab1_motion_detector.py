import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)
start = time.time()
#prev_time = int(datetime.now().strftime('%S'))
count=0
k = 15
n = 3

ret, last_frame = cap.read()

if last_frame is None:
    exit()


#while(cap.isOpened()):
#while(True):
while(True):
    ret, frame = cap.read()

    if frame is None:
        break

    a = 255-cv2.absdiff(last_frame, frame)
    
    curr_time = time.time()
    time_to_write = int(curr_time - start)
    if (time_to_write == n and time_to_write < k):
        name = "frame%d.jpg"%count
        cv2.imwrite(name, a)
        start = curr_time
#      
    count+=1

#    cv2.imshow('frame', frame)
#    cv2.imshow('a', a)
#
#    if cv2.waitKey(33) >= 0:
#        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    last_frame = frame

cap.release()
cv2.destroyAllWindows()






#import numpy as np
#
#import cv2
#
# 
#
#sdThresh = 10
#
#font = cv2.FONT_HERSHEY_SIMPLEX
#
##TODO: Face Detection 1
#
# 
#
#def distMap(frame1, frame2):
#
#    """outputs pythagorean distance between two frames"""
#
#    frame1_32 = np.float32(frame1)
#
#    frame2_32 = np.float32(frame2)
#
#    diff32 = frame1_32 - frame2_32
#
#    norm32 = np.sqrt(diff32[:,:,0]**2 + diff32[:,:,1]**2 + diff32[:,:,2]**2)/np.sqrt(255**2 + 255**2 + 255**2)
#
#    dist = np.uint8(norm32*255)
#
#    return dist
#
# 
#
#cv2.namedWindow('frame')
#
#cv2.namedWindow('dist')
#
# 
#
##capture video stream from camera source. 0 refers to first camera, 1 referes to 2nd and so on.
#
#cap = cv2.VideoCapture(0)
#
# 
#
#_, frame1 = cap.read()
#
#_, frame2 = cap.read()
#
# 
#
#facecount = 0
#
#while(True):
#
#    _, frame3 = cap.read()
#
#    rows, cols, _ = np.shape(frame3)
#
#    cv2.imshow('dist', frame3)
#
#    dist = distMap(frame1, frame3)
#
# 
#
#    frame1 = frame2
#
#    frame2 = frame3
#
# 
#
#    # apply Gaussian smoothing
#
#    mod = cv2.GaussianBlur(dist, (9,9), 0)
#
# 
#
#    # apply thresholding
#
#    _, thresh = cv2.threshold(mod, 100, 255, 0)
#
# 
#
#    # calculate st dev test
#
#    _, stDev = cv2.meanStdDev(mod)
#
# 
#
#    cv2.imshow('dist', mod)
#
#    cv2.putText(frame2, "Standard Deviation - {}".format(round(stDev[0][0],0)), (70, 70), font, 1, (255, 0, 255), 1, cv2.LINE_AA)
#
#    if stDev > sdThresh:
#
#            print("Motion detected.. Do something!!!");
#
#            #TODO: Face Detection 2
#
# 
#
#    cv2.imshow('frame', frame2)
#
#    if cv2.waitKey(1) & 0xFF == 27:
#
#        break
#
#
#cap.release()
#
#cv2.destroyAllWindows()














import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)
start = time.time()
count=0
k = 15
n = 3

ret, last_frame = cap.read()

if last_frame is None:
    exit()


while(True):
    ret, frame = cap.read()

    if frame is None:
        break

    a = 255-cv2.absdiff(last_frame, frame)
    
    curr_time = time.time()
    time_to_write = int(curr_time - start)
    if (time_to_write == n and time_to_write < k):
        name = "frame%d.jpg"%count
        cv2.imwrite(name, a)
        start = curr_time
    
    count+=1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    last_frame = frame

cap.release()
cv2.destroyAllWindows()










