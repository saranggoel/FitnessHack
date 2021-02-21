import cv2
import time
import numpy as np
import argparse
import math

parser = argparse.ArgumentParser(description='Run keypoint detection')
parser.add_argument("--device", default="gpu", help="Device to inference on")
parser.add_argument("--video_file", default="path + input video", help="Input Video")
parser.add_argument("--exercise", default="none", help="Enter the name of the exercise (in real application, this will be automatically detected)")

args = parser.parse_args()

MODE = "MPI"


protoFile = "path + pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "path + pose_iter_160000.caffemodel"
nPoints = 15
POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]


inWidth = 368
inHeight = 368
threshold = 0.1

exercise = args.exercise
input_source = args.video_file
cap = cv2.VideoCapture(input_source)
hasFrame, frame = cap.read()

vid_writer = cv2.VideoWriter('path + output video',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

if args.device == "cpu":
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU device")
elif args.device == "gpu":
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")

planktotalbackerror = 0
planktotallegerror = 0

count = 0
lastleftshoulder = []
lastrightshoulder = []
lastleftelbow = []
lastrightelbow = []
lastleftwrist = []
lastrightwrist = []

holderleftshoulder = []
holderrightshoulder = []
holderleftelbow = []
holderrightelbow = []
holderleftwrist = []
holderrightwrist = []

while cv2.waitKey(1) < 0:

    t = time.time()
    hasFrame, frame = cap.read()
    frameCopy = np.copy(frame)
    if not hasFrame:
        cv2.waitKey()
        break


    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]
    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        def slope(x1, y1, x2, y2):
            denom = (x2 - x1)
            if denom == 0:
                denom = 0.001
            return (float)(y2 - y1) / denom

        # store coordinates of different body parts
        if i == 0:
            head = [x, y]
        elif i == 1:
            neck = [x, y]
        elif i == 5:
            leftshoulder = [x, y]
        elif i == 2:
            rightshoulder = [x, y]
        elif i == 6:
            leftelbow = [x, y]
        elif i == 3:
            rightelbow = [x, y]
        elif i == 7:
            leftwrist = [x, y]
        elif i == 4:
            rightwrist = [x, y]
        elif i == 14:
            chest = [x, y]
        elif i == 11:
            lefthip = [x, y]
        elif i == 8:
            righthip = [x, y]
        elif i == 12:
            leftknee = [x, y]
        elif i == 9:
            rightknee = [x, y]
        elif i == 13:
            leftankle = [x, y]
        elif i == 10:
            rightankle = [x, y]

        if prob > threshold : 
            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)


    # example analysis on 3 different exercises (more will be added soon)

    if exercise == "plank":
        backerror = 0
        legerror = 0
        if not 0 <= slope(neck[0], neck[1], chest[0], chest[1]) <= 0.15:
            backerror += 1
        if not 0 <= slope(chest[0], chest[1], lefthip[0], lefthip[1]) <= 0.15:
            backerror += 1
        if not 0 <= slope(chest[0], chest[1], righthip[0], righthip[1]) <= 0.15:
            backerror += 1
        if not 0.05 <= slope(lefthip[0], lefthip[1], leftknee[0], leftknee[1]) <= 1:
            legerror += 1
        if not 0.05 <= slope(righthip[0], righthip[1], rightknee[0], rightknee[1]) <= 1:
            legerror += 1
        if not 0.05 <= slope(leftknee[0], leftknee[1], leftankle[0], leftankle[1]) <= 1:
            legerror += 1
        if not 0.05 <= slope(rightknee[0], rightknee[1], rightankle[0], rightankle[1]) <= 0.5:
            legerror += 1
        if backerror > 1:
            backtext = "Raise or lower back into proper position"
        else:
            backtext = "Back is currently in proper position"

        if legerror > 1:
            legtext = "Raise or lower leg into proper position"
        else:
            legtext = "Leg is currently in proper position"
        planktotalbackerror += backerror
        planktotallegerror += legerror

    elif exercise == "pullup":
        armerror = 0
        legerror = 0
        if not 2 <= slope(lefthip[0], lefthip[1], leftknee[0], leftknee[1]):
            legerror += 1
        if not 2 <= slope(righthip[0], righthip[1], rightknee[0], rightknee[1]):
            legerror += 1
        if not 2 <= slope(leftknee[0], leftknee[1], leftankle[0], leftankle[1]):
            legerror += 1
        if not 2 <= slope(rightknee[0], rightknee[1], rightankle[0], rightankle[1]):
            legerror += 1
        if legerror > 1:
            legtext = "Legs are not fully extended"
        else:
            legtext = "Current leg position is fine"


        lasttolastleftshoulder = lastleftshoulder[:]
        lasttolastrightshoulder = lastrightshoulder[:]
        lastleftshoulder = holderleftshoulder[:]
        lastrightshoulder = holderrightshoulder[:]
        holderleftshoulder = leftshoulder[:]
        holderrightshoulder = rightshoulder[:]
        lasttolastleftelbow = lastleftelbow[:]
        lasttolastrightelbow = lastrightelbow[:]
        lastleftelbow = holderleftelbow[:]
        lastrightelbow = holderrightelbow[:]
        holderleftelbow = leftelbow[:]
        holderrightelbow = rightelbow[:]
        lasttolastleftwrist = lastleftwrist[:]
        lasttolastrightwrist = lastrightwrist[:]
        lastleftwrist = holderleftwrist[:]
        lastrightwrist = holderrightwrist[:]
        holderleftwrist = leftwrist[:]
        holderrightwrist = rightwrist[:]
        if count >= 2:
            leftdeg1 = math.degrees(math.atan(slope(lastleftelbow[0], lastleftelbow[1], lastleftwrist[0], lastleftwrist[1])))
            leftdeg2 = math.degrees(math.atan(slope(lastleftshoulder[0], lastleftshoulder[1], lastleftelbow[0], lastleftelbow[1])))

            rightdeg1 = math.degrees(math.atan(slope(lastrightelbow[0], lastrightelbow[1], lastrightwrist[0], lastrightwrist[1])))
            rightdeg2 = math.degrees(math.atan(slope(lastrightshoulder[0], lastrightshoulder[1], lastrightelbow[0], lastrightelbow[1])))


            leftbet = 180 - (leftdeg1 + leftdeg2)
            rightbet = 180 - (rightdeg1 + rightdeg2)

            print(rightbet)
            print(leftbet)

            if leftbet < 150 or rightbet < 150:
                armtext = "Did not go all the way up or down"
            print(count)
            print(leftshoulder[1], lastleftshoulder[1], lasttolastleftshoulder[1])
            if (leftshoulder[1] < lastleftshoulder[1] and lastleftshoulder[1] > lasttolastleftshoulder[1]) and (rightshoulder[1] < lastrightshoulder[1] and lastrightshoulder[1] > lasttolastrightshoulder[1]):
                print("hi")
                leftdeg1 = math.degrees(math.atan(slope(lastleftelbow[0], lastleftelbow[1], lastleftwrist[0], lastleftwrist[1])))
                leftdeg2 = math.degrees(math.atan(slope(lastleftshoulder[0], lastleftshoulder[1], lastleftelbow[0], lastleftelbow[1])))

                rightdeg1 = math.degrees(math.atan(slope(lastrightelbow[0], lastrightelbow[1], lastrightwrist[0], lastrightwrist[1])))
                rightdeg2 = math.degrees(math.atan(slope(lastrightshoulder[0], lastrightshoulder[1], lastrightelbow[0], lastrightelbow[1])))

                leftbet = 180 - (leftdeg1 + leftdeg2)
                rightbet = 180 - (rightdeg1 + rightdeg2)

                print(rightbet)
                print(leftbet)

                if leftbet < 150 or rightbet < 150:
                    armtext = "Did not go all the way up or down"
        else:
            armtext = "Current arm position is fine"
            count += 1
    elif exercise == "wallsit":
        if righthip[0] <= leftwrist[0] <= lefthip[0] or righthip[0] <= leftwrist[0] <= lefthip[0]:
            handtext = "Keeps hands off your lap"
        else:
            handtext = "Hand position is fine"
        leftdeg1 = math.degrees(math.atan(slope(lefthip[0], lefthip[1], leftknee[0], leftknee[1], )))
        leftdeg2 = math.degrees(math.atan(slope(leftankle[0], leftankle[1], leftknee[0], leftknee[1], )))

        rightdeg1 = math.degrees(math.atan(slope(righthip[0], righthip[1], rightknee[0], rightknee[1], )))
        rightdeg2 = math.degrees(math.atan(slope(rightankle[0], rightankle[1], rightknee[0], rightknee[1], )))



        leftdif = 180 - (leftdeg1 - leftdeg2)
        rightdif = 180 - (rightdeg1 - rightdeg2)

        print(rightdif, leftdif)

        if 75 < leftdif < 110 or 75 < rightdif < 110:
            legtext = "Leg position is fine"
        else:
            legtext = "Align legs with knees"

    # print(planktotalbackerror, planktotallegerror)

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    if exercise == "plank":
        cv2.putText(frame, backtext, (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        cv2.putText(frame, legtext, (50, 80), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    elif exercise == "pullup":
        cv2.putText(frame, armtext, (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        cv2.putText(frame, legtext, (50, 80), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    elif exercise == "wallsit":
        cv2.putText(frame, handtext, (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        cv2.putText(frame, legtext, (50, 80), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.putText(frame, "OpenPose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.imshow('Output-Keypoints', frameCopy)
    cv2.imshow('Output-Skeleton', frame)

    vid_writer.write(frame)

    done = 0

vid_writer.release()