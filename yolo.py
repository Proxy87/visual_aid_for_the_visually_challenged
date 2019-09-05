import cv2 as cv
import numpy as np
import time


Idp=0

classesFile = "G:\D&I\YOLO\yolo-coco\coco.names"

with open(classesFile,'rt') as f:
    classes = f.read().split('\n')

modelConf = "G:\D&I\YOLO\yolo-coco\yolov3.cfg" #setting configuration and weights of the network
modelWeights = "G:\D&I\YOLO\yolo-coco\yolov3.weights"
opfile = "C:\\Users\\Darshan\\Desktop\\Windows\\output.txt"

net = cv.dnn.readNetFromDarknet(modelConf, modelWeights) # setting up the network using opencv's dnn
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)

winName = 'Say Hello' #setting display window
cv.namedWindow(winName, cv.WINDOW_NORMAL)
cv.resizeWindow(winName, 600,400)

confThreshold = 0.25 #Setting thresholds
nmsThreshold = 0.40
inpWidth = 256
inpHeight = 256


def postprocess(frame, outs): #Taking individual frames and its corresponding outputs  as arguments
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIDs = []
    confidences = []
    boxes = []

    for out in outs: #for every output vector the class id,confidence... is stripped
        for detection in out:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > confThreshold: #loop used to neglect detections with confidences less than threshold
                centerX = int(detection[0] * frameWidth)
                centerY = int(detection[1] * frameHeight)

                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)

                left = int(centerX - width / 2)
                top = int(centerY - height / 2)
                if centerX < (.3*frameWidth): #to detect the position of the detected object
                    l = "left"
                elif centerX > (.9*frameWidth):
                    l = "right"
                else:
                    l = "front"

                if width > 0.2*frameWidth and height > 0.2*frameHeight:
                    dis = "near"
                else:
                    dis = ""


                classIDs.append(classID)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        drawPred(classIDs[i], confidences[i], left, top, left + width, top + height,l,dis)

def drawPred(classId, conf, left, top, right, bottom,loc,dis): #used to draw bounding boxes around the objects detected

    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    label = '%.2f' % conf

    if classes:
        assert (classId < len(classes))

        label = '%s %s %s' % (classes[classId],loc,dis)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        FileWriter(classId,label,opfile)
        print(label)


def FileWriter(Id,label,opfile): #saving the output to file with earlier mentioned destination

    file = open(opfile, "w")
    file.write("%s\n"%label)
    file.close()


def getOutputsNames(net):
    layersNames = net.getLayerNames()

    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

url = "http://192.168.43.1:8080/video" #Setting up input
cap = cv.VideoCapture(url)
start_time = time.time() #Starting timer to calculate fps
x = 1
counter = 0

while cv.waitKey(1) < 0:

    hasFrame, frame = cap.read() #converting video input to frames
    frame = cv.flip(frame, 1)

    blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False) #converting image/frame to blob so that it can be used in the network
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))

    postprocess(frame, outs)

    cv.imshow(winName, frame)
    counter += 1
    if (time.time() - start_time) > x:
        #print("FPS: ", counter / (time.time() - start_time))
        counter = 0
        start_time = time.time()

