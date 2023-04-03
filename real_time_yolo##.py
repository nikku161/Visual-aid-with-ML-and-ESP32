
import cv2
import numpy as np
import pyttsx3
import random

net = cv2.dnn.readNet("./yolov3.weights", "cfg/yolov3.cfg")

classes = []
lis = []
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))
count =0
error =0 
text_speech = pyttsx3.init()
while True:
    _, img = cap.read()
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if (round(confidences[i],2) > 0.65):
                lis.append(label)

                count = count + 1 
                if count==10:
                    s = set(lis)
                    l = list(s)
                    if len(l)==0:
                        print("Nothing Detected")
                        text_speech.say("Nothing Detected")
                        text_speech.runAndWait()
                    else:
                        print("\nThere is " ,end="")
                        text_speech.say("There is ")
                        
                        for index,item in enumerate(l):
                            if index==0:
                                print(item,end="")
                                text_speech.say(item)
                                text_speech.runAndWait()
                            else:
                                print(" and " + item ,end=" ")
                                text_speech.say(" and " + item)
                                text_speech.runAndWait()
                        
                        # print( "at a distance of " + str(round(random.uniform(350,600),2)))
                        # text_speech.say("at a distance of " + str(round(random.uniform(350,600),2)))
                        # text_speech.runAndWait()
                        # count =0
                        # lis.clear()
                        
            elif(error < 10):
                error = error +1
                if error==10:
                     print("Nothing Detected")
                     text_speech.say("Nothing Detected")
                     text_speech.runAndWait()
                     error = 0
            else:
                print("Maa mujhe kuchh dikhayi nahi de raha hai!!!")
                text_speech.say("Maa mujhe kuchh dikhayi nahi de raha hai!!!")
                text_speech.runAndWait()
                    
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)
            
        
                

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()