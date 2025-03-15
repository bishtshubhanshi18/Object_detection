import cv2
import numpy as np
import pygame
from gtts import gTTS
import os
import time

# Initialize Pygame
pygame.mixer.init()

# Load YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize webcam feed
cap = cv2.VideoCapture()
last_detected = ""

while True:
    ret, frame = cap.read()
    height, width, channels = frame.shape

    # Detect objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)

            # Text to Speech - Only if new detection or after a delay
            if label != last_detected:
                last_detected = label
                tts = gTTS(text=label, lang='en')
                tts.save("detected.mp3")
                pygame.mixer.music.load("detected.mp3")
                pygame.mixer.music.play()

    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)  # Small delay for smoother display
    if key == 27:  # Escape key to exit
        break

cap.release()
cv2.destroyAllWindows()
