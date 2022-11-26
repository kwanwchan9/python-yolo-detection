import cv2
import numpy as np

# Import configuartion and weight file for YOLOv3 download from officaial website
# Please download the "yolov3.weights" on your own
yolo = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")

# Import trained class in the coco.names files
classes = []
with open("coco.names", 'r') as f:
    classes = f.read().splitlines()

# Loading the input image
img = cv2.imread('myphoto_1.jpg')
height, width, channels = img.shape

# From BGR to RGB and resize the image to 320 x 320 and use blob to extract features
blob = cv2.dnn.blobFromImage(img, 1/255, (320,320), (0,0,0), swapRB=True, crop=False)

yolo.setInput(blob)
output_layers_name = yolo.getUnconnectedOutLayersNames()
layeroutput = yolo.forward(output_layers_name)

# Capture bounding box and infromation to input image
boxes = []
confidences = []
class_ids = []

# Find that probability of characters for the classs in the boxes
for output in layeroutput:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5: #Threshold for probability of object detectedd
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)

            # Coordinate of Rectangle
            x = int(center_x - w/2)
            y = int(center_y - h/2)

            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Non maximum suppresion to remove noise
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size = (len(boxes), 3))

# Add bounding box to each object detected, label with classes and confidence
if len(indexes) > 0:
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confi_value = str(round(confidences[i], 2))
        color = colors[i]

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        cv2.putText(img, label + " "+ confi_value, (x, y + 20), font, 1.5, (255,255,255), 2)

cv2.imshow("Image", img)
# save = cv2.imwrite("output_1.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

