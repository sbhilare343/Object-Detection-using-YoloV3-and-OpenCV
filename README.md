# Title: Object Detection Using YoloV3 and OpenCV

### 1. **Importing Libraries**
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```
- **cv2**: This is the OpenCV library, used here for image processing and computer vision tasks.
- **numpy**: This library is used for numerical operations, particularly for handling arrays and matrices.
- **matplotlib.pyplot**: This library is used for plotting and displaying images.

### 2. **Loading the YOLO Model**
```python
yolo = cv2.dnn.readNet('./yolov3.weights', './yolov3.cfg')
```
- **cv2.dnn.readNet**: Loads the pre-trained YOLOv3 model.
  - `'./yolov3.weights'`: This file contains the weights of the YOLO model.
  - `'./yolov3.cfg'`: This file contains the configuration of the YOLO model architecture.

### 3. **Loading Class Labels**
```python
classes = []
with open('./coco.names','r') as namefile:
  classes = namefile.read().splitlines()
```
- **classes**: A list that will store the names of the object classes the YOLO model can detect.
- **coco.names**: This file contains the names of the classes used by YOLO, each on a new line. The code reads this file and splits it into a list of strings.

### 4. **Loading and Preparing the Image**
```python
image_path = input("Path of IMAGE you want to detect object?(./filename.jpg)")
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```
- **image_path**: Prompts the user to input the path of the image they want to analyze.
- **cv2.imread(image_path)**: Reads the image from the provided path and stores it in the variable `img`.
- **cv2.cvtColor(img, cv2.COLOR_BGR2RGB)**: `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` converts an image from BGR to RGB color space.

```python
blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), (0,0,0), swapRB=True, crop=False)
blob.shape
```
- **blobFromImage**: Converts the image into a blob, which is a 4D array that can be passed through the neural network.
  - `1/255`: Normalizes the image's pixel values to the range [0, 1].
  - `(416,416)`: Resizes the image to 416x416 pixels, the input size required by YOLO.
  - `(0,0,0)`: Sets the mean subtraction values to zero (no mean subtraction).
  - `swapRB=True`: Swaps the Red and Blue channels since OpenCV uses BGR by default, while most models expect RGB.
  - `crop=False`: Ensures that the image is not cropped after resizing.

### 5. **Setting the Blob as Input to the Model**
```python
yolo.setInput(blob)
```
- **setInput**: This sets the prepared blob as the input to the YOLO model for further processing.

### 6. **Getting Output Layer Names and Forward Pass**
```python
output_layer_name = yolo.getUnconnectedOutLayersNames()
layeroutput = yolo.forward(output_layer_name)
```
- **getUnconnectedOutLayersNames**: Retrieves the names of the YOLO model’s output layers.
- **forward**: Performs a forward pass through the network, using the blob as input, and gets the output from the specified output layers.

### 7. **Preparing for Bounding Box Detection**
```python
boxes = []
confidences = []
class_ids = []
height, width, _ = img.shape
```
- **boxes**: A list to store the coordinates of the bounding boxes for detected objects.
- **confidences**: A list to store the confidence scores for detected objects.
- **class_ids**: A list to store the class IDs for the detected objects.
- **height, width, _**: These values store the dimensions of the original image.

### 8. **Processing Each Detection**
```python
for output in layeroutput:
  for detection in output:
    score = detection[5:]
    class_id = np.argmax(score)
    confidence = score[class_id]
    if confidence > 0.7:
      center_x = int(detection[0]*width)
      center_y = int(detection[1]*height)
      w = int(detection[2]*width)
      h = int(detection[3]*height)

      x = int(center_x - w/2)
      y = int(center_y - h/2)

      boxes.append([x,y,w,h])
      confidences.append((float(confidence)))
      class_ids.append(class_id)
```
- **for output in layeroutput**: Loops through each output from the YOLO model.
- **for detection in output**: Loops through each detection within the output.
- **score = detection[5:]**: Extracts the scores for each class from the detection.
- **class_id = np.argmax(score)**: Determines which class has the highest score (i.e., the most likely class for this detection).
- **confidence = score[class_id]**: Retrieves the confidence score for the detected class.
- **if confidence > 0.7**: Only considers detections with a confidence score above 0.7.
- **center_x, center_y**: Calculates the center coordinates of the detected object.
- **w, h**: Calculates the width and height of the bounding box.
- **x, y**: Calculates the top-left corner of the bounding box.
- **boxes.append([x,y,w,h])**: Stores the bounding box coordinates.
- **confidences.append(float(confidence))**: Stores the confidence score.
- **class_ids.append(class_id)**: Stores the class ID.

### 9. **Non-Maximum Suppression (NMS)**
```python
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
```
- **NMSBoxes**: Applies Non-Maximum Suppression to eliminate redundant overlapping boxes with lower confidence scores.
  - `0.5`: The threshold for the confidence score.
  - `0.4`: The threshold for the Intersection over Union (IoU) overlap.

### 10. **Drawing Bounding Boxes and Labels**
```python
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(boxes), 3))

for i in indexes.flatten():
  x,y,w,h = boxes[i]
  label = str(classes[class_ids[i]])
  confidence = str(round(confidences[i], 2))
  color = colors[i]
  cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
  cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)
```
- **font**: Specifies the font type for the text.
- **colors**: Generates random colors for the bounding boxes.
- **for i in indexes.flatten()**: Iterates through the indices of the final selected bounding boxes after NMS.
- **cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)**: Draws a rectangle (bounding box) around the detected object.
- **cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)**: Adds a label with the class name and confidence score above the bounding box.

### 11. **Displaying the Final Image**
```python
plt.imshow(img)
```
- **plt.imshow(img)**: Displays the final image with the detected objects, bounding boxes, and labels using Matplotlib.

This code performs object detection on an input image using the YOLOv3 model, drawing bounding boxes around the detected objects and displaying the image with these annotations.
