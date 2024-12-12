# import cv2
# import numpy as np
# import time
# from datetime import datetime


# print('Start Time====', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# #net has the YOLO loaded
# net = cv2.dnn.readNet("object_detection_model/weights/yolov3.weights", 
#                       "object_detection_model/config/yolov3.cfg")

# #classes that we have to detect using Object Detection Model
# label_classes = []

# with open("object_detection_model/objectLabels/coco.names","r") as file:
#     label_classes = [name.strip() for name in file.readlines()]

# layer_names = net.getLayerNames()
# # print(type(layer_names))
# # layer_names = [i for i in layer_names]
# # print(type(layer_names))
# output_layers = [layer_names[layer-1] for layer in net.getUnconnectedOutLayers()]

# colors = np.random.uniform(0,255,size=(len(label_classes),3))

# font = cv2.FONT_HERSHEY_PLAIN
# start_time = time.time()
# frame_id = 0

# def detectObject(frame):

#     labels_this_frame = []

#     height, width, channels = frame.shape

#     blob = cv2.dnn.blobFromImage(frame, 0.00392, (220,220), (0,0,0), True, crop=False)

#     #Feeding Blob as an input to our Yolov3-tiny model
#     net.setInput(blob)

#     #Output labels received at the output of model
#     outs = net.forward(output_layers)

#     #show informations on the screen
#     class_ids = []
#     confidences = []
#     boxes = []

#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]

#             if confidence > 0.5:

#                 #object detected
#                 center_x = int(detection[0]*width)
#                 center_y = int(detection[1]*height)

#                 w = int(detection[2]*width)
#                 h = int(detection[3]*height)

#                 #rectangle co-ordinates
#                 x = int(center_x - w/2)
#                 y = int(center_y - h/2)

#                 boxes.append([x,y,w,h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

#     for i in range(len(boxes)):
#         if i in indexes:
#             #show the box only if it comes in non-max supression box
#             x,y,w,h = boxes[i]
#             label = str(label_classes[class_ids[i]])

#             color = colors[class_ids[i]]
#             labels_this_frame.append((label, confidences[i]))
#             cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#             cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)

#     return labels_this_frame, frame 










from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime

# Load the YOLO11 model
model = YOLO("object_detection_model/yolo11m.pt")
device = model.device

print('Start Time====', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

colors = np.random.randint(0, 255, size=(len(model.names), 3), dtype=int)

def detectObject(frame):
    labels_this_frame = []

    # Resize frame to a smaller resolution for faster processing
    input_size = 640  # YOLO models often perform well with this size
    original_height, original_width = frame.shape[:2]
    scaling_factor = input_size / max(original_height, original_width)
    resized_frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor)

    # Convert frame to RGB as required by the YOLO model
    frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Inference
    results = model(frame_rgb)

    # Process detections
    for detection in results[0].boxes.data:
        x1, y1, x2, y2, confidence, class_id = detection.tolist()
        label = model.names[int(class_id)]
        labels_this_frame.append((label, confidence))

        # Rescale bounding box coordinates to the original frame size
        x1, y1, x2, y2 = (x1 / scaling_factor, y1 / scaling_factor, x2 / scaling_factor, y2 / scaling_factor)

        # Draw bounding box and label
        color = colors[int(class_id)].tolist()
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return labels_this_frame  # Return only the labels and confidence scores


# frame = cv2.imread('c:/Users/a1666/Downloads/people-are-driving-different-types-of-vehicles-in-heavy-traffic.jpg')
# labels, annotated_frame = detectObject(frame)
# print("Detected objects:", labels)

# height, width = annotated_frame.shape[:2]
# max_display_size = 800  # Set maximum display width or height
# if height > max_display_size or width > max_display_size:
#     scaling_factor = max_display_size / max(height, width)
#     annotated_frame = cv2.resize(annotated_frame, None, fx=scaling_factor, fy=scaling_factor)

# cv2.imshow("YOLO11 Detection", annotated_frame)

# print('End Time====', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# cv2.waitKey(0)
# cv2.destroyAllWindows()

