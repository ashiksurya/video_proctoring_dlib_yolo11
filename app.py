import cv2
import imutils
import time
from facial_detections import detectFace
from blink_detection import isBlinking
from mouth_tracking import mouthTrack
from object_detection import detectObject
from eye_tracker import gazeDetection
from head_pose_estimation import head_pose_detection 
import winsound
from datetime import datetime

global data_record
data_record = []

#For Beeping
frequency = 2500
duration = 1000

#OpenCV videocapture for the webcam
cam = cv2.VideoCapture(0)

#If camera is already opened
if (cam.isOpened() == False):
    cam.open()

#Face Count If-else conditions
def faceCount_detection(faceCount):
    if faceCount > 1:
        time.sleep(5)
        remark = "Multiple faces has been detected."
        winsound.Beep(frequency, duration)
    elif faceCount == 0:
        remark = "No face has been detected."
        time.sleep(3)
        winsound.Beep(frequency, duration)
    else:
        remark = "Face detecting properly."
    return remark


#Main function 
def proctoringAlgo():

    blinkCount = 0

    while True:
        ret, frame = cam.read()
        frame = imutils.resize(frame, width=450)

        record = []

        #Reading the Current time
        current_time = datetime.now().strftime("%H:%M:%S")
        print("Current Time is:", current_time)
        record.append(current_time)

        #Returns the face count and will detect the face.
        faceCount, faces = detectFace(frame)
        print('faceCount_detection =', faceCount_detection(faceCount))
        record.append(faceCount_detection(faceCount))
        print('faceCount =', faceCount)

        if faceCount == 1:

            #Blink Detection
            blinkStatus = isBlinking(faces, frame)
            print('blinkStatus =', blinkStatus[2])

            if blinkStatus[2] == "Blink":
                blinkCount += 1
                record.append(blinkStatus[2] + " count: " + str(blinkCount))
            else:
                record.append(blinkStatus[2])


            # Gaze Detection
            eyeStatus = (gazeDetection(faces, frame))
            print('eyeStatus =', eyeStatus)
            record.append(eyeStatus)

            #Mouth Position Detection
            print('mouthTrack =', mouthTrack(faces, frame))
            record.append(mouthTrack(faces, frame))

            #Object detection using YOLO
            objectName = detectObject(frame)
            print('objectName =', objectName)
            record.append(objectName)

            if len(objectName) > 1:
                time.sleep(4)
                winsound.Beep(frequency, duration)
                continue

            # Head Pose estimation
            print('head_pose_detection =', head_pose_detection(faces, frame))
            record.append(head_pose_detection(faces, frame))

        else:
            data_record.append(record)
            continue

        data_record.append(record)
        # eyeStatus = gazeDetection(faces, frame)
        # print(eyeStatus)
        # print(objectName) 

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    proctoringAlgo()

    # Convert the list to a string with each element on a new line
    activityVal = "\n".join(map(str, data_record))
    # print(activityVal)

    with open('activity.txt', 'w') as file:
        file.write(str(activityVal))











# import cv2

# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))  # 20 fps, frame size

# # Open the default webcam
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if ret:
#         # Write the frame to the output file
#         out.write(frame)

#         # Display the frame on the screen
#         cv2.imshow('Recording', frame)

#         # Press 'q' to stop recording
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break

# # Release everything when done
# cap.release()
# out.release()
# cv2.destroyAllWindows()





