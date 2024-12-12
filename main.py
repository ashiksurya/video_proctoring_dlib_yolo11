import time
import winsound
import mysql.connector
from facial_detections import detectFace
from blink_detection import isBlinking
from mouth_tracking import mouthTrack
from object_detection import detectObject
from eye_tracker import gazeDetection
from head_pose_estimation import head_pose_detection
from datetime import datetime


global data_record, running, proctoring_running, blinkCount
data_record = []
running = True
proctoring_running = True
blinkCount = 0

frequency = 2500
duration = 1000


def save_to_database(record):
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="#ashikS10",
        database="quizo"
    )
    cursor = conn.cursor()

    sql = """
        INSERT INTO proctoring_activity (
            activity_time,
            face_status,
            face_count,
            blink_status,
            eye_status,
            mouth_status,
            object_detected,
            head_pose_status
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
    values = (
        record['activity_time'],
        record.get('face_status', 'N/A'),
        record.get('face_count', 'N/A'),
        record.get('blink_status', 'N/A'),
        record.get('eye_status', 'N/A'),
        record.get('mouth_status', 'N/A'),
        str(record.get('object_detected', 'N/A')),
        record.get('head_pose_status', 'N/A')
    )

    cursor.execute(sql, values)
    conn.commit()
    cursor.close()
    conn.close()

def faceCount_detection(faceCount):
    if faceCount > 1:
        # time.sleep(2)
        remark = "Multiple faces has been detected."
        # winsound.Beep(frequency, duration)
    elif faceCount == 0:
        remark = "No face has been detected."
        # time.sleep(2)
        # winsound.Beep(frequency, duration)
    else:
        remark = "Face detecting properly."
    return remark


def proctoringAlgo(frame):
    """ Process each frame for proctoring analysis """
    global blinkCount
    record = {}

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    print("Current time is:", current_time)
    record['activity_time'] = current_time

    faceCount, faces = detectFace(frame)
    print('faceCount_detection:',faceCount_detection(faceCount))
    record['face_status'] = faceCount_detection(faceCount)
    record['face_count'] = faceCount
    objectName =[]
    # print(faceCount)

    if faceCount == 1:
        blinkStatus = isBlinking(faces, frame)
        print('blinkStatus:',blinkStatus[2])

        if blinkStatus[2] == "Blink":
            blinkCount += 1
            record['blink_status'] = f"Blink count: {blinkCount}"
        else:
            record['blink_status'] = blinkStatus[2]

        eyeStatus = (gazeDetection(faces, frame))
        print('eyeStatus:',eyeStatus)
        record['eye_status'] = eyeStatus

        print('mouthTrack:',mouthTrack(faces, frame))
        mouthStatus = mouthTrack(faces, frame)
        record['mouth_status'] = mouthStatus
        # mouthTrack(faces, frame)

        objectName = detectObject(frame)
        print('objectName:',objectName)
        record['object_detected'] = objectName

        # person_count = sum(1 for obj, conf in objectName if obj == 'person')
        # if person_count > 1:
        #     pass
            # time.sleep(2)
            # winsound.Beep(frequency, duration)

        print('head_pose_detection:',head_pose_detection(faces, frame))
        head_pose = head_pose_detection(faces, frame)
        record['head_pose_status'] = head_pose
    else:
        print("No face has been detected.")

    data_record.append(record)
    save_to_database(record)

    with open('activity.txt', 'w') as file:
        for entry in data_record:
            for key, value in entry.items():
                file.write(f"{key}: {value}\n")
            file.write("\n")

    return faceCount, objectName, current_time, record

