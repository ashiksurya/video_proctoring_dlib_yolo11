from flask import Flask, render_template, request, redirect, url_for
from datetime import datetime
import threading
import os
import cv2
from main import proctoringAlgo

app = Flask(__name__)

if not os.path.exists("recordings"):
    os.makedirs("recordings")

if not os.path.exists("flagged_frames"):
    os.makedirs("flagged_frames")

if not os.path.exists("flagged_frames/face_detect"):
    os.makedirs("flagged_frames/face_detect")

if not os.path.exists("flagged_frames/person_detect"):    
    os.makedirs("flagged_frames/person_detect")

if not os.path.exists("flagged_frames/sus_object"):    
    os.makedirs("flagged_frames/sus_object")    

frame_width = 640
frame_height = 480
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Global variables for managing the camera, recording, and proctoring states
recording_active = False
proctoring_running = False
video_path = None
out = None
live_cam = None
proctoring_thread = None

def start_camera():
    """Initialize the camera feed."""
    global live_cam
    live_cam = cv2.VideoCapture(0)
    if not live_cam.isOpened():
        raise Exception("Error: Could not open video stream.")

def stop_camera():
    """Release the camera and close any open windows."""
    global live_cam, out
    if live_cam:
        live_cam.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

def process_camera_feed():
    """Continuously process the camera feed for proctoring analysis."""
    global recording_active, proctoring_running, out

    video_path = os.path.join("recordings", f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi")
    out = cv2.VideoWriter(video_path, fourcc, 1.0, (frame_width, frame_height))

    while recording_active and proctoring_running:
        ret, frame = live_cam.read()
        if ret:
            # Resize and save frame to output video
            out.write(cv2.resize(frame, (frame_width, frame_height)))
            # Run proctoring algorithm on the frame
            faceCount, objectName, current_time, record = proctoringAlgo(frame)

            person_count = sum(1 for obj, conf in objectName if obj == "person")

            timestamp = current_time.replace(":", "-").replace(".", "-")

            if faceCount > 1:
                image_filename = os.path.join('flagged_frames/face_detect', f"Multiple_face_{timestamp}.jpg")
                cv2.imwrite(image_filename, frame)
                print(f"Image saved to the folder: {image_filename}.")

            if faceCount == 0:
                image_filename = os.path.join('flagged_frames/face_detect', f"No_face_{timestamp}.jpg")
                cv2.imwrite(image_filename, frame)
                print(f"Image saved to the folder: {image_filename}.")

            if person_count > 1:
                image_filename = os.path.join('flagged_frames/person_detect', f"Multiple_Person_{timestamp}.jpg")
                cv2.imwrite(image_filename, frame)
                print(f"Image saved to the folder: {image_filename}.")

            for obj, conf in objectName:
                if obj in ["laptop", "cell phone", "tablet"]:
                    image_filename = os.path.join('flagged_frames/sus_object', f"suspected_object_{obj}_{timestamp}.jpg")
                    cv2.imwrite(image_filename, frame)
                    print(f"Image saved to the folder: {image_filename}.")                    

        else:
            print("Error: Failed to read frame from camera.")
            break

    # Release resources when done
    stop_camera()

@app.route('/')
def index():
    """Home page route that starts camera and proctoring immediately upon accessing URL."""
    global recording_active, proctoring_running, proctoring_thread

    # Only start the proctoring thread if it's not already running
    if not recording_active and not proctoring_running:
        recording_active = True
        proctoring_running = True
        start_camera()  # Ensure a fresh camera session

        # Start the proctoring in a background thread
        proctoring_thread = threading.Thread(target=process_camera_feed, daemon=True)
        proctoring_thread.start()

    return render_template('index.html')

@app.route('/submit_quiz', methods=['POST'])
def submit_quiz():
    """Route to handle quiz submission, stop camera, and save quiz results."""
    global recording_active, proctoring_running, proctoring_thread

    # Stop recording and proctoring
    recording_active = False
    proctoring_running = False

    # Ensure the camera and thread are properly released
    if proctoring_thread and proctoring_thread.is_alive():
        proctoring_thread.join()  # Wait for the thread to finish
    stop_camera()

    # Collect quiz answers
    q1 = request.form.get('q1')
    q2 = request.form.get('q2')
    q3 = request.form.get('q3')
    q4 = request.form.get('q4')
    q5 = request.form.get('q5')
    result = f"Your answers: Q1 = {q1}, Q2 = {q2}, Q3 = {q3}, Q4 = {q4}, Q5 = {q5}"
    
    # Save results with timestamp
    with open('quiz_results.txt', 'a') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {result}\n")

    return redirect(url_for('thank_you'))

@app.route('/thank_you')
def thank_you():
    """Thank you page after quiz submission."""
    return "<h1>Thank you for completing the quiz!</h1>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
