import time
import threading
import cv2
import dlib
import numpy as np
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
import socket
import argparse
import smtplib
import ssl
from email.mime.text import MIMEText
from datetime import datetime

total_faces_detected_locally = 0
total_faces_detected_by_peer = 0
peer_ip_address = ""
max_occupancy = 100
run_program = True

# load our serialized model from disk
print("[INFO] loading model...")
prototxt = "models/MobileNetSSD_deploy.prototxt"
model = "models/MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)
time.sleep(2.0)

ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

totalFrames = 0
totalDown = 0
totalUp = 0
totalPeople = 0


now = datetime.now().time()
print("now =", now)
hr = now.hour
min = now.minute


def thread_for_capturing_face():
    print("[INFO] Running Thread 1...")
    global net
    global total_faces_detected_locally
    global CLASSES
    global vs
    global ct
    global trackers
    global trackableObjects
    global totalFrames
    global totalDown
    global totalUp
    global totalPeople
    global run_program

    while run_program:
        ret, frame = vs.read()

        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        (H, W) = frame.shape[:2]

        status = "Waiting"
        rects = []

        if totalFrames % 10 == 0:
            status = "Detecting"
            trackers = []

            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    idx = int(detections[0, 0, i, 1])
                    if CLASSES[idx] != "person":
                        continue
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)
                    trackers.append(tracker)
        else:
            # loop over the trackers
            for tracker in trackers:
                status = "Tracking"
                tracker.update(rgb)
                pos = tracker.get_position()
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                rects.append((startX, startY, endX, endY))

        cv2.line(frame, (W // 2, 0), (W // 2, H), (0, 255, 255), 2)
        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)
            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                y = [c[1] for c in to.centroids]
                x = [c[0] for c in to.centroids]
                direction = centroid[0] - np.mean(x)
                print("Direction of person:", direction)
                print("Current Centroid {} vs. Middle {}".format(centroid[0], H //2))
                to.centroids.append(centroid)
                if not to.counted:
                    if direction < 0 and centroid[0] < H // 2:
                        totalUp += 1
                        totalPeople += 1
                        total_faces_detected_locally -= 1
                        # print(type(totalPeople))
                        to.counted = True
                    elif direction > 0 and centroid[0] > H // 2:
                        totalPeople -= 1
                        totalDown += 1
                        total_faces_detected_locally += 1
                        to.counted = True

            trackableObjects[objectID] = to
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)


        info = [
            ("Up", totalUp),
            ("Down", totalDown),
            ("Num Of people in the building: ", total_faces_detected_locally),
            ("Status", status),
        ]

        #for (i, (k, v)) in enumerate(info):
            #text = "{}: {}".format(k, v)
            #cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # print("{} people entered in today".format(totalPeople))

        totalFrames += 1
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)


def thread_for_receiving_face_detected_by_peer():
    print("[INFO] Running Thread 2...")
    global total_faces_detected_by_peer
    global run_program
    # Initialize a TCP server socket using SOCK_STREAM
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Bind the socket to the port
    server_address = ('', 10000)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print('Server Thread2: starting up on {} port {}'.format(*server_address))
        s.bind(server_address)
        s.listen(1)
        print('Server Thread2: Waiting for a connection')
        conn, addr = s.accept()
        with conn:
            print('Server Thread2: connection from', addr)
            while run_program:
                data = conn.recv(1024)
                if data:
                    print('Server Thread2: received {} from peer {}.'.format(data, addr))
                    data = data.decode('utf-8')
                    total_faces_detected_by_peer = int(data)
                    print("Server Thread2: total_faces_detected_by_peer = {}".format(total_faces_detected_by_peer))
                else:
                    print("server Thread2: data is Null")


def thread_for_transmitting_face_detected_locally():
    print("Client [INFO] Running Thread 3...")
    global total_faces_detected_locally
    global run_program
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Connect the socket to the port where the server is listening
    server_address = (peer_ip_address, 10000)


    curr_count = 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        successfully_connected_to_peer = False
        while not successfully_connected_to_peer:
            try:
                print('Client Thread3: connecting to {} port {}'.format(*server_address))
                s.connect(server_address)
                successfully_connected_to_peer = True
            except:
                time.sleep(5)
        while run_program:
            try:
                if total_faces_detected_locally != curr_count:
                    # Send the count
                    print("Client Thread3: Sending total_faces_detected_locally={} to peer ip={}, port={}.".format(
                        total_faces_detected_locally,
                        *server_address))
                    s.sendall(str(total_faces_detected_locally).encode())
                    curr_count = total_faces_detected_locally
                    time.sleep(1)
            except:
                print('Client Thread3: Exception: closing client socket')
                s.close()

def thread_for_debug():
    global total_faces_detected_locally
    while run_program:
        print("Thread5: DEBUG: Enter the total number of faces detected locally.\n")
        try:
            total_faces_detected_locally = int(input("Enter a number: "))
        except:
            continue


def thread_for_comparing_local_face_detected_and_global_face_detected():
    global total_faces_detected_by_peer
    global total_faces_detected_locally
    global max_occupancy
    global run_program
    while run_program:
        total_faces_detected = total_faces_detected_locally + total_faces_detected_by_peer
        print("Thead4: Compute total faces detected by both cameras: {}".format(total_faces_detected))
        if total_faces_detected  >  max_occupancy:
            print("Please wait because the occupancy is greater than {}".format(max_occupancy))
            # Play an audio message.
        time.sleep(5)

def thread_for_sending_email():
    global total_faces_detected_by_peer
    global total_faces_detected_locally
    global run_program


    GMAIL_USERNAME = 'maskdetector101@gmail.com'
    GMAIL_PASSWORD = 'LearnIOT06!'

    to = 'srinivassriram06@gmail.com'
    msg = str(total_faces_detected_locally + total_faces_detected_by_peer)

    context = ssl.create_default_context() # Try with a context
    smptserver = smtplib.SMTP("smtp.gmail.com",587) 
    smptserver.ehlo()
    smptserver.starttls(context=context)
    smptserver.ehlo
    smptserver.login(GMAIL_USERNAME, GMAIL_PASSWORD)


    while run_program:
        msg = str(total_faces_detected_locally + total_faces_detected_by_peer)

        now = datetime.now().time()
        hr = now.hour
        min = now.minute

        if min == 5:
            for x in range(1):
                smptserver.sendmail(GMAIL_USERNAME, to, msg)


if __name__ == "__main__":
    # thread_for_capturing_face()
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--peer_ip_address", type=str, help="Provide the IP address of the remote raspberry PI.")
    parser.add_argument("-d", "--debug", type=bool, required=False, default=False, help="debug the application at run time by incrementing the total_faces_detected_locally.")
    parser.add_argument("-e","--email", type=bool, required=False, default=False, help="send email every hour containing the number of people that entered so far")
    args = parser.parse_args()
    peer_ip_address = args.peer_ip_address

    t1 = threading.Thread(target=thread_for_capturing_face)
    t2 = threading.Thread(target=thread_for_receiving_face_detected_by_peer)
    t3 = threading.Thread(target=thread_for_transmitting_face_detected_locally)
    t4 = threading.Thread(target=thread_for_comparing_local_face_detected_and_global_face_detected)

    # starting thread 1
    t1.start()
    # starting thread 2
    t2.start()
    # starting thread 3
    t3.start()
    # starting thread 4
    t4.start()

    if args.debug:
        t5 = threading.Thread(target=thread_for_debug)
        t5.start()
    if args.email:
        t6 = threading.Thread(target=thread_for_sending_email)
        # starting thread 6
        t6.start()


    # wait until thread 1 is completely executed
    t1.join()
    # wait until thread 2 is completely executed
    t2.join()
    # wait until thread 3 is completely executed
    t3.join()
    # wait until thread 4 is completely executed
    t4.join()


    if args.debug:
        t5.join()
    if args.email:
        t6.join()

    # both threads completely executed
    print("Done!")

