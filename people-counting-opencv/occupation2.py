import time

import cv2
import dlib
import imutils
import numpy as np
import zmq
from imutils.video import VideoStream, FPS
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

total_faces_detected_locally = 0
total_faces_detected_by_peer = 0
peer_ip_address = "tcp://192.168.6.158:5555"
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
vs = VideoStream(src=0).start()
time.sleep(2.0)

W = None
H = None

ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

totalFrames = 0
totalDown = 0
totalUp = 0
totalPeople = 0


def thread_for_capturing_face():
    global net
    global total_faces_detected_locally
    global CLASSES
    global vs
    global W
    global H
    global ct
    global trackers
    global trackableObjects
    global totalFrames
    global totalDown
    global totalUp
    global totalPeople

    fps = FPS().start()

    while True:
        frame = vs.read()

        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        status = "Waiting"
        rects = []

        if totalFrames % 40 == 0:
            status = "Detecting"
            trackers = []

            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in np.arange(0, detections.shape[2]):

                confidence = detections[0, 0, i, 2]

                if confidence > 0.4:

                    idx = int(detections[0, 0, i, 1])

                    if CLASSES[idx] != "person":
                        continue

                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")

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

        cv2.line(frame, (0, W // 2), (H, W // 2), (0, 255, 255), 2)

        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():

            to = trackableObjects.get(objectID, None)

            if to is None:
                to = TrackableObject(objectID, centroid)

            else:

                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                if not to.counted:

                    if direction < 0 and centroid[1] < W // 2:
                        totalUp += 1
                        totalPeople += 1
                        print(type(totalPeople))
                        to.counted = True

                    elif direction > 0 and centroid[1] > W // 2:
                        totalPeople -= 1
                        totalDown += 1
                        to.counted = True

            if totalPeople == 5:
                print('Sorry you cannot enter.')
            trackableObjects[objectID] = to

            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)


        info = [
            ("Up", totalUp),
            ("Down", totalDown),
            ("Status", status),
        ]

        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        print("{} people entered in today".format(totalPeople))

        totalFrames += 1
        fps.update()

        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        cv2.destroyAllWindows()


def thread_for_zmq_for_receiving_face_detected_by_peer():
    global total_faces_detected_by_peer
    while run_program:
        #  Wait for next request from client
        message = socket.recv()
        print("Received request: %s" % message)
        total_faces_detected_by_peer = int(message)
        message = socket.recv()


def thread_for_zmq_for_transmitting_face_detected_locally():
    global total_faces_detected_locally
    context = zmq.Context()
    while run_program:
        #  Socket to talk to server
        print("Connecting to hello world server…")
        socket = context.socket(zmq.REQ)
        socket.connect(peer_ip_address)
        time.sleep(1)
        curr_count = 0
        if total_faces_detected_locally > curr_count:
            #  Send the count
            socket.send(str(total_faces_detected_locally))
            curr_count = total_faces_detected_locally


if __name__ == "__main__":
    thread_for_capturing_face()
    # t1 = threading.Thread(target=thread_for_capturing_face)
    # t2 = threading.Thread(target=thread_for_zmq_for_receiving_face_detected_by_peer)
    # t3 = threading.Thread(target=thread_for_zmq_for_transmitting_face_detected_locally)

    # starting thread 1
    # t1.start()
    # starting thread 2
    # t2.start()
    # starting thread 3
    # t3.start()

    # wait until thread 1 is completely executed
    # t1.join()
    # wait until thread 2 is completely executed
    # t2.join()
    # wait until thread 3 is completely executed
    # t3.join()

    # both threads completely executed
    print("Done!")
