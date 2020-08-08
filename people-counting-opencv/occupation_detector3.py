import time
import threading
import cv2
import dlib
import numpy as np
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
import socket
import argparse
import simpleaudio as sa
import pandas as pd
from datetime import datetime
import pytz
import smtplib,ssl
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import formatdate
from email import encoders
import os
 
 
def send_email(excelsheet1, excelsheet2):

   print("[INFO]: Running send_email function")
   print(excelsheet1)
   print(excelsheet2)
   msg = MIMEMultipart()
   sender_email = "maskdetector101@gmail.com"
   receiver_email = "srinivassriram06@gmail.com"
   password = "LearnIOT06!"
   body = 'Here is an excel sheet which contains the attendance sheet for today'
   msg['From'] = 'maskdetector101@gmail.com'
   msg['To'] = 'srinivassriram06@gmail.com'
   msg['Date'] = formatdate(localtime = True)
   msg['Subject'] = 'Here is the attendance list for today'

   part = MIMEBase('application', "octet-stream")
   part.set_payload(open(excelsheet1, "rb").read())
   encoders.encode_base64(part)
   part.add_header('Content-Disposition', 'attachment; filename="enter_file.xlsx"')
   part2 = MIMEBase('application', "octet-stream")
   part2.set_payload(open(excelsheet2, "rb").read())
   encoders.encode_base64(part2)
   part2.add_header('Content-Disposition', 'attachment; filename="exit_file.xlsx"')
   msg.attach(part)
   msg.attach(part2)
   msg.attach(MIMEText(body,"plain"))


   context = ssl.create_default_context()
   with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
       server.login(sender_email, password)
       server.sendmail(sender_email, receiver_email, msg.as_string())
 
 
 
tz_NY = pytz.timezone('America/New_York')
datetime_NY = datetime.now(tz_NY)
Time = datetime_NY.strftime("%H:%M %p")
Date = datetime_NY.strftime("%m")+"/"+datetime_NY.strftime("%d")+"/"+datetime_NY.strftime("%y")
Enter = {"Date":[], "Time":[]}
Exit = {"Date":[], "Time":[]}
 
 
total_faces_detected_locally = 0
total_faces_detected_by_peer = 0
peer_ip_address = ""
max_occupancy = 2
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
centroid_list = []
 
totalFrames = 0
totalDown = 0
totalUp = 0
totalPeople = 0
 
moveDict = {}
 
filename = 'speech.wav'
wave_obj = sa.WaveObject.from_wave_file(filename)
arr = []
emailSent = False
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
   global centroid_list
   global datetime_NY
   global emailSent

   while True:
       ret, frame = vs.read()
 
       #frame = frame
       frame = cv2.resize(frame, (240, 240), interpolation=cv2.INTER_AREA)
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
 
       cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
 
       objects = ct.update(rects)
 
       for (objectID, centroid) in objects.items():
           if objectID in moveDict:
               values = moveDict[objectID]
               values.append(centroid[1])
               moveDict[objectID] = values
           else:
               moveDict[objectID] = [centroid[1]]
           #print("[MOVE DICTIONARY]: ", moveDict)
           to = trackableObjects.get(objectID, None)
           if to is None:
               to = TrackableObject(objectID, centroid)
           else:
               #y = [c[1] for c in to.centroids]
               #x = [c[0] for c in to.centroids]
               #direction = centroid[0] - np.mean(x)
               #print("Direction of person:", direction)
               #print("Current Centroids 1: {} 2: {} vs. Middle {}".format(centroid[0], centroid[1], W //2))
               centroid_list.append(centroid[0])
               to.centroids.append(centroid)
               if not to.counted:
                   """
                   final_centroid = centroid_list[-1]
                   beginning_centroid = centroid_list[0]
                   if final_centroid < H // 2:
                       #print("[SRINI]: ", len(centroid_list))
                       #print("[MILAN]: ", (final_centroid - beginning_centroid))
                       if len(centroid_list) > 100 and final_centroid != beginning_centroid:
                           total_faces_detected_locally += 1
                           #print("[SRINI]: Number of people in =", total_faces_detected_locally)
                           to.counted = True
                           centroid_list.clear()
                   elif final_centroid > H // 2:
                       if len(centroid_list) > 100 and final_centroid != beginning_centroid:
                           total_faces_detected_locally -= 1
                           print("[SRINI]: Number of people in =", total_faces_detected_locally)
                           to.counted = True
                           centroid_list.clear()
 
                   """
                   for keyName in moveDict:
                       keyVals = moveDict[keyName]
                       # for i in range(len(keyVals)):
                           # keyVals[i] = keyVals[i].item()
                       if "Counted" in keyVals:
                           pass
                       elif (keyVals[0] < W // 2) and (keyVals[-1] > W // 2):
                           totalUp += 1
                           totalPeople += 1
                           total_faces_detected_locally -=1
                           # TODO
                           Time = datetime_NY.strftime("%H:%M %p")
                           Date =datetime_NY.strftime("%m")+"/"+datetime_NY.strftime("%d")+"/"+datetime_NY.strftime("%y")
                           enterD = Enter["Date"]
                           enterD.append(Date)
                           Enter["Date"] = enterD
                           enterT = Enter["Time"]
                           enterT.append(Time)
                           Enter["Time"] = enterT
 
                           # FINISH CODE BEFORE THIS
                           values = moveDict[keyName]
                           values.append("Counted")
                           moveDict[keyName] = values
                           to.counted = True
                       elif (keyVals[0] > W // 2) and (keyVals[-1] < W // 2):
                           totalPeople -= 1
                           totalDown += 1
                           total_faces_detected_locally += 1
                           # TODO
                           Time = datetime_NY.strftime("%H:%M %p")
                           Date =datetime_NY.strftime("%m")+"/"+datetime_NY.strftime("%d")+"/"+datetime_NY.strftime("%y")
                           exitD = Exit["Date"]
                           exitD.append(Date)
                           Exit["Date"] = exitD
                           exitT = Exit["Time"]
                           exitT.append(Time)
                           Exit["Time"] = exitT
                          
                           # FINISH CODE BEFORE THIS
                           values = moveDict[keyName]
                           values.append("Counted")
                           moveDict[keyName] = values
                           to.counted = True
 
           trackableObjects[objectID] = to
           text = "ID {}".format(objectID)
           cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
           cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
 
       totalFrames += 1

       cv2.imshow("Frame", frame)
       cv2.waitKey(1)

       now = datetime.now().time()
       hr = now.hour
       min = now.minute

       if hr == 23 and min == 37 and not(emailSent):
           enterDataframe = pd.DataFrame(Enter)
           exitDataframe = pd.DataFrame(Exit)
           enterfile = enterDataframe.to_excel("enter_file.xlsx")
           exitfile = exitDataframe.to_excel("exit_file.xlsx")
           send_email("enter_file.xlsx", "exit_file.xlsx")
           emailSent = True


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
       print("[INFO D 1]: ", total_faces_detected)
       print("[INFO L 2]: ", total_faces_detected_locally)
       print("[INFO P 3]: ", total_faces_detected_by_peer)
       print("Thead4: Compute total faces detected by both cameras: {}".format(total_faces_detected))
       if total_faces_detected  >  max_occupancy:
           print("Please wait because the occupancy is greater than {}".format(max_occupancy))
           play_obj = wave_obj.play()
           play_obj.wait_done()
       time.sleep(5)
 
 
if __name__ == "__main__":
   # thread_for_capturing_face()
   parser = argparse.ArgumentParser()
   parser.add_argument("-p", "--peer_ip_address", type=str, help="Provide the IP address of the remote raspberry PI.")
   parser.add_argument("-d", "--debug", type=bool, required=False, default=False, help="debug the application at run time by incrementing the total_faces_detected_locally.")
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
   # both threads completely executed
   print("Done!")
 


