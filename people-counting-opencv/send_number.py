import os
import pickle


send_number = 710

with open('cheenu.txt', 'w') as f:
  f.write('%d' % send_number)

os.system("sshpass -p 'raspberry' scp cheenu.txt pi@192.168.6.159:/home/pi")

"""
Mask Detector ip: 192.168.6.159
Occupancy Entrance Detector ip: 192.168.6.227
Social Distancing Detector ip: 192.168.5.0
"""