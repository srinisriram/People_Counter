import time
import threading
import zmq


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

time.sleep(1)
hello_message = socket.recv()



def server_send_to_peer():
    global socket
    global number
    time.sleep(1)
    while True:
        #  Wait for next request from client
        message = socket.recv()
        if message != b"Hello":
            print("Received request: %s" % message)

            print(int(message))

            #  Do some 'work'
            time.sleep(1)

            #  Send reply back to client
            socket.send(number)

def enter_number():
    while True:
        number = input("Enter a number")
        number = str(number).encode('utf-8')

if __name__ == "__main__":
    number_thread = threading.Thread(target=enter_number).start()
    server_thread = threading.Thread(target=server_send_to_peer).start()

