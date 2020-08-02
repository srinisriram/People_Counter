import zmq

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server…")
socket = context.socket(zmq.REQ)
socket.connect("tcp://192.168.6.158:5555")

socket.send(b"Hello")
#  Do 10 requests, waiting each time for a response
#  Get the reply.
while True:
    message = socket.recv()
    print("Received reply: %s" % message)

