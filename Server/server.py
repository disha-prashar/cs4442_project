# Created by Youssef Elashry to allow two-way communication between Python3 and Unity to send and receive strings
# Two-way communication between Python 3 and Unity (C#) - Y. T. Elashry
# It would be appreciated if you send me how you have used this in your projects (e.g. Machine Learning) at youssef.elashry@gmail.com

# Edits made by Western University Students

import UdpComms as U
import time

# Create UDP socket to use for sending (and receiving)
sock = U.UdpComms(udpIP="127.0.0.1", portTX=8000, portRX=8001, enableRX=True, suppressWarnings=True)
print("Python Server started")

# for testing purposes

while True:
    data = sock.ReadReceivedData() # read data

    if data != None: # if NEW data has been received since last ReadReceivedData function call
        print(data) # print new received data
        
        if data == "Hey":
            sock.SendData("Hello I am a Potion seller") # Send this string to other application

        elif data == "test":
            sock.SendData("I sell many things!")
        
        else:
            sock.SendData("Is there anything else you need?")

    time.sleep(1)
