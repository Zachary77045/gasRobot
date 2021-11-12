#sudo chmod a+rw /dev/ttyUSB0
#ipython palmSerF3.py /dev/ttyUSB0
import os, sys, time, threading, socket, serial, argparse, csv
print(sys.executable)
#from IPython.display import display, clear_output
import numpy as np
from datetime import datetime
from pathlib import Path

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("comPort", default=0)
#parser.add_argument("udpPort", default=9800)
args = parser.parse_args()
#####################################################################
#initial veriables

V_offset = 15e-3 #V
UDP_IP = '127.0.0.2'
UDP_Port = 9800+int(args.comPort)
myCOM='COM3'
if os.name == 'nt':
    myCOM = 'COM'+str(args.comPort)
else:
    myCOM = '/dev/ttyUSB'+str(args.comPort) #/dev/ttyUSB

#myCOM=args.comPort

print('Com port = ', myCOM)


print("UDP channel is ", UDP_IP, UDP_Port)
def write_csv(fileName,data):
    with open(fileName, 'a') as outfile:
        writer = csv.writer(outfile,  lineterminator='\n')
        writer.writerow(data)

ser = serial.Serial(myCOM, baudrate=115200)

#fileName = args.fileName
timestr = time.strftime("%Y%m%d_%H%M%S")
fileNameRaw = "data_"+str(args.comPort)+"/gasMatrix_"+timestr+"_raw.csv"
fileNameRes = "data_"+str(args.comPort)+"/gasMatrix_"+timestr+"_res.csv"
Path("data_"+str(args.comPort)).mkdir(parents=True, exist_ok=True)
print("File will be created:", fileNameRaw)
print("File will be created:", fileNameRes)

def threadCall():
    threading.Timer(0.05,uartTask).start() #Call itself
#####################################################################
# Main function
compImg = []
compImgR= []
initOK = False
temp=0
rh=0
def uartTask():
    #timestr = time.strftime("%Y%m%d_%H%M%S")
    #fileName = "data/gasMatrix_"+timestr+".csv"
    msgByte=ser.readline()
    msg=str(msgByte)\
            .replace('b\'', '')\
            .replace('\\n', '')\
            .replace('\\r', '')\
            .replace('\'', '')\
            .replace('\\x', '')\
            .split(',') 
    global initOK
    if int(msg[0],16)==0:
        initOK = True
    if initOK == False:
        print('waiting for first package')
        threadCall()
        return
    
    global temp, rh, compImgR, compImg
    if int(msg[0],16)==0:
        temp = float(msg[1])
        rh = float(msg[2])
        compImg = []
        compImgR = []
    else:
        #Send to UDP plot
        myUdp = [((10*4.096*int(adc,16)/65535.0)/(4.0972-4.096*int(adc,16)/65535.0)) for adc in msg[1:-1]]
        myUdp = ["{:.4E}".format(float(r)) for r in myUdp]
        compImgR = compImgR+myUdp
        myUdp = [str(int(msg[0],16)-1)] + myUdp
        # udp_compImgR=','.join(compImgR)
        print("myUdp", myUdp)
        udpMsg=','.join(myUdp)
        print(msg[1:-1])
        print("udpMsg:", len(udpMsg))
        # print("compImgR:", len(udp_compImgR))
        # sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
        # sock.sendto(udpMsg.encode(), (UDP_IP, UDP_Port))
        # sock.close()

        compImg = compImg+msg[1:-1]
        #received last char, save to csv
        if msg[-1]==';':
            udp_compImgR=','.join(compImgR[0:20])
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
            sock.sendto(udp_compImgR.encode(), (UDP_IP, UDP_Port))
            sock.close()
            firstData = np.array([time.strftime("%Y-%m-%d_%H:%M:%S"), temp,rh])
            data = np.concatenate((firstData, np.array(compImg)))
            write_csv(fileNameRaw,data)
            data = np.concatenate((firstData, np.array(compImgR)))
            write_csv(fileNameRes,data)
            # print(myCOM, datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
            # print(temp,'deg C ',rh,'% ',compImgR[0],'MOhm')
    threadCall()
    '''
    global initOK
    if initOK == False:
        rx = ser.read(1)
        while rx !=b'\n':
            rx = ser.read(1)
            #print(rx)
            initOK = False
        initOK = True
        print('init ok')
        
    if initOK == True:
        msgByte=ser.readline()
        msg=str(msgByte)\
            .replace('b\'', '')\
            .replace('\\n', '')\
            .replace('\\r', '')\
            .replace('\\x', '')\
            .replace('\'', '')\
            .split(',')

        temp=0
        rh=0
        try:
            temp = float(msg[0])
            rh = float(msg[1])
        except:
            threading.Timer(0.05,uartTask).start() #Call itself
            return
        #Save csv
        firstData = np.array(["{:.2f}".format(time.time()), temp,rh])
        data = np.concatenate((firstData, np.array(msg[2:])))
        write_csv(fileNameRaw,data)

        resList = [((10*4.096*int(adc,16)/65535)/(4.0972-4.096*int(adc,16)/65535)) for adc in np.array(msg[2:])]
        data = np.concatenate((firstData, np.array(resList)))
        write_csv(fileNameRes,data)
        print(myCOM, datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        print(temp,'deg C ',rh,'% ',round(resList[0],4),'MOhm')
        
        #UDP package
        dataLst = data[3:].tolist()
        dataLst = ["{:.4E}".format(float(r)) for r in dataLst]
        #print(dataLst)
        for i in range(10):
            myUdp = [str(i)]+dataLst[i*10:i*10+10]
            udpMsg=','.join(myUdp)
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
            sock.sendto(udpMsg.encode(), (UDP_IP, UDP_Port))
            sock.close()
            time.sleep(0.001)
    threading.Timer(0.05,uartTask).start() #Call itself
    '''
    

#Run forever
uartTask()