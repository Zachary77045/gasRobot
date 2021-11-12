#bokeh serve --show gasMatrixPlot.py --port 9700 --args 9700
from re import S
from bokeh.plotting import figure, output_file, show, curdoc
import numpy as np
from bokeh.driving import linear
from bokeh.models import ColorBar, BasicTicker, LinearColorMapper, ColumnDataSource,Range1d
from bokeh.layouts import gridplot, column, row
from bokeh.models import TextInput,ColumnDataSource
import socket
import time, csv

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("comPort", default=0)
args = parser.parse_args()

xSize=ySize=10

UDP_IP = '127.0.0.2'
UDP_Port = 9800+int(args.comPort)

color_mapper = LinearColorMapper(palette="Turbo256")  #Viridis256 , low=0, high=30
color_bar = ColorBar(color_mapper=color_mapper, ticker= BasicTicker(),location=(0,0))

pSensor = []
rs = []
ds = []
numSensor = 2
sensor_No = 0
title= ["Spectrum Histogram","1px Signal, sensor_No: ( "+str(sensor_No)+' )',"Normalized Spectrum Histogram"]
xLab = ["Resistance", "Resistance", "Resistance"]
yLab = ["Sensor ID", "Time (n)", "Sensor ID"]
yLim = [30, 10, 256]

for i in range(numSensor):
    if i != 1:
        pSensor.append(figure(title=title[i],y_range=(0,yLim[i]), plot_width=600, plot_height=360,  output_backend="webgl", toolbar_location="above")) #y_range=(-2, 2),
    else:
        pSensor.append(figure(title=title[i], plot_width=600, plot_height=360,  output_backend="webgl", toolbar_location="above"))
    pSensor[i].yaxis.axis_label = xLab[i]
    pSensor[i].xaxis.axis_label = yLab[i]
    rs.append(pSensor[i].line([], [], line_width=2))
    #ds.append(rs[i].data_source)
    #ds[i].data['y'] = np.linspace(0,xSize*ySize,xSize*ySize, dtype=int)
    #ds[i].data['x'] = np.linspace(0,xSize*ySize,xSize*ySize, dtype=int)

srcBar = ColumnDataSource(dict(x=np.linspace(0,xSize*ySize-1,xSize*ySize, dtype=int),top=np.ones((xSize*ySize,), dtype=int),))

pSensor[0].vbar(x="x", top="top",
    width = .9,
    fill_alpha = .5,
    #fill_color = 'salmon',
    line_alpha = .5,
    #line_color='green',
    #line_dash='dashed',
    source=srcBar)
#pSensor[0].y_range = Range1d(start=-2, end=2)
dsT=rs[1].data_source

def write_csv(fileName,data):
    with open(fileName, 'a') as outfile:
        writer = csv.writer(outfile,  lineterminator='\n')
        writer.writerow(data)
timestr = time.strftime("%Y%m%d_%H%M%S")
fileNameRes = "dataset/gasMatrix_"+timestr+"_res.csv"

labels = "atmosphere"
label_list = ["atmosphere"]
text_input = TextInput(value="atmosphere.", title="Label: "+labels+". (label_list: "+str(label_list)+").")
def my_text_input_handler(attr, old, new):
    global labels
    # print("Previous label: " + old)
    # print("Updated label: " + new)
    labels = new
    text_input.title = "Label: "+labels+"(label_list: "+str(label_list)+")."
    if labels not in label_list:
        label_list.append(labels)
        # print(label_list)
        text_input.title = "Label: "+labels+"(label_list: "+str(label_list)+")."
text_input.on_change("value", my_text_input_handler)

sensorNo_input = TextInput(value="0-99", title="Sensor No.:")
def change_sensorNo_handler(attr, old, new):
    global sensor_No
    if new.isdigit() & ((sensor_No > 0)or(sensor_No == 0)) & (sensor_No < 100):
        sensor_No = int(new)
        pSensor[1].title.text = "1px Signal, sensor_No: ( "+str(sensor_No)+' )'
        while len(dsT.data['y']) != 0:
            dsT.data['y'].pop(0)
            dsT.data['x'].pop(0)
sensorNo_input.on_change("value", change_sensorNo_handler)

barChart = []
@linear()
def update(step):
    global barChart, sensor_No, label_list
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((UDP_IP, UDP_Port))
    data, addr = s.recvfrom(1024)
    RVec = data.split(b',')[1:]
    RVec = [float(R) for R in RVec]
    s.close()
    # print(len(RVec), RVec)

    barChart=barChart+RVec

    ###### save data #######################################################################
    if len(barChart) == 100:
        #2D plot
        # 1D graph
        dsT.data['y'].append(float(barChart[sensor_No]))
        dsT.data['x'].append(int(step/10))
        if len(dsT.data['y'])>60*5:
            dsT.data['y'].pop(0)
            dsT.data['x'].pop(0)
        dsT.trigger('data', dsT.data, dsT.data)
        # Bar chart
        srcBar.data['top'] = barChart

        firstData = np.array([time.strftime("%Y-%m-%d_%H:%M:%S")])
        labels_save = np.array([labels]) # .reshape[1, -1]
        label_order = np.array([label_list.index(labels)])
        # print(label_order)
        data = np.concatenate((firstData, label_order,labels_save,label_order,(np.array(barChart)).flatten())) # 2nd label_order is placeholder
        write_csv(fileNameRes, data)
        barChart = []
#############################################################################################


# ptot = column(text_input)
ptot = column(text_input,row(pSensor[0],pSensor[1]),sensorNo_input)
curdoc().add_root(ptot)
curdoc().add_periodic_callback(update, 50)
