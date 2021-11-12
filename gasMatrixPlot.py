#bokeh serve --show gasMatrixPlot.py --port 9700 --args 9700
from re import S
from bokeh.plotting import figure, output_file, show, curdoc
import numpy as np
from bokeh.driving import linear
from bokeh.models import ColorBar, BasicTicker, LinearColorMapper, ColumnDataSource,Range1d
from bokeh.layouts import gridplot, column, row
from bokeh.models import DatetimeTickFormatter, Div
import socket
import time
import pandas as pd
from bokeh.models import ColumnDataSource, Grid, LinearAxis, Plot, Text

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("comPort", default=0)
args = parser.parse_args()

VOC_label = ['ambient gases', 'apple']
titles = Div(text='<h1 style="text-align: center">target gases: <b>ambient gases</b></h1>', align="center",width=400, height=100, background="white", margin=(150, 15, 15, 5))
# titles = Div(text="target gases: "+VOC_label[0], width=1000, height=100)

xSize=ySize=10

UDP_IP = '127.0.0.2'
UDP_Port = 9800+int(args.comPort)

sta_sec_BL = 10 # seconds when after the program
num_of_frameworks = 10 # num of frameworks to calculate BL
img_BL = []
start_second = int(time.time())
Norm_data = np.array(pd.read_csv('Norm_data.csv', header=None).iloc[0])
print('Norm_data', Norm_data)

#imgMean=np.load('imgMean.npy')
#imgStd=np.load('imgStd.npy')

gasMap = np.random.rand(xSize,ySize) #np.zeros((xSize,ySize), dtype=float)
source = ColumnDataSource(data=dict(image=[gasMap]))
source_N = ColumnDataSource(data=dict(image=[gasMap]))

color_mapper = LinearColorMapper(palette="Turbo256")  #Viridis256 , low=0, high=30
color_mapper_N = LinearColorMapper(palette="Turbo256", low=0, high=256)  #Viridis256
color_bar = ColorBar(color_mapper=color_mapper, ticker= BasicTicker(),location=(0,0))

pGasMap = figure(title='Resistance Pattern', x_range=(0,xSize), y_range=(0,ySize),aspect_scale=1, output_backend="webgl", toolbar_location="below") #
pGasMap.title.text_font_size = "38px"
pGasMap.title.align = "center"
pGasMap.title.background_fill_color = "darkgrey"
pGasMap.title.text_color = "white"
pclrBar = figure(output_backend="webgl")
pGasMap.image(image='image', dh=xSize, dw=ySize, x=0, y=0,color_mapper=color_mapper,  source=source) # fixed range color
#pGasMap.image(image='image', dh=xSize, dw=ySize, x=0, y=0,  source=source) #auto range color
#Plot style
pGasMap.xaxis.visible = False
pGasMap.xgrid.visible = False
pGasMap.yaxis.visible = False
pGasMap.ygrid.visible = False

pGasMap_N = figure(x_range=(0,xSize), y_range=(0,ySize),aspect_scale=1, output_backend="webgl") #
pGasMap_N = figure(title='Normalized Pattern', x_range=(0,xSize), y_range=(0,ySize),aspect_scale=1, output_backend="webgl", toolbar_location="below") #
pGasMap_N.title.text_font_size = "38px"
pGasMap_N.title.align = "center"
pGasMap_N.title.background_fill_color = "navajowhite" # "Coral"
pGasMap_N.title.text_color = "black"
pclrBar = figure(output_backend="webgl")
# pclrBar = figure(output_backend="webgl")
pGasMap_N.image(image='image', dh=xSize, dw=ySize, x=0, y=0,color_mapper=color_mapper_N,  source=source_N) # fixed range color
#pGasMap.image(image='image', dh=xSize, dw=ySize, x=0, y=0,  source=source) #auto range color
#Plot style
pGasMap_N.xaxis.visible = False
pGasMap_N.xgrid.visible = False
pGasMap_N.yaxis.visible = False
pGasMap_N.ygrid.visible = False

pSensor = []
rs = []
ds = []
numSensor = 3
singlePx = [0,0] #(X,Y)
title= ["Spectrum Histogram","1px Signal ("+str(singlePx[0])+','+str(singlePx[0])+')',"Normalized Spectrum Histogram"]
xLab = ["Resistance", "Resistance", "Resistance"]
yLab = ["Sensor ID", "Time (n)", "Sensor ID"]
yLim = [30, 10, 256]

for i in range(numSensor):
    if i != 1:
        pSensor.append(figure(title=title[i],y_range=(0,yLim[i]), plot_width=600, plot_height=210,  output_backend="webgl", toolbar_location="above")) #y_range=(-2, 2),
    else:
        pSensor.append(figure(title=title[i], plot_width=600, plot_height=210,  output_backend="webgl", toolbar_location="above"))
    pSensor[i].yaxis.axis_label = xLab[i]
    pSensor[i].xaxis.axis_label = yLab[i]
    rs.append(pSensor[i].line([], [], line_width=2))
    #ds.append(rs[i].data_source)
    #ds[i].data['y'] = np.linspace(0,xSize*ySize,xSize*ySize, dtype=int)
    #ds[i].data['x'] = np.linspace(0,xSize*ySize,xSize*ySize, dtype=int)

srcBar = ColumnDataSource(dict(x=np.linspace(0,xSize*ySize-1,xSize*ySize, dtype=int),top=np.ones((xSize*ySize,), dtype=int),))
srcBar_N = ColumnDataSource(dict(x=np.linspace(0,xSize*ySize-1,xSize*ySize, dtype=int),top=np.ones((xSize*ySize,), dtype=int),))

pSensor[0].vbar(x="x", top="top",
    width = .9,
    fill_alpha = .5,
    #fill_color = 'salmon',
    line_alpha = .5,
    #line_color='green',
    #line_dash='dashed',
    source=srcBar)
#pSensor[0].y_range = Range1d(start=-2, end=2)

pSensor[2].vbar(x="x", top="top",
    width = .9,
    fill_alpha = .5,
    #fill_color = 'salmon',
    line_alpha = .5,
    #line_color='green',
    #line_dash='dashed',
    source=srcBar_N)

dsT=rs[1].data_source
# global titles
@linear()
def update(step):
    global num_of_frameworks, sta_sec_BL, img_BL, start_second, titles
    now_second = int(time.time())
    # start_second = datetime.fromtimestamp(timestamp).strftime("%H:%M")
    #Image 
    #gasMap = np.random.rand(xSize,ySize) #np.zeros((xSize,ySize), dtype=float)
    imgStr = []
    barChart = []
    barChart_N = []
    rowID = 99
    initOK = False
    while(1):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind((UDP_IP, UDP_Port))
        data, addr = s.recvfrom(1024)
        RVec = data.split(b',')[1:]
        RVec = [float(R) for R in RVec]
        s.close()
        if int(data.split(b',')[0])==0:
            imgStr = []
            barChart = []
            barChart_N = []
            initOK = True

        if initOK==False:
            pass
        barChart=barChart+RVec
        imgStr.append(RVec)

        if len(imgStr)==ySize:
            break

    # get the normalized BL
    if now_second - start_second > sta_sec_BL:
        # titles = Div(text='<h1 style="text-align: center">target gases: apple</h1>')
        titles.text = '<h1 style="text-align: center">target gases: <b>apple</b></h1>'
        if num_of_frameworks != 0:
            if len(img_BL) == 0:
                img_BL = np.array(imgStr)
                # print('2', img_BL)
            else:
                img_BL = (img_BL+np.array(imgStr))/2
                # print('3', img_BL)
            num_of_frameworks -= 1
        else:
            barChart_N_tmp = (np.array(imgStr)-img_BL*0.9).flatten() / Norm_data
            value_max = np.max(barChart_N_tmp)
            value_min = np.min(barChart_N_tmp)
            # print('4',  value_max, value_min, (np.max(barChart_N_tmp) - np.min(barChart_N_tmp)), barChart_N_tmp)
            # print('6',(value_max - value_min)*(value_max-value_min)*255)
            # print('6',(8 - 6)*(8-6)*255)
            barChart_N = ( (barChart_N_tmp-value_min)*255/(value_max - value_min) ).tolist()
            # print('5', np.max(barChart_N), np.min(barChart_N), barChart_N)
            imgStr_N = np.matrix(np.array(barChart_N).reshape([10,10]), dtype='float')
            source_N.data = dict(image=[imgStr_N])

    if len(barChart_N) == 0:
        barChart_N = np.ones((100)).tolist()
        imgStr_N = np.matrix(np.ones((10,10)), dtype='float')
        source_N.data = dict(image=[imgStr_N])

    imgStr = np.matrix(imgStr, dtype='float')
    #2D plot
    source.data = dict(image=[imgStr])
    # 1D graph
    dsT.data['y'].append(imgStr[singlePx[0],singlePx[0]])
    dsT.data['x'].append(step)
    if len(dsT.data['y'])>60*5:
        dsT.data['y'].pop(0)
        dsT.data['x'].pop(0)
    dsT.trigger('data', dsT.data, dsT.data)
    # Bar chart
    srcBar.data['top'] = barChart
    srcBar_N.data['top'] = barChart_N
    '''
    #UDP socket
    for i in range(xSize): #UDP cannot support long data package
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind((UDP_IP, UDP_Port))
        data, addr = s.recvfrom(1024)
        s.close()
        rowID = int(data.split(b',')[0])
        if rowID!= i:
            print('not align yet..')
            return
        row = data.split(b',')[1:]
        #row = [(float(row[j])-imgMean[i*xSize+j])/imgStd[i*xSize+j] for j in range(ySize)]
        rowf = [float(row[j]) for j in range(xSize)]
        barChart = barChart+rowf
        imgStr.append(rowf)

    imgStr = np.matrix(imgStr, dtype='float')

    source.data = dict(image=[imgStr])
    # 1D graph
    dsT.data['y'].append(imgStr[0,0])
    dsT.data['x'].append(step)
    if len(dsT.data['y'])>60*5:
        dsT.data['y'].pop(0)
        dsT.data['x'].pop(0)
    dsT.trigger('data', dsT.data, dsT.data)
    #Bar chart
    srcBar.data['top'] = barChart
    '''

#show(pGasMap)
glyph = Text(x="x", y="y", text="text", angle=0.3, text_color="#96deb3")

ptot = column(row(pGasMap,column(pSensor[0], pSensor[1], pSensor[2]), pGasMap_N), titles)
# ptot = column(titles,row(pSensor[0],pSensor[1]))
curdoc().add_root(ptot)
curdoc().add_periodic_callback(update, 100)
