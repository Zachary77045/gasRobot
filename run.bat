@echo off 
ECHO ********************************************************
ECHO FAN Group Sensor Group, HKUST, 2021
ECHO Gas Matrix Project, stm32
ECHO Project Author:    Vinson
ECHO Programming:       Jonathan
ECHO Hardware:          Andrew
ECHO ********************************************************
@echo on

set comPort=3
set /a "servPort=9800+%comPort%"
call D:\software\Anaconda\Scripts\activate.bat
call activate GasDataPro
start bokeh serve --show gasMatrixPlot_models.py --port %servPort% --args %comPort%
start python gasMatrixStm32.py %comPort% 

PAUSE

