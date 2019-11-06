#!/user/bin/env python
'''
	File name: .py
	Author: Chris Harris and Harrison Huang
	Date created: 07/13/18
	Date last modified: 10/03/18
	Python Version: 2.7

	changed motor movement send bit
	implemented graph function
		mrn and endpoint inputs
		select range
		sort, average, plot
		export image
'''

#TODO switch the bit to flip the "programming running" bit off

import sys
import cv2
import time
import pypylon.pylon as py
import numpy as np
import os
import minimalmodbus
import serial
import csv
import datetime
import shutil
import pandas
import random
import glob
import subprocess


#from fastai.vision import *

from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import cv2
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from threading import Thread
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.uic import loadUi

AUTO_DETECTION_ENABLED = True
LOAD_SETUP_ON_START = True
PLC_PSUEDO_ENABLED = False
CAMERA_ENABLED = True
FULLSCREEN = False
MENUBAR_ENABLED = False
STATUSBAR_ENABLED = False

UI_FILE = 'mainwindow6.ui'

DATETIME = ""
DATE = ""
TIME = ""
MRN = ""

FONT = QFont()
FONT.setPointSize(14)

BTN_STYLESHEET = {"transparent": "QPushButton {background: transparent}", "red toggle": "QPushButton {background: transparent}" + 
				"QPushButton:checked {background-color: red; border-style: outset; border-width: 2px; border-radius 100px; border-color: beige}"}

BTN_STYLESHEET_START = "QPushButton {background: #00FF00; border-color: gray; border-style: solid; border-width: 1px}"

BTN_STYLESHEET_STOP = "QPushButton {background: red; border-color: gray; border-style: solid; border-width: 1px}"

BTN_STYLESHEET_RESET = "QPushButton {background: orange; border-color: gray; border-style: solid; border-width: 1px}"

LABEL_STYLESHEET = {"disabled" : "QLabel {background-color: gray}", "on" : "QLabel {background-color: green}", "off" : "QLabel {background-color: red}"}

SCREEN_RESOLUTION = (768,1366)
IMAGE_RESOLUTION = {'y': 3036, 'x': 4024}

ADDRESS_REF = { 1010: "rot axis speed",
		1012: "rot axis acceleration",
		1014: "rot axis direction",
		1000: "tilt axis speed",
		1002: "tilt axis acceleration",
		1004: "tilt axis up degrees",
		1006: "tilt axis down degrees",
		1008: "tilt axis plane offset",
		1020: "set temperature",
		1022: "temperature read",
		1030: "white light",
		1032: "red light", 	#TODO
		1034: "green light",	#TODO
		1036: "blue light",	#TODO
		2290: "tilt axis jog step",
		2291: "tilt axis jog dir",
		2294: "rot axis jog step",
		2295: "rot axis jog direction",
		2297: "momentary",
		2299: "test toggle", #added this on maintenance page R15B.
		2304: "drawer jog step",
		2305: "drawer jog dir",
		2309: "carriage is homed",
		2310: "home drawer", 
		2311: "buzzer",
		2312: "home drawer finish", #send after 250ms to terminate homing
		2371: "Load Drawer", #made this change for rbit drawer loading
		2372: "Timer On and start test",	#added this to send timer on signal to plc
		2373: "Drawer open", #not hitting the limit switch on drawer
		2374: "Drawer in motion",#when 1, drawer is in motion
		2377: "start program", #send when program starts
		2378: "heart beat signal", #when the program reboots
		
		
} 


		 
ROOT_DIR = "/home/bha/Desktop/"


class MyApp(QMainWindow):
	def __init__(self):
		super(MyApp, self).__init__()
		loadUi(UI_FILE,self)
		updateDateTime()
		self.updateMRN()
		if FULLSCREEN:
			self.showFullScreen()
		if STATUSBAR_ENABLED:
			self.statusBar = QStatusBar()
			self.setStatusBar(self.statusBar)
			self.statusBar.showMessage("System Status | Normal")
		self.generalSetup()		
		self.dataTableSetup()
		if MENUBAR_ENABLED:
			self.menuSetup()
		self.cameraSetup()
		self.timerSetup()
		self.loggerSetup()
		self.displaySetup()
		self.labTechSetup()
		self.hardwareSetup()
		self.configSetup()
		self.pollSetup()
		self.maintenanceSetup()
		self.keyboardSetup()
		self.graphSetup()	
		self.heartbeatSetup()
		
#initial homing check
#------------------------------------------------------------------------------------------------

	def heartbeatSetup(self):
		#program start
		self.plc.send(2377, 1, "rbit");
		QTimer.singleShot(200, lambda : self.plc.send(2377, 0, "rbit"))

		self.heartbeat = False
		self.heartbeatTimer = QTimer()
		self.heartbeatTimer.timeout.connect(self.heartbeatPulse)
		self.heartbeatTimer.start(436) # random

	def heartbeatPulse(self):
		if self.heartbeat == True:
			self.plc.send(2378, 1, "rbit", priority = False)
			self.heartbeat = False
		else:
			self.plc.send(2378, 0, "rbit", priority = False)
			self.heartbeat = True
			
		
#graph
#------------------------------------------------------------------------------------------------

	def graphSetup(self):
		self.graphSearchParamMRN = ["","",""]
		self.graphSearchParamEndPoint = ["","",""]
		self.graphSearchParamMRNStruct = [self.editExpMedNumGraph, self.editExpMedNumGraph_2, self.editExpMedNumGraph_3]
		self.graphSearchParamEndpointStruct = [self.comboBoxGraph, self.comboBoxGraph_2, self.comboBoxGraph_3]
		self.graphFileRange = []
		self.btnChooseFiles.clicked.connect(self.getFileRange)
		self.btnPlot.clicked.connect(self.startPlot)
		self.btnExportGraph.clicked.connect(self.exportGraphCall)
		self.plot({},{}) #blank

	def getFileRange(self):
		self.graphFileRange = []
		fileDialog = FileDialog()
		fileDialog.exec_()
		if fileDialog.result():
			for path in list(fileDialog.selectedFiles()):
				result = glob.glob("{}/*[0-9].csv".format(path))
				if result:
					csv = result[0]
					self.graphFileRange.append(csv)
	
	def exportGraphCall(self):
		self.temp = InputDialogKeyboard("Name of Graph")
		self.temp.show()
		self.temp.move(600,200)
		self.temp.returnMsg.connect(self.exportGraphReturn)

	def exportGraphReturn(self, string):
		shutil.copy('graph.png', 
				ROOT_DIR+"graphs/{}.png".format(string))

	def startPlot(self):
		self.updateSearchParam()
		data = self.findData()
		formattedData = self.formattedData(data)
		averagedData = self.averageData(formattedData)
		unAveragedData = self.unAveragedData(formattedData)
		alphaDataAvg = self.alphaData(averagedData)
		alphaDataUnAvg = self.alphaData(unAveragedData)
		self.plot(alphaDataAvg, alphaDataUnAvg)

	def updateSearchParam(self):
		for i, mrnTextField in enumerate(self.graphSearchParamMRNStruct):
			if str(mrnTextField.text()).isdigit():
				self.graphSearchParamMRN[i] = str(mrnTextField.text())
			else:
				self.graphSearchParamMRN[i] = ""
		for i, endpointComboBox in enumerate(self.graphSearchParamEndpointStruct):
			if endpointComboBox.currentIndex():
				self.graphSearchParamEndPoint[i] = str(endpointComboBox.currentText())
			else:
				self.graphSearchParamEndPoint[i] = ""
		print("MSG	mrn selected: {}".format(self.graphSearchParamMRN))
		print("MSG	endpoint selected: {}".format(self.graphSearchParamEndPoint))

	def findData(self):
		nameToCSVindexDict = {"MRN": 0, "Blood": 5, "Clump" : 8, "Clump Stick" : 11, "Clot": 14, "4th": 17, "5th": 21}
		out = []
		for csv in self.graphFileRange:
			with open(csv, "r") as csvFile:
				next(csvFile) #skip header
				data = next(csvFile).split(',')
				dataMrn = data[nameToCSVindexDict["MRN"]]
				dataBlood = data[nameToCSVindexDict["Blood"]]
				for mrn, endpoint in zip(self.graphSearchParamMRN, self.graphSearchParamEndPoint):
					if mrn == dataMrn and endpoint:
						dataTime = data[nameToCSVindexDict[endpoint]]
						out.append((mrn, endpoint, dataBlood, dataTime))				
		print("MSG	data found: {}".format(out))
		return out

	def formattedData(self, data):
		#format data
		#data structure: {"label": {blood: [times, ...]}}
		formattedData = {}
		for mrn, endpoint, blood, time in data:
			if (time == "-"):
				time = 0
			label = "{} {}".format(mrn, endpoint)
			if label in formattedData.keys():
				if int(blood) in formattedData[label].keys():
					formattedData[label][int(blood)].append(float(time))
				else:
					formattedData[label][int(blood)] = [float(time)]
			else:
				formattedData[label] = {int(blood): [float(time)]}
		
		print("MSG	formatted data: {}".format(formattedData))
		return formattedData
	
	def averageData(self, data):
		#average duplicated
		#data structure: {"label": [(blood,times), ...]}
		averagedData = {}
		for label, value in data.items():
			for blo, timeArr in value.items():
				if label in averagedData.keys():
					averagedData[label].append((blo,sum(timeArr)/len(timeArr)))
				else:
					averagedData[label] = [(blo,sum(timeArr)/len(timeArr))]
		print(averagedData)
		print("MSG	averaged data: {}".format(averagedData))
		return averagedData

	def unAveragedData(self, data):
		#unAveraged duplicated
		#data structure: {"label": [(blood,times), ...]}
		unAveragedData = {}
		for label, value in data.items():
			for blo, timeArr in value.items():
				for time in timeArr:
					if label in unAveragedData.keys():
						unAveragedData[label].append((blo, time))
					else:
						unAveragedData[label] = [(blo,time)]
		print("MSG	un-averaged data: {}".format(unAveragedData))
		return unAveragedData

	def alphaData(self, data):
		#alpha
		#data structure: {"label": [[blood,...],[times,...]]}
		alphaData = {}
		for key, value in data.items():
			sortedVal = sorted(value, key=lambda x: x[0])
			alphaData[key] = [[x for x, y in sortedVal],[y for x, y in sortedVal]]
		print("MSG	alpha data: {}".format(alphaData))
		return alphaData
	
	def plot(self, alphaDataAvg, alphaDataUnAvg):
		colorLine = ["b-", "g-", "r-"]
		colorDot  = ["bo", "go", "ro"]
		plt.clf()
		plt.ylabel('Endpoint Times')
		plt.xlabel('Time since blood drawn')
		lines = []
		for i, (key, value) in enumerate(alphaDataAvg.items()):
			print(value)
			line, = plt.plot(value[0], value[1], colorLine[i], label=key)
			lines.append(line)
		for i, (key, value) in enumerate(alphaDataUnAvg.items()):
			print(value)
			plt.plot(value[0], value[1], colorDot[i], label=key)
		plt.legend(loc='lower right')
		plt.savefig('graph.png')
		pixmap = QPixmap('graph.png')
   		self.labGraph.setPixmap(pixmap)
   		self.labGraph.show()
		print("MSG	graph plotted")

#------------------------------------------------------------------------------------------------


#general setup
#------------------------------------------------------------------------------------------------

	def generalSetup(self):

		#update GUI time
		self.editExpDate.setText("{}".format(DATE))

		#start up on main tab
		self.tabConfig.setCurrentIndex(0)

		#make folders
		createFolder("pics", deleteExisting=True)
		createFolder(ROOT_DIR+"tests")
		createFolder("usb_EMULATED")
		createFolder(ROOT_DIR+"graphs")
		createFolder("temp", deleteExisting=True)
		
		#jog (NEW)
		self.TiltAxis_CCW.pressed.connect(lambda : self.plc.sendMulti([2291], [1], "rbit"))
		self.TiltAxis_CW.pressed.connect(lambda: self.plc.sendMulti([2290], [1], "rbit"))
		self.Rotation_CCW.pressed.connect(lambda: self.plc.sendMulti([2295], [1], "rbit"))
		self.Rotation_CW.pressed.connect(lambda: self.plc.sendMulti([2294], [1], "rbit"))

		#jog (OLD)
		"""
		self.TiltAxis_CCW.pressed.connect(lambda : self.plc.sendMulti([2290,2291], [1,0], "rbit"))
		self.TiltAxis_CW.pressed.connect(lambda: self.plc.sendMulti([2290,2291], [1,1], "rbit"))
		self.Rotation_CCW.pressed.connect(lambda: self.plc.sendMulti([2294,2295], [1,0], "rbit"))
		self.Rotation_CW.pressed.connect(lambda: self.plc.sendMulti([2294,2295], [1,1], "rbit"))
		"""
		self.TiltAxis_CCW.released.connect(lambda: self.plc.sendMulti([2290,2291], [0,0], "rbit"))
		self.TiltAxis_CW.released.connect(lambda: self.plc.sendMulti([2290,2291], [0,0], "rbit"))
		self.Rotation_CCW.released.connect(lambda: self.plc.sendMulti([2294,2295], [0,0], "rbit"))
		self.Rotation_CW.released.connect(lambda: self.plc.sendMulti([2294,2295], [0,0], "rbit"))

		#momentary
		self.btnMoment.pressed.connect(lambda: self.plc.send(2297, 1, "rbit"))
		self.btnMoment.released.connect(lambda: self.plc.send(2297, 0, "rbit"))

		#drawer
		self.btnLoad.clicked.connect(self.drawer)

		#drawer maintenance
		#jog (NEW)
		self.DrawerEject.pressed.connect(lambda : self.plc.sendMulti([2305], [1], "rbit"))
		self.DrawerLoad.pressed.connect(lambda: self.plc.sendMulti([2304], [1], "rbit"))
		self.DrawerEject.released.connect(lambda: self.plc.sendMulti([2304,2305], [0,0], "rbit"))
		self.DrawerLoad.released.connect(lambda: self.plc.sendMulti([2304,2305], [0,0], "rbit"))

		#self.DrawerHome.released.connect(lambda : self.plc.send(2310, 1, "rbit"))

		#fullscreen and hide
		self.btnFullscreen.clicked.connect(self.showFullScreen)
		self.btnWindowed.clicked.connect(self.showNormal)
		self.btnHide.clicked.connect(self.showMinimized)

		#close program 
		#TODO switch the bit to flip the "programming running" bit off
		self.btnCloseProgram.clicked.connect(lambda: quit() if Message("Exiting Program", "Are you sure?").run()
 else None)
		
		
		#shutdown
		self.btnShutdown.clicked.connect(lambda: os.system('sudo shutdown now -h') if Message("Shutting down", "Are you sure?").run()
 else None)
		self.btnShutdown_2.clicked.connect(lambda: os.system('sudo shutdown now -h') if Message("Shutting down", "Are you sure?").run() else None)

		#Test toggle button
		self.Test_button.setCheckable(True)
		#if self.Test_button.isChecked():
		#	self.plc.sendMulti([2299], [1], "rbit")
		#else:
		#	self.plc.sendMulti([2299, 2300], [0,0], "rbit")
		self.Test_button.toggle()
		self.Test_button.pressed.connect(lambda: self.plc.sendMulti([2299], [1], "rbit"))
		self.Test_button.released.connect(lambda: self.plc.sendMulti([2299, 2300], [0,0], "rbit"))

#------------------------------------------------------------------------------------------------


#keyboard
#------------------------------------------------------------------------------------------------

	def keyboardSetup(self):
		#for config
		self.btnMRN.clicked.connect(lambda _: self.showKeyboard(_, self.editExpMedNum, self.editExpMedNum.text(), "MRN"))
		self.btnMRN.setStyleSheet(BTN_STYLESHEET["transparent"])
		
		self.btnExpDate.clicked.connect(lambda _: self.showKeyboard(_, self.editExpDate, self.editExpDate.text(), "Date"))
		self.btnExpDate.setStyleSheet(BTN_STYLESHEET["transparent"])
		
		self.btnExpTestType.clicked.connect(lambda _: self.showKeyboard(_, self.editExpTestType, self.editExpTestType.text(), "Test Type"))
		self.btnExpTestType.setStyleSheet(BTN_STYLESHEET["transparent"])
		
		self.btnExpTimeBloodDraw.clicked.connect(lambda _: self.showKeyboard(_, self.editExpTimeBloodDraw, self.editExpTimeBloodDraw.text(), "Time of Blood Draw"))
		self.btnExpTimeBloodDraw.setStyleSheet(BTN_STYLESHEET["transparent"])
		
		self.btnExpComment.clicked.connect(lambda _: self.showKeyboard(_, self.editExpComment, self.editExpComment.text(), "Comments"))
		self.btnExpComment.setStyleSheet(BTN_STYLESHEET["transparent"])
		
		#for graph
		self.btnMRNGraph.clicked.connect(lambda _: self.showKeyboard(_, self.editExpMedNumGraph, self.editExpMedNumGraph.text(), "MRN Line 1"))
		self.btnMRNGraph.setStyleSheet(BTN_STYLESHEET["transparent"])
		
		self.btnMRNGraph_2.clicked.connect(lambda _: self.showKeyboard(_, self.editExpMedNumGraph_2, self.editExpMedNumGraph_2.text(), "MRN Line 2"))
		self.btnMRNGraph_2.setStyleSheet(BTN_STYLESHEET["transparent"])
		
		self.btnMRNGraph_3.clicked.connect(lambda _: self.showKeyboard(_, self.editExpMedNumGraph_3, self.editExpMedNumGraph_3.text(), "MRN Line 3"))
		self.btnMRNGraph_3.setStyleSheet(BTN_STYLESHEET["transparent"])
		
	def showKeyboard(self, _, label, text, title):
		keyboard = VirtualKeyboard(label, text, title)
    		keyboard.show()

#------------------------------------------------------------------------------------------------
	

#poll
#---------------------------------------------------------------------------------------

	def pollSetup(self):
		#{address: {"labels": [array of labels on interface], "decimal": if needed, "value": last val}} 
		self.pollDict = {	1022: {	"labels" : [self.labTempGet, self.labTempGet_2, self.labTempGet_3],
						"decimal" : True, "value": -1}}
		self.IO_Dict = {'xbit': [{"value" : -1, "label": self.X_0_label_2}, 
				{"value" : -1, "label": self.X_1_label_2}, 
				{"value" : -1, "label": self.X_2_label_2}, 
				{"value" : -1, "label": self.X_3_label_2}, 
				{"value" : -1, "label": self.X_4_label_2}, 
				{"value" : -1, "label": self.X_5_label_2}, 
				{"value" : -1, "label": self.X_6_label_2}, 
				{"value" : -1, "label": self.X_7_label_2}, 
				{"value" : -1, "label": self.X_8_label_2}, 
				{"value" : -1, "label": self.X_9_label_2}, 
				{"value" : -1, "label": self.X_10_label_2}, 
				{"value" : -1, "label": self.X_11_label_2}, 
				{"value" : -1, "label": self.X_12_label_2}, 
				{"value" : -1, "label": self.X_13_label_2}, 
				{"value" : -1, "label": self.X_14_label_2}, 
				{"value" : -1, "label": self.X_15_label_2}],
				'ybit': [{"value" : -1, "label": self.Y_0_label_2}, 
				{"value" : -1, "label": self.Y_1_label_2}, 
				{"value" : -1, "label": self.Y_2_label_2}, 
				{"value" : -1, "label": self.Y_3_label_2}, 
				{"value" : -1, "label": self.Y_4_label_2}, 
				{"value" : -1, "label": self.Y_5_label_2}, 
				{"value" : -1, "label": self.Y_6_label_2}, 
				{"value" : -1, "label": self.Y_7_label_2}, 
				{"value" : -1, "label": self.Y_8_label_2}, 
				{"value" : -1, "label": self.Y_9_label_2}, 
				{"value" : -1, "label": self.Y_10_label_2}, 
				{"value" : -1, "label": self.Y_11_label_2}, 
				{"value" : -1, "label": self.Y_12_label_2}, 
				{"value" : -1, "label": self.Y_13_label_2}, 
				{"value" : -1, "label": self.Y_14_label_2}, 
				{"value" : -1, "label": self.Y_15_label_2}]}
		self.polling = Polling(self.plc, self.pollDict, self.IO_Dict)
		self.polling.pollSignal.connect(self.pollToInter)
		self.polling.start()

	#called when new value from poll is found
	def pollToInter(self, address, value, type):
		if type == "DT":
			for label in self.pollDict[address]["labels"]:
				if self.pollDict[address]["decimal"]:
					label.setText(str(value / 10.0))
				else:
					label.setText(str(value))
		elif type == "xbit":
			if value:
				self.IO_Dict[str(type)][address]["label"].setStyleSheet(LABEL_STYLESHEET['on'])
			else:
				self.IO_Dict[str(type)][address]["label"].setStyleSheet(LABEL_STYLESHEET['off'])
		elif type == "ybit":
			if value:
				self.IO_Dict[str(type)][address]["label"].setStyleSheet(LABEL_STYLESHEET['on'])
			else:
				self.IO_Dict[str(type)][address]["label"].setStyleSheet(LABEL_STYLESHEET['off'])

#---------------------------------------------------------------------------------------


#maintenance
#---------------------------------------------------------------------------------------

	def maintenanceSetup(self):
		self.Maintenance_Enab.clicked.connect(lambda _: self.polling.maintenanceToggle(_, True))
		self.Maintenance_Disab.clicked.connect(self.disableMaintenance)
		self.disableMaintenance()
		
	def disableMaintenance(self):
		for bitType, valueLabelArr in self.IO_Dict.items():
			for valueLabel in valueLabelArr:
				valueLabel["label"].setStyleSheet(LABEL_STYLESHEET['disabled'])
		self.polling.maintenanceToggle('', False)

#---------------------------------------------------------------------------------------
	
	
#dataTable
#---------------------------------------------------------------------------------------

	def dataTableSetup(self):		
		self.btnExportHighlighted.clicked.connect(self._exportHighlighted)
		self.btnImportCSV.clicked.connect(self._importFiles)
		
		#self.columns also used by export
		self.columns = [
			"MRN","Date","Time","Lab Tech",
			"Test Type","Blood Draw Time",
			"Manual Timer 1 Tube 1","Manual Timer 1 Tube 2","Manual Timer 1 Average",
			"Manual Timer 2 Tube 1","Manual Timer 2 Tube 2","Manual Timer 2 Average",
			"Manual Timer 3 Tube 1","Manual Timer 3 Tube 2","Manual Timer 3 Average",
			"Manual Timer 4 Tube 1","Manual Timer 4 Tube 2","Manual Timer 4 Average",
			"Manual Timer 5 Tube 1","Manual Timer 5 Tube 2","Manual Timer 5 Average",
			"Auto Timer 1 Tube 1","Auto Timer 1 Tube 2","Timer 1 Average",
			"Auto Timer 2 Tube 1","Auto Timer 2 Tube 2","Timer 2 Average",
			"Auto Timer 3 Tube 1","Auto Timer 3 Tube 2","Timer 3 Average",
			"Tube 1 Excluded 1","Tube 1 Excluded 2","Tube 1 Excluded 3","Tube 1 Excluded 4","Tube 1 Excluded 5",
			"Tube 2 Excluded 1","Tube 2 Excluded 2","Tube 2 Excluded 3","Tube 2 Excluded 4","Tube 2 Excluded 5",
			"Comments"]
		self.columnsCondensed= [
			"MRN","Date","Time","Lab Tech",
			"Test Type","Blood Draw Time",
			"M T1 TT1","M T1 TT2","M T1 A",
			"M T2 TT1","M T2 TT2","M T2 A",
			"M T3 TT1","M T3 TT2","M T3 A",
			"M T4 TT1","M T4 TT2","M T4 A",
			"M T5 TT1","M T5 TT2","M T5 A",
			"A T1 TT1","A T1 TT2","A T1 A",
			"A T2 TT1","A T2 TT2","A T2 A",
			"A T3 TT1","A T3 TT2","A T3 A",
			"TT1 Excluded 1","TT1 Excluded 2","TT1 Excluded 3","TT1 Excluded 4","TT1 Excluded 5",
			"TT2 Excluded 1","TT2 Excluded 2","TT2 Excluded 3","TT2 Excluded 4","TT2 Excluded 5",
			"Comments"]

		#tableView setup
		self.dfData = pandas.DataFrame(columns=self.columns)
		self.proxySortModel = QSortFilterProxyModel()
		self.proxySortModel.setSourceModel(PandasModel(self.dfData))
		self.tableData.setModel(self.proxySortModel)
		self.tableData.setSortingEnabled(True)

		#auto import on startup
		csvList = [ROOT_DIR+"tests/{}".format(folder) for folder in os.listdir(ROOT_DIR+"tests")]
		self.importCSVtoDF(csvList)

	#called by export button
	def _exportHighlighted(self):
		dirName = str(QFileDialog.getExistingDirectory(self, "Select Directory to save to"))
		if not dirName:
			return #terminate if no selection
		sourceIndexed = self.tableData.selectionModel().selectedRows()
		existedCounter = 0
		for sourceIndex in sourceIndexed:
			row = sourceIndex.row()
			mrn  = self.proxySortModel.data(self.proxySortModel.index(row, 0)).toString()
			date = self.proxySortModel.data(self.proxySortModel.index(row, 1)).toString()
			time = self.proxySortModel.data(self.proxySortModel.index(row, 2)).toString()
			name = testLookUp(mrn, date, time) #finds folder via information from csv			
			if name:
				#copies only if not pre-existing
				if os.path.isdir("{}/{}".format(dirName, name)): 
					existedCounter+=1
					print("MSG	test {} is already in the folder".format(name))
				else:
					shutil.copytree(ROOT_DIR+"tests/{}".format(name), 
					"{}/{}".format(dirName, name))
		print("MSG	Exported {} new test(s) into USB, {} already exported".format((len(sourceIndexed)-existedCounter), existedCounter))
	
	#called by import button
	def _importFiles(self):
		importDialog = FileDialog()
		importDialog.exec_()
		if importDialog.result(): #if clicked choose
			#move file to tests folder
			selectedFilesList = []
			for path in list(importDialog.selectedFiles()):
				if not os.path.isdir(ROOT_DIR+"tests/{}".format(str(path).split('/')[-1])):
					shutil.copytree(str(path), ROOT_DIR+"tests/{}".format(str(path).split('/')[-1]))
					selectedFilesList.append(path)
			self.importCSVtoDF(selectedFilesList)

	#called when test end, program startup, import clicked
	#takes in the folder/folder list, will auto find csv
	def importCSVtoDF(self, fnameList):
		print("MSG \t adding csv to DF")
		nulled = 0
		for fname in fnameList:
			fname = fname+"/"+fname.split("/")[-1]+".csv" #folder -> csv file
			try:
				data = pandas.read_csv(str(fname), nrows=1, dtype = {"MRN":str})
				self.dfData = self.dfData.append(data, sort=False)
			except:
				print("ERROR\tPandas importCSVtoDF failed")
		#print("ERROR\tfname:{}\tdata:{}".format(fname, data))
		print("MSG	Imported {} test(s) to table".format(len(fnameList) - nulled))
		self.proxySortModel.setSourceModel(PandasModel(self.dfData))

#---------------------------------------------------------------------------------------

	
#menu setup
#---------------------------------------------------------------------------------------

	def menuSetup(self):
		menuBar = QMenuBar()
		menuBar.setNativeMenuBar(False)
		self.setMenuBar(menuBar)

		rightMenuBar = QMenuBar()

		closeAction = QAction("&Close", self)
        	closeAction.triggered.connect(sys.exit)
		rightMenuBar.addAction(closeAction)
		
    		menuBar.setCornerWidget(rightMenuBar, Qt.TopRightCorner)
	
#---------------------------------------------------------------------------------------


#timer
#---------------------------------------------------------------------------------------

	def timerSetup(self):
		#timer1
		self.timer1 = Timer(self.labTimer1, self.btnTimer1Start, self.btnTimer1Stop)
		self.btnTimer1Start.clicked.connect(self.timer1.startTimer)
		self.btnTimer1Start.setStyleSheet(BTN_STYLESHEET_START)
		self.btnTimer1Stop.clicked.connect(self.timer1.stopTimer)
		self.btnTimer1Stop.setStyleSheet(BTN_STYLESHEET_STOP)
		
		if CAMERA_ENABLED:
			self.btnTimer1Start.clicked.connect(lambda _: self._startTesting(_, 0))
			self.btnTimer1Stop.clicked.connect(lambda _: self.camera.stopSaveThread(_, 0))

		#timer2
		self.timer2 = Timer(self.labTimer2, self.btnTimer2Start, self.btnTimer2Stop)
		self.btnTimer2Start.clicked.connect(self.timer2.startTimer)
		self.btnTimer2Start.setStyleSheet(BTN_STYLESHEET_START)
		self.btnTimer2Stop.clicked.connect(self.timer2.stopTimer)
		self.btnTimer2Stop.setStyleSheet(BTN_STYLESHEET_STOP)
		if CAMERA_ENABLED:
			self.btnTimer2Start.clicked.connect(lambda _: self._startTesting(_, 1))
			self.btnTimer2Stop.clicked.connect(lambda _: self.camera.stopSaveThread(_, 1))
			
		#beeping
		self.btnTimer1Start.pressed.connect(lambda : self.plc.send(2311, 1, type="rbit"))
		self.btnTimer1Start.released.connect(lambda : self.plc.send(2311, 0, type="rbit"))
		self.btnTimer2Start.pressed.connect(lambda : self.plc.send(2311, 1, type="rbit"))
		self.btnTimer2Start.released.connect(lambda : self.plc.send(2311, 0, type="rbit"))
		self.btnTimer1Stop.pressed.connect(lambda : self.plc.send(2311, 1, type="rbit"))
		self.btnTimer1Stop.released.connect(lambda : self.plc.send(2311, 0, type="rbit"))
		self.btnTimer2Stop.pressed.connect(lambda : self.plc.send(2311, 1, type="rbit"))
		self.btnTimer2Stop.released.connect(lambda : self.plc.send(2311, 0, type="rbit"))

		self.timerArr = [self.timer1, self.timer2]
	
	#called by timer buttons
	def _startTesting(self, _, index):
		print("sdkfjlasdjf ;lajfl;asdjf sf;asf")
		#check for pre-existing starts, will only start if no timer started
		if not any(self.camera.timerState):
			#update globals
			updateDateTime()
			self.updateMRN()
			createFolder(ROOT_DIR+"tests/{}_{}".format(MRN, DATETIME))
			self.camera.startThread("", index)
			self.plc.send(2372, 1, type="rbit") # added to send timer on signal to plc
			self.btnTimer1Stop.setStyleSheet(BTN_STYLESHEET_STOP)
			self.btnTimer1Stop.setText("Stop")
			self.btnTimer2Stop.setStyleSheet(BTN_STYLESHEET_STOP)
			self.btnTimer2Stop.setText("Stop")
		self.camera.timerState[index] = True
		self.plc.send(2372, 1, type="rbit")
		self.camera.startThread("", index)
		
	#should be called everytime the timer is started
	def updateMRN(self):
		global MRN
		MRN = self.editExpMedNum.text()
		print("MSG	MRN updated to: {}".format(MRN))
		
#---------------------------------------------------------------------------------------


#Lab tech
#---------------------------------------------------------------------------------------

	def labTechSetup(self):
		self.readLabTech()
		self.btnAddLabTech.clicked.connect(self.addLabTechCall)
		self.btnRemoveLabTech.clicked.connect(self.removeLabTech)	

	def readLabTech(self):
		with open("labTechs.csv", "r") as labTechNames:
			self.labTechNamesArr = labTechNames.read().strip().split(',')
			self.editExpOperName.clear()
			self.editExpOperName.addItems(self.labTechNamesArr)

	#attached to button
	def addLabTechCall(self):
		self.temp = InputDialogKeyboard("Add a Lab Tech")
		self.temp.show()
		self.temp.move(600,200)
		self.temp.returnMsg.connect(self.addLabTech)

	#called by signal
	def addLabTech(self, name):
		self.labTechNamesArr.append(name)
		with open("labTechs.csv", mode='w') as namesFile:
			namesWriter = csv.writer(namesFile,
				delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			namesWriter.writerow(self.labTechNamesArr)
		self.readLabTech()

	def removeLabTech(self):
		confirm, name = ComboBoxDialog("Removing a Lab Tech",self.labTechNamesArr).run()
		if confirm:
			self.labTechNamesArr.remove(name)
			with open("labTechs.csv", mode='w') as namesFile:
				namesWriter = csv.writer(namesFile,
					delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
				namesWriter.writerow(self.labTechNamesArr)
			self.readLabTech()
		
#---------------------------------------------------------------------------------------

	
#generate test files
#---------------------------------------------------------------------------------------

	#called by signal from camera class
	#creates csv and pics
	def generateTest(self):
		self.plc.send(2372, 0, type="rbit") #added for timer off signal to plc
		manLogBtns = self.logManTimerArr
		autoLogBtns = self.logAutoTimerArr
		fileName = "{}_{}".format(MRN, DATETIME)

		#data structure of endpoint times
		#[[0,1,2,3,4],[0,1,2,3,4]]
		manLogTimes = [[str(manLogBtns[0][0].text()),str(manLogBtns[0][1].text()),str(manLogBtns[0][2].text()),
				str(manLogBtns[0][3].text()),str(manLogBtns[0][4].text())
				],
				[str(manLogBtns[1][0].text()),str(manLogBtns[1][1].text()),str(manLogBtns[1][2].text()),
				str(manLogBtns[1][3].text()),str(manLogBtns[1][4].text())]]
		manLogTimesAvg = ["-" for i in range(len(manLogTimes[0]))]
		autoLogTimes = [[str(autoLogBtns[0][0].text()),str(autoLogBtns[0][1].text()),str(autoLogBtns[0][2].text()),
				str(autoLogBtns[0][3].text())],
				[str(autoLogBtns[1][0].text()),str(autoLogBtns[1][1].text()),str(autoLogBtns[1][2].text()),
				str(autoLogBtns[1][3].text())]]
		autoLogTimesAvg = ["-" for i in range(len(autoLogTimes[0]))]

		#toss values and calc avgs
		manLogTimes, manLogTimesAvg, manTossed = self._handleTossedValue(manLogBtns, manLogTimes, manLogTimesAvg)
		manLogTimes, manLogTimesAvg = self._calcAvgs(manLogTimes, manLogTimesAvg)
		autoLogTimes, autoLogTimesAvg = self._calcAvgs(autoLogTimes, autoLogTimesAvg)

		#export
		self._generateCSV(self.columns, MRN, self.editExpOperName.currentText(),
				self.editExpTestType.text(),self.editExpTimeBloodDraw.text(), self.editExpComment.text(),
				manLogTimes, manLogTimesAvg, manTossed, autoLogTimes, autoLogTimesAvg)
		self._generateLoggedPics(self.pictureLog, time, "Man")
		self._generateLoggedPics(self.pictureLogAuto, time, "Auto")

		#add to table
		#self.importCSVtoDF([ROOT_DIR+"tests/{}_{}/{}.csv".format(MRN, DATETIME,fileName)])
		self.importCSVtoDF([ROOT_DIR+"tests/{}_{}".format(MRN, DATETIME)])

	def _handleTossedValue(self, manLogBtns, manLogTimes, manLogTimesAvg):
		#handle tossed values
		manTossed = [["-" for j in range(len(manLogTimes[i]))] for i in range(len(manLogTimes))]
		manTossedLocation = [(i,j) for i in range(len(manLogBtns)) for j in range(len(manLogBtns[i])) if manLogBtns[i][j].isChecked()]
		for i,j in reversed(manTossedLocation):
			manTossed[i].insert(0,manLogTimes[i].pop(j))
			manTossed[i].pop()
			manLogTimes[i].append("-")
		return (manLogTimes, manLogTimesAvg, manTossed)

	def _calcAvgs(self, array, avgs):
		#calc avgs if available
		for i in range(len(max(array[0],array[1]))):
			if array[0][i] != '-' and array[1][i] != '-':
				#a+b/2
				avgs[i] = tenthsToDisplay(int(round((displayToTenths(array[0][i]) + displayToTenths(array[1][i]))/2.0)))
			elif array[0][i] != '-':
				avgs[i] = array[0][i]
			elif array[1][i] != '-':
				avgs[i] = array[1][i]
			#default timerManAvg[i] = '-'
		return (array, avgs)

	def _generateCSV(self, headers, MRN, labTech, testType, bloodTime, comment, timerManArr, timerManAvg, timerManCancelled, timerAutoArr, timerAutoAvg):
		with open(ROOT_DIR+'tests/{}_{}/{}_{}.csv'.format(MRN, DATETIME, MRN,
				DATETIME),
					mode='w') as testFile:
			Endpoint_Output = csv.writer(testFile,
				delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			Endpoint_Output.writerow(headers)
			Endpoint_Output.writerow([
				MRN,
				DATE,
				TIME,
				labTech,
				testType,
				bloodTime,
				timerManArr[0][0],timerManArr[1][0],timerManAvg[0],
				timerManArr[0][1],timerManArr[1][1],timerManAvg[1],
				timerManArr[0][2],timerManArr[1][2],timerManAvg[2],
				timerManArr[0][3],timerManArr[1][3],timerManAvg[3],
				timerManArr[0][4],timerManArr[1][4],timerManAvg[4],
				timerAutoArr[0][0],timerAutoArr[1][0],timerAutoAvg[0],
				timerAutoArr[0][1],timerAutoArr[1][1],timerAutoAvg[1],
				timerAutoArr[0][2],timerAutoArr[1][2],timerAutoAvg[2],
				timerManCancelled[0][0],timerManCancelled[0][1],timerManCancelled[0][2],timerManCancelled[0][3],timerManCancelled[0][4],
				timerManCancelled[1][0],timerManCancelled[1][1],timerManCancelled[1][2],timerManCancelled[1][3],timerManCancelled[1][4],
				comment])
			Endpoint_Output.writerow([])
			Endpoint_Output.writerow(['MRN', MRN])
			Endpoint_Output.writerow(['Lab Tech', labTech])
			Endpoint_Output.writerow(['Date', DATE])
			Endpoint_Output.writerow(['Test Type', testType])
			Endpoint_Output.writerow(['Time since Blood Draw', bloodTime])
			Endpoint_Output.writerow(['Comments', comment])
			Endpoint_Output.writerow(['Manual'])
			Endpoint_Output.writerow(['', 'Test Tube 1', 'Test Tube 2', 'Average', 'Test Tube 1 Excluded', 'Test Tube 2 Excluded'])
			Endpoint_Output.writerow(['Time 1', timerManArr[0][0], timerManArr[1][0], timerManAvg[0], timerManCancelled[0][0], timerManCancelled[1][0]])
			Endpoint_Output.writerow(['Time 2', timerManArr[0][1], timerManArr[1][1], timerManAvg[1], timerManCancelled[0][1], timerManCancelled[1][1]])
			Endpoint_Output.writerow(['Time 3', timerManArr[0][2], timerManArr[1][2], timerManAvg[2], timerManCancelled[0][2], timerManCancelled[1][2]])
			Endpoint_Output.writerow(['Time 4', timerManArr[0][3], timerManArr[1][3], timerManAvg[3], timerManCancelled[0][3], timerManCancelled[1][3]])
			Endpoint_Output.writerow(['Time 5', timerManArr[0][4], timerManArr[1][4], timerManAvg[4], timerManCancelled[0][4], timerManCancelled[1][4]])
			Endpoint_Output.writerow([''])
			Endpoint_Output.writerow(['Automated'])
			Endpoint_Output.writerow(['', 'Test Tube 1', 'Test Tube 2', 'Average'])
			Endpoint_Output.writerow(['Time 1', timerAutoArr[0][0], timerAutoArr[1][0], timerAutoAvg[0]])
			Endpoint_Output.writerow(['Time 2', timerAutoArr[0][1], timerAutoArr[1][1], timerAutoAvg[1]])
			Endpoint_Output.writerow(['Time 3', timerAutoArr[0][2], timerAutoArr[1][2], timerAutoAvg[2]])
			Endpoint_Output.writerow([''])
			Endpoint_Output.writerow([''])
			Endpoint_Output.writerow(["Combo Box Selection 0",self.displayArr[0]["imageSelection"]])
			Endpoint_Output.writerow(["Combo Box Selection 1",self.displayArr[1]["imageSelection"]])
			Endpoint_Output.writerow(["Combo Box Selection 2",self.displayArr[2]["imageSelection"]])
			Endpoint_Output.writerow(["Combo Box Selection 3",self.displayArr[3]["imageSelection"]])
			Endpoint_Output.writerow(["Panel Mirror 0",self.displayArr[0]["mirror"]])
			Endpoint_Output.writerow(["Panel Mirror 1",self.displayArr[1]["mirror"]])
			Endpoint_Output.writerow(["Panel Mirror 2",self.displayArr[2]["mirror"]])
			Endpoint_Output.writerow(["Panel Mirror 3",self.displayArr[3]["mirror"]])
			Endpoint_Output.writerow(["Rotation Speed",self.hardwareDict["rotationSpeed"].getValue()])
			Endpoint_Output.writerow(["Rotation Direction",self.hardwareDict["rotationDirection"].getValue()])
			Endpoint_Output.writerow(["Rotation Acceleration",self.hardwareDict["rotationAcceleration"].getValue()])			
			Endpoint_Output.writerow(["Tilt Speed",self.hardwareDict["tiltSpeed"].getValue()])
			Endpoint_Output.writerow(["Tilt Acceleration",self.hardwareDict["tiltAcceleration"].getValue()])
			Endpoint_Output.writerow(["Tilt Up Degrees",self.hardwareDict["tiltUpDegree"].getValue()])
			Endpoint_Output.writerow(["Tilt Down Degrees",self.hardwareDict["tiltDownDegree"].getValue()])
			Endpoint_Output.writerow(["Tilt Plane Offset Degrees",self.hardwareDict["tiltPlaneOffset"].getValue()])
			Endpoint_Output.writerow(["Temperature Setting",self.hardwareDict["temperature"].getValue()])
			Endpoint_Output.writerow(["Light White Intensity",self.hardwareDict["lightWhite"].getValue()])
			Endpoint_Output.writerow(["Light Red Intensity",self.hardwareDict["lightRed"].getValue()])			
			Endpoint_Output.writerow(["Light Green Intensity",self.hardwareDict["lightGreen"].getValue()])			
			Endpoint_Output.writerow(["Light Blue Intensity",self.hardwareDict["lightBlue"].getValue()])

	def _generateLoggedPics(self, log, time, method):
		for i in range(len(log)):
			for j, (pic, time) in enumerate(log[i]):
				shutil.copy('pics/{}'.format(pic), 
				ROOT_DIR+"tests/{}_{}/{}_{}_tube{}_{}.jpeg".format(MRN, DATETIME, MRN, method,
				i, time))

				shutil.copy('pics/{}'.format(pic[:-4]+".tiff"), #knock off the .jpg and put on .tiff, shorthand, fix later 
				ROOT_DIR+"tests/{}_{}/{}_{}_tube{}_{}.tiff".format(MRN, DATETIME, MRN, method,
				i, time))

#---------------------------------------------------------------------------------------


#drawer
#---------------------------------------------------------------------------------------	

	def drawer(self):
		#currently the toggle(main screen) works as follows:
		#no commands will send unless the drawerUnloadReady bit is zero
		#if drawer rbit is Loaded(0), I will send it to Unload(1)
		#if drawer rbit is Unloaded(1), I will send it to Load(0)
		#on the plc side, it should watch for when the drawer r bit changes and move accordingly,
		
		drawerOpen = self.plc.read(2373, type="rbit")
		carriageIsHomed = self.plc.read(2309, type="rbit")
		drawerInMotion = self.plc.read(2374, type="rbit")

		if not drawerOpen and not Message("Requesting to move Drawer", "Are you sure?").run():
			return 
		if carriageIsHomed and not drawerInMotion: 
			if drawerOpen:
				if STATUSBAR_ENABLED:
					self.statusBar.showMessage("Closing Drawer...",2000)
				self.plc.send(2371, 0, type="rbit") #was 1042, changed to 2371/R203, trying to figure out.
				self.btnLoad.setText("Unload")
			else:
				if STATUSBAR_ENABLED:
					self.statusBar.showMessage("Opening Drawer...",2000)
				self.plc.send(2371, 1, type="rbit")
				self.btnLoad.setText("Load")
			
				
		else:
			print("MSG\tDrawer not ready, one or more motors are moving")
			print("drawerInMotion: " + str(drawerInMotion))
			print("carriageIsHomed" + str(carriageIsHomed))
			if STATUSBAR_ENABLED:
				self.statusBar.showMessage("Drawer is currently in motion, Wait.",1000)

#---------------------------------------------------------------------------------------
             

#Logger	
#---------------------------------------------------------------------------------------

	def loggerSetup(self):
		self.btnTimerLog1.clicked.connect(lambda _: self.logTime(_, 0))
		self.btnTimerLog2.clicked.connect(lambda _: self.logTime(_, 1))
		self.btnTimerLogReset.clicked.connect(self.logReset)
		self.logEntryCounter = [0,0]
		self.logAutoEntryCounter = [0,0]
		self.pictureLog = [[],[]]
		self.pictureLogAuto = [[],[]]
		
		#data structure for Manual Timer buttons
		#[[0,1,2,3,4],[0,1,2,3,4]]
		self.logManTimerArr = [
				[self.btnManTimer1__0, self.btnManTimer1__1,
					self.btnManTimer1__2, self.btnManTimer1__3,
					self.btnManTimer1__4],
				[self.btnManTimer2__0, self.btnManTimer2__1,
					self.btnManTimer2__2, self.btnManTimer2__3,
					self.btnManTimer2__4]]

		#data structure for Automated Timer Labels
		#[[0,1,2,3],[0,1,2,3]]
		self.logAutoTimerArr = [
				[self.labAutoTimer1_0, self.labAutoTimer1_1,
					self.labAutoTimer1_2, self.labAutoTimer1_3],
				[self.labAutoTimer2_0, self.labAutoTimer2_1,
					self.labAutoTimer2_2, self.labAutoTimer2_3]]

		#styling for red button press
		for i in range(len(self.logManTimerArr)):
			for j in range(len(self.logManTimerArr[i])):
				self.logManTimerArr[i][j].setStyleSheet(BTN_STYLESHEET["red toggle"])
	
	def logTime(self, _, index):
		if not self.timerArr[index].getExistence(): #log only when that timer is running
			return
		if self.logEntryCounter[index] < 5: #limit to 5
			self.logManTimerArr[index][self.logEntryCounter[index]].setText(self.timerArr[index].getTime())
			self.pictureLog[index].append((self.camera.getCurrImgNumber(), self.timerArr[index].getTime()))
			self.logEntryCounter[index] += 1			

	def logReset(self):
        	if not Message("Are you sure", "Reset timer results?").run():
			return
        	for i in range(len(self.logManTimerArr)):
			self.logEntryCounter[i] = 0
			self.pictureLog[i] = []
			for j in range(len(self.logManTimerArr[i])):
				self.logManTimerArr[i][j].setChecked(False)
				self.logManTimerArr[i][j].setText("-")
		for i in range(len(self.logAutoTimerArr)):
			self.logAutoEntryCounter[i] = 0
			self.pictureLogAuto[i] = []
			for j in range(len(self.logAutoTimerArr[i])):
				self.logAutoTimerArr[i][j].setText("-")

	#called by signal
	def logAutoTimePoint(self, index):
		if not self.logAutoEntryCounter[index] < 3: #limit to 3
			return
		self.logAutoTimerArr[index][self.logAutoEntryCounter[index]].setText(self.timerArr[index].getTime())
		self.pictureLogAuto[index].append((self.camera.getCurrImgNumber(),self.timerArr[index].getTime()))
		self.logAutoEntryCounter[index] += 1	

#---------------------------------------------------------------------------------------


#Camera
#---------------------------------------------------------------------------------------

	def cameraSetup(self):
		#Camera QThread setup
		self.camera = Camera()
		self.camera.imageSignal.connect(self.displayImg)
		self.camera.timerStoppedSignal.connect(self.generateTest)
		self.camera.autoTimePointSignal.connect(self.logAutoTimePoint)
		if CAMERA_ENABLED:
			self.camera.start()

#---------------------------------------------------------------------------------------


#Display
#---------------------------------------------------------------------------------------

	def displaySetup(self):
		#data structure	[{"comboBox": combobox_0, "imageSelection": M1, "pixmap": pixmap, "panels": [list of panels], "mirror": "Normal"}] 
		self.displayArr = [
			{"comboBox": self.comboBox_0, "imageSelection": "M1", "pixmap": None, "panels": [self.labMainVideo_0, self.labMotorVideo_0, self.labTempVideo_0], "mirror": "Normal"},
			{"comboBox": self.comboBox_1, "imageSelection": "T1", "pixmap": None, "panels": [self.labMainVideo_1, self.labMotorVideo_1, self.labTempVideo_1], "mirror": "Normal"},
			{"comboBox": self.comboBox_2, "imageSelection": "T2", "pixmap": None, "panels": [self.labMainVideo_2, self.labMotorVideo_2, self.labTempVideo_2], "mirror": "Normal"},
			{"comboBox": self.comboBox_3, "imageSelection": "M2", "pixmap": None, "panels": [self.labMainVideo_3, self.labMotorVideo_3, self.labTempVideo_3], "mirror": "Normal"}]
		
		#used for setting comboBox as index reference
		self.comboBoxRef = ["M1", "T1", "T2", "M2"]

		#connect comboBox events
		for i, display in enumerate(self.displayArr):
			display["comboBox"].currentIndexChanged.connect(lambda _, i=i: self.comboBoxSelection(_, i))

		#connect mirror btns
		self.btnMirrorArr = [self.btnMirror_0, self.btnMirror_1, self.btnMirror_2, self.btnMirror_3]
		for i, btn in enumerate(self.btnMirrorArr):
			btn.clicked.connect(lambda _, i=i: self.mirrorFlip(_, i))

		#pixmap temporary holder
		self.pixmapDict = {"M1": "pixmap", "T1": "pixmap", "T2": "pixmap", "M2": "pixmap"}
		

	#change panel based on comboBox selection
	def comboBoxSelection(self, _, index):
		self.displayArr[index]["imageSelection"] = str(self.displayArr[index]["comboBox"].currentText())

	#mirror on btn press
	def mirrorFlip(self, _, index):
		#forward image = normal
		#reverse iamge = mirror
		if self.displayArr[index]["mirror"] == "Mirror":
			self.displayArr[index]["mirror"] = "Normal"
			self.btnMirrorArr[index].setText("Forward\nImage")
		else:
			self.displayArr[index]["mirror"] = "Mirror"
			self.btnMirrorArr[index].setText("Reverse\nImage")

	#called by signal
	def displayImg(self, imgObject):

		#CV2img to QImg + scaling and rgb swap
		self.img=QImage(imgObject, imgObject.shape[1], 
				imgObject.shape[0], imgObject.strides[0], 
				QImage.Format_RGB888).scaled(1590,1200).rgbSwapped()

		#rotate
		self.pixmapOG = QPixmap.fromImage(self.img).transformed(
			QTransform().rotate(270), 
			Qt.SmoothTransformation)

		#take exact pixels (x,y,width,height) from image and scaled to fit panel size (currently 231x611)
		self.pixmapDict["M1"] = self.pixmapOG.copy(950,280,240,960).scaled(160, 641, Qt.KeepAspectRatio)
		self.pixmapDict["T1"] = self.pixmapOG.copy(640,80,345,1380).scaled(160, 641, Qt.KeepAspectRatio)
		self.pixmapDict["T2"] = self.pixmapOG.copy(180,80,345,1380).scaled(160, 641, Qt.KeepAspectRatio).transformed(QTransform().scale(-1, 1))
		self.pixmapDict["M2"] = self.pixmapOG.copy(0,280,240,960).scaled(160, 641, Qt.KeepAspectRatio).transformed(QTransform().scale(-1, 1))

		#put pixmap into dict based on imageSelection and mirror
		for displayDict in self.displayArr:
			displayDict["pixmap"] = self.pixmapDict[displayDict["imageSelection"]]
			if displayDict["mirror"] == "Mirror":
				displayDict["pixmap"] = displayDict["pixmap"].transformed(QTransform().scale(-1, 1))
		
		#output pixmap to panel
		for displayDict in self.displayArr:
			for panel in displayDict["panels"]:
				panel.setPixmap(displayDict["pixmap"])

	#update mirror on display index and value eg. (0, "Normal")
	def updateMirrorValue(self, index, value):
		self.displayArr[index]["mirror"] = value
		if value == "Normal":
			self.btnMirrorArr[index].setText("Forward\nImage")
		else:
			self.btnMirrorArr[index].setText("Reverse\nImage")

	#update comboBox on display index and value eg. (0, "M1")
	def updateComboBoxValue(self, index, value):
		self.displayArr[index]["comboBox"].setCurrentIndex(self.comboBoxRef.index(value))
		
#---------------------------------------------------------------------------------------


#hardware setup (motor, light, temp)
#---------------------------------------------------------------------------------------
	
	def hardwareSetup(self):
		self.plc = PLC()
		
		self.hardwareDict = dict()

		self.hardwareDict["rotationSpeed"] = 		HardwareNormal(15.0, 1010, self.plc, [self.labRotSpeed, self.labRotSpeedMain], (5.0,30.0), 
								incButtons=[self.btnRotSpeedInc], decButtons=[self.btnRotSpeedDec],
								superIncButtons=[self.btnRotSpeedInc2], superDecButtons=[self.btnRotSpeedDec2])
		self.hardwareDict["rotationDirection"] =  	HardwareRotate(0, 1014, self.plc, [self.btnRotDirIcon], (0,1), 
								self.btnRotDirIcon, QIcon('button2.png'), QIcon('button1.png'))
		self.hardwareDict["rotationAcceleration"] =  	HardwareNormal(5.0, 1012, self.plc, [self.labRotAcc], (0.0,50.0), 
								incButtons=[self.btnRotAccInc], decButtons=[self.btnRotAccDec],
								superIncButtons=[self.btnRotAccInc2], superDecButtons=[self.btnRotAccDec2])
		self.hardwareDict["tiltSpeed"] =  		HardwareNormal(15.0, 1000, self.plc, [self.labTilSpeed, self.labTilSpeedMain], (5.0,30.0), 
								incButtons=[self.btnTilSpeedInc], decButtons=[self.btnTilSpeedDec],
								superIncButtons=[self.btnTilSpeedInc2], superDecButtons=[self.btnTilSpeedDec2])
		self.hardwareDict["tiltAcceleration"] =  	HardwareNormal(5.0, 1002, self.plc, [self.labTilAcc], (0.0,50.0), 
								incButtons=[self.btnTilAccInc], decButtons=[self.btnTilAccDec],
								superIncButtons=[self.btnTilAccInc2], superDecButtons=[self.btnTilAccDec2])
		self.hardwareDict["tiltUpDegree"] =  	HardwareNormal(15.0, 1004, self.plc, [self.labTilForw], (0.1,10.0), 
								incButtons=[self.btnTilForwInc], decButtons=[self.btnTilForwDec],
								superIncButtons=[self.btnTilForwInc2], superDecButtons=[self.btnTilForwDec2])
		self.hardwareDict["tiltDownDegree"] =  	HardwareNormal(15.0, 1006, self.plc, [self.labTilBack], (0.1,10.0), 
								incButtons=[self.btnTilBackInc], decButtons=[self.btnTilBackDec],
								superIncButtons=[self.btnTilBackInc2], superDecButtons=[self.btnTilBackDec2])
		self.hardwareDict["tiltPlaneOffset"] =  	HardwareNormal(0.0, 1008, self.plc, [self.labTilPlane], (-4.0,4.0), 
								incButtons=[self.btnTilPlaneInc], decButtons=[self.btnTilPlaneDec],
								superIncButtons=[self.btnTilPlaneInc2], superDecButtons=[self.btnTilPlaneDec2])
		self.hardwareDict["temperature"] =  		HardwareNormal(30.0, 1020, self.plc, [self.labTempSet, self.labTempSet_2, self.labTempMain], (20.0,50.0), 
								incButtons=[self.btnTempSetInc, self.btnTempSetInc_2], decButtons=[self.btnTempSetDec, self.btnTempSetDec_2],
								superIncButtons=[self.btnTempSetInc2, self.btnTempSetInc2_2], superDecButtons=[self.btnTempSetDec2, self.btnTempSetDec2_2])
		self.hardwareDict["lightWhite"] =  		HardwareLight(0, 1030, self.plc, [self.labLightWhite], (0,100), 
								incButtons=[self.btnLightWhiteInc], decButtons=[self.btnLightWhiteDec],
								superIncButtons=[self.btnLightWhiteInc2], superDecButtons=[self.btnLightWhiteDec2])
		self.hardwareDict["lightRed"] =  		HardwareLight(0, 1032, self.plc, [self.labLightRed], (0,100),
								incButtons=[self.btnLightRedInc], decButtons=[self.btnLightRedDec],
								superIncButtons=[self.btnLightRedInc2], superDecButtons=[self.btnLightRedDec2])
		self.hardwareDict["lightGreen"] =  		HardwareLight(0, 1034, self.plc, [self.labLightGreen], (0,100), 
								incButtons=[self.btnLightGreenInc], decButtons=[self.btnLightGreenDec],
								superIncButtons=[self.btnLightGreenInc2], superDecButtons=[self.btnLightGreenDec2])
		self.hardwareDict["lightBlue"] =  		HardwareLight(0, 1036, self.plc, [self.labLightBlue], (0,100), 
								incButtons=[self.btnLightBlueInc], decButtons=[self.btnLightBlueDec],
								superIncButtons=[self.btnLightBlueInc2], superDecButtons=[self.btnLightBlueDec2])

#---------------------------------------------------------------------------------------


#configuration/settings
#---------------------------------------------------------------------------------------

	def configSetup(self):
		#load each config name into combobox
		self.readConfig()
		#load the first one by default
		if LOAD_SETUP_ON_START:
			self.loadConfig(0)
		self.comboBoxConfig.currentIndexChanged.connect(self.loadConfig)
		self.btnAddConfig.clicked.connect(self.addConfigCall)
		self.btnRemoveConfig.clicked.connect(self.removeConfig)

	def readConfig(self):
		self.dfConfig = pandas.read_csv('testConfig.csv')
		self.configNameArr = self.dfConfig.columns.values[1:]		
		self.comboBoxConfig.clear()
		self.comboBoxConfig.addItems(self.configNameArr)
		
	def loadConfig(self, index):
		#print(self.dfConfig[self.configNameArr[index]])
		config = self.dfConfig[self.configNameArr[index]]
		self.editExpTestType.setText(config[0])
		try:
			self.editExpOperName.setCurrentIndex(self.labTechNamesArr.index(config[1]))
		except ValueError:
			self.editExpOperName.setCurrentIndex(0)
			print("defaulted to first name")
		self.updateComboBoxValue(0, config[2])
		self.updateComboBoxValue(1, config[3])
		self.updateComboBoxValue(2, config[4])
		self.updateComboBoxValue(3, config[5])
		self.updateMirrorValue(0, config[6])
		self.updateMirrorValue(1, config[7])
		self.updateMirrorValue(2, config[8])
		self.updateMirrorValue(3, config[9])
		
		self.hardwareDict["rotationSpeed"].setValue(float(config[10]))
		self.hardwareDict["rotationDirection"].setValue(int(config[11]))
		self.hardwareDict["rotationAcceleration"].setValue(float(config[12]))
		self.hardwareDict["tiltSpeed"].setValue(float(config[13]))
		self.hardwareDict["tiltAcceleration"].setValue(float(config[14]))
		self.hardwareDict["tiltUpDegree"].setValue(float(config[15]))
		self.hardwareDict["tiltDownDegree"].setValue(float(config[16]))
		self.hardwareDict["tiltPlaneOffset"].setValue(float(config[17]))
		self.hardwareDict["temperature"].setValue(float(config[18]))
		self.hardwareDict["lightWhite"].setValue(int(config[19]))
		self.hardwareDict["lightRed"].setValue(int(config[20]))
		self.hardwareDict["lightGreen"].setValue(int(config[21]))
		self.hardwareDict["lightBlue"].setValue(int(config[22]))

		self.updateAllPLC(self.hardwareDict.values())
	
	#will send all config values to PLC
	def updateAllPLC(self, hardwareList):
		for hardware in hardwareList:
			hardware.sendToPLC()

	#attached to button
	def addConfigCall(self):
		self.temp = InputDialogKeyboard("Add Config")
		self.temp.show()
		self.temp.move(600,200)
		self.temp.returnMsg.connect(self.addConfig)

	#called by signal
	def addConfig(self, configName):
		self.currentConfigArr = [
			str(self.editExpTestType.text()),
			str(self.editExpOperName.currentText()),
			self.displayArr[0]["imageSelection"],
			self.displayArr[1]["imageSelection"],
			self.displayArr[2]["imageSelection"],
			self.displayArr[3]["imageSelection"],
			self.displayArr[0]["mirror"],
			self.displayArr[1]["mirror"],
			self.displayArr[2]["mirror"],
			self.displayArr[3]["mirror"],
			self.hardwareDict["rotationSpeed"].getValue(),
			self.hardwareDict["rotationDirection"].getValue(),
			self.hardwareDict["rotationAcceleration"].getValue(),
			self.hardwareDict["tiltSpeed"].getValue(),
			self.hardwareDict["tiltAcceleration"].getValue(),
			self.hardwareDict["tiltUpDegree"].getValue(),
			self.hardwareDict["tiltDownDegree"].getValue(),
			self.hardwareDict["tiltPlaneOffset"].getValue(),
			self.hardwareDict["temperature"].getValue(),
			self.hardwareDict["lightWhite"].getValue(),
			self.hardwareDict["lightRed"].getValue(),
			self.hardwareDict["lightGreen"].getValue(),
			self.hardwareDict["lightBlue"].getValue()]
		self.dfConfig[str(configName)] = self.currentConfigArr
		self.dfConfig.to_csv("testConfig.csv", encoding='utf-8', index=False)
		self.readConfig()
		self.comboBoxConfig.setCurrentIndex(self.comboBoxConfig.count()-1)

	def removeConfig(self):
		confirm, configName = ComboBoxDialog("Removing Configuration", self.configNameArr).run()
		if not confirm:
			return
		self.dfConfig.drop([str(configName)], axis = 1, inplace = True)
		self.dfConfig.to_csv("testConfig.csv", encoding='utf-8', index=False)
		self.readConfig()

#---------------------------------------------------------------------------------------




#CLASSES

#Pandas
#------------------------------------------------------------------------------------------------

#copy pasted, give credit TODO find source
class PandasModel(QAbstractTableModel):
	def __init__(self, data, parent=None):
		QAbstractTableModel.__init__(self, parent)
		self
		self._data = data

	def rowCount(self, parent=None):
		return self._data.shape[0]

	def columnCount(self, parent=None):
		return self._data.shape[1]-1 #-1 bc of leading 0 from pandas export

	def data(self, index, role=Qt.DisplayRole):
		if index.isValid():
			if role == Qt.DisplayRole:
				return str(self._data.iloc[index.row(), index.column()])
		return None

	def getData(self, index):
		return str(self._data.iloc[index.row(), index.column()])

	def headerData(self, col, orientation, role):
		if orientation == Qt.Horizontal and role == Qt.DisplayRole:
			return str(self._data.columns[col])
		return None	

#------------------------------------------------------------------------------------------------


#Message Dialog
#------------------------------------------------------------------------------------------------

#used by log reset confirm
class Message():
	def __init__(self, title, message):
    		self.inputDialog = QMessageBox(None)
    		self.inputDialog.setWindowTitle(title)
    		self.inputDialog.setText(message)
    		self.inputDialog.setFont(FONT)
		self.inputDialog.addButton(QMessageBox.No)
		self.inputDialog.addButton(QMessageBox.Yes)

	def run(self):
    		if self.inputDialog.exec_() == 16384: #mapped to yes
			return True
		return False

#------------------------------------------------------------------------------------------------


#ComboBox Dialog
#------------------------------------------------------------------------------------------------

#used for combobox deletion (labtech and config)
class ComboBoxDialog():
	def __init__(self, title, items):
    		self.inputDialog = QInputDialog(None)
    		self.inputDialog.setWindowTitle(title)
    		self.inputDialog.setFont(FONT)
		self.inputDialog.setComboBoxItems(items)
	def run(self):
		confirm = self.inputDialog.exec_()
    		out = self.inputDialog.textValue()
		return (confirm, out)

#------------------------------------------------------------------------------------------------


#Input Dialog w/ keyboard
#------------------------------------------------------------------------------------------------

class InputDialogKeyboard(QWidget):
	returnMsg = pyqtSignal(str)

	def __init__(self, title):
		super(InputDialogKeyboard, self).__init__()
		loadUi('dialog.ui',self)
    		self.setFont(FONT)
		self.setWindowTitle(title)
		self.buttonBox.buttons()[0].clicked.connect(self.accepted)
		self.buttonBox.buttons()[1].clicked.connect(self.rejected)
		self.keyboard = VirtualKeyboard(self.lineEdit, "", title)
		self.keyboard.move(400,350)
		self.keyboard.show()

	def accepted(self):
		self.keyboard.close()
		self.close()
		self.returnMsg.emit(self.lineEdit.text())
	
	def rejected(self):
		self.keyboard.close()
		self.close()

#------------------------------------------------------------------------------------------------


#Timer
#------------------------------------------------------------------------------------------------

class Timer():

	def __init__(self, timeLabel, startBtn, stopBtn):
		self.exists = False
		self.startBtn = startBtn
		self.timeLabel = timeLabel
		self.stopBtn = stopBtn
		self.base = 0
		self.time = 0
		self.holder = 0

	def startTimer(self):
		if self.exists:
			return
		self.startBtn.setText("Testing IP")
		self.exists = True
		self.timer = QTimer()
		self.timer.timeout.connect(self._updateTimer)
		self.timer.start(100)
		self.base = time.time() * 10.0

	def stopTimer(self):
		if self.exists:
			self.exists = False
			self.timer.stop()
			self.holder = self.time
			self.stopBtn.setStyleSheet(BTN_STYLESHEET_RESET)
			self.stopBtn.setText("Reset Time")
		else:
			self._resetTimer()
			self.stopBtn.setStyleSheet(BTN_STYLESHEET_STOP)
			self.startBtn.setText("Start")			
			self.stopBtn.setText("Stop")
	
	def getRawTime(self):
		return self.time
	
	def getTime(self):
		return tenthsToDisplay(self.time)
	
	def getExistence(self):
		return self.exists

	def _resetTimer(self):
		self.time = 0
		self.holder = 0
		self.timeLabel.setText("00.0")

	def _displayTimeToLabel(self):
		self.timeLabel.setText(tenthsToDisplay(self.time))

	def _updateTimer(self):
		self.time = int(((time.time() * 10.0) - self.base)) + self.holder
		self._displayTimeToLabel()

#------------------------------------------------------------------------------------------------


#Camera QThread
#------------------------------------------------------------------------------------------------

class Camera(QThread):

	#signal slot to send imgObject to GUI
	imageSignal = pyqtSignal(np.ndarray)
	timerStoppedSignal = pyqtSignal()
	autoTimePointSignal = pyqtSignal(int)

	def __init__(self, parent=None):
		super(Camera, self).__init__(parent)

		#general camera vars
		self.isThereNewImg = False
		self.imgNumber = 0
		self.imgObject = 0

		#saveThread vars
		self.saveThreadExist = False
		self.timerState = [False, False]

	#main Camera
	#automatically runs on qthread start()
	def run(self):

		#camera setup
		self.first_device = py.TlFactory.GetInstance().CreateFirstDevice()
		self.instant_camera = py.InstantCamera(self.first_device)
		self.instant_camera.Open()
		self.instant_camera.PixelFormat = "BayerRG8"
		self.instant_camera.ExposureAuto = "Off"
		self.instant_camera.StartGrabbing(py.GrabStrategy_LatestImages)
		
		self.isThereNewImg = False
		self.isThereNewVidImg = False
		self.isThereNewAnalysisImg = False
		
		#main camera grab
		while True:
			if self.instant_camera.NumReadyBuffers:
			    res = self.instant_camera.RetrieveResult(200)
			    if res:
				try:
				    if res.GrabSucceeded():
				        self.imgObject = cv2.cvtColor(res.Array, cv2.COLOR_BAYER_RG2RGB)
				#	self.imgObject2 = cv2.cvtColor(res.Array, cv2.COLOR_BAYER_RG2GRAY)
					self.imageSignal.emit(self.imgObject)
					self.isThereNewImg = True
					self.isThereNewVidImg = True
					self.isThereNewAnalysisImg = True
					self.isThereNewAnalysisImg1 = True
					
				finally:
				    res.Release()

	#start thread
	def startThread(self, _, index):
		#True only if both timerState is False

		
		#might not need this prop

		if not self.saveThreadExist:
			self.saveThreadExist = True #tracks existence
			self.threadSav = Thread(target=self.saveThread_benchmark, args=())
			self.threadSav.daemon = True
			self.threadSav.start()
			self.threadVid = Thread(target=self.saveVidThread, args=())
			self.threadVid.daemon = True
			self.threadVid.start()
			self.threadAnaT0 = Thread(target=self.analysisThread, args=([0]))
			self.threadAnaT0.daemon = True
			if AUTO_DETECTION_ENABLED:
				self.threadAnaT0.start()
			#stress test
			"""
			self.threadAnaT0_1 = Thread(target=self.analysisThread1, args=([0]))
			self.threadAnaT0_1.daemon = True
			self.threadAnaT0_1.start()
			self.threadAnaT1_1 = Thread(target=self.analysisThread2, args=([1]))
			self.threadAnaT1_1.daemon = True
			self.threadAnaT1_1.start()
			"""
		

	#stop thread
	def stopSaveThread(self, _, index):
		self.timerState[index] = False
		if not any(self.timerState):
			#emit only right before thread exit
			if self.saveThreadExist:
				self.timerStoppedSignal.emit()
			self.saveThreadExist = False

	#main saving thread
	def saveThread(self):
		while self.saveThreadExist:
			if self.isThereNewImg:
				
				self.isThereNewImg = False
				self.imgNumber += 1
				imgName = 'pics/Vtest'+str(self.imgNumber)+'.jpg'
				cv2.imwrite(imgName, self.imgObject) 
					
				cv2.imwrite('pics/Vtest'+str(self.imgNumber)+'.tiff', self.imgObject)		
				print('MSG	saved {}'.format(imgName))
				#self.isThereNewAnalysisImg = True
				print(image_saved)

	#main saving thread
	def saveThread_benchmark(self):
		counter = 0
		c = time.time()
		while self.saveThreadExist:
			if self.isThereNewImg:
				counter+=1
				self.isThereNewImg = False
				self.imgNumber += 1
				imgName = 'pics/Vtest'+str(self.imgNumber)+'.jpg'
				cv2.imwrite(imgName, self.imgObject) 

				
				#crop images and write to pics
				#top left corner
				x1, y1 = 400, 700
				#bottom right corner
				x2, y2 = 3900, 1500
				
				
				# cv2 BGR to RGB
				#img = cv2.cvtColor(self.imgObject, cv2.COLOR_BGR2RGB)
					

				# resize main cv2 image into m1 m2 tubes
				cropped_imgM1 = self.imgObject[y1:y2, x1:x2]
#				imgM2 = self.img[x2:width, y2:height]

				#write cropped frame to analysis folder				
				cropName = 'analysis/cropped_pics/crop'+str(self.imgNumber)+'.jpg'
				cv2.imwrite(cropName, cropped_imgM1) 
				
				# see if its cropped properly
#				cv2.imshow("m1", imgM1)
#				cv2.imshow("m2", imgM2)
				
				#create a virtual environment and run analysis script to analyze, only call on first loop, and continue,
				
				

				#cv2.imwrite('pics/Vtest'+str(self.imgNumber)+'.tiff', self.imgObject)		
				#print('MSG	saved {}'.format(imgName))
				if counter == 10:
					counter = 0
					print(str(1.0/((time.time() - c)/10.0)))
					c = time.time()
				#self.isThereNewAnalysisImg = True
				
				

	#main video making thread
	def saveVidThread(self):
		vidtest = cv2.VideoWriter(ROOT_DIR+'tests/{}_{}/{}_{}.avi'.format(MRN, DATETIME, MRN, DATETIME), cv2.VideoWriter_fourcc('X','V','I','D'), 10, (IMAGE_RESOLUTION['x'], IMAGE_RESOLUTION['y'])) #new addition, may need to remove
		while self.saveThreadExist:
			if self.isThereNewVidImg:
				self.isThereNewVidImg = False
				vidtest.write(self.imgObject)
		vidtest.release() #new attempt, may remove

	#debugging heat stress test
	"""
	def analysisThread1(self, a):
		vidtest2 = cv2.VideoWriter('tests/{}/{}{}.avi'.format(DATETIME,DATETIME,a), cv2.VideoWriter_fourcc('X','V','I','D'), 10, (IMAGE_RESOLUTION['x'], IMAGE_RESOLUTION['y'])) #new addition, may need to remove
		while self.saveThreadExist:
			if self.isThereNewAnalysisImg:
				self.isThereNewAnalysisImg = False
				vidtest2.write(self.imgObject)
		vidtest2.release() #new attempt, may remove
	def analysisThread2(self, a):
		vidtest2 = cv2.VideoWriter('tests/{}/{}{}.avi'.format(DATETIME,DATETIME,a), cv2.VideoWriter_fourcc('X','V','I','D'), 10, (IMAGE_RESOLUTION['x'], IMAGE_RESOLUTION['y'])) #new addition, may need to remove
		while self.saveThreadExist:
			if self.isThereNewAnalysisImg1:
				self.isThereNewAnalysisImg1 = False
				vidtest2.write(self.imgObject)
		vidtest2.release() #new attempt, may remove
	"""

	#analysis thread
	def analysisThread(self, tubeNumber):
		stage = 0
		found = False	
		print("in analysis thread")
		while self.saveThreadExist:
			if self.isThereNewAnalysisImg:
				self.isThereNewAnalysisImg = False
		if self.saveThreadExist:	
			subprocess.Popen(["/home/bha/fastai36/bin/python", "/home/bha/Desktop/lluFolder/masterProgram/01_26_19/analysis_script.py"]) 
			#subprocess.Popen(["/home/llu-2/anaconda3/envs/tf_cpu/bin/python", "/home/llu-2/Desktop/lluFolder/tensorflow/models/research/object_detection/object_detection_image.py"]) 				




	#report back to main class the img file name
	def getCurrImgNumber(self):
		return 'Vtest'+str(self.imgNumber)+'.jpg'

#------------------------------------------------------------------------------------------------


#Hardware
#------------------------------------------------------------------------------------------------

class Hardware():
	def __init__(self, value, address, plc, labels, limits):
		self._value = value
		self._labels = labels
		self._address = address
		self._limits = limits
		self._plc = plc

	def setValue(self, value):
		self._value = value
		self._checkLimits()
		self._show()
	
	def getValue(self):
		return self._value

	def getAddress(self):
		return self._address

	def sendToPLC(self, timesTen=True):
		if self._address:
			if timesTen:
				self._plc.send(self._address, int(self._value*10))
			else:
				self._plc.send(self._address, self._value)

	def _show(self):
		#display
		for label in self._labels:
			label.setText(str(self._value))
	
	def _checkLimits(self):
		#check limits
		if self._value < self._limits[0]:
			self._value = self._limits[0]
		elif self._value > self._limits[1]:
			self._value = self._limits[1]

class HardwareNormal(Hardware):
	def __init__(self, value, address, plc, labels, limits, incButtons, decButtons, superIncButtons, superDecButtons):
		Hardware.__init__(self, value, address, plc, labels, limits)		
		self._incButtons = incButtons
		self._decButtons = decButtons
		self._superIncButtons = superIncButtons
		self._superDecButtons = superDecButtons
		
		self._attachButtons()
		self._checkLimits()
		self._show()

	def _attachButtons(self):
		for button in self._incButtons:
			button.clicked.connect(self._btnFunction("+"))
		for button in self._decButtons:
			button.clicked.connect(self._btnFunction("-"))
		for button in self._superIncButtons:
			button.clicked.connect(self._btnFunction("++"))
		for button in self._superDecButtons:
			button.clicked.connect(self._btnFunction("--"))

	def _btnFunction(self, command):
		def function():
			#inc or dec
			if command == "+":
				self._value += 0.1
			elif command == "-":
				self._value -= 0.1
			elif command == "++":
				self._value += 1
			elif command == "--":
				self._value -= 1
			self._value = round(self._value,1)
			self._checkLimits()
			self._show()
			self.sendToPLC()
		return function


class HardwareRotate(Hardware):
	def __init__(self, value, address, plc, labels, limits, toggle, normalIcon, invertedIcon):
		Hardware.__init__(self, value, address, plc, labels, limits)
		self._toggle = toggle	
		#self._normalIcon = normalIcon
		#self._invertedIcon = invertedIcon			
		self._attachButtons()
		self._checkLimits()
		self._show()
		
	def _attachButtons(self):
		self._toggle.clicked.connect(self._rotate)

	def _rotate(self):
		self.setValue(1) if self._value == 0 else self.setValue(0)
		self.sendToPLC()

	def _show(self):
		if self._value == 0:
			self._toggle.setText("Counter\nClockwise")
			#self._toggle.setIcon(self._normalIcon)
			#self._toggle.setIconSize(QSize(100,100))
		elif self._value == 1:
			self._toggle.setText("Clockwise")			
			#self._toggle.setIcon(self._invertedIcon)
			#self._toggle.setIconSize(QSize(100,100))

class HardwareLight(HardwareNormal):
	#increment lights by 1,10 instead of default .1,1
	def __init__(self, value, address, plc, labels, limits, incButtons, decButtons, superIncButtons, superDecButtons):
		HardwareNormal.__init__(self, value, address, plc, labels, limits, incButtons, decButtons, superIncButtons, superDecButtons)

	def _btnFunction(self, command):
		def function():
			#inc or dec
			if command == "+":
				self._value += 1
			elif command == "-":
				self._value -= 1
			elif command == "++":
				self._value += 10
			elif command == "--":
				self._value -= 10
			self._checkLimits()
			self._show()			
			self.sendToPLC(timesTen=True) #new change, all signals sent are auto mult by 10
			
		return function	

#------------------------------------------------------------------------------------------------


#PLC
#------------------------------------------------------------------------------------------------

class PLC():
	def __init__(self):
		if PLC_PSUEDO_ENABLED:
			pass
		else:
			self.instrument = minimalmodbus.Instrument('/dev/ttyUSB0', 1) # port name, slave address (in decimal)
			self.instrument.serial.baudrate = 19200   # Baud
			self.instrument.serial.bytesize = 8
			self.instrument.serial.parity   = serial.PARITY_ODD
			self.instrument.serial.stopbits = 1
			self.instrument.serial.timeout  = 0.05   # seconds
			self.instrument.mode = minimalmodbus.MODE_RTU   # rtu mode

	def send(self, address, value, type="DT", priority=True):
		#priority will attempt to resend if collided
		if PLC_PSUEDO_ENABLED:
			print("PSUEDO	SENT	{} {} {}".format(address, value, type))
		else:
			try:
				if type == "rbit":
					self.instrument.write_bit(address, value)
				elif type == "DT":
					self.instrument.write_register(address, value, functioncode=6)
				print("SENT	{} {} {}".format(address, value, type))
			except:
				print("MSG	send {} collided, will reattempt to send {} to {}".format(address, value, address))
				if priority:
					self.send(address, value, type=type, priority=priority) #recursive, watch for recursion depth, shouldn't happen

	def read(self, address, type="DT", priority=True):
		#priority will attempt to resend if collided
		if PLC_PSUEDO_ENABLED:
			rand = random.randint(0,1)
			print("PSUEDO	READ	{} {} {}".format(address, rand, type))
			return rand
		else:
			try:
				if type == "rbit":
					returned = self.instrument.read_bit(address, functioncode=1)
				elif type == "xbit":
					returned = self.instrument.read_bit(address, functioncode=2)
				elif type == "ybit":
					returned = self.instrument.read_bit(address, functioncode=1)
				elif type == "DT":
					returned = self.instrument.read_register(address, functioncode=3)
				#print("READ	{} {} {}".format(address, returned, type))
				return returned
			except:
				print("MSG	read {} collided, will reattempt to read {}".format(address, address))
				if priority:
					return self.read(address, type=type, priority=priority)
				return -1

	def sendMulti(self, address, value, type="DT", priority=True):
		for i in range(len(address)):
			self.send(address[i], value[i], type=type, priority=priority)

	def readMulti(self, address, type="DT", priority=True):
		values = []
		for i in range(len(address)):
			values.append(self.read(address[i], type=type, priority=priority))
		return values
			
#------------------------------------------------------------------------------------------------


#Polling
#------------------------------------------------------------------------------------------------

class Polling(QThread):
	#(address, value, type)
	pollSignal = pyqtSignal(int, int, str)

	def __init__(self, plc, pollDict, IO_Dict):
		super(Polling, self).__init__()
		self.plc = plc
		self.maintenanceEnabled = False
		self.pollDict = pollDict
		self.IO_Dict = IO_Dict
	
	def maintenanceToggle(self, _, toggle):
		self.maintenanceEnabled = toggle

	#automatically runs on qthread start()
	def run(self):
		POLLING_RATE = 0.1  #100ms
		while True:
			for address, dataDict in self.pollDict.items(): 
				time.sleep(POLLING_RATE)
				currentValue = self.plc.read(address, priority = False)
				if currentValue != -1 and currentValue != dataDict["value"]:
					self.pollDict[address]["value"] = currentValue
					self.pollSignal.emit(address, currentValue, "DT")
			if self.maintenanceEnabled:
				for bitType, valueLabelArr in self.IO_Dict.items():
					for address, valueLabel in enumerate(valueLabelArr):
						time.sleep(POLLING_RATE)
						if not self.maintenanceEnabled:
							break
						currentValue = self.plc.read(address, type=bitType, priority = False)
						#pass collided info
						if currentValue == -1:
							continue
						self.IO_Dict[bitType][address]["value"] = currentValue
						self.pollSignal.emit(address, currentValue, bitType)
			
#------------------------------------------------------------------------------------------------


#Keyboard
#------------------------------------------------------------------------------------------------

#TODO give creds
class KeyButton(QPushButton):
    sigKeyButtonClicked =  pyqtSignal()

    def __init__(self, key):

        super(KeyButton, self).__init__()
    	self.setFont(FONT)

        self._key = key
        self._activeSize = QSize(60,60)
        self.clicked.connect(self.emitKey)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))

    def emitKey(self):
        self.sigKeyButtonClicked.emit()

    def enterEvent(self, event):
        self.setFixedSize(self._activeSize)

    def leaveEvent(self, event):
        self.setFixedSize(self.sizeHint())

    def sizeHint(self):
        return QSize(55, 55)

class VirtualKeyboard(QWidget):

    def __init__(self, lineEdit, text, title):
        super(VirtualKeyboard, self).__init__()
	self.setWindowFlags(Qt.WindowStaysOnTopHint)
	self.setWindowTitle("Keyboard: " + title)
        self.globalLayout = QVBoxLayout(self)
        self.keysLayout = QGridLayout()
        self.buttonLayout = QHBoxLayout()

        self.keyListByLines = [
		    [('1','!'), ('2','@'), ('3','#'), ('4','$'), ('5','%'), ('6','*'), ('7','('), ('8',')'), ('9','_'), ('0','/')],
                    [('q','Q'), ('w','W'), ('e','E'), ('r','R'), ('t','T'), ('y','Y'), ('u','U'), ('i','I'), ('o','O'), ('p','P')],
                    [('a','A'), ('s','S'), ('d','D'), ('f','F'), ('g','G'), ('h','H'), ('j','J'), ('k','K'), ('l','L'), ('\'','"')],
                    [('z','Z'), ('x','X'), ('c','C'), ('v','V'), ('b','B'), ('n','N'), ('m','M'), (',','-'), ('.','+'), ('?','=')],
                ]

	self.inputLine = lineEdit
        self.inputString = text
	self.state = False

        self.stateButton = QPushButton()
        self.stateButton.setText('Caps Lock')
	self.stateButton.setFont(FONT)
        self.backButton = QPushButton()
        self.backButton.setText('BackSpace')
	self.backButton.setFont(FONT)
	self.spaceButton = QPushButton()
        self.spaceButton.setText("Space")
	self.spaceButton.setFont(FONT)
        self.cancelButton = QPushButton()
        self.cancelButton.setText("Close")
	self.cancelButton.setFont(FONT)

        for lineIndex, line in enumerate(self.keyListByLines):
            for keyIndex, key in enumerate(line):
                buttonName = "keyButton" + str(key)
                self.__setattr__(buttonName, KeyButton(key))
                self.keysLayout.addWidget(self.getButtonByKey(key), self.keyListByLines.index(line), line.index(key))
                self.getButtonByKey(key).setText(key[self.state])
		self.getButtonByKey(key).sigKeyButtonClicked.connect(lambda v=key: self.addInputByKey(v))
                self.keysLayout.setColumnMinimumWidth(keyIndex, 60)
            self.keysLayout.setRowMinimumHeight(lineIndex, 60)

        self.stateButton.clicked.connect(self.switchState)
	self.spaceButton.clicked.connect(self.space)
        self.backButton.clicked.connect(self.backspace)
        self.cancelButton.clicked.connect(self.close)

	self.buttonLayout.addWidget(self.stateButton)
        self.buttonLayout.addWidget(self.backButton)
	self.buttonLayout.addWidget(self.spaceButton)
	self.buttonLayout.addWidget(self.cancelButton)
        
        self.globalLayout.addLayout(self.keysLayout)
        self.globalLayout.addLayout(self.buttonLayout)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))

    def getButtonByKey(self, key):
        return getattr(self, "keyButton" + str(key))

    def switchState(self):
	self.state = not self.state
	for line in self.keyListByLines:
            	for key in line:
		        self.getButtonByKey(key).setText(key[self.state])
	
    def addInputByKey(self, key):
        self.inputString += key[self.state]
        self.inputLine.setText(self.inputString)

    def backspace(self):
        self.inputLine.backspace()
        self.inputString = self.inputString[:-1]

    def space(self):
	self.inputString += " "
	self.inputLine.setText(self.inputString)

    def sizeHint(self):
        #return QSize(480,272)
	return QSize(500,300)

#------------------------------------------------------------------------------------------------


#File Dialog for multi-directory

#------------------------------------------------------------------------------------------------

#TODO give cred
class FileDialog(QFileDialog):
    def __init__(self, *args):
        QFileDialog.__init__(self, *args)
        self.setOption(self.DontUseNativeDialog, True)
        self.setFileMode(self.DirectoryOnly)

        for view in self.findChildren((QListView, QTreeView)):
            if isinstance(view.model(), QFileSystemModel):
                view.setSelectionMode(QAbstractItemView.ExtendedSelection)
		view.setDragEnabled(False)

#------------------------------------------------------------------------------------------------




#GLOBAL METHODS
#------------------------------------------------------------------------------------------------
"""
def displayToTenths(s):
	sArr = str(s).replace(':',' ').replace('.',' ').split()
	return (int(sArr[0]) * 600) + (int(sArr[1]) * 10) +  int(sArr[2])

def displayToSecs(s):
	sArr = str(s).replace(':',' ').replace('.',' ').strip("\"").split()
	return (int(sArr[0]) * 60) + (int(sArr[1])) + (int(sArr[2]) * 0.1)

def tenthsToDisplay(i):
	return "{}:{:02d}.{}".format(i / 600, (i / 10) % 60, i % 10)
"""

def displayToTenths(s):
	sArr = str(s).split('.')
	return (int(sArr[0]) * 600) + (int(sArr[1]) * 10)

def tenthsToDisplay(i):
	return "{:02d}.{}".format((i / 10), i % 10)

def createFolder(directory, deleteExisting=False):
	#be careful with deleting
	try:
		if not os.path.exists(directory):
			os.makedirs(directory)
		elif deleteExisting:
			shutil.rmtree(directory)
			os.makedirs(directory)
	except OSError:
		print ('ERROR	cannot create directory. ' + directory)

def updateDateTime():
	global DATETIME, DATE, TIME
	t = datetime.datetime.now()
	DATETIME = "{}-{}-{}_{:02d}.{:02d}.{:02d}".format(
				t.month, t.day, t.year,
				t.hour, t.minute, t.second)
	TIME = "{:02d}.{:02d}.{:02d}".format(t.hour, t.minute, t.second)
	DATE = "{}-{}-{}".format(t.month, t.day, t.year)
	print("Msg	Updated time to: " + DATETIME)

def testLookUp(mrn, date, time):
	name = "{}_{}_{}".format(mrn, date, time)
	if os.path.isdir(ROOT_DIR+"tests/{}".format(name)):
		return name
	else:
		print("ERROR	test folder {} could not be found".format(name))
	
#------------------------------------------------------------------------------------------------


if __name__ == "__main__":
	app = QApplication(sys.argv)
	window = MyApp()
	window.show()
	sys.exit(app.exec_())
	
