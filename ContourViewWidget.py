from __future__ import print_function
import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTableWidget,QTableWidgetItem,QVBoxLayout
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os
import pickle
import copy
import time

class ContourGraph(pg.GraphItem):
    def __init__(self):
        pg.GraphItem.__init__(self)
        
    def setData(self, **kwds):
        self.data = copy.deepcopy(kwds)
        if 'pos' in self.data:
            npts = self.data['pos'].shape[0]
            self.data['data'] = np.empty(npts, dtype=[('index', int)])
            self.data['data']['index'] = np.arange(npts)
        pg.GraphItem.setData(self, **self.data)

class ContourViewWidget(pg.GraphicsWindow):

	def __init__(self, parent=None):
		pg.GraphicsWindow.__init__(self)
		self.setParent(parent)

		self.w_sub = self.addLayout(row=0,col=0)

		self.mov_nr = 0

		self.v_list = []
		self.img_list = []
		self.contour_list_dest_body = []
		self.contour_list_dest_wing_L = []
		self.contour_list_dest_wing_R = []
		self.contour_list_init_body = []
		self.contour_list_init_wing_L = []
		self.contour_list_init_wing_R = []
		self.contour_list_outer = []

		self.contour_list_dest = []
		self.contour_list_init = []

		self.init_check = False
		self.dest_check = False
		self.src_check  = False

	def loadFLT(self,flt):
		self.flt = flt

	def setMovNR(self,mov_nr):
		self.mov_nr = mov_nr-1

	def set_init_view(self,check):
		self.init_check = check

	def set_dest_view(self,check):
		self.dest_check = check

	def set_src_view(self,check):
		self.src_check = check

	def add_frame(self,frame_nr):
		self.flt.load_frame(self.mov_nr,frame_nr)
		self.image_size = []
		frame_list = self.flt.get_frame()
		for i, frame in enumerate(frame_list):
			self.image_size.append(np.array([frame.shape[0],frame.shape[1]]))
			self.v_list.append(self.w_sub.addViewBox(row=1,col=i,lockAspect=True))
			self.img_list.append(pg.ImageItem(np.transpose(np.flipud(frame))))
			self.v_list[i].addItem(self.img_list[i])
			self.v_list[i].disableAutoRange('xy')
			self.v_list[i].autoRange()

	def update_frame(self,frame_nr):
		self.flt.load_frame(self.mov_nr,frame_nr)
		frame_list = self.flt.get_frame()
		for i, frame in enumerate(frame_list):
			self.img_list[i].setImage(np.transpose(np.flipud(frame)))
		self.update_contour()

	def add_contour(self, frame_nr):
		self.flt.load_frame(self.mov_nr,frame_nr)
		self.flt.segment_frame()
		self.flt.project_frame_2_pcl()
		self.flt.find_initial_state()
		dest_contour_list = self.flt.return_dest_contour()
		init_contour_list = self.flt.return_init_contour()
		# Create 10 empty contours in each window
		for i, contour_list in enumerate(dest_contour_list):
			self.contour_list_dest.append([])
			for j in range(10):
				self.contour_list_dest[i].append(pg.PlotCurveItem())
				self.contour_list_dest[i][j].setData(x=np.asarray([0.0]),y=np.asarray([0.0]),pen=(0,0,255))
				self.v_list[i].addItem(self.contour_list_dest[i][j])
		for i, contour_list in enumerate(init_contour_list):
			self.contour_list_init.append([])
			for j in range(10):
				self.contour_list_init[i].append(pg.PlotCurveItem())
				self.contour_list_init[i][j].setData(x=np.asarray([0.0]),y=np.asarray([0.0]),pen=(0,0,255))
				self.v_list[i].addItem(self.contour_list_init[i][j])

	def update_contour(self):
		self.flt.segment_frame()
		self.flt.project_frame_2_pcl()
		self.flt.find_initial_state()
		color_list = [(0,0,255), (255,0,0), (0,255,0)]
		dest_contour_list = self.flt.return_dest_contour()
		init_contour_list = self.flt.return_init_contour()
		
		for i, contour_list in enumerate(dest_contour_list):
			N_items = len(contour_list)
			for j in range(10):
				if (j<N_items):
					if (np.amax(contour_list[j][2,:])>0):
						color_now = color_list[int(np.amax(contour_list[j][2,:])-1)]
						self.contour_list_dest[i][j].setData(x=contour_list[j][0,:],y=self.image_size[i][1]-contour_list[j][1,:],pen=color_now)
					else:
						self.contour_list_dest[i][j].setData(x=np.asarray([0.0]),y=np.asarray([0.0]),pen=(0,0,255))
				else:
					self.contour_list_dest[i][j].setData(x=np.asarray([0.0]),y=np.asarray([0.0]),pen=(0,0,255))
		for i, contour_list in enumerate(init_contour_list):
			N_items = len(contour_list)
			for j in range(10):
				if (j<N_items):
					if (np.amax(contour_list[j][2,:])>0):
						color_now = color_list[int(np.amax(contour_list[j][2,:])-1)]
						self.contour_list_init[i][j].setData(x=contour_list[j][0,:],y=self.image_size[i][1]-contour_list[j][1,:],pen=color_now)
					else:
						self.contour_list_init[i][j].setData(x=np.asarray([0.0]),y=np.asarray([0.0]),pen=(0,0,255))
				else:
					self.contour_list_init[i][j].setData(x=np.asarray([0.0]),y=np.asarray([0.0]),pen=(0,0,255))