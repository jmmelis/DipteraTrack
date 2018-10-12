from __future__ import print_function
import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTableWidget,QTableWidgetItem,QVBoxLayout
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import time

class ImageSegmentWidget(pg.GraphicsWindow):

	def __init__(self, parent=None):
		pg.GraphicsWindow.__init__(self)
		self.setParent(parent)

		self.w_sub = self.addLayout(row=0,col=0)

		self.mov_nr = 0
		self.cam_nr = 0
		self.seg_nr = 0

		self.v_list = []
		self.img_list = []
		self.frame_list = []

	def loadFLT(self,flt):
		self.flt = flt

	def setBodyThresh(self,body_thresh):
		self.body_thresh = body_thresh

	def setWingThresh(self,wing_thresh):
		self.wing_thresh = wing_thresh

	def setSigma(self,sigma):
		self.sigma = sigma

	def setK(self,K):
		self.K = K

	def setMinBodySize(self,min_body_size):
		self.min_body_size = min_body_size

	def setMinWingSize(self,min_wing_size):
		self.min_wing_size = min_wing_size

	def setTethered(self,tethered):
		self.tethered = tethered

	def update_mov(self,mov_nr):
		self.mov_nr = mov_nr-1

	def add_frame(self,frame_nr):
		self.frame_nr = frame_nr
		self.N_cam = self.flt.N_cam
		self.image_size = self.flt.get_image_size()
		self.flt.load_frame(self.mov_nr, frame_nr)
		self.flt.set_segmentation_param(self.body_thresh,self.wing_thresh,self.sigma,self.K,self.min_body_size,self.min_wing_size)
		self.flt.segment_frame()
		frame_list = self.flt.return_segmented_frame()
		for i, frame in enumerate(frame_list):
			self.v_list.append(self.w_sub.addViewBox(row=1,col=i,lockAspect=True))
			frame_jet = self.jet_color(np.transpose(np.flipud(frame)))
			self.frame_list.append(frame)
			self.img_list.append(pg.ImageItem(frame_jet))
			self.v_list[i].addItem(self.img_list[i])
			self.v_list[i].disableAutoRange('xy')
			self.v_list[i].autoRange()

	def set_mask_segment(self,segment_nr):
		self.show_segment_view()
		self.seg_nr = segment_nr
		self.remove_segment_from_view(segment_nr)

	def set_mask_view(self,cam_nr):
		self.cam_nr = cam_nr-1
		self.update_spin_max()

	def add_mask(self):
		self.flt.set_mask(self.cam_nr,self.seg_nr)
		self.update_frame(self.frame_nr)
		print("added mask: " + str(self.seg_nr))

	def add_mask_spin(self,mask_spin,N_cam):
		self.mask_spin = mask_spin
		self.spin_max_list = []
		for i in range(N_cam):
			self.spin_max_list.append(1)

	def update_spin_max(self):
		self.mask_spin.setMaximum(self.spin_max_list[self.cam_nr])

	def reset_mask(self):
		self.flt.reset_mask()
		try:
			self.update_frame(self.frame_nr)
		except:
			print("")
	
	def update_frame(self,frame_nr):
		self.frame_nr = frame_nr
		self.flt.load_frame(self.mov_nr, frame_nr)
		self.flt.set_segmentation_param(self.body_thresh,self.wing_thresh,self.sigma,self.K,self.min_body_size,self.min_wing_size)
		self.flt.segment_frame()
		frame_list = self.flt.return_segmented_frame()
		for i, frame in enumerate(frame_list):
			frame_jet = self.jet_color(np.transpose(np.flipud(frame)))
			self.frame_list[i] = frame
			self.img_list[i].setImage(frame_jet)
			self.spin_max_list[i] = np.amax(frame)
		self.update_spin_max()

	def show_segment_view(self):
		frame = self.frame_list[self.cam_nr].copy()
		frame_jet = self.jet_color(np.transpose(np.flipud(frame)))
		self.img_list[self.cam_nr].setImage(frame_jet)

	def remove_segment_from_view(self,segment_nr):
		frame = self.frame_list[self.cam_nr].copy()
		frame[frame == segment_nr] = 0
		frame_jet = self.jet_color(np.transpose(np.flipud(frame)))
		self.img_list[self.cam_nr].setImage(frame_jet)

	def update_parameters(self):
		self.flt.set_segmentation_param(self.body_thresh,self.wing_thresh,self.sigma,self.K,self.min_body_size,self.min_wing_size)
		self.flt.segment_frame()
		frame_list = self.flt.return_segmented_frame()
		for i, frame in enumerate(frame_list):
			frame_jet = self.jet_color(np.transpose(np.flipud(frame)))
			self.img_list[i].setImage(frame_jet)

	def jet_color(self,frame):
		norm = mpl.colors.Normalize()
		color_img = plt.cm.jet(norm(frame.astype(float)))*255
		return color_img