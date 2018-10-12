from __future__ import print_function
import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTableWidget,QTableWidgetItem,QVBoxLayout
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import copy
import time

class Graph(pg.GraphItem):
    def __init__(self,graph_nr):
    	self.graph_nr = graph_nr
        self.dragPoint = None
        self.dragOffset = None
        self.textItems = []
        pg.GraphItem.__init__(self)
        self.scatter.sigClicked.connect(self.clicked)
        self.onMouseDragCb = None
        
    def setData(self, **kwds):
        self.text = kwds.pop('text', [])
        self.data = copy.deepcopy(kwds)
        
        if 'pos' in self.data:
            npts = self.data['pos'].shape[0]
            self.data['data'] = np.empty(npts, dtype=[('index', int)])
            self.data['data']['index'] = np.arange(npts)
        self.setTexts(self.text,self.data)
        self.updateGraph()
        
    def setTexts(self, text, data):
        for i in self.textItems:
            i.scene().removeItem(i)
        self.textItems = []
        #for t in text:
        for i,t in enumerate(text):
            item = pg.TextItem(t)
            if len(data.keys())>0:
            	item.setColor(data['textcolor'][i])
            self.textItems.append(item)
            item.setParentItem(self)
        
    def updateGraph(self):
        pg.GraphItem.setData(self, **self.data)
        for i,item in enumerate(self.textItems):
            item.setPos(*self.data['pos'][i])

    def setOnMouseDragCallback(self, callback):
    	self.onMouseDragCb = callback
        
    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return
        
        if ev.isStart():
            # We are already one step into the drag.
            # Find the point(s) at the mouse cursor when the button was first 
            # pressed:
            pos = ev.buttonDownPos()
            pts = self.scatter.pointsAt(pos)
            if len(pts) == 0:
                ev.ignore()
                return
            self.dragPoint = pts[0]
            ind = pts[0].data()[0]
            self.dragOffset = self.data['pos'][ind] - pos
        elif ev.isFinish():
            self.dragPoint = None
            return
        else:
            if self.dragPoint is None:
                ev.ignore()
                return
        
        ind = self.dragPoint.data()[0]
        self.data['pos'][ind] = ev.pos() + self.dragOffset
        self.updateGraph()
        ev.accept()
        if self.onMouseDragCb:
        	PosData = self.data['pos'][ind]
        	PosData = np.append(PosData,ind)
        	PosData = np.append(PosData,self.graph_nr)
        	self.onMouseDragCb(PosData)
        
    def clicked(self, pts):
        print("clicked: %s" % pts)

class ScaleModelWidget(pg.GraphicsWindow):

	def __init__(self, parent=None):
		pg.GraphicsWindow.__init__(self)
		self.setParent(parent)

		self.w_sub = self.addLayout(row=0,col=0)

		self.mov_nr = 0

		self.v_list = []
		self.img_list = []
		self.graph_list = []

	def loadFLT(self,flt):
		self.flt = flt

	def setFrameParam(self):
		self.N_cam = self.flt.N_cam
		self.image_size = self.flt.get_image_size()

	def setPos(self,xyz_pos):
		self.N_drag_points = xyz_pos.shape[0]
		self.xyz_pos = xyz_pos
		self.pos = []
		for i in range(self.N_cam):
			self.pos.append(np.empty([self.N_drag_points,2]))
			for j in range(self.N_drag_points):
				uv_points = self.flt.convert_3D_point_2_uv(xyz_pos[j,0],xyz_pos[j,1],xyz_pos[j,2])
				self.pos[i][j,0] = uv_points[i,0]
				self.pos[i][j,1] = self.image_size[i,1]-uv_points[i,1]

	def setAdj(self,adj):
		self.adj = adj

	def setLines(self,lines):
		self.lines = lines

	def setSymbols(self,symbols):
		self.symbols = symbols

	def setTexts(self,texts):
		self.texts = texts

	def setTextColor(self,text_color):
		self.txt_color = text_color

	def setScaleCalc(self,scale_calc):
		self.scale_calc = scale_calc
		self.scales = []
		for i in range(len(self.scale_calc)):
			self.scales.append(0.0)

	def setLengthCalc(self,length_calc):
		self.length_calc = length_calc
		self.lengths = []
		for i in range(len(self.length_calc)):
			self.lengths.append(0.0)

	def setOriginInd(self,ind_list):
		self.origin_ind = []
		for ind in ind_list:
			self.origin_ind.append(ind)

	def setSessionLoc(self,session_loc):
		self.session_loc = session_loc

	def setMovNR(self,mov_nr):
		self.mov_nr = mov_nr-1

	def add_frame(self,frame_nr):
		self.flt.load_frame(self.mov_nr,frame_nr)
		frame_list = self.flt.get_frame()
		for i, frame in enumerate(frame_list):
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

	def add_graph(self):
		for i in range(self.N_cam):
			self.graph_list.append(Graph(i))
			self.v_list[i].addItem(self.graph_list[i])
			self.graph_list[i].setData(pos=self.pos[i], adj=self.adj, pen=self.lines, size=3, symbol=self.symbols, pxMode=False, text=self.texts, textcolor=self.txt_color)

	def update_graph(self,drag_data):

		# Calculate new 3D position and new uv positions

		cam_nr = int(drag_data[3])
		point_nr = int(drag_data[2])
		u_now = int(drag_data[0])
		v_now = int(self.image_size[cam_nr,1]-drag_data[1])
		u_prev = int(self.pos[cam_nr][point_nr,0])
		v_prev = int(self.image_size[cam_nr,1]-self.pos[cam_nr][point_nr,1])
		x_prev = self.xyz_pos[point_nr,0]
		y_prev = self.xyz_pos[point_nr,1]
		z_prev = self.xyz_pos[point_nr,2]

		new_pos = self.flt.drag_point_3D(cam_nr, u_now, v_now, u_prev, v_prev, x_prev, y_prev, z_prev)

		# Update new 3D position and new uv positions

		self.xyz_pos[point_nr,:] = new_pos[:,0]

		self.calculate_scales()

		self.calculate_lengths()

		self.update_table()
		
		for i in range(self.N_cam):
			self.pos[i][point_nr,0] = new_pos[0,i+1]
			self.pos[i][point_nr,1] = self.image_size[i,1]-new_pos[1,i+1]

		for i in range(self.N_cam):
			self.graph_list[i].data['pos'][point_nr,:] = self.pos[i][point_nr,:]
			self.graph_list[i].updateGraph()

	def setMouseCallbacks(self):
		def onMouseDragCallback(data):
			self.update_graph(data)

		for i in range(self.N_cam):
			self.graph_list[i].setOnMouseDragCallback(onMouseDragCallback)

	def calculate_scales(self):
		for i,calc in enumerate(self.scale_calc):
			self.scales[i] = np.sqrt((self.xyz_pos[calc[0],0]-self.xyz_pos[calc[1],0])**2+
								(self.xyz_pos[calc[0],1]-self.xyz_pos[calc[1],1])**2+
								(self.xyz_pos[calc[0],2]-self.xyz_pos[calc[1],2])**2)

	def calculate_lengths(self):
		for i,calc in enumerate(self.length_calc):
			self.lengths[i] = np.sqrt((self.xyz_pos[calc[0],0]-self.xyz_pos[calc[1],0])**2+
								(self.xyz_pos[calc[0],1]-self.xyz_pos[calc[1],1])**2+
								(self.xyz_pos[calc[0],2]-self.xyz_pos[calc[1],2])**2)

	def connect_table(self,table_item):
		self.table_item = table_item

	def update_table(self):
		for i in range(len(self.scales)):
			self.table_item.setItem(1,i,QTableWidgetItem(str(self.scales[i])))
	
	def update_model_scale(self):
		self.flt.set_body_length(self.lengths[0])
		self.flt.set_wing_length(self.lengths[1],self.lengths[2])
		self.flt.set_model_scale(self.scales)
		self.flt.set_model_joint_locs(self.xyz_pos[self.origin_ind[0],:],self.xyz_pos[self.origin_ind[1],:],self.xyz_pos[self.origin_ind[2],:],self.xyz_pos[self.origin_ind[3],:])

	def save_model_scale(self):
		# Save model scale file
		try:
			scale_dict = {
				'lengths': self.lengths,
				'scales': self.scales,
				'xyz_pos': self.xyz_pos,
				'origin_ind': self.origin_ind
			}
			print(scale_dict)
			print(self.session_loc)
			os.chdir(self.session_loc)
			scale_pickle = open('model_scale.pickle','wb')
			pickle.dump(scale_dict,scale_pickle)
			scale_pickle.close()
			print('saved model_scale.pickle')
		except:
			print('could not save model_scale.pickle')

	def load_model_scale(self):
		# Load model scale file
		try:
			os.chdir(self.session_loc)
			pickle_in = open('model_scale.pickle','rb')
			scale_dict = pickle.load(pickle_in)
			self.lengths = scale_dict['lengths']
			self.scales = scale_dict['scales']
			self.xyz_pos = scale_dict['xyz_pos']
			self.origin_ind = scale_dict['origin_ind']
			print('loaded model_scale.pickle')
			self.update_model_scale()
		except:
			print("could not load model_scale.pickle")