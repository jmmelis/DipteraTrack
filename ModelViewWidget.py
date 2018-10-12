from __future__ import print_function
import sys
import vtk
from PyQt5 import QtCore, QtGui
from PyQt5 import Qt
import numpy as np
import os
import copy
import time

from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

class ModelViewWidget(Qt.QFrame):

	def __init__(self, parent=None):
		Qt.QFrame.__init__(self, parent)

		self.vl = Qt.QVBoxLayout()
		self.vtkWidget = QVTKRenderWindowInteractor(self)
		self.vl.addWidget(self.vtkWidget)

		self.ren = vtk.vtkRenderer()
		self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
		self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()

		axes = vtk.vtkAxesActor()
		axes.SetAxisLabels(0)
		self.ren.AddActor(axes)

		self.ren.ResetCamera()

		self.model_list = []

		# Set the background color
		self.background = (0.1,0.2,0.4)
		self.ren.SetBackground(*self.background)
		self.setLayout(self.vl)
		self.show()

		self.iren.Initialize()
		self.ren.ResetCameraClippingRange()
		self.vtkWidget.Render()

	def set_file_loc(self,file_loc):
		self.file_loc = file_loc

	def load_stl_files(self,file_list):
		os.chdir(self.file_loc)
		self.stl_list = []
		self.mapper_list = []
		self.stl_actor_list = []
		self.stl_properties = []
		for stl_file in file_list:
			stl_reader = vtk.vtkSTLReader()
			stl_reader.SetFileName(stl_file)
			self.stl_list.append(stl_reader)
			mapper = vtk.vtkPolyDataMapper()
			mapper.ScalarVisibilityOff()
			mapper.SetInputConnection(stl_reader.GetOutputPort())
			self.mapper_list.append(mapper)
			stl_actor = vtk.vtkActor()
			stl_actor.SetMapper(mapper)
			stl_actor.GetProperty().SetOpacity(1.0)
			self.stl_actor_list.append(stl_actor)
			self.stl_properties.append(stl_actor.GetProperty())

	def show_stl_files(self):
		M = self.flt.get_start_state()
		scale = self.flt.get_model_scales()
		for i, actor in enumerate(self.stl_actor_list):
			M_vtk = vtk.vtkMatrix4x4()
			M_vtk.SetElement(0,0,M[0,i])
			M_vtk.SetElement(0,1,M[1,i])
			M_vtk.SetElement(0,2,M[2,i])
			M_vtk.SetElement(0,3,M[3,i])
			M_vtk.SetElement(1,0,M[4,i])
			M_vtk.SetElement(1,1,M[5,i])
			M_vtk.SetElement(1,2,M[6,i])
			M_vtk.SetElement(1,3,M[7,i])
			M_vtk.SetElement(2,0,M[8,i])
			M_vtk.SetElement(2,1,M[9,i])
			M_vtk.SetElement(2,2,M[10,i])
			M_vtk.SetElement(2,3,M[11,i])
			M_vtk.SetElement(3,0,M[12,i])
			M_vtk.SetElement(3,1,M[13,i])
			M_vtk.SetElement(3,2,M[14,i])
			M_vtk.SetElement(3,3,M[15,i])
			actor.SetUserMatrix(M_vtk)
			actor.SetScale(scale[i],scale[i],scale[i])
			actor.Modified()
			self.ren.AddActor(actor)

	def add_model(self,new_model):
		self.model_list.append(new_model)

	def loadFLT(self,flt):
		self.flt = flt

	def load_model(self):
		self.clear_window()
		stl_list = self.flt.get_stl_list()
		self.load_stl_files(stl_list)
		self.show_stl_files()

	def clear_window(self):
		actors = self.ren.GetActors()
		for actor in actors:
			self.ren.RemoveActor(actor)