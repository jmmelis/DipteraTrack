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

class BBoxWidget(Qt.QFrame):

	def __init__(self, parent=None):
		Qt.QFrame.__init__(self, parent)

		self.vl = Qt.QVBoxLayout()
		self.vtkWidget = QVTKRenderWindowInteractor(self)
		self.vl.addWidget(self.vtkWidget)

		self.ren = vtk.vtkRenderer()
		self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
		self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()

		orig_axes = vtk.vtkAxesActor()
		orig_axes.SetAxisLabels(0)

		self.ren.AddActor(orig_axes)

		self.ren.ResetCamera()

		# Set the background color
		self.background = (0.1,0.2,0.4)
		self.ren.SetBackground(*self.background)

		self.pcl_view_on = False
		self.bbox_view_on = False
		self.model_view_on = False

		self.setLayout(self.vl)

		self.show()

		self.iren.Initialize()
		self.ren.ResetCameraClippingRange()
		self.vtkWidget.Render()

	def load_pointcloud(self,pointCloud,pcl_in):
		for k in range(pcl_in.shape[1]):
			point = np.array([pcl_in[0,k],pcl_in[1,k],pcl_in[2,k],pcl_in[3,k]])
			pointCloud.addPoint(point)
		return pointCloud

	def show_pointcloud(self,pcl_in):
		pointCloud = self.VtkPointCloud(np.amax(pcl_in[3,:]))
		pointCloud = self.load_pointcloud(pointCloud,pcl_in)
		self.ren.AddActor(pointCloud.vtkActor)

	def show_SRF_axes(self):
		M_SRF = self.flt.return_SRF_orientation()
		self.show_axes(M_SRF)

	def show_joints(self):
		joint_locs = self.flt.return_fixed_joint_locs()
		jL = vtk.vtkSphereSource()
		jL.SetCenter(joint_locs[0,0],joint_locs[1,0],joint_locs[2,0])
		jL.SetRadius(0.1)
		jL_mapper = vtk.vtkPolyDataMapper()
		jL_mapper.SetInputConnection(jL.GetOutputPort())
		jL_actor = vtk.vtkActor()
		jL_actor.SetMapper(jL_mapper)
		self.ren.AddActor(jL_actor)
		jR = vtk.vtkSphereSource()
		jR.SetCenter(joint_locs[0,1],joint_locs[1,1],joint_locs[2,1])
		jR.SetRadius(0.1)
		jR_mapper = vtk.vtkPolyDataMapper()
		jR_mapper.SetInputConnection(jR.GetOutputPort())
		jR_actor = vtk.vtkActor()
		jR_actor.SetMapper(jR_mapper)
		self.ren.AddActor(jR_actor)

	def show_bboxes(self,corner_points):
		N_boxes = corner_points.shape[1]
		for i in range(N_boxes):
			corner_mat = np.empty([8,3])
			for j in range(8):
				corner_mat[j,0] = corner_points[j*3,i]
				corner_mat[j,1] = corner_points[j*3+1,i]
				corner_mat[j,2] = corner_points[j*3+2,i]
			box = self.BoundingBox()
			box.addBox(corner_mat)
			self.ren.AddActor(box.vtkActor)

	def show_axes(self,M):
		axes = vtk.vtkAxesActor()
		axes.SetAxisLabels(0)
		M_vtk = vtk.vtkMatrix4x4()
		M_vtk.SetElement(0,0,M[0,0])
		M_vtk.SetElement(0,1,M[0,1])
		M_vtk.SetElement(0,2,M[0,2])
		M_vtk.SetElement(0,3,M[0,3])
		M_vtk.SetElement(1,0,M[1,0])
		M_vtk.SetElement(1,1,M[1,1])
		M_vtk.SetElement(1,2,M[1,2])
		M_vtk.SetElement(1,3,M[1,3])
		M_vtk.SetElement(2,0,M[2,0])
		M_vtk.SetElement(2,1,M[2,1])
		M_vtk.SetElement(2,2,M[2,2])
		M_vtk.SetElement(2,3,M[2,3])
		M_vtk.SetElement(3,0,M[3,0])
		M_vtk.SetElement(3,1,M[3,1])
		M_vtk.SetElement(3,2,M[3,2])
		M_vtk.SetElement(3,3,M[3,3])
		axes.SetUserMatrix(M_vtk)
		axes.Modified()
		self.ren.AddActor(axes)

	def clear_window(self):
		actors = self.ren.GetActors()
		for actor in actors:
			self.ren.RemoveActor(actor)

	def loadFLT(self,flt):
		self.flt = flt

	def setMov(self,mov_nr):
		self.mov_nr = mov_nr

	def setPCLView(self,pcl_view_on):
		self.pcl_view_on = pcl_view_on
		if pcl_view_on == True:
			self.bbox_view_on = False
			self.model_view_on = False

	def setBBoxView(self,bbox_view_on):
		self.bbox_view_on = bbox_view_on
		if bbox_view_on == True:
			self.pcl_view_on = False
			self.model_view_on = False

	def setModelView(self,model_view_on):
		self.model_view_on = model_view_on
		if model_view_on == True:
			self.pcl_view_on = False
			self.bbox_view_on = False

	def setMinNrPoints(self,min_nr_points):
		self.flt.set_min_nr_points(min_nr_points)

	def setMinConnPoints(self,min_conn_points):
		self.flt.set_min_conn_points(min_conn_points)

	def setSphereRadius(self,radius):
		self.flt.set_sphere_radius(radius)

	def setStrokeBound(self,bound_val):
		self.flt.set_stroke_bound(bound_val)

	def setDevBound(self,bound_val):
		self.flt.set_deviation_bound(bound_val)

	def setWingPitchBound(self,bound_val):
		self.flt.set_wing_pitch_bound(bound_val)

	def write_pcl_to_file(self,file_path,pcl_in):
		file_out = open(file_path,'w')
		for k in range(pcl_in.shape[1]):
			file_out.write('%.6f %.6f %.6f %.6f\n' % (pcl_in[0,k],pcl_in[1,k],pcl_in[2,k],pcl_in[3,k]) )
		file_out.close()

	def loadFrame(self,frame_nr):
		self.clear_window()
		self.flt.load_frame(self.mov_nr,frame_nr)
		self.flt.segment_frame()
		self.flt.project_frame_2_pcl()
		self.flt.find_initial_state()
		if self.pcl_view_on:
			seg_pcl = self.flt.return_seg_pcl()
			frame_name = "/home/johan/Documents/NEF_polyhedron_test/pcl_files/pcl_frame_" + str(frame_nr) + ".txt"
			if seg_pcl.shape[1]>0:
				self.show_pointcloud(seg_pcl)
				self.write_pcl_to_file(frame_name,seg_pcl)
		if self.bbox_view_on:
			self.show_SRF_axes()
			self.show_joints()
			wt_pcl = self.flt.return_wingtip_pcls()
			#frame_name = "/home/johan/Documents/NEF_polyhedron_test/pcl_files/pcl_frame_" + str(frame_nr) + ".txt"
			if wt_pcl.shape[0]>0:
				self.show_pointcloud(wt_pcl)
				#self.write_pcl_to_file(frame_name,wt_pcl)
			wt_bbox = self.flt.return_wingtip_boxes()
			if wt_bbox.shape[0]>0:
				self.show_bboxes(wt_bbox)
		if self.model_view_on:
			self.show_SRF_axes()
			self.show_joints()
			self.show_stl_files()
			seg_pcl = self.flt.return_seg_pcl()
			if seg_pcl.shape[1]>0:
				self.show_pointcloud(seg_pcl)
		self.vtkWidget.Render()

	def load_stl_files(self,file_list,file_loc):
		os.chdir(file_loc)
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
			self.stl_actor_list.append(stl_actor)
			self.stl_properties.append(stl_actor.GetProperty())

	def show_stl_files(self):
		M = self.flt.return_initial_state()
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

	class VtkPointCloud:
		def __init__(self,scalar_range):
			self.vtkPolyData = vtk.vtkPolyData()
			self.clearPoints()
			mapper = vtk.vtkPolyDataMapper()
			mapper.SetInputData(self.vtkPolyData)
			mapper.SetColorModeToDefault()
			mapper.SetScalarRange(0.0,scalar_range)
			mapper.SetScalarVisibility(1)
			self.vtkActor = vtk.vtkActor()
			self.vtkActor.SetMapper(mapper)

		def addPoint(self,point):
			pointID = self.vtkPoints.InsertNextPoint(point[0:3])
			self.vtkDepth.InsertNextValue(point[3])
			self.vtkCells.InsertNextCell(1)
			self.vtkCells.InsertCellPoint(pointID)
			self.vtkCells.Modified()
			self.vtkPoints.Modified()
			self.vtkDepth.Modified()

		def addNormal(self,point,normal,scale):
			pointID1 = self.vtkPoints.InsertNextPoint(point[0:3])
			pointID2 = self.vtkPoints.InsertNextPoint([point[0]+scale*normal[0],point[1]+scale*normal[1],point[2]+scale*normal[2]])
			self.vtkDepth.InsertNextValue(point[3])
			self.vtkCells.InsertNextCell(2)
			self.vtkCells.InsertCellPoint(pointID1)
			self.vtkCells.InsertCellPoint(pointID2)
			self.vtkCells.Modified()
			self.vtkPoints.Modified()
			self.vtkDepth.Modified()

		def clearPoints(self):
			self.vtkPoints = vtk.vtkPoints()
			self.vtkCells = vtk.vtkCellArray()
			self.vtkDepth = vtk.vtkDoubleArray()
			self.vtkDepth.SetName('DepthArray')
			self.vtkPolyData.SetPoints(self.vtkPoints)
			self.vtkPolyData.SetVerts(self.vtkCells)
			#self.vtkPolyData.SetLines(self.vtkCells)
			self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
			self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')

	class BoundingBox:
		def __init__(self):
			self.mapper = vtk.vtkPolyDataMapper()
			self.vtkActor = vtk.vtkActor()
			self.vtkActor.SetMapper(self.mapper)

		def addBox(self,corner_points):
			# Add a bounding box
			points = vtk.vtkPoints()
			points.SetNumberOfPoints(8)
			points.SetPoint(0,corner_points[0,0],corner_points[0,1],corner_points[0,2])
			points.SetPoint(1,corner_points[1,0],corner_points[1,1],corner_points[1,2])
			points.SetPoint(2,corner_points[2,0],corner_points[2,1],corner_points[2,2])
			points.SetPoint(3,corner_points[3,0],corner_points[3,1],corner_points[3,2])
			points.SetPoint(4,corner_points[4,0],corner_points[4,1],corner_points[4,2])
			points.SetPoint(5,corner_points[5,0],corner_points[5,1],corner_points[5,2])
			points.SetPoint(6,corner_points[6,0],corner_points[6,1],corner_points[6,2])
			points.SetPoint(7,corner_points[7,0],corner_points[7,1],corner_points[7,2])
			lines = vtk.vtkCellArray()
			lines.InsertNextCell(5)
			lines.InsertCellPoint(0)
			lines.InsertCellPoint(1)
			lines.InsertCellPoint(2)
			lines.InsertCellPoint(3)
			lines.InsertCellPoint(0)
			lines.InsertNextCell(5)
			lines.InsertCellPoint(4)
			lines.InsertCellPoint(5)
			lines.InsertCellPoint(6)
			lines.InsertCellPoint(7)
			lines.InsertCellPoint(4)
			lines.InsertNextCell(5)
			lines.InsertCellPoint(0)
			lines.InsertCellPoint(4)
			lines.InsertCellPoint(7)
			lines.InsertCellPoint(3)
			lines.InsertCellPoint(0)
			lines.InsertNextCell(5)
			lines.InsertCellPoint(1)
			lines.InsertCellPoint(5)
			lines.InsertCellPoint(6)
			lines.InsertCellPoint(2)
			lines.InsertCellPoint(1)
			lines.InsertNextCell(5)
			lines.InsertCellPoint(0)
			lines.InsertCellPoint(1)
			lines.InsertCellPoint(5)
			lines.InsertCellPoint(4)
			lines.InsertCellPoint(0)
			lines.InsertNextCell(5)
			lines.InsertCellPoint(3)
			lines.InsertCellPoint(2)
			lines.InsertCellPoint(6)
			lines.InsertCellPoint(7)
			lines.InsertCellPoint(3)
			polygon = vtk.vtkPolyData()
			polygon.SetPoints(points)
			polygon.SetLines(lines)
			self.mapper.SetInputData(polygon)
			self.mapper.Update()