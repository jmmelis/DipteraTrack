from __future__ import print_function
import sys
import vtk
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from PyQt5 import Qt
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTreeView, QFileSystemModel, QTableWidget, QTableWidgetItem, QVBoxLayout
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import time

from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from diptera_track_ui import Ui_MainWindow

class SessionParamWidget(Qt.QWidget):

	def __init__(self, parent=None):
		Qt.QWidget.__init__(self, parent)

		self.file_model = QFileSystemModel()

	def set_directory(self,directory):
		self.directory = directory
		self.file_model.setRootPath(directory)

	def setup_tree(self):
		self.folder_select_tree.setModel(self.file_model)
		self.folder_select_tree.setRootIndex(self.file_model.index(self.directory));

	def connect_tree(self):
		self.folder_select_tree.clicked.connect(self.set_session_folder)

	def set_session_folder(self, index):
		indexItem = self.file_model.index(index.row(), 0, index.parent())
		fileName = self.file_model.fileName(indexItem)
		filePath = self.file_model.filePath(indexItem)

		print(fileName)
		print(filePath)

class CheckableDirModel(QtGui.QDirModel):
	def __init__(self, parent=None):
	    QtGui.QDirModel.__init__(self, None)
	    self.checks = {}

	def data(self, index, role=QtCore.Qt.DisplayRole):
	    if role != QtCore.Qt.CheckStateRole:
	        return QtGui.QDirModel.data(self, index, role)
	    else:
	        if index.column() == 0:
	            return self.checkState(index)

	def flags(self, index):
	    return QtGui.QDirModel.flags(self, index) | QtCore.Qt.ItemIsUserCheckable

	def checkState(self, index):
	    if index in self.checks:
	        return self.checks[index]
	    else:
	        return QtCore.Qt.Unchecked

	def setData(self, index, value, role):
	    if (role == QtCore.Qt.CheckStateRole and index.column() == 0):
	        self.checks[index] = value
	        self.emit(QtCore.SIGNAL("dataChanged(QModelIndex,QModelIndex)"), index, index)
	        return True 

	    return QtGui.QDirModel.setData(self, index, value, role)