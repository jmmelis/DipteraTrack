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
import math
import pickle
import os
import copy
import time

from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from flight_tracker_class import Flight_Tracker_Class
from diptera_track_ui import Ui_MainWindow
from SessionParamWidget import SessionParamWidget
from ScaleModelWidget import ScaleModelWidget
from ModelViewWidget import ModelViewWidget
from ImageSegmentWidget import ImageSegmentWidget
from BoundingBoxWidget import BBoxWidget
from ContourViewWidget import ContourViewWidget

class DipteraTrack(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(DipteraTrack,self).__init__(parent)
        self.setupUi(self)

        # Session parameters:
        self.session_loc = '...'
        self.session_name = '...'
        self.bckg_loc = '...'
        self.bckg_img_format = '...'
        self.cal_loc = '...'
        self.cal_name = '...'
        self.cal_img_format = '...'
        self.bckg_img_list = []
        self.mov_name_list = []
        self.cam_name_list = []
        self.N_mov = 0
        self.N_cam = 0
        self.frame_name = '...'
        self.frame_img_format = '...'
        self.model_loc = '...'
        self.model_name = '...'
        self.start_frame = 0
        self.trig_frame = 0
        self.end_frame = 0
        self.trigger_mode = '...'

        self.mov_1_name = '...'
        self.mov_2_name = '...'
        self.mov_3_name = '...'
        self.mov_4_name = '...'
        self.mov_5_name = '...'
        self.mov_6_name = '...'
        self.mov_7_name = '...'
        self.mov_8_name = '...'

        self.cam_1_name = '...'
        self.cam_2_name = '...'
        self.cam_3_name = '...'
        self.cam_4_name = '...'
        self.cam_5_name = '...'
        self.cam_6_name = '...'

        self.img_formats = ['...','bmp','tif']

        self.trig_modes = ['...','start','center','end']

        self.ses_par_file = ''

        # Focal grid parameters:
        self.nx = None
        self.ny = None
        self.nz = None
        self.ds = None
        self.x0 = None
        self.y0 = None
        self.z0 = None
        self.pixel_size = None

        self.flt = Flight_Tracker_Class()

        self.initialize()

        self.connectActions()

    def initialize(self):
        self.add_session_param_gui()
        self.add_focal_grid_gui()
        self.add_model_scale_gui()
        self.add_model_view_gui()
        self.add_image_segment_gui()
        self.add_pcl_view_gui()
        self.add_contour_view_gui()

    def connectActions(self):
        self.connect_session_param_gui()
        self.connect_focal_grid_gui()
        self.connect_model_scale_gui()
        self.connect_model_view_gui()
        self.connect_image_segment_gui()
        self.connect_pcl_view_gui()
        self.connect_contour_view_gui()

    def add_session_param_gui(self):
        self.ses_folder_rbtn.toggled.connect(lambda:self.ses_folder_rbtn)
        self.bckg_folder_rbtn.toggled.connect(lambda:self.bckg_folder_rbtn)
        self.cal_folder_rbtn.toggled.connect(lambda:self.cal_folder_rbtn)
        self.mov_folder1_rbtn.toggled.connect(lambda:self.mov_folder1_rbtn)
        self.mov_folder2_rbtn.toggled.connect(lambda:self.mov_folder2_rbtn)
        self.mov_folder3_rbtn.toggled.connect(lambda:self.mov_folder3_rbtn)
        self.mov_folder4_rbtn.toggled.connect(lambda:self.mov_folder4_rbtn)
        self.mov_folder5_rbtn.toggled.connect(lambda:self.mov_folder5_rbtn)
        self.mov_folder6_rbtn.toggled.connect(lambda:self.mov_folder6_rbtn)
        self.mov_folder7_rbtn.toggled.connect(lambda:self.mov_folder7_rbtn)
        self.mov_folder8_rbtn.toggled.connect(lambda:self.mov_folder8_rbtn)
        self.cam_folder1_rbtn.toggled.connect(lambda:self.cam_folder1_rbtn)
        self.cam_folder2_rbtn.toggled.connect(lambda:self.cam_folder2_rbtn)
        self.cam_folder3_rbtn.toggled.connect(lambda:self.cam_folder3_rbtn)
        self.cam_folder4_rbtn.toggled.connect(lambda:self.cam_folder4_rbtn)
        self.cam_folder5_rbtn.toggled.connect(lambda:self.cam_folder5_rbtn)
        self.cam_folder6_rbtn.toggled.connect(lambda:self.cam_folder6_rbtn)
        self.frame_name_rbtn.toggled.connect(lambda:self.frame_name_rbtn)
        self.mdl_loc_rbtn.toggled.connect(lambda:self.mdl_loc_rbtn)
        self.mdl_name_rbtn.toggled.connect(lambda:self.mdl_name_rbtn)
        self.load_settings_rbtn.toggled.connect(lambda:self.load_settings_rbtn)

        self.bck_img_fmt_box.addItem(self.img_formats[0])
        self.bck_img_fmt_box.addItem(self.img_formats[1])
        self.bck_img_fmt_box.addItem(self.img_formats[2])

        self.cal_img_fmt_box.addItem(self.img_formats[0])
        self.cal_img_fmt_box.addItem(self.img_formats[1])
        self.cal_img_fmt_box.addItem(self.img_formats[2])

        self.frame_img_fmt_box.addItem(self.img_formats[0])
        self.frame_img_fmt_box.addItem(self.img_formats[1])
        self.frame_img_fmt_box.addItem(self.img_formats[2])

        self.trig_mode_box.addItem(self.trig_modes[0])
        self.trig_mode_box.addItem(self.trig_modes[1])
        self.trig_mode_box.addItem(self.trig_modes[2])
        self.trig_mode_box.addItem(self.trig_modes[3])

        self.start_frame_spin.setMinimum(0)
        self.start_frame_spin.setMaximum(0)
        self.start_frame_spin.setValue(0)
        self.start_frame_spin.valueChanged.connect(self.set_start_frame)

        self.trig_frame_spin.setMinimum(0)
        self.trig_frame_spin.setMaximum(0)
        self.trig_frame_spin.setValue(0)
        self.trig_frame_spin.valueChanged.connect(self.set_trig_frame)

        self.end_frame_spin.setMinimum(0)
        self.end_frame_spin.setMaximum(0)
        self.end_frame_spin.setValue(0)
        self.end_frame_spin.valueChanged.connect(self.set_end_frame)

        self.reset_selection_push_btn.clicked.connect(self.reset_selection)
        self.load_settings_push_btn.clicked.connect(self.load_session_file)
        self.save_settings_push_btn.clicked.connect(self.save_session_file)

        directory = '/media/flyami'
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(directory)
        self.folder_select_tree.setModel(self.file_model)
        self.folder_select_tree.setRootIndex(self.file_model.index(directory));

    def connect_session_param_gui(self):
        self.tabs.setTabEnabled(1, False)
        self.folder_select_tree.clicked.connect(self.select_folder_file)
        self.bck_img_fmt_box.currentIndexChanged.connect(self.set_bckg_img_format)
        self.cal_img_fmt_box.currentIndexChanged.connect(self.set_cal_img_format)
        self.frame_img_fmt_box.currentIndexChanged.connect(self.set_frame_img_format)
        self.trig_mode_box.currentIndexChanged.connect(self.set_trigger_mode)
        self.start_session_push_btn.clicked.connect(self.start_session)

    def select_folder_file(self, index):
        indexItem = self.file_model.index(index.row(), 0, index.parent())
        fileName = self.file_model.fileName(indexItem)
        filePath = self.file_model.filePath(indexItem)
        if self.ses_folder_rbtn.isChecked():
            self.session_loc = filePath
            self.session_name = fileName
            self.ses_folder_label.setText(self.session_loc)
            self.ses_name_label.setText(self.session_name)
        elif self.bckg_folder_rbtn.isChecked():
            self.bckg_loc = fileName
            self.bckg_folder_label.setText(self.bckg_loc)
        elif self.cal_folder_rbtn.isChecked():
            self.cal_loc = fileName
            self.cal_folder_label.setText(self.cal_loc)
            try:
                for file_name in os.listdir(self.session_loc + '/' + self.cal_loc):
                    if file_name.endswith("calib.txt"):
                        self.cal_name = file_name
                        self.cal_file_label.setText(self.cal_name)
            except:
                self.cal_name = '...'
                self.cal_file_label.setText(self.cal_name)
        elif self.mov_folder1_rbtn.isChecked():
            self.mov_1_name = fileName
            self.mov_folder1_label.setText(fileName)
        elif self.mov_folder2_rbtn.isChecked():
            self.mov_2_name = fileName
            self.mov_folder2_label.setText(fileName)
        elif self.mov_folder3_rbtn.isChecked():
            self.mov_3_name = fileName
            self.mov_folder3_label.setText(fileName)
        elif self.mov_folder4_rbtn.isChecked():
            self.mov_4_name = fileName
            self.mov_folder4_label.setText(fileName)
        elif self.mov_folder5_rbtn.isChecked():
            self.mov_5_name = fileName
            self.mov_folder5_label.setText(fileName)
        elif self.mov_folder6_rbtn.isChecked():
            self.mov_6_name = fileName
            self.mov_folder6_label.setText(fileName)
        elif self.mov_folder7_rbtn.isChecked():
            self.mov_7_name = fileName
            self.mov_folder7_label.setText(fileName)
        elif self.mov_folder8_rbtn.isChecked():
            self.mov_8_name = fileName
            self.mov_folder8_label.setText(fileName)
        elif self.cam_folder1_rbtn.isChecked():
            self.cam_1_name = fileName
            self.cam_folder1_label.setText(fileName)
        elif self.cam_folder2_rbtn.isChecked():
            self.cam_2_name = fileName
            self.cam_folder2_label.setText(fileName)
        elif self.cam_folder3_rbtn.isChecked():
            self.cam_3_name = fileName
            self.cam_folder3_label.setText(fileName)
        elif self.cam_folder4_rbtn.isChecked():
            self.cam_4_name = fileName
            self.cam_folder4_label.setText(fileName)
        elif self.cam_folder5_rbtn.isChecked():
            self.cam_5_name = fileName
            self.cam_folder5_label.setText(fileName)
        elif self.cam_folder6_rbtn.isChecked():
            self.cam_6_name = fileName
            self.cam_folder6_label.setText(fileName)
        elif self.frame_name_rbtn.isChecked():
            try:
                frame_name_split = fileName.split('_',1)
                self.frame_name = frame_name_split[0] + '_'
                self.frame_name_label.setText(self.frame_name)
            except:
                self.frame_name = '...'
                self.frame_name_label.setText(self.frame_name)
        elif self.mdl_loc_rbtn.isChecked():
            self.model_loc = filePath
            self.mdl_loc_label.setText(self.model_loc)
        elif self.mdl_name_rbtn.isChecked():
            self.model_name = fileName
            self.mdl_name_label.setText(self.model_name)
        elif self.load_settings_rbtn.isChecked():
            self.ses_par_file = filePath
            self.load_settings_file_label.setText(self.ses_par_file)

    def reset_selection(self):
        if self.ses_folder_rbtn.isChecked():
            self.session_loc = '...'
            self.session_name = '...'
            self.ses_folder_label.setText(self.session_loc)
            self.ses_name_label.setText(self.session_name)
        elif self.bckg_folder_rbtn.isChecked():
            self.bckg_loc = '...'
            self.bckg_folder_label.setText(self.bckg_loc)
        elif self.cal_folder_rbtn.isChecked():
            self.cal_loc = '...'
            self.cal_folder_label.setText(self.cal_loc)
            self.cal_name = '...'
            self.cal_file_label.setText(self.cal_name)
        elif self.mov_folder1_rbtn.isChecked():
            self.mov_1_name = '...'
            self.mov_folder1_label.setText(self.mov_1_name)
        elif self.mov_folder2_rbtn.isChecked():
            self.mov_2_name = '...'
            self.mov_folder2_label.setText(self.mov_2_name)
        elif self.mov_folder3_rbtn.isChecked():
            self.mov_3_name = '...'
            self.mov_folder3_label.setText(self.mov_3_name)
        elif self.mov_folder4_rbtn.isChecked():
            self.mov_4_name = '...'
            self.mov_folder4_label.setText(self.mov_4_name)
        elif self.mov_folder5_rbtn.isChecked():
            self.mov_5_name = '...'
            self.mov_folder5_label.setText(self.mov_5_name)
        elif self.mov_folder6_rbtn.isChecked():
            self.mov_6_name = '...'
            self.mov_folder6_label.setText(self.mov_6_name)
        elif self.mov_folder7_rbtn.isChecked():
            self.mov_7_name = '...'
            self.mov_folder7_label.setText(self.mov_7_name)
        elif self.mov_folder8_rbtn.isChecked():
            self.mov_8_name = '...'
            self.mov_folder8_label.setText(self.mov_8_name)
        elif self.cam_folder1_rbtn.isChecked():
            self.cam_1_name = '...'
            self.cam_folder1_label.setText(self.cam_1_name)
        elif self.cam_folder2_rbtn.isChecked():
            self.cam_2_name = '...'
            self.cam_folder2_label.setText(self.cam_2_name)
        elif self.cam_folder3_rbtn.isChecked():
            self.cam_3_name = '...'
            self.cam_folder3_label.setText(self.cam_3_name)
        elif self.cam_folder4_rbtn.isChecked():
            self.cam_4_name = '...'
            self.cam_folder4_label.setText(self.cam_4_name)
        elif self.cam_folder5_rbtn.isChecked():
            self.cam_5_name = '...'
            self.cam_folder5_label.setText(self.cam_5_name)
        elif self.cam_folder6_rbtn.isChecked():
            self.cam_6_name = '...'
            self.cam_folder6_label.setText(self.cam_6_name)
        elif self.frame_name_rbtn.isChecked():
            self.frame_name = '...'
            self.frame_name_label.setText(self.frame_name)
        elif self.mdl_loc_rbtn.isChecked():
            self.model_loc = '...'
            self.mdl_loc_label.setText(self.model_loc)
        elif self.mdl_name_rbtn.isChecked():
            self.model_name = '...'
            self.mdl_name_label.setText(self.model_name)

    def set_start_frame(self,val):
        self.start_frame = val

    def set_trig_frame(self,val):
        self.trig_frame = val

    def set_end_frame(self,val):
        self.end_frame = val

    def set_bckg_img_format(self,ind):
        self.bckg_img_format = self.img_formats[ind]

    def set_cal_img_format(self,ind):
        self.cal_img_format = self.img_formats[ind]

    def set_frame_img_format(self,ind):
        self.frame_img_format = self.img_formats[ind]

    def set_trigger_mode(self,ind):
        self.trigger_mode = self.trig_modes[ind]
        if (ind == 0):
            self.start_frame_spin.setMinimum(0)
            self.start_frame_spin.setMaximum(0)
            self.start_frame_spin.setValue(0)
            self.trig_frame_spin.setMinimum(0)
            self.trig_frame_spin.setMaximum(0)
            self.trig_frame_spin.setValue(0)
            self.end_frame_spin.setMinimum(0)
            self.end_frame_spin.setMaximum(0)
            self.end_frame_spin.setValue(0)
            self.start_frame = 0
            self.trig_frame = 0
            self.end_frame = 0
        elif (ind == 1):
            # Start trigger mode:
            try:
                file_count = 0
                for file_name in os.listdir(self.session_loc + '/' + self.mov_1_name + '/' + self.cam_1_name):
                    if file_name.endswith(self.frame_img_format):
                        file_count += 1
                frame_start = 0
                frame_trig = 0
                frame_end = file_count-1
            except:
                frame_start = 0
                frame_trig = 0
                frame_end = 0
            self.start_frame_spin.setMinimum(frame_start)
            self.start_frame_spin.setMaximum(frame_trig)
            self.start_frame_spin.setValue(frame_start)
            self.trig_frame_spin.setMinimum(frame_start)
            self.trig_frame_spin.setMaximum(frame_end)
            self.trig_frame_spin.setValue(frame_trig)
            self.end_frame_spin.setMinimum(frame_trig)
            self.end_frame_spin.setMaximum(frame_end)
            self.end_frame_spin.setValue(frame_end)
            self.start_frame = frame_start
            self.trig_frame = frame_trig
            self.end_frame = frame_end
        elif (ind == 2):
            # Center trigger mode:
            try:
                file_count = 0
                for file_name in os.listdir(self.session_loc + '/' + self.mov_1_name + '/' + self.cam_1_name):
                    if file_name.endswith(self.frame_img_format):
                        file_count += 1
                frame_start = 0
                frame_trig = int(np.ceil(file_count/2.0))
                frame_end = file_count-1
            except:
                frame_start = 0
                frame_trig = 0
                frame_end = 0
            self.start_frame_spin.setMinimum(frame_start)
            self.start_frame_spin.setMaximum(frame_trig)
            self.start_frame_spin.setValue(frame_start)
            self.trig_frame_spin.setMinimum(frame_start)
            self.trig_frame_spin.setMaximum(frame_end)
            self.trig_frame_spin.setValue(frame_trig)
            self.end_frame_spin.setMinimum(frame_trig)
            self.end_frame_spin.setMaximum(frame_end)
            self.end_frame_spin.setValue(frame_end)
            self.start_frame = frame_start
            self.trig_frame = frame_trig
            self.end_frame = frame_end
        elif (ind == 3):
            # End trigger mode:
            try:
                file_count = 0
                for file_name in os.listdir(self.session_loc + '/' + self.mov_1_name + '/' + self.cam_1_name):
                    if file_name.endswith(self.frame_img_format):
                        file_count += 1
                frame_start = 0
                frame_trig = file_count-1
                frame_end = file_count-1
            except:
                frame_start = 0
                frame_trig = 0
                frame_end = 0
            self.start_frame_spin.setMinimum(frame_start)
            self.start_frame_spin.setMaximum(frame_trig)
            self.start_frame_spin.setValue(frame_start)
            self.trig_frame_spin.setMinimum(frame_start)
            self.trig_frame_spin.setMaximum(frame_end)
            self.trig_frame_spin.setValue(frame_trig)
            self.end_frame_spin.setMinimum(frame_trig)
            self.end_frame_spin.setMaximum(frame_end)
            self.end_frame_spin.setValue(frame_end)
            self.start_frame = frame_start
            self.trig_frame = frame_trig
            self.end_frame = frame_end

    def start_session(self):
        # Check parameters
        par_okay = True

        if (self.session_loc == '...'):
            par_okay = False
            print('No session folder')
        if (self.session_name == '...'):
            par_okay = False
            print('No session name')
        if (self.bckg_loc == '...'):
            par_okay = False
            print('No background folder')
        if (self.bckg_img_format == '...'):
            par_okay = False
            print('No background image format')
        if (self.cal_loc == '...'):
            par_okay = False
            print('No calibration folder')
        if (self.cal_name == '...'):
            par_okay = False
            print('No calibration file')
        if (self.cal_img_format == '...'):
            par_okay = False
            print('No calibration image format')
        if (self.mov_1_name == '...'):
            par_okay = False
            print('No movie folders selected')
        if (self.cam_1_name == '...'):
            par_okay = False
            print('No camera folders selected')
        if (self.frame_name == '...'):
            par_okay = False
            print('No frame name')
        if (self.frame_img_format == '...'):
            par_okay = False
            print('No frame image format')
        if (self.model_loc == '...'):
            par_okay = False
            print('No model folder')
        if (self.model_name == '...'):
            par_okay = False
            print('No model selected')
        if (self.trigger_mode == '...'):
            par_okay = False
            print('No trigger mode selected')

        try:
            self.bckg_img_list = []
            for file_name in os.listdir(self.session_loc + '/' + self.bckg_loc):
                if file_name.endswith(self.bckg_img_format):
                    self.bckg_img_list.append(str(file_name))
            self.bckg_img_list.sort()
            #print(self.bckg_img_list)
        except:
            par_okay = False
            self.bckg_img_list = []
            print('Could not find background images')

        
        if (par_okay):

            if (self.mov_1_name != '...'):
                self.mov_name_list.append(str(self.mov_1_name))
            if (self.mov_2_name != '...'):
                self.mov_name_list.append(str(self.mov_2_name))
            if (self.mov_3_name != '...'):
                self.mov_name_list.append(str(self.mov_3_name))
            if (self.mov_4_name != '...'):
                self.mov_name_list.append(str(self.mov_4_name))
            if (self.mov_5_name != '...'):
                self.mov_name_list.append(str(self.mov_5_name))
            if (self.mov_6_name != '...'):
                self.mov_name_list.append(str(self.mov_6_name))
            if (self.mov_7_name != '...'):
                self.mov_name_list.append(str(self.mov_7_name))
            if (self.mov_8_name != '...'):
                self.mov_name_list.append(str(self.mov_8_name))

            self.mov_name_list.sort()

            self.N_mov = len(self.mov_name_list)

            print('Number of movies: ' + str(self.N_mov))
            print(self.mov_name_list)

            if (self.cam_1_name != '...'):
                self.cam_name_list.append(str(self.cam_1_name))
            if (self.cam_2_name != '...'):
                self.cam_name_list.append(str(self.cam_2_name))
            if (self.cam_3_name != '...'):
                self.cam_name_list.append(str(self.cam_3_name))
            if (self.cam_4_name != '...'):
                self.cam_name_list.append(str(self.cam_4_name))
            if (self.cam_5_name != '...'):
                self.cam_name_list.append(str(self.cam_5_name))
            if (self.cam_6_name != '...'):
                self.cam_name_list.append(str(self.cam_6_name))

            self.cam_name_list.sort()

            self.N_cam = len(self.cam_name_list)

            print('Number of cameras: ' + str(self.N_cam))
            print(self.cam_name_list)

            # Transfer parameters to the flight_tracker_class.py
            self.flt.N_cam = self.N_cam
            self.flt.N_mov = self.N_mov
            self.flt.session_loc = str(self.session_loc)
            self.flt.session_name = str(self.session_name)
            self.flt.bckg_loc = str(self.bckg_loc)
            self.flt.bckg_img_list = self.bckg_img_list
            self.flt.bckg_img_format = str(self.bckg_img_format)
            self.flt.cal_loc = str(self.cal_loc)
            self.flt.cal_name = str(self.cal_name)
            self.flt.cal_img_format = str(self.cal_img_format)
            self.flt.mov_name_list = self.mov_name_list
            self.flt.cam_name_list = self.cam_name_list
            self.flt.frame_name = str(self.frame_name)
            self.flt.frame_img_format = str(self.frame_img_format)
            self.flt.model_loc = str(self.model_loc)
            self.flt.model_name = str(self.model_name)
            self.flt.start_point = self.start_frame
            self.flt.trig_point = self.trig_frame
            self.flt.end_point = self.end_frame
            self.flt.trigger_mode = str(self.trigger_mode)

            self.flt.set_parameters()

            self.flt.print_parameters()

            self.tabs.setTabEnabled(1, True)

    def load_session_file(self):
        # Load session parameter file
        try:
            pickle_in = open(self.ses_par_file,'rb')
            ses_dict = pickle.load(pickle_in)
            self.session_loc = ses_dict['session_loc']
            self.session_name = ses_dict['session_name']
            self.bckg_loc = ses_dict['bckg_loc']
            self.bckg_img_list = ses_dict['bckg_img_list']
            self.bckg_img_format = ses_dict['bckg_img_format']
            self.cal_loc = ses_dict['cal_loc']
            self.cal_name = ses_dict['cal_name']
            self.cal_img_format = ses_dict['cal_img_format']
            self.mov_1_name = ses_dict['mov_1_name']
            self.mov_2_name = ses_dict['mov_2_name']
            self.mov_3_name = ses_dict['mov_3_name']
            self.mov_4_name = ses_dict['mov_4_name']
            self.mov_5_name = ses_dict['mov_5_name']
            self.mov_6_name = ses_dict['mov_6_name']
            self.mov_7_name = ses_dict['mov_7_name']
            self.mov_8_name = ses_dict['mov_8_name']
            self.cam_1_name = ses_dict['cam_1_name']
            self.cam_2_name = ses_dict['cam_2_name']
            self.cam_3_name = ses_dict['cam_3_name']
            self.cam_4_name = ses_dict['cam_4_name']
            self.cam_5_name = ses_dict['cam_5_name']
            self.cam_6_name = ses_dict['cam_6_name']
            self.frame_name = ses_dict['frame_name']
            self.frame_img_format = ses_dict['frame_img_format']
            self.model_loc = ses_dict['model_loc']
            self.model_name = ses_dict['model_name']
            self.trigger_mode = ses_dict['trigger_mode']
            self.start_frame = ses_dict['start_frame']
            self.trig_frame = ses_dict['trig_frame']
            self.end_frame = ses_dict['end_frame']
            print('loaded session parameters')
        except:
            print('could not load session parameter file')

    def save_session_file(self):
        # Save session paramter file
        try:
            ses_dict = {
                'session_loc': self.session_loc,
                'session_name': self.session_name,
                'bckg_loc': self.bckg_loc,
                'bckg_img_list': self.bckg_img_list,
                'bckg_img_format': self.bckg_img_format,
                'cal_loc': self.cal_loc,
                'cal_name': self.cal_name,
                'cal_img_format': self.cal_img_format,
                'mov_1_name': self.mov_1_name,
                'mov_2_name': self.mov_2_name,
                'mov_3_name': self.mov_3_name,
                'mov_4_name': self.mov_4_name,
                'mov_5_name': self.mov_5_name,
                'mov_6_name': self.mov_6_name,
                'mov_7_name': self.mov_7_name,
                'mov_8_name': self.mov_8_name,
                'cam_1_name': self.cam_1_name,
                'cam_2_name': self.cam_2_name,
                'cam_3_name': self.cam_3_name,
                'cam_4_name': self.cam_4_name,
                'cam_5_name': self.cam_5_name,
                'cam_6_name': self.cam_6_name,
                'frame_name': self.frame_name,
                'frame_img_format': self.frame_img_format,
                'model_loc': self.model_loc,
                'model_name': self.model_name,
                'trigger_mode': self.trigger_mode,
                'start_frame': self.start_frame,
                'trig_frame': self.trig_frame,
                'end_frame': self.end_frame
            }
            os.chdir(self.session_loc)
            ses_pickle = open('session_parameters.pickle','wb')
            pickle.dump(ses_dict,ses_pickle)
            ses_pickle.close()
            print('saved session_parameters.pickle')
        except:
            print('could not save session parameter file')

    def set_nx(self,val):
        self.nx = val

    def set_ny(self,val):
        self.ny = val

    def set_nz(self,val):
        self.nz = val

    def set_ds(self,val):
        self.ds = val

    def set_x0(self,val):
        self.x0 = val

    def set_y0(self,val):
        self.y0 = val

    def set_z0(self,val):
        self.z0 = val

    def set_pixel_size(self,val):
        self.pixel_size = val

    def add_focal_grid_gui(self):
        self.nx = 320
        self.ny = 320
        self.nz = 320
        self.ds = 0.032
        self.x0 = 0.0
        self.y0 = 0.0
        self.z0 = 0.0
        self.pixel_size = 0.040

    def connect_focal_grid_gui(self):
        self.tabs.setTabEnabled(2, False)

        self.nx_spin.setMinimum(1)
        self.nx_spin.setMaximum(2048)
        self.nx_spin.setValue(self.nx)
        self.nx_spin.valueChanged.connect(self.set_nx)

        self.ny_spin.setMinimum(1)
        self.ny_spin.setMaximum(2048)
        self.ny_spin.setValue(self.ny)
        self.ny_spin.valueChanged.connect(self.set_ny)

        self.nz_spin.setMinimum(1)
        self.nz_spin.setMaximum(2048)
        self.nz_spin.setValue(self.nz)
        self.nz_spin.valueChanged.connect(self.set_nz)

        self.ds_spin.setMinimum(0.001)
        self.ds_spin.setMaximum(1.000)
        self.ds_spin.setSingleStep(0.001)
        self.ds_spin.setValue(self.ds)
        self.ds_spin.valueChanged.connect(self.set_ds)

        self.x0_spin.setRange(-10.0,10.0)
        self.x0_spin.setSingleStep(0.01)
        self.x0_spin.setValue(self.x0)
        self.x0_spin.valueChanged.connect(self.set_x0)

        self.y0_spin.setRange(-10.0,10.0)
        self.y0_spin.setSingleStep(0.01)
        self.y0_spin.setValue(self.y0)
        self.y0_spin.valueChanged.connect(self.set_y0)

        self.z0_spin.setRange(-10.0,10.0)
        self.z0_spin.setSingleStep(0.01)
        self.z0_spin.setValue(self.z0)
        self.z0_spin.valueChanged.connect(self.set_z0)

        self.pixel_size_spin.setRange(0.001,0.1)
        self.pixel_size_spin.setSingleStep(0.001)
        self.pixel_size_spin.setValue(self.pixel_size)
        self.pixel_size_spin.valueChanged.connect(self.set_pixel_size)

        self.calc_vox_btn.clicked.connect(self.calculate_focal_grid)

        self.vox_progress_bar.setRange(0,100)
        self.vox_progress_bar.setValue(0)

    def update_grid_progress_bar(self,progress):
        self.vox_progress_bar.setValue(progress)

    def calculate_focal_grid(self):
        self.flt.nx = self.nx
        self.flt.ny = self.ny
        self.flt.nz = self.nz
        self.flt.ds = self.ds
        self.flt.x0 = self.x0
        self.flt.y0 = self.y0
        self.flt.z0 = self.z0
        self.flt.pixel_size = self.pixel_size
        self.flt.set_focal_grid_parameters()
        self.flt.set_pixel_size(self.pixel_size)
        self.flt.calculate_focal_grid(self.update_grid_progress_bar)
        self.tabs.setTabEnabled(2, True)
        self.rawFrameView.setFrameParam()
        self.set_model_scale_gui()

    def add_model_scale_gui(self):
        self.tabs.setTabEnabled(2, False)

    def connect_model_scale_gui(self):
        self.tabs.setTabEnabled(3, False)
        self.rawFrameView.loadFLT(self.flt)

    def set_model_scale_gui(self):

        drag_pnt_data = self.flt.get_drag_point_pars()
        symbols = drag_pnt_data[0]
        labels = drag_pnt_data[1]
        adj = drag_pnt_data[4]
        scale_texts = drag_pnt_data[6]
        scale_calc = drag_pnt_data[7]
        length_calc = drag_pnt_data[8]
        origin_ind = drag_pnt_data[9]
        contour_calc = drag_pnt_data[10]
        scale_data = []
        for i in range(len(scale_texts)):
            scale_data.append(1.0)

        xyz_pos = drag_pnt_data[3]
        pos_list = []
        for i in range(xyz_pos.shape[0]):
            pos_list.append((xyz_pos[i,0],xyz_pos[i,1],xyz_pos[i,2]))
        start_pos = np.array(pos_list)

        p_clr = drag_pnt_data[2]
        p_clr_list = []
        for i in range(p_clr.shape[0]):
            p_clr_list.append((p_clr[i,0],p_clr[i,1],p_clr[i,2]))
        pnt_clr = np.array([p_clr_list], dtype=[('red',np.ubyte),('green',np.ubyte),('blue',np.ubyte)])

        l_clr = drag_pnt_data[5]
        l_clr_list = []
        for i in range(l_clr.shape[0]):
            l_clr_list.append((l_clr[i,0],l_clr[i,1],l_clr[i,2],l_clr[i,3],l_clr[i,4]))
        line_clr = np.array([l_clr_list],
            dtype=[('red',np.ubyte),('green',np.ubyte),('blue',np.ubyte),('alpha',np.ubyte),('width',float)])

        self.rawFrameView.setSessionLoc(self.session_loc)
        self.rawFrameView.setPos(start_pos)
        self.rawFrameView.setAdj(adj)
        self.rawFrameView.setLines(l_clr)
        self.rawFrameView.setSymbols(symbols)
        self.rawFrameView.setTexts(labels)
        self.rawFrameView.setTextColor(p_clr)
        self.rawFrameView.setScaleCalc(scale_calc)
        self.rawFrameView.setLengthCalc(length_calc)
        self.rawFrameView.setOriginInd(origin_ind)
        self.rawFrameView.add_frame(self.start_frame)
        self.rawFrameView.add_graph()
        self.rawFrameView.setMouseCallbacks()

        self.raw_mov_spin.setMinimum(1)
        self.raw_mov_spin.setMaximum(self.N_mov)
        self.raw_mov_spin.setValue(1)
        self.raw_mov_spin.valueChanged.connect(self.rawFrameView.setMovNR)

        self.raw_frame_spin.setMinimum(self.start_frame)
        self.raw_frame_spin.setMaximum(self.end_frame)
        self.raw_frame_spin.setValue(self.start_frame)
        self.raw_frame_spin.valueChanged.connect(self.rawFrameView.update_frame)

        self.scaleTable.setRowCount(2)
        self.scaleTable.setColumnCount(len(scale_texts))
        for i in range(len(scale_texts)):
            self.scaleTable.setItem(0,i,QTableWidgetItem(scale_texts[i]))
            self.scaleTable.setItem(1,i,QTableWidgetItem(str(scale_data[i])))
        self.rawFrameView.connect_table(self.scaleTable)

        self.load_scale_btn.clicked.connect(self.rawFrameView.load_model_scale)
        self.save_scale_btn.clicked.connect(self.rawFrameView.save_model_scale)
        self.set_model_btn.clicked.connect(self.update_model_scale)

    def update_model_scale(self):
        self.tabs.setTabEnabled(3, True)
        self.model_view_window.set_file_loc(self.model_loc)
        self.rawFrameView.update_model_scale()
        self.model_view_window.load_model()
        self.tabs.setTabEnabled(4, True)
        self.set_image_segment_gui()

    def add_model_view_gui(self):
        self.tabs.setTabEnabled(3, False)

    def connect_model_view_gui(self):
        self.tabs.setTabEnabled(4, False)
        self.model_view_window.loadFLT(self.flt)

    def add_image_segment_gui(self):
        self.tabs.setTabEnabled(4, False)

    def connect_image_segment_gui(self):
        self.tabs.setTabEnabled(5, False)

        self.body_thresh = 40
        self.wing_thresh = 20
        self.sigma = 0.05
        self.K = 3000
        self.min_body_size = 50
        self.min_wing_size = 3

        self.seg_view.loadFLT(self.flt)
        self.seg_view.setBodyThresh(self.body_thresh)
        self.seg_view.setWingThresh(self.wing_thresh)
        self.seg_view.setSigma(self.sigma)
        self.seg_view.setK(self.K)
        self.seg_view.setMinBodySize(self.min_body_size)
        self.seg_view.setMinWingSize(self.min_wing_size)
        #self.seg_view.setTethered(self.tethered)

    def set_image_segment_gui(self):
        print("resetting mask")
        self.seg_view.reset_mask()
        self.seg_view.add_frame(self.flt.start_point)

        self.seg_mov_spin.setMinimum(1)
        self.seg_mov_spin.setMaximum(self.N_mov)
        self.seg_mov_spin.setValue(1)
        self.seg_mov_spin.valueChanged.connect(self.seg_view.update_mov)

        self.seg_frame_spin.setMinimum(self.flt.start_point)
        self.seg_frame_spin.setMaximum(self.flt.end_point)
        self.seg_mov_spin.setValue(self.flt.start_point)
        self.seg_frame_spin.valueChanged.connect(self.seg_view.update_frame)

        self.body_thresh_spin.setMinimum(0)
        self.body_thresh_spin.setMaximum(255)
        self.body_thresh_spin.setValue(self.body_thresh)
        self.body_thresh_spin.valueChanged.connect(self.seg_view.setBodyThresh)

        self.wing_thresh_spin.setMinimum(0)
        self.wing_thresh_spin.setMaximum(255)
        self.wing_thresh_spin.setValue(self.wing_thresh)
        self.wing_thresh_spin.valueChanged.connect(self.seg_view.setWingThresh)

        self.sigma_spin.setMinimum(0.0)
        self.sigma_spin.setMaximum(2.0)
        self.sigma_spin.setSingleStep(0.01)
        self.sigma_spin.setValue(self.sigma)
        self.sigma_spin.valueChanged.connect(self.seg_view.setSigma)

        self.K_spin.setMinimum(0)
        self.K_spin.setMaximum(10000)
        self.K_spin.setSingleStep(100)
        self.K_spin.setValue(self.K)
        self.K_spin.valueChanged.connect(self.seg_view.setK)

        self.min_body_spin.setMinimum(0)
        self.min_body_spin.setMaximum(10000)
        self.min_body_spin.setValue(self.min_body_size)
        self.min_body_spin.valueChanged.connect(self.seg_view.setMinBodySize)

        self.min_wing_spin.setMinimum(0)
        self.min_wing_spin.setMaximum(10000)
        self.min_wing_spin.setValue(self.min_wing_size)
        self.min_wing_spin.valueChanged.connect(self.seg_view.setMinWingSize)

        #self.tethered_check.setChecked(self.tethered)
        #self.tethered_check.stateChanged.connect(self.seg_view.setTethered)

        self.seg_update_btn.clicked.connect(self.seg_view.update_parameters)

        self.mask_cam_nr_spin.setMinimum(1)
        self.mask_cam_nr_spin.setMaximum(self.flt.N_cam)
        self.mask_cam_nr_spin.setValue(1)
        self.mask_cam_nr_spin.valueChanged.connect(self.seg_view.set_mask_view)

        self.mask_seg_nr_spin.setMinimum(1)
        self.mask_seg_nr_spin.setMaximum(1)
        self.mask_seg_nr_spin.setValue(1)
        self.mask_seg_nr_spin.valueChanged.connect(self.seg_view.set_mask_segment)

        self.seg_view.add_mask_spin(self.mask_seg_nr_spin,self.flt.N_cam)

        self.add_mask_btn.clicked.connect(self.seg_view.add_mask)

        self.reset_mask_btn.clicked.connect(self.seg_view.reset_mask)

        self.continue_btn.clicked.connect(self.set_pcl_view_gui)

    def add_pcl_view_gui(self):
        self.tabs.setTabEnabled(5, False)

    def connect_pcl_view_gui(self):
        self.min_nr_points = 50
        self.min_conn_points = 50
        self.sphere_radius = 0.25
        self.phi_bound = 120
        self.theta_bound = 50
        self.eta_bound = 180
        self.tethered = True

        self.pcl_view.loadFLT(self.flt)
        self.flt.set_tethered_flight(self.tethered)
        self.flt.set_min_conn_points(self.min_nr_points)
        self.flt.set_min_nr_points(self.min_conn_points)
        self.flt.set_stroke_bound(self.phi_bound)
        self.flt.set_deviation_bound(self.theta_bound)
        self.flt.set_wing_pitch_bound(self.eta_bound)

    def set_pcl_view_gui(self):
        self.tabs.setTabEnabled(5, True)

        self.pcl_view.setMov(1)
        self.pcl_view.loadFrame(self.flt.start_point)

        self.pcl_mov_spin.setMinimum(1)
        self.pcl_mov_spin.setMaximum(self.N_mov)
        self.pcl_mov_spin.setValue(1)
        self.pcl_mov_spin.valueChanged.connect(self.pcl_view.setMov)

        self.pcl_frame_spin.setMinimum(self.flt.start_point)
        self.pcl_frame_spin.setMaximum(self.flt.end_point)
        self.pcl_frame_spin.setValue(self.flt.start_point)
        self.pcl_frame_spin.valueChanged.connect(self.load_pcl_view_frame)

        self.sphere_radius_spin.setMinimum(0.0)
        self.sphere_radius_spin.setMaximum(2.0)
        self.sphere_radius_spin.setSingleStep(0.01)
        self.sphere_radius_spin.setValue(self.sphere_radius)
        self.pcl_view.setSphereRadius(self.sphere_radius)
        self.sphere_radius_spin.valueChanged.connect(self.pcl_view.setSphereRadius)

        self.tethered_radio_btn.toggled.connect(self.set_tethered_flight)
        self.free_radio_btn.toggled.connect(self.set_free_flight)

        self.stroke_bound_spin.setMinimum(0)
        self.stroke_bound_spin.setMaximum(120)
        self.stroke_bound_spin.setValue(self.phi_bound)
        self.stroke_bound_spin.valueChanged.connect(self.pcl_view.setStrokeBound)

        self.dev_bound_spin.setMinimum(0)
        self.dev_bound_spin.setMaximum(90)
        self.dev_bound_spin.setValue(self.theta_bound)
        self.dev_bound_spin.valueChanged.connect(self.pcl_view.setDevBound)

        self.wing_pitch_bound_spin.setMinimum(0)
        self.wing_pitch_bound_spin.setMaximum(180)
        self.wing_pitch_bound_spin.setValue(self.eta_bound)
        self.wing_pitch_bound_spin.valueChanged.connect(self.pcl_view.setWingPitchBound)

        self.pcl_view_btn.toggled.connect(lambda:self.pcl_view_btn)
        self.bbox_view_btn.toggled.connect(lambda:self.bbox_view_btn)
        self.model_view_btn.toggled.connect(lambda:self.model_view_btn)

        stl_list = self.flt.get_stl_list()
        self.pcl_view.load_stl_files(stl_list,self.model_loc)

        self.set_contour_view_gui()

    def set_free_flight(self):
        self.tethered = False
        self.flt.set_tethered_flight(self.tethered)

    def set_tethered_flight(self):
        self.tethered = True
        self.flt.set_tethered_flight(self.tethered)

    def load_pcl_view_frame(self,frame_nr):
        if self.pcl_view_btn.isChecked():
            self.pcl_view.setPCLView(True)
        elif self.bbox_view_btn.isChecked():
            self.pcl_view.setBBoxView(True)
        elif self.model_view_btn.isChecked():
            self.pcl_view.setModelView(True)
        self.pcl_view.loadFrame(frame_nr)

    def add_contour_view_gui(self):
        self.tabs.setTabEnabled(6, False)

    def connect_contour_view_gui(self):
        self.contour_view.loadFLT(self.flt)

    def set_contour_view_gui(self):

        self.tabs.setTabEnabled(6, True)

        self.opt_mov_spin.setMinimum(1)
        self.opt_mov_spin.setMaximum(self.N_mov)
        self.opt_mov_spin.setValue(1)
        self.opt_mov_spin.valueChanged.connect(self.contour_view.setMovNR)

        self.contour_view.add_frame(self.start_frame)

        self.contour_view.add_contour(self.start_frame)

        self.opt_frame_spin.setMinimum(self.start_frame)
        self.opt_frame_spin.setMaximum(self.end_frame)
        self.opt_frame_spin.setValue(self.start_frame)
        self.opt_frame_spin.valueChanged.connect(self.contour_view.update_frame)

        self.init_view_check.toggled.connect(self.contour_view.set_init_view)
        self.dest_view_check.toggled.connect(self.contour_view.set_dest_view)
        self.src_view_check.toggled.connect(self.contour_view.set_src_view)

def appMain():
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = DipteraTrack()
    mainWindow.show()
    app.exec_()

# -------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    appMain()