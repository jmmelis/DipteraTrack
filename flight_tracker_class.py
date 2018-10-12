import os
import numpy as np
import math
import time
import vtk
import threading
from threading import Thread
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
from time import sleep
print(os.getcwd())
sys.path.append(os.getcwd()+'/build')
import FlightTracker_lib

class Flight_Tracker_Class():

    # initialization

    def __init__(self):

        self.N_cam = None
        self.N_mov = None

        self.session_loc = ""
        self.session_name = ""
        self.bckg_loc = ""
        self.bckg_img_list = []
        self.bckg_img_format = ""
        self.cal_loc = ""
        self.cal_name = ""
        self.cal_img_format = ""
        self.mov_name_list = []
        self.cam_name_list = []
        self.frame_name = ""
        self.frame_img_format = ""
        self.model_loc = ""
        self.model_name = ""

        self.start_point = None
        self.trig_point = None
        self.end_point = None
        self.trigger_mode = ""

        self.nx = None
        self.ny = None
        self.nz = None
        self.ds = None
        self.x0 = None
        self.y0 = None
        self.z0 = None

        self.N_threads = None

        # C++ class
        self.flt = FlightTracker_lib.FLT()

    def set_parameters(self):
        self.flt.set_session_loc(self.session_loc, self.session_name)
        self.flt.set_backgroud_subtract(self.bckg_loc, self.bckg_img_list, self.bckg_img_format)
        self.flt.set_calibration(self.cal_loc, self.cal_name, self.cal_img_format)
        self.flt.set_movie_list(self.mov_name_list)
        self.flt.set_cam_list(self.cam_name_list)
        self.flt.set_frame_par(self.frame_name, self.frame_img_format)
        self.flt.set_model_par(self.model_loc, self.model_name)
        self.flt.set_trigger_par(self.trigger_mode, self.start_point, self.trig_point, self.end_point)
        self.flt.set_session_par()
        self.flt.set_frame_loader()
        self.flt.load_model()

    def print_parameters(self):
        self.flt.display_session_par()

    def set_focal_grid_parameters(self):
        self.flt.set_focal_grid(self.nx, self.ny, self.nz, self.ds, self.x0, self.y0, self.z0)
        self.flt.display_focal_grid_par()

    def set_pixel_size(self,pixel_size):
        self.flt.set_pixel_size(pixel_size)

    def calculate_focal_grid(self,progress_callback):
        self.flt.calc_focal_grid(progress_callback)
        return 1.0

    def convert_3D_point_2_uv(self,x_in,y_in,z_in):
        return self.flt.xyz_2_uv(x_in,y_in,z_in)

    def drag_point_3D(self,cam_nr, u_now, v_now, u_prev, v_prev, x_prev, y_prev, z_prev):
        return self.flt.drag_point_3d(cam_nr, u_now, v_now, u_prev, v_prev, x_prev, y_prev, z_prev)

    def get_image_size(self):
        return self.flt.get_image_size()

    def load_frame(self,mov_nr,frame_nr):
        self.flt.load_frame(mov_nr,frame_nr)

    def get_frame(self):
        frame = []
        for i in range(0,self.N_cam):
            frame.append(self.flt.return_frame(i))
        return frame

    def get_drag_point_pars(self):
        drag_point_data = []
        drag_point_data.append(self.flt.drag_pt_symbols())
        drag_point_data.append(self.flt.drag_pt_labels())
        drag_point_data.append(self.flt.drag_pt_colors())
        drag_point_data.append(self.flt.drag_pt_start())
        drag_point_data.append(self.flt.drag_ln_connect())
        drag_point_data.append(self.flt.drag_ln_colors())
        drag_point_data.append(self.flt.drag_scale_texts())
        drag_point_data.append(self.flt.drag_scale_calc())
        drag_point_data.append(self.flt.drag_length_calc())
        drag_point_data.append(self.flt.drag_origin_ind())
        drag_point_data.append(self.flt.drag_contour_calc())
        return drag_point_data

    def set_body_length(self,body_length):
        self.flt.set_body_length(body_length)

    def set_wing_length(self,wing_length_L,wing_length_R):
        self.flt.set_wing_length(wing_length_L,wing_length_R)

    def set_stroke_bound(self,stroke_bound):
        phi_bound = stroke_bound*(np.pi/180.0)
        self.flt.set_stroke_bound(phi_bound)

    def set_deviation_bound(self,deviation_bound):
        theta_bound = deviation_bound*(np.pi/180.0)
        self.flt.set_deviation_bound(theta_bound)

    def set_wing_pitch_bound(self,wing_pitch_bound):
        eta_bound = wing_pitch_bound*(np.pi/180.0)
        self.flt.set_wing_pitch_bound(eta_bound)

    def set_model_scale(self,model_scales):
        self.flt.set_model_scale(model_scales)

    def set_model_joint_locs(self,origin,neck_loc,joint_L,joint_R):
        self.flt.set_model_origin(origin[0],origin[1],origin[2])
        self.flt.set_neck_loc(neck_loc[0],neck_loc[1],neck_loc[2])
        self.flt.set_joint_L_loc(joint_L[0],joint_L[1],joint_L[2])
        self.flt.set_joint_R_loc(joint_R[0],joint_R[1],joint_R[2])
        self.flt.set_fixed_body_points()

    def get_stl_list(self):
        return self.flt.get_stl_list()

    def get_start_state(self):
        return self.flt.get_start_state()

    def get_init_state(self):
        return self.flt.get_init_state()

    def get_model_scales(self):
        return self.flt.get_model_scale()

    def segment_frame(self):
        self.flt.segment_frame()

    def set_segmentation_param(self,body_thresh,wing_thresh,Sigma,K,min_body_size,min_wing_size):
        self.flt.set_seg_param(body_thresh,wing_thresh,Sigma,K,min_body_size,min_wing_size)

    def set_mask(self,cam_nr,segment_nr):
        self.flt.set_mask(cam_nr,segment_nr)

    def reset_mask(self):
        self.flt.reset_mask()

    def return_segmented_frame(self):
        frame = []
        for i in range(0,self.N_cam):
            frame.append(self.flt.return_seg_frame(i))
        return frame

    def project_frame_2_pcl(self):
        self.flt.project_frame_2_pcl()

    def set_min_nr_points(self,min_nr_points):
        self.flt.set_min_nr_points(min_nr_points)

    def set_min_conn_points(self,min_conn_points):
        self.flt.set_min_conn_points(min_conn_points)

    def set_sphere_radius(self,sphere_radius):
        self.flt.set_sphere_radius(sphere_radius)

    def set_tethered_flight(self,tethered_check):
        self.flt.set_tethered_flight(tethered_check)

    def find_initial_state(self):
        self.flt.find_initial_state()

    def return_initial_state(self):
        return self.flt.get_initial_state()

    def return_wingtip_boxes(self):
        return self.flt.return_wingtip_boxes()

    def return_wingtip_pcls(self):
        return self.flt.return_wingtip_pcls()

    def return_seg_pcl(self):
        return self.flt.return_seg_pcl()

    def return_fixed_joint_locs(self):
        return self.flt.return_fixed_joints()

    def return_SRF_orientation(self):
        return self.flt.return_SRF_orientation()

    def return_dest_contour(self):
        contour_list = []
        for i in range(self.N_cam):
            contour_list.append(self.flt.return_dest_contour(i))
        return contour_list

    def return_init_contour(self):
        contour_list = []
        for i in range(self.N_cam):
            contour_list.append(self.flt.return_init_contour(i))
        return contour_list

    def return_outer_contour(self):
        contour_list = []
        for i in range(self.N_cam):
            contour_list.append(self.flt.return_outer_contour(i))
        return contour_list
    