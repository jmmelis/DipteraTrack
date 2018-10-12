#include "flight_tracker_class.h"

#include <string>
#include <stdint.h>
#include <stdlib.h>
#include <chrono>
#include <thread>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <dirent.h>
#include <vector>
#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/call.hpp>

using namespace std;

namespace p = boost::python;
namespace np = boost::python::numpy;

// ------------------------------------------------------------------------------------------

FLT::FLT() {
}

// Session functions:

void FLT::SetSessionLocation(string SessionLoc, string SessionName) {
	session_loc = SessionLoc;
	session_name = SessionName;
}

void FLT::SetBackgroundSubtraction(string BckgLoc, p::object& BckgImgList, string BckgImgFmt) {
	bckg_img_names.clear();
	bckg_loc = BckgLoc;
	bckg_img_names = FLT::py_list_to_std_vector_string(BckgImgList);
	bckg_img_format = BckgImgFmt;
}

void FLT::SetCalibrationParameters(string CalLoc, string CalName, string CalImgFmt) {
	cal_loc = CalLoc;
	cal_name = CalName;
	cal_img_format = CalImgFmt;
}

void FLT::SetMovieList(p::object& MovList) {
	mov_names.clear();
	mov_names = FLT::py_list_to_std_vector_string(MovList);
	N_movies = mov_names.size();
}

void FLT::SetCamList(p::object& CamList) {
	cam_names.clear();
	cam_names = FLT::py_list_to_std_vector_string(CamList);
	N_cam = cam_names.size();
}

void FLT::SetFrameParameters(string FrameName, string FrameImgFmt) {
	frame_name = FrameName;
	frame_img_format = FrameImgFmt;
}

void FLT::SetModelParameters(string ModelLoc, string ModelName) {
	model_loc = ModelLoc;
	model_name = ModelName;
}

bool FLT::SetTriggerParameters(string TriggerMode, int StartFrame, int TrigFrame, int EndFrame) {
	bool success = true;
	try {
		vector<int> chrono_frame_nr;
		trigger_mode = TriggerMode;
		start_point = StartFrame;
		trig_point = TrigFrame;
		end_point = EndFrame;
		chrono_frames.clear();
		if (TriggerMode == "start") {
			for (int i = start_point; i<=end_point; i++) {
				chrono_frame_nr.push_back(i);
			}
			chrono_frames = chrono_frame_nr;
		}
		else if (TriggerMode == "center") {
			for (int i = trig_point; i<=end_point; i++) {
				chrono_frame_nr.push_back(i);
			}
			for (int j = start_point; j<trig_point; j++) {
				chrono_frame_nr.push_back(j);
			}
			chrono_frames = chrono_frame_nr;
		}
		else if (TriggerMode == "end") {
			for (int i = start_point; i<=end_point; i++) {
				chrono_frame_nr.push_back(i);
			}
			chrono_frames = chrono_frame_nr;
		}
		else {
			success = false;
		}
	}
	catch (...) {
		success = false;
	}
	return success;
}

bool FLT::SetSessionParameters() {

	bool success = true;

	try {
		// Load session parameters in the session struct:
		session.session_loc 	 = session_loc;
		session.session_name 	 = session_name;
		session.bckg_loc 		 = bckg_loc;
		session.bckg_img_names 	 = bckg_img_names;
		session.bckg_img_format  = bckg_img_format;
		session.cal_loc 		 = cal_loc;
		session.cal_name 		 = cal_name;
		session.cal_img_format 	 = cal_img_format;
		session.mov_names 		 = mov_names;
		session.cam_names 		 = cam_names;
		session.frame_name 		 = frame_name;
		session.frame_img_format = frame_img_format;
		session.model_loc 		 = model_loc;
		session.model_name 		 = model_name;
		session.N_cam 			 = N_cam;
		session.N_movies 		 = N_movies;
		session.trigger_mode 	 = trigger_mode;
		session.start_point 	 = start_point;
		session.trig_point 		 = trig_point;
		session.end_point 		 = end_point;
		session.chrono_frames 	 = chrono_frames;
	}
	catch (...) {
		success = false;
	}

	return success;
}

void FLT::PrintSessionParameters() {

	cout << " " << endl;
	cout << "------------------------------------------------" << endl;
	cout << "FLT session parameters:" << endl;
	cout << "" << endl;
	cout << "" << endl;
	cout << "Session location: " << session.session_loc << endl;
	cout << "Session name: " << session.session_name << endl;
	cout << "Background location: " << session.bckg_loc << endl;
	for (int i=0; i<session.bckg_img_names.size(); i++) {
		cout << "Background image: " << session.bckg_img_names[i] << endl;
	}
	cout << "Background image format: " << session.bckg_img_format << endl;
	cout << "Calibration location: " << session.cal_loc << endl;
	cout << "Calibration file: " << session.cal_name << endl;
	cout << "Calibration image format " << session.cal_img_format << endl;
	cout << "Number of movies: " << session.N_movies << endl;
	for (int i=0; i<session.mov_names.size(); i++) {
		cout << "Movie name: " << session.mov_names[i] << endl;
	}
	cout << "Number of cameras: "  << session.N_cam << endl;
	for (int i=0; i<session.cam_names.size(); i++) {
		cout << "Cameras: " << session.cam_names[i] << endl;
	}
	cout << "Frame name: " << session.frame_name << endl;
	cout << "Frame image format: " << session.frame_img_format << endl;
	cout << "Model location: " << session.model_loc << endl;
	cout << "Model name: " << session.model_name << endl;
	cout << "Trigger mode: " << session.trigger_mode << endl;
	cout << "Start frame: " << session.start_point << endl;
	cout << "Trigger frame: " << session.trig_point << endl;
	cout << "End frame: " << session.end_point << endl;
	cout << "------------------------------------------------" << endl;
	cout << " " << endl;

}

// Frameloader functions:

bool FLT::SetFrameLoader() {

	bool success = true;

	try {
		fl.LoadBackground(session);
		fl.SetFrame(session,frame);
	}
	catch(...) {
		success = false;
	}
	return success;
}

np::ndarray FLT::ReturnImageSize() {
	p::tuple shape = p::make_tuple(session.N_cam,2);
	np::dtype dtype = np::dtype::get_builtin<int>();
	np::ndarray size_mat = np::zeros(shape,dtype);

	for (int i=0; i<session.N_cam; i++) {
		cout << get<0>(frame.image_size[i]) << endl;
		cout << get<1>(frame.image_size[i]) << endl;
		size_mat[i][0] = get<0>(frame.image_size[i]);
		size_mat[i][1] = get<1>(frame.image_size[i]);
	}

	return size_mat;
}

void FLT::LoadFrame(int mov_nr, int frame_nr) {
	fl.LoadFrame(session, frame, mov_nr, frame_nr);
}

np::ndarray FLT::ReturnFrame(int cam_nr) {
	return fl.ReturnFrame(session, frame, cam_nr);
}

// Focal grid functions:

bool FLT::SetFocalGrid(int nx, int ny, int nz, double ds, double x0, double y0, double z0) {

	bool success = true;

	try {
		fg.SetGridParameters(session, frame, nx, ny, nz, ds, x0, y0, z0);
	}
	catch(...) {
		success = false;
	}
	return success;
}

void FLT::SetPixelSizeCamera(double PixelSize) {
	pixel_size = PixelSize;
}

void FLT::PrintFocalGridParameters() {

	cout << " " << endl;
	cout << "------------------------------------------------" << endl;
	cout << "Focal grid parameters:" << endl;
	cout << "" << endl;
	cout << "Nx: " << fg.nx << endl;
	cout << "Ny: " << fg.ny << endl;
	cout << "Nz: " << fg.nz << endl;
	cout << "ds: " << fg.ds << endl;
	cout << "x0: " << fg.x0 << endl;
	cout << "y0: " << fg.y0 << endl;
	cout << "z0: " << fg.z0 << endl;
	cout << "Number of voxels: " << fg.nx*fg.ny*fg.nz << endl;
	cout << "------------------------------------------------" << endl;
	cout << " " << endl;

}

bool FLT::CalculateFocalGrid(PyObject* progress_func) {

	bool success = true;

	try {
		fg.ConstructFocalGrid(progress_func);
	}
	catch(...) {
		success = false;
	}
	return success;
}

np::ndarray FLT::XYZ2UV(double x_in, double y_in, double z_in) {

	arma::Col<double> xyz = {x_in, y_in, z_in, 1.0};

	arma::Mat<int> uv_mat = fg.TransformXYZ2UV(xyz);

	p::tuple shape = p::make_tuple(N_cam,2);
	np::dtype dtype = np::dtype::get_builtin<int>();
	np::ndarray point_mat = np::zeros(shape,dtype);

	for (int i=0; i<N_cam; i++) {
		point_mat[i][0] = uv_mat(0,i);
		point_mat[i][1] = uv_mat(1,i);
	}

	return point_mat;
}

np::ndarray FLT::DragPoint3D(int cam_nr, int u_now, int v_now, int u_prev, int v_prev, double x_prev, double y_prev, double z_prev) {

	tuple<arma::Mat<int>, arma::Col<double>> new_point;

	arma::Col<double> xyz_pos_prev = {x_prev,y_prev,z_prev,1.0};
	arma::Col<double> uv_pos_prev = {double(u_prev),double(v_prev),1.0};
	arma::Col<double> uv_pos_now = {double(u_now),double(v_now),1.0};

	new_point = fg.RayCasting(cam_nr, xyz_pos_prev, uv_pos_prev, uv_pos_now);

	p::tuple shape = p::make_tuple(3,session.N_cam+1);
	np::dtype dtype = np::dtype::get_builtin<double>();
	np::ndarray point_mat = np::zeros(shape,dtype);

	point_mat[0][0] = get<1>(new_point)(0);
	point_mat[1][0] = get<1>(new_point)(1);
	point_mat[2][0] = get<1>(new_point)(2);

	for (int i=0; i<session.N_cam; i++) {
		point_mat[0][i+1] = get<0>(new_point)(0,i);
		point_mat[1][i+1] = get<0>(new_point)(1,i);
		point_mat[2][i+1] = get<0>(new_point)(2,i);
	}

	return point_mat;
}

// ModelClass:

void FLT::LoadModel() {
	mdl.LoadModel(session);
}

np::ndarray FLT::StartState() {
	vector<arma::Mat<double>> M_vec = mdl.ReturnStartState();
	int N_states = M_vec.size();
	p::tuple shape = p::make_tuple(16,N_states);
	np::dtype dtype = np::dtype::get_builtin<double>();
	np::ndarray M_out = np::zeros(shape,dtype);
	for (int i=0; i<N_states; i++) {
		M_out[0][i] = M_vec[i](0,0);
		M_out[1][i] = M_vec[i](0,1);
		M_out[2][i] = M_vec[i](0,2);
		M_out[3][i] = M_vec[i](0,3);
		M_out[4][i] = M_vec[i](1,0);
		M_out[5][i] = M_vec[i](1,1);
		M_out[6][i] = M_vec[i](1,2);
		M_out[7][i] = M_vec[i](1,3);
		M_out[8][i] = M_vec[i](2,0);
		M_out[9][i] = M_vec[i](2,1);
		M_out[10][i] = M_vec[i](2,2);
		M_out[11][i] = M_vec[i](2,3);
		M_out[12][i] = M_vec[i](3,0);
		M_out[13][i] = M_vec[i](3,1);
		M_out[14][i] = M_vec[i](3,2);
		M_out[15][i] = M_vec[i](3,3);
	}
	return M_out;
}

void FLT::SetBodyLength(double BodyLength) {
	body_length = BodyLength;
}

void FLT::SetWingLength(double WingLengthL, double WingLengthR) {
	wing_length_L = WingLengthL;
	wing_length_R = WingLengthR;
}

void FLT::SetStrokeBounds(double StrokeBound) {
	mdl.SetStrokeBounds(StrokeBound);
}

void FLT::SetDeviationBounds(double DeviationBound) {
	mdl.SetDeviationBounds(DeviationBound);
}

void FLT::SetWingPitchBounds(double WingPitchBound) {
	mdl.SetWingPitchBounds(WingPitchBound);
}

void FLT::SetModelScales(p::list scale_list) {
	vector<double> scale_vec = FLT::py_list_to_std_vector_double(scale_list);
	mdl.SetScale(scale_vec);
	mdl.SetModel(session);
}

p::list FLT::GetModelScales() {
	p::list scale_list;
	vector<double> scale_vec = mdl.GetModelScale();
	for (int i=0; i<scale_vec.size(); i++) {
		scale_list.append(scale_vec[i]);
	}
	return scale_list;
}

void FLT::SetOrigin(double x_in, double y_in, double z_in) {
	origin_loc = {x_in, y_in, z_in, 1.0};
}

void FLT::SetNeckLoc(double x_in, double y_in, double z_in) {
	neck_loc = {x_in, y_in, z_in, 1.0};
}

void FLT::SetJointLeftLoc(double x_in, double y_in, double z_in) {
	joint_L_loc = {x_in, y_in, z_in, 1.0};
}

void FLT::SetJointRightLoc(double x_in, double y_in, double z_in) {
	joint_R_loc = {x_in, y_in, z_in, 1.0};
}

p::list FLT::ReturnSTLList() {
	p::list stl_list;
	vector<string> stl_vec = mdl.stl_list;
	for (int i=0; i<stl_vec.size(); i++) {
		stl_list.append(stl_vec[i]);
	}
	return stl_list;
}

p::list FLT::ReturnPointLabels() {
	p::list label_list;
	vector<string> point_labels = mdl.GetPointLabels();
	for (int i=0; i<point_labels.size(); i++) {
		label_list.append(point_labels[i]);
	}
	return label_list;
}

p::list FLT::ReturnPointSymbols() {
	p::list symbol_list;
	vector<string> point_symbols = mdl.GetPointSymbols();
	for (int i=0; i<point_symbols.size(); i++) {
		symbol_list.append(point_symbols[i]);
	}
	return symbol_list;
}

np::ndarray FLT::ReturnPointColors() {
	vector<vector<int>> point_colors = mdl.GetPointColors();
	int N_colors = point_colors.size();
	p::tuple shape = p::make_tuple(N_colors,3);
	np::dtype dtype = np::dtype::get_builtin<int>();
	np::ndarray colors_out = np::zeros(shape,dtype);
	for (int i=0; i<N_colors; i++) {
		for (int j=0; j<3; j++) {
			colors_out[i][j] = point_colors[i][j];
		}
	}
	return colors_out;
}

np::ndarray FLT::ReturnPointStartPos() {
	vector<vector<double>> point_pos = mdl.GetPointStartPos();
	int N_pos = point_pos.size();
	p::tuple shape = p::make_tuple(N_pos,3);
	np::dtype dtype = np::dtype::get_builtin<double>();
	np::ndarray pos_out = np::zeros(shape,dtype);
	for (int i=0; i<N_pos; i++) {
		for (int j=0; j<3; j++) {
			pos_out[i][j] = point_pos[i][j];
		}
	}
	return pos_out;
}

np::ndarray FLT::ReturnLineConnectivity() {
	vector<vector<int>> line_con = mdl.GetLineConnectivity();
	int N_lines = line_con.size();
	p::tuple shape = p::make_tuple(N_lines,2);
	np::dtype dtype = np::dtype::get_builtin<int>();
	np::ndarray con_out = np::zeros(shape,dtype);
	for (int i=0; i<N_lines; i++) {
		for (int j=0; j<2; j++) {
			con_out[i][j] = line_con[i][j];
		}
	}
	return con_out;
}

np::ndarray FLT::ReturnLineColors() {
	vector<vector<int>> line_colors = mdl.GetLineColors();
	int N_colors = line_colors.size();
	p::tuple shape = p::make_tuple(N_colors,5);
	np::dtype dtype = np::dtype::get_builtin<int>();
	np::ndarray colors_out = np::zeros(shape,dtype);
	for (int i=0; i<N_colors; i++) {
		for (int j=0; j<5; j++) {
			colors_out[i][j] = line_colors[i][j];
		}
	}
	return colors_out;
}

p::list FLT::ReturnScaleTexts() {
	p::list scale_text_list;
	vector<string> scale_texts = mdl.GetScaleTexts();
	for (int i=0; i<scale_texts.size(); i++) {
		scale_text_list.append(scale_texts[i]);
	}
	return scale_text_list;
}

np::ndarray FLT::ReturnScaleCalc() {
	vector<vector<int>> scale_calc = mdl.GetScaleCalc();
	int N_calcs = scale_calc.size();
	p::tuple shape = p::make_tuple(N_calcs,2);
	np::dtype dtype = np::dtype::get_builtin<int>();
	np::ndarray calc_out = np::zeros(shape,dtype);
	for (int i=0; i<N_calcs; i++) {
		for (int j=0; j<2; j++) {
			calc_out[i][j] = scale_calc[i][j];
		}
	}
	return calc_out;
}

np::ndarray FLT::ReturnLengthCalc() {
	vector<vector<int>> length_calc = mdl.GetLengthCalc();
	int N_calcs = length_calc.size();
	p::tuple shape = p::make_tuple(N_calcs,2);
	np::dtype dtype = np::dtype::get_builtin<int>();
	np::ndarray calc_out = np::zeros(shape,dtype);
	for (int i=0; i<N_calcs; i++) {
		for (int j=0; j<2; j++) {
			calc_out[i][j] = length_calc[i][j];
		}
	}
	return calc_out;
}

np::ndarray FLT::ReturnContourCalc() {
	vector<int> contour_calc = mdl.GetContourCalc();
	int N_calcs = contour_calc.size();
	p::tuple shape = p::make_tuple(N_calcs,1);
	np::dtype dtype = np::dtype::get_builtin<int>();
	np::ndarray calc_out = np::zeros(shape,dtype);
	for (int i=0; i<N_calcs; i++) {
		calc_out[i] = contour_calc[i];
	}
	return calc_out;
}

p::list FLT::ReturnOriginInd() {
	p::list origin_ind_list;
	vector<int> origin_ind = mdl.GetOriginInd();
	for (int i=0; i<origin_ind.size(); i++) {
		origin_ind_list.append(origin_ind[i]);
	}
	return origin_ind_list;
}

// Image segmentation functions:

bool FLT::SegmentFrame() {
	bool success = true;
	try {
		seg.SegmentFrame(frame);
	}
	catch (...) {
		success = false;
	}
	return success;
}

void FLT::SetImagSegmParam(int body_thresh, int wing_thresh, double Sigma, int K, int min_body_size, int min_wing_size) {

	arma::Col<double> xyz_pos = {origin_loc(0), origin_loc(1), origin_loc(2), 1.0};
	arma::Mat<int> uv_mat = fg.TransformXYZ2UV(xyz_pos);

	vector<tuple<double,double>> Origin;

	for (int n=0; n<session.N_cam; n++) {
		Origin.push_back(make_tuple(uv_mat(0,n),uv_mat(1,n)));
	}

	double BodyLength = body_length/pixel_size;
	double WingLength = (wing_length_L+wing_length_R)/(2.0*pixel_size);

	seg.SetBodySegmentationParam(body_thresh, 2, 5, Sigma, K, min_body_size, BodyLength, Origin);
	seg.SetWingSegmentationParam(wing_thresh, 2, Sigma, K, min_wing_size, WingLength);

}

void FLT::SegImageMask(int cam_nr, int seg_nr) {
	seg.SetImageMask(frame, cam_nr, seg_nr);
}

void FLT::ResetImageMask() {
	seg.ResetImageMask(frame);
}

np::ndarray FLT::ReturnSegmentedFrame(int cam_nr) {
	return seg.ReturnSegFrame(frame, cam_nr);
}

// InitState

void FLT::SetMinimumNumberOfPoints(int NrMinPoints) {
	init.SetMinimumNumberOfPoints(NrMinPoints);
}

void FLT::SetMinimumConnectionPoints(int MinConnPoints) {
	init.SetMinimumConnectionPoints(MinConnPoints);
}

void FLT::SetFixedBodyPoints() {
	vector<arma::Col<double>> fixed_locs;
	fixed_locs.push_back(origin_loc);
	fixed_locs.push_back(neck_loc);
	fixed_locs.push_back(joint_L_loc);
	fixed_locs.push_back(joint_R_loc);
	init.SetFixedBodyLocs(fixed_locs,wing_length_L,wing_length_R);
	init.SetMstrk(mdl);
	init.SetBounds(mdl);
}

bool FLT::ProjectFrame2PCL() {
	bool success = true;
	try {
		init.ProjectFrame2PCL(fg, frame);
	}
	catch (...) {
		success = false;
	}
	return success;
}

void FLT::FindInitialState() {
	init.FindInitialState(frame, fg, mdl);
	opt.OptimizeState(fg,frame,mdl, seg);
}

np::ndarray FLT::GetInitState() {
	vector<arma::Mat<double>> M_vec = mdl.ReturnInitState(frame);
	int N_states = M_vec.size();
	p::tuple shape = p::make_tuple(16,N_states);
	np::dtype dtype = np::dtype::get_builtin<double>();
	np::ndarray M_out = np::zeros(shape,dtype);
	for (int i=0; i<N_states; i++) {
		M_out[0][i] = M_vec[i](0,0);
		M_out[1][i] = M_vec[i](0,1);
		M_out[2][i] = M_vec[i](0,2);
		M_out[3][i] = M_vec[i](0,3);
		M_out[4][i] = M_vec[i](1,0);
		M_out[5][i] = M_vec[i](1,1);
		M_out[6][i] = M_vec[i](1,2);
		M_out[7][i] = M_vec[i](1,3);
		M_out[8][i] = M_vec[i](2,0);
		M_out[9][i] = M_vec[i](2,1);
		M_out[10][i] = M_vec[i](2,2);
		M_out[11][i] = M_vec[i](2,3);
		M_out[12][i] = M_vec[i](3,0);
		M_out[13][i] = M_vec[i](3,1);
		M_out[14][i] = M_vec[i](3,2);
		M_out[15][i] = M_vec[i](3,3);
	}
	return M_out;
}

void FLT::SetSphereRadius(double SphereRadius) {
	init.SetSphereRadius(SphereRadius);
}

void FLT::SetTetheredFlight(bool TetheredCheck) {
	tethered_flight = TetheredCheck;
}

np::ndarray FLT::ReturnWingtipPCLS() {
	return init.ReturnWingtipPCLS();
}

np::ndarray FLT::ReturnWingtipBoxes() {
	return init.ReturnWingtipBoxes();
}

np::ndarray FLT::ReturnSegPCL() {
	return init.ReturnSegPCL(frame);
}

np::ndarray FLT::ReturnFixedJointLocs() {
	return init.ReturnJointLocs();
}

np::ndarray FLT::ReturnSRFOrientation() {
	return init.ReturnSRF();
}

// ContourOpt:

void FLT::OptimizeState() {
	opt.OptimizeState(fg,frame,mdl, seg);
}

np::ndarray FLT::ReturnOuterContour(int cam_nr) {
	return seg.ReturnOuterContour(frame, cam_nr);
}

p::list FLT::ReturnDestContour(int cam_nr) {
	return opt.ReturnDestContour(cam_nr);
}

p::list FLT::ReturnInitContour(int cam_nr) {
	return opt.ReturnInitContour(cam_nr);
}

// Helper functions:

vector<double> FLT::py_list_to_std_vector_double( p::object& iterable ) {
    return vector<double>( p::stl_input_iterator< double >(iterable), p::stl_input_iterator< double >( ));
}

vector<string> FLT::py_list_to_std_vector_string( p::object& iterable ) {
    return vector<string>( p::stl_input_iterator< string >(iterable), p::stl_input_iterator< string >( ));
}