#ifndef FLIGHT_TRACKER_CLASS_H
#define FLIGHT_TRACKER_CLASS_H

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/call.hpp>
#include <iostream>
#include <string>
#include <stdint.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <armadillo>

#include "session_data.h"
#include "frame_loader.h"
#include "frame_data.h"
#include "focal_grid.h"
#include "model_class.h"
#include "image_segmentation.h"
#include "initial_state.h"
#include "contour_optimization.h"

namespace p = boost::python;
namespace np = boost::python::numpy;

using namespace std;

class FLT
{
	
	public:

		// Data structures:
		struct session_data session;
		struct frame_data frame;

		// Classes:
		FrameLoader fl;
		FocalGrid fg;
		ModelClass mdl;
		ImagSegm seg;
		InitState init;
		ContourOpt opt;

		// Session parameters:
		string session_loc;
		string session_name;
		string bckg_loc;
		vector<string> bckg_img_names;
		string bckg_img_format;
		string cal_loc;
		string cal_name;
		string cal_img_format;
		vector<string> mov_names;
		vector<string> cam_names;
		string frame_name;
		string frame_img_format;
		string model_loc;
		string model_name;

		int N_cam;
		int N_movies;

		string trigger_mode;
		int start_point;
		int trig_point;
		int end_point;
		vector<int> chrono_frames;
		double pixel_size;

		// Segmentation parameters:
		bool tethered_flight;
		double body_length;
		double wing_length_L;
		double wing_length_R;
		arma::Col<double> origin_loc;
		arma::Col<double> neck_loc;
		arma::Col<double> joint_L_loc;
		arma::Col<double> joint_R_loc;

		FLT();

		// Session
		void SetSessionLocation(string SessionLoc, string SessionName);
		void SetBackgroundSubtraction(string BckgLoc, p::object& BckgImgList, string BckgImgFmt);
		void SetCalibrationParameters(string CalLoc, string CalName, string CalImgFmt);
		void SetMovieList(p::object& MovList);
		void SetCamList(p::object& CamList);
		void SetFrameParameters(string FrameName, string FrameImgFmt);
		void SetModelParameters(string ModelLoc, string ModelName);
		bool SetTriggerParameters(string TriggerMode, int StartFrame, int TrigFrame, int EndFrame);
		bool SetSessionParameters();
		void PrintSessionParameters();

		// FrameLoader
		bool SetFrameLoader();
		np::ndarray ReturnImageSize();
		void LoadFrame(int mov_nr, int frame_nr);
		np::ndarray ReturnFrame(int cam_nr);

		// FocalGrid
		bool SetFocalGrid(int nx, int ny, int nz, double ds, double x0, double y0, double z0);
		void SetPixelSizeCamera(double PixelSize);
		void PrintFocalGridParameters();
		bool CalculateFocalGrid(PyObject* progress_func);
		np::ndarray XYZ2UV(double x_in, double y_in, double z_in);
		np::ndarray DragPoint3D(int cam_nr, int u_now, int v_now, int u_prev, int v_prev, double x_prev, double y_prev, double z_prev);

		// ModelClass
		void LoadModel();
		np::ndarray StartState();
		void SetBodyLength(double BodyLength);
		void SetWingLength(double WingLengthL, double WingLengthR);
		void SetStrokeBounds(double StrokeBound);
		void SetDeviationBounds(double DeviationBound);
		void SetWingPitchBounds(double WingPitchBound);
		void SetModelScales(p::list scale_list);
		p::list GetModelScales();
		void SetOrigin(double x_in, double y_in, double z_in);
		void SetNeckLoc(double x_in, double y_in, double z_in);
		void SetJointLeftLoc(double x_in, double y_in, double z_in);
		void SetJointRightLoc(double x_in, double y_in, double z_in);
		p::list ReturnSTLList();
		p::list ReturnPointLabels();
		p::list ReturnPointSymbols();
		np::ndarray ReturnPointColors();
		np::ndarray ReturnPointStartPos();
		np::ndarray ReturnLineConnectivity();
		np::ndarray ReturnLineColors();
		p::list ReturnScaleTexts();
		np::ndarray ReturnScaleCalc();
		np::ndarray ReturnLengthCalc();
		np::ndarray ReturnContourCalc();
		p::list ReturnOriginInd();

		// ImagSegm
		bool SegmentFrame();
		void SetImagSegmParam(int body_thresh, int wing_thresh, double Sigma, int K, int min_body_size, int min_wing_size);
		void SegImageMask(int cam_nr, int seg_nr);
		void ResetImageMask();
		np::ndarray ReturnSegmentedFrame(int cam_nr);

		// InitState
		void SetMinimumNumberOfPoints(int NrMinPoints);
		void SetMinimumConnectionPoints(int MinConnPoints);
		void SetFixedBodyPoints();
		bool ProjectFrame2PCL();
		void FindInitialState();
		void SetSphereRadius(double SphereRadius);
		void SetTetheredFlight(bool TetheredCheck);
		np::ndarray GetInitState();
		np::ndarray ReturnWingtipPCLS();
		np::ndarray ReturnWingtipBoxes();
		np::ndarray ReturnSegPCL();
		np::ndarray ReturnFixedJointLocs();
		np::ndarray ReturnSRFOrientation();

		// ContourOpt
		void SetAlpha(double AlphaIn);
		void OptimizeState();
		np::ndarray ReturnBodyDestContour(int cam_nr);
		np::ndarray ReturnWingLDestContour(int cam_nr);
		np::ndarray ReturnWingRDestContour(int cam_nr);
		np::ndarray ReturnOuterContour(int cam_nr);
		np::ndarray ReturnBodyInitContour(int cam_nr);
		np::ndarray ReturnWingLInitContour(int cam_nr);
		np::ndarray ReturnWingRInitContour(int cam_nr);
		p::list ReturnDestContour(int cam_nr);
		p::list ReturnInitContour(int cam_nr);

		// Helper functions
		vector<double> py_list_to_std_vector_double( p::object& iterable );
		vector<string> py_list_to_std_vector_string( p::object& iterable );

};
#endif