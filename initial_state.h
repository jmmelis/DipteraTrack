#ifndef INITIAL_STATE_H
#define INITIAL_STATE_H

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <string>
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <thread>
#include <chrono>
#include <algorithm>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <armadillo>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Alpha_shape_3.h>
#include <CGAL/Alpha_shape_cell_base_3.h>
#include <CGAL/Alpha_shape_vertex_base_3.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <cassert>
#include <list>

#include "session_data.h"
#include "frame_data.h"
#include "focal_grid.h"
#include "model_class.h"

using namespace std;

namespace p = boost::python;
namespace np = boost::python::numpy;

typedef CGAL::Exact_predicates_inexact_constructions_kernel Gt;
typedef CGAL::Alpha_shape_vertex_base_3<Gt>          Vb;
typedef CGAL::Alpha_shape_cell_base_3<Gt>            Fb;
typedef CGAL::Triangulation_data_structure_3<Vb,Fb>  Tds;
typedef CGAL::Delaunay_triangulation_3<Gt,Tds>       Triangulation_3;
typedef CGAL::Alpha_shape_3<Triangulation_3>         Alpha_shape_3;
typedef Gt::Point_3                                  Point;
typedef Alpha_shape_3::Alpha_iterator                Alpha_iterator;

struct bbox {
	double box_center [3];
	double box_corners [3][8];
	double q_box [4];
	double R_box [3][3];
	double mass_center [3];
	double eigen_values [3];
	double eigen_vectors [3][3];
	double moment_of_inertia [3];
	double eccentricity [3];
	double volume;
};

class InitState
{

	public:

		// Class
		InitState();

		// Parameters
		double sphere_radius;
		bool tethered;
		struct bbox bbox_body;
		arma::Mat<double> body_pcl;
		struct bbox bbox_wingtip_L;
		struct bbox bbox_wingtip_R;
		arma::Mat<double> wingtip_pcl_L;
		arma::Mat<double> wingtip_pcl_R;
		double wing_length_L;
		double wing_length_R;
		int min_nr_points;
		int min_connect_points;
		arma::Col<double> origin_loc;
		arma::Col<double> neck_loc;
		arma::Col<double> joint_L_loc;
		arma::Col<double> joint_R_loc;
		arma::Col<double> wingtip_L_loc;
		arma::Col<double> wingtip_R_loc;
		arma::Mat<double> M_body_fixed;
		arma::Mat<double> M_body;
		arma::Mat<double> M_strk;
		arma::Mat<double> M_90;
		arma::Mat<double> M_wing_L;
		arma::Mat<double> M_wing_R;
		double phi_bound;
		double theta_bound;
		double eta_bound;

		// Functions
		void SetWingLength(double WingLength_L, double WingLength_R);
		void SetTethered(bool TetheredCheck);
		void SetBounds(ModelClass &mdl);
		void SetMstrk(ModelClass &mdl);
		void SetMinimumNumberOfPoints(int NrMinPoints);
		void SetMinimumConnectionPoints(int MinConnPoints);
		void SetFixedBodyLocs(vector<arma::Col<double>> FixedLocs, double WingLengthL, double WingLengthR);
		void ProjectFrame2PCL(FocalGrid &fg, frame_data &frame_in);
		void SetSphereRadius(double SphereRadius);
		void FindInitialState(frame_data &frame_in, FocalGrid &fg, ModelClass &mdl);
		void SetInitState(ModelClass &mdl, frame_data &frame_in);
		bool FindBodyBoundingBox(frame_data &frame_in);
		bool FindWingBoundingBoxes(frame_data &frame_in, FocalGrid &fg);
		void CalculateFixedBodyRefFrame();
		vector<arma::Mat<double>> ConnectPointclouds(FocalGrid &fg, vector<arma::Mat<double>> &pcl_vec);
		tuple<vector<arma::Mat<double>>,vector<arma::Col<double>>> SelectWingTipsTethered(vector<arma::Mat<double>> &pcl_vec);
		vector<arma::Mat<double>> SelectWingTips(vector<arma::Mat<double>> &pcl_vec);
		arma::Mat<double> OrthoNormalizeR(arma::Mat<double> R_in);
		void Alpha3D(arma::Mat<double> &pcl_in);
		bbox BoundingBox(arma::Mat<double> &pcl_in);
		np::ndarray ReturnBBox(vector<struct bbox> &bbox_in);
		np::ndarray ReturnJointLocs();
		np::ndarray ReturnSRF();
		np::ndarray ReturnWingtipPCLS();
		np::ndarray ReturnWingtipBoxes();
		np::ndarray ReturnSegPCL(frame_data &frame_in);

};
#endif