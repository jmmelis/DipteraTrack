#include "initial_state.h"

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

#include "frame_data.h"
#include "focal_grid.h"
#include "model_class.h"

#define PI 3.14159265

using namespace std;

namespace p = boost::python;
namespace np = boost::python::numpy;

typedef CGAL::Exact_predicates_inexact_constructions_kernel Gt;
typedef CGAL::Alpha_shape_vertex_base_3<Gt>          		Vb;
typedef CGAL::Alpha_shape_cell_base_3<Gt>            		Fb;
typedef CGAL::Triangulation_data_structure_3<Vb,Fb>  		Tds;
typedef CGAL::Delaunay_triangulation_3<Gt,Tds>       		Triangulation_3;
typedef CGAL::Alpha_shape_3<Triangulation_3>         		Alpha_shape_3;
typedef Gt::Point_3                                  		Point;
typedef Alpha_shape_3::Alpha_iterator                		Alpha_iterator;

InitState::InitState() {
	// empty
}

void InitState::SetTethered(bool TetheredCheck) {
	tethered = TetheredCheck;
}

void InitState::SetMstrk(ModelClass &mdl) {
	double beta = mdl.srf_angle;

	M_strk.eye(4,4);
	M_strk(0,0) = cos(beta);
	M_strk(0,2) = sin(beta);
	M_strk(2,0) = -sin(beta);
	M_strk(2,2) = cos(beta);

	M_90.eye(4,4);
	M_90(0,0) = 0.0;
	M_90(0,2) = -1.0;
	M_90(2,0) = 1.0;
	M_90(2,2) = 0.0;

}

void InitState::SetBounds(ModelClass &mdl) {
	phi_bound = mdl.stroke_bound;
	theta_bound = mdl.deviation_bound;
	eta_bound = mdl.wing_pitch_bound;
}

void InitState::SetMinimumNumberOfPoints(int NrMinPoints) {
	min_nr_points = NrMinPoints;
}

void InitState::SetMinimumConnectionPoints(int MinConnPoints) {
	min_connect_points = MinConnPoints;
}

void InitState::SetFixedBodyLocs(vector<arma::Col<double>> FixedLocs, double WingLengthL, double WingLengthR) {
	origin_loc.set_size(3);
	neck_loc.set_size(3);
	joint_L_loc.set_size(3);
	joint_R_loc.set_size(3);
	wingtip_L_loc.set_size(3);
	wingtip_R_loc.set_size(3);
	origin_loc(0) = FixedLocs[0](0);
	origin_loc(1) = FixedLocs[0](1);
	origin_loc(2) = FixedLocs[0](2);
	neck_loc(0) = FixedLocs[1](0);
	neck_loc(1) = FixedLocs[1](1);
	neck_loc(2) = FixedLocs[1](2);
	joint_L_loc(0) = FixedLocs[2](0);
	joint_L_loc(1) = FixedLocs[2](1);
	joint_L_loc(2) = FixedLocs[2](2);
	joint_R_loc(0) = FixedLocs[3](0);
	joint_R_loc(1) = FixedLocs[3](1);
	joint_R_loc(2) = FixedLocs[3](2);
	InitState::CalculateFixedBodyRefFrame();
	wing_length_L = WingLengthL;
	wing_length_R = WingLengthR;
}

void InitState::ProjectFrame2PCL(FocalGrid &fg, frame_data &frame_in) {
	frame_in.seg_pcl.clear();
	vector<tuple<double,double,double,int>> pcl_out = fg.ProjectFrames2Cloud(frame_in.seg_frame);
	frame_in.seg_pcl = fg.ConvertVector2Mat(pcl_out);
}

void InitState::SetSphereRadius(double SphereRadius) {
	sphere_radius = SphereRadius*wing_length_L;
}

void InitState::FindInitialState(frame_data &frame_in, FocalGrid &fg, ModelClass &mdl) {
	bool body_bbox_found = InitState::FindBodyBoundingBox(frame_in);
	bool wingtip_bbox_found = InitState::FindWingBoundingBoxes(frame_in, fg);
	if (body_bbox_found && wingtip_bbox_found) {
		InitState::SetInitState(mdl,frame_in);
	}
}

void InitState::SetInitState(ModelClass &mdl, frame_data &frame_in) {
	frame_in.init_state.reset();
	frame_in.init_state.zeros(7,3);
	frame_in.body_and_wing_pcls.reset();
	frame_in.body_and_wing_pcls.zeros(4,1);
	if (arma::trace(M_body_fixed)!=0.0) {
		frame_in.init_state.col(0) = mdl.GetStateFromM(M_body);
		frame_in.body_and_wing_pcls.insert_cols(frame_in.body_and_wing_pcls.n_cols,body_pcl);
	}
	if (arma::trace(M_wing_L)!=0.0) {
		frame_in.init_state.col(1) = mdl.GetStateFromM(M_wing_L);
		frame_in.body_and_wing_pcls.insert_cols(frame_in.body_and_wing_pcls.n_cols,wingtip_pcl_L);
	}
	if (arma::trace(M_wing_R)!=0.0) {
		frame_in.init_state.col(2) = mdl.GetStateFromM(M_wing_R);
		frame_in.body_and_wing_pcls.insert_cols(frame_in.body_and_wing_pcls.n_cols,wingtip_pcl_R);
	}
}

bool InitState::FindBodyBoundingBox(frame_data &frame_in) {
	bool success = true;
	body_pcl.reset();
	M_body.reset();
	try {
		arma::Row<double> seg_ids = arma::unique(frame_in.seg_pcl.row(3));
		int N_seg = seg_ids.n_cols;
		double body_id = seg_ids.min();
		arma::uvec body_pt_ids = arma::find(frame_in.seg_pcl.row(3) == body_id);
		body_pcl = frame_in.seg_pcl.cols(body_pt_ids);
		body_pcl.row(3).fill(1.0);
		bbox_body = InitState::BoundingBox(body_pcl);
		M_body.eye(4,4);
		M_body.submat(0,0,2,2) = M_body_fixed.submat(0,0,2,2);
		M_body(0,3) = M_body_fixed(0,3);
		M_body(1,3) = M_body_fixed(1,3);
		M_body(2,3) = M_body_fixed(2,3);
		//InitState::Alpha3D(body_pcl);
	}
	catch (...) {
		success = false;
		M_body.eye(4,4);
		body_pcl.zeros(4,1);
	}
	return success;
}

bool InitState::FindWingBoundingBoxes(frame_data &frame_in, FocalGrid &fg) {
	bool success = true;
	wingtip_pcl_L.reset();
	wingtip_pcl_R.reset();
	M_wing_L.reset();
	M_wing_R.reset();
	try {
		if (tethered = true) {

			vector<arma::Mat<double>> pcl_vec;

			arma::Mat<double> pcl_mat = frame_in.seg_pcl;
			int N_pts = pcl_mat.n_cols;

			arma::Row<double> seg_ids = arma::unique(pcl_mat.row(3));
			int N_seg = seg_ids.n_cols;

			// Remove body points
			double body_id = seg_ids.min();
			arma::uvec wing_pt_ids1 = arma::find(pcl_mat.row(3) > body_id);
			arma::Mat<double> wing_tip_pcl1 = pcl_mat.cols(wing_pt_ids1);

			// Remove points within the cylinder around the body's longitudinal axis:
			arma::Col<double> body_ref_center = {M_body_fixed(0,3), M_body_fixed(1,3), M_body_fixed(2,3)};

			arma::Mat<double> xyz_body_ref = M_body_fixed.submat(0,0,2,2).t()*(wing_tip_pcl1.rows(0,2)-arma::repmat(body_ref_center,1,wing_tip_pcl1.n_cols));
			arma::Row<double> radius_yz = (xyz_body_ref.row(1)%xyz_body_ref.row(1))+(xyz_body_ref.row(2)%xyz_body_ref.row(2));
			arma::uvec wing_pt_ids2 = arma::find(radius_yz > pow(sphere_radius,2));
			//arma::Mat<double> wing_tip_pcl2 = wing_tip_pcl1.cols(wing_pt_ids2);

			arma::Mat<double> wing_tip_pcl2 = wing_tip_pcl1;

			arma::Row<double> wing_seg_ids = arma::unique(wing_tip_pcl2.row(3));
			int N_wing_segs = wing_seg_ids.n_cols;

			for (int j=0; j<N_wing_segs; j++) {
				arma::uvec wing_pt_ids3 = arma::find(wing_tip_pcl2.row(3) == wing_seg_ids(j));
				if (wing_pt_ids3.n_rows>min_nr_points) {
					arma::Mat<double> wing_tip_pcl3 = wing_tip_pcl2.cols(wing_pt_ids3);
					pcl_vec.push_back(wing_tip_pcl3);
				}
			}

			// Find connected matrices:
			vector<arma::Mat<double>> pcl_vec2 = InitState::ConnectPointclouds(fg, pcl_vec);
			tuple<vector<arma::Mat<double>>,vector<arma::Col<double>>> wing_tip_LR = InitState::SelectWingTipsTethered(pcl_vec2);
			vector<arma::Mat<double>> wingtip_pcl_LR = get<0>(wing_tip_LR);
			vector<arma::Col<double>> wingtip_info_LR = get<1>(wing_tip_LR);

			if (wingtip_info_LR[0](7)>1.0) {
				bbox_wingtip_L = InitState::BoundingBox(wingtip_pcl_LR[0]);
				wingtip_pcl_L  = wingtip_pcl_LR[0];
				double phi_L = wingtip_info_LR[0](4);
				double theta_L = wingtip_info_LR[0](5);
				double eta_L = wingtip_info_LR[0](6);
				arma::Mat<double> R_phi_L = {{1.0, 0.0, 0.0},
					{0.0, cos(phi_L), -sin(phi_L)},
					{0.0, sin(phi_L), cos(phi_L)}};
				arma::Mat<double> R_theta_L = {{cos(-theta_L), -sin(-theta_L), 0.0},
					{sin(-theta_L), cos(-theta_L), 0.0},
					{0.0, 0.0, 1.0}};
				arma::Mat<double> R_eta_L = {{cos(eta_L), 0.0, sin(eta_L)},
					{0.0, 1.0, 0.0},
					{-sin(eta_L), 0.0, sin(eta_L)}};
				arma::Mat<double> R_wing_L = M_strk.submat(0,0,2,2)*M_90.submat(0,0,2,2)*R_phi_L*R_theta_L;
				M_wing_L.eye(4,4);
				M_wing_L.submat(0,0,2,2) = R_wing_L;
			}
			else {
				wingtip_pcl_L.zeros(4,1);
				M_wing_L.zeros(4,4);
			}

			if (wingtip_info_LR[1](7)>1.0) {
				bbox_wingtip_R = InitState::BoundingBox(wingtip_pcl_LR[1]);
				wingtip_pcl_R  = wingtip_pcl_LR[1];
				double phi_R = wingtip_info_LR[1](4);
				double theta_R = wingtip_info_LR[1](5);
				double eta_R = wingtip_info_LR[1](6);
				arma::Mat<double> R_phi_R = {{1.0, 0.0, 0.0},
					{0.0, cos(-phi_R), -sin(-phi_R)},
					{0.0, sin(-phi_R), cos(-phi_R)}};
				arma::Mat<double> R_theta_R = {{cos(theta_R), -sin(theta_R), 0.0},
					{sin(theta_R), cos(theta_R), 0.0},
					{0.0, 0.0, 1.0}};
				arma::Mat<double> R_eta_R = {{cos(eta_R), 0.0, sin(eta_R)},
					{0.0, 1.0, 0.0},
					{-sin(eta_R), 0.0, sin(eta_R)}};
				arma::Mat<double> R_wing_R = M_strk.submat(0,0,2,2)*M_90.submat(0,0,2,2)*R_phi_R*R_theta_R;
				M_wing_R.eye(4,4);
				M_wing_R.submat(0,0,2,2) = R_wing_R;
			}
			else {
				wingtip_pcl_R.zeros(4,1);
				M_wing_R.zeros(4,4);
			} 

		}
		else {
			// Free flight algorithm
		}
	}
	catch (...) {
		success = false;
		wingtip_pcl_L.zeros(4,1);
		wingtip_pcl_R.zeros(4,1);
		M_wing_L.eye(4,4);
		M_wing_R.eye(4,4);
	}
	return success;
}

void InitState::CalculateFixedBodyRefFrame() {

	arma::Col<double> x_vector;
	arma::Col<double> y_vector;
	arma::Col<double> z_vector;

	x_vector = (neck_loc-origin_loc)/arma::norm(neck_loc-origin_loc);
	x_vector = x_vector/arma::norm(x_vector);
	y_vector = (joint_L_loc-joint_R_loc)/arma::norm(joint_L_loc-joint_R_loc);
	y_vector = y_vector/arma::norm(y_vector);
	z_vector = arma::cross(x_vector,y_vector)/(arma::norm(x_vector)*arma::norm(y_vector));
	z_vector = z_vector/arma::norm(z_vector);

	// Construct the rotation matrix from the reference frame:
	arma::Mat<double> M_body(3,3);
	M_body.col(0) = x_vector;
	M_body.col(1) = y_vector;
	M_body.col(2) = z_vector;

	// Find the mid-point between the two hinges:
	arma::Col<double> center_loc;
	center_loc = (joint_L_loc+joint_R_loc)/2.0;

	M_body_fixed.eye(4,4);

	M_body_fixed.submat(0,0,2,2) = M_body;
	M_body_fixed(0,3) = center_loc(0);
	M_body_fixed(1,3) = center_loc(1);
	M_body_fixed(2,3) = center_loc(2);
}

vector<arma::Mat<double>> InitState::ConnectPointclouds(FocalGrid &fg, vector<arma::Mat<double>> &pcl_vec) {

	int N_pcls = pcl_vec.size();

	arma::Mat<double> pcl_mat;

	vector<arma::Mat<double>> pcl_out;

	if (N_pcls>1) {
		arma::Mat<double> pcl_mat;
		pcl_mat = pcl_vec[0];
		for (int i=1; i<N_pcls; i++) {
			pcl_mat = arma::join_rows(pcl_mat,pcl_vec[i]);
		}
		arma::Mat<int> connectivity_mat = fg.FindConnectedPointClouds(pcl_mat);

		arma::Col<int> pcl_codes = connectivity_mat.diag();

		vector<vector<int>> conn_vec;
		vector<int> conn_row;
		vector<int>::iterator it;
		bool is_new = true;
		int vec_ind;

		for (int i=0; i<N_pcls; i++) {

			if (i==0) {
				conn_row.clear();
				conn_row.push_back(i);
				conn_vec.push_back(conn_row);
				vec_ind = 0;
			}
			else {
				is_new = true;
				for (int k=0; k<conn_vec.size(); k++) {
					it = find(conn_vec[k].begin(),conn_vec[k].end(),i);
					if (it != conn_vec[k].end()) {
						is_new = false;
						vec_ind = k;
					}
				}
				if (is_new==true) {
					conn_row.clear();
					conn_row.push_back(i);
					conn_vec.push_back(conn_row);
					vec_ind = conn_vec.size()-1;
				}
			}

			for (int j=0; j<N_pcls; j++) {
				if (i != j) {
					if (connectivity_mat(i,j) > min_connect_points) {
						it = find(conn_vec[vec_ind].begin(),conn_vec[vec_ind].end(),j);
						if (it == conn_vec[vec_ind].end()) {
							conn_vec[vec_ind].push_back(j);
						}
					}
				}
			}
		}

		int N_conn_pcls = conn_vec.size();
		arma::Mat<double> temp_pcl;

		for (int m=0; m<N_conn_pcls; m++) {
			for (int n=0; n<conn_vec[m].size(); n++) {
			}
		}

		for (int m=0; m<N_conn_pcls; m++) {
			temp_pcl = pcl_vec[conn_vec[m][0]];
			if (conn_vec[m].size() > 1) {
				for (int n=1; n<conn_vec[m].size(); n++) {
					temp_pcl = arma::join_rows(temp_pcl,pcl_vec[conn_vec[m][n]]);
				}
			}
			temp_pcl.row(3).fill(m+2);
			pcl_out.push_back(temp_pcl);
		}
	}
	else if (N_pcls == 1) {
		pcl_out.push_back(pcl_vec[0]);
	}
	else {
		// Do nothing
	}

	return pcl_out;
}


tuple<vector<arma::Mat<double>>,vector<arma::Col<double>>> InitState::SelectWingTipsTethered(vector<arma::Mat<double>> &pcl_vec) {

	int N_segs = pcl_vec.size();

	vector<arma::Mat<double>> pcl_wingtip_L;
	vector<arma::Col<double>> potential_wingtip_L;
	vector<arma::Col<double>> wingtip_info_L;
	vector<arma::Mat<double>> pcl_wingtip_R;
	vector<arma::Col<double>> potential_wingtip_R;
	vector<arma::Col<double>> wingtip_info_R;

	arma::Col<double> wingtip_info(8);

	for (int i=0; i<N_segs; i++) {
		arma::Row<double> joint_L_dist = arma::sum((pcl_vec[i].rows(0,2).each_col()-joint_L_loc)%(pcl_vec[i].rows(0,2).each_col()-joint_L_loc));
		arma::Row<double> joint_R_dist = arma::sum((pcl_vec[i].rows(0,2).each_col()-joint_R_loc)%(pcl_vec[i].rows(0,2).each_col()-joint_R_loc));
		arma::uvec joint_L_dist_sort = arma::sort_index(joint_L_dist,"descend");
		arma::uvec joint_R_dist_sort = arma::sort_index(joint_R_dist,"descend");
		if (joint_L_dist_sort.n_rows > 10) {
			arma::Col<double> potential_L_wingtip(4);
			arma::Col<double> potential_R_wingtip(4);
			potential_L_wingtip = arma::mean(pcl_vec[i].cols(joint_L_dist_sort.rows(0,9)),1);
			potential_L_wingtip(3) = sqrt(arma::mean(joint_L_dist(joint_L_dist_sort.rows(0,9))));
			potential_R_wingtip = arma::mean(pcl_vec[i].cols(joint_R_dist_sort.rows(0,9)),1);
			potential_R_wingtip(3) = sqrt(arma::mean(joint_R_dist(joint_R_dist_sort.rows(0,9))));

			// Calculate the wing kinematic angles w.r.t. the fixed body ref frame
			arma::Col<double> wing_L_axis = M_90.submat(0,0,2,2).t()*M_strk.submat(0,0,2,2).t()*M_body_fixed.submat(0,0,2,2).t()*(potential_L_wingtip.rows(0,2)-joint_L_loc);
			arma::Col<double> wing_R_axis = M_90.submat(0,0,2,2).t()*M_strk.submat(0,0,2,2).t()*M_body_fixed.submat(0,0,2,2).t()*(potential_R_wingtip.rows(0,2)-joint_R_loc);

			//double phi_L   = atan2(-wing_L_axis(0),wing_L_axis(1));
			double phi_L   = atan2(wing_L_axis(2),wing_L_axis(1));
			//double theta_L = atan2(wing_L_axis(2),sqrt(pow(wing_L_axis(0),2)+pow(wing_L_axis(1),2)));
			double theta_L = atan2(wing_L_axis(0),sqrt(pow(wing_L_axis(1),2)+pow(wing_L_axis(2),2)));
			//double phi_R   = atan2(-wing_R_axis(0),-wing_R_axis(1));
			double phi_R   = atan2(wing_R_axis(2),-wing_R_axis(1));
			//double theta_R = atan2(wing_R_axis(2),sqrt(pow(wing_R_axis(0),2)+pow(wing_R_axis(1),2)));
			double theta_R = atan2(wing_R_axis(0),sqrt(pow(wing_R_axis(1),2)+pow(wing_R_axis(2),2)));

			// To determine eta convert to wing reference frame and determine the principal components in the x-z plane:

			arma::Mat<double> M_phi_L = {{1.0, 0.0, 0.0},
				{0.0, cos(phi_L), -sin(phi_L)},
				{0.0, sin(phi_L), cos(phi_L)}};

			arma::Mat<double> M_theta_L = {{cos(-theta_L), -sin(-theta_L), 0.0},
				{sin(-theta_L), cos(-theta_L), 0.0},
				{0.0, 0.0, 1.0}};

			arma::Mat<double> M_phi_R = {{1.0, 0.0, 0.0},
				{0.0, cos(-phi_R), -sin(-phi_R)},
				{0.0, sin(-phi_R), cos(-phi_R)}};

			arma::Mat<double> M_theta_R = {{cos(theta_R), -sin(theta_R), 0.0},
				{sin(theta_R), cos(theta_R), 0.0},
				{0.0, 0.0, 1.0}};

			arma::Mat<double> wing_L_pts = M_theta_L.t()*M_phi_L.t()*M_90.submat(0,0,2,2).t()*M_strk.submat(0,0,2,2).t()*M_body_fixed.submat(0,0,2,2).t()*(pcl_vec[i].rows(0,2).each_col()-joint_L_loc);
			arma::Mat<double> wing_R_pts = M_theta_R.t()*M_phi_R.t()*M_90.submat(0,0,2,2).t()*M_strk.submat(0,0,2,2).t()*M_body_fixed.submat(0,0,2,2).t()*(pcl_vec[i].rows(0,2).each_col()-joint_R_loc);

			// PCA
			//arma::Mat<double> pca_coeff_L;
			//arma::Mat<double> pca_score_L;
			//arma::Col<double> pca_latent_L;
			//arma::Col<double> pca_tsquared_L;

			//arma::Mat<double> A_L = arma::join_cols(wing_L_pts.row(0)-arma::mean(wing_L_pts.row(0)), wing_L_pts.row(2)-arma::mean(wing_L_pts.row(2)));

			//arma::princomp(pca_coeff_L, pca_score_L, pca_latent_L, pca_tsquared_L, A_L.t());

			//cout << "eigenvectors" << endl;
			//cout << arma::eig_gen(pca_coeff_L) << endl;

			double eta_L = atan2(arma::mean(wing_L_pts.row(2)),-arma::mean(wing_L_pts.row(0)));
			double eta_R = atan2(arma::mean(wing_R_pts.row(2)),-arma::mean(wing_R_pts.row(0)));

			/*
			if (eta_L < -eta_bound) {
				eta_L = eta_L+PI;
			}
			else if (eta_L > eta_bound) {
				eta_L = eta_L-PI;
			}

			if (eta_R < -eta_bound) {
				eta_R = eta_L+PI;
			}
			else if (eta_R > eta_bound) {
				eta_R = eta_R-PI;
			}
			*/

			bool L_candidate = false;
			bool R_candidate = false;

			if ((abs(phi_L) <= phi_bound) && (abs(theta_L) <= theta_bound)) {
				if ((potential_L_wingtip(3)>0.7*wing_length_L)&& (potential_L_wingtip(3)<1.3*wing_length_L)) {
					L_candidate = true;
				}
			}

			if ((abs(phi_R) <= phi_bound) && (abs(theta_R) <= theta_bound)) {
				if ((potential_R_wingtip(3)>0.7*wing_length_R)&& (potential_R_wingtip(3)<1.3*wing_length_R)) {
					R_candidate = true;
				}
			}

			if ((L_candidate==true) && (R_candidate==false)) {
				pcl_wingtip_L.push_back(pcl_vec[i]);
				wingtip_info(0) = potential_L_wingtip(0);
				wingtip_info(1) = potential_L_wingtip(1);
				wingtip_info(2) = potential_L_wingtip(2);
				wingtip_info(3) = potential_L_wingtip(3);
				wingtip_info(4) = phi_L;
				wingtip_info(5) = theta_L;
				wingtip_info(6) = eta_L;
				wingtip_info(7) = pcl_vec[i].n_cols;
				wingtip_info_L.push_back(wingtip_info);
			}
			else if ((R_candidate==true) && (L_candidate==false)) {
				pcl_wingtip_R.push_back(pcl_vec[i]);
				wingtip_info(0) = potential_R_wingtip(0);
				wingtip_info(1) = potential_R_wingtip(1);
				wingtip_info(2) = potential_R_wingtip(2);
				wingtip_info(3) = potential_R_wingtip(3);
				wingtip_info(4) = phi_R;
				wingtip_info(5) = theta_R;
				wingtip_info(6) = eta_R;
				wingtip_info(7) = pcl_vec[i].n_cols;
				wingtip_info_R.push_back(wingtip_info);
			}
			else if ((L_candidate == true) && (R_candidate==true)) {

				//bool select_L = false;
				//bool select_R = false;

				if (abs(phi_L)>abs(phi_R)) {
					pcl_wingtip_R.push_back(pcl_vec[i]);
					wingtip_info(0) = potential_R_wingtip(0);
					wingtip_info(1) = potential_R_wingtip(1);
					wingtip_info(2) = potential_R_wingtip(2);
					wingtip_info(3) = potential_R_wingtip(3);
					wingtip_info(4) = phi_R;
					wingtip_info(5) = theta_R;
					wingtip_info(6) = eta_R;
					wingtip_info(7) = pcl_vec[i].n_cols;
					wingtip_info_R.push_back(wingtip_info);
				}
				else if (abs(phi_L)<abs(phi_R)) {
					pcl_wingtip_L.push_back(pcl_vec[i]);
					wingtip_info(0) = potential_L_wingtip(0);
					wingtip_info(1) = potential_L_wingtip(1);
					wingtip_info(2) = potential_L_wingtip(2);
					wingtip_info(3) = potential_L_wingtip(3);
					wingtip_info(4) = phi_L;
					wingtip_info(5) = theta_L;
					wingtip_info(6) = eta_L;
					wingtip_info(7) = pcl_vec[i].n_cols;
					wingtip_info_L.push_back(wingtip_info);
				}
				else {
					// do nothing
				}
			}
			else {
				// do nothing
			}
		}
	}

	arma::Mat<double> wingtip_pcl_left;
	arma::Mat<double> wingtip_pcl_right;
	arma::Col<double> wingtip_info_left;
	arma::Col<double> wingtip_info_right;

	if (pcl_wingtip_L.size()==1) {
		wingtip_pcl_left = pcl_wingtip_L[0];
		wingtip_pcl_left.row(3).fill(2.0);
		wingtip_info_left = wingtip_info_L[0];
	}
	else if (pcl_wingtip_L.size()>1) {
		int N_segs_L = pcl_wingtip_L.size();
		arma::Row<double> seg_size_L(N_segs_L);
		for (int j=0; j<N_segs_L; j++) {
			seg_size_L(j) = wingtip_info_L[j](7);
		}
		arma::uvec L_sort = arma::sort_index(seg_size_L,"descend");
		wingtip_pcl_left = pcl_wingtip_L[L_sort[0]];
		wingtip_pcl_left.row(3).fill(2.0);
		wingtip_info_left = wingtip_info_L[L_sort[0]];
	}
	else {
		wingtip_pcl_left.zeros(4,1);
		wingtip_info_left.zeros(7);
	}

	if (pcl_wingtip_R.size()==1) {
		wingtip_pcl_right = pcl_wingtip_R[0];
		wingtip_pcl_right.row(3).fill(3.0);
		wingtip_info_right = wingtip_info_R[0];
	}
	else if (pcl_wingtip_R.size()>1) {
		int N_segs_R = pcl_wingtip_R.size();
		arma::Row<double> seg_size_R(N_segs_R);
		for (int j=0; j<N_segs_R; j++) {
			seg_size_R(j) = wingtip_info_R[j](7);
		}
		arma::uvec R_sort = arma::sort_index(seg_size_R,"descend");
		wingtip_pcl_right = pcl_wingtip_R[R_sort[0]];
		wingtip_pcl_right.row(3).fill(3.0);
		wingtip_info_right = wingtip_info_R[R_sort[0]];
	}
	else {
		wingtip_pcl_right.zeros(4,1);
		wingtip_info_right.zeros(8);
	}

	vector<arma::Mat<double>> pcl_out;
	pcl_out.push_back(wingtip_pcl_left);
	pcl_out.push_back(wingtip_pcl_right);
	
	vector<arma::Col<double>> info_out;
	info_out.push_back(wingtip_info_left);
	info_out.push_back(wingtip_info_right);

	tuple<vector<arma::Mat<double>>,vector<arma::Col<double>>> wingtips_out = make_tuple(pcl_out,info_out);

	return wingtips_out;
}


vector<arma::Mat<double>> InitState::SelectWingTips(vector<arma::Mat<double>> &pcl_vec) {

	int N_segs = pcl_vec.size();

	arma::Row<double> seg_size;
	seg_size.zeros(N_segs);

	arma::Col<double> body_center = {bbox_body.box_center[0], bbox_body.box_center[1], bbox_body.box_center[2]};

	vector<arma::Mat<double>> pcl_wingtip;
	vector<arma::Col<double>> potential_wingtip;

	for (int i=0; i<N_segs; i++) {
		arma::Row<double> center_dist = arma::sum((pcl_vec[i].rows(0,2).each_col()-body_center)%(pcl_vec[i].rows(0,2).each_col()-body_center));
		arma::uvec dist_sort = arma::sort_index(center_dist,"descend");
		if (dist_sort.n_rows > 10) {
			arma::Col<double> potential_wingtip_pt(4);
			potential_wingtip_pt = arma::mean(pcl_vec[i].cols(dist_sort.rows(0,9)),1);
			potential_wingtip_pt(3) = sqrt(arma::mean(center_dist(dist_sort.rows(0,9))));
			if (potential_wingtip_pt(3) > (0.8*wing_length_L) && potential_wingtip_pt(3) < (1.2*wing_length_L)) {
				seg_size(i) = pcl_vec[i].n_cols;
				pcl_wingtip.push_back(pcl_vec[i]);
				potential_wingtip.push_back(potential_wingtip_pt);
			}
			else {
				seg_size(i) = 0;
				pcl_wingtip.push_back(pcl_vec[i]);
				potential_wingtip.push_back(potential_wingtip_pt);
			}
		}
	}

	arma::uvec seg_size_sort = arma::sort_index(seg_size,"descend");

	// Select the biggest pointcloud, calculate the distance of the other pointclouds to the wingtip 
	// and pick the largest product between distance*pointcloud_size

	vector<arma::Mat<double>> wing_tip_vec;
	vector<arma::Col<double>> wing_tip_pos;

	if (N_segs == 1) {
		wing_tip_vec.push_back(pcl_wingtip[seg_size_sort(0)]);
		wing_tip_pos.push_back(potential_wingtip[seg_size_sort(0)]);
	}
	else if (N_segs > 1) {
		wing_tip_vec.push_back(pcl_wingtip[seg_size_sort(0)]);
		wing_tip_pos.push_back(potential_wingtip[seg_size_sort(0)]);
		arma::Col<double> wt_1 = potential_wingtip[seg_size_sort(0)];
		arma::Col<double> wt_2(4);
		arma::Row<double> candidate_score;
		candidate_score.zeros(N_segs);
		for (int j=0; j<N_segs; j++) {
			wt_2 = potential_wingtip[j];
			candidate_score(j) = sqrt(pow(wt_1(0)-wt_2(0),2)+pow(wt_1(1)-wt_2(1),2)+pow(wt_1(2)-wt_2(2),2))*seg_size(j);
		}
		wing_tip_vec.push_back(pcl_wingtip[candidate_score.index_max()]);
		wing_tip_pos.push_back(potential_wingtip[candidate_score.index_max()]);
	}

	return wing_tip_vec;
}

void InitState::Alpha3D(arma::Mat<double> &pcl_in) {
	// Convert pcl_in to a point list:
	list<Point> lp;
	for (int i=0; i<pcl_in.n_cols; i++) {
		lp.push_back(Point(pcl_in(0,i),pcl_in(1,i),pcl_in(2,i)));
	}

	// Compute alpha shape
	Alpha_shape_3 as(lp.begin(),lp.end());
	cout << "Alpha shape computed in REGULARIZED mode by default" << endl;

	// Find optimal alpha value
	Alpha_iterator opt = as.find_optimal_alpha(1);
	cout << "Optimal alpha value to get one connected component is " << *opt << endl;

	as.set_alpha(*opt);
	assert(as.number_of_solid_components() == 1);
}

bbox InitState::BoundingBox(arma::Mat<double> &pcl_in) {

  	// Return bounding box structure

  	struct bbox bounding_box;

  	bounding_box.box_center[0] = 0.0;
  	bounding_box.box_center[1] = 0.0;
  	bounding_box.box_center[2] = 0.0;
  	bounding_box.box_corners[0][0] = 0.0;
  	bounding_box.box_corners[1][0] = 0.0;
  	bounding_box.box_corners[2][0] = 0.0;
	bounding_box.box_corners[0][1] = 0.0;
  	bounding_box.box_corners[1][1] = 0.0;
  	bounding_box.box_corners[2][1] = 0.0;
  	bounding_box.box_corners[0][2] = 0.0;
  	bounding_box.box_corners[1][2] = 0.0;
  	bounding_box.box_corners[2][2] = 0.0;
  	bounding_box.box_corners[0][3] = 0.0;
  	bounding_box.box_corners[1][3] = 0.0;
  	bounding_box.box_corners[2][3] = 0.0;
  	bounding_box.box_corners[0][4] = 0.0;
  	bounding_box.box_corners[1][4] = 0.0;
  	bounding_box.box_corners[2][4] = 0.0;
  	bounding_box.box_corners[0][5] = 0.0;
  	bounding_box.box_corners[1][5] = 0.0;
  	bounding_box.box_corners[2][5] = 0.0;
  	bounding_box.box_corners[0][6] = 0.0;
  	bounding_box.box_corners[1][6] = 0.0;
  	bounding_box.box_corners[2][6] = 0.0;
  	bounding_box.box_corners[0][7] = 0.0;
  	bounding_box.box_corners[1][7] = 0.0;
  	bounding_box.box_corners[2][7] = 0.0;
	bounding_box.q_box[0] = 0.0;
	bounding_box.q_box[1] = 0.0;
	bounding_box.q_box[2] = 0.0;
	bounding_box.q_box[3] = 0.0;
	bounding_box.R_box[0][0] = 0.0;
	bounding_box.R_box[0][1] = 0.0;
	bounding_box.R_box[0][2] = 0.0;
	bounding_box.R_box[1][0] = 0.0;
	bounding_box.R_box[1][1] = 0.0;
	bounding_box.R_box[1][2] = 0.0;
	bounding_box.R_box[2][0] = 0.0;
	bounding_box.R_box[2][1] = 0.0;
	bounding_box.R_box[2][2] = 0.0;
	bounding_box.mass_center[0] = 0.0;
	bounding_box.mass_center[1] = 0.0;
	bounding_box.mass_center[2] = 0.0;
	bounding_box.eigen_values[0] = 0.0;
	bounding_box.eigen_values[1] = 0.0;
	bounding_box.eigen_values[2] = 0.0;
	bounding_box.eigen_vectors[0][0] = 0.0;
	bounding_box.eigen_vectors[0][1] = 0.0;
	bounding_box.eigen_vectors[0][2] = 0.0;
	bounding_box.eigen_vectors[1][0] = 0.0;
	bounding_box.eigen_vectors[1][1] = 0.0;
	bounding_box.eigen_vectors[1][2] = 0.0;
	bounding_box.eigen_vectors[2][0] = 0.0;
	bounding_box.eigen_vectors[2][1] = 0.0;
	bounding_box.eigen_vectors[2][2] = 0.0;
	bounding_box.moment_of_inertia[0] = 0.0;
	bounding_box.moment_of_inertia[1] = 0.0;
	bounding_box.moment_of_inertia[2] = 0.0;
	bounding_box.eccentricity[0] = 0.0;
	bounding_box.eccentricity[1] = 0.0;
	bounding_box.eccentricity[2] = 0.0;
	bounding_box.volume =  0.0;

	return bounding_box;
}

np::ndarray InitState::ReturnBBox(vector<struct bbox> &bbox_in) {
	int N_boxes = bbox_in.size();
	p::tuple shape = p::make_tuple(24,1);
	np::dtype dtype = np::dtype::get_builtin<double>();
	np::ndarray bbox_out = np::zeros(shape,dtype);

	if (N_boxes > 0) {
		p::tuple shape = p::make_tuple(24,N_boxes);
		np::dtype dtype = np::dtype::get_builtin<double>();
		bbox_out = np::zeros(shape,dtype);
		for (int i=0; i<N_boxes; i++) {
			arma::Mat<double> rot_mat = {{bbox_in[i].R_box[0][0], bbox_in[i].R_box[0][1], bbox_in[i].R_box[0][2], bbox_in[i].box_center[0]},
										 {bbox_in[i].R_box[1][0], bbox_in[i].R_box[1][1], bbox_in[i].R_box[1][2], bbox_in[i].box_center[1]},
										 {bbox_in[i].R_box[2][0], bbox_in[i].R_box[2][1], bbox_in[i].R_box[2][2], bbox_in[i].box_center[2]},
										 {0.0, 0.0, 0.0, 1.0}};
			for (int j=0; j<8; j++) {
				arma::Col<double> corner_point = {bbox_in[i].box_corners[0][j], bbox_in[i].box_corners[1][j], bbox_in[i].box_corners[2][j], 1.0};
				arma::Col<double> new_corner_point = rot_mat*corner_point;
				bbox_out[j*3][i]   = new_corner_point(0);
				bbox_out[j*3+1][i] = new_corner_point(1);
				bbox_out[j*3+2][i] = new_corner_point(2);
			}
		}
	}
	else {
		for (int j=0; j<8; j++) {
			bbox_out[j*3]   = 0.0;
			bbox_out[j*3+1] = 0.0;
			bbox_out[j*3+2] = 0.0;
		}
	}
	return bbox_out;
}

np::ndarray InitState::ReturnJointLocs() {
	p::tuple shape = p::make_tuple(3,2);
	np::dtype dtype = np::dtype::get_builtin<double>();
	np::ndarray loc_out = np::zeros(shape,dtype);

	loc_out[0][0] = joint_L_loc(0);
	loc_out[1][0] = joint_L_loc(1);
	loc_out[2][0] = joint_L_loc(2);
	loc_out[0][1] = joint_R_loc(0);
	loc_out[1][1] = joint_R_loc(1);
	loc_out[2][1] = joint_R_loc(2);

	return loc_out;
}

np::ndarray InitState::ReturnSRF() {
	p::tuple shape = p::make_tuple(4,4);
	np::dtype dtype = np::dtype::get_builtin<double>();
	np::ndarray M_out = np::zeros(shape,dtype);

	arma::Mat<double> M_stroke_out;
	M_stroke_out.eye(4,4);

	M_stroke_out.submat(0,0,2,2) = M_body_fixed.submat(0,0,2,2)*M_strk.submat(0,0,2,2);

	M_stroke_out(0,3) = M_body_fixed(0,3);
	M_stroke_out(1,3) = M_body_fixed(1,3);
	M_stroke_out(2,3) = M_body_fixed(2,3);

	M_out[0][0] = M_stroke_out(0,0);
	M_out[0][1] = M_stroke_out(0,1);
	M_out[0][2] = M_stroke_out(0,2);
	M_out[0][3] = M_stroke_out(0,3);
	M_out[1][0] = M_stroke_out(1,0);
	M_out[1][1] = M_stroke_out(1,1);
	M_out[1][2] = M_stroke_out(1,2);
	M_out[1][3] = M_stroke_out(1,3);
	M_out[2][0] = M_stroke_out(2,0);
	M_out[2][1] = M_stroke_out(2,1);
	M_out[2][2] = M_stroke_out(2,2);
	M_out[2][3] = M_stroke_out(2,3);
	M_out[3][0] = M_stroke_out(3,0);
	M_out[3][1] = M_stroke_out(3,1);
	M_out[3][2] = M_stroke_out(3,2);
	M_out[3][3] = M_stroke_out(3,3);

	return M_out;
}

np::ndarray InitState::ReturnWingtipPCLS() {
	int N = 0;
	bool left_wt_exists = false;
	bool right_wt_exists = false;

	if (wingtip_pcl_L.n_cols > 1) {
		N = N + wingtip_pcl_L.n_cols;
		left_wt_exists = true;
	}

	if (wingtip_pcl_R.n_cols > 1) {
		N = N + wingtip_pcl_R.n_cols;
		right_wt_exists = true;
	}

	if (N==0) {
		N = 2;
	}

	p::tuple shape = p::make_tuple(4,N);
	np::dtype dtype = np::dtype::get_builtin<double>();
	np::ndarray pcl_out = np::zeros(shape,dtype);

	if (left_wt_exists == true && right_wt_exists == true) {

		int k=0;

		for (int i=0; i<wingtip_pcl_L.n_cols; i++) {
			pcl_out[0][k] = wingtip_pcl_L(0,i);
			pcl_out[1][k] = wingtip_pcl_L(1,i);
			pcl_out[2][k] = wingtip_pcl_L(2,i);
			pcl_out[3][k] = 1.0;
			k++;
		}
		for (int j=0; j<wingtip_pcl_R.n_cols; j++) {
			pcl_out[0][k] = wingtip_pcl_R(0,j);
			pcl_out[1][k] = wingtip_pcl_R(1,j);
			pcl_out[2][k] = wingtip_pcl_R(2,j);
			pcl_out[3][k] = 2.0;
			k++;
		}
	}
	else if (left_wt_exists == true && right_wt_exists == false) {

		for (int i=0; i<wingtip_pcl_L.n_cols; i++) {
			pcl_out[0][i] = wingtip_pcl_L(0,i);
			pcl_out[1][i] = wingtip_pcl_L(1,i);
			pcl_out[2][i] = wingtip_pcl_L(2,i);
			pcl_out[3][i] = 1.0;
		}

		pcl_out[0][N-1] = 0.0;
		pcl_out[1][N-1] = 0.0;
		pcl_out[2][N-1] = 0.0;
		pcl_out[3][N-1] = 2.0;
	}
	else if (left_wt_exists == false && right_wt_exists == true) {

		for (int i=0; i<wingtip_pcl_R.n_cols; i++) {
			pcl_out[0][i] = wingtip_pcl_R(0,i);
			pcl_out[1][i] = wingtip_pcl_R(1,i);
			pcl_out[2][i] = wingtip_pcl_R(2,i);
			pcl_out[3][i] = 2.0;
		}

		pcl_out[0][N-1] = 0.0;
		pcl_out[1][N-1] = 0.0;
		pcl_out[2][N-1] = 0.0;
		pcl_out[3][N-1] = 1.0;
	}
	else {

		pcl_out[0][0] = 0.0;
		pcl_out[1][0] = 0.0;
		pcl_out[2][0] = 0.0;
		pcl_out[3][0] = 1.0;

		pcl_out[0][0] = 0.0;
		pcl_out[1][0] = 0.0;
		pcl_out[2][0] = 0.0;
		pcl_out[3][0] = 2.0;
	}

	return pcl_out;
}

np::ndarray InitState::ReturnWingtipBoxes() {
	int N = 0;
	bool left_wt_exists = false;
	bool right_wt_exists = false;

	if (wingtip_pcl_L.n_cols > 1) {
		N = N + 1;
		left_wt_exists = true;
	}

	if (wingtip_pcl_R.n_cols > 1) {
		N = N + 1;
		right_wt_exists = true;
	}

	if (N==0) {
		N=1;
	}

	p::tuple shape = p::make_tuple(24,N);
	np::dtype dtype = np::dtype::get_builtin<double>();
	np::ndarray bbox_out = np::zeros(shape,dtype);

	if (left_wt_exists == true && right_wt_exists == true) {

		arma::Mat<double> rot_L = {{bbox_wingtip_L.R_box[0][0], bbox_wingtip_L.R_box[0][1], bbox_wingtip_L.R_box[0][2], bbox_wingtip_L.box_center[0]},
								   {bbox_wingtip_L.R_box[1][0], bbox_wingtip_L.R_box[1][1], bbox_wingtip_L.R_box[1][2], bbox_wingtip_L.box_center[1]},
								   {bbox_wingtip_L.R_box[2][0], bbox_wingtip_L.R_box[2][1], bbox_wingtip_L.R_box[2][2], bbox_wingtip_L.box_center[2]},
								   {0.0, 0.0, 0.0, 1.0}};

		for (int j=0; j<8; j++) {
			arma::Col<double> corner_point = {bbox_wingtip_L.box_corners[0][j], bbox_wingtip_L.box_corners[1][j], bbox_wingtip_L.box_corners[2][j], 1.0};
			arma::Col<double> new_corner_point = rot_L*corner_point;
			bbox_out[j*3][0]   = new_corner_point(0);
			bbox_out[j*3+1][0] = new_corner_point(1);
			bbox_out[j*3+2][0] = new_corner_point(2);
		}

		arma::Mat<double> rot_R = {{bbox_wingtip_R.R_box[0][0], bbox_wingtip_R.R_box[0][1], bbox_wingtip_R.R_box[0][2], bbox_wingtip_R.box_center[0]},
								   {bbox_wingtip_R.R_box[1][0], bbox_wingtip_R.R_box[1][1], bbox_wingtip_R.R_box[1][2], bbox_wingtip_R.box_center[1]},
								   {bbox_wingtip_R.R_box[2][0], bbox_wingtip_R.R_box[2][1], bbox_wingtip_R.R_box[2][2], bbox_wingtip_R.box_center[2]},
								   {0.0, 0.0, 0.0, 1.0}};

		for (int j=0; j<8; j++) {
			arma::Col<double> corner_point = {bbox_wingtip_R.box_corners[0][j], bbox_wingtip_R.box_corners[1][j], bbox_wingtip_R.box_corners[2][j], 1.0};
			arma::Col<double> new_corner_point = rot_R*corner_point;
			bbox_out[j*3][1]   = new_corner_point(0);
			bbox_out[j*3+1][1] = new_corner_point(1);
			bbox_out[j*3+2][1] = new_corner_point(2);
		}
	}
	else if (left_wt_exists == true && right_wt_exists == false) {

		arma::Mat<double> rot_L = {{bbox_wingtip_L.R_box[0][0], bbox_wingtip_L.R_box[0][1], bbox_wingtip_L.R_box[0][2], bbox_wingtip_L.box_center[0]},
								   {bbox_wingtip_L.R_box[1][0], bbox_wingtip_L.R_box[1][1], bbox_wingtip_L.R_box[1][2], bbox_wingtip_L.box_center[1]},
								   {bbox_wingtip_L.R_box[2][0], bbox_wingtip_L.R_box[2][1], bbox_wingtip_L.R_box[2][2], bbox_wingtip_L.box_center[2]},
								   {0.0, 0.0, 0.0, 1.0}};

		for (int j=0; j<8; j++) {
			arma::Col<double> corner_point = {bbox_wingtip_L.box_corners[0][j], bbox_wingtip_L.box_corners[1][j], bbox_wingtip_L.box_corners[2][j], 1.0};
			arma::Col<double> new_corner_point = rot_L*corner_point;
			bbox_out[j*3][0]   = new_corner_point(0);
			bbox_out[j*3+1][0] = new_corner_point(1);
			bbox_out[j*3+2][0] = new_corner_point(2);
		}
	}
	else if (left_wt_exists == false && right_wt_exists == true) {

		arma::Mat<double> rot_R = {{bbox_wingtip_R.R_box[0][0], bbox_wingtip_R.R_box[0][1], bbox_wingtip_R.R_box[0][2], bbox_wingtip_R.box_center[0]},
								   {bbox_wingtip_R.R_box[1][0], bbox_wingtip_R.R_box[1][1], bbox_wingtip_R.R_box[1][2], bbox_wingtip_R.box_center[1]},
								   {bbox_wingtip_R.R_box[2][0], bbox_wingtip_R.R_box[2][1], bbox_wingtip_R.R_box[2][2], bbox_wingtip_R.box_center[2]},
								   {0.0, 0.0, 0.0, 1.0}};

		for (int j=0; j<8; j++) {
			arma::Col<double> corner_point = {bbox_wingtip_R.box_corners[0][j], bbox_wingtip_R.box_corners[1][j], bbox_wingtip_R.box_corners[2][j], 1.0};
			arma::Col<double> new_corner_point = rot_R*corner_point;
			bbox_out[j*3][0]   = new_corner_point(0);
			bbox_out[j*3+1][0] = new_corner_point(1);
			bbox_out[j*3+2][0] = new_corner_point(2);
		}
	}
	else {
		for (int j=0; j<8; j++) {
			bbox_out[j*3][0]   = 0.0;
			bbox_out[j*3+1][0] = 0.0;
			bbox_out[j*3+2][0] = 0.0;
		}
	}

	return bbox_out;
}

arma::Mat<double> InitState::OrthoNormalizeR(arma::Mat<double> R_in) {
	arma::Mat<double> R_out(3,3);
	R_out = arma::real(arma::expmat(0.5*(arma::real(arma::logmat(R_in))-arma::trans(arma::real(arma::logmat(R_in))))));
	return R_out;
}

np::ndarray InitState::ReturnSegPCL(frame_data &frame_in) {
	int N_pts = frame_in.seg_pcl.n_cols;
	p::tuple shape = p::make_tuple(4,N_pts);
	np::dtype dtype = np::dtype::get_builtin<double>();
	np::ndarray array_out = np::zeros(shape,dtype);
	for (int i=0; i<N_pts; i++) {
		array_out[0][i] = frame_in.seg_pcl(0,i);
		array_out[1][i] = frame_in.seg_pcl(1,i);
		array_out[2][i] = frame_in.seg_pcl(2,i);
		array_out[3][i] = frame_in.seg_pcl(3,i);
	}
	return array_out;
}