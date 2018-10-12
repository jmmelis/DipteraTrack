#include "contour_optimization.h"

#include "session_data.h"
#include "frame_data.h"
#include "focal_grid.h"
#include "image_segmentation.h"
#include "model_class.h"

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <string>
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <list>
#include <thread>
#include <chrono>
#include <algorithm>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <armadillo>

#include <nlopt.hpp>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Boolean_set_operations_2.h>
#include <CGAL/algorithm.h>
#include <CGAL/assertions.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Polygon_with_holes_2.h>
#include <CGAL/Polygon_set_2.h>



#define PI 3.14159265

using namespace std;

namespace p = boost::python;
namespace np = boost::python::numpy;


typedef CGAL::Exact_predicates_inexact_constructions_kernel KI;
//typedef CGAL::Exact_predicates_exact_constructions_kernel KE;

typedef KI::Point_2 				   Point_2;
typedef CGAL::Polygon_2<KI>            Polygon_2;
typedef CGAL::Polygon_with_holes_2<KI> Polygon_with_holes_2;
typedef CGAL::Polygon_set_2<KI> 	   Polygon_set_2;

typedef struct {
	FocalGrid* fg;
	ContourOpt* opt;
	ModelClass* mdl;
	int N_cam;
	const vector<list<Polygon_with_holes_2>> body_dest;
	const vector<list<Polygon_with_holes_2>> wing_L_dest;
	const vector<list<Polygon_with_holes_2>> wing_R_dest;
} cost_func_data;

double cost_function(const vector<double> &x, vector<double> &grad, void *data) {

	clock_t start;
	double duration;

	start = clock();

	cost_func_data *d = reinterpret_cast<cost_func_data*>(data);
	FocalGrid* fg = d->fg;
	ContourOpt* opt = d->opt;
	ModelClass* mdl = d->mdl;
	int N_cam = d->N_cam;
	//vector<list<Polygon_with_holes_2>> body_dest = d->body_dest;
	//vector<list<Polygon_with_holes_2>> wing_L_dest = d->wing_L_dest;
	//vector<list<Polygon_with_holes_2>> wing_R_dest = d->wing_R_dest;

	vector<double> state_now;
	state_now.push_back(x[0]);
	state_now.push_back(x[1]);
	state_now.push_back(x[2]);
	state_now.push_back(x[3]);
	state_now.push_back(x[4]);
	state_now.push_back(x[5]);
	state_now.push_back(x[6]);
	state_now.push_back(x[7]);
	state_now.push_back(x[8]);
	state_now.push_back(x[9]);
	state_now.push_back(x[10]);
	state_now.push_back(x[11]);
	state_now.push_back(x[12]);
	state_now.push_back(x[13]);
	state_now.push_back(x[14]);

	vector<list<Polygon_with_holes_2>> body_src;
	vector<list<Polygon_with_holes_2>> wing_L_src;
	vector<list<Polygon_with_holes_2>> wing_R_src;

	vector<list<Polygon_with_holes_2>> body_dest; 
	vector<list<Polygon_with_holes_2>> wing_L_dest; 
	vector<list<Polygon_with_holes_2>> wing_R_dest;

	// Convert state x to affine matrix transforms:
	cout << "return src state" << endl;
	vector<double> src_state = mdl->ReturnSRCState(state_now);
	cout << "size src state" << src_state.size() << endl;

	cout << "get src contours" << endl;
	for (int i=0; i<N_cam; i++) {
		cout << "get silhouette" << endl;
		vector<Polygon_2> m_silhouette = mdl->ReturnSilhouette2(*fg,src_state,i);
		cout << "size silhouette" << m_silhouette.size() << endl;
		vector<Polygon_2> m_body;
		for (int j=0; j<3; j++) {
			m_body.push_back(m_silhouette[j]);
		}
		list<Polygon_with_holes_2> body_contour_poly = opt->CalcUnion(m_body);
		body_src.push_back(body_contour_poly);
		body_dest.push_back(d->body_dest[i]);
		vector<Polygon_2> m_wing_L;
		m_wing_L.push_back(m_silhouette[3]);
		list<Polygon_with_holes_2> wing_L_contour_poly = opt->CalcUnion(m_wing_L);
		list<Polygon_with_holes_2> wing_L_complement = opt->CalcComplement(body_contour_poly, wing_L_contour_poly);
		wing_L_src.push_back(wing_L_complement);
		wing_L_dest.push_back(d->wing_L_dest[i]);
		vector<Polygon_2> m_wing_R;
		m_wing_R.push_back(m_silhouette[4]);
		list<Polygon_with_holes_2> wing_R_contour_poly = opt->CalcUnion(m_wing_R);
		list<Polygon_with_holes_2> wing_R_complement = opt->CalcComplement(body_contour_poly, wing_R_contour_poly);
		wing_R_src.push_back(wing_R_complement);
		wing_R_dest.push_back(d->wing_R_dest[i]);
	}

	// Calculate the cost by calculating the symmetric difference per view per polygon list:
	double cost = 0.0;

	list<Polygon_with_holes_2> body_diff;
	list<Polygon_with_holes_2> wing_L_diff;
	list<Polygon_with_holes_2> wing_R_diff;

	cout << "body src vec size: " << body_src.size() << endl;
	cout << "body dest vec size: " << body_dest.size() << endl;

	for (int i=0; i<N_cam; i++) {
		cout << "i " << i << endl;
		if (body_dest[i].size()>0 && body_src[i].size()>0) {
			//body_diff.clear();
			cout << "Calc symmetric difference body " << endl;
			list<Polygon_with_holes_2> body_diff = opt->CalcSymmetricDifference(body_dest[i],body_src[i]);
			if (body_diff.size() > 0) {
				for (list<Polygon_with_holes_2>::iterator it1 = body_diff.begin(); it1 != body_diff.end(); ++it1) {
					cout << "area: " << it1->outer_boundary().area() << endl;
					//cost += it->outer_boundary().area();
				}
			}
		}
		/*
		if (wing_L_dest[i].size()>0 && wing_L_src[i].size()>0) {
			//wing_L_diff.clear();
			cout << "Calc symmetric difference wing_L " << endl;
			list<Polygon_with_holes_2> wing_L_diff = opt->CalcSymmetricDifference(wing_L_dest[i],wing_L_src[i]);
			if (wing_L_diff.size() > 0) {
				for (list<Polygon_with_holes_2>::iterator it2 = wing_L_diff.begin(); it2 != wing_L_diff.end(); ++it2) {
					cout << "area: " << it2->outer_boundary().area() << endl;
					//cost += it->outer_boundary().area();
				}
			}
		}
		cout << "dest polygons size " << wing_R_dest[i].size() << endl;
		cout << "src polygons size "<< wing_R_src[i].size() << endl;
		if (wing_R_dest[i].size()>0 && wing_R_src[i].size()>0) {
			//wing_R_diff.clear();
			cout << "wing_R_dest" << endl;
			cout << wing_R_dest[i].front() << endl;
			cout << "wing_R_src" << endl;
			cout << wing_R_src[i].front() << endl;
			cout << "Calc symmetric difference wing_R " << endl;
			list<Polygon_with_holes_2> wing_R_diff = opt->CalcSymmetricDifference(wing_R_dest[i],wing_R_src[i]);
			cout << "wing_R_diff size " << wing_R_diff.size() << endl;
			if (wing_R_diff.size() > 0) {
				for (list<Polygon_with_holes_2>::iterator it3 = wing_R_diff.begin(); it3 != wing_R_diff.end(); ++it3) {
					cout << "area: " << it3->outer_boundary().area() << endl;
					//cost += it->outer_boundary().area();
				}
			}
		}
		*/
	}

	cout << "cost: " << cost << endl;
	duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;
    cout<<"printf: "<< duration <<'\n';

	return cost;
}

double quat_constraint_body(const std::vector<double> &x, std::vector<double> &grad, void *data) {
	double q_norm = sqrt(pow(x[0],2)+pow(x[1],2)+pow(x[2],2)+pow(x[3],2));
	return (1.0-q_norm);
}

double quat_constraint_wing_L(const std::vector<double> &x, std::vector<double> &grad, void *data) {
	double q_norm = sqrt(pow(x[7],2)+pow(x[8],2)+pow(x[9],2)+pow(x[10],2));
	return (1.0-q_norm);
}

double quat_constraint_wing_R(const std::vector<double> &x, std::vector<double> &grad, void *data) {
	double q_norm = sqrt(pow(x[11],2)+pow(x[12],2)+pow(x[13],2)+pow(x[14],2));
	return (1.0-q_norm);
}

/*
void Optimize(FocalGrid &fg, frame_data &frame_in, ModelClass &mdl, ContourOpt &opt) {

	int N_cam = frame_in.N_cam;

	cost_func_data cost_data = {fg, opt, mdl, N_cam, opt.body_dest_polygons, opt.wing_L_dest_polygons, opt.wing_R_dest_polygons};

    vector<double> lb(15);
    lb[0] = -1.0;
	lb[1] = -1.0;
	lb[2] = -1.0;
	lb[3] = -1.0;
	lb[4] = -10.0;
	lb[5] = -10.0;
	lb[6] = -10.0;
	lb[7] = -1.0;
	lb[8] = -1.0;
	lb[9] = -1.0;
	lb[10] = -1.0;
	lb[11] = -1.0;
	lb[12] = -1.0;
	lb[13] = -1.0;
	lb[14] = -1.0;

	vector<double> ub(15);
	ub[0] = -1.0;
	ub[1] = -1.0;
	ub[2] = -1.0;
	ub[3] = -1.0;
	ub[4] = -10.0;
	ub[5] = -10.0;
	ub[6] = -10.0;
	ub[7] = -1.0;
	ub[8] = -1.0;
	ub[9] = -1.0;
	ub[10] = -1.0;
	ub[11] = -1.0;
	ub[12] = -1.0;
	ub[13] = -1.0;
	ub[14] = -1.0;

	vector<double> x(15);
	x[0] = frame_in.init_state(0,0);
	x[1] = frame_in.init_state(1,0);
	x[2] = frame_in.init_state(2,0);
	x[3] = frame_in.init_state(3,0);
	x[4] = frame_in.init_state(4,0);
	x[5] = frame_in.init_state(5,0);
	x[6] = frame_in.init_state(6,0);
	x[7] = frame_in.init_state(0,1);
	x[8] = frame_in.init_state(1,1);
	x[9] = frame_in.init_state(2,1);
	x[10] = frame_in.init_state(3,1);
	x[11] = frame_in.init_state(0,2);
	x[12] = frame_in.init_state(1,2);
	x[13] = frame_in.init_state(2,2);
	x[14] = frame_in.init_state(3,2);

	nlopt::opt opt_alg(nlopt::LN_COBYLA, 15);
	opt_alg.set_lower_bounds(lb);
	opt_alg.set_upper_bounds(ub);
	opt_alg.set_min_objective(cost_function, &cost_data);
	opt_alg.add_equality_constraint(quat_constraint_body, NULL, 1e-6);
	opt_alg.add_equality_constraint(quat_constraint_wing_L, NULL, 1e-6);
	opt_alg.add_equality_constraint(quat_constraint_wing_R, NULL, 1e-6);
	opt_alg.set_xtol_rel(1e-5);

	cout << " -------------------- " << endl;
	cout << "x start: " << x[0] << ", " << x[1] << ", " << x[2] << ", " << x[3] << ", " << x[4] 
		<< ", " << x[5] << ", " << x[6] << endl;
	cout << x[7] << ", " << x[8] << ", " << x[9] << ", " << x[10] << endl;
	cout << x[11] << ", " << x[12] << ", " << x[13] << ", " << x[14] << endl;

	double minf;

	nlopt::result result = opt_alg.optimize(x, minf);

	cout << "x opt: " << x[0] << ", " << x[1] << ", " << x[2] << ", " << x[3] << ", " << x[4] 
		<< ", " << x[5] << ", " << x[6] << endl;
	cout << x[7] << ", " << x[8] << ", " << x[9] << ", " << x[10] << endl;
	cout << x[11] << ", " << x[12] << ", " << x[13] << ", " << x[14] << endl;

	cout << "quaternion norm body: " << sqrt(pow(x[0],2)+pow(x[1],2)+pow(x[2],2)+pow(x[3],2)) << endl;
	cout << "quaternion norm wing L: " << sqrt(pow(x[7],2)+pow(x[8],2)+pow(x[9],2)+pow(x[10],2)) << endl;
	cout << "quaternion norm wing R: " << sqrt(pow(x[11],2)+pow(x[12],2)+pow(x[13],2)+pow(x[14],2)) << endl;
	cout << " -------------------- " << endl;
	cout << endl;
}
*/

ContourOpt::ContourOpt() {
	// Empty
}

void ContourOpt::OptimizeState(FocalGrid &fg, frame_data &frame_in, ModelClass &mdl, ImagSegm &seg) {
	ContourOpt::FindDestinationContour(fg, frame_in, seg);
	ContourOpt::FindInitContour(fg, frame_in, mdl, seg);
	ContourOpt::Optimize(fg, frame_in, mdl);
}

void ContourOpt::Optimize(FocalGrid &fg, frame_data &frame_in, ModelClass &mdl) {

	int N_cam = frame_in.N_cam;

	cost_func_data cost_data = {&fg, this, &mdl, N_cam, body_dest_polygons, wing_L_dest_polygons, wing_R_dest_polygons};

    vector<double> lb(15);
    lb[0] = -1.0;
	lb[1] = -1.0;
	lb[2] = -1.0;
	lb[3] = -1.0;
	lb[4] = -10.0;
	lb[5] = -10.0;
	lb[6] = -10.0;
	lb[7] = -1.0;
	lb[8] = -1.0;
	lb[9] = -1.0;
	lb[10] = -1.0;
	lb[11] = -1.0;
	lb[12] = -1.0;
	lb[13] = -1.0;
	lb[14] = -1.0;

	vector<double> ub(15);
	ub[0] = 1.0;
	ub[1] = 1.0;
	ub[2] = 1.0;
	ub[3] = 1.0;
	ub[4] = 10.0;
	ub[5] = 10.0;
	ub[6] = 10.0;
	ub[7] = 1.0;
	ub[8] = 1.0;
	ub[9] = 1.0;
	ub[10] = 1.0;
	ub[11] = 1.0;
	ub[12] = 1.0;
	ub[13] = 1.0;
	ub[14] = 1.0;

	vector<double> x(15);
	x[0] = frame_in.init_state(0,0);
	x[1] = frame_in.init_state(1,0);
	x[2] = frame_in.init_state(2,0);
	x[3] = frame_in.init_state(3,0);
	x[4] = frame_in.init_state(4,0);
	x[5] = frame_in.init_state(5,0);
	x[6] = frame_in.init_state(6,0);
	x[7] = frame_in.init_state(0,1);
	x[8] = frame_in.init_state(1,1);
	x[9] = frame_in.init_state(2,1);
	x[10] = frame_in.init_state(3,1);
	x[11] = frame_in.init_state(0,2);
	x[12] = frame_in.init_state(1,2);
	x[13] = frame_in.init_state(2,2);
	x[14] = frame_in.init_state(3,2);

	nlopt::opt opt_alg(nlopt::LN_COBYLA, 15);
	opt_alg.set_lower_bounds(lb);
	opt_alg.set_upper_bounds(ub);
	opt_alg.set_min_objective(cost_function, &cost_data);
	opt_alg.add_equality_constraint(quat_constraint_body, NULL, 1e-6);
	opt_alg.add_equality_constraint(quat_constraint_wing_L, NULL, 1e-6);
	opt_alg.add_equality_constraint(quat_constraint_wing_R, NULL, 1e-6);
	opt_alg.set_xtol_rel(1e-5);

	cout << " -------------------- " << endl;
	cout << "x start: " << x[0] << ", " << x[1] << ", " << x[2] << ", " << x[3] << ", " << x[4] 
		<< ", " << x[5] << ", " << x[6] << endl;
	cout << x[7] << ", " << x[8] << ", " << x[9] << ", " << x[10] << endl;
	cout << x[11] << ", " << x[12] << ", " << x[13] << ", " << x[14] << endl;

	double minf;

	nlopt::result result = opt_alg.optimize(x, minf);

	cout << "x opt: " << x[0] << ", " << x[1] << ", " << x[2] << ", " << x[3] << ", " << x[4] 
		<< ", " << x[5] << ", " << x[6] << endl;
	cout << x[7] << ", " << x[8] << ", " << x[9] << ", " << x[10] << endl;
	cout << x[11] << ", " << x[12] << ", " << x[13] << ", " << x[14] << endl;

	cout << "quaternion norm body: " << sqrt(pow(x[0],2)+pow(x[1],2)+pow(x[2],2)+pow(x[3],2)) << endl;
	cout << "quaternion norm wing L: " << sqrt(pow(x[7],2)+pow(x[8],2)+pow(x[9],2)+pow(x[10],2)) << endl;
	cout << "quaternion norm wing R: " << sqrt(pow(x[11],2)+pow(x[12],2)+pow(x[13],2)+pow(x[14],2)) << endl;
	cout << " -------------------- " << endl;
	cout << endl;
}

void ContourOpt::FindDestinationContour(FocalGrid &fg, frame_data &frame_in, ImagSegm &seg) {

	clock_t start;
	double duration;

	start = clock();

	body_dest_polygons.clear();
	wing_L_dest_polygons.clear();
	wing_R_dest_polygons.clear();
	body_dest_contours.clear();
	wing_L_dest_contours.clear();
	wing_R_dest_contours.clear();

	int N_cam = frame_in.N_cam;

	// Get the body and wing pointclouds
	arma::uvec body_pt_ids   = arma::find(frame_in.body_and_wing_pcls.row(3) == 1.0);
	arma::uvec wing_L_pt_ids = arma::find(frame_in.body_and_wing_pcls.row(3) == 2.0);
	arma::uvec wing_R_pt_ids = arma::find(frame_in.body_and_wing_pcls.row(3) == 3.0);

	arma::Mat<double> body_pcl = frame_in.body_and_wing_pcls.cols(body_pt_ids);
	arma::Mat<double> wing_L_pcl = frame_in.body_and_wing_pcls.cols(wing_L_pt_ids);
	arma::Mat<double> wing_R_pcl = frame_in.body_and_wing_pcls.cols(wing_R_pt_ids);

	//arma::Mat<double> means = ContourOpt::KMeans(body_pcl, 3, 10);

	//cout << means << endl;

	vector<arma::Col<int>> body_proj = fg.FocalGrid::ProjectCloud2Frames(body_pcl);
	vector<arma::Col<int>> wing_L_proj = fg.FocalGrid::ProjectCloud2Frames(wing_L_pcl);
	vector<arma::Col<int>> wing_R_proj = fg.FocalGrid::ProjectCloud2Frames(wing_R_pcl);

	vector<vector<arma::Mat<double>>> contour_vec = ContourOpt::FindContours(body_proj, wing_L_proj, wing_R_proj, frame_in, seg);

	for (int i=0; i<N_cam; i++) {
		body_dest_polygons.push_back(ContourOpt::Convert2PolygonWithHoles(contour_vec[i][0]));
		wing_L_dest_polygons.push_back(ContourOpt::Convert2PolygonWithHoles(contour_vec[i][1]));
		wing_R_dest_polygons.push_back(ContourOpt::Convert2PolygonWithHoles(contour_vec[i][2]));
		body_dest_contours.push_back(contour_vec[i][0]);
		wing_L_dest_contours.push_back(contour_vec[i][1]);
		wing_R_dest_contours.push_back(contour_vec[i][2]);
	}

	duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;
    cout<<"printf: "<< duration <<'\n';

}

arma::Mat<double> ContourOpt::KMeans(arma::Mat<double> &pcl_in, int N_clusters, int N_iter) {

	arma::Mat<double> data = pcl_in.rows(0,2);

	arma::Mat<double> means;

	bool status =arma::kmeans(means,data,N_clusters,arma::random_subset,N_iter,false);

	if (status==false) {
		cout << "clustering failed" << endl;
	}

	return means;
}

void ContourOpt::FindInitContour(FocalGrid &fg, frame_data &frame_in, ModelClass &mdl, ImagSegm &seg) {

	clock_t start;
	double duration;

	start = clock();

	body_init_contours.clear();
	wing_L_init_contours.clear();
	wing_R_init_contours.clear();

	int N_cam = frame_in.N_cam;

	// Get model pcl's in initial state:

	vector<arma::Mat<double>> M_init_vec = mdl.ReturnInitState(frame_in);

	for (int i=0; i<N_cam; i++) {
		vector<Polygon_2> m_silhouette = mdl.ReturnSilhouette(fg, M_init_vec,fg.CalculateViewVector(i),i);
		vector<Polygon_2> m_body;
		for (int j=0; j<3; j++) {
			m_body.push_back(m_silhouette[j]);
		}
		list<Polygon_with_holes_2> body_contour_poly = ContourOpt::CalcUnion(m_body);
		body_init_contours.push_back(ContourOpt::ConvertPolygonWithHoles2Mat(body_contour_poly));
		vector<Polygon_2> m_wing_L;
		m_wing_L.push_back(m_silhouette[3]);
		list<Polygon_with_holes_2> wing_L_contour_poly = ContourOpt::CalcUnion(m_wing_L);
		list<Polygon_with_holes_2> wing_L_complement = ContourOpt::CalcComplement(body_contour_poly, wing_L_contour_poly);
		wing_L_init_contours.push_back(ContourOpt::ConvertPolygonWithHoles2Mat(wing_L_complement));
		vector<Polygon_2> m_wing_R;
		m_wing_R.push_back(m_silhouette[4]);
		list<Polygon_with_holes_2> wing_R_contour_poly = ContourOpt::CalcUnion(m_wing_R);
		list<Polygon_with_holes_2> wing_R_complement = ContourOpt::CalcComplement(body_contour_poly, wing_R_contour_poly);
		wing_R_init_contours.push_back(ContourOpt::ConvertPolygonWithHoles2Mat(wing_R_complement));
	}

	duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;
    cout<<"printf: "<< duration <<'\n';
}

vector<vector<arma::Mat<double>>> ContourOpt::FindContours(vector<arma::Col<int>> &body_proj, vector<arma::Col<int>> &wing_L_proj, vector<arma::Col<int>> &wing_R_proj, frame_data &frame_in, ImagSegm &seg) {

	int N_cam = frame_in.N_cam;

	vector<vector<arma::Mat<double>>> contour_vec_out;

	vector<arma::Mat<double>> contour_vec_now;

	arma::Mat<double> contour_now;

	arma::Col<int> proj_body_img;
	arma::Col<int> proj_wing_L_img;
	arma::Col<int> proj_wing_R_img;

	for (int i=0; i<N_cam; i++) {

		contour_vec_now.clear();

		int N_row = get<0>(frame_in.image_size[i]);
		int N_col = get<1>(frame_in.image_size[i]);

		proj_body_img = body_proj[i];
		proj_wing_L_img = wing_L_proj[i]%(1-body_proj[i]);
		proj_wing_R_img = wing_R_proj[i]%(1-body_proj[i]);

		contour_vec_now.push_back(seg.FindContours(proj_body_img,N_row,N_col));
		contour_vec_now.push_back(seg.FindContours(proj_wing_L_img,N_row,N_col));
		contour_vec_now.push_back(seg.FindContours(proj_wing_R_img,N_row,N_col));

		contour_vec_out.push_back(contour_vec_now);
	}
	return contour_vec_out;
}

arma::Mat<double> ContourOpt::CalculateContour(arma::Mat<double> &pts_in, double d_u, double d_v) {

	int N_pts = pts_in.n_cols;

	// find starting point (minimum u index):

	arma::Col<double> uv_start = pts_in.col(arma::index_min(pts_in.row(0)));

	arma::Col<double> uv_now = uv_start;

	arma::Col<double> uv_prev = uv_start;

	arma::uvec inds_u_plus;
	arma::uvec inds_u_min;
	arma::uvec inds_v_plus;
	arma::uvec inds_v_min;

	arma::uvec inds_neighbors;

	double theta_now;

	double theta_min;

	int theta_min_ind;

	arma::Mat<double> contour_pts;

	contour_pts = uv_start;

	int iter = 0;

	int N_contour_pts = 0;

	bool contour_closed = false;

	while (iter<N_pts && contour_closed==false) {

		inds_u_plus = arma::find(pts_in.row(0)>(uv_now(0)-d_u));
		inds_u_min = arma::find(pts_in.row(0)<(uv_now(0)+d_u));
		inds_v_plus = arma::find(pts_in.row(1)>(uv_now(1)-d_v));
		inds_v_min = arma::find(pts_in.row(1)<(uv_now(1)+d_v));

		inds_neighbors = arma::intersect(arma::intersect(inds_u_plus,inds_u_min),arma::intersect(inds_v_plus,inds_v_min));

		if (inds_neighbors.n_rows>0) {

			theta_min = 2*PI;
			theta_min_ind = -1;

			for (int i=0; i<inds_neighbors.n_rows; i++) {
				theta_now = atan2(pts_in(1,inds_neighbors(i))-uv_prev(1),pts_in(0,inds_neighbors(i))-uv_prev(0));
				if (isnan(theta_now)==0) {
					if (theta_now<theta_min) {
						theta_min = theta_now;
						theta_min_ind = i;
					}
				}
			}

			uv_prev = uv_now;

			uv_now = pts_in.col(inds_neighbors(theta_min_ind));

			contour_pts.insert_cols(N_contour_pts,uv_now);
			N_contour_pts++;
			if (uv_now(0)==uv_start(0) && uv_now(1)==uv_start(1)) {
				contour_closed = true;
			}
		}
		else {
			iter = N_pts;
			cout << "could not close contour" << endl;
		}
		iter++;
	}

	return contour_pts;
}

list<Polygon_with_holes_2> ContourOpt::CalcUnion(vector<Polygon_2> &polygons_in) {

	int N_c = polygons_in.size();

	Polygon_set_2 S;

	for (int i=0; i<N_c; i++) {
		if (i==0) {
			S.insert(polygons_in[i]);
		}
		else {
			S.join(polygons_in[i]);
		}
	}

	//S.union()

	list<Polygon_with_holes_2> res;
	S.polygons_with_holes (back_inserter (res));

	return res;
}

list<Polygon_with_holes_2> ContourOpt::CalcComplement(list<Polygon_with_holes_2> &pol_A, list<Polygon_with_holes_2> &pol_B) {

	Polygon_set_2 S;

	int iter_1 =0;
	for (list<Polygon_with_holes_2>::iterator it_1=pol_B.begin(); it_1 !=pol_B.end(); ++it_1) {
		//cout << "pol B area " << it_1->outer_boundary().area() << endl;
		if (iter_1==0) {
			S.insert(*it_1);
		}
		else {
			S.join(*it_1);
		}
		iter_1++;
	}
	
	int iter_2 = 0;
	for (list<Polygon_with_holes_2>::iterator it_2=pol_A.begin(); it_2 !=pol_A.end(); ++it_2) {
		//cout << "pol A area " << it_2->outer_boundary().area() << endl;
		S.difference(*it_2);
	}

	list<Polygon_with_holes_2> res;
	S.polygons_with_holes (back_inserter (res));

	return res;
}

list<Polygon_with_holes_2> ContourOpt::CalcSymmetricDifference(list<Polygon_with_holes_2> &pol_A, list<Polygon_with_holes_2> &pol_B) {

	/*
	Polygon_set_2 S;

	int iter_1 =0;
	for (list<Polygon_with_holes_2>::iterator it_1=pol_B.begin(); it_1 !=pol_B.end(); ++it_1) {
		if (iter_1==0) {
			S.insert(*it_1);
		}
		else {
			S.join(*it_1);
		}
		iter_1++;
	}
	
	int iter_2 = 0;
	for (list<Polygon_with_holes_2>::iterator it_2=pol_A.begin(); it_2 !=pol_A.end(); ++it_2) {
		S.symmetric_difference(*it_2);
	}

	list<Polygon_with_holes_2> res;
	S.polygons_with_holes (back_inserter (res));
	*/
	//cout << "caclulating symmetric difference" << endl;
	list<Polygon_with_holes_2> res_out;

	//if (!pol_A.empty() && !pol_B.empty()){
  	for (list<Polygon_with_holes_2>::iterator it_1=pol_B.begin(); it_1 !=pol_B.end(); ++it_1) {
  		for (list<Polygon_with_holes_2>::iterator it_2=pol_A.begin(); it_2 !=pol_A.end(); ++it_2) {
  			//if (CGAL::do_intersect(*it_1, *it_2)) {
  			//cout << "do intersect " << CGAL::do_intersect(*it_1, *it_2) << endl;
			//cout << "area pol_A " << it_2->outer_boundary().area() << endl;
			//cout << "area pol_B " << it_1->outer_boundary().area() << endl;
			CGAL::symmetric_difference (*it_1, *it_2, back_inserter(res_out));
			//}
			//else {
			//	cout << "do not intersect " << endl;
			//}
		}
	}
	//}

	cout << "Res size: " << res_out.size() << endl;

	return res_out;
}

arma::Mat<double> ContourOpt::ConvertPolygonWithHoles2Mat(list<Polygon_with_holes_2> &Pol_list_in) {

	int N_poly = Pol_list_in.size();
	int N_v;
	arma::Mat<double> mat_out;

	if (N_poly>0) {
		int iter = 0;
		vector<double> output_iterator;
		for (list<Polygon_with_holes_2>::iterator it=Pol_list_in.begin(); it !=Pol_list_in.end(); ++it) {
			//cout << it->is_unbounded() << endl;
			N_v = it->outer_boundary().size();
			arma::Mat<double> curr_pol_mat(3,N_v);
			for (int j=0; j<N_v; j++) {
				curr_pol_mat(0,j) = static_cast<double>(it->outer_boundary().vertex(j).x());
				curr_pol_mat(1,j) = static_cast<double>(it->outer_boundary().vertex(j).y());
				curr_pol_mat(2,j) = iter+1;
			}
			if (iter==0) {
				mat_out = curr_pol_mat;
			}
			else {
				mat_out = arma::join_rows(mat_out, curr_pol_mat);
			}
			iter++;
		}
	}
	else {
		mat_out.zeros(3,1);
	}

	return mat_out;
}

arma::Mat<double> ContourOpt::ConvertPolygon2Mat(Polygon_2 &Pol_in) {

	int N_v = Pol_in.size();

	arma::Mat<double> mat_out;
	mat_out.zeros(3,N_v);

	//int k=0;

	//for (VertexIterator vi = Pol_in.vertices_begin(); vi != Pol_in.vertices_end(); ++vi) {
	for (int i=0; i<N_v; i++) {

		mat_out(0,i) = static_cast<double>(Pol_in.vertex(i).x());
		mat_out(1,i) = static_cast<double>(Pol_in.vertex(i).y());
		mat_out(2,i) = 1.0;

		//k++;
	}

	return mat_out;
}

list<Polygon_with_holes_2> ContourOpt::Convert2PolygonWithHoles(arma::Mat<double> &contour_mat) {

	int N_contours = contour_mat.row(2).max();

	arma::uvec temp_inds;
	arma::Mat<double> temp_mat;

	list<Polygon_with_holes_2> poly_list_out;

	for (int i=0; i<N_contours; i++) {
		temp_inds = arma::find(contour_mat.row(2)==i+1);
		temp_mat = contour_mat.cols(temp_inds);
		Polygon_with_holes_2 t_poly(ContourOpt::Convert2Polygon(temp_mat));
		poly_list_out.push_back(t_poly);
	}

	return poly_list_out;
}

Polygon_2 ContourOpt::Convert2Polygon(arma::Mat<double> &contour_mat) {

	// Run through the points and construct a Polygon_2 object:
	Polygon_2 P;

	int N_pts = contour_mat.n_cols;

	//for (int i=0; i<N_pts; i++) {
	for (int i=(N_pts-1); i>=0; i--) {
		P.push_back(Point_2 (contour_mat(0,i),contour_mat(1,i)));
	}

	return P;
}

p::list ContourOpt::ReturnInitContour(int cam_nr) {

	p::list init_contour_list;

	arma::Mat<double> body_c = body_init_contours[cam_nr];
	arma::Mat<double> wing_L_c = wing_L_init_contours[cam_nr];
	arma::Mat<double> wing_R_c = wing_R_init_contours[cam_nr];

	arma::uvec found_ids;
	int N_found;

	for (int i=0; i<body_c.row(2).max(); i++) {
		found_ids = arma::find(body_c.row(2)==i+1);
		N_found = found_ids.n_rows;
		if (N_found>1) {
			p::tuple shape = p::make_tuple(3,N_found);
			np::dtype dtype = np::dtype::get_builtin<double>();
			np::ndarray c_array = np::zeros(shape,dtype);
			for (int j=0; j<N_found; j++) {
				c_array[0][j] = body_c(0,found_ids(j));
				c_array[1][j] = body_c(1,found_ids(j));
				c_array[2][j] = 1;
			}
			init_contour_list.append(c_array);
		}
	}

	for (int i=0; i<wing_L_c.row(2).max(); i++) {
		found_ids = arma::find(wing_L_c.row(2)==i+1);
		N_found = found_ids.n_rows;
		if (N_found>1) {
			p::tuple shape = p::make_tuple(3,N_found);
			np::dtype dtype = np::dtype::get_builtin<double>();
			np::ndarray c_array = np::zeros(shape,dtype);
			for (int j=0; j<N_found; j++) {
				c_array[0][j] = wing_L_c(0,found_ids(j));
				c_array[1][j] = wing_L_c(1,found_ids(j));
				c_array[2][j] = 2;
			}
			init_contour_list.append(c_array);
		}
	}

	for (int i=0; i<wing_R_c.row(2).max(); i++) {
		found_ids = arma::find(wing_R_c.row(2)==i+1);
		N_found = found_ids.n_rows;
		if (N_found>1) {
			p::tuple shape = p::make_tuple(3,N_found);
			np::dtype dtype = np::dtype::get_builtin<double>();
			np::ndarray c_array = np::zeros(shape,dtype);
			for (int j=0; j<N_found; j++) {
				c_array[0][j] = wing_R_c(0,found_ids(j));
				c_array[1][j] = wing_R_c(1,found_ids(j));
				c_array[2][j] = 3;
			}
			init_contour_list.append(c_array);
		}
	}

	return init_contour_list;
}

p::list ContourOpt::ReturnDestContour(int cam_nr) {

	p::list dest_contour_list;

	arma::Mat<double> body_c = body_dest_contours[cam_nr];
	arma::Mat<double> wing_L_c = wing_L_dest_contours[cam_nr];
	arma::Mat<double> wing_R_c = wing_R_dest_contours[cam_nr];

	arma::uvec found_ids;
	int N_found;

	for (int i=0; i<body_c.row(2).max(); i++) {
		found_ids = arma::find(body_c.row(2)==i+1);
		N_found = found_ids.n_rows;
		if (N_found>1) {
			p::tuple shape = p::make_tuple(3,N_found);
			np::dtype dtype = np::dtype::get_builtin<double>();
			np::ndarray c_array = np::zeros(shape,dtype);
			for (int j=0; j<N_found; j++) {
				c_array[0][j] = body_c(0,found_ids(j));
				c_array[1][j] = body_c(1,found_ids(j));
				c_array[2][j] = 1;
			}
			dest_contour_list.append(c_array);
		}
	}

	for (int i=0; i<wing_L_c.row(2).max(); i++) {
		found_ids = arma::find(wing_L_c.row(2)==i+1);
		N_found = found_ids.n_rows;
		if (N_found>1) {
			p::tuple shape = p::make_tuple(3,N_found);
			np::dtype dtype = np::dtype::get_builtin<double>();
			np::ndarray c_array = np::zeros(shape,dtype);
			for (int j=0; j<N_found; j++) {
				c_array[0][j] = wing_L_c(0,found_ids(j));
				c_array[1][j] = wing_L_c(1,found_ids(j));
				c_array[2][j] = 2;
			}
			dest_contour_list.append(c_array);
		}
	}

	for (int i=0; i<wing_R_c.row(2).max(); i++) {
		found_ids = arma::find(wing_R_c.row(2)==i+1);
		N_found = found_ids.n_rows;
		if (N_found>1) {
			p::tuple shape = p::make_tuple(3,N_found);
			np::dtype dtype = np::dtype::get_builtin<double>();
			np::ndarray c_array = np::zeros(shape,dtype);
			for (int j=0; j<N_found; j++) {
				c_array[0][j] = wing_R_c(0,found_ids(j));
				c_array[1][j] = wing_R_c(1,found_ids(j));
				c_array[2][j] = 3;
			}
			dest_contour_list.append(c_array);
		}
	}

	return dest_contour_list;
}