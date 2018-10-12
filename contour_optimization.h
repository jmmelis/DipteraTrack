#ifndef CONTOUR_OPT_H
#define CONTOUR_OPT_H

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
#include <armadillo>

#include <nlopt.hpp>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Boolean_set_operations_2.h>

using namespace std;

namespace p = boost::python;
namespace np = boost::python::numpy;

typedef CGAL::Exact_predicates_inexact_constructions_kernel KI;

typedef KI::Point_2 Point_2;
typedef CGAL::Polygon_2<KI> Polygon_2;
typedef CGAL::Polygon_with_holes_2<KI> Polygon_with_holes_2;

class ContourOpt
{
	
	public:

		// Class
		ContourOpt();

		// Parameters
		double alpha;

		bool output_c_on_off;

		vector<arma::Mat<double>> body_dest_contours;
		vector<arma::Mat<double>> wing_L_dest_contours;
		vector<arma::Mat<double>> wing_R_dest_contours;
		vector<arma::Mat<double>> body_init_contours;
		vector<arma::Mat<double>> wing_L_init_contours;
		vector<arma::Mat<double>> wing_R_init_contours;
		vector<arma::Mat<double>> body_src_contours;
		vector<arma::Mat<double>> wing_L_src_contours;
		vector<arma::Mat<double>> wing_R_src_contours;

		vector<list<Polygon_with_holes_2>> body_dest_polygons;
		vector<list<Polygon_with_holes_2>> wing_L_dest_polygons;
		vector<list<Polygon_with_holes_2>> wing_R_dest_polygons;
		vector<list<Polygon_with_holes_2>> body_init_polygons;
		vector<list<Polygon_with_holes_2>> wing_L_init_polygons;
		vector<list<Polygon_with_holes_2>> wing_R_init_polygons;
		vector<list<Polygon_with_holes_2>> body_src_polygons;
		vector<list<Polygon_with_holes_2>> wing_L_src_polygons;
		vector<list<Polygon_with_holes_2>> wing_R_src_polygons;

		// Functions
		void OptimizeState(FocalGrid &fg, frame_data &frame_in, ModelClass &mdl, ImagSegm &seg);
		void Optimize(FocalGrid &fg, frame_data &frame_in, ModelClass &mdl);
		void FindDestinationContour(FocalGrid &fg, frame_data &frame_in, ImagSegm &seg);
		arma::Mat<double> KMeans(arma::Mat<double> &pcl_in, int N_clusters, int N_iter);
		void FindInitContour(FocalGrid &fg, frame_data &frame_in, ModelClass &mdl, ImagSegm &seg);
		vector<vector<arma::Mat<double>>> FindContours(vector<arma::Col<int>> &body_proj, vector<arma::Col<int>> &wing_L_proj, vector<arma::Col<int>> &wing_R_proj, frame_data &frame_in, ImagSegm &seg);
		arma::Mat<double> CalculateContour(arma::Mat<double> &pts_in, double d_u, double d_v);
		list<Polygon_with_holes_2> CalcUnion(vector<Polygon_2> &polygons_in);
		arma::Mat<double> ConvertPolygon2Mat(Polygon_2 &Pol_in);
		arma::Mat<double> ConvertPolygonWithHoles2Mat(list<Polygon_with_holes_2> &Pol_list_in);
		list<Polygon_with_holes_2> Convert2PolygonWithHoles(arma::Mat<double> &contour_mat);
		Polygon_2 Convert2Polygon(arma::Mat<double> &contour_mat);
		list<Polygon_with_holes_2> CalcComplement(list<Polygon_with_holes_2> &pol_A, list<Polygon_with_holes_2> &pol_B);
		list<Polygon_with_holes_2> CalcSymmetricDifference(list<Polygon_with_holes_2> &pol_A, list<Polygon_with_holes_2> &pol_B);
		
		p::list ReturnDestContour(int cam_nr);
		p::list ReturnInitContour(int cam_nr);

};
#endif
