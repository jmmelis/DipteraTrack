#ifndef MODEL_CLASS_H
#define MODEL_CLASS_H

#include "session_data.h"
#include "frame_data.h"
#include "focal_grid.h"

#include <string>
#include <stdint.h>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <dirent.h>
#include <armadillo>
#include "json.hpp"

#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyDataSilhouette.h>
#include <vtkSmartPointer.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkSTLReader.h>
#include <vtkCleanPolyData.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Boolean_set_operations_2.h>
#include <CGAL/algorithm.h>
#include <CGAL/assertions.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Polygon_with_holes_2.h>
#include <CGAL/Polygon_set_2.h>


typedef CGAL::Exact_predicates_inexact_constructions_kernel KI;
//typedef CGAL::Exact_predicates_exact_constructions_kernel KE;

typedef KI::Point_2 				   Point_2;
typedef CGAL::Polygon_2<KI>            Polygon_2;
typedef CGAL::Polygon_with_holes_2<KI> Polygon_with_holes_2;
typedef CGAL::Polygon_set_2<KI> 	   Polygon_set_2;

typedef Polygon_2::Vertex_iterator VertexIterator;
typedef Polygon_2::Edge_const_iterator EdgeIterator;

using namespace std;

class ModelClass
{
	
	public:

		// Class

		ModelClass();

		// Parameters
		string model_name;
		int N_components;
		int N_joints;
		vector<string> stl_list;
		vector<vector<int>> parent_list;
		vector<int> joint_type_list;
		vector<vector<double>> joint_param_parent;
		vector<vector<double>> joint_param_child;

		vector<string> drag_point_labels;
		vector<string> drag_point_symbols;
		vector<vector<int>> drag_point_colors;
		vector<vector<double>> drag_point_start_pos;
		vector<vector<int>> drag_line_connectivity;
		vector<vector<int>> drag_line_colors;
		vector<string> scale_texts;
		vector<vector<int>> scale_calc;
		vector<vector<int>> length_calc;
		vector<int> origin_ind;
		vector<int> contour_calc;
		vector<int> state_calc;

		double srf_angle;
		double stroke_bound;
		double deviation_bound;
		double wing_pitch_bound;
		vector<vtkSmartPointer<vtkPolyData>> model_polydata;
		vector<double> model_scale;
		vector<arma::Mat<double>> model_pcls;

		// Functions
		bool LoadModel(session_data &session);
		vector<arma::Mat<double>> ReturnStartState();
		vector<arma::Mat<double>> ReturnInitState(frame_data &frame_in);
		vector<double> ReturnSRCState(vector<double> state_in);
		vector<Polygon_2> ReturnSilhouette(FocalGrid &fg, vector<arma::Mat<double>> M_state, arma::Col<double> view_vec, int cam_nr);
		vector<Polygon_2> ReturnSilhouette2(FocalGrid &fg, vector<double> state_vec, int cam_nr);
		void SetStrokeBounds(double StrokeBound);
		void SetDeviationBounds(double DeviationBound);
		void SetWingPitchBounds(double WingPitchBound);
		bool SetModel(session_data &session);
		bool LoadModelMesh(session_data &session);
		vtkSmartPointer<vtkPolyData> LoadSTLFile(string FileLoc, string FileName);
		void SetScale(vector<double> scale_in);
		vector<double> GetModelScale();
		vector<arma::Mat<double>> ReturnModelPCL(vector<arma::Mat<double>> M_in);
		vector<arma::Mat<double>> ReturnInitPCL(vector<arma::Mat<double>> M_in);
		bool ScaleModelPCL();
		vtkSmartPointer<vtkPolyData> TransformPolyData(vtkSmartPointer<vtkPolyData> polyData, arma::Mat<double> M_in, double scale_in);
		Polygon_2 GetSilhouette(FocalGrid &fg, vtkSmartPointer<vtkPolyData> polyData, double view_vector[3], int cam_nr);
		arma::Mat<double> GetM(arma::Col<double> state_in);
		arma::Col<double> GetStateFromM(arma::Mat<double> M_in);
		arma::Mat<double> MultiplyM(arma::Mat<double> MA, arma::Mat<double> MB);
		arma::Mat<double> TransposeM(arma::Mat<double> M_in);
		vtkSmartPointer<vtkPolyData> stlReader(string FileLoc, string FileName);
		void TestPointInsidePolygonSpeed(arma::Mat<double> &pcl_in);
		void Test3DIntersection();
		void TestPoissonSurfaceReconstruction(arma::Mat<double> &pcl_in);
		vector<string> GetPointLabels();
		vector<string> GetPointSymbols();
		vector<vector<int>> GetPointColors();
		vector<vector<double>> GetPointStartPos();
		vector<vector<int>> GetLineConnectivity();
		vector<vector<int>> GetLineColors();
		vector<string> GetScaleTexts();
		vector<vector<int>> GetScaleCalc();
		vector<vector<int>> GetLengthCalc();
		vector<int> GetContourCalc();
		vector<int> GetOriginInd();

};
#endif