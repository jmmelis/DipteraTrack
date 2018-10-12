#include "model_class.h"

#include "frame_data.h"
#include "focal_grid.h"

#include <string>
#include <stdint.h>
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <list>
#include <ctime>
#include <algorithm>
#include <dirent.h>
#include <armadillo>
#include "json.hpp"

#include <vtkPolyData.h>
#include <vtkSTLReader.h>
#include <vtkSmartPointer.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyDataSilhouette.h>
#include <vtkMatrix4x4.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkPoints.h>
#include <vtkLine.h>
#include <vtkCellArray.h>
#include <vtkExtractEdges.h>
#include <vtkCleanPolyData.h>
#include <vtkPolyDataConnectivityFilter.h>

#include <vtkPointData.h>
#include <vtkSelectEnclosedPoints.h>
#include <vtkIntArray.h>
#include <vtkDataArray.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkProperty.h>

#include <vtkSurfaceReconstructionFilter.h>
#include <vtkProgrammableSource.h>
#include <vtkContourFilter.h>
#include <vtkReverseSense.h>

#include <vtkBooleanOperationPolyDataFilter.h>

#include <vtkCenterOfMass.h>

#include <CGAL/basic.h>
//#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Boolean_set_operations_2.h>
#include <CGAL/algorithm.h>
#include <CGAL/assertions.h>
#include <CGAL/Line_2.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Polygon_with_holes_2.h>
#include <CGAL/Polygon_set_2.h>
#include <CGAL/connect_holes.h>

#include <CGAL/Cartesian.h>
#include <CGAL/MP_Float.h>
#include <CGAL/Quotient.h>
#include <CGAL/Arr_segment_traits_2.h>
#include <CGAL/Arrangement_2.h>
#include <CGAL/Arr_walk_along_line_point_location.h>
//#include "arr_print.h"

#include <boost/math/special_functions/factorials.hpp>

typedef CGAL::Exact_predicates_inexact_constructions_kernel KI;
typedef CGAL::Exact_predicates_exact_constructions_kernel KE;

typedef KI::Point_2 				   Point_2;
typedef KI::Line_2					   Line_2;
typedef CGAL::Polygon_2<KI>            Polygon_2;
typedef CGAL::Polygon_with_holes_2<KI> Polygon_with_holes_2;
typedef CGAL::Polygon_set_2<KI> 	   Polygon_set_2;

typedef Polygon_2::Vertex_iterator 			VertexIterator;
typedef Polygon_2::Edge_const_iterator 		EdgeIterator;
typedef Polygon_2::Edge_const_circulator 	EdgeCirculator;

typedef CGAL::Quotient<int>                                     Number_type;
typedef CGAL::Cartesian<Number_type>                            Kernel;
typedef CGAL::Arr_segment_traits_2<KE>                      	Traits_2;
typedef Traits_2::Point_2                                       Point_arr_2;
typedef Traits_2::X_monotone_curve_2                            Segment_2;
typedef CGAL::Arrangement_2<Traits_2>                           Arrangement_2;
typedef CGAL::Arr_walk_along_line_point_location<Arrangement_2> Walk_pl;
typedef CGAL::Arr_segment_traits_2<KE> 							Segment_traits_2;



#define PI 3.14159265

using namespace std;

using json = nlohmann::json;

ModelClass::ModelClass() {
	// empty
}

bool ModelClass::LoadModel(session_data &session) {
	bool model_loaded = true;
	try {

		string mdl_name = session.model_name;
		size_t dot_pos = mdl_name.find(".");

		cout << mdl_name.erase(dot_pos,5) << endl;
		
		chdir(session.model_loc.c_str());

		// Load JSON file:
		ifstream in_file(session.model_name.c_str());
		json loaded_file;
		in_file >> loaded_file;

		// Set parameters:
		model_name = mdl_name.erase(dot_pos,5);

		N_components = loaded_file["N_components"];
		N_joints = loaded_file["N_joints"];

		stl_list.clear();
		parent_list.clear();

		for (int i=0; i<N_components; i++) {
			stl_list.push_back(loaded_file["stl_list"][i]);
			int N_temp = loaded_file["parent_list"][i].size();
			vector<int> kin_tree;
			for (int k=0; k<N_temp; k++) {
				kin_tree.push_back(loaded_file["parent_list"][i][k]);
			}
			parent_list.push_back(kin_tree);
		}

		joint_type_list.clear();
		joint_param_parent.clear();
		joint_param_child.clear();

		for (int j=0; j<N_joints; j++) {
			joint_type_list.push_back(loaded_file["joint_type_list"][j]);
			int N_temp = loaded_file["joint_param_parent"][j].size();
			vector<double> parent_param;
			vector<double> child_param;
			for (int k=0; k<N_temp; k++) {
				parent_param.push_back(loaded_file["joint_param_parent"][j][k]);
				child_param.push_back(loaded_file["joint_param_child"][j][k]);
			}
			joint_param_parent.push_back(parent_param);
			joint_param_child.push_back(child_param);
		}

		
		drag_point_labels.clear();
		drag_point_symbols.clear();
		drag_point_colors.clear();
		drag_point_start_pos.clear();

		int N_drag_pts = loaded_file["drag_point_labels"].size();

		for (int i=0; i<N_drag_pts; i++) {
			drag_point_labels.push_back(loaded_file["drag_point_labels"][i]);
			drag_point_symbols.push_back(loaded_file["drag_point_symbols"][i]);
			vector<int> pt_color;
			vector<double> start_pos;
			for (int k=0; k<3; k++) {
				pt_color.push_back(loaded_file["drag_point_colors"][i][k]);
				start_pos.push_back(loaded_file["drag_point_start_pos"][i][k]);
			}
			drag_point_colors.push_back(pt_color);
			drag_point_start_pos.push_back(start_pos);
		}


		drag_line_connectivity.clear();
		drag_line_colors.clear();
		scale_calc.clear();

		int N_drag_lns = loaded_file["drag_line_connectivity"].size();

		for (int j=0; j<N_drag_lns; j++) {
			vector<int> connectivity;
			vector<int> line_color;
			for (int k=0; k<5; k++) {
				if (k<2) {
					connectivity.push_back(loaded_file["drag_line_connectivity"][j][k]);
				}
				line_color.push_back(loaded_file["drag_line_colors"][j][k]);
			}
			drag_line_connectivity.push_back(connectivity);
			drag_line_colors.push_back(line_color);
		}

		scale_texts.clear();
		scale_calc.clear();

		for (int i=0; i<loaded_file["scale_calc"].size(); i++) {
			scale_texts.push_back(loaded_file["scale_texts"][i]);
			scale_calc.push_back(loaded_file["scale_calc"][i]);
		}

		length_calc.clear();

		for (int j=0; j<loaded_file["length_calc"].size(); j++) {
			length_calc.push_back(loaded_file["length_calc"][j]);
		}

		origin_ind.clear();

		for (int i=0; i<loaded_file["origin_indices"].size(); i++) {
			origin_ind.push_back(loaded_file["origin_indices"][i]);
		}

		contour_calc.clear();

		for (int j=0; j<loaded_file["contour_calc"].size(); j++) {
			contour_calc.push_back(loaded_file["contour_calc"][j]);
		}

		srf_angle = loaded_file["srf_angle"];

		state_calc.clear();

		for (int i=0; i<loaded_file["state_calc"].size(); i++) {
			state_calc.push_back(loaded_file["state_calc"][i]);
		}

	}
	catch(...) {
		model_loaded = false;
	}
	return model_loaded;
}

vector<arma::Mat<double>> ModelClass::ReturnStartState() {
	int N_parents;
	double scale_now;
	double scale_parent;
	double scale_child;
	arma::Col<double> state_parent(7);
	arma::Col<double> state_child(7);
	arma::Mat<double> M_parent(4,4);
	arma::Mat<double> M_child(4,4);
	arma::Mat<double> M_out;
	vector<arma::Mat<double>> M_start;
	for (int i=0; i<N_components; i++) {
		N_parents = parent_list[i].size();
		M_out.eye(4,4);
		for (int j=0; j<N_parents; j++) {
			if (j>0) {
				scale_parent = model_scale[parent_list[i][j-1]];
				state_parent(0) = joint_param_parent[parent_list[i][j]][0];
				state_parent(1) = joint_param_parent[parent_list[i][j]][1];
				state_parent(2) = joint_param_parent[parent_list[i][j]][2];
				state_parent(3) = joint_param_parent[parent_list[i][j]][3];
				state_parent(4) = joint_param_parent[parent_list[i][j]][4]*scale_parent;
				state_parent(5) = joint_param_parent[parent_list[i][j]][5]*scale_parent;
				state_parent(6) = joint_param_parent[parent_list[i][j]][6]*scale_parent;
				M_parent = ModelClass::GetM(state_parent);
				scale_child = model_scale[parent_list[i][j]];
				state_child(0) = joint_param_child[parent_list[i][j]][0];
				state_child(1) = joint_param_child[parent_list[i][j]][1];
				state_child(2) = joint_param_child[parent_list[i][j]][2];
				state_child(3) = joint_param_child[parent_list[i][j]][3];
				state_child(4) = joint_param_child[parent_list[i][j]][4]*scale_child;
				state_child(5) = joint_param_child[parent_list[i][j]][5]*scale_child;
				state_child(6) = joint_param_child[parent_list[i][j]][6]*scale_child;
				M_child = ModelClass::GetM(state_child);
				M_out = ModelClass::MultiplyM(M_out,ModelClass::MultiplyM(M_parent,M_child));
			}
			else {
				//scale_now = model_scale[parent_list[i][j]];
				scale_parent = 1.0;
				state_parent(0) = joint_param_parent[parent_list[i][j]][0];
				state_parent(1) = joint_param_parent[parent_list[i][j]][1];
				state_parent(2) = joint_param_parent[parent_list[i][j]][2];
				state_parent(3) = joint_param_parent[parent_list[i][j]][3];
				state_parent(4) = joint_param_parent[parent_list[i][j]][4]*scale_parent;
				state_parent(5) = joint_param_parent[parent_list[i][j]][5]*scale_parent;
				state_parent(6) = joint_param_parent[parent_list[i][j]][6]*scale_parent;
				M_parent = ModelClass::GetM(state_parent);
				scale_child = model_scale[parent_list[i][j]];
				state_child(0) = joint_param_child[parent_list[i][j]][0];
				state_child(1) = joint_param_child[parent_list[i][j]][1];
				state_child(2) = joint_param_child[parent_list[i][j]][2];
				state_child(3) = joint_param_child[parent_list[i][j]][3];
				state_child(4) = joint_param_child[parent_list[i][j]][4]*scale_child;
				state_child(5) = joint_param_child[parent_list[i][j]][5]*scale_child;
				state_child(6) = joint_param_child[parent_list[i][j]][6]*scale_child;
				M_child = ModelClass::GetM(state_child);
				M_out = ModelClass::MultiplyM(M_out,ModelClass::MultiplyM(M_parent,M_child));
			}
		}
		M_start.push_back(M_out);
	}
	return M_start;
}

vector<double> ModelClass::ReturnSRCState(vector<double> state_in) {

	int N_parents;
	double scale_now;
	double scale_parent;
	double scale_child;
	vector<double> state_out;

	arma::Mat<double> X_mat;
	X_mat.zeros(7,5);
	
	X_mat(0,0) = state_in[0];
	X_mat(1,0) = state_in[1];
	X_mat(2,0) = state_in[2];
	X_mat(3,0) = state_in[3];
	X_mat(4,0) = state_in[4];
	X_mat(5,0) = state_in[5];
	X_mat(6,0) = state_in[6];
	X_mat(0,3) = state_in[7];
	X_mat(1,3) = state_in[8];
	X_mat(2,3) = state_in[9];
	X_mat(3,3) = state_in[10];
	X_mat(0,4) = state_in[11];
	X_mat(1,4) = state_in[12];
	X_mat(2,4) = state_in[13];
	X_mat(3,4) = state_in[14];

	arma::Col<double> state_now;
	state_now.zeros(7);
	arma::Col<double> state_parent;
	state_parent.zeros(7);
	arma::Col<double> state_child;
	state_child.zeros(7);
	arma::Mat<double> M_parent(4,4);
	arma::Mat<double> M_child(4,4);
	arma::Mat<double> M_state(4,4);
	arma::Mat<double> M_out;
	for (int i=0; i<N_components; i++) {
		N_parents = parent_list[i].size();
		M_out.eye(4,4);
		for (int j=0; j<N_parents; j++) {
			if (j>0) {
				if (state_calc[parent_list[i][j]] >= 0) {
					state_now = X_mat.col(parent_list[i][j]);
				}
				else {
					state_now = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
				}
				M_state = ModelClass::GetM(state_now);
				scale_parent = model_scale[parent_list[i][j-1]];
				state_parent(0) = joint_param_parent[parent_list[i][j]][0];
				state_parent(1) = joint_param_parent[parent_list[i][j]][1];
				state_parent(2) = joint_param_parent[parent_list[i][j]][2];
				state_parent(3) = joint_param_parent[parent_list[i][j]][3];
				state_parent(4) = joint_param_parent[parent_list[i][j]][4]*scale_parent;
				state_parent(5) = joint_param_parent[parent_list[i][j]][5]*scale_parent;
				state_parent(6) = joint_param_parent[parent_list[i][j]][6]*scale_parent;
				M_parent = ModelClass::GetM(state_parent);
				scale_child = model_scale[parent_list[i][j]];
				state_child(0) = joint_param_child[parent_list[i][j]][0];
				state_child(1) = joint_param_child[parent_list[i][j]][1];
				state_child(2) = joint_param_child[parent_list[i][j]][2];
				state_child(3) = joint_param_child[parent_list[i][j]][3];
				state_child(4) = joint_param_child[parent_list[i][j]][4]*scale_child;
				state_child(5) = joint_param_child[parent_list[i][j]][5]*scale_child;
				state_child(6) = joint_param_child[parent_list[i][j]][6]*scale_child;
				M_child = ModelClass::GetM(state_child);
				M_out = ModelClass::MultiplyM(M_out,ModelClass::MultiplyM(M_parent,ModelClass::MultiplyM(M_state,M_child)));
			}
			else {
				if (state_calc[parent_list[i][j]] >= 0) {
					state_now = X_mat.col(parent_list[i][j]);
				}
				else {
					state_now = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
				}
				M_state = ModelClass::GetM(state_now);
				scale_child = model_scale[parent_list[i][j]];
				state_child(0) = joint_param_child[parent_list[i][j]][0];
				state_child(1) = joint_param_child[parent_list[i][j]][1];
				state_child(2) = joint_param_child[parent_list[i][j]][2];
				state_child(3) = joint_param_child[parent_list[i][j]][3];
				state_child(4) = joint_param_child[parent_list[i][j]][4]*scale_child;
				state_child(5) = joint_param_child[parent_list[i][j]][5]*scale_child;
				state_child(6) = joint_param_child[parent_list[i][j]][6]*scale_child;
				M_child = ModelClass::GetM(state_child);
				M_out = ModelClass::MultiplyM(M_out,ModelClass::MultiplyM(M_state,M_child));
			}
		}
		arma::Col<double> state_vec =  ModelClass::GetStateFromM(M_out);
		state_out.push_back(state_vec(0));
		state_out.push_back(state_vec(1));
		state_out.push_back(state_vec(2));
		state_out.push_back(state_vec(3));
		state_out.push_back(state_vec(4));
		state_out.push_back(state_vec(5));
		state_out.push_back(state_vec(6));
	}
	return state_out;
}

vector<arma::Mat<double>> ModelClass::ReturnInitState(frame_data &frame_in) {
	int N_parents;
	double scale_now;
	double scale_parent;
	double scale_child;
	arma::Col<double> state_now(7);
	arma::Col<double> state_parent(7);
	arma::Col<double> state_child(7);
	arma::Mat<double> M_parent(4,4);
	arma::Mat<double> M_child(4,4);
	arma::Mat<double> M_state(4,4);
	arma::Mat<double> M_out;
	vector<arma::Mat<double>> M_init;
	for (int i=0; i<N_components; i++) {
		N_parents = parent_list[i].size();
		M_out.eye(4,4);
		for (int j=0; j<N_parents; j++) {
			if (j>0) {
				if (state_calc[parent_list[i][j]] >= 0) {
					state_now = frame_in.init_state.col(state_calc[parent_list[i][j]]);
				}
				else {
					state_now = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
				}
				M_state = ModelClass::GetM(state_now);
				scale_parent = model_scale[parent_list[i][j-1]];
				state_parent(0) = joint_param_parent[parent_list[i][j]][0];
				state_parent(1) = joint_param_parent[parent_list[i][j]][1];
				state_parent(2) = joint_param_parent[parent_list[i][j]][2];
				state_parent(3) = joint_param_parent[parent_list[i][j]][3];
				state_parent(4) = joint_param_parent[parent_list[i][j]][4]*scale_parent;
				state_parent(5) = joint_param_parent[parent_list[i][j]][5]*scale_parent;
				state_parent(6) = joint_param_parent[parent_list[i][j]][6]*scale_parent;
				M_parent = ModelClass::GetM(state_parent);
				scale_child = model_scale[parent_list[i][j]];
				state_child(0) = joint_param_child[parent_list[i][j]][0];
				state_child(1) = joint_param_child[parent_list[i][j]][1];
				state_child(2) = joint_param_child[parent_list[i][j]][2];
				state_child(3) = joint_param_child[parent_list[i][j]][3];
				state_child(4) = joint_param_child[parent_list[i][j]][4]*scale_child;
				state_child(5) = joint_param_child[parent_list[i][j]][5]*scale_child;
				state_child(6) = joint_param_child[parent_list[i][j]][6]*scale_child;
				M_child = ModelClass::GetM(state_child);
				M_out = ModelClass::MultiplyM(M_out,ModelClass::MultiplyM(M_parent,ModelClass::MultiplyM(M_state,M_child)));
			}
			else {
				if (state_calc[parent_list[i][j]] >= 0) {
					state_now = frame_in.init_state.col(state_calc[parent_list[i][j]]);
				}
				else {
					state_now = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
				}
				M_state = ModelClass::GetM(state_now);
				scale_child = model_scale[parent_list[i][j]];
				state_child(0) = joint_param_child[parent_list[i][j]][0];
				state_child(1) = joint_param_child[parent_list[i][j]][1];
				state_child(2) = joint_param_child[parent_list[i][j]][2];
				state_child(3) = joint_param_child[parent_list[i][j]][3];
				state_child(4) = joint_param_child[parent_list[i][j]][4]*scale_child;
				state_child(5) = joint_param_child[parent_list[i][j]][5]*scale_child;
				state_child(6) = joint_param_child[parent_list[i][j]][6]*scale_child;
				M_child = ModelClass::GetM(state_child);
				M_out = ModelClass::MultiplyM(M_out,ModelClass::MultiplyM(M_state,M_child));
			}
		}
		M_init.push_back(M_out);
	}
	return M_init;
}

vector<Polygon_2> ModelClass::ReturnSilhouette(FocalGrid &fg, vector<arma::Mat<double>> M_state, arma::Col<double> view_vec, int cam_nr) {
	vector<Polygon_2> silhouette_vec;

	for (int i=0; i<N_components; i++) {
		double view_vector[3] = {view_vec(0),view_vec(1),view_vec(2)};
		vtkSmartPointer<vtkPolyData> polyDataTrans = ModelClass::TransformPolyData(model_polydata[i],M_state[i],model_scale[i]);
		silhouette_vec.push_back(ModelClass::GetSilhouette(fg, polyDataTrans, view_vector, cam_nr));
	}

	return silhouette_vec;
}

vector<Polygon_2> ModelClass::ReturnSilhouette2(FocalGrid &fg, vector<double> state_vec, int cam_nr) {
	vector<Polygon_2> silhouette_vec;
	arma::Col<double> view_vec = fg.CalculateViewVector(cam_nr);
	for (int i=0; i<N_components; i++) {
		arma::Col<double> state_in = {state_vec[i*7],state_vec[i*7+1],state_vec[i*7+2],state_vec[i*7+3],state_vec[i*7+4],state_vec[i*7+5],state_vec[i*7+6]};
		double view_vector[3] = {view_vec(0),view_vec(1),view_vec(2)};
		vtkSmartPointer<vtkPolyData> polyDataTrans = ModelClass::TransformPolyData(model_polydata[i],ModelClass::GetM(state_in),model_scale[i]);
		silhouette_vec.push_back(ModelClass::GetSilhouette(fg, polyDataTrans, view_vector, cam_nr));
	}

	return silhouette_vec;
}

void ModelClass::SetStrokeBounds(double StrokeBound) {
	stroke_bound = StrokeBound;
}

void ModelClass::SetDeviationBounds(double DeviationBound) {
	deviation_bound = DeviationBound;
}

void ModelClass::SetWingPitchBounds(double WingPitchBound) {
	wing_pitch_bound = WingPitchBound;
}

bool ModelClass::SetModel(session_data &session) {
	bool success = true;
	try {
		ModelClass::LoadModelMesh(session);
	}
	catch(...) {
		success = false;
	}
	return success;
}

bool ModelClass::LoadModelMesh(session_data &session) {
	bool success = true;
	try {
		model_polydata.clear();
		for (int i=0; i<N_components; i++) {
			model_polydata.push_back(ModelClass::LoadSTLFile(session.model_loc,stl_list[i]));
		}
	}
	catch(...) {
		success = false;
	}
	return success;
}

void ModelClass::TestPointInsidePolygonSpeed(arma::Mat<double> &pcl_in) {

	int N_points = pcl_in.n_cols;

	cout << "N_points " << N_points << endl;

	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

	for (int i=0; i<N_points; i++) {
		double t_point[3] = {pcl_in(0,i), pcl_in(1,i), pcl_in(2,i)};
		points->InsertNextPoint(t_point);
	}

	vtkSmartPointer<vtkPolyData> pointsPolydata = vtkSmartPointer<vtkPolyData>::New();
  	pointsPolydata->SetPoints(points);

	//Points inside test
	vtkSmartPointer<vtkSelectEnclosedPoints> selectEnclosedPoints = vtkSmartPointer<vtkSelectEnclosedPoints>::New();

	selectEnclosedPoints->SetInputData(pointsPolydata);

	selectEnclosedPoints->SetSurfaceData(model_polydata[0]);

	selectEnclosedPoints->Update();

	// Select enclosed points
	vtkDataArray* insideArray = vtkDataArray::SafeDownCast(selectEnclosedPoints->GetOutput()->GetPointData()->GetArray("SelectedPoints"));

	arma::Row<int> pts_inside_array(insideArray->GetNumberOfTuples());

  	for(vtkIdType j = 0; j < insideArray->GetNumberOfTuples(); j++)
    { 
    	pts_inside_array(j) = insideArray->GetComponent(j,0);
    }

}

void ModelClass::Test3DIntersection() {

	vtkSmartPointer<vtkBooleanOperationPolyDataFilter> booleanOperation = vtkSmartPointer<vtkBooleanOperationPolyDataFilter>::New();
	booleanOperation->SetOperationToIntersection();
	booleanOperation->SetInputData( 0, model_polydata[0] );
	booleanOperation->SetInputData( 1, model_polydata[2] );
	booleanOperation->Update();

}

void ModelClass::TestPoissonSurfaceReconstruction(arma::Mat<double> &pcl_in) {

	int N_points = pcl_in.n_cols;

	cout << "N_points " << N_points << endl;

	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

	for (int i=0; i<N_points; i++) {
		double t_point[3] = {pcl_in(0,i), pcl_in(1,i), pcl_in(2,i)};
		points->InsertNextPoint(t_point);
	}

	vtkSmartPointer<vtkPolyData> pointsPolydata = vtkSmartPointer<vtkPolyData>::New();
  	pointsPolydata->SetPoints(points);

  	// Construct the surface and create isosurface.	
	vtkSmartPointer<vtkSurfaceReconstructionFilter> surf = vtkSmartPointer<vtkSurfaceReconstructionFilter>::New();\
	surf->SetInputData(pointsPolydata);

	vtkSmartPointer<vtkContourFilter> cf = vtkSmartPointer<vtkContourFilter>::New();
	cf->SetInputConnection(surf->GetOutputPort());
	cf->SetValue(0, 0.0);

	cf->Update();

	// Sometimes the contouring algorithm can create a volume whose gradient
  	// vector and ordering of polygon (using the right hand rule) are
  	// inconsistent. vtkReverseSense cures this problem.
  	vtkSmartPointer<vtkReverseSense> reverse = vtkSmartPointer<vtkReverseSense>::New();
	reverse->SetInputConnection(cf->GetOutputPort());
	reverse->ReverseCellsOn();
	reverse->ReverseNormalsOn();

	reverse->Update();

	//vtkSmartPointer<vtkPolyData> poissonPoly = vtkSmartPointer<vtkPolyData>::New();
	//poissonPoly->SetPoints(reverse);

	vtkSmartPointer<vtkCenterOfMass> centerOfMassFilter = vtkSmartPointer<vtkCenterOfMass>::New();

	centerOfMassFilter->SetInputConnection(reverse->GetOutputPort());

	centerOfMassFilter->SetUseScalarsAsWeights(false);
	centerOfMassFilter->Update();
	 
	double center[3];
	centerOfMassFilter->GetCenter(center);

	std::cout << "Center of mass is " << center[0] << " " << center[1] << " " << center[2] << std::endl;

}

vtkSmartPointer<vtkPolyData> ModelClass::LoadSTLFile(string FileLoc, string FileName) {

	vtkSmartPointer<vtkPolyData> polyData = ModelClass::stlReader(FileLoc, FileName);

	return polyData;
}

vector<arma::Mat<double>> ModelClass::ReturnModelPCL(vector<arma::Mat<double>> M_in) {

	vector<arma::Mat<double>> pcl_out;

	arma::Mat<double> pcl_now;

	for (int i=0; i<N_components; i++) {
		pcl_now = model_pcls[i];
		pcl_out.push_back(M_in[i]*pcl_now);
	}

	return pcl_out;
}

vector<arma::Mat<double>> ModelClass::ReturnInitPCL(vector<arma::Mat<double>> M_in) {

	arma::Mat<double> init_body_pcl;
	arma::Mat<double> init_wing_L_pcl;
	arma::Mat<double> init_wing_R_pcl;

	arma::Mat<double> pcl_now;
	arma::Mat<double> pcl_scaled;

	for (int i=0; i<N_components; i++) {
		if (state_calc[i] == 0) {
			pcl_now = model_pcls[i];
			pcl_scaled = pcl_now*model_scale[i];
			pcl_scaled.row(3).fill(1.0);
			init_body_pcl = M_in[i]*pcl_scaled;
		}
		else if (state_calc[i] == 1) {
			pcl_now = model_pcls[i];
			pcl_scaled = pcl_now*model_scale[i];
			pcl_scaled.row(3).fill(1.0);
			init_wing_L_pcl = M_in[i]*pcl_scaled;
		}
		else if (state_calc[i] == 2) {
			pcl_now = model_pcls[i];
			pcl_scaled = pcl_now*model_scale[i];
			pcl_scaled.row(3).fill(1.0);
			init_wing_R_pcl = M_in[i]*pcl_scaled;
		}
		else {
			pcl_now = model_pcls[i];
			pcl_scaled = pcl_now*model_scale[i];
			pcl_scaled.row(3).fill(1.0);
			init_body_pcl = arma::join_rows(init_body_pcl,M_in[i]*pcl_scaled);
		}
	}

	vector<arma::Mat<double>> pcl_out;

	pcl_out.push_back(init_body_pcl);
	pcl_out.push_back(init_wing_L_pcl);
	pcl_out.push_back(init_wing_R_pcl);

	return pcl_out;

}

vtkSmartPointer<vtkPolyData> ModelClass::TransformPolyData(vtkSmartPointer<vtkPolyData> polyData, arma::Mat<double> M_in, double scale_in) {

	vtkSmartPointer<vtkMatrix4x4> M_t = vtkSmartPointer<vtkMatrix4x4>::New();

	M_t->SetElement(0,0,M_in(0,0));
	M_t->SetElement(0,1,M_in(0,1));
	M_t->SetElement(0,2,M_in(0,2));
	M_t->SetElement(0,3,M_in(0,3));
	M_t->SetElement(1,0,M_in(1,0));
	M_t->SetElement(1,1,M_in(1,1));
	M_t->SetElement(1,2,M_in(1,2));
	M_t->SetElement(1,3,M_in(1,3));
	M_t->SetElement(2,0,M_in(2,0));
	M_t->SetElement(2,1,M_in(2,1));
	M_t->SetElement(2,2,M_in(2,2));
	M_t->SetElement(2,3,M_in(2,3));
	M_t->SetElement(3,0,M_in(3,0));
	M_t->SetElement(3,1,M_in(3,1));
	M_t->SetElement(3,2,M_in(3,2));
	M_t->SetElement(3,3,M_in(3,3));

	vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();

	transform->SetMatrix(M_t);
	transform->Scale(scale_in,scale_in,scale_in);

	vtkSmartPointer<vtkTransformPolyDataFilter> transformPD = vtkSmartPointer<vtkTransformPolyDataFilter>::New();

	transformPD->SetInputData(polyData);
	transformPD->SetTransform(transform);
	transformPD->Update();

	//vtkPolyData* polyDataOut = transformPD->GetOutput();

	return transformPD->GetOutput();
}

Polygon_2 ModelClass::GetSilhouette(FocalGrid &fg, vtkSmartPointer<vtkPolyData> polyData, double view_vector[3], int cam_nr) {
 
	vtkSmartPointer<vtkPolyDataSilhouette> silhouette = vtkSmartPointer<vtkPolyDataSilhouette>::New();
	silhouette->SetInputData(polyData);
	silhouette->SetDirectionToSpecifiedVector();
	silhouette->SetVector(view_vector);
	silhouette->SetEnableFeatureAngle(0);
	//silhouette->BorderEdgesOff();
	silhouette->Update();

	vtkPolyData* silhdata = silhouette->GetOutput();

	int N_lines = silhdata->GetNumberOfLines();

	arma::Mat<double> sil_mat(4,N_lines*2);
	arma::Row<int> id_list(N_lines*2);

	double p1[3];
	double p2[3];

	vtkSmartPointer<vtkIdList> idList = vtkSmartPointer<vtkIdList>::New();
	int iter = 0;
	while (silhdata->GetLines()->GetNextCell(idList)) {
		silhdata->GetPoint(idList->GetId(0),p1);
		silhdata->GetPoint(idList->GetId(1),p2);
		id_list(iter*2) = idList->GetId(0);
		id_list(iter*2+1) = idList->GetId(1);
		sil_mat(0,iter*2) = p1[0];
		sil_mat(1,iter*2) = p1[1];
		sil_mat(2,iter*2) = p1[2];
		sil_mat(3,iter*2) = 1.0;
		sil_mat(0,iter*2+1) = p2[0];
		sil_mat(1,iter*2+1) = p2[1];
		sil_mat(2,iter*2+1) = p2[2];
		sil_mat(3,iter*2+1) = 1.0;
		iter++;
	}

	arma::Mat<double> silh_pts = fg.ProjectCloud2UVdouble(sil_mat,cam_nr);

	// Insert the 2d segments in a 2d arrangement
	Arrangement_2 arr;
	Walk_pl pl(arr);
	for (int i=0; i<N_lines; i++) {
		Segment_2 s_now (Point_arr_2 (silh_pts(0,i*2), silh_pts(1,i*2)), Point_arr_2 (silh_pts(0,i*2+1), silh_pts(1,i*2+1)));
		insert(arr, s_now, pl);
	}

	Arrangement_2::Vertex_const_iterator vit;

	arma::Row<int> vertex_degree;
	vertex_degree.zeros(arr.number_of_vertices());
	arma::Mat<double> vertex_mat;
	vertex_mat.zeros(2,arr.number_of_vertices());
	iter = 0;
	for (vit = arr.vertices_begin(); vit != arr.vertices_end(); ++vit) {
  		vertex_degree(iter) = vit->degree();
  		vertex_mat(0,iter) = CGAL::to_double((vit->point()).x());
  		vertex_mat(1,iter) = CGAL::to_double((vit->point()).y());
  		iter++;
  	}

  	//cout << vertex_mat << endl;

  	//cout << vertex_degree << endl;

  	// Find open ends:
  	arma::uvec open_ends = arma::find(vertex_degree==1);

  	//cout << open_ends << endl;

  	if (open_ends.is_empty()) {
  		//cout << "no open ends" << endl;
  	}
  	else {
	  	int N_open_ends = open_ends.n_rows;
	  	//cout << "no of open ends: " << N_open_ends << endl;
	  	if (N_open_ends%2==0) {
	  		int comb_arr[N_open_ends];
	  		for (int k=0; k<N_open_ends; k++) {
	  			comb_arr[k] = k;
	  		}

	  		sort(comb_arr,comb_arr+N_open_ends);

	  		do {
	  			int add_2_mat = 0;
				for (int i=0; i<(N_open_ends/2); i++) {
					if (comb_arr[i*2]<comb_arr[i*2+1]) {
						add_2_mat++;
					}
				}
				if (add_2_mat==(N_open_ends/2)) {
		    		for (int j=0; j<(N_open_ends/2); j++) {
		    			//cout << comb_arr[j*2] << " " << comb_arr[j*2+1] << " ";
		    			Segment_2 s_now (Point_arr_2 (vertex_mat(0,open_ends(comb_arr[j*2])), vertex_mat(1,open_ends(comb_arr[j*2]))), 
		    			Point_arr_2 (vertex_mat(0,open_ends(comb_arr[j*2+1])), vertex_mat(1,open_ends(comb_arr[j*2+1]))));
		    			insert(arr, s_now, pl);
		    		}
		    		//cout << endl;
		    	}
	  		} while (next_permutation(comb_arr,comb_arr+N_open_ends-1));


  		}
  		else {
  			cout << "odd number of open ends" << endl;
  		}
  	}

  	// Remove internal edges:
  	Arrangement_2::Face_handle unb_face = arr.unbounded_face();
  	Arrangement_2::Edge_iterator eit;
  	for (eit = arr.edges_begin(); eit!=arr.edges_end(); ++eit) {
  		Arrangement_2::Halfedge_handle he = eit;
  		if ((he->face()!=unb_face)&&(he->twin()->face()!=unb_face)) {
  			arr.remove_edge(eit);
  		}
  	}

  	Polygon_2 P_out;

  	//cout << "Bounded faces: " << endl;
  	Arrangement_2::Face_iterator fit;
  	Arrangement_2::Ccb_halfedge_const_circulator  curr;
  	for (fit = arr.faces_begin(); fit != arr.faces_end(); ++fit) {
  		if(fit->is_unbounded()) {
  			//cout << "unbounded face" << endl;
  		}
  		else {
  			//cout << "bounded face" << endl;
  			curr = fit->outer_ccb();
  			//cout << "Direction " << curr->direction() << endl;
      		//cout << curr->source()->point();
      		//P_out.push_back(Point_2 (CGAL::to_double((curr->source()->point()).x()), CGAL::to_double((curr->source()->point()).y())));
      		do {
        		//cout << " --> " << curr->target()->point();
        		//cout << "  direction " << curr->direction();
        		P_out.push_back(Point_2 (CGAL::to_double((curr->target()->point()).x()), CGAL::to_double((curr->target()->point()).y())));
        		++curr;
      		} while (curr != fit->outer_ccb());
      		//cout << endl;
  		}
  	}

	//cout << "N lines = " << N_lines << endl;

	// Print the size of the arrangement.
	//cout << "The arrangement size:" << std::endl
    //	<< "   V = " << arr.number_of_vertices()
    //    << ",  E = " << arr.number_of_edges()
    //    << ",  F = " << arr.number_of_faces() << endl;

    //Polygon_2 P_out;
    
    //Arrangement_2::Vertex_iterator vit2;
    
    //for (vit2 = arr.vertices_begin(); vit2 != arr.vertices_end(); ++vit2) {
  		//Arrangement_2::Vertex_handle ve = vit2;
  		//P_out.push_back(Point_2 (CGAL::to_double((vit2->point()).x()), CGAL::to_double((vit2->point()).y())));
  		//cout << vit2->point() << endl;
  	//}

  	//cout << "P_out is simple: " << P_out.is_simple() << endl;

    //arma::uvec unique_ids = arma::find_unique(id_list);
    //arma::Mat<double> unique_pts = silh_pts.cols(unique_ids);

    //for (int j=0; j<unique_pts.n_cols; j++) {
    //	P_out.push_back(Point_2 (unique_pts(0,j), unique_pts(1,j)));
    //}

    return P_out;
}

/*

Polygon_2 ModelClass::GetSilhouette(FocalGrid &fg, vtkSmartPointer<vtkPolyData> polyData, double view_vector[3], int cam_nr) {
 
	vtkSmartPointer<vtkPolyDataSilhouette> silhouette = vtkSmartPointer<vtkPolyDataSilhouette>::New();
	silhouette->SetInputData(polyData);
	silhouette->SetDirectionToSpecifiedVector();
	silhouette->SetVector(view_vector);
	silhouette->SetEnableFeatureAngle(0);
	//silhouette->BorderEdgesOff();
	silhouette->Update();

	vtkPolyData* silhdata = silhouette->GetOutput();

	//vtkSmartPointer<vtkPolyDataConnectivityFilter> connectivityFilter = 
    //vtkSmartPointer<vtkPolyDataConnectivityFilter>::New();
  	//connectivityFilter->SetInputConnection(silhouette->GetOutputPort());
  	//connectivityFilter->SetExtractionModeToLargestRegion(); 
  	//connectivityFilter->SetExtractionModeToClosestPointRegion();
  	//connectivityFilter->SetExtractionModeToAllRegions();
  	//connectivityFilter->Update();

  	//vtkPolyData* silhdata = connectivityFilter->GetOutput();

  	//vtkSmartPointer<vtkCleanPolyData> cleaner = vtkSmartPointer<vtkCleanPolyData>::New();
  	//cleaner->SetInputConnection(silhouette->GetOutputPort());
  	//cleaner->Update();

  	//vtkPolyData* silhdata = cleaner->GetOutput();

	int N_lines = silhdata->GetNumberOfLines();

	arma::Mat<double> sil_mat(4,N_lines*2);
	arma::Row<int> id_list(N_lines*2);

	double p1[3];
	double p2[3];

	vtkSmartPointer<vtkIdList> idList = vtkSmartPointer<vtkIdList>::New();
	int iter = 0;
	while (silhdata->GetLines()->GetNextCell(idList)) {
		silhdata->GetPoint(idList->GetId(0),p1);
		silhdata->GetPoint(idList->GetId(1),p2);
		id_list(iter*2) = idList->GetId(0);
		id_list(iter*2+1) = idList->GetId(1);
		sil_mat(0,iter*2) = p1[0];
		sil_mat(1,iter*2) = p1[1];
		sil_mat(2,iter*2) = p1[2];
		sil_mat(3,iter*2) = 1.0;
		sil_mat(0,iter*2+1) = p2[0];
		sil_mat(1,iter*2+1) = p2[1];
		sil_mat(2,iter*2+1) = p2[2];
		sil_mat(3,iter*2+1) = 1.0;
		iter++;
	}

	cout << "id list" << endl;
	cout << id_list << endl;

	arma::Mat<double> silh_pts = fg.ProjectCloud2UVdouble(sil_mat,cam_nr);

	vector<list<int>> line_seg_vec;
	list<int> line_seg;

	int prev_id = id_list(0);
	int next_id = id_list(1);
	id_list(0) = -1;
	id_list(1) = -1;
	int count = 0;

	while (count<(silh_pts.n_cols/2)) {

		if (prev_id<0 && next_id<0) {
			line_seg_vec.push_back(line_seg);
			line_seg.clear();
			arma::uvec ids_left = arma::find(id_list>-1);
			if (ids_left.is_empty()) {
				count = silh_pts.n_cols/2;
			}
			else {
				prev_id = id_list(ids_left(0));
				next_id = id_list(ids_left(0)+1);
			}
		}
		else {
			arma::uvec found_prev_ids = arma::find(id_list==prev_id);

			if (found_prev_ids.is_empty()) {
				prev_id = -10;
			}
			else {
					if (found_prev_ids(0)%2==0) {
						line_seg.push_front(found_prev_ids(0));
						prev_id = id_list(found_prev_ids(0)+1);
						id_list(found_prev_ids(0)) = -1;
						id_list(found_prev_ids(0)+1) = -1;
					}
					else {
						line_seg.push_front(found_prev_ids(0)-1);
						prev_id = id_list(found_prev_ids(0)-1);
						id_list(found_prev_ids(0)) = -1;
						id_list(found_prev_ids(0)-1) = -1;
					}
					count++;
			}

			arma::uvec found_next_ids = arma::find(id_list==next_id);

			if (found_next_ids.is_empty()) {
				next_id = -10;
			}
			else {
					if (found_next_ids(0)%2==0) {
						line_seg.push_back(found_next_ids(0)+1);
						next_id = id_list(found_next_ids(0)+1);
						id_list(found_next_ids(0)) = -1;
						id_list(found_next_ids(0)+1) = -1;
					}
					else {
						line_seg.push_back(found_next_ids(0)-1);
						next_id = id_list(found_next_ids(0)-1);
						id_list(found_next_ids(0)) = -1;
						id_list(found_next_ids(0)-1) = -1;
					}
					count++;
			}
		}

		if (prev_id==next_id && prev_id+next_id>=0) {
			list<int>::iterator it1 = line_seg.begin();
			it1 = line_seg.erase(it1);
		}

	}

	cout << "finished" << endl;

	int N_seg = line_seg_vec.size();

	arma::Row<int> connectivity_vec;
	connectivity_vec.zeros(2*N_seg);

	if (N_seg>1) {

		arma::Mat<double> connectivity_pts(3,2*N_seg);
		arma::Row<int> connectivity_ids(2*N_seg);
		int connectivity_arr[2*(N_seg-1)];
		for (int i=0; i<N_seg; i++) {
			if (i==0) {
				connectivity_pts.col(i*2) = silh_pts.col(line_seg_vec[i].front());
				connectivity_pts.col(i*2+1) = silh_pts.col(line_seg_vec[i].back());
				connectivity_ids(i*2) = line_seg_vec[i].front();
				connectivity_ids(i*2+1) = line_seg_vec[i].back();
			}
			else {
				connectivity_arr[(i-1)*2] = i*2;
				connectivity_arr[(i-1)*2+1] = i*2+1;
				connectivity_pts.col(i*2) = silh_pts.col(line_seg_vec[i].front());
				connectivity_pts.col(i*2+1) = silh_pts.col(line_seg_vec[i].back());
				connectivity_ids(i*2) = line_seg_vec[i].front();
				connectivity_ids(i*2+1) = line_seg_vec[i].back();
			}
		}

		sort(connectivity_arr, connectivity_arr+2*(N_seg-1));

		arma::Mat<int> connectivity_mat;
		connectivity_mat.zeros(1,2*N_seg);

		int row_count = 0;
		do {
			int add_2_mat = 0;
			for (int j=0; j<(N_seg-1); j++) {
				if (abs(connectivity_arr[j*2]-connectivity_arr[j*2+1])==1) {
					add_2_mat++;
				}
			}
			if (add_2_mat==(N_seg-1)) {
				if (row_count==0) {
					connectivity_mat(row_count,0) = 0;
					connectivity_mat(row_count,1) = 1;
					for (int j=0; j<(N_seg-1); j++) {
						connectivity_mat(row_count,(j+1)*2) = connectivity_arr[j*2];
						connectivity_mat(row_count,(j+1)*2+1) = connectivity_arr[j*2+1];
					}
				}
				else {
					arma::Row<int> temp_row(2*N_seg);
					temp_row(0) = 0;
					temp_row(1) = 1;
					for (int j=0; j<(N_seg-1); j++) {
						temp_row((j+1)*2) = connectivity_arr[j*2];
						temp_row((j+1)*2+1) = connectivity_arr[j*2+1];
					}
					connectivity_mat.insert_rows(row_count,temp_row);
				}

				// Check if lines intersect:
				Polygon_2 P_test;
				double pt_diff;
				for (int i=0; i<N_seg; i++) {
					if (connectivity_ids(connectivity_mat(row_count,i*2))==connectivity_ids(connectivity_mat(row_count,i*2+1))) {
						P_test.push_back(Point_2 (connectivity_pts(0,connectivity_mat(row_count,i*2)), connectivity_pts(1,connectivity_mat(row_count,i*2))));
					}
					else {
						P_test.push_back(Point_2 (connectivity_pts(0,connectivity_mat(row_count,i*2)), connectivity_pts(1,connectivity_mat(row_count,i*2))));
						P_test.push_back(Point_2 (connectivity_pts(0,connectivity_mat(row_count,i*2+1)), connectivity_pts(1,connectivity_mat(row_count,i*2+1))));
					}
				}

				if (P_test.is_simple()==true) {
					connectivity_vec = connectivity_mat.row(row_count);
					break;
				}

				row_count++;
			}
		} while (next_permutation(connectivity_arr, connectivity_arr+2*(N_seg-1)));

	}
	else {
		connectivity_vec(0) = 0;
		connectivity_vec(1) = 1;
	}

	cout << connectivity_vec << endl;

	Polygon_2 P_temp;

	int list_nr;

	list<Segment_2> segment_list;

	cout << "printing segments" << endl;
	cout << N_seg << endl;

	for (int i=0; i<N_seg; i++) {
		if (connectivity_vec(i*2)<connectivity_vec(i*2+1)) {
			list_nr = connectivity_vec(i*2)/2;
			for (list<int>::iterator it=line_seg_vec[i].begin(); it!=line_seg_vec[i].end(); ++it) {
				cout << " " << *it;
				Point_2 t_p1 (silh_pts(0,*it), silh_pts(1,*it));
				Point_2 t_p2 (silh_pts(0,next(*it)), silh_pts(1,next(*it)));
				segment_list.push_back(Segment_2 (t_p1, t_p2));
				P_temp.push_back(Point_2 (silh_pts(0,*it), silh_pts(1,*it)));
			}
		}
		else {
			list_nr = (connectivity_vec(i*2)-1)/2;
			for (list<int>::reverse_iterator rit=line_seg_vec[i].rbegin(); rit!=line_seg_vec[i].rend(); ++rit) {
				cout << " " << *rit;
				segment_list.push_back(Point_2 (silh_pts(0,*rit), silh_pts(1,*rit)));
				P_temp.push_back(Point_2 (silh_pts(0,*rit), silh_pts(1,*rit)));
			}
		}
	}

	Arrangement_2 arr;
	insert(arr, segment_list.begin(), segment_list.end());
	for (auto it = arr.begin_vertices(); it != arr.end_vertices(); ++it) {
		if (4 == it->degree()) {
			cout << "intersection vertex " << *it << endl;
		}
	}

	Polygon_2 P_temp;

	cout << "printing segments" << endl;
	cout << N_seg << endl;

	int list_nr;

	for (int i=0; i<N_seg; i++) {
		if (connectivity_vec(i*2)<connectivity_vec(i*2+1)) {
			list_nr = connectivity_vec(i*2)/2;
			for (list<int>::iterator it=line_seg_vec[i].begin(); it!=line_seg_vec[i].end(); ++it) {
				cout << " " << *it;
				P_temp.push_back(Point_2 (silh_pts(0,*it), silh_pts(1,*it)));
			}
		}
		else {
			list_nr = (connectivity_vec(i*2)-1)/2;
			for (list<int>::reverse_iterator rit=line_seg_vec[i].rbegin(); rit!=line_seg_vec[i].rend(); ++rit) {
				cout << " " << *rit;
				P_temp.push_back(Point_2 (silh_pts(0,*rit), silh_pts(1,*rit)));
			}
		}
	}

	// Iterate through the vertices, remove vertices which lay inside the contour:
	//for (VertexIterator vi = P_temp.vertices_begin(); vi != P_temp.vertices_end(); ++vi) {
		//cout << P_temp.bounded_side(*vi) << endl;
		//cout << CGAL::bounded_side_2(P_temp.vertices_begin(), P_temp.vertices_end(), *vi, KI()) << endl;
	//}

	//Polygon_with_holes_2 P_holes(P_temp);

	Polygon_2 P_out = P_temp;

	cout << " " << endl;

	cout << "P_out is simple : " << P_out.is_simple() << endl;

	return P_out;
}

*/

vtkSmartPointer<vtkPolyData> ModelClass::stlReader(string FileLoc, string FileName) {

	vtkSmartPointer<vtkPolyData> polyData;

	string inputFilename = FileLoc + "/" + FileName;
	vtkSmartPointer<vtkSTLReader> reader = vtkSmartPointer<vtkSTLReader>::New();
	reader->SetFileName(inputFilename.c_str());
	reader->Update();

	vtkSmartPointer<vtkCleanPolyData> clean = vtkSmartPointer<vtkCleanPolyData>::New();
	clean->SetInputConnection(reader->GetOutputPort());
	clean->Update();

	polyData = clean->GetOutput();

	return polyData;
}

void ModelClass::SetScale(vector<double> scale_in) {
	model_scale.clear();
	cout << "" << endl;
	cout << "------------------------------------------------" << endl;
	cout << "" << endl;
	cout << "model scales: " << endl;
	for (int i=0; i<N_components; i++) {
		cout << stl_list[i] << ": " << to_string(scale_in[i]) << endl;
		model_scale.push_back(scale_in[i]);
	}
	cout << "" << endl;
	cout << "------------------------------------------------" << endl;
	cout << "" << endl;
}

vector<double> ModelClass::GetModelScale() {
	return model_scale;
}

bool ModelClass::ScaleModelPCL() {
	bool success = true;
	try {
		for (int i=0; i<N_components; i++) {
			arma::Mat<double> scale_mat;
			scale_mat.eye(4,4);
			scale_mat(0,0) = model_scale[i];
			scale_mat(1,1) = model_scale[i];
			scale_mat(2,2) = model_scale[i];
			model_pcls[i] = scale_mat*model_pcls[i];
		}
	}
	catch(...) {
		success = false;
	}
	return success;
}

arma::Mat<double> ModelClass::GetM(arma::Col<double> state_in) {

	arma::Col<double> q_vec = {state_in(0),state_in(1),state_in(2),state_in(3)};
	q_vec = q_vec/sqrt(pow(q_vec(0),2)+pow(q_vec(1),2)+pow(q_vec(2),2)+pow(q_vec(3),2));

	double q0 = q_vec(0);
	double q1 = q_vec(1);
	double q2 = q_vec(2);
	double q3 = q_vec(3);
	double tx = state_in(4);
	double ty = state_in(5);
	double tz = state_in(6);

	arma::Mat<double> M = {{2.0*pow(q0,2.0)-1.0+2.0*pow(q1,2.0), 2.0*q1*q2+2.0*q0*q3, 2.0*q1*q3-2.0*q0*q2, tx},
		{2.0*q1*q2-2.0*q0*q3, 2.0*pow(q0,2.0)-1.0+2.0*pow(q2,2.0), 2.0*q2*q3+2.0*q0*q1, ty},
		{2.0*q1*q3+2.0*q0*q2, 2.0*q2*q3-2.0*q0*q1, 2.0*pow(q0,2.0)-1.0+2.0*pow(q3,2.0), tz},
		{0.0, 0.0, 0.0, 1.0}};

	return M;
}

arma::Col<double> ModelClass::GetStateFromM(arma::Mat<double> M_in) {

	double q0 = 0.5*sqrt(M_in(0,0)+M_in(1,1)+M_in(2,2)+1.0);
	double q1 = (M_in(1,2)-M_in(2,1))/(4.0*q0);
	double q2 = (M_in(2,0)-M_in(0,2))/(4.0*q0);
	double q3 = (M_in(0,1)-M_in(1,0))/(4.0*q0);
	double tx = M_in(0,3);
	double ty = M_in(1,3);
	double tz = M_in(2,3);

	arma::Col<double> q = {q0, q1, q2, q3};
	q = q/arma::norm(q);

	arma::Col<double> state_out = {q(0), q(1), q(2), q(3), tx, ty, tz};

	return state_out;
}

arma::Mat<double> ModelClass::MultiplyM(arma::Mat<double> MA, arma::Mat<double> MB) {

	arma::Mat<double> RA = MA.submat(0,0,2,2);
	arma::Mat<double> RB = MB.submat(0,0,2,2);
	arma::Mat<double> RC = RA*RB;
	arma::Mat<double> MC;
	MC.eye(4,4);
	MC.submat(0,0,2,2) = RC;
	arma::Col<double> TA(3);
	arma::Col<double> TB(3);
	TA(0) = MA(0,3);
	TA(1) = MA(1,3);
	TA(2) = MA(2,3);
	TB(0) = MB(0,3);
	TB(1) = MB(1,3);
	TB(2) = MB(2,3);
	arma::Col<double> TC = TA+RA*TB;
	MC(0,3) = TC(0);
	MC(1,3) = TC(1);
	MC(2,3) = TC(2);
	return MC;
}

arma::Mat<double> ModelClass::TransposeM(arma::Mat<double> M_in) {
	arma::Mat<double> M_out;
	M_out.eye(4,4);
	M_out.submat(0,0,2,2) = M_in.submat(0,0,2,2).t();
	M_out(0,3) = M_in(0,3);
	M_out(1,3) = M_in(1,3);
	M_out(2,3) = M_in(2,3);
	return M_out;
}

vector<string> ModelClass::GetPointLabels() {
	return drag_point_labels;
}

vector<string> ModelClass::GetPointSymbols() {
	return drag_point_symbols;
}

vector<vector<int>> ModelClass::GetPointColors() {
	return drag_point_colors;
}

vector<vector<double>> ModelClass::GetPointStartPos() {
	return drag_point_start_pos;
}

vector<vector<int>> ModelClass::GetLineConnectivity() {
	return drag_line_connectivity;
}

vector<vector<int>> ModelClass::GetLineColors() {
	return drag_line_colors;
}

vector<string> ModelClass::GetScaleTexts() {
	return scale_texts;
}

vector<vector<int>> ModelClass::GetScaleCalc() {
	return scale_calc;
}

vector<vector<int>> ModelClass::GetLengthCalc() {
	return length_calc;
}

vector<int> ModelClass::GetContourCalc() {
	return contour_calc;
}

vector<int> ModelClass::GetOriginInd() {
	return origin_ind;
}