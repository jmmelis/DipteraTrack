#include <string>
#include <stdint.h>
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <dirent.h>
#include <armadillo>
#include "json.hpp"

using namespace std;

using json = nlohmann::json;

#define PI 3.141592653589793

struct model_dat_layout {

	int N_components;
	vector<string> stl_list;
	vector<vector<int>> parent_list;
	vector<int> joint_type_list;
	vector<vector<double>> joint_param_parent;
	vector<vector<double>> joint_param_child;

};

int main ()
{

	//----------------------------------------------------------------------------------------------------
	// Drosophila model rigid wing
	//----------------------------------------------------------------------------------------------------
	int N_components = 5;

	int N_joints = 5;

	// List the stl files of the model
	vector<string> stl_list = {"thorax.stl",
		"head.stl",
		"abdomen.stl",
		"wing_L.stl",
		"wing_R.stl"};

	// List with the kinematic tree structure
	vector<vector<int>> parent_list = {{0},
		{0,1},
		{0,2},
		{0,3},
		{0,4}};

	// List with the different joint types (6 DOF = 0, Ball joint = 1, Spherical surface joint = 2, 
	// Single axis of rotation joint = 3, Double axis of rotation joint = 4, Rigid joint = 5)
	vector<int> joint_type_list = {	0,
		1,
		4,
		2,
		2};

	vector<vector<double>> joint_param_parent = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{1.0, 0.0, 0.0, 0.0, 0.69, 0.0, 0.0},
		{1.0, 0.0, 0.0, 0.0, -0.25, 0.0, -0.30},
		{1.0, 0.0, 0.0, 0.0, 0.0, 0.50, 0.0},
		{1.0, 0.0, 0.0, 0.0, 0.0, -0.50, 0.0}};

	vector<vector<double>> joint_param_child = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{0.9397, 0.0, 0.3420, 0.0, 0.0, 0.0, 0.0},
		{1.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.0},
		{1.0, 0.0, 0.0, 0.0, 0.0, -0.15, 0.0}};

	vector<string> drag_point_labels = {"origin","neck","head tip","abdomen hinge","abdomen tip","joint L","tip L","joint R", "tip R"};

	vector<string> drag_point_symbols = {"o", "o", "o", "o", "o", "o", "o", "o", "o"};

	vector<vector<int>> drag_point_colors = {{0,0,255},
		{0,0,255},
		{0,0,255},
		{0,0,255},
		{0,0,255},
		{255,0,0},
		{255,0,0},
		{0,255,0},
		{0,255,0}};

	vector<vector<double>> drag_point_start_pos = {{0.0,0.0,0.0},
		{0.5,0.5,0.5},
		{1.0,1.0,1.0},
		{-0.5,-0.5,-0.5},
		{-1.0,-1.0,-1.0},
		{0.5,0.5,-0.5},
		{1.0,1.0,-1.0},
		{-0.5,-0.5,0.5},
		{-1.0,-1.0,1.0}};

	vector<vector<int>> drag_line_connectivity = {{0,1},
		{0,3},
		{0,5},
		{0,7},
		{1,2},
		{3,4},
		{5,6},
		{7,8}};

	vector<vector<int>> drag_line_colors = {{0,0,255,255,2},
		{0,0,255,255,2},
		{0,0,255,255,2},
		{0,0,255,255,2},
		{0,0,255,255,2},
		{0,0,255,255,2},
		{255,0,0,255,2},
		{0,255,0,255,2}};

	vector<string> scale_texts = {"thorax scale: ", "head scale: ", "abdomen scale", "left wing scale: ", "right wing scale: "};

	vector<vector<int>> scale_calc = {{5,7}, // thorax
		{1,2}, // head
		{3,4}, // abdomen
		{5,6}, // left wing
		{7,8}};// right wing

	vector<vector<int>> length_calc = {{2,4}, // body length
		{5,6}, // left wing
		{7,8}}; // right wing

	vector<int> origin_joint_indices = {0,1,5,7};

	vector<int> contour_calc = {3,4};

	double SRF_angle = 45.0*(PI/180.0);

	vector<int> state_calc = {0,-1,-1,1,2};

	// Create json object:

	json config_file;

	config_file["N_components"] = N_components;
	config_file["N_joints"] = N_joints;
	config_file["stl_list"] = stl_list;
	config_file["parent_list"] = parent_list;
	config_file["joint_type_list"] = joint_type_list;
	config_file["joint_param_parent"] = joint_param_parent;
	config_file["joint_param_child"] = joint_param_child;
	config_file["drag_point_labels"] = drag_point_labels;
	config_file["drag_point_symbols"] = drag_point_symbols;
	config_file["drag_point_colors"] = drag_point_colors;
	config_file["drag_point_start_pos"] = drag_point_start_pos;
	config_file["drag_line_connectivity"] = drag_line_connectivity;
	config_file["drag_line_colors"] = drag_line_colors;
	config_file["scale_texts"] = scale_texts;
	config_file["scale_calc"] = scale_calc;
	config_file["length_calc"] = length_calc;
	config_file["origin_indices"] = origin_joint_indices;
	config_file["contour_calc"] = contour_calc;
	config_file["srf_angle"] = SRF_angle;
	config_file["state_calc"] = state_calc;

	cout << config_file.dump(2) << endl;

	// save json object

	chdir("/home/jmm/Documents/insect_models/drosophila_melanogaster");

	ofstream outfile("drosophila_melanogaster.json");
	outfile << config_file.dump(2) << endl;

	return 0;
}