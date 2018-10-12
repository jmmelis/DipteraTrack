#ifndef SESSION_DATA_H
#define SESSION_DATA_H

#include <string>
#include <stdint.h>
#include <vector>
#include <armadillo>

using namespace std;

struct session_data {

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

};
#endif