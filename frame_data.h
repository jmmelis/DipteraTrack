#ifndef FRAME_DATA_H
#define FRAME_DATA_H

#include <string>
#include <stdint.h>
#include <vector>
#include <armadillo>

using namespace std;

struct frame_data {

	int N_cam;
	int mov_nr;
	int frame_nr;
	vector<tuple<int,int>> image_size;
	vector<arma::Col<int>> raw_frame;
	vector<arma::Col<int>> seg_frame;
	vector<arma::Mat<double>> seg_contours;
	arma::Mat<double> seg_pcl;
	arma::Mat<double> init_state;
	arma::Mat<double> body_and_wing_pcls;
};
#endif