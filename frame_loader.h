#ifndef FRAME_LOADER_CLASS_H
#define FRAME_LOADER_CLASS_H

#include "session_data.h"
#include "frame_data.h"

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <string>
#include <stdint.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <armadillo>

using namespace std;

namespace p = boost::python;
namespace np = boost::python::numpy;

class FrameLoader {

	public:

		vector<tuple<int,int>> image_size;
		vector<arma::Col<int>> bckg_images;

		FrameLoader();

		bool LoadBackground(session_data &session);
		void SetFrame(session_data &session, frame_data &frame);
		bool LoadFrame(session_data &session, frame_data &frame, int MovNR, int FrameNR);
		arma::Col<int> BackgroundSubtract(arma::Col<int> &img_vec, arma::Col<int> &bckg_vec);
		np::ndarray ReturnFrame(session_data &session, frame_data &frame, int CamNR);
		arma::Col<int> CVMat2Vector(cv::Mat &img);
		cv::Mat Vector2CVMat(arma::Col<int> &img_vec, int N_row, int N_col);

};
#endif