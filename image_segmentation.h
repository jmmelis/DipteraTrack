#ifndef IMAGE_SEGMENTATION_H
#define IMAGE_SEGMENTATION_H

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
#include "opencv2/ximgproc/segmentation.hpp"
#include <opencv2/video/background_segm.hpp>
#include <armadillo>

using namespace cv::ximgproc::segmentation;
using namespace std;

namespace p = boost::python;
namespace np = boost::python::numpy;

class ImagSegm {

	public:

		ImagSegm();

		int body_thresh;
		int body_blur_sigma;
		int body_blur_window;
		double body_sigma;
		int body_K;
		int min_body_size;
		double body_length;
		vector<tuple<double,double>> origin;
		int wing_thresh;
		int body_dilation;
		double wing_sigma;
		int wing_K;
		int min_wing_size;
		double wing_length;
		vector<arma::Col<int>> image_masks;

		//cv::setUseOptimized(true);
		//cv::setNumThreads(8);

		cv::Ptr<GraphSegmentation> gs = createGraphSegmentation();

		void SetBodySegmentationParam(int BodyThresh, int BodyBlurSigma, int BodyBlurWindow, double BodySigma, int BodyK, int MinBodySize, double BodyLength, vector<tuple<double,double>> Origin);
		void SetWingSegmentationParam(int WingThresh, int BodyDilation, double WingSigma, int WingK, int MinWingSize, double WingLength);
		void SegmentFrame(frame_data &frame_in);
		void SetImageMask(frame_data &frame_in, int cam_nr, int seg_nr);
		arma::Col<int> ApplyMask(arma::Col<int> &img_in, int cam_nr);
		void ResetImageMask(frame_data &frame_in);
		tuple<arma::Col<double>,arma::Col<int>> SelectBody(arma::Col<int> &frame_in, int N_row, int N_col, arma::Mat<double> seg_prop, tuple<double,double> origin_loc);
		arma::Mat<double> FindBodyAndWingContours(arma::Col<int> &raw_img_vec, arma::Col<int> &body_img_vec, int N_row, int N_col);
		tuple<arma::Mat<double>,arma::Mat<int>> SelectWing(arma::Col<int> &frame_in, int N_row, int N_col, arma::Mat<double> seg_prop, arma::Col<double> body_prop);
		arma::Col<int> BodyThresh(arma::Col<int> &frame_in, int N_row, int N_col, int body_thresh, int sigma_blur, int blur_window);
		arma::Col<int> GlueSegments(arma::Col<int> body_seg, arma::Mat<int> wing_seg, arma::Col<int> raw_img_vec, int N_row, int N_col);
		arma::Col<int> WingThresh(arma::Col<int> &frame_in, arma::Col<int> &body_frame_in, int N_row, int N_col, int wing_thresh, int body_dilation, arma::Col<double> body_prop);
		arma::Mat<double> GetSegmentProperties(arma::Col<int> &frame_in, int N_row, int N_col);
		arma::Col<int> GraphSeg(arma::Col<int> &frame_in, int N_row, int N_col, double Sigma, int K, int minsize);
		arma::Mat<double> FindContours(arma::Col<int> &img_in, int N_row, int N_col);
		arma::Mat<double> FindDoubleContours(arma::Mat<double> &img_mat_in, int N_row, int N_col);
		arma::Col<int> CVMat2Vector(cv::Mat &img);
		cv::Mat Vector2CVMat(arma::Col<int> &img_vec, int N_row, int N_col);
		np::ndarray ReturnSegFrame(frame_data &frame_now, int cam_nr);
		np::ndarray ReturnOuterContour(frame_data &frame_now, int cam_nr);

};
#endif