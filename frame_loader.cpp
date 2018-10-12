#include "frame_loader.h"

#include <string>
#include <stdint.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <armadillo>

using namespace std;

FrameLoader::FrameLoader() {
	// empty
}

bool FrameLoader::LoadBackground(session_data &session) {

	bool bckg_loaded = true;

	image_size.clear();
	bckg_images.clear();

	for (int i=0; i<session.N_cam; i++) {

		string img_name = session.session_loc+"/"+session.bckg_loc+"/"+session.bckg_img_names[i];

		cv::Mat image = cv::imread(img_name, cv::IMREAD_GRAYSCALE);

		if (image.empty()) {
			bckg_loaded = false;
		}
		else {
			arma::Col<int> bckg_vec = FrameLoader::CVMat2Vector(image);
			bckg_images.push_back(bckg_vec);
			tuple<int, int> img_size = make_tuple(image.rows, image.cols);
			image_size.push_back(img_size);
		}

	}

	return bckg_loaded;
}

void FrameLoader::SetFrame(session_data &session, frame_data &frame) {
	frame.N_cam = session.N_cam;
	frame.image_size.clear();
	for (int i=0; i<session.N_cam; i++) {
		frame.image_size.push_back(image_size[i]);
	}
}

bool FrameLoader::LoadFrame(session_data &session, frame_data &frame, int MovNR, int FrameNR) {
	bool success;
	try {
		frame.mov_nr = MovNR;
		frame.frame_nr = FrameNR;
		frame.raw_frame.clear();
		for (int i=0; i<frame.N_cam; i++) {
			string img_name = 	session.session_loc+"/"+session.mov_names[MovNR]+"/"+session.cam_names[i]+"/"+
								session.frame_name+to_string(session.chrono_frames[FrameNR])+"."+session.frame_img_format;
			cv::Mat image = cv::imread(img_name, cv::IMREAD_GRAYSCALE);
			if (image.empty()) {
				success = false;
			}
			else {
				arma::Col<int> img_vec = FrameLoader::CVMat2Vector(image);
				frame.raw_frame.push_back(FrameLoader::BackgroundSubtract(img_vec,bckg_images[i]));
			}
		}
	}
	catch(...) {
		success = false;
	}
	return success;
}

arma::Col<int> FrameLoader::BackgroundSubtract(arma::Col<int> &img_vec, arma::Col<int> &bckg_vec) {
	arma::Col<int> s255;
	s255.set_size(img_vec.n_rows);
	s255.fill(255);
	arma::Col<int> sub_img = img_vec-(bckg_vec-s255);
	return sub_img;
}

np::ndarray FrameLoader::ReturnFrame(session_data &session, frame_data &frame, int CamNR) {

	int N_row = get<0>(image_size[CamNR]);
	int N_col = get<1>(image_size[CamNR]);

	p::tuple shape = p::make_tuple(N_row,N_col);
	np::dtype dtype = np::dtype::get_builtin<int>();
	np::ndarray array_out = np::zeros(shape,dtype);

	for (int i=0; i<N_row; i++) {
		for (int j=0; j<N_col; j++) {
			array_out[i][j] = frame.raw_frame[CamNR](i*N_row+j);
		}
	}

	return array_out;
}

arma::Col<int> FrameLoader::CVMat2Vector(cv::Mat &img) {

	arma::Col<int> img_vec;

	img.convertTo(img, CV_8UC1);
	arma::Mat<uint8_t> img_mat(reinterpret_cast<uint8_t*>(img.data), img.rows, img.cols);
	img_vec = arma::vectorise(arma::conv_to<arma::Mat<int>>::from(img_mat));

	return img_vec;
}

cv::Mat FrameLoader::Vector2CVMat(arma::Col<int> &img_vec, int N_row, int N_col) {

	arma::Mat<int> img_mat;

	img_mat = img_vec;

	img_mat.reshape(N_row,N_col);

	cv::Mat img(img_mat.n_rows, img_mat.n_cols, CV_32SC1, img_mat.memptr());

	img.convertTo(img, CV_8UC1);

	return img;
}