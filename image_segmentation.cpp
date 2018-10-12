#include "image_segmentation.h"

#include "frame_data.h"

#include <string>
#include <stdint.h>
#include <tuple>
#include <iostream>
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/ximgproc/segmentation.hpp"
#include <opencv2/video/background_segm.hpp>
#include <armadillo>

using namespace cv::ximgproc::segmentation;
using namespace std;

ImagSegm::ImagSegm() {
	// empty
}

void ImagSegm::SetBodySegmentationParam(int BodyThresh, int BodyBlurSigma, int BodyBlurWindow, double BodySigma, int BodyK, int MinBodySize, double BodyLength, vector<tuple<double,double>> Origin) {

	body_thresh = BodyThresh;
	body_blur_sigma = BodyBlurSigma;
	body_blur_window = BodyBlurWindow;
	body_sigma = BodySigma;
	body_K = BodyK;
	min_body_size = MinBodySize;
	body_length = BodyLength;
	origin = Origin;

}

void ImagSegm::SetWingSegmentationParam(int WingThresh, int BodyDilation, double WingSigma, int WingK, int MinWingSize, double WingLength) {

	wing_thresh = WingThresh;
	body_dilation = BodyDilation;
	wing_sigma = WingSigma;
	wing_K = WingK;
	min_wing_size = MinWingSize;
	wing_length = WingLength;

}

void ImagSegm::SegmentFrame(frame_data &frame_in) {

	int N_cam = frame_in.N_cam;

	frame_in.seg_frame.clear();
	frame_in.seg_contours.clear();

	for (int n=0; n<N_cam; n++) {

		//arma::Col<int> img_vec = frame_in.raw_frame[n];
		arma::Col<int> img_vec = ImagSegm::ApplyMask(frame_in.raw_frame[n], n);
		int N_row = get<0>(frame_in.image_size[n]);
		int N_col = get<1>(frame_in.image_size[n]);

		// Get body segment:
		arma::Col<int> body_thresh_img = ImagSegm::BodyThresh(img_vec, N_row, N_col, body_thresh, body_blur_sigma, body_blur_window);
		arma::Col<int> body_seg_img = ImagSegm::GraphSeg(body_thresh_img, N_row, N_col, body_sigma, body_K, min_body_size);
		arma::Mat<double> body_seg_prop = ImagSegm::GetSegmentProperties(body_seg_img, N_row, N_col);
		tuple<arma::Col<double>,arma::Col<int>> body_seg = ImagSegm::SelectBody(body_seg_img, N_row, N_col, body_seg_prop, origin[n]);

		arma::Mat<double> frame_contour = ImagSegm::FindBodyAndWingContours(img_vec, get<1>(body_seg), N_row, N_col);
		frame_in.seg_contours.push_back(frame_contour);

		// Get wing segments:
		arma::Col<int> wing_thresh_img = ImagSegm::WingThresh(img_vec, get<1>(body_seg), N_row, N_col, wing_thresh, body_dilation, get<0>(body_seg));
		arma::Col<int> wing_seg_img = ImagSegm::GraphSeg(wing_thresh_img, N_row, N_col, wing_sigma, wing_K, min_wing_size);
		arma::Mat<double> wing_seg_prop = ImagSegm::GetSegmentProperties(wing_seg_img, N_row, N_col);
		tuple<arma::Mat<double>,arma::Mat<int>> wing_seg = ImagSegm::SelectWing(wing_seg_img, N_row, N_col, wing_seg_prop, get<0>(body_seg));

		arma::Col<int> seg_frame = ImagSegm::GlueSegments(get<1>(body_seg), get<1>(wing_seg), img_vec, N_row, N_col);

		// Load segmented frame into the frames struct:
		frame_in.seg_frame.push_back(seg_frame);

	}

}

void ImagSegm::SetImageMask(frame_data &frame_in, int cam_nr, int seg_nr) {

	arma::Col<int> seg_frame = frame_in.seg_frame[cam_nr];
	int N_row = get<0>(frame_in.image_size[cam_nr]);
	int N_col = get<1>(frame_in.image_size[cam_nr]);

	arma::Col<int> mask_vec;

	cv::Mat seg_img, bin_img;
	seg_img = ImagSegm::Vector2CVMat(seg_frame,N_row,N_col);
	cv::inRange(seg_img,seg_nr,seg_nr,bin_img);
	mask_vec = ImagSegm::CVMat2Vector(bin_img);

	arma::Col<int> mask(N_row*N_col);

	image_masks[cam_nr] = image_masks[cam_nr] % (1-(mask_vec/255));
}

arma::Col<int> ImagSegm::ApplyMask(arma::Col<int> &img_in, int cam_nr) {
	arma::Col<int> masked_img = (255-img_in) % image_masks[cam_nr];
	arma::Col<int> img_out = (255-masked_img);
	return img_out;
}

void ImagSegm::ResetImageMask(frame_data &frame_in) {

	image_masks.clear();

	for (int i=0; i<frame_in.N_cam; i++) {
		int N_row = get<0>(frame_in.image_size[i]);
		int N_col = get<1>(frame_in.image_size[i]);
		arma::Col<int> mask;
		mask.ones(N_row*N_col);
		image_masks.push_back(mask);
	}
}

tuple<arma::Col<double>,arma::Col<int>> ImagSegm::SelectBody(arma::Col<int> &frame_in, int N_row, int N_col, arma::Mat<double> seg_prop, tuple<double,double> origin_loc) {

	int N_seg = frame_in.max()+1;
	bool is_body = true;
	tuple<arma::Col<double>, arma::Col<int>> body_img;

	vector<tuple<int,double>> pos_body_segs;

	if (N_seg == 1) {
		cout << "no body segment found" << endl;
		arma::Col<int> body_img_vec(N_row*N_col);
		body_img_vec.zeros();
		arma::Col<double> body_prop_vec(4);
		body_prop_vec.ones();
		body_img = make_tuple((body_prop_vec*-1.0),body_img_vec);
	}
	else {
		for (int i=0; i<N_seg; i++) {
			is_body = true;

			//if (seg_prop(2,i)<(0.1*pow(body_length,2))) {
			//	is_body = false;
			//}
			//if (seg_prop(2,i)>(0.75*pow(body_length,2))) {
			//	is_body = false;
			//}
			//if (seg_prop(3,i)>0) {
			//	is_body = false;
			//}

			if (is_body == true) {
				pos_body_segs.push_back(make_tuple(i,seg_prop(2,i)));
			}
		}

		if (pos_body_segs.size()==1) {
			// Only one candidate for the body segment.
			cv::Mat seg_img, bin_img;
			seg_img = ImagSegm::Vector2CVMat(frame_in,N_row,N_col);
			cv::inRange(seg_img,get<0>(pos_body_segs[0]),get<0>(pos_body_segs[0]),bin_img);
			arma::Col<int> body_img_vec = ImagSegm::CVMat2Vector(bin_img);
			body_img = make_tuple(seg_prop.col(get<0>(pos_body_segs[0])),body_img_vec);
		}
		else if (pos_body_segs.size()>1) {
			// More than one candidate for the body segment, select the segment that is closest to 0.25*pow(body_length,2)
			arma::Col<double> area_diff(pos_body_segs.size());
			for (int j=0; j<pos_body_segs.size(); j++) {
				area_diff(j) = abs(get<1>(pos_body_segs[j])-0.25*pow(body_length,2));
			}
			int min_ind = area_diff.index_min();
			cv::Mat seg_img, bin_img;
			seg_img = ImagSegm::Vector2CVMat(frame_in,N_row,N_col);
			cv::inRange(seg_img,get<0>(pos_body_segs[min_ind]),get<0>(pos_body_segs[min_ind]),bin_img);
			arma::Col<int> body_img_vec = ImagSegm::CVMat2Vector(bin_img);
			body_img = make_tuple(seg_prop.col(min_ind),body_img_vec);
		}
		else {
			// No candidate for the body segment
			arma::Col<int> body_img_vec(N_row*N_col);
			body_img_vec.zeros();
			arma::Col<double> body_prop_vec(4);
			body_prop_vec.ones();
			body_img = make_tuple((body_prop_vec*-1.0),body_img_vec);
		}
	}

	return body_img;	
}

arma::Mat<double> ImagSegm::FindBodyAndWingContours(arma::Col<int> &raw_img_vec, arma::Col<int> &body_img_vec, int N_row, int N_col) {

	cv::Mat img_raw, img_thresh;

	img_raw = ImagSegm::Vector2CVMat(raw_img_vec, N_row, N_col);

	cv::threshold(img_raw, img_thresh, 255-wing_thresh, 255, cv::THRESH_TOZERO_INV);

	vector<vector<cv::Point>> contours_outer;

	cv::findContours(img_thresh, contours_outer, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

	int N_seg = contours_outer.size();

	arma::Row<int> contour_size(N_seg);

	for (int i=0; i<N_seg; i++) {
		contour_size(i) = contours_outer[i].size();
	}

	int max_ind = arma::index_max(contour_size);

	int N_pts = contour_size(max_ind);

	arma::Mat<double> contour_pts(N_pts,2);

	for (int j=0; j<N_pts; j++) {
		contour_pts(j,0) = contours_outer[max_ind][j].x;
		contour_pts(j,1) = contours_outer[max_ind][j].y;
	}

	return contour_pts;
}

tuple<arma::Mat<double>,arma::Mat<int>> ImagSegm::SelectWing(arma::Col<int> &frame_in, int N_row, int N_col, arma::Mat<double> seg_prop, arma::Col<double> body_prop) {

	int N_seg = frame_in.max()+1;
	bool is_wing = true;
	tuple<arma::Mat<double>,arma::Mat<int>> wing_img;

	vector<int> wing_segs;

	if (N_seg == 1) {
		cout << "no wing segment found" << endl;
		arma::Mat<int> wing_img_vec(N_row*N_col,1);
		wing_img_vec.zeros();
		arma::Mat<double> wing_prop_vec(4,1);
		wing_prop_vec.ones();
		wing_img = make_tuple((wing_prop_vec*-1.0),wing_img_vec);
	}
	else {
		for (int i=0; i<N_seg; i++) {
			is_wing = true;

			double dist_cg = sqrt(pow(seg_prop(0,i)-body_prop(0),2)+pow(seg_prop(1,i)-body_prop(1),2));

			//if (dist_cg>(1.5*wing_length)) {
			//	is_wing = false;
			//}
			//if (seg_prop(2,i)>(2.0*pow(wing_length,2))) {
			//	is_wing = false;
			//}
			//if (seg_prop(3,i)>0) {
			//	is_wing = false;
			//}
			
			if (is_wing == true) {
				wing_segs.push_back(i);
			}
		}

		if (wing_segs.size()>0) {
			arma::Mat<int> wing_img_vec(N_row*N_col,wing_segs.size());
			arma::Mat<double> wing_prop_vec(4,wing_segs.size());
			for (int j=0; j<wing_segs.size(); j++) {
				cv::Mat seg_img, bin_img;
				seg_img = ImagSegm::Vector2CVMat(frame_in,N_row,N_col);
				cv::inRange(seg_img,wing_segs[j],wing_segs[j],bin_img);
				wing_img_vec.col(j) = ImagSegm::CVMat2Vector(bin_img);
				wing_prop_vec.col(j) = seg_prop.col(wing_segs[j]);
			}
			wing_img = make_tuple(wing_prop_vec,wing_img_vec);
		}
		else {
			cout << "no wing segment found" << endl;
			arma::Mat<int> wing_img_vec(N_row*N_col,1);
			wing_img_vec.zeros();
			arma::Mat<double> wing_prop_vec(4,1);
			wing_prop_vec.ones();
			wing_img = make_tuple((wing_prop_vec*-1.0),wing_img_vec);
		}
	}

	return wing_img;
}


arma::Col<int> ImagSegm::GlueSegments(arma::Col<int> body_seg, arma::Mat<int> wing_seg, arma::Col<int> raw_img_vec, int N_row, int N_col) {

	int seg_dilation = 15;

	// Dilate the body and wing segments, merge the segments which thouch:
	int N_seg = wing_seg.n_cols;
	int body_seg_max = arma::max(body_seg);
	arma::Col<int> img_out;

	if (body_seg_max > 0) {

		cv::Mat img_body, img_contour, img_i;

		img_body = ImagSegm::Vector2CVMat(body_seg, N_row, N_col);

		cv::Mat RawImg, ThreshImg;

		RawImg = ImagSegm::Vector2CVMat(raw_img_vec,N_row,N_col);

		cv::threshold(RawImg, ThreshImg, 255-wing_thresh, 255, cv::THRESH_TOZERO_INV);

		//cv::threshold(255-RawImg, ThreshImg, wing_thresh, 255, cv::THRESH_TOZERO);

		vector<vector<cv::Point>> contours_outer;

		cv::findContours(ThreshImg, contours_outer, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

		arma::Col<int> img_c_vec;

		if (contours_outer.size() > 0) {

			arma::Row<double> contours_size(contours_outer.size());

			for (int k=0; k<contours_outer.size(); k++) {
				contours_size(k) = contours_outer[k].size();
			}

			int c_max_ind;

			c_max_ind = arma::index_max(contours_size);

			img_contour = cv::Mat::zeros(ThreshImg.size(), CV_8UC1);

			cv::drawContours(img_contour, contours_outer, c_max_ind, cv::Scalar(255), cv::FILLED);

			arma::Col<int> img_contour_vec = arma::clamp(ImagSegm::CVMat2Vector(img_contour),0,1);

			img_c_vec = img_contour_vec;

		}
		else {
			img_c_vec.zeros(N_row*N_col);
		}

		/*
		cv::Mat RawImg, ThreshImg;

		RawImg = ImagSegm::Vector2CVMat(raw_img_vec,N_row,N_col);

		// Perform thresholding:
		cv::threshold(255-RawImg, img_contour, wing_thresh, 255, cv::THRESH_TOZERO);

		arma::Col<int> img_c_vec = arma::clamp(ImagSegm::CVMat2Vector(img_contour),0,1);
		*/

		if (body_dilation > 0) {
			cv::dilate(img_body, img_body, cv::Mat(), cv::Point(-1, -1), body_dilation, 1, 1);
		}

		arma::Col<int> img_b_vec = img_c_vec%arma::clamp(ImagSegm::CVMat2Vector(img_body),0,1);

		if (N_seg > 0) {

			// Iterate through the segments and dilate by 1 pixel at the time:

			arma::Col<int> wing_seg_sum;

			wing_seg_sum.zeros(N_row*N_col);

			for (int i=0; i<N_seg; i++) {

				arma::Col<int> wing_sec_col = wing_seg.col(i);

				img_i = ImagSegm::Vector2CVMat(wing_sec_col, N_row, N_col);

				cv::dilate(img_i, img_i, cv::Mat(), cv::Point(-1, -1), seg_dilation, 1, 1);

				wing_seg_sum += (img_c_vec%arma::clamp(ImagSegm::CVMat2Vector(img_i),0,1))-img_b_vec;

			}

			arma::Col<int> glue_img_vec = arma::clamp(wing_seg_sum,0,1);

			cv::Mat glued_img = ImagSegm::Vector2CVMat(glue_img_vec, N_row, N_col);

			vector<vector<cv::Point>> contours;

			cv::findContours(glued_img, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

			int N_cont = contours.size();

			img_out = img_b_vec;

			int seg_count = 2;

			if (N_cont>0) {
				for (int j=0; j<N_cont; j++) {
					cv::Mat contour_img = cv::Mat::zeros(N_row,N_col, CV_8UC1);
					if (cv::contourArea(contours[j])>min_wing_size) {
						cv::drawContours(contour_img,contours,j,cv::Scalar(seg_count), cv::FILLED);
						img_out += ImagSegm::CVMat2Vector(contour_img);
						seg_count++;
					}
				}
			}

		}
		else {
			img_out = img_b_vec;
		}
	}
	else {
		img_out.zeros(N_col*N_row);
	}
	return img_out;
}

arma::Col<int> ImagSegm::BodyThresh(arma::Col<int> &frame_in, int N_row, int N_col, int body_thresh, int sigma_blur, int blur_window) {

	cv::Mat img, ThreshImg;

	img = ImagSegm::Vector2CVMat(frame_in,N_row,N_col);

	cv::GaussianBlur(img, img, cv::Size(blur_window,blur_window),sigma_blur,sigma_blur);

	cv::threshold(img, ThreshImg, body_thresh, 0, cv::THRESH_TOZERO_INV);

	arma::Col<int> thresh_vec; 

	thresh_vec = ImagSegm::CVMat2Vector(ThreshImg);

	return thresh_vec;
}

arma::Col<int> ImagSegm::WingThresh(arma::Col<int> &frame_in, arma::Col<int> &body_frame_in, int N_row, int N_col, int wing_thresh, int body_dilation, arma::Col<double> body_prop) {

	cv::Mat img, img_coords_body, body_img, circle_mask, ThreshImg;

	img = ImagSegm::Vector2CVMat(frame_in,N_row,N_col);

	body_img = cv::Mat::zeros(N_row, N_col, CV_8UC1 );

	body_img = body_img+ImagSegm::Vector2CVMat(body_frame_in,N_row,N_col);

	// Dilate the image a bit:
	if (body_dilation > 0) {
		cv::dilate(body_img, body_img, cv::Mat(), cv::Point(-1, -1), body_dilation, 1, 1);
	}

	// Perform thresholding:
	cv::threshold(255-(img+body_img), ThreshImg, wing_thresh, 255, cv::THRESH_TOZERO);

	arma::Col<int> thresh_vec;

	thresh_vec = ImagSegm::CVMat2Vector(ThreshImg);

	return thresh_vec;
}

arma::Mat<double> ImagSegm::GetSegmentProperties(arma::Col<int> &frame_in, int N_row, int N_col) {

	int N_seg = frame_in.max()+1;

	arma::Mat<double> seg_prop(4,N_seg);

	cv::Mat seg_img = ImagSegm::Vector2CVMat(frame_in, N_row, N_col);

	for (int i=0; i<N_seg; i++) {

		cv::Mat bin_img;

		cv::inRange(seg_img,i,i,bin_img);

		double minVal;
		double maxVal;
		cv::Point minLoc;
		cv::Point maxLoc;

		cv::minMaxLoc(bin_img, &minVal, &maxVal, &minLoc, &maxLoc);

		double cg_x, cg_y, Area, Mx, My, border_sum;

		// Calculate area

		Area = cv::sum(bin_img)[0]/maxVal;

		Mx = 0.0;
		for (int j=0; j<N_row; j++) {
			Mx+=(cv::sum(bin_img.row(j))[0]*j)/maxVal;
		}

		My = 0.0;
		for (int k=0; k<N_col; k++) {
			My+=(cv::sum(bin_img.col(k))[0]*k)/maxVal;
		}

		// Calculate cg

		cg_x = My/Area;
		cg_y = Mx/Area;

		// Check if the segment hits one of the borders
		border_sum = cv::sum(bin_img.row(0))[0]+cv::sum(bin_img.row(N_row-1))[0]+cv::sum(bin_img.col(0))[0]+cv::sum(bin_img.col(N_col-1))[0];

		seg_prop(0,i) = cg_x;
		seg_prop(1,i) = cg_y;
		seg_prop(2,i) = Area;
		seg_prop(3,i) = border_sum;

	}
	return seg_prop;
}

arma::Col<int> ImagSegm::GraphSeg(arma::Col<int> &frame_in, int N_row, int N_col, double Sigma, int K, int minsize) {

	gs->setSigma(Sigma);
	gs->setK(K);
	gs->setMinSize(minsize);

	cv::Mat input_img, output_img;

	input_img = ImagSegm::Vector2CVMat(frame_in,N_row,N_col);

	gs->processImage(input_img,output_img);

	arma::Col<int> segmented_frame;
	segmented_frame = ImagSegm::CVMat2Vector(output_img);

	return segmented_frame;
}

arma::Mat<double> ImagSegm::FindDoubleContours(arma::Mat<double> &img_mat_in, int N_row, int N_col) {

	arma::Col<int> bin_img_vec = arma::conv_to<arma::Col<int>>::from(arma::clamp(img_mat_in.row(2).t(),0,1)*255);

	cv::Mat ThreshImg;

	ThreshImg = ImagSegm::Vector2CVMat(bin_img_vec,N_row,N_col);

	vector<vector<cv::Point>> contours;

	cv::findContours(ThreshImg, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

	int N_contours = contours.size();

	int N_pts;

	arma::Mat<double> contour_now;

	arma::Mat<double> contour_temp;

	int contour_count = 1;

	int uv_ind;

	cout << N_contours << endl;

	if (N_contours > 0) {
		for (int i=0; i<N_contours; i++) {
			if (i==0) {
				N_pts = contours[i].size();
				contour_now.zeros(3,N_pts+1);
				for (int j=0; j<=N_pts; j++) {
					if (j<N_pts) {
						uv_ind = contours[i][j].y*N_row+contours[i][j].x;
						contour_now(0,j) = img_mat_in(3,uv_ind);
						contour_now(1,j) = img_mat_in(4,uv_ind);
						contour_now(2,j) = contour_count;
					}
					else {
						contour_now(0,j) = contour_now(0,0);
						contour_now(1,j) = contour_now(1,0);
						contour_now(2,j) = contour_now(2,0);
					}
				}
			}
			else {
				N_pts = contours[i].size();
				contour_temp.zeros(3,N_pts+1);
				for (int j=0; j<=N_pts; j++) {
					if (j<N_pts) {
						uv_ind = contours[i][j].y*N_row+contours[i][j].x;
						contour_temp(0,j) = img_mat_in(3,uv_ind);
						contour_temp(1,j) = img_mat_in(4,uv_ind);
						contour_temp(2,j) = contour_count;
					}
					else {
						contour_temp(0,j) = contour_temp(0,0);
						contour_temp(1,j) = contour_temp(1,0);
						contour_temp(2,j) = contour_temp(2,0);
					}
				}
				arma::join_rows(contour_now,contour_temp);
				contour_count++;
			}
		}
	}
	else {
		contour_now.zeros(3,1);
	}
	return contour_now;
}

arma::Mat<double> ImagSegm::FindContours(arma::Col<int> &img_in, int N_row, int N_col) {

	// Clamp the image between 0 and 255
	arma::Col<int> img_thresh = arma::clamp(img_in,0,1)*255;

	cv::Mat ThreshImg;

	ThreshImg = ImagSegm::Vector2CVMat(img_thresh,N_row,N_col);

	vector<vector<cv::Point>> contours;

	cv::findContours(ThreshImg, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

	int N_contours = contours.size();

	int N_pts;

	arma::Mat<double> contour_now;

	arma::Mat<double> contour_temp;

	int contour_count = 1;

	if (N_contours > 0) {
		for (int i=0; i<N_contours; i++) {
			if (i==0) {
				N_pts = contours[i].size();
				contour_now.zeros(3,N_pts+1);
				for (int j=0; j<=N_pts; j++) {
					if (j<N_pts) {
						contour_now(0,j) = contours[i][j].x;
						contour_now(1,j) = contours[i][j].y;
						contour_now(2,j) = contour_count;
					}
					else {
						contour_now(0,j) = contour_now(0,0);
						contour_now(1,j) = contour_now(1,0);
						contour_now(2,j) = contour_now(2,0);
					}
				}
			}
			else {
				if (cv::contourArea(contours[i])>pow(min_wing_size,2)) {
					N_pts = contours[i].size();
					contour_temp.zeros(3,N_pts+1);
					for (int j=0; j<=N_pts; j++) {
						if (j<N_pts) {
							contour_temp(0,j) = contours[i][j].x;
							contour_temp(1,j) = contours[i][j].y;
							contour_temp(2,j) = contour_count;
						}
						else {
							contour_temp(0,j) = contour_temp(0,0);
							contour_temp(1,j) = contour_temp(1,0);
							contour_temp(2,j) = contour_temp(2,0);
						}
					}
					arma::join_rows(contour_now,contour_temp);
					contour_count++;
				}
			}
		}
	}
	else {
		contour_now.zeros(3,1);
	}
	return contour_now;
}

arma::Col<int> ImagSegm::CVMat2Vector(cv::Mat &img) {

	arma::Col<int> img_vec;

	img.convertTo(img, CV_8UC1);
	arma::Mat<uint8_t> img_mat(reinterpret_cast<uint8_t*>(img.data), img.rows, img.cols);
	img_vec = arma::vectorise(arma::conv_to<arma::Mat<int>>::from(img_mat));

	return img_vec;
}

cv::Mat ImagSegm::Vector2CVMat(arma::Col<int> &img_vec, int N_row, int N_col) {

	arma::Mat<int> img_mat;

	img_mat = img_vec;

	img_mat.reshape(N_row,N_col);

	cv::Mat img(img_mat.n_rows, img_mat.n_cols, CV_32SC1, img_mat.memptr());

	img.convertTo(img, CV_8UC1);

	return img;
}

np::ndarray ImagSegm::ReturnSegFrame(frame_data &frame_now, int cam_nr) {

	int N_row = get<0>(frame_now.image_size[cam_nr]);
	int N_col = get<1>(frame_now.image_size[cam_nr]);

	p::tuple shape = p::make_tuple(N_row,N_col);
	np::dtype dtype = np::dtype::get_builtin<int>();
	np::ndarray array_out = np::zeros(shape,dtype);

	for (int i=0; i<N_row; i++) {
		for (int j=0; j<N_col; j++) {
			array_out[i][j] = frame_now.seg_frame[cam_nr](i*N_row+j);
		}
	}

	return array_out;
}

np::ndarray ImagSegm::ReturnOuterContour(frame_data &frame_now, int cam_nr) {

	int N_pts = frame_now.seg_contours[cam_nr].n_rows;

	p::tuple shape = p::make_tuple(N_pts,2);
	np::dtype dtype = np::dtype::get_builtin<int>();
	np::ndarray array_out = np::zeros(shape,dtype);

	for (int i=0; i<N_pts; i++) {
		array_out[i][0] = frame_now.seg_contours[cam_nr](i,0);
		array_out[i][1] = frame_now.seg_contours[cam_nr](i,1);
	}

	return array_out;
}