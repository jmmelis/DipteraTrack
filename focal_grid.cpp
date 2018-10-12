#include "focal_grid.h"

#include <string>
#include <stdint.h>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <chrono>
#include <vector>
#include <map>
#include <unordered_map>
#include <iterator>
#include <future>
#include <thread>
#include <algorithm>
#include <dirent.h>
#include <armadillo>
#include <boost/python/call.hpp>

using namespace std;

FocalGrid::FocalGrid() {
	// empty
}

bool FocalGrid::SetGridParameters(session_data &session, frame_data &frame, int NX, int NY, int NZ, double DS, double X0, double Y0, double Z0) {

	bool grid_loaded = true;

	// Set the focal grid parameters:
	try {

		// grid parameters:
		nx = NX;
		ny = NY;
		nz = NZ;
		ds = DS;
		x0 = X0;
		y0 = Y0;
		z0 = Z0;

		// load calibration file:
		string calib_file = session.session_loc + "/" + session.cal_loc + "/" + session.cal_name;

		arma::Mat<double> CalibMatrix;

		CalibMatrix.load(calib_file);

		// Clear vectors:
		image_size.clear();
		calib_mat.clear();
		X_xyz.clear();
		X_uv.clear();
		uv_offset.clear();

		// Set N_cam
		N_cam = session.N_cam;

		for (int i=0; i<N_cam; i++) {
			image_size.push_back(make_tuple(get<0>(frame.image_size[i]),get<1>(frame.image_size[i])));
			calib_mat.push_back(CalibMatrix.col(i));
			X_xyz.push_back(FocalGrid::Camera2WorldMatrix(CalibMatrix.col(i)));
			X_uv.push_back(FocalGrid::World2CameraMatrix(CalibMatrix.col(i)));
			arma::Col<double> uv_off_i = {CalibMatrix(10,i)/2.0-CalibMatrix(12,i)/2.0, CalibMatrix(9,i)/2.0-CalibMatrix(11,i)/2.0, 0.0};
			uv_offset.push_back(uv_off_i);
		}


	}
	catch (...) {
		cout << "Could not set focal grid parameters." << endl;
		grid_loaded = false;
	}

	return grid_loaded;
}

bool FocalGrid::ConstructFocalGrid(PyObject* progress_func) {

	bool grid_build = true;

	int voxel_ind = 0;

	int progress = 0;

	try {

		pix2vox.clear();
		vox2pix.clear();

		for (int k=0; k<nz; k++) {
			progress = (int) 100.0*(((k+1)*1.0)/(nz*1.0));
			boost::python::call<void>(progress_func,progress);
			for (int j=0; j<ny; j++) {
				for (int i=0; i<nx; i++) {
					vector<int> uv_voxel = FocalGrid::CheckVoxel(i, j, k);
					if (uv_voxel.size()==N_cam) {
						voxel_ind = k*nx*ny+j*nx+i;
						pix2vox.insert(pair<int,int>(uv_voxel[0],voxel_ind));
						vox2pix.insert(pair<int,vector<int>>(voxel_ind,uv_voxel));
					}
				}
			}
		}
		cout << "focal grid size: " << pix2vox.size() << endl;
		cout << "focal grid has been constructed" << endl;
	}
	catch (...) {
		grid_build = false;
		cout << "could not construct focal grid" << endl;
	}

	return grid_build;
}

vector<int> FocalGrid::CheckVoxel(int i, int j, int k) {

	vector<int> uv_out;

	int uv_ind = -1;

	arma::Col<double> xyz(4);

	xyz = {x0-((nx-1)/2.0)*ds+i*ds, 
		y0-((ny-1)/2.0)*ds+j*ds,
		z0-((nz-1)/2.0)*ds+k*ds,
		1.0};

	arma::Col<double> uv(3);

	for (int n=0; n<N_cam; n++) {
		uv = X_uv[n]*xyz-uv_offset[n];
		if (uv(0)>=0.0 && uv(0)<(1.0*get<1>(image_size[n]))) {
			if (uv(1)>=0.0 && uv(1)<(1.0*get<0>(image_size[n]))) {
				uv_ind = ((int) uv(1))*get<1>(image_size[n])+((int) uv(0));
				uv_out.push_back(uv_ind);
			}
		}
	}

	return uv_out;
}

vector<arma::Col<int>> FocalGrid::ProjectCloud2Frames(arma::Mat<double> &cloud_in) {

	vector<arma::Col<int>> frame_now;

	int N_vox = cloud_in.n_cols;

	for (int n=0; n<N_cam; n++) {
		arma::Col<int> frame_n;
		frame_n.zeros(get<0>(image_size[n])*get<1>(image_size[n]));
		frame_now.push_back(frame_n);
	}

	arma::Col<double> uv;
	arma::Col<double> xyz;
	int u = 0;
	int v = 0;

	for (int i=0; i<N_vox; i++) {
		for (int j=0; j<N_cam; j++) {
			xyz = {cloud_in(0,i),cloud_in(1,i),cloud_in(2,i),1.0};
			uv = X_uv[j]*xyz-uv_offset[j];
			if (uv(0)>=0 && uv(0)<(get<1>(image_size[j]))) {
				if (uv(1)>=0 && uv(1)<(get<0>(image_size[j]))) {
					u = (int) uv(0);
					v = (int) uv(1);
					if (frame_now[j](get<1>(image_size[j])*v+u)==0) {
						frame_now[j](get<1>(image_size[j])*v+u) = (int) cloud_in(3,i);
					}
					else {
						if (frame_now[j](get<1>(image_size[j])*v+u) > cloud_in(3,i)) {
							frame_now[j](get<1>(image_size[j])*v+u) = (int) cloud_in(3,i);
						}
					}
				}
			}
		}
	}
	return frame_now;
}

arma::Mat<double> FocalGrid::ProjectCloud2UVdouble(arma::Mat<double> &cloud_in, int cam_nr) {

	int N_vox = cloud_in.n_cols;

	arma::Mat<double> uv_doubles(3,N_vox);

	arma::Mat<double> xyz_coords = cloud_in;
	xyz_coords.row(3).fill(1.0);
	arma::Mat<double> M_mat =  X_uv[cam_nr];
	M_mat.col(3) += -uv_offset[cam_nr];
	uv_doubles = M_mat*xyz_coords;

	return uv_doubles;
}

vector<tuple<double,double,double,int>> FocalGrid::ProjectFrames2Cloud(vector<arma::Col<int>> &frame_in) {

	vector<tuple<double,double,double,int>> pcl_now;

	unordered_map<int,int> pcl_voxels;

	int N_row = get<0>(image_size[0]);
	int N_col = get<1>(image_size[1]);

	pair<multimap<int,int>::iterator, multimap<int,int>::iterator> voxels_i;

	int vox_now;
	vector<int> uv_now;

	int n = 0;
	bool is_voxel = true;

	int frame_val_0 = 0;
	int frame_val_n = 0;

	int code_now = 0;

	int count = 0;

	// Insert voxels into an unordered map and give them a segment code:
	for (int i=0; i<(N_row*N_col); i++) {
		frame_val_0 = frame_in[0](i);
		if (frame_val_0>0) {
			voxels_i = pix2vox.equal_range(i);
			for (multimap<int,int>::iterator it=voxels_i.first; it != voxels_i.second; ++it) {
				vox_now = it->second;
				uv_now = vox2pix[vox_now];
				n = 1;
				is_voxel = true;
				code_now = frame_val_0;
				int body_view_count = 0;
				if (frame_val_0 == 1) {
					body_view_count = 1;
				}
				while (is_voxel==true && n < N_cam) {
					frame_val_n = frame_in[n](uv_now[n]);
					if (frame_val_n>0) {
						code_now = code_now+pow(max_n_seg,n)*frame_val_n;
					}
					else {
						is_voxel = false;
					}
					// Reject voxels which have N_cam-1 views formed by the body:
					if (frame_val_n==1) {
						body_view_count++;
					}
					n++;
				}
				if (body_view_count == (N_cam-1)) {
					is_voxel = false;
				}
				if (is_voxel==true) {
					count++;
					pcl_voxels.insert(pair<int,int>(vox_now,code_now));
				}
			}
		}
	}

	// For each voxel in the pcl_voxels map:
	// -> find the neighboring voxels
	// -> add the voxel to pcl_now if it has more than 1 neighbor and less tan 6 neighbors

	unordered_map<int,int>::iterator nb_it;

	arma::Col<int> neighbors(27);

	int nb_sum;

	arma::Col<double> xyz_pos(3);

	for (nb_it = pcl_voxels.begin(); nb_it != pcl_voxels.end(); nb_it++) {
		vox_now = nb_it->first;
		code_now = nb_it->second;
		neighbors = FocalGrid::FindNeighbors(vox_now);
		if (neighbors(13)>0) {
			nb_sum = 0;
			for (int m=0; m<27; m++) {
				if (pcl_voxels.find(neighbors(m)) != pcl_voxels.end()) {
					// Do nothing
				}
				else {
					nb_sum++;
				}
			}
			if (nb_sum > 2 && nb_sum < 27) {
				xyz_pos = FocalGrid::CalculatePosition(vox_now);
				pcl_now.push_back(make_tuple(xyz_pos(0),xyz_pos(1),xyz_pos(2),code_now));
			}
		}
	}

	if (pcl_now.size() == 0) {
		pcl_now.push_back(make_tuple(0.0,0.0,0.0,0));
	}

	return pcl_now;
}

arma::Mat<int> FocalGrid::FindConnectedPointClouds(arma::Mat<double> &pcl_in) {

	int N_pts = pcl_in.n_cols;

	arma::Row<double> seg_ids = arma::unique(pcl_in.row(3));
	int N_seg = seg_ids.n_cols;

	arma::Mat<int> connectivity_mat;

	// Set the diagonals of the connectivity mat to the unique codes:
	if (N_seg>1) {

		connectivity_mat.zeros(N_seg,N_seg);

		for (int j=0; j<N_seg; j++) {
			connectivity_mat(j,j) = seg_ids(j);
		}

		double central_x;
		double central_y;
		double central_z;
		double central_code;

		arma::uvec dx_ids;
		arma::Mat<double> dx_points;

		arma::uvec dxy_ids;
		arma::Mat<double> dxy_points;

		arma::uvec dxyz_ids;
		arma::Mat<double> dxyz_points;

		arma::Row<double> neighbor_codes;
		int N_codes;

		arma::uvec central_code_ind;
		arma::uvec neighbor_code_ind;

		for (int i=0; i<N_pts; i++) {
			// Iterate through the voxels, find any points which are at (+dx,-dx) subsequently (+dy,-dy) and finally (+dz,-dz).
			// Points which show up in all 3 conditions are neighboring points.
			// Than check the codes of the neighboring points and add scores to the connectivity mat if their codes differ from the central point.
			central_x = pcl_in(0,i);
			central_y = pcl_in(1,i);
			central_z = pcl_in(2,i);
			central_code = pcl_in(3,i);

			dx_ids = arma::find((pcl_in.row(0)>(central_x-(1.1*ds))) && (pcl_in.row(0)<(central_x+(1.1*ds))));
			if (dx_ids.n_rows>1) {
				dx_points = pcl_in.cols(dx_ids);
				dxy_ids = arma::find((dx_points.row(1)>(central_y-(1.1*ds))) && (dx_points.row(1)<(central_y+(1.1*ds))));
				if (dxy_ids.n_rows>1) {
					dxy_points = dx_points.cols(dxy_ids);
					dxyz_ids = arma::find((dxy_points.row(2)>(central_z-(1.1*ds))) && (dxy_points.row(2)<(central_z+(1.1*ds))));
					if (dxyz_ids.n_rows>1) {
						dxyz_points = dxy_points.cols(dxyz_ids);
						neighbor_codes = arma::unique(dxyz_points.row(3));
						N_codes = neighbor_codes.n_cols;
						central_code_ind = arma::find(seg_ids == central_code);
						for (int k=0; k<N_codes; k++) {
							neighbor_code_ind = arma::find(seg_ids == neighbor_codes(k));
							if (central_code_ind(0) != neighbor_code_ind(0)) {
								connectivity_mat(central_code_ind,neighbor_code_ind) += 1;
								connectivity_mat(neighbor_code_ind,central_code_ind) += 1;
							}
						}
					}
				}
			}
		}
	}
	else {
		connectivity_mat.zeros(1,1);
	}
	return connectivity_mat;
}

arma::Mat<double> FocalGrid::ConvertVector2Mat(vector<tuple<double,double,double,int>> pcl_in) {
	int N_pts = pcl_in.size();
	arma::Mat<double> pcl_out(4,N_pts);
	for (int j=0; j<N_pts; j++) {
		pcl_out(0,j) = get<0>(pcl_in[j]);
		pcl_out(1,j) = get<1>(pcl_in[j]);
		pcl_out(2,j) = get<2>(pcl_in[j]);
		pcl_out(3,j) = get<3>(pcl_in[j]);
	}
	return pcl_out;
}

arma::Col<int> FocalGrid::FindNeighbors(int vox_ind) {

	// find the neighboring voxels:

	arma::Col<int> vox_int_mat(27);

	int i = vox_ind % nx;
	int j = ((vox_ind-i)/nx) % ny;
	int k = (vox_ind-i-j*nx)/(nx*ny);

	if ((i>0 && i<(nx-1)) && (j>0 && j<(ny-1)) && (k>0 && k<(nz-1))) {

		vox_int_mat(0) = vox_ind-nx*ny-nx-1;
		vox_int_mat(1) = vox_ind-nx*ny-nx;
		vox_int_mat(2) = vox_ind-nx*ny-nx+1;
		vox_int_mat(3) = vox_ind-nx*ny-1;
		vox_int_mat(4) = vox_ind-nx*ny;
		vox_int_mat(5) = vox_ind-nx*ny+1;
		vox_int_mat(6) = vox_ind-nx*ny+nx-1;
		vox_int_mat(7) = vox_ind-nx*ny+nx;
		vox_int_mat(8) = vox_ind-nx*ny+nx+1;
		vox_int_mat(9) = vox_ind-nx-1;
		vox_int_mat(10) = vox_ind-nx;
		vox_int_mat(11) = vox_ind-nx+1;
		vox_int_mat(12) = vox_ind-1;
		vox_int_mat(13) = vox_ind;
		vox_int_mat(14) = vox_ind+1;
		vox_int_mat(15) = vox_ind+nx-1;
		vox_int_mat(16) = vox_ind+nx;
		vox_int_mat(17) = vox_ind+nx+1;
		vox_int_mat(18) = vox_ind+nx*ny-nx-1;
		vox_int_mat(19) = vox_ind+nx*ny-nx;
		vox_int_mat(20) = vox_ind+nx*ny-nx+1;
		vox_int_mat(21) = vox_ind+nx*ny-1;
		vox_int_mat(22) = vox_ind+nx*ny;
		vox_int_mat(23) = vox_ind+nx*ny+1;
		vox_int_mat(24) = vox_ind+nx*ny+nx-1;
		vox_int_mat(25) = vox_ind+nx*ny+nx;
		vox_int_mat(26) = vox_ind+nx*ny+nx+1;

	}
	else {
		vox_int_mat.zeros();
	}

	return vox_int_mat;
}

arma::Col<double> FocalGrid::CalculatePosition(int vox_ind) {

	int i = vox_ind % nx;
	int j = ((vox_ind-i)/nx) % ny;
	int k = (vox_ind-i-j*nx)/(nx*ny);

	arma::Col<double> xyz_pos(3);

	xyz_pos(0) = x0-((nx-1)/2.0)*ds+i*ds;
	xyz_pos(1) = y0-((ny-1)/2.0)*ds+j*ds;
	xyz_pos(2) = z0-((nz-1)/2.0)*ds+k*ds;

	return xyz_pos;
}

arma::Mat<int> FocalGrid::TransformXYZ2UV(arma::Col<double> xyz_pos) {

	arma::Mat<int> uv_mat(2,N_cam);

	for (int n=0; n<N_cam; n++) {
		arma::Col<double> uv = X_uv[n]*xyz_pos-uv_offset[n];
		uv_mat(0,n) = int(uv(0));
		uv_mat(1,n) = int(uv(1));
	}

	return uv_mat;
}

tuple<arma::Mat<int>, arma::Col<double>> FocalGrid::RayCasting(int cam_nr, arma::Col<double> xyz_pos_prev, arma::Col<double> uv_pos_prev, arma::Col<double> uv_pos_now) {

	// Calculate the 3D translation vector:
	arma::Col<double> xyz_uv_prev = X_xyz[cam_nr]*(uv_pos_prev+uv_offset[cam_nr]);
	arma::Col<double> xyz_uv_now = X_xyz[cam_nr]*(uv_pos_now+uv_offset[cam_nr]);
	arma::Col<double> trans_vec = xyz_uv_now-xyz_uv_prev;
	trans_vec(3) = 0.0;

	// Add the translation to the xyz position:
	arma::Col<double> xyz_pos_now = xyz_pos_prev + trans_vec;

	// Project the new position back to the camera views:

	arma::Mat<int> uv_mat(3,N_cam);

	for (int n=0; n<N_cam; n++) {
		arma::Col<double> uv = X_uv[n]*xyz_pos_now-uv_offset[n];
		if (uv(0)>=0 && uv(0)<(get<1>(image_size[n])) && uv(1)>=0 && uv(1)<(get<0>(image_size[n]))) {
			uv_mat(0,n) = int(uv(0));
			uv_mat(1,n) = int(uv(1));
			uv_mat(2,n) = int(uv(2));
		}
		else {
			arma::Col<double> uv_old = X_uv[n]*xyz_pos_prev-uv_offset[n];
			uv_mat(0,n) = int(uv_old(0));
			uv_mat(1,n) = int(uv_old(1));
			uv_mat(2,n) = int(uv_old(2));
		}
	}

	return make_tuple(uv_mat,xyz_pos_now);
}

arma::Col<double> FocalGrid::CalculateViewVector(int cam_nr) {

	arma::Col<double> u_vec = {1.0,0.0,0.0};
	arma::Col<double> v_vec = {0.0,1.0,0.0};

	arma::Col<double> u_vec_world = X_xyz[cam_nr]*u_vec;
	arma::Col<double> v_vec_world = X_xyz[cam_nr]*v_vec;

	arma::Col<double> view_vec = arma::cross(v_vec_world.rows(0,2),u_vec_world.rows(0,2))/(arma::norm(u_vec_world.rows(0,2))*arma::norm(v_vec_world.rows(0,2)));

	return arma::normalise(view_vec);

}

arma::Mat<double> FocalGrid::Camera2WorldMatrix(arma::Col<double> calib_param) {

	// return the world to camera projection matrix

	arma::Mat<double> C = {{calib_param(0), calib_param(2), 0, 0},
		 			{0, calib_param(1), 0, 0},
		 			{0, 0, 0, 1}};

	double theta = sqrt(pow(calib_param(3),2)+pow(calib_param(4),2)+pow(calib_param(5),2));

	arma::Mat<double> omega = {{0, -calib_param(5), calib_param(4)},
			 			{calib_param(5), 0, -calib_param(3)},
			 			{-calib_param(4), calib_param(3), 0}};

	arma::Mat<double> R(3,3); R.eye();

	R = R+(sin(theta)/theta)*omega+((1-cos(theta))/pow(theta,2))*(omega*omega);

	arma::Col<double> T = {calib_param(6), calib_param(7), calib_param(8)};

	arma::Mat<double> K = {{R(0,0), R(0,1), R(0,2), T(0)},
		 				{R(1,0), R(1,1), R(1,2), T(1)},
		 				{R(2,0), R(2,1), R(2,2), T(2)},
		 				{0, 0, 0, 1}};

	return arma::inv(K)*arma::pinv(C);

}

arma::Mat<double> FocalGrid::World2CameraMatrix(arma::Col<double> calib_param) {

	// return the world to camera projection matrix

	arma::Mat<double> C = {{calib_param(0), calib_param(2), 0, 0},
		 			{0, calib_param(1), 0, 0},
		 			{0, 0, 0, 1}};

	double theta = sqrt(pow(calib_param(3),2)+pow(calib_param(4),2)+pow(calib_param(5),2));

	arma::Mat<double> omega = {{0, -calib_param(5), calib_param(4)},
			 			{calib_param(5), 0, -calib_param(3)},
			 			{-calib_param(4), calib_param(3), 0}};

	arma::Mat<double> R(3,3); R.eye();

	R = R+(sin(theta)/theta)*omega+((1-cos(theta))/pow(theta,2))*(omega*omega);

	arma::Col<double> T = {calib_param(6), calib_param(7), calib_param(8)};

	arma::Mat<double> K = {{R(0,0), R(0,1), R(0,2), T(0)},
		 				{R(1,0), R(1,1), R(1,2), T(1)},
		 				{R(2,0), R(2,1), R(2,2), T(2)},
		 				{0, 0, 0, 1}};

	return C*K;

}