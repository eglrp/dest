#if !defined(BUNDLEADJUSTMENT_H )
#define BUNDLEADJUSTMENT_H
#pragma once

#define GLOG_NO_ABBREVIATED_SEVERITIES
#define NOMINMAX

//#include <Windows.h>
#include <stdio.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

#include "ceres/ceres.h"
#include "ceres/types.h"
#include "ceres/rotation.h"
//#include "gflags/gflags.h"
#include "glog/logging.h"

using namespace std;

//options for intrinsic parameters
#define BA_OPT_INTRINSIC_ALL             0 //Default
#define BA_OPT_INTRINSIC_ALL_FIXED       1
#define BA_OPT_INTRINSIC_SKEW_FIXED      2
#define BA_OPT_INTRINSIC_SKEW_ZERO_FIXED 3
#define BA_OPT_INTRINSIC_CENTER_FIXED    4

#define BA_OPT_LOAD_INTRINSICS_BUT_JUST_AS_INITIAL_GUESS 10

//options for lens distortion
#define BA_OPT_LENSDIST_RADIAL_AND_TANGENTIAL 0 //Default
#define BA_OPT_LENSDIST_ALL_FIXED             1
#define BA_OPT_LENSDIST_RADIAL_1ST_ONLY       2
#define BA_OPT_LENSDIST_RADIAL_ONLY           3
//#define BA_OPT_LENSDIST_TANGENTIAL_ZERO       3
//#define BA_OPT_LENSDIST_PRISM_ZERO            4

//#define BA_OPT_CONVERT_1ST_RADIAL_BEFORE_BA   true


//options for extrinsic parameters
#define BA_OPT_EXTRINSIC_ALL        0//Default
#define BA_OPT_EXTRINSIC_ALL_FIXED  1//All extrinsic params are not optimized.
#define BA_OPT_EXTRINSIC_R_FIXED    2
#define BA_OPT_EXTRINSIC_T_FIXED    3


namespace BA
{

	struct CameraData
	{
		string filename;
		int imgWidth;
		int imgHeight;

		//intrinsic
		double FocalLength[2];
		double OpticalCenter[2];
		double Skew;

		//lens distortion
		double Radialfirst;
		double Radialothers[2];
		double Tangential[2];
		double Prism[2];

		//extrinsic
		double AngleAxis[3];
		double Translation[3];

		//constraints
		int opt_intrinsic;
		int opt_lensdistortion;
		int opt_extrinsic;

		//bool convert_1st_radial;

		bool available;//if false, not used in BA
		//int lenstype;

		//point info (used for BA)
		vector< vector<double> >  point2D;//observed image points Nx2 [ptID][2]
		vector<int>               ptID;   //ID for corresponding 3D point
		vector<bool>              inlier; //ptID[i] is used in BA or not

		//point info (used for NVM file output)
		//vector< vector<unsigned char> > rgb;//RGB color of the image points Nx3
		//vector<int>                     fID;//ID for the image points

	};


	struct NVM
	{
		int nCamera;
		vector<string> filenames;
		vector<double> focallength;
		vector< vector<double> > quaternion;
		vector< vector<double> > position;
		vector<double> firstradial;

		map<string, int> filename_id;

		int n3dPoints;
		vector< vector<double> >  xyz;
		vector< vector<int> >     rgb;
		vector<string> measurementinfo;
	};


	struct residualData
	{
		vector< vector<int> >    ID;   //[point, camera] Nx2

		vector< vector<double> > observed_pt;   //[x, y]  Nx2
		vector< vector<double> > reprojected_pt;//[x, y]  Nx2

		vector< vector<double> > error;//[err_x, err_y]  Nx2
		double mean_abs_error[2];//(err_x, err_y)
	};


	template <typename T>
	void PinholeReprojection(
		const T* const FocalLength,
		const T* const OpticalCenter,
		const T* const Skew,
		const T* const Radialfirst,
		const T* const Radialothers,
		const T* const Tangential,
		const T* const Prism,
		const T* const AngleAxis,
		const T* const Translation,
		const T* const point3D,
		T* reprojected2D);

	bool loadNVM(const string filepath, NVM &nvmdata,  int sharedIntrinsics = 0);
	bool loadInitialIntrinsics(const string intrinsicfile, const map<string, int> &filename_id, vector<CameraData> &camera, int sharedIntrinisc = 0);
	bool loadIntrinsics(const string intrinsicfile, vector<CameraData> &camera);
	bool loadExtrinsics(const string extrinsicfile, vector<CameraData> &camera);
	bool loadAllCameraParams(const string cameraparamfile, vector<CameraData> &camera);
	bool initCameraData(const NVM &nvmdata, const string filepath, vector<CameraData> &camera, int sharedIntrinisc = 0);


	void dispCameraParams(const CameraData &camera);

	bool saveAllData(const string path, const vector<CameraData> &camera, const vector< vector<double> > &xyz, const residualData &res, const string prefix, const bool After);
	bool saveCameraAllParams(const string file_fullpath, const string separator, const vector<CameraData> &camera);
	bool saveCameraIntrinsics(const string file_fullpath, const string separator, const vector<CameraData> &camera);
	bool saveCameraExtrinsics(const string file_fullpath, const string separator, const vector<CameraData> &camera);
	bool save3Dpoints(const string file_fullpath, const string separator, const vector< vector<double> > &xyz);
	bool saveReprojectionError(const string file_fullpath, const string separator, const residualData &res, const vector<CameraData> &camera, const int order);

	bool saveNVM(const string path, const string inputnvmname, const vector<CameraData> &camera, const vector< vector<double> > &xyz, const NVM &nvmdata);
	bool saveNVM(const string path, const string outputnvmname, const vector<CameraData> &camera, const vector< vector<double> > &xyz, const vector< vector<double> > &uv);//uv is nPoints x 2*nCameras matrix


	void convertLendsModel(vector<CameraData> &camera, vector< vector<double> > &xyz);

	void calcReprojectionError(const vector<CameraData> &camera, const vector< vector<double> > &xyz, residualData &res);
	void runBundleAdjustment(vector<CameraData> &camera, vector< vector<double> > &xyz, const ceres::Solver::Options &options, ceres::Solver::Summary &summary, const double thresh = 3.0);
	void runBundleAdjustment(vector<CameraData> &camera, vector< vector<double> > &xyz, vector< vector<bool> > &visMap, const double thresh, const ceres::Solver::Options &options, ceres::Solver::Summary &summary);
	void checkOutlier(vector<CameraData> &camera, vector< vector<double> > &xyz, vector< vector<bool> > &visMap, const double thresh);


	void setConstantParams(CameraData &cam, ceres::Problem &problem);

	void setCeresOption(const NVM &nvmdata, ceres::Solver::Options &options);
	void setCeresOption(const int nCameras, ceres::Solver::Options &options);


	void copyIntrinsic(const CameraData &src, CameraData &dst);
	void copyExtrinsic(const CameraData &src, CameraData &dst);

}
#endif
