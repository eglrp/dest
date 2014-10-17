#include <cstdlib>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <complex>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/flann/flann.hpp"
#include "precomp.hpp"
#include "ceres/ceres.h"
#include "glog/logging.h"
#include "ceres/types.h"
#include "ceres/rotation.h"
#include "gflags/gflags.h"
#include "precomp.hpp"
#include "_modelest.h"

#include "DataStructure.h"
#include "ImagePro.h"
#include "SiftGPU\src\SiftGPU\SiftGPU.h"

using namespace cv;

#if !defined(GEOMETRY_H )
#define GEOMETRY_H

void FishEyeUndistortionPoint(double *K, double* invK, double omega, Point2d *Points, int npts); /////////////////////////////calib.txt
void FishEyeUndistortionPoint(double omega, double DistCtrX, double DistCtrY, Point2d *Points, int npts); /////////////////////////////calib_new.txt

void FishEyeDistortionPoint(double omega, double DistCtrX, double DistCtrY, Point2d *Points, int npts);
void FishEyeDistortionPoint(double *K, double* invK, double omega, Point2d *Points, int npts);

void FishEyeUndistortion(unsigned char *Img, int width, int height, int nchannels, double omega, double DistCtrX, double DistCtrY,int intepAlgo, double ImgMag, double Contscale, double *Para = NULL);
void FishEyeUndistortion(unsigned char *Img, int width, int height, int nchannels, double *K, double* invK, double omega, int intepAlgo, double ImgMag, double Contscale, double *Para = NULL);

void LensDistortionPoint(Point2d *img_point, double *camera, double *distortion, int npts = 1);
void LensCorrectionPoint(Point2d *uv, double *camera, double *distortion, int npts = 1); 
void LensUndistortion(unsigned char *Img, int width, int height, int nchannels, double *K, double *distortion, int intepAlgo, double ImgMag, double Contscale, double *Para = NULL);

void computeFmatfromKRT(CameraData *CameraInfo, int nvews, int *selectedCams, double *Fmat);

int GeneratePointsCorrespondenceMatrix(char *Path, int nviews, int timeID);
void GenerateViewCorrespondenceMatrix(char *Path, int nviews, int timeID);
int ExtractSiftGPUfromVideoFrames(char *Path, int nviews, int start, int nframes);
int BatchExtractSiftGPU(char *Path, int viewID, int TimeLength);
int GeneratePointsCorrespondenceMatrix_SiftGPU1(char *Path, int nviews, int timeID, float nndrRatio = 0.8, bool distortionCorrected = true, int OulierRemoveTestMethod = 1, int nCams = 10, int cameraToScan = 1);
int GeneratePointsCorrespondenceMatrix_SiftGPU2(char *Path, int nviews, int timeID, float nndrRatio = 0.8, bool distortionCorrected = true, int OulierRemoveTestMethod = 1, int nCams = 10, int cameraToScan = 1);
void GenerateMatchingTable(char *Path, int nviews, int timeID);
void BestPairFinder(char *Path, int nviews, int timeID, int &viewPair);
int NextViewFinder(char *Path, int nviews, int timeID, int currentView, int &maxPoints, vector<int> usedPairs);

int GetPoint2DPairCorrespondence(char *Path, int timeID, vector<int>viewID, vector<KeyPoint>&keypoints1, vector<KeyPoint>&keypoints2, vector<int>&CorrespondencesID);
int GetPoint3D2DPairCorrespondence(char *Path, int nviews, int timeID, vector<int> cumulativePts, vector<int> viewID, Point3d *ThreeD, vector<KeyPoint>&keypoints1, vector<KeyPoint>&keypoints2, vector<int>&TwoDCorrespondencesID, vector<int> &ThreeDCorrespondencesID, vector<int>&SelectedIndex, bool SwapView);
int GetPoint3D2DAllCorrespondence(char *Path, int nviews, int timeID, vector<int> cumulativePts, Point3d *ThreeD, vector<int> AvailViews, vector<int>&Selected3DIndex, vector<Point2d> *selected2D, vector<int>*nselectedviews, int &nselectedPts);

Mat findEssentialMat(InputArray points1, InputArray points2, Mat K1, Mat K2, int method = CV_RANSAC, double prob = 0.999, double threshold = 1, int maxIters = 100, OutputArray mask = noArray());
void decomposeEssentialMat( const Mat & E, Mat & R1, Mat & R2, Mat & t ); 
int recoverPose( const Mat & E, InputArray points1, InputArray points2, Mat & R, Mat & t,  Mat K1, Mat K2, InputOutputArray mask = noArray()); 
int EssentialMatOutliersRemove(char *Path, int timeID, int id1, int id2, int nCams, int cameraToScan = -1, int ninlierThresh = 40, bool distortionCorrected = true, bool needDuplicateRemove = false);

void ProjectandDistort(Point3d WC, Point2d *pts, double *P, double *camera = NULL, double *distortion = NULL, int nviews = 1);
void Stereo_Triangulation(Point2d *pts1, Point2d *pts2, double *P1, double *P2, Point3d *WC, int npts = 1);
void TwoViewTriangulationQualityCheck(Point2d *pts1, Point2d *pts2, Point3d *WC, double *P1, double *P2, double *K1, double *K2, double *distortion1, double *distortion2, bool *GoodPoints, int npts, double thresh);
void NviewTriangulation(Point2d *pts, double *P, Point3d *WC, int nview, int npts, double *Cov, double *A, double *B);
void NviewTriangulationRANSAC(Point2d *pts, double *P, Point3d *WC, bool *PassedTri, vector<int> *Inliers, int nview, int npts, int MaxRanSacIter, double inlierPercent, double threshold, double *A = NULL, double *B = NULL, double *tP = NULL);
void NviewTriangulationRANSAC(vector<Point2d> *pts, double *P, Point3d *WC, int nview, int npts, int MaxRanSacIter, double inlierPercent, double threshold, double *A = NULL, double *B = NULL);
int MultiViewQualityCheck(Point2d *Pts, double *Pmat, double *K, double *distortion, bool *PassedPoints, int nviews, int npts, double thresh, Point2d *apts, Point2d *bkapts, int *DeviceMask, double *tK, double *tdistortion, double *tP, double *A, double *B);

int TwoCameraReconstruction(char *Path, CameraData *AllViewsParas, int nviews, int timeID, vector<int> cumulativePts, vector<int> AvailViews, Point3d *ThreeD);
void DetermineDevicePose(double *K, double *distortion, int LensModel, double *R, double *T, Point2d *pts, Point3d *ThreeD, int npts, double thresh, int &ninliers);
int AddNewViewReconstruction(char *Path, CameraData *AllViewsParas, int nviews, int timeID, vector<int> cumulativePts, Point3d *ThreeD, double threshold, vector<int> &availViews);
int IncrementalBA(char *Path, int nviews, int timeID, CameraData *AllViewsParas, vector<int> AvailViews, vector<int> Selected3DIndex, Point3d *All3D, vector<Point2d> *selected2D, vector<int>*nSelectedViews, int nSelectedPts, int totalPts, bool fixSkew, bool fixIntrinsic, bool fixDistortion, bool debug);
void IncrementalBundleAdjustment(char *Path, int nviews, int timeID, int maxKeypoints);

int BuildCorpus(char *Path, int nCameras, int CameraToScan, int width, int height, bool distortionCorrected, int NDplus = 5);
int PoseBA(char *Path, CameraData &camera, vector<Point3d>  Vxyz, vector<Point2d> uvAll3D, vector<bool> &Good, bool fixIntrinsic, bool fixDistortion, bool debug);
int LocalizeCameraFromCorpus(char *Path, Corpus CorpusData, CameraData  &cameraParas, int cameraID, int timeID, vector<int> CorpusViewToMatch, const float nndrRatio = 0.6f);
int LocalizeCameraFromCorpusDriver(char *Path, int StartTime, int StopTime, int nviews, int selectedView, bool distortionCorrected);

class epnp 
{
public:
	epnp(void);
	~epnp();

	void set_internal_parameters(const double uc, const double vc,
		const double fu, const double fv);

	void set_maximum_number_of_correspondences(const int n);
	void reset_correspondences(void);
	void add_correspondence(const double X, const double Y, const double Z,
		const double u, const double v);

	double compute_pose(double R[3][3], double T[3]);

	void relative_error(double & rot_err, double & transl_err,
		const double Rtrue[3][3], const double ttrue[3],
		const double Rest[3][3], const double test[3]);

	void print_pose(const double R[3][3], const double t[3]);
	double reprojection_error(const double R[3][3], const double t[3]);

private:
	void choose_control_points(void);
	void compute_barycentric_coordinates(void);
	void fill_M(CvMat * M, const int row, const double * alphas, const double u, const double v);
	void compute_ccs(const double * betas, const double * ut);
	void compute_pcs(void);

	void solve_for_sign(void);

	void find_betas_approx_1(const CvMat * L_6x10, const CvMat * Rho, double * betas);
	void find_betas_approx_2(const CvMat * L_6x10, const CvMat * Rho, double * betas);
	void find_betas_approx_3(const CvMat * L_6x10, const CvMat * Rho, double * betas);
	void qr_solve(CvMat * A, CvMat * b, CvMat * X);

	double dot(const double * v1, const double * v2);
	double dist2(const double * p1, const double * p2);

	void compute_rho(double * rho);
	void compute_L_6x10(const double * ut, double * l_6x10);

	void gauss_newton(const CvMat * L_6x10, const CvMat * Rho, double current_betas[4]);
	void compute_A_and_b_gauss_newton(const double * l_6x10, const double * rho,
		double cb[4], CvMat * A, CvMat * b);

	double compute_R_and_t(const double * ut, const double * betas,
		double R[3][3], double t[3]);

	void estimate_R_and_t(double R[3][3], double t[3]);

	void copy_R_and_t(const double R_dst[3][3], const double t_dst[3],
		double R_src[3][3], double t_src[3]);

	void mat_to_quat(const double R[3][3], double q[4]);


	double uc, vc, fu, fv;

	double * pws, *us, *alphas, *pcs;
	int maximum_number_of_correspondences;
	int number_of_correspondences;

	double cws[4][3], ccs[4][3];
	double cws_determinant;
};

void Test(char *Path);
#endif