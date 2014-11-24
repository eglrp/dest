#if !defined(GEOMETRY_H )
#define GEOMETRY_H

#include <cstdlib>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <complex>
#include <omp.h>
#include <algorithm>
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
#include "USAC.h"
#include "FundamentalMatrixEstimator.h"
#include "HomographyEstimator.h"

using namespace cv;

/////////////////////////////calib_new.txt
void FishEyeDistortionPoint(Point2d *Points, double omega, double DistCtrX, double DistCtrY, int npts = 1);
void FishEyeCorrectionPoint(Point2d *Points, double omega, double DistCtrX, double DistCtrY, int npts = 1);
void FishEyeDistortionPoint(vector<Point2d> &Points, double omega, double DistCtrX, double DistCtrY);
void FishEyeCorrectionPoint(vector<Point2d> &Points, double omega, double DistCtrX, double DistCtrY);
void FishEyeCorrection(unsigned char *Img, int width, int height, int nchannels, double omega, double DistCtrX, double DistCtrY, int intepAlgo, double ImgMag, double Contscale, double *Para = NULL);

/////////////////////////////calib.txt
void FishEyeCorrectionPoint(Point2d *Points, double *K, double* invK, double omega, int npts = 1);
void FishEyeDistortionPoint(Point2d *Points, double *K, double* invK, double omega, int npts = 1);
void FishEyeCorrection(unsigned char *Img, int width, int height, int nchannels, double *K, double* invK, double omega, int intepAlgo, double ImgMag, double Contscale, double *Para = NULL);

void LensDistortionPoint(Point2d *img_point, double *K, double *distortion, int npts = 1);
void LensDistortionPoint(vector<Point2d> &img_point, double *K, double *distortion);
void LensCorrectionPoint(Point2d *uv, double *K, double *distortion, int npts = 1);
void LensCorrectionPoint(vector<Point2d> &uv, double *K, double *distortion);
void LensUndistortion(unsigned char *Img, int width, int height, int nchannels, double *K, double *distortion, int intepAlgo, double ImgMag, double Contscale, double *Para = NULL);

void computeFmatfromKRT(CameraData *CameraInfo, int nvews, int *selectedIDs, double *Fmat);
void computeFmatfromKRT(CorpusandVideo &CorpusandVideoInfo, int *selectedCams, int *seletectedTime, int ChooseCorpusView1, int ChooseCorpusView2, double *Fmat);

int USAC_FindFundamentalMatrix(ConfigParamsFund cfg, vector<Point2d> pts1, vector<Point2d>pts2, double *Fmat, vector<int>&InlierIndicator, int &ninliers);
int USAC_FindFundamentalDriver(char *Path, int id1, int id2, int timeID);
int USAC_FindHomography(ConfigParamsHomog cfg, vector<Point2d> pts1, vector<Point2d>pts2, double *Hmat, vector<int>&InlierIndicator, int &ninliers);
int USAC_FindHomographyDriver(char *Path, int id1, int id2, int timeID);

int GeneratePointsCorrespondenceMatrix(char *Path, int nviews, int timeID);
void GenerateViewCorrespondenceMatrix(char *Path, int nviews, int timeID);
int ExtractSiftGPUfromExtractedFrames(char *Path, vector<int> nviews, int startF, int stopF, int HistogramEqual = 1);
int GeneratePointsCorrespondenceMatrix_SiftGPU1(char *Path, int nviews, int timeID, float nndrRatio = 0.8, int distortionCorrected = 1, int OulierRemoveTestMethod = 1, int nCams = 10, int cameraToScan = 1);
int GeneratePointsCorrespondenceMatrix_SiftGPU2(char *Path, int nviews, int timeID, int HistogramEqual, float nndrRatio = 0.8, int LensType = RADIAL_TANGENTIAL_PRISM, int distortionCorrected = 1, int OulierRemoveTestMethod = 1, int nCams = 10, int cameraToScan = 1);
void GenerateMatchingTable(char *Path, int nviews, int timeID);
void BestPairFinder(char *Path, int nviews, int timeID, int &viewPair);
int NextViewFinder(char *Path, int nviews, int timeID, int currentView, int &maxPoints, vector<int> usedPairs);

int GetPoint2DPairCorrespondence(char *Path, int timeID, vector<int>viewID, vector<KeyPoint>&keypoints1, vector<KeyPoint>&keypoints2, vector<int>&CorrespondencesID);
int GetPoint3D2DPairCorrespondence(char *Path, int nviews, int timeID, vector<int> cumulativePts, vector<int> viewID, Point3d *ThreeD, vector<KeyPoint>&keypoints1, vector<KeyPoint>&keypoints2, vector<int>&TwoDCorrespondencesID, vector<int> &ThreeDCorrespondencesID, vector<int>&SelectedIndex, bool SwapView);
int GetPoint3D2DAllCorrespondence(char *Path, int nviews, int timeID, vector<int> cumulativePts, Point3d *ThreeD, vector<int> AvailViews, vector<int>&Selected3DIndex, vector<Point2d> *selected2D, vector<int>*nselectedviews, int &nselectedPts);

Mat findEssentialMat(InputArray points1, InputArray points2, Mat K1, Mat K2, int method = CV_RANSAC, double prob = 0.999, double threshold = 1, int maxIters = 100, OutputArray mask = noArray());
void decomposeEssentialMat(const Mat & E, Mat & R1, Mat & R2, Mat & t);
int recoverPose(const Mat & E, InputArray points1, InputArray points2, Mat & R, Mat & t, Mat K1, Mat K2, InputOutputArray mask = noArray());
int EssentialMatOutliersRemove(char *Path, int timeID, int id1, int id2, int nCams, int cameraToScan = -1, int ninlierThresh = 40, int distortionCorrected = 0, bool needDuplicateRemove = false);
int FundamentalMatOutliersRemove(char *Path, int timeID, int id1, int id2, int ninlierThresh = 40, int LensType = RADIAL_TANGENTIAL_PRISM, int distortionCorrected = 0, bool needDuplicateRemove = false, int nCams = 1, int cameraToScan = -1);


void ProjectandDistort(Point3d WC, Point2d *pts, double *P, double *camera = NULL, double *distortion = NULL, int nviews = 1);
void Stereo_Triangulation(Point2d *pts1, Point2d *pts2, double *P1, double *P2, Point3d *WC, int npts = 1);
void TwoViewTriangulationQualityCheck(Point2d *pts1, Point2d *pts2, Point3d *WC, double *P1, double *P2, double *K1, double *K2, double *distortion1, double *distortion2, bool *GoodPoints, int npts, double thresh);
void NviewTriangulation(Point2d *pts, double *P, Point3d *WC, int nview, int npts, double *Cov, double *A, double *B);
void NviewTriangulationRANSAC(Point2d *pts, double *P, Point3d *WC, bool *PassedTri, vector<int> *Inliers, int nview, int npts, int MaxRanSacIter, double inlierPercent, double threshold, double *A = NULL, double *B = NULL, double *tP = NULL);
void NviewTriangulationRANSAC(vector<Point2d> *pts, double *P, Point3d *WC, int nview, int npts, int MaxRanSacIter, double inlierPercent, double threshold, double *A = NULL, double *B = NULL);
void MultiViewQualityCheck(Point2d *Pts, double *Pmat, int LensType, double *K, double *distortion, bool *PassedPoints, int nviews, int npts, double thresh, Point3d *aWC, Point2d *apts = 0, Point2d *bkapts = 0, int *DeviceMask = 0, double *tK = 0, double *tdistortion = 0, double *tP = 0, double *A = 0, double *B = 0);

int TwoCameraReconstruction(char *Path, CameraData *AllViewsParas, int nviews, int timeID, vector<int> cumulativePts, vector<int> AvailViews, Point3d *ThreeD);
void DetermineDevicePose(double *K, double *distortion, int LensModel, double *R, double *T, Point2d *pts, Point3d *ThreeD, int npts, int distortionCorrected, double thresh, int &ninliers);
int AddNewViewReconstruction(char *Path, CameraData *AllViewsParas, int nviews, int timeID, vector<int> cumulativePts, Point3d *ThreeD, double threshold, vector<int> &availViews);
int IncrementalBA(char *Path, int nviews, int timeID, CameraData *AllViewsParas, vector<int> AvailViews, vector<int> Selected3DIndex, Point3d *All3D, vector<Point2d> *selected2D, vector<int>*nSelectedViews, int nSelectedPts, int totalPts, bool fixSkew, bool fixIntrinsic, bool fixDistortion, bool debug);
void IncrementalBundleAdjustment(char *Path, int nviews, int timeID, int maxKeypoints);

int BuildCorpus(char *Path, int nCameras, int CameraToScan, int distortionCorrected, int NDplus = 5);
int Build3DFromSyncedImages(char *Path, int nviews, int startTime, int stopTime, int LensType, int distortionCorrected, double Reprojectionthreshold, bool Gen3DPatchFile = false, double Patch_World_Unit = 1.0);
int PoseBA(char *Path, CameraData &camera, vector<Point3d>  Vxyz, vector<Point2d> uvAll3D, vector<bool> &Good, bool fixIntrinsic, bool fixDistortion, int distortionCorrected, bool debug);

int MatchCameraToCorpus(char *Path, Corpus &corpusData, CameraData *camera, int cameraID, int timeID, int distortionCorrected, vector<int> CorpusViewToMatch, const float nndrRatio = 0.6f, const int ninlierThresh = 40);
int EstimateCameraPoseFromCorpus(char *Path, Corpus corpusData, CameraData  &cameraParas, int cameraID, bool fixedIntrinsc, bool fixedDistortion, int distortionCorrected, int sharedIntriniscOptim, int timeID);
int LocalizeCameraFromCorpusDriver(char *Path, int StartTime, int StopTime, bool RunMatching, int nCams, int selectedCams, int distortionCorrected, int sharedIntriniscOptim, int LensType);

#endif