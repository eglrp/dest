#if !defined(GEOMETRY_H )
#define GEOMETRY_H

#include <cstdlib>
#include <vector>
#include <iostream>
#include <complex>
#include <omp.h>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include <ceres/ceres.h>
#include <ceres/types.h>
#include <ceres/rotation.h>
#include "glog/logging.h"

#include "DataStructure.h"
#include "ImagePro.h"

#include "_modelest.h"
#include "USAC.h"
#include "FundamentalMatrixEstimator.h"
#include "HomographyEstimator.h"

#include "SiftGPU/src/SiftGPU/SiftGPU.h"

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif


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
void LensDistortionPoint2(Point2d *img_point, double *Intrinsic, double *distortion, int npts = 1);
void LensCorrectionPoint(vector<Point2f> &uv, double *K, double *distortion);
void LensDistortionPoint(vector<Point2d> &img_point, double *K, double *distortion);
void LensCorrectionPoint(Point2d *uv, double *K, double *distortion, int npts = 1);
void LensCorrectionPoint(vector<Point2d> &uv, double *K, double *distortion);
void LensUndistortion(unsigned char *Img, int width, int height, int nchannels, double *K, double *distortion, int intepAlgo, double ImgMag, double Contscale, double *Para = NULL);

double FmatPointError(double *Fmat, Point2d p1, Point2d p2);
void computeFmat(CameraData Cam1, CameraData Cam2, double *Fmat);
void computeFmatfromKRT(CameraData *CameraInfo, int nvews, int *selectedIDs, double *Fmat);
void computeFmatfromKRT(CorpusandVideo &CorpusandVideoInfo, int *selectedCams, int *seletectedTime, int ChooseCorpusView1, int ChooseCorpusView2, double *Fmat);

int TwoCamerasReconstructionFmat(CameraData *AllViewsInfo, Point2d *PCcorres, Point3d *ThreeD, int *CameraPair, int nCams, int nProjectors, int nPpts);

int USAC_FindFundamentalMatrix(ConfigParamsFund cfg, vector<Point2d> pts1, vector<Point2d>pts2, double *Fmat, vector<int>&InlierIndicator, int &ninliers);
int USAC_FindFundamentalDriver(char *Path, int id1, int id2, int timeID);
int USAC_FindHomography(ConfigParamsHomog cfg, vector<Point2d> pts1, vector<Point2d>pts2, double *Hmat, vector<int>&InlierIndicator, int &ninliers);
int USAC_FindHomographyDriver(char *Path, int id1, int id2, int timeID);

int GeneratePointsCorrespondenceMatrix(char *Path, int nviews, int timeID);
void GenerateViewCorrespondenceMatrix(char *Path, int nviews, int timeID);
int ExtractSiftGPUfromExtractedFrames(char *Path, vector<int> &nviews, int startF, int stopF, int increF, int HistogramEqual = 1);

int SiftGPUPair(char *Path, char *Fname1, char *Fname2, float nndrRatio, int timeID, double density = 0.5, bool flipCoordinate = false);
int GeneratePointsCorrespondenceMatrix_SiftGPU1(char *Path, int nviews, int timeID, float nndrRatio = 0.8, int distortionCorrected = 1, int OulierRemoveTestMethod = 1, int nCams = 10, int cameraToScan = 1);
int GeneratePointsCorrespondenceMatrix_SiftGPU2(char *Path, int nviews, int timeID, int HistogramEqual, float nndrRatio = 0.8, int *FrameOffset = NULL);

void BestPairFinder(char *Path, int nviews, int timeID, int &viewPair);
int NextViewFinder(char *Path, int nviews, int timeID, int currentView, int &maxPoints, vector<int> usedPairs);

int GetPoint2DPairCorrespondence(char *Path, int timeID, vector<int>viewID, vector<KeyPoint>&keypoints1, vector<KeyPoint>&keypoints2, vector<int>&CorrespondencesID);
int GetPoint3D2DPairCorrespondence(char *Path, int nviews, int timeID, vector<int> cumulativePts, vector<int> viewID, Point3d *ThreeD, vector<KeyPoint>&keypoints1, vector<KeyPoint>&keypoints2, vector<int>&TwoDCorrespondencesID, vector<int> &ThreeDCorrespondencesID, vector<int>&SelectedIndex, bool SwapView);
int GetPoint3D2DAllCorrespondence(char *Path, int nviews, int timeID, vector<int> cumulativePts, Point3d *ThreeD, vector<int> AvailViews, vector<int>&Selected3DIndex, vector<Point2d> *selected2D, vector<int>*nselectedviews, int &nselectedPts);

Mat findEssentialMat(InputArray points1, InputArray points2, Mat K1, Mat K2, int method = CV_RANSAC, double prob = 0.999, double threshold = 1, int maxIters = 100, OutputArray mask = noArray());
void decomposeEssentialMat(const Mat & E, Mat & R1, Mat & R2, Mat & t);
int recoverPose(const Mat & E, InputArray points1, InputArray points2, Mat & R, Mat & t, Mat K1, Mat K2, InputOutputArray mask = noArray());
int EssentialMatOutliersRemove(char *Path, int timeID, int id1, int id2, int nCams, int cameraToScan = -1, int ninlierThresh = 40, int distortionCorrected = 0, bool needDuplicateRemove = false);
int FundamentalMatOutliersRemove(char *Path, int timeID, int id1, int id2, int ninlierThresh = 40, int LensType = RADIAL_TANGENTIAL_PRISM, int distortionCorrected = 0, bool needDuplicateRemove = false, int nCams = 1, int cameraToScan = -1, int *FrameOffset = NULL);


void ProjectandDistort(Point3d WC, Point2d *pts, double *P, double *camera = NULL, double *distortion = NULL, int nviews = 1);
void ProjectandDistort(vector<Point3d> WC, Point2d *pts, double *P, double *camera = NULL, double *distortion = NULL, int nviews = 1);
void Stereo_Triangulation(Point2d *pts1, Point2d *pts2, double *P1, double *P2, Point3d *WC, int npts = 1);
void Stereo_Triangulation(vector<Point2d> pts1, vector<Point2d> pts2, double *P1, double *P2, vector<Point3d> &WC);
void TwoViewTriangulationQualityCheck(Point2d *pts1, Point2d *pts2, Point3d *WC, double *P1, double *P2, bool *GoodPoints, double thresh, int npts = 1, double *K1 = 0, double *K2 = 0, double *distortion1 = 0, double *distortion2 = 0);
void NviewTriangulation(Point2d *pts, double *P, Point3d *WC, int nview, int npts, double *Cov, double *A, double *B);
void NviewTriangulation(vector<Point2d> *pts, double *P, Point3d *WC, int nview, int npts, double *Cov, double *A, double *B);
void NviewTriangulation(CameraData *ViewInfo, int AvailViews, vector <vector<int> > &viewIdAll3D, vector<vector<Point2d> > &uvAll3D, vector<Point3d> &AllP3D, bool CayleyRS = false);
double NviewTriangulationRANSAC(Point2d *pts, double *P, Point3d *WC, bool *PassedTri, vector<int> *Inliers, int nview, int npts, int MaxRanSacIter, double inlierPercent, double threshold, double *A = NULL, double *B = NULL, double *tP = NULL, bool nonlinear = false, bool refineRanSac = false);
double NviewTriangulationRANSAC(vector<Point2d> *pts, double *P, Point3d *WC, bool *PassedTri, vector<int> *Inliers, int nview, int npts, int MaxRanSacIter, double inlierPercent, double threshold, double *A = NULL, double *B = NULL, double *tP = NULL, bool nonlinear = false, bool refineRanSac = false);
void NviewTriangulationNonLinear(double *P, double *Point2D, double *Point3D, double *ReprojectionError, int nviews, int npts = 1);
void NviewTriangulationNonLinearCayley(CameraData *camInfo, double *Point2D, double *Point3D, double *ReprojectionError, int nviews, int npts);
void MultiViewQualityCheck(Point2d *Pts, double *Pmat, int LensType, double *K, double *distortion, bool *PassedPoints, int nviews, int npts, double thresh, Point3d *aWC, Point2d *apts = 0, Point2d *bkapts = 0, int *DeviceMask = 0, double *tK = 0, double *tdistortion = 0, double *tP = 0, double *A = 0, double *B = 0);
double MinDistanceTwoLines(double *P0, double *u, double *Q0, double *v, double &s, double &t);

int TwoCameraReconstruction(char *Path, CameraData *AllViewsParas, int nviews, int timeID, vector<int> cumulativePts, vector<int> AvailViews, Point3d *ThreeD);
int TwoViewsClean3DReconstructionFmat(CameraData &View1, CameraData &View2, vector<Point2d>imgpts1, vector<Point2d> imgpts2, vector<Point3d> &P3D);
void DetermineDevicePose(double *K, double *distortion, int LensModel, double *R, double *T, Point2d *pts, Point3d *ThreeD, int npts, int distortionCorrected, double thresh, int &ninliers);
void DetermineDevicePose(double *K, double *distortion, int LensModel, double *R, double *T, vector<Point2d> pts, vector<Point3d> ThreeD, int distortionCorrected, double thresh, int &ninliers, bool directMethod = false);
int AddNewViewReconstruction(char *Path, CameraData *AllViewsParas, int nviews, int timeID, vector<int> cumulativePts, Point3d *ThreeD, double threshold, vector<int> &availViews);
int IncrementalBA(char *Path, int nviews, int timeID, CameraData *AllViewsParas, vector<int> AvailViews, vector<int> Selected3DIndex, Point3d *All3D, vector<Point2d> *selected2D, vector<int>*nSelectedViews, int nSelectedPts, int totalPts, bool fixSkew, bool fixIntrinsic, bool fixDistortion, bool debug);

double PinholeReprojectionErrorSimpleDebug(double *P, Point3d Point, Point2d uv);
void PinholeReprojectionDebug(double *intrinsic, double* rt, Point2d observed, Point3d Point, double *residuals);
int CayleyReprojectionDebug(double *intrinsic, double* rt, double *wt, Point2d observed, Point3d Point, int width, int height, double *residuals);
int CayleyDistortionReprojectionDebug(double *intrinsic, double* distortion, double* rt, double *wt, Point2d observed, Point3d Point, int width, int height, double *residuals);
void PinholeDistortionReprojectionDebug(double *intrinsic, double* distortion, double* rt, Point2d observed, Point3d Point, double *residuals);
int GlobalShutterBundleAdjustment(char *Path, CameraData *camera, vector<Point3d>  &Vxyz, vector < vector<int> > viewIdAll3D, vector<vector<Point2d> > uvAll3D, vector<vector<double> >scaleAll3D, vector<int> SharedIntrinsicCamID, int nviews, int fixIntrinsicHD, int fixDistortion, int fixPose, int fixFirstCamPose, int fix3D, int distortionCorrected, int LossType, bool debug = false, bool silent = true);
int ProjectionCayLeyReProjection(double *intrinsic, double* distortion, double* rt, double *wt, Point2d &predicted, Point3d Point, int width, int height);
int CayleyRollingShutterBundleAdjustment(char *Path, CameraData *camera, vector<Point3d>  &Vxyz, vector < vector<int> > viewIdAll3D, vector<vector<Point2d> > uvAll3D, vector<vector<double> >scaleAll3D, vector<int> SharedIntrinsicCamID, int nviews, int fixIntrinsicHD, int fixDistortion, int fixPose, int fixFirstCamPose, int fixLocalPose, int fix3D, int distortionCorrected, int LossType, bool debug = false, bool silent = true);

void IncrementalBundleAdjustment(char *Path, int nviews, int timeID, int maxKeypoints);
int GlobalShutterBundleAdjustmentDriver(char *Path, int nViews, int distortionCorrected, vector< int> SharedIntrinsicCamID, int LossType = 0);
int CayleyRollingShutterBundleAdjustmentDriver(char *Path, int nViews, int distortionCorrected, vector< int> SharedIntrinsicCamID, int LossType = 0);

int BuildCorpus(char *Path, int distortionCorrected, int ShutterModel, int SharedIntrinsic = 1, int NDplus = 5, int LossType = 1);
int BuildCorpusVisualSfm(char *Path, int distortionCorrected, int ShutterModel, int SharedIntrinsic = 1, int NDplus = 5, int LossType = 0);
int Build3DFromSyncedImages(char *Path, int nviews, int startTime, int stopTime, int timeStep, int LensType, int distortionCorrected, int NDplus, double Reprojectionthreshold, double DepthThresh, int *FrameOffset = NULL, bool Save2DCorres = false, bool Gen3DPatchFile = false, double Patch_World_Unit = 1.0, bool useRANSAC = true);

int MatchCameraToCorpus(char *Path, Corpus &corpusData, CameraData *camera, int cameraID, int timeID, int distortionCorrected, vector<int> &CorpusViewToMatch, const float nndrRatio = 0.6f, const int ninlierThresh = 40);
int CameraPose_GSBA(char *Path, CameraData &camera, vector<Point3d>  Vxyz, vector<Point2d> &uvAll3D, vector<double>&scaleAll3D, vector<bool> &Good, bool fixIntrinsic, bool fixDistortion, int distortionCorrected, bool debug);
int CameraPose_RSBA(char *Path, CameraData &camera, vector<Point3d>  Vxyz, vector<Point2d> &uvAll3D, vector<double> &scaleAll3D, vector<bool> &Good, bool fixIntrinsic, bool fixDistortion, int distortionCorrected, bool debug);
int EstimateCameraPoseFromCorpus(char *Path, Corpus &corpusData, CameraData  &cameraParas, int cameraID, bool fixedIntrinsc, bool fixedDistortion, int distortionCorrected, int sharedIntriniscOptim, int timeID);
int VideoPose_GSBA(char *Path, int camID, int startTime, int stopTime, int fixedIntrinisc, int fixedDistortion, int fixed3D, int distortionCorrected, double threshold);
int VideoPose_RS_Cayley_BA(char *Path, int camID, int startF, int stopF, int fixIntrinsic, int fixDistortion, int fixPose, int fixfirstCamPose, int fix3D, int distortionCorrected, bool doubleRefinement, double threshold);
int VideoSplineRSBA(char *Path, int startFrame, int stopFrame, int selectedCams, int distortionCorrected, int fixedIntrinisc, int fixedDistortion, double threshold, int controlStep = 5, int SplineOrder = 4, bool se3 = false, bool debug = false);
int VideoDCTRSBA(char *Path, int startFrame, int stopFrame, int selectedCams, int distortionCorrected, int fixedIntrinisc, int fixedDistortion, double threshold, int sampleStep = 5, double lamda = 0.1, bool debug = false);
int LocalizeCameraFromCorpusDriver(char *Path, int StartFrame, int StopFrame, int IncreFrame, int module, int nCams, int selectedCams, int distortionCorrected, int GetIntrinsicFromCorpus, int sharedIntriniscOptim, int LensType);

int BundleAdjustDomeTableCorres(char *Path, int startF_HD, int stopF_HD, int startF_VGA, int stopF_VGA, bool fixIntrinsic, bool fixDistortion, bool fixPose, bool fixIntrinsicVGA, bool fixDistortionVGA, bool fixPoseVGA, bool debug);
int BundleAdjustDomeMultiNVM(char *Path, int nNvm, int maxPtsPerNvM, bool fixIntrinsic, bool fixDistortion, bool fixPose, bool debug);
int ReCalibratedFromGroundTruthCorrespondences(char *Path, int camID, int startFrame, int stopFrame, int Allnpts, int ShutterModel = 0);
int RefineVisualSfM(char *Path, int nimages, int nplus, int ShutterModel = 0, double threshold = 5.0, bool sharedInstrinsic = true, bool fixedIntrinsc = false, bool fixedDistortion = false, bool fixedPose = false, bool fixedfirstCamPose = true, bool distortionCorrected = false, bool doubleRefinement = true);
int RefineVisualSfMAndCreateCorpus(char *Path, int nimages, int nplus, int ShutterModel = 0, double threshold = 5.0, bool sharedInstrinsic = true, bool fixedIntrinsc = false, bool fixedDistortion = false, bool fixedPose = false, bool fixedfirstCamPose = true, bool distortionCorrected = false, bool doubleRefinement = true);

int SparsePointTrackingDriver(char *Path, vector<Point2d> &Tracks, vector<float*> &ImgPara, int viewID, int startF, int stopF, LKParameters LKArg, int &width, int &height, int nchannels);
#endif
