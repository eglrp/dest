#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <vl/generic.h>
#include <vl/mathop.h>
#include <vl/sift.h>
#include <vl/covdet.h>


using namespace cv;
using namespace std;

#if !defined( DATASTRUCTURE_H )
#define DATASTRUCTURE_H

#define MaxNFeatures 100000
#define SIFTBINS 128
#define FISHEYE  -1
#define RADIAL_TANGENTIAL_PRISM  0
#define VisSFMLens 1
#define LUT  2
#define LIMIT3D 1e-6
#define Pi 3.1415926535897932
#define MaxnFrames 601
#define MaxnCams 10000
#define MaxnTrajectories 100000
#define MaxSharedIntrinsicGroup 20
#define OPTICALFLOW_BIDIRECT_DIST_THRESH 3.0

#define MOTION_TRANSLATION 0
#define MOTION_EUCLIDEAN 1
#define MOTION_AFFINE 2
#define MOTION_HOMOGRAPHY 3


struct SiftFeature
{
	SiftFeature()
	{
		CurrentMaxFeatures = 0, Kpts = 0, Desc = 0;
	}
	int CurrentMaxFeatures;
	double* Kpts;
	vl_uint8* Desc;
};
struct CovFeature
{
	CovFeature()
	{
		Affine = 0, Orientation = 1, CurrentMaxFeatures = 0;

		method = VL_COVDET_METHOD_HARRIS_LAPLACE;
		doubleImage = 1;
		octaveResolution = 3, patchResolution = 15;
		edgeThreshold = 10, peakThreshold = 2e-6, lapPeakThreshold = 0.01;
		patchRelativeExtent = 7.5, patchRelativeSmoothing = 1.0, boundaryMargin = 2.0;

		Kpts = 0;
		Desc = 0;

	}
	int Affine, Orientation, CurrentMaxFeatures;

	VlCovDetMethod method;
	vl_bool doubleImage;
	vl_index octaveResolution, patchResolution;
	double edgeThreshold, peakThreshold, lapPeakThreshold;
	double patchRelativeExtent, patchRelativeSmoothing, boundaryMargin;

	double* Kpts;
	float* Desc;
};

struct FeatureDesc
{
	float desc[128];
};
struct ImgPyr
{
	bool rgb;
	int nscales;
	vector<double *> ImgPyrPara;
	vector<unsigned char *> ImgPyrImg;
	vector<Point2i> wh;
	vector<double> factor;
};
struct LKParameters
{
	LKParameters()	{}

	LKParameters(int hsubset, int nscales, int scaleStep, int DIC_Algo, int InterpAlgo, double Gsigma, int Convergence_Criteria, int  IterMax, int Analysis_Speed, double ZNCCThreshold, double PSSDab_thresh, int DisplacementThresh) :
		hsubset(hsubset), nscales(nscales), scaleStep(scaleStep), DIC_Algo(DIC_Algo), InterpAlgo(InterpAlgo), Gsigma(Gsigma), Convergence_Criteria(Convergence_Criteria), IterMax(IterMax), Analysis_Speed(Analysis_Speed), ZNCCThreshold(ZNCCThreshold), PSSDab_thresh(PSSDab_thresh), DisplacementThresh(DisplacementThresh){}

	//DIC_Algo: 
	//0 epipolar search with translation model
	//1 Affine model with epipolar constraint
	//2 translation model
	//3 Affine model without epipolar constraint
	bool checkZNCC;
	int step, nscales, scaleStep, hsubset, npass, npass2, searchRangeScale, searchRangeScale2, searchRange, DisplacementThresh, Incomplete_Subset_Handling, Convergence_Criteria, Analysis_Speed, IterMax, InterpAlgo, DIC_Algo, EpipEnforce;
	double ZNCCThreshold, PSSDab_thresh, ssigThresh, Gsigma, ProjectorGsigma;
};


struct CameraData
{
	CameraData(){
		valid = false;
		for (int ii = 0; ii < 6; ii++)
			wt[ii] = 0.0;
		for (int ii = 0; ii < 9; ii++)
			R[ii] = 0.0, K[ii] = 0.0;
		for (int ii = 0; ii < 6; ii++)
			rt[ii] = 0.0;
		for (int ii = 0; ii < 5; ii++)
			intrinsic[ii] = 0.0;
	}

	double K[9], distortion[7], R[9], Quat[4], T[3], rt[6], wt[6], P[12], intrinsic[5], invK[9], invR[9];
	double Rgl[16], camCenter[3];
	int LensModel;
	int ShutterModel; //0: Global, 1: Cayley, 2: Spline
	double threshold, ninlierThresh;
	std::string filename;
	int nviews, width, height;
	bool notCalibrated, valid;
};

struct Corpus
{
	int nCameras, n3dPoints;
	vector<int> IDCumView;
	vector<string> filenames;
	CameraData *camera;

	vector<Point3d>  xyz;
	vector<Point3i >  rgb;
	vector <vector<int> > viewIdAll3D; //3D -> visiable views index
	vector <vector<int> > pointIdAll3D; //3D -> 2D index in those visible views
	vector<vector<Point2d> > uvAll3D; //3D -> uv of that point in those visible views
	vector<vector<double> > scaleAll3D; //3D -> uv of that point in those visible views
	vector<Mat>DescAll3D; //desc for all 3d

	vector<Point2d> *uvAllViews; //all views valid 2D points
	vector<double> *scaleAllViews; //all views valid 2D points
	vector<int> *threeDIdAllViews; //2D point in visible view -> 3D index
	vector<FeatureDesc> *DescAllViews;//all views valid desc

	//Mat SiftDesc, SurfDesc;
};
struct CorpusandVideo
{
	int nViewsCorpus, nVideos, startTime, stopTime, CorpusSharedIntrinsics;
	CameraData *CorpusInfo;
	CameraData *VideoInfo;
};
struct VideoData
{
	int nVideos, startTime, stopTime, nframesI;
	CameraData *VideoInfo;
};

struct CamInfo
{
	int frameID;
	float camCenter[3];
	float Rgl[16];
};
struct MultiViewPtEle
{
	vector<int> viewID, frameID;
	vector<Point2d> pt2D;
	Point3d pt3D;
};
struct ImgPtEle
{
	ImgPtEle(){
		pixelSizeToMm = 1.0e3, std2D = 1.0, std3D = -1, scale = 7.0, canonicalScale = 7.0;
	}

	int viewID, frameID, imWidth, imHeight, shutterModel;
	Point2d pt2D;
	Point3d pt3D;
	double ray[3], camcenter[3], d, timeStamp, scale, canonicalScale, std2D, std3D, pixelSizeToMm;
	double K[9], R[9], Quat[4], T[3], P[12], Q[6], u[2];
};

struct XYZD
{
	Point3d xyz;
	double d;
};
struct Pmat
{
	double P[12];
};
struct KMatrix
{
	double K[9];
};
struct CamCenter
{
	double C[3];
};
struct RotMatrix
{
	double R[9];
};
struct Quaternion
{
	double quad[4];
};
struct Track2D
{
	int *frameID;
	Point2d *uv;
	double *ParaX, *ParaY;

	int nf;
};
struct Track3D
{
	double *xyz;
	int *frameID;
	int nf;
};
struct Track4D
{
	double *xyzt;
	int npts;
};

struct PerCamNonRigidTrajectory
{
	vector<Pmat> P;
	vector<KMatrix> K;
	vector<RotMatrix >R;
	vector<Quaternion> Q;
	vector<CamCenter>C;

	Track3D *Track3DInfo;
	Track2D *Track2DInfo;
	Track3D *camcenter;
	Track4D *quaternion;
	double F;
	int npts;
};
struct Trajectory2D
{
	int timeID, nViews;
	vector<int>viewIDs, frameID;
	vector<Point2d> uv;
	vector<float>angle;
};
struct Trajectory3D
{
	double timeID, viewID;
	int frameID;
	vector<int>viewIDs;
	vector<Point2d> uv;
	Point3d WC, STD;
	Point3f rgb;
};
struct TrajectoryData
{
	vector<Point3d> *cpThreeD;
	vector<Point3d> *fThreeD;
	vector<Point3d> *cpNormal;
	vector<Point3d> *fNormal;
	vector<Point2d> *cpTwoD;
	vector<Point2d> *fTwoD;
	vector<vector<int> > *cpVis;
	vector<vector<int> > *fVis;
	int nTrajectories, nViews;
	vector<Trajectory2D> *trajectoryUnit;
};
struct VisualizationManager
{
	int catPointCurrentTime;
	Point3d g_trajactoryCenter;
	vector<Point3d> CorpusPointPosition, CorpusPointPosition2, PointPosition, PointPosition2, PointPosition3;
	vector<Point3f> CorpusPointColor, CorpusPointColor2, PointColor, PointColor2, PointColor3;
	vector<Point3d>PointNormal, PointNormal2, PointNormal3;
	vector<CamInfo> glCorpusCameraInfo, *glCameraPoseInfo, *glCameraPoseInfo2;
	vector<Point3d> *catPointPosition, *catPointPosition2;
	vector<Trajectory3D* > Traject3D;
	vector<int> Track3DLength;
	vector<Trajectory3D* > Traject3D2;
	vector<int> Track3DLength2;
	vector<Trajectory3D* > Traject3D3;
	vector<int> Track3DLength3;
	vector<Trajectory3D* > Traject3D4;
	vector<int> Track3DLength4;
	vector<Trajectory3D* > Traject3D5;
	vector<int> Track3DLength5;
};

struct SurfDesc
{
	float desc[64];
};
struct SiftDesc
{
	float desc[128];
};

#endif 
