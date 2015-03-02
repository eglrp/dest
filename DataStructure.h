#include <cstdlib>
#include <opencv\cv.hpp>
#include "GL\glut.h"


using namespace cv;
using namespace std;

#if !defined( DATASTRUCTURE_H )
#define DATASTRUCTURE_H

#define MAXSIFTPTS 15000
#define SIFTBINS 128
#define FISHEYE  -1
#define RADIAL_TANGENTIAL_PRISM  0
#define VisSFMLens 1
#define LUT  2
#define LIMIT3D 1e-6
#define Pi 3.1415926535897932
#define MaxnFrame 3000

struct LKParameters
{
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
	double K[9], distortion[7], R[9], T[3], rt[6], P[12], intrinsic[5], invK[9], invR[9];
	double Rgl[16], camCenter[3];
	int LensModel;
	double threshold, ninlierThresh;
	std::string filename;
	int nviews, width, height;
	bool notCalibrated;
};

struct SurfDesc
{
	float desc[64];
};
struct SiftDesc
{
	float desc[128];
};

struct Corpus
{
	int nCamera, n3dPoints;
	vector<int> IDCumView;
	vector<string> filenames;
	CameraData *camera;

	vector<Point3d>  xyz;
	vector<Point3i >  rgb;
	vector < vector<int>> viewIdAll3D; //3D -> visiable views index
	vector < vector<int>> pointIdAll3D; //3D -> 2D index in those visible views
	vector<vector<Point2d>> uvAll3D; //3D -> uv of that point in those visible views
	vector < vector<int>>threeDIdAllViews; //2D point in visible view -> 3D index
	vector < vector<int>> orgtwoDIdAll3D; //visualsfm output order
	vector<vector<Point2d>> uvAllViews;
	Mat SiftDesc, SurfDesc;
};

struct CorpusandVideo
{
	int nViewsCorpus, nVideos, startTime, stopTime, CorpusSharedIntrinsics;
	CameraData *CorpusInfo;
	CameraData *VideoInfo;
};
struct VideoData
{
	int nVideos, startTime, stopTime;
	CameraData *VideoInfo;
};

struct CamInfo
{
	float camCenter[3];
	GLfloat Rgl[16];
};
struct Trajectory2D
{
	int timeID, nViews;
	vector<int>viewIDs;
	vector<Point2d> uv;
	vector<float>angle;
};

struct ImgPtEle
{
	Point2d pt2D;
	Point3d pt3D;
	double ray[3], C[3], d;
	double K[9], R[9], P[12], Q[6], u[2];//Jack notation
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
struct Track3D
{
	double *xyz;
	int npts;
};
struct Track4D
{
	double *xyzt;
	int npts;
};
struct Track2D
{
	double *ParaX, *ParaY;
	Point2d *uv;
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
	Track3D *CamCenter;
	Track4D *quaternion;
	double F;
	int nTracks;
};
struct Trajectory3D
{
	int timeID;
	vector<int>viewIDs;
	vector<Point2d> uv;
	Point3d WC;
};

struct TrajectoryData
{
	vector<Point3d> *cpThreeD;
	vector<Point3d> *fThreeD;
	vector<Point3d> *cpNormal;
	vector<Point3d> *fNormal;
	vector<Point2d> *cpTwoD;
	vector<Point2d> *fTwoD;
	vector<vector<int>> *cpVis;
	vector<vector<int>> *fVis;
	int nTrajectories, nViews;
	vector<Trajectory2D> *trajectoryUnit;
};

struct VisualizationManager
{
	Point3d g_trajactoryCenter;
	vector<Point3d> PointPosition, PointPosition2, PointPosition3;
	vector<Point3f> PointColor, PointColor2, PointColor3;
	vector<Point3d>PointNormal, PointNormal2, PointNormal3;
	vector<CamInfo> glCameraInfo;
	vector<CamInfo> *glCameraPoseInfo;
	vector<Point3d> *catPointPosition, *catPointPosition3;
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



#endif 