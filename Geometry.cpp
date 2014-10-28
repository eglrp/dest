#include "Geometry.h"
#include "Ultility.h"
#include "Visualization.h"
#include "Eigen\Sparse"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::CauchyLoss;
using ceres::SoftLOneLoss;
using ceres::HuberLoss;
using ceres::Problem;
using ceres::Solver;

using namespace std;
using namespace cv;
using namespace Eigen;

bool useGPU = true;
//SiftGPU 
#define SIFTGPU_DLL_RUNTIME// Load at runtime if the above macro defined comment the macro above to use static linking
#ifdef _WIN32
#ifdef SIFTGPU_DLL_RUNTIME
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define FREE_MYLIB FreeLibrary
#define GET_MYPROC GetProcAddress
#endif
#else
#ifdef SIFTGPU_DLL_RUNTIME
#include <dlfcn.h>
#define FREE_MYLIB dlclose
#define GET_MYPROC dlsym
#endif
#endif


int siftgpu()
{

#ifdef _WIN32
#ifdef _DEBUG
	HMODULE  hsiftgpu = LoadLibrary("siftgpu_d.dll");
#else
	HMODULE  hsiftgpu = LoadLibrary("siftgpu.dll");
#endif
#else
	void * hsiftgpu = dlopen("libsiftgpu.so", RTLD_LAZY);
#endif

	if (hsiftgpu == NULL)
		return 0;

	SiftGPU* (*pCreateNewSiftGPU)(int) = NULL;
	SiftMatchGPU* (*pCreateNewSiftMatchGPU)(int) = NULL;
	pCreateNewSiftGPU = (SiftGPU* (*) (int)) GET_MYPROC(hsiftgpu, "CreateNewSiftGPU");
	pCreateNewSiftMatchGPU = (SiftMatchGPU* (*)(int)) GET_MYPROC(hsiftgpu, "CreateNewSiftMatchGPU");
	SiftGPU* sift = pCreateNewSiftGPU(1);
	SiftMatchGPU* matcher = pCreateNewSiftMatchGPU(8192);

	vector<float > descriptors1(1), descriptors2(1);
	vector<SiftGPU::SiftKeypoint> keys1(1), keys2(1);
	int num1 = 0, num2 = 0;

	//process parameters
	//The following parameters are default in V340
	//-m,       up to 2 orientations for each feature (change to single orientation by using -m 1)
	//-s        enable subpixel subscale (disable by using -s 0)

	// Allocation size to the largest width and largest height 1920x1080
	// Maximum working dimension. All the SIFT octaves that needs a larger texture size will be skipped. maxd = 2560 <-> 768MB of graphic memory. 
	char * argv[] = { "-fo", "-1", "-v", "1", "-p", "3072x2048", "-maxd", "3200" };


	//-fo -1    staring from -1 octave 
	//-v 1      only print out # feature and overall time
	//-loweo    add a (.5, .5) offset
	//-tc <num> set a soft limit to number of detected features

	//NEW:  parameters for  GPU-selection
	//1. CUDA.                   Use parameter "-cuda", "[device_id]"
	//2. OpenGL.				 Use "-Display", "display_name" to select monitor/GPU (XLIB/GLUT) on windows the display name would be something like \\.\DISPLAY4

	//You use CUDA for nVidia graphic cards by specifying
	//-cuda   : cuda implementation (fastest for smaller images)
	//          CUDA-implementation allows you to create multiple instances for multiple threads. Checkout src\TestWin\MultiThreadSIFT

	////////////////////////Two Important Parameters///////////////////////////
	// Second, there is a parameter you may not be aware of: the allowed maximum working
	// dimension. All the SIFT octaves that needs a larger texture size will be skipped.
	// The default prameter is 2560 for the unpacked implementation and 3200 for the packed.
	// Those two default parameter is tuned to for 768MB of graphic memory. You should adjust
	// it for your own GPU memory. You can also use this to keep/skip the small featuers.
	// To change this, call function SetMaxDimension or use parameter "-maxd".

	// NEW: by default SiftGPU will try to fit the cap of GPU memory, and reduce the working 
	// dimension so as to not allocate too much. This feature can be disabled by -nomc
	//////////////////////////////////////////////////////////////////////////////////////

	int argc = sizeof(argv) / sizeof(char*);
	sift->ParseParam(argc, argv);

	//Only the following parameters can be changed after initialization (by calling ParseParam):-dw, -ofix, -ofix-not, -fo, -unn, -maxd, -b
	//to change other parameters at runtime, you need to first unload the dynamically loaded libaray reload the libarary, then create a new siftgpu instance


	//Create a context for computation, and SiftGPU will be initialized automatically. The same context can be used by SiftMatchGPU
	if (sift->CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED)
		return 0;

	if (sift->RunSIFT("C:/temp/fountain_dense/0001.png"))
	{
		num1 = sift->GetFeatureNum(); //sift->SaveSIFT("C:/temp/fountain_dense/_0000.sift"); //Note that saving Lowe's ASCII format is slow
		keys1.resize(num1);    descriptors1.resize(SIFTBINS * num1);
		sift->GetFeatureVector(&keys1[0], &descriptors1[0]);
	}

	//You can have at most one OpenGL-based SiftGPU (per process)--> no omp can be used
	float u, v, s, o;
	FILE *fp = fopen("C:/temp/k2.txt", "r");
	SiftGPU::SiftKeypoint mykeys[4507];
	for (int i = 0; i < 4507; ++i)
	{
		fscanf(fp, "%f %f %f %f ", &u, &v, &s, &o);
		mykeys[i].s = s;
		mykeys[i].o = o;
		mykeys[i].x = u;
		mykeys[i].y = v;
	}
	sift->SetKeypointList(4507, mykeys, 1);
	sift->RunSIFT("C:/temp/fountain_dense/S_6.png");
	num2 = sift->GetFeatureNum();
	keys2.resize(num2);    descriptors2.resize(SIFTBINS * num2);
	sift->GetFeatureVector(&keys2[0], &descriptors2[0]);

	/*if (sift->RunSIFT("C:/temp/fountain_dense/0001.png"))
	{
	num2 = sift->GetFeatureNum();
	keys2.resize(num2);    descriptors2.resize(SIFTBINS * num2);
	sift->GetFeatureVector(&keys2[0], &descriptors2[0]);
	}*/

	//SiftGPU::SiftKeypoint mykeys[100];
	//for(int i = 0; i < 100; ++i){
	//    mykeys[i].s = 1.0f;mykeys[i].o = 0.0f;
	//    mykeys[i].x = (i%10)*10.0f+50.0f;
	//    mykeys[i].y = (i/10)*10.0f+50.0f;
	//}
	//sift->SetKeypointList(100, mykeys, 0);
	//sift->RunSIFT("../data/800-1.jpg");                    sift->SaveSIFT("../data/800-1.sift.2");


	//**********************GPU SIFT MATCHING*********************************
	//**************************select shader language*************************
	//SiftMatchGPU will use the same shader lanaguage as SiftGPU by default
	//Before initialization, you can choose between glsl, and CUDA(if compiled). 
	//matcher->SetLanguage(SiftMatchGPU::SIFTMATCH_CUDA); // +i for the (i+1)-th device

	//Verify current OpenGL Context and initialize the Matcher. If you don't have an OpenGL Context, call matcher->CreateContextGL instead;
	matcher->VerifyContextGL(); //must call once

	//Set descriptors to match, the first argument must be either 0 or 1
	//if you want to use more than 4096 or less than 4096
	//call matcher->SetMaxSift() to change the limit before calling setdescriptor
	matcher->SetDescriptors(0, num1, &descriptors1[0]); //image 1
	matcher->SetDescriptors(1, num2, &descriptors2[0]); //image 2

	//match and get result.    
	int(*match_buf)[2] = new int[num1][2];
	//use the default thresholds. Check the declaration in SiftGPU.h
	int num_match = matcher->GetSiftMatch(num1, match_buf);
	std::cout << num_match << " sift matches were found;\n";

	//enumerate all the feature matches
	for (int i = 0; i < num_match; ++i)
	{
		//How to get the feature matches: 
		//SiftGPU::SiftKeypoint & key1 = keys1[match_buf[i][0]];
		//SiftGPU::SiftKeypoint & key2 = keys2[match_buf[i][1]];
		//key1 in the first image matches with key2 in the second image
	}


	fp = fopen("C:/temp/corres.txt", "w+");
	KeyPoint key;
	vector<int> CorresID;
	vector<KeyPoint> keypoints1, keypoints2;
	for (int i = 0; i < num_match; ++i)
	{
		SiftGPU::SiftKeypoint & key1 = keys1[match_buf[i][0]];
		SiftGPU::SiftKeypoint & key2 = keys2[match_buf[i][1]];
		key.pt.x = key1.x, key.pt.y = key1.y; keypoints1.push_back(key);
		key.pt.x = key2.x, key.pt.y = key2.y; keypoints2.push_back(key);
		CorresID.push_back(i), CorresID.push_back(i);
		fprintf(fp, "%.2f %.2f %.2f %.2f\n", keypoints1.at(i).pt.x, keypoints1.at(i).pt.y, keypoints2.at(i).pt.x, keypoints2.at(i).pt.y);
	}
	fclose(fp);

	int nchannels = 3;
	IplImage *Img1 = cvLoadImage("C:/temp/fountain_dense/0001.png", nchannels == 3 ? 1 : 0);
	if (Img1->imageData == NULL)
	{
		printf("Cannot load image 1\n");
		return 1;
	}
	IplImage *Img2 = cvLoadImage("C:/temp/fountain_dense/S_6.png", nchannels == 3 ? 1 : 0);
	if (Img2->imageData == NULL)
	{
		printf("Cannot load image 2\n");
		return 1;
	}

	IplImage* correspond = cvCreateImage(cvSize(Img1->width + Img2->width, Img1->height), 8, nchannels);
	cvSetImageROI(correspond, cvRect(0, 0, Img1->width, Img1->height));
	cvCopy(Img1, correspond);
	cvSetImageROI(correspond, cvRect(Img1->width, 0, correspond->width, correspond->height));
	cvCopy(Img2, correspond);
	cvResetImageROI(correspond);
	DisplayImageCorrespondence(correspond, Img1->width, 0, keypoints1, keypoints2, CorresID, 1.0);

	delete[] match_buf;
	delete sift;
	delete matcher;
	FREE_MYLIB(hsiftgpu);

	return 0;
}
class CvEMEstimator : public CvModelEstimator2
{
public:
	CvEMEstimator();
	virtual int runKernel(const CvMat* m1, const CvMat* m2, CvMat* model);
	virtual int run5Point(const CvMat* _q1, const CvMat* _q2, CvMat* _ematrix);
	//protected: 
	bool reliable(const CvMat* m1, const CvMat* m2, const CvMat* model);
	virtual void getCoeffMat(double *eet, double* a);
	virtual void computeReprojError(const CvMat* m1, const CvMat* m2,
		const CvMat* model, CvMat* error);
};

CvEMEstimator::CvEMEstimator()
	: CvModelEstimator2(5, cvSize(3, 3), 10)
{
}
int CvEMEstimator::runKernel(const CvMat* m1, const CvMat* m2, CvMat* model)
{
	return run5Point(m1, m2, model);
}
// Notice to keep compatibility with opencv ransac, q1 and q2 have to be of 1 row x n col x 2 channel. 
int CvEMEstimator::run5Point(const CvMat* q1, const CvMat* q2, CvMat* ematrix)
{
	Mat Q1 = Mat(q1).reshape(1, q1->cols);
	Mat Q2 = Mat(q2).reshape(1, q2->cols);

	int n = Q1.rows;
	Mat Q(n, 9, CV_64F);
	Q.col(0) = Q1.col(0).mul(Q2.col(0));
	Q.col(1) = Q1.col(1).mul(Q2.col(0));
	Q.col(2) = Q2.col(0) * 1.0;
	Q.col(3) = Q1.col(0).mul(Q2.col(1));
	Q.col(4) = Q1.col(1).mul(Q2.col(1));
	Q.col(5) = Q2.col(1) * 1.0;
	Q.col(6) = Q1.col(0) * 1.0;
	Q.col(7) = Q1.col(1) * 1.0;
	Q.col(8) = 1.0;

	Mat U, W, Vt;
	SVD::compute(Q, W, U, Vt, SVD::MODIFY_A | SVD::FULL_UV);

	Mat EE = Mat(Vt.t()).colRange(5, 9) * 1.0;
	Mat A(10, 20, CV_64F);
	EE = EE.t();
	getCoeffMat((double*)EE.data, (double*)A.data);
	EE = EE.t();

	A = A.colRange(0, 10).inv() * A.colRange(10, 20);

	double b[3 * 13];
	Mat B(3, 13, CV_64F, b);
	for (int i = 0; i < 3; i++)
	{
		Mat arow1 = A.row(i * 2 + 4) * 1.0;
		Mat arow2 = A.row(i * 2 + 5) * 1.0;
		Mat row1(1, 13, CV_64F, Scalar(0.0));
		Mat row2(1, 13, CV_64F, Scalar(0.0));

		row1.colRange(1, 4) = arow1.colRange(0, 3) * 1.0;
		row1.colRange(5, 8) = arow1.colRange(3, 6) * 1.0;
		row1.colRange(9, 13) = arow1.colRange(6, 10) * 1.0;

		row2.colRange(0, 3) = arow2.colRange(0, 3) * 1.0;
		row2.colRange(4, 7) = arow2.colRange(3, 6) * 1.0;
		row2.colRange(8, 12) = arow2.colRange(6, 10) * 1.0;

		B.row(i) = row1 - row2;
	}

	double c[11];
	Mat coeffs(1, 11, CV_64F, c);
	c[10] = (b[0] * b[17] * b[34] + b[26] * b[4] * b[21] - b[26] * b[17] * b[8] - b[13] * b[4] * b[34] - b[0] * b[21] * b[30] + b[13] * b[30] * b[8]);
	c[9] = (b[26] * b[4] * b[22] + b[14] * b[30] * b[8] + b[13] * b[31] * b[8] + b[1] * b[17] * b[34] - b[13] * b[5] * b[34] + b[26] * b[5] * b[21] - b[0] * b[21] * b[31] - b[26] * b[17] * b[9] - b[1] * b[21] * b[30] + b[27] * b[4] * b[21] + b[0] * b[17] * b[35] - b[0] * b[22] * b[30] + b[13] * b[30] * b[9] + b[0] * b[18] * b[34] - b[27] * b[17] * b[8] - b[14] * b[4] * b[34] - b[13] * b[4] * b[35] - b[26] * b[18] * b[8]);
	c[8] = (b[14] * b[30] * b[9] + b[14] * b[31] * b[8] + b[13] * b[31] * b[9] - b[13] * b[4] * b[36] - b[13] * b[5] * b[35] + b[15] * b[30] * b[8] - b[13] * b[6] * b[34] + b[13] * b[30] * b[10] + b[13] * b[32] * b[8] - b[14] * b[4] * b[35] - b[14] * b[5] * b[34] + b[26] * b[4] * b[23] + b[26] * b[5] * b[22] + b[26] * b[6] * b[21] - b[26] * b[17] * b[10] - b[15] * b[4] * b[34] - b[26] * b[18] * b[9] - b[26] * b[19] * b[8] + b[27] * b[4] * b[22] + b[27] * b[5] * b[21] - b[27] * b[17] * b[9] - b[27] * b[18] * b[8] - b[1] * b[21] * b[31] - b[0] * b[23] * b[30] - b[0] * b[21] * b[32] + b[28] * b[4] * b[21] - b[28] * b[17] * b[8] + b[2] * b[17] * b[34] + b[0] * b[18] * b[35] - b[0] * b[22] * b[31] + b[0] * b[17] * b[36] + b[0] * b[19] * b[34] - b[1] * b[22] * b[30] + b[1] * b[18] * b[34] + b[1] * b[17] * b[35] - b[2] * b[21] * b[30]);
	c[7] = (b[14] * b[30] * b[10] + b[14] * b[32] * b[8] - b[3] * b[21] * b[30] + b[3] * b[17] * b[34] + b[13] * b[32] * b[9] + b[13] * b[33] * b[8] - b[13] * b[4] * b[37] - b[13] * b[5] * b[36] + b[15] * b[30] * b[9] + b[15] * b[31] * b[8] - b[16] * b[4] * b[34] - b[13] * b[6] * b[35] - b[13] * b[7] * b[34] + b[13] * b[30] * b[11] + b[13] * b[31] * b[10] + b[14] * b[31] * b[9] - b[14] * b[4] * b[36] - b[14] * b[5] * b[35] - b[14] * b[6] * b[34] + b[16] * b[30] * b[8] - b[26] * b[20] * b[8] + b[26] * b[4] * b[24] + b[26] * b[5] * b[23] + b[26] * b[6] * b[22] + b[26] * b[7] * b[21] - b[26] * b[17] * b[11] - b[15] * b[4] * b[35] - b[15] * b[5] * b[34] - b[26] * b[18] * b[10] - b[26] * b[19] * b[9] + b[27] * b[4] * b[23] + b[27] * b[5] * b[22] + b[27] * b[6] * b[21] - b[27] * b[17] * b[10] - b[27] * b[18] * b[9] - b[27] * b[19] * b[8] + b[0] * b[17] * b[37] - b[0] * b[23] * b[31] - b[0] * b[24] * b[30] - b[0] * b[21] * b[33] - b[29] * b[17] * b[8] + b[28] * b[4] * b[22] + b[28] * b[5] * b[21] - b[28] * b[17] * b[9] - b[28] * b[18] * b[8] + b[29] * b[4] * b[21] + b[1] * b[19] * b[34] - b[2] * b[21] * b[31] + b[0] * b[20] * b[34] + b[0] * b[19] * b[35] + b[0] * b[18] * b[36] - b[0] * b[22] * b[32] - b[1] * b[23] * b[30] - b[1] * b[21] * b[32] + b[1] * b[18] * b[35] - b[1] * b[22] * b[31] - b[2] * b[22] * b[30] + b[2] * b[17] * b[35] + b[1] * b[17] * b[36] + b[2] * b[18] * b[34]);
	c[6] = (-b[14] * b[6] * b[35] - b[14] * b[7] * b[34] - b[3] * b[22] * b[30] - b[3] * b[21] * b[31] + b[3] * b[17] * b[35] + b[3] * b[18] * b[34] + b[13] * b[32] * b[10] + b[13] * b[33] * b[9] - b[13] * b[4] * b[38] - b[13] * b[5] * b[37] - b[15] * b[6] * b[34] + b[15] * b[30] * b[10] + b[15] * b[32] * b[8] - b[16] * b[4] * b[35] - b[13] * b[6] * b[36] - b[13] * b[7] * b[35] + b[13] * b[31] * b[11] + b[13] * b[30] * b[12] + b[14] * b[32] * b[9] + b[14] * b[33] * b[8] - b[14] * b[4] * b[37] - b[14] * b[5] * b[36] + b[16] * b[30] * b[9] + b[16] * b[31] * b[8] - b[26] * b[20] * b[9] + b[26] * b[4] * b[25] + b[26] * b[5] * b[24] + b[26] * b[6] * b[23] + b[26] * b[7] * b[22] - b[26] * b[17] * b[12] + b[14] * b[30] * b[11] + b[14] * b[31] * b[10] + b[15] * b[31] * b[9] - b[15] * b[4] * b[36] - b[15] * b[5] * b[35] - b[26] * b[18] * b[11] - b[26] * b[19] * b[10] - b[27] * b[20] * b[8] + b[27] * b[4] * b[24] + b[27] * b[5] * b[23] + b[27] * b[6] * b[22] + b[27] * b[7] * b[21] - b[27] * b[17] * b[11] - b[27] * b[18] * b[10] - b[27] * b[19] * b[9] - b[16] * b[5] * b[34] - b[29] * b[17] * b[9] - b[29] * b[18] * b[8] + b[28] * b[4] * b[23] + b[28] * b[5] * b[22] + b[28] * b[6] * b[21] - b[28] * b[17] * b[10] - b[28] * b[18] * b[9] - b[28] * b[19] * b[8] + b[29] * b[4] * b[22] + b[29] * b[5] * b[21] - b[2] * b[23] * b[30] + b[2] * b[18] * b[35] - b[1] * b[22] * b[32] - b[2] * b[21] * b[32] + b[2] * b[19] * b[34] + b[0] * b[19] * b[36] - b[0] * b[22] * b[33] + b[0] * b[20] * b[35] - b[0] * b[23] * b[32] - b[0] * b[25] * b[30] + b[0] * b[17] * b[38] + b[0] * b[18] * b[37] - b[0] * b[24] * b[31] + b[1] * b[17] * b[37] - b[1] * b[23] * b[31] - b[1] * b[24] * b[30] - b[1] * b[21] * b[33] + b[1] * b[20] * b[34] + b[1] * b[19] * b[35] + b[1] * b[18] * b[36] + b[2] * b[17] * b[36] - b[2] * b[22] * b[31]);
	c[5] = (-b[14] * b[6] * b[36] - b[14] * b[7] * b[35] + b[14] * b[31] * b[11] - b[3] * b[23] * b[30] - b[3] * b[21] * b[32] + b[3] * b[18] * b[35] - b[3] * b[22] * b[31] + b[3] * b[17] * b[36] + b[3] * b[19] * b[34] + b[13] * b[32] * b[11] + b[13] * b[33] * b[10] - b[13] * b[5] * b[38] - b[15] * b[6] * b[35] - b[15] * b[7] * b[34] + b[15] * b[30] * b[11] + b[15] * b[31] * b[10] + b[16] * b[31] * b[9] - b[13] * b[6] * b[37] - b[13] * b[7] * b[36] + b[13] * b[31] * b[12] + b[14] * b[32] * b[10] + b[14] * b[33] * b[9] - b[14] * b[4] * b[38] - b[14] * b[5] * b[37] - b[16] * b[6] * b[34] + b[16] * b[30] * b[10] + b[16] * b[32] * b[8] - b[26] * b[20] * b[10] + b[26] * b[5] * b[25] + b[26] * b[6] * b[24] + b[26] * b[7] * b[23] + b[14] * b[30] * b[12] + b[15] * b[32] * b[9] + b[15] * b[33] * b[8] - b[15] * b[4] * b[37] - b[15] * b[5] * b[36] + b[29] * b[5] * b[22] + b[29] * b[6] * b[21] - b[26] * b[18] * b[12] - b[26] * b[19] * b[11] - b[27] * b[20] * b[9] + b[27] * b[4] * b[25] + b[27] * b[5] * b[24] + b[27] * b[6] * b[23] + b[27] * b[7] * b[22] - b[27] * b[17] * b[12] - b[27] * b[18] * b[11] - b[27] * b[19] * b[10] - b[28] * b[20] * b[8] - b[16] * b[4] * b[36] - b[16] * b[5] * b[35] - b[29] * b[17] * b[10] - b[29] * b[18] * b[9] - b[29] * b[19] * b[8] + b[28] * b[4] * b[24] + b[28] * b[5] * b[23] + b[28] * b[6] * b[22] + b[28] * b[7] * b[21] - b[28] * b[17] * b[11] - b[28] * b[18] * b[10] - b[28] * b[19] * b[9] + b[29] * b[4] * b[23] - b[2] * b[22] * b[32] - b[2] * b[21] * b[33] - b[1] * b[24] * b[31] + b[0] * b[18] * b[38] - b[0] * b[24] * b[32] + b[0] * b[19] * b[37] + b[0] * b[20] * b[36] - b[0] * b[25] * b[31] - b[0] * b[23] * b[33] + b[1] * b[19] * b[36] - b[1] * b[22] * b[33] + b[1] * b[20] * b[35] + b[2] * b[19] * b[35] - b[2] * b[24] * b[30] - b[2] * b[23] * b[31] + b[2] * b[20] * b[34] + b[2] * b[17] * b[37] - b[1] * b[25] * b[30] + b[1] * b[18] * b[37] + b[1] * b[17] * b[38] - b[1] * b[23] * b[32] + b[2] * b[18] * b[36]);
	c[4] = (-b[14] * b[6] * b[37] - b[14] * b[7] * b[36] + b[14] * b[31] * b[12] + b[3] * b[17] * b[37] - b[3] * b[23] * b[31] - b[3] * b[24] * b[30] - b[3] * b[21] * b[33] + b[3] * b[20] * b[34] + b[3] * b[19] * b[35] + b[3] * b[18] * b[36] - b[3] * b[22] * b[32] + b[13] * b[32] * b[12] + b[13] * b[33] * b[11] - b[15] * b[6] * b[36] - b[15] * b[7] * b[35] + b[15] * b[31] * b[11] + b[15] * b[30] * b[12] + b[16] * b[32] * b[9] + b[16] * b[33] * b[8] - b[13] * b[6] * b[38] - b[13] * b[7] * b[37] + b[14] * b[32] * b[11] + b[14] * b[33] * b[10] - b[14] * b[5] * b[38] - b[16] * b[6] * b[35] - b[16] * b[7] * b[34] + b[16] * b[30] * b[11] + b[16] * b[31] * b[10] - b[26] * b[19] * b[12] - b[26] * b[20] * b[11] + b[26] * b[6] * b[25] + b[26] * b[7] * b[24] + b[15] * b[32] * b[10] + b[15] * b[33] * b[9] - b[15] * b[4] * b[38] - b[15] * b[5] * b[37] + b[29] * b[5] * b[23] + b[29] * b[6] * b[22] + b[29] * b[7] * b[21] - b[27] * b[20] * b[10] + b[27] * b[5] * b[25] + b[27] * b[6] * b[24] + b[27] * b[7] * b[23] - b[27] * b[18] * b[12] - b[27] * b[19] * b[11] - b[28] * b[20] * b[9] - b[16] * b[4] * b[37] - b[16] * b[5] * b[36] + b[0] * b[19] * b[38] - b[0] * b[24] * b[33] + b[0] * b[20] * b[37] - b[29] * b[17] * b[11] - b[29] * b[18] * b[10] - b[29] * b[19] * b[9] + b[28] * b[4] * b[25] + b[28] * b[5] * b[24] + b[28] * b[6] * b[23] + b[28] * b[7] * b[22] - b[28] * b[17] * b[12] - b[28] * b[18] * b[11] - b[28] * b[19] * b[10] - b[29] * b[20] * b[8] + b[29] * b[4] * b[24] + b[2] * b[18] * b[37] - b[0] * b[25] * b[32] + b[1] * b[18] * b[38] - b[1] * b[24] * b[32] + b[1] * b[19] * b[37] + b[1] * b[20] * b[36] - b[1] * b[25] * b[31] + b[2] * b[17] * b[38] + b[2] * b[19] * b[36] - b[2] * b[24] * b[31] - b[2] * b[22] * b[33] - b[2] * b[23] * b[32] + b[2] * b[20] * b[35] - b[1] * b[23] * b[33] - b[2] * b[25] * b[30]);
	c[3] = (-b[14] * b[6] * b[38] - b[14] * b[7] * b[37] + b[3] * b[19] * b[36] - b[3] * b[22] * b[33] + b[3] * b[20] * b[35] - b[3] * b[23] * b[32] - b[3] * b[25] * b[30] + b[3] * b[17] * b[38] + b[3] * b[18] * b[37] - b[3] * b[24] * b[31] - b[15] * b[6] * b[37] - b[15] * b[7] * b[36] + b[15] * b[31] * b[12] + b[16] * b[32] * b[10] + b[16] * b[33] * b[9] + b[13] * b[33] * b[12] - b[13] * b[7] * b[38] + b[14] * b[32] * b[12] + b[14] * b[33] * b[11] - b[16] * b[6] * b[36] - b[16] * b[7] * b[35] + b[16] * b[31] * b[11] + b[16] * b[30] * b[12] + b[15] * b[32] * b[11] + b[15] * b[33] * b[10] - b[15] * b[5] * b[38] + b[29] * b[5] * b[24] + b[29] * b[6] * b[23] - b[26] * b[20] * b[12] + b[26] * b[7] * b[25] - b[27] * b[19] * b[12] - b[27] * b[20] * b[11] + b[27] * b[6] * b[25] + b[27] * b[7] * b[24] - b[28] * b[20] * b[10] - b[16] * b[4] * b[38] - b[16] * b[5] * b[37] + b[29] * b[7] * b[22] - b[29] * b[17] * b[12] - b[29] * b[18] * b[11] - b[29] * b[19] * b[10] + b[28] * b[5] * b[25] + b[28] * b[6] * b[24] + b[28] * b[7] * b[23] - b[28] * b[18] * b[12] - b[28] * b[19] * b[11] - b[29] * b[20] * b[9] + b[29] * b[4] * b[25] - b[2] * b[24] * b[32] + b[0] * b[20] * b[38] - b[0] * b[25] * b[33] + b[1] * b[19] * b[38] - b[1] * b[24] * b[33] + b[1] * b[20] * b[37] - b[2] * b[25] * b[31] + b[2] * b[20] * b[36] - b[1] * b[25] * b[32] + b[2] * b[19] * b[37] + b[2] * b[18] * b[38] - b[2] * b[23] * b[33]);
	c[2] = (b[3] * b[18] * b[38] - b[3] * b[24] * b[32] + b[3] * b[19] * b[37] + b[3] * b[20] * b[36] - b[3] * b[25] * b[31] - b[3] * b[23] * b[33] - b[15] * b[6] * b[38] - b[15] * b[7] * b[37] + b[16] * b[32] * b[11] + b[16] * b[33] * b[10] - b[16] * b[5] * b[38] - b[16] * b[6] * b[37] - b[16] * b[7] * b[36] + b[16] * b[31] * b[12] + b[14] * b[33] * b[12] - b[14] * b[7] * b[38] + b[15] * b[32] * b[12] + b[15] * b[33] * b[11] + b[29] * b[5] * b[25] + b[29] * b[6] * b[24] - b[27] * b[20] * b[12] + b[27] * b[7] * b[25] - b[28] * b[19] * b[12] - b[28] * b[20] * b[11] + b[29] * b[7] * b[23] - b[29] * b[18] * b[12] - b[29] * b[19] * b[11] + b[28] * b[6] * b[25] + b[28] * b[7] * b[24] - b[29] * b[20] * b[10] + b[2] * b[19] * b[38] - b[1] * b[25] * b[33] + b[2] * b[20] * b[37] - b[2] * b[24] * b[33] - b[2] * b[25] * b[32] + b[1] * b[20] * b[38]);
	c[1] = (b[29] * b[7] * b[24] - b[29] * b[20] * b[11] + b[2] * b[20] * b[38] - b[2] * b[25] * b[33] - b[28] * b[20] * b[12] + b[28] * b[7] * b[25] - b[29] * b[19] * b[12] - b[3] * b[24] * b[33] + b[15] * b[33] * b[12] + b[3] * b[19] * b[38] - b[16] * b[6] * b[38] + b[3] * b[20] * b[37] + b[16] * b[32] * b[12] + b[29] * b[6] * b[25] - b[16] * b[7] * b[37] - b[3] * b[25] * b[32] - b[15] * b[7] * b[38] + b[16] * b[33] * b[11]);
	c[0] = -b[29] * b[20] * b[12] + b[29] * b[7] * b[25] + b[16] * b[33] * b[12] - b[16] * b[7] * b[38] + b[3] * b[20] * b[38] - b[3] * b[25] * b[33];

	std::vector<std::complex<double> > roots;
	solvePoly(coeffs, roots);

	std::vector<double> xs, ys, zs;
	int count = 0;
	double * e = ematrix->data.db;
	for (int i = 0; i < roots.size(); i++)
	{
		if (fabs(roots[i].imag()) > 1e-10) continue;
		double z1 = roots[i].real();
		double z2 = z1 * z1;
		double z3 = z2 * z1;
		double z4 = z3 * z1;

		double bz[3][3];
		for (int j = 0; j < 3; j++)
		{
			const double * br = b + j * 13;
			bz[j][0] = br[0] * z3 + br[1] * z2 + br[2] * z1 + br[3];
			bz[j][1] = br[4] * z3 + br[5] * z2 + br[6] * z1 + br[7];
			bz[j][2] = br[8] * z4 + br[9] * z3 + br[10] * z2 + br[11] * z1 + br[12];
		}

		Mat Bz(3, 3, CV_64F, bz);
		cv::Mat xy1;
		SVD::solveZ(Bz, xy1);

		if (fabs(xy1.at<double>(2)) < 1e-10) continue;
		xs.push_back(xy1.at<double>(0) / xy1.at<double>(2));
		ys.push_back(xy1.at<double>(1) / xy1.at<double>(2));
		zs.push_back(z1);

		cv::Mat Evec = EE.col(0) * xs.back() + EE.col(1) * ys.back() + EE.col(2) * zs.back() + EE.col(3);
		Evec /= norm(Evec);

		memcpy(e + count * 9, Evec.data, 9 * sizeof(double));
		count++;
	}

	return count;

}
// Same as the runKernel (run5Point), m1 and m2 should be 1 row x n col x 2 channels. And also, error has to be of CV_32FC1. 
void CvEMEstimator::computeReprojError(const CvMat* m1, const CvMat* m2, const CvMat* model, CvMat* error)
{
	Mat X1(m1), X2(m2);
	int n = X1.cols;
	X1 = X1.reshape(1, n);
	X2 = X2.reshape(1, n);

	X1.convertTo(X1, CV_64F);
	X2.convertTo(X2, CV_64F);

	Mat E(model);
	for (int i = 0; i < n; i++)
	{
		Mat x1 = (Mat_<double>(3, 1) << X1.at<double>(i, 0), X1.at<double>(i, 1), 1.0);
		Mat x2 = (Mat_<double>(3, 1) << X2.at<double>(i, 0), X2.at<double>(i, 1), 1.0);
		double x2tEx1 = x2.dot(E * x1);
		Mat Ex1 = E * x1;
		Mat Etx2 = E * x2;
		double a = Ex1.at<double>(0) * Ex1.at<double>(0);
		double b = Ex1.at<double>(1) * Ex1.at<double>(1);
		double c = Etx2.at<double>(0) * Etx2.at<double>(0);
		double d = Etx2.at<double>(0) * Etx2.at<double>(0);

		error->data.fl[i] = x2tEx1 * x2tEx1 / (a + b + c + d);
	}

	/*	Eigen::MatrixXd X1t, X2t;
	cv2eigen(Mat(m1).reshape(1, m1->cols), X1t);
	cv2eigen(Mat(m2).reshape(1, m2->cols), X2t);
	Eigen::MatrixXd X1(3, X1t.rows());
	Eigen::MatrixXd X2(3, X2t.rows());
	X1.topRows(2) = X1t.transpose();
	X2.topRows(2) = X2t.transpose();
	X1.row(2).setOnes();
	X2.row(2).setOnes();

	Eigen::MatrixXd E;
	cv2eigen(Mat(model), E);

	// Compute Simpson's error
	Eigen::MatrixXd Ex1, x2tEx1, Etx2, SimpsonError;
	Ex1 = E * X1;
	x2tEx1 = (X2.array() * Ex1.array()).matrix().colwise().sum();
	Etx2 = E.transpose() * X2;
	SimpsonError = x2tEx1.array().square() / (Ex1.row(0).array().square() + Ex1.row(1).array().square() + Etx2.row(0).array().square() + Etx2.row(1).array().square());

	assert( CV_IS_MAT_CONT(error->type) );
	Mat isInliers, R, t;
	for (int i = 0; i < SimpsonError.cols(); i++)
	{
	error->data.fl[i] = SimpsonError(0, i);
	}
	*/
}
void CvEMEstimator::getCoeffMat(double *e, double *A)
{
	double ep2[36], ep3[36];
	for (int i = 0; i < 36; i++)
	{
		ep2[i] = e[i] * e[i];
		ep3[i] = ep2[i] * e[i];
	}

	A[0] = e[33] * e[28] * e[32] - e[33] * e[31] * e[29] + e[30] * e[34] * e[29] - e[30] * e[28] * e[35] - e[27] * e[32] * e[34] + e[27] * e[31] * e[35];
	A[146] = .5000000000*e[6] * ep2[8] - .5000000000*e[6] * ep2[5] + .5000000000*ep3[6] + .5000000000*e[6] * ep2[7] - .5000000000*e[6] * ep2[4] + e[0] * e[2] * e[8] + e[3] * e[4] * e[7] + e[3] * e[5] * e[8] + e[0] * e[1] * e[7] - .5000000000*e[6] * ep2[1] - .5000000000*e[6] * ep2[2] + .5000000000*ep2[0] * e[6] + .5000000000*ep2[3] * e[6];
	A[1] = e[30] * e[34] * e[2] + e[33] * e[1] * e[32] - e[3] * e[28] * e[35] + e[0] * e[31] * e[35] + e[3] * e[34] * e[29] - e[30] * e[1] * e[35] + e[27] * e[31] * e[8] - e[27] * e[32] * e[7] - e[30] * e[28] * e[8] - e[33] * e[31] * e[2] - e[0] * e[32] * e[34] + e[6] * e[28] * e[32] - e[33] * e[4] * e[29] + e[33] * e[28] * e[5] + e[30] * e[7] * e[29] + e[27] * e[4] * e[35] - e[27] * e[5] * e[34] - e[6] * e[31] * e[29];
	A[147] = e[9] * e[27] * e[15] + e[9] * e[29] * e[17] + e[9] * e[11] * e[35] + e[9] * e[28] * e[16] + e[9] * e[10] * e[34] + e[27] * e[11] * e[17] + e[27] * e[10] * e[16] + e[12] * e[30] * e[15] + e[12] * e[32] * e[17] + e[12] * e[14] * e[35] + e[12] * e[31] * e[16] + e[12] * e[13] * e[34] + e[30] * e[14] * e[17] + e[30] * e[13] * e[16] + e[15] * e[35] * e[17] + e[15] * e[34] * e[16] - 1.*e[15] * e[28] * e[10] - 1.*e[15] * e[31] * e[13] - 1.*e[15] * e[32] * e[14] - 1.*e[15] * e[29] * e[11] + .5000000000*ep2[9] * e[33] + .5000000000*e[33] * ep2[16] - .5000000000*e[33] * ep2[11] + .5000000000*e[33] * ep2[12] + 1.500000000*e[33] * ep2[15] + .5000000000*e[33] * ep2[17] - .5000000000*e[33] * ep2[10] - .5000000000*e[33] * ep2[14] - .5000000000*e[33] * ep2[13];
	A[2] = -e[33] * e[22] * e[29] - e[33] * e[31] * e[20] - e[27] * e[32] * e[25] + e[27] * e[22] * e[35] - e[27] * e[23] * e[34] + e[27] * e[31] * e[26] + e[33] * e[28] * e[23] - e[21] * e[28] * e[35] + e[30] * e[25] * e[29] + e[24] * e[28] * e[32] - e[24] * e[31] * e[29] + e[18] * e[31] * e[35] - e[30] * e[28] * e[26] - e[30] * e[19] * e[35] + e[21] * e[34] * e[29] + e[33] * e[19] * e[32] - e[18] * e[32] * e[34] + e[30] * e[34] * e[20];
	A[144] = e[18] * e[2] * e[17] + e[3] * e[21] * e[15] + e[3] * e[12] * e[24] + e[3] * e[23] * e[17] + e[3] * e[14] * e[26] + e[3] * e[22] * e[16] + e[3] * e[13] * e[25] + 3.*e[6] * e[24] * e[15] + e[6] * e[26] * e[17] + e[6] * e[25] * e[16] + e[0] * e[20] * e[17] + e[0] * e[11] * e[26] + e[0] * e[19] * e[16] + e[0] * e[10] * e[25] + e[15] * e[26] * e[8] - 1.*e[15] * e[20] * e[2] - 1.*e[15] * e[19] * e[1] - 1.*e[15] * e[22] * e[4] + e[15] * e[25] * e[7] - 1.*e[15] * e[23] * e[5] + e[12] * e[21] * e[6] + e[12] * e[22] * e[7] + e[12] * e[4] * e[25] + e[12] * e[23] * e[8] + e[12] * e[5] * e[26] - 1.*e[24] * e[11] * e[2] - 1.*e[24] * e[10] * e[1] - 1.*e[24] * e[13] * e[4] + e[24] * e[16] * e[7] - 1.*e[24] * e[14] * e[5] + e[24] * e[17] * e[8] + e[21] * e[13] * e[7] + e[21] * e[4] * e[16] + e[21] * e[14] * e[8] + e[21] * e[5] * e[17] - 1.*e[6] * e[23] * e[14] - 1.*e[6] * e[20] * e[11] - 1.*e[6] * e[19] * e[10] - 1.*e[6] * e[22] * e[13] + e[9] * e[18] * e[6] + e[9] * e[0] * e[24] + e[9] * e[19] * e[7] + e[9] * e[1] * e[25] + e[9] * e[20] * e[8] + e[9] * e[2] * e[26] + e[18] * e[0] * e[15] + e[18] * e[10] * e[7] + e[18] * e[1] * e[16] + e[18] * e[11] * e[8];
	A[3] = e[33] * e[10] * e[32] + e[33] * e[28] * e[14] - e[33] * e[13] * e[29] - e[33] * e[31] * e[11] + e[9] * e[31] * e[35] - e[9] * e[32] * e[34] + e[27] * e[13] * e[35] - e[27] * e[32] * e[16] + e[27] * e[31] * e[17] - e[27] * e[14] * e[34] + e[12] * e[34] * e[29] - e[12] * e[28] * e[35] + e[30] * e[34] * e[11] + e[30] * e[16] * e[29] - e[30] * e[10] * e[35] - e[30] * e[28] * e[17] + e[15] * e[28] * e[32] - e[15] * e[31] * e[29];
	A[145] = e[0] * e[27] * e[6] + e[0] * e[28] * e[7] + e[0] * e[1] * e[34] + e[0] * e[29] * e[8] + e[0] * e[2] * e[35] + e[6] * e[34] * e[7] - 1.*e[6] * e[32] * e[5] + e[6] * e[30] * e[3] + e[6] * e[35] * e[8] - 1.*e[6] * e[29] * e[2] - 1.*e[6] * e[28] * e[1] - 1.*e[6] * e[31] * e[4] + e[27] * e[1] * e[7] + e[27] * e[2] * e[8] + e[3] * e[31] * e[7] + e[3] * e[4] * e[34] + e[3] * e[32] * e[8] + e[3] * e[5] * e[35] + e[30] * e[4] * e[7] + e[30] * e[5] * e[8] + .5000000000*ep2[0] * e[33] + 1.500000000*e[33] * ep2[6] - .5000000000*e[33] * ep2[4] - .5000000000*e[33] * ep2[5] - .5000000000*e[33] * ep2[1] + .5000000000*e[33] * ep2[7] + .5000000000*e[33] * ep2[3] - .5000000000*e[33] * ep2[2] + .5000000000*e[33] * ep2[8];
	A[4] = -e[0] * e[23] * e[16] + e[9] * e[4] * e[26] + e[9] * e[22] * e[8] - e[9] * e[5] * e[25] - e[9] * e[23] * e[7] + e[18] * e[4] * e[17] + e[18] * e[13] * e[8] - e[18] * e[5] * e[16] - e[18] * e[14] * e[7] + e[3] * e[16] * e[20] + e[3] * e[25] * e[11] - e[3] * e[10] * e[26] - e[3] * e[19] * e[17] + e[12] * e[7] * e[20] + e[12] * e[25] * e[2] - e[12] * e[1] * e[26] - e[12] * e[19] * e[8] + e[21] * e[7] * e[11] + e[21] * e[16] * e[2] - e[21] * e[1] * e[17] - e[21] * e[10] * e[8] + e[6] * e[10] * e[23] + e[6] * e[19] * e[14] - e[6] * e[13] * e[20] - e[6] * e[22] * e[11] + e[15] * e[1] * e[23] + e[15] * e[19] * e[5] - e[15] * e[4] * e[20] - e[15] * e[22] * e[2] + e[24] * e[1] * e[14] + e[24] * e[10] * e[5] - e[24] * e[4] * e[11] - e[24] * e[13] * e[2] + e[0] * e[13] * e[26] + e[0] * e[22] * e[17] - e[0] * e[14] * e[25];
	A[150] = e[18] * e[19] * e[25] + .5000000000*ep3[24] - .5000000000*e[24] * ep2[23] + e[18] * e[20] * e[26] + e[21] * e[22] * e[25] + e[21] * e[23] * e[26] - .5000000000*e[24] * ep2[19] + .5000000000*ep2[21] * e[24] + .5000000000*e[24] * ep2[26] - .5000000000*e[24] * ep2[20] + .5000000000*ep2[18] * e[24] - .5000000000*e[24] * ep2[22] + .5000000000*e[24] * ep2[25];
	A[5] = -e[3] * e[1] * e[35] - e[0] * e[32] * e[7] + e[27] * e[4] * e[8] + e[33] * e[1] * e[5] - e[33] * e[4] * e[2] + e[0] * e[4] * e[35] + e[3] * e[34] * e[2] - e[30] * e[1] * e[8] + e[30] * e[7] * e[2] - e[6] * e[4] * e[29] + e[3] * e[7] * e[29] + e[6] * e[1] * e[32] - e[0] * e[5] * e[34] - e[3] * e[28] * e[8] + e[0] * e[31] * e[8] + e[6] * e[28] * e[5] - e[6] * e[31] * e[2] - e[27] * e[5] * e[7];
	A[151] = e[33] * e[16] * e[7] - 1.*e[33] * e[14] * e[5] + e[33] * e[17] * e[8] + e[30] * e[13] * e[7] + e[30] * e[4] * e[16] + e[30] * e[14] * e[8] + e[30] * e[5] * e[17] + e[6] * e[27] * e[9] - 1.*e[6] * e[28] * e[10] - 1.*e[6] * e[31] * e[13] - 1.*e[6] * e[32] * e[14] - 1.*e[6] * e[29] * e[11] + e[9] * e[28] * e[7] + e[9] * e[1] * e[34] + e[9] * e[29] * e[8] + e[9] * e[2] * e[35] + e[27] * e[10] * e[7] + e[27] * e[1] * e[16] + e[27] * e[11] * e[8] + e[27] * e[2] * e[17] + e[3] * e[30] * e[15] + e[3] * e[12] * e[33] + e[3] * e[32] * e[17] + e[3] * e[14] * e[35] + e[3] * e[31] * e[16] + e[3] * e[13] * e[34] + 3.*e[6] * e[33] * e[15] + e[6] * e[35] * e[17] + e[6] * e[34] * e[16] + e[0] * e[27] * e[15] + e[0] * e[9] * e[33] + e[0] * e[29] * e[17] + e[0] * e[11] * e[35] + e[0] * e[28] * e[16] + e[0] * e[10] * e[34] + e[15] * e[34] * e[7] - 1.*e[15] * e[32] * e[5] + e[15] * e[35] * e[8] - 1.*e[15] * e[29] * e[2] - 1.*e[15] * e[28] * e[1] - 1.*e[15] * e[31] * e[4] + e[12] * e[30] * e[6] + e[12] * e[31] * e[7] + e[12] * e[4] * e[34] + e[12] * e[32] * e[8] + e[12] * e[5] * e[35] - 1.*e[33] * e[11] * e[2] - 1.*e[33] * e[10] * e[1] - 1.*e[33] * e[13] * e[4];
	A[6] = e[6] * e[1] * e[5] - e[6] * e[4] * e[2] + e[3] * e[7] * e[2] + e[0] * e[4] * e[8] - e[0] * e[5] * e[7] - e[3] * e[1] * e[8];
	A[148] = .5000000000*ep3[15] + e[9] * e[10] * e[16] - .5000000000*e[15] * ep2[11] + e[9] * e[11] * e[17] + .5000000000*ep2[12] * e[15] + .5000000000*e[15] * ep2[16] + .5000000000*e[15] * ep2[17] - .5000000000*e[15] * ep2[13] + .5000000000*ep2[9] * e[15] + e[12] * e[14] * e[17] - .5000000000*e[15] * ep2[10] - .5000000000*e[15] * ep2[14] + e[12] * e[13] * e[16];
	A[7] = e[15] * e[28] * e[14] - e[15] * e[13] * e[29] - e[15] * e[31] * e[11] + e[33] * e[10] * e[14] - e[33] * e[13] * e[11] + e[9] * e[13] * e[35] - e[9] * e[32] * e[16] + e[9] * e[31] * e[17] - e[9] * e[14] * e[34] + e[27] * e[13] * e[17] - e[27] * e[14] * e[16] + e[12] * e[34] * e[11] + e[12] * e[16] * e[29] - e[12] * e[10] * e[35] - e[12] * e[28] * e[17] + e[30] * e[16] * e[11] - e[30] * e[10] * e[17] + e[15] * e[10] * e[32];
	A[149] = e[18] * e[27] * e[24] + e[18] * e[28] * e[25] + e[18] * e[19] * e[34] + e[18] * e[29] * e[26] + e[18] * e[20] * e[35] + e[27] * e[19] * e[25] + e[27] * e[20] * e[26] + e[21] * e[30] * e[24] + e[21] * e[31] * e[25] + e[21] * e[22] * e[34] + e[21] * e[32] * e[26] + e[21] * e[23] * e[35] + e[30] * e[22] * e[25] + e[30] * e[23] * e[26] + e[24] * e[34] * e[25] + e[24] * e[35] * e[26] - 1.*e[24] * e[29] * e[20] - 1.*e[24] * e[31] * e[22] - 1.*e[24] * e[32] * e[23] - 1.*e[24] * e[28] * e[19] + 1.500000000*e[33] * ep2[24] + .5000000000*e[33] * ep2[25] + .5000000000*e[33] * ep2[26] - .5000000000*e[33] * ep2[23] - .5000000000*e[33] * ep2[19] - .5000000000*e[33] * ep2[20] - .5000000000*e[33] * ep2[22] + .5000000000*ep2[18] * e[33] + .5000000000*ep2[21] * e[33];
	A[9] = e[21] * e[25] * e[29] - e[27] * e[23] * e[25] + e[24] * e[19] * e[32] - e[21] * e[28] * e[26] - e[21] * e[19] * e[35] + e[18] * e[31] * e[26] - e[30] * e[19] * e[26] - e[24] * e[31] * e[20] + e[24] * e[28] * e[23] + e[27] * e[22] * e[26] + e[30] * e[25] * e[20] - e[33] * e[22] * e[20] + e[33] * e[19] * e[23] + e[21] * e[34] * e[20] - e[18] * e[23] * e[34] - e[24] * e[22] * e[29] - e[18] * e[32] * e[25] + e[18] * e[22] * e[35];
	A[155] = e[12] * e[14] * e[8] + e[12] * e[5] * e[17] + e[15] * e[16] * e[7] + e[15] * e[17] * e[8] + e[0] * e[11] * e[17] + e[0] * e[9] * e[15] + e[0] * e[10] * e[16] + e[3] * e[14] * e[17] + e[3] * e[13] * e[16] + e[9] * e[10] * e[7] + e[9] * e[1] * e[16] + e[9] * e[11] * e[8] + e[9] * e[2] * e[17] - 1.*e[15] * e[11] * e[2] - 1.*e[15] * e[10] * e[1] - 1.*e[15] * e[13] * e[4] - 1.*e[15] * e[14] * e[5] + e[12] * e[3] * e[15] + e[12] * e[13] * e[7] + e[12] * e[4] * e[16] + .5000000000*ep2[12] * e[6] + 1.500000000*ep2[15] * e[6] + .5000000000*e[6] * ep2[17] + .5000000000*e[6] * ep2[16] + .5000000000*e[6] * ep2[9] - .5000000000*e[6] * ep2[11] - .5000000000*e[6] * ep2[10] - .5000000000*e[6] * ep2[14] - .5000000000*e[6] * ep2[13];
	A[8] = -e[9] * e[14] * e[16] - e[12] * e[10] * e[17] + e[9] * e[13] * e[17] - e[15] * e[13] * e[11] + e[15] * e[10] * e[14] + e[12] * e[16] * e[11];
	A[154] = e[21] * e[14] * e[17] + e[21] * e[13] * e[16] + e[15] * e[26] * e[17] + e[15] * e[25] * e[16] - 1.*e[15] * e[23] * e[14] - 1.*e[15] * e[20] * e[11] - 1.*e[15] * e[19] * e[10] - 1.*e[15] * e[22] * e[13] + e[9] * e[20] * e[17] + e[9] * e[11] * e[26] + e[9] * e[19] * e[16] + e[9] * e[10] * e[25] + .5000000000*ep2[12] * e[24] + 1.500000000*e[24] * ep2[15] + .5000000000*e[24] * ep2[17] + .5000000000*e[24] * ep2[16] + .5000000000*ep2[9] * e[24] - .5000000000*e[24] * ep2[11] - .5000000000*e[24] * ep2[10] - .5000000000*e[24] * ep2[14] - .5000000000*e[24] * ep2[13] + e[18] * e[11] * e[17] + e[18] * e[9] * e[15] + e[18] * e[10] * e[16] + e[12] * e[21] * e[15] + e[12] * e[23] * e[17] + e[12] * e[14] * e[26] + e[12] * e[22] * e[16] + e[12] * e[13] * e[25];
	A[11] = -e[9] * e[5] * e[34] + e[9] * e[31] * e[8] - e[9] * e[32] * e[7] + e[27] * e[4] * e[17] + e[27] * e[13] * e[8] - e[27] * e[5] * e[16] - e[27] * e[14] * e[7] + e[0] * e[13] * e[35] - e[0] * e[32] * e[16] + e[0] * e[31] * e[17] - e[0] * e[14] * e[34] + e[9] * e[4] * e[35] + e[6] * e[10] * e[32] + e[6] * e[28] * e[14] - e[6] * e[13] * e[29] - e[6] * e[31] * e[11] + e[15] * e[1] * e[32] + e[3] * e[34] * e[11] + e[3] * e[16] * e[29] - e[3] * e[10] * e[35] - e[3] * e[28] * e[17] - e[12] * e[1] * e[35] + e[12] * e[7] * e[29] + e[12] * e[34] * e[2] - e[12] * e[28] * e[8] + e[15] * e[28] * e[5] - e[15] * e[4] * e[29] - e[15] * e[31] * e[2] + e[33] * e[1] * e[14] + e[33] * e[10] * e[5] - e[33] * e[4] * e[11] - e[33] * e[13] * e[2] + e[30] * e[7] * e[11] + e[30] * e[16] * e[2] - e[30] * e[1] * e[17] - e[30] * e[10] * e[8];
	A[153] = e[21] * e[31] * e[7] + e[21] * e[4] * e[34] + e[21] * e[32] * e[8] + e[21] * e[5] * e[35] + e[30] * e[22] * e[7] + e[30] * e[4] * e[25] + e[30] * e[23] * e[8] + e[30] * e[5] * e[26] + 3.*e[24] * e[33] * e[6] + e[24] * e[34] * e[7] + e[24] * e[35] * e[8] + e[33] * e[25] * e[7] + e[33] * e[26] * e[8] + e[0] * e[27] * e[24] + e[0] * e[18] * e[33] + e[0] * e[28] * e[25] + e[0] * e[19] * e[34] + e[0] * e[29] * e[26] + e[0] * e[20] * e[35] + e[18] * e[27] * e[6] + e[18] * e[28] * e[7] + e[18] * e[1] * e[34] + e[18] * e[29] * e[8] + e[18] * e[2] * e[35] + e[27] * e[19] * e[7] + e[27] * e[1] * e[25] + e[27] * e[20] * e[8] + e[27] * e[2] * e[26] + e[3] * e[30] * e[24] + e[3] * e[21] * e[33] + e[3] * e[31] * e[25] + e[3] * e[22] * e[34] + e[3] * e[32] * e[26] + e[3] * e[23] * e[35] + e[6] * e[30] * e[21] - 1.*e[6] * e[29] * e[20] + e[6] * e[35] * e[26] - 1.*e[6] * e[31] * e[22] - 1.*e[6] * e[32] * e[23] - 1.*e[6] * e[28] * e[19] + e[6] * e[34] * e[25] - 1.*e[24] * e[32] * e[5] - 1.*e[24] * e[29] * e[2] - 1.*e[24] * e[28] * e[1] - 1.*e[24] * e[31] * e[4] - 1.*e[33] * e[20] * e[2] - 1.*e[33] * e[19] * e[1] - 1.*e[33] * e[22] * e[4] - 1.*e[33] * e[23] * e[5];
	A[10] = e[21] * e[25] * e[20] - e[21] * e[19] * e[26] + e[18] * e[22] * e[26] - e[18] * e[23] * e[25] - e[24] * e[22] * e[20] + e[24] * e[19] * e[23];
	A[152] = e[3] * e[4] * e[25] + e[3] * e[23] * e[8] + e[3] * e[5] * e[26] + e[21] * e[4] * e[7] + e[21] * e[5] * e[8] + e[6] * e[25] * e[7] + e[6] * e[26] * e[8] + e[0] * e[19] * e[7] + e[0] * e[1] * e[25] + e[0] * e[20] * e[8] + e[0] * e[2] * e[26] - 1.*e[6] * e[20] * e[2] - 1.*e[6] * e[19] * e[1] - 1.*e[6] * e[22] * e[4] - 1.*e[6] * e[23] * e[5] + e[18] * e[1] * e[7] + e[18] * e[0] * e[6] + e[18] * e[2] * e[8] + e[3] * e[21] * e[6] + e[3] * e[22] * e[7] - .5000000000*e[24] * ep2[4] + .5000000000*e[24] * ep2[0] + 1.500000000*e[24] * ep2[6] - .5000000000*e[24] * ep2[5] - .5000000000*e[24] * ep2[1] + .5000000000*e[24] * ep2[7] + .5000000000*e[24] * ep2[3] - .5000000000*e[24] * ep2[2] + .5000000000*e[24] * ep2[8];
	A[13] = e[6] * e[28] * e[23] - e[6] * e[22] * e[29] - e[6] * e[31] * e[20] - e[3] * e[19] * e[35] + e[3] * e[34] * e[20] + e[3] * e[25] * e[29] - e[21] * e[1] * e[35] + e[21] * e[7] * e[29] + e[21] * e[34] * e[2] + e[24] * e[1] * e[32] + e[24] * e[28] * e[5] - e[24] * e[4] * e[29] - e[24] * e[31] * e[2] + e[33] * e[1] * e[23] + e[33] * e[19] * e[5] - e[33] * e[4] * e[20] - e[33] * e[22] * e[2] - e[21] * e[28] * e[8] + e[30] * e[7] * e[20] + e[30] * e[25] * e[2] - e[30] * e[1] * e[26] + e[18] * e[4] * e[35] - e[18] * e[5] * e[34] + e[18] * e[31] * e[8] - e[18] * e[32] * e[7] + e[27] * e[4] * e[26] + e[27] * e[22] * e[8] - e[27] * e[5] * e[25] - e[27] * e[23] * e[7] - e[3] * e[28] * e[26] - e[0] * e[32] * e[25] + e[0] * e[22] * e[35] - e[0] * e[23] * e[34] + e[0] * e[31] * e[26] - e[30] * e[19] * e[8] + e[6] * e[19] * e[32];
	A[159] = .5000000000*ep2[18] * e[6] + .5000000000*ep2[21] * e[6] + 1.500000000*ep2[24] * e[6] + .5000000000*e[6] * ep2[26] - .5000000000*e[6] * ep2[23] - .5000000000*e[6] * ep2[19] - .5000000000*e[6] * ep2[20] - .5000000000*e[6] * ep2[22] + .5000000000*e[6] * ep2[25] + e[21] * e[3] * e[24] + e[18] * e[20] * e[8] + e[21] * e[4] * e[25] + e[18] * e[19] * e[7] + e[18] * e[1] * e[25] + e[21] * e[22] * e[7] + e[21] * e[23] * e[8] + e[18] * e[0] * e[24] + e[18] * e[2] * e[26] + e[21] * e[5] * e[26] + e[24] * e[26] * e[8] - 1.*e[24] * e[20] * e[2] - 1.*e[24] * e[19] * e[1] - 1.*e[24] * e[22] * e[4] + e[24] * e[25] * e[7] - 1.*e[24] * e[23] * e[5] + e[0] * e[19] * e[25] + e[0] * e[20] * e[26] + e[3] * e[22] * e[25] + e[3] * e[23] * e[26];
	A[12] = e[18] * e[4] * e[8] + e[3] * e[7] * e[20] + e[3] * e[25] * e[2] - e[3] * e[1] * e[26] - e[18] * e[5] * e[7] + e[6] * e[1] * e[23] + e[6] * e[19] * e[5] - e[6] * e[4] * e[20] - e[6] * e[22] * e[2] + e[21] * e[7] * e[2] - e[21] * e[1] * e[8] + e[24] * e[1] * e[5] - e[24] * e[4] * e[2] - e[3] * e[19] * e[8] + e[0] * e[4] * e[26] + e[0] * e[22] * e[8] - e[0] * e[5] * e[25] - e[0] * e[23] * e[7];
	A[158] = e[9] * e[1] * e[7] + e[9] * e[0] * e[6] + e[9] * e[2] * e[8] + e[3] * e[12] * e[6] + e[3] * e[13] * e[7] + e[3] * e[4] * e[16] + e[3] * e[14] * e[8] + e[3] * e[5] * e[17] + e[12] * e[4] * e[7] + e[12] * e[5] * e[8] + e[6] * e[16] * e[7] + e[6] * e[17] * e[8] - 1.*e[6] * e[11] * e[2] - 1.*e[6] * e[10] * e[1] - 1.*e[6] * e[13] * e[4] - 1.*e[6] * e[14] * e[5] + e[0] * e[10] * e[7] + e[0] * e[1] * e[16] + e[0] * e[11] * e[8] + e[0] * e[2] * e[17] + .5000000000*ep2[3] * e[15] + 1.500000000*e[15] * ep2[6] + .5000000000*e[15] * ep2[7] + .5000000000*e[15] * ep2[8] + .5000000000*ep2[0] * e[15] - .5000000000*e[15] * ep2[4] - .5000000000*e[15] * ep2[5] - .5000000000*e[15] * ep2[1] - .5000000000*e[15] * ep2[2];
	A[15] = -e[15] * e[13] * e[2] - e[6] * e[13] * e[11] - e[15] * e[4] * e[11] + e[12] * e[16] * e[2] - e[3] * e[10] * e[17] + e[3] * e[16] * e[11] + e[0] * e[13] * e[17] - e[0] * e[14] * e[16] + e[15] * e[1] * e[14] - e[12] * e[10] * e[8] + e[9] * e[4] * e[17] + e[9] * e[13] * e[8] - e[9] * e[5] * e[16] - e[9] * e[14] * e[7] + e[15] * e[10] * e[5] + e[12] * e[7] * e[11] + e[6] * e[10] * e[14] - e[12] * e[1] * e[17];
	A[157] = e[12] * e[30] * e[24] + e[12] * e[21] * e[33] + e[12] * e[31] * e[25] + e[12] * e[22] * e[34] + e[12] * e[32] * e[26] + e[12] * e[23] * e[35] + e[9] * e[27] * e[24] + e[9] * e[18] * e[33] + e[9] * e[28] * e[25] + e[9] * e[19] * e[34] + e[9] * e[29] * e[26] + e[9] * e[20] * e[35] + e[21] * e[30] * e[15] + e[21] * e[32] * e[17] + e[21] * e[14] * e[35] + e[21] * e[31] * e[16] + e[21] * e[13] * e[34] + e[30] * e[23] * e[17] + e[30] * e[14] * e[26] + e[30] * e[22] * e[16] + e[30] * e[13] * e[25] + e[15] * e[27] * e[18] + 3.*e[15] * e[33] * e[24] - 1.*e[15] * e[29] * e[20] + e[15] * e[35] * e[26] - 1.*e[15] * e[31] * e[22] - 1.*e[15] * e[32] * e[23] - 1.*e[15] * e[28] * e[19] + e[15] * e[34] * e[25] + e[18] * e[29] * e[17] + e[18] * e[11] * e[35] + e[18] * e[28] * e[16] + e[18] * e[10] * e[34] + e[27] * e[20] * e[17] + e[27] * e[11] * e[26] + e[27] * e[19] * e[16] + e[27] * e[10] * e[25] - 1.*e[24] * e[28] * e[10] - 1.*e[24] * e[31] * e[13] - 1.*e[24] * e[32] * e[14] + e[24] * e[34] * e[16] + e[24] * e[35] * e[17] - 1.*e[24] * e[29] * e[11] - 1.*e[33] * e[23] * e[14] + e[33] * e[25] * e[16] + e[33] * e[26] * e[17] - 1.*e[33] * e[20] * e[11] - 1.*e[33] * e[19] * e[10] - 1.*e[33] * e[22] * e[13];
	A[14] = e[18] * e[13] * e[17] + e[9] * e[13] * e[26] + e[9] * e[22] * e[17] - e[9] * e[14] * e[25] - e[18] * e[14] * e[16] - e[15] * e[13] * e[20] - e[15] * e[22] * e[11] + e[12] * e[16] * e[20] + e[12] * e[25] * e[11] - e[12] * e[10] * e[26] - e[12] * e[19] * e[17] + e[21] * e[16] * e[11] - e[21] * e[10] * e[17] - e[9] * e[23] * e[16] + e[24] * e[10] * e[14] - e[24] * e[13] * e[11] + e[15] * e[10] * e[23] + e[15] * e[19] * e[14];
	A[156] = e[21] * e[12] * e[24] + e[21] * e[23] * e[17] + e[21] * e[14] * e[26] + e[21] * e[22] * e[16] + e[21] * e[13] * e[25] + e[24] * e[26] * e[17] + e[24] * e[25] * e[16] + e[9] * e[19] * e[25] + e[9] * e[18] * e[24] + e[9] * e[20] * e[26] + e[12] * e[22] * e[25] + e[12] * e[23] * e[26] + e[18] * e[20] * e[17] + e[18] * e[11] * e[26] + e[18] * e[19] * e[16] + e[18] * e[10] * e[25] - 1.*e[24] * e[23] * e[14] - 1.*e[24] * e[20] * e[11] - 1.*e[24] * e[19] * e[10] - 1.*e[24] * e[22] * e[13] + .5000000000*ep2[21] * e[15] + 1.500000000*ep2[24] * e[15] + .5000000000*e[15] * ep2[25] + .5000000000*e[15] * ep2[26] + .5000000000*e[15] * ep2[18] - .5000000000*e[15] * ep2[23] - .5000000000*e[15] * ep2[19] - .5000000000*e[15] * ep2[20] - .5000000000*e[15] * ep2[22];
	A[18] = e[6] * e[1] * e[14] + e[15] * e[1] * e[5] - e[0] * e[5] * e[16] - e[0] * e[14] * e[7] + e[0] * e[13] * e[8] - e[15] * e[4] * e[2] + e[12] * e[7] * e[2] + e[6] * e[10] * e[5] + e[3] * e[7] * e[11] - e[6] * e[4] * e[11] + e[3] * e[16] * e[2] - e[6] * e[13] * e[2] - e[3] * e[1] * e[17] - e[9] * e[5] * e[7] - e[3] * e[10] * e[8] - e[12] * e[1] * e[8] + e[0] * e[4] * e[17] + e[9] * e[4] * e[8];
	A[128] = -.5000000000*e[14] * ep2[16] - .5000000000*e[14] * ep2[10] - .5000000000*e[14] * ep2[9] + e[11] * e[9] * e[12] + .5000000000*ep3[14] + e[17] * e[13] * e[16] + .5000000000*e[14] * ep2[12] + e[11] * e[10] * e[13] - .5000000000*e[14] * ep2[15] + .5000000000*e[14] * ep2[17] + e[17] * e[12] * e[15] + .5000000000*ep2[11] * e[14] + .5000000000*e[14] * ep2[13];
	A[19] = -e[21] * e[19] * e[8] + e[18] * e[4] * e[26] - e[18] * e[5] * e[25] - e[18] * e[23] * e[7] + e[21] * e[25] * e[2] - e[21] * e[1] * e[26] + e[6] * e[19] * e[23] + e[18] * e[22] * e[8] - e[0] * e[23] * e[25] - e[6] * e[22] * e[20] + e[24] * e[1] * e[23] + e[24] * e[19] * e[5] - e[24] * e[4] * e[20] - e[24] * e[22] * e[2] + e[3] * e[25] * e[20] - e[3] * e[19] * e[26] + e[0] * e[22] * e[26] + e[21] * e[7] * e[20];
	A[129] = .5000000000*ep2[20] * e[32] + 1.500000000*e[32] * ep2[23] + .5000000000*e[32] * ep2[22] + .5000000000*e[32] * ep2[21] + .5000000000*e[32] * ep2[26] - .5000000000*e[32] * ep2[18] - .5000000000*e[32] * ep2[19] - .5000000000*e[32] * ep2[24] - .5000000000*e[32] * ep2[25] + e[20] * e[27] * e[21] + e[20] * e[18] * e[30] + e[20] * e[28] * e[22] + e[20] * e[19] * e[31] + e[20] * e[29] * e[23] + e[29] * e[19] * e[22] + e[29] * e[18] * e[21] + e[23] * e[30] * e[21] + e[23] * e[31] * e[22] + e[26] * e[30] * e[24] + e[26] * e[21] * e[33] + e[26] * e[31] * e[25] + e[26] * e[22] * e[34] + e[26] * e[23] * e[35] + e[35] * e[22] * e[25] + e[35] * e[21] * e[24] - 1.*e[23] * e[27] * e[18] - 1.*e[23] * e[33] * e[24] - 1.*e[23] * e[28] * e[19] - 1.*e[23] * e[34] * e[25];
	A[16] = -e[9] * e[23] * e[25] - e[21] * e[10] * e[26] - e[21] * e[19] * e[17] - e[18] * e[23] * e[16] + e[18] * e[13] * e[26] + e[12] * e[25] * e[20] - e[12] * e[19] * e[26] - e[15] * e[22] * e[20] + e[21] * e[16] * e[20] + e[21] * e[25] * e[11] + e[24] * e[10] * e[23] + e[24] * e[19] * e[14] - e[24] * e[13] * e[20] - e[24] * e[22] * e[11] + e[18] * e[22] * e[17] - e[18] * e[14] * e[25] + e[9] * e[22] * e[26] + e[15] * e[19] * e[23];
	A[130] = .5000000000*e[23] * ep2[21] + e[20] * e[19] * e[22] + e[20] * e[18] * e[21] + .5000000000*ep3[23] + e[26] * e[22] * e[25] + .5000000000*e[23] * ep2[26] - .5000000000*e[23] * ep2[18] + .5000000000*e[23] * ep2[22] - .5000000000*e[23] * ep2[19] + e[26] * e[21] * e[24] + .5000000000*ep2[20] * e[23] - .5000000000*e[23] * ep2[24] - .5000000000*e[23] * ep2[25];
	A[17] = e[18] * e[13] * e[35] - e[18] * e[32] * e[16] + e[18] * e[31] * e[17] - e[18] * e[14] * e[34] + e[27] * e[13] * e[26] + e[27] * e[22] * e[17] - e[27] * e[14] * e[25] - e[27] * e[23] * e[16] - e[9] * e[32] * e[25] + e[9] * e[22] * e[35] - e[9] * e[23] * e[34] + e[9] * e[31] * e[26] + e[15] * e[19] * e[32] + e[15] * e[28] * e[23] - e[15] * e[22] * e[29] - e[15] * e[31] * e[20] + e[24] * e[10] * e[32] + e[24] * e[28] * e[14] - e[24] * e[13] * e[29] - e[24] * e[31] * e[11] + e[33] * e[10] * e[23] + e[33] * e[19] * e[14] - e[33] * e[13] * e[20] - e[33] * e[22] * e[11] + e[21] * e[16] * e[29] - e[21] * e[10] * e[35] - e[21] * e[28] * e[17] + e[30] * e[16] * e[20] + e[30] * e[25] * e[11] - e[30] * e[10] * e[26] - e[30] * e[19] * e[17] - e[12] * e[28] * e[26] - e[12] * e[19] * e[35] + e[12] * e[34] * e[20] + e[12] * e[25] * e[29] + e[21] * e[34] * e[11];
	A[131] = -1.*e[32] * e[10] * e[1] + e[32] * e[13] * e[4] - 1.*e[32] * e[16] * e[7] - 1.*e[32] * e[15] * e[6] - 1.*e[32] * e[9] * e[0] + e[32] * e[12] * e[3] + e[17] * e[30] * e[6] + e[17] * e[3] * e[33] + e[17] * e[31] * e[7] + e[17] * e[4] * e[34] + e[17] * e[5] * e[35] - 1.*e[5] * e[27] * e[9] - 1.*e[5] * e[28] * e[10] - 1.*e[5] * e[33] * e[15] - 1.*e[5] * e[34] * e[16] + e[5] * e[29] * e[11] + e[35] * e[12] * e[6] + e[35] * e[3] * e[15] + e[35] * e[13] * e[7] + e[35] * e[4] * e[16] + e[11] * e[27] * e[3] + e[11] * e[0] * e[30] + e[11] * e[28] * e[4] + e[11] * e[1] * e[31] + e[29] * e[9] * e[3] + e[29] * e[0] * e[12] + e[29] * e[10] * e[4] + e[29] * e[1] * e[13] + e[5] * e[30] * e[12] + 3.*e[5] * e[32] * e[14] + e[5] * e[31] * e[13] + e[8] * e[30] * e[15] + e[8] * e[12] * e[33] + e[8] * e[32] * e[17] + e[8] * e[14] * e[35] + e[8] * e[31] * e[16] + e[8] * e[13] * e[34] + e[2] * e[27] * e[12] + e[2] * e[9] * e[30] + e[2] * e[29] * e[14] + e[2] * e[11] * e[32] + e[2] * e[28] * e[13] + e[2] * e[10] * e[31] - 1.*e[14] * e[27] * e[0] - 1.*e[14] * e[34] * e[7] - 1.*e[14] * e[33] * e[6] + e[14] * e[30] * e[3] - 1.*e[14] * e[28] * e[1] + e[14] * e[31] * e[4];
	A[22] = .5000000000*e[18] * ep2[29] + .5000000000*e[18] * ep2[28] + .5000000000*e[18] * ep2[30] + .5000000000*e[18] * ep2[33] - .5000000000*e[18] * ep2[32] - .5000000000*e[18] * ep2[31] - .5000000000*e[18] * ep2[34] - .5000000000*e[18] * ep2[35] + 1.500000000*e[18] * ep2[27] + e[27] * e[28] * e[19] + e[27] * e[29] * e[20] + e[21] * e[27] * e[30] + e[21] * e[29] * e[32] + e[21] * e[28] * e[31] + e[30] * e[28] * e[22] + e[30] * e[19] * e[31] + e[30] * e[29] * e[23] + e[30] * e[20] * e[32] + e[24] * e[27] * e[33] + e[24] * e[29] * e[35] + e[24] * e[28] * e[34] + e[33] * e[28] * e[25] + e[33] * e[19] * e[34] + e[33] * e[29] * e[26] + e[33] * e[20] * e[35] - 1.*e[27] * e[35] * e[26] - 1.*e[27] * e[31] * e[22] - 1.*e[27] * e[32] * e[23] - 1.*e[27] * e[34] * e[25];
	A[132] = e[20] * e[1] * e[4] + e[20] * e[0] * e[3] + e[20] * e[2] * e[5] + e[5] * e[21] * e[3] + e[5] * e[22] * e[4] + e[8] * e[21] * e[6] + e[8] * e[3] * e[24] + e[8] * e[22] * e[7] + e[8] * e[4] * e[25] + e[8] * e[5] * e[26] + e[26] * e[4] * e[7] + e[26] * e[3] * e[6] + e[2] * e[18] * e[3] + e[2] * e[0] * e[21] + e[2] * e[19] * e[4] + e[2] * e[1] * e[22] - 1.*e[5] * e[19] * e[1] - 1.*e[5] * e[18] * e[0] - 1.*e[5] * e[25] * e[7] - 1.*e[5] * e[24] * e[6] + .5000000000*e[23] * ep2[4] - .5000000000*e[23] * ep2[0] - .5000000000*e[23] * ep2[6] + 1.500000000*e[23] * ep2[5] - .5000000000*e[23] * ep2[1] - .5000000000*e[23] * ep2[7] + .5000000000*e[23] * ep2[3] + .5000000000*e[23] * ep2[2] + .5000000000*e[23] * ep2[8];
	A[23] = 1.500000000*e[9] * ep2[27] + .5000000000*e[9] * ep2[29] + .5000000000*e[9] * ep2[28] - .5000000000*e[9] * ep2[32] - .5000000000*e[9] * ep2[31] + .5000000000*e[9] * ep2[33] + .5000000000*e[9] * ep2[30] - .5000000000*e[9] * ep2[34] - .5000000000*e[9] * ep2[35] + e[33] * e[27] * e[15] + e[33] * e[29] * e[17] + e[33] * e[11] * e[35] + e[33] * e[28] * e[16] + e[33] * e[10] * e[34] + e[27] * e[29] * e[11] + e[27] * e[28] * e[10] + e[27] * e[30] * e[12] - 1.*e[27] * e[31] * e[13] - 1.*e[27] * e[32] * e[14] - 1.*e[27] * e[34] * e[16] - 1.*e[27] * e[35] * e[17] + e[30] * e[29] * e[14] + e[30] * e[11] * e[32] + e[30] * e[28] * e[13] + e[30] * e[10] * e[31] + e[12] * e[29] * e[32] + e[12] * e[28] * e[31] + e[15] * e[29] * e[35] + e[15] * e[28] * e[34];
	A[133] = -1.*e[32] * e[24] * e[6] + e[8] * e[30] * e[24] + e[8] * e[21] * e[33] + e[8] * e[31] * e[25] + e[8] * e[22] * e[34] + e[26] * e[30] * e[6] + e[26] * e[3] * e[33] + e[26] * e[31] * e[7] + e[26] * e[4] * e[34] + e[26] * e[32] * e[8] + e[26] * e[5] * e[35] + e[35] * e[21] * e[6] + e[35] * e[3] * e[24] + e[35] * e[22] * e[7] + e[35] * e[4] * e[25] + e[35] * e[23] * e[8] + e[2] * e[27] * e[21] + e[2] * e[18] * e[30] + e[2] * e[28] * e[22] + e[2] * e[19] * e[31] + e[2] * e[29] * e[23] + e[2] * e[20] * e[32] + e[20] * e[27] * e[3] + e[20] * e[0] * e[30] + e[20] * e[28] * e[4] + e[20] * e[1] * e[31] + e[20] * e[29] * e[5] + e[29] * e[18] * e[3] + e[29] * e[0] * e[21] + e[29] * e[19] * e[4] + e[29] * e[1] * e[22] + e[5] * e[30] * e[21] + e[5] * e[31] * e[22] + 3.*e[5] * e[32] * e[23] - 1.*e[5] * e[27] * e[18] - 1.*e[5] * e[33] * e[24] - 1.*e[5] * e[28] * e[19] - 1.*e[5] * e[34] * e[25] - 1.*e[23] * e[27] * e[0] - 1.*e[23] * e[34] * e[7] - 1.*e[23] * e[33] * e[6] + e[23] * e[30] * e[3] - 1.*e[23] * e[28] * e[1] + e[23] * e[31] * e[4] + e[32] * e[21] * e[3] - 1.*e[32] * e[19] * e[1] + e[32] * e[22] * e[4] - 1.*e[32] * e[18] * e[0] - 1.*e[32] * e[25] * e[7];
	A[20] = .5000000000*e[27] * ep2[33] - .5000000000*e[27] * ep2[32] - .5000000000*e[27] * ep2[31] - .5000000000*e[27] * ep2[34] - .5000000000*e[27] * ep2[35] + e[33] * e[29] * e[35] + .5000000000*e[27] * ep2[29] + e[30] * e[29] * e[32] + e[30] * e[28] * e[31] + e[33] * e[28] * e[34] + .5000000000*e[27] * ep2[28] + .5000000000*e[27] * ep2[30] + .5000000000*ep3[27];
	A[134] = e[14] * e[21] * e[12] + e[14] * e[22] * e[13] + e[17] * e[21] * e[15] + e[17] * e[12] * e[24] + e[17] * e[14] * e[26] + e[17] * e[22] * e[16] + e[17] * e[13] * e[25] + e[26] * e[12] * e[15] + e[26] * e[13] * e[16] - 1.*e[14] * e[24] * e[15] - 1.*e[14] * e[25] * e[16] - 1.*e[14] * e[18] * e[9] - 1.*e[14] * e[19] * e[10] + e[11] * e[18] * e[12] + e[11] * e[9] * e[21] + e[11] * e[19] * e[13] + e[11] * e[10] * e[22] + e[20] * e[11] * e[14] + e[20] * e[9] * e[12] + e[20] * e[10] * e[13] + 1.500000000*e[23] * ep2[14] + .5000000000*e[23] * ep2[12] + .5000000000*e[23] * ep2[13] + .5000000000*e[23] * ep2[17] + .5000000000*ep2[11] * e[23] - .5000000000*e[23] * ep2[16] - .5000000000*e[23] * ep2[9] - .5000000000*e[23] * ep2[15] - .5000000000*e[23] * ep2[10];
	A[21] = 1.500000000*e[0] * ep2[27] + .5000000000*e[0] * ep2[29] + .5000000000*e[0] * ep2[28] + .5000000000*e[0] * ep2[30] - .5000000000*e[0] * ep2[32] - .5000000000*e[0] * ep2[31] + .5000000000*e[0] * ep2[33] - .5000000000*e[0] * ep2[34] - .5000000000*e[0] * ep2[35] - 1.*e[27] * e[31] * e[4] + e[3] * e[27] * e[30] + e[3] * e[29] * e[32] + e[3] * e[28] * e[31] + e[30] * e[28] * e[4] + e[30] * e[1] * e[31] + e[30] * e[29] * e[5] + e[30] * e[2] * e[32] + e[6] * e[27] * e[33] + e[6] * e[29] * e[35] + e[6] * e[28] * e[34] + e[27] * e[28] * e[1] + e[27] * e[29] * e[2] + e[33] * e[28] * e[7] + e[33] * e[1] * e[34] + e[33] * e[29] * e[8] + e[33] * e[2] * e[35] - 1.*e[27] * e[34] * e[7] - 1.*e[27] * e[32] * e[5] - 1.*e[27] * e[35] * e[8];
	A[135] = e[14] * e[12] * e[3] + e[14] * e[13] * e[4] + e[17] * e[12] * e[6] + e[17] * e[3] * e[15] + e[17] * e[13] * e[7] + e[17] * e[4] * e[16] + e[17] * e[14] * e[8] + e[8] * e[12] * e[15] + e[8] * e[13] * e[16] + e[2] * e[11] * e[14] + e[2] * e[9] * e[12] + e[2] * e[10] * e[13] + e[11] * e[9] * e[3] + e[11] * e[0] * e[12] + e[11] * e[10] * e[4] + e[11] * e[1] * e[13] - 1.*e[14] * e[10] * e[1] - 1.*e[14] * e[16] * e[7] - 1.*e[14] * e[15] * e[6] - 1.*e[14] * e[9] * e[0] - .5000000000*e[5] * ep2[16] - .5000000000*e[5] * ep2[9] + .5000000000*e[5] * ep2[11] + .5000000000*e[5] * ep2[12] - .5000000000*e[5] * ep2[15] - .5000000000*e[5] * ep2[10] + .5000000000*e[5] * ep2[13] + 1.500000000*ep2[14] * e[5] + .5000000000*e[5] * ep2[17];
	A[27] = 1.500000000*e[27] * ep2[9] - .5000000000*e[27] * ep2[16] + .5000000000*e[27] * ep2[11] + .5000000000*e[27] * ep2[12] + .5000000000*e[27] * ep2[15] - .5000000000*e[27] * ep2[17] + .5000000000*e[27] * ep2[10] - .5000000000*e[27] * ep2[14] - .5000000000*e[27] * ep2[13] + e[12] * e[10] * e[31] + e[30] * e[11] * e[14] + e[30] * e[10] * e[13] + e[15] * e[9] * e[33] + e[15] * e[29] * e[17] + e[15] * e[11] * e[35] + e[15] * e[28] * e[16] + e[15] * e[10] * e[34] + e[33] * e[11] * e[17] + e[33] * e[10] * e[16] - 1.*e[9] * e[31] * e[13] - 1.*e[9] * e[32] * e[14] - 1.*e[9] * e[34] * e[16] - 1.*e[9] * e[35] * e[17] + e[9] * e[29] * e[11] + e[9] * e[28] * e[10] + e[12] * e[9] * e[30] + e[12] * e[29] * e[14] + e[12] * e[11] * e[32] + e[12] * e[28] * e[13];
	A[137] = e[29] * e[18] * e[12] + e[29] * e[9] * e[21] + e[29] * e[19] * e[13] + e[29] * e[10] * e[22] + e[17] * e[30] * e[24] + e[17] * e[21] * e[33] + e[17] * e[31] * e[25] + e[17] * e[22] * e[34] + e[17] * e[32] * e[26] + e[17] * e[23] * e[35] - 1.*e[23] * e[27] * e[9] - 1.*e[23] * e[28] * e[10] - 1.*e[23] * e[33] * e[15] - 1.*e[23] * e[34] * e[16] - 1.*e[32] * e[24] * e[15] - 1.*e[32] * e[25] * e[16] - 1.*e[32] * e[18] * e[9] - 1.*e[32] * e[19] * e[10] + e[26] * e[30] * e[15] + e[26] * e[12] * e[33] + e[26] * e[31] * e[16] + e[26] * e[13] * e[34] + e[35] * e[21] * e[15] + e[35] * e[12] * e[24] + e[35] * e[22] * e[16] + e[35] * e[13] * e[25] + e[14] * e[30] * e[21] + e[14] * e[31] * e[22] + 3.*e[14] * e[32] * e[23] + e[11] * e[27] * e[21] + e[11] * e[18] * e[30] + e[11] * e[28] * e[22] + e[11] * e[19] * e[31] + e[11] * e[29] * e[23] + e[11] * e[20] * e[32] + e[23] * e[30] * e[12] + e[23] * e[31] * e[13] + e[32] * e[21] * e[12] + e[32] * e[22] * e[13] - 1.*e[14] * e[27] * e[18] - 1.*e[14] * e[33] * e[24] + e[14] * e[29] * e[20] + e[14] * e[35] * e[26] - 1.*e[14] * e[28] * e[19] - 1.*e[14] * e[34] * e[25] + e[20] * e[27] * e[12] + e[20] * e[9] * e[30] + e[20] * e[28] * e[13] + e[20] * e[10] * e[31];
	A[26] = .5000000000*e[0] * ep2[1] + .5000000000*e[0] * ep2[2] + e[6] * e[2] * e[8] + e[6] * e[1] * e[7] + .5000000000*e[0] * ep2[3] + e[3] * e[1] * e[4] + .5000000000*e[0] * ep2[6] + e[3] * e[2] * e[5] - .5000000000*e[0] * ep2[5] - .5000000000*e[0] * ep2[8] + .5000000000*ep3[0] - .5000000000*e[0] * ep2[7] - .5000000000*e[0] * ep2[4];
	A[136] = 1.500000000*ep2[23] * e[14] + .5000000000*e[14] * ep2[26] - .5000000000*e[14] * ep2[18] - .5000000000*e[14] * ep2[19] + .5000000000*e[14] * ep2[20] + .5000000000*e[14] * ep2[22] - .5000000000*e[14] * ep2[24] + .5000000000*e[14] * ep2[21] - .5000000000*e[14] * ep2[25] + e[23] * e[21] * e[12] + e[23] * e[22] * e[13] + e[26] * e[21] * e[15] + e[26] * e[12] * e[24] + e[26] * e[23] * e[17] + e[26] * e[22] * e[16] + e[26] * e[13] * e[25] + e[17] * e[22] * e[25] + e[17] * e[21] * e[24] + e[11] * e[19] * e[22] + e[11] * e[18] * e[21] + e[11] * e[20] * e[23] + e[20] * e[18] * e[12] + e[20] * e[9] * e[21] + e[20] * e[19] * e[13] + e[20] * e[10] * e[22] - 1.*e[23] * e[24] * e[15] - 1.*e[23] * e[25] * e[16] - 1.*e[23] * e[18] * e[9] - 1.*e[23] * e[19] * e[10];
	A[25] = 1.500000000*e[27] * ep2[0] - .5000000000*e[27] * ep2[4] + .5000000000*e[27] * ep2[6] - .5000000000*e[27] * ep2[5] + .5000000000*e[27] * ep2[1] - .5000000000*e[27] * ep2[7] + .5000000000*e[27] * ep2[3] + .5000000000*e[27] * ep2[2] - .5000000000*e[27] * ep2[8] + e[0] * e[33] * e[6] + e[0] * e[30] * e[3] - 1.*e[0] * e[35] * e[8] - 1.*e[0] * e[31] * e[4] + e[3] * e[28] * e[4] + e[3] * e[1] * e[31] + e[3] * e[29] * e[5] + e[3] * e[2] * e[32] + e[30] * e[1] * e[4] + e[30] * e[2] * e[5] + e[6] * e[28] * e[7] + e[6] * e[1] * e[34] + e[6] * e[29] * e[8] + e[6] * e[2] * e[35] + e[33] * e[1] * e[7] + e[33] * e[2] * e[8] + e[0] * e[28] * e[1] + e[0] * e[29] * e[2] - 1.*e[0] * e[34] * e[7] - 1.*e[0] * e[32] * e[5];
	A[139] = e[8] * e[22] * e[25] + e[8] * e[21] * e[24] + e[20] * e[18] * e[3] + e[20] * e[0] * e[21] + e[20] * e[19] * e[4] + e[20] * e[1] * e[22] + e[20] * e[2] * e[23] + e[23] * e[21] * e[3] + e[23] * e[22] * e[4] + e[23] * e[26] * e[8] - 1.*e[23] * e[19] * e[1] - 1.*e[23] * e[18] * e[0] - 1.*e[23] * e[25] * e[7] - 1.*e[23] * e[24] * e[6] + e[2] * e[19] * e[22] + e[2] * e[18] * e[21] + e[26] * e[21] * e[6] + e[26] * e[3] * e[24] + e[26] * e[22] * e[7] + e[26] * e[4] * e[25] + .5000000000*ep2[20] * e[5] + 1.500000000*ep2[23] * e[5] + .5000000000*e[5] * ep2[22] + .5000000000*e[5] * ep2[21] + .5000000000*e[5] * ep2[26] - .5000000000*e[5] * ep2[18] - .5000000000*e[5] * ep2[19] - .5000000000*e[5] * ep2[24] - .5000000000*e[5] * ep2[25];
	A[24] = e[24] * e[11] * e[8] + e[24] * e[2] * e[17] + 3.*e[9] * e[18] * e[0] + e[9] * e[19] * e[1] + e[9] * e[20] * e[2] + e[18] * e[10] * e[1] + e[18] * e[11] * e[2] + e[3] * e[18] * e[12] + e[3] * e[9] * e[21] + e[3] * e[20] * e[14] + e[3] * e[11] * e[23] + e[3] * e[19] * e[13] + e[3] * e[10] * e[22] + e[6] * e[18] * e[15] + e[6] * e[9] * e[24] + e[6] * e[20] * e[17] + e[6] * e[11] * e[26] + e[6] * e[19] * e[16] + e[6] * e[10] * e[25] + e[0] * e[20] * e[11] + e[0] * e[19] * e[10] - 1.*e[9] * e[26] * e[8] - 1.*e[9] * e[22] * e[4] - 1.*e[9] * e[25] * e[7] - 1.*e[9] * e[23] * e[5] + e[12] * e[0] * e[21] + e[12] * e[19] * e[4] + e[12] * e[1] * e[22] + e[12] * e[20] * e[5] + e[12] * e[2] * e[23] - 1.*e[18] * e[13] * e[4] - 1.*e[18] * e[16] * e[7] - 1.*e[18] * e[14] * e[5] - 1.*e[18] * e[17] * e[8] + e[21] * e[10] * e[4] + e[21] * e[1] * e[13] + e[21] * e[11] * e[5] + e[21] * e[2] * e[14] + e[15] * e[0] * e[24] + e[15] * e[19] * e[7] + e[15] * e[1] * e[25] + e[15] * e[20] * e[8] + e[15] * e[2] * e[26] - 1.*e[0] * e[23] * e[14] - 1.*e[0] * e[25] * e[16] - 1.*e[0] * e[26] * e[17] - 1.*e[0] * e[22] * e[13] + e[24] * e[10] * e[7] + e[24] * e[1] * e[16];
	A[138] = e[11] * e[1] * e[4] + e[11] * e[0] * e[3] + e[11] * e[2] * e[5] + e[5] * e[12] * e[3] + e[5] * e[13] * e[4] + e[8] * e[12] * e[6] + e[8] * e[3] * e[15] + e[8] * e[13] * e[7] + e[8] * e[4] * e[16] + e[8] * e[5] * e[17] + e[17] * e[4] * e[7] + e[17] * e[3] * e[6] - 1.*e[5] * e[10] * e[1] - 1.*e[5] * e[16] * e[7] - 1.*e[5] * e[15] * e[6] - 1.*e[5] * e[9] * e[0] + e[2] * e[9] * e[3] + e[2] * e[0] * e[12] + e[2] * e[10] * e[4] + e[2] * e[1] * e[13] + .5000000000*ep2[2] * e[14] - .5000000000*e[14] * ep2[0] - .5000000000*e[14] * ep2[6] - .5000000000*e[14] * ep2[1] - .5000000000*e[14] * ep2[7] + 1.500000000*e[14] * ep2[5] + .5000000000*e[14] * ep2[4] + .5000000000*e[14] * ep2[3] + .5000000000*e[14] * ep2[8];
	A[31] = e[3] * e[27] * e[12] + e[3] * e[9] * e[30] + e[3] * e[29] * e[14] + e[3] * e[11] * e[32] + e[3] * e[28] * e[13] + e[3] * e[10] * e[31] + e[6] * e[27] * e[15] + e[6] * e[9] * e[33] + e[6] * e[29] * e[17] + e[6] * e[11] * e[35] + e[6] * e[28] * e[16] + e[6] * e[10] * e[34] + 3.*e[0] * e[27] * e[9] + e[0] * e[29] * e[11] + e[0] * e[28] * e[10] - 1.*e[9] * e[34] * e[7] - 1.*e[9] * e[32] * e[5] - 1.*e[9] * e[35] * e[8] + e[9] * e[29] * e[2] + e[9] * e[28] * e[1] - 1.*e[9] * e[31] * e[4] + e[12] * e[0] * e[30] + e[12] * e[28] * e[4] + e[12] * e[1] * e[31] + e[12] * e[29] * e[5] + e[12] * e[2] * e[32] + e[27] * e[11] * e[2] + e[27] * e[10] * e[1] - 1.*e[27] * e[13] * e[4] - 1.*e[27] * e[16] * e[7] - 1.*e[27] * e[14] * e[5] - 1.*e[27] * e[17] * e[8] + e[30] * e[10] * e[4] + e[30] * e[1] * e[13] + e[30] * e[11] * e[5] + e[30] * e[2] * e[14] + e[15] * e[0] * e[33] + e[15] * e[28] * e[7] + e[15] * e[1] * e[34] + e[15] * e[29] * e[8] + e[15] * e[2] * e[35] - 1.*e[0] * e[31] * e[13] - 1.*e[0] * e[32] * e[14] - 1.*e[0] * e[34] * e[16] - 1.*e[0] * e[35] * e[17] + e[33] * e[10] * e[7] + e[33] * e[1] * e[16] + e[33] * e[11] * e[8] + e[33] * e[2] * e[17];
	A[141] = .5000000000*ep2[30] * e[6] + .5000000000*e[6] * ep2[27] - .5000000000*e[6] * ep2[32] - .5000000000*e[6] * ep2[28] - .5000000000*e[6] * ep2[29] - .5000000000*e[6] * ep2[31] + 1.500000000*e[6] * ep2[33] + .5000000000*e[6] * ep2[34] + .5000000000*e[6] * ep2[35] + e[0] * e[27] * e[33] + e[0] * e[29] * e[35] + e[0] * e[28] * e[34] + e[3] * e[30] * e[33] + e[3] * e[32] * e[35] + e[3] * e[31] * e[34] + e[30] * e[31] * e[7] + e[30] * e[4] * e[34] + e[30] * e[32] * e[8] + e[30] * e[5] * e[35] + e[27] * e[28] * e[7] + e[27] * e[1] * e[34] + e[27] * e[29] * e[8] + e[27] * e[2] * e[35] + e[33] * e[34] * e[7] + e[33] * e[35] * e[8] - 1.*e[33] * e[32] * e[5] - 1.*e[33] * e[29] * e[2] - 1.*e[33] * e[28] * e[1] - 1.*e[33] * e[31] * e[4];
	A[30] = e[24] * e[20] * e[26] + e[21] * e[19] * e[22] - .5000000000*e[18] * ep2[22] - .5000000000*e[18] * ep2[25] + .5000000000*ep3[18] + .5000000000*e[18] * ep2[21] + e[21] * e[20] * e[23] + .5000000000*e[18] * ep2[20] + .5000000000*e[18] * ep2[19] + .5000000000*e[18] * ep2[24] + e[24] * e[19] * e[25] - .5000000000*e[18] * ep2[23] - .5000000000*e[18] * ep2[26];
	A[140] = .5000000000*e[33] * ep2[35] + .5000000000*ep3[33] + .5000000000*ep2[27] * e[33] + .5000000000*ep2[30] * e[33] - .5000000000*e[33] * ep2[29] + .5000000000*e[33] * ep2[34] - .5000000000*e[33] * ep2[32] - .5000000000*e[33] * ep2[28] + e[30] * e[32] * e[35] - .5000000000*e[33] * ep2[31] + e[27] * e[29] * e[35] + e[27] * e[28] * e[34] + e[30] * e[31] * e[34];
	A[29] = 1.500000000*e[27] * ep2[18] + .5000000000*e[27] * ep2[19] + .5000000000*e[27] * ep2[20] + .5000000000*e[27] * ep2[21] + .5000000000*e[27] * ep2[24] - .5000000000*e[27] * ep2[26] - .5000000000*e[27] * ep2[23] - .5000000000*e[27] * ep2[22] - .5000000000*e[27] * ep2[25] + e[33] * e[20] * e[26] - 1.*e[18] * e[35] * e[26] - 1.*e[18] * e[31] * e[22] - 1.*e[18] * e[32] * e[23] - 1.*e[18] * e[34] * e[25] + e[18] * e[28] * e[19] + e[18] * e[29] * e[20] + e[21] * e[18] * e[30] + e[21] * e[28] * e[22] + e[21] * e[19] * e[31] + e[21] * e[29] * e[23] + e[21] * e[20] * e[32] + e[30] * e[19] * e[22] + e[30] * e[20] * e[23] + e[24] * e[18] * e[33] + e[24] * e[28] * e[25] + e[24] * e[19] * e[34] + e[24] * e[29] * e[26] + e[24] * e[20] * e[35] + e[33] * e[19] * e[25];
	A[143] = e[9] * e[27] * e[33] + e[9] * e[29] * e[35] + e[9] * e[28] * e[34] + e[33] * e[35] * e[17] + e[33] * e[34] * e[16] + e[27] * e[29] * e[17] + e[27] * e[11] * e[35] + e[27] * e[28] * e[16] + e[27] * e[10] * e[34] + e[33] * e[30] * e[12] - 1.*e[33] * e[28] * e[10] - 1.*e[33] * e[31] * e[13] - 1.*e[33] * e[32] * e[14] - 1.*e[33] * e[29] * e[11] + e[30] * e[32] * e[17] + e[30] * e[14] * e[35] + e[30] * e[31] * e[16] + e[30] * e[13] * e[34] + e[12] * e[32] * e[35] + e[12] * e[31] * e[34] + .5000000000*e[15] * ep2[27] - .5000000000*e[15] * ep2[32] - .5000000000*e[15] * ep2[28] - .5000000000*e[15] * ep2[29] - .5000000000*e[15] * ep2[31] + 1.500000000*e[15] * ep2[33] + .5000000000*e[15] * ep2[30] + .5000000000*e[15] * ep2[34] + .5000000000*e[15] * ep2[35];
	A[28] = .5000000000*e[9] * ep2[12] - .5000000000*e[9] * ep2[16] + .5000000000*e[9] * ep2[10] - .5000000000*e[9] * ep2[17] - .5000000000*e[9] * ep2[13] + e[15] * e[10] * e[16] + e[12] * e[11] * e[14] + .5000000000*e[9] * ep2[11] + .5000000000*e[9] * ep2[15] - .5000000000*e[9] * ep2[14] + e[15] * e[11] * e[17] + .5000000000*ep3[9] + e[12] * e[10] * e[13];
	A[142] = e[18] * e[27] * e[33] + e[18] * e[29] * e[35] + e[18] * e[28] * e[34] + e[27] * e[28] * e[25] + e[27] * e[19] * e[34] + e[27] * e[29] * e[26] + e[27] * e[20] * e[35] + e[21] * e[30] * e[33] + e[21] * e[32] * e[35] + e[21] * e[31] * e[34] + e[30] * e[31] * e[25] + e[30] * e[22] * e[34] + e[30] * e[32] * e[26] + e[30] * e[23] * e[35] + e[33] * e[34] * e[25] + e[33] * e[35] * e[26] - 1.*e[33] * e[29] * e[20] - 1.*e[33] * e[31] * e[22] - 1.*e[33] * e[32] * e[23] - 1.*e[33] * e[28] * e[19] + .5000000000*ep2[27] * e[24] + .5000000000*ep2[30] * e[24] + 1.500000000*e[24] * ep2[33] + .5000000000*e[24] * ep2[35] + .5000000000*e[24] * ep2[34] - .5000000000*e[24] * ep2[32] - .5000000000*e[24] * ep2[28] - .5000000000*e[24] * ep2[29] - .5000000000*e[24] * ep2[31];
	A[36] = .5000000000*e[9] * ep2[21] + .5000000000*e[9] * ep2[24] + .5000000000*e[9] * ep2[19] + 1.500000000*e[9] * ep2[18] + .5000000000*e[9] * ep2[20] - .5000000000*e[9] * ep2[26] - .5000000000*e[9] * ep2[23] - .5000000000*e[9] * ep2[22] - .5000000000*e[9] * ep2[25] + e[21] * e[18] * e[12] + e[21] * e[20] * e[14] + e[21] * e[11] * e[23] + e[21] * e[19] * e[13] + e[21] * e[10] * e[22] + e[24] * e[18] * e[15] + e[24] * e[20] * e[17] + e[24] * e[11] * e[26] + e[24] * e[19] * e[16] + e[24] * e[10] * e[25] + e[15] * e[19] * e[25] + e[15] * e[20] * e[26] + e[12] * e[19] * e[22] + e[12] * e[20] * e[23] + e[18] * e[20] * e[11] + e[18] * e[19] * e[10] - 1.*e[18] * e[23] * e[14] - 1.*e[18] * e[25] * e[16] - 1.*e[18] * e[26] * e[17] - 1.*e[18] * e[22] * e[13];
	A[182] = .5000000000*ep2[29] * e[26] + .5000000000*ep2[32] * e[26] + .5000000000*e[26] * ep2[33] + 1.500000000*e[26] * ep2[35] + .5000000000*e[26] * ep2[34] - .5000000000*e[26] * ep2[27] - .5000000000*e[26] * ep2[28] - .5000000000*e[26] * ep2[31] - .5000000000*e[26] * ep2[30] + e[20] * e[27] * e[33] + e[20] * e[29] * e[35] + e[20] * e[28] * e[34] + e[29] * e[27] * e[24] + e[29] * e[18] * e[33] + e[29] * e[28] * e[25] + e[29] * e[19] * e[34] + e[23] * e[30] * e[33] + e[23] * e[32] * e[35] + e[23] * e[31] * e[34] + e[32] * e[30] * e[24] + e[32] * e[21] * e[33] + e[32] * e[31] * e[25] + e[32] * e[22] * e[34] + e[35] * e[33] * e[24] + e[35] * e[34] * e[25] - 1.*e[35] * e[27] * e[18] - 1.*e[35] * e[30] * e[21] - 1.*e[35] * e[31] * e[22] - 1.*e[35] * e[28] * e[19];
	A[37] = e[12] * e[19] * e[31] + e[12] * e[29] * e[23] + e[12] * e[20] * e[32] + 3.*e[9] * e[27] * e[18] + e[9] * e[28] * e[19] + e[9] * e[29] * e[20] + e[21] * e[9] * e[30] + e[21] * e[29] * e[14] + e[21] * e[11] * e[32] + e[21] * e[28] * e[13] + e[21] * e[10] * e[31] + e[30] * e[20] * e[14] + e[30] * e[11] * e[23] + e[30] * e[19] * e[13] + e[30] * e[10] * e[22] + e[9] * e[33] * e[24] - 1.*e[9] * e[35] * e[26] - 1.*e[9] * e[31] * e[22] - 1.*e[9] * e[32] * e[23] - 1.*e[9] * e[34] * e[25] + e[18] * e[29] * e[11] + e[18] * e[28] * e[10] + e[27] * e[20] * e[11] + e[27] * e[19] * e[10] + e[15] * e[27] * e[24] + e[15] * e[18] * e[33] + e[15] * e[28] * e[25] + e[15] * e[19] * e[34] + e[15] * e[29] * e[26] + e[15] * e[20] * e[35] - 1.*e[18] * e[31] * e[13] - 1.*e[18] * e[32] * e[14] - 1.*e[18] * e[34] * e[16] - 1.*e[18] * e[35] * e[17] - 1.*e[27] * e[23] * e[14] - 1.*e[27] * e[25] * e[16] - 1.*e[27] * e[26] * e[17] - 1.*e[27] * e[22] * e[13] + e[24] * e[29] * e[17] + e[24] * e[11] * e[35] + e[24] * e[28] * e[16] + e[24] * e[10] * e[34] + e[33] * e[20] * e[17] + e[33] * e[11] * e[26] + e[33] * e[19] * e[16] + e[33] * e[10] * e[25] + e[12] * e[27] * e[21] + e[12] * e[18] * e[30] + e[12] * e[28] * e[22];
	A[183] = -.5000000000*e[17] * ep2[27] + .5000000000*e[17] * ep2[32] - .5000000000*e[17] * ep2[28] + .5000000000*e[17] * ep2[29] - .5000000000*e[17] * ep2[31] + .5000000000*e[17] * ep2[33] - .5000000000*e[17] * ep2[30] + .5000000000*e[17] * ep2[34] + 1.500000000*e[17] * ep2[35] + e[32] * e[30] * e[15] + e[32] * e[12] * e[33] + e[32] * e[31] * e[16] + e[32] * e[13] * e[34] + e[14] * e[30] * e[33] + e[14] * e[31] * e[34] + e[11] * e[27] * e[33] + e[11] * e[29] * e[35] + e[11] * e[28] * e[34] + e[35] * e[33] * e[15] + e[35] * e[34] * e[16] + e[29] * e[27] * e[15] + e[29] * e[9] * e[33] + e[29] * e[28] * e[16] + e[29] * e[10] * e[34] - 1.*e[35] * e[27] * e[9] - 1.*e[35] * e[30] * e[12] - 1.*e[35] * e[28] * e[10] - 1.*e[35] * e[31] * e[13] + e[35] * e[32] * e[14];
	A[38] = .5000000000*e[9] * ep2[1] + 1.500000000*e[9] * ep2[0] + .5000000000*e[9] * ep2[2] + .5000000000*e[9] * ep2[3] + .5000000000*e[9] * ep2[6] - .5000000000*e[9] * ep2[4] - .5000000000*e[9] * ep2[5] - .5000000000*e[9] * ep2[7] - .5000000000*e[9] * ep2[8] + e[6] * e[0] * e[15] + e[6] * e[10] * e[7] + e[6] * e[1] * e[16] + e[6] * e[11] * e[8] + e[6] * e[2] * e[17] + e[15] * e[1] * e[7] + e[15] * e[2] * e[8] + e[0] * e[11] * e[2] + e[0] * e[10] * e[1] - 1.*e[0] * e[13] * e[4] - 1.*e[0] * e[16] * e[7] - 1.*e[0] * e[14] * e[5] - 1.*e[0] * e[17] * e[8] + e[3] * e[0] * e[12] + e[3] * e[10] * e[4] + e[3] * e[1] * e[13] + e[3] * e[11] * e[5] + e[3] * e[2] * e[14] + e[12] * e[1] * e[4] + e[12] * e[2] * e[5];
	A[180] = .5000000000*e[35] * ep2[33] + .5000000000*e[35] * ep2[34] - .5000000000*e[35] * ep2[27] - .5000000000*e[35] * ep2[28] - .5000000000*e[35] * ep2[31] - .5000000000*e[35] * ep2[30] + e[32] * e[31] * e[34] + .5000000000*ep2[29] * e[35] + .5000000000*ep2[32] * e[35] + e[29] * e[28] * e[34] + e[32] * e[30] * e[33] + .5000000000*ep3[35] + e[29] * e[27] * e[33];
	A[39] = .5000000000*e[0] * ep2[19] + .5000000000*e[0] * ep2[20] + .5000000000*e[0] * ep2[24] - .5000000000*e[0] * ep2[26] - .5000000000*e[0] * ep2[23] - .5000000000*e[0] * ep2[22] - .5000000000*e[0] * ep2[25] + 1.500000000*ep2[18] * e[0] + .5000000000*e[0] * ep2[21] + e[18] * e[19] * e[1] + e[18] * e[20] * e[2] + e[21] * e[18] * e[3] + e[21] * e[19] * e[4] + e[21] * e[1] * e[22] + e[21] * e[20] * e[5] + e[21] * e[2] * e[23] - 1.*e[18] * e[26] * e[8] - 1.*e[18] * e[22] * e[4] - 1.*e[18] * e[25] * e[7] - 1.*e[18] * e[23] * e[5] + e[18] * e[24] * e[6] + e[3] * e[19] * e[22] + e[3] * e[20] * e[23] + e[24] * e[19] * e[7] + e[24] * e[1] * e[25] + e[24] * e[20] * e[8] + e[24] * e[2] * e[26] + e[6] * e[19] * e[25] + e[6] * e[20] * e[26];
	A[181] = .5000000000*ep2[32] * e[8] - .5000000000*e[8] * ep2[27] - .5000000000*e[8] * ep2[28] + .5000000000*e[8] * ep2[29] - .5000000000*e[8] * ep2[31] + .5000000000*e[8] * ep2[33] - .5000000000*e[8] * ep2[30] + .5000000000*e[8] * ep2[34] + 1.500000000*e[8] * ep2[35] + e[2] * e[27] * e[33] + e[2] * e[29] * e[35] + e[2] * e[28] * e[34] + e[5] * e[30] * e[33] + e[5] * e[32] * e[35] + e[5] * e[31] * e[34] + e[32] * e[30] * e[6] + e[32] * e[3] * e[33] + e[32] * e[31] * e[7] + e[32] * e[4] * e[34] + e[29] * e[27] * e[6] + e[29] * e[0] * e[33] + e[29] * e[28] * e[7] + e[29] * e[1] * e[34] + e[35] * e[33] * e[6] + e[35] * e[34] * e[7] - 1.*e[35] * e[27] * e[0] - 1.*e[35] * e[30] * e[3] - 1.*e[35] * e[28] * e[1] - 1.*e[35] * e[31] * e[4];
	A[32] = -.5000000000*e[18] * ep2[4] + 1.500000000*e[18] * ep2[0] + .5000000000*e[18] * ep2[6] - .5000000000*e[18] * ep2[5] + .5000000000*e[18] * ep2[1] - .5000000000*e[18] * ep2[7] + .5000000000*e[18] * ep2[3] + .5000000000*e[18] * ep2[2] - .5000000000*e[18] * ep2[8] + e[3] * e[0] * e[21] + e[3] * e[19] * e[4] + e[3] * e[1] * e[22] + e[3] * e[20] * e[5] + e[3] * e[2] * e[23] + e[21] * e[1] * e[4] + e[21] * e[2] * e[5] + e[6] * e[0] * e[24] + e[6] * e[19] * e[7] + e[6] * e[1] * e[25] + e[6] * e[20] * e[8] + e[6] * e[2] * e[26] + e[24] * e[1] * e[7] + e[24] * e[2] * e[8] + e[0] * e[19] * e[1] + e[0] * e[20] * e[2] - 1.*e[0] * e[26] * e[8] - 1.*e[0] * e[22] * e[4] - 1.*e[0] * e[25] * e[7] - 1.*e[0] * e[23] * e[5];
	A[178] = e[10] * e[1] * e[7] + e[10] * e[0] * e[6] + e[10] * e[2] * e[8] + e[4] * e[12] * e[6] + e[4] * e[3] * e[15] + e[4] * e[13] * e[7] + e[4] * e[14] * e[8] + e[4] * e[5] * e[17] + e[13] * e[3] * e[6] + e[13] * e[5] * e[8] + e[7] * e[15] * e[6] + e[7] * e[17] * e[8] - 1.*e[7] * e[11] * e[2] - 1.*e[7] * e[9] * e[0] - 1.*e[7] * e[14] * e[5] - 1.*e[7] * e[12] * e[3] + e[1] * e[9] * e[6] + e[1] * e[0] * e[15] + e[1] * e[11] * e[8] + e[1] * e[2] * e[17] + 1.500000000*e[16] * ep2[7] + .5000000000*e[16] * ep2[6] + .5000000000*e[16] * ep2[8] + .5000000000*ep2[1] * e[16] - .5000000000*e[16] * ep2[0] - .5000000000*e[16] * ep2[5] - .5000000000*e[16] * ep2[3] - .5000000000*e[16] * ep2[2] + .5000000000*ep2[4] * e[16];
	A[33] = e[0] * e[30] * e[21] - 1.*e[0] * e[35] * e[26] - 1.*e[0] * e[31] * e[22] - 1.*e[0] * e[32] * e[23] - 1.*e[0] * e[34] * e[25] - 1.*e[18] * e[34] * e[7] - 1.*e[18] * e[32] * e[5] - 1.*e[18] * e[35] * e[8] - 1.*e[18] * e[31] * e[4] - 1.*e[27] * e[26] * e[8] - 1.*e[27] * e[22] * e[4] - 1.*e[27] * e[25] * e[7] - 1.*e[27] * e[23] * e[5] + e[6] * e[28] * e[25] + e[6] * e[19] * e[34] + e[6] * e[29] * e[26] + e[6] * e[20] * e[35] + e[21] * e[28] * e[4] + e[21] * e[1] * e[31] + e[21] * e[29] * e[5] + e[21] * e[2] * e[32] + e[30] * e[19] * e[4] + e[30] * e[1] * e[22] + e[30] * e[20] * e[5] + e[30] * e[2] * e[23] + e[24] * e[27] * e[6] + e[24] * e[0] * e[33] + e[24] * e[28] * e[7] + e[24] * e[1] * e[34] + e[24] * e[29] * e[8] + e[24] * e[2] * e[35] + e[33] * e[18] * e[6] + e[33] * e[19] * e[7] + e[33] * e[1] * e[25] + e[33] * e[20] * e[8] + e[33] * e[2] * e[26] + 3.*e[0] * e[27] * e[18] + e[0] * e[28] * e[19] + e[0] * e[29] * e[20] + e[18] * e[28] * e[1] + e[18] * e[29] * e[2] + e[27] * e[19] * e[1] + e[27] * e[20] * e[2] + e[3] * e[27] * e[21] + e[3] * e[18] * e[30] + e[3] * e[28] * e[22] + e[3] * e[19] * e[31] + e[3] * e[29] * e[23] + e[3] * e[20] * e[32];
	A[179] = e[19] * e[18] * e[6] + e[19] * e[0] * e[24] + e[19] * e[1] * e[25] + e[19] * e[20] * e[8] + e[19] * e[2] * e[26] + e[22] * e[21] * e[6] + e[22] * e[3] * e[24] + e[22] * e[4] * e[25] + e[22] * e[23] * e[8] + e[22] * e[5] * e[26] - 1.*e[25] * e[21] * e[3] + e[25] * e[26] * e[8] - 1.*e[25] * e[20] * e[2] - 1.*e[25] * e[18] * e[0] - 1.*e[25] * e[23] * e[5] + e[25] * e[24] * e[6] + e[1] * e[18] * e[24] + e[1] * e[20] * e[26] + e[4] * e[21] * e[24] + e[4] * e[23] * e[26] + .5000000000*ep2[19] * e[7] + .5000000000*ep2[22] * e[7] + 1.500000000*ep2[25] * e[7] + .5000000000*e[7] * ep2[26] - .5000000000*e[7] * ep2[18] - .5000000000*e[7] * ep2[23] - .5000000000*e[7] * ep2[20] + .5000000000*e[7] * ep2[24] - .5000000000*e[7] * ep2[21];
	A[34] = .5000000000*e[18] * ep2[11] + 1.500000000*e[18] * ep2[9] + .5000000000*e[18] * ep2[10] + .5000000000*e[18] * ep2[12] + .5000000000*e[18] * ep2[15] - .5000000000*e[18] * ep2[16] - .5000000000*e[18] * ep2[17] - .5000000000*e[18] * ep2[14] - .5000000000*e[18] * ep2[13] + e[12] * e[9] * e[21] + e[12] * e[20] * e[14] + e[12] * e[11] * e[23] + e[12] * e[19] * e[13] + e[12] * e[10] * e[22] + e[21] * e[11] * e[14] + e[21] * e[10] * e[13] + e[15] * e[9] * e[24] + e[15] * e[20] * e[17] + e[15] * e[11] * e[26] + e[15] * e[19] * e[16] + e[15] * e[10] * e[25] + e[24] * e[11] * e[17] + e[24] * e[10] * e[16] - 1.*e[9] * e[23] * e[14] - 1.*e[9] * e[25] * e[16] - 1.*e[9] * e[26] * e[17] + e[9] * e[20] * e[11] + e[9] * e[19] * e[10] - 1.*e[9] * e[22] * e[13];
	A[176] = e[13] * e[21] * e[24] + e[13] * e[23] * e[26] + e[19] * e[18] * e[15] + e[19] * e[9] * e[24] + e[19] * e[20] * e[17] + e[19] * e[11] * e[26] - 1.*e[25] * e[23] * e[14] - 1.*e[25] * e[20] * e[11] - 1.*e[25] * e[18] * e[9] - 1.*e[25] * e[21] * e[12] + e[22] * e[21] * e[15] + e[22] * e[12] * e[24] + e[22] * e[23] * e[17] + e[22] * e[14] * e[26] + e[22] * e[13] * e[25] + e[25] * e[24] * e[15] + e[25] * e[26] * e[17] + e[10] * e[19] * e[25] + e[10] * e[18] * e[24] + e[10] * e[20] * e[26] - .5000000000*e[16] * ep2[18] - .5000000000*e[16] * ep2[23] + .5000000000*e[16] * ep2[19] - .5000000000*e[16] * ep2[20] - .5000000000*e[16] * ep2[21] + .5000000000*ep2[22] * e[16] + 1.500000000*ep2[25] * e[16] + .5000000000*e[16] * ep2[24] + .5000000000*e[16] * ep2[26];
	A[35] = .5000000000*e[0] * ep2[12] + .5000000000*e[0] * ep2[15] + .5000000000*e[0] * ep2[11] + 1.500000000*e[0] * ep2[9] + .5000000000*e[0] * ep2[10] - .5000000000*e[0] * ep2[16] - .5000000000*e[0] * ep2[17] - .5000000000*e[0] * ep2[14] - .5000000000*e[0] * ep2[13] + e[12] * e[9] * e[3] + e[12] * e[10] * e[4] + e[12] * e[1] * e[13] + e[12] * e[11] * e[5] + e[12] * e[2] * e[14] + e[15] * e[9] * e[6] + e[15] * e[10] * e[7] + e[15] * e[1] * e[16] + e[15] * e[11] * e[8] + e[15] * e[2] * e[17] + e[6] * e[11] * e[17] + e[6] * e[10] * e[16] + e[3] * e[11] * e[14] + e[3] * e[10] * e[13] + e[9] * e[10] * e[1] + e[9] * e[11] * e[2] - 1.*e[9] * e[13] * e[4] - 1.*e[9] * e[16] * e[7] - 1.*e[9] * e[14] * e[5] - 1.*e[9] * e[17] * e[8];
	A[177] = e[19] * e[11] * e[35] + e[28] * e[18] * e[15] + e[28] * e[9] * e[24] + e[28] * e[20] * e[17] + e[28] * e[11] * e[26] - 1.*e[25] * e[27] * e[9] - 1.*e[25] * e[30] * e[12] - 1.*e[25] * e[32] * e[14] + e[25] * e[33] * e[15] + e[25] * e[35] * e[17] - 1.*e[25] * e[29] * e[11] - 1.*e[34] * e[23] * e[14] + e[34] * e[24] * e[15] + e[34] * e[26] * e[17] - 1.*e[34] * e[20] * e[11] - 1.*e[34] * e[18] * e[9] - 1.*e[34] * e[21] * e[12] + e[13] * e[30] * e[24] + e[13] * e[21] * e[33] + e[13] * e[31] * e[25] + e[13] * e[22] * e[34] + e[13] * e[32] * e[26] + e[13] * e[23] * e[35] + e[10] * e[27] * e[24] + e[10] * e[18] * e[33] + e[10] * e[28] * e[25] + e[10] * e[19] * e[34] + e[10] * e[29] * e[26] + e[10] * e[20] * e[35] + e[22] * e[30] * e[15] + e[22] * e[12] * e[33] + e[22] * e[32] * e[17] + e[22] * e[14] * e[35] + e[22] * e[31] * e[16] + e[31] * e[21] * e[15] + e[31] * e[12] * e[24] + e[31] * e[23] * e[17] + e[31] * e[14] * e[26] - 1.*e[16] * e[27] * e[18] + e[16] * e[33] * e[24] - 1.*e[16] * e[30] * e[21] - 1.*e[16] * e[29] * e[20] + e[16] * e[35] * e[26] - 1.*e[16] * e[32] * e[23] + e[16] * e[28] * e[19] + 3.*e[16] * e[34] * e[25] + e[19] * e[27] * e[15] + e[19] * e[9] * e[33] + e[19] * e[29] * e[17];
	A[45] = e[4] * e[27] * e[3] + e[4] * e[0] * e[30] + e[4] * e[29] * e[5] + e[4] * e[2] * e[32] + e[31] * e[0] * e[3] + e[31] * e[2] * e[5] + e[7] * e[27] * e[6] + e[7] * e[0] * e[33] + e[7] * e[29] * e[8] + e[7] * e[2] * e[35] + e[34] * e[0] * e[6] + e[34] * e[2] * e[8] + e[1] * e[27] * e[0] + e[1] * e[29] * e[2] + e[1] * e[34] * e[7] - 1.*e[1] * e[32] * e[5] - 1.*e[1] * e[33] * e[6] - 1.*e[1] * e[30] * e[3] - 1.*e[1] * e[35] * e[8] + e[1] * e[31] * e[4] + 1.500000000*e[28] * ep2[1] + .5000000000*e[28] * ep2[4] + .5000000000*e[28] * ep2[0] - .5000000000*e[28] * ep2[6] - .5000000000*e[28] * ep2[5] + .5000000000*e[28] * ep2[7] - .5000000000*e[28] * ep2[3] + .5000000000*e[28] * ep2[2] - .5000000000*e[28] * ep2[8];
	A[191] = -1.*e[35] * e[10] * e[1] - 1.*e[35] * e[13] * e[4] + e[35] * e[16] * e[7] + e[35] * e[15] * e[6] - 1.*e[35] * e[9] * e[0] - 1.*e[35] * e[12] * e[3] + e[32] * e[12] * e[6] + e[32] * e[3] * e[15] + e[32] * e[13] * e[7] + e[32] * e[4] * e[16] - 1.*e[8] * e[27] * e[9] - 1.*e[8] * e[30] * e[12] - 1.*e[8] * e[28] * e[10] - 1.*e[8] * e[31] * e[13] + e[8] * e[29] * e[11] + e[11] * e[27] * e[6] + e[11] * e[0] * e[33] + e[11] * e[28] * e[7] + e[11] * e[1] * e[34] + e[29] * e[9] * e[6] + e[29] * e[0] * e[15] + e[29] * e[10] * e[7] + e[29] * e[1] * e[16] + e[5] * e[30] * e[15] + e[5] * e[12] * e[33] + e[5] * e[32] * e[17] + e[5] * e[14] * e[35] + e[5] * e[31] * e[16] + e[5] * e[13] * e[34] + e[8] * e[33] * e[15] + 3.*e[8] * e[35] * e[17] + e[8] * e[34] * e[16] + e[2] * e[27] * e[15] + e[2] * e[9] * e[33] + e[2] * e[29] * e[17] + e[2] * e[11] * e[35] + e[2] * e[28] * e[16] + e[2] * e[10] * e[34] - 1.*e[17] * e[27] * e[0] + e[17] * e[34] * e[7] + e[17] * e[33] * e[6] - 1.*e[17] * e[30] * e[3] - 1.*e[17] * e[28] * e[1] - 1.*e[17] * e[31] * e[4] + e[14] * e[30] * e[6] + e[14] * e[3] * e[33] + e[14] * e[31] * e[7] + e[14] * e[4] * e[34] + e[14] * e[32] * e[8];
	A[44] = e[19] * e[11] * e[2] + e[4] * e[18] * e[12] + e[4] * e[9] * e[21] + e[4] * e[20] * e[14] + e[4] * e[11] * e[23] + e[4] * e[19] * e[13] + e[4] * e[10] * e[22] + e[7] * e[18] * e[15] + e[7] * e[9] * e[24] + e[7] * e[20] * e[17] + e[7] * e[11] * e[26] + e[7] * e[19] * e[16] + e[7] * e[10] * e[25] + e[1] * e[18] * e[9] + e[1] * e[20] * e[11] - 1.*e[10] * e[21] * e[3] - 1.*e[10] * e[26] * e[8] - 1.*e[10] * e[23] * e[5] - 1.*e[10] * e[24] * e[6] + e[13] * e[18] * e[3] + e[13] * e[0] * e[21] + e[13] * e[1] * e[22] + e[13] * e[20] * e[5] + e[13] * e[2] * e[23] - 1.*e[19] * e[15] * e[6] - 1.*e[19] * e[14] * e[5] - 1.*e[19] * e[12] * e[3] - 1.*e[19] * e[17] * e[8] + e[22] * e[9] * e[3] + e[22] * e[0] * e[12] + e[22] * e[11] * e[5] + e[22] * e[2] * e[14] + e[16] * e[18] * e[6] + e[16] * e[0] * e[24] + e[16] * e[1] * e[25] + e[16] * e[20] * e[8] + e[16] * e[2] * e[26] - 1.*e[1] * e[23] * e[14] - 1.*e[1] * e[24] * e[15] - 1.*e[1] * e[26] * e[17] - 1.*e[1] * e[21] * e[12] + e[25] * e[9] * e[6] + e[25] * e[0] * e[15] + e[25] * e[11] * e[8] + e[25] * e[2] * e[17] + e[10] * e[18] * e[0] + 3.*e[10] * e[19] * e[1] + e[10] * e[20] * e[2] + e[19] * e[9] * e[0];
	A[190] = .5000000000*ep2[23] * e[26] + .5000000000*e[26] * ep2[25] + .5000000000*ep2[20] * e[26] - .5000000000*e[26] * ep2[18] + .5000000000*ep3[26] + .5000000000*e[26] * ep2[24] + e[20] * e[19] * e[25] - .5000000000*e[26] * ep2[19] - .5000000000*e[26] * ep2[21] + e[20] * e[18] * e[24] - .5000000000*e[26] * ep2[22] + e[23] * e[21] * e[24] + e[23] * e[22] * e[25];
	A[47] = e[16] * e[9] * e[33] + e[16] * e[29] * e[17] + e[16] * e[11] * e[35] + e[16] * e[10] * e[34] + e[34] * e[11] * e[17] + e[34] * e[9] * e[15] - 1.*e[10] * e[30] * e[12] - 1.*e[10] * e[32] * e[14] - 1.*e[10] * e[33] * e[15] - 1.*e[10] * e[35] * e[17] + e[10] * e[27] * e[9] + e[10] * e[29] * e[11] + e[13] * e[27] * e[12] + e[13] * e[9] * e[30] + e[13] * e[29] * e[14] + e[13] * e[11] * e[32] + e[13] * e[10] * e[31] + e[31] * e[11] * e[14] + e[31] * e[9] * e[12] + e[16] * e[27] * e[15] + 1.500000000*e[28] * ep2[10] + .5000000000*e[28] * ep2[16] + .5000000000*e[28] * ep2[9] + .5000000000*e[28] * ep2[11] - .5000000000*e[28] * ep2[12] - .5000000000*e[28] * ep2[15] - .5000000000*e[28] * ep2[17] - .5000000000*e[28] * ep2[14] + .5000000000*e[28] * ep2[13];
	A[189] = .5000000000*ep2[20] * e[35] + .5000000000*ep2[23] * e[35] + 1.500000000*e[35] * ep2[26] + .5000000000*e[35] * ep2[25] + .5000000000*e[35] * ep2[24] - .5000000000*e[35] * ep2[18] - .5000000000*e[35] * ep2[19] - .5000000000*e[35] * ep2[22] - .5000000000*e[35] * ep2[21] + e[20] * e[27] * e[24] + e[20] * e[18] * e[33] + e[20] * e[28] * e[25] + e[20] * e[19] * e[34] + e[20] * e[29] * e[26] + e[29] * e[19] * e[25] + e[29] * e[18] * e[24] + e[23] * e[30] * e[24] + e[23] * e[21] * e[33] + e[23] * e[31] * e[25] + e[23] * e[22] * e[34] + e[23] * e[32] * e[26] + e[32] * e[22] * e[25] + e[32] * e[21] * e[24] + e[26] * e[33] * e[24] + e[26] * e[34] * e[25] - 1.*e[26] * e[27] * e[18] - 1.*e[26] * e[30] * e[21] - 1.*e[26] * e[31] * e[22] - 1.*e[26] * e[28] * e[19];
	A[46] = e[4] * e[2] * e[5] + .5000000000*e[1] * ep2[0] - .5000000000*e[1] * ep2[6] + e[7] * e[0] * e[6] + .5000000000*e[1] * ep2[7] + .5000000000*e[1] * ep2[4] - .5000000000*e[1] * ep2[8] + .5000000000*e[1] * ep2[2] - .5000000000*e[1] * ep2[3] + .5000000000*ep3[1] + e[7] * e[2] * e[8] - .5000000000*e[1] * ep2[5] + e[4] * e[0] * e[3];
	A[188] = -.5000000000*e[17] * ep2[13] - .5000000000*e[17] * ep2[9] + .5000000000*e[17] * ep2[16] + .5000000000*e[17] * ep2[15] + .5000000000*ep3[17] - .5000000000*e[17] * ep2[10] + e[14] * e[13] * e[16] + e[14] * e[12] * e[15] + .5000000000*ep2[14] * e[17] + e[11] * e[10] * e[16] - .5000000000*e[17] * ep2[12] + .5000000000*ep2[11] * e[17] + e[11] * e[9] * e[15];
	A[41] = e[4] * e[27] * e[30] + e[4] * e[29] * e[32] + e[4] * e[28] * e[31] + e[31] * e[27] * e[3] + e[31] * e[0] * e[30] + e[31] * e[29] * e[5] + e[31] * e[2] * e[32] + e[7] * e[27] * e[33] + e[7] * e[29] * e[35] + e[7] * e[28] * e[34] + e[28] * e[27] * e[0] + e[28] * e[29] * e[2] + e[34] * e[27] * e[6] + e[34] * e[0] * e[33] + e[34] * e[29] * e[8] + e[34] * e[2] * e[35] - 1.*e[28] * e[32] * e[5] - 1.*e[28] * e[33] * e[6] - 1.*e[28] * e[30] * e[3] - 1.*e[28] * e[35] * e[8] + .5000000000*e[1] * ep2[27] + .5000000000*e[1] * ep2[29] + 1.500000000*e[1] * ep2[28] + .5000000000*e[1] * ep2[31] - .5000000000*e[1] * ep2[32] - .5000000000*e[1] * ep2[33] - .5000000000*e[1] * ep2[30] + .5000000000*e[1] * ep2[34] - .5000000000*e[1] * ep2[35];
	A[187] = .5000000000*ep2[11] * e[35] + .5000000000*e[35] * ep2[16] - .5000000000*e[35] * ep2[9] - .5000000000*e[35] * ep2[12] + .5000000000*e[35] * ep2[15] + 1.500000000*e[35] * ep2[17] - .5000000000*e[35] * ep2[10] + .5000000000*e[35] * ep2[14] - .5000000000*e[35] * ep2[13] + e[11] * e[27] * e[15] + e[11] * e[9] * e[33] + e[11] * e[29] * e[17] + e[11] * e[28] * e[16] + e[11] * e[10] * e[34] + e[29] * e[9] * e[15] + e[29] * e[10] * e[16] + e[14] * e[30] * e[15] + e[14] * e[12] * e[33] + e[14] * e[32] * e[17] + e[14] * e[31] * e[16] + e[14] * e[13] * e[34] + e[32] * e[12] * e[15] + e[32] * e[13] * e[16] + e[17] * e[33] * e[15] + e[17] * e[34] * e[16] - 1.*e[17] * e[27] * e[9] - 1.*e[17] * e[30] * e[12] - 1.*e[17] * e[28] * e[10] - 1.*e[17] * e[31] * e[13];
	A[40] = e[34] * e[27] * e[33] + e[34] * e[29] * e[35] - .5000000000*e[28] * ep2[30] - .5000000000*e[28] * ep2[35] + .5000000000*ep3[28] + .5000000000*e[28] * ep2[27] + .5000000000*e[28] * ep2[29] + e[31] * e[27] * e[30] + e[31] * e[29] * e[32] - .5000000000*e[28] * ep2[32] - .5000000000*e[28] * ep2[33] + .5000000000*e[28] * ep2[31] + .5000000000*e[28] * ep2[34];
	A[186] = .5000000000*ep2[5] * e[8] + e[2] * e[0] * e[6] + .5000000000*ep2[2] * e[8] + .5000000000*ep3[8] - .5000000000*e[8] * ep2[0] + e[5] * e[4] * e[7] + e[5] * e[3] * e[6] + .5000000000*e[8] * ep2[7] + e[2] * e[1] * e[7] - .5000000000*e[8] * ep2[1] - .5000000000*e[8] * ep2[4] - .5000000000*e[8] * ep2[3] + .5000000000*e[8] * ep2[6];
	A[43] = e[28] * e[27] * e[9] + e[28] * e[29] * e[11] - 1.*e[28] * e[30] * e[12] + e[28] * e[31] * e[13] - 1.*e[28] * e[32] * e[14] - 1.*e[28] * e[33] * e[15] - 1.*e[28] * e[35] * e[17] + e[31] * e[27] * e[12] + e[31] * e[9] * e[30] + e[31] * e[29] * e[14] + e[31] * e[11] * e[32] + e[13] * e[27] * e[30] + e[13] * e[29] * e[32] + e[16] * e[27] * e[33] + e[16] * e[29] * e[35] + e[34] * e[27] * e[15] + e[34] * e[9] * e[33] + e[34] * e[29] * e[17] + e[34] * e[11] * e[35] + e[34] * e[28] * e[16] + .5000000000*e[10] * ep2[27] + .5000000000*e[10] * ep2[29] + 1.500000000*e[10] * ep2[28] - .5000000000*e[10] * ep2[32] + .5000000000*e[10] * ep2[31] - .5000000000*e[10] * ep2[33] - .5000000000*e[10] * ep2[30] + .5000000000*e[10] * ep2[34] - .5000000000*e[10] * ep2[35];
	A[185] = -.5000000000*e[35] * ep2[1] + .5000000000*e[35] * ep2[7] - .5000000000*e[35] * ep2[3] + .5000000000*ep2[2] * e[35] + 1.500000000*e[35] * ep2[8] - .5000000000*e[35] * ep2[4] - .5000000000*e[35] * ep2[0] + .5000000000*e[35] * ep2[6] + .5000000000*e[35] * ep2[5] + e[2] * e[27] * e[6] + e[2] * e[0] * e[33] + e[2] * e[28] * e[7] + e[2] * e[1] * e[34] + e[2] * e[29] * e[8] - 1.*e[8] * e[27] * e[0] + e[8] * e[34] * e[7] + e[8] * e[32] * e[5] + e[8] * e[33] * e[6] - 1.*e[8] * e[30] * e[3] - 1.*e[8] * e[28] * e[1] - 1.*e[8] * e[31] * e[4] + e[29] * e[1] * e[7] + e[29] * e[0] * e[6] + e[5] * e[30] * e[6] + e[5] * e[3] * e[33] + e[5] * e[31] * e[7] + e[5] * e[4] * e[34] + e[32] * e[4] * e[7] + e[32] * e[3] * e[6];
	A[42] = e[28] * e[27] * e[18] + e[28] * e[29] * e[20] + e[22] * e[27] * e[30] + e[22] * e[29] * e[32] + e[22] * e[28] * e[31] + e[31] * e[27] * e[21] + e[31] * e[18] * e[30] + e[31] * e[29] * e[23] + e[31] * e[20] * e[32] + e[25] * e[27] * e[33] + e[25] * e[29] * e[35] + e[25] * e[28] * e[34] + e[34] * e[27] * e[24] + e[34] * e[18] * e[33] + e[34] * e[29] * e[26] + e[34] * e[20] * e[35] - 1.*e[28] * e[33] * e[24] - 1.*e[28] * e[30] * e[21] - 1.*e[28] * e[35] * e[26] - 1.*e[28] * e[32] * e[23] - .5000000000*e[19] * ep2[33] - .5000000000*e[19] * ep2[30] - .5000000000*e[19] * ep2[35] + .5000000000*e[19] * ep2[27] + .5000000000*e[19] * ep2[29] + 1.500000000*e[19] * ep2[28] + .5000000000*e[19] * ep2[31] + .5000000000*e[19] * ep2[34] - .5000000000*e[19] * ep2[32];
	A[184] = e[23] * e[3] * e[15] - 1.*e[17] * e[19] * e[1] - 1.*e[17] * e[22] * e[4] - 1.*e[17] * e[18] * e[0] + e[17] * e[25] * e[7] + e[17] * e[24] * e[6] + e[14] * e[21] * e[6] + e[14] * e[3] * e[24] + e[14] * e[22] * e[7] + e[14] * e[4] * e[25] + e[14] * e[23] * e[8] - 1.*e[26] * e[10] * e[1] - 1.*e[26] * e[13] * e[4] + e[26] * e[16] * e[7] + e[26] * e[15] * e[6] - 1.*e[26] * e[9] * e[0] - 1.*e[26] * e[12] * e[3] + e[23] * e[12] * e[6] + e[11] * e[18] * e[6] + e[11] * e[0] * e[24] + e[11] * e[19] * e[7] + e[11] * e[1] * e[25] + e[11] * e[20] * e[8] + e[11] * e[2] * e[26] + e[20] * e[9] * e[6] + e[20] * e[0] * e[15] + e[20] * e[10] * e[7] + e[20] * e[1] * e[16] + e[20] * e[2] * e[17] + e[5] * e[21] * e[15] + e[5] * e[12] * e[24] + e[5] * e[23] * e[17] + e[5] * e[14] * e[26] + e[5] * e[22] * e[16] + e[5] * e[13] * e[25] + e[8] * e[24] * e[15] + 3.*e[8] * e[26] * e[17] + e[8] * e[25] * e[16] + e[2] * e[18] * e[15] + e[2] * e[9] * e[24] + e[2] * e[19] * e[16] + e[2] * e[10] * e[25] - 1.*e[17] * e[21] * e[3] + e[23] * e[4] * e[16] + e[23] * e[13] * e[7] - 1.*e[8] * e[18] * e[9] - 1.*e[8] * e[21] * e[12] - 1.*e[8] * e[19] * e[10] - 1.*e[8] * e[22] * e[13];
	A[54] = e[13] * e[18] * e[12] + e[13] * e[9] * e[21] + e[13] * e[20] * e[14] + e[13] * e[11] * e[23] + e[13] * e[10] * e[22] + e[22] * e[11] * e[14] + e[22] * e[9] * e[12] + e[16] * e[18] * e[15] + e[16] * e[9] * e[24] + e[16] * e[20] * e[17] + e[16] * e[11] * e[26] + e[16] * e[10] * e[25] + e[25] * e[11] * e[17] + e[25] * e[9] * e[15] - 1.*e[10] * e[23] * e[14] - 1.*e[10] * e[24] * e[15] - 1.*e[10] * e[26] * e[17] + e[10] * e[20] * e[11] + e[10] * e[18] * e[9] - 1.*e[10] * e[21] * e[12] + .5000000000*e[19] * ep2[11] + .5000000000*e[19] * ep2[9] + 1.500000000*e[19] * ep2[10] + .5000000000*e[19] * ep2[13] + .5000000000*e[19] * ep2[16] - .5000000000*e[19] * ep2[12] - .5000000000*e[19] * ep2[15] - .5000000000*e[19] * ep2[17] - .5000000000*e[19] * ep2[14];
	A[164] = e[10] * e[18] * e[6] + e[10] * e[0] * e[24] + e[10] * e[19] * e[7] + e[10] * e[1] * e[25] + e[10] * e[20] * e[8] + e[10] * e[2] * e[26] + e[19] * e[9] * e[6] + e[19] * e[0] * e[15] + e[19] * e[1] * e[16] + e[19] * e[11] * e[8] + e[19] * e[2] * e[17] + e[4] * e[21] * e[15] + e[4] * e[12] * e[24] + e[4] * e[23] * e[17] + e[4] * e[14] * e[26] + e[4] * e[22] * e[16] + e[4] * e[13] * e[25] + e[7] * e[24] * e[15] + e[7] * e[26] * e[17] + 3.*e[7] * e[25] * e[16] + e[1] * e[18] * e[15] + e[1] * e[9] * e[24] + e[1] * e[20] * e[17] + e[1] * e[11] * e[26] - 1.*e[16] * e[21] * e[3] + e[16] * e[26] * e[8] - 1.*e[16] * e[20] * e[2] - 1.*e[16] * e[18] * e[0] - 1.*e[16] * e[23] * e[5] + e[16] * e[24] * e[6] + e[13] * e[21] * e[6] + e[13] * e[3] * e[24] + e[13] * e[22] * e[7] + e[13] * e[23] * e[8] + e[13] * e[5] * e[26] - 1.*e[25] * e[11] * e[2] + e[25] * e[15] * e[6] - 1.*e[25] * e[9] * e[0] - 1.*e[25] * e[14] * e[5] - 1.*e[25] * e[12] * e[3] + e[25] * e[17] * e[8] + e[22] * e[12] * e[6] + e[22] * e[3] * e[15] + e[22] * e[14] * e[8] + e[22] * e[5] * e[17] - 1.*e[7] * e[23] * e[14] - 1.*e[7] * e[20] * e[11] - 1.*e[7] * e[18] * e[9] - 1.*e[7] * e[21] * e[12];
	A[55] = e[13] * e[9] * e[3] + e[13] * e[0] * e[12] + e[13] * e[10] * e[4] + e[13] * e[11] * e[5] + e[13] * e[2] * e[14] + e[16] * e[9] * e[6] + e[16] * e[0] * e[15] + e[16] * e[10] * e[7] + e[16] * e[11] * e[8] + e[16] * e[2] * e[17] + e[7] * e[11] * e[17] + e[7] * e[9] * e[15] + e[4] * e[11] * e[14] + e[4] * e[9] * e[12] + e[10] * e[9] * e[0] + e[10] * e[11] * e[2] - 1.*e[10] * e[15] * e[6] - 1.*e[10] * e[14] * e[5] - 1.*e[10] * e[12] * e[3] - 1.*e[10] * e[17] * e[8] + .5000000000*e[1] * ep2[11] + .5000000000*e[1] * ep2[9] + 1.500000000*e[1] * ep2[10] - .5000000000*e[1] * ep2[12] - .5000000000*e[1] * ep2[15] - .5000000000*e[1] * ep2[17] - .5000000000*e[1] * ep2[14] + .5000000000*e[1] * ep2[13] + .5000000000*e[1] * ep2[16];
	A[165] = e[1] * e[27] * e[6] + e[1] * e[0] * e[33] + e[1] * e[28] * e[7] + e[1] * e[29] * e[8] + e[1] * e[2] * e[35] - 1.*e[7] * e[27] * e[0] - 1.*e[7] * e[32] * e[5] + e[7] * e[33] * e[6] - 1.*e[7] * e[30] * e[3] + e[7] * e[35] * e[8] - 1.*e[7] * e[29] * e[2] + e[7] * e[31] * e[4] + e[28] * e[0] * e[6] + e[28] * e[2] * e[8] + e[4] * e[30] * e[6] + e[4] * e[3] * e[33] + e[4] * e[32] * e[8] + e[4] * e[5] * e[35] + e[31] * e[3] * e[6] + e[31] * e[5] * e[8] + .5000000000*ep2[1] * e[34] + 1.500000000*e[34] * ep2[7] + .5000000000*e[34] * ep2[4] - .5000000000*e[34] * ep2[0] + .5000000000*e[34] * ep2[6] - .5000000000*e[34] * ep2[5] - .5000000000*e[34] * ep2[3] - .5000000000*e[34] * ep2[2] + .5000000000*e[34] * ep2[8];
	A[52] = e[4] * e[18] * e[3] + e[4] * e[0] * e[21] + e[4] * e[1] * e[22] + e[4] * e[20] * e[5] + e[4] * e[2] * e[23] + e[22] * e[0] * e[3] + e[22] * e[2] * e[5] + e[7] * e[18] * e[6] + e[7] * e[0] * e[24] + e[7] * e[1] * e[25] + e[7] * e[20] * e[8] + e[7] * e[2] * e[26] + e[25] * e[0] * e[6] + e[25] * e[2] * e[8] + e[1] * e[18] * e[0] + e[1] * e[20] * e[2] - 1.*e[1] * e[21] * e[3] - 1.*e[1] * e[26] * e[8] - 1.*e[1] * e[23] * e[5] - 1.*e[1] * e[24] * e[6] + .5000000000*e[19] * ep2[4] + .5000000000*e[19] * ep2[0] - .5000000000*e[19] * ep2[6] - .5000000000*e[19] * ep2[5] + 1.500000000*e[19] * ep2[1] + .5000000000*e[19] * ep2[7] - .5000000000*e[19] * ep2[3] + .5000000000*e[19] * ep2[2] - .5000000000*e[19] * ep2[8];
	A[166] = -.5000000000*e[7] * ep2[0] + e[4] * e[5] * e[8] + .5000000000*ep2[4] * e[7] - .5000000000*e[7] * ep2[2] + .5000000000*e[7] * ep2[8] - .5000000000*e[7] * ep2[5] + .5000000000*e[7] * ep2[6] + e[1] * e[0] * e[6] + .5000000000*ep3[7] + e[4] * e[3] * e[6] + e[1] * e[2] * e[8] - .5000000000*e[7] * ep2[3] + .5000000000*ep2[1] * e[7];
	A[53] = -1.*e[1] * e[32] * e[23] - 1.*e[19] * e[32] * e[5] - 1.*e[19] * e[33] * e[6] - 1.*e[19] * e[30] * e[3] - 1.*e[19] * e[35] * e[8] - 1.*e[28] * e[21] * e[3] - 1.*e[28] * e[26] * e[8] - 1.*e[28] * e[23] * e[5] - 1.*e[28] * e[24] * e[6] + e[7] * e[27] * e[24] + e[7] * e[18] * e[33] + e[7] * e[29] * e[26] + e[7] * e[20] * e[35] + e[22] * e[27] * e[3] + e[22] * e[0] * e[30] + e[22] * e[29] * e[5] + e[22] * e[2] * e[32] + e[31] * e[18] * e[3] + e[31] * e[0] * e[21] + e[31] * e[20] * e[5] + e[31] * e[2] * e[23] + e[25] * e[27] * e[6] + e[25] * e[0] * e[33] + e[25] * e[28] * e[7] + e[25] * e[1] * e[34] + e[25] * e[29] * e[8] + e[25] * e[2] * e[35] + e[34] * e[18] * e[6] + e[34] * e[0] * e[24] + e[34] * e[19] * e[7] + e[34] * e[20] * e[8] + e[34] * e[2] * e[26] + e[1] * e[27] * e[18] + 3.*e[1] * e[28] * e[19] + e[1] * e[29] * e[20] + e[19] * e[27] * e[0] + e[19] * e[29] * e[2] + e[28] * e[18] * e[0] + e[28] * e[20] * e[2] + e[4] * e[27] * e[21] + e[4] * e[18] * e[30] + e[4] * e[28] * e[22] + e[4] * e[19] * e[31] + e[4] * e[29] * e[23] + e[4] * e[20] * e[32] - 1.*e[1] * e[33] * e[24] - 1.*e[1] * e[30] * e[21] - 1.*e[1] * e[35] * e[26] + e[1] * e[31] * e[22];
	A[167] = e[10] * e[27] * e[15] + e[10] * e[9] * e[33] + e[10] * e[29] * e[17] + e[10] * e[11] * e[35] + e[10] * e[28] * e[16] + e[28] * e[11] * e[17] + e[28] * e[9] * e[15] + e[13] * e[30] * e[15] + e[13] * e[12] * e[33] + e[13] * e[32] * e[17] + e[13] * e[14] * e[35] + e[13] * e[31] * e[16] + e[31] * e[14] * e[17] + e[31] * e[12] * e[15] + e[16] * e[33] * e[15] + e[16] * e[35] * e[17] - 1.*e[16] * e[27] * e[9] - 1.*e[16] * e[30] * e[12] - 1.*e[16] * e[32] * e[14] - 1.*e[16] * e[29] * e[11] + .5000000000*ep2[10] * e[34] + 1.500000000*e[34] * ep2[16] - .5000000000*e[34] * ep2[9] - .5000000000*e[34] * ep2[11] - .5000000000*e[34] * ep2[12] + .5000000000*e[34] * ep2[15] + .5000000000*e[34] * ep2[17] - .5000000000*e[34] * ep2[14] + .5000000000*e[34] * ep2[13];
	A[50] = .5000000000*e[19] * ep2[18] + .5000000000*e[19] * ep2[25] + .5000000000*e[19] * ep2[22] + e[25] * e[20] * e[26] - .5000000000*e[19] * ep2[21] + .5000000000*e[19] * ep2[20] - .5000000000*e[19] * ep2[26] - .5000000000*e[19] * ep2[23] - .5000000000*e[19] * ep2[24] + .5000000000*ep3[19] + e[22] * e[20] * e[23] + e[25] * e[18] * e[24] + e[22] * e[18] * e[21];
	A[160] = .5000000000*e[34] * ep2[33] + .5000000000*e[34] * ep2[35] - .5000000000*e[34] * ep2[27] - .5000000000*e[34] * ep2[32] - .5000000000*e[34] * ep2[29] - .5000000000*e[34] * ep2[30] + .5000000000*ep2[28] * e[34] + e[31] * e[30] * e[33] + e[31] * e[32] * e[35] + e[28] * e[27] * e[33] + .5000000000*ep3[34] + e[28] * e[29] * e[35] + .5000000000*ep2[31] * e[34];
	A[51] = e[4] * e[28] * e[13] + e[4] * e[10] * e[31] + e[7] * e[27] * e[15] + e[7] * e[9] * e[33] + e[7] * e[29] * e[17] + e[7] * e[11] * e[35] + e[7] * e[28] * e[16] + e[7] * e[10] * e[34] + e[1] * e[27] * e[9] + e[1] * e[29] * e[11] + 3.*e[1] * e[28] * e[10] + e[10] * e[27] * e[0] - 1.*e[10] * e[32] * e[5] - 1.*e[10] * e[33] * e[6] - 1.*e[10] * e[30] * e[3] - 1.*e[10] * e[35] * e[8] + e[10] * e[29] * e[2] + e[13] * e[27] * e[3] + e[13] * e[0] * e[30] + e[13] * e[1] * e[31] + e[13] * e[29] * e[5] + e[13] * e[2] * e[32] + e[28] * e[11] * e[2] - 1.*e[28] * e[15] * e[6] + e[28] * e[9] * e[0] - 1.*e[28] * e[14] * e[5] - 1.*e[28] * e[12] * e[3] - 1.*e[28] * e[17] * e[8] + e[31] * e[9] * e[3] + e[31] * e[0] * e[12] + e[31] * e[11] * e[5] + e[31] * e[2] * e[14] + e[16] * e[27] * e[6] + e[16] * e[0] * e[33] + e[16] * e[1] * e[34] + e[16] * e[29] * e[8] + e[16] * e[2] * e[35] - 1.*e[1] * e[30] * e[12] - 1.*e[1] * e[32] * e[14] - 1.*e[1] * e[33] * e[15] - 1.*e[1] * e[35] * e[17] + e[34] * e[9] * e[6] + e[34] * e[0] * e[15] + e[34] * e[11] * e[8] + e[34] * e[2] * e[17] + e[4] * e[27] * e[12] + e[4] * e[9] * e[30] + e[4] * e[29] * e[14] + e[4] * e[11] * e[32];
	A[161] = e[4] * e[30] * e[33] + e[4] * e[32] * e[35] + e[4] * e[31] * e[34] + e[31] * e[30] * e[6] + e[31] * e[3] * e[33] + e[31] * e[32] * e[8] + e[31] * e[5] * e[35] + e[28] * e[27] * e[6] + e[28] * e[0] * e[33] + e[28] * e[29] * e[8] + e[28] * e[2] * e[35] + e[34] * e[33] * e[6] + e[34] * e[35] * e[8] - 1.*e[34] * e[27] * e[0] - 1.*e[34] * e[32] * e[5] - 1.*e[34] * e[30] * e[3] - 1.*e[34] * e[29] * e[2] + e[1] * e[27] * e[33] + e[1] * e[29] * e[35] + e[1] * e[28] * e[34] + .5000000000*ep2[31] * e[7] - .5000000000*e[7] * ep2[27] - .5000000000*e[7] * ep2[32] + .5000000000*e[7] * ep2[28] - .5000000000*e[7] * ep2[29] + .5000000000*e[7] * ep2[33] - .5000000000*e[7] * ep2[30] + 1.500000000*e[7] * ep2[34] + .5000000000*e[7] * ep2[35];
	A[48] = -.5000000000*e[10] * ep2[14] - .5000000000*e[10] * ep2[17] - .5000000000*e[10] * ep2[15] + e[13] * e[11] * e[14] + e[16] * e[11] * e[17] + .5000000000*e[10] * ep2[13] + e[13] * e[9] * e[12] - .5000000000*e[10] * ep2[12] + .5000000000*ep3[10] + e[16] * e[9] * e[15] + .5000000000*e[10] * ep2[16] + .5000000000*e[10] * ep2[11] + .5000000000*e[10] * ep2[9];
	A[162] = e[22] * e[32] * e[35] + e[22] * e[31] * e[34] + e[31] * e[30] * e[24] + e[31] * e[21] * e[33] + e[31] * e[32] * e[26] + e[31] * e[23] * e[35] + e[34] * e[33] * e[24] + e[34] * e[35] * e[26] - 1.*e[34] * e[27] * e[18] - 1.*e[34] * e[30] * e[21] - 1.*e[34] * e[29] * e[20] - 1.*e[34] * e[32] * e[23] + e[19] * e[27] * e[33] + e[19] * e[29] * e[35] + e[19] * e[28] * e[34] + e[28] * e[27] * e[24] + e[28] * e[18] * e[33] + e[28] * e[29] * e[26] + e[28] * e[20] * e[35] + e[22] * e[30] * e[33] + .5000000000*ep2[28] * e[25] + .5000000000*ep2[31] * e[25] + .5000000000*e[25] * ep2[33] + .5000000000*e[25] * ep2[35] + 1.500000000*e[25] * ep2[34] - .5000000000*e[25] * ep2[27] - .5000000000*e[25] * ep2[32] - .5000000000*e[25] * ep2[29] - .5000000000*e[25] * ep2[30];
	A[49] = -1.*e[19] * e[35] * e[26] - 1.*e[19] * e[32] * e[23] + e[19] * e[27] * e[18] + e[19] * e[29] * e[20] + e[22] * e[27] * e[21] + e[22] * e[18] * e[30] + e[22] * e[19] * e[31] + e[22] * e[29] * e[23] + e[22] * e[20] * e[32] + e[31] * e[18] * e[21] + e[31] * e[20] * e[23] + e[25] * e[27] * e[24] + e[25] * e[18] * e[33] + e[25] * e[19] * e[34] + e[25] * e[29] * e[26] + e[25] * e[20] * e[35] + e[34] * e[18] * e[24] + e[34] * e[20] * e[26] - 1.*e[19] * e[33] * e[24] - 1.*e[19] * e[30] * e[21] + 1.500000000*e[28] * ep2[19] + .5000000000*e[28] * ep2[18] + .5000000000*e[28] * ep2[20] + .5000000000*e[28] * ep2[22] + .5000000000*e[28] * ep2[25] - .5000000000*e[28] * ep2[26] - .5000000000*e[28] * ep2[23] - .5000000000*e[28] * ep2[24] - .5000000000*e[28] * ep2[21];
	A[163] = e[10] * e[27] * e[33] + e[10] * e[29] * e[35] + e[10] * e[28] * e[34] + e[34] * e[33] * e[15] + e[34] * e[35] * e[17] + e[28] * e[27] * e[15] + e[28] * e[9] * e[33] + e[28] * e[29] * e[17] + e[28] * e[11] * e[35] - 1.*e[34] * e[27] * e[9] - 1.*e[34] * e[30] * e[12] + e[34] * e[31] * e[13] - 1.*e[34] * e[32] * e[14] - 1.*e[34] * e[29] * e[11] + e[31] * e[30] * e[15] + e[31] * e[12] * e[33] + e[31] * e[32] * e[17] + e[31] * e[14] * e[35] + e[13] * e[30] * e[33] + e[13] * e[32] * e[35] - .5000000000*e[16] * ep2[27] - .5000000000*e[16] * ep2[32] + .5000000000*e[16] * ep2[28] - .5000000000*e[16] * ep2[29] + .5000000000*e[16] * ep2[31] + .5000000000*e[16] * ep2[33] - .5000000000*e[16] * ep2[30] + 1.500000000*e[16] * ep2[34] + .5000000000*e[16] * ep2[35];
	A[63] = e[29] * e[32] * e[14] - 1.*e[29] * e[33] * e[15] - 1.*e[29] * e[34] * e[16] + e[32] * e[27] * e[12] + e[32] * e[9] * e[30] + e[32] * e[28] * e[13] + e[32] * e[10] * e[31] + e[14] * e[27] * e[30] + e[14] * e[28] * e[31] + e[17] * e[27] * e[33] + e[17] * e[28] * e[34] + e[35] * e[27] * e[15] + e[35] * e[9] * e[33] + e[35] * e[29] * e[17] + e[35] * e[28] * e[16] + e[35] * e[10] * e[34] + e[29] * e[27] * e[9] + e[29] * e[28] * e[10] - 1.*e[29] * e[30] * e[12] - 1.*e[29] * e[31] * e[13] + .5000000000*e[11] * ep2[27] + 1.500000000*e[11] * ep2[29] + .5000000000*e[11] * ep2[28] + .5000000000*e[11] * ep2[32] - .5000000000*e[11] * ep2[31] - .5000000000*e[11] * ep2[33] - .5000000000*e[11] * ep2[30] - .5000000000*e[11] * ep2[34] + .5000000000*e[11] * ep2[35];
	A[173] = e[1] * e[20] * e[35] + e[19] * e[27] * e[6] + e[19] * e[0] * e[33] + e[19] * e[28] * e[7] + e[19] * e[29] * e[8] + e[19] * e[2] * e[35] + e[28] * e[18] * e[6] + e[28] * e[0] * e[24] + e[28] * e[20] * e[8] + e[28] * e[2] * e[26] + e[4] * e[30] * e[24] + e[4] * e[21] * e[33] + e[4] * e[31] * e[25] + e[4] * e[22] * e[34] + e[4] * e[32] * e[26] + e[4] * e[23] * e[35] - 1.*e[7] * e[27] * e[18] + e[7] * e[33] * e[24] - 1.*e[7] * e[30] * e[21] - 1.*e[7] * e[29] * e[20] + e[7] * e[35] * e[26] + e[7] * e[31] * e[22] - 1.*e[7] * e[32] * e[23] - 1.*e[25] * e[27] * e[0] - 1.*e[25] * e[32] * e[5] - 1.*e[25] * e[30] * e[3] - 1.*e[25] * e[29] * e[2] - 1.*e[34] * e[21] * e[3] - 1.*e[34] * e[20] * e[2] - 1.*e[34] * e[18] * e[0] - 1.*e[34] * e[23] * e[5] + e[22] * e[30] * e[6] + e[22] * e[3] * e[33] + e[22] * e[32] * e[8] + e[22] * e[5] * e[35] + e[31] * e[21] * e[6] + e[31] * e[3] * e[24] + e[31] * e[23] * e[8] + e[31] * e[5] * e[26] + e[34] * e[26] * e[8] + e[1] * e[27] * e[24] + e[1] * e[18] * e[33] + e[1] * e[28] * e[25] + e[1] * e[19] * e[34] + e[1] * e[29] * e[26] + e[34] * e[24] * e[6] + e[25] * e[33] * e[6] + 3.*e[25] * e[34] * e[7] + e[25] * e[35] * e[8];
	A[62] = .5000000000*e[20] * ep2[27] + 1.500000000*e[20] * ep2[29] + .5000000000*e[20] * ep2[28] + .5000000000*e[20] * ep2[32] + .5000000000*e[20] * ep2[35] - .5000000000*e[20] * ep2[31] - .5000000000*e[20] * ep2[33] - .5000000000*e[20] * ep2[30] - .5000000000*e[20] * ep2[34] + e[29] * e[27] * e[18] + e[29] * e[28] * e[19] + e[23] * e[27] * e[30] + e[23] * e[29] * e[32] + e[23] * e[28] * e[31] + e[32] * e[27] * e[21] + e[32] * e[18] * e[30] + e[32] * e[28] * e[22] + e[32] * e[19] * e[31] + e[26] * e[27] * e[33] + e[26] * e[29] * e[35] + e[26] * e[28] * e[34] + e[35] * e[27] * e[24] + e[35] * e[18] * e[33] + e[35] * e[28] * e[25] + e[35] * e[19] * e[34] - 1.*e[29] * e[33] * e[24] - 1.*e[29] * e[30] * e[21] - 1.*e[29] * e[31] * e[22] - 1.*e[29] * e[34] * e[25];
	A[172] = e[19] * e[1] * e[7] + e[19] * e[0] * e[6] + e[19] * e[2] * e[8] + e[4] * e[21] * e[6] + e[4] * e[3] * e[24] + e[4] * e[22] * e[7] + e[4] * e[23] * e[8] + e[4] * e[5] * e[26] + e[22] * e[3] * e[6] + e[22] * e[5] * e[8] + e[7] * e[24] * e[6] + e[7] * e[26] * e[8] + e[1] * e[18] * e[6] + e[1] * e[0] * e[24] + e[1] * e[20] * e[8] + e[1] * e[2] * e[26] - 1.*e[7] * e[21] * e[3] - 1.*e[7] * e[20] * e[2] - 1.*e[7] * e[18] * e[0] - 1.*e[7] * e[23] * e[5] + .5000000000*e[25] * ep2[4] - .5000000000*e[25] * ep2[0] + .5000000000*e[25] * ep2[6] - .5000000000*e[25] * ep2[5] + .5000000000*e[25] * ep2[1] + 1.500000000*e[25] * ep2[7] - .5000000000*e[25] * ep2[3] - .5000000000*e[25] * ep2[2] + .5000000000*e[25] * ep2[8];
	A[61] = e[5] * e[27] * e[30] + e[5] * e[29] * e[32] + e[5] * e[28] * e[31] + e[32] * e[27] * e[3] + e[32] * e[0] * e[30] + e[32] * e[28] * e[4] + e[32] * e[1] * e[31] + e[8] * e[27] * e[33] + e[8] * e[29] * e[35] + e[8] * e[28] * e[34] + e[29] * e[27] * e[0] + e[29] * e[28] * e[1] + e[35] * e[27] * e[6] + e[35] * e[0] * e[33] + e[35] * e[28] * e[7] + e[35] * e[1] * e[34] - 1.*e[29] * e[34] * e[7] - 1.*e[29] * e[33] * e[6] - 1.*e[29] * e[30] * e[3] - 1.*e[29] * e[31] * e[4] + .5000000000*e[2] * ep2[27] + 1.500000000*e[2] * ep2[29] + .5000000000*e[2] * ep2[28] + .5000000000*e[2] * ep2[32] - .5000000000*e[2] * ep2[31] - .5000000000*e[2] * ep2[33] - .5000000000*e[2] * ep2[30] - .5000000000*e[2] * ep2[34] + .5000000000*e[2] * ep2[35];
	A[175] = e[13] * e[12] * e[6] + e[13] * e[3] * e[15] + e[13] * e[4] * e[16] + e[13] * e[14] * e[8] + e[13] * e[5] * e[17] + e[16] * e[15] * e[6] + e[16] * e[17] * e[8] + e[1] * e[11] * e[17] + e[1] * e[9] * e[15] + e[1] * e[10] * e[16] + e[4] * e[14] * e[17] + e[4] * e[12] * e[15] + e[10] * e[9] * e[6] + e[10] * e[0] * e[15] + e[10] * e[11] * e[8] + e[10] * e[2] * e[17] - 1.*e[16] * e[11] * e[2] - 1.*e[16] * e[9] * e[0] - 1.*e[16] * e[14] * e[5] - 1.*e[16] * e[12] * e[3] + .5000000000*ep2[13] * e[7] + 1.500000000*ep2[16] * e[7] + .5000000000*e[7] * ep2[17] + .5000000000*e[7] * ep2[15] - .5000000000*e[7] * ep2[9] - .5000000000*e[7] * ep2[11] - .5000000000*e[7] * ep2[12] + .5000000000*e[7] * ep2[10] - .5000000000*e[7] * ep2[14];
	A[60] = .5000000000*e[29] * ep2[32] + .5000000000*e[29] * ep2[35] - .5000000000*e[29] * ep2[31] - .5000000000*e[29] * ep2[33] - .5000000000*e[29] * ep2[30] - .5000000000*e[29] * ep2[34] + e[32] * e[27] * e[30] + .5000000000*ep3[29] + .5000000000*e[29] * ep2[28] + e[35] * e[28] * e[34] + .5000000000*e[29] * ep2[27] + e[35] * e[27] * e[33] + e[32] * e[28] * e[31];
	A[174] = -1.*e[16] * e[21] * e[12] + e[10] * e[18] * e[15] + e[10] * e[9] * e[24] + e[10] * e[20] * e[17] + e[10] * e[11] * e[26] + e[19] * e[11] * e[17] + e[19] * e[9] * e[15] + e[19] * e[10] * e[16] + e[13] * e[21] * e[15] + e[13] * e[12] * e[24] + e[13] * e[23] * e[17] + e[13] * e[14] * e[26] + e[13] * e[22] * e[16] + e[22] * e[14] * e[17] + e[22] * e[12] * e[15] + e[16] * e[24] * e[15] + e[16] * e[26] * e[17] - 1.*e[16] * e[23] * e[14] - 1.*e[16] * e[20] * e[11] - 1.*e[16] * e[18] * e[9] + .5000000000*ep2[13] * e[25] + 1.500000000*e[25] * ep2[16] + .5000000000*e[25] * ep2[17] + .5000000000*e[25] * ep2[15] + .5000000000*ep2[10] * e[25] - .5000000000*e[25] * ep2[9] - .5000000000*e[25] * ep2[11] - .5000000000*e[25] * ep2[12] - .5000000000*e[25] * ep2[14];
	A[59] = e[19] * e[20] * e[2] + e[22] * e[18] * e[3] + e[22] * e[0] * e[21] + e[22] * e[19] * e[4] + e[22] * e[20] * e[5] + e[22] * e[2] * e[23] - 1.*e[19] * e[21] * e[3] - 1.*e[19] * e[26] * e[8] + e[19] * e[25] * e[7] - 1.*e[19] * e[23] * e[5] - 1.*e[19] * e[24] * e[6] + e[4] * e[18] * e[21] + e[4] * e[20] * e[23] + e[25] * e[18] * e[6] + e[25] * e[0] * e[24] + e[25] * e[20] * e[8] + e[25] * e[2] * e[26] + e[7] * e[18] * e[24] + e[7] * e[20] * e[26] + e[19] * e[18] * e[0] + 1.500000000*ep2[19] * e[1] + .5000000000*e[1] * ep2[22] + .5000000000*e[1] * ep2[18] + .5000000000*e[1] * ep2[20] + .5000000000*e[1] * ep2[25] - .5000000000*e[1] * ep2[26] - .5000000000*e[1] * ep2[23] - .5000000000*e[1] * ep2[24] - .5000000000*e[1] * ep2[21];
	A[169] = e[19] * e[27] * e[24] + e[19] * e[18] * e[33] + e[19] * e[28] * e[25] + e[19] * e[29] * e[26] + e[19] * e[20] * e[35] + e[28] * e[18] * e[24] + e[28] * e[20] * e[26] + e[22] * e[30] * e[24] + e[22] * e[21] * e[33] + e[22] * e[31] * e[25] + e[22] * e[32] * e[26] + e[22] * e[23] * e[35] + e[31] * e[21] * e[24] + e[31] * e[23] * e[26] + e[25] * e[33] * e[24] + e[25] * e[35] * e[26] - 1.*e[25] * e[27] * e[18] - 1.*e[25] * e[30] * e[21] - 1.*e[25] * e[29] * e[20] - 1.*e[25] * e[32] * e[23] - .5000000000*e[34] * ep2[18] - .5000000000*e[34] * ep2[23] - .5000000000*e[34] * ep2[20] - .5000000000*e[34] * ep2[21] + .5000000000*ep2[19] * e[34] + .5000000000*ep2[22] * e[34] + 1.500000000*e[34] * ep2[25] + .5000000000*e[34] * ep2[24] + .5000000000*e[34] * ep2[26];
	A[58] = e[16] * e[0] * e[6] + e[16] * e[2] * e[8] + e[1] * e[11] * e[2] - 1.*e[1] * e[15] * e[6] + e[1] * e[9] * e[0] - 1.*e[1] * e[14] * e[5] - 1.*e[1] * e[12] * e[3] - 1.*e[1] * e[17] * e[8] + e[4] * e[9] * e[3] + e[4] * e[0] * e[12] + e[4] * e[1] * e[13] + e[4] * e[11] * e[5] + e[4] * e[2] * e[14] + e[13] * e[0] * e[3] + e[13] * e[2] * e[5] + e[7] * e[9] * e[6] + e[7] * e[0] * e[15] + e[7] * e[1] * e[16] + e[7] * e[11] * e[8] + e[7] * e[2] * e[17] - .5000000000*e[10] * ep2[6] - .5000000000*e[10] * ep2[5] - .5000000000*e[10] * ep2[3] - .5000000000*e[10] * ep2[8] + 1.500000000*e[10] * ep2[1] + .5000000000*e[10] * ep2[0] + .5000000000*e[10] * ep2[2] + .5000000000*e[10] * ep2[4] + .5000000000*e[10] * ep2[7];
	A[168] = e[13] * e[14] * e[17] + e[13] * e[12] * e[15] + e[10] * e[9] * e[15] + .5000000000*e[16] * ep2[15] - .5000000000*e[16] * ep2[11] - .5000000000*e[16] * ep2[12] - .5000000000*e[16] * ep2[14] + e[10] * e[11] * e[17] + .5000000000*ep2[10] * e[16] + .5000000000*ep3[16] - .5000000000*e[16] * ep2[9] + .5000000000*e[16] * ep2[17] + .5000000000*ep2[13] * e[16];
	A[57] = e[10] * e[29] * e[20] + e[22] * e[27] * e[12] + e[22] * e[9] * e[30] + e[22] * e[29] * e[14] + e[22] * e[11] * e[32] + e[22] * e[10] * e[31] + e[31] * e[18] * e[12] + e[31] * e[9] * e[21] + e[31] * e[20] * e[14] + e[31] * e[11] * e[23] - 1.*e[10] * e[33] * e[24] - 1.*e[10] * e[30] * e[21] - 1.*e[10] * e[35] * e[26] - 1.*e[10] * e[32] * e[23] + e[10] * e[34] * e[25] + e[19] * e[27] * e[9] + e[19] * e[29] * e[11] + e[28] * e[18] * e[9] + e[28] * e[20] * e[11] + e[16] * e[27] * e[24] + e[16] * e[18] * e[33] + e[16] * e[28] * e[25] + e[16] * e[19] * e[34] + e[16] * e[29] * e[26] + e[16] * e[20] * e[35] - 1.*e[19] * e[30] * e[12] - 1.*e[19] * e[32] * e[14] - 1.*e[19] * e[33] * e[15] - 1.*e[19] * e[35] * e[17] - 1.*e[28] * e[23] * e[14] - 1.*e[28] * e[24] * e[15] - 1.*e[28] * e[26] * e[17] - 1.*e[28] * e[21] * e[12] + e[25] * e[27] * e[15] + e[25] * e[9] * e[33] + e[25] * e[29] * e[17] + e[25] * e[11] * e[35] + e[34] * e[18] * e[15] + e[34] * e[9] * e[24] + e[34] * e[20] * e[17] + e[34] * e[11] * e[26] + e[13] * e[27] * e[21] + e[13] * e[18] * e[30] + e[13] * e[28] * e[22] + e[13] * e[19] * e[31] + e[13] * e[29] * e[23] + e[13] * e[20] * e[32] + e[10] * e[27] * e[18] + 3.*e[10] * e[28] * e[19];
	A[171] = e[4] * e[30] * e[15] + e[4] * e[12] * e[33] + e[4] * e[32] * e[17] + e[4] * e[14] * e[35] + e[4] * e[31] * e[16] + e[4] * e[13] * e[34] + e[7] * e[33] * e[15] + e[7] * e[35] * e[17] + 3.*e[7] * e[34] * e[16] + e[1] * e[27] * e[15] + e[1] * e[9] * e[33] + e[1] * e[29] * e[17] + e[1] * e[11] * e[35] + e[1] * e[28] * e[16] + e[1] * e[10] * e[34] - 1.*e[16] * e[27] * e[0] - 1.*e[16] * e[32] * e[5] + e[16] * e[33] * e[6] - 1.*e[16] * e[30] * e[3] + e[16] * e[35] * e[8] - 1.*e[16] * e[29] * e[2] + e[13] * e[30] * e[6] + e[13] * e[3] * e[33] + e[13] * e[31] * e[7] + e[13] * e[32] * e[8] + e[13] * e[5] * e[35] - 1.*e[34] * e[11] * e[2] + e[34] * e[15] * e[6] - 1.*e[34] * e[9] * e[0] - 1.*e[34] * e[14] * e[5] - 1.*e[34] * e[12] * e[3] + e[34] * e[17] * e[8] + e[31] * e[12] * e[6] + e[31] * e[3] * e[15] + e[31] * e[14] * e[8] + e[31] * e[5] * e[17] - 1.*e[7] * e[27] * e[9] - 1.*e[7] * e[30] * e[12] + e[7] * e[28] * e[10] - 1.*e[7] * e[32] * e[14] + e[10] * e[27] * e[6] + e[10] * e[0] * e[33] + e[10] * e[29] * e[8] + e[10] * e[2] * e[35] + e[28] * e[9] * e[6] + e[28] * e[0] * e[15] + e[28] * e[11] * e[8] + e[28] * e[2] * e[17] - 1.*e[7] * e[29] * e[11];
	A[56] = e[22] * e[18] * e[12] + e[22] * e[9] * e[21] + e[22] * e[20] * e[14] + e[22] * e[11] * e[23] + e[22] * e[19] * e[13] + e[25] * e[18] * e[15] + e[25] * e[9] * e[24] + e[25] * e[20] * e[17] + e[25] * e[11] * e[26] + e[25] * e[19] * e[16] + e[16] * e[18] * e[24] + e[16] * e[20] * e[26] + e[13] * e[18] * e[21] + e[13] * e[20] * e[23] + e[19] * e[18] * e[9] + e[19] * e[20] * e[11] - 1.*e[19] * e[23] * e[14] - 1.*e[19] * e[24] * e[15] - 1.*e[19] * e[26] * e[17] - 1.*e[19] * e[21] * e[12] + .5000000000*e[10] * ep2[22] + .5000000000*e[10] * ep2[25] + 1.500000000*e[10] * ep2[19] + .5000000000*e[10] * ep2[18] + .5000000000*e[10] * ep2[20] - .5000000000*e[10] * ep2[26] - .5000000000*e[10] * ep2[23] - .5000000000*e[10] * ep2[24] - .5000000000*e[10] * ep2[21];
	A[170] = e[19] * e[20] * e[26] - .5000000000*e[25] * ep2[20] + e[22] * e[21] * e[24] + e[19] * e[18] * e[24] + .5000000000*ep2[22] * e[25] - .5000000000*e[25] * ep2[21] - .5000000000*e[25] * ep2[23] + .5000000000*ep2[19] * e[25] - .5000000000*e[25] * ep2[18] + .5000000000*e[25] * ep2[24] + .5000000000*e[25] * ep2[26] + .5000000000*ep3[25] + e[22] * e[23] * e[26];
	A[73] = -1.*e[20] * e[33] * e[6] - 1.*e[20] * e[30] * e[3] - 1.*e[20] * e[31] * e[4] - 1.*e[29] * e[21] * e[3] - 1.*e[29] * e[22] * e[4] - 1.*e[29] * e[25] * e[7] - 1.*e[29] * e[24] * e[6] + e[8] * e[27] * e[24] + e[8] * e[18] * e[33] + e[8] * e[28] * e[25] + e[8] * e[19] * e[34] + e[23] * e[27] * e[3] + e[23] * e[0] * e[30] + e[23] * e[28] * e[4] + e[23] * e[1] * e[31] + e[32] * e[18] * e[3] + e[32] * e[0] * e[21] + e[32] * e[19] * e[4] + e[32] * e[1] * e[22] + e[26] * e[27] * e[6] + e[26] * e[0] * e[33] + e[26] * e[28] * e[7] + e[26] * e[1] * e[34] + e[26] * e[29] * e[8] + e[26] * e[2] * e[35] + e[35] * e[18] * e[6] + e[35] * e[0] * e[24] + e[35] * e[19] * e[7] + e[35] * e[1] * e[25] + e[35] * e[20] * e[8] + e[2] * e[27] * e[18] + e[2] * e[28] * e[19] + 3.*e[2] * e[29] * e[20] + e[20] * e[27] * e[0] + e[20] * e[28] * e[1] + e[29] * e[18] * e[0] + e[29] * e[19] * e[1] + e[5] * e[27] * e[21] + e[5] * e[18] * e[30] + e[5] * e[28] * e[22] + e[5] * e[19] * e[31] + e[5] * e[29] * e[23] + e[5] * e[20] * e[32] - 1.*e[2] * e[33] * e[24] - 1.*e[2] * e[30] * e[21] - 1.*e[2] * e[31] * e[22] + e[2] * e[32] * e[23] - 1.*e[2] * e[34] * e[25] - 1.*e[20] * e[34] * e[7];
	A[72] = e[5] * e[18] * e[3] + e[5] * e[0] * e[21] + e[5] * e[19] * e[4] + e[5] * e[1] * e[22] + e[5] * e[2] * e[23] + e[23] * e[1] * e[4] + e[23] * e[0] * e[3] + e[8] * e[18] * e[6] + e[8] * e[0] * e[24] + e[8] * e[19] * e[7] + e[8] * e[1] * e[25] + e[8] * e[2] * e[26] + e[26] * e[1] * e[7] + e[26] * e[0] * e[6] + e[2] * e[18] * e[0] + e[2] * e[19] * e[1] - 1.*e[2] * e[21] * e[3] - 1.*e[2] * e[22] * e[4] - 1.*e[2] * e[25] * e[7] - 1.*e[2] * e[24] * e[6] - .5000000000*e[20] * ep2[4] + .5000000000*e[20] * ep2[0] - .5000000000*e[20] * ep2[6] + .5000000000*e[20] * ep2[5] + .5000000000*e[20] * ep2[1] - .5000000000*e[20] * ep2[7] - .5000000000*e[20] * ep2[3] + 1.500000000*e[20] * ep2[2] + .5000000000*e[20] * ep2[8];
	A[75] = e[14] * e[9] * e[3] + e[14] * e[0] * e[12] + e[14] * e[10] * e[4] + e[14] * e[1] * e[13] + e[14] * e[11] * e[5] + e[17] * e[9] * e[6] + e[17] * e[0] * e[15] + e[17] * e[10] * e[7] + e[17] * e[1] * e[16] + e[17] * e[11] * e[8] + e[8] * e[9] * e[15] + e[8] * e[10] * e[16] + e[5] * e[9] * e[12] + e[5] * e[10] * e[13] + e[11] * e[9] * e[0] + e[11] * e[10] * e[1] - 1.*e[11] * e[13] * e[4] - 1.*e[11] * e[16] * e[7] - 1.*e[11] * e[15] * e[6] - 1.*e[11] * e[12] * e[3] + .5000000000*e[2] * ep2[14] + .5000000000*e[2] * ep2[17] + 1.500000000*e[2] * ep2[11] + .5000000000*e[2] * ep2[9] + .5000000000*e[2] * ep2[10] - .5000000000*e[2] * ep2[16] - .5000000000*e[2] * ep2[12] - .5000000000*e[2] * ep2[15] - .5000000000*e[2] * ep2[13];
	A[74] = e[14] * e[18] * e[12] + e[14] * e[9] * e[21] + e[14] * e[11] * e[23] + e[14] * e[19] * e[13] + e[14] * e[10] * e[22] + e[23] * e[9] * e[12] + e[23] * e[10] * e[13] + e[17] * e[18] * e[15] + e[17] * e[9] * e[24] + e[17] * e[11] * e[26] + e[17] * e[19] * e[16] + e[17] * e[10] * e[25] + e[26] * e[9] * e[15] + e[26] * e[10] * e[16] - 1.*e[11] * e[24] * e[15] - 1.*e[11] * e[25] * e[16] + e[11] * e[18] * e[9] - 1.*e[11] * e[21] * e[12] + e[11] * e[19] * e[10] - 1.*e[11] * e[22] * e[13] + 1.500000000*e[20] * ep2[11] + .5000000000*e[20] * ep2[9] + .5000000000*e[20] * ep2[10] + .5000000000*e[20] * ep2[14] + .5000000000*e[20] * ep2[17] - .5000000000*e[20] * ep2[16] - .5000000000*e[20] * ep2[12] - .5000000000*e[20] * ep2[15] - .5000000000*e[20] * ep2[13];
	A[77] = e[23] * e[10] * e[31] + e[32] * e[18] * e[12] + e[32] * e[9] * e[21] + e[32] * e[19] * e[13] + e[32] * e[10] * e[22] - 1.*e[11] * e[33] * e[24] - 1.*e[11] * e[30] * e[21] + e[11] * e[35] * e[26] - 1.*e[11] * e[31] * e[22] - 1.*e[11] * e[34] * e[25] + e[20] * e[27] * e[9] + e[20] * e[28] * e[10] + e[29] * e[18] * e[9] + e[29] * e[19] * e[10] + e[17] * e[27] * e[24] + e[17] * e[18] * e[33] + e[17] * e[28] * e[25] + e[17] * e[19] * e[34] + e[17] * e[29] * e[26] + e[17] * e[20] * e[35] - 1.*e[20] * e[30] * e[12] - 1.*e[20] * e[31] * e[13] - 1.*e[20] * e[33] * e[15] - 1.*e[20] * e[34] * e[16] - 1.*e[29] * e[24] * e[15] - 1.*e[29] * e[25] * e[16] - 1.*e[29] * e[21] * e[12] - 1.*e[29] * e[22] * e[13] + e[26] * e[27] * e[15] + e[26] * e[9] * e[33] + e[26] * e[28] * e[16] + e[26] * e[10] * e[34] + e[35] * e[18] * e[15] + e[35] * e[9] * e[24] + e[35] * e[19] * e[16] + e[35] * e[10] * e[25] + e[14] * e[27] * e[21] + e[14] * e[18] * e[30] + e[14] * e[28] * e[22] + e[14] * e[19] * e[31] + e[14] * e[29] * e[23] + e[14] * e[20] * e[32] + e[11] * e[27] * e[18] + e[11] * e[28] * e[19] + 3.*e[11] * e[29] * e[20] + e[23] * e[27] * e[12] + e[23] * e[9] * e[30] + e[23] * e[11] * e[32] + e[23] * e[28] * e[13];
	A[76] = e[23] * e[18] * e[12] + e[23] * e[9] * e[21] + e[23] * e[20] * e[14] + e[23] * e[19] * e[13] + e[23] * e[10] * e[22] + e[26] * e[18] * e[15] + e[26] * e[9] * e[24] + e[26] * e[20] * e[17] + e[26] * e[19] * e[16] + e[26] * e[10] * e[25] + e[17] * e[19] * e[25] + e[17] * e[18] * e[24] + e[14] * e[19] * e[22] + e[14] * e[18] * e[21] + e[20] * e[18] * e[9] + e[20] * e[19] * e[10] - 1.*e[20] * e[24] * e[15] - 1.*e[20] * e[25] * e[16] - 1.*e[20] * e[21] * e[12] - 1.*e[20] * e[22] * e[13] + .5000000000*e[11] * ep2[23] + .5000000000*e[11] * ep2[26] + .5000000000*e[11] * ep2[19] + .5000000000*e[11] * ep2[18] + 1.500000000*e[11] * ep2[20] - .5000000000*e[11] * ep2[22] - .5000000000*e[11] * ep2[24] - .5000000000*e[11] * ep2[21] - .5000000000*e[11] * ep2[25];
	A[79] = -1.*e[20] * e[21] * e[3] + e[20] * e[26] * e[8] - 1.*e[20] * e[22] * e[4] - 1.*e[20] * e[25] * e[7] - 1.*e[20] * e[24] * e[6] + e[5] * e[19] * e[22] + e[5] * e[18] * e[21] + e[26] * e[18] * e[6] + e[26] * e[0] * e[24] + e[26] * e[19] * e[7] + e[26] * e[1] * e[25] + e[8] * e[19] * e[25] + e[8] * e[18] * e[24] + e[20] * e[18] * e[0] + e[20] * e[19] * e[1] + e[23] * e[18] * e[3] + e[23] * e[0] * e[21] + e[23] * e[19] * e[4] + e[23] * e[1] * e[22] + e[23] * e[20] * e[5] + 1.500000000*ep2[20] * e[2] + .5000000000*e[2] * ep2[23] + .5000000000*e[2] * ep2[19] + .5000000000*e[2] * ep2[18] + .5000000000*e[2] * ep2[26] - .5000000000*e[2] * ep2[22] - .5000000000*e[2] * ep2[24] - .5000000000*e[2] * ep2[21] - .5000000000*e[2] * ep2[25];
	A[78] = -1.*e[2] * e[15] * e[6] + e[2] * e[9] * e[0] - 1.*e[2] * e[12] * e[3] + e[5] * e[9] * e[3] + e[5] * e[0] * e[12] + e[5] * e[10] * e[4] + e[5] * e[1] * e[13] + e[5] * e[2] * e[14] + e[14] * e[1] * e[4] + e[14] * e[0] * e[3] + e[8] * e[9] * e[6] + e[8] * e[0] * e[15] + e[8] * e[10] * e[7] + e[8] * e[1] * e[16] + e[8] * e[2] * e[17] + e[17] * e[1] * e[7] + e[17] * e[0] * e[6] + e[2] * e[10] * e[1] - 1.*e[2] * e[13] * e[4] - 1.*e[2] * e[16] * e[7] + .5000000000*e[11] * ep2[1] + .5000000000*e[11] * ep2[0] + 1.500000000*e[11] * ep2[2] + .5000000000*e[11] * ep2[5] + .5000000000*e[11] * ep2[8] - .5000000000*e[11] * ep2[4] - .5000000000*e[11] * ep2[6] - .5000000000*e[11] * ep2[7] - .5000000000*e[11] * ep2[3];
	A[64] = e[5] * e[19] * e[13] + e[5] * e[10] * e[22] + e[8] * e[18] * e[15] + e[8] * e[9] * e[24] + e[8] * e[20] * e[17] + e[8] * e[11] * e[26] + e[8] * e[19] * e[16] + e[8] * e[10] * e[25] + e[2] * e[18] * e[9] + e[2] * e[19] * e[10] - 1.*e[11] * e[21] * e[3] - 1.*e[11] * e[22] * e[4] - 1.*e[11] * e[25] * e[7] - 1.*e[11] * e[24] * e[6] + e[14] * e[18] * e[3] + e[14] * e[0] * e[21] + e[14] * e[19] * e[4] + e[14] * e[1] * e[22] + e[14] * e[2] * e[23] - 1.*e[20] * e[13] * e[4] - 1.*e[20] * e[16] * e[7] - 1.*e[20] * e[15] * e[6] - 1.*e[20] * e[12] * e[3] + e[23] * e[9] * e[3] + e[23] * e[0] * e[12] + e[23] * e[10] * e[4] + e[23] * e[1] * e[13] + e[17] * e[18] * e[6] + e[17] * e[0] * e[24] + e[17] * e[19] * e[7] + e[17] * e[1] * e[25] + e[17] * e[2] * e[26] - 1.*e[2] * e[24] * e[15] - 1.*e[2] * e[25] * e[16] - 1.*e[2] * e[21] * e[12] - 1.*e[2] * e[22] * e[13] + e[26] * e[9] * e[6] + e[26] * e[0] * e[15] + e[26] * e[10] * e[7] + e[26] * e[1] * e[16] + e[11] * e[18] * e[0] + e[11] * e[19] * e[1] + 3.*e[11] * e[20] * e[2] + e[20] * e[9] * e[0] + e[20] * e[10] * e[1] + e[5] * e[18] * e[12] + e[5] * e[9] * e[21] + e[5] * e[20] * e[14] + e[5] * e[11] * e[23];
	A[65] = e[32] * e[1] * e[4] + e[32] * e[0] * e[3] + e[8] * e[27] * e[6] + e[8] * e[0] * e[33] + e[8] * e[28] * e[7] + e[8] * e[1] * e[34] + e[35] * e[1] * e[7] + e[35] * e[0] * e[6] + e[2] * e[27] * e[0] + e[2] * e[28] * e[1] - 1.*e[2] * e[34] * e[7] + e[2] * e[32] * e[5] - 1.*e[2] * e[33] * e[6] - 1.*e[2] * e[30] * e[3] + e[2] * e[35] * e[8] - 1.*e[2] * e[31] * e[4] + e[5] * e[27] * e[3] + e[5] * e[0] * e[30] + e[5] * e[28] * e[4] + e[5] * e[1] * e[31] + 1.500000000*e[29] * ep2[2] - .5000000000*e[29] * ep2[4] + .5000000000*e[29] * ep2[0] - .5000000000*e[29] * ep2[6] + .5000000000*e[29] * ep2[5] + .5000000000*e[29] * ep2[1] - .5000000000*e[29] * ep2[7] - .5000000000*e[29] * ep2[3] + .5000000000*e[29] * ep2[8];
	A[66] = e[5] * e[0] * e[3] + e[8] * e[1] * e[7] + e[8] * e[0] * e[6] + e[5] * e[1] * e[4] - .5000000000*e[2] * ep2[4] + .5000000000*ep3[2] + .5000000000*e[2] * ep2[1] - .5000000000*e[2] * ep2[3] + .5000000000*e[2] * ep2[0] + .5000000000*e[2] * ep2[8] + .5000000000*e[2] * ep2[5] - .5000000000*e[2] * ep2[6] - .5000000000*e[2] * ep2[7];
	A[67] = e[35] * e[9] * e[15] + e[35] * e[10] * e[16] - 1.*e[11] * e[30] * e[12] - 1.*e[11] * e[31] * e[13] - 1.*e[11] * e[33] * e[15] - 1.*e[11] * e[34] * e[16] + e[11] * e[27] * e[9] + e[11] * e[28] * e[10] + e[14] * e[27] * e[12] + e[14] * e[9] * e[30] + e[14] * e[11] * e[32] + e[14] * e[28] * e[13] + e[14] * e[10] * e[31] + e[32] * e[9] * e[12] + e[32] * e[10] * e[13] + e[17] * e[27] * e[15] + e[17] * e[9] * e[33] + e[17] * e[11] * e[35] + e[17] * e[28] * e[16] + e[17] * e[10] * e[34] + 1.500000000*e[29] * ep2[11] - .5000000000*e[29] * ep2[16] + .5000000000*e[29] * ep2[9] - .5000000000*e[29] * ep2[12] - .5000000000*e[29] * ep2[15] + .5000000000*e[29] * ep2[17] + .5000000000*e[29] * ep2[10] + .5000000000*e[29] * ep2[14] - .5000000000*e[29] * ep2[13];
	A[68] = e[14] * e[9] * e[12] + e[17] * e[10] * e[16] + e[17] * e[9] * e[15] + .5000000000*ep3[11] + e[14] * e[10] * e[13] + .5000000000*e[11] * ep2[10] - .5000000000*e[11] * ep2[15] + .5000000000*e[11] * ep2[14] - .5000000000*e[11] * ep2[13] - .5000000000*e[11] * ep2[12] + .5000000000*e[11] * ep2[9] - .5000000000*e[11] * ep2[16] + .5000000000*e[11] * ep2[17];
	A[69] = e[20] * e[27] * e[18] + e[20] * e[28] * e[19] + e[23] * e[27] * e[21] + e[23] * e[18] * e[30] + e[23] * e[28] * e[22] + e[23] * e[19] * e[31] + e[23] * e[20] * e[32] + e[32] * e[19] * e[22] + e[32] * e[18] * e[21] + e[26] * e[27] * e[24] + e[26] * e[18] * e[33] + e[26] * e[28] * e[25] + e[26] * e[19] * e[34] + e[26] * e[20] * e[35] + e[35] * e[19] * e[25] + e[35] * e[18] * e[24] - 1.*e[20] * e[33] * e[24] - 1.*e[20] * e[30] * e[21] - 1.*e[20] * e[31] * e[22] - 1.*e[20] * e[34] * e[25] + .5000000000*e[29] * ep2[23] + .5000000000*e[29] * ep2[26] - .5000000000*e[29] * ep2[22] - .5000000000*e[29] * ep2[24] - .5000000000*e[29] * ep2[21] - .5000000000*e[29] * ep2[25] + 1.500000000*e[29] * ep2[20] + .5000000000*e[29] * ep2[19] + .5000000000*e[29] * ep2[18];
	A[70] = .5000000000*e[20] * ep2[26] + .5000000000*e[20] * ep2[18] + .5000000000*ep3[20] + .5000000000*e[20] * ep2[19] + e[26] * e[18] * e[24] + .5000000000*e[20] * ep2[23] - .5000000000*e[20] * ep2[25] + e[23] * e[19] * e[22] - .5000000000*e[20] * ep2[24] - .5000000000*e[20] * ep2[21] - .5000000000*e[20] * ep2[22] + e[23] * e[18] * e[21] + e[26] * e[19] * e[25];
	A[71] = e[8] * e[28] * e[16] + e[8] * e[10] * e[34] + e[2] * e[27] * e[9] + 3.*e[2] * e[29] * e[11] + e[2] * e[28] * e[10] + e[11] * e[27] * e[0] - 1.*e[11] * e[34] * e[7] - 1.*e[11] * e[33] * e[6] - 1.*e[11] * e[30] * e[3] + e[11] * e[28] * e[1] - 1.*e[11] * e[31] * e[4] + e[14] * e[27] * e[3] + e[14] * e[0] * e[30] + e[14] * e[28] * e[4] + e[14] * e[1] * e[31] + e[14] * e[2] * e[32] + e[29] * e[10] * e[1] - 1.*e[29] * e[13] * e[4] - 1.*e[29] * e[16] * e[7] - 1.*e[29] * e[15] * e[6] + e[29] * e[9] * e[0] - 1.*e[29] * e[12] * e[3] + e[32] * e[9] * e[3] + e[32] * e[0] * e[12] + e[32] * e[10] * e[4] + e[32] * e[1] * e[13] + e[17] * e[27] * e[6] + e[17] * e[0] * e[33] + e[17] * e[28] * e[7] + e[17] * e[1] * e[34] + e[17] * e[2] * e[35] - 1.*e[2] * e[30] * e[12] - 1.*e[2] * e[31] * e[13] - 1.*e[2] * e[33] * e[15] - 1.*e[2] * e[34] * e[16] + e[35] * e[9] * e[6] + e[35] * e[0] * e[15] + e[35] * e[10] * e[7] + e[35] * e[1] * e[16] + e[5] * e[27] * e[12] + e[5] * e[9] * e[30] + e[5] * e[29] * e[14] + e[5] * e[11] * e[32] + e[5] * e[28] * e[13] + e[5] * e[10] * e[31] + e[8] * e[27] * e[15] + e[8] * e[9] * e[33] + e[8] * e[29] * e[17] + e[8] * e[11] * e[35];
	A[91] = -1.*e[12] * e[34] * e[7] + e[12] * e[32] * e[5] - 1.*e[12] * e[35] * e[8] - 1.*e[12] * e[29] * e[2] - 1.*e[12] * e[28] * e[1] + e[12] * e[31] * e[4] - 1.*e[30] * e[11] * e[2] - 1.*e[30] * e[10] * e[1] + e[30] * e[13] * e[4] - 1.*e[30] * e[16] * e[7] + e[30] * e[14] * e[5] - 1.*e[30] * e[17] * e[8] + e[15] * e[3] * e[33] + e[15] * e[31] * e[7] + e[15] * e[4] * e[34] + e[15] * e[32] * e[8] + e[15] * e[5] * e[35] + e[3] * e[27] * e[9] - 1.*e[3] * e[28] * e[10] - 1.*e[3] * e[34] * e[16] - 1.*e[3] * e[35] * e[17] - 1.*e[3] * e[29] * e[11] + e[33] * e[13] * e[7] + e[33] * e[4] * e[16] + e[33] * e[14] * e[8] + e[33] * e[5] * e[17] + e[9] * e[28] * e[4] + e[9] * e[1] * e[31] + e[9] * e[29] * e[5] + e[9] * e[2] * e[32] + e[27] * e[10] * e[4] + e[27] * e[1] * e[13] + e[27] * e[11] * e[5] + e[27] * e[2] * e[14] + 3.*e[3] * e[30] * e[12] + e[3] * e[32] * e[14] + e[3] * e[31] * e[13] + e[6] * e[30] * e[15] + e[6] * e[12] * e[33] + e[6] * e[32] * e[17] + e[6] * e[14] * e[35] + e[6] * e[31] * e[16] + e[6] * e[13] * e[34] + e[0] * e[27] * e[12] + e[0] * e[9] * e[30] + e[0] * e[29] * e[14] + e[0] * e[11] * e[32] + e[0] * e[28] * e[13] + e[0] * e[10] * e[31];
	A[90] = .5000000000*e[21] * ep2[24] - .5000000000*e[21] * ep2[25] + .5000000000*e[21] * ep2[23] - .5000000000*e[21] * ep2[26] + .5000000000*ep2[18] * e[21] + .5000000000*e[21] * ep2[22] - .5000000000*e[21] * ep2[20] + e[24] * e[22] * e[25] + e[24] * e[23] * e[26] - .5000000000*e[21] * ep2[19] + e[18] * e[19] * e[22] + e[18] * e[20] * e[23] + .5000000000*ep3[21];
	A[89] = -.5000000000*e[30] * ep2[26] - .5000000000*e[30] * ep2[19] - .5000000000*e[30] * ep2[20] - .5000000000*e[30] * ep2[25] + .5000000000*ep2[18] * e[30] + 1.500000000*e[30] * ep2[21] + .5000000000*e[30] * ep2[22] + .5000000000*e[30] * ep2[23] + .5000000000*e[30] * ep2[24] + e[18] * e[27] * e[21] + e[18] * e[28] * e[22] + e[18] * e[19] * e[31] + e[18] * e[29] * e[23] + e[18] * e[20] * e[32] + e[27] * e[19] * e[22] + e[27] * e[20] * e[23] + e[21] * e[31] * e[22] + e[21] * e[32] * e[23] + e[24] * e[21] * e[33] + e[24] * e[31] * e[25] + e[24] * e[22] * e[34] + e[24] * e[32] * e[26] + e[24] * e[23] * e[35] + e[33] * e[22] * e[25] + e[33] * e[23] * e[26] - 1.*e[21] * e[29] * e[20] - 1.*e[21] * e[35] * e[26] - 1.*e[21] * e[28] * e[19] - 1.*e[21] * e[34] * e[25];
	A[88] = .5000000000*e[12] * ep2[15] - .5000000000*e[12] * ep2[17] + e[15] * e[13] * e[16] - .5000000000*e[12] * ep2[10] + e[15] * e[14] * e[17] - .5000000000*e[12] * ep2[16] - .5000000000*e[12] * ep2[11] + e[9] * e[10] * e[13] + .5000000000*e[12] * ep2[13] + .5000000000*ep2[9] * e[12] + .5000000000*ep3[12] + e[9] * e[11] * e[14] + .5000000000*e[12] * ep2[14];
	A[95] = e[12] * e[13] * e[4] + e[12] * e[14] * e[5] + e[15] * e[12] * e[6] + e[15] * e[13] * e[7] + e[15] * e[4] * e[16] + e[15] * e[14] * e[8] + e[15] * e[5] * e[17] + e[6] * e[14] * e[17] + e[6] * e[13] * e[16] + e[0] * e[11] * e[14] + e[0] * e[9] * e[12] + e[0] * e[10] * e[13] + e[9] * e[10] * e[4] + e[9] * e[1] * e[13] + e[9] * e[11] * e[5] + e[9] * e[2] * e[14] - 1.*e[12] * e[11] * e[2] - 1.*e[12] * e[10] * e[1] - 1.*e[12] * e[16] * e[7] - 1.*e[12] * e[17] * e[8] + 1.500000000*ep2[12] * e[3] + .5000000000*e[3] * ep2[15] - .5000000000*e[3] * ep2[16] + .5000000000*e[3] * ep2[9] - .5000000000*e[3] * ep2[11] - .5000000000*e[3] * ep2[17] - .5000000000*e[3] * ep2[10] + .5000000000*e[3] * ep2[14] + .5000000000*e[3] * ep2[13];
	A[94] = e[18] * e[11] * e[14] + e[18] * e[9] * e[12] + e[18] * e[10] * e[13] + e[12] * e[23] * e[14] + e[12] * e[22] * e[13] + e[15] * e[12] * e[24] + e[15] * e[23] * e[17] + e[15] * e[14] * e[26] + e[15] * e[22] * e[16] + e[15] * e[13] * e[25] + e[24] * e[14] * e[17] + e[24] * e[13] * e[16] - 1.*e[12] * e[25] * e[16] - 1.*e[12] * e[26] * e[17] - 1.*e[12] * e[20] * e[11] - 1.*e[12] * e[19] * e[10] + e[9] * e[20] * e[14] + e[9] * e[11] * e[23] + e[9] * e[19] * e[13] + e[9] * e[10] * e[22] + .5000000000*ep2[9] * e[21] - .5000000000*e[21] * ep2[16] - .5000000000*e[21] * ep2[11] - .5000000000*e[21] * ep2[17] - .5000000000*e[21] * ep2[10] + 1.500000000*e[21] * ep2[12] + .5000000000*e[21] * ep2[14] + .5000000000*e[21] * ep2[13] + .5000000000*e[21] * ep2[15];
	A[93] = -1.*e[21] * e[35] * e[8] - 1.*e[21] * e[29] * e[2] - 1.*e[21] * e[28] * e[1] + e[21] * e[31] * e[4] - 1.*e[30] * e[26] * e[8] - 1.*e[30] * e[20] * e[2] - 1.*e[30] * e[19] * e[1] + e[30] * e[22] * e[4] - 1.*e[30] * e[25] * e[7] + e[30] * e[23] * e[5] + e[6] * e[31] * e[25] + e[6] * e[22] * e[34] + e[6] * e[32] * e[26] + e[6] * e[23] * e[35] + e[24] * e[30] * e[6] + e[24] * e[3] * e[33] + e[24] * e[31] * e[7] + e[24] * e[4] * e[34] + e[24] * e[32] * e[8] + e[24] * e[5] * e[35] + e[33] * e[21] * e[6] + e[33] * e[22] * e[7] + e[33] * e[4] * e[25] + e[33] * e[23] * e[8] + e[33] * e[5] * e[26] + e[0] * e[27] * e[21] + e[0] * e[18] * e[30] + e[0] * e[28] * e[22] + e[0] * e[19] * e[31] + e[0] * e[29] * e[23] + e[0] * e[20] * e[32] + e[18] * e[27] * e[3] + e[18] * e[28] * e[4] + e[18] * e[1] * e[31] + e[18] * e[29] * e[5] + e[18] * e[2] * e[32] + e[27] * e[19] * e[4] + e[27] * e[1] * e[22] + e[27] * e[20] * e[5] + e[27] * e[2] * e[23] + 3.*e[3] * e[30] * e[21] + e[3] * e[31] * e[22] + e[3] * e[32] * e[23] - 1.*e[3] * e[29] * e[20] - 1.*e[3] * e[35] * e[26] - 1.*e[3] * e[28] * e[19] - 1.*e[3] * e[34] * e[25] - 1.*e[21] * e[34] * e[7] + e[21] * e[32] * e[5];
	A[92] = e[18] * e[1] * e[4] + e[18] * e[0] * e[3] + e[18] * e[2] * e[5] + e[3] * e[22] * e[4] + e[3] * e[23] * e[5] + e[6] * e[3] * e[24] + e[6] * e[22] * e[7] + e[6] * e[4] * e[25] + e[6] * e[23] * e[8] + e[6] * e[5] * e[26] + e[24] * e[4] * e[7] + e[24] * e[5] * e[8] + e[0] * e[19] * e[4] + e[0] * e[1] * e[22] + e[0] * e[20] * e[5] + e[0] * e[2] * e[23] - 1.*e[3] * e[26] * e[8] - 1.*e[3] * e[20] * e[2] - 1.*e[3] * e[19] * e[1] - 1.*e[3] * e[25] * e[7] + .5000000000*e[21] * ep2[4] + .5000000000*e[21] * ep2[0] + .5000000000*e[21] * ep2[6] + .5000000000*e[21] * ep2[5] - .5000000000*e[21] * ep2[1] - .5000000000*e[21] * ep2[7] + 1.500000000*e[21] * ep2[3] - .5000000000*e[21] * ep2[2] - .5000000000*e[21] * ep2[8];
	A[82] = .5000000000*ep2[27] * e[21] + 1.500000000*e[21] * ep2[30] + .5000000000*e[21] * ep2[32] + .5000000000*e[21] * ep2[31] + .5000000000*e[21] * ep2[33] - .5000000000*e[21] * ep2[28] - .5000000000*e[21] * ep2[29] - .5000000000*e[21] * ep2[34] - .5000000000*e[21] * ep2[35] + e[18] * e[27] * e[30] + e[18] * e[29] * e[32] + e[18] * e[28] * e[31] + e[27] * e[28] * e[22] + e[27] * e[19] * e[31] + e[27] * e[29] * e[23] + e[27] * e[20] * e[32] + e[30] * e[31] * e[22] + e[30] * e[32] * e[23] + e[24] * e[30] * e[33] + e[24] * e[32] * e[35] + e[24] * e[31] * e[34] + e[33] * e[31] * e[25] + e[33] * e[22] * e[34] + e[33] * e[32] * e[26] + e[33] * e[23] * e[35] - 1.*e[30] * e[29] * e[20] - 1.*e[30] * e[35] * e[26] - 1.*e[30] * e[28] * e[19] - 1.*e[30] * e[34] * e[25];
	A[192] = -.5000000000*e[26] * ep2[4] - .5000000000*e[26] * ep2[0] + .5000000000*e[26] * ep2[6] + .5000000000*e[26] * ep2[5] - .5000000000*e[26] * ep2[1] + .5000000000*e[26] * ep2[7] - .5000000000*e[26] * ep2[3] + .5000000000*e[26] * ep2[2] + 1.500000000*e[26] * ep2[8] + e[20] * e[0] * e[6] + e[20] * e[2] * e[8] + e[5] * e[21] * e[6] + e[5] * e[3] * e[24] + e[5] * e[22] * e[7] + e[5] * e[4] * e[25] + e[5] * e[23] * e[8] + e[23] * e[4] * e[7] + e[23] * e[3] * e[6] + e[8] * e[24] * e[6] + e[8] * e[25] * e[7] + e[2] * e[18] * e[6] + e[2] * e[0] * e[24] + e[2] * e[19] * e[7] + e[2] * e[1] * e[25] - 1.*e[8] * e[21] * e[3] - 1.*e[8] * e[19] * e[1] - 1.*e[8] * e[22] * e[4] - 1.*e[8] * e[18] * e[0] + e[20] * e[1] * e[7];
	A[83] = e[9] * e[27] * e[30] + e[9] * e[29] * e[32] + e[9] * e[28] * e[31] + e[33] * e[30] * e[15] + e[33] * e[32] * e[17] + e[33] * e[14] * e[35] + e[33] * e[31] * e[16] + e[33] * e[13] * e[34] + e[27] * e[29] * e[14] + e[27] * e[11] * e[32] + e[27] * e[28] * e[13] + e[27] * e[10] * e[31] - 1.*e[30] * e[28] * e[10] + e[30] * e[31] * e[13] + e[30] * e[32] * e[14] - 1.*e[30] * e[34] * e[16] - 1.*e[30] * e[35] * e[17] - 1.*e[30] * e[29] * e[11] + e[15] * e[32] * e[35] + e[15] * e[31] * e[34] - .5000000000*e[12] * ep2[34] - .5000000000*e[12] * ep2[35] + .5000000000*e[12] * ep2[27] + .5000000000*e[12] * ep2[32] - .5000000000*e[12] * ep2[28] - .5000000000*e[12] * ep2[29] + .5000000000*e[12] * ep2[31] + .5000000000*e[12] * ep2[33] + 1.500000000*e[12] * ep2[30];
	A[193] = e[23] * e[30] * e[6] + e[23] * e[3] * e[33] + e[23] * e[31] * e[7] + e[23] * e[4] * e[34] + e[32] * e[21] * e[6] + e[32] * e[3] * e[24] + e[32] * e[22] * e[7] + e[32] * e[4] * e[25] + e[26] * e[33] * e[6] + e[26] * e[34] * e[7] + 3.*e[26] * e[35] * e[8] + e[35] * e[24] * e[6] + e[35] * e[25] * e[7] + e[2] * e[27] * e[24] + e[2] * e[18] * e[33] + e[2] * e[28] * e[25] + e[2] * e[19] * e[34] + e[2] * e[29] * e[26] + e[2] * e[20] * e[35] + e[20] * e[27] * e[6] + e[20] * e[0] * e[33] + e[20] * e[28] * e[7] + e[20] * e[1] * e[34] + e[20] * e[29] * e[8] + e[29] * e[18] * e[6] + e[29] * e[0] * e[24] + e[29] * e[19] * e[7] + e[29] * e[1] * e[25] + e[5] * e[30] * e[24] + e[5] * e[21] * e[33] + e[5] * e[31] * e[25] + e[5] * e[22] * e[34] + e[5] * e[32] * e[26] + e[5] * e[23] * e[35] - 1.*e[8] * e[27] * e[18] + e[8] * e[33] * e[24] - 1.*e[8] * e[30] * e[21] - 1.*e[8] * e[31] * e[22] + e[8] * e[32] * e[23] - 1.*e[8] * e[28] * e[19] + e[8] * e[34] * e[25] - 1.*e[26] * e[27] * e[0] - 1.*e[26] * e[30] * e[3] - 1.*e[26] * e[28] * e[1] - 1.*e[26] * e[31] * e[4] - 1.*e[35] * e[21] * e[3] - 1.*e[35] * e[19] * e[1] - 1.*e[35] * e[22] * e[4] - 1.*e[35] * e[18] * e[0];
	A[80] = e[27] * e[29] * e[32] + e[27] * e[28] * e[31] + e[33] * e[32] * e[35] + e[33] * e[31] * e[34] + .5000000000*ep3[30] - .5000000000*e[30] * ep2[28] - .5000000000*e[30] * ep2[29] - .5000000000*e[30] * ep2[34] + .5000000000*e[30] * ep2[33] + .5000000000*ep2[27] * e[30] + .5000000000*e[30] * ep2[32] + .5000000000*e[30] * ep2[31] - .5000000000*e[30] * ep2[35];
	A[194] = .5000000000*ep2[14] * e[26] + 1.500000000*e[26] * ep2[17] + .5000000000*e[26] * ep2[15] + .5000000000*e[26] * ep2[16] + .5000000000*ep2[11] * e[26] - .5000000000*e[26] * ep2[9] - .5000000000*e[26] * ep2[12] - .5000000000*e[26] * ep2[10] - .5000000000*e[26] * ep2[13] + e[20] * e[11] * e[17] + e[20] * e[9] * e[15] + e[20] * e[10] * e[16] + e[14] * e[21] * e[15] + e[14] * e[12] * e[24] + e[14] * e[23] * e[17] + e[14] * e[22] * e[16] + e[14] * e[13] * e[25] + e[23] * e[12] * e[15] + e[23] * e[13] * e[16] + e[17] * e[24] * e[15] + e[17] * e[25] * e[16] - 1.*e[17] * e[18] * e[9] - 1.*e[17] * e[21] * e[12] - 1.*e[17] * e[19] * e[10] - 1.*e[17] * e[22] * e[13] + e[11] * e[18] * e[15] + e[11] * e[9] * e[24] + e[11] * e[19] * e[16] + e[11] * e[10] * e[25];
	A[81] = e[0] * e[27] * e[30] + e[0] * e[29] * e[32] + e[0] * e[28] * e[31] + e[30] * e[31] * e[4] + e[30] * e[32] * e[5] + e[6] * e[30] * e[33] + e[6] * e[32] * e[35] + e[6] * e[31] * e[34] + e[27] * e[28] * e[4] + e[27] * e[1] * e[31] + e[27] * e[29] * e[5] + e[27] * e[2] * e[32] + e[33] * e[31] * e[7] + e[33] * e[4] * e[34] + e[33] * e[32] * e[8] + e[33] * e[5] * e[35] - 1.*e[30] * e[34] * e[7] - 1.*e[30] * e[35] * e[8] - 1.*e[30] * e[29] * e[2] - 1.*e[30] * e[28] * e[1] + 1.500000000*e[3] * ep2[30] + .5000000000*e[3] * ep2[32] + .5000000000*e[3] * ep2[31] + .5000000000*e[3] * ep2[27] - .5000000000*e[3] * ep2[28] - .5000000000*e[3] * ep2[29] + .5000000000*e[3] * ep2[33] - .5000000000*e[3] * ep2[34] - .5000000000*e[3] * ep2[35];
	A[195] = .5000000000*ep2[14] * e[8] + 1.500000000*ep2[17] * e[8] + .5000000000*e[8] * ep2[15] + .5000000000*e[8] * ep2[16] - .5000000000*e[8] * ep2[9] + .5000000000*e[8] * ep2[11] - .5000000000*e[8] * ep2[12] - .5000000000*e[8] * ep2[10] - .5000000000*e[8] * ep2[13] + e[14] * e[12] * e[6] + e[14] * e[3] * e[15] + e[14] * e[13] * e[7] + e[14] * e[4] * e[16] + e[14] * e[5] * e[17] + e[17] * e[15] * e[6] + e[17] * e[16] * e[7] + e[2] * e[11] * e[17] + e[2] * e[9] * e[15] + e[2] * e[10] * e[16] + e[5] * e[12] * e[15] + e[5] * e[13] * e[16] + e[11] * e[9] * e[6] + e[11] * e[0] * e[15] + e[11] * e[10] * e[7] + e[11] * e[1] * e[16] - 1.*e[17] * e[10] * e[1] - 1.*e[17] * e[13] * e[4] - 1.*e[17] * e[9] * e[0] - 1.*e[17] * e[12] * e[3];
	A[86] = -.5000000000*e[3] * ep2[1] - .5000000000*e[3] * ep2[7] + .5000000000*ep3[3] - .5000000000*e[3] * ep2[8] + e[0] * e[2] * e[5] + .5000000000*e[3] * ep2[6] + .5000000000*e[3] * ep2[4] - .5000000000*e[3] * ep2[2] + e[0] * e[1] * e[4] + e[6] * e[4] * e[7] + .5000000000*ep2[0] * e[3] + .5000000000*e[3] * ep2[5] + e[6] * e[5] * e[8];
	A[196] = .5000000000*ep2[23] * e[17] + 1.500000000*ep2[26] * e[17] + .5000000000*e[17] * ep2[25] + .5000000000*e[17] * ep2[24] - .5000000000*e[17] * ep2[18] - .5000000000*e[17] * ep2[19] + .5000000000*e[17] * ep2[20] - .5000000000*e[17] * ep2[22] - .5000000000*e[17] * ep2[21] + e[23] * e[21] * e[15] + e[23] * e[12] * e[24] + e[23] * e[14] * e[26] + e[23] * e[22] * e[16] + e[23] * e[13] * e[25] + e[26] * e[24] * e[15] + e[26] * e[25] * e[16] + e[11] * e[19] * e[25] + e[11] * e[18] * e[24] + e[11] * e[20] * e[26] + e[14] * e[22] * e[25] + e[14] * e[21] * e[24] + e[20] * e[18] * e[15] + e[20] * e[9] * e[24] + e[20] * e[19] * e[16] + e[20] * e[10] * e[25] - 1.*e[26] * e[18] * e[9] - 1.*e[26] * e[21] * e[12] - 1.*e[26] * e[19] * e[10] - 1.*e[26] * e[22] * e[13];
	A[87] = -1.*e[12] * e[34] * e[16] - 1.*e[12] * e[35] * e[17] - 1.*e[12] * e[29] * e[11] + e[9] * e[27] * e[12] + e[9] * e[29] * e[14] + e[9] * e[11] * e[32] + e[9] * e[28] * e[13] + e[9] * e[10] * e[31] + e[27] * e[11] * e[14] + e[27] * e[10] * e[13] + e[12] * e[32] * e[14] + e[12] * e[31] * e[13] + e[15] * e[12] * e[33] + e[15] * e[32] * e[17] + e[15] * e[14] * e[35] + e[15] * e[31] * e[16] + e[15] * e[13] * e[34] + e[33] * e[14] * e[17] + e[33] * e[13] * e[16] - 1.*e[12] * e[28] * e[10] + .5000000000*ep2[9] * e[30] - .5000000000*e[30] * ep2[16] - .5000000000*e[30] * ep2[11] + 1.500000000*e[30] * ep2[12] + .5000000000*e[30] * ep2[15] - .5000000000*e[30] * ep2[17] - .5000000000*e[30] * ep2[10] + .5000000000*e[30] * ep2[14] + .5000000000*e[30] * ep2[13];
	A[197] = e[32] * e[22] * e[16] + e[32] * e[13] * e[25] - 1.*e[17] * e[27] * e[18] + e[17] * e[33] * e[24] - 1.*e[17] * e[30] * e[21] + e[17] * e[29] * e[20] + 3.*e[17] * e[35] * e[26] - 1.*e[17] * e[31] * e[22] - 1.*e[17] * e[28] * e[19] + e[17] * e[34] * e[25] + e[20] * e[27] * e[15] + e[20] * e[9] * e[33] + e[20] * e[28] * e[16] + e[20] * e[10] * e[34] + e[29] * e[18] * e[15] + e[29] * e[9] * e[24] + e[29] * e[19] * e[16] + e[29] * e[10] * e[25] - 1.*e[26] * e[27] * e[9] - 1.*e[26] * e[30] * e[12] - 1.*e[26] * e[28] * e[10] - 1.*e[26] * e[31] * e[13] + e[26] * e[33] * e[15] + e[26] * e[34] * e[16] + e[35] * e[24] * e[15] + e[35] * e[25] * e[16] - 1.*e[35] * e[18] * e[9] - 1.*e[35] * e[21] * e[12] - 1.*e[35] * e[19] * e[10] - 1.*e[35] * e[22] * e[13] + e[14] * e[30] * e[24] + e[14] * e[21] * e[33] + e[14] * e[31] * e[25] + e[14] * e[22] * e[34] + e[14] * e[32] * e[26] + e[14] * e[23] * e[35] + e[11] * e[27] * e[24] + e[11] * e[18] * e[33] + e[11] * e[28] * e[25] + e[11] * e[19] * e[34] + e[11] * e[29] * e[26] + e[11] * e[20] * e[35] + e[23] * e[30] * e[15] + e[23] * e[12] * e[33] + e[23] * e[32] * e[17] + e[23] * e[31] * e[16] + e[23] * e[13] * e[34] + e[32] * e[21] * e[15] + e[32] * e[12] * e[24];
	A[84] = e[6] * e[23] * e[17] + e[6] * e[14] * e[26] + e[6] * e[22] * e[16] + e[6] * e[13] * e[25] + e[0] * e[20] * e[14] + e[0] * e[11] * e[23] + e[0] * e[19] * e[13] + e[0] * e[10] * e[22] - 1.*e[12] * e[26] * e[8] - 1.*e[12] * e[20] * e[2] - 1.*e[12] * e[19] * e[1] + e[12] * e[22] * e[4] - 1.*e[12] * e[25] * e[7] + e[12] * e[23] * e[5] - 1.*e[21] * e[11] * e[2] - 1.*e[21] * e[10] * e[1] + e[21] * e[13] * e[4] - 1.*e[21] * e[16] * e[7] + e[21] * e[14] * e[5] - 1.*e[21] * e[17] * e[8] + e[15] * e[3] * e[24] + e[15] * e[22] * e[7] + e[15] * e[4] * e[25] + e[15] * e[23] * e[8] + e[15] * e[5] * e[26] - 1.*e[3] * e[25] * e[16] - 1.*e[3] * e[26] * e[17] - 1.*e[3] * e[20] * e[11] - 1.*e[3] * e[19] * e[10] + e[24] * e[13] * e[7] + e[24] * e[4] * e[16] + e[24] * e[14] * e[8] + e[24] * e[5] * e[17] + e[9] * e[18] * e[3] + e[9] * e[0] * e[21] + e[9] * e[19] * e[4] + e[9] * e[1] * e[22] + e[9] * e[20] * e[5] + e[9] * e[2] * e[23] + e[18] * e[0] * e[12] + e[18] * e[10] * e[4] + e[18] * e[1] * e[13] + e[18] * e[11] * e[5] + e[18] * e[2] * e[14] + 3.*e[3] * e[21] * e[12] + e[3] * e[23] * e[14] + e[3] * e[22] * e[13] + e[6] * e[21] * e[15] + e[6] * e[12] * e[24];
	A[198] = .5000000000*ep2[5] * e[17] + 1.500000000*e[17] * ep2[8] + .5000000000*e[17] * ep2[7] + .5000000000*e[17] * ep2[6] + .5000000000*ep2[2] * e[17] - .5000000000*e[17] * ep2[4] - .5000000000*e[17] * ep2[0] - .5000000000*e[17] * ep2[1] - .5000000000*e[17] * ep2[3] + e[11] * e[1] * e[7] + e[11] * e[0] * e[6] + e[11] * e[2] * e[8] + e[5] * e[12] * e[6] + e[5] * e[3] * e[15] + e[5] * e[13] * e[7] + e[5] * e[4] * e[16] + e[5] * e[14] * e[8] + e[14] * e[4] * e[7] + e[14] * e[3] * e[6] + e[8] * e[15] * e[6] + e[8] * e[16] * e[7] - 1.*e[8] * e[10] * e[1] - 1.*e[8] * e[13] * e[4] - 1.*e[8] * e[9] * e[0] - 1.*e[8] * e[12] * e[3] + e[2] * e[9] * e[6] + e[2] * e[0] * e[15] + e[2] * e[10] * e[7] + e[2] * e[1] * e[16];
	A[85] = e[6] * e[4] * e[34] + e[6] * e[32] * e[8] + e[6] * e[5] * e[35] + e[33] * e[4] * e[7] + e[33] * e[5] * e[8] + e[0] * e[27] * e[3] + e[0] * e[28] * e[4] + e[0] * e[1] * e[31] + e[0] * e[29] * e[5] + e[0] * e[2] * e[32] - 1.*e[3] * e[34] * e[7] + e[3] * e[32] * e[5] + e[3] * e[33] * e[6] - 1.*e[3] * e[35] * e[8] - 1.*e[3] * e[29] * e[2] - 1.*e[3] * e[28] * e[1] + e[3] * e[31] * e[4] + e[27] * e[1] * e[4] + e[27] * e[2] * e[5] + e[6] * e[31] * e[7] + .5000000000*e[30] * ep2[4] + .5000000000*e[30] * ep2[6] + .5000000000*e[30] * ep2[5] - .5000000000*e[30] * ep2[1] - .5000000000*e[30] * ep2[7] - .5000000000*e[30] * ep2[2] - .5000000000*e[30] * ep2[8] + .5000000000*ep2[0] * e[30] + 1.500000000*e[30] * ep2[3];
	A[199] = .5000000000*ep2[23] * e[8] + 1.500000000*ep2[26] * e[8] - .5000000000*e[8] * ep2[18] - .5000000000*e[8] * ep2[19] - .5000000000*e[8] * ep2[22] + .5000000000*e[8] * ep2[24] - .5000000000*e[8] * ep2[21] + .5000000000*e[8] * ep2[25] + .5000000000*ep2[20] * e[8] + e[20] * e[18] * e[6] + e[20] * e[0] * e[24] + e[20] * e[19] * e[7] + e[20] * e[1] * e[25] + e[20] * e[2] * e[26] + e[23] * e[21] * e[6] + e[23] * e[3] * e[24] + e[23] * e[22] * e[7] + e[23] * e[4] * e[25] + e[23] * e[5] * e[26] - 1.*e[26] * e[21] * e[3] - 1.*e[26] * e[19] * e[1] - 1.*e[26] * e[22] * e[4] - 1.*e[26] * e[18] * e[0] + e[26] * e[25] * e[7] + e[26] * e[24] * e[6] + e[2] * e[19] * e[25] + e[2] * e[18] * e[24] + e[5] * e[22] * e[25] + e[5] * e[21] * e[24];
	A[109] = e[19] * e[27] * e[21] + e[19] * e[18] * e[30] + e[19] * e[28] * e[22] + e[19] * e[29] * e[23] + e[19] * e[20] * e[32] + e[28] * e[18] * e[21] + e[28] * e[20] * e[23] + e[22] * e[30] * e[21] + e[22] * e[32] * e[23] + e[25] * e[30] * e[24] + e[25] * e[21] * e[33] + e[25] * e[22] * e[34] + e[25] * e[32] * e[26] + e[25] * e[23] * e[35] + e[34] * e[21] * e[24] + e[34] * e[23] * e[26] - 1.*e[22] * e[27] * e[18] - 1.*e[22] * e[33] * e[24] - 1.*e[22] * e[29] * e[20] - 1.*e[22] * e[35] * e[26] + .5000000000*ep2[19] * e[31] + 1.500000000*e[31] * ep2[22] + .5000000000*e[31] * ep2[21] + .5000000000*e[31] * ep2[23] + .5000000000*e[31] * ep2[25] - .5000000000*e[31] * ep2[26] - .5000000000*e[31] * ep2[18] - .5000000000*e[31] * ep2[20] - .5000000000*e[31] * ep2[24];
	A[108] = -.5000000000*e[13] * ep2[15] + .5000000000*e[13] * ep2[16] + .5000000000*e[13] * ep2[12] + e[16] * e[12] * e[15] + .5000000000*ep3[13] + e[10] * e[11] * e[14] + .5000000000*e[13] * ep2[14] - .5000000000*e[13] * ep2[17] - .5000000000*e[13] * ep2[11] - .5000000000*e[13] * ep2[9] + .5000000000*ep2[10] * e[13] + e[10] * e[9] * e[12] + e[16] * e[14] * e[17];
	A[111] = -1.*e[13] * e[29] * e[2] - 1.*e[31] * e[11] * e[2] - 1.*e[31] * e[15] * e[6] - 1.*e[31] * e[9] * e[0] + e[31] * e[14] * e[5] + e[31] * e[12] * e[3] - 1.*e[31] * e[17] * e[8] + e[16] * e[30] * e[6] + e[16] * e[3] * e[33] + e[16] * e[4] * e[34] + e[16] * e[32] * e[8] + e[16] * e[5] * e[35] - 1.*e[4] * e[27] * e[9] + e[4] * e[28] * e[10] - 1.*e[4] * e[33] * e[15] - 1.*e[4] * e[35] * e[17] - 1.*e[4] * e[29] * e[11] + e[34] * e[12] * e[6] + e[34] * e[3] * e[15] + e[34] * e[14] * e[8] + e[34] * e[5] * e[17] + e[10] * e[27] * e[3] + e[10] * e[0] * e[30] + e[10] * e[29] * e[5] + e[10] * e[2] * e[32] + e[28] * e[9] * e[3] + e[28] * e[0] * e[12] + e[28] * e[11] * e[5] + e[28] * e[2] * e[14] + e[4] * e[30] * e[12] + e[4] * e[32] * e[14] + 3.*e[4] * e[31] * e[13] + e[7] * e[30] * e[15] + e[7] * e[12] * e[33] + e[7] * e[32] * e[17] + e[7] * e[14] * e[35] + e[7] * e[31] * e[16] + e[7] * e[13] * e[34] + e[1] * e[27] * e[12] + e[1] * e[9] * e[30] + e[1] * e[29] * e[14] + e[1] * e[11] * e[32] + e[1] * e[28] * e[13] + e[1] * e[10] * e[31] - 1.*e[13] * e[27] * e[0] + e[13] * e[32] * e[5] - 1.*e[13] * e[33] * e[6] + e[13] * e[30] * e[3] - 1.*e[13] * e[35] * e[8];
	A[110] = e[25] * e[23] * e[26] + e[19] * e[20] * e[23] + e[19] * e[18] * e[21] + e[25] * e[21] * e[24] + .5000000000*ep3[22] + .5000000000*e[22] * ep2[23] + .5000000000*ep2[19] * e[22] - .5000000000*e[22] * ep2[18] - .5000000000*e[22] * ep2[24] + .5000000000*e[22] * ep2[21] + .5000000000*e[22] * ep2[25] - .5000000000*e[22] * ep2[20] - .5000000000*e[22] * ep2[26];
	A[105] = e[34] * e[5] * e[8] + e[1] * e[27] * e[3] + e[1] * e[0] * e[30] + e[1] * e[28] * e[4] + e[1] * e[29] * e[5] + e[1] * e[2] * e[32] - 1.*e[4] * e[27] * e[0] + e[4] * e[34] * e[7] + e[4] * e[32] * e[5] - 1.*e[4] * e[33] * e[6] + e[4] * e[30] * e[3] - 1.*e[4] * e[35] * e[8] - 1.*e[4] * e[29] * e[2] + e[28] * e[0] * e[3] + e[28] * e[2] * e[5] + e[7] * e[30] * e[6] + e[7] * e[3] * e[33] + e[7] * e[32] * e[8] + e[7] * e[5] * e[35] + e[34] * e[3] * e[6] + .5000000000*ep2[1] * e[31] + 1.500000000*e[31] * ep2[4] - .5000000000*e[31] * ep2[0] - .5000000000*e[31] * ep2[6] + .5000000000*e[31] * ep2[5] + .5000000000*e[31] * ep2[7] + .5000000000*e[31] * ep2[3] - .5000000000*e[31] * ep2[2] - .5000000000*e[31] * ep2[8];
	A[104] = e[1] * e[20] * e[14] + e[1] * e[11] * e[23] + e[13] * e[21] * e[3] - 1.*e[13] * e[26] * e[8] - 1.*e[13] * e[20] * e[2] - 1.*e[13] * e[18] * e[0] + e[13] * e[23] * e[5] - 1.*e[13] * e[24] * e[6] - 1.*e[22] * e[11] * e[2] - 1.*e[22] * e[15] * e[6] - 1.*e[22] * e[9] * e[0] + e[22] * e[14] * e[5] + e[22] * e[12] * e[3] - 1.*e[22] * e[17] * e[8] + e[16] * e[21] * e[6] + e[16] * e[3] * e[24] + e[16] * e[4] * e[25] + e[16] * e[23] * e[8] + e[16] * e[5] * e[26] - 1.*e[4] * e[24] * e[15] - 1.*e[4] * e[26] * e[17] - 1.*e[4] * e[20] * e[11] - 1.*e[4] * e[18] * e[9] + e[25] * e[12] * e[6] + e[25] * e[3] * e[15] + e[25] * e[14] * e[8] + e[25] * e[5] * e[17] + e[10] * e[18] * e[3] + e[10] * e[0] * e[21] + e[10] * e[19] * e[4] + e[10] * e[1] * e[22] + e[10] * e[20] * e[5] + e[10] * e[2] * e[23] + e[19] * e[9] * e[3] + e[19] * e[0] * e[12] + e[19] * e[1] * e[13] + e[19] * e[11] * e[5] + e[19] * e[2] * e[14] + e[4] * e[21] * e[12] + e[4] * e[23] * e[14] + 3.*e[4] * e[22] * e[13] + e[7] * e[21] * e[15] + e[7] * e[12] * e[24] + e[7] * e[23] * e[17] + e[7] * e[14] * e[26] + e[7] * e[22] * e[16] + e[7] * e[13] * e[25] + e[1] * e[18] * e[12] + e[1] * e[9] * e[21];
	A[107] = e[10] * e[27] * e[12] + e[10] * e[9] * e[30] + e[10] * e[29] * e[14] + e[10] * e[11] * e[32] + e[10] * e[28] * e[13] + e[28] * e[11] * e[14] + e[28] * e[9] * e[12] + e[13] * e[30] * e[12] + e[13] * e[32] * e[14] + e[16] * e[30] * e[15] + e[16] * e[12] * e[33] + e[16] * e[32] * e[17] + e[16] * e[14] * e[35] + e[16] * e[13] * e[34] + e[34] * e[14] * e[17] + e[34] * e[12] * e[15] - 1.*e[13] * e[27] * e[9] - 1.*e[13] * e[33] * e[15] - 1.*e[13] * e[35] * e[17] - 1.*e[13] * e[29] * e[11] + .5000000000*ep2[10] * e[31] + .5000000000*e[31] * ep2[16] - .5000000000*e[31] * ep2[9] - .5000000000*e[31] * ep2[11] + .5000000000*e[31] * ep2[12] - .5000000000*e[31] * ep2[15] - .5000000000*e[31] * ep2[17] + .5000000000*e[31] * ep2[14] + 1.500000000*e[31] * ep2[13];
	A[106] = -.5000000000*e[4] * ep2[6] - .5000000000*e[4] * ep2[0] + e[1] * e[2] * e[5] + .5000000000*e[4] * ep2[7] + e[1] * e[0] * e[3] + e[7] * e[5] * e[8] - .5000000000*e[4] * ep2[8] + .5000000000*e[4] * ep2[3] + .5000000000*e[4] * ep2[5] + e[7] * e[3] * e[6] - .5000000000*e[4] * ep2[2] + .5000000000*ep3[4] + .5000000000*ep2[1] * e[4];
	A[100] = e[34] * e[32] * e[35] - .5000000000*e[31] * ep2[35] + .5000000000*e[31] * ep2[34] + .5000000000*ep2[28] * e[31] + .5000000000*ep3[31] + .5000000000*e[31] * ep2[32] + e[34] * e[30] * e[33] - .5000000000*e[31] * ep2[27] + .5000000000*e[31] * ep2[30] - .5000000000*e[31] * ep2[33] - .5000000000*e[31] * ep2[29] + e[28] * e[29] * e[32] + e[28] * e[27] * e[30];
	A[101] = e[1] * e[27] * e[30] + e[1] * e[29] * e[32] + e[1] * e[28] * e[31] + e[31] * e[30] * e[3] + e[31] * e[32] * e[5] + e[7] * e[30] * e[33] + e[7] * e[32] * e[35] + e[7] * e[31] * e[34] + e[28] * e[27] * e[3] + e[28] * e[0] * e[30] + e[28] * e[29] * e[5] + e[28] * e[2] * e[32] + e[34] * e[30] * e[6] + e[34] * e[3] * e[33] + e[34] * e[32] * e[8] + e[34] * e[5] * e[35] - 1.*e[31] * e[27] * e[0] - 1.*e[31] * e[33] * e[6] - 1.*e[31] * e[35] * e[8] - 1.*e[31] * e[29] * e[2] + .5000000000*e[4] * ep2[30] + .5000000000*e[4] * ep2[32] + 1.500000000*e[4] * ep2[31] - .5000000000*e[4] * ep2[27] + .5000000000*e[4] * ep2[28] - .5000000000*e[4] * ep2[29] - .5000000000*e[4] * ep2[33] + .5000000000*e[4] * ep2[34] - .5000000000*e[4] * ep2[35];
	A[102] = .5000000000*e[22] * ep2[30] + .5000000000*e[22] * ep2[32] + 1.500000000*e[22] * ep2[31] + .5000000000*e[22] * ep2[34] - .5000000000*e[22] * ep2[27] - .5000000000*e[22] * ep2[29] - .5000000000*e[22] * ep2[33] - .5000000000*e[22] * ep2[35] + e[28] * e[18] * e[30] + e[28] * e[29] * e[23] + e[28] * e[20] * e[32] + e[31] * e[30] * e[21] + e[31] * e[32] * e[23] + e[25] * e[30] * e[33] + e[25] * e[32] * e[35] + e[25] * e[31] * e[34] + e[34] * e[30] * e[24] + e[34] * e[21] * e[33] + e[34] * e[32] * e[26] + e[34] * e[23] * e[35] - 1.*e[31] * e[27] * e[18] - 1.*e[31] * e[33] * e[24] - 1.*e[31] * e[29] * e[20] - 1.*e[31] * e[35] * e[26] + e[19] * e[27] * e[30] + e[19] * e[29] * e[32] + e[19] * e[28] * e[31] + e[28] * e[27] * e[21] + .5000000000*ep2[28] * e[22];
	A[103] = e[16] * e[30] * e[33] + e[16] * e[32] * e[35] + e[10] * e[27] * e[30] + e[10] * e[29] * e[32] + e[10] * e[28] * e[31] + e[34] * e[30] * e[15] + e[34] * e[12] * e[33] + e[34] * e[32] * e[17] + e[34] * e[14] * e[35] + e[34] * e[31] * e[16] + e[28] * e[27] * e[12] + e[28] * e[9] * e[30] + e[28] * e[29] * e[14] + e[28] * e[11] * e[32] - 1.*e[31] * e[27] * e[9] + e[31] * e[30] * e[12] + e[31] * e[32] * e[14] - 1.*e[31] * e[33] * e[15] - 1.*e[31] * e[35] * e[17] - 1.*e[31] * e[29] * e[11] - .5000000000*e[13] * ep2[27] + .5000000000*e[13] * ep2[32] + .5000000000*e[13] * ep2[28] - .5000000000*e[13] * ep2[29] + 1.500000000*e[13] * ep2[31] - .5000000000*e[13] * ep2[33] + .5000000000*e[13] * ep2[30] + .5000000000*e[13] * ep2[34] - .5000000000*e[13] * ep2[35];
	A[96] = e[21] * e[23] * e[14] + e[21] * e[22] * e[13] + e[24] * e[21] * e[15] + e[24] * e[23] * e[17] + e[24] * e[14] * e[26] + e[24] * e[22] * e[16] + e[24] * e[13] * e[25] + e[15] * e[22] * e[25] + e[15] * e[23] * e[26] + e[9] * e[19] * e[22] + e[9] * e[18] * e[21] + e[9] * e[20] * e[23] + e[18] * e[20] * e[14] + e[18] * e[11] * e[23] + e[18] * e[19] * e[13] + e[18] * e[10] * e[22] - 1.*e[21] * e[25] * e[16] - 1.*e[21] * e[26] * e[17] - 1.*e[21] * e[20] * e[11] - 1.*e[21] * e[19] * e[10] + 1.500000000*ep2[21] * e[12] + .5000000000*e[12] * ep2[24] - .5000000000*e[12] * ep2[26] + .5000000000*e[12] * ep2[18] + .5000000000*e[12] * ep2[23] - .5000000000*e[12] * ep2[19] - .5000000000*e[12] * ep2[20] + .5000000000*e[12] * ep2[22] - .5000000000*e[12] * ep2[25];
	A[97] = -1.*e[12] * e[29] * e[20] - 1.*e[12] * e[35] * e[26] - 1.*e[12] * e[28] * e[19] - 1.*e[12] * e[34] * e[25] + e[18] * e[29] * e[14] + e[18] * e[11] * e[32] + e[18] * e[28] * e[13] + e[18] * e[10] * e[31] + e[27] * e[20] * e[14] + e[27] * e[11] * e[23] + e[27] * e[19] * e[13] + e[27] * e[10] * e[22] + e[15] * e[30] * e[24] + e[15] * e[21] * e[33] + e[15] * e[31] * e[25] + e[15] * e[22] * e[34] + e[15] * e[32] * e[26] + e[15] * e[23] * e[35] - 1.*e[21] * e[28] * e[10] - 1.*e[21] * e[34] * e[16] - 1.*e[21] * e[35] * e[17] - 1.*e[21] * e[29] * e[11] - 1.*e[30] * e[25] * e[16] - 1.*e[30] * e[26] * e[17] - 1.*e[30] * e[20] * e[11] - 1.*e[30] * e[19] * e[10] + e[24] * e[32] * e[17] + e[24] * e[14] * e[35] + e[24] * e[31] * e[16] + e[24] * e[13] * e[34] + e[33] * e[23] * e[17] + e[33] * e[14] * e[26] + e[33] * e[22] * e[16] + e[33] * e[13] * e[25] + 3.*e[12] * e[30] * e[21] + e[12] * e[31] * e[22] + e[12] * e[32] * e[23] + e[9] * e[27] * e[21] + e[9] * e[18] * e[30] + e[9] * e[28] * e[22] + e[9] * e[19] * e[31] + e[9] * e[29] * e[23] + e[9] * e[20] * e[32] + e[21] * e[32] * e[14] + e[21] * e[31] * e[13] + e[30] * e[23] * e[14] + e[30] * e[22] * e[13] + e[12] * e[27] * e[18] + e[12] * e[33] * e[24];
	A[98] = e[0] * e[11] * e[5] + e[0] * e[2] * e[14] + e[9] * e[1] * e[4] + e[9] * e[0] * e[3] + e[9] * e[2] * e[5] + e[3] * e[13] * e[4] + e[3] * e[14] * e[5] + e[6] * e[3] * e[15] + e[6] * e[13] * e[7] + e[6] * e[4] * e[16] + e[6] * e[14] * e[8] + e[6] * e[5] * e[17] + e[15] * e[4] * e[7] + e[15] * e[5] * e[8] - 1.*e[3] * e[11] * e[2] - 1.*e[3] * e[10] * e[1] - 1.*e[3] * e[16] * e[7] - 1.*e[3] * e[17] * e[8] + e[0] * e[10] * e[4] + e[0] * e[1] * e[13] + 1.500000000*e[12] * ep2[3] + .5000000000*e[12] * ep2[4] + .5000000000*e[12] * ep2[5] + .5000000000*e[12] * ep2[6] + .5000000000*ep2[0] * e[12] - .5000000000*e[12] * ep2[1] - .5000000000*e[12] * ep2[7] - .5000000000*e[12] * ep2[2] - .5000000000*e[12] * ep2[8];
	A[99] = e[21] * e[24] * e[6] + e[0] * e[19] * e[22] + e[0] * e[20] * e[23] + e[24] * e[22] * e[7] + e[24] * e[4] * e[25] + e[24] * e[23] * e[8] + e[24] * e[5] * e[26] + e[6] * e[22] * e[25] + e[6] * e[23] * e[26] + e[18] * e[0] * e[21] + e[18] * e[19] * e[4] + e[18] * e[1] * e[22] + e[18] * e[20] * e[5] + e[18] * e[2] * e[23] + e[21] * e[22] * e[4] + e[21] * e[23] * e[5] - 1.*e[21] * e[26] * e[8] - 1.*e[21] * e[20] * e[2] - 1.*e[21] * e[19] * e[1] - 1.*e[21] * e[25] * e[7] + 1.500000000*ep2[21] * e[3] + .5000000000*e[3] * ep2[22] + .5000000000*e[3] * ep2[23] + .5000000000*e[3] * ep2[24] - .5000000000*e[3] * ep2[26] - .5000000000*e[3] * ep2[19] - .5000000000*e[3] * ep2[20] - .5000000000*e[3] * ep2[25] + .5000000000*ep2[18] * e[3];
	A[127] = e[11] * e[27] * e[12] + e[11] * e[9] * e[30] + e[11] * e[29] * e[14] + e[11] * e[28] * e[13] + e[11] * e[10] * e[31] + e[29] * e[9] * e[12] + e[29] * e[10] * e[13] + e[14] * e[30] * e[12] + e[14] * e[31] * e[13] + e[17] * e[30] * e[15] + e[17] * e[12] * e[33] + e[17] * e[14] * e[35] + e[17] * e[31] * e[16] + e[17] * e[13] * e[34] + e[35] * e[12] * e[15] + e[35] * e[13] * e[16] - 1.*e[14] * e[27] * e[9] - 1.*e[14] * e[28] * e[10] - 1.*e[14] * e[33] * e[15] - 1.*e[14] * e[34] * e[16] + .5000000000*ep2[11] * e[32] - .5000000000*e[32] * ep2[16] - .5000000000*e[32] * ep2[9] + .5000000000*e[32] * ep2[12] - .5000000000*e[32] * ep2[15] + .5000000000*e[32] * ep2[17] - .5000000000*e[32] * ep2[10] + 1.500000000*e[32] * ep2[14] + .5000000000*e[32] * ep2[13];
	A[126] = e[8] * e[3] * e[6] + .5000000000*ep2[2] * e[5] - .5000000000*e[5] * ep2[0] + .5000000000*e[5] * ep2[4] - .5000000000*e[5] * ep2[6] + .5000000000*e[5] * ep2[8] + e[8] * e[4] * e[7] + .5000000000*ep3[5] + e[2] * e[0] * e[3] + .5000000000*e[5] * ep2[3] - .5000000000*e[5] * ep2[7] + e[2] * e[1] * e[4] - .5000000000*e[5] * ep2[1];
	A[125] = e[2] * e[27] * e[3] + e[2] * e[0] * e[30] + e[2] * e[28] * e[4] + e[2] * e[1] * e[31] + e[2] * e[29] * e[5] - 1.*e[5] * e[27] * e[0] - 1.*e[5] * e[34] * e[7] - 1.*e[5] * e[33] * e[6] + e[5] * e[30] * e[3] + e[5] * e[35] * e[8] - 1.*e[5] * e[28] * e[1] + e[5] * e[31] * e[4] + e[29] * e[1] * e[4] + e[29] * e[0] * e[3] + e[8] * e[30] * e[6] + e[8] * e[3] * e[33] + e[8] * e[31] * e[7] + e[8] * e[4] * e[34] + e[35] * e[4] * e[7] + e[35] * e[3] * e[6] + .5000000000*ep2[2] * e[32] + 1.500000000*e[32] * ep2[5] + .5000000000*e[32] * ep2[4] - .5000000000*e[32] * ep2[0] - .5000000000*e[32] * ep2[6] - .5000000000*e[32] * ep2[1] - .5000000000*e[32] * ep2[7] + .5000000000*e[32] * ep2[3] + .5000000000*e[32] * ep2[8];
	A[124] = -1.*e[14] * e[19] * e[1] + e[14] * e[22] * e[4] - 1.*e[14] * e[18] * e[0] - 1.*e[14] * e[25] * e[7] - 1.*e[14] * e[24] * e[6] - 1.*e[23] * e[10] * e[1] + e[23] * e[13] * e[4] - 1.*e[23] * e[16] * e[7] - 1.*e[23] * e[15] * e[6] - 1.*e[23] * e[9] * e[0] + e[23] * e[12] * e[3] + e[17] * e[21] * e[6] + e[17] * e[3] * e[24] + e[17] * e[22] * e[7] + e[17] * e[4] * e[25] + e[17] * e[5] * e[26] - 1.*e[5] * e[24] * e[15] - 1.*e[5] * e[25] * e[16] - 1.*e[5] * e[18] * e[9] - 1.*e[5] * e[19] * e[10] + e[26] * e[12] * e[6] + e[26] * e[3] * e[15] + e[26] * e[13] * e[7] + e[26] * e[4] * e[16] + e[11] * e[18] * e[3] + e[11] * e[0] * e[21] + e[11] * e[19] * e[4] + e[11] * e[1] * e[22] + e[11] * e[20] * e[5] + e[11] * e[2] * e[23] + e[20] * e[9] * e[3] + e[20] * e[0] * e[12] + e[20] * e[10] * e[4] + e[20] * e[1] * e[13] + e[20] * e[2] * e[14] + e[5] * e[21] * e[12] + 3.*e[5] * e[23] * e[14] + e[5] * e[22] * e[13] + e[8] * e[21] * e[15] + e[8] * e[12] * e[24] + e[8] * e[23] * e[17] + e[8] * e[14] * e[26] + e[8] * e[22] * e[16] + e[8] * e[13] * e[25] + e[2] * e[18] * e[12] + e[2] * e[9] * e[21] + e[2] * e[19] * e[13] + e[2] * e[10] * e[22] + e[14] * e[21] * e[3];
	A[123] = -.5000000000*e[14] * ep2[27] + 1.500000000*e[14] * ep2[32] - .5000000000*e[14] * ep2[28] + .5000000000*e[14] * ep2[29] + .5000000000*e[14] * ep2[31] - .5000000000*e[14] * ep2[33] + .5000000000*e[14] * ep2[30] - .5000000000*e[14] * ep2[34] + .5000000000*e[14] * ep2[35] + e[11] * e[27] * e[30] + e[11] * e[29] * e[32] + e[11] * e[28] * e[31] + e[35] * e[30] * e[15] + e[35] * e[12] * e[33] + e[35] * e[32] * e[17] + e[35] * e[31] * e[16] + e[35] * e[13] * e[34] + e[29] * e[27] * e[12] + e[29] * e[9] * e[30] + e[29] * e[28] * e[13] + e[29] * e[10] * e[31] - 1.*e[32] * e[27] * e[9] + e[32] * e[30] * e[12] - 1.*e[32] * e[28] * e[10] + e[32] * e[31] * e[13] - 1.*e[32] * e[33] * e[15] - 1.*e[32] * e[34] * e[16] + e[17] * e[30] * e[33] + e[17] * e[31] * e[34];
	A[122] = -.5000000000*e[23] * ep2[33] - .5000000000*e[23] * ep2[34] + .5000000000*ep2[29] * e[23] + .5000000000*e[23] * ep2[30] + 1.500000000*e[23] * ep2[32] + .5000000000*e[23] * ep2[31] + .5000000000*e[23] * ep2[35] - .5000000000*e[23] * ep2[27] - .5000000000*e[23] * ep2[28] + e[32] * e[30] * e[21] + e[32] * e[31] * e[22] + e[26] * e[30] * e[33] + e[26] * e[32] * e[35] + e[26] * e[31] * e[34] + e[35] * e[30] * e[24] + e[35] * e[21] * e[33] + e[35] * e[31] * e[25] + e[35] * e[22] * e[34] - 1.*e[32] * e[27] * e[18] - 1.*e[32] * e[33] * e[24] - 1.*e[32] * e[28] * e[19] - 1.*e[32] * e[34] * e[25] + e[20] * e[27] * e[30] + e[20] * e[29] * e[32] + e[20] * e[28] * e[31] + e[29] * e[27] * e[21] + e[29] * e[18] * e[30] + e[29] * e[28] * e[22] + e[29] * e[19] * e[31];
	A[121] = e[2] * e[27] * e[30] + e[2] * e[29] * e[32] + e[2] * e[28] * e[31] + e[32] * e[30] * e[3] + e[32] * e[31] * e[4] + e[8] * e[30] * e[33] + e[8] * e[32] * e[35] + e[8] * e[31] * e[34] + e[29] * e[27] * e[3] + e[29] * e[0] * e[30] + e[29] * e[28] * e[4] + e[29] * e[1] * e[31] + e[35] * e[30] * e[6] + e[35] * e[3] * e[33] + e[35] * e[31] * e[7] + e[35] * e[4] * e[34] - 1.*e[32] * e[27] * e[0] - 1.*e[32] * e[34] * e[7] - 1.*e[32] * e[33] * e[6] - 1.*e[32] * e[28] * e[1] + .5000000000*e[5] * ep2[30] + 1.500000000*e[5] * ep2[32] + .5000000000*e[5] * ep2[31] - .5000000000*e[5] * ep2[27] - .5000000000*e[5] * ep2[28] + .5000000000*e[5] * ep2[29] - .5000000000*e[5] * ep2[33] - .5000000000*e[5] * ep2[34] + .5000000000*e[5] * ep2[35];
	A[120] = .5000000000*e[32] * ep2[31] + .5000000000*e[32] * ep2[35] - .5000000000*e[32] * ep2[27] + e[29] * e[27] * e[30] + e[29] * e[28] * e[31] + e[35] * e[30] * e[33] + e[35] * e[31] * e[34] + .5000000000*ep2[29] * e[32] + .5000000000*ep3[32] - .5000000000*e[32] * ep2[33] - .5000000000*e[32] * ep2[34] + .5000000000*e[32] * ep2[30] - .5000000000*e[32] * ep2[28];
	A[118] = e[10] * e[1] * e[4] + e[10] * e[0] * e[3] + e[10] * e[2] * e[5] + e[4] * e[12] * e[3] + e[4] * e[14] * e[5] + e[7] * e[12] * e[6] + e[7] * e[3] * e[15] + e[7] * e[4] * e[16] + e[7] * e[14] * e[8] + e[7] * e[5] * e[17] + e[16] * e[3] * e[6] + e[16] * e[5] * e[8] - 1.*e[4] * e[11] * e[2] - 1.*e[4] * e[15] * e[6] - 1.*e[4] * e[9] * e[0] - 1.*e[4] * e[17] * e[8] + e[1] * e[9] * e[3] + e[1] * e[0] * e[12] + e[1] * e[11] * e[5] + e[1] * e[2] * e[14] + 1.500000000*e[13] * ep2[4] + .5000000000*e[13] * ep2[3] + .5000000000*e[13] * ep2[5] + .5000000000*e[13] * ep2[7] + .5000000000*ep2[1] * e[13] - .5000000000*e[13] * ep2[0] - .5000000000*e[13] * ep2[6] - .5000000000*e[13] * ep2[2] - .5000000000*e[13] * ep2[8];
	A[119] = e[25] * e[21] * e[6] + e[25] * e[3] * e[24] + e[25] * e[23] * e[8] + e[25] * e[5] * e[26] + e[7] * e[21] * e[24] + e[7] * e[23] * e[26] + e[19] * e[18] * e[3] + e[19] * e[0] * e[21] + e[19] * e[1] * e[22] + e[19] * e[20] * e[5] + e[19] * e[2] * e[23] + e[22] * e[21] * e[3] + e[22] * e[23] * e[5] - 1.*e[22] * e[26] * e[8] - 1.*e[22] * e[20] * e[2] - 1.*e[22] * e[18] * e[0] + e[22] * e[25] * e[7] - 1.*e[22] * e[24] * e[6] + e[1] * e[18] * e[21] + e[1] * e[20] * e[23] + .5000000000*e[4] * ep2[25] - .5000000000*e[4] * ep2[26] - .5000000000*e[4] * ep2[18] - .5000000000*e[4] * ep2[20] - .5000000000*e[4] * ep2[24] + .5000000000*ep2[19] * e[4] + 1.500000000*ep2[22] * e[4] + .5000000000*e[4] * ep2[21] + .5000000000*e[4] * ep2[23];
	A[116] = e[22] * e[21] * e[12] + e[22] * e[23] * e[14] + e[25] * e[21] * e[15] + e[25] * e[12] * e[24] + e[25] * e[23] * e[17] + e[25] * e[14] * e[26] + e[25] * e[22] * e[16] + e[16] * e[21] * e[24] + e[16] * e[23] * e[26] + e[10] * e[19] * e[22] + e[10] * e[18] * e[21] + e[10] * e[20] * e[23] + e[19] * e[18] * e[12] + e[19] * e[9] * e[21] + e[19] * e[20] * e[14] + e[19] * e[11] * e[23] - 1.*e[22] * e[24] * e[15] - 1.*e[22] * e[26] * e[17] - 1.*e[22] * e[20] * e[11] - 1.*e[22] * e[18] * e[9] - .5000000000*e[13] * ep2[26] - .5000000000*e[13] * ep2[18] + .5000000000*e[13] * ep2[23] + .5000000000*e[13] * ep2[19] - .5000000000*e[13] * ep2[20] - .5000000000*e[13] * ep2[24] + .5000000000*e[13] * ep2[21] + 1.500000000*ep2[22] * e[13] + .5000000000*e[13] * ep2[25];
	A[117] = e[13] * e[30] * e[21] + 3.*e[13] * e[31] * e[22] + e[13] * e[32] * e[23] + e[10] * e[27] * e[21] + e[10] * e[18] * e[30] + e[10] * e[28] * e[22] + e[10] * e[19] * e[31] + e[10] * e[29] * e[23] + e[10] * e[20] * e[32] + e[22] * e[30] * e[12] + e[22] * e[32] * e[14] + e[31] * e[21] * e[12] + e[31] * e[23] * e[14] - 1.*e[13] * e[27] * e[18] - 1.*e[13] * e[33] * e[24] - 1.*e[13] * e[29] * e[20] - 1.*e[13] * e[35] * e[26] + e[13] * e[28] * e[19] + e[13] * e[34] * e[25] + e[19] * e[27] * e[12] + e[19] * e[9] * e[30] + e[19] * e[29] * e[14] + e[19] * e[11] * e[32] + e[28] * e[18] * e[12] + e[28] * e[9] * e[21] + e[28] * e[20] * e[14] + e[28] * e[11] * e[23] + e[16] * e[30] * e[24] + e[16] * e[21] * e[33] + e[16] * e[31] * e[25] + e[16] * e[22] * e[34] + e[16] * e[32] * e[26] + e[16] * e[23] * e[35] - 1.*e[22] * e[27] * e[9] - 1.*e[22] * e[33] * e[15] - 1.*e[22] * e[35] * e[17] - 1.*e[22] * e[29] * e[11] - 1.*e[31] * e[24] * e[15] - 1.*e[31] * e[26] * e[17] - 1.*e[31] * e[20] * e[11] - 1.*e[31] * e[18] * e[9] + e[25] * e[30] * e[15] + e[25] * e[12] * e[33] + e[25] * e[32] * e[17] + e[25] * e[14] * e[35] + e[34] * e[21] * e[15] + e[34] * e[12] * e[24] + e[34] * e[23] * e[17] + e[34] * e[14] * e[26];
	A[114] = e[19] * e[11] * e[14] + e[19] * e[9] * e[12] + e[19] * e[10] * e[13] + e[13] * e[21] * e[12] + e[13] * e[23] * e[14] + e[16] * e[21] * e[15] + e[16] * e[12] * e[24] + e[16] * e[23] * e[17] + e[16] * e[14] * e[26] + e[16] * e[13] * e[25] + e[25] * e[14] * e[17] + e[25] * e[12] * e[15] - 1.*e[13] * e[24] * e[15] - 1.*e[13] * e[26] * e[17] - 1.*e[13] * e[20] * e[11] - 1.*e[13] * e[18] * e[9] + e[10] * e[18] * e[12] + e[10] * e[9] * e[21] + e[10] * e[20] * e[14] + e[10] * e[11] * e[23] + 1.500000000*e[22] * ep2[13] + .5000000000*e[22] * ep2[14] + .5000000000*e[22] * ep2[12] + .5000000000*e[22] * ep2[16] + .5000000000*ep2[10] * e[22] - .5000000000*e[22] * ep2[9] - .5000000000*e[22] * ep2[11] - .5000000000*e[22] * ep2[15] - .5000000000*e[22] * ep2[17];
	A[115] = e[13] * e[12] * e[3] + e[13] * e[14] * e[5] + e[16] * e[12] * e[6] + e[16] * e[3] * e[15] + e[16] * e[13] * e[7] + e[16] * e[14] * e[8] + e[16] * e[5] * e[17] + e[7] * e[14] * e[17] + e[7] * e[12] * e[15] + e[1] * e[11] * e[14] + e[1] * e[9] * e[12] + e[1] * e[10] * e[13] + e[10] * e[9] * e[3] + e[10] * e[0] * e[12] + e[10] * e[11] * e[5] + e[10] * e[2] * e[14] - 1.*e[13] * e[11] * e[2] - 1.*e[13] * e[15] * e[6] - 1.*e[13] * e[9] * e[0] - 1.*e[13] * e[17] * e[8] + 1.500000000*ep2[13] * e[4] + .5000000000*e[4] * ep2[16] - .5000000000*e[4] * ep2[9] - .5000000000*e[4] * ep2[11] + .5000000000*e[4] * ep2[12] - .5000000000*e[4] * ep2[15] - .5000000000*e[4] * ep2[17] + .5000000000*e[4] * ep2[10] + .5000000000*e[4] * ep2[14];
	A[112] = e[19] * e[1] * e[4] + e[19] * e[0] * e[3] + e[19] * e[2] * e[5] + e[4] * e[21] * e[3] + e[4] * e[23] * e[5] + e[7] * e[21] * e[6] + e[7] * e[3] * e[24] + e[7] * e[4] * e[25] + e[7] * e[23] * e[8] + e[7] * e[5] * e[26] + e[25] * e[3] * e[6] + e[25] * e[5] * e[8] + e[1] * e[18] * e[3] + e[1] * e[0] * e[21] + e[1] * e[20] * e[5] + e[1] * e[2] * e[23] - 1.*e[4] * e[26] * e[8] - 1.*e[4] * e[20] * e[2] - 1.*e[4] * e[18] * e[0] - 1.*e[4] * e[24] * e[6] + 1.500000000*e[22] * ep2[4] - .5000000000*e[22] * ep2[0] - .5000000000*e[22] * ep2[6] + .5000000000*e[22] * ep2[5] + .5000000000*e[22] * ep2[1] + .5000000000*e[22] * ep2[7] + .5000000000*e[22] * ep2[3] - .5000000000*e[22] * ep2[2] - .5000000000*e[22] * ep2[8];
	A[113] = -1.*e[31] * e[20] * e[2] - 1.*e[31] * e[18] * e[0] + e[31] * e[23] * e[5] - 1.*e[31] * e[24] * e[6] + e[7] * e[30] * e[24] + e[7] * e[21] * e[33] + e[7] * e[32] * e[26] + e[7] * e[23] * e[35] + e[25] * e[30] * e[6] + e[25] * e[3] * e[33] + e[25] * e[31] * e[7] + e[25] * e[4] * e[34] + e[25] * e[32] * e[8] + e[25] * e[5] * e[35] + e[34] * e[21] * e[6] + e[34] * e[3] * e[24] + e[34] * e[22] * e[7] + e[34] * e[23] * e[8] + e[34] * e[5] * e[26] + e[1] * e[27] * e[21] + e[1] * e[18] * e[30] + e[1] * e[28] * e[22] + e[1] * e[19] * e[31] + e[1] * e[29] * e[23] + e[1] * e[20] * e[32] + e[19] * e[27] * e[3] + e[19] * e[0] * e[30] + e[19] * e[28] * e[4] + e[19] * e[29] * e[5] + e[19] * e[2] * e[32] + e[28] * e[18] * e[3] + e[28] * e[0] * e[21] + e[28] * e[20] * e[5] + e[28] * e[2] * e[23] + e[4] * e[30] * e[21] + 3.*e[4] * e[31] * e[22] + e[4] * e[32] * e[23] - 1.*e[4] * e[27] * e[18] - 1.*e[4] * e[33] * e[24] - 1.*e[4] * e[29] * e[20] - 1.*e[4] * e[35] * e[26] - 1.*e[22] * e[27] * e[0] + e[22] * e[32] * e[5] - 1.*e[22] * e[33] * e[6] + e[22] * e[30] * e[3] - 1.*e[22] * e[35] * e[8] - 1.*e[22] * e[29] * e[2] + e[31] * e[21] * e[3] - 1.*e[31] * e[26] * e[8];

	int perm[20] = { 6, 8, 18, 15, 12, 5, 14, 7, 4, 11, 19, 13, 1, 16, 17, 3, 10, 9, 2, 0 };
	double AA[200];
	for (int i = 0; i < 20; i++)
	{
		for (int j = 0; j < 10; j++) AA[i + j * 20] = A[perm[i] + j * 20];
	}

	for (int i = 0; i < 200; i++)
	{
		A[i] = AA[i];
	}
}

// Input should be a vector of n 2D points or a Nx2 matrix
Mat findEssentialMat(InputArray _points1, InputArray _points2, Mat K1, Mat K2, int method, double prob, double threshold, int maxIters, OutputArray _mask)
{
	Mat points1, points2;
	_points1.getMat().copyTo(points1);
	_points2.getMat().copyTo(points2);

	int npoints = points1.checkVector(2);
	CV_Assert(npoints >= 5 && points2.checkVector(2) == npoints &&
		points1.type() == points2.type());

	if (points1.channels() > 1)
	{
		points1 = points1.reshape(1, npoints);
		points2 = points2.reshape(1, npoints);
	}
	points1.convertTo(points1, CV_64F);
	points2.convertTo(points2, CV_64F);

	double f1 = K1.at<double>(0), f2 = K2.at<double>(0), u1 = K1.at<double>(2), v1 = K1.at<double>(4);
	points1.col(0) = (points1.col(0) - K1.at<double>(2)) / K1.at<double>(0);
	points1.col(1) = (points1.col(1) - K1.at<double>(5)) / K1.at<double>(4);
	points2.col(0) = (points2.col(0) - K2.at<double>(2)) / K2.at<double>(0);
	points2.col(1) = (points2.col(1) - K2.at<double>(5)) / K2.at<double>(4);

	// Reshape data to fit opencv ransac function
	points1 = points1.reshape(2, 1);
	points2 = points2.reshape(2, 1);

	Mat E(3, 3, CV_64F);
	CvEMEstimator estimator;

	CvMat p1 = points1;
	CvMat p2 = points2;
	CvMat _E = E;
	CvMat* tempMask = cvCreateMat(1, npoints, CV_8U);

	assert(npoints >= 5);
	threshold /= 0.25*(K1.at<double>(0) + K1.at<double>(4) + K2.at<double>(0) + K2.at<double>(4));
	int count = 1;
	if (npoints == 5)
	{
		E.create(3 * 10, 3, CV_64F);
		_E = E;
		count = estimator.runKernel(&p1, &p2, &_E);
		E = E.rowRange(0, 3 * count) * 1.0;
		Mat(tempMask).setTo(true);
	}
	else if (method == CV_RANSAC)
	{
		estimator.runRANSAC(&p1, &p2, &_E, tempMask, threshold, prob, maxIters);
	}
	else
	{
		estimator.runLMeDS(&p1, &p2, &_E, tempMask, prob);
	}

	if (_mask.needed())
	{
		_mask.create(1, npoints, CV_8U, -1, true);
		Mat mask = _mask.getMat();
		Mat(tempMask).copyTo(mask);
	}


	return E;

}
void decomposeEssentialMat(const Mat & E, Mat & R1, Mat & R2, Mat & t)
{
	assert(E.cols == 3 && E.rows == 3);
	Mat D, U, Vt;
	SVD::compute(E, D, U, Vt);
	if (determinant(U) < 0) U = -U;
	if (determinant(Vt) < 0) Vt = -Vt;
	Mat W = (Mat_<double>(3, 3) << 0, 1, 0, -1, 0, 0, 0, 0, 1);
	W.convertTo(W, E.type());
	R1 = U * W * Vt;
	R2 = U * W.t() * Vt;
	t = U.col(2) * 1.0;
}
int recoverPose(const Mat & E, InputArray _points1, InputArray _points2, Mat & _R, Mat & _t, Mat K1, Mat K2, InputOutputArray _mask)
{
	Mat points1, points2;
	_points1.getMat().copyTo(points1);
	_points2.getMat().copyTo(points2);
	int npoints = points1.checkVector(2);
	CV_Assert(npoints >= 0 && points2.checkVector(2) == npoints &&
		points1.type() == points2.type());

	if (points1.channels() > 1)
	{
		points1 = points1.reshape(1, npoints);
		points2 = points2.reshape(1, npoints);
	}
	points1.convertTo(points1, CV_64F);
	points2.convertTo(points2, CV_64F);

	points1.col(0) = (points1.col(0) - K1.at<double>(2)) / K1.at<double>(0);
	points1.col(1) = (points1.col(1) - K1.at<double>(5)) / K1.at<double>(4);
	points2.col(0) = (points2.col(0) - K2.at<double>(2)) / K2.at<double>(0);
	points2.col(1) = (points2.col(1) - K2.at<double>(5)) / K2.at<double>(4);

	points1 = points1.t();
	points2 = points2.t();

	Mat R1, R2, t;
	decomposeEssentialMat(E, R1, R2, t);
	Mat P0 = Mat::eye(3, 4, R1.type());
	Mat P1(3, 4, R1.type()), P2(3, 4, R1.type()), P3(3, 4, R1.type()), P4(3, 4, R1.type());
	P1(Range::all(), Range(0, 3)) = R1 * 1.0; P1.col(3) = t * 1.0;
	P2(Range::all(), Range(0, 3)) = R2 * 1.0; P2.col(3) = t * 1.0;
	P3(Range::all(), Range(0, 3)) = R1 * 1.0; P3.col(3) = -t * 1.0;
	P4(Range::all(), Range(0, 3)) = R2 * 1.0; P4.col(3) = -t * 1.0;

	// Do the cheirality check. 
	// Notice here a threshold dist is used to filter
	// out far away points (i.e. infinite points) since 
	// there depth may vary between postive and negtive. 
	double dist = 50.0;
	Mat Q;
	triangulatePoints(P0, P1, points1, points2, Q);
	Mat mask1 = Q.row(2).mul(Q.row(3)) > 0;
	Q.row(0) /= Q.row(3);
	Q.row(1) /= Q.row(3);
	Q.row(2) /= Q.row(3);
	Q.row(3) /= Q.row(3);
	mask1 = (Q.row(2) < dist) & mask1;
	Q = P1 * Q;
	mask1 = (Q.row(2) > 0) & mask1;
	mask1 = (Q.row(2) < dist) & mask1;

	triangulatePoints(P0, P2, points1, points2, Q);
	Mat mask2 = Q.row(2).mul(Q.row(3)) > 0;
	Q.row(0) /= Q.row(3);
	Q.row(1) /= Q.row(3);
	Q.row(2) /= Q.row(3);
	Q.row(3) /= Q.row(3);
	mask2 = (Q.row(2) < dist) & mask2;
	Q = P2 * Q;
	mask2 = (Q.row(2) > 0) & mask2;
	mask2 = (Q.row(2) < dist) & mask2;

	triangulatePoints(P0, P3, points1, points2, Q);
	Mat mask3 = Q.row(2).mul(Q.row(3)) > 0;
	Q.row(0) /= Q.row(3);
	Q.row(1) /= Q.row(3);
	Q.row(2) /= Q.row(3);
	Q.row(3) /= Q.row(3);
	mask3 = (Q.row(2) < dist) & mask3;
	Q = P3 * Q;
	mask3 = (Q.row(2) > 0) & mask3;
	mask3 = (Q.row(2) < dist) & mask3;

	triangulatePoints(P0, P4, points1, points2, Q);
	Mat mask4 = Q.row(2).mul(Q.row(3)) > 0;
	Q.row(0) /= Q.row(3);
	Q.row(1) /= Q.row(3);
	Q.row(2) /= Q.row(3);
	Q.row(3) /= Q.row(3);
	mask4 = (Q.row(2) < dist) & mask4;
	Q = P4 * Q;
	mask4 = (Q.row(2) > 0) & mask4;
	mask4 = (Q.row(2) < dist) & mask4;

	// If _mask is given, then use it to filter outliers. 
	if (_mask.needed())
	{
		_mask.create(1, npoints, CV_8U, -1, true);
		Mat mask = _mask.getMat();
		bitwise_and(mask, mask1, mask1);
		bitwise_and(mask, mask2, mask2);
		bitwise_and(mask, mask3, mask3);
		bitwise_and(mask, mask4, mask4);
	}

	int good1 = countNonZero(mask1);
	int good2 = countNonZero(mask2);
	int good3 = countNonZero(mask3);
	int good4 = countNonZero(mask4);
	if (good1 >= good2 && good1 >= good3 && good1 >= good4)
	{
		_R = R1; _t = t;
		if (_mask.needed()) mask1.copyTo(_mask.getMat());
		return good1;
	}
	else if (good2 >= good1 && good2 >= good3 && good2 >= good4)
	{
		_R = R2; _t = t;
		if (_mask.needed()) mask2.copyTo(_mask.getMat());
		return good2;
	}
	else if (good3 >= good1 && good3 >= good2 && good3 >= good4)
	{
		_R = R1; _t = -t;
		if (_mask.needed()) mask3.copyTo(_mask.getMat());
		return good3;
	}
	else
	{
		_R = R2; _t = -t;
		if (_mask.needed()) mask4.copyTo(_mask.getMat());
		return good4;
	}

}

int TwoCamerasReconstructionFmat(CameraData *AllViewsInfo, Point2d *PCcorres, Point3d *ThreeD, int *CameraPair, int nCams, int nProjectors, int nPpts)
{
	int ii, jj;

	//Estimate fundamental matrix
	vector<int> pid;
	vector<Point2f>imgpts1, imgpts2;
	for (ii = 0; ii < nPpts*nProjectors; ii++)
	{
		if (PCcorres[CameraPair[0] + ii*(nProjectors + nCams)].x > 0.0001 && PCcorres[CameraPair[1] + ii*(nProjectors + nCams)].x > 0.0001)
		{
			pid.push_back(ii);
			imgpts1.push_back(Point2f(PCcorres[CameraPair[0] + ii*(nProjectors + nCams)].x, PCcorres[CameraPair[0] + ii*(nProjectors + nCams)].y));
			imgpts2.push_back(Point2f(PCcorres[CameraPair[1] + ii*(nProjectors + nCams)].x, PCcorres[CameraPair[1] + ii*(nProjectors + nCams)].y));
		}
	}
	Mat cvF = findFundamentalMat(imgpts1, imgpts2, FM_8POINT, 0.1, 0.99);

	cout << "Fmat: " << endl;
	cout << cvF << endl << endl;

	//Extract essential matrix
	Mat cvK1(3, 3, CV_64F, AllViewsInfo[CameraPair[0]].K);
	Mat cvK2(3, 3, CV_64F, AllViewsInfo[CameraPair[1]].K);

	Mat cvE = cvK2.t()*cvF*cvK1;
	cout << "Emat: " << endl;
	cout << cvE << endl << endl;

	//Decompose into RT
	SVD svd(cvE, SVD::MODIFY_A);
	double m = (svd.w.at<double>(0) + svd.w.at<double>(1)) / 2;
	double nW[9] = { m, 0, 0, 0, m, 0, 0, 0, 0 };
	Mat cvnW(3, 3, CV_64F, nW);
	Mat cvnE = svd.u*cvnW*svd.vt;
	SVD svd2(cvnE, SVD::MODIFY_A);

	cvnW.at<double>(0, 0) = 0.0, cvnW.at<double>(0, 1) = -1.0, cvnW.at<double>(0, 2) = 0.0;
	cvnW.at<double>(1, 0) = 1.0, cvnW.at<double>(1, 1) = 0.0, cvnW.at<double>(1, 2) = 0.0;
	cvnW.at<double>(2, 0) = 0.0, cvnW.at<double>(2, 1) = 0.0, cvnW.at<double>(2, 2) = 1.0;

	//Make sure we return rotation matrices with det(R) == 1
	Mat UWVt = svd2.u*cvnW*svd2.vt;
	if (determinant(UWVt) < 0.0)
	{
		cvnW.at<double>(0, 0) = 0.0, cvnW.at<double>(0, 1) = 1.0, cvnW.at<double>(0, 2) = 0.0;
		cvnW.at<double>(1, 0) = -1.0, cvnW.at<double>(1, 1) = 0.0, cvnW.at<double>(1, 2) = 0.0;
		cvnW.at<double>(2, 0) = 0.0, cvnW.at<double>(2, 1) = 0.0, cvnW.at<double>(2, 2) = -1.0;
	}

	UWVt = svd2.u*cvnW*svd2.vt;
	Mat UWtVt = svd2.u*cvnW.t()*svd2.vt;

	double maxU = 0.0;
	for (ii = 0; ii < 3; ii++)
		if (maxU < abs(svd2.u.at<double>(ii, 2)))
			maxU = abs(svd2.u.at<double>(ii, 2));

	//There are 4 possible cases
	double RT2[] = { UWVt.at<double>(0, 0), UWVt.at<double>(0, 1), UWVt.at<double>(0, 2), svd2.u.at<double>(0, 2) / maxU,
		UWVt.at<double>(1, 0), UWVt.at<double>(1, 1), UWVt.at<double>(1, 2), svd2.u.at<double>(1, 2) / maxU,
		UWVt.at<double>(2, 0), UWVt.at<double>(2, 1), UWVt.at<double>(2, 2), svd2.u.at<double>(2, 2) / maxU,

		UWVt.at<double>(0, 0), UWVt.at<double>(0, 1), UWVt.at<double>(0, 2), -svd2.u.at<double>(0, 2) / maxU,
		UWVt.at<double>(1, 0), UWVt.at<double>(1, 1), UWVt.at<double>(1, 2), -svd2.u.at<double>(1, 2) / maxU,
		UWVt.at<double>(2, 0), UWVt.at<double>(2, 1), UWVt.at<double>(2, 2), -svd2.u.at<double>(2, 2) / maxU,

		UWtVt.at<double>(0, 0), UWtVt.at<double>(0, 1), UWtVt.at<double>(0, 2), svd2.u.at<double>(0, 2) / maxU,
		UWtVt.at<double>(1, 0), UWtVt.at<double>(1, 1), UWtVt.at<double>(1, 2), svd2.u.at<double>(1, 2) / maxU,
		UWtVt.at<double>(2, 0), UWtVt.at<double>(2, 1), UWtVt.at<double>(2, 2), svd2.u.at<double>(2, 2) / maxU,

		UWtVt.at<double>(0, 0), UWtVt.at<double>(0, 1), UWtVt.at<double>(0, 2), -svd2.u.at<double>(0, 2) / maxU,
		UWtVt.at<double>(1, 0), UWtVt.at<double>(1, 1), UWtVt.at<double>(1, 2), -svd2.u.at<double>(1, 2) / maxU,
		UWtVt.at<double>(2, 0), UWtVt.at<double>(2, 1), UWtVt.at<double>(2, 2), -svd2.u.at<double>(2, 2) / maxU };

	//Cherality check
	double P1[12], P2[48], RT1[] = { 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 };
	mat_mul(AllViewsInfo[CameraPair[0]].K, RT1, P1, 3, 3, 4);
	for (ii = 0; ii < 4; ii++)
		mat_mul(AllViewsInfo[CameraPair[1]].K, RT2 + 12 * ii, P2 + 12 * ii, 3, 3, 4);

	Point2d pt1, pt2; Point3d WC;
	int positivePoints[4] = { 0, 0, 0, 0 }, RTid[4] = { 0, 1, 2, 3 };
	for (ii = 0; ii < imgpts1.size(); ii++)
	{
		pt1.x = imgpts1.at(ii).x, pt1.y = imgpts1.at(ii).y;
		pt2.x = imgpts2.at(ii).x, pt2.y = imgpts2.at(ii).y;

		for (jj = 0; jj<4; jj++)
		{
			//Stereo_Triangulation2(&pt1, &pt2, P1, P2+12*jj, &WC); 
			if (WC.z > 0.0)
				positivePoints[jj] ++;
		}
	}

	Quick_Sort_Int(positivePoints, RTid, 0, 3);
	if (positivePoints[3] < 10)
	{
		cout << "Something wrong. Only a few points have positve z values" << endl;
		return 1;
	}

	/*for(ii=0; ii<6; ii++)
	AllViewsInfo[CameraPair[0]].rt[ii] = 0.0;

	Mat rvec, Rmat = (Mat_<double>(3,3) << RT2[RTid[3]*12], RT2[RTid[3]*12+1], RT2[RTid[3]*12+2],
	RT2[RTid[3]*12+4], RT2[RTid[3]*12+5], RT2[RTid[3]*12+6],
	RT2[RTid[3]*12+8], RT2[RTid[3]*12+9], RT2[RTid[3]*12+10]);
	Rodrigues( Rmat, rvec);

	for(ii=0; ii<3; ii++)
	{
	AllViewsInfo[CameraPair[1]].RT[ii] = rvec.at<double>(ii);
	AllViewsInfo[CameraPair[1]].RT[ii+3] = RT2[RTid[3]*12+3+4*ii];
	}

	//Triangulate
	for(ii = 0; ii<nPpts*nProjectors; ii++ )
	{
	if(PCcorres[CameraPair[0]+ii*(nProjectors+nCams)].x > 0.0001 && PCcorres[CameraPair[1]+ii*(nProjectors+nCams)].x > 0.0001)
	;//Stereo_Triangulation(&PCcorres[CameraPair[0]+ii*(nProjectors+nCams)], &PCcorres[CameraPair[1]+ii*(nProjectors+nCams)], P1, P2+12*RTid[3], &ThreeD[ii]);
	else
	{
	ThreeD[ii].x = 0.0, ThreeD[ii].y = 0.0, ThreeD[ii].z = 0.0;
	}
	}*/
	return 0;
}
int EMatTest()
{
	int N = 500;
	double bound_2d = 5;

	double focal1 = 300, focal2 = 350, u1 = 0, v1 = 0, u2 = 1, v2 = 2;
	Point2d pp(0, 0);

	Mat rvec = (cv::Mat_<double>(3, 1) << 0.1, 0.2, 0.3);
	Mat tvec = (cv::Mat_<double>(3, 1) << 0.4, 0.5, 0.6);
	normalize(tvec, tvec);
	std::cout << "Expected rvec: " << rvec << std::endl;
	std::cout << "Expected tvec: " << tvec << std::endl;

	Mat rmat;
	Rodrigues(rvec, rmat);

	Mat K1 = (Mat_<double>(3, 3) << focal1, 0, u1, 0, focal1, v1, 0, 0, 1);
	Mat K2 = (Mat_<double>(3, 3) << focal2, 0, u2, 0, focal2, v2, 0, 0, 1);

	RNG rng;
	Mat Xs(N, 3, CV_32F);
	rng.fill(Xs, RNG::UNIFORM, -bound_2d, bound_2d);

	Mat x1s = K1 * Xs.t();
	Mat x2s = rmat * Xs.t();
	for (int j = 0; j < x2s.cols; j++) x2s.col(j) += tvec;
	x2s = K2 * x2s;

	x1s.row(0) /= x1s.row(2);
	x1s.row(1) /= x1s.row(2);
	x1s.row(2) /= x1s.row(2);

	x2s.row(0) /= x2s.row(2);
	x2s.row(1) /= x2s.row(2);
	x2s.row(2) /= x2s.row(2);

	x1s = x1s.t();
	x2s = x2s.t();

	x1s = x1s.colRange(0, 2) * 1.0;
	x2s = x2s.colRange(0, 2) * 1.0;

	double start = omp_get_wtime();
	Mat E = findEssentialMat(x1s, x2s, K1, K2, CV_RANSAC, 0.99, 1, 200, noArray());
	printf("Time: %.6fs\n", omp_get_wtime() - start);

	std::cout << "=====================================================" << std::endl;
	Mat R1_5pt, R2_5pt, tvec_5pt, rvec1_5pt, rvec2_5pt;
	decomposeEssentialMat(E, R1_5pt, R2_5pt, tvec_5pt);
	Rodrigues(R1_5pt, rvec1_5pt);
	Rodrigues(R2_5pt, rvec2_5pt);
	std::cout << "5-pt-nister rvec: " << std::endl;
	std::cout << rvec1_5pt << std::endl;
	std::cout << rvec2_5pt << std::endl;
	std::cout << "5-pt-nister tvec: " << std::endl;
	std::cout << tvec_5pt << std::endl;
	std::cout << -tvec_5pt << std::endl;


	start = omp_get_wtime();
	Mat R_5pt, rvec_5pt;
	recoverPose(E, x1s, x2s, R_5pt, tvec_5pt, K1, K2, noArray());
	printf("Time: %.6fs\n", omp_get_wtime() - start);

	Rodrigues(R_5pt, rvec_5pt);
	std::cout << "5-pt-nister rvec: " << std::endl;
	std::cout << rvec_5pt << std::endl;
	std::cout << "5-pt-nister tvec: " << std::endl;
	std::cout << tvec_5pt << std::endl;

	return 0;
}

int USAC_FindFundamentalMatrix(ConfigParamsFund cfg, vector<Point2d> pts1, vector<Point2d>pts2, double *Fmat, vector<int>&Inlier)
{
	srand((unsigned int)time(NULL));

	FundMatrixEstimator* fund = new FundMatrixEstimator;
	fund->initParamsUSAC(cfg);

	// set up the fundamental matrix estimation problem
	std::vector<double> point_data; point_data.reserve(6 * cfg.common.numDataPoints);
	for (int ii = 0; ii < cfg.common.numDataPoints; ii++)
	{
		point_data.push_back(pts1.at(ii).x), point_data.push_back(pts1.at(ii).y), point_data.push_back(1.0);
		point_data.push_back(pts2.at(ii).x), point_data.push_back(pts2.at(ii).y), point_data.push_back(1.0);
	}

	fund->initDataUSAC(cfg);
	fund->initProblem(cfg, &point_data[0]);
	if (!fund->solve())
		return 1;

	// write out results
	for (unsigned int i = 0; i < 3; ++i)
		for (unsigned int j = 0; j < 3; ++j)
			Fmat[3 * i + j] = fund->final_model_params_[3 * i + j];

	Inlier.reserve(cfg.common.numDataPoints);
	for (unsigned int i = 0; i < cfg.common.numDataPoints; ++i)
		Inlier.push_back(fund->usac_results_.inlier_flags_[i]);

	// clean up
	point_data.clear();
	//prosac_data.clear();
	fund->cleanupProblem();
	delete fund;

	return 0;
}
int USAC_FindFundamentalDriver(char *Path, int id1, int id2, int timeID)
{
	ConfigParamsFund cfg;
	bool USEPROSAC = false, USESPRT = true, USELOSAC = true;
	/// store common parameters
	cfg.common.confThreshold = 0.99;
	cfg.common.minSampleSize = 7;
	cfg.common.inlierThreshold = 1.5;
	cfg.common.maxHypotheses = 850000;
	cfg.common.maxSolutionsPerSample = 3;
	cfg.common.prevalidateSample = true;
	cfg.common.prevalidateModel = true;
	cfg.common.testDegeneracy = true;
	cfg.common.randomSamplingMethod = USACConfig::SAMP_UNIFORM;
	cfg.common.verifMethod = USACConfig::VERIF_SPRT;
	cfg.common.localOptMethod = USACConfig::LO_LOSAC;

	// read in PROSAC parameters if required
	if (USEPROSAC)
	{
		cfg.prosac.maxSamples;
		cfg.prosac.beta;
		cfg.prosac.nonRandConf;
		cfg.prosac.minStopLen;
	}

	// read in SPRT parameters if required
	if (USESPRT)
	{
		cfg.sprt.tM = 200.0;
		cfg.sprt.mS = 2.38;
		cfg.sprt.delta = 0.05;
		cfg.sprt.epsilon = 0.15;
	}

	// read in LO parameters if required
	if (USELOSAC)
	{
		cfg.losac.innerSampleSize = 15;
		cfg.losac.innerRansacRepetitions = 5;
		cfg.losac.thresholdMultiplier = 2.0;
		cfg.losac.numStepsIterative = 4;
	}
	cfg.fund.inputFilePath = Path;// "C:/temp/test1/orig_pts.txt";

	// read data from from file
	char Fname[200];

	vector<KeyPoint> Keys1, Keys2;
	if (timeID < 0)
		sprintf(Fname, "%s/K%d.dat", Path, id1);
	else
		sprintf(Fname, "%s/%d/K%d.dat", Path, id1, timeID);
	if (!ReadKPointsBinarySIFTGPU(Fname, Keys1))
		return 1;

	if (timeID < 0)
		sprintf(Fname, "%s/K%d.dat", Path, id2);
	else
		sprintf(Fname, "%s/%d/K%d.dat", Path, id2, timeID);
	if (!ReadKPointsBinarySIFTGPU(Fname, Keys2))
		return 1;

	if (timeID < 0)
		sprintf(Fname, "%s/M_%d_%d.dat", Path, id1, id2);
	else
		sprintf(Fname, "%s/M%d_%d_%d.dat", Path, timeID, id1, id2);

	int npts, pid1, pid2;
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot open %s\n", Fname);
		return 1;
	}
	fscanf(fp, "%d ", &npts);
	cfg.common.numDataPoints = npts;

	vector<Point2d>pts1, pts2;
	pts1.reserve(npts); pts2.reserve(npts);
	while (fscanf(fp, "%d %d ", &pid1, &pid2) != EOF)
	{
		pts1.push_back(Point2d(Keys1[pid1].pt.x, Keys1[pid1].pt.y));
		pts2.push_back(Point2d(Keys2[pid2].pt.x, Keys2[pid2].pt.y));
	}
	fclose(fp);

	std::vector<unsigned int> prosac_data;
	if (USEPROSAC)
	{
		prosac_data.resize(cfg.common.numDataPoints);
		if (!readPROSACDataFromFile(cfg.prosac.sortedPointsFile, cfg.common.numDataPoints, prosac_data))
			return 1;
		cfg.prosac.sortedPointIndices = &prosac_data[0];
	}
	else
		cfg.prosac.sortedPointIndices = NULL;

	double Fmat[9];
	vector<int> Inlier;
	USAC_FindFundamentalMatrix(cfg, pts1, pts2, Fmat, Inlier);

	/*sprintf(Fname, "%s/orig_pts.txt", Path); fp = fopen(Fname, "w+");
	fprintf(fp, "%d\n", cfg.common.numDataPoints);
	for (int ii = 0; ii < cfg.common.numDataPoints; ii++)
	fprintf(fp, "%.2f %.2f %.2f %.2f\n", pts1[ii].x, pts1[ii].y, pts2[ii].x, pts2[ii].y);
	fclose(fp);*/

	// write out results
	sprintf(Fname, "%s/F.txt", Path); fp = fopen(Fname, "w+");
	for (int ii = 0; ii < 9; ii++)
		fprintf(fp, "%.8f ", Fmat[ii]);
	fclose(fp);

	sprintf(Fname, "%s/inliers.txt", Path); fp = fopen(Fname, "w+");
	for (int ii = 0; ii < cfg.common.numDataPoints; ii++)
		fprintf(fp, "%d\n", Inlier[ii]);
	fclose(fp);

	return 0;
}
int USAC_FindHomography(ConfigParamsHomog cfg, vector<Point2d> pts1, vector<Point2d>pts2, double *Hmat, vector<int>&Inlier)
{
	srand((unsigned int)time(NULL));

	HomogEstimator* homog = new HomogEstimator;
	homog->initParamsUSAC(cfg);

	// set up the homography estimation problem
	std::vector<double> point_data; point_data.reserve(6 * cfg.common.numDataPoints);
	for (int ii = 0; ii < cfg.common.numDataPoints; ii++)
	{
		point_data.push_back(pts1.at(ii).x), point_data.push_back(pts1.at(ii).y), point_data.push_back(1.0);
		point_data.push_back(pts2.at(ii).x), point_data.push_back(pts2.at(ii).y), point_data.push_back(1.0);
	}

	homog->initDataUSAC(cfg);
	homog->initProblem(cfg, &point_data[0]);
	if (!homog->solve())
		return 1;

	// write out results
	for (unsigned int i = 0; i < 3; ++i)
		for (unsigned int j = 0; j < 3; ++j)
			Hmat[3 * i + j] = homog->final_model_params_[3 * i + j];

	Inlier.reserve(cfg.common.numDataPoints);
	for (unsigned int i = 0; i < cfg.common.numDataPoints; ++i)
		Inlier.push_back(homog->usac_results_.inlier_flags_[i]);

	// clean up
	point_data.clear();
	//prosac_data.clear();
	homog->cleanupProblem();
	delete homog;

	return 0;
}
int USAC_FindHomographyDriver(char *Path, int id1, int id2, int timeID)
{
	bool USEPROSAC = false, USESPRT = true, USELOSAC = true;

	ConfigParamsHomog cfg;
	/// store common parameters
	cfg.common.confThreshold = 0.99;
	cfg.common.minSampleSize = 4;
	cfg.common.inlierThreshold = 2.0;
	cfg.common.maxHypotheses = 850000;
	cfg.common.maxSolutionsPerSample = 1;
	cfg.common.prevalidateSample = true;
	cfg.common.prevalidateModel = true;
	cfg.common.testDegeneracy = true;
	cfg.common.randomSamplingMethod = USACConfig::SAMP_UNIFORM;
	cfg.common.verifMethod = USACConfig::VERIF_SPRT;
	cfg.common.localOptMethod = USACConfig::LO_LOSAC;

	// read in PROSAC parameters if required
	if (USEPROSAC)
	{
		cfg.prosac.maxSamples;
		cfg.prosac.beta;
		cfg.prosac.nonRandConf;
		cfg.prosac.minStopLen;
	}

	// read in SPRT parameters if required
	if (USESPRT)
	{
		cfg.sprt.tM = 100.0;
		cfg.sprt.mS = 1.0;
		cfg.sprt.delta = 0.01;
		cfg.sprt.epsilon = 0.2;
	}

	// read in LO parameters if required
	if (USELOSAC)
	{
		cfg.losac.innerSampleSize = 12;
		cfg.losac.innerRansacRepetitions = 3;
		cfg.losac.thresholdMultiplier = 2.0;
		cfg.losac.numStepsIterative = 4;
	}
	cfg.homog.inputFilePath = Path;// "C:/temp/test1/orig_pts.txt";

	// read data from from file
	char Fname[200];

	vector<KeyPoint> Keys1, Keys2;
	if (timeID < 0)
		sprintf(Fname, "%s/K%d.dat", Path, id1);
	else
		sprintf(Fname, "%s/%d/K%d.dat", Path, id1, timeID);
	if (!ReadKPointsBinarySIFTGPU(Fname, Keys1))
		return 1;

	if (timeID < 0)
		sprintf(Fname, "%s/K%d.dat", Path, id2);
	else
		sprintf(Fname, "%s/%d/K%d.dat", Path, id2, timeID);
	if (!ReadKPointsBinarySIFTGPU(Fname, Keys2))
		return 1;

	if (timeID < 0)
		sprintf(Fname, "%s/M_%d_%d.dat", Path, id1, id2);
	else
		sprintf(Fname, "%s/M%d_%d_%d.dat", Path, timeID, id1, id2);

	int npts, pid1, pid2;
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot open %s\n", Fname);
		return 1;
	}
	fscanf(fp, "%d ", &npts);
	cfg.common.numDataPoints = npts;

	vector<Point2d>pts1, pts2;
	pts1.reserve(npts); pts2.reserve(npts);
	while (fscanf(fp, "%d %d ", &pid1, &pid2) != EOF)
	{
		pts1.push_back(Point2d(Keys1[pid1].pt.x, Keys1[pid1].pt.y));
		pts2.push_back(Point2d(Keys2[pid2].pt.x, Keys2[pid2].pt.y));
	}
	fclose(fp);

	std::vector<unsigned int> prosac_data;
	if (USEPROSAC)
	{
		prosac_data.resize(cfg.common.numDataPoints);
		if (!readPROSACDataFromFile(cfg.prosac.sortedPointsFile, cfg.common.numDataPoints, prosac_data))
			return 1;
		cfg.prosac.sortedPointIndices = &prosac_data[0];
	}
	else
		cfg.prosac.sortedPointIndices = NULL;

	double Hmat[9];
	vector<int> Inlier;
	USAC_FindHomography(cfg, pts1, pts2, Hmat, Inlier);

	/*sprintf(Fname, "%s/orig_pts.txt", Path); fp = fopen(Fname, "w+");
	fprintf(fp, "%d\n", cfg.common.numDataPoints);
	for (int ii = 0; ii < cfg.common.numDataPoints; ii++)
	fprintf(fp, "%.2f %.2f %.2f %.2f\n", pts1[ii].x, pts1[ii].y, pts2[ii].x, pts2[ii].y);
	fclose(fp);*/

	// write out results
	sprintf(Fname, "%s/H.txt", Path); fp = fopen(Fname, "w+");
	for (int ii = 0; ii < 9; ii++)
		fprintf(fp, "%.8f ", Hmat[ii]);
	fclose(fp);

	sprintf(Fname, "%s/inliers.txt", Path); fp = fopen(Fname, "w+");
	for (int ii = 0; ii < cfg.common.numDataPoints; ii++)
		fprintf(fp, "%d\n", Inlier[ii]);
	fclose(fp);

	return 0;
}

void FishEyeUndistortionPoint(double omega, double DistCtrX, double DistCtrY, Point2d *Points, int npts)
{
	double x, y, ru, rd, x_u, y_u, t;
	for (int iPoint = 0; iPoint < npts; iPoint++)
	{
		x = Points[iPoint].x - DistCtrX, y = Points[iPoint].y - DistCtrY;
		ru = sqrt(x*x + y*y), rd = tan(ru*omega) / 2 / tan(omega / 2);
		t = rd / ru;
		x_u = t*x, y_u = t*y;
		Points[iPoint].x = x_u + DistCtrX, Points[iPoint].y = y_u + DistCtrY;
	}
}
void FishEyeUndistortionPoint(double *K, double* invK, double omega, Point2d *Points, int npts)
{
	double x_n, y_n, ru, rd, x_u, y_u, t;
	for (int iPoint = 0; iPoint < npts; iPoint++)
	{
		t = invK[6] * Points[iPoint].x + invK[7] * Points[iPoint].y + invK[8];
		x_n = (invK[0] * Points[iPoint].x + invK[1] * Points[iPoint].y + invK[2]) / t;
		y_n = (invK[3] * Points[iPoint].x + invK[4] * Points[iPoint].y + invK[5]) / t;

		ru = sqrt(x_n*x_n + y_n*y_n), rd = tan(ru*omega) / 2 / tan(omega / 2);
		t = rd / ru;
		x_u = t*x_n, y_u = t*y_n;

		t = K[6] * x_u + K[7] * y_u + K[8];
		Points[iPoint].x = (K[0] * x_u + K[1] * y_u + K[2]) / t;
		Points[iPoint].y = (K[3] * x_u + K[4] * y_u + K[5]) / t;
	}
}

void FishEyeDistortionPoint(double omega, double DistCtrX, double DistCtrY, Point2d *Points, int npts)
{
	double x, y, ru, rd, x_u, y_u, t;
	for (int iPoint = 0; iPoint < npts; iPoint++)
	{
		x = Points[iPoint].x - DistCtrX, y = Points[iPoint].y - DistCtrY;
		ru = sqrt(x*x + y*y), rd = atan(2.0*ru*tan(0.5*omega)) / omega;
		t = rd / ru;
		x_u = t*x, y_u = t*y;
		Points[iPoint].x = x_u + DistCtrX, Points[iPoint].y = y_u + DistCtrY;
	}
}
void FishEyeDistortionPoint(double *K, double* invK, double omega, Point2d *Points, int npts)
{
	double x, y, ru, rd, x_u, y_u, t;
	for (int iPoint = 0; iPoint < npts; iPoint++)
	{
		t = invK[6] * Points[iPoint].x + invK[7] * Points[iPoint].y + invK[8];
		x = (invK[0] * Points[iPoint].x + invK[1] * Points[iPoint].y + invK[2]) / t;
		y = (invK[3] * Points[iPoint].x + invK[4] * Points[iPoint].y + invK[5]) / t;

		ru = sqrt(x*x + y*y), rd = atan(2.0*ru*tan(0.5*omega)) / omega;
		t = rd / ru;
		x_u = t*x, y_u = t*y;

		t = K[6] * x_u + K[7] * y_u + K[8];
		Points[iPoint].x = (K[0] * x_u + K[1] * y_u + K[2]) / t;
		Points[iPoint].y = (K[3] * x_u + K[4] * y_u + K[5]) / t;
	}
}

void FishEyeUndistortion(unsigned char *Img, int width, int height, int nchannels, double omega, double DistCtrX, double DistCtrY, int intepAlgo, double ImgMag, double Contscale, double *Para)
{
	Contscale = 1.0 / Contscale;
	int ii, jj, kk, length = width*height, Mwidth = (int)(width*ImgMag), Mheight = (int)(height*ImgMag), Mlength = Mwidth*Mheight;
	bool createMem = false;
	if (Para == NULL)
	{
		createMem = true;
		Para = new double[length*nchannels];

		for (kk = 0; kk < nchannels; kk++)
			Generate_Para_Spline(Img + kk*length, Para + kk*length, width, height, intepAlgo);
	}


	double S[3];
	Point2d ImgPt;
	double H[9] = { Contscale, 0, width / 2 - Mwidth / 2 * Contscale, 0, Contscale, height / 2 - Mheight / 2 * Contscale, 0, 0, 1 };
	for (jj = 0; jj < Mheight; jj++)
	{
		for (ii = 0; ii < Mwidth; ii++)
		{
			ImgPt.x = H[0] * ii + H[1] * jj + H[2], ImgPt.y = H[3] * ii + H[4] * jj + H[5];
			FishEyeDistortionPoint(omega, DistCtrX, DistCtrY, &ImgPt, 1);
			if (ImgPt.x < 0 || ImgPt.x > width - 1 || ImgPt.y<0.0 || ImgPt.y > height - 1)
			{
				for (kk = 0; kk < nchannels; kk++)
					Img[ii + jj*Mwidth + kk*Mlength] = (unsigned char)0;
			}
			else
			{
				for (kk = 0; kk < nchannels; kk++)
				{
					Get_Value_Spline(Para + kk*length, width, height, ImgPt.x, ImgPt.y, S, -1, intepAlgo);
					S[0] = min(max(S[0], 0.0), 255.0);
					Img[ii + jj*Mwidth + kk*Mlength] = (unsigned char)MyFtoI(S[0]);
				}
			}
		}
	}

	if (createMem)
		delete[]Para;

	return;
}
void FishEyeUndistortion(unsigned char *Img, int width, int height, int nchannels, double *K, double* invK, double omega, int intepAlgo, double ImgMag, double Contscale, double *Para)
{
	Contscale = 1.0 / Contscale;
	int ii, jj, kk, length = width*height, Mwidth = width*ImgMag, Mheight = height*ImgMag, Mlength = Mwidth*Mheight;
	bool createMem = false;
	if (Para == NULL)
	{
		createMem = true;
		Para = new double[length*nchannels];

		for (kk = 0; kk < nchannels; kk++)
			Generate_Para_Spline(Img + kk*length, Para + kk*length, width, height, intepAlgo);
	}


	double S[3];
	Point2d ImgPt;
	double H[9] = { Contscale, 0, width / 2 - Mwidth / 2 * Contscale, 0, Contscale, height / 2 - Mheight / 2 * Contscale, 0, 0, 1 };
	for (jj = 0; jj < Mheight; jj++)
	{
		for (ii = 0; ii < Mwidth; ii++)
		{
			ImgPt.x = H[0] * ii + H[1] * jj + H[2], ImgPt.y = H[3] * ii + H[4] * jj + H[5];
			FishEyeDistortionPoint(K, invK, omega, &ImgPt, 1);
			if (ImgPt.x < 0 || ImgPt.x > width - 1 || ImgPt.y<0.0 || ImgPt.y > height - 1)
			{
				for (kk = 0; kk < nchannels; kk++)
					Img[ii + jj*Mwidth + kk*Mlength] = (unsigned char)0;
			}
			else
			{
				for (kk = 0; kk < nchannels; kk++)
				{
					Get_Value_Spline(Para + kk*length, width, height, ImgPt.x, ImgPt.y, S, -1, intepAlgo);
					S[0] = min(max(S[0], 0.0), 255.0);
					Img[ii + jj*Mwidth + kk*Mlength] = (unsigned char)MyFtoI(S[0]);
				}
			}
		}
	}

	if (createMem)
		delete[]Para;

	return;
}

//The camera is parameterized using 16 parameters: 2 for  focal length, 1 for skew, 2 for principal point, 3 for radial distortion, 2 for tangential distortion, 2 for prism, and 6 for rotation translation.
void LensDistortionPoint(Point2d *img_point, double *camera, double *distortion, int npts)
{
	double alpha = camera[0], beta = camera[4], gamma = camera[1], u0 = camera[2], v0 = camera[5];

	for (int ii = 0; ii < npts; ii++)
	{
		double ycn = (img_point[ii].y - v0) / beta;
		double xcn = (img_point[ii].x - u0 - gamma*ycn) / alpha;

		double r2 = xcn*xcn + ycn*ycn, r4 = r2*r2, r6 = r2*r4, X2 = xcn*xcn, Y2 = ycn*ycn, XY = xcn*ycn;

		double a0 = distortion[0], a1 = distortion[1], a2 = distortion[2];
		double p0 = distortion[3], p1 = distortion[4];
		double s0 = distortion[5], s1 = distortion[6];

		double radial = 1 + a0*r2 + a1*r4 + a2*r6;
		double tangential_x = p0*(r2 + 2.0*X2) + 2.0*p1*XY;
		double tangential_y = p1*(r2 + 2.0*Y2) + 2.0*p0*XY;
		double prism_x = s0*r2;
		double prism_y = s1*r2;

		double xcn_ = radial*xcn + tangential_x + prism_x;
		double ycn_ = radial*ycn + tangential_y + prism_y;

		img_point[ii].x = alpha*xcn_ + gamma*ycn_ + u0;
		img_point[ii].y = beta*ycn_ + v0;
	}

	return;
}
void CC_Calculate_xcn_ycn_from_i_j(double i, double j, double &xcn, double &ycn, double *A, double *distortion, int Method)
{
	int k;
	double Xcn, Ycn, r2, r4, r6, x2, y2, xy, x0, y0, t_x, t_y;
	double radial, tangential_x, tangential_y, prism_x, prism_y;
	double a0 = distortion[0], a1 = distortion[1], a2 = distortion[2];
	double p0 = distortion[3], p1 = distortion[4];
	double s0 = distortion[5], s1 = distortion[6];

	Ycn = (j - A[4]) / A[1];
	Xcn = (i - A[3] - A[2] * Ycn) / A[0];

	xcn = Xcn;
	ycn = Ycn;
	for (k = 0; k < 20; k++)
	{
		x0 = xcn;
		y0 = ycn;
		r2 = xcn*xcn + ycn*ycn, r4 = r2*r2, r6 = r2*r4, x2 = xcn*xcn, y2 = ycn*ycn, xy = xcn*ycn;

		radial = 1.0 + a0*r2 + a1*r4 + a2*r6;
		tangential_x = p0*(r2 + 2.0*x2) + 2.0*p1*xy;
		tangential_y = p1*(r2 + 2.0*y2) + 2.0*p0*xy;
		prism_x = s0*r2;
		prism_y = s1*r2;

		if (Method == 0)
		{
			xcn = (Xcn - tangential_x - prism_x) / radial;
			ycn = (Ycn - tangential_y - prism_y) / radial;
			t_x = xcn - x0;
			t_y = ycn - y0;
		}

		if (fabs(t_x) < fabs(xcn*1e-16) && fabs(t_y) < fabs(ycn*1e-16))
			break;
	}
	return;
}
void LensCorrectionPoint(Point2d *uv, double *camera, double *distortion, int npts)
{
	double xcn, ycn, A[] = { camera[0], camera[4], camera[1], camera[2], camera[5] };

	for (int ii = 0; ii < npts; ii++)
	{
		CC_Calculate_xcn_ycn_from_i_j(uv[ii].x, uv[ii].y, xcn, ycn, A, distortion, 0);

		uv[ii].x = A[0] * xcn + A[2] * ycn + A[3];
		uv[ii].y = A[1] * ycn + A[4];
	}

	return;
}
void LensUndistortion(unsigned char *Img, int width, int height, int nchannels, double *K, double *distortion, int intepAlgo, double ImgMag, double Contscale, double *Para)
{
	Contscale = 1.0 / Contscale;
	int ii, jj, kk, length = width*height, Mwidth = (int)(width*ImgMag), Mheight = (int)(height*ImgMag), Mlength = Mwidth*Mheight;
	bool createMem = false;
	if (Para == NULL)
	{
		createMem = true;
		Para = new double[length*nchannels];

		for (kk = 0; kk < nchannels; kk++)
			Generate_Para_Spline(Img + kk*length, Para + kk*length, width, height, intepAlgo);
	}

	double S[3];
	Point2d ImgPt;
	double H[9] = { Contscale, 0, width / 2 - Mwidth / 2 * Contscale, 0, Contscale, height / 2 - Mheight / 2 * Contscale, 0, 0, 1 };
	for (jj = 0; jj < Mheight; jj++)
	{
		for (ii = 0; ii < Mwidth; ii++)
		{
			ImgPt.x = H[0] * ii + H[1] * jj + H[2], ImgPt.y = H[3] * ii + H[4] * jj + H[5];
			LensDistortionPoint(&ImgPt, K, distortion, 1);

			if (ImgPt.x < 0 || ImgPt.x > width - 1 || ImgPt.y<0.0 || ImgPt.y > height - 1)
			{
				for (kk = 0; kk < nchannels; kk++)
					Img[ii + jj*Mwidth + kk*Mlength] = (unsigned char)0;
			}
			else
			{
				for (kk = 0; kk < nchannels; kk++)
				{
					Get_Value_Spline(Para + kk*length, width, height, ImgPt.x, ImgPt.y, S, -1, intepAlgo);
					S[0] = min(max(S[0], 0.0), 255.0);
					Img[ii + jj*Mwidth + kk*Mlength] = (unsigned char)MyFtoI(S[0]);
				}
			}
		}
	}

	if (createMem)
		delete[]Para;

	return;
}

int EssentialMatOutliersRemove(char *Path, int timeID, int id1, int id2, int nCams, int cameraToScan, int ninlierThresh, bool distortionCorrected, bool needDuplicateRemove)
{
	CameraData *camera = new CameraData[nCams];
	if (ReadIntrinsicResults(Path, camera, nCams) != 0)
		return 1;

	if (distortionCorrected)
		for (int ii = 0; ii < nCams; ii++)
			for (int jj = 0; jj < 7; jj++)
				camera[ii].distortion[jj] = 0.0;

	for (int ii = 0; ii < nCams; ii++)
		camera[ii].LensModel = RADIAL_TANGENTIAL_PRISM, camera[ii].threshold = 2.0, camera[ii].ninlierThresh = 50;
	//printf("Load camera intriniscs for Essential matrix outlier removal.\n");

	char Fname[200];
	vector<Point2i> RawPairWiseMatchID;
	if (timeID < 0)
		sprintf(Fname, "%s/M_%d_%d.dat", Path, id1, id2);
	else
		sprintf(Fname, "%s/M%d_%d_%d.dat", Path, timeID, id1, id2);

	int pid1, pid2, npts;
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
		return 1;
	fscanf(fp, "%d ", &npts);
	RawPairWiseMatchID.reserve(npts);
	while (fscanf(fp, "%d %d ", &pid1, &pid2) != EOF)
		RawPairWiseMatchID.push_back(Point2i(pid1, pid2));
	fclose(fp);

	if (npts < 40)
		return 1;

	vector<KeyPoint> Keys1, Keys2;
	if (timeID < 0)
		sprintf(Fname, "%s/K%d.dat", Path, id1);
	else
		sprintf(Fname, "%s/%d/K%d.dat", Path, id1, timeID);

	if (!ReadKPointsBinarySIFTGPU(Fname, Keys1))
		return 1;

	if (timeID < 0)
		sprintf(Fname, "%s/K%d.dat", Path, id2);
	else
		sprintf(Fname, "%s/%d/K%d.dat", Path, id2, timeID);
	if (!ReadKPointsBinarySIFTGPU(Fname, Keys2))
		return 1;

	if (needDuplicateRemove)
	{
		int SortingVec[20000], tId[20000]; //should be more than enough
		vector<Point2i> SRawPairWiseMatchID; SRawPairWiseMatchID.reserve(RawPairWiseMatchID.size());

		//To remove the nonsense case of every point matchces to 1 point-->IT HAPPENED
		SRawPairWiseMatchID.push_back(RawPairWiseMatchID.at(0));
		for (int i = 1; i < min(npts, 20000); i++)
			if (RawPairWiseMatchID.at(i).x != RawPairWiseMatchID.at(i - 1).x)
				SRawPairWiseMatchID.push_back(RawPairWiseMatchID.at(i));

		if (SRawPairWiseMatchID.size() < ninlierThresh)
			return 1;

		//Start sorting
		int nsPairwiseMatchID = SRawPairWiseMatchID.size();
		for (int i = 0; i < min(nsPairwiseMatchID, 20000); i++)
		{
			SortingVec[i] = SRawPairWiseMatchID.at(i).x;
			tId[i] = i;
		}
		Quick_Sort_Int(SortingVec, tId, 0, min(nsPairwiseMatchID, 20000) - 1);

		//Store sorted vector
		RawPairWiseMatchID.push_back(SRawPairWiseMatchID.at(tId[0]));
		for (unsigned int i = 1; i < min(nsPairwiseMatchID, 20000); i++)
			if (SortingVec[i] != SortingVec[i - 1])
				RawPairWiseMatchID.push_back(SRawPairWiseMatchID.at(tId[i]));

		npts = RawPairWiseMatchID.size();
		if (npts < ninlierThresh)
			return 1;

		if (timeID < 0)
			sprintf(Fname, "%s/M_%d_%d.dat", Path, id1, id2);
		else
			sprintf(Fname, "%s/M%d_%d_%d.dat", Path, timeID, id1, id2);
		fp = fopen(Fname, "w+");
		fprintf(fp, "%d\n", npts);
		for (int ii = 0; ii < npts; ii++)
			fprintf(fp, "%d %d\n", RawPairWiseMatchID[ii].x, RawPairWiseMatchID[ii].y);
		fclose(fp);
	}

	Point2d *pts1 = new Point2d[npts], *pts2 = new Point2d[npts];
	for (int ii = 0; ii < npts; ii++)
	{
		int id1 = RawPairWiseMatchID[ii].x, id2 = RawPairWiseMatchID[ii].y;
		pts1[ii].x = Keys1.at(id1).pt.x, pts1[ii].y = Keys1.at(id1).pt.y;
		pts2[ii].x = Keys2.at(id2).pt.x, pts2[ii].y = Keys2.at(id2).pt.y;
	}

	if (cameraToScan != -1)
	{
		if (!distortionCorrected && camera[cameraToScan].LensModel == RADIAL_TANGENTIAL_PRISM)
		{
			LensCorrectionPoint(pts1, camera[cameraToScan].K, camera[cameraToScan].distortion, npts);
			LensCorrectionPoint(pts2, camera[cameraToScan].K, camera[cameraToScan].distortion, npts);
		}
	}
	else
	{
		if (!distortionCorrected && camera[id1].LensModel == RADIAL_TANGENTIAL_PRISM)
			LensCorrectionPoint(pts1, camera[id1].K, camera[id1].distortion, npts);
		if (!distortionCorrected && camera[id2].LensModel == RADIAL_TANGENTIAL_PRISM)
			LensCorrectionPoint(pts2, camera[id2].K, camera[id2].distortion, npts);
	}

	Mat x1s(npts, 2, CV_64F), x2s(npts, 2, CV_64F);
	for (int ii = 0; ii < npts; ii++)
	{
		x1s.at<double>(ii, 0) = pts1[ii].x, x1s.at<double>(ii, 1) = pts1[ii].y;
		x2s.at<double>(ii, 0) = pts2[ii].x, x2s.at<double>(ii, 1) = pts2[ii].y;
	}

	double start = omp_get_wtime();
	Mat Inliers, E;
	if (cameraToScan != -1)
	{
		Mat cvK1 = Mat(3, 3, CV_64F, camera[cameraToScan].K);
		Mat cvK2 = Mat(3, 3, CV_64F, camera[cameraToScan].K);
		E = findEssentialMat(x1s, x2s, cvK1, cvK2, CV_RANSAC, 0.99, 1, 200, Inliers);
	}
	else
	{
		Mat cvK1 = Mat(3, 3, CV_64F, camera[id1].K);
		Mat cvK2 = Mat(3, 3, CV_64F, camera[id2].K);
		E = findEssentialMat(x1s, x2s, cvK1, cvK2, CV_RANSAC, 0.99, 1, 200, Inliers);
	}

	int ninliers = 0;
	//fp = fopen("C:/temp/inliers.txt", "w+");
	for (int ii = 0; ii < Inliers.cols; ii++)
	{
		if (Inliers.at<bool>(ii))
		{
			ninliers++;
			//fprintf(fp, "%d\n", ii);
		}
	}
	//fclose(fp);
	//if (ninliers > 40)
	//	printf("Essential matrix succeeds....%d inliers\n", ninliers);
	//else
	//	printf("Essential matrix fails....%d inliers\n", ninliers);

	if (timeID < 0)
		sprintf(Fname, "%s/_M_%d_%d.dat", Path, id1, id2);
	else
		sprintf(Fname, "%s/M%d_%d_%d.dat", Path, timeID, id1, id2);
	fp = fopen(Fname, "w+");
	fprintf(fp, "%d\n", ninliers);
	for (int ii = 0; ii < Inliers.cols; ii++)
		if (Inliers.at<bool>(ii))
			fprintf(fp, "%d %d\n", RawPairWiseMatchID[ii].x, RawPairWiseMatchID[ii].y);
	fclose(fp);

#pragma omp critical
	printf("View (%d, %d) of frame %d...%d matches... %.2fs\n", id1, id2, timeID, ninliers, omp_get_wtime() - start);
	return 0;
}
static void flannFindPairs(const CvSeq*objectKpts, const CvSeq* objectDescriptors, const CvSeq*imageKpts, const CvSeq* imageDescriptors, vector<int>& ptpairs)
{
	int length = (int)(objectDescriptors->elem_size / sizeof(float));

	cv::Mat m_object(objectDescriptors->total, length, CV_32F);
	cv::Mat m_image(imageDescriptors->total, length, CV_32F);


	// copy descriptors
	CvSeqReader obj_reader;
	float* obj_ptr = m_object.ptr<float>(0);
	cvStartReadSeq(objectDescriptors, &obj_reader);
	for (int i = 0; i < objectDescriptors->total; i++)
	{
		const float* descriptor = (const float*)obj_reader.ptr;
		CV_NEXT_SEQ_ELEM(obj_reader.seq->elem_size, obj_reader);
		memcpy(obj_ptr, descriptor, length*sizeof(float));
		obj_ptr += length;
	}
	CvSeqReader img_reader;
	float* img_ptr = m_image.ptr<float>(0);
	cvStartReadSeq(imageDescriptors, &img_reader);
	for (int i = 0; i < imageDescriptors->total; i++)
	{
		const float* descriptor = (const float*)img_reader.ptr;
		CV_NEXT_SEQ_ELEM(img_reader.seq->elem_size, img_reader);
		memcpy(img_ptr, descriptor, length*sizeof(float));
		img_ptr += length;
	}

	// find nearest neighbors using FLANN
	cv::Mat m_indices(objectDescriptors->total, 2, CV_32S);
	cv::Mat m_dists(objectDescriptors->total, 2, CV_32F);
	cv::flann::Index flann_index(m_image, cv::flann::KDTreeIndexParams(4));  // using 4 randomized kdtrees
	flann_index.knnSearch(m_object, m_indices, m_dists, 2, cv::flann::SearchParams(64)); // maximum number of leafs checked

	int* indices_ptr = m_indices.ptr<int>(0);
	float* dists_ptr = m_dists.ptr<float>(0);
	for (int i = 0; i < m_indices.rows; ++i) {
		if (dists_ptr[2 * i] < 0.6*dists_ptr[2 * i + 1]) {
			ptpairs.push_back(i);
			ptpairs.push_back(indices_ptr[2 * i]);
		}
	}
}
int GeneratePointsCorrespondenceMatrix(char *Path, int nviews, int timeID)
{
	char Fname[200];
	FILE *fp = 0;
	int ii, jj;

	bool BinaryDesc = false;

	int minHessian = 400, descriptorSize = 128;
	SiftFeatureDetector detector(MAXSIFTPTS);
	SiftDescriptorExtractor extractor;

	Mat img, imgGray, equalizedImg;
	vector<int> cumulativePts;

	omp_set_num_threads(omp_get_max_threads());

	int *PtsPerView = new int[nviews];
	double start = omp_get_wtime();
#pragma omp parallel for
	for (int ii = 0; ii < nviews; ii++)
	{
		char Fname[200];
		if (timeID < 0)
			sprintf(Fname, "%s/S/%d.png", Path, ii);
		else
			sprintf(Fname, "%s/%d/%d.png", Path, ii, timeID);
		Mat img = imread(Fname, CV_LOAD_IMAGE_COLOR);
		if (img.empty())
		{
			printf("Can't read %s\n", Fname);
			continue;
		}

		//cvtColor(img, imgGray, CV_BGR2GRAY);
		////equalizeHist(imgGray, equalizedImg);

		double start = omp_get_wtime();

		vector<KeyPoint> keypoints; keypoints.reserve(MAXSIFTPTS);
		Mat descriptors(MAXSIFTPTS, descriptorSize, CV_32F);
		detector.detect(img, keypoints);
		extractor.compute(img, keypoints, descriptors);

#pragma omp critical
		{
			if (timeID < 0)
			{
				sprintf(Fname, "%s/K%d.dat", Path, ii), WriteKPointsBinary(Fname, keypoints, false);
				sprintf(Fname, "%s/D%d.dat", Path, ii), WriteDescriptorBinary(Fname, descriptors, false);
			}
			else
			{
				sprintf(Fname, "%s/%d/K%d.dat", Path, ii, timeID), WriteKPointsBinary(Fname, keypoints, false);
				sprintf(Fname, "%s/%d/D%d.dat", Path, ii, timeID), WriteDescriptorBinary(Fname, descriptors, false);
			}
			printf("Obtain %d points for view %d frame %d  ... wrote to files. Take %.2fs\n", keypoints.size(), ii + 1, timeID, omp_get_wtime() - start);
		}

		PtsPerView[ii] = keypoints.size();
	}
	printf("Finished extracting feature points ... in %.2fs\n", omp_get_wtime() - start);

	int totalPts = 0;
	for (int ii = 0; ii < nviews; ii++)
	{
		cumulativePts.push_back(totalPts);
		totalPts += PtsPerView[ii];
	}
	cumulativePts.push_back(totalPts);

	if (timeID < 0)
		sprintf(Fname, "%s/CumlativePoints.txt", Path);
	else
		sprintf(Fname, "%s/CumlativePoints_%d.txt", Path, timeID);
	fp = fopen(Fname, "w+");
	for (ii = 0; ii < cumulativePts.size(); ii++)
		fprintf(fp, "%d\n", cumulativePts.at(ii));
	fclose(fp);

	// NEAREST NEIGHBOR MATCHING USING FLANN LIBRARY :  match descriptor2 to descriptor1
	vector<int> *MatchingMatrix = new vector<int>[totalPts];

	bool useBFMatcher = false; // SET TO TRUE TO USE BRUTE FORCE MATCHER
	const int knn = 2, ntrees = 4, maxLeafCheck = 128;
	const float nndrRatio = 0.6f;

	start = omp_get_wtime();
	printf("Running feature matching...\n");
	Mat descriptors1;
	for (int jj = 0; jj < nviews - 1; jj++)
	{
		if (timeID < 0)
			sprintf(Fname, "%s/D%d.dat", Path, jj);
		else
			sprintf(Fname, "%s/%d/D%d.dat", Path, jj, timeID);
		descriptors1 = ReadDescriptorBinary(Fname, descriptorSize);
		if (descriptors1.empty())
			continue;

#pragma omp parallel for
		for (int ii = jj + 1; ii < nviews; ii++)
		{
			char Fname[200];
			if (timeID < 0)
				sprintf(Fname, "%s/D%d.dat", Path, ii);
			else
				sprintf(Fname, "%s/%d/D%d.dat", Path, ii, timeID);
			Mat descriptors2 = ReadDescriptorBinary(Fname, descriptorSize);
			if (descriptors2.empty())
				continue;

			double start = omp_get_wtime();
			//Finding nearest neighbor
			Mat indices, dists;
			vector<vector<DMatch> > matches;
			if (BinaryDesc)
			{
				//printf("Binary descriptors detected...\n");// ORB, Brief, BRISK, FREAK
				if (useBFMatcher)
				{
					cv::BFMatcher matcher(cv::NORM_HAMMING); // use cv::NORM_HAMMING2 for ORB descriptor with WTA_K == 3 or 4 (see ORB constructor)
					matcher.knnMatch(descriptors2, descriptors1, matches, knn);
				}
				else
				{
					// Create Flann LSH index
					cv::flann::Index flannIndex(descriptors1, cv::flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);
					flannIndex.knnSearch(descriptors2, indices, dists, knn, cv::flann::SearchParams());
				}
			}
			else
			{
				if (useBFMatcher)
				{
					cv::BFMatcher matcher(cv::NORM_L2);
					matcher.knnMatch(descriptors2, descriptors1, matches, knn);
				}
				else
				{
					// Create Flann KDTree index
					cv::flann::Index flannIndex(descriptors1, cv::flann::KDTreeIndexParams(ntrees));//, cvflann::FLANN_DIST_EUCLIDEAN);
					flannIndex.knnSearch(descriptors2, indices, dists, knn, cv::flann::SearchParams(maxLeafCheck));
				}
			}

			// Find correspondences by NNDR (Nearest Neighbor Distance Ratio)
			int count = 0;
			if (!useBFMatcher)
			{
				for (int i = 0; i < descriptors2.rows; ++i)
				{
					//printf("q=%d dist1=%f dist2=%f\n", i, dists.at<float>(i,0), dists.at<float>(i,1));
					int ind1 = indices.at<int>(i, 0);
					if (indices.at<int>(i, 0) >= 0 && indices.at<int>(i, 1) >= 0 && dists.at<float>(i, 0) <= nndrRatio * dists.at<float>(i, 1))
					{
						MatchingMatrix[cumulativePts.at(jj) + ind1].push_back(cumulativePts.at(ii) + i);
						count++;
					}
				}
			}
			else
			{
				for (unsigned int i = 0; i < matches.size(); ++i)
				{
					//printf("q=%d dist1=%f dist2=%f\n", matches.at(i).at(0).queryIdx, matches.at(i).at(0).distance, matches.at(i).at(1).distance);
					if (matches.at(i).size() == 2 && matches.at(i).at(0).distance <= nndrRatio * matches.at(i).at(1).distance)
					{
						MatchingMatrix[cumulativePts.at(jj) + matches.at(i).at(0).trainIdx].push_back(cumulativePts.at(ii) + i);
						count++;
					}
				}
			}
#pragma omp critical
			printf("Matching view %d to view %d of frame %d has %d points in %.2fs.\n", jj + 1, ii + 1, timeID, count, omp_get_wtime() - start);
		}
	}
	printf("Finished matching feature points ... in %.2fs\n", omp_get_wtime() - start);

	if (timeID < 0)
		sprintf(Fname, "%s/PM.txt", Path);
	else
		sprintf(Fname, "%s/PM_%d.txt", Path, timeID);
	fp = fopen(Fname, "w+");
	if (fp != NULL)
	{
		for (jj = 0; jj < totalPts; jj++)
		{
			int nmatches = MatchingMatrix[jj].size();
			fprintf(fp, "%d ", nmatches);
			sort(MatchingMatrix[jj].begin(), MatchingMatrix[jj].end());
			for (ii = 0; ii < nmatches; ii++)
				fprintf(fp, "%d ", MatchingMatrix[jj].at(ii));
			fprintf(fp, "\n");
		}
		fclose(fp);
	}
	printf("Finished generateing point correspondence matrix\n");

	delete[]MatchingMatrix;

	return 0;
}
int ExtractSiftGPUfromExtractedFrames(char *Path, vector<int> nviews, int startF, int nframes)
{
	// Allocation size to the largest width and largest height 1920x1080
	// Maximum working dimension. All the SIFT octaves that needs a larger texture size will be skipped. maxd = 2560 <-> 768MB of graphic memory. 
	char * argv[] = { "-fo", "-1", "-v", "0", "-p", "1920x1080", "-maxd", "4096" };
	//-fo -1    staring from -1 octave 
	//-v 1      only print out # feature and overall time
	//-loweo    add a (.5, .5) offset
	//-tc <num> set a soft limit to number of detected features
	//-m,       up to 2 orientations for each feature (change to single orientation by using -m 1)
	//-s        enable subpixel subscale (disable by using -s 0)
	//"-cuda", "[device_id]"  : cuda implementation (fastest for smaller images). CUDA-implementation allows you to create multiple instances for multiple threads. Checkout src\TestWin\MultiThreadSIFT
	// "-Display", "display_name" (for OPENGL) to select monitor/GPU (XLIB/GLUT) on windows the display name would be something like \\.\DISPLAY4
	//Only the following parameters can be changed after initialization (by calling ParseParam):-dw, -ofix, -ofix-not, -fo, -unn, -maxd, -b
	//to change other parameters at runtime, you need to first unload the dynamically loaded libaray reload the libarary, then create a new siftgpu instance

	//Init SiftGPU: START
#ifdef _WIN32
#ifdef _DEBUG
	HMODULE  hsiftgpu = LoadLibrary("siftgpu_d.dll");
#else
	HMODULE  hsiftgpu = LoadLibrary("siftgpu.dll");
#endif
#else
	void * hsiftgpu = dlopen("libsiftgpu.so", RTLD_LAZY);
#endif

	if (hsiftgpu == NULL)
		return 0;

	SiftGPU* (*pCreateNewSiftGPU)(int) = NULL;
	SiftMatchGPU* (*pCreateNewSiftMatchGPU)(int) = NULL;
	pCreateNewSiftGPU = (SiftGPU* (*) (int)) GET_MYPROC(hsiftgpu, "CreateNewSiftGPU");
	pCreateNewSiftMatchGPU = (SiftMatchGPU* (*)(int)) GET_MYPROC(hsiftgpu, "CreateNewSiftMatchGPU");
	SiftGPU* sift = pCreateNewSiftGPU(1);

	int argc = sizeof(argv) / sizeof(char*);
	sift->ParseParam(argc, argv);
	if (sift->CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED)
		return 0;
	//Init SiftGPU: END

	//SIFT DECTION: START
	int numKeys, descriptorSize = SIFTBINS;
	vector<float > descriptors; descriptors.reserve(MAXSIFTPTS * descriptorSize);
	vector<SiftGPU::SiftKeypoint> keys; keys.reserve(MAXSIFTPTS);

	char Fname[200];
	double start = omp_get_wtime();
	for (int timeID = startF; timeID < startF + nframes; timeID++)
	{
		for (int ii = 0; ii < nviews.size(); ii++)
		{
			int viewID = nviews[ii];
			keys.clear(), descriptors.clear();
			double start = omp_get_wtime();

			sprintf(Fname, "%s/%d/%d.png", Path, viewID, timeID);
			if (sift->RunSIFT(Fname)) //You can have at most one OpenGL-based SiftGPU (per process)--> no omp can be used
			{
				numKeys = sift->GetFeatureNum();
				keys.resize(numKeys);    descriptors.resize(descriptorSize * numKeys);
				sift->GetFeatureVector(&keys[0], &descriptors[0]);

				sprintf(Fname, "%s/%d/K%d.dat", Path, viewID, timeID); WriteKPointsBinarySIFTGPU(Fname, keys);
				sprintf(Fname, "%s/%d/D%d.dat", Path, viewID, timeID); WriteDescriptorBinarySIFTGPU(Fname, descriptors);
				printf("View (%d, %d): %d points ... Wrote to files. Take %.2fs\n", viewID, timeID, numKeys, omp_get_wtime() - start);
			}
			else
				printf("Cannot load %s", Fname);
		}
	}
	printf("Total time: %.2fs\n", omp_get_wtime() - start);

	return 0;
}

//Use GPU for brute force Sift matching
int GeneratePointsCorrespondenceMatrix_SiftGPU1(char *Path, int nviews, int timeID, float nndrRatio, bool distortionCorrected, int OulierRemoveTestMethod, int cameraToScan)
{

	// Allocation size to the largest width and largest height 1920x1080
	// Maximum working dimension. All the SIFT octaves that needs a larger texture size will be skipped. maxd = 2560 <-> 768MB of graphic memory. 
	char * argv[] = { "-fo", "-1", "-v", "0", "-p", "1920x1080", "-maxd", "3200" };
	//-fo -1    staring from -1 octave 
	//-v 1      only print out # feature and overall time
	//-loweo    add a (.5, .5) offset
	//-tc <num> set a soft limit to number of detected features
	//-m,       up to 2 orientations for each feature (change to single orientation by using -m 1)
	//-s        enable subpixel subscale (disable by using -s 0)
	//"-cuda", "[device_id]"  : cuda implementation (fastest for smaller images). CUDA-implementation allows you to create multiple instances for multiple threads. Checkout src\TestWin\MultiThreadSIFT
	// "-Display", "display_name" (for OPENGL) to select monitor/GPU (XLIB/GLUT) on windows the display name would be something like \\.\DISPLAY4
	//Only the following parameters can be changed after initialization (by calling ParseParam):-dw, -ofix, -ofix-not, -fo, -unn, -maxd, -b
	//to change other parameters at runtime, you need to first unload the dynamically loaded libaray reload the libarary, then create a new siftgpu instance

	//Init SiftGPU: START
#ifdef _WIN32
#ifdef _DEBUG
	HMODULE  hsiftgpu = LoadLibrary("siftgpu_d.dll");
#else
	HMODULE  hsiftgpu = LoadLibrary("siftgpu.dll");
#endif
#else
	void * hsiftgpu = dlopen("libsiftgpu.so", RTLD_LAZY);
#endif

	if (hsiftgpu == NULL)
		return 0;

	SiftGPU* (*pCreateNewSiftGPU)(int) = NULL;
	SiftMatchGPU* (*pCreateNewSiftMatchGPU)(int) = NULL;
	pCreateNewSiftGPU = (SiftGPU* (*) (int)) GET_MYPROC(hsiftgpu, "CreateNewSiftGPU");
	pCreateNewSiftMatchGPU = (SiftMatchGPU* (*)(int)) GET_MYPROC(hsiftgpu, "CreateNewSiftMatchGPU");
	SiftGPU* sift = pCreateNewSiftGPU(1);

	int argc = sizeof(argv) / sizeof(char*);
	sift->ParseParam(argc, argv);
	if (sift->CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED)
		return 0;
	//Init SiftGPU: END

	//SIFT DECTION: START
	int numKeys, descriptorSize = SIFTBINS;
	vector<float > descriptors; descriptors.reserve(MAXSIFTPTS * descriptorSize);
	vector<SiftGPU::SiftKeypoint> keys; keys.reserve(MAXSIFTPTS);

	vector<int>cumulativePts;
	vector<int>PtsPerView;

	int totalPts = 0;
	char Fname[200];

	double start;
	for (int ii = 0; ii < nviews; ii++)
	{
		keys.clear(), descriptors.clear();
		start = omp_get_wtime();

		//Try to read all sift points if available
		if (timeID < 0)
			sprintf(Fname, "%s/K%d.dat", Path, ii);
		else
			sprintf(Fname, "%s/%d/K%d.dat", Path, ii, timeID);
		if (ReadKPointsBinarySIFTGPU(Fname, keys))
		{
			cumulativePts.push_back(totalPts);
			totalPts += keys.size();
			PtsPerView.push_back(keys.size());
			continue; //Sift availble, move one
		}

		if (timeID < 0)
			sprintf(Fname, "%s/%d.png", Path, ii);
		else
			sprintf(Fname, "%s/%d/%d.png", Path, ii, timeID);
		if (sift->RunSIFT(Fname)) //You can have at most one OpenGL-based SiftGPU (per process)--> no omp can be used
		{
			numKeys = sift->GetFeatureNum();
			keys.resize(numKeys);    descriptors.resize(descriptorSize * numKeys);
			sift->GetFeatureVector(&keys[0], &descriptors[0]);

			if (timeID < 0)
			{
				sprintf(Fname, "%s/K%d.dat", Path, ii); WriteKPointsBinarySIFTGPU(Fname, keys);
				sprintf(Fname, "%s/D%d.dat", Path, ii); WriteDescriptorBinarySIFTGPU(Fname, descriptors);
			}
			else
			{
				sprintf(Fname, "%s/%d/K%d.dat", Path, ii, timeID); WriteKPointsBinarySIFTGPU(Fname, keys);
				sprintf(Fname, "%s/%d/D%d.dat", Path, ii, timeID); WriteDescriptorBinarySIFTGPU(Fname, descriptors);
			}

			printf("View %d: %d points ... Wrote to files. Take %.2fs\n", ii, numKeys, omp_get_wtime() - start);

			cumulativePts.push_back(totalPts);
			totalPts += numKeys;
			PtsPerView.push_back(numKeys);
		}
		else
			printf("Cannot load %s", Fname);
	}
	cumulativePts.push_back(totalPts);

	if (timeID < 0)
		sprintf(Fname, "%s/CumlativePoints.txt", Path);
	else
		sprintf(Fname, "%s/CumlativePoints_%d.txt", Path, timeID);
	FILE* fp = fopen(Fname, "w+");
	for (int ii = 0; ii < cumulativePts.size(); ii++)
		fprintf(fp, "%d\n", cumulativePts.at(ii));
	fclose(fp);
	//SIFT DECTION: ENDS

	///SIFT MATCHING: START
	SiftMatchGPU* matcher = pCreateNewSiftMatchGPU(8192);
	matcher->VerifyContextGL(); //must call once
	int(*match_buf)[2] = new int[MAXSIFTPTS][2];

	vector<float > descriptors1, descriptors2;
	descriptors1.reserve(MAXSIFTPTS * descriptorSize), descriptors2.reserve(MAXSIFTPTS * descriptorSize);

	vector<Point2i> RawPairWiseMatchID;	RawPairWiseMatchID.reserve(10000);

	start = omp_get_wtime();
	printf("Running feature matching...\n");
	for (int jj = 0; jj < nviews - 1; jj++)
	{
		if (timeID < 0)
			sprintf(Fname, "%s/D%d.dat", Path, jj);
		else
			sprintf(Fname, "%s/%d/D%d.dat", Path, jj, timeID);
		if (!ReadDescriptorBinarySIFTGPU(Fname, descriptors1))
			continue;

		int num1 = PtsPerView.at(jj);
		for (int ii = jj + 1; ii < nviews; ii++)
		{
			if (timeID < 0)
				sprintf(Fname, "%s/D%d.dat", Path, ii);
			else
				sprintf(Fname, "%s/%d/D%d.dat", Path, ii, timeID);
			if (!ReadDescriptorBinarySIFTGPU(Fname, descriptors2))
				continue;

			double start = omp_get_wtime();
			printf("View (%d, %d) of frame %d ...", jj, ii, timeID);

			int num2 = PtsPerView.at(ii);
			//Finding nearest neighbor. call matcher->SetMaxSift() to change the limit before calling setdescriptor if you want change maxMatch
			matcher->SetDescriptors(0, num1, &descriptors1[0]); //image 1
			matcher->SetDescriptors(1, num2, &descriptors2[0]); //image 2

			int num_match = matcher->GetSiftMatch(num1, match_buf, 0.7f, nndrRatio);
			for (int i = 0; i < num_match; ++i)
			{
				int id1 = match_buf[i][0], id2 = match_buf[i][1];
				RawPairWiseMatchID.push_back(Point2i(id1, id2));
			}

			printf("%d matches .... %.2fs\n", num_match, omp_get_wtime() - start);
			if (timeID < 0)
				sprintf(Fname, "%s/M_%d_%d.dat", Path, jj, ii);
			else
				sprintf(Fname, "%s/M%d_%d_%d.dat", Path, timeID, jj, ii);
			FILE *fp = fopen(Fname, "w+");
			fprintf(fp, "%d\n", RawPairWiseMatchID.size());
			for (int i = 0; i < RawPairWiseMatchID.size(); i++)
				fprintf(fp, "%d %d\n", RawPairWiseMatchID.at(i).x, RawPairWiseMatchID.at(i).y);
			fclose(fp);
		}
	}
	printf("Finished matching feature points ... in %.2fs\n", omp_get_wtime() - start);
	///SIFT MATCHING: ENDS

	GenerateMatchingTable(Path, nviews, timeID);

	delete[] match_buf;
	delete sift;
	delete matcher;
	FREE_MYLIB(hsiftgpu);

	return 0;
}
//Use CPU for flann Sift matching
//OulierRemoveTestMethod 0: no test, 1: Emat, 2: Fmat
int GeneratePointsCorrespondenceMatrix_SiftGPU2(char *Path, int nviews, int timeID, float nndrRatio, bool distortionCorrected, int OulierRemoveTestMethod, int nCams, int cameraToScan)
{
	// Allocation size to the largest width and largest height 1920x1080
	// Maximum working dimension. All the SIFT octaves that needs a larger texture size will be skipped. maxd = 2560 <-> 768MB of graphic memory. 
	char * argv[] = { "-fo", "-1", "-v", "0", "-p", "3072x2048", "-maxd", "4096" };
	//-fo -1    staring from -1 octave 
	//-v 1      only print out # feature and overall time
	//-loweo    add a (.5, .5) offset
	//-tc <num> set a soft limit to number of detected features
	//-m,       up to 2 orientations for each feature (change to single orientation by using -m 1)
	//-s        enable subpixel subscale (disable by using -s 0)
	//"-cuda", "[device_id]"  : cuda implementation (fastest for smaller images). CUDA-implementation allows you to create multiple instances for multiple threads. Checkout src\TestWin\MultiThreadSIFT
	// "-Display", "display_name" (for OPENGL) to select monitor/GPU (XLIB/GLUT) on windows the display name would be something like \\.\DISPLAY4
	//Only the following parameters can be changed after initialization (by calling ParseParam):-dw, -ofix, -ofix-not, -fo, -unn, -maxd, -b
	//to change other parameters at runtime, you need to first unload the dynamically loaded libaray reload the libarary, then create a new siftgpu instance

	//Init SiftGPU: START
#ifdef _WIN32
#ifdef _DEBUG
	HMODULE  hsiftgpu = LoadLibrary("siftgpu_d.dll");
#else
	HMODULE  hsiftgpu = LoadLibrary("siftgpu.dll");
#endif
#else
	void * hsiftgpu = dlopen("libsiftgpu.so", RTLD_LAZY);
#endif

	if (hsiftgpu == NULL)
		return 0;

	SiftGPU* (*pCreateNewSiftGPU)(int) = NULL;
	SiftMatchGPU* (*pCreateNewSiftMatchGPU)(int) = NULL;
	pCreateNewSiftGPU = (SiftGPU* (*) (int)) GET_MYPROC(hsiftgpu, "CreateNewSiftGPU");
	pCreateNewSiftMatchGPU = (SiftMatchGPU* (*)(int)) GET_MYPROC(hsiftgpu, "CreateNewSiftMatchGPU");
	SiftGPU* sift = pCreateNewSiftGPU(1);

	int argc = sizeof(argv) / sizeof(char*);
	sift->ParseParam(argc, argv);
	if (sift->CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED)
		return 0;
	//Init SiftGPU: END

	//SIFT DECTION: START
	int numKeys, descriptorSize = SIFTBINS;
	vector<float > descriptors; descriptors.reserve(MAXSIFTPTS * descriptorSize);
	vector<SiftGPU::SiftKeypoint> keys; keys.reserve(MAXSIFTPTS);

	vector<int>cumulativePts;
	vector<int>PtsPerView;

	int totalPts = 0;
	char Fname[200];

	double start;
	for (int ii = 0; ii < nviews; ii++)
	{
		keys.clear(), descriptors.clear();
		start = omp_get_wtime();

		//Try to read all sift points if available
		if (timeID < 0)
			sprintf(Fname, "%s/K%d.dat", Path, ii);
		else
			sprintf(Fname, "%s/%d/K%d.dat", Path, ii, timeID);
		if (ReadKPointsBinarySIFTGPU(Fname, keys))
		{
			printf("Loaded %s with %d SIFTs\n", Fname, keys.size());
			cumulativePts.push_back(totalPts);
			totalPts += keys.size();
			PtsPerView.push_back(keys.size());
			continue; //Sift availble, move one
		}

		if (timeID < 0)
			sprintf(Fname, "%s/%d.png", Path, ii);
		else
			sprintf(Fname, "%s/%d/%d.png", Path, ii, timeID);
		if (sift->RunSIFT(Fname)) //You can have at most one OpenGL-based SiftGPU (per process)--> no omp can be used
		{
			//sprintf(Fname, "%s/%d.sift", Path, ii);sift->SaveSIFT(Fname);
			numKeys = sift->GetFeatureNum();
			keys.resize(numKeys);    descriptors.resize(descriptorSize * numKeys);
			sift->GetFeatureVector(&keys[0], &descriptors[0]);

			if (timeID < 0)
			{
				sprintf(Fname, "%s/K%d.dat", Path, ii); WriteKPointsBinarySIFTGPU(Fname, keys);
				sprintf(Fname, "%s/D%d.dat", Path, ii); WriteDescriptorBinarySIFTGPU(Fname, descriptors);
			}
			else
			{
				sprintf(Fname, "%s/%d/K%d.dat", Path, ii, timeID); WriteKPointsBinarySIFTGPU(Fname, keys);
				sprintf(Fname, "%s/%d/D%d.dat", Path, ii, timeID); WriteDescriptorBinarySIFTGPU(Fname, descriptors);
			}

			printf("View %d: %d points ... Wrote to files. Take %.2fs\n", ii, numKeys, omp_get_wtime() - start);

			cumulativePts.push_back(totalPts);
			totalPts += numKeys;
			PtsPerView.push_back(numKeys);
		}
		else
			printf("Cannot load %s", Fname);
	}
	cumulativePts.push_back(totalPts);

	if (timeID < 0)
		sprintf(Fname, "%s/CumlativePoints.txt", Path);
	else
		sprintf(Fname, "%s/CumlativePoints_%d.txt", Path, timeID);
	FILE* fp = fopen(Fname, "w+");
	for (int ii = 0; ii < cumulativePts.size(); ii++)
		fprintf(fp, "%d\n", cumulativePts.at(ii));
	fclose(fp);
	//SIFT DECTION: ENDS

	///SIFT MATCHING: START
	int nthreads = omp_get_max_threads();
	omp_set_num_threads(nthreads);

	vector<KeyPoint> Keys1, Keys2;
	SiftMatchGPU* matcher = pCreateNewSiftMatchGPU(8192);
	matcher->VerifyContextGL(); //must call once
	int(*match_buf)[2] = new int[MAXSIFTPTS][2];

	vector<Point2i> *RawPairWiseMatchID = new vector<Point2i>[nthreads];
	for (int ii = 0; ii < nthreads; ii++)
		RawPairWiseMatchID[ii].reserve(10000);
	vector<Point2i> *SRawPairWiseMatchID = new vector<Point2i>[nthreads];
	for (int ii = 0; ii < nthreads; ii++)
		SRawPairWiseMatchID[ii].reserve(10000);

	const int ninlierThesh = 50;
	int *SortingVec = new int[50000 * nthreads]; //should be more than enough
	int *tId = new int[50000 * nthreads];

	bool BinaryDesc = false, useBFMatcher = false; // SET TO TRUE TO USE BRUTE FORCE MATCHER
	const int knn = 2, ntrees = 4, maxLeafCheck = 128;

	start = omp_get_wtime();
	printf("Running feature matching...\n");
	Mat descriptors1;
	for (int jj = 0; jj < nviews - 1; jj++)
	{
		if (timeID < 0)
			sprintf(Fname, "%s/D%d.dat", Path, jj);
		else
			sprintf(Fname, "%s/%d/D%d.dat", Path, jj, timeID);
		Mat descriptors1 = ReadDescriptorBinarySIFTGPU(Fname);
		if (descriptors1.rows == 1)
			continue;

#pragma omp parallel for
		for (int ii = jj + 1; ii < nviews; ii++)
		{
			char Fname[200];
			if (timeID < 0)
				sprintf(Fname, "%s/D%d.dat", Path, ii);
			else
				sprintf(Fname, "%s/%d/D%d.dat", Path, ii, timeID);
			Mat descriptors2 = ReadDescriptorBinarySIFTGPU(Fname);
			if (descriptors2.rows == 1)
				continue;

			double start = omp_get_wtime();
			int threadID = omp_get_thread_num();
			RawPairWiseMatchID[threadID].clear(), SRawPairWiseMatchID[threadID].clear();

			//Finding nearest neighbor
			Mat indices, dists;
			vector<vector<DMatch> > matches;
			if (BinaryDesc)
			{
				//printf("Binary descriptors detected...\n");// ORB, Brief, BRISK, FREAK
				if (useBFMatcher)
				{
					cv::BFMatcher matcher(cv::NORM_HAMMING); // use cv::NORM_HAMMING2 for ORB descriptor with WTA_K == 3 or 4 (see ORB constructor)
					matcher.knnMatch(descriptors2, descriptors1, matches, knn);
				}
				else
				{
					// Create Flann LSH index
					cv::flann::Index flannIndex(descriptors1, cv::flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);
					flannIndex.knnSearch(descriptors2, indices, dists, knn, cv::flann::SearchParams());
				}
			}
			else
			{
				if (useBFMatcher)
				{
					cv::BFMatcher matcher(cv::NORM_L2);
					matcher.knnMatch(descriptors2, descriptors1, matches, knn);
				}
				else
				{
					// Create Flann KDTree index
					cv::flann::Index flannIndex(descriptors1, cv::flann::KDTreeIndexParams(ntrees));//, cvflann::FLANN_DIST_EUCLIDEAN);
					flannIndex.knnSearch(descriptors2, indices, dists, knn, cv::flann::SearchParams(maxLeafCheck));
				}
			}

			// Find correspondences by NNDR (Nearest Neighbor Distance Ratio)
			int count = ii - jj - 1;
			for (int i = 0; i <= jj - 1; i++)
				count += nviews - i - 1;

			if (!useBFMatcher)
			{
				for (int i = 0; i < descriptors2.rows; ++i)
				{
					int ind1 = indices.at<int>(i, 0);
					if (indices.at<int>(i, 0) >= 0 && indices.at<int>(i, 1) >= 0 && dists.at<float>(i, 0) <= nndrRatio * dists.at<float>(i, 1))
						RawPairWiseMatchID[threadID].push_back(Point2i(ind1, i));
				}
			}
			else
			{
				for (unsigned int i = 0; i < matches.size(); ++i)
					if (matches.at(i).size() == 2 && matches.at(i).at(0).distance <= nndrRatio * matches.at(i).at(1).distance)
						RawPairWiseMatchID[threadID].push_back(Point2i(matches.at(i).at(0).trainIdx, i));
			}

			//To remove the nonsense case of every point matchces to 1 point-->IT HAPPENED
			SRawPairWiseMatchID[threadID].push_back(RawPairWiseMatchID[threadID].at(0));
			for (unsigned int i = 1; i < min(RawPairWiseMatchID[threadID].size(), 50000); i++)
				if (RawPairWiseMatchID[threadID].at(i).x != RawPairWiseMatchID[threadID].at(i - 1).x)
					SRawPairWiseMatchID[threadID].push_back(RawPairWiseMatchID[threadID].at(i));

			if (SRawPairWiseMatchID[threadID].size() < ninlierThesh)
				continue;

			//Start sorting
			for (unsigned int i = 0; i < min(SRawPairWiseMatchID[threadID].size(), 50000); i++)
			{
				SortingVec[i + 50000 * threadID] = SRawPairWiseMatchID[threadID].at(i).x;
				tId[i + 50000 * threadID] = i;
			}
			Quick_Sort_Int(SortingVec + 50000 * threadID, tId + 50000 * threadID, 0, min(SRawPairWiseMatchID[threadID].size(), 50000) - 1);

			//Store sorted vector
			RawPairWiseMatchID[threadID].push_back(SRawPairWiseMatchID[threadID].at(tId[0 + 50000 * threadID]));
			for (unsigned int i = 1; i < min(SRawPairWiseMatchID[threadID].size(), 50000); i++)
				if (SortingVec[i + 50000 * threadID] != SortingVec[i - 1 + 50000 * threadID])
					RawPairWiseMatchID[threadID].push_back(SRawPairWiseMatchID[threadID].at(tId[i + 50000 * threadID]));

#pragma omp critical
			{
				printf("View (%d, %d) of frame %d...%d matches... %.2fs\n", jj, ii, timeID, SRawPairWiseMatchID[threadID].size(), omp_get_wtime() - start);
				if (timeID < 0)
					sprintf(Fname, "%s/M_%d_%d.dat", Path, jj, ii);
				else
					sprintf(Fname, "%s/M%d_%d_%d.dat", Path, timeID, jj, ii);
				FILE *fp = fopen(Fname, "w+");
				fprintf(fp, "%d\n", SRawPairWiseMatchID[threadID].size());
				for (int i = 0; i < SRawPairWiseMatchID[threadID].size(); i++)
					fprintf(fp, "%d %d\n", SRawPairWiseMatchID[threadID].at(i).x, SRawPairWiseMatchID[threadID].at(i).y);
				fclose(fp);
			}
		}
	}
	printf("Finished matching feature points ... in %.2fs\n", omp_get_wtime() - start);
	delete[]SortingVec;
	delete[]tId;
	delete[]RawPairWiseMatchID, delete[]SRawPairWiseMatchID;
	///SIFT MATCHING: ENDS

	start = omp_get_wtime();
	for (int jj = 0; jj < nviews - 1; jj++)
#pragma omp parallel for
		for (int ii = jj + 1; ii < nviews; ii++)
			EssentialMatOutliersRemove(Path, timeID, jj, ii, nCams, cameraToScan, ninlierThesh, distortionCorrected, false);
	printf("Finished pruning matches ... in %.2fs\n", omp_get_wtime() - start);

	GenerateMatchingTable(Path, nviews, timeID);
	delete[] match_buf;
	delete sift;
	delete matcher;
	FREE_MYLIB(hsiftgpu);

	return 0;
}

void GenerateMatchingTable(char *Path, int nviews, int timeID)
{
	char Fname[200];

	int totalPts;
	vector<int> cumulativePts;
	ReadCumulativePoints(Path, nviews, timeID, cumulativePts);
	totalPts = cumulativePts.at(nviews);

	vector<Point2i> *AllPairWiseMatchingId = new vector<Point2i>[nviews*(nviews - 1) / 2];
	for (int ii = 0; ii < nviews*(nviews - 1) / 2; ii++)
		AllPairWiseMatchingId[ii].reserve(10000);

	int percent = 10, incre = 10;
	int nfiles = nviews*(nviews - 1) / 2, filesCount = 0;
	double start = omp_get_wtime();
	for (int jj = 0; jj < nviews - 1; jj++)
	{
		for (int ii = jj + 1; ii < nviews; ii++)
		{
			if (100.0*filesCount / nfiles >= percent)
			{
				printf("@\r# %.2f%% (%.2fs) Reading pairwise matches....", 100.0*filesCount / nfiles, omp_get_wtime() - start);
				percent += incre;
			}
			filesCount++;
			if (timeID < 0)
				sprintf(Fname, "%s/_M_%d_%d.dat", Path, jj, ii);
			else
				sprintf(Fname, "%s/M%d_%d_%d.dat", Path, timeID, jj, ii);

			int count = ii - jj - 1;
			for (int i = 0; i <= jj - 1; i++)
				count += nviews - i - 1;

			int id1, id2, npts;
			FILE *fp = fopen(Fname, "r");
			if (fp == NULL)
				continue;
			fscanf(fp, "%d ", &npts);
			AllPairWiseMatchingId[count].reserve(npts);
			while (fscanf(fp, "%d %d ", &id1, &id2) != EOF)
				AllPairWiseMatchingId[count].push_back(Point2i(id1, id2));
			fclose(fp);
		}
	}

	//Generate Visbible Points Table
	vector<int> *KeysBelongTo3DPoint = new vector <int>[nviews];
	for (int jj = 0; jj < nviews; jj++)
	{
		KeysBelongTo3DPoint[jj].reserve(cumulativePts[jj + 1] - cumulativePts[jj]);
		for (int ii = 0; ii < cumulativePts[jj + 1] - cumulativePts[jj]; ii++)
			KeysBelongTo3DPoint[jj].push_back(-1);
	}

	vector<int>*ViewMatch = new vector<int>[totalPts]; //cotains all visible views of 1 3D point
	vector<int>*PointIDMatch = new vector<int>[totalPts];//cotains all keyID of the visible views of 1 3D point
	int count3D = 0;

	for (int jj = 0; jj < nviews; jj++)
	{
		for (int ii = jj + 1; ii < nviews; ii++)
		{
			int PairWiseID = ii - jj - 1;
			for (int i = 0; i <= jj - 1; i++)
				PairWiseID += nviews - i - 1;
			//printf("@(%d, %d) with %d 3+ points ...TE: %.2fs\n ", jj, ii, count3D, omp_get_wtime() - start);
			for (int kk = 0; kk < AllPairWiseMatchingId[PairWiseID].size(); kk++)
			{
				int id1 = AllPairWiseMatchingId[PairWiseID].at(kk).x;
				int id2 = AllPairWiseMatchingId[PairWiseID].at(kk).y;
				int ID3D1 = KeysBelongTo3DPoint[jj].at(id1), ID3D2 = KeysBelongTo3DPoint[ii].at(id2);
				if (ID3D1 == -1 && ID3D2 == -1) //Both are never seeen before
				{
					ViewMatch[count3D].push_back(jj), ViewMatch[count3D].push_back(ii);
					PointIDMatch[count3D].push_back(id1), PointIDMatch[count3D].push_back(id2);
					KeysBelongTo3DPoint[jj].at(id1) = count3D, KeysBelongTo3DPoint[ii].at(id2) = count3D; //this pair of corres constitutes 3D point #count
					count3D++;
				}
				else if (ID3D1 == -1 && ID3D2 != -1)
				{
					ViewMatch[ID3D2].push_back(jj);
					PointIDMatch[ID3D2].push_back(id1);
					KeysBelongTo3DPoint[jj].at(id1) = ID3D2; //this point constitutes 3D point #ID3D2
				}
				else if (ID3D1 != -1 && ID3D2 == -1)
				{
					ViewMatch[ID3D1].push_back(ii);
					PointIDMatch[ID3D1].push_back(id2);
					KeysBelongTo3DPoint[ii].at(id2) = ID3D1; //this point constitutes 3D point #ID3D2
				}
				else if (ID3D1 != -1 && ID3D2 != -1 && ID3D1 != ID3D2)//Strange case where 1 point (usually not vey discrimitive or repeating points) is matched to multiple points in the same view pair 
					//--> Just concatanate the one with fewer points to largrer one and hope MultiTriangulationRansac can do sth.
				{
					if (ViewMatch[ID3D1].size() >= ViewMatch[ID3D2].size())
					{
						int nmatches = ViewMatch[ID3D2].size();
						for (int ll = 0; ll < nmatches; ll++)
						{
							ViewMatch[ID3D1].push_back(ViewMatch[ID3D2].at(ll));
							PointIDMatch[ID3D1].push_back(PointIDMatch[ID3D2].at(ll));
						}
						ViewMatch[ID3D2].clear(), PointIDMatch[ID3D2].clear();
					}
					else
					{
						int nmatches = ViewMatch[ID3D1].size();
						for (int ll = 0; ll < nmatches; ll++)
						{
							ViewMatch[ID3D2].push_back(ViewMatch[ID3D1].at(ll));
							PointIDMatch[ID3D2].push_back(PointIDMatch[ID3D1].at(ll));
						}
						ViewMatch[ID3D1].clear(), PointIDMatch[ID3D1].clear();
					}
				}
				else//(ID3D1 == ID3D2): cycle in the corres, i.e. a-b, a-c, and b-c
					continue;
			}
		}
	}
	printf("Merged correspondences in %.2fs\n ", count3D, omp_get_wtime() - start);

	int count = 0, maxmatches = 0, npts = 0;
	if (timeID < 0)
		sprintf(Fname, "%s/ViewPM.txt", Path);
	else
		sprintf(Fname, "%s/ViewPM_%d.txt", Path, timeID);
	FILE *fp = fopen(Fname, "w+");
	if (fp != NULL)
	{
		for (int jj = 0; jj < count3D; jj++)
		{
			int nmatches = ViewMatch[jj].size();
			if (nmatches < 2 || nmatches > nviews * 2)
				continue;

			npts++;
			if (nmatches > 2)
				count++;
			if (nmatches > maxmatches)
				maxmatches = nmatches;

			fprintf(fp, "%d ", nmatches);
			for (int ii = 0; ii < nmatches; ii++)
				fprintf(fp, "%d ", ViewMatch[jj].at(ii));
			fprintf(fp, "\n");
		}
		fclose(fp);
	}
	printf("#3+ points: %d. Max #matches views:  %d. #matches point: %d\n", count, maxmatches, npts);


	if (timeID < 0)
		sprintf(Fname, "%s/IDPM.txt", Path);
	else
		sprintf(Fname, "%s/IDPM_%d.txt", Path, timeID);
	fp = fopen(Fname, "w+");
	if (fp != NULL)
	{
		for (int jj = 0; jj < count3D; jj++)
		{
			int nmatches = PointIDMatch[jj].size();
			if (nmatches < 2 || nmatches > nviews * 2)
				continue;

			fprintf(fp, "%d ", nmatches);
			for (int ii = 0; ii < nmatches; ii++)
				fprintf(fp, "%d ", PointIDMatch[jj].at(ii));
			fprintf(fp, "\n");
		}
		fclose(fp);
	}
	printf("Finished generateing point correspondence matrix\n");

	delete[]ViewMatch;
	delete[]PointIDMatch;

	return;
}
void GenerateViewCorrespondenceMatrix(char *Path, int nviews, int timeID)
{
	int ii, jj, kk, ll, mm, nn;
	char Fname[200];

	vector<int> cumulativePts, PtsView;
	if (timeID < 0)
		sprintf(Fname, "%s/CumlativePoints.txt", Path);
	else
		sprintf(Fname, "%s/CumlativePoints_%d.txt", Path, timeID);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot open %s", Fname);
		exit(1);
	}
	for (ii = 0; ii < nviews + 1; ii++)
	{
		fscanf(fp, "%d\n", &jj);
		cumulativePts.push_back(jj);
	}
	fclose(fp);

	Mat viewMatrix(nviews, nviews, CV_32S);
	viewMatrix = Scalar::all(0);

	vector<int>matches; matches.reserve(nviews * 2);
	for (mm = 0; mm < nviews - 1; mm++)
	{
		for (nn = mm + 1; nn < nviews; nn++)
		{
			int totalPts = cumulativePts.at(nviews);

			int count = 0;
			if (timeID < 0)
				sprintf(Fname, "%s/PM.txt", Path);
			else
				sprintf(Fname, "%s/PM_%d.txt", Path, timeID);
			fp = fopen(Fname, "r");
			if (fp == NULL)
			{
				printf("Cannot open %s", Fname);
				exit(1);
			}
			for (jj = 0; jj < totalPts; jj++)
			{
				kk = 0; matches.clear();
				fscanf(fp, "%d ", &kk);
				for (ii = 0; ii < kk; ii++)
				{
					fscanf(fp, "%d ", &ll);
					matches.push_back(ll);
				}

				if (jj >= cumulativePts.at(mm) && jj < cumulativePts.at(mm + 1))
				{
					for (ii = 0; ii < kk; ii++)
					{
						int match = matches.at(ii);
						if (match >= cumulativePts.at(nn) && match < cumulativePts.at(nn + 1))
							viewMatrix.at<int>(mm + nn*nviews) += 1;
					}
				}
			}
			fclose(fp);

		}
	}
	completeSymm(viewMatrix, true);

	if (timeID < 0)
		sprintf(Fname, "%s/VM.txt", Path);
	else
		sprintf(Fname, "%s/VM_%d.txt", Path, timeID);
	fp = fopen(Fname, "w+");
	for (jj = 0; jj < nviews; jj++)
	{
		for (ii = 0; ii < nviews; ii++)
			fprintf(fp, "%d ", viewMatrix.at<int>(ii + jj*nviews));
		fprintf(fp, "\n");
	}
	fclose(fp);

	return;
}
void BestPairFinder(char *Path, int nviews, int timeID, int *viewPair)
{
	char Fname[200];
	int ii, jj;

	int *viewMatrix = new int[nviews*nviews];

	if (timeID < 0)
		sprintf(Fname, "%s/VM.txt", Path);
	else
		sprintf(Fname, "%s/VM_%d.txt", Path, timeID);
	FILE *fp = fopen(Fname, "r");
	for (jj = 0; jj < nviews; jj++)
		for (ii = 0; ii < nviews; ii++)
			fscanf(fp, "%d ", &viewMatrix[ii + jj*nviews]);
	fclose(fp);

	int bestCount = 0;
	for (jj = 0; jj < nviews; jj++)
	{
		for (ii = 0; ii < nviews; ii++)
		{
			if (viewMatrix[ii + jj*nviews] > bestCount)
			{
				bestCount = viewMatrix[ii + jj*nviews];
				viewPair[0] = ii, viewPair[1] = jj;
			}
		}
	}

	delete[]viewMatrix;

	return;
}
int NextViewFinder(char *Path, int nviews, int timeID, int currentView, int &maxPoints, vector<int> usedViews)
{
	char Fname[200];
	int ii, jj, kk;

	int *viewMatrix = new int[nviews*nviews];

	if (timeID < 0)
		sprintf(Fname, "%s/VM.txt", Path);
	else
		sprintf(Fname, "%s/VM_%d.txt", Path, timeID);
	FILE *fp = fopen(Fname, "r");
	for (jj = 0; jj < nviews; jj++){
		for (ii = 0; ii < nviews; ii++){
			fscanf(fp, "%d ", &viewMatrix[ii + jj*nviews]);
		}
	}
	fclose(fp);

	for (ii = 0; ii < usedViews.size(); ii++){
		for (jj = 0; jj < usedViews.size(); jj++){
			if (jj != ii){
				viewMatrix[usedViews.at(ii) + usedViews.at(jj)*nviews] = 0, viewMatrix[usedViews.at(jj) + usedViews.at(ii)*nviews] = 0;
			}
		}
	}

	jj = 0;
	for (ii = 0; ii < nviews; ii++)
	{
		if (viewMatrix[ii + currentView*nviews] > jj)
		{
			jj = viewMatrix[ii + currentView*nviews];
			kk = ii;
		}
	}

	maxPoints = jj;

	delete[]viewMatrix;

	return kk;
}

int GetPoint2DPairCorrespondence(char *Path, int timeID, vector<int>viewID, vector<KeyPoint>&keypoints1, vector<KeyPoint>&keypoints2, vector<int>&CorrespondencesID)
{
	//SelectedIndex: index of correspondenceID in the total points pool
	keypoints1.clear(), keypoints2.clear(), CorrespondencesID.clear();
	char Fname[200];

	if (timeID < 0)
		sprintf(Fname, "%s/K%d.dat", Path, viewID.at(0));
	else
		sprintf(Fname, "%s/%d/K%d.dat", Path, viewID.at(0), timeID);
	if (useGPU)
		ReadKPointsBinarySIFTGPU(Fname, keypoints1);
	else
		ReadKPointsBinary(Fname, keypoints1);

	if (timeID < 0)
		sprintf(Fname, "%s/K%d.dat", Path, viewID.at(1));
	else
		sprintf(Fname, "%s/%d/K%d.dat", Path, viewID.at(1), timeID);
	if (useGPU)
		ReadKPointsBinarySIFTGPU(Fname, keypoints2);
	else
		ReadKPointsBinary(Fname, keypoints2);

	vector<int>matches; matches.reserve(500);//Cannot be found in more than 500 views!

	if (timeID < 0)
		sprintf(Fname, "%s/M_%d_%d.dat", Path, viewID.at(0), viewID.at(1));
	else
		sprintf(Fname, "%s/M%d_%d_%d.dat", Path, timeID, viewID.at(0), viewID.at(1));

	int npts, id1, id2;
	FILE *fp = fopen(Fname, "r");
	fscanf(fp, "%d ", &npts);
	CorrespondencesID.reserve(npts * 2);
	while (fscanf(fp, "%d %d ", &id1, &id2) != EOF)
		CorrespondencesID.push_back(id1), CorrespondencesID.push_back(id2);
	fclose(fp);

	return 0;
}
int GetPoint3D2DPairCorrespondence(char *Path, int nviews, int timeID, vector<int> cumulativePts, vector<int> viewID, Point3d *ThreeD, vector<KeyPoint>&keypoints1, vector<KeyPoint>&keypoints2, vector<int>&TwoDCorrespondencesID, vector<int> &ThreeDCorrespondencesID, vector<int>&SelectedIndex, bool SwapView)
{
	//SelectedIndex: index of correspondenceID in the total points pool
	keypoints1.clear(), keypoints2.clear(), TwoDCorrespondencesID.clear(), ThreeDCorrespondencesID.clear();

	int ii, jj, kk, ll, id;
	char Fname[200];

	if (timeID < 0)
		sprintf(Fname, "%s/K%d.dat", Path, viewID.at(0));
	else
		sprintf(Fname, "%s/%d/K%d.dat", Path, viewID.at(0), timeID);
	if (useGPU)
		ReadKPointsBinarySIFTGPU(Fname, keypoints1);
	else
		ReadKPointsBinary(Fname, keypoints1);

	if (timeID < 0)
		sprintf(Fname, "%s/K%d.dat", Path, viewID.at(1));
	else
		sprintf(Fname, "%s/%d/K%d.dat", Path, viewID.at(1), timeID);
	if (useGPU)
		ReadKPointsBinarySIFTGPU(Fname, keypoints2);
	else
		ReadKPointsBinary(Fname, keypoints2);

	int totalPts = cumulativePts.at(nviews);
	vector<int>matches; matches.reserve(500);//Cannot be found in more than 500 views!
	//vector<int>CorrespondencesID;CorrespondencesID.reserve((cumulativePts.at(viewID.at(1)+1)-cumulativePts.at(viewID.at(0)+1))*2);

	if (timeID < 0)
		sprintf(Fname, "%s/PM.txt", Path);
	else
		sprintf(Fname, "%s/PM_%d.txt", Path, timeID);
	FILE *fp = fopen(Fname, "r");
	for (jj = 0; jj < totalPts; jj++)
	{
		kk = 0; matches.clear();
		fscanf(fp, "%d ", &kk);
		for (ii = 0; ii < kk; ii++)
		{
			fscanf(fp, "%d ", &ll);
			matches.push_back(ll);
		}

		if (jj >= cumulativePts.at(viewID.at(0)) && jj < cumulativePts.at(viewID.at(0) + 1))
		{
			for (ii = 0; ii < matches.size(); ii++)
			{
				int match = matches.at(ii);
				if (match >= cumulativePts.at(viewID.at(1)) && match < cumulativePts.at(viewID.at(1) + 1))
				{
					TwoDCorrespondencesID.push_back(jj - cumulativePts.at(viewID.at(0)));
					TwoDCorrespondencesID.push_back(match - cumulativePts.at(viewID.at(1)));
					SelectedIndex.push_back(jj);

					if (abs(ThreeD[jj].z) > 0.0 && !SwapView)
					{
						id = match - cumulativePts.at(viewID.at(1));
						ThreeDCorrespondencesID.push_back(id);
						ThreeDCorrespondencesID.push_back(jj);
					}
					else if (abs(ThreeD[match].z) > 0.0 && SwapView)
					{
						id = jj - cumulativePts.at(viewID.at(0));
						ThreeDCorrespondencesID.push_back(id);
						ThreeDCorrespondencesID.push_back(match);
					}
				}
			}
		}
	}
	fclose(fp);

	return 0;
}
int GetPoint3D2DAllCorrespondence(char *Path, int nviews, int timeID, vector<int> cumulativePts, Point3d *ThreeD, vector<int> availViews, vector<int>&Selected3DIndex, vector<Point2d> *selected2D, vector<int>*nSelectedViews, int &nSelectedPts)
{
	//SelectedIndex: index of correspondenceID in the total points pool
	Selected3DIndex.clear();
	int ii, jj, kk, ll;
	char Fname[200];

	bool PointAdded, PointAdded2, once;
	int viewID1, viewID2, match, totalPts = cumulativePts.at(nviews);

	vector<int>matches; matches.reserve(500);//Cannot be found in more than 500 views!
	//vector<int>CorrespondencesID;CorrespondencesID.reserve((cumulativePts.at(viewsID[1]+1)-cumulativePts.at(viewsID[0]+1))*2);
	vector<int> *selected2Did = new vector<int>[totalPts];
	for (ii = 0; ii < totalPts; ii++)
		selected2Did[ii].reserve(20);

	//fill in selected3D, select3Dindex, index of 2d points in available views
	if (timeID < 0)
		sprintf(Fname, "%s/PM.txt", Path);
	else
		sprintf(Fname, "%s/PM_%d.txt", Path, timeID);
	FILE* fp = fopen(Fname, "r");
	nSelectedPts = 0;
	for (jj = 0; jj < totalPts; jj++)
	{
		kk = 0; matches.clear();
		fscanf(fp, "%d ", &kk);
		for (ii = 0; ii < kk; ii++)
		{
			fscanf(fp, "%d ", &match);
			matches.push_back(match);
		}

		if (abs(ThreeD[jj].z) > 0.0 && matches.size() > 0)
		{
			once = true, PointAdded = false, PointAdded2 = false;
			for (kk = 0; kk < availViews.size(); kk++)
			{
				viewID1 = availViews.at(kk);
				if (jj >= cumulativePts.at(viewID1) && jj < cumulativePts.at(viewID1 + 1))
				{
					for (ii = 0; ii < matches.size(); ii++)
					{
						PointAdded = false;
						match = matches.at(ii);
						for (ll = 0; ll < availViews.size(); ll++)
						{
							if (ll == kk)
								continue;

							viewID2 = availViews.at(ll);
							if (match >= cumulativePts.at(viewID2) && match < cumulativePts.at(viewID2 + 1))
							{
								if (once)
								{
									once = false, PointAdded = true, PointAdded2 = true;
									Selected3DIndex.push_back(jj);
									nSelectedViews[nSelectedPts].clear();  nSelectedViews[nSelectedPts].push_back(viewID1);
									selected2Did[nSelectedPts].push_back(jj - cumulativePts.at(viewID1));
								}
								nSelectedViews[nSelectedPts].push_back(viewID2);
								selected2Did[nSelectedPts].push_back(match - cumulativePts.at(viewID2));
							}
							if (PointAdded)
								break;
						}
					}
				}
				if (PointAdded2)
					break;
			}
			if (PointAdded2)
				nSelectedPts++;
		}
	}
	fclose(fp);
	//fill in select2D: points seen in available views
	vector<KeyPoint> keypoints; keypoints.reserve(10000);

	for (ii = 0; ii < nSelectedPts; ii++)
	{
		int nviews = nSelectedViews[ii].size();
		selected2D[ii].clear(); selected2D[ii].reserve(nviews);
		for (jj = 0; jj < nviews; jj++)
			selected2D[ii].push_back(Point2d(0, 0));
	}

	for (kk = 0; kk < availViews.size(); kk++)
	{
		int viewID = availViews.at(kk); keypoints.clear();
		if (timeID < 0)
			sprintf(Fname, "%s/K%d.dat", Path, viewID);
		else
			sprintf(Fname, "%s/%d/K%d.dat", Path, viewID, timeID);

		if (useGPU)
			ReadKPointsBinarySIFTGPU(Fname, keypoints);
		else
			ReadKPointsBinary(Fname, keypoints);
		for (ll = 0; ll < nSelectedPts; ll++)
		{
			for (jj = 0; jj < nSelectedViews[ll].size(); jj++)
			{
				if (nSelectedViews[ll].at(jj) == viewID)
				{
					int poindID = selected2Did[ll].at(jj);
					selected2D[ll].at(jj).x = keypoints.at(poindID).pt.x;
					selected2D[ll].at(jj).y = keypoints.at(poindID).pt.y;
					break;
				}
			}
		}
	}

	delete[]selected2Did;
	return 0;
}

void ProjectandDistort(Point3d WC, Point2d *pts, double *P, double *camera, double *distortion, int nviews)
{
	int ii;
	double num1, num2, denum;

	for (ii = 0; ii < nviews; ii++)
	{
		num1 = P[ii * 12 + 0] * WC.x + P[ii * 12 + 1] * WC.y + P[ii * 12 + 2] * WC.z + P[ii * 12 + 3];
		num2 = P[ii * 12 + 4] * WC.x + P[ii * 12 + 5] * WC.y + P[ii * 12 + 6] * WC.z + P[ii * 12 + 7];
		denum = P[ii * 12 + 8] * WC.x + P[ii * 12 + 9] * WC.y + P[ii * 12 + 10] * WC.z + P[ii * 12 + 11];

		pts[ii].x = num1 / denum, pts[ii].y = num2 / denum;
		if (camera != NULL)
			LensDistortionPoint(&pts[ii], camera + ii * 9, distortion + ii * 13);
	}

	return;
}
void Stereo_Triangulation(Point2d *pts1, Point2d *pts2, double *P1, double *P2, Point3d *WC, int npts)
{
	int ii;
	double A[12], B[4], u1, v1, u2, v2;
	double p11 = P1[0], p12 = P1[1], p13 = P1[2], p14 = P1[3];
	double p21 = P1[4], p22 = P1[5], p23 = P1[6], p24 = P1[7];
	double p31 = P1[8], p32 = P1[9], p33 = P1[10], p34 = P1[11];

	double P11 = P2[0], P12 = P2[1], P13 = P2[2], P14 = P2[3];
	double P21 = P2[4], P22 = P2[5], P23 = P2[6], P24 = P2[7];
	double P31 = P2[8], P32 = P2[9], P33 = P2[10], P34 = P2[11];

	for (ii = 0; ii < npts; ii++)
	{
		u1 = pts1[ii].x, v1 = pts1[ii].y;
		u2 = pts2[ii].x, v2 = pts2[ii].y;

		A[0] = p11 - u1*p31;
		A[1] = p12 - u1*p32;
		A[2] = p13 - u1*p33;
		A[3] = p21 - v1*p31;
		A[4] = p22 - v1*p32;
		A[5] = p23 - v1*p33;

		A[6] = P11 - u2*P31;
		A[7] = P12 - u2*P32;
		A[8] = P13 - u2*P33;
		A[9] = P21 - v2*P31;
		A[10] = P22 - v2*P32;
		A[11] = P23 - v2*P33;

		B[0] = u1*p34 - p14;
		B[1] = v1*p34 - p24;
		B[2] = u2*P34 - P14;
		B[3] = v2*P34 - P24;

		QR_Solution_Double(A, B, 4, 3);

		WC[ii].x = B[0];
		WC[ii].y = B[1];
		WC[ii].z = B[2];
	}

	return;
}
void TwoViewTriangulationQualityCheck(Point2d *pts1, Point2d *pts2, Point3d *WC, double *P1, double *P2, double *K1, double *K2, double *distortion1, double *distortion2, bool *GoodPoints, int npts, double thresh)
{
	double u1, v1, u2, v2, denum, reprojectionError;
	for (int ii = 0; ii < npts; ii++)
	{
		Point2d p1(pts1[ii].x, pts1[ii].y), p2(pts2[ii].x, pts2[ii].y);
		if (distortion1 != NULL)
		{
			LensCorrectionPoint(&p1, K1, distortion1);
			LensCorrectionPoint(&p2, K2, distortion2);
		}

		//Project to 1st view
		denum = P1[8] * WC[ii].x + P1[9] * WC[ii].y + P1[10] * WC[ii].z + P1[11];
		u1 = (P1[0] * WC[ii].x + P1[1] * WC[ii].y + P1[2] * WC[ii].z + P1[3]) / denum;
		v1 = (P1[4] * WC[ii].x + P1[5] * WC[ii].y + P1[6] * WC[ii].z + P1[7]) / denum;

		//Project to 2nd view
		denum = P2[8] * WC[ii].x + P2[9] * WC[ii].y + P2[10] * WC[ii].z + P2[11];
		u2 = (P2[0] * WC[ii].x + P2[1] * WC[ii].y + P2[2] * WC[ii].z + P2[3]) / denum;
		v2 = (P2[4] * WC[ii].x + P2[5] * WC[ii].y + P2[6] * WC[ii].z + P2[7]) / denum;

		reprojectionError = (abs(u2 - p2.x) + abs(v2 - p2.y) + abs(u1 - p1.x) + abs(v1 - p1.y)) / 4.0;
		if (reprojectionError > thresh)
			GoodPoints[ii] = false;
		else
			GoodPoints[ii] = true;
	}
	return;
}
void NviewTriangulation(Point2d *pts, double *P, Point3d *WC, int nview, int npts, double *Cov, double *A, double *B)
{
	int ii, jj, kk;
	bool MenCreated = false;
	if (A == NULL)
	{
		MenCreated = true;
		A = new double[6 * nview];
		B = new double[2 * nview];
	}
	double u, v;

	if (Cov == NULL)
	{
		for (ii = 0; ii < npts; ii++)
		{
			for (jj = 0; jj < nview; jj++)
			{
				u = pts[ii + jj*npts].x, v = pts[ii + jj*npts].y;

				A[6 * jj + 0] = P[12 * jj] - u*P[12 * jj + 8];
				A[6 * jj + 1] = P[12 * jj + 1] - u*P[12 * jj + 9];
				A[6 * jj + 2] = P[12 * jj + 2] - u*P[12 * jj + 10];
				A[6 * jj + 3] = P[12 * jj + 4] - v*P[12 * jj + 8];
				A[6 * jj + 4] = P[12 * jj + 5] - v*P[12 * jj + 9];
				A[6 * jj + 5] = P[12 * jj + 6] - v*P[12 * jj + 10];
				B[2 * jj + 0] = u*P[12 * jj + 11] - P[12 * jj + 3];
				B[2 * jj + 1] = v*P[12 * jj + 11] - P[12 * jj + 7];
			}

			QR_Solution_Double(A, B, 2 * nview, 3);

			WC[ii].x = B[0];
			WC[ii].y = B[1];
			WC[ii].z = B[2];
		}
	}
	else
	{
		double mse = 0.0;
		double *At = new double[6 * nview];
		double *Bt = new double[2 * nview];
		double *t1 = new double[4 * nview*nview];
		double *t2 = new double[4 * nview*nview];
		double *Identity = new double[4 * nview*nview];
		double AtA[9], iAtA[9];
		for (ii = 0; ii < 4 * nview*nview; ii++)
			Identity[ii] = 0.0;
		for (ii = 0; ii < 2 * nview; ii++)
			Identity[ii + ii * 2 * nview] = 1.0;

		for (ii = 0; ii < npts; ii++)
		{
			for (jj = 0; jj < nview; jj++)
			{
				u = pts[ii + jj*npts].x, v = pts[ii + jj*npts].y;

				A[6 * jj + 0] = P[12 * jj] - u*P[12 * jj + 8];
				A[6 * jj + 1] = P[12 * jj + 1] - u*P[12 * jj + 9];
				A[6 * jj + 2] = P[12 * jj + 2] - u*P[12 * jj + 10];
				A[6 * jj + 3] = P[12 * jj + 4] - v*P[12 * jj + 8];
				A[6 * jj + 4] = P[12 * jj + 5] - v*P[12 * jj + 9];
				A[6 * jj + 5] = P[12 * jj + 6] - v*P[12 * jj + 10];
				B[2 * jj + 0] = u*P[12 * jj + 11] - P[12 * jj + 3];
				B[2 * jj + 1] = v*P[12 * jj + 11] - P[12 * jj + 7];
			}

			mat_transpose(A, At, nview * 2, 3);
			mat_transpose(B, Bt, nview * 2, 1);
			mat_mul(At, A, AtA, 3, 2 * nview, 3);
			mat_invert(AtA, iAtA);
			mat_mul(A, iAtA, t1, 2 * nview, 3, 3);
			mat_mul(t1, At, t2, 2 * nview, 3, 2 * nview);
			mat_subtract(Identity, t2, t1, 2 * nview, 2 * nview);
			mat_mul(Bt, t1, t2, 1, 2 * nview, 2 * nview);
			mat_mul(Bt, t2, t1, 1, 2 * nview, 1);
			mse = t1[0] / (2 * nview - 3);

			for (jj = 0; jj < 3; jj++)
				for (kk = 0; kk < 3; kk++)
					Cov[kk + jj * 3] = iAtA[kk + jj * 3] * mse;

			QR_Solution_Double(A, B, 2 * nview, 3);

			WC[ii].x = B[0];
			WC[ii].y = B[1];
			WC[ii].z = B[2];
		}

		delete[]At;
		delete[]Bt;
		delete[]t1;
		delete[]t2;
		delete[]Identity;
	}

	if (MenCreated)
		delete[]A, delete[]B;

	return;
}
void NviewTriangulationRANSAC(Point2d *pts, double *P, Point3d *WC, bool *PassedTri, vector<int> *Inliers, int nview, int npts, int MaxRanSacIter, double inlierPercent, double threshold, double *A, double *B, double *tP)
{
	int ii, jj, kk, ll, goodCount, bestCount;
	double u, v;
	Point2d _pt;
	Point3d t3D, b3D;
	bool MenCreated = false;
	if (A == NULL)
	{
		MenCreated = true;
		A = new double[6 * nview];
		B = new double[2 * nview];
		tP = new double[12 * nview];
	}
	int *GoodViewID = new int[nview];
	int *BestViewID = new int[nview];
	Point2d *goodpts2d = new Point2d[nview];
	Point2d *goodpts2dbk = new Point2d[nview];

	for (ii = 0; ii < npts; ii++)
	{
		//Pick a random pair to triangulate
		Point pair;
		bestCount = 0;
		for (kk = 0; kk < MaxRanSacIter; kk++)
		{
			pair.x = int(UniformNoise(nview, 0));
			while (true)
			{
				pair.y = int(UniformNoise(nview, 0));
				if (pair.y != pair.x)
					break;
			}
			Stereo_Triangulation(&pts[ii + pair.x*npts], &pts[ii + pair.y*npts], &P[12 * pair.x], &P[12 * pair.y], &t3D);

			//Project to other views
			goodCount = 0;
			for (jj = 0; jj < nview; jj++)
			{
				_pt = pts[ii + jj*npts];
				ProjectandDistort(t3D, &_pt, &P[12 * jj], NULL, NULL, 1);
				if (abs(_pt.x - pts[ii + jj*npts].x) < threshold && abs(_pt.y - pts[ii + jj*npts].y) < threshold)
					GoodViewID[jj] = true, goodCount++;
				else
					GoodViewID[jj] = false;
			}

			if (goodCount > bestCount)
			{
				bestCount = goodCount, b3D = t3D;
				for (ll = 0; ll<nview; ll++)
					BestViewID[ll] = GoodViewID[ll];
			}
			if (bestCount> nview*2.0*inlierPercent)
				break;
		}

		int t = nview*inlierPercent;
		if (bestCount >= max(t, 2))
		{
			int count = 0; Inliers[ii].reserve(nview);
			for (jj = 0; jj < nview; jj++)
			{
				if (!BestViewID[jj])
				{
					pts[ii + jj*npts].x = 0.0, pts[ii + jj*npts].y = 0.0;
					Inliers[ii].push_back(0);
					continue;
				}
				Inliers[ii].push_back(1);

				u = pts[ii + jj*npts].x, v = pts[ii + jj*npts].y;
				goodpts2d[count] = pts[jj], goodpts2dbk[count] = pts[jj];
				for (ll = 0; ll < 12; ll++)
					tP[12 * count + ll] = P[12 * jj + ll];
				count++;
			}

			for (jj = 0; jj < count; jj++)
			{
				u = goodpts2d[jj].x, v = goodpts2d[jj].y;
				A[6 * jj + 0] = tP[12 * jj] - u*tP[12 * jj + 8];
				A[6 * jj + 1] = tP[12 * jj + 1] - u*tP[12 * jj + 9];
				A[6 * jj + 2] = tP[12 * jj + 2] - u*tP[12 * jj + 10];
				A[6 * jj + 3] = tP[12 * jj + 4] - v*tP[12 * jj + 8];
				A[6 * jj + 4] = tP[12 * jj + 5] - v*tP[12 * jj + 9];
				A[6 * jj + 5] = tP[12 * jj + 6] - v*tP[12 * jj + 10];
				B[2 * jj + 0] = u*tP[12 * jj + 11] - tP[12 * jj + 3];
				B[2 * jj + 1] = v*tP[12 * jj + 11] - tP[12 * jj + 7];
			}
			QR_Solution_Double(A, B, 2 * bestCount, 3);
			WC[ii].x = B[0], WC[ii].y = B[1], WC[ii].z = B[2];

			ProjectandDistort(WC[ii], goodpts2d, tP, NULL, NULL, count);

			double error = 0.0;
			for (jj = 0; jj < count; jj++)
				error += pow(goodpts2dbk[jj].x - goodpts2d[jj].x, 2) + pow(goodpts2dbk[jj].y - goodpts2d[jj].y, 2);
			error = sqrt(error / count);
			if (error < threshold)
				PassedTri[ii] = true;
			else
				PassedTri[ii] = false;
		}
		else
			PassedTri[ii] = false;
	}

	delete[]GoodViewID;
	delete[]BestViewID;
	delete[]goodpts2d;
	delete[]goodpts2dbk;
	if (MenCreated)
		delete[]A, delete[]B, delete[]tP;
	return;
}
void NviewTriangulationRANSAC(vector<Point2d> *pts, double *P, Point3d *WC, int nview, int npts, int MaxRanSacIter, double inlierPercent, double threshold, double *A, double *B)
{
	int ii, jj, kk, ll;
	double u, v;
	Point2d _pt;
	Point3d t3D, b3D;
	bool MenCreated = false;
	if (A == NULL)
	{
		MenCreated = true;
		A = new double[6 * nview];
		B = new double[2 * nview];
	}
	int *GoodViewID = new int[nview];
	int *BestViewID = new int[nview];
	Point2d *goodpts2d = new Point2d[nview];
	if (MaxRanSacIter < 30)
		MaxRanSacIter = 30;

	for (ii = 0; ii < npts; ii++)
	{
		//Pick a random pair to triangulate
		int pair[2], bestCount = 0;
		for (kk = 0; kk < MaxRanSacIter; kk++)
		{
			pair[0] = int(UniformNoise(nview, 0));
			while (true)
			{
				pair[1] = int(UniformNoise(nview, 0));
				if (pair[1] != pair[0])
					break;
			}

			Stereo_Triangulation(&pts[ii].at(pair[0]), &pts[ii].at(pair[1]), &P[12 * pair[0]], &P[12 * pair[1]], &t3D);

			//Project to other views
			int goodCount = 0;
			for (jj = 0; jj < nview; jj++)
			{
				_pt = pts[ii].at(jj);
				ProjectandDistort(t3D, &_pt, &P[12 * jj], NULL, NULL, 1);
				if (abs(_pt.x - pts[ii].at(jj).x) < threshold || abs(_pt.y - pts[ii].at(jj).y) < threshold)
					GoodViewID[jj] = true, goodCount++;
				else
					GoodViewID[jj] = false;
			}

			if (goodCount > bestCount)
			{
				bestCount = goodCount, b3D = t3D;
				for (ll = 0; ll<nview; ll++)
					BestViewID[ll] = GoodViewID[ll];
			}
			if (bestCount> nview*inlierPercent)
				break;
		}

		int count = 0;
		for (jj = 0; jj < nview; jj++)
		{
			if (!BestViewID[jj])
			{
				pts[ii].at(jj).x = 0.0, pts[ii].at(jj).y = 0.0;
				continue;
			}

			u = pts[ii].at(jj).x, v = pts[ii].at(jj).y;
			goodpts2d[count] = pts[ii].at(jj);
			count++;

			A[6 * jj + 0] = P[12 * jj] - u*P[12 * jj + 8];
			A[6 * jj + 1] = P[12 * jj + 1] - u*P[12 * jj + 9];
			A[6 * jj + 2] = P[12 * jj + 2] - u*P[12 * jj + 10];
			A[6 * jj + 3] = P[12 * jj + 4] - v*P[12 * jj + 8];
			A[6 * jj + 4] = P[12 * jj + 5] - v*P[12 * jj + 9];
			A[6 * jj + 5] = P[12 * jj + 6] - v*P[12 * jj + 10];
			B[2 * jj + 0] = u*P[12 * jj + 11] - P[12 * jj + 3];
			B[2 * jj + 1] = v*P[12 * jj + 11] - P[12 * jj + 7];
		}
	}

	delete[]GoodViewID;
	delete[]BestViewID;
	if (MenCreated)
		delete[]A, delete[]B;
	return;
}
int MultiViewQualityCheck(Point2d *Pts, double *Pmat, double *K, double *distortion, bool *PassedPoints, int nviews, int npts, double thresh, Point2d *apts, Point2d *bkapts, int *DeviceMask, double *tK, double *tdistortion, double *tP, double *A, double *B)
{
	//Pts: [pts1; pts2; ...; ptsN]: (npts X nviews) matrix
	int ii, jj, kk, devCount;
	Point3d WC;
	bool createMem = false;
	double error;

	if (apts == NULL)
	{
		createMem = true;
		apts = new Point2d[nviews], bkapts = new Point2d[nviews], DeviceMask = new int[nviews];
		tK = new double[9 * nviews], tdistortion = new double[13 * nviews], tP = new double[12 * nviews];
	}

	for (ii = 0; ii < npts; ii++)
	{
		devCount = 0;
		for (jj = 0; jj < nviews; jj++)
		{
			DeviceMask[jj] = 1;
			if (Pts[jj + ii*nviews].x < 1 || Pts[jj + ii*nviews].y < 1)
				DeviceMask[jj] = 0;
			else
			{
				bkapts[devCount].x = Pts[jj + ii*nviews].x, bkapts[devCount].y = Pts[jj + ii*nviews].y;
				apts[devCount].x = bkapts[devCount].x, apts[devCount].y = bkapts[devCount].y;
				LensCorrectionPoint(&apts[devCount], K + 9 * jj, distortion + 13 * jj);

				for (kk = 0; kk < 12; kk++)
					tP[12 * devCount + kk] = Pmat[12 * jj + kk];
				for (kk = 0; kk < 9; kk++)
					tK[9 * devCount + kk] = K[9 * jj + kk];
				for (kk = 0; kk < 13; kk++)
					tdistortion[13 * devCount + kk] = distortion[13 * jj + kk];
				devCount++;
			}
		}

		NviewTriangulation(apts, tP, &WC, devCount, 1, NULL, A, B);
		ProjectandDistort(WC, apts, tP, tK, tdistortion, devCount);

		error = 0.0;
		for (jj = 0; jj < devCount; jj++)
			error += pow(bkapts[jj].x - apts[jj].x, 2) + pow(bkapts[jj].y - apts[jj].y, 2);
		error = sqrt(error / devCount);
		if (error < thresh)
			PassedPoints[ii] = true;
		else
			PassedPoints[ii] = false;
	}

	if (createMem)
	{
		delete[]apts, delete[]bkapts, delete[]DeviceMask;
		delete[]tK, delete[]tdistortion, delete[]tP;
	}

	return 0;
}
void NviewTriangulationRANSACDriver(CameraData *AllViewsInfo, vector<int>Selected3DIndex, vector<int> *nSelectedViews, vector<Point2d> *Selected2D, int nviews)
{
	double *A = new double[6 * nviews * 2];
	double *B = new double[2 * nviews * 2];
	double *P = new double[12 * nviews * 2];
	double *tP = new double[12 * nviews * 2];
	int *BestViewID = new int[nviews * 2];
	bool *passedTri = new bool[nviews * 2];

	srand(1);
	Point3d t3D;
	for (int ii = 0; ii < Selected3DIndex.size(); ii++)
	{
		int nviewsII = nSelectedViews[ii].size();
		if (nviewsII <= 2 * nviews)
		{
			for (int jj = 0; jj < nviewsII; jj++)
			{
				int viewID = nSelectedViews[ii].at(jj);
				for (int kk = 0; kk < 12; kk++)
					P[12 * jj + kk] = AllViewsInfo[viewID].P[kk];
			}

			//NviewTriangulationRANSAC(&Selected2D[ii], P, &t3D, passedTri, BestViewID, nviewsII, 1, (int)nChoosek(nviewsII, 2), 0.7, AllViewsInfo[0].threshold, A, B, tP);
		}
		else
		{
			double *A = new double[6 * nviewsII];
			double *B = new double[2 * nviewsII];
			double *P = new double[12 * nviewsII];
			for (int jj = 0; jj < nviewsII; jj++)
			{
				int viewID = nSelectedViews[ii].at(jj);
				for (int kk = 0; kk < 12; kk++)
					P[12 * jj + kk] = AllViewsInfo[viewID].P[kk];
			}
			//NviewTriangulationRANSAC(&Selected2D[ii], P, &t3D, passedTri, BestViewID, nviewsII, 1, (int)nChoosek(nviewsII, 2), 0.7, AllViewsInfo[0].threshold, A, B, tP);

			delete[]A, delete[]B, delete[]P;
		}

	}
	delete[]A, delete[]B, delete[]P, delete[]BestViewID;

	return;
}

int TwoCameraReconstruction(char *Path, CameraData *AllViewsInfo, int nviews, int timeID, vector<int> cumulativePts, vector<int>*PointCorres, vector<int> availViews, Point3d *ThreeD)
{
	vector<int>CorrespondencesID, SelectedIndex;
	vector<KeyPoint>keypoints1, keypoints2;
	CorrespondencesID.reserve(10000), SelectedIndex.reserve(10000), keypoints1.reserve(20000), keypoints1.reserve(20000);
	//GetPoint2DPairCorrespondence(Path, nviews, timeID, cumulativePts, availViews, keypoints1, keypoints2, CorrespondencesID, SelectedIndex, true);

	int npts = CorrespondencesID.size() / 2;
	Point2d *pts1 = new Point2d[npts], *pts2 = new Point2d[npts];
	for (int ii = 0; ii < npts; ii++)
	{
		int id1 = CorrespondencesID.at(2 * ii), id2 = CorrespondencesID.at(2 * ii + 1);
		pts1[ii].x = keypoints1.at(id1).pt.x, pts1[ii].y = keypoints1.at(id1).pt.y;
		pts2[ii].x = keypoints2.at(id2).pt.x, pts2[ii].y = keypoints2.at(id2).pt.y;
	}

	if (AllViewsInfo[availViews.at(0)].LensModel == RADIAL_TANGENTIAL_PRISM)
	{
		LensCorrectionPoint(pts1, AllViewsInfo[availViews.at(0)].K, AllViewsInfo[availViews.at(0)].distortion, npts);
		LensCorrectionPoint(pts2, AllViewsInfo[availViews.at(1)].K, AllViewsInfo[availViews.at(1)].distortion, npts);
	}

	Mat x1s(npts, 2, CV_64F), x2s(npts, 2, CV_64F);
	for (int ii = 0; ii < npts; ii++)
	{
		x1s.at<double>(ii, 0) = pts1[ii].x, x1s.at<double>(ii, 1) = pts1[ii].y;
		x2s.at<double>(ii, 0) = pts2[ii].x, x2s.at<double>(ii, 1) = pts2[ii].y;
	}

	Mat cvK1 = Mat(3, 3, CV_64F, AllViewsInfo[availViews.at(0)].K);
	Mat cvK2 = Mat(3, 3, CV_64F, AllViewsInfo[availViews.at(1)].K);

	Mat E = findEssentialMat(x1s, x2s, cvK1, cvK2, CV_RANSAC, 0.99, 1, 200, noArray());

	Mat R_5pt, rvec_5pt, tvec_5pt;
	recoverPose(E, x1s, x2s, R_5pt, tvec_5pt, cvK1, cvK2, noArray());
	//Rodrigues(R_5pt, rvec_5pt);
	//DisplayMatrix("5pts Algo rvec: ", rvec_5pt);
	//DisplayMatrix("5pts Algo tvec: ", tvec_5pt);

	Mat cvP1(3, 4, CV_64F), cvP2(3, 4, CV_64F);
	cvP1 = Mat::eye(3, 4, CV_64F); cvP1 = cvK1*cvP1;
	cvP2(Range::all(), Range(0, 3)) = R_5pt * 1.0; cvP2.col(3) = tvec_5pt * 1.0;  cvP2 = cvK2*cvP2;

	Mat cvThreeD;
	x1s = x1s.t(), x2s = x2s.t();
	triangulatePoints(cvP1, cvP2, x1s, x2s, cvThreeD);

	cvThreeD.row(0) /= cvThreeD.row(3);
	cvThreeD.row(1) /= cvThreeD.row(3);
	cvThreeD.row(2) /= cvThreeD.row(3);
	cvThreeD.row(3) /= cvThreeD.row(3);

	bool *goodPoints = new bool[npts];
	Point3d *t3D = new Point3d[npts];
	for (int ii = 0; ii < npts; ii++)
	{
		t3D[ii].x = cvThreeD.at<double>(0, ii);
		t3D[ii].y = cvThreeD.at<double>(1, ii);
		t3D[ii].z = cvThreeD.at<double>(2, ii);
	}

	double P1[12], P2[12];
	for (int ii = 0; ii < 12; ii++)
		P1[ii] = cvP1.at<double>(ii), P2[ii] = cvP2.at<double>(ii);

	double threshold = AllViewsInfo[0].threshold;
	TwoViewTriangulationQualityCheck(pts1, pts2, t3D, P1, P2, NULL, NULL, NULL, NULL, goodPoints, npts, threshold);

	int count = 0;
	for (int ii = 0; ii < npts; ii++)
	{
		if (goodPoints[ii])
		{
			count++;
			int id = SelectedIndex.at(ii);
			ThreeD[id].x = cvThreeD.at<double>(0, ii);
			ThreeD[id].y = cvThreeD.at<double>(1, ii);
			ThreeD[id].z = cvThreeD.at<double>(2, ii);
			for (int jj = 0; jj < PointCorres[id].size(); jj++)
				ThreeD[PointCorres[id].at(jj)] = ThreeD[id];
		}
	}

	AllViewsInfo[availViews.at(0)].R[0] = 1.0, AllViewsInfo[availViews.at(0)].R[1] = 0.0, AllViewsInfo[availViews.at(0)].R[2] = 0.0, AllViewsInfo[availViews.at(0)].T[0] = 0.0;
	AllViewsInfo[availViews.at(0)].R[3] = 0.0, AllViewsInfo[availViews.at(0)].R[4] = 1.0, AllViewsInfo[availViews.at(0)].R[5] = 0.0, AllViewsInfo[availViews.at(0)].T[1] = 0.0;
	AllViewsInfo[availViews.at(0)].R[6] = 0.0, AllViewsInfo[availViews.at(0)].R[7] = 0.0, AllViewsInfo[availViews.at(0)].R[8] = 1.0, AllViewsInfo[availViews.at(0)].T[2] = 0.0;

	for (int ii = 0; ii < 9; ii++)
		AllViewsInfo[availViews.at(1)].R[ii] = R_5pt.at<double>(ii);
	for (int ii = 0; ii < 3; ii++)
		AllViewsInfo[availViews.at(1)].T[ii] = tvec_5pt.at<double>(ii);

	//Update ViewParas
	GetrtFromRT(AllViewsInfo, availViews);
	GetIntrinsicFromK(AllViewsInfo, availViews);
	for (int ii = 0; ii < availViews.size(); ii++)
		AssembleP(AllViewsInfo[availViews.at(ii)].K, AllViewsInfo[availViews.at(ii)].R, AllViewsInfo[availViews.at(ii)].T, AllViewsInfo[availViews.at(ii)].P);

	/*FILE *fp = fopen("C:/temp/2d3D.txt", "w+");
	for (int ii = 0; ii < CorrespondencesID.size() / 2; ii++)
	if (goodPoints[ii])
	fprintf(fp, "%d %d\n", CorrespondencesID.at(2 * ii), SelectedIndex.at(ii));
	fclose(fp);

	fp = fopen("C:/temp/2d3dCorres.txt", "w+");
	for (int ii = 0; ii < npts; ii++)
	if (goodPoints[ii])
	fprintf(fp, "%.6f %.6f %.6f %.6f %.6f\n", pts1[ii].x, pts1[ii].y, cvThreeD.at<double>(0, ii), cvThreeD.at<double>(1, ii), cvThreeD.at<double>(2, ii));
	fclose(fp);*/

	delete[]pts1, delete[]pts2, delete[]t3D, delete[]goodPoints;

	if (count < AllViewsInfo[0].ninlierThresh)
	{
		printf("Stiching from %d to %d fails due to low number of inliers (%d)\n", availViews.at(0), availViews.at(1), count);
		return 1;
	}
	else
		return 0;
}
void DetermineDevicePose(double *K, double *distortion, int LensModel, double *R, double *T, Point2d *pts, Point3d *ThreeD, int npts, double thresh, int &ninliers)
{
	int ii;
	Mat cvpts(npts, 2, CV_32F), cv3D(npts, 3, CV_32F);

	if (distortion != NULL && LensModel == RADIAL_TANGENTIAL_PRISM)
		LensCorrectionPoint(pts, K, distortion, npts);

	for (ii = 0; ii < npts; ii++)
	{
		cvpts.at<float>(ii, 0) = pts[ii].x, cvpts.at<float>(ii, 1) = pts[ii].y;
		cv3D.at<float>(ii, 0) = ThreeD[ii].x, cv3D.at<float>(ii, 1) = ThreeD[ii].y, cv3D.at<float>(ii, 2) = ThreeD[ii].z;
	}

	/*FILE *fp = fopen("C:/temp/_2d3dCorres.txt", "w+");
	for (int ii = 0; ii < npts; ii++)
	fprintf(fp, "%.6f %.6f %.6f %.6f %.6f\n", cvpts.at<float>(ii, 0), cvpts.at<float>(ii, 1), cv3D.at<float>(ii,0), cv3D.at<float>(ii,1), cv3D.at<float>(ii,2));
	fclose(fp);*/

	Mat cvK = Mat(3, 3, CV_32F), rvec(1, 3, CV_32F), tvec(1, 3, CV_32F);
	for (ii = 0; ii < 9; ii++)
		cvK.at<float>(ii) = (float)K[ii];

	Mat Inliers;
	double ProThresh = 0.995, PercentInlier = 0.4;
	int iterMax = (int)(log(1.0 - ProThresh) / log(1.0 - pow(PercentInlier, 4)) + 0.5); //log(1-eps) / log(1 - (inlier%)^min_pts_requires)
	solvePnPRansac(cv3D, cvpts, cvK, Mat(), rvec, tvec, false, iterMax, thresh, npts*PercentInlier, Inliers, CV_EPNP);// CV_ITERATIVE);

	ninliers = Inliers.rows;
	//cout << rvec << endl;
	//cout << tvec << endl;

	Mat Rmat(3, 3, CV_64F);
	Rodrigues(rvec, Rmat);
	for (ii = 0; ii < 9; ii++)
		R[ii] = Rmat.at<double>(ii);
	for (ii = 0; ii < 3; ii++)
		T[ii] = tvec.at<double>(ii);

	return;
}

int AddNewViewReconstruction(char *Path, CameraData *AllViewsInfo, int nviews, int timeID, vector<int> cumulativePts, vector<int>*PointCorres, Point3d *All3D, double threshold, vector<int> &availViews)
{
	int ii, jj, kk, ll;

	//Determine next view with highest number of correspondences
	int maxPoints = 0;
	vector<int>viewID; viewID.reserve(2); viewID.push_back(0), viewID.push_back(0);
	vector<int>checkedViews; checkedViews.reserve(100);
	for (ii = 0; ii < availViews.size(); ii++)
	{
		for (jj = 0; jj<checkedViews.size(); jj++)
		{
			if (availViews.at(ii) == checkedViews.at(jj))
				break;
		}

		if (jj == checkedViews.size())
		{
			checkedViews.push_back(availViews.at(ii));
			kk = NextViewFinder(Path, nviews, timeID, availViews.at(ii), ll, availViews);
			if (ll>maxPoints)
				maxPoints = ll, viewID.at(0) = availViews.at(ii), viewID.at(1) = kk;
		}
	}
	availViews.push_back(viewID.at(1));
	sort(availViews.begin(), availViews.end());
	printf("Adding view %d to the list...", viewID.at(1));

	bool SwapView = viewID.at(0) < viewID.at(1) ? false : true;
	sort(viewID.begin(), viewID.end());

	//Get correspondences and their indices
	vector<KeyPoint> keypoints1, keypoints2;
	vector<int>TwoDcorrespondencesID, ThreeDCorrespondencesID, SelectedIndex;
	keypoints1.reserve(10000), keypoints2.reserve(10000);
	TwoDcorrespondencesID.reserve(10000), ThreeDCorrespondencesID.reserve(10000), SelectedIndex.reserve(10000);

	GetPoint3D2DPairCorrespondence(Path, nviews, timeID, cumulativePts, viewID, All3D, keypoints1, keypoints2, TwoDcorrespondencesID, ThreeDCorrespondencesID, SelectedIndex, SwapView);
	if (ThreeDCorrespondencesID.size() / 2 < AllViewsInfo[0].ninlierThresh)
	{
		printf("Stiching from %d to %d fails due to low number of inliers (%d)\n", viewID.at(0), viewID.at(1), ThreeDCorrespondencesID.size() / 2);
		return 1;
	}

	//Run PnP for pose estimation
	int npts = ThreeDCorrespondencesID.size() / 2;
	Point2d *pts = new Point2d[npts];
	Point3d *t3D = new Point3d[npts];

	for (ii = 0; ii < npts; ii++)
	{
		int id1 = ThreeDCorrespondencesID.at(2 * ii), id2 = ThreeDCorrespondencesID.at(2 * ii + 1);
		if (SwapView)
			pts[ii].x = keypoints1.at(id1).pt.x, pts[ii].y = keypoints1.at(id1).pt.y;
		else
			pts[ii].x = keypoints2.at(id1).pt.x, pts[ii].y = keypoints2.at(id1).pt.y;
		t3D[ii] = All3D[id2];
	}

	if (AllViewsInfo[viewID.at(1)].LensModel == RADIAL_TANGENTIAL_PRISM)
	{
		if (SwapView)
			LensCorrectionPoint(pts, AllViewsInfo[viewID.at(0)].K, AllViewsInfo[viewID.at(0)].distortion, npts);
		else
			LensCorrectionPoint(pts, AllViewsInfo[viewID.at(1)].K, AllViewsInfo[viewID.at(1)].distortion, npts);
	}

	int ninliers;
	if (SwapView)
		DetermineDevicePose(AllViewsInfo[viewID.at(0)].K, NULL, AllViewsInfo[viewID.at(0)].LensModel, AllViewsInfo[viewID.at(0)].R, AllViewsInfo[viewID.at(0)].T, pts, t3D, npts, AllViewsInfo[0].threshold * 2, ninliers);
	else
		DetermineDevicePose(AllViewsInfo[viewID.at(1)].K, NULL, AllViewsInfo[viewID.at(1)].LensModel, AllViewsInfo[viewID.at(1)].R, AllViewsInfo[viewID.at(1)].T, pts, t3D, npts, AllViewsInfo[0].threshold * 2, ninliers);
	delete[]pts, delete[]t3D;

	//Triangulate new points
	npts = TwoDcorrespondencesID.size() / 2;
	Point2d *pts1 = new Point2d[npts], *pts2 = new Point2d[npts];
	for (int ii = 0; ii < npts; ii++)
	{
		int id1 = TwoDcorrespondencesID.at(2 * ii), id2 = TwoDcorrespondencesID.at(2 * ii + 1);
		pts1[ii].x = keypoints1.at(id1).pt.x, pts1[ii].y = keypoints1.at(id1).pt.y;
		pts2[ii].x = keypoints2.at(id2).pt.x, pts2[ii].y = keypoints2.at(id2).pt.y;
	}

	if (AllViewsInfo[viewID.at(0)].LensModel == RADIAL_TANGENTIAL_PRISM)
	{
		LensCorrectionPoint(pts1, AllViewsInfo[viewID.at(0)].K, AllViewsInfo[viewID.at(0)].distortion, npts);
		LensCorrectionPoint(pts2, AllViewsInfo[viewID.at(1)].K, AllViewsInfo[viewID.at(1)].distortion, npts);
	}

	double P1[12], P2[12], RT1[12], RT2[12];
	AssembleRT(AllViewsInfo[viewID.at(0)].R, AllViewsInfo[viewID.at(0)].T, RT1);
	AssembleRT(AllViewsInfo[viewID.at(1)].R, AllViewsInfo[viewID.at(1)].T, RT2);

	mat_mul(AllViewsInfo[viewID.at(0)].K, RT1, P1, 3, 3, 4);
	mat_mul(AllViewsInfo[viewID.at(1)].K, RT2, P2, 3, 3, 4);

	//Triangulate and remove bad points quality
	bool *goodPoints = new bool[npts];
	Point3d *_t3D = new Point3d[npts];
	Stereo_Triangulation(pts1, pts2, P1, P2, _t3D, npts);
	TwoViewTriangulationQualityCheck(pts1, pts2, _t3D, P1, P2, NULL, NULL, NULL, NULL, goodPoints, npts, threshold);

	int count = 0;
	for (int ii = 0; ii < npts; ii++)
	{
		if (goodPoints[ii])
		{
			count++;
			int id = SelectedIndex.at(ii);
			All3D[id] = _t3D[ii];
			for (int jj = 0; jj < PointCorres[id].size(); jj++)
				All3D[PointCorres[id].at(jj)] = All3D[id];
		}
	}

	//Update ViewParas
	GetrtFromRT(AllViewsInfo, availViews);
	GetIntrinsicFromK(AllViewsInfo, availViews);
	for (int ii = 0; ii < availViews.size(); ii++)
		AssembleP(AllViewsInfo[availViews.at(ii)].K, AllViewsInfo[availViews.at(ii)].R, AllViewsInfo[availViews.at(ii)].T, AllViewsInfo[availViews.at(ii)].P);

	delete[]pts1, delete[]pts2, delete[]_t3D, delete[]goodPoints;

	sort(availViews.begin(), availViews.end());
	if (count < AllViewsInfo[0].ninlierThresh)
	{
		printf("Stiching from %d to %d fails due to low number of inliers (%d)\n", viewID.at(0), viewID.at(1), ThreeDCorrespondencesID.size() / 2);
		return 1;
	}
	else
		return 0;
}

struct PinholeReprojectionError {
	PinholeReprojectionError(double observed_x, double observed_y) : observed_x(observed_x), observed_y(observed_y) {}
	template <typename T>	bool operator()(const T* const intrinsic, const T* const distortion, const T* const RT, const T* const point, T* residuals) const
	{
		// camera[0,1,2] are the angle-axis rotation.
		T p[3];
		ceres::AngleAxisRotatePoint(RT, point, p);

		// camera[3,4,5] are the translation.
		p[0] += RT[3], p[1] += RT[4], p[2] += RT[5];

		// Project to normalize coordinate
		T xcn = p[0] / p[2];
		T ycn = p[1] / p[2];

		// Apply second and fourth order radial distortion.
		T xcn2 = xcn*xcn, ycn2 = ycn*ycn, xycn = xcn*ycn, r2 = xcn2 + ycn2, r4 = r2*r2, r6 = r2*r4;
		T radial = T(1.0) + distortion[0] * r2 + distortion[1] * r4 + distortion[2] * r6;
		T tangentialX = T(2.0)*distortion[3] * xycn + distortion[4] * (r2 + T(2.0)*xcn2);
		T tangentailY = distortion[3] * (r2 + T(2.0)*ycn2) + T(2.0)*distortion[4] * xycn;
		T prismX = distortion[5] * r2;
		T prismY = distortion[6] * r2;
		T xcn_ = radial*xcn + tangentialX + prismX;
		T ycn_ = radial*ycn + tangentailY + prismY;

		// Compute final projected point position.
		T predicted_x = intrinsic[0] * xcn_ + intrinsic[2] * ycn_ + intrinsic[3];
		T predicted_y = intrinsic[1] * ycn_ + intrinsic[4];

		// The error is the difference between the predicted and observed position.
		residuals[0] = predicted_x - T(observed_x);
		residuals[1] = predicted_y - T(observed_y);

		return true;
	}
	static ceres::CostFunction* Create(const double observed_x, const double observed_y) {
		return (new ceres::AutoDiffCostFunction<PinholeReprojectionError, 2, 5, 7, 6, 3>(new PinholeReprojectionError(observed_x, observed_y)));
	}
	double observed_x, observed_y;
};
struct PinholeReprojectionError2 {
	PinholeReprojectionError2(double observed_x, double observed_y) : observed_x(observed_x), observed_y(observed_y) {}
	template <typename T>	bool operator()(const T* const fxfy, const T* const skew, const T* const uv0, const T* const Radial12, const T* const Tangential12, const T*const Radial3Prism, const T* const RT, const T* const point, T* residuals) const
	{
		// camera[0,1,2] are the angle-axis rotation.
		T p[3];
		ceres::AngleAxisRotatePoint(RT, point, p);

		// camera[3,4,5] are the translation.
		p[0] += RT[3], p[1] += RT[4], p[2] += RT[5];

		// Project to normalize coordinate
		T xcn = p[0] / p[2];
		T ycn = p[1] / p[2];

		// Apply second and fourth order radial distortion.
		T xcn2 = xcn*xcn, ycn2 = ycn*ycn, xycn = xcn*ycn, r2 = xcn2 + ycn2, r4 = r2*r2, r6 = r2*r4;
		T radial = T(1.0) + Radial12[0] * r2 + Radial12[1] * r4 + Radial3Prism[0] * r6;
		T tangentialX = T(2.0)*Tangential12[0] * xycn + Tangential12[1] * (r2 + T(2.0)*xcn2);
		T tangentailY = Tangential12[0] * (r2 + T(2.0)*ycn2) + T(2.0)*Tangential12[1] * xycn;
		T prismX = Radial3Prism[1] * r2;
		T prismY = Radial3Prism[2] * r2;
		T xcn_ = radial*xcn + tangentialX + prismX;
		T ycn_ = radial*ycn + tangentailY + prismY;

		// Compute final projected point position.
		T predicted_x = fxfy[0] * xcn_ + skew[0] * ycn_ + uv0[0];
		T predicted_y = fxfy[1] * ycn_ + uv0[1];

		// The error is the difference between the predicted and observed position.
		residuals[0] = predicted_x - T(observed_x);
		residuals[1] = predicted_y - T(observed_y);

		return true;
	}
	static ceres::CostFunction* Create(const double observed_x, const double observed_y) {
		return (new ceres::AutoDiffCostFunction<PinholeReprojectionError2, 2, 2, 1, 2, 2, 2, 3, 6, 3>(new PinholeReprojectionError2(observed_x, observed_y)));
	}
	double observed_x, observed_y;
};
void PinholeReprojectionDebug(double *intrinsic, double* distortion, double* rt, Point2d observed, Point3d Point, double *residuals)
{
	// camera[0,1,2] are the angle-axis rotation.
	double p[3];
	double point[3] = { Point.x, Point.y, Point.z };
	ceres::AngleAxisRotatePoint(rt, point, p);

	// camera[3,4,5] are the translation.
	p[0] += rt[3], p[1] += rt[4], p[2] += rt[5];

	// Project to image coordinate
	double xcn = p[0] / p[2], ycn = p[1] / p[2];
	Point2d uv(intrinsic[0] * xcn + intrinsic[2] * ycn + intrinsic[3], intrinsic[1] * ycn + intrinsic[4]);

	// Deal with distortion
	double K[9] = { intrinsic[0], intrinsic[2], intrinsic[3], 0.0, intrinsic[1], intrinsic[4], 0.0, 0.0, 1.0 };
	double distortionParas[7] = { distortion[0], distortion[1], distortion[2], distortion[3], distortion[4], distortion[5], distortion[6] };

	LensDistortionPoint(&uv, K, distortionParas);

	// The error is the difference between the predicted and observed position.
	residuals[0] = uv.x - observed.x, residuals[1] = uv.y - observed.y;

	return;
}
int IncrementalBA(char *Path, int nviews, int timeID, CameraData *AllViewsInfo, vector<int> availViews, vector<int>*PointCorres, vector<int>mask, vector<int> Selected3DIndex, Point3d *All3D, vector<Point2d> *selected2D, vector<int>*nSelectedViews, int nSelectedPts, int totalPts, bool fixIntrinsic, bool fixDistortion, bool showReProjectionError, bool debug)
{
	char Fname[200]; FILE *fp = 0;
	int ii, jj, match, id3d, viewID, npts = Selected3DIndex.size();
	double residuals[2];

	double *seleted3D = new double[npts * 3];
	for (ii = 0; ii < npts; ii++)
	{
		id3d = Selected3DIndex.at(ii);
		seleted3D[3 * ii] = All3D[id3d].x, seleted3D[3 * ii + 1] = All3D[id3d].y, seleted3D[3 * ii + 2] = All3D[id3d].z;
	}

	printf("Set up BA ...");
	ceres::Problem problem;

	if (debug)
		sprintf(Fname, "C:/temp/reprojectionB_%d.txt", availViews.size()), fp = fopen(Fname, "w+");

	bool *discard3Dpoint = new bool[npts];
	vector<bool> *notGood = new vector<bool>[npts];
	for (int jj = 0; jj < npts; jj++)
		discard3Dpoint[jj] = false, notGood[jj].reserve(nSelectedViews[jj].size());

	vector<int>::iterator it;
	for (int jj = 0; jj < npts; jj++)
	{
		id3d = Selected3DIndex.at(jj);
		if (abs(All3D[id3d].x) > LIMIT3D)
		{
			it = find(mask.begin(), mask.end(), id3d);
			if (it != mask.end())
				continue; //the parent of the points has been processed

			//screening: if there are only 2 points and 1 of them fails, discard the pair
			for (ii = 0; ii < nSelectedViews[jj].size(); ii++)
			{
				if (selected2D[jj].at(ii).x < 1 || selected2D[jj].at(ii).y < 1)
				{
					notGood[jj].push_back(false);
					continue;
				}

				viewID = nSelectedViews[jj].at(ii);
				PinholeReprojectionDebug(AllViewsInfo[viewID].intrinsic, AllViewsInfo[viewID].distortion, AllViewsInfo[viewID].rt, Point2d(selected2D[jj].at(ii).x, selected2D[jj].at(ii).y), All3D[id3d], residuals);
				if (abs(residuals[0]) > 3 * AllViewsInfo[0].threshold || abs(residuals[1]) > 3 * AllViewsInfo[0].threshold)
					notGood[jj].push_back(false);
				else
					notGood[jj].push_back(true);
			}

			//Discard point 
			int count = 0;
			for (ii = 0; ii < nSelectedViews[jj].size(); ii++)
				if (notGood[jj].at(ii) == true)
					count++;

			discard3Dpoint[jj] = false;
			if (count < 2)
			{
				discard3Dpoint[jj] = true;
				continue;
			}

			//All good, add point and its 2D projections to Ceres
			bool once = true;
			for (ii = 0; ii < nSelectedViews[jj].size(); ii++)
			{
				if (!notGood[jj].at(ii))
					continue;

				viewID = nSelectedViews[jj].at(ii);
				ceres::CostFunction* cost_function = PinholeReprojectionError::Create(selected2D[jj].at(ii).x, selected2D[jj].at(ii).y);
				problem.AddResidualBlock(cost_function, NULL, AllViewsInfo[viewID].intrinsic, AllViewsInfo[viewID].distortion, AllViewsInfo[viewID].rt, &seleted3D[3 * jj]);

				if (debug)
				{
					PinholeReprojectionDebug(AllViewsInfo[viewID].intrinsic, AllViewsInfo[viewID].distortion, AllViewsInfo[viewID].rt, Point2d(selected2D[jj].at(ii).x, selected2D[jj].at(ii).y), All3D[id3d], residuals);
					if (once)
					{
						once = false;
						fprintf(fp, "%d %.4f %.4f %.4f ", id3d, All3D[id3d].x, All3D[id3d].y, All3D[id3d].z);
					}
					fprintf(fp, "%d %.4f %.4f %.4f %.4f ", viewID, selected2D[jj].at(ii).x, selected2D[jj].at(ii).y, residuals[0], residuals[1]);
				}
			}
			if (!once)
				fprintf(fp, "\n");
		}
	}
	if (debug)
		fclose(fp);

	//Set up constant parameters:
	printf("...set up fixed parameters ...");
	for (int ii = 0; ii < availViews.size(); ii++)
	{
		int viewID = availViews.at(ii);
		if (fixIntrinsic)
			problem.SetParameterBlockConstant(AllViewsInfo[viewID].intrinsic);
		if (fixDistortion)
			problem.SetParameterBlockConstant(AllViewsInfo[viewID].distortion);
	}

	printf("...run BA...\n");
	ceres::Solver::Options options;
	options.num_threads = 1;
	options.max_num_iterations = 100;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = false;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.BriefReport() << "\n";

	GetKFromIntrinsic(AllViewsInfo, availViews);
	GetRTFromrt(AllViewsInfo, availViews);
	for (int ii = 0; ii < availViews.size(); ii++)
		AssembleP(AllViewsInfo[availViews.at(ii)].K, AllViewsInfo[availViews.at(ii)].R, AllViewsInfo[availViews.at(ii)].T, AllViewsInfo[availViews.at(ii)].P);

	//2d points belong to 1 3D point-> distribute 3D to its 2d matches
	for (ii = 0; ii < npts; ii++)
	{
		id3d = Selected3DIndex.at(ii);
		All3D[id3d].x = seleted3D[3 * ii], All3D[id3d].y = seleted3D[3 * ii + 1], All3D[id3d].z = seleted3D[3 * ii + 2];
		for (jj = 0; jj < PointCorres[id3d].size(); jj++)
		{
			match = PointCorres[id3d].at(jj);
			All3D[match] = All3D[id3d];
		}
	}

	vector<double> ReProjectionError; ReProjectionError.reserve(npts);
	if (debug || showReProjectionError)
	{
		if (debug)
			sprintf(Fname, "C:/temp/reprojectionA_%d.txt", availViews.size()), fp = fopen(Fname, "w+");
		for (int jj = 0; jj < npts; jj++)
		{
			id3d = Selected3DIndex.at(jj);
			if (abs(All3D[id3d].x) > LIMIT3D && !discard3Dpoint[jj])
			{
				it = find(mask.begin(), mask.end(), id3d);
				if (it != mask.end())
					continue; //the parent of the points has been processed

				bool once = true;
				int validViewcount = 0;
				double pointErr = 0.0;
				for (int ii = 0; ii < nSelectedViews[jj].size(); ii++)
				{
					if (!notGood[jj].at(ii))
						continue;

					viewID = nSelectedViews[jj].at(ii);
					PinholeReprojectionDebug(AllViewsInfo[viewID].intrinsic, AllViewsInfo[viewID].distortion, AllViewsInfo[viewID].rt, Point2d(selected2D[jj].at(ii).x, selected2D[jj].at(ii).y), All3D[id3d], residuals);

					validViewcount++;
					pointErr += residuals[0] * residuals[0] + residuals[1] * residuals[1];
					if (once && debug)
					{
						once = false;
						fprintf(fp, "%d %.4f %.4f %.4f ", id3d, All3D[id3d].x, All3D[id3d].y, All3D[id3d].z);
					}
					if (debug)
						fprintf(fp, "%d %.4f %.4f %.4f %.4f ", viewID, selected2D[jj].at(ii).x, selected2D[jj].at(ii).y, residuals[0], residuals[1]);
				}
				if (!once &&debug)
					fprintf(fp, "\n");

				ReProjectionError.push_back(sqrt(pointErr / validViewcount));
			}
		}
		if (debug)
			fclose(fp);

		if (debug)
			sprintf(Fname, "C:/temp/visSfm_%d.txt", availViews.size()), fp = fopen(Fname, "w+");
		for (int jj = 0; jj < npts; jj++)
		{
			id3d = Selected3DIndex.at(jj);
			if (abs(All3D[id3d].x) > LIMIT3D && !discard3Dpoint[jj])
			{
				it = find(mask.begin(), mask.end(), id3d);
				if (it != mask.end())
					continue; //the parent of the points has been processed

				bool once = true;
				int validViewcount = 0;
				double pointErr = 0.0;
				for (int ii = 0; ii < nSelectedViews[jj].size(); ii++)
				{
					if (!notGood[jj].at(ii))
						continue;
					validViewcount++;
				}
				for (int ii = 0; ii < nSelectedViews[jj].size(); ii++)
				{
					if (!notGood[jj].at(ii))
						continue;

					viewID = nSelectedViews[jj].at(ii);
					PinholeReprojectionDebug(AllViewsInfo[viewID].intrinsic, AllViewsInfo[viewID].distortion, AllViewsInfo[viewID].rt, Point2d(selected2D[jj].at(ii).x, selected2D[jj].at(ii).y), All3D[id3d], residuals);

					pointErr += residuals[0] * residuals[0] + residuals[1] * residuals[1];
					if (once && debug)
					{
						once = false;
						fprintf(fp, "%.4f %.4f %.4f 0  255 0 %d ", All3D[id3d].x, All3D[id3d].y, All3D[id3d].z, validViewcount);
					}
					if (debug)
						fprintf(fp, "%d %d %.4f %.4f ", viewID, (int)(UniformNoise(10000, 0)), selected2D[jj].at(ii).x - 1536, selected2D[jj].at(ii).y - 1024);
				}
				if (!once &&debug)
					fprintf(fp, "\n");
			}
		}
		if (debug)
			fclose(fp);
	}

	if (showReProjectionError)
	{
		double mini = *min_element(ReProjectionError.begin(), ReProjectionError.end());
		double maxi = *max_element(ReProjectionError.begin(), ReProjectionError.end());
		double avg = MeanArray(ReProjectionError);
		double std = sqrt(VarianceArray(ReProjectionError, avg));
		printf("Reprojection error: %.2f %.2f %.2f %.2f\n", mini, maxi, avg, std);
	}

	delete[]discard3Dpoint, delete[]notGood;
	return 0;
}
void IncrementalBundleAdjustment(char *Path, int nviews, int timeID, int maxKeypoints)
{
	int totalPts;
	vector<int> cumulativePts;
	ReadCumulativePoints(Path, nviews, timeID, cumulativePts);
	totalPts = cumulativePts.at(nviews);

	vector<int>CeresDuplicateAddInMask;
	vector<int>*PointCorres = new vector<int>[totalPts];
	//vector<int>PointCorres[191872];
	ReadPointCorrespondences(Path, nviews, timeID, PointCorres, CeresDuplicateAddInMask, totalPts);

	int viewPair[2];
	BestPairFinder(Path, nviews, timeID, viewPair);

	vector<int> availViews; availViews.reserve(nviews);
	availViews.push_back(viewPair[0]), availViews.push_back(viewPair[1]);
	sort(availViews.begin(), availViews.end());

	int nSelectedPts;
	vector<int>Selected3DIndex; Selected3DIndex.reserve(totalPts);
	vector<Point2d> *Selected2D = new vector<Point2d>[totalPts];
	vector<int> *nSelectedViews = new vector<int>[totalPts];

	CameraData *AllViewsInfo = new CameraData[nviews];
	if (ReadIntrinsicResults(Path, AllViewsInfo, nviews) != 0)
		return;
	for (int ii = 0; ii < nviews; ii++)
		AllViewsInfo[ii].LensModel = RADIAL_TANGENTIAL_PRISM, AllViewsInfo[ii].threshold = 2.0, AllViewsInfo[ii].ninlierThresh = 50;

	Point3d *All3D = new Point3d[totalPts];
	for (int ii = 0; ii < totalPts; ii++)
		All3D[ii].x = 0.0, All3D[ii].y = 0.0, All3D[ii].z = 0.0;

	TwoCameraReconstruction(Path, AllViewsInfo, nviews, timeID, cumulativePts, PointCorres, availViews, All3D);
	GetPoint3D2DAllCorrespondence(Path, nviews, timeID, cumulativePts, All3D, availViews, Selected3DIndex, Selected2D, nSelectedViews, nSelectedPts);
	NviewTriangulationRANSACDriver(AllViewsInfo, Selected3DIndex, nSelectedViews, Selected2D, nviews);
	IncrementalBA(Path, nviews, timeID, AllViewsInfo, availViews, PointCorres, CeresDuplicateAddInMask, Selected3DIndex, All3D, Selected2D, nSelectedViews, nSelectedPts, totalPts, true, true, true, false);

	int startnum = 2, addedDevices = startnum;
	for (int ii = 0; ii < nviews - startnum; ii++)
	{
		if (AddNewViewReconstruction(Path, AllViewsInfo, nviews, timeID, cumulativePts, PointCorres, All3D, AllViewsInfo[0].threshold, availViews) == 0)
		{
			printf("succeed!\n");
			addedDevices++;
		}

		if (addedDevices % 2 == 0) // Do BA after every 2 views being added
		{
			GetPoint3D2DAllCorrespondence(Path, nviews, timeID, cumulativePts, All3D, availViews, Selected3DIndex, Selected2D, nSelectedViews, nSelectedPts);
			NviewTriangulationRANSACDriver(AllViewsInfo, Selected3DIndex, nSelectedViews, Selected2D, nviews);
			IncrementalBA(Path, nviews, timeID, AllViewsInfo, availViews, PointCorres, CeresDuplicateAddInMask, Selected3DIndex, All3D, Selected2D, nSelectedViews, nSelectedPts, totalPts, true, true, true, false);
		}
	}
	printf("Done!\n");

	//Final BA
	GetPoint3D2DAllCorrespondence(Path, nviews, timeID, cumulativePts, All3D, availViews, Selected3DIndex, Selected2D, nSelectedViews, nSelectedPts);
	NviewTriangulationRANSACDriver(AllViewsInfo, Selected3DIndex, nSelectedViews, Selected2D, nviews);
	IncrementalBA(Path, nviews, timeID, AllViewsInfo, availViews, PointCorres, CeresDuplicateAddInMask, Selected3DIndex, All3D, Selected2D, nSelectedViews, nSelectedPts, totalPts, true, true, true, true);

	SaveCurrentSfmGL(Path, AllViewsInfo, availViews, All3D, NULL, totalPts);
	SaveCurrentSfmInfo(Path, AllViewsInfo, availViews, All3D, totalPts);
	//saveNVM("C:/temp", "fountain.nvm", AllViewsInfo, availViews);
	delete[]All3D, delete[]Selected2D, delete[]nSelectedViews;

	return;
}

int AllViewsBA(char *Path, CameraData *camera, vector<Point3d>  Vxyz, vector < vector<int>> viewIdAll3D, vector<vector<Point2d>> uvAll3D, int nviews, bool fixIntrinsic, bool fixDistortion, bool debug)
{
	char Fname[200]; FILE *fp = 0;
	int ii, viewID, npts = Vxyz.size();
	double residuals[2];

	double *xyz = new double[npts * 3];
	for (ii = 0; ii < npts; ii++)
		xyz[3 * ii] = Vxyz[ii].x, xyz[3 * ii + 1] = Vxyz[ii].y, xyz[3 * ii + 2] = Vxyz[ii].z;

	printf("Set up BA ...");
	ceres::Problem problem;

	if (debug)
		sprintf(Fname, "C:/temp/reprojectionB.txt"), fp = fopen(Fname, "w+");

	bool *discard3Dpoint = new bool[npts];
	vector<bool> *notGood = new vector<bool>[npts];
	for (int jj = 0; jj < npts; jj++)
		discard3Dpoint[jj] = false, notGood[ii].reserve(viewIdAll3D[ii].size());

	int nBadCounts = 0;
	vector<double> ReProjectionErrorX; ReProjectionErrorX.reserve(npts);
	vector<double> ReProjectionErrorY; ReProjectionErrorY.reserve(npts);
	double maxOutlierX = 0.0, maxOutlierY = 0.0;
	for (int jj = 0; jj < npts; jj++)
	{
		for (ii = 0; ii < viewIdAll3D[jj].size(); ii++)
		{
			if (uvAll3D[jj].at(ii).x < 1 || uvAll3D[jj].at(ii).y < 1)
			{
				notGood[jj].push_back(false);
				continue;
			}

			viewID = viewIdAll3D[jj].at(ii);
			PinholeReprojectionDebug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, uvAll3D[jj].at(ii), Point3d(xyz[3 * jj], xyz[3 * jj + 1], xyz[3 * jj + 2]), residuals);
			if (abs(residuals[0]) > 1.5*camera[0].threshold || abs(residuals[1]) > 1.5*camera[0].threshold)
			{
				notGood[jj].push_back(false);
				//printf("\n@P %d (%.3f %.3f %.3f):  %.2f %.2f", jj, xyz[3 * jj], xyz[3 * jj + 1], xyz[3 * jj + 2], residuals[0], residuals[1]);
				if (abs(residuals[0]) > maxOutlierX)
					maxOutlierX = residuals[0];
				if (abs(residuals[1]) > maxOutlierY)
					maxOutlierY = residuals[1];
				nBadCounts++;
			}
			else
				notGood[jj].push_back(true);
		}

		//Discard point 
		int count = 0;
		for (ii = 0; ii < viewIdAll3D[jj].size(); ii++)
			if (notGood[jj].at(ii) == true)
				count++;

		discard3Dpoint[jj] = false;
		if (count < 2)
		{
			discard3Dpoint[jj] = true;
			continue;
		}

		//add 3D point and its 2D projections to Ceres
		bool once = true;
		int validViewcount = 0;
		double pointErrX = 0.0, pointErrY = 0.0;
		for (ii = 0; ii < viewIdAll3D[jj].size(); ii++)
		{
			if (!notGood[jj].at(ii))
				continue;

			viewID = viewIdAll3D[jj].at(ii);
			ceres::CostFunction* cost_function = PinholeReprojectionError::Create(uvAll3D[jj].at(ii).x, uvAll3D[jj].at(ii).y);
			problem.AddResidualBlock(cost_function, NULL, camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, &xyz[3 * jj]);

			//Set up constant parameters:-->move to here in case some views don't have any 3D points
			if (fixIntrinsic)
				problem.SetParameterBlockConstant(camera[viewID].intrinsic);
			if (fixDistortion)
				problem.SetParameterBlockConstant(camera[viewID].distortion);

			PinholeReprojectionDebug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, uvAll3D[jj].at(ii), Point3d(xyz[3 * jj], xyz[3 * jj + 1], xyz[3 * jj + 2]), residuals);
			validViewcount++;
			pointErrX += pow(residuals[0], 2), pointErrY += pow(residuals[1], 2);

			if (debug)
			{
				if (once)
				{
					once = false;
					fprintf(fp, "%d %.4f %.4f %.4f ", jj, xyz[3 * jj], xyz[3 * jj + 1], xyz[3 * jj + 2]);
				}
				fprintf(fp, "%d %.4f %.4f %.4f %.4f ", viewID, uvAll3D[jj].at(ii).x, uvAll3D[jj].at(ii).y, residuals[0], residuals[1]);
			}
		}
		if (validViewcount > 0)
		{
			ReProjectionErrorX.push_back(sqrt(pointErrX / validViewcount));
			ReProjectionErrorY.push_back(sqrt(pointErrY / validViewcount));
		}

		if (!once)
			fprintf(fp, "\n");
	}
	if (debug)
		fclose(fp);

	sprintf(Fname, "%s/notGood.txt", Path);
	fp = fopen(Fname, "w+");
	for (int jj = 0; jj < npts; jj++)
	{
		fprintf(fp, "%d ", jj);
		for (int ii = 0; ii < notGood[jj].size(); ii++)
		{
			if (notGood[jj].at(ii) == false)
				fprintf(fp, "%d ", ii);
		}
		fprintf(fp, "-1\n");
	}
	fclose(fp);

	printf("\n %d bad points detected with maximum reprojection error of (%.2f %.2f) \n", nBadCounts, maxOutlierX, maxOutlierY);

	double miniX = *min_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
	double maxiX = *max_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
	double avgX = MeanArray(ReProjectionErrorX);
	double stdX = sqrt(VarianceArray(ReProjectionErrorX, avgX));
	double miniY = *min_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
	double maxiY = *max_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
	double avgY = MeanArray(ReProjectionErrorY);
	double stdY = sqrt(VarianceArray(ReProjectionErrorY, avgY));
	printf("Reprojection error before BA \n Min: (%.2f, %.2f) Max: (%.2f,%.2f) Mean: (%.2f,%.2f) Std: (%.2f,%.2f)\n", miniX, miniY, maxiX, maxiY, avgX, avgY, stdX, stdY);

	printf("...run BA...\n");
	ceres::Solver::Options options;
	options.num_threads = 4;
	options.max_num_iterations = 30;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	options.trust_region_strategy_type = ceres::DOGLEG;
	options.use_nonmonotonic_steps = false;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	//std::cout << summary.BriefReport() << "\n";
	std::cout << summary.FullReport() << "\n";

	//Store refined parameters
	GetKFromIntrinsic(camera, nviews);
	GetRTFromrt(camera, nviews);
	for (int ii = 0; ii < nviews; ii++)
		AssembleP(camera[ii].K, camera[ii].R, camera[ii].T, camera[ii].P);
	for (int ii = 0; ii < npts; ii++)
		Vxyz.at(ii) = Point3d(xyz[3 * ii], xyz[3 * ii + 1], xyz[3 * ii + 2]);

	ReProjectionErrorX.clear(), ReProjectionErrorY.clear();
	if (debug)
		sprintf(Fname, "C:/temp/reprojectionA.txt"), fp = fopen(Fname, "w+");
	for (int jj = 0; jj < npts; jj++)
	{
		if (abs(xyz[3 * jj]) > LIMIT3D && !discard3Dpoint[jj])
		{
			bool once = true;
			int validViewcount = 0;
			double pointErrX = 0.0, pointErrY = 0.0;
			for (int ii = 0; ii < viewIdAll3D[jj].size(); ii++)
			{
				if (!notGood[jj].at(ii))
					continue;

				viewID = viewIdAll3D[jj].at(ii);
				PinholeReprojectionDebug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, uvAll3D[jj].at(ii), Point3d(xyz[3 * jj], xyz[3 * jj + 1], xyz[3 * jj + 2]), residuals);

				validViewcount++;
				pointErrX += pow(residuals[0], 2), pointErrY += pow(residuals[1], 2);
				if (once && debug)
				{
					once = false;
					fprintf(fp, "%d %.4f %.4f %.4f ", jj, xyz[3 * jj], xyz[3 * jj + 1], xyz[3 * jj + 2]);
				}
				if (debug)
					fprintf(fp, "%d %.4f %.4f %.4f %.4f ", viewID, uvAll3D[jj].at(ii).x, uvAll3D[jj].at(ii).y, residuals[0], residuals[1]);
			}
			if (!once &&debug)
				fprintf(fp, "\n");

			ReProjectionErrorX.push_back(sqrt(pointErrX / validViewcount));
			ReProjectionErrorY.push_back(sqrt(pointErrY / validViewcount));
		}
	}
	if (debug)
		fclose(fp);

	miniX = *min_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
	maxiX = *max_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
	avgX = MeanArray(ReProjectionErrorX);
	stdX = sqrt(VarianceArray(ReProjectionErrorX, avgX));
	miniY = *min_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
	maxiY = *max_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
	avgY = MeanArray(ReProjectionErrorY);
	stdY = sqrt(VarianceArray(ReProjectionErrorY, avgY));
	printf("Reprojection error after BA \n Min: (%.2f, %.2f) Max: (%.2f,%.2f) Mean: (%.2f,%.2f) Std: (%.2f,%.2f)\n", miniX, miniY, maxiX, maxiY, avgX, avgY, stdX, stdY);

	delete[]xyz, delete[]discard3Dpoint;
	delete[]notGood;
	return 0;
}
int BuildCorpus(char *Path, int nCameras, int CameraToScan, int width, int height, bool IntrinsicCalibrated, bool distortionCorrected, int NDplus)
{
	printf("Reading Corpus and camera info");
	char Fname[200];

	Corpus corpusData;
	if (IntrinsicCalibrated)
	{
		sprintf(Fname, "%s/Corpus.nvm", Path);
		string nvmfile = Fname;
		if (!loadNVMLite(nvmfile, corpusData, 1))
			return 1;
		int nviews = corpusData.nCamera;

		CameraData *cameraInfo = new CameraData[nCameras];
		if (ReadIntrinsicResults(Path, cameraInfo, nCameras) != 0)
			return 1;
		if (CameraToScan != -1)
		{
			for (int ii = 0; ii < nviews; ii++)
			{
				for (int jj = 0; jj < 9; jj++)
					corpusData.camera[ii].K[jj] = cameraInfo[CameraToScan].K[jj];
				for (int jj = 0; jj < 7; jj++)
					corpusData.camera[ii].distortion[jj] = cameraInfo[CameraToScan].distortion[jj];
			}
		}
		else
		{
			for (int ii = 0; ii < nviews; ii++)
			{
				for (int jj = 0; jj < 9; jj++)
					corpusData.camera[ii].K[jj] = cameraInfo[ii].K[jj];
				for (int jj = 0; jj < 7; jj++)
					corpusData.camera[ii].distortion[jj] = cameraInfo[ii].distortion[jj];
			}
		}
	}
	else
	{
		sprintf(Fname, "%s/BA_Camera_AllParams_after.txt", Path);
		if (!loadBundleAdjustedNVMResults(Fname, corpusData))
			return 1;
	}
	int nviews = corpusData.nCamera;

	for (int ii = 0; ii < nviews; ii++)
	{
		corpusData.camera[ii].LensModel = RADIAL_TANGENTIAL_PRISM, corpusData.camera[ii].threshold = 1.5, corpusData.camera[ii].ninlierThresh = 50, corpusData.camera[ii];
		GetrtFromRT(corpusData.camera[ii].rt, corpusData.camera[ii].R, corpusData.camera[ii].T);
		GetIntrinsicFromK(corpusData.camera[ii]);
		AssembleP(corpusData.camera[ii].K, corpusData.camera[ii].R, corpusData.camera[ii].T, corpusData.camera[ii].P);
		if (distortionCorrected)
			for (int jj = 0; jj < 7; jj++)
				corpusData.camera[ii].distortion[jj] = 0.0;
	}
	printf("...Done\n");

	int totalPts;
	vector<int> cumulativePts;
	ReadCumulativePoints(Path, nviews, -1, cumulativePts);
	totalPts = cumulativePts.at(nviews);

	vector<int>*ViewMatch = new vector<int>[totalPts];
	vector<int>*PointIDMatch = new vector<int>[totalPts];

	printf("Reading Matching table....");
	sprintf(Fname, "%s/ViewPM.txt", Path); FILE *fp = fopen(Fname, "r");
	int nviewsi, viewi, n3D = 0;
	while (fscanf(fp, "%d ", &nviewsi) != EOF)
	{
		ViewMatch[n3D].reserve(nviewsi);
		for (int ii = 0; ii < nviewsi; ii++)
		{
			fscanf(fp, "%d ", &viewi);
			ViewMatch[n3D].push_back(viewi);
		}
		n3D++;
	}
	fclose(fp);

	sprintf(Fname, "%s/IDPM.txt", Path); fp = fopen(Fname, "r");
	int np, pi;
	n3D = 0;
	while (fscanf(fp, "%d ", &np) != EOF)
	{
		PointIDMatch[n3D].reserve(np);
		for (int ii = 0; ii < np; ii++)
		{
			fscanf(fp, "%d ", &pi);
			PointIDMatch[n3D].push_back(pi);
		}
		n3D++;
	}
	fclose(fp);
	printf("...Done\n");

	//Read all sift points
	printf("Reading SIFT keys....");
	vector<SiftKeypoint> *AllKeys = new vector < SiftKeypoint >[nviews];
	for (int ii = 0; ii < nviews; ii++)
	{
		sprintf(Fname, "%s/K%d.dat", Path, ii);
		ReadKPointsBinarySIFTGPU(Fname, AllKeys[ii]);
	}
	printf("...Done\n");

	//Triangulate points from estimated camera poses
	printf("Triangulating the Corpus...");
	Point3d xyz;
	double *A = new double[6 * nviews * 2];
	double *B = new double[2 * nviews * 2];
	double *tPs = new double[12 * nviews * 2];
	bool *passed = new bool[nviews * 2];
	double *Ps = new double[12 * nviews * 2];

	vector<int>Inliers[1];  Inliers[0].reserve(nviews * 2);
	Point2d *match2Dpts = new Point2d[nviews * 2];

	//double Ps[12 * 2 * 190];
	//Point2d match2Dpts[190 * 2];

	corpusData.xyz.reserve(n3D);
	corpusData.rgb.reserve(n3D);
	corpusData.viewIdAll3D.reserve(n3D);
	corpusData.pointIdAll3D.reserve(n3D);

	vector<int>viewIDs, pointIDs, orgId, threeDid;
	vector<Point2d> uvPer3D, uvperView;

	for (int ii = 0; ii < n3D; ii++)
	{
		corpusData.viewIdAll3D.push_back(viewIDs), corpusData.viewIdAll3D.at(ii).reserve(nviews);
		corpusData.pointIdAll3D.push_back(pointIDs), corpusData.pointIdAll3D.at(ii).reserve(nviews);
		corpusData.uvAll3D.push_back(uvPer3D), corpusData.uvAll3D.at(ii).reserve(nviews);
	}

	printf("Start: \n");
	double ProThresh = 0.99, PercentInlier = 0.25;
	int goodNDplus = 0, iterMax = (int)(log(1.0 - ProThresh) / log(1.0 - pow(PercentInlier, 2)) + 0.5); //log(1-eps) / log(1 - (inlier%)^min_pts_requires)
	bool printout = 0;
	double start = omp_get_wtime();
	for (int jj = 0; jj < n3D; jj++)
	{
		if (jj % 1000 == 0)
			printf("@\r# %.2f%% (%.2fs) Triangualating corpus..", 100.0*jj / n3D, omp_get_wtime() - start);
		int nviewsi = ViewMatch[jj].size();
		if (nviewsi >= NDplus)
		{
			Inliers[0].clear();
			for (int ii = 0; ii < nviewsi; ii++)
			{
				viewi = ViewMatch[jj].at(ii);
				for (int kk = 0; kk < 12; kk++)
					Ps[12 * ii + kk] = corpusData.camera[viewi].P[kk];

				pi = PointIDMatch[jj].at(ii);
				match2Dpts[ii] = Point2d(AllKeys[viewi].at(pi).x, AllKeys[viewi].at(pi).y);
			}
			if (printout)
			{
				FILE *fp = fopen("C:/temp/corres.txt", "w+");
				for (int ii = 0; ii < nviewsi; ii++)
					fprintf(fp, "%.1f %.1f\n", match2Dpts[ii].x, match2Dpts[ii].y);
				for (int ii = 0; ii < nviewsi; ii++)
				{
					for (int jj = 0; jj < 12; jj++)
						fprintf(fp, "%.4f ", Ps[jj + ii * 12]);
					fprintf(fp, "\n");
				}

				fclose(fp);
			}

			NviewTriangulationRANSAC(match2Dpts, Ps, &xyz, passed, Inliers, nviewsi, 1, iterMax, PercentInlier, corpusData.camera[0].threshold, A, B, tPs);
			if (passed[0])
			{
				int ninlier = 0;
				for (int ii = 0; ii < Inliers[0].size(); ii++)
					if (Inliers[0].at(ii))
						ninlier++;
				if (ninlier < NDplus)
					continue; //Corpus needs NDplus+ points!
				corpusData.xyz.push_back(xyz);

				for (int ii = 0; ii < nviewsi; ii++)
				{
					if (Inliers[0].at(ii))
					{
						viewi = ViewMatch[jj].at(ii), pi = PointIDMatch[jj].at(ii);

						corpusData.viewIdAll3D.at(goodNDplus).push_back(viewi);
						corpusData.pointIdAll3D.at(goodNDplus).push_back(pi);
						corpusData.uvAll3D.at(goodNDplus).push_back(match2Dpts[ii]);
					}
				}
				goodNDplus++;
			}
		}
	}
	printf("@\r# %.2f%% (%.2fs) \n", 100.0, omp_get_wtime() - start);
	printf("Found %d (%d+) points.\n", goodNDplus, NDplus);

	fp = fopen("C:/temp/BA.txt", "w+");
	for (int ii = 0; ii < nviews; ii++)
	{
		for (int jj = 0; jj < 5; jj++)
			fprintf(fp, "%.8f ", corpusData.camera[ii].intrinsic[jj]);
		for (int jj = 0; jj < 7; jj++)
			fprintf(fp, "%.8f ", corpusData.camera[ii].distortion[jj]);
		for (int jj = 0; jj < 6; jj++)
			fprintf(fp, "%.8f ", corpusData.camera[ii].rt[jj]);
		fprintf(fp, "\n");
	}
	for (int ii = 0; ii < goodNDplus; ii++)
	{
		int nviewsi = corpusData.viewIdAll3D[ii].size();
		fprintf(fp, "%.6f %.6f %.6f %d ", corpusData.xyz[ii].x, corpusData.xyz[ii].y, corpusData.xyz[ii].z, nviewsi);
		for (int jj = 0; jj < nviewsi; jj++)
			fprintf(fp, "%d %.4f %.4f ", corpusData.viewIdAll3D[ii].at(jj), corpusData.uvAll3D[ii].at(jj).x, corpusData.uvAll3D[ii].at(jj).y);
		fprintf(fp, "\n");
	}
	fclose(fp);

	printf("Runing BA on the triangulated points...");
	AllViewsBA(Path, corpusData.camera, corpusData.xyz, corpusData.viewIdAll3D, corpusData.uvAll3D, nviews, true, true, false);
	if (CameraToScan != -1)
		SaveIntrinsicResults(Path, corpusData.camera, 1);
	else
		SaveIntrinsicResults(Path, corpusData.camera, nviews);
	printf("...Done!");

	printf("Remove not good points ...");
	sprintf(Fname, "%s/notGood.txt", Path);	fp = fopen(Fname, "r");
	vector<int> *notGood = new vector<int>[goodNDplus];
	for (int jj = 0; jj < goodNDplus; jj++)
	{
		int pid, ii;
		fscanf(fp, "%d %d", &pid, &ii);
		while (ii != -1)
		{
			notGood[jj].push_back(ii);
			fscanf(fp, "%d ", &ii);
		}
	}
	fclose(fp);

	for (int jj = 0; jj < goodNDplus; jj++)
	{
		for (int ii = 0; ii < notGood[jj].size(); ii++)
		{
			int viewID = notGood[jj].at(ii);
			corpusData.viewIdAll3D.at(jj).erase(corpusData.viewIdAll3D.at(jj).begin() + viewID);
			corpusData.pointIdAll3D.at(jj).erase(corpusData.pointIdAll3D.at(jj).begin() + viewID);
			corpusData.uvAll3D.at(jj).erase(corpusData.uvAll3D.at(jj).begin() + viewID);
		}
	}
	delete[]notGood;

	//And generate 3D id, uv, sift id for all views
	printf("and generate Corpus visibility info....");
	vector<vector<int>> siftIDAllViews;
	corpusData.threeDIdAllViews.reserve(nviews);
	corpusData.uvAllViews.reserve(nviews);
	siftIDAllViews.reserve(nviews);

	for (int ii = 0; ii < nviews; ii++)
	{
		corpusData.threeDIdAllViews.push_back(threeDid);
		corpusData.uvAllViews.push_back(uvperView);
		siftIDAllViews.push_back(orgId);

		corpusData.threeDIdAllViews.at(ii).reserve(10000);
		corpusData.uvAllViews.at(ii).reserve(10000);
		siftIDAllViews.at(ii).reserve(10000);
	}

	Point2d uv;
	for (int jj = 0; jj < goodNDplus; jj++)
	{
		for (int ii = 0; ii < corpusData.viewIdAll3D[jj].size(); ii++)
		{
			viewi = corpusData.viewIdAll3D[jj].at(ii), pi = corpusData.pointIdAll3D[jj].at(ii), uv = corpusData.uvAll3D[jj].at(ii);

			corpusData.threeDIdAllViews.at(viewi).push_back(jj);
			corpusData.uvAllViews.at(viewi).push_back(uv);
			siftIDAllViews.at(viewi).push_back(pi);
		}
	}
	printf("Done!\n");

	//Get the color info
	printf("Getting color info...\n");
	int length = width*height;
	Mat cvImg;
	unsigned char *AllImages = new unsigned char[nviews*length * 3];
	start = omp_get_wtime();
	for (int kk = 0; kk < nviews; kk++)
	{
		sprintf(Fname, "%s/%d.png", Path, kk);
		myImgReader(Fname, AllImages + kk * 3 * length, width, height, 3);
		printf("@\r# %.2f%% (%.2fs) Reading images...", 100.0*kk / nviews, omp_get_wtime() - start);
	}
	printf("@\r# %.2f%% (%.2fs) Loaded all images...\n", 100.0, omp_get_wtime() - start);

	corpusData.rgb.reserve(goodNDplus);
	for (int kk = 0; kk < goodNDplus; kk++)
	{
		int viewID = corpusData.viewIdAll3D.at(kk).at(0);
		int x = corpusData.uvAll3D.at(kk).at(0).x;
		int y = corpusData.uvAll3D.at(kk).at(0).y;
		int id = x + y*width;
		Point3i rgb;
		rgb.z = AllImages[viewID * 3 * length + id];//b
		rgb.y = AllImages[(viewID * 3 + 1) * length + id];//g
		rgb.x = AllImages[(viewID * 3 + 2)* length + id];//r
		corpusData.rgb.push_back(rgb);
	}
	delete[]AllImages;

	vector<int> AvailViews; AvailViews.reserve(nviews);
	for (int ii = 0; ii < nviews; ii++)
		AvailViews.push_back(ii);
	SaveCurrentSfmGL2(Path, corpusData.camera, AvailViews, corpusData.xyz, corpusData.rgb);

	//Get sift matrix for all views
	printf("Prune SIFT descriptors for only Corpus points....");
	int nSift, totalSift = 0, maxSift = 0;
	corpusData.IDCumView.reserve(nviews + 1);
	for (int ii = 0; ii < nviews; ii++)
	{
		corpusData.IDCumView.push_back(totalSift);
		nSift = siftIDAllViews.at(ii).size();
		if (nSift > maxSift)
			maxSift = nSift;
		totalSift += nSift;
	}
	corpusData.IDCumView.push_back(totalSift);

	float d;
	corpusData.SiftDesc.create(totalSift, SIFTBINS, CV_32F);
	vector<float> desc; desc.reserve(maxSift*SIFTBINS);
	for (int ii = 0; ii < nviews; ii++)
	{
		desc.clear();
		sprintf(Fname, "%s/D%d.dat", Path, ii), ReadDescriptorBinarySIFTGPU(Fname, desc);

		int curPid = corpusData.IDCumView.at(ii), nSift = siftIDAllViews.at(ii).size();
		for (int j = 0; j < nSift; ++j)
		{
			int pid = siftIDAllViews.at(ii).at(j);
			for (int i = 0; i < SIFTBINS; i++)
			{
				d = desc.at(pid*SIFTBINS + i);
				corpusData.SiftDesc.at<float>(curPid + j, i) = d;
			}
		}
	}
	printf("...Done\n");

	SaveCorpusInfo(Path, corpusData);

	delete[]ViewMatch, delete[]PointIDMatch, delete[]AllKeys;
	delete[]A, delete[]B, delete[]tPs, delete[]passed, delete[]Ps, delete[]match2Dpts;
	return 0;
}
int PoseBA(char *Path, CameraData &camera, vector<Point3d>  Vxyz, vector<Point2d> uvAll3D, vector<bool> &Good, bool fixIntrinsic, bool fixDistortion, bool debug)
{
	char Fname[200]; FILE *fp = 0;
	int ii, npts = Vxyz.size();
	double residuals[2];

	double *xyz = new double[npts * 3];
	for (ii = 0; ii < npts; ii++)
		xyz[3 * ii] = Vxyz[ii].x, xyz[3 * ii + 1] = Vxyz[ii].y, xyz[3 * ii + 2] = Vxyz[ii].z;

	//printf("Set up Pose BA ...");
	ceres::Problem problem;

	int nBadCounts = 0, validPtsCount = 0;
	vector<double> ReProjectionErrorX; ReProjectionErrorX.reserve(npts);
	vector<double> ReProjectionErrorY; ReProjectionErrorY.reserve(npts);
	double maxOutlierX = 0.0, maxOutlierY = 0.0, pointErrX = 0.0, pointErrY = 0.0;

	double fxfy[2] = { camera.intrinsic[0], camera.intrinsic[1] };
	double skew = camera.intrinsic[2];
	double uv0[2] = { camera.intrinsic[3], camera.intrinsic[4] };
	double Radial12[2] = { camera.distortion[0], camera.distortion[1] };
	double Tangential[2] = { camera.distortion[3], camera.distortion[4] };
	double Radial3Prism[3] = { camera.distortion[2], camera.distortion[5], camera.distortion[6] };

	if (debug)
		sprintf(Fname, "C:/temp/reprojectionB.txt"), fp = fopen(Fname, "w+");
	for (int jj = 0; jj < npts; jj++)
	{
		if (uvAll3D[jj].x < 1 || uvAll3D[jj].y < 1)
		{
			Good.push_back(false);
			continue;
		}

		PinholeReprojectionDebug(camera.intrinsic, camera.distortion, camera.rt, uvAll3D[jj], Point3d(xyz[3 * jj], xyz[3 * jj + 1], xyz[3 * jj + 2]), residuals);
		if (abs(residuals[0]) > 1.25*camera.threshold || abs(residuals[1]) > 1.25*camera.threshold)
		{
			Good.push_back(false);
			if (abs(residuals[0]) > maxOutlierX)
				maxOutlierX = residuals[0];
			if (abs(residuals[1]) > maxOutlierY)
				maxOutlierY = residuals[1];
			nBadCounts++;
		}
		else
		{
			Good.push_back(true);
			//ceres::CostFunction* cost_function = PinholeReprojectionError::Create(uvAll3D[jj].x, uvAll3D[jj].y);
			//problem.AddResidualBlock(cost_function, NULL, camera.intrinsic, camera.distortion, camera.rt, &xyz[3 * jj]);
			ceres::CostFunction* cost_function = PinholeReprojectionError2::Create(uvAll3D[jj].x, uvAll3D[jj].y);
			problem.AddResidualBlock(cost_function, NULL, fxfy, &skew, uv0, Radial12, Tangential, Radial3Prism, camera.rt, &xyz[3 * jj]);

			validPtsCount++;
			ReProjectionErrorX.push_back(abs(residuals[0]));
			ReProjectionErrorY.push_back(abs(residuals[1]));
		}

		if (debug)
			fprintf(fp, "%d %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n", jj, xyz[3 * jj], xyz[3 * jj + 1], xyz[3 * jj + 2], uvAll3D[jj].x, uvAll3D[jj].y, abs(residuals[0]), abs(residuals[1]));
	}
	if (debug)
		fclose(fp);

	double miniX = *min_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
	double maxiX = *max_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
	double avgX = MeanArray(ReProjectionErrorX);
	double stdX = sqrt(VarianceArray(ReProjectionErrorX, avgX));
	double miniY = *min_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
	double maxiY = *max_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
	double avgY = MeanArray(ReProjectionErrorY);
	double stdY = sqrt(VarianceArray(ReProjectionErrorY, avgY));

#pragma omp critical
	{
		printf("\n %d bad points detected with maximum reprojection error of (%.2f %.2f) \n", nBadCounts, maxOutlierX, maxOutlierY);
		printf("Reprojection error before BA \n Min: (%.2f, %.2f) Max: (%.2f,%.2f) Mean: (%.2f,%.2f) Std: (%.2f,%.2f)\n", miniX, miniY, maxiX, maxiY, avgX, avgY, stdX, stdY);
	}

	//Set up constant parameters:
	printf("...set up fixed parameters ...");
	/*if (fixIntrinsic)
		problem.SetParameterBlockConstant(camera.intrinsic);
		if (fixDistortion)
		problem.SetParameterBlockConstant(camera.distortion);*/
	problem.SetParameterBlockConstant(&skew);
	problem.SetParameterBlockConstant(Radial3Prism);

	for (int ii = 0; ii < npts; ii++)
		if (Good.at(ii))
			problem.SetParameterBlockConstant(xyz + 3 * ii);


	//printf("...run \n");
	ceres::Solver::Options options;
	options.num_threads = 4;
	options.max_num_iterations = 30;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = false;
	options.trust_region_strategy_type = ceres::DOGLEG;
	options.use_nonmonotonic_steps = false;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	//std::cout << summary.BriefReport() << "\n";
	std::cout << summary.BriefReport() << "\n";

	//Store refined parameters
	camera.intrinsic[0] = fxfy[0], camera.intrinsic[1] = fxfy[1];
	camera.intrinsic[2] = skew;
	camera.intrinsic[3] = uv0[0], camera.intrinsic[4] = uv0[1];
	camera.distortion[0] = Radial12[0], camera.distortion[1] = Radial12[1];
	camera.distortion[3] = Tangential[0], camera.distortion[4] = Tangential[1];
	camera.distortion[2] = Radial3Prism[0], camera.distortion[5] = Radial3Prism[1], camera.distortion[6] = Radial3Prism[2];

	GetKFromIntrinsic(&camera, 1);
	GetRTFromrt(&camera, 1);
	AssembleP(camera.K, camera.R, camera.T, camera.P);
	for (int ii = 0; ii < npts; ii++)
		Vxyz.at(ii) = Point3d(xyz[3 * ii], xyz[3 * ii + 1], xyz[3 * ii + 2]);

	ReProjectionErrorX.clear(), ReProjectionErrorY.clear();
	pointErrX = 0.0, pointErrY = 0.0;

	if (debug)
		sprintf(Fname, "C:/temp/reprojectionA.txt"), fp = fopen(Fname, "w+");
	for (int ii = 0; ii < npts; ii++)
	{
		if (abs(xyz[3 * ii]) > LIMIT3D)
		{
			if (!Good.at(ii))
				continue;

			PinholeReprojectionDebug(camera.intrinsic, camera.distortion, camera.rt, uvAll3D[ii], Point3d(xyz[3 * ii], xyz[3 * ii + 1], xyz[3 * ii + 2]), residuals);

			ReProjectionErrorX.push_back(abs(residuals[0]));
			ReProjectionErrorY.push_back(abs(residuals[1]));
			if (debug)
				fprintf(fp, "%d %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n", ii, xyz[3 * ii], xyz[3 * ii + 1], xyz[3 * ii + 2], uvAll3D[ii].x, uvAll3D[ii].y, abs(residuals[0]), abs(residuals[1]));
		}
	}
	if (debug)
		fclose(fp);


	miniX = *min_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
	maxiX = *max_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
	avgX = MeanArray(ReProjectionErrorX);
	stdX = sqrt(VarianceArray(ReProjectionErrorX, avgX));
	miniY = *min_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
	maxiY = *max_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
	avgY = MeanArray(ReProjectionErrorY);
	stdY = sqrt(VarianceArray(ReProjectionErrorY, avgY));
#pragma omp critical
	printf("Reprojection error after BA \n Min: (%.2f, %.2f) Max: (%.2f,%.2f) Mean: (%.2f,%.2f) Std: (%.2f,%.2f)\n", miniX, miniY, maxiX, maxiY, avgX, avgY, stdX, stdY);

	delete[]xyz;
	return 0;
}
int MatchCameraToCorpus(char *Path, Corpus &corpusData, int cameraID, int timeID, vector<int> CorpusViewToMatch, const float nndrRatio, const int ninlierThresh)
{
	//Load image and extract features
	const int descriptorSize = SIFTBINS;

	char Fname[200];
	sprintf(Fname, "%s/%d/%d.png", Path, cameraID, timeID);
	Mat img = imread(Fname, CV_LOAD_IMAGE_COLOR);
	if (img.empty())
	{
		printf("Can't read %s\n", Fname);
		return 1;
	}

	//Mat imgGray, equalizedImg;
	//cvtColor(img, imgGray, CV_BGR2GRAY);
	////equalizeHist(imgGray, equalizedImg);

	double start = omp_get_wtime();
	if (timeID < 0)
		sprintf(Fname, "%s/%d/K.dat", Path, cameraID);
	else
		sprintf(Fname, "%s/%d/K%d.dat", Path, cameraID, timeID);

	bool readsucces = false;
	vector<KeyPoint> keypoints1; keypoints1.reserve(MAXSIFTPTS);
	if (useGPU)
		readsucces = ReadKPointsBinarySIFTGPU(Fname, keypoints1);
	else
		readsucces = ReadKPointsBinary(Fname, keypoints1);
	if (!readsucces)
	{
		printf("%s does not have SIFT points. Please precompute it!\n");
		exit(1);
	}

	if (timeID < 0)
		sprintf(Fname, "%s/D%d.dat", Path, cameraID);
	else
		sprintf(Fname, "%s/%d/D%d.dat", Path, cameraID, timeID);
	Mat descriptors1 = ReadDescriptorBinarySIFTGPU(Fname);
	if (descriptors1.rows == 1)
	{
		printf("%s does not have SIFT points. Please precompute it!\n");
		exit(1);
	}

	//USAC config
	bool USEPROSAC = false, USESPRT = true, USELOSAC = true;
	ConfigParamsFund cfg;
	cfg.common.confThreshold = 0.99, cfg.common.minSampleSize = 7, cfg.common.inlierThreshold = 1.5;
	cfg.common.maxHypotheses = 850000, cfg.common.maxSolutionsPerSample = 3;
	cfg.common.prevalidateSample = true, cfg.common.prevalidateModel = true, cfg.common.testDegeneracy = true;
	cfg.common.randomSamplingMethod = USACConfig::SAMP_UNIFORM, cfg.common.verifMethod = USACConfig::VERIF_SPRT, cfg.common.localOptMethod = USACConfig::LO_LOSAC;

	if (USEPROSAC)
		cfg.prosac.maxSamples, cfg.prosac.beta, cfg.prosac.nonRandConf, cfg.prosac.minStopLen;
	if (USESPRT)
		cfg.sprt.tM = 200.0, cfg.sprt.mS = 2.38, cfg.sprt.delta = 0.05, cfg.sprt.epsilon = 0.15;
	if (USELOSAC)
		cfg.losac.innerSampleSize = 15, cfg.losac.innerRansacRepetitions = 5, cfg.losac.thresholdMultiplier = 2.0, cfg.losac.numStepsIterative = 4;

	//Match extracted features with Corpus
	const bool useBFMatcher = false;
	const int knn = 2, ntrees = 4, maxLeafCheck = 128;

	vector<Point2f> twoD; twoD.reserve(5000);
	vector<int> threeDiD; threeDiD.reserve(5000);
	vector<int>viewID; viewID.reserve(5000);

	//Finding nearest neighbor
	bool ShowCorrespondence = 0;
	vector<Point2d>key1, key2;
	vector<int>CorrespondencesID;
	double Fmat[9];
	vector<int>cur3Ds, Inliers;
	key1.reserve(5000), key2.reserve(5000);
	CorrespondencesID.reserve(5000), cur3Ds.reserve(5000), Inliers.reserve(5000);

	for (int ii = 0; ii < CorpusViewToMatch.size(); ii++)
	{
		key1.clear(), key2.clear();
		cur3Ds.clear(), Inliers.clear(), CorrespondencesID.clear();

		int camera2ID = CorpusViewToMatch.at(ii);
		int startID = corpusData.IDCumView.at(camera2ID), endID = corpusData.IDCumView.at(camera2ID + 1);
		Mat descriptors2(endID - startID, SIFTBINS, CV_32F);

		for (int jj = startID; jj < endID; jj++)
			for (int kk = 0; kk < SIFTBINS; kk++)
				descriptors2.at<float>(jj - startID, kk) = corpusData.SiftDesc.at<float>(jj, kk);

		double start = omp_get_wtime();
		Mat indices, dists;
		vector<vector<DMatch> > matches;
		if (useBFMatcher)
		{
			cv::BFMatcher matcher(cv::NORM_L2);
			matcher.knnMatch(descriptors2, descriptors1, matches, knn);
		}
		else
		{
			cv::flann::Index flannIndex(descriptors1, cv::flann::KDTreeIndexParams(ntrees));//, cvflann::FLANN_DIST_EUCLIDEAN);
			flannIndex.knnSearch(descriptors2, indices, dists, knn, cv::flann::SearchParams(maxLeafCheck));//Search in desc1 for every desc in 2
		}

		int count = 0;
		if (!useBFMatcher)
		{
			for (int i = 0; i < descriptors2.rows; ++i)
			{
				int ind1 = indices.at<int>(i, 0);
				if (indices.at<int>(i, 0) >= 0 && indices.at<int>(i, 1) >= 0 && dists.at<float>(i, 0) <= nndrRatio * dists.at<float>(i, 1))
				{
					int cur3Did = corpusData.threeDIdAllViews.at(camera2ID).at(i);
					cur3Ds.push_back(cur3Did);

					key1.push_back(Point2d(keypoints1.at(ind1).pt.x, keypoints1.at(ind1).pt.y));
					key2.push_back(corpusData.uvAllViews.at(camera2ID).at(i));
				}
			}
		}
		else
		{
			for (unsigned int i = 0; i < matches.size(); ++i)
			{
				if (matches.at(i).size() == 2 && matches.at(i).at(0).distance <= nndrRatio * matches.at(i).at(1).distance)
				{
					int cur3Did = corpusData.threeDIdAllViews.at(camera2ID).at(i);
					cur3Ds.push_back(cur3Did);

					int ind1 = matches.at(i).at(0).trainIdx;
					key1.push_back(Point2d(keypoints1.at(ind1).pt.x, keypoints1.at(ind1).pt.y));
					key2.push_back(corpusData.uvAllViews.at(camera2ID).at(i));
				}
			}
		}

		//TO DO: Work to set threshold on the ransac
		cfg.common.numDataPoints = key1.size();
		USAC_FindFundamentalMatrix(cfg, key1, key2, Fmat, Inliers);

		/*sprintf(Fname, "%s/orig_pts.txt", Path); FILE *fp = fopen(Fname, "w+");
		fprintf(fp, "%d\n", cfg.common.numDataPoints);
		for (int ii = 0; ii < cfg.common.numDataPoints; ii++)
		fprintf(fp, "%.2f %.2f %.2f %.2f\n", key1[ii].x, key1[ii].y, key2[ii].x, key2[ii].y);
		fclose(fp);

		sprintf(Fname, "%s/F.txt", Path); fp = fopen(Fname, "w+");
		for (int ii = 0; ii < 9; ii++)
		fprintf(fp, "%.8f ", Fmat[ii]);
		fclose(fp);

		sprintf(Fname, "%s/inliers.txt", Path); fp = fopen(Fname, "w+");
		for (int ii = 0; ii < cfg.common.numDataPoints; ii++)
		fprintf(fp, "%d\n", Inliers[ii]);
		fclose(fp);*/

		int FittedPts = 0;
		for (int ii = 0; ii < cfg.common.numDataPoints; ii++)
			if (Inliers[ii] == 1)
				FittedPts++;
		if (FittedPts < ninlierThresh)
		{
			printf("(%d, %d) to Corpus %d: failed Fundamental matrix test\n\n", cameraID, timeID, camera2ID);
			continue;
		}

		/*sprintf(Fname, "C:/temp/%d_%d.txt", cameraID, camera2ID);
		FILE *fp = fopen(Fname, "w+");
		for (int jj = 0; jj < Inliers.size(); jj++)
		{
		if (Inliers[jj] == 1)
		fprintf(fp, "%d %.2f %.2f %.2f %.2f\n", cur3Ds[jj], key1[jj].x, key1[jj].y, key2[jj].x, key2[jj].y);
		}
		fclose(fp);*/

		//Add matches to 2d-3d list
		for (int jj = 0; jj < Inliers.size(); jj++)
		{
			if (Inliers[jj] == 1)
			{
				int cur3Did = cur3Ds[jj];
				bool used = false;
				for (int kk = 0; kk < threeDiD.size(); kk++)
				{
					if (cur3Did == threeDiD.at(kk))
					{
						used = true;
						break;
					}
				}
				if (used)
					continue;

				twoD.push_back(Point2f(key1[jj].x, key1[jj].y));
				threeDiD.push_back(cur3Did);
				viewID.push_back(camera2ID);
				count++;
			}
		}

		if (ShowCorrespondence)
		{
			int nchannels = 3;
			sprintf(Fname, "%s/%d/%d.png", Path, cameraID, timeID);
			IplImage *Img1 = cvLoadImage(Fname, nchannels == 3 ? 1 : 0);
			if (Img1->imageData == NULL)
			{
				printf("Cannot load %s\n", Fname);
				return 1;
			}
			sprintf(Fname, "%s/%d.png", Path, camera2ID);
			IplImage *Img2 = cvLoadImage(Fname, nchannels == 3 ? 1 : 0);
			if (Img2->imageData == NULL)
			{
				printf("Cannot load %s\n", Fname);
				return 1;
			}

			CorrespondencesID.clear();
			for (int ii = 0; ii < key1.size(); ii++)
				if (Inliers[ii] == 1)
					CorrespondencesID.push_back(ii), CorrespondencesID.push_back(ii);

			IplImage* correspond = cvCreateImage(cvSize(Img1->width + Img2->width, Img1->height), 8, nchannels);
			cvSetImageROI(correspond, cvRect(0, 0, Img1->width, Img1->height));
			cvCopy(Img1, correspond);
			cvSetImageROI(correspond, cvRect(Img1->width, 0, correspond->width, correspond->height));
			cvCopy(Img2, correspond);
			cvResetImageROI(correspond);
			DisplayImageCorrespondence(correspond, Img1->width, 0, key1, key2, CorrespondencesID, 1.0);
		}
#pragma omp critical
		printf("(%d, %d) to Corpus %d: %d 3+ points in %.2fs.\n\n", cameraID, timeID, camera2ID, count, omp_get_wtime() - start);
	}

	sprintf(Fname, "%s/%d/3D2D_%d.txt", Path, cameraID, timeID);
	FILE *fp = fopen(Fname, "w+");
	fprintf(fp, "%d\n", threeDiD.size());
	for (int jj = 0; jj < threeDiD.size(); jj++)
		fprintf(fp, "%d %.16f %.16f \n", threeDiD[jj], twoD[jj].x, twoD[jj].y);
	fclose(fp);

	/*sprintf(Fname, "%s/%d/_3D2D_%d.txt", Path, cameraID, timeID); fp = fopen(Fname, "w+");
	fprintf(fp, "%d\n", threeDiD.size());
	for (int jj = 0; jj < threeDiD.size(); jj++)
	{
		int pid = threeDiD[jj], vid = viewID[jj];
		Point2d twoDCorpus;
		for (int i = 0; i < corpusData.viewIdAll3D[pid].size(); i++)
		{
			if (corpusData.viewIdAll3D[pid].at(i) == vid)
			{
				twoDCorpus = corpusData.uvAll3D[pid].at(i);
				break;
			}
		}

		fprintf(fp, "%d %.2f %.2f %d %.2f %.2f\n", threeDiD[jj], twoD[jj].x, twoD[jj].y, vid, twoDCorpus.x, twoDCorpus.y);
	}
	fclose(fp);*/

	return 0;
}
int EstimateCameraPoseFromCorpus(char *Path, Corpus corpusData, CameraData  &cameraParas, int cameraID, bool fixedIntrinsc, bool fixedDistortion, int timeID)
{
	char Fname[200];
	int threeDid, npts, ptsCount = 0;
	double u, v;

	sprintf(Fname, "%s/%d/3D2D_%d.txt", Path, cameraID, timeID);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return 1;
	}
	fscanf(fp, "%d ", &npts);
	Point2d *pts = new Point2d[npts];
	Point3d *t3D = new Point3d[npts];
	while (fscanf(fp, "%d %lf %lf ", &threeDid, &u, &v) != EOF)
	{
		pts[ptsCount].x = u, pts[ptsCount].y = v;
		t3D[ptsCount].x = corpusData.xyz.at(threeDid).x, t3D[ptsCount].y = corpusData.xyz.at(threeDid).y, t3D[ptsCount].z = corpusData.xyz.at(threeDid).z;
		ptsCount++;
	}
	fclose(fp);

	if (cameraParas.LensModel == RADIAL_TANGENTIAL_PRISM && abs(cameraParas.distortion[0]) > 0.001)
		LensCorrectionPoint(pts, cameraParas.K, cameraParas.distortion, npts);

	int ninliers;
	DetermineDevicePose(cameraParas.K, NULL, cameraParas.LensModel, cameraParas.R, cameraParas.T, pts, t3D, npts, cameraParas.threshold, ninliers);
	//printf("EPnP + Ransac for View (%d, %d)...", cameraID, timeID);
	GetrtFromRT(&cameraParas, 1);

	vector<bool> Good; Good.reserve(npts);
	vector<Point3d> Vxyz; Vxyz.reserve(npts);
	vector<Point2d> uv; uv.reserve(npts);
	for (int ii = 0; ii < npts; ii++)
	{
		Vxyz.push_back(Point3d(t3D[ii].x, t3D[ii].y, t3D[ii].z));
		uv.push_back(Point2d(pts[ii].x, pts[ii].y));
	}

	PoseBA(Path, cameraParas, Vxyz, uv, Good, fixedIntrinsc, fixedDistortion, false);
	printf("Intrinsic: %.1f %.1f %.1f %.1f %.1f\n", cameraParas.intrinsic[0], cameraParas.intrinsic[1], cameraParas.intrinsic[2], cameraParas.intrinsic[3], cameraParas.intrinsic[4]);
	printf("Distortion: ");
	for (int ii = 0; ii < 7; ii++)
		printf("%.1e ", cameraParas.distortion[ii]);
	printf("\n");

	fp = fopen("C:/temp/intrinsic.txt", "a+");
	for (int ii = 0; ii < 5; ii++)
		fprintf(fp, "%.1f ", cameraParas.intrinsic[ii]);
	for (int ii = 0; ii < 7; ii++)
		fprintf(fp, "%.1e ", cameraParas.distortion[ii]);
	fprintf(fp, "\n");
	fclose(fp);

	ninliers = 0;
	for (int ii = 0; ii < npts; ii++)
		if (Good[ii])
			ninliers++;

	double iR[9], center[3];
	mat_invert(cameraParas.R, iR);

	cameraParas.Rgl[0] = cameraParas.R[0], cameraParas.Rgl[1] = cameraParas.R[1], cameraParas.Rgl[2] = cameraParas.R[2], cameraParas.Rgl[3] = 0.0;
	cameraParas.Rgl[4] = cameraParas.R[3], cameraParas.Rgl[5] = cameraParas.R[4], cameraParas.Rgl[6] = cameraParas.R[5], cameraParas.Rgl[7] = 0.0;
	cameraParas.Rgl[8] = cameraParas.R[6], cameraParas.Rgl[9] = cameraParas.R[7], cameraParas.Rgl[10] = cameraParas.R[8], cameraParas.Rgl[11] = 0.0;
	cameraParas.Rgl[12] = 0, cameraParas.Rgl[13] = 0, cameraParas.Rgl[14] = 0, cameraParas.Rgl[15] = 1.0;

	mat_mul(iR, cameraParas.T, center, 3, 3, 1); //Center = -iR*T 
	cameraParas.camCenter[0] = -center[0], cameraParas.camCenter[1] = -center[1], cameraParas.camCenter[2] = -center[2];

	delete[]pts, delete[]t3D;
	if (ninliers < cameraParas.ninlierThresh)
	{
		printf("Estimated pose for View (%d, %d).. fails ... low inliers (%d/%d). Camera center: %.4f %.4f %.4f \n\n", cameraID, timeID, ninliers, npts, cameraParas.T[0], cameraParas.T[1], cameraParas.T[2]);
		return 1;
	}
	else
	{
		printf("Estimated pose for View (%d, %d).. succeds ... inliers (%d/%d). Camera center: %.4f %.4f %.4f \n\n", cameraID, timeID, ninliers, npts, cameraParas.T[0], cameraParas.T[1], cameraParas.T[2]);
		return 0;
	}
}
int LocalizeCameraFromCorpusDriver(char *Path, int StartTime, int StopTime, bool RunMatching, int nCams, int selectedCams, bool distortionCorrected)
{
	int width = 1920, height = 1080;
	Corpus corpusData;
	if (RunMatching)
		ReadCorpusInfo(Path, corpusData, false, false);
	else
		ReadCorpusInfo(Path, corpusData, false, true);

	double start = omp_get_wtime();
	char Fname[200];

	if (RunMatching)
	{
		int toMatch;
		const int ninlierThresh = 40;
		for (int timeID = StartTime; timeID <= StopTime; timeID++)
		{
			vector<int> CorpusViewToMatch;
			CorpusViewToMatch.reserve(corpusData.nCamera);

			sprintf(Fname, "%s/%d/ToMatch.txt", Path, selectedCams);
			FILE *fp = fopen(Fname, "r");
			if (fp == NULL)
				printf("Cannot read %s\n", Fname);
			while (fscanf(fp, "%d ", &toMatch) != EOF)
				CorpusViewToMatch.push_back(toMatch);
			fclose(fp);

			MatchCameraToCorpus(Path, corpusData, selectedCams, timeID, CorpusViewToMatch, ninlierThresh);
		}
	}
	else
	{
		bool fixedIntrinsc = true, fixedDistortion = true;
		CameraData *AllCamsInfo = new CameraData[nCams];
		if (ReadIntrinsicResults(Path, AllCamsInfo, nCams) != 0)
		{
			//Uncalibrated cam-->have to search for focal length
			fixedIntrinsc = false, fixedDistortion = false;
			for (int ii = 0; ii < nCams; ii++)
			{
				double focal = 0.945*max(width, height);
				AllCamsInfo[ii].intrinsic[0] = focal, AllCamsInfo[ii].intrinsic[1] = focal, AllCamsInfo[ii].intrinsic[2] = 0,
					AllCamsInfo[ii].intrinsic[3] = width / 2, AllCamsInfo[ii].intrinsic[4] = height / 2;
				GetKFromIntrinsic(AllCamsInfo[ii]);

				AllCamsInfo[ii].threshold = 3.0, AllCamsInfo[ii].ninlierThresh = 40;
				for (int jj = 0; jj < 7; jj++)
					AllCamsInfo[ii].distortion[jj] = 0.0;
			}
		}
		else
		{
			for (int ii = 0; ii < nCams; ii++)
			{
				AllCamsInfo[ii].LensModel = RADIAL_TANGENTIAL_PRISM, AllCamsInfo[ii].threshold = 3.0, AllCamsInfo[ii].ninlierThresh = 40;
				if (distortionCorrected)
					for (int jj = 0; jj < 7; jj++)
						AllCamsInfo[ii].distortion[jj] = 0.0;
			}
		}

		vector<int> computedTime; computedTime.reserve(StopTime - StartTime + 1);
		CameraData *SelectedCameraInfo = new CameraData[StopTime - StartTime + 1];
		for (int timeID = StartTime; timeID <= StopTime; timeID++)
		{
			computedTime.clear();
			CopyCamereInfo(AllCamsInfo[selectedCams], SelectedCameraInfo[timeID - StartTime]);
			if (EstimateCameraPoseFromCorpus(Path, corpusData, SelectedCameraInfo[timeID - StartTime], selectedCams, fixedIntrinsc, fixedDistortion, timeID))
				computedTime.push_back(-1);
			else
				computedTime.push_back(timeID - StartTime);

			SaveVideoCameraPosesGL(Path, SelectedCameraInfo, computedTime, selectedCams, StartTime);
		}
		printf("Finished estimating poses for camera %d\n", selectedCams);
		SaveVideoCameraPosesGL(Path, SelectedCameraInfo, computedTime, selectedCams, StartTime);

		delete[]AllCamsInfo, delete[]SelectedCameraInfo;
	}

	printf("Total time %d: %.3fs\n", selectedCams, omp_get_wtime() - start);

	return 0;
}

void computeFmatfromKRT(CameraData *CameraInfo, int nvews, int *selectedCams, double *Fmat)
{
	int ii;
	double tmat[9], tmat2[9];
	double K1[9] = { CameraInfo[selectedCams[0]].K[0], CameraInfo[selectedCams[0]].K[1], CameraInfo[selectedCams[0]].K[2],
		0, CameraInfo[selectedCams[0]].K[4], CameraInfo[selectedCams[0]].K[5],
		0, 0, 1.0 };
	double K2[9] = { CameraInfo[selectedCams[1]].K[0], CameraInfo[selectedCams[1]].K[1], CameraInfo[selectedCams[1]].K[2],
		0, CameraInfo[selectedCams[1]].K[4], CameraInfo[selectedCams[1]].K[5],
		0, 0, 1.0 };
	double rt1[6] = { CameraInfo[selectedCams[0]].rt[0], CameraInfo[selectedCams[0]].rt[1], CameraInfo[selectedCams[0]].rt[2],
		CameraInfo[selectedCams[0]].rt[3], CameraInfo[selectedCams[0]].rt[4], CameraInfo[selectedCams[0]].rt[5] };
	double rt2[6] = { CameraInfo[selectedCams[1]].rt[0], CameraInfo[selectedCams[1]].rt[1], CameraInfo[selectedCams[1]].rt[2],
		CameraInfo[selectedCams[1]].rt[3], CameraInfo[selectedCams[1]].rt[4], CameraInfo[selectedCams[1]].rt[5] };

	double RT1[16], RT2[16], R1[9], R2[9], T1[3], T2[3];
	GetRTFromrt(rt1, R1, T1);
	RT1[0] = R1[0], RT1[1] = R1[1], RT1[2] = R1[2], RT1[3] = T1[0];
	RT1[4] = R1[3], RT1[5] = R1[4], RT1[6] = R1[5], RT1[7] = T1[1];
	RT1[8] = R1[6], RT1[9] = R1[7], RT1[10] = R1[8], RT1[11] = T1[2];
	RT1[12] = 0, RT1[13] = 0, RT1[14] = 0, RT1[15] = 1;

	GetRTFromrt(rt2, R2, T2);
	RT2[0] = R2[0], RT2[1] = R2[1], RT2[2] = R2[2], RT2[3] = T2[0];
	RT2[4] = R2[3], RT2[5] = R2[4], RT2[6] = R2[5], RT2[7] = T2[1];
	RT2[8] = R2[6], RT2[9] = R2[7], RT2[10] = R2[8], RT2[11] = T2[2];
	RT2[12] = 0, RT2[13] = 0, RT2[14] = 0, RT2[15] = 1;

	double iRT1[16], RT12[16], R12[9], T12[3];
	mat_invert(RT1, iRT1, 4);
	mat_mul(RT2, iRT1, RT12, 4, 4, 4);
	DesembleRT(R12, T12, RT12);

	double Emat12[9], Tx[9];
	Tx[0] = 0.0, Tx[1] = -T12[2], Tx[2] = T12[1];
	Tx[3] = T12[2], Tx[4] = 0.0, Tx[5] = -T12[0];
	Tx[6] = -T12[1], Tx[7] = T12[0], Tx[8] = 0.0;

	mat_mul(Tx, R12, Emat12, 3, 3, 3);

	double iK1[9], iK2[9];
	mat_invert(K1, iK1, 3);
	mat_invert(K2, iK2, 3);
	mat_transpose(iK2, tmat, 3, 3);
	mat_mul(tmat, Emat12, tmat2, 3, 3, 3);
	mat_mul(tmat2, iK1, Fmat, 3, 3, 3);

	for (ii = 0; ii < 9; ii++)
		Fmat[ii] = Fmat[ii] / Fmat[8];

	return;
}
//if ChooseCorpusView != -1, selectedCams and seletectedTime will be overwritten
void computeFmatfromKRT(CorpusandVideo &CorpusandVideoInfo, int *selectedCams, int *seletectedTime, int ChooseCorpusView1, int ChooseCorpusView2, double *Fmat)
{
	int ii, startTime = CorpusandVideoInfo.startTime, stopTime = CorpusandVideoInfo.stopTime;
	double tmat[9], tmat2[9];
	double K1[9], K2[9], rt1[6], rt2[6];

	if (ChooseCorpusView1 != -1)
	{
		K1[0] = CorpusandVideoInfo.CorpusInfo[ChooseCorpusView1].K[0], K1[1] = CorpusandVideoInfo.CorpusInfo[ChooseCorpusView1].K[1], K1[2] = CorpusandVideoInfo.CorpusInfo[ChooseCorpusView1].K[2],
			K1[3] = 0, K1[4] = CorpusandVideoInfo.CorpusInfo[ChooseCorpusView1].K[4], K1[5] = CorpusandVideoInfo.CorpusInfo[ChooseCorpusView1].K[5],
			K1[6] = 0, K1[7] = 0, K1[8] = 1.0;

		rt1[0] = CorpusandVideoInfo.CorpusInfo[ChooseCorpusView1].rt[0], rt1[1] = CorpusandVideoInfo.CorpusInfo[ChooseCorpusView1].rt[1], rt1[2] = CorpusandVideoInfo.CorpusInfo[ChooseCorpusView1].rt[2],
			rt1[3] = CorpusandVideoInfo.CorpusInfo[ChooseCorpusView1].rt[3], rt1[4] = CorpusandVideoInfo.CorpusInfo[ChooseCorpusView1].rt[4], rt1[5] = CorpusandVideoInfo.CorpusInfo[ChooseCorpusView1].rt[5];
	}
	else
	{
		int ID = selectedCams[0] * (stopTime - startTime + 1) + seletectedTime[0];
		K1[0] = CorpusandVideoInfo.VideoInfo[ID].K[0], K1[1] = CorpusandVideoInfo.VideoInfo[ID].K[1], K1[2] = CorpusandVideoInfo.VideoInfo[ID].K[2],
			K1[3] = 0, K1[4] = CorpusandVideoInfo.VideoInfo[ID].K[4], K1[5] = CorpusandVideoInfo.VideoInfo[ID].K[5],
			K1[6] = 0, K1[7] = 0, K1[8] = 1.0;

		rt1[0] = CorpusandVideoInfo.VideoInfo[ID].rt[0], rt1[1] = CorpusandVideoInfo.VideoInfo[ID].rt[1], rt1[2] = CorpusandVideoInfo.VideoInfo[ID].rt[2],
			rt1[3] = CorpusandVideoInfo.VideoInfo[ID].rt[3], rt1[4] = CorpusandVideoInfo.VideoInfo[ID].rt[4], rt1[5] = CorpusandVideoInfo.VideoInfo[ID].rt[5];
	}

	if (ChooseCorpusView2 != -1)
	{
		K2[0] = CorpusandVideoInfo.CorpusInfo[ChooseCorpusView2].K[0], K2[1] = CorpusandVideoInfo.CorpusInfo[ChooseCorpusView2].K[1], K2[2] = CorpusandVideoInfo.CorpusInfo[ChooseCorpusView2].K[2],
			K2[3] = 0, K2[4] = CorpusandVideoInfo.CorpusInfo[ChooseCorpusView2].K[4], K2[5] = CorpusandVideoInfo.CorpusInfo[ChooseCorpusView2].K[5],
			K2[6] = 0, K2[7] = 0, K2[8] = 1.0;

		rt2[0] = CorpusandVideoInfo.CorpusInfo[ChooseCorpusView2].rt[0], rt2[1] = CorpusandVideoInfo.CorpusInfo[ChooseCorpusView2].rt[1], rt2[2] = CorpusandVideoInfo.CorpusInfo[ChooseCorpusView2].rt[2],
			rt2[3] = CorpusandVideoInfo.CorpusInfo[ChooseCorpusView2].rt[3], rt2[4] = CorpusandVideoInfo.CorpusInfo[ChooseCorpusView2].rt[4], rt2[5] = CorpusandVideoInfo.CorpusInfo[ChooseCorpusView2].rt[5];
	}
	else
	{
		int ID = selectedCams[1] * (stopTime - startTime + 1) + seletectedTime[1];
		K2[0] = CorpusandVideoInfo.VideoInfo[ID].K[0], K2[1] = CorpusandVideoInfo.VideoInfo[ID].K[1], K2[2] = CorpusandVideoInfo.VideoInfo[ID].K[2],
			K2[3] = 0, K2[4] = CorpusandVideoInfo.VideoInfo[ID].K[4], K2[5] = CorpusandVideoInfo.VideoInfo[ID].K[5],
			K2[6] = 0, K2[7] = 0, K2[8] = 1.0;

		rt2[0] = CorpusandVideoInfo.VideoInfo[ID].rt[0], rt2[1] = CorpusandVideoInfo.VideoInfo[ID].rt[1], rt2[2] = CorpusandVideoInfo.VideoInfo[ID].rt[2],
			rt2[3] = CorpusandVideoInfo.VideoInfo[ID].rt[3], rt2[4] = CorpusandVideoInfo.VideoInfo[ID].rt[4], rt2[5] = CorpusandVideoInfo.VideoInfo[ID].rt[5];
	}


	double RT1[16], RT2[16], R1[9], R2[9], T1[3], T2[3];
	GetRTFromrt(rt1, R1, T1);
	RT1[0] = R1[0], RT1[1] = R1[1], RT1[2] = R1[2], RT1[3] = T1[0];
	RT1[4] = R1[3], RT1[5] = R1[4], RT1[6] = R1[5], RT1[7] = T1[1];
	RT1[8] = R1[6], RT1[9] = R1[7], RT1[10] = R1[8], RT1[11] = T1[2];
	RT1[12] = 0, RT1[13] = 0, RT1[14] = 0, RT1[15] = 1;

	GetRTFromrt(rt2, R2, T2);
	RT2[0] = R2[0], RT2[1] = R2[1], RT2[2] = R2[2], RT2[3] = T2[0];
	RT2[4] = R2[3], RT2[5] = R2[4], RT2[6] = R2[5], RT2[7] = T2[1];
	RT2[8] = R2[6], RT2[9] = R2[7], RT2[10] = R2[8], RT2[11] = T2[2];
	RT2[12] = 0, RT2[13] = 0, RT2[14] = 0, RT2[15] = 1;

	double iRT1[16], RT12[16], R12[9], T12[3];
	mat_invert(RT1, iRT1, 4);
	mat_mul(RT2, iRT1, RT12, 4, 4, 4);
	DesembleRT(R12, T12, RT12);

	double Emat12[9], Tx[9];
	Tx[0] = 0.0, Tx[1] = -T12[2], Tx[2] = T12[1];
	Tx[3] = T12[2], Tx[4] = 0.0, Tx[5] = -T12[0];
	Tx[6] = -T12[1], Tx[7] = T12[0], Tx[8] = 0.0;

	mat_mul(Tx, R12, Emat12, 3, 3, 3);

	double iK1[9], iK2[9];
	mat_invert(K1, iK1, 3);
	mat_invert(K2, iK2, 3);
	mat_transpose(iK2, tmat, 3, 3);
	mat_mul(tmat, Emat12, tmat2, 3, 3, 3);
	mat_mul(tmat2, iK1, Fmat, 3, 3, 3);

	for (ii = 0; ii < 9; ii++)
		Fmat[ii] = Fmat[ii] / Fmat[8];

	return;
}




