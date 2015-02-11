#include "Ultility.h"
#include "ImagePro.h"
#include "Geometry.h"
#include "BAutil.h"
using namespace cv;
using namespace std;

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
#include <d.8fcn.h>
#define FREE_MYLIB dlclose
#define GET_MYPROC dlsym
#endif
#endif

int siftgpu(char *Fname1, char *Fname2, const float nndrRatio, const double fractionMatchesDisplayed)

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
	int numKeys1, numKeys2, descriptorSize = SIFTBINS;
	vector<float > desc1, desc2; desc1.reserve(MAXSIFTPTS * descriptorSize), desc2.reserve(MAXSIFTPTS * descriptorSize);
	vector<SiftGPU::SiftKeypoint> keys1, keys2; keys1.reserve(MAXSIFTPTS), keys2.reserve(MAXSIFTPTS);

	int totalPts = 0;
	char Fname[200];

	double start;
	if (sift->RunSIFT(Fname1)) //You can have at most one OpenGL-based SiftGPU (per process)--> no omp can be used
	{
		//sprintf(Fname, "%s/%d.sift", Path, ii);sift->SaveSIFT(Fname);
		numKeys1 = sift->GetFeatureNum();
		keys1.resize(numKeys1);    desc1.resize(descriptorSize * numKeys1);
		sift->GetFeatureVector(&keys1[0], &desc1[0]);

		//sprintf(Fname, "K%d.dat", 0); WriteKPointsBinarySIFTGPU(Fname, keys1);
		sprintf(Fname, "D%d.dat", 0); WriteDescriptorBinarySIFTGPU(Fname, desc1);
		printf("#%d sift deteced...\n", numKeys1);
	}
	if (sift->RunSIFT(Fname2)) //You can have at most one OpenGL-based SiftGPU (per process)--> no omp can be used
	{
		//sprintf(Fname, "%s/%d.sift", Path, ii);sift->SaveSIFT(Fname);
		numKeys2 = sift->GetFeatureNum();
		keys2.resize(numKeys2);    desc2.resize(descriptorSize * numKeys2);
		sift->GetFeatureVector(&keys2[0], &desc2[0]);

		//sprintf(Fname, "K%d.dat", 1); WriteKPointsBinarySIFTGPU(Fname, keys2);
		sprintf(Fname, "D%d.dat", 1); WriteDescriptorBinarySIFTGPU(Fname, desc2);
		printf("#%d sift deteced...\n", numKeys2);
	}
	//SIFT DECTION: ENDS

	///SIFT MATCHING: START
	SiftMatchGPU* matcher = pCreateNewSiftMatchGPU(8192);
	matcher->VerifyContextGL(); //must call once
	int(*match_buf)[2] = new int[MAXSIFTPTS][2];

	vector<Point2i> RawPairWiseMatchID; RawPairWiseMatchID.reserve(10000);

	const int ninlierThesh = 50;
	bool BinaryDesc = false, useBFMatcher = false; // SET TO TRUE TO USE BRUTE FORCE MATCHER
	const int knn = 2, ntrees = 4, maxLeafCheck = 128;

	start = omp_get_wtime();
	printf("Running feature matching...\n");
	sprintf(Fname, "D%d.dat", 0);
	Mat descriptors1 = ReadDescriptorBinarySIFTGPU(Fname);
	if (descriptors1.rows == 1)
		return 1;

	sprintf(Fname, "D%d.dat", 1);
	Mat descriptors2 = ReadDescriptorBinarySIFTGPU(Fname);
	if (descriptors2.rows == 1)
		return 1;

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
	if (!useBFMatcher)
	{
		for (int i = 0; i < descriptors2.rows; ++i)
		{
			int ind1 = indices.at<int>(i, 0);
			if (indices.at<int>(i, 0) >= 0 && indices.at<int>(i, 1) >= 0 && dists.at<float>(i, 0) <= nndrRatio * dists.at<float>(i, 1))
				RawPairWiseMatchID.push_back(Point2i(ind1, i));
		}
	}
	else
	{
		for (unsigned int i = 0; i < matches.size(); ++i)
			if (matches.at(i).size() == 2 && matches.at(i).at(0).distance <= nndrRatio * matches.at(i).at(1).distance)
				RawPairWiseMatchID.push_back(Point2i(matches.at(i).at(0).trainIdx, i));
	}
	printf("%d matches found... in %.2fs\n", RawPairWiseMatchID.size(), omp_get_wtime() - start);

	KeyPoint key;
	vector<int> CorresID;
	vector<Point2d> Keys1, Keys2;
	for (int i = 0; i < keys1.size(); i++)
		Keys1.push_back(Point2d(keys1[i].x, keys1[i].y));
	for (int i = 0; i < keys2.size(); i++)
		Keys2.push_back(Point2d(keys2[i].x, keys2[i].y));
	for (int i = 0; i < RawPairWiseMatchID.size(); ++i)
		CorresID.push_back(RawPairWiseMatchID[i].x), CorresID.push_back(RawPairWiseMatchID[i].y);

	int nchannels = 3;
	IplImage *Img1 = cvLoadImage(Fname1, nchannels == 3 ? 1 : 0);
	if (Img1->imageData == NULL)
	{
		printf("Cannot load %s\n", Fname1);
		return 1;
	}
	IplImage *Img2 = cvLoadImage(Fname2, nchannels == 3 ? 1 : 0);
	if (Img2->imageData == NULL)
	{
		printf("Cannot load %s\n", Fname2);
		return 1;
	}

	IplImage* correspond = cvCreateImage(cvSize(Img1->width + Img2->width, Img1->height), 8, nchannels);
	cvSetImageROI(correspond, cvRect(0, 0, Img1->width, Img1->height));
	cvCopy(Img1, correspond);
	cvSetImageROI(correspond, cvRect(Img1->width, 0, correspond->width, correspond->height));
	cvCopy(Img2, correspond);
	cvResetImageROI(correspond);
	DisplayImageCorrespondence(correspond, Img1->width, 0, Keys1, Keys2, CorresID, fractionMatchesDisplayed);

	delete[] match_buf;
	delete sift;
	delete matcher;
	FREE_MYLIB(hsiftgpu);

	return 0;
}

void dec2bin(int dec, int*bin, int num_bin)
{
	bool stop = false;
	int ii, digit = 0;
	int temp[32];

	while (!stop)
	{
		temp[digit] = dec % 2;
		dec /= 2;
		digit++;
		if (dec == 0)
			stop = true;
	}

	if (digit > num_bin)
		Beep(1000, 200);

	for (ii = 0; ii < num_bin - digit; ii++)
		bin[ii] = 0;
	for (ii = digit - 1; ii >= 0; ii--)
		bin[num_bin - ii - 1] = temp[ii];

	return;
}

double nChoosek(int n, int k)
{
	if (n < 0 || k < 0)
		return 0.0;
	if (n < k)
		return 0.0;  // special case
	if (n == k)
		return 1.0;

	int iMax;
	double delta;

	if (k < n - k) // ex: Choose(100,3)
	{
		delta = 1.0*(n - k);
		iMax = k;
	}
	else         // ex: Choose(100,97)
	{
		delta = 1.0*k;
		iMax = n - k;
	}

	double res = delta + 1.0;
	for (int i = 2; i <= iMax; i++)
		res = res * (delta + i) / i;

	return res;
}
int MyFtoI(double W)
{
	if (W >= 0.0)
		return (int)(W + 0.5);
	else
		return (int)(W - 0.5);

	return 0;
}
bool IsNumber(double x)
{
	// This looks like it should always be true, but it's false if x is a NaN.
	return (x == x);
}
bool IsFiniteNumber(double x)
{
	return (x <= DBL_MAX && x >= -DBL_MAX);
}

double UniformNoise(double High, double Low)
{
	double noise = 1.0*rand() / RAND_MAX;
	return (High - Low)*noise + Low;
}
double gaussian_noise(double mean, double std)
{
	double u1 = 0.0, u2 = 0.0;
	while (abs(u1) < DBL_EPSILON || abs(u2) < DBL_EPSILON) //avoid 0.0 case since log(0) = inf
	{
		u1 = 1.0 * rand() / RAND_MAX;
		u2 = 1.0 * rand() / RAND_MAX;
	}

	double normal_noise = sqrt(-2.0 * log(u1)) * cos(2.0 * Pi * u2);
	return mean + std * normal_noise;
}

double L1norm(vector<double>A)
{
	double res = 0.0;
	for (int ii = 0; ii < A.size(); ii++)
		res += abs(A[ii]);
	return res;
}
void normalize(double *x, int dim)
{
	double tt = 0;
	for (int ii = 0; ii < dim; ii++)
		tt += x[ii] * x[ii];
	tt = sqrt(tt);
	for (int ii = 0; ii < dim; ii++)
		x[ii] = x[ii] / tt;
	return;
}
float MeanArray(float *data, int length)
{
	double mean = 0.0;
	for (int ii = 0; ii < length; ii++)
		mean += data[ii];
	return mean / length;
}
double MeanArray(double *data, int length)
{
	double mean = 0.0;
	for (int ii = 0; ii < length; ii++)
		mean += data[ii];
	return mean / length;
}
double VarianceArray(double *data, int length, double mean)
{
	if (mean == NULL)
		mean = MeanArray(data, length);

	double var = 0.0;
	for (int ii = 0; ii < length; ii++)
		var += pow(data[ii] - mean, 2);
	return var / (length - 1);
}
double MeanArray(vector<double>&data)
{
	double mean = 0.0;
	for (int ii = 0; ii < data.size(); ii++)
		mean += data[ii];
	return mean / data.size();
}
double VarianceArray(vector<double>&data, double mean)
{
	if (mean == NULL)
		mean = MeanArray(data);

	double var = 0.0;
	for (int ii = 0; ii < data.size(); ii++)
		var += pow(data[ii] - mean, 2);
	return var / (data.size() - 1);
}
double norm_dot_product(double *x, double *y, int dim)
{
	double nx = 0.0, ny = 0.0, dxy = 0.0;
	for (int ii = 0; ii < dim; ii++)
	{
		nx += x[ii] * x[ii];
		ny += y[ii] * y[ii];
		dxy += x[ii] * y[ii];
	}
	double radian = dxy / sqrt(nx*ny);

	return radian;
}
void cross_product(double *x, double *y, double *xy)
{
	xy[0] = x[1] * y[2] - x[2] * y[1];
	xy[1] = x[2] * y[0] - x[0] * y[2];
	xy[2] = x[0] * y[1] - x[1] * y[0];

	return;
}
void conv(float *A, int lenA, float *B, int lenB, float *C)
{
	int nconv;
	int i, j, i1;
	double tmp;

	nconv = lenA + lenB - 1;
	for (i = 0; i < nconv; i++)
	{
		i1 = i;
		tmp = 0.0;
		for (j = 0; j < lenB; j++)
		{
			if (i1 >= 0 && i1 < lenA)
				tmp = tmp + (A[i1] * B[j]);

			i1 = i1 - 1;
			C[i] = (float)tmp;
		}
	}

	return;
}
void conv(double *A, int lenA, double *B, int lenB, double *C)
{
	int nconv;
	int i, j, i1;
	double tmp;

	nconv = lenA + lenB - 1;
	for (i = 0; i < nconv; i++)
	{
		i1 = i;
		tmp = 0.0;
		for (j = 0; j < lenB; j++)
		{
			if (i1 >= 0 && i1 < lenA)
				tmp = tmp + (A[i1] * B[j]);

			i1 = i1 - 1;
			C[i] = tmp;
		}
	}

	return;
}
void ZNCC1D(float *A, const int dimA, float *B, const int dimB, float *Result, float *nB)
{
	//Matlab normxcorr2
	const int sdimA = dimA - 1, dimnB = 2 * (dimA - 1) + dimB, dimRes = dimB + dimA - 1;
	bool createMem = false;
	if (nB == NULL)
	{
		createMem = true;
		nB = new float[dimnB];
	}

	omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for
	for (int ii = 0; ii < sdimA; ii++)
		nB[ii] = 0;
#pragma omp parallel for
	for (int ii = sdimA; ii < sdimA + dimB; ii++)
		nB[ii] = B[ii - sdimA];
#pragma omp parallel for
	for (int ii = sdimA + dimB; ii < dimnB; ii++)
		nB[ii] = 0;

	Mat ma(1, dimA, CV_32F, A);
	Mat mb(1, dimnB, CV_32F, nB);

	Mat result(1, dimRes, CV_32F);
	matchTemplate(mb, ma, result, 5);

	for (int ii = 0; ii < dimRes; ii++)
		Result[ii] = result.at<float>(ii);

	if (createMem)
		delete[]nB;

	return;
}
void ZNCC1D(double *A, int Asize, double *B, int Bsize, double *Result)
{
	//Matlab normxcorr2
	int ii, jj;

	double A2 = 0.0, meanA = MeanArray(A, Asize);
	double *ZNA = new double[Asize], *ZNB = new double[Asize];
	for (ii = 0; ii < Asize; ii++)
	{
		ZNA[ii] = A[ii] - meanA;
		A2 += pow(ZNA[ii], 2);
	}

	for (ii = 0; ii < Asize; ii++)
	{
		double meanB = 0.0, allZeros = 0;
		for (jj = 0; jj <= ii; jj++)
		{
			meanB += B[jj];
			allZeros += abs(B[jj]);
		}
		if (allZeros < 1e-6)
			Result[ii] = 0.0;
		else
		{
			meanB = meanB / Asize;

			for (jj = 0; jj < Asize - ii - 1; jj++)
				ZNB[jj] = 0.0 - meanB;
			for (jj = 0; jj <= ii; jj++)
				ZNB[Asize - ii - 1 + jj] = B[jj] - meanB;

			double B2 = 0, AB = 0.0;
			for (jj = 0; jj < Asize; jj++)
				AB += ZNA[jj] * ZNB[jj], B2 += pow(ZNB[jj], 2);

			double zncc = AB / sqrt(A2*B2);
			Result[ii] = zncc;
		}
	}

	for (ii = 1; ii < Bsize - Asize + 1; ii++)
	{
		double meanB = 0.0, allZeros = 0;
		for (jj = ii; jj < ii + Asize; jj++)
		{
			meanB += B[jj];
			allZeros += abs(B[jj]);
		}
		if (allZeros < 1.0e-6)
			Result[ii - 1 + Asize] = 0.0;
		else
		{
			meanB = meanB / Asize;

			for (jj = 0; jj < Asize; jj++)
				ZNB[jj] = B[jj + ii] - meanB;

			double B2 = 0, AB = 0.0;
			for (jj = 0; jj < Asize; jj++)
				AB += ZNA[jj] * ZNB[jj], B2 += pow(ZNB[jj], 2);

			double zncc = AB / sqrt(A2*B2);
			Result[ii - 1 + Asize] = zncc;
		}
	}

	for (ii = 1; ii < Asize; ii++)
	{
		double meanB = 0.0, allZeros = 0;
		for (jj = Asize - ii; jj > 0; jj--)
		{
			meanB += B[Bsize - jj];
			allZeros += abs(B[Bsize - jj]);
		}
		if (allZeros < 1e-6)
			Result[ii - 1 + Bsize] = 0.0;
		else
		{
			meanB = meanB / Asize;

			for (jj = Asize - ii; jj > 0; jj--)
				ZNB[Asize - ii - jj] = B[Bsize - jj] - meanB;
			for (jj = Asize - ii; jj < Asize; jj++)
				ZNB[jj] = 0.0 - meanB;

			double B2 = 0, AB = 0.0;
			for (jj = 0; jj < Asize; jj++)
				AB += ZNA[jj] * ZNB[jj], B2 += pow(ZNB[jj], 2);

			double zncc = AB / sqrt(A2*B2);
			Result[ii - 1 + Bsize] = zncc;
		}
	}

	delete[]ZNA, delete[]ZNB;

	return;
}
double ComputeZNCCPatch(double *RefPatch, double *TarPatch, int hsubset, int nchannels, double *T)
{
	int i, kk, iii, jjj;

	FILE *fp1, *fp2;
	bool printout = false;
	if (printout)
	{
		fp1 = fopen("C:/temp/src.txt", "w+");
		fp2 = fopen("C:/temp/tar.txt", "w+");
	}

	int TimgS = 2 * hsubset + 1, Tlength = TimgS*TimgS;
	bool createMem = false;
	if (T == NULL)
	{
		createMem = true;
		T = new double[2 * Tlength*nchannels];
	}
	double ZNCC = 0.0;

	int m = 0;
	double t_1, t_2, t_3, t_4, t_5, t_f = 0.0, t_g = 0.0;
	for (jjj = 0; jjj < TimgS; jjj++)
	{
		for (iii = 0; iii < TimgS; iii++)
		{
			for (kk = 0; kk < nchannels; kk++)
			{
				i = iii + jjj*TimgS + kk*Tlength;
				T[2 * m] = RefPatch[i], T[2 * m + 1] = TarPatch[i];
				t_f += T[2 * m], t_g += T[2 * m + 1];

				if (printout)
					fprintf(fp1, "%.4f ", T[2 * m]), fprintf(fp2, "%.4f ", T[2 * m + 1]);
				m++;
			}
		}
		if (printout)
		{
			fprintf(fp1, "\n"), fprintf(fp2, "\n");
		}
	}
	if (printout)
	{
		fclose(fp1), fclose(fp2);
	}

	t_f = t_f / m;
	t_g = t_g / m;
	t_1 = 0.0, t_2 = 0.0, t_3 = 0.0;
	for (i = 0; i < m; i++)
	{
		t_4 = T[2 * i] - t_f, t_5 = T[2 * i + 1] - t_g;
		t_1 += 1.0*t_4*t_5, t_2 += 1.0*t_4*t_4, t_3 += 1.0*t_5*t_5;
	}

	t_2 = sqrt(t_2*t_3);
	if (t_2 < 1e-10)
		t_2 = 1e-10;

	ZNCC = t_1 / t_2; //This is the zncc score
	if (abs(ZNCC) > 1.0)
		ZNCC = 0.0;

	if (createMem)
		delete[]T;

	return ZNCC;
}
void mat_invert(double* mat, double* imat, int dims)
{
	if (dims == 3)
	{
		// only work for 3x3
		double a = mat[0], b = mat[1], c = mat[2], d = mat[3], e = mat[4], f = mat[5], g = mat[6], h = mat[7], k = mat[8];
		double A = e*k - f*h, B = c*h - b*k, C = b*f - c*e;
		double D = f*g - d*k, E = a*k - c*g, F = c*d - a*f;
		double G = d*h - e*g, H = b*g - a*h, K = a*e - b*d;
		double DET = a*A + b*D + c*G;
		imat[0] = A / DET, imat[1] = B / DET, imat[2] = C / DET;
		imat[3] = D / DET, imat[4] = E / DET, imat[5] = F / DET,
			imat[6] = G / DET, imat[7] = H / DET, imat[8] = K / DET;
	}
	else
	{
		Mat inMat = Mat(dims, dims, CV_64FC1, mat);
		Mat outMat = inMat.inv(DECOMP_SVD);
		for (int jj = 0; jj < dims; jj++)
			for (int ii = 0; ii < dims; ii++)
				imat[ii + jj*dims] = outMat.at<double>(jj, ii);
	}

	return;
}
void mat_invert(float* mat, float* imat, int dims)
{
	if (dims == 3)
	{
		// only work for 3x3
		float a = mat[0], b = mat[1], c = mat[2], d = mat[3], e = mat[4], f = mat[5], g = mat[6], h = mat[7], k = mat[8];
		float A = e*k - f*h, B = c*h - b*k, C = b*f - c*e;
		float D = f*g - d*k, E = a*k - c*g, F = c*d - a*f;
		float G = d*h - e*g, H = b*g - a*h, K = a*e - b*d;
		float DET = a*A + b*D + c*G;
		imat[0] = A / DET, imat[1] = B / DET, imat[2] = C / DET;
		imat[3] = D / DET, imat[4] = E / DET, imat[5] = F / DET,
			imat[6] = G / DET, imat[7] = H / DET, imat[8] = K / DET;
	}
	else
	{
		Mat inMat = Mat(dims, dims, CV_32FC1, mat);
		Mat outMat = inMat.inv(DECOMP_SVD);
		for (int jj = 0; jj < dims; jj++)
			for (int ii = 0; ii < dims; ii++)
				imat[ii + jj*dims] = outMat.at<float>(jj, ii);
	}

	return;
}
void mat_mul(double *aa, double *bb, double *out, int rowa, int col_row, int colb)
{
	int ii, jj, kk;
	for (ii = 0; ii < rowa*colb; ii++)
		out[ii] = 0;

	for (ii = 0; ii < rowa; ii++)
	{
		for (jj = 0; jj < colb; jj++)
		{
			for (kk = 0; kk < col_row; kk++)
				out[ii*colb + jj] += aa[ii*col_row + kk] * bb[kk*colb + jj];
		}
	}

	return;
}
void mat_add(double *aa, double *bb, double* cc, int row, int col, double scale_a, double scale_b)
{
	int ii, jj;

	for (ii = 0; ii < row; ii++)
		for (jj = 0; jj < col; jj++)
			cc[ii*col + jj] = scale_a*aa[ii*col + jj] + scale_b*bb[ii*col + jj];

	return;
}
void mat_subtract(double *aa, double *bb, double* cc, int row, int col, double scale_a, double scale_b)
{
	int ii, jj;

	for (ii = 0; ii < row; ii++)
		for (jj = 0; jj < col; jj++)
			cc[ii*col + jj] = scale_a*aa[ii*col + jj] - scale_b*bb[ii*col + jj];

	return;
}
void mat_transpose(double *in, double *out, int row_in, int col_in)
{
	int ii, jj;
	for (jj = 0; jj < row_in; jj++)
		for (ii = 0; ii < col_in; ii++)
			out[ii*row_in + jj] = in[jj*col_in + ii];
	return;
}
void mat_completeSym(double *mat, int size, bool upper)
{
	if (upper)
	{
		for (int jj = 0; jj < size; jj++)
			for (int ii = jj; ii < size; ii++)
				mat[jj + ii*size] = mat[ii + jj*size];
	}
	else
	{
		for (int jj = 0; jj < size; jj++)
			for (int ii = jj; ii < size; ii++)
				mat[ii + jj*size] = mat[jj + ii*size];
	}
	return;
}

void LS_Solution_Double(double *lpA, double *lpB, int m, int n)
{
	if (m == n)
	{
		QR_Solution_Double(lpA, lpB, n, n);
		return;
	}

	int i, j, k, n2 = n*n;
	double *A = new double[n2];
	double *B = new double[n];

	for (i = 0; i < n2; i++)
		*(A + i) = 0.0;
	for (i = 0; i < n; i++)
		*(B + i) = 0.0;

	for (k = 0; k < m; k++)
	{
		for (j = 0; j < n; j++)
		{
			for (i = 0; i < n; i++)
			{
				*(A + j*n + i) += (*(lpA + k*n + i))*(*(lpA + k*n + j));
			}

			*(B + j) += (*(lpB + k))*(*(lpA + k*n + j));
		}
	}

	QR_Solution_Double(A, B, n, n);

	for (i = 0; i < n; i++)
		*(lpB + i) = *(B + i);

	delete[]B;
	delete[]A;
	return;
}
void QR_Solution_Double(double *lpA, double *lpB, int m, int n)
{
	if (m > 3000)
	{
		LS_Solution_Double(lpA, lpB, m, n);
		return;
	}

	m_TemplateClass_1<double> m_TempClass;
	m_TempClass.QR_Solution(lpA, lpB, m, n);
	return;
}

void Quick_Sort_Int(int * A, int *B, int low, int high)
{
	m_TemplateClass_1<int> m_TempClass;
	m_TempClass.Quick_Sort(A, B, low, high);
	return;
}
void Quick_Sort_Float(float * A, int *B, int low, int high)
{
	m_TemplateClass_1<float> m_TempClass;
	m_TempClass.Quick_Sort(A, B, low, high);
	return;
}
void Quick_Sort_Double(double * A, int *B, int low, int high)
{
	m_TemplateClass_1<double> m_TempClass;
	m_TempClass.Quick_Sort(A, B, low, high);
	return;
}

bool in_polygon(double u, double v, Point2d *vertex, int num_vertex)
{
	int ii;
	bool position;
	double pi = 3.1415926535897932384626433832795;

	for (ii = 0; ii < num_vertex; ii++)
	{
		if (abs(u - vertex[ii].x) < 0.01 && abs(v - vertex[ii].y) < 0.01)
			return 1;
	}
	double dot = (u - vertex[0].x)*(u - vertex[num_vertex - 1].x) + (v - vertex[0].y)*(v - vertex[num_vertex - 1].y);
	double square1 = (u - vertex[0].x)*(u - vertex[0].x) + (v - vertex[0].y)*(v - vertex[0].y);
	double square2 = (u - vertex[num_vertex - 1].x)*(u - vertex[num_vertex - 1].x) + (v - vertex[num_vertex - 1].y)*(v - vertex[num_vertex - 1].y);
	double angle = acos(dot / sqrt(square1*square2));

	for (ii = 0; ii < num_vertex - 1; ii++)
	{
		dot = (u - vertex[ii].x)*(u - vertex[ii + 1].x) + (v - vertex[ii].y)*(v - vertex[ii + 1].y);
		square1 = (u - vertex[ii].x)*(u - vertex[ii].x) + (v - vertex[ii].y)*(v - vertex[ii].y);
		square2 = (u - vertex[ii + 1].x)*(u - vertex[ii + 1].x) + (v - vertex[ii + 1].y)*(v - vertex[ii + 1].y);

		angle += acos(dot / sqrt(square1*square2));
	}

	angle = angle * 180 / pi;
	if (fabs(angle - 360) <= 2.0)
		position = 1;
	else
		position = 0;

	return position;
}

int ReadDomeVGACalibFile(char *Path, CameraData *AllCamInfo)
{
	const int nCams = 480, nPanels = 20, nCamsPanel = 24;
	char Fname[200];

	double Quaterunion[4], CamCenter[3], T[3];
	for (int jj = 0; jj < nPanels; jj++)
	{
		for (int ii = 0; ii < nCamsPanel; ii++)
		{
			int camID = jj*nCamsPanel + ii;

			sprintf(Fname, "%s/In/Calib/%.2d_%.2d.txt", Path, jj + 1, ii + 1); FILE *fp = fopen(Fname, "r");
			if (fp == NULL)
			{
				printf("Cannot load %s\n", Fname);
				return 1;
			}
			for (int kk = 0; kk < 9; kk++)
				fscanf(fp, "%lf ", &AllCamInfo[camID].K[kk]);
			fclose(fp);
			for (int kk = 0; kk < 7; kk++)
				AllCamInfo[camID].distortion[kk] = 0.0;
			AllCamInfo[camID].LensModel = RADIAL_TANGENTIAL_PRISM;

			sprintf(Fname, "%s/In/Calib/%.2d_%.2d_ext.txt", Path, jj + 1, ii + 1); fp = fopen(Fname, "r");
			if (fp == NULL)
			{
				printf("Cannot load %s\n", Fname);
				return 1;
			}
			fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf ", &Quaterunion[0], &Quaterunion[1], &Quaterunion[2], &Quaterunion[3], &CamCenter[0], &CamCenter[1], &CamCenter[2]);
			fclose(fp);
			ceres::QuaternionToRotation(Quaterunion, AllCamInfo[camID].R);
			mat_mul(AllCamInfo[camID].R, CamCenter, T, 3, 3, 1); //t = -RC
			AllCamInfo[camID].T[0] = -T[0], AllCamInfo[camID].T[1] = -T[1], AllCamInfo[camID].T[2] = -T[2];

			GetIntrinsicFromK(AllCamInfo[camID]);
			GetrtFromRT(AllCamInfo[camID].rt, AllCamInfo[camID].R, AllCamInfo[camID].T);
			AssembleP(AllCamInfo[camID].K, AllCamInfo[camID].R, AllCamInfo[camID].T, AllCamInfo[camID].P);
			GetRCGL(AllCamInfo[camID]);
		}
	}


	sprintf(Fname, "%s/PinfoGL_%d.txt", Path, 0);
	FILE *fp = fopen(Fname, "a+");
	for (int ii = 0; ii < nCams; ii++)
	{
		fprintf(fp, "%d ", ii);
		for (int jj = 0; jj < 16; jj++)
			fprintf(fp, "%.16f ", AllCamInfo[ii].Rgl[jj]);
		for (int jj = 0; jj < 3; jj++)
			fprintf(fp, "%.16f ", AllCamInfo[ii].camCenter[jj]);
		fprintf(fp, "\n");
	}
	fclose(fp);
	return 0;
}
bool LoadTrackData(char* filePath, int CurrentFrame, TrajectoryData &TrajectoryInfo, bool loadVis)
{
	char trackingFilePath[512];
	sprintf(trackingFilePath, "%s/Out/reconResult%.8d.track", filePath, CurrentFrame);
	ifstream fin(trackingFilePath);
	if (fin.is_open() == false)
	{
		printf("There is no trackdata %s\n", trackingFilePath);
		return false;
	}

	char dummy[200];
	fin >> dummy;
	float ver;
	fin >> ver;
	fin >> dummy >> TrajectoryInfo.nViews; //CamNum

	int pt3DNum = 0;
	fin >> dummy >> pt3DNum; //TotalPtNum
	fin >> dummy >> TrajectoryInfo.nTrajectories; //TrackNum

	TrajectoryInfo.cpThreeD = new vector<Point3d>[TrajectoryInfo.nTrajectories];
	TrajectoryInfo.fThreeD = new vector<Point3d>[TrajectoryInfo.nTrajectories];
	TrajectoryInfo.cpNormal = new vector<Point3d>[TrajectoryInfo.nTrajectories];
	TrajectoryInfo.fNormal = new vector<Point3d>[TrajectoryInfo.nTrajectories];

	vector<Point3d>cpThreeD, fThreeD;
	vector<Point3d>cpNormal, fNormal;
	for (int i = 0; i < TrajectoryInfo.nTrajectories; ++i)
	{
		if (i % 100 == 0)
			printf("Loading Tracjectory: %d/%d \r", i, TrajectoryInfo.nTrajectories);

		cpThreeD.clear(), fThreeD.clear(), cpNormal.clear(), fNormal.clear();

		int ptIdx, trackedNum;
		Point3d t3D, cur3D, past3D, future3D;

		//For cuurrentTrackUnit
		fin >> dummy >> ptIdx;  //"Pt3d"
		fin >> cur3D.x >> cur3D.y >> cur3D.z;
		fin >> t3D.x >> t3D.y >> t3D.z >> t3D.x >> t3D.y >> t3D.z >> t3D.x >> t3D.y >> t3D.z;
		//cpThreeD.push_back(cur3D);

		fin >> dummy >> trackedNum;  //"prevTracked"
		for (int t = 0; t < trackedNum; ++t)
		{
			fin >> past3D.x >> past3D.y >> past3D.z;
			fin >> t3D.x >> t3D.y >> t3D.z >> t3D.x >> t3D.y >> t3D.z >> t3D.x >> t3D.y >> t3D.z;
			cpThreeD.push_back(past3D);
			cpNormal.push_back(t3D);
		}
		TrajectoryInfo.cpThreeD[i] = cpThreeD;
		TrajectoryInfo.cpNormal[i] = cpNormal;

		fin >> dummy >> trackedNum;  //"nextTracked"
		for (int t = 0; t < trackedNum; ++t)
		{
			fin >> future3D.x >> future3D.y >> future3D.z;
			fin >> t3D.x >> t3D.y >> t3D.z >> t3D.x >> t3D.y >> t3D.z >> t3D.x >> t3D.y >> t3D.z;
			fThreeD.push_back(future3D);
			fNormal.push_back(t3D);
		}
		TrajectoryInfo.fThreeD[i] = fThreeD;
		TrajectoryInfo.fNormal[i] = fNormal;
	}

	//Load Visibility
	if (!loadVis)
		return true;

	int TrueCamID[480], ii = 0;
	FILE *fp = fopen("D:/Y/camId.txt", "r");
	while (fscanf(fp, "%d ", &TrueCamID[ii]) != EOF)
		ii++;
	fclose(fp);

	TrajectoryInfo.cpVis = new vector<vector<int>>[TrajectoryInfo.nTrajectories];
	TrajectoryInfo.fVis = new vector<vector<int>>[TrajectoryInfo.nTrajectories];

	vector<int>cpVis, fVis;
	fin >> dummy; //Visiblity
	for (int i = 0; i < TrajectoryInfo.nTrajectories; ++i)
	{
		if (i % 100 == 0)
			printf("Loading Visibility: %d/%d \r", i, TrajectoryInfo.nTrajectories);

		int ptIdx, trackedNum, visibleCamNum, visibleCamIdx;
		fin >> dummy >> ptIdx; //PtIdx

		//fin >> dummy >> trackedNum;  //"curTrackedVisibleCam"

		fin >> dummy >> trackedNum;  //"prevTrackedVisibleCam"
		TrajectoryInfo.cpVis[i].reserve(trackedNum);
		for (int t = 0; t < trackedNum; ++t)
		{
			fin >> visibleCamNum;
			for (int v = 0; v < visibleCamNum; ++v)
			{
				fin >> visibleCamIdx;
				//cpVis.push_back visibleCamIdx);
				int trueID = TrueCamID[visibleCamIdx];
				cpVis.push_back(trueID);
			}
			TrajectoryInfo.cpVis[i].push_back(cpVis);
			cpVis.clear();
		}

		fin >> dummy >> trackedNum;  //"nextTrackedVisibleCam"
		TrajectoryInfo.fVis[i].reserve(trackedNum);
		for (int t = 0; t < trackedNum; ++t)
		{
			fin >> visibleCamNum;
			for (int v = 0; v < visibleCamNum; ++v)
			{
				fin >> visibleCamIdx;
				//fVis.push_back(visibleCamIdx);
				int trueID = TrueCamID[visibleCamIdx];
				fVis.push_back(trueID);
			}
			TrajectoryInfo.fVis[i].push_back(fVis);
			fVis.clear();
		}
	}

	return true;
}
void Write3DMemAtThatTime(char *Path, TrajectoryData &TrajectoryInfo, CameraData *AllCamInfo, int refFrame, int curFrame)
{
	double angleThreshold = 0.5;
	char Fname[200];
	double normNormal, normPtoC, angle;
	Point3d t3D, n3D, PtoC;

	sprintf(Fname, "%s/3dGL_%d.xyz", Path, curFrame);	FILE *fp = fopen(Fname, "w+");
	if (curFrame > refFrame)
	{
		int timeOffset = curFrame - refFrame - 1;
		for (int kk = 0; kk < TrajectoryInfo.nTrajectories; kk++)
		{
			if (TrajectoryInfo.fThreeD[kk].size() > timeOffset)
			{
				fprintf(fp, "%f %f %f ", TrajectoryInfo.fThreeD[kk].at(timeOffset).x, TrajectoryInfo.fThreeD[kk].at(timeOffset).y, TrajectoryInfo.fThreeD[kk].at(timeOffset).z);
				fprintf(fp, "%f %f %f ", TrajectoryInfo.fNormal[kk].at(timeOffset).x, TrajectoryInfo.fNormal[kk].at(timeOffset).y, TrajectoryInfo.fNormal[kk].at(timeOffset).z);

				int viewID = 0;
				t3D = TrajectoryInfo.fThreeD[kk].at(timeOffset);
				n3D = TrajectoryInfo.fNormal[kk].at(timeOffset);
				PtoC = Point3d(AllCamInfo[viewID].camCenter[0] - t3D.x, AllCamInfo[viewID].camCenter[1] - t3D.y, AllCamInfo[viewID].camCenter[2] - t3D.z);
				normPtoC = sqrt(pow(PtoC.x, 2) + pow(PtoC.y, 2) + pow(PtoC.z, 2));
				normNormal = sqrt(pow(n3D.x, 2) + pow(n3D.y, 2) + pow(n3D.z, 2));
				angle = (n3D.x*PtoC.x + n3D.y*PtoC.y + n3D.z*PtoC.z) / normNormal / normPtoC;
				if (angle > angleThreshold)
					fprintf(fp, "255 0 0  \n");
				else
					fprintf(fp, "0 255 0  \n");
			}
		}
	}
	else if (curFrame < refFrame)
	{
		int timeOffset = refFrame - curFrame - 1;
		for (int kk = 0; kk < TrajectoryInfo.nTrajectories; kk++)
		{
			if (TrajectoryInfo.cpThreeD[kk].size() > timeOffset)
			{
				fprintf(fp, "%f %f %f ", TrajectoryInfo.cpThreeD[kk].at(timeOffset).x, TrajectoryInfo.cpThreeD[kk].at(timeOffset).y, TrajectoryInfo.cpThreeD[kk].at(timeOffset).z);
				fprintf(fp, "%f %f %f ", TrajectoryInfo.cpNormal[kk].at(timeOffset).x, TrajectoryInfo.cpNormal[kk].at(timeOffset).y, TrajectoryInfo.cpNormal[kk].at(timeOffset).z);

				int viewID = 0;
				t3D = TrajectoryInfo.cpThreeD[kk].at(timeOffset);
				n3D = TrajectoryInfo.cpNormal[kk].at(timeOffset);
				PtoC = Point3d(AllCamInfo[viewID].camCenter[0] - t3D.x, AllCamInfo[viewID].camCenter[1] - t3D.y, AllCamInfo[viewID].camCenter[2] - t3D.z);
				normPtoC = sqrt(pow(PtoC.x, 2) + pow(PtoC.y, 2) + pow(PtoC.z, 2));
				normNormal = sqrt(pow(n3D.x, 2) + pow(n3D.y, 2) + pow(n3D.z, 2));
				angle = (n3D.x*PtoC.x + n3D.y*PtoC.y + n3D.z*PtoC.z) / normNormal / normPtoC;
				if (angle > angleThreshold)
					fprintf(fp, "255 0 0  \n");
				else
					fprintf(fp, "0 255 0  \n");
			}
		}
	}
	fclose(fp);
	return;
}
void Genrate2DTrajectory(char *Path, int CurrentFrame, TrajectoryData InfoTraj, CameraData *AllCamInfo, vector<int> trajectoriesUsed)
{
	char Fname[200];
	int ntrajectoriesUsed = trajectoriesUsed.size();
	if (ntrajectoriesUsed > InfoTraj.nTrajectories)
	{
		printf("# trajectories input error\n");
		return;
	}

	vector<Point2d> *Traj2D = new vector<Point2d>[InfoTraj.nViews];
	vector<int> *TimeLine = new vector<int>[InfoTraj.nViews];

	for (int kk = 0; kk < ntrajectoriesUsed; kk++)
	{
		int sTraj = trajectoriesUsed[kk];
		std::reverse(InfoTraj.cpNormal[sTraj].begin(), InfoTraj.cpNormal[sTraj].end());
		std::reverse(InfoTraj.cpThreeD[sTraj].begin(), InfoTraj.cpThreeD[sTraj].end());
		std::reverse(InfoTraj.cpVis[sTraj].begin(), InfoTraj.cpVis[sTraj].end());

		for (int jj = 0; jj < InfoTraj.nViews; jj++)
			Traj2D[jj].clear(), TimeLine[jj].clear();

		double normNormal, normPtoC, angle;
		Point3d t3D, n3D, PtoC;
		Point2d pt;
		sprintf(Fname, "%s/Traject_%d.txt", Path, kk); FILE *fp = fopen(Fname, "w+");
		int ntracks = InfoTraj.cpVis[sTraj].size();
		for (int jj = 0; jj < ntracks; jj++)
		{
			t3D = InfoTraj.cpThreeD[sTraj].at(jj);
			n3D = InfoTraj.cpNormal[sTraj].at(jj);
			normNormal = sqrt(pow(n3D.x, 2) + pow(n3D.y, 2) + pow(n3D.z, 2));
			int nvis = InfoTraj.cpVis[sTraj].at(jj).size();
			fprintf(fp, "%d %d ", CurrentFrame - ntracks + jj, nvis);
			for (int ii = 0; ii < nvis; ii++)
			{
				int viewID = InfoTraj.cpVis[sTraj].at(jj).at(ii);
				PtoC = Point3d(AllCamInfo[viewID].camCenter[0] - t3D.x, AllCamInfo[viewID].camCenter[1] - t3D.y, AllCamInfo[viewID].camCenter[2] - t3D.z);
				normPtoC = sqrt(pow(PtoC.x, 2) + pow(PtoC.y, 2) + pow(PtoC.z, 2));
				angle = (n3D.x*PtoC.x + n3D.y*PtoC.y + n3D.z*PtoC.z) / normNormal / normPtoC;
				//if (angle > angleThreshold)
				//continue;

				ProjectandDistort(t3D, &pt, AllCamInfo[viewID].P);
				fprintf(fp, "%d %.2f %.2f %.2f ", viewID, pt.x, pt.y, angle);
			}
			fprintf(fp, "\n");
		}

		for (int jj = 0; jj < InfoTraj.fThreeD[sTraj].size(); jj++)
		{
			t3D = InfoTraj.fThreeD[sTraj].at(jj);
			n3D = InfoTraj.fNormal[sTraj].at(jj);
			normNormal = sqrt(pow(n3D.x, 2) + pow(n3D.y, 2) + pow(n3D.z, 2));
			int nvis = InfoTraj.fVis[sTraj].at(jj).size();
			fprintf(fp, "%d %d ", CurrentFrame + jj + 1, nvis);
			for (int ii = 0; ii < nvis; ii++)
			{
				int viewID = InfoTraj.fVis[sTraj].at(jj).at(ii);
				PtoC = Point3d(AllCamInfo[viewID].camCenter[0] - t3D.x, AllCamInfo[viewID].camCenter[1] - t3D.y, AllCamInfo[viewID].camCenter[2] - t3D.z);
				normPtoC = sqrt(pow(PtoC.x, 2) + pow(PtoC.y, 2) + pow(PtoC.z, 2));
				angle = (n3D.x*PtoC.x + n3D.y*PtoC.y + n3D.z*PtoC.z) / normNormal / normPtoC;
				//if (angle > angleThreshold)
				//	continue;
				ProjectandDistort(t3D, &pt, AllCamInfo[viewID].P);
				fprintf(fp, "%d %.2f %.2f %.2f ", viewID, pt.x, pt.y, angle);
			}
			fprintf(fp, "\n");
		}
		fclose(fp);
	}
	return;
}
int Load2DTrajectory(char *Path, TrajectoryData &inoTraj, int ntrajectories)
{
	inoTraj.trajectoryUnit = new vector<Trajectory2D>[ntrajectories];
	int timeID, nvis, camID;
	float u, v, angle;
	char Fname[200];
	for (int kk = 0; kk < ntrajectories; kk++)
	{
		if (kk % 100 == 0)
			printf("Loading traj file # %d (%.2f%%) \r", kk, 100.0*kk / ntrajectories);
		sprintf(Fname, "%s/Traject_%d.txt", Path, kk);
		FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("Cannot load %s", Fname);
			continue;// return 1;
		}
		while (fscanf(fp, "%d %d", &timeID, &nvis) != EOF)
		{
			Trajectory2D OneTrajectory;
			OneTrajectory.timeID = timeID, OneTrajectory.nViews = nvis;
			OneTrajectory.uv.reserve(nvis), OneTrajectory.angle.reserve(nvis), OneTrajectory.viewIDs.reserve(nvis);
			for (int jj = 0; jj < nvis; jj++)
			{
				fscanf(fp, "%d %f %f %f", &camID, &u, &v, &angle);
				OneTrajectory.viewIDs.push_back(camID);
				OneTrajectory.uv.push_back(Point2d(u, v));
				OneTrajectory.angle.push_back(angle);
			}
			inoTraj.trajectoryUnit[kk].push_back(OneTrajectory);
		}
		fclose(fp);
	}

	return 0;
}
int GetImagePatchIntensityColorVar(char *Path, TrajectoryData infoTraj, int nTraj, int minFrame, int maxFrame, int *cameraPair, int *range)// , vector<Point3i>& Argb1, vector<Point3i>& Argb2)
{
	printf("Getting  Trajectory Color profile\n");
	char Fname[200];

	static CvScalar colors[] =
	{
		{ { 0, 0, 255 } },
		{ { 0, 128, 255 } },
		{ { 0, 255, 255 } },
		{ { 0, 255, 0 } },
		{ { 255, 128, 0 } },
		{ { 255, 255, 0 } },
		{ { 255, 0, 0 } },
		{ { 255, 0, 255 } },
		{ { 255, 255, 255 } }
	};

	const int nCamsPerPanel = 24, width = 640, height = 480;
	int timeID, nvis, camID, panelID, camIDInPanel;
	Point3d t3D;
	IplImage *Img = 0;
	float u, v;
	vector<Point3i> rgb1, rgb2;

	printf("Loading images to memory\n");
	vector<IplImage*> AllImagePtr;
	for (int ii = 0; ii < 2; ii++)
	{
		int viewID = cameraPair[ii];
		for (int timeID = 0; timeID <= maxFrame; timeID++)
		{
			panelID = viewID / nCamsPerPanel,
				camIDInPanel = viewID%nCamsPerPanel;
			sprintf(Fname, "%s/In/%.8d/%.8d_%.02d_%.02d.png", Path, timeID, timeID, panelID + 1, camIDInPanel + 1);
			Img = cvLoadImage(Fname, 1);
			if (Img == NULL)
				;// printf("Cannot load %s\n", Fname);
			else
				printf("View %d: %.2f %% completed \r", viewID, 100.0*timeID / 209);
			AllImagePtr.push_back(Img);
		}
		printf("View %d:  completed \n", viewID);
	}
	int hsubset = 2, patchlength = (2 * hsubset + 1)*(2 * hsubset + 1);
	double *T = new double[2 * patchlength * 3];
	double *RGB = new double[2 * 3 * (2 * hsubset + 1)*(2 * hsubset + 1)];

	IplImage *drawing = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 3);
	sprintf(Fname, "%s/dif_%d_%d.txt", Path, cameraPair[0], cameraPair[1]); FILE *fp3 = fopen(Fname, "w+");
	if (fp3 == NULL)
	{
		printf("Cannot open %s\n", Fname);
		return 1;
	}
	fprintf(fp3, "%d %f\n", 1, 999.0);
	fclose(fp3);

	for (int temporalOffset = range[0]; temporalOffset <= range[1]; temporalOffset++)
	{
		double zncc = 0.0;
		double start = omp_get_wtime();
		//sprintf(Fname, "%s/TrajectC%d_%d_%d.txt", Path, temporalOffset, cameraPair[0], cameraPair[1]); FILE *fp2 = fopen(Fname, "w+");
		for (int ii = 0; ii < nTraj; ii++)
		{
			if (ii % 100 == 0)
				printf("Time offset %d: @%.2f%% \r", temporalOffset, 100.0*ii / nTraj);
			rgb1.clear(), rgb2.clear();
			for (int kk = 0; kk < infoTraj.trajectoryUnit[ii].size(); kk++)
			{
				int timeID = infoTraj.trajectoryUnit[ii].at(kk).timeID, nvis = infoTraj.trajectoryUnit[ii].at(kk).nViews;

				int count = 0, iu, iv;
				for (int jj = 0; jj < nvis; jj++)
				{
					camID = infoTraj.trajectoryUnit[ii].at(kk).viewIDs.at(jj);
					u = infoTraj.trajectoryUnit[ii].at(kk).uv.at(jj).x, v = infoTraj.trajectoryUnit[ii].at(kk).uv.at(jj).y;
					panelID = camID / nCamsPerPanel,
						camIDInPanel = camID%nCamsPerPanel;

					if (timeID >maxFrame || timeID + temporalOffset > maxFrame || timeID + temporalOffset < minFrame)
						continue;

					if (camID == cameraPair[0] || camID == cameraPair[1])
					{
						int ind = camID == cameraPair[0] ? 0 : 1;
						int id = ind* maxFrame + timeID + ind*temporalOffset;
						Img = AllImagePtr[ind* (maxFrame + 1) + timeID + ind*temporalOffset];
						//sprintf(Fname, "%s/In/%.8d/%.8d_%.02d_%.02d.png", Path, timeID + ind*temporalOffset, timeID + ind* temporalOffset, panelID + 1, camIDInPanel + 1);
						if (Img == NULL)
							continue;

						iu = (int)u, iv = (int)v;
						if (iu<5 || iv<5 || iu>width - 5 || iv>height - 5)
							continue;
						int pcount = 0;
						for (int mm = -hsubset; mm <= hsubset; mm++)
						{
							for (int nn = -hsubset; nn <= hsubset; nn++)
							{
								RGB[ind*patchlength * 3 + 3 * pcount] = Img->imageData[3 * ((iv + mm)*width + iu + nn) + 0];
								RGB[ind*patchlength * 3 + 3 * pcount + 1] = Img->imageData[3 * ((iv + mm)*width + iu + nn) + 1];
								RGB[ind*patchlength * 3 + 3 * pcount + 2] = Img->imageData[3 * ((iv + mm)*width + iu + nn) + 2];
								pcount += 3;
							}
						}
						count++;

						/*cvCopy(Img, drawing);
						cvCircle(drawing, Point2i(u, v), 2, colors[rand() % 9], 1, 8, 0);
						CvFont font = cvFont(2.0 * 640 / 2048, 2);
						cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 2.0 * 640 / 2048, 2.0 * 640 / 2048, 0, 2, 8);
						CvPoint text_origin = { 640 / 30, 640 / 30 };
						sprintf(Fname, "%.2d_%.02d_%.02d %d/%d NVis of Traj %d", timeID + ind*temporalOffset, panelID + 1, camIDInPanel + 1, jj + 1, nvis, ii + 1);
						cvPutText(drawing, Fname, text_origin, &font, CV_RGB(255, 0, 0));
						char Fname2[200]; sprintf(Fname2, "Image %d", ind);
						cvShowImage(Fname2, drawing); waitKey(-1);
						int a = 0;*/
					}

					if (count == 2)
						zncc += ComputeZNCCPatch(RGB, RGB + 3 * patchlength, hsubset, 3, T);
					if (count == 2) //Keep on reading until the end of that point
						break;
				}
			}

			/*if (rgb1.size() > 1 && rgb1.size() == rgb2.size())
			{
			fprintf(fp2, "%d %d \n", ii, rgb1.size());
			for (int jj = 0; jj < rgb1.size(); jj++)
			fprintf(fp2, "%d %d %d ", rgb1[jj].x, rgb1[jj].y, rgb1[jj].z);
			fprintf(fp2, "\n");
			for (int jj = 0; jj < rgb2.size(); jj++)
			fprintf(fp2, "%d %d %d ", rgb2[jj].x, rgb2[jj].y, rgb2[jj].z);
			fprintf(fp2, "\n");
			}*/
		}
		//fclose(fp2);
		sprintf(Fname, "%s/dif_%d_%d.txt", Path, cameraPair[0], cameraPair[1]);
		FILE *fp3 = fopen(Fname, "a");
		if (fp3 == NULL)
		{
			printf("Cannot open %s\n", Fname);
			return 1;
		}
		fprintf(fp3, "%d %f\n", temporalOffset, zncc);
		fclose(fp3);
		printf("Time offset %d: @%.2f%% in %.2fs\n", temporalOffset, 100.0, omp_get_wtime() - start);
	}


	for (int ii = 0; ii < AllImagePtr.size(); ii++)
		cvReleaseImage(&AllImagePtr[ii]);

	return 0;
}
int Compute3DTrajectoryErrorColorVar(char *Path, vector<int> SyncOff, int *pair)
{
	char Fname[200];
	int TrajID, nframes, r, g, b;
	vector<int> rgb1, rgb2;
	vector<double> dr, db, dg;

	double *dif = new double[SyncOff.size()];
	for (int ii = 0; ii < SyncOff.size(); ii++)
	{
		dif[ii] = 0.0;
		sprintf(Fname, "%s/TrajectC%d_%d_%d.txt", Path, SyncOff[ii], pair[0], pair[1]); FILE *fp = fopen(Fname, "r");
		while (fscanf(fp, "%d %d", &TrajID, &nframes) != EOF)
		{
			rgb1.clear(), rgb2.clear();
			dr.clear(), db.clear(), dg.clear();
			for (int jj = 0; jj < nframes; jj++)
			{
				fscanf(fp, "%d %d %d ", &r, &g, &b);
				b = b / 9;
				rgb1.push_back(r), rgb1.push_back(g), rgb1.push_back(b);
			}
			for (int jj = 0; jj < nframes; jj++)
			{
				fscanf(fp, "%d %d %d ", &r, &g, &b);
				b = b / 9;
				rgb2.push_back(r), rgb2.push_back(g), rgb2.push_back(b);
			}

			for (int jj = 0; jj < nframes; jj++)
			{
				dr.push_back(rgb1[3 * jj] - rgb2[3 * jj]);
				dg.push_back(rgb1[3 * jj + 1] - rgb2[3 * jj] + 1);
				db.push_back(rgb1[3 * jj + 2] - rgb2[3 * jj + 2]);
			}
			dif[ii] += sqrt(VarianceArray(dr) + VarianceArray(dg) + VarianceArray(db));
			//dif[ii] += L1norm(dr) + L1norm(dg) + L1norm(db);
		}
		fclose(fp);
	}

	sprintf(Fname, "%s/dif_%d_%d.txt", Path, pair[0], pair[1]);
	FILE *fp = fopen(Fname, "w+");
	for (int ii = 0; ii < SyncOff.size(); ii++)
		fprintf(fp, "%d %lf\n", SyncOff[ii], dif[ii]);
	fclose(fp);

	return 0;
}
int Compute3DTrajectoryErrorZNCC(char *Path, TrajectoryData infoTraj, int nTraj, int minFrame, int maxFrame, int *cameraPair, int *range)// , vector<Point3i>& Argb1, vector<Point3i>& Argb2)
{
	printf("Getting  Trajectory Color profile\n");
	char Fname[200];

	int syncOff[480];
	FILE *fp = fopen("D:/Y/Out/syncOff.txt", "r");
	for (int ii = 0; ii < 480; ii++)
		fscanf(fp, "%d ", &syncOff[ii]);
	fclose(fp);

	/*int TrueCamID[480];
	fp = fopen("D:/Y/camId.txt", "r");
	for (int ii = 0; ii < 24; ii++)
	fscanf(fp, "%d ", &TrueCamID[ii]);
	fclose(fp);*/

	static CvScalar colors[] =
	{
		{ { 0, 0, 255 } },
		{ { 0, 128, 255 } },
		{ { 0, 255, 255 } },
		{ { 0, 255, 0 } },
		{ { 255, 128, 0 } },
		{ { 255, 255, 0 } },
		{ { 255, 0, 0 } },
		{ { 255, 0, 255 } },
		{ { 255, 255, 255 } }
	};

	const int nCamsPerPanel = 24, width = 640, height = 480, length = width*height;
	int timeID, nvis, camID, panelID, camIDInPanel;
	Point3d t3D;
	IplImage *Img = 0;
	float u, v, angle;
	vector<Point3i> rgb1, rgb2;

	printf("Loading images to memory\n");
	int nchannels = 1;
	vector<IplImage*> AllImagePtr;
	vector<float*> AllImageParaPtr;
	float *tImg = new float[width*height*nchannels];
	for (int ii = 0; ii < 2; ii++)
	{
		int viewID = cameraPair[ii];
		for (int timeID = 0; timeID <= maxFrame; timeID++)
		{
			panelID = viewID / nCamsPerPanel,
				camIDInPanel = viewID%nCamsPerPanel;
			sprintf(Fname, "%s/In/%.8d/%.8d_%.02d_%.02d.png", Path, timeID, timeID, panelID + 1, camIDInPanel + 1);
			Img = cvLoadImage(Fname, nchannels == 1 ? 0 : 1);
			float *Para = 0;
			if (Img == NULL)
				;// printf("Cannot load %s\n", Fname);
			else
			{
				Para = new float[nchannels * length];
				for (int kk = 0; kk < nchannels; kk++)
				{
					for (int jj = 0; jj < height; jj++)
						for (int ii = 0; ii < width; ii++)
							tImg[ii + jj*width + kk*length] = Img->imageData[nchannels*ii + nchannels*jj*width + kk];
					Generate_Para_Spline(tImg + kk*length, Para + kk*length, width, height, 1);
				}
				printf("View %d: %.2f %% completed \r", viewID, 100.0*timeID / 209);
			}

			AllImagePtr.push_back(Img);
			AllImageParaPtr.push_back(Para);
		}
		printf("View %d:  completed \n", viewID);
	}
	IplImage *drawing = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 3);

	int hsubset = 5, patchSize = (2 * hsubset + 1), patchlength = (2 * hsubset + 1)*(2 * hsubset + 1);
	double *T = new double[2 * patchlength * 3];
	double *RGB = new double[2 * 3 * (2 * hsubset + 1)*(2 * hsubset + 1)];

	bool Save = false;
	vector<int> temporal;
	vector<double>ZNCCv;
	for (int temporalOffset = range[0]; temporalOffset <= range[1]; temporalOffset++)
	{
		double zncc = 0.0;

		float *Para = 0;
		double start = omp_get_wtime();
		for (int ii = 0; ii < nTraj; ii++)
		{
			if (ii % 100 == 0)
				printf("Time offset %d: @%.2f%% \r", temporalOffset, 100.0*ii / nTraj);
			rgb1.clear(), rgb2.clear();
			for (int kk = 0; kk < infoTraj.trajectoryUnit[ii].size(); kk++)
			{
				int timeID = infoTraj.trajectoryUnit[ii].at(kk).timeID, nvis = infoTraj.trajectoryUnit[ii].at(kk).nViews;

				int count = 0, iu, iv;
				for (int jj = 0; jj < nvis; jj++)
				{
					camID = infoTraj.trajectoryUnit[ii].at(kk).viewIDs.at(jj);
					panelID = camID / nCamsPerPanel, camIDInPanel = camID%nCamsPerPanel;

					u = infoTraj.trajectoryUnit[ii].at(kk).uv.at(jj).x, v = infoTraj.trajectoryUnit[ii].at(kk).uv.at(jj).y, angle = infoTraj.trajectoryUnit[ii].at(kk).angle.at(jj);
					if (u<hsubset || v<hsubset || u>width - hsubset || v>height - hsubset)
						break;
					if (angle < 0.5)
						break;
					if (timeID + syncOff[camID] > maxFrame || timeID + temporalOffset + syncOff[camID] > maxFrame || timeID + temporalOffset + syncOff[camID] < minFrame)
						continue;

					if (camID == cameraPair[0] || camID == cameraPair[1])
					{
						int ind = camID == cameraPair[0] ? 0 : 1;
						int id = ind* maxFrame + timeID + ind*temporalOffset;
						Img = AllImagePtr[ind* (maxFrame + 1) + timeID + ind*temporalOffset + syncOff[camID]];
						if (Img == NULL)
							continue;

						u = infoTraj.trajectoryUnit[ii].at(kk).uv.at(jj).x, v = infoTraj.trajectoryUnit[ii].at(kk).uv.at(jj).y;
						if (u<hsubset || v<hsubset || u>width - hsubset || v>height - hsubset)
							continue;
						int pcount = 0; double S[3];
						Para = AllImageParaPtr[ind* (maxFrame + 1) + timeID + ind*temporalOffset + syncOff[camID]];

						if (Save)
						{
							for (int mm = -0; mm < height; mm++)
							{
								for (int nn = -0; nn < width; nn++)
								{
									Get_Value_Spline(Para, width, height, nn, mm, S, -1, 1);
									tImg[nn + mm*width] = S[0];
								}
							}
							cvSaveImage("C:/temp/img1.png", Img);
							SaveDataToImage("C:/temp/img2.png", tImg, width, height);
						}

						for (int ll = 0; ll < nchannels; ll++)
							for (int mm = -hsubset; mm <= hsubset; mm++)
								for (int nn = -hsubset; nn <= hsubset; nn++)
									Get_Value_Spline(Para + ll*length, width, height, u + nn, v + mm, S, -1, 1),
									RGB[(3 * ind + ll)*patchlength + (mm + hsubset)*patchSize + nn + hsubset] = S[0];

						/*cvCopy(Img, drawing);
						cvCircle(drawing, Point2i(u, v), 2, colors[rand() % 9], 1, 8, 0);
						CvFont font = cvFont(2.0 * 640 / 2048, 2);
						cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 2.0 * 640 / 2048, 2.0 * 640 / 2048, 0, 2, 8);
						CvPoint text_origin = { 640 / 30, 640 / 30 };
						sprintf(Fname, "%.2d_%.02d_%.02d %d/%d NVis of Traj %d", timeID + ind*temporalOffset, panelID + 1, camIDInPanel + 1, jj + 1, nvis, ii + 1);
						cvPutText(drawing, Fname, text_origin, &font, CV_RGB(255, 0, 0));
						char Fname2[200]; sprintf(Fname2, "Image %d", ind);
						cvShowImage(Fname2, drawing); waitKey(-1);*/
						count++;
					}

					if (count == 2)
						zncc += ComputeZNCCPatch(RGB, RGB + 3 * patchlength, hsubset, nchannels, T);
					if (count == 2) //Keep on reading until the end of that point
						break;
				}
			}
		}
		temporal.push_back(temporalOffset);
		ZNCCv.push_back(zncc);

		printf("Time offset %d: @%.2f%% Zncc: %f TE: %.2fs\n", temporalOffset, 100.0, zncc, omp_get_wtime() - start);
	}

	sprintf(Fname, "%s/dif_%d_%d.txt", Path, cameraPair[0], cameraPair[1]);
	FILE *fp3 = fopen(Fname, "w+");
	if (fp3 == NULL)
	{
		printf("Cannot open %s\n", Fname);
		return 1;
	}
	for (int ii = 0; ii < ZNCCv.size(); ii++)
		fprintf(fp3, "%d %f\n", temporal[ii], ZNCCv[ii]);
	fclose(fp3);

	delete[]T, delete[]RGB, delete[]tImg;
	cvReleaseImage(&drawing);
	for (int ii = 0; ii < AllImagePtr.size(); ii++)
	{
		cvReleaseImage(&AllImagePtr[ii]);
		delete[]AllImageParaPtr[ii];
	}

	return 0;
}
int Compute3DTrajectoryError2DTracking(char *Path, TrajectoryData infoTraj, int nTraj, int minFrame, int maxFrame, int SelectedViewID, int *range)
{
	printf("Getting  Trajectory Color profile\n");
	char Fname[200];

	int syncOff[480];
	FILE *fp = fopen("D:/Y/Out/syncOff.txt", "r");
	for (int ii = 0; ii < 480; ii++)
		//fscanf(fp, "%d ", &syncOff[ii]);
		syncOff[ii] = 0;
	fclose(fp);

	const int nCamsPerPanel = 24, width = 640, height = 480, length = width*height;
	int timeID, nvis, viewID, panelID, camIDInPanel;
	float u, v, angle;
	Point3d t3D;
	IplImage *Img = 0, *drawing = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 3);

	printf("Loading images to memory\n");
	int nchannels = 1;
	vector<IplImage*> AllImagePtr;
	vector<float*> AllImageParaPtr;
	float *tImg = new float[width*height*nchannels];
	for (int timeID = 0; timeID <= maxFrame; timeID++)
	{
		panelID = SelectedViewID / nCamsPerPanel, camIDInPanel = SelectedViewID%nCamsPerPanel;
		sprintf(Fname, "%s/In/%.8d/%.8d_%.02d_%.02d.png", Path, timeID, timeID, panelID + 1, camIDInPanel + 1);
		Img = cvLoadImage(Fname, nchannels == 1 ? 0 : 1);
		float *Para = 0;
		if (Img == NULL)
			;// printf("Cannot load %s\n", Fname);
		else
		{
			Para = new float[nchannels * length];
			for (int kk = 0; kk < nchannels; kk++)
			{
				for (int jj = 0; jj < height; jj++)
					for (int ii = 0; ii < width; ii++)
						tImg[ii + jj*width + kk*length] = Img->imageData[nchannels*ii + nchannels*jj*width + kk];
				Generate_Para_Spline(tImg + kk*length, Para + kk*length, width, height, 1);
			}
			printf("View %d: %.2f %% completed \r", SelectedViewID, 100.0*timeID / 209);
		}

		AllImagePtr.push_back(Img);
		AllImageParaPtr.push_back(Para);
	}
	printf("View %d:  completed \n", SelectedViewID);

	int hsubset = 7, patchSize = (2 * hsubset + 1), patchlength = (2 * hsubset + 1)*(2 * hsubset + 1);
	double *T = new double[2 * patchlength * nchannels];
	double *RGB = new double[2 * nchannels * patchlength];
	double *Timg = new double[patchlength*nchannels];
	double *Znssd_reqd = new double[6 * patchlength];

	vector<int> temporal;
	vector<double>Error2D;
	vector<IplImage*> VisImagePtr;
	vector<float*> VisImageParaPtr;
	vector<Point2d> Projected2DLoc, Tracked2DLoc;

	for (int temporalOffset = range[0]; temporalOffset <= range[1]; temporalOffset++)
	{
		double error = 0.0;
		VisImagePtr.clear(), VisImageParaPtr.clear(), Projected2DLoc.clear(), Tracked2DLoc.clear();

		double start = omp_get_wtime();
		for (int ii = 0; ii < nTraj; ii++)
		{
			if (ii % 100 == 0)
				printf("Time offset %d: @%.2f%% \r", temporalOffset, 100.0*ii / nTraj);

			bool found = false;////See if the required camera is visible
			int kk;
			for (kk = 0; kk < infoTraj.trajectoryUnit[ii].size(); kk++)
			{
				int timeID = infoTraj.trajectoryUnit[ii].at(kk).timeID;
				for (int jj = 0; jj < infoTraj.trajectoryUnit[ii].at(kk).nViews; jj++)
				{
					viewID = infoTraj.trajectoryUnit[ii].at(kk).viewIDs.at(jj);
					if (timeID + syncOff[viewID] > maxFrame || timeID + temporalOffset + syncOff[viewID] > maxFrame || timeID + temporalOffset + syncOff[viewID] < minFrame)
						break;
					if (viewID == SelectedViewID)
					{
						u = infoTraj.trajectoryUnit[ii].at(kk).uv.at(jj).x, v = infoTraj.trajectoryUnit[ii].at(kk).uv.at(jj).y, angle = infoTraj.trajectoryUnit[ii].at(kk).angle.at(jj);
						if (u<hsubset || v<hsubset || u>width - hsubset || v>height - hsubset)
							break;
						if (angle < 0.5)
							break;
						Projected2DLoc.push_back(infoTraj.trajectoryUnit[ii].at(kk).uv.at(jj));

						VisImagePtr.push_back(AllImagePtr[timeID + temporalOffset + syncOff[viewID]]);
						VisImageParaPtr.push_back(AllImageParaPtr[timeID + temporalOffset + syncOff[viewID]]);
						if (VisImagePtr.back() == NULL || VisImageParaPtr.back() == NULL)
							break;

						found = true;
						break;
					}
				}
				if (found)
					break;
			}
			if (!found)
				continue;
			double fufv[2];
			Point2d PR = Projected2DLoc[0], PT = PR;
			Tracked2DLoc.push_back(PR);

			//See how many frames it is visible
			found = false;
			int visibleFrameCount = 1;
			for (kk; kk < infoTraj.trajectoryUnit[ii].size(); kk++)
			{
				//See if the required camera is visible
				int timeID = infoTraj.trajectoryUnit[ii].at(kk).timeID;
				for (int jj = 0; jj < infoTraj.trajectoryUnit[ii].at(kk).nViews; jj++)
				{
					viewID = infoTraj.trajectoryUnit[ii].at(kk).viewIDs.at(jj);
					if (timeID + syncOff[viewID] > maxFrame || timeID + temporalOffset + syncOff[viewID] > maxFrame || timeID + temporalOffset + syncOff[viewID] < minFrame)
						break;
					if (viewID == SelectedViewID)
					{
						u = infoTraj.trajectoryUnit[ii].at(kk).uv.at(jj).x, v = infoTraj.trajectoryUnit[ii].at(kk).uv.at(jj).y, angle = infoTraj.trajectoryUnit[ii].at(kk).angle.at(jj);
						if (u<hsubset || v<hsubset || u>width - hsubset || v>height - hsubset)
							break;
						if (angle < 0.5)
							break;
						Projected2DLoc.push_back(infoTraj.trajectoryUnit[ii].at(kk).uv.at(jj));

						VisImagePtr.push_back(AllImagePtr[timeID + temporalOffset + syncOff[viewID]]);
						VisImageParaPtr.push_back(AllImageParaPtr[timeID + temporalOffset + syncOff[viewID]]);
						if (VisImagePtr.back() == NULL || VisImageParaPtr.back() == NULL)
							break;

						found = true;
						visibleFrameCount++;
						break;
					}
				}

				if (!found) //stop the search if the point disapperas
					break;
				found = false;
			}

			if (visibleFrameCount < 5)
				continue;

			//Let track the 2D points and compute their differnces
			for (int kk = 0; kk < visibleFrameCount - 1; kk++)
			{
				if (TrackingByLK(VisImageParaPtr[kk], VisImageParaPtr[kk + 1], hsubset, width, height, width, height, nchannels, PR, PT, 3, 1, 0.85, 30, 1, fufv, false, NULL, NULL, Timg, T, Znssd_reqd) < 0.9)
					break;//tracking fails
				PR.x = PR.x + fufv[0], PR.y = PR.y + fufv[1];
				Tracked2DLoc.push_back(PR);
				error += sqrt(pow(Projected2DLoc[kk + 1].x - PR.x, 2) + pow(Projected2DLoc[kk + 1].y - PR.y, 2));
			}
		}
		Error2D.push_back(error);
		temporal.push_back(temporalOffset);

		printf("Time offset %d: @%.2f%% in %.2fs\n", temporalOffset, 100.0, omp_get_wtime() - start);
	}

	sprintf(Fname, "%s/dif_%d.txt", Path, SelectedViewID);
	FILE *fp3 = fopen(Fname, "w+");
	if (fp3 == NULL)
	{
		printf("Cannot open %s\n", Fname);
		return 1;
	}
	for (int ii = 0; ii < temporal.size(); ii++)
		fprintf(fp, "%d %f\n", temporal[ii], Error2D[ii]);
	fclose(fp3);

	delete[]T, delete[]RGB, delete[]Timg; delete[]Znssd_reqd;  delete[]tImg;
	cvReleaseImage(&drawing);
	for (int ii = 0; ii < AllImagePtr.size(); ii++)
	{
		cvReleaseImage(&AllImagePtr[ii]);
		delete[]AllImageParaPtr[ii];
	}

	return 0;
}

bool myImgReader(char *fname, unsigned char *Img, int width, int height, int nchannels)
{
	Mat view = imread(fname, nchannels == 1 ? 0 : 1);
	if (view.data == NULL)
	{
		cout << "Cannot load: " << fname << endl;
		return false;
	}
	int length = width*height;
	for (int kk = 0; kk < nchannels; kk++)
	{
		for (int jj = 0; jj < height; jj++)
			for (int ii = 0; ii < width; ii++)
				Img[ii + jj*width + kk*length] = view.data[nchannels*ii + jj*nchannels*width + kk];
	}

	return true;
}
bool myImgReader(char *fname, float *Img, int width, int height, int nchannels)
{
	Mat view = imread(fname, nchannels == 1 ? 0 : 1);
	if (view.data == NULL)
	{
		cout << "Cannot load: " << fname << endl;
		return false;
	}
	int length = width*height;
	for (int kk = 0; kk < nchannels; kk++)
	{
		for (int jj = 0; jj < height; jj++)
			for (int ii = 0; ii < width; ii++)
				Img[ii + jj*width + kk*length] = (float)(int)view.data[nchannels*ii + jj*nchannels*width + kk];
	}

	return true;
}
bool myImgReader(char *fname, double *Img, int width, int height, int nchannels)
{
	Mat view = imread(fname, nchannels == 1 ? 0 : 1);
	if (view.data == NULL)
	{
		cout << "Cannot load: " << fname << endl;
		return false;
	}
	int length = width*height;
	for (int kk = 0; kk < nchannels; kk++)
	{
		for (int jj = 0; jj < height; jj++)
			for (int ii = 0; ii < width; ii++)
				Img[ii + jj*width + kk*length] = (double)(int)view.data[nchannels*ii + jj*nchannels*width + kk];
	}

	return true;
}
void ShowDataToImage(char *Fname, char *Img, int width, int height, int nchannels, IplImage *cvImg)
{
	//Need to call waitkey
	int ii, jj, kk, length = width*height;

	bool createMem = false;
	if (cvImg == 0)
	{
		createMem = true;
		cvImg = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, nchannels);
	}
	for (jj = 0; jj < height; jj++)
		for (ii = 0; ii < width; ii++)
			for (kk = 0; kk < nchannels; kk++)
				cvImg->imageData[nchannels*ii + kk + nchannels*jj*width] = Img[nchannels*ii + kk + nchannels*jj*width];//Img[ii+jj*width+kk*length];

	//cvSaveImage("C:/temp/x.png", cvImg);
	cvShowImage(Fname, cvImg);

	return;
}
bool SaveDataToImage(char *fname, char *Img, int width, int height, int nchannels)
{
	int ii, jj, kk, length = width*height;

	Mat M = Mat::zeros(height, width, nchannels == 1 ? CV_8UC1 : CV_8UC3);
	for (jj = 0; jj < height; jj++)
		for (ii = 0; ii < width; ii++)
			for (kk = 0; kk < nchannels; kk++)
				M.data[nchannels*ii + kk + nchannels*jj*width] = Img[nchannels*ii + kk + nchannels*jj*width];//ii+(height-1-jj)*width+kk*length];

	return imwrite(fname, M);
}
bool SaveDataToImage(char *fname, unsigned char *Img, int width, int height, int nchannels)
{
	int ii, jj, kk, length = width*height;

	Mat M = Mat::zeros(height, width, nchannels == 1 ? CV_8UC1 : CV_8UC3);
	for (jj = 0; jj < height; jj++)
		for (ii = 0; ii < width; ii++)
			for (kk = 0; kk < nchannels; kk++)
				M.data[nchannels*ii + kk + nchannels*jj*width] = Img[ii + (height - 1 - jj)*width + kk*length];

	return imwrite(fname, M);
}
bool SaveDataToImage(char *fname, float *Img, int width, int height, int nchannels)
{
	int ii, jj, kk, length = width*height;

	Mat M = Mat::zeros(height, width, nchannels == 1 ? CV_8UC1 : CV_8UC3);
	for (jj = 0; jj < height; jj++)
		for (ii = 0; ii < width; ii++)
			for (kk = 0; kk < nchannels; kk++)
				M.data[nchannels*ii + kk + nchannels*jj*width] = (unsigned char)(int)(Img[ii + jj*width + kk*length] + 0.5);

	return imwrite(fname, M);
}
bool SaveDataToImage(char *fname, double *Img, int width, int height, int nchannels)
{
	int ii, jj, kk, length = width*height;

	Mat M = Mat::zeros(height, width, nchannels == 1 ? CV_8UC1 : CV_8UC3);
	for (jj = 0; jj < height; jj++)
		for (ii = 0; ii < width; ii++)
			for (kk = 0; kk < nchannels; kk++)
				M.data[nchannels*ii + kk + nchannels*jj*width] = (unsigned char)(int)(Img[ii + jj*width + kk*length] + 0.5);

	return imwrite(fname, M);
}

bool WriteKPointsBinary(char *fn, vector<KeyPoint>kpts, bool silent)
{
	ofstream fout;
	fout.open(fn, ios::binary);
	if (!fout.is_open())
	{
		if (silent)
			cout << "Cannot write: " << fn << endl;
		return false;
	}

	int npts = kpts.size();
	KeyPoint kpt;
	fout.write(reinterpret_cast<char *>(&npts), sizeof(int));
	for (int j = 0; j < npts; ++j)
	{
		fout.write(reinterpret_cast<char *>(&kpts.at(j).pt.x), sizeof(float));
		fout.write(reinterpret_cast<char *>(&kpts.at(j).pt.y), sizeof(float));
		fout.write(reinterpret_cast<char *>(&kpts.at(j).size), sizeof(float));
	}
	fout.close();

	return true;
}
bool ReadKPointsBinary(char *fn, vector<KeyPoint> &kpts, bool silent)
{
	ifstream fin;
	fin.open(fn, ios::binary);
	if (!fin.is_open())
	{
		cout << "Cannot open: " << fn << endl;
		return false;
	}
	if (silent)
		cout << "Load " << fn << endl;

	int npts;
	fin.read(reinterpret_cast<char *>(&npts), sizeof(int));
	kpts.reserve(npts);
	float x, y, scale;
	KeyPoint kpt;
	for (int ii = 0; ii < npts; ii++)
	{
		fin.read(reinterpret_cast<char *>(&x), sizeof(float));
		fin.read(reinterpret_cast<char *>(&y), sizeof(float));
		fin.read(reinterpret_cast<char *>(&scale), sizeof(float));
		kpt.pt.x = x, kpt.pt.y = y, kpt.size = scale;
		kpts.push_back(kpt);
	}

	return true;
}
bool WriteDescriptorBinary(char *fn, Mat descriptor, bool silent)
{
	ofstream fout;
	fout.open(fn, ios::binary);
	if (!fout.is_open())
	{
		if (silent)
			cout << "Cannot write: " << fn << endl;
		return false;
	}

	int npts = descriptor.rows, descriptorSize = descriptor.cols;
	fout.write(reinterpret_cast<char *>(&npts), sizeof(int));
	for (int j = 0; j < npts; ++j)
		for (int i = 0; i < descriptorSize; i++)
		{
		float x = descriptor.at<float>(j, i);
		fout.write(reinterpret_cast<char *>(&x), sizeof(float));
		}
	fout.close();

	return true;
}
Mat ReadDescriptorBinary(char *fn, int descriptorSize, bool silent)
{
	ifstream fin;
	fin.open(fn, ios::binary);
	if (!fin.is_open())
	{
		cout << "Cannot open: " << fn << endl;
		Mat descriptor(1, descriptorSize, CV_32F);
		return descriptor;
	}
	else
	{
		if (silent)
			cout << "Load " << fn << endl;

		int npts;
		fin.read(reinterpret_cast<char *>(&npts), sizeof(int));
		Mat descriptor(npts, descriptorSize, CV_32F);
		for (int j = 0; j < npts; j++)
			for (int i = 0; i < descriptorSize; i++)
				fin.read(reinterpret_cast<char *>(&descriptor.at<float>(j, i)), sizeof(float));
		fin.close();

		return descriptor;
	}
}

bool WriteKPointsBinarySIFTGPU(char *fn, vector<SiftKeypoint>kpts, bool silent)
{
	ofstream fout;
	fout.open(fn, ios::binary);
	if (!fout.is_open())
	{
		if (silent)
			cout << "Cannot write: " << fn << endl;
		return false;
	}

	int npts = kpts.size();
	fout.write(reinterpret_cast<char *>(&npts), sizeof(int));
	for (int j = 0; j < npts; ++j)
	{
		fout.write(reinterpret_cast<char *>(&kpts.at(j).x), sizeof(float));
		fout.write(reinterpret_cast<char *>(&kpts.at(j).y), sizeof(float));
		fout.write(reinterpret_cast<char *>(&kpts.at(j).o), sizeof(float));
		fout.write(reinterpret_cast<char *>(&kpts.at(j).s), sizeof(float));
	}
	fout.close();

	return true;
}
bool ReadKPointsBinarySIFTGPU(char *fn, vector<SiftKeypoint> &kpts, bool silent)
{
	ifstream fin;
	fin.open(fn, ios::binary);
	if (!fin.is_open())
	{
		cout << "Cannot open: " << fn << endl;
		return false;
	}
	if (silent)
		cout << "Load " << fn << endl;

	float x, y, orirent, scale;
	SiftKeypoint kpt;

	int npts;
	fin.read(reinterpret_cast<char *>(&npts), sizeof(int));
	kpts.reserve(npts); kpts.clear();
	for (int ii = 0; ii < npts; ii++)
	{
		fin.read(reinterpret_cast<char *>(&x), sizeof(float));
		fin.read(reinterpret_cast<char *>(&y), sizeof(float));
		fin.read(reinterpret_cast<char *>(&orirent), sizeof(float));
		fin.read(reinterpret_cast<char *>(&scale), sizeof(float));
		kpt.x = x, kpt.y = y, kpt.o = orirent, kpt.s = scale;
		kpts.push_back(kpt);
	}

	return true;
}
bool WriteKPointsBinarySIFTGPU(char *fn, vector<KeyPoint>kpts, bool silent)
{
	ofstream fout;
	fout.open(fn, ios::binary);
	if (!fout.is_open())
	{
		if (silent)
			cout << "Cannot write: " << fn << endl;
		return false;
	}

	int npts = kpts.size();
	fout.write(reinterpret_cast<char *>(&npts), sizeof(int));
	for (int j = 0; j < npts; ++j)
	{
		fout.write(reinterpret_cast<char *>(&kpts.at(j).pt.x), sizeof(float));
		fout.write(reinterpret_cast<char *>(&kpts.at(j).pt.y), sizeof(float));
		fout.write(reinterpret_cast<char *>(&kpts.at(j).angle), sizeof(float));
		fout.write(reinterpret_cast<char *>(&kpts.at(j).size), sizeof(float));
	}
	fout.close();

	return true;
}
bool ReadKPointsBinarySIFTGPU(char *fn, vector<KeyPoint> &kpts, bool silent)
{
	ifstream fin;
	fin.open(fn, ios::binary);
	if (!fin.is_open())
	{
		cout << "Cannot open: " << fn << endl;
		return false;
	}
	if (silent)
		cout << "Load " << fn << endl;

	float x, y, orirent, scale;
	KeyPoint kpt;

	int npts;
	fin.read(reinterpret_cast<char *>(&npts), sizeof(int));
	kpts.reserve(npts); kpts.clear();
	for (int ii = 0; ii < npts; ii++)
	{
		fin.read(reinterpret_cast<char *>(&x), sizeof(float));
		fin.read(reinterpret_cast<char *>(&y), sizeof(float));
		fin.read(reinterpret_cast<char *>(&orirent), sizeof(float));
		fin.read(reinterpret_cast<char *>(&scale), sizeof(float));
		kpt.pt.x = x, kpt.pt.y = y, kpt.angle = orirent, kpt.size = scale;
		kpts.push_back(kpt);
	}

	return true;
}
bool WriteDescriptorBinarySIFTGPU(char *fn, vector<float > descriptors, bool silent)
{
	ofstream fout;
	fout.open(fn, ios::binary);
	if (!fout.is_open())
	{
		if (silent)
			cout << "Cannot write: " << fn << endl;
		return false;
	}

	int descriptorSize = SIFTBINS, npts = descriptors.size() / descriptorSize;
	fout.write(reinterpret_cast<char *>(&npts), sizeof(int));
	for (int j = 0; j < npts; ++j)
		for (int i = 0; i < descriptorSize; i++)
			fout.write(reinterpret_cast<char *>(&descriptors.at(i + j*descriptorSize)), sizeof(float));
	fout.close();

	return true;
}
bool ReadDescriptorBinarySIFTGPU(char *fn, vector<float > &descriptors, bool silent)
{
	descriptors.clear();
	ifstream fin;
	fin.open(fn, ios::binary);
	if (!fin.is_open())
	{
		cout << "Cannot open: " << fn << endl;
		return false;
	}
	if (silent)
		cout << "Load " << fn << endl;

	int npts, descriptorSize = SIFTBINS;
	float val;

	fin.read(reinterpret_cast<char *>(&npts), sizeof(int));
	descriptors.reserve(descriptorSize * npts);
	for (int j = 0; j < npts; j++)
	{
		for (int i = 0; i < descriptorSize; i++)
		{
			fin.read(reinterpret_cast<char *>(&val), sizeof(float));
			descriptors.push_back(val);
		}
	}
	fin.close();

	return true;
}
Mat ReadDescriptorBinarySIFTGPU(char *fn, bool silent)
{
	ifstream fin;
	fin.open(fn, ios::binary);
	if (!fin.is_open())
	{
		cout << "Cannot open: " << fn << endl;
		Mat descriptors(1, 128, CV_32F);
		return descriptors;
	}
	if (silent)
		cout << "Load " << fn << endl;

	int npts, descriptorSize = SIFTBINS;
	float val;

	fin.read(reinterpret_cast<char *>(&npts), sizeof(int));
	Mat descriptors(npts, 128, CV_32F);
	for (int j = 0; j < npts; j++)
	{
		for (int i = 0; i < descriptorSize; i++)
		{
			fin.read(reinterpret_cast<char *>(&val), sizeof(float));
			descriptors.at<float>(j, i) = val;
		}
	}
	fin.close();

	return descriptors;
}

void GenereteKeyPointsRGB(char *ImgName, char *KName, char *KeyRGBName)
{
	Mat view = imread(ImgName, 1);
	if (view.data == NULL)
	{
		cout << "Cannot load: " << ImgName << endl;
		return;
	}
	int width = view.cols, height = view.rows, length = width*height, nchannels = 3;
	unsigned char *Img = new unsigned char[length*nchannels];
	for (int kk = 0; kk < nchannels; kk++)
	{
		for (int jj = 0; jj < height; jj++)
			for (int ii = 0; ii < width; ii++)
				Img[ii + jj*width + kk*length] = view.data[nchannels*ii + jj*nchannels*width + kk];
	}

	vector<KeyPoint>kpts; kpts.reserve(30000);
	if (!ReadKPointsBinarySIFTGPU(KName, kpts))
		return;

	Point3i rgb;
	vector<Point3i>Argb; Argb.reserve(kpts.size());
	for (int kk = 0; kk < kpts.size(); kk++)
	{
		int x = kpts[kk].pt.x;
		int y = kpts[kk].pt.y;
		int id = x + y*width;

		rgb.z = Img[id];//b
		rgb.y = Img[length + id];//g
		rgb.x = Img[2 * length + id];//r
		Argb.push_back(rgb);
	}

	//WriteKPointsRGBBinarySIFTGPU(KeyRGBName, kpts, Argb);
	WriteRGBBinarySIFTGPU(KeyRGBName, Argb);
	delete[]Img;
}
bool WriteRGBBinarySIFTGPU(char *fn, vector<Point3i> rgb, bool silent)
{
	ofstream fout;
	fout.open(fn, ios::binary);
	if (!fout.is_open())
	{
		if (silent)
			cout << "Cannot write: " << fn << endl;
		return false;
	}

	int npts = rgb.size();
	fout.write(reinterpret_cast<char *>(&npts), sizeof(int));
	for (int j = 0; j < npts; ++j)
	{
		fout.write(reinterpret_cast<char *>(&rgb.at(j).x), sizeof(int));
		fout.write(reinterpret_cast<char *>(&rgb.at(j).y), sizeof(int));
		fout.write(reinterpret_cast<char *>(&rgb.at(j).z), sizeof(int));
	}
	fout.close();

	return true;
}
bool ReadRGBBinarySIFTGPU(char *fn, vector<Point3i> &rgb, bool silent)
{
	ifstream fin;
	fin.open(fn, ios::binary);
	if (!fin.is_open())
	{
		cout << "Cannot open: " << fn << endl;
		return false;
	}
	if (silent)
		cout << "Load " << fn << endl;

	int r, g, b, npts;
	fin.read(reinterpret_cast<char *>(&npts), sizeof(int));
	rgb.reserve(npts); rgb.clear();
	for (int ii = 0; ii < npts; ii++)
	{
		fin.read(reinterpret_cast<char *>(&r), sizeof(int));
		fin.read(reinterpret_cast<char *>(&g), sizeof(int));
		fin.read(reinterpret_cast<char *>(&b), sizeof(int));
		rgb.push_back(Point3i(r, g, b));
	}

	return true;
}
bool WriteKPointsRGBBinarySIFTGPU(char *fn, vector<SiftKeypoint>kpts, vector<Point3i> rgb, bool silent)
{
	ofstream fout;
	fout.open(fn, ios::binary);
	if (!fout.is_open())
	{
		if (silent)
			cout << "Cannot write: " << fn << endl;
		return false;
	}

	int npts = kpts.size();
	fout.write(reinterpret_cast<char *>(&npts), sizeof(int));
	for (int j = 0; j < npts; ++j)
	{
		fout.write(reinterpret_cast<char *>(&kpts.at(j).x), sizeof(float));
		fout.write(reinterpret_cast<char *>(&kpts.at(j).y), sizeof(float));
		fout.write(reinterpret_cast<char *>(&kpts.at(j).o), sizeof(float));
		fout.write(reinterpret_cast<char *>(&kpts.at(j).s), sizeof(float));
		fout.write(reinterpret_cast<char *>(&rgb.at(j).x), sizeof(int));
		fout.write(reinterpret_cast<char *>(&rgb.at(j).y), sizeof(int));
		fout.write(reinterpret_cast<char *>(&rgb.at(j).z), sizeof(int));
	}
	fout.close();

	return true;
}
bool ReadKPointsRGBBinarySIFTGPU(char *fn, vector<SiftKeypoint> &kpts, vector<Point3i> &rgb, bool silent)
{
	ifstream fin;
	fin.open(fn, ios::binary);
	if (!fin.is_open())
	{
		cout << "Cannot open: " << fn << endl;
		return false;
	}
	if (silent)
		cout << "Load " << fn << endl;

	int r, g, b;
	float x, y, orirent, scale;
	SiftKeypoint kpt;

	int npts;
	fin.read(reinterpret_cast<char *>(&npts), sizeof(int));
	kpts.reserve(npts); rgb.reserve(npts);  kpts.clear(); rgb.clear();
	for (int ii = 0; ii < npts; ii++)
	{
		fin.read(reinterpret_cast<char *>(&x), sizeof(float));
		fin.read(reinterpret_cast<char *>(&y), sizeof(float));
		fin.read(reinterpret_cast<char *>(&orirent), sizeof(float));
		fin.read(reinterpret_cast<char *>(&scale), sizeof(float));
		fin.read(reinterpret_cast<char *>(&r), sizeof(int));
		fin.read(reinterpret_cast<char *>(&g), sizeof(int));
		fin.read(reinterpret_cast<char *>(&b), sizeof(int));
		kpt.x = x, kpt.y = y, kpt.o = orirent, kpt.s = scale;
		kpts.push_back(kpt);
		rgb.push_back(Point3i(r, g, b));
	}

	return true;
}
bool WriteKPointsRGBBinarySIFTGPU(char *fn, vector<KeyPoint>kpts, vector<Point3i> rgb, bool silent)
{
	ofstream fout;
	fout.open(fn, ios::binary);
	if (!fout.is_open())
	{
		if (silent)
			cout << "Cannot write: " << fn << endl;
		return false;
	}

	int npts = kpts.size();
	fout.write(reinterpret_cast<char *>(&npts), sizeof(int));
	for (int j = 0; j < npts; ++j)
	{
		fout.write(reinterpret_cast<char *>(&kpts.at(j).pt.x), sizeof(float));
		fout.write(reinterpret_cast<char *>(&kpts.at(j).pt.y), sizeof(float));
		fout.write(reinterpret_cast<char *>(&kpts.at(j).angle), sizeof(float));
		fout.write(reinterpret_cast<char *>(&kpts.at(j).size), sizeof(float));
		fout.write(reinterpret_cast<char *>(&rgb.at(j).x), sizeof(int));
		fout.write(reinterpret_cast<char *>(&rgb.at(j).y), sizeof(int));
		fout.write(reinterpret_cast<char *>(&rgb.at(j).z), sizeof(int));
	}
	fout.close();

	return true;
}
bool ReadKPointsRGBBinarySIFTGPU(char *fn, vector<KeyPoint> &kpts, vector<Point3i> &rgb, bool silent)
{
	ifstream fin;
	fin.open(fn, ios::binary);
	if (!fin.is_open())
	{
		cout << "Cannot open: " << fn << endl;
		return false;
	}
	if (silent)
		cout << "Load " << fn << endl;

	int r, g, b;
	float x, y, orirent, scale;
	KeyPoint kpt;

	int npts;
	fin.read(reinterpret_cast<char *>(&npts), sizeof(int));
	kpts.reserve(npts); rgb.reserve(npts);  kpts.clear(); rgb.clear();
	for (int ii = 0; ii < npts; ii++)
	{
		fin.read(reinterpret_cast<char *>(&x), sizeof(float));
		fin.read(reinterpret_cast<char *>(&y), sizeof(float));
		fin.read(reinterpret_cast<char *>(&orirent), sizeof(float));
		fin.read(reinterpret_cast<char *>(&scale), sizeof(float));
		fin.read(reinterpret_cast<char *>(&r), sizeof(int));
		fin.read(reinterpret_cast<char *>(&g), sizeof(int));
		fin.read(reinterpret_cast<char *>(&b), sizeof(int));
		kpt.pt.x = x, kpt.pt.y = y, kpt.angle = orirent, kpt.size = scale;
		kpts.push_back(kpt);
		rgb.push_back(Point3i(r, g, b));
	}

	return true;
}

void ResizeImage(unsigned char *Image, unsigned char *OutImage, int width, int height, int nchannels, double Rfactor, int InterpAlgo, double *InPara)
{
	bool createMem = false;
	int length = width*height;
	if (InPara == NULL)
	{
		createMem = true;
		InPara = new double[length*nchannels];
		for (int kk = 0; kk < nchannels; kk++)
			Generate_Para_Spline(Image + kk*length, InPara + kk*length, width, height, InterpAlgo);
	}

	double S[3];
	int nwidth = width*Rfactor, nheight = height*Rfactor, nlength = nwidth*nheight;
	for (int kk = 0; kk < nchannels; kk++)
		for (int jj = 0; jj < nheight; jj++)
		{
		for (int ii = 0; ii<nwidth; ii++)
		{
			Get_Value_Spline(InPara + kk*length, width, height, 1.0*ii / Rfactor, 1.0*jj / Rfactor, S, -1, InterpAlgo);
			if (S[0]>255.0)
				OutImage[ii + jj*nwidth + kk*nlength] = 255;
			else if (S[0] < 0.0)
				OutImage[ii + jj*nwidth + kk*nlength] = 0;
			else
				OutImage[ii + jj*nwidth + kk*nlength] = (unsigned char)(int)(S[0] + 0.5);
		}
		}

	if (createMem)
		delete[]InPara;

	return;
}
void ResizeImage(float *Image, float *OutImage, int width, int height, int nchannels, double Rfactor, int InterpAlgo, float *InPara)
{
	bool createMem = false;
	int length = width*height;
	if (InPara == NULL)
	{
		createMem = true;
		InPara = new float[width*height*nchannels];
		for (int kk = 0; kk < nchannels; kk++)
			Generate_Para_Spline(Image + kk*length, InPara + kk*length, width, height, InterpAlgo);
	}

	double S[3];
	int nwidth = width*Rfactor, nheight = height*Rfactor, nlength = nwidth*nheight;
	for (int kk = 0; kk < nchannels; kk++)
		for (int jj = 0; jj < nheight; jj++)
		{
		for (int ii = 0; ii<nwidth; ii++)
		{
			Get_Value_Spline(InPara + kk*length, width, height, 1.0*ii / Rfactor, 1.0*jj / Rfactor, S, -1, InterpAlgo);
			if (S[0]>255.0)
				OutImage[ii + jj*nwidth + kk*nlength] = 255;
			else if (S[0] < 0.0)
				OutImage[ii + jj*nwidth + kk*nlength] = 0;
			else
				OutImage[ii + jj*nwidth + kk*nlength] = (float)S[0];
		}
		}

	if (createMem)
		delete[]InPara;

	return;
}
void ResizeImage(double *Image, double *OutImage, int width, int height, int nchannels, double Rfactor, int InterpAlgo, double *InPara)
{
	bool createMem = false;
	int length = width*height;
	if (InPara == NULL)
	{
		createMem = true;
		InPara = new double[width*height*nchannels];
		for (int kk = 0; kk < nchannels; kk++)
			Generate_Para_Spline(Image + kk*length, InPara + kk*length, width, height, InterpAlgo);
	}

	double S[3];
	int nwidth = width*Rfactor, nheight = height*Rfactor, nlength = nwidth*nheight;
	for (int kk = 0; kk < nchannels; kk++)
		for (int jj = 0; jj < nheight; jj++)
		{
		for (int ii = 0; ii<nwidth; ii++)
		{
			Get_Value_Spline(InPara + kk*length, width, height, 1.0*ii / Rfactor, 1.0*jj / Rfactor, S, -1, InterpAlgo);
			if (S[0]>255.0)
				OutImage[ii + jj*nwidth + kk*nlength] = 255;
			else if (S[0] < 0.0)
				OutImage[ii + jj*nwidth + kk*nlength] = 0;
			else
				OutImage[ii + jj*nwidth + kk*nlength] = S[0];
		}
		}

	if (createMem)
		delete[]InPara;

	return;
}

double interpolate(double val, double y0, double x0, double y1, double x1)
{
	return (val - x0)*(y1 - y0) / (x1 - x0) + y0;
}
double base(double val)
{
	if (val <= -0.75) return 0;
	else if (val <= -0.25) return interpolate(val, 0.0, -0.75, 1.0, -0.25);
	else if (val <= 0.25) return 1.0;
	else if (val <= 0.75) return interpolate(val, 1.0, 0.25, 0.0, 0.75);
	else return 0.0;
}
double red(double gray)
{
	return base(gray - 0.5);
}
double green(double gray)
{
	return base(gray);
}
double blue(double gray)
{
	return base(gray + 0.5);
}
void ConvertToHeatMap(double *Map, unsigned char *ColorMap, int width, int height, bool *mask)
{
	int ii, jj;
	double gray;
	if (mask)
	{
		for (jj = 0; jj < height; jj++)
			for (ii = 0; ii < width; ii++)
			{
			if (mask[ii + jj*width])
			{
				ColorMap[3 * ii + 3 * jj*width] = 0;
				ColorMap[3 * ii + 3 * jj*width + 1] = 0;
				ColorMap[3 * ii + 3 * jj*width + 2] = 0;
			}
			else
			{
				gray = Map[ii + jj*width];
				ColorMap[3 * ii + 3 * jj*width] = (unsigned char)(int)(255.0*red(gray) + 0.5);
				ColorMap[3 * ii + 3 * jj*width + 1] = (unsigned char)(int)(255.0*green(gray) + 0.5);
				ColorMap[3 * ii + 3 * jj*width + 2] = (unsigned char)(int)(255.0*blue(gray) + 0.5);
			}
			}
	}
	else
	{
		for (jj = 0; jj < height; jj++)
			for (ii = 0; ii < width; ii++)
			{
			gray = Map[ii + jj*width];
			ColorMap[3 * ii + 3 * jj*width] = (unsigned char)(int)(255.0*red(gray) + 0.5);
			ColorMap[3 * ii + 3 * jj*width + 1] = (unsigned char)(int)(255.0*green(gray) + 0.5);
			ColorMap[3 * ii + 3 * jj*width + 2] = (unsigned char)(int)(255.0*blue(gray) + 0.5);
			}
	}

	return;
}

int PickStaticImagesFromVideo(char *PATH, char *VideoName, int SaveFrameDif, int redetectInterval, double percentile, double MovingThresh2, int &nNonBlurImages, bool visCamual)
{
	Mat colorImg, gray, prevGray, tImg, backGround, bestFrameInWind;
	vector<Point2f> points[2];
	vector<double>flowMag2;
	vector<uchar> status;
	vector<float> err;

	char Fname[200];
	sprintf(Fname, "%s/%s", PATH, VideoName);
	VideoCapture  capture(Fname);
	if (!capture.isOpened())  // if not success, exit program
	{
		printf("Cannot open %s\n", Fname);
		return -1;
	}

	cvNamedWindow("Static Image detection with LK", WINDOW_NORMAL);

	bool needToInit = true;
	int MAX_COUNT = 5000, frameID = 0, lastSaveframe = -SaveFrameDif - 1;

	TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
	Size subPixWinSize(21, 21), winSize(31, 31);

	nNonBlurImages = 0;
	int bestframeID;
	vector<double> distance; distance.reserve(500);
	double Movement, smallestMovement = 1000.0;
	while (true)
	{
		if (!capture.read(colorImg))
			break;
		cvtColor(colorImg, gray, CV_BGR2GRAY);

		if (visCamual) //Create background
			cvtColor(gray, backGround, CV_GRAY2BGR);

		if (frameID == 0) // automatic initialization
		{
			goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
			cornerSubPix(gray, points[1], subPixWinSize, Size(-1, -1), termcrit);
			if (visCamual)
				for (int jj = 0; jj < points[1].size() && visCamual; jj++)
					circle(backGround, points[1][jj], 5, Scalar(83, 185, 255), -1, 8);
		}

		if (!points[0].empty())
		{
			status.clear(); err.clear();
			if (prevGray.empty())
				gray.copyTo(prevGray);
			calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize, 1, termcrit, 0, 0.001);

			size_t i, k;
			flowMag2.clear();
			for (i = k = 0; i < points[1].size(); i++)
			{
				if (!status[i])
					continue;

				flowMag2.push_back((points[1][i].x - points[0][i].x)*(points[1][i].x - points[0][i].x) + (points[1][i].y - points[0][i].y)*(points[1][i].y - points[0][i].y));

				points[1][k++] = points[1][i];
				if (visCamual)
					circle(backGround, points[1][i], 5, Scalar(83, 185, 255), -1, 8);
			}
			points[1].resize(k);

			sort(flowMag2.begin(), flowMag2.end());
			if (flowMag2.size()>0)
				distance.push_back(flowMag2.at((int)(percentile*flowMag2.size())));
			else
				distance.push_back(-1);
			printf("@frame %d: %.3f\n", frameID, distance.at(frameID - 1));


			if (flowMag2.size() > 0)
			{
				Movement = flowMag2.at((int)(percentile*flowMag2.size()));
				if (smallestMovement > Movement)
				{
					bestFrameInWind = colorImg;
					smallestMovement = Movement;
					bestframeID = frameID;
				}

				if (0.3*Movement > smallestMovement && smallestMovement < MovingThresh2 && frameID - lastSaveframe > SaveFrameDif)
				{
					printf("Saving frame %d\n", bestframeID);
					sprintf(Fname, "%s/_%d.png", PATH, nNonBlurImages);
					imwrite(Fname, bestFrameInWind);
					lastSaveframe = frameID;
					smallestMovement = 1000.0;
					nNonBlurImages++;
				}

				if (flowMag2.size() < 50 || frameID%redetectInterval == 0)
				{
					goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
					cornerSubPix(gray, points[1], subPixWinSize, Size(-1, -1), termcrit);
				}
			}
			else
			{
				goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
				cornerSubPix(gray, points[1], subPixWinSize, Size(-1, -1), termcrit);
			}
		}

		needToInit = false;
		if (visCamual)
		{
			imshow("Static Image detection with LK", backGround);
			char c = (char)waitKey(10);
			if (c == 27)
				break;
		}

		std::swap(points[1], points[0]);
		swap(prevGray, gray);
		frameID++;
	}
	return 0;
}
int PickStaticImagesFromImages(char *PATH, int SaveFrameDif, int redetectInterval, double percentile, double MovingThresh2, bool visCamual)
{
	Mat colorImg, gray, prevGray, tImg, backGround, bestFrameInWind;
	vector<Point2f> points[2];
	vector<double>flowMag2;
	vector<uchar> status;
	vector<float> err;

	char Fname[200];

	cvNamedWindow("Static Image detection with LK", WINDOW_NORMAL);

	bool needToInit = true;
	int MAX_COUNT = 5000, frameID = 0, lastSaveframe = -SaveFrameDif - 1;

	TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
	Size subPixWinSize(21, 21), winSize(31, 31);

	vector<double> distance; distance.reserve(500);
	int bestframeID = 0;
	double Movement, smallestMovement = 1000.0;
	while (true)
	{
		sprintf(Fname, "%s/%d.png", PATH, frameID + 1);
		colorImg = imread(Fname, 1);
		if (colorImg.empty())
			break;

		cvtColor(colorImg, gray, CV_BGR2GRAY);

		if (visCamual) //Create background
			cvtColor(gray, backGround, CV_GRAY2BGR);

		if (frameID == 0) // automatic initialization
		{
			goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
			cornerSubPix(gray, points[1], subPixWinSize, Size(-1, -1), termcrit);
			if (visCamual)
				for (int jj = 0; jj < points[1].size() && visCamual; jj++)
					circle(backGround, points[1][jj], 5, Scalar(83, 185, 255), -1, 8);
		}

		if (!points[0].empty())
		{
			status.clear(); err.clear();
			if (prevGray.empty())
				gray.copyTo(prevGray);
			calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize, 1, termcrit, 0, 0.001);

			size_t i, k;
			flowMag2.clear();
			for (i = k = 0; i < points[1].size(); i++)
			{
				if (!status[i])
					continue;

				flowMag2.push_back((points[1][i].x - points[0][i].x)*(points[1][i].x - points[0][i].x) + (points[1][i].y - points[0][i].y)*(points[1][i].y - points[0][i].y));

				points[1][k++] = points[1][i];
				if (visCamual)
					circle(backGround, points[1][i], 5, Scalar(83, 185, 255), -1, 8);
			}
			points[1].resize(k);

			sort(flowMag2.begin(), flowMag2.end());
			if (flowMag2.size()>0)
				distance.push_back(flowMag2.at((int)(percentile*flowMag2.size())));
			else
				distance.push_back(-1);
			printf("@frame %d: %.3f\n", frameID, distance.at(frameID - 1));

			if (flowMag2.size() > 0)
			{
				Movement = flowMag2.at((int)(percentile*flowMag2.size()));
				if (smallestMovement > Movement)
				{
					bestFrameInWind = colorImg;
					smallestMovement = Movement;
					bestframeID = frameID;
				}

				if (0.3*Movement > smallestMovement && smallestMovement < MovingThresh2 && frameID - lastSaveframe > SaveFrameDif)
				{
					printf("Saving frame %d\n", bestframeID);
					sprintf(Fname, "%s/_%d.png", PATH, bestframeID);
					imwrite(Fname, bestFrameInWind);
					lastSaveframe = frameID;
					smallestMovement = 1000.0;
				}

				if (flowMag2.size() < 50 || frameID%redetectInterval == 0)
				{
					goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
					cornerSubPix(gray, points[1], subPixWinSize, Size(-1, -1), termcrit);
				}
			}
			else
			{
				printf("Redected @ frame %d due to low # of features", frameID);
				goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
				cornerSubPix(gray, points[1], subPixWinSize, Size(-1, -1), termcrit);
			}
		}

		needToInit = false;
		if (visCamual)
		{
			imshow("Static Image detection with LK", backGround);
			char c = (char)waitKey(10);
			if (c == 27)
				break;
		}

		std::swap(points[1], points[0]);
		swap(prevGray, gray);
		frameID++;
	}

	FILE *fp = fopen("C:/temp/distance.txt", "w+");
	for (int ii = 0; ii < distance.size(); ii++)
		fprintf(fp, "%.3f\n", distance.at(ii));
	fclose(fp);

	return 0;
}

void nonMaximaSuppression(const Mat& src, const int sz, Mat& dst, const Mat mask)
{

	// initialise the block mask and destination
	const int M = src.rows;
	const int N = src.cols;
	const bool masked = !mask.empty();
	Mat block = 255 * Mat_<unsigned char>::ones(Size(2 * sz + 1, 2 * sz + 1));
	dst = Mat_<unsigned char>::zeros(src.size());

	// iterate over image blocks
	for (int m = 0; m < M; m += sz + 1)
	{
		for (int n = 0; n < N; n += sz + 1)
		{
			Point  ijmax;
			double vcmax, vnmax;

			// get the maximal candidate within the block
			Range ic(m, min(m + sz + 1, M));
			Range jc(n, min(n + sz + 1, N));
			minMaxLoc(src(ic, jc), NULL, &vcmax, NULL, &ijmax, masked ? mask(ic, jc) : noArray());
			Point cc = ijmax + Point(jc.start, ic.start);

			// search the neighbours centered around the candidate for the true maxima
			Range in(max(cc.y - sz, 0), min(cc.y + sz + 1, M));
			Range jn(max(cc.x - sz, 0), min(cc.x + sz + 1, N));

			// mask out the block whose maxima we already know
			Mat_<unsigned char> blockmask;
			block(Range(0, in.size()), Range(0, jn.size())).copyTo(blockmask);
			Range iis(ic.start - in.start, min(ic.start - in.start + sz + 1, in.size()));
			Range jis(jc.start - jn.start, min(jc.start - jn.start + sz + 1, jn.size()));
			blockmask(iis, jis) = Mat_<unsigned char>::zeros(Size(jis.size(), iis.size()));

			minMaxLoc(src(in, jn), NULL, &vnmax, NULL, &ijmax, masked ? mask(in, jn).mul(blockmask) : blockmask);
			Point cn = ijmax + Point(jn.start, in.start);

			// if the block centre is also the neighbour centre, then it's a local maxima
			if (vcmax > vnmax) {
				dst.at<unsigned char>(cc.y, cc.x) = 255;
			}
		}
	}

	return;
}
int LensCorrectionVideoDriver(char *Path, char *VideoName, double *K, double *distortion, int LensType, int nimages, double Imgscale, double Contscale, int interpAlgo)
{
	char Fname[200];
	double iK[9];

	mat_invert(K, iK, 3);
	double omega, DistCtr[2];
	if (LensType == 1)
		omega = distortion[0], DistCtr[0] = distortion[1], DistCtr[1] = distortion[2];
	else if (LensType == 2)
		omega = distortion[0];

	Mat cvImg;
	unsigned char *Img = 0;
	double *Para = 0;

	VideoCapture  capture(VideoName);
	if (!capture.isOpened())  // if not success, exit program
	{
		printf("Cannot open %s\n", Fname);
		return -1;
	}

	for (int Id = 0; Id < nimages; Id++)
	{
		if (!capture.read(cvImg))
			break;

		int width = cvImg.cols, height = cvImg.rows, nchannels = cvImg.channels();
		int Mwidth = Imgscale*width, Mheight = Imgscale*height, Mlength = Mwidth*Mheight;
		if (Id == 0)
		{
			Img = new unsigned char[Mlength*nchannels];
			Para = new double[Mlength*nchannels];
		}

		for (int kk = 0; kk < nchannels; kk++)
		{
			for (int jj = 0; jj < height; jj++)
				for (int ii = 0; ii < width; ii++)
					Img[ii + jj*width + kk*width*height] = cvImg.data[ii*nchannels + jj*width*nchannels + kk];
			if (Para != NULL)
				Generate_Para_Spline(Img + kk*width*height, Para + kk*width*height, width, height, interpAlgo);
		}

		if (LensType == 0)
			LensUndistortion(Img, width, height, nchannels, K, distortion, interpAlgo, Imgscale, Contscale, Para);
		else if (LensType == 1)
			FishEyeCorrection(Img, cvImg.cols, cvImg.rows, 3, omega, DistCtr[0], DistCtr[1], interpAlgo, Imgscale, Contscale, Para);
		else if (LensType == 2)
			FishEyeCorrection(Img, cvImg.cols, cvImg.rows, 3, K, iK, omega, interpAlgo, Imgscale, Contscale, Para);
		else
			return 1;

		Mat nImg(Mheight, Mwidth, CV_8UC3);
		for (int kk = 0; kk < nchannels; kk++)
			for (int jj = 0; jj < Mheight; jj++)
				for (int ii = 0; ii < Mwidth; ii++)
					nImg.data[ii*nchannels + jj*Mwidth*nchannels + kk] = Img[ii + jj*Mwidth + kk*Mlength];

		sprintf(Fname, "%s/%d.png", Path, Id + 1);
		imwrite(Fname, nImg);
	}

	delete[]Img;

	return 0;
}
int LensCorrectionImageSequenceDriver(char *Path, double *K, double *distortion, int LensType, int StartFrame, int StopFrame, double Imgscale, double Contscale, int interpAlgo)
{
	char Fname[200];
	double iK[9];

	mat_invert(K, iK, 3);
	double omega, DistCtr[2];
	if (LensType == FISHEYE)
		omega = distortion[0], DistCtr[0] = distortion[1], DistCtr[1] = distortion[2];
	else if (LensType == RADIAL_TANGENTIAL_PRISM)
		omega = distortion[0];

	Mat cvImg;
	unsigned char *Img = 0;
	double *Para = 0;

	for (int Id = StartFrame; Id <= StopFrame; Id++)
	{
		sprintf(Fname, "%s/%d.ppm", Path, Id);	cvImg = imread(Fname, 1);
		if (cvImg.empty())
		{
			printf("Cannot load %s\n", Fname);
			continue;
		}

		int width = cvImg.cols, height = cvImg.rows, nchannels = cvImg.channels();
		int Mwidth = Imgscale*width, Mheight = Imgscale*height, Mlength = Mwidth*Mheight;
		if (Id == StartFrame)
		{
			Img = new unsigned char[Mlength*nchannels];
			Para = new double[Mlength*nchannels];
		}

		for (int kk = 0; kk < nchannels; kk++)
		{
			for (int jj = 0; jj < height; jj++)
				for (int ii = 0; ii < width; ii++)
					Img[ii + jj*width + kk*width*height] = cvImg.data[ii*nchannels + jj*width*nchannels + kk];
			if (Para != NULL)
				Generate_Para_Spline(Img + kk*width*height, Para + kk*width*height, width, height, interpAlgo);
		}

		if (LensType == RADIAL_TANGENTIAL_PRISM)
			LensUndistortion(Img, width, height, nchannels, K, distortion, interpAlgo, Imgscale, Contscale, Para);
		else if (LensType == FISHEYE)
			FishEyeCorrection(Img, cvImg.cols, cvImg.rows, 3, omega, DistCtr[0], DistCtr[1], interpAlgo, Imgscale, Contscale, Para);
		else if (LensType == 2)
			FishEyeCorrection(Img, cvImg.cols, cvImg.rows, 3, K, iK, omega, interpAlgo, Imgscale, Contscale, Para);
		else
			return 1;

		Mat nImg(Mheight, Mwidth, CV_8UC3);
		for (int kk = 0; kk < nchannels; kk++)
			for (int jj = 0; jj < Mheight; jj++)
				for (int ii = 0; ii < Mwidth; ii++)
					nImg.data[ii*nchannels + jj*Mwidth*nchannels + kk] = Img[ii + jj*Mwidth + kk*Mlength];

		sprintf(Fname, "%s/%d.png", Path, Id);
		imwrite(Fname, nImg);
	}

	delete[]Img;

	return 0;
}
int LensCorrectionDriver(char *Path, double *K, double *distortion, int LensType, int startID, int stopID, double Imgscale, double Contscale, int interpAlgo)
{
	char Fname[200];

	//double Imgscale = 1.0, Contscale = 1.0, iK[9];
	//mat_invert(K, iK, 3);

	Mat cvImg;
	unsigned char *Img = 0;
	double *Para = 0;
	for (int Id = startID; Id <= stopID; Id++)
	{
		sprintf(Fname, "%s/%d.png", Path, Id);
		cvImg = imread(Fname, CV_LOAD_IMAGE_COLOR);
		if (cvImg.data == NULL)
		{
			printf("Cannot read %s\n", Fname);
			return 1;
		}
		else
			printf("Loaded %s\n", Fname);

		int width = cvImg.cols, height = cvImg.rows, nchannels = cvImg.channels();
		int Mwidth = Imgscale*width, Mheight = Imgscale*height, Mlength = Mwidth*Mheight;
		if (Id == startID)
		{
			Img = new unsigned char[Mlength*nchannels];
			Para = new double[Mlength*nchannels];
		}

		for (int kk = 0; kk < nchannels; kk++)
		{
			for (int jj = 0; jj < height; jj++)
				for (int ii = 0; ii < width; ii++)
					Img[ii + jj*width + kk*width*height] = cvImg.data[ii*nchannels + jj*width*nchannels + kk];
			if (Para != NULL)
				Generate_Para_Spline(Img + kk*width*height, Para + kk*width*height, width, height, interpAlgo);
		}

		if (LensType == RADIAL_TANGENTIAL_PRISM)
			LensUndistortion(Img, width, height, nchannels, K, distortion, interpAlgo, Imgscale, Contscale, Para);
		else if (LensType == FISHEYE)
			FishEyeCorrection(Img, cvImg.cols, cvImg.rows, 3, distortion[0], distortion[1], distortion[2], interpAlgo, Imgscale, 1.0);
		else if (LensType == 2)
			printf("This lens model is not supported right now!");// FishEyeCorrection(Img, cvImg.cols, cvImg.rows, 3, K, iK, omega, interpAlgo, Imgscale, Contscale);

		Mat nImg(Mheight, Mwidth, CV_8UC3);
		for (int kk = 0; kk < nchannels; kk++)
			for (int jj = 0; jj < Mheight; jj++)
				for (int ii = 0; ii < Mwidth; ii++)
					nImg.data[ii*nchannels + jj*Mwidth*nchannels + kk] = Img[ii + jj*Mwidth + kk*Mlength];

		sprintf(Fname, "%s/U%d.png", Path, Id);
		imwrite(Fname, nImg);
	}

	delete[]Img;

	return 0;
}

int DisplayImageCorrespondence(IplImage* correspond, int offsetX, int offsetY, vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2, vector<int>pair, double density)
{
	static CvScalar colors[] =
	{
		{ { 0, 0, 255 } },
		{ { 0, 128, 255 } },
		{ { 0, 255, 255 } },
		{ { 0, 255, 0 } },
		{ { 255, 128, 0 } },
		{ { 255, 255, 0 } },
		{ { 255, 0, 0 } },
		{ { 255, 0, 255 } },
		{ { 255, 255, 255 } }
	};
	int nmatches = pair.size() / 2, step = (int)((2.0 / density)) / 2 * 2;
	step = step > 0 ? step : 2;
	cout << step << endl;

	FILE *fp = fopen("C:/temp/corres.txt", "w+");
	for (int ii = 0; ii < pair.size(); ii += 2)
	{
		int x1 = keypoints1.at(pair.at(ii)).pt.x, y1 = keypoints1.at(pair.at(ii)).pt.y;
		int x2 = keypoints2.at(pair.at(ii + 1)).pt.x + offsetX, y2 = keypoints2.at(pair.at(ii + 1)).pt.y + offsetY;
		fprintf(fp, "%.1f %.1f %1.f %.1f\n", keypoints1.at(pair.at(ii)).pt.x, keypoints1.at(pair.at(ii)).pt.y, keypoints2.at(pair.at(ii + 1)).pt.x, keypoints2.at(pair.at(ii + 1)).pt.y);
	}
	fclose(fp);

	for (int ii = 0; ii < pair.size(); ii += step)
	{
		int x1 = keypoints1.at(pair.at(ii)).pt.x, y1 = keypoints1.at(pair.at(ii)).pt.y;
		int x2 = keypoints2.at(pair.at(ii + 1)).pt.x + offsetX, y2 = keypoints2.at(pair.at(ii + 1)).pt.y + offsetY;
		cvLine(correspond, cvPoint(x1, y1), cvPoint(x2, y2), colors[ii % 9], 1, 4);
	}

	cvNamedWindow("Correspondence", CV_WINDOW_NORMAL);
	cvShowImage("Correspondence", correspond);
	cvWaitKey(-1);
	printf("Images closed\n");
	return 0;
}
int DisplayImageCorrespondence(IplImage* correspond, int offsetX, int offsetY, vector<Point2d> keypoints1, vector<Point2d> keypoints2, vector<int>pair, double density)
{
	static CvScalar colors[] =
	{
		{ { 0, 0, 255 } },
		{ { 0, 128, 255 } },
		{ { 0, 255, 255 } },
		{ { 0, 255, 0 } },
		{ { 255, 128, 0 } },
		{ { 255, 255, 0 } },
		{ { 255, 0, 0 } },
		{ { 255, 0, 255 } },
		{ { 255, 255, 255 } }
	};
	int nmatches = pair.size() / 2, step = (int)((2.0 / density)) / 2 * 2;
	step = step > 0 ? step : 2;
	cout << step << endl;

	FILE *fp = fopen("C:/temp/corres.txt", "w+");
	for (int ii = 0; ii < pair.size(); ii += 2)
	{
		int x1 = keypoints1.at(pair.at(ii)).x, y1 = keypoints1.at(pair.at(ii)).y;
		int x2 = keypoints2.at(pair.at(ii + 1)).x + offsetX, y2 = keypoints2.at(pair.at(ii + 1)).y + offsetY;
		fprintf(fp, "%.1f %.1f %1.f %.1f\n", keypoints1.at(pair.at(ii)).x, keypoints1.at(pair.at(ii)).y, keypoints2.at(pair.at(ii + 1)).x, keypoints2.at(pair.at(ii + 1)).y);
	}
	fclose(fp);

	for (int ii = 0; ii < pair.size(); ii += step)
	{
		int x1 = keypoints1.at(pair.at(ii)).x, y1 = keypoints1.at(pair.at(ii)).y;
		int x2 = keypoints2.at(pair.at(ii + 1)).x + offsetX, y2 = keypoints2.at(pair.at(ii + 1)).y + offsetY;
		cvLine(correspond, cvPoint(x1, y1), cvPoint(x2, y2), colors[ii % 9], 1, 4);
	}

	cvNamedWindow("Correspondence", CV_WINDOW_NORMAL);
	cvShowImage("Correspondence", correspond);
	cvWaitKey(-1);
	printf("Images closed\n");
	return 0;
}
int DisplayImageCorrespondencesDriver(char *Path, vector<int>AvailViews, int timeID, int nchannels, double density)
{
	char Fname[200];

	vector<int>CorrespondencesID;
	vector<KeyPoint>keypoints1, keypoints2;
	GetPoint2DPairCorrespondence(Path, timeID, AvailViews, keypoints1, keypoints2, CorrespondencesID);

	if (timeID < 0)
		sprintf(Fname, "%s/%d.png", Path, AvailViews.at(0));
	else
		sprintf(Fname, "%s/%d/%d.png", Path, AvailViews.at(0), timeID);
	IplImage *Img1 = cvLoadImage(Fname, nchannels == 3 ? 1 : 0);
	if (Img1->imageData == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return 1;
	}

	if (timeID < 0)
		sprintf(Fname, "%s/%d.png", Path, AvailViews.at(1));
	else
		sprintf(Fname, "%s/%d/%d.png", Path, AvailViews.at(1), timeID);
	IplImage *Img2 = cvLoadImage(Fname, nchannels == 3 ? 1 : 0);
	if (Img2->imageData == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return 1;
	}

	IplImage* correspond = cvCreateImage(cvSize(Img1->width + Img2->width, Img1->height), 8, nchannels);
	cvSetImageROI(correspond, cvRect(0, 0, Img1->width, Img1->height));
	cvCopy(Img1, correspond);
	cvSetImageROI(correspond, cvRect(Img1->width, 0, correspond->width, correspond->height));
	cvCopy(Img2, correspond);
	cvResetImageROI(correspond);

	DisplayImageCorrespondence(correspond, Img1->width, 0, keypoints1, keypoints2, CorrespondencesID, density);

	return 0;
}

int ReadIntrinsicResults(char *path, CameraData *AllViewsParas)
{
	//Note that visCamualSfm use different lens model than openCV or matlab or yours (inverse model)
	char Fname[200];
	int id = 0, lensType, width, height;
	double fx, fy, skew, u0, v0, r0, r1, r2, t0, t1, p0, p1, omega, DistCtrX, DistCtrY;

	sprintf(Fname, "%s/DevicesIntrinsics.txt", path); FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		cout << "Cannot load " << Fname << endl;
		return 1;
	}
	while (fscanf(fp, "%d %d %d %lf %lf %lf %lf %lf ", &lensType, &width, &height, &fx, &fy, &skew, &u0, &v0) != EOF)
	{
		AllViewsParas[id].LensModel = lensType, AllViewsParas[id].width = width, AllViewsParas[id].height = height;
		AllViewsParas[id].K[0] = fx, AllViewsParas[id].K[1] = skew, AllViewsParas[id].K[2] = u0,
			AllViewsParas[id].K[3] = 0.0, AllViewsParas[id].K[4] = fy, AllViewsParas[id].K[5] = v0,
			AllViewsParas[id].K[6] = 0.0, AllViewsParas[id].K[7] = 0.0, AllViewsParas[id].K[8] = 1.0;

		GetIntrinsicFromK(AllViewsParas[id]);
		//mat_invert(AllViewsParas[id].K, AllViewsParas[id].iK);
		if (lensType == RADIAL_TANGENTIAL_PRISM)
		{
			fscanf(fp, " %lf %lf %lf %lf %lf %lf %lf ", &r0, &r1, &r2, &t0, &t1, &p0, &p1);
			AllViewsParas[id].distortion[0] = r0, AllViewsParas[id].distortion[1] = r1, AllViewsParas[id].distortion[2] = r2;
			AllViewsParas[id].distortion[3] = t0, AllViewsParas[id].distortion[4] = t1;
			AllViewsParas[id].distortion[5] = p0, AllViewsParas[id].distortion[6] = p1;
		}
		else
		{
			fscanf(fp, " %lf %lf %lf ", &omega, &DistCtrX, &DistCtrY);
			AllViewsParas[id].distortion[0] = omega, AllViewsParas[id].distortion[1] = DistCtrX, AllViewsParas[id].distortion[2] = DistCtrY;
			AllViewsParas[id].distortion[3] = 0, AllViewsParas[id].distortion[4] = 0;
			AllViewsParas[id].distortion[5] = 0, AllViewsParas[id].distortion[6] = 0;
		}
		id++;
	}
	fclose(fp);

	return 0;
}
int SaveIntrinsicResults(char *path, CameraData *AllViewsParas, int nCams)
{
	//Note that visCamualSfm use different lens model than openCV or matlab or yours (inverse model)
	char Fname[200];
	int id = 0, LensType;
	double fx, fy, skew, u0, v0, r0, r1, r2, t0, t1, p0, p1, omega, DistCtrX, DistCtrY;

	sprintf(Fname, "%s/DevicesIntrinsics.txt", path); FILE *fp = fopen(Fname, "w+");
	if (fp == NULL)
	{
		cout << "Cannot load " << Fname << endl;
		return 1;
	}
	for (int ii = 0; ii < nCams; ii++)
	{
		LensType = AllViewsParas[id].LensModel;
		fx = AllViewsParas[id].K[0], fy = AllViewsParas[id].K[4], skew = AllViewsParas[id].K[1], u0 = AllViewsParas[id].K[2], v0 = AllViewsParas[id].K[5];

		if (LensType == RADIAL_TANGENTIAL_PRISM)
		{
			r0 = AllViewsParas[id].distortion[0], r1 = AllViewsParas[id].distortion[1], r2 = AllViewsParas[id].distortion[2];
			t0 = AllViewsParas[id].distortion[3], t1 = AllViewsParas[id].distortion[4];
			p0 = AllViewsParas[id].distortion[5], p1 = AllViewsParas[id].distortion[6];
			fprintf(fp, "%d %d %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf \n", LensType, AllViewsParas[id].width, AllViewsParas[id].height, fx, fy, skew, u0, v0, r0, r1, r2, t0, t1, p0, p1);
		}
		else
		{
			omega = AllViewsParas[id].distortion[0], DistCtrX = AllViewsParas[id].distortion[1], DistCtrY = AllViewsParas[id].distortion[2];
			fprintf(fp, "%d %d %d %lf %lf %lf %lf %lf %lf %lf %lf \n", LensType, AllViewsParas[id].width, AllViewsParas[id].height, fx, fy, skew, u0, v0, omega, DistCtrX, DistCtrY);
		}
	}
	fclose(fp);

	return 0;
}
void SaveCurrentSfmInfo(char *path, CameraData *AllViewParas, vector<int>AvailViews, Point3d *All3D, int npts)
{
	char Fname[200];

	sprintf(Fname, "%s/Dinfo.txt", path);
	FILE *fp = fopen(Fname, "w+");
	for (int ii = 0; ii < AvailViews.size(); ii++)
	{
		int viewID = AvailViews.at(ii);
		fprintf(fp, "%d: ", viewID);
		for (int jj = 0; jj < 5; jj++)
			fprintf(fp, "%.16f ", AllViewParas[viewID].intrinsic[jj]);
		for (int jj = 0; jj < 7; jj++)
			fprintf(fp, "%.16f ", AllViewParas[viewID].distortion[jj]);
		for (int jj = 0; jj < 6; jj++)
			fprintf(fp, "%.16f ", AllViewParas[viewID].rt[jj]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	if (All3D != NULL)
	{
		sprintf(Fname, "%s/3d.xyz", path);
		fp = fopen(Fname, "w+");
		for (int ii = 0; ii < npts; ii++)
		{
			if (abs(All3D[ii].x) + abs(All3D[ii].y) + abs(All3D[ii].z) < 0.001)
				continue;
			fprintf(fp, "%.16f %.16f %.16f \n", All3D[ii].x, All3D[ii].y, All3D[ii].z);
		}
		fclose(fp);
	}

	return;
}
void ReadCurrentSfmInfo(char *path, CameraData *AllViewParas, vector<int>&AvailViews, Point3d *All3D, int npts)
{
	char Fname[200];
	int viewID;

	AvailViews.clear();
	sprintf(Fname, "%s/Dinfo.txt", path);
	FILE *fp = fopen(Fname, "r");
	while (fscanf(fp, "%d: ", &viewID) != EOF)
	{
		AvailViews.push_back(viewID);
		for (int jj = 0; jj < 5; jj++)
			fscanf(fp, "%lf ", &AllViewParas[viewID].intrinsic[jj]);
		for (int jj = 0; jj < 7; jj++)
			fscanf(fp, "%lf ", &AllViewParas[viewID].distortion[jj]);
		for (int jj = 0; jj < 6; jj++)
			fscanf(fp, "%lf ", &AllViewParas[viewID].rt[jj]);
	}
	fclose(fp);
	sort(AvailViews.begin(), AvailViews.end());

	GetKFromIntrinsic(AllViewParas, AvailViews);
	GetRTFromrt(AllViewParas, AvailViews);

	if (All3D != NULL)
	{
		sprintf(Fname, "%s/3d.xyz", path);
		fp = fopen(Fname, "r");
		for (int ii = 0; ii < npts; ii++)
			fscanf(fp, "%lf %lf %lf ", &All3D[ii].x, &All3D[ii].y, &All3D[ii].z);
		fclose(fp);
	}

	return;
}
int ReadCumulativePoints(char *Path, int nviews, int timeID, vector<int>&cumulativePts)
{
	int ii, jj;
	char Fname[200];
	if (timeID < 0)
		sprintf(Fname, "%s/CumlativePoints.txt", Path);
	else
		sprintf(Fname, "%s/CumlativePoints_%d.txt", Path, timeID);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s. Abort program!\n", Fname);
		return 1;
	}
	for (ii = 0; ii < nviews + 1; ii++)
	{
		fscanf(fp, "%d\n", &jj);
		cumulativePts.push_back(jj);
	}
	fclose(fp);

	return 0;
}
void ReadPointCorrespondences(char *Path, int nviews, int timeID, vector<int> *PointCorres, vector<int>&CeresDuplicateAddInMask, int totalPts, bool Merge)
{
	int ii, jj, kk, match;
	char Fname[200];

	for (ii = 0; ii < totalPts; ii++)
		PointCorres[ii].reserve(nviews * 2);

	if (!Merge)
	{
		if (timeID < 0)
			sprintf(Fname, "%s/PM.txt", Path);
		else
			sprintf(Fname, "%s/PM_%ds.txt", Path, timeID);
	}
	else
		if (timeID < 0)
			sprintf(Fname, "%s/MPM.txt", Path);
		else
			sprintf(Fname, "%s/MPM_%ds.txt", Path, timeID);

	CeresDuplicateAddInMask.reserve(totalPts * 30);
	FILE *fp = fopen(Fname, "r");
	for (jj = 0; jj < totalPts; jj++)
	{
		fscanf(fp, "%d ", &kk);
		for (ii = 0; ii < kk; ii++)
		{
			fscanf(fp, "%d ", &match);
			PointCorres[jj].push_back(match);
			CeresDuplicateAddInMask.push_back(match);
		}
	}
	return;
}
void ReadPointCorrespondences(char *Path, int nviews, int timeID, vector<int> *PointCorres, int totalPts, bool Merge)
{
	int ii, jj, kk, match;
	char Fname[200];

	for (ii = 0; ii < totalPts; ii++)
		PointCorres[ii].reserve(nviews * 2);

	if (!Merge)
	{
		if (timeID < 0)
			sprintf(Fname, "%s/notMergePM.txt", Path);
		else
			sprintf(Fname, "%s/notMergePM_%d.txt", Path, timeID);
	}
	else
		if (timeID < 0)
			sprintf(Fname, "%s/MPM.txt", Path);
		else
			sprintf(Fname, "%s/MPM_%d.txt", Path, timeID);

	FILE *fp = fopen(Fname, "r");
	for (jj = 0; jj < totalPts; jj++)
	{
		fscanf(fp, "%d ", &kk);
		for (ii = 0; ii < kk; ii++)
		{
			fscanf(fp, "%d ", &match);
			PointCorres[jj].push_back(match);
		}
	}
	return;
}
void GenerateMergePointCorrespondences(vector<int> *MergePointCorres, vector<int> *PointCorres, int totalPts)
{
	//Merging
	for (int kk = 0; kk < totalPts; kk++)
	{
		int nmatches = PointCorres[kk].size();
		if (nmatches > 0) //if that point has matches
		{
			for (int jj = 0; jj < kk; jj++) //look back to previous point
			{
				for (int ii = 0; ii < PointCorres[jj].size(); ii++) //look into all of that previous point matches
				{
					if (PointCorres[jj].at(ii) == kk) //if it has the same ID as the current point-->merge points
					{
						//printf("Merging %d (%d matches) to %d (%d matches)\n", kk, PointCorres[kk].size(), jj, PointCorres[jj].size());
						for (int i = 0; i < PointCorres[kk].size(); i++)
							PointCorres[jj].push_back(PointCorres[kk].at(i));
						PointCorres[kk].clear();//earse matches of the currrent point
						break;
					}
				}
			}
		}
	}

	//Removing duplicated points and sort them
	for (int kk = 0; kk < totalPts; kk++)
	{
		std::sort(PointCorres[kk].begin(), PointCorres[kk].end());
		for (int jj = 0; jj < PointCorres[kk].size(); jj++)
		{
			if (jj == 0)
				MergePointCorres[kk].push_back(PointCorres[kk].at(0));
			else if (jj> 0 && PointCorres[kk].at(jj) != PointCorres[kk].at(jj - 1))
				MergePointCorres[kk].push_back(PointCorres[kk].at(jj));
		}
	}
	return;
}
void GenerateViewandPointCorrespondences(vector<int> *ViewCorres, vector<int> *PointIDCorres, vector<int> *PointCorres, vector<int> CumIDView, int totalPts)
{
	int viewID, PointID, curPID;
	for (int jj = 0; jj < totalPts; jj++)
	{
		for (int ii = 0; ii < PointCorres[jj].size(); ii++)
		{
			curPID = PointCorres[jj].at(ii);
			for (int j = 0; j < CumIDView.size() - 1; j++)
			{
				if (curPID >= CumIDView.at(j) && curPID < CumIDView.at(j + 1))
				{
					viewID = j;
					PointID = curPID - CumIDView.at(j);
					break;
				}
			}
			ViewCorres[jj].push_back(viewID);
			PointIDCorres[jj].push_back(PointID);
		}
	}

	return;
}
void Save3DPoints(char *Path, Point3d *All3D, vector<int>Selected3DIndex)
{
	char Fname[200];
	sprintf(Fname, "%s/3D.xyz", Path);
	FILE *fp = fopen(Fname, "w+");
	for (int ii = 0; ii < Selected3DIndex.size(); ii++)
	{
		int pID = Selected3DIndex.at(ii);
		fprintf(fp, "%.3f %.3f %.3f\n", All3D[pID].x, All3D[pID].y, All3D[pID].z);
	}
	fclose(fp);
}
void DisplayMatrix(char *Fname, Mat m)
{
	printf("%s: ", Fname), cout << m << endl;
}

void convertRvectoRmat(double *r, double *R)
{
	Mat Rmat(3, 3, CV_64F), rvec(3, 1, CV_64F);
	for (int jj = 0; jj < 3; jj++)
		rvec.at<double>(jj) = r[jj];

	Rodrigues(rvec, Rmat);

	for (int jj = 0; jj < 9; jj++)
		R[jj] = Rmat.at<double>(jj);

	return;
}
void GetIntrinsicFromK(CameraData *AllViewsParas, vector<int> AvailViews)
{
	for (int ii = 0; ii < AvailViews.size(); ii++)
	{
		int viewID = AvailViews.at(ii);
		AllViewsParas[viewID].intrinsic[0] = AllViewsParas[viewID].K[0];
		AllViewsParas[viewID].intrinsic[1] = AllViewsParas[viewID].K[4];
		AllViewsParas[viewID].intrinsic[2] = AllViewsParas[viewID].K[1];
		AllViewsParas[viewID].intrinsic[3] = AllViewsParas[viewID].K[2];
		AllViewsParas[viewID].intrinsic[4] = AllViewsParas[viewID].K[5];
	}
	return;
}
void GetIntrinsicFromK(CameraData *AllViewsParas, int nviews)
{
	for (int viewID = 0; viewID < nviews; viewID++)
	{
		AllViewsParas[viewID].intrinsic[0] = AllViewsParas[viewID].K[0];
		AllViewsParas[viewID].intrinsic[1] = AllViewsParas[viewID].K[4];
		AllViewsParas[viewID].intrinsic[2] = AllViewsParas[viewID].K[1];
		AllViewsParas[viewID].intrinsic[3] = AllViewsParas[viewID].K[2];
		AllViewsParas[viewID].intrinsic[4] = AllViewsParas[viewID].K[5];
	}
	return;
}
void GetKFromIntrinsic(CameraData *AllViewsParas, vector<int> AvailViews)
{
	for (int ii = 0; ii < AvailViews.size(); ii++)
	{
		int viewID = AvailViews.at(ii);
		AllViewsParas[viewID].K[0] = AllViewsParas[viewID].intrinsic[0];
		AllViewsParas[viewID].K[4] = AllViewsParas[viewID].intrinsic[1];
		AllViewsParas[viewID].K[1] = AllViewsParas[viewID].intrinsic[2];
		AllViewsParas[viewID].K[2] = AllViewsParas[viewID].intrinsic[3];
		AllViewsParas[viewID].K[5] = AllViewsParas[viewID].intrinsic[4];
	}
	return;
}
void GetKFromIntrinsic(CameraData *AllViewsParas, int nviews)
{
	for (int viewID = 0; viewID < nviews; viewID++)
	{
		AllViewsParas[viewID].K[0] = AllViewsParas[viewID].intrinsic[0];
		AllViewsParas[viewID].K[4] = AllViewsParas[viewID].intrinsic[1];
		AllViewsParas[viewID].K[1] = AllViewsParas[viewID].intrinsic[2];
		AllViewsParas[viewID].K[2] = AllViewsParas[viewID].intrinsic[3];
		AllViewsParas[viewID].K[5] = AllViewsParas[viewID].intrinsic[4];
	}
	return;
}
void GetIntrinsicFromK(CameraData &camera)
{
	camera.intrinsic[0] = camera.K[0];
	camera.intrinsic[1] = camera.K[4];
	camera.intrinsic[2] = camera.K[1];
	camera.intrinsic[3] = camera.K[2];
	camera.intrinsic[4] = camera.K[5];
	return;
}
void GetKFromIntrinsic(CameraData &camera)
{
	camera.K[0] = camera.intrinsic[0], camera.K[1] = camera.intrinsic[2], camera.K[2] = camera.intrinsic[3];
	camera.K[3] = 0.0, camera.K[4] = camera.intrinsic[1], camera.K[5] = camera.intrinsic[4];
	camera.K[6] = 0.0, camera.K[7] = 0.0, camera.K[8] = 1.0;
	return;
}
void GetrtFromRT(CameraData *AllViewsParas, vector<int> AvailViews)
{
	Mat R(3, 3, CV_64F), r(3, 1, CV_64F);

	for (int ii = 0; ii < AvailViews.size(); ii++)
	{
		int viewID = AvailViews.at(ii);
		for (int jj = 0; jj < 9; jj++)
			R.at<double>(jj) = AllViewsParas[viewID].R[jj];

		Rodrigues(R, r);

		for (int jj = 0; jj < 3; jj++)
			AllViewsParas[viewID].rt[jj] = r.at<double>(jj), AllViewsParas[viewID].rt[3 + jj] = AllViewsParas[viewID].T[jj];
	}
}
void GetrtFromRT(CameraData *AllViewsParas, int nviews)
{
	Mat R(3, 3, CV_64F), r(3, 1, CV_64F);

	for (int viewID = 0; viewID < nviews; viewID++)
	{
		for (int jj = 0; jj < 9; jj++)
			R.at<double>(jj) = AllViewsParas[viewID].R[jj];

		Rodrigues(R, r);

		for (int jj = 0; jj < 3; jj++)
			AllViewsParas[viewID].rt[jj] = r.at<double>(jj), AllViewsParas[viewID].rt[3 + jj] = AllViewsParas[viewID].T[jj];
	}
}
void GetrtFromRT(double *rt, double *R, double *T)
{
	Mat Rmat(3, 3, CV_64F), r(3, 1, CV_64F);

	for (int jj = 0; jj < 9; jj++)
		Rmat.at<double>(jj) = R[jj];

	Rodrigues(Rmat, r);

	for (int jj = 0; jj < 3; jj++)
		rt[jj] = r.at<double>(jj), rt[3 + jj] = T[jj];

	return;
}

void GetRTFromrt(CameraData &camera)
{
	Mat Rmat(3, 3, CV_64F), rvec(3, 1, CV_64F);
	for (int jj = 0; jj < 3; jj++)
		rvec.at<double>(jj) = camera.rt[jj];

	Rodrigues(rvec, Rmat);

	for (int jj = 0; jj < 9; jj++)
		camera.R[jj] = Rmat.at<double>(jj);
	for (int jj = 0; jj < 3; jj++)
		camera.T[jj] = camera.rt[jj + 3];

	return;
}
void GetRTFromrt(double *rt, double *R, double *T)
{
	Mat Rmat(3, 3, CV_64F), rvec(3, 1, CV_64F);
	for (int jj = 0; jj < 3; jj++)
		rvec.at<double>(jj) = rt[jj];

	Rodrigues(rvec, Rmat);

	for (int jj = 0; jj < 9; jj++)
		R[jj] = Rmat.at<double>(jj);
	for (int jj = 0; jj < 3; jj++)
		T[jj] = rt[jj + 3];

	return;
}
void GetRTFromrt(CameraData *AllViewsParas, vector<int> AvailViews)
{
	Mat R(3, 3, CV_64F), r(3, 1, CV_64F);

	for (int ii = 0; ii < AvailViews.size(); ii++)
	{
		int viewID = AvailViews.at(ii);
		for (int jj = 0; jj < 3; jj++)
			r.at<double>(jj) = AllViewsParas[viewID].rt[jj];

		Rodrigues(r, R);

		for (int jj = 0; jj < 9; jj++)
			AllViewsParas[viewID].R[jj] = R.at<double>(jj);
		for (int jj = 0; jj < 3; jj++)
			AllViewsParas[viewID].T[jj] = AllViewsParas[viewID].rt[jj + 3];
	}

	return;
}
void GetRTFromrt(CameraData *AllViewsParas, int nviews)
{
	Mat R(3, 3, CV_64F), r(3, 1, CV_64F);

	for (int viewID = 0; viewID < nviews; viewID++)
	{
		for (int jj = 0; jj < 3; jj++)
			r.at<double>(jj) = AllViewsParas[viewID].rt[jj];

		Rodrigues(r, R);

		for (int jj = 0; jj < 9; jj++)
			AllViewsParas[viewID].R[jj] = R.at<double>(jj);
		for (int jj = 0; jj < 3; jj++)
			AllViewsParas[viewID].T[jj] = AllViewsParas[viewID].rt[jj + 3];
	}

	return;
}
void AssembleRT(double *R, double *T, double *RT)
{
	RT[0] = R[0], RT[1] = R[1], RT[2] = R[2], RT[3] = T[0];
	RT[4] = R[3], RT[5] = R[4], RT[6] = R[5], RT[7] = T[1];
	RT[8] = R[6], RT[9] = R[7], RT[10] = R[8], RT[11] = T[2];
}
void DesembleRT(double *R, double *T, double *RT)
{
	R[0] = RT[0], R[1] = RT[1], R[2] = RT[2], T[0] = RT[3];
	R[3] = RT[4], R[4] = RT[5], R[5] = RT[6], T[1] = RT[7];
	R[6] = RT[8], R[7] = RT[9], R[8] = RT[10], T[2] = RT[11];
}
void AssembleP(CameraData &camera)
{
	double RT[12];
	Set_Sub_Mat(camera.R, RT, 3, 3, 4, 0, 0);
	Set_Sub_Mat(camera.T, RT, 1, 3, 4, 3, 0);
	mat_mul(camera.K, RT, camera.P, 3, 3, 4);
	return;
}
void AssembleP(double *K, double *R, double *T, double *P)
{
	double RT[12];
	Set_Sub_Mat(R, RT, 3, 3, 4, 0, 0);
	Set_Sub_Mat(T, RT, 1, 3, 4, 3, 0);
	mat_mul(K, RT, P, 3, 3, 4);
	return;
}


int clickCount = 0;
Point2i ClickPos[3];
void ClickLocation(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
		ClickPos[clickCount].x = x, ClickPos[clickCount].y = y;
		clickCount++;
	}
	if (clickCount > 3)
		waitKey(-27);
}
double DistanceOfTwoPointsSfM(char *Path, int id1, int id2, int id3)
{
	printf("Reading Corpus and camera info");
	char Fname[200];

	Corpus corpusData;
	sprintf(Fname, "%s/BA_Camera_AllParams_after.txt", Path);
	if (!loadBundleAdjustedNVMResults(Fname, corpusData))
		return 1;

	int nviews = corpusData.nCamera;
	for (int ii = 0; ii < nviews; ii++)
	{
		corpusData.camera[ii].threshold = 1.5, corpusData.camera[ii].ninlierThresh = 50, corpusData.camera[ii];
		GetrtFromRT(corpusData.camera[ii].rt, corpusData.camera[ii].R, corpusData.camera[ii].T);
		GetIntrinsicFromK(corpusData.camera[ii]);
		AssembleP(corpusData.camera[ii].K, corpusData.camera[ii].R, corpusData.camera[ii].T, corpusData.camera[ii].P);
	}
	printf("...Done\n");


	namedWindow("Image", CV_WINDOW_NORMAL);
	setMouseCallback("Image", ClickLocation, NULL);
	printf("Left button to click, ESC  to stop\n");

	clickCount = 0;
	sprintf(Fname, "%s/%d.png", Path, id1);
	Mat img = imread(Fname, 1);
	if (img.empty())
		printf("Cannot load %s\n", Fname);
	imshow("Image", img);
	waitKey(0);
	Point2d pos1[2] = { Point2d(ClickPos[0].x, ClickPos[0].y), Point2d(ClickPos[1].x, ClickPos[1].y) };

	clickCount = 0;
	sprintf(Fname, "%s/%d.png", Path, id2);
	img = imread(Fname, 1);
	if (img.empty())
		printf("Cannot load %s\n", Fname);
	imshow("Image", img);
	waitKey(0);
	Point2d pos2[2] = { Point2d(ClickPos[0].x, ClickPos[0].y), Point2d(ClickPos[1].x, ClickPos[1].y) };

	clickCount = 0;
	sprintf(Fname, "%s/%d.png", Path, id3);
	img = imread(Fname, 1);
	if (img.empty())
		printf("Cannot load %s\n", Fname);
	imshow("Image", img);
	waitKey(0);
	Point2d pos3[2] = { Point2d(ClickPos[0].x, ClickPos[0].y), Point2d(ClickPos[1].x, ClickPos[1].y) };

	Point3d WC[2];
	Point2d allPts[2 * 3] = { pos1[0], pos2[0], pos3[0], pos1[1], pos2[1], pos3[1] };
	bool passedPoints[3];
	int allId[3] = { id1, id2, id3 };
	double allP[12 * 3], allK[9 * 3], allDistortion[7 * 3];
	for (int ii = 0; ii < 3; ii++)
	{
		for (int jj = 0; jj < 12; jj++)
			allP[12 * ii + jj] = corpusData.camera[allId[ii]].P[jj];
		for (int jj = 0; jj < 9; jj++)
			allK[9 * ii + jj] = corpusData.camera[allId[ii]].K[jj];
		for (int jj = 0; jj < 7; jj++)
			allDistortion[7 * ii + jj] = corpusData.camera[allId[ii]].distortion[jj];
	}

	MultiViewQualityCheck(allPts, allP, corpusData.camera[0].LensModel, allK, allDistortion, passedPoints, 3, 2, 3.0, WC);

	return sqrt(pow(WC[0].x - WC[1].x, 2) + pow(WC[0].y - WC[1].y, 2) + pow(WC[0].z - WC[1].z, 2));
}


void BlurDetectionDriver(char *Path, int nimages, int width, int height, float blurThresh)
{
	char Fname[200];
	Mat cvImg;
	unsigned char *Img = new unsigned char[width*height];
	vector<int>blurredImgVector; blurredImgVector.reserve(nimages);

	for (int kk = 0; kk < nimages; kk++)
	{
		sprintf(Fname, "%s/_%d.png", Path, kk + 1);
		cvImg = imread(Fname, 0);
		if (cvImg.empty())
		{
			printf("Cannot load %s\n", Fname);
			continue;
		}
		for (int jj = 0; jj < cvImg.rows; jj++)
			for (int ii = 0; ii < cvImg.cols; ii++)
				Img[ii + jj*cvImg.cols] = cvImg.data[ii + jj*cvImg.cols];

		float blur, extent;
		int blurred = IsBlurred(Img, cvImg.cols, cvImg.rows, blur, extent, blurThresh);
		printf("@frame %d: blur coeff: %.3f\n", kk + 1, blur);
		if (blurred)
		{
			sprintf(Fname, "%s/ (%d).png", Path, kk + 1);
			cvImg = imread(Fname, 1);
			sprintf(Fname, "%s/B(%d).png", Path, kk + 1);
			imwrite(Fname, cvImg);
		}
		else
		{
			sprintf(Fname, "%s/B%d.png", Path, kk + 1);
			cvImg = imread(Fname, 1);
			sprintf(Fname, "%s/%d.png", Path, kk + 1);
			imwrite(Fname, cvImg);
		}
		blurredImgVector.push_back(blurred);
	}

	delete[]Img;
	return;
}

// <Path>  : Where nvm file is located. All results are output here.
// <nvmName>: result from Visual SFM
// <camInfo>: filename width height [model of intrinsic parameters] [ model of lends distortion] [model of extrinsic parameters] availability
// <IntrinsicInfo>: (optional)  filename fx fy skew u0 v0 radiral(1,2,3) tangential(1,2) prism(1,2)
// lensconversion: convert from visSfm to Opencv format
// sharedIntrinsics: 0 (not share), 1 (shared)
int BAVisualSfMDriver(char *Path, char *nvmName, char *camInfo, char *IntrinsicInfo, bool lensconversion, int sharedIntrinsics)
{
	//Path = "C:/temp/X",
	//	nvmName = "fountain.nvm",
	//	camInfo = "camInfo.txt",
	//	IntrinsicInfo = "Intrinsics.txt";

	string rootfolder(Path);

	//---------load an NVM file--------------
	cout << "Loading the NVM file...";
	string nvmfile = nvmName;
	string nvmpath = rootfolder + "/" + nvmfile;

	BA::NVM    nvmdata;
	if (!BA::loadNVM(nvmpath, nvmdata))
		return 1;
	cout << "Done." << endl;
	//-------------------------------------

	//---------convert NVM data to CameraData---------
	cout << "Converting NVM data to CameraData...";
	string imginfofile = rootfolder + "/" + camInfo;
	vector<BA::CameraData> camera_before;
	if (!BA::initCameraData(nvmdata, imginfofile, camera_before, sharedIntrinsics))
		return 1;

	cout << "Done." << endl;
	//-------------------------------------


	//---------load initial parameters if applicable---------
	if (IntrinsicInfo != NULL)
	{
		cout << "Loading initial intrinsic parameters...";
		string intrinsicfile = rootfolder + "/" + IntrinsicInfo;
		if (!BA::loadInitialIntrinsics(intrinsicfile, nvmdata.filename_id, camera_before, sharedIntrinsics))
			return 1;

		cout << "Done." << endl;
	}
	//-------------------------------------

	//---------perform bundle adjustment---------
	vector< vector<double> > xyz_before(nvmdata.xyz);
	vector< vector<double> > xyz_after(nvmdata.xyz);
	vector<BA::CameraData> camera_after(camera_before);

	if (lensconversion)
	{
		cout << "\n" << "Converting Lens Model...";
		convertLendsModel(camera_before, xyz_before);
		cout << "Done." << endl;
	}


	//---------reprojection error before BA---------
	cout << "Calculating reprojection erros (before) and Saving the result...";
	BA::residualData resdata_before;
	BA::calcReprojectionError(camera_before, xyz_before, resdata_before);
	BA::saveAllData(rootfolder, camera_before, xyz_before, resdata_before, "BA_", false);

	double mean_err_before[2] = { resdata_before.mean_abs_error[0], resdata_before.mean_abs_error[1] };

	ceres::Solver::Summary summary;
	ceres::Solver::Options options;
	BA::setCeresOption(nvmdata, options);
	//options.use_nonmonotonic_steps = false;


	cout << "\n" << "Run Bundle Adjustment..." << endl;
	cout << "\t# of Cameras: " << camera_before.size() << "\n" << "\t# of Points : " << nvmdata.n3dPoints << "\n" << endl;

	double thresh = lensconversion ? 10.0 : -1.0; //if minus value, all points from NVM are regarded as inliers.
	//double thresh = -1.0;

	BA::runBundleAdjustment(camera_after, xyz_after, options, summary, thresh);
	cout << summary.FullReport() << "\n";

	string ceres_report = rootfolder + "/BA_CeresReport.txt";
	ofstream ofs(ceres_report);
	if (ofs.fail())
		cerr << "Cannot write " << ceres_report << endl;
	else
		ofs << summary.FullReport();
	ofs.close();
	//-------------------------------------


	//--------save as NVM format------------------
	cout << "\n";
	cout << "Saving the result as NVM format...";
	BA::saveNVM(rootfolder, nvmfile, camera_after, xyz_after, nvmdata);
	cout << "Done." << endl;
	//-------------------------------------


	//---------reprojection error after BA---------
	cout << "Calculating reprojection erros ( after) and Saving the result...";
	BA::residualData resdata_after;
	BA::calcReprojectionError(camera_after, xyz_after, resdata_after);
	BA::saveAllData(rootfolder, camera_after, xyz_after, resdata_after, "BA_", true);
	double mean_err_after[2] = { resdata_after.mean_abs_error[0], resdata_after.mean_abs_error[1] };

	cout << "Done." << endl;
	//-------------------------------------


	cout << "Mean Absolute Reprojection Errors (x,y)\n" << fixed << setprecision(3)
		<< "\t" << "before: (" << mean_err_before[0] << ", " << mean_err_before[1] << ")\n"
		<< "\t" << "after : (" << mean_err_after[0] << ", " << mean_err_after[1] << ")" << endl;

	cout << "X" << endl;
	return 0;
}

bool loadNVMLite(const string filepath, Corpus &CorpusData, int sharedIntrinsics)
{
	ifstream ifs(filepath);
	if (ifs.fail())
	{
		cerr << "Cannot load " << filepath << endl;
		return false;
	}

	string token;
	ifs >> token; //NVM_V3
	if (token != "NVM_V3")
	{
		cerr << "Can only load NVM_V3" << endl;
		return false;
	}

	if (sharedIntrinsics == 1)
	{
		double fx, fy, u0, v0, radial1;
		ifs >> token >> fx >> u0 >> fy >> v0 >> radial1;
	}

	//loading camera parameters
	int nviews;
	ifs >> nviews;
	if (nviews <= 1)
	{
		cerr << "# of cameras must be more than 1." << endl;
		return false;
	}
	CorpusData.nCamera = nviews;
	CorpusData.camera = new CameraData[nviews];
	double Quaterunion[4], CamCenter[3], T[3];
	for (int ii = 0; ii < nviews; ii++)
	{
		string filename;
		double f;
		vector<double> q(4), c(3), d(2);
		ifs >> filename >> f >> Quaterunion[0] >> Quaterunion[1] >> Quaterunion[2] >> Quaterunion[3] >> CamCenter[0] >> CamCenter[1] >> CamCenter[2] >> d[0] >> d[1];

		std::size_t pos = filename.find(".ppm");
		filename.erase(pos, 4);
		const char * str = filename.c_str();
		int viewID = atoi(str);

		ceres::QuaternionToRotation(Quaterunion, CorpusData.camera[viewID].R);
		mat_mul(CorpusData.camera[viewID].R, CamCenter, T, 3, 3, 1); //t = -RC
		CorpusData.camera[viewID].T[0] = -T[0], CorpusData.camera[viewID].T[1] = -T[1], CorpusData.camera[viewID].T[2] = -T[2];
	}

	return true;
}
bool loadBundleAdjustedNVMResults(char *BAfileName, Corpus &CorpusData)
{
	FILE *fp = fopen(BAfileName, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n");
		return false;
	}

	char Fname[200];
	int lensType, width, height;
	double fx, fy, skew, u0, v0, r1, r2, r3, t1, t2, p1, p2, omega, DistCtrX, DistCtrY, rv[3], T[3];

	fscanf(fp, "%d ", &CorpusData.nCamera);
	CorpusData.camera = new CameraData[CorpusData.nCamera];

	for (int ii = 0; ii < CorpusData.nCamera; ii++)
	{
		if (fscanf(fp, "%s %d %d %d", &Fname, &lensType, &width, &height) == EOF)
			break;
		string filename = Fname;
		std::size_t pos = filename.find(".ppm");
		filename.erase(pos, 4);
		const char * str = filename.c_str();
		int viewID = atoi(str);

		CorpusData.camera[viewID].LensModel = lensType;
		CorpusData.camera[viewID].width = width, CorpusData.camera[viewID].height = height;
		if (lensType == RADIAL_TANGENTIAL_PRISM)
		{
			fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf ", &fx, &fy, &skew, &u0, &v0,
				&r1, &r2, &r3, &t1, &t2, &p1, &p2,
				&rv[0], &rv[1], &rv[2], &T[0], &T[1], &T[2]);

			CorpusData.camera[viewID].distortion[0] = r1,
				CorpusData.camera[viewID].distortion[1] = r2,
				CorpusData.camera[viewID].distortion[2] = r3,
				CorpusData.camera[viewID].distortion[3] = t1,
				CorpusData.camera[viewID].distortion[4] = t2,
				CorpusData.camera[viewID].distortion[5] = p1,
				CorpusData.camera[viewID].distortion[6] = p2;
		}
		else
		{
			fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf ", &fx, &fy, &skew, &u0, &v0,
				&omega, &DistCtrX, &DistCtrY,
				&rv[0], &rv[1], &rv[2], &T[0], &T[1], &T[2]);

			CorpusData.camera[viewID].distortion[0] = omega,
				CorpusData.camera[viewID].distortion[1] = DistCtrX,
				CorpusData.camera[viewID].distortion[2] = DistCtrY;
			for (int jj = 3; jj < 7; jj++)
				CorpusData.camera[viewID].distortion[jj] = 0;
		}

		CorpusData.camera[viewID].intrinsic[0] = fx,
			CorpusData.camera[viewID].intrinsic[1] = fy,
			CorpusData.camera[viewID].intrinsic[2] = skew,
			CorpusData.camera[viewID].intrinsic[3] = u0,
			CorpusData.camera[viewID].intrinsic[4] = v0;

		for (int jj = 0; jj < 3; jj++)
		{
			CorpusData.camera[viewID].rt[jj] = rv[jj];
			CorpusData.camera[viewID].rt[jj + 3] = T[jj];
		}

		GetKFromIntrinsic(CorpusData.camera[viewID]);
		GetRTFromrt(CorpusData.camera[viewID].rt, CorpusData.camera[viewID].R, CorpusData.camera[viewID].T);
	}
	return true;
}
bool rewriteBundleAdjustedNVMResults(char *Path, char *BAfileName, Corpus &CorpusData)
{
	FILE *fp = fopen(BAfileName, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n");
		return false;
	}

	char Fname[200], iFname[200];
	int lensType, width, height;
	double fx, fy, skew, u0, v0, r1, r2, r3, t1, t2, p1, p2, omega, DistCtrX, DistCtrY, rv[3], T[3];

	fscanf(fp, "%d ", &CorpusData.nCamera);
	CorpusData.camera = new CameraData[CorpusData.nCamera];

	Mat Img;
	for (int viewID = 0; viewID < CorpusData.nCamera; viewID++)
	{
		if (fscanf(fp, "%s %d %d %d", &iFname, &lensType, &width, &height) == EOF)
			break;

		/*sprintf(Fname, "%s/%s", Path, iFname);
		Img = imread(Fname, 1);
		if (Img.empty())
		{
		printf("Cannot load %s\n", Fname);
		abort();
		}
		sprintf(Fname, "%s/%d.png", Path, viewID); imwrite(Fname, Img);*/

		CorpusData.camera[viewID].LensModel = lensType;
		CorpusData.camera[viewID].width = width, CorpusData.camera[viewID].height = height;
		if (lensType == RADIAL_TANGENTIAL_PRISM)
		{
			fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf ", &fx, &fy, &skew, &u0, &v0,
				&r1, &r2, &r3, &t1, &t2, &p1, &p2,
				&rv[0], &rv[1], &rv[2], &T[0], &T[1], &T[2]);

			CorpusData.camera[viewID].distortion[0] = r1, CorpusData.camera[viewID].distortion[1] = r2, CorpusData.camera[viewID].distortion[2] = r3,
				CorpusData.camera[viewID].distortion[3] = t1, CorpusData.camera[viewID].distortion[4] = t2,
				CorpusData.camera[viewID].distortion[5] = p1, CorpusData.camera[viewID].distortion[6] = p2;
		}
		else
		{
			fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf ", &fx, &fy, &skew, &u0, &v0,
				&omega, &DistCtrX, &DistCtrY,
				&rv[0], &rv[1], &rv[2], &T[0], &T[1], &T[2]);
			CorpusData.camera[viewID].distortion[0] = omega, CorpusData.camera[viewID].distortion[1] = DistCtrX, CorpusData.camera[viewID].distortion[2] = DistCtrY;
			for (int jj = 3; jj < 7; jj++)
				CorpusData.camera[viewID].distortion[jj] = 0;
		}

		CorpusData.camera[viewID].intrinsic[0] = fx,
			CorpusData.camera[viewID].intrinsic[1] = fy,
			CorpusData.camera[viewID].intrinsic[2] = skew,
			CorpusData.camera[viewID].intrinsic[3] = u0,
			CorpusData.camera[viewID].intrinsic[4] = v0;

		for (int jj = 0; jj < 3; jj++)
		{
			CorpusData.camera[viewID].rt[jj] = rv[jj];
			CorpusData.camera[viewID].rt[jj + 3] = T[jj];
		}

		GetKFromIntrinsic(CorpusData.camera[viewID]);
		GetRTFromrt(CorpusData.camera[viewID].rt, CorpusData.camera[viewID].R, CorpusData.camera[viewID].T);
	}

	fp = fopen(BAfileName, "w+");
	fprintf(fp, "%d \n", CorpusData.nCamera);

	for (int viewID = 0; viewID < CorpusData.nCamera; viewID++)
	{
		fprintf(fp, "%d.png %d %d %d ", viewID, CorpusData.camera[viewID].LensModel, CorpusData.camera[viewID].width, CorpusData.camera[viewID].height);

		fx = CorpusData.camera[viewID].intrinsic[0], fy = CorpusData.camera[viewID].intrinsic[1],
			skew = CorpusData.camera[viewID].intrinsic[2],
			u0 = CorpusData.camera[viewID].intrinsic[3], v0 = CorpusData.camera[viewID].intrinsic[4];

		for (int jj = 0; jj < 3; jj++)
			rv[jj] = CorpusData.camera[viewID].rt[jj], T[jj] = CorpusData.camera[viewID].rt[jj + 3];

		if (CorpusData.camera[viewID].LensModel == RADIAL_TANGENTIAL_PRISM)
		{
			r1 = CorpusData.camera[viewID].distortion[0], r2 = CorpusData.camera[viewID].distortion[1], r3 = CorpusData.camera[viewID].distortion[2],
				t1 = CorpusData.camera[viewID].distortion[3], t2 = CorpusData.camera[viewID].distortion[4],
				p1 = CorpusData.camera[viewID].distortion[5], p2 = CorpusData.camera[viewID].distortion[6];
			fprintf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf \n", fx, fy, skew, u0, v0,
				r1, r2, r3, t1, t2, p1, p2,
				rv[0], rv[1], rv[2], T[0], T[1], T[2]);
		}
		else
		{
			omega = CorpusData.camera[viewID].distortion[0], DistCtrX = CorpusData.camera[viewID].distortion[1], DistCtrY = CorpusData.camera[viewID].distortion[2];
			fprintf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf \n", fx, fy, skew, u0, v0,
				omega, DistCtrX, DistCtrY,
				rv[0], rv[1], rv[2], T[0], T[1], T[2]);
		}
	}
	return true;
}
int SaveCorpusInfo(char *Path, Corpus &CorpusData, bool notbinary)
{
	int ii, jj, kk;
	char Fname[200];
	sprintf(Fname, "%s/Corpus_3D.txt", Path);	FILE *fp = fopen(Fname, "w+");
	CorpusData.n3dPoints = CorpusData.xyz.size();
	fprintf(fp, "%d %d ", CorpusData.nCamera, CorpusData.n3dPoints);

	//xyz rgb viewid3D pointid3D 3dId2D cumpoint
	if (CorpusData.rgb.size() == 0)
	{
		fprintf(fp, "0\n");
		for (jj = 0; jj < CorpusData.xyz.size(); jj++)
			fprintf(fp, "%lf %lf %lf \n", CorpusData.xyz.at(jj).x, CorpusData.xyz.at(jj).y, CorpusData.xyz.at(jj).z);
	}
	else
	{
		fprintf(fp, "1\n");
		for (jj = 0; jj < CorpusData.xyz.size(); jj++)
			fprintf(fp, "%lf %lf %lf %d %d %d\n", CorpusData.xyz.at(jj).x, CorpusData.xyz.at(jj).y, CorpusData.xyz.at(jj).z, CorpusData.rgb.at(jj).x, CorpusData.rgb.at(jj).y, CorpusData.rgb.at(jj).z);
	}
	fclose(fp);

	sprintf(Fname, "%s/Corpus_viewIdAll3D.txt", Path); fp = fopen(Fname, "w+");
	for (jj = 0; jj < CorpusData.n3dPoints; jj++)
	{
		int nviews = CorpusData.viewIdAll3D.at(jj).size();
		fprintf(fp, "%d ", nviews);
		for (ii = 0; ii < nviews; ii++)
			fprintf(fp, "%d ", CorpusData.viewIdAll3D.at(jj).at(ii));
		fprintf(fp, "\n");
	}
	fclose(fp);

	sprintf(Fname, "%s/Corpus_pointIdAll3D.txt", Path); fp = fopen(Fname, "w+");
	for (jj = 0; jj < CorpusData.n3dPoints; jj++)
	{
		int npts = CorpusData.pointIdAll3D.at(jj).size();
		fprintf(fp, "%d ", npts);
		for (ii = 0; ii < npts; ii++)
			fprintf(fp, "%d ", CorpusData.pointIdAll3D.at(jj).at(ii));
		fprintf(fp, "\n");
	}
	fclose(fp);

	sprintf(Fname, "%s/Corpus_uvAll3D.txt", Path); fp = fopen(Fname, "w+");
	for (jj = 0; jj < CorpusData.n3dPoints; jj++)
	{
		int npts = CorpusData.uvAll3D.at(jj).size();
		fprintf(fp, "%d ", npts);
		for (ii = 0; ii < npts; ii++)
			fprintf(fp, "%lf %lf ", CorpusData.uvAll3D.at(jj).at(ii).x, CorpusData.uvAll3D.at(jj).at(ii).y);
		fprintf(fp, "\n");
	}
	fclose(fp);

	sprintf(Fname, "%s/Corpus_threeDIdAllViews.txt", Path); fp = fopen(Fname, "w+");
	for (jj = 0; jj < CorpusData.nCamera; jj++)
	{
		int n3D = CorpusData.threeDIdAllViews.at(jj).size();
		fprintf(fp, "%d\n", n3D);
		for (ii = 0; ii < n3D; ii++)
			fprintf(fp, "%d ", CorpusData.threeDIdAllViews.at(jj).at(ii));
		fprintf(fp, "\n");
	}
	fclose(fp);

	sprintf(Fname, "%s/Corpus_cum.txt", Path); fp = fopen(Fname, "w+");
	for (int ii = 0; ii < CorpusData.IDCumView.size(); ii++)
		fprintf(fp, "%d ", CorpusData.IDCumView.at(ii));
	fclose(fp);

	for (ii = 0; ii < CorpusData.nCamera; ii++)
	{
		sprintf(Fname, "%s/CorpusK_%d.txt", Path, ii);
		FILE *fp = fopen(Fname, "w+");
		int npts = CorpusData.uvAllViews.at(ii).size();
		for (int jj = 0; jj < npts; jj++)
			fprintf(fp, "%.4f %.4f\n", CorpusData.uvAllViews.at(ii).at(jj).x, CorpusData.uvAllViews.at(ii).at(jj).y);
		fclose(fp);
	}

	sprintf(Fname, "%s/Corpus_Intrinsics.txt", Path); fp = fopen(Fname, "w+");
	for (int viewID = 0; viewID < CorpusData.nCamera; viewID++)
	{
		fprintf(fp, "%d ", CorpusData.camera[viewID].LensModel);
		for (int ii = 0; ii < 5; ii++)
			fprintf(fp, "%lf ", CorpusData.camera[viewID].intrinsic[ii]);

		if (CorpusData.camera[viewID].LensModel == RADIAL_TANGENTIAL_PRISM)
			for (int ii = 0; ii < 7; ii++)
				fprintf(fp, "%lf ", CorpusData.camera[viewID].distortion[ii]);
		else
		{
			for (int ii = 0; ii < 3; ii++)
				fprintf(fp, "%lf ", CorpusData.camera[viewID].distortion[ii]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	if (notbinary)
	{
		for (kk = 0; kk < CorpusData.nCamera; kk++)
		{
			sprintf(Fname, "%s/CorpusD_%d.txt", Path, kk);	fp = fopen(Fname, "w+");
			int npts = CorpusData.threeDIdAllViews.at(kk).size(), curPid = CorpusData.IDCumView.at(kk);
			fprintf(fp, "%d\n", npts);
			for (jj = 0; jj < npts; jj++)
			{
				fprintf(fp, "%d ", CorpusData.threeDIdAllViews.at(kk).at(jj));
				for (ii = 0; ii < SIFTBINS; ii++)
					fprintf(fp, "%.5f ", CorpusData.SiftDesc.at<float>(curPid + jj, ii));
				fprintf(fp, "\n");
			}
			fclose(fp);
		}
	}
	else
	{
		for (kk = 0; kk < CorpusData.nCamera; kk++)
		{
			sprintf(Fname, "%s/CorpusD_%d.txt", Path, kk);
			ofstream fout; fout.open(Fname, ios::binary);

			int npts = CorpusData.threeDIdAllViews.at(kk).size(), curPid = CorpusData.IDCumView.at(kk);
			fout.write(reinterpret_cast<char *>(&npts), sizeof(int));
			for (jj = 0; jj < npts; jj++)
			{
				fout.write(reinterpret_cast<char *>(&CorpusData.threeDIdAllViews.at(kk).at(jj)), sizeof(int));
				for (ii = 0; ii < SIFTBINS; ii++)
					fout.write(reinterpret_cast<char *>(&CorpusData.SiftDesc.at<float>(curPid + jj, ii)), sizeof(float));
			}
			fout.close();
		}
	}

	return 0;
}
int ReadCorpusInfo(char *Path, Corpus &CorpusData, bool notbinary, bool notReadDescriptor)
{
	int ii, jj, kk, nCameras, nPoints, useColor;
	char Fname[200];
	sprintf(Fname, "%s/Corpus_3D.txt", Path); FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return 1;
	}
	fscanf(fp, "%d %d %d", &nCameras, &nPoints, &useColor);
	CorpusData.nCamera = nCameras;
	CorpusData.n3dPoints = nPoints;
	//xyz rgb viewid3D pointid3D 3dId2D cumpoint

	Point3d xyz;
	Point3i rgb;
	CorpusData.xyz.reserve(nPoints);
	if (useColor)
	{
		CorpusData.rgb.reserve(nPoints);
		for (jj = 0; jj < nPoints; jj++)
		{
			fscanf(fp, "%lf %lf %lf %d %d %d", &xyz.x, &xyz.y, &xyz.z, &rgb.x, &rgb.y, &rgb.z);
			CorpusData.xyz.push_back(xyz);
			CorpusData.rgb.push_back(rgb);
		}
	}
	else
	{
		CorpusData.rgb.reserve(nPoints);
		for (jj = 0; jj < nPoints; jj++)
		{
			fscanf(fp, "%lf %lf %lf ", &xyz.x, &xyz.y, &xyz.z);
			CorpusData.xyz.push_back(xyz);
		}
	}

	sprintf(Fname, "%s/Corpus_viewIdAll3D.txt", Path); fp = fopen(Fname, "r");
	int nviews, viewID;
	vector<int>viewIDs; viewIDs.reserve(nCameras / 10);
	for (jj = 0; jj < nPoints; jj++)
	{
		viewIDs.clear();
		fscanf(fp, "%d ", &nviews);
		for (ii = 0; ii < nviews; ii++)
		{
			fscanf(fp, "%d ", &viewID);
			viewIDs.push_back(viewID);
		}
		CorpusData.viewIdAll3D.push_back(viewIDs);
	}
	fclose(fp);

	sprintf(Fname, "%s/Corpus_pointIdAll3D.txt", Path); fp = fopen(Fname, "r");
	int npts, pid;
	vector<int>pointIDs;
	for (jj = 0; jj < nPoints; jj++)
	{
		pointIDs.clear();
		fscanf(fp, "%d ", &npts);
		for (ii = 0; ii < npts; ii++)
		{
			fscanf(fp, "%d ", &pid);
			pointIDs.push_back(pid);
		}
		CorpusData.pointIdAll3D.push_back(pointIDs);
	}
	fclose(fp);

	sprintf(Fname, "%s/Corpus_uvAll3D.txt", Path); fp = fopen(Fname, "r");
	Point2d uv;
	vector<Point2d> uvVector; uvVector.reserve(50);
	for (jj = 0; jj < CorpusData.n3dPoints; jj++)
	{
		uvVector.clear();
		fscanf(fp, "%d ", &npts);
		for (ii = 0; ii < npts; ii++)
		{
			fscanf(fp, "%lf %lf ", &uv.x, &uv.y);
			uvVector.push_back(uv);
		}
		CorpusData.uvAll3D.push_back(uvVector);
	}
	fclose(fp);

	sprintf(Fname, "%s/Corpus_threeDIdAllViews.txt", Path); fp = fopen(Fname, "r");
	int n3D, id3D;
	vector<int>threeDid;
	for (jj = 0; jj < CorpusData.nCamera; jj++)
	{
		threeDid.clear();
		fscanf(fp, "%d ", &n3D);
		for (ii = 0; ii < n3D; ii++)
		{
			fscanf(fp, "%d ", &id3D);
			threeDid.push_back(id3D);
		}
		CorpusData.threeDIdAllViews.push_back(threeDid);
	}
	fclose(fp);

	sprintf(Fname, "%s/Corpus_cum.txt", Path); fp = fopen(Fname, "r");
	int totalPts = 0;
	CorpusData.IDCumView.reserve(nCameras + 1);
	for (kk = 0; kk < nCameras + 1; kk++)
	{
		fscanf(fp, "%d ", &totalPts);
		CorpusData.IDCumView.push_back(totalPts);
	}
	fclose(fp);

	uvVector.reserve(2000);
	for (ii = 0; ii < CorpusData.nCamera; ii++)
	{
		uvVector.clear();
		sprintf(Fname, "%s/CorpusK_%d.txt", Path, ii);
		FILE *fp = fopen(Fname, "r");
		while (fscanf(fp, "%lf %lf ", &uv.x, &uv.y) != EOF)
			uvVector.push_back(uv);
		fclose(fp);
		CorpusData.uvAllViews.push_back(uvVector);
	}

	sprintf(Fname, "%s/Corpus_Intrinsics.txt", Path); fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return 1;
	}
	CorpusData.camera = new CameraData[nCameras];
	for (viewID = 0; viewID < nCameras; viewID++)
	{
		fscanf(fp, "%d ", &CorpusData.camera[viewID].LensModel);
		for (int ii = 0; ii < 5; ii++)
			fscanf(fp, "%lf ", &CorpusData.camera[viewID].intrinsic[ii]);

		if (CorpusData.camera[viewID].LensModel == RADIAL_TANGENTIAL_PRISM)
			for (int ii = 0; ii < 7; ii++)
				fscanf(fp, "%lf ", &CorpusData.camera[viewID].distortion[ii]);
		else
		{
			for (int ii = 0; ii < 3; ii++)
				fscanf(fp, "%lf ", &CorpusData.camera[viewID].distortion[ii]);
		}
		GetKFromIntrinsic(CorpusData.camera[viewID]);
	}
	fclose(fp);

	if (notReadDescriptor)
		return 0;

	float desc;
	CorpusData.SiftDesc.create(totalPts, SIFTBINS, CV_32F);
	totalPts = 0;
	if (notbinary)
	{
		for (kk = 0; kk < nCameras; kk++)
		{
			threeDid.clear();
			sprintf(Fname, "%s/CorpusD_%d.txt", Path, kk);	fp = fopen(Fname, "r");

			int npts; fscanf(fp, "%d ", &npts);
			for (jj = 0; jj < npts; jj++)
			{
				fscanf(fp, "%d ", &id3D);
				threeDid.push_back(id3D);
				for (ii = 0; ii < SIFTBINS; ii++)
				{
					fscanf(fp, "%f ", &desc);
					CorpusData.SiftDesc.at<float>(totalPts, ii) = desc;
				}
				totalPts++;
			}
			fclose(fp);
			CorpusData.threeDIdAllViews.push_back(threeDid);
		}
	}
	else
	{
		for (kk = 0; kk < nCameras; kk++)
		{
			threeDid.clear();
			sprintf(Fname, "%s/CorpusD_%d.txt", Path, kk);
			ifstream fin; fin.open(Fname, ios::binary);
			if (!fin.is_open())
			{
				cout << "Cannot open: " << Fname << endl;
				abort();
			}

			int npts; fin.read(reinterpret_cast<char *>(&npts), sizeof(int));
			for (jj = 0; jj < npts; jj++)
			{
				fin.read(reinterpret_cast<char *>(&id3D), sizeof(int));
				threeDid.push_back(id3D);
				for (ii = 0; ii < SIFTBINS; ii++)
				{
					fin.read(reinterpret_cast<char *>(&desc), sizeof(float));
					CorpusData.SiftDesc.at<float>(totalPts, ii) = desc;
				}
				totalPts++;
			}
			fin.close();
			CorpusData.threeDIdAllViews.push_back(threeDid);
		}
	}

	return 0;
}
bool loadIndividualNVMforpose(char *Path, CameraData *CameraInfo, vector<int>availViews, int timeIDstart, int timeIDstop, int nviews, bool sharedIntrinsics)
{
	char Fname[200];
	bool *Avail = new bool[timeIDstop*nviews];
	for (int ii = 0; ii < timeIDstop*nviews; ii++)
		Avail[ii] = false;

	for (int kk = 0; kk < availViews.size(); kk++)
	{
		int camID = availViews.at(kk);
		for (int timeID = timeIDstart; timeID < timeIDstop; timeID++)
		{
			sprintf(Fname, "%s/%d_%d.nvm", Path, camID, timeID);
			ifstream ifs(Fname);
			if (ifs.fail())
			{
				cerr << "Cannot load " << Fname << endl;
				continue;
			}


			string token;
			ifs >> token; //NVM_V3
			if (token != "NVM_V3")
			{
				cerr << "Can only load NVM_V3" << endl;
				return false;
			}

			if (sharedIntrinsics == 1)
			{
				double fx, fy, u0, v0, radial1;
				ifs >> token >> fx >> u0 >> fy >> v0 >> radial1;
			}

			//loading camera parameters
			int nviews;
			ifs >> nviews;
			if (nviews <= 1)
			{
				cerr << "# of cameras must be more than 1." << endl;
				return false;
			}

			sprintf(Fname, "%d_%d", camID, timeID); //filename we are looking for
			double Quaterunion[4], CamCenter[3], T[3];
			for (int ii = 0; ii < nviews; ii++)
			{
				string filename;
				double f;
				vector<double> q(4), c(3), d(2);
				ifs >> filename >> f >> Quaterunion[0] >> Quaterunion[1] >> Quaterunion[2] >> Quaterunion[3] >> CamCenter[0] >> CamCenter[1] >> CamCenter[2] >> d[0] >> d[1];

				std::size_t pos = filename.find(".ppm");
				filename.erase(pos, 4);
				const char * str = filename.c_str();

				if (strcmp(str, Fname) == 0)
				{
					Avail[camID*timeIDstop + timeID] = true;
					ceres::QuaternionToRotation(Quaterunion, CameraInfo[camID*timeIDstop + timeID].R);
					mat_mul(CameraInfo[camID*timeIDstop + timeID].R, CamCenter, T, 3, 3, 1); //t = -RC
					CameraInfo[camID*timeIDstop + timeID].T[0] = -T[0], CameraInfo[camID*timeIDstop + timeID].T[1] = -T[1], CameraInfo[camID*timeIDstop + timeID].T[2] = -T[2];
					break;
				}
			}
		}
	}

	sprintf(Fname, "%s/PinfoGL.txt", Path);
	FILE *fp = fopen(Fname, "w+");
	double iR[9], center[3];
	for (int jj = 0; jj < nviews; jj++)
	{
		for (int ii = timeIDstart; ii < timeIDstop; ii++)
		{
			int viewID = jj*timeIDstop + ii;
			if (Avail[viewID])
			{
				mat_invert(CameraInfo[viewID].R, iR);
				CameraInfo[viewID].Rgl[0] = CameraInfo[viewID].R[0], CameraInfo[viewID].Rgl[1] = CameraInfo[viewID].R[1], CameraInfo[viewID].Rgl[2] = CameraInfo[viewID].R[2], CameraInfo[viewID].Rgl[3] = 0.0;
				CameraInfo[viewID].Rgl[4] = CameraInfo[viewID].R[3], CameraInfo[viewID].Rgl[5] = CameraInfo[viewID].R[4], CameraInfo[viewID].Rgl[6] = CameraInfo[viewID].R[5], CameraInfo[viewID].Rgl[7] = 0.0;
				CameraInfo[viewID].Rgl[8] = CameraInfo[viewID].R[6], CameraInfo[viewID].Rgl[9] = CameraInfo[viewID].R[7], CameraInfo[viewID].Rgl[10] = CameraInfo[viewID].R[8], CameraInfo[viewID].Rgl[11] = 0.0;
				CameraInfo[viewID].Rgl[12] = 0, CameraInfo[viewID].Rgl[13] = 0, CameraInfo[viewID].Rgl[14] = 0, CameraInfo[viewID].Rgl[15] = 1.0;

				mat_mul(iR, CameraInfo[viewID].T, center, 3, 3, 1);
				CameraInfo[viewID].camCenter[0] = -center[0], CameraInfo[viewID].camCenter[1] = -center[1], CameraInfo[viewID].camCenter[2] = -center[2];

				fprintf(fp, "%d %d: ", jj, ii);
				for (int jj = 0; jj < 16; jj++)
					fprintf(fp, "%.16f ", CameraInfo[viewID].Rgl[jj]);
				for (int jj = 0; jj < 3; jj++)
					fprintf(fp, "%.16f ", CameraInfo[viewID].camCenter[jj]);
				fprintf(fp, "\n");
			}
		}
	}
	fclose(fp);

	sprintf(Fname, "%s/Pinfo.txt", Path);
	fp = fopen(Fname, "w+");
	Mat rvec(1, 3, CV_64F), Rmat(3, 3, CV_64F);
	for (int jj = 0; jj < nviews; jj++)
	{
		for (int ii = timeIDstart; ii < timeIDstop; ii++)
		{
			int viewID = jj*timeIDstop + ii;
			if (Avail[viewID])
			{
				Rmat.at<double>(0, 0) = CameraInfo[viewID].R[0], Rmat.at<double>(0, 1) = CameraInfo[viewID].R[1], Rmat.at<double>(0, 2) = CameraInfo[viewID].R[2];
				Rmat.at<double>(1, 0) = CameraInfo[viewID].R[3], Rmat.at<double>(1, 1) = CameraInfo[viewID].R[4], Rmat.at<double>(1, 2) = CameraInfo[viewID].R[5];
				Rmat.at<double>(2, 0) = CameraInfo[viewID].R[6], Rmat.at<double>(2, 1) = CameraInfo[viewID].R[7], Rmat.at<double>(2, 2) = CameraInfo[viewID].R[8];

				Rodrigues(Rmat, rvec);

				fprintf(fp, "%d %d: ", jj, ii);
				for (int kk = 0; kk < 3; kk++)
					fprintf(fp, "%.16f ", rvec.at<double>(kk));
				for (int kk = 0; kk < 3; kk++)
					fprintf(fp, "%.16f ", CameraInfo[viewID].T[kk]);
				fprintf(fp, "\n");
			}
		}
	}
	fclose(fp);

	return true;
}


int GenerateCorpusVisualWords(char *Path, int nimages)
{
	char Fname[200];
	Mat img;

	vector<KeyPoint> keypoints;
	Mat descriptors, featuresUnclustered;
	SiftDescriptorExtractor detector;


	for (int ii = 0; ii < nimages; ii++)//= nimages / 5)
	{
		keypoints.clear();
		sprintf(Fname, "%s/%d.jpg", Path, ii);
		img = imread(Fname, CV_LOAD_IMAGE_GRAYSCALE);
		detector.detect(img, keypoints);
		detector.compute(img, keypoints, descriptors);
		featuresUnclustered.push_back(descriptors);
		printf("%.2f %%percent done\n", 100.0*ii / nimages);
	}

	//Construct BOWKMeansTrainer
	int dictionarySize = 1000, retries = 1, flags = KMEANS_PP_CENTERS;
	TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);

	//Create the BoW (or BoF) trainer
	double start = omp_get_wtime();
	BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
	Mat dictionary = bowTrainer.cluster(featuresUnclustered);
	printf("Finished generating the dictionary .... in %.2fs", omp_get_wtime() - start);

	//store the vocabulary
	start = omp_get_wtime();
	sprintf(Fname, "%s/dictionary.yml", Path);
	FileStorage fs(Fname, FileStorage::WRITE);
	fs << "vocabulary" << dictionary;
	fs.release();
	printf("Saving the dictionary .... in %.2fs", omp_get_wtime() - start);

	return 0;
}
int ComputeWordsHistogram(char *Path, int nimages)
{
	char filename[200];

	//prepare BOW descriptor extractor from the dictionary    
	Mat dictionary;
	sprintf(filename, "%s/dictionary.yml", Path);
	FileStorage fs(filename, FileStorage::READ);
	fs["vocabulary"] >> dictionary;
	fs.release();


	Mat img;
	for (int ii = 0; ii < nimages; ii++)
	{
		double start = omp_get_wtime();
		sprintf(filename, "%s/T%d.jpg", Path, ii);
		img = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

		Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
		Ptr<FeatureDetector> detector(new SiftFeatureDetector());
		Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);
		BOWImgDescriptorExtractor bowDE(extractor, matcher);
		bowDE.setVocabulary(dictionary);

		vector<KeyPoint> keypoints;
		detector->detect(img, keypoints);
		Mat bowDescriptor;
		bowDE.compute(img, keypoints, bowDescriptor);

		sprintf(filename, "%s/TH_%d.dat", Path, ii);
		float hist[1000];
		for (int jj = 0; jj < 1000; jj++)
			hist[jj] = bowDescriptor.at<float>(0, jj);
		WriteGridBinary(filename, hist, 1000, 1);
		printf("Finished generating histogram feature for frame %d.... in %.2fs\n", ii, omp_get_wtime() - start);
	}
	return 0;
}
int ReadCorpusAndVideoData(char *Path, CorpusandVideo &CorpusandVideoInfo, int ScannedCopursCam, int nVideoViews, int startTime, int stopTime, int LensModel, int distortionCorrected)
{
	char Fname[200];

	//READ INTRINSIC: START
	CameraData *IntrinsicInfo = new CameraData[nVideoViews];
	if (ReadIntrinsicResults(Path, IntrinsicInfo) != 0)
		return 1;
	for (int ii = 0; ii < nVideoViews; ii++)
	{
		IntrinsicInfo[ii].LensModel = LensModel, IntrinsicInfo[ii].threshold = 3.0, IntrinsicInfo[ii].ninlierThresh = 40;
		if (distortionCorrected == 1)
			for (int jj = 0; jj < 7; jj++)
				IntrinsicInfo[ii].distortion[jj] = 0.0;
	}
	//END

	//READ POSE FROM CORPUS: START
	sprintf(Fname, "%s/Corpus.nvm", Path);
	ifstream ifs(Fname);
	if (ifs.fail())
	{
		printf("Cannot load %s\n", Fname);
		return 1;
	}

	string token;
	ifs >> token; //NVM_V3
	if (token != "NVM_V3")
	{
		printf("Can only load NVM_V3\n");
		return 1;
	}
	double fx, fy, u0, v0, radial1;
	ifs >> token >> fx >> u0 >> fy >> v0 >> radial1;

	//loading camera parameters
	ifs >> CorpusandVideoInfo.nViewsCorpus;
	if (CorpusandVideoInfo.nViewsCorpus <= 1)
	{
		cerr << "# of cameras must be more than 1." << endl;
		return false;
	}
	CorpusandVideoInfo.CorpusInfo = new CameraData[CorpusandVideoInfo.nViewsCorpus];

	double Quaterunion[4], CamCenter[3], T[3];
	for (int ii = 0; ii < CorpusandVideoInfo.nViewsCorpus; ii++)
	{
		string filename;
		double f;
		vector<double> q(4), c(3), d(2);
		ifs >> filename >> f >> Quaterunion[0] >> Quaterunion[1] >> Quaterunion[2] >> Quaterunion[3] >> CamCenter[0] >> CamCenter[1] >> CamCenter[2] >> d[0] >> d[1];

		std::size_t pos = filename.find(".ppm");
		filename.erase(pos, 4);
		const char * str = filename.c_str();
		int viewID = atoi(str);

		ceres::QuaternionToRotation(Quaterunion, CorpusandVideoInfo.CorpusInfo[viewID].R);
		mat_mul(CorpusandVideoInfo.CorpusInfo[viewID].R, CamCenter, T, 3, 3, 1); //t = -RC
		CorpusandVideoInfo.CorpusInfo[viewID].T[0] = -T[0], CorpusandVideoInfo.CorpusInfo[viewID].T[1] = -T[1], CorpusandVideoInfo.CorpusInfo[viewID].T[2] = -T[2];

		for (int jj = 0; jj < 5; jj++)
			CorpusandVideoInfo.CorpusInfo[viewID].intrinsic[jj] = IntrinsicInfo[ScannedCopursCam].intrinsic[jj];
		for (int jj = 0; jj < 7; jj++)
			CorpusandVideoInfo.CorpusInfo[viewID].distortion[jj] = IntrinsicInfo[ScannedCopursCam].distortion[jj];

		GetKFromIntrinsic(CorpusandVideoInfo.CorpusInfo[viewID]);
		GetrtFromRT(CorpusandVideoInfo.CorpusInfo[viewID].rt, CorpusandVideoInfo.CorpusInfo[viewID].R, CorpusandVideoInfo.CorpusInfo[viewID].T);
		AssembleP(CorpusandVideoInfo.CorpusInfo[viewID].K, CorpusandVideoInfo.CorpusInfo[viewID].R, CorpusandVideoInfo.CorpusInfo[viewID].T, CorpusandVideoInfo.CorpusInfo[viewID].P);
	}
	//READ POSE FROM CORPUS: END

	//READ POSE FROM VIDEO POSE: START
	CorpusandVideoInfo.startTime = startTime, CorpusandVideoInfo.stopTime = stopTime, CorpusandVideoInfo.nVideos = nVideoViews;
	CorpusandVideoInfo.VideoInfo = new CameraData[nVideoViews*(stopTime - startTime + 1)];
	int id;
	double R[9], C[3], t1, t2, t3, t4, t5, t6, t7;
	for (int viewID = 0; viewID < nVideoViews; viewID++)
	{
		sprintf(Fname, "%s/PinfoGL_%d.txt", Path, viewID);
		FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("Cannot load %s\n", Fname);
			continue;
		}
		while (fscanf(fp, "%d: %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf ",
			&id, &R[0], &R[1], &R[2], &t1, &R[3], &R[4], &R[5], &t2, &R[6], &R[7], &R[8], &t3, &t4, &t5, &t6, &t7, &C[0], &C[1], &C[2]) != EOF)
		{
			for (int jj = 0; jj < 9; jj++)
				CorpusandVideoInfo.VideoInfo[id + viewID*(stopTime - startTime + 1)].R[jj] = R[jj];

			//T = -R*Center;
			mat_mul(R, C, CorpusandVideoInfo.VideoInfo[id + viewID*(stopTime - startTime + 1)].T, 3, 3, 1);
			for (int jj = 0; jj < 3; jj++)
				CorpusandVideoInfo.VideoInfo[id + viewID*(stopTime - startTime + 1)].T[jj] = -CorpusandVideoInfo.VideoInfo[id + viewID*(stopTime - startTime + 1)].T[jj];

			for (int jj = 0; jj < 5; jj++)
				CorpusandVideoInfo.VideoInfo[id + viewID*(stopTime - startTime + 1)].intrinsic[jj] = IntrinsicInfo[viewID].intrinsic[jj];
			for (int jj = 0; jj < 7; jj++)
				CorpusandVideoInfo.VideoInfo[id + viewID*(stopTime - startTime + 1)].distortion[jj] = IntrinsicInfo[viewID].distortion[jj];

			GetKFromIntrinsic(CorpusandVideoInfo.VideoInfo[id + viewID*(stopTime - startTime + 1)]);
			GetrtFromRT(CorpusandVideoInfo.VideoInfo[id + viewID*(stopTime - startTime + 1)].rt,
				CorpusandVideoInfo.VideoInfo[id + viewID*(stopTime - startTime + 1)].R, CorpusandVideoInfo.VideoInfo[id + viewID*(stopTime - startTime + 1)].T);
			AssembleP(CorpusandVideoInfo.VideoInfo[id + viewID*(stopTime - startTime + 1)].K, CorpusandVideoInfo.VideoInfo[id + viewID*(stopTime - startTime + 1)].R,
				CorpusandVideoInfo.VideoInfo[id + viewID*(stopTime - startTime + 1)].T, CorpusandVideoInfo.VideoInfo[id + viewID*(stopTime - startTime + 1)].P);
		}
	}
	//READ FROM VIDEO POSE: END

	return 0;
}
int ReadVideoData(char *Path, VideoData &AllVideoInfo, int nVideoViews, int startTime, int stopTime)
{
	char Fname[200];
	int videoID, frameID, LensType, width, height;

	AllVideoInfo.startTime = startTime, AllVideoInfo.stopTime = stopTime, AllVideoInfo.nVideos = nVideoViews;
	AllVideoInfo.VideoInfo = new CameraData[nVideoViews*(stopTime - startTime + 1)];

	//READ INTRINSIC: START
	int count = 0;
	for (int viewID = 0; viewID < nVideoViews; viewID++)
	{
		videoID = (stopTime - startTime + 1)*viewID, frameID = 0;
		sprintf(Fname, "%s/intrinsic_%d.txt", Path, viewID); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			cout << "Cannot load " << Fname << endl;
			count++;
			continue;
		}
		double fx, fy, skew, u0, v0, r0, r1, r2, t0, t1, p0, p1, omega, DistCtrX, DistCtrY;
		while (fscanf(fp, "%d %d %lf %lf %lf %lf %lf ", &frameID, &LensType, &fx, &fy, &skew, &u0, &v0) != EOF)
		{
			AllVideoInfo.VideoInfo[frameID + videoID].K[0] = fx, AllVideoInfo.VideoInfo[frameID + videoID].K[1] = skew, AllVideoInfo.VideoInfo[frameID + videoID].K[2] = u0,
				AllVideoInfo.VideoInfo[frameID + videoID].K[3] = 0.0, AllVideoInfo.VideoInfo[frameID + videoID].K[4] = fy, AllVideoInfo.VideoInfo[frameID + videoID].K[5] = v0,
				AllVideoInfo.VideoInfo[frameID + videoID].K[6] = 0.0, AllVideoInfo.VideoInfo[frameID + videoID].K[7] = 0.0, AllVideoInfo.VideoInfo[frameID + videoID].K[8] = 1.0;

			mat_invert(AllVideoInfo.VideoInfo[frameID + videoID].K, AllVideoInfo.VideoInfo[frameID + videoID].invK, 3);
			GetIntrinsicFromK(AllVideoInfo.VideoInfo[frameID + videoID]);
			//mat_invert(AllViewsParas[frameID].K, AllViewsParas[frameID].iK);

			AllVideoInfo.VideoInfo[frameID + videoID].LensModel = LensType, AllVideoInfo.VideoInfo[frameID + videoID].threshold = 3.0, AllVideoInfo.VideoInfo[frameID + videoID].ninlierThresh = 40;
			if (LensType == RADIAL_TANGENTIAL_PRISM)
			{
				fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf ", &r0, &r1, &r2, &t0, &t1, &p0, &p1);
				AllVideoInfo.VideoInfo[frameID + videoID].distortion[0] = r0, AllVideoInfo.VideoInfo[frameID + videoID].distortion[1] = r1, AllVideoInfo.VideoInfo[frameID + videoID].distortion[2] = r2;
				AllVideoInfo.VideoInfo[frameID + videoID].distortion[3] = t0, AllVideoInfo.VideoInfo[frameID + videoID].distortion[4] = t1;
				AllVideoInfo.VideoInfo[frameID + videoID].distortion[5] = p0, AllVideoInfo.VideoInfo[frameID + videoID].distortion[6] = p1;
			}
			else
			{
				fscanf(fp, "%lf %lf %lf ", &omega, &DistCtrX, &DistCtrY);
				AllVideoInfo.VideoInfo[frameID + videoID].distortion[0] = omega, AllVideoInfo.VideoInfo[frameID + videoID].distortion[1] = DistCtrX, AllVideoInfo.VideoInfo[frameID + videoID].distortion[2] = DistCtrY;
			}
			fscanf(fp, "%d %d ", &width, &height);
			AllVideoInfo.VideoInfo[frameID + videoID].width = width, AllVideoInfo.VideoInfo[frameID + videoID].height = height;
			if (frameID > stopTime - startTime)
				break;
		}
		fclose(fp);
	}
	if (count == nVideoViews)
		return 1;
	//END


	//READ POSE FROM VIDEO POSE: START
	count = 0;
	for (int viewID = 0; viewID < nVideoViews; viewID++)
	{
		videoID = (stopTime - startTime + 1)*viewID;
		sprintf(Fname, "%s/PinfoGL_%d.txt", Path, viewID);
		FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("Cannot load %s\n", Fname);
			count++;
			continue;
		}
		double R[9], C[3], t1, t2, t3, t4, t5, t6, t7;
		while (fscanf(fp, "%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf ",
			&frameID, &R[0], &R[1], &R[2], &t1, &R[3], &R[4], &R[5], &t2, &R[6], &R[7], &R[8], &t3, &t4, &t5, &t6, &t7, &C[0], &C[1], &C[2]) != EOF)
		{
			for (int jj = 0; jj < 9; jj++)
				AllVideoInfo.VideoInfo[frameID + videoID].R[jj] = R[jj];

			//T = -R*Center;
			mat_mul(R, C, AllVideoInfo.VideoInfo[frameID + videoID].T, 3, 3, 1);
			for (int jj = 0; jj < 3; jj++)
				AllVideoInfo.VideoInfo[frameID + videoID].T[jj] = -AllVideoInfo.VideoInfo[frameID + videoID].T[jj];

			mat_invert(AllVideoInfo.VideoInfo[frameID + videoID].R, AllVideoInfo.VideoInfo[frameID + videoID].invR);
			GetrtFromRT(AllVideoInfo.VideoInfo[frameID + videoID].rt,
				AllVideoInfo.VideoInfo[frameID + videoID].R, AllVideoInfo.VideoInfo[frameID + videoID].T);
			AssembleP(AllVideoInfo.VideoInfo[frameID + videoID].K, AllVideoInfo.VideoInfo[frameID + videoID].R,
				AllVideoInfo.VideoInfo[frameID + videoID].T, AllVideoInfo.VideoInfo[frameID + videoID].P);

			if (frameID > stopTime - startTime)
				break;
		}
	}
	if (count == nVideoViews)
		return 1;
	//READ FROM VIDEO POSE: END

	return 0;
}


void TMatchingSuperCoarse(double *Pattern, int pattern_size, int hsubset, double *Image, int width, int height, Point2i POI, int search_area, double thresh, double &zncc)
{
	//No interpolation at all, just slide the template around to compute the ZNCC
	int m, i, j, ii, jj, iii, jjj, II, JJ, length = width*height;
	double t_f, t_g, t_1, t_2, t_3, t_4, t_5, m_F, m_G;

	Point2d w_pt, ima_pt;
	int Pattern_cen_x = pattern_size / 2;
	int Pattern_cen_y = pattern_size / 2;

	FILE *fp1, *fp2;
	bool printout = false;


	double *T = new double[2 * (2 * hsubset + 1)*(2 * hsubset + 1)];
	zncc = 0.0;
	for (j = -search_area; j <= search_area; j++)
	{
		for (i = -search_area; i <= search_area; i++)
		{
			m = -1;
			t_f = 0.0;
			t_g = 0.0;

			if (printout)
			{
				fp1 = fopen("C:/temp/src.txt", "w+");
				fp2 = fopen("C:/temp/tar.txt", "w+");
			}

			for (jjj = -hsubset; jjj <= hsubset; jjj++)
			{
				for (iii = -hsubset; iii <= hsubset; iii++)
				{
					jj = Pattern_cen_y + jjj;
					ii = Pattern_cen_x + iii;

					JJ = POI.y + jjj + j;
					II = POI.x + iii + i;

					m_F = Pattern[ii + jj*pattern_size];
					m_G = Image[II + JJ*width];

					if (printout)
					{
						fprintf(fp1, "%.2f ", m_F);
						fprintf(fp2, "%.2f ", m_G);
					}
					m++;
					*(T + 2 * m + 0) = m_F;
					*(T + 2 * m + 1) = m_G;
					t_f += m_F;
					t_g += m_G;
				}
				if (printout)
				{
					fprintf(fp1, "\n");
					fprintf(fp2, "\n");
				}
			}
			if (printout)
			{
				fclose(fp1); fclose(fp2);
			}

			t_f = t_f / (m + 1);
			t_g = t_g / (m + 1);
			t_1 = 0.0;
			t_2 = 0.0;
			t_3 = 0.0;
			for (iii = 0; iii <= m; iii++)
			{
				t_4 = *(T + 2 * iii + 0) - t_f;
				t_5 = *(T + 2 * iii + 1) - t_g;
				t_1 += (t_4*t_5);
				t_2 += (t_4*t_4);
				t_3 += (t_5*t_5);
			}

			t_2 = sqrt(t_2*t_3);
			if (t_2 < 1e-10)
				t_2 = 1e-10;

			t_3 = t_1 / t_2;

			if (t_3 > 1.0 || t_3 < -1.0)
				t_3 = 0.0;

			if (t_3>thresh && t_3 > zncc)
				zncc = t_3;
			else if (t_3 < -thresh && t_3 < zncc)
				zncc = t_3;
		}
	}

	zncc = abs(zncc);

	delete[]T;
	return;
}
void LaplacianOfGaussian(double *LOG, int sigma)
{
	int n = ceil(sigma * 3), Size = 2 * n + 1;
	//[x, y] = meshgrid(-ceil(sigma * 3) : ceil(sigma * 3));
	//	L = -(1 - (x. ^ 2 + y. ^ 2) / 2 / sigma ^ 2) / pi / sigma^4.*exp(-(x. ^ 2 + y. ^ 2) / 2 / sigma ^ 2);

	double ii2, jj2, Twosigma2 = 2.0*sigma*sigma, sigma4 = pow(sigma, 4);
	for (int jj = -n; jj <= n; jj++)
	{
		for (int ii = -n; ii <= n; ii++)
		{
			ii2 = ii*ii, jj2 = jj*jj;
			LOG[(ii + n) + (jj + n)*Size] = (ii2 + jj2 - Twosigma2) / sigma4*exp(-(ii2 + jj2) / Twosigma2);
		}
	}

	return;
}
void DetectBlobCorrelation(double *img, int width, int height, Point2d *Checker, int &npts, double sigma, int search_area, int NMS_BW, double thresh)
{
	int i, j, ii, jj, kk, jump = 1, nMaxCorners = npts, numPatterns = 1;

	int hsubset = ceil(sigma * 3), PatternSize = hsubset * 2 + 1, PatternLength = PatternSize*PatternSize;
	double *maskSmooth = new double[PatternLength*numPatterns];
	LaplacianOfGaussian(maskSmooth, sigma);
	/*FILE *fp = fopen("C:/temp/LOG.txt", "r");
	for (int jj = 0; jj < PatternSize; jj++)
	for (int ii = 0; ii <PatternSize; ii++)
	fscanf(fp, "%lf ", &maskSmooth[ii + jj*PatternSize]);
	fclose(fp);*/

	SaveDataToImage("C:/temp/temp.png", maskSmooth, PatternSize, PatternSize, 1);

	//synthesize_square_mask(maskSmooth, bi_graylevel, PatternSize, 1.0, 0, false);
	double *Cornerness = new double[width*height];
	for (ii = 0; ii < width*height; ii++)
		Cornerness[ii] = 0.0;

	double zncc;
	Point2i POI;
	for (kk = 0; kk < numPatterns; kk++)
	{
		for (jj = 3 * hsubset; jj < height - 3 * hsubset; jj += jump)
		{
			for (ii = 3 * hsubset; ii < width - 3 * hsubset; ii += jump)
			{
				POI.x = ii, POI.y = jj;
				TMatchingSuperCoarse(maskSmooth + kk*PatternLength, PatternSize, hsubset, img, width, height, POI, search_area, thresh, zncc);
				Cornerness[ii + jj*width] = max(zncc, Cornerness[ii + jj*width]);
			}
		}
	}

	//ReadGridBinary("C:/temp/cornerness.dat", Cornerness, width, height);
	double *Cornerness2 = new double[width*height];
	for (ii = 0; ii < width*height; ii++)
		Cornerness2[ii] = Cornerness[ii];
	WriteGridBinary("C:/temp/cornerness.dat", Cornerness, width, height);

	//Non-max suppression
	bool breakflag;
	for (jj = 3 * hsubset; jj < height - 3 * hsubset; jj += jump)
	{
		for (ii = 3 * hsubset; ii < width - 3 * hsubset; ii += jump)
		{
			breakflag = false;
			if (Cornerness[ii + jj*width] < thresh)
			{
				Cornerness[ii + jj*width] = 0.0;
				Cornerness2[ii + jj*width] = 0.0;
			}
			else
			{
				for (j = -NMS_BW; j <= NMS_BW; j += jump)
				{
					for (i = -NMS_BW; i <= NMS_BW; i += jump)
					{
						if (Cornerness[ii + jj*width] < Cornerness[ii + i + (jj + j)*width] - 0.001) //avoid comparing with itse.8f
						{
							Cornerness2[ii + jj*width] = 0.0;
							breakflag = true;
							break;
						}
					}
				}
			}
			if (breakflag == true)
				break;
		}
	}
	WriteGridBinary("C:/temp/NMS.dat", Cornerness2, width, height);

	npts = 0;
	for (jj = 3 * hsubset; jj < height - 3 * hsubset; jj += jump)
	{
		for (ii = 3 * hsubset; ii < width - 3 * hsubset; ii += jump)
		{
			if (Cornerness2[ii + jj*width] > thresh)
			{
				Checker[npts].x = ii;
				Checker[npts].y = jj;
				npts++;
			}
			if (npts > nMaxCorners)
				break;
		}
	}

	delete[]maskSmooth;
	delete[]Cornerness;
	delete[]Cornerness2;

	return;
}

int ImportCalibDatafromHanFormat(char *Path, VideoData &AllVideoInfo, int nVGAPanels, int nVGACamsPerPanel, int nHDs)
{
	char Fname[200];
	int offset = 0;

	for (unsigned int viewID = 0; viewID < nHDs; viewID++)
	{
		sprintf(Fname, "%s/In/Calib/00_%02d.txt", Path, viewID); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("cannot load %s\n", Fname);
			continue;
		}
		//KMatrix load
		for (int j = 0; j < 9; j++)
			fscanf(fp, "%lf ", &AllVideoInfo.VideoInfo[viewID].K[j]);
		fscanf(fp, "%lf %lf ", &AllVideoInfo.VideoInfo[viewID].distortion[0], &AllVideoInfo.VideoInfo[viewID].distortion[1]);//lens distortion parameter

		//RT load
		double Quaterunion[4];
		sprintf(Fname, "%s/In/Calib/00_%02d_ext.txt", Path, viewID); fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("cannot load %s\n", Fname);
			return 1;
		}
		for (int j = 0; j < 4; j++)
			fscanf(fp, "%lf ", &Quaterunion[j]);
		for (int j = 0; j < 3; j++)
			fscanf(fp, "%lf ", &AllVideoInfo.VideoInfo[viewID].camCenter[j]);
		fclose(fp);

		ceres::QuaternionToAngleAxis(Quaterunion, AllVideoInfo.VideoInfo[viewID].rt);
		ceres::QuaternionRotatePoint(Quaterunion, AllVideoInfo.VideoInfo[viewID].camCenter, AllVideoInfo.VideoInfo[viewID].rt + 3);
		for (int j = 0; j < 3; j++) //position to translation t=-R*c
			AllVideoInfo.VideoInfo[viewID].rt[j + 3] = -AllVideoInfo.VideoInfo[viewID].rt[j + 3];

		AllVideoInfo.VideoInfo[viewID].LensModel = VisSFMLens;
		GetIntrinsicFromK(AllVideoInfo.VideoInfo[viewID]);
		GetRTFromrt(AllVideoInfo.VideoInfo[viewID]);
		AssembleP(AllVideoInfo.VideoInfo[viewID]);
	}

	for (int panelID = 0; panelID < nVGAPanels; panelID++)
	{
		for (int camID = 0; camID < nVGACamsPerPanel; camID++)
		{
			int viewID = panelID*nVGACamsPerPanel + camID+nHDs;
			sprintf(Fname, "%s/In/Calib/%02d_%02d.txt", Path, panelID+1, camID+1); FILE *fp = fopen(Fname, "r");
			if (fp == NULL)
			{
				printf("cannot load %s\n", Fname);
				continue;
			}

			//KMatrix load
			for (int j = 0; j < 9; j++)
				fscanf(fp, "%lf ", &AllVideoInfo.VideoInfo[viewID].K[j]);
			fscanf(fp, "%lf %lf ", &AllVideoInfo.VideoInfo[viewID].distortion[0], &AllVideoInfo.VideoInfo[viewID].distortion[1]);//lens distortion parameter

			//RT load
			double Quaterunion[4];
			sprintf(Fname, "%s/In/Calib/%02d_%02d_ext.txt", Path, panelID+1, camID+1); fp = fopen(Fname, "r");
			if (fp == NULL)
			{
				printf("cannot load %s\n", Fname);
				return 1;
			}
			for (int j = 0; j < 4; j++)
				fscanf(fp, "%lf ", &Quaterunion[j]);
			for (int j = 0; j < 3; j++)
				fscanf(fp, "%lf ", &AllVideoInfo.VideoInfo[viewID].camCenter[j]);
			fclose(fp);

			ceres::QuaternionToAngleAxis(Quaterunion, AllVideoInfo.VideoInfo[viewID].rt);
			ceres::QuaternionRotatePoint(Quaterunion, AllVideoInfo.VideoInfo[viewID].camCenter, AllVideoInfo.VideoInfo[viewID].rt + 3);
			for (int j = 0; j < 3; j++)//position to translation t=-R*c
				AllVideoInfo.VideoInfo[viewID].rt[j + 3] = -AllVideoInfo.VideoInfo[viewID].rt[j + 3];

			AllVideoInfo.VideoInfo[viewID].LensModel = VisSFMLens;
			GetIntrinsicFromK(AllVideoInfo.VideoInfo[viewID]);
			GetRTFromrt(AllVideoInfo.VideoInfo[viewID]);
			AssembleP(AllVideoInfo.VideoInfo[viewID]);
		}
	}

	return 0;
}
void ExportCalibDatatoHanFormat(char *Path, VideoData &AllVideoInfo, int nVideoViews, int startTime, int stopTime)
{
	char Fname[200];
	int offset = 0;

	for (unsigned int viewID = 0; viewID < nVideoViews; viewID++)
	{
		for (int frameID = startTime; frameID <= stopTime - offset; frameID++)
		{
			sprintf(Fname, "%s/Mem/Pinfo_%d%_%d.txt", Path, viewID, frameID); FILE *fp = fopen(Fname, "w+");
			if (fp == NULL)
			{
				sprintf("cannot load %s\n", Fname);
				continue;
			}
			int videoID = (stopTime - startTime + 1)*viewID;

			//Projection Matrix 	
			for (int j = 0; j < 12; j++)
				fprintf(fp, "%lf ", AllVideoInfo.VideoInfo[frameID + offset + videoID].P[j]);
			fprintf(fp, "\n");

			//KMatrix load
			for (int j = 0; j < 9; j++)
				fprintf(fp, "%lf ", AllVideoInfo.VideoInfo[frameID + offset + videoID].K[j]);
			fprintf(fp, "\n");

			fprintf(fp, "%lf %lf\n", 0.0, 0.0);//lens distortion parameter

			//RMatrix load
			for (int j = 0; j < 9; j++)
				fprintf(fp, "%lf ", AllVideoInfo.VideoInfo[frameID + offset + videoID].R[j]);
			fprintf(fp, "\n");

			//T Matrix load
			for (int j = 0; j < 3; j++)
				fprintf(fp, "%lf ", AllVideoInfo.VideoInfo[frameID + offset + videoID].T[j]);
			fclose(fp);


			sprintf(Fname, "%s/Mem/00_%.2d%_%d.txt", Path, viewID - 1, frameID); fp = fopen(Fname, "w+");
			for (int j = 0; j < 9; j++)
				fprintf(fp, "%lf ", AllVideoInfo.VideoInfo[frameID + offset + videoID].K[j]);
			fprintf(fp, "%lf %lf\n", 0.0, 0.0);//lens distortion parameter
			fclose(fp);

			double iR[9], center[3], Quaterunion[4];
			mat_invert(AllVideoInfo.VideoInfo[frameID + offset + videoID].R, iR);
			mat_mul(iR, AllVideoInfo.VideoInfo[frameID + offset + videoID].T, center, 3, 3, 1);
			AllVideoInfo.VideoInfo[frameID + offset + videoID].camCenter[0] = -center[0], AllVideoInfo.VideoInfo[frameID + offset + videoID].camCenter[1] = -center[1], AllVideoInfo.VideoInfo[frameID + offset + videoID].camCenter[2] = -center[2];

			ceres::AngleAxisToQuaternion(AllVideoInfo.VideoInfo[frameID + offset + videoID].rt, Quaterunion);

			sprintf(Fname, "%s/Mem/00_%.2d%_%d_ext.txt", Path, viewID - 1, frameID); fp = fopen(Fname, "w+");
			for (int j = 0; j < 4; j++)
				fprintf(fp, "%lf ", Quaterunion[j]);
			for (int j = 0; j < 3; j++)
				fprintf(fp, "%lf ", AllVideoInfo.VideoInfo[frameID + offset + videoID].camCenter[j]);
			fclose(fp);

			sprintf(Fname, "%s/%d", Path, viewID);
			//LensCorrectionDriver(Fname, AllVideoInfo.VideoInfo[videoID].K, AllVideoInfo.VideoInfo[videoID].distortion, AllVideoInfo.VideoInfo[videoID].LensModel, frameID, frameID, 1.0, 1.0, 5);
		}
	}
	return;


	Point3d xyz;
	Point3i rgb;
	vector<Point3d> Allxyz;
	vector<Point3i> Allrgb;
	vector<vector<int>> AllVis;
	for (int frameID = startTime; frameID <= stopTime; frameID++)
	{
		Allxyz.clear(), Allrgb.clear();
		AllVis.clear();

		sprintf(Fname, "%s/3dGL_%d.xyz", Path, frameID); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("Cannot load %s\n", Fname);
			continue;
		}
		while (fscanf(fp, "%lf %lf %lf %d %d %d ", &xyz.x, &xyz.y, &xyz.z, &rgb.x, &rgb.y, &rgb.z) != EOF)
			Allxyz.push_back(xyz), Allrgb.push_back(rgb);
		fclose(fp);

		sprintf(Fname, "%s/3dVis_%d.txt", Path, frameID); fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("Cannot load %s\n", Fname);
			continue;
		}
		int nvis, ivis;
		vector<int> Vis;
		while (fscanf(fp, "%d ", &nvis) != EOF)
		{
			Vis.clear();
			for (int ii = 0; ii < nvis; ii++)
			{
				fscanf(fp, "%d ", &ivis);
				Vis.push_back(ivis);
			}
			AllVis.push_back(Vis);
		}
		fclose(fp);

		sprintf(Fname, "%s/Mem/reconResult%08d.mem", Path, frameID); fp = fopen(Fname, "w+");
		fprintf(fp, "ver 1.0\n %d\n", nVideoViews - 1);
		for (unsigned int viewID = 1; viewID < nVideoViews; viewID++)
		{
			int videoID = (stopTime - startTime + 1)*viewID;
			double iR[9], center[3];
			mat_invert(AllVideoInfo.VideoInfo[videoID].R, iR);

			AllVideoInfo.VideoInfo[videoID].Rgl[0] = AllVideoInfo.VideoInfo[videoID].R[0], AllVideoInfo.VideoInfo[videoID].Rgl[1] = AllVideoInfo.VideoInfo[videoID].R[1], AllVideoInfo.VideoInfo[videoID].Rgl[2] = AllVideoInfo.VideoInfo[videoID].R[2], AllVideoInfo.VideoInfo[videoID].Rgl[3] = 0.0;
			AllVideoInfo.VideoInfo[videoID].Rgl[4] = AllVideoInfo.VideoInfo[videoID].R[3], AllVideoInfo.VideoInfo[videoID].Rgl[5] = AllVideoInfo.VideoInfo[videoID].R[4], AllVideoInfo.VideoInfo[videoID].Rgl[6] = AllVideoInfo.VideoInfo[videoID].R[5], AllVideoInfo.VideoInfo[videoID].Rgl[7] = 0.0;
			AllVideoInfo.VideoInfo[videoID].Rgl[8] = AllVideoInfo.VideoInfo[videoID].R[6], AllVideoInfo.VideoInfo[videoID].Rgl[9] = AllVideoInfo.VideoInfo[videoID].R[7], AllVideoInfo.VideoInfo[videoID].Rgl[10] = AllVideoInfo.VideoInfo[videoID].R[8], AllVideoInfo.VideoInfo[videoID].Rgl[11] = 0.0;
			AllVideoInfo.VideoInfo[videoID].Rgl[12] = 0, AllVideoInfo.VideoInfo[videoID].Rgl[13] = 0, AllVideoInfo.VideoInfo[videoID].Rgl[14] = 0, AllVideoInfo.VideoInfo[videoID].Rgl[15] = 1.0;

			mat_mul(iR, AllVideoInfo.VideoInfo[videoID].T, center, 3, 3, 1);
			AllVideoInfo.VideoInfo[videoID].camCenter[0] = -center[0], AllVideoInfo.VideoInfo[videoID].camCenter[1] = -center[1], AllVideoInfo.VideoInfo[videoID].camCenter[2] = -center[2];

			for (int ii = 0; ii < 3; ii++)
				fprintf(fp, "%lf ", AllVideoInfo.VideoInfo[videoID].camCenter[ii]);
			fprintf(fp, "\n");
			for (int ii = 0; ii < 16; ii++)
				fprintf(fp, "%lf ", AllVideoInfo.VideoInfo[videoID].Rgl[ii]);
			fprintf(fp, "\n");
			for (int ii = 0; ii < 12; ii++)
				fprintf(fp, "%lf ", AllVideoInfo.VideoInfo[videoID].P[ii]);
			fprintf(fp, "\n");
			for (int ii = 0; ii < 9; ii++)
				fprintf(fp, "%lf ", AllVideoInfo.VideoInfo[videoID].K[ii]);
			fprintf(fp, "\n 0.0 0.0\n");
			for (int ii = 0; ii < 9; ii++)
				fprintf(fp, "%lf ", AllVideoInfo.VideoInfo[videoID].R[ii]);
			fprintf(fp, "\n");
			for (int ii = 0; ii < 3; ii++)
				fprintf(fp, "%lf ", AllVideoInfo.VideoInfo[videoID].T[ii]);
			fprintf(fp, "\n");
			fprintf(fp, "D:/Z/In/%.8d/%.8d_%.2d_%.2d.png\n", frameID, frameID, 0, viewID - 1);
		}

		double nx1 = 0.0, ny1 = 0.0, nz1 = 0.0, nx2 = 0.0, ny2 = 0.0, nz2 = 0.0;
		fprintf(fp, "%d \n", Allxyz.size());
		for (int ii = 0; ii < Allxyz.size(); ii++)
		{
			fprintf(fp, "Pt3D %d %.4f %.4f %.4f %3f %.3f %.3f %.1f %.3f %.3f %.3f %.3f %.3f %.3f", ii, Allxyz[ii].x, Allxyz[ii].y, Allxyz[ii].z,
				1.0*Allrgb[ii].x / 255, 1.0*Allrgb[ii].y / 255, 1.0*Allrgb[ii].z / 255, 3.0,
				nx1, ny1, nz1,
				nx2, ny2, nz2);
			fprintf(fp, "\n%d ", AllVis[ii].size());
			for (int jj = 0; jj < AllVis[ii].size(); jj++)
				fprintf(fp, "%d 0.0 0.0 ", AllVis[ii].at(jj) - 1);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}
}

void GetRCGL(CameraData &camInfo)
{
	double iR[9], center[3];
	mat_invert(camInfo.R, iR);

	camInfo.Rgl[0] = camInfo.R[0], camInfo.Rgl[1] = camInfo.R[1], camInfo.Rgl[2] = camInfo.R[2], camInfo.Rgl[3] = 0.0;
	camInfo.Rgl[4] = camInfo.R[3], camInfo.Rgl[5] = camInfo.R[4], camInfo.Rgl[6] = camInfo.R[5], camInfo.Rgl[7] = 0.0;
	camInfo.Rgl[8] = camInfo.R[6], camInfo.Rgl[9] = camInfo.R[7], camInfo.Rgl[10] = camInfo.R[8], camInfo.Rgl[11] = 0.0;
	camInfo.Rgl[12] = 0, camInfo.Rgl[13] = 0, camInfo.Rgl[14] = 0, camInfo.Rgl[15] = 1.0;

	mat_mul(iR, camInfo.T, center, 3, 3, 1);
	camInfo.camCenter[0] = -center[0], camInfo.camCenter[1] = -center[1], camInfo.camCenter[2] = -center[2];
	return;
}
void CopyCamereInfo(CameraData Src, CameraData &Dst)
{
	int ii;
	for (ii = 0; ii < 9; ii++)
		Dst.K[ii] = Src.K[ii];
	for (ii = 0; ii < 7; ii++)
		Dst.distortion[ii] = Src.distortion[ii];
	for (ii = 0; ii < 5; ii++)
		Dst.intrinsic[ii] = Src.intrinsic[ii];
	for (ii = 0; ii < 9; ii++)
		Dst.R[ii] = Src.R[ii];
	for (ii = 0; ii < 3; ii++)
		Dst.T[ii] = Src.T[ii];
	for (ii = 0; ii < 6; ii++)
		Dst.rt[ii] = Src.rt[ii];
	for (ii = 0; ii < 12; ii++)
		Dst.P[ii] = Src.P[ii];
	for (ii = 0; ii < 16; ii++)
		Dst.Rgl[ii] = Src.Rgl[ii];
	for (ii = 0; ii < 3; ii++)
		Dst.camCenter[ii] = Src.camCenter[ii];
	Dst.LensModel = Src.LensModel;
	Dst.ninlierThresh = Src.ninlierThresh;
	Dst.threshold = Src.threshold;
	Dst.width = Src.width, Dst.height = Src.height;
}
void GenerateViewAll_3D_2DInliers(char *Path, int viewID, int startID, int stopID, int n3Dcorpus)
{
	char Fname[200];
	vector<int> *All3DviewIDper3D = new vector<int>[n3Dcorpus];
	vector<Point2d> *Alluvper3D = new vector<Point2d>[n3Dcorpus];

	int threeDid;
	double u, v;
	for (int timeID = startID; timeID <= stopID; timeID++)
	{
		sprintf(Fname, "%s/%d/Inliers_3D2D_%d.txt", Path, viewID, timeID);
		FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("Cannot load %s\n", Fname);
			continue;
		}
		while (fscanf(fp, "%d %lf %lf", &threeDid, &u, &v) != EOF)
		{
			All3DviewIDper3D[threeDid].push_back(timeID);
			Alluvper3D[threeDid].push_back(Point2d(u, v));
		}
		fclose(fp);
	}

	sprintf(Fname, "%s/%d/Inliers_3D2D.txt", Path, viewID);
	FILE *fp = fopen(Fname, "w+");
	for (int ii = 0; ii < n3Dcorpus; ii++)
	{
		fprintf(fp, "%d\n", All3DviewIDper3D[ii].size());
		for (int jj = 0; jj < All3DviewIDper3D[ii].size(); jj++)
			fprintf(fp, "%d %f %f ", All3DviewIDper3D[ii].at(jj), Alluvper3D[ii].at(jj).x, Alluvper3D[ii].at(jj).y);
		if (All3DviewIDper3D[ii].size() != 0)
			fprintf(fp, "\n");
	}
	fclose(fp);

	return;
}