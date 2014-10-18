#if !defined(ULTILITY_H )
#define ULTILITY_H
#pragma once

#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <limits>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <opencv2/opencv.hpp>
#include "opencv2\nonfree\features2d.hpp"
#include <opencv2/nonfree/nonfree.hpp>
#include "SiftGPU\src\SiftGPU\SiftGPU.h"
#include "DataStructure.h"

using namespace cv;
using namespace std;

bool myImgReader(char *fname, unsigned char *Img, int width, int height, int nchannels);
bool myImgReader(char *fname, float *Img, int width, int height, int nchannels);
bool myImgReader(char *fname, double *Img, int width, int height, int nchannels);

void ShowDataToImage(char *Fname, char *Img, int width, int height, int nchannels, IplImage *cvImg = 0);

bool SaveDataToImage(char *fname, char *Img, int width, int height, int nchannels = 1);
bool SaveDataToImage(char *fname, unsigned char *Img, int width, int height, int nchannels = 1);
bool SaveDataToImage(char *fname, float *Img, int width, int height, int nchannels = 1);
bool SaveDataToImage(char *fname, double *Img, int width, int height, int nchannels = 1);

template <class myType> bool WriteGridBinary(char *fn, myType *data, int width, int height, bool silent = false)
{
	ofstream fout;
	fout.open(fn, ios::binary);
	if (!fout.is_open())
	{
		if (silent)
			cout << "Cannot write: " << fn << endl;
		return false;
	}

	for (int j = 0; j < height; ++j)
		for (int i = 0; i < width; ++i)
			fout.write(reinterpret_cast<char *>(&data[i + j*width]), sizeof(myType));
	fout.close();

	return true;
}
template <class myType> bool ReadGridBinary(char *fn, myType *data, int width, int height, bool silent = false)
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

	for (int j = 0; j < height; ++j)
		for (int i = 0; i < width; ++i)
			fin.read(reinterpret_cast<char *>(&data[i + j*width]), sizeof(myType));
	fin.close();

	return true;
}

bool WriteKPointsBinary(char *fn, vector<KeyPoint>kpts, bool silent = false);
bool ReadKPointsBinary(char *fn, vector<KeyPoint> &kpts, bool silent = false);
bool WriteDescriptorBinary(char *fn, Mat descriptor, bool silent = false);
Mat ReadDescriptorBinary(char *fn, int descriptorSize, bool silent = false);

bool WriteKPointsBinarySIFTGPU(char *fn, vector<SiftGPU::SiftKeypoint>kpts, bool silent = false);
bool ReadKPointsBinarySIFTGPU(char *fn, vector<SiftGPU::SiftKeypoint> &kpts, bool silent = false);
bool WriteKPointsBinarySIFTGPU(char *fn, vector<KeyPoint>kpts, bool silent = false);
bool ReadKPointsBinarySIFTGPU(char *fn, vector<KeyPoint> &kpts, bool silent = false);
bool WriteDescriptorBinarySIFTGPU(char *fn, vector<float > descriptors, bool silent = false);
bool ReadDescriptorBinarySIFTGPU(char *fn, vector<float > &descriptors, bool silent = false);
Mat ReadDescriptorBinarySIFTGPU(char *fn, bool silent = false);

double nChoosek(int n, int k);
int MyFtoI(double W);
bool IsNumber(double x);
bool IsFiniteNumber(double x);
double UniformNoise(double High, double Low);
double gaussian_noise(double mean, double std);

float MeanArray(float *data, int length);
double MeanArray(double *data, int length);
double VarianceArray(double *data, int length, double mean = NULL);
double MeanArray(vector<double>data);
double VarianceArray(vector<double>data, double mean = NULL);
void normalize(double *x, int dim = 3);
double norm_dot_product(double *x, double *y, int dim = 3);
void cross_product(double *x, double *y, double *xy);
void ZNCC1D(float *A, const int dimA, float *B, const int dimB, float *Result, float *nB = NULL);
void ZNCC1D(double *A, int Asize, double *B, int Bsize, double *Result);

void mat_invert(double* mat, double* imat, int dims = 3);
void mat_invert(float* mat, float* imat, int dims = 3);
void mat_mul(double *aa, double *bb, double *out, int rowa, int col_row, int colb);
void mat_add(double *aa, double *bb, double* cc, int row, int col, double scale_a = 1.0, double scale_b = 1.0);
void mat_subtract(double *aa, double *bb, double* cc, int row, int col, double scale_a = 1.0, double scale_b = 1.0);
void mat_transpose(double *in, double *out, int row_in, int col_in);
void mat_completeSym(double *mat, int size, bool upper = true);
void Rodrigues_trans(double *RT_vec, double *R_mat, bool vec2mat, double *dR_dm = NULL);


void LS_Solution_Double(double *lpA, double *lpB, int m, int n);
void QR_Solution_Double(double *lpA, double *lpB, int m, int n);
void Quick_Sort_Double(double * A, int *B, int low, int high);
void Quick_Sort_Float(float * A, int *B, int low, int high);
void Quick_Sort_Int(int * A, int *B, int low, int high);

bool in_polygon(double u, double v, Point2d *vertex, int num_vertex);

void ConvertToHeatMap(double *Map, unsigned char *ColorMap, int width, int height, bool *mask = 0);

void ResizeImage(unsigned char *Image, unsigned char *OutImage, int width, int height, int nchannels, double Rfactor, int InterpAlgo, double *InPara = 0);
void ResizeImage(float *Image, float *OutImage, int width, int height, int channels, double Rfactor, int InterpAlgo, float *Para = 0);
void ResizeImage(double *Image, double *OutImage, int width, int height, int nchannels, double Rfactor, int InterpAlgo, double *Para = 0);

template <class myType>void Get_Sub_Mat(myType *srcMat, myType *dstMat, int srcWidth, int srcHeight, int dstWidth, int startCol, int startRow)
{
	int ii, jj;

	for (jj = startRow; jj < startRow + srcHeight; jj++)
		for (ii = startCol; ii < startCol + srcWidth; ii++)
			dstMat[ii - startCol + (jj - startRow)*dstWidth] = srcMat[ii + jj*srcWidth];

	return;
}
template <class myType>void Set_Sub_Mat(myType *srcMat, myType *dstMat, int srcWidth, int srcHeight, int dstWidth, int startCol, int startRow)
{
	int ii, jj;

	for (jj = 0; jj < srcHeight; jj++)
	{
		for (ii = 0; ii < srcWidth; ii++)
		{
			dstMat[ii + startCol + (jj + startRow)*dstWidth] = srcMat[ii + jj*srcWidth];
		}
	}

	return;
}

template <class m_Type> class m_TemplateClass_1
{
public:
	void Quick_Sort(m_Type* A, int *B, int low, int high);
	void QR_Solution(m_Type *lpA, m_Type *lpB, int m, int n);
	void QR_Solution_2(m_Type *lpA, m_Type *lpB, int m, int n, int k);
};

template <class m_Type> void m_TemplateClass_1<m_Type>::Quick_Sort(m_Type* A, int *B, int low, int high)
//A: array to be sorted (from min to max); B: index of the original array; low and high: array range
//After sorting, A: sorted array; B: re-sorted index of the original array, e.g., the m-th element of
// new A[] is the original n-th element in old A[]. B[m-1]=n-1;
//B[] is useless for most sorting, it is added here for the special application in this program.  
{
	m_Type A_pivot, A_S;
	int B_pivot, B_S;
	int scanUp, scanDown;
	int mid;
	if (high - low <= 0)
		return;
	else if (high - low == 1)
	{
		if (A[high] < A[low])
		{
			//	Swap(A[low],A[high]);
			//	Swap(B[low],B[high]);
			A_S = A[low];
			A[low] = A[high];
			A[high] = A_S;
			B_S = B[low];
			B[low] = B[high];
			B[high] = B_S;
		}
		return;
	}
	mid = (low + high) / 2;
	A_pivot = A[mid];
	B_pivot = B[mid];

	//	Swap(A[mid],A[low]);
	//	Swap(B[mid],B[low]);
	A_S = A[mid];
	A[mid] = A[low];
	A[low] = A_S;
	B_S = B[mid];
	B[mid] = B[low];
	B[low] = B_S;

	scanUp = low + 1;
	scanDown = high;
	do
	{
		while (scanUp <= scanDown && A[scanUp] <= A_pivot)
			scanUp++;
		while (A_pivot < A[scanDown])
			scanDown--;
		if (scanUp < scanDown)
		{
			//	Swap(A[scanUp],A[scanDown]);
			//	Swap(B[scanUp],B[scanDown]);
			A_S = A[scanUp];
			A[scanUp] = A[scanDown];
			A[scanDown] = A_S;
			B_S = B[scanUp];
			B[scanUp] = B[scanDown];
			B[scanDown] = B_S;
		}
	} while (scanUp < scanDown);

	A[low] = A[scanDown];
	B[low] = B[scanDown];
	A[scanDown] = A_pivot;
	B[scanDown] = B_pivot;
	if (low < scanDown - 1)
		Quick_Sort(A, B, low, scanDown - 1);
	if (scanDown + 1 < high)
		Quick_Sort(A, B, scanDown + 1, high);
}
template <class m_Type> void m_TemplateClass_1<m_Type>::QR_Solution(m_Type *lpA, m_Type *lpB, int m, int n)
{
	int ii, jj, mm, kk;
	m_Type t, d, alpha, u;
	m_Type *lpC = new m_Type[n];
	m_Type *lpQ = new m_Type[m*m];

	for (ii = 0; ii < m; ii++)
	{
		for (jj = 0; jj < m; jj++)
		{
			*(lpQ + ii*m + jj) = (m_Type)0;
			if (ii == jj)
				*(lpQ + ii*m + jj) = (m_Type)1;
		}
	}

	for (kk = 0; kk<n; kk++)
	{
		u = (m_Type)0;
		for (ii = kk; ii<m; ii++)
		{
			if (fabs(*(lpA + ii*n + kk))>u)
				u = (m_Type)(fabs(*(lpA + ii*n + kk)));
		}

		alpha = (m_Type)0;
		for (ii = kk; ii < m; ii++)
		{
			t = *(lpA + ii*n + kk) / u;
			alpha = alpha + t*t;
		}
		if (*(lpA + kk*n + kk) > (m_Type)0)
			u = -u;
		alpha = (m_Type)(u*sqrt(alpha));
		u = (m_Type)(sqrt(2.0*alpha*(alpha - *(lpA + kk*n + kk))));
		if (fabs(u)>1e-8)
		{
			*(lpA + kk*n + kk) = (*(lpA + kk*n + kk) - alpha) / u;
			for (ii = kk + 1; ii < m; ii++)
				*(lpA + ii*n + kk) = *(lpA + ii*n + kk) / u;
			for (jj = 0; jj < m; jj++)
			{
				t = (m_Type)0;
				for (mm = kk; mm < m; mm++)
					t = t + *(lpA + mm*n + kk)*(*(lpQ + mm*m + jj));
				for (ii = kk; ii < m; ii++)
					*(lpQ + ii*m + jj) = *(lpQ + ii*m + jj) - (m_Type)(2.0*t*(*(lpA + ii*n + kk)));
			}
			for (jj = kk + 1; jj < n; jj++)
			{
				t = (m_Type)0;
				for (mm = kk; mm < m; mm++)
					t = t + *(lpA + mm*n + kk)*(*(lpA + mm*n + jj));
				for (ii = kk; ii < m; ii++)
					*(lpA + ii*n + jj) = *(lpA + ii*n + jj) - (m_Type)(2.0*t*(*(lpA + ii*n + kk)));
			}
			*(lpA + kk*n + kk) = alpha;
			for (ii = kk + 1; ii < m; ii++)
				*(lpA + ii*n + kk) = (m_Type)0;
		}
	}
	for (ii = 0; ii < m - 1; ii++)
	{
		for (jj = ii + 1; jj < m; jj++)
		{
			t = *(lpQ + ii*m + jj);
			*(lpQ + ii*m + jj) = *(lpQ + jj*m + ii);
			*(lpQ + jj*m + ii) = t;
		}
	}
	//Solve the equation
	for (ii = 0; ii < n; ii++)
	{
		d = (m_Type)0;
		for (jj = 0; jj < m; jj++)
			d = d + *(lpQ + jj*m + ii)*(*(lpB + jj));
		*(lpC + ii) = d;
	}
	*(lpB + n - 1) = *(lpC + n - 1) / (*(lpA + (n - 1)*n + n - 1));
	for (ii = n - 2; ii >= 0; ii--)
	{
		d = (m_Type)0;
		for (jj = ii + 1; jj < n; jj++)
			d = d + *(lpA + ii*n + jj)*(*(lpB + jj));
		*(lpB + ii) = (*(lpC + ii) - d) / (*(lpA + ii*n + ii));
	}

	delete[]lpQ;
	delete[]lpC;
	return;
}
template <class m_Type> void m_TemplateClass_1<m_Type>::QR_Solution_2(m_Type *lpA, m_Type *lpB, int m, int n, int k)
{
	int ii, jj, mm, kk;
	m_Type t, d, alpha, u;
	m_Type *lpC = new m_Type[n];
	m_Type *lpQ = new m_Type[m*m];

	for (ii = 0; ii < m; ii++)
	{
		for (jj = 0; jj < m; jj++)
		{
			*(lpQ + ii*m + jj) = (m_Type)0;
			if (ii == jj)
				*(lpQ + ii*m + jj) = (m_Type)1;
		}
	}

	for (kk = 0; kk<n; kk++)
	{
		u = (m_Type)0;
		for (ii = kk; ii<m; ii++)
		{
			if (fabs(*(lpA + ii*n + kk))>u)
				u = (m_Type)(fabs(*(lpA + ii*n + kk)));
		}

		alpha = (m_Type)0;
		for (ii = kk; ii < m; ii++)
		{
			t = *(lpA + ii*n + kk) / u;
			alpha = alpha + t*t;
		}
		if (*(lpA + kk*n + kk) > (m_Type)0)
			u = -u;
		alpha = (m_Type)(u*sqrt(alpha));
		u = (m_Type)(sqrt(2.0*alpha*(alpha - *(lpA + kk*n + kk))));
		if (fabs(u)>1e-8)
		{
			*(lpA + kk*n + kk) = (*(lpA + kk*n + kk) - alpha) / u;
			for (ii = kk + 1; ii < m; ii++)
				*(lpA + ii*n + kk) = *(lpA + ii*n + kk) / u;
			for (jj = 0; jj < m; jj++)
			{
				t = (m_Type)0;
				for (mm = kk; mm < m; mm++)
					t = t + *(lpA + mm*n + kk)*(*(lpQ + mm*m + jj));
				for (ii = kk; ii < m; ii++)
					*(lpQ + ii*m + jj) = *(lpQ + ii*m + jj) - (m_Type)(2.0*t*(*(lpA + ii*n + kk)));
			}
			for (jj = kk + 1; jj < n; jj++)
			{
				t = (m_Type)0;
				for (mm = kk; mm < m; mm++)
					t = t + *(lpA + mm*n + kk)*(*(lpA + mm*n + jj));
				for (ii = kk; ii < m; ii++)
					*(lpA + ii*n + jj) = *(lpA + ii*n + jj) - (m_Type)(2.0*t*(*(lpA + ii*n + kk)));
			}
			*(lpA + kk*n + kk) = alpha;
			for (ii = kk + 1; ii < m; ii++)
				*(lpA + ii*n + kk) = (m_Type)0;
		}
	}
	for (ii = 0; ii < m - 1; ii++)
	{
		for (jj = ii + 1; jj < m; jj++)
		{
			t = *(lpQ + ii*m + jj);
			*(lpQ + ii*m + jj) = *(lpQ + jj*m + ii);
			*(lpQ + jj*m + ii) = t;
		}
	}
	//Solve the equation

	m_Type *lpBB;
	for (mm = 0; mm < k; mm++)
	{
		lpBB = lpB + mm*m;

		for (ii = 0; ii < n; ii++)
		{
			d = (m_Type)0;
			for (jj = 0; jj < m; jj++)
				d = d + *(lpQ + jj*m + ii)*(*(lpBB + jj));
			*(lpC + ii) = d;
		}
		*(lpBB + n - 1) = *(lpC + n - 1) / (*(lpA + (n - 1)*n + n - 1));
		for (ii = n - 2; ii >= 0; ii--)
		{
			d = (m_Type)0;
			for (jj = ii + 1; jj < n; jj++)
				d = d + *(lpA + ii*n + jj)*(*(lpBB + jj));
			*(lpBB + ii) = (*(lpC + ii) - d) / (*(lpA + ii*n + ii));
		}
	}

	delete[]lpQ;
	delete[]lpC;
	return;
}


int PickStaticImagesFromVideo(char *PATH, char *VideoName, int SaveFrameDif, int redetectInterval, double percentile, double MovingThresh2, int &nNonBlurImages, bool visual);
int PickStaticImagesFromImages(char *PATH, int SaveFrameDif, int redetectInterval, double percentile, double MovingThresh2, bool visual);
template <class myType>void nonMinimaSuppression1D(myType *src, int nsample, int *MinEle, int &nMinEle, int halfWnd)
{
	int i = 0, minInd = 0, srcCnt = 0, ele;
	nMinEle = 0;
	while (i < nsample)
	{
		if (minInd < i - halfWnd)
			minInd = i - halfWnd;

		ele = min(i + halfWnd, nsample);
		while (minInd <= ele)
		{
			srcCnt++;
			if (src[minInd] < src[i])
				break;
			minInd++;
		}

		if (minInd > ele) // src(i) is a maxima in the search window
		{
			MinEle[nMinEle] = i, nMinEle++; // the loop above suppressed the maximum, so set it back
			minInd = i + 1;
			i += halfWnd;
		}
		i++;
	}

	return;
}
template <class myType>void nonMaximaSuppression1D(myType *src, int nsample, int *MaxEle, int &nMaxEle, int hWind)
{
	myType *src2 = new myType[nsample];
	for (int ii = 0; ii < nsample; ii++)
		src2[ii] = -src[ii];

	nonMinimaSuppression1D(src2, nsample, MaxEle, nMaxEle, hWind);

	return;
}
void nonMaximaSuppression(const Mat& src, const int sz, Mat& dst, const Mat mask);

int LensCorrectionVideoDriver(char *Path, char *VideoName, double *K, double *distortion, int LensType, int nimages, int interpAlgo);
int LensCorrectionDriver(char *Path, double *K, double *distortion, int LensType, int nimages, int interpAlgo);
int DisplayImageCorrespondence(IplImage* correspond, int offsetX, int offsetY, vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2, vector<int>pair, double density);
int DisplayImageCorrespondencesDriver(char *Path, vector<int>viewsID, int timeID, int nchannels, double density = 0.25);

int ReadIntrinsicResults(char *path, CameraData *DeviceParas, int nCam);
int SaveIntrinsicResults(char *path, CameraData *AllViewsParas, int nCams);
void SaveCurrentSfmInfo(char *path, CameraData *AllViewParas, vector<int>AvailViews, Point3d *All3D, int npts);
void ReadCurrentSfmInfo(char *path, CameraData *AllViewParas, vector<int>&AvailViews, Point3d *All3D, int npts);
void ReadCumulativePoints(char *Path, int nviews, int timeID, vector<int>&cumulativePts);
void ReadPointCorrespondences(char *Path, int nviews, int timeID, vector<int> *PointCorres, vector<int>&mask, int totalPts, bool Merge =false);
void ReadPointCorrespondences(char *Path, int nviews, int timeID, vector<int> *PointCorres, int totalPts, bool Merge);
void GenerateMergePointCorrespondences(vector<int> *MergePointCorres, vector<int> *PointCorres, int totalPts);
void GenerateViewandPointCorrespondences(vector<int> *ViewCorres, vector<int> *PointIDCorres, vector<int> *PointCorres, vector<int> CumIDView, int totalPts);
void Save3DPoints(char *Path, Point3d *All3D, vector<int>Selected3DIndex);
void DisplayMatrix(char *Fname, Mat m);

void GetIntrinsicFromK(CameraData *AllViewsParas, vector<int> AvailViews);
void GetKFromIntrinsic(CameraData *AllViewsParas, vector<int> AvailViews);
void GetIntrinsicFromK(CameraData &camera);
void GetIntrinsicFromK(CameraData *AllViewsParas, int nviews);
void GetKFromIntrinsic(CameraData *AllViewsParas, int nviews);

void GetrtFromRT(CameraData *AllViewsParas, vector<int> AvailViews);
void GetrtFromRT(CameraData *AllViewsParas, int nviews);
void GetrtFromRT(double *rt, double *R, double *T);
void GetRTFromrt(CameraData *AllViewsParas, vector<int> AvailViews);
void GetRTFromrt(CameraData *AllViewsParas, int nviews);
void GetRTFromrt(double *rt, double *R, double *T);
void AssembleRT(double *R, double *T, double *RT);
void DesembleRT(double *R, double *T, double *RT);
void AssembleP(double *K, double *R, double *T, double *P);
void CopyCamereInfo(CameraData Src, CameraData &Dst);

void BlurDetectionDriver(char *Path, int nimages, int width, int height, float blurThresh);

int BAVisualSfMDriver(char *Path, char *nvmName, char *camInfo, char *IntrinsicInfo = NULL, bool lensconversion = 1, int sharedIntrinsics = 0);
bool loadNVMLite(const string filepath, Corpus &CorpusData, int width, int height, int sharedIntrinsics);
int SaveCorpusInfo(char *Path, Corpus &CorpusData, bool outputtext = false);
int ReadCorpusInfo(char *Path, Corpus &CorpusData, bool inputtext = false, bool notReadDescriptor = false);
bool loadIndividualNVMforpose(char *Path, CameraData *CameraInfo, vector<int>availViews, int timeIDstart, int timeIDstop, int nviews, bool sharedIntrinsics);

int GenerateCorpusVisualWords(char *Path, int nimages);
int ComputeWordsHistogram(char *Path, int nimages);
///////////////////////////////////////////////////////////////////
//DESCRIPTOR TYPE
typedef unsigned char	DTYPE;
//FEATURE LOCATION TYPE
typedef float LTYPE;

class FeatureData
{
	typedef struct sift_fileheader_v2
	{
		int	 szFeature;
		int  szVersion;
		int  npoint;
		int  nLocDim;
		int  nDesDim;
	}sift_fileheader_v2;
	typedef struct sift_fileheader_v1
	{
		int  npoint;
		int  nLocDim;
		int  nDesDim;
	}sift_fileheader_v1;

	enum
	{
		//		READ_BUFFER_SIZE = 0x100000,
		SIFT_NAME = ('S' + ('I' << 8) + ('F' << 16) + ('T' << 24)),
		MSER_NAME = ('M' + ('S' << 8) + ('E' << 16) + ('R' << 24)),
		RECT_NAME = ('R' + ('E' << 8) + ('C' << 16) + ('T' << 24)),
		//SIFT_VERSION_2=('V'+('2'<<8)+('.'<<16)+('0'<<24)),
		//SIFT_VERSION_3=('V'+('3'<<8)+('.'<<16)+('0'<<24)),
		SIFT_VERSION_4 = ('V' + ('4' << 8) + ('.' << 16) + ('0' << 24)),
		SIFT_VERSION_5 = ('V' + ('5' << 8) + ('.' << 16) + ('0' << 24)),
		SIFT_EOF = (0xff + ('E' << 8) + ('O' << 16) + ('F' << 24)),
	};

	//	static char readBuf[READ_BUFFER_SIZE];
	//	static char sift_version[8];
	static inline int IsValidFeatureName(int value)
	{
		return value == SIFT_NAME || value == MSER_NAME;
	}
	static inline int IsValidVersionName(int value)
	{
		return value == SIFT_VERSION_4 || value == SIFT_VERSION_5;
	}

public:
	class LocationData : public Points < LTYPE >
	{
	public:
		int              _file_version;
	public:
		//each feature has x,y, z, size, orientation
		//for rect feature, there is x, y, width, height
		//for eclips feature, there is u,v,a,b,c +z
	public:
		void glPaint2D(int style);
		LocationData(int d, int n) :Points<LTYPE>(d, n), _file_version(0){	};
		LocationData(LocationData& sup, int index[], int n) : Points<LTYPE>(sup, index, n), _file_version(0){};
		static void glPaintTexSIFT(const float * loc, float td[3]);
		static void glPaintTexFrontalVIP(const float * loc, float td[3]);
		static void glPaintSIFT(const LTYPE* loc);
		static void glPaintSIFTSQ(LTYPE*  loc);
		static void glPaintELIPS(LTYPE*  loc);
		static void glPaintRECT(LTYPE* loc);
		static void SetPaintColor(LTYPE sz);
	};

	class DescriptorData : public Points < DTYPE >
	{
		//generally d is 128
	public:
		DescriptorData(DescriptorData& sup, int index[], int n) : Points<DTYPE>(sup, index, n){};
		DescriptorData(int d, int n) :Points<DTYPE>(d, n){};
	};
	static float gSiftDisplayScale;
	static int	 gSiftVisualStyle;
protected:
	// the set of feature descriptors
	DescriptorData * _desData;
	// the set of feature locations
	LocationData *   _locData;
	int              _npoint;
	int				 _updated;
public:
	void SetUpdated(){ _updated = 1; }
	int  GetUpdated(){ return _updated; }
	void CopyToFeatureData(FeatureData &fd);
	int  appendSIFTB(const char* szFile, int pos);
	int  validate()	{ return _locData && _desData; }
	void ResizeFeatureData(int npoint, int locDim = 5, int desDim = 128)
	{
		if (npoint == 0)
		{
			if (_locData) delete _locData;
			if (_desData) delete _desData;
			_locData = NULL;
			_desData = NULL;
		}
		else
		{
			if (_locData)
				_locData->resize(locDim, npoint);
			else
				_locData = new LocationData(locDim, npoint);
			if (_desData)
				_desData->resize(desDim, npoint);
			else
				_desData = new DescriptorData(desDim, npoint);
			_locData->_file_version = SIFT_VERSION_4;
		}
		_npoint = npoint;

	}
	void operator = (FeatureData& ref) { ref.CopyToFeatureData(*this); }
	void ResizeLocationData(int npoint, int locDim)
	{
		if (npoint == 0)
		{
			if (_locData) delete _locData;
			if (_desData) delete _desData;
			_locData = NULL;
			_desData = NULL;
		}
		else
		{
			if (_locData)
				_locData->resize(locDim, npoint);
			else
				_locData = new LocationData(locDim, npoint);
			if (_desData)
			{
				delete _desData;
				_desData = NULL;
			}
			_locData->_file_version = SIFT_VERSION_4;
		}
	}

	void ShrinkLocationData(int ndim = 2, int npoint = -1);
	void ReleaseDescriptorData()
	{
		if (_desData)
		{
			delete _desData;
			_desData = NULL;
		}
	}
public:

	FeatureData();
	virtual ~FeatureData();
	void ReleaseFeatureData();
	DescriptorData&  getDescriptorData() { return *_desData; }
	LocationData&   getLocationData()const { return *_locData; }
	int		IsValidFeatureData(){ return getFeatureNum() > 0; }
	int		getFeatureNum(){ return _npoint; }
	int		getLoadedFeatureNum(){ return _locData ? _locData->npoint() : 0; }
	int  ReadSIFTB_LOCT(const char* szFile, int fmax);
	int  ReadSIFTB_DES(const char* szFile, int fmax);
	static int ReadSIFTB_DES(const char* szFile, unsigned char * buf, int nmax);
	static int ReadSIFTB_LOC(const char* szFile, float * buf, int &nmax);
	static int ReadSIFTB(const char* szFile, float * locbuf, unsigned char * desbuf);
	int  ReadSIFTB(const char* szFile);
	void saveSIFTB2(const char* szFile);
};


typedef FeatureData::LocationData FeatureLocationData;
typedef FeatureData::DescriptorData FeatureDescriptorData;

#endif
