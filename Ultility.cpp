#include "Ultility.h"
#include "ImagePro.h"
#include "Geometry.h"
#include "BAutil.h"
using namespace cv;
using namespace std;

#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>
float FeatureData::gSiftDisplayScale = 6.0f;
int FeatureData::gSiftVisualStyle = 0;

FeatureData::FeatureData()
{
	_desData = NULL;
	_locData = NULL;
	_updated = 0;
	_npoint = 0;
}
FeatureData::~FeatureData()
{

	if (_desData) delete _desData;
	if (_locData) delete _locData;
}
void FeatureData::ReleaseFeatureData()
{
	if (_locData) delete _locData;
	if (_desData) delete _desData;
	_locData = NULL;
	_desData = NULL;
	_npoint = 0;
}

void FeatureData::saveSIFTB2(const char* szFile)
{

	int i, j, sift_eof = SIFT_EOF;
	sift_fileheader_v2 sfh;
	int fd = _open(szFile, _O_BINARY | _O_CREAT | _O_WRONLY | _O_TRUNC, _S_IREAD | _S_IWRITE);
	if (fd < 0) return;

	///
	sfh.szFeature = SIFT_NAME;
	sfh.szVersion = SIFT_VERSION_4;
	sfh.npoint = _locData->npoint();
	sfh.nLocDim = _locData->ndim();
	sfh.nDesDim = _desData->ndim();
	_write(fd, &sfh, sizeof(sfh));
	////
	LTYPE * lp;
	DTYPE * dp;
	unsigned char* fph;
	float  *fp;
	unsigned char * ucp;
	int Max, MemNum;
	lp = _locData->data();
	MemNum = sfh.nDesDim * sfh.npoint;
	Max = sfh.npoint*sfh.nLocDim;
	fph = new unsigned char[MemNum]; //MemCollect::Malloc(MemNum);
	fp = (float*)fph;

	for (i = 0; i < sfh.npoint; i++)
	{
		lp = (*_locData)[i];
		for (j = 0; j < sfh.nLocDim; j++)
		{
			*fp++ = (float)*lp++;
		}
	}
	_write(fd, fph, sizeof(float)*Max);

	dp = _desData->data();
	Max = sfh.npoint*sfh.nDesDim;
	ucp = (unsigned char*)fph;
	for (i = 0; i < sfh.npoint; i++)
	{
		dp = (*_desData)[i];
		for (j = 0; j < sfh.nDesDim; j++, dp++)
		{
			*ucp++ = *dp;
		}
	}
	_write(fd, fph, sizeof(unsigned char)*Max);
	_write(fd, &sift_eof, sizeof(int));
	_close(fd);
}
int FeatureData::ReadSIFTB(const char* szFile)
{
	int name, version, npoint, nLocDim, nDesDim, sift_eof, sorted = 0;
	int fd = _open(szFile, _O_BINARY | _O_RDONLY, _S_IREAD);
	if (fd < 0)
		return 0;

	_read(fd, &name, sizeof(int));
	_read(fd, &version, sizeof(int));
	if (IsValidFeatureName(name) && IsValidVersionName(version))
	{
		//version 2 file
		_read(fd, &npoint, sizeof(int));
		_read(fd, &nLocDim, sizeof(int));
		_read(fd, &nDesDim, sizeof(int));
		if (npoint > 0 && nLocDim > 0 && nDesDim == SIFTBINS)
		{
			ResizeFeatureData(npoint, nLocDim, nDesDim);
			_read(fd, _locData->data(), nLocDim *npoint*sizeof(float));
			_read(fd, _desData->data(), nDesDim*npoint*sizeof(unsigned char));
			_read(fd, &sift_eof, sizeof(int));
			_close(fd);
			_locData->_file_version = version;
			SetUpdated();
		}
		else
		{
			ResizeFeatureData(0, 0, 0);
			_close(fd);
			return 0;
		}
		return 1;
	}
	else
	{
		_close(fd);
		return 0;
	}
}
int FeatureData::ReadSIFTB_LOC(const char* szFile, float * buf, int &nmax)
{
	sift_fileheader_v2 sfh;
	int fd = _open(szFile, _O_BINARY | _O_RDONLY, _S_IREAD);
	if (fd < 0)
		return 0;

	_read(fd, &sfh, sizeof(sfh));

	nmax = min(nmax, sfh.npoint);

	_read(fd, buf, sfh.nLocDim *sfh.npoint*sizeof(float));
	_close(fd);
	return nmax;
}
int FeatureData::ReadSIFTB_DES(const char* szFile, unsigned char * buf, int nmax)
{
	sift_fileheader_v2 sfh;
	int fd = _open(szFile, _O_BINARY | _O_RDONLY, _S_IREAD);
	if (fd < 0) return 0;
	///
	_read(fd, &sfh, sizeof(sfh));

	nmax = min(nmax, sfh.npoint);

	_lseek(fd, sfh.nLocDim *sfh.npoint*sizeof(float), SEEK_CUR);
	_read(fd, buf, sfh.nDesDim* nmax*sizeof(unsigned char));
	_close(fd);
	return nmax;
}
int FeatureData::ReadSIFTB(const char* szFile, float * locbuf, unsigned char * desbuf)
{
	sift_fileheader_v2 sfh;
	int fd = _open(szFile, _O_BINARY | _O_RDONLY, _S_IREAD);
	if (fd < 0) return 0;
	///
	_read(fd, &sfh, sizeof(sfh));

	_read(fd, locbuf, sfh.nLocDim *sfh.npoint*sizeof(float));
	_read(fd, desbuf, sfh.nDesDim* sfh.npoint*sizeof(unsigned char));
	_close(fd);
	return sfh.npoint;

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
	omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for
	for (int ii = 0; ii < length; ii++)
		mean += data[ii];
	return mean / length;
}
double MeanArray(double *data, int length)
{
	double mean = 0.0;
	omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for
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
double MeanArray(vector<double>data)
{
	double mean = 0.0;
	omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for
	for (int ii = 0; ii < data.size(); ii++)
		mean += data[ii];
	return mean / data.size();
}
double VarianceArray(vector<double>data, double mean)
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
				M.data[nchannels*ii + kk + nchannels*jj*width] = (unsigned char)(int)(Img[ii + (height - 1 - jj)*width + kk*length] + 0.5);

	return imwrite(fname, M);
}
bool SaveDataToImage(char *fname, double *Img, int width, int height, int nchannels)
{
	int ii, jj, kk, length = width*height;

	Mat M = Mat::zeros(height, width, nchannels == 1 ? CV_8UC1 : CV_8UC3);
	for (jj = 0; jj < height; jj++)
		for (ii = 0; ii < width; ii++)
			for (kk = 0; kk < nchannels; kk++)
				M.data[nchannels*ii + kk + nchannels*jj*width] = (unsigned char)(int)(Img[ii + (height - 1 - jj)*width + kk*length] + 0.5);

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
					sprintf(Fname, "%s/_%d.png", PATH, bestframeID);
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
int LensCorrectionVideoDriver(char *Path, char *VideoName, double *K, double *distortion, int LensType, int nimages, int interpAlgo)
{
	char Fname[200];
	double Imgscale = 1.0, Contscale = 1.0, iK[9];

	mat_invert(K, iK, 3);
	//double omega = 0.993183;

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
			printf("This lens model is not supported right now!");// FishEyeUndistortion(Img, cvImg.cols, cvImg.rows, 3, omega, CenterOfDistortionX, CenterOfDistortionY, interpAlgo, Imgscale, 1.0);
		else if (LensType == 2)
			printf("This lens model is not supported right now!");// FishEyeUndistortion(Img, cvImg.cols, cvImg.rows, 3, K, iK, omega, interpAlgo, Imgscale, Contscale);
		if (LensType)
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
int LensCorrectionDriver(char *Path, double *K, double *distortion, int LensType, int nimages, int interpAlgo)
{
	char Fname[200];
	double Imgscale = 1.0, Contscale = 1.0, iK[9];

	mat_invert(K, iK, 3);
	//double omega = 0.993183;

	Mat cvImg;
	unsigned char *Img = 0;
	double *Para = 0;
	for (int Id = 0; Id < nimages; Id++)
	{
		sprintf(Fname, "%s/D%d.png", Path, Id + 1);
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
			printf("This lens model is not supported right now!");// FishEyeUndistortion(Img, cvImg.cols, cvImg.rows, 3, omega, CenterOfDistortionX, CenterOfDistortionY, interpAlgo, Imgscale, 1.0);
		else if (LensType == 2)
			printf("This lens model is not supported right now!");// FishEyeUndistortion(Img, cvImg.cols, cvImg.rows, 3, K, iK, omega, interpAlgo, Imgscale, Contscale);
		if (LensType)
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

int ReadIntrinsicResults(char *path, CameraData *AllViewsParas, int nHDs)
{
	//Note that visCamualSfm use different lens model than openCV or matlab or yours (inverse model)
	char Fname[200];
	int id = 0;
	double fx, fy, skew, u0, v0, r0, r1, r2, t0, t1, p0, p1;

	sprintf(Fname, "%s/DevicesIntrinsics.txt", path); FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		cout << "Cannot load " << Fname << endl;
		return 1;
	}
	while (fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf ", &fx, &fy, &skew, &u0, &v0, &r0, &r1, &r2, &t0, &t1, &p0, &p1) != EOF)
	{
		AllViewsParas[id].K[0] = fx, AllViewsParas[id].K[1] = skew, AllViewsParas[id].K[2] = u0,
			AllViewsParas[id].K[3] = 0.0, AllViewsParas[id].K[4] = fy, AllViewsParas[id].K[5] = v0,
			AllViewsParas[id].K[6] = 0.0, AllViewsParas[id].K[7] = 0.0, AllViewsParas[id].K[8] = 1.0;

		GetIntrinsicFromK(AllViewsParas[id]);
		//mat_invert(AllViewsParas[id].K, AllViewsParas[id].iK);

		AllViewsParas[id].distortion[0] = r0, AllViewsParas[id].distortion[1] = r1, AllViewsParas[id].distortion[2] = r2;
		AllViewsParas[id].distortion[3] = t0, AllViewsParas[id].distortion[4] = t1;
		AllViewsParas[id].distortion[5] = p0, AllViewsParas[id].distortion[6] = p1;
		id++;
	}
	fclose(fp);

	return 0;
}
int SaveIntrinsicResults(char *path, CameraData *AllViewsParas, int nCams)
{
	//Note that visCamualSfm use different lens model than openCV or matlab or yours (inverse model)
	char Fname[200];
	int id = 0;
	double fx, fy, skew, u0, v0, r0, r1, r2, t0, t1, p0, p1;

	sprintf(Fname, "%s/R_DevicesIntrinsics.txt", path); FILE *fp = fopen(Fname, "w+");
	if (fp == NULL)
	{
		cout << "Cannot load " << Fname << endl;
		return 1;
	}
	for (int ii = 0; ii < nCams; ii++)
	{
		fx = AllViewsParas[id].K[0], fy = AllViewsParas[id].K[4], skew = AllViewsParas[id].K[1], u0 = AllViewsParas[id].K[2], v0 = AllViewsParas[id].K[5];

		r0 = AllViewsParas[id].distortion[0], r1 = AllViewsParas[id].distortion[1], r2 = AllViewsParas[id].distortion[2];
		t0 = AllViewsParas[id].distortion[3], t1 = AllViewsParas[id].distortion[4];
		p0 = AllViewsParas[id].distortion[5], p1 = AllViewsParas[id].distortion[6];
		fprintf(fp, "%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\n", fx, fy, skew, u0, v0, r0, r1, r2, t0, t1, p0, p1);
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
void ReadCumulativePoints(char *Path, int nviews, int timeID, vector<int>&cumulativePts)
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
		abort();
	}
	for (ii = 0; ii < nviews + 1; ii++)
	{
		fscanf(fp, "%d\n", &jj);
		cumulativePts.push_back(jj);
	}
	fclose(fp);

	return;
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
void GetIntrinsicFromK(CameraData &camera)
{
	camera.intrinsic[0] = camera.K[0];
	camera.intrinsic[1] = camera.K[4];
	camera.intrinsic[2] = camera.K[1];
	camera.intrinsic[3] = camera.K[2];
	camera.intrinsic[4] = camera.K[5];
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
void AssembleP(double *K, double *R, double *T, double *P)
{
	double RT[12];
	Set_Sub_Mat(R, RT, 3, 3, 4, 0, 0);
	Set_Sub_Mat(T, RT, 1, 3, 4, 3, 0);
	mat_mul(K, RT, P, 3, 3, 4);
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
}
void BlurDetectionDriver(char *Path, int nimages, int width, int height, float blurThresh)
{
	char Fname[200];
	Mat cvImg;
	unsigned char *Img = new unsigned char[width*height];
	vector<int>blurredImgVector; blurredImgVector.reserve(nimages);

	for (int kk = 0; kk < nimages; kk++)
	{
		sprintf(Fname, "%s/ (%d).png", Path, kk + 1);
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
			sprintf(Fname, "%s/ (%d).png", Path, kk + 1);
			cvImg = imread(Fname, 1);
			sprintf(Fname, "%s/NB(%d).png", Path, kk + 1);
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

bool loadNVMLite(const string filepath, Corpus &CorpusData, int width, int height, int sharedIntrinsics)
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
		if (strcmp(str, "7_50") == 0)
		{
			int viewID = 190;

			ceres::QuaternionToRotation(Quaterunion, CorpusData.camera[viewID].R);
			mat_mul(CorpusData.camera[viewID].R, CamCenter, T, 3, 3, 1); //t = -RC
			CorpusData.camera[viewID].T[0] = -T[0], CorpusData.camera[viewID].T[1] = -T[1], CorpusData.camera[viewID].T[2] = -T[2];
		}
		else
		{
			int viewID = atoi(str) - 1;

			ceres::QuaternionToRotation(Quaterunion, CorpusData.camera[viewID].R);
			mat_mul(CorpusData.camera[viewID].R, CamCenter, T, 3, 3, 1); //t = -RC
			CorpusData.camera[viewID].T[0] = -T[0], CorpusData.camera[viewID].T[1] = -T[1], CorpusData.camera[viewID].T[2] = -T[2];
		}
	}


	return true;
}
int SaveCorpusInfo(char *Path, Corpus &CorpusData, bool notbinary)
{
	int ii, jj, kk;
	char Fname[200];
	sprintf(Fname, "%s/Corpus.txt", Path);
	FILE *fp = fopen(Fname, "w+");
	CorpusData.n3dPoints = CorpusData.xyz.size();
	fprintf(fp, "%d %d ", CorpusData.nCamera, CorpusData.n3dPoints);

	//xyz rgb viewid3D pointid3D 3dId2D cumpoint
	if (CorpusData.rgb.size() == 0)
	{
		fprintf(fp, "0\n");
		for (jj = 0; jj < CorpusData.xyz.size(); jj++)
			fprintf(fp, "%.8f %.8f %.8f \n", CorpusData.xyz.at(jj).x, CorpusData.xyz.at(jj).y, CorpusData.xyz.at(jj).z);
	}
	else
	{
		fprintf(fp, "1\n");
		for (jj = 0; jj < CorpusData.xyz.size(); jj++)
			fprintf(fp, "%.8f %.8f %.8f %d %d %d\n", CorpusData.xyz.at(jj).x, CorpusData.xyz.at(jj).y, CorpusData.xyz.at(jj).z, CorpusData.rgb.at(jj).x, CorpusData.rgb.at(jj).y, CorpusData.rgb.at(jj).z);
	}

	for (jj = 0; jj < CorpusData.n3dPoints; jj++)
	{
		int nviews = CorpusData.viewIdAll3D.at(jj).size();
		fprintf(fp, "%d ", nviews);
		for (ii = 0; ii < nviews; ii++)
			fprintf(fp, "%d ", CorpusData.viewIdAll3D.at(jj).at(ii));
		fprintf(fp, "\n");
	}

	for (jj = 0; jj < CorpusData.n3dPoints; jj++)
	{
		int npts = CorpusData.pointIdAll3D.at(jj).size();
		fprintf(fp, "%d ", npts);
		for (ii = 0; ii < npts; ii++)
			fprintf(fp, "%d ", CorpusData.pointIdAll3D.at(jj).at(ii));
		fprintf(fp, "\n");
	}

	for (jj = 0; jj < CorpusData.n3dPoints; jj++)
	{
		int npts = CorpusData.uvAll3D.at(jj).size();
		fprintf(fp, "%d ", npts);
		for (ii = 0; ii < npts; ii++)
			fprintf(fp, "%.8f %.8f ", CorpusData.uvAll3D.at(jj).at(ii).x, CorpusData.uvAll3D.at(jj).at(ii).y);
		fprintf(fp, "\n");
	}

	for (jj = 0; jj < CorpusData.nCamera; jj++)
	{
		int n3D = CorpusData.threeDIdAllViews.at(jj).size();
		fprintf(fp, "%d\n", n3D);
		for (ii = 0; ii < n3D; ii++)
			fprintf(fp, "%d ", CorpusData.threeDIdAllViews.at(jj).at(ii));
		fprintf(fp, "\n");
	}

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
	sprintf(Fname, "%s/Corpus.txt", Path);
	FILE *fp = fopen(Fname, "r");
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

	
	for (int ii = 0; ii < nimages; ii ++)//= nimages / 5)
	{
		keypoints.clear();
		sprintf(Fname, "%s/%d.jpg", Path, ii);
		img = imread(Fname, CV_LOAD_IMAGE_GRAYSCALE); 		
		detector.detect(img, keypoints);
		detector.compute(img, keypoints, descriptors);
		featuresUnclustered.push_back(descriptors);
		printf("%.2f %%percent done\n", 100.0*ii/nimages);
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
	for (int ii = 0; ii < nimages; ii ++)
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















