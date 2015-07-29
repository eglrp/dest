#include "Ultility.h"
#include "ImagePro.h"
#include "Geometry.h"
#include "BAutil.h"

using namespace std;
using namespace cv;
using namespace Eigen;

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

void makeDir(char *Fname)
{
#ifdef _WIN32
	mkdir(Fname);
#else
	mkdir(Fname, 0755);
#endif
	return;
}

char LOG_FILE_NAME[512];
void printfLog(const char *strLog, char *Path, ...)
{
#ifdef _WIN32
	static bool FirstTimeLog = true;

	ofstream fout;
	char finalLog[1024];
	va_list marker;
	SYSTEMTIME st;

	va_start(marker, strLog);
	vsprintf(finalLog, strLog, marker);
	printf("%s", finalLog);

	if (FirstTimeLog)
	{
		GetLocalTime(&st);
		if (Path == NULL)
			sprintf(LOG_FILE_NAME, "%d_%d_%d.txt", st.wDay, st.wHour, st.wMinute);
		else
			sprintf(LOG_FILE_NAME, "%s/%d_%d_%d.txt", Path, st.wDay, st.wHour, st.wMinute);
		fout.open(LOG_FILE_NAME, std::ios_base::trunc);

		char startLog[1024];
		sprintf(startLog, "Log Start(%d/%d %d:%d) \n", st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
		fout << startLog;

		FirstTimeLog = false;
	}
	else
		fout.open(LOG_FILE_NAME, std::ios_base::app);

	fout << finalLog;
	fout.close();


#else
	char finalLog[1024];
	va_list marker;

	va_start(marker, strLog);
	vsprintf(finalLog, strLog, marker);
	printf("%s", finalLog);
#endif
}

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

	IplImage* correspond = cvCreateImage(cvSize(Img1->width + Img2->width, max(Img1->height, Img2->height)), 8, nchannels);
	cvSetImageROI(correspond, cvRect(0, 0, Img1->width, Img1->height));	cvCopy(Img1, correspond);
	cvSetImageROI(correspond, cvRect(Img1->width, 0, Img2->width, Img2->height));	cvCopy(Img2, correspond);
	cvResetImageROI(correspond);
	DisplayImageCorrespondence(correspond, Img1->width, 0, Keys1, Keys2, CorresID, fractionMatchesDisplayed);

	delete[] match_buf;
	delete sift;
	delete matcher;
	FREE_MYLIB(hsiftgpu);

	return 0;
}

//DCT
void GenerateDCTBasis(int nsamples, double *Basis, double *Weight)
{
	if (Basis != NULL)
	{
		double s = sqrt(1.0 / nsamples);
		for (int ll = 0; ll < nsamples; ll++)
			Basis[ll] = s;

		for (int kk = 1; kk < nsamples; kk++)
		{
			double s = sqrt(2.0 / nsamples);
			for (int ll = 0; ll < nsamples; ll++)
				Basis[kk*nsamples + ll] = s*cos(Pi*kk *(1.0*ll - 0.5) / nsamples);
		}
	}

	if (Weight != NULL)
	{
		for (int ll = 0; ll < nsamples; ll++)
			Weight[ll] = 2.0*(cos(Pi*(ll - 1) / nsamples) - 1.0);
	}

	return;
}
void GenerateiDCTBasis(double *Basis, int nsamples, double t)
{
	Basis[0] = sqrt(1.0 / nsamples);

	double s = sqrt(2.0 / nsamples);
	for (int kk = 1; kk < nsamples; kk++)
		Basis[kk] = s*cos(Pi*kk *(t + 0.5) / nsamples);

	return;
}

//BSpline
void GenerateSplineBasisWithBreakPts(double *Basis, double *DBasis, double *ResampledPts, double *BreakPts, int nResamples, int nbreaks, int SplineOrder, int DerivativeOrder)
{
	if (ResampledPts[nResamples - 1] > BreakPts[nbreaks - 1])
	{
		cout << "Element of the resampled data is out the break point range!" << endl;
		abort();
	}

	const int ncoeffs = nbreaks + SplineOrder - 2;
	gsl_bspline_workspace *bw = gsl_bspline_alloc(SplineOrder, nbreaks);
	gsl_vector *gsl_BreakPts = gsl_vector_alloc(nbreaks);

	gsl_bspline_deriv_workspace *dw = dw = gsl_bspline_deriv_alloc(SplineOrder);
	gsl_matrix  *dBi = gsl_matrix_alloc(ncoeffs, DerivativeOrder + 1);

	//copy data to gsl format
	for (int ii = 0; ii < nbreaks; ii++)
		gsl_vector_set(gsl_BreakPts, ii, BreakPts[ii]);

	//contruct knots
	gsl_bspline_knots(gsl_BreakPts, bw);

	//construct basis matrix
	for (int ii = 0; ii < nResamples; ii++)
	{
		gsl_bspline_deriv_eval(ResampledPts[ii], DerivativeOrder, dBi, bw, dw);//compute basis vector for point i

		if (Basis != NULL) //sometimes, the 0th order is not needed
			for (int jj = 0; jj < ncoeffs; jj++)
				Basis[jj + ii*ncoeffs] = gsl_matrix_get(dBi, jj, 0);

		if (DerivativeOrder == 1)
			for (int jj = 0; jj < ncoeffs; jj++)
				DBasis[jj + ii*ncoeffs] = gsl_matrix_get(dBi, jj, 1);
	}

	/*FILE	*fp = fopen("C:/temp/gsl_B.txt", "w + ");
	for (int jj = 0; jj < nResamples; jj++)
	{
	for (int ii = 0; ii < ncoeffs; ii++)
	fprintf(fp, "%.16f ", Basis[ii + jj*ncoeffs]);
	fprintf(fp, "\n");
	}
	fclose(fp);

	fp = fopen("C:/temp/gsl_dB.txt", "w + ");
	for (int jj = 0; jj < nResamples; jj++)
	{
	for(int ii = 0; ii < ncoeffs; ii++)
	fprintf(fp, "%.16f ", DBasis[ii + jj*ncoeffs]);
	fprintf(fp, "\n");
	}
	fclose(fp);*/

	gsl_bspline_free(bw);
	gsl_bspline_deriv_free(dw);
	gsl_matrix_free(dBi);

	return;
}
void GenerateResamplingSplineBasisWithBreakPts(double *Basis, double *ResampledPts, double *BreakPts, int nResamples, int nbreaks, int SplineOrder)
{
	if (ResampledPts[nResamples - 1] > BreakPts[nbreaks - 1])
	{
		cout << "Element of the resampled data is out the break point range!" << endl;
		abort();
	}

	gsl_bspline_workspace *bw = gsl_bspline_alloc(SplineOrder, nbreaks);

	gsl_vector *Bi = gsl_vector_alloc(nbreaks + SplineOrder - 2);
	gsl_vector *gsl_BreakPts = gsl_vector_alloc(nbreaks);
	//gsl_matrix *gsl_Basis = gsl_matrix_alloc(nResamples, nbreaks + SplineOrder - 2);

	//copy data to gsl format
	for (int ii = 0; ii < nbreaks; ii++)
		gsl_vector_set(gsl_BreakPts, ii, BreakPts[ii]);

	//contruct knots
	gsl_bspline_knots(gsl_BreakPts, bw);

	//construct basis matrix
	int ncoeffs = nbreaks + 2;
	for (int ii = 0; ii < nResamples; ii++)
	{
		gsl_bspline_eval(ResampledPts[ii], Bi, bw); //compute basis vector for point i

		//for (int jj = 0; jj < ncoeffs; jj++)
		//	gsl_matrix_set(gsl_Basis, ii, jj, gsl_vector_get(Bi, jj));

		for (int jj = 0; jj < ncoeffs; jj++)
			Basis[jj + ii*ncoeffs] = gsl_vector_get(Bi, jj);
	}

	/*FILE *fp = fopen("C:/temp/gsl_breakpts.txt", "w + ");
	for (int ii = 0; ii < nbreaks; ii++)
	fprintf(fp, "%.16f \n", gsl_vector_get(gsl_BreakPts, ii));
	fclose(fp);

	fp = fopen("C:/temp/gsl_resampled.txt", "w + ");
	for (int ii = 0; ii < nResamples; ii++)
	fprintf(fp, "%.16f \n", gsl_vector_get(gsl_ResampledPts, ii));
	fclose(fp);

	fp = fopen("C:/temp/gsl_knots.txt", "w + ");
	for (int ii = 0; ii < nbreaks + 2 * SplineOrder - 2; ii++)
	fprintf(fp, "%.16f \n", gsl_vector_get(bw->knots, ii));
	fclose(fp);

	fp = fopen("C:/temp/gsl_B.txt", "w + ");
	for (int jj = 0; jj < nResamples; jj++)
	{
	for (int ii = 0; ii < nbreaks + 2; ii++)
	fprintf(fp, "%.16f ", gsl_matrix_get(gsl_Basis, jj, ii));
	fprintf(fp, "\n");
	}
	fclose(fp);*/

	gsl_bspline_free(bw);
	gsl_vector_free(Bi);
	//gsl_matrix_free(gsl_Basis);

	return;
}
void GenerateResamplingSplineBasisWithBreakPts(double *Basis, vector<double> ResampledPts, vector<double>BreakPts, int SplineOrder)
{
	int nResamples = (int)ResampledPts.size(), nbreaks = (int)BreakPts.size();
	if (ResampledPts[nResamples - 1] > BreakPts[nbreaks - 1])
	{
		//cout << "Element of the resampled data is out the break point range!" << endl;
		abort();
	}

	gsl_bspline_workspace *bw = gsl_bspline_alloc(SplineOrder, nbreaks);

	gsl_vector *Bi = gsl_vector_alloc(nbreaks + 2);
	gsl_vector *gsl_BreakPts = gsl_vector_alloc(nbreaks);
	gsl_matrix *gsl_Basis = gsl_matrix_alloc(nResamples, nbreaks + 2);

	//copy data to gsl format
	for (int ii = 0; ii < nbreaks; ii++)
		gsl_vector_set(gsl_BreakPts, ii, BreakPts[ii]);

	//contruct knots
	gsl_bspline_knots(gsl_BreakPts, bw);

	//construct basis matrix
	for (int ii = 0; ii < nResamples; ii++)
	{
		gsl_bspline_eval(ResampledPts[ii], Bi, bw); //compute basis vector for point i

		for (int jj = 0; jj < nbreaks + 2; jj++)
			gsl_matrix_set(gsl_Basis, ii, jj, gsl_vector_get(Bi, jj));
	}

	for (int jj = 0; jj < nResamples; jj++)
		for (int ii = 0; ii < nbreaks + 2; ii++)
			Basis[ii + jj*(nbreaks + 2)] = gsl_matrix_get(gsl_Basis, jj, ii);

	/*FILE *fp = fopen("C:/temp/gsl_breakpts.txt", "w + ");
	for (int ii = 0; ii < nbreaks; ii++)
	fprintf(fp, "%.16f \n", gsl_vector_get(gsl_BreakPts, ii));
	fclose(fp);

	fp = fopen("C:/temp/gsl_resampled.txt", "w + ");
	for (int ii = 0; ii < nResamples; ii++)
	fprintf(fp, "%.16f \n", gsl_vector_get(gsl_ResampledPts, ii));
	fclose(fp);

	fp = fopen("C:/temp/gsl_knots.txt", "w + ");
	for (int ii = 0; ii < nbreaks + 2 * SplineOrder - 2; ii++)
	fprintf(fp, "%.16f \n", gsl_vector_get(bw->knots, ii));
	fclose(fp);

	fp = fopen("C:/temp/gsl_B.txt", "w + ");
	for (int jj = 0; jj < nResamples; jj++)
	{
	for (int ii = 0; ii < nbreaks + 2; ii++)
	fprintf(fp, "%.16f ", gsl_matrix_get(gsl_Basis, jj, ii));
	fprintf(fp, "\n");
	}
	fclose(fp);*/
	gsl_bspline_free(bw);
	gsl_vector_free(Bi);
	gsl_matrix_free(gsl_Basis);

	return;
}
void GenerateResamplingSplineBasisWithBreakPtsExample()
{
	int nResamples = 200, nbreaks = 15, SplineOrder = 4;
	double *Basis = new double[nResamples*(nbreaks + 2)];
	double *ResampledPts = new double[nResamples];
	double *BreakPts = new double[nbreaks];

	for (int ii = 0; ii < nbreaks; ii++)
		BreakPts[ii] = 15.0*ii / 14.0;

	for (int ii = 0; ii < nResamples; ii++)
		ResampledPts[ii] = (15.0 / (nResamples - 1)) * ii;

	GenerateResamplingSplineBasisWithBreakPts(Basis, ResampledPts, BreakPts, nResamples, nbreaks, SplineOrder);

	FILE *fp = fopen("C:/temp/gsl_B.txt", "w + ");
	for (int jj = 0; jj < nResamples; jj++)
	{
		for (int ii = 0; ii < nbreaks + 2; ii++)
			fprintf(fp, "%.16f ", Basis[ii + jj*(nbreaks + 2)]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	return;
}
int FindActingControlPts(double t, int *ActingID, int ncontrols, gsl_bspline_workspace *bw, gsl_vector *Bi, int splineOrder, int extraNControls)
{
	gsl_bspline_eval(t, Bi, bw);

	int startID = -1;
	for (int ii = 0; ii < ncontrols; ii++)
	{
		double bi = gsl_vector_get(Bi, ii);
		if (bi > 1.0e-9)
		{
			startID = ii;
			break;
		}
	}

	//add points at the two ends of acting controls just in case
	while (startID - extraNControls / 2 < 0) //at the begining,  ---> need to decrese starting point
		startID++;
	while (startID + splineOrder + extraNControls / 2 > ncontrols) //at the end, #control points is less than SplineOrder ---> need to decrese starting point
		startID--;

	for (int ii = 0; ii < splineOrder + extraNControls; ii++) //assume # controls >> Splineorder + extraNControl
		ActingID[ii] = startID - extraNControls / 2 + ii;

	return 0;
}

static void BSplineLVB(const double * t, const int jhigh, const int index, const double x, const int left, int * j, double * deltal, double * deltar, double * biatx)
{
	int i;
	double saved;
	double term;

	if (index == 1)
	{
		*j = 0;
		biatx[0] = 1.0;
	}

	for (; *j < jhigh - 1; *j += 1)
	{
		deltar[*j] = t[left + *j + 1] - x;
		deltal[*j] = x - t[left - *j];

		saved = 0.0;

		for (i = 0; i <= *j; i++)
		{
			term = biatx[i] / (deltar[i] + deltal[*j - i]);

			biatx[i] = saved + deltar[i] * term;

			saved = deltal[*j - i] * term;
		}

		biatx[*j + 1] = saved;
	}

	return;
}
static void BSplineLVD(const double * knots, const int SplineOrder, const double x, const int left, double * deltal, double * deltar, double * a, double * dbiatx, const int nderiv)
{
	int i, ideriv, il, j, jlow, jp1mid, kmm, ldummy, m, mhigh;
	double factor, fkmm, sum;

	int bsplvb_j;
	double *dbcol = dbiatx;

	mhigh = min(nderiv, SplineOrder - 1);
	BSplineLVB(knots, SplineOrder - mhigh, 1, x, left, &bsplvb_j, deltal, deltar, dbcol);
	if (mhigh > 0)
	{
		ideriv = mhigh;
		for (m = 1; m <= mhigh; m++)
		{
			for (j = ideriv, jp1mid = 0; j < (int)SplineOrder; j++, jp1mid++)
				dbiatx[j + ideriv*SplineOrder] = dbiatx[jp1mid];

			ideriv--;
			BSplineLVB(knots, SplineOrder - ideriv, 2, x, left, &bsplvb_j, deltal, deltar, dbcol);
		}

		jlow = 0;
		for (i = 0; i < (int)SplineOrder; i++)
		{
			for (j = jlow; j < (int)SplineOrder; j++)
				a[i + j*SplineOrder] = 0.0;
			jlow = i;
			a[i + i*SplineOrder] = 1.0;
		}

		for (m = 1; m <= mhigh; m++)
		{
			kmm = SplineOrder - m;
			fkmm = (float)kmm;
			il = left;
			i = SplineOrder - 1;

			for (ldummy = 0; ldummy < kmm; ldummy++)
			{
				factor = fkmm / (knots[il + kmm] - knots[il]);

				for (j = 0; j <= i; j++)
					a[j + i*SplineOrder] = factor*(a[j + i*SplineOrder] - a[j + (i - 1)*SplineOrder]);

				il--;
				i--;
			}

			for (i = 0; i < (int)SplineOrder; i++)
			{
				sum = 0;
				jlow = max(i, m);
				for (j = jlow; j < (int)SplineOrder; j++)
					sum += a[i + j*SplineOrder] * dbiatx[j + m*SplineOrder];

				dbiatx[i + m*SplineOrder] = sum;
			}
		}
	}

	return;

}
void BSplineFindActiveCtrl(int *ActingID, const double x, double *knots, int nbreaks, int nControls, int SplineOrder, int extraNControls)
{
	int i;
	int nknots = nControls + SplineOrder, nPolyPieces = nbreaks - 1;

	// find i such that t_i <= x < t_{i+1} 
	for (i = SplineOrder - 1; i < SplineOrder + nPolyPieces - 1; i++)
	{
		const double ti = knots[i];
		const double tip1 = knots[i + 1];

		if (ti <= x && x < tip1)
			break;
		if (ti < x && x == tip1 && tip1 == knots[SplineOrder + nPolyPieces - 1])
			break;
	}

	int startID = i - SplineOrder + 1;
	while (startID - extraNControls / 2 < 0) //at the begining,  ---> need to decrese starting point
		startID++;
	while (startID + SplineOrder + extraNControls / 2 > nControls) //at the end, #control points is less than SplineOrder ---> need to decrese starting point
		startID--;

	startID = startID - extraNControls / 2;
	for (int ii = 0; ii < SplineOrder + extraNControls; ii++)
		ActingID[ii] = startID + ii;

	return;
}
static inline int BSplineFindInterval(const double x, int *flag, double *knots, int nbreaks, int nControls, int SplineOrder)
{
	int i;
	int nknots = nControls + SplineOrder, nPolyPieces = nbreaks - 1;
	if (x < knots[0])
	{
		*flag = -1;
		return 0;
	}

	// find i such that t_i <= x < t_{i+1} 
	for (i = SplineOrder - 1; i < SplineOrder + nPolyPieces - 1; i++)
	{
		const double ti = knots[i];
		const double tip1 = knots[i + 1];

		if (tip1 < ti)
		{
			printf("knots vector is not increasing"); abort();
		}

		if (ti <= x && x < tip1)
			break;

		if (ti < x && x == tip1 && tip1 == knots[SplineOrder + nPolyPieces - 1])//if (ti < x && x == tip1 && tip1 == gsl_vector_get(knots, SplineOrder + nPolyPieces	- 1))
			break;
	}

	if (i == SplineOrder + nPolyPieces - 1)
		*flag = 1;
	else
		*flag = 0;

	return i;
}
static inline int BSplineEvalInterval(const double x, int * i, const int flag, double *knots, int nbreaks, int nControls, int SplineOrder)
{
	if (flag == -1)
	{
		printf("x outside of knot interval"); abort();
	}
	else if (flag == 1)
	{
		if (x <= knots[*i] + DBL_EPSILON)
			*i -= 1;
		else
		{
			printf("x outside of knot interval"); abort();
		}
	}

	if (knots[*i] == knots[*i + 1])
	{
		printf("knot(i) = knot(i+1) will result in division by zero"); abort();
	}

	return 0;
}
int BSplineGetKnots(double *knots, double *BreakLoc, int nbreaks, int nControls, int SplineOrder)
{
	int i;
	for (i = 0; i < SplineOrder; i++)
		knots[i] = BreakLoc[0];

	int nPolyPieces = nbreaks - 1;
	for (i = 1; i < nPolyPieces; i++)
		knots[i + SplineOrder - 1] = BreakLoc[i];

	for (i = nControls; i < nControls + SplineOrder; i++)
		knots[i] = BreakLoc[nPolyPieces];
	return 0;
}
int BSplineGetNonZeroBasis(const double x, double * dB, int * istart, int * iend, double *knots, int nbreaks, int nControls, int SplineOrder, const int nderiv)
{
	int flag = 0;

	int i = BSplineFindInterval(x, &flag, knots, nbreaks, nControls, SplineOrder);
	int error = BSplineEvalInterval(x, &i, flag, knots, nbreaks, nControls, SplineOrder);
	if (error)
		return error;

	*istart = i - SplineOrder + 1;
	*iend = i;

	double deltal[4], deltar[4], A[16];//Assuming cubi B spline
	BSplineLVD(knots, SplineOrder, x, *iend, deltal, deltar, A, dB, nderiv);

	return 0;
}
int BSplineGetBasis(const double x, double * B, double *knots, int nbreaks, int nControls, int SplineOrder, const int nderiv)
{
	int i, j, istart, iend, error;
	double Bi[4 * 3];//up to 2nd der of cubic spline

	error = BSplineGetNonZeroBasis(x, Bi, &istart, &iend, knots, nbreaks, nControls, SplineOrder, nderiv);
	if (error)
		return error;

	for (j = 0; j <= nderiv; j++)
	{
		for (i = 0; i < istart; i++)
			B[i + j*nControls] = 0.0;
		for (i = istart; i <= iend; i++)
			B[i + j*nControls] = Bi[(i - istart) + j*SplineOrder];
		for (i = iend + 1; i < nControls; i++)
			B[i + j*nControls] = 0.0;
	}

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
		cout << '\a';

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

	if (k < n - k) // eg: Choose(100,3)
	{
		delta = 1.0*(n - k);
		iMax = k;
	}
	else         // eg: Choose(100,97)
	{
		delta = 1.0*k;
		iMax = n - k;
	}

	double res = delta + 1.0;
	for (int i = 2; i <= iMax; i++)
		res = res * (delta + i) / i;

	return res;
}
double nPermutek(int n, int k)
{
	double res = n;
	for (int ii = 0; ii < k; ii++)
		res *= 1.0*(n - ii);
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

double Distance2D(Point2d X, Point2d Y)
{
	Point2d Dif = X - Y;
	return sqrt(Dif.x*Dif.x + Dif.y * Dif.y);
}
double Distance3D(Point3d X, Point3d Y)
{
	Point3d Dif = X - Y;
	return sqrt(Dif.x*Dif.x + Dif.y * Dif.y + Dif.z *Dif.z);
}
double Distance3D(double *X, double * Y)
{
	Point3d Dif(X[0] - Y[0], X[1] - Y[1], X[2] - Y[2]);
	return sqrt(Dif.x*Dif.x + Dif.y * Dif.y + Dif.z *Dif.z);
}
Point3d ProductPoint3d(Point3d X, Point3d Y)
{
	return Point3d(X.x*Y.x, X.y*Y.y, X.z*Y.z);
}
Point3d DividePoint3d(Point3d X, Point3d Y)
{
	return Point3d(X.x / Y.x, X.y / Y.y, X.z / Y.z);
}
double L1norm(vector<double>A)
{
	double res = 0.0;
	for (int ii = 0; ii < A.size(); ii++)
		res += abs(A[ii]);
	return res;
}
double L2norm(double *A, int dim)
{
	double res = 0.0;
	for (int ii = 0; ii < dim; ii++)
		res += A[ii] * A[ii];
	return sqrt(res);
}
void normalize(double *x, int dim)
{
	double tt = 0;
	for (int ii = 0; ii < dim; ii++)
		tt += x[ii] * x[ii];
	tt = sqrt(tt);
	if (tt < FLT_EPSILON)
		return;
	for (int ii = 0; ii < dim; ii++)
		x[ii] = x[ii] / tt;
	return;
}
float MeanArray(float *data, int length)
{
	double mean = 0.0;
	for (int ii = 0; ii < length; ii++)
		mean += data[ii];
	return (float)(mean / length);
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
double dotProduct(double *x, double *y, int dim)
{
	double res = 0.0;
	for (int ii = 0; ii < dim; ii++)
		res += x[ii] * y[ii];
	return res;
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
void XCORR1D(float *s, const int sdim, float *b, const int bdim, float *res)
{
	Mat ms(1, bdim, CV_32F, Scalar(0.0));
	Mat mb(1, bdim * 3 - 2, CV_32F, Scalar(0.0));
	Mat result(1, 2 * bdim - 1, CV_32F);

#pragma omp parallel for
	for (int ii = 0; ii < sdim; ii++)
		ms.at<float>(ii) = s[ii];

#pragma omp parallel for
	for (int ii = 0; ii < bdim; ii++)
		mb.at<float>(ii + bdim - 1) = b[ii];

	matchTemplate(mb, ms, result, CV_TM_CCORR);

#pragma omp parallel for
	for (int ii = 0; ii < 2 * bdim - 1; ii++)
		res[ii] = mb.at<float>(ii);

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
void mat_mul(float *aa, float *bb, float *out, int rowa, int col_row, int colb)
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
void mat_mul_symetric(double *A, double *B, int row, int column)
{
	for (int I = 0; I < row; I++)
	{
		for (int J = 0; J < row; J++)
		{
			if (J < I)
				continue;

			B[I*row + J] = 0.0;
			for (int K = 0; K < column; K++)
				B[I*row + J] += A[I*column + K] * A[J*column + K];
		}
	}

	return;
}
void mat_add_symetric(double *A, double * B, double *C, int row, int column)
{
	for (int I = 0; I < row; I++)
	{
		for (int J = 0; J < column; J++)
		{
			if (J < I)
				continue;

			C[I*column + J] = A[I*column + J] + B[I*column + J];
		}
	}

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
void RescaleMat(double *mat, double nmin, double nmax, int length)
{
	double minv = 9e9, maxv = -9e9;
	for (int ii = 0; ii < length; ii++)
	{
		if (mat[ii] > maxv)
			maxv = mat[ii];
		if (mat[ii] < minv)
			minv = mat[ii];
	}

	for (int ii = 0; ii < length; ii++)
		mat[ii] = (mat[ii] - minv) / (maxv - minv)*(nmax - nmin) + nmin;
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
				*(A + j*n + i) += (*(lpA + k*n + i))*(*(lpA + k*n + j));
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

double SimpsonThreeEightIntegration(double *y, double step, int npts)
{
	if (npts < 4)
	{
		printf("Not enough supprintg points (4) for this integration method. Abort()");
		abort();
	}

	if (npts == 4)
		return  step*(3.0 / 8.0*y[0] + 9.0 / 8.0*y[1] + 9.0 / 8.0*y[2] + 3.0 / 8.0*y[3]);


	double result = 3.0 / 8.0*y[0] + 7.0 / 6.0*y[1] + 23.0 / 24.0*y[2] + 23.0 / 24.0*y[npts - 3] + 7.0 / 6.0*y[npts - 2] + 3.0 / 8.0*y[npts - 1];
	for (int ii = 3; ii < npts - 3; ii++)
		result += y[ii];

	return step*result;
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

int ReadDomeCalibFile(char *Path, CameraData *AllCamInfo)
{
	const int nHDs = 30, nVGAs = 480, nPanels = 20, nCamsPanel = 24;
	char Fname[200];

	double Quaterunion[4], CamCenter[3], T[3];
	for (int camID = 0; camID < nHDs; camID++)
	{
		sprintf(Fname, "%s/In/Calib/%.2d_%.2d.txt", Path, 00, camID); FILE *fp = fopen(Fname, "r");
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

		sprintf(Fname, "%s/In/Calib/%.2d_%.2d_ext.txt", Path, 00, camID); fp = fopen(Fname, "r");
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

	for (int jj = 0; jj < nPanels; jj++)
	{
		for (int ii = 0; ii < nCamsPanel; ii++)
		{
			int camID = jj*nCamsPanel + ii + nHDs;

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


	sprintf(Fname, "%s/CamPose_%d.txt", Path, 0);
	FILE *fp = fopen(Fname, "a+");
	for (int ii = 0; ii < nHDs + nVGAs; ii++)
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
	sprintf(trackingFilePath, "%s/%d.track", filePath, CurrentFrame);
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

		//For currentTrackUnit
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

	int TrueCamID[480 + 30], ii = 0;
	for (int ii = 0; ii < 8; ii++)
		TrueCamID[ii] = ii;
	/*sprintf(trackingFilePath, "%s/camId.txt", filePath);
	FILE *fp = fopen(trackingFilePath, "r");//read from Han flow program
	while (fscanf(fp, "%d ", &TrueCamID[ii]) != EOF)
	ii++;
	fclose(fp);*/

	TrajectoryInfo.cpVis = new vector<vector<int> >[TrajectoryInfo.nTrajectories];
	TrajectoryInfo.fVis = new vector<vector<int> >[TrajectoryInfo.nTrajectories];

	vector<int>cpVis, fVis;
	fin >> dummy; //Visiblity
	for (int i = 0; i < TrajectoryInfo.nTrajectories; ++i)
	{
		if (i % 100 == 0)
			printf("Loading Visibility: %d/%d \r", i, TrajectoryInfo.nTrajectories);

		int ptIdx, trackedNum, visibleCamNum, visibleCamIdx;
		fin >> dummy >> ptIdx; //PtIdx

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
void Genrate2DTrajectoryBK(char *Path, int CurrentFrame, TrajectoryData InfoTraj, CameraData *AllCamInfo, vector<int> trajectoriesUsed)
{
	char Fname[200];
	int ntrajectoriesUsed = trajectoriesUsed.size();
	if (ntrajectoriesUsed > InfoTraj.nTrajectories)
	{
		printf("# trajectories input error\n");
		return;
	}

	int TrueCamID[480 + 30], ii = 0;
	sprintf(Fname, "%s/camId.txt", Path);
	FILE *fp = fopen(Fname, "r");//read from Han flow program
	while (fscanf(fp, "%d ", &TrueCamID[ii]) != EOF)
		ii++;
	fclose(fp);

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
			t3D = InfoTraj.cpThreeD[sTraj][jj];
			n3D = InfoTraj.cpNormal[sTraj][jj];
			normNormal = sqrt(pow(n3D.x, 2) + pow(n3D.y, 2) + pow(n3D.z, 2));
			int nvis = InfoTraj.cpVis[sTraj][jj].size();
			fprintf(fp, "%d %d ", CurrentFrame - ntracks + jj, nvis);
			for (int ii = 0; ii < nvis; ii++)
			{
				int viewID = InfoTraj.cpVis[sTraj][jj][ii];
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
			t3D = InfoTraj.fThreeD[sTraj][jj];
			n3D = InfoTraj.fNormal[sTraj][jj];
			normNormal = sqrt(pow(n3D.x, 2) + pow(n3D.y, 2) + pow(n3D.z, 2));
			int nvis = InfoTraj.fVis[sTraj][jj].size();
			fprintf(fp, "%d %d ", CurrentFrame + jj + 1, nvis);
			for (int ii = 0; ii < nvis; ii++)
			{
				int viewID = InfoTraj.fVis[sTraj][jj][ii];
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
void Genrate2DTrajectory(char *Path, int CurrentFrame, TrajectoryData InfoTraj, CameraData *AllCamInfo, vector<int> trajectoriesUsed)
{
	double angleThreshold = 0.8;
	char Fname[200];
	int ntrajectoriesUsed = trajectoriesUsed.size();
	if (ntrajectoriesUsed > InfoTraj.nTrajectories)
	{
		printf("# trajectories input error\n");
		return;
	}

	int TrueCamID[480 + 30], ii = 0;
	for (int ii = 0; ii < 9; ii++)
		TrueCamID[ii] = ii;
	/*sprintf(Fname, "%s/camId.txt", Path);
	FILE *fp = fopen(Fname, "r");//read from Han flow program
	while (fscanf(fp, "%d ", &TrueCamID[ii]) != EOF)
	ii++;
	fclose(fp);*/

	vector<Point2d> *Traj2D = new vector<Point2d>[InfoTraj.nViews];
	vector<int> *TimeLine = new vector<int>[InfoTraj.nViews];

	vector<int>viewIDs;
	vector<Point3d> PtA;
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
			t3D = InfoTraj.cpThreeD[sTraj][jj];
			n3D = InfoTraj.cpNormal[sTraj][jj];
			normNormal = sqrt(pow(n3D.x, 2) + pow(n3D.y, 2) + pow(n3D.z, 2));
			viewIDs.clear(), PtA.clear();
			for (int ii = 0; ii < InfoTraj.nViews; ii++)
			{
				int viewID = TrueCamID[ii];
				PtoC = Point3d(AllCamInfo[viewID].camCenter[0] - t3D.x, AllCamInfo[viewID].camCenter[1] - t3D.y, AllCamInfo[viewID].camCenter[2] - t3D.z);
				normPtoC = sqrt(pow(PtoC.x, 2) + pow(PtoC.y, 2) + pow(PtoC.z, 2));
				angle = (n3D.x*PtoC.x + n3D.y*PtoC.y + n3D.z*PtoC.z) / normNormal / normPtoC;
				if (angle < angleThreshold)
					continue;

				ProjectandDistort(t3D, &pt, AllCamInfo[viewID].P);
				viewIDs.push_back(viewID);
				PtA.push_back(Point3d(pt.x, pt.y, angle));
			}

			fprintf(fp, "%d %d ", CurrentFrame - ntracks + jj, viewIDs.size());
			for (int ii = 0; ii < viewIDs.size(); ii++)
				fprintf(fp, "%d %.2f %.2f %.2f ", viewIDs[ii], PtA[ii].x, PtA[ii].y, PtA[ii].z);
			fprintf(fp, "\n");
		}

		for (int jj = 0; jj < InfoTraj.fThreeD[sTraj].size(); jj++)
		{
			t3D = InfoTraj.fThreeD[sTraj][jj];
			n3D = InfoTraj.fNormal[sTraj][jj];
			normNormal = sqrt(pow(n3D.x, 2) + pow(n3D.y, 2) + pow(n3D.z, 2));

			viewIDs.clear(), PtA.clear();
			for (int ii = 0; ii < InfoTraj.nViews; ii++)
			{
				int viewID = TrueCamID[ii];
				PtoC = Point3d(AllCamInfo[viewID].camCenter[0] - t3D.x, AllCamInfo[viewID].camCenter[1] - t3D.y, AllCamInfo[viewID].camCenter[2] - t3D.z);
				normPtoC = sqrt(pow(PtoC.x, 2) + pow(PtoC.y, 2) + pow(PtoC.z, 2));
				angle = (n3D.x*PtoC.x + n3D.y*PtoC.y + n3D.z*PtoC.z) / normNormal / normPtoC;
				if (angle < angleThreshold)
					continue;

				ProjectandDistort(t3D, &pt, AllCamInfo[viewID].P);
				viewIDs.push_back(viewID);
				PtA.push_back(Point3d(pt.x, pt.y, angle));
			}

			fprintf(fp, "%d %d ", CurrentFrame + jj + 1, viewIDs.size());
			for (int ii = 0; ii < viewIDs.size(); ii++)
				fprintf(fp, "%d %.2f %.2f %.2f ", viewIDs[ii], PtA[ii].x, PtA[ii].y, PtA[ii].z);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}
	return;
}
void Genrate2DTrajectory2(char *Path, int CurrentFrame, TrajectoryData InfoTraj, VideoData &AllVideoData, vector<int> trajectoriesUsed)
{
	double angleThreshold = 0.8;
	char Fname[200];
	int ntrajectoriesUsed = trajectoriesUsed.size();
	if (ntrajectoriesUsed > InfoTraj.nTrajectories)
	{
		printf("# trajectories input error\n");
		return;
	}

	int TrueCamID[480 + 30], ii = 0;
	for (int ii = 0; ii < 9; ii++)
		TrueCamID[ii] = ii;

	int nframes = MaxnFrames;

	vector<Point2d> *Traj2D = new vector<Point2d>[InfoTraj.nViews];
	vector<int> *TimeLine = new vector<int>[InfoTraj.nViews];

	vector<int>viewIDs;
	vector<Point3d> PtA;
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
			t3D = InfoTraj.cpThreeD[sTraj][jj];
			n3D = InfoTraj.cpNormal[sTraj][jj];
			normNormal = sqrt(pow(n3D.x, 2) + pow(n3D.y, 2) + pow(n3D.z, 2));
			viewIDs.clear(), PtA.clear();
			for (int ii = 0; ii < InfoTraj.nViews; ii++)
			{
				int camID = TrueCamID[ii] * nframes + CurrentFrame;
				PtoC = Point3d(AllVideoData.VideoInfo[camID].camCenter[0] - t3D.x, AllVideoData.VideoInfo[camID].camCenter[1] - t3D.y, AllVideoData.VideoInfo[camID].camCenter[2] - t3D.z);
				normPtoC = sqrt(pow(PtoC.x, 2) + pow(PtoC.y, 2) + pow(PtoC.z, 2));
				angle = (n3D.x*PtoC.x + n3D.y*PtoC.y + n3D.z*PtoC.z) / normNormal / normPtoC;
				//	if (angle < angleThreshold)
				//	continue;

				ProjectandDistort(t3D, &pt, AllVideoData.VideoInfo[camID].P);
				viewIDs.push_back(TrueCamID[ii]);
				PtA.push_back(Point3d(pt.x, pt.y, angle));
			}

			fprintf(fp, "%d %d ", CurrentFrame - ntracks + jj, viewIDs.size());
			for (int ii = 0; ii < viewIDs.size(); ii++)
				fprintf(fp, "%d %.2f %.2f %.2f ", viewIDs[ii], PtA[ii].x, PtA[ii].y, PtA[ii].z);
			fprintf(fp, "\n");
		}

		for (int jj = 0; jj < InfoTraj.fThreeD[sTraj].size(); jj++)
		{
			t3D = InfoTraj.fThreeD[sTraj][jj];
			n3D = InfoTraj.fNormal[sTraj][jj];
			normNormal = sqrt(pow(n3D.x, 2) + pow(n3D.y, 2) + pow(n3D.z, 2));

			viewIDs.clear(), PtA.clear();
			for (int ii = 0; ii < InfoTraj.nViews; ii++)
			{
				int camID = TrueCamID[ii] * nframes + CurrentFrame;
				PtoC = Point3d(AllVideoData.VideoInfo[camID].camCenter[0] - t3D.x, AllVideoData.VideoInfo[camID].camCenter[1] - t3D.y, AllVideoData.VideoInfo[camID].camCenter[2] - t3D.z);
				normPtoC = sqrt(pow(PtoC.x, 2) + pow(PtoC.y, 2) + pow(PtoC.z, 2));
				angle = (n3D.x*PtoC.x + n3D.y*PtoC.y + n3D.z*PtoC.z) / normNormal / normPtoC;
				//if (angle < angleThreshold)
				//	continue;

				ProjectandDistort(t3D, &pt, AllVideoData.VideoInfo[camID].P);
				viewIDs.push_back(TrueCamID[ii]);
				PtA.push_back(Point3d(pt.x, pt.y, angle));
			}

			fprintf(fp, "%d %d ", CurrentFrame + jj + 1, viewIDs.size());
			for (int ii = 0; ii < viewIDs.size(); ii++)
				fprintf(fp, "%d %.2f %.2f %.2f ", viewIDs[ii], PtA[ii].x, PtA[ii].y, PtA[ii].z);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}
	return;
}
void Genrate2DTrajectory3(char *Path, int CurrentFrame, TrajectoryData InfoTraj, VideoData &AllVideoData, vector<int> trajectoriesUsed)
{
	char Fname[200];
	int ntrajectoriesUsed = trajectoriesUsed.size();
	if (ntrajectoriesUsed > InfoTraj.nTrajectories)
	{
		printf("# trajectories input error\n");
		return;
	}

	int TrueCamID[50];
	for (int ii = 0; ii < 9; ii++)
		TrueCamID[ii] = ii;

	int nframes = MaxnFrames;

	vector<Point2d> *Traj2D = new vector<Point2d>[InfoTraj.nViews];
	vector<int> *TimeLine = new vector<int>[InfoTraj.nViews];

	int frameOffset[] = { -30, -32, 29, -25, 0, -17, -10, -35 };

	vector<int>viewIDs;
	vector<Point3d> PtA;
	for (int ll = 0; ll < 8; ll++)
	{
		sprintf(Fname, "%s/CT%d.txt", Path, ll); FILE *fp = fopen(Fname, "a+");
		for (int kk = 0; kk < ntrajectoriesUsed; kk++)
		{
			int sTraj = trajectoriesUsed[kk];
			int nftracked = InfoTraj.cpNormal[sTraj].size() + InfoTraj.fThreeD[sTraj].size();
			if (nftracked < 4)
				fprintf(fp, "%d %d \n", kk, 0);
			else
			{
				fprintf(fp, "%d %d ", kk, nftracked);

				std::reverse(InfoTraj.cpNormal[sTraj].begin(), InfoTraj.cpNormal[sTraj].end());
				std::reverse(InfoTraj.cpThreeD[sTraj].begin(), InfoTraj.cpThreeD[sTraj].end());
				std::reverse(InfoTraj.cpVis[sTraj].begin(), InfoTraj.cpVis[sTraj].end());

				for (int jj = 0; jj < InfoTraj.nViews; jj++)
					Traj2D[jj].clear(), TimeLine[jj].clear();

				double normNormal, normPtoC, angle;
				Point3d t3D, n3D, PtoC;
				Point2d pt;

				int ntracks = InfoTraj.cpVis[sTraj].size();
				for (int jj = 0; jj < ntracks; jj++)
				{
					t3D = InfoTraj.cpThreeD[sTraj][jj];
					n3D = InfoTraj.cpNormal[sTraj][jj];
					normNormal = sqrt(pow(n3D.x, 2) + pow(n3D.y, 2) + pow(n3D.z, 2));
					viewIDs.clear(), PtA.clear();
					for (int ii = 0; ii < InfoTraj.nViews; ii++)
					{
						if (TrueCamID[ii] != ll)
							continue;
						int camID = TrueCamID[ii] * nframes + CurrentFrame - ntracks + jj + frameOffset[ll];
						PtoC = Point3d(AllVideoData.VideoInfo[camID].camCenter[0] - t3D.x, AllVideoData.VideoInfo[camID].camCenter[1] - t3D.y, AllVideoData.VideoInfo[camID].camCenter[2] - t3D.z);
						normPtoC = sqrt(pow(PtoC.x, 2) + pow(PtoC.y, 2) + pow(PtoC.z, 2));
						angle = (n3D.x*PtoC.x + n3D.y*PtoC.y + n3D.z*PtoC.z) / normNormal / normPtoC;
						//	if (angle < angleThreshold)
						//	continue;

						ProjectandDistort(t3D, &pt, AllVideoData.VideoInfo[camID].P);
						viewIDs.push_back(TrueCamID[ii]);
						PtA.push_back(Point3d(pt.x, pt.y, angle));
					}

					if (viewIDs.size() > 0)
						fprintf(fp, "%d %.3f %.3f ", CurrentFrame - ntracks + jj + frameOffset[ll], PtA[0].x, PtA[0].y);
				}

				for (int jj = 0; jj < InfoTraj.fThreeD[sTraj].size(); jj++)
				{
					t3D = InfoTraj.fThreeD[sTraj][jj];
					n3D = InfoTraj.fNormal[sTraj][jj];
					normNormal = sqrt(pow(n3D.x, 2) + pow(n3D.y, 2) + pow(n3D.z, 2));

					viewIDs.clear(), PtA.clear();
					for (int ii = 0; ii < InfoTraj.nViews; ii++)
					{
						if (TrueCamID[ii] != ll)
							continue;

						int camID = TrueCamID[ii] * nframes + CurrentFrame + jj + frameOffset[ll];
						PtoC = Point3d(AllVideoData.VideoInfo[camID].camCenter[0] - t3D.x, AllVideoData.VideoInfo[camID].camCenter[1] - t3D.y, AllVideoData.VideoInfo[camID].camCenter[2] - t3D.z);
						normPtoC = sqrt(pow(PtoC.x, 2) + pow(PtoC.y, 2) + pow(PtoC.z, 2));
						angle = (n3D.x*PtoC.x + n3D.y*PtoC.y + n3D.z*PtoC.z) / normNormal / normPtoC;
						//if (angle < angleThreshold)
						//	continue;

						ProjectandDistort(t3D, &pt, AllVideoData.VideoInfo[camID].P);
						viewIDs.push_back(TrueCamID[ii]);
						PtA.push_back(Point3d(pt.x, pt.y, angle));
					}

					if (viewIDs.size() > 0)
						fprintf(fp, "%d %.3f %.3f ", CurrentFrame + jj + frameOffset[ll], PtA[0].x, PtA[0].y);
				}
				fprintf(fp, "\n");
			}
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
	int camID, panelID, camIDInPanel;
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
				int timeID = infoTraj.trajectoryUnit[ii][kk].timeID, nvis = infoTraj.trajectoryUnit[ii][kk].nViews;

				int count = 0, iu, iv;
				for (int jj = 0; jj < nvis; jj++)
				{
					camID = infoTraj.trajectoryUnit[ii][kk].viewIDs[jj];
					u = (float)infoTraj.trajectoryUnit[ii][kk].uv[jj].x, v = (float)infoTraj.trajectoryUnit[ii][kk].uv[jj].y;
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
	int nHDs = 30;

	int syncOff[510];
	sprintf(Fname, "%s/Out/syncOff.txt", Path);
	FILE *fp = fopen("D:/DomeSync/Out/syncOff.txt", "r");
	for (int ii = 0; ii < 510; ii++)
		fscanf(fp, "%d ", &syncOff[ii]);
	fclose(fp);

	int TrueCamID[510];
	sprintf(Fname, "%s/camId.txt", Path);
	fp = fopen(Fname, "r");
	int ii = 0;
	while (fscanf(fp, "%d ", &TrueCamID[ii]) != EOF)
		ii++;
	fclose(fp);

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

	const int nCamsPerPanel = 24, width = 640, height = 480, length = width*height, HDwidth = 1920, HDheight = 1080, HDlength = HDwidth*HDheight;
	int camID, panelID, camIDInPanel;
	Point3d t3D;
	IplImage *Img = 0, *HDImg = 0;
	float u, v, angle;
	vector<Point3i> rgb1, rgb2;

	printf("Loading images to memory\n");
	int nchannels = 1;
	vector<IplImage*> AllImagePtr;
	vector<float*> AllImageParaPtr;
	float *tImg = new float[width*height*nchannels];
	float *HDtImg = new float[HDwidth*HDheight*nchannels];
	for (int ii = 0; ii < 2; ii++)
	{
		int viewID = cameraPair[ii];
		if (viewID >= nHDs)
		{
			for (int timeID = minFrame; timeID <= maxFrame; timeID++)
			{
				panelID = (viewID - nHDs) / nCamsPerPanel, camIDInPanel = (viewID - nHDs) % nCamsPerPanel;
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
					printf("View %d: %.2f %% completed \r", viewID, 100.0*(timeID - minFrame) / (maxFrame - minFrame));
				}

				AllImagePtr.push_back(Img);
				AllImageParaPtr.push_back(Para);
			}
		}
		else
		{
			for (int timeID = minFrame; timeID <= maxFrame; timeID++)
			{
				sprintf(Fname, "%s/In/%.8d/%.8d_%.02d_%.02d.png", Path, timeID, timeID, 0, viewID);
				HDImg = cvLoadImage(Fname, nchannels == 1 ? 0 : 1);
				float *Para = 0;
				if (HDImg == NULL)
					;// printf("Cannot load %s\n", Fname);
				else
				{
					Para = new float[nchannels * HDlength];
					for (int kk = 0; kk < nchannels; kk++)
					{
						for (int jj = 0; jj < HDheight; jj++)
							for (int ii = 0; ii < HDwidth; ii++)
								HDtImg[ii + jj*HDwidth + kk*HDlength] = HDImg->imageData[nchannels*ii + nchannels*jj*HDwidth + kk];
						Generate_Para_Spline(HDtImg + kk*HDlength, Para + kk*HDlength, HDwidth, HDheight, 1);
					}
					printf("View %d: %d : %.2f %% completed \r", viewID, timeID, 100.0*(timeID - minFrame) / (maxFrame - minFrame));
				}

				AllImagePtr.push_back(HDImg);
				AllImageParaPtr.push_back(Para);
			}
		}
		printf("View %d:  completed \n", viewID);
	}

	IplImage *drawing = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
	IplImage *HDdrawing = cvCreateImage(cvSize(HDwidth, HDheight), IPL_DEPTH_8U, 3);

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
				int timeID = infoTraj.trajectoryUnit[ii][kk].timeID, nvis = infoTraj.trajectoryUnit[ii][kk].nViews;

				int count = 0;
				for (int jj = 0; jj < nvis; jj++)
				{
					camID = infoTraj.trajectoryUnit[ii][kk].viewIDs[jj];
					u = infoTraj.trajectoryUnit[ii][kk].uv[jj].x, v = infoTraj.trajectoryUnit[ii][kk].uv[jj].y, angle = infoTraj.trajectoryUnit[ii][kk].angle[jj];
					if (camID >= nHDs && (u<hsubset || v<hsubset || u>width - hsubset || v>height - hsubset))
						break;
					if (camID < nHDs && (u<hsubset || v<hsubset || u>HDwidth - hsubset || v>HDheight - hsubset))
						break;
					if (angle < 0.5)
						break;
					if (timeID + syncOff[camID] > maxFrame || timeID + temporalOffset + syncOff[camID] > maxFrame || timeID + temporalOffset + syncOff[camID] < minFrame)
						continue;

					if (camID == cameraPair[0] || camID == cameraPair[1])
					{
						int ind = camID == cameraPair[0] ? 0 : 1;
						int id = ind* maxFrame + timeID + ind*temporalOffset;
						int index = ind* (maxFrame - minFrame + 1) + timeID - minFrame + ind*temporalOffset + syncOff[camID];
						if (camID >= nHDs)
						{
							Img = AllImagePtr[index];
							if (Img == NULL)
								continue;
						}
						else
						{
							HDImg = AllImagePtr[index];
							if (HDImg == NULL)
								continue;
						}

						u = (float)infoTraj.trajectoryUnit[ii][kk].uv[jj].x, v = (float)infoTraj.trajectoryUnit[ii][kk].uv[jj].y;
						if (camID >= nHDs && (u<hsubset || v<hsubset || u>width - hsubset || v>height - hsubset))
							continue;
						if (camID < nHDs && (u<hsubset || v<hsubset || u>HDwidth - hsubset || v>HDheight - hsubset))
							continue;
						int pcount = 0; double S[3];
						Para = AllImageParaPtr[index];

						if (Save)
						{
							if (camID >= nHDs)
							{
								for (int mm = -0; mm < height; mm++)
									for (int nn = -0; nn < width; nn++)
										Get_Value_Spline(Para, width, height, nn, mm, S, -1, 1), tImg[nn + mm*width] = (float)S[0];

								cvSaveImage("C:/temp/img1.png", Img);
								SaveDataToImage("C:/temp/img2.png", tImg, width, height);
							}
							else
							{
								for (int mm = -0; mm < HDheight; mm++)
									for (int nn = -0; nn < HDwidth; nn++)
										Get_Value_Spline(Para, HDwidth, HDheight, nn, mm, S, -1, 1), HDtImg[nn + mm*HDwidth] = (float)S[0];
								cvSaveImage("C:/temp/img1.png", HDImg);
								SaveDataToImage("C:/temp/img2.png", HDtImg, HDwidth, HDheight);
							}
						}

						if (camID >= nHDs)
							for (int ll = 0; ll < nchannels; ll++)
								for (int mm = -hsubset; mm <= hsubset; mm++)
									for (int nn = -hsubset; nn <= hsubset; nn++)
										Get_Value_Spline(Para + ll*length, width, height, u + nn, v + mm, S, -1, 1), RGB[(3 * ind + ll)*patchlength + (mm + hsubset)*patchSize + nn + hsubset] = S[0];
						else
							for (int ll = 0; ll < nchannels; ll++)
								for (int mm = -hsubset; mm <= hsubset; mm++)
									for (int nn = -hsubset; nn <= hsubset; nn++)
										Get_Value_Spline(Para + ll*HDlength, HDwidth, HDheight, u + nn, v + mm, S, -1, 1), RGB[(3 * ind + ll)*patchlength + (mm + hsubset)*patchSize + nn + hsubset] = S[0];

						/*if (camID >= nHDs)
						{
						cvCvtColor(Img, drawing, CV_GRAY2RGB);
						cvCircle(drawing, Point2i(u, v), 2, colors[rand() % 9], 1, 8, 0);
						CvFont font = cvFont(2.0 * 640 / 2048, 2);
						cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 2.0 * 640 / 2048, 2.0 * 640 / 2048, 0, 2, 8);
						CvPoint text_origin = { 640 / 30, 640 / 30 };
						panelID = camID / nCamsPerPanel, camIDInPanel = camID % nCamsPerPanel;
						sprintf(Fname, "%.2d_%.02d_%.02d %d/%d NVis of Traj %d", timeID + ind*temporalOffset, panelID + 1, camIDInPanel + 1, jj + 1, nvis, ii + 1);
						cvPutText(drawing, Fname, text_origin, &font, CV_RGB(255, 0, 0));
						char Fname2[200]; sprintf(Fname2, "Image %d", ind);
						cvShowImage(Fname2, drawing); waitKey(-1);
						}
						else
						{
						cvCvtColor(HDImg, HDdrawing, CV_GRAY2RGB);
						cvCircle(HDdrawing, Point2i(u, v), 2, colors[rand() % 9], 1, 8, 0);
						CvFont font = cvFont(2.0 * 1920 / 2048, 2);
						cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 2.0 * 1920 / 2048, 2.0 * 1080 / 2048, 0, 2, 8);
						CvPoint text_origin = { 1920 / 30, 1080 / 30 };
						sprintf(Fname, "%.2d_%.02d_%.02d %d/%d NVis of Traj %d", timeID + ind*temporalOffset, 0, camID, jj + 1, nvis, ii + 1);
						cvPutText(HDdrawing, Fname, text_origin, &font, CV_RGB(255, 0, 0));
						char Fname2[200]; sprintf(Fname2, "Image %d", ind);
						cvShowImage(Fname2, HDdrawing); waitKey(-1);
						}*/
						count++;
					}

					if (count == 2)
						zncc += ComputeZNCCPatch(RGB, RGB + 3 * patchlength, hsubset, nchannels, T);
					if (count == 2) //Keep on reading until the end of that point
						break;
				}
				//if (count == 1)
				//printf("Miss the other view!\n");
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
int Compute3DTrajectoryErrorZNCC2(char *Path, TrajectoryData infoTraj, int nTraj, int minFrame, int maxFrame, int *cameraPair, int *range)// , vector<Point3i>& Argb1, vector<Point3i>& Argb2)
{
	printf("Getting  Trajectory Color profile\n");
	char Fname[200];
	int nHDs = 30;

	int syncOff[510];

	/*sprintf(Fname, "%s/Out/syncOff.txt", Path);
	FILE *fp = fopen("D:/DomeSync/Out/syncOff.txt", "r");
	for (int ii = 0; ii < 510; ii++)
	fscanf(fp, "%d ", &syncOff[ii]);
	fclose(fp);*/

	int TrueCamID[510];
	/*sprintf(Fname, "%s/camId.txt", Path);
	fp = fopen(Fname, "r");
	int ii = 0;
	while (fscanf(fp, "%d ", &TrueCamID[ii]) != EOF)
	ii++;
	fclose(fp);*/

	for (int ii = 0; ii < 9; ii++)
		syncOff[ii] = 0, TrueCamID[ii] = ii;
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

	const int nCamsPerPanel = 24, width = 640, height = 480, length = width*height, HDwidth = 1920, HDheight = 1080, HDlength = HDwidth*HDheight;
	int camID, panelID, camIDInPanel;
	Point3d t3D;
	IplImage *Img = 0, *HDImg = 0;
	float u, v, angle;
	vector<Point3i> rgb1, rgb2;

	printf("Loading images to memory\n");
	int nchannels = 1;
	vector<IplImage*> AllImagePtr;
	vector<float*> AllImageParaPtr;
	float *tImg = new float[width*height*nchannels];
	float *HDtImg = new float[HDwidth*HDheight*nchannels];
	for (int ii = 0; ii < 2; ii++)
	{
		int viewID = cameraPair[ii];
		if (viewID >= nHDs)
		{
			for (int timeID = minFrame; timeID <= maxFrame; timeID++)
			{
				panelID = (viewID - nHDs) / nCamsPerPanel, camIDInPanel = (viewID - nHDs) % nCamsPerPanel;
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
					printf("View %d: %.2f %% completed \r", viewID, 100.0*(timeID - minFrame) / (maxFrame - minFrame));
				}

				AllImagePtr.push_back(Img);
				AllImageParaPtr.push_back(Para);
			}
		}
		else
		{
			for (int timeID = minFrame; timeID <= maxFrame; timeID++)
			{
				sprintf(Fname, "%s/In/%.8d/%.8d_%.02d_%.02d.png", Path, timeID, timeID, 0, viewID);
				HDImg = cvLoadImage(Fname, nchannels == 1 ? 0 : 1);
				float *Para = 0;
				if (HDImg == NULL)
					;// printf("Cannot load %s\n", Fname);
				else
				{
					Para = new float[nchannels * HDlength];
					for (int kk = 0; kk < nchannels; kk++)
					{
						for (int jj = 0; jj < HDheight; jj++)
							for (int ii = 0; ii < HDwidth; ii++)
								HDtImg[ii + jj*HDwidth + kk*HDlength] = HDImg->imageData[nchannels*ii + nchannels*jj*HDwidth + kk];
						Generate_Para_Spline(HDtImg + kk*HDlength, Para + kk*HDlength, HDwidth, HDheight, 1);
					}
					printf("View %d: %d : %.2f %% completed \r", viewID, timeID, 100.0*(timeID - minFrame) / (maxFrame - minFrame));
				}

				AllImagePtr.push_back(HDImg);
				AllImageParaPtr.push_back(Para);
			}
		}
		printf("\r View %d:  100%% \n", viewID);
	}

	IplImage *drawing = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
	IplImage *HDdrawing = cvCreateImage(cvSize(HDwidth, HDheight), IPL_DEPTH_8U, 3);

	int hsubset = 5, patchSize = (2 * hsubset + 1), patchlength = (2 * hsubset + 1)*(2 * hsubset + 1);
	double *T = new double[2 * patchlength * 3];
	double *RGB = new double[2 * 3 * (2 * hsubset + 1)*(2 * hsubset + 1)];

	bool Save = false, draw = false;
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
				int timeID = infoTraj.trajectoryUnit[ii][kk].timeID, nvis = infoTraj.trajectoryUnit[ii][kk].nViews;

				int count = 0;
				for (int jj = 0; jj < nvis; jj++)
				{
					camID = infoTraj.trajectoryUnit[ii][kk].viewIDs[jj];
					u = (float)infoTraj.trajectoryUnit[ii][kk].uv[jj].x, v = (float)infoTraj.trajectoryUnit[ii][kk].uv[jj].y, angle = infoTraj.trajectoryUnit[ii][kk].angle[jj];
					if (camID >= nHDs && (u<hsubset || v<hsubset || u>width - hsubset || v>height - hsubset))
						break;
					if (camID < nHDs && (u<hsubset || v<hsubset || u>HDwidth - hsubset || v>HDheight - hsubset))
						break;
					if (angle < 0.5)
						break;
					if (timeID + syncOff[camID] > maxFrame || timeID + temporalOffset + syncOff[camID] > maxFrame || timeID + temporalOffset + syncOff[camID] < minFrame)
						continue;

					if (camID == cameraPair[0] || camID == cameraPair[1])
					{
						int ind = camID == cameraPair[0] ? 0 : 1;
						int id = ind* maxFrame + timeID + ind*temporalOffset;
						int index = ind* (maxFrame - minFrame + 1) + timeID - minFrame + ind*temporalOffset + syncOff[camID];
						if (camID >= nHDs)
						{
							Img = AllImagePtr[index];
							if (Img == NULL)
								continue;
						}
						else
						{
							HDImg = AllImagePtr[index];
							if (HDImg == NULL)
								continue;
						}

						u = (float)infoTraj.trajectoryUnit[ii][kk].uv[jj].x, v = (float)infoTraj.trajectoryUnit[ii][kk].uv[jj].y;
						if (camID >= nHDs && (u<hsubset || v<hsubset || u>width - hsubset || v>height - hsubset))
							continue;
						if (camID < nHDs && (u<hsubset || v<hsubset || u>HDwidth - hsubset || v>HDheight - hsubset))
							continue;
						int pcount = 0; double S[3];
						Para = AllImageParaPtr[index];

						if (Save)
						{
							if (camID >= nHDs)
							{
								for (int mm = -0; mm < height; mm++)
									for (int nn = -0; nn < width; nn++)
										Get_Value_Spline(Para, width, height, nn, mm, S, -1, 1), tImg[nn + mm*width] = (float)S[0];

								cvSaveImage("C:/temp/img1.png", Img);
								SaveDataToImage("C:/temp/img2.png", tImg, width, height);
							}
							else
							{
								for (int mm = -0; mm < HDheight; mm++)
									for (int nn = -0; nn < HDwidth; nn++)
										Get_Value_Spline(Para, HDwidth, HDheight, nn, mm, S, -1, 1), HDtImg[nn + mm*HDwidth] = (float)S[0];
								cvSaveImage("C:/temp/img1.png", HDImg);
								SaveDataToImage("C:/temp/img2.png", HDtImg, HDwidth, HDheight);
							}
						}

						if (camID >= nHDs)
							for (int ll = 0; ll < nchannels; ll++)
								for (int mm = -hsubset; mm <= hsubset; mm++)
									for (int nn = -hsubset; nn <= hsubset; nn++)
										Get_Value_Spline(Para + ll*length, width, height, u + nn, v + mm, S, -1, 1), RGB[(3 * ind + ll)*patchlength + (mm + hsubset)*patchSize + nn + hsubset] = S[0];
						else
							for (int ll = 0; ll < nchannels; ll++)
								for (int mm = -hsubset; mm <= hsubset; mm++)
									for (int nn = -hsubset; nn <= hsubset; nn++)
										Get_Value_Spline(Para + ll*HDlength, HDwidth, HDheight, u + nn, v + mm, S, -1, 1), RGB[(3 * ind + ll)*patchlength + (mm + hsubset)*patchSize + nn + hsubset] = S[0];

						if (draw)
						{
							if (camID >= nHDs)
							{
								cvCvtColor(Img, drawing, CV_GRAY2RGB);
								cvCircle(drawing, Point2i(u, v), 2, colors[rand() % 9], 1, 8, 0);
								CvFont font = cvFont(2.0 * 640 / 2048, 2);
								cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 2.0 * 640 / 2048, 2.0 * 640 / 2048, 0, 2, 8);
								CvPoint text_origin = { 640 / 30, 640 / 30 };
								panelID = camID / nCamsPerPanel, camIDInPanel = camID % nCamsPerPanel;
								sprintf(Fname, "%.2d_%.02d_%.02d %d/%d NVis of Traj %d", timeID + ind*temporalOffset, panelID + 1, camIDInPanel + 1, jj + 1, nvis, ii + 1);
								cvPutText(drawing, Fname, text_origin, &font, CV_RGB(255, 0, 0));
								char Fname2[200]; sprintf(Fname2, "Image %d", ind);
								cvShowImage(Fname2, drawing); waitKey(-1);
							}
							else
							{
								cvCvtColor(HDImg, HDdrawing, CV_GRAY2RGB);
								cvCircle(HDdrawing, Point2i(u, v), 2, colors[rand() % 9], 1, 8, 0);
								CvFont font = cvFont(2.0 * 1920 / 2048, 2);
								cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 2.0 * 1920 / 2048, 2.0 * 1080 / 2048, 0, 2, 8);
								CvPoint text_origin = { 1920 / 30, 1080 / 30 };
								sprintf(Fname, "%.2d_%.02d_%.02d %d/%d NVis of Traj %d", timeID + ind*temporalOffset, 0, camID, jj + 1, nvis, ii + 1);
								cvPutText(HDdrawing, Fname, text_origin, &font, CV_RGB(255, 0, 0));
								char Fname2[200]; sprintf(Fname2, "Image %d", ind);
								cvShowImage(Fname2, HDdrawing); waitKey(-1);
							}
						}
						count++;
					}

					if (count == 2)
						zncc += ComputeZNCCPatch(RGB, RGB + 3 * patchlength, hsubset, nchannels, T);
					if (count == 2) //Keep on reading until the end of that point
						break;
				}
				//if (count == 1)
				//printf("Miss the other view!\n");
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
int Compute3DTrajectoryErrorZNCCDif(char *Path, TrajectoryData infoTraj, int nTraj, int minFrame, int maxFrame, int viewID, int *range)
{
	printf("Getting  Trajectory Color profile\n");
	char Fname[200];
	int nHDs = 30;

	int syncOff[510];
	sprintf(Fname, "%s/Out/syncOff.txt", Path);
	FILE *fp = fopen("D:/DomeSync/Out/syncOff.txt", "r");
	for (int ii = 0; ii < 510; ii++)
		fscanf(fp, "%d ", &syncOff[ii]);
	fclose(fp);

	int TrueCamID[510];
	sprintf(Fname, "%s/camId.txt", Path);
	fp = fopen(Fname, "r");
	int ii = 0;
	while (fscanf(fp, "%d ", &TrueCamID[ii]) != EOF)
		ii++;
	fclose(fp);

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

	const int nCamsPerPanel = 24, width = 640, height = 480, length = width*height, HDwidth = 1920, HDheight = 1080, HDlength = HDwidth*HDheight;
	int camID, panelID, camIDInPanel;
	Point3d t3D;
	IplImage *Img = 0, *HDImg = 0;
	float u, v, angle;
	vector<Point3i> rgb1, rgb2;

	printf("Loading images to memory\n");
	int nchannels = 1;
	vector<IplImage*> AllImagePtr;
	vector<float*> AllImageParaPtr;
	float *tImg = new float[width*height*nchannels];
	float *HDtImg = new float[HDwidth*HDheight*nchannels];
	if (viewID >= nHDs)
	{
		for (int timeID = minFrame; timeID <= maxFrame; timeID++)
		{
			panelID = (viewID - nHDs) / nCamsPerPanel, camIDInPanel = (viewID - nHDs) % nCamsPerPanel;
			sprintf(Fname, "%s/In/%.8d/%.8d_%.02d_%.02d.png", Path, timeID, timeID, panelID + 1, camIDInPanel + 1);
			Img = cvLoadImage(Fname, nchannels == 1 ? 0 : 1);
			float *Para = 0;
			if (Img == NULL)
				printf("Cannot load %s\n", Fname);
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
				printf("View %d: %.2f %% completed \r", viewID, 100.0*(timeID - minFrame) / (maxFrame - minFrame));
			}

			AllImagePtr.push_back(Img);
			AllImageParaPtr.push_back(Para);
		}
	}
	else
	{
		for (int timeID = minFrame; timeID <= maxFrame; timeID++)
		{
			sprintf(Fname, "%s/In/%.8d/%.8d_%.02d_%.02d.png", Path, timeID, timeID, 0, viewID);
			HDImg = cvLoadImage(Fname, nchannels == 1 ? 0 : 1);
			float *Para = 0;
			if (HDImg == NULL)
				;// printf("Cannot load %s\n", Fname);
			else
			{
				Para = new float[nchannels * HDlength];
				for (int kk = 0; kk < nchannels; kk++)
				{
					for (int jj = 0; jj < HDheight; jj++)
						for (int ii = 0; ii < HDwidth; ii++)
							HDtImg[ii + jj*HDwidth + kk*HDlength] = HDImg->imageData[nchannels*ii + nchannels*jj*HDwidth + kk];
					Generate_Para_Spline(HDtImg + kk*HDlength, Para + kk*HDlength, HDwidth, HDheight, 1);
				}
				printf("View %d: %d : %.2f %% completed \r", viewID, timeID, 100.0*(timeID - minFrame) / (maxFrame - minFrame));
			}

			AllImagePtr.push_back(HDImg);
			AllImageParaPtr.push_back(Para);
		}
	}
	printf("View %d:  completed \n", viewID);

	IplImage *drawing = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
	IplImage *HDdrawing = cvCreateImage(cvSize(HDwidth, HDheight), IPL_DEPTH_8U, 3);

	int hsubset = 10, patchSize = (2 * hsubset + 1), patchlength = (2 * hsubset + 1)*(2 * hsubset + 1);
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
			int count = 0;
			for (int kk = 0; kk < infoTraj.trajectoryUnit[ii].size() - 1; kk++)
			{
				int timeID = infoTraj.trajectoryUnit[ii][kk].timeID, nvis = infoTraj.trajectoryUnit[ii][kk].nViews;

				int jj;
				for (jj = 0; jj < nvis; jj++)
				{
					camID = infoTraj.trajectoryUnit[ii][kk].viewIDs[jj];
					u = infoTraj.trajectoryUnit[ii][kk].uv[jj].x, v = infoTraj.trajectoryUnit[ii][kk].uv[jj].y, angle = infoTraj.trajectoryUnit[ii][kk].angle[jj];
					if (camID >= nHDs && (u<hsubset || v<hsubset || u>width - hsubset || v>height - hsubset))
						break;
					if (camID < nHDs && (u<hsubset || v<hsubset || u>HDwidth - hsubset || v>HDheight - hsubset))
						break;
					if (angle < 0.5)
						break;
					if (timeID + syncOff[camID] > maxFrame || timeID + temporalOffset + syncOff[camID] > maxFrame || timeID + temporalOffset + syncOff[camID] < minFrame)
						continue;

					if (camID == viewID)
					{
						int index = timeID - minFrame + syncOff[camID];
						if (camID >= nHDs)
						{
							Img = AllImagePtr[index];
							if (Img == NULL)
								continue;
						}
						else
						{
							HDImg = AllImagePtr[index];
							if (HDImg == NULL)
								continue;
						}

						u = infoTraj.trajectoryUnit[ii][kk].uv[jj].x, v = infoTraj.trajectoryUnit[ii][kk].uv[jj].y;
						if (camID >= nHDs && (u<hsubset || v<hsubset || u>width - hsubset || v>height - hsubset))
							continue;
						if (camID < nHDs && (u<hsubset || v<hsubset || u>HDwidth - hsubset || v>HDheight - hsubset))
							continue;
						int pcount = 0; double S[3];
						Para = AllImageParaPtr[index];

						if (camID >= nHDs)
							for (int ll = 0; ll < nchannels; ll++)
								for (int mm = -hsubset; mm <= hsubset; mm++)
									for (int nn = -hsubset; nn <= hsubset; nn++)
										Get_Value_Spline(Para + ll*length, width, height, u + nn, v + mm, S, -1, 1), RGB[(3 * count + ll)*patchlength + (mm + hsubset)*patchSize + nn + hsubset] = S[0];
						else
							for (int ll = 0; ll < nchannels; ll++)
								for (int mm = -hsubset; mm <= hsubset; mm++)
									for (int nn = -hsubset; nn <= hsubset; nn++)
										Get_Value_Spline(Para + ll*HDlength, HDwidth, HDheight, u + nn, v + mm, S, -1, 1), RGB[(3 * count + ll)*patchlength + (mm + hsubset)*patchSize + nn + hsubset] = S[0];

						if (camID >= nHDs)
						{
							cvCvtColor(Img, drawing, CV_GRAY2RGB);
							cvCircle(drawing, Point2i(u, v), 2, colors[rand() % 9], 1, 8, 0);
							CvFont font = cvFont(2.0 * 640 / 2048, 2);
							cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 2.0 * 640 / 2048, 2.0 * 640 / 2048, 0, 2, 8);
							CvPoint text_origin = { 640 / 30, 640 / 30 };
							panelID = camID / nCamsPerPanel, camIDInPanel = camID % nCamsPerPanel;
							sprintf(Fname, "%.2d_%.02d_%.02d %d/%d NVis of Traj %d", timeID, panelID + 1, camIDInPanel + 1, jj + 1, nvis, ii + 1);
							cvPutText(drawing, Fname, text_origin, &font, CV_RGB(255, 0, 0));
							cvShowImage("Image", drawing); waitKey(300);
						}
						else
						{
							cvCvtColor(HDImg, HDdrawing, CV_GRAY2RGB);
							cvCircle(HDdrawing, Point2i(u, v), 2, colors[rand() % 9], 1, 8, 0);
							CvFont font = cvFont(2.0 * 1920 / 2048, 2);
							cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 2.0 * 1920 / 2048, 2.0 * 1080 / 2048, 0, 2, 8);
							CvPoint text_origin = { 1920 / 30, 1080 / 30 };
							sprintf(Fname, "%.2d_%.02d_%.02d %d/%d NVis of Traj %d", timeID, 0, camID, jj + 1, nvis, ii + 1);
							cvPutText(HDdrawing, Fname, text_origin, &font, CV_RGB(255, 0, 0));
							cvShowImage("Image", HDdrawing); waitKey(300);
						}

						count++;
						break;
					}
				}
				if (count == 1 && jj == nvis)//Miss the other view
					count = 0;

				if (count == 2)
				{
					zncc += ComputeZNCCPatch(RGB, RGB + 3 * patchlength, hsubset, nchannels, T);
					count = 0;
				}
			}
		}
		temporal.push_back(temporalOffset);
		ZNCCv.push_back(zncc);

		printf("Time offset %d: @%.2f%% Zncc: %f TE: %.2fs\n", temporalOffset, 100.0, zncc, omp_get_wtime() - start);
	}

	sprintf(Fname, "%s/dif_%d.txt", Path, viewID);
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
	int  viewID, panelID, camIDInPanel;
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
	double *Znssd_reqd = new double[9 * patchlength];

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
				int timeID = infoTraj.trajectoryUnit[ii][kk].timeID;
				for (int jj = 0; jj < infoTraj.trajectoryUnit[ii][kk].nViews; jj++)
				{
					viewID = infoTraj.trajectoryUnit[ii][kk].viewIDs[jj];
					if (timeID + syncOff[viewID] > maxFrame || timeID + temporalOffset + syncOff[viewID] > maxFrame || timeID + temporalOffset + syncOff[viewID] < minFrame)
						break;
					if (viewID == SelectedViewID)
					{
						u = infoTraj.trajectoryUnit[ii][kk].uv[jj].x, v = infoTraj.trajectoryUnit[ii][kk].uv[jj].y, angle = infoTraj.trajectoryUnit[ii][kk].angle[jj];
						if (u<hsubset || v<hsubset || u>width - hsubset || v>height - hsubset)
							break;
						if (angle < 0.5)
							break;
						Projected2DLoc.push_back(infoTraj.trajectoryUnit[ii][kk].uv[jj]);

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
				int timeID = infoTraj.trajectoryUnit[ii][kk].timeID;
				for (int jj = 0; jj < infoTraj.trajectoryUnit[ii][kk].nViews; jj++)
				{
					viewID = infoTraj.trajectoryUnit[ii][kk].viewIDs[jj];
					if (timeID + syncOff[viewID] > maxFrame || timeID + temporalOffset + syncOff[viewID] > maxFrame || timeID + temporalOffset + syncOff[viewID] < minFrame)
						break;
					if (viewID == SelectedViewID)
					{
						u = infoTraj.trajectoryUnit[ii][kk].uv[jj].x, v = infoTraj.trajectoryUnit[ii][kk].uv[jj].y, angle = infoTraj.trajectoryUnit[ii][kk].angle[jj];
						if (u<hsubset || v<hsubset || u>width - hsubset || v>height - hsubset)
							break;
						if (angle < 0.5)
							break;
						Projected2DLoc.push_back(infoTraj.trajectoryUnit[ii][kk].uv[jj]);

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

void Average_Filtering_All(char *lpD, int width, int height, int ni, int HSize, int VSize)
{
	int length = width*height;
	int i, j, k, m, n, ii, jj, s;

	char *T = new char[length];

	for (k = 0; k < ni; k++)
	{
		/// Testing shows that memcpy() is NOT faster than the conventional way
		// memcpy(T, lpD+k*length, length);
		for (n = 0; n < length; n++)
			*(T + n) = *(lpD + k*length + n);

		/*
		for(j=0; j<height; j++)
		{
		for(i=0; i<width; i++)
		{
		s = 0;
		m = 0;
		for(jj=-VSize; jj<=VSize; jj++)
		{
		for(ii=-HSize; ii<=HSize; ii++)
		{
		if( (j+jj)>=0 && (j+jj)<height && (i+ii)>=0 && (i+ii)<width )
		{
		s +=  (int)((unsigned char)*(lpD+k*length+(j+jj)*width+i+ii));
		m++;
		}
		}
		}
		s = int( (double)s/m + 0.5 );
		*(T+j*width+i) = (char)s;
		}
		}*/

		/// Do not consider the boundary issue to increase the processing speed
		m = (2 * HSize + 1)*(2 * VSize + 1);
		for (j = VSize; j < height - VSize; j++)
		{
			for (i = HSize; i < width - HSize; i++)
			{
				s = 0;
				for (jj = -VSize; jj <= VSize; jj++)
				{
					for (ii = -HSize; ii <= HSize; ii++)
					{
						s += (int)((unsigned char)*(lpD + k*length + (j + jj)*width + i + ii));
					}
				}
				s = int((double)s / m + 0.5);
				*(T + j*width + i) = (char)s;
			}
		}

		for (n = 0; n < length; n++)
			*(lpD + k*length + n) = *(T + n);
	}

	delete[]T;

	return;
}
void MConventional_PhaseShifting(char *lpD, char *lpPBM, double* lpFO, int nipf, int length, int Mask_Threshold, double *f_atan2)
{
	int n, ni, nj;
	int I1, I2, I3, I4, I5, f1, f2;
	double q;

	if (nipf == 3)
	{
		ni = 510;
		nj = 255;
		for (n = 0; n < length; n++)
		{
			I1 = (int)((unsigned char)*(lpD + n));
			I2 = (int)((unsigned char)*(lpD + length + n));
			I3 = (int)((unsigned char)*(lpD + 2 * length + n));
			if (abs(I1 - I2) < Mask_Threshold && abs(I2 - I3) < Mask_Threshold && abs(I3 - I1) < Mask_Threshold)
				*(lpPBM + n) = (char)0;

			f1 = 2 * I1 - I2 - I3;
			f2 = I3 - I2;
			q = f_atan2[(f2 + nj)*(2 * ni + 1) + f1 + ni];

			if (q < 0.0)
				q = q + 2 * Pi;
			*(lpFO + n) = q / (2 * Pi);
		}
	}
	else if (nipf == 4)
	{
		ni = 255;
		nj = 255;

		for (n = 0; n < length; n++)
		{
			I1 = (int)((unsigned char)*(lpD + n));
			I2 = (int)((unsigned char)*(lpD + length + n));
			I3 = (int)((unsigned char)*(lpD + 2 * length + n));
			I4 = (int)((unsigned char)*(lpD + 3 * length + n));
			if (abs(I1 - I2) < Mask_Threshold && abs(I2 - I3) < Mask_Threshold && abs(I3 - I4) < Mask_Threshold && abs(I4 - I1) < Mask_Threshold)
			{
				*(lpPBM + n) = (char)0;
			}

			f1 = I1 - I3;
			f2 = I4 - I2;
			q = *(f_atan2 + (f2 + nj)*(2 * ni + 1) + f1 + ni);
			if (q < 0.0)
				q = q + 2 * Pi;
			*(lpFO + n) = q / (2 * Pi);
		}
	}
	else if (nipf == 5)
	{
		ni = 510;
		nj = 510;

		for (n = 0; n < length; n++)
		{
			I1 = (int)((unsigned char)*(lpD + n));
			I2 = (int)((unsigned char)*(lpD + length + n));
			I3 = (int)((unsigned char)*(lpD + 2 * length + n));
			I4 = (int)((unsigned char)*(lpD + 3 * length + n));
			I5 = (int)((unsigned char)*(lpD + 4 * length + n));
			if (abs(I1 - I2) < Mask_Threshold && abs(I2 - I3) < Mask_Threshold && abs(I3 - I4) < Mask_Threshold && abs(I4 - I5) < Mask_Threshold && abs(I5 - I1) < Mask_Threshold)
			{
				*(lpPBM + n) = (char)0;
			}

			f1 = I1 + I5 - 2 * I3;
			f2 = 2 * (I4 - I2);
			q = *(f_atan2 + (f2 + nj)*(2 * ni + 1) + f1 + ni);
			if (q < 0.0)
				q = q + 2 * Pi;
			*(lpFO + n) = q / (2 * Pi);
		}
	}
	else
	{
		int I_max, I_min, i;
		double ff1, ff2;
		I_max = 0;
		I_min = 255;
		int *I = new int[nipf];

		for (n = 0; n < length; n++)
		{
			for (i = 0; i<nipf; i++)
			{
				I[i] = (int)((unsigned char)*(lpD + i*length + n));
				if (I[i]>I_max)
					I_max = I[i];
				if (I[i] < I_min)
					I_min = I[i];
			}
			if ((I_max - I_min) < Mask_Threshold)
				*(lpPBM + n) = (char)0;

			ff1 = 0.0;
			ff2 = 0.0;
			for (i = 0; i < nipf; i++)
			{
				ff1 = ff1 + I[i] * cos(2.0*i*Pi / nipf);
				ff2 = ff2 + I[i] * sin(2.0*i*Pi / nipf);
			}
			q = atan2(-ff2, ff1);
			if (q < 0.0)
				q = q + 2 * Pi;
			*(lpFO + n) = q / (2.0*Pi);
		}
		delete[]I;
	}

	return;
}
void DecodePhaseShift2(char *Image, char *PBM, double *PhaseUW, int width, int height, int *frequency, int nfrequency, int sstep, int LFstep, int half_filter_size, int m_mask)
{
	int ii, jj, k, m, n, length = width*height;
	int ni = sstep*(nfrequency - 1) + LFstep;
	double *PhaseW = new double[nfrequency*length];

	double s;
	if (sstep == 3)
	{
		m = 510; n = 255; s = sqrt(3.0);
	}
	else if (sstep == 4)
	{
		m = 255; n = 255; s = 1.0;
	}
	else if (sstep == 5)
	{
		m = 510; n = 510; s = 1.0;
	}
	else
	{
		m = 1; n = 1;
	}

	double *f_atan2 = new double[(2 * m + 1)*(2 * n + 1)];
	for (jj = -n; jj <= n; jj++)
	{
		for (ii = -m; ii <= m; ii++)
		{
			f_atan2[(jj + n)*(2 * m + 1) + ii + m] = atan2(s*jj, 1.0*ii);
		}
	}

	// Filtering all images first
	if (half_filter_size >= 1)
		Average_Filtering_All(Image, width, height, ni, half_filter_size, half_filter_size);

	m = nfrequency*length;
	for (n = 0; n < m; n++)
		*(PBM + n) = (char)255;

	//Phase warpping
	for (k = 0; k < nfrequency; k++)
	{
		if (k < nfrequency - 1)
			MConventional_PhaseShifting(Image + k*sstep*length, PBM + k*length, PhaseW + k*length, sstep, length, m_mask, f_atan2);
		else
			MConventional_PhaseShifting(Image + k*sstep*length, PBM + k*length, PhaseW + k*length, LFstep, length, m_mask, f_atan2);
	}

	//Incremental phase unwarpped
	double m_s;
	for (k = 1; k < nfrequency; k++)
	{
		for (n = 0; n < length; n++)
		{
			if (PBM[(k - 1)*length + n] == (char)0)
			{
				PBM[k*length + n] = (char)0;
				PhaseW[k*length + n] = 1000.0f;
			}
			else
			{
				m_s = PhaseW[(k - 1)*length + n] * frequency[k] / frequency[k - 1] - PhaseW[k*length + n];
				if (m_s >= 0.0)
					PhaseW[k*length + n] += (int)(m_s + 0.5);
				else
					PhaseW[k*length + n] += (int)(m_s - 0.5);
			}
		}
	}

	for (n = 0; n < length; n++)
		PhaseUW[n] = PhaseW[(nfrequency - 1)*length + n];

	delete[]PhaseW;
	delete[]f_atan2;
	return;
}

bool RoateImage180(char *fname, char *Img, int &width, int &height, int nchannels, bool silent)
{
	Mat view = imread(fname, nchannels == 1 ? 0 : 1);
	if (view.data == NULL)
	{
		if (!silent)
			cout << "Cannot load: " << fname << endl;
		return false;
	}
	width = view.cols, height = view.rows;
	if (Img == NULL)
		Img = new char[width*height*nchannels];
	int length = width*height;
	for (int kk = 0; kk < nchannels; kk++)
	{
		for (int jj = 0; jj < height; jj++)
			for (int ii = 0; ii < width; ii++)
				Img[(width - 1 - ii) + (height - 1 - jj)*width + kk*length] = (char)view.data[nchannels*ii + jj*nchannels*width + kk];
	}

	for (int jj = 0; jj < height; jj++)
		for (int ii = 0; ii < width; ii++)
			for (int kk = 0; kk < nchannels; kk++)
				view.data[nchannels*ii + kk + nchannels*jj*width] = Img[ii + jj*width + kk*length];

	return imwrite(fname, view);
}
bool GrabImageCVFormat(char *fname, char *Img, int &width, int &height, int nchannels, bool silent)
{
	Mat view = imread(fname, nchannels == 1 ? 0 : 1);
	if (view.data == NULL)
	{
		if (!silent)
			cout << "Cannot load: " << fname << endl;
		return false;
	}
	width = view.cols, height = view.rows;
	if (Img == NULL)
		Img = new char[width*height*nchannels];
	int length = width*height;
	for (int kk = 0; kk < nchannels; kk++)
	{
		for (int jj = 0; jj < height; jj++)
			for (int ii = 0; ii < width; ii++)
				Img[nchannels*ii + jj*nchannels*width + kk] = (char)view.data[nchannels*ii + jj*nchannels*width + kk];
	}

	return true;
}
bool GrabImage(char *fname, char *Img, int &width, int &height, int nchannels, bool silent)
{
	Mat view = imread(fname, nchannels == 1 ? 0 : 1);
	if (view.data == NULL)
	{
		if (!silent)
			cout << "Cannot load: " << fname << endl;
		return false;
	}
	if (Img == NULL)
	{
		width = view.cols, height = view.rows;
		Img = new char[width*height*nchannels];
	}
	int length = width*height;
	for (int kk = 0; kk < nchannels; kk++)
	{
		for (int jj = 0; jj < height; jj++)
			for (int ii = 0; ii < width; ii++)
				Img[ii + jj*width + kk*length] = (char)view.data[nchannels*ii + jj*nchannels*width + kk];
	}

	return true;
}
bool GrabImage(char *fname, unsigned char *Img, int &width, int &height, int nchannels, bool silent)
{
	Mat view = imread(fname, nchannels == 1 ? 0 : 1);
	if (view.data == NULL)
	{
		if (!silent)
			cout << "Cannot load: " << fname << endl;
		return false;
	}
	if (Img == NULL)
	{
		width = view.cols, height = view.rows;
		Img = new unsigned char[width*height*nchannels];
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
bool GrabImage(char *fname, float *Img, int &width, int &height, int nchannels, bool silent)
{
	Mat view = imread(fname, nchannels == 1 ? 0 : 1);
	if (view.data == NULL)
	{
		if (!silent)
			cout << "Cannot load: " << fname << endl;
		return false;
	}
	if (Img == NULL)
	{
		width = view.cols, height = view.rows;
		Img = new float[width*height*nchannels];
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
bool GrabImage(char *fname, double *Img, int &width, int &height, int nchannels, bool silent)
{
	Mat view = imread(fname, nchannels == 1 ? 0 : 1);
	if (view.data == NULL)
	{
		if (!silent)
			cout << "Cannot load: " << fname << endl;
		return false;
	}
	width = view.cols, height = view.rows;
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
bool SaveDataToImageCVFormat(char *fname, char *Img, int width, int height, int nchannels)
{
	int ii, jj, kk, length = width*height;

	Mat M = Mat::zeros(height, width, nchannels == 1 ? CV_8UC1 : CV_8UC3);
	for (jj = 0; jj < height; jj++)
		for (ii = 0; ii < width; ii++)
			for (kk = 0; kk < nchannels; kk++)
				M.data[nchannels*ii + kk + nchannels*jj*width] = (unsigned char)Img[nchannels*ii + kk + nchannels*jj*width];

	return imwrite(fname, M);
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
				M.data[nchannels*ii + kk + nchannels*jj*width] = Img[ii + jj*width + kk*length];

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

void ShowDataAsImage(char *fname, unsigned char *Img, int width, int height, int nchannels)
{
	int ii, jj, kk, length = width*height;

	Mat M = Mat::zeros(height, width, nchannels == 1 ? CV_8UC1 : CV_8UC3);
	for (jj = 0; jj < height; jj++)
		for (ii = 0; ii < width; ii++)
			for (kk = 0; kk < nchannels; kk++)
				M.data[nchannels*ii + kk + nchannels*jj*width] = Img[ii + jj*width + kk*length];

	if (nchannels == 3)
	{
		imshow(fname, M);
		waitKey(-1);
		destroyWindow(fname);
	}
	else
	{
		Mat cM;
		cvtColor(M, cM, CV_GRAY2RGB);
		imshow(fname, cM);
		waitKey(-1);
		destroyWindow(fname);
	}

	return;
}
void ShowDataAsImage(char *fname, double *Img, int width, int height, int nchannels)
{
	int ii, jj, kk, length = width*height;

	Mat M = Mat::zeros(height, width, nchannels == 1 ? CV_8UC1 : CV_8UC3);
	for (jj = 0; jj < height; jj++)
		for (ii = 0; ii < width; ii++)
			for (kk = 0; kk < nchannels; kk++)
				M.data[nchannels*ii + kk + nchannels*jj*width] = (unsigned char)(int)(Img[ii + jj*width + kk*length] + 0.5);

	if (nchannels == 3)
	{
		imshow(fname, M);
		waitKey(-1);
		destroyWindow(fname);
	}
	else
	{
		Mat cM;
		cvtColor(M, cM, CV_GRAY2RGB);
		imshow(fname, cM);
		waitKey(-1);
		destroyWindow(fname);
	}

	return;
}

void RemoveNoiseMedianFilter(float *data, int width, int height, int ksize, float thresh)
{
	Mat src = Mat(height, width, CV_32F, data);
	Mat dst = Mat(10, 10, CV_32F, Scalar(0));

	if (ksize > 5)
		ksize = 5;

	medianBlur(src, dst, ksize);

	for (int jj = 0; jj < height; jj++)
		for (int ii = 0; ii<width; ii++)
			if (abs(data[ii + jj*width] - dst.at<float>(jj, ii))>thresh)
				data[ii + jj*width] = 0.0;

	return;
}
void RemoveNoiseMedianFilter(double *data, int width, int height, int ksize, float thresh, float *fdata)
{
	bool createMem = false;
	if (fdata == NULL)
	{
		createMem = true;
		fdata = new float[width*height];
	}

	for (int ii = 0; ii < width*height; ii++)
		fdata[ii] = (float)data[ii];

	Mat src = Mat(height, width, CV_32F, fdata);
	Mat dst = Mat(10, 10, CV_32F, Scalar(0));

	if (ksize > 5)
		ksize = 5;

	medianBlur(src, dst, ksize);

	for (int jj = 0; jj < height; jj++)
		for (int ii = 0; ii<width; ii++)
			if (abs(data[ii + jj*width] - dst.at<float>(jj, ii))>thresh)
				data[ii + jj*width] = 0.0;

	if (createMem)
		delete[]fdata;
	return;
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

int readVisualSFMSift(char *fn, vector<KeyPoint>&kpts, Mat &descriptors, bool silent)
{
	ifstream fin;
	fin.open(fn, ios::binary);
	if (!fin.is_open())
	{
		cout << "Cannot open: " << fn << endl;
		return 1;
	}
	if (!silent)
		cout << "Load " << fn << endl;

	int dummy, npts, descriptorSize = SIFTBINS;
	float val;

	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));//SIFT
	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));///V4.0
	fin.read(reinterpret_cast<char *>(&npts), sizeof(int));//npts
	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));//5numbers
	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));//descriptorSize

	kpts.reserve(npts);
	KeyPoint kpt;
	for (int ii = 0; ii < npts; ii++)
	{
		fin.read(reinterpret_cast<char *>(&kpt.pt.x), sizeof(float));
		fin.read(reinterpret_cast<char *>(&kpt.pt.y), sizeof(float));
		fin.read(reinterpret_cast<char *>(&val), sizeof(float));
		fin.read(reinterpret_cast<char *>(&kpt.size), sizeof(float));
		fin.read(reinterpret_cast<char *>(&kpt.angle), sizeof(float));
		kpts.push_back(kpt);
	}


	uint8_t d;
	float desci[SIFTBINS];
	descriptors.create(npts, SIFTBINS, CV_32F);
	for (int j = 0; j < npts; j++)
	{
		val = 0.0;
		for (int i = 0; i < descriptorSize; i++)
		{
			fin.read(reinterpret_cast<char *>(&d), sizeof(uint8_t));
			dummy = (int)d;
			desci[i] = (float)(int)d;
			val += desci[i] * desci[i];
		}
		val = sqrt(val);

		for (int i = 0; i < descriptorSize; i++)
			descriptors.at<float>(j, i) = desci[i] / val;
	}
	fin.close();

	return 0;
}
bool WriteKPointsSIFTGPU(char *fn, vector<SiftKeypoint>kpts, bool silent)
{
	FILE *fp = fopen(fn, "w+");
	if (fp == NULL)
	{
		if (silent)
			cout << "Cannot write: " << fn << endl;
		return false;
	}

	int npts = kpts.size();
	fprintf(fp, "%d\n", npts);
	for (int j = 0; j < npts; ++j)
		fprintf(fp, "%.4f %.4f %.4f %.4f\n", kpts.at(j).x, kpts.at(j).y, kpts.at(j).o, kpts.at(j).s);
	fclose(fp);

	return true;
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
		int x = (int)kpts[kk].pt.x;
		int y = (int)kpts[kk].pt.y;
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

void ResizeImage(unsigned char *Image, unsigned char *OutImage, int width, int height, int nchannels, double Rfactor, double sigma, int InterpAlgo, double *InPara)
{
	bool createMem = false;
	int length = width*height;
	if (InPara == NULL)
	{
		createMem = true;
		InPara = new double[length*nchannels];
		if (sigma == 0)
			for (int kk = 0; kk < nchannels; kk++)
				Generate_Para_Spline(Image + kk*length, InPara + kk*length, width, height, InterpAlgo);
		else
		{
			double *SmoothImg = new double[length];
			for (int kk = 0; kk < nchannels; kk++)
			{
				Gaussian_smooth(Image + kk*length, SmoothImg, height, width, 255.0, sigma);
				Generate_Para_Spline(SmoothImg, InPara + kk*length, width, height, InterpAlgo);
			}
			delete[]SmoothImg;
		}
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
void ResizeImage(float *Image, float *OutImage, int width, int height, int nchannels, double Rfactor, double sigma, int InterpAlgo, float *InPara)
{
	bool createMem = false;
	int length = width*height;
	if (InPara == NULL)
	{
		createMem = true;
		InPara = new float[width*height*nchannels];
		if (sigma < 0.001)
			for (int kk = 0; kk < nchannels; kk++)
				Generate_Para_Spline(Image + kk*length, InPara + kk*length, width, height, InterpAlgo);
		else
		{
			printf("Image resize type not supported\n");
			/*float *SmoothImg = new float[length*nchannels];
			for (int kk = 0; kk < nchannels; kk++)
			{
			Gaussian_smooth(Image + kk*length, SmoothImg, height, width, 255.0, sigma);
			Generate_Para_Spline(SmoothImg, InPara + kk*length, width, height, InterpAlgo);
			}
			delete[]SmoothImg;*/
		}
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
void ResizeImage(double *Image, double *OutImage, int width, int height, int nchannels, double Rfactor, double sigma, int InterpAlgo, double *InPara)
{
	bool createMem = false;
	int length = width*height;
	if (InPara == NULL)
	{
		createMem = true;
		InPara = new double[width*height*nchannels];

		if (sigma < 0.001)
			for (int kk = 0; kk < nchannels; kk++)
				Generate_Para_Spline(Image + kk*length, InPara + kk*length, width, height, InterpAlgo);
		else
		{
			double *SmoothImg = new double[length];
			for (int kk = 0; kk < nchannels; kk++)
			{
				Gaussian_smooth(Image + kk*length, SmoothImg, height, width, 255.0, sigma);
				Generate_Para_Spline(SmoothImg, InPara + kk*length, width, height, InterpAlgo);
			}
			delete[]SmoothImg;
		}
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
			if (flowMag2.size() > 0)
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
					sprintf(Fname, "%s/%d.png", PATH, nNonBlurImages);
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
			if (flowMag2.size() > 0)
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
		fprintf(fp, "%.3f\n", distance[ii]);
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

	int percent = 10, increment = 10;
	for (int Id = StartFrame; Id <= StopFrame; Id++)
	{
		sprintf(Fname, "%s/%d.png", Path, Id);	cvImg = imread(Fname, 1);
		if (cvImg.empty())
		{
			printf("Cannot load %s\n", Fname);
			continue;
		}

		int per = 100 * (Id - StartFrame) / (StopFrame - StartFrame + 1);
		if (per >= percent)
		{
			percent += increment;
			printf("\rWorking on %s: %d%%", Path, per);
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

		sprintf(Fname, "%s/U_%d.png", Path, Id);
		imwrite(Fname, nImg);
	}
	printf("\rWorking on %s: 100%%\n", Fname);

	delete[]Img;

	return 0;
}
int LensCorrectionDriver(char *Path, double *K, double *distortion, int LensType, int startID, int stopID, double Imgscale, double Contscale, int interpAlgo)
{
	char Fname[200];

	//double Imgscale = 1.0, Contscale = 1.0, iK[9];
	//mat_invert(K, iK, 3);

	Mat cvImg;
	Mat nImg;// (Mheight, Mwidth, CV_8UC3);
	unsigned char *Img = 0;
	double *Para = 0;
	bool firsttime = true;
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

		if (firsttime)
		{
			nImg.create(Mheight, Mwidth, CV_8UC3);
			firsttime = false;
		}
		for (int kk = 0; kk < nchannels; kk++)
			for (int jj = 0; jj < Mheight; jj++)
				for (int ii = 0; ii < Mwidth; ii++)
					nImg.data[ii*nchannels + jj*Mwidth*nchannels + kk] = Img[ii + jj*Mwidth + kk*Mlength];

		sprintf(Fname, "%s/U%d.png", Path, Id);
		imwrite(Fname, nImg);
	}

	delete[]Img;
	delete[]Para;

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

	for (int ii = 0; ii < pair.size(); ii += 2)
	{
		int x1 = keypoints1.at(pair[ii]).pt.x, y1 = keypoints1.at(pair[ii]).pt.y;
		int x2 = keypoints2.at(pair.at(ii + 1)).pt.x + offsetX, y2 = keypoints2.at(pair.at(ii + 1)).pt.y + offsetY;
	}

	for (int ii = 0; ii < pair.size(); ii += step)
	{
		int x1 = keypoints1.at(pair[ii]).pt.x, y1 = keypoints1.at(pair[ii]).pt.y;
		int x2 = keypoints2.at(pair.at(ii + 1)).pt.x + offsetX, y2 = keypoints2.at(pair.at(ii + 1)).pt.y + offsetY;

		cvCircle(correspond, cvPoint(x1, y1), 2, colors[ii % 9], 2), cvCircle(correspond, cvPoint(x2, y2), 2, colors[ii % 9], 2);
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

	for (int ii = 0; ii < pair.size(); ii += 2)
	{
		int x1 = keypoints1.at(pair[ii]).x, y1 = keypoints1.at(pair[ii]).y;
		int x2 = keypoints2.at(pair.at(ii + 1)).x + offsetX, y2 = keypoints2.at(pair.at(ii + 1)).y + offsetY;
	}

	for (int ii = 0; ii < pair.size(); ii += step)
	{
		int x1 = keypoints1.at(pair[ii]).x, y1 = keypoints1.at(pair[ii]).y;
		int x2 = keypoints2.at(pair.at(ii + 1)).x + offsetX, y2 = keypoints2.at(pair.at(ii + 1)).y + offsetY;
		cvLine(correspond, cvPoint(x1, y1), cvPoint(x2, y2), colors[ii % 9], 1, 4);
	}

	cvNamedWindow("Correspondence", CV_WINDOW_NORMAL);
	cvShowImage("Correspondence", correspond);
	cvWaitKey(-1);
	printf("Images closed\n");

	cvSaveImage("C:/temp/sift.png", correspond);
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

bool ReadIntrinsicResults(char *path, CameraData *AllViewsParas)
{
	//Note that visCamualSfm use different lens model than openCV or matlab or yours (inverse model)
	char Fname[200], dummy[200];
	int id = 0, lensType, width, height;
	double fx, fy, skew, u0, v0, r0, r1, r2, t0, t1, p0, p1, omega, DistCtrX, DistCtrY;

	sprintf(Fname, "%s/DevicesIntrinsics.txt", path); FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		cout << "Cannot load " << Fname << endl;
		return false;
	}
	while (fscanf(fp, "%s %d %d %d %lf %lf %lf %lf %lf ", dummy, &lensType, &width, &height, &fx, &fy, &skew, &u0, &v0) != EOF)
	{
		AllViewsParas[id].LensModel = lensType, AllViewsParas[id].width = width, AllViewsParas[id].height = height;
		AllViewsParas[id].K[0] = fx, AllViewsParas[id].K[1] = skew, AllViewsParas[id].K[2] = u0,
			AllViewsParas[id].K[3] = 0.0, AllViewsParas[id].K[4] = fy, AllViewsParas[id].K[5] = v0,
			AllViewsParas[id].K[6] = 0.0, AllViewsParas[id].K[7] = 0.0, AllViewsParas[id].K[8] = 1.0;

		GetIntrinsicFromK(AllViewsParas[id]);
		mat_invert(AllViewsParas[id].K, AllViewsParas[id].invK);
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

	return true;
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
			fprintf(fp, "%d.png %d %d %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf \n", ii, LensType, AllViewsParas[id].width, AllViewsParas[id].height, fx, fy, skew, u0, v0, r0, r1, r2, t0, t1, p0, p1);
		}
		else
		{
			omega = AllViewsParas[id].distortion[0], DistCtrX = AllViewsParas[id].distortion[1], DistCtrY = AllViewsParas[id].distortion[2];
			fprintf(fp, "%d.png %d %d %d %lf %lf %lf %lf %lf %lf %lf %lf \n", ii, LensType, AllViewsParas[id].width, AllViewsParas[id].height, fx, fy, skew, u0, v0, omega, DistCtrX, DistCtrY);
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
		int viewID = AvailViews[ii];
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
		sprintf(Fname, "%s/Dynamic/CumlativePoints_%d.txt", Path, timeID);
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
			sprintf(Fname, "%s/Corpus/PM.txt", Path);
		else
			sprintf(Fname, "%s/PM_%ds.txt", Path, timeID);
	}
	else
		if (timeID < 0)
			sprintf(Fname, "%s/Corpus/MPM.txt", Path);
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
					if (PointCorres[jj][ii] == kk) //if it has the same ID as the current point-->merge points
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
			else if (jj> 0 && PointCorres[kk][jj] != PointCorres[kk].at(jj - 1))
				MergePointCorres[kk].push_back(PointCorres[kk][jj]);
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
			curPID = PointCorres[jj][ii];
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
		int pID = Selected3DIndex[ii];
		fprintf(fp, "%.3f %.3f %.3f\n", All3D[pID].x, All3D[pID].y, All3D[pID].z);
	}
	fclose(fp);
}

void convertRTToTwist(double *R, double *T, double *twist)
{
	//OpenCV code to handle log map for SO(3)
	Map < Matrix < double, 3, 3, RowMajor > > matR(R); //matR is referenced to R;
	JacobiSVD<MatrixXd> svd(matR, ComputeFullU | ComputeFullV);
	//Matrix3d S = svd.singularValues().asDiagonal();
	matR = svd.matrixU()*svd.matrixV().transpose();//Project R to SO(3)

	double rx = R[7] - R[5], ry = R[2] - R[6], rz = R[3] - R[1];
	double s = sqrt((rx*rx + ry*ry + rz*rz)*0.25);
	double c = (R[0] + R[4] + R[8] - 1)*0.5;
	c = c > 1. ? 1. : c < -1. ? -1. : c;
	double theta = acos(c);

	if (s < 1e-5)
	{
		double t;
		if (c > 0)
			rx = ry = rz = 0.0;
		else
		{
			t = (R[0] + 1)*0.5, rx = sqrt(MAX(t, 0.));
			t = (R[4] + 1)*0.5, ry = sqrt(MAX(t, 0.))*(R[1] < 0 ? -1. : 1.);
			t = (R[8] + 1)*0.5, rz = sqrt(MAX(t, 0.))*(R[2] < 0 ? -1. : 1.);
			if (fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R[5] > 0) != (ry*rz > 0))
				rz = -rz;
			theta /= sqrt(rx*rx + ry*ry + rz*rz);
			rx *= theta, ry *= theta, rz *= theta;
		}
	}
	else
	{
		double vth = 1.0 / (2.0 * s);
		vth *= theta;
		rx *= vth; ry *= vth; rz *= vth;
	}
	twist[3] = rx, twist[4] = ry, twist[5] = rz;

	//Compute V
	double theta2 = theta* theta;
	double wx[9] = { 0.0, -rz, ry, rz, 0.0, -rx, -ry, rx, 0.0 };
	double wx2[9]; mat_mul(wx, wx, wx2, 3, 3, 3);

	double V[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
	if (theta < 1.0e-9)
		twist[0] = T[0], twist[1] = T[1], twist[2] = T[2];
	else
	{
		double A = sin(theta) / theta, B = (1.0 - cos(theta)) / theta2, C = (1.0 - A) / theta2;
		for (int ii = 0; ii < 9; ii++)
			V[ii] += B*wx[ii] + C*wx2[ii];
	}

	//solve Vt = T;
	Map < Matrix < double, 3, 3, RowMajor > > matV(V);
	Map<Vector3d> matT(T), matt(twist);
	matt = matV.lu().solve(matT);

	return;
}
void convertTwistToRT(double *twist, double *R, double *T)
{
	double t[3] = { twist[0], twist[1], twist[2] };
	double w[3] = { twist[3], twist[4], twist[5] };

	double theta = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]), theta2 = theta* theta;
	double wx[9] = { 0.0, -w[2], w[1], w[2], 0.0, -w[0], -w[1], w[0], 0.0 };
	double wx2[9]; mat_mul(wx, wx, wx2, 3, 3, 3);

	R[0] = 1.0, R[1] = 0.0, R[2] = 0.0;
	R[3] = 0.0, R[4] = 1.0, R[5] = 0.0;
	R[6] = 0.0, R[7] = 0.0, R[8] = 1.0;

	double V[9] = { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };
	if (theta < 1.0e-9)
		T[0] = t[0], T[1] = t[1], T[2] = t[2]; //Rotation is idenity
	else
	{
		double A = sin(theta) / theta, B = (1.0 - cos(theta)) / theta2, C = (1.0 - A) / theta2;
		for (int ii = 0; ii < 9; ii++)
			R[ii] += A*wx[ii] + B*wx2[ii];

		for (int ii = 0; ii < 9; ii++)
			V[ii] += B*wx[ii] + C*wx2[ii];
		mat_mul(V, t, T, 3, 3, 1);
	}

	return;
}
void convertRmatToRvec(double *R, double *r)
{
	//Project R to SO(3)
	Map < Matrix < double, 3, 3, RowMajor > > matR(R); //matR is referenced to R;
	JacobiSVD<MatrixXd> svd(matR, ComputeFullU | ComputeFullV);
	matR = svd.matrixU()*svd.matrixV().transpose();

	double rx = R[7] - R[5], ry = R[2] - R[6], rz = R[3] - R[1];
	double s = sqrt((rx*rx + ry*ry + rz*rz)*0.25);
	double c = (R[0] + R[4] + R[8] - 1)*0.5;
	c = c > 1. ? 1. : c < -1. ? -1. : c;
	double theta = acos(c);

	if (s < 1e-5)
	{
		double t;
		if (c > 0)
			rx = ry = rz = 0.0;
		else
		{
			t = (R[0] + 1)*0.5, rx = sqrt(MAX(t, 0.));
			t = (R[4] + 1)*0.5, ry = sqrt(MAX(t, 0.))*(R[1] < 0 ? -1. : 1.);
			t = (R[8] + 1)*0.5, rz = sqrt(MAX(t, 0.))*(R[2] < 0 ? -1. : 1.);
			if (fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R[5] > 0) != (ry*rz > 0))
				rz = -rz;
			theta /= sqrt(rx*rx + ry*ry + rz*rz);
			rx *= theta, ry *= theta, rz *= theta;
		}
	}
	else
	{
		double vth = 1.0 / (2.0 * s);
		vth *= theta;
		rx *= vth; ry *= vth; rz *= vth;
	}
	r[0] = rx, r[1] = ry, r[2] = rz;

	return;
}
void convertRvecToRmat(double *r, double *R)
{
	/*Mat Rmat(3, 3, CV_64F), rvec(3, 1, CV_64F);
	for (int jj = 0; jj < 3; jj++)
	rvec.at<double>(jj) = r[jj];

	Rodrigues(rvec, Rmat);

	for (int jj = 0; jj < 9; jj++)
	R[jj] = Rmat.at<double>(jj);*/

	double theta = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]), theta2 = theta* theta;
	double rx[9] = { 0.0, -r[2], r[1], r[2], 0.0, -r[0], -r[1], r[0], 0.0 };
	double rx2[9]; mat_mul(rx, rx, rx2, 3, 3, 3);

	R[0] = 1.0, R[1] = 0.0, R[2] = 0.0;
	R[3] = 0.0, R[4] = 1.0, R[5] = 0.0;
	R[6] = 0.0, R[7] = 0.0, R[8] = 1.0;

	if (theta < 1.0e-9)
		return;
	else
	{
		double A = sin(theta) / theta, B = (1.0 - cos(theta)) / theta2, C = (1.0 - A) / theta2;
		for (int ii = 0; ii < 9; ii++)
			R[ii] += A*rx[ii] + B*rx2[ii];
	}

	return;
}
void SetIntrinisc(CameraData &CamInfo, double *Intrinsic)
{
	for (int ii = 0; ii < 5; ii++)
		CamInfo.intrinsic[ii] = Intrinsic[ii];
	GetKFromIntrinsic(CamInfo);
	return;
}
void GetIntrinsicFromK(CameraData *AllViewsParas, vector<int> AvailViews)
{
	for (int ii = 0; ii < AvailViews.size(); ii++)
	{
		int viewID = AvailViews[ii];
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
		int viewID = AvailViews[ii];
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
		int viewID = AvailViews[ii];
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
void GetrtFromRT(CameraData &cam)
{
	Mat Rmat(3, 3, CV_64F), r(3, 1, CV_64F);

	for (int jj = 0; jj < 9; jj++)
		Rmat.at<double>(jj) = cam.R[jj];

	Rodrigues(Rmat, r);

	for (int jj = 0; jj < 3; jj++)
		cam.rt[jj] = r.at<double>(jj), cam.rt[3 + jj] = cam.T[jj];

	return;
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
void GetRCGL(double *R, double *T, double *Rgl, double *C)
{
	double iR[9], center[3];
	mat_invert(R, iR);

	Rgl[0] = R[0], Rgl[1] = R[1], Rgl[2] = R[2], Rgl[3] = 0.0;
	Rgl[4] = R[3], Rgl[5] = R[4], Rgl[6] = R[5], Rgl[7] = 0.0;
	Rgl[8] = R[6], Rgl[9] = R[7], Rgl[10] = R[8], Rgl[11] = 0.0;
	Rgl[12] = 0, Rgl[13] = 0, Rgl[14] = 0, Rgl[15] = 1.0;

	mat_mul(iR, T, center, 3, 3, 1);
	C[0] = -center[0], C[1] = -center[1], C[2] = -center[2];
	return;
}
void GetPosesGL(double *R, double *T, double *poseGL)
{
	poseGL[0] = R[0], poseGL[1] = R[1], poseGL[2] = R[2], poseGL[3] = 0.0;
	poseGL[4] = R[3], poseGL[5] = R[4], poseGL[6] = R[5], poseGL[7] = 0.0;
	poseGL[8] = R[6], poseGL[9] = R[7], poseGL[10] = R[8], poseGL[11] = 0.0;
	poseGL[12] = 0, poseGL[13] = 0, poseGL[14] = 0, poseGL[15] = 1.0;

	//Center = -iR*T 
	double iR[9], center[3];
	mat_invert(R, iR);
	mat_mul(iR, T, center, 3, 3, 1);
	poseGL[16] = -center[0], poseGL[17] = -center[1], poseGL[18] = -center[2];

	return;
}
void GetTfromC(CameraData &camInfo)
{
	double T[3];

	mat_mul(camInfo.R, camInfo.camCenter, T, 3, 3, 1);
	camInfo.T[0] = -T[0], camInfo.T[1] = -T[1], camInfo.T[2] = -T[2];
	return;
}
void InvertCameraPose(double *R, double *T, double *iR, double *iT)
{
	double RT[16] = { R[0], R[1], R[2], T[0],
		R[3], R[4], R[5], T[1],
		R[6], R[7], R[8], T[2],
		0, 0, 0, 1 };

	double iRT[16];
	mat_invert(RT, iRT, 4);

	iR[0] = iRT[0], iR[1] = iRT[1], iR[2] = iRT[2], iT[0] = iRT[3];
	iR[3] = iRT[4], iR[4] = iRT[5], iR[5] = iRT[6], iT[1] = iRT[7];
	iR[6] = iRT[8], iR[7] = iRT[9], iR[8] = iRT[10], iT[2] = iRT[11];

	return;
}
void CopyCamereInfo(CameraData Src, CameraData &Dst, bool Extrinsic)
{
	int ii;
	for (ii = 0; ii < 9; ii++)
		Dst.K[ii] = Src.K[ii];
	for (ii = 0; ii < 7; ii++)
		Dst.distortion[ii] = Src.distortion[ii];
	for (ii = 0; ii < 5; ii++)
		Dst.intrinsic[ii] = Src.intrinsic[ii];

	Dst.LensModel = Src.LensModel;
	Dst.ninlierThresh = Src.ninlierThresh;
	Dst.threshold = Src.threshold;
	Dst.width = Src.width, Dst.height = Src.height;

	if (Extrinsic)
	{
		for (ii = 0; ii < 9; ii++)
			Dst.R[ii] = Src.R[ii];
		for (ii = 0; ii < 3; ii++)
			Dst.T[ii] = Src.T[ii];
		for (ii = 0; ii < 6; ii++)
			Dst.rt[ii] = Src.rt[ii];
		for (ii = 0; ii < 6; ii++)
			Dst.wt[ii] = Src.wt[ii];
		for (ii = 0; ii < 12; ii++)
			Dst.P[ii] = Src.P[ii];
		for (ii = 0; ii < 16; ii++)
			Dst.Rgl[ii] = Src.Rgl[ii];
		for (ii = 0; ii < 3; ii++)
			Dst.camCenter[ii] = Src.camCenter[ii];
	}
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
		int viewID = AvailViews[ii];
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
void AssembleRT(double *R, double *T, double *RT, bool GivenCenter)
{
	if (!GivenCenter)
	{
		RT[0] = R[0], RT[1] = R[1], RT[2] = R[2], RT[3] = T[0];
		RT[4] = R[3], RT[5] = R[4], RT[6] = R[5], RT[7] = T[1];
		RT[8] = R[6], RT[9] = R[7], RT[10] = R[8], RT[11] = T[2];
	}
	else//RT = [R, -R*C];
	{
		double mT[3];
		mat_mul(R, T, mT, 3, 3, 1);
		RT[0] = R[0], RT[1] = R[1], RT[2] = R[2], RT[3] = -mT[0];
		RT[4] = R[3], RT[5] = R[4], RT[6] = R[5], RT[7] = -mT[1];
		RT[8] = R[6], RT[9] = R[7], RT[10] = R[8], RT[11] = -mT[2];
	}
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
void AssembleP(double *K, double *RT, double *P)
{
	mat_mul(K, RT, P, 3, 3, 4);
	return;
}

#define QW_ZERO 1e-6
void Rotation2Quaternion(double *R, double *q)
{
	double r11 = R[0], r12 = R[1], r13 = R[2];
	double r21 = R[3], r22 = R[4], r23 = R[5];
	double r31 = R[6], r32 = R[7], r33 = R[8];

	double qw = sqrt(abs(1.0 + r11 + r22 + r33)) / 2;
	double qx, qy, qz;
	if (qw > QW_ZERO)
	{
		qx = (r32 - r23) / 4 / qw;
		qy = (r13 - r31) / 4 / qw;
		qz = (r21 - r12) / 4 / qw;
	}
	else
	{
		double d = sqrt((r12*r12*r13*r13 + r12*r12*r23*r23 + r13*r13*r23*r23));
		qx = r12*r13 / d;
		qy = r12*r23 / d;
		qz = r13*r23 / d;
	}

	q[0] = qw, q[1] = qx, q[2] = qy, q[3] = qz;

	normalize(q, 4);
}
void Quaternion2Rotation(double *q, double *R)
{
	normalize(q, 4);

	double qw = q[0], qx = q[1], qy = q[2], qz = q[3];

	R[0] = 1.0 - 2 * qy*qy - 2 * qz*qz;
	R[1] = 2 * qx*qy - 2 * qz*qw;
	R[2] = 2 * qx*qz + 2 * qy*qw;

	R[3] = 2 * qx*qy + 2 * qz*qw;
	R[4] = 1.0 - 2 * qx*qx - 2 * qz*qz;
	R[5] = 2 * qz*qy - 2 * qx*qw;

	R[6] = 2 * qx*qz - 2 * qy*qw;
	R[7] = 2 * qy*qz + 2 * qx*qw;
	R[8] = 1.0 - 2 * qx*qx - 2 * qy*qy;
}
void QuaternionLinearInterp(double *quad1, double *quad2, double *quadi, double u)
{
	const double DOT_THRESHOLD = 0.9995;

	double C_phi = dotProduct(quad1, quad2);
	if (C_phi > DOT_THRESHOLD) //do linear interp
		for (int ii = 0; ii < 4; ii++)
			quadi[ii] = (1.0 - u)*quad1[ii] + u*quad2[ii];
	else
	{
		double phi = acos(C_phi);
		double  S_phi = sin(phi), S_1uphi = sin((1.0 - u)*phi) / S_phi, S_uphi = sin(u*phi) / S_phi;
		for (int ii = 0; ii < 4; ii++)
			quadi[ii] = S_1uphi*quad1[ii] + S_uphi*quad2[ii];
		normalize(quadi, 4);
		if (dotProduct(quad1, quadi) < 0.0)
			for (int ii = 0; ii < 4; ii++)
				quadi[ii] = -quadi[ii];
	}
	return;
}

int Pose_se_BSplineInterpolation(char *Fname1, char *Fname2, int nsamples, char *Fname3)
{
	int nControls, nbreaks, SplineOrder, se3;

	//Read data
	FILE *fp = fopen(Fname1, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", Fname1);
		return 1;
	}
	fscanf(fp, "%d %d %d %d\n", &nControls, &nbreaks, &SplineOrder, &se3);

	double *BreakLoc = new double[nbreaks];
	double *ControlLoc = new double[nControls];
	double *ControlPose = new double[nControls * 6];
	for (int ii = 0; ii < nControls; ii++)
	{
		fscanf(fp, "%lf ", &ControlLoc[ii]);
		for (int jj = 0; jj < 6; jj++)
			fscanf(fp, "%lf ", &ControlPose[ii * 6 + jj]);
	}
	for (int ii = 0; ii < nbreaks; ii++)
		fscanf(fp, "%lf", &BreakLoc[ii]);
	fclose(fp);

	//Bspline generator
	gsl_bspline_workspace *bw = gsl_bspline_alloc(SplineOrder, nbreaks);
	gsl_vector *Bi = gsl_vector_alloc(nControls);
	gsl_vector *gsl_BreakPts = gsl_vector_alloc(nbreaks);

	for (int ii = 0; ii < nbreaks; ii++)
		gsl_vector_set(gsl_BreakPts, ii, BreakLoc[ii]);
	gsl_bspline_knots(gsl_BreakPts, bw);

	//Start interpolation pose
	double *SampleLoc = new double[nsamples];
	double step = (BreakLoc[nbreaks - 1] - BreakLoc[0]) / (nsamples - 1);
	for (int ii = 0; ii < nsamples; ii++)
		SampleLoc[ii] = BreakLoc[0] + step*ii;

	int ActingID[4];
	double twist[6], tr[6], R[9], T[3], poseGL[19];

	fp = fopen(Fname2, "w+");
	for (int ii = 0; ii < nsamples; ii++)
	{
		FindActingControlPts(SampleLoc[ii], ActingID, nControls, bw, Bi, SplineOrder, 0);

		gsl_bspline_eval(SampleLoc[ii], Bi, bw);
		for (int jj = 0; jj < 6; jj++)
		{
			if (se3 == 1)
			{
				twist[jj] = 0.0;
				for (int kk = 0; kk < 4; kk++)
					twist[jj] += ControlPose[jj + 6 * ActingID[kk]] * gsl_vector_get(Bi, ActingID[kk]);
			}
			else
			{
				tr[jj] = 0;
				for (int kk = 0; kk < 4; kk++)
					tr[jj] += ControlPose[jj + 6 * ActingID[kk]] * gsl_vector_get(Bi, ActingID[kk]);
			}
		}

		if (se3 == 1)
			convertTwistToRT(twist, R, T);
		else
		{
			convertRvecToRmat(tr + 3, R);
			for (int jj = 0; jj < 3; jj++)
				T[jj] = tr[jj];
		}

		GetPosesGL(R, T, poseGL);

		fprintf(fp, "%d ", (int)SampleLoc[ii]);
		for (int jj = 0; jj < 19; jj++)
			fprintf(fp, "%.16e ", poseGL[jj]);
		fprintf(fp, "\n");
	}
	fclose(fp);


	fp = fopen(Fname3, "w+");
	for (int ii = 0; ii < nControls; ii++)
	{
		if (se3 == 1)
			convertTwistToRT(&ControlPose[6 * ii], R, T);
		else
		{
			convertRvecToRmat(&ControlPose[6 * ii], R);
			for (int jj = 0; jj < 3; jj++)
				T[jj] = ControlPose[6 * ii + jj];
		}
		GetPosesGL(R, T, poseGL);

		fprintf(fp, "%d ", (int)(ControlLoc[ii] + 0.5));
		for (int jj = 0; jj < 19; jj++)
			fprintf(fp, "%.16e ", poseGL[jj]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	return 0;
}
int Pose_se_DCTInterpolation(char *FnameIn, char *FnameOut, int nsamples)
{
	int startFrame, nCoeffs, sampleStep;

	FILE *fp = fopen(FnameIn, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", FnameIn);
		return 1;
	}

	fscanf(fp, "%d %d %d ", &startFrame, &nCoeffs, &sampleStep);
	double *C = new double[6 * nCoeffs];
	for (int jj = 0; jj < 6; jj++)
		for (int ii = 0; ii < nCoeffs; ii++)
			fscanf(fp, "%lf ", &C[ii + jj*nCoeffs]);
	fclose(fp);

	double *iBi = new double[nCoeffs];
	double twist[6], R[9], T[3], poseGL[19];
	double stopFrame = sampleStep*(nCoeffs - 1) + startFrame, resampleStep = 1.0*nCoeffs / (stopFrame - startFrame);

	fp = fopen(FnameOut, "w+");
	for (int ii = 0; ii < nsamples; ii++)
	{
		double loc = 1.0*ii * resampleStep; //linspace(0, ncoeffs-1, nsamples)
		GenerateiDCTBasis(iBi, nCoeffs, loc);
		for (int jj = 0; jj < 6; jj++)
		{
			twist[jj] = 0.0;
			for (int ii = 0; ii < nCoeffs; ii++)
				twist[jj] += C[ii + jj*nCoeffs] * iBi[ii];
		}

		convertTwistToRT(twist, R, T);
		GetPosesGL(R, T, poseGL);

		loc = loc / resampleStep + startFrame; //linspace(startF, stopF, nsamples)
		fprintf(fp, "%d ", (int)(loc + 0.5));
		for (int jj = 0; jj < 19; jj++)
			fprintf(fp, "%.16e ", poseGL[jj]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	delete[]iBi;
	return 0;
}
void ComputeInterCamerasPose(double *R1, double *T1, double *R2, double *T2, double *R21, double *T21)
{
	//C2 = RT21*C1
	double RT1[16] = { R1[0], R1[1], R1[2], T1[0],
		R1[3], R1[4], R1[5], T1[1],
		R1[6], R1[7], R1[8], T1[2],
		0, 0, 0, 1 };
	double RT2[16] = { R2[0], R2[1], R2[2], T2[0],
		R2[3], R2[4], R2[5], T2[1],
		R2[6], R2[7], R2[8], T2[2],
		0, 0, 0, 1 };

	double iRT2[16], RT21[16];
	mat_invert(RT2, iRT2, 4);

	mat_mul(RT1, iRT2, RT21, 4, 4, 4);

	R21[0] = RT21[0], R21[1] = RT21[1], R21[2] = RT21[2], T21[0] = RT21[3];
	R21[3] = RT21[4], R21[4] = RT21[5], R21[5] = RT21[6], T21[1] = RT21[7];
	R21[6] = RT21[8], R21[7] = RT21[9], R21[8] = RT21[10], T21[2] = RT21[11];

	return;
}
// Author: Xiaotao Duan
//
// This library contains image processing method to detect
// image blurriness.
//
// This library is *not* thread safe because static memory is
// used for performance.
//
// A method to detect whether a given image is blurred or not.
// The algorithm is based on H. Tong, M. Li, H. Zhang, J. He,
// and C. Zhang. "Blur detection for digital images using wavelet
// transform".
//
// To achieve better performance on client side, the method
// is running on four 128x128 portions which compose the 256x256
// central area of the given image. On Nexus One, average time
// to process a single image is ~5 milliseconds.
static const int kDecomposition = 3;
static const int kThreshold = 35;

static const int kMaximumWidth = 2048;
static const int kMaximumHeight = 1536;

static int32_t _smatrix[kMaximumWidth * kMaximumHeight];
static int32_t _arow[kMaximumWidth > kMaximumHeight ? kMaximumWidth : kMaximumHeight];

// Does Haar Wavelet Transformation in place on a given row of a matrix. The matrix is in size of matrix_height * matrix_width and represented in a linear array. 
//Parameter offset_row indicates transformation is performed on which row. offset_column and num_columns indicate column range of the given row.
inline void Haar1DX(int* matrix, int matrix_height, int matrix_width, int offset_row, int offset_column, int num_columns) {
	int32_t* ptr_a = _arow;
	int32_t* ptr_matrix = matrix + offset_row * matrix_width + offset_column;
	int half_num_columns = num_columns / 2;

	int32_t* a_tmp = ptr_a;
	int32_t* matrix_tmp = ptr_matrix;
	for (int j = 0; j < half_num_columns; ++j) {
		*a_tmp++ = (matrix_tmp[0] + matrix_tmp[1]) / 2;
		matrix_tmp += 2;
	}

	int32_t* average = ptr_a;
	a_tmp = ptr_a + half_num_columns;
	matrix_tmp = ptr_matrix;
	for (int j = 0; j < half_num_columns; ++j) {
		*a_tmp++ = *matrix_tmp - *average++;
		matrix_tmp += 2;
	}

	memcpy(ptr_matrix, ptr_a, sizeof(int32_t)* num_columns);
}
// Does Haar Wavelet Transformation in place on a given column of a matrix.
inline void Haar1DY(int* matrix, int matrix_height, int matrix_width, int offset_column, int offset_row, int num_rows) {
	int32_t* ptr_a = _arow;
	int32_t* ptr_matrix = matrix + offset_row * matrix_width + offset_column;
	int half_num_rows = num_rows / 2;
	int two_line_width = matrix_width * 2;

	int32_t* a_tmp = ptr_a;
	int32_t* matrix_tmp = ptr_matrix;
	for (int j = 0; j < half_num_rows; ++j) {
		*a_tmp++ = (matrix_tmp[matrix_width] + matrix_tmp[0]) / 2;
		matrix_tmp += two_line_width;
	}

	int32_t* average = ptr_a;
	a_tmp = ptr_a + half_num_rows;
	matrix_tmp = ptr_matrix;
	for (int j = 0; j < num_rows; j += 2) {
		*a_tmp++ = *matrix_tmp - *average++;
		matrix_tmp += two_line_width;
	}

	for (int j = 0; j < num_rows; ++j) {
		*ptr_matrix = *ptr_a++;
		ptr_matrix += matrix_width;
	}
}
// Does Haar Wavelet Transformation in place for a specified area of a matrix. The matrix size is specified by matrix_width and matrix_height.
// The area on which the transformation is performed is specified by offset_column, num_columns, offset_row and num_rows.
void Haar2D(int* matrix, int matrix_height, int matrix_width, int offset_column, int num_columns, int offset_row, int num_rows) {
	for (int i = offset_row; i < offset_row + num_rows; ++i) {
		Haar1DX(matrix, matrix_height, matrix_width, i, offset_column, num_columns);
	}

	for (int i = offset_column; i < offset_column + num_columns; ++i){
		Haar1DY(matrix, matrix_height, matrix_width, i, offset_row, num_rows);
	}
}
// Reads in a given matrix, does first round HWT and outputs result matrix into target array. This function is used for optimization by avoiding a memory copy. 
//The input matrix has height rows and width columns. The transformation is performed on the given area specified by offset_column, num_columns, offset_row, num_rows. 
// After transformation, the output matrix has num_columns columns and num_rows rows.
void HwtFirstRound(const unsigned char* const data, int height, int width, int offset_column, int num_columns, int offset_row, int num_rows, int32_t* matrix)
{
	int32_t* ptr_a = _arow;
	const unsigned char* ptr_data = data + offset_row * width + offset_column;
	int half_num_columns = num_columns / 2;

	for (int i = 0; i < num_rows; ++i)
	{
		int32_t* a_tmp = ptr_a;
		const unsigned char* data_tmp = ptr_data;
		for (int j = 0; j < half_num_columns; ++j)
		{
			*a_tmp++ = (int32_t)((data_tmp[0] + data_tmp[1]) / 2);
			data_tmp += 2;
		}

		int32_t* average = ptr_a;
		a_tmp = ptr_a + half_num_columns;
		data_tmp = ptr_data;
		for (int j = 0; j < half_num_columns; ++j)
		{
			*a_tmp++ = *data_tmp - *average++;
			data_tmp += 2;
		}

		int32_t* ptr_matrix = matrix + i * num_columns;
		a_tmp = ptr_a;
		for (int j = 0; j < num_columns; ++j)
		{
			*ptr_matrix++ = *a_tmp++;
		}

		ptr_data += width;
	}

	// Column transformation does not involve input data.
	for (int i = 0; i < num_columns; ++i)
		Haar1DY(matrix, num_rows, num_columns, i, 0, num_rows);
}
// Returns the weight of a given point in a certain scale of a matrix after wavelet transformation.
// The point is specified by k and l which are y and x coordinate respectively. Parameter scale tells in which scale the weight is computed, must be 1, 2 or 3 which stands respectively for 1/2, 1/4, and 1/8 of original size.
int ComputeEdgePointWeight(int* matrix, int width, int height, int k, int l, int scale) {
	int r = k >> scale;
	int c = l >> scale;
	int window_row = height >> scale;
	int window_column = width >> scale;

	int v_top_right = pow(matrix[r * width + c + window_column], 2);
	int v_bot_left = pow(matrix[(r + window_row) * width + c], 2);
	int v_bot_right = pow(matrix[(r + window_row) * width + c + window_column], 2);

	int v = sqrt(v_top_right + v_bot_left + v_bot_right);
	return v;
}
// Computes point with maximum weight for a given local window for a given scale. Parameter scaled_width and scaled_height define scaled image size of a certain decomposition level. 
//The window size is defined by window_size. Output value k and l store row (y coordinate) and column (x coordinate) respectively of the point with maximum weight. The maximum weight is returned.
int ComputeLocalMaximum(int* matrix, int width, int height, int scaled_width, int scaled_height, int top, int left, int window_size, int* k, int* l) {
	int max = -1;
	*k = top;
	*l = left;

	for (int i = 0; i < window_size; ++i) {
		for (int j = 0; j < window_size; ++j) {
			int r = top + i;
			int c = left + j;

			int v_top_right = abs(matrix[r * width + c + scaled_width]);
			int v_bot_left = abs(matrix[(r + scaled_height) * width + c]);
			int v_bot_right =
				abs(matrix[(r + scaled_height) * width + c + scaled_width]);
			int v = v_top_right + v_bot_left + v_bot_right;

			if (v > max) {
				max = v;
				*k = r;
				*l = c;
			}
		}
	}

	int r = *k;
	int c = *l;
	int v_top_right = pow(matrix[r * width + c + scaled_width], 2);
	int v_bot_left = pow(matrix[(r + scaled_height) * width + c], 2);
	int v_bot_right = pow(matrix[(r + scaled_height) * width + c + scaled_width], 2);
	int v = sqrt(v_top_right + v_bot_left + v_bot_right);

	return v;
}
// Detects blurriness of a transformed matrix. Blur confidence and extent will be returned through blur_conf and blur_extent. 1 is returned while input matrix is blurred.
int DetectBlur(int* matrix, int width, int height, float* blur_conf, float* blur_extent, float blurThresh) {
	int nedge = 0;
	int nda = 0;
	int nrg = 0;
	int nbrg = 0;

	// For each scale
	for (int current_scale = kDecomposition; current_scale > 0; --current_scale) {
		int scaled_width = width >> current_scale;
		int scaled_height = height >> current_scale;
		int window_size = 16 >> current_scale;  // 2, 4, 8
		// For each window
		for (int r = 0; r + window_size < scaled_height; r += window_size) {
			for (int c = 0; c + window_size < scaled_width; c += window_size) {
				int k, l;
				int emax = ComputeLocalMaximum(matrix, width, height,
					scaled_width, scaled_height, r, c, window_size, &k, &l);
				if (emax > kThreshold) {
					int emax1, emax2, emax3;
					switch (current_scale) {
					case 1:
						emax1 = emax;
						emax2 = ComputeEdgePointWeight(matrix, width, height,
							k << current_scale, l << current_scale, 2);
						emax3 = ComputeEdgePointWeight(matrix, width, height,
							k << current_scale, l << current_scale, 3);
						break;
					case 2:
						emax1 = ComputeEdgePointWeight(matrix, width, height,
							k << current_scale, l << current_scale, 1);
						emax2 = emax;
						emax3 = ComputeEdgePointWeight(matrix, width, height,
							k << current_scale, l << current_scale, 3);
						break;
					case 3:
						emax1 = ComputeEdgePointWeight(matrix, width, height,
							k << current_scale, l << current_scale, 1);
						emax2 = ComputeEdgePointWeight(matrix, width, height,
							k << current_scale, l << current_scale, 2);
						emax3 = emax;
						break;
					}

					nedge++;
					if (emax1 > emax2 && emax2 > emax3) {
						nda++;
					}
					if (emax1 < emax2 && emax2 < emax3) {
						nrg++;
						if (emax1 < kThreshold) {
							nbrg++;
						}
					}
					if (emax2 > emax1 && emax2 > emax3) {
						nrg++;
						if (emax1 < kThreshold) {
							nbrg++;
						}
					}
				}
			}
		}
	}

	// TODO(xiaotao): No edge point at all, blurred or not?
	float per = nedge == 0 ? 0 : (float)nda / nedge;

	*blur_conf = per;
	*blur_extent = (float)nbrg / nrg;

	return per < blurThresh;
}
// Detects blurriness of a given portion of a luminance matrix.
int IsBlurredInner(const unsigned char* const luminance, const int width, const int height, const int left, const int top, const int width_wanted, const int height_wanted, float* const blur, float* const extent, float blurThresh) {
	int32_t* matrix = _smatrix;

	HwtFirstRound(luminance, height, width, left, width_wanted, top, height_wanted, matrix);
	Haar2D(matrix, height_wanted, width_wanted, 0, width_wanted >> 1, 0, height_wanted >> 1);
	Haar2D(matrix, height_wanted, width_wanted, 0, width_wanted >> 2, 0, height_wanted >> 2);

	int blurred = DetectBlur(matrix, width_wanted, height_wanted, blur, extent, blurThresh);

	return blurred;
}
int IsBlurred(const unsigned char* const luminance, const int width, const int height, float &blur, float &extent, float blurThresh) {

	int desired_width = min(kMaximumWidth, width);
	int desired_height = min(kMaximumHeight, height);
	int left = (width - desired_width) >> 1;
	int top = (height - desired_height) >> 1;

	float conf1, extent1;
	int blur1 = IsBlurredInner(luminance, width, height, left, top, desired_width >> 1, desired_height >> 1, &conf1, &extent1, blurThresh);
	float conf2, extent2;
	int blur2 = IsBlurredInner(luminance, width, height, left + (desired_width >> 1), top, desired_width >> 1, desired_height >> 1, &conf2, &extent2, blurThresh);
	float conf3, extent3;
	int blur3 = IsBlurredInner(luminance, width, height, left, top + (desired_height >> 1), desired_width >> 1, desired_height >> 1, &conf3, &extent3, blurThresh);
	float conf4, extent4;
	int blur4 = IsBlurredInner(luminance, width, height, left + (desired_width >> 1), top + (desired_height >> 1), desired_width >> 1, desired_height >> 1, &conf4, &extent4, blurThresh);

	blur = (conf1 + conf2 + conf3 + conf4) / 4;
	extent = (extent1 + extent2 + extent3 + extent4) / 4;
	return blur < blurThresh;
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

int GenerateVisualSFMinput(char *path, int startFrame, int stopFrame, int npts)
{
	//This function will write down points with upper left coor.
	int ii, jj, mm, kk;
	char Fname[200], Fname1[200], Fname2[200];

	//nptsxnviews
	int  nframes = stopFrame - startFrame + 1;
	Point2d *Correspondences = new Point2d[npts*nframes];

	for (ii = startFrame; ii <= stopFrame; ii++)
	{
		sprintf(Fname, "%s/CV_%d.txt", path, ii); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("Cannot open %s", Fname);
			return 0;
		}
		for (jj = 0; jj < npts; jj++)
			fscanf(fp, "%lf %lf ", &Correspondences[ii + jj*nframes].x, &Correspondences[ii + jj*nframes].y);
		fclose(fp);
	}

	//Write out sift format
	for (jj = startFrame; jj <= stopFrame; jj++)
	{
		sprintf(Fname, "%s/%d.sift", path, jj);	FILE *fp = fopen(Fname, "w+");
		fprintf(fp, "%d 128\n", npts);
		for (ii = 0; ii < npts; ii++)
		{
			fprintf(fp, "%.6f %.6f 0.0 0.0\n", Correspondences[jj + ii*nframes].x, Correspondences[jj + ii*nframes].y);
			for (kk = 0; kk < 128; kk++)
				fprintf(fp, "%d ", 0);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	//Write out pair of correspondences
	vector<int> matchID;
	sprintf(Fname, "%s/FeatureMatches.txt", path); FILE *fp = fopen(Fname, "w+");
	for (jj = startFrame; jj < stopFrame; jj++)
	{
		for (ii = jj + 1; ii <= stopFrame; ii++)
		{
			sprintf(Fname1, "%s/%d.jpg", path, jj);
			sprintf(Fname2, "%s/%d.jpg", path, ii);

			fprintf(fp, "%s\n", Fname1);
			fprintf(fp, "%s\n", Fname2);
			matchID.clear();
			for (mm = 0; mm < npts; mm++)
				if (Correspondences[mm*nframes + jj].x > 0.0 && Correspondences[mm*nframes + jj].y > 0.0 && Correspondences[mm*nframes + ii].x > 0.0 && Correspondences[mm*nframes + ii].y > 0.0)
					matchID.push_back(mm);

			fprintf(fp, "%d\n", matchID.size());
			for (mm = 0; mm < matchID.size(); mm++)
				fprintf(fp, "%d ", matchID.at(mm));
			fprintf(fp, "\n");
			for (mm = 0; mm < matchID.size(); mm++)
				fprintf(fp, "%d ", matchID.at(mm));
			fprintf(fp, "\n\n");
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	delete[]Correspondences;
	return 0;
}
bool loadNVMLite(const char *filepath, Corpus &CorpusData, int sharedIntrinsics, int nHDs, int nVGAs, int nPanels)
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
	CorpusData.nCameras = nviews;
	CorpusData.camera = new CameraData[nviews];
	double Quaterunion[4], CamCenter[3], T[3];
	for (int ii = 0; ii < nviews; ii++)
	{
		string filename;
		double f;
		vector<double> q(4), c(3), d(2);
		ifs >> filename >> f >> Quaterunion[0] >> Quaterunion[1] >> Quaterunion[2] >> Quaterunion[3] >> CamCenter[0] >> CamCenter[1] >> CamCenter[2] >> d[0] >> d[1];

		int viewID, panelID, camID, width, height;
		std::size_t posDot = filename.find(".");
		if (posDot < 5)
		{
			std::size_t pos = filename.find(".ppm");
			if (pos != string::npos)
			{
				filename.erase(pos, 4);
				const char * str = filename.c_str();
				viewID = atoi(str);
			}
			else
			{
				pos = filename.find(".jpg");
				if (pos != string::npos)
				{
					filename.erase(pos, 4);
					const char * str = filename.c_str();
					viewID = atoi(str);
				}
				else
				{
					printf("cannot find %s\n", filename.c_str());
					abort();
				}
			}
		}
		else
		{
			std::size_t pos1 = filename.find("_");
			string PanelName; PanelName = filename.substr(0, 2);
			const char * str = PanelName.c_str();
			panelID = atoi(str);

			string CamName; CamName = filename.substr(pos1 + 1, 2);
			str = CamName.c_str();
			camID = atoi(str);

			viewID = panelID == 0 ? camID : nHDs + nVGAs*(panelID - 1) + camID - 1;
			width = viewID > nHDs ? 640 : 1920, height = viewID > nHDs ? 480 : 1080;
		}

		CorpusData.camera[viewID].intrinsic[0] = f, CorpusData.camera[viewID].intrinsic[1] = f,
			CorpusData.camera[viewID].intrinsic[2] = 0, CorpusData.camera[viewID].intrinsic[3] = width / 2, CorpusData.camera[viewID].intrinsic[4] = height / 2;

		ceres::QuaternionToRotation(Quaterunion, CorpusData.camera[viewID].R);
		mat_mul(CorpusData.camera[viewID].R, CamCenter, T, 3, 3, 1); //t = -RC
		CorpusData.camera[viewID].T[0] = -T[0], CorpusData.camera[viewID].T[1] = -T[1], CorpusData.camera[viewID].T[2] = -T[2];

		CorpusData.camera[viewID].LensModel = 0;
		CorpusData.camera[viewID].distortion[0] = -d[0];
		for (int jj = 1; jj < 7; jj++)
			CorpusData.camera[viewID].distortion[jj] = 0.0;

		GetrtFromRT(CorpusData.camera[viewID]);
		GetKFromIntrinsic(CorpusData.camera[viewID]);
		AssembleP(CorpusData.camera[viewID]);
	}

	return true;
}
bool loadNVM(const char *Fname, Corpus &CorpusData, vector<Point2i> &ImgSize, int sharedIntrinsics, vector<KeyPoint> *AllKeyPts, Mat *AllDesc)
{
	ifstream ifs(Fname);
	if (ifs.fail())
	{
		cerr << "Cannot load " << Fname << endl;
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
	cout << "Loading nvm cameras" << endl;
	int nviews;
	ifs >> nviews;
	if (nviews <= 1)
	{
		cerr << "# of cameras must be more than 1." << endl;
		return false;
	}
	CorpusData.nCameras = nviews;
	CorpusData.camera = new CameraData[nviews + 1];//1 is just in case camera start at 1
	double Quaterunion[4], CamCenter[3], T[3];
	vector<int> CameraOrder;

	for (int ii = 0; ii < nviews; ii++)
		CorpusData.camera[ii].valid = false;

	for (int ii = 0; ii < nviews; ii++)
	{
		string filename;
		double f;
		vector<double> q(4), c(3), d(2);
		ifs >> filename >> f >> Quaterunion[0] >> Quaterunion[1] >> Quaterunion[2] >> Quaterunion[3] >> CamCenter[0] >> CamCenter[1] >> CamCenter[2] >> d[0] >> d[1];

		int viewID;
		std::size_t posDot = filename.find(".");
		if (posDot < 5)
		{
			std::size_t pos = filename.find(".ppm");
			if (pos != string::npos)
			{
				filename.erase(pos, 4);
				const char * str = filename.c_str();
				viewID = atoi(str);
			}
			else
			{
				pos = filename.find(".jpg");
				if (pos != string::npos)
				{
					filename.erase(pos, 4);
					const char * str = filename.c_str();
					viewID = atoi(str);
				}
				else
				{
					printf("cannot find %s\n", filename.c_str());
					abort();
				}
			}
		}
		CameraOrder.push_back(viewID);

		CorpusData.camera[viewID].valid = true;
		CorpusData.camera[viewID].intrinsic[0] = f, CorpusData.camera[viewID].intrinsic[1] = f,
			CorpusData.camera[viewID].intrinsic[2] = 0, CorpusData.camera[viewID].intrinsic[3] = 1.0*ImgSize[viewID].x / 2 - 0.5, CorpusData.camera[viewID].intrinsic[4] = 1.0*ImgSize[viewID].y / 2 - 0.5;
		CorpusData.camera[viewID].width = ImgSize[viewID].x, CorpusData.camera[viewID].height = ImgSize[viewID].y;

		ceres::QuaternionToRotation(Quaterunion, CorpusData.camera[viewID].R);
		mat_mul(CorpusData.camera[viewID].R, CamCenter, T, 3, 3, 1); //t = -RC
		CorpusData.camera[viewID].T[0] = -T[0], CorpusData.camera[viewID].T[1] = -T[1], CorpusData.camera[viewID].T[2] = -T[2];

		CorpusData.camera[viewID].LensModel = 0;
		CorpusData.camera[viewID].distortion[0] = -d[0];
		for (int jj = 1; jj < 7; jj++)
			CorpusData.camera[viewID].distortion[jj] = 0.0;

		GetrtFromRT(CorpusData.camera[viewID]);
		GetKFromIntrinsic(CorpusData.camera[viewID]);
		AssembleP(CorpusData.camera[viewID]);
	}

	cout << "Loading nvm points" << endl;
	int nPoints, viewID, pid;
	ifs >> nPoints;
	CorpusData.n3dPoints = nPoints;
	CorpusData.xyz.reserve(nPoints);
	CorpusData.viewIdAll3D.reserve(nPoints);
	CorpusData.pointIdAll3D.reserve(nPoints);
	CorpusData.uvAll3D.reserve(nPoints);
	CorpusData.scaleAll3D.reserve(nPoints);

	Point2d uv;
	Point3d xyz;
	Point3i rgb;
	vector<int>viewID3D, pid3D;
	vector<Point2d> uv3D;
	vector<double> scale3D;

	FeatureDesc desci;
	if (AllDesc != NULL)
	{
		CorpusData.scaleAllViews2 = new vector<double>[CorpusData.nCameras];
		CorpusData.uvAllViews2 = new vector<Point2d>[CorpusData.nCameras];
		CorpusData.threeDIdAllViews2 = new vector<int>[CorpusData.nCameras];
		CorpusData.DescAllViews2 = new vector<FeatureDesc>[CorpusData.nCameras];
		for (int ii = 0; ii < CorpusData.nCameras; ii++)
		{
			CorpusData.scaleAllViews2[ii].reserve(5000);
			CorpusData.uvAllViews2[ii].reserve(5000);
			CorpusData.threeDIdAllViews2[ii].reserve(5000);
			CorpusData.DescAllViews2[ii].reserve(5000);
		}
	}

	for (int i = 0; i < nPoints; i++)
	{
		viewID3D.clear(), pid3D.clear(), uv3D.clear(), scale3D.clear();
		ifs >> xyz.x >> xyz.y >> xyz.z >> rgb.x >> rgb.y >> rgb.z;

		CorpusData.xyz.push_back(xyz);
		CorpusData.rgb.push_back(rgb);

		ifs >> nviews;
		//if (AllDesc != NULL) Desc3Di.create(nviews, 128, CV_32F);

		int cur3DID = CorpusData.viewIdAll3D.size();
		for (int ii = 0; ii < nviews; ii++)
		{
			ifs >> viewID >> pid >> uv.x >> uv.y;
			uv.x += 0.5*(CorpusData.camera[CameraOrder[viewID]].width - 1.0);
			uv.y += 0.5*(CorpusData.camera[CameraOrder[viewID]].height - 1.0);

			viewID3D.push_back(CameraOrder[viewID]);
			pid3D.push_back((pid));
			uv3D.push_back(uv);
			scale3D.push_back(1.0);

			if (AllDesc != NULL)
			{
				//for (int jj = 0; jj < 128; jj++)  Desc3Di.at<float>(ii, jj) = AllDesc[viewID].at<float>(pid, jj);
				for (int jj = 0; jj < 128; jj++)
					desci.desc[jj] = AllDesc[CameraOrder[viewID]].at<float>(pid, jj);

				CorpusData.uvAllViews2[CameraOrder[viewID]].push_back(uv);
				CorpusData.scaleAllViews2[CameraOrder[viewID]].push_back(AllKeyPts[CameraOrder[viewID]][pid].size);
				CorpusData.threeDIdAllViews2[CameraOrder[viewID]].push_back(cur3DID);
				CorpusData.DescAllViews2[CameraOrder[viewID]].push_back(desci);
			}
		}

		CorpusData.viewIdAll3D.push_back(viewID3D);
		CorpusData.uvAll3D.push_back(uv3D);
		CorpusData.scaleAll3D.push_back(scale3D);
		CorpusData.pointIdAll3D.push_back(pid3D);
		//if (AllDesc != NULL) 	CorpusData.DescAll3D.push_back(Desc3Di);
	}

	printf("Done with nvm\n");

	return true;
}


bool loadBundleAdjustedNVMResults(char *BAfileName, Corpus &CorpusData)
{
	const int nHDs = 30, nVGAs = 24, nPanels = 20;
	FILE *fp = fopen(BAfileName, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", BAfileName);
		return false;
	}

	char Fname[200];
	int lensType, shutterModel, width, height;
	double fx, fy, skew, u0, v0, r1, r2, r3, t1, t2, p1, p2, omega, DistCtrX, DistCtrY, rv[3], T[3];

	fscanf(fp, "%d ", &CorpusData.nCameras);
	CorpusData.camera = new CameraData[CorpusData.nCameras + 20];

	for (int ii = 0; ii < CorpusData.nCameras; ii++)
		CorpusData.camera[ii].valid = false;

	for (int ii = 0; ii < CorpusData.nCameras; ii++)
	{
		if (fscanf(fp, "%s %d %d %d %d", &Fname, &lensType, &shutterModel, &width, &height) == EOF)
			break;
		string filename = Fname;
		size_t dotpos = filename.find(".");
		int viewID, camID, panelID;
		if (dotpos < 4)
		{
			std::size_t pos = filename.find(".ppm");
			if (pos > 1000)
			{
				pos = filename.find(".png");
				if (pos > 1000)
				{
					pos = filename.find(".jpg");
					if (pos > 1000)
					{
						printf("Something wrong with the image name in the BA file!\n");
						abort();
					}
				}
			}
			filename.erase(pos, 4);
			const char * str = filename.c_str();
			viewID = atoi(str);
		}
		else
		{
			std::size_t pos1 = filename.find("_");
			string PanelName; PanelName = filename.substr(0, 2);
			const char * str = PanelName.c_str();
			panelID = atoi(str);

			string CamName; CamName = filename.substr(pos1 + 1, 2);
			str = CamName.c_str();
			camID = atoi(str);

			viewID = panelID == 0 ? camID : nHDs + nVGAs*(panelID - 1) + camID - 1;
		}

		CorpusData.camera[viewID].valid = true;
		CorpusData.camera[viewID].LensModel = lensType, CorpusData.camera[viewID].ShutterModel = shutterModel;
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
		if (CorpusData.camera[viewID].ShutterModel == 1)
			fscanf(fp, "%lf %lf %lf %lf %lf %lf ", &CorpusData.camera[viewID].wt[0], &CorpusData.camera[viewID].wt[1], &CorpusData.camera[viewID].wt[2], &CorpusData.camera[viewID].wt[3], &CorpusData.camera[viewID].wt[4], &CorpusData.camera[viewID].wt[5]);
		else
			for (int jj = 0; jj < 6; jj++)
				CorpusData.camera[viewID].wt[jj] = 0.0;

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
		GetRCGL(CorpusData.camera[viewID]);
		AssembleP(CorpusData.camera[viewID]);
	}
	fclose(fp);

	return true;
}
bool saveBundleAdjustedNVMResults(char *BAfileName, Corpus &CorpusData)
{
	double fx, fy, skew, u0, v0, r1, r2, r3, t1, t2, p1, p2, omega, DistCtrX, DistCtrY, rv[3], T[3];

	FILE *fp = fopen(BAfileName, "w+");
	fprintf(fp, "%d \n", CorpusData.nCameras);

	for (int viewID = 0; viewID < CorpusData.nCameras; viewID++)
	{
		fprintf(fp, "%d.png %d %d %d %d ", viewID, CorpusData.camera[viewID].LensModel, CorpusData.camera[viewID].ShutterModel, CorpusData.camera[viewID].width, CorpusData.camera[viewID].height);

		fx = CorpusData.camera[viewID].intrinsic[0], fy = CorpusData.camera[viewID].intrinsic[1],
			skew = CorpusData.camera[viewID].intrinsic[2],
			u0 = CorpusData.camera[viewID].intrinsic[3], v0 = CorpusData.camera[viewID].intrinsic[4];

		//Scale data
		for (int jj = 0; jj < 3; jj++)
			rv[jj] = CorpusData.camera[viewID].rt[jj], T[jj] = CorpusData.camera[viewID].rt[jj + 3];

		if (CorpusData.camera[viewID].LensModel == RADIAL_TANGENTIAL_PRISM)
		{
			r1 = CorpusData.camera[viewID].distortion[0], r2 = CorpusData.camera[viewID].distortion[1], r3 = CorpusData.camera[viewID].distortion[2],
				t1 = CorpusData.camera[viewID].distortion[3], t2 = CorpusData.camera[viewID].distortion[4],
				p1 = CorpusData.camera[viewID].distortion[5], p2 = CorpusData.camera[viewID].distortion[6];
			fprintf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %.16f %.16f %.16f %.16f %.16f %.16f ", fx, fy, skew, u0, v0,
				r1, r2, r3, t1, t2, p1, p2,
				rv[0], rv[1], rv[2], T[0], T[1], T[2]);
			if (CorpusData.camera[viewID].ShutterModel == 1)
				fprintf(fp, "%.16f %.16f  %.16f %.16f %.16f %.16f \n", CorpusData.camera[viewID].wt[0], CorpusData.camera[viewID].wt[1], CorpusData.camera[viewID].wt[2],
				CorpusData.camera[viewID].wt[3], CorpusData.camera[viewID].wt[4], CorpusData.camera[viewID].wt[5]);
			else
				fprintf(fp, "\n");
		}
		else
		{
			omega = CorpusData.camera[viewID].distortion[0], DistCtrX = CorpusData.camera[viewID].distortion[1], DistCtrY = CorpusData.camera[viewID].distortion[2];
			fprintf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %.16f %.16f %.16f %.16f %.16f %.16f ", fx, fy, skew, u0, v0,
				omega, DistCtrX, DistCtrY,
				rv[0], rv[1], rv[2], T[0], T[1], T[2]);
			if (CorpusData.camera[viewID].ShutterModel == 1)
				fprintf(fp, "%.16f %.16f  %.16f %.16f %.16f %.16f \n", CorpusData.camera[viewID].wt[0], CorpusData.camera[viewID].wt[1], CorpusData.camera[viewID].wt[2],
				CorpusData.camera[viewID].wt[3], CorpusData.camera[viewID].wt[4], CorpusData.camera[viewID].wt[5]);
			else
				fprintf(fp, "\n");
		}
	}
	return true;
}
bool ReSaveBundleAdjustedNVMResults(char *BAfileName, double ScaleFactor)
{
	FILE *fp = fopen(BAfileName, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", BAfileName);
		return false;
	}

	char Fname[200];
	int lensType, shutterModel, width, height;
	double fx, fy, skew, u0, v0, r1, r2, r3, t1, t2, p1, p2, omega, DistCtrX, DistCtrY, rv[3], T[3];

	Corpus CorpusData;
	fscanf(fp, "%d ", &CorpusData.nCameras);
	CorpusData.camera = new CameraData[CorpusData.nCameras];

	for (int ii = 0; ii < CorpusData.nCameras; ii++)
	{
		if (fscanf(fp, "%s %d %d %d %d", &Fname, &lensType, &shutterModel, &width, &height) == EOF)
			break;
		string filename = Fname;
		std::size_t pos = filename.find(".ppm");
		if (pos > 1000)
		{
			pos = filename.find(".png");
			if (pos > 1000)
			{
				pos = filename.find(".jpg");
				if (pos > 100)
				{
					printf("Something wrong with the image name in the BA file!\n");
					abort();
				}
			}
		}
		filename.erase(pos, 4);
		const char * str = filename.c_str();
		int viewID = atoi(str);

		CorpusData.camera[viewID].LensModel = lensType, CorpusData.camera[viewID].ShutterModel = shutterModel;
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
	fclose(fp);

	fp = fopen(BAfileName, "w+");
	fprintf(fp, "%d \n", CorpusData.nCameras);
	for (int viewID = 0; viewID < CorpusData.nCameras; viewID++)
	{
		fprintf(fp, "%d.png %d %d %d %d ", viewID, CorpusData.camera[viewID].LensModel, CorpusData.camera[viewID].ShutterModel, CorpusData.camera[viewID].width, CorpusData.camera[viewID].height);

		fx = CorpusData.camera[viewID].intrinsic[0], fy = CorpusData.camera[viewID].intrinsic[1],
			skew = CorpusData.camera[viewID].intrinsic[2],
			u0 = CorpusData.camera[viewID].intrinsic[3], v0 = CorpusData.camera[viewID].intrinsic[4];

		//Scale data
		for (int jj = 0; jj < 3; jj++)
			rv[jj] = CorpusData.camera[viewID].rt[jj], T[jj] = CorpusData.camera[viewID].rt[jj + 3] * ScaleFactor;

		if (CorpusData.camera[viewID].LensModel == RADIAL_TANGENTIAL_PRISM)
		{
			r1 = CorpusData.camera[viewID].distortion[0], r2 = CorpusData.camera[viewID].distortion[1], r3 = CorpusData.camera[viewID].distortion[2],
				t1 = CorpusData.camera[viewID].distortion[3], t2 = CorpusData.camera[viewID].distortion[4],
				p1 = CorpusData.camera[viewID].distortion[5], p2 = CorpusData.camera[viewID].distortion[6];
			fprintf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %.16f %.16f %.16f %.16f %.16f %.16f ", fx, fy, skew, u0, v0,
				r1, r2, r3, t1, t2, p1, p2,
				rv[0], rv[1], rv[2], T[0], T[1], T[2]);
			if (CorpusData.camera[viewID].ShutterModel == 1)
				fprintf(fp, "%.16f %.16f  %.16f %.16f %.16f %.16f \n", CorpusData.camera[viewID].wt[0], CorpusData.camera[viewID].wt[1], CorpusData.camera[viewID].wt[2],
				CorpusData.camera[viewID].wt[3], CorpusData.camera[viewID].wt[4], CorpusData.camera[viewID].wt[5]);
			else
				fprintf(fp, "\n");
		}
		else
		{
			omega = CorpusData.camera[viewID].distortion[0], DistCtrX = CorpusData.camera[viewID].distortion[1], DistCtrY = CorpusData.camera[viewID].distortion[2];
			fprintf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %.16f %.16f %.16f %.16f %.16f %.16f ", fx, fy, skew, u0, v0,
				omega, DistCtrX, DistCtrY,
				rv[0], rv[1], rv[2], T[0], T[1], T[2]);
			if (CorpusData.camera[viewID].ShutterModel == 1)
				fprintf(fp, "%.16f %.16f  %.16f %.16f %.16f %.16f \n", CorpusData.camera[viewID].wt[0], CorpusData.camera[viewID].wt[1], CorpusData.camera[viewID].wt[2],
				CorpusData.camera[viewID].wt[3], CorpusData.camera[viewID].wt[4], CorpusData.camera[viewID].wt[5]);
			else
				fprintf(fp, "\n");
		}
	}
	fclose(fp);

	return true;
}
bool ReSaveBundleAdjustedNVMResults(char *BAfileName, Corpus &CorpusData, double ScaleFactor)
{
	double fx, fy, skew, u0, v0, r1, r2, r3, t1, t2, p1, p2, omega, DistCtrX, DistCtrY, rv[3], T[3];

	FILE *fp = fopen(BAfileName, "w+");
	fprintf(fp, "%d \n", CorpusData.nCameras);

	for (int viewID = 0; viewID < CorpusData.nCameras; viewID++)
	{
		if (CorpusData.camera[viewID].valid)
		{
			fprintf(fp, "%d.png %d %d %d %d ", viewID, CorpusData.camera[viewID].LensModel, CorpusData.camera[viewID].ShutterModel, CorpusData.camera[viewID].width, CorpusData.camera[viewID].height);

			fx = CorpusData.camera[viewID].intrinsic[0], fy = CorpusData.camera[viewID].intrinsic[1],
				skew = CorpusData.camera[viewID].intrinsic[2],
				u0 = CorpusData.camera[viewID].intrinsic[3], v0 = CorpusData.camera[viewID].intrinsic[4];

			//Scale data
			for (int jj = 0; jj < 3; jj++)
				rv[jj] = CorpusData.camera[viewID].rt[jj], T[jj] = CorpusData.camera[viewID].rt[jj + 3] * ScaleFactor;

			if (CorpusData.camera[viewID].LensModel == RADIAL_TANGENTIAL_PRISM)
			{
				r1 = CorpusData.camera[viewID].distortion[0], r2 = CorpusData.camera[viewID].distortion[1], r3 = CorpusData.camera[viewID].distortion[2],
					t1 = CorpusData.camera[viewID].distortion[3], t2 = CorpusData.camera[viewID].distortion[4],
					p1 = CorpusData.camera[viewID].distortion[5], p2 = CorpusData.camera[viewID].distortion[6];
				fprintf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %.16f %.16f %.16f %.16f %.16f %.16f ", fx, fy, skew, u0, v0,
					r1, r2, r3, t1, t2, p1, p2,
					rv[0], rv[1], rv[2], T[0], T[1], T[2]);
				if (CorpusData.camera[viewID].ShutterModel == 1)
					fprintf(fp, "%.16f %.16f  %.16f %.16f %.16f %.16f \n", CorpusData.camera[viewID].wt[0], CorpusData.camera[viewID].wt[1], CorpusData.camera[viewID].wt[2],
					CorpusData.camera[viewID].wt[3], CorpusData.camera[viewID].wt[4], CorpusData.camera[viewID].wt[5]);
				else
					fprintf(fp, "\n");
			}
			else
			{
				omega = CorpusData.camera[viewID].distortion[0], DistCtrX = CorpusData.camera[viewID].distortion[1], DistCtrY = CorpusData.camera[viewID].distortion[2];
				fprintf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %.16f %.16f %.16f %.16f %.16f %.16f ", fx, fy, skew, u0, v0,
					omega, DistCtrX, DistCtrY,
					rv[0], rv[1], rv[2], T[0], T[1], T[2]);
				if (CorpusData.camera[viewID].ShutterModel == 1)
					fprintf(fp, "%.16f %.16f  %.16f %.16f %.16f %.16f \n", CorpusData.camera[viewID].wt[0], CorpusData.camera[viewID].wt[1], CorpusData.camera[viewID].wt[2],
					CorpusData.camera[viewID].wt[3], CorpusData.camera[viewID].wt[4], CorpusData.camera[viewID].wt[5]);
				else
					fprintf(fp, "\n");
			}
		}
	}
	fclose(fp);
	return true;
}

int SaveCorpusInfo(char *Path, Corpus &CorpusData, bool notbinary, bool saveDescriptor)
{
	int ii, jj, kk;
	char Fname[200];
	sprintf(Fname, "%s/Corpus_3D.txt", Path);	FILE *fp = fopen(Fname, "w+");
	CorpusData.n3dPoints = CorpusData.xyz.size();
	fprintf(fp, "%d %d ", CorpusData.nCameras, CorpusData.n3dPoints);

	//xyz rgb viewid3D pointid3D 3dId2D cumpoint
	if (CorpusData.rgb.size() == 0)
	{
		fprintf(fp, "0\n");
		for (jj = 0; jj < CorpusData.xyz.size(); jj++)
			fprintf(fp, "%lf %lf %lf \n", CorpusData.xyz[jj].x, CorpusData.xyz[jj].y, CorpusData.xyz[jj].z);
	}
	else
	{
		fprintf(fp, "1\n");
		for (jj = 0; jj < CorpusData.xyz.size(); jj++)
			fprintf(fp, "%lf %lf %lf %d %d %d\n", CorpusData.xyz[jj].x, CorpusData.xyz[jj].y, CorpusData.xyz[jj].z, CorpusData.rgb[jj].x, CorpusData.rgb[jj].y, CorpusData.rgb[jj].z);
	}
	fclose(fp);

	sprintf(Fname, "%s/Corpus_viewIdAll3D.txt", Path); fp = fopen(Fname, "w+");
	for (jj = 0; jj < CorpusData.n3dPoints; jj++)
	{
		int nviews = CorpusData.viewIdAll3D[jj].size();
		fprintf(fp, "%d ", nviews);
		for (ii = 0; ii < nviews; ii++)
			fprintf(fp, "%d ", CorpusData.viewIdAll3D[jj][ii]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	sprintf(Fname, "%s/Corpus_pointIdAll3D.txt", Path); fp = fopen(Fname, "w+");
	for (jj = 0; jj < CorpusData.n3dPoints; jj++)
	{
		int npts = CorpusData.pointIdAll3D[jj].size();
		fprintf(fp, "%d ", npts);
		for (ii = 0; ii < npts; ii++)
			fprintf(fp, "%d ", CorpusData.pointIdAll3D[jj][ii]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	sprintf(Fname, "%s/Corpus_uvAll3D.txt", Path); fp = fopen(Fname, "w+");
	for (jj = 0; jj < CorpusData.n3dPoints; jj++)
	{
		int npts = CorpusData.uvAll3D[jj].size();
		fprintf(fp, "%d ", npts);
		for (ii = 0; ii < npts; ii++)
			fprintf(fp, "%8f %8f %.2f ", CorpusData.uvAll3D[jj][ii].x, CorpusData.uvAll3D[jj][ii].y, CorpusData.scaleAll3D[jj][ii]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	sprintf(Fname, "%s/Corpus_threeDIdAllViews.txt", Path); fp = fopen(Fname, "w+");
	if (CorpusData.threeDIdAllViews.size() != 0)
	{
		for (jj = 0; jj < CorpusData.nCameras; jj++)
		{
			int n3D = CorpusData.threeDIdAllViews[jj].size();
			fprintf(fp, "%d\n", n3D);
			for (ii = 0; ii < n3D; ii++)
				fprintf(fp, "%d ", CorpusData.threeDIdAllViews[jj][ii]);
			fprintf(fp, "\n");
		}
	}
	else
	{
		for (jj = 0; jj < CorpusData.nCameras; jj++)
		{
			int n3D = CorpusData.threeDIdAllViews2[jj].size();
			fprintf(fp, "%d\n", n3D);
			for (ii = 0; ii < n3D; ii++)
				fprintf(fp, "%d ", CorpusData.threeDIdAllViews2[jj][ii]);
			fprintf(fp, "\n");
		}
	}
	fclose(fp);

	sprintf(Fname, "%s/Corpus_cum.txt", Path); fp = fopen(Fname, "w+");
	for (int ii = 0; ii < CorpusData.IDCumView.size(); ii++)
		fprintf(fp, "%d ", CorpusData.IDCumView[ii]);
	fclose(fp);

	for (ii = 0; ii < CorpusData.nCameras; ii++)
	{
		sprintf(Fname, "%s/CorpusK_%d.txt", Path, ii); FILE *fp = fopen(Fname, "w+");
		if (CorpusData.uvAllViews.size() != 0)
		{
			int npts = CorpusData.uvAllViews[ii].size();
			for (int jj = 0; jj < npts; jj++)
				fprintf(fp, "%.4f %.4f %.2f\n", CorpusData.uvAllViews[ii][jj].x, CorpusData.uvAllViews[ii][jj].y, CorpusData.scaleAllViews[ii][jj]);
		}
		else
		{
			int npts = CorpusData.uvAllViews2[ii].size();
			for (int jj = 0; jj < npts; jj++)
				fprintf(fp, "%.4f %.4f %.2f\n", CorpusData.uvAllViews2[ii][jj].x, CorpusData.uvAllViews2[ii][jj].y, CorpusData.scaleAllViews2[ii][jj]);
		}
		fclose(fp);
	}

	sprintf(Fname, "%s/Corpus_Intrinsics.txt", Path); fp = fopen(Fname, "w+");
	for (int viewID = 0; viewID < CorpusData.nCameras; viewID++)
	{
		fprintf(fp, "%d %d %d ", CorpusData.camera[viewID].LensModel, CorpusData.camera[viewID].width, CorpusData.camera[viewID].height);
		for (int ii = 0; ii < 5; ii++)
			fprintf(fp, "%.6e ", CorpusData.camera[viewID].intrinsic[ii]);

		if (CorpusData.camera[viewID].LensModel == RADIAL_TANGENTIAL_PRISM)
			for (int ii = 0; ii < 7; ii++)
				fprintf(fp, "%.6e ", CorpusData.camera[viewID].distortion[ii]);
		else
		{
			for (int ii = 0; ii < 3; ii++)
				fprintf(fp, "%.6e ", CorpusData.camera[viewID].distortion[ii]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	sprintf(Fname, "%s/Corpus_Extrinsics.txt", Path); fp = fopen(Fname, "w+");
	for (int viewID = 0; viewID < CorpusData.nCameras; viewID++)
	{
		for (int ii = 0; ii < 6; ii++)
			fprintf(fp, "%.16e ", CorpusData.camera[viewID].rt[ii]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	sprintf(Fname, "%s/Corpus_extendedExtrinsics.txt", Path); fp = fopen(Fname, "w+");
	for (int viewID = 0; viewID < CorpusData.nCameras; viewID++)
	{
		for (int ii = 0; ii < 6; ii++)
			fprintf(fp, "%.16e ", CorpusData.camera[viewID].wt[ii]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	if (!saveDescriptor)
		return 0;

	if (notbinary)
	{
		for (kk = 0; kk < CorpusData.nCameras; kk++)
		{
			if (CorpusData.SiftDesc.rows != 0)
			{
				sprintf(Fname, "%s/CorpusD_%d.txt", Path, kk);	fp = fopen(Fname, "w+");
				int npts = CorpusData.threeDIdAllViews[kk].size(), curPid = CorpusData.IDCumView[kk];
				fprintf(fp, "%d\n", npts);
				for (jj = 0; jj < npts; jj++)
				{
					fprintf(fp, "%d ", CorpusData.threeDIdAllViews[kk][jj]);
					for (ii = 0; ii < SIFTBINS; ii++)
						fprintf(fp, "%.5f ", CorpusData.SiftDesc.at<float>(curPid + jj, ii));
					fprintf(fp, "\n");
				}
				fclose(fp);
			}
			else
			{
				sprintf(Fname, "%s/CorpusD_%d.txt", Path, kk);	fp = fopen(Fname, "w+");
				int npts = CorpusData.threeDIdAllViews2[kk].size();
				fprintf(fp, "%d\n", npts);
				for (jj = 0; jj < npts; jj++)
				{
					fprintf(fp, "%d ", CorpusData.threeDIdAllViews2[kk][jj]);
					for (ii = 0; ii < SIFTBINS; ii++)
						fprintf(fp, "%.5f ", CorpusData.DescAllViews2[kk][jj].desc[ii]);
					fprintf(fp, "\n");
				}
				fclose(fp);
			}
		}
	}
	else
	{
		for (kk = 0; kk < CorpusData.nCameras; kk++)
		{
			if (CorpusData.threeDIdAllViews.size() != 0)
			{
				sprintf(Fname, "%s/CorpusD_%d.txt", Path, kk); ofstream fout; fout.open(Fname, ios::binary);
				int npts = CorpusData.threeDIdAllViews[kk].size(), curPid = CorpusData.IDCumView[kk];
				fout.write(reinterpret_cast<char *>(&npts), sizeof(int));
				for (jj = 0; jj < npts; jj++)
				{
					fout.write(reinterpret_cast<char *>(&CorpusData.threeDIdAllViews[kk][jj]), sizeof(int));
					for (ii = 0; ii < SIFTBINS; ii++)
						fout.write(reinterpret_cast<char *>(&CorpusData.SiftDesc.at<float>(curPid + jj, ii)), sizeof(float));
				}
				fout.close();
			}
			else
			{
				sprintf(Fname, "%s/CorpusD_%d.txt", Path, kk); ofstream fout; fout.open(Fname, ios::binary);
				int npts = CorpusData.threeDIdAllViews2[kk].size();
				fout.write(reinterpret_cast<char *>(&npts), sizeof(int));
				for (jj = 0; jj < npts; jj++)
				{
					fout.write(reinterpret_cast<char *>(&CorpusData.threeDIdAllViews2[kk][jj]), sizeof(int));
					for (ii = 0; ii < SIFTBINS; ii++)
						fout.write(reinterpret_cast<char *>(&CorpusData.DescAllViews2[kk][jj].desc[ii]), sizeof(float));
				}
				fout.close();
			}
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
	CorpusData.nCameras = nCameras;
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
	Point2d uv;	vector<Point2d> uvVector; uvVector.reserve(50);
	double scale = 1.0;  vector<double> scaleVector; scaleVector.reserve(2000);
	for (jj = 0; jj < CorpusData.n3dPoints; jj++)
	{
		uvVector.clear(), scaleVector.clear();
		fscanf(fp, "%d ", &npts);
		for (ii = 0; ii < npts; ii++)
		{
			fscanf(fp, "%lf %lf %lf", &uv.x, &uv.y, &scale);
			//fscanf(fp, "%lf %lf", &uv.x, &uv.y);
			uvVector.push_back(uv), scaleVector.push_back(scale);
		}
		CorpusData.uvAll3D.push_back(uvVector);
		CorpusData.scaleAll3D.push_back(scaleVector);
	}
	fclose(fp);

	sprintf(Fname, "%s/Corpus_threeDIdAllViews.txt", Path); fp = fopen(Fname, "r");
	int n3D, id3D;
	vector<int>threeDid;
	for (jj = 0; jj < CorpusData.nCameras; jj++)
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
	for (ii = 0; ii < CorpusData.nCameras; ii++)
	{
		uvVector.clear(), scaleVector.clear();
		sprintf(Fname, "%s/CorpusK_%d.txt", Path, ii);
		FILE *fp = fopen(Fname, "r");
		while (fscanf(fp, "%lf %lf %lf", &uv.x, &uv.y, &scale) != EOF)
			uvVector.push_back(uv), scaleVector.push_back(scale);
		//while (fscanf(fp, "%lf %lf ", &uv.x, &uv.y) != EOF)
		//uvVector.push_back(uv), scaleVector.push_back(1.0);
		fclose(fp);
		CorpusData.uvAllViews.push_back(uvVector);
		CorpusData.scaleAllViews.push_back(scaleVector);
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
		fscanf(fp, "%d %d %d ", &CorpusData.camera[viewID].LensModel, &CorpusData.camera[viewID].width, &CorpusData.camera[viewID].height);
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

	sprintf(Fname, "%s/Corpus_Extrinsics.txt", Path); fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return 1;
	}
	for (viewID = 0; viewID < nCameras; viewID++)
	{
		for (int ii = 0; ii < 6; ii++)
			fscanf(fp, "%lf ", &CorpusData.camera[viewID].rt[ii]);

		GetRTFromrt(CorpusData.camera[viewID]);
	}
	fclose(fp);

	sprintf(Fname, "%s/Corpus_extendedExtrinsics.txt", Path); fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		for (viewID = 0; viewID < nCameras; viewID++)
			for (int ii = 0; ii < 6; ii++)
				fscanf(fp, "%lf ", &CorpusData.camera[viewID].wt[ii]);
		fclose(fp);
	}
	else
	{
		for (viewID = 0; viewID < nCameras; viewID++)
			for (int ii = 0; ii < 6; ii++)
				CorpusData.camera[viewID].wt[ii] = 0.0;
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
	CorpusandVideoInfo.nVideos = nVideoViews;
	CorpusandVideoInfo.VideoInfo = new CameraData[nVideoViews*MaxnFrames];
	int id;
	double R[9], C[3], t1, t2, t3, t4, t5, t6, t7;
	for (int viewID = 0; viewID < nVideoViews; viewID++)
	{
		sprintf(Fname, "%s/CamPose_%d.txt", Path, viewID);
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
				CorpusandVideoInfo.VideoInfo[id + viewID*MaxnFrames].R[jj] = R[jj];

			//T = -R*Center;
			mat_mul(R, C, CorpusandVideoInfo.VideoInfo[id + viewID*MaxnFrames].T, 3, 3, 1);
			for (int jj = 0; jj < 3; jj++)
				CorpusandVideoInfo.VideoInfo[id + viewID*MaxnFrames].T[jj] = -CorpusandVideoInfo.VideoInfo[id + viewID*MaxnFrames].T[jj];

			for (int jj = 0; jj < 5; jj++)
				CorpusandVideoInfo.VideoInfo[id + viewID*MaxnFrames].intrinsic[jj] = IntrinsicInfo[viewID].intrinsic[jj];
			for (int jj = 0; jj < 7; jj++)
				CorpusandVideoInfo.VideoInfo[id + viewID*MaxnFrames].distortion[jj] = IntrinsicInfo[viewID].distortion[jj];

			GetKFromIntrinsic(CorpusandVideoInfo.VideoInfo[id + viewID*MaxnFrames]);
			GetrtFromRT(CorpusandVideoInfo.VideoInfo[id + viewID*MaxnFrames].rt,
				CorpusandVideoInfo.VideoInfo[id + viewID*MaxnFrames].R, CorpusandVideoInfo.VideoInfo[id + viewID*MaxnFrames].T);
			AssembleP(CorpusandVideoInfo.VideoInfo[id + viewID*MaxnFrames].K, CorpusandVideoInfo.VideoInfo[id + viewID*MaxnFrames].R,
				CorpusandVideoInfo.VideoInfo[id + viewID*MaxnFrames].T, CorpusandVideoInfo.VideoInfo[id + viewID*MaxnFrames].P);
		}
	}
	//READ FROM VIDEO POSE: END

	return 0;
}
int ReadVideoData(char *Path, VideoData &AllVideoInfo, int nVideoViews, int startTime, int stopTime)
{
	char Fname[200];
	int videoID, frameID, LensType, width, height;
	int nframes = max(MaxnFrames, stopTime);

	AllVideoInfo.nVideos = nVideoViews;
	AllVideoInfo.VideoInfo = new CameraData[nVideoViews*nframes];

	for (int ii = 0; ii < nVideoViews*nframes; ii++)
		AllVideoInfo.VideoInfo[ii].valid = false;

	//READ INTRINSIC: START
	int count = 0;
	for (int viewID = 0; viewID < nVideoViews; viewID++)
	{
		videoID = nframes*viewID;
		sprintf(Fname, "%s/Intrinsic_%d.txt", Path, viewID); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			cout << "Cannot load " << Fname << endl;
			count++;
			continue;
		}
		double fx, fy, skew, u0, v0, r0, r1, r2, t0, t1, p0, p1, omega, DistCtrX, DistCtrY;
		while (fscanf(fp, "%d %d %d %d %lf %lf %lf %lf %lf ", &frameID, &LensType, &width, &height, &fx, &fy, &skew, &u0, &v0) != EOF)
		{
			AllVideoInfo.VideoInfo[frameID + videoID].valid = true;
			AllVideoInfo.VideoInfo[frameID + videoID].K[0] = fx, AllVideoInfo.VideoInfo[frameID + videoID].K[1] = skew, AllVideoInfo.VideoInfo[frameID + videoID].K[2] = u0,
				AllVideoInfo.VideoInfo[frameID + videoID].K[3] = 0.0, AllVideoInfo.VideoInfo[frameID + videoID].K[4] = fy, AllVideoInfo.VideoInfo[frameID + videoID].K[5] = v0,
				AllVideoInfo.VideoInfo[frameID + videoID].K[6] = 0.0, AllVideoInfo.VideoInfo[frameID + videoID].K[7] = 0.0, AllVideoInfo.VideoInfo[frameID + videoID].K[8] = 1.0;

			mat_invert(AllVideoInfo.VideoInfo[frameID + videoID].K, AllVideoInfo.VideoInfo[frameID + videoID].invK, 3);
			GetIntrinsicFromK(AllVideoInfo.VideoInfo[frameID + videoID]);
			mat_invert(AllVideoInfo.VideoInfo[frameID + videoID].K, AllVideoInfo.VideoInfo[frameID + videoID].invK);

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
			AllVideoInfo.VideoInfo[frameID + videoID].width = width, AllVideoInfo.VideoInfo[frameID + videoID].height = height;
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
		videoID = nframes*viewID;
		sprintf(Fname, "%s/CamPose_%d.txt", Path, viewID);
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

			for (int jj = 0; jj < 3; jj++)
				AllVideoInfo.VideoInfo[frameID + videoID].camCenter[jj] = C[jj];

			//T = -R*Center;
			mat_mul(R, C, AllVideoInfo.VideoInfo[frameID + videoID].T, 3, 3, 1);
			for (int jj = 0; jj < 3; jj++)
				AllVideoInfo.VideoInfo[frameID + videoID].T[jj] = -AllVideoInfo.VideoInfo[frameID + videoID].T[jj];

			mat_invert(AllVideoInfo.VideoInfo[frameID + videoID].R, AllVideoInfo.VideoInfo[frameID + videoID].invR);
			GetrtFromRT(AllVideoInfo.VideoInfo[frameID + videoID].rt,
				AllVideoInfo.VideoInfo[frameID + videoID].R, AllVideoInfo.VideoInfo[frameID + videoID].T);
			AssembleP(AllVideoInfo.VideoInfo[frameID + videoID].K, AllVideoInfo.VideoInfo[frameID + videoID].R,
				AllVideoInfo.VideoInfo[frameID + videoID].T, AllVideoInfo.VideoInfo[frameID + videoID].P);
		}
		fclose(fp);
	}
	if (count == nVideoViews)
		return 1;
	//READ FROM VIDEO POSE: END

	return 0;
}
int ReadVideoDataI(char *Path, VideoData &VideoInfo, int viewID, int startTime, int stopTime)
{
	char Fname[200];
	int frameID, LensType, width, height;
	int nframes = max(MaxnFrames, stopTime);

	VideoInfo.VideoInfo = new CameraData[nframes];
	for (int ii = 0; ii < nframes; ii++)
		VideoInfo.VideoInfo[ii].valid = false;

	//READ INTRINSIC: START
	sprintf(Fname, "%s/Intrinsic_%d.txt", Path, viewID); FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		cout << "Cannot load " << Fname << endl;
		return 1;
	}
	double fx, fy, skew, u0, v0, r0, r1, r2, t0, t1, p0, p1, omega, DistCtrX, DistCtrY;
	while (fscanf(fp, "%d %d %d %d %lf %lf %lf %lf %lf ", &frameID, &LensType, &width, &height, &fx, &fy, &skew, &u0, &v0) != EOF)
	{
		if (frameID >= startTime && frameID <= stopTime)
		{
			VideoInfo.VideoInfo[frameID].K[0] = fx, VideoInfo.VideoInfo[frameID].K[1] = skew, VideoInfo.VideoInfo[frameID].K[2] = u0,
				VideoInfo.VideoInfo[frameID].K[3] = 0.0, VideoInfo.VideoInfo[frameID].K[4] = fy, VideoInfo.VideoInfo[frameID].K[5] = v0,
				VideoInfo.VideoInfo[frameID].K[6] = 0.0, VideoInfo.VideoInfo[frameID].K[7] = 0.0, VideoInfo.VideoInfo[frameID].K[8] = 1.0;

			mat_invert(VideoInfo.VideoInfo[frameID].K, VideoInfo.VideoInfo[frameID].invK, 3);
			GetIntrinsicFromK(VideoInfo.VideoInfo[frameID]);
			mat_invert(VideoInfo.VideoInfo[frameID].K, VideoInfo.VideoInfo[frameID].invK);

			VideoInfo.VideoInfo[frameID].LensModel = LensType, VideoInfo.VideoInfo[frameID].threshold = 3.0, VideoInfo.VideoInfo[frameID].ninlierThresh = 40;
			VideoInfo.VideoInfo[frameID].valid = true;
		}

		if (LensType == RADIAL_TANGENTIAL_PRISM)
		{
			fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf ", &r0, &r1, &r2, &t0, &t1, &p0, &p1);
			if (frameID >= startTime && frameID <= stopTime)
			{
				VideoInfo.VideoInfo[frameID].distortion[0] = r0, VideoInfo.VideoInfo[frameID].distortion[1] = r1, VideoInfo.VideoInfo[frameID].distortion[2] = r2;
				VideoInfo.VideoInfo[frameID].distortion[3] = t0, VideoInfo.VideoInfo[frameID].distortion[4] = t1;
				VideoInfo.VideoInfo[frameID].distortion[5] = p0, VideoInfo.VideoInfo[frameID].distortion[6] = p1;
			}
		}
		else
		{
			fscanf(fp, "%lf %lf %lf ", &omega, &DistCtrX, &DistCtrY);
			if (frameID >= startTime && frameID <= stopTime)
				VideoInfo.VideoInfo[frameID].distortion[0] = omega, VideoInfo.VideoInfo[frameID].distortion[1] = DistCtrX, VideoInfo.VideoInfo[frameID].distortion[2] = DistCtrY;
		}
		if (frameID >= startTime && frameID <= stopTime)
			VideoInfo.VideoInfo[frameID].width = width, VideoInfo.VideoInfo[frameID].height = height;
	}
	fclose(fp);
	//END

	//READ POSE FROM VIDEO POSE: START
	sprintf(Fname, "%s/CamPose_%d.txt", Path, viewID); fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return 1;
	}
	double R[9], C[3], t2, t3, t4, t5, t6, t7;
	while (fscanf(fp, "%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf ",
		&frameID, &R[0], &R[1], &R[2], &t1, &R[3], &R[4], &R[5], &t2, &R[6], &R[7], &R[8], &t3, &t4, &t5, &t6, &t7, &C[0], &C[1], &C[2]) != EOF)
	{
		if (frameID >= startTime && frameID <= stopTime)
		{
			if (abs(C[0]) + abs(C[1]) + abs(C[2]) > 0.001)
				VideoInfo.VideoInfo[frameID].valid = true;
			else
				VideoInfo.VideoInfo[frameID].valid = false;

			for (int jj = 0; jj < 9; jj++)
				VideoInfo.VideoInfo[frameID].R[jj] = R[jj];

			for (int jj = 0; jj < 3; jj++)
				VideoInfo.VideoInfo[frameID].camCenter[jj] = C[jj];

			//T = -R*Center;
			mat_mul(R, C, VideoInfo.VideoInfo[frameID].T, 3, 3, 1);
			for (int jj = 0; jj < 3; jj++)
				VideoInfo.VideoInfo[frameID].T[jj] = -VideoInfo.VideoInfo[frameID].T[jj];

			Rotation2Quaternion(VideoInfo.VideoInfo[frameID].R, VideoInfo.VideoInfo[frameID].Quat);

			mat_invert(VideoInfo.VideoInfo[frameID].R, VideoInfo.VideoInfo[frameID].invR);
			GetrtFromRT(VideoInfo.VideoInfo[frameID].rt,
				VideoInfo.VideoInfo[frameID].R, VideoInfo.VideoInfo[frameID].T);
			GetRCGL(VideoInfo.VideoInfo[frameID]);
			AssembleP(VideoInfo.VideoInfo[frameID].K, VideoInfo.VideoInfo[frameID].R,
				VideoInfo.VideoInfo[frameID].T, VideoInfo.VideoInfo[frameID].P);
		}
	}
	fclose(fp);
	//READ FROM VIDEO POSE: END

	return 0;
}
int WriteVideoDataI(char *Path, VideoData &VideoInfo, int viewID, int startTime, int stopTime)
{
	//WRITE  INTRINSIC
	char Fname[200]; sprintf(Fname, "%s/_Intrinsic_%d.txt", Path, viewID); FILE *fp = fopen(Fname, "w+");
	for (int fid = startTime; fid <= stopTime; fid++)
	{
		if (!VideoInfo.VideoInfo[fid].valid)
			continue;
		fprintf(fp, "%d %d %d %d %.8f %.8f %.8f %.8f %.8f  ", fid, VideoInfo.VideoInfo[fid].LensModel, VideoInfo.VideoInfo[fid].width, VideoInfo.VideoInfo[fid].height,
			VideoInfo.VideoInfo[fid].K[0], VideoInfo.VideoInfo[fid].K[4], VideoInfo.VideoInfo[fid].K[1], VideoInfo.VideoInfo[fid].K[2], VideoInfo.VideoInfo[fid].K[5]);

		if (VideoInfo.VideoInfo[fid].LensModel == RADIAL_TANGENTIAL_PRISM)
			fprintf(fp, "%.6f %.6f %.6f %.6f %.6f %.6f %.6f ", VideoInfo.VideoInfo[fid].distortion[0], VideoInfo.VideoInfo[fid].distortion[1], VideoInfo.VideoInfo[fid].distortion[2],
			VideoInfo.VideoInfo[fid].distortion[3], VideoInfo.VideoInfo[fid].distortion[4], VideoInfo.VideoInfo[fid].distortion[5], VideoInfo.VideoInfo[fid].distortion[6]);
		else
			fprintf(fp, "%.8f %.8f %.8f \n", VideoInfo.VideoInfo[fid].distortion[0], VideoInfo.VideoInfo[fid].distortion[1], VideoInfo.VideoInfo[fid].distortion[2]);
	}
	fclose(fp);

	//WRITE VIDEO POSE
	sprintf(Fname, "%s/_CamPose_%d.txt", Path, viewID); fp = fopen(Fname, "w+");
	for (int fid = startTime; fid <= stopTime; fid++)
	{
		if (!VideoInfo.VideoInfo[fid].valid)
			continue;

		fprintf(fp, "%d %.16f %.16f %.16f 0.0 %.16f %.16f %.16f 0.0 %.16f %.16f %.16f 0.0 0.0 0.0 0.0 1.0 %.16f %.16f %.16f ", fid,
			VideoInfo.VideoInfo[fid].R[0], VideoInfo.VideoInfo[fid].R[1], VideoInfo.VideoInfo[fid].R[2],
			VideoInfo.VideoInfo[fid].R[3], VideoInfo.VideoInfo[fid].R[4], VideoInfo.VideoInfo[fid].R[5],
			VideoInfo.VideoInfo[fid].R[6], VideoInfo.VideoInfo[fid].R[7], VideoInfo.VideoInfo[fid].R[8],
			VideoInfo.VideoInfo[fid].camCenter[0], VideoInfo.VideoInfo[fid].camCenter[1], VideoInfo.VideoInfo[fid].camCenter[2]);
	}
	fclose(fp);

	return 0;
}
void SaveCurrentPosesGL(char *path, CameraData *AllViewParas, vector<int>AvailViews, int timeID)
{
	char Fname[200];

	//Center = -iR*T 
	double iR[9], center[3];
	for (int ii = 0; ii < AvailViews.size(); ii++)
	{
		int viewID = AvailViews[ii];
		if (viewID < 0)
			continue;
		mat_invert(AllViewParas[viewID].R, iR);

		AllViewParas[viewID].Rgl[0] = AllViewParas[viewID].R[0], AllViewParas[viewID].Rgl[1] = AllViewParas[viewID].R[1], AllViewParas[viewID].Rgl[2] = AllViewParas[viewID].R[2], AllViewParas[viewID].Rgl[3] = 0.0;
		AllViewParas[viewID].Rgl[4] = AllViewParas[viewID].R[3], AllViewParas[viewID].Rgl[5] = AllViewParas[viewID].R[4], AllViewParas[viewID].Rgl[6] = AllViewParas[viewID].R[5], AllViewParas[viewID].Rgl[7] = 0.0;
		AllViewParas[viewID].Rgl[8] = AllViewParas[viewID].R[6], AllViewParas[viewID].Rgl[9] = AllViewParas[viewID].R[7], AllViewParas[viewID].Rgl[10] = AllViewParas[viewID].R[8], AllViewParas[viewID].Rgl[11] = 0.0;
		AllViewParas[viewID].Rgl[12] = 0, AllViewParas[viewID].Rgl[13] = 0, AllViewParas[viewID].Rgl[14] = 0, AllViewParas[viewID].Rgl[15] = 1.0;

		mat_mul(iR, AllViewParas[viewID].T, center, 3, 3, 1);
		AllViewParas[viewID].camCenter[0] = -center[0], AllViewParas[viewID].camCenter[1] = -center[1], AllViewParas[viewID].camCenter[2] = -center[2];
	}

	sprintf(Fname, "%s/DinfoGL_%d.txt", path, timeID);
	FILE *fp = fopen(Fname, "w+");
	for (int ii = 0; ii < AvailViews.size(); ii++)
	{
		int viewID = AvailViews[ii];
		fprintf(fp, "%d ", viewID);
		for (int jj = 0; jj < 16; jj++)
			fprintf(fp, "%.16f ", AllViewParas[viewID].Rgl[jj]);
		for (int jj = 0; jj < 3; jj++)
			fprintf(fp, "%.16f ", AllViewParas[viewID].camCenter[jj]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	return;
}
void SaveVideoCameraPosesGL(char *path, CameraData *AllViewParas, vector<int>&AvailTime, int camID, int StartTime)
{
	char Fname[200];

	//Center = -iR*T 
	double iR[9], center[3];
	for (int ii = 0; ii < AvailTime.size(); ii++)
	{
		int timeID = AvailTime[ii];
		if (timeID < 0)
			continue;
		mat_invert(AllViewParas[timeID].R, iR);

		AllViewParas[timeID].Rgl[0] = AllViewParas[timeID].R[0], AllViewParas[timeID].Rgl[1] = AllViewParas[timeID].R[1], AllViewParas[timeID].Rgl[2] = AllViewParas[timeID].R[2], AllViewParas[timeID].Rgl[3] = 0.0;
		AllViewParas[timeID].Rgl[4] = AllViewParas[timeID].R[3], AllViewParas[timeID].Rgl[5] = AllViewParas[timeID].R[4], AllViewParas[timeID].Rgl[6] = AllViewParas[timeID].R[5], AllViewParas[timeID].Rgl[7] = 0.0;
		AllViewParas[timeID].Rgl[8] = AllViewParas[timeID].R[6], AllViewParas[timeID].Rgl[9] = AllViewParas[timeID].R[7], AllViewParas[timeID].Rgl[10] = AllViewParas[timeID].R[8], AllViewParas[timeID].Rgl[11] = 0.0;
		AllViewParas[timeID].Rgl[12] = 0, AllViewParas[timeID].Rgl[13] = 0, AllViewParas[timeID].Rgl[14] = 0, AllViewParas[timeID].Rgl[15] = 1.0;

		mat_mul(iR, AllViewParas[timeID].T, center, 3, 3, 1);
		AllViewParas[timeID].camCenter[0] = -center[0], AllViewParas[timeID].camCenter[1] = -center[1], AllViewParas[timeID].camCenter[2] = -center[2];
	}

	sprintf(Fname, "%s/CamPose_%d.txt", path, camID);
	FILE *fp = fopen(Fname, "a+");
	for (int ii = 0; ii < AvailTime.size(); ii++)
	{
		int timeID = AvailTime[ii];
		fprintf(fp, "%d ", timeID + StartTime);
		for (int jj = 0; jj < 16; jj++)
			fprintf(fp, "%.16f ", AllViewParas[timeID].Rgl[jj]);
		for (int jj = 0; jj < 3; jj++)
			fprintf(fp, "%.16f ", AllViewParas[timeID].camCenter[jj]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	return;
}
int DownSampleSpatialCalib(char *Path, int nviews, int startFrame, int stopFrame, int Factor)
{
	char Fname[200];
	for (int viewID = 0; viewID < nviews; viewID++)
	{
		VideoData VideoDataInfo;
		if (ReadVideoDataI(Path, VideoDataInfo, viewID, startFrame, stopFrame) == 1)
			return 1;

		sprintf(Fname, "%s/sCamPose_%d.txt", Path, viewID); FILE *fp = fopen(Fname, "w+");
		for (int fid = startFrame; fid < stopFrame; fid++)
		{
			if ((fid - 1) % Factor == 0 && VideoDataInfo.VideoInfo[fid].valid)
			{
				fprintf(fp, "%d ", (fid - 1) / Factor + 1);
				for (int jj = 0; jj < 16; jj++)
					fprintf(fp, "%.16f ", VideoDataInfo.VideoInfo[fid].Rgl[jj]);
				for (int jj = 0; jj < 3; jj++)
					fprintf(fp, "%.16f ", VideoDataInfo.VideoInfo[fid].camCenter[jj]);
				fprintf(fp, "\n");
			}
		}
		fclose(fp);
	}

	return 0;
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
			int viewID = panelID*nVGACamsPerPanel + camID + nHDs;
			sprintf(Fname, "%s/In/Calib/%02d_%02d.txt", Path, panelID + 1, camID + 1); FILE *fp = fopen(Fname, "r");
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
			sprintf(Fname, "%s/In/Calib/%02d_%02d_ext.txt", Path, panelID + 1, camID + 1); fp = fopen(Fname, "r");
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
#ifdef _WINDOWS
void ExportCalibDatatoHanFormat(char *Path, VideoData &AllVideoInfo, int nVideoViews, int startTime, int stopTime, int chosencamera)
{
	char Fname[200];
	int offset = 0;
	int nframes = max(MaxnFrames, stopTime);

	for (unsigned int viewID = 0; viewID < nVideoViews; viewID++)
	{
		if (chosencamera != -1)
			viewID = chosencamera;
		for (int frameID = startTime; frameID <= stopTime - offset; frameID++)
		{
			int videoID = nframes*viewID;
			if (!AllVideoInfo.VideoInfo[frameID + offset + videoID].valid)
				continue;

			sprintf(Fname, "%s/Calib/Pinfo_%d%_%d.txt", Path, viewID, frameID); FILE *fp = fopen(Fname, "w+");
			if (fp == NULL)
			{
				sprintf(Fname, "%s/Calib", Path), mkdir(Fname);
				sprintf(Fname, "%s/Calib/Pinfo_%d%_%d.txt", Path, viewID, frameID);
			}

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


			sprintf(Fname, "%s/Calib/%d%_%d.txt", Path, viewID, frameID); fp = fopen(Fname, "w+");
			for (int j = 0; j < 9; j++)
				fprintf(fp, "%lf ", AllVideoInfo.VideoInfo[frameID + offset + videoID].K[j]);
			fprintf(fp, "%lf %lf\n", 0.0, 0.0);//lens distortion parameter
			fclose(fp);

			double iR[9], center[3], Quaterunion[4];
			mat_invert(AllVideoInfo.VideoInfo[frameID + offset + videoID].R, iR);
			mat_mul(iR, AllVideoInfo.VideoInfo[frameID + offset + videoID].T, center, 3, 3, 1);
			AllVideoInfo.VideoInfo[frameID + offset + videoID].camCenter[0] = -center[0], AllVideoInfo.VideoInfo[frameID + offset + videoID].camCenter[1] = -center[1], AllVideoInfo.VideoInfo[frameID + offset + videoID].camCenter[2] = -center[2];

			ceres::AngleAxisToQuaternion(AllVideoInfo.VideoInfo[frameID + offset + videoID].rt, Quaterunion);

			sprintf(Fname, "%s/Calib/%d%_%d_ext.txt", Path, viewID, frameID); fp = fopen(Fname, "w+");
			for (int j = 0; j < 4; j++)
				fprintf(fp, "%lf ", Quaterunion[j]);
			for (int j = 0; j < 3; j++)
				fprintf(fp, "%lf ", AllVideoInfo.VideoInfo[frameID + offset + videoID].camCenter[j]);
			fclose(fp);

			sprintf(Fname, "D:/CellPhone/In/%d", viewID);
			;// LensCorrectionDriver(Fname, AllVideoInfo.VideoInfo[videoID].K, AllVideoInfo.VideoInfo[videoID].distortion, AllVideoInfo.VideoInfo[videoID].LensModel, frameID, frameID, 1.0, 1.0, 5);
		}
		if (chosencamera != -1)
			break;
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
			int videoID = MaxnFrames*viewID;
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
				fprintf(fp, "%d 0.0 0.0 ", AllVis[ii][jj] - 1);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}
}
#endif

void GenerateViewAll_3D_2DInliers(char *Path, int viewID, int startID, int stopID, int n3Dcorpus)
{
	char Fname[200];
	vector<int> *All3DviewIDper3D = new vector<int>[n3Dcorpus];
	vector<Point2d> *Alluvper3D = new vector<Point2d>[n3Dcorpus];
	vector<double> *AllscalePer3D = new vector<double>[n3Dcorpus];

	int threeDid;
	double u, v, scale;
	for (int timeID = startID; timeID <= stopID; timeID++)
	{
		sprintf(Fname, "%s/%d/Inliers_3D2D_%d.txt", Path, viewID, timeID);
		FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("Cannot load %s\n", Fname);
			continue;
		}
		while (fscanf(fp, "%d %lf %lf", &threeDid, &u, &v, &scale) != EOF)
		{
			All3DviewIDper3D[threeDid].push_back(timeID);
			Alluvper3D[threeDid].push_back(Point2d(u, v));
			AllscalePer3D[threeDid].push_back(scale);
		}
		fclose(fp);
	}

	sprintf(Fname, "%s/%d/Inliers_3D2D.txt", Path, viewID);	FILE *fp = fopen(Fname, "w+");
	for (int ii = 0; ii < n3Dcorpus; ii++)
	{
		fprintf(fp, "%d\n", All3DviewIDper3D[ii].size());
		for (int jj = 0; jj < All3DviewIDper3D[ii].size(); jj++)
			fprintf(fp, "%d %f %f %f", All3DviewIDper3D[ii][jj], Alluvper3D[ii][jj].x, Alluvper3D[ii][jj].y, AllscalePer3D[ii][jj]);
		if (All3DviewIDper3D[ii].size() != 0)
			fprintf(fp, "\n");
	}
	fclose(fp);

	return;
}

//Image pyramid
int BuildImgPyr(char *ImgName, ImgPyr &Pyrad, int nOtaves, int nPerOctaves, bool color, int interpAlgo, double sigma)
{
	int width, height, nw, nh, nchannels = color ? 3 : 1;

	Mat view = imread(ImgName, nchannels == 1 ? 0 : 1);
	if (view.data == NULL)
	{
		cout << "Cannot load: " << ImgName << endl;
		return false;
	}
	width = view.cols, height = view.rows;

	unsigned char*	Img = new unsigned char[width*height*nchannels];
	int length = width*height;
	for (int kk = 0; kk < nchannels; kk++)
	{
		for (int jj = 0; jj < height; jj++)
			for (int ii = 0; ii < width; ii++)
				Img[ii + jj*width + kk*length] = view.data[nchannels*ii + jj*nchannels*width + kk];
	}
	Pyrad.factor.push_back(1.0);
	Pyrad.wh.push_back(Point2i(width, height));
	Pyrad.ImgPyrImg.push_back(Img);

	int nlayers = nOtaves*nPerOctaves, count = 0;
	double scalePerOctave = pow(2.0, 1.0 / nPerOctaves), factor;
	for (int jj = 1; jj <= nOtaves; jj++)
	{
		for (int ii = nPerOctaves - 1; ii >= 0; ii--)
		{
			factor = pow(scalePerOctave, ii) / pow(2.0, jj);
			nw = (int)(factor*width), nh = (int)(factor*height);

			unsigned char *smallImg = new uchar[nw*nh*nchannels];
			ResizeImage(Pyrad.ImgPyrImg[0], smallImg, width, height, nchannels, factor, sigma / factor, interpAlgo);

			Pyrad.factor.push_back(factor);
			Pyrad.wh.push_back(Point2i(nw, nh));
			Pyrad.ImgPyrImg.push_back(smallImg);

			//sprintf(Fname, "C:/temp/_L%d.png", count+1);
			//SaveDataToImage(Fname, Pyrad.ImgPyrImg[count+1], nw, nh, nchannels);
			count++;
		}
	}

	Pyrad.nscales = count + 1;
	return 0;
}

//Features dection and tracking
void LaplacianOfGaussian(double *LOG, int sigma)
{
	int n = ceil(sigma * 3), Size = 2 * n + 1;
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
void LaplacianOfGaussian(double *LOG, int sigma, int PatternSize)
{
	int n = ceil(sigma * 3), Size = 2 * n + 1, hsubset = PatternSize / 2;
	double ii2, jj2, Twosigma2 = 2.0*sigma*sigma, sigma4 = pow(sigma, 4);
	for (int jj = -hsubset; jj <= hsubset; jj++)
	{
		for (int ii = -hsubset; ii <= hsubset; ii++)
		{
			ii2 = ii*ii, jj2 = jj*jj;
			LOG[(ii + hsubset) + (jj + hsubset)*PatternSize] = (ii2 + jj2 - Twosigma2) / sigma4*exp(-(ii2 + jj2) / Twosigma2);
		}
	}

	return;
}
void Gaussian(double *G, int sigma, int PatternSize)
{
	int ii2, jj2, size = MyFtoI(6.0*sigma + 1) / 2 * 2 + 1;
	double sigma2 = 2.0*sigma*sigma, sqrt2Pi_sigma = sqrt(2.0*Pi)*sigma;
	int hsubset = PatternSize / 2;

	for (int jj = -hsubset; jj <= hsubset; jj++)
	{
		for (int ii = -hsubset; ii <= hsubset; ii++)
		{
			ii2 = ii*ii, jj2 = jj*jj;
			G[(ii + hsubset) + (jj + hsubset)*PatternSize] = exp(-(ii2 + jj2) / sigma2) / sqrt2Pi_sigma;
		}
	}

	return;
}
void synthesize_concentric_circles_mask(double *ring_mask_smooth, int *pattern_bi_graylevel, int pattern_size, double sigma, double scale, double *ring_info, int flag, int num_ring_edge)
{
	//ring_mask's size:[Pattern_size,Pattern_size] 
	int ii, jj, kk;
	int es_cen_x = pattern_size / 2;
	int es_cen_y = pattern_size / 2;
	int dark = pattern_bi_graylevel[0];
	int bright = pattern_bi_graylevel[1];
	int hb = 1;	// half-band

	double t = scale / 2.0;
	double r0[10], r1[9];

	for (ii = 0; ii <= num_ring_edge; ii++)
	{
		r0[ii] = ring_info[ii] * t;
		if (ii >= 1)
			r1[ii - 1] = ring_info[ii] * t;
	}

	char *ring_mask = new char[pattern_size*pattern_size];
	for (ii = 0; ii < pattern_size*pattern_size; ii++)
		ring_mask[ii] = (char)bright;

	int g1, g2;
	if (flag == 0)
	{
		for (jj = 0; jj < pattern_size; jj++)
		{
			for (ii = 0; ii < pattern_size; ii++)
			{
				t = sqrt((double)((ii - es_cen_x)*(ii - es_cen_x) + (jj - es_cen_y)*(jj - es_cen_y)));
				for (kk = 0; kk <= num_ring_edge; kk++)
				{
					if (kk % 2 == 0)
					{
						g1 = dark;
						g2 = bright;
					}
					else
					{
						g1 = bright;
						g2 = dark;
					}

					if (kk <= num_ring_edge - 1 && t <= r0[kk] - hb && t > r0[kk + 1] + hb)
						ring_mask[ii + jj*pattern_size] = (char)g1;
					else if (t <= r0[kk] + hb && t > r0[kk] - hb)
						ring_mask[ii + jj*pattern_size] = (char)(MyFtoI(g2*(t + hb - r0[kk]) / (2.0*hb) + g1*(r0[kk] + hb - t) / (2.0*hb)));
				}
				if (t <= r0[num_ring_edge] - hb)
					ring_mask[ii + jj*pattern_size] = (char)(num_ring_edge % 2 == 0 ? dark : bright);
			}
		}
	}
	else
	{
		for (jj = 0; jj < pattern_size; jj++)
		{
			for (ii = 0; ii < pattern_size; ii++)
			{
				t = sqrt((double)((ii - es_cen_x)*(ii - es_cen_x) + (jj - es_cen_y)*(jj - es_cen_y)));
				for (kk = 0; kk < num_ring_edge; kk++)
				{
					if (kk % 2 == 0)
					{
						g1 = dark;
						g2 = bright;
					}
					else
					{
						g1 = bright;
						g2 = dark;
					}

					if (kk<num_ring_edge - 1 && t <= r1[kk] - hb && t > r1[kk + 1] + hb)
						ring_mask[ii + jj*pattern_size] = (char)g1;
					else if (t <= r1[kk] + hb && t > r1[kk] - hb)
						ring_mask[ii + jj*pattern_size] = (char)(MyFtoI(g2*(t + hb - r1[kk]) / (2.0*hb) + g1*(r1[kk] + hb - t) / (2.0*hb)));
				}
				if (t <= r1[num_ring_edge - 1] - hb)
					ring_mask[ii + jj*pattern_size] = (char)(num_ring_edge % 2 == 0 ? bright : dark);
			}
		}
	}

	//SaveDataToGreyImage(ring_mask, pattern_size, pattern_size, TempPathName+(flag==0?_T("syn_ring_1_0.png"):_T("syn_ring_2_0.png")));

	// Gaussian smooth
	Gaussian_smooth(ring_mask, ring_mask_smooth, pattern_size, pattern_size, pattern_bi_graylevel[1], sigma);
	delete[]ring_mask;
	return;
}
void synthesize_square_mask(double *square_mask_smooth, int *pattern_bi_graylevel, int Pattern_size, double sigma, int flag, bool OpenMP)
{
	int ii, jj;
	int es_con_x = Pattern_size / 2;
	int es_con_y = Pattern_size / 2;
	char dark = (char)pattern_bi_graylevel[0];
	char bright = (char)pattern_bi_graylevel[1];
	char mid = (char)((pattern_bi_graylevel[0] + pattern_bi_graylevel[1]) / 2);

	char *square_mask = new char[Pattern_size*Pattern_size];

	for (jj = 0; jj < Pattern_size; jj++)
	{
		for (ii = 0; ii < Pattern_size; ii++)
		{
			if ((ii<es_con_x && jj<es_con_y) || (ii>es_con_x && jj>es_con_y))
				square_mask[ii + jj*Pattern_size] = (flag == 0 ? bright : dark);
			else if (ii == es_con_x || jj == es_con_y)
				square_mask[ii + jj*Pattern_size] = mid;
			else
				square_mask[ii + jj*Pattern_size] = (flag == 0 ? dark : bright);
		}
	}
	//SaveDataToImage("C:/temp/t.png", square_mask, Pattern_size, Pattern_size, 1);

	// Gaussian smooth
	Gaussian_smooth(square_mask, square_mask_smooth, Pattern_size, Pattern_size, pattern_bi_graylevel[1], sigma);

	for (jj = 0; jj < Pattern_size; jj++)
		for (ii = 0; ii < Pattern_size; ii++)
			square_mask[ii + jj*Pattern_size] = (char)MyFtoI((square_mask_smooth[ii + jj*Pattern_size]));

	//SaveDataToImage("C:/temp/st.png", square_mask, Pattern_size, Pattern_size, 1);
	delete[]square_mask;

	return;
}
void synthesize_pattern(int pattern, double *Pattern, int *Pattern_size, int *pattern_bi_graylevel, int *Subset, double *ctrl_pts_info, double scale, int board_width, int board_height, double sigma, int num_ring_edge, int num_target = 0, bool OpenMP = false)
{
	int i, addon = MyFtoI(ctrl_pts_info[0] * scale*0.2) / 4 * 4;
	if (pattern == 0)//CHECKER
	{
		Pattern_size[0] = MyFtoI(ctrl_pts_info[0] * scale) / 4 * 4 + addon;
		Subset[0] = MyFtoI(ctrl_pts_info[0] * scale) / 4 * 2 + addon / 4;
		if (Subset[0] > 20)
			Subset[0] = 20;

		Pattern_size[1] = Pattern_size[0];
		Subset[1] = Subset[0];
	}
	else if (pattern == 1)//RING
	{
		for (i = 0; i <= 1; i++)
		{
			Pattern_size[i] = MyFtoI(ctrl_pts_info[i] * scale) / 4 * 4 + addon;
			Subset[i] = MyFtoI(ctrl_pts_info[i] * scale) / 4 * 2 + addon / 4;
		}
	}

	if (pattern == 0)
		for (i = 0; i <= 1; i++)
			synthesize_square_mask(Pattern + Pattern_size[0] * Pattern_size[0] * i, pattern_bi_graylevel, Pattern_size[0], sigma, i, OpenMP);
	else if (pattern == 1)
		for (i = 0; i <= 1; i++)
			synthesize_concentric_circles_mask(Pattern + Pattern_size[0] * Pattern_size[0] * i, pattern_bi_graylevel, Pattern_size[i], sigma, scale, ctrl_pts_info, i, num_ring_edge);

	return;
}

//ECC image alignment
static void image_jacobian_homo_ECC(const Mat& src1, const Mat& src2, const Mat& src3, const Mat& src4, const Mat& src5, Mat& dst)
{


	CV_Assert(src1.size() == src2.size());
	CV_Assert(src1.size() == src3.size());
	CV_Assert(src1.size() == src4.size());

	CV_Assert(src1.rows == dst.rows);
	CV_Assert(dst.cols == (src1.cols * 8));
	CV_Assert(dst.type() == CV_32FC1);

	CV_Assert(src5.isContinuous());


	const float* hptr = src5.ptr<float>(0);

	const float h0_ = hptr[0];
	const float h1_ = hptr[3];
	const float h2_ = hptr[6];
	const float h3_ = hptr[1];
	const float h4_ = hptr[4];
	const float h5_ = hptr[7];
	const float h6_ = hptr[2];
	const float h7_ = hptr[5];

	const int w = src1.cols;


	//create denominator for all points as a block
	Mat den_ = src3*h2_ + src4*h5_ + 1.0;//check the time of this! otherwise use addWeighted

	//create projected points
	Mat hatX_ = -src3*h0_ - src4*h3_ - h6_;
	divide(hatX_, den_, hatX_);
	Mat hatY_ = -src3*h1_ - src4*h4_ - h7_;
	divide(hatY_, den_, hatY_);


	//instead of dividing each block with den,
	//just pre-devide the block of gradients (it's more efficient)

	Mat src1Divided_;
	Mat src2Divided_;

	divide(src1, den_, src1Divided_);
	divide(src2, den_, src2Divided_);


	//compute Jacobian blocks (8 blocks)

	dst.colRange(0, w) = src1Divided_.mul(src3);//1

	dst.colRange(w, 2 * w) = src2Divided_.mul(src3);//2

	Mat temp_ = (hatX_.mul(src1Divided_) + hatY_.mul(src2Divided_));
	dst.colRange(2 * w, 3 * w) = temp_.mul(src3);//3

	hatX_.release();
	hatY_.release();

	dst.colRange(3 * w, 4 * w) = src1Divided_.mul(src4);//4

	dst.colRange(4 * w, 5 * w) = src2Divided_.mul(src4);//5

	dst.colRange(5 * w, 6 * w) = temp_.mul(src4);//6

	src1Divided_.copyTo(dst.colRange(6 * w, 7 * w));//7

	src2Divided_.copyTo(dst.colRange(7 * w, 8 * w));//8
}
static void image_jacobian_euclidean_ECC(const Mat& src1, const Mat& src2, const Mat& src3, const Mat& src4, const Mat& src5, Mat& dst)
{

	CV_Assert(src1.size() == src2.size());
	CV_Assert(src1.size() == src3.size());
	CV_Assert(src1.size() == src4.size());

	CV_Assert(src1.rows == dst.rows);
	CV_Assert(dst.cols == (src1.cols * 3));
	CV_Assert(dst.type() == CV_32FC1);

	CV_Assert(src5.isContinuous());

	const float* hptr = src5.ptr<float>(0);

	const float h0 = hptr[0];//cos(theta)
	const float h1 = hptr[3];//sin(theta)

	const int w = src1.cols;

	//create -sin(theta)*X -cos(theta)*Y for all points as a block -> hatX
	Mat hatX = -(src3*h1) - (src4*h0);

	//create cos(theta)*X -sin(theta)*Y for all points as a block -> hatY
	Mat hatY = (src3*h0) - (src4*h1);


	//compute Jacobian blocks (3 blocks)
	dst.colRange(0, w) = (src1.mul(hatX)) + (src2.mul(hatY));//1

	src1.copyTo(dst.colRange(w, 2 * w));//2
	src2.copyTo(dst.colRange(2 * w, 3 * w));//3
}
static void image_jacobian_affine_ECC(const Mat& src1, const Mat& src2, const Mat& src3, const Mat& src4, Mat& dst)
{

	CV_Assert(src1.size() == src2.size());
	CV_Assert(src1.size() == src3.size());
	CV_Assert(src1.size() == src4.size());

	CV_Assert(src1.rows == dst.rows);
	CV_Assert(dst.cols == (6 * src1.cols));

	CV_Assert(dst.type() == CV_32FC1);


	const int w = src1.cols;

	//compute Jacobian blocks (6 blocks)

	dst.colRange(0, w) = src1.mul(src3);//1
	dst.colRange(w, 2 * w) = src2.mul(src3);//2
	dst.colRange(2 * w, 3 * w) = src1.mul(src4);//3
	dst.colRange(3 * w, 4 * w) = src2.mul(src4);//4
	src1.copyTo(dst.colRange(4 * w, 5 * w));//5
	src2.copyTo(dst.colRange(5 * w, 6 * w));//6
}
static void image_jacobian_translation_ECC(const Mat& src1, const Mat& src2, Mat& dst)
{

	CV_Assert(src1.size() == src2.size());

	CV_Assert(src1.rows == dst.rows);
	CV_Assert(dst.cols == (src1.cols * 2));
	CV_Assert(dst.type() == CV_32FC1);

	const int w = src1.cols;

	//compute Jacobian blocks (2 blocks)
	src1.copyTo(dst.colRange(0, w));
	src2.copyTo(dst.colRange(w, 2 * w));
}
static void project_onto_jacobian_ECC(const Mat& src1, const Mat& src2, Mat& dst)
{
	/* this functions is used for two types of projections. If src1.cols ==src.cols
	it does a blockwise multiplication (like in the outer product of vectors)
	of the blocks in matrices src1 and src2 and dst
	has size (number_of_blcks x number_of_blocks), otherwise dst is a vector of size
	(number_of_blocks x 1) since src2 is "multiplied"(dot) with each block of src1.
	The number_of_blocks is equal to the number of parameters we are lloking for
	(i.e. rtanslation:2, euclidean: 3, affine: 6, homography: 8)
	*/
	CV_Assert(src1.rows == src2.rows);
	CV_Assert((src1.cols % src2.cols) == 0);
	int w;

	float* dstPtr = dst.ptr<float>(0);

	if (src1.cols != src2.cols){//dst.cols==1
		w = src2.cols;
		for (int i = 0; i < dst.rows; i++){
			dstPtr[i] = (float)src2.dot(src1.colRange(i*w, (i + 1)*w));
		}
	}

	else {
		CV_Assert(dst.cols == dst.rows); //dst is square (and symmetric)
		w = src2.cols / dst.cols;
		Mat mat;
		for (int i = 0; i < dst.rows; i++){

			mat = Mat(src1.colRange(i*w, (i + 1)*w));
			dstPtr[i*(dst.rows + 1)] = (float)pow(norm(mat), 2); //diagonal elements

			for (int j = i + 1; j < dst.cols; j++){ //j starts from i+1
				dstPtr[i*dst.cols + j] = (float)mat.dot(src2.colRange(j*w, (j + 1)*w));
				dstPtr[j*dst.cols + i] = dstPtr[i*dst.cols + j]; //due to symmetry
			}
		}
	}
}
static void update_warping_matrix_ECC(Mat& map_matrix, const Mat& update, const int motionType)
{
	CV_Assert(map_matrix.type() == CV_32FC1);
	CV_Assert(update.type() == CV_32FC1);

	CV_Assert(motionType == MOTION_TRANSLATION || motionType == MOTION_EUCLIDEAN ||
		motionType == MOTION_AFFINE || motionType == MOTION_HOMOGRAPHY);

	if (motionType == MOTION_HOMOGRAPHY)
		CV_Assert(map_matrix.rows == 3 && update.rows == 8);
	else if (motionType == MOTION_AFFINE)
		CV_Assert(map_matrix.rows == 2 && update.rows == 6);
	else if (motionType == MOTION_EUCLIDEAN)
		CV_Assert(map_matrix.rows == 2 && update.rows == 3);
	else
		CV_Assert(map_matrix.rows == 2 && update.rows == 2);

	CV_Assert(update.cols == 1);

	CV_Assert(map_matrix.isContinuous());
	CV_Assert(update.isContinuous());


	float* mapPtr = map_matrix.ptr<float>(0);
	const float* updatePtr = update.ptr<float>(0);


	if (motionType == MOTION_TRANSLATION){
		mapPtr[2] += updatePtr[0];
		mapPtr[5] += updatePtr[1];
	}
	if (motionType == MOTION_AFFINE) {
		mapPtr[0] += updatePtr[0];
		mapPtr[3] += updatePtr[1];
		mapPtr[1] += updatePtr[2];
		mapPtr[4] += updatePtr[3];
		mapPtr[2] += updatePtr[4];
		mapPtr[5] += updatePtr[5];
	}
	if (motionType == MOTION_HOMOGRAPHY) {
		mapPtr[0] += updatePtr[0];
		mapPtr[3] += updatePtr[1];
		mapPtr[6] += updatePtr[2];
		mapPtr[1] += updatePtr[3];
		mapPtr[4] += updatePtr[4];
		mapPtr[7] += updatePtr[5];
		mapPtr[2] += updatePtr[6];
		mapPtr[5] += updatePtr[7];
	}
	if (motionType == MOTION_EUCLIDEAN) {
		double new_theta = updatePtr[0];
		if (mapPtr[3] > 0)
			new_theta += acos(mapPtr[0]);

		if (mapPtr[3] < 0)
			new_theta -= acos(mapPtr[0]);

		mapPtr[2] += updatePtr[1];
		mapPtr[5] += updatePtr[2];
		mapPtr[0] = mapPtr[4] = (float)cos(new_theta);
		mapPtr[3] = (float)sin(new_theta);
		mapPtr[1] = -mapPtr[3];
	}
}
double findTransformECC(InputArray templateImage, InputArray inputImage, InputOutputArray warpMatrix, int motionType, TermCriteria criteria)
{
	Mat src = templateImage.getMat();//template iamge
	Mat dst = inputImage.getMat(); //input image (to be warped)
	Mat map = warpMatrix.getMat(); //warp (transformation)

	CV_Assert(!src.empty());
	CV_Assert(!dst.empty());


	if (!(src.type() == dst.type()))
		printf("Both input images must have the same data type");

	//accept only 1-channel images
	if (src.type() != CV_8UC1 && src.type() != CV_32FC1)
		printf("Images must have 8uC1 or 32fC1 type");

	if (map.type() != CV_32FC1)
		printf("warpMatrix must be single-channel floating-point matrix");

	CV_Assert(map.cols == 3);
	CV_Assert(map.rows == 2 || map.rows == 3);

	CV_Assert(motionType == MOTION_AFFINE || motionType == MOTION_HOMOGRAPHY ||
		motionType == MOTION_EUCLIDEAN || motionType == MOTION_TRANSLATION);

	if (motionType == MOTION_HOMOGRAPHY){
		CV_Assert(map.rows == 3);
	}

	CV_Assert(criteria.type & TermCriteria::COUNT || criteria.type & TermCriteria::EPS);
	const int    numberOfIterations = (criteria.type & TermCriteria::COUNT) ? criteria.maxCount : 200;
	const double termination_eps = (criteria.type & TermCriteria::EPS) ? criteria.epsilon : -1;

	int paramTemp = 6;//default: affine
	switch (motionType){
	case MOTION_TRANSLATION:
		paramTemp = 2;
		break;
	case MOTION_EUCLIDEAN:
		paramTemp = 3;
		break;
	case MOTION_HOMOGRAPHY:
		paramTemp = 8;
		break;
	}


	const int numberOfParameters = paramTemp;

	const int ws = src.cols;
	const int hs = src.rows;
	const int wd = dst.cols;
	const int hd = dst.rows;

	Mat Xcoord = Mat(1, ws, CV_32F);
	Mat Ycoord = Mat(hs, 1, CV_32F);
	Mat Xgrid = Mat(hs, ws, CV_32F);
	Mat Ygrid = Mat(hs, ws, CV_32F);

	float* XcoPtr = Xcoord.ptr<float>(0);
	float* YcoPtr = Ycoord.ptr<float>(0);
	int j;
	for (j = 0; j < ws; j++)
		XcoPtr[j] = (float)j;
	for (j = 0; j < hs; j++)
		YcoPtr[j] = (float)j;

	repeat(Xcoord, hs, 1, Xgrid);
	repeat(Ycoord, 1, ws, Ygrid);

	Xcoord.release();
	Ycoord.release();

	Mat templateZM = Mat(hs, ws, CV_32F);// to store the (smoothed)zero-mean version of template
	Mat templateFloat = Mat(hs, ws, CV_32F);// to store the (smoothed) template
	Mat imageFloat = Mat(hd, wd, CV_32F);// to store the (smoothed) input image
	Mat imageWarped = Mat(hs, ws, CV_32F);// to store the warped zero-mean input image
	Mat allOnes = Mat::ones(hd, wd, CV_8U); //to use it for mask warping
	Mat imageMask = Mat(hs, ws, CV_8U); //to store the final mask

	//gaussian filtering is optional
	src.convertTo(templateFloat, templateFloat.type());
	GaussianBlur(templateFloat, templateFloat, Size(5, 5), 0, 0);//is in-place filtering slower?

	dst.convertTo(imageFloat, imageFloat.type());
	GaussianBlur(imageFloat, imageFloat, Size(5, 5), 0, 0);

	// needed matrices for gradients and warped gradients
	Mat gradientX = Mat::zeros(hd, wd, CV_32FC1);
	Mat gradientY = Mat::zeros(hd, wd, CV_32FC1);
	Mat gradientXWarped = Mat(hs, ws, CV_32FC1);
	Mat gradientYWarped = Mat(hs, ws, CV_32FC1);


	// calculate first order image derivatives
	Matx13f dx(-0.5f, 0.0f, 0.5f);

	filter2D(imageFloat, gradientX, -1, dx);
	filter2D(imageFloat, gradientY, -1, dx.t());


	// matrices needed for solving linear equation system for maximizing ECC
	Mat jacobian = Mat(hs, ws*numberOfParameters, CV_32F);
	Mat hessian = Mat(numberOfParameters, numberOfParameters, CV_32F);
	Mat hessianInv = Mat(numberOfParameters, numberOfParameters, CV_32F);
	Mat imageProjection = Mat(numberOfParameters, 1, CV_32F);
	Mat templateProjection = Mat(numberOfParameters, 1, CV_32F);
	Mat imageProjectionHessian = Mat(numberOfParameters, 1, CV_32F);
	Mat errorProjection = Mat(numberOfParameters, 1, CV_32F);

	Mat deltaP = Mat(numberOfParameters, 1, CV_32F);//transformation parameter correction
	Mat error = Mat(hs, ws, CV_32F);//error as 2D matrix

	const int imageFlags = INTER_LINEAR + WARP_INVERSE_MAP;
	const int maskFlags = INTER_NEAREST + WARP_INVERSE_MAP;


	// iteratively update map_matrix
	double rho = -1;
	double last_rho = -termination_eps;
	for (int i = 1; (i <= numberOfIterations) && (fabs(rho - last_rho) >= termination_eps); i++)
	{

		// warp-back portion of the inputImage and gradients to the coordinate space of the templateImage
		if (motionType != MOTION_HOMOGRAPHY)
		{
			warpAffine(imageFloat, imageWarped, map, imageWarped.size(), imageFlags);
			warpAffine(gradientX, gradientXWarped, map, gradientXWarped.size(), imageFlags);
			warpAffine(gradientY, gradientYWarped, map, gradientYWarped.size(), imageFlags);
			warpAffine(allOnes, imageMask, map, imageMask.size(), maskFlags);
		}
		else
		{
			warpPerspective(imageFloat, imageWarped, map, imageWarped.size(), imageFlags);
			warpPerspective(gradientX, gradientXWarped, map, gradientXWarped.size(), imageFlags);
			warpPerspective(gradientY, gradientYWarped, map, gradientYWarped.size(), imageFlags);
			warpPerspective(allOnes, imageMask, map, imageMask.size(), maskFlags);
		}


		Scalar imgMean, imgStd, tmpMean, tmpStd;
		meanStdDev(imageWarped, imgMean, imgStd, imageMask);
		meanStdDev(templateFloat, tmpMean, tmpStd, imageMask);

		subtract(imageWarped, imgMean, imageWarped, imageMask);//zero-mean input
		templateZM = Mat::zeros(templateZM.rows, templateZM.cols, templateZM.type());
		subtract(templateFloat, tmpMean, templateZM, imageMask);//zero-mean template

		const double tmpNorm = std::sqrt(countNonZero(imageMask)*(tmpStd.val[0])*(tmpStd.val[0]));
		const double imgNorm = std::sqrt(countNonZero(imageMask)*(imgStd.val[0])*(imgStd.val[0]));

		// calculate jacobian of image wrt parameters
		switch (motionType){
		case MOTION_AFFINE:
			image_jacobian_affine_ECC(gradientXWarped, gradientYWarped, Xgrid, Ygrid, jacobian);
			break;
		case MOTION_HOMOGRAPHY:
			image_jacobian_homo_ECC(gradientXWarped, gradientYWarped, Xgrid, Ygrid, map, jacobian);
			break;
		case MOTION_TRANSLATION:
			image_jacobian_translation_ECC(gradientXWarped, gradientYWarped, jacobian);
			break;
		case MOTION_EUCLIDEAN:
			image_jacobian_euclidean_ECC(gradientXWarped, gradientYWarped, Xgrid, Ygrid, map, jacobian);
			break;
		}

		// calculate Hessian and its inverse
		project_onto_jacobian_ECC(jacobian, jacobian, hessian);

		hessianInv = hessian.inv();

		const double correlation = templateZM.dot(imageWarped);

		// calculate enhanced correlation coefficiont (ECC)->rho
		last_rho = rho;
		rho = correlation / (imgNorm*tmpNorm);

		// project images into jacobian
		project_onto_jacobian_ECC(jacobian, imageWarped, imageProjection);
		project_onto_jacobian_ECC(jacobian, templateZM, templateProjection);


		// calculate the parameter lambda to account for illumination variation
		imageProjectionHessian = hessianInv*imageProjection;
		const double lambda_n = (imgNorm*imgNorm) - imageProjection.dot(imageProjectionHessian);
		const double lambda_d = correlation - templateProjection.dot(imageProjectionHessian);
		if (lambda_d <= 0.0)
		{
			rho = -1;
			printf("The algorithm stopped before its convergence. The correlation is going to be minimized. Images may be uncorrelated or non-overlapped");
		}
		const double lambda = (lambda_n / lambda_d);

		// estimate the update step delta_p
		error = lambda*templateZM - imageWarped;
		project_onto_jacobian_ECC(jacobian, error, errorProjection);
		deltaP = hessianInv * errorProjection;

		// update warping matrix
		update_warping_matrix_ECC(map, deltaP, motionType);
	}

	// return final correlation coefficient
	return rho;
}

double TMatchingSuperCoarse(double *Pattern, int pattern_size, int hsubset, double *Image, int width, int height, int nchannels, Point2i &POI, int search_area, double thresh, double *T)
{
	//No interpolation at all, just slide the template around to compute the ZNCC
	int m, i, j, ii, jj, iii, jjj, II, JJ, length = width*height, patternLength = pattern_size*pattern_size;
	double t_f, t_g, t_1, t_2, t_3, t_4, t_5, m_F, m_G;

	Point2d w_pt, ima_pt;
	int Pattern_cen_x = pattern_size / 2;
	int Pattern_cen_y = pattern_size / 2;

	FILE *fp1, *fp2;
	bool printout = false;

	Point2i orgPOI = POI;
	bool createdMem = false;
	if (T == NULL)
		T = new double[2 * (2 * hsubset + 1)*(2 * hsubset + 1)*nchannels], createdMem = true;

	double zncc = 0.0;
	for (j = -search_area; j <= search_area; j++)
	{
		for (i = -search_area; i <= search_area; i++)
		{
			m = -1;
			t_f = 0.0, t_g = 0.0;

			if (printout)
			{
				fp1 = fopen("C:/temp/src.txt", "w+");
				fp2 = fopen("C:/temp/tar.txt", "w+");
			}

			for (jjj = -hsubset; jjj <= hsubset; jjj++)
			{
				for (iii = -hsubset; iii <= hsubset; iii++)
				{
					for (int kk = 0; kk < nchannels; kk++)
					{
						jj = Pattern_cen_y + jjj, ii = Pattern_cen_x + iii;
						JJ = orgPOI.y + jjj + j, II = orgPOI.x + iii + i;

						m_F = Pattern[ii + jj*pattern_size + kk*patternLength], m_G = Image[II + JJ*width + kk*length];

						if (printout)
							fprintf(fp1, "%.2f ", m_F), fprintf(fp2, "%.2f ", m_G);

						m++;
						T[2 * m] = m_F, T[2 * m + 1] = m_G;
						t_f += m_F, t_g += m_G;
					}
				}
				if (printout)
					fprintf(fp1, "\n"), fprintf(fp2, "\n");
			}
			if (printout)
				fclose(fp1), fclose(fp2);

			t_f = t_f / (m + 1);
			t_g = t_g / (m + 1);
			t_1 = 0.0, t_2 = 0.0, t_3 = 0.0;
			for (iii = 0; iii <= m; iii++)
			{
				t_4 = *(T + 2 * iii + 0) - t_f;
				t_5 = *(T + 2 * iii + 1) - t_g;
				t_1 += t_4*t_5, t_2 += t_4*t_4, t_3 += t_5*t_5;
			}

			t_2 = sqrt(t_2*t_3);
			if (t_2 < 1e-10)
				t_2 = 1e-10;

			t_3 = t_1 / t_2;
			if (t_3 > 1.0 || t_3 < -1.0)
				t_3 = 0.0;

			if (t_3>thresh && t_3 > zncc)
			{
				zncc = t_3;
				POI.x = orgPOI.x + i, POI.y = orgPOI.y + j;
			}
			else if (t_3 < -thresh && t_3 < zncc)
			{
				zncc = t_3;
				POI.x = orgPOI.x + i, POI.y = orgPOI.y + j;
			}
		}
	}
	zncc = abs(zncc);

	if (createdMem)
		delete[]T;
	return zncc;
}
int TMatchingCoarse(double *Pattern, int pattern_size, int hsubset, double *Para, int width, int height, int nchannels, Point2d &POI, int search_area, double thresh, double &zncc, int InterpAlgo, double *InitPara = NULL, double *maxZNCC = NULL)
{
	//Compute the zncc in a local region (5x5). No iteration is used to solve for shape parameters
	//InitPara: 3x3 homography matrix
	int i, j, ii, jj, iii, jjj, kkk, length = width*height, pattern_length = pattern_size*pattern_size, pjump = search_area > 5 ? 2 : 1;
	double II, JJ, t_1, t_2, t_3, t_4, m_F, m_G, S[6];

	Point2d w_pt, ima_pt;
	int Pattern_cen_x = pattern_size / 2;
	int Pattern_cen_y = pattern_size / 2;

	bool printout = false;
	FILE *fp1 = 0, *fp2 = 0;
	int m;
	double t_f, t_g, t_5, CeresResamplingSpline2 = 0.0, yyy = 0.0;
	double *T = new double[2 * (2 * hsubset + 1)*(2 * hsubset + 1)];

	zncc = 0.0;
	for (j = -search_area; j <= search_area; j += pjump)
	{
		for (i = -search_area; i <= search_area; i += pjump)
		{
			m = -1;
			t_f = 0.0, t_g = 0.0;

			if (printout)
				fp1 = fopen("C:/temp/src.txt", "w+"), fp2 = fopen("C:/temp/tar.txt", "w+");

			for (jjj = -hsubset; jjj <= hsubset; jjj++)
			{
				for (iii = -hsubset; iii <= hsubset; iii++)
				{
					for (kkk = 0; kkk < nchannels; kkk++)
					{
						jj = Pattern_cen_y + jjj;
						ii = Pattern_cen_x + iii;

						if (InitPara == NULL)
						{
							II = (int)(POI.x + 0.5) + iii + i;
							JJ = (int)(POI.y + 0.5) + jjj + j;
						}
						else
						{
							II = (InitPara[0] * iii + InitPara[1] * jjj + InitPara[2]) / (InitPara[6] * iii + InitPara[7] * jjj + InitPara[8]);
							JJ = (InitPara[3] * iii + InitPara[4] * jjj + InitPara[5]) / (InitPara[6] * iii + InitPara[7] * jjj + InitPara[8]);
						}

						Get_Value_Spline(Para + kkk*length, width, height, II, JJ, S, -1, InterpAlgo);

						m_F = Pattern[ii + jj*pattern_size + kkk*pattern_length], m_G = S[0];
						m++;
						T[2 * m] = m_F, T[2 * m + 1] = m_G;
						t_f += m_F, t_g += m_G;

						if (printout)
							fprintf(fp1, "%.2f ", m_G), fprintf(fp2, "%.2f ", m_F);
					}
				}
				if (printout)
					fprintf(fp1, "\n"), fprintf(fp2, "\n");
			}
			if (printout)
				fclose(fp1), fclose(fp2);

			t_f = t_f / (m + 1), t_g = t_g / (m + 1);
			t_1 = 0.0, t_2 = 0.0, t_3 = 0.0;
			for (iii = 0; iii <= m; iii++)
			{
				t_4 = *(T + 2 * iii + 0) - t_f, t_5 = *(T + 2 * iii + 1) - t_g;
				t_1 += (t_4*t_5), t_2 += (t_4*t_4), t_3 += (t_5*t_5);
			}

			t_2 = sqrt(t_2*t_3);
			if (t_2 < 1e-10)
				t_2 = 1e-10;

			t_3 = t_1 / t_2;
			if (t_3 > 1.0 || t_3 < -1.0)
				t_3 = 0.0;

			if (t_3>thresh && t_3 > zncc)
			{
				zncc = t_3;
				CeresResamplingSpline2 = i, yyy = j;
			}
			else if (t_3 < -thresh && abs(t_3) > zncc)
			{
				zncc = t_3;
				CeresResamplingSpline2 = i, yyy = j;
			}
		}
	}
	if (InitPara != NULL)
		maxZNCC[0] = abs(zncc);

	delete[]T;
	if (zncc > thresh)
	{
		POI.x = (int)(POI.x + 0.5) + CeresResamplingSpline2;
		POI.y = (int)(POI.y + 0.5) + yyy;
		zncc = abs(zncc);
		return 0;
	}
	else if (zncc < -thresh)
	{
		POI.x = (int)(POI.x + 0.5) + CeresResamplingSpline2;
		POI.y = (int)(POI.y + 0.5) + yyy;
		zncc = abs(zncc);
		return 1;
	}
	else
		return -1;
}
double TMatchingFine_ZNCC(double *Pattern, int pattern_size, int hsubset, double *Para, int width, int height, int nchannels, Point2d &POI, int advanced_tech, int Convergence_Criteria, double ZNCCthresh, int InterpAlgo, double *Znssd_reqd)
{
	int i, j, k, m, ii, jj, kk, iii, jjj, iii2, jjj2;
	double II, JJ, iii_n, jjj_n, gx, gy, DIC_Coeff, DIC_Coeff_min, t_1, t_2, t_3, t_4, t_5, t_6, numx, numy, denum, denum2, t_7, m_F, m_G, t_f, t_ff, t_g, S[6];
	double conv_crit_1 = pow(10.0, -Convergence_Criteria - 2);
	double conv_crit_2 = conv_crit_1*0.1;
	int NN[] = { 3, 6, 12, 8 }, P_Jump_Incr[] = { 1, 1, 1, 1 };
	int nn = NN[advanced_tech], _iter = 0, Iter_Max = 20;
	int p_jump, p_jump_0 = 1, p_jump_incr = P_Jump_Incr[advanced_tech];
	int length = width*height, pattern_length = pattern_size*pattern_size;

	double AA[144], BB[12], CC[12];

	bool createMem = false;
	if (Znssd_reqd == NULL)
	{
		createMem = true;
		Znssd_reqd = new double[9 * (2 * hsubset + 1)*(2 * hsubset + 1)*nchannels];
	}

	int Pattern_cen_x = pattern_size / 2;
	int Pattern_cen_y = pattern_size / 2;

	double p[12], p_best[12];
	for (i = 0; i < 12; i++)
		p[i] = 0.0;

	nn = NN[advanced_tech];
	int pixel_increment_in_subset[] = { 1, 2, 2, 3 };

	bool printout = false;
	FILE *fp1 = 0, *fp2 = 0;

	/// Iteration: Begin
	bool Break_Flag = false;
	DIC_Coeff_min = 4.0;
	for (p_jump = p_jump_0; p_jump > 0; p_jump -= p_jump_incr)
	{
		for (k = 0; k < Iter_Max; k++)
		{
			m = -1;
			t_1 = 0.0, t_2 = 0.0;
			for (iii = 0; iii < 144; iii++)
				AA[iii] = 0.0;
			for (iii = 0; iii < 12; iii++)
				BB[iii] = 0.0;

			if (printout)
				fp1 = fopen("C:/temp/src.txt", "w+"), fp2 = fopen("C:/temp/tar.txt", "w+");

			for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
			{
				for (iii = -hsubset; iii <= hsubset; iii += p_jump)
				{
					ii = Pattern_cen_x + iii, jj = Pattern_cen_y + jjj;
					if (ii<0 || ii>(width - 1) || jj<0 || jj>(height - 1))
						continue;

					iii2 = iii*iii, jjj2 = jjj*jjj;
					if (advanced_tech == 0)
					{
						II = POI.x + iii + p[0] + p[2] * iii;
						JJ = POI.y + jjj + p[1] + p[2] * jjj;
					}
					else if (advanced_tech == 1)
					{
						II = POI.x + iii + p[0] + p[2] * iii + p[3] * jjj;
						JJ = POI.y + jjj + p[1] + p[4] * iii + p[5] * jjj;
					}
					else if (advanced_tech == 2)
					{
						II = POI.x + iii + p[0] + p[2] * iii + p[3] * jjj + p[6] * iii2*0.5 + p[7] * jjj2*0.5 + p[8] * iii*jjj;
						JJ = POI.y + jjj + p[1] + p[4] * iii + p[5] * jjj + p[9] * iii2*0.5 + p[10] * jjj2*0.5 + p[11] * iii*jjj;
					}
					else
					{
						denum = 1.0 + p[6] * iii + p[7] * jjj;
						numx = POI.x + iii + p[0] + p[2] * iii + p[3] * jjj;
						numy = POI.y + jjj + p[1] + p[4] * iii + p[5] * jjj;
						II = numx / denum;
						JJ = numy / denum;
					}

					if (II<0.0 || II>(double)(width - 1) - (1e-10) || JJ<0.0 || JJ>(double)(height - 1) - (1e-10))
						continue;

					for (kk = 0; kk < nchannels; kk++)
					{
						Get_Value_Spline(Para + kk*length, width, height, II, JJ, S, 0, InterpAlgo);
						m_F = Pattern[ii + jj*pattern_size + kk*pattern_length];
						m_G = S[0], gx = S[1], gy = S[2];
						m++;

						Znssd_reqd[9 * m + 0] = m_F, Znssd_reqd[9 * m + 1] = m_G;
						Znssd_reqd[9 * m + 2] = gx, Znssd_reqd[9 * m + 3] = gy;
						Znssd_reqd[9 * m + 4] = (double)iii, Znssd_reqd[9 * m + 5] = (double)jjj;
						if (advanced_tech == 3)
							Znssd_reqd[9 * m + 6] = numx, Znssd_reqd[9 * m + 7] = numy, Znssd_reqd[9 * m + 8] = denum;

						t_1 += m_F, t_2 += m_G;

						if (printout)
							fprintf(fp1, "%e ", m_F), fprintf(fp2, "%e ", m_G);
					}
				}
				if (printout)
					fprintf(fp1, "\n"), fprintf(fp2, "\n");
			}
			if (printout)
				fclose(fp1), fclose(fp2);

			if (k == 0)
			{
				t_f = t_1 / (m + 1);
				t_1 = 0.0;
				for (iii = 0; iii <= m; iii++)
				{
					t_4 = Znssd_reqd[9 * iii + 0] - t_f;
					t_1 += t_4*t_4;
				}
				t_ff = sqrt(t_1);
			}

			t_g = t_2 / (m + 1);
			t_2 = 0.0;
			for (iii = 0; iii <= m; iii++)
			{
				t_5 = Znssd_reqd[9 * iii + 1] - t_g;
				t_2 += t_5*t_5;
			}
			t_2 = sqrt(t_2);

			DIC_Coeff = 0.0;
			for (iii = 0; iii <= m; iii++)
			{
				t_4 = Znssd_reqd[9 * iii + 0] - t_f;
				t_5 = Znssd_reqd[9 * iii + 1] - t_g;
				t_6 = t_5 / t_2 - t_4 / t_ff;
				t_3 = t_6 / t_2;
				gx = Znssd_reqd[9 * iii + 2], gy = Znssd_reqd[9 * iii + 3];
				iii_n = Znssd_reqd[9 * iii + 4], jjj_n = Znssd_reqd[9 * iii + 5];
				if (advanced_tech == 3)
				{
					denum = Znssd_reqd[9 * ii + 8];
					denum2 = denum*denum;
					t_7 = (gx*Znssd_reqd[9 * iii + 6] + gy*Znssd_reqd[9 * ii + 7]) / denum2;
					CC[0] = gx / denum, CC[1] = gy / denum;
					CC[2] = gx*iii_n / denum, CC[3] = gx*jjj_n / denum;
					CC[4] = gy*iii_n / denum, CC[5] = gy*jjj_n / denum;
					CC[6] = -t_7*iii_n;
					CC[7] = -t_7*jjj_n;
				}
				else
				{
					CC[0] = gx, CC[1] = gy;
					if (advanced_tech == 0)
						CC[2] = gx*iii_n + gy*jjj_n;
					if (advanced_tech == 1 || advanced_tech == 2)
					{
						CC[2] = gx*iii_n, CC[3] = gx*jjj_n;
						CC[4] = gy*iii_n, CC[5] = gy*jjj_n;
					}
					if (advanced_tech == 2)
					{
						CC[6] = gx*iii_n*iii_n*0.5, CC[7] = gx*jjj_n*jjj_n*0.5, CC[8] = gx*iii_n*jjj_n;
						CC[9] = gy*iii_n*iii_n*0.5, CC[10] = gy*jjj_n*jjj_n*0.5, CC[11] = gy*iii_n*jjj_n;
					}
				}

				for (j = 0; j < nn; j++)
				{
					BB[j] += t_3*CC[j];
					for (i = 0; i < nn; i++)
						AA[j*nn + i] += CC[i] * CC[j] / (t_2*t_2);
				}

				DIC_Coeff += t_6*t_6;
			}

			QR_Solution_Double(AA, BB, nn, nn);
			for (iii = 0; iii < nn; iii++)
				p[iii] -= BB[iii];

			if (!IsNumber(p[0]) || abs(p[0]) > hsubset || abs(p[1]) > hsubset)
			{
				if (createMem)
					delete[]Znssd_reqd;
				return false;
			}

			if (DIC_Coeff < DIC_Coeff_min)	// If the iteration does not converge, this can be helpful
			{
				DIC_Coeff_min = DIC_Coeff;
				for (iii = 0; iii < nn; iii++)
					p_best[iii] = p[iii];
			}

			if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
			{
				for (iii = 2; iii < nn; iii++)
				{
					if (fabs(BB[iii]) > conv_crit_2)
						break;
				}
				if (iii == nn)
					Break_Flag = true;
			}

			if (Break_Flag)
				break;
		}
		// In case the iteration converges to "wrong" points, always use the data that lead to the least-square value.
		for (iii = 0; iii < nn; iii++)
			p[iii] = p_best[iii];
	}
	/// Iteration: End

	if (createMem)
		delete[]Znssd_reqd;
	if (abs(p[0]) > hsubset || abs(p[1]) > hsubset || p[0] != p[0] || p[1] != p[1] || 1.0 - 0.5*DIC_Coeff_min < ZNCCthresh)
		return false;

	POI.x += p[0], POI.y += p[1];

	return 1.0 - 0.5*DIC_Coeff_min;
}
double TrackingByLK(double *RefPara, double *TarPara, int hsubset, int widthRef, int heightRef, int widthTar, int heightTar, int nchannels, Point2d PR, Point2d PT, int advanced_tech, int Convergence_Criteria, double ZNCCThreshold, int Iter_Max, int InterpAlgo, double *fufv, bool greedySearch, double *ShapePara, double *oPara, double *Timg, double *T, double *Znssd_reqd)
{
	//Also a fine ImgRef matching,.... some differences in the input as compared to TMatchingFine though
	// NOTE: initial guess is of the form of the homography

	int i, j, k, m, ii, kk, iii, jjj, iii_n, jjj_n, iii2, jjj2, ij;
	double II, JJ, a, b, gx, gy, DIC_Coeff, DIC_Coeff_min, t_1, t_2, t_3, t_4, t_5, t_6, t_f, t_ff, t_g, m_F, m_G, S[6];
	double conv_crit_1 = pow(10.0, -Convergence_Criteria - 2);
	double conv_crit_2 = conv_crit_1*0.01;
	int NN[] = { 8, 14, 6, 12 };
	int nn = NN[advanced_tech - 1], nExtraParas = advanced_tech > 2 ? 0 : 2, _iter = 0;
	int p_jump, p_jump_0 = 1, p_jump_incr = 1;
	int TimgS = 2 * hsubset + 1, Tlength = TimgS*TimgS, RefLength = widthRef*heightRef, TarLength = widthTar*heightTar;

	double 	AA[196 * 196], BB[14], CC[14], p[14], ip[14], p_best[14];
	if (ShapePara == NULL)
	{
		for (ii = 0; ii < nn; ii++)
			p[ii] = (ii == nn - nExtraParas ? 1.0 : 0.0);
	}
	else
	{
		if (advanced_tech == 1) //These are basically taylor approximation of the denumerator
		{
			p[0] = ShapePara[2] - PT.x, p[1] = ShapePara[5] - PT.y;
			p[2] = ShapePara[0] - ShapePara[2] * ShapePara[6] - 1.0;
			p[3] = ShapePara[1] - ShapePara[2] * ShapePara[7];
			p[4] = ShapePara[3] - ShapePara[5] * ShapePara[6];
			p[5] = ShapePara[4] - ShapePara[5] * ShapePara[7] - 1.0;
			p[6] = 1.0, p[7] = 0.0;
		}
		else
		{
			p[0] = ShapePara[2] - PT.x, p[1] = ShapePara[5] - PT.y;
			p[2] = ShapePara[0] - ShapePara[2] * ShapePara[6] - 1.0;
			p[3] = ShapePara[1] - ShapePara[2] * ShapePara[7];
			p[4] = ShapePara[3] - ShapePara[5] * ShapePara[6];
			p[5] = ShapePara[4] - ShapePara[5] * ShapePara[7] - 1.0;
			p[6] = -0.5*ShapePara[0] * ShapePara[6];
			p[7] = -0.5*ShapePara[1] * ShapePara[7];
			p[8] = -(ShapePara[0] * ShapePara[7] + ShapePara[1] * ShapePara[6]);
			p[9] = -0.5*ShapePara[3] * ShapePara[6];
			p[10] = -0.5*ShapePara[4] * ShapePara[7];
			p[11] = -(ShapePara[3] * ShapePara[7] + ShapePara[4] * ShapePara[6]);
			p[12] = 1.0, p[13] = 0.0;
		}
	}
	for (i = 0; i < nn; i++)
		ip[i] = p[i];

	bool createMem = false;
	if (Timg == NULL)
	{
		Timg = new double[Tlength*nchannels];
		T = new double[2 * Tlength*nchannels];
		Znssd_reqd = new double[9 * Tlength];
		createMem = true;
	}

	for (jjj = -hsubset; jjj <= hsubset; jjj++)
	{
		for (iii = -hsubset; iii <= hsubset; iii++)
		{
			II = PR.x + iii, JJ = PR.y + jjj;
			for (kk = 0; kk < nchannels; kk++)
			{
				Get_Value_Spline(RefPara + kk*RefLength, widthRef, heightRef, II, JJ, S, -1, InterpAlgo);
				Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength] = S[0];
			}
		}
	}

	bool printout = false; FILE *fp = 0;
	if (printout)
	{
		fp = fopen("C:/temp/src.txt", "w+");
		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
				for (kk = 0; kk < nchannels; kk++)
					fprintf(fp, "%.2f ", Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	if (greedySearch)
	{
		/// Let's start with only translation and only match the at the highest level of the pyramid
		double zncc, znccMin;
		for (p_jump = p_jump_0; p_jump > 0; p_jump -= (advanced_tech == 0 ? 1 : 2))
		{
			znccMin = 1e10;
			for (k = 0; k < Iter_Max; k++)
			{
				t_1 = 0.0;
				t_2 = 0.0;
				for (i = 0; i < 4; i++)
					AA[i] = 0.0;
				for (i = 0; i < 2; i++)
					BB[i] = 0.0;

				for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
				{
					for (iii = -hsubset; iii <= hsubset; iii += p_jump)
					{
						II = PT.x + iii + p[0], JJ = PT.y + jjj + p[1];
						if (II<0.0 || II>(double)(widthTar - 2) || JJ<0.0 || JJ>(double)(heightTar - 2))
							continue;

						for (kk = 0; kk < nchannels; kk++)
						{
							Get_Value_Spline(TarPara + kk*TarLength, widthTar, heightTar, II, JJ, S, 0, InterpAlgo);

							m_F = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength];
							m_G = S[0];

							t_3 = m_G - m_F;
							CC[0] = S[1], CC[1] = S[2];

							for (i = 0; i < 2; i++)
								BB[i] += t_3*CC[i];

							for (j = 0; j < 2; j++)
								for (i = 0; i < 2; i++)
									AA[j * 2 + i] += CC[i] * CC[j];

							t_1 += t_3*t_3, t_2 += m_F*m_F;
						}
					}
				}
				zncc = t_1 / t_2;

				QR_Solution_Double(AA, BB, 2, 2);
				for (i = 0; i < 2; i++)
					p[i] -= BB[i];

				if (abs(p[0]) > 0.005*widthTar || abs(p[1]) > 0.005*widthTar || !IsFiniteNumber(p[0]))
				{
					if (createMem)
					{
						delete[]Timg;
						delete[]T;
					}
					return 0.0;
				}

				if (zncc < znccMin)	// If the iteration does not converge, this can be helpful
				{
					znccMin = zncc;
					p_best[0] = p[0], p_best[1] = p[1];
				}

				if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
					break;
			}
		}
		p[0] = p_best[0], p[1] = p_best[1];
	}

	/// DIC Iteration: Begin
	bool Break_Flag;
	DIC_Coeff_min = 1e10;
	for (p_jump = p_jump_0; p_jump > 0; p_jump -= p_jump_incr)
	{
		Break_Flag = false;
		for (k = 0; k < Iter_Max; k++)
		{
			m = -1, t_1 = 0.0, t_2 = 0.0;
			for (iii = 0; iii < nn*nn; iii++)
				AA[iii] = 0.0;
			for (iii = 0; iii < nn; iii++)
				BB[iii] = 0.0;

			a = p[nn - 2], b = p[nn - 1];
			if (printout)
				fp = fopen("C:/temp/tar.txt", "w+");

			for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
			{
				for (iii = -hsubset; iii <= hsubset; iii += p_jump)
				{
					if (advanced_tech % 2 == 1)
						II = PT.x + iii + p[0] + p[2] * iii + p[3] * jjj, JJ = PT.y + jjj + p[1] + p[4] * iii + p[5] * jjj;
					else if (advanced_tech == 0)
					{
						iii2 = iii*iii, jjj2 = jjj*jjj, ij = iii*jjj;
						II = PT.x + iii + p[0] + p[2] * iii + p[3] * jjj + p[6] * iii2*0.5 + p[7] * jjj2*0.5 + p[8] * ij;
						JJ = PT.y + jjj + p[1] + p[4] * iii + p[5] * jjj + p[9] * iii2*0.5 + p[10] * jjj2*0.5 + p[11] * ij;
					}

					if (II<5.0 || II>(double)(widthTar - 5) || JJ<5.0 || JJ>(double)(heightTar - 5))
						continue;

					for (kk = 0; kk < nchannels; kk++)
					{
						m_F = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength];
						Get_Value_Spline(TarPara + kk*TarLength, widthTar, heightTar, II, JJ, S, 0, InterpAlgo);
						m_G = S[0], gx = S[1], gy = S[2];
						m++;

						if (printout)
							fprintf(fp, "%.2f ", m_G);

						if (advanced_tech < 2)
						{
							t_3 = a*m_G + b - m_F, t_4 = a;

							t_5 = t_4*gx, t_6 = t_4*gy;
							CC[0] = t_5, CC[1] = t_6;
							CC[2] = t_5*iii, CC[3] = t_5*jjj;
							CC[4] = t_6*iii, CC[5] = t_6*jjj;
							CC[6] = m_G, CC[7] = 1.0;

							for (j = 0; j < nn; j++)
								BB[j] += t_3*CC[j];

							for (j = 0; j < nn; j++)
								for (i = 0; i < nn; i++)
									AA[j*nn + i] += CC[i] * CC[j];

							t_1 += t_3*t_3, t_2 += m_F*m_F;
						}
						else
						{
							Znssd_reqd[9 * m + 0] = m_F, Znssd_reqd[9 * m + 1] = m_G;
							Znssd_reqd[9 * m + 2] = gx, Znssd_reqd[9 * m + 3] = gy;
							Znssd_reqd[9 * m + 4] = (double)iii, Znssd_reqd[9 * m + 5] = (double)jjj;
							t_1 += m_F, t_2 += m_G;
						}
					}
				}
				if (printout)
					fprintf(fp, "\n");
			}
			if (printout)
				fclose(fp);

			if (advanced_tech < 3)
			{
				DIC_Coeff = t_1 / t_2;
				if (t_2 < 10.0e-9)
					break;
			}
			else
			{
				if (k == 0)
				{
					t_f = t_1 / (m + 1);
					t_1 = 0.0;
					for (iii = 0; iii <= m; iii++)
					{
						t_4 = Znssd_reqd[9 * iii + 0] - t_f;
						t_1 += t_4*t_4;
					}
					t_ff = sqrt(t_1);
				}
				t_g = t_2 / (m + 1);
				t_2 = 0.0;
				for (iii = 0; iii <= m; iii++)
				{
					t_5 = Znssd_reqd[9 * iii + 1] - t_g;
					t_2 += t_5*t_5;
				}
				t_2 = sqrt(t_2);

				DIC_Coeff = 0.0;
				for (iii = 0; iii <= m; iii++)
				{
					t_4 = Znssd_reqd[9 * iii + 0] - t_f;
					t_5 = Znssd_reqd[9 * iii + 1] - t_g;
					t_6 = t_5 / t_2 - t_4 / t_ff;
					t_3 = t_6 / t_2;
					gx = Znssd_reqd[9 * iii + 2], gy = Znssd_reqd[9 * iii + 3];
					iii_n = Znssd_reqd[9 * iii + 4], jjj_n = Znssd_reqd[9 * iii + 5];
					CC[0] = gx, CC[1] = gy;
					CC[2] = gx*iii_n, CC[3] = gx*jjj_n;
					CC[4] = gy*iii_n, CC[5] = gy*jjj_n;
					if (advanced_tech == 4)
					{
						CC[6] = gx*iii_n*iii_n*0.5, CC[7] = gx*jjj_n*jjj_n*0.5, CC[8] = gx*iii_n*jjj_n;
						CC[9] = gy*iii_n*iii_n*0.5, CC[10] = gy*jjj_n*jjj_n*0.5, CC[11] = gy*iii_n*jjj_n;
					}
					for (j = 0; j < nn; j++)
					{
						BB[j] += t_3*CC[j];
						for (i = 0; i < nn; i++)
							AA[j*nn + i] += CC[i] * CC[j] / (t_2*t_2);
					}

					DIC_Coeff += t_6*t_6;
				}
				if (!IsNumber(DIC_Coeff))
					return 9e9;
				if (!IsFiniteNumber(DIC_Coeff))
					return 9e9;
			}

			QR_Solution_Double(AA, BB, nn, nn);
			for (iii = 0; iii < nn; iii++)
				p[iii] -= BB[iii];

			if (abs(p[0]) > 0.005*widthTar || abs(p[1]) > 0.005*widthTar || !IsFiniteNumber(p[0]))
			{
				if (createMem)
				{
					delete[]Timg;
					delete[]T;
				}
				return 0.0;
			}

			if (DIC_Coeff < DIC_Coeff_min)	// If the iteration does not converge, this can be helpful
			{
				DIC_Coeff_min = DIC_Coeff;
				for (iii = 0; iii < nn; iii++)
					p_best[iii] = p[iii];
				if (!IsNumber(p[0]) || !IsNumber(p[1]))
					return 9e9;
			}

			if (advanced_tech < 3)
			{
				if (abs(p[0] - ip[0]) > hsubset || abs(p[1] - ip[1]) > hsubset)
					return 9e9;
				if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
				{
					for (iii = 2; iii < nn - nExtraParas; iii++)
					{
						if (fabs(BB[iii]) > conv_crit_2)
							break;
					}
					if (iii == nn - nExtraParas)
						Break_Flag = true;
				}
			}
			else if (advanced_tech == 3)
			{
				if (abs(p[0] - ip[0]) > hsubset || abs(p[1] - ip[1]) > hsubset)
					return 9e9;
				if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
				{
					for (iii = 2; iii < nn - nExtraParas; iii++)
						if (fabs(BB[iii]) > conv_crit_2)
							break;
					if (iii == nn - nExtraParas)
						Break_Flag = true;
				}
			}
			else
			{
				if (abs(p[0] - ip[0]) > hsubset || abs(p[1] - ip[1]) > hsubset)
					return 9e9;
				if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
				{
					for (iii = 2; iii < nn; iii++)
						if (fabs(BB[iii]) > conv_crit_2)
							break;
					if (iii == nn)
						Break_Flag = true;
				}
			}
			if (Break_Flag)
				break;
		}
		_iter += k;

		// In case the iteration converges to "wrong" points, always use the data that lead to the least-square value.
		for (iii = 0; iii < nn; iii++)
			p[iii] = p_best[iii];
	}
	/// DIC Iteration: End

	//Now, dont really trust the pssad error too much, compute zncc score instead! 
	//They are usually close when the convergence goes smothly, but in case of trouble, zncc is more reliable.
	double ZNCC;
	if (advanced_tech < 3)
	{
		int m = 0;
		double t_1, t_2, t_3, t_4, t_5, t_f = 0.0, t_g = 0.0;
		if (printout)
			fp = fopen("C:/temp/tar.txt", "w+");
		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
			{
				II = PT.x + iii + p[0] + p[2] * iii + p[3] * jjj;
				JJ = PT.y + jjj + p[1] + p[4] * iii + p[5] * jjj;

				if (II<0.0 || II>(double)(widthTar - 1) || JJ<0.0 || JJ>(double)(heightTar - 1))
					continue;

				for (kk = 0; kk < nchannels; kk++)
				{
					Get_Value_Spline(TarPara + kk*TarLength, widthTar, heightTar, II, JJ, S, -1, InterpAlgo);
					if (printout)
						fprintf(fp, "%.4f ", S[0]);

					T[2 * m] = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength];
					T[2 * m + 1] = S[0];
					t_f += T[2 * m];
					t_g += T[2 * m + 1];
					m++;
				}
			}
			if (printout)
				fprintf(fp, "\n");
		}
		if (printout)
			fclose(fp);

		t_f = t_f / m;
		t_g = t_g / m;
		t_1 = 0.0, t_2 = 0.0, t_3 = 0.0;
		for (i = 0; i < m; i++)
		{
			t_4 = T[2 * i] - t_f;
			t_5 = T[2 * i + 1] - t_g;
			t_1 += 1.0*t_4*t_5;
			t_2 += 1.0*t_4*t_4;
			t_3 += 1.0*t_5*t_5;
		}

		t_2 = sqrt(t_2*t_3);
		if (t_2 < 1e-10)
			t_2 = 1e-10;

		ZNCC = t_1 / t_2; //This is the zncc score
		if (abs(ZNCC) > 1.0)
			ZNCC = 0.0;
	}
	else
		ZNCC = 1.0 - 0.5*DIC_Coeff_min; //from ZNSSD to ZNCC

	if (abs(p[0]) > 0.005*widthTar || abs(p[1]) > 0.005*widthTar || p[0] != p[0] || p[1] != p[1] || ZNCC < ZNCCThreshold)
	{
		if (createMem)
		{
			delete[]Timg;
			delete[]T;
		}
		return 0.0;
	}

	fufv[0] = p_best[0], fufv[1] = p_best[1];
	if (oPara != NULL)
		for (ii = 0; ii < 8; ii++)
			oPara[ii] = p_best[ii];

	if (createMem)
	{
		delete[]Timg;
		delete[]T;
	}
	return ZNCC;
}
double TrackingByLK(float *RefPara, float *TarPara, int hsubset, int widthRef, int heightRef, int widthTar, int heightTar, int nchannels, Point2d PR, Point2d PT, int advanced_tech, int Convergence_Criteria, double ZNCCThreshold, int Iter_Max, int InterpAlgo, double *fufv, bool greedySearch, double *ShapePara, double *oPara, double *Timg, double *T, double *Znssd_reqd)
{
	//Also a fine ImgRef matching,.... some differences in the input as compared to TMatchingFine though
	// NOTE: initial guess is of the form of the homography

	int i, j, k, m, ii, kk, iii, jjj, iii_n, jjj_n, iii2, jjj2, ij;
	double II, JJ, a, b, gx, gy, DIC_Coeff, DIC_Coeff_min, t_1, t_2, t_3, t_4, t_5, t_6, t_f, t_ff, t_g, m_F, m_G, S[6];
	double conv_crit_1 = pow(10.0, -Convergence_Criteria - 2);
	double conv_crit_2 = conv_crit_1*0.01;
	int NN[] = { 8, 14, 6, 12 };
	int nn = NN[advanced_tech - 1], nExtraParas = advanced_tech > 2 ? 0 : 2, _iter = 0;
	int p_jump, p_jump_0 = 1, p_jump_incr = 1;
	int TimgS = 2 * hsubset + 1, Tlength = TimgS*TimgS, RefLength = widthRef*heightRef, TarLength = widthTar*heightTar;

	double 	AA[196 * 196], BB[14], CC[14], p[14], ip[14], p_best[14];
	if (ShapePara == NULL)
	{
		for (ii = 0; ii < nn; ii++)
			p[ii] = (ii == nn - nExtraParas ? 1.0 : 0.0);
	}
	else
	{
		if (advanced_tech == 1) //These are basically taylor approximation of the denumerator
		{
			p[0] = ShapePara[2] - PT.x, p[1] = ShapePara[5] - PT.y;
			p[2] = ShapePara[0] - ShapePara[2] * ShapePara[6] - 1.0;
			p[3] = ShapePara[1] - ShapePara[2] * ShapePara[7];
			p[4] = ShapePara[3] - ShapePara[5] * ShapePara[6];
			p[5] = ShapePara[4] - ShapePara[5] * ShapePara[7] - 1.0;
			p[6] = 1.0, p[7] = 0.0;
		}
		else
		{
			p[0] = ShapePara[2] - PT.x, p[1] = ShapePara[5] - PT.y;
			p[2] = ShapePara[0] - ShapePara[2] * ShapePara[6] - 1.0;
			p[3] = ShapePara[1] - ShapePara[2] * ShapePara[7];
			p[4] = ShapePara[3] - ShapePara[5] * ShapePara[6];
			p[5] = ShapePara[4] - ShapePara[5] * ShapePara[7] - 1.0;
			p[6] = -0.5*ShapePara[0] * ShapePara[6];
			p[7] = -0.5*ShapePara[1] * ShapePara[7];
			p[8] = -(ShapePara[0] * ShapePara[7] + ShapePara[1] * ShapePara[6]);
			p[9] = -0.5*ShapePara[3] * ShapePara[6];
			p[10] = -0.5*ShapePara[4] * ShapePara[7];
			p[11] = -(ShapePara[3] * ShapePara[7] + ShapePara[4] * ShapePara[6]);
			p[12] = 1.0, p[13] = 0.0;
		}
	}
	for (i = 0; i < nn; i++)
		ip[i] = p[i];

	bool createMem = false;
	if (Timg == NULL)
	{
		Timg = new double[Tlength*nchannels];
		T = new double[2 * Tlength*nchannels];
		Znssd_reqd = new double[9 * Tlength];
		createMem = true;
	}

	for (jjj = -hsubset; jjj <= hsubset; jjj++)
	{
		for (iii = -hsubset; iii <= hsubset; iii++)
		{
			II = PR.x + iii, JJ = PR.y + jjj;
			for (kk = 0; kk < nchannels; kk++)
			{
				Get_Value_Spline(RefPara + kk*RefLength, widthRef, heightRef, II, JJ, S, -1, InterpAlgo);
				Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength] = S[0];
			}
		}
	}

	bool printout = false; FILE *fp = 0;
	if (printout)
	{
		fp = fopen("C:/temp/src.txt", "w+");
		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
				for (kk = 0; kk < nchannels; kk++)
					fprintf(fp, "%.2f ", Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	if (greedySearch)
	{
		/// Let's start with only translation and only match the at the highest level of the pyramid
		double zncc, znccMin;
		for (p_jump = p_jump_0; p_jump > 0; p_jump -= (advanced_tech == 0 ? 1 : 2))
		{
			znccMin = 1e10;
			for (k = 0; k < Iter_Max; k++)
			{
				t_1 = 0.0;
				t_2 = 0.0;
				for (i = 0; i < 4; i++)
					AA[i] = 0.0;
				for (i = 0; i < 2; i++)
					BB[i] = 0.0;

				for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
				{
					for (iii = -hsubset; iii <= hsubset; iii += p_jump)
					{
						II = PT.x + iii + p[0], JJ = PT.y + jjj + p[1];
						if (II<0.0 || II>(double)(widthTar - 2) || JJ<0.0 || JJ>(double)(heightTar - 2))
							continue;

						for (kk = 0; kk < nchannels; kk++)
						{
							Get_Value_Spline(TarPara + kk*TarLength, widthTar, heightTar, II, JJ, S, 0, InterpAlgo);

							m_F = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength];
							m_G = S[0];

							t_3 = m_G - m_F;
							CC[0] = S[1], CC[1] = S[2];

							for (i = 0; i < 2; i++)
								BB[i] += t_3*CC[i];

							for (j = 0; j < 2; j++)
								for (i = 0; i < 2; i++)
									AA[j * 2 + i] += CC[i] * CC[j];

							t_1 += t_3*t_3, t_2 += m_F*m_F;
						}
					}
				}
				zncc = t_1 / t_2;

				QR_Solution_Double(AA, BB, 2, 2);
				for (i = 0; i < 2; i++)
					p[i] -= BB[i];

				if (abs(p[0]) > 0.005*widthTar || abs(p[1]) > 0.005*widthTar || !IsFiniteNumber(p[0]))
				{
					if (createMem)
					{
						delete[]Timg;
						delete[]T;
					}
					return 0.0;
				}

				if (zncc < znccMin)	// If the iteration does not converge, this can be helpful
				{
					znccMin = zncc;
					p_best[0] = p[0], p_best[1] = p[1];
				}

				if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
					break;
			}
		}
		p[0] = p_best[0], p[1] = p_best[1];
	}

	/// DIC Iteration: Begin
	bool Break_Flag;
	DIC_Coeff_min = 1e10;
	for (p_jump = p_jump_0; p_jump > 0; p_jump -= p_jump_incr)
	{
		Break_Flag = false;
		for (k = 0; k < Iter_Max; k++)
		{
			m = -1, t_1 = 0.0, t_2 = 0.0;
			for (iii = 0; iii < nn*nn; iii++)
				AA[iii] = 0.0;
			for (iii = 0; iii < nn; iii++)
				BB[iii] = 0.0;

			a = p[nn - 2], b = p[nn - 1];
			if (printout)
				fp = fopen("C:/temp/tar.txt", "w+");

			for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
			{
				for (iii = -hsubset; iii <= hsubset; iii += p_jump)
				{
					if (advanced_tech % 2 == 1)
						II = PT.x + iii + p[0] + p[2] * iii + p[3] * jjj, JJ = PT.y + jjj + p[1] + p[4] * iii + p[5] * jjj;
					else if (advanced_tech == 0)
					{
						iii2 = iii*iii, jjj2 = jjj*jjj, ij = iii*jjj;
						II = PT.x + iii + p[0] + p[2] * iii + p[3] * jjj + p[6] * iii2*0.5 + p[7] * jjj2*0.5 + p[8] * ij;
						JJ = PT.y + jjj + p[1] + p[4] * iii + p[5] * jjj + p[9] * iii2*0.5 + p[10] * jjj2*0.5 + p[11] * ij;
					}

					if (II<5.0 || II>(double)(widthTar - 5) || JJ<5.0 || JJ>(double)(heightTar - 5))
						continue;

					for (kk = 0; kk < nchannels; kk++)
					{
						m_F = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength];
						Get_Value_Spline(TarPara + kk*TarLength, widthTar, heightTar, II, JJ, S, 0, InterpAlgo);
						m_G = S[0], gx = S[1], gy = S[2];
						m++;

						if (printout)
							fprintf(fp, "%.2f ", m_G);

						if (advanced_tech < 2)
						{
							t_3 = a*m_G + b - m_F, t_4 = a;

							t_5 = t_4*gx, t_6 = t_4*gy;
							CC[0] = t_5, CC[1] = t_6;
							CC[2] = t_5*iii, CC[3] = t_5*jjj;
							CC[4] = t_6*iii, CC[5] = t_6*jjj;
							CC[6] = m_G, CC[7] = 1.0;

							for (j = 0; j < nn; j++)
								BB[j] += t_3*CC[j];

							for (j = 0; j < nn; j++)
								for (i = 0; i < nn; i++)
									AA[j*nn + i] += CC[i] * CC[j];

							t_1 += t_3*t_3, t_2 += m_F*m_F;
						}
						else
						{
							Znssd_reqd[9 * m + 0] = m_F, Znssd_reqd[9 * m + 1] = m_G;
							Znssd_reqd[9 * m + 2] = gx, Znssd_reqd[9 * m + 3] = gy;
							Znssd_reqd[9 * m + 4] = (double)iii, Znssd_reqd[9 * m + 5] = (double)jjj;
							t_1 += m_F, t_2 += m_G;
						}
					}
				}
				if (printout)
					fprintf(fp, "\n");
			}
			if (printout)
				fclose(fp);

			if (advanced_tech < 3)
			{
				DIC_Coeff = t_1 / t_2;
				if (t_2 < 10.0e-9)
					break;
			}
			else
			{
				if (k == 0)
				{
					t_f = t_1 / (m + 1);
					t_1 = 0.0;
					for (iii = 0; iii <= m; iii++)
					{
						t_4 = Znssd_reqd[9 * iii + 0] - t_f;
						t_1 += t_4*t_4;
					}
					t_ff = sqrt(t_1);
				}
				t_g = t_2 / (m + 1);
				t_2 = 0.0;
				for (iii = 0; iii <= m; iii++)
				{
					t_5 = Znssd_reqd[9 * iii + 1] - t_g;
					t_2 += t_5*t_5;
				}
				t_2 = sqrt(t_2);

				DIC_Coeff = 0.0;
				for (iii = 0; iii <= m; iii++)
				{
					t_4 = Znssd_reqd[9 * iii + 0] - t_f;
					t_5 = Znssd_reqd[9 * iii + 1] - t_g;
					t_6 = t_5 / t_2 - t_4 / t_ff;
					t_3 = t_6 / t_2;
					gx = Znssd_reqd[9 * iii + 2], gy = Znssd_reqd[9 * iii + 3];
					iii_n = Znssd_reqd[9 * iii + 4], jjj_n = Znssd_reqd[9 * iii + 5];
					CC[0] = gx, CC[1] = gy;
					CC[2] = gx*iii_n, CC[3] = gx*jjj_n;
					CC[4] = gy*iii_n, CC[5] = gy*jjj_n;
					if (advanced_tech == 4)
					{
						CC[6] = gx*iii_n*iii_n*0.5, CC[7] = gx*jjj_n*jjj_n*0.5, CC[8] = gx*iii_n*jjj_n;
						CC[9] = gy*iii_n*iii_n*0.5, CC[10] = gy*jjj_n*jjj_n*0.5, CC[11] = gy*iii_n*jjj_n;
					}
					for (j = 0; j < nn; j++)
					{
						BB[j] += t_3*CC[j];
						for (i = 0; i < nn; i++)
							AA[j*nn + i] += CC[i] * CC[j] / (t_2*t_2);
					}

					DIC_Coeff += t_6*t_6;
				}
				if (!IsNumber(DIC_Coeff))
					return 9e9;
				if (!IsFiniteNumber(DIC_Coeff))
					return 9e9;
			}

			QR_Solution_Double(AA, BB, nn, nn);
			for (iii = 0; iii < nn; iii++)
				p[iii] -= BB[iii];

			if (abs(p[0]) > 0.005*widthTar || abs(p[1]) > 0.005*widthTar || !IsFiniteNumber(p[0]))
			{
				if (createMem)
				{
					delete[]Timg;
					delete[]T;
				}
				return 0.0;
			}

			if (DIC_Coeff < DIC_Coeff_min)	// If the iteration does not converge, this can be helpful
			{
				DIC_Coeff_min = DIC_Coeff;
				for (iii = 0; iii < nn; iii++)
					p_best[iii] = p[iii];
				if (!IsNumber(p[0]) || !IsNumber(p[1]))
					return 0.0;
			}

			if (advanced_tech < 3)
			{
				if (abs(p[0] - ip[0]) > hsubset || abs(p[1] - ip[1]) > hsubset)
					return 0.0;
				if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
				{
					for (iii = 2; iii < nn - nExtraParas; iii++)
					{
						if (fabs(BB[iii]) > conv_crit_2)
							break;
					}
					if (iii == nn - nExtraParas)
						Break_Flag = true;
				}
			}
			else if (advanced_tech == 3)
			{
				if (abs(p[0] - ip[0]) > hsubset || abs(p[1] - ip[1]) > hsubset)
					return 0.0;
				if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
				{
					for (iii = 2; iii < nn - nExtraParas; iii++)
						if (fabs(BB[iii]) > conv_crit_2)
							break;
					if (iii == nn - nExtraParas)
						Break_Flag = true;
				}
			}
			else
			{
				if (abs(p[0] - ip[0]) > hsubset || abs(p[1] - ip[1]) > hsubset)
					return 0.0;
				if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
				{
					for (iii = 2; iii < nn; iii++)
						if (fabs(BB[iii]) > conv_crit_2)
							break;
					if (iii == nn)
						Break_Flag = true;
				}
			}
			if (Break_Flag)
				break;
		}
		_iter += k;

		// In case the iteration converges to "wrong" points, always use the data that lead to the least-square value.
		for (iii = 0; iii < nn; iii++)
			p[iii] = p_best[iii];
	}
	/// DIC Iteration: End

	//Now, dont really trust the pssad error too much, compute zncc score instead! 
	//They are usually close when the convergence goes smothly, but in case of trouble, zncc is more reliable.
	double ZNCC;
	if (advanced_tech < 3)
	{
		int m = 0;
		double t_1, t_2, t_3, t_4, t_5, t_f = 0.0, t_g = 0.0;
		if (printout)
			fp = fopen("C:/temp/tar.txt", "w+");
		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
			{
				II = PT.x + iii + p[0] + p[2] * iii + p[3] * jjj;
				JJ = PT.y + jjj + p[1] + p[4] * iii + p[5] * jjj;

				if (II<0.0 || II>(double)(widthTar - 1) || JJ<0.0 || JJ>(double)(heightTar - 1))
					continue;

				for (kk = 0; kk < nchannels; kk++)
				{
					Get_Value_Spline(TarPara + kk*TarLength, widthTar, heightTar, II, JJ, S, -1, InterpAlgo);
					if (printout)
						fprintf(fp, "%.4f ", S[0]);

					T[2 * m] = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength];
					T[2 * m + 1] = S[0];
					t_f += T[2 * m];
					t_g += T[2 * m + 1];
					m++;
				}
			}
			if (printout)
				fprintf(fp, "\n");
		}
		if (printout)
			fclose(fp);

		t_f = t_f / m;
		t_g = t_g / m;
		t_1 = 0.0, t_2 = 0.0, t_3 = 0.0;
		for (i = 0; i < m; i++)
		{
			t_4 = T[2 * i] - t_f;
			t_5 = T[2 * i + 1] - t_g;
			t_1 += 1.0*t_4*t_5;
			t_2 += 1.0*t_4*t_4;
			t_3 += 1.0*t_5*t_5;
		}

		t_2 = sqrt(t_2*t_3);
		if (t_2 < 1e-10)
			t_2 = 1e-10;

		ZNCC = t_1 / t_2; //This is the zncc score
		if (abs(ZNCC) > 1.0)
			ZNCC = 0.0;
	}
	else
		ZNCC = 1.0 - 0.5*DIC_Coeff_min; //from ZNSSD to ZNCC

	if (abs(p[0]) > 0.005*widthTar || abs(p[1]) > 0.005*widthTar || p[0] != p[0] || p[1] != p[1] || ZNCC < ZNCCThreshold)
	{
		if (createMem)
		{
			delete[]Timg;
			delete[]T;
		}
		return 0.0;
	}

	fufv[0] = p_best[0], fufv[1] = p_best[1];
	if (oPara != NULL)
		for (ii = 0; ii < 8; ii++)
			oPara[ii] = p_best[ii];

	if (createMem)
	{
		delete[]Timg;
		delete[]T;
	}
	return ZNCC;
}
int TrackOpenCVLK(char *Path, int startFrame, int stopFrame)
{
	Mat colorImg, gray, prevGray, tImg, backGround, bestFrameInWind;
	vector<Point2f> points[2];
	vector<uchar> status;
	vector<float> err;

	char Fname[200];
	cvNamedWindow("Static Image detection with LK", WINDOW_NORMAL);

	bool needToInit = true;
	int MAX_COUNT = 5000, frameID = 0;

	TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
	Size subPixWinSize(21, 21), winSize(31, 31);

	for (int frameID = startFrame; frameID <= stopFrame; frameID++)
	{
		sprintf(Fname, "%s/%d.png", Path, frameID); colorImg = imread(Fname, 1);
		if (colorImg.empty())
			break;

		cvtColor(colorImg, gray, CV_BGR2GRAY);
		cvtColor(gray, backGround, CV_GRAY2BGR);

		if (frameID == startFrame) // automatic initialization
		{
			goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.05, 50, Mat(), 21, 0, 0.04);
			cornerSubPix(gray, points[1], subPixWinSize, Size(-1, -1), termcrit);
			for (int jj = 0; jj < points[1].size(); jj++)
				circle(backGround, points[1][jj], 5, Scalar(83, 185, 255), -1, 8);

			sprintf(Fname, "%s/%d.txt", Path, frameID);	FILE *fp = fopen(Fname, "w+");
			for (int jj = 0; jj < points[1].size(); jj++)
				fprintf(fp, "%.3f %.3f\n", points[1][jj].x, points[1][jj].y);
			fclose(fp);
		}

		if (!points[0].empty())
		{
			status.clear(); err.clear();
			if (prevGray.empty())
				gray.copyTo(prevGray);
			calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize, 3, termcrit, 0, 0.001);

			size_t i, k;
			int succeded = 0;
			for (i = k = 0; i < points[1].size(); i++)
			{
				if (!status[i])
					continue;
				succeded++;
				points[1][k++] = points[1][i];
				circle(backGround, points[1][i], 5, Scalar(83, 185, 255), -1, 8);
			}
			points[1].resize(k);

			sprintf(Fname, "%s/%d.txt", Path, frameID);	FILE *fp = fopen(Fname, "w+");
			for (int jj = 0; jj < points[1].size(); jj++)
				fprintf(fp, "%.3f %.3f\n", points[1][jj].x, points[1][jj].y);
			fclose(fp);

			if (succeded < 0.5*points[1].size())
			{
				printf("Redected @ frame %d due to low # of features", frameID);
				goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
				cornerSubPix(gray, points[1], subPixWinSize, Size(-1, -1), termcrit);
			}
		}

		needToInit = false;
		imshow("Static Image detection with LK", backGround);
		char c = (char)waitKey(10);
		if (c == 27)
			break;

		sprintf(Fname, "%s/_%d.png", Path, frameID);
		imwrite(Fname, backGround);

		std::swap(points[1], points[0]);
		swap(prevGray, gray);
	}

	return 0;
}

void TransformImage(double *oImg, int Owidth, int Oheight, double *iImg, int Iwidth, int Iheight, double *Trans, int nchannels, int interpAlgo, double *iPara)
{
	//Trans if of a 3x3 matrix with Trans[8] = 1
	int ii, jj, kk, Ilength = Iwidth*Iheight, Olength = Owidth*Oheight;

	bool createMem = false;
	if (iPara == NULL)
	{
		createMem = true;
		iPara = new double[Ilength*nchannels];
	}

	for (ii = 0; ii < nchannels; ii++)
		Generate_Para_Spline(iImg + ii*Ilength, iPara + ii*Ilength, Iwidth, Iheight, interpAlgo);

	double u, v, denum, val[3];
	for (jj = 0; jj < Oheight; jj++)
	{
		for (ii = 0; ii < Owidth; ii++)
		{
			denum = Trans[6] * ii + Trans[7] * jj + 1.0;
			u = (Trans[0] * ii + Trans[1] * jj + Trans[2]) / denum;
			v = (Trans[3] * ii + Trans[4] * jj + Trans[5]) / denum;
			if (u<0 || u>Iwidth - 1 || v<0 || v>Iheight - 1)
			{
				for (kk = 0; kk < nchannels; kk++)
					oImg[ii + jj*Owidth + kk*Olength] = 0.0;
				continue;
			}

			for (kk = 0; kk < nchannels; kk++)
			{
				Get_Value_Spline(iPara + kk*Ilength, Iwidth, Iheight, u, v, val, -1, interpAlgo);
				val[0] = min(max(val[0], 0.0), 255.0);
				oImg[ii + jj*Owidth + kk*Olength] = val[0];
			}
		}
	}

	if (createMem)
		delete[]iPara;

	return;
}
void DetectCornersCorrelation(double *img, int width, int height, int nchannels, Point2d *Checker, int &npts, vector<double> PatternAngles, int hsubset, int search_area, double thresh)
{
	int i, j, ii, jj, kk, jump = 2, nMaxCorners = npts, numPatterns = PatternAngles.size();

	int bi_graylevel[2] = { 0, 255 }, PatternSize = 48, PatternLength = PatternSize*PatternSize; //Note that the pattern size is deliberately make bigger than the subset because small size give very blurry checkercorner
	double *maskSmooth = new double[PatternLength*numPatterns];

	synthesize_square_mask(maskSmooth, bi_graylevel, PatternSize, 1.0, 0, false);
	double trans[9], temp[9], iH1[9], H1[9] = { 1, 0, -PatternSize / 2, 0, 1, -PatternSize / 2, 0, 0, 1 };
	for (ii = 1; ii < PatternAngles.size(); ii++)
	{
		double c = cos(PatternAngles[ii] * 3.14159265359 / 180), s = sin(PatternAngles[ii] * 3.14159265359 / 180);
		double H2[9] = { c, -s, 0, s, c, 0, 0, 0, 1 };
		mat_invert(H1, iH1, 3);
		mat_mul(H2, H1, temp, 3, 3, 3);
		mat_mul(iH1, temp, trans, 3, 3, 3);
		TransformImage(maskSmooth + ii*PatternLength, PatternSize, PatternSize, maskSmooth, PatternSize, PatternSize, trans, 1, 1, NULL);
		//char Fname[200];  sprintf(Fname, "C:/temp/rS_%d.png", ii);
		//SaveDataToImage(Fname, maskSmooth + ii*PatternLength, PatternSize, PatternSize, 1);
	}

	double *Cornerness = new double[width*height];
	for (ii = 0; ii < width*height; ii++)
		Cornerness[ii] = 0.0;

	double zncc;
	Point2i POI;
	double *T = new double[2 * (2 * hsubset + 1)*(2 * hsubset + 1)*nchannels];
	for (jj = hsubset + search_area + 1; jj < height - hsubset - search_area - 1; jj += jump)
	{
		for (ii = hsubset + search_area + 1; ii < width - hsubset - search_area - 1; ii += jump)
		{
			for (kk = 0; kk < numPatterns; kk++)
			{
				POI.x = ii, POI.y = jj;
				zncc = abs(TMatchingSuperCoarse(maskSmooth + kk*PatternLength, PatternSize, hsubset, img, width, height, nchannels, POI, search_area, thresh, T));
				Cornerness[ii + jj*width] = max(zncc, Cornerness[ii + jj*width]);
			}
		}
	}

	double *Cornerness2 = new double[width*height];
	for (ii = 0; ii < width*height; ii++)
		Cornerness2[ii] = Cornerness[ii];
	//WriteGridBinary("C:/temp/cornerness.dat", Cornerness, width, height);

	//Non-max suppression
	bool breakflag;
	for (jj = hsubset + search_area + 1; jj < height - hsubset - search_area - 1; jj += jump)
	{
		for (ii = hsubset + search_area + 1; ii < width - hsubset - search_area - 1; ii += jump)
		{
			breakflag = false;
			if (Cornerness[ii + jj*width] < thresh)
			{
				Cornerness[ii + jj*width] = 0.0;
				Cornerness2[ii + jj*width] = 0.0;
			}
			else
			{
				for (j = -jump; j <= jump; j += jump)
				{
					for (i = -jump; i <= jump; i += jump)
					{
						if (Cornerness[ii + jj*width] < Cornerness[ii + i + (jj + j)*width] - 0.001) //avoid comparing with itself
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

	npts = 0;
	for (jj = hsubset + search_area + 1; jj < height - hsubset - search_area - 1; jj += jump)
	{
		for (ii = hsubset + search_area + 1; ii < width - hsubset - search_area - 1; ii += jump)
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
void RefineCorners(double *Para, int width, int height, int nchannels, Point2d *Checker, Point2d *Fcorners, int *FStype, int &npts, vector<double>PatternAngles, int hsubset1, int hsubset2, int searchArea, double ZNCCCoarseThresh, double ZNCCthresh, int InterpAlgo)
{
	int ii, jj, kk, boundary = hsubset2 + 2;
	int numPatterns = PatternAngles.size();
	int bi_graylevel[2] = { 0, 255 }, PatternSize = 48, PatternLength = PatternSize*PatternSize; //Note that the pattern size is deliberately make bigger than the hsubset because small size give very blurry checkercorner
	double *maskSmooth = new double[PatternLength*numPatterns * 2];

	synthesize_square_mask(maskSmooth, bi_graylevel, PatternSize, 1.0, 0, false);
	synthesize_square_mask(maskSmooth + PatternLength, bi_graylevel, PatternSize, 1.0, 1, false);

	double trans[9], temp[9], iH1[9], H1[9] = { 1, 0, -PatternSize / 2, 0, 1, -PatternSize / 2, 0, 0, 1 };
	for (ii = 1; ii < PatternAngles.size(); ii++)
	{
		double c = cos(PatternAngles[ii] * 3.14159265359 / 180), s = sin(PatternAngles[ii] * 3.14159265359 / 180);
		double H2[9] = { c, -s, 0, s, c, 0, 0, 0, 1 };
		mat_invert(H1, iH1, 3), mat_mul(H2, H1, temp, 3, 3, 3), mat_mul(iH1, temp, trans, 3, 3, 3);
		TransformImage(maskSmooth + 2 * ii*PatternLength, PatternSize, PatternSize, maskSmooth, PatternSize, PatternSize, trans, 1, 1, NULL);
		TransformImage(maskSmooth + (2 * ii + 1)*PatternLength, PatternSize, PatternSize, maskSmooth + PatternLength, PatternSize, PatternSize, trans, 1, 1, NULL);
	}
	/*	FILE *fp = fopen("C:/temp/coarse.txt", "w+");
	for(ii=0; ii<npts; ii++)
	fprintf(fp, "%.2f %.2f \n", Checker[ii].x, Checker[ii].y);
	fclose(fp);*/

	//Detect coarse corners:
	int *goodCandiates = new int[npts];
	Point2d *goodCorners = new Point2d[npts];
	int count = 0, ngoodCandiates = 0, squaretype;

	int percent = 10, increP = 10;
	double start = omp_get_wtime(), elapsed;
	//#pragma omp critical
	//cout << "Coarse refinement ..." << endl;

	double zncc, bestzncc;
	for (ii = 0; ii < npts; ii++)
	{
		if ((Checker[ii].x < boundary) || (Checker[ii].y < boundary) || (Checker[ii].x > 1.0*width - boundary) || (Checker[ii].y > 1.0*height - boundary))
			continue;

		zncc = 0.0, bestzncc = 0.0;
		for (jj = 0; jj<numPatterns; jj++)
		{
			squaretype = TMatchingCoarse(maskSmooth + 2 * jj*PatternLength, PatternSize, hsubset1, Para, width, height, nchannels, Checker[ii], searchArea, ZNCCCoarseThresh, zncc, InterpAlgo);
			if (squaretype >-1 && zncc > bestzncc)
			{
				goodCorners[count].x = Checker[ii].x;
				goodCorners[count].y = Checker[ii].y;
				goodCandiates[count] = squaretype + 2 * jj;
				bestzncc = zncc;
			}
		}
		if (bestzncc > ZNCCCoarseThresh)
			count++;
	}
	ngoodCandiates = count;
	elapsed = omp_get_wtime() - start;

	/*FILE *fp = fopen("C:/temp/coarseR.txt", "w+");
	for (ii = 0; ii < ngoodCandiates; ii++)
	fprintf(fp, "%.2f %.2f %d\n", goodCorners[ii].x, goodCorners[ii].y, goodCandiates[ii]);
	fclose(fp);*/

	//Merege coarsely detected candidates:
	npts = ngoodCandiates;
	int STACK[30]; //Maximum KNN
	int *squareType = new int[npts];
	Point2d *mergeCorners = new Point2d[npts];
	int *marker = new int[2 * npts];
	for (jj = 0; jj < 2 * npts; jj++)
		marker[jj] = -1;

	int flag, KNN;
	double t1, t2, megre_thresh = 5.0;
	count = 0, ngoodCandiates = 0;
	for (jj = 0; jj < npts; jj++)
	{
		KNN = 0;
		flag = 0;
		for (ii = 0; ii < count; ii++)
		{
			if (marker[ii] == jj)
			{
				flag = 1;
				break;
			}
		}
		if (flag == 1)
			continue;

		for (ii = jj + 1; ii < npts; ii++)
		{
			t1 = goodCorners[ii].x - goodCorners[jj].x;
			t2 = goodCorners[ii].y - goodCorners[jj].y;

			if (t1*t1 + t2*t2 < megre_thresh*megre_thresh &&goodCandiates[ii] == goodCandiates[jj])
			{
				STACK[KNN] = ii;
				KNN++;
			}
		}
		STACK[KNN] = jj;// include itself

		for (kk = 0; kk < KNN + 1; kk++)
		{
			marker[count] = STACK[kk];
			count++;
		}

		mergeCorners[ngoodCandiates].x = 0.0, mergeCorners[ngoodCandiates].y = 0.0;
		for (kk = 0; kk <= KNN; kk++)
		{
			mergeCorners[ngoodCandiates].x += goodCorners[STACK[kk]].x;
			mergeCorners[ngoodCandiates].y += goodCorners[STACK[kk]].y;
		}
		mergeCorners[ngoodCandiates].x /= (KNN + 1);
		mergeCorners[ngoodCandiates].y /= (KNN + 1);
		squareType[ngoodCandiates] = goodCandiates[jj];
		ngoodCandiates++;
	}

	/*fp = fopen("c:/temp/coarseRM.txt", "w+");
	for (ii = 0; ii < ngoodCandiates; ii++)
	fprintf(fp, "%lf %lf %d\n", mergeCorners[ii].x, mergeCorners[ii].y, squareType[ii]);
	fclose(fp);*/

	//Refine corners:
	int advanced_tech = 3; // affine only
	count = 0;
	double *Znssd_reqd = new double[9 * PatternLength];

	percent = 10;
	start = omp_get_wtime();
	for (ii = 0; ii < ngoodCandiates; ii++)
	{
		if ((mergeCorners[ii].x < boundary) || (mergeCorners[ii].y < boundary) || (mergeCorners[ii].x > 1.0*width - boundary) || (mergeCorners[ii].y > 1.0*height - boundary))
			continue;

		zncc = TMatchingFine_ZNCC(maskSmooth + squareType[ii] * PatternLength, PatternSize, hsubset2, Para, width, height, 1, mergeCorners[ii], advanced_tech, 1, ZNCCthresh, InterpAlgo, Znssd_reqd);
		if (zncc > ZNCCthresh)
		{
			squareType[ii] = squareType[ii];
			count++;
		}
		else
			squareType[ii] = -1;
	}
	delete[]Znssd_reqd;
	elapsed = omp_get_wtime() - start;

	//Final merging:
	count = 0;
	for (ii = 0; ii < ngoodCandiates; ii++)
	{
		if (squareType[ii] != -1)
		{
			goodCorners[count].x = mergeCorners[ii].x;
			goodCorners[count].y = mergeCorners[ii].y;
			goodCandiates[count] = squareType[ii];
			count++;
		}
	}

	npts = count;
	for (jj = 0; jj < npts; jj++)
		marker[jj] = -1;

	megre_thresh = 4.0, count = 0, ngoodCandiates = 0;
	for (jj = 0; jj < npts; jj++)
	{
		KNN = 0, flag = 0;
		for (ii = 0; ii < count; ii++)
		{
			if (marker[ii] == jj)
			{
				flag = 1;
				break;
			}
		}
		if (flag == 1)
			continue;

		for (ii = jj + 1; ii < npts; ii++)
		{
			t1 = goodCorners[ii].x - goodCorners[jj].x;
			t2 = goodCorners[ii].y - goodCorners[jj].y;
			if (t1*t1 + t2*t2 < megre_thresh*megre_thresh)
			{
				STACK[KNN] = ii;
			}
		}
		STACK[KNN] = jj;// include itself

		for (kk = 0; kk < KNN + 1; kk++)
		{
			marker[count] = STACK[kk];
			count++;
		}

		Fcorners[ngoodCandiates].x = goodCorners[jj].x, Fcorners[ngoodCandiates].y = goodCorners[jj].y;
		for (kk = 0; kk < KNN; kk++)
		{
			Fcorners[ngoodCandiates].x += goodCorners[STACK[kk]].x;
			Fcorners[ngoodCandiates].y += goodCorners[STACK[kk]].y;
		}
		Fcorners[ngoodCandiates].x /= (KNN + 1);
		Fcorners[ngoodCandiates].y /= (KNN + 1);
		FStype[ngoodCandiates] = goodCandiates[jj];
		ngoodCandiates++;
	}
	npts = ngoodCandiates;

	delete[]maskSmooth;
	delete[]goodCorners;
	delete[]goodCandiates;
	delete[]marker;
	delete[]squareType;
	delete[]mergeCorners;

	return;
}
void RefineCornersFromInit(double *Para, int width, int height, int nchannels, Point2d *Checker, int &npts, vector<double>PatternAngles, int hsubset1, int hsubset2, int searchArea, double ZNCCCoarseThresh, double ZNCCthresh, int InterpAlgo)
{
	int numPatterns = PatternAngles.size();
	int bi_graylevel[2] = { 0, 255 }, PatternSize = 48, PatternLength = PatternSize*PatternSize; //Note that the pattern size is deliberately make bigger than the hsubset because small size give very blurry checkercorner

	double *maskSmooth = new double[PatternLength*numPatterns];
	synthesize_square_mask(maskSmooth, bi_graylevel, PatternSize, 1.0, 0, false);

	double trans[9], temp[9], iH1[9], H1[9] = { 1, 0, -PatternSize / 2, 0, 1, -PatternSize / 2, 0, 0, 1 };
	for (int ii = 1; ii < numPatterns; ii++)
	{
		double c = cos(PatternAngles[ii] * 3.14159265359 / 180), s = sin(PatternAngles[ii] * 3.14159265359 / 180);
		double H2[9] = { c, -s, 0, s, c, 0, 0, 0, 1 };
		mat_invert(H1, iH1, 3), mat_mul(H2, H1, temp, 3, 3, 3), mat_mul(iH1, temp, trans, 3, 3, 3);
		TransformImage(maskSmooth + ii*PatternLength, PatternSize, PatternSize, maskSmooth, PatternSize, PatternSize, trans, 1, 1, NULL);
	}

	//Detect coarse corners:
	double zncc, bestzncc;
	Point2d bestPts, bkPt;
	int advanced_tech = 3; // affine only
	double *Znssd_reqd = new double[9 * PatternLength];

	for (int ii = 0; ii < npts; ii++)
	{
		bestzncc = 0.0;
		for (int jj = 0; jj < numPatterns; jj++)
		{
			bkPt = Checker[ii];
			zncc = TMatchingFine_ZNCC(maskSmooth + jj* PatternLength, PatternSize, hsubset2, Para, width, height, nchannels, bkPt, advanced_tech, 1, ZNCCthresh, InterpAlgo, Znssd_reqd);
			if (zncc > bestzncc)
			{
				bestzncc = zncc;
				bestPts = bkPt;
			}
		}

		if (bestzncc < ZNCCthresh)
			Checker[ii] = Point2d(-1, -1);
		else
			Checker[ii] = bestPts;
	}
	delete[]Znssd_reqd;
	delete[]maskSmooth;

	return;
}
void RunCornersDetector(Point2d *CornerPts, int *CornersType, int &nCpts, double *Img, double *IPara, int width, int height, int nchannels, vector<double>PatternAngles, int hsubset1, int hsubset2, int searchArea, double ZNCCCoarseThresh, double ZNCCThresh, int InterpAlgo)
{
	int npts = 500000;
	Point2d *Checker = new Point2d[npts];

	//#pragma omp critical
	//cout << "Sliding window for detection..." << endl;

	DetectCornersCorrelation(Img, width, height, nchannels, Checker, npts, PatternAngles, hsubset1, searchArea, ZNCCCoarseThresh);
	/*FILE *fp = fopen("C:/temp/cornerCorr.txt", "w+");
	for (int ii = 0; ii < npts; ii++)
	fprintf(fp, "%.1f %1f\n", Checker[ii].x, Checker[ii].y);
	fclose(fp);
	FILE *fp = fopen("C:/temp/cornerCorr.txt", "r");
	npts = 0;
	while (fscanf(fp, "%lf %lf ", &Checker[npts].x, &Checker[npts].y) != EOF)
	npts++;
	fclose(fp);*/

	//#pragma omp critical
	//cout << "finished width " << npts << " points. Refine detected corners..." << endl;

	RefineCorners(IPara, width, height, nchannels, Checker, CornerPts, CornersType, npts, PatternAngles, hsubset1, hsubset2, searchArea, ZNCCCoarseThresh, ZNCCThresh, InterpAlgo);
	nCpts = npts;

	delete[]Checker;
	return;
}
int CornerDetectorDriver(char *Path, int checkerSize, double ZNCCThreshold, int startF, int stopF, int width, int height)
{
	char Fname[200];

	const int maxPts = 5000, SearchArea = 1, InterpAlgo = 1;
	double Gsigma = 1.0;
	vector<double> PatternAngles;
	for (int ii = 0; ii < 7; ii++)
		PatternAngles.push_back(10 * ii);
	int CheckerhSubset = (int)(0.5*checkerSize + 0.5);

	int nPts, nCCorres = maxPts;
	int CType[maxPts];
	Point2d Pts[maxPts];

	double *Img = new double[width*height];
	double *SImg = new double[width*height];
	double *IPara = new double[width*height];

	sprintf(Fname, "%s/Corner", Path), makeDir(Fname);
	for (int fid = startF; fid <= stopF; fid++)
	{
		sprintf(Fname, "%s/%d.png", Path, fid);
		if (!GrabImage(Fname, Img, width, height, 1))
			continue;

		Gaussian_smooth(Img, SImg, height, width, 255.0, 0.707);
		Generate_Para_Spline(SImg, IPara, width, height, InterpAlgo);
		//ShowDataAsImage("C:/temp/x.png", Img, width, height, 1);

		RunCornersDetector(Pts, CType, nPts, SImg, IPara, width, height, 1, PatternAngles, CheckerhSubset, CheckerhSubset, SearchArea, ZNCCThreshold - 0.35, ZNCCThreshold, InterpAlgo);

#pragma omp critical
		printf("%s: %d points\n", Fname, nPts);

		sprintf(Fname, "%s/Corner/%d.txt", Path, fid); FILE *fp = fopen(Fname, "w+");
		for (int ii = 0; ii < nPts; ii++)
		{
			if (Pts[ii].x<1.5*CheckerhSubset || Pts[ii].x > width - 1.5*CheckerhSubset || Pts[ii].y < 1.5*CheckerhSubset || Pts[ii].y > height - 1.5*CheckerhSubset)
				continue;
			fprintf(fp, "%d %.3f %.3f\n", CType[ii], Pts[ii].x, Pts[ii].y);
		}
		fclose(fp);
	}

	delete[]Img, delete[]SImg, delete[]IPara;

	return 0;
}
int DetectMarkersandCorrespondence(char *PATH, Point2d *Pcorners, int *PcorType, double *PPara, LKParameters LKArg, double *CamProScale, vector<double>PatternAngles, int checkerSize, int nPpts, int PSncols, Point2i *ProjectorCorressBoundary, double EpipThresh, int frameID, int CamID, int nPros, int width, int height, int pwidth, int pheight)
{
	int length = width*height;
	const int maxPts = 5000, SearchArea = 2;
	int CheckerhSubset1 = (int)(0.5*checkerSize + 0.5), CheckerhSubset2 = (int)(0.5*checkerSize + 0.5); //A reasonable checkerhsubset gives better result than using the huge subset

	int  ii, jj, nCPts, nCCorres = maxPts, CType1[maxPts];
	Point2d CPts1[maxPts], Ppts[maxPts], CPts[maxPts * 2];
	Point2i CCorresID[maxPts];
	int T[maxPts], TT[maxPts];

	IplImage *view = 0;
	char *Img = new char[2 * width*height];
	double *SImg = new double[width*height];
	double *IPara = new double[2 * width*height];

	char Fname[100];
	sprintf(Fname, "%s/Image/C%d_%05d.png", PATH, CamID + 1, frameID);
	view = cvLoadImage(Fname, 0);
	if (view == NULL)
	{
		cout << "cannot load " << Fname << endl;
		delete[]Img;
		delete[]SImg;
		delete[]IPara;
		return 1;
	}
	cout << "Loaded " << Fname << endl;
	for (jj = 0; jj < height; jj++)
		for (ii = 0; ii < width; ii++)
			Img[ii + (height - 1 - jj)*width] = view->imageData[ii + jj*width];
	cvReleaseImage(&view);

	Gaussian_smooth(Img, SImg, height, width, 255.0, LKArg.Gsigma);
	Generate_Para_Spline(SImg, IPara, width, height, LKArg.InterpAlgo);

	RunCornersDetector(CPts1, CType1, nCPts, SImg, IPara, width, height, 1, PatternAngles, CheckerhSubset1, CheckerhSubset2, SearchArea, LKArg.ZNCCThreshold - 0.35, LKArg.ZNCCThreshold, LKArg.InterpAlgo);
	/*FILE *fp = fopen("C:/temp/pts.txt", "w+");
	for (int ii = 0; ii < nCPts; ii++)
	fprintf(fp, "%f %f %d \n", CPts1[ii].x, CPts1[ii].y, CType1[ii]);
	fclose(fp);

	{
	nCPts = 0;
	FILE *fp = fopen("C:/temp/pts.txt", "r");
	while (fscanf(fp, "lf %lf %d %", &CPts1[nCPts].x, &CPts1[nCPts].y, &CType1[nCPts]) != EOF)
	nCPts++;
	fclose(fp);
	}*/
	for (int ProID = 0; ProID < nPros; ProID++)
	{
		for (ii = 0; ii < nCCorres; ii++)
		{
			T[ii] = CCorresID[ii].y;
			TT[ii] = ii;
		}
		Quick_Sort_Int(T, TT, 0, nCCorres - 1);

		for (ii = 0; ii < nCCorres; ii++)
		{
			Ppts[ii].x = Pcorners[CCorresID[TT[ii]].y].x;
			Ppts[ii].y = Pcorners[CCorresID[TT[ii]].y].y;
			CPts[ii].x = CPts1[CCorresID[TT[ii]].x].x;
			CPts[ii].y = CPts1[CCorresID[TT[ii]].x].y;
		}

		sprintf(Fname, "%s/Sparse/P%dC%d_%05d.txt", PATH, ProID + 1, CamID + 1, frameID);
		FILE *fp = fopen(Fname, "w+");
		for (ii = 0; ii < nCCorres; ii++)
			fprintf(fp, "%d %.8f %.8f \n", CCorresID[TT[ii]].y, Ppts[ii].x, Ppts[ii].y);
		fclose(fp);

		sprintf(Fname, "%s/Sparse/C%dP%d_%05d.txt", PATH, CamID + 1, ProID + 1, frameID);
		fp = fopen(Fname, "w+");
		for (ii = 0; ii < nCCorres; ii++)
			fprintf(fp, "%.8f %.8f \n", CPts[ii].x, CPts[ii].y);
		fclose(fp);

		if (nCCorres < 10)
			cout << "Something wrong with #sparse points." << endl;
		cout << nCCorres << " pts dected for frame " << frameID << endl;
	}

	delete[]Img;
	delete[]SImg;
	delete[]IPara;

	return 0;
}

int ReadCorresAndRunTracking(char *Path, int nviews, int startFrame, int beginFrame, int endFrame, int *FrameOffset, int HighFrameRateFactor)
{
	char Fname[200];

	double u, v;
	int nvis, viewid, TrajCount = 0;
	sprintf(Fname, "%s/Dynamic/Corres_%d.txt", Path, startFrame); FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return 1;
	}
	while (fscanf(fp, "%d ", &nvis) != EOF)
	{
		for (int ii = 0; ii < nvis; ii++)
			fscanf(fp, "%d %lf %lf ", &viewid, &u, &v);
		TrajCount++;
	}
	fclose(fp);

	TrajectoryData TrajInfo;
	TrajInfo.nTrajectories = TrajCount;
	TrajInfo.trajectoryUnit = new vector<Trajectory2D>[TrajCount];

	TrajCount = 0;
	sprintf(Fname, "%s/Dynamic/Corres_%d.txt", Path, startFrame); fp = fopen(Fname, "r");
	while (fscanf(fp, "%d ", &nvis) != EOF)
	{
		Trajectory2D traj2d;
		traj2d.nViews = nvis;

		for (int ii = 0; ii < nvis; ii++)
		{
			fscanf(fp, "%d %lf %lf ", &viewid, &u, &v);
			traj2d.viewIDs.push_back(viewid);
			traj2d.uv.push_back(Point2d(u, v));
			traj2d.frameID.push_back((startFrame + FrameOffset[ii] - 1)*HighFrameRateFactor + 1);
		}

		TrajInfo.trajectoryUnit[TrajCount].push_back(traj2d);
		TrajCount++;
	}
	fclose(fp);
	//May need to do refinement on the inital corres

	Mat *cImg = new Mat[nviews], *nImg = new Mat[nviews];
	static const int arr[] = { 15, 19, 23 };
	vector<int> pyrSizeVector(arr, arr + sizeof(arr) / sizeof(arr[0]));

	Mat Oimg, Nimg; int c = 0;

	//forward track
	double starttime = omp_get_wtime();
	printf("Forward flow: ");
	for (int frameID = (startFrame - 1)*HighFrameRateFactor + 1; frameID <= (endFrame - 1)*HighFrameRateFactor + 1 - 1; frameID++)
	{
		//Read Image
		printf("%d ... ", frameID);
		if (frameID == (startFrame - 1)*HighFrameRateFactor + 1)
		{
			for (int jj = 0; jj < nviews; jj++)
			{
				if (HighFrameRateFactor == 1)
					sprintf(Fname, "%s/%d/%d.png", Path, jj, frameID + FrameOffset[jj]);
				else
					sprintf(Fname, "%s/%d/F/%d.png", Path, jj, frameID + FrameOffset[jj] * HighFrameRateFactor);
				cImg[jj] = imread(Fname, 0);
				if (cImg[jj].empty() && frameID + FrameOffset[jj] * HighFrameRateFactor <= (endFrame - 1)*HighFrameRateFactor)
					printf("cannot load %s\n", Fname);
			}
		}
		for (int jj = 0; jj < nviews; jj++)
		{
			if (HighFrameRateFactor == 1)
				sprintf(Fname, "%s/%d/%d.png", Path, jj, frameID + FrameOffset[jj]);
			else
				sprintf(Fname, "%s/%d/F/%d.png", Path, jj, frameID + FrameOffset[jj] * HighFrameRateFactor + 1);
			nImg[jj] = imread(Fname, 0);
			if (nImg[jj].empty() && frameID + FrameOffset[jj] * HighFrameRateFactor + 1 <= (endFrame - 1)*HighFrameRateFactor)
				printf("cannot load %s\n", Fname);
		}

		//Run tracking
		bool addedPt = false;
		for (int trajID = 0; trajID < TrajCount; trajID++)
		{
			Trajectory2D traj2d;
#pragma omp parallel for
			for (int view = 0; view < TrajInfo.trajectoryUnit[trajID].back().viewIDs.size(); view++)
			{
				int viewID = TrajInfo.trajectoryUnit[trajID].back().viewIDs[view];
				Mat cImage = cImg[viewID].clone(), nImage = nImg[viewID].clone();
				if (cImage.empty() || nImage.empty())
					continue;

				vector<Point2f> prePts; prePts.push_back(TrajInfo.trajectoryUnit[trajID].back().uv[view]);
				vector<Point2f> newPts; newPts.push_back(Point2f(0, 0));

				bool success = false;
				double MinFlowBackFore = 9e9;
				for (int pyamidIdx = 0; pyamidIdx < pyrSizeVector.size(); pyamidIdx++)
				{
					vector<Point2f> foreTrackPts, backTrackedPts;
					vector<uchar> tempOutputStatus, tempOutputStatus_2;
					vector<float> tempOutputError, tempOutputError_2;

					int tempPyramidSize = pyrSizeVector[pyamidIdx];
					calcOpticalFlowPyrLK(cImage, nImage, prePts, foreTrackPts, tempOutputStatus, tempOutputError, cvSize(tempPyramidSize, tempPyramidSize), 2);
					calcOpticalFlowPyrLK(nImage, cImage, foreTrackPts, backTrackedPts, tempOutputStatus_2, tempOutputError_2, cvSize(tempPyramidSize, tempPyramidSize), 2);

					double tempBackTrackDist = Distance2D(prePts[0], backTrackedPts[0]);
					if (!tempOutputStatus[0] || !tempOutputStatus_2[0] || tempBackTrackDist > OPTICALFLOW_BIDIRECT_DIST_THRESH)  //invalid optical flow
						continue;
					else
					{
						if (MinFlowBackFore > tempBackTrackDist)
						{
							MinFlowBackFore = tempBackTrackDist;
							newPts[0] = foreTrackPts[0], success = true;
						}
					}
				}

#pragma omp critical
				if (success)
				{
					traj2d.viewIDs.push_back(viewID);
					traj2d.uv.push_back(newPts[0]);
					traj2d.frameID.push_back(frameID + FrameOffset[viewID] * HighFrameRateFactor + 1);
					if (frameID + FrameOffset[viewID] * HighFrameRateFactor + 1 >= 599)
						int a = 0;

					if (0)
					{
						if (c == 0)
						{
							cvtColor(cImage, Oimg, CV_GRAY2BGR);
							circle(Oimg, prePts[0], 5, Scalar(83, 185, 255), 1, 8);
							sprintf(Fname, "C:/temp/%d.png", c); c++;
							imwrite(Fname, Oimg);
						}

						cvtColor(nImage, Nimg, CV_GRAY2BGR);
						circle(Nimg, newPts[0], 5, Scalar(83, 185, 255), 1, 8);
						sprintf(Fname, "C:/temp/%d.png", c); c++;
						imwrite(Fname, Nimg);
					}
				}
			}

#pragma omp critical
			if (traj2d.viewIDs.size() > 0)
			{
				addedPt = true;
				TrajInfo.trajectoryUnit[trajID].push_back(traj2d);
			}
		}

		if (!addedPt)
			break;

		for (int jj = 0; jj < nviews; jj++)
			swap(nImg[jj], cImg[jj]);
	}
	printf("takes %.2fs\n", omp_get_wtime() - starttime);

	//backward track
	printf("\nBackward flow: ");
	starttime = omp_get_wtime();
	for (int frameID = (startFrame - 1)*HighFrameRateFactor + 1; frameID >= (beginFrame - 1)*HighFrameRateFactor + 1 + 1; frameID--)
	{
		//Read Image
		printf("%d ... ", frameID);
		if (frameID == (startFrame - 1)*HighFrameRateFactor + 1)
		{
			for (int jj = 0; jj < nviews; jj++)
			{
				if (HighFrameRateFactor == 1)
					sprintf(Fname, "%s/%d/%d.png", Path, jj, frameID + FrameOffset[jj]);
				else
					sprintf(Fname, "%s/%d/F/%d.png", Path, jj, frameID + FrameOffset[jj] * HighFrameRateFactor);
				cImg[jj] = imread(Fname, 0);
				if (cImg[jj].empty() && frameID + FrameOffset[jj] * HighFrameRateFactor >= (beginFrame - 1)*HighFrameRateFactor + 1)
					printf("cannot load %s\n", Fname);
			}
		}
		for (int jj = 0; jj < nviews; jj++)
		{
			if (HighFrameRateFactor == 1)
				sprintf(Fname, "%s/%d/%d.png", Path, jj, frameID + FrameOffset[jj]);
			else
				sprintf(Fname, "%s/%d/F/%d.png", Path, jj, frameID + FrameOffset[jj] * HighFrameRateFactor - 1);
			nImg[jj] = imread(Fname, 0);
			if (nImg[jj].empty() && frameID + FrameOffset[jj] * HighFrameRateFactor + 1 >= (beginFrame - 1)*HighFrameRateFactor + 1)
				printf("cannot load %s\n", Fname);
		}

		//Run tracking
		bool addedPt = false;
#pragma omp parallel for
		for (int trajID = 0; trajID < TrajCount; trajID++)
		{
			Trajectory2D traj2d;
			int firstnVis = (frameID == (startFrame - 1)*HighFrameRateFactor + 1) ? TrajInfo.trajectoryUnit[trajID][0].viewIDs.size() : TrajInfo.trajectoryUnit[trajID].back().viewIDs.size();
			for (int view = 0; view < firstnVis; view++)
			{
				int viewID = (frameID == (startFrame - 1)*HighFrameRateFactor + 1) ? TrajInfo.trajectoryUnit[trajID][0].viewIDs[view] : TrajInfo.trajectoryUnit[trajID].back().viewIDs[view];
				Mat cImage = cImg[viewID].clone(), nImage = nImg[viewID].clone();
				if (cImage.empty() || nImage.empty())
					continue;
				//imwrite("C:/temp/c.png", cImage), imwrite("C:/temp/n.png", nImage);

				vector<Point2f> prePts;
				if (frameID == (startFrame - 1)*HighFrameRateFactor + 1)
					prePts.push_back(TrajInfo.trajectoryUnit[trajID][0].uv[view]);
				else
					prePts.push_back(TrajInfo.trajectoryUnit[trajID].back().uv[view]);

				vector<Point2f> newPts; newPts.push_back(Point2f(0, 0));

				bool success = false;
				double MinFlowBackFore = 9e9;
				for (int pyamidIdx = 0; pyamidIdx < pyrSizeVector.size(); pyamidIdx++)
				{
					vector<Point2f> foreTrackPts, backTrackedPts;
					vector<uchar> tempOutputStatus, tempOutputStatus_2;
					vector<float> tempOutputError, tempOutputError_2;

					int tempPyramidSize = pyrSizeVector[pyamidIdx];
					calcOpticalFlowPyrLK(cImage, nImage, prePts, foreTrackPts, tempOutputStatus, tempOutputError, cvSize(tempPyramidSize, tempPyramidSize), 3);
					calcOpticalFlowPyrLK(nImage, cImage, foreTrackPts, backTrackedPts, tempOutputStatus_2, tempOutputError_2, cvSize(tempPyramidSize, tempPyramidSize), 3);

					double tempBackTrackDist = Distance2D(prePts[0], backTrackedPts[0]);
					if (!tempOutputStatus[0] || !tempOutputStatus_2[0] || tempBackTrackDist > OPTICALFLOW_BIDIRECT_DIST_THRESH)  //invalid optical flow
						continue;
					else
					{
						if (MinFlowBackFore > tempBackTrackDist)
						{
							MinFlowBackFore = tempBackTrackDist;
							newPts[0] = foreTrackPts[0], success = true;
						}
					}
				}

#pragma omp critical
				if (success)
				{
					traj2d.viewIDs.push_back(viewID);
					traj2d.uv.push_back(newPts[0]);
					traj2d.frameID.push_back(frameID + FrameOffset[viewID] * HighFrameRateFactor - 1);

					if (0)
					{
						cvtColor(nImage, Nimg, CV_GRAY2BGR);
						circle(Nimg, newPts[0], 5, Scalar(83, 185, 255), 1, 8);
						sprintf(Fname, "C:/temp/%d.png", c); c++;
						imwrite(Fname, Nimg);
					}
				}
			}

#pragma omp critical
			if (traj2d.viewIDs.size() > 0)
			{
				addedPt = true;
				TrajInfo.trajectoryUnit[trajID].push_back(traj2d);
			}
		}

		if (!addedPt)
			break;

		for (int jj = 0; jj < nviews; jj++)
			swap(nImg[jj], cImg[jj]);
	}
	printf("takes %.2fs\n", omp_get_wtime() - starttime);

	//Write TrajData
	printf("\nWrite results ...");
	for (int viewID = 0; viewID < nviews; viewID++)
	{
		sprintf(Fname, "%s/Track2D/%d.txt", Path, viewID); FILE *fp = fopen(Fname, "a");
		for (int trajID = 0; trajID < TrajCount; trajID++) // for all trajectories
		{
			int count = 0;
			for (int fID = 0; fID < TrajInfo.trajectoryUnit[trajID].size(); fID++) //for all time instances in that trajectory
				for (int vid = 0; vid < TrajInfo.trajectoryUnit[trajID][fID].viewIDs.size(); vid++) //for all visible cameras of that time instances
					if (TrajInfo.trajectoryUnit[trajID][fID].viewIDs[vid] == viewID)
						count++;

			fprintf(fp, "%d %d", trajID, count);
			for (int fID = 0; fID < TrajInfo.trajectoryUnit[trajID].size(); fID++) //for all time instances in that trajectory
			{
				for (int vid = 0; vid < TrajInfo.trajectoryUnit[trajID][fID].viewIDs.size(); vid++) //for all visible cameras of that time instances
				{
					if (TrajInfo.trajectoryUnit[trajID][fID].viewIDs[vid] == viewID)
						fprintf(fp, "%d %.3f %.3f ", TrajInfo.trajectoryUnit[trajID][fID].frameID[vid], TrajInfo.trajectoryUnit[trajID][fID].uv[vid].x, TrajInfo.trajectoryUnit[trajID][fID].uv[vid].y);
				}
			}
			fprintf(fp, "\n");
		}
		fclose(fp);
	}
	printf("done!\n");

	delete[]cImg, delete[]nImg;

	return 0;
}
int CleanUp2DTrackingByGradientConsistency(char *Path, int nviews, int ntrajects)
{
	double magThresh = 15.0;
	int d1, frameCount;
	double t1, t2;
	char Fname[512];
	vector<Point2d> UVList;

	int ID[MaxnFrames], frameIDList[MaxnFrames];
	double uList[MaxnFrames], vList[MaxnFrames];
	double duList[MaxnFrames], dvList[MaxnFrames], direc[MaxnFrames], mag[MaxnFrames];

	for (int viewID = 0; viewID < nviews; viewID++)
	{
		sprintf(Fname, "%s/Track2D/%d.txt", Path, viewID); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("Cannot load %s\n", Fname);
			return 1;
		}
		printf("Working on %s ...\n", Fname);
		sprintf(Fname, "%s/Track2D/R_%d.txt", Path, viewID); FILE *fp2 = fopen(Fname, "w+");

		fscanf(fp, "%d ", &d1);
		if (abs(d1 + 1) < 0.01)
			fscanf(fp, "%d ", &d1);

		for (int trajID = 0; trajID < ntrajects; trajID++)
		{
			UVList.clear();
			frameCount = 0;
			while (true)
			{
				if (fscanf(fp, "%d ", &d1) == EOF)
					break;

				if (d1 == -1)
				{
					fscanf(fp, "%d ", &d1);//end of this traj;
					break;
				}
				else
				{
					fscanf(fp, "%lf %lf ", &t1, &t2);
					frameIDList[frameCount] = d1;
					ID[frameCount] = frameCount;
					UVList.push_back(Point2d(t1, t2));
					frameCount++;
				}
			}

			//Sort tracked points
			Quick_Sort_Int(frameIDList, ID, 0, frameCount - 1);
			for (int frameID = 0; frameID < frameCount; frameID++)
				uList[frameID] = UVList[ID[frameID]].x, vList[frameID] = UVList[ID[frameID]].y;

			//Compute the gradient
			for (int frameID = 0; frameID < frameCount - 1; frameID++)
				duList[frameID] = uList[frameID + 1] - uList[frameID], dvList[frameID] = vList[frameID + 1] - vList[frameID];

			//Compute direction and mag
			for (int frameID = 0; frameID < frameCount - 1; frameID++)
				direc[frameID] = abs(atan2(dvList[frameID], duList[frameID])),
				mag[frameID] = sqrt(pow(duList[frameID], 2) + pow(dvList[frameID], 2));

			//check for large changes
			int breakFrame = 0;
			for (int frameID = 0; frameID < frameCount - 2; frameID++)
				if (mag[frameID] * direc[frameID] > 0.3* mag[frameID + 1] * direc[frameID + 1] || mag[frameID + 1] < magThresh || mag[frameID] < 0.01)
					breakFrame++;
				else
					break;
			breakFrame -= 10;

			fprintf(fp2, "%d %d ", trajID, breakFrame);
			for (int frameID = 0; frameID < breakFrame; frameID++)
				fprintf(fp2, "%d %.3f %.3f ", frameIDList[frameID], UVList[ID[frameID]].x, UVList[ID[frameID]].y);
			fprintf(fp2, "\n");
		}
		fclose(fp), fclose(fp2);
	}

	return 0;
}
int DownSampleTracking(char *Path, int nviews, int ntrajects, int Factor)
{
	char Fname[200];
	int d, nframes;
	int frameIDList[MaxnFrames];
	Point2d uvList[MaxnFrames];

	for (int viewID = 0; viewID < nviews; viewID++)
	{
		sprintf(Fname, "%s/Track2D/C_%d.txt", Path, viewID); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("Cannot load %s\n", Fname);
			return 1;
		}
		printf("Working on %s ...\n", Fname);
		sprintf(Fname, "%s/Track2D/CL_%d.txt", Path, viewID); FILE *fp2 = fopen(Fname, "w+");

		for (int trajID = 0; trajID < ntrajects; trajID++)
		{
			fscanf(fp, "%d %d ", &d, &nframes);
			for (int fid = 0; fid < nframes; fid++)
				fscanf(fp, "%d %lf %lf ", &frameIDList[fid], &uvList[fid].x, &uvList[fid].y);

			int framecount = 0;
			for (int fid = 0; fid < nframes; fid++)
				if ((frameIDList[fid] - 1) % Factor == 0)
					framecount++;

			fprintf(fp2, "%d %d ", trajID, framecount);
			for (int fid = 0; fid < nframes; fid++)
				if ((frameIDList[fid] - 1) % Factor == 0)
					fprintf(fp2, "%d %.3f %.3f ", (frameIDList[fid] - 1) / Factor + 1, uvList[fid].x, uvList[fid].y);
			fprintf(fp2, "\n");
		}
		fclose(fp), fclose(fp2);
	}
	return 0;
}
int DeletePointsOf2DTracks(char *Path, int nCams, int npts)
{
	char Fname[200];

	int pid;
	vector<int>ChosenPid;
	sprintf(Fname, "%s/chosen.txt", Path);  FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return 1;
	}
	while (fscanf(fp, "%d ", &pid) != EOF)
		ChosenPid.push_back(pid);
	fclose(fp);

	double u, v;
	int fid, nf;
	ImgPtEle ptele;
	for (int camID = 0; camID < nCams; camID++)
	{
		vector<ImgPtEle> *Track2D = new vector<ImgPtEle>[npts];
		sprintf(Fname, "%s/Track2D/C_%d.txt", Path, camID); fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("Cannot load %s\n", Fname);
			continue;
		}
		for (int jj = 0; jj < npts; jj++)
		{
			fscanf(fp, "%d %d ", &pid, &nf);
			if (pid != jj)
				printf("Problem at Point %d of Cam %d", jj, camID);
			for (int ii = 0; ii < nf; ii++)
			{
				fscanf(fp, "%d %lf %lf ", &fid, &u, &v);
				ptele.frameID = fid, ptele.pt2D = Point2d(u, v);
				Track2D[jj].push_back(ptele);
			}
		}
		fclose(fp);

		sprintf(Fname, "%s/Track2D/C_%d.txt", Path, camID); fp = fopen(Fname, "w+");
		for (int pid = 0; pid < ChosenPid.size(); pid++)
		{
			int trackID = ChosenPid[pid];
			fprintf(fp, "%d %d ", pid, Track2D[trackID].size());
			for (int fid = 0; fid < Track2D[trackID].size(); fid++)
				fprintf(fp, "%d %.3f %.3f ", Track2D[trackID][fid].frameID, Track2D[trackID][fid].pt2D.x, Track2D[trackID][fid].pt2D.y);
			fprintf(fp, "\n");
		}
		fclose(fp);

		delete[]Track2D;
	}

	return 0;
}

void DetectBlobCorrelation(char *ImgName, vector<KeyPoint> &kpts, int nOctaveLayers, int nScalePerOctave, double sigma, int PatternSize, int NMS_BW, double thresh)
{
	int jump = 1, numPatterns = 1;
	char Fname[200];

	ImgPyr imgpyrad;
	double starttime = omp_get_wtime();
	BuildImgPyr(ImgName, imgpyrad, nOctaveLayers, nScalePerOctave, false, 1, 1.0);
	for (int ii = 0; ii < imgpyrad.ImgPyrImg.size(); ii++)
	{
		sprintf(Fname, "C:/temp/L%d.png", ii);
		SaveDataToImage(Fname, imgpyrad.ImgPyrImg[ii], imgpyrad.wh[ii].x, imgpyrad.wh[ii].y, 1);
	}
	printf("Building Image pyramid: %.fs\n", omp_get_wtime() - starttime);

	//build template
	int hsubset = PatternSize / 2, PatternLength = PatternSize*PatternSize;
	int IntensityProfile[] = { 10, 240 };
	double RingInfo[1] = { 0.9 };
	double *maskSmooth = new double[PatternLength];
	synthesize_concentric_circles_mask(maskSmooth, IntensityProfile, PatternSize, 1, PatternSize, RingInfo, 0, 1);
	//SaveDataToImage("C:/temp/mask.png", maskSmooth, PatternSize, PatternSize);

	Mat tpl = Mat::zeros(PatternSize, PatternSize, CV_32F);
	for (int ii = 0; ii < PatternLength; ii++)
		tpl.at<float>(ii) = maskSmooth[ii];

	int width = imgpyrad.wh[0].x, height = imgpyrad.wh[0].y;
	float *response = new float[width*height];

	for (int scaleID = 0; scaleID < imgpyrad.ImgPyrImg.size(); scaleID++)
	{
		starttime = omp_get_wtime();
		printf("Layer %d ....", scaleID);
		int width = imgpyrad.wh[scaleID].x, height = imgpyrad.wh[scaleID].y;

		Mat ref = Mat::zeros(height, width, CV_32F);
		for (int ii = 0; ii < width*height; ii++)
			ref.at<float>(ii) = (float)(int)imgpyrad.ImgPyrImg[scaleID][ii];

		Mat dst;
		cv::matchTemplate(ref, tpl, dst, CV_TM_CCORR_NORMED);
		for (int ii = 0; ii < dst.rows*dst.cols; ii++)
			response[ii] = dst.at<float>(ii);

		//sprintf(Fname, "C:/temp/x_%d.dat", scaleID);
		//WriteGridBinary(Fname, response, dst.cols, dst.rows);

		//Non-max suppression:
		bool breakflag;
		int ScoreW = dst.cols, ScoreH = dst.rows;
		for (int jj = hsubset; jj < ScoreH - hsubset; jj += jump)
		{
			for (int ii = hsubset; ii < ScoreW - hsubset; ii += jump)
			{
				breakflag = false;
				if (response[ii + jj*ScoreW] < thresh)
					response[ii + jj*ScoreW] = 0.0;
				else
				{
					for (int j = -NMS_BW; j <= NMS_BW && !breakflag; j += jump)
					{
						for (int i = -NMS_BW; i <= NMS_BW&& !breakflag; i += jump)
						{
							if (i == 0 && j == 0)
								continue;
							if (ii + i< 0 || ii + i>ScoreW || jj + j < 0 || jj + j>ScoreH)
								continue;
							if (response[ii + jj*ScoreW] < response[(ii + i) + (jj + j)*ScoreW])
							{
								response[ii + jj*ScoreW] = 0.0;
								breakflag = true;
								break;
							}
						}
					}
				}
			}
		}
		//WriteGridBinary("C:/temp/x.dat", response, dst.cols, dst.rows);

		for (int jj = hsubset; jj < ScoreH - hsubset; jj += jump)
		{
			for (int ii = hsubset; ii < ScoreW - hsubset; ii += jump)
			{
				if (response[ii + jj*ScoreW] > thresh)
				{
					KeyPoint kpt;
					kpt.pt.x = (1.0*ii + PatternSize / 2) / imgpyrad.factor[scaleID];
					kpt.pt.y = (1.0*jj + PatternSize / 2) / imgpyrad.factor[scaleID];
					kpt.size = 1.0*PatternSize * imgpyrad.factor[scaleID];
					kpts.push_back(kpt);
				}
			}
		}
		printf(" %.2fs\n", omp_get_wtime() - starttime);
	}
	printf(" %.2fs\n", omp_get_wtime() - starttime);
	delete[]maskSmooth, delete[]response;

	width = imgpyrad.wh[0].x, height = imgpyrad.wh[0].y;
	Mat ref_gray = Mat::zeros(height, width, CV_8UC1);
	for (int ii = 0; ii < width*height; ii++)
		ref_gray.data[ii] = imgpyrad.ImgPyrImg[0][ii];
	cv::cvtColor(ref_gray, ref_gray, CV_GRAY2BGR);
	for (int ii = 0; ii < kpts.size(); ii++)
	{
		KeyPoint kpt = kpts[ii];
		int startX = kpt.pt.x - kpt.size / 2, startY = kpt.pt.y - kpt.size / 2;
		int stopX = kpt.pt.x + kpt.size / 2, stopY = kpt.pt.y + kpt.size / 2;
		cv::rectangle(ref_gray, Point2i(startX, startY), cv::Point(stopX, stopY), CV_RGB(0, 255, 0), 2);
	}
	cvNamedWindow("result", CV_WINDOW_NORMAL);
	imshow("result", ref_gray); waitKey();


	return;
}
int DetectRGBBallCorrelation(char *ImgName, vector<KeyPoint> &kpts, vector<int> &ballType, int nOctaveLayers, int nScalePerOctave, double sigma, int PatternSize, int NMS_BW, double thresh, bool visualize)
{
	char Fname[200];
	int nscales = nOctaveLayers*nScalePerOctave + 1;
	ImgPyr imgpyrad;
	double starttime = omp_get_wtime();
	if (BuildImgPyr(ImgName, imgpyrad, nOctaveLayers, nScalePerOctave, false, 1, 1.0) == 1)
		return 1;
	for (int ii = 0; ii < imgpyrad.ImgPyrImg.size() && visualize; ii++)
	{
		sprintf(Fname, "C:/temp/L%d.png", ii);
		SaveDataToImage(Fname, imgpyrad.ImgPyrImg[ii], imgpyrad.wh[ii].x, imgpyrad.wh[ii].y, 1);
	}
	printf("Building Image pyramid: %.fs\n", omp_get_wtime() - starttime);

	//build template
	int hsubset = PatternSize / 2, PatternLength = PatternSize*PatternSize;
	int IntensityProfile[] = { 10, 240 };
	double RingInfo[1] = { 0.9 };
	double *maskSmooth = new double[PatternLength*nscales*nscales];
	synthesize_concentric_circles_mask(maskSmooth, IntensityProfile, PatternSize, 1, PatternSize, RingInfo, 0, 1);
	SaveDataToImage("C:/temp/mask.png", maskSmooth, PatternSize, PatternSize);


	Mat tpl = Mat::zeros(PatternSize, PatternSize, CV_32F);
	for (int ii = 0; ii < PatternLength; ii++)
		tpl.at<float>(ii) = maskSmooth[ii];

	// Standard multiscale correlation detection
	vector<KeyPoint> potentialPts;
	int width = imgpyrad.wh[0].x, height = imgpyrad.wh[0].y, length = width*height;
	/*float *response = new float[width*height];
	for (int scaleID = 0; scaleID < imgpyrad.ImgPyrImg.size(); scaleID++)
	{
	starttime = omp_get_wtime();
	printf("Layer %d ....", scaleID);
	int width = imgpyrad.wh[scaleID].x, height = imgpyrad.wh[scaleID].y;

	Mat ref = Mat::zeros(height, width, CV_32F);
	for (int ii = 0; ii < width*height; ii++)
	ref.at<float>(ii) = (float)(int)imgpyrad.ImgPyrImg[scaleID][ii];

	Mat dst;
	cv::matchTemplate(ref, tpl, dst, CV_TM_CCOEFF_NORMED);
	for (int ii = 0; ii < dst.rows*dst.cols; ii++)
	response[ii] = dst.at<float>(ii);
	sprintf(Fname, "C:/temp/L_%d.dat", scaleID);	WriteGridBinary(Fname, response, dst.cols, dst.rows);

	//Non-max suppression:
	bool breakflag;
	int ScoreW = dst.cols, ScoreH = dst.rows;
	for (int jj = hsubset; jj < ScoreH - hsubset; jj ++)
	{
	for (int ii = hsubset; ii < ScoreW - hsubset; ii ++)
	{
	breakflag = false;
	if (response[ii + jj*ScoreW] < thresh)
	response[ii + jj*ScoreW] = 0.0;
	else
	{
	for (int j = -NMS_BW; j <= NMS_BW && !breakflag; j ++)
	{
	for (int i = -NMS_BW; i <= NMS_BW&& !breakflag; i ++)
	{
	if (i == 0 && j == 0)
	continue;
	if (ii + i< 0 || ii + i>ScoreW || jj + j < 0 || jj + j>ScoreH)
	continue;
	if (response[ii + jj*ScoreW] < response[(ii + i) + (jj + j)*ScoreW])
	{
	response[ii + jj*ScoreW] = 0.0;
	breakflag = true;
	break;
	}
	}
	}
	}
	}
	}


	Mat ref_gray = Mat::zeros(height, width, CV_8UC1);
	for (int ii = 0; ii < width*height; ii++)
	ref_gray.data[ii] = imgpyrad.ImgPyrImg[scaleID][ii];
	cv::cvtColor(ref_gray, ref_gray, CV_GRAY2BGR);

	for (int jj = hsubset; jj < ScoreH - hsubset; jj ++)
	{
	for (int ii = hsubset; ii < ScoreW - hsubset; ii ++)
	{
	if (response[ii + jj*ScoreW] > thresh)
	{
	KeyPoint kpt;
	kpt.pt.x = (1.0*ii + PatternSize / 2) / imgpyrad.factor[scaleID];
	kpt.pt.y = (1.0*jj + PatternSize / 2) / imgpyrad.factor[scaleID];
	kpt.size = 1.0*PatternSize / imgpyrad.factor[scaleID]; //size of what the template should be
	kpt.response = response[ii + jj*ScoreW];
	kpt.octave = scaleID;
	cv::rectangle(ref_gray, Point2i(ii, jj), cv::Point(ii + PatternSize, jj + PatternSize), CV_RGB(0, 255, 0), 2);
	potentialPts.push_back(kpt);
	}
	}
	}

	cvNamedWindow("result", CV_WINDOW_NORMAL);
	imshow("result", ref_gray); waitKey();
	printf(" %.2fs\n", omp_get_wtime() - starttime);
	}
	delete[]response;

	FILE *fp = fopen("C:/temp/kpts.txt", "w+");
	for (int kk = 0; kk < potentialPts.size(); kk++)
	fprintf(fp, "%.1f %.1f %.3f %.1f %d\n", potentialPts[kk].pt.x, potentialPts[kk].pt.y, potentialPts[kk].response, potentialPts[kk].size, potentialPts[kk].octave);
	fclose(fp);*/


	FILE *fp = fopen("C:/temp/kpts.txt", "r");
	int oct;
	float x, y, s, r;
	while (fscanf(fp, "%f %f %f %f %d", &x, &y, &r, &s, &oct) != EOF)
	{
		KeyPoint kpt;
		kpt.pt.x = x, kpt.pt.y = y, kpt.response = r, kpt.size = s;
		kpt.octave = oct;
		potentialPts.push_back(kpt);
	}
	fclose(fp);

	//Prune close-by points with lower response;
	vector<int> pointsToBeRemoved;
	for (int ll = 0; ll < potentialPts.size() - 1; ll++)
	{
		for (int kk = ll + 1; kk < potentialPts.size(); kk++)
		{
			if (abs(potentialPts[ll].pt.x - potentialPts[kk].pt.x) < potentialPts[ll].size / 2 && abs(potentialPts[ll].pt.y - potentialPts[kk].pt.y) < potentialPts[ll].size / 2)
				if (potentialPts[ll].response > potentialPts[kk].response)
					pointsToBeRemoved.push_back(kk);
				else
					pointsToBeRemoved.push_back(ll);
		}
	}

	vector<KeyPoint> potentialPts2;
	for (int ii = 0; ii < potentialPts.size(); ii++)
	{
		int jj;
		for (jj = 0; jj < pointsToBeRemoved.size(); jj++)
			if (ii == pointsToBeRemoved[jj])
				break;
		if (jj == pointsToBeRemoved.size())
			potentialPts2.push_back(potentialPts[ii]);
	}

	//Figure out RGB Ball:  the small regions around the detected point must have good color matching
	width = imgpyrad.wh[0].x, height = imgpyrad.wh[0].y;
	unsigned char *Img = new unsigned char[width*height * 3];
	GrabImage(ImgName, Img, width, height, 3);

	int count = 0, bw = width > 640 ? 5 : 3;
	vector<KeyPoint>potentialPts3;
	vector<int> potentialBallType;
	for (int kk = 0; kk < potentialPts2.size(); kk++)
	{
		KeyPoint kpt = potentialPts2[kk];
		int startX = kpt.pt.x - bw, startY = kpt.pt.y - bw;
		int stopX = kpt.pt.x + bw, stopY = kpt.pt.y + bw;

		double r = 0, g = 0, b = 0;
		for (int jj = startY; jj < stopY; jj++)
		{
			for (int ii = startX; ii < stopX; ii++)
			{
				b += Img[ii + jj*width], //b
					g += Img[ii + jj*width + length],//g
					r += Img[ii + jj*width + 2 * length];//r
			}
		}
		if (r>1.2*g && r>1.2*b)
			potentialPts3.push_back(kpt), potentialBallType.push_back(0);
		if (g > 1.2*r && g > 1.2*b)
			potentialPts3.push_back(kpt), potentialBallType.push_back(1);
		if (b > 1.2*g && b > 1.2*r)
			potentialPts3.push_back(kpt), potentialBallType.push_back(2);
	}
	if (potentialPts3.size() < 3)
		printf("Cannot detect all the balls in %s\n", ImgName);

	//Refine the detection
	double *DImg = new double[width*height];
	GrabImage(ImgName, DImg, width, height, 1); //Get gray-scale imge

	int InterpAlgo = 1;
	double *ImgPara = new double[width*height];
	Generate_Para_Spline(DImg, ImgPara, width, height, InterpAlgo);

	double *Znssd_reqd = new double[9 * PatternLength*nscales*nscales];
	pointsToBeRemoved.clear();
	for (int kk = 0; kk < potentialPts3.size(); kk++)
	{
		int PatternSize = potentialPts3[kk].size, hsubset = PatternSize / 2, PatternLength = PatternSize*PatternSize;
		synthesize_concentric_circles_mask(maskSmooth, IntensityProfile, PatternSize, 1, PatternSize, RingInfo, 0, 1);
		//SaveDataToImage("C:/temp/mask.png", maskSmooth, PatternSize, PatternSize);

		Point2i POI(potentialPts3[kk].pt.x, potentialPts3[kk].pt.y);
		double zncc = TMatchingSuperCoarse(maskSmooth, PatternSize, hsubset - 1, DImg, width, height, 1, POI, 5, 0.5);
		if (zncc < thresh)
		{
			printf("Cannot refine the %d balls in %s\n", kk + 1, ImgName);
			pointsToBeRemoved.push_back(kk);
			continue;
		}

		Point2d pt = POI;
		zncc = TMatchingFine_ZNCC(maskSmooth, PatternSize, hsubset - 1, ImgPara, width, height, 1, pt, 0, 1, thresh, InterpAlgo, Znssd_reqd);
		if (zncc < thresh)
		{
			printf("Cannot refine the %d balls in %s\n", kk + 1, ImgName);
			pointsToBeRemoved.push_back(kk);
		}
		else
			potentialPts3[kk].pt.x = pt.x, potentialPts3[kk].pt.y = pt.y;
	}
	delete[]maskSmooth, delete[]DImg, delete[]ImgPara, delete[]Znssd_reqd;

	for (int ii = 0; ii < potentialPts3.size(); ii++)
	{
		int jj;
		for (jj = 0; jj < pointsToBeRemoved.size(); jj++)
			if (ii == pointsToBeRemoved[jj])
				break;
		if (jj == pointsToBeRemoved.size())
			kpts.push_back(potentialPts3[ii]), ballType.push_back(potentialBallType[ii]);
	}

	if (visualize)
	{
		Mat ref_gray = Mat::zeros(height, width, CV_8UC1);
		for (int ii = 0; ii < width*height; ii++)
			ref_gray.data[ii] = imgpyrad.ImgPyrImg[0][ii];
		cv::cvtColor(ref_gray, ref_gray, CV_GRAY2BGR);
		for (int ii = 0; ii < kpts.size(); ii++)
		{
			KeyPoint kpt = kpts[ii];
			int startX = kpt.pt.x - kpt.size / 2, startY = kpt.pt.y - kpt.size / 2;
			int stopX = kpt.pt.x + kpt.size / 2, stopY = kpt.pt.y + kpt.size / 2;
			circle(ref_gray, Point2i(kpt.pt.x, kpt.pt.y), 2, CV_RGB(0, 255, 0), 2);
			rectangle(ref_gray, Point2i(startX, startY), cv::Point(stopX, stopY), CV_RGB(0, 255, 0), 2);
		}
		cvNamedWindow("result", CV_WINDOW_NORMAL);
		imshow("result", ref_gray); waitKey();
	}

	return 0;
}

int CleanUpCheckerboardCorner(char *Path, int startF, int stopF)
{
	char Fname[200];
	vector<int> GoodId;
	vector<double> X, Y;

	//Scattered points signify outliers
	int t;
	double u, v, mx, my, vx, vy, v2, distance;
	for (int fid = startF; fid < stopF; fid++)
	{
		sprintf(Fname, "%s/Corner/%d.txt", Path, fid); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("Cannot load %s\n", Fname);
			continue;
		}
		while (fscanf(fp, "%d %lf %lf ", &t, &u, &v) != EOF)
			X.push_back(u), Y.push_back(v);
		fclose(fp);

		mx = MeanArray(X), my = MeanArray(Y);
		vx = VarianceArray(X, mx), vy = VarianceArray(Y, my);
		v2 = sqrt(vx*vx + vy*vy);

		sprintf(Fname, "%s/Corner/_%d.txt", Path, fid); fp = fopen(Fname, "w+");
		for (int ii = 0; ii < X.size(); ii++)
		{
			distance = sqrt(pow(X[ii] - mx, 2) + pow(Y[ii] - my, 2));
			if (distance < 3 * v2)
				fprintf(fp, "%.3f %.3f\n", X[ii], Y[ii]);
		}
		fclose(fp);

		GoodId.clear(), X.clear(), Y.clear();
	}

	return 0;
}
int ComputeAverageImage(char *Path, unsigned char *MeanImg, int width, int height, int camID, int panelID, int startF, int stopF)
{
	char Fname[200];
	int length = width*height;

	sprintf(Fname, "%s/%02d_%02d.png", Path, panelID, camID);
	if (GrabImage(Fname, MeanImg, width, height, 3, true))
		return 0;

	float *Mean = new float[length * 3];
	for (int ii = 0; ii < length * 3; ii++)
		Mean[ii] = 0.0f;

	int count = 0;
	float	*Img = new float[length * 3];
	for (int frameID = startF; frameID <= stopF; frameID++)
	{
		sprintf(Fname, "%s/%08d/%08d_%02d_%02d.png", Path, frameID, frameID, panelID, camID);
		if (!GrabImage(Fname, Img, width, height, 3, true))
			continue;

		count++;
		for (int kk = 0; kk < 3; kk++)
			for (int jj = 0; jj < height; jj++)
				for (int ii = 0; ii < width; ii++)
					Mean[ii + jj*width + kk*length] += Img[ii + jj*width + kk*length];
	}

	if (count < 10)
	{
		//#pragma omp critical
		//printf("Cannot gather sufficient statistic for Cam %d Panel %d\n", camID, panelID);
		return 1;
	}

	for (int ii = 0; ii < length * 3; ii++)
		MeanImg[ii] = (unsigned char)(int)(Mean[ii] / count + 0.5);

	sprintf(Fname, "%s/%02d_%02d.png", Path, panelID, camID);
	SaveDataToImage(Fname, MeanImg, width, height, 3);

	delete[]Mean;
	return 0;
}
int DetectRedLaserCorrelationMultiScale(char *ImgName, int width, int height, unsigned char *MeanImg, vector<Point2d> &kpts, double sigma, int PatternSize, int nscales, int NMS_BW, double thresh, bool visualize, unsigned char *ColorImg, float *colorResponse, double *DImg, double *ImgPara, double *maskSmooth, double *Znssd_reqd)
{
	int length = width*height;

	bool createMem = false;
	if (ColorImg == NULL)
	{
		createMem = true;
		ColorImg = new unsigned char[length * 3];
		colorResponse = new float[width*height];
		DImg = new double[width*height];
		ImgPara = new double[width*height];
		Znssd_reqd = new double[9 * PatternSize*PatternSize];
		maskSmooth = new double[PatternSize*PatternSize*nscales];
	}

	if (createMem)
	{
		Mat view = imread(ImgName);
		if (view.data == NULL)
		{
			//cout << "Cannot load: " << ImgName << endl;
			return 1;
		}
		for (int kk = 0; kk < 3; kk++)
		{
			for (int jj = 0; jj < height; jj++)
			{
				for (int ii = 0; ii < width; ii++)
				{
					ColorImg[ii + jj*width + kk*length] = view.data[3 * ii + jj * 3 * width + kk];
				}
			}
		}
	}

	//Find places with red color through NMS
	float r, g, b;
	for (int jj = NMS_BW; jj < height - NMS_BW; jj++)
	{
		for (int ii = NMS_BW; ii < width - NMS_BW; ii++)
		{
			r = ColorImg[ii + jj*width + 2 * length] - MeanImg[ii + jj*width + 2 * length],
				g = ColorImg[ii + jj*width + length] - MeanImg[ii + jj*width + length],
				b = ColorImg[ii + jj*width] - MeanImg[ii + jj*width];
			colorResponse[ii + jj*width] = 2.0*(int)r - (int)g - (int)b;
		}
	}

	bool breakflag;
	for (int jj = NMS_BW; jj < height - NMS_BW; jj++)
	{
		for (int ii = NMS_BW; ii < width - NMS_BW; ii++)
		{
			breakflag = false;
			if (colorResponse[ii + jj*width] < 20)
				colorResponse[ii + jj*width] = 0.0;
			else
			{
				for (int j = -NMS_BW; j <= NMS_BW && !breakflag; j++)
				{
					for (int i = -NMS_BW; i <= NMS_BW&& !breakflag; i++)
					{
						if (i == 0 && j == 0)
							continue;
						if (ii + i< 0 || ii + i>width || jj + j < 0 || jj + j>height)
							continue;
						if (colorResponse[ii + jj*width] <= colorResponse[(ii + i) + (jj + j)*width])
						{
							colorResponse[ii + jj*width] = 0.0;
							breakflag = true;
							break;
						}
					}
				}
			}
		}
	}

	//find points left:
	vector<Point2d> redPoints;
	for (int jj = NMS_BW; jj < height - NMS_BW; jj++)
	{
		for (int ii = NMS_BW; ii < width - NMS_BW; ii++)
		{
			if (colorResponse[ii + jj*width] >0.0)
				redPoints.push_back(Point2d(ii, jj));
		}
	}

	if (visualize)
	{
		Mat view = imread(ImgName);
		if (view.data == NULL)
			cout << "Cannot load: " << ImgName << endl;
		else
		{
			for (int ii = 0; ii < redPoints.size(); ii++)
				rectangle(view, Point2i(redPoints[ii].x - 10, redPoints[ii].y - 10), cv::Point(redPoints[ii].x + 10, redPoints[ii].y + 10), CV_RGB(0, 255, 0), 2);
			cvNamedWindow("result", CV_WINDOW_NORMAL);
			imshow("result", view); waitKey();
		}
	}

	//build template
	int PatternLength = PatternSize*PatternSize;
	int IntensityProfile[] = { 10, 240 };
	int *hsubset = new int[nscales];
	double *RingInfo = new double[nscales];
	double maxScale = 0.9, minScale = 0.1, step = (maxScale - minScale) / nscales;
	for (int ii = 0; ii < nscales; ii++)
		RingInfo[ii] = maxScale - ii*step, hsubset[ii] = RingInfo[ii] / 2 * PatternSize;

	//double *maskSmooth = new double[PatternLength*nscales];
	for (int ii = 0; ii < nscales; ii++)
	{
		if (hsubset[ii] > 10)
			synthesize_concentric_circles_mask(maskSmooth + PatternLength*ii, IntensityProfile, PatternSize, 1, PatternSize, &RingInfo[ii], 0, 1);
		else
		{
			Gaussian(maskSmooth + PatternLength*ii, hsubset[ii], PatternSize);//LaplacianOfGaussian(maskSmooth + PatternLength*ii, hsubset[ii], PatternSize);
			RescaleMat(maskSmooth + PatternLength*ii, 0, 255, PatternLength);
		}

		//sprintf(Fname, "C:/temp/mask_%d.png", ii + 1);	SaveDataToImage(Fname, maskSmooth + PatternLength*ii, PatternSize, PatternSize);
	}

	//Refine the detection
	int InterpAlgo = 1;
	GrabImage(ImgName, DImg, width, height, 1, true); //Get gray-scale imge
	Generate_Para_Spline(DImg, ImgPara, width, height, InterpAlgo);

	Point2i POI;
	double zncc, bestzncc = 0.0;
	int bestscale = -1, bestPattern = -1;
	for (int kk = 0; kk < redPoints.size(); kk++)
	{
		//find the scale with the best response
		for (int jj = 0; jj < nscales; jj++)
		{
			POI.x = redPoints[kk].x, POI.y = redPoints[kk].y;
			zncc = TMatchingSuperCoarse(maskSmooth + jj*PatternLength, PatternSize, hsubset[jj], DImg, width, height, 1, POI, 3, 0.5);
			if (zncc > bestzncc)
				bestzncc = abs(zncc), bestscale = hsubset[jj], bestPattern = jj;
		}
		if (bestzncc < thresh)
			continue;

		Point2d pt = POI;
		bestzncc = TMatchingFine_ZNCC(maskSmooth + bestPattern*PatternLength, PatternSize, 2 * (bestscale + 2) > PatternSize ? bestscale : bestscale + 2, ImgPara, width, height, 1, pt, 0, 1, thresh, InterpAlgo, Znssd_reqd);
		if (bestzncc > thresh)
			kpts.push_back(pt);
	}

	if (createMem)
		delete[]ColorImg, delete[]colorResponse, delete[]DImg, delete[] ImgPara, delete[]Znssd_reqd, delete[]maskSmooth;

	//if (kpts.size() == 0)
	//	printf("Cannot find the laser point in %s\n", ImgName);

	if (visualize)
	{
		Mat view = imread(ImgName);
		if (view.data == NULL)
			cout << "Cannot load: " << ImgName << endl;
		else
		{
			for (int ii = 0; ii < kpts.size(); ii++)
				rectangle(view, Point2i(kpts[ii].x - 5, kpts[ii].y - 5), cv::Point(kpts[ii].x + 5, kpts[ii].y + 5), CV_RGB(0, 255, 0), 1);
			cvNamedWindow("result", CV_WINDOW_NORMAL);
			imshow("result", view); waitKey();
		}
	}

	return 0;
}


int RefineCheckBoardDetection2(char *Path, int viewID, int startF, int stopF)
{
	char Fname[200];

	//sprintf(Fname, "%s/%d", Path, viewID);
	//int checkerSize = 18; double znccThresh = 0.93;
	//CornerDetectorDriver(Fname, checkerSize, znccThresh, startF, stopF, 1280, 720);
	const int npts = 84;
	int width = 1280, height = 720, length = width*height, nchannels = 1;
	int interpAlgo = 1, hsubset = 10, searchArea = 1;
	double ZNCCThresh = 0.9;

	vector<double> PatternAngles;
	for (int ii = 0; ii < 9; ii++)
		PatternAngles.push_back(10 * ii);


	double *Img = new double[length*nchannels], *Para = new double[length*nchannels];

	int count, nptsI;
	double u, v;
	Point2d cvPts[npts];
	vector<ImgPtEle> *AllPts = NULL;
	for (int fid = startF; fid < stopF; fid++)
	{
		sprintf(Fname, "%s/%d/Corner/CV_%d.txt", Path, viewID, fid); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
			continue;
		count = 0;
		while (fscanf(fp, "%lf %lf", &u, &v) != EOF)
			cvPts[count] = Point2d(u, v), count++;
		fclose(fp);

		if (AllPts == NULL)
			AllPts = new vector<ImgPtEle>[npts];

		//refine those corners with correlation
		sprintf(Fname, "%s/%d/%d.png", Path, viewID, fid); GrabImage(Fname, Img, width, height, nchannels);
		for (int kk = 0; kk < nchannels; kk++)
			Generate_Para_Spline(Img + kk*length, Para + kk*length, width, height, interpAlgo);

		nptsI = npts;
		RefineCornersFromInit(Para, width, height, nchannels, cvPts, nptsI, PatternAngles, hsubset, hsubset, searchArea, ZNCCThresh - 0.3, ZNCCThresh, interpAlgo);

		sprintf(Fname, "%s/%d/Corner/%d.txt", Path, viewID, fid); fp = fopen(Fname, "w+");
		for (int ii = 0; ii < npts; ii++)
			fprintf(fp, "%.3f %.3f \n", cvPts[ii].x, cvPts[ii].y);
		fclose(fp);

		for (int ii = 0; ii < nptsI; ii++)
		{
			ImgPtEle impt; impt.frameID = fid;
			if (cvPts[ii].x > 1 && cvPts[ii].y > 1 && cvPts[ii].x < width - 1 && cvPts[ii].y < height - 1)
			{
				impt.pt2D = cvPts[ii];
				AllPts[ii].push_back(impt);
			}
		}
	}

	sprintf(Fname, "%s/Track2D", Path), makeDir(Fname);
	sprintf(Fname, "%s/Track2D/%d.txt", Path, viewID); FILE *fp = fopen(Fname, "w+");
	for (int ii = 0; ii < npts; ii++)
	{
		fprintf(fp, "%d %d ", ii, AllPts[ii].size());
		for (int jj = 0; jj < AllPts[ii].size(); jj++)
			fprintf(fp, "%d %.3f %.3f ", AllPts[ii][jj].frameID, AllPts[ii][jj].pt2D.x, AllPts[ii][jj].pt2D.y);
		fprintf(fp, "%\n");
	}
	fclose(fp);

	return 0;
}
int CleanCheckBoardDetection3(char *Path, int viewID, int startF, int stopF)
{
	char Fname[200];

	//sprintf(Fname, "%s/%d", Path, viewID);
	//int checkerSize = 18; double znccThresh = 0.93;
	//CornerDetectorDriver(Fname, checkerSize, znccThresh, startF, stopF, 1280, 720);

	//Merge points 
	int npts = 0;
	double u, v;
	vector<int> goodId;
	vector<Point2d> cvPts, TemplPts;
	vector<ImgPtEle> *AllPts = NULL;
	for (int fid = startF; fid < stopF; fid++)
	{
		sprintf(Fname, "%s/%d/Corner/CV2_%d.txt", Path, viewID, fid); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
			continue;
		while (fscanf(fp, "%lf %lf", &u, &v) != EOF)
			cvPts.push_back(Point2d(u, v));
		fclose(fp);

		if (AllPts == NULL)
			npts = cvPts.size(), AllPts = new vector<ImgPtEle>[cvPts.size()];

		sprintf(Fname, "%s/%d/Corner/CV2_%d.txt", Path, viewID, fid); fp = fopen(Fname, "r");
		if (fp == NULL)
			continue;
		while (fscanf(fp, "%lf %lf", &u, &v) != EOF)
			TemplPts.push_back(Point2d(u, v));
		fclose(fp);

		int bestID;
		double distance2, mindistance2;
		for (int ii = 0; ii < cvPts.size(); ii++)
		{
			bestID = -1, mindistance2 = 9e9;
			for (int jj = 0; jj < TemplPts.size(); jj++)
			{
				distance2 = pow(cvPts[ii].x - TemplPts[jj].x, 2) + pow(cvPts[ii].y - TemplPts[jj].y, 2);
				if (distance2 < mindistance2)
				{
					mindistance2 = distance2;
					bestID = jj;
				}
			}

			if (mindistance2 < 9)
			{
				cvPts[ii] = TemplPts[bestID];
				goodId.push_back(ii);
			}
			else
			{
				cvPts[ii] = Point2d(0, 0);
				goodId.push_back(-1);
			}
		}

		for (int ii = 0; ii < cvPts.size(); ii++)
		{
			ImgPtEle impt; impt.frameID = fid;
			if (goodId[ii] >-1)
			{
				impt.pt2D = cvPts[ii];
				AllPts[ii].push_back(impt);
			}
		}
		goodId.clear(), cvPts.clear(), TemplPts.clear();
	}

	sprintf(Fname, "%s/Track2D", Path), makeDir(Fname);
	sprintf(Fname, "%s/Track2D/%d.txt", Path, viewID); FILE *fp = fopen(Fname, "w+");
	for (int ii = 0; ii < npts; ii++)
	{
		fprintf(fp, "%d %d ", ii, AllPts[ii].size());
		for (int jj = 0; jj < AllPts[ii].size(); jj++)
			fprintf(fp, "%d %.3f %.3f ", AllPts[ii][jj].frameID, AllPts[ii][jj].pt2D.x - 1, AllPts[ii][jj].pt2D.y - 1);//matlab
		//fprintf(fp, "%d %.3f %.3f ", AllPts[ii][jj].frameID, AllPts[ii][jj].pt2D.x, AllPts[ii][jj].pt2D.y );//C++
		fprintf(fp, "%\n");
	}
	fclose(fp);

	return 0;
}
int CleanCheckBoardDetection(char *Path, int viewID, int startF, int stopF)
{
	char Fname[200];

	int npts = 0;
	double u, v;
	vector<Point2d> cvPts;
	vector<ImgPtEle> *AllPts = NULL;
	for (int fid = startF; fid <= stopF; fid++)
	{
		sprintf(Fname, "%s/%d/Corner/CV2_%d.txt", Path, viewID, fid); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
			continue;
		while (fscanf(fp, "%lf %lf", &u, &v) != EOF)
			cvPts.push_back(Point2d(u, v));
		fclose(fp);

		if (AllPts == NULL)
			npts = cvPts.size(), AllPts = new vector<ImgPtEle>[cvPts.size()];

		for (int ii = 0; ii < cvPts.size(); ii++)
		{
			ImgPtEle impt; impt.frameID = fid;
			impt.pt2D = cvPts[ii];
			AllPts[ii].push_back(impt);
		}
		cvPts.clear();
	}

	sprintf(Fname, "%s/Track2D", Path), makeDir(Fname);
	sprintf(Fname, "%s/Track2D/%d.txt", Path, viewID); FILE *fp = fopen(Fname, "w+");
	for (int ii = 0; ii < npts; ii++)
	{
		fprintf(fp, "%d %d ", ii, AllPts[ii].size());
		for (int jj = 0; jj < AllPts[ii].size(); jj++)
			fprintf(fp, "%d %.8f %.8f ", AllPts[ii][jj].frameID, AllPts[ii][jj].pt2D.x - 1, AllPts[ii][jj].pt2D.y - 1);//matlab
		//fprintf(fp, "%d %.3f %.3f ", AllPts[ii][jj].frameID, AllPts[ii][jj].pt2D.x, AllPts[ii][jj].pt2D.y );//C++
		fprintf(fp, "%\n");
	}
	fclose(fp);

	return 0;
}
void CheckBoardMatchingDriver(char *Path, int viewID, int fid)
{
	char Fname[200];

	int width = 1280, height = 720, PatternW = 290, PatternH = 184, length = width*height, PatternL = PatternH*PatternW, nchannels = 1;
	int interpAlgo = 1;
	double *Img = new double[length], *Pattern = new double[PatternL];

	sprintf(Fname, "%s/checkerboard.png", Path);	GrabImage(Fname, Pattern, PatternW, PatternH, 1);
	sprintf(Fname, "%s/%d/%d.png", Path, viewID, fid); GrabImage(Fname, Img, width, height, 1);

	vector<Point2f> ImgPts; ImgPts.push_back(Point2f(1133.6, 262.2));
	ImgPts.push_back(Point2f(1264.7, 301.8));
	ImgPts.push_back(Point2f(1243.5, 380.3));
	ImgPts.push_back(Point2f(1109.4, 336.6));

	vector<Point2f> TempPts; TempPts.push_back(Point2f(19.5, 19.5));
	TempPts.push_back(Point2f(239.5, 19.5));
	TempPts.push_back(Point2f(239.5, 139.5));
	TempPts.push_back(Point2f(19.5, 139.5));

	Mat mH = findHomography(TempPts, ImgPts);
	cout << mH << endl;

	double H[9] = { mH.at<double>(0, 0), mH.at<double>(0, 1), mH.at<double>(0, 2),
		mH.at<double>(1, 0), mH.at<double>(1, 1), mH.at<double>(1, 2),
		mH.at<double>(2, 0), mH.at<double>(2, 1), mH.at<double>(2, 2) };

	double *oImg = new double[PatternL];
	double *iPara = new double[length*nchannels];

	TransformImage(oImg, PatternW, PatternH, Img, width, height, H, 1, 1, iPara);
	ShowDataAsImage("Img", oImg, PatternW, PatternH, 1);

	return;
}
int DetectBalls(char *Path, int camID, const int startFrame, const int stopFrame, int search_area = 10, double threshold = 0.75)
{
	char Fname[100];
	int width = 1920, height = 1080, nchannels = 3, length = width*height, patternSizeMax = 50;
	double *Img1 = new double[length * 3];
	double *Para = new double[length * 3];
	double *PatternR = new double[patternSizeMax *patternSizeMax * 3];
	double *PatternG = new double[patternSizeMax *patternSizeMax * 3];
	double *PatternB = new double[patternSizeMax *patternSizeMax * 3];

	int t1, t2, t3;
	Point2i pti;
	Point2d pts[241 * 3], pt1, pt2, pt3;
	sprintf(Fname, "%s/%d/pts.txt", Path, camID);  FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return 1;
	}
	while (fscanf(fp, "%d %lf %lf %d %lf %lf %d %lf %lf", &t1, &pt1.x, &pt1.y, &t2, &pt2.x, &pt2.y, &t3, &pt3.x, &pt3.y) != EOF)
		pts[3 * t1] = pt1, pts[3 * t1 + 1] = pt2, pts[3 * t1 + 2] = pt3;
	fclose(fp);

	int patternSizeR, patternSizeG, patternSizeB;
	sprintf(Fname, "%s/%d/Red.png", Path, camID); GrabImage(Fname, PatternR, patternSizeR, patternSizeR, nchannels, true);
	sprintf(Fname, "%s/%d/Green.png", Path, camID); GrabImage(Fname, PatternG, patternSizeG, patternSizeG, nchannels, true);
	sprintf(Fname, "%s/%d/Blue.png", Path, camID); GrabImage(Fname, PatternB, patternSizeB, patternSizeB, nchannels, true);
	int hsubsetR = patternSizeR / 2 - 1, hsubsetG = patternSizeG / 2 - 1, hsubsetB = patternSizeB / 2 - 1;

	double score, start = omp_get_wtime();
	for (int frameID = startFrame; frameID <= stopFrame; frameID++)
	{
		printf("Working on %d. Time elapsed %.2f ....", frameID, omp_get_wtime() - start);
		sprintf(Fname, "%s/%d/%d.png", Path, camID, frameID); GrabImage(Fname, Img1, width, height, nchannels, true);
		for (int kk = 0; kk < nchannels; kk++)
			Generate_Para_Spline(Img1 + kk*length, Para + kk*length, width, height, 1);

		int detected = 0;
		pt1 = pts[3 * frameID]; pti = pt1;
		score = TMatchingSuperCoarse(PatternR, patternSizeR, hsubsetR, Img1, width, height, nchannels, pti, search_area, threshold);
		if (score < threshold)
			pts[3 * frameID] = Point2d(-1, -1);
		else
		{
			pt1 = pti;
			score = TMatchingFine_ZNCC(PatternR, patternSizeR, hsubsetR, Para, width, height, nchannels, pt1, 0, 1, threshold, 1);
			if (score < threshold)
				pts[3 * frameID] = Point2d(-1, -1);
			else
				pts[3 * frameID] = pt1, printf("got Red..."), detected++;
		}

		pt2 = pts[3 * frameID + 1]; pti = pt2;
		score = TMatchingSuperCoarse(PatternG, patternSizeG, hsubsetG, Img1, width, height, nchannels, pti, search_area, threshold);
		if (score < threshold)
			pts[3 * frameID + 1] = Point2d(-1, -1);
		else
		{
			pt2 = pti;
			score = TMatchingFine_ZNCC(PatternG, patternSizeG, hsubsetG, Para, width, height, nchannels, pt2, 0, 1, threshold, 1);
			if (score < threshold)
				pts[3 * frameID + 1] = Point2d(-1, -1);
			else
				pts[3 * frameID + 1] = pt2, printf("got Green..."), detected++;

		}

		pt3 = pts[3 * frameID + 2]; pti = pt3;
		score = TMatchingSuperCoarse(PatternB, patternSizeB, hsubsetB, Img1, width, height, nchannels, pti, search_area, threshold);
		if (score < threshold)
			pts[3 * frameID + 2] = Point2d(-1, -1);
		else
		{
			pt3 = pti;
			score = TMatchingFine_ZNCC(PatternB, patternSizeB, hsubsetB, Para, width, height, nchannels, pt3, 0, 1, threshold, 1);
			if (score < threshold)
				pts[3 * frameID + 2] = Point2d(-1, -1);
			else
				pts[3 * frameID + 2] = pt3, printf("got Blue..."), detected++;
		}
		if (detected == 0)
			printf("CAUTION!");
		printf("\n");
	}

	sprintf(Fname, "%s/%d/Rpts.txt", Path, camID);  fp = fopen(Fname, "w+");
	for (int ii = 0; ii < 241; ii++)
		fprintf(fp, "%d %f %f %f %f %f %f\n", ii, pts[3 * ii].x, pts[3 * ii].y, pts[3 * ii + 1].x, pts[3 * ii + 1].y, pts[3 * ii + 2].x, pts[3 * ii + 2].y);
	fclose(fp);

	return 0;
}
int TrajectoryTrackingDriver(char *Path)
{
	char Fname[512];
	int FrameOffset[8] = { 0, -1, 15, 1, 8, 3, 5, -2 };
	sprintf(Fname, "%s/Track2D", Path), makeDir(Fname);
	for (int viewID = 0; viewID < 8; viewID++)
	{
		sprintf(Fname, "%s/Track2D/%d.txt", Path, viewID); FILE *fp = fopen(Fname, "w+"); fclose(fp);
	}
	ReadCorresAndRunTracking(Path, 8, 70, 70, 150, FrameOffset, 4);
	ReadCorresAndRunTracking(Path, 8, 90, 70, 150, FrameOffset, 4);
	ReadCorresAndRunTracking(Path, 8, 110, 70, 150, FrameOffset, 4);
	ReadCorresAndRunTracking(Path, 8, 130, 70, 150, FrameOffset, 4);

	CleanUp2DTrackingByGradientConsistency(Path, 8, 939);
	DownSampleTracking(Path, 8, 939, 4);
	return 0;
}

#define HOMO_VECTOR(H, x, y)\
    H.at<float>(0,0) = (float)(x);\
    H.at<float>(1,0) = (float)(y);\
    H.at<float>(2,0) = 1.;

#define GET_HOMO_VALUES(X, x, y)\
    (x) = static_cast<float> (X.at<float>(0,0)/X.at<float>(2,0));\
    (y) = static_cast<float> (X.at<float>(1,0)/X.at<float>(2,0));

static int readWarp(string iFilename, Mat& warp, int motionType){

	// it reads from file a specific number of raw values:
	// 9 values for homography, 6 otherwise
	CV_Assert(warp.type() == CV_32FC1);
	int numOfElements;
	if (motionType == MOTION_HOMOGRAPHY)
		numOfElements = 9;
	else
		numOfElements = 6;

	int i;
	int ret_value;

	ifstream myfile(iFilename.c_str());
	if (myfile.is_open()){
		float* matPtr = warp.ptr<float>(0);
		for (i = 0; i < numOfElements; i++){
			myfile >> matPtr[i];
		}
		ret_value = 1;
	}
	else {
		cout << "Unable to open file " << iFilename.c_str() << endl;
		ret_value = 0;
	}
	return ret_value;
}
static int saveWarp(string fileName, const Mat& warp, int motionType)
{
	// it saves the raw matrix elements in a file
	CV_Assert(warp.type() == CV_32FC1);

	const float* matPtr = warp.ptr<float>(0);
	int ret_value;

	ofstream outfile(fileName.c_str());
	if (!outfile) {
		cerr << "error in saving "
			<< "Couldn't open file '" << fileName.c_str() << "'!" << endl;
		ret_value = 0;
	}
	else {//save the warp's elements
		outfile << matPtr[0] << " " << matPtr[1] << " " << matPtr[2] << endl;
		outfile << matPtr[3] << " " << matPtr[4] << " " << matPtr[5] << endl;
		if (motionType == MOTION_HOMOGRAPHY){
			outfile << matPtr[6] << " " << matPtr[7] << " " << matPtr[8] << endl;
		}
		ret_value = 1;
	}
	return ret_value;

}
static void draw_warped_roi(Mat& image, const int width, const int height, Mat& W)
{
	Point2f top_left, top_right, bottom_left, bottom_right;

	Mat  H = Mat(3, 1, CV_32F);
	Mat  U = Mat(3, 1, CV_32F);

	Mat warp_mat = Mat::eye(3, 3, CV_32F);

	for (int y = 0; y < W.rows; y++)
		for (int x = 0; x < W.cols; x++)
			warp_mat.at<float>(y, x) = W.at<float>(y, x);

	//warp the corners of rectangle

	// top-left
	HOMO_VECTOR(H, 1, 1);
	gemm(warp_mat, H, 1, 0, 0, U);
	GET_HOMO_VALUES(U, top_left.x, top_left.y);

	// top-right
	HOMO_VECTOR(H, width, 1);
	gemm(warp_mat, H, 1, 0, 0, U);
	GET_HOMO_VALUES(U, top_right.x, top_right.y);

	// bottom-left
	HOMO_VECTOR(H, 1, height);
	gemm(warp_mat, H, 1, 0, 0, U);
	GET_HOMO_VALUES(U, bottom_left.x, bottom_left.y);

	// bottom-right
	HOMO_VECTOR(H, width, height);
	gemm(warp_mat, H, 1, 0, 0, U);
	GET_HOMO_VALUES(U, bottom_right.x, bottom_right.y);

	// draw the warped perimeter
	line(image, top_left, top_right, Scalar(255, 0, 255));
	line(image, top_right, bottom_right, Scalar(255, 0, 255));
	line(image, bottom_right, bottom_left, Scalar(255, 0, 255));
	line(image, bottom_left, top_left, Scalar(255, 0, 255));
}
int TemplateMatchingECCDriver(char *Path, double *H, int matchingType, int MaxIter = 70, double termination_eps = 1e-6)
{
	string imgFile = "C:/temp/21.png";
	string tempImgFile = "C:/temp/checkerboard.png";
	string inWarpFile = "c:/temp/Hi.txt";
	string finalWarp = "C:/temp/H.txt";

	Mat inputImage = imread(imgFile, 0);
	if (inputImage.empty())
	{
		cerr << "Unable to load the inputImage" << endl;
		return -1;
	}

	Mat target_image;
	Mat template_image;

	if (tempImgFile != "")
	{
		inputImage.copyTo(target_image);
		template_image = imread(tempImgFile, 0);
		if (template_image.empty()){
			cerr << "Unable to load the template image" << endl;
			return -1;
		}
	}
	else
	{
		//apply random waro to input image
		resize(inputImage, target_image, Size(216, 216));
		Mat warpGround;
		cv::RNG rng;
		double angle;
		switch (matchingType) {
		case MOTION_TRANSLATION:
			warpGround = (Mat_<float>(2, 3) << 1, 0, (rng.uniform(10.f, 20.f)),
				0, 1, (rng.uniform(10.f, 20.f)));
			warpAffine(target_image, template_image, warpGround,
				Size(200, 200), INTER_LINEAR + WARP_INVERSE_MAP);
			break;
		case MOTION_EUCLIDEAN:
			angle = CV_PI / 30 + CV_PI*rng.uniform((double)-2.f, (double)2.f) / 180;

			warpGround = (Mat_<float>(2, 3) << cos(angle), -sin(angle), (rng.uniform(10.f, 20.f)),
				sin(angle), cos(angle), (rng.uniform(10.f, 20.f)));
			warpAffine(target_image, template_image, warpGround,
				Size(200, 200), INTER_LINEAR + WARP_INVERSE_MAP);
			break;
		case MOTION_AFFINE:

			warpGround = (Mat_<float>(2, 3) << (1 - rng.uniform(-0.05f, 0.05f)),
				(rng.uniform(-0.03f, 0.03f)), (rng.uniform(10.f, 20.f)),
				(rng.uniform(-0.03f, 0.03f)), (1 - rng.uniform(-0.05f, 0.05f)),
				(rng.uniform(10.f, 20.f)));
			warpAffine(target_image, template_image, warpGround,
				Size(200, 200), INTER_LINEAR + WARP_INVERSE_MAP);
			break;
		case MOTION_HOMOGRAPHY:
			warpGround = (Mat_<float>(3, 3) << (1 - rng.uniform(-0.05f, 0.05f)),
				(rng.uniform(-0.03f, 0.03f)), (rng.uniform(10.f, 20.f)),
				(rng.uniform(-0.03f, 0.03f)), (1 - rng.uniform(-0.05f, 0.05f)), (rng.uniform(10.f, 20.f)),
				(rng.uniform(0.0001f, 0.0003f)), (rng.uniform(0.0001f, 0.0003f)), 1.f);
			warpPerspective(target_image, template_image, warpGround,
				Size(200, 200), INTER_LINEAR + WARP_INVERSE_MAP);
			break;
		}
	}

	// initialize or load the warp matrix
	Mat warp_matrix;
	if (matchingType == 3)
		warp_matrix = Mat::eye(3, 3, CV_32F);
	else
		warp_matrix = Mat::eye(2, 3, CV_32F);

	if (inWarpFile != "")
	{
		int readflag = readWarp(inWarpFile, warp_matrix, matchingType);
		if ((!readflag) || warp_matrix.empty())
		{
			cerr << "-> Check warp initialization file" << endl << flush;
			return -1;
		}
	}
	else
		printf("\n ->Perfomarnce Warning: Identity warp ideally assumes images of similar size. If the deformation is strong, the identity warp may not \n");

	if (MaxIter > 200)
		cout << "-> Warning: too many iterations " << endl;

	if (matchingType != MOTION_HOMOGRAPHY)
		warp_matrix.rows = 2;

	double CorrelationScore = findTransformECC(template_image, target_image, warp_matrix, matchingType, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, MaxIter, termination_eps));

	if (CorrelationScore == -1)
	{
		cerr << "The execution was interrupted. The correlation value is going to be minimized." << endl;
		cerr << "Check the warp initialization and/or the size of images." << endl << flush;
	}

	// save the final warp matrix
	saveWarp(finalWarp, warp_matrix, matchingType);

	// save the final warped image
	Mat warped_image = Mat(template_image.rows, template_image.cols, CV_32FC1);
	if (matchingType != MOTION_HOMOGRAPHY)
		warpAffine(target_image, warped_image, warp_matrix, warped_image.size(), INTER_LINEAR + WARP_INVERSE_MAP);
	else
		warpPerspective(target_image, warped_image, warp_matrix, warped_image.size(), INTER_LINEAR + WARP_INVERSE_MAP);


	// display resulting images
	if (1)
	{
		namedWindow("image", WINDOW_AUTOSIZE);
		namedWindow("template", WINDOW_AUTOSIZE);
		namedWindow("warped image", WINDOW_AUTOSIZE);
		namedWindow("error (black: no error)", WINDOW_AUTOSIZE);

		moveWindow("template", 350, 350);
		moveWindow("warped image", 600, 300);
		moveWindow("error (black: no error)", 900, 300);

		// draw boundaries of corresponding regions
		Mat identity_matrix = Mat::eye(3, 3, CV_32F);

		draw_warped_roi(target_image, template_image.cols - 2, template_image.rows - 2, warp_matrix);
		draw_warped_roi(template_image, template_image.cols - 2, template_image.rows - 2, identity_matrix);

		Mat errorImage;
		subtract(template_image, warped_image, errorImage);
		double max_of_error;
		minMaxLoc(errorImage, NULL, &max_of_error);

		cout << "Press any key to exit the demo." << endl << flush;
		imshow("image", target_image);
		waitKey(200);
		imshow("template", template_image);
		waitKey(200);
		imshow("warped image", warped_image);
		waitKey(200);
		imshow("error (black: no error)", abs(errorImage) * 255 / max_of_error);
		waitKey(0);
	}

	return 0;
}


void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners, int boardType)
{
	corners.clear();

	switch (boardType)
	{
	case 1://CHESSBOARD:
	case 2://CIRCLES_GRID:
		for (int i = 0; i < boardSize.height; ++i)
			for (int j = 0; j < boardSize.width; ++j)
				corners.push_back(Point3f(float(j*squareSize), float(i*squareSize), 0));
		break;

	case 3://ASYMMETRIC_CIRCLES_GRID
		for (int i = 0; i < boardSize.height; i++)
			for (int j = 0; j < boardSize.width; j++)
				corners.push_back(Point3f(float((2 * j + i % 2)*squareSize), float(i*squareSize), 0));
		break;
	default:
		break;
	}
}
double computeReprojectionErrors(const vector<vector<Point3f> >& objectPoints, const vector<vector<Point2f> >& imagePoints, const vector<Mat>& rvecs, const vector<Mat>& tvecs, const Mat& cameraMatrix, const Mat& distCoeffs, vector<float>& perViewErrors)
{
	vector<Point2f> imagePoints2;
	int i, totalPoints = 0;
	double totalErr = 0, err;
	perViewErrors.resize(objectPoints.size());

	for (i = 0; i < (int)objectPoints.size(); ++i)
	{
		projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);
		err = norm(Mat(imagePoints[i]), Mat(imagePoints2), CV_L2);

		int n = (int)objectPoints[i].size();
		perViewErrors[i] = (float)std::sqrt(err*err / n);
		totalErr += err*err;
		totalPoints += n;
	}

	return std::sqrt(totalErr / totalPoints);
}
int CheckerBoardDetection(char *Path, int viewID, int startF, int stopF, int bw, int bh)
{
	char Fname[1024];

	// Initializations
	int elem_size, found, count;
	CvSize board_size = { bw, bh }, img_size = { 0, 0 };
	CvMemStorage* storage;
	CvSeq* image_points_seq = 0;
	CvPoint2D32f* cornerPts = 0;

	// Allocate memory
	elem_size = board_size.width*board_size.height*sizeof(cornerPts[0]);
	storage = cvCreateMemStorage(MAX(elem_size * 4, 1 << 16));
	cornerPts = (CvPoint2D32f*)cvAlloc(elem_size);
	image_points_seq = cvCreateSeq(0, sizeof(CvSeq), elem_size, storage);

	bool firsttime = true;
	IplImage *view = 0, *viewGray = 0;
	for (int fid = startF; fid < stopF; fid++)
	{
		sprintf(Fname, "%s/%d/%d.png", Path, viewID, fid);	view = cvLoadImage(Fname, 1);
		if (view == NULL)
			continue;
		if (firsttime)
		{
			viewGray = cvCreateImage(cvGetSize(view), 8, 1), firsttime = false;
			sprintf(Fname, "%s/%d/Corner", Path, viewID), makeDir(Fname);
		}

		img_size = cvGetSize(view);
		found = cvFindChessboardCorners(view, board_size, cornerPts, &count, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);

		if (found == 1)
		{
			//not so good result for low res images
			cvCvtColor(view, viewGray, CV_BGR2GRAY);
			cvFindCornerSubPix(viewGray, cornerPts, count, cvSize(11, 11), cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 100, 0.001));

			cvDrawChessboardCorners(view, board_size, cornerPts, count, found);
			cvShowImage("Detected corners", view);
			cvWaitKey(1);

			sprintf(Fname, "%s/%d/Corner/CV_%d.png", Path, viewID, fid); cvSaveImage(Fname, view);

			sprintf(Fname, "%s/%d/Corner/CV_%d.txt", Path, viewID, fid); FILE *fp = fopen(Fname, "w+");
			for (int ii = 0; ii < count; ii++)
				fprintf(fp, "%.3f %.3f\n", cornerPts[ii].x, cornerPts[ii].y);
			fclose(fp);
		}

		cvReleaseImage(&view);
	}

	return 0;
}
int SingleCameraCalibration(char *Path, int camID, int startFrame, int stopFrame, int bw, int bh, bool hasPoint, int step, float squareSize, int calibrationPattern, int width, int height, bool showUndistorsed)
{
	int calibFlag = 0;
	calibFlag |= CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5; //CV_CALIB_FIX_PRINCIPAL_POINT

	char Fname[512];
	const Scalar RED(0, 0, 255);
	Size imageSize, boardSize(bw, bh);

	vector<int>ValidFrame;
	vector<Point2f> pointBuf;
	vector<vector<Point2f> > imagePoints;
	vector<vector<Point3f> > objectPoints(1);

	vector<Mat> rvecs, tvecs;
	Mat cameraMatrix = Mat::eye(3, 3, CV_64F), distCoeffs = Mat::zeros(8, 1, CV_64F);

	//Get 2d points
	if (!hasPoint)
	{
		Mat view, viewGray;
		for (int ii = startFrame; ii <= stopFrame; ii += step)
		{
			sprintf(Fname, "%s/%d/%d.png", Path, camID, ii);
			view = imread(Fname);
			if (view.empty())
				continue;

			imageSize = view.size();  // Format input image.
			pointBuf.clear();

			bool found;
			switch (calibrationPattern) // Find feature points on the input format
			{
			case 1: //Checkerboard
				found = findChessboardCorners(view, boardSize, pointBuf, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
				break;
			case 2: //CIRCLES_GRID :
				found = findCirclesGrid(view, boardSize, pointBuf);
				break;
			case 3://ASYMMETRIC_CIRCLES_GRID:
				found = findCirclesGrid(view, boardSize, pointBuf, CALIB_CB_ASYMMETRIC_GRID);
				break;
			}

			if (found)
			{
				if (calibrationPattern == 1) // improve the found corners' coordinate accuracy for chessboard
				{
					cvtColor(view, viewGray, COLOR_BGR2GRAY);
					cornerSubPix(viewGray, pointBuf, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
				}

				imagePoints.push_back(pointBuf);
				drawChessboardCorners(view, boardSize, Mat(pointBuf), found);

				sprintf(Fname, "%s/%d/Corner/CV_%d.txt", Path, camID, ii); FILE *fp = fopen(Fname, "w+");
				for (int jj = 0; jj < bw*bh; jj++)
					fprintf(fp, "%f %f ", pointBuf[jj].x, pointBuf[jj].y);
				fclose(fp);
			}

			int baseLine = 0;
			sprintf(Fname, "Frame: %d/%d", ii / step + 1, (stopFrame - startFrame) / step);
			Size textSize = getTextSize(Fname, 1, 1, 1, &baseLine);
			Point textOrigin(view.cols - 2 * textSize.width - 10, view.rows - 2 * baseLine - 10);
			putText(view, Fname, textOrigin, 1, 1, RED);

			imshow("Image View", view);
			if (waitKey(10) == 27)
				break;

			ValidFrame.push_back(ii);
			pointBuf.clear();
		}
	}
	else
	{
		float u, v;
		for (int ii = startFrame; ii <= stopFrame; ii += step)
		{
			sprintf(Fname, "%s/%d/Corner/CV_%d.txt", Path, camID, ii); FILE *fp = fopen(Fname, "r");
			if (fp == NULL)
				continue;
			for (int jj = 0; jj < bw*bh; jj++)
			{
				fscanf(fp, "%f %f ", &u, &v);
				pointBuf.push_back(Point2f(u, v));
			}
			fclose(fp);

			imagePoints.push_back(pointBuf);

			ValidFrame.push_back(ii);
			pointBuf.clear();
		}

		imageSize.width = width, imageSize.height = height;
	}

	//Calibration routine:
	switch (calibrationPattern)//Create 3D pattern
	{
	case 1://CHESSBOARD:
	case 2://CIRCLES_GRID:
		if (0)//Matlab detection
			for (int i = 0; i < boardSize.width; ++i)
				for (int j = 0; j < boardSize.height; ++j)
					objectPoints[0].push_back(Point3f(float(i*squareSize), float(j*squareSize), 0));
		else//OpenCV detection
			for (int i = 0; i < boardSize.height; ++i)
				for (int j = 0; j < boardSize.width; ++j)
					objectPoints[0].push_back(Point3f(float(j*squareSize), float(i*squareSize), 0));
		break;

	case 3://ASYMMETRIC_CIRCLES_GRID
		for (int i = 0; i < boardSize.height; i++)
			for (int j = 0; j < boardSize.width; j++)
				objectPoints[0].push_back(Point3f(float((2 * j + i % 2)*squareSize), float(i*squareSize), 0));
		break;
	default:
		break;
	}

	objectPoints.resize(imagePoints.size(), objectPoints[0]); 	//Make sure points size match

	TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 300, DBL_EPSILON);
	double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, calibFlag, criteria);

	vector<float> reprojErrs; //all reprojection error
	double totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints, rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

#pragma omp critical
	{
		if (!(checkRange(cameraMatrix) && checkRange(distCoeffs)))
		{
			printf("Problem with camera %d calibration. Abort!", camID);
			abort();
		}
		printf("Cam %d.  Avg reprojection error: %.3f\n", camID, totalAvgErr);
		cout << cameraMatrix << "\t" << distCoeffs << endl;

		//Save Data
		sprintf(Fname, "%s/Intrinsic_%d.txt", Path, camID);	FILE *fp = fopen(Fname, "w+");
		for (int ii = 0; ii < (int)ValidFrame.size(); ii++)
			fprintf(fp, "%d 0 %d %d %.6f %.6f 0.0 %.6f %.6f  %.6f %.6f  %.6f %.6f %.6f 0.0 0.0\n", ValidFrame[ii], imageSize.width, imageSize.height, cameraMatrix.at<double>(0, 0), cameraMatrix.at<double>(1, 1), cameraMatrix.at<double>(0, 2), cameraMatrix.at<double>(1, 2),
			distCoeffs.at<double>(0), distCoeffs.at<double>(1), distCoeffs.at<double>(4), distCoeffs.at<double>(3), distCoeffs.at<double>(2)); //OpenCV tangential distortion is a bit different from mine.
		fclose(fp);

		double  Rvec[3], Rmat[9], T[3], Rgl[16], C[3];
		sprintf(Fname, "%s/CamPose_%d.txt", Path, camID); fp = fopen(Fname, "w+");
		for (int ii = 0; ii < (int)ValidFrame.size(); ii++)
		{
			for (int jj = 0; jj < 3; jj++)
				Rvec[jj] = rvecs[ii].at<double>(jj), T[jj] = tvecs[ii].at<double>(jj);

			//ceres::AngleAxisToRotationMatrix(Rvec, RTmat); //actually return the transpose of what you get with rodrigues
			//mat_transpose(RTmat, Rmat, 3, 3);
			convertRvecToRmat(Rvec, Rmat);
			GetRCGL(Rmat, T, Rgl, C);

			fprintf(fp, "%d %.16f %.16f %.16f %.1f %.16f %.16f %.16f %.1f %.16f %.16f %.16f %.1f 0.0 0.0 0.0 1.0 %.16f %.16f %.16f\n",
				ValidFrame[ii], Rgl[0], Rgl[1], Rgl[2], Rgl[3], Rgl[4], Rgl[5], Rgl[6], Rgl[7], Rgl[8], Rgl[9], Rgl[10], Rgl[11], C[0], C[1], C[2]);
		}
		fclose(fp);

		//Additional data for rolling shutter BA: vector<Point3d>  &Vxyz, vector < vector<int> > viewIdAll3D, vector<vector<Point2d> > uvAll3D,

	}

	//Visualization routine
	if (!hasPoint && showUndistorsed)
	{
		Mat view, rview, map1, map2;
		initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(), getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0), imageSize, CV_16SC2, map1, map2);

		for (int ii = startFrame; ii <= stopFrame; ii += step)
		{
			sprintf(Fname, "%s/%d/%d.png", Path, camID, ii);
			view = imread(Fname);
			if (view.empty())
				continue;

			remap(view, rview, map1, map2, INTER_LINEAR);
			imshow("Image View", rview);
			if (waitKey(500) == 27)
				break;
		}
	}

	return 0;
}





