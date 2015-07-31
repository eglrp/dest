#include <stdio.h>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include "MarkerDetector.h"

using namespace alvar;
using namespace std;
using namespace cv;

#ifdef _WIN32
#ifdef _DEBUG
#pragma comment(lib, "alvar200d.lib")
#else
#pragma comment(lib, "alvar200.lib")
#endif
#else
#pragma comment(lib, "alvar200.lib")
#endif

int LoadCalib(const char *calibfile, Camera &camera)
{
	double fx, fy, skew, u0, v0, r0, r1, t0, t1;

	FILE *fp = fopen(calibfile, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", calibfile);
		return 1;
	}
	fscanf(fp, "%lf %lf %lf %lf %lf", &fx, &fy, &skew, &u0, &v0);
	fscanf(fp, "%lf %lf %lf %lf", &r0, &r1, &t0, &t1);

	// K Intrinsic
	cvmSet(&camera.calib_K, 0, 0, fx);
	cvmSet(&camera.calib_K, 0, 1, skew);
	cvmSet(&camera.calib_K, 0, 2, u0);
	cvmSet(&camera.calib_K, 1, 0, 0.0);
	cvmSet(&camera.calib_K, 1, 1, fy);
	cvmSet(&camera.calib_K, 1, 2, v0);
	cvmSet(&camera.calib_K, 2, 0, 0.0);
	cvmSet(&camera.calib_K, 2, 1, 0.0);
	cvmSet(&camera.calib_K, 2, 2, 1.0);

	// D Distortion
	cvmSet(&camera.calib_D, 0, 0, r0);
	cvmSet(&camera.calib_D, 1, 0, r1);
	cvmSet(&camera.calib_D, 2, 0, t0);
	cvmSet(&camera.calib_D, 3, 0, t1);

	return 0;
}

int main(int argc, char** argv)
{
	/*char *Path = argv[1];
	int startF = atoi(argv[2]),
	stopF = atoi(argv[3]),
	startV = atoi(argv[4]),
	stopV = atoi(argv[5]);*/
	char Path[200] = "E:/ARTag";
	int startF = 1, stopF = 200, startV = 1, stopV = 2;

	int nvideos = stopV - startV + 1;

	//omp_set_num_threads(omp_get_max_threads());
	//#pragma omp parallel for
	for (int vid = startV; vid <= stopV; vid++)
	{
		char Fname[200];
		IplImage *image = 0;
		Camera camera;
		MarkerDetector<MarkerData> markerDetector;

		sprintf(Fname, "%s/calib_%d.txt", Path, vid);
		LoadCalib(Fname, camera);

		for (int fid = startF; fid <= stopF; fid++)
		{
			if (nvideos == 1)
				sprintf(Fname, "%s/%d.png", Path, fid);
			else
				sprintf(Fname, "%s/%d/%d.png", Path, vid, fid);
			image = cvLoadImage(Fname);

			camera.SetRes(image->width, image->height);
			

			markerDetector.SetMarkerSize(15);
			markerDetector.Detect(image, &camera, false, false);

			if (nvideos == 1)
				sprintf(Fname, "%s/%d.txt", Path, fid);
			else
				sprintf(Fname, "%s/%d/%d.txt", Path, vid, fid);
			FILE *fp = fopen(Fname, "w+");
			for (int jj = 0; jj < markerDetector.markers[0].size(); jj++)
				for (int kk = 0; kk < markerDetector.markers[0][jj].marker_corners_img.size(); kk++)
					fprintf(fp, "%.2f %.2f %d\n", markerDetector.markers[0][jj].marker_corners_img[kk].x, markerDetector.markers[0][jj].marker_corners_img[kk].y, markerDetector.markers[0][jj].data.id);
			fclose(fp);

#pragma omp critical
			if (nvideos == 1)
				printf("Frame %d: %d markers\n", fid, (int)markerDetector.markers[0].size());
			else
				printf("(%d, %d): %d markers\n", vid, fid, (int)markerDetector.markers[0].size());

			cvReleaseImage(&image);
		}
	}

	return 0;
}
