#include <iostream>
#include <fstream>
#include <omp.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/flann/flann.hpp"
#include "ImagePro.h"
#include "VideoSequence.h"
#include "Ultility.h"
#include "Geometry.h"
#include "Visualization.h"
#include "BAutil.h"
#include "Eigen\Sparse"

using namespace std;
using namespace cv;

/*
#ifdef _DEBUG
#pragma comment(lib, "../ffmpeg/lib/avcodec.lib")
#else
#pragma comment(lib, "../ffmpeg/lib/avcodec.lib")
#endif*/

bool autoplay = false;
void AutomaticPlay(int state, void* userdata)
{
	autoplay = !autoplay;
}
int ShowSyncMem()
{
	char Fname[200], DataPATH[] = "C:/temp/Wedding";
	const int nsequences = 2, refSeq = 0;
	int seqName[3] = { 3, 5, 5 };

	int width = 1920, height = 1080, nchannels = 3;
	int frameInterval = 4, ngetframes = 300, nframes = 0;
	int WBlock = 0, HBlock = 0, nBlockX = 2;

	Sequence mySeq[nsequences];
	for (int ii = 0; ii < nsequences; ii++)
		mySeq[ii].Img = new char[width*height*nchannels*ngetframes];
	//mySeq[0].TimeAlignPara[0] = 100.0, mySeq[0].TimeAlignPara[1] = 331.0/frameInterval;
	//mySeq[1].TimeAlignPara[0] = 60.0, mySeq[1].TimeAlignPara[1] = 0.0/frameInterval;
	//mySeq[2].TimeAlignPara[0] = 60.0, mySeq[2].TimeAlignPara[1] = 661.0/frameInterval;

	mySeq[0].TimeAlignPara[0] = 60.0, mySeq[0].TimeAlignPara[1] = 0.0 / frameInterval;
	mySeq[1].TimeAlignPara[0] = 60.0, mySeq[1].TimeAlignPara[1] = 661.0 / frameInterval;
	mySeq[2].TimeAlignPara[0] = 60.0, mySeq[2].TimeAlignPara[1] = 661.0 / frameInterval;

	//Read video sequences
	width = 0, height = 0;
	nBlockX = nsequences < nBlockX ? nsequences : nBlockX;
	for (int ii = 0; ii < nsequences; ii++)
	{
		sprintf(Fname, "%s/%d.mp4", DataPATH, seqName[ii]);
		if (!GrabVideoFrame2Mem(Fname, mySeq[ii].Img, mySeq[ii].width, mySeq[ii].height, mySeq[ii].nchannels, mySeq[ii].nframes, frameInterval, ngetframes))
			return -1;
		width += mySeq[ii].width, height += mySeq[ii].height, nframes = max(mySeq[ii].nframes + (int)(mySeq[ii].TimeAlignPara[1] + 0.5), nframes);
		WBlock = max(WBlock, mySeq[ii].width), HBlock = max(HBlock, mySeq[ii].height);
	}


	//Initialize display canvas
	int nBlockY = (1.0*nsequences / nBlockX > nsequences / nBlockX) ? nsequences / nBlockX + 1 : nsequences / nBlockX;
	width = WBlock*nBlockX, height = HBlock*nBlockY;
	char *BigImg = new char[width*height*nchannels];
	char *BlackImage = new char[WBlock*HBlock*nchannels];
	for (int ii = 0; ii < width*height*nchannels; ii++)
		BigImg[ii] = (char)0;
	for (int ii = 0; ii < WBlock*HBlock*nchannels; ii++)
		BlackImage[ii] = (char)0;

	//Create display window
	int oFrameID[nsequences + 1], FrameID[nsequences + 1];
	for (int ii = 0; ii < nsequences + 1; ii++)
		oFrameID[ii] = 0, FrameID[ii] = 0;
	cvNamedWindow("VideoSequences", CV_WINDOW_NORMAL);
	cvCreateTrackbar("Global frame", "VideoSequences", &FrameID[0], nframes - 1, NULL);
	for (int ii = 0; ii < nsequences; ii++)
	{
		sprintf(Fname, "Seq %d", ii + 1);
		cvCreateTrackbar(Fname, "VideoSequences", &FrameID[ii + 1], nframes - 1, NULL);
		cvSetTrackbarPos(Fname, "VideoSequences", 0);
	}

	int BlockXID, BlockYID, clength, setframeID, setSeqFrame;
	IplImage *cvImg = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, nchannels);

	bool GlobalSlider[nsequences]; //True: global slider, false: local slider
	for (int ii = 0; ii < nsequences; ii++)
		GlobalSlider[ii] = true;

	while (waitKey(33) != 27)
	{
		for (int ii = 0; ii < nsequences; ii++)
		{
			clength = mySeq[ii].width*mySeq[ii].height*mySeq[ii].nchannels;
			BlockXID = ii%nBlockX, BlockYID = ii / nBlockX;

			if (GlobalSlider[ii])
				setframeID = FrameID[0]; //global frame
			else
				setframeID = FrameID[ii + 1];

			if (oFrameID[0] != FrameID[0])
				FrameID[ii + 1] = FrameID[0], GlobalSlider[ii] = true;

			if (oFrameID[ii + 1] != FrameID[ii + 1]) //but if local slider moves
				setframeID = FrameID[ii + 1], GlobalSlider[ii] = false;

			sprintf(Fname, "Seq %d", ii + 1);
			setSeqFrame = (int)(mySeq[ii].TimeAlignPara[0] / mySeq[refSeq].TimeAlignPara[0] * (setframeID - (int)(mySeq[ii].TimeAlignPara[1] + 0.5)) + 0.5); //setframeID-SeqFrameOffset[ii];
			printf("Sequence %d frame %d\n", ii + 1, setSeqFrame);
			if (setSeqFrame <= 0)
			{
				cvSetTrackbarPos(Fname, "VideoSequences", (int)(mySeq[ii].TimeAlignPara[1] + 0.5));// SeqFrameOffset[ii]);
				Set_Sub_Mat(BlackImage, BigImg, nchannels*mySeq[ii].width, mySeq[ii].height, nchannels*width, nchannels*BlockXID*WBlock, BlockYID*HBlock);
			}
			else if (setSeqFrame >= mySeq[ii].nframes)
			{
				cvSetTrackbarPos(Fname, "VideoSequences", (int)(mySeq[ii].TimeAlignPara[1] + 0.5) + mySeq[ii].nframes);
				Set_Sub_Mat(BlackImage, BigImg, nchannels*mySeq[ii].width, mySeq[ii].height, nchannels*width, nchannels*BlockXID*WBlock, BlockYID*HBlock);
			}
			else
			{
				oFrameID[ii + 1] = FrameID[ii + 1];
				cvSetTrackbarPos(Fname, "VideoSequences", oFrameID[ii + 1]);
				Set_Sub_Mat(mySeq[ii].Img + setSeqFrame*clength, BigImg, nchannels*mySeq[ii].width, mySeq[ii].height, nchannels*width, nchannels*BlockXID*WBlock, BlockYID*HBlock);
			}
		}
		oFrameID[0] = FrameID[0];
		ShowDataToImage("VideoSequences", BigImg, width, height, nchannels, cvImg);
	}

	cvReleaseImage(&cvImg);
	delete[]BigImg;
	delete[]BlackImage;
	return 0;
}
int ShowSyncLoad(char *DataPATH)
{
	//bad frames: 1510 1741 1765 1899
	char Fname[2000];
	const int playBackSpeed = 1, nsequences = 2, refSeq = 1;
	int seqName[nsequences] = { 1, 2 };// 1, 2, 3, 4, 5, 6, 7, 8, 9

	int WBlock = 1920, HBlock = 1080, nBlockX = 3, nchannels = 3, MaxFrames = 5500;

	//Make sure to use 59.94 instead of 60 
	Sequence mySeq[nsequences];
	/*mySeq[0].InitSeq(47.95, 4.761);
	mySeq[1].InitSeq(47.95, - 8.3183);
	mySeq[2].InitSeq(47.95,  4.761+0.80816);
	mySeq[3].InitSeq(47.95,  0.0);*/

	/*mySeq[0].InitSeq(47.95, -1);///aligned by hyunsoo
	mySeq[1].InitSeq(47.95, -14);
	mySeq[2].InitSeq(47.95,  0);
	mySeq[3].InitSeq(47.95,  -6);*/

	/*//My wedding sequences: 1 3 5 7
	mySeq[0].InitSeq(100.0,   333.3126);
	mySeq[1].InitSeq(59.94, 0);
	mySeq[2].InitSeq(59.94,  660.8185);
	mySeq[3].InitSeq(59.94,  700.2928);*/

	// soccer sequences: 
	/*mySeq[0].InitSeq(47.95, 0);
	mySeq[1].InitSeq(47.95, 72.78);
	mySeq[2].InitSeq(47.95, 110.19);
	mySeq[3].InitSeq(47.95, 90.42);
	mySeq[4].InitSeq(47.95, 90.38);
	mySeq[5].InitSeq(47.95, 89.52);
	mySeq[6].InitSeq(47.95, 79.56);
	mySeq[7].InitSeq(47.95, 90.74);
	mySeq[8].InitSeq(47.95, 89.62);*/

	mySeq[0].InitSeq(29.97, 18.08);
	mySeq[1].InitSeq(29.97, 0);

	//Read video sequences
	int width = 0, height = 0;
	nBlockX = nsequences < nBlockX ? nsequences : nBlockX;
	for (int ii = 0; ii < nsequences; ii++)
	{
		//sprintf(Fname, "%s/%d/s2.mp4", DataPATH, seqName[ii], seqName[ii]);
		width += WBlock, height += HBlock;
	}

	//Initialize display canvas
	int nBlockY = (1.0*nsequences / nBlockX > nsequences / nBlockX) ? nsequences / nBlockX + 1 : nsequences / nBlockX;
	width = WBlock*nBlockX, height = HBlock*nBlockY;
	char *BigImg = new char[width*height*nchannels];
	char *BlackImage = new char[WBlock*HBlock*nchannels], *SubImage = new char[WBlock*HBlock*nchannels];
	IplImage *cvImg = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, nchannels);

	for (int ii = 0; ii < width*height*nchannels; ii++)
		BigImg[ii] = (char)0;
	for (int ii = 0; ii < WBlock*HBlock*nchannels; ii++)
		BlackImage[ii] = (char)0;

	//Create display window
	int oFrameID[nsequences + 1], FrameID[nsequences + 1];
	for (int ii = 0; ii < nsequences + 1; ii++)
		oFrameID[ii] = 0, FrameID[ii] = 0;
	cvNamedWindow("VideoSequences", CV_WINDOW_NORMAL);
	cvCreateTrackbar("Global frame", "VideoSequences", &FrameID[0], MaxFrames - 1, NULL);
	for (int ii = 0; ii < nsequences; ii++)
	{
		sprintf(Fname, "Seq %d", ii + 1);
		cvCreateTrackbar(Fname, "VideoSequences", &FrameID[ii + 1], MaxFrames - 1, NULL);
		cvSetTrackbarPos(Fname, "VideoSequences", 0);
	}
	char* nameb1 = "Play/Stop";
	createButton(nameb1, AutomaticPlay, nameb1, CV_CHECKBOX, 1);

	int BlockXID, BlockYID, setframeID, setSeqFrame, same, noUpdate, swidth, sheight;
	bool GlobalSlider[nsequences]; //True: global slider, false: local slider
	for (int ii = 0; ii < nsequences; ii++)
		GlobalSlider[ii] = true;

	while (waitKey(17) != 27)
	{
		noUpdate = 0;
		for (int ii = 0; ii < nsequences; ii++)
		{
			BlockXID = ii%nBlockX, BlockYID = ii / nBlockX;

			same = 0;
			if (GlobalSlider[ii])
				setframeID = FrameID[0]; //global frame
			else
				setframeID = FrameID[ii + 1];

			if (oFrameID[0] != FrameID[0])
				FrameID[ii + 1] = FrameID[0], GlobalSlider[ii] = true;
			else
				same += 1;

			if (oFrameID[ii + 1] != FrameID[ii + 1]) //but if local slider moves
				setframeID = FrameID[ii + 1], GlobalSlider[ii] = false;
			else
				same += 1;

			sprintf(Fname, "Seq %d", ii + 1);
			setSeqFrame = (int)(mySeq[ii].TimeAlignPara[0] / mySeq[refSeq].TimeAlignPara[0] * (1.0*setframeID - mySeq[ii].TimeAlignPara[1]) + 0.5); //setframeID-SeqFrameOffset[ii];
			printf("Sequence %d frame %d\n", ii + 1, setSeqFrame);
			if (same == 2)
			{
				noUpdate++;
				continue;
			}
			if (setSeqFrame <= 0)
			{
				cvSetTrackbarPos(Fname, "VideoSequences", (int)(mySeq[ii].TimeAlignPara[1] + 0.5));
				Set_Sub_Mat(BlackImage, BigImg, nchannels*WBlock, HBlock, nchannels*width, nchannels*BlockXID*WBlock, BlockYID*HBlock);
			}
			else
			{
				oFrameID[ii + 1] = FrameID[ii + 1];
				cvSetTrackbarPos(Fname, "VideoSequences", oFrameID[ii + 1]);
				sprintf(Fname, "%s/%d/%d.png", DataPATH, seqName[ii], setSeqFrame);
				if (GrabImage(Fname, SubImage, swidth, sheight, nchannels))
					Set_Sub_Mat(SubImage, BigImg, nchannels*swidth, sheight, nchannels*width, nchannels*BlockXID*WBlock, BlockYID*HBlock);
				else
					Set_Sub_Mat(BlackImage, BigImg, nchannels*WBlock, HBlock, nchannels*width, nchannels*BlockXID*WBlock, BlockYID*HBlock);
			}
		}
		oFrameID[0] = FrameID[0];
		if (noUpdate != nsequences)
			ShowDataToImage("VideoSequences", BigImg, width, height, nchannels, cvImg);
		if (autoplay)
		{
			cvSetTrackbarPos("Global frame", "VideoSequences", FrameID[0]);
			FrameID[0] += playBackSpeed;
		}
	}

	cvReleaseImage(&cvImg);
	delete[]BigImg;
	delete[]BlackImage;
	delete[]SubImage;

	return 0;
}

int TestPnP(char *Path, int nCams, int selectedCams)
{
	char Fname[200];
	sprintf(Fname, "%s/Corpus.nvm", Path);
	Corpus corpusData;
	string nvmfile = Fname;
	if (!loadNVMLite(nvmfile, corpusData, 1920, 1080, 1))
		return 1;
	int nviews = corpusData.nCamera;

	CameraData *cameraInfo = new CameraData[nCams];
	if (ReadIntrinsicResults(Path, cameraInfo, nCams) != 0)
		return 1;
	if (selectedCams != -1)
	{
		for (int ii = 0; ii < nviews; ii++)
		{
			for (int jj = 0; jj < 9; jj++)
				corpusData.camera[ii].K[jj] = cameraInfo[selectedCams].K[jj];
			for (int jj = 0; jj < 7; jj++)
				corpusData.camera[ii].distortion[jj] = cameraInfo[selectedCams].distortion[jj];
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

	for (int ii = 0; ii < nviews; ii++)
	{
		corpusData.camera[ii].LensModel = RADIAL_TANGENTIAL_PRISM, corpusData.camera[ii].threshold = 3.0, corpusData.camera[ii].ninlierThresh = 50, corpusData.camera[ii];
		GetrtFromRT(corpusData.camera[ii].rt, corpusData.camera[ii].R, corpusData.camera[ii].T);
		GetIntrinsicFromK(corpusData.camera[ii]);
		AssembleP(corpusData.camera[ii].K, corpusData.camera[ii].R, corpusData.camera[ii].T, corpusData.camera[ii].P);
		if (true)
			for (int jj = 0; jj < 7; jj++)
				corpusData.camera[ii].distortion[jj] = 0.0;
	}
	printf("...Done\n");

	int n3D = 240, nviewsPerPt, viewID, ptsCount = 0;;
	double x, y, z, u, v;
	vector<Point3d> t3D(n3D);
	vector<Point2d> uv(n3D);
	vector<int> *viewIDAll3D = new vector<int>[n3D];
	vector<Point2d> *uvAll3D = new vector<Point2d>[n3D];

	FILE *fp = fopen("C:/temp/3D2D.txt", "r");
	while (fscanf(fp, "%lf %lf %lf %lf %lf %d ", &x, &y, &z, &u, &v, &nviewsPerPt) != EOF)
	{
		t3D.push_back(Point3d(x, y, z));
		uv.push_back(Point2d(u, v));
		for (int ii = 0; ii < nviewsPerPt; ii++)
		{
			fscanf(fp, "%d %lf %lf ", &viewID, &u, &v);
			viewIDAll3D[ptsCount].push_back(viewID);
			uvAll3D[ptsCount].push_back(Point2d(u, v));
		}
	
		ptsCount++;
		if (ptsCount > n3D-1)
			break;
	
	}
	fclose(fp);

	//Test if 3D is correct
	Point3d xyz;
	double *A = new double[6 * 50 * 2];
	double *B = new double[2 * 50 * 2];
	double *tPs = new double[12 * 50 * 2];
	bool *passed = new bool[50 * 2];
	double Ps[12 * 50 * 2];
	Point2d match2Dpts[50 * 2];

	vector<int>Inliers[1];  Inliers[0].reserve(20 * 2);
	double ProThresh = 0.99, PercentInlier = 0.25;
	int goodNDplus = 0, iterMax = (int)(log(1.0 - ProThresh) / log(1.0 - pow(PercentInlier, 2)) + 0.5); //log(1-eps) / log(1 - (inlier%)^min_pts_requires)
	for (int jj = 0; jj < n3D; jj++)
	{
		int nviewsi = viewIDAll3D[jj].size();
		Inliers[0].clear();
		for (int ii = 0; ii < nviewsi; ii++)
		{
			viewID = viewIDAll3D[jj].at(ii);
			for (int kk = 0; kk < 12; kk++)
				Ps[12 * ii + kk] = corpusData.camera[viewID].P[kk];

			match2Dpts[ii] = uvAll3D[jj].at(ii);
		}

		NviewTriangulationRANSAC(match2Dpts, Ps, &xyz, passed, Inliers, nviewsi, 1, iterMax, PercentInlier, corpusData.camera[0].threshold, A, B, tPs);
		if (passed[0])
		{
			if (pow(xyz.y - t3D.at(jj).y, 2) + pow(xyz.y - t3D.at(jj).y, 2) + pow(xyz.y - t3D.at(jj).y, 2) > 0.1)
				goodNDplus++;
		}
	}
	printf("Nbad: %d ", goodNDplus);

	return 0;
}
int main(int argc, char* argv[])
{
	char Path[] = "D:/S", Fname[200], Fname2[200];

	int nSviews = 190;
	//TestPnP(Path, 10, 6);
	//GenerateCorpusVisualWords("C:/temp/BOW", 24);
	//ComputeWordsHistogram("C:/temp/BOW", 3);

	//Undistort and compute sift for all extracted frames
	/*int nviews = 10, nframes = 2000;
	CameraData *AllViewsInfo = new CameraData[nviews];
	if (ReadIntrinsicResults(Path, AllViewsInfo, nviews) != 0)
	return 1;
	for (int ii = 0; ii < nviews; ii++)
	{
	AllViewsInfo[ii].LensModel = RADIAL_TANGENTIAL_PRISM;
	sprintf(Fname, "%s/%d", Path, ii);
	sprintf(Fname2, "%s/%d/s2.mp4", Path, ii);
	LensCorrectionVideoDriver(Fname, Fname2, AllViewsInfo[ii].K, AllViewsInfo[ii].distortion, 0, nframes, 5); //e.g. D1.png -->1.png
	}*/
	//ExtractSiftGPUfromVideoFrames(Path, nSviews, 50, 200); 

	//BatchExtractSiftGPU(Path, 3, 200);
	//BatchExtractSiftGPU(Path,6, 200);
	//GeneratePointsCorrespondenceMatrix_SiftGPU2(Path, nSviews, -1, 0.6, 1, 1, 1, 0);
	/*double start = omp_get_wtime();
	start = omp_get_wtime();
	for (int jj = 0; jj < nSviews - 1; jj++)
	#pragma omp parallel for
	for (int ii = jj + 1; ii < nSviews; ii++)
	EssentialMatOutliersRemove(Path, -1, jj, ii, 1, 0, 50, 1, 0);
	printf("Finished pruning matches ... in %.2fs\n", omp_get_wtime() - start);
	GenerateMatchingTable(Path, nSviews, -1);*/

	//BuildCorpus(Path, 10, 6, 1920, 1080, 5);
	LocalizeCameraFromCorpusDriver(Path, 1, 1, 10, 3, true);
	//LocalizeCameraFromCorpusDriver(Path, 1, 199, 10, 6, true);
	//visualizationDriver(Path, 1, 200);

	//google::InitGoogleLogging("SfM");
	//IncrementalBundleAdjustment(Path, nviews, -1, 20000);

	//vector<int>viewsID; viewsID.push_back(1), viewsID.push_back(3);
	//DisplayImageCorrespondencesDriver(Path, viewsID, -1, 3, 1.0);
	/*double offset = 0.0;
	for (int ii = 1; ii < 2; ii++)
	{
	sprintf(Fname, "%s/%d/audio.wav", Path, 1);
	sprintf(Fname2, "%s/%d/audio.wav", Path, ii + 1);
	if (SynAudio(Fname, Fname2, 47.95, 47.95, 3e6, offset, 0.25) != 0)
	printf("Not succeed\n");
	else
	printf("Succeed\n");
	}*/

	//ShowSyncLoad(Path); 
	//Sequence mySeq[1]; mySeq[0].InitSeq(59.94, 0.0);

	/*for (int ii = 2; ii <= 10; ii++)
	{
	sprintf(Fname, "%s/%d", Path, ii);
	PickStaticImagesFromVideo(Fname, "c1.mp4", 20, 15, 30, 50, true);
	BlurDetectionDriver("E:/s7/C", 241, 1920, 1080, 0.1);
	}*/

	/*const int nviews = 191;
	CameraData CameraInfo[nviews];
	vector<int> AvailViews;
	for (int ii = 0; ii < nviews; ii++)
	AvailViews.push_back(ii);
	ReadCurrentSfmInfo(Path, CameraInfo, AvailViews, NULL, 0);

	double Fmat[9];
	int selectedViews[] = { 4, 190 };
	computeFmatfromKRT(CameraInfo, 191, selectedViews, Fmat);*/

	return 0;
}

