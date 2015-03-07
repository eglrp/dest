#include <iostream>
#include <fstream>
#include <omp.h>
#include <vector>
#include <cstdlib>
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
#include <direct.h>
using namespace std;
using namespace cv;

/*
#ifdef _DEBUG
#pragma comment(lib, "../ffmpeg/lib/avcodec.lib")
#else
#pragma comment(lib, "../ffmpeg/lib/avcodec.lib")
#endif*/

bool autoplay = false, saveFrame = false;
int MousePosX, MousePosY;
static void onMouse(int event, int x, int y, int, void*)
{
	if (event == EVENT_LBUTTONDBLCLK)
	{
		MousePosX = x, MousePosY = y;
		printf("Selected: %d %d\n", x, y);
		cout << "\a";
	}
}
int TriangulatePointsFromCalibratedCameras(char *Path, int distortionCorrected, int maxPts, double threshold = 2.0)
{
	char Fname[200];
	Corpus corpusData;
	sprintf(Fname, "%s/BA_Camera_AllParams_after.txt", Path);
	if (!loadBundleAdjustedNVMResults(Fname, corpusData))
		return 1;

	int nviews = corpusData.nCamera;
	for (int ii = 0; ii < nviews; ii++)
	{
		corpusData.camera[ii].threshold = threshold, corpusData.camera[ii].ninlierThresh = 50, corpusData.camera[ii];
		GetrtFromRT(corpusData.camera[ii].rt, corpusData.camera[ii].R, corpusData.camera[ii].T);
		GetIntrinsicFromK(corpusData.camera[ii]);
		AssembleP(corpusData.camera[ii].K, corpusData.camera[ii].R, corpusData.camera[ii].T, corpusData.camera[ii].P);
		if (distortionCorrected == 1)
			for (int jj = 0; jj < 7; jj++)
				corpusData.camera[ii].distortion[jj] = 0.0;
	}
	printf("...Done\n");

	sprintf(Fname, "%s/ImageList.txt", Path);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return 1;
	}
	int imgID;
	vector<int> ImageIDList;
	while (fscanf(fp, "%d ", &imgID) != EOF)
		ImageIDList.push_back(imgID);
	fclose(fp);

	int n3D = 1, viewID, ptsCount = 0;
	vector<Point3d> t3D;
	vector<Point2d> uv;
	vector<int> viewIDAll3D;
	vector<Point2d>uvAll3D;

	sprintf(Fname, "%s/Points.txt", Path);
	ofstream ofs(Fname);
	if (ofs.fail())
		cerr << "Cannot write " << Fname << endl;
	for (int npts = 0; npts < maxPts; npts++)
	{
		t3D.clear(), uv.clear(), viewIDAll3D.clear(), uvAll3D.clear();
		for (int ii = 0; ii < ImageIDList.size(); ii++)
		{
			sprintf(Fname, "%s/%d.png", Path, ImageIDList[ii]);
			cvNamedWindow("Image", CV_WINDOW_NORMAL); setMouseCallback("Image", onMouse);
			Mat Img = imread(Fname);
			if (Img.empty())
			{
				printf("Cannot load %s\n", Fname);
				return 1;
			}

			CvPoint text_origin = { Img.cols / 30, Img.cols / 30 };
			sprintf(Fname, "Point %d/%d of Image %d/%d", npts + 1, maxPts, ii + 1, ImageIDList.size());
			putText(Img, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 3.0 * 640 / Img.cols, CV_RGB(255, 0, 0), 2);
			imshow("Image", Img), waitKey(0);
			viewIDAll3D.push_back(ImageIDList[ii]);
			uvAll3D.push_back(Point2d(MousePosX, MousePosY));
			ofs << MousePosX << " " << MousePosY << endl;
		}

		//Test if 3D is correct
		ptsCount = uvAll3D.size();
		Point3d xyz;
		double *A = new double[6 * ptsCount];
		double *B = new double[2 * ptsCount];
		double *tPs = new double[12 * ptsCount];
		bool *passed = new bool[ptsCount];
		double *Ps = new double[12 * ptsCount];
		Point2d *match2Dpts = new Point2d[ptsCount];

		vector<int>Inliers[1];  Inliers[0].reserve(ptsCount * 2);
		double ProThresh = 0.99, PercentInlier = 0.25;
		int goodNDplus = 0, iterMax = (int)(log(1.0 - ProThresh) / log(1.0 - pow(PercentInlier, 2)) + 0.5); //log(1-eps) / log(1 - (inlier%)^min_pts_requires)
		int nviewsi = viewIDAll3D.size();
		Inliers[0].clear();
		for (int ii = 0; ii < nviewsi; ii++)
		{
			viewID = viewIDAll3D.at(ii);
			for (int kk = 0; kk < 12; kk++)
				Ps[12 * ii + kk] = corpusData.camera[viewID].P[kk];

			match2Dpts[ii] = uvAll3D.at(ii);
			if (corpusData.camera[viewID].LensModel == RADIAL_TANGENTIAL_PRISM)
				LensCorrectionPoint(&match2Dpts[ii], corpusData.camera[viewID].K, corpusData.camera[viewID].distortion);
			else
				FishEyeCorrectionPoint(&match2Dpts[ii], corpusData.camera[viewID].distortion[0], corpusData.camera[viewID].distortion[1], corpusData.camera[viewID].distortion[2]);
		}

		double error = NviewTriangulationRANSAC(match2Dpts, Ps, &xyz, passed, Inliers, nviewsi, 1, iterMax, PercentInlier, corpusData.camera[0].threshold, A, B, tPs);
		if (passed[0])
		{
			printf("3D: %f %f %f Error: %f\n", xyz.x, xyz.y, xyz.z, error);
			ofs << xyz.x << " " << xyz.y << " " << xyz.z << endl;
		}
	}
	ofs.close();
	return 0;
}
void AutomaticPlay(int state, void* userdata)
{
	autoplay = !autoplay;
}
void AutomaticSave(int state, void* userdata)
{
	saveFrame = !saveFrame;
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
int ShowSyncLoad(char *DataPATH, char *SavePATH, int refSeq = -1)
{
	char Fname[2000];
	const int nsequences = 9;
	int  playBackSpeed = 1, seqName[nsequences] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	int WBlock = 1280, HBlock = 720, nBlockX = 3, nchannels = 3, MaxFrames = 2500;


	/*sprintf(Fname, "%s/FaudioSync.txt", DataPATH);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
	printf("Cannot open %s\n", Fname);
	return 1;
	}
	while (fscanf(fp, "%d %lf ", &id, &offset) != EOF)
	Offset[id] = offset;
	fclose(fp);
	double Offset[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 }, offset;
	Sequence mySeq[nsequences];
	if (refSeq == -1)
	{
	for (int ii = 0; ii < nsequences; ii++)
	{
	//mySeq[ii].InitSeq(59.94, Offset[ii]);
	mySeq[ii].InitSeq(1, Offset[ii]);
	//if (Offset[ii] < 0.1)
	//refSeq = ii;
	}
	}
	refSeq = 2;*/

	double Offset[] = { 0.315, 0.000, 0.547, 0.477, 0.046, 0.225, 0.187, 0.118, 0.150 };
	//double Offset[] = { 0.501, 0.431, 0.000, 0.179 }; int seqName[nsequences] = { 2, 3, 4, 5 }
	Sequence mySeq[nsequences];
	if (refSeq == -1)
		for (int ii = 0; ii < nsequences; ii++)
			mySeq[ii].InitSeq(119.88, Offset[ii]);
	refSeq = 2;

	//Read video sequences
	int width = 0, height = 0;
	nBlockX = nsequences < nBlockX ? nsequences : nBlockX;
	for (int ii = 0; ii < nsequences; ii++)
		width += WBlock, height += HBlock;

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
	cvNamedWindow("VideoSequences2", CV_WINDOW_NORMAL);
	cvNamedWindow("Control2", CV_WINDOW_NORMAL);
	cvCreateTrackbar("Speed", "Control2", &playBackSpeed, 10, NULL);
	cvCreateTrackbar("Global frame", "Control2", &FrameID[0], MaxFrames - 1, NULL);
	for (int ii = 0; ii < nsequences; ii++)
	{
		sprintf(Fname, "Seq %d", ii + 1);
		cvCreateTrackbar(Fname, "Control2", &FrameID[ii + 1], MaxFrames - 1, NULL);
		cvSetTrackbarPos(Fname, "Control2", 0);
	}
	char* nameb1 = "Play/Stop";
	createButton(nameb1, AutomaticPlay, nameb1, CV_CHECKBOX, 0);
	char* nameb2 = "Not Save/Save";
	createButton(nameb2, AutomaticSave, nameb2, CV_CHECKBOX, 0);

	int BlockXID, BlockYID, setReferenceFrame, setSeqFrame, same, noUpdate, swidth, sheight;
	bool GlobalSlider[nsequences]; //True: global slider, false: local slider
	for (int ii = 0; ii < nsequences; ii++)
		GlobalSlider[ii] = true;

	int SaveFrameCount = 0;
	while (waitKey(17) != 27)
	{
		noUpdate = 0;
		if (playBackSpeed < 1)
			playBackSpeed = 1;
		for (int ii = 0; ii < nsequences; ii++)
		{
			BlockXID = ii%nBlockX, BlockYID = ii / nBlockX;

			same = 0;
			if (GlobalSlider[ii])
				setReferenceFrame = FrameID[0]; //global frame
			else
				setReferenceFrame = FrameID[ii + 1];

			if (oFrameID[0] != FrameID[0])
				FrameID[ii + 1] = FrameID[0], GlobalSlider[ii] = true;
			else
				same += 1;

			if (oFrameID[ii + 1] != FrameID[ii + 1]) //but if local slider moves
			{
				setReferenceFrame = FrameID[ii + 1];
				if (same == 0 && GlobalSlider[ii])
					GlobalSlider[ii] = true;
				else
					GlobalSlider[ii] = false;
			}
			else
				same += 1;

			sprintf(Fname, "Seq %d", ii + 1);
			setSeqFrame = (int)((1.0*setReferenceFrame / mySeq[refSeq].TimeAlignPara[0] - mySeq[ii].TimeAlignPara[1]) * mySeq[ii].TimeAlignPara[0] + 0.5); //(refFrame/fps_ref - current)*fps_current
			printf("Sequence %d frame %d\n", ii + 1, setSeqFrame);

			if (same == 2)
			{
				noUpdate++;
				if (autoplay)
				{
					sprintf(Fname, "Seq %d", ii + 1);
					cvSetTrackbarPos(Fname, "Control2", FrameID[ii + 1]);
					FrameID[ii + 1] += playBackSpeed;
				}
				continue;
			}
			if (setSeqFrame <= 0)
			{
				cvSetTrackbarPos(Fname, "Control2", (int)(mySeq[ii].TimeAlignPara[1] + 0.5));
				Set_Sub_Mat(BlackImage, BigImg, nchannels*WBlock, HBlock, nchannels*width, nchannels*BlockXID*WBlock, BlockYID*HBlock);
			}
			else
			{
				oFrameID[ii + 1] = FrameID[ii + 1];
				cvSetTrackbarPos(Fname, "Control2", oFrameID[ii + 1]);
				sprintf(Fname, "%s/%d/%d.png", DataPATH, seqName[ii], setSeqFrame);
				if (GrabImage(Fname, SubImage, swidth, sheight, nchannels))
					Set_Sub_Mat(SubImage, BigImg, nchannels*swidth, sheight, nchannels*width, nchannels*BlockXID*WBlock, BlockYID*HBlock);
				else
					Set_Sub_Mat(BlackImage, BigImg, nchannels*WBlock, HBlock, nchannels*width, nchannels*BlockXID*WBlock, BlockYID*HBlock);

				if (saveFrame)
				{
					sprintf(Fname, "%s/%d/%d.png", SavePATH, seqName[ii], SaveFrameCount / nsequences);
					SaveDataToImage(Fname, SubImage, swidth, sheight, nchannels);
					SaveFrameCount++;
				}
				else
					SaveFrameCount = 0;
			}
			if (autoplay)
			{
				sprintf(Fname, "Seq %d", ii + 1);
				cvSetTrackbarPos(Fname, "Control2", FrameID[ii + 1]);
				FrameID[ii + 1] += playBackSpeed;
			}
		}
		oFrameID[0] = FrameID[0];
		if (noUpdate != nsequences)
			ShowDataToImage("VideoSequences2", BigImg, width, height, nchannels, cvImg);

		if (autoplay)
		{
			int ii;
			for (ii = 0; ii < nsequences; ii++)
				if (!GlobalSlider[ii])
					break;
			if (ii == nsequences)
			{
				cvSetTrackbarPos("Global frame", "Control2", FrameID[0]);
				FrameID[0] += playBackSpeed;
			}
		}
	}

	cvReleaseImage(&cvImg);
	delete[]BigImg;
	delete[]BlackImage;
	delete[]SubImage;

	return 0;
}
int ShowSyncLoadDTW(char *DataPATH)
{
	char Fname[2000];
	const int playBackSpeed = 1, nsequences = 2, refSeq = 1;
	int seqName[nsequences] = { 2, 7 };

	int WBlock = 1920, HBlock = 1080, nBlockX = 3, nchannels = 3, MaxFrames = 202;

	Sequence mySeq[nsequences];
	// soccer sequences: 
	mySeq[0].InitSeq(47.95, 0);
	mySeq[1].InitSeq(47.95, 72.78);

	//Read video sequences
	int width = 0, height = 0;
	nBlockX = nsequences < nBlockX ? nsequences : nBlockX;
	for (int ii = 0; ii < nsequences; ii++)
		width += WBlock, height += HBlock;

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
		oFrameID[ii] = -1, FrameID[ii] = 0;
	cvNamedWindow("VideoSequences", CV_WINDOW_NORMAL);
	cvCreateTrackbar("Global frame", "VideoSequences", &FrameID[0], MaxFrames - 1, NULL);
	char* nameb1 = "Play/Stop";
	createButton(nameb1, AutomaticPlay, nameb1, CV_CHECKBOX, 1);

	int BlockXID, BlockYID, setframeID, setSeqFrame, swidth, sheight;

	int id1, id2;
	vector<int> seq[nsequences]; seq[0].reserve(100), seq[1].reserve(100);
	sprintf(Fname, "%s/SyncSeq.txt", DataPATH);
	FILE *fp = fopen(Fname, "r");
	while (fscanf(fp, "%d %d ", &id1, &id2) != EOF)
		seq[1].push_back(id1), seq[0].push_back(id2);
	fclose(fp);

	int frameID = 0;
	while (FrameID[0] < seq[0].size() && waitKey(17) != 27)
	{
		if (FrameID[0] == oFrameID[0])
			continue;
		for (int ii = 0; ii < nsequences; ii++)
		{
			BlockXID = ii%nBlockX, BlockYID = ii / nBlockX;

			setframeID = FrameID[0]; //global frame
			setSeqFrame = seq[ii].at(setframeID);
			printf("Sequence %d frame %d\n", ii + 1, setSeqFrame);

			oFrameID[ii + 1] = setframeID;

			sprintf(Fname, "%s/RawImages/%d/%d.png", DataPATH, seqName[ii], setSeqFrame);
			if (GrabImage(Fname, SubImage, swidth, sheight, nchannels))
				Set_Sub_Mat(SubImage, BigImg, nchannels*swidth, sheight, nchannels*width, nchannels*BlockXID*WBlock, BlockYID*HBlock);
			else
				Set_Sub_Mat(BlackImage, BigImg, nchannels*WBlock, HBlock, nchannels*width, nchannels*BlockXID*WBlock, BlockYID*HBlock);
		}
		oFrameID[0] = FrameID[0];
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
	if (!loadNVMLite(nvmfile, corpusData, 1))
		return 1;
	int nviews = corpusData.nCamera;

	CameraData *cameraInfo = new CameraData[nCams];
	if (!ReadIntrinsicResults(Path, cameraInfo))
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
		if (ptsCount > n3D - 1)
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
int TriangulatePointsFromCalibratedCameras2(char *Path, int nCams, int selectedCams, int distortionCorrected, double threshold = 2.0)
{
	char Fname[200];
	Corpus corpusData;
	sprintf(Fname, "%s/BA_Camera_AllParams_after.txt", Path);
	if (!loadBundleAdjustedNVMResults(Fname, corpusData))
		return 1;

	int nviews = corpusData.nCamera;
	for (int ii = 0; ii < nviews; ii++)
	{
		corpusData.camera[ii].threshold = threshold, corpusData.camera[ii].ninlierThresh = 50, corpusData.camera[ii];
		GetrtFromRT(corpusData.camera[ii].rt, corpusData.camera[ii].R, corpusData.camera[ii].T);
		GetIntrinsicFromK(corpusData.camera[ii]);
		AssembleP(corpusData.camera[ii].K, corpusData.camera[ii].R, corpusData.camera[ii].T, corpusData.camera[ii].P);
		if (distortionCorrected == 1)
			for (int jj = 0; jj < 7; jj++)
				corpusData.camera[ii].distortion[jj] = 0.0;
	}
	printf("...Done\n");
	FILE *fp = fopen("C:/temp/PInfo.txt", "w+");
	for (int ii = 0; ii < nviews; ii++)
	{
		fprintf(fp, "%d ", ii);
		for (int jj = 0; jj < 12; jj++)
			fprintf(fp, "%.8f ", corpusData.camera[ii].P[jj]);
		fprintf(fp, "\n");
	}
	fclose(fp);
	fp = fopen("C:/temp/KInfo.txt", "w+");
	for (int ii = 0; ii < nviews; ii++)
	{
		fprintf(fp, "%d ", ii);
		for (int jj = 0; jj < 9; jj++)
			fprintf(fp, "%.8f ", corpusData.camera[ii].K[jj]);
		fprintf(fp, "\n");
	}
	fclose(fp);
	fp = fopen("C:/temp/DistortionInfo.txt", "w+");
	for (int ii = 0; ii < nviews; ii++)
	{
		fprintf(fp, "%d ", ii);
		for (int jj = 0; jj < 7; jj++)
			fprintf(fp, "%.8f ", corpusData.camera[ii].distortion[jj]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	int n3D = 1, viewID, ptsCount = 0;
	double  u, v;
	vector<Point3d> t3D;
	vector<Point2d> uv;
	vector<int> viewIDAll3D;
	vector<Point2d>uvAll3D;

	while (true)
	{
		t3D.clear(), uv.clear(), viewIDAll3D.clear(), uvAll3D.clear();
		FILE *fp = fopen("C:/temp/3D2D.txt", "r");
		while (fscanf(fp, "%d %lf %lf ", &viewID, &u, &v) != EOF)
		{
			viewIDAll3D.push_back(viewID);
			uvAll3D.push_back(Point2d(u, v));
		}
		fclose(fp);

		//Test if 3D is correct
		ptsCount = uvAll3D.size();
		Point3d xyz;
		double *A = new double[6 * ptsCount];
		double *B = new double[2 * ptsCount];
		double *tPs = new double[12 * ptsCount];
		bool *passed = new bool[ptsCount];
		double *Ps = new double[12 * ptsCount];
		Point2d *match2Dpts = new Point2d[ptsCount];

		vector<int>Inliers[1];  Inliers[0].reserve(ptsCount * 2);
		double ProThresh = 0.99, PercentInlier = 0.25;
		int goodNDplus = 0, iterMax = (int)(log(1.0 - ProThresh) / log(1.0 - pow(PercentInlier, 2)) + 0.5); //log(1-eps) / log(1 - (inlier%)^min_pts_requires)
		int nviewsi = viewIDAll3D.size();
		Inliers[0].clear();
		for (int ii = 0; ii < nviewsi; ii++)
		{
			viewID = viewIDAll3D.at(ii);
			for (int kk = 0; kk < 12; kk++)
				Ps[12 * ii + kk] = corpusData.camera[viewID].P[kk];

			match2Dpts[ii] = uvAll3D.at(ii);
			LensCorrectionPoint(&match2Dpts[ii], corpusData.camera[viewID].K, corpusData.camera[viewID].distortion);
		}

		NviewTriangulationRANSAC(match2Dpts, Ps, &xyz, passed, Inliers, nviewsi, 1, iterMax, PercentInlier, corpusData.camera[0].threshold, A, B, tPs);
		if (passed[0])
			printf("%f %f %f\n", xyz.x, xyz.y, xyz.z);
	}

	return 0;
}
void VisualizeCleanMatches(char *Path, int view1, int view2, int timeID, double fractionMatchesDisplayed = 0.5)
{
	char Fname[200];
	if (timeID < 0)
		sprintf(Fname, "%s/Corpus/M_%d_%d.dat", Path, view1, view2);
	else
		sprintf(Fname, "%s/M%d_%d_%d.dat", Path, timeID, view1, view2);

	vector<Point2i> PairWiseMatchID; PairWiseMatchID.reserve(10000);
	int id1, id2, npts;
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return;
	}
	fscanf(fp, "%d ", &npts);
	PairWiseMatchID.reserve(npts);
	while (fscanf(fp, "%d %d ", &id1, &id2) != EOF)
		PairWiseMatchID.push_back(Point2i(id1, id2));
	fclose(fp);

	if (timeID < 0)
		sprintf(Fname, "%s/Corpus/K%d.dat", Path, view1);
	else
		sprintf(Fname, "%s/%d/K%d.dat", Path, view1, timeID);

	bool readsucces = false;
	vector<KeyPoint> keypoints1; keypoints1.reserve(MAXSIFTPTS);
	readsucces = ReadKPointsBinarySIFTGPU(Fname, keypoints1);
	if (!readsucces)
	{
		printf("%s does not have SIFT points. Please precompute it!\n", Fname);
		return;
	}

	if (timeID < 0)
		sprintf(Fname, "%s/Corpus/K%d.dat", Path, view2);
	else
		sprintf(Fname, "%s/%d/K%d.dat", Path, view2, timeID);
	vector<KeyPoint> keypoints2; keypoints2.reserve(MAXSIFTPTS);
	readsucces = ReadKPointsBinarySIFTGPU(Fname, keypoints2);
	if (!readsucces)
	{
		printf("%s does not have SIFT points. Please precompute it!\n", Fname);
		return;
	}

	vector<int> CorresID;
	for (int i = 0; i < PairWiseMatchID.size(); ++i)
		CorresID.push_back(PairWiseMatchID[i].x), CorresID.push_back(PairWiseMatchID[i].y);

	int nchannels = 3;
	if (timeID < 0)
		sprintf(Fname, "%s/Corpus/%d.png", Path, view1);
	else
		sprintf(Fname, "%s/%d/%d.png", Path, view1, timeID);
	IplImage *Img1 = cvLoadImage(Fname, nchannels == 3 ? 1 : 0);
	if (Img1->imageData == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return;
	}
	if (timeID < 0)
		sprintf(Fname, "%s/Corpus/%d.png", Path, view2);
	else
		sprintf(Fname, "%s/%d/%d.png", Path, view2, timeID);
	IplImage *Img2 = cvLoadImage(Fname, nchannels == 3 ? 1 : 0);
	if (Img2->imageData == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return;
	}

	IplImage* correspond = cvCreateImage(cvSize(Img1->width + Img2->width, Img1->height), 8, nchannels);
	cvSetImageROI(correspond, cvRect(0, 0, Img1->width, Img1->height));
	cvCopy(Img1, correspond);
	cvSetImageROI(correspond, cvRect(Img1->width, 0, correspond->width, correspond->height));
	cvCopy(Img2, correspond);
	cvResetImageROI(correspond);
	DisplayImageCorrespondence(correspond, Img1->width, 0, keypoints1, keypoints2, CorresID, fractionMatchesDisplayed);

	return;
}
void ConvertImagetoHanFormat(char *Path, int nviews, int beginTime, int endTime)
{
	char Fname[200];
	for (int ii = beginTime; ii <= endTime; ii++)
	{
		sprintf(Fname, "%s/In/%08d", Path, ii);
		mkdir(Fname);
	}

	Mat img; double incre = 5.0;
	for (int jj = 0; jj < nviews; jj++)
	{
		for (int ii = beginTime; ii <= endTime; ii++)
		{
			sprintf(Fname, "%s/Minh/%d/%d.png", Path, jj, ii);
			img = imread(Fname, 1);

			sprintf(Fname, "%s/In/%08d/%08d_%02d_%02d.png", Path, ii, ii, 0, jj);
			imwrite(Fname, img);

			double percent = 100.0*(ii - beginTime) / (endTime - beginTime);
			if (percent >= incre)
				printf("\rView %d: %.1f%% ", jj, percent), incre += 5.0;
		}
		printf("\rView %d: 100%% \n", jj);
	}

	return;
}

int GenerateCommonCameraForNPlusPoint(char *Path, int nviews, int nplus, int timeID)
{
	const int MaxCam = 100;
	char Fname[200];
	int ii, jj, kk, pid, vid, ninliers;
	Point3d xyz, rgb, arrow1, arrow2;
	double scale, u, v;

	int counter, visibleCam[MaxCam], T[MaxCam];
	Point2d imageCorres[MaxCam], bk_imageCorres[MaxCam];

	int NPlusViewerCount = 0, npossibleNPlusViewer = pow(2, nviews);
	vector<int> *ListNPlusViewer = new vector<int>[npossibleNPlusViewer];
	vector<Point2d> *ListNPlusViewerPoints = new vector<Point2d>[npossibleNPlusViewer];

	sprintf(Fname, "%s/3DMem_%d.txt", Path, timeID);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return 1;
	}
	int currentPt = 0;
	while (fscanf(fp, "%s %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf ",
		Fname, &pid, &xyz.x, &xyz.y, &xyz.z, &rgb.x, &rgb.y, &rgb.z,
		&scale, &arrow1.x, &arrow1.y, &arrow1.z,
		&arrow2.x, &arrow2.y, &arrow2.z) != EOF)
	{
		currentPt++;
		fscanf(fp, "%d ", &ninliers);
		if (ninliers >= 3)
			counter = 0;

		for (ii = 0; ii < ninliers; ii++)
		{
			fscanf(fp, "%d %lf %lf ", &vid, &u, &v);
			if (ninliers >= 3)
			{
				visibleCam[counter] = vid;
				T[counter] = counter;
				imageCorres[counter] = Point2d(u, v);
				counter++;
			}
		}

		if (ninliers >= 3)
		{
			//Sort them for easier indexing
			Quick_Sort_Int(visibleCam, T, 0, counter - 1);
			for (ii = 0; ii < counter; ii++)
				bk_imageCorres[ii] = imageCorres[ii];
			for (ii = 0; ii < counter; ii++)
				imageCorres[ii] = bk_imageCorres[T[ii]];

			//Put them to list
			int CurrentIdinList = 0;
			bool NotfoundnewViewer = true;
			if (NPlusViewerCount == 0)
				NotfoundnewViewer = false;
			else
			{
				NotfoundnewViewer = false;
				for (ii = 0; ii < NPlusViewerCount; ii++)//Try to find if the new visible set already exists
				{
					int ListSize = ListNPlusViewer[ii].size();
					for (jj = 0; jj < min(counter, ListSize); jj++)
						if (visibleCam[jj] != ListNPlusViewer[ii][jj])
							break;
					if (jj == counter && jj == ListSize)
					{
						CurrentIdinList = ii;
						NotfoundnewViewer = true;
						break;
					}
				}
			}

			if (!NotfoundnewViewer)//found new N+viewers
			{
				CurrentIdinList = NPlusViewerCount;
				NPlusViewerCount++;
				for (jj = 0; jj < counter; jj++)
					ListNPlusViewer[CurrentIdinList].push_back(visibleCam[jj]);
			}
			else
				int a = 0;


			//dump the 2d correspondences to that N+viewer
			for (jj = 0; jj < counter; jj++)
				ListNPlusViewerPoints[CurrentIdinList].push_back(imageCorres[jj]);
		}
	}

	//Write down the list:
	sprintf(Fname, "%s/NPlusViewer/", Path);  mkdir(Fname);
	for (ii = 0; ii < NPlusViewerCount; ii++)
	{
		sprintf(Fname, "%s/NPlusViewer/%d.txt", Path, ii); fp = fopen(Fname, "w+");
		int nviewers = ListNPlusViewer[ii].size(), ncorres = ListNPlusViewerPoints[ii].size() / nviewers;
		fprintf(fp, "%d ", nviewers);
		for (jj = 0; jj < nviewers; jj++)
			fprintf(fp, "%d ", ListNPlusViewer[ii][jj]);
		fprintf(fp, "\n%d\n", ListNPlusViewerPoints[ii].size() / nviewers);
		for (jj = 0; jj < ncorres; jj++)
		{
			for (kk = 0; kk < nviewers; kk++)
				fprintf(fp, "%.3f %.3f ", ListNPlusViewerPoints[ii][kk + jj*nviewers].x, ListNPlusViewerPoints[ii][kk + jj*nviewers].y);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}
	return 0;
}
int GenerateTracksForNPlusPoint(char *Path, int nviews, int nplus, int timeID, int fileID)
{
	char Fname[200];

	vector<float*> ImgPara;
	LKParameters LKArg;
	LKArg.hsubset = 7, LKArg.nscales = 2, LKArg.scaleStep = 5, LKArg.DisplacementThresh = 1, LKArg.DIC_Algo = 3, LKArg.InterpAlgo = 1, LKArg.EpipEnforce = 0;
	LKArg.Incomplete_Subset_Handling = 0, LKArg.Convergence_Criteria = 0, LKArg.Analysis_Speed = 0, LKArg.IterMax = 15;
	LKArg.Gsigma = 1.0, LKArg.ssigThresh = 30.0;

	int  nviewers, vid, npts, orgfileID = fileID;
	double u, v;
	vector<int> viewerList;
	vector<Point2d> ptsList;
	while (true)
	{
		printf("\n***** File #: %d******\n", fileID);
		sprintf(Fname, "%s/NPlusViewer/%d.txt", Path, fileID); FILE *fp = fopen(Fname, "r");
		ifstream testFin(Fname);
		if (testFin.is_open())
			testFin.close();
		else
			break;

		ImgPara.clear(), viewerList.clear(), ptsList.clear();
		fscanf(fp, "%d ", &nviewers);
		for (int ii = 0; ii < nviewers; ii++)
		{
			fscanf(fp, "%d ", &vid);
			viewerList.push_back(vid);
		}
		fscanf(fp, "%d ", &npts);
		for (int ii = 0; ii < npts; ii++)
		{
			for (int jj = 0; jj < nviewers; jj++)
			{
				fscanf(fp, "%lf %lf ", &u, &v);
				ptsList.push_back(Point2d(u, v));
			}
		}
		fclose(fp);

		//Now, run tracking
		vector<Point2d> *Tracks = new vector<Point2d>[nviewers*npts];
		for (int jj = 0; jj < nviewers; jj++)
		{
			for (int ii = 0; ii < ImgPara.size(); ii++)
				delete ImgPara[ii];
			ImgPara.clear();

			int width, height;
			for (int ii = 0; ii < npts; ii++)
			{
				Tracks[ii*nviewers + jj].push_back(ptsList[ii*nviewers + jj]);
				//sprintf(Fname, "%s/%d", Path, viewerList[jj]);
				SparsePointTrackingDriver(Path, Tracks[ii*nviewers + jj], ImgPara, viewerList[jj], timeID, timeID + 500, LKArg, width, height, 1);
			}
		}

		sprintf(Fname, "%s/Track", Path), mkdir(Fname);
		sprintf(Fname, "%s/Track/%d_%d.txt", Path, timeID, fileID);
		FILE *fp2 = fopen(Fname, "w+");
		fprintf(fp2, "%d ", nviewers);
		for (int jj = 0; jj < nviewers; jj++)
			fprintf(fp2, "%d ", viewerList[jj]);
		fprintf(fp2, "\n");
		fprintf(fp2, "%d\n", npts);
		for (int ii = 0; ii < npts; ii++)
		{
			for (int jj = 0; jj < nviewers; jj++)
			{
				fprintf(fp2, "%d %d %d\n", viewerList[jj], timeID, timeID + Tracks[ii*nviewers + jj].size() - 1);
				for (int kk = 0; kk < Tracks[ii*nviewers + jj].size(); kk++)
					fprintf(fp2, "%.3f %.3f ", Tracks[ii*nviewers + jj][kk].x, Tracks[ii*nviewers + jj][kk].y);
				fprintf(fp2, "\n");
			}
		}
		fclose(fp2);

		delete[]Tracks;
		//	if (fileID - orgfileID >= 50)
		break;
		fileID++;
	}

	return 0;
}
int GenarateTrajectoryFrom2DTracks(char *Path, int nviews, int startTime, int stopTime, int timeID, int fileID, double Moing3DPointThresh = 10, double Moving2DPointThresh = 1.0)
{
	char Fname[200];
	int nviewers, vid, npts, maxTime = 0, stopID;
	double u, v;
	vector<int> ViewerList;

	VideoData AllVideoInfo;
	if (ReadVideoData(Path, AllVideoInfo, nviews, startTime, stopTime) == 1)
		return 1;
	int nframes = max(MaxnFrame, stopTime);

	sprintf(Fname, "%s/Track/N_%d_%d.txt", Path, timeID, fileID);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("%s\n", Fname);
		return 0;
	}
	fscanf(fp, "%d ", &nviewers);
	for (int ii = 0; ii < nviewers; ii++)
	{
		fscanf(fp, "%d ", &vid);
		ViewerList.push_back(vid);
	}
	fscanf(fp, "%d ", &npts);
	vector<Point2d> *Tracks = new vector<Point2d>[npts*nviewers];
	for (int kk = 0; kk < npts; kk++)
	{
		for (int jj = 0; jj < nviewers; jj++)
		{
			fscanf(fp, "%d %d %d ", &vid, &timeID, &stopID);
			for (int ii = 0; ii < stopID - timeID + 1; ii++)
			{
				fscanf(fp, "%lf %lf ", &u, &v);
				Tracks[jj + kk*nviewers].push_back(Point2d(u, v));
			}

			if (maxTime < stopID)
				maxTime = stopID;
		}
	}
	fclose(fp);


	//Verify correspondences
	bool passed;
	Point2d *PutativeCorres = new Point2d[nviewers];
	double *P = new double[12 * nviewers];
	double *A = new double[6 * nviews];
	double *B = new double[2 * nviews];
	double *tPs = new double[12 * nviews];
	vector<int> inliers, viewId;
	Point3d WC;
	vector<Trajectory3D> *traject3D = new vector<Trajectory3D>[npts];

	for (int jj = 0; jj < npts; jj++)
	{
		for (int ii = 0; ii < maxTime - timeID + 1; ii++)
		{
			int nvisible = 0;

			viewId.clear();
			for (int kk = 0; kk < nviewers; kk++)
			{
				int videoID = nframes*ViewerList[kk];
				if (Tracks[kk + jj*nviewers].size() > ii + 1)
				{
					PutativeCorres[nvisible] = Tracks[kk + jj*nviewers][ii];
					for (int ll = 0; ll < 12; ll++)
						P[12 * nvisible + ll] = AllVideoInfo.VideoInfo[videoID + ii + timeID].P[ll];
					viewId.push_back(kk);
					nvisible++;
				}
			}

			if (nvisible < 2) //not useful-->kill the tracjectory
			{
				for (int kk = 0; kk < nvisible; kk++)
					Tracks[viewId[kk] + jj*nviewers].erase(Tracks[viewId[kk] + jj*nviewers].begin() + ii, Tracks[viewId[kk] + jj*nviewers].end());
				break;
			}
			else
			{
				inliers.clear();
				double error = NviewTriangulationRANSAC(PutativeCorres, P, &WC, &passed, &inliers, nvisible, 1, 10, 0.75, 8.0, A, B, tPs);
				if (!passed) //not useful-->kill the tracjectory
				{
					for (int kk = 0; kk < nvisible; kk++)
						Tracks[viewId[kk] + jj*nviewers].erase(Tracks[viewId[kk] + jj*nviewers].begin() + ii, Tracks[viewId[kk] + jj*nviewers].end());
					break;
				}
				else
				{
					int ninlier = 0;
					for (int kk = 0; kk < nvisible; kk++)
						if (inliers.at(kk) == 1)
							ninlier++;

					if (ninlier < 2) //not useful-->kill the trajectory
					{
						for (int kk = 0; kk < nvisible; kk++)
							Tracks[viewId[kk] + jj*nviewers].erase(Tracks[viewId[kk] + jj*nviewers].begin() + ii, Tracks[viewId[kk] + jj*nviewers].end());
						break;
					}
					else
					{
						Trajectory3D T;
						T.timeID = ii + timeID; T.WC = WC;
						T.viewIDs.reserve(ninlier), T.uv.reserve(ninlier);
						for (int kk = 0; kk < nvisible; kk++)
							if (inliers.at(kk) == 1)
								T.viewIDs.push_back(viewId[kk]), T.uv.push_back(PutativeCorres[kk]);

						traject3D[jj].push_back(T);
					}
				}
			}
		}
	}

	//Remove not moving points
	double varX, varY, varZ;
	vector<double> X, Y, Z;
	for (int kk = 0; kk < npts; kk++)
	{
		X.clear(), Y.clear(), Z.clear();
		for (int jj = 0; jj < traject3D[kk].size(); jj++)
			X.push_back(traject3D[kk][jj].WC.x), Y.push_back(traject3D[kk][jj].WC.y), Z.push_back(traject3D[kk][jj].WC.z);

		varX = VarianceArray(X), varY = VarianceArray(Y), varZ = VarianceArray(Z);
		if (sqrt(varX + varY + varZ) < Moing3DPointThresh)
			traject3D[kk].clear(); //stationary points removed
	}

	//remove not moving points: 2D
	for (int ii = 0; ii < npts; ii++)
	{
		int tracksLength = traject3D[ii].size();
		if (tracksLength < 1)
			continue;
		double distance = 0.0;
		for (int jj = 0; jj < tracksLength - 1; jj++)
		{
			int nvis = min(traject3D[ii][jj].viewIDs.size(), traject3D[ii][jj + 1].viewIDs.size());
			for (int kk = 0; kk < nvis; kk++)
				distance += (sqrt(pow(traject3D[ii][jj].uv[kk].x - traject3D[ii][jj + 1].uv[kk].x, 2) + pow(traject3D[ii][jj].uv[kk].y - traject3D[ii][jj + 1].uv[kk].y, 2))) / nvis;
		}
		distance /= tracksLength - 1;
		if (distance < Moving2DPointThresh)
			traject3D[ii].clear();
	}

	//remove very short (long) trajectory
	for (int ii = 0; ii < npts; ii++)
	{
		int tracksLength = traject3D[ii].size();
		if (tracksLength < 10 || tracksLength > 490)
			traject3D[ii].clear();
	}

	//Write Trajectory recon 
	sprintf(Fname, "%s/Traject3D", Path), mkdir(Fname);
	sprintf(Fname, "%s/Traject3D/3D_%d_%d.txt", Path, timeID, fileID); fp = fopen(Fname, "w+");

	int validNpts = 0;
	for (int kk = 0; kk < npts; kk++)
	{
		if (traject3D[kk].size() == 0)
			continue;
		validNpts++;
	}
	fprintf(fp, "3D %d\n", validNpts);
	for (int kk = 0; kk < npts; kk++)
	{
		if (traject3D[kk].size() == 0)
			continue;
		fprintf(fp, "%d %d\n", timeID, traject3D[kk].size());
		for (int jj = 0; jj < traject3D[kk].size(); jj++)
			fprintf(fp, "%.3f %.3f %.3f ", traject3D[kk][jj].WC.x, traject3D[kk][jj].WC.y, traject3D[kk][jj].WC.z);
		fprintf(fp, "\n");
	}
	fprintf(fp, "Visibility %d\n", npts);
	for (int kk = 0; kk < npts; kk++)
	{
		if (traject3D[kk].size() == 0)
			continue;
		fprintf(fp, "%d %d\n", timeID, traject3D[kk].size());
		for (int jj = 0; jj < traject3D[kk].size(); jj++)
		{
			fprintf(fp, "%d ", traject3D[kk][jj].viewIDs.size());
			for (int ii = 0; ii < traject3D[kk][jj].viewIDs.size(); ii++)
				fprintf(fp, "%d ", traject3D[kk][jj].viewIDs[ii]);
			fprintf(fp, "\n");
		}
	}
	fprintf(fp, "2D %d\n", npts);
	for (int kk = 0; kk < npts; kk++)
	{
		if (traject3D[kk].size() == 0)
			continue;
		fprintf(fp, "%d %d\n", timeID, traject3D[kk].size());
		for (int jj = 0; jj < traject3D[kk].size(); jj++)
		{
			fprintf(fp, "%d ", traject3D[kk][jj].viewIDs.size());
			for (int ii = 0; ii < traject3D[kk][jj].viewIDs.size(); ii++)
				fprintf(fp, "%.2f %.2f ", traject3D[kk][jj].uv[ii].x, traject3D[kk][jj].uv[ii].y);
			fprintf(fp, "\n");
		}
	}
	fclose(fp);

	//Rewrite Tracks
	int new_npts = 0, ptsCount = 0, tracksLength, nvis, viewID;
	for (int ii = 0; ii < npts; ii++)
		if (traject3D[ii].size() > 0)
			new_npts++;
	vector<Point2d> *NewTracks = new vector<Point2d>[new_npts*nviewers];

	for (int ii = 0; ii < npts; ii++)
	{
		tracksLength = traject3D[ii].size();
		if (tracksLength < 1)
			continue;
		for (int jj = 0; jj < tracksLength; jj++)
		{
			nvis = traject3D[ii][jj].viewIDs.size();
			for (int kk = 0; kk < nvis; kk++)
			{
				viewID = traject3D[ii][jj].viewIDs[kk];
				NewTracks[ptsCount*nviewers + viewID].push_back(traject3D[ii][jj].uv[kk]);
			}
		}
		ptsCount++;
	}
	if (ptsCount > 0)
	{
		sprintf(Fname, "%s/Track", Path), mkdir(Fname);
		sprintf(Fname, "%s/Track/N_%d_%d.txt", Path, timeID, fileID);
		FILE *fp2 = fopen(Fname, "w+");
		fprintf(fp2, "%d ", nviewers);
		for (int jj = 0; jj < nviewers; jj++)
			fprintf(fp2, "%d ", ViewerList[jj]);
		fprintf(fp2, "\n");
		fprintf(fp2, "%d\n", new_npts);
		for (int ii = 0; ii < new_npts; ii++)
		{
			for (int jj = 0; jj < nviewers; jj++)
			{
				fprintf(fp2, "%d %d %d\n", ViewerList[jj], timeID, timeID + NewTracks[ii*nviewers + jj].size() - 1);
				for (int kk = 0; kk < NewTracks[ii*nviewers + jj].size(); kk++)
					fprintf(fp2, "%.4f %.4f ", NewTracks[ii*nviewers + jj][kk].x, NewTracks[ii*nviewers + jj][kk].y);
				fprintf(fp2, "\n");
			}
		}
		fclose(fp2);
	}

	delete[]traject3D, delete[]Tracks, delete[]NewTracks;
	delete[]PutativeCorres, delete[]P, delete[]A, delete[]B, delete[]tPs;

	return 0;
}
int Interpolation3DTrajectory(int trackID)
{
	char Fname[200];
	double x, y, z;
	vector<double>X, Y, Z;

	sprintf(Fname, "C:/temp/Sim/f%d.txt", trackID);
	FILE *fp = fopen(Fname, "r");
	while (fscanf(fp, "%lf %lf %lf", &x, &y, &z) != EOF)
		X.push_back(x), Y.push_back(y), Z.push_back(z);
	fclose(fp);

	int npts = X.size();
	double *vX = new double[npts], *vY = new double[npts], *vZ = new double[npts];
	for (int ii = 0; ii < npts; ii++)
		vX[ii] = X[ii], vY[ii] = Y[ii], vZ[ii] = Z[ii];

	double *pX = new double[npts], *pY = new double[npts], *pZ = new double[npts];
	Generate_Para_Spline(vX, pX, npts, 1, 1);
	Generate_Para_Spline(vY, pY, npts, 1, 1);
	Generate_Para_Spline(vZ, pZ, npts, 1, 1);

	int Upsample = 10;
	double S[3];
	double *uX = new double[npts*Upsample], *uY = new double[npts*Upsample], *uZ = new double[npts*Upsample];
	for (int ii = 0; ii < npts*Upsample; ii++)
	{
		Get_Value_Spline(pX, npts, 1, 1.0*ii / Upsample, 0.0, S, -1, 1); uX[ii] = S[0];
		Get_Value_Spline(pY, npts, 1, 1.0*ii / Upsample, 0.0, S, -1, 1); uY[ii] = S[0];
		Get_Value_Spline(pZ, npts, 1, 1.0*ii / Upsample, 0.0, S, -1, 1); uZ[ii] = S[0];
	}

	sprintf(Fname, "C:/temp/Sim/if%d.txt", trackID);
	fp = fopen(Fname, "w+");
	for (int ii = 0; ii < npts*Upsample; ii++)
		fprintf(fp, "%.6f %.6f %.6f\n", uX[ii], uY[ii], uZ[ii]);
	fclose(fp);

	return 0;
}
int Generate2DTracksFrom3D_Simu(char *Path, int nviews, int trackID, int timeID, int startTime, int stopTime)
{
	char Fname[200];

	int nf;
	double x, y, z, mX = 0.0, mY = 0.0, mZ = 0.0;
	vector<Point3d> XYZ;
	sprintf(Fname, "C:/temp/Sim/i%d.txt", trackID); FILE *fp = fopen(Fname, "r");
	//fscanf(fp, "%s %d %d %d", Fname, &nf, &nf, &nf);
	while (fscanf(fp, "%lf %lf %lf", &x, &y, &z) != EOF)
	{
		XYZ.push_back(Point3d(x, y, z));
		mX += x, mY += y, mZ += z;
	}
	fclose(fp);
	nf = XYZ.size();

	mX = mX / nf, mY = mY / nf, mZ = mZ / nf;
	for (int jj = 0; jj < nf; jj++)
	{
		XYZ[jj].x = 1.0*(XYZ[jj].x - mX) + mX;
		XYZ[jj].y = 1.0*(XYZ[jj].y - mY) + mY;
		XYZ[jj].z = 1.0*(XYZ[jj].z - mZ) + mZ;
	}

	VideoData AllVideoInfo;
	if (ReadVideoData(Path, AllVideoInfo, nviews, startTime, stopTime) == 1)
		return 1;
	int nframes = max(MaxnFrame, stopTime);

	double P[12];
	Point2d pts;
	sprintf(Fname, "C:/temp/Sim/iTrack_%d.txt", trackID); fp = fopen(Fname, "w+");
	/*fprintf(fp, "%d ", nviews);
	for (int kk = 0; kk < nviews; kk++)
	fprintf(fp, "%d ", kk);
	fprintf(fp, "\n1\n");*/
	fprintf(fp, "%d %d \n", timeID, nf);
	for (int kk = 0; kk < nviews; kk++)
	{
		//fprintf(fp, "%d %d %d \n", kk, timeID, nf);
		int videoID = kk*nframes;
		for (int jj = 0; jj < nf; jj++)
		{
			for (int ii = 0; ii < 12; ii++)
				/*if (kk>1 && jj>499)
				P[ii] = AllVideoInfo.VideoInfo[videoID + 499].P[ii];
				else
				P[ii] = AllVideoInfo.VideoInfo[videoID + jj + timeID].P[ii];*/
				P[ii] = AllVideoInfo.VideoInfo[videoID + timeID].P[ii];

			ProjectandDistort(XYZ[jj], &pts, P, NULL, NULL, 1);
			fprintf(fp, "%.16f %.16f ", pts.x, pts.y);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	return 0;
}
int GenarateTrajectoryInput(char *Path, int nviews, int startTime, int stopTime, int timeID, int fileID, int syncUncertainty, double alpha)
{
	char Fname[200];
	int nviewers, vid, npts, maxTime = 0;
	double u, v;
	vector<int> ViewerList;

	VideoData AllVideoInfo;
	if (ReadVideoData(Path, AllVideoInfo, nviews, startTime, stopTime) == 1)
		return 1;
	int nframes = max(MaxnFrame, stopTime);

	//sprintf(Fname, "%s/2view_smothness_temporal/N_%d_%d.txt", Path, timeID, fileID);
	sprintf(Fname, "C:/temp/Sim/iTrack_%d.txt", fileID); FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("%s\n", Fname);
		return 0;
	}
	fscanf(fp, "%d ", &nviewers);
	for (int ii = 0; ii < nviewers; ii++)
	{
		fscanf(fp, "%d ", &vid);
		ViewerList.push_back(vid);
	}
	fscanf(fp, "%d ", &npts);

	vector<Point2d> *Tracks = new vector<Point2d>[npts*nviewers];
	vector<Point3d> *Tracks3D = new vector<Point3d>[npts];
	for (int kk = 0; kk < npts; kk++)
	{
		for (int jj = 0; jj < nviewers; jj++)
		{
			fscanf(fp, "%d %d %d ", &vid, &timeID, &stopTime);
			for (int ii = 0; ii < stopTime - timeID + 1; ii++)
			{
				fscanf(fp, "%lf %lf ", &u, &v);
				Tracks[jj + kk*nviewers].push_back(Point2d(u, v));
			}

			if (maxTime < stopTime)
				maxTime = stopTime;
		}
	}
	fclose(fp);

	//Verify correspondences
	bool passed;
	Point2d *PutativeCorres = new Point2d[nviewers];
	double *P = new double[12 * nviewers];
	double *A = new double[6 * nviews];
	double *B = new double[2 * nviews];
	double *tPs = new double[12 * nviews];
	vector<int> inliers, viewId;
	Point3d WC;
	bool nonlinear = true;
	double *PInliers = new double[12 * nviewers];
	double *uvInliers = new double[2 * nviewers];
	for (int jj = 0; jj < npts; jj++)
	{
		for (int ii = 0; ii < stopTime - timeID + 1; ii++)
		{
			int nvisible = 0;

			viewId.clear();
			for (int kk = 0; kk < nviewers; kk++)
			{
				int videoID = nframes * ViewerList[kk];
				if (Tracks[kk + jj*nviewers].size() > ii)
				{
					PutativeCorres[nvisible] = Tracks[kk + jj*nviewers][ii];
					for (int ll = 0; ll < 12; ll++)
						/*if (kk>1 && ii>499)
						P[12 * nvisible + ll] = AllVideoInfo.VideoInfo[videoID + 499 + timeID].P[ll];
						else
						P[12 * nvisible + ll] = AllVideoInfo.VideoInfo[videoID + ii + timeID].P[ll];*/
						P[12 * nvisible + ll] = AllVideoInfo.VideoInfo[videoID + timeID].P[ll];
					viewId.push_back(kk);
					nvisible++;
				}
			}

			if (nvisible < 1) //not useful-->kill the tracjectory
			{
				for (int kk = 0; kk < nvisible; kk++)
					Tracks[viewId[kk] + jj*nviewers].erase(Tracks[viewId[kk] + jj*nviewers].begin() + ii, Tracks[viewId[kk] + jj*nviewers].end());
				break;
			}
			else
			{
				inliers.clear();
				double error = NviewTriangulationRANSAC(PutativeCorres, P, &WC, &passed, &inliers, nvisible, 1, 10, 0.75, 8.0, A, B, tPs);
				if (!passed) //not useful-->kill the tracjectory
				{
					for (int kk = 0; kk < nvisible; kk++)
						Tracks[viewId[kk] + jj*nviewers].erase(Tracks[viewId[kk] + jj*nviewers].begin() + ii, Tracks[viewId[kk] + jj*nviewers].end());
					break;
				}
				else
				{
					int ninlier = 0;
					for (int kk = 0; kk < nvisible; kk++)
						if (inliers.at(kk) == 1)
							ninlier++;

					if (ninlier < 1) //not useful-->kill the trajectory
					{
						for (int kk = 0; kk < nvisible; kk++)
							Tracks[viewId[kk] + jj*nviewers].erase(Tracks[viewId[kk] + jj*nviewers].begin() + ii, Tracks[viewId[kk] + jj*nviewers].end());
						break;
					}
					else
					{
						for (int kk = 0; kk < nvisible; kk++)
						{
							if (inliers.at(kk) == 0)//remove tracks for that view
								Tracks[viewId[kk] + jj*nviewers].erase(Tracks[viewId[kk] + jj*nviewers].begin() + ii, Tracks[viewId[kk] + jj*nviewers].end());
						}
						Tracks3D[jj].push_back(WC);
					}
				}
			}
		}
	}

	//Write Trajectory recon input
	sprintf(Fname, "C:/temp/Sim/TrackIn2_%d.txt", fileID); fp = fopen(Fname, "w+");
	for (int ii = 0; ii < nviewers; ii++)
	{
		int count = 0;
		for (int jj = timeID; jj < maxTime; jj++)//= (2 * syncUncertainty + 1)* nviewers)
		{
			//printf calibration data
			int videoID = nframes*ViewerList[ii];
			int tid = jj;// +(2 * syncUncertainty + 1)* ii;
			if (ii == 1)
				tid -= 3;
			int reachTrackFailture = 0;
			for (int kk = 0; kk < npts; kk++)
				if (tid - timeID + 1> Tracks[kk*nviewers + ii].size())
					reachTrackFailture++;
			if (reachTrackFailture == npts)
				break;
			fprintf(fp, "%d %d\n", ii, count);
			//if (ii == 0)
			//fprintf(fp, "%d %d\n", ii, tid);
			//else
			//fprintf(fp, "%d %d\n", ii, tid + fileID);

			for (int kk = 0; kk < 3; kk++)
				//if (ii>1 && tid>499)
				//fprintf(fp, "%f ", AllVideoInfo.VideoInfo[videoID + 499].camCenter[kk]);
				//else
				//fprintf(fp, "%f ", AllVideoInfo.VideoInfo[videoID + tid].camCenter[kk]);
				fprintf(fp, "%f ", AllVideoInfo.VideoInfo[videoID + timeID].camCenter[kk]);
			fprintf(fp, "\n");
			for (int kk = 0; kk < 3; kk++)
			{
				for (int ll = 0; ll < 3; ll++)
					//if (ii>1 && tid>499)
					//	fprintf(fp, "%f ", AllVideoInfo.VideoInfo[videoID + 499].R[ll + kk * 3]);
					//else
					//fprintf(fp, "%f ", AllVideoInfo.VideoInfo[videoID + tid].R[ll + kk * 3]);
					fprintf(fp, "%f ", AllVideoInfo.VideoInfo[videoID + timeID].R[ll + kk * 3]);
				fprintf(fp, "\n");
			}
			for (int kk = 0; kk < 3; kk++)
			{
				for (int ll = 0; ll < 3; ll++)
					//fprintf(fp, "%f ", AllVideoInfo.VideoInfo[videoID + tid].K[ll + kk * 3]);
					fprintf(fp, "%f ", AllVideoInfo.VideoInfo[videoID + timeID].K[ll + kk * 3]);
				fprintf(fp, "\n");
			}
			for (int kk = 0; kk < 3; kk++)
			{
				for (int ll = 0; ll < 4; ll++)
					//if (ii>1 && tid>499)
					//	fprintf(fp, "%f ", AllVideoInfo.VideoInfo[videoID + 499].P[ll + kk * 4]);
					//else
					//	fprintf(fp, "%f ", AllVideoInfo.VideoInfo[videoID + tid].P[ll + kk * 4]);
					fprintf(fp, "%.16f ", AllVideoInfo.VideoInfo[videoID + timeID].P[ll + kk * 4]);
				fprintf(fp, "\n");
			}

			//print points
			int ninlier = 0;
			for (int kk = 0; kk < npts; kk++)
			{
				if (tid - timeID + 1> Tracks[kk*nviewers + ii].size())
					fprintf(fp, "%d %d ", -1, -1);
				else
					fprintf(fp, "%.16f %.16f ", Tracks[kk*nviewers + ii][tid - timeID].x, Tracks[kk*nviewers + ii][tid - timeID].y);
				fprintf(fp, "%.16f %.16f %.16f %.16f\n", Tracks3D[kk][tid - timeID].x, Tracks3D[kk][tid - timeID].y, Tracks3D[kk][tid - timeID].z, 1.0*count / alpha);

				double P[12], XYZ[3] = { Tracks3D[kk][tid - timeID].x, Tracks3D[kk][tid - timeID].y, Tracks3D[kk][tid - timeID].z };
				for (int ll = 0; ll < 12; ll++)
					//if (ii>1 && tid > 499)
					//	P[ll] = AllVideoInfo.VideoInfo[videoID + 499].P[ll];
					//else
					//	P[ll] = AllVideoInfo.VideoInfo[videoID + tid].P[ll];
					P[ll] = AllVideoInfo.VideoInfo[videoID + timeID].P[ll];

				double denum = P[8] * XYZ[0] + P[9] * XYZ[1] + P[10] * XYZ[2] + P[11];
				double numX = P[0] * XYZ[0] + P[1] * XYZ[1] + P[2] * XYZ[2] + P[3];
				double numY = P[4] * XYZ[0] + P[5] * XYZ[1] + P[6] * XYZ[2] + P[7];

				double residualsX = (numX / denum - Tracks[kk*nviewers + ii][tid - timeID].x);
				double residualsY = (numY / denum - Tracks[kk*nviewers + ii][tid - timeID].y);
				double Residual = residualsX*residualsX + residualsY *residualsY;
				if (Residual > 10)
					int a = 0;
			}
			count++;
		}
	}
	fclose(fp);

	/*for (int offset = 0; offset >= -100; offset -= 1)
	{
	sprintf(Fname, "C:/temp/x_%d_%d.txt", offset, fileID); fp = fopen(Fname, "w+");
	sprintf(Fname, "C:/temp/xyz_%d_%d.txt", offset, fileID); FILE *fp2 = fopen(Fname, "w+");
	for (int kk = 0; kk < npts; kk++)
	{
	for (int jj = timeID; jj < maxTime; jj += (2 * syncUncertainty + 1)* nviewers)
	{
	int ninlier = 0;
	double C0[3], C1[3], pcn[4], r0[3], r1[3], s, t;
	for (int ii = 0; ii < nviewers; ii++)
	{
	int videoID = nframes*ViewerList[ii];
	int tid = jj;
	if (ii == 1)
	tid += offset;
	if (tid - timeID + 1> Tracks[kk*nviewers + ii].size())
	continue;

	double XYZ[3] = { Tracks3D[kk][tid - timeID].x, Tracks3D[kk][tid - timeID].y, Tracks3D[kk][tid - timeID].z };
	for (int ll = 0; ll < 12; ll++)
	//if (ii>1 && tid > 499)
	//PInliers[12 * ninlier + ll] = AllVideoInfo.VideoInfo[videoID + 499].P[ll];
	//else
	//PInliers[12 * ninlier + ll] = AllVideoInfo.VideoInfo[videoID + tid].P[ll];
	PInliers[12 * ninlier + ll] = AllVideoInfo.VideoInfo[videoID + timeID].P[ll];

	uvInliers[2 * ninlier] = Tracks[kk*nviewers + ii][tid - timeID].x, uvInliers[2 * ninlier + 1] = Tracks[kk*nviewers + ii][tid - timeID].y;

	//d = iR*(lamda*iK*[u,v,1] - T) - C
	pcn[0] = AllVideoInfo.VideoInfo[videoID + tid].invK[0] * uvInliers[2 * ninlier] + AllVideoInfo.VideoInfo[videoID + tid].invK[1] * uvInliers[2 * ninlier + 1] + AllVideoInfo.VideoInfo[videoID + tid].invK[2];
	pcn[1] = AllVideoInfo.VideoInfo[videoID + tid].invK[4] * uvInliers[2 * ninlier + 1] + AllVideoInfo.VideoInfo[videoID + tid].invK[5];
	pcn[2] = 1.0;

	double iR[9], tt[3], ttt[3];
	//mat_transpose(AllVideoInfo.VideoInfo[videoID + tid].R, iR, 3, 3);
	mat_transpose(AllVideoInfo.VideoInfo[videoID + timeID].R, iR, 3, 3);
	//mat_subtract(pcn, AllVideoInfo.VideoInfo[videoID + tid].T, tt, 3, 1, 1000.0); //Scaling the pcn results in better numerical precision
	mat_subtract(pcn, AllVideoInfo.VideoInfo[videoID + timeID].T, tt, 3, 1, 1000.0); //Scaling the pcn results in better numerical precision
	mat_mul(iR, tt, ttt, 3, 3, 1);

	for (int ll = 0; ll < 3; ll++)
	if (ii == 0)
	//r0[ll] = ttt[ll], C0[ll] = AllVideoInfo.VideoInfo[videoID + tid].camCenter[ll];
	r0[ll] = ttt[ll], C0[ll] = AllVideoInfo.VideoInfo[videoID + timeID].camCenter[ll];
	else
	//r1[ll] = ttt[ll], C1[ll] = AllVideoInfo.VideoInfo[videoID + tid].camCenter[ll];
	r1[ll] = ttt[ll], C1[ll] = AllVideoInfo.VideoInfo[videoID + timeID].camCenter[ll];

	ninlier++;
	Point3D[0] = XYZ[0], Point3D[1] = XYZ[1], Point3D[2] = XYZ[2];
	}

	if (ninlier > 1)
	{
	for (int ll = 0; ll < 3; ll++)
	r0[ll] = C0[ll] - r0[ll], r1[ll] = C1[ll] - r1[ll];
	normalize(r0), normalize(r1);
	double reprojError1 = MinDistanceTwoLines(C0, r0, C1, r1, s, t);
	NviewTriangulationNonLinear(PInliers, uvInliers, Point3D, &reprojError, ninlier);
	fprintf(fp, "%.16f %.16f\n", reprojError1, reprojError);
	fprintf(fp2, "%.16f %.16f %.16f\n", C0[0] + r0[0] * s, C0[1] + r0[1] * s, C0[2] + r0[2] * s);
	}
	}

	}
	fclose(fp);
	fclose(fp2);
	}*/

	delete[]Tracks;
	delete[]PutativeCorres, delete[]P, delete[]A, delete[]B, delete[]tPs;

	return 0;
}
int TestTrajectoryProjection(char *Path, int nviews, int startTime, int stopTime)
{
	char Fname[200];
	int ntimes, npts, frameID, vid, timeID, ntracks;
	double t;
	vector<int> frameIDList;
	vector<int> viewID;

	sprintf(Fname, "%s/TrackIn/In_1_41.txt", Path);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return 1;
	}
	fscanf(fp, "%s %d", Fname, &ntimes);
	fscanf(fp, "%s %d", Fname, &npts);
	for (int jj = 0; jj < ntimes; jj++)
	{
		fscanf(fp, "%d %d", &vid, &frameID);
		viewID.push_back(vid);
		frameIDList.push_back(frameID);
		for (int ii = 0; ii < 21; ii++)
			fscanf(fp, "%lf ", &t);
		for (int ii = 0; ii < npts; ii++)
			fscanf(fp, "%lf %lf ", &t, &t);
	}
	fclose(fp);

	int *sortedTime = new int[ntimes];
	int *sortedView = new int[ntimes];
	int *T = new int[ntimes];

	for (int ii = 0; ii < ntimes; ii++)
	{
		sortedTime[ii] = frameIDList[ii];
		T[ii] = ii;
	}
	Quick_Sort_Int(sortedTime, T, 0, ntimes - 1);

	for (int ii = 0; ii < ntimes; ii++)
		sortedView[ii] = viewID[T[ii]];

	fp = fopen("C:/temp/Fil_1_3.txt", "w+");
	if (fp == NULL)
	{
		return 1;
	}
	fscanf(fp, "%s %d", Fname, &npts);
	double x, y, z;
	//vector<Point3d> *vWC = new vector<Point3d>[npts];
	vector<Point3d> vWC[1];
	for (int ii = 0; ii < npts; ii++)
	{
		fscanf(fp, "%d %d", &timeID, &ntracks);
		for (int jj = 0; jj < ntracks; jj++)
		{
			fscanf(fp, "%lf %lf %lf ", &x, &y, &z);
			vWC[ii].push_back(Point3d(x, y, z));
		}
	}
	fclose(fp);

	VideoData AllVideoInfo;
	if (ReadVideoData(Path, AllVideoInfo, nviews, startTime, stopTime) == 1)
		return 1;
	int nframes = max(MaxnFrame, stopTime);

	double P[12];
	Point2d pts;
	Mat img, colorImg;
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

	for (int kk = 0; kk < viewID.size(); kk++)
	{
		vid = viewID[kk];
		int videoID = vid*nframes;
		for (int jj = 0; jj < ntracks; jj++)
		{
			for (int ii = 0; ii < 12; ii++)
				P[ii] = AllVideoInfo.VideoInfo[videoID + jj + timeID].P[ii];

			sprintf(Fname, "%s/%08d/%08d_00_%02d.png", Path, jj + timeID, jj + timeID, vid);	img = imread(Fname, 0);
			if (img.empty())
			{
				printf("Cannot load %s\n", Fname);
				continue;
			}
			cvtColor(img, colorImg, CV_GRAY2RGB);

			for (int ii = 0; ii < npts; ii++)
			{
				ProjectandDistort(vWC[ii][jj], &pts, P, NULL, NULL, 1);
				circle(colorImg, pts, 4, colors[ii % 9], 1, 8, 0);
			}
			sprintf(Fname, "d:/Phuong/%d_%d.png", vid, jj + timeID); imwrite(Fname, colorImg);
		}
	}
	return 0;
}
int ReadTrajectory(PerCamNonRigidTrajectory *CamTraj, int nCams, int nTracks, int trackID = 0)
{
	char Fname[200];
	int ii, jj, camID, frameID;
	double x, y, z, t, u, v;
	CamCenter Ccenter;
	RotMatrix Rmat;
	Quaternion Qmat;
	KMatrix Kmat;
	Pmat P;

	vector<double> *xyz = new vector<double>[nCams];
	vector<double> *uv = new vector<double>[nCams];

	for (ii = 0; ii < nCams; ii++)
		CamTraj[ii].nTracks = 0,
		CamTraj[ii].Track2DInfo = new Track2D[nTracks],
		CamTraj[ii].Track3DInfo = new Track3D[nTracks];

	for (ii = 0; ii < nTracks; ii++)
	{
		for (jj = 0; jj < nCams; jj++)
			xyz[jj].clear(), uv[jj].clear();

		sprintf(Fname, "C:/temp/Sim/TrackIn2_%d.txt", ii + 1); FILE *fp = fopen(Fname, "r");
		//sprintf(Fname, "C:/temp/2view_smothness_temporal/TrackIn3_%d.txt", trackID); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("Cannot load %s\n", Fname);
			continue;
		}
		fscanf(fp, "%s %d", Fname, &jj);
		fscanf(fp, "%s %d", Fname, &jj);
		while (fscanf(fp, "%d %d ", &camID, &frameID) != EOF)
		{
			for (int jj = 0; jj < 3; jj++)
				fscanf(fp, "%lf ", &Ccenter.C[jj]);
			for (int jj = 0; jj < 9; jj++)
				fscanf(fp, "%lf ", &Rmat.R[jj]);
			for (int jj = 0; jj < 9; jj++)
				fscanf(fp, "%lf ", &Kmat.K[jj]);
			for (int jj = 0; jj < 12; jj++)
				fscanf(fp, "%lf ", &P.P[jj]);

			fscanf(fp, "%lf %lf ", &u, &v);
			fscanf(fp, "%lf %lf %lf %lf ", &x, &y, &z, &t);

			//x += 50.0*(1.0*rand() / RAND_MAX - 0.5);
			//y += 50.0*(1.0*rand() / RAND_MAX - 0.5);
			//z += 50.0*(1.0*rand() / RAND_MAX - 0.5);

			if (frameID == 0)
				CamTraj[camID].F = t;

			if (frameID >= CamTraj[camID].P.size())
			{
				CamTraj[camID].P.push_back(P);
				CamTraj[camID].K.push_back(Kmat);
				CamTraj[camID].C.push_back(Ccenter);
				CamTraj[camID].R.push_back(Rmat);
				Rotation2Quaternion(Rmat.R, Qmat.quad);
				CamTraj[camID].Q.push_back(Qmat);
			}

			xyz[camID].push_back(x), xyz[camID].push_back(y), xyz[camID].push_back(z);
			uv[camID].push_back(u), uv[camID].push_back(v);
		}
		fclose(fp);

		//Copy to trajectory data
		for (int jj = 0; jj < nCams; jj++)
		{
			int npts = uv[jj].size() / 2;

			CamTraj[jj].Track2DInfo[CamTraj[jj].nTracks].npts = npts;
			CamTraj[jj].Track3DInfo[CamTraj[jj].nTracks].npts = npts;

			CamTraj[jj].Track2DInfo[CamTraj[jj].nTracks].uv = new Point2d[npts];
			CamTraj[jj].Track3DInfo[CamTraj[jj].nTracks].xyz = new double[npts * 3];

			for (int kk = 0; kk < xyz[jj].size(); kk++)
				CamTraj[jj].Track3DInfo[CamTraj[jj].nTracks].xyz[kk] = xyz[jj][kk];
			for (int kk = 0; kk < uv[jj].size() / 2; kk++)
				CamTraj[jj].Track2DInfo[CamTraj[jj].nTracks].uv[kk].x = uv[jj][2 * kk],
				CamTraj[jj].Track2DInfo[CamTraj[jj].nTracks].uv[kk].y = uv[jj][2 * kk + 1];

			CamTraj[jj].nTracks++;
		}
	}

	delete[]xyz, delete[]uv;
	return 0;
}
int WriteTrajectory(PerCamNonRigidTrajectory *CamTraj, int nCams, int nTracks, double alpha = 1.0)
{
	int camID, trackID, frameID;
	double ialpha = 1.0 / alpha;
	FILE *fp = fopen("C:/temp/Sim/Results.txt", "w+");
	fprintf(fp, "Temporal alignment (ms): ");
	for (camID = 0; camID < nCams; camID++)
		fprintf(fp, "%f ", CamTraj[camID].F);
	fprintf(fp, "\n");

	int maxPts = 0;
	for (trackID = 0; trackID < nTracks; trackID++)
	{
		for (camID = 0; camID < nCams; camID++)
		{
			int npts = 0;
			for (frameID = 0; frameID < CamTraj[camID].Track3DInfo[trackID].npts; frameID++)
				npts++;
			if (npts > maxPts)
				maxPts = npts;
		}
	}
	int *index = new int[maxPts];
	double *CapturedTime = new double[maxPts];
	Point3d *XYZ = new Point3d[maxPts];

	for (trackID = 0; trackID < nTracks; trackID++)
	{
		int npts = 0;
		for (camID = 0; camID < nCams; camID++)
		{
			for (frameID = 0; frameID < CamTraj[camID].Track3DInfo[trackID].npts; frameID++)
			{
				XYZ[npts].x = CamTraj[camID].Track3DInfo[trackID].xyz[3 * frameID];
				XYZ[npts].y = CamTraj[camID].Track3DInfo[trackID].xyz[3 * frameID + 1];
				XYZ[npts].z = CamTraj[camID].Track3DInfo[trackID].xyz[3 * frameID + 2];
				CapturedTime[npts] = CamTraj[camID].F + ialpha*frameID;
				index[npts] = npts;
				npts++;
			}
		}
		Quick_Sort_Double(CapturedTime, index, 0, npts - 1);

		fprintf(fp, "3D track %d \n", trackID);
		for (int ii = 0; ii < npts; ii++)
			fprintf(fp, "%.3f %.3f %.3f ", XYZ[index[ii]].x, XYZ[index[ii]].y, XYZ[index[ii]].z);
		fprintf(fp, "\n");
	}
	fclose(fp);

	delete[]index, delete[]CapturedTime, delete[]XYZ;

	return 0;
}
struct SpatialSmoothnessCost {
	SpatialSmoothnessCost(int fid1, int fid2, double ialpha, double lambda) : fid1(fid1), fid2(fid2), ialpha(ialpha), lambda(lambda){}

	template <typename T>	bool operator()(const T* const X_ijk, const T* const X2_ijk, const T* const F_k, const T* const F2_k, T* residuals) 	const
	{
		for (int ii = 0; ii < 3; ii++)
			residuals[ii] = lambda*(X_ijk[ii] - X2_ijk[ii]) / (F_k[0] - F2_k[0] + ialpha*(fid1 - fid2));

		return true;
	}

	// Factory to hide the construction of the CostFunction object from the client code.
	static ceres::CostFunction* Create(const int fid1, const int fid2, const double ialpha, double lambda)
	{
		return (new ceres::AutoDiffCostFunction<SpatialSmoothnessCost, 3, 3, 3, 1, 1>(new SpatialSmoothnessCost(fid1, fid2, ialpha, lambda)));
	}

	int fid1, fid2;
	double ialpha, lambda;
};
struct PerCamSpatialSmoothnessCost {
	PerCamSpatialSmoothnessCost(double ialpha) : ialpha(ialpha) {}

	template <typename T>	bool operator()(const T* const X_KMinus, const T* const X_K, const T* const X_KPlus, T* residuals) 	const
	{
		T dX = -X_KMinus[0] + T(2.0) * X_K[0] - X_KPlus[0];
		T dY = -X_KMinus[1] + T(2.0) * X_K[1] - X_KPlus[1];
		T dZ = -X_KMinus[2] + T(2.0) * X_K[2] - X_KPlus[2];
		residuals[0] = sqrt(dX*dX + dY*dY + dZ*dZ);

		return true;
	}

	// Factory to hide the construction of the CostFunction object from the client code.
	static ceres::CostFunction* Create(const double ialpha)
	{
		return (new ceres::AutoDiffCostFunction<PerCamSpatialSmoothnessCost, 1, 3, 3, 3>(new PerCamSpatialSmoothnessCost(ialpha)));
	}

	double ialpha;
};
struct TemporalSmoothnessCost {
	TemporalSmoothnessCost() {}

	template <typename T>	bool operator()(const T* const F_k, const T* const F2_k, T* residuals) 	const
	{
		residuals[0] = F_k[0] - F2_k[0];

		return true;
	}

	// Factory to hide the construction of the CostFunction object from the client code.
	static ceres::CostFunction* Create()
	{
		return (new ceres::AutoDiffCostFunction<TemporalSmoothnessCost, 1, 1, 1>(new TemporalSmoothnessCost()));
	}
};
struct ImageProjectionCost {
	ImageProjectionCost(const double *inP, double u, double v, double inlambda)
	{
		for (int ii = 0; ii < 12; ii++)
			P[ii] = inP[ii];
		observed_x = u, observed_y = v;
		lambda = inlambda;
	}

	template <typename T>	bool operator()(const T* const XYZ, T* residuals) 	const
	{
		// Project to normalize coordinate
		T denum = P[8] * XYZ[0] + P[9] * XYZ[1] + P[10] * XYZ[2] + P[11];
		T numX = P[0] * XYZ[0] + P[1] * XYZ[1] + P[2] * XYZ[2] + P[3];
		T numY = P[4] * XYZ[0] + P[5] * XYZ[1] + P[6] * XYZ[2] + P[7];

		residuals[0] = (T)lambda*(numX / denum - T(observed_x));
		residuals[1] = (T)lambda*(numY / denum - T(observed_y));

		return true;
	}

	// Factory to hide the construction of the CostFunction object from the client code.
	static ceres::CostFunction* Create(const double *P, const double observed_x, const double observed_y, double lambda)
	{
		return (new ceres::AutoDiffCostFunction<ImageProjectionCost, 2, 3>(new ImageProjectionCost(P, observed_x, observed_y, lambda)));
	}

	double observed_x, observed_y, P[12], lambda;
};
int NonRigidTemporalCost(PerCamNonRigidTrajectory *CamTraj, int nCams, int nTracks, int TemporalRange, double ialpha, double lambda1, double lambda2)
{
	//Spatial smothness term
	double Cost1 = 0.0;
	for (int jj = 0; jj < nCams; jj++)
	{
		for (int ii = 0; ii < CamTraj[ii].nTracks; ii++)
		{
			for (int kk = 1; kk < CamTraj[jj].Track3DInfo[ii].npts - 1; kk++)
			{
				double residuals[3], Residual;
				for (int ll = 0; ll < 3; ll++)
					residuals[ll] = -CamTraj[jj].Track3DInfo[ii].xyz[3 * (kk - 1) + ll] + 2.0 * CamTraj[jj].Track3DInfo[ii].xyz[3 * kk + ll] - CamTraj[jj].Track3DInfo[ii].xyz[3 * (kk + 1) + ll];
				Residual = residuals[0] * residuals[0] + residuals[1] * residuals[1] + residuals[2] * residuals[2];
				Cost1 += Residual;
			}
		}
	}

	//k_X^J1_i - (k+T)_X^J2_i
	double Cost2 = 0.0;
	for (int jj = 0; jj < nCams; jj++)
	{
		for (int ii = 0; ii < CamTraj[ii].nTracks; ii++)
		{
			for (int jj_ = 0; jj_ < nCams; jj_++)
			{
				if (jj_ == jj)//Intra camera case
					continue;

				for (int kk = 0; kk < CamTraj[jj].Track3DInfo[ii].npts; kk++)
				{
					for (int kk_ = kk - TemporalRange; kk_ <= kk + TemporalRange; kk_++)
					{
						if (kk_ < 0)
							continue;
						if (kk_ >= CamTraj[jj_].Track3DInfo[ii].npts)
							continue;

						double residuals[3];
						for (int ll = 0; ll < 3; ll++)
							residuals[ll] = lambda2*(CamTraj[jj].Track3DInfo[ii].xyz[3 * kk + ll] - CamTraj[jj_].Track3DInfo[ii].xyz[3 * kk_ + ll]) / (CamTraj[jj].F - CamTraj[jj_].F + ialpha*(kk - kk_));
						double Resdiual = residuals[0] * residuals[0] + residuals[1] * residuals[1] + residuals[2] * residuals[2];
						Cost2 += Resdiual*Resdiual;
					}
				}
			}
		}
	}

	double Cost3 = 0.0;
	for (int jj = 0; jj < nCams; jj++)
	{
		for (int ii = 0; ii < nCams; ii++)
		{
			if (jj == ii)
				continue;
			Cost3 += pow(CamTraj[jj].F - CamTraj[ii].F, 2);
		}
	}

	double Cost4 = 0.0; //Image constraint
	for (int jj = 0; jj < nCams; jj++)
	{
		for (int ii = 0; ii < CamTraj[ii].nTracks; ii++)
		{
			for (int kk = 0; kk < CamTraj[jj].Track3DInfo[ii].npts; kk++)
			{
				double P[12], XYZ[3] = { CamTraj[jj].Track3DInfo[ii].xyz[3 * kk], CamTraj[jj].Track3DInfo[ii].xyz[3 * kk + 1], CamTraj[jj].Track3DInfo[ii].xyz[3 * kk + 2] };
				for (int ll = 0; ll < 12; ll++)
					P[ll] = CamTraj[jj].P[kk].P[ll];

				double denum = P[8] * XYZ[0] + P[9] * XYZ[1] + P[10] * XYZ[2] + P[11];
				double numX = P[0] * XYZ[0] + P[1] * XYZ[1] + P[2] * XYZ[2] + P[3];
				double numY = P[4] * XYZ[0] + P[5] * XYZ[1] + P[6] * XYZ[2] + P[7];

				double residualsX = lambda1*(numX / denum - CamTraj[jj].Track2DInfo[ii].uv[kk].x);
				double residualsY = lambda1*(numY / denum - CamTraj[jj].Track2DInfo[ii].uv[kk].y);
				double Residual = residualsX*residualsX + residualsY *residualsY;
				Cost4 += Residual;
			}
		}
	}
	printf("%f %f %f %f\n", Cost1, Cost2, Cost3, Cost4);
	return 0;
}
int NonRigidTemporalOptim()
{
	const int nCams = 2, nTracks = 4, TemporalRange = 1;
	const double Tscale = 1000.0, alpha = 50.0, ialpha = Tscale / alpha;
	double sqrtlambda = 1000.0, sqrtlambda2 = 100.0;
	PerCamNonRigidTrajectory CamTraj[nCams];

	ReadTrajectory(CamTraj, nCams, nTracks);

	//for (int ii = 1; ii < nCams; ii++)
	//	CamTraj[ii].F = Tscale * 2 * ii;
	CamTraj[0].F = 0, CamTraj[1].F = 1;//ialpha * 2;
	NonRigidTemporalCost(CamTraj, nCams, nTracks, TemporalRange, ialpha, sqrtlambda, sqrtlambda2);

	printf("Set up BA ...\n");
	//google::InitGoogleLogging("Sync");
	ceres::Problem problem;

	//Image projection constraint
	printf("Adding image projection cost...\n");
	for (int jj = 0; jj < nCams; jj++)
	{
		for (int ii = 0; ii < CamTraj[ii].nTracks; ii++)
		{
			for (int kk = 0; kk < CamTraj[jj].Track3DInfo[ii].npts; kk++)
			{
				ceres::CostFunction* cost_function = ImageProjectionCost::Create(CamTraj[jj].P[kk].P, CamTraj[jj].Track2DInfo[ii].uv[kk].x, CamTraj[jj].Track2DInfo[ii].uv[kk].y, sqrtlambda);
				problem.AddResidualBlock(cost_function, NULL, CamTraj[jj].Track3DInfo[ii].xyz + 3 * kk);
			}
		}
	}

	//Spatial smothness term
	/*printf("Adding intra camera spatial-temporal smoothness cost...\n");
	for (int jj = 0; jj < nCams; jj++)
	{
	for (int ii = 0; ii < CamTraj[ii].nTracks; ii++)
	{
	for (int kk = 1; kk < CamTraj[jj].Track3DInfo[ii].npts - 1; kk++)
	{
	ceres::CostFunction* cost_function = PerCamSpatialSmoothnessCost::Create(ialpha);
	problem.AddResidualBlock(cost_function, NULL, CamTraj[jj].Track3DInfo[ii].xyz + 3 * (kk - 1), CamTraj[jj].Track3DInfo[ii].xyz + 3 * kk, CamTraj[jj].Track3DInfo[ii].xyz + 3 * (kk + 1));
	}
	}
	}*/

	printf("Adding inter camera spatial-temporal smoothness cost...\n"); //k_X^J1_i - (k+T)_X^J2_i
	for (int jj = 0; jj < nCams; jj++)
	{
		for (int ii = 0; ii < CamTraj[ii].nTracks; ii++)
		{
			for (int jj_ = 0; jj_ < nCams; jj_++)
			{
				if (jj_ == jj)//Intra camera case
					continue;

				for (int kk = 0; kk < CamTraj[jj].Track3DInfo[ii].npts; kk++)
				{
					for (int kk_ = kk - TemporalRange; kk_ <= kk + TemporalRange; kk_++)
					{
						if (kk_ < 0)
							continue;
						if (kk_ >= CamTraj[jj_].Track3DInfo[ii].npts)
							continue;

						ceres::CostFunction* cost_function = SpatialSmoothnessCost::Create(kk, kk_, ialpha, sqrtlambda2);
						problem.AddResidualBlock(cost_function, NULL, CamTraj[jj].Track3DInfo[ii].xyz + 3 * kk, CamTraj[jj_].Track3DInfo[ii].xyz + 3 * kk_, &CamTraj[jj].F, &CamTraj[jj_].F);
					}
				}
			}
		}
	}

	//Temporal smoothness
	printf("Adding inter camera similar start time cost...\n");
	for (int jj = 0; jj < nCams; jj++)
	{
		for (int ii = 0; ii < nCams; ii++)
		{
			if (jj == ii)
				continue;
			ceres::CostFunction* cost_function = TemporalSmoothnessCost::Create();
			problem.AddResidualBlock(cost_function, NULL, &CamTraj[jj].F, &CamTraj[ii].F);
		}
	}

	//Add fixed parameter
	printf("Setting fixed parameters...\n");
	problem.SetParameterBlockConstant(&CamTraj[0].F);

	printf("Running optim..\n");
	ceres::Solver::Options options;
	options.num_threads = 4;
	options.max_num_iterations = 50;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	options.trust_region_strategy_type = ceres::DOGLEG;
	options.use_nonmonotonic_steps = true;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";

	NonRigidTemporalCost(CamTraj, nCams, nTracks, TemporalRange, ialpha, sqrtlambda, sqrtlambda2);

	//Store the results	
	printf("Write results ....\n");
	WriteTrajectory(CamTraj, nCams, nTracks, alpha / Tscale);

	printf("Done!\n");

	return 0;
}

//Nonlinear Optimization for 3D Patch estimation
struct TemporalOptimInterpCeres {
	TemporalOptimInterpCeres(double *Pin, double *ParaXin, double *ParaYin, int frameIDin, int nframesIn, int interpAlgoIn)
	{
		P = Pin;
		ParaX = ParaXin, ParaY = ParaYin;
		double x = ParaXin[0], y = ParaX[0];
		frameID = frameIDin, nframes = nframesIn, interpAlgo = interpAlgoIn;
	}

	template <typename T>	bool operator()(const T* const XYZ, const T* const F, T* residuals) 	const
	{
		double denum = P[8] * XYZ[0] + P[9] * XYZ[1] + P[10] * XYZ[2] + P[11];
		double numX = P[0] * XYZ[0] + P[1] * XYZ[1] + P[2] * XYZ[2] + P[3];
		double numY = P[4] * XYZ[0] + P[5] * XYZ[1] + P[6] * XYZ[2] + P[7];

		double Fi = F[0] + frameID;
		double Sx[3], Sy[3];
		if (Fi < 0.0)
			Fi = 0.0;
		if (Fi>nframes - 1)
			Fi = nframes - 1;

		Get_Value_Spline(ParaX, nframes, 1, Fi, 0, Sx, -1, interpAlgo);
		Get_Value_Spline(ParaY, nframes, 1, Fi, 0, Sy, -1, interpAlgo);

		residuals[0] = numX / denum - Sx[0];
		residuals[1] = numY / denum - Sy[0];


		return true;
	}

	// Factory to hide the construction of the CostFunction object from the client code.
	static ceres::CostFunction* Create(double *Pin, double *ParaXin, double *ParaYin, int frameIDin, int nframesIn, int interpAlgoIn)
	{
		return (new ceres::NumericDiffCostFunction<TemporalOptimInterpCeres, ceres::CENTRAL, 2, 3, 1>(new TemporalOptimInterpCeres(Pin, ParaXin, ParaYin, frameIDin, nframesIn, interpAlgoIn)));
	}

	int frameID, nframes, interpAlgo;
	double F;
	double *ParaX, *ParaY, *P;
};
struct TemporalOptimInterpCeres2 {
	TemporalOptimInterpCeres2(double *Pin, double *ParaXin, double *ParaYin, int frameIDin, int nframesIn, int interpAlgoIn)
	{
		P = Pin;
		ParaX = ParaXin, ParaY = ParaYin;
		double x = ParaXin[0], y = ParaX[0];
		frameID = frameIDin, nframes = nframesIn, interpAlgo = interpAlgoIn;
	}

	template <typename T>	bool operator()(const T* const XYZ, const T* const F, T* residuals) 	const
	{
		double Fi = F[0] + frameID;
		if (Fi < 0.0)
			Fi = 0.0;
		if (Fi>nframes - 1)
			Fi = nframes - 1;

		int rFi = (int)(Fi + 0.5);
		double iP[] = { P[12 * rFi], P[12 * rFi + 1], P[12 * rFi + 2], P[12 * rFi + 3],
			P[12 * rFi + 4], P[12 * rFi + 5], P[12 * rFi + 6], P[12 * rFi + 7],
			P[12 * rFi + 8], P[12 * rFi + 9], P[12 * rFi + 10], P[12 * rFi + 11] };

		double denum = iP[8] * XYZ[0] + iP[9] * XYZ[1] + iP[10] * XYZ[2] + iP[11];
		double numX = iP[0] * XYZ[0] + iP[1] * XYZ[1] + iP[2] * XYZ[2] + iP[3];
		double numY = iP[4] * XYZ[0] + iP[5] * XYZ[1] + iP[6] * XYZ[2] + iP[7];

		double Sx[3], Sy[3];
		Get_Value_Spline(ParaX, nframes, 1, Fi, 0, Sx, -1, interpAlgo);
		Get_Value_Spline(ParaY, nframes, 1, Fi, 0, Sy, -1, interpAlgo);

		residuals[0] = numX / denum - Sx[0];
		residuals[1] = numY / denum - Sy[0];


		return true;
	}

	// Factory to hide the construction of the CostFunction object from the client code.
	static ceres::CostFunction* Create(double *Pin, double *ParaXin, double *ParaYin, int frameIDin, int nframesIn, int interpAlgoIn)
	{
		return (new ceres::NumericDiffCostFunction<TemporalOptimInterpCeres2, ceres::CENTRAL, 2, 3, 1>(new TemporalOptimInterpCeres2(Pin, ParaXin, ParaYin, frameIDin, nframesIn, interpAlgoIn)));
	}

	int frameID, nframes, interpAlgo;
	double F;
	double *ParaX, *ParaY, *P;
};
struct TemporalOptimInterpCeres3 {
	TemporalOptimInterpCeres3(double *AllPin, double *AllKin, double *AllQin, double *AllRin, double *AllCin, double *ParaCamCenterXIn, double *ParaCamCenterYIn, double *ParaCamCenterZIn, double *ParaXin, double *ParaYin, int frameIDin, int nframesIn, int interpAlgoIn)
	{
		AllP = AllPin, AllK = AllKin, AllQ = AllQin, AllR = AllRin, AllC = AllCin;
		ParaCamCenterX = ParaCamCenterXIn, ParaCamCenterY = ParaCamCenterYIn, ParaCamCenterZ = ParaCamCenterZIn, ParaX = ParaXin, ParaY = ParaYin;
		frameID = frameIDin, nframes = nframesIn, interpAlgo = interpAlgoIn;
	}

	template <typename T>	bool operator()(const T* const XYZ, const T* const F, T* residuals) 	const
	{
		double Fi = F[0] + frameID;
		if (Fi < 0.0)
			Fi = 0.0;
		if (Fi>nframes - 2)
			Fi = nframes - 2;
		int lFi = (int)Fi, uFi = lFi + 1;
		double fFi = Fi - lFi;

		if (lFi < 0)
		{
			residuals[0] = 0.0;
			residuals[1] = 0.0;
			return true;
		}
		else if (uFi > nframes - 2)
		{
			residuals[0] = 0.0;
			residuals[1] = 0.0;
			return true;
		}

		double K[9], C[3], R[9], RT[12], P[12], Q[4];
		for (int ll = 0; ll < 9; ll++)
			K[ll] = AllK[9 * lFi + ll];

		for (int ll = 0; ll < 3; ll++)
			C[ll] = (1.0 - fFi)*AllC[3 * lFi + ll] + fFi*AllC[3 * uFi + ll]; //linear interpolation
		//Get_Value_Spline(ParaCamCenterX, nframes, 1, Fi, 0, &C[0], -1, interpAlgo);
		//Get_Value_Spline(ParaCamCenterY, nframes, 1, Fi, 0, &C[1], -1, interpAlgo);
		//Get_Value_Spline(ParaCamCenterZ, nframes, 1, Fi, 0, &C[2], -1, interpAlgo);


		for (int ll = 0; ll < 4; ll++)
			Q[ll] = AllQ[4 * lFi + ll];
		//QuaternionLinearInterp(&AllQ[4 * lFi], &AllQ[4 * uFi], Q, fFi);//linear interpolation

		/*//Give good result given 1frame offset--> strange so I use rigorous interplation instead
		lFi = (int)(Fi + 0.5);
		for (int ll = 0; ll < 3; ll++)
		C[ll] = AllC[3 * lFi + ll];
		for (int ll = 0; ll < 4; ll++)
		Q[ll] = AllQ[4 * lFi + ll];*/

		Quaternion2Rotation(Q, R);
		AssembleRT(R, C, RT, true);
		AssembleP(K, RT, P);

		double numX = P[0] * XYZ[0] + P[1] * XYZ[1] + P[2] * XYZ[2] + P[3];
		double numY = P[4] * XYZ[0] + P[5] * XYZ[1] + P[6] * XYZ[2] + P[7];
		double denum = P[8] * XYZ[0] + P[9] * XYZ[1] + P[10] * XYZ[2] + P[11];

		double Sx[3], Sy[3];
		Get_Value_Spline(ParaX, nframes, 1, Fi, 0, Sx, -1, interpAlgo);
		Get_Value_Spline(ParaY, nframes, 1, Fi, 0, Sy, -1, interpAlgo);

		residuals[0] = numX / denum - Sx[0];
		residuals[1] = numY / denum - Sy[0];
		if (abs(residuals[0]) > 5 || abs(residuals[1]) > 5)
			int a = 0;
		return true;
	}

	// Factory to hide the construction of the CostFunction object from the client code.
	static ceres::CostFunction* Create(double *AllPin, double *AllKin, double *AllQin, double *AllRin, double *AllCin, double *ParaCamCenterXIn, double *ParaCamCenterYin, double *ParaCamCenterZin, double *ParaXin, double *ParaYin, int frameIDin, int nframesIn, int interpAlgoIn)
	{
		return (new ceres::NumericDiffCostFunction<TemporalOptimInterpCeres3, ceres::CENTRAL, 2, 3, 1>(new TemporalOptimInterpCeres3(AllPin, AllKin, AllQin, AllRin, AllCin, ParaCamCenterXIn, ParaCamCenterYin, ParaCamCenterZin, ParaXin, ParaYin, frameIDin, nframesIn, interpAlgoIn)));
	}

	int frameID, nframes, interpAlgo;
	double F;
	double *ParaCamCenterX, *ParaCamCenterY, *ParaCamCenterZ, *ParaX, *ParaY;
	double *AllP, *AllK, *AllQ, *AllR, *AllC;
};
int TemporalOptimInterp()
{
	const int nCams = 3, nTracks = 4;
	PerCamNonRigidTrajectory CamTraj[nCams];

	ReadTrajectory(CamTraj, nCams, nTracks);

	//Interpolate the trajectory of 2d tracks
	int maxPts = 0;
	for (int ii = 0; ii < nTracks; ii++)
	{
		for (int jj = 0; jj < nCams; jj++)
		{
			int npts = 0;
			for (int kk = 0; kk < CamTraj[jj].Track3DInfo[ii].npts; kk++)
				npts++;
			if (npts > maxPts)
				maxPts = npts;
		}
	}

	int InterpAlgo = 1;
	double *x = new double[maxPts], *y = new double[maxPts];
	double* AllQuaternion = new double[4 * nCams*maxPts];
	double* AllRotationMat = new double[9 * nCams*maxPts];
	double* AllCamCenter = new double[3 * nCams*maxPts];
	double *AllKMatrix = new double[9 * nCams*maxPts];
	double *AllPMatrix = new double[12 * nCams*maxPts];

	double *z = new double[maxPts];
	double *ParaCamCenterX = new double[nCams*maxPts];
	double *ParaCamCenterY = new double[nCams*maxPts];
	double *ParaCamCenterZ = new double[nCams*maxPts];
	for (int jj = 0; jj < nCams; jj++)
	{
		for (int ii = 0; ii < CamTraj[jj].nTracks; ii++)
		{
			int npts = CamTraj[jj].Track3DInfo[ii].npts;
			CamTraj[jj].Track2DInfo[ii].ParaX = new double[npts];
			CamTraj[jj].Track2DInfo[ii].ParaY = new double[npts];

			for (int kk = 0; kk < npts; kk++)
				x[kk] = CamTraj[jj].Track2DInfo[ii].uv[kk].x, y[kk] = CamTraj[jj].Track2DInfo[ii].uv[kk].y;
			Generate_Para_Spline(x, CamTraj[jj].Track2DInfo[ii].ParaX, npts, 1, InterpAlgo);
			Generate_Para_Spline(y, CamTraj[jj].Track2DInfo[ii].ParaY, npts, 1, InterpAlgo);
		}
		for (int ii = 0; ii < maxPts; ii++)
		{
			if (ii >= CamTraj[jj].R.size())
				continue;
			for (int kk = 0; kk < 9; kk++)
				AllKMatrix[9 * jj * maxPts + 9 * ii + kk] = CamTraj[jj].K[ii].K[kk];
			for (int kk = 0; kk < 3; kk++)
				AllCamCenter[3 * jj * maxPts + 3 * ii + kk] = CamTraj[jj].C[ii].C[kk];
			for (int kk = 0; kk < 9; kk++)
				AllRotationMat[9 * jj * maxPts + 9 * ii + kk] = CamTraj[jj].R[ii].R[kk];
			for (int kk = 0; kk < 4; kk++)
				AllQuaternion[4 * jj * maxPts + 4 * ii + kk] = CamTraj[jj].Q[ii].quad[kk];
			for (int kk = 0; kk < 12; kk++)
				AllPMatrix[12 * jj * maxPts + 12 * ii + kk] = CamTraj[jj].P[ii].P[kk];
		}

		for (int ii = 0; ii < maxPts; ii++)
		{
			if (ii >= CamTraj[jj].R.size())
				continue;
			x[ii] = AllCamCenter[3 * jj * maxPts + 3 * ii];
			y[ii] = AllCamCenter[3 * jj * maxPts + 3 * ii + 1];
			z[ii] = AllCamCenter[3 * jj * maxPts + 3 * ii + 2];
		}
		Generate_Para_Spline(x, ParaCamCenterX + jj*maxPts, maxPts, 1, InterpAlgo);
		Generate_Para_Spline(y, ParaCamCenterY + jj*maxPts, maxPts, 1, InterpAlgo);
		Generate_Para_Spline(z, ParaCamCenterZ + jj*maxPts, maxPts, 1, InterpAlgo);
	}
	delete[]x, delete[]y;
	delete[]z;

	//Initialize temporal info
	for (int ii = 0; ii < nCams; ii++)
		CamTraj[ii].F = round(10.0*(1.0*rand() / RAND_MAX - 0.5));
	CamTraj[0].F = 0, CamTraj[1].F = -3.0, CamTraj[2].F = 2.0;

	printf("Inital offset: ");
	for (int ii = 0; ii < nCams; ii++)
		printf("%f ", CamTraj[ii].F);
	printf("\n");

	//printf("Set up BA ...\n");
	ceres::Problem problem;

	//printf("Adding image projection cost ...\n");
	double Error = 0.0;
	for (int ii = 0; ii < nTracks; ii++)
	{
		//find maxtracks
		int maxTracks = 0, maxCam = 0;
		for (int jj = 0; jj < nCams; jj++)
			if (maxTracks < CamTraj[jj].Track3DInfo[ii].npts)
				maxTracks = CamTraj[jj].Track3DInfo[ii].npts, maxCam = jj;

		for (int kk = 0; kk < maxTracks; kk++)
		{
			for (int jj = 0; jj < nCams; jj++)
			{
				if (kk>CamTraj[jj].Track3DInfo[ii].npts || kk >= CamTraj[jj].R.size())
					continue;

				double Fi = CamTraj[jj].F + kk;
				if (Fi < 0.0)
					Fi = 0.0;
				if (Fi>CamTraj[jj].Track3DInfo[ii].npts - 2)
					Fi = CamTraj[jj].Track3DInfo[ii].npts - 2;
				int lFi = (int)Fi, uFi = lFi + 1, rFi = (int)(Fi + 0.5);
				double fFi = Fi - lFi;

				double K[9], C[3], Q[4], R[9], RT[12], P[12];
				for (int ll = 0; ll < 9; ll++)
					K[ll] = AllKMatrix[9 * jj*maxPts + 9 * lFi + ll];
				for (int ll = 0; ll < 3; ll++)
					C[ll] = (1.0 - fFi)*AllCamCenter[3 * jj*maxPts + 3 * lFi + ll] + fFi*AllCamCenter[3 * jj*maxPts + 3 * uFi + ll]; //linear interpolation

				for (int ll = 0; ll < 4; ll++)
					Q[ll] = AllQuaternion[4 * jj*maxPts + 4 * lFi + ll];
				//QuaternionLinearInterp(&AllQuaternion[4 * jj*maxPts + 4 * lFi], &AllQuaternion[4 * jj*maxPts + 4 * uFi], Q, fFi);//linear interpolation
				//Get_Value_Spline(ParaCamCenterX + jj*maxPts, maxPts, 1, Fi, 0, &C[0], -1, InterpAlgo);
				//Get_Value_Spline(ParaCamCenterY + jj*maxPts, maxPts, 1, Fi, 0, &C[1], -1, InterpAlgo);
				//Get_Value_Spline(ParaCamCenterZ + jj*maxPts, maxPts, 1, Fi, 0, &C[2], -1, InterpAlgo);

				Quaternion2Rotation(Q, R);
				AssembleRT(R, C, RT, true);
				AssembleP(K, RT, P);

				double XYZ[] = { CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk], CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk + 1], CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk + 2] };
				double denum = P[8] * XYZ[0] + P[9] * XYZ[1] + P[10] * XYZ[2] + P[11];
				double numX = P[0] * XYZ[0] + P[1] * XYZ[1] + P[2] * XYZ[2] + P[3];
				double numY = P[4] * XYZ[0] + P[5] * XYZ[1] + P[6] * XYZ[2] + P[7];

				double Sx, Sy;
				Get_Value_Spline(CamTraj[jj].Track2DInfo[ii].ParaX, CamTraj[jj].Track3DInfo[ii].npts, 1, Fi, 0, &Sx, -1, InterpAlgo);
				Get_Value_Spline(CamTraj[jj].Track2DInfo[ii].ParaY, CamTraj[jj].Track3DInfo[ii].npts, 1, Fi, 0, &Sy, -1, InterpAlgo);

				double residualsX = numX / denum - Sx;
				double residualsY = numY / denum - Sy;
				double Residual = residualsX*residualsX + residualsY*residualsY;
				Error += Residual;

				//ceres::CostFunction* cost_function = TemporalOptimInterpCeres2::Create(CamTraj[jj].P[kk].P, CamTraj[jj].Track2DInfo[ii].ParaX, CamTraj[jj].Track2DInfo[ii].ParaY, kk, CamTraj[jj].Track3DInfo[ii].npts, InterpAlgo);
				ceres::CostFunction* cost_function = TemporalOptimInterpCeres3::Create(&AllPMatrix[12 * jj*maxPts], &AllKMatrix[9 * jj*maxPts], &AllQuaternion[4 * jj*maxPts], &AllRotationMat[9 * jj*maxPts], &AllCamCenter[3 * jj*maxPts],
					ParaCamCenterX + jj*maxPts, ParaCamCenterY + jj*maxPts, ParaCamCenterZ + jj*maxPts, CamTraj[jj].Track2DInfo[ii].ParaX, CamTraj[jj].Track2DInfo[ii].ParaY, kk, CamTraj[jj].Track3DInfo[ii].npts, InterpAlgo);
				problem.AddResidualBlock(cost_function, NULL, CamTraj[maxCam].Track3DInfo[ii].xyz + 3 * kk, &CamTraj[jj].F);
			}
		}
	}
	printf("Initial error: %.6e\n", Error);

	//printf("Setting fixed parameters...\n");
	problem.SetParameterBlockConstant(&CamTraj[0].F);

	//printf("Running optim..\n");
	ceres::Solver::Options options;
	options.num_threads = 4;
	options.max_num_iterations = 50;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	options.trust_region_strategy_type = ceres::DOGLEG;
	options.use_nonmonotonic_steps = true;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";

	Error = 0.0;
	FILE *fp = fopen("C:/temp/Sim/Results.txt", "w+");
	fprintf(fp, "Temporal alignment (ms): ");
	for (int ii = 0; ii < nCams; ii++)
		fprintf(fp, "%f ", CamTraj[ii].F);
	fprintf(fp, "\n");
	for (int ii = 0; ii < nTracks; ii++)
	{
		//find maxtracks
		int maxTracks = 0, maxCam = 0;
		for (int jj = 0; jj < nCams; jj++)
			if (maxTracks < CamTraj[jj].Track3DInfo[ii].npts)
				maxTracks = CamTraj[jj].Track3DInfo[ii].npts, maxCam = jj;

		fprintf(fp, "3D track %d \n", ii);
		for (int kk = 0; kk < maxTracks; kk++)
		{
			for (int jj = 0; jj < nCams; jj++)
			{
				if (kk>CamTraj[jj].Track3DInfo[ii].npts || kk >= CamTraj[jj].R.size())
					continue;

				double Fi = CamTraj[jj].F + kk;
				if (Fi < 0.0)
					Fi = 0.0;
				if (Fi>CamTraj[jj].Track3DInfo[ii].npts - 2)
					Fi = CamTraj[jj].Track3DInfo[ii].npts - 2;
				int lFi = (int)Fi, uFi = lFi + 1, rFi = (int)(Fi + 0.5);
				double fFi = Fi - lFi;

				double K[9], C[3], Q[4], R[9], RT[12], P[12];
				for (int ll = 0; ll < 9; ll++)
					K[ll] = AllKMatrix[9 * jj*maxPts + 9 * lFi + ll];
				for (int ll = 0; ll < 3; ll++)
					C[ll] = (1.0 - fFi)*AllCamCenter[3 * jj*maxPts + 3 * lFi + ll] + fFi*AllCamCenter[3 * jj*maxPts + 3 * uFi + ll]; //linear interpolation

				for (int ll = 0; ll < 4; ll++)
					Q[ll] = AllQuaternion[4 * jj*maxPts + 4 * lFi + ll];
				//QuaternionLinearInterp(&AllQuaternion[4 * jj*maxPts + 4 * lFi], &AllQuaternion[4 * jj*maxPts + 4 * uFi], Q, fFi);//linear interpolation
				//Get_Value_Spline(ParaCamCenterX + jj*maxPts, maxPts, 1, Fi, 0, &C[0], -1, InterpAlgo);
				//Get_Value_Spline(ParaCamCenterY + jj*maxPts, maxPts, 1, Fi, 0, &C[1], -1, InterpAlgo);
				//Get_Value_Spline(ParaCamCenterZ + jj*maxPts, maxPts, 1, Fi, 0, &C[2], -1, InterpAlgo);

				Quaternion2Rotation(Q, R);
				AssembleRT(R, C, RT, true);
				AssembleP(K, RT, P);

				double XYZ[] = { CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk], CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk + 1], CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk + 2] };
				double denum = P[8] * XYZ[0] + P[9] * XYZ[1] + P[10] * XYZ[2] + P[11];
				double numX = P[0] * XYZ[0] + P[1] * XYZ[1] + P[2] * XYZ[2] + P[3];
				double numY = P[4] * XYZ[0] + P[5] * XYZ[1] + P[6] * XYZ[2] + P[7];

				double Sx, Sy;
				Get_Value_Spline(CamTraj[jj].Track2DInfo[ii].ParaX, CamTraj[jj].Track3DInfo[ii].npts, 1, Fi, 0, &Sx, -1, InterpAlgo);
				Get_Value_Spline(CamTraj[jj].Track2DInfo[ii].ParaY, CamTraj[jj].Track3DInfo[ii].npts, 1, Fi, 0, &Sy, -1, InterpAlgo);

				double residualsX = numX / denum - Sx;
				double residualsY = numY / denum - Sy;
				double Residual = residualsX*residualsX + residualsY*residualsY;
				Error += Residual;

				if (jj == 0)
					fprintf(fp, "%.4f %.4f %.4f ", CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk], CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk + 1], CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk + 2]);
			}
		}
		fprintf(fp, "\n");
	}
	printf("Final error: %.6e\n", Error);
	printf("Final offset: ");
	for (int ii = 0; ii < nCams; ii++)
		printf("%f ", CamTraj[ii].F);
	printf("\n");

	//printf("Write results ....\n");
	//WriteTrajectory(CamTraj, nCams, nTracks, 1.0);
	delete[]AllRotationMat, delete[]AllPMatrix, delete[]AllKMatrix, delete[]AllQuaternion, delete[]AllCamCenter;
	delete[]ParaCamCenterX, delete[]ParaCamCenterY, delete[]ParaCamCenterZ;

	return 0;
}
void Simulate3DHelix(vector<Point3d> &Traj, double A, double T, double V, int npts, double timeStep)
{
	double x, y, z;
	const double w = 2.0*Pi / T;
	Traj.reserve(npts);
	for (int ii = 0; ii < npts; ii++)
	{
		x = A*cos(w*ii*timeStep);
		y = A*sin(w*ii*timeStep);
		z = V*ii*timeStep;
		Traj.push_back(Point3d(x, y, z));
	}
	return;
}
double ComputeActionEnergy(vector<Point3d> traj, vector<double>Time, double eps, double g, int ApproxOrder)
{
	double Cost = 0.0;
	if (ApproxOrder == 1)
		for (int ii = 0; ii < traj.size() - 1; ii++)
		{
			double num = (pow(traj[ii].x - traj[ii + 1].x, 2) + pow(traj[ii].y - traj[ii + 1].y, 2) + pow(traj[ii].z - traj[ii + 1].z, 2));
			double denum = Time[ii + 1] - Time[ii] + eps;
			double cost = num / denum - g*(traj[ii].y + traj[ii + 1].y)*(Time[ii + 1] - Time[ii]);
			Cost += cost;
		}
	else
	{
		for (int ii = 0; ii < traj.size() - 2; ii++)
		{
			double  t1 = Time[ii], t2 = Time[ii + 1], t3 = Time[ii + 2];

			double num1 = (t2 - t3), num2 = (2.0*t2 - t1 - t3), num3 = (t2 - t1);
			double denum1 = ((t1 - t2)*(t1 - t3) + eps), denum2 = ((t2 - t1)*(t2 - t3) + eps), denum3 = ((t3 - t1)*(t3 - t2) + eps);
			double a1 = num1 / denum1, a2 = num2 / denum2, a3 = num3 / denum3;

			double velX = a1*traj[ii].x + a2*traj[ii + 1].x + a3*traj[ii + 2].x;
			double velY = a1*traj[ii].y + a2*traj[ii + 1].y + a3*traj[ii + 2].y;
			double velZ = a1*traj[ii].z + a2*traj[ii + 1].z + a3*traj[ii + 2].z;
			double vel = sqrt(velX*velX + velY*velY + velZ*velZ);
			double cost = (velX*velX + velY*velY + velZ*velZ)* abs(t3 - t1 + eps);
			Cost += cost;
		}
	}
	return Cost;
}
double ComputeActionEnergy(vector<XYZD> traj, vector<double>Time, double eps)
{
	double Cost = 0.0;
	for (int ii = 0; ii < traj.size() - 1; ii++)
	{
		double num = (pow(traj[ii].xyz.x - traj[ii + 1].xyz.x, 2) + pow(traj[ii].xyz.y - traj[ii + 1].xyz.y, 2) + pow(traj[ii].xyz.z - traj[ii + 1].xyz.z, 2));
		double denum = Time[ii + 1] - Time[ii] + eps;
		double cost = num / denum;
		Cost += cost;
	}
	return Cost;
}
void RecursiveUpdateCameraOffset(int *currentOffset, int BruteForceTimeWindow, int currentCam, int nCams)
{
	if (currentOffset[currentCam] > BruteForceTimeWindow)
	{
		currentOffset[currentCam] = -BruteForceTimeWindow;
		if (currentCam < nCams - 1)
			currentOffset[currentCam + 1] ++;
		RecursiveUpdateCameraOffset(currentOffset, BruteForceTimeWindow, currentCam + 1, nCams);
	}

	return;
}
int TestLeastActionConstraint(char *Path, int nCams, int ntracks, double stdev, int nRep = 10000)
{
	int approxOrder = 1;

	char Fname[200];
	VideoData AllVideoInfo;
	sprintf(Fname, "%s/Calib", Path);
	if (ReadVideoData(Fname, AllVideoInfo, nCams, 0, 3000) == 1)
		return 1;
	int nframes = max(MaxnFrame, 7000);
	double P[12];

	srand(time(NULL));
	const double Tscale = 1.0, ialpha = 1.0 / 120, eps = 1.0e-6, rate = 10;
	const int  BruteForceTimeWindow = 10, nCostSize = pow(2 * BruteForceTimeWindow + 1, (nCams - 1));

	vector<Point2d> uv;
	vector<int> ViewerList;
	vector<Point3d> XYZ, XYZBK, Traj;
	vector<double>Time;
	Point3d XYZAll[2000];

	int gtOff[] = { 0, 6, 2, 8, 8 };
	double *AllCost = new double[nCostSize];
	int *currentFrame = new int[nCams], *currentOffset = new int[nCams], *currentOffsetBK = new int[nCams], *SortID = new int[nCams], *PerCam_nf = new int[nCams];

	vector<Point2d> *PerCam_UV_All = new vector<Point2d>[nCams], *PerCam_UV = new vector<Point2d>[nCams];
	vector<Point3d> *PerCam_XYZ = new vector<Point3d>[nCams];

	//Make sure that no gtOff is smaller than 0
	for (int ii = 0; ii < nCams; ii++)
		if (rate*ii + gtOff[ii] < 0)
			gtOff[ii] += rate;

	for (int gI = 0; gI < 10; gI++)
	{
		for (int off = 0; off <= 10; off++)
		{
			gtOff[1] = off;
			for (int ii = 0; ii < nCostSize; ii++)
				AllCost[ii] = 0.0;
			for (int trackID = 0; trackID < ntracks; trackID++)
			{
				//Read 3D
				double x, y, z;
				XYZ.clear();
				sprintf(Fname, "%s/3DTracks/%d.txt", Path, trackID);
				FILE *fp = fopen(Fname, "r");
				while (fscanf(fp, "%lf %lf %lf", &x, &y, &z) != EOF)
					XYZ.push_back(Point3d(x, y, z));
				fclose(fp);
				//Simulate3DHelix(XYZ, A, T, V, 1000, ialpha);
				//double T = 0.8, w = 2.0*Pi / T;
				//for (int ii = 0; ii < 1000; ii++)
				//	XYZ.push_back(Point3d(0, 1.0*A*sin(w*ii*ialpha), 1.0*ii * 40 * ialpha));
				int nf = XYZ.size();

				//Generate 2D points
				ViewerList.clear();
				for (int jj = 0; jj < nCams; jj++)
				{
					int videoID = jj*nframes;
					PerCam_UV_All[jj].clear();
					for (int ii = 0; ii < nf; ii++)
					{
						for (int kk = 0; kk < 12; kk++)
							P[kk] = AllVideoInfo.VideoInfo[videoID + 1].P[kk];

						Point2d pts;
						ProjectandDistort(XYZ[ii], &pts, P, NULL, NULL, 1);
						PerCam_UV_All[jj].push_back(pts);
					}
				}

				//Assign 2D to cameras
				for (int ii = 0; ii < nCams; ii++)
				{
					PerCam_UV[ii].clear();
					for (int jj = 0; jj < nf; jj += rate)
					{
						int tid = jj + gtOff[ii];
						if (tid >= nf || tid < 0)
							continue;
						PerCam_UV[ii].push_back(PerCam_UV_All[ii][tid]);
					}
				}

				//Initiate internal data
				for (int kk = 1; kk < nCams; kk++)
					currentOffset[kk] = -BruteForceTimeWindow;
				for (int rep = 0; rep < nRep; rep++)
				{
					//Add noise to 3D data:
					XYZBK.clear();
					for (int ii = 0; ii < nf; ii++)
					{
						XYZBK.push_back(XYZ[ii]);
						XYZBK[ii].x += max(min(gaussian_noise(0.0, stdev), 3.0*stdev), -3.0*stdev);
						XYZBK[ii].y += max(min(gaussian_noise(0.0, stdev), 3.0*stdev), -3.0*stdev);
						XYZBK[ii].z += max(min(gaussian_noise(0.0, stdev), 3.0*stdev), -3.0*stdev);
					}

					//Assign 3D to cameras
					for (int ii = 0; ii < nCams; ii++)
					{
						PerCam_XYZ[ii].clear();
						for (int jj = 0; jj < nf; jj += rate)
						{
							int tid = jj + gtOff[ii];
							if (tid >= nf || tid < 0)
								continue;
							PerCam_XYZ[ii].push_back(XYZBK[tid]);
						}
					}
					for (int ii = 0; ii < nCams; ii++)
						PerCam_nf[ii] = PerCam_XYZ[ii].size();

					//Image projection cost:
					double ImCost = 0.0;
					for (int jj = 0; jj < nCams; jj++)
					{
						double *P = AllVideoInfo.VideoInfo[jj*nframes + 1].P;
						for (int ii = 0; ii < PerCam_nf[jj]; ii++)
						{
							double XYZ[3] = { PerCam_XYZ[jj][ii].x, PerCam_XYZ[jj][ii].y, PerCam_XYZ[jj][ii].z };
							double denum = P[8] * XYZ[0] + P[9] * XYZ[1] + P[10] * XYZ[2] + P[11];
							double numX = P[0] * XYZ[0] + P[1] * XYZ[1] + P[2] * XYZ[2] + P[3];
							double numY = P[4] * XYZ[0] + P[5] * XYZ[1] + P[6] * XYZ[2] + P[7];

							double residualsX = numX / denum - PerCam_UV[jj][ii].x;
							double residualsY = numY / denum - PerCam_UV[jj][ii].y;
							double Residual = sqrt(residualsX*residualsX + residualsY *residualsY);
							ImCost += Residual;
							if (Residual > 50)
								printf("%d %.2f %f\n", nCams, stdev, Residual);
						}
					}

					//Time alignment cost
					int count = 0;
					while (true)
					{
						Traj.clear(); Time.clear();
						//find the order of the streams
						currentOffset[0] = 0;
						for (int jj = 0; jj < nCams; jj++)
						{
							SortID[jj] = jj;
							currentOffsetBK[jj] = currentOffset[jj];
						}
						Quick_Sort_Int(currentOffsetBK, SortID, 0, nCams - 1);

						//Assemble trajactory and time from all Cameras
						for (int jj = 0; jj < nCams; jj++)
							currentFrame[jj] = 0;
						vector<int> availStream;
						availStream.push_back(SortID[0]), availStream.push_back(SortID[1]);

						double currentTime;
						vector<int>::iterator it;

						bool allOverlapped = false, doneAdding = false;
						while (!doneAdding)
						{
							int camID, NextCamID = availStream[availStream.size() - 1];
							bool reachedNewStream = false;

							//Add 3d and T in the streamlist to global stream until new stream is seen
							while (!reachedNewStream)
							{
								int nstream = availStream.size() - (allOverlapped ? 0 : 1);
								for (int ii = 0; ii < nstream; ii++)
								{
									camID = SortID[ii];
									if (currentFrame[camID] >= PerCam_nf[camID])
										continue;

									currentTime = 1.0*currentOffset[camID] * ialpha*Tscale + 1.0*currentFrame[camID] * ialpha*Tscale*rate;
									if (!allOverlapped && currentTime >= 1.0*currentOffset[NextCamID] * ialpha*Tscale)
									{
										reachedNewStream = true;
										if (nCams > nstream + 1)
											availStream.push_back(SortID[nstream + 1]);
										else
											allOverlapped = true;

										//re-sort the order 
										nstream = availStream.size() - (allOverlapped ? 0 : 1);
										for (int jj = 0; jj < nstream; jj++)
											currentOffsetBK[jj] = currentOffset[SortID[jj]] + currentFrame[SortID[jj]] * rate;
										Quick_Sort_Int(currentOffsetBK, SortID, 0, nstream - 1);
										break;
									}

									Time.push_back(currentTime);
									Traj.push_back(PerCam_XYZ[camID][currentFrame[camID]]);

									currentFrame[camID]++;
								}

								int nStreamEnd = 0;
								for (int ii = 0; ii < nCams; ii++)
									if (currentFrame[ii] >= PerCam_nf[ii])
										nStreamEnd++;
								if (nStreamEnd == nCams)
								{
									doneAdding = true;
									break;
								}
							}
						}

						//Compute energy
						AllCost[count] += ComputeActionEnergy(Traj, Time, eps, 9.8 * gI, approxOrder) + ImCost;
						//double cost2 = ComputeActionEnergy(Traj, Time, eps, 1);

						currentOffset[1]++;
						RecursiveUpdateCameraOffset(currentOffset, BruteForceTimeWindow, 1, nCams);

						count++;
						if (count == nCostSize)
							break;
					}
				}
			}

			//Parse the Cost into offset and print
			int count = 0;
			for (int kk = 1; kk < nCams; kk++)
				currentOffset[kk] = -BruteForceTimeWindow;

			sprintf(Fname, "C:/temp/%d_%d_cost_%d_%.1f.txt", gI, gtOff[1], nCams, stdev);  FILE *fp = fopen(Fname, "w+");
			while (true)
			{
				for (int kk = 1; kk < nCams; kk++)
					fprintf(fp, "%d ", currentOffset[kk]);
				fprintf(fp, "%.16f\n", AllCost[count] / nRep);

				currentOffset[1]++;
				RecursiveUpdateCameraOffset(currentOffset, BruteForceTimeWindow, 1, nCams);

				count++;
				if (count == nCostSize)
					break;
			}
			fclose(fp);
		}
	}

	printf("Done with %s\n", Path);
	delete[]currentFrame, delete[]currentOffset, delete[]PerCam_nf, delete[]SortID, delete[]currentOffsetBK, delete[]AllCost;
	delete[]PerCam_UV, delete[]PerCam_UV_All, delete[]PerCam_XYZ;

	return 0;
}
void RotY(double *Rmat, double theta)
{
	double c = cos(theta), s = sin(theta);
	Rmat[0] = c, Rmat[1] = 0.0, Rmat[2] = s;
	Rmat[3] = 0, Rmat[4] = 1.0, Rmat[5] = 0;
	Rmat[6] = -s, Rmat[7] = 0.0, Rmat[8] = c;
	return;
}
void GenerateCamerasExtrinsicOnCircle(CameraData &CameraInfo, double theta, double radius, Point3d center, Point3d LookAtTarget, Point3d Noise3D)
{
	//Adapted from Jack's code
	double Rmat[9], RmatT[9];
	RotY(Rmat, theta); mat_transpose(Rmat, RmatT, 3, 3);
	double CameraPosition[3] = { -(RmatT[0] * 0.0 + RmatT[1] * 0.0 + RmatT[2] * radius) + center.x + Noise3D.x,
		-(RmatT[3] * 0.0 + RmatT[4] * 0.0 + RmatT[5] * radius) + center.y + Noise3D.y,
		-(RmatT[6] * 0.0 + RmatT[7] * 0.0 + RmatT[8] * radius) + center.z + Noise3D.z };

	//Look at vector
	double k[3] = { LookAtTarget.x - CameraPosition[0], LookAtTarget.y - CameraPosition[1], LookAtTarget.z - CameraPosition[2] };
	normalize(k, 3);

	//Up vector
	double j[3] = { 0.0, 1.0, 0.0 };

	//Sideway vector
	double i[3]; cross_product(j, k, i);

	//Camera rotation matrix
	CameraInfo.R[0] = i[0], CameraInfo.R[1] = i[1], CameraInfo.R[2] = i[2];
	CameraInfo.R[3] = j[0], CameraInfo.R[4] = j[1], CameraInfo.R[5] = j[2];
	CameraInfo.R[6] = k[0], CameraInfo.R[7] = k[1], CameraInfo.R[8] = k[2];

	//Translation vector
	mat_mul(CameraInfo.R, CameraPosition, CameraInfo.T, 3, 3, 1);
	CameraInfo.T[0] = -CameraInfo.T[0], CameraInfo.T[1] = -CameraInfo.T[1], CameraInfo.T[2] = -CameraInfo.T[2];
	CameraInfo.camCenter[0] = CameraPosition[0], CameraInfo.camCenter[1] = CameraPosition[1], CameraInfo.camCenter[2] = CameraPosition[2];
	return;
}
void SimulateCamerasAnd2DPointsForMoCap(char *Path, int nCams, int n3DTracks, double *Intrinsic, int width, int height, double radius = 5e3, bool show2DImage = false, bool save2D = false)
{
	char Fname[200];
	double noise3DMag = 20 / 60;
	double x, y, z, cx = 0, cy = 0, cz = 0;
	vector<Point3d> XYZ;
	sprintf(Fname, "%s/3DTracks/%d.txt", Path, 0); FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
		printf("Cannot load %s\n", Fname);
	while (fscanf(fp, "%lf %lf %lf", &x, &y, &z) != EOF)
	{
		XYZ.push_back(Point3d(x, y, z));
		cx += x, cy += y, cz += z;
	}
	fclose(fp);
	int nframes = XYZ.size();
	cx /= nframes, cy /= nframes, cz /= nframes;

	CameraData *Camera = new CameraData[nframes*nCams];
	vector<int> angleList;
	vector<Point3d> Center;
	for (int frameID = 0; frameID < nframes; frameID++)
	{
		angleList.clear(), Center.clear();
		for (int camID = 0; camID < nCams; camID++)
		{
			int count, angleID;
			while (true)
			{
				count = 0, angleID = 5.0*cos(2.0*Pi / 100 * frameID) + 360 * camID / nCams;
				for (int ii = 0; ii < angleList.size(); ii++)
					if (angleID == angleList[ii])
						count++;
				if (count == 0)
					break;
			}
			angleList.push_back(angleID);

			double theta = 1.0*angleID / 180 * Pi;
			Point3d Noise3D(gaussian_noise(0.0, noise3DMag), gaussian_noise(0.0, noise3DMag), gaussian_noise(0.0, noise3DMag));
			if (Noise3D.x > 3.0*noise3DMag)
				Noise3D.x = 3.0*noise3DMag;
			else if (Noise3D.x < -3.0 *noise3DMag)
				Noise3D.x = -3.0*noise3DMag;
			if (Noise3D.y > 3.0*noise3DMag)
				Noise3D.y = 3.0*noise3DMag;
			else if (Noise3D.y < -3.0 *noise3DMag)
				Noise3D.y = -3.0*noise3DMag;
			if (Noise3D.z > 3.0*noise3DMag)
				Noise3D.z = 3.0*noise3DMag;
			else if (Noise3D.z < -3.0 *noise3DMag)
				Noise3D.z = -3.0*noise3DMag;

			//GenerateCamerasExtrinsicOnCircle(Camera[frameID + nframes*camID], theta, radius, XYZ[frameID], Point3d(cx, cy, cz), Noise3D);
			GenerateCamerasExtrinsicOnCircle(Camera[frameID + nframes*camID], theta, radius, XYZ[frameID], XYZ[frameID], Noise3D);
			SetIntrinisc(Camera[frameID + nframes*camID], Intrinsic);
			AssembleP(Camera[frameID + nframes*camID]);
			Center.push_back(Point3d(Camera[frameID + nframes*camID].camCenter[0], Camera[frameID + nframes*camID].camCenter[1], Camera[frameID + nframes*camID].camCenter[2]));
		}
		angleList;
		Center;
		if (frameID == 19)
			int a = 0;
	}

	if (show2DImage)
	{
		Mat Img(height, width, CV_8UC3, Scalar(0, 0, 0)), displayImg;
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
		cvNamedWindow("Image", CV_WINDOW_NORMAL);

		for (int camID = 0; camID < nCams; camID++)
		{
			for (int frameID = 0; frameID < nframes; frameID++)
			{
				XYZ.clear();
				sprintf(Fname, "%s/3DPoints/3dGL_%d.txt", Path, frameID); fp = fopen(Fname, "r");
				if (fp == NULL)
				{
					printf("Cannot load %s\n", Fname);
					continue;
				}
				while (fscanf(fp, "%lf %lf %lf ", &x, &y, &z) != EOF)
					XYZ.push_back(Point3d(x, y, z));
				fclose(fp);

				Point2d *ImgPts = new Point2d[XYZ.size()];
				ProjectandDistort(XYZ, ImgPts, Camera[frameID + nframes*camID].P);

				displayImg = Img.clone();
				for (int pid = 0; pid < XYZ.size(); pid++)
					circle(displayImg, ImgPts[pid], 4, colors[pid % 9], 1, 8, 0);

				sprintf(Fname, "Cam %d: frame d", camID, frameID);
				CvPoint text_origin = { width / 30, height / 30 };
				putText(displayImg, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 3.0 * 640 / Img.cols, CV_RGB(255, 0, 0), 2);
				imshow("Image", displayImg);
				waitKey(1);

				delete[]ImgPts;
			}
		}
	}

	for (int camID = 0; camID < nCams; camID++)
	{
		sprintf(Fname, "%s/Calib/intrinsic_%d.txt", Path, camID); fp = fopen(Fname, "w+");
		for (int frameID = 0; frameID < nframes; frameID++)
		{
			fprintf(fp, "%d 0 ", frameID);
			for (int ii = 0; ii < 5; ii++)
				fprintf(fp, "%.16f ", Camera[frameID + nframes*camID].intrinsic[ii]);
			for (int ii = 0; ii < 7; ii++)
				fprintf(fp, "%.1f ", 0.0);
			fprintf(fp, "%d %d\n", width, height);
		}
		fclose(fp);
	}

	vector<int>AvailFrame;
	for (int frameID = 0; frameID < nframes; frameID++)
		AvailFrame.push_back(frameID);
	for (int camID = 0; camID < nCams; camID++)
	{
		sprintf(Fname, "%s/Calib/PinfoGL_%d.txt", Path, camID); fp = fopen(Fname, "w+");  fclose(fp);
		sprintf(Fname, "%s/Calib", Path);
		SaveVideoCameraPosesGL(Fname, Camera + camID*nframes, AvailFrame, camID, 0);
	}

	if (save2D)
	{
		for (int trackID = 0; trackID < n3DTracks; trackID++)
		{
			XYZ.clear();
			sprintf(Fname, "%s/3DTracks/%d.txt", Path, trackID); fp = fopen(Fname, "r");
			if (fp == NULL)
				printf("Cannot load %s\n", Fname);
			while (fscanf(fp, "%lf %lf %lf", &x, &y, &z) != EOF)
				XYZ.push_back(Point3d(x, y, z));
			fclose(fp);

			int nframes = XYZ.size();
			Point2d *ImPts = new Point2d[nframes];
			Point3d wc;
			Point2d pt;
			for (int camID = 0; camID < nCams; camID++)
			{
				for (int frameID = 0; frameID < nframes; frameID++)
				{
					wc = XYZ[frameID];
					ProjectandDistort(XYZ[frameID], &pt, Camera[frameID + camID*nframes].P);
					ImPts[frameID] = pt;
				}

				sprintf(Fname, "%s/2DTracks/%d_%d.txt", Path, camID, trackID); fp = fopen(Fname, "w+");
				for (int frameID = 0; frameID < nframes; frameID++)
					fprintf(fp, "%.16f %.16f\n", ImPts[frameID].x, ImPts[frameID].y);
				fclose(fp);
			}
			delete[]ImPts;
		}
	}

	return;
}

double LeastActionError(double *xyz1, double *xyz2, double *timeStamp1, double *timeStamp2, int frameID1, int frameID2, double ialpha, double Tscale, double timeStep, double eps)
{
	double difX = xyz2[0] - xyz1[0], difY = xyz2[1] - xyz1[1], difZ = xyz2[2] - xyz1[2];
	double  t1 = (timeStamp1[0] + timeStep*frameID1) * ialpha*Tscale;
	double  t2 = (timeStamp2[0] + timeStep*frameID2) * ialpha*Tscale;

	double vel = sqrt(difX*difX + difY*difY + difZ*difZ) / (t2 - t1 + eps);
	double cost = pow(difX*difX + difY*difY + difZ*difZ, 2) / (pow(t2 - t1, 3) + eps);

	return cost;
}
struct LeastActionCostCeres {
	LeastActionCostCeres(int fID1, int fID2, double ial, double tScale, double tStep, double eps)
	{
		ialpha = ial, Tscale = tScale, timeStep = tStep;
		epsilon = eps;
		frameID1 = fID1, frameID2 = fID2;
	}

	template <typename T>	bool operator()(const T* const xyz1, const T* const xyz2, const T* const timeStamp1, const T* const timeStamp2, T* residuals) 	const
	{
		T difX = xyz2[0] - xyz1[0], difY = xyz2[1] - xyz1[1], difZ = xyz2[2] - xyz1[2];
		T  t1 = (T)((timeStamp1[0] + timeStep*frameID1) * ialpha*Tscale);
		T  t2 = (T)((timeStamp2[0] + timeStep*frameID2) * ialpha*Tscale);

		T cost = pow(difX*difX + difY*difY + difZ*difZ, 2) / abs(pow(t2 - t1, 3) + (T)epsilon);
		//T cost = (difX*difX + difY*difY + difZ*difZ) / abs(t2 - t1 + (T)epsilon);
		residuals[0] = sqrt(cost);
		return true;
	}
	template <typename T>	bool operator()(const T* const xyz1, const T* const xyz2, const T* const timeStamp, T* residuals) 	const
	{
		T difX = xyz2[0] - xyz1[0], difY = xyz2[1] - xyz1[1], difZ = xyz2[2] - xyz1[2];
		T  t1 = (T)((timeStamp[0] + timeStep*frameID1) * ialpha*Tscale);
		T  t2 = (T)((timeStamp[0] + timeStep*frameID2) * ialpha*Tscale);

		T cost = pow(difX*difX + difY*difY + difZ*difZ, 2) / abs(pow(t2 - t1, 3) + (T)epsilon);
		//T cost = (difX*difX + difY*difY + difZ*difZ) / abs(t2 - t1 + (T)epsilon);
		residuals[0] = sqrt(cost);

		return true;
	}

	static ceres::CostFunction* CreateAutoDiff(int frameID1, int frameID2, double ialpha, double Tscale, double timeStep, double epsilon)
	{
		return (new ceres::AutoDiffCostFunction<LeastActionCostCeres, 1, 3, 3, 1, 1>(new LeastActionCostCeres(frameID1, frameID2, ialpha, Tscale, timeStep, epsilon)));
	}
	static ceres::CostFunction* CreateAutoDiffSame(int frameID1, int frameID2, double ialpha, double Tscale, double timeStep, double epsilon)
	{
		return (new ceres::AutoDiffCostFunction<LeastActionCostCeres, 1, 3, 3, 1>(new LeastActionCostCeres(frameID1, frameID2, ialpha, Tscale, timeStep, epsilon)));
	}

	static ceres::CostFunction* CreateNumerDiff(int frameID1, int frameID2, double ialpha, double Tscale, double timeStep, double epsilon)
	{
		return (new ceres::NumericDiffCostFunction<LeastActionCostCeres, ceres::CENTRAL, 1, 3, 3, 1, 1>(new LeastActionCostCeres(frameID1, frameID2, ialpha, Tscale, timeStep, epsilon)));
	}

	int frameID1, frameID2;
	double ialpha, Tscale, timeStep, epsilon;
};
struct TimeCyclicCostCerest {
	TimeCyclicCostCerest(double ilamda){
		lamda = ilamda;
	}

	template <typename T>	bool operator()(const T* const timeStamp1, const T* const timeStamp2, const T* const timeStamp3, T* residuals) 	const
	{
		T dif12 = timeStamp1[0] - timeStamp2[0];
		T dif23 = timeStamp2[0] - timeStamp3[0];
		T dif13 = timeStamp1[0] - timeStamp3[0];
		residuals[0] = (T)lamda*(dif12 + dif23 - dif13);
		return true;
	}
	static ceres::CostFunction* CreateAutoDiff(double lamda){
		return (new ceres::AutoDiffCostFunction<TimeCyclicCostCerest, 1, 1, 1, 1>(new TimeCyclicCostCerest(lamda)));
	}

	static ceres::CostFunction* CreateNumerDiff(double lamda){
		return (new ceres::NumericDiffCostFunction<TimeCyclicCostCerest, ceres::CENTRAL, 1, 1, 1, 1>(new TimeCyclicCostCerest(lamda)));
	}

	double lamda;
};
struct IdealGeoProjectionCeres {
	IdealGeoProjectionCeres(double *Pmat, Point2d pt2D, double ilamda)
	{
		P = Pmat, observed_x = pt2D.x, observed_y = pt2D.y;
		lamda = ilamda;
	}

	template <typename T>	bool operator()(const T* const pt3D, T* residuals) 	const
	{
		T numX = (T)P[0] * pt3D[0] + (T)P[1] * pt3D[1] + (T)P[2] * pt3D[2] + (T)P[3];
		T numY = (T)P[4] * pt3D[0] + (T)P[5] * pt3D[1] + (T)P[6] * pt3D[2] + (T)P[7];
		T denum = (T)P[8] * pt3D[0] + (T)P[9] * pt3D[1] + (T)P[10] * pt3D[2] + (T)P[11];

		residuals[0] = (T)(lamda)*(numX / denum - T(observed_x));
		residuals[1] = (T)(lamda)*(numY / denum - T(observed_y));

		return true;
	}

	static ceres::CostFunction* Create(double *Pmat, const Point2d pt2D, double lamda)
	{
		return (new ceres::AutoDiffCostFunction<IdealGeoProjectionCeres, 2, 3>(new IdealGeoProjectionCeres(Pmat, pt2D, lamda)));
	}

	double observed_x, observed_y, *P, lamda;
};
struct IdealAlgebraicReprojectionCeres {
	IdealAlgebraicReprojectionCeres(double *iQ, double *iU, double ilamda)
	{
		Q = iQ, U = iU;
		lamda = ilamda;
	}

	template <typename T>	bool operator()(const T* const pt3D, T* residuals) 	const
	{
		residuals[0] = (T)(lamda)*(Q[0] * pt3D[0] + Q[1] * pt3D[1] + Q[2] * pt3D[2] - U[0]);
		residuals[1] = (T)(lamda)*(Q[3] * pt3D[0] + Q[4] * pt3D[1] + Q[5] * pt3D[2] - U[1]);

		return true;
	}

	static ceres::CostFunction* Create(double *Qmat, double* Umat, double lamda)
	{
		return (new ceres::AutoDiffCostFunction<IdealAlgebraicReprojectionCeres, 2, 3>(new IdealAlgebraicReprojectionCeres(Qmat, Umat, lamda)));
	}

	double *Q, *U, lamda;
};

void LeastActionOptimXYZT(char *Path, vector<double*> &Allpt3D, vector<ImgPtEle> *PerCam_UV, vector<int> &PointsPerTrack, double *currentOffset, int ntracks, bool fixedTime, bool fixed3D, bool non_monotonicDescent, int nCams, double Tscale, double ialpha, double rate, double eps, double lamda2DCost, double lamdaTimeTripletCost)
{
	double ActionCost = 0.0, ProjCost = 0.0, costiX, costiY, costi;

	int *currentFrame = new int[nCams], *PerCam_nf = new int[nCams], *SortID = new int[nCams];
	double *currentOffsetBK = new double[nCams];
	Point3d P3D;
	ImgPtEle ptEle;

	vector<int>triangulatedList;
	vector<double>AllError3D, VectorTime;
	vector<int> *VectorCamID = new vector<int>[ntracks], *VectorFrameID = new vector<int>[ntracks];
	vector<ImgPtEle> *Traj2DAll = new vector<ImgPtEle>[ntracks];

	ceres::Problem problem;
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		for (int camID = 0; camID < nCams; camID++)
			PerCam_nf[camID] = PerCam_UV[camID*ntracks + trackID].size();

		//Time alignment:find the order of the streams
		VectorTime.clear();
		currentOffset[0] = 0;
		for (int jj = 0; jj < nCams; jj++)
			SortID[jj] = jj, currentOffsetBK[jj] = currentOffset[jj];
		Quick_Sort_Double(currentOffsetBK, SortID, 0, nCams - 1);

		//Assemble trajactory and time from all Cameras
		for (int jj = 0; jj < nCams; jj++)
			currentFrame[jj] = 0;
		vector<int> availStream;
		availStream.push_back(SortID[0]), availStream.push_back(SortID[1]);

		double currentTime;
		vector<int>::iterator it;
		bool allOverlapped = false, doneAdding = false;
		while (!doneAdding)
		{
			int camID, NextCamID = availStream[availStream.size() - 1];
			bool reachedNewStream = false;

			//Add 3D and T in the streamlist to global stream until new stream is seen
			while (!reachedNewStream)
			{
				int nstream = availStream.size() - (allOverlapped ? 0 : 1);
				for (int ii = 0; ii < nstream; ii++)
				{
					camID = SortID[ii];
					if (currentFrame[camID] >= PerCam_nf[camID])
						continue;

					currentTime = 1.0*currentOffset[camID] * ialpha*Tscale + 1.0*currentFrame[camID] * ialpha*Tscale*rate;
					if (!allOverlapped && currentTime >= 1.0*currentOffset[NextCamID] * ialpha*Tscale)
					{
						reachedNewStream = true;
						if (nCams > nstream + 1)
							availStream.push_back(SortID[nstream + 1]);
						else
							allOverlapped = true;

						//re-sort the order 
						nstream = availStream.size() - (allOverlapped ? 0 : 1);
						for (int jj = 0; jj < nstream; jj++)
							currentOffsetBK[jj] = currentOffset[SortID[jj]] + currentFrame[SortID[jj]] * rate;
						Quick_Sort_Double(currentOffsetBK, SortID, 0, nstream - 1);
						break;
					}

					VectorTime.push_back(currentTime);
					VectorFrameID[trackID].push_back(currentFrame[camID]);
					VectorCamID[trackID].push_back(camID);
					Traj2DAll[trackID].push_back(PerCam_UV[camID*ntracks + trackID][currentFrame[camID]]);
					currentFrame[camID]++;
				}

				int nStreamEnd = 0;
				for (int ii = 0; ii < nCams; ii++)
					if (currentFrame[ii] >= PerCam_nf[ii])
						nStreamEnd++;
				if (nStreamEnd == nCams)
				{
					doneAdding = true;
					break;
				}
			}
		}

		int npts = Traj2DAll[trackID].size();
		Allpt3D[trackID] = new double[3 * npts];
		for (int ll = 0; ll < npts; ll++)
			Allpt3D[trackID][3 * ll] = Traj2DAll[trackID][ll].pt3D.x, Allpt3D[trackID][3 * ll + 1] = Traj2DAll[trackID][ll].pt3D.y, Allpt3D[trackID][3 * ll + 2] = Traj2DAll[trackID][ll].pt3D.z;

		//1st order approx of v
		double *Q1, *Q2, *U1, *U2;
		for (int ll = 0; ll < npts - 1; ll++)
		{
			int camID1 = VectorCamID[trackID][ll], camID2 = VectorCamID[trackID][ll + 1];
			int frameID1 = VectorFrameID[trackID][ll], frameID2 = VectorFrameID[trackID][ll + 1];

			costi = LeastActionError(&Allpt3D[trackID][3 * ll], &Allpt3D[trackID][3 * ll + 3], &currentOffset[camID1], &currentOffset[camID2], VectorFrameID[trackID][ll], VectorFrameID[trackID][ll + 1], ialpha, Tscale, rate, eps);
			ActionCost += costi;
			Q1 = Traj2DAll[trackID][ll].Q, Q2 = Traj2DAll[trackID][ll + 1].Q, U1 = Traj2DAll[trackID][ll].u, U2 = Traj2DAll[trackID][ll + 1].u;

			costiX = lamda2DCost*(Q1[0] * Allpt3D[trackID][3 * ll] + Q1[1] * Allpt3D[trackID][3 * ll + 1] + Q1[2] * Allpt3D[trackID][3 * ll + 2] - U1[0]);
			costiY = lamda2DCost*(Q1[3] * Allpt3D[trackID][3 * ll] + Q1[4] * Allpt3D[trackID][3 * ll + 1] + Q1[5] * Allpt3D[trackID][3 * ll + 2] - U1[1]);
			costi = sqrt(costiX*costiX + costiY*costiY);
			ProjCost += costi;

			costiX = lamda2DCost*(Q2[0] * Allpt3D[trackID][3 * ll + 3] + Q2[1] * Allpt3D[trackID][3 * ll + 4] + Q2[2] * Allpt3D[trackID][3 * ll + 5] - U2[0]);
			costiY = lamda2DCost*(Q2[3] * Allpt3D[trackID][3 * ll + 3] + Q2[4] * Allpt3D[trackID][3 * ll + 4] + Q2[5] * Allpt3D[trackID][3 * ll + 5] - U2[1]);
			costi = sqrt(costiX*costiX + costiY*costiY);
			ProjCost += costi;

			if (camID1 == camID2)
			{
				ceres::CostFunction* cost_function = LeastActionCostCeres::CreateAutoDiffSame(VectorFrameID[trackID][ll], VectorFrameID[trackID][ll + 1], ialpha, Tscale, rate, eps);
				problem.AddResidualBlock(cost_function, NULL, &Allpt3D[trackID][3 * ll], &Allpt3D[trackID][3 * (ll + 1)], &currentOffset[camID1]);
			}
			else
			{
				ceres::CostFunction* cost_function = LeastActionCostCeres::CreateAutoDiff(VectorFrameID[trackID][ll], VectorFrameID[trackID][ll + 1], ialpha, Tscale, rate, eps);
				problem.AddResidualBlock(cost_function, NULL, &Allpt3D[trackID][3 * ll], &Allpt3D[trackID][3 * (ll + 1)], &currentOffset[camID1], &currentOffset[camID2]);
			}

			ceres::CostFunction* cost_function2 = IdealAlgebraicReprojectionCeres::Create(Traj2DAll[trackID][ll].Q, Traj2DAll[trackID][ll].u, lamda2DCost);
			problem.AddResidualBlock(cost_function2, NULL, Allpt3D[trackID] + 3 * ll);

			ceres::CostFunction* cost_function3 = IdealAlgebraicReprojectionCeres::Create(Traj2DAll[trackID][ll + 1].Q, Traj2DAll[trackID][ll + 1].u, lamda2DCost);
			problem.AddResidualBlock(cost_function3, NULL, Allpt3D[trackID] + 3 * ll + 3);
		}
	}
	//Add triplet cyclic time constraint: AB + BC = AC
	if (nCams > 2)
	{
		const int k = 3; //nchoosek
		std::vector<bool> v(nCams);
		std::fill(v.begin() + k, v.end(), true);

		int nTriplets = nChoosek(nCams, k);
		vector<int>*TripletList = new vector<int>[nTriplets];
		int count = 0;
		do {
			for (int i = 0; i < nCams; ++i)
				if (!v[i])
					TripletList[count].push_back(i);
			count++;
		} while (std::next_permutation(v.begin(), v.end()));

		for (int ii = 0; ii < nTriplets; ii++)
		{
			ceres::CostFunction* cost_function = TimeCyclicCostCerest::CreateAutoDiff(lamdaTimeTripletCost);
			problem.AddResidualBlock(cost_function, NULL, &currentOffset[TripletList[ii][0]], &currentOffset[TripletList[ii][1]], &currentOffset[TripletList[ii][2]]);
		}
	}
	//Set bound on the time 
	for (int camID = 1; camID < nCams; camID++)
		problem.SetParameterLowerBound(&currentOffset[camID], 0, 0.0), problem.SetParameterUpperBound(&currentOffset[camID], 0, rate);

	//Set fixed parameters
	problem.SetParameterBlockConstant(&currentOffset[0]);
	if (fixedTime)
		for (int camID = 1; camID < nCams; camID++)
			problem.SetParameterBlockConstant(&currentOffset[camID]);

	for (int trackID = 0; trackID < ntracks; trackID++)
		PointsPerTrack.push_back(Traj2DAll[trackID].size());
	if (fixed3D)
	{
		for (int trackID = 0; trackID < ntracks; trackID++)
		{
			int npts = PointsPerTrack[trackID];
			for (int ll = 0; ll < npts; ll++)
				problem.SetParameterBlockConstant(&Allpt3D[trackID][3 * ll]);
		}
	}

	//printf("ActionCost and ProjCost before: %f %f\n", ActionCost, ProjCost);
	printf("%f %f ", ActionCost, ProjCost);

	ceres::Solver::Options options;
	options.num_threads = 4;
	options.max_num_iterations = 150;
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	options.minimizer_progress_to_stdout = true;
	options.trust_region_strategy_type = ceres::DOGLEG;
	options.use_nonmonotonic_steps = non_monotonicDescent;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";

	ActionCost = 0.0, ProjCost = 0.0;
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		int npts = PointsPerTrack[trackID];
		double costi;
		for (int ll = 0; ll < npts - 1; ll++)
		{
			int camID1 = VectorCamID[trackID][ll], camID2 = VectorCamID[trackID][ll + 1];

			costi = LeastActionError(&Allpt3D[trackID][3 * ll], &Allpt3D[trackID][3 * ll + 3], &currentOffset[camID1], &currentOffset[camID2], VectorFrameID[trackID][ll], VectorFrameID[trackID][ll + 1], ialpha, Tscale, rate, eps);
			ActionCost += costi;

			costiX = lamda2DCost*(Traj2DAll[trackID][ll].Q[0] * Allpt3D[trackID][3 * ll] + Traj2DAll[trackID][ll].Q[1] * Allpt3D[trackID][3 * ll + 1] + Traj2DAll[trackID][ll].Q[2] * Allpt3D[trackID][3 * ll + 2] - Traj2DAll[trackID][ll].u[0]);
			costiY = lamda2DCost*(Traj2DAll[trackID][ll].Q[3] * Allpt3D[trackID][3 * ll] + Traj2DAll[trackID][ll].Q[4] * Allpt3D[trackID][3 * ll + 1] + Traj2DAll[trackID][ll].Q[5] * Allpt3D[trackID][3 * ll + 2] - Traj2DAll[trackID][ll].u[1]);
			costi = sqrt(costiX*costiX + costiY*costiY);
			ProjCost += costi;

			costiX = lamda2DCost*(Traj2DAll[trackID][ll + 1].Q[0] * Allpt3D[trackID][3 * ll + 3] + Traj2DAll[trackID][ll + 1].Q[1] * Allpt3D[trackID][3 * ll + 4] + Traj2DAll[trackID][ll + 1].Q[2] * Allpt3D[trackID][3 * ll + 5] - Traj2DAll[trackID][ll + 1].u[0]);
			costiY = lamda2DCost*(Traj2DAll[trackID][ll + 1].Q[3] * Allpt3D[trackID][3 * ll + 3] + Traj2DAll[trackID][ll + 1].Q[4] * Allpt3D[trackID][3 * ll + 4] + Traj2DAll[trackID][ll + 1].Q[5] * Allpt3D[trackID][3 * ll + 5] - Traj2DAll[trackID][ll + 1].u[1]);
			costi = sqrt(costiX*costiX + costiY*costiY);
			ProjCost += costi;
		}
	}

	//printf("ActionCost and ProjCost after: %f %f\n", ActionCost, ProjCost);
	printf("%f %f ", ActionCost, ProjCost);
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		for (int ll = 0; ll < PointsPerTrack[trackID]; ll++)
		{
			int camID = VectorCamID[trackID][ll], frameID = VectorFrameID[trackID][ll];
			PerCam_UV[camID*ntracks + trackID][frameID].pt3D = Point3d(Allpt3D[trackID][3 * ll], Allpt3D[trackID][3 * ll + 1], Allpt3D[trackID][3 * ll + 2]);
		}
	}

	double distanceCost = 0.0;
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		int npts = Traj2DAll[trackID].size();
		double costi;
		for (int ll = 0; ll < npts - 1; ll++)
		{
			costi = sqrt(pow(Allpt3D[trackID][3 * ll] - Allpt3D[trackID][3 * ll + 3], 2) + pow(Allpt3D[trackID][3 * ll + 1] - Allpt3D[trackID][3 * ll + 4], 2) + pow(Allpt3D[trackID][3 * ll + 2] - Allpt3D[trackID][3 * ll + 5], 2));
			distanceCost += costi;
		}
	}
	//printf("Distance Cost: %e\n", distanceCost);
	printf("%e ", distanceCost);

	double directionCost = 0.0;
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		int npts = Traj2DAll[trackID].size();
		double direct1[3], direct2[3];
		for (int ll = 0; ll < npts - 2; ll++)
		{
			direct1[0] = Allpt3D[trackID][3 * ll] - Allpt3D[trackID][3 * ll + 3], direct1[1] = Allpt3D[trackID][3 * ll + 1] - Allpt3D[trackID][3 * ll + 4], direct1[2] = Allpt3D[trackID][3 * ll + 2] - Allpt3D[trackID][3 * ll + 5];
			direct2[0] = Allpt3D[trackID][3 * ll + 3] - Allpt3D[trackID][3 * ll + 6], direct2[1] = Allpt3D[trackID][3 * ll + 4] - Allpt3D[trackID][3 * ll + 7], direct2[2] = Allpt3D[trackID][3 * ll + 5] - Allpt3D[trackID][3 * ll + 8];
			normalize(direct1), normalize(direct2);
			directionCost += abs(dotProduct(direct1, direct2));
		}
	}
	//printf("Direction Cost: %e\n", directionCost);
	printf("%e\n", directionCost);

	delete[]VectorCamID, delete[]VectorFrameID, delete[]Traj2DAll;
	delete[]currentFrame, delete[]PerCam_nf, delete[]SortID, delete[]currentOffsetBK;

	return;
}
int SimultaneousSyncReconc(char *Path, int nCams, int ntracks, double stdev)
{
	srand(time(NULL));
	bool save3D = true, fixed3D = true, fixedTime = false, non_monotonicDescent = true;
	int gtOff[] = { 0, 1, 4, 8, 9 };
	double currentOffset[5] = { 0, 0.1, .2, .3, .4 };

	const double Tscale = 1000.0, ialpha = 1.0 / 120, rate = 10, eps = 1.0e-1, lamda2DCost = 10.0, lamdaTimeTripletCost = 10000.0;

	char Fname[200];
	VideoData AllVideoInfo;
	sprintf(Fname, "%s/Calib", Path);
	if (ReadVideoData(Fname, AllVideoInfo, nCams, 0, 7000) == 1)
		return 1;
	int nframes = max(MaxnFrame, 7000);
	double ActionCost = 0.0, TProjCost = 0.0, costi, x, y, z;
	double P[12], AA[6], bb[2], ccT[3], dd[1];
	double *Q = new double[6 * nCams], *U = new double[2 * nCams], *AllP = new double[12 * nCams], *A = new double[6 * nCams], *B = new double[2 * nCams];

	Point2d *ImgPts = new Point2d[nCams];
	Point3d P3D;
	ImgPtEle ptEle;
	vector<int>triangulatedList;

	vector<int>VectorCamID, VectorFrameID;
	vector<double> AllError2D;
	vector<ImgPtEle> *PerCamUV_All = new vector<ImgPtEle>[nCams], *PerCam_UV = new vector<ImgPtEle>[nCams*ntracks];
	vector<XYZD> *PerCam_XYZ = new vector<XYZD>[nCams], *XYZ = new vector<XYZD>[ntracks], XYZBK;

	//Make sure that no gtOff is smaller than 0
	for (int ii = 0; ii < nCams; ii++)
		if (rate*ii + gtOff[ii] < 0)
			gtOff[ii] += rate;

	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		//Read 3D
		sprintf(Fname, "%s/3DTracks/%d.txt", Path, trackID); FILE *fp = fopen(Fname, "r");
		while (fscanf(fp, "%lf %lf %lf", &x, &y, &z) != EOF)
		{
			XYZD t; t.xyz.x = x, t.xyz.y = y, t.xyz.z = z, t.d = 0.0;
			XYZ[trackID].push_back(t);
		}
		fclose(fp);
		int nf = XYZ[trackID].size();

		//Read 2D Points
		for (int camID = 0; camID < nCams; camID++)
		{
			PerCamUV_All[camID].clear();
			sprintf(Fname, "%s/2DTracks/%d_%d.txt", Path, camID, trackID); fp = fopen(Fname, "r");
			while (fscanf(fp, "%lf %lf ", &x, &y) != EOF)
			{
				ptEle.pt2D.x = x, ptEle.pt2D.y = y;
				PerCamUV_All[camID].push_back(ptEle);
			}
			fclose(fp);
		}

		//Get ray direction, Q, U, P
		for (int camID = 0; camID < nCams; camID++)
		{
			for (int frameID = 0; frameID < nf; frameID++)
			{
				for (int kk = 0; kk < 12; kk++)
				{
					P[kk] = AllVideoInfo.VideoInfo[camID*nframes + frameID].P[kk];
					PerCamUV_All[camID][frameID].P[kk] = P[kk];
				}
				for (int kk = 0; kk < 9; kk++)
					PerCamUV_All[camID][frameID].K[kk] = AllVideoInfo.VideoInfo[camID*nframes + frameID].K[kk],
					PerCamUV_All[camID][frameID].R[kk] = AllVideoInfo.VideoInfo[camID*nframes + frameID].R[kk];

				//Q, U
				AA[0] = P[0], AA[1] = P[1], AA[2] = P[2], bb[0] = P[3];
				AA[3] = P[4], AA[4] = P[5], AA[5] = P[6], bb[1] = P[7];
				ccT[0] = P[8], ccT[1] = P[9], ccT[2] = P[10], dd[0] = P[11];

				PerCamUV_All[camID][frameID].Q[0] = AA[0] - PerCamUV_All[camID][frameID].pt2D.x*ccT[0], PerCamUV_All[camID][frameID].Q[1] = AA[1] - PerCamUV_All[camID][frameID].pt2D.x*ccT[1], PerCamUV_All[camID][frameID].Q[2] = AA[2] - PerCamUV_All[camID][frameID].pt2D.x*ccT[2];
				PerCamUV_All[camID][frameID].Q[3] = AA[3] - PerCamUV_All[camID][frameID].pt2D.y*ccT[0], PerCamUV_All[camID][frameID].Q[4] = AA[4] - PerCamUV_All[camID][frameID].pt2D.y*ccT[1], PerCamUV_All[camID][frameID].Q[5] = AA[5] - PerCamUV_All[camID][frameID].pt2D.y*ccT[2];
				PerCamUV_All[camID][frameID].u[0] = dd[0] * PerCamUV_All[camID][frameID].pt2D.x - bb[0], PerCamUV_All[camID][frameID].u[1] = dd[0] * PerCamUV_All[camID][frameID].pt2D.y - bb[1];
				PerCamUV_All[camID][frameID].pt3D = XYZ[trackID][frameID].xyz;
			}
		}

		//Check if 2D is ok
		for (int frameID = 0; frameID < nf; frameID++)
		{
			double TpreprojErr = 0.0, preprojErr;
			//Test Q U
			for (int camID = 0; camID < nCams; camID++)
			{
				for (int ii = 0; ii < 6; ii++)
					Q[camID * 6 + ii] = PerCamUV_All[camID][frameID].Q[ii];
				U[camID * 2] = PerCamUV_All[camID][frameID].u[0], U[camID * 2 + 1] = PerCamUV_All[camID][frameID].u[1];
			}
			LS_Solution_Double(Q, U, nCams * 2, 3);

			for (int camID = 0; camID < nCams; camID++)
			{
				ProjectandDistort(Point3d(U[0], U[1], U[2]), ImgPts, AllVideoInfo.VideoInfo[camID*nframes + frameID].P);
				preprojErr = Distance2D(PerCamUV_All[camID][frameID].pt2D, ImgPts[0]);
				TpreprojErr += preprojErr;
			}
			if (TpreprojErr / nCams > 10.0)
				printf("Algebraic 3D-2D projection warning\n");

			//Test P
			TpreprojErr = 0.0;
			for (int camID = 0; camID < nCams; camID++)
			{
				ProjectandDistort(XYZ[trackID][frameID].xyz, ImgPts, AllVideoInfo.VideoInfo[camID*nframes + frameID].P);
				preprojErr = Distance2D(PerCamUV_All[camID][frameID].pt2D, ImgPts[0]);
				TpreprojErr += preprojErr;
			}
			if (TpreprojErr / nCams > 10.0)
				printf("Geometric 3D-2D projection warning\n");
		}

		//Assign 2D to cameras
		int maxPerCamFrames = 0;
		for (int camID = 0; camID < nCams; camID++)
		{
			for (int frameID = 0; frameID < nf; frameID += rate)
			{
				int tid = frameID + gtOff[camID];
				if (tid >= nf || tid < 0)
					continue;
				PerCam_UV[camID*ntracks + trackID].push_back(PerCamUV_All[camID][tid]);
			}
			if (PerCam_UV[camID*ntracks + trackID].size() > maxPerCamFrames)
				maxPerCamFrames = PerCam_UV[camID*ntracks + trackID].size();
		}

		//Add noise to 3d data 
		if (0)
		{
			XYZBK.clear();
			for (int frameID = 0; frameID < nf; frameID++)
			{
				XYZBK.push_back(XYZ[trackID][frameID]);
				XYZBK[frameID].xyz.x += max(min(gaussian_noise(0.0, stdev), 3.0*stdev), -3.0*stdev);
				XYZBK[frameID].xyz.y += max(min(gaussian_noise(0.0, stdev), 3.0*stdev), -3.0*stdev);
				XYZBK[frameID].xyz.z += max(min(gaussian_noise(0.0, stdev), 3.0*stdev), -3.0*stdev);
			}

			for (int frameID = 0; frameID < nf; frameID++)
			{
				for (int camID = 0; camID < nCams; camID++)
				{
					for (int kk = 0; kk < 12; kk++)
						P[kk] = PerCamUV_All[camID][frameID].P[kk];

					ImgPtEle ptEle;
					ptEle.pt2D.x = PerCamUV_All[camID][frameID].pt2D.x, ptEle.pt2D.y = PerCamUV_All[camID][frameID].pt2D.y;
					for (int ii = 0; ii < 3; ii++)
						ptEle.C[ii] = PerCamUV_All[camID][frameID].C[ii], ptEle.ray[ii] = PerCamUV_All[camID][frameID].ray[ii];

					PerCamUV_All[camID][frameID].pt3D = XYZBK[frameID].xyz;
				}
			}

			//Update 2D to cameras
			for (int camID = 0; camID < nCams; camID++)
			{
				PerCam_UV[camID*ntracks + trackID].clear();
				for (int frameID = 0; frameID < nf; frameID += rate)
				{
					int tid = frameID + gtOff[camID];
					if (tid >= nf || tid < 0)
						continue;
					PerCam_UV[camID*ntracks + trackID].push_back(PerCamUV_All[camID][tid]);
				}
			}

			//Tell me how bad reprojection error is
			for (int frameID = 0; frameID < nf; frameID++)
			{
				costi = 0.0;
				for (int camID = 0; camID < nCams; camID++)
				{
					ProjectandDistort(PerCamUV_All[camID][frameID].pt3D, ImgPts, PerCamUV_All[camID][frameID].P);
					double preprojErr = Distance2D(PerCamUV_All[camID][frameID].pt2D, ImgPts[0]);
					costi += preprojErr;
				}
				if (costi / nCams > 2.0*(stdev + 1))
					printf("3D-2D projection warning\n");
				TProjCost += costi / nCams;
			}
		}
		else
		{
			//Triangulate to get depth from badly syn cameras
			AllError2D.clear();
			for (int frameID = 0; frameID < maxPerCamFrames; frameID++)
			{
				triangulatedList.clear();
				int triangulatable = 0;
				for (int camID = 0; camID < nCams; camID++)
				{
					if (PerCam_UV[camID*ntracks + trackID].size() > frameID)
					{
						for (int ii = 0; ii < 12; ii++)
							AllP[triangulatable * 12 + ii] = PerCam_UV[camID*ntracks + trackID][frameID].P[ii];
						ImgPts[triangulatable] = PerCam_UV[camID*ntracks + trackID][frameID].pt2D;
						triangulatedList.push_back(camID);
						triangulatable++;
					}
				}

				if (triangulatable == 0)
					continue;
				if (triangulatable == 1)
					PerCam_UV[triangulatedList[0] * ntracks + trackID][frameID].d = PerCam_UV[triangulatedList[0] * ntracks + trackID][frameID - 1].d;
				else
				{
					NviewTriangulation(ImgPts, AllP, &P3D, triangulatable, 1, NULL, A, B);
					double pt2d[10], pt3d[3], reProjErr[5];
					for (int ii = 0; ii < triangulatable; ii++)
						pt2d[2 * ii] = ImgPts[ii].x, pt2d[2 * ii + 1] = ImgPts[ii].y;
					pt3d[0] = P3D.x, pt3d[1] = P3D.y, pt3d[2] = P3D.z;
					NviewTriangulationNonLinear(AllP, pt2d, pt3d, reProjErr, triangulatable);
					P3D.x = pt3d[0], P3D.y = pt3d[1], P3D.z = pt3d[2];

					for (int jj = 0; jj < triangulatable; jj++)
					{
						int camID = triangulatedList[jj];
						ptEle.pt2D = PerCam_UV[camID*ntracks + trackID][frameID].pt2D;
						for (int ii = 0; ii < 3; ii++)
							ptEle.C[ii] = PerCam_UV[camID*ntracks + trackID][frameID].C[ii], ptEle.ray[ii] = PerCam_UV[camID*ntracks + trackID][frameID].ray[ii];

						double stdA = 0.00001;
						PerCam_UV[camID*ntracks + trackID][frameID].pt3D = Point3d(P3D.x + gaussian_noise(0.0, stdA), P3D.y + gaussian_noise(0.0, stdA), P3D.z + gaussian_noise(0.0, stdA));

						ProjectandDistort(P3D, ImgPts, PerCam_UV[camID*ntracks + trackID][frameID].P);
						double preprojErr = Distance2D(PerCam_UV[camID*ntracks + trackID][frameID].pt2D, ImgPts[0]);
						AllError2D.push_back(preprojErr);
					}
				}
			}
			double m2D = MeanArray(AllError2D);
			double std2D = sqrt(VarianceArray(AllError2D));
			printf("Track %d error statistic: (m, std)  = (%.2f %.2f)\n", trackID, m2D, std2D);
		}

		//In case GT 3D is needed
		if (save3D)
		{
			sprintf(Fname, "%s/GTTrack_%d.txt", Path, trackID), fp = fopen(Fname, "w+");
			for (int ii = 0; ii < nCams; ii++)
			{
				for (int jj = 0; jj < nf; jj += rate)
				{
					int tid = jj + gtOff[ii];
					if (tid >= nf || tid < 0)
						continue;
					fprintf(fp, "%f %f %f %d\n", XYZ[trackID][tid].xyz.x, XYZ[trackID][tid].xyz.y, XYZ[trackID][tid].xyz.z, tid);
				}
			}
			fclose(fp);
		}
	}

	//printf("Initial time:\n");
	//for (int ll = 0; ll < nCams; ll++)
	//	printf("%d %f\n", ll, currentOffset[ll]);

	vector<int> PointsPerTrack;
	vector<int *> AllFrameID(ntracks);
	vector<double*> All3D(ntracks);
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		int npts = 0;
		for (int camID = 0; camID < nCams; camID++)
			npts += PerCam_UV[camID*ntracks + trackID].size();
		AllFrameID[trackID] = new int[npts];
		All3D[trackID] = new double[3 * npts];
	}

	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		sprintf(Fname, "%s/BTrack_%d.txt", Path, trackID);  FILE *fp = fopen(Fname, "w+");
		for (int camID = 0; camID < nCams; camID++)
			for (int fid = 0; fid < PerCam_UV[camID*ntracks + trackID].size(); fid++)
				fprintf(fp, "%.3f %.3f %.3f %.1f\n", PerCam_UV[camID*ntracks + trackID][fid].pt3D.x, PerCam_UV[camID*ntracks + trackID][fid].pt3D.y, PerCam_UV[camID*ntracks + trackID][fid].pt3D.z, 1.0*currentOffset[camID] * ialpha*Tscale + 1.0*fid * ialpha*Tscale*rate);
		fclose(fp);
	}

	fixedTime = false, fixed3D = false;
	LeastActionOptimXYZT(Path, All3D, PerCam_UV, PointsPerTrack, currentOffset, ntracks, fixedTime, fixed3D, non_monotonicDescent, nCams, Tscale, ialpha, rate, eps, lamda2DCost, lamdaTimeTripletCost);

	//printf("Saving results...\n");
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		sprintf(Fname, "%s/ATrack_%d.txt", Path, trackID);  FILE *fp = fopen(Fname, "w+");
		for (int camID = 0; camID < nCams; camID++)
			for (int fid = 0; fid < PerCam_UV[camID*ntracks + trackID].size(); fid++)
				fprintf(fp, "%.3f %.3f %.3f %.1f\n", PerCam_UV[camID*ntracks + trackID][fid].pt3D.x, PerCam_UV[camID*ntracks + trackID][fid].pt3D.y, PerCam_UV[camID*ntracks + trackID][fid].pt3D.z, 1.0*currentOffset[camID] * ialpha*Tscale + 1.0*fid * ialpha*Tscale*rate);
		fclose(fp);
	}

	/*printf("Time after 1-iter: \n");
	for (int ll = 0; ll < nCams; ll++)
	printf("%d %f\n", ll, currentOffset[ll]);

	fixedTime = true, fixed3D = false;
	LeastActionOptimXYZT(Path, All3D, PerCam_UV, PointsPerTrack, currentOffset, ntracks, fixedTime, fixed3D, non_monotonicDescent, nCams, Tscale, ialpha, rate, eps, lamda2DCost);

	for (int trackID = 0; trackID < ntracks; trackID++)
	{
	sprintf(Fname, "%s/A2Track_%d.txt", Path, trackID);  FILE *fp = fopen(Fname, "w+");
	for (int camID = 0; camID < nCams; camID++)
	for (int fid = 0; fid < PerCam_UV[camID*ntracks + trackID].size(); fid++)
	fprintf(fp, "%.3f %.3f %.3f %.1f\n", PerCam_UV[camID*ntracks + trackID][fid].pt3D.x, PerCam_UV[camID*ntracks + trackID][fid].pt3D.y, PerCam_UV[camID*ntracks + trackID][fid].pt3D.z, 1.0*currentOffset[camID] * ialpha*Tscale + 1.0*fid * ialpha*Tscale*rate);
	fclose(fp);
	}

	fixedTime = false, fixed3D = false;
	LeastActionOptimXYZT(Path, All3D, PerCam_UV, PointsPerTrack, currentOffset, ntracks, fixedTime, fixed3D, non_monotonicDescent, nCams, Tscale, ialpha, rate, eps, lamda2DCost);
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
	sprintf(Fname, "%s/A3Track_%d.txt", Path, trackID);  FILE *fp = fopen(Fname, "w+");
	for (int camID = 0; camID < nCams; camID++)
	for (int fid = 0; fid < PerCam_UV[camID*ntracks + trackID].size(); fid++)
	fprintf(fp, "%.3f %.3f %.3f %.1f\n", PerCam_UV[camID*ntracks + trackID][fid].pt3D.x, PerCam_UV[camID*ntracks + trackID][fid].pt3D.y, PerCam_UV[camID*ntracks + trackID][fid].pt3D.z, 1.0*currentOffset[camID] * ialpha*Tscale + 1.0*fid * ialpha*Tscale*rate);
	fclose(fp);
	}*/
	printf("Final results\n");
	for (int ll = 0; ll < nCams; ll++)
		printf("%d %f\n", ll, currentOffset[ll]);

	for (int trackID = 0; trackID < ntracks; trackID++)
		delete[]All3D[trackID];
	for (int ii = 0; ii < ntracks*nCams; ii++)
		PerCam_UV[ii].clear();
	for (int ii = 0; ii < nCams; ii++)
		PerCam_XYZ[ii].clear();

	delete[]PerCam_UV, delete[]PerCam_XYZ;
	delete[]Q, delete[]U, delete[]AllP, delete[]A, delete[]B, delete[]ImgPts;

	return 0;
}
int main(int argc, char* argv[])
{
	char Path[] = "E:/Data/Phuong", Fname[200], Fname2[200];

	/*srand((unsigned int)time(NULL));
#pragma omp parallel for
	for (int ii = 1; ii <= 13; ii++)
	{
	//Interpolation3DTrajectory(ii);
	//Generate2DTracksFrom3D_Simu(Path, 5, ii, 1, 1, 2600);
	//GenarateTrajectoryInput(Path, 2, 1, 2600, 1, ii, 0, 50.0);
	}

	int nCams = 5, nTracks = 13, width = 1920, height = 1080;
	double Intrinsic[5] = { 1500, 1500, 0, 960, 540 }, radius = 3000;
	//SimulateCamerasAnd2DPointsForMoCap(Path, nCams, nTracks, Intrinsic, width, height, radius, false, true);
	//visualizationDriver(Path, nCams, 0,7000, false, false, true, true, false, 0);
	//return 0;

	if (argc == 2)
	SimultaneousSyncReconc(Path, nCams, nTracks, atof(argv[1]));
	else
	visualizationDriver(argv[1], nCams, 0, atoi(argv[2]), false, false, false, true, false, 0);
	return 0;*/


	/*int actionID[] = { 9 };
	double stdev[] = { 0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5 };
	for (int nCams = 2; nCams <= 2; nCams++)
	{
	for (int kk = 0; kk < 1; kk++)
	{
	TestLeastActionConstraint(nCams, stdev[0], actionID[kk], 1);
	#pragma omp parallel for
	for (int jj = 1; jj < 7; jj++)
	TestLeastActionConstraint(nCams, stdev[jj], actionID[kk], 1000);
	}
	}
	return 0;*/

	int mode = 5;// atoi(argv[1]);

	srand((unsigned int)time(NULL));
	if (mode == 0) //Step 1: sync all sequences with audio
	{
		int computeSync = 0;// atoi(argv[2]);
		if (computeSync == 1)
		{
			int pair = 0;// atoi(argv[3]);
			int minSegLength = 3e6;
			if (pair == 1)
			{
				int srcID1 = 1,// atoi(argv[4]),
					srcID2 = 14;// atoi(argv[5]);
				double fps1 = 1,//atof(argv[6]),
					fps2 = 1,// atof(argv[7]),
					minZNCC = 0.3;// atof(argv[8]);

				double offset = 0.0, ZNCC = 0.0;
				sprintf(Fname, "%s/%d/audio.wav", Path, srcID1);
				sprintf(Fname2, "%s/%d/audio.wav", Path, srcID2);
				if (SynAudio(Fname, Fname2, fps1, fps2, minSegLength, offset, ZNCC, minZNCC) != 0)
					printf("Not succeed\n");
				else
					printf("Succeed\n");
			}
			else
			{
				double	minZNCC = 0.15;// atof(argv[4]);
				int minSegLength = 3e6;

				const int nvideos = 4;
				int camID[] = { 2, 3, 4, 5 };
				double fps[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };//to avoid drift at this stage
				double offset, ZNCC;

				sprintf(Fname, "%s/audioSync.txt", Path);
				FILE *fp = fopen(Fname, "w+");
				for (int jj = 0; jj < nvideos - 1; jj++)
				{
					for (int ii = jj + 1; ii < nvideos; ii++)
					{
						sprintf(Fname, "%s/%d/audio.wav", Path, camID[jj]);
						sprintf(Fname2, "%s/%d/audio.wav", Path, camID[ii]);
						if (SynAudio(Fname, Fname2, fps[jj], fps[ii], minSegLength, offset, ZNCC, minZNCC) != 0)
						{
							printf("Between %d and %d: not succeed\n", camID[jj], camID[ii]);
							fprintf(fp, "%d %d %.4f\n", camID[jj], camID[ii], 1.0*rand());
						}
						else
							fprintf(fp, "%d %d %.4f\n", camID[jj], camID[ii], offset);
					}
				}
				fclose(fp);

				PrismMST(Path, nvideos);
				AssignOffsetFromMST(Path, nvideos);
			}
		}
		else
		{
			//int writeSyncedImages = atoi(argv[3]);
			ShowSyncLoad(Path, Path);
		}
		return 0;
	}
	else if (mode == 1) //step 2: calibrate camera individually if needed
	{
		char *Path = argv[2],
			*VideoName = argv[3];
		int SaveFrameDif = atoi(argv[4]);//20;
		int nNonBlurIma = 0;
		PickStaticImagesFromVideo(Path, VideoName, SaveFrameDif, 15, .3, 50, nNonBlurIma, true);
		//BlurDetectionDriver(Path, nNonBlurIma, 1920, 1080, 0.1);
	}
	else if (mode == 2) //Step 3: do BA of visSfM and rescale the result
	{
		printf("Input the Path name and number of points per image\n");
		TriangulatePointsFromCalibratedCameras("E:/Data/Phuong/Corpus", 0, 2, 2.0);
		cout << "Input conversion ratio (real unit / visSfM): ";
		double ratio; cin >> ratio;
		ReSaveBundleAdjustedNVMResults("E:/Data/Phuong/Corpus/BA_Camera_AllParams_after.txt", ratio);
		return 0;
	}
	else if (mode == 3) //step 4: generate corpus
	{
		int nviews = 94,//atoi(argv[2]),
			NPplus = 5,//atoi(argv[3]),
			runMatching = 0,//atoi(argv[4]),
			distortionCorrected = 0;//atoi(argv[5]);

		sprintf(Fname, "%s/Corpus", Path);
		if (runMatching == 1)
		{
			int HistogramEqual = 0,//atoi(argv[6]),
				OulierRemoveTestMethod = 2,//atoi(argv[7]), //Fmat is more prefered. USAC is much faster then 5-pts :(
				LensType = 0;// atoi(argv[8]);
			double ratioThresh = 0.8;// atof(argv[9]);

			int nCams = nviews, cameraToScan = -1;
			if (OulierRemoveTestMethod == 2)
				nCams = 94,//atoi(argv[10]),
				cameraToScan = -1;// atoi(argv[11]);

			int timeID = -1, ninlierThesh = 50;
			GeneratePointsCorrespondenceMatrix_SiftGPU2(Fname, nviews, -1, HistogramEqual, ratioThresh);

#pragma omp parallel
			{
#pragma omp for nowait
				for (int jj = 0; jj < nviews - 1; jj++)
				{
					for (int ii = jj + 1; ii < nviews; ii++)
					{
						if (OulierRemoveTestMethod == 1)
							EssentialMatOutliersRemove(Fname, timeID, jj, ii, nCams, cameraToScan, ninlierThesh, distortionCorrected, false);
						else if (OulierRemoveTestMethod == 2)
							FundamentalMatOutliersRemove(Fname, timeID, jj, ii, ninlierThesh, LensType, distortionCorrected, false, nCams, cameraToScan);
					}
				}
			}
			GenerateMatchingTable(Fname, nviews, timeID);
		}
		else
		{
			BuildCorpus(Fname, -1, distortionCorrected, NPplus);
			visualizationDriver(Fname, 1, -1, -1, true, false, false, false);
		}
		return 0;
	}
	else if (mode == 4) //Step 5: Get features data for test sequences
	{
		int nviews = 9,//atoi(argv[2]),
			selectedView = 0,//atoi(argv[3]),
			startF = 1,//atoi(argv[4]),
			stopF = 2500,//atoi(argv[5]),
			HistogrameEqual = 0;// atoi(argv[6]);

		Corpus corpusData;
		sprintf(Fname, "%s/BA_Camera_AllParams_after.txt", Path);
		if (!loadBundleAdjustedNVMResults(Fname, corpusData))
		{
			corpusData.camera = new CameraData[nviews];
			if (!ReadIntrinsicResults(Path, corpusData.camera))
				return 1;
		}

		vector<int> availViews;
		availViews.push_back(selectedView);
		sprintf(Fname, "%s/%d", Path, selectedView);
		printf("Working on %s: ", Fname);

		//LensCorrectionImageSequenceDriver(Fname, corpusData.camera[selectedView].K, corpusData.camera[selectedView].distortion, corpusData.camera[selectedView].LensModel, startF, stopF, 1.0, 1.0, 5);
		ExtractSiftGPUfromExtractedFrames(Path, availViews, startF, stopF, HistogrameEqual);

		return 0;
	}
	else if (mode == 5) //Step 6: Localize test sequence wrst to corpus
	{
		int StartTime = 1,//atoi(argv[2]),
			StopTime = 2500,//atoi(argv[3]),
			seletectedCam = 0,//atoi(argv[4]),
			nCams = 9,//atoi(argv[5]),
			runMatching = -1,//atoi(argv[6]),
			distortionCorrected = 0,//atoi(argv[7]), //must set to 1 when runMatching != 1;
			sharedIntriniscOptim = 0,//atoi(argv[8]),
			LensType = 0;// atoi(argv[9]);// RADIAL_TANGENTIAL_PRISM;// FISHEYE;

		//sprintf(Fname, "%s/intrinsic_%d.txt", Path, seletectedCam); FILE*fp = fopen(Fname, "w+");	fclose(fp);
		if (runMatching >= 0)
			LocalizeCameraFromCorpusDriver(Path, StartTime, StopTime, runMatching, nCams, seletectedCam, distortionCorrected, sharedIntriniscOptim, LensType);
		if (runMatching != 1)
			visualizationDriver(Path, nCams, StartTime, StopTime, true, false, true, false, false, StartTime);
		return 0;
	}
	else if (mode == 6) //step 7: generate 3d data from test sequences
	{
		int StartTime = 1,// atoi(argv[2]),
			StopTime = 500,//atoi(argv[3]),
			timeStep = 50, //atoi(argv[4])
			RunMatching = 0;// atoi(argv[5]);

		int HistogrameEqual = 0,
			distortionCorrected = 1,
			OulierRemoveTestMethod = 2, ninlierThesh = 40, //fmat test
			LensType = RADIAL_TANGENTIAL_PRISM,
			NViewPlus = 3,
			nviews = 9;
		double ratioThresh = 0.4, reprojectionThreshold = 3;

		if (RunMatching == 1)
		{
			for (int timeID = StartTime; timeID <= StopTime; timeID += timeStep)
			{
				GeneratePointsCorrespondenceMatrix_SiftGPU2(Path, nviews, timeID, HistogrameEqual, 0.6);
#pragma omp parallel 
				{
#pragma omp for nowait
					for (int jj = 0; jj < nviews - 1; jj++)
					{
						for (int ii = jj + 1; ii < nviews; ii++)
						{
							if (OulierRemoveTestMethod == 1)
								EssentialMatOutliersRemove(Path, timeID, jj, ii, nviews, 0, ninlierThesh, distortionCorrected, false);
							if (OulierRemoveTestMethod == 2)
								FundamentalMatOutliersRemove(Path, timeID, jj, ii, ninlierThesh, LensType, distortionCorrected, false, nviews, 0);
						}
					}
					GenerateMatchingTable(Path, nviews, timeID);
				}
			}
		}
		else
			Build3DFromSyncedImages(Path, nviews, StartTime, StopTime, timeStep, LensType, distortionCorrected, NViewPlus, reprojectionThreshold, true, 10.0);

		return 0;
	}
	else if (mode == 7)
	{
		int nVideoCams = 5,
			startTime = 0, stopTime = 199,
			LensModel = RADIAL_TANGENTIAL_PRISM;

		int selectedCams[] = { 1, 2 }, selectedTime[2] = { 0, 0 }, ChooseCorpusView1 = -1, ChooseCorpusView2 = -1;

		VideoData AllVideoInfo;
		ReadVideoData(Path, AllVideoInfo, nVideoCams, startTime, stopTime);

		double Fmat[9];
		int seletectedIDs[2] = { selectedCams[0] * (stopTime - startTime + 1) + selectedTime[0], selectedCams[1] * (stopTime - startTime + 1) + selectedTime[1] };
		computeFmatfromKRT(AllVideoInfo.VideoInfo, 5, seletectedIDs, Fmat);
		return 0;
	}
	else if (mode == 8)
	{
		GenerateCorpusVisualWords("C:/temp/BOW", 24);
		ComputeWordsHistogram("C:/temp/BOW", 3);
	}
	else if (mode == 9)
	{
		//siftgpu("D:/1_0.png", "D:/2_0.png", 0.5, 0.5);
		//VisualizeCleanMatches(Path, 1, 2, 0, 0.01);

		/*int submode = atoi(argv[2]);
		if (submode == -1)
		ShowSyncLoad2("C:/Data/GoPro", 0);
		else if (submode == 0)
		ShowSyncLoad(Path, 0);
		else if (submode == 1)
		visualizationDriver(Path, 5, 0, 199, true, true, false);
		else if (submode == 2)
		visualizationDriver(Path, 1, 0, 198, true, false, true);*/

		/*char ImgName[200], Kname[200], RGBName[200];
		for (int ii = 0; ii < 10; ii++)
		{
		for (int jj = 0; jj <= 290; jj++)
		{
		sprintf(ImgName, "%s/%d/%d.png", Path, ii, jj);
		sprintf(Kname, "%s/%d/K%d.dat", Path, ii, jj);
		sprintf(RGBName, "%s/%d/RGB%d.dat", Path, ii, jj);
		GenereteKeyPointsRGB(ImgName, Kname, RGBName);
		printf("Wrote %s\n", RGBName);
		}
		}*/
		/*double K[] = { 5.9894787542305494e+002, 0.0000000000000000e+000, 6.2486606812417028e+002,
		0, 5.9623146393896582e+002, 3.6152639154296548e+002,
		0, 0, 1 };

		double Distortion[] = { 1.6472154585326839e-003, 6.2420964417695950e+002, 3.6077115979767234e+002 };
		int startFrame = 0, stopFrame = 10;

		LensCorrectionImageSequenceDriver("D:/Disney", K, Distortion, FISHEYE, startFrame, stopFrame, 1.5, 1.0, 5);
		return 0;*/

		const int nHDs = 30, nCamsPerPanel = 24;
		TrajectoryData InfoTraj;
		//CameraData CameraInfo[480 + nHDs];

		int nviews = 9, startTime = 1, stopTime = 500, timeID = 0;// atoi(argv[2]);
		VideoData AllVideoInfo;
		sprintf(Fname, "%s/In/Calib", Path);
		if (ReadVideoData(Fname, AllVideoInfo, nviews, startTime, stopTime) == 1)
			return 1;

		/*LoadTrackData(Path, timeID, InfoTraj, true);
		//Write3DMemAtThatTime(Path, InfoTraj, CameraInfo, 100, 101);

		vector<int> TrajectUsed;
		for (int ii = 0; ii < InfoTraj.nTrajectories; ii++)
		TrajectUsed.push_back(ii);

		Genrate2DTrajectory2(Path, timeID, InfoTraj, AllVideoInfo, TrajectUsed);
		//Genrate2DTrajectory(Path, 120, InfoTraj, CameraInfo, TrajectUsed);//
		return 0;*/

		int ntraj = 5917;
		Load2DTrajectory(Path, InfoTraj, ntraj);

		int minFrame = 1, maxFrame = 400;
		int temporalOffsetRange[] = { -6, 6 };
		int list[] = { 0, 1, 3, 4, 5, 6, 7, 8 };
		for (int jj = 0; jj < 8; jj++)
		{
			int cameraPair[] = { 2, list[jj] };
			Compute3DTrajectoryErrorZNCC2(Path, InfoTraj, ntraj, minFrame, maxFrame, cameraPair, temporalOffsetRange);
		}

		/*int list[] = { 0, 1, 8, 10, 11, 14, 16, 17, 19, 23, 24, 25, 29 };
		for (int jj = 0; jj < 13; jj++)
		{
		int cameraPair[] = { 30, list[jj] };
		Compute3DTrajectoryErrorZNCC(Path, InfoTraj, ntraj, minFrame, maxFrame, cameraPair, temporalOffsetRange);
		}

		for (int kk = 0; kk < 6; kk++)
		{
		for (int jj = 1; jj < nCamsPerPanel; jj++)
		{
		int cameraPair[] = { kk * nCamsPerPanel + nHDs, kk * nCamsPerPanel + jj + nHDs };
		Compute3DTrajectoryErrorZNCC(Path, InfoTraj, ntraj, minFrame, maxFrame, cameraPair, temporalOffsetRange);
		}
		}

		for (int kk = 0; kk < 6; kk++)
		{
		int cameraPair[] = { kk * nCamsPerPanel + nHDs, (kk + 1) * nCamsPerPanel + nHDs };
		Compute3DTrajectoryErrorZNCC(Path, InfoTraj, ntraj, minFrame, maxFrame, cameraPair, temporalOffsetRange);
		}*/
		return 0;

	}


	return 0;
}
