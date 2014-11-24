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
int ShowSyncLoad(char *DataPATH, int writeSyncedFrames = 0)
{
	//bad frames: 1510 1741 1765 1899
	char Fname[2000];
	const int playBackSpeed = 1, nsequences = 10, refSeq = 1;
	int seqName[nsequences] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }; //{ 2, 7 };

	int WBlock = 1920, HBlock = 1080, nBlockX = 5, nchannels = 3, MaxFrames = 5500;

	//Make sure to use 59.94 instead of 60 
	Sequence mySeq[nsequences];
	/*mySeq[0].InitSeq(47.95, 4.761);
	mySeq[1].InitSeq(47.95, - 8.3183);
	mySeq[2].InitSeq(47.95,  4.761+0.80816);
	mySeq[3].InitSeq(47.95,  0.0);*/

	/*//My wedding sequences: 1 3 5 7
	mySeq[0].InitSeq(100.0,   333.3126);
	mySeq[1].InitSeq(59.94, 0);
	mySeq[2].InitSeq(59.94,  660.8185);
	mySeq[3].InitSeq(59.94,  700.2928);*/

	// soccer sequences: 
	mySeq[0].InitSeq(47.95, 0);
	mySeq[1].InitSeq(47.95, 72.78);
	mySeq[2].InitSeq(47.95, 108.19);
	mySeq[3].InitSeq(47.95, 91.42);
	mySeq[4].InitSeq(47.95, 91.38);
	mySeq[5].InitSeq(47.95, 89.52);
	mySeq[6].InitSeq(47.95, 79.56);
	mySeq[7].InitSeq(47.95, 90.74);
	mySeq[8].InitSeq(47.95, 89.62);
	mySeq[9].InitSeq(47.95, 78.93);
	/*mySeq[0].InitSeq(47.95, 110.19 - 90.74);
	mySeq[1].InitSeq(47.95, 90.74 - 90.74);*/

	//clock with srinivas
	//mySeq[0].InitSeq(29.97, 18.08);
	//mySeq[1].InitSeq(29.97, 0);

	//Juggling
	//mySeq[0].InitSeq(29.97, 0);
	//mySeq[1].InitSeq(29.97, 0);// 70.57 + 383.24);
	//mySeq[2].InitSeq(29.97, 0);// 70.57 + 167.79);
	//mySeq[3].InitSeq(29.97, 0);// 70.57 + 367.67);

	int allFramesOn = 0;
	if (mySeq[0].TimeAlignPara[0] == 47.95)
		allFramesOn = 150;

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
		oFrameID[ii] = allFramesOn, FrameID[ii] = allFramesOn;
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

	int SaveFrameCount = 0;
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
				sprintf(Fname, "%s/Corrected/%d/%d.png", DataPATH, seqName[ii], setSeqFrame);
				if (GrabImage(Fname, SubImage, swidth, sheight, nchannels) && ii == 5)
					Set_Sub_Mat(SubImage, BigImg, nchannels*swidth, sheight, nchannels*width, nchannels*BlockXID*WBlock, BlockYID*HBlock);
				else
					Set_Sub_Mat(BlackImage, BigImg, nchannels*WBlock, HBlock, nchannels*width, nchannels*BlockXID*WBlock, BlockYID*HBlock);

				if (writeSyncedFrames == 1 && ii == 5)
				{
					//Han's format
					sprintf(Fname, "%s/%.8d", DataPATH, SaveFrameCount / nsequences);	_mkdir(Fname);
					sprintf(Fname, "%s/%.8d/%.8d_00_%.2d.png", DataPATH, SaveFrameCount / nsequences, SaveFrameCount / nsequences, seqName[ii]);
					SaveDataToImage(Fname, SubImage, swidth, sheight, nchannels);

					sprintf(Fname, "%s/%d", DataPATH, seqName[ii]);	_mkdir(Fname);
					sprintf(Fname, "%s/%d/%d.png", DataPATH, seqName[ii], SaveFrameCount / nsequences);
					SaveDataToImage(Fname, SubImage, swidth, sheight, nchannels);
					SaveFrameCount++;
				}
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
int ShowSyncLoad2(char *DataPATH, int writeSyncedFrames = 0)
{
	//bad frames: 1510 1741 1765 1899
	char Fname[2000];
	const int playBackSpeed = 1, nsequences = 2, refSeq = 1;
	int seqName[nsequences] = { 2, 7 };

	int WBlock = 1920, HBlock = 1080, nBlockX = 2, nchannels = 3, MaxFrames = 5500;

	//Make sure to use 59.94 instead of 60 
	Sequence mySeq[nsequences];
	/*mySeq[0].InitSeq(47.95, 4.761);
	mySeq[1].InitSeq(47.95, - 8.3183);
	mySeq[2].InitSeq(47.95,  4.761+0.80816);
	mySeq[3].InitSeq(47.95,  0.0);*/

	/*//My wedding sequences: 1 3 5 7
	mySeq[0].InitSeq(100.0,   333.3126);
	mySeq[1].InitSeq(59.94, 0);
	mySeq[2].InitSeq(59.94,  660.8185);
	mySeq[3].InitSeq(59.94,  700.2928);*/

	// soccer sequences: 
	mySeq[0].InitSeq(47.95, 110.19 - 90.74);
	mySeq[1].InitSeq(47.95, 90.74 - 90.74);

	//clock with srinivas
	//mySeq[0].InitSeq(29.97, 18.08);
	//mySeq[1].InitSeq(29.97, 0);


	int allFramesOn = 0;
	if (mySeq[0].TimeAlignPara[0] == 47.95)
		allFramesOn = 150;

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
		oFrameID[ii] = allFramesOn, FrameID[ii] = allFramesOn;
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

	int SaveFrameCount = 0;
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
				sprintf(Fname, "%s/RawImages/%d/%d.png", DataPATH, seqName[ii], setSeqFrame);
				if (GrabImage(Fname, SubImage, swidth, sheight, nchannels))
					Set_Sub_Mat(SubImage, BigImg, nchannels*swidth, sheight, nchannels*width, nchannels*BlockXID*WBlock, BlockYID*HBlock);
				else
					Set_Sub_Mat(BlackImage, BigImg, nchannels*WBlock, HBlock, nchannels*width, nchannels*BlockXID*WBlock, BlockYID*HBlock);

				if (writeSyncedFrames == 1)
				{
					sprintf(Fname, "%s/%d/%d.png", DataPATH, seqName[ii], SaveFrameCount / nsequences);
					SaveDataToImage(Fname, SubImage, swidth, sheight, nchannels);
					SaveFrameCount++;
				}
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
	if (ReadIntrinsicResults(Path, cameraInfo) != 0)
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
void VisualizeCleanMatches(char *Path, int view1, int view2, int timeID, double fractionMatchesDisplayed = 0.5)
{
	char Fname[200];
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
		sprintf(Fname, "%s/K%d.dat", Path, view1);
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
		sprintf(Fname, "%s/K%d.dat", Path, view2);
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
		sprintf(Fname, "%s/%d.png", Path, view1);
	else
		sprintf(Fname, "%s/%d/%d.png", Path, view1, timeID);
	IplImage *Img1 = cvLoadImage(Fname, nchannels == 3 ? 1 : 0);
	if (Img1->imageData == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return;
	}
	if (timeID < 0)
		sprintf(Fname, "%s/%d.png", Path, view2);
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

int main(int argc, char* argv[])
{

	char Path[] = "E:/Data/GoPro", Fname[200], Fname2[200];

	/*VideoData AllVideoData;
	ReadVideoData(Path, AllVideoData, 10, 0, 290);
	ExportCalibDatatoHanFormat(Path, AllVideoData, 10, 0, 290);
	return 0;*/

	if (argc == 1)
	{
		visualizationDriver(Path, 10, 0, 290, true, false, true, false);
		return 0;
	}
	int mode = atoi(argv[1]);

	srand((unsigned int)time(NULL));
	if (mode == 0)
	{
		visualizationDriver(Path, 480, 0, 0, true, true, true, true);
		return 0;

		int computeSync = 0;// atoi(argv[2]);
		if (computeSync == 1)
		{
			int srcID1 = atoi(argv[3]),
				srcID2 = atoi(argv[4]);
			double fps1 = atof(argv[5]),
				fps2 = atof(argv[6]),
				minZNCC = atof(argv[7]);

			int minSegLength = 3e6;
			double offset = 0.0;
			sprintf(Fname, "%s/%d/audio.wav", Path, srcID1);
			sprintf(Fname2, "%s/%d/audio.wav", Path, srcID2);
			if (SynAudio(Fname, Fname2, fps1, fps2, minSegLength, offset, minZNCC) != 0)
				printf("Not succeed\n");
			else
				printf("Succeed\n");
		}
		else
		{
			int writeSyncedImages = 1;// atoi(argv[3]);
			ShowSyncLoad(Path, writeSyncedImages);
		}
		return 0;
	}
	else if (mode == 1)
	{
		char *Fname = argv[2],
			*Fname2 = argv[3];
		int SaveFrameDif = atoi(argv[4]);//20;
		int nNonBlurIma = 0;
		PickStaticImagesFromVideo(Fname, Fname2, SaveFrameDif, 15, .3, 50, nNonBlurIma, true);
		BlurDetectionDriver(Fname, nNonBlurIma, 1920, 1080, 0.1);
	}
	else if (mode == 2)
	{
		int nviews = atoi(argv[2]),
			startF = atoi(argv[3]),
			stopF = atoi(argv[4]),
			HistogrameEqual = atoi(argv[5]);

		vector<int> availViews;
		for (int ii = 0; ii < nviews; ii++)
			availViews.push_back(ii);

		ExtractSiftGPUfromExtractedFrames(Path, availViews, startF, stopF, HistogrameEqual);
		return 0;
	}
	else if (mode == 3)
	{
		int nviews = atoi(argv[2]),
			NPplus = atoi(argv[3]),
			runMatching = atoi(argv[4]),
			distortionCorrected = atoi(argv[5]);

		if (runMatching == 1)
		{
			int HistogramEqual = atoi(argv[6]),
				OulierRemoveTestMethod = atoi(argv[7]);
			double ratioThresh = atof(argv[8]);

			int nCams = 1, cameraToScan = 0;
			Corpus corpusData;
			sprintf(Fname, "%s/BA_Camera_AllParams_after.txt", Path);
			if (!rewriteBundleAdjustedNVMResults(Path, Fname, corpusData))
				GeneratePointsCorrespondenceMatrix_SiftGPU2(Path, nviews, -1, HistogramEqual, ratioThresh, distortionCorrected, OulierRemoveTestMethod, nCams, cameraToScan);

			int timeID = -1, ninlierThesh = 50, LensType = -1;
#pragma omp parallel
			{
#pragma omp for nowait
				for (int jj = 0; jj < nviews - 1; jj++)
				{
					for (int ii = jj + 1; ii < nviews; ii++)
					{
						if (OulierRemoveTestMethod == 1)
							EssentialMatOutliersRemove(Path, timeID, jj, ii, nCams, cameraToScan, ninlierThesh, distortionCorrected, false);
						if (OulierRemoveTestMethod == 2)
							FundamentalMatOutliersRemove(Path, timeID, jj, ii, ninlierThesh, LensType, distortionCorrected, false, nCams, cameraToScan);
					}
				}
			}
			GenerateMatchingTable(Path, nviews, timeID);
		}
		else
		{
			BuildCorpus(Path, 1, 0, distortionCorrected, NPplus);
			visualizationDriver(Path, 1, -1, -1, true, false, false, false);
		}
		return 0;
	}
	else if (mode == 4)
	{
		int StartTime = atoi(argv[2]),
			StopTime = atoi(argv[3]),
			seletectedCam = atoi(argv[4]),
			runMatching = atoi(argv[5]),
			distortionCorrected = atoi(argv[6]), //must set to 1 when runMatching != 1;
			sharedIntriniscOptim = atoi(argv[7]);
		int nCams = 10,
			LensType = RADIAL_TANGENTIAL_PRISM;// RADIAL_TANGENTIAL_PRISM;// FISHEYE;

		sprintf(Fname, "%s/intrinsic_%d.txt", Path, seletectedCam);
		FILE*fp = fopen(Fname, "w+");	fclose(fp);
		if (runMatching >= 0)
			LocalizeCameraFromCorpusDriver(Path, StartTime, StopTime, runMatching, nCams, seletectedCam, distortionCorrected, sharedIntriniscOptim, LensType);
		if (runMatching != 1)
			visualizationDriver(Path, nCams, StartTime, StopTime, true, false, true, false);
		return 0;
	}
	else if (mode == 5)
	{
		int StartTime = atoi(argv[2]),
			StopTime = atoi(argv[3]),
			RunMatching = atoi(argv[4]);

		int HistogrameEqual = 0,
			distortionCorrected = 1,
			OulierRemoveTestMethod = 2, ninlierThesh = 40, //fmat test
			LensType = FISHEYE,
			nviews = 1;
		double ratioThresh = 0.4, reprojectionThreshold = 5;

		if (RunMatching == 1)
		{
			for (int timeID = StartTime; timeID <= StopTime; timeID++)
			{
				GeneratePointsCorrespondenceMatrix_SiftGPU2(Path, nviews, timeID, HistogrameEqual, 0.6, LensType, distortionCorrected, OulierRemoveTestMethod, nviews, 0);
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
			Build3DFromSyncedImages(Path, nviews, StartTime, StopTime, LensType, distortionCorrected, reprojectionThreshold, true, 1.0);

		return 0;
	}
	else if (mode == 6)
	{
		//google::InitGoogleLogging("SfM");
		//IncrementalBundleAdjustment(Path, nviews, -1, 20000);
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
	else if (mode == 7)
	{
		GenerateCorpusVisualWords("C:/temp/BOW", 24);
		ComputeWordsHistogram("C:/temp/BOW", 3);
	}
	else
	{
		//
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
		/*double K[] = { 5.6899782975011715e+002, 0, 6.3491168470047603e+002, 0, 5.7154638026417649e+002, 4.8530525211848123e+002, 0, 0, 1 };
		double Distortion[] = { 1.6037658238481549e-003, 6.3388270804725789e+002, 4.8009018607311287e+002 };
		int startFrame = 0, stopFrame = 38;

		LensCorrectionImageSequenceDriver("E:/Disney", K, Distortion, 1, startFrame, stopFrame, 1.0, 1.0, 5);
		return 0;*/

		TrajectoryData InfoTraj;
		/*CameraData CameraInfo[480];
		ReadDomeVGACalibFile(Path, CameraInfo);
		LoadTrackData(Path, 100, InfoTraj, true);
		//Write3DMemAtThatTime(Path, InfoTraj, CameraInfo, 100, 101);
		//return 0;

		vector<int> TrajectUsed;
		for (int ii = 0; ii < InfoTraj.nTrajectories; ii++)
		TrajectUsed.push_back(ii);
		double angleThreshold = 05;
		Genrate2DTrajectory(Path, 100, InfoTraj, CameraInfo, TrajectUsed, angleThreshold);
		return 0;*/
		int ntraj = 3409;
		Load2DTrajectory(Path, InfoTraj, ntraj);

		int minFrame = 60, maxFrame = 209;
		for (int kk = atoi(argv[2]); kk < atoi(argv[2]) + 1; kk++)
		{
			for (int jj = 1; jj <= 23; jj++)
			{
				int cameraPair[] = { kk * 24, kk * 24 + jj }, temporalOffsetRange[] = { -6, 6 };
				GetImagePatchIntensity(Path, InfoTraj, ntraj, minFrame, maxFrame, cameraPair, temporalOffsetRange);

				//vector<int> syncOff;
				//for (int ii = temporalOffsetRange[0]; ii <= temporalOffsetRange[1]; ii++)
				//	syncOff.push_back(ii);
				//ComputeErrorIntensity(Path, syncOff, cameraPair);
			}
		}
		return 0;

	}

	return 0;
}

