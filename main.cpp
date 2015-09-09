#include "DataStructure.h"
#include "ImagePro.h"
#include "VideoSequence.h"
#include "Ultility.h"
#include "Geometry.h"
#include "Visualization.h"


#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

using namespace std;
using namespace cv;
using namespace Eigen;

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
int ShowSyncLoad(char *DataPATH, char *SynFileName, char *SavePATH, int nsequences, double fps = -1)
{
	char Fname[2000];
	int WBlock = 1920, HBlock = 1080, nBlockX = 3, nchannels = 3, MaxFrames = 1000, playBackSpeed = 1, id;

	int *seqName = new int[nsequences];
	double offset, *Offset = new double[nsequences];
	Sequence *mySeq = new Sequence[nsequences];
	for (int ii = 0; ii < nsequences; ii++)
		seqName[ii] = ii;

	printf("Please input offset info in the format time-stamp format (f = f_ref - offset)!\n");
	sprintf(Fname, "%s/%s.txt", DataPATH, SynFileName);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot open %s\n", Fname);
		return 1;
	}
	while (fscanf(fp, "%d %lf ", &id, &offset) != EOF)
		Offset[id] = offset;
	fclose(fp);

	int refSeq = 0;
	double earliestTime = DBL_MAX;
	for (int ii = 0; ii < nsequences; ii++)
	{
		mySeq[ii].InitSeq(fps, Offset[ii]);
		if (earliestTime>Offset[ii])
			earliestTime = Offset[ii], refSeq = ii;
	}

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
	int *oFrameID = new int[nsequences + 1], *FrameID = new int[nsequences + 1];
	for (int ii = 0; ii < nsequences + 1; ii++)
		oFrameID[ii] = 0, FrameID[ii] = 0;
	cvNamedWindow("VideoSequences", CV_WINDOW_NORMAL);
	cvNamedWindow("Control", CV_WINDOW_NORMAL);
	cvCreateTrackbar("Speed", "Control", &playBackSpeed, 10, NULL);
	cvCreateTrackbar("Global frame", "Control", &FrameID[0], MaxFrames - 1, NULL);
	for (int ii = 0; ii < nsequences; ii++)
	{
		sprintf(Fname, "Seq %d", ii);
		cvCreateTrackbar(Fname, "Control", &FrameID[ii + 1], MaxFrames - 1, NULL);
		cvSetTrackbarPos(Fname, "Control", 0);
	}
	char* nameb1 = "Play/Stop";
	createButton(nameb1, AutomaticPlay, nameb1, CV_CHECKBOX, 0);
	char* nameb2 = "Not Save/Save";
	createButton(nameb2, AutomaticSave, nameb2, CV_CHECKBOX, 0);

	int BlockXID, BlockYID, setReferenceFrame, setSeqFrame, same, noUpdate, swidth, sheight;
	bool *GlobalSlider = new bool[nsequences]; //True: global slider, false: local slider
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

			sprintf(Fname, "Seq %d", ii);
			setSeqFrame = (int)((1.0*setReferenceFrame / mySeq[refSeq].TimeAlignPara[0] - mySeq[ii].TimeAlignPara[1]) * mySeq[ii].TimeAlignPara[0] + 0.5); //(refFrame/fps_ref - offset_i)*fps_i

			if (same == 2)
			{
				noUpdate++;
				if (autoplay)
				{
					sprintf(Fname, "Seq %d", ii);
					cvSetTrackbarPos(Fname, "Control", FrameID[ii + 1]);
					FrameID[ii + 1] += playBackSpeed;
				}
				continue;
			}

			if (setSeqFrame <= 0)
			{
				cvSetTrackbarPos(Fname, "Control", (int)(mySeq[ii].TimeAlignPara[1] + 0.5));
				Set_Sub_Mat(BlackImage, BigImg, nchannels*WBlock, HBlock, nchannels*width, nchannels*BlockXID*WBlock, BlockYID*HBlock);
			}
			else
			{
				printf("Sequence %d frame %d\n", ii, setSeqFrame);
				oFrameID[ii + 1] = FrameID[ii + 1];
				cvSetTrackbarPos(Fname, "Control", oFrameID[ii + 1]);
				sprintf(Fname, "%s/%d/%d.png", DataPATH, seqName[ii], setSeqFrame);
				if (GrabImageCVFormat(Fname, SubImage, swidth, sheight, nchannels))
					Set_Sub_Mat(SubImage, BigImg, nchannels*swidth, sheight, nchannels*width, nchannels*BlockXID*WBlock, BlockYID*HBlock);
				else
					Set_Sub_Mat(BlackImage, BigImg, nchannels*WBlock, HBlock, nchannels*width, nchannels*BlockXID*WBlock, BlockYID*HBlock);

				if (saveFrame)
				{
					sprintf(Fname, "%s/%d/%d.png", SavePATH, seqName[ii], SaveFrameCount / nsequences);
					SaveDataToImageCVFormat(Fname, SubImage, swidth, sheight, nchannels);
					SaveFrameCount++;
				}
				else
					SaveFrameCount = 0;
			}
			if (autoplay)
			{
				sprintf(Fname, "Seq %d", ii);
				cvSetTrackbarPos(Fname, "Control", FrameID[ii + 1]);
				FrameID[ii + 1] += playBackSpeed;
			}
		}
		oFrameID[0] = FrameID[0];
		if (noUpdate != nsequences)
			ShowDataToImage("VideoSequences", BigImg, width, height, nchannels, cvImg);

		if (autoplay)
		{
			int ii;
			for (ii = 0; ii < nsequences; ii++)
				if (!GlobalSlider[ii])
					break;
			if (ii == nsequences)
			{
				cvSetTrackbarPos("Global frame", "Control", FrameID[0]);
				FrameID[0] += playBackSpeed;
			}

			bool xxx = false;
			if (xxx)
			{
				char Fname[200];  sprintf(Fname, "C:/temp/%d.png", FrameID[0]);
				SaveDataToImage(Fname, BigImg, width, height, 3);
			}
		}
	}

	cvReleaseImage(&cvImg);
	delete[]seqName, delete[]Offset, delete[]oFrameID, delete[]FrameID, delete[]GlobalSlider;
	delete[]BigImg, delete[]BlackImage, delete[]SubImage;
	delete[]mySeq;

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
double TriangulatePointsFromCalibratedCameras(char *Path, int distortionCorrected, int maxPts, double threshold = 2.0)
{
	char Fname[200];
	Corpus corpusData;
	sprintf(Fname, "%s/BA_Camera_AllParams_after.txt", Path);
	if (!loadBundleAdjustedNVMResults(Fname, corpusData))
		return -1;

	int nviews = corpusData.nCameras;
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
		return -1;
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
	vector<Point3d>TwoPoints;

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
			putText(Img, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 1.0 * Img.cols / 640, CV_RGB(255, 0, 0), 1);
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
			if (maxPts == 2)
				TwoPoints.push_back(xyz);
		}
	}
	ofs.close();

	if (maxPts == 2)
		return Distance3D(TwoPoints[0], TwoPoints[1]);
	else
		return 0;
}
int TriangulatePointsFromCalibratedCameras2(char *Path, vector<int> SelectedCameras, int *DelayOffset, int startFrame, int stopFrame, int refFid = 50, int distortionCorrected = 0)
{
	char Fname[200];

	int nCams = (int)SelectedCameras.size();
	VideoData *VideoInfo = new VideoData[nCams];
	for (int camID = 0; camID < nCams; camID++)
	{
		if (ReadVideoDataI(Path, VideoInfo[camID], SelectedCameras[camID], startFrame, stopFrame) == 1)
			return 1;
	}

	Point3d xyz;
	vector<Point2d> uvAll[1], puvAll;
	double *A = new double[6 * (int)SelectedCameras.size()];
	double *B = new double[2 * (int)SelectedCameras.size()];
	double *Ps = new double[12 * (int)SelectedCameras.size()];

	int npts = 0;
	while (true)
	{
		uvAll[0].clear();
		bool breakflag = false;
		for (int ii = 0; ii < SelectedCameras.size(); ii++)
		{
			sprintf(Fname, "%s/%d/%d.png", Path, SelectedCameras[ii], refFid + DelayOffset[ii]);
			cvNamedWindow("Image", CV_WINDOW_NORMAL); setMouseCallback("Image", onMouse);
			Mat Img = imread(Fname);
			if (Img.empty())
			{
				printf("Cannot load %s\n", Fname);
				return 1;
			}

			CvPoint text_origin = { Img.cols / 30, Img.cols / 30 };
			sprintf(Fname, "Point %d/%d of (C, f): (%d, %d)", ii + 1, SelectedCameras.size(), SelectedCameras[ii], refFid + DelayOffset[ii]);
			putText(Img, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 0.5 * Img.cols / 640, CV_RGB(255, 0, 0), 3);
			imshow("Image", Img);
			if (waitKey(0) == 27)
			{
				breakflag = true;
				break;
			}
			uvAll[0].push_back(Point2d(MousePosX, MousePosY));
		}

		if (breakflag)
			break;

		for (int ii = 0; ii < SelectedCameras.size(); ii++)
		{
			if (VideoInfo[ii].VideoInfo[refFid + DelayOffset[ii]].LensModel == RADIAL_TANGENTIAL_PRISM)
				LensCorrectionPoint(&uvAll[0][ii], VideoInfo[ii].VideoInfo[refFid + DelayOffset[ii]].K, VideoInfo[ii].VideoInfo[refFid + DelayOffset[ii]].distortion);
			else
				FishEyeCorrectionPoint(&uvAll[0][ii], VideoInfo[ii].VideoInfo[refFid + DelayOffset[ii]].distortion[0], VideoInfo[ii].VideoInfo[refFid + DelayOffset[ii]].distortion[1], VideoInfo[ii].VideoInfo[refFid + DelayOffset[ii]].distortion[2]);

			if (VideoInfo[ii].VideoInfo[refFid + DelayOffset[ii]].ShutterModel == 0)
			{
				for (int kk = 0; kk < 12; kk++)
					Ps[12 * ii + kk] = VideoInfo[ii].VideoInfo[refFid + DelayOffset[ii]].P[kk];
			}
			else if (VideoInfo[ii].VideoInfo[refFid + DelayOffset[ii]].ShutterModel == 1)
			{
				double *K = VideoInfo[ii].VideoInfo[refFid + DelayOffset[ii]].K;
				double ycn = (uvAll[0][ii].y - K[5]) / K[4];
				double xcn = (uvAll[0][ii].x - K[2] - K[1] * ycn) / K[0];

				double *wt = VideoInfo[ii].VideoInfo[refFid + DelayOffset[ii]].wt;
				double *Rcenter = VideoInfo[ii].VideoInfo[refFid + DelayOffset[ii]].R;
				double *Tcenter = VideoInfo[ii].VideoInfo[refFid + DelayOffset[ii]].T;

				double wx = ycn*wt[0], wy = ycn*wt[1], wz = ycn*wt[2];
				double wx2 = wx*wx, wy2 = wy*wy, wz2 = wz*wz, wxz = wx*wz, wxy = wx*wy, wyz = wy*wz;
				double denum = 1.0 + wx2 + wy2 + wz2;

				double Rw[9] = { 1.0 + wx2 - wy2 - wz2, 2.0 * wxy - 2.0 * wz, 2.0 * wy + 2.0 * wxz,
					2.0 * wz + 2.0 * wxy, 1.0 - wx2 + wy2 - wz2, 2.0 * wyz - 2.0 * wx,
					2.0 * wxz - 2.0 * wy, 2.0 * wx + 2.0 * wyz, 1.0 - wx2 - wy2 + wz2 };

				for (int jj = 0; jj < 9; jj++)
					Rw[jj] = Rw[jj] / denum;

				double R[9];  mat_mul(Rw, Rcenter, R, 3, 3, 3);
				double T[3] = { Tcenter[0] + ycn*wt[3], Tcenter[1] + ycn*wt[4], Tcenter[2] + ycn*wt[5] };

				AssembleP(K, R, T, Ps + 12 * ii);
			}
			else
				printf("Not supported model for motion prior sync\n");
		}

		NviewTriangulation(uvAll, Ps, &xyz, (int)SelectedCameras.size(), 1, NULL, A, B);

		puvAll = uvAll[0];
		ProjectandDistort(xyz, puvAll, Ps, NULL, NULL);

		double finalerror = 0.0;
		for (int ll = 0; ll < (int)SelectedCameras.size(); ll++)
			finalerror += pow(puvAll[ll].x - uvAll[0][ll].x, 2) + pow(puvAll[ll].y - uvAll[0][ll].y, 2);
		finalerror = sqrt(finalerror / (int)SelectedCameras.size());
		printf("3D: %f %f %f Error: %f\n", xyz.x, xyz.y, xyz.z, finalerror);
	}

	return 0;
}
void VisualizeCleanMatches(char *Path, int view1, int view2, int timeID, double fractionMatchesDisplayed = 0.5, int FrameOffset1 = 0, int FrameOffset2 = 0)
{
	char Fname[200];
	if (timeID < 0)
		sprintf(Fname, "%s/Corpus/M_%d_%d.txt", Path, view1, view2);
	else
		sprintf(Fname, "%s/Dynamic/M%d_%d_%d.txt", Path, timeID, view1, view2);

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
		sprintf(Fname, "%s/%d/K%d.dat", Path, view1, timeID + FrameOffset1);

	bool readsucces = false;
	vector<KeyPoint> keypoints1; keypoints1.reserve(MaxNFeatures);
	readsucces = ReadKPointsBinarySIFT(Fname, keypoints1);
	if (!readsucces)
	{
		printf("%s does not have SIFT points. Please precompute it!\n", Fname);
		return;
	}

	if (timeID < 0)
		sprintf(Fname, "%s/Corpus/K%d.dat", Path, view2);
	else
		sprintf(Fname, "%s/%d/K%d.dat", Path, view2, timeID + FrameOffset2);
	vector<KeyPoint> keypoints2; keypoints2.reserve(MaxNFeatures);
	readsucces = ReadKPointsBinarySIFT(Fname, keypoints2);
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
		sprintf(Fname, "%s/%d/%d.png", Path, view1, timeID + FrameOffset1);
	IplImage *Img1 = cvLoadImage(Fname, nchannels == 3 ? 1 : 0);
	if (Img1->imageData == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return;
	}
	if (timeID < 0)
		sprintf(Fname, "%s/Corpus/%d.png", Path, view2);
	else
		sprintf(Fname, "%s/%d/%d.png", Path, view2, timeID + FrameOffset2);
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
int VisualizePnPMatches(char *Path, int cameraID, int timeID)
{
	char Fname[200];
	Corpus corpusData;
	sprintf(Fname, "%s/Corpus", Path);	ReadCorpusInfo(Fname, corpusData, false, false);

	int npts, threeDid, ptsCount = 0;
	double u, v, scale;
	sprintf(Fname, "%s/%d/Inliers_3D2D_%d.txt", Path, cameraID, timeID); FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return -1;
	}
	fscanf(fp, "%d ", &npts);
	vector<int> threeDidVec;
	Point2d *pts = new Point2d[npts];
	Point3d *t3D = new Point3d[npts];
	while (fscanf(fp, "%d %lf %lf ", &threeDid, &u, &v, &scale) != EOF)
	{
		threeDidVec.push_back(threeDid);
		pts[ptsCount].x = u, pts[ptsCount].y = v;
		ptsCount++;
	}
	fclose(fp);



	printf("Reading corpus images....\n");
	Mat Img;
	Mat *CorpusImg = new Mat[corpusData.nCameras];
	for (int ii = 0; ii < corpusData.nCameras; ii++)
	{
		sprintf(Fname, "%s/Corpus/%d.png", Path, ii);	 CorpusImg[ii] = imread(Fname);
		if (CorpusImg[ii].empty())
		{
			printf("Cannot load %s\n", Fname);
			return 1;
		}
	}
	sprintf(Fname, "%s/%d/%d.png", Path, cameraID, timeID);	Img = imread(Fname);
	if (Img.empty())
	{
		printf("Cannot load %s\n", Fname);
		return 1;
	}

	static CvScalar colors[] = { { { 0, 0, 255 } } };
	namedWindow("PnPTest", CV_WINDOW_NORMAL);
	cv::Mat BImage(Img.rows, Img.cols * 2, Img.type());
	for (int ii = 0; ii < threeDidVec.size(); ii++)
	{
		threeDid = threeDidVec[ii];
		int viewID = corpusData.viewIdAll3D[threeDid][0];
		Point2d uv = corpusData.uvAll3D[threeDid][0];

		Img.copyTo(BImage(cv::Rect(0, 0, Img.cols, Img.rows)));
		CorpusImg[viewID].copyTo(BImage(cv::Rect(Img.cols, 0, Img.cols, Img.rows)));

		circle(BImage, cv::Point(pts[ii].x, pts[ii].y), 3, colors[0], 8);
		circle(BImage, cv::Point(uv.x + Img.cols, uv.y), 3, colors[0], 8);

		imshow("PnPTest", BImage);
		waitKey(-1);
	}

	return 0;
}
void ConvertImagetoHanFormat(char *Path, int nviews, int beginTime, int endTime)
{
	char Fname[200];
	for (int ii = beginTime; ii <= endTime; ii++)
		sprintf(Fname, "%s/In/%08d", Path, ii), makeDir(Fname);

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
	sprintf(Fname, "%s/NPlusViewer/", Path), makeDir(Fname);
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
				fprintf(fp, "%.4f %.4f ", ListNPlusViewerPoints[ii][kk + jj*nviewers].x, ListNPlusViewerPoints[ii][kk + jj*nviewers].y);
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

		sprintf(Fname, "%s/Track", Path), makeDir(Fname);
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
					fprintf(fp2, "%.4f %.4f ", Tracks[ii*nviewers + jj][kk].x, Tracks[ii*nviewers + jj][kk].y);
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
	int nframes = max(MaxnFrames, stopTime);

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
	sprintf(Fname, "%s/Traject3D", Path), makeDir(Fname);
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
			fprintf(fp, "%.4f %.4f %.4f ", traject3D[kk][jj].WC.x, traject3D[kk][jj].WC.y, traject3D[kk][jj].WC.z);
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
		sprintf(Fname, "%s/Track", Path), makeDir(Fname);
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

	int nTimeInstrances;
	double x, y, z, mX = 0.0, mY = 0.0, mZ = 0.0;
	vector<Point3d> XYZ;
	sprintf(Fname, "C:/temp/Sim/i%d.txt", trackID); FILE *fp = fopen(Fname, "r");
	//fscanf(fp, "%s %d %d %d", Fname, &nTimeInstrances, &nTimeInstrances, &nTimeInstrances);
	while (fscanf(fp, "%lf %lf %lf", &x, &y, &z) != EOF)
	{
		XYZ.push_back(Point3d(x, y, z));
		mX += x, mY += y, mZ += z;
	}
	fclose(fp);
	nTimeInstrances = XYZ.size();

	mX = mX / nTimeInstrances, mY = mY / nTimeInstrances, mZ = mZ / nTimeInstrances;
	for (int jj = 0; jj < nTimeInstrances; jj++)
	{
		XYZ[jj].x = 1.0*(XYZ[jj].x - mX) + mX;
		XYZ[jj].y = 1.0*(XYZ[jj].y - mY) + mY;
		XYZ[jj].z = 1.0*(XYZ[jj].z - mZ) + mZ;
	}

	VideoData AllVideoInfo;
	if (ReadVideoData(Path, AllVideoInfo, nviews, startTime, stopTime) == 1)
		return 1;
	int nframes = max(MaxnFrames, stopTime);

	double P[12];
	Point2d pts;
	sprintf(Fname, "C:/temp/Sim/iTrack_%d.txt", trackID); fp = fopen(Fname, "w+");
	/*fprintf(fp, "%d ", nviews);
	for (int kk = 0; kk < nviews; kk++)
	fprintf(fp, "%d ", kk);
	fprintf(fp, "\n1\n");*/
	fprintf(fp, "%d %d \n", timeID, nTimeInstrances);
	for (int kk = 0; kk < nviews; kk++)
	{
		//fprintf(fp, "%d %d %d \n", kk, timeID, nTimeInstrances);
		int videoID = kk*nframes;
		for (int jj = 0; jj < nTimeInstrances; jj++)
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
	int nframes = max(MaxnFrames, stopTime);

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
	int nframes = max(MaxnFrames, stopTime);

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
			for (frameID = 0; frameID < CamTraj[camID].Track3DInfo[trackID].nf; frameID++)
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
			for (frameID = 0; frameID < CamTraj[camID].Track3DInfo[trackID].nf; frameID++)
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
			fprintf(fp, "%.4f %.4f %.4f ", XYZ[index[ii]].x, XYZ[index[ii]].y, XYZ[index[ii]].z);
		fprintf(fp, "\n");
	}
	fclose(fp);

	delete[]index, delete[]CapturedTime, delete[]XYZ;

	return 0;
}
int TestFrameSyncTriangulation(char *Path)
{
	int nCams = 3, npts = 1, width = 1920, height = 1080, rate = 10;
	double Intrinsic[5] = { 1500, 1500, 0, 960, 540 }, distortion[7] = { 0, 0, 0, 0, 0, 0, 0 }, radius = 3000;
	vector<int> SelectedCams, UnSyncFrameTimeStamp;
	//SimulateCamerasAnd2DPointsForMoCap(Path, nCams, npts, Intrinsic, distortion, width, height, radius, true, false, true, rate, UnSyncFrameTimeStamp, SyncFrameOffset);

	vector<int> SyncFrameOffset;
	for (int ii = 0; ii < (int)SelectedCams.size(); ii++)
		SyncFrameOffset.push_back(-UnSyncFrameTimeStamp[ii] / 10);

	//double OffsetInfo[3];
	//LeastActionSyncBruteForce2DTriplet(Path, SelectedCams, 0, 53, 1, OffsetInfo);
	TriangulateFrameSync2DTrajectories(Path, SelectedCams, SyncFrameOffset, 0, 54, npts);
	visualizationDriver(Path, 3, 0, 400, true, false, true, false, true, false, 0);
	return 0;
}

void CheckerBoardFormFrameSynceCorrespondingInfo4BA(char *Path, vector <vector<int> > &viewIdAll3D, vector<vector<Point2d> > &uvAll3D, int camID, int nframes, int npts, int *Offset)
{
	char Fname[200];
	int id, frameID, nf;

	double u, v;
	ImgPtEle ptEle;
	vector<ImgPtEle> *PerCam_UV = new vector<ImgPtEle>[npts];

	//Get 2D info
	for (int pid = 0; pid < npts; pid++)
		PerCam_UV[pid].reserve(nframes);

	sprintf(Fname, "%s/Track2D/%d.txt", Path, camID); FILE *fp = fopen(Fname, "r");
	for (int pid = 0; pid < npts; pid++)
	{
		fscanf(fp, "%d %d ", &id, &nf);
		if (id != pid)
			printf("Problem at Point %d of Cam %d", id, camID);
		for (int pid = 0; pid < nf; pid++)
		{
			fscanf(fp, "%d %lf %lf ", &frameID, &u, &v);
			if (frameID < 0 || frameID>nframes)
				continue;

			ptEle.pt2D.x = u, ptEle.pt2D.y = v, ptEle.frameID = frameID;
			PerCam_UV[id].push_back(ptEle);
		}
	}
	fclose(fp);

	for (int fid = 0; fid < PerCam_UV[0].size(); fid++)
	{
		int syncFid = PerCam_UV[0][fid].frameID + Offset[camID];
		if (syncFid<0 || syncFid>nframes)
			continue;

		for (int ii = 0; ii < npts; ii++)
		{
			viewIdAll3D[syncFid + ii*nframes].push_back(camID);
			uvAll3D[syncFid + ii*nframes].push_back(PerCam_UV[ii][fid].pt2D);
		}
	}

	delete[]PerCam_UV;
	return;
}
void CheckerBoardAssemble2D_2DCorresInfo(int *SelectedPair, int nframes, int npts, vector <vector<int> > &viewIdAll3D, vector<vector<Point2d> > &uvAll3D, vector<Point2d> &pts1, vector<Point2d> &pts2, vector<int> &takenFrame)
{
	int count, Found2Views[2];
	for (int fid = 0; fid < nframes; fid++)
	{
		for (int pid = 0; pid < npts; pid++)
		{
			count = 0;
			for (int vid = 0; vid < viewIdAll3D[fid + pid*nframes].size(); vid++)
			{
				if (viewIdAll3D[fid + pid*nframes][vid] == SelectedPair[0])
					Found2Views[0] = vid, count++;
				else if (viewIdAll3D[fid + pid*nframes][vid] == SelectedPair[1])
					Found2Views[1] = vid, count++;

				if (count == 2)
					break;
			}

			if (count == 2)
			{
				pts2.push_back(uvAll3D[fid + pid*nframes][Found2Views[0]]);
				pts1.push_back(uvAll3D[fid + pid*nframes][Found2Views[1]]);
				takenFrame.push_back(fid + pid*nframes);
			}
		}
	}

	return;
}
void CheckerBoardAssemble2D_3DCorresInfo(char *Path, vector <vector<int> > &viewIdAll3D, vector<vector<Point2d> > &uvAll3D, vector<Point3d> AllP3D, vector<Point2d> &p2d, vector<Point3d> &p3d, int camID, int nframes, int npts, int *Offset)
{
	char Fname[200];
	int id, frameID, nf;

	double u, v;
	ImgPtEle ptEle;
	vector<ImgPtEle> *PerCam_UV = new vector<ImgPtEle>[npts];

	//Get 2D info
	for (int pid = 0; pid < npts; pid++)
		PerCam_UV[pid].reserve(nframes);

	sprintf(Fname, "%s/Track2D/%d.txt", Path, camID); FILE *fp = fopen(Fname, "r");
	for (int pid = 0; pid < npts; pid++)
	{
		fscanf(fp, "%d %d ", &id, &nf);
		if (id != pid)
			printf("Problem at Point %d of Cam %d", id, camID);
		for (int pid = 0; pid < nf; pid++)
		{
			fscanf(fp, "%d %lf %lf ", &frameID, &u, &v);
			if (frameID < 0 || frameID>nframes)
				continue;

			ptEle.pt2D.x = u, ptEle.pt2D.y = v, ptEle.frameID = frameID;
			PerCam_UV[id].push_back(ptEle);
		}
	}
	fclose(fp);

	for (int fid = 0; fid < PerCam_UV[0].size(); fid++)
	{
		int syncFid = PerCam_UV[0][fid].frameID + Offset[camID];
		if (syncFid<0 || syncFid>nframes)
			continue;

		for (int ii = 0; ii < npts; ii++)
		{
			if (viewIdAll3D[syncFid + ii*nframes].size() > 1) //3D has been reconstructed
			{
				p3d.push_back(AllP3D[syncFid + ii*nframes]);
				p2d.push_back(PerCam_UV[ii][fid].pt2D);
			}
			viewIdAll3D[syncFid + ii*nframes].push_back(camID);
			uvAll3D[syncFid + ii*nframes].push_back(PerCam_UV[ii][fid].pt2D);
		}
	}

	delete[]PerCam_UV;
	return;
}
int CheckerBoardMultiviewSpatialTemporalCalibration2(char *Path, int nCams, int nframes)
{
	char Fname[1024]; FILE *fp = 0;
	const double square_size = 50.8;
	const int width = 1920, height = 1080, bh = 8, bw = 11, npts = bh*bw, TemporalSearchRange = 30, LossType = 0; //Huber loss
	bool fixIntrinsic = true, fixDistortion = true, fixPose = false, fixfirstCamPose = true, distortionCorrected = false;

	//Single Camera calibration
	/*int sampleCalibFrameStep = 1;
	if (nframes > 100)
	sampleCalibFrameStep = nframes / 50;

	omp_set_num_threads(omp_get_max_threads());
	#pragma omp parallel for
	for (int camID = 0; camID < nCams; camID++)
	SingleCameraCalibration(Path, camID, nframes, bw, bh, true, sampleCalibFrameStep, square_size, 1, width, height);


	//Create 2D trajectories
	for (int camID = 0; camID < nCams; camID++)
	CleanCheckBoardDetection(Path, camID, 0, nframes);

	//Brute force Fmat sync
	GeometricConstraintSyncDriver(Path, nCams, npts, 0, 0, nframes, TemporalSearchRange, false);*/

	//Register all cameras into a common coordinate base on temporal sync results: START
	VideoData *FrameInfo = new VideoData[nCams];
	for (int camID = 0; camID < nCams; camID++)
		ReadVideoDataI(Path, FrameInfo[camID], camID, 0, nframes);

	//Compute pairwise transformation base on temporal sync results
	int FrameSync[9];
	float dummy;
	sprintf(Fname, "%s/FGeoSync.txt", Path); fp = fopen(Fname, "r");
	for (int ii = 0; ii < nCams; ii++)
	{
		fscanf(fp, "%f %f ", &dummy, &dummy);
		FrameSync[ii] = -dummy;
	}
	fclose(fp);

	//Copy camera info of the 1st valid video frame. Assume all cameras are stationary during the calibration
	CameraData StationaryCamInfo[MaxnCams];
	for (int camID = 0; camID < nCams; camID++)
	{
		for (int ii = 0; ii < nframes; ii++)
		{
			if (FrameInfo[camID].VideoInfo[ii].valid)
			{
				CopyCamereInfo(FrameInfo[camID].VideoInfo[ii], StationaryCamInfo[camID], false);
				StationaryCamInfo[camID].threshold = 1000000.0; //make sure that all points are inliers
				break;
			}
		}
	}

	/*//Transform pose of all other camers to camera 1: R_w1 = R_i1*R_wi
	for (int camID = 1; camID < nCams; camID++)
	{
	double Ri1[9], RTi1[9], RT_w1[16], RT_wi[16], RT_i1[16];
	for (int fid = 0; fid < nframes; fid++)
	{
	if (!FrameInfo[camID].VideoInfo[fid].valid)
	continue;
	ceres::AngleAxisToRotationMatrix(allri1 + 3 * (camID - 1), RTi1);
	mat_transpose(RTi1, Ri1, 3, 3);

	RT_i1[0] = Ri1[0], RT_i1[0] = Ri1[1], RT_i1[0] = Ri1[2], RT_i1[0] = allTi1[3 * (camID - 1)],
	RT_i1[0] = Ri1[3], RT_i1[0] = Ri1[4], RT_i1[0] = Ri1[5], RT_i1[0] = allTi1[3 * (camID - 1) + 1],
	RT_i1[0] = Ri1[6], RT_i1[0] = Ri1[7], RT_i1[0] = Ri1[8], RT_i1[0] = allTi1[3 * (camID - 1) + 2],
	RT_i1[0] = 0, RT_i1[0] = 0, RT_i1[0] = 0, RT_i1[0] = 1;

	RT_wi[0] = FrameInfo[camID].VideoInfo[fid].R[0], RT_wi[0] = FrameInfo[camID].VideoInfo[fid].R[1], RT_wi[0] = FrameInfo[camID].VideoInfo[fid].R[2], RT_wi[0] = FrameInfo[camID].VideoInfo[fid].T[0],
	RT_wi[0] = FrameInfo[camID].VideoInfo[fid].R[3], RT_wi[0] = FrameInfo[camID].VideoInfo[fid].R[4], RT_wi[0] = FrameInfo[camID].VideoInfo[fid].R[5], RT_wi[0] = FrameInfo[camID].VideoInfo[fid].T[1],
	RT_wi[0] = FrameInfo[camID].VideoInfo[fid].R[6], RT_wi[0] = FrameInfo[camID].VideoInfo[fid].R[7], RT_wi[0] = FrameInfo[camID].VideoInfo[fid].R[8], RT_wi[0] = FrameInfo[camID].VideoInfo[fid].T[2],
	RT_wi[0] = 0, RT_wi[0] = 0, RT_wi[0] = 0, RT_wi[0] = 1;

	mat_mul(RT_i1, RT_wi, RT_w1, 4, 4, 4);

	FrameInfo[camID].VideoInfo[fid].R[0] = RT_w1[0], FrameInfo[camID].VideoInfo[fid].R[1] = RT_w1[1], FrameInfo[camID].VideoInfo[fid].R[2] = RT_w1[2], FrameInfo[camID].VideoInfo[fid].T[0] = RT_w1[3],
	FrameInfo[camID].VideoInfo[fid].R[3] = RT_w1[4], FrameInfo[camID].VideoInfo[fid].R[4] = RT_w1[5], FrameInfo[camID].VideoInfo[fid].R[5] = RT_w1[6], FrameInfo[camID].VideoInfo[fid].T[0] = RT_w1[7],
	FrameInfo[camID].VideoInfo[fid].R[6] = RT_w1[8], FrameInfo[camID].VideoInfo[fid].R[7] = RT_w1[9], FrameInfo[camID].VideoInfo[fid].R[8] = RT_w1[10], FrameInfo[camID].VideoInfo[fid].T[0] = RT_w1[11];
	}
	}

	for (int camID = 0; camID < nCams; camID++)
	{
	//Invert "camea pose wrst to checkerboard coordinate"
	for (int fid = 0; fid < nframes; fid++)
	if (FrameInfo[camID].VideoInfo[fid].valid)
	InvertCameraPose(FrameInfo[camID].VideoInfo[fid].R, FrameInfo[camID].VideoInfo[fid].T, FrameInfo[camID].VideoInfo[fid].R, FrameInfo[camID].VideoInfo[fid].T);

	WriteVideoDataI(Path, FrameInfo[camID], camID, 0, nframes);
	}*/

	//Setup 2d point correspondences for all views according to the frame-level sync result
	int ninlers;
	vector<Point3d> P3D(nframes*npts);
	vector <vector<int> > viewIdAll3D(nframes*npts);
	vector<vector<Point2d> > uvAll3D(nframes*npts);
	vector<vector<double> > scaleAll3D(nframes*npts);
	vector<int>TakenFrame, sharedCam;
	vector<bool>GoodPoints;
	vector < Point3d> p3d;
	vector<Point2d> pts1, pts2;
	vector<double> featureScale;

	//Inital SfM with two views
	vector<int> AvailableViews;

	//Use pose from single cam calib to estimate inter-pose of the corresponding synced frame in the other cam
	double allri1[3 * (9 - 1)], allTi1[3 * (9 - 1)];
	for (int camID = 1; camID < nCams; camID++)
	{
		vector<Point3d> rvec, tvec;
		double R21[9], r21[3], T21[3];
		for (int fid1 = 0; fid1 < nframes; fid1++)
		{
			int fid2 = fid1 + FrameSync[0] - FrameSync[camID];
			if (fid1 <0 || fid1 >nframes || fid2<0 || fid2>nframes)
				continue;
			if (!FrameInfo[0].VideoInfo[fid1].valid || !FrameInfo[camID].VideoInfo[fid2].valid)
				continue;

			ComputeInterCamerasPose(FrameInfo[0].VideoInfo[fid1].R, FrameInfo[0].VideoInfo[fid1].T, FrameInfo[camID].VideoInfo[fid2].R, FrameInfo[camID].VideoInfo[fid2].T, R21, T21);
			ceres::RotationMatrixToAngleAxis(R21, r21);

			rvec.push_back(Point3d(r21[0], r21[1], r21[2]));
			tvec.push_back(Point3d(T21[0], T21[1], T21[2]));
		}

		//average the relative pose
		r21[0] = 0, r21[1] = 0, r21[2] = 0, T21[0] = 0, T21[1] = 0, T21[2] = 0;
		int nposes = (int)rvec.size();
		for (int ii = 0; ii < nposes; ii++)
		{
			r21[0] += rvec[ii].x, r21[1] += rvec[ii].y, r21[2] += rvec[ii].z;
			T21[0] += tvec[ii].x, T21[1] += tvec[ii].y, T21[2] += tvec[ii].z;
		}
		allri1[3 * (camID - 1)] = r21[0] / nposes, allri1[3 * (camID - 1) + 1] = r21[1] / nposes, allri1[3 * (camID - 1) + 2] = r21[2] / nposes;
		allTi1[3 * (camID - 1)] = T21[0] / nposes, allTi1[3 * (camID - 1) + 1] = T21[1] / nposes, allTi1[3 * (camID - 1) + 2] = T21[2] / nposes;
	}

	int Mode = 0; //0: optimize all views directly from the relative poses, 1: incremental sfm PNP
	if (Mode == 0)
	{
		for (int jj = 0; jj < 3; jj++)
			StationaryCamInfo[0].rt[jj] = 0, StationaryCamInfo[0].rt[jj + 3] = 0;
		for (int ii = 1; ii < nCams; ii++)
			for (int jj = 0; jj < 3; jj++)
				StationaryCamInfo[ii].rt[jj] = allri1[3 * (ii - 1) + jj], StationaryCamInfo[ii].rt[jj + 3] = allTi1[3 * (ii - 1) + jj];
		for (int ii = 0; ii < nCams; ii++)
			GetRTFromrt(StationaryCamInfo[ii]), GetRTFromrt(StationaryCamInfo[ii]);

		for (int ii = 0; ii < nCams; ii++)
		{
			AvailableViews.push_back(ii);
			CheckerBoardFormFrameSynceCorrespondingInfo4BA(Path, viewIdAll3D, uvAll3D, AvailableViews[ii], nframes, npts, FrameSync);
		}

		for (int ii = 0; ii < AvailableViews.size(); ii++)
			AssembleP(StationaryCamInfo[AvailableViews[ii]]);

		NviewTriangulation(StationaryCamInfo, AvailableViews.size(), viewIdAll3D, uvAll3D, P3D);
		for (int ii = 0; ii < (int)uvAll3D.size(); ii++)
		{
			featureScale.clear();
			for (int jj = 0; jj < (int)uvAll3D[ii].size(); jj++)
				featureScale.push_back(1.0);
			scaleAll3D.push_back(featureScale);
		}
		GlobalShutterBundleAdjustment(Path, StationaryCamInfo, P3D, viewIdAll3D, uvAll3D, scaleAll3D, sharedCam, (int)AvailableViews.size(), fixIntrinsic, fixDistortion, fixPose, fixfirstCamPose, distortionCorrected, LossType, false);
	}
	else if (Mode == 1)
	{
		AvailableViews.push_back(0), AvailableViews.push_back(1);
		for (int ii = 0; ii < 2; ii++)
			CheckerBoardFormFrameSynceCorrespondingInfo4BA(Path, viewIdAll3D, uvAll3D, AvailableViews[ii], nframes, npts, FrameSync);

		if (0) //two view recon gives very bad results
		{
			int SelectedPair[2] = { AvailableViews[0], AvailableViews[1] };
			CheckerBoardAssemble2D_2DCorresInfo(SelectedPair, nframes, npts, viewIdAll3D, uvAll3D, pts1, pts2, TakenFrame);
			TwoViewsClean3DReconstructionFmat(StationaryCamInfo[AvailableViews[0]], StationaryCamInfo[AvailableViews[1]], pts1, pts2, p3d);
		}
		else //hack
		{
			for (int jj = 0; jj < 3; jj++)
				StationaryCamInfo[0].rt[jj] = 0, StationaryCamInfo[0].rt[jj + 3] = 0;
			GetRTFromrt(StationaryCamInfo[AvailableViews[0]]);

			for (int ii = 1; ii < 4; ii++)
				for (int jj = 0; jj < 3; jj++)
					StationaryCamInfo[ii].rt[jj] = allri1[3 * (ii - 1) + jj], StationaryCamInfo[ii].rt[jj + 3] = allTi1[3 * (ii - 1) + jj];
			GetRTFromrt(StationaryCamInfo[AvailableViews[1]]);
		}

		for (int ii = 0; ii < AvailableViews.size(); ii++)
			AssembleP(StationaryCamInfo[AvailableViews[ii]]);

		NviewTriangulation(StationaryCamInfo, AvailableViews.size(), viewIdAll3D, uvAll3D, P3D);
		for (int ii = 0; ii < (int)uvAll3D.size(); ii++)
		{
			featureScale.clear();
			for (int jj = 0; jj < (int)uvAll3D[ii].size(); jj++)
				featureScale.push_back(1.0);
			scaleAll3D.push_back(featureScale);
		}
		GlobalShutterBundleAdjustment(Path, StationaryCamInfo, P3D, viewIdAll3D, uvAll3D, scaleAll3D, sharedCam, (int)AvailableViews.size(), true, true, false, true, false, LossType, false);

		for (int addingCamID = 2; addingCamID < nCams; addingCamID++)
		{
			pts1.clear(), p3d.clear(), featureScale.clear();
			AvailableViews.push_back(addingCamID);

			CheckerBoardAssemble2D_3DCorresInfo(Path, viewIdAll3D, uvAll3D, P3D, pts1, p3d, addingCamID, nframes, npts, FrameSync);
			for (int ii = 0; ii < (int)pts1.size(); ii++)
				featureScale.push_back(1.0);

			DetermineDevicePose(StationaryCamInfo[addingCamID].K, StationaryCamInfo[addingCamID].distortion, StationaryCamInfo[addingCamID].LensModel, StationaryCamInfo[addingCamID].R, StationaryCamInfo[addingCamID].T, pts1, p3d, 0, 1000000.0, ninlers, true);
			GetrtFromRT(StationaryCamInfo[addingCamID]);

			fixIntrinsic = true, fixDistortion = true, distortionCorrected = false;
			if (CameraPose_GSBA(Path, StationaryCamInfo[addingCamID], p3d, pts1, featureScale, GoodPoints, fixIntrinsic, fixDistortion, distortionCorrected, false) == 1)
				printf("Something seriously wrong happend when adding new cameras\n"), abort();

			for (int ii = 0; ii < AvailableViews.size(); ii++)
				AssembleP(StationaryCamInfo[AvailableViews[ii]]);

			NviewTriangulation(StationaryCamInfo, AvailableViews.size(), viewIdAll3D, uvAll3D, P3D);
			for (int ii = 0; ii < (int)uvAll3D.size(); ii++)
			{
				featureScale.clear();
				for (int jj = 0; jj < (int)uvAll3D[ii].size(); jj++)
					featureScale.push_back(1.0);
				scaleAll3D.push_back(featureScale);
			}
			GlobalShutterBundleAdjustment(Path, StationaryCamInfo, P3D, viewIdAll3D, uvAll3D, scaleAll3D, sharedCam, (int)AvailableViews.size(), fixIntrinsic, fixDistortion, fixPose, fixfirstCamPose, distortionCorrected, LossType, false);
		}

		//Scale the 3D to physical unit of mm
		double estimatedDistance = Distance3D(P3D[0], P3D[(bh - 1)*nframes]); //Matlab detection
		double scale = square_size*(bh - 1) / estimatedDistance;
		for (int ii = 1; ii < nCams; ii++)
		{
			StationaryCamInfo[ii].rt[3] *= scale, StationaryCamInfo[ii].rt[4] *= scale, StationaryCamInfo[ii].rt[5] *= scale;
			GetRTFromrt(StationaryCamInfo[ii]);
		}

		for (int ii = 0; ii < AvailableViews.size(); ii++)
			AssembleP(StationaryCamInfo[AvailableViews[ii]]);

		NviewTriangulation(StationaryCamInfo, AvailableViews.size(), viewIdAll3D, uvAll3D, P3D);
	}

	//Save Data
	sprintf(Fname, "%s/3dGL.xyz", Path);  fp = fopen(Fname, "w+");
	for (int ii = 0; ii < P3D.size(); ii++)
		if (Distance3D(P3D[ii], Point3d(0, 0, 0))>0.00001)
			fprintf(fp, "%.3f %.3f %.3f\n", P3D[ii].x, P3D[ii].y, P3D[ii].z);
	fclose(fp);

	sprintf(Fname, "%s/Intrinsic.txt", Path); FILE *fp1 = fopen(Fname, "w+");
	sprintf(Fname, "%s/CamPose.txt", Path); FILE *fp2 = fopen(Fname, "w+");
	for (int ii = 0; ii < (int)AvailableViews.size(); ii++)
	{
		int viewID = AvailableViews[ii];
		GetRTFromrt(StationaryCamInfo[viewID]), GetRCGL(StationaryCamInfo[viewID]);

		fprintf(fp1, "%d %d %d %d %.8f %.8f %.8f %.8f %.8f  ", ii, StationaryCamInfo[viewID].LensModel, StationaryCamInfo[viewID].width, StationaryCamInfo[viewID].height,
			StationaryCamInfo[viewID].K[0], StationaryCamInfo[viewID].K[4], StationaryCamInfo[viewID].K[1], StationaryCamInfo[viewID].K[2], StationaryCamInfo[viewID].K[5]);
		fprintf(fp1, "%.6f %.6f %.6f %.6f %.6f %.6f %.6f \n", StationaryCamInfo[viewID].distortion[0], StationaryCamInfo[viewID].distortion[1], StationaryCamInfo[viewID].distortion[2],
			StationaryCamInfo[viewID].distortion[3], StationaryCamInfo[viewID].distortion[4], StationaryCamInfo[viewID].distortion[5], StationaryCamInfo[viewID].distortion[6]);

		fprintf(fp2, "%d %.16f %.16f %.16f 0.0 %.16f %.16f %.16f 0.0 %.16f %.16f %.16f 0.0 0.0 0.0 0.0 1.0 %.16f %.16f %.16f \n", ii,
			StationaryCamInfo[viewID].R[0], StationaryCamInfo[viewID].R[1], StationaryCamInfo[viewID].R[2],
			StationaryCamInfo[viewID].R[3], StationaryCamInfo[viewID].R[4], StationaryCamInfo[viewID].R[5],
			StationaryCamInfo[viewID].R[6], StationaryCamInfo[viewID].R[7], StationaryCamInfo[viewID].R[8],
			StationaryCamInfo[viewID].camCenter[0], StationaryCamInfo[viewID].camCenter[1], StationaryCamInfo[viewID].camCenter[2]);
	}
	fclose(fp1), fclose(fp2);

	return 0;
}

int IncrementalLASyncDriverJump(char *Path, int sampleStartID, int nsamples)
{
	const int nCams = 8, nTracks = 70, motionPriorPower = 2;
	int SelectedCams[] = { 4, 5, 6, 7, 3, 1, 0, 2 }, startFrame = 75, stopFrame = 600;
	//int FrameOffset[nCams] = {0, 4.13, 2.60, 8.78, 6.32, 8.17, 7.70, -7.53}; //30fps accurate according to the down sampling offset of 120fps

	double lamda = 0.8;
	bool EstUncertainty = true;
	if (EstUncertainty)
	{
		double OffsetInfo[8] = { 0, 4.2495, 2.6490, 8.4026, 6.3424, 8.1850, 7.7668, -7.934 };
		//Generate3DUncertaintyFromRandomSampling(Path, SelectedCams, nCams, OffsetInfo, 0, stopFrame, nTracks, sampleStartID, nsamples);
		//Combine3DStatisticFromRandomSampling(Path, nCams, nTracks);
		//TrajectoryTriangulation(Path, SelectedCams, nCams, 0, stopFrame, nTracks, OffsetInfo, lamda, motionPriorPower);
	}
	else
	{
		int PairOne[] = { 4, 5 };
		double OffsetPairOne[] = { 0, 4.2495 };
		//LeastActionSyncBruteForce2DStereo(Path, PairOne, 0, stopFrame, nTracks, OffsetPairOne, -10, 10, 0.1, lamda, motionPriorPower, dummy);
		printf("(%d, %d): %.4f %.4f\n", PairOne[0], PairOne[1], OffsetPairOne[0], OffsetPairOne[1]);

		int PairTwo[] = { 4, 6 };
		double OffsetPairTwo[] = { 0, 2.6490 };
		//LeastActionSyncBruteForce2DStereo(Path, PairTwo, 0, 60, nTracks, OffsetPairTwo, -10, 10, 0.1, lamda, motionPriorPower, dummy);
		printf("(%d, %d): %.4f %.4f\n", PairTwo[0], PairTwo[1], OffsetPairTwo[0], OffsetPairTwo[1]);

		int TripletOne[] = { PairOne[0], PairOne[1], PairTwo[1] };
		double OffsetTripletOne[] = { OffsetPairOne[0], OffsetPairOne[1], OffsetPairTwo[1] };
		//LeastActionSyncBruteForce2DTriplet(Path, TripletOne, 0, stopFrame, nTracks, OffsetTripletOne, -5, 5, 0.1, lamda, motionPriorPower);
		printf("(%d, %d, %d): %.4f %.4f %.4f \n", TripletOne[0], TripletOne[1], TripletOne[2], OffsetTripletOne[0], OffsetTripletOne[1], OffsetTripletOne[2]);

		//IncrementalLeastActionSyncDiscreteContinous2D(Path, SelectedCams, 3, 0, stopFrame, nTracks, OffsetTripletOne, 0, 0, 0, lamda, motionPriorPower, false);
		printf("Triplet refinement: %.4f %.4f %.4f \n", OffsetTripletOne[0], OffsetTripletOne[1], OffsetTripletOne[2]);

		double OffsetInfo[nCams];
		for (int cnCams = 0; cnCams < 3; cnCams++)
			OffsetInfo[cnCams] = OffsetTripletOne[cnCams];

		//OffsetInfo[0] = 0, OffsetInfo[1] = 4.256, OffsetInfo[2] = 2.539;
		//OffsetInfo[0] = 0, OffsetInfo[1] = 4.2530, OffsetInfo[2] = 2.5399, OffsetInfo[3] = 8.3946
		//OffsetInfo[0] = 0, OffsetInfo[1] = 4.2388, OffsetInfo[2] = 2.5734, OffsetInfo[3] = 8.4574, OffsetInfo[4] = 6.305;
		//OffsetInfo[0] = 0, OffsetInfo[1] = 4.2411, OffsetInfo[2] = 2.5746, OffsetInfo[3] = 8.4596, OffsetInfo[4] = 6.3623, OffsetInfo[5] = 8.2;
		//OffsetInfo[0] = 0, OffsetInfo[1] = 4.2750, OffsetInfo[2] = 2.6158, OffsetInfo[3] = 8.4980, OffsetInfo[4] = 6.3919, OffsetInfo[5] = 8.2020, OffsetInfo[6] = 7.7;
		//OffsetInfo[0] = 0, OffsetInfo[1] = 4.2747, OffsetInfo[2] = 2.6155, OffsetInfo[3] = 8.4961, OffsetInfo[4] = 6.3920, OffsetInfo[5] = 8.2024, OffsetInfo[6] = 7.703, OffsetInfo[7] = -7.5;
		OffsetInfo[0] = 0, OffsetInfo[1] = 4.2495, OffsetInfo[2] = 2.6490, OffsetInfo[3] = 8.4026, OffsetInfo[4] = 6.3424, OffsetInfo[5] = 8.1850, OffsetInfo[6] = 7.7668, OffsetInfo[7] = -7.934;//Final
		//for (int addedCams = 7; addedCams < nCams; addedCams++)
		//IncrementalLeastActionSyncDiscreteContinous2D(Path, SelectedCams, addedCams + 1, 0, stopFrame, nTracks, OffsetInfo, 0, 0, 0.1);
		//TrajectoryTriangulation(Path, SelectedCams, nCams, 0, stopFrame, nTracks, OffsetInfo);
	}

	visualizationDriver(Path, nCams, 0, 600, true, false, true, false, true, true, 0);

	return 0;
}
int IncrementalLASyncDriver(char *Path, int sampleStartID, int nsamples)
{
	const int nCams = 3, nTracks = 84, motionPriorPower = 2;
	int SelectedCams[] = { 5, 2, 7 }, startFrame = 0, stopFrame = 500;
	//int FrameOffset[nCams] = {-13, -16, -20, -1, -10, 0, -5, -25, 12}; 

	double lamda = 0.8;
	bool EstUncertainty = false;
	if (EstUncertainty)
	{
		double OffsetInfo[8] = { 0, -19.4488, -24.6508 };
		//Generate3DUncertaintyFromRandomSampling(Path, SelectedCams, nCams, OffsetInfo, 0, stopFrame, nTracks, sampleStartID, nsamples);
		//Combine3DStatisticFromRandomSampling(Path, nCams, nTracks);
		//TrajectoryTriangulation(Path, SelectedCams, nCams, 0, stopFrame, nTracks, OffsetInfo, lamda, motionPriorPower);
	}
	else
	{
		int PairOne[] = { 5, 2 };
		double OffsetPairOne[] = { 0, -19.7 };
		//LeastActionSyncBruteForce2DStereo(Path, PairOne, 0, stopFrame, nTracks, OffsetPairOne, -20,20, 0.1, motionPriorPower);
		//printf("(%d, %d): %.4f %.4f\n", PairOne[0], PairOne[1], OffsetPairOne[0], OffsetPairOne[1]);

		int PairTwo[] = { 5, 7 };
		double OffsetPairTwo[] = { 0, -25.3 };
		//LeastActionSyncBruteForce2DStereo(Path, PairTwo, 0, 60, nTracks, OffsetPairTwo, -20, 20, 0.1, motionPriorPower);
		//printf("(%d, %d): %.4f %.4f\n", PairTwo[0], PairTwo[1], OffsetPairTwo[0], OffsetPairTwo[1]);

		int TripletOne[] = { PairOne[0], PairOne[1], PairTwo[1] };
		double OffsetTripletOne[] = { OffsetPairOne[0], OffsetPairOne[1], OffsetPairTwo[1] };

		//LeastActionSyncBruteForce2DTriplet(Path, TripletOne, 0, stopFrame, nTracks, OffsetTripletOne, -100, 100, 0.1, lamda, motionPriorPower);
		printf("(%d, %d, %d): %.4f %.4f %.4f \n", TripletOne[0], TripletOne[1], TripletOne[2], OffsetTripletOne[0], OffsetTripletOne[1], OffsetTripletOne[2]);

		return 0;
		//IncrementalLeastActionSyncDiscreteContinous2D(Path, SelectedCams, 3, 0, stopFrame, nTracks, OffsetTripletOne, 0, 0, 0, lamda, motionPriorPower, false);
		printf("Triplet refinement: %.4f %.4f %.4f \n", OffsetTripletOne[0], OffsetTripletOne[1], OffsetTripletOne[2]);

		double OffsetInfo[nCams];
		for (int cnCams = 0; cnCams < 3; cnCams++)
			OffsetInfo[cnCams] = OffsetTripletOne[cnCams];

		for (int addedCams = 3; addedCams < nCams; addedCams++)
			;// IncrementalLeastActionSyncDiscreteContinous2D(Path, SelectedCams, addedCams + 1, 0, stopFrame, nTracks, OffsetInfo, 0, 0, 0.1, lamda, motionPriorPower);

		//TrajectoryTriangulation(Path, SelectedCams, nCams, 0, stopFrame, nTracks, OffsetInfo, lamda, motionPriorPower);
	}
	visualizationDriver(Path, nCams, 0, 600, true, false, true, false, true, true, 0);

	return 0;
}
int ConvertTrajectoryToPointCloudTime(char *Path, int npts)
{
	char Fname[512];

	double x, y, z, t;
	vector<double> timeStamp;
	std::vector<int>::iterator it;

	//Read all the time possible
	for (int pid = 0; pid < npts; pid++)
	{
		sprintf(Fname, "%s/_ATrack_%d_0.txt", Path, pid);  FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("Cannot load %s\n", Fname);
			return 1;
		}
		while (fscanf(fp, "%lf %lf %lf %lf", &x, &y, &z, &t) != EOF)
		{
			bool found = false;
			for (int ii = 0; ii < timeStamp.size(); ii++)
			{
				if (abs(timeStamp[ii] - t) < 0.01)
				{
					found = true;
					break;
				}
			}

			if (!found)
				timeStamp.push_back(t);
		}
		fclose(fp);
	}
	sort(timeStamp.begin(), timeStamp.end());

	int ntimes = timeStamp.size();
	vector<int> *PoindID = new vector<int>[ntimes];
	vector<Point3d> *PointCloudTime = new vector<Point3d>[ntimes];
	for (int pid = 0; pid < npts; pid++)
	{
		sprintf(Fname, "%s/_ATrack_%d_0.txt", Path, pid);  FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("Cannot load %s\n", Fname);
			return 1;
		}
		while (fscanf(fp, "%lf %lf %lf %lf", &x, &y, &z, &t) != EOF)
		{
			for (int ii = 0; ii < timeStamp.size(); ii++)
			{
				if (abs(timeStamp[ii] - t) < 0.01)
				{
					PoindID[ii].push_back(pid);
					PointCloudTime[ii].push_back(Point3d(x, y, z));
					break;
				}
			}
		}
		fclose(fp);
	}

	for (int ii = 0; ii < ntimes; ii++)
	{
		sprintf(Fname, "%s/Dynamic/HP3D_%d.xyz", Path, ii);  FILE *fp = fopen(Fname, "w+");
		for (int jj = 0; jj < PointCloudTime[ii].size(); jj++)
			fprintf(fp, "%d %.4f %.4f %.4f\n", PoindID[ii][jj], PointCloudTime[ii][jj].x, PointCloudTime[ii][jj].y, PointCloudTime[ii][jj].z);
		fclose(fp);
	}

	return 0;
}
int RenderSuperCameraFromMultipleUnSyncedCamerasA(char *Path, int startFrame, int stopFrame, int playBackSpeed, bool HighQualityOutputImage = false)
{
	char Fname[200];
	const double fps = 1.0;

	vector<int> SelectedCamera;
	vector<double> TimeStampInfoVector;

	sprintf(Fname, "%s/FMotionPriorSync.txt", Path);	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n. Abort!", Fname);
		abort();
	}
	int selected; double offsetValue;
	while (fscanf(fp, "%d %lf ", &selected, &offsetValue) != EOF)
		SelectedCamera.push_back(selected), TimeStampInfoVector.push_back(offsetValue);
	fclose(fp);


	//Arrange timestamps into a sequence
	int nCams = (int)SelectedCamera.size();

	vector<double>VectorTime;
	vector<int>VectorCamID, VectorFrameID;
	VectorCamID.reserve(nCams*(stopFrame - startFrame)), VectorFrameID.reserve(nCams*(stopFrame - startFrame)), VectorTime.reserve(nCams*(stopFrame - startFrame));

	int *CurrentCameraIFrameID = new int[nCams];
	for (int ii = 0; ii < nCams; ii++)
		CurrentCameraIFrameID[ii] = startFrame;

	while (true)
	{
		//Determine the next camera
		int earliestID, earliestCamFrameID, nfinishedCams = 0;
		double earliestTime = DBL_MAX;
		for (int ii = 0; ii < nCams; ii++)
		{
			if (CurrentCameraIFrameID[ii] > stopFrame)
			{
				nfinishedCams++;
				continue;
			}

			//Time:
			int cframeID = CurrentCameraIFrameID[ii];
			double currentTime = (TimeStampInfoVector[ii] + cframeID) / fps;

			if (currentTime < earliestTime)
			{
				earliestTime = currentTime;
				earliestID = ii;
				earliestCamFrameID = cframeID;
			}
		}

		//If all cameras are done
		if (nfinishedCams == nCams)
			break;

		//Add new data to the sequence
		VectorTime.push_back(earliestTime);
		VectorCamID.push_back(SelectedCamera[earliestID]);
		VectorFrameID.push_back(earliestCamFrameID);

		CurrentCameraIFrameID[earliestID]++;
	}

	//Create super video
	bool noUpdate = false, GlobalSlider = true;
	int oframeID = 0, cframeID = 0;
	sprintf(Fname, "%s/SuperVideo", Path);		makeDir(Fname);

	namedWindow("VideoSequences", CV_WINDOW_NORMAL);
	createTrackbar("Speed", "VideoSequences", &playBackSpeed, 10, NULL);
	createTrackbar("Global frame", "VideoSequences", &cframeID, nCams*(stopFrame - startFrame) - 1, NULL);

	char* nameb1 = "Play/Stop";
	createButton(nameb1, AutomaticPlay, nameb1, CV_CHECKBOX, 0);
	char* nameb2 = "Not Save/Save";
	createButton(nameb2, AutomaticSave, nameb2, CV_CHECKBOX, 0);

	Mat Img;
	while (waitKey(17) != 27)
	{
		if (autoplay)
		{
			playBackSpeed = max(1, playBackSpeed);
			cframeID += playBackSpeed;
			cvSetTrackbarPos(Fname, "VideoSequences", cframeID);
		}

		if (oframeID == cframeID)
			continue;
		cframeID = min(cframeID, (int)VectorFrameID.size() - 1);

		cvSetTrackbarPos("Global frame", "VideoSequences", cframeID);
		sprintf(Fname, "%s/%d/%d.png", Path, VectorCamID[cframeID], VectorFrameID[cframeID]); Img = imread(Fname);
		if (Img.empty())
			continue;

		imshow("VideoSequences", Img);

		oframeID = cframeID;

		if (saveFrame)
		{
			if (HighQualityOutputImage)
				sprintf(Fname, "%s/SuperVideo/%d.png", Path, cframeID);
			else
				sprintf(Fname, "%s/SuperVideo/%d.jpg", Path, cframeID);
			imwrite(Fname, Img);
		}
		printf("Global: (cframeID, time): (%d, %.2fs) \tLocal: (CamID, cframeID): (%d, %d)\n", cframeID, VectorTime[cframeID], VectorCamID[cframeID], VectorFrameID[cframeID]);
	}

	destroyWindow("VideoSequences");

	delete[]CurrentCameraIFrameID;
	return 0;
}

int DomeLeastActionSyncBruteForce3D(char *Path, int npts, double *OffsetInfo)
{
	char Fname[200]; FILE *fp = 0;

	const int approxOrder = 1, gI = 0.0;
	const int LowBound = -80, UpBound = 80, globalOff[2] = { 0, 16 };
	const double timeSize = 0.05, Tscale = 1.0, fps[2] = { 30.00, 25 }, eps = 1.0e-6, scale3d = 1.0;

	double x, y, z;
	int frameID, frameCount;
	PerCamNonRigidTrajectory PerCam_XYZ[2];

	/*vector<Point3d> XYZ;
	sprintf(Fname, "C:/temp/Sim/Dance/3DTracks/%d.txt", 1); fp = fopen(Fname, "r");
	while (fscanf(fp, "%lf %lf %lf", &x, &y, &z) != EOF)
	XYZ.push_back(Point3d(x, y, z));
	fclose(fp);

	sprintf(Fname, "%s/C%d_%d.txt", Path, 0, 0); fp = fopen(Fname, "w+");
	for (int ii = 0; ii < XYZ.size(); ii += 10)
	fprintf(fp, "%d %f %f %f\n", ii / 10, XYZ[ii].x, XYZ[ii].y, XYZ[ii].z);
	fclose(fp);

	sprintf(Fname, "%s/C%d_%d.txt", Path, 1, 0); fp = fopen(Fname, "w+");
	for (int ii = 2; ii < XYZ.size(); ii += 10)
	fprintf(fp, "%d %f %f %f\n", ii / 10, XYZ[ii].x, XYZ[ii].y, XYZ[ii].z);
	fclose(fp);*/

	for (int camID = 0; camID < 2; camID++)
	{
		PerCam_XYZ[camID].npts = npts;
		PerCam_XYZ[camID].Track3DInfo = new Track3D[npts];

		for (int trackID = 0; trackID < npts; trackID++)
		{
			frameCount = 0;
			sprintf(Fname, "%s/C%d_%d.txt", Path, camID, trackID); FILE *fp = fopen(Fname, "r");
			while (fscanf(fp, "%d %lf %lf %lf", &frameID, &x, &y, &z) != EOF)
				frameCount++;
			fclose(fp);

			PerCam_XYZ[camID].Track3DInfo[trackID].nf = frameCount;
			PerCam_XYZ[camID].Track3DInfo[trackID].frameID = new int[frameCount];
			PerCam_XYZ[camID].Track3DInfo[trackID].xyz = new double[3 * frameCount];

			fp = fopen(Fname, "r");
			for (int ii = 0; ii < frameCount; ii++)
				fscanf(fp, "%d %lf %lf %lf", &PerCam_XYZ[camID].Track3DInfo[trackID].frameID[ii],
				&PerCam_XYZ[camID].Track3DInfo[trackID].xyz[3 * ii], &PerCam_XYZ[camID].Track3DInfo[trackID].xyz[3 * ii + 1], &PerCam_XYZ[camID].Track3DInfo[trackID].xyz[3 * ii + 2]);
			fclose(fp);
		}
	}

	int nframes, maxnFrames = 0;
	for (int trackID = 0; trackID < npts; trackID++)
	{
		nframes = 0;
		for (int camID = 0; camID < 2; camID++)
			nframes += PerCam_XYZ[camID].Track3DInfo[trackID].nf;
		if (maxnFrames < nframes)
			maxnFrames = nframes;
	}
	int *TrajectoryPointID = new int[maxnFrames];
	double *TimeStamp = new double[maxnFrames];
	Point3d *Trajectory3D = new Point3d[3 * maxnFrames];

	//int TrajectoryPointID[2380];
	//double TimeStamp[2380];
	//int OffsetValue [UpBound - LowBound + 1];
	//double AllCost[UpBound - LowBound + 1];

	int currentOffset[2];
	int *OffsetValue = new int[UpBound - LowBound + 1];
	double *AllCost = new double[UpBound - LowBound + 1];
	vector<double> VTimeStamp; VTimeStamp.reserve(maxnFrames);
	vector<Point3d> VTrajectory3D; VTrajectory3D.reserve(maxnFrames);
	for (int off = LowBound; off <= UpBound; off++)
	{
		AllCost[off - LowBound] = 0.0;
		OffsetValue[off - LowBound] = off;
		currentOffset[0] = 0, currentOffset[1] = off;

		for (int trackID = 0; trackID < npts; trackID++)
		{
			//Time alignment
			int PointCount = 0;
			for (int camID = 0; camID < 2; camID++)
			{
				for (int frameID = 0; frameID < PerCam_XYZ[camID].Track3DInfo[trackID].nf; frameID++)
				{
					TrajectoryPointID[PointCount] = PointCount;
					TimeStamp[PointCount] = (PerCam_XYZ[camID].Track3DInfo[trackID].frameID[frameID] + globalOff[camID] + timeSize*currentOffset[camID]) / fps[camID];
					Trajectory3D[PointCount].x = scale3d*PerCam_XYZ[camID].Track3DInfo[trackID].xyz[3 * frameID],
						Trajectory3D[PointCount].y = scale3d*PerCam_XYZ[camID].Track3DInfo[trackID].xyz[3 * frameID + 1],
						Trajectory3D[PointCount].z = scale3d*PerCam_XYZ[camID].Track3DInfo[trackID].xyz[3 * frameID + 2];

					PointCount++;
				}
			}
			Quick_Sort_Double(TimeStamp, TrajectoryPointID, 0, PointCount - 1);

			//Re-oder points
			VTrajectory3D.clear(); VTimeStamp.clear();
			for (int ii = 0; ii < PointCount; ii++)
			{
				int pid = TrajectoryPointID[ii];
				VTimeStamp.push_back(TimeStamp[ii]);
				VTrajectory3D.push_back(Trajectory3D[pid]);
			}

			sprintf(Fname, "C:/temp/Off_%d.txt", off); fp = fopen(Fname, "w+");
			for (int ii = 0; ii < PointCount; ii++)
				fprintf(fp, "%f %f %f\n", VTrajectory3D[ii].x, VTrajectory3D[ii].y, VTrajectory3D[ii].z);
			fclose(fp);

			//Compute energy
			double cost1 = ComputeActionEnergy(VTrajectory3D, VTimeStamp, eps, 9800 * gI, 1);
			//double cost2 = ComputeActionEnergy(VTrajectory3D, VTimeStamp, eps, 9800 * gI, 2);
			AllCost[off - LowBound] += cost1;
		}
	}

	Quick_Sort_Double(AllCost, OffsetValue, 0, UpBound - LowBound);
	printf("Least cost: %e @ offset: %f(s) or %f frames \nRatio: %f\n", AllCost[0], (timeSize*OffsetValue[0] + globalOff[1]) / fps[1], timeSize*OffsetValue[0] + globalOff[1], AllCost[0] / AllCost[1]);
	printf("%d %d \n", OffsetValue[0], globalOff[1]);

	OffsetInfo[0] = (timeSize*OffsetValue[0] + globalOff[1]) / fps[1];
	OffsetInfo[1] = timeSize*OffsetValue[0] + globalOff[1];

	delete[]TrajectoryPointID, delete[]TimeStamp, delete[]Trajectory3D, delete[]AllCost;

	return 0;
}
int TestLeastActionConstraint2D(char *Path, int nCams, int npts, double stdev, int nRep = 10000)
{
	int approxOrder = 1;

	char Fname[200];
	VideoData AllVideoInfo;
	if (ReadVideoData(Path, AllVideoInfo, nCams, 0, 3000) == 1)
		return 1;
	int nframes = max(MaxnFrames, 7000);
	double P[12];

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
	for (int power = 2; power <= 10; power += 2)
	{
		for (int gI = 0; gI <= 0; gI++)
		{
			for (int off = 0; off <= 10; off++)
			{
				gtOff[1] = off;
				for (int ii = 0; ii < nCostSize; ii++)
					AllCost[ii] = 0.0;
				//int trackID = 1;
				for (int trackID = 0; trackID < npts; trackID++)
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
					/*double T = 0.8, w = 2.0*Pi / T, A = 50.0;
					for (int ii = 0; ii < 1000; ii++)
					XYZ.push_back(Point3d(0, 1.0*A*sin(w*ii*ialpha), 1.0*ii * 40 * ialpha));*/
					int nTimeInstrances = XYZ.size();

					//Generate 2D points
					ViewerList.clear();
					for (int jj = 0; jj < nCams; jj++)
					{
						int videoID = jj*nframes;
						PerCam_UV_All[jj].clear();
						for (int ii = 0; ii < nTimeInstrances; ii++)
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
						for (int jj = 0; jj < nTimeInstrances; jj += rate)
						{
							int tid = jj + gtOff[ii];
							if (tid >= nTimeInstrances || tid < 0)
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
						for (int ii = 0; ii < nTimeInstrances; ii++)
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
							for (int jj = 0; jj < nTimeInstrances; jj += rate)
							{
								int tid = jj + gtOff[ii];
								if (tid >= nTimeInstrances || tid < 0)
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
							/*fp = fopen("C:/temp/T.txt", "w+");
							for (int ii = 0; ii < Traj.size(); ii++)
							fprintf(fp, "%e %e %e %e\n", Traj[ii].x, Traj[ii].y, Traj[ii].z, Time[ii]);
							fclose(fp);*/
							double cost1 = ComputeActionEnergy(Traj, Time, eps, 9.8 * gI, approxOrder);
							double cost2 = ComputeActionEnergy(Traj, Time, eps, 9.8 * gI, 2);
							AllCost[count] += cost1 + ImCost;

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

				sprintf(Fname, "C:/temp/%d_%d_cost_%d_%.1f.txt", power, gtOff[1], nCams, stdev);  FILE *fp = fopen(Fname, "w+");
				while (true)
				{
					for (int kk = 1; kk < nCams; kk++)
						fprintf(fp, "%d ", currentOffset[kk]);
					fprintf(fp, "%.16e\n", AllCost[count] / nRep);

					currentOffset[1]++;
					RecursiveUpdateCameraOffset(currentOffset, BruteForceTimeWindow, 1, nCams);

					count++;
					if (count == nCostSize)
						break;
				}
				fclose(fp);
			}
		}
	}

	printf("Done with %s\n", Path);
	delete[]currentFrame, delete[]currentOffset, delete[]PerCam_nf, delete[]SortID, delete[]currentOffsetBK, delete[]AllCost;
	delete[]PerCam_UV, delete[]PerCam_UV_All, delete[]PerCam_XYZ;

	return 0;
}
int TestLeastActionConstraint3D(char *Path, int ActionCategory, int ActionID, double g, int approxOrder, double noiseSTD, int nsamples, double *OffsetErrorMean, double*OffsetErrorStd)
{
	char Fname[200]; FILE *fp = 0;

	const int nCams = 2, npts = 31;
	const double fps[2] = { 10.0, 10.0 }, Tscale = 1000.0, eps = 1.0e-16;

	vector<Point3d> *allCleanXYZ = new vector<Point3d>[npts];
	vector<Point3d> *allXYZ = new vector<Point3d>[npts];
	vector<ImgPtEle> PerCam_UV[nCams*npts];
	vector<double> VectorTime;
	vector<Point3d> Vector3D;


	double x, y, z;
	for (int pid = 0; pid < npts; pid++)
	{
		allXYZ[pid].clear();
		sprintf(Fname, "%s/%d_%d/Track3D/%d.txt", Path, ActionCategory, ActionID, pid); fp = fopen(Fname, "r");
		if (fp == NULL)
			return 1;
		while (fscanf(fp, "%lf %lf %lf", &x, &y, &z) != EOF)
			allCleanXYZ[pid].push_back(Point3d(x, y, z));
		fclose(fp);
	}

	double *OffsetError = new double[9 * nsamples];
	for (int sid = 0; sid < nsamples; sid++)
	{
		for (int pid = 0; pid < npts; pid++)
		{
			allXYZ[pid].clear();
			for (int fid = 0; fid < allCleanXYZ[pid].size(); fid++)
			{
				x = allCleanXYZ[pid][fid].x, y = allCleanXYZ[pid][fid].y, z = allCleanXYZ[pid][fid].z;
				x += max(min(3.0*noiseSTD, gaussian_noise(0.0, noiseSTD)), -0.3*noiseSTD);
				y += max(min(3.0*noiseSTD, gaussian_noise(0.0, noiseSTD)), -0.3*noiseSTD);
				z += max(min(3.0*noiseSTD, gaussian_noise(0.0, noiseSTD)), -0.3*noiseSTD);
				allXYZ[pid].push_back(Point3d(x, y, z));
			}
		}

		int OffsetErrorI[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
		for (int offset = 1; offset < 10; offset++)
		{
			ImgPtEle ptEle;
			for (int ii = 0; ii < nCams*npts; ii++)
				PerCam_UV[ii].clear();

			for (int camID = 0; camID < 2; camID++)
			{
				for (int pid = 0; pid<npts; pid++)
				{
					int currentFid = 0;
					while (true)
					{
						int fid = 10 * currentFid + camID*offset;
						if (fid > allXYZ[pid].size() - 1)
							break;

						ptEle.frameID = currentFid;
						ptEle.pt3D = allXYZ[pid][fid];

						PerCam_UV[camID*npts + pid].push_back(ptEle);

						currentFid++;
					}
				}
			}

			int bestOffset = -1;
			double cost, bestCost = 9e99;
			int PerCam_nf[2], currentPID_InTrack[2];
			for (int runningOffset = 1; runningOffset < 10; runningOffset++)
			{
				double earliestTime, currentTime;
				int earliestCamFrameID, frameID, nfinishedCams, earliestCamID;

				cost = 0.0;
				for (int pid = 0; pid < npts; pid++)
				{
					VectorTime.clear(), Vector3D.clear();
					for (int camID = 0; camID < nCams; camID++)
						PerCam_nf[camID] = PerCam_UV[camID*npts + pid].size();

					//Assemble trajactory and time from all Cameras
					for (int jj = 0; jj < nCams; jj++)
						currentPID_InTrack[jj] = 0;

					while (true)
					{
						//Determine the next camera
						nfinishedCams = 0, earliestCamID, earliestTime = 9e9;
						for (int camID = 0; camID < nCams; camID++)
						{
							if (currentPID_InTrack[camID] == PerCam_nf[camID])
							{
								nfinishedCams++;
								continue;
							}

							//Time:
							frameID = PerCam_UV[camID*npts + pid][currentPID_InTrack[camID]].frameID;
							currentTime = (0.1*runningOffset*camID + frameID) / fps[camID] * Tscale;

							if (currentTime < earliestTime)
							{
								earliestTime = currentTime;
								earliestCamID = camID;
								earliestCamFrameID = frameID;
							}
						}

						//If all cameras are done
						if (nfinishedCams == nCams)
							break;

						//Add new point to the sequence
						VectorTime.push_back(earliestTime);
						Vector3D.push_back(PerCam_UV[earliestCamID*npts + pid][currentPID_InTrack[earliestCamID]].pt3D);

						currentPID_InTrack[earliestCamID]++;
					}

					cost += ComputeActionEnergy(Vector3D, VectorTime, eps, g, 1);
				}

				if (cost < bestCost)
					bestOffset = runningOffset, bestCost = cost;
			}
			OffsetErrorI[offset - 1] = bestOffset - offset;
		}
		for (int ii = 0; ii < 9; ii++)
			OffsetError[sid * 9 + ii] = OffsetErrorI[ii];
	}

	for (int sid = 0; sid < nsamples; sid++)
		for (int ii = 0; ii < 9; ii++)
			OffsetErrorMean[ii] += 1.0 / nsamples*OffsetError[sid * 9 + ii];

	for (int sid = 0; sid < nsamples; sid++)
		for (int ii = 0; ii < 9; ii++)
			OffsetErrorStd[ii] += 1.0 / (nsamples - 1)*pow(OffsetError[sid * 9 + ii] - OffsetErrorMean[ii], 2);

	for (int ii = 0; ii < 9; ii++)
		OffsetErrorStd[ii] = sqrt(OffsetErrorStd[ii]);

	//sprintf(Fname, "%s/%d_%d_%d.txt", Path, noiseSTD, ActionCategory, ActionID); fp = fopen(Fname, "w+");
	//for (int ii = 0; ii < 9; ii++)
	//	fprintf(fp, "%.6f ", SuccessRate[ii]);
	//fclose(fp);

	delete[]allCleanXYZ, delete[]allXYZ;

	return 0;
}
int TestMotionPrior3DDriver(char *Path)
{
	srand(time(NULL));

	int numthreads = 1;// omp_get_max_threads();
	omp_set_num_threads(numthreads);

	const int nLevels = 1, nsamples = 50;

	//Least Kinetic
	/*for (int noiseLevel = 0; noiseLevel < nLevels; noiseLevel++)
	{
	int *notvalid = new int[150 * 50];
	for (int ii = 0; ii < 150 * 50; ii++)
	notvalid[ii] = 1;
	double *OffsetErrorMean = new double[150 * 50 * 9], *OffsetErrorStd = new double[150 * 50 * 9];
	for (int ii = 0; ii < 150 * 50 * 9; ii++)
	OffsetErrorMean[ii] = 0.0, OffsetErrorStd[ii] = 0.0;

	double percent = 2.5, incre = 2.5, startTime = omp_get_wtime();
	#pragma omp parallel for
	for (int catergoryID = 1; catergoryID < 150; catergoryID++)
	{
	#pragma omp critical
	if (omp_get_thread_num() == 0 && 100.0*catergoryID / 150 >= percent)
	{
	printf("Level %d: %.2f%% (%.2fs)....\n", noiseLevel, 1.0*percent * numthreads, omp_get_wtime() - startTime);
	percent += incre;
	}

	for (int actionID = 1; actionID < 50; actionID++)
	{
	double LocalOffsetErrorMean[9] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
	double LocalOffsetErrorStd[9] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
	notvalid[catergoryID * 50 + actionID] = TestLeastActionConstraint3D(Path, catergoryID, actionID, 0.0, 1, 5.0*noiseLevel, noiseLevel == 0 ? 1 : nsamples, LocalOffsetErrorMean, LocalOffsetErrorStd);
	for (int ii = 0; ii < 9; ii++)
	OffsetErrorMean[(catergoryID * 50 + actionID) * 9 + ii] = LocalOffsetErrorMean[ii],
	OffsetErrorStd[(catergoryID * 50 + actionID) * 9 + ii] = LocalOffsetErrorStd[ii];
	}
	}
	printf("Level %d: Done\n", noiseLevel);

	char Fname[200]; sprintf(Fname, "%s/Level_%d.txt", Path, noiseLevel); FILE *fp = fopen(Fname, "w+");
	for (int catergoryID = 1; catergoryID < 150; catergoryID++)
	{
	for (int actionID = 1; actionID < 50; actionID++)
	{
	if (notvalid[catergoryID * 50 + actionID] == 1)
	continue;

	fprintf(fp, "%d %d ", catergoryID, actionID);
	for (int jj = 0; jj < 9; jj++)
	fprintf(fp, "%.4f ", OffsetErrorMean[(catergoryID * 50 + actionID) * 9 + jj]);
	for (int jj = 0; jj < 9; jj++)
	fprintf(fp, "%.4f ", OffsetErrorStd[(catergoryID * 50 + actionID) * 9 + jj]);
	fprintf(fp, "%\n");
	}
	}
	fclose(fp);

	delete[]notvalid, delete[]OffsetErrorMean, delete[]OffsetErrorStd;
	}*/

	//Least action
	for (int noiseLevel = 0; noiseLevel < nLevels; noiseLevel++)
	{
		int *notvalid = new int[150 * 50];
		for (int ii = 0; ii < 150 * 50; ii++)
			notvalid[ii] = 1;
		double *OffsetErrorMean = new double[150 * 50 * 9], *OffsetErrorStd = new double[150 * 50 * 9];
		for (int ii = 0; ii < 150 * 50 * 9; ii++)
			OffsetErrorMean[ii] = 0.0, OffsetErrorStd[ii] = 0.0;

		double percent = 2.5, incre = 2.5, startTime = omp_get_wtime();
#pragma omp parallel for
		for (int catergoryID = 1; catergoryID < 150; catergoryID++)
		{
#pragma omp critical
			if (omp_get_thread_num() == 0 && 100.0*catergoryID / 150 >= percent)
			{
				printf("Level %d: %.2f%% (%.2fs)....\n", noiseLevel, 1.0*percent * numthreads, omp_get_wtime() - startTime);
				percent += incre;
			}

			for (int actionID = 1; actionID < 50; actionID++)
			{
				double LocalOffsetErrorMean[9] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
				double LocalOffsetErrorStd[9] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
				notvalid[catergoryID * 50 + actionID] = TestLeastActionConstraint3D(Path, catergoryID, actionID, 0.0098, 1, 5.0*noiseLevel, noiseLevel == 0 ? 1 : nsamples, LocalOffsetErrorMean, LocalOffsetErrorStd);
				for (int ii = 0; ii < 9; ii++)
					OffsetErrorMean[(catergoryID * 50 + actionID) * 9 + ii] = LocalOffsetErrorMean[ii],
					OffsetErrorStd[(catergoryID * 50 + actionID) * 9 + ii] = LocalOffsetErrorStd[ii];
			}
		}
		printf("Level %d: Done\n", noiseLevel);

		char Fname[200]; sprintf(Fname, "%s/gLevel_%d.txt", Path, noiseLevel); FILE *fp = fopen(Fname, "w+");
		for (int catergoryID = 1; catergoryID < 150; catergoryID++)
		{
			for (int actionID = 1; actionID < 50; actionID++)
			{
				if (notvalid[catergoryID * 50 + actionID] == 1)
					continue;

				fprintf(fp, "%d %d ", catergoryID, actionID);
				for (int jj = 0; jj < 9; jj++)
					fprintf(fp, "%.4f ", OffsetErrorMean[(catergoryID * 50 + actionID) * 9 + jj]);
				for (int jj = 0; jj < 9; jj++)
					fprintf(fp, "%.4f ", OffsetErrorStd[(catergoryID * 50 + actionID) * 9 + jj]);
				fprintf(fp, "%\n");
			}
		}
		fclose(fp);

		delete[]notvalid, delete[]OffsetErrorMean, delete[]OffsetErrorStd;
	}

	//Least force
	/*for (int noiseLevel = 0; noiseLevel < nLevels; noiseLevel++)
	{
	int *notvalid = new int[150 * 50];
	for (int ii = 0; ii < 150 * 50; ii++)
	notvalid[ii] = 1;
	double *OffsetErrorMean = new double[150 * 50 * 9], *OffsetErrorStd = new double[150 * 50 * 9];
	for (int ii = 0; ii < 150 * 50 * 9; ii++)
	OffsetErrorMean[ii] = 0.0, OffsetErrorStd[ii] = 0.0;

	double percent = 2.5, incre = 2.5, startTime = omp_get_wtime();
	#pragma omp parallel for
	for (int catergoryID = 1; catergoryID < 150; catergoryID++)
	{
	#pragma omp critical
	if (omp_get_thread_num() == 0 && 100.0*catergoryID / 150 >= percent)
	{
	printf("Level %d: %.2f%% (%.2fs)....\n", noiseLevel, 1.0*percent * numthreads, omp_get_wtime() - startTime);
	percent += incre;
	}

	for (int actionID = 1; actionID < 50; actionID++)
	{
	double LocalOffsetErrorMean[9] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
	double LocalOffsetErrorStd[9] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
	notvalid[catergoryID * 50 + actionID] = TestLeastActionConstraint3D(Path, catergoryID, actionID, 0.0, 2, 5.0*noiseLevel, noiseLevel == 0 ? 1 : nsamples, LocalOffsetErrorMean, LocalOffsetErrorStd);
	for (int ii = 0; ii < 9; ii++)
	OffsetErrorMean[(catergoryID * 50 + actionID) * 9 + ii] = LocalOffsetErrorMean[ii],
	OffsetErrorStd[(catergoryID * 50 + actionID) * 9 + ii] = LocalOffsetErrorStd[ii];
	}
	}
	printf("Level %d: Done\n", noiseLevel);

	char Fname[200]; sprintf(Fname, "%s/aLevel_%d.txt", Path, noiseLevel); FILE *fp = fopen(Fname, "w+");
	for (int catergoryID = 1; catergoryID < 150; catergoryID++)
	{
	for (int actionID = 1; actionID < 50; actionID++)
	{
	if (notvalid[catergoryID * 50 + actionID] == 1)
	continue;

	fprintf(fp, "%d %d ", catergoryID, actionID);
	for (int jj = 0; jj < 9; jj++)
	fprintf(fp, "%.4f ", OffsetErrorMean[(catergoryID * 50 + actionID) * 9 + jj]);
	for (int jj = 0; jj < 9; jj++)
	fprintf(fp, "%.4f ", OffsetErrorStd[(catergoryID * 50 + actionID) * 9 + jj]);
	fprintf(fp, "%\n");
	}
	}
	fclose(fp);

	delete[]notvalid, delete[]OffsetErrorMean, delete[]OffsetErrorStd;
	}*/

	return 0;
}

//Test Least action on still images
int GenerateHynSooData(char *Path, int nCams, int ntracks)
{
	bool save3D = false, fixed3D = false, fixedTime = false, non_monotonicDescent = true;
	int gtOff[] = { 0, 1, 4, 8, 9 };

	const double Tscale = 1000.0, ialpha = 1.0 / 120, rate = 10, eps = 1.0e-6, lamda = .8;

	char Fname[200];
	VideoData AllVideoInfo;
	sprintf(Fname, "%s/Calib", Path);
	if (ReadVideoData(Fname, AllVideoInfo, nCams, 0, 7000) == 1)
		return 1;

	int nframes = max(MaxnFrames, 7000);
	double ActionCost = 0.0, TProjCost = 0.0, x, y, z;
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

	int nTimeInstrances;
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
		nTimeInstrances = XYZ[trackID].size();

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

		//Get C, P, Q, U
		for (int camID = 0; camID < nCams; camID++)
		{
			for (int frameID = 0; frameID < nTimeInstrances; frameID++)
			{
				for (int kk = 0; kk < 12; kk++)
				{
					P[kk] = AllVideoInfo.VideoInfo[camID*nframes + frameID].P[kk];
					PerCamUV_All[camID][frameID].P[kk] = P[kk];
				}
				for (int kk = 0; kk < 9; kk++)
					PerCamUV_All[camID][frameID].K[kk] = AllVideoInfo.VideoInfo[camID*nframes + frameID].K[kk],

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
		for (int frameID = 0; frameID < nTimeInstrances; frameID++)
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
			for (int frameID = 0; frameID < nTimeInstrances; frameID += rate)
			{
				int tid = frameID + gtOff[camID];
				if (tid >= nTimeInstrances || tid < 0)
					continue;
				PerCam_UV[camID*ntracks + trackID].push_back(PerCamUV_All[camID][tid]);
			}
			if (PerCam_UV[camID*ntracks + trackID].size() > maxPerCamFrames)
				maxPerCamFrames = PerCam_UV[camID*ntracks + trackID].size();
		}
	}
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		sprintf(Fname, "%s/GTTrack_%d.txt", Path, trackID); FILE *fp = fopen(Fname, "w+");
		for (int ii = 0; ii < nCams; ii++)
		{
			for (int jj = 0; jj < nTimeInstrances; jj += rate)
			{
				int tid = jj + gtOff[ii];
				if (tid >= nTimeInstrances || tid < 0)
					continue;
				fprintf(fp, "%f %f %f %d\n", XYZ[trackID][tid].xyz.x, XYZ[trackID][tid].xyz.y, XYZ[trackID][tid].xyz.z, tid);
			}
		}
		fclose(fp);
	}

	int cumpts = 0;
	for (int camID = 0; camID < nCams; camID++)
		cumpts += PerCam_UV[camID*ntracks].size();

	sprintf(Fname, "%s/HS_data.txt", Path); FILE *fp = fopen(Fname, "w+");
	fprintf(fp, "nProjections: %d\n", cumpts);
	fprintf(fp, "nPoints: %d\n", ntracks);
	int nOff[] = { 0, 2, 4, 6, 8 };
	for (int camID = 0; camID < nCams; camID++)
	{
		for (int frameID = 0; frameID < PerCam_UV[camID].size(); frameID++)
		{
			if (frameID>PerCam_UV[camID*ntracks].size() - 1)
				continue;

			double timeStamp = //1.0*gtOff[camID] * ialpha*Tscale + 1.0*frameID * ialpha*Tscale*rate;
				1.0*nOff[camID] * ialpha*Tscale + 1.0*frameID * ialpha*Tscale*rate;
			fprintf(fp, "%d_%d %f\n", camID, frameID, timeStamp);

			fprintf(fp, "%.16f %.16f %.16f\n", PerCam_UV[camID*ntracks][frameID].camcenter[0], PerCam_UV[camID*ntracks][frameID].camcenter[1], PerCam_UV[camID*ntracks][frameID].camcenter[2]);
			for (int ii = 0; ii < 3; ii++)
				fprintf(fp, "%.16f %.16f %.16f\n", PerCam_UV[camID*ntracks][frameID].R[3 * ii + 0], PerCam_UV[camID*ntracks][frameID].R[3 * ii + 1], PerCam_UV[camID*ntracks][frameID].R[3 * ii + 2]);
			fprintf(fp, "%.16f %.16f %.16f \n0.0 %.16f %.16f \n0.0 0.0 1.0\n", PerCam_UV[camID*ntracks][frameID].K[0], PerCam_UV[camID*ntracks][frameID].K[1], PerCam_UV[camID*ntracks][frameID].K[2], PerCam_UV[camID*ntracks][frameID].K[4], PerCam_UV[camID*ntracks][frameID].K[5]);
			for (int trackID = 0; trackID < ntracks; trackID++)
				fprintf(fp, "%f %f ", PerCam_UV[camID*ntracks + trackID][frameID].pt2D.x, PerCam_UV[camID*ntracks + trackID][frameID].pt2D.y);
			fprintf(fp, "\n");
		}
	}
	fclose(fp);

	return 0;
}
int ReadHyunSooData(char *Fname, VideoData &VData, PerCamNonRigidTrajectory &Traject2D, vector<double> &TimeStamp, int &nTimeInstances, int &ntracks)
{
	char str[200];
	double ts;

	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot open %s\n", Fname);
		return 1;
	}
	fscanf(fp, "%s %d", str, &nTimeInstances);
	fscanf(fp, "%s %d", str, &ntracks);


	TimeStamp.reserve(nTimeInstances);
	VData.nVideos = nTimeInstances;
	VData.VideoInfo = new CameraData[nTimeInstances];

	Traject2D.npts = ntracks;
	Traject2D.Track2DInfo = new Track2D[ntracks];
	for (int ii = 0; ii < ntracks; ii++)
		Traject2D.Track2DInfo[ii].uv = new Point2d[nTimeInstances];

	for (int ii = 0; ii < nTimeInstances; ii++)
	{
		fscanf(fp, "%s %lf", str, &ts); TimeStamp.push_back(ts);
		fscanf(fp, "%lf %lf %lf ", &VData.VideoInfo[ii].camCenter[0], &VData.VideoInfo[ii].camCenter[1], &VData.VideoInfo[ii].camCenter[2]);
		for (int jj = 0; jj < 9; jj++)
			fscanf(fp, "%lf ", &VData.VideoInfo[ii].R[jj]);
		for (int jj = 0; jj < 9; jj++)
			fscanf(fp, "%lf ", &VData.VideoInfo[ii].K[jj]);

		GetTfromC(VData.VideoInfo[ii]);
		AssembleP(VData.VideoInfo[ii]);

		for (int jj = 0; jj < ntracks; jj++)
			fscanf(fp, "%lf %lf ", &Traject2D.Track2DInfo[jj].uv[ii].x, &Traject2D.Track2DInfo[jj].uv[ii].y);
	}

	return 0;
}
int SimultaneousSyncReconcHyunSooData(char *Path)
{
	const double fps = 1.0, ialpha = 1.0 / fps, Tscale = 1.0, eps = 1.0e-6, lamda = .8;
	bool save3D = false, fixed3D = false, fixedTime = false, non_monotonicDescent = true;

	char Fname[200];

	VideoData AllVideoInfo;
	PerCamNonRigidTrajectory Traject2D;
	vector<double> TimeStampGT;
	int nTimeInstrances, ntracks;
	sprintf(Fname, "%s/HS_data.txt", Path);
	ReadHyunSooData(Fname, AllVideoInfo, Traject2D, TimeStampGT, nTimeInstrances, ntracks);

	int nCams = AllVideoInfo.nVideos, nframes = 1;

	double ActionCost = 0.0, TProjCost = 0.0;
	double AA[6], bb[2], ccT[3], dd[1], *P = new double[12], *Q = new double[6 * nCams], *U = new double[2 * nCams];

	Point2d pt2D;
	ImgPtEle ptEle;
	vector<ImgPtEle> *PerCam_UV = new vector<ImgPtEle>[nCams*ntracks];

	//Assuming for one time instance, one camera sees at least one point
	double *currentOffset = new double[nCams];
	for (int ii = 0; ii < nCams; ii++)
	{
		currentOffset[ii] = TimeStampGT[ii]; //ordering is given in this case

		//Add noise to the time
		double t = gaussian_noise(0, 8.3 * 2);
		t = min(max(t, -8.2), 8.2);
		currentOffset[ii] += t;
	}
	currentOffset[0] = 0;

	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		for (int camID = 0; camID < nCams; camID++)
		{
			pt2D = Traject2D.Track2DInfo[trackID].uv[camID];
			if (pt2D.x < 1 && pt2D.y < 1)
				continue;

			ptEle.pt2D = Traject2D.Track2DInfo[trackID].uv[camID];
			double stdA = 0.01;
			ptEle.pt3D = Point3d(gaussian_noise(0.0, stdA), gaussian_noise(0.0, stdA), gaussian_noise(0.0, stdA)); //Even this works because of convex cost function
			//ptEle.pt3D = Point3d(0, 0, 0);

			for (int kk = 0; kk < 12; kk++)
				ptEle.P[kk] = AllVideoInfo.VideoInfo[camID].P[kk];
			for (int kk = 0; kk < 9; kk++)
				ptEle.K[kk] = AllVideoInfo.VideoInfo[camID].K[kk], ptEle.R[kk] = AllVideoInfo.VideoInfo[camID].R[kk];

			//Q, U
			P = ptEle.P;
			AA[0] = P[0], AA[1] = P[1], AA[2] = P[2], bb[0] = P[3];
			AA[3] = P[4], AA[4] = P[5], AA[5] = P[6], bb[1] = P[7];
			ccT[0] = P[8], ccT[1] = P[9], ccT[2] = P[10], dd[0] = P[11];

			ptEle.Q[0] = AA[0] - pt2D.x*ccT[0], ptEle.Q[1] = AA[1] - pt2D.x*ccT[1], ptEle.Q[2] = AA[2] - pt2D.x*ccT[2];
			ptEle.Q[3] = AA[3] - pt2D.y*ccT[0], ptEle.Q[4] = AA[4] - pt2D.y*ccT[1], ptEle.Q[5] = AA[5] - pt2D.y*ccT[2];
			ptEle.u[0] = dd[0] * pt2D.x - bb[0], ptEle.u[1] = dd[0] * pt2D.y - bb[1];

			PerCam_UV[camID*ntracks + trackID].push_back(ptEle);
		}
	}

	vector<int> PerPoint_nFrames;
	vector<int *> PerTrackFrameID(ntracks);
	vector<double*> PerTrack3D(ntracks);
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		int npts = 0;
		for (int camID = 0; camID < nCams; camID++)
			npts += PerCam_UV[camID*ntracks + trackID].size();
		PerTrackFrameID[trackID] = new int[npts];
		PerTrack3D[trackID] = new double[3 * npts];
	}

	double rate = 8.3;
	double Cost[4];
	MotionPrior_Optim_SpatialStructure_Algebraic(Path, PerTrack3D, PerCam_UV, PerPoint_nFrames, currentOffset, ntracks, non_monotonicDescent, nCams, 4, Tscale, ialpha, eps, lamda, Cost, true, false);
	MotionPrior_ML_Weighting(PerCam_UV, ntracks, nCams);
	MotionPrior_Optim_SpatialStructure_Geometric(Path, PerTrack3D, PerCam_UV, PerPoint_nFrames, currentOffset, ntracks, non_monotonicDescent, nCams, 4, Tscale, ialpha, eps, lamda, Cost, true, false);

	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		sprintf(Fname, "%s/A1Track_%d.txt", Path, trackID);  FILE *fp = fopen(Fname, "w+");
		for (int camID = 0; camID < nCams; camID++)
			for (int frameID = 0; frameID < PerCam_UV[camID*ntracks + trackID].size(); frameID++)
				fprintf(fp, "%.4f %.4f %.4f %.1f\n", PerCam_UV[camID*ntracks + trackID][frameID].pt3D.x, PerCam_UV[camID*ntracks + trackID][frameID].pt3D.y, PerCam_UV[camID*ntracks + trackID][frameID].pt3D.z, 1.0*currentOffset[camID] * ialpha*Tscale + 1.0*frameID * ialpha*Tscale*rate);
		fclose(fp);
	}

	sprintf(Fname, "%s/TS_1.txt", Path);	FILE *fp = fopen(Fname, "w+");
	for (int ll = 0; ll < nCams; ll++)
		fprintf(fp, "%d %f\n", ll, currentOffset[ll]);
	fclose(fp);

	printf("Layer 2\n");
	MotionPrior_Optim_SpatialStructure_Algebraic(Path, PerTrack3D, PerCam_UV, PerPoint_nFrames, currentOffset, ntracks, non_monotonicDescent, nCams, 4, Tscale, ialpha, eps, lamda, Cost, true, false);
	MotionPrior_ML_Weighting(PerCam_UV, ntracks, nCams);
	MotionPrior_Optim_SpatialStructure_Geometric(Path, PerTrack3D, PerCam_UV, PerPoint_nFrames, currentOffset, ntracks, non_monotonicDescent, nCams, 4, Tscale, ialpha, eps, lamda, Cost, true, false);
	MotionPrior_Optim_ST_Geometric(Path, PerTrack3D, PerCam_UV, PerPoint_nFrames, currentOffset, ntracks, non_monotonicDescent, nCams, 4, Tscale, ialpha, eps, lamda, Cost, true, false);

	//printf("Saving results...\n");
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		sprintf(Fname, "%s/A2Track_%d.txt", Path, trackID);  FILE *fp = fopen(Fname, "w+");
		for (int camID = 0; camID < nCams; camID++)
			for (int frameID = 0; frameID < 1; frameID++)
				fprintf(fp, "%.4f %.4f %.4f %.1f\n", PerCam_UV[camID*ntracks + trackID][frameID].pt3D.x, PerCam_UV[camID*ntracks + trackID][frameID].pt3D.y, PerCam_UV[camID*ntracks + trackID][frameID].pt3D.z, 1.0*currentOffset[camID] * ialpha*Tscale + 1.0*frameID * ialpha*Tscale*rate);
		fclose(fp);
	}

	//printf("Final results\n");
	//for (int ll = 0; ll < nCams; ll++)
	///	printf("%d %f\n", ll, currentOffset[ll]);

	sprintf(Fname, "%s/TS.txt", Path);	fp = fopen(Fname, "w+");
	for (int ll = 0; ll < nCams; ll++)
		fprintf(fp, "%d %f\n", ll, currentOffset[ll]);
	fclose(fp);

	for (int trackID = 0; trackID < ntracks; trackID++)
		delete[]PerTrack3D[trackID];
	for (int ii = 0; ii < ntracks*nCams; ii++)
		PerCam_UV[ii].clear();

	delete[]PerCam_UV, delete[]Q, delete[]U, delete[]currentOffset;

	return 0;
}
int TestLeastActionStillImages(char *Path)
{
	int nCams = 5, nTracks = 31;
	GenerateHynSooData(Path, nCams, nTracks);
	SimultaneousSyncReconcHyunSooData(Path);
	visualizationDriver(Path, 5, 0, 269, false, false, false, true, false, false, 0);
	return 0;
}

//Least Action synthetic data
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
int SimulateCamerasAnd2DPointsForMoCap(char *Path, int nCams, int n3DTracks, double *Intrinsic, double *distortion, int width, int height, double radius = 5e3, bool saveGT3D = true, bool show2DImage = false, int Rate = 1, double PMissingData = 0.0, double Noise2D = 2.0, int *UnSyncFrameTimeStamp = NULL)
{
	if (UnSyncFrameTimeStamp == NULL)
	{
		UnSyncFrameTimeStamp = new int[nCams];
		for (int ii = 0; ii < nCams; ii++)
			UnSyncFrameTimeStamp[ii] = 0;
	}

	char Fname[200];
	double noise3D_CamShake = 20 / 60;
	double x, y, z, cx = 0, cy = 0, cz = 0;
	vector<Point3d> XYZ;
	sprintf(Fname, "%s/Track3D/%d.txt", Path, 0); FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return 1;
	}
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
			Point3d Noise3D(gaussian_noise(0.0, noise3D_CamShake), gaussian_noise(0.0, noise3D_CamShake), gaussian_noise(0.0, noise3D_CamShake));
			if (Noise3D.x > 3.0*noise3D_CamShake)
				Noise3D.x = 3.0*noise3D_CamShake;
			else if (Noise3D.x < -3.0 *noise3D_CamShake)
				Noise3D.x = -3.0*noise3D_CamShake;
			if (Noise3D.y > 3.0*noise3D_CamShake)
				Noise3D.y = 3.0*noise3D_CamShake;
			else if (Noise3D.y < -3.0 *noise3D_CamShake)
				Noise3D.y = -3.0*noise3D_CamShake;
			if (Noise3D.z > 3.0*noise3D_CamShake)
				Noise3D.z = 3.0*noise3D_CamShake;
			else if (Noise3D.z < -3.0 *noise3D_CamShake)
				Noise3D.z = -3.0*noise3D_CamShake;

			Camera[frameID + nframes*camID].valid = true;
			GenerateCamerasExtrinsicOnCircle(Camera[frameID + nframes*camID], theta, radius, XYZ[frameID], XYZ[frameID], Noise3D);
			SetIntrinisc(Camera[frameID + nframes*camID], Intrinsic);
			GetKFromIntrinsic(Camera[frameID + nframes*camID]);
			for (int ii = 0; ii < 7; ii++)
				Camera[frameID + nframes*camID].distortion[ii] = distortion[ii];
			AssembleP(Camera[frameID + nframes*camID]);
			Center.push_back(Point3d(Camera[frameID + nframes*camID].camCenter[0], Camera[frameID + nframes*camID].camCenter[1], Camera[frameID + nframes*camID].camCenter[2]));
		}
		angleList;
		Center;
	}

	Point2d pt;
	Point3d p3d;
	vector<Point3d> *allXYZ = new vector<Point3d>[n3DTracks];
	for (int trackID = 0; trackID < n3DTracks; trackID++)
	{
		sprintf(Fname, "%s/Track3D/%d.txt", Path, trackID); fp = fopen(Fname, "r");
		if (fp == NULL)
			printf("Cannot load %s\n", Fname);
		while (fscanf(fp, "%lf %lf %lf", &x, &y, &z) != EOF)
			allXYZ[trackID].push_back(Point3d(x, y, z));
		fclose(fp);
	}

	vector<int> UsedFrames;
	for (int camID = 0; camID < nCams; camID++)
	{
		sprintf(Fname, "%s/Intrinsic_%d.txt", Path, camID); fp = fopen(Fname, "w+");
		for (int frameID = 0; frameID < nframes; frameID += Rate)
		{
			if (frameID + UnSyncFrameTimeStamp[camID] > nframes || !Camera[frameID + UnSyncFrameTimeStamp[camID] + nframes*camID].valid)
				continue;

			UsedFrames.push_back(frameID + UnSyncFrameTimeStamp[camID]);
			fprintf(fp, "%d 0 0 %d %d ", frameID / Rate, width, height);
			for (int ii = 0; ii < 5; ii++)
				fprintf(fp, "%f ", Camera[frameID + UnSyncFrameTimeStamp[camID] + nframes*camID].intrinsic[ii]);
			for (int ii = 0; ii < 7; ii++)
				fprintf(fp, "%f ", distortion[ii]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	for (int camID = 0; camID < nCams; camID++)
	{
		sprintf(Fname, "%s/CamPose_%d.txt", Path, camID); fp = fopen(Fname, "w+");
		for (int frameID = 0; frameID < nframes; frameID += Rate)
		{
			if (frameID + UnSyncFrameTimeStamp[camID] > nframes || !Camera[frameID + UnSyncFrameTimeStamp[camID] + nframes*camID].valid)
				continue;

			GetRCGL(Camera[frameID + UnSyncFrameTimeStamp[camID] + nframes*camID]);

			fprintf(fp, "%d ", frameID / Rate);
			for (int jj = 0; jj < 16; jj++)
				fprintf(fp, "%.16f ", Camera[frameID + UnSyncFrameTimeStamp[camID] + nframes*camID].Rgl[jj]);
			for (int jj = 0; jj < 3; jj++)
				fprintf(fp, "%.16f ", Camera[frameID + UnSyncFrameTimeStamp[camID] + nframes*camID].camCenter[jj]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	vector<ImgPtEle> *PerCam_UV = new vector<ImgPtEle>[nCams*n3DTracks];
	for (int camID = 0; camID < nCams; camID++)
	{
		for (int trackID = 0; trackID < n3DTracks; trackID++)
		{
			int nf = 0;
			for (int frameID = 0; frameID < nframes; frameID += Rate)
			{
				if (frameID + UnSyncFrameTimeStamp[camID] > allXYZ[trackID].size() - 1)
					continue;
				if (frameID + UnSyncFrameTimeStamp[camID] > nframes)
					continue;
				nf++;
			}

			//Simulate random missing data
			vector<int> randomNumber;
			for (int ii = 0; ii < nf; ii++)
				randomNumber.push_back(ii);
			random_shuffle(randomNumber.begin(), randomNumber.end());

			int nMissingData = (int)(PMissingData*nf);
			sort(randomNumber.begin(), randomNumber.begin() + nMissingData);

			for (int frameID = 0; frameID < nframes; frameID += Rate)
			{
				if (frameID + UnSyncFrameTimeStamp[camID] > allXYZ[trackID].size() - 1)
					continue;
				if (frameID + UnSyncFrameTimeStamp[camID] > nframes)
					continue;

				bool missed = false;
				for (int ii = 0; ii < nMissingData; ii++)
				{
					if (randomNumber[ii] == frameID / Rate)
					{
						missed = true; break;
					}
				}
				if (missed)
					continue;

				ProjectandDistort(allXYZ[trackID][frameID + UnSyncFrameTimeStamp[camID]], &pt, Camera[frameID + UnSyncFrameTimeStamp[camID] + camID*nframes].P, Camera[frameID + UnSyncFrameTimeStamp[camID] + camID*nframes].K, Camera[frameID + UnSyncFrameTimeStamp[camID] + camID*nframes].distortion);
				Point2d Noise(gaussian_noise(0.0, Noise2D), gaussian_noise(0.0, Noise2D));
				if (Noise.x > 3.0*Noise2D)
					Noise.x = 3.0*Noise2D;
				else if (Noise.x < -3.0 *Noise2D)
					Noise.x = -3.0*Noise2D;
				if (Noise.y > 3.0*Noise2D)
					Noise.y = 3.0*Noise2D;
				else if (Noise.y < -3.0 *Noise2D)
					Noise.y = -3.0*Noise2D;
				pt.x += Noise.x, pt.y += Noise.y;

				ImgPtEle ptEle;
				ptEle.pt2D = pt, ptEle.viewID = camID, ptEle.frameID = frameID / Rate,
					ptEle.imWidth = width, ptEle.imHeight = height;
				ptEle.pt3D = allXYZ[trackID][frameID + UnSyncFrameTimeStamp[camID]];
				ptEle.timeStamp = frameID + UnSyncFrameTimeStamp[camID];
				PerCam_UV[camID*n3DTracks + trackID].push_back(ptEle);
			}
		}
	}

	sprintf(Fname, "%s/Track2D", Path), makeDir(Fname);
	for (int camID = 0; camID < nCams; camID++)
	{
		sprintf(Fname, "%s/Track2D/%d.txt", Path, camID); fp = fopen(Fname, "w+");
		for (int trackID = 0; trackID < n3DTracks; trackID++)
		{
			fprintf(fp, "%d %d ", trackID, (int)PerCam_UV[camID*n3DTracks + trackID].size());
			for (int fid = 0; fid < (int)PerCam_UV[camID*n3DTracks + trackID].size(); fid++)
				fprintf(fp, "%d %.16f %.16f ", PerCam_UV[camID*n3DTracks + trackID][fid].frameID, PerCam_UV[camID*n3DTracks + trackID][fid].pt2D.x, PerCam_UV[camID*n3DTracks + trackID][fid].pt2D.y);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	if (saveGT3D)
	{
		for (int trackID = 0; trackID < n3DTracks; trackID++)
		{
			sprintf(Fname, "%s/GTTrack_%d.txt", Path, trackID); remove(Fname);	fp = fopen(Fname, "w+");
			for (int camID = 0; camID < nCams; camID++)
			{
				for (int ii = 0; ii < (int)PerCam_UV[camID*n3DTracks + trackID].size(); ii++)
				{
					ImgPtEle &x = PerCam_UV[camID*n3DTracks + trackID][ii];
					fprintf(fp, "%.4f %.4f %.4f %.4f %d %d\n", PerCam_UV[camID*n3DTracks + trackID][ii].pt3D.x, PerCam_UV[camID*n3DTracks + trackID][ii].pt3D.y, PerCam_UV[camID*n3DTracks + trackID][ii].pt3D.z,
						PerCam_UV[camID*n3DTracks + trackID][ii].timeStamp, PerCam_UV[camID*n3DTracks + trackID][ii].viewID, PerCam_UV[camID*n3DTracks + trackID][ii].frameID);
				}
			}
			fclose(fp);
		}
	}

	/*sprintf(Fname, "%s/Track2D", Path), makeDir(Fname);
	for (int camID = 0; camID < nCams; camID++)
	{
	sprintf(Fname, "%s/Track2D/%d.txt", Path, camID); fp = fopen(Fname, "w+");
	for (int trackID = 0; trackID < n3DTracks; trackID++)
	{
	int nf = 0;
	for (int frameID = 0; frameID < nframes; frameID += Rate)
	{
	if (frameID + UnSyncFrameTimeStamp[camID] > allXYZ[trackID].size() - 1)
	continue;
	if (frameID + UnSyncFrameTimeStamp[camID] > nframes)
	continue;
	nf++;
	}

	//Simulate random missing data
	vector<int> randomNumber;
	for (int ii = 0; ii < nf; ii++)
	randomNumber.push_back(ii);
	random_shuffle(randomNumber.begin(), randomNumber.end());

	int nMissingData = (int)(PMissingData*nf);
	sort(randomNumber.begin(), randomNumber.begin() + nMissingData);

	fprintf(fp, "%d %d ", trackID, nf - nMissingData);
	for (int frameID = 0; frameID < nframes; frameID += Rate)
	{
	if (frameID + UnSyncFrameTimeStamp[camID] > allXYZ[trackID].size() - 1)
	continue;
	if (frameID + UnSyncFrameTimeStamp[camID] > nframes)
	continue;

	bool missed = false;
	for (int ii = 0; ii < nMissingData; ii++)
	{
	if (randomNumber[ii] == frameID / Rate)
	{
	missed = true; break;
	}
	}
	if (missed)
	continue;

	ProjectandDistort(allXYZ[trackID][frameID + UnSyncFrameTimeStamp[camID]], &pt, Camera[frameID + UnSyncFrameTimeStamp[camID] + camID*nframes].P, Camera[frameID + UnSyncFrameTimeStamp[camID] + camID*nframes].K, Camera[frameID + UnSyncFrameTimeStamp[camID] + camID*nframes].distortion);
	Point2d Noise(gaussian_noise(0.0, Noise2D), gaussian_noise(0.0, Noise2D));
	if (Noise.x > 3.0*Noise2D)
	Noise.x = 3.0*Noise2D;
	else if (Noise.x < -3.0 *Noise2D)
	Noise.x = -3.0*Noise2D;
	if (Noise.y > 3.0*Noise2D)
	Noise.y = 3.0*Noise2D;
	else if (Noise.y < -3.0 *Noise2D)
	Noise.y = -3.0*Noise2D;
	pt.x += Noise.x, pt.y += Noise.y;
	fprintf(fp, "%d %.16f %.16f ", frameID / Rate, pt.x, pt.y);
	}
	fprintf(fp, "\n");
	}
	fclose(fp);
	}*/

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
				displayImg = Img.clone();
				for (int trackID = 0; trackID < n3DTracks; trackID++)
				{
					ProjectandDistort(allXYZ[trackID][frameID + UnSyncFrameTimeStamp[camID]], &pt, Camera[frameID + nframes*camID].P, Camera[frameID + UnSyncFrameTimeStamp[camID] + camID*nframes].K, Camera[frameID + UnSyncFrameTimeStamp[camID] + camID*nframes].distortion);
					circle(displayImg, pt, 4, colors[trackID % 9], 1, 8, 0);
				}

				sprintf(Fname, "Cam %d: frame %d", camID, frameID);
				CvPoint text_origin = { width / 30, height / 30 };
				putText(displayImg, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 3.0 * 640 / Img.cols, CV_RGB(255, 0, 0), 2);
				imshow("Image", displayImg);
				waitKey(1);
			}
		}
	}
	return 0;
}
int TestLeastActionOnSyntheticData(char *Path, int STCalibration)
{
	//General setup
	const int nCams = 10, npts = 31, width = 1920, height = 1080, startFrame = 0, stopFrame = 55, rate = 10;
	double Intrinsic[5] = { 1500, 1500, 0, 960, 540 }, distortion[7] = { 0, 0, 0, 0, 0, 0, 0 }, radius = 3000, PMissingData = 0.0, noise2D = 2.0;
	int UnSyncFrameTimeStamp[] = { 0, 1, 29, 5, 16, 3, 27, 2, 18, 4 };
	//SimulateCamerasAnd2DPointsForMoCap(Path, nCams, npts, Intrinsic, distortion, width, height, radius, true, false, rate, PMissingData, noise2D, UnSyncFrameTimeStamp);

	//Input Parameters
	const int PriorOrder = 1, motionPriorPower = 2, SearchRange = 10;
	double lamdaData = 0.85, lamdaPrior = 1.0 - lamdaData, SearchStep = 0.1;

	if (STCalibration == 1)
	{
		double startTime = omp_get_wtime();
		double FrameLevelOffsetInfo[nCams];
		GeometricConstraintSyncDriver(Path, nCams, npts, startFrame, startFrame, stopFrame, SearchRange, true, FrameLevelOffsetInfo, false);

		//for some reasons, only work with motionPriorPower = 2
		/*EvaluateAllPairCost(Path, nCams, npts, startFrame, stopFrame, SearchRange, SearchStep, lamdaData, motionPriorPower, FrameLevelOffsetInfo);*/

		vector<int>cameraOrdering;
		vector<double> InitTimeStampInfoVector;
		DetermineCameraOrderingForGreedySTBA(Path, "PairwiseCost", nCams, cameraOrdering, InitTimeStampInfoVector);

		vector<int> SelectedCamera;
		vector<double>TimeStampInfoVector;
		for (int ii = 0; ii < 3; ii++)
			SelectedCamera.push_back(cameraOrdering[ii]), TimeStampInfoVector.push_back(InitTimeStampInfoVector[ii]);

		printf("\nCoarse ST estimation for 3 cameras (%d, %d, %d):\n", SelectedCamera[0], SelectedCamera[1], SelectedCamera[2]);
		LeastActionSyncBruteForce2DTriplet(Path, SelectedCamera, 0, stopFrame, npts, TimeStampInfoVector, -SearchRange / 2, SearchRange / 2, SearchStep, lamdaData, motionPriorPower);
		printf("Coarse ST estimation for 3 cameras (%d, %d, %d): %.4f %.4f %.4f \n\n", SelectedCamera[0], SelectedCamera[1], SelectedCamera[2], TimeStampInfoVector[0], TimeStampInfoVector[1], TimeStampInfoVector[2]);

		IncrementalLeastActionSyncDiscreteContinous2D(Path, SelectedCamera, startFrame, stopFrame, npts, TimeStampInfoVector, 0, 0, 0, lamdaData, motionPriorPower, false);

		char Fname[200]; sprintf(Fname, "%s/MotionPriorSyncProgress.txt", Path);	FILE *fp = fopen(Fname, "w+");
		fprintf(fp, "%d %d %d %d %.4f %.4f %.4f \n", 3, SelectedCamera[0], SelectedCamera[1], SelectedCamera[2], TimeStampInfoVector[0], TimeStampInfoVector[1], TimeStampInfoVector[2]);

		int orderingChangeTrials = 0;
		for (int currentCamID = 3; currentCamID < nCams; currentCamID++)
		{
			SelectedCamera.push_back(cameraOrdering[currentCamID]);
			TimeStampInfoVector.push_back(InitTimeStampInfoVector[currentCamID]);
			int NotSuccess = IncrementalLeastActionSyncDiscreteContinous2D(Path, SelectedCamera, startFrame, stopFrame, npts, TimeStampInfoVector, -(int)(SearchStep* SearchRange), (int)(SearchStep* SearchRange), 0.0, lamdaData, motionPriorPower);

			//Push the current ordering ID to the end of the stack
			if (NotSuccess == 1)
			{
				SelectedCamera.erase(SelectedCamera.end() - 1);
				TimeStampInfoVector.erase(TimeStampInfoVector.end() - 1);

				//Nothing can be done with this last camera
				if (currentCamID == nCams - 1)
					break;

				std::vector<int>::iterator it1;
				it1 = cameraOrdering.end();
				cameraOrdering.insert(it1, cameraOrdering[currentCamID]);
				cameraOrdering.erase(cameraOrdering.begin() + currentCamID);

				std::vector<double>::iterator it2;
				it2 = InitTimeStampInfoVector.end();
				InitTimeStampInfoVector.insert(it2, InitTimeStampInfoVector[currentCamID]);
				InitTimeStampInfoVector.erase(InitTimeStampInfoVector.begin() + currentCamID);

				currentCamID--;

				orderingChangeTrials++;
				if (orderingChangeTrials == nCams - 3) //Already tried other ordering options but failed
					break;
			}
			else
			{
				fprintf(fp, "%d ", (int)SelectedCamera.size());
				for (int ii = 0; ii < (int)SelectedCamera.size(); ii++)
					fprintf(fp, "%d ", SelectedCamera[ii]);
				for (int ii = 0; ii < (int)SelectedCamera.size(); ii++)
					fprintf(fp, "%.4f ", TimeStampInfoVector[ii]);
				fprintf(fp, "\n");
			}
		}
		fclose(fp);

		double earliest = 9e9;
		vector<int> SortedSelectedCamera;
		vector<double> SortedTimeStampInfoVector;
		SortedSelectedCamera = SelectedCamera;
		sort(SortedSelectedCamera.begin(), SortedSelectedCamera.end());
		for (int ii = 0; ii < (int)SelectedCamera.size(); ii++)
		{
			for (int jj = 0; jj < (int)SelectedCamera.size(); jj++)
			{
				if (SortedSelectedCamera[ii] == SelectedCamera[jj])
				{
					SortedTimeStampInfoVector.push_back(TimeStampInfoVector[jj]);
					if (earliest > TimeStampInfoVector[jj])
						earliest = TimeStampInfoVector[jj];
					break;
				}
			}
		}
		for (int ii = (int)SortedSelectedCamera.size() - 1; ii >= 0; ii--)
			SortedTimeStampInfoVector[ii] -= earliest;

		sprintf(Fname, "%s/FMotionPriorSync.txt", Path);	fp = fopen(Fname, "w+");
		for (int ii = 0; ii < (int)SortedSelectedCamera.size(); ii++)
			fprintf(fp, "%d %.4f\n", SortedSelectedCamera[ii], SortedTimeStampInfoVector[ii]);
		fclose(fp);

		//Triangulate trajectories
		TrajectoryTriangulation(Path, SortedSelectedCamera, SortedTimeStampInfoVector, npts, startFrame, stopFrame, lamdaData, motionPriorPower);
		ResamplingOf3DTrajectorySplineDriver(Path, SortedSelectedCamera, SortedTimeStampInfoVector, startFrame, stopFrame, npts, lamdaData);
		ResamplingOf3DTrajectoryDCTDriver(Path, SortedSelectedCamera, SortedTimeStampInfoVector, PriorOrder, startFrame, stopFrame, npts, lamdaData, lamdaPrior);

		printf("Total time: %.2fs\n\n", omp_get_wtime() - startTime);
	}
	else if (STCalibration > 1)
	{
		double startTime = omp_get_wtime();

		vector<int> SelectedCamera;
		vector<double> TimeStampInfoVector;

		char Fname[200]; sprintf(Fname, "%s/FMotionPriorSync.txt", Path);	FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("Cannot load %s\n. Abort!", Fname);
			abort();
		}
		int selected; double offsetValue;
		while (fscanf(fp, "%d %lf ", &selected, &offsetValue) != EOF)
			SelectedCamera.push_back(selected), TimeStampInfoVector.push_back(offsetValue);
		fclose(fp);

		int nCams = (int)SelectedCamera.size();
		vector<int>  sortedSelectedCamera(nCams);
		vector<double> sortedTimeStampInfoVector(nCams);
		for (int ii = 0; ii < nCams; ii++)
			sortedSelectedCamera[ii] = ii,
			sortedTimeStampInfoVector[SelectedCamera[ii]] = TimeStampInfoVector[ii];

		if (STCalibration == 2)
			TrajectoryTriangulation(Path, sortedSelectedCamera, sortedTimeStampInfoVector, npts, startFrame, stopFrame, lamdaData, motionPriorPower);
		else if (STCalibration == 3)
			ResamplingOf3DTrajectorySplineDriver(Path, sortedSelectedCamera, sortedTimeStampInfoVector, startFrame, stopFrame, npts, lamdaData);
		else if (STCalibration == 4)
			ResamplingOf3DTrajectoryDCTDriver(Path, sortedSelectedCamera, sortedTimeStampInfoVector, PriorOrder, startFrame, stopFrame, npts, lamdaData, lamdaPrior);
		else if (STCalibration == 4)
			Generate3DUncertaintyFromRandomSampling(Path, sortedSelectedCamera, sortedTimeStampInfoVector, startFrame, stopFrame, npts, 0, 100, motionPriorPower);
		else if (STCalibration == 6)
		{
			vector<int>DelayInfoVector;
			for (int ii = 0; ii < nCams; ii++)
				DelayInfoVector.push_back((int)(-(1.0*sortedTimeStampInfoVector[ii] + 0.5)));

			double Rate = 10.0, ialpha = 1.0;
			double GTFrameTimeStamp[10];
			for (int ii = 0; ii < (int)sortedTimeStampInfoVector.size(); ii++)
				GTFrameTimeStamp[ii] = sortedTimeStampInfoVector[ii];

			TriangulateFrameSync2DTrajectories(Path, sortedSelectedCamera, DelayInfoVector, startFrame, stopFrame, npts, false, GTFrameTimeStamp, &ialpha, &Rate);
		}
		printf("Total time: %.2fs\n\n", omp_get_wtime() - startTime);
	}

	visualizationDriver(Path, nCams, startFrame, stopFrame, false, false, true, false, false, false, 0);
	return 0;
}

int SimulateRollingShutterCameraAnd2DPointsForMoCap(char *Path, int nCams, int n3DTracks, double *Intrinsic, double *distortion, int width, int height, double radius = 5e3, bool saveGT3D = true, bool show2DImage = false, int Rate = 1, double PMissingData = 0.0, double Noise2D = 2.0, int *UnSyncFrameTimeStamp = NULL, bool RollingShutter = true)
{
	if (UnSyncFrameTimeStamp == NULL)
	{
		UnSyncFrameTimeStamp = new int[nCams];
		for (int ii = 0; ii < nCams; ii++)
			UnSyncFrameTimeStamp[ii] = 0;
	}

	char Fname[200];
	double noise3D_CamShake = 20 / 60;
	double x, y, z, cx = 0, cy = 0, cz = 0;

	sprintf(Fname, "%s/RollingShutterPose", Path), makeDir(Fname);

	int nframes;
	Point2d pt;
	Point3d p3d;
	vector<Point3d> *allXYZ = new vector<Point3d>[n3DTracks];
	for (int pid = 0; pid < n3DTracks; pid++)
	{
		sprintf(Fname, "%s/Track3D/%d.txt", Path, pid); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
			printf("Cannot load %s\n", Fname);
		while (fscanf(fp, "%lf %lf %lf", &x, &y, &z) != EOF)
		{
			if (pid == 0)
				cx += x, cy += y, cz += z;
			else
				x -= cx, y -= cy, z -= cz;

			allXYZ[pid].push_back(Point3d(x, y, z));
		}
		fclose(fp);

		if (pid == 0)
		{
			nframes = (int)allXYZ[pid].size();
			cx /= nframes, cy /= nframes, cz /= nframes;
		}
	}

	for (int camID = 0; camID < nCams; camID++)
	{
		sprintf(Fname, "%s/Intrinsic_%d.txt", Path, camID); FILE *fp1 = fopen(Fname, "w+");
		sprintf(Fname, "%s/CamPose_%d.txt", Path, camID); FILE *fp2 = fopen(Fname, "w+");
		for (int frameID = 0; frameID < nframes; frameID += Rate)
		{
			CameraData CamI;
			double theta = 20.0*cos(2.0*Pi / 25 * frameID) + 360 * camID / nCams;
			GenerateCamerasExtrinsicOnCircle(CamI, theta / 180.0 * Pi, radius, Point3d(cx, cy - 700, cz), Point3d(cx, cy - 700, cz), Point3d(0, 0, 0));

			SetIntrinisc(CamI, Intrinsic);
			GetKFromIntrinsic(CamI);
			fprintf(fp1, "%d 0 0 %d %d ", frameID / Rate, width, height);
			for (int ii = 0; ii < 5; ii++)
				fprintf(fp1, "%f ", CamI.intrinsic[ii]);
			for (int ii = 0; ii < 7; ii++)
				fprintf(fp1, "%f ", distortion[ii]);
			fprintf(fp1, "\n");

			getrFromR(CamI.R, CamI.rt);
			GetRCGL(CamI);
			fprintf(fp2, "%d ", frameID / Rate);
			for (int jj = 0; jj < 3; jj++)
				fprintf(fp2, "%.16f ", CamI.rt[jj]);
			for (int jj = 0; jj < 3; jj++)
				fprintf(fp2, "%.16f ", CamI.camCenter[jj]);
			fprintf(fp2, "\n");
		}
		fclose(fp1), fclose(fp2);
	}

	vector<ImgPtEle> *PerCam_UV = new vector<ImgPtEle>[nCams*nframes];
	for (int camID = 0; camID < nCams; camID++)
	{
		for (int pid = 0; pid < n3DTracks; pid++)
		{
			//Simulate random missing data
			vector<int> randomNumber;
			for (int ii = 0; ii < nframes; ii++)
				randomNumber.push_back(ii);
			random_shuffle(randomNumber.begin(), randomNumber.end());

			int nMissingData = (int)(PMissingData*nframes);
			sort(randomNumber.begin(), randomNumber.begin() + nMissingData);

			for (int frameID = 0; frameID < nframes; frameID += Rate)
			{
				if (frameID + UnSyncFrameTimeStamp[camID] > allXYZ[pid].size() - 1)
					continue;
				if (frameID + UnSyncFrameTimeStamp[camID] > nframes)
					continue;

				bool missed = false;
				for (int ii = 0; ii < nMissingData; ii++)
				{
					if (randomNumber[ii] == frameID / Rate)
					{
						missed = true; break;
					}
				}
				if (missed)
					continue;

				int iter;
				CameraData CamI;
				double relativeDiff, theta = 20.0*cos(2.0*Pi / 25 * frameID) + 360 * camID / nCams,
					subframe = 0.0, osubframe = subframe;

				for (iter = 0; iter < 20; iter++) //iterative fix point solver since we don't know the time
				{
					GenerateCamerasExtrinsicOnCircle(CamI, theta / 180.0 * Pi, radius, Point3d(cx, cy - 700, cz), Point3d(cx, cy - 700, cz), Point3d(0, 0, 0));
					SetIntrinisc(CamI, Intrinsic);
					GetKFromIntrinsic(CamI);
					for (int ii = 0; ii < 7; ii++)
						CamI.distortion[ii] = distortion[ii];
					AssembleP(CamI);
					ProjectandDistort(allXYZ[pid][frameID + UnSyncFrameTimeStamp[camID]], &pt, CamI.P, CamI.K, CamI.distortion);
					subframe = RollingShutter ? pt.y / height : 0.0;
					relativeDiff = RollingShutter ? abs((subframe - osubframe) / subframe) : 0.0;
					if (relativeDiff < 1.e-9)
						break;
					osubframe = subframe;
					theta = 20.0*cos(2.0*Pi / 25 * (subframe + frameID)) + 360 * camID / nCams;
				}
				if (iter > 19)
					continue;

				Point2d Noise(gaussian_noise(0.0, Noise2D), gaussian_noise(0.0, Noise2D));
				if (Noise.x > 3.0*Noise2D)
					Noise.x = 3.0*Noise2D;
				else if (Noise.x < -3.0 *Noise2D)
					Noise.x = -3.0*Noise2D;
				if (Noise.y > 3.0*Noise2D)
					Noise.y = 3.0*Noise2D;
				else if (Noise.y < -3.0 *Noise2D)
					Noise.y = -3.0*Noise2D;
				pt.x += Noise.x, pt.y += Noise.y;

				ImgPtEle ptEle;
				ptEle.pt2D = pt, ptEle.viewID = camID, ptEle.frameID = frameID / Rate, ptEle.imWidth = width, ptEle.imHeight = height;
				ptEle.pt3D = allXYZ[pid][frameID + UnSyncFrameTimeStamp[camID]];
				ptEle.timeStamp = (RollingShutter ? subframe : 0.0 + frameID) / Rate + UnSyncFrameTimeStamp[camID];

				for (int ii = 0; ii < 9; ii++)
					ptEle.R[ii] = CamI.R[ii];
				for (int ii = 0; ii < 3; ii++)
					ptEle.T[ii] = CamI.T[ii];
				PerCam_UV[camID*n3DTracks + pid].push_back(ptEle);
			}
		}
	}

	sprintf(Fname, "%s/Track2D", Path), makeDir(Fname);
	for (int camID = 0; camID < nCams; camID++)
	{
		if (RollingShutter)
			sprintf(Fname, "%s/Track2D/R_%d.txt", Path, camID);
		else
			sprintf(Fname, "%s/Track2D/G_%d.txt", Path, camID);
		FILE *fp = fopen(Fname, "w+");
		for (int trackID = 0; trackID < n3DTracks; trackID++)
		{
			int nf = (int)PerCam_UV[camID*n3DTracks + trackID].size();
			fprintf(fp, "%d %d ", trackID, nf);
			for (int fid = 0; fid < nf; fid++)
				fprintf(fp, "%d %.16f %.16f ", PerCam_UV[camID*n3DTracks + trackID][fid].frameID, PerCam_UV[camID*n3DTracks + trackID][fid].pt2D.x, PerCam_UV[camID*n3DTracks + trackID][fid].pt2D.y);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	if (saveGT3D)
	{
		for (int trackID = 0; trackID < n3DTracks; trackID++)
		{
			sprintf(Fname, "%s/GTTrack_%d.txt", Path, trackID); remove(Fname);	FILE *fp = fopen(Fname, "w+");
			for (int camID = 0; camID < nCams; camID++)
			{
				for (int ii = 0; ii < (int)PerCam_UV[camID*n3DTracks + trackID].size(); ii++)
				{
					fprintf(fp, "%.4f %.4f %.4f %.4f %d %d\n", PerCam_UV[camID*n3DTracks + trackID][ii].pt3D.x, PerCam_UV[camID*n3DTracks + trackID][ii].pt3D.y, PerCam_UV[camID*n3DTracks + trackID][ii].pt3D.z,
						PerCam_UV[camID*n3DTracks + trackID][ii].timeStamp, PerCam_UV[camID*n3DTracks + trackID][ii].viewID, PerCam_UV[camID*n3DTracks + trackID][ii].frameID);
				}
			}
			fclose(fp);
		}
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

		for (int frameID = 0; frameID < nframes; frameID++)
		{
			displayImg = Img.clone();
			for (int pid = 0; pid < nframes; pid++)
			{
				for (int ii = 0; ii < (int)PerCam_UV[pid].size(); ii++)
				{
					if (frameID == PerCam_UV[pid][ii].frameID)
					{
						circle(displayImg, PerCam_UV[pid][ii].pt2D, 4, colors[pid % 9], 2, 8, 0);
						break;
					}
				}
			}

			sprintf(Fname, "frame %d", frameID);
			CvPoint text_origin = { width / 30, height / 30 };
			putText(displayImg, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 3.0 * 640 / Img.cols, CV_RGB(255, 0, 0), 2);
			imshow("Image", displayImg);
			waitKey(10);
		}
	}
	return 0;
}
int TestRollingShutterSpatialTemporalReconOnSyntheticData(char *Path, int STCalibration)
{
	const int nCams = 10, npts = 31, width = 1920, height = 1080, startFrame = 0, stopFrame = 539;
	double Intrinsic[5] = { 1500, 1500, 0, 960, 540 }, distortion[7] = { 0, 0, 0, 0, 0, 0, 0 }, radius = 3000, noise2D = 0.0;
	int UnSyncFrameTimeStamp[] = { 0, 1, 29, 5, 16, 3, 27, 2, 18, 4 };
	SimulateRollingShutterCameraAnd2DPointsForMoCap(Path, nCams, npts, Intrinsic, distortion, width, height, radius, 1, 0, 10, 0.0, noise2D, UnSyncFrameTimeStamp, true);

	//Input Parameters
	const int PriorOrder = 1, motionPriorPower = 2, SearchRange = 10;
	double lamdaData = 0.85, lamdaPrior = 1.0 - lamdaData, SearchStep = 0.1;

	if (STCalibration == 1)
	{
		double startTime = omp_get_wtime();
		double FrameLevelOffsetInfo[nCams];
		GeometricConstraintSyncDriver(Path, nCams, npts, startFrame, startFrame, stopFrame, SearchRange, true, FrameLevelOffsetInfo, false);

		//Convert from time-delay format to  time-stamp format
		for (int ii = 0; ii < nCams; ii++)
			FrameLevelOffsetInfo[ii] = -FrameLevelOffsetInfo[ii];

		//for some reasons, only work with motionPriorPower = 2
		EvaluateAllPairCost(Path, nCams, npts, startFrame, stopFrame, SearchRange, SearchStep, lamdaData, motionPriorPower, FrameLevelOffsetInfo);

		vector<int>cameraOrdering;
		vector<double> InitTimeStampInfoVector;
		DetermineCameraOrderingForGreedySTBA(Path, "PairwiseCost", nCams, cameraOrdering, InitTimeStampInfoVector);

		vector<int> SelectedCamera;
		vector<double>TimeStampInfoVector;
		for (int ii = 0; ii < 3; ii++)
			SelectedCamera.push_back(cameraOrdering[ii]), TimeStampInfoVector.push_back(InitTimeStampInfoVector[ii]);

		printf("\nCoarse ST estimation for 3 cameras (%d, %d, %d):\n", SelectedCamera[0], SelectedCamera[1], SelectedCamera[2]);
		LeastActionSyncBruteForce2DTriplet(Path, SelectedCamera, 0, stopFrame, npts, TimeStampInfoVector, -SearchRange / 2, SearchRange / 2, SearchStep, lamdaData, motionPriorPower);
		printf("Coarse ST estimation for 3 cameras (%d, %d, %d): %.4f %.4f %.4f \n\n", SelectedCamera[0], SelectedCamera[1], SelectedCamera[2], TimeStampInfoVector[0], TimeStampInfoVector[1], TimeStampInfoVector[2]);

		IncrementalLeastActionSyncDiscreteContinous2D(Path, SelectedCamera, startFrame, stopFrame, npts, TimeStampInfoVector, 0, 0, 0, lamdaData, motionPriorPower, false);

		char Fname[200]; sprintf(Fname, "%s/MotionPriorSyncProgress.txt", Path);	FILE *fp = fopen(Fname, "w+");
		fprintf(fp, "%d %d %d %d %.4f %.4f %.4f \n", 3, SelectedCamera[0], SelectedCamera[1], SelectedCamera[2], TimeStampInfoVector[0], TimeStampInfoVector[1], TimeStampInfoVector[2]);

		int orderingChangeTrials = 0;
		for (int currentCamID = 3; currentCamID < nCams; currentCamID++)
		{
			SelectedCamera.push_back(cameraOrdering[currentCamID]);
			TimeStampInfoVector.push_back(InitTimeStampInfoVector[currentCamID]);
			int NotSuccess = IncrementalLeastActionSyncDiscreteContinous2D(Path, SelectedCamera, startFrame, stopFrame, npts, TimeStampInfoVector, -(int)(SearchStep* SearchRange), (int)(SearchStep* SearchRange), 0.0, lamdaData, motionPriorPower);

			//Push the current ordering ID to the end of the stack
			if (NotSuccess == 1)
			{
				SelectedCamera.erase(SelectedCamera.end() - 1);
				TimeStampInfoVector.erase(TimeStampInfoVector.end() - 1);

				//Nothing can be done with this last camera
				if (currentCamID == nCams - 1)
					break;

				std::vector<int>::iterator it1;
				it1 = cameraOrdering.end();
				cameraOrdering.insert(it1, cameraOrdering[currentCamID]);
				cameraOrdering.erase(cameraOrdering.begin() + currentCamID);

				std::vector<double>::iterator it2;
				it2 = InitTimeStampInfoVector.end();
				InitTimeStampInfoVector.insert(it2, InitTimeStampInfoVector[currentCamID]);
				InitTimeStampInfoVector.erase(InitTimeStampInfoVector.begin() + currentCamID);

				currentCamID--;

				orderingChangeTrials++;
				if (orderingChangeTrials == nCams - 3) //Already tried other ordering options but failed
					break;
			}
			else
			{
				fprintf(fp, "%d ", (int)SelectedCamera.size());
				for (int ii = 0; ii < (int)SelectedCamera.size(); ii++)
					fprintf(fp, "%d ", SelectedCamera[ii]);
				for (int ii = 0; ii < (int)SelectedCamera.size(); ii++)
					fprintf(fp, "%.4f ", TimeStampInfoVector[ii]);
				fprintf(fp, "\n");
			}
		}
		fclose(fp);

		double earliest = 9e9;
		vector<int> SortedSelectedCamera;
		vector<double> SortedTimeStampInfoVector;
		SortedSelectedCamera = SelectedCamera;
		sort(SortedSelectedCamera.begin(), SortedSelectedCamera.end());
		for (int ii = 0; ii < (int)SelectedCamera.size(); ii++)
		{
			for (int jj = 0; jj < (int)SelectedCamera.size(); jj++)
			{
				if (SortedSelectedCamera[ii] == SelectedCamera[jj])
				{
					SortedTimeStampInfoVector.push_back(TimeStampInfoVector[jj]);
					if (earliest > TimeStampInfoVector[jj])
						earliest = TimeStampInfoVector[jj];
					break;
				}
			}
		}
		for (int ii = (int)SortedSelectedCamera.size() - 1; ii >= 0; ii--)
			SortedTimeStampInfoVector[ii] -= earliest;

		sprintf(Fname, "%s/FMotionPriorSync.txt", Path);	fp = fopen(Fname, "w+");
		for (int ii = 0; ii < (int)SortedSelectedCamera.size(); ii++)
			fprintf(fp, "%d %.4f\n", SortedSelectedCamera[ii], SortedTimeStampInfoVector[ii]);
		fclose(fp);

		//Triangulate trajectories
		TrajectoryTriangulation(Path, SortedSelectedCamera, SortedTimeStampInfoVector, npts, startFrame, stopFrame, lamdaData, motionPriorPower);
		ResamplingOf3DTrajectorySplineDriver(Path, SortedSelectedCamera, SortedTimeStampInfoVector, startFrame, stopFrame, npts, lamdaData);
		ResamplingOf3DTrajectoryDCTDriver(Path, SortedSelectedCamera, SortedTimeStampInfoVector, PriorOrder, startFrame, stopFrame, npts, lamdaData, lamdaPrior);

		printf("Total time: %.2fs\n\n", omp_get_wtime() - startTime);
	}
	else if (STCalibration > 1)
	{
		double startTime = omp_get_wtime();

		vector<int> SelectedCamera;
		vector<double> TimeStampInfoVector;

		char Fname[200]; sprintf(Fname, "%s/FMotionPriorSync.txt", Path);	FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("Cannot load %s\n. Abort!", Fname);
			abort();
		}
		int selected; double offsetValue;
		while (fscanf(fp, "%d %lf ", &selected, &offsetValue) != EOF)
			SelectedCamera.push_back(selected), TimeStampInfoVector.push_back(offsetValue);
		fclose(fp);

		int nCams = (int)SelectedCamera.size();
		vector<int>  sortedSelectedCamera(nCams);
		vector<double> sortedTimeStampInfoVector(nCams);
		for (int ii = 0; ii < nCams; ii++)
			sortedSelectedCamera[ii] = ii,
			sortedTimeStampInfoVector[SelectedCamera[ii]] = TimeStampInfoVector[ii];

		if (STCalibration == 2)
			TrajectoryTriangulation(Path, sortedSelectedCamera, sortedTimeStampInfoVector, npts, startFrame, stopFrame, lamdaData, motionPriorPower);
		else if (STCalibration == 3)
			ResamplingOf3DTrajectorySplineDriver(Path, sortedSelectedCamera, sortedTimeStampInfoVector, startFrame, stopFrame, npts, lamdaData);
		else if (STCalibration == 4)
			ResamplingOf3DTrajectoryDCTDriver(Path, sortedSelectedCamera, sortedTimeStampInfoVector, PriorOrder, startFrame, stopFrame, npts, lamdaData, lamdaPrior);
		else if (STCalibration == 4)
			Generate3DUncertaintyFromRandomSampling(Path, sortedSelectedCamera, sortedTimeStampInfoVector, startFrame, stopFrame, npts, 0, 100, motionPriorPower);
		else if (STCalibration == 6)
		{
			vector<int>DelayInfoVector;
			for (int ii = 0; ii < nCams; ii++)
				DelayInfoVector.push_back((int)(-(1.0*sortedTimeStampInfoVector[ii] + 0.5)));

			double Rate = 10.0, ialpha = 1.0;
			double GTFrameTimeStamp[10];
			for (int ii = 0; ii < (int)sortedTimeStampInfoVector.size(); ii++)
				GTFrameTimeStamp[ii] = sortedTimeStampInfoVector[ii];

			TriangulateFrameSync2DTrajectories(Path, sortedSelectedCamera, DelayInfoVector, startFrame, stopFrame, npts, false, GTFrameTimeStamp, &ialpha, &Rate);
		}
		printf("Total time: %.2fs\n\n", omp_get_wtime() - startTime);
	}

	visualizationDriver(Path, nCams, startFrame, stopFrame, false, false, true, false, false, false, 0);

	return 0;
}

//Least Action dome data
int LaserDetectionCorrelationDriver(char *Path)
{
	omp_set_num_threads(omp_get_max_threads());

	double sigma = 1.0, thresh = 0.7;
	int PatternSize = 15, nscales = 3, NMS_BW = 40;
	int nPanels = 20, nHDs = 30;
	int startF = 1, stopF = 1800, width = 1920, height = 1080;

	//HD images
#pragma omp parallel for
	for (int camID = 0; camID <= nHDs; camID++)
	{
#pragma omp critical
		printf("Averaging HDcam %d\n", camID);
		unsigned char *mImg = new unsigned char[width * height * 3];
		ComputeAverageImage(Path, mImg, width, height, camID, 0, startF, startF + 200);//compute mean images
		delete[]mImg;
	}

	int percent = 10, incre = 10, jobCount = 0;
	double startTime = omp_get_wtime();

#pragma omp parallel for
	for (int camID = 0; camID <= nHDs; camID++)
	{
#pragma omp critical
		if (100.0*jobCount / nHDs >= percent)
		{
			printf("\r %.2f%% (%.2fs) jobs finished....", 100.0*jobCount / nHDs, omp_get_wtime() - startTime);
			percent += incre;
		}
		jobCount++;

		unsigned char *MeanImg = new unsigned char[width * height * 3];
		if (ComputeAverageImage(Path, MeanImg, width, height, camID, 0, startF, stopF) == 1) //Look up for mean images
		{
			delete[]MeanImg;
			continue;
		}

		double startTime = omp_get_wtime();
		vector<Point2d> *kpts = new vector<Point2d>[stopF - startF + 1];
		unsigned char *ColorImg = new unsigned char[width*height * 3];
		float *colorResponse = new float[width*height];
		double *DImg = new double[width*height], *ImgPara = new double[width*height];
		double *maskSmooth = new double[PatternSize*PatternSize*nscales], *Znssd_reqd = new double[9 * PatternSize*PatternSize];

		for (int frameID = startF; frameID <= stopF; frameID++)
		{
			char Fname[200];  sprintf(Fname, "%s/%08d/%08d_%02d_%02d.png", Path, frameID, frameID, 0, camID);
			DetectRedLaserCorrelationMultiScale(Fname, width, height, MeanImg, kpts[frameID - startF], sigma, PatternSize, nscales, NMS_BW, thresh, false, ColorImg, colorResponse, DImg, ImgPara, maskSmooth, Znssd_reqd);
		}
		printf("\rHDCam %d: %.2f%% (%.2fs) jobs finished....", camID, 100.0, omp_get_wtime() - startTime); printf("\n");

		char Fname[200];  sprintf(Fname, "%s/%02d_%02d.txt", Path, 0, camID);	FILE *fp = fopen(Fname, "w+");
		for (int frameID = startF; frameID <= stopF; frameID++)
			for (int ii = 0; ii < kpts[frameID - startF].size(); ii++)
				fprintf(fp, "%d %f %f\n", frameID, kpts[frameID - startF][ii].x, kpts[frameID - startF][ii].y);
		fclose(fp);

		delete[]MeanImg, delete[]kpts, delete[]ColorImg, delete[]colorResponse, delete[]DImg, delete[]ImgPara, delete[]maskSmooth, delete[]Znssd_reqd;
	}
	printf("%.2f%% (%.2fs) jobs finished....", 100.0, omp_get_wtime() - startTime);


	//VGA images
	width = 640, height = 480, PatternSize = 15, nscales = 3, NMS_BW = 20;

#pragma omp parallel for
	for (int panelID = 15; panelID <= 15; panelID++)
	{
#pragma omp critical
		printf("Averaging VGA panel %d\n", panelID);
		unsigned char *mImg = new unsigned char[width * height * 3];
		for (int camID = 1; camID <= 20; camID++)
			ComputeAverageImage(Path, mImg, width, height, camID, panelID, startF, startF + 200);//compute mean images
		delete[]mImg;
	}

	percent = 10, incre = 10, jobCount = 0;
	startTime = omp_get_wtime();

#pragma omp parallel for
	for (int panelID = 1; panelID <= nPanels; panelID++)
	{
#pragma omp critical
		if (100.0*jobCount / nPanels >= percent)
		{
			printf("/r%.2f%% (%.2fs) jobs finished....\n", 100.0*jobCount / nPanels, omp_get_wtime() - startTime);
			percent += incre;
		}
		jobCount++;

		vector<Point2d> *kpts = new vector<Point2d>[stopF - startF + 1];
		unsigned char *MeanImg = new unsigned char[width * height * 3];
		unsigned char *ColorImg = new unsigned char[width*height * 3];
		float *colorResponse = new float[width*height];
		double *DImg = new double[width*height], *ImgPara = new double[width*height];
		double *maskSmooth = new double[PatternSize*PatternSize*nscales], *Znssd_reqd = new double[9 * PatternSize*PatternSize];

		for (int camID = 1; camID <= 20; camID++)
		{
			for (int frameID = startF; frameID <= stopF; frameID++)
				kpts[frameID - startF].clear();

			if (ComputeAverageImage(Path, MeanImg, width, height, camID, panelID, startF, stopF) == 1)//look up for mean images
				continue;

			for (int frameID = startF; frameID <= stopF; frameID++)
			{
				char Fname[200];  sprintf(Fname, "%s/%08d/%08d_%02d_%02d.png", Path, frameID, frameID, panelID, camID);
				DetectRedLaserCorrelationMultiScale(Fname, width, height, MeanImg, kpts[frameID - startF], sigma, PatternSize, nscales, NMS_BW, thresh, false, ColorImg, colorResponse, DImg, ImgPara, maskSmooth, Znssd_reqd);
			}

			char Fname[200];  sprintf(Fname, "%s/%02d_%02d.txt", Path, panelID, camID);	FILE *fp = fopen(Fname, "w+");
			for (int frameID = startF; frameID <= stopF; frameID++)
				for (int ii = 0; ii < kpts[frameID - startF].size(); ii++)
					fprintf(fp, "%d %f %f\n", frameID, kpts[frameID - startF][ii].x, kpts[frameID - startF][ii].y);
			fclose(fp);
		}

		delete[]MeanImg, delete[]kpts, delete[]ColorImg, delete[]colorResponse, delete[]DImg, delete[]ImgPara, delete[]maskSmooth, delete[]Znssd_reqd;
	}
	printf("%.2f%% (%.2fs) jobs finished....\n", 100.0, omp_get_wtime() - startTime);

	return 0;
}
int BallDetectionDriver(char *dpath)
{
	char Fname[200];

	int noctaves = 2, nPerOctaves = 2, nscales = noctaves*nPerOctaves + 1, basePatternSize = 24;
	double sigma = 1.0, thresh = 0.5;
	vector<KeyPoint> kpts;
	vector<int> ballType;

	int startFrame = 34, stopFrame = 36, camID = 24;
	sprintf(Fname, "%s/%08d_00_%02d.png", dpath, startFrame, camID);
	DetectRGBBallCorrelation(Fname, kpts, ballType, noctaves, nPerOctaves, sigma, basePatternSize, basePatternSize, thresh, true);

	sprintf(Fname, "%s/%08d_00_%02d.txt", dpath, startFrame, camID); FILE *fp = fopen(Fname, "w+");
	for (int kk = 0; kk < 3; kk++)
	{
		if (kpts[kk].response < 0.01)
			continue;
		fprintf(fp, "%d %.2f %.2f %.4f %.1f %d\n", kk, kpts[kk].pt.x, kpts[kk].pt.y, kpts[kk].response, kpts[kk].size, kpts[kk].octave);
	}
	fclose(fp);

	for (int frameID = startFrame + 2; frameID <= stopFrame; frameID += 2)
	{
		/*KeyPoint kpt;
		FILE *fp = fopen("C:/temp/kpts.txt", "r");
		for (int ii = 0; ii < 3; ii++)
		{
		fscanf(fp, "%f %f %f %f %d", &kpt.pt.x, &kpt.pt.y, &kpt.response, &kpt.size, &kpt.octave);
		kpts[ii] = kpt;
		}
		fclose(fp);*/

		//Check if all balls are detected
		int count = 0;
		for (int ii = 0; ii < 3; ii++)
			if (kpts[ii].response > 0.01)
				count++;

		sprintf(Fname, "%s/%08d_00_%02d.png", dpath, frameID, camID);
		if (count == 3)//Track the point
		{
			int width = 1920, height = 1080;
			double *Img = new double[width*height];
			GrabImage(Fname, Img, width, height, 1);

			int InterpAlgo = 1;
			double *ImgPara = new double[width*height];
			Generate_Para_Spline(Img, ImgPara, width, height, InterpAlgo);


			int IntensityProfile[] = { 10, 240 };
			double RingInfo[1] = { 0.9 };
			double *maskSmooth = new double[basePatternSize*basePatternSize*nscales*nscales];

			Point2d pt;
			for (int ii = 0; ii < 3; ii++)
			{
				int PatternSize = kpts[ii].size, hsubset = PatternSize / 2, PatternLength = PatternSize*PatternSize;
				synthesize_concentric_circles_mask(maskSmooth, IntensityProfile, PatternSize, 1, PatternSize, RingInfo, 0, 1);

				pt.x = kpts[ii].pt.x, pt.y = kpts[ii].pt.y;
				double zncc = TMatchingFine_ZNCC(maskSmooth, PatternSize, hsubset - 1, ImgPara, width, height, 1, pt, 0, 1, 0.6, InterpAlgo);
				if (zncc < thresh)
					printf("Cannot refine the %d balls in %s\n", ii + 1, Fname);
				else
					kpts[ii].pt.x = pt.x, kpts[ii].pt.y = pt.y;
			}
		}
		else //Do detection
			DetectRGBBallCorrelation(Fname, kpts, ballType, noctaves, nPerOctaves, sigma, basePatternSize, basePatternSize, thresh, false);

		sprintf(Fname, "%s/%08d_00_%02d.txt", dpath, frameID, camID); fp = fopen(Fname, "w+");
		for (int kk = 0; kk < 3; kk++)
			fprintf(fp, "%.2f %.2f %.4f %.1f %d\n", kpts[kk].pt.x, kpts[kk].pt.y, kpts[kk].response, kpts[kk].size, kpts[kk].octave);
		fclose(fp);
	}

	return 0;
}
int CleanLaserDetection(char *Path)
{
	int nHDs = 30, nPanels = 20, nVGAPanel = 24;
	char Fname[200];

	vector<int>frameID;
	vector<Point2d> pts;
	int id, currentId; double x, y;
	for (int camID = 0; camID < nHDs; camID++)
	{
		frameID.clear(), pts.clear();

		sprintf(Fname, "%s/00_%02d.txt", Path, camID); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("Cannot load %s\n", Fname);
			continue;
		}
		else
			printf("Loaded %s\n", Fname);
		while (fscanf(fp, "%d %lf %lf ", &id, &x, &y) != EOF)
		{
			currentId = frameID.size() - 1;
			if (currentId < 0)
				frameID.push_back(id), pts.push_back(Point2d(x, y));
			else
			{
				if (id != frameID[currentId])
					frameID.push_back(id), pts.push_back(Point2d(x, y));
			}
		}
		fclose(fp);


		sprintf(Fname, "%s/00_%02d.txt", Path, camID); fp = fopen(Fname, "w+");
		for (int ii = 0; ii < frameID.size(); ii++)
			fprintf(fp, "%d %f %f\n", frameID[ii], pts[ii].x, pts[ii].y);
		fclose(fp);
	}

	for (int panelID = 1; panelID <= 20; panelID++)
	{
		for (int camID = 1; camID < 24; camID++)
		{
			frameID.clear(), pts.clear();

			sprintf(Fname, "%s/%02d_%02d.txt", Path, panelID, camID); FILE *fp = fopen(Fname, "r");
			if (fp == NULL)
			{
				printf("Cannot load %s\n", Fname);
				continue;
			}
			else
				printf("Loaded %s\n", Fname);
			while (fscanf(fp, "%d %lf %lf ", &id, &x, &y) != EOF)
			{
				currentId = frameID.size() - 1;
				if (currentId < 0)
					frameID.push_back(id), pts.push_back(Point2d(x, y));
				else
				{
					if (id != frameID[currentId])
						frameID.push_back(id), pts.push_back(Point2d(x, y));
				}
			}
			fclose(fp);


			sprintf(Fname, "%s/%02d_%02d.txt", Path, panelID, camID); fp = fopen(Fname, "w+");
			for (int ii = 0; ii < frameID.size(); ii++)
				fprintf(fp, "%d %f %f\n", frameID[ii], pts[ii].x, pts[ii].y);
			fclose(fp);
		}
	}


	return 0;
}
int GenerateCorresTableLaser(char *Path, int startFHD, int stopFHD, int startFVGA, int stopFVGA)
{
	int nHDs = 30, nPanels = 20, nVGAPanel = 24, nVGAs = nPanels*nVGAPanel;
	char Fname[200];

	int *TableHDFrameID = new int[(stopFHD - startFHD + 1)], *TableVGAFrameID = new int[(stopFVGA - startFVGA + 1)];
	Point2d *TableHD = new Point2d[(stopFHD - startFHD + 1)*nHDs], *TableVGA = new Point2d[(stopFVGA - startFVGA + 1)*nVGAs];

	for (int ii = startFHD; ii <= stopFHD; ii++)
		TableHDFrameID[ii - startFHD] = ii;
	for (int ii = 0; ii <= (stopFHD - startFHD + 1)*nHDs; ii++)
		TableHD[ii] = Point2d(-1, -1);

	for (int ii = startFVGA; ii <= stopFVGA; ii++)
		TableVGAFrameID[ii - startFVGA] = ii;
	for (int ii = 0; ii <= (stopFVGA - startFVGA + 1)*nVGAs; ii++)
		TableVGA[ii] = Point2d(-1, -1);

	int id; double x, y;
	for (int camID = 0; camID < nHDs; camID++)
	{
		sprintf(Fname, "%s/00_%02d.txt", Path, camID); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("Cannot load %s\n", Fname);
			continue;
		}
		else
			;// printf("Loaded %s\n", Fname);
		while (fscanf(fp, "%d %lf %lf ", &id, &x, &y) != EOF)
		{
			if (TableHDFrameID[id - startFHD] != id)
				printf("Error\n");
			TableHD[(id - startFHD)*nHDs + camID].x = x, TableHD[(id - startFHD)*nHDs + camID].y = y;
		}
		fclose(fp);
	}

	for (int panelID = 0; panelID < 20; panelID++)
	{
		for (int camID = 0; camID < 24; camID++)
		{
			int globalCamID = nVGAPanel*panelID + camID;
			sprintf(Fname, "%s/%02d_%02d.txt", Path, panelID + 1, camID + 1); FILE *fp = fopen(Fname, "r");
			if (fp == NULL)
			{
				printf("Cannot load %s\n", Fname);
				continue;
			}
			else
				;// printf("Loaded %s\n", Fname);
			while (fscanf(fp, "%d %lf %lf ", &id, &x, &y) != EOF)
			{
				if (TableVGAFrameID[id - startFVGA] != id)
					printf("Error\n");
				TableVGA[(id - startFVGA)*nVGAs + globalCamID].x = x, TableVGA[(id - startFVGA)*nVGAs + globalCamID].y = y;
			}
			fclose(fp);
		}
	}

	sprintf(Fname, "%s/Correspondences_HD.txt", Path); FILE *fp = fopen(Fname, "w+");
	for (int ii = startFHD; ii <= stopFHD; ii++)
	{
		fprintf(fp, "%d ", TableHDFrameID[ii - startFHD]);
		for (int jj = 0; jj < nHDs; jj++)
			fprintf(fp, "%f %f ", TableHD[(ii - startFHD)*nHDs + jj].x, TableHD[(ii - startFHD)*nHDs + jj].y);
		fprintf(fp, "\n");
	}
	fclose(fp);

	sprintf(Fname, "%s/Correspondences_VGA.txt", Path); fp = fopen(Fname, "w+");
	for (int ii = startFVGA; ii <= stopFVGA; ii++)
	{
		fprintf(fp, "%d ", TableVGAFrameID[ii - startFVGA]);
		for (int jj = 0; jj < nVGAs; jj++)
			fprintf(fp, "%f %f ", TableVGA[(ii - startFVGA)*nVGAs + jj].x, TableVGA[(ii - startFVGA)*nVGAs + jj].y);
		fprintf(fp, "\n");
	}
	fclose(fp);

	printf("Done\n");

	//delete[]TableHD, delete[]TableVGA;
	//delete[]TableHDFrameID, delete[]TableVGAFrameID;

	return 0;
}
int ReadDomeCalib(char *BAfileName, Corpus &CorpusData)
{
	int nHDs = 30, nPanels = 20, nVGAPanel = 24, nCams = nHDs + nPanels*nVGAPanel;
	char Fname[200];

	FILE *fp = fopen(BAfileName, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", BAfileName);
		return false;
	}

	int width, height;
	double fx, fy, skew, u0, v0, r1, r2, r3, t1, t2, p1, p2, rv[3], T[3];

	CorpusData.nCameras = nCams;
	CorpusData.camera = new CameraData[CorpusData.nCameras];

	for (int ii = 0; ii < CorpusData.nCameras; ii++)
	{
		if (fscanf(fp, "%s %d %d", Fname, &width, &height) == EOF)
			break;
		string filename = Fname;
		std::size_t pos1 = filename.find("_");
		string PanelName; PanelName = filename.substr(0, 2);
		const char * str = PanelName.c_str();
		int panelID = atoi(str);

		std::size_t pos2 = filename.find(".jpg");
		string CamName; CamName = filename.substr(pos1 + 1, 2);

		str = CamName.c_str();
		int camID = atoi(str);

		int viewID = panelID == 0 ? camID : nHDs + nVGAPanel*(panelID - 1) + camID - 1;

		CorpusData.camera[viewID].LensModel = RADIAL_TANGENTIAL_PRISM;
		CorpusData.camera[viewID].width = width, CorpusData.camera[viewID].height = height;
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
		AssembleP(CorpusData.camera[viewID]);
		GetRCGL(CorpusData.camera[viewID]);
	}
	fclose(fp);

	return true;
}
int DomeTriangulation(char *Path, Corpus CorpusData, int startF_HD, int stopF_HD, int startF_VGA, int stopF_VGA)
{
	char Fname[200];
	const int nHDs = 30, nPanels = 20, nVGAPanel = 24, nVGAs = nPanels*nVGAPanel;
	const int nHDsUsed = 20;

	Point3d wc;
	//Read Corres TableHD
	int frameID;
	Point2d *TableHD = new Point2d[(stopF_HD - startF_HD + 1)*nHDs];
	int *frame3D_HD = new int[stopF_HD - startF_HD + 1];
	for (int ii = startF_HD; ii <= stopF_HD; ii++)
	{
		frame3D_HD[ii - startF_HD] = ii;
		for (int jj = 0; jj < nHDs; jj++)
			TableHD[(ii - startF_HD)*nHDs + jj].x = -1, TableHD[(ii - startF_HD)*nHDs + jj].y = -1;
	}

	sprintf(Fname, "%s/Correspondences_HD.txt", Path);	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return 1;
	}
	while (fscanf(fp, "%d ", &frameID) != EOF)
	{
		frame3D_HD[frameID - startF_HD] = frameID;
		for (int jj = 0; jj < nHDs; jj++)
			fscanf(fp, "%lf %lf ", &TableHD[(frameID - startF_HD)*nHDs + jj].x, &TableHD[(frameID - startF_HD)*nHDs + jj].y);
	}
	fclose(fp);

	//Triangulate HD cameras:
	Point3d *WC_HD = new Point3d[stopF_HD - startF_HD + 1];
	{
		int minInliers = 999;
		double minInliersPercent = 1.0;
		double maxError = 0.0;
		vector<int>Inliers[1];  Inliers[0].reserve(nHDs);
		vector<int>CamID;
		Point2d* pts = new Point2d[nHDs];
		bool *passed = new bool[nHDs];
		double *A = new double[6 * nHDs], *B = new double[2 * nHDs], *tP = new double[12 * nHDs];
		double P_HD[12 * nHDs];
		vector<double> ReProjectionError; ReProjectionError.reserve(stopF_HD - startF_HD + 1);

		for (int frameID = startF_HD; frameID <= stopF_HD; frameID++)
		{
			WC_HD[frameID - startF_HD] = Point3d(0, 0, 0);
			int nViews = 0;
			CamID.clear();
			for (int jj = 0; jj < nHDsUsed; jj++)
			{
				if (TableHD[(frameID - startF_HD)*nHDs + jj].x > 0 && TableHD[(frameID - startF_HD)*nHDs + jj].y > 0)
				{
					CamID.push_back(jj);
					pts[nViews] = TableHD[(frameID - startF_HD)*nHDs + jj];
					LensCorrectionPoint(&pts[nViews], CorpusData.camera[jj].K, CorpusData.camera[jj].distortion);
					for (int kk = 0; kk < 12; kk++)
						P_HD[12 * nViews + kk] = CorpusData.camera[jj].P[kk];
					nViews++;
				}
			}
			if (nViews < 2)
				continue;
			Inliers[0].clear();
			double avgerror = NviewTriangulationRANSAC(pts, P_HD, &wc, passed, Inliers, nViews, 1, 100, 0.7, 10, A, B, tP, false, false);
			if (!passed[0])
				for (int jj = 0; jj < nHDs; jj++)
					TableHD[(frameID - startF_HD)*nHDs + jj].x = -1, TableHD[(frameID - startF_HD)*nHDs + jj].y = -1;
			else
			{
				int ninlier = 0;
				for (int kk = 0; kk < Inliers[0].size(); kk++)
					if (Inliers[0].at(kk))
						ninlier++;
				if (ninlier < 3)
				{
					printf("Error at frame %d\n", frameID);
					for (int kk = 0; kk < nHDs; kk++)
						TableHD[(frameID - startF_HD)*nHDs + kk].x = -1, TableHD[(frameID - startF_HD)*nHDs + kk].y = -1;
					continue; //Corpus needs NDplus+ points!
				}
				else
					WC_HD[frameID - startF_HD] = wc;

				ReProjectionError.push_back(avgerror);
				double inlierPercent = 1.0*ninlier / Inliers[0].size();
				if (minInliersPercent > inlierPercent)
					minInliersPercent = inlierPercent, minInliers = ninlier;
				if (maxError < avgerror)
					maxError = avgerror;

				for (int kk = 0; kk < nViews; kk++)
					if (Inliers[0].at(kk) == 0)
						TableHD[(frameID - startF_HD)*nHDs + CamID[kk]].x = -1, TableHD[(frameID - startF_HD)*nHDs + CamID[kk]].y = -1;
			}
		}
		double miniE = *min_element(ReProjectionError.begin(), ReProjectionError.end()), maxiE = *max_element(ReProjectionError.begin(), ReProjectionError.end());
		double avgE = MeanArray(ReProjectionError), stdE = sqrt(VarianceArray(ReProjectionError, avgE));
		printf("Reprojection error: Min: %.2f Max: %.2f Mean: %.2f Std: %.2f with min nliners percent %.2f (%d pts)\n", miniE, maxiE, avgE, stdE, minInliersPercent, minInliers);
		delete[]passed, delete[]pts, delete[]A, delete[]B, delete[]tP;// , delete[] P_HD;
	}
	sprintf(Fname, "%s/C0_0.txt", Path);	fp = fopen(Fname, "w+");
	for (int ii = startF_HD; ii <= stopF_HD; ii++)
	{
		if (frame3D_HD[ii - startF_HD] != -1)
		{
			if (abs(WC_HD[ii - startF_HD].x) + abs(WC_HD[ii - startF_HD].y) + abs(WC_HD[ii - startF_HD].z) < 0.01)
				continue;
			fprintf(fp, "%d %f %f %f\n", frame3D_HD[ii - startF_HD], WC_HD[ii - startF_HD].x, WC_HD[ii - startF_HD].y, WC_HD[ii - startF_HD].z);
		}
	}
	fclose(fp);

	/*sprintf(Fname, "%s/Correspondences_HD.txt", Path); fp = fopen(Fname, "w+");
	for (int ii = startF_HD; ii <= stopF_HD; ii++)
	{
	fprintf(fp, "%d ", frame3D_HD[ii - startF_HD]);
	for (int jj = 0; jj < nHDs; jj++)
	fprintf(fp, "%f %f ", TableHD[(ii - startF_HD)*nHDs + jj].x, TableHD[(ii - startF_HD)*nHDs + jj].y);
	fprintf(fp, "\n");
	}
	fclose(fp);*/


	//VGA
	Point2d *TableVGA = new Point2d[(stopF_VGA - startF_VGA + 1)*nVGAs];
	int *frame3D_VGA = new int[stopF_VGA - startF_VGA + 1];
	for (int ii = startF_VGA; ii <= stopF_VGA; ii++)
	{
		frame3D_VGA[ii - startF_VGA] = ii;
		for (int jj = 0; jj < nVGAs; jj++)
			TableVGA[(ii - startF_VGA)*nVGAs + jj].x = -1.0, TableVGA[(ii - startF_VGA)*nVGAs + jj].y = -1.0;
	}

	sprintf(Fname, "%s/Correspondences_VGA.txt", Path);	fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return 1;
	}
	while (fscanf(fp, "%d ", &frameID) != EOF)
	{
		frame3D_VGA[frameID - startF_VGA] = frameID;
		for (int jj = 0; jj < nVGAs; jj++)
			fscanf(fp, "%lf %lf ", &TableVGA[(frameID - startF_VGA)*nVGAs + jj].x, &TableVGA[(frameID - startF_VGA)*nVGAs + jj].y);
	}
	fclose(fp);

	Point3d *WC_VGA = new Point3d[stopF_VGA - startF_VGA + 1];
	{
		int minInliers = 999;
		double minInliersPercent = 1.0;
		double maxError = 0.0;
		vector<int>Inliers[1];  Inliers[0].reserve(nVGAs);
		vector<int>CamID;
		Point2d* pts = new Point2d[nVGAs];
		bool *passed = new bool[nVGAs];
		double *P_VGA = new double[12 * nVGAs], *A = new double[6 * nVGAs], *B = new double[2 * nVGAs], *tP = new double[12 * nVGAs];
		vector<double> ReProjectionError; ReProjectionError.reserve(stopF_VGA - startF_VGA + 1);
		for (int frameID = startF_VGA; frameID <= stopF_VGA; frameID++)
		{
			WC_VGA[frameID - startF_VGA] = Point3d(0, 0, 0);
			int nViews = 0;
			CamID.clear();
			for (int jj = 0; jj < nVGAs; jj++)
			{
				if (TableVGA[(frameID - startF_VGA)*nVGAs + jj].x > 0 && TableVGA[(frameID - startF_VGA)*nVGAs + jj].y > 0)
				{
					CamID.push_back(jj);
					pts[nViews] = TableVGA[(frameID - startF_VGA)*nVGAs + jj];
					LensCorrectionPoint(&pts[nViews], CorpusData.camera[jj + nHDs].K, CorpusData.camera[jj + nHDs].distortion);
					for (int kk = 0; kk < 12; kk++)
						P_VGA[12 * nViews + kk] = CorpusData.camera[jj + nHDs].P[kk];
					nViews++;
				}
			}
			if (nViews < 2)
				continue;
			Inliers[0].clear();
			double avgerror = NviewTriangulationRANSAC(pts, P_VGA, &wc, passed, Inliers, nViews, 1, 100, 0.7, 4, A, B, tP);
			if (!passed[0])
				for (int jj = 0; jj < nVGAs; jj++)
					;// TableVGA[(frameID - startF_VGA)*nVGAs + jj].x = -1, TableVGA[(frameID - startF_VGA)*nVGAs + jj].y = -1;
			else
			{
				int ninlier = 0;
				for (int kk = 0; kk < Inliers[0].size(); kk++)
					if (Inliers[0].at(kk))
						ninlier++;
				if (ninlier < 3)
				{
					printf("Error at frame %d\n", frameID);
					for (int kk = 0; kk < nVGAs; kk++)
						TableVGA[(frameID - startF_VGA)*nVGAs + kk].x = -1, TableVGA[(frameID - startF_VGA)*nVGAs + kk].y = -1;
					continue; //Corpus needs NDplus+ points!
				}
				else
					WC_VGA[frameID - startF_VGA] = wc;

				ReProjectionError.push_back(avgerror);
				double inlierPercent = 1.0*ninlier / Inliers[0].size();
				if (minInliersPercent > inlierPercent)
					minInliersPercent = inlierPercent, minInliers = ninlier;
				if (maxError < avgerror)
					maxError = avgerror;

				for (int kk = 0; kk < nViews; kk++)
					if (Inliers[0].at(kk) == 0)
						TableVGA[(frameID - startF_VGA)*nVGAs + CamID[kk]].x = -1, TableVGA[(frameID - startF_VGA)*nVGAs + +CamID[kk]].y = -1;
			}
		}
		double miniE = *min_element(ReProjectionError.begin(), ReProjectionError.end()), maxiE = *max_element(ReProjectionError.begin(), ReProjectionError.end());
		double avgE = MeanArray(ReProjectionError), stdE = sqrt(VarianceArray(ReProjectionError, avgE));
		printf("Reprojection error VGA: Min: %.2f Max: %.2f Mean: %.2f Std: %.2f with min nliners percent %.2f (%d pts)\n", miniE, maxiE, avgE, stdE, minInliersPercent, minInliers);

		delete[]passed, delete[]pts, delete[]A, delete[]B, delete[]tP, delete[]P_VGA;
	}

	sprintf(Fname, "%s/C1_0.txt", Path);	fp = fopen(Fname, "w+");
	for (int ii = startF_VGA; ii <= stopF_VGA; ii++)
	{
		if (frame3D_VGA[ii - startF_VGA] != -1)
			if (abs(WC_VGA[ii - startF_VGA].x) + abs(WC_VGA[ii - startF_VGA].y) + abs(WC_VGA[ii - startF_VGA].z) < 0.01)
				continue;
		fprintf(fp, "%d %f %f %f\n", frame3D_VGA[ii - startF_VGA], WC_VGA[ii - startF_VGA].x, WC_VGA[ii - startF_VGA].y, WC_VGA[ii - startF_VGA].z);
	}
	fclose(fp);

	/*sprintf(Fname, "%s/Correspondences_VGA.txt", Path); fp = fopen(Fname, "w+");
	for (int ii = startF_VGA; ii <= stopF_VGA; ii++)
	{
	fprintf(fp, "%d ", frame3D_VGA[ii - startF_VGA]);
	for (int CamID = 0; CamID < nVGAs; CamID++)
	fprintf(fp, "%f %f ", TableVGA[(frameID - startF_VGA)*nVGAs + CamID].x, TableVGA[(frameID - startF_VGA)*nVGAs + CamID].y);
	fprintf(fp, "\n");
	}
	fclose(fp);*/

	return 0;
}
int DomeSyncGroundTruth()
{
	char Path[] = "E:/DomeLaser5/beforeBA";
	int startHD = 1, stopHD = 4210, startVGA = 1, stopVGA = 3601;
	Corpus CorpusData;

	//CleanLaserDetection("E:/DomeLaser5/X/2");
	//GenerateCorresTableLaser("E:/DomeLaser5/X/2", startHD, stopHD, startVGA, stopVGA);
	//sprintf(Fname, "%s/BA_Camera_AllParams_after.txt", Path); loadBundleAdjustedNVMResults(Fname, CorpusData);
	//sprintf(Fname, "%s/calib.txt", Path); ReadDomeCalib(Fname, CorpusData);
	//double scale =  120.0 / Distance3D(CorpusData.camera[30].camCenter, CorpusData.camera[31].camCenter);
	//sprintf(Fname, "%s/BA_Camera_AllParams_after.txt", Path);
	//ReSaveBundleAdjustedNVMResults(Fname, CorpusData, scale);

	//sprintf(Fname, "%s/BA_Camera_AllParams_after.txt", Path);
	//loadBundleAdjustedNVMResults(Fname, CorpusData);
	//DomeTriangulation(Path, CorpusData, startHD, stopHD, startVGA, stopVGA);
	//visualizationDriver(Path, 510, 0, stopHD, false, false, false, false, false, 0);

	//BundleAdjustDomeTableCorres(Path, startHD, stopHD, startVGA, stopVGA, false, true, false, false, true, false, true);
	//BundleAdjustDomeMultiNVM("C:/temp/Dome", 3, 81576, true, true, false, true);
	//visualizationDriver("C:/temp/Dome", 510, 0, 4210, false, false, false, false, false, 0);

	double Offset[2];;
	DomeLeastActionSyncBruteForce3D(Path, 1, Offset);
	visualizationDriver(Path, 510, -80, 80, true, false, false, true, false, false, 0);

	return 0;
}

int SimulateRollingShutterCameraAnd2DPointsOnMoCapForCalib(char *Path, int npts, double *Intrinsic, double *distortion, int width, int height, double radius = 5e3, bool saveGT3D = true, bool show2DImage = false, int Rate = 1, double Noise2D = 2.0)
{
	char Fname[200];
	double noise3D_CamShake = 20 / 60;
	double x, y, z, cx = 0, cy = 0, cz = 0;

	int nframes;
	Point2d pt;
	Point3d p3d;
	vector<Point3d> allXYZ;
	for (int pid = 0; pid < npts; pid++)
	{
		sprintf(Fname, "%s/Track3D/%d.txt", Path, pid); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
			printf("Cannot load %s\n", Fname);
		while (fscanf(fp, "%lf %lf %lf", &x, &y, &z) != EOF)
		{
			if (pid == 0)
				cx += x, cy += y, cz += z;
			else
				x -= cx, y -= cy, z -= cz;

			allXYZ.push_back(Point3d(x, y, z));
		}
		fclose(fp);

		if (pid == 0)
		{
			nframes = (int)allXYZ.size();
			cx /= nframes, cy /= nframes, cz /= nframes;
		}
	}


	CameraData *Camera = new CameraData[nframes];
	for (int frameID = 0; frameID < nframes; frameID++)
	{
		Camera[frameID].valid = true;
		GenerateCamerasExtrinsicOnCircle(Camera[frameID], 180.0*frameID / nframes / 180.0 * Pi, radius, Point3d(cx, cy - 700, cz), Point3d(cx, cy - 700, cz), Point3d(0, 0, 0));
		SetIntrinisc(Camera[frameID], Intrinsic);
		GetKFromIntrinsic(Camera[frameID]);
		for (int ii = 0; ii < 7; ii++)
			Camera[frameID].distortion[ii] = distortion[ii];
		AssembleP(Camera[frameID]);
	}


	vector<int> UsedFrames;
	sprintf(Fname, "%s/Intrinsic_0.txt", Path); FILE *fp = fopen(Fname, "w+");
	for (int frameID = 0; frameID < nframes; frameID++)
	{
		if (!Camera[frameID].valid)
			continue;

		UsedFrames.push_back(frameID);
		fprintf(fp, "%d 0 %d %d ", frameID, width, height);
		for (int ii = 0; ii < 5; ii++)
			fprintf(fp, "%.2f ", Camera[frameID].intrinsic[ii]);
		for (int ii = 0; ii < 7; ii++)
			fprintf(fp, "%.6f ", distortion[ii]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	sprintf(Fname, "%s/CamPose_0.txt", Path); fp = fopen(Fname, "w+");
	for (int frameID = 0; frameID < nframes; frameID++)
	{
		if (!Camera[frameID].valid)
			continue;

		GetRCGL(Camera[frameID]);

		fprintf(fp, "%d ", frameID);
		for (int jj = 0; jj < 16; jj++)
			fprintf(fp, "%.16f ", Camera[frameID].Rgl[jj]);
		for (int jj = 0; jj < 3; jj++)
			fprintf(fp, "%.16f ", Camera[frameID].camCenter[jj]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	vector<ImgPtEle> *PerCam_UV = new vector<ImgPtEle>[(int)allXYZ.size()];
	for (int pid = 0; pid < (int)allXYZ.size(); pid++)
	{
		for (int frameID = 0; frameID < nframes; frameID++)
		{
			int iter;
			CameraData CamI;
			double relativeDiff, theta = 1.0*frameID / nframes, subframe = 0.0, osubframe = subframe;
			for (iter = 0; iter < 20; iter++)
			{
				GenerateCamerasExtrinsicOnCircle(CamI, 180.0*theta / 180.0 * Pi, radius, Point3d(cx, cy - 700, cz), Point3d(cx, cy - 700, cz), Point3d(0, 0, 0));
				SetIntrinisc(CamI, Intrinsic);
				GetKFromIntrinsic(CamI);
				for (int ii = 0; ii < 7; ii++)
					CamI.distortion[ii] = distortion[ii];
				AssembleP(CamI);
				ProjectandDistort(allXYZ[pid], &pt, CamI.P, CamI.K, CamI.distortion);
				subframe = pt.y / height;
				relativeDiff = abs((subframe - osubframe) / subframe);
				if (relativeDiff < 1.e-9)
					break;
				osubframe = subframe;
				theta = (subframe + frameID) / nframes;
			}
			if (iter > 19)
				continue;
			if (subframe<0.0 || subframe>1.0)
				continue;

			Point2d Noise(gaussian_noise(0.0, Noise2D), gaussian_noise(0.0, Noise2D));
			if (Noise.x > 3.0*Noise2D)
				Noise.x = 3.0*Noise2D;
			else if (Noise.x < -3.0 *Noise2D)
				Noise.x = -3.0*Noise2D;
			if (Noise.y > 3.0*Noise2D)
				Noise.y = 3.0*Noise2D;
			else if (Noise.y < -3.0 *Noise2D)
				Noise.y = -3.0*Noise2D;
			pt.x += Noise.x, pt.y += Noise.y;

			ImgPtEle ptEle;
			ptEle.pt2D = pt, ptEle.viewID = 0, ptEle.frameID = frameID, ptEle.imWidth = width, ptEle.imHeight = height;
			ptEle.pt3D = allXYZ[pid];
			ptEle.timeStamp = frameID;

			for (int ii = 0; ii < 9; ii++)
				ptEle.R[ii] = CamI.R[ii];
			for (int ii = 0; ii < 3; ii++)
				ptEle.T[ii] = CamI.T[ii];
			PerCam_UV[pid].push_back(ptEle);
		}
	}

	sprintf(Fname, "%s/RollingShutterPose", Path), makeDir(Fname);
	sprintf(Fname, "%s/RollingShutterPose/CamPose.txt", Path); fp = fopen(Fname, "w+");
	for (int pid = 0; pid < (int)allXYZ.size(); pid++)
	{
		for (int fid = 0; fid < (int)PerCam_UV[pid].size(); fid++)
		{
			double twist[6];  getTwistFromRT(PerCam_UV[pid][fid].R, PerCam_UV[pid][fid].T, twist);
			fprintf(fp, "%.8f ", PerCam_UV[pid][fid].frameID + PerCam_UV[pid][fid].pt2D.y / height);
			for (int ii = 0; ii < 6; ii++)
				fprintf(fp, "%.16f ", twist[ii]);
			fprintf(fp, "\n");
		}
	}
	fclose(fp);


	//Generate rolling shutter 2d data
	bool binary = true;
	if (binary)
	{
		sprintf(Fname, "%s/VideoPose_Optim_Input.dat", Path);
		ofstream fout; fout.open(Fname, ios::binary);
		int n = (int)allXYZ.size();
		fout.write(reinterpret_cast<char *>(&n), sizeof(int));
		for (int pid = 0; pid < (int)allXYZ.size(); pid++)
		{
			int nvisibles = PerCam_UV[pid].size();
			float X = (float)allXYZ[pid].x, Y = (float)allXYZ[pid].y, Z = (float)allXYZ[pid].z;

			fout.write(reinterpret_cast<char *>(&nvisibles), sizeof(int));
			fout.write(reinterpret_cast<char *>(&X), sizeof(float));
			fout.write(reinterpret_cast<char *>(&Y), sizeof(float));
			fout.write(reinterpret_cast<char *>(&Z), sizeof(float));
			for (int fid = 0; fid < (int)PerCam_UV[pid].size(); fid += Rate)
			{
				float u = (float)PerCam_UV[pid][fid].pt2D.x, v = (float)PerCam_UV[pid][fid].pt2D.y, s = (float)1.0;
				fout.write(reinterpret_cast<char *>(&PerCam_UV[pid][fid].frameID), sizeof(int));
				fout.write(reinterpret_cast<char *>(&u), sizeof(float));
				fout.write(reinterpret_cast<char *>(&v), sizeof(float));
				fout.write(reinterpret_cast<char *>(&s), sizeof(float));
			}
		}
		fout.close();
	}
	else
	{
		sprintf(Fname, "%s/VideoPose_Optim_Input.txt", Path);
		fp = fopen(Fname, "w+");
		fprintf(fp, "%d \n", (int)allXYZ.size());
		for (int pid = 0; pid < (int)allXYZ.size(); pid++)
		{
			int nvisibles = PerCam_UV[pid].size();
			float X = (float)allXYZ[pid].x, Y = (float)allXYZ[pid].y, Z = (float)allXYZ[pid].z;

			fprintf(fp, "%d %.4f %.4f %.4f ", nvisibles, X, Y, Z);
			for (int fid = 0; fid < (int)PerCam_UV[pid].size(); fid += Rate)
			{
				float u = (float)PerCam_UV[pid][fid].pt2D.x, v = (float)PerCam_UV[pid][fid].pt2D.y, s = (float)1.0;
				fprintf(fp, "%d %.3f %.3f %.1f ", PerCam_UV[pid][fid].frameID, u, v, s);
			}
			fprintf(fp, "\n");
		}
		fclose(fp);
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

		for (int frameID = 0; frameID < nframes; frameID++)
		{
			displayImg = Img.clone();
			for (int pid = 0; pid < (int)allXYZ.size(); pid++)
			{
				for (int ii = 0; ii < (int)PerCam_UV[pid].size(); ii++)
				{
					if (frameID == PerCam_UV[pid][ii].frameID)
					{
						circle(displayImg, PerCam_UV[pid][ii].pt2D, 4, colors[pid % 9], 2, 8, 0);
						break;
					}
				}
			}

			sprintf(Fname, "frame %d", frameID);
			CvPoint text_origin = { width / 30, height / 30 };
			putText(displayImg, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 3.0 * 640 / Img.cols, CV_RGB(255, 0, 0), 2);
			imshow("Image", displayImg);
			waitKey(10);
		}
	}
	return 0;
}
int TestRollingShutterCalibOnSyntheticData(char *Path, int se3)
{
	const int nCams = 10, npts = 2, width = 1920, height = 1080, startFrame = 0, stopFrame = 539;
	double Intrinsic[5] = { 1500, 1500, 0, 960, 540 }, distortion[7] = { 0, 0, 0, 0, 0, 0, 0 }, radius = 3000, noise2D = 0.0;
	//SimulateRollingShutterCameraAnd2DPointsOnMoCapForCalib(Path, npts, Intrinsic, distortion, width, height, radius, 1, 1, noise2D);

	VideoSplineRSBA(Path, startFrame, stopFrame, 0, 1, 0, 0, 10.0, 2, 4, se3 == 1, false);
	if (se3 == 1)
		Pose_se_BSplineInterpolation("E:/SimRollingShutter/CamPoseS_se3_0.txt", "E:/SimRollingShutter/CamPose_se3_1.txt", stopFrame, "E:/SimRollingShutter/CamPose_se3_2.txt");
	else
		Pose_se_BSplineInterpolation("E:/SimRollingShutter/CamPoseS_so3_0.txt", "E:/SimRollingShutter/CamPose_so3_1.txt", stopFrame, "E:/SimRollingShutter/CamPose_so3_2.txt");

	//double lamda = 0.2;
	//VideoDCTRSBA(Path, startFrame, stopFrame, 0, 1, 0, 1, 1,100.0, 2, lamda, false);
	//Pose_se_DCTInterpolation("E:/SimRollingShutter2/CamPoseDCT_0.txt", "E:/SimRollingShutter2/CamPose_1.txt", stopFrame - startFrame + 1);

	//visualizationDriver(Path, 2, 1, stopFrame, true, false, true, false, false, 1);

	return 0;
}

int SpatialTemporalCalibInTheWildDriver(char *Path, int nCams, int startF, int stopF, int increF, bool debug = false)
{
	//0. Obtain spatial calibration info for each cameras

	//1. Audio sync or very rough manual sync
	int FrameOffset[] = { 39.781, 32.813, 3.019, 9.236, 0.000, 18.297, 32.209 }; //in time delay format: f = f_ref + offset

	//2. Detect and match feature points
	int HistogrameEqual = 0, distortionCorrected = 0, OulierRemoveTestMethod = 2, ninlierThesh = 40, //fmat test
		NViewPlus = 3, LensType = RADIAL_TANGENTIAL_PRISM;
	double ratioThresh = 0.6, reprojectionThreshold = 5,
		minScale = 2.0, maxScale = 7.5; //too small features are likely to be noise, too large features are likely to be very unstable

	/*for (int timeID = startF; timeID <= stopF; timeID += increF)
	{
		//#ifdef _WIN32
		//GeneratePointsCorrespondenceMatrix_SiftGPU(Path, nCams, timeID, HistogrameEqual, ratioThresh, FrameOffset);
		//#else
		GeneratePointsCorrespondenceMatrix_CPU(Path, nCams, timeID, HistogrameEqual, ratioThresh, FrameOffset, 2);
		//#endif

#pragma omp parallel
		{
#pragma omp for nowait
			for (int jj = 0; jj < nCams - 1; jj++)
			{
				for (int ii = jj + 1; ii < nCams; ii++)
				{
					if (OulierRemoveTestMethod == 1)
						EssentialMatOutliersRemove(Path, timeID, jj, ii, nCams, 0, ninlierThesh, distortionCorrected, false);
					if (OulierRemoveTestMethod == 2)
						FundamentalMatOutliersRemove(Path, timeID, jj, ii, ninlierThesh, LensType, distortionCorrected, false, nCams, -1, FrameOffset);
				}
			}
		}
		GenerateMatchingTable(Path, nCams, timeID);
	}
	GetPutativeMatchesForEachView(Path, nCams, startF, stopF, increF, Point2d(minScale, maxScale), NViewPlus, FrameOffset);
	*/
	//3. Track all the features: NEED TO INCORPERATE SIFT DESCRIPTOR TO REMOVE BAD TRACKs
	int fps = 60, TrackTime = 2;
	int  WinSize = 31, nWins = 3, WinStep = 3, PyrLevel = 5;
	double MeanSSGThresh = 500;

	/*omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for
	for (int timeID = startF; timeID <= stopF; timeID += increF)
	{
		printf("***Working on frame %d***\n", timeID);
		for (int cid = 0; cid < nCams; cid++)
		{
			printf("***Working on view %d***\n", cid);
			int npts = TrackAllPointsWithRefTemplateDriver(Path, cid, timeID, fps, TrackTime, WinSize, nWins, WinStep, PyrLevel, MeanSSGThresh);
			if (debug)
				VisualizeTracking(Path, cid, timeID, fps, TrackTime, npts);
			printf("***Done with view %d***\n\n", cid);
		}
		printf("***Finish frame %d***\n", timeID);
	}
	return 0;*/

	int fid, nm;
	vector<Point2i> matches;
	char Fname[200];  sprintf(Fname, "%s/Dynamic/nMatches.txt", Path); 	FILE *fp = fopen(Fname, "r");
	while (fscanf(fp, "%d %d ", &fid, &nm) != EOF)
		matches.push_back(Point2i(fid, nm));
	fclose(fp);

	//4. Remove sure stationary features via triangulation and 3D statistic
	vector<int> SelectedCameras, vFrameOffset;
	for (int ii = 0; ii < nCams; ii++)
		SelectedCameras.push_back(ii), vFrameOffset.push_back(FrameOffset[ii]);

	for (int timeID = startF; timeID <= stopF; timeID += increF)
		for (int jj = 0; jj < (int)matches.size(); jj++)
			if (timeID == matches[jj].x)
				ReconAndClassifyPointsFromTriangulation(Path, SelectedCameras, matches[jj].y, vFrameOffset, timeID - TrackTime*fps, timeID + TrackTime*fps, timeID, 3, 10.0, 0.75, 30);
	return 0;

	//5. Refine sync using geometric constraint
	double FrameLevelOffsetInfo[4];
	vector<int>RealStartFrame;
	for (int timeID = startF; timeID <= stopF; timeID += increF)
		RealStartFrame.push_back(timeID);

	GeometricConstraintSyncDriver2(Path, nCams, 10000, RealStartFrame, 0, 600, 10 * 6, true, FrameLevelOffsetInfo, false);

	return 0;
}
int main(int argc, char** argv)
{
	srand(0);//srand(time(NULL));
	char Path[] = "E:/Sim2", Fname[200], Fname2[200];

	TestLeastActionOnSyntheticData(Path, 1);
	return 0;
	///	ExtractVideoFrames(Path, 5, 1, 3);
	//	return 0;
	//SpatialTemporalCalibInTheWildDriver(Path, 7, atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
	//SpatialTemporalCalibInTheWildDriver(Path, 7, 30, 30, 30);

	int fps = 60, TrackTime = 2;
	int  WinSize = 31, nWins = 3, WinStep = 3, PyrLevel = 5;
	double MeanSSGThresh = 300;
	//int npts = TrackAllPointsWithRefTemplateDriver(Path, atoi(argv[1]), atoi(argv[2]), fps, TrackTime, WinSize, nWins, WinStep, PyrLevel, MeanSSGThresh);
	//VisualizeTracking(Path, atoi(argv[1]), atoi(argv[2]), fps, TrackTime, 5000, 0);
	visualizationDriver(Path, 7, 0, 600, true, false, false, false, false, false, 0);
	return 0;
	/*{
	Mat img = imread("E:/Shirt/0/140.png", CV_LOAD_IMAGE_GRAYSCALE);
	KeyPoint key; key.pt = Point2f(1306.33410000000, 287.756000000000);
	float desc[128];
	ComputeFeatureScaleAndDescriptor(img, key, desc);
	}*/
	/*{
		int startF = 140, npts = 7964, fps = 60;
		int pid, nf, fid;
		int trueStartF[4] = { 140, 136, 137, 140 };
		Point2f uv;
		for (int viewID = 0; viewID < 4; viewID++)
		{
		vector<Point2f> *ForeTrackUV = new vector<Point2f>[7964], *BackTrackUV = new vector<Point2f>[7964];

		//Write data
		sprintf(Fname, "%s/Track2D/FT_%d_%d.txt", Path, viewID, startF); FILE *fp = fopen(Fname, "r");
		while (fscanf(fp, "%d %d ", &pid, &nf) != EOF)
		{
		for (int ii = 0; ii < nf; ii++)
		{
		fscanf(fp, "%d %f %f ", &fid, &uv.x, &uv.y);
		ForeTrackUV[pid].push_back(uv);
		}
		}
		fclose(fp);

		sprintf(Fname, "%s/Track2D/BT_%d_%d.txt", Path, viewID, startF);; fp = fopen(Fname, "r");
		while (fscanf(fp, "%d %d ", &pid, &nf) != EOF)
		{
		for (int ii = 0; ii < nf; ii++)
		{
		fscanf(fp, "%d %f %f ", &fid, &uv.x, &uv.y);
		BackTrackUV[pid].push_back(uv);
		}
		}
		fclose(fp);

		sprintf(Fname, "%s/Track2D/%d_%d.txt", Path, viewID, startF); fp = fopen(Fname, "w+");
		for (int pid = 0; pid < npts; pid++)
		{
		if ((int)BackTrackUV[pid].size() + (int)ForeTrackUV[pid].size() < fps / 3)
		continue;
		else
		{
		fprintf(fp, "%d %d ", pid, (int)BackTrackUV[pid].size() + ForeTrackUV[pid].size());
		for (int fid = 0; fid < (int)ForeTrackUV[pid].size(); fid++)
		fprintf(fp, "%d %.4f %.4f ", trueStartF[viewID] + fid, ForeTrackUV[pid][fid].x, ForeTrackUV[pid][fid].y);
		for (int fid = 0; fid < (int)BackTrackUV[pid].size(); fid++)
		fprintf(fp, "%d %.4f %.4f ", trueStartF[viewID] - fid, BackTrackUV[pid][fid].x, BackTrackUV[pid][fid].y);
		fprintf(fp, "\n");
		}
		}
		fclose(fp);
		delete[]ForeTrackUV, delete[]BackTrackUV;
		}
		}*/
	//GenerateVisibilityImage(Path, 4, 140, 120 * 2, 7964);
	//SiftOpenCVPair("C:/temp/X/Corpus/0.png", "C:/temp/X/Corpus/5.png", 0.7, 1.0);
	//SiftGPUPair("C:/temp/X/0.jpg", "C:/temp/X/5.jpg", 0.7, 1.0);
	//vfFeatPair("E:/Shirt/Corpus/0.png", "E:/Shirt/Corpus/1.png", 0.8, 1.0);
	//SiftMatch("E:/Shirt/Corpus", "0", "3", 0.7, 1.0);
	//SpatialTemporalCalibration(Path, 8, 1, 650, 6);
	/*vector<int> SelectedCameras; SelectedCameras.push_back(0); SelectedCameras.push_back(1);
	vector<int> CameraID, DelayInfoVector;
	sprintf(Fname, "%s/FGeoSync.txt", Path);	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
	printf("Cannot load %s\n. Abort!", Fname);
	abort();
	}
	int selected; double offsetValue;
	while (fscanf(fp, "%d %lf ", &selected, &offsetValue) != EOF)
	CameraID.push_back(selected), DelayInfoVector.push_back(offsetValue);
	fclose(fp);

	int DelayOffset[10];
	for (int ii = 0; ii < SelectedCameras.size(); ii++)
	DelayOffset[ii] = DelayInfoVector[SelectedCameras[ii]];
	TriangulatePointsFromCalibratedCameras2(Path, SelectedCameras, DelayOffset, 1, 650, 25, 0);*/
	//TrackLocalizedCameraSIFT(Path, 0, 16, 15, Point2d(40.0, 1.0), 3.0, 1);
	//TestRollingShutterCalibOnSyntheticData(Path, atoi(argv[1]));
	//TestRollingShutterCalibOnSyntheticData(Path, 1);
	//TestRollingShutterCalibOnSyntheticData(Path, 0);
	//CheckerBoardDetection(Path, 0, 0, 150, 11, 8);
	//SingleCameraCalibration(Path, 0, 1, 132, 11, 8, true, 1, 50.8, 1);
	//GenerateVisualSFMinput("E:/RollingShutter3/0/Corner", 0, 131, 88);
	//ReCalibratedFromGroundTruthCorrespondences(Path, 0, 0, 131, 88, atoi(argv[1]));
	//TestMotionPrior3DDriver("D:/cmu_mocap");
	//Pose_se_BSplineInterpolation("E:/SimRollingShutter/_CamPoseS_0.txt", "E:/SimRollingShutter/__CamPose_0.txt", 500, "E:/SimRollingShutter/CamPose_1.txt");
	//visualizationDriver(Path, 2, 1, 500, true, false, true, false, false, 1);
	//CheckerBoardMultiviewSpatialTemporalCalibration(Path, 7, 0, 300, -1);
	//visualizationDriver(Path, 10, 0, 540, false, false, true, false, false, 0);
	//visualizationDriver(Path, 10, 0, 519, false, false, true, true, false, 0);
	//TestLeastActionOnSyntheticData(Path, atoi(argv[1]));
	//ShowSyncLoad(Path, "FMotionPriorSync", Path, 7);
	//RenderSuperCameraFromMultipleUnSyncedCamerasA(Path, 0, 1200, 1, false);
	//TestRollingShutterSpatialTemporalReconOnSyntheticData(Path);
	//return 0;

	int mode = atoi(argv[1]);
	if (mode == 0) //Step 1: sync all sequences with audio
	{
#ifdef _WINDOWS
		int computeSync = atoi(argv[2]);
		if (computeSync == 1)
		{
			int pair = atoi(argv[3]);
			int minSegLength = 3e6;
			if (pair == 1)
			{
				int srcID1 = atoi(argv[4]), srcID2 = atoi(argv[5]);
				double fps1 = atof(argv[6]), fps2 = atof(argv[7]), minZNCC = atof(argv[8]);

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
				double	minZNCC = atof(argv[4]);
				int minSegLength = 20e6;

				int nvideos = atoi(argv[5]);
				int camID[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
				double fps[] = { 60.01, 60.01, 60.01, 60.01, 60.01, 60.01, 59.99 };//to avoid drift at this stage
				double offset, ZNCC;

				sprintf(Fname, "%s/audioSync.txt", Path); 	FILE *fp = fopen(Fname, "w+");
				for (int jj = 0; jj < nvideos - 1; jj++)
				{
					for (int ii = jj + 1; ii < nvideos; ii++)
					{
						sprintf(Fname, "%s/%d/audio.wav", Path, camID[jj]);
						sprintf(Fname2, "%s/%d/audio.wav", Path, camID[ii]);
						if (SynAudio(Fname, Fname2, 1.0, 1.0, minSegLength, offset, ZNCC, minZNCC) != 0)
						{
							printf("Between %d and %d: not succeed\n\n", camID[jj], camID[ii]);
							fprintf(fp, "%d %d %.4f\n", camID[jj], camID[ii], 1000.0*rand());
						}
						else
							fprintf(fp, "%d %d %.4f\n", camID[jj], camID[ii], -offset);
					}
				}
				fclose(fp);

				PrismMST(Path, "audioSync", nvideos);
				AssignOffsetFromMST(Path, "audioSync", nvideos, NULL, fps);
			}
		}
		else
		{
			int nvideos = atoi(argv[3]);
			ShowSyncLoad(Path, argv[4], Path, nvideos, 1.0);
		}
#else
		printf("This mode is not supported in Linux\n");
#endif
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
		sprintf(Fname, "%s/Corpus", Path);

		//int nviews = atoi(argv[2]), NPplus = atoi(argv[3]), ShutterModel = atoi(argv[4]), LossType = atoi(argv[5]);
		//double threshold = atof(argv[6]);

		int nviews = 5, NPplus = 2, ShutterModel = 0, LossType = 0;
		double threshold = 5.0;
		bool sharedIntrinisc = false, fixIntrinsic = false, fixDistortion = false, fixPose = false, distortionCorrected = false, doubleRefinement = true;

		RefineVisualSfM(Fname, nviews, NPplus, ShutterModel, threshold, sharedIntrinisc, fixIntrinsic, fixDistortion, fixPose, true, distortionCorrected, doubleRefinement, LossType);
		//ReCalibratedFromGroundTruthCorrespondences(Path, 1, 0, 161, 88, ShutterModel);


		double SfMdistance = TriangulatePointsFromCalibratedCameras(Fname, 0, 2, 2.0);
		printf("SfM measured distance: %.3f\nPlease input the physcial distance (mm): ", SfMdistance);
		double Physicaldistance; cin >> Physicaldistance;
		double ratio = Physicaldistance / SfMdistance;
		sprintf(Fname, "%s/Corpus/BA_Camera_AllParams_after.txt", Path);
		ReSaveBundleAdjustedNVMResults(Fname, ratio);
		return 0;
	}
	else if (mode == 3) //step 4: generate corpus
	{
		int nviews = atoi(argv[2]), NPplus = atoi(argv[3]), module = atoi(argv[4]);
		//int nviews = 7, NPplus = 3, module = 0;

		sprintf(Fname, "%s/Corpus", Path);
		if (module == 0)
		{
			int ShutterModel = atoi(argv[5]), reTriangulate = atoi(argv[6]), LossType = atoi(argv[7]);
			double threshold = atof(argv[8]);
			bool sharedIntrinisc = false, fixIntrinsic = false, fixDistortion = false, fixPose = false, distortionCorrected = false, doubleRefinement = true;

			//int ShutterModel = 0, reTriangulate = 1, LossType = 0;
			//double threshold = 5.0;
			if (reTriangulate == 0)
			{
				RefineVisualSfMAndCreateCorpus(Fname, nviews, NPplus, ShutterModel, threshold, sharedIntrinisc, fixIntrinsic, fixDistortion, fixPose, true, distortionCorrected, doubleRefinement);

				double SfMdistance = TriangulatePointsFromCalibratedCameras(Fname, 0, 2, 2.0);
				printf("SfM measured distance: %.3f\nPlease input the physcial distance (mm): ", SfMdistance);
				double Physicaldistance; cin >> Physicaldistance;
				double ratio = Physicaldistance / SfMdistance;
				sprintf(Fname, "%s/Corpus/BA_Camera_AllParams_after.txt", Path);
				ReSaveBundleAdjustedNVMResults(Fname, ratio);
			}
			else
			{
				RefineVisualSfM(Fname, nviews, NPplus, ShutterModel, threshold, sharedIntrinisc, fixIntrinsic, fixDistortion, fixPose, true, distortionCorrected, doubleRefinement, LossType);

				double SfMdistance = TriangulatePointsFromCalibratedCameras(Fname, 0, 2, 2.0);
				printf("SfM measured distance: %.3f\nPlease input the physcial distance (mm): ", SfMdistance);
				double Physicaldistance; cin >> Physicaldistance;
				double ratio = Physicaldistance / SfMdistance;
				sprintf(Fname, "%s/Corpus/BA_Camera_AllParams_after.txt", Path);
				ReSaveBundleAdjustedNVMResults(Fname, ratio);

				//Remember to clean up the match.txt file before running match table
				sprintf(Fname, "%s/Corpus", Path);
				GenerateMatchingTableVisualSfM(Fname, nviews);
				BuildCorpusVisualSfm(Fname, distortionCorrected, ShutterModel, sharedIntrinisc, NPplus, LossType);
			}
			visualizationDriver(Path, 1, -1, -1, true, false, false, false, false, false, -1);
		}
		else if (module == 1)
		{
			int distortionCorrected = atoi(argv[5]), HistogramEqual = atoi(argv[6]), OulierRemoveTestMethod = atoi(argv[7]), //Fmat is more prefered. USAC is much faster then 5-pts :(
				LensType = atoi(argv[8]);
			double ratioThresh = atof(argv[9]);

			int nCams = nviews, cameraToScan = -1;
			if (OulierRemoveTestMethod == 2)
				nCams = atoi(argv[10]), cameraToScan = atoi(argv[11]);

			/*int distortionCorrected = 0, HistogramEqual = 0, OulierRemoveTestMethod = 2, LensType = 0;
			double ratioThresh = 0.7;

			int nCams = nviews, cameraToScan = -1;
			if (OulierRemoveTestMethod == 2)
			nCams = 1, cameraToScan = 0;*/

			int timeID = -1, ninlierThesh = 50;
			//GeneratePointsCorrespondenceMatrix_SiftGPU(Fname, nviews, -1, HistogramEqual, ratioThresh);
			GeneratePointsCorrespondenceMatrix_CPU(Fname, nviews, -1, HistogramEqual, ratioThresh, NULL, 1);

			omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for
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

			GenerateMatchingTable(Fname, nviews, timeID);
		}
		else
		{
			int distortionCorrected = atoi(argv[5]), ShutterModel = atoi(argv[6]), SharedIntrinsic = atoi(argv[7]), LossType = atoi(argv[8]); //0: NULL, 1: Huber
			//int distortionCorrected = 0, ShutterModel = 0, SharedIntrinsic = 1, LossType = 0; //0: NULL, 1: Huber

			BuildCorpus(Fname, distortionCorrected, ShutterModel, SharedIntrinsic, NPplus, LossType);
			visualizationDriver(Path, 1, -1, -1, true, false, false, false, false, false, -1);
		}
		return 0;
	}
	else if (mode == 4) //Step 5: Get features data for test sequences
	{
		int module = atoi(argv[2]);
		if (module == 0)
		{
			int startF = atoi(argv[3]), stopF = atoi(argv[4]), HistogramEqual = atoi(argv[5]), Covdet = atoi(argv[6]);
			//int startF = 0, stopF = 5, HistogramEqual = 0, Covdet = 1;

			int nthreads = omp_get_max_threads();
			int segLength = max((stopF - startF + 1) / nthreads, 1);
			vector<Point2i> seg;
			int segcount = 0;
			while (true)
			{
				seg.push_back(Point2i(startF + segLength*segcount, startF + segLength*(segcount + 1) - 1));
				if (startF + segLength*(segcount + 1) - 1 >= stopF)
					break;
				segcount++;
			}
			sprintf(Fname, "%s/Corpus", Path);

			omp_set_num_threads(nthreads);
#pragma omp parallel for
			for (int ii = 0; ii < (int)seg.size(); ii++)
				ExtractFeatureForVisualSfM(Fname, seg[ii].x, seg[ii].y, 1, HistogramEqual, Covdet);
		}
		else
		{
			int nviews = atoi(argv[3]), selectedView = atoi(argv[4]), startF = atoi(argv[5]), stopF = atoi(argv[6]), increF = atoi(argv[7]), HistogramEqual = atoi(argv[8]);
			//int nviews = 1, selectedView = 0, startF = 0, stopF = 25, increF = 1, HistogramEqual = 0;

			vector<int> availViews;
			if (selectedView < 0)
				for (int ii = 0; ii < nviews; ii++)
					availViews.push_back(ii);
			else
				availViews.push_back(selectedView);

			//if (!ReadIntrinsicResults(Path, corpusData.camera))
			//return 1;
			//LensCorrectionImageSequenceDriver(Fname, corpusData.camera[selectedView].K, corpusData.camera[selectedView].distortion, corpusData.camera[selectedView].LensModel, startF, stopF, 1.0, 1.0, 5);
			if (module == 1) //vlfeat covdet
			{
				int nthreads = omp_get_max_threads();
				int segLength = max((stopF - startF + 1) / nthreads, 1);
				vector<Point2i> seg;
				int segcount = 0;
				while (true)
				{
					seg.push_back(Point2i(startF + segLength*segcount, startF + segLength*(segcount + 1) - 1));
					if (startF + segLength*(segcount + 1) - 1 >= stopF)
						break;
					segcount++;
				}

				omp_set_num_threads(nthreads);
#pragma omp parallel for
				for (int ii = 0; ii < (int)seg.size(); ii++)
					ExtractFeaturefromExtractedFrames(Path, availViews, seg[ii].x, seg[ii].y, increF, HistogramEqual);
			}
			if (module == 2) //SIFTGPU
				ExtractSiftGPUfromExtractedFrames(Path, availViews, startF, stopF, increF, HistogramEqual);
		}
		return 0;
	}
	else if (mode == 5) //Step 6: Localize test sequence wrst to corpus
	{
		int startFrame = atoi(argv[2]), stopFrame = atoi(argv[3]), IncreFrame = atoi(argv[4]), selectedCam = atoi(argv[5]), nCams = atoi(argv[6]), //ncameras used to scan to corpus
			module = atoi(argv[7]); //0: Matching, 1: Localize, 2+3+4: refine on video, -1: visualize

		//int startFrame = 0, stopFrame = 25, IncreFrame = 1, selectedCam = 0, nCams = 1, module = 2;
		if (startFrame == 1 && module == 1)
		{
			sprintf(Fname, "%s/Intrinsic_%d.txt", Path, selectedCam); FILE*fp = fopen(Fname, "w+");	fclose(fp);
			sprintf(Fname, "%s/CamPose_%d.txt", Path, selectedCam); fp = fopen(Fname, "w+"); fclose(fp);
			sprintf(Fname, "%s/CamPose_RSCayley_%d.txt", Path, selectedCam); fp = fopen(Fname, "w+");	fclose(fp);
			sprintf(Fname, "%s/CamPose_sRSCayley_%d.txt", Path, selectedCam); fp = fopen(Fname, "w+"); fclose(fp);
		}

		if (module < 0)
			visualizationDriver(Path, nCams, startFrame, stopFrame, true, false, true, false, false, false, startFrame);
		if (module == 0 || module == 1)
		{
			int distortionCorrected = atoi(argv[8]), //Must set to 1 after corpus matching with calibrated caminfo
				GetIntrinsicFromCorpus = atoi(argv[9]), sharedIntriniscOptim = atoi(argv[10]),
				LensType = atoi(argv[11]);// 0:RADIAL_TANGENTIAL_PRISM, 1: FISHEYE;

			//int distortionCorrected = 1, GetIntrinsicFromCorpus = 1, sharedIntriniscOptim = 1, LensType = 0;
			LocalizeCameraToCorpusDriver(Path, startFrame, stopFrame, IncreFrame, module, nCams, selectedCam, distortionCorrected, GetIntrinsicFromCorpus, sharedIntriniscOptim, LensType);
		}
		if (module == 2)
		{
			int fixIntrinsic = atoi(argv[8]), fixDistortion = atoi(argv[9]), fix3D = atoi(argv[10]), distortionCorrected = atoi(argv[11]), RobustLoss = atoi(argv[12]);
			double threshold = atof(argv[13]);

			//int  fixIntrinsic = 0, fixDistortion = 0, fix3D = 0, distortionCorrected = 0, RobustLoss = 1; //Recomemded
			//double threshold = 5.0;

			if (distortionCorrected == 1)
				fixIntrinsic = 1, fixDistortion = 1;
			VideoPose_GSBA(Path, selectedCam, startFrame, stopFrame, fixIntrinsic, fixDistortion, fix3D, distortionCorrected, threshold, RobustLoss);
		}
		if (module == 3)
		{
			int fixIntrinsic = atoi(argv[8]), fixDistortion = atoi(argv[9]), fixPose = atoi(argv[10]), fix3D = atoi(argv[11]),
				distortionCorrected = atoi(argv[12]), RobustLoss = atoi(argv[13]), fixfirstCamPose = 0;
			double threshold = atof(argv[14]);
			bool doubleRefinement = true;

			//double threshold = 5.0;
			//int fixIntrinsic = 0, fixDistortion = 0, fixPose = 0, fixfirstCamPose = 1, fix3D = 0, distortionCorrected = 1;
			VideoPose_RS_Cayley_BA(Path, selectedCam, startFrame, stopFrame, fixIntrinsic, fixDistortion, fixPose, fixfirstCamPose, fix3D, distortionCorrected, RobustLoss, doubleRefinement, threshold);
		}
		if (module == 4)
		{
			int fixIntrinsic = atoi(argv[8]), fixDistortion = atoi(argv[9]), fixPose = atoi(argv[10]), fix3D = atoi(argv[11]),
				distortionCorrected = atoi(argv[12]), RobustLoss = atoi(argv[13]), fixfirstCamPose = 0;
			double threshold = atof(argv[14]);
			bool doubleRefinement = true;

			AllVideoPose_RS_Cayley_BA(Path, nCams, startFrame, stopFrame, fixIntrinsic, fixDistortion, fixPose, fixfirstCamPose, fix3D, distortionCorrected, RobustLoss, doubleRefinement, threshold);
		}
		if (module == 5)
		{
			//double threshold = atof(argv[13]);
			//int fixIntrinsic = atoi(argv[8]), fixDistortion = atoi(argv[9]), fix3D = atoi(argv[10]), distortionCorrected = atoi(argv[11]), controlStep = atoi(argv[12]);
			double threshold = 5.0;
			int  fixIntrinisc = 0, fixDistortion = 0, fixed3D = 0, distortionCorrected = 0, controlStep = 3; //Recomemded

			VideoSplineRSBA(Path, startFrame, stopFrame, selectedCam, distortionCorrected, false, false, threshold, controlStep);
			//Pose_se_BSplineInterpolation("E:/RollingShutter/_CamPoseS_0.txt", "E:/RollingShutter/__CamPose_0.txt", 100, "E:/RollingShutter/Pose.txt");
		}

		return 0;
	}
	else if (mode == 6) //step 7: generate 3d data from test sequences
	{
		int startFrame = 140,//atoi(argv[2]),
			stopFrame = 140,//atoi(argv[3]),
			increF = 1,// atoi(argv[4]),
			module = 1;// atoi(argv[5]); 0: matching, 2 : triangulation

		int HistogrameEqual = 0,
			distortionCorrected = 0,
			OulierRemoveTestMethod = 2, ninlierThesh = 40, //fmat test
			LensType = RADIAL_TANGENTIAL_PRISM,
			NViewPlus = 2,
			nviews = 4;
		double ratioThresh = 0.8, reprojectionThreshold = 5;

		int FrameOffset[] = { 0, -4, -3, 0 };
		if (module == 0)
		{
			for (int timeID = startFrame; timeID <= stopFrame; timeID += increF)
			{
				GeneratePointsCorrespondenceMatrix_SiftGPU(Path, nviews, timeID, HistogrameEqual, 0.6, FrameOffset);
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
								FundamentalMatOutliersRemove(Path, timeID, jj, ii, ninlierThesh, LensType, distortionCorrected, false, nviews, -1, FrameOffset);
						}
					}
				}
				GenerateMatchingTable(Path, nviews, timeID);
			}
		}
		else if (module == 1) //Get the matched points
			GetPutativeMatchesForEachView(Path, nviews, startFrame, stopFrame, increF, Point2d(2.0, 7.5), NViewPlus, FrameOffset);
		else
		{
			bool save2DCorres = false;
			double MaxDepthThresh = 99000; //mm
			Build3DFromSyncedImages(Path, nviews, startFrame, stopFrame, increF, LensType, distortionCorrected, NViewPlus, reprojectionThreshold, MaxDepthThresh, FrameOffset, save2DCorres, false, 10.0, false);
			visualizationDriver(Path, nviews, 0, 600, true, false, true, false, true, false, startFrame);
		}

		return 0;
	}
	else if (mode == 7)
	{
		int nVideoCams = 5,
			startFrame = 0, stopFrame = 199,
			LensModel = RADIAL_TANGENTIAL_PRISM;

		int selectedCam[] = { 1, 2 }, selectedTime[2] = { 0, 0 }, ChooseCorpusView1 = -1, ChooseCorpusView2 = -1;

		VideoData AllVideoInfo;
		ReadVideoData(Path, AllVideoInfo, nVideoCams, startFrame, stopFrame);

		double Fmat[9];
		int seletectedIDs[2] = { selectedCam[0] * max(MaxnFrames, stopFrame) + selectedTime[0], selectedCam[1] * max(MaxnFrames, stopFrame) + selectedTime[1] };
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

		int nviews = 8, startFrame = 1, stopFrame = 600, timeID = 450;// atoi(argv[2]);
		VideoData AllVideoInfo;
		if (ReadVideoData(Path, AllVideoInfo, nviews, startFrame, stopFrame) == 1)
			return 1;

		for (int timeID = 450; timeID <= 450; timeID += 10)
		{
			LoadTrackData(Path, timeID, InfoTraj, true);
			//Write3DMemAtThatTime(Path, InfoTraj, CameraInfo, 100, 101);

			vector<int> TrajectUsed;
			for (int ii = 0; ii < InfoTraj.nTrajectories; ii++)
				TrajectUsed.push_back(ii);

			Genrate2DTrajectory3(Path, timeID, InfoTraj, AllVideoInfo, TrajectUsed);
			//Genrate2DTrajectory2(Path, timeID, InfoTraj, AllVideoInfo, TrajectUsed);
			//Genrate2DTrajectory(Path, 120, InfoTraj, CameraInfo, TrajectUsed);//
		}
		return 0;

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

/*// f(x,y) = (1-x)^2 + 100(y - x^2)^2;
struct MyScalarCostFunctor {
MyScalarCostFunctor(){}

template <typename T>	bool operator()(const T* const xy, T* residuals) 	const
{
residuals[0] = ((T)1.0 - xy[0]) * ((T)1.0 - xy[0]) + (T)100.0 * (xy[1] - xy[0] * xy[0]) * (xy[1] - xy[0] * xy[0]);

return true;
}

static ceres::CostFunction* Create()
{
return (new ceres::AutoDiffCostFunction<MyScalarCostFunctor, 1, 2>(new MyScalarCostFunctor()));
}
};
class Rosenbrock : public ceres::FirstOrderFunction {
public:
virtual ~Rosenbrock() {}

virtual bool Evaluate(const double* parameters, double* cost, double* gradient) const
{
const double x = parameters[0], y = parameters[1];
cost[0] = (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x);
if (gradient != NULL)
{
gradient[0] =  -2.0 * (1.0 - x) - 200.0 * (y - x * x) * 2.0 * x;
gradient[1] = 200.0 * (y - x * x);
}

ceres::CostFunction* cost_function = MyScalarCostFunctor::Create();
cost_function->Evaluate(&parameters, &cost[0], NULL);

if (gradient != NULL)
{
cost_function->Evaluate(&parameters, &cost[0], &gradient);
}
return true;
}

virtual int NumParameters() const
{
return 2;
}
};
int CeresNonlinearOptimExample()
{
double parameters[2] = { -4, 5 };

ceres::GradientProblemSolver::Options options;
options.minimizer_progress_to_stdout = true;
options.max_num_iterations = 1000;

ceres::GradientProblemSolver::Summary summary;
ceres::GradientProblem problem(new Rosenbrock());
ceres::Solve(options, problem, parameters, &summary);

std::cout << summary.FullReport() << "\n";
std::cout << "Initial x: " << -3.2 << " y: " << 1.0 << "\n";
std::cout << "Final   x: " << parameters[0]
<< " y: " << parameters[1] << "\n";
return 0;
}

struct Xsolver {
Xsolver(double measuredValue1, double measuredValue2) :measuredValue1(measuredValue1), measuredValue2(measuredValue2){}

template <typename T>    bool operator()(T const* const* X, T* residuals)     const
{
residuals[0] = X[0][0] - (T)measuredValue1;
residuals[1] = X[1][0] - (T)measuredValue2;

return true;
}
double measuredValue1, measuredValue2;
};
int main(int argc, char** argv) {


ceres::Problem problem;

const int ntimes = 4;
double x = 0.0;
double y[ntimes];
for (int ii = 0; ii < ntimes; ii++)
y[ii] = 0.0;
y[0] = -10.0;

for (int ii = 0; ii < ntimes; ii++)
{
vector<double*> parameter_blocks;
parameter_blocks.push_back(&x);
parameter_blocks.push_back(&y[ii]);
ceres::DynamicNumericDiffCostFunction<Xsolver, ceres::CENTRAL> *cost_function = new ceres::DynamicNumericDiffCostFunction<Xsolver, ceres::CENTRAL>(new Xsolver(2.0, 3.0*ii));
cost_function->AddParameterBlock(1);
cost_function->AddParameterBlock(1);
cost_function->SetNumResiduals(2);
problem.AddResidualBlock(cost_function, NULL, parameter_blocks);
}

problem.SetParameterLowerBound(&x, 0, -2), problem.SetParameterUpperBound(&x, 0, -1);
problem.SetParameterBlockConstant(&y[0]);

ceres::Solver::Options solver_options;
solver_options.minimizer_progress_to_stdout = true;
solver_options.gradient_tolerance = 1.0e-9;

ceres::Solver::Summary summary;
printf("Solving...\n");
Solve(solver_options, &problem, &summary);
printf("Done.\n");
std::cout << summary.FullReport() << "\n";

printf("Final values:\n");
printf("%.3f ", x);
for (int ii = 0; ii < ntimes; ii++)
printf("%.3f ", y[ii]);


return 0;
}*/
/*int main(int argc, char **argv)
{
gsl_bspline_workspace *bw = gsl_bspline_alloc(SplineOrder, nknots);
gsl_vector *Bi = gsl_vector_alloc(nknots + SplineOrder - 2);
gsl_vector *gsl_BreakPts = gsl_vector_alloc(nknots);

//copy data to gsl format
for (int ii = 0; ii < nknots; ii++)
gsl_vector_set(gsl_BreakPts, ii, BreakPts[ii]);

//contruct knots
gsl_BSplineGetKnots(gsl_BreakPts, bw);

//construct basis matrix
for (int ii = 0; ii < nsamples; ii++)
{
gsl_bspline_eval(ControlPts[ii], Bi, bw); //compute basis vector for point i
for (int jj = 0; jj < ncoeffs; jj++)
Basis[jj + ii*ncoeffs] = gsl_vector_get(Bi, jj);
}
return 0;
}*/
/*//Calculate the blending value, this is done recursively.
double SplineBlend(int k, int t, int *u, double v)
{
double value;

if (t == 1)
{
if ((u[k] <= v) && (v < u[k + 1]))
value = 1;
else
value = 0;
}
else
{
if ((u[k + t - 1] == u[k]) && (u[k + t] == u[k + 1]))
value = 0;
else if (u[k + t - 1] == u[k])
value = (u[k + t] - v) / (u[k + t] - u[k + 1]) * SplineBlend(k + 1, t - 1, u, v);
else if (u[k + t] == u[k + 1])
value = (v - u[k]) / (u[k + t - 1] - u[k]) * SplineBlend(k, t - 1, u, v);
else
value = (v - u[k]) / (u[k + t - 1] - u[k]) * SplineBlend(k, t - 1, u, v) + (u[k + t] - v) / (u[k + t] - u[k + 1]) * SplineBlend(k + 1, t - 1, u, v);
}
return(value);
}

//This returns the point "output" on the spline curve.The parameter "v" indicates the position, it ranges from 0 to n-t+2
void SplinePoint(int *u, int n, int t, double v, Point3d *control, Point3d *output)
{
int k;
double b;

output[0].x = 0;
output[0].y = 0;
output[0].z = 0;

for (k = 0; k <= n; k++)
{
b = SplineBlend(k, t, u, v);
if (b == 1.0)
int a = 0;
output[0].x += control[k].x * b;
output[0].y += control[k].y * b;
output[0].z += control[k].z * b;
}
return;
}

void BSplineGetKnots(int *u, int n, int t)
{
int j;

for (j = 0; j <= n + t; j++)
{
if (j < t)
u[j] = 0;
else if (j <= n)
u[j] = j - t + 1;
else if (j > n)
u[j] = n - t + 2;
}
}
void SplineCurve(Point3d *inp, int n, int *knots, int t, Point3d *outp, int res)
{
int i;
double interval, increment;

interval = 0;
increment = (n - t + 2) / (double)(res - 1);
for (i = 0; i < res - 1; i++) {
SplinePoint(knots, n, t, interval, inp, &(outp[i]));
interval += increment;
}
outp[res - 1] = inp[n];
}

#define N 3
Point3d inp[(N + 1)] = { Point3d(0.0, 0.0, 0.0), Point3d(1.0, 0.0, 3.0), Point3d(2.0, 0.0, 1.0), Point3d(4.0, 0.0, 4.0) };
#define T 3
int knots[N + T + 1];
#define RESOLUTION 200
Point3d outp[RESOLUTION];

int main(int argc, char **argv)
{
BSplineGetKnots(knots, N, T);

double interval = 0;
double increment = (N - T + 2) / (double)(RESOLUTION - 1);
for (int i = 0; i < RESOLUTION - 1; i++)
{
SplinePoint(knots, N, T, interval, inp, &(outp[i]));
interval += increment;
}
outp[RESOLUTION - 1] = inp[N];

//SplineCurve(inp, N, knots, T, outp, RESOLUTION);

return 0;
}*/
/*cv::VideoCapture cap;	// Camera object
	cap = cv::VideoCapture("E:/Shirt2/11/IMG_0004.mov");
	if (!cap.isOpened())
	throw(runtime_error("camera or video cannot be openned."));

	// Press Esc to quit
	int ii = 0;
	while (true)
	{
	// Current frame
	cv::Mat frame;
	// Get a new frame from camera
	cap >> frame;
	if (frame.empty())
	break;

	sprintf(Fname, "E:/Shirt2/11/%d.png", ii);
	imwrite(Fname, frame);
	ii++;
	}
	return 0;*/

