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
int ShowSyncLoad(char *DataPATH, char *SynFileName, char *SavePATH, int nsequences = 7, int refSeq = -1)
{
	char Fname[2000];
	int WBlock = 1920, HBlock = 1080, nBlockX = 3, nchannels = 3, MaxFrames = 300, playBackSpeed = 1, id;

	int *seqName = new int[nsequences];
	double offset, *Offset = new double[nsequences];
	Sequence *mySeq = new Sequence[nsequences];
	for (int ii = 0; ii < nsequences; ii++)
		seqName[ii] = ii;

	printf("Please input offset info in the format time-stamp format!\n");
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

	if (refSeq == -1)
	{
		double earliestTime = DBL_MAX;
		for (int ii = 0; ii < nsequences; ii++)
		{
			mySeq[ii].InitSeq(1, Offset[ii]);
			if (earliestTime>Offset[ii])
				earliestTime = Offset[ii], refSeq = ii;
		}
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
		sprintf(Fname, "Seq %d", ii + 1);
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

			sprintf(Fname, "Seq %d", ii + 1);
			setSeqFrame = (int)((1.0*setReferenceFrame / mySeq[refSeq].TimeAlignPara[0] - mySeq[ii].TimeAlignPara[1]) * mySeq[ii].TimeAlignPara[0] + 0.5); //(refFrame/fps_ref - offset_i)*fps_i

			if (same == 2)
			{
				noUpdate++;
				if (autoplay)
				{
					sprintf(Fname, "Seq %d", ii + 1);
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
				printf("Sequence %d frame %d\n", ii + 1, setSeqFrame);
				oFrameID[ii + 1] = FrameID[ii + 1];
				cvSetTrackbarPos(Fname, "Control", oFrameID[ii + 1]);
				sprintf(Fname, "%s/%d/%d.png", DataPATH, seqName[ii], setSeqFrame);
				if (GrabImageCVFormat(Fname, SubImage, swidth, sheight, nchannels))
					Set_Sub_Mat(SubImage, BigImg, nchannels*swidth, sheight, nchannels*width, nchannels*BlockXID*WBlock, BlockYID*HBlock);
				else
					Set_Sub_Mat(BlackImage, BigImg, nchannels*WBlock, HBlock, nchannels*width, nchannels*BlockXID*WBlock, BlockYID*HBlock);

				if (saveFrame)
				{
					sprintf(Fname, "%s/%d/_%d.png", SavePATH, seqName[ii], SaveFrameCount / nsequences);
					SaveDataToImageCVFormat(Fname, SubImage, swidth, sheight, nchannels);
					SaveFrameCount++;
				}
				else
					SaveFrameCount = 0;
			}
			if (autoplay)
			{
				sprintf(Fname, "Seq %d", ii + 1);
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

			if (0)
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
int TriangulatePointsFromCalibratedCameras2(char *Path, int nCams, int selectedCams, int distortionCorrected, double threshold = 2.0)
{
	char Fname[200];
	Corpus corpusData;
	sprintf(Fname, "%s/BA_Camera_AllParams_after.txt", Path);
	if (!loadBundleAdjustedNVMResults(Fname, corpusData))
		return 1;

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
void VisualizeCleanMatches(char *Path, int view1, int view2, int timeID, double fractionMatchesDisplayed = 0.5, int FrameOffset1 = 0, int FrameOffset2 = 0)
{
	char Fname[200];
	if (timeID < 0)
		sprintf(Fname, "%s/Corpus/M_%d_%d.dat", Path, view1, view2);
	else
		sprintf(Fname, "%s/Dynamic/M%d_%d_%d.dat", Path, timeID, view1, view2);

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
		sprintf(Fname, "%s/%d/K%d.dat", Path, view2, timeID + FrameOffset2);
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
int PrepareTrajectoryInfo(char *Path, VideoData *VideoInfo, PerCamNonRigidTrajectory *CamTraj, double *OffsetInfo, int nCams, int npts, int startFrame, int stopFrame)
{
	char Fname[200];
	int id, nf, frameID;
	//CamCenter Ccenter;
	//RotMatrix Rmat;
	//Quaternion Qmat;
	//KMatrix Kmat;
	//Pmat P;

	for (int camID = 0; camID < nCams; camID++)
		if (ReadVideoDataI(Path, VideoInfo[camID], camID, startFrame, stopFrame) == 1)
			return 1;

	for (int ii = 0; ii < nCams; ii++)
		CamTraj[ii].npts = npts,
		CamTraj[ii].Track2DInfo = new Track2D[npts],
		CamTraj[ii].Track3DInfo = new Track3D[npts];

	Point2d uv;
	vector<Point2d> uvAll;
	vector<int> frameIDAll;
	for (int camID = 0; camID < nCams; camID++)
	{
		sprintf(Fname, "%s/Track2D/C_%d.txt", Path, camID); FILE *fp = fopen(Fname, "r");
		for (int pid = 0; pid < npts; pid++)
		{
			fscanf(fp, "%d %d ", &id, &nf);
			if (id != pid)
				printf("Problem at Point %d of Cam %d", id, camID);

			uvAll.clear(), frameIDAll.clear();
			for (int fid = 0; fid < nf; fid++)
			{
				fscanf(fp, "%d %lf %lf ", &frameID, &uv.x, &uv.y);
				if (frameID < startFrame || frameID>stopFrame)
					continue;
				if (!VideoInfo[camID].VideoInfo[frameID].valid)
					continue; //camera not localized

				if (uv.x > 0 && uv.y > 0)
				{
					LensCorrectionPoint(&uv, VideoInfo[camID].VideoInfo[frameID].K, VideoInfo[camID].VideoInfo[frameID].distortion);
					uvAll.push_back(uv);
					frameIDAll.push_back(frameID);
				}
			}

			nf = uvAll.size();
			CamTraj[camID].Track2DInfo[pid].nf = nf;
			CamTraj[camID].Track3DInfo[pid].nf = nf;
			CamTraj[camID].Track2DInfo[pid].uv = new Point2d[nf];
			CamTraj[camID].Track3DInfo[pid].xyz = new double[nf * 3];

			for (int kk = 0; kk < nf; kk++)
				CamTraj[camID].Track2DInfo[pid].uv[kk] = uvAll[kk];
		}
		fclose(fp);
	}

	//Triangulate 3D data
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
int FmatSyncBruteForce2DStereo(char *Path, int *SelectedCams, int startFrame, int stopFrame, int ntracks, int *OffsetInfo, int LowBound, int UpBound, bool GivenF, bool silent = true)
{
	char Fname[200]; FILE *fp = 0;
	const int nCams = 2;

	//Read calib info
	VideoData VideoInfo[2];
	if (ReadVideoDataI(Path, VideoInfo[0], SelectedCams[0], startFrame, stopFrame) == 1)
		return 1;
	if (ReadVideoDataI(Path, VideoInfo[1], SelectedCams[1], startFrame, stopFrame) == 1)
		return 1;

	int id, frameID, npts;
	int nframes = max(MaxnFrames, stopFrame);

	double u, v;
	ImgPtEle ptEle;
	vector<ImgPtEle> *PerCam_UV = new vector<ImgPtEle>[nCams*ntracks];
	vector<XYZD> *PerCam_XYZ = new vector<XYZD>[nCams], *XYZ = new vector<XYZD>[ntracks], XYZBK;

	//Get 2D info
	for (int camID = 0; camID < nCams; camID++)
	{
		for (int trackID = 0; trackID < ntracks; trackID++)
			PerCam_UV[camID*ntracks + trackID].reserve(stopFrame - startFrame + 1);

		int  FirstValidFrame;
		if (!GivenF)
		{
			for (int fid = startFrame; fid < stopFrame; fid++)
			{
				if (VideoInfo[camID].VideoInfo[fid].valid)
				{
					FirstValidFrame = fid;
					break;
				}
			}
		}

		sprintf(Fname, "%s/Track2D/%d.txt", Path, SelectedCams[camID]); FILE *fp = fopen(Fname, "r");
		for (int trackID = 0; trackID < ntracks; trackID++)
		{
			fscanf(fp, "%d %d ", &id, &npts);
			if (id != trackID)
				printf("Problem at Point %d of Cam %d", id, camID);
			for (int pid = 0; pid < npts; pid++)
			{
				fscanf(fp, "%d %lf %lf ", &frameID, &u, &v);
				if (frameID < startFrame || frameID>stopFrame)
					continue;
				if (GivenF && !VideoInfo[camID].VideoInfo[frameID].valid)
					continue; //camera not localized

				if (u > 0 && v > 0)
				{
					ptEle.pt2D.x = u, ptEle.pt2D.y = v, ptEle.frameID = frameID;
					if (GivenF)
						LensCorrectionPoint(&ptEle.pt2D, VideoInfo[camID].VideoInfo[frameID].K, VideoInfo[camID].VideoInfo[frameID].distortion);
					else
						LensCorrectionPoint(&ptEle.pt2D, VideoInfo[camID].VideoInfo[FirstValidFrame].K, VideoInfo[camID].VideoInfo[FirstValidFrame].distortion);

					PerCam_UV[camID*ntracks + id].push_back(ptEle);
				}
			}
		}
		fclose(fp);
	}

	//Generate Calib Info
	if (GivenF)
	{
		for (int trackID = 0; trackID < ntracks; trackID++)
		{
			int count = 0;
			for (int camID = 0; camID < nCams; camID++)
			{
				for (int frameID = 0; frameID < PerCam_UV[camID*ntracks + trackID].size(); frameID++)
				{
					int RealFrameID = PerCam_UV[camID*ntracks + trackID][frameID].frameID;

					for (int kk = 0; kk < 9; kk++)
						PerCam_UV[camID*ntracks + trackID][frameID].K[kk] = VideoInfo[camID].VideoInfo[RealFrameID].K[kk],
						PerCam_UV[camID*ntracks + trackID][frameID].R[kk] = VideoInfo[camID].VideoInfo[RealFrameID].R[kk];

					for (int kk = 0; kk < 3; kk++)
						PerCam_UV[camID*ntracks + trackID][frameID].camcenter[kk] = VideoInfo[camID].VideoInfo[RealFrameID].camCenter[kk],
						PerCam_UV[camID*ntracks + trackID][frameID].T[kk] = VideoInfo[camID].VideoInfo[RealFrameID].T[kk];
				}
			}
		}
	}

	//Start sliding
	int *OffsetID = new int[UpBound - LowBound + 1];
	double*AllFmatCost = new double[UpBound - LowBound + 1];

	int BestOffset = 0;
	double minError = 9e9;
	int count = 0;
	for (int off = LowBound; off <= UpBound; off++)
	{
		double Fmat[9], error, cumError = 0.0, usedPointsCount = 0;
		if (GivenF)
		{
			for (int trackID = 0; trackID < ntracks; trackID++)
			{
				for (int pid = 0; pid < PerCam_UV[trackID].size(); pid++)
				{
					int currentFrame = PerCam_UV[trackID][pid].frameID + OffsetInfo[0], otherCameraFrame = currentFrame + off + OffsetInfo[1];

					//see if the corresponding frame in the other camera has point
					error = 0.0;
					for (int pid2 = 0; pid2 < PerCam_UV[ntracks + trackID].size(); pid2++)
					{
						if (PerCam_UV[ntracks + trackID][pid2].frameID == otherCameraFrame)
						{
							computeFmat(VideoInfo[0].VideoInfo[currentFrame], VideoInfo[1].VideoInfo[otherCameraFrame], Fmat);
							error = FmatPointError(Fmat, PerCam_UV[trackID][pid].pt2D, PerCam_UV[ntracks + trackID][pid2].pt2D);
							usedPointsCount++;
							break;
						}
					}
					cumError += error;
				}
			}
		}
		else
		{
			//Compute Fmat
			vector<Point2d> points1, points2;
			for (int fid1 = 0; fid1 < (int)PerCam_UV[0].size(); fid1++)
			{
				for (int fid2 = 0; fid2 < (int)PerCam_UV[ntracks].size(); fid2++)
				{
					if (PerCam_UV[0][fid1].frameID + OffsetInfo[0] == PerCam_UV[ntracks][fid2].frameID + off + OffsetInfo[1])
					{
						for (int i = 0; i < ntracks; i++)
						{
							points1.push_back(PerCam_UV[i][fid1].pt2D);
							points2.push_back(PerCam_UV[i + ntracks][fid2].pt2D);
						}
						break;
					}
				}
			}

			if (points1.size() < 8)
			{
				cumError = 9e9;
				usedPointsCount = 0;
			}
			else
			{
				Mat cvFmat = findFundamentalMat(points1, points2, CV_FM_8POINT, 3, 0.99);

				double Fmat[9];
				for (int ii = 0; ii < 9; ii++)
					Fmat[ii] = cvFmat.at<double>(ii);

				for (int ii = 0; ii < (int)points1.size(); ii++)
					cumError += FmatPointError(Fmat, points1[ii], points2[ii]);
				usedPointsCount = points1.size();
			}
		}
		cumError = cumError / (0.0000001 + usedPointsCount);
		if (cumError < minError)
			minError = cumError, BestOffset = off;

		AllFmatCost[count] = cumError;
		if (silent)
			printf("@off %d (id: %d): %.5f -- Best offset: %d\n", off, count, cumError, BestOffset);
		count++;
	}
	printf("Pair (%d, %d): %d\n", SelectedCams[0], SelectedCams[1], BestOffset);

	OffsetInfo[1] = BestOffset;

	delete[]PerCam_UV, delete[]PerCam_XYZ;
	delete[]OffsetID, delete[]AllFmatCost;

	return 0;
}
int GeometricConstraintSyncDriver(char *Path, int nCams, int npts, int startFrame, int stopTime, int Range, bool GivenF, double *OffsetInfo, bool HasInitOffset = false)
{
	if (OffsetInfo == NULL)
	{
		OffsetInfo = new double[nCams];
		for (int ii = 0; ii < nCams; ii++)
			OffsetInfo[ii] = 0;
	}
	if (!HasInitOffset)
		for (int ii = 0; ii < nCams; ii++)
			OffsetInfo[ii] = 0;

	printf("Geometric sync:\n");
	char Fname[200]; sprintf(Fname, "%s/GeoSync.txt", Path); 	FILE *fp = fopen(Fname, "w+");
	for (int jj = 0; jj < nCams - 1; jj++)
	{
		for (int ii = jj + 1; ii < nCams; ii++)
		{
			int SelectedCams[2] = { jj, ii }, Offset[] = { OffsetInfo[jj], OffsetInfo[ii] };
			FmatSyncBruteForce2DStereo(Path, SelectedCams, startFrame, stopTime, npts, Offset, -Range, Range, GivenF);
			printf("Between (%d, %d): %d\n", jj, ii, Offset[1] - Offset[0]);
			fprintf(fp, "%d %d %d\n", jj, ii, Offset[1] - Offset[0]);
		}
	}
	fclose(fp);

	PrismMST(Path, "GeoSync", nCams);
	AssignOffsetFromMST(Path, "GeoSync", nCams, OffsetInfo);
	printf("\n");

	return 0;
}
int TriangulateFrameSync2DTrajectories(char *Path, vector<int> SelectedCams, vector<int> FrameOffset, int startFrame, int stopFrame, int npts, bool CleanCorrespondencesByTriangulationTest = false, double *GTFrameOffset = 0, double *ialpha = 0, double*Tscale = 0)
{
	int nCams = (int)SelectedCams.size();
	bool createdMem = false;
	if (GTFrameOffset == NULL)
	{
		createdMem = true;
		GTFrameOffset = new double[nCams];
		ialpha = new double[1], Tscale = new double[1];
		ialpha[0] = 1.0, Tscale[0] = 1.0;
		for (int ii = 0; ii < nCams; ii++)
			GTFrameOffset[ii] = 0.0;
	}

	char Fname[512];
	VideoData *VideoInfo = new VideoData[nCams];
	for (int camID = 0; camID < nCams; camID++)
	{
		if (ReadVideoDataI(Path, VideoInfo[camID], SelectedCams[camID], startFrame, stopFrame) == 1)
			return 1;
	}

	vector<int> *RealframeID = new vector<int>[MaxnFrames*npts];
	vector<int> *CamID = new vector<int>[MaxnFrames*npts];
	vector<Point2d> *FrameSyncedPoints = new vector<Point2d>[MaxnFrames*npts];
	vector<Pmat> *Pmatrix = new vector<Pmat>[MaxnFrames*npts];

	Pmat Pm;
	Point2d uv;
	int pid, cid, fid, nf;
	for (int ii = 0; ii < nCams; ii++)
	{
		sprintf(Fname, "%s/Track2D/%d.txt", Path, SelectedCams[ii]);	FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("Cannot load %s\n", Fname);
			return 1;
		}
		for (pid = 0; pid < npts; pid++)
		{
			fscanf(fp, "%d %d ", &pid, &nf);
			for (int kk = 0; kk < nf; kk++)
			{
				fscanf(fp, "%d %lf %lf ", &fid, &uv.x, &uv.y);
				if (fid<0 || fid>MaxnFrames || !VideoInfo[ii].VideoInfo[fid].valid)
					continue;
				LensCorrectionPoint(&uv, VideoInfo[ii].VideoInfo[fid].K, VideoInfo[ii].VideoInfo[fid].distortion);

				for (int ll = 0; ll < 12; ll++)
					Pm.P[ll] = VideoInfo[ii].VideoInfo[fid].P[ll];

				int fake_fid = fid - FrameOffset[ii]; //fake the order
				if (fake_fid<0 || fake_fid>MaxnFrames)
					continue;

				CamID[fake_fid + pid*MaxnFrames].push_back(ii);
				RealframeID[fake_fid + pid*MaxnFrames].push_back(fid);
				FrameSyncedPoints[fake_fid + pid*MaxnFrames].push_back(uv);
				Pmatrix[fake_fid + pid*MaxnFrames].push_back(Pm);
			}
		}
		fclose(fp);
	}

	Point3d P3d;
	Point2d *pts = new Point2d[nCams * 2], *ppts = new Point2d[nCams * 2];
	double *A = new double[6 * nCams * 2];
	double *B = new double[2 * nCams * 2];
	double *P = new double[12 * nCams * 2];
	double *tP = new double[12 * nCams * 2];
	bool *passed = new bool[nCams * 2];
	vector<int>Inliers[1];  Inliers[0].reserve(nCams * 2);

	vector<int> *CleanFrameSynID;
	vector<Point2d> *CleanedFrameSyncedPoints;
	if (CleanCorrespondencesByTriangulationTest)
		CleanFrameSynID = new vector<int>[nCams*npts], CleanedFrameSyncedPoints = new vector<Point2d>[nCams*npts];

	for (pid = 0; pid < npts; pid++)
	{
		sprintf(Fname, "%s/frameSynced_Track_%d.txt", Path, pid);  FILE *fp = fopen(Fname, "w+");
		for (int kk = 0; kk <= stopFrame - startFrame; kk++)
		{
			int nvis = FrameSyncedPoints[kk + pid*MaxnFrames].size();
			if (nvis < 2)
				continue;

			vector<int> cameraID, frameID;
			for (int ii = 0; ii < nvis; ii++)
			{
				cameraID.push_back(CamID[kk + pid*MaxnFrames][ii]);
				frameID.push_back(RealframeID[kk + pid*MaxnFrames][ii]);
				uv = FrameSyncedPoints[kk + pid*MaxnFrames][ii];
				pts[ii] = FrameSyncedPoints[kk + pid*MaxnFrames][ii];

				for (int ll = 0; ll < 12; ll++)
					P[12 * ii + ll] = Pmatrix[kk + pid*MaxnFrames][ii].P[ll];
			}

			/*finalerror = NviewTriangulationRANSAC(pts, P, &P3d, passed, Inliers, nvis, 1, 10, 0.5, 10, A, B, tP);
			int ninlier = 0;
			for (int ii = 0; ii < Inliers[0].size(); ii++)
			if (Inliers[0].at(ii))
			ninlier++;
			Inliers[0].clear();
			if (ninlier>=5)*/
			NviewTriangulation(pts, P, &P3d, nvis, 1, NULL, A, B);
			ProjectandDistort(P3d, ppts, P, NULL, NULL, nvis);

			int nvalid = 0;
			double finalerror = 0.0;
			for (int ll = 0; ll < nvis; ll++)
				finalerror += pow(ppts[ll].x - pts[ll].x, 2) + pow(ppts[ll].y - pts[ll].y, 2), nvalid++;
			finalerror = sqrt(finalerror / nvis);

			if (finalerror < 1000 && nvalid >= 2)
			{
				for (int ii = 0; ii < nvis; ii++)
				{
					cid = CamID[kk + pid*MaxnFrames][ii], fid = RealframeID[kk + pid*MaxnFrames][ii];
					fprintf(fp, "%.4f %.4f %.4f %.2f %d %d\n", P3d.x, P3d.y, P3d.z, (GTFrameOffset[cid] + fid)*ialpha[0] * Tscale[0], SelectedCams[cid], fid);
				}

				for (int ii = 0; ii < nvis && CleanCorrespondencesByTriangulationTest; ii++)
				{
					cid = CamID[kk + pid*MaxnFrames][ii], fid = RealframeID[kk + pid*MaxnFrames][ii];
					CleanFrameSynID[cid*npts + pid].push_back(fid);
					CleanedFrameSyncedPoints[cid*npts + pid].push_back(FrameSyncedPoints[kk + pid*MaxnFrames][ii]);
				}
			}
		}
		fclose(fp);
	}

	for (int ii = 0; ii < nCams &&CleanCorrespondencesByTriangulationTest; ii++)
	{
		sprintf(Fname, "%s/Track2D/C_%d.txt", Path, SelectedCams[ii]);	FILE *fp = fopen(Fname, "w+");
		for (int jj = 0; jj < npts; jj++)
		{
			fprintf(fp, "%d %d ", jj, CleanedFrameSyncedPoints[ii*npts + jj].size());
			for (int kk = 0; kk < CleanedFrameSyncedPoints[ii*npts + jj].size(); kk++)
				fprintf(fp, "%d %.4f %.4f ", CleanFrameSynID[ii*npts + jj][kk], CleanedFrameSyncedPoints[ii*npts + jj][kk].x, CleanedFrameSyncedPoints[ii*npts + jj][kk].y);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	delete[]A, delete[]B, delete[]P, delete[]tP, delete[]passed;
	if (createdMem)
		delete[]CleanFrameSynID, delete[]CleanedFrameSyncedPoints;

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
	GeometricConstraintSyncDriver(Path, nCams, npts, 0, nframes, TemporalSearchRange, false);*/

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
		FrameSync[ii] = dummy;
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

//Nonlinear Optimization for Temporal Alignement using BA geometric constraint
struct TemporalOptimInterpStationaryCameraCeres {
	TemporalOptimInterpStationaryCameraCeres(double *Pin, double *ParaXin, double *ParaYin, int frameIDin, int nframesIn, int interpAlgoIn)
	{
		P = Pin;
		ParaX = ParaXin, ParaY = ParaYin;
		double x = ParaXin[0], y = ParaX[0];
		frameID = frameIDin, nframes = nframesIn, interpAlgo = interpAlgoIn;
	}

	template <typename T>	bool operator()(const T* const XYZ, const T* const F, T* residuals) 	const
	{
		double Fi = F[0] + frameID;
		double Sx[3], Sy[3];
		if (Fi < 0.0)
			Fi = 0.0;
		if (Fi>nframes - 1)
			Fi = nframes - 1;

		double denum = P[8] * XYZ[0] + P[9] * XYZ[1] + P[10] * XYZ[2] + P[11];
		double numX = P[0] * XYZ[0] + P[1] * XYZ[1] + P[2] * XYZ[2] + P[3];
		double numY = P[4] * XYZ[0] + P[5] * XYZ[1] + P[6] * XYZ[2] + P[7];

		Get_Value_Spline(ParaX, nframes, 1, Fi, 0, Sx, -1, interpAlgo);
		Get_Value_Spline(ParaY, nframes, 1, Fi, 0, Sy, -1, interpAlgo);

		residuals[0] = numX / denum - Sx[0];
		residuals[1] = numY / denum - Sy[0];


		return true;
	}

	static ceres::CostFunction* Create(double *Pin, double *ParaXin, double *ParaYin, int frameIDin, int nframesIn, int interpAlgoIn)
	{
		return (new ceres::NumericDiffCostFunction<TemporalOptimInterpStationaryCameraCeres, ceres::CENTRAL, 2, 3, 1>(new TemporalOptimInterpStationaryCameraCeres(Pin, ParaXin, ParaYin, frameIDin, nframesIn, interpAlgoIn)));
	}

	int frameID, nframes, interpAlgo;
	double F;
	double *ParaX, *ParaY, *P;
};
struct TemporalOptimInterpMovingCameraCeres {
	TemporalOptimInterpMovingCameraCeres(double *AllPin, double *AllKin, double *AllQin, double *AllRin, double *AllCin, double *ParaCamCenterXIn, double *ParaCamCenterYIn, double *ParaCamCenterZIn, double *ParaXin, double *ParaYin, int frameIDin, int nframesIn, int interpAlgoIn)
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
		return (new ceres::NumericDiffCostFunction<TemporalOptimInterpMovingCameraCeres, ceres::CENTRAL, 2, 3, 1>(new TemporalOptimInterpMovingCameraCeres(AllPin, AllKin, AllQin, AllRin, AllCin, ParaCamCenterXIn, ParaCamCenterYin, ParaCamCenterZin, ParaXin, ParaYin, frameIDin, nframesIn, interpAlgoIn)));
	}

	int frameID, nframes, interpAlgo;
	double F;
	double *ParaCamCenterX, *ParaCamCenterY, *ParaCamCenterZ, *ParaX, *ParaY;
	double *AllP, *AllK, *AllQ, *AllR, *AllC;
};
int TemporalOptimInterp()
{
	const int nCams = 3, npts = 4;
	PerCamNonRigidTrajectory CamTraj[nCams];

	//PrepareTrajectoryInfo(CamTraj, nCams, npts);

	//Interpolate the trajectory of 2d tracks
	int maxPts = 0;
	for (int ii = 0; ii < npts; ii++)
	{
		for (int jj = 0; jj < nCams; jj++)
		{
			int npts = 0;
			for (int kk = 0; kk < CamTraj[jj].Track3DInfo[ii].nf; kk++)
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
		for (int ii = 0; ii < CamTraj[jj].npts; ii++)
		{
			int nf = CamTraj[jj].Track3DInfo[ii].nf;
			CamTraj[jj].Track2DInfo[ii].ParaX = new double[nf];
			CamTraj[jj].Track2DInfo[ii].ParaY = new double[nf];

			for (int kk = 0; kk < nf; kk++)
				x[kk] = CamTraj[jj].Track2DInfo[ii].uv[kk].x, y[kk] = CamTraj[jj].Track2DInfo[ii].uv[kk].y;
			Generate_Para_Spline(x, CamTraj[jj].Track2DInfo[ii].ParaX, nf, 1, InterpAlgo);
			Generate_Para_Spline(y, CamTraj[jj].Track2DInfo[ii].ParaY, nf, 1, InterpAlgo);
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


	ceres::Problem problem;
	double Error = 0.0;
	for (int ii = 0; ii < npts; ii++)
	{
		//find maxnf
		int maxnf = 0, maxCam = 0;
		for (int jj = 0; jj < nCams; jj++)
			if (maxnf < CamTraj[jj].Track3DInfo[ii].nf)
				maxnf = CamTraj[jj].Track3DInfo[ii].nf, maxCam = jj;

		for (int kk = 0; kk < maxnf; kk++)
		{
			for (int jj = 0; jj < nCams; jj++)
			{
				if (kk>CamTraj[jj].Track3DInfo[ii].nf || kk >= CamTraj[jj].R.size())
					continue;

				double Fi = CamTraj[jj].F + kk;
				if (Fi < 0.0)
					Fi = 0.0;
				if (Fi>CamTraj[jj].Track3DInfo[ii].nf - 2)
					Fi = CamTraj[jj].Track3DInfo[ii].nf - 2;
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
				Get_Value_Spline(CamTraj[jj].Track2DInfo[ii].ParaX, CamTraj[jj].Track3DInfo[ii].nf, 1, Fi, 0, &Sx, -1, InterpAlgo);
				Get_Value_Spline(CamTraj[jj].Track2DInfo[ii].ParaY, CamTraj[jj].Track3DInfo[ii].nf, 1, Fi, 0, &Sy, -1, InterpAlgo);

				double residualsX = numX / denum - Sx;
				double residualsY = numY / denum - Sy;
				double Residual = residualsX*residualsX + residualsY*residualsY;
				Error += Residual;

				//ceres::CostFunction* cost_function = TemporalOptimInterpStationaryCameraCeres::Create(CamTraj[jj].P[kk].P, CamTraj[jj].Track2DInfo[ii].ParaX, CamTraj[jj].Track2DInfo[ii].ParaY, kk, CamTraj[jj].Track3DInfo[ii].npts, InterpAlgo);
				ceres::CostFunction* cost_function = TemporalOptimInterpMovingCameraCeres::Create(&AllPMatrix[12 * jj*maxPts], &AllKMatrix[9 * jj*maxPts], &AllQuaternion[4 * jj*maxPts], &AllRotationMat[9 * jj*maxPts], &AllCamCenter[3 * jj*maxPts],
					ParaCamCenterX + jj*maxPts, ParaCamCenterY + jj*maxPts, ParaCamCenterZ + jj*maxPts, CamTraj[jj].Track2DInfo[ii].ParaX, CamTraj[jj].Track2DInfo[ii].ParaY, kk, CamTraj[jj].Track3DInfo[ii].nf, InterpAlgo);
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
	for (int ii = 0; ii < npts; ii++)
	{
		//find maxnf
		int maxnf = 0, maxCam = 0;
		for (int jj = 0; jj < nCams; jj++)
			if (maxnf < CamTraj[jj].Track3DInfo[ii].nf)
				maxnf = CamTraj[jj].Track3DInfo[ii].nf, maxCam = jj;

		fprintf(fp, "3D track %d \n", ii);
		for (int kk = 0; kk < maxnf; kk++)
		{
			for (int jj = 0; jj < nCams; jj++)
			{
				if (kk>CamTraj[jj].Track3DInfo[ii].nf || kk >= CamTraj[jj].R.size())
					continue;

				double Fi = CamTraj[jj].F + kk;
				if (Fi < 0.0)
					Fi = 0.0;
				if (Fi>CamTraj[jj].Track3DInfo[ii].nf - 2)
					Fi = CamTraj[jj].Track3DInfo[ii].nf - 2;
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
				Get_Value_Spline(CamTraj[jj].Track2DInfo[ii].ParaX, CamTraj[jj].Track3DInfo[ii].nf, 1, Fi, 0, &Sx, -1, InterpAlgo);
				Get_Value_Spline(CamTraj[jj].Track2DInfo[ii].ParaY, CamTraj[jj].Track3DInfo[ii].nf, 1, Fi, 0, &Sy, -1, InterpAlgo);

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
	//WriteTrajectory(CamTraj, nCams, npts, 1.0);
	delete[]AllRotationMat, delete[]AllPMatrix, delete[]AllKMatrix, delete[]AllQuaternion, delete[]AllCamCenter;
	delete[]ParaCamCenterX, delete[]ParaCamCenterY, delete[]ParaCamCenterZ;

	return 0;
}
int TemporalOptimInterpNew(char *Path, double *Offset)
{
	const int nCams = 3, npts = 4, startF = 0, stopF = 150;

	VideoData *VideoInfo = new VideoData[nCams];
	PerCamNonRigidTrajectory CamTraj[nCams];
	PrepareTrajectoryInfo(Path, VideoInfo, CamTraj, Offset, nCams, npts, startF, stopF);

	//Initialize temporal info
	for (int ii = 0; ii < nCams; ii++)
		CamTraj[ii].F = Offset[ii];

	int InterpAlgo = 1;
	double *x = new double[stopF], *y = new double[stopF], *z = new double[stopF];
	double* AllQuaternion = new double[4 * nCams*stopF], *AllRotationMat = new double[9 * nCams*stopF], *AllCamCenter = new double[3 * nCams*stopF];
	double *AllKMatrix = new double[9 * nCams*stopF], *AllPMatrix = new double[12 * nCams*stopF];
	double *ParaCamCenterX = new double[nCams*stopF], *ParaCamCenterY = new double[nCams*stopF], *ParaCamCenterZ = new double[nCams*stopF];

	for (int jj = 0; jj < nCams; jj++)
	{
		//Interpolate the trajectory of 2d tracks
		for (int ii = 0; ii < CamTraj[jj].npts; ii++)
		{
			int nf = CamTraj[jj].Track3DInfo[ii].nf;
			CamTraj[jj].Track2DInfo[ii].ParaX = new double[nf];
			CamTraj[jj].Track2DInfo[ii].ParaY = new double[nf];

			for (int kk = 0; kk < nf; kk++)
				x[kk] = CamTraj[jj].Track2DInfo[ii].uv[kk].x, y[kk] = CamTraj[jj].Track2DInfo[ii].uv[kk].y;
			Generate_Para_Spline(x, CamTraj[jj].Track2DInfo[ii].ParaX, nf, 1, InterpAlgo);
			Generate_Para_Spline(y, CamTraj[jj].Track2DInfo[ii].ParaY, nf, 1, InterpAlgo);
		}

		for (int ii = 0; ii < stopF; ii++)
		{
			if (ii >= CamTraj[jj].R.size())
				continue;
			for (int kk = 0; kk < 9; kk++)
				AllKMatrix[9 * jj * stopF + 9 * ii + kk] = CamTraj[jj].K[ii].K[kk];
			for (int kk = 0; kk < 3; kk++)
				AllCamCenter[3 * jj * stopF + 3 * ii + kk] = CamTraj[jj].C[ii].C[kk];
			for (int kk = 0; kk < 9; kk++)
				AllRotationMat[9 * jj * stopF + 9 * ii + kk] = CamTraj[jj].R[ii].R[kk];
			for (int kk = 0; kk < 4; kk++)
				AllQuaternion[4 * jj * stopF + 4 * ii + kk] = CamTraj[jj].Q[ii].quad[kk];
			for (int kk = 0; kk < 12; kk++)
				AllPMatrix[12 * jj * stopF + 12 * ii + kk] = CamTraj[jj].P[ii].P[kk];
		}

		for (int ii = 0; ii < stopF; ii++)
		{
			if (ii >= CamTraj[jj].R.size())
				continue;
			x[ii] = AllCamCenter[3 * jj * stopF + 3 * ii];
			y[ii] = AllCamCenter[3 * jj * stopF + 3 * ii + 1];
			z[ii] = AllCamCenter[3 * jj * stopF + 3 * ii + 2];
		}
		Generate_Para_Spline(x, ParaCamCenterX + jj*stopF, stopF, 1, InterpAlgo);
		Generate_Para_Spline(y, ParaCamCenterY + jj*stopF, stopF, 1, InterpAlgo);
		Generate_Para_Spline(z, ParaCamCenterZ + jj*stopF, stopF, 1, InterpAlgo);
	}
	delete[]x, delete[]y, delete[]z;

	printf("Inital offset: ");
	for (int ii = 0; ii < nCams; ii++)
		printf("%f ", CamTraj[ii].F);
	printf("\n");


	ceres::Problem problem;
	double Error = 0.0;
	for (int ii = 0; ii < npts; ii++)
	{
		//find maxtracks
		int maxTracks = 0, maxCam = 0;
		for (int jj = 0; jj < nCams; jj++)
			if (maxTracks < CamTraj[jj].Track3DInfo[ii].nf)
				maxTracks = CamTraj[jj].Track3DInfo[ii].nf, maxCam = jj;

		for (int kk = 0; kk < maxTracks; kk++)
		{
			for (int jj = 0; jj < nCams; jj++)
			{
				if (kk>CamTraj[jj].Track3DInfo[ii].nf || kk >= CamTraj[jj].R.size())
					continue;

				double Fi = CamTraj[jj].F + kk;
				if (Fi < 0.0)
					Fi = 0.0;
				if (Fi>CamTraj[jj].Track3DInfo[ii].nf - 2)
					Fi = CamTraj[jj].Track3DInfo[ii].nf - 2;
				int lFi = (int)Fi, uFi = lFi + 1, rFi = (int)(Fi + 0.5);
				double fFi = Fi - lFi;

				double K[9], C[3], Q[4], R[9], RT[12], P[12];
				for (int ll = 0; ll < 9; ll++)
					K[ll] = AllKMatrix[9 * jj*stopF + 9 * lFi + ll];
				for (int ll = 0; ll < 3; ll++)
					C[ll] = (1.0 - fFi)*AllCamCenter[3 * jj*stopF + 3 * lFi + ll] + fFi*AllCamCenter[3 * jj*stopF + 3 * uFi + ll]; //linear interpolation

				for (int ll = 0; ll < 4; ll++)
					Q[ll] = AllQuaternion[4 * jj*stopF + 4 * lFi + ll];

				//QuaternionLinearInterp(&AllQuaternion[4 * jj*stopF + 4 * lFi], &AllQuaternion[4 * jj*stopF + 4 * uFi], Q, fFi);//linear interpolation
				//Get_Value_Spline(ParaCamCenterX + jj*stopF, stopF, 1, Fi, 0, &C[0], -1, InterpAlgo);
				//Get_Value_Spline(ParaCamCenterY + jj*stopF, stopF, 1, Fi, 0, &C[1], -1, InterpAlgo);
				//Get_Value_Spline(ParaCamCenterZ + jj*stopF, stopF, 1, Fi, 0, &C[2], -1, InterpAlgo);

				Quaternion2Rotation(Q, R);
				AssembleRT(R, C, RT, true);
				AssembleP(K, RT, P);

				double XYZ[] = { CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk], CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk + 1], CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk + 2] };
				double denum = P[8] * XYZ[0] + P[9] * XYZ[1] + P[10] * XYZ[2] + P[11];
				double numX = P[0] * XYZ[0] + P[1] * XYZ[1] + P[2] * XYZ[2] + P[3];
				double numY = P[4] * XYZ[0] + P[5] * XYZ[1] + P[6] * XYZ[2] + P[7];

				double Sx, Sy;
				Get_Value_Spline(CamTraj[jj].Track2DInfo[ii].ParaX, CamTraj[jj].Track3DInfo[ii].nf, 1, Fi, 0, &Sx, -1, InterpAlgo);
				Get_Value_Spline(CamTraj[jj].Track2DInfo[ii].ParaY, CamTraj[jj].Track3DInfo[ii].nf, 1, Fi, 0, &Sy, -1, InterpAlgo);

				double residualsX = numX / denum - Sx;
				double residualsY = numY / denum - Sy;
				double Residual = residualsX*residualsX + residualsY*residualsY;
				Error += Residual;

				//ceres::CostFunction* cost_function = TemporalOptimInterpStationaryCameraCeres::Create(CamTraj[jj].P[kk].P, CamTraj[jj].Track2DInfo[ii].ParaX, CamTraj[jj].Track2DInfo[ii].ParaY, kk, CamTraj[jj].Track3DInfo[ii].nf, InterpAlgo);
				ceres::CostFunction* cost_function = TemporalOptimInterpMovingCameraCeres::Create(&AllPMatrix[12 * jj*stopF], &AllKMatrix[9 * jj*stopF], &AllQuaternion[4 * jj*stopF], &AllRotationMat[9 * jj*stopF], &AllCamCenter[3 * jj*stopF],
					ParaCamCenterX + jj*stopF, ParaCamCenterY + jj*stopF, ParaCamCenterZ + jj*stopF, CamTraj[jj].Track2DInfo[ii].ParaX, CamTraj[jj].Track2DInfo[ii].ParaY, kk, CamTraj[jj].Track3DInfo[ii].nf, InterpAlgo);
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
	for (int ii = 0; ii < npts; ii++)
	{
		//find maxtracks
		int maxTracks = 0, maxCam = 0;
		for (int jj = 0; jj < nCams; jj++)
			if (maxTracks < CamTraj[jj].Track3DInfo[ii].nf)
				maxTracks = CamTraj[jj].Track3DInfo[ii].nf, maxCam = jj;

		fprintf(fp, "3D track %d \n", ii);
		for (int kk = 0; kk < maxTracks; kk++)
		{
			for (int jj = 0; jj < nCams; jj++)
			{
				if (kk>CamTraj[jj].Track3DInfo[ii].nf || kk >= CamTraj[jj].R.size())
					continue;

				double Fi = CamTraj[jj].F + kk;
				if (Fi < 0.0)
					Fi = 0.0;
				if (Fi>CamTraj[jj].Track3DInfo[ii].nf - 2)
					Fi = CamTraj[jj].Track3DInfo[ii].nf - 2;
				int lFi = (int)Fi, uFi = lFi + 1, rFi = (int)(Fi + 0.5);
				double fFi = Fi - lFi;

				double K[9], C[3], Q[4], R[9], RT[12], P[12];
				for (int ll = 0; ll < 9; ll++)
					K[ll] = AllKMatrix[9 * jj*stopF + 9 * lFi + ll];
				for (int ll = 0; ll < 3; ll++)
					C[ll] = (1.0 - fFi)*AllCamCenter[3 * jj*stopF + 3 * lFi + ll] + fFi*AllCamCenter[3 * jj*stopF + 3 * uFi + ll]; //linear interpolation

				for (int ll = 0; ll < 4; ll++)
					Q[ll] = AllQuaternion[4 * jj*stopF + 4 * lFi + ll];
				//QuaternionLinearInterp(&AllQuaternion[4 * jj*stopF + 4 * lFi], &AllQuaternion[4 * jj*stopF + 4 * uFi], Q, fFi);//linear interpolation
				//Get_Value_Spline(ParaCamCenterX + jj*stopF, stopF, 1, Fi, 0, &C[0], -1, InterpAlgo);
				//Get_Value_Spline(ParaCamCenterY + jj*stopF, stopF, 1, Fi, 0, &C[1], -1, InterpAlgo);
				//Get_Value_Spline(ParaCamCenterZ + jj*stopF, stopF, 1, Fi, 0, &C[2], -1, InterpAlgo);

				Quaternion2Rotation(Q, R);
				AssembleRT(R, C, RT, true);
				AssembleP(K, RT, P);

				double XYZ[] = { CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk], CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk + 1], CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk + 2] };
				double denum = P[8] * XYZ[0] + P[9] * XYZ[1] + P[10] * XYZ[2] + P[11];
				double numX = P[0] * XYZ[0] + P[1] * XYZ[1] + P[2] * XYZ[2] + P[3];
				double numY = P[4] * XYZ[0] + P[5] * XYZ[1] + P[6] * XYZ[2] + P[7];

				double Sx, Sy;
				Get_Value_Spline(CamTraj[jj].Track2DInfo[ii].ParaX, CamTraj[jj].Track3DInfo[ii].nf, 1, Fi, 0, &Sx, -1, InterpAlgo);
				Get_Value_Spline(CamTraj[jj].Track2DInfo[ii].ParaY, CamTraj[jj].Track3DInfo[ii].nf, 1, Fi, 0, &Sy, -1, InterpAlgo);

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
	//WriteTrajectory(CamTraj, nCams, npts, 1.0);
	delete[]AllRotationMat, delete[]AllPMatrix, delete[]AllKMatrix, delete[]AllQuaternion, delete[]AllCamCenter;
	delete[]ParaCamCenterX, delete[]ParaCamCenterY, delete[]ParaCamCenterZ;

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

struct LeastActionCostCeres {
	LeastActionCostCeres(int frameID1, int frameID2, double ialpha, double Tscale, double epsilon, double lamda, int motionPriorPower) : frameID1(frameID1), frameID2(frameID2), ialpha(ialpha), Tscale(Tscale), epsilon(epsilon), lamda(lamda), motionPriorPower(motionPriorPower){	}

	template <typename T>	bool operator()(const T* const xyz1, const T* const xyz2, const T* const timeStamp1, const T* const timeStamp2, T* residuals) 	const
	{
		T difX = xyz2[0] - xyz1[0], difY = xyz2[1] - xyz1[1], difZ = xyz2[2] - xyz1[2];
		T  t1 = (T)((timeStamp1[0] + (T)(1.0*frameID1)) * ialpha*Tscale);
		T  t2 = (T)((timeStamp2[0] + (T)(1.0*frameID2)) * ialpha*Tscale);

		T cost;
		if (motionPriorPower == 4)
			cost = pow(difX*difX + difY*difY + difZ*difZ, 2) / abs(pow(t2 - t1, 3) + (T)epsilon); //mv4dt
		else if (motionPriorPower == 2)
			cost = (difX*difX + difY*difY + difZ*difZ) / abs(t2 - t1 + (T)epsilon);
		residuals[0] = sqrt(cost) / (T)lamda;

		return true;
	}
	template <typename T>	bool operator()(const T* const xyz1, const T* const xyz2, const T* const timeStamp, T* residuals) 	const
	{
		T difX = xyz2[0] - xyz1[0], difY = xyz2[1] - xyz1[1], difZ = xyz2[2] - xyz1[2];
		T  t1 = (T)((timeStamp[0] + (T)(1.0*frameID1))* ialpha*Tscale);
		T  t2 = (T)((timeStamp[0] + (T)(1.0* frameID2))* ialpha*Tscale);

		T cost;
		if (motionPriorPower == 4)
			cost = pow(difX*difX + difY*difY + difZ*difZ, 2) / abs(pow(t2 - t1, 3) + (T)epsilon); //mv4dt
		else if (motionPriorPower == 2)
			cost = (difX*difX + difY*difY + difZ*difZ) / abs(t2 - t1 + (T)epsilon);
		residuals[0] = sqrt(cost) / (T)lamda;

		return true;
	}
	static ceres::CostFunction* CreateAutoDiff(int frameID1, int frameID2, double ialpha, double Tscale, double epsilon, double lamda, int motionPriorPower)
	{
		return (new ceres::AutoDiffCostFunction<LeastActionCostCeres, 1, 3, 3, 1, 1>(new LeastActionCostCeres(frameID1, frameID2, ialpha, Tscale, epsilon, lamda, motionPriorPower)));
	}
	static ceres::CostFunction* CreateAutoDiffSame(int frameID1, int frameID2, double ialpha, double Tscale, double epsilon, double lamda, int motionPriorPower)
	{
		return (new ceres::AutoDiffCostFunction<LeastActionCostCeres, 1, 3, 3, 1>(new LeastActionCostCeres(frameID1, frameID2, ialpha, Tscale, epsilon, lamda, motionPriorPower)));
	}
	static ceres::CostFunction* CreateNumerDiff(int frameID1, int frameID2, double ialpha, double Tscale, double epsilon, double lamda, int motionPriorPower)
	{
		return (new ceres::NumericDiffCostFunction<LeastActionCostCeres, ceres::CENTRAL, 1, 3, 3, 1, 1>(new LeastActionCostCeres(frameID1, frameID2, ialpha, Tscale, epsilon, lamda, motionPriorPower)));
	}
	static ceres::CostFunction* CreateNumerDiffSame(int frameID1, int frameID2, double ialpha, double Tscale, double epsilon, double lamda, int motionPriorPower)
	{
		return (new ceres::NumericDiffCostFunction<LeastActionCostCeres, ceres::CENTRAL, 1, 3, 3, 1>(new LeastActionCostCeres(frameID1, frameID2, ialpha, Tscale, epsilon, lamda, motionPriorPower)));
	}

	int frameID1, frameID2, motionPriorPower;
	double ialpha, Tscale, epsilon, lamda;
};
struct LeastActionCostDynamicCeres {
	LeastActionCostDynamicCeres(int *VectorCamIDIn, int *VectorFrameIDIn, double *PIn, Point2d *pt2DIn, double lamda, double ialpha, double Tscale, int npts) :lamda(lamda), ialpha(ialpha), Tscale(Tscale), npts(npts)
	{
		VectorCamID = VectorCamIDIn, VectorFrameID = VectorFrameIDIn, P = PIn, pt2D = pt2DIn;
	}

	template <typename T>    bool operator()(T const* const* Parameters, T* residuals)     const
	{
		//cost = lamda*(PX-u)^2 + (1-lamda)*Prior
		T x, y, z, xo, yo, zo, numX, numY, denum, sqrtlamda1 = sqrt((T)lamda), lamda2 = (T)1.0 - lamda;

		//Projection cost 
		for (int ii = 0; ii < npts; ii++)
		{
			x = Parameters[ii][0], y = Parameters[ii][1], z = Parameters[ii][2];
			numX = (T)P[12 * ii] * x + (T)P[12 * ii + 1] * y + (T)P[12 * ii + 2] * z + (T)P[12 * ii + 3];
			numY = (T)P[12 * ii + 4] * x + (T)P[12 * ii + 5] * y + (T)P[12 * ii + 6] * z + (T)P[12 * ii + 7];
			denum = (T)P[12 * ii + 8] * x + (T)P[12 * ii + 9] * y + (T)P[12 * ii + 10] * z + (T)P[12 * ii + 11];

			residuals[2 * ii] = (T)sqrtlamda1*(numX / denum - (T)pt2D[ii].x);
			residuals[2 * ii + 1] = (T)sqrtlamda1*(numY / denum - (T)pt2D[ii].y);
		}

		//Action cost: START
		const int SplineOrder = 4;
		for (int ii = 1; ii < npts; ii++)
		{
			int camID = VectorCamID[ii - 1], frameID = VectorFrameID[ii - 1];
			T T1 = (Parameters[npts][camID] + (T)frameID)* (T)(ialpha*Tscale);
			xo = Parameters[ii - 1][0], yo = Parameters[ii - 1][1], zo = Parameters[ii - 1][2];

			camID = VectorCamID[ii], frameID = VectorFrameID[ii];
			T T2 = (Parameters[npts][camID] + (T)frameID)* (T)(ialpha*Tscale);;
			x = Parameters[ii][0], y = Parameters[ii][1], z = Parameters[ii][2];

			T dx = x - xo, dy = y - yo, dz = z - zo;
			T error = (T)lamda2*(dx*dx + dy*dy + dz*dz) / (T2 - T1 + (T)1.0e-16);
			residuals[2 * npts + ii - 1] = sqrt(max((T)(1.0e-16), error));
		}

		return true;
	}

	int npts;
	double lamda, ialpha, Tscale;
	int *VectorCamID, *VectorFrameID;
	double  *P;
	Point2d *pt2D;
};
struct LeastActionCostSplineCeres {
	LeastActionCostSplineCeres(int *VectorCamIDIn, int *VectorFrameIDIn, double *PIn, Point2d *pt2DIn, double *BreakPtsIn, double *PhiDataIn, double *PhiPriorIn, double *TimeStampDatain, double *TimeStampPriorIn, double *CoeffsIn, double *TempIn,
		double lamda, double ialpha, double Tscale, int nData, int UpsamplingRate, int nBreakPts) :lamda(lamda), ialpha(ialpha), Tscale(Tscale), nData(nData), nBreakPts(nBreakPts), UpsamplingRate(UpsamplingRate)
	{
		VectorCamID = VectorCamIDIn, VectorFrameID = VectorFrameIDIn, P = PIn, pt2D = pt2DIn, BreakPts = BreakPtsIn;
		PhiData = PhiDataIn, PhiPrior = PhiPriorIn, TimeStampData = TimeStampDatain, TimeStampPrior = TimeStampPriorIn, Coeffs = CoeffsIn, Temp = TempIn;//just to reserve ram space
	}

	template <typename T>    bool operator()(T const* const* Parameters, T* residuals)     const
	{
		//cost = lamda*(PX-u)^2 + (1-lamda)*Prior
		T x, y, z, numX, numY, denum, sqrtlamda1 = sqrt((T)lamda), lamda2 = (T)1.0 - lamda;

		//Projection cost 
		for (int ii = 0; ii < nData; ii++)
		{
			x = Parameters[0][3 * ii], y = Parameters[0][3 * ii + 1], z = Parameters[0][3 * ii + 2];
			numX = (T)P[12 * ii] * x + (T)P[12 * ii + 1] * y + (T)P[12 * ii + 2] * z + (T)P[12 * ii + 3];
			numY = (T)P[12 * ii + 4] * x + (T)P[12 * ii + 5] * y + (T)P[12 * ii + 6] * z + (T)P[12 * ii + 7];
			denum = (T)P[12 * ii + 8] * x + (T)P[12 * ii + 9] * y + (T)P[12 * ii + 10] * z + (T)P[12 * ii + 11];

			residuals[2 * ii] = (T)sqrtlamda1*(numX / denum - (T)pt2D[ii].x);
			residuals[2 * ii + 1] = (T)sqrtlamda1*(numY / denum - (T)pt2D[ii].y);
		}

		//Action cost: START
		const int SplineOrder = 4;
		for (int ii = 0; ii < nData; ii++)
		{
			int camID = VectorCamID[ii], frameID = VectorFrameID[ii];
			TimeStampData[ii] = (Parameters[1][camID] + frameID)* ialpha*Tscale;
		}
		//GenerateSplineBasisWithBreakPts(PhiData, NULL, TimeStampData, BreakPts, nData, nBreakPts, SplineOrder, 0);
		BSplineGetAllBasis(PhiData, TimeStampData, BreakPts, nData, nBreakPts, SplineOrder);

		int nCoeffs = nBreakPts + 2;
		for (int jj = 0; jj < 3; jj++)
		{
			for (int ii = 0; ii < nData; ii++)
				Temp[ii] = Parameters[0][3 * ii + jj];
			LS_Solution_Double(PhiData, Temp, nData, nCoeffs);
			for (int ii = 0; ii < nCoeffs; ii++)
				Coeffs[ii + jj*nCoeffs] = Temp[ii];
		}

		int nResamples = nData*UpsamplingRate;
		double PriorStep = (TimeStampData[nData - 1] - TimeStampData[0]) / nResamples;
		for (int ii = 0; ii < nResamples; ii++)
			TimeStampPrior[ii] = TimeStampData[0] + PriorStep*ii;
		//GenerateSplineBasisWithBreakPts(NULL, PhiPrior, TimeStampPrior, BreakPts, nResamples, nBreakPts, SplineOrder, 1);
		BSplineGetAllBasis(PhiPrior, TimeStampPrior, BreakPts, nResamples, nBreakPts, SplineOrder, 1, PhiPrior);

		double dx, dy, dz;
		for (int ii = 0; ii < nResamples; ii++)
		{
			dx = 0.0, dy = 0.0, dz = 0.0;
			for (int jj = 0; jj < nCoeffs; jj++)
			{
				if (PhiPrior[jj + ii*nCoeffs] < 0.00001)
					continue;
				dx += PhiPrior[jj + ii*nCoeffs] * Coeffs[jj],
					dy += PhiPrior[jj + ii*nCoeffs] * Coeffs[jj + nCoeffs],
					dz += PhiPrior[jj + ii*nCoeffs] * Coeffs[jj + 2 * nCoeffs];
			}
			Temp[ii] = dx*dx + dy*dy + dz*dz; //v^2
		}
		residuals[2 * nData] = sqrt(max(lamda2*SimpsonThreeEightIntegration(Temp, PriorStep, nResamples), 1.0e-16));

		return true;
	}

	int nData, UpsamplingRate, nBreakPts;
	double lamda, ialpha, Tscale;
	int *VectorCamID, *VectorFrameID;
	double  *P, *PhiData, *PhiPrior, *TimeStampData, *TimeStampPrior, *BreakPts, *Coeffs, *Temp;
	Point2d *pt2D;
};
struct LeastActionCost3DCeres {
	LeastActionCost3DCeres(double timeStamp1, double timeStamp2, double epsilon, double lamda, int motionPriorPower) : timeStamp1(timeStamp1), timeStamp2(timeStamp2), epsilon(epsilon), lamda(lamda), motionPriorPower(motionPriorPower){}

	template <typename T>	bool operator()(const T* const xyz1, const T* const xyz2, T* residuals) 	const
	{
		T difX = xyz2[0] - xyz1[0], difY = xyz2[1] - xyz1[1], difZ = xyz2[2] - xyz1[2];

		T cost;
		if (motionPriorPower == 4)
			cost = pow(difX*difX + difY*difY + difZ*difZ, 2) / abs(pow(timeStamp2 - timeStamp1, 3) + (T)epsilon); //mv4dt
		else if (motionPriorPower == 2)
			cost = (difX*difX + difY*difY + difZ*difZ) / abs(timeStamp2 - timeStamp1 + (T)epsilon);
		residuals[0] = sqrt(cost) / (T)lamda;
		return true;
	}
	static ceres::CostFunction* CreateAutoDiff(double Stamp1, double Stamp2, double epsilon, double lamda, int motionPriorPower)
	{
		return (new ceres::AutoDiffCostFunction<LeastActionCost3DCeres, 1, 3, 3>(new LeastActionCost3DCeres(Stamp1, Stamp2, epsilon, lamda, motionPriorPower)));
	}
	static ceres::CostFunction* CreateNumerDiff(double Stamp1, double Stamp2, double epsilon, double lamda, int motionPriorPower)
	{
		return (new ceres::NumericDiffCostFunction<LeastActionCost3DCeres, ceres::CENTRAL, 1, 3, 3>(new LeastActionCost3DCeres(Stamp1, Stamp2, epsilon, lamda, motionPriorPower)));
	}
	int motionPriorPower;
	double timeStamp1, timeStamp2, epsilon, lamda;
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

		residuals[0] = (numX / denum - T(observed_x)) / (T)(lamda);
		residuals[1] = (numY / denum - T(observed_y)) / (T)(lamda);

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
struct KineticEnergy {
	KineticEnergy(){}

	template <typename T>	bool operator()(const T* const xy, T* residuals) 	const
	{
		residuals[0] = ((T)1.0 - xy[0]) * ((T)1.0 - xy[0]) + (T)100.0 * (xy[1] - xy[0] * xy[0]) * (xy[1] - xy[0] * xy[0]);

		return true;
	}

	static ceres::CostFunction* Create()
	{
		return (new ceres::AutoDiffCostFunction<KineticEnergy, 1, 2>(new KineticEnergy()));
	}
};
struct PotentialEnergy {
	PotentialEnergy(double g) :g(g){}

	template <typename T>	bool operator()(const T* const Y, T* residuals) 	const
	{
		residuals[0] = ((T)-g)*Y[0];

		return true;
	}

	static ceres::CostFunction* Create(double g)
	{
		return (new ceres::AutoDiffCostFunction<PotentialEnergy, 1, 1>(new PotentialEnergy(g)));
	}
	double g;
};
class LeastActionProblem : public ceres::FirstOrderFunction {
public:
	LeastActionProblem(double *AllQ, double *AllU, int *PtsPerTrack, int totalPts, int nCams, int nPperTracks, int npts, double lamdaI) :lamdaImg(lamdaI), totalPts(totalPts), nCams(nCams), nPperTracks(nPperTracks)
	{
		gravity = 9.88;
		lamdaImg = lamdaI;
		PointsPerTrack = PtsPerTrack;
		AllQmat = AllQ, AllUmat = AllU;
	}
	virtual ~LeastActionProblem() {}

	virtual bool Evaluate(const double* parameters, double* cost, double* gradient) const
	{
		/*//Kinetic energy depends on velocity computed at multiple points--> get splited
		for (int trackId = 0; trackId < npts; trackId++)
		{
		for (int pid = 0; pid < PointsPerTrack[trackId]; pid++)
		{
		ceres::CostFunction* cost_functionE = KineticEnergy::Create();
		cost_functionE->Evaluate(&parameters, &cost[0], NULL);
		if (gradient != NULL)
		cost_functionE->Evaluate(&parameters, &cost[0], &gradient);
		}
		}

		//Potential energy
		for (int trackId = 0; trackId < npts; trackId++)
		{
		for (int pid = 0; pid < PointsPerTrack[trackId]; pid++)
		{
		ceres::CostFunction* cost_functionP = PotentialEnergy::Create(gravity);
		cost_functionP->Evaluate(&parameters, &cost[0], NULL);

		if (gradient != NULL)
		cost_functionP->Evaluate(&parameters, &cost[0], &gradient);
		}
		}

		//Potential energy + Image constraint
		int currentPts = 0;
		for (int trackId = 0; trackId < npts; trackId++)
		{
		for (int pid = 0; pid < PointsPerTrack[trackId]; pid++)
		{
		ceres::CostFunction* cost_functionI = IdealAlgebraicReprojectionCeres::Create(&AllQmat[6 * pid], &AllUmat[2 * pid], lamdaImg);

		if (gradient != NULL)
		cost_functionI->Evaluate(&parameters, &cost[currentPts], &gradient);
		else
		cost_functionI->Evaluate(&parameters, &cost[currentPts], NULL);
		currentPts++;
		}
		}*/

		return true;
	}

	virtual int NumParameters() const
	{
		return totalPts + nCams;
	}

	int nCams, totalPts, nPperTracks, npts;
	int *PointsPerTrack;
	double *AllQmat, *AllUmat, lamdaImg, gravity;
};
int CeresLeastActionNonlinearOptim()
{
	int totalPts = 100, nCams = 2, nPperTracks = 100, npts = 1;

	//double *AllQ, *AllU;
	int *PointsPerTrack = new int[npts];


	double lamdaI = 10.0;
	for (int ii = 0; ii < npts; ii++)
		PointsPerTrack[ii] = nPperTracks;

	double *parameters = new double[totalPts * 3 + nCams];

	ceres::GradientProblemSolver::Options options;
	options.minimizer_progress_to_stdout = true;
	options.max_num_iterations = 1000;

	ceres::GradientProblemSolver::Summary summary;
	//ceres::GradientProblem problem(new LeastActionProblem(AllQ, AllU, PointsPerTrack, totalPts, nCams, nPperTracks, npts, lamdaI));
	//ceres::Solve(options, problem, parameters, &summary);

	std::cout << summary.FullReport() << "\n";

	return 0;
}

double LeastActionError(double *xyz1, double *xyz2, double *timeStamp1, double *timeStamp2, int frameID1, int frameID2, double ialpha, double Tscale, double eps, int motionPriorPower)
{
	double difX = xyz2[0] - xyz1[0], difY = xyz2[1] - xyz1[1], difZ = xyz2[2] - xyz1[2];
	double  t1 = (timeStamp1[0] + frameID1) * ialpha*Tscale;
	double  t2 = (timeStamp2[0] + frameID2) * ialpha*Tscale;

	double cost;
	if (motionPriorPower == 4)
		cost = pow(difX*difX + difY*difY + difZ*difZ, 2) / (pow(t2 - t1, 3) + eps); //mv^4*dt
	else if (motionPriorPower == 2)
		cost = (difX*difX + difY*difY + difZ*difZ) / (t2 - t1 + eps); //mv^2*dt

	return cost;
}
double LeastActionError(double *xyz1, double *xyz2, double timeStamp1, double timeStamp2, double eps, int motionPriorPower)
{
	double difX = xyz2[0] - xyz1[0], difY = xyz2[1] - xyz1[1], difZ = xyz2[2] - xyz1[2];
	double cost;
	if (motionPriorPower == 4)
		cost = pow(difX*difX + difY*difY + difZ*difZ, 2) / (pow(timeStamp2 - timeStamp1, 3) + eps); //mv^4*dt
	else if (motionPriorPower == 2)
		cost = (difX*difX + difY*difY + difZ*difZ) / (timeStamp2 - timeStamp1 + eps); //mv^2*dt

	return cost;
}
double LeastActionError(double *xyz1, double *xyz2, double *xyz3, double *timeStamp1, double *timeStamp2, double *timeStamp3, int frameID1, int frameID2, int frameID3, double ialpha, double Tscale, double eps, int motionPriorPower)
{
	double  t1 = (timeStamp1[0] + frameID1) * ialpha*Tscale;
	double  t2 = (timeStamp2[0] + frameID2) * ialpha*Tscale;
	double  t3 = (timeStamp3[0] + frameID3) * ialpha*Tscale;
	Point3d X1(xyz1[0], xyz1[1], xyz1[2]), X2(xyz2[0], xyz2[1], xyz2[2]), X3(xyz3[0], xyz3[1], xyz3[2]);
	Point3d num1 = X2 - X3, denum1 = ProductPoint3d(X1 - X2, X1 - X3),
		num2 = 2.0*X2 - X1 - X3, denum2 = ProductPoint3d(X2 - X1, X2 - X3),
		num3 = X2 - X1, denum3 = ProductPoint3d(X3 - X1, X3 - X2);
	Point3d dv = ProductPoint3d(X1, DividePoint3d(num1, denum1)) + ProductPoint3d(X2, DividePoint3d(num2, denum2)) + ProductPoint3d(X3, DividePoint3d(num3, denum3));

	double cost;
	if (motionPriorPower == 4)
		cost = pow(dv.x*dv.x + dv.y*dv.y + dv.z*dv.z, 2) / (pow(t2 - t1, 3) + eps); //mv^4*dt
	else if (motionPriorPower == 2)
		cost = (dv.x*dv.x + dv.y*dv.y + dv.z*dv.z) / (t2 - t1 + eps); //mv^2*dt

	return cost;
}
double ComputeActionCostDriver(char *Fname, int motionPriorPower = 2)
{
	double x, y, z, t;
	int count = 0;
	const int nframes = 5000;

	int id[nframes];
	Point3d P3dTemp[nframes];
	double timeStamp[nframes], P3d[3 * nframes];

	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return 1;
	}
	while (fscanf(fp, "%lf %lf %lf %lf", &x, &y, &z, &t) != EOF)
	{
		id[count] = count;
		timeStamp[count] = t;
		P3dTemp[count].x = x, P3dTemp[count].y = y, P3dTemp[count].z = z;
		count++;
	}
	fclose(fp);

	Quick_Sort_Double(timeStamp, id, 0, count - 1);

	for (int ii = 0; ii < count; ii++)
		P3d[3 * ii] = P3dTemp[id[ii]].x, P3d[3 * ii + 1] = P3dTemp[id[ii]].y, P3d[3 * ii + 2] = P3dTemp[id[ii]].z;

	double ActionCost = 0.0, costi;
	for (int ii = 0; ii < count - 1; ii++)
	{
		costi = LeastActionError(&P3d[3 * ii], &P3d[3 * ii + 3], &timeStamp[ii], &timeStamp[ii + 1], 0, 0, 1, 1, 1.0e-6, motionPriorPower);
		ActionCost += costi;
	}
	printf("Ac: %.6e\n", ActionCost);

	return ActionCost;
}
double ComputeActionEnergy(vector<Point3d> traj, vector<double>Time, double eps, double g, int ApproxOrder)
{
	double Cost = 0.0;
	if (ApproxOrder == 1)
	{
		double T = 0.0, V = 0.0;
		for (int ii = 0; ii < traj.size() - 1; ii++)
		{
			double dT = Time[ii + 1] - Time[ii];
			double dx = traj[ii].x - traj[ii + 1].x, dy = traj[ii].y - traj[ii + 1].y, dz = traj[ii].z - traj[ii + 1].z;
			double vx = dx / (dT + eps), vy = dy / (dT + eps), vz = dz / (dT + eps);
			double v = sqrt(vx*vx + vy*vy + vz*vz);
			double cost = pow(v, 2)*(dT + eps);
			T += cost;
		}
		for (int ii = 0; ii < traj.size() - 1; ii++)
		{
			double dT = Time[ii + 1] - Time[ii];
			double cost = g*(traj[ii].y + traj[ii + 1].y)*dT;
			V += cost;
		}
		Cost = T - V;
	}
	else
	{
		/*//Second order approxiamtaion of speed
		Point3d vel[2000];
		for (int ii = 0; ii < traj.size() - 2; ii++)
		{
		double  t1 = Time[ii], t2 = Time[ii + 1], t3 = Time[ii + 2];

		double num1 = t2 - t3, num2 = 2.0*t2 - t1 - t3, num3 = t2 - t1;
		double denum1 = (t1 - t2)*(t1 - t3) + eps, denum2 = (t2 - t1)*(t2 - t3) + eps, denum3 = (t3 - t1)*(t3 - t2) + eps;
		double a1 = num1 / denum1, a2 = num2 / denum2, a3 = num3 / denum3;

		double velX = a1*traj[ii].x + a2*traj[ii + 1].x + a3*traj[ii + 2].x;
		double velY = a1*traj[ii].y + a2*traj[ii + 1].y + a3*traj[ii + 2].y;
		double velZ = a1*traj[ii].z + a2*traj[ii + 1].z + a3*traj[ii + 2].z;

		vel[ii + 1].x = velX, vel[ii + 1].y = velY, vel[ii + 1].z = velZ;
		}
		vel[0] = vel[1]; vel[traj.size() - 1] = vel[traj.size() - 2];

		for (int ii = 0; ii < traj.size() - 1; ii++)
		{
		Point3d vel1 = vel[ii], vel2 = vel[ii + 1];
		double dT = Time[ii + 1] - Time[ii];
		double cost = 0.5*(pow(vel1.x, 2) + pow(vel2.x, 2))*(dT + eps) + 0.5*(pow(vel1.y, 2) + pow(vel2.y, 2))*(dT + eps) + 0.5*(pow(vel1.z, 2) + pow(vel2.z, 2))*(dT + eps);
		Cost += cost;
		}*/

		//accelatiaon via lagrange interpolation
		double accel[2000];
		double t1, t2, t3, denum1, denum3;
		int npts = traj.size();
		t1 = Time[0], t2 = Time[1];
		denum1 = pow(t2 - t1, 2);
		accel[0] = (-traj[0].x + traj[1].x) / denum1 + (-traj[0].y + traj[1].y) / denum1 + (-traj[0].z + traj[1].z) / denum1;

		for (int ii = 1; ii < npts - 1; ii++)
		{
			double  t1 = Time[ii - 1], t2 = Time[ii], t3 = Time[ii + 1];
			double denum1 = (t1 - t2)*(t1 - t3), denum2 = (t2 - t1)*(t2 - t3), denum3 = (t3 - t1)*(t3 - t2);

			double accelX = traj[ii - 1].x / denum1 + traj[ii].x / denum2 + traj[ii + 1].x / denum3;
			double accelY = traj[ii - 1].y / denum1 + traj[ii].y / denum2 + traj[ii + 1].y / denum3;
			double accelZ = traj[ii - 1].z / denum1 + traj[ii].z / denum2 + traj[ii + 1].z / denum3;

			accel[ii] = accelX*accelX + accelY*accelY + accelZ*accelZ;
		}
		t2 = Time[npts - 2], t3 = Time[npts - 1];
		denum3 = pow(t3 - t2, 2);
		accel[npts - 1] = (traj[npts - 2].x - traj[npts - 1].x) / denum3 + (traj[npts - 2].y - traj[npts - 1].y) / denum3 + (traj[npts - 2].z - traj[npts - 1].z) / denum3;

		Cost = 0.0;
		for (int ii = 0; ii <npts - 1; ii++)
			Cost += 0.5*(accel[ii] + accel[ii + 1])*(Time[ii + 1] - Time[ii]);
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

void MotionPrior_ML_Weighting(vector<ImgPtEle> *PerCam_UV, int ntracks, int nCams)
{
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		for (int camID = 0; camID < nCams; camID++)
		{
			for (int kk = 0; kk < PerCam_UV[camID*ntracks + trackID].size(); kk++)
			{
				ImgPtEle ptEle = PerCam_UV[camID*ntracks + trackID][kk];
				ptEle.K[2] = 0, ptEle.K[5] = 0;

				double Scale = PerCam_UV[camID*ntracks + trackID][kk].scale, canonicalScale = PerCam_UV[camID*ntracks + trackID][kk].canonicalScale;
				double std2d = PerCam_UV[camID*ntracks + trackID][kk].std2D;

				if (Scale >= canonicalScale)
				{
					double sigmaRetina[9], sigma2d[9] = { std2d*Scale / canonicalScale, 0, 0, 0, std2d*Scale / canonicalScale, 0, 0, 0, 1 };

					Map < Matrix < double, 3, 3, RowMajor > > eK(ptEle.K);
					Map < Matrix < double, 3, 3, RowMajor > > esigma2d(sigma2d);
					Map < Matrix < double, 3, 3, RowMajor > > esigmaRetina(sigmaRetina);
					Matrix3d eiK = eK.inverse();
					esigmaRetina = eiK*esigma2d*eiK.transpose();

					double depth = sqrt(pow(ptEle.pt3D.x - ptEle.camcenter[0], 2) + pow(ptEle.pt3D.y - ptEle.camcenter[1], 2) + pow(ptEle.pt3D.z - ptEle.camcenter[2], 2));
					PerCam_UV[camID*ntracks + trackID][kk].std3D = max(sigmaRetina[0], sigmaRetina[4]) * PerCam_UV[camID*ntracks + trackID][kk].pixelSizeToMm*depth;
				}
				else
				{
					PerCam_UV[camID*ntracks + trackID][kk].std2D = 0.0;
					PerCam_UV[camID*ntracks + trackID][kk].std3D = 0.0; //disregard this point
				}
			}
		}
	}

	return;
}
void MotionPrior_Optim_SpatialStructure_NoSimulatenousPoints(char *Path, vector<double*> &Allpt3D, vector<ImgPtEle> *PerCam_UV, vector<int> &PerPoint_nFrames, double *currentOffset, int ntracks, bool non_monotonicDescent, int nCams, int motionPriorPower, double Tscale, double ialpha, double eps, double lamda, double *Cost, bool StillImages = false, bool silent = true)
{
	double ActionCost = 0.0, ProjCost = 0.0, costiX, costiY, costi;

	int *currentFrame = new int[nCams], *PerCam_nf = new int[nCams], *currentPID_InTrack = new int[nCams];
	Point3d P3D;
	ImgPtEle ptEle;

	vector<int>triangulatedList;
	vector<double>AllError3D, VectorTime;
	vector<int> *VectorCamID = new vector<int>[ntracks], *VectorFrameID = new vector<int>[ntracks];
	vector<ImgPtEle> *Traj2DAll = new vector<ImgPtEle>[ntracks];


	ceres::Problem problem;

	double earliestTime, currentTime;
	int earliestCamFrameID, frameID, nfinishedCams, earliestCamID;

	int *StillImageTimeOrderID = 0;
	double *StillImageTimeOrder = 0;
	if (StillImages)
	{
		StillImageTimeOrderID = new int[nCams];
		StillImageTimeOrder = new double[nCams];
		for (int camID = 0; camID < nCams; camID++)
		{
			StillImageTimeOrder[camID] = currentOffset[camID];
			StillImageTimeOrderID[camID] = camID;
		}
		Quick_Sort_Double(StillImageTimeOrder, StillImageTimeOrderID, 0, nCams - 1);
	}

	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		if (StillImages)
		{
			VectorTime.clear();

			for (int ii = 0; ii < nCams; ii++)
			{
				int camID = StillImageTimeOrderID[ii];
				double currentTime = StillImageTimeOrder[ii];
				VectorTime.push_back(currentTime);
				VectorCamID[trackID].push_back(camID);
				VectorFrameID[trackID].push_back(0);
				Traj2DAll[trackID].push_back(PerCam_UV[camID*ntracks + trackID][0]);
			}
		}
		else
		{
			for (int camID = 0; camID < nCams; camID++)
				PerCam_nf[camID] = PerCam_UV[camID*ntracks + trackID].size();

			//Assemble trajactory and time from all Cameras
			VectorTime.clear();
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
					frameID = PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].frameID;
					currentTime = 1.0*currentOffset[camID] * ialpha*Tscale + 1.0*frameID * ialpha*Tscale;

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
				VectorCamID[trackID].push_back(earliestCamID);
				VectorFrameID[trackID].push_back(earliestCamFrameID);
				Traj2DAll[trackID].push_back(PerCam_UV[earliestCamID*ntracks + trackID][currentPID_InTrack[earliestCamID]]);

				currentPID_InTrack[earliestCamID]++;
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

			costi = LeastActionError(&Allpt3D[trackID][3 * ll], &Allpt3D[trackID][3 * ll + 3], &currentOffset[camID1], &currentOffset[camID2], VectorFrameID[trackID][ll], VectorFrameID[trackID][ll + 1], ialpha, Tscale, eps, motionPriorPower);
			ActionCost += (1.0 - lamda)*costi;
			Q1 = Traj2DAll[trackID][ll].Q, Q2 = Traj2DAll[trackID][ll + 1].Q, U1 = Traj2DAll[trackID][ll].u, U2 = Traj2DAll[trackID][ll + 1].u;

			costiX = sqrt(lamda)*(Q1[0] * Allpt3D[trackID][3 * ll] + Q1[1] * Allpt3D[trackID][3 * ll + 1] + Q1[2] * Allpt3D[trackID][3 * ll + 2] - U1[0]);
			costiY = sqrt(lamda)*(Q1[3] * Allpt3D[trackID][3 * ll] + Q1[4] * Allpt3D[trackID][3 * ll + 1] + Q1[5] * Allpt3D[trackID][3 * ll + 2] - U1[1]);
			costi = sqrt(costiX*costiX + costiY*costiY);
			ProjCost += costi;

			costiX = sqrt(lamda)*(Q2[0] * Allpt3D[trackID][3 * ll + 3] + Q2[1] * Allpt3D[trackID][3 * ll + 4] + Q2[2] * Allpt3D[trackID][3 * ll + 5] - U2[0]);
			costiY = sqrt(lamda)*(Q2[3] * Allpt3D[trackID][3 * ll + 3] + Q2[4] * Allpt3D[trackID][3 * ll + 4] + Q2[5] * Allpt3D[trackID][3 * ll + 5] - U2[1]);
			costi = sqrt(costiX*costiX + costiY*costiY);
			ProjCost += costi;

			double  t1 = (currentOffset[camID1] + VectorFrameID[trackID][ll]) * ialpha*Tscale;
			double  t2 = (currentOffset[camID2] + VectorFrameID[trackID][ll + 1]) * ialpha*Tscale;

			ceres::CostFunction* cost_function = LeastActionCost3DCeres::CreateAutoDiff(t1, t2, eps, sqrt(1.0 - lamda), motionPriorPower);
			//ceres::CostFunction* cost_function = LeastActionCostCeres::CreateNumerDiffSame(VectorFrameID[trackID][ll], VectorFrameID[trackID][ll + 1], ialpha, Tscale, rate, eps, motionPriorPower);
			problem.AddResidualBlock(cost_function, NULL, &Allpt3D[trackID][3 * ll], &Allpt3D[trackID][3 * (ll + 1)]);

			ceres::CostFunction* cost_function2 = IdealAlgebraicReprojectionCeres::Create(Traj2DAll[trackID][ll].Q, Traj2DAll[trackID][ll].u, sqrt(lamda));
			problem.AddResidualBlock(cost_function2, NULL, Allpt3D[trackID] + 3 * ll);

			ceres::CostFunction* cost_function3 = IdealAlgebraicReprojectionCeres::Create(Traj2DAll[trackID][ll + 1].Q, Traj2DAll[trackID][ll + 1].u, sqrt(lamda));
			problem.AddResidualBlock(cost_function3, NULL, Allpt3D[trackID] + 3 * ll + 3);
		}

		/*//Set fixed parameters
		ceres::Solver::Options options;
		options.num_threads = 4;
		options.max_num_iterations = 1000;
		options.linear_solver_type = ceres::SPARSE_SCHUR; //SPARSE_NORMAL_CHOLESKY;
		options.minimizer_progress_to_stdout = false;// silent ? false : true;
		options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
		options.use_nonmonotonic_steps = true;

		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);

		if (!silent)
		cout << "Point: " << trackID << " (" << npts << ") frames " << summary.BriefReport() << "\n";

		for (int ii = 0; ii < npts; ii++)
		{
		int camID = VectorCamID[trackID][ii], frameID = VectorFrameID[trackID][ii];

		bool found = false;
		for (int kk = 0; kk < PerCam_UV[camID*ntracks + trackID].size(); kk++)
		{
		if (frameID == PerCam_UV[camID*ntracks + trackID][kk].frameID)
		{
		//if (summary.final_cost < 1e7)
		PerCam_UV[camID*ntracks + trackID][kk].pt3D = Point3d(Allpt3D[trackID][3 * ii], Allpt3D[trackID][3 * ii + 1], Allpt3D[trackID][3 * ii + 2]);
		//else
		//	PerCam_UV[camID*ntracks + trackID][kk].pt3D = Point3d(0, 0, 0);
		found = true;
		break;
		}
		}
		if (!found)
		{
		printf("Serious bug in point-camera-frame association\n");
		abort();
		}
		}*/
	}
	for (int trackID = 0; trackID < ntracks; trackID++)
		PerPoint_nFrames.push_back(Traj2DAll[trackID].size());

	ceres::Solver::Options options;
	options.num_threads = 4;
	options.num_linear_solver_threads = 4;
	options.max_num_iterations = 500;
	options.linear_solver_type = ceres::SPARSE_SCHUR; //SPARSE_NORMAL_CHOLESKY;
	options.minimizer_progress_to_stdout = false;// silent ? false : true;
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	options.use_nonmonotonic_steps = true;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	if (!silent)
		std::cout << summary.FullReport() << "\n";
	else
		std::cout << summary.BriefReport() << "\n";

	//Save data
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		int npts = Traj2DAll[trackID].size();
		for (int ii = 0; ii < npts; ii++)
		{
			int camID = VectorCamID[trackID][ii], frameID = VectorFrameID[trackID][ii];

			bool found = false;
			for (int kk = 0; kk < PerCam_UV[camID*ntracks + trackID].size(); kk++)
			{
				if (frameID == PerCam_UV[camID*ntracks + trackID][kk].frameID)
				{
					PerCam_UV[camID*ntracks + trackID][kk].pt3D = Point3d(Allpt3D[trackID][3 * ii], Allpt3D[trackID][3 * ii + 1], Allpt3D[trackID][3 * ii + 2]);
					found = true;
					break;
				}
			}
			if (!found)
			{
				printf("Serious bug in point-camera-frame association\n");
				abort();
			}
		}
	}

	//Compute cost after optim
	ActionCost = 0.0, ProjCost = 0.0;
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		int npts = PerPoint_nFrames[trackID];
		double costi;
		for (int ll = 0; ll < npts - 1; ll++)
		{
			int camID1 = VectorCamID[trackID][ll], camID2 = VectorCamID[trackID][ll + 1];

			costi = LeastActionError(&Allpt3D[trackID][3 * ll], &Allpt3D[trackID][3 * ll + 3], &currentOffset[camID1], &currentOffset[camID2], VectorFrameID[trackID][ll], VectorFrameID[trackID][ll + 1], ialpha, Tscale, eps, motionPriorPower);
			ActionCost += costi;

			costiX = sqrt(lamda)*(Traj2DAll[trackID][ll].Q[0] * Allpt3D[trackID][3 * ll] + Traj2DAll[trackID][ll].Q[1] * Allpt3D[trackID][3 * ll + 1] + Traj2DAll[trackID][ll].Q[2] * Allpt3D[trackID][3 * ll + 2] - Traj2DAll[trackID][ll].u[0]);
			costiY = sqrt(lamda)*(Traj2DAll[trackID][ll].Q[3] * Allpt3D[trackID][3 * ll] + Traj2DAll[trackID][ll].Q[4] * Allpt3D[trackID][3 * ll + 1] + Traj2DAll[trackID][ll].Q[5] * Allpt3D[trackID][3 * ll + 2] - Traj2DAll[trackID][ll].u[1]);
			costi = sqrt(costiX*costiX + costiY*costiY);
			ProjCost += costi;

			costiX = sqrt(lamda)*(Traj2DAll[trackID][ll + 1].Q[0] * Allpt3D[trackID][3 * ll + 3] + Traj2DAll[trackID][ll + 1].Q[1] * Allpt3D[trackID][3 * ll + 4] + Traj2DAll[trackID][ll + 1].Q[2] * Allpt3D[trackID][3 * ll + 5] - Traj2DAll[trackID][ll + 1].u[0]);
			costiY = sqrt(lamda)*(Traj2DAll[trackID][ll + 1].Q[3] * Allpt3D[trackID][3 * ll + 3] + Traj2DAll[trackID][ll + 1].Q[4] * Allpt3D[trackID][3 * ll + 4] + Traj2DAll[trackID][ll + 1].Q[5] * Allpt3D[trackID][3 * ll + 5] - Traj2DAll[trackID][ll + 1].u[1]);
			costi = sqrt(costiX*costiX + costiY*costiY);
			ProjCost += costi;
		}
	}

	if (!silent)
		printf("Action cost: %f \nProjection cost: %f ", ActionCost, ProjCost);

	double lengthCost = 0.0;
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		int npts = Traj2DAll[trackID].size();
		double costi;
		for (int ll = 0; ll < npts - 1; ll++)
		{
			costi = sqrt(pow(Allpt3D[trackID][3 * ll] - Allpt3D[trackID][3 * ll + 3], 2) + pow(Allpt3D[trackID][3 * ll + 1] - Allpt3D[trackID][3 * ll + 4], 2) + pow(Allpt3D[trackID][3 * ll + 2] - Allpt3D[trackID][3 * ll + 5], 2));
			lengthCost += costi;
		}
	}
	if (!silent)
		printf("Distance Cost: %e\n", lengthCost);

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
	if (!silent)
		printf("Direction Cost: %e\n", directionCost);

	Cost[0] = ActionCost, Cost[1] = ProjCost, Cost[2] = lengthCost, Cost[3] = directionCost;

	delete[]VectorCamID, delete[]VectorFrameID, delete[]Traj2DAll;
	delete[]currentFrame, delete[]PerCam_nf;

	return;
}
double MotionPrior_Optim_SpatialStructure_Algebraic(char *Path, vector<double*> &Allpt3D, vector<ImgPtEle> *PerCam_UV, vector<int> &PerPoint_nFrames, double *currentOffset, int ntracks, bool non_monotonicDescent, int nCams, int motionPriorPower, double Tscale, double ialpha, double eps, double lamda, double *Cost, bool StillImages = false, bool silent = true)
{
	vector<double> *VectorTime = new vector<double>[ntracks];
	vector<int> *VectorCamID = new vector<int>[ntracks], *VectorFrameID = new vector<int>[ntracks], *simulatneousPoints = new vector<int>[ntracks];
	vector<ImgPtEle> *Traj2DAll = new vector<ImgPtEle>[ntracks];

	int *StillImageTimeOrderID = 0;
	double *StillImageTimeOrder = 0;
	if (StillImages)
	{
		StillImageTimeOrderID = new int[nCams];
		StillImageTimeOrder = new double[nCams];
		for (int camID = 0; camID < nCams; camID++)
		{
			StillImageTimeOrder[camID] = currentOffset[camID];
			StillImageTimeOrderID[camID] = camID;
		}
		Quick_Sort_Double(StillImageTimeOrder, StillImageTimeOrderID, 0, nCams - 1);
	}

	double CeresCost = 0.0;
	omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		bool UsedCam[MaxnCams];
		int PerCam_nf[MaxnCams], currentPID_InTrack[MaxnCams];
		Point3d P3D;
		ImgPtEle ptEle;
		double earliestTime, currentTime, RollingShutterOffset;
		int earliestCamFrameID, frameID, nfinishedCams, earliestCamID;

		ceres::Problem problem;
		if (StillImages)
		{
			for (int ii = 0; ii < nCams; ii++)
			{
				int camID = StillImageTimeOrderID[ii];
				double currentTime = StillImageTimeOrder[ii];
				VectorTime[trackID].push_back(currentTime);
				VectorCamID[trackID].push_back(camID);
				VectorFrameID[trackID].push_back(0);
				Traj2DAll[trackID].push_back(PerCam_UV[camID*ntracks + trackID][0]);
			}
		}
		else
		{
			for (int camID = 0; camID < nCams; camID++)
				PerCam_nf[camID] = PerCam_UV[camID*ntracks + trackID].size();

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
					frameID = PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].frameID;
					RollingShutterOffset = 0;// PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].pt2D.y / PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].imHeight;
					currentTime = (currentOffset[camID] + frameID + RollingShutterOffset) * ialpha*Tscale;

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
				VectorTime[trackID].push_back(earliestTime);
				VectorCamID[trackID].push_back(earliestCamID);
				VectorFrameID[trackID].push_back(earliestCamFrameID);
				Traj2DAll[trackID].push_back(PerCam_UV[earliestCamID*ntracks + trackID][currentPID_InTrack[earliestCamID]]);

				currentPID_InTrack[earliestCamID]++;
			}
		}

		int npts = Traj2DAll[trackID].size();
		Allpt3D[trackID] = new double[3 * npts];
		for (int ll = 0; ll < npts; ll++)
			Allpt3D[trackID][3 * ll] = Traj2DAll[trackID][ll].pt3D.x, Allpt3D[trackID][3 * ll + 1] = Traj2DAll[trackID][ll].pt3D.y, Allpt3D[trackID][3 * ll + 2] = Traj2DAll[trackID][ll].pt3D.z;

		//Detect points captured simulatenously
		int groupCount = 0;
		for (int ll = 0; ll < npts; ll++)
			simulatneousPoints[trackID].push_back(-1);

		for (int ll = 0; ll < npts - 1; ll++)
		{
			int naddedPoints = 0; bool found = false;
			for (int kk = ll + 1; kk < npts; kk++)
			{
				naddedPoints++;
				if (VectorTime[trackID][kk] - VectorTime[trackID][ll] > FLT_EPSILON)
					break;
				else
				{
					if (kk - 1 == ll)
						simulatneousPoints[trackID][ll] = groupCount;

					simulatneousPoints[trackID][kk] = groupCount;
					found = true;
				}
			}

			if (found)
			{
				ll += naddedPoints - 1;
				groupCount++;
			}
		}

		double ActionCost = 0, ProjCost = 0;

		//1st order approx of v
		int oldtype = simulatneousPoints[trackID][0];
		for (int ll = 0; ll < nCams; ll++)
			UsedCam[ll] = false;

		for (int ll = 0; ll < npts - 1; ll++)
		{
			int incre = 1;
			while (ll + incre < npts)
			{
				if (simulatneousPoints[trackID][ll + incre] == -1 || simulatneousPoints[trackID][ll + incre] != oldtype)
				{
					oldtype = simulatneousPoints[trackID][ll + incre];
					break;
				}
				else
				{
					ceres::CostFunction* cost_function3 = IdealAlgebraicReprojectionCeres::Create(Traj2DAll[trackID][ll + incre].Q, Traj2DAll[trackID][ll + incre].u, sqrt(lamda));
					problem.AddResidualBlock(cost_function3, NULL, Allpt3D[trackID] + 3 * ll);
				}
				incre++;
			}
			if (ll + incre == npts)
				break;


			int camID1 = VectorCamID[trackID][ll], camID2 = VectorCamID[trackID][ll + incre];

			double shutterOffset1 = 0,//Traj2DAll[trackID][ll].pt2D.y / Traj2DAll[trackID][ll].imHeight,
				shutterOffset2 = 0;// Traj2DAll[trackID][ll + incre].pt2D.y / Traj2DAll[trackID][ll + incre].imHeight;
			double  t1 = (currentOffset[camID1] + VectorFrameID[trackID][ll] + shutterOffset1) * ialpha*Tscale;
			double  t2 = (currentOffset[camID2] + VectorFrameID[trackID][ll + incre] + shutterOffset2) * ialpha*Tscale;

			ceres::CostFunction* cost_function = LeastActionCost3DCeres::CreateAutoDiff(t1, t2, eps, sqrt(1.0 - lamda), motionPriorPower);
			//ceres::CostFunction* cost_function = LeastActionCostCeres::CreateNumerDiffSame(t1, t2, eps);
			problem.AddResidualBlock(cost_function, NULL, &Allpt3D[trackID][3 * ll], &Allpt3D[trackID][3 * (ll + incre)]);

			ceres::CostFunction* cost_function2 = IdealAlgebraicReprojectionCeres::Create(Traj2DAll[trackID][ll].Q, Traj2DAll[trackID][ll].u, sqrt(lamda));
			problem.AddResidualBlock(cost_function2, NULL, &Allpt3D[trackID][3 * ll]);

			ceres::CostFunction* cost_function3 = IdealAlgebraicReprojectionCeres::Create(Traj2DAll[trackID][ll + incre].Q, Traj2DAll[trackID][ll + incre].u, sqrt(lamda));
			problem.AddResidualBlock(cost_function3, NULL, &Allpt3D[trackID][3 * (ll + incre)]);

			ll += incre - 1;
		}

		ceres::Solver::Options options;
		options.num_threads = 2;
		options.num_linear_solver_threads = 2;
		options.max_num_iterations = 500;
		options.linear_solver_type = ceres::SPARSE_SCHUR;
		options.minimizer_progress_to_stdout = false;
		options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
		options.use_nonmonotonic_steps = true;

		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);

#pragma omp critical
		{
			CeresCost += summary.final_cost;
			if (!silent)
				cout << "Point: " << trackID << " " << summary.BriefReport() << "\n";// cout << "Point: " << trackID << " " << summary.FullReport() << "\n";
		}

		//copy simultaneous triggered points
		for (int ll = 0; ll < npts - 1; ll++)
		{
			if (simulatneousPoints[trackID][ll] != -1)
			{
				int nPoint = 0;
				for (int kk = ll + 1; kk < npts; kk++)
				{
					nPoint++;
					if (simulatneousPoints[trackID][kk] != simulatneousPoints[trackID][ll])
						break;
					else
					{
						Allpt3D[trackID][3 * kk] = Allpt3D[trackID][3 * ll];
						Allpt3D[trackID][3 * kk + 1] = Allpt3D[trackID][3 * ll + 1];
						Allpt3D[trackID][3 * kk + 2] = Allpt3D[trackID][3 * ll + 2];
					}
				}
				ll += nPoint - 1;
			}
		}

		for (int ii = 0; ii < Traj2DAll[trackID].size(); ii++)
		{
			int camID = VectorCamID[trackID][ii], frameID = VectorFrameID[trackID][ii];

			bool found = false;
			for (int kk = 0; kk < PerCam_UV[camID*ntracks + trackID].size(); kk++)
			{
				if (frameID == PerCam_UV[camID*ntracks + trackID][kk].frameID)
				{
					PerCam_UV[camID*ntracks + trackID][kk].pt3D = Point3d(Allpt3D[trackID][3 * ii], Allpt3D[trackID][3 * ii + 1], Allpt3D[trackID][3 * ii + 2]);
					found = true;
					break;
				}
			}
			if (!found)
			{
				printf("Serious bug in point-camera-frame association\n");
				abort();
			}
		}
	}
	//printf("Alge: END\n");
	for (int trackID = 0; trackID < ntracks; trackID++)
		PerPoint_nFrames.push_back(Traj2DAll[trackID].size());

	//Compute cost after optim
	double ActionCost = 0.0, ProjCost = 0.0, lengthCost = 0.0, directionCost = 0;
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		int npts = PerPoint_nFrames[trackID], oldtype = simulatneousPoints[trackID][0];
		for (int ll = 0; ll < npts - 1; ll++)
		{
			int incre = 1;
			while (ll + incre < npts)
			{
				if (simulatneousPoints[trackID][ll + incre] == -1 || simulatneousPoints[trackID][ll + incre] != oldtype)
				{
					oldtype = simulatneousPoints[trackID][ll + incre];
					break;
				}
				incre++;
			}
			if (ll + incre == npts)
				break;

			int camID1 = VectorCamID[trackID][ll], camID2 = VectorCamID[trackID][ll + incre];
			double costi = LeastActionError(&Allpt3D[trackID][3 * ll], &Allpt3D[trackID][3 * (ll + incre)], &currentOffset[camID1], &currentOffset[camID2], VectorFrameID[trackID][ll], VectorFrameID[trackID][ll + incre], ialpha, Tscale, eps, motionPriorPower);
			ActionCost += costi;

			ll += incre - 1;
		}
	}
	if (!silent)
		printf("Action cost: %f ", ActionCost);

	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		int npts = PerPoint_nFrames[trackID];
		double costi_1, costi_2;
		for (int ll = 0; ll < npts - 1; ll++)
		{
			Point2d P2d_1(Traj2DAll[trackID][ll].pt2D.x, Traj2DAll[trackID][ll].pt2D.y);
			Point3d P3d_1(Allpt3D[trackID][3 * ll], Allpt3D[trackID][3 * ll + 1], Allpt3D[trackID][3 * ll + 2]);
			costi_1 = PinholeReprojectionErrorSimpleDebug(Traj2DAll[trackID][ll].P, P3d_1, P2d_1);
			ProjCost += costi_1;

			Point2d P2d_2(Traj2DAll[trackID][ll].pt2D.x, Traj2DAll[trackID][ll].pt2D.y);
			Point3d P3d_2(Allpt3D[trackID][3 * ll], Allpt3D[trackID][3 * ll + 1], Allpt3D[trackID][3 * ll + 2]);
			costi_2 = PinholeReprojectionErrorSimpleDebug(Traj2DAll[trackID][ll].P, P3d_2, P2d_2);
			ProjCost += costi_2;
		}
	}
	if (!silent)
		printf("Projection cost: %f ", ProjCost);

	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		int npts = Traj2DAll[trackID].size();
		double costi;
		for (int ll = 0; ll < npts - 1; ll++)
		{
			costi = sqrt(pow(Allpt3D[trackID][3 * ll] - Allpt3D[trackID][3 * ll + 3], 2) + pow(Allpt3D[trackID][3 * ll + 1] - Allpt3D[trackID][3 * ll + 4], 2) + pow(Allpt3D[trackID][3 * ll + 2] - Allpt3D[trackID][3 * ll + 5], 2));
			lengthCost += costi;
		}
	}
	if (!silent)
		printf("Distance Cost: %e.", lengthCost);

	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		int npts = PerPoint_nFrames[trackID], oldtype = simulatneousPoints[trackID][0];
		vector<int>Knot;
		Knot.push_back(0);
		for (int ll = 0; ll < npts; ll++)
		{
			int incre = 1;
			while (ll + incre < npts)
			{
				if (simulatneousPoints[trackID][ll + incre] == -1 || simulatneousPoints[trackID][ll + incre] != oldtype)
				{
					oldtype = simulatneousPoints[trackID][ll + incre];
					break;
				}
				incre++;
			}
			if (ll + incre == npts)
				break;
			Knot.push_back(ll + incre);
			ll += incre - 1;
		}

		for (int ll = 0; ll < Knot.size() - 2; ll++)
		{
			int fid1 = Knot[ll], fid2 = Knot[ll + 1], fid3 = Knot[ll + 2];
			double direct1[] = { Allpt3D[trackID][3 * fid1] - Allpt3D[trackID][3 * fid2], Allpt3D[trackID][3 * fid1 + 1] - Allpt3D[trackID][3 * fid2 + 1], Allpt3D[trackID][3 * fid1 + 2] - Allpt3D[trackID][3 * fid2 + 2] };
			double direct2[] = { Allpt3D[trackID][3 * fid2] - Allpt3D[trackID][3 * fid3], Allpt3D[trackID][3 * fid2 + 1] - Allpt3D[trackID][3 * fid3 + 1], Allpt3D[trackID][3 * fid2 + 2] - Allpt3D[trackID][3 * fid3 + 2] };
			normalize(direct1), normalize(direct2);
			directionCost += abs(dotProduct(direct1, direct2));
		}
	}
	if (!silent)
		printf("Direction Cost: %e\n", directionCost);

	Cost[0] = ActionCost, Cost[1] = ProjCost, Cost[2] = lengthCost, Cost[3] = directionCost;

	delete[]VectorTime, delete[]VectorCamID, delete[]VectorFrameID, delete[]simulatneousPoints, delete[]Traj2DAll;

	return CeresCost;
}
double MotionPrior_Optim_SpatialStructure_Geometric(char *Path, vector<double*> &Allpt3D, vector<ImgPtEle> *PerCam_UV, vector<int> &PerPoint_nFrames, double *currentOffset, int npts, bool non_monotonicDescent, int nCams, int motionPriorPower, double Tscale, double ialpha, double eps, double lamda, double *Cost, bool StillImages = false, bool silent = true)
{
	vector<double> *VectorTime = new vector<double>[npts];
	vector<int> *VectorCamID = new vector<int>[npts], *VectorFrameID = new vector<int>[npts], *simulatneousPoints = new vector<int>[npts];
	vector<ImgPtEle> *Traj2DAll = new vector<ImgPtEle>[npts];

	int *StillImageTimeOrderID = 0;
	double *StillImageTimeOrder = 0;
	if (StillImages)
	{
		StillImageTimeOrderID = new int[nCams];
		StillImageTimeOrder = new double[nCams];
		for (int camID = 0; camID < nCams; camID++)
		{
			StillImageTimeOrder[camID] = currentOffset[camID];
			StillImageTimeOrderID[camID] = camID;
		}
		Quick_Sort_Double(StillImageTimeOrder, StillImageTimeOrderID, 0, nCams - 1);
	}

	double CeresCost = 0.0;

	omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for
	for (int pid = 0; pid < npts; pid++)
	{
		//printf("%d ... ", pid);
		bool UsedCam[MaxnCams];
		int PerCam_nf[MaxnCams], currentFID_InPoint[MaxnCams];
		Point3d P3D;
		ImgPtEle ptEle;

		double earliestTime, currentTime, RollingShutterOffset;
		int earliestCamFrameID, frameID, nfinishedCams, earliestCamID;

		ceres::Problem problem;
		if (StillImages)
		{
			for (int ii = 0; ii < nCams; ii++)
			{
				int camID = StillImageTimeOrderID[ii];
				double currentTime = StillImageTimeOrder[ii];
				VectorTime[pid].push_back(currentTime);
				VectorCamID[pid].push_back(camID);
				VectorFrameID[pid].push_back(0);
				Traj2DAll[pid].push_back(PerCam_UV[camID*npts + pid][0]);
			}
		}
		else
		{
			for (int camID = 0; camID < nCams; camID++)
				PerCam_nf[camID] = PerCam_UV[camID*npts + pid].size();

			//Assemble trajactory and time from all Cameras
			for (int jj = 0; jj < nCams; jj++)
				currentFID_InPoint[jj] = 0;

			while (true)
			{
				//Determine the next camera
				nfinishedCams = 0, earliestCamID, earliestTime = 9e9;
				for (int camID = 0; camID < nCams; camID++)
				{
					if (currentFID_InPoint[camID] == PerCam_nf[camID])
					{
						nfinishedCams++;
						continue;
					}

					//Time:
					frameID = PerCam_UV[camID*npts + pid][currentFID_InPoint[camID]].frameID;
					RollingShutterOffset = 0;// PerCam_UV[camID*npts + pid][currentFID_InPoint[camID]].pt2D.y / PerCam_UV[camID*npts + pid][currentFID_InPoint[camID]].imHeight;
					currentTime = (currentOffset[camID] + frameID + RollingShutterOffset) * ialpha*Tscale;

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
				VectorTime[pid].push_back(earliestTime);
				VectorCamID[pid].push_back(earliestCamID);
				VectorFrameID[pid].push_back(earliestCamFrameID);
				Traj2DAll[pid].push_back(PerCam_UV[earliestCamID*npts + pid][currentFID_InPoint[earliestCamID]]);
				currentFID_InPoint[earliestCamID]++;
			}
		}

		int nf = Traj2DAll[pid].size();
		Allpt3D[pid] = new double[3 * nf];
		for (int ll = 0; ll < nf; ll++)
			Allpt3D[pid][3 * ll] = Traj2DAll[pid][ll].pt3D.x, Allpt3D[pid][3 * ll + 1] = Traj2DAll[pid][ll].pt3D.y, Allpt3D[pid][3 * ll + 2] = Traj2DAll[pid][ll].pt3D.z;

		//Detect points captured simulatenously
		int groupCount = 0;
		for (int ll = 0; ll < nf; ll++)
			simulatneousPoints[pid].push_back(-1);

		for (int ll = 0; ll < nf - 1; ll++)
		{
			int naddedPoints = 0; bool found = false;
			for (int kk = ll + 1; kk < nf; kk++)
			{
				naddedPoints++;
				if (VectorTime[pid][kk] - VectorTime[pid][ll] > FLT_EPSILON)
					break;
				else
				{
					if (kk - 1 == ll)
						simulatneousPoints[pid][ll] = groupCount;

					simulatneousPoints[pid][kk] = groupCount;
					found = true;
				}
			}

			if (found)
			{
				ll += naddedPoints - 1;
				groupCount++;
			}
		}

		//1st order approx of v
		double ActionCost = 0, ProjCost = 0;
		int oldtype = simulatneousPoints[pid][0];
		for (int ll = 0; ll < nCams; ll++)
			UsedCam[ll] = false;

		for (int ll = 0; ll < nf - 1; ll++)
		{
			int incre = 1;
			while (ll + incre < nf)
			{
				if (simulatneousPoints[pid][ll + incre] == -1 || simulatneousPoints[pid][ll + incre] != oldtype)
				{
					oldtype = simulatneousPoints[pid][ll + incre];
					break;
				}
				else
				{
					if (Traj2DAll[pid][ll + incre].std3D < 0.0)
					{
						ceres::CostFunction* cost_function = IdealGeoProjectionCeres::Create(Traj2DAll[pid][ll + incre].P, Traj2DAll[pid][ll + incre].pt2D, 1.0 / sqrt(lamda));
						problem.AddResidualBlock(cost_function, NULL, Allpt3D[pid] + 3 * ll);
					}
					else
					{
						ceres::CostFunction* cost_function = IdealGeoProjectionCeres::Create(Traj2DAll[pid][ll + incre].P, Traj2DAll[pid][ll + incre].pt2D, Traj2DAll[pid][ll + incre].std2D);
						problem.AddResidualBlock(cost_function, NULL, Allpt3D[pid] + 3 * ll);
					}
				}
				incre++;
			}
			if (ll + incre == nf)
				break;

			int camID1 = VectorCamID[pid][ll], camID2 = VectorCamID[pid][ll + incre];
			double costi = LeastActionError(&Allpt3D[pid][3 * ll], &Allpt3D[pid][3 * ll + 3], &currentOffset[camID1], &currentOffset[camID2], VectorFrameID[pid][ll], VectorFrameID[pid][ll + incre], ialpha, Tscale, eps, motionPriorPower);
			ActionCost += costi;

			Point2d P2d_1(Traj2DAll[pid][ll].pt2D.x, Traj2DAll[pid][ll].pt2D.y);
			Point3d P3d_1(Allpt3D[pid][3 * ll], Allpt3D[pid][3 * ll + 1], Allpt3D[pid][3 * ll + 2]);
			costi = PinholeReprojectionErrorSimpleDebug(Traj2DAll[pid][ll].P, P3d_1, P2d_1);
			ProjCost += costi;

			Point2d P2d_2(Traj2DAll[pid][(ll + incre)].pt2D.x, Traj2DAll[pid][(ll + incre)].pt2D.y);
			Point3d P3d_2(Allpt3D[pid][3 * (ll + incre)], Allpt3D[pid][3 * (ll + incre) + 1], Allpt3D[pid][3 * (ll + incre) + 2]);
			costi = PinholeReprojectionErrorSimpleDebug(Traj2DAll[pid][(ll + incre)].P, P3d_2, P2d_2);
			ProjCost += costi;

			double shutterOffset1 = 0,//Traj2DAll[pid][ll].pt2D.y / Traj2DAll[pid][ll].imHeight,
				shutterOffset2 = 0;// Traj2DAll[pid][ll + incre].pt2D.y / Traj2DAll[pid][ll + incre].imHeight;
			double  t1 = (currentOffset[camID1] + VectorFrameID[pid][ll] + shutterOffset1) * ialpha*Tscale;
			double  t2 = (currentOffset[camID2] + VectorFrameID[pid][ll + incre] + shutterOffset2) * ialpha*Tscale;

			if (Traj2DAll[pid][ll + incre].std3D < 0.0)
			{
				ceres::CostFunction* cost_function = LeastActionCost3DCeres::CreateAutoDiff(t1, t2, eps, 1.0 / sqrt(1.0 - lamda), motionPriorPower);
				//ceres::CostFunction* cost_function = LeastActionCostCeres::CreateNumerDiffSame(VectorFrameID[pid][ll], VectorFrameID[pid][ll + incre], ialpha, Tscale, eps, motionPriorPower);
				problem.AddResidualBlock(cost_function, NULL, &Allpt3D[pid][3 * ll], &Allpt3D[pid][3 * (ll + incre)]);

				ceres::CostFunction* cost_function2 = IdealGeoProjectionCeres::Create(Traj2DAll[pid][ll].P, Traj2DAll[pid][ll].pt2D, 1.0 / sqrt(lamda));
				problem.AddResidualBlock(cost_function2, NULL, Allpt3D[pid] + 3 * ll);

				ceres::CostFunction* cost_function3 = IdealGeoProjectionCeres::Create(Traj2DAll[pid][ll + incre].P, Traj2DAll[pid][ll + incre].pt2D, 1.0 / sqrt(lamda));
				problem.AddResidualBlock(cost_function3, NULL, Allpt3D[pid] + 3 * (ll + incre));
			}
			else
			{
				ceres::CostFunction* cost_function = LeastActionCost3DCeres::CreateAutoDiff(t1, t2, eps, Traj2DAll[pid][ll].std3D, motionPriorPower);
				problem.AddResidualBlock(cost_function, NULL, &Allpt3D[pid][3 * ll], &Allpt3D[pid][3 * (ll + incre)]);

				ceres::CostFunction* cost_function2 = IdealGeoProjectionCeres::Create(Traj2DAll[pid][ll].P, Traj2DAll[pid][ll].pt2D, Traj2DAll[pid][ll].std2D);
				problem.AddResidualBlock(cost_function2, NULL, Allpt3D[pid] + 3 * ll);

				ceres::CostFunction* cost_function3 = IdealGeoProjectionCeres::Create(Traj2DAll[pid][ll + incre].P, Traj2DAll[pid][ll + incre].pt2D, Traj2DAll[pid][(ll + incre)].std2D);
				problem.AddResidualBlock(cost_function3, NULL, Allpt3D[pid] + 3 * (ll + incre));
			}

			ll += incre - 1;
		}

		ceres::Solver::Options options;
		options.num_threads = 2;
		options.num_linear_solver_threads = 2;
		options.max_num_iterations = 500;
		options.linear_solver_type = ceres::SPARSE_SCHUR;
		options.minimizer_progress_to_stdout = false;
		options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
		options.use_nonmonotonic_steps = true;

		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);

#pragma omp critical
		{
			CeresCost += summary.final_cost;
			if (!silent)
				cout << "Point: " << pid << " " << summary.BriefReport() << "\n";// cout << "Point: " << pid << " " << summary.FullReport() << "\n";
		}

		//copy simultaneous triggered points
		for (int ll = 0; ll < nf - 1; ll++)
		{
			if (simulatneousPoints[pid][ll] != -1)
			{
				int nPoint = 0;
				for (int kk = ll + 1; kk < nf; kk++)
				{
					nPoint++;
					if (simulatneousPoints[pid][kk] != simulatneousPoints[pid][ll])
						break;
					else
					{
						Allpt3D[pid][3 * kk] = Allpt3D[pid][3 * ll];
						Allpt3D[pid][3 * kk + 1] = Allpt3D[pid][3 * ll + 1];
						Allpt3D[pid][3 * kk + 2] = Allpt3D[pid][3 * ll + 2];
					}
				}
				ll += nPoint - 1;
			}
		}

		for (int ii = 0; ii < Traj2DAll[pid].size(); ii++)
		{
			int camID = VectorCamID[pid][ii], frameID = VectorFrameID[pid][ii];

			bool found = false;
			for (int kk = 0; kk < PerCam_UV[camID*npts + pid].size(); kk++)
			{
				if (frameID == PerCam_UV[camID*npts + pid][kk].frameID)
				{
					PerCam_UV[camID*npts + pid][kk].pt3D = Point3d(Allpt3D[pid][3 * ii], Allpt3D[pid][3 * ii + 1], Allpt3D[pid][3 * ii + 2]);
					found = true;
					break;
				}
			}
			if (!found)
			{
				printf("Serious bug in point-camera-frame association\n");
				abort();
			}
		}
	}
	for (int pid = 0; pid < npts; pid++)
		PerPoint_nFrames.push_back(Traj2DAll[pid].size());

	//Compute cost after optim
	double ActionCost = 0.0, ProjCost = 0.0, lengthCost = 0.0, directionCost = 0;
	for (int pid = 0; pid < npts; pid++)
	{
		int nf = PerPoint_nFrames[pid], oldtype = simulatneousPoints[pid][0];
		for (int ll = 0; ll < nf - 1; ll++)
		{
			int incre = 1;
			while (ll + incre < nf)
			{
				if (simulatneousPoints[pid][ll + incre] == -1 || simulatneousPoints[pid][ll + incre] != oldtype)
				{
					oldtype = simulatneousPoints[pid][ll + incre];
					break;
				}
				incre++;
			}
			if (ll + incre == nf)
				break;

			int camID1 = VectorCamID[pid][ll], camID2 = VectorCamID[pid][ll + incre];
			double costi = LeastActionError(&Allpt3D[pid][3 * ll], &Allpt3D[pid][3 * (ll + incre)], &currentOffset[camID1], &currentOffset[camID2], VectorFrameID[pid][ll], VectorFrameID[pid][ll + incre], ialpha, Tscale, eps, motionPriorPower);
			ActionCost += costi;

			ll += incre - 1;
		}
	}
	if (!silent)
		printf("Action cost: %f ", ActionCost);

	for (int pid = 0; pid < npts; pid++)
	{
		int nf = PerPoint_nFrames[pid];
		double costi;
		for (int ll = 0; ll < nf - 1; ll++)
		{
			int viewID = Traj2DAll[pid][ll].viewID, frameID = Traj2DAll[pid][ll].frameID;
			Point2d P2d(Traj2DAll[pid][ll].pt2D.x, Traj2DAll[pid][ll].pt2D.y);
			Point3d P3d(Allpt3D[pid][3 * ll], Allpt3D[pid][3 * ll + 1], Allpt3D[pid][3 * ll + 2]);
			costi = PinholeReprojectionErrorSimpleDebug(Traj2DAll[pid][ll].P, P3d, P2d);
			ProjCost += costi;
		}
	}
	if (!silent)
		printf("Projection cost: %f ", ProjCost);

	for (int pid = 0; pid < npts; pid++)
	{
		int nf = Traj2DAll[pid].size();
		double costi;
		for (int ll = 0; ll < nf - 1; ll++)
		{
			costi = sqrt(pow(Allpt3D[pid][3 * ll] - Allpt3D[pid][3 * ll + 3], 2) + pow(Allpt3D[pid][3 * ll + 1] - Allpt3D[pid][3 * ll + 4], 2) + pow(Allpt3D[pid][3 * ll + 2] - Allpt3D[pid][3 * ll + 5], 2));
			lengthCost += costi;
		}
	}
	if (!silent)
		printf("Distance Cost: %e.", lengthCost);

	for (int pid = 0; pid < npts; pid++)
	{
		int nf = PerPoint_nFrames[pid], oldtype = simulatneousPoints[pid][0];
		vector<int>Knot;
		Knot.push_back(0);
		for (int ll = 0; ll < nf; ll++)
		{
			int incre = 1;
			while (ll + incre < nf)
			{
				if (simulatneousPoints[pid][ll + incre] == -1 || simulatneousPoints[pid][ll + incre] != oldtype)
				{
					oldtype = simulatneousPoints[pid][ll + incre];
					break;
				}
				incre++;
			}
			if (ll + incre == nf)
				break;
			Knot.push_back(ll + incre);
			ll += incre - 1;
		}

		for (int ll = 0; ll < Knot.size() - 2; ll++)
		{
			int fid1 = Knot[ll], fid2 = Knot[ll + 1], fid3 = Knot[ll + 2];
			double direct1[] = { Allpt3D[pid][3 * fid1] - Allpt3D[pid][3 * fid2], Allpt3D[pid][3 * fid1 + 1] - Allpt3D[pid][3 * fid2 + 1], Allpt3D[pid][3 * fid1 + 2] - Allpt3D[pid][3 * fid2 + 2] };
			double direct2[] = { Allpt3D[pid][3 * fid2] - Allpt3D[pid][3 * fid3], Allpt3D[pid][3 * fid2 + 1] - Allpt3D[pid][3 * fid3 + 1], Allpt3D[pid][3 * fid2 + 2] - Allpt3D[pid][3 * fid3 + 2] };
			normalize(direct1), normalize(direct2);
			directionCost += abs(dotProduct(direct1, direct2));
		}
	}
	if (!silent)
		printf("Direction Cost: %e\n", directionCost);

	Cost[0] = ActionCost, Cost[1] = ProjCost, Cost[2] = lengthCost, Cost[3] = directionCost;

	delete[]VectorTime, delete[]VectorCamID, delete[]VectorFrameID, delete[]simulatneousPoints, delete[]Traj2DAll;

	return CeresCost;
}
double MotionPrior_Optim_ST_Geometric(char *Path, vector<double*> &Allpt3D, vector<ImgPtEle> *PerCam_UV, vector<int> &PerPoint_nFrames, double *currentOffset, int ntracks, bool non_monotonicDescent, int nCams, int motionPriorPower, double Tscale, double ialpha, double eps, double lamda, double *Cost, bool StillImages = false, bool silent = true)
{
	double ActionCost = 0.0, ProjCost = 0.0, costi;

	int *currentFrame = new int[nCams], *PerCam_nf = new int[nCams], *currentPID_InTrack = new int[nCams];
	Point3d P3D;
	ImgPtEle ptEle;

	vector<double> VectorTime;
	vector<int> *VectorCamID = new vector<int>[ntracks], *VectorFrameID = new vector<int>[ntracks];
	vector<ImgPtEle> *Traj2DAll = new vector<ImgPtEle>[ntracks];

	double earliestTime, currentTime, RollingShutterOffset;
	int earliestCamFrameID, frameID, nfinishedCams, earliestCamID;

	int *StillImageTimeOrderID = 0;
	double *StillImageTimeOrder = 0;
	if (StillImages)
	{
		StillImageTimeOrderID = new int[nCams];
		StillImageTimeOrder = new double[nCams];
		for (int camID = 0; camID < nCams; camID++)
		{
			StillImageTimeOrder[camID] = currentOffset[camID];
			StillImageTimeOrderID[camID] = camID;
		}
		Quick_Sort_Double(StillImageTimeOrder, StillImageTimeOrderID, 0, nCams - 1);
	}

	ceres::Problem problem;
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		if (StillImages)
		{
			VectorTime.clear();

			for (int ii = 0; ii < nCams; ii++)
			{
				int camID = StillImageTimeOrderID[ii];
				double currentTime = StillImageTimeOrder[ii];
				VectorTime.push_back(currentTime);
				VectorCamID[trackID].push_back(camID);
				VectorFrameID[trackID].push_back(0);
				Traj2DAll[trackID].push_back(PerCam_UV[camID*ntracks + trackID][0]);
			}
		}
		else
		{
			for (int camID = 0; camID < nCams; camID++)
				PerCam_nf[camID] = PerCam_UV[camID*ntracks + trackID].size();

			//Assemble trajactory and time from all Cameras
			VectorTime.clear();
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
					frameID = PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].frameID;
					RollingShutterOffset = 0;// PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].pt2D.y / PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].imHeight;
					currentTime = (currentOffset[camID] + frameID + RollingShutterOffset) * ialpha*Tscale;

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
				VectorCamID[trackID].push_back(earliestCamID);
				VectorFrameID[trackID].push_back(earliestCamFrameID);
				Traj2DAll[trackID].push_back(PerCam_UV[earliestCamID*ntracks + trackID][currentPID_InTrack[earliestCamID]]);

				currentPID_InTrack[earliestCamID]++;
			}
		}

		int npts = Traj2DAll[trackID].size();
		Allpt3D[trackID] = new double[3 * npts];
		for (int ll = 0; ll < npts; ll++)
			Allpt3D[trackID][3 * ll] = Traj2DAll[trackID][ll].pt3D.x, Allpt3D[trackID][3 * ll + 1] = Traj2DAll[trackID][ll].pt3D.y, Allpt3D[trackID][3 * ll + 2] = Traj2DAll[trackID][ll].pt3D.z;

		for (int ll = 0; ll < npts - 1; ll++)//1st order approx of v
		{
			int camID1 = VectorCamID[trackID][ll], camID2 = VectorCamID[trackID][ll + 1];
			costi = LeastActionError(&Allpt3D[trackID][3 * ll], &Allpt3D[trackID][3 * ll + 3], &currentOffset[camID1], &currentOffset[camID2], VectorFrameID[trackID][ll], VectorFrameID[trackID][ll + 1], ialpha, Tscale, eps, motionPriorPower);
			ActionCost += costi;

			if (camID1 == camID2)
			{
				if (Traj2DAll[trackID][ll].std3D < 0.0)
				{
					ceres::CostFunction* cost_function = LeastActionCostCeres::CreateAutoDiffSame(VectorFrameID[trackID][ll], VectorFrameID[trackID][ll + 1], ialpha, Tscale, eps, 1.0 / sqrt(1.0 - lamda), motionPriorPower);
					//ceres::CostFunction* cost_function = LeastActionCostCeres::CreateNumerDiffSame(VectorFrameID[trackID][ll], VectorFrameID[trackID][ll + 1], ialpha, Tscale, rate, eps, motionPriorPower);
					problem.AddResidualBlock(cost_function, NULL, &Allpt3D[trackID][3 * ll], &Allpt3D[trackID][3 * (ll + 1)], &currentOffset[camID1]);
				}
				else
				{
					ceres::CostFunction* cost_function = LeastActionCostCeres::CreateAutoDiffSame(VectorFrameID[trackID][ll], VectorFrameID[trackID][ll + 1], ialpha, Tscale, eps, Traj2DAll[trackID][ll].std3D, motionPriorPower);
					problem.AddResidualBlock(cost_function, NULL, &Allpt3D[trackID][3 * ll], &Allpt3D[trackID][3 * (ll + 1)], &currentOffset[camID1]);
				}

			}
			else
			{
				if (Traj2DAll[trackID][ll].std3D < 0.0)
				{
					ceres::CostFunction* cost_function = LeastActionCostCeres::CreateAutoDiff(VectorFrameID[trackID][ll], VectorFrameID[trackID][ll + 1], ialpha, Tscale, eps, 1.0 / sqrt(1.0 - lamda), motionPriorPower);
					//ceres::CostFunction* cost_function = LeastActionCostCeres::CreateNumerDiff(VectorFrameID[trackID][ll], VectorFrameID[trackID][ll + 1], ialpha, Tscale, rate, eps, motionPriorPower);
					problem.AddResidualBlock(cost_function, NULL, &Allpt3D[trackID][3 * ll], &Allpt3D[trackID][3 * (ll + 1)], &currentOffset[camID1], &currentOffset[camID2]);
				}
				else
				{
					ceres::CostFunction* cost_function = LeastActionCostCeres::CreateAutoDiff(VectorFrameID[trackID][ll], VectorFrameID[trackID][ll + 1], ialpha, Tscale, eps, Traj2DAll[trackID][ll].std3D, motionPriorPower);
					problem.AddResidualBlock(cost_function, NULL, &Allpt3D[trackID][3 * ll], &Allpt3D[trackID][3 * (ll + 1)], &currentOffset[camID1], &currentOffset[camID2]);
				}
			}
		}

		for (int ll = 0; ll < npts; ll++)
		{
			Point2d P2d(Traj2DAll[trackID][ll].pt2D.x, Traj2DAll[trackID][ll].pt2D.y);
			Point3d P3d(Allpt3D[trackID][3 * ll], Allpt3D[trackID][3 * ll + 1], Allpt3D[trackID][3 * ll + 2]);
			costi = PinholeReprojectionErrorSimpleDebug(Traj2DAll[trackID][ll].P, P3d, P2d);
			ProjCost += costi;

			if (Traj2DAll[trackID][ll].std3D < 0.0)
			{
				ceres::CostFunction* cost_function = IdealGeoProjectionCeres::Create(Traj2DAll[trackID][ll].P, Traj2DAll[trackID][ll].pt2D, 1.0 / sqrt(lamda));
				problem.AddResidualBlock(cost_function, NULL, Allpt3D[trackID] + 3 * ll);
			}
			else
			{
				ceres::CostFunction* cost_function = IdealGeoProjectionCeres::Create(Traj2DAll[trackID][ll].P, Traj2DAll[trackID][ll].pt2D, Traj2DAll[trackID][ll].std2D);
				problem.AddResidualBlock(cost_function, NULL, Allpt3D[trackID] + 3 * ll);
			}
		}
	}
	for (int trackID = 0; trackID < ntracks; trackID++)
		PerPoint_nFrames.push_back(Traj2DAll[trackID].size());
	//ceres::LossFunction* loss_function = new ceres::HuberLoss(10.0);

	//Set bound on the time 
	double Initoffset[1000];
	Initoffset[0] = 0;
	for (int camID = 1; camID < nCams; camID++)
	{
		Initoffset[camID] = currentOffset[camID];
		//problem.SetParameterLowerBound(&currentOffset[camID], 0, Initoffset[camID] - 0.5), problem.SetParameterUpperBound(&currentOffset[camID], 0, Initoffset[camID] + 0.5);
		problem.SetParameterLowerBound(&currentOffset[camID], 0, floor(Initoffset[camID])), problem.SetParameterUpperBound(&currentOffset[camID], 0, ceil(Initoffset[camID])); //lock it in the frame
	}

	//Set fixed parameters
	problem.SetParameterBlockConstant(&currentOffset[0]);

	if (!silent)
		printf("Action cost: %e Projection cost: %e\n", ActionCost, ProjCost);

	ceres::Solver::Options options;
	options.num_threads = omp_get_max_threads();
	options.num_linear_solver_threads = omp_get_max_threads();
	options.max_num_iterations = 3000;
	options.linear_solver_type = ceres::SPARSE_SCHUR;//SPARSE_NORMAL_CHOLESKY
	options.minimizer_progress_to_stdout = !silent;
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	options.use_nonmonotonic_steps = non_monotonicDescent;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	if (!silent)
		std::cout << summary.FullReport() << "\n";
	else
		std::cout << summary.BriefReport() << "\n";

	double CeresCost = summary.final_cost;
	//Save data
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		int npts = Traj2DAll[trackID].size();
		for (int ii = 0; ii < npts; ii++)
		{
			int camID = VectorCamID[trackID][ii], frameID = VectorFrameID[trackID][ii];

			bool found = false;
			for (int kk = 0; kk < PerCam_UV[camID*ntracks + trackID].size(); kk++)
			{
				if (frameID == PerCam_UV[camID*ntracks + trackID][kk].frameID)
				{
					PerCam_UV[camID*ntracks + trackID][kk].pt3D = Point3d(Allpt3D[trackID][3 * ii], Allpt3D[trackID][3 * ii + 1], Allpt3D[trackID][3 * ii + 2]);
					found = true;
					break;
				}
			}
			if (!found)
			{
				printf("Serious bug in point-camera-frame association\n");
				abort();
			}
		}
	}


	//Compute cost after optim
	ActionCost = 0.0, ProjCost = 0.0;
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		int npts = PerPoint_nFrames[trackID];
		double costi;
		for (int ll = 0; ll < npts - 1; ll++)
		{
			int camID1 = VectorCamID[trackID][ll], camID2 = VectorCamID[trackID][ll + 1];

			costi = LeastActionError(&Allpt3D[trackID][3 * ll], &Allpt3D[trackID][3 * ll + 3], &currentOffset[camID1], &currentOffset[camID2], VectorFrameID[trackID][ll], VectorFrameID[trackID][ll + 1], ialpha, Tscale, eps, motionPriorPower);
			ActionCost += costi;

			Point2d P2d_1(Traj2DAll[trackID][ll].pt2D.x, Traj2DAll[trackID][ll].pt2D.y);
			Point3d P3d_1(Allpt3D[trackID][3 * ll], Allpt3D[trackID][3 * ll + 1], Allpt3D[trackID][3 * ll + 2]);
			costi = PinholeReprojectionErrorSimpleDebug(Traj2DAll[trackID][ll].P, P3d_1, P2d_1);
			ProjCost += costi;

			Point2d P2d_2(Traj2DAll[trackID][ll].pt2D.x, Traj2DAll[trackID][ll].pt2D.y);
			Point3d P3d_2(Allpt3D[trackID][3 * ll], Allpt3D[trackID][3 * ll + 1], Allpt3D[trackID][3 * ll + 2]);
			costi = PinholeReprojectionErrorSimpleDebug(Traj2DAll[trackID][ll].P, P3d_2, P2d_2);
			ProjCost += costi;
		}
	}

	double lengthCost = 0.0;
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		int npts = Traj2DAll[trackID].size();
		double costi;
		for (int ll = 0; ll < npts - 1; ll++)
		{
			costi = sqrt(pow(Allpt3D[trackID][3 * ll] - Allpt3D[trackID][3 * ll + 3], 2) + pow(Allpt3D[trackID][3 * ll + 1] - Allpt3D[trackID][3 * ll + 4], 2) + pow(Allpt3D[trackID][3 * ll + 2] - Allpt3D[trackID][3 * ll + 5], 2));
			lengthCost += costi;
		}
	}

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

	if (!silent)
		printf("Action cost: %e Projection cost: %e Distance Cost: %e Direction Cost %e\n ", ActionCost, ProjCost, lengthCost, directionCost);

	Cost[0] = ActionCost, Cost[1] = ProjCost, Cost[2] = lengthCost, Cost[3] = directionCost;

	delete[]VectorCamID, delete[]VectorFrameID, delete[]Traj2DAll;
	delete[]currentFrame, delete[]PerCam_nf, delete[]currentPID_InTrack;

	if (StillImages)
		delete[]StillImageTimeOrderID, delete[]StillImageTimeOrder;

	return CeresCost;
}
double MotionPrior_Optim_ST_GeometricDynamic(char *Path, vector<double*> &Allpt3D, vector<ImgPtEle> *PerCam_UV, vector<int> &PerPoint_nFrames, double *currentOffset, int ntracks, bool non_monotonicDescent, int nCams, int motionPriorPower, double Tscale, double ialpha, double eps, double lamda, double *Cost, bool StillImages = false, bool silent = true)
{
	double ActionCost = 0.0, ProjCost = 0.0;

	int *currentFrame = new int[nCams], *PerCam_nf = new int[nCams], *currentPID_InTrack = new int[nCams];
	Point3d P3D;
	ImgPtEle ptEle;

	vector<double> VectorTime;
	vector<int> *VectorCamID = new vector<int>[ntracks], *VectorFrameID = new vector<int>[ntracks];
	vector<ImgPtEle> *Traj2DAll = new vector<ImgPtEle>[ntracks];
	vector<double *> PmatDataAll;
	vector<Point2d*>p2dCatAll;

	int npts, maxnpts = 0;
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		npts = 0;
		for (int camID = 0; camID < nCams; camID++)
			npts += PerCam_UV[camID*ntracks + trackID].size();
		maxnpts = max(maxnpts, npts);
	}
	int *ArrayCamID = new int[maxnpts], *ArrayFrameID = new int[maxnpts];

	double earliestTime, currentTime, RollingShutterOffset;
	int earliestCamFrameID, frameID, nfinishedCams, earliestCamID;

	int *StillImageTimeOrderID = 0;
	double *StillImageTimeOrder = 0;
	if (StillImages)
	{
		StillImageTimeOrderID = new int[nCams];
		StillImageTimeOrder = new double[nCams];
		for (int camID = 0; camID < nCams; camID++)
		{
			StillImageTimeOrder[camID] = currentOffset[camID];
			StillImageTimeOrderID[camID] = camID;
		}
		Quick_Sort_Double(StillImageTimeOrder, StillImageTimeOrderID, 0, nCams - 1);
	}

	ceres::Problem problem;
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		if (StillImages)
		{
			VectorTime.clear();

			for (int ii = 0; ii < nCams; ii++)
			{
				int camID = StillImageTimeOrderID[ii];
				double currentTime = StillImageTimeOrder[ii];
				VectorTime.push_back(currentTime);
				VectorCamID[trackID].push_back(camID);
				VectorFrameID[trackID].push_back(0);
				Traj2DAll[trackID].push_back(PerCam_UV[camID*ntracks + trackID][0]);
			}
		}
		else
		{
			for (int camID = 0; camID < nCams; camID++)
				PerCam_nf[camID] = PerCam_UV[camID*ntracks + trackID].size();

			//Assemble trajactory and time from all Cameras
			VectorTime.clear();
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
					frameID = PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].frameID;
					RollingShutterOffset = 0;// PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].pt2D.y / PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].imHeight;
					currentTime = (currentOffset[camID] + frameID + RollingShutterOffset) * ialpha*Tscale;

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
				VectorCamID[trackID].push_back(earliestCamID);
				VectorFrameID[trackID].push_back(earliestCamFrameID);
				Traj2DAll[trackID].push_back(PerCam_UV[earliestCamID*ntracks + trackID][currentPID_InTrack[earliestCamID]]);

				currentPID_InTrack[earliestCamID]++;
			}
		}

		//Initialize memory for b spline
		int npts = Traj2DAll[trackID].size();
		Allpt3D[trackID] = new double[3 * npts];
		Point2d *p2dCat = new Point2d[npts];
		double *PmatData = new double[12 * npts];

		PmatDataAll.push_back(PmatData);
		p2dCatAll.push_back(p2dCat);

		for (int jj = 0; jj < npts; jj++)
		{
			for (int ii = 0; ii < 12; ii++)
				PmatData[12 * jj + ii] = Traj2DAll[trackID][jj].P[ii];

			p2dCat[jj] = Traj2DAll[trackID][jj].pt2D;

			Allpt3D[trackID][3 * jj] = Traj2DAll[trackID][jj].pt3D.x,
				Allpt3D[trackID][3 * jj + 1] = Traj2DAll[trackID][jj].pt3D.y,
				Allpt3D[trackID][3 * jj + 2] = Traj2DAll[trackID][jj].pt3D.z;

			ArrayCamID[jj] = VectorCamID[trackID][jj],
				ArrayFrameID[jj] = VectorFrameID[trackID][jj];
		}

		//ceres::DynamicNumericDiffCostFunction<LeastActionCostDynamicCeres, ceres::CENTRAL> *cost_function = new ceres::DynamicNumericDiffCostFunction<LeastActionCostDynamicCeres, ceres::CENTRAL>(new LeastActionCostDynamicCeres(ArrayCamID, ArrayFrameID, PmatData, p2dCat, lamda, ialpha, Tscale, npts));
		ceres::DynamicAutoDiffCostFunction<LeastActionCostDynamicCeres, 4> *cost_function = new ceres::DynamicAutoDiffCostFunction<LeastActionCostDynamicCeres, 4>(new LeastActionCostDynamicCeres(ArrayCamID, ArrayFrameID, PmatData, p2dCat, lamda, ialpha, Tscale, npts));
		vector<double*> parameter_blocks;
		for (int ii = 0; ii < npts; ii++)
			parameter_blocks.push_back(&Allpt3D[trackID][3 * ii]), cost_function->AddParameterBlock(3);
		parameter_blocks.push_back(currentOffset), cost_function->AddParameterBlock(nCams);
		cost_function->SetNumResiduals(3 * npts - 1);
		problem.AddResidualBlock(cost_function, NULL, parameter_blocks);

		//Set bound on the time 
		for (int camID = 1; camID < nCams; camID++)
		{
			double InitialOffset = currentOffset[camID];
			problem.SetParameterLowerBound(parameter_blocks[npts], camID, floor(InitialOffset)), problem.SetParameterUpperBound(parameter_blocks[npts], camID, ceil(InitialOffset)); //lock it in the frame
		}

		//Set fixed parameters
		problem.SetParameterBlockConstant(&parameter_blocks[npts][0]);
	}

	for (int trackID = 0; trackID < ntracks; trackID++)
		PerPoint_nFrames.push_back(Traj2DAll[trackID].size());


	if (!silent)
		printf("Action cost: %e Projection cost: %e\n", ActionCost, ProjCost);

	ceres::Solver::Options options;
	options.num_threads = omp_get_max_threads();
	options.num_linear_solver_threads = omp_get_max_threads();
	options.max_num_iterations = 5000;
	options.linear_solver_type = ceres::SPARSE_SCHUR;//SPARSE_NORMAL_CHOLESKY
	options.minimizer_progress_to_stdout = true;
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	options.use_nonmonotonic_steps = non_monotonicDescent;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	//if (!silent)
	std::cout << summary.FullReport() << "\n";
	//else
	//	std::cout << summary.BriefReport() << "\n";
	double CeresCost = summary.final_cost;

	//Save data
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		int npts = Traj2DAll[trackID].size();
		for (int ii = 0; ii < npts; ii++)
		{
			int camID = VectorCamID[trackID][ii], frameID = VectorFrameID[trackID][ii];

			bool found = false;
			for (int kk = 0; kk < PerCam_UV[camID*ntracks + trackID].size(); kk++)
			{
				if (frameID == PerCam_UV[camID*ntracks + trackID][kk].frameID)
				{
					PerCam_UV[camID*ntracks + trackID][kk].pt3D = Point3d(Allpt3D[trackID][3 * ii], Allpt3D[trackID][3 * ii + 1], Allpt3D[trackID][3 * ii + 2]);
					found = true;
					break;
				}
			}
			if (!found)
			{
				printf("Serious bug in point-camera-frame association\n");
				abort();
			}
		}
	}


	//Compute cost after optim
	ActionCost = 0.0, ProjCost = 0.0;
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		int npts = PerPoint_nFrames[trackID];
		double costi;
		for (int ll = 0; ll < npts - 1; ll++)
		{
			int camID1 = VectorCamID[trackID][ll], camID2 = VectorCamID[trackID][ll + 1];

			costi = LeastActionError(&Allpt3D[trackID][3 * ll], &Allpt3D[trackID][3 * ll + 3], &currentOffset[camID1], &currentOffset[camID2], VectorFrameID[trackID][ll], VectorFrameID[trackID][ll + 1], ialpha, Tscale, eps, motionPriorPower);
			ActionCost += costi;

			Point2d P2d_1(Traj2DAll[trackID][ll].pt2D.x, Traj2DAll[trackID][ll].pt2D.y);
			Point3d P3d_1(Allpt3D[trackID][3 * ll], Allpt3D[trackID][3 * ll + 1], Allpt3D[trackID][3 * ll + 2]);
			costi = PinholeReprojectionErrorSimpleDebug(Traj2DAll[trackID][ll].P, P3d_1, P2d_1);
			ProjCost += costi;

			Point2d P2d_2(Traj2DAll[trackID][ll].pt2D.x, Traj2DAll[trackID][ll].pt2D.y);
			Point3d P3d_2(Allpt3D[trackID][3 * ll], Allpt3D[trackID][3 * ll + 1], Allpt3D[trackID][3 * ll + 2]);
			costi = PinholeReprojectionErrorSimpleDebug(Traj2DAll[trackID][ll].P, P3d_2, P2d_2);
			ProjCost += costi;
		}
	}

	double lengthCost = 0.0;
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		int npts = Traj2DAll[trackID].size();
		double costi;
		for (int ll = 0; ll < npts - 1; ll++)
		{
			costi = sqrt(pow(Allpt3D[trackID][3 * ll] - Allpt3D[trackID][3 * ll + 3], 2) + pow(Allpt3D[trackID][3 * ll + 1] - Allpt3D[trackID][3 * ll + 4], 2) + pow(Allpt3D[trackID][3 * ll + 2] - Allpt3D[trackID][3 * ll + 5], 2));
			lengthCost += costi;
		}
	}

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

	if (!silent)
		printf("Action cost: %e Projection cost: %e Distance Cost: %e Direction Cost %e\n ", ActionCost, ProjCost, lengthCost, directionCost);

	Cost[0] = ActionCost, Cost[1] = ProjCost, Cost[2] = lengthCost, Cost[3] = directionCost;

	delete[]VectorCamID, delete[]VectorFrameID, delete[]Traj2DAll;
	delete[]ArrayCamID, delete[]ArrayFrameID, delete[]currentFrame, delete[]PerCam_nf, delete[]currentPID_InTrack;

	for (int ii = 0; ii < ntracks; ii++)
		delete PmatDataAll[ii], delete[]p2dCatAll[ii];

	if (StillImages)
		delete[]StillImageTimeOrderID, delete[]StillImageTimeOrder;

	return CeresCost;
}
double MotionPrior_Optim_ST_GeometricSpline(char *Path, vector<double*> &Allpt3D, vector<ImgPtEle> *PerCam_UV, vector<int> &PerPoint_nFrames, double *currentOffset, int ntracks, bool non_monotonicDescent, int nCams, int motionPriorPower, double Tscale, double ialpha, double eps, double lamda, double *Cost, bool StillImages = false, bool silent = true)
{
	const int UpsamplingRate = 2;
	double ActionCost = 0.0, ProjCost = 0.0;

	int *currentFrame = new int[nCams], *PerCam_nf = new int[nCams], *currentPID_InTrack = new int[nCams];
	Point3d P3D;
	ImgPtEle ptEle;

	vector<int>IdToDel;
	vector<double> VectorTime;
	vector<int> *VectorCamID = new vector<int>[ntracks], *VectorFrameID = new vector<int>[ntracks];
	vector<ImgPtEle> *Traj2DAll = new vector<ImgPtEle>[ntracks];
	vector<double *> PmatDataAll, BreakPtsAll, PhiDataAll, PhiPriorAll, TimeStampDataAll, TimeStampPriorAll, CoeffsAll, TempAll;

	int npts, maxnpts = 0;
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		npts = 0;
		for (int camID = 0; camID < nCams; camID++)
			npts += PerCam_UV[camID*ntracks + trackID].size();
		maxnpts = max(maxnpts, npts);
	}
	int *ArrayCamID = new int[maxnpts], *ArrayFrameID = new int[maxnpts];

	double earliestTime, currentTime, RollingShutterOffset;
	int earliestCamFrameID, frameID, nfinishedCams, earliestCamID;

	int *StillImageTimeOrderID = 0;
	double *StillImageTimeOrder = 0;
	if (StillImages)
	{
		StillImageTimeOrderID = new int[nCams];
		StillImageTimeOrder = new double[nCams];
		for (int camID = 0; camID < nCams; camID++)
		{
			StillImageTimeOrder[camID] = currentOffset[camID];
			StillImageTimeOrderID[camID] = camID;
		}
		Quick_Sort_Double(StillImageTimeOrder, StillImageTimeOrderID, 0, nCams - 1);
	}

	ceres::Problem problem;
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		if (StillImages)
		{
			VectorTime.clear();

			for (int ii = 0; ii < nCams; ii++)
			{
				int camID = StillImageTimeOrderID[ii];
				double currentTime = StillImageTimeOrder[ii];
				VectorTime.push_back(currentTime);
				VectorCamID[trackID].push_back(camID);
				VectorFrameID[trackID].push_back(0);
				Traj2DAll[trackID].push_back(PerCam_UV[camID*ntracks + trackID][0]);
			}
		}
		else
		{
			for (int camID = 0; camID < nCams; camID++)
				PerCam_nf[camID] = PerCam_UV[camID*ntracks + trackID].size();

			//Assemble trajactory and time from all Cameras
			VectorTime.clear();
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
					frameID = PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].frameID;
					RollingShutterOffset = 0;// PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].pt2D.y / PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].imHeight;
					currentTime = (currentOffset[camID] + frameID + RollingShutterOffset) * ialpha*Tscale;

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
				VectorCamID[trackID].push_back(earliestCamID);
				VectorFrameID[trackID].push_back(earliestCamFrameID);
				Traj2DAll[trackID].push_back(PerCam_UV[earliestCamID*ntracks + trackID][currentPID_InTrack[earliestCamID]]);

				currentPID_InTrack[earliestCamID]++;
			}
		}

		//Initialize memory for b spline
		int npts = Traj2DAll[trackID].size();
		Allpt3D[trackID] = new double[3 * npts];
		Point2d *p2dCat = new Point2d[npts];
		double *PmatData = new double[12 * npts];
		int nBreaks = npts / nCams, nPriorPts = npts*UpsamplingRate;
		double *BreakPts = new double[nBreaks];
		double *TimeStampData = new double[npts];
		double *TimeStampPrior = new double[nPriorPts];
		double *PhiData = new double[npts*(nBreaks + 2)];
		double *PhiPrior = new double[nPriorPts*(nBreaks + 2)];
		double *Coeffs = new double[3 * (nBreaks + 2)];
		double *TempIn = new double[3 * npts];

		PmatDataAll.push_back(PmatData);
		PhiDataAll.push_back(PhiData);
		PhiPriorAll.push_back(PhiPrior);
		BreakPtsAll.push_back(BreakPts);
		TimeStampDataAll.push_back(TimeStampData);
		TimeStampPriorAll.push_back(TimeStampPrior);
		CoeffsAll.push_back(Coeffs);
		TempAll.push_back(TempIn);

		for (int jj = 0; jj < npts; jj++)
		{
			for (int ii = 0; ii < 12; ii++)
				PmatData[12 * jj + ii] = Traj2DAll[trackID][jj].P[ii];

			p2dCat[jj] = Traj2DAll[trackID][jj].pt2D;

			Allpt3D[trackID][3 * jj] = Traj2DAll[trackID][jj].pt3D.x,
				Allpt3D[trackID][3 * jj + 1] = Traj2DAll[trackID][jj].pt3D.y,
				Allpt3D[trackID][3 * jj + 2] = Traj2DAll[trackID][jj].pt3D.z;

			ArrayCamID[jj] = VectorCamID[trackID][jj],
				ArrayFrameID[jj] = VectorFrameID[trackID][jj];
		}

		//Compute break points
		double Break_Step = (VectorTime.back() - VectorTime.front() + 2.0*ialpha*Tscale) / (nBreaks - 1);//intentionally expand the break location to avoid time instances to go out of the time brackets
		for (int ii = 0; ii < nBreaks; ii++)
			BreakPts[ii] = floor(VectorTime[0] - ialpha*Tscale) + Break_Step*ii;

		//Find and delete breakpts with no data in between, which causes the basis matrix to has high condition value
		IdToDel.clear();
		for (int ii = 0; ii < nBreaks - 1; ii++)
		{
			bool found = false;
			for (int jj = 0; jj < npts; jj++)
			{
				if ((BreakPts[ii] < VectorTime[jj] && VectorTime[jj] < BreakPts[ii + 1]))
				{
					found = true;
					break;
				}
			}
			if (!found)
				IdToDel.push_back(ii + 1);
		}
		if ((int)IdToDel.size() > 0)
		{
			vector<double> tBreakPts;
			for (int ii = 0; ii < nBreaks; ii++)
				tBreakPts.push_back(BreakPts[ii]);

			for (int ii = (int)IdToDel.size() - 1; ii >= 0; ii--)
				tBreakPts.erase(tBreakPts.begin() + IdToDel[ii]);

			nBreaks = (int)tBreakPts.size();
			for (int ii = 0; ii < nBreaks; ii++)
				BreakPts[ii] = tBreakPts[ii];
		}

		vector<double*> parameter_blocks;
		parameter_blocks.push_back(Allpt3D[trackID]);
		parameter_blocks.push_back(currentOffset);
		ceres::DynamicNumericDiffCostFunction<LeastActionCostSplineCeres, ceres::CENTRAL> *cost_function = new ceres::DynamicNumericDiffCostFunction<LeastActionCostSplineCeres, ceres::CENTRAL>
			(new LeastActionCostSplineCeres(ArrayCamID, ArrayFrameID, PmatData, p2dCat, BreakPts, PhiData, PhiPrior, TimeStampData, TimeStampPrior, Coeffs, TempIn, lamda, ialpha, Tscale, npts, UpsamplingRate, nBreaks));
		cost_function->AddParameterBlock(3 * npts);
		cost_function->AddParameterBlock(nCams);
		cost_function->SetNumResiduals(2 * npts + 1);
		problem.AddResidualBlock(cost_function, NULL, parameter_blocks);

		//Set bound on the time 
		double Initoffset[1000];
		Initoffset[0] = 0;
		for (int camID = 1; camID < nCams; camID++)
		{
			Initoffset[camID] = currentOffset[camID];
			//problem.SetParameterLowerBound(&currentOffset[camID], 0, Initoffset[camID] - 0.5), problem.SetParameterUpperBound(&currentOffset[camID], 0, Initoffset[camID] + 0.5);
			//problem.SetParameterLowerBound(&currentOffset[camID], 0, floor(Initoffset[camID])), problem.SetParameterUpperBound(&currentOffset[camID], 0, ceil(Initoffset[camID])); //lock it in the frame
			problem.SetParameterLowerBound(parameter_blocks[1], camID, floor(Initoffset[camID])), problem.SetParameterUpperBound(parameter_blocks[1], camID, ceil(Initoffset[camID])); //lock it in the frame
		}

		//Set fixed parameters
		//problem.SetParameterLowerBound(parameter_blocks[1], 0, 0.0), problem.SetParameterUpperBound(parameter_blocks[1], 0, 0.0); 
		problem.SetParameterBlockConstant(&parameter_blocks[1][0]);
	}

	for (int trackID = 0; trackID < ntracks; trackID++)
		PerPoint_nFrames.push_back(Traj2DAll[trackID].size());
	//ceres::LossFunction* loss_function = new ceres::HuberLoss(10.0);



	if (!silent)
		printf("Action cost: %e Projection cost: %e\n", ActionCost, ProjCost);

	ceres::Solver::Options options;
	options.num_threads = omp_get_max_threads();
	options.num_linear_solver_threads = omp_get_max_threads();
	options.max_num_iterations = 5000;
	options.linear_solver_type = ceres::SPARSE_SCHUR;//SPARSE_NORMAL_CHOLESKY
	options.minimizer_progress_to_stdout = true;
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	options.use_nonmonotonic_steps = non_monotonicDescent;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	//if (!silent)
	std::cout << summary.FullReport() << "\n";
	//else
	//	std::cout << summary.BriefReport() << "\n";

	double CeresCost = summary.final_cost;
	//Save data
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		int npts = Traj2DAll[trackID].size();
		for (int ii = 0; ii < npts; ii++)
		{
			int camID = VectorCamID[trackID][ii], frameID = VectorFrameID[trackID][ii];

			bool found = false;
			for (int kk = 0; kk < PerCam_UV[camID*ntracks + trackID].size(); kk++)
			{
				if (frameID == PerCam_UV[camID*ntracks + trackID][kk].frameID)
				{
					PerCam_UV[camID*ntracks + trackID][kk].pt3D = Point3d(Allpt3D[trackID][3 * ii], Allpt3D[trackID][3 * ii + 1], Allpt3D[trackID][3 * ii + 2]);
					found = true;
					break;
				}
			}
			if (!found)
			{
				printf("Serious bug in point-camera-frame association\n");
				abort();
			}
		}
	}


	//Compute cost after optim
	ActionCost = 0.0, ProjCost = 0.0;
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		int npts = PerPoint_nFrames[trackID];
		double costi;
		for (int ll = 0; ll < npts - 1; ll++)
		{
			int camID1 = VectorCamID[trackID][ll], camID2 = VectorCamID[trackID][ll + 1];

			costi = LeastActionError(&Allpt3D[trackID][3 * ll], &Allpt3D[trackID][3 * ll + 3], &currentOffset[camID1], &currentOffset[camID2], VectorFrameID[trackID][ll], VectorFrameID[trackID][ll + 1], ialpha, Tscale, eps, motionPriorPower);
			ActionCost += costi;

			Point2d P2d_1(Traj2DAll[trackID][ll].pt2D.x, Traj2DAll[trackID][ll].pt2D.y);
			Point3d P3d_1(Allpt3D[trackID][3 * ll], Allpt3D[trackID][3 * ll + 1], Allpt3D[trackID][3 * ll + 2]);
			costi = PinholeReprojectionErrorSimpleDebug(Traj2DAll[trackID][ll].P, P3d_1, P2d_1);
			ProjCost += costi;

			Point2d P2d_2(Traj2DAll[trackID][ll].pt2D.x, Traj2DAll[trackID][ll].pt2D.y);
			Point3d P3d_2(Allpt3D[trackID][3 * ll], Allpt3D[trackID][3 * ll + 1], Allpt3D[trackID][3 * ll + 2]);
			costi = PinholeReprojectionErrorSimpleDebug(Traj2DAll[trackID][ll].P, P3d_2, P2d_2);
			ProjCost += costi;
		}
	}

	double lengthCost = 0.0;
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		int npts = Traj2DAll[trackID].size();
		double costi;
		for (int ll = 0; ll < npts - 1; ll++)
		{
			costi = sqrt(pow(Allpt3D[trackID][3 * ll] - Allpt3D[trackID][3 * ll + 3], 2) + pow(Allpt3D[trackID][3 * ll + 1] - Allpt3D[trackID][3 * ll + 4], 2) + pow(Allpt3D[trackID][3 * ll + 2] - Allpt3D[trackID][3 * ll + 5], 2));
			lengthCost += costi;
		}
	}

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

	if (!silent)
		printf("Action cost: %e Projection cost: %e Distance Cost: %e Direction Cost %e\n ", ActionCost, ProjCost, lengthCost, directionCost);

	Cost[0] = ActionCost, Cost[1] = ProjCost, Cost[2] = lengthCost, Cost[3] = directionCost;

	delete[]VectorCamID, delete[]VectorFrameID, delete[]Traj2DAll;
	delete[]ArrayCamID, delete[]ArrayFrameID, delete[]currentFrame, delete[]PerCam_nf, delete[]currentPID_InTrack;

	for (int ii = 0; ii < ntracks; ii++)
		delete PmatDataAll[ii], delete[]BreakPtsAll[ii], delete[]PhiDataAll[ii], delete[]TimeStampDataAll[ii], delete[]TimeStampPriorAll[ii], delete[]CoeffsAll[ii], delete[]TempAll[ii];

	if (StillImages)
		delete[]StillImageTimeOrderID, delete[]StillImageTimeOrder;

	return CeresCost;
}

double LeastActionSyncBruteForce2DStereo(char *Path, vector<int> &SelectedCams, int startFrame, int stopFrame, int ntracks, vector<double> &OffsetInfo, int LowBound, int UpBound, double frameSize, double lamda, int motionPriorPower, int &totalPoints, bool silient = true)
{
	//Offset is in timestamp format
	const double Tscale = 1000.0, fps = 60, ialpha = 1.0 / fps, eps = 1.0e-6;
	char Fname[200]; FILE *fp = 0;
	const int nCams = 2;

	//Read calib info
	VideoData VideoInfo[2];
	if (ReadVideoDataI(Path, VideoInfo[0], SelectedCams[0], startFrame, stopFrame) == 1)
		abort();
	if (ReadVideoDataI(Path, VideoInfo[1], SelectedCams[1], startFrame, stopFrame) == 1)
		abort();

	int id, frameID, npts;
	int nframes = max(MaxnFrames, stopFrame);

	double u, v;
	ImgPtEle ptEle;
	vector<ImgPtEle> *PerCam_UV = new vector<ImgPtEle>[nCams*ntracks];
	vector<XYZD> *PerCam_XYZ = new vector<XYZD>[nCams], *XYZ = new vector<XYZD>[ntracks], XYZBK;

	//Get 2D info
	for (int camID = 0; camID < nCams; camID++)
	{
		for (int trackID = 0; trackID < ntracks; trackID++)
			PerCam_UV[camID*ntracks + trackID].reserve(stopFrame - startFrame + 1);

		sprintf(Fname, "%s/Track2D/%d.txt", Path, SelectedCams[camID]); FILE *fp = fopen(Fname, "r");
		for (int trackID = 0; trackID < ntracks; trackID++)
		{
			fscanf(fp, "%d %d ", &id, &npts);
			if (id != trackID)
				printf("Problem at Point %d of Cam %d", id, camID);
			for (int pid = 0; pid < npts; pid++)
			{
				fscanf(fp, "%d %lf %lf ", &frameID, &u, &v);
				if (frameID < startFrame)
					continue;
				if (abs(VideoInfo[camID].VideoInfo[frameID].camCenter[0]) + abs(VideoInfo[camID].VideoInfo[frameID].camCenter[1]) + abs(VideoInfo[camID].VideoInfo[frameID].camCenter[2]) < 2)
					continue; //camera not localized

				if (u > 0 && v > 0)
				{
					ptEle.pt2D.x = u, ptEle.pt2D.y = v, ptEle.viewID = camID, ptEle.frameID = frameID;
					LensCorrectionPoint(&ptEle.pt2D, VideoInfo[camID].VideoInfo[frameID].K, VideoInfo[camID].VideoInfo[frameID].distortion);
					PerCam_UV[camID*ntracks + id].push_back(ptEle);
				}
			}
		}
		fclose(fp);
	}

	//Generate Calib Info
	double P[12], AA[6], bb[2], ccT[3], dd[1];
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		//Get ray direction, Q, U, P
		int count = 0;
		for (int camID = 0; camID < nCams; camID++)
		{
			for (int frameID = 0; frameID < PerCam_UV[camID*ntracks + trackID].size(); frameID++)
			{
				int RealFrameID = PerCam_UV[camID*ntracks + trackID][frameID].frameID;

				for (int kk = 0; kk < 12; kk++)
				{
					P[kk] = VideoInfo[camID].VideoInfo[RealFrameID].P[kk];
					PerCam_UV[camID*ntracks + trackID][frameID].P[kk] = P[kk];
				}

				for (int kk = 0; kk < 9; kk++)
					PerCam_UV[camID*ntracks + trackID][frameID].K[kk] = VideoInfo[camID].VideoInfo[RealFrameID].K[kk],
					PerCam_UV[camID*ntracks + trackID][frameID].R[kk] = VideoInfo[camID].VideoInfo[RealFrameID].R[kk];

				for (int kk = 0; kk < 3; kk++)
					PerCam_UV[camID*ntracks + trackID][frameID].camcenter[kk] = VideoInfo[camID].VideoInfo[RealFrameID].camCenter[kk];

				//Q, U
				AA[0] = P[0], AA[1] = P[1], AA[2] = P[2], bb[0] = P[3];
				AA[3] = P[4], AA[4] = P[5], AA[5] = P[6], bb[1] = P[7];
				ccT[0] = P[8], ccT[1] = P[9], ccT[2] = P[10], dd[0] = P[11];

				PerCam_UV[camID*ntracks + trackID][frameID].Q[0] = AA[0] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.x*ccT[0],
					PerCam_UV[camID*ntracks + trackID][frameID].Q[1] = AA[1] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.x*ccT[1],
					PerCam_UV[camID*ntracks + trackID][frameID].Q[2] = AA[2] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.x*ccT[2];
				PerCam_UV[camID*ntracks + trackID][frameID].Q[3] = AA[3] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.y*ccT[0],
					PerCam_UV[camID*ntracks + trackID][frameID].Q[4] = AA[4] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.y*ccT[1],
					PerCam_UV[camID*ntracks + trackID][frameID].Q[5] = AA[5] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.y*ccT[2];
				PerCam_UV[camID*ntracks + trackID][frameID].u[0] = dd[0] * PerCam_UV[camID*ntracks + trackID][frameID].pt2D.x - bb[0],
					PerCam_UV[camID*ntracks + trackID][frameID].u[1] = dd[0] * PerCam_UV[camID*ntracks + trackID][frameID].pt2D.y - bb[1];

				PerCam_UV[camID*ntracks + trackID][frameID].pt3D = Point3d(count, count, count);//Interestingly, Ceres does not work if all the input are the same
				count++;
			}
		}
	}

	//Initialize data for optim
	totalPoints = 0;
	vector<int> PointsPerTrack;
	vector<int *> PerTrackFrameID(ntracks);
	vector<double*> All3D(ntracks);
	int ntimeinstances, maxntimeinstances = 0;
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		ntimeinstances = 0;
		for (int camID = 0; camID < nCams; camID++)
			ntimeinstances += PerCam_UV[camID*ntracks + trackID].size();
		totalPoints += ntimeinstances;

		if (maxntimeinstances < ntimeinstances)
			maxntimeinstances = ntimeinstances;

		PerTrackFrameID[trackID] = new int[ntimeinstances];
		All3D[trackID] = new double[3 * ntimeinstances];
	}


	//Start sliding
	double currentOffset[2], APLDCost[4], ceresCost;
	vector<double> VTimeStamp; VTimeStamp.reserve(maxntimeinstances);
	vector<Point3d> VTrajectory3D; VTrajectory3D.reserve(maxntimeinstances);
	int *OffsetID = new int[UpBound - LowBound + 1];
	double*AllCost = new double[UpBound - LowBound + 1],
		*AllACost = new double[UpBound - LowBound + 1],
		*AllPCost = new double[UpBound - LowBound + 1],
		*AllLCost = new double[UpBound - LowBound + 1],
		*AllDCost = new double[UpBound - LowBound + 1];

	int count = 0;
	for (int off = LowBound; off <= UpBound; off++)
	{
		OffsetID[off - LowBound] = off;
		currentOffset[0] = OffsetInfo[0], currentOffset[1] = off *frameSize + OffsetInfo[1];

		PointsPerTrack.clear();
		ceresCost = MotionPrior_Optim_SpatialStructure_Algebraic(Path, All3D, PerCam_UV, PointsPerTrack, currentOffset, ntracks, false, nCams, motionPriorPower, Tscale, ialpha, eps, lamda, APLDCost, false, true);
		MotionPrior_ML_Weighting(PerCam_UV, ntracks, nCams);
		ceresCost = MotionPrior_Optim_SpatialStructure_Geometric(Path, All3D, PerCam_UV, PointsPerTrack, currentOffset, ntracks, false, nCams, motionPriorPower, Tscale, ialpha, eps, lamda, APLDCost, false, true);

		if (silient)
			printf("@off %d (id: %d): C: %.5e Ac: %.5e Pc: %.5e Lc: %.5f Dc: %.5f\n", off, count, ceresCost, APLDCost[0], APLDCost[1], APLDCost[2], APLDCost[3]);
		count++;

		//Clean estimated 3D
		for (int trackID = 0; trackID < ntracks; trackID++)
		{
			int dummy = 0;
			for (int camID = 0; camID < nCams; camID++)
			{
				for (int frameID = 0; frameID < PerCam_UV[camID*ntracks + trackID].size(); frameID++)
				{
					PerCam_UV[camID*ntracks + trackID][frameID].pt3D = Point3d(dummy, dummy, dummy);
					dummy++;
				}
			}
		}

		AllCost[off - LowBound] = ceresCost, AllACost[off - LowBound] = APLDCost[0], AllPCost[off - LowBound] = APLDCost[1], AllLCost[off - LowBound] = APLDCost[2], AllDCost[off - LowBound] = APLDCost[3];
	}

	//Compute minium cost
	Quick_Sort_Double(AllCost, OffsetID, 0, count - 1);

	printf("(%d %d): Min Acost: %.5e, Offset: %.4f (id: %d)\n", SelectedCams[0], SelectedCams[1], AllCost[0], frameSize*OffsetID[0] + OffsetInfo[1], OffsetID[0] - LowBound);
	OffsetInfo[1] += frameSize*OffsetID[0];

	double finalCost = AllCost[0];

	delete[]PerCam_UV, delete[]PerCam_XYZ;
	delete[]OffsetID, delete[]AllCost, delete[]AllACost, delete[]AllPCost, delete[]AllLCost, delete[]AllDCost;

	return finalCost;
}
int LeastActionSyncBruteForce2DTriplet(char *Path, vector<int> &SelectedCams, int startFrame, int stopFrame, int ntracks, vector<double> &OffsetInfo, int LowBound, int UpBound, double frameSize, double lamda, int motionPriorPower)
{
	//Offset is in timestamp format
	const double Tscale = 1000.0, fps = 60, ialpha = 1.0 / fps, eps = 1.0e-6;

	char Fname[200]; FILE *fp = 0;
	const int nCams = 3;

	//Read calib info
	VideoData VideoInfo[3];
	if (ReadVideoDataI(Path, VideoInfo[0], SelectedCams[0], startFrame, stopFrame) == 1)
		return 1;
	if (ReadVideoDataI(Path, VideoInfo[1], SelectedCams[1], startFrame, stopFrame) == 1)
		return 1;
	if (ReadVideoDataI(Path, VideoInfo[2], SelectedCams[2], startFrame, stopFrame) == 1)
		return 1;

	int id, frameID, npts;
	int nframes = max(MaxnFrames, stopFrame);

	double u, v;
	vector<int>VectorCamID, VectorFrameID;
	vector<double> AllError2D;
	ImgPtEle ptEle;
	vector<ImgPtEle> *PerCam_UV = new vector<ImgPtEle>[nCams*ntracks];
	vector<XYZD> *PerCam_XYZ = new vector<XYZD>[nCams], *XYZ = new vector<XYZD>[ntracks];

	//Get 2D info
	for (int camID = 0; camID < nCams; camID++)
	{
		for (int trackID = 0; trackID < ntracks; trackID++)
			PerCam_UV[camID*ntracks + trackID].reserve(stopFrame - startFrame + 1);

		sprintf(Fname, "%s/Track2D/%d.txt", Path, SelectedCams[camID]); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("Cannot open %s\n", Fname);
			return 1;
		}
		for (int trackID = 0; trackID < ntracks; trackID++)
		{
			fscanf(fp, "%d %d ", &id, &npts);
			if (id != trackID)
				printf("Problem at Point %d of Cam %d", id, camID);
			for (int pid = 0; pid < npts; pid++)
			{
				fscanf(fp, "%d %lf %lf ", &frameID, &u, &v);
				if (frameID < 0)
					continue;
				if (!VideoInfo[camID].VideoInfo[frameID].valid)
					continue; //camera not localized

				if (u > 0 && v > 0)
				{
					ptEle.pt2D.x = u, ptEle.pt2D.y = v, ptEle.frameID = frameID;
					LensCorrectionPoint(&ptEle.pt2D, VideoInfo[camID].VideoInfo[frameID].K, VideoInfo[camID].VideoInfo[frameID].distortion);
					PerCam_UV[camID*ntracks + id].push_back(ptEle);
				}
			}
		}
		fclose(fp);
	}


	//Generate Calib Info
	double P[12], AA[6], bb[2], ccT[3], dd[1];
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		//Get ray direction, Q, U, P
		int pcount = 0;
		for (int camID = 0; camID < nCams; camID++)
		{
			for (int frameID = 0; frameID < PerCam_UV[camID*ntracks + trackID].size(); frameID++)
			{
				int RealFrameID = PerCam_UV[camID*ntracks + trackID][frameID].frameID;

				for (int kk = 0; kk < 12; kk++)
				{
					P[kk] = VideoInfo[camID].VideoInfo[RealFrameID].P[kk];
					PerCam_UV[camID*ntracks + trackID][frameID].P[kk] = P[kk];
				}

				for (int kk = 0; kk < 9; kk++)
					PerCam_UV[camID*ntracks + trackID][frameID].K[kk] = VideoInfo[camID].VideoInfo[RealFrameID].K[kk],
					PerCam_UV[camID*ntracks + trackID][frameID].R[kk] = VideoInfo[camID].VideoInfo[RealFrameID].R[kk];

				for (int kk = 0; kk < 3; kk++)
					PerCam_UV[camID*ntracks + trackID][frameID].camcenter[kk] = VideoInfo[camID].VideoInfo[RealFrameID].camCenter[kk];

				//Q, U
				AA[0] = P[0], AA[1] = P[1], AA[2] = P[2], bb[0] = P[3];
				AA[3] = P[4], AA[4] = P[5], AA[5] = P[6], bb[1] = P[7];
				ccT[0] = P[8], ccT[1] = P[9], ccT[2] = P[10], dd[0] = P[11];

				PerCam_UV[camID*ntracks + trackID][frameID].Q[0] = AA[0] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.x*ccT[0],
					PerCam_UV[camID*ntracks + trackID][frameID].Q[1] = AA[1] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.x*ccT[1],
					PerCam_UV[camID*ntracks + trackID][frameID].Q[2] = AA[2] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.x*ccT[2];
				PerCam_UV[camID*ntracks + trackID][frameID].Q[3] = AA[3] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.y*ccT[0],
					PerCam_UV[camID*ntracks + trackID][frameID].Q[4] = AA[4] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.y*ccT[1],
					PerCam_UV[camID*ntracks + trackID][frameID].Q[5] = AA[5] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.y*ccT[2];
				PerCam_UV[camID*ntracks + trackID][frameID].u[0] = dd[0] * PerCam_UV[camID*ntracks + trackID][frameID].pt2D.x - bb[0],
					PerCam_UV[camID*ntracks + trackID][frameID].u[1] = dd[0] * PerCam_UV[camID*ntracks + trackID][frameID].pt2D.y - bb[1];

				PerCam_UV[camID*ntracks + trackID][frameID].pt3D = Point3d(pcount, pcount, pcount);
				pcount++;
			}
		}
	}

	//Initialize data for optim
	vector<int> PointsPerTrack;
	vector<int *> PerTrackFrameID(ntracks);
	vector<double*> All3D(ntracks);
	int ntimeinstances, maxntimeinstances = 0;
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		ntimeinstances = 0;
		for (int camID = 0; camID < nCams; camID++)
			ntimeinstances += PerCam_UV[camID*ntracks + trackID].size();

		if (maxntimeinstances < ntimeinstances)
			maxntimeinstances = ntimeinstances;

		PerTrackFrameID[trackID] = new int[ntimeinstances];
		All3D[trackID] = new double[3 * ntimeinstances];
	}

	//Start sliding
	double currentOffset[3], APLDCost[4], ceresCost;
	int *OffsetID = new int[(UpBound - LowBound + 1)*(UpBound - LowBound + 1)];
	Point2d *OffsetValue = new Point2d[(UpBound - LowBound + 1)*(UpBound - LowBound + 1)];
	double *AllCost = new double[(UpBound - LowBound + 1)*(UpBound - LowBound + 1)];
	double *AllACost = new double[(UpBound - LowBound + 1)*(UpBound - LowBound + 1)];
	double *AllPCost = new double[(UpBound - LowBound + 1)*(UpBound - LowBound + 1)];
	double *AllLCost = new double[(UpBound - LowBound + 1)*(UpBound - LowBound + 1)];
	double *AllDCost = new double[(UpBound - LowBound + 1)*(UpBound - LowBound + 1)];
	vector<double> VTimeStamp; VTimeStamp.reserve(maxntimeinstances);
	vector<Point3d> VTrajectory3D; VTrajectory3D.reserve(maxntimeinstances);

	int count = 0;
	//for (int lamdaID = 1; lamdaID <= 100; lamdaID++)
	{
		//lamda = lamdaID*0.01;
		for (int off2 = LowBound; off2 <= UpBound; off2++)
		{
			for (int off1 = LowBound; off1 <= UpBound; off1++)
			{
				OffsetID[count] = count;
				OffsetValue[count] = Point2d(off1, off2);
				currentOffset[0] = OffsetInfo[0], currentOffset[1] = off1*frameSize + OffsetInfo[1], currentOffset[2] = off2*frameSize + OffsetInfo[2];

				PointsPerTrack.clear();
				MotionPrior_Optim_SpatialStructure_Algebraic(Path, All3D, PerCam_UV, PointsPerTrack, currentOffset, ntracks, false, nCams, motionPriorPower, Tscale, ialpha, eps, lamda, APLDCost, false, true);
				MotionPrior_ML_Weighting(PerCam_UV, ntracks, nCams);
				ceresCost = MotionPrior_Optim_SpatialStructure_Geometric(Path, All3D, PerCam_UV, PointsPerTrack, currentOffset, ntracks, false, nCams, motionPriorPower, Tscale, ialpha, eps, lamda, APLDCost, false, true);

				/*for (int trackID = 0; trackID < ntracks; trackID++)
				{
				sprintf(Fname, "%s/ATrack_%d_%d.txt", Path, trackID, count);  FILE *fp = fopen(Fname, "w+");
				for (int camID = 0; camID < nCams; camID++)
				for (int fid = 0; fid < PerCam_UV[camID*ntracks + trackID].size(); fid++)
				fprintf(fp, "%.4f %.4f %.4f %.2f %d %d\n", PerCam_UV[camID*ntracks + trackID][fid].pt3D.x, PerCam_UV[camID*ntracks + trackID][fid].pt3D.y,
				PerCam_UV[camID*ntracks + trackID][fid].pt3D.z, 1.0*OffsetInfo[camID] * ialpha*Tscale + 1.0*PerCam_UV[camID*ntracks + trackID][fid].frameID * ialpha*Tscale, camID, PerCam_UV[camID*ntracks + trackID][fid].frameID);
				fclose(fp);
				}*/

				printf("@(%d, %d) (id: %d): C: %.5e Ac: %.5e Pc: %.5e Lc: %.5f Dc: %.5f\n", off1, off2, count, ceresCost, APLDCost[0], APLDCost[1], APLDCost[2], APLDCost[3]);
				count++;

				//Clean estimated 3D
				for (int trackID = 0; trackID < ntracks; trackID++)
				{
					int pcount = 0;
					for (int camID = 0; camID < nCams; camID++)
					{
						for (int frameID = 0; frameID < PerCam_UV[camID*ntracks + trackID].size(); frameID++)
						{
							PerCam_UV[camID*ntracks + trackID][frameID].pt3D = Point3d(pcount, pcount, pcount);
							pcount++;
						}
					}
				}

				AllCost[(off1 - LowBound) + (off2 - LowBound)*(UpBound - LowBound + 1)] = ceresCost,
					AllACost[(off1 - LowBound) + (off2 - LowBound)*(UpBound - LowBound + 1)] = APLDCost[0],
					AllPCost[(off1 - LowBound) + (off2 - LowBound)*(UpBound - LowBound + 1)] = APLDCost[1],
					AllLCost[(off1 - LowBound) + (off2 - LowBound)*(UpBound - LowBound + 1)] = APLDCost[2],
					AllDCost[(off1 - LowBound) + (off2 - LowBound)*(UpBound - LowBound + 1)] = APLDCost[3];
			}
		}
	}

	/*fp = fopen("C:/temp/cost.txt", "w+");
	for (int off2 = LowBound; off2 <= UpBound; off2++)
	{
	for (int off1 = LowBound; off1 <= UpBound; off1++)
	{
	int id = off1 - LowBound + (off2 - LowBound)*(UpBound - LowBound + 1);
	fprintf(fp, "%.3f %.3f %.5e %.5e %.5e %.5e %.5e\n", frameSize*OffsetValue[id].x + OffsetInfo[1], frameSize*OffsetValue[id].y + OffsetInfo[2], AllCost[id], AllACost[id], AllPCost[id], AllLCost[id], AllDCost[id]);
	}
	fprintf(fp, "\n");
	}
	fclose(fp);*/

	Quick_Sort_Double(AllCost, OffsetID, 0, count - 1);
	printf("Min Acost of %.5e with Offset of (%.4f, %.4f) (id: %d)\n", AllCost[0], frameSize*OffsetValue[OffsetID[0]].x + OffsetInfo[1], frameSize*OffsetValue[OffsetID[0]].y + OffsetInfo[2], OffsetID[0]);
	OffsetInfo[0] = 0, OffsetInfo[1] += frameSize*OffsetValue[OffsetID[0]].x, OffsetInfo[2] += frameSize*OffsetValue[OffsetID[0]].y;

	delete[]PerCam_UV, delete[]PerCam_XYZ;
	delete[]OffsetID, delete[]AllCost, delete[]AllACost, delete[]AllPCost, delete[]AllLCost, delete[]AllDCost;

	return 0;
}
int IncrementalLeastActionSyncDiscreteContinous2D(char *Path, vector<int> &SelectedCams, int startFrame, int stopFrame, int npts, vector<double> &OffsetInfo, int LowBound, int UpBound, double frameSize, double lamda, int motionPriorPower, bool RefineConsiderOrdering = true)
{
	//Offset is in timestamp format
	char Fname[200]; FILE *fp = 0;
	const double Tscale = 1000.0, fps = 60, ialpha = 1.0 / fps, eps = 1.0e-9;

	//Read calib info
	int nCams = (int)SelectedCams.size();
	VideoData *VideoInfo = new VideoData[nCams];
	for (int camID = 0; camID < nCams; camID++)
		if (ReadVideoDataI(Path, VideoInfo[camID], SelectedCams[camID], startFrame, stopFrame) == 1)
			return 1;

	int frameID, id, nf;
	int nframes = max(MaxnFrames, stopFrame);

	double u, v;
	vector<int>VectorCamID, VectorFrameID;
	vector<double> AllError2D;
	ImgPtEle ptEle;
	vector<ImgPtEle> *PerCam_UV = new vector<ImgPtEle>[nCams*npts];
	vector<XYZD> *PerCam_XYZ = new vector<XYZD>[nCams], *XYZ = new vector<XYZD>[npts];

	for (int camID = 0; camID < nCams; camID++)
	{
		for (int trackID = 0; trackID < npts; trackID++)
			PerCam_UV[camID*npts + trackID].reserve(stopFrame - startFrame + 1);

		sprintf(Fname, "%s/Track2D/%d.txt", Path, SelectedCams[camID]); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("Cannot load %s\n", Fname);
			return 1;
		}
		for (int trackID = 0; trackID < npts; trackID++)
		{
			fscanf(fp, "%d %d ", &id, &nf);
			if (id != trackID)
				printf("Problem at Point %d of Cam %d", id, camID);
			for (int fid = 0; fid < nf; fid++)
			{
				fscanf(fp, "%d %lf %lf ", &frameID, &u, &v);
				if (frameID < 0)
					continue;
				if (!VideoInfo[camID].VideoInfo[frameID].valid)
					continue; //camera not localized

				if (u > 0 && v > 0)
				{
					ptEle.pt2D.x = u, ptEle.pt2D.y = v, ptEle.frameID = frameID, ptEle.imHeight = VideoInfo[camID].VideoInfo[frameID].height, ptEle.imWidth = VideoInfo[camID].VideoInfo[frameID].width;
					LensCorrectionPoint(&ptEle.pt2D, VideoInfo[camID].VideoInfo[frameID].K, VideoInfo[camID].VideoInfo[frameID].distortion);
					PerCam_UV[camID*npts + id].push_back(ptEle);
				}
			}
		}
		fclose(fp);
	}

	double P[12], AA[6], bb[2], ccT[3], dd[1];
	for (int trackID = 0; trackID < npts; trackID++)
	{
		int count = 0;
		for (int camID = 0; camID < nCams; camID++)
		{
			for (int frameID = 0; frameID < PerCam_UV[camID*npts + trackID].size(); frameID++)
			{
				int RealFrameID = PerCam_UV[camID*npts + trackID][frameID].frameID;

				for (int kk = 0; kk < 12; kk++)
				{
					P[kk] = VideoInfo[camID].VideoInfo[RealFrameID].P[kk];
					PerCam_UV[camID*npts + trackID][frameID].P[kk] = P[kk];
				}

				for (int kk = 0; kk < 9; kk++)
					PerCam_UV[camID*npts + trackID][frameID].K[kk] = VideoInfo[camID].VideoInfo[RealFrameID].K[kk],
					PerCam_UV[camID*npts + trackID][frameID].R[kk] = VideoInfo[camID].VideoInfo[RealFrameID].R[kk];

				for (int kk = 0; kk < 3; kk++)
					PerCam_UV[camID*npts + trackID][frameID].camcenter[kk] = VideoInfo[camID].VideoInfo[RealFrameID].camCenter[kk];

				//Q, U
				AA[0] = P[0], AA[1] = P[1], AA[2] = P[2], bb[0] = P[3];
				AA[3] = P[4], AA[4] = P[5], AA[5] = P[6], bb[1] = P[7];
				ccT[0] = P[8], ccT[1] = P[9], ccT[2] = P[10], dd[0] = P[11];

				PerCam_UV[camID*npts + trackID][frameID].Q[0] = AA[0] - PerCam_UV[camID*npts + trackID][frameID].pt2D.x*ccT[0],
					PerCam_UV[camID*npts + trackID][frameID].Q[1] = AA[1] - PerCam_UV[camID*npts + trackID][frameID].pt2D.x*ccT[1],
					PerCam_UV[camID*npts + trackID][frameID].Q[2] = AA[2] - PerCam_UV[camID*npts + trackID][frameID].pt2D.x*ccT[2];
				PerCam_UV[camID*npts + trackID][frameID].Q[3] = AA[3] - PerCam_UV[camID*npts + trackID][frameID].pt2D.y*ccT[0],
					PerCam_UV[camID*npts + trackID][frameID].Q[4] = AA[4] - PerCam_UV[camID*npts + trackID][frameID].pt2D.y*ccT[1],
					PerCam_UV[camID*npts + trackID][frameID].Q[5] = AA[5] - PerCam_UV[camID*npts + trackID][frameID].pt2D.y*ccT[2];
				PerCam_UV[camID*npts + trackID][frameID].u[0] = dd[0] * PerCam_UV[camID*npts + trackID][frameID].pt2D.x - bb[0],
					PerCam_UV[camID*npts + trackID][frameID].u[1] = dd[0] * PerCam_UV[camID*npts + trackID][frameID].pt2D.y - bb[1];

				PerCam_UV[camID*npts + trackID][frameID].pt3D = Point3d(count, count, count);
				count++;
			}
		}
	}

	//Initialize data for optim
	vector<int> PointsPerTrack;
	vector<int *> PerTrackFrameID(npts);
	vector<double*> All3D(npts);
	int ntimeinstances, maxntimeinstances = 0;
	for (int trackID = 0; trackID < npts; trackID++)
	{
		ntimeinstances = 0;
		for (int camID = 0; camID < nCams; camID++)
			ntimeinstances += PerCam_UV[camID*npts + trackID].size();

		if (maxntimeinstances < ntimeinstances)
			maxntimeinstances = ntimeinstances;

		PerTrackFrameID[trackID] = new int[ntimeinstances];
		All3D[trackID] = new double[3 * ntimeinstances];
	}

	//Step 1: Start sliding to make sure that you are at frame level accurate
	double APLDCost[4];
	double currentOffset[MaxnCams];
	vector<double> VTimeStamp; VTimeStamp.reserve(maxntimeinstances);
	vector<Point3d> VTrajectory3D; VTrajectory3D.reserve(maxntimeinstances);

	printf("ST estimation for %d camaras ( ", nCams);
	for (int ii = 0; ii < nCams; ii++)
		printf("%d ", SelectedCams[ii]);
	printf("): \n");

	int NotSuccess = 0;
	if (RefineConsiderOrdering)
	{
		//Step 1: brute force search for the best time stamp. May not work well enough if the distrubution of the correct time stamps are dramatically skew
		if (frameSize > FLT_EPSILON)
		{
			int *OffsetID = new int[UpBound - LowBound + 1];
			double*AllACost = new double[UpBound - LowBound + 1],
				*AllPCost = new double[UpBound - LowBound + 1];

			for (int ii = 0; ii < nCams; ii++)
				currentOffset[ii] = OffsetInfo[ii];

			int count = 0;
			for (int off = LowBound; off <= UpBound; off++)
			{
				//Clean estimated 3D for the next trial
				for (int trackID = 0; trackID < npts; trackID++)
					for (int camID = 0; camID < nCams; camID++)
						for (int frameID = 0; frameID < PerCam_UV[camID*npts + trackID].size(); frameID++)
							PerCam_UV[camID*npts + trackID][frameID].pt3D = Point3d(gaussian_noise(0.0, 1), gaussian_noise(0.0, 1), gaussian_noise(0.0, 1));

				OffsetID[count] = off;
				currentOffset[nCams - 1] = off *frameSize + OffsetInfo[nCams - 1];

				PointsPerTrack.clear();
				printf("Trial %d (%.2f): ", off, off *frameSize + OffsetInfo[nCams - 1]);
				MotionPrior_Optim_SpatialStructure_Algebraic(Path, All3D, PerCam_UV, PointsPerTrack, currentOffset, npts, false, nCams, motionPriorPower, Tscale, ialpha, eps, lamda, APLDCost, false, true);
				printf("Ac: %.5e Pc: %.5e Lc: %.5f Dc: %.5f\n", APLDCost[0], APLDCost[1], APLDCost[2], APLDCost[3]);

				AllACost[count] = APLDCost[0], AllPCost[count] = APLDCost[1];
				count++;
			}

			if (count > 1)
			{
				Quick_Sort_Double(AllACost, OffsetID, 0, count - 1);
				OffsetInfo[nCams - 1] += frameSize*OffsetID[0];
				printf("@Brute-force: \t Least action cost: %.5e \t Offset: %.4f\n\n", AllACost[0], OffsetInfo[nCams - 1]);
			}
		}


		//Step 2: insert the new camera to the pre-order camears since brute force sliding may not guarantee correct ordering
		int naddedCams = nCams - 1;
		double subframeRemander[MaxnCams], subframeRemander2[MaxnCams];
		int subframeRemanderID[MaxnCams], subframeRemanderID2[MaxnCams];
		for (int ii = 0; ii < naddedCams; ii++)
		{
			subframeRemanderID[ii] = ii;
			subframeRemander[ii] = OffsetInfo[ii] - floor(OffsetInfo[ii]);
		}
		Quick_Sort_Double(subframeRemander, subframeRemanderID, 0, naddedCams - 1);

		double subframeSlots[MaxnCams];
		for (int ii = 0; ii < naddedCams - 1; ii++)
			subframeSlots[ii] = 0.5*(subframeRemander[ii] + subframeRemander[ii + 1]);
		subframeSlots[naddedCams - 1] = 0.5*(subframeRemander[naddedCams - 1] + subframeRemander[0] + 1.0);

		int bestID = 0;
		double CeresCost, bestCeresCost = 9e20, bestAcost = 9e20;
		double *BestOffset = new double[nCams];
		vector<int> iTimeOrdering, fTimeOrdering;
		int NotFlippedTimes = 0;
		for (int kk = LowBound; kk <= UpBound; kk++)
		{
			for (int ii = 0; ii < naddedCams; ii++)//try different slots
			{
				for (int jj = 0; jj < naddedCams; jj++)
					currentOffset[jj] = OffsetInfo[jj];
				currentOffset[nCams - 1] = floor(OffsetInfo[nCams - 1]) + subframeSlots[ii] + kk;

				for (int jj = 0; jj < nCams; jj++)
				{
					subframeRemanderID[jj] = jj;
					subframeRemander[jj] = currentOffset[jj] - floor(currentOffset[jj]);
				}
				Quick_Sort_Double(subframeRemander, subframeRemanderID, 0, nCams - 1);

				printf("Trial %d/%d (%.4f): ", ii + (kk - LowBound)*naddedCams, (UpBound - LowBound + 1)*naddedCams, currentOffset[nCams - 1]);
				MotionPrior_Optim_SpatialStructure_Algebraic(Path, All3D, PerCam_UV, PointsPerTrack, currentOffset, npts, false, nCams, motionPriorPower, Tscale, ialpha, eps, lamda, APLDCost, false, true);
				MotionPrior_ML_Weighting(PerCam_UV, npts, nCams);
				CeresCost = MotionPrior_Optim_ST_Geometric(Path, All3D, PerCam_UV, PointsPerTrack, currentOffset, npts, true, nCams, motionPriorPower, Tscale, ialpha, eps, lamda, APLDCost, false, true);

				printf("Cost: %.5e Ac: %.5e Pc: %.5e Lc: %.5f Dc: %.5f\nOffsets:", CeresCost, APLDCost[0], APLDCost[1], APLDCost[2], APLDCost[3]);
				for (int jj = 0; jj < nCams; jj++)
					printf("%.4f ", currentOffset[jj]);
				printf("\n");

				//check if time order has been flipped
				for (int jj = 0; jj < nCams; jj++)
				{
					subframeRemanderID2[jj] = jj;
					subframeRemander2[jj] = currentOffset[jj] - floor(currentOffset[jj]);
				}
				Quick_Sort_Double(subframeRemander2, subframeRemanderID2, 0, nCams - 1);

				bool flipped = false;
				for (int jj = 0; jj < nCams; jj++)
				{
					if (subframeRemanderID[jj] != subframeRemanderID2[jj])
					{
						flipped = true;
						break;
					}
				}
				if (flipped)
				{
					printf("Flipping occurs!\n");
					continue;
				}
				else
					NotFlippedTimes++;

				if (bestCeresCost > CeresCost)
				{
					bestID = ii, bestCeresCost = CeresCost;
					for (int jj = 0; jj < nCams; jj++)
						BestOffset[jj] = currentOffset[jj];
				}

				//Clean estimated 3D
				for (int trackID = 0; trackID < npts; trackID++)
					for (int camID = 0; camID < nCams; camID++)
						for (int frameID = 0; frameID < PerCam_UV[camID*npts + trackID].size(); frameID++)
							PerCam_UV[camID*npts + trackID][frameID].pt3D = Point3d(gaussian_noise(0.0, 1), gaussian_noise(0.0, 1), gaussian_noise(0.0, 1));
			}
		}
		if (NotFlippedTimes == 0)
			NotSuccess = 1;

		if (bestCeresCost < 9e20)
			for (int jj = 0; jj < nCams; jj++)
				OffsetInfo[jj] = BestOffset[jj];
	}
	else
	{
		for (int ii = 0; ii < nCams; ii++)
			currentOffset[ii] = OffsetInfo[ii];

		MotionPrior_Optim_SpatialStructure_Algebraic(Path, All3D, PerCam_UV, PointsPerTrack, currentOffset, npts, false, nCams, motionPriorPower, Tscale, ialpha, eps, lamda, APLDCost, false, true);
		MotionPrior_ML_Weighting(PerCam_UV, npts, nCams);
		MotionPrior_Optim_ST_Geometric(Path, All3D, PerCam_UV, PointsPerTrack, currentOffset, npts, true, nCams, motionPriorPower, Tscale, ialpha, eps, lamda, APLDCost, false, true);

		for (int jj = 0; jj < nCams; jj++)
			OffsetInfo[jj] = currentOffset[jj];
	}

	if (NotSuccess == 0)
	{
		printf("Final temporal estimation: ");
		for (int ii = 0; ii < nCams; ii++)
			printf("%.4f ", OffsetInfo[ii]);
		printf("\n\n");
	}
	else
		printf("Flipping occurs for all trials. Start to shuffle the ordering stack and retry.\n");

	return NotSuccess;
}
int TrajectoryTriangulation(char *Path, vector<int> &SelectedCams, vector<double> &TimeStampInfoVector, int npts, int startFrame, int stopFrame, double lamda, int motionPriorPower)
{
	char Fname[200]; FILE *fp = 0;
	//const double Tscale = 1000.0, fps = 60.0, ialpha = 1.0 / fps, eps = 1.0e-6;
	const double Tscale = 10.0, fps = 1.0, ialpha = 1.0 / fps, eps = 1.0e-6;

	//Read calib info
	int nCams = (int)SelectedCams.size();
	VideoData *VideoInfo = new VideoData[nCams];
	for (int camID = 0; camID < nCams; camID++)
		if (ReadVideoDataI(Path, VideoInfo[camID], SelectedCams[camID], startFrame, stopFrame) == 1)
			return 1;

	int frameID, id, nf;
	int nframes = max(MaxnFrames, stopFrame);

	double u, v;
	vector<int>VectorCamID, VectorFrameID;
	vector<double> AllError2D;
	ImgPtEle ptEle;
	vector<ImgPtEle> *PerCam_UV = new vector<ImgPtEle>[nCams*npts];

	for (int camID = 0; camID < nCams; camID++)
	{
		for (int pid = 0; pid < npts; pid++)
			PerCam_UV[camID*npts + pid].reserve(stopFrame - startFrame + 1);

		sprintf(Fname, "%s/Track2D/%d.txt", Path, SelectedCams[camID]); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("Cannot load %s\n", Fname);
			return 1;
		}
		for (int pid = 0; pid < npts; pid++)
		{
			fscanf(fp, "%d %d ", &id, &nf);
			if (id != pid)
				printf("Problem at Point %d of Cam %d", id, camID);
			for (int fid = 0; fid < nf; fid++)
			{
				fscanf(fp, "%d %lf %lf ", &frameID, &u, &v);
				if (frameID < 0)
					continue;
				if (!VideoInfo[camID].VideoInfo[frameID].valid)
					continue; //camera not localized

				if (u > 0 && v > 0)
				{
					ptEle.pt2D.x = u, ptEle.pt2D.y = v, ptEle.viewID = camID, ptEle.frameID = frameID, ptEle.imWidth = VideoInfo[camID].VideoInfo[frameID].width, ptEle.imHeight = VideoInfo[camID].VideoInfo[frameID].height;
					LensCorrectionPoint(&ptEle.pt2D, VideoInfo[camID].VideoInfo[frameID].K, VideoInfo[camID].VideoInfo[frameID].distortion);
					PerCam_UV[camID*npts + id].push_back(ptEle);
				}
			}
		}
		fclose(fp);
	}

	//Generate Calib Info
	double P[12], AA[6], bb[2], ccT[3], dd[1];
	for (int pid = 0; pid < npts; pid++)
	{
		//Get ray direction, Q, U, P
		for (int camID = 0; camID < nCams; camID++)
		{
			for (int frameID = 0; frameID < PerCam_UV[camID*npts + pid].size(); frameID++)
			{
				int RealFrameID = PerCam_UV[camID*npts + pid][frameID].frameID;

				for (int kk = 0; kk < 12; kk++)
				{
					P[kk] = VideoInfo[camID].VideoInfo[RealFrameID].P[kk];
					PerCam_UV[camID*npts + pid][frameID].P[kk] = P[kk];
				}

				for (int kk = 0; kk < 9; kk++)
					PerCam_UV[camID*npts + pid][frameID].K[kk] = VideoInfo[camID].VideoInfo[RealFrameID].K[kk],
					PerCam_UV[camID*npts + pid][frameID].R[kk] = VideoInfo[camID].VideoInfo[RealFrameID].R[kk];

				for (int kk = 0; kk < 3; kk++)
					PerCam_UV[camID*npts + pid][frameID].camcenter[kk] = VideoInfo[camID].VideoInfo[RealFrameID].camCenter[kk];

				//Q, U
				AA[0] = P[0], AA[1] = P[1], AA[2] = P[2], bb[0] = P[3];
				AA[3] = P[4], AA[4] = P[5], AA[5] = P[6], bb[1] = P[7];
				ccT[0] = P[8], ccT[1] = P[9], ccT[2] = P[10], dd[0] = P[11];

				PerCam_UV[camID*npts + pid][frameID].Q[0] = AA[0] - PerCam_UV[camID*npts + pid][frameID].pt2D.x*ccT[0],
					PerCam_UV[camID*npts + pid][frameID].Q[1] = AA[1] - PerCam_UV[camID*npts + pid][frameID].pt2D.x*ccT[1],
					PerCam_UV[camID*npts + pid][frameID].Q[2] = AA[2] - PerCam_UV[camID*npts + pid][frameID].pt2D.x*ccT[2];
				PerCam_UV[camID*npts + pid][frameID].Q[3] = AA[3] - PerCam_UV[camID*npts + pid][frameID].pt2D.y*ccT[0],
					PerCam_UV[camID*npts + pid][frameID].Q[4] = AA[4] - PerCam_UV[camID*npts + pid][frameID].pt2D.y*ccT[1],
					PerCam_UV[camID*npts + pid][frameID].Q[5] = AA[5] - PerCam_UV[camID*npts + pid][frameID].pt2D.y*ccT[2];
				PerCam_UV[camID*npts + pid][frameID].u[0] = dd[0] * PerCam_UV[camID*npts + pid][frameID].pt2D.x - bb[0],
					PerCam_UV[camID*npts + pid][frameID].u[1] = dd[0] * PerCam_UV[camID*npts + pid][frameID].pt2D.y - bb[1];

				double stdA = 100.0;//Interestingly, Ceres does not work if all the input are the same-->need some random perturbation. 
				PerCam_UV[camID*npts + pid][frameID].pt3D = Point3d(gaussian_noise(0.0, stdA), gaussian_noise(0.0, stdA), gaussian_noise(0.0, stdA));
			}
		}
	}

	//Initialize data for optim
	vector<int> PointsPerTrack;
	vector<int *> PerTrackFrameID(npts);
	vector<double*> All3D(npts);
	int ntimeinstances, maxntimeinstances = 0;
	for (int pid = 0; pid < npts; pid++)
	{
		ntimeinstances = 0;
		for (int camID = 0; camID < nCams; camID++)
			ntimeinstances += PerCam_UV[camID*npts + pid].size();

		if (maxntimeinstances < ntimeinstances)
			maxntimeinstances = ntimeinstances;

		PerTrackFrameID[pid] = new int[ntimeinstances];
		All3D[pid] = new double[3 * ntimeinstances];
	}

	double APLDCost[4];
	vector<double> VTimeStamp; VTimeStamp.reserve(maxntimeinstances);
	vector<Point3d> VTrajectory3D; VTrajectory3D.reserve(maxntimeinstances);

	double *OffsetInfo = new double[nCams];
	for (int ii = 0; ii < nCams; ii++)
		OffsetInfo[ii] = TimeStampInfoVector[ii];

	//MotionPrior_Optim_SpatialStructure_NoSimulatenousPoints(Path, All3D, PerCam_UV, PointsPerTrack, OffsetInfo, npts, false, nCams, Tscale, ialpha, eps, lamda, APLDCost, false, false);
	printf("Algrebraic triangulation:\n");
	MotionPrior_Optim_SpatialStructure_Algebraic(Path, All3D, PerCam_UV, PointsPerTrack, OffsetInfo, npts, false, nCams, motionPriorPower, Tscale, ialpha, eps, lamda, APLDCost, false, false);

	printf("\nGeometric triangulation:\n");
	MotionPrior_ML_Weighting(PerCam_UV, npts, nCams);
	MotionPrior_Optim_SpatialStructure_Geometric(Path, All3D, PerCam_UV, PointsPerTrack, OffsetInfo, npts, false, nCams, motionPriorPower, Tscale, ialpha, eps, lamda, APLDCost, false, false);

	for (int pid = 0; pid < npts; pid++)
	{
		sprintf(Fname, "%s/OptimizedRaw_Track_%d.txt", Path, pid);  FILE *fp = fopen(Fname, "w+");
		for (int camID = 0; camID < nCams; camID++)
			for (int fid = 0; fid < PerCam_UV[camID*npts + pid].size(); fid++)
				fprintf(fp, "%.8f %.8f %.8f %.4f %d %d\n", PerCam_UV[camID*npts + pid][fid].pt3D.x, PerCam_UV[camID*npts + pid][fid].pt3D.y, PerCam_UV[camID*npts + pid][fid].pt3D.z,
				1.0*OffsetInfo[camID] * ialpha*Tscale + 1.0*PerCam_UV[camID*npts + pid][fid].frameID * ialpha*Tscale, SelectedCams[camID], PerCam_UV[camID*npts + pid][fid].frameID);
		fclose(fp);
	}

	delete[]PerCam_UV, delete[]OffsetInfo;
	return 0;
}

int EvaluateAllPairCost(char *Path, int nCams, int nTracks, int startFrame, int stopFrame, int SearchRange, double SearchStep, double lamda, int motionPriorPower, double *InitialOffset)
{
	char Fname[200];
	int  totalPts;
	double cost;
	vector<int>Pair(2);
	vector<double> PairOffset(2), baseline;

	//Base on cameras' baseline
	VideoData *VideoIInfo = new VideoData[nCams];
	for (int ii = 0; ii < nCams; ii++)
	{
		if (ReadVideoDataI(Path, VideoIInfo[ii], ii, startFrame, stopFrame) == 1)
		{
			abort();
		}
	}

	printf("Motion prior sync:\n");
	sprintf(Fname, "%s/PairwiseCost.txt", Path); FILE *fp = fopen(Fname, "w+");
	for (int ii = 0; ii < nCams - 1; ii++)
	{
		for (int jj = ii + 1; jj < nCams; jj++)
		{
			Pair[0] = ii, Pair[1] = jj;
			PairOffset[0] = InitialOffset[ii], PairOffset[1] = InitialOffset[jj];

			//Base on Motion prior
			cost = LeastActionSyncBruteForce2DStereo(Path, Pair, startFrame, stopFrame, nTracks, PairOffset, -SearchRange, SearchRange, SearchStep, lamda, motionPriorPower, totalPts);

			for (int fid1 = startFrame; fid1 <= stopFrame; fid1++)
			{
				int fid2 = fid1 - (int)(InitialOffset[ii] - InitialOffset[jj] + 0.5);
				if (fid2 < 0 || fid2>stopFrame)
					continue;
				if (VideoIInfo[ii].VideoInfo[fid1].valid && VideoIInfo[jj].VideoInfo[fid2].valid)
					baseline.push_back(Distance3D(VideoIInfo[ii].VideoInfo[fid1].camCenter, VideoIInfo[jj].VideoInfo[fid2].camCenter));
			}
			double avgBaseline = MeanArray(baseline);

			fprintf(fp, "%d %d %d %.3e %.8e %.3f\n", Pair[0], Pair[1], totalPts, avgBaseline, cost, PairOffset[1] - PairOffset[0]);
			baseline.clear();
		}
	}
	fclose(fp);

	printf("\n");
	return 0;
}
int DetermineCameraOrderingForGreedySTBA(char *Path, char *PairwiseSyncFilename, int nCams, vector<int>&CameraOrder, vector<double> &OffsetInfo)
{
	//Offset info is corresponded to the camera order
	typedef boost::adjacency_list < boost::vecS, boost::vecS, boost::undirectedS, boost::property<boost::vertex_distance_t, int>, boost::property < boost::edge_weight_t, double > > Graph;
	typedef boost::graph_traits < Graph >::edge_descriptor Edge;
	typedef boost::graph_traits < Graph >::vertex_descriptor Vertex;
	typedef std::pair < int, int >E;

	int v1, v2, nvalidPts;
	double baseline, TrajCost, offset;
	char Fname[200];
	int *nValidPts = new int[nCams*nCams];
	double *TimeOffset = new double[nCams*nCams],
		*BaseLine = new double[nCams*nCams],
		*Traj3dCost = new double[nCams*nCams];
	for (int ii = 0; ii < nCams*nCams; ii++)
		TimeOffset[ii] = 0, BaseLine[ii] = 0, Traj3dCost[ii] = 0;

	sprintf(Fname, "%s/%s.txt", Path, PairwiseSyncFilename);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot open %s\n", Fname);
		return 1;
	}
	while (fscanf(fp, "%d %d %d %lf %lf %lf ", &v1, &v2, &nvalidPts, &baseline, &TrajCost, &offset) != EOF)
		TimeOffset[v1 + v2*nCams] = offset, TimeOffset[v2 + v1*nCams] = offset,
		nValidPts[v1 + v2*nCams] = nvalidPts, nValidPts[v2 + v1*nCams] = nvalidPts,
		BaseLine[v1 + v2*nCams] = baseline, BaseLine[v2 + v1*nCams] = baseline,
		Traj3dCost[v1 + v2*nCams] = TrajCost, Traj3dCost[v2 + v1*nCams] = TrajCost;
	fclose(fp);

#ifdef ENABLE_DEBUG_FLAG
	sprintf(Fname, "%s/timeConstrantoffset.txt", Path);	fp = fopen(Fname, "w+");
	for (int kk = 0; kk < nCams; kk++)
	{
		for (int ll = 0; ll < nCams; ll++)
			fprintf(fp, "%.4f ", TimeOffset[kk + ll*nCams]);
		fprintf(fp, "\n");
	}
	fclose(fp);
#endif

	//Form edges weight based on the consistency of the triplet
	int num_nodes = nCams, nedges = nCams*(nCams - 1) / 2;
	E *edges = new E[nedges];
	double *weightTable = new double[nCams*nCams];
	double *weights = new double[nedges];
	for (int ii = 0; ii < nCams*nCams; ii++)
		weightTable[ii] = 0;

	int count = 0;
	for (int kk = 0; kk < nCams - 1; kk++)
	{
		for (int ll = kk + 1; ll < nCams; ll++)
		{
			edges[count] = E(kk, ll), weights[count] = 0.0;
			//Consistency_score_kl = sum_j(Offset_kj+Offset_jl);
			for (int jj = 0; jj < nCams; jj++)
			{
				if (jj == ll || jj == kk)
					continue;
				if (jj >= ll) //kl = kj-lj
					weights[count] += abs(TimeOffset[kk + jj*nCams] - TimeOffset[ll + jj*nCams] - TimeOffset[kk + ll*nCams]);
				else if (jj <= kk) //kl = -jk + jl
					weights[count] += abs(-TimeOffset[jj + kk*nCams] + TimeOffset[jj + ll*nCams] - TimeOffset[kk + ll*nCams]);
				else //kl = kj+jl
					weights[count] += abs(TimeOffset[kk + jj*nCams] + TimeOffset[jj + ll*nCams] - TimeOffset[kk + ll*nCams]);
			}

			//Weight them by the # visible points along all trajectories and the average baseline between cameras
			weights[count] = weights[count] / (BaseLine[kk + ll*nCams] * nValidPts[kk + ll*nCams] + DBL_EPSILON);
			weightTable[kk + ll*nCams] = weights[count], weightTable[ll + kk*nCams] = weights[count];
			count++;
		}
	}

#ifdef ENABLE_DEBUG_FLAG
	sprintf(Fname, "%s/weightTable.txt", Path);	fp = fopen(Fname, "w+");
	for (int kk = 0; kk < nCams; kk++)
	{
		for (int ll = 0; ll < nCams; ll++)
			fprintf(fp, "%.4e ", weightTable[kk + ll*nCams]);
		fprintf(fp, "\n");
	}
	fclose(fp);
#endif

	//Estimate incremental camera order by Kruskal MST 
	Graph g(edges, edges + sizeof(E)*nedges / sizeof(E), weights, num_nodes);
	boost::property_map<Graph, boost::edge_weight_t>::type weightmap = get(boost::edge_weight, g);
	std::vector < Edge > spanning_tree;

	boost::kruskal_minimum_spanning_tree(g, std::back_inserter(spanning_tree));

	cout << "Print the edges in the MST:" << endl;
	for (vector < Edge >::iterator ei = spanning_tree.begin(); ei != spanning_tree.end(); ei++)
		cout << source(*ei, g) << " <--> " << target(*ei, g) << " with weight of " << weightmap[*ei] << endl;


	//Store the ordering and subframe offset info. Note that source id is always smaller than target id
	int RootCamera = spanning_tree[0].m_source;
	CameraOrder.push_back(RootCamera); OffsetInfo.push_back(0.0);
	for (int ii = 0; ii < spanning_tree.size(); ii++)
	{
		bool added = false;
		int cam1 = spanning_tree[ii].m_source, cam2 = spanning_tree[ii].m_target;
		for (int jj = 0; jj < (int)CameraOrder.size(); jj++)
		{
			if (CameraOrder[jj] == cam1)
			{
				added = true;
				break;
			}
		}

		if (!added)
		{
			CameraOrder.push_back(cam1);
			if (RootCamera < cam1)
				OffsetInfo.push_back(TimeOffset[RootCamera + cam1*nCams]);
			else
				OffsetInfo.push_back(-TimeOffset[RootCamera + cam1*nCams]);
		}

		added = false;
		for (int jj = 0; jj < (int)CameraOrder.size(); jj++)
		{
			if (CameraOrder[jj] == cam2)
			{
				added = true;
				break;
			}
		}

		if (!added)
		{
			CameraOrder.push_back(cam2);
			if (RootCamera < cam2)
				OffsetInfo.push_back(TimeOffset[RootCamera + cam2*nCams]);
			else
				OffsetInfo.push_back(-TimeOffset[RootCamera + cam2*nCams]);
		}
	}

	sprintf(Fname, "%s/MotionPriorSync.txt", Path);	fp = fopen(Fname, "w+");
	for (int kk = 0; kk < nCams; kk++)
		fprintf(fp, "%d %.4e \n", CameraOrder[kk], OffsetInfo[kk]);
	fclose(fp);

	delete[]weights, delete[]nValidPts, delete[]BaseLine, delete[]Traj3dCost, delete[]weightTable, delete[]TimeOffset, delete[]edges;

	return 0;
}

int Combine3DStatisticFromRandomSampling(char *Path, int nCams, int ntracks)
{
	vector<int> availFileID;
	char Fname[200];
	for (int ii = 0; ii < 10000; ii++)
	{
		sprintf(Fname, "%s/ATrack_0_%d.txt", Path, ii); FILE *fp = fopen(Fname, "r");
		if (fp != NULL)
		{
			availFileID.push_back(ii);
			fclose(fp);
		}
	}

	int navailFiles = availFileID.size(), fileCount = 0, nframes;
	double x, y, z, t;
	vector<double> *timeStamp = new vector<double>[ntracks];
	vector<Point3d>*Traject3D_iTrial = new vector<Point3d>[navailFiles*ntracks];
	for (int ii = 0; ii < navailFiles; ii++)
	{
		for (int tid = 0; tid < ntracks; tid++)
		{
			sprintf(Fname, "%s/ATrack_%d_%d.txt", Path, tid, availFileID[ii]); FILE *fp = fopen(Fname, "r");
			nframes = 0;
			while (fscanf(fp, "%lf %lf %lf %lf", &x, &y, &z, &t) != EOF)
			{
				if (fileCount == 0)
					timeStamp[tid].push_back(t);
				else	if (abs(timeStamp[tid][nframes] - t) > 0.1)
				{
					printf("Something wrong with the time stamp!");
					abort();
				}

				Traject3D_iTrial[fileCount*ntracks + tid].push_back(Point3d(x, y, z));
				nframes++;
			}
			fclose(fp);
		}
		fileCount++;
	}

	vector<Point3d> *P3D_Mean = new vector<Point3d>[ntracks], *P3D_STD = new vector<Point3d>[ntracks];
	for (int tid = 0; tid < ntracks; tid++)
		for (int fid = 0; fid < Traject3D_iTrial[tid].size(); fid++)
			P3D_Mean[tid].push_back(Point3d(0, 0, 0)), P3D_STD[tid].push_back(Point3d(0, 0, 0));

	for (int fileCount = 0; fileCount < navailFiles; fileCount++)
		for (int tid = 0; tid < ntracks; tid++)
			for (int fid = 0; fid < Traject3D_iTrial[fileCount*ntracks + tid].size(); fid++)
				P3D_Mean[tid][fid].x += Traject3D_iTrial[fileCount*ntracks + tid][fid].x / navailFiles,
				P3D_Mean[tid][fid].y += Traject3D_iTrial[fileCount*ntracks + tid][fid].y / navailFiles,
				P3D_Mean[tid][fid].z += Traject3D_iTrial[fileCount*ntracks + tid][fid].z / navailFiles;

	for (int fileCount = 0; fileCount < navailFiles; fileCount++)
		for (int tid = 0; tid < ntracks; tid++)
			for (int fid = 0; fid < Traject3D_iTrial[fileCount*ntracks + tid].size(); fid++)
				P3D_STD[tid][fid].x += pow(Traject3D_iTrial[fileCount*ntracks + tid][fid].x - P3D_Mean[tid][fid].x, 2) / (navailFiles - 1),
				P3D_STD[tid][fid].y += pow(Traject3D_iTrial[fileCount*ntracks + tid][fid].y - P3D_Mean[tid][fid].y, 2) / (navailFiles - 1),
				P3D_STD[tid][fid].z += pow(Traject3D_iTrial[fileCount*ntracks + tid][fid].z - P3D_Mean[tid][fid].z, 2) / (navailFiles - 1);


	for (int tid = 0; tid < ntracks; tid++)
	{
		sprintf(Fname, "%s/ATrackMSTD_%d.txt", Path, tid);  FILE *fp = fopen(Fname, "w+");
		for (int fid = 0; fid < P3D_Mean[tid].size(); fid++)
			fprintf(fp, "%.4f %.4f %.4f %.6f %.6f %.6f %.2f\n", P3D_Mean[tid][fid].x, P3D_Mean[tid][fid].y, P3D_Mean[tid][fid].z, sqrt(P3D_STD[tid][fid].x), sqrt(P3D_STD[tid][fid].y), sqrt(P3D_STD[tid][fid].z), timeStamp[tid][fid]);
		fclose(fp);
	}

	return 0;
}
int Generate3DUncertaintyFromRandomSampling(char *Path, vector<int> SelectedCams, vector<double> OffsetInfo, int startFrame, int stopFrame, int ntracks, int startSample = 0, int nSamples = 100, int motionPriorPower = 2)
{
	char Fname[200]; FILE *fp = 0;
	const double Tscale = 1000.0, fps = 60, ialpha = 1.0 / fps, eps = 1.0e-6, rate = 1.0, lamda = .3;

	//Read calib info
	int nCams = (int)SelectedCams.size();
	VideoData *VideoInfo = new VideoData[nCams];
	for (int camID = 0; camID < nCams; camID++)
		if (ReadVideoDataI(Path, VideoInfo[camID], SelectedCams[camID], startFrame, stopFrame) == 1)
			return 1;

	int frameID, id, npts;
	int nframes = max(MaxnFrames, stopFrame);

	double u, v;
	ImgPtEle ptEle;
	vector<ImgPtEle> *orgPerCam_UV = new vector<ImgPtEle>[nCams*ntracks];

	for (int camID = 0; camID < nCams; camID++)
	{
		for (int trackID = 0; trackID < ntracks; trackID++)
			orgPerCam_UV[camID*ntracks + trackID].reserve(stopFrame - startFrame + 1);

		sprintf(Fname, "%s/Track2D/C_%d.txt", Path, SelectedCams[camID]); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("Cannot load %s\n", Fname);
			return 1;
		}
		for (int trackID = 0; trackID < ntracks; trackID++)
		{
			fscanf(fp, "%d %d ", &id, &npts);
			if (id != trackID)
				printf("Problem at Point %d of Cam %d", id, camID);
			for (int pid = 0; pid < npts; pid++)
			{
				fscanf(fp, "%d %lf %lf ", &frameID, &u, &v);
				if (frameID < 0)
					continue;
				if (!VideoInfo[camID].VideoInfo[frameID].valid)
					continue; //camera not localized

				if (u > 0 && v > 0)
				{
					ptEle.pt2D.x = u, ptEle.pt2D.y = v, ptEle.frameID = frameID, ptEle.imHeight = VideoInfo[camID].VideoInfo[frameID].height, ptEle.imWidth = VideoInfo[camID].VideoInfo[frameID].width;
					LensCorrectionPoint(&ptEle.pt2D, VideoInfo[camID].VideoInfo[frameID].K, VideoInfo[camID].VideoInfo[frameID].distortion);
					orgPerCam_UV[camID*ntracks + id].push_back(ptEle);
				}
			}
		}
		fclose(fp);
	}

	//Sample 2d points with gaussian noise
	const double NoiseMag = 0;
	double startTime = omp_get_wtime();

	int numThreads = 1;
	for (int trialID = 0; trialID < nSamples; trialID++)
	{
		vector<ImgPtEle> *PerCam_UV = new vector<ImgPtEle>[nCams*ntracks];
		for (int camID = 0; camID < nCams; camID++)
		{
			for (int trackID = 0; trackID < ntracks; trackID++)
				PerCam_UV[camID*ntracks + trackID].reserve(orgPerCam_UV[camID*ntracks + trackID].size());

			for (int trackID = 0; trackID < ntracks; trackID++)
			{
				for (int pid = 0; pid < orgPerCam_UV[camID*ntracks + trackID].size(); pid++)
				{
					ptEle.pt2D.x = orgPerCam_UV[camID*ntracks + trackID][pid].pt2D.x + max(min(gaussian_noise(0, NoiseMag), 4.0*NoiseMag), -4.0*NoiseMag);
					ptEle.pt2D.y = orgPerCam_UV[camID*ntracks + trackID][pid].pt2D.y + max(min(gaussian_noise(0, NoiseMag), 4.0*NoiseMag), -4.0*NoiseMag);
					ptEle.frameID = orgPerCam_UV[camID*ntracks + trackID][pid].frameID;
					PerCam_UV[camID*ntracks + trackID].push_back(ptEle);
				}
			}
		}

		double P[12], AA[6], bb[2], ccT[3], dd[1];
		for (int trackID = 0; trackID < ntracks; trackID++)
		{
			for (int camID = 0; camID < nCams; camID++)
			{
				for (int frameID = 0; frameID < PerCam_UV[camID*ntracks + trackID].size(); frameID++)
				{
					int RealFrameID = PerCam_UV[camID*ntracks + trackID][frameID].frameID;

					for (int kk = 0; kk < 12; kk++)
					{
						P[kk] = VideoInfo[camID].VideoInfo[RealFrameID].P[kk];
						PerCam_UV[camID*ntracks + trackID][frameID].P[kk] = P[kk];
					}

					for (int kk = 0; kk < 9; kk++)
						PerCam_UV[camID*ntracks + trackID][frameID].K[kk] = VideoInfo[camID].VideoInfo[RealFrameID].K[kk],
						PerCam_UV[camID*ntracks + trackID][frameID].R[kk] = VideoInfo[camID].VideoInfo[RealFrameID].R[kk];

					for (int kk = 0; kk < 3; kk++)
						PerCam_UV[camID*ntracks + trackID][frameID].camcenter[kk] = VideoInfo[camID].VideoInfo[RealFrameID].camCenter[kk];

					//Q, U
					AA[0] = P[0], AA[1] = P[1], AA[2] = P[2], bb[0] = P[3];
					AA[3] = P[4], AA[4] = P[5], AA[5] = P[6], bb[1] = P[7];
					ccT[0] = P[8], ccT[1] = P[9], ccT[2] = P[10], dd[0] = P[11];

					PerCam_UV[camID*ntracks + trackID][frameID].Q[0] = AA[0] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.x*ccT[0],
						PerCam_UV[camID*ntracks + trackID][frameID].Q[1] = AA[1] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.x*ccT[1],
						PerCam_UV[camID*ntracks + trackID][frameID].Q[2] = AA[2] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.x*ccT[2];
					PerCam_UV[camID*ntracks + trackID][frameID].Q[3] = AA[3] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.y*ccT[0],
						PerCam_UV[camID*ntracks + trackID][frameID].Q[4] = AA[4] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.y*ccT[1],
						PerCam_UV[camID*ntracks + trackID][frameID].Q[5] = AA[5] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.y*ccT[2];
					PerCam_UV[camID*ntracks + trackID][frameID].u[0] = dd[0] * PerCam_UV[camID*ntracks + trackID][frameID].pt2D.x - bb[0],
						PerCam_UV[camID*ntracks + trackID][frameID].u[1] = dd[0] * PerCam_UV[camID*ntracks + trackID][frameID].pt2D.y - bb[1];

					double stdA = 100.0;//Interestingly, Ceres does not work if all the input are the same-->need some random perturbation. 
					PerCam_UV[camID*ntracks + trackID][frameID].pt3D = Point3d(gaussian_noise(0.0, stdA), gaussian_noise(0.0, stdA), gaussian_noise(0.0, stdA));
				}
			}
		}

		//Initialize data for optim
		vector<int> PointsPerTrack;
		vector<int *> PerTrackFrameID(ntracks);
		vector<double*> All3D(ntracks);
		int ntimeinstances, maxntimeinstances = 0;
		for (int trackID = 0; trackID < ntracks; trackID++)
		{
			ntimeinstances = 0;
			for (int camID = 0; camID < nCams; camID++)
				ntimeinstances += PerCam_UV[camID*ntracks + trackID].size();

			if (maxntimeinstances < ntimeinstances)
				maxntimeinstances = ntimeinstances;

			PerTrackFrameID[trackID] = new int[ntimeinstances];
			All3D[trackID] = new double[3 * ntimeinstances];
		}

		double currentOffset[MaxnCams], APLDCost[4];
		for (int ii = 0; ii < nCams; ii++)
			currentOffset[ii] = OffsetInfo[ii];

		MotionPrior_Optim_SpatialStructure_Algebraic(Path, All3D, PerCam_UV, PointsPerTrack, currentOffset, ntracks, false, nCams, motionPriorPower, Tscale, ialpha, eps, lamda, APLDCost, false, false);
		MotionPrior_ML_Weighting(PerCam_UV, ntracks, nCams);
		//MotionPrior_Optim_SpatialStructure_Geometric(Path, All3D, PerCam_UV, PointsPerTrack, currentOffset, ntracks, false, nCams, Tscale, ialpha, eps, lamda, APLDCost, false, false);

		for (int trackID = 0; trackID < ntracks; trackID++)
		{
			sprintf(Fname, "%s/ATrack_%d_%d.txt", Path, trackID, trialID + startSample);  FILE *fp = fopen(Fname, "w+");
			for (int camID = 0; camID < nCams; camID++)
				for (int fid = 0; fid < PerCam_UV[camID*ntracks + trackID].size(); fid++)
					fprintf(fp, "%.4f %.4f %.4f %.2f %d %d\n", PerCam_UV[camID*ntracks + trackID][fid].pt3D.x, PerCam_UV[camID*ntracks + trackID][fid].pt3D.y, PerCam_UV[camID*ntracks + trackID][fid].pt3D.z,
					1.0*currentOffset[camID] * ialpha*Tscale + 1.0*PerCam_UV[camID*ntracks + trackID][fid].frameID * ialpha*Tscale*rate, camID, PerCam_UV[camID*ntracks + trackID][fid].frameID);
			fclose(fp);
		}

		printf("%.1f%% ... TR: %.2fs\n", 100.0*trialID / nSamples, (omp_get_wtime() - startTime) / (trialID + 0.000001)*(nSamples - trialID));
		delete[]PerCam_UV;
	}

	return 0;
}

struct CeresResamplingSpline {
	CeresResamplingSpline(double *XIn, double *PIn, double *PhiDataIn, double *PhiPriorIn, double *TimeStampDatain, double *TimeStampPriorIn, Point2d *pt2DIn, double lamda, int nData, int nResamples, int nCoeffs) :lamda(lamda), nData(nData), nResamples(nResamples), nCoeffs(nCoeffs)
	{
		X = XIn, P = PIn, PhiData = PhiDataIn, PhiPrior = PhiPriorIn, TimeStampData = TimeStampDatain, TimeStampPrior = TimeStampPriorIn, pt2D = pt2DIn;
	}

	template <typename T>    bool operator()(T const* const* C, T* residuals)     const
	{
		//cost = lamda*(PBC-u)^2 + (1-lamda)*Prior
		T x, y, z, numX, numY, denum, sqrtlamda1 = sqrt((T)lamda), lamda2 = (T)1.0 - lamda, lamda3 = (T)2.0;

		//Projection cost 
		for (int ii = 0; ii < nData; ii++)
		{
			x = (T) 0.0, y = (T) 0.0, z = (T)0.0;
			for (int jj = 0; jj < nCoeffs; jj++)
			{
				if (PhiData[jj + ii*nCoeffs] < 0.00001)
					continue;
				x += (T)PhiData[jj + ii*nCoeffs] * C[0][jj],
					y += (T)PhiData[jj + ii*nCoeffs] * C[0][jj + nCoeffs],
					z += (T)PhiData[jj + ii*nCoeffs] * C[0][jj + 2 * nCoeffs];
			}


			numX = (T)P[12 * ii] * x + (T)P[12 * ii + 1] * y + (T)P[12 * ii + 2] * z + (T)P[12 * ii + 3];
			numY = (T)P[12 * ii + 4] * x + (T)P[12 * ii + 5] * y + (T)P[12 * ii + 6] * z + (T)P[12 * ii + 7];
			denum = (T)P[12 * ii + 8] * x + (T)P[12 * ii + 9] * y + (T)P[12 * ii + 10] * z + (T)P[12 * ii + 11];

			residuals[2 * ii] = (T)sqrtlamda1*(numX / denum - (T)pt2D[ii].x);
			residuals[2 * ii + 1] = (T)sqrtlamda1*(numY / denum - (T)pt2D[ii].y);
		}

		//Motion cost 
		/*T xo, yo, zo;
		for (int ii = 0; ii < nResamples; ii++)
		{
		x = (T) 0.0, y = (T) 0.0, z = (T)0.0;
		for (int jj = 0; jj < nCoeffs; jj++)
		{
		if (PhiPrior[jj + ii*nCoeffs] < 0.00001)
		continue;
		x += (T)PhiPrior[jj + ii*nCoeffs] * C[0][jj],
		y += (T)PhiPrior[jj + ii*nCoeffs] * C[0][jj + nCoeffs],
		z += (T)PhiPrior[jj + ii*nCoeffs] * C[0][jj + 2 * nCoeffs];
		}

		if (ii > 0)
		{
		T dx = x - xo, dy = y - yo, dz = z - zo;
		T error = (T)lamda2*(dx*dx + dy*dy + dz*dz) / (T)(TimeStampPrior[ii] - TimeStampPrior[ii - 1]);
		residuals[2 * nData + ii - 1] = sqrt(max((T)(1.0e-16), error));
		}
		xo = x, yo = y, zo = z;
		}*/

		return true;
	}

	int nData, nResamples, nCoeffs;
	double lamda;
	double  *X, *P, *PhiData, *PhiPrior, *TimeStampData, *TimeStampPrior;
	Point2d *pt2D;
};
void ResamplingOf3DTrajectorySpline(vector<ImgPtEle> &Traj3D, bool non_monotonicDescent, double Break_Step, double Resample_Step, double lamda, bool silent = true)
{
	const int SplineOrder = 4;

	double earliest = Traj3D[0].timeStamp, latest = Traj3D.back().timeStamp;
	int nBreaks = (int)(ceil((ceil(latest) - floor(earliest)) / Break_Step)) + 1, nCoeffs = nBreaks + 2, nData = (int)Traj3D.size(), nptsPrior = (int)((latest - earliest) / Resample_Step);

	double*BreakPts = new double[nBreaks], *X = new double[nData * 3];
	double *PhiData = new double[nData*nCoeffs], *PhiPrior = new double[nptsPrior*nCoeffs], *C = new double[3 * nCoeffs];
	double *PmatData = new double[nData * 12], *TimeStampData = new double[nData], *TimeStampPrior = new double[nptsPrior];
	Point2d *pt2dData = new Point2d[nData];

	for (int ii = 0; ii < nBreaks; ii++)
		BreakPts[ii] = floor(earliest) + Break_Step*ii;

	for (int ii = 0; ii < nData; ii++)
	{
		for (int jj = 0; jj < 12; jj++)
			PmatData[12 * ii + jj] = Traj3D[ii].P[jj];

		TimeStampData[ii] = Traj3D[ii].timeStamp;

		pt2dData[ii] = Traj3D[ii].pt2D;

		X[ii] = Traj3D[ii].pt3D.x, X[ii + nData] = Traj3D[ii].pt3D.y, X[ii + 2 * nData] = Traj3D[ii].pt3D.z;
	}

	for (int ii = 0; ii < nptsPrior; ii++)
		TimeStampPrior[ii] = floor(earliest) + Resample_Step*ii;

	//Find and delete breakpts with no data in between (this makes the basis matrix to be ill-condition)
	vector<int>IdToDel;
	for (int ii = 0; ii < nBreaks - 1; ii++)
	{
		bool found = false;
		for (int jj = 0; jj < nData; jj++)
		{
			if ((BreakPts[ii] < TimeStampData[jj] && TimeStampData[jj] < BreakPts[ii + 1]))
			{
				found = true;
				break;
			}
		}
		if (!found)
			IdToDel.push_back(ii + 1);
	}
	if ((int)IdToDel.size() > 0)
	{
		vector<double> tBreakPts;
		for (int ii = 0; ii < nBreaks; ii++)
			tBreakPts.push_back(BreakPts[ii]);

		for (int ii = (int)IdToDel.size() - 1; ii >= 0; ii--)
			tBreakPts.erase(tBreakPts.begin() + IdToDel[ii]);

		nBreaks = (int)tBreakPts.size(), nCoeffs = nBreaks + 2;
		for (int ii = 0; ii < nBreaks; ii++)
			BreakPts[ii] = tBreakPts[ii];
	}

	//Generate Spline Basis
	//GenerateResamplingSplineBasisWithBreakPts(PhiData, TimeStampData, BreakPts, nData, nBreaks, SplineOrder);
	//GenerateResamplingSplineBasisWithBreakPts(PhiPrior, TimeStampPrior, BreakPts, nptsPrior, nBreaks, SplineOrder);
	BSplineGetAllBasis(PhiData, TimeStampData, BreakPts, nData, nBreaks, SplineOrder);
	BSplineGetAllBasis(PhiPrior, TimeStampPrior, BreakPts, nptsPrior, nBreaks, SplineOrder);


	//Initialize basis coefficients: X_data = Phi_data*C
	for (int jj = 0; jj < 3; jj++)
	{
		LS_Solution_Double(PhiData, X + jj*nData, nData, nCoeffs);
		for (int ii = 0; ii < nCoeffs; ii++)
			C[ii + jj*nCoeffs] = X[ii + jj*nData];
	}

#ifdef ENABLE_DEBUG_FLAG
	double ActionCost = 0, ProjCost = 0;
	for (int ii = 0; ii < nData - 1; ii++)
	{
		double xyz1[] = { Traj3D[ii].pt3D.x, Traj3D[ii].pt3D.y, Traj3D[ii].pt3D.z };
		double xyz2[] = { Traj3D[ii + 1].pt3D.x, Traj3D[ii + 1].pt3D.y, Traj3D[ii + 1].pt3D.z };
		double costi = LeastActionError(xyz1, xyz2, Traj3D[ii].timeStamp, Traj3D[ii + 1].timeStamp, 1.0e-6, 2);
		ActionCost += costi;
	}

	for (int ii = 0; ii < nData; ii++)
	{
		int  camID = Traj3D[ii].viewID, frameID = Traj3D[ii].frameID;
		Point2d p2d = Traj3D[ii].pt2D;
		Point3d p3d = Traj3D[ii].pt3D;
		double	err = PinholeReprojectionErrorSimpleDebug(Traj3D[ii].P, Traj3D[ii].pt3D, Traj3D[ii].pt2D);
		ProjCost += err*err;
	}
	printf("(Before Spline: Action cost, projection cost): %.4e %.4e\n", ActionCost, sqrt(ProjCost / nData));

	ProjCost = 0.0;
	for (int ii = 0; ii < nData; ii++)
	{
		double x = 0.0, y = 0.0, z = 0.0;
		for (int jj = 0; jj < nCoeffs; jj++)
		{
			if (PhiData[jj + ii*nCoeffs] < 1e-6)
				continue;
			x += PhiData[jj + ii*nCoeffs] * C[jj],
				y += PhiData[jj + ii*nCoeffs] * C[jj + nCoeffs],
				z += PhiData[jj + ii*nCoeffs] * C[jj + 2 * nCoeffs];
		}

		double numX = PmatData[12 * ii] * x + PmatData[12 * ii + 1] * y + PmatData[12 * ii + 2] * z + PmatData[12 * ii + 3];
		double numY = PmatData[12 * ii + 4] * x + PmatData[12 * ii + 5] * y + PmatData[12 * ii + 6] * z + PmatData[12 * ii + 7];
		double denum = PmatData[12 * ii + 8] * x + PmatData[12 * ii + 9] * y + PmatData[12 * ii + 10] * z + PmatData[12 * ii + 11];

		double errX = (numX / denum - pt2dData[ii].x);
		double errY = (numY / denum - pt2dData[ii].y);
		ProjCost += errX*errX + errY*errY;
	}

	double xo, yo, zo;
	for (int ii = 0; ii < nptsPrior; ii++)
	{
		double x = 0.0, y = 0.0, z = 0.0;
		for (int jj = 0; jj < nCoeffs; jj++)
		{
			if (PhiPrior[jj + ii*nCoeffs] < 1e-6)
				continue;
			x += PhiPrior[jj + ii*nCoeffs] * C[jj],
				y += PhiPrior[jj + ii*nCoeffs] * C[jj + nCoeffs],
				z += PhiPrior[jj + ii*nCoeffs] * C[jj + 2 * nCoeffs];
		}

		if (ii > 0)
			ActionCost += (pow(x - xo, 2) + pow(y - yo, 2) + pow(z - zo, 2)) / (TimeStampPrior[ii] - TimeStampPrior[ii - 1]);
		xo = x, yo = y, zo = z;

	}
	printf("(Spline: Action cost, projection cost, Totalcost): %.4e %.4e %.4e\n", ActionCost, sqrt(ProjCost / nData), lamda*ProjCost + (1.0 - lamda)*ActionCost);
#endif

	for (int ii = 0; ii < nData; ii++)
		X[ii] = Traj3D[ii].pt3D.x, X[ii + nData] = Traj3D[ii].pt3D.y, X[ii + 2 * nData] = Traj3D[ii].pt3D.z;

	//Run ceres optimization
	ceres::Problem problem;

	vector<double*> parameter_blocks;
	parameter_blocks.push_back(C);
	ceres::DynamicAutoDiffCostFunction<CeresResamplingSpline, 4> *cost_function = new ceres::DynamicAutoDiffCostFunction<CeresResamplingSpline, 4>(new CeresResamplingSpline(X, PmatData, PhiData, PhiPrior, TimeStampData, TimeStampPrior, pt2dData, lamda, nData, nptsPrior, nCoeffs));
	//ceres::DynamicNumericDiffCostFunction<CeresResamplingSpline, ceres::CENTRAL> *cost_function = new ceres::DynamicNumericDiffCostFunction<CeresResamplingSpline, ceres::CENTRAL>(new CeresResamplingSpline(X, PmatData, PhiData, PhiPrior, TimeStampData, TimeStampPrior, pt2dData, lamda, nData, nptsPrior, nCoeffs));
	cost_function->AddParameterBlock(3 * nCoeffs);
	//cost_function->SetNumResiduals(2 * nData + nptsPrior - 1);
	cost_function->SetNumResiduals(2 * nData);
	problem.AddResidualBlock(cost_function, NULL, parameter_blocks);

	ceres::Solver::Options options;
	options.max_num_iterations = 100;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = silent ? false : true;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
#pragma omp critical
	if (silent)
		std::cout << summary.BriefReport() << "\n";
	else
		std::cout << summary.FullReport() << "\n";

	//Final curve
	mat_mul(PhiData, C, X, nData, nCoeffs, 1);
	mat_mul(PhiData, C + nCoeffs, X + nData, nData, nCoeffs, 1);
	mat_mul(PhiData, C + 2 * nCoeffs, X + 2 * nData, nData, nCoeffs, 1);

	for (int ii = 0; ii < nData; ii++)
		Traj3D[ii].pt3D.x = X[ii], Traj3D[ii].pt3D.y = X[ii + nData], Traj3D[ii].pt3D.z = X[ii + 2 * nData];

#ifdef ENABLE_DEBUG_FLAG 
	{double ProjCost = 0.0;
	for (int ii = 0; ii < nData; ii++)
	{
		double x = 0.0, y = 0.0, z = 0.0;
		for (int jj = 0; jj < nCoeffs; jj++)
		{
			if (PhiData[jj + ii*nCoeffs] < 1e-6)
				continue;
			x += PhiData[jj + ii*nCoeffs] * C[jj],
				y += PhiData[jj + ii*nCoeffs] * C[jj + nCoeffs],
				z += PhiData[jj + ii*nCoeffs] * C[jj + 2 * nCoeffs];
		}

		double numX = PmatData[12 * ii] * x + PmatData[12 * ii + 1] * y + PmatData[12 * ii + 2] * z + PmatData[12 * ii + 3];
		double numY = PmatData[12 * ii + 4] * x + PmatData[12 * ii + 5] * y + PmatData[12 * ii + 6] * z + PmatData[12 * ii + 7];
		double denum = PmatData[12 * ii + 8] * x + PmatData[12 * ii + 9] * y + PmatData[12 * ii + 10] * z + PmatData[12 * ii + 11];

		double errX = (numX / denum - pt2dData[ii].x);
		double errY = (numY / denum - pt2dData[ii].y);
		ProjCost += errX*errX + errY*errY;
	}

	double xo, yo, zo, ActionCost = 0.0;
	for (int ii = 0; ii < nptsPrior; ii++)
	{
		double x = 0.0, y = 0.0, z = 0.0;
		for (int jj = 0; jj < nCoeffs; jj++)
		{
			if (PhiPrior[jj + ii*nCoeffs] < 1e-6)
				continue;
			x += PhiPrior[jj + ii*nCoeffs] * C[jj],
				y += PhiPrior[jj + ii*nCoeffs] * C[jj + nCoeffs],
				z += PhiPrior[jj + ii*nCoeffs] * C[jj + 2 * nCoeffs];
		}

		if (ii > 0)
			ActionCost += (pow(x - xo, 2) + pow(y - yo, 2) + pow(z - zo, 2)) / (TimeStampPrior[ii] - TimeStampPrior[ii - 1]);
		xo = x, yo = y, zo = z;

	}
	printf("(After Spline: Action cost, projection cost, Totalcost): %.4e %.4e %.4e\n", ActionCost, sqrt(ProjCost / nData), lamda*ProjCost + (1.0 - lamda)*ActionCost);
	}
#endif

	delete[]X, delete[]BreakPts, delete[]PhiData, delete[]PhiPrior, delete[]C;
	delete[]PmatData, delete[]TimeStampData, delete[]pt2dData;

	return;
}
int ResamplingOf3DTrajectorySplineDriver(char *Path, vector<int> &SelectedCams, vector<double> &OffsetInfo, int startFrame, int stopFrame, int ntracks, double lamda)
{
	char Fname[200]; FILE *fp = 0;
	const double Tscale = 1000.0, fps = 60.0, ialpha = 1.0 / fps;

	//Read calib info
	int nCams = (int)SelectedCams.size();
	VideoData *VideoInfo = new VideoData[nCams];
	for (int camID = 0; camID < nCams; camID++)
		if (ReadVideoDataI(Path, VideoInfo[camID], SelectedCams[camID], startFrame, stopFrame) == 1)
			return 1;

	int frameID, id, npts;
	int nframes = max(MaxnFrames, stopFrame);

	double u, v;
	vector<int>VectorCamID, VectorFrameID;
	vector<double> AllError2D;
	ImgPtEle ptEle;
	vector<ImgPtEle> *PerCam_UV = new vector<ImgPtEle>[nCams*ntracks];

	printf("Get 2D ...");
	for (int camID = 0; camID < nCams; camID++)
	{
		for (int trackID = 0; trackID < ntracks; trackID++)
			PerCam_UV[camID*ntracks + trackID].reserve(stopFrame - startFrame + 1);

		sprintf(Fname, "%s/Track2D/%d.txt", Path, SelectedCams[camID]); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("Cannot load %s\n", Fname);
			return 1;
		}
		for (int trackID = 0; trackID < ntracks; trackID++)
		{
			fscanf(fp, "%d %d ", &id, &npts);
			if (id != trackID)
				printf("Problem at Point %d of Cam %d", id, camID);
			for (int pid = 0; pid < npts; pid++)
			{
				fscanf(fp, "%d %lf %lf ", &frameID, &u, &v);
				if (frameID < 0)
					continue;
				if (!VideoInfo[camID].VideoInfo[frameID].valid)
					continue; //camera not localized

				if (u > 0 && v > 0)
				{
					ptEle.pt2D.x = u, ptEle.pt2D.y = v, ptEle.frameID = frameID, ptEle.imWidth = VideoInfo[camID].VideoInfo[frameID].width, ptEle.imHeight = VideoInfo[camID].VideoInfo[frameID].height;
					LensCorrectionPoint(&ptEle.pt2D, VideoInfo[camID].VideoInfo[frameID].K, VideoInfo[camID].VideoInfo[frameID].distortion);
					PerCam_UV[camID*ntracks + id].push_back(ptEle);
				}
			}
		}
		fclose(fp);
	}

	printf("Get rays info ....");
	double P[12], AA[6], bb[2], ccT[3], dd[1];
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		int count = 0;
		for (int camID = 0; camID < nCams; camID++)
		{
			for (int frameID = 0; frameID < PerCam_UV[camID*ntracks + trackID].size(); frameID++)
			{
				int RealFrameID = PerCam_UV[camID*ntracks + trackID][frameID].frameID;

				for (int kk = 0; kk < 12; kk++)
				{
					P[kk] = VideoInfo[camID].VideoInfo[RealFrameID].P[kk];
					PerCam_UV[camID*ntracks + trackID][frameID].P[kk] = P[kk];
				}

				for (int kk = 0; kk < 9; kk++)
					PerCam_UV[camID*ntracks + trackID][frameID].K[kk] = VideoInfo[camID].VideoInfo[RealFrameID].K[kk],
					PerCam_UV[camID*ntracks + trackID][frameID].R[kk] = VideoInfo[camID].VideoInfo[RealFrameID].R[kk];

				for (int kk = 0; kk < 3; kk++)
					PerCam_UV[camID*ntracks + trackID][frameID].camcenter[kk] = VideoInfo[camID].VideoInfo[RealFrameID].camCenter[kk];

				AA[0] = P[0], AA[1] = P[1], AA[2] = P[2], bb[0] = P[3];
				AA[3] = P[4], AA[4] = P[5], AA[5] = P[6], bb[1] = P[7];
				ccT[0] = P[8], ccT[1] = P[9], ccT[2] = P[10], dd[0] = P[11];

				PerCam_UV[camID*ntracks + trackID][frameID].Q[0] = AA[0] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.x*ccT[0],
					PerCam_UV[camID*ntracks + trackID][frameID].Q[1] = AA[1] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.x*ccT[1],
					PerCam_UV[camID*ntracks + trackID][frameID].Q[2] = AA[2] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.x*ccT[2];
				PerCam_UV[camID*ntracks + trackID][frameID].Q[3] = AA[3] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.y*ccT[0],
					PerCam_UV[camID*ntracks + trackID][frameID].Q[4] = AA[4] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.y*ccT[1],
					PerCam_UV[camID*ntracks + trackID][frameID].Q[5] = AA[5] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.y*ccT[2];
				PerCam_UV[camID*ntracks + trackID][frameID].u[0] = dd[0] * PerCam_UV[camID*ntracks + trackID][frameID].pt2D.x - bb[0],
					PerCam_UV[camID*ntracks + trackID][frameID].u[1] = dd[0] * PerCam_UV[camID*ntracks + trackID][frameID].pt2D.y - bb[1];

				PerCam_UV[camID*ntracks + trackID][frameID].pt3D = Point3d(count, count, count);//Interestingly, Ceres does not work if all the input are the same
				count++;
			}
		}
	}

	//Initialize data for optim
	vector<int> PointsPerTrack;
	vector<int *> PerTrackFrameID(ntracks);
	vector<double*> All3D(ntracks);
	int ntimeinstances, maxntimeinstances = 0;
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		ntimeinstances = 0;
		for (int camID = 0; camID < nCams; camID++)
			ntimeinstances += PerCam_UV[camID*ntracks + trackID].size();

		if (maxntimeinstances < ntimeinstances)
			maxntimeinstances = ntimeinstances;

		PerTrackFrameID[trackID] = new int[ntimeinstances];
		All3D[trackID] = new double[3 * ntimeinstances];
	}

	vector<int> VisCamID, VisLocalFrameID;
	vector<double> TimeStamp;
	vector<Point3d> T3D;
	vector<ImgPtEle> *Traj3D = new vector<ImgPtEle>[ntracks];
	int cID, fID;
	double x, y, z, t;
	int dummy[10000];
	double ts[10000];
	ImgPtEle iele;

	printf("Get 3D data:\n");
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		TimeStamp.clear(), T3D.clear(), VisCamID.clear(), VisLocalFrameID.clear();

		sprintf(Fname, "%s/OptimizedRaw_Track_%d.txt", Path, trackID); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("Cannot load %s\n", Fname);
			break;
		}
		while (fscanf(fp, "%lf %lf %lf %lf %d %d", &x, &y, &z, &t, &cID, &fID) != EOF)
		{
			TimeStamp.push_back(t), VisCamID.push_back(cID), VisLocalFrameID.push_back(fID);
			T3D.push_back(Point3d(x, y, z));
		}
		fclose(fp);

		for (int ii = 0; ii < (int)TimeStamp.size(); ii++)
			ts[ii] = TimeStamp[ii], dummy[ii] = ii;
		Quick_Sort_Double(ts, dummy, 0, (int)TimeStamp.size() - 1);

		for (int ii = 0; ii < (int)TimeStamp.size(); ii++)
		{
			int id = dummy[ii], camID = VisCamID[id], frameID = VisLocalFrameID[id];
			iele.viewID = camID, iele.frameID = frameID, iele.timeStamp = TimeStamp[id];

			iele.pt3D = T3D[id];
			for (int fid = 0; fid < (int)PerCam_UV[camID*ntracks + trackID].size(); fid++)
			{
				if (PerCam_UV[camID*ntracks + trackID][fid].frameID == frameID)
				{
					iele.pt2D = PerCam_UV[camID*ntracks + trackID][fid].pt2D;
					for (int jj = 0; jj < 12; jj++)
						iele.P[jj] = PerCam_UV[camID*ntracks + trackID][fid].P[jj];
					break;
				}
			}
			Traj3D[trackID].push_back(iele);
		}
	}

	printf("Cubic Bspline resampling:\n");

	omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
#pragma omp critical
		printf("%d: ", trackID);

		ResamplingOf3DTrajectorySpline(Traj3D[trackID], true, ialpha *Tscale, ialpha *Tscale / nCams, lamda);

#pragma omp critical
		{
			sprintf(Fname, "%s/SplineResampled_Track_%d.txt", Path, trackID); remove(Fname);
			fp = fopen(Fname, "w+");
			for (int ii = 0; ii < (int)Traj3D[trackID].size(); ii++)
				fprintf(fp, "%.4f %.4f %.4f %.4f %d %d\n", Traj3D[trackID][ii].pt3D.x, Traj3D[trackID][ii].pt3D.y, Traj3D[trackID][ii].pt3D.z, Traj3D[trackID][ii].timeStamp, Traj3D[trackID][ii].viewID, Traj3D[trackID][ii].frameID);
			fclose(fp);
		}
	}
	printf("Done!\n\n");

	delete[]VideoInfo;
	delete[]PerCam_UV;
	delete[]Traj3D;

	return 0;
}
struct CeresResamplingDCT {
	CeresResamplingDCT(double *XIn, double *PIn, double *iBDataIn, double *sqrtWeightIn, Point2d *pt2DIn, double lamda1, double lamda2, int nData, int nResamples) :lamda1(lamda1), lamda2(lamda2), nData(nData), nResamples(nResamples)
	{
		X = XIn, P = PIn, iBData = iBDataIn, sqrtWeight = sqrtWeightIn, pt2D = pt2DIn;
	}

	template <typename T>    bool operator()(T const* const* C, T* residuals)     const
	{
		//cost = lamda1*(PiBC-u)^2 + lamda2*Prior
		T x, y, z, numX, numY, denum, sqrtlamda1 = sqrt((T)lamda1), sqrtlamda2 = sqrt((T)lamda2);

		//Projection cost : lamda1*(PiBC-u)^2
		for (int ii = 0; ii < nData; ii++)
		{
			//X = iB*C
			x = (T) 0.0, y = (T) 0.0, z = (T)0.0;
			for (int jj = 0; jj < nResamples; jj++)
			{
				x += (T)iBData[jj + ii*nResamples] * C[0][jj],
					y += (T)iBData[jj + ii*nResamples] * C[0][jj + nResamples],
					z += (T)iBData[jj + ii*nResamples] * C[0][jj + 2 * nResamples];
			}

			numX = (T)P[12 * ii] * x + (T)P[12 * ii + 1] * y + (T)P[12 * ii + 2] * z + (T)P[12 * ii + 3];
			numY = (T)P[12 * ii + 4] * x + (T)P[12 * ii + 5] * y + (T)P[12 * ii + 6] * z + (T)P[12 * ii + 7];
			denum = (T)P[12 * ii + 8] * x + (T)P[12 * ii + 9] * y + (T)P[12 * ii + 10] * z + (T)P[12 * ii + 11];

			residuals[2 * ii] = (T)sqrtlamda1*(numX / denum - (T)pt2D[ii].x);
			residuals[2 * ii + 1] = (T)sqrtlamda1*(numY / denum - (T)pt2D[ii].y);
		}

		//Motion prior in DCT form: CT*W*C
		for (int ii = 0; ii < nResamples; ii++)
		{
			T lamdaW = sqrtlamda2* sqrtWeight[ii];
			residuals[2 * nData + 3 * ii] = lamdaW* C[0][ii];
			residuals[2 * nData + 3 * ii + 1] = lamdaW * C[0][ii + nResamples];
			residuals[2 * nData + 3 * ii + 2] = lamdaW * C[0][ii + 2 * nResamples];
		}

		return true;
	}

	int nData, nResamples;
	double  *X, *P, *iBData, *sqrtWeight, lamda1, lamda2;
	Point2d *pt2D;
};
void ResamplingOf3DTrajectoryDCT(vector<ImgPtEle> &Traj3D, int PriorOrder, bool non_monotonicDescent, double Resample_Step, double lamda1, double lamda2, bool silent = true)
{
	double earliest = Traj3D[0].timeStamp, latest = Traj3D.back().timeStamp;
	int nData = (int)Traj3D.size(), nResamples = (int)((latest - earliest) / Resample_Step);

	//if (PriorOrder == 1)
	//	lamda2 = lamda2*Resample_Step;//scale the weight to take into account the temporal integration of uniformedly sampled motion prior 

	double *X = new double[nData * 3], *C = new double[3 * nResamples];
	double *iBData = new double[nData*nResamples], *BResampled = new double[nResamples*nResamples], *sqrtWeight = new double[nResamples];
	double *PmatData = new double[nData * 12], *TimeStampData = new double[nData];
	Point2d *pt2dData = new Point2d[nData];

	for (int ii = 0; ii < nData; ii++)
	{
		for (int jj = 0; jj < 12; jj++)
			PmatData[12 * ii + jj] = Traj3D[ii].P[jj];

		pt2dData[ii] = Traj3D[ii].pt2D;
		X[ii] = Traj3D[ii].pt3D.x, X[ii + nData] = Traj3D[ii].pt3D.y, X[ii + 2 * nData] = Traj3D[ii].pt3D.z;
		TimeStampData[ii] = (Traj3D[ii].timeStamp - earliest) / (latest - earliest)*(nResamples - 1);//Normalize to [0, n-1] range
	}

	//Generate DCT Basis
	GenerateDCTBasis(nResamples, BResampled, sqrtWeight);
	for (int ii = 0; ii < nResamples; ii++)
		if (PriorOrder == 1)
			sqrtWeight[ii] = sqrt(-sqrtWeight[ii]); //(1) using precomputed sqrt is better for ceres' squaring residual square nature; (2) weigths are negative, but that does not matter for ctwc optim.
		else
			sqrtWeight[ii] = -sqrtWeight[ii]; //ctw.^2c-->ceres: residuals = c*W;

	for (int ii = 0; ii < nData; ii++)
		GenerateiDCTBasis(iBData + ii*nResamples, nResamples, TimeStampData[ii]);

	//Initialize basis coefficients: iBd(:, 1:activeBasis)*C =  X_d
	const int nactiveBasis = 20;
	Map < Matrix < double, Dynamic, Dynamic, RowMajor > > eiBData(iBData, nData, nResamples);
	MatrixXd etiBData = eiBData.block(0, 0, nData, nactiveBasis);
	JacobiSVD<MatrixXd> etiP_svd(etiBData, ComputeThinU | ComputeThinV);
	for (int ii = 0; ii < 3; ii++)
	{
		for (int jj = nactiveBasis; jj < nResamples; jj++)
			C[jj + nResamples*ii] = 0.0; //set coeffs outside active basis to 0

		Map<VectorXd> eX(X + nData*ii, nData);
		Map<VectorXd> eC(C + nResamples*ii, nactiveBasis);
		eC = etiP_svd.solve(eX);
	}

#ifdef ENABLE_DEBUG_FLAG
	{
		double x, y, z, numX, numY, denum;

		//Projection cost : (PiBC-u)^2
		double projectionCost = 0.0;
		for (int ii = 0; ii < nData; ii++)
		{
			//X = iB*C
			x = 0.0, y = 0.0, z = 0.0;
			for (int jj = 0; jj < nResamples; jj++)
			{
				x += iBData[jj + ii*nResamples] * C[jj],
					y += iBData[jj + ii*nResamples] * C[jj + nResamples],
					z += iBData[jj + ii*nResamples] * C[jj + 2 * nResamples];
			}

			numX = PmatData[12 * ii] * x + PmatData[12 * ii + 1] * y + PmatData[12 * ii + 2] * z + PmatData[12 * ii + 3];
			numY = PmatData[12 * ii + 4] * x + PmatData[12 * ii + 5] * y + PmatData[12 * ii + 6] * z + PmatData[12 * ii + 7];
			denum = PmatData[12 * ii + 8] * x + PmatData[12 * ii + 9] * y + PmatData[12 * ii + 10] * z + PmatData[12 * ii + 11];

			projectionCost += (pow(numX / denum - pt2dData[ii].x, 2) + pow(numY / denum - pt2dData[ii].y, 2));
		}

		//Motion prior in DCT form: CT*W*C
		double MotionPrior = 0.0;
		for (int ii = 0; ii < nResamples; ii++)
			MotionPrior += pow(sqrtWeight[ii], 2)*(pow(C[ii], 2) + pow(C[ii + nResamples], 2) + pow(C[ii + 2 * nResamples], 2));

#pragma omp critical
		printf("(DCT before: Motion cost, projection cost, Totalcost): %.4e %.4e %.4e\n", MotionPrior, sqrt(projectionCost / nData), lamda1*projectionCost + lamda2*MotionPrior);
	}
#endif

	//Run ceres optimization
	ceres::Problem problem;

	vector<double*> parameter_blocks;
	parameter_blocks.push_back(C);
	ceres::DynamicAutoDiffCostFunction<CeresResamplingDCT, 4> *cost_function = new ceres::DynamicAutoDiffCostFunction<CeresResamplingDCT, 4>(
		new CeresResamplingDCT(X, PmatData, iBData, sqrtWeight, pt2dData, lamda1, lamda2, nData, nResamples));
	cost_function->AddParameterBlock(3 * nResamples);
	cost_function->SetNumResiduals(2 * nData + 3 * nResamples);
	problem.AddResidualBlock(cost_function, NULL, parameter_blocks);

	ceres::Solver::Options options;
	options.max_num_iterations = 100;
	//options.num_threads = omp_get_max_threads();
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = silent ? false : true;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

#pragma omp critical
	if (silent)
		std::cout << summary.BriefReport() << "\n";
	else
		std::cout << summary.FullReport() << "\n";

#ifdef ENABLE_DEBUG_FLAG
	{double x, y, z, numX, numY, denum;

	//Projection cost : (PiBC-u)^2
	double projectionCost = 0.0;
	for (int ii = 0; ii < nData; ii++)
	{
		//X = iB*C
		x = 0.0, y = 0.0, z = 0.0;
		for (int jj = 0; jj < nResamples; jj++)
		{
			x += iBData[jj + ii*nResamples] * C[jj],
				y += iBData[jj + ii*nResamples] * C[jj + nResamples],
				z += iBData[jj + ii*nResamples] * C[jj + 2 * nResamples];
		}

		numX = PmatData[12 * ii] * x + PmatData[12 * ii + 1] * y + PmatData[12 * ii + 2] * z + PmatData[12 * ii + 3];
		numY = PmatData[12 * ii + 4] * x + PmatData[12 * ii + 5] * y + PmatData[12 * ii + 6] * z + PmatData[12 * ii + 7];
		denum = PmatData[12 * ii + 8] * x + PmatData[12 * ii + 9] * y + PmatData[12 * ii + 10] * z + PmatData[12 * ii + 11];

		projectionCost += (pow(numX / denum - pt2dData[ii].x, 2) + pow(numY / denum - pt2dData[ii].y, 2));
	}

	//Motion prior in DCT form: CT*W*C
	double MotionPrior = 0.0;
	for (int ii = 0; ii < nResamples; ii++)
		MotionPrior += pow(sqrtWeight[ii], 2)*(pow(C[ii], 2) + pow(C[ii + nResamples], 2) + pow(C[ii + 2 * nResamples], 2));

#pragma omp critical
	printf("(DCT after: Motion cost, projection cost, Totalcost): %.4e %.4e %.4e\n\n", MotionPrior, sqrt(projectionCost / nData), lamda1*projectionCost + lamda2*MotionPrior);
	}
#endif

	//Final curve: iB*C =  X
	for (int ii = 0; ii < 3; ii++)
	{
		Map<VectorXd> eX(X + nData*ii, nData);
		Map<VectorXd> eC(C + nResamples*ii, nResamples);
		eX = eiBData*eC;
	}

	for (int ii = 0; ii < nData; ii++)
		Traj3D[ii].pt3D.x = X[ii], Traj3D[ii].pt3D.y = X[ii + nData], Traj3D[ii].pt3D.z = X[ii + 2 * nData];

	delete[]X, delete[]iBData, delete[]BResampled, delete[]C;
	delete[]PmatData, delete[]TimeStampData, delete[]pt2dData;

	return;
}
int ResamplingOf3DTrajectoryDCTDriver(char *Path, vector<int> &SelectedCams, vector<double> &OffsetInfo, int PriorOrder, int startFrame, int stopFrame, int ntracks, double lamda1, double lamda2)
{
	char Fname[200]; FILE *fp = 0;

	//Read calib info
	int nCams = (int)SelectedCams.size();
	VideoData *VideoInfo = new VideoData[nCams];
	for (int camID = 0; camID < nCams; camID++)
		if (ReadVideoDataI(Path, VideoInfo[camID], SelectedCams[camID], startFrame, stopFrame) == 1)
			return 1;

	int frameID, id, npts;
	int nframes = max(MaxnFrames, stopFrame);

	double u, v;
	vector<int>VectorCamID, VectorFrameID;
	vector<double> AllError2D;
	ImgPtEle ptEle;
	vector<ImgPtEle> *PerCam_UV = new vector<ImgPtEle>[nCams*ntracks];

	printf("Get 2D ...");
	for (int camID = 0; camID < nCams; camID++)
	{
		for (int trackID = 0; trackID < ntracks; trackID++)
			PerCam_UV[camID*ntracks + trackID].reserve(stopFrame - startFrame + 1);

		sprintf(Fname, "%s/Track2D/%d.txt", Path, SelectedCams[camID]); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("Cannot load %s\n", Fname);
			return 1;
		}
		for (int trackID = 0; trackID < ntracks; trackID++)
		{
			fscanf(fp, "%d %d ", &id, &npts);
			if (id != trackID)
				printf("Problem at Point %d of Cam %d", id, camID);
			for (int pid = 0; pid < npts; pid++)
			{
				fscanf(fp, "%d %lf %lf ", &frameID, &u, &v);
				if (frameID < 0)
					continue;
				if (!VideoInfo[camID].VideoInfo[frameID].valid)
					continue; //camera not localized

				if (u > 0 && v > 0)
				{
					ptEle.pt2D.x = u, ptEle.pt2D.y = v, ptEle.frameID = frameID, ptEle.imWidth = VideoInfo[camID].VideoInfo[frameID].width, ptEle.imHeight = VideoInfo[camID].VideoInfo[frameID].height;
					LensCorrectionPoint(&ptEle.pt2D, VideoInfo[camID].VideoInfo[frameID].K, VideoInfo[camID].VideoInfo[frameID].distortion);
					PerCam_UV[camID*ntracks + id].push_back(ptEle);
				}
			}
		}
		fclose(fp);
	}

	printf("Get rays info ....");
	double P[12], AA[6], bb[2], ccT[3], dd[1];
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		int count = 0;
		for (int camID = 0; camID < nCams; camID++)
		{
			for (int frameID = 0; frameID < PerCam_UV[camID*ntracks + trackID].size(); frameID++)
			{
				int RealFrameID = PerCam_UV[camID*ntracks + trackID][frameID].frameID;

				for (int kk = 0; kk < 12; kk++)
				{
					P[kk] = VideoInfo[camID].VideoInfo[RealFrameID].P[kk];
					PerCam_UV[camID*ntracks + trackID][frameID].P[kk] = P[kk];
				}

				for (int kk = 0; kk < 9; kk++)
					PerCam_UV[camID*ntracks + trackID][frameID].K[kk] = VideoInfo[camID].VideoInfo[RealFrameID].K[kk],
					PerCam_UV[camID*ntracks + trackID][frameID].R[kk] = VideoInfo[camID].VideoInfo[RealFrameID].R[kk];

				for (int kk = 0; kk < 3; kk++)
					PerCam_UV[camID*ntracks + trackID][frameID].camcenter[kk] = VideoInfo[camID].VideoInfo[RealFrameID].camCenter[kk];

				AA[0] = P[0], AA[1] = P[1], AA[2] = P[2], bb[0] = P[3];
				AA[3] = P[4], AA[4] = P[5], AA[5] = P[6], bb[1] = P[7];
				ccT[0] = P[8], ccT[1] = P[9], ccT[2] = P[10], dd[0] = P[11];

				PerCam_UV[camID*ntracks + trackID][frameID].Q[0] = AA[0] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.x*ccT[0],
					PerCam_UV[camID*ntracks + trackID][frameID].Q[1] = AA[1] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.x*ccT[1],
					PerCam_UV[camID*ntracks + trackID][frameID].Q[2] = AA[2] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.x*ccT[2];
				PerCam_UV[camID*ntracks + trackID][frameID].Q[3] = AA[3] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.y*ccT[0],
					PerCam_UV[camID*ntracks + trackID][frameID].Q[4] = AA[4] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.y*ccT[1],
					PerCam_UV[camID*ntracks + trackID][frameID].Q[5] = AA[5] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.y*ccT[2];
				PerCam_UV[camID*ntracks + trackID][frameID].u[0] = dd[0] * PerCam_UV[camID*ntracks + trackID][frameID].pt2D.x - bb[0],
					PerCam_UV[camID*ntracks + trackID][frameID].u[1] = dd[0] * PerCam_UV[camID*ntracks + trackID][frameID].pt2D.y - bb[1];

				PerCam_UV[camID*ntracks + trackID][frameID].pt3D = Point3d(count, count, count);//Interestingly, Ceres does not work if all the input are the same
				count++;
			}
		}
	}

	//Initialize data for optim
	vector<int> PointsPerTrack;
	vector<int *> PerTrackFrameID(ntracks);
	vector<double*> All3D(ntracks);
	int ntimeinstances, maxntimeinstances = 0;
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		ntimeinstances = 0;
		for (int camID = 0; camID < nCams; camID++)
			ntimeinstances += PerCam_UV[camID*ntracks + trackID].size();

		if (maxntimeinstances < ntimeinstances)
			maxntimeinstances = ntimeinstances;

		PerTrackFrameID[trackID] = new int[ntimeinstances];
		All3D[trackID] = new double[3 * ntimeinstances];
	}

	vector<int> VisCamID, VisLocalFrameID;
	vector<double> TimeStamp;
	vector<Point3d> T3D;
	vector<ImgPtEle> *Traj3D = new vector<ImgPtEle>[ntracks];
	int cID, fID;
	double x, y, z, t;
	int dummy[10000];
	double ts[10000];
	ImgPtEle iele;

	printf("Get 3D data:\n");
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		TimeStamp.clear(), T3D.clear(), VisCamID.clear(), VisLocalFrameID.clear();

		sprintf(Fname, "%s/OptimizedRaw_Track_%d.txt", Path, trackID); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("Cannot load %s\n", Fname);
			break;
		}
		while (fscanf(fp, "%lf %lf %lf %lf %d %d", &x, &y, &z, &t, &cID, &fID) != EOF)
		{
			TimeStamp.push_back(t), VisCamID.push_back(cID), VisLocalFrameID.push_back(fID);
			T3D.push_back(Point3d(x, y, z));
		}
		fclose(fp);

		for (int ii = 0; ii < (int)TimeStamp.size(); ii++)
			ts[ii] = TimeStamp[ii], dummy[ii] = ii;
		Quick_Sort_Double(ts, dummy, 0, (int)TimeStamp.size() - 1);

		for (int ii = 0; ii < (int)TimeStamp.size(); ii++)
		{
			int id = dummy[ii], camID = VisCamID[id], frameID = VisLocalFrameID[id];
			iele.viewID = camID, iele.frameID = frameID, iele.timeStamp = TimeStamp[id];

			iele.pt3D = T3D[id];
			for (int fid = 0; fid < (int)PerCam_UV[camID*ntracks + trackID].size(); fid++)
			{
				if (PerCam_UV[camID*ntracks + trackID][fid].frameID == frameID)
				{
					iele.pt2D = PerCam_UV[camID*ntracks + trackID][fid].pt2D;
					for (int jj = 0; jj < 12; jj++)
						iele.P[jj] = PerCam_UV[camID*ntracks + trackID][fid].P[jj];
					break;
				}
			}
			Traj3D[trackID].push_back(iele);
		}
	}

	printf("DCT-based resampling:\n");
	//for (int trial = 0; trial <= 50; trial++)
	//{
	//	lamda1 = 0.1 + 0.9 / 50 * trial; lamda2 = 1.0 - lamda2;

	omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
#pragma omp critical
		printf("%d: ", trackID);

		double earliest = Traj3D[trackID][0].timeStamp, latest = Traj3D[trackID].back().timeStamp;
		int nData = (int)Traj3D[trackID].size();
		double resamplingStep = (latest - earliest) / nData;

		double lamda2_ = lamda2*resamplingStep;
		ResamplingOf3DTrajectoryDCT(Traj3D[trackID], PriorOrder, true, resamplingStep, lamda1, lamda2_, true);

#pragma omp critical
		{
			sprintf(Fname, "%s/DCTResampled_Track_%d.txt", Path, trackID); remove(Fname);
			fp = fopen(Fname, "w+");
			for (int ii = 0; ii < (int)Traj3D[trackID].size(); ii++)
				fprintf(fp, "%.4f %.4f %.4f %.4f %d %d\n", Traj3D[trackID][ii].pt3D.x, Traj3D[trackID][ii].pt3D.y, Traj3D[trackID][ii].pt3D.z, Traj3D[trackID][ii].timeStamp, Traj3D[trackID][ii].viewID, Traj3D[trackID][ii].frameID);
			fclose(fp);
		}
	}
	//}
	printf("Done!\n\n");

	delete[]VideoInfo;
	delete[]PerCam_UV;
	delete[]Traj3D;

	return 0;
}

int GenerateCheckerBoardFreeImage(char *Path, int camID, int npts, int startFrame, int stopFrame)
{
	char Fname[200];

	int width = 0, height = 0;
	Mat cvImg;
	for (int fid = startFrame; fid <= stopFrame; fid++)
	{
		sprintf(Fname, "%s/%d/%d.png", Path, camID, fid); cvImg = imread(Fname, 0);
		if (!cvImg.empty())
		{
			width = cvImg.cols, height = cvImg.rows;
			break;
		}
	}

	if (width == 0 || height == 0)
	{
		printf("Found no images\n");
		return 1;
	}

	int *PixelCount = new int[width*height];
	double *AvgImage = new double[width*height];
	for (int ii = 0; ii < width*height; ii++)
		PixelCount[ii] = 0, AvgImage[ii] = 0.0;

	float x, y;
	int maxX, minX, maxY, minY;
	for (int fid = startFrame; fid <= stopFrame; fid++)
	{
		sprintf(Fname, "%s/%d/Corner/CV2_%d.txt", Path, camID, fid); 	FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
			continue;

		maxX = 0, minX = width, maxY = 0, minY = height;
		for (int ii = 0; ii < npts; ii++)
		{
			fscanf(fp, "%f %f ", &x, &y);
			if (x > maxX)
				maxX = min((int)x, width - 1);
			if (x < minX)
				minX = max((int)x, 0);
			if (y > maxY)
				maxY = min((int)y, height - 1);
			if (y < minY)
				minY = max((int)y, 0);
		}

		sprintf(Fname, "%s/%d/%d.png", Path, camID, fid); cvImg = imread(Fname, 0);
		if (!cvImg.empty())
			continue;

		for (int jj = 0; jj < height; jj++)
		{
			for (int ii = 0; ii < width; ii++)
			{
				if (ii>minX && ii <maxX && jj>minY && jj < maxY)
					continue;
				AvgImage[ii + jj*width] += (double)(int)cvImg.data[ii + jj*width];
				PixelCount[ii + jj*width]++;
			}
		}
	}


	for (int ii = 0; ii < width*height; ii++)
		AvgImage[ii] = AvgImage[ii] / PixelCount[ii];

	sprintf(Fname, "%s/Corpus/", Path); makeDir(Fname);
	sprintf(Fname, "%s/Corpus/%d.png", Path, camID);
	SaveDataToImage(Fname, AvgImage, width, height, 1);

	delete[]PixelCount, delete[]AvgImage;
	return 0;
}
int CheckerBoardMultiviewSpatialTemporalCalibration(char *Path, int nCams, int startFrame, int stopFrame, int STCalibration)
{
	const double square_size = 50.8;
	const int width = 1920, height = 1080, bh = 8, bw = 11, npts = bh*bw, TemporalSearchRange = 30, LossType = 0; //Huber loss
	bool fixIntrinsic = true, fixDistortion = true, fixPose = false, fixfirstCamPose = true, distortionCorrected = false;

	//Input Parameters
	const int PriorOrder = 1, motionPriorPower = 2, SearchRange = 10;
	double lamdaData = 0.85, lamdaPrior = 1.0 - lamdaData, SearchStep = 0.1;

	if (STCalibration == 1)
	{
		double startTime = omp_get_wtime();

		/*printf("Estimate camera instrinsic parameters: ");
		int sampleCalibFrameStep = 1;
		if (stopFrame - startFrame > 100)
		sampleCalibFrameStep = (stopFrame - startFrame) / 50;

		omp_set_num_threads(omp_get_max_threads());
		#pragma omp parallel for
		for (int camID = 0; camID < nCams; camID++)
		SingleCameraCalibration(Path, camID, startFrame, stopFrame, bw, bh, true, sampleCalibFrameStep, square_size, 1, width, height);

		printf("Generating checkerboard free images ... ");
		omp_set_num_threads(omp_get_max_threads());
		#pragma omp parallel for
		for (int camID = 0; camID < nCams; camID++)
		GenerateCheckerBoardFreeImage(Path, camID, bh*bw, startFrame, stopFrame);
		printf("Done. \n");

		printf("Create 2D trajectories from detected checkerboard corners ...");
		for (int camID = 0; camID < nCams; camID++)
		CleanCheckBoardDetection(Path, camID, startFrame, stopFrame);
		printf("Done. \n");*/

		double FrameLevelOffsetInfo[MaxnCams];
		GeometricConstraintSyncDriver(Path, nCams, npts, startFrame, stopFrame, 3 * SearchRange, true, FrameLevelOffsetInfo, false);

		//Convert between time-delay format and time-stamp format
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

		//Resort the time with the earliest camera placed first
		vector<int> SortedSelectedCamera;
		vector<double> SortedTimeStampInfoVector = TimeStampInfoVector;
		sort(SortedTimeStampInfoVector.begin(), SortedTimeStampInfoVector.end());
		for (int ii = 0; ii < (int)SelectedCamera.size(); ii++)
		{
			for (int jj = 0; jj < (int)SelectedCamera.size(); jj++)
			{
				if (TimeStampInfoVector[jj] == SortedTimeStampInfoVector[ii])
				{
					SortedSelectedCamera.push_back(SelectedCamera[jj]);
					break;
				}
			}
		}
		for (int ii = (int)SortedSelectedCamera.size() - 1; ii >= 0; ii--)
			SortedTimeStampInfoVector[ii] -= SortedTimeStampInfoVector[0];

		//Triangulate trajectories
		TrajectoryTriangulation(Path, SortedSelectedCamera, SortedTimeStampInfoVector, npts, startFrame, stopFrame, lamdaData, motionPriorPower);
		//ResamplingOf3DTrajectorySplineDriver(Path, SortedSelectedCamera, SortedTimeStampInfoVector, startFrame, stopFrame, npts, lamdaData);
		ResamplingOf3DTrajectoryDCTDriver(Path, SortedSelectedCamera, SortedTimeStampInfoVector, PriorOrder, startFrame, stopFrame, npts, lamdaData, lamdaPrior);

		sprintf(Fname, "%s/FMotionPriorSync.txt", Path);	fp = fopen(Fname, "w+");
		for (int ii = 0; ii < (int)SortedSelectedCamera.size(); ii++)
			fprintf(fp, "%d %.4f\n", SortedSelectedCamera[ii], SortedTimeStampInfoVector[ii]);
		fclose(fp);

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
			sortedSelectedCamera[ii] = SelectedCamera[ii],
			sortedTimeStampInfoVector[ii] = TimeStampInfoVector[ii];

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

	visualizationDriver(Path, nCams, startFrame, stopFrame, true, false, true, false, false, true, 0);
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
					PerCamUV_All[camID][frameID].R[kk] = AllVideoInfo.VideoInfo[camID*nframes + frameID].R[kk];
				for (int kk = 0; kk < 3; kk++)
					PerCamUV_All[camID][frameID].camcenter[kk] = AllVideoInfo.VideoInfo[camID*nframes + frameID].camCenter[kk];

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
		GeometricConstraintSyncDriver(Path, nCams, npts, startFrame, stopFrame, SearchRange, true, FrameLevelOffsetInfo, false);

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

int SimulateRollingShutterCameraAnd2DPointsForMoCap(char *Path, int npts, double *Intrinsic, double *distortion, int width, int height, double radius = 5e3, bool saveGT3D = true, bool show2DImage = false, double Noise2D = 2.0)
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

	/*sprintf(Fname, "%s/RollingShutterPose", Path), makeDir(Fname);
	for (int pid = 0; pid < (int)allXYZ.size(); pid++)
	{
	sprintf(Fname, "%s/RollingShutterPose/CamPose_%d.txt", Path, pid); fp = fopen(Fname, "w+");
	for (int fid = 0; fid < (int)PerCam_UV[pid].size(); fid++)
	{
	double Rgl[16], C[3];
	GetRCGL(PerCam_UV[pid][fid].R, PerCam_UV[pid][fid].T, Rgl, C);
	fprintf(fp, "%.8f ", PerCam_UV[pid][fid].frameID + PerCam_UV[pid][fid].pt2D.y / height);
	for (int ii = 0; ii < 16; ii++)
	fprintf(fp, "%.16f ", Rgl[ii]);
	for (int ii = 0; ii < 3; ii++)
	fprintf(fp, "%.16f ", C[ii]);
	fprintf(fp, "\n");
	}
	fclose(fp);
	}*/

	sprintf(Fname, "%s/RollingShutterPose", Path), makeDir(Fname);
	sprintf(Fname, "%s/RollingShutterPose/CamPose.txt", Path); fp = fopen(Fname, "w+");
	for (int pid = 0; pid < (int)allXYZ.size(); pid++)
	{
		for (int fid = 0; fid < (int)PerCam_UV[pid].size(); fid++)
		{
			double twist[6];  convertRTToTwist(PerCam_UV[pid][fid].R, PerCam_UV[pid][fid].T, twist);
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
			for (int fid = 0; fid < (int)PerCam_UV[pid].size(); fid++)
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
			for (int fid = 0; fid < (int)PerCam_UV[pid].size(); fid++)
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
	//SimulateRollingShutterCameraAnd2DPointsForMoCap(Path, npts, Intrinsic, distortion, width, height, radius, 1, 1, noise2D);

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

int TrackLocalizedCameraSIFT(char *Path, int viewID, int startF, int bw, Point2d scaleThresh, double backforeThresh2, int backward)
{
	char Fname[200];

	int pid; double s;
	Point2f uv; Point3d xyz;

	vector<int> ThreeDiD3D;
	vector<Point2f> uv3D;
	vector<Point3d> xyz3D;
	vector<float> scale3D;

	sprintf(Fname, "%s/%d/Inliers_3D2D_%d.txt", Path, viewID, startF); FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return 1;
	}

	while (fscanf(fp, "%d %lf %lf %lf %f %f %lf", &pid, &xyz.x, &xyz.y, &xyz.z, &uv.x, &uv.y, &s) != EOF)
	{
		if (s > scaleThresh.x || s < scaleThresh.y)
			continue;

		ThreeDiD3D.push_back(pid);
		uv3D.push_back(uv);
		xyz3D.push_back(xyz);
		scale3D.push_back(s);
	}
	fclose(fp);


	Mat RefImg, PreImg, NewImg;
	vector<float> err;
	vector<uchar> status;
	vector<Point2f> prePt, newPt, backPt;

	Size subPixWinSize(21, 21), winSize(31, 31);
	TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);

	sprintf(Fname, "%s/%d/%d.png", Path, viewID, startF); RefImg = imread(Fname, 0);
	if (RefImg.empty())
	{
		printf("Cannot load %s\n", Fname);
		return 1;
	}
	PreImg = RefImg;

	vector<Point2f> *BackTrackUV = 0, *ForeTrackUV = new vector<Point2f>[uv3D.size()];
	for (int ii = 0; ii < (int)uv3D.size(); ii++)
		ForeTrackUV[ii].reserve(bw), ForeTrackUV[ii].push_back(uv3D[ii]);

	for (int ii = 1; ii <= bw; ii++)
	{
		printf("%d ...", ii);
		sprintf(Fname, "%s/%d/%d.png", Path, viewID, ii + startF); NewImg = imread(Fname, 0);
		if (NewImg.empty())
			break;

		for (int pid = 0; pid < (int)uv3D.size(); pid++)
		{
			if ((int)ForeTrackUV[pid].size() != ii)
				continue;

			err.clear(), status.clear(), prePt.clear(), newPt.clear();
			prePt.push_back(ForeTrackUV[pid].back());
			newPt.push_back(ForeTrackUV[pid].back());

			calcOpticalFlowPyrLK(PreImg, NewImg, prePt, newPt, status, err, winSize, 3, termcrit, 0, 0.001);

			if (status[0])
			{
				//Track again from the ref image to avoid drift
				err.clear(), status.clear(), prePt.clear();
				prePt.push_back(ForeTrackUV[pid][0]);

				calcOpticalFlowPyrLK(RefImg, NewImg, prePt, newPt, status, err, winSize, 3, termcrit, 0, 0.001);

				//forward-backward consistency
				if (status[0])
				{
					err.clear(), status.clear(), backPt.clear();
					backPt.push_back(ForeTrackUV[pid][0]);

					calcOpticalFlowPyrLK(NewImg, RefImg, newPt, backPt, status, err, winSize, 3, termcrit, 0, 0.001);
					if (status[0])
					{
						double dist2 = pow(backPt[0].x - ForeTrackUV[pid][0].x, 2) + pow(backPt[0].y - ForeTrackUV[pid][0].y, 2);
						if (dist2 < backforeThresh2)
							ForeTrackUV[pid].push_back(newPt.back());
					}
				}
			}
		}
		PreImg = NewImg;
	}

	if (backward)
	{
		BackTrackUV = new vector<Point2f>[uv3D.size()];
		for (int ii = 0; ii < (int)uv3D.size(); ii++)
			BackTrackUV[ii].reserve(bw), BackTrackUV[ii].push_back(uv3D[ii]);

		PreImg = RefImg;
		for (int ii = -1; ii >= -bw; ii--)
		{
			printf("%d ...", ii);
			sprintf(Fname, "%s/%d/%d.png", Path, viewID, ii + startF); NewImg = imread(Fname, 0);
			if (NewImg.empty())
				break;

			for (int pid = 0; pid < (int)uv3D.size(); pid++)
			{
				if ((int)BackTrackUV[pid].size() != -ii)
					continue;

				err.clear(), status.clear(), prePt.clear(), newPt.clear();
				prePt.push_back(BackTrackUV[pid].back());
				newPt.push_back(BackTrackUV[pid].back());

				calcOpticalFlowPyrLK(PreImg, NewImg, prePt, newPt, status, err, winSize, 3, termcrit, 0, 0.001);

				if (status[0])
				{
					//Track again from the ref image
					err.clear(), status.clear(), prePt.clear();
					prePt.push_back(BackTrackUV[pid][0]);

					calcOpticalFlowPyrLK(RefImg, NewImg, prePt, newPt, status, err, winSize, 3, termcrit, 0, 0.001);

					//forward-backward consistency
					if (status[0])
					{
						err.clear(), status.clear(), backPt.clear();
						backPt.push_back(BackTrackUV[pid][0]);

						calcOpticalFlowPyrLK(NewImg, RefImg, newPt, backPt, status, err, winSize, 3, termcrit, 0, 0.001);
						if (status[0])
						{
							double dist2 = pow(backPt[0].x - BackTrackUV[pid][0].x, 2) + pow(backPt[0].y - BackTrackUV[pid][0].y, 2);
							if (dist2 < backforeThresh2)
								BackTrackUV[pid].push_back(newPt.back());
						}
					}
				}
			}
			PreImg = NewImg;
		}
	}

	//Write data
	for (int ii = 1; ii <= bw; ii++)
	{
		sprintf(Fname, "%s/%d/Inliers_3D2D_%d.txt", Path, viewID, startF + ii); FILE *fp = fopen(Fname, "a+");
		for (int pid = 0; pid < (int)uv3D.size(); pid++)
		{
			if ((int)ForeTrackUV[pid].size() >= ii)
			{
				int i = ThreeDiD3D[pid];
				double x = xyz3D[pid].x, y = xyz3D[pid].y, z = xyz3D[pid].z, u = ForeTrackUV[pid][ii].x, v = ForeTrackUV[pid][ii].y, s = scale3D[pid];
				fprintf(fp, "%d %.6f %.6f %.6f %.4f %.4f %.2f\n", ThreeDiD3D[pid], xyz3D[pid].x, xyz3D[pid].y, xyz3D[pid].z, ForeTrackUV[pid][ii].x, ForeTrackUV[pid][ii].y, scale3D[pid]);
			}
		}
		fclose(fp);
	}

	if (backward)
	{
		for (int ii = -1; ii > -bw; ii--)
		{
			sprintf(Fname, "%s/%d/Inliers_3D2D_%d.txt", Path, viewID, startF + ii); FILE *fp = fopen(Fname, "a+");
			for (int pid = 0; pid < (int)uv3D.size(); pid++)
			{
				if ((int)BackTrackUV[pid].size() >= -ii)
					fprintf(fp, "%d %.6f %.6f %.6f %.4f %.4f %.2f\n", ThreeDiD3D[pid], xyz3D[pid].x, xyz3D[pid].y, xyz3D[pid].z, BackTrackUV[pid][ii].x, BackTrackUV[pid][ii].y, scale3D[pid]);
			}
			fclose(fp);
		}
	}

	delete[]BackTrackUV, delete[]ForeTrackUV;

	return 0;
}

int TestPnP(char *Path, int camID, int nCams, int frameID, double thresh = 5.0)
{
	int nCameras = 200;

	char Fname[200];
	CameraData *AllCamsInfo = new CameraData[nCams];
	if (!ReadIntrinsicResults(Path, AllCamsInfo))
		return 0;

	int id, npts, ninliers;
	double x, y, z, u, v, s, residuals[2];
	vector<Point3d> t3D; t3D.reserve(3000);
	vector<Point2d> uv; uv.reserve(3000);
	vector<double> scale; scale.reserve(3000);
	vector<bool> Good; Good.reserve(3000);

	sprintf(Fname, "%s/%d/Inliers_3D2D_%d.txt", Path, camID, frameID);
	FILE *fp = fopen(Fname, "r");
	while (fscanf(fp, "%d %lf %lf %lf %lf %lf %lf ", &id, &x, &y, &z, &u, &v, &s) != EOF)
	{
		t3D.push_back(Point3d(x, y, z));
		uv.push_back(Point2d(u, v));
		scale.push_back(s);
	}
	fclose(fp);
	npts = t3D.size();

	//Test if 3D is correct
	Mat cvpts(npts, 2, CV_32F), cv3D(npts, 3, CV_32F);
	for (int ii = 0; ii < npts; ii++)
	{
		cvpts.at<float>(ii, 0) = uv[ii].x, cvpts.at<float>(ii, 1) = uv[ii].y;
		cv3D.at<float>(ii, 0) = t3D[ii].x, cv3D.at<float>(ii, 1) = t3D[ii].y, cv3D.at<float>(ii, 2) = t3D[ii].z;
	}

	Mat cvK = Mat(3, 3, CV_32F), rvec(1, 3, CV_32F), tvec(1, 3, CV_32F);
	for (int ii = 0; ii < 9; ii++)
		cvK.at<float>(ii) = (float)AllCamsInfo[camID].K[ii];

	Mat Inliers;
	double ProThresh = 0.995, PercentInlier = 0.4;
	int iterMax = (int)(log(1.0 - ProThresh) / log(1.0 - pow(PercentInlier, 4)) + 0.5); //log(1-eps) / log(1 - (inlier%)^min_pts_requires)
	solvePnPRansac(cv3D, cvpts, cvK, Mat(), rvec, tvec, false, iterMax, thresh, npts*PercentInlier, Inliers, CV_EPNP);// CV_ITERATIVE);

	ninliers = Inliers.rows;
	printf("With pnp: (%d/%d)\n", ninliers, npts);
	cout << rvec << endl << tvec << endl;

	for (int ii = 0; ii < 3; ii++)
		AllCamsInfo[camID].rt[ii] = rvec.at<double>(ii);
	for (int ii = 0; ii < 3; ii++)
		AllCamsInfo[camID].rt[ii + 3] = tvec.at<double>(ii);

	AllCamsInfo[camID].threshold = thresh * 2;
	if (CameraPose_GSBA(Path, AllCamsInfo[camID], t3D, uv, scale, Good, 1, 1, 1, true))
		//if (CameraPose_RSBA(Path, AllCamsInfo[camID], t3D, uv, scale, Good, 1, 1, 1, true))
		return -1;

	ninliers = 0;
	for (int ii = 0; ii < npts; ii++)
	{
		PinholeReprojectionDebug(AllCamsInfo[camID].intrinsic, AllCamsInfo[camID].rt, uv[ii], t3D[ii], residuals);
		//CayleyReprojectionDebug(AllCamsInfo[camID].intrinsic, AllCamsInfo[camID].rt, AllCamsInfo[camID].wt, uv[ii], t3D[ii], AllCamsInfo[camID].width, AllCamsInfo[camID].height, residuals);
		if (abs(residuals[0]) < thresh && abs(residuals[1]) < thresh)
			ninliers++;
	}
	for (int ii = 0; ii < 6; ii++)
		printf("%f ", AllCamsInfo[camID].rt[ii]);
	printf("\n");
	for (int ii = 0; ii < 6; ii++)
		printf("%f ", AllCamsInfo[camID].wt[ii]);
	printf("\n");
	printf("With BA: (%d/%d)\n", ninliers, npts);

	double rt_gt[6] = { 0.2240511253258971, -0.0428236581421133, 0.0369851631942179, -3.3586379214701299, 0.9992682352604219, -3.6783490920648232 };
	double wt_gt[6] = { 0.0003875752556760, 0.0177087294372213, 0.0054402561932053, -0.1244087467843000, -0.0414196631281985, 0.1433284329690574 };
	for (int ii = 0; ii < 6; ii++)
		AllCamsInfo[camID].rt[ii] = rt_gt[ii], AllCamsInfo[camID].wt[ii] = wt_gt[ii];

	ninliers = 0;
	for (int ii = 0; ii < npts; ii++)
	{
		//LensCorrectionPoint();
		//CayleyDistortionReprojectionDebug(AllCamsInfo[camID].intrinsic, AllCamsInfo[camID].distortion, AllCamsInfo[camID].rt, AllCamsInfo[camID].wt, uv[ii], t3D[ii], AllCamsInfo[camID].width, AllCamsInfo[camID].height, residuals);
		CayleyReprojectionDebug(AllCamsInfo[camID].intrinsic, AllCamsInfo[camID].rt, AllCamsInfo[camID].wt, uv[ii], t3D[ii], AllCamsInfo[camID].width, AllCamsInfo[camID].height, residuals);
		if (abs(residuals[0]) < thresh && abs(residuals[1]) < thresh)
			ninliers++;
	}
	printf("With groundtruth: (%d/%d)", ninliers, npts);

	return 0;
}
int TestPnP2(char *Path, int camID, double thresh = 5.0)
{
	char Fname[200];

	Corpus corpusData;
	sprintf(Fname, "%s/Corpus", Path);
	ReadCorpusInfo(Fname, corpusData, false, true);

	int npts = corpusData.threeDIdAllViews[camID].size(),
		npts2 = corpusData.uvAllViews[camID].size();

	vector<Point2d>uv;
	vector<Point3d> xyz;
	for (int ii = 0; ii < npts; ii++)
	{
		int p3Did = corpusData.threeDIdAllViews[camID][ii];
		xyz.push_back(corpusData.xyz[p3Did]);
		uv.push_back(corpusData.uvAllViews[camID][ii]);
	}

	int ninliers = 0;
	double residuals[2];
	for (int ii = 0; ii < npts; ii++)
	{
		CayleyReprojectionDebug(corpusData.camera[camID].intrinsic, corpusData.camera[camID].rt, corpusData.camera[camID].wt, uv[ii], xyz[ii], corpusData.camera[camID].width, corpusData.camera[camID].height, residuals);
		//PinholeReprojectionDebug(corpusData.camera[camID].intrinsic, corpusData.camera[camID].rt, uv[ii], xyz[ii], residuals);
		if (abs(residuals[0]) < thresh && abs(residuals[1]) < thresh)
			ninliers++;
	}
	printf("With groundtruth: (%d/%d)", ninliers, npts);

	sprintf(Fname, "%s/%d/Inliers_3D2D_%d_.txt", Path, 1, 1); FILE *fp = fopen(Fname, "w+");
	for (int ii = 0; ii < npts; ii++)
		fprintf(fp, "%d %f %f %f %f %f %.2f\n", ii, xyz[ii].x, xyz[ii].y, xyz[ii].z, uv[ii].x, uv[ii].y, 1.0);
	fclose(fp);

	return 0;
}
int main(int argc, char** argv)
{
	srand(0);//srand(time(NULL));
	char Path[] = "E:/ARTag", Fname[200], Fname2[200];

	//return 0;
	//TrackLocalizedCameraSIFT(Path, 0, 16, 15, Point2d(40.0, 1.0), 3.0, 1);
	//TrackOpenCVLK(Path, 270, 400);
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
	//return 0;
	int mode = atoi(argv[1]);
	if (mode == 0) //Step 1: sync all sequences with audio
	{
#ifdef _WINDOWS
		int computeSync = 0; //atoi(argv[2]);
		if (computeSync == 1)
		{
			int pair = atoi(argv[3]);
			int minSegLength = 3e6;
			if (pair == 1)
			{
				int srcID1 = atoi(argv[4]),
					srcID2 = atoi(argv[5]);
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

				const int nvideos = 8;
				int camID[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
				double fps[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };//to avoid drift at this stage
				double offset, ZNCC;


				sprintf(Fname, "%s/audioSync.txt", Path); 	FILE *fp = fopen(Fname, "w+");
				for (int jj = 0; jj < nvideos - 1; jj++)
				{
					for (int ii = jj + 1; ii < nvideos; ii++)
					{
						sprintf(Fname, "%s/%d/audio.wav", Path, camID[jj]);
						sprintf(Fname2, "%s/%d/audio.wav", Path, camID[ii]);
						if (SynAudio(Fname, Fname2, fps[jj], fps[ii], minSegLength, offset, ZNCC, minZNCC) != 0)
						{
							printf("Between %d and %d: not succeed\n", camID[jj], camID[ii]);
							fprintf(fp, "%d %d %.4f\n", camID[jj], camID[ii], 1000.0*rand());
						}
						else
							fprintf(fp, "%d %d %.4f\n", camID[jj], camID[ii], -offset);
					}
				}
				fclose(fp);

				PrismMST(Path, "audioSync", nvideos);
				AssignOffsetFromMST(Path, "audioSync", nvideos);
			}
		}
		else
		{
			//int writeSyncedImages = atoi(argv[3]);
			int nvideos = atoi(argv[2]);
			ShowSyncLoad(Path, "FaudioSync", Path, nvideos);
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
		int ShutterModel = atoi(argv[2]);
		RefineVisualSfM(Path, ShutterModel, 10.0);
		//ReCalibratedFromGroundTruthCorrespondences(Path, 1, 0, 161, 88, ShutterModel);

		sprintf(Fname, "%s/Corpus", Path);
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
		int nviews = atoi(argv[2]),
			NPplus = atoi(argv[3]),
			mode = atoi(argv[4]);

		if (mode == 0)
		{
			int ShutterModel = atoi(argv[5]),
				reTriangulate = atoi(argv[6]);
			double threshold = atof(argv[7]);
			bool sharedIntrinisc = true, fixIntrinsic = false, fixDistortion = false, fixPose = false, distortionCorrected = false, doubleRefinement = true;

			sprintf(Fname, "%s/Corpus", Path);
			if (reTriangulate == 0)
				RefineVisualSfMAndCreateCorpus(Fname, nviews, NPplus, ShutterModel, threshold, sharedIntrinisc, fixIntrinsic, fixDistortion, fixPose, true, distortionCorrected, doubleRefinement);
			else
			{
				RefineVisualSfM(Fname, nviews, NPplus, ShutterModel, threshold, sharedIntrinisc, fixIntrinsic, fixDistortion, fixPose, true, distortionCorrected, doubleRefinement);

				//Remember to clean up the match.txt file before running match table
				GenerateMatchingTableVisualSfM(Fname, nviews);
				BuildCorpusVisualSfm(Fname, distortionCorrected, ShutterModel, sharedIntrinisc, NPplus, 0);
			}
			visualizationDriver(Path, 1, -1, -1, true, false, false, false, false, false, -1);
		}
		else if (mode == 1)
		{
			int distortionCorrected = 0,// atoi(argv[5]),
				HistogramEqual = 0,//atoi(argv[6]),
				OulierRemoveTestMethod = 2,//atoi(argv[7]), //Fmat is more prefered. USAC is much faster then 5-pts :(
				LensType = 0;// atoi(argv[8]);
			double ratioThresh = 0.8;// atof(argv[9]);

			int nCams = nviews, cameraToScan = -1;
			if (OulierRemoveTestMethod == 2)
				nCams = 1,//atoi(argv[10]),
				cameraToScan = 0;// atoi(argv[11]);

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
			int distortionCorrected = atoi(argv[5]),
				ShutterModel = atoi(argv[6]),
				SharedIntrinsic = atoi(argv[7]),
				LossType = atoi(argv[8]); //0: NULL, 1: Huber

			BuildCorpus(Fname, distortionCorrected, ShutterModel, SharedIntrinsic, NPplus, LossType);
			visualizationDriver(Path, 1, -1, -1, true, false, false, false, false, false, -1);
		}
		return 0;
	}
	else if (mode == 4) //Step 5: Get features data for test sequences
	{
		int nviews = atoi(argv[2]),
			selectedView = atoi(argv[3]),
			startF = atoi(argv[4]),
			stopF = atoi(argv[5]),
			increF = atoi(argv[6]),
			HistogramEqual = atoi(argv[7]);

		//int nviews = 1, selectedView = 0, startF =0, stopF = 0, increF = 1, HistogramEqual = 0;
		vector<int> availViews;
		if (selectedView < 0)
			for (int ii = 0; ii < nviews; ii++)
				availViews.push_back(ii);
		else
			availViews.push_back(selectedView);


		//if (!ReadIntrinsicResults(Path, corpusData.camera))
		//return 1;
		//LensCorrectionImageSequenceDriver(Fname, corpusData.camera[selectedView].K, corpusData.camera[selectedView].distortion, corpusData.camera[selectedView].LensModel, startF, stopF, 1.0, 1.0, 5);
		ExtractSiftGPUfromExtractedFrames(Path, availViews, startF, stopF, increF, HistogramEqual);
		return 0;
	}
	else if (mode == 5) //Step 6: Localize test sequence wrst to corpus
	{
		int startFrame = atoi(argv[2]),
			stopFrame = atoi(argv[3]),
			IncreFrame = atoi(argv[4]),
			seletectedCam = atoi(argv[5]),
			nCams = atoi(argv[6]), //ncameras used to scan to corpus
			module = atoi(argv[7]); //0: Matching, 1: Localize, 2+3: refine on video, -1: visualize

		//int startFrame = 1, stopFrame = 660, IncreFrame = 1, seletectedCam = 1, nCams = 8, module = 2;
		if (startFrame == 1 && module == 1)
		{
			sprintf(Fname, "%s/Intrinsic_%d.txt", Path, seletectedCam); FILE*fp = fopen(Fname, "w+");	fclose(fp);
			sprintf(Fname, "%s/CamPose_%d.txt", Path, seletectedCam); fp = fopen(Fname, "w+"); fclose(fp);
		}

		if (module < 0)
			visualizationDriver(Path, nCams, startFrame, stopFrame, true, false, true, false, false, false, startFrame);
		if (module == 0 || module == 1)
		{
			int distortionCorrected = atoi(argv[8]), //Must set to 1 after corpus matching with calibrated caminfo
				GetIntrinsicFromCorpus = atoi(argv[9]), sharedIntriniscOptim = atoi(argv[10]),
				LensType = atoi(argv[11]);// 0:RADIAL_TANGENTIAL_PRISM, 1: FISHEYE;

			//int distortionCorrected = 1, GetIntrinsicFromCorpus = 0, sharedIntriniscOptim = 1, LensType = 0;
			LocalizeCameraFromCorpusDriver(Path, startFrame, stopFrame, IncreFrame, module, nCams, seletectedCam, distortionCorrected, GetIntrinsicFromCorpus, sharedIntriniscOptim, LensType);
		}
		if (module == 2)
		{
			int fixIntrinsic = atoi(argv[8]), fixDistortion = atoi(argv[9]), fixPose = atoi(argv[10]),
				fixfirstCamPose = atoi(argv[11]), distortionCorrected = atoi(argv[12]), fix3D = 0;
			double threshold = atof(argv[13]);
			bool doubleRefinement = true;

			//double threshold = 5.0;
			//int fixIntrinsic = 0, fixDistortion = 0, fixPose = 0, fixfirstCamPose = 1, fix3D = 0, distortionCorrected = 1;
			VideoPose_RS_Cayley_BA(Path, seletectedCam, startFrame, stopFrame, fixIntrinsic, fixDistortion, fixPose, fixfirstCamPose, fix3D, distortionCorrected, doubleRefinement, threshold);
		}
		if (module == 3)
		{
			//double threshold = atof(argv[12]);
			//int fixIntrinsic = atoi(argv[8]), fixDistortion = atoi(argv[9]), fix3D = atoi(argv[10]), distortionCorrected = atoi(argv[11]);
			double threshold = 5.0;
			int  fixIntrinisc = 0, fixDistortion = 0, fixed3D = 0, distortionCorrected = 0; //Recomemded

			if (distortionCorrected == 1)
				fixIntrinisc = 1, fixDistortion = 1;
			VideoPose_GSBA(Path, seletectedCam, startFrame, stopFrame, fixIntrinisc, fixDistortion, fixed3D, distortionCorrected, threshold);
		}
		if (module == 4)
		{
			//double threshold = atof(argv[13]);
			//int fixIntrinsic = atoi(argv[8]), fixDistortion = atoi(argv[9]), fix3D = atoi(argv[10]), distortionCorrected = atoi(argv[11]), controlStep = atoi(argv[12]);
			double threshold = 5.0;
			int  fixIntrinisc = 0, fixDistortion = 0, fixed3D = 0, distortionCorrected = 0, controlStep = 3; //Recomemded

			VideoSplineRSBA(Path, startFrame, stopFrame, seletectedCam, distortionCorrected, false, false, threshold, controlStep);
			//Pose_se_BSplineInterpolation("E:/RollingShutter/_CamPoseS_0.txt", "E:/RollingShutter/__CamPose_0.txt", 100, "E:/RollingShutter/Pose.txt");
		}

		return 0;
	}
	else if (mode == 6) //step 7: generate 3d data from test sequences
	{
		int startFrame = 1,//atoi(argv[2]),
			stopFrame = 120,//atoi(argv[3]),
			timeStep = 1,// atoi(argv[4]),
			module = 0;// atoi(argv[5]); 1: matching, 0 : triangulation

		int HistogrameEqual = 0,
			distortionCorrected = 0,
			OulierRemoveTestMethod = 2, ninlierThesh = 40, //fmat test
			LensType = RADIAL_TANGENTIAL_PRISM,
			NViewPlus = 2,
			nviews = 2;
		double ratioThresh = 0.8, reprojectionThreshold = 5;

		int FrameOffset[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
		if (module == 1)
		{
			for (int timeID = startFrame; timeID <= stopFrame; timeID += timeStep)
			{
				GeneratePointsCorrespondenceMatrix_SiftGPU2(Path, nviews, timeID, HistogrameEqual, 0.6, FrameOffset);
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
		else
		{
			bool save2DCorres = false;
			double MaxDepthThresh = 99000; //mm
			Build3DFromSyncedImages(Path, nviews, startFrame, stopFrame, timeStep, LensType, distortionCorrected, NViewPlus, reprojectionThreshold, MaxDepthThresh, FrameOffset, save2DCorres, false, 10.0, false);
			visualizationDriver(Path, nviews, 0, 600, true, false, true, false, true, false, startFrame);
		}

		return 0;
	}
	else if (mode == 7)
	{
		int nVideoCams = 5,
			startFrame = 0, stopFrame = 199,
			LensModel = RADIAL_TANGENTIAL_PRISM;

		int selectedCams[] = { 1, 2 }, selectedTime[2] = { 0, 0 }, ChooseCorpusView1 = -1, ChooseCorpusView2 = -1;

		VideoData AllVideoInfo;
		ReadVideoData(Path, AllVideoInfo, nVideoCams, startFrame, stopFrame);

		double Fmat[9];
		int seletectedIDs[2] = { selectedCams[0] * max(MaxnFrames, stopFrame) + selectedTime[0], selectedCams[1] * max(MaxnFrames, stopFrame) + selectedTime[1] };
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
