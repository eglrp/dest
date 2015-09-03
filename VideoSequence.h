#if !defined(SEQUENCE_H )
#define SEQUENCE_H
#pragma once

#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <opencv2/opencv.hpp>
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>

using namespace std;
using namespace cv;

class Sequence
{
public:
	int width, height, nchannels, nframes, nsamples, sampleRate;
	char *Img;
	float *Audio;
	double TimeAlignPara[2];

	void InitSeq(double fps, double offset)
	{
		TimeAlignPara[0] = fps,TimeAlignPara[1] = offset;
		Img = 0, Audio = 0;
		return ;
	}

	~Sequence()
	{
		delete []Img;
		delete []Audio;
	}
};

#ifdef _WIN32
#include "libsndfile/include/sndfile.h"
#pragma comment(lib, "../libsndfile/lib/libsndfile-1.lib")
#endif
int ReadAudio(char *Fin, Sequence &mySeq, char *Fout = 0);
int SynAudio(char *Fname1, char *Fname2, double fps1, double fps2, int MinSample, double &finalframeOffset, double &MaxZNCC, double reliableThreshold = 0.25);

bool GrabVideoFrame2Mem(char *fname, char *Data, int &width, int &height, int &nchannels, int &nframes, int frameSample = 1, int fixnframes = 99999999);

int PrismMST(char *Path, char *PairwiseSyncFilename, int nvideos);
int AssignOffsetFromMST(char *Path, char *PairwiseSyncFilename, int nvideos, double *OffsetInfo = 0, double *fps = 0);

void DynamicTimeWarping3Step(Mat pM, vector<int>&p, vector<int> &q);
void DynamicTimeWarping5Step(Mat pM, vector<int>&p, vector<int> &q);

#endif