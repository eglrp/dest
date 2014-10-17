#pragma once
#include <iostream>
#include <fstream>
#include <cstdio>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "libsndfile/include/sndfile.h"

#pragma comment(lib, "../libsndfile/lib/libsndfile-1.lib")
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

bool myImgReader(char *fname, char *Img, int &width, int &height, int nchannels);
bool GrabImage(char *fname, char *Img, int &width, int &height, int nchannels);
bool GrabVideoFrame2Mem(char *fname, char *Data, int &width, int &height, int &nchannels, int &nframes, int frameSample = 1, int fixnframes = 99999999);

int ReadAudio(char *Fin, Sequence &mySeq, char *Fout = 0);
int SynAudio(char *Fname1, char *Fname2, double fps1, double fps2, int MinSample, double &finalframeOffset, double reliableThreshold = 0.25);



