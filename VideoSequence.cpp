#include "VideoSequence.h"
#include "Ultility.h"

using namespace cv;
using namespace std;

bool GrabImage(char *fname, char *Img, int &width, int &height, int nchannels)
{
	Mat view = imread(fname, nchannels==1?0:1);
	if(view.data == NULL)
	{
		cout<<"Cannot load: "<<fname<<endl;
		return false;
	}
	width = view.cols, height = view.rows;
	int length = width *height*nchannels;
	for(int ii=0; ii<length; ii++)
		Img[ii] = view.data[ii];

	return true;
}
bool GrabVideoFrame2Mem(char *fname, char *Data, int &width, int &height, int &nchannels, int &nframes, int frameSample, int fixnframes)
{
	IplImage  *frame = 0;
	CvCapture *capture = cvCaptureFromFile(fname );
	if( !capture ) 
		return false;	

	bool flag = false;
	int length, frameID = 0, frameID2 = 0;
	while(true && fixnframes > frameID)
	{
		IplImage  *frame = cvQueryFrame( capture );
		if( !frame ) 
		{
			cvReleaseImage(&frame);
			return true;
		}

		if(frameID ==0)
			width = frame->width, height = frame->height, nchannels = frame->nChannels, length = width*height*nchannels;

		for(int ii=0; ii<length; ii++)
			Data[ii+length*frameID] = frame->imageData[ii];

		frameID2++;
		if(frameID2 == frameSample)
			frameID++, frameID2= 0;
		nframes = frameID;	
	}

	//cvReleaseImage(&frame);
	cvReleaseCapture( &capture );

	return true;
}

int ReadAudio(char *Fin, Sequence &mySeq,char *Fout)
{   
	SNDFILE      *infile ;
	SF_INFO      sinfo ;

	int nchannels;
	if (! (infile = sf_open (Fin, SFM_READ, &sinfo)))
	{  
		printf ("Not able to open input file %s.\n", Fin) ;
		return  1 ;
	}
	else
	{
		mySeq.nsamples = (int)sinfo.frames, mySeq.sampleRate = (int)sinfo.samplerate, nchannels = (int)sinfo.channels;
		printf("Number of sample per channel=%d, Samplerate=%d, Channels=%d\n",mySeq.nsamples, mySeq.sampleRate, nchannels);
	}

	float *buf = (float *) malloc(mySeq.nsamples*nchannels*sizeof(float));
	int num = (int)sf_read_float(infile, buf, mySeq.nsamples*nchannels);

	//I want only 1 channel
	mySeq.Audio = new float[mySeq.nsamples];
	for (int i = 0; i < mySeq.nsamples; i++)
		mySeq.Audio[i] = buf[nchannels*i];
	delete []buf;

	if(Fout!= NULL)
		WriteGridBinary(Fout, mySeq.Audio, 1, mySeq.nsamples);

	return 0 ;
} 
int SynAudio(char *Fname1, char *Fname2, double fps1, double fps2, int MinSample, double &finalfoffset, double reliableThreshold)
{
	omp_set_num_threads(omp_get_max_threads());

	int ii, jj;
	Sequence Seq1, Seq2;
	Seq1.InitSeq(fps1, 0.0);
	Seq2.InitSeq(fps2, 0.0);
	if(ReadAudio(Fname1, Seq1)!=0)
		return 1;
	if(ReadAudio(Fname2, Seq2)!=0)
		return 1;

	if(Seq1.sampleRate != Seq2.sampleRate)
	{
		printf("Sample rate of %s and %s do not match. Stop!\n", Fname1, Fname2);
		return 1;
	}
	double sampleRate = Seq1.sampleRate;

	MinSample = min(MinSample, min(Seq1.nsamples,  Seq2.nsamples));
	int nSpliting =(int) floor(1.0*min(Seq1.nsamples,  Seq2.nsamples) /MinSample);
	nSpliting = nSpliting==0?1:nSpliting;

	//Take gradient of signals: somehow, this seems to be robust
	float *Grad1 = new float[Seq1.nsamples+1], *Grad2 = new float[Seq2.nsamples+1];
	Grad1[0] = Seq1.Audio[0], Grad2[0] = Seq2.Audio[0];
	Grad1[Seq1.nsamples] = Seq1.Audio[Seq1.nsamples-1], Grad2[Seq2.nsamples] = Seq2.Audio[Seq2.nsamples-1];

#pragma omp parallel for
	for(ii=0; ii<Seq1.nsamples-1; ii++)
		Grad1[ii+1] = abs(Seq1.Audio[ii+1] - Seq1.Audio[ii]);
#pragma omp parallel for
	for(ii=0; ii<Seq2.nsamples-1; ii++)
		Grad2[ii+1] = abs(Seq2.Audio[ii+1] - Seq2.Audio[ii]);

	int ns3, ns4;
	double fps3, fps4;
	float *Seq3, *Seq4;

	bool Switch = false;
	if (Seq1.nsamples <= Seq2.nsamples)
	{
		fps3 = fps1, fps4 = fps2;
		ns3 = Seq1.nsamples+1;
		Seq3 = new float[ns3 ];
#pragma omp parallel for
		for (int i = 0; i <ns3 ; i++)
			Seq3[i] = Grad1[i];

		ns4 = Seq2.nsamples+1;
		Seq4 = new float[ns4 ];
#pragma omp parallel for
		for (int i = 0; i <ns4 ; i++)
			Seq4[i] =Grad2[i];
	}
	else
	{
		Switch = true;
		fps3 = fps2, fps4 = fps1;
		ns3 = Seq2.nsamples+1;
		Seq3 = new float[ns3 ];
#pragma omp parallel for
		for (int i = 0; i <ns3 ; i++)
			Seq3[i] =Grad2[i];

		ns4 = Seq1.nsamples+1;
		Seq4 = new float[ns4 ];
#pragma omp parallel for
		for (int i = 0; i <ns4 ; i++)
			Seq4[i] = Grad1[i];
	}

	const int hbandwidth = sampleRate/100;
	int nsamplesSubSeq, nMaxLoc;
	bool *Goodness = new bool[nSpliting];
	int *soffset = new int[nSpliting];
	double *MaxCorr = new double[nSpliting], *foffset = new double[nSpliting];
	int *MaxLocID = new int[ns4+max(MinSample, ns4-MinSample*(nSpliting -1)) -1];
	float *SubSeq = new float[max(MinSample, ns4-MinSample*(nSpliting -1))];
	float *res = new float[ns4+max(MinSample, ns4-MinSample*(nSpliting -1)) -1];
	float *nres = new float[ns4+max(MinSample, ns4-MinSample*(nSpliting -1)) -1];

	for(ii=0; ii<nSpliting; ii++)
	{
		nMaxLoc = 0;
		if(ii == nSpliting -1)
			nsamplesSubSeq = ns4-MinSample*(nSpliting -1);
		else
			nsamplesSubSeq = MinSample;

		for(jj=0; jj<nsamplesSubSeq; jj++)
			SubSeq[jj] = Seq4[jj+MinSample*ii];

		//Correlate the subsequence with the smaller sequence
		ZNCC1D(SubSeq, nsamplesSubSeq,Seq3, ns3, res);
		
		//Quality check: how many peaks, are they close?
		nonMaximaSuppression1D(res, ns3+nsamplesSubSeq-1, MaxLocID, nMaxLoc, hbandwidth);
		for(jj=0; jj<ns3+nsamplesSubSeq-1; jj++)
			nres[jj] = 0.0;
		for(jj=0; jj<nMaxLoc; jj++)
			nres[MaxLocID[jj]] = res[MaxLocID[jj]];

		/// Localizing the best match with minMaxLoc
		double minVal; double maxVal, maxVal2; Point minLoc; Point maxLoc, maxLoc2;
		Mat zncc(1, ns3+nsamplesSubSeq-1, CV_32F, nres);

		minMaxLoc( zncc, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
		MaxCorr[ii] = maxVal;
		soffset[ii] = maxLoc.x - nsamplesSubSeq+1-MinSample*ii;
		foffset[ii] = 1.0*(soffset[ii])/sampleRate*fps4;

		zncc.at<float>(maxLoc.x) = 0.0;
		minMaxLoc( zncc, &minVal, &maxVal2, &minLoc, &maxLoc2, Mat() );
		if(maxVal2/maxVal>0.5 ||  abs(maxLoc2.x - maxLoc.x) < hbandwidth*2+1)
		{
			Goodness[ii] = false;
			printf("Caution! Distance to the 2nd best peak (%.4f /%.4f): %d or %.2fs\n", maxVal, maxVal2, abs(maxLoc2.x - maxLoc.x), 1.0*abs(maxLoc2.x - maxLoc.x)/sampleRate*fps4);
		}
		else
			Goodness[ii] = true;

		if(!Switch && soffset[ii]<0)
			printf("Split #%d (%d samples): %s is behind of %s %d samples or %.2f frames\n", ii+1, nsamplesSubSeq, Fname1, Fname2, abs(soffset[ii]), foffset[ii]);
		if(!Switch && soffset[ii]>=0)
			printf("Split #%d (%d samples): %s is ahead of %s %d samples or %.2f frames\n", ii+1, nsamplesSubSeq, Fname1, Fname2, abs(soffset[ii]), foffset[ii]);
		if(Switch && soffset[ii]<0)
			printf("Split #%d (%d samples): %s is behind of %s %d samples or %.2f frames\n", ii+1, nsamplesSubSeq, Fname2, Fname1, abs(soffset[ii]), foffset[ii]);
		if(Switch && soffset[ii]>=0)
			printf("Split #%d(%d samples): %s is ahead of %s %d samples or %.2f frames\n", ii+1, nsamplesSubSeq, Fname2, Fname1, abs(soffset[ii]), foffset[ii]);
	}

	//Pick the one with highest correlation score
	int *index = new int[nSpliting];
	for(int ii=0; ii<nSpliting; ii++)
	{
		index[ii] = ii;
		if(!Goodness[ii])
			MaxCorr[ii] = 0.0;
	}
	Quick_Sort_Double(MaxCorr, index, 0, nSpliting-1);

	if(MaxCorr[nSpliting-1] < reliableThreshold)
	{
		printf("The result is very unreliable (ZNCC = %.2f)! No offset will be generated.", MaxCorr[nSpliting - 1]);

		delete []index, delete []Goodness;
		delete []Seq3, delete []Seq4;
		delete []SubSeq, delete []res;

		return 1;
	}
	else
	{
		int fsoffset = soffset[index[nSpliting-1]];
		finalfoffset = 1.0*fsoffset/sampleRate*fps4;
		printf("Final offset: %d samples or %.2f frames with ZNCC score %.4f\n", fsoffset, finalfoffset, MaxCorr[nSpliting-1]);

		delete []index, delete []Goodness;
		delete []Seq3, delete []Seq4;
		delete []SubSeq, delete []res;

		return 0;
	}
}
