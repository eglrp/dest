#ifdef _WINDOWS

#include "VideoSequence.h"
#include "Ultility.h"
#include <boost/config.hpp>
#include <iostream>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>

using namespace cv;
using namespace std;

bool GrabVideoFrame2Mem(char *fname, char *Data, int &width, int &height, int &nchannels, int &nframes, int frameSample, int fixnframes)
{
	IplImage  *frame = 0;
	CvCapture *capture = cvCaptureFromFile(fname);
	if (!capture)
		return false;

	bool flag = false;
	int length, frameID = 0, frameID2 = 0;
	while (true && fixnframes > frameID)
	{
		IplImage  *frame = cvQueryFrame(capture);
		if (!frame)
		{
			cvReleaseImage(&frame);
			return true;
		}

		if (frameID == 0)
			width = frame->width, height = frame->height, nchannels = frame->nChannels, length = width*height*nchannels;

		for (int ii = 0; ii < length; ii++)
			Data[ii + length*frameID] = frame->imageData[ii];

		frameID2++;
		if (frameID2 == frameSample)
			frameID++, frameID2 = 0;
		nframes = frameID;
	}

	//cvReleaseImage(&frame);
	cvReleaseCapture(&capture);

	return true;
}
int ReadAudio(char *Fin, Sequence &mySeq, char *Fout)
{
	SNDFILE      *infile;
	SF_INFO      sinfo;

	int nchannels;
	if (!(infile = sf_open(Fin, SFM_READ, &sinfo)))
	{
		printf("Not able to open input file %s.\n", Fin);
		return  1;
	}
	else
	{
		mySeq.nsamples = (int)sinfo.frames, mySeq.sampleRate = (int)sinfo.samplerate, nchannels = (int)sinfo.channels;
		printf("Number of sample per channel=%d, Samplerate=%d, Channels=%d\n", mySeq.nsamples, mySeq.sampleRate, nchannels);
	}

	float *buf = (float *)malloc(mySeq.nsamples*nchannels*sizeof(float));
	int num = (int)sf_read_float(infile, buf, mySeq.nsamples*nchannels);

	//I want only 1 channel
	mySeq.Audio = new float[mySeq.nsamples];
	for (int i = 0; i < mySeq.nsamples; i++)
		mySeq.Audio[i] = buf[nchannels*i];
	delete[]buf;

	if (Fout != NULL)
		WriteGridBinary(Fout, mySeq.Audio, 1, mySeq.nsamples);

	return 0;
}

int PrismMST(char *Path, int nvideos)
{
	using namespace boost;
	typedef adjacency_list < vecS, vecS, undirectedS, property<vertex_distance_t, int>, property < edge_weight_t, double > > Graph;
	typedef std::pair < int, int >E;

	int v1, v2; double offset;
	char Fname[200];
	double *TimeOffset = new double[nvideos*nvideos];
	sprintf(Fname, "%s/audioSync.txt", Path);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot open %s\n", Fname);
		return 1;
	}
	while (fscanf(fp, "%d %d %lf ", &v1, &v2, &offset) != EOF)
		TimeOffset[v1 + v2*nvideos] = offset, TimeOffset[v2 + v1*nvideos] = offset;
	fclose(fp);

	int num_nodes = nvideos, nedges = nvideos*(nvideos - 1) / 2;
	E *edges = new E[nedges];
	double *weightTable = new double[nvideos*nvideos];
	double *weights = new double[nedges];

	sprintf(Fname, "%s/timeConstrantoffset.txt", Path);	fp = fopen(Fname, "w+");
	for (int kk = 0; kk < nvideos; kk++)
	{
	for (int ll = 0; ll < nvideos; ll++)
	fprintf(fp, "%.4f ", TimeOffset[kk + ll*nvideos]);
	fprintf(fp, "\n");
	}
	fclose(fp);

	//Form edges weight based on the consistency of the triplet
	int count = 0;
	for (int kk = 0; kk < nvideos - 1; kk++)
	{
		for (int ll = kk + 1; ll < nvideos; ll++)
		{
			edges[count] = E(kk, ll);
			weights[count] = 0.0;
			for (int jj = 0; jj < nvideos; jj++)
			{
				if (jj == ll || jj == kk)
					continue;
				if (jj >= ll) //ab = ac+cb;
					weights[count] += abs(TimeOffset[ll + jj*nvideos] + TimeOffset[kk + ll*nvideos] - TimeOffset[kk + jj*nvideos]);
				else if (jj <= kk)
					weights[count] += abs(TimeOffset[jj + kk*nvideos] + TimeOffset[kk + ll*nvideos] - TimeOffset[jj + ll*nvideos]);
				else
					weights[count] += abs(TimeOffset[kk + jj*nvideos] + TimeOffset[ll + jj*nvideos] - TimeOffset[kk + ll*nvideos]);
			}
			weightTable[kk + ll*nvideos] = weights[count];
			weightTable[ll + kk*nvideos] = weights[count];
			count++;
		}
	}

	/*fp = fopen("C:/temp/weightTable.txt", "w+");
	for (int kk = 0; kk < nvideos; kk++)
	{
	for (int ll = 0; ll < nvideos; ll++)
	fprintf(fp, "%.2f ", weightTable[kk + ll*nvideos]);
	fprintf(fp, "\n");
	}
	fclose(fp);*/

	Graph g(edges, edges + sizeof(E)*nedges / sizeof(E), weights, num_nodes);
	property_map<Graph, edge_weight_t>::type weightmap = get(edge_weight, g);
	std::vector < graph_traits < Graph >::vertex_descriptor >p(num_vertices(g));

	prim_minimum_spanning_tree(g, &p[0]);

	sprintf(Fname, "%s/MST_audioSync.txt", Path);
	fp = fopen(Fname, "w+");
	for (std::size_t i = 0; i != p.size(); ++i)
	{
		if (p[i] != i)
		{
			std::cout << "parent[" << i << "] = " << p[i] << std::endl;
			fprintf(fp, "%d %d\n", p[i], i);
		}
		else
		{
			std::cout << "parent[" << i << "] = no parent" << std::endl;
			fprintf(fp, "%d %d\n", i, i);
		}
	}
	fclose(fp);

	delete[]weights, delete[]weightTable, delete[]TimeOffset, delete []edges;

	return 0;
}
int SynAudio(char *Fname1, char *Fname2, double fps1, double fps2, int MinSample, double &finalfoffset, double &MaxZNCC, double reliableThreshold)
{
	omp_set_num_threads(omp_get_max_threads());

	int ii, jj;
	Sequence Seq1, Seq2;
	Seq1.InitSeq(fps1, 0.0);
	Seq2.InitSeq(fps2, 0.0);
	if (ReadAudio(Fname1, Seq1) != 0)
		return 1;
	if (ReadAudio(Fname2, Seq2) != 0)
		return 1;

	if (Seq1.sampleRate != Seq2.sampleRate)
	{
		printf("Sample rate of %s and %s do not match. Stop!\n", Fname1, Fname2);
		return 1;
	}
	double sampleRate = Seq1.sampleRate;

	MinSample = min(MinSample, min(Seq1.nsamples, Seq2.nsamples));
	int nSpliting = (int)floor(1.0*min(Seq1.nsamples, Seq2.nsamples) / MinSample);
	nSpliting = nSpliting == 0 ? 1 : nSpliting;

	//Take gradient of signals: somehow, this seems to be robust
	int filterSize = 6;
	float GaussianDfilter[] = { -0.0219, -0.0764, -0.0638, 0.0638, 0.0764, 0.0219 };
	float *Grad1 = new float[Seq1.nsamples + filterSize - 1], *Grad2 = new float[Seq2.nsamples + filterSize - 1];
	conv(Seq1.Audio, Seq1.nsamples, GaussianDfilter, filterSize, Grad1);
	conv(Seq2.Audio, Seq2.nsamples, GaussianDfilter, filterSize, Grad2);

	for (int ii = 0; ii < Seq1.nsamples + filterSize - 1; ii++)
		Grad1[ii] = abs(Grad1[ii]);
	for (int ii = 0; ii < Seq2.nsamples + filterSize - 1; ii++)
		Grad2[ii] = abs(Grad2[ii]);

	int ns3, ns4;
	double fps3, fps4;
	float *Seq3, *Seq4;

	bool Switch = false;
	if (Seq1.nsamples <= Seq2.nsamples)
	{
		fps3 = fps1, fps4 = fps2;
		ns3 = Seq1.nsamples + filterSize - 1;
		Seq3 = new float[ns3];
#pragma omp parallel for
		for (int i = 0; i < ns3; i++)
			Seq3[i] = Grad1[i];

		ns4 = Seq2.nsamples + filterSize - 1;
		Seq4 = new float[ns4];
#pragma omp parallel for
		for (int i = 0; i < ns4; i++)
			Seq4[i] = Grad2[i];
	}
	else
	{
		Switch = true;
		fps3 = fps2, fps4 = fps1;
		ns3 = Seq2.nsamples + filterSize - 1;
		Seq3 = new float[ns3];
#pragma omp parallel for
		for (int i = 0; i < ns3; i++)
			Seq3[i] = Grad2[i];

		ns4 = Seq1.nsamples + filterSize - 1;
		Seq4 = new float[ns4];
#pragma omp parallel for
		for (int i = 0; i < ns4; i++)
			Seq4[i] = Grad1[i];
	}

	const int hbandwidth = sampleRate / 60; //usually 30fps, so this give 0.5 frame accuracy
	int nsamplesSubSeq, nMaxLoc;
	bool *Goodness = new bool[nSpliting];
	int *soffset = new int[nSpliting];
	double *MaxCorr = new double[nSpliting], *foffset = new double[nSpliting];
	int *MaxLocID = new int[ns4 + max(MinSample, ns4 - MinSample*(nSpliting - 1)) - 1];
	float *SubSeq = new float[max(MinSample, ns4 - MinSample*(nSpliting - 1))];
	float *res = new float[ns4 + max(MinSample, ns4 - MinSample*(nSpliting - 1)) - 1];
	float *nres = new float[ns4 + max(MinSample, ns4 - MinSample*(nSpliting - 1)) - 1];

	for (ii = 0; ii < nSpliting; ii++)
	{
		nMaxLoc = 0;
		if (ii == nSpliting - 1)
			nsamplesSubSeq = ns4 - MinSample*(nSpliting - 1);
		else
			nsamplesSubSeq = MinSample;

		for (jj = 0; jj < nsamplesSubSeq; jj++)
			SubSeq[jj] = Seq4[jj + MinSample*ii];

		//Correlate the subsequence with the smaller sequence
		ZNCC1D(SubSeq, nsamplesSubSeq, Seq3, ns3, res);

		float a = res[jj], b = Seq3[jj], c = SubSeq[jj];
		//Quality check: how many peaks, are they close?
		nonMaximaSuppression1D(res, ns3 + nsamplesSubSeq - 1, MaxLocID, nMaxLoc, hbandwidth);
		for (jj = 0; jj < ns3 + nsamplesSubSeq - 1; jj++)
			nres[jj] = 0.0;
		for (jj = 0; jj<nMaxLoc; jj++)
			nres[MaxLocID[jj]] = res[MaxLocID[jj]];

		/// Localizing the best match with minMaxLoc
		double minVal; double maxVal, maxVal2; Point minLoc; Point maxLoc, maxLoc2;
		Mat zncc(1, ns3 + nsamplesSubSeq - 1, CV_32F, nres);

		minMaxLoc(zncc, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
		MaxCorr[ii] = maxVal;
		soffset[ii] = maxLoc.x - nsamplesSubSeq + 1 - MinSample*ii;
		foffset[ii] = 1.0*(soffset[ii]) / sampleRate*fps4;

		zncc.at<float>(maxLoc.x) = 0.0;
		minMaxLoc(zncc, &minVal, &maxVal2, &minLoc, &maxLoc2, Mat());
		if (maxVal2 / maxVal>0.5 && abs(maxLoc2.x - maxLoc.x) < hbandwidth * 2 + 1)
		{
			Goodness[ii] = false;
			printf("Caution! Distance to the 2nd best peak (%.4f /%.4f): %d or %.2fs\n", maxVal, maxVal2, abs(maxLoc2.x - maxLoc.x), 1.0*abs(maxLoc2.x - maxLoc.x) / sampleRate*fps4);
		}
		else
			Goodness[ii] = true;

		if (!Switch && soffset[ii] < 0)
			printf("Split #%d (%d samples): %s is behind of %s %d samples or %.4f sec \n", ii + 1, nsamplesSubSeq, Fname1, Fname2, abs(soffset[ii]), foffset[ii]);
		if (!Switch && soffset[ii] >= 0)
			printf("Split #%d (%d samples): %s is ahead of %s %d samples or %.4f sec \n", ii + 1, nsamplesSubSeq, Fname1, Fname2, abs(soffset[ii]), foffset[ii]);
		if (Switch && soffset[ii] < 0)
			printf("Split #%d (%d samples): %s is ahead of %s %d samples or %.4f sec \n", ii + 1, nsamplesSubSeq, Fname1, Fname2, abs(soffset[ii]), -foffset[ii]);
		if (Switch && soffset[ii] >= 0)
			printf("Split #%d(%d samples): %s is behind of %s %d samples or %.4f sec \n", ii + 1, nsamplesSubSeq, Fname1, Fname2, abs(soffset[ii]), -foffset[ii]);
	}

	//Pick the one with highest correlation score
	int *index = new int[nSpliting];
	for (int ii = 0; ii < nSpliting; ii++)
	{
		index[ii] = ii;
		if (!Goodness[ii])
			MaxCorr[ii] = 0.0;
	}
	Quick_Sort_Double(MaxCorr, index, 0, nSpliting - 1);

	if (MaxCorr[nSpliting - 1] < reliableThreshold)
	{
		printf("The result is very unreliable (ZNCC = %.2f)! No offset will be generated.", MaxCorr[nSpliting - 1]);

		delete[]Grad1, delete[]Grad2;
		delete[]index, delete[]Goodness, delete[]soffset;
		delete[]MaxCorr, delete[]MaxLocID;
		delete[]Seq3, delete[]Seq4;
		delete[]SubSeq, delete[]res, delete[]nres;

		return 1;
	}
	else
	{
		int fsoffset = soffset[index[nSpliting - 1]];
		finalfoffset = Switch ? -1.0*fsoffset / sampleRate*fps3 : 1.0*fsoffset / sampleRate*fps4;
		printf("Final offset: %d samples or %.2f frames with ZNCC score %.4f\n", fsoffset, finalfoffset, MaxCorr[nSpliting - 1]);

		MaxZNCC = MaxCorr[nSpliting - 1];

		delete[]Grad1, delete[]Grad2;
		delete[]index, delete[]Goodness, delete[]soffset;
		delete[]MaxCorr, delete[]MaxLocID;
		delete[]Seq3, delete[]Seq4;
		delete[]SubSeq, delete[]res, delete[]nres;

		return 0;
	}
}
int AssignOffsetFromMST(char *Path, int nvideos)
{
	char Fname[200];
	vector<Point2i> ParentChild;

	sprintf(Fname, "%s/MST_audioSync.txt", Path);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot open %s\n", Fname);
		return 1;
	}
	int parent, child;
	while (fscanf(fp, "%d %d ", &parent, &child) != EOF)
		ParentChild.push_back(Point2i(parent, child));
	fclose(fp);

	double *Offset = new double[nvideos*nvideos], t;
	sprintf(Fname, "%s/audioSync.txt", Path);
	fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot open %s\n", Fname);
		return 1;
	}
	while (fscanf(fp, "%d %d %lf", &parent, &child, &t) != EOF)
		Offset[parent + child*nvideos] = t, Offset[child + parent*nvideos] = t;
	fclose(fp);

	double *OrderedOffset = new double[nvideos];
	OrderedOffset[ParentChild[0].x] = 0;

	int ncollected = 1;
	vector<int>currentParent, currentChild, tcurrentChild;
	currentParent.push_back(ParentChild[0].x);
	while (ncollected != nvideos)
	{
		//search for children node
		for (int jj = 0; jj < currentParent.size(); jj++)
		{
			for (int ii = 1; ii < ParentChild.size(); ii++)
			{
				if (ParentChild[ii].x == currentParent[jj])
				{
					tcurrentChild.push_back(ParentChild[ii].y);
					ncollected++;
				}
			}

			//assign offset to children
			for (int ii = 0; ii < tcurrentChild.size(); ii++)
			{
				if (tcurrentChild[ii]>currentParent[jj])
					OrderedOffset[tcurrentChild[ii]] = OrderedOffset[currentParent[jj]] + Offset[currentParent[jj] + tcurrentChild[ii] * nvideos];
				else if (tcurrentChild[ii] < currentParent[jj])
					OrderedOffset[tcurrentChild[ii]] = OrderedOffset[currentParent[jj]] - Offset[currentParent[jj] + tcurrentChild[ii] * nvideos];
				else
					printf("Error: parent is child!\n");
			}

			for (int ii = 0; ii < tcurrentChild.size(); ii++)
				currentChild.push_back(tcurrentChild[ii]);
			tcurrentChild.clear();
		}

		//replace parent with children
		currentParent.clear();
		currentParent = currentChild;
		currentChild.clear();
	}

	//Find the video started earliest and that it as reference
	double earliest = 999.9;
	for (int ii = 0; ii < nvideos; ii++)
		if (OrderedOffset[ii] < earliest)
			earliest = OrderedOffset[ii];

	for (int ii = 0; ii < nvideos; ii++)
		OrderedOffset[ii] -= earliest;

	//Write results:
	sprintf(Fname, "%s/FaudioSync.txt", Path);
	fp = fopen(Fname, "w+");
	for (int ii = 0; ii < nvideos; ii++)
		fprintf(fp, "%d %.3f\n", ii, OrderedOffset[ii]);
	fclose(fp);

	delete[]Offset, delete[]OrderedOffset;

	return 0;
}

void DynamicTimeWarping3Step(Mat pM, vector<int>&pp, vector<int> &qq)
{
	int ii, jj;
	int nrows = pM.rows, ncols = pM.cols;

	Mat DMatrix(nrows + 1, ncols + 1, CV_64F);
	for (ii = 0; ii < nrows + 1; ii++)
		DMatrix.at<double>(ii, 0) = 10.0e16;
	for (ii = 0; ii < ncols + 1; ii++)
		DMatrix.at<double>(0, ii) = 10.0e16;
	DMatrix.at<double>(0, 0) = 0.0;
	for (jj = 0; jj < nrows; jj++)
		for (ii = 0; ii < ncols; ii++)
			DMatrix.at<double>(jj + 1, ii + 1) = pM.at<double>(jj, ii);

	// traceback
	Mat phi = Mat::zeros(nrows, ncols, CV_32S);

	int id[3]; double val[3];
	for (ii = 0; ii < nrows; ii++)
	{
		for (jj = 0; jj < ncols; jj++)
		{
			double dd = DMatrix.at<double>(ii, jj);
			//find min of sub block
			val[0] = DMatrix.at<double>(ii, jj); id[0] = 0;
			val[1] = DMatrix.at<double>(ii, jj + 1); id[1] = 1;
			val[2] = DMatrix.at<double>(ii + 1, jj); id[2] = 2;

			Quick_Sort_Double(val, id, 0, 2);
			DMatrix.at<double>(ii + 1, jj + 1) += val[0];
			phi.at<int>(ii, jj) = id[0] + 1;
			//cout << phi << endl << endl;
		}
	}

	//Traceback from top left
	{
		int jj = nrows - 1;
		int ii = ncols - 1;
		vector<int>p, q;
		p.reserve(max(nrows, ncols));
		q.reserve(max(nrows, ncols));
		p.push_back(ii);
		q.push_back(jj);
		while (ii > 0 && jj > 0)
		{
			int tb = phi.at<int>(ii, jj);

			if (tb == 1)
				ii = ii - 1, jj = jj - 1;
			else if (tb == 2)
				ii = ii - 1;
			else if (tb == 3)
				jj = jj - 1;
			else
			{
				printf("Problem in finding path of DTW\n");
				abort();
			}
			p.push_back(ii);
			q.push_back(jj);
		}

		// Strip off the edges of the D matrix before returning
		//DMatrix = D(2:(r + 1), 2 : (c + 1));

		//flip the vector, substract 1 and store
		int nele = q.size();
		pp.reserve(nele); qq.reserve(nele);
		for (int ii = 0; ii < nele; ii++)
			pp.push_back(p[nele - 1 - ii]), qq.push_back(q[nele - 1 - ii]);
	}
	return;
}
void DynamicTimeWarping5Step(Mat pM, vector<int>&pp, vector<int> &qq)
{
	int ii, jj;
	int nrows = pM.rows, ncols = pM.cols;

	Mat DMatrix(nrows + 1, ncols + 1, CV_64F);
	for (ii = 0; ii < nrows + 1; ii++)
		DMatrix.at<double>(ii, 0) = 10.0e16;
	for (ii = 0; ii < ncols + 1; ii++)
		DMatrix.at<double>(0, ii) = 10.0e16;
	DMatrix.at<double>(0, 0) = 0.0;
	for (jj = 0; jj < nrows; jj++)
		for (ii = 0; ii < ncols; ii++)
			DMatrix.at<double>(jj + 1, ii + 1) = pM.at<double>(jj, ii);

	// traceback
	Mat phi = Mat::zeros(nrows + 1, ncols + 1, CV_32S);

	int id[5]; double val[5];
	//Scale the 'longer' steps to discourage skipping ahead
	int kk1 = 2, kk2 = 1;
	for (ii = 1; ii < nrows + 1; ii++)
	{
		for (jj = 1; jj < ncols + 1; jj++)
		{
			double dd = DMatrix.at<double>(ii, jj);
			//find min of sub block
			val[0] = DMatrix.at<double>(ii - 1, jj - 1) + dd; id[0] = 0;
			val[1] = DMatrix.at<double>(max(0, ii - 2), jj - 1) + dd*kk1; id[1] = 1;
			val[2] = DMatrix.at<double>(ii - 1, max(0, jj - 2)) + dd*kk1; id[2] = 2;
			val[3] = DMatrix.at<double>(ii - 1, jj) + dd*kk2; id[3] = 3;
			val[4] = DMatrix.at<double>(ii, jj - 1) + dd*kk2; id[4] = 4;

			Quick_Sort_Double(val, id, 0, 4);
			DMatrix.at<double>(ii, jj) = val[0];
			phi.at<int>(ii, jj) = id[0] + 1;
			//cout << phi << endl << endl;
		}
	}
	//cout << phi << endl << endl;
	//Traceback from top left
	jj = nrows;
	ii = ncols;
	vector<int>p, q;
	p.reserve(max(nrows, ncols));
	q.reserve(max(nrows, ncols));
	p.push_back(ii);
	q.push_back(jj);
	while (ii > 1 && jj > 1)
	{
		int tb = phi.at<int>(ii, jj);

		if (tb == 1)
			ii = ii - 1, jj = jj - 1;
		else if (tb == 2)
			ii = ii - 2, jj = jj - 1;
		else if (tb == 3)
			ii = ii - 1, jj = jj - 2;
		else if (tb == 4)
			ii = ii - 1;
		else if (tb == 5)
			jj = jj - 1;
		else
		{
			printf("Problem in finding path of DTW\n");
			abort();
		}
		p.push_back(ii);
		q.push_back(jj);
	}

	// Strip off the edges of the D matrix before returning
	//DMatrix = D(2:(r + 1), 2 : (c + 1));

	//flip the vector, substract 1 and store
	int nele = q.size();
	pp.reserve(nele); qq.reserve(nele);
	for (int ii = 0; ii < nele; ii++)
		pp.push_back(p[nele - 1 - ii] - 1), qq.push_back(q[nele - 1 - ii] - 1);

	return;
}
#endif
