#include "ImagePro.h"
#include "Ultility.h"

void filter1D_row_Double(double *kernel, int k_size, double *in, double *out, int width, int height)
{
	int ii, jj, i_in, iKernel, k_mid = (k_size-1)/2;
	double t_value;

	for(ii=0; ii<width*height; ii++)
		out[ii] = 0.0;

	for(jj=0; jj<height; jj++)
	{
		for(ii=k_mid; ii<width-k_mid; ii++)
		{
			t_value=0.0;
			for(iKernel=0; iKernel<k_size; iKernel++)
			{
				i_in = ii + (iKernel-k_mid); 			
				t_value += kernel[iKernel]*in[i_in+jj*width];
			}
			out[ii+jj*width] = t_value;
		}
	}

	return;
}
void filter1D_row(double *kernel, int k_size, char *in, double *out, int width, int height)
{


	int ii, jj, i_in, iKernel, k_mid = (k_size-1)/2;
	double t_value;

	for(ii=0; ii<width*height; ii++)
		out[ii] = 0.0;

	for(jj=0; jj<height; jj++)
	{
		for(ii=k_mid; ii<width-k_mid; ii++)
		{
			t_value=0.0;
			for(iKernel=0; iKernel<k_size; iKernel++)
			{
				i_in = ii + (iKernel-k_mid); 			
				t_value += kernel[iKernel]*((int)((unsigned char)(in[i_in+jj*width])));
			}
			out[ii+jj*width] = t_value;
		}
	}

	return;
}
void filter1D_col(double *kernel, int k_size, double *in, double *out, int width, int height, double &i_max)
{


	int ii, jj, j_in, jKernel, k_mid = (k_size-1)/2;
	double t_value;

	for(ii=0; ii<width*height; ii++)
		out[ii] = 0.0;

	i_max = 1.0;
	for(ii=k_mid; ii<width-k_mid;ii++)
	{
		for(jj=k_mid; jj<height-k_mid; jj++)	
		{
			t_value=0;
			for(jKernel=0; jKernel<k_size; jKernel++)
			{
				j_in = jj + (jKernel-k_mid); 			
				t_value += kernel[jKernel]*in[ii+j_in*width];
			}
			out[ii+jj*width] = t_value;
			if(t_value > i_max)
				i_max = t_value;
		}
	}

	return;
}
void Gaussian_smooth(char* data, double* out_data, int height, int width, double max_i, double sigma)
{
	int ii, jj, size = MyFtoI(6.0*sigma+1)/2*2+1;
	double max_filter, sigma2 = 2.0*sigma*sigma, sqrt2Pi_sigma = sqrt(2.0*Pi)*sigma;
	double *kernel=new double[size];
	double *temp = new double[width*height];
	int kk = (size-1)/2;

	for(ii=-kk; ii<=kk;ii++)
		kernel[ii+kk] = exp(-(ii*ii)/sigma2) / sqrt2Pi_sigma;

	if(abs(max_i - 255.0)<1.0)
	{
		for(ii=0; ii<width*height; ii++)
			if(data[ii] > max_i)
				max_i = (unsigned char) data[ii];
	}

	filter1D_row(kernel, size, data, temp, width, height);
	filter1D_col(kernel, size, temp, out_data, width, height, max_filter);

	for(jj=kk; jj<height-kk; jj++)
		for(ii=kk; ii<width-kk; ii++)
			out_data[ii+jj*width] = out_data[ii+jj*width]*max_i/max_filter;

	for(ii=0; ii<width; ii++)
	{
		for(jj=0; jj<kk; jj++)
			out_data[ii+jj*width] = (double)((int)((unsigned char)(data[ii+jj*width])));
		for(jj=height-kk; jj<height; jj++)
			out_data[ii+jj*width] = (double)((int)((unsigned char)(data[ii+jj*width])));
	}

	for(jj=kk; jj<height-kk; jj++)
	{
		for(ii=0; ii<kk; ii++)
			out_data[ii+jj*width] = (double)((int)((unsigned char)(data[ii+jj*width])));
		for(ii=width-kk; ii<width; ii++)
			out_data[ii+jj*width] = (double)((int)((unsigned char)(data[ii+jj*width])));
	}

	delete []temp;
	delete []kernel;
	return;
}
void Gaussian_smooth_Double(double* data, double* out_data, int height, int width, double max_i, double sigma)
{
	int ii, jj, size = MyFtoI(6.0*sigma+1)/2*2+1;
	double max_filter, sigma2 = 2.0*sigma*sigma, sqrt2Pi_sigma = sqrt(2.0*Pi)*sigma;
	double *kernel=new double[size];
	double *temp = new double[width*height];
	int kk = (size-1)/2;

	for(ii=-kk; ii<=kk;ii++)
		kernel[ii+kk] = exp(-(ii*ii)/sigma2) / sqrt2Pi_sigma;

	if(abs(max_i - 255.0)<1.0)
	{
		for(ii=0; ii<width*height; ii++)
			if(data[ii] > max_i)
				max_i = data[ii];
	}

	filter1D_row_Double(kernel, size, data, temp, width, height);
	filter1D_col(kernel, size, temp, out_data, width, height, max_filter);

	for(jj=kk; jj<height-kk; jj++)
		for(ii=kk; ii<width-kk; ii++)
			out_data[ii+jj*width] = out_data[ii+jj*width]*max_i/max_filter;

	for(ii=0; ii<width; ii++)
	{
		for(jj=0; jj<kk; jj++)
			out_data[ii+jj*width] = data[ii+jj*width];
		for(jj=height-kk; jj<height; jj++)
			out_data[ii+jj*width] = data[ii+jj*width];
	}

	for(jj=kk; jj<height-kk; jj++)
	{
		for(ii=0; ii<kk; ii++)
			out_data[ii+jj*width] = data[ii+jj*width];
		for(ii=width-kk; ii<width; ii++)
			out_data[ii+jj*width] = data[ii+jj*width];
	}

	delete []temp;
	delete []kernel;
	return;
}
double InitialCausalCoefficient(double *sample,int length,double pole, double tolerance)
{
	double zn,iz,z2n;
	double FirstCausalCoef;
	int n,horizon;
	horizon = (int)( ceil(log(tolerance) / log(fabs(pole))) + 0.01 );
	if (horizon < length) {
		/* accelerated loop */
		zn = pole;
		FirstCausalCoef = *(sample);
		for (n = 1; n < horizon; n++) {
			FirstCausalCoef += zn * (*(sample + n));
			zn *= pole;
		}
	}
	else {
		/* full loop */
		zn = pole;
		iz = 1.0 / pole;
		z2n = pow(pole, (double)(length - 1));
		FirstCausalCoef = sample[0] + z2n * sample[length - 1];
		z2n *= z2n * iz;
		for (n = 1; n <= length - 2; n++) {
			FirstCausalCoef += (zn + z2n) * sample[n];
			zn *= pole;
			z2n *= iz;
		}
	}
	return FirstCausalCoef;
}
double InitialAnticausalCoefficient(double *CausalCoef,int length,double pole)
{
	return((pole / (pole * pole - 1.0)) * (pole * CausalCoef[length - 2] + CausalCoef[length - 1]));
}
// prefilter for 4-tap, 6-tap, 8-tap, optimized 4-tap, and optimized 6-tap
void Prefilter_1D( double *coefficient,int length,double *pole, double tolerance,int nPoles)
{
	int i,n,k;
	double Lambda;
	Lambda = 1;
	if (length == 1)
		return;
	/* compute the overall gain */
	for (k = 0; k < nPoles; k++) 
		Lambda = Lambda * (1.0 - pole[k]) * (1.0 - 1.0 / pole[k]);

	// Applying the gain to original image
	for (i = 0; i < length; i++)
		*(coefficient + i) =  (*(coefficient + i)) * Lambda;

	for(k = 0; k < nPoles; k++)
	{
		// Compute the first causal coefficient
		*(coefficient) = InitialCausalCoefficient(coefficient,length, pole[k],tolerance);

		// Causal prefilter
		for (n = 1; n < length; n++) 
			coefficient[n] += pole[k] * coefficient[n - 1];

		//Compute the first anticausal coefficient
		*(coefficient + length - 1) = InitialAnticausalCoefficient(coefficient,length,pole[k]);

		//Anticausal prefilter
		for (n = length - 2; n >= 0; n--) 
			coefficient[n] = pole[k] * (coefficient[n + 1] - coefficient[n]);
	}
}
// Prefilter for modified 4-tap
void Prefilter_1Dm( double *coefficient,int length,double *pole, double tolerance,double gamma)
{
	int i,n,k;
	double Lambda;
	Lambda = 6.0 / (6.0 * gamma + 1.0);
	if (length == 1)
		return;

	// Applying the gain to original image
	for (i = 0; i < length; i++)
		*(coefficient + i) =  (*(coefficient + i)) * Lambda;

	for(k = 0; k < 1; k++)
	{
		// Compute the first causal coefficient
		*(coefficient) = InitialCausalCoefficient(coefficient,length, pole[k],tolerance);

		// Causal prefilter
		for (n = 1; n < length; n++) 
			coefficient[n] += pole[k] * coefficient[n - 1];

		//Compute the first anticausal coefficient
		*(coefficient + length - 1) = InitialAnticausalCoefficient(coefficient,length,pole[k]);

		//Anticausal prefilter
		for (n = length - 2; n >= 0; n--) 
			coefficient[n] = pole[k] * (coefficient[n + 1] - coefficient[n]);
	}
}

void Generate_Para_Spline(double *Image, double *Para, int width, int height, int Interpolation_Algorithm)
{
	double tolerance;
	int i, j, nPoles;
	int length = width * height;
	double pole[2], a, gamma;
	tolerance = 1e-4;
	if (Interpolation_Algorithm == 1) // 4-tap
	{
		nPoles = 1;
		pole[0] = sqrt(3.0) - 2.0;
	}
	else if (Interpolation_Algorithm == 2) // 6-tap
	{
		nPoles = 2;
		pole[0] = sqrt(664.0 - sqrt(438976.0)) + sqrt(304.0) - 19.0;
		pole[1] = sqrt(664.0 + sqrt(438976.0)) - sqrt(304.0) - 19.0;
	}
	else if (Interpolation_Algorithm == 3) // modified 4-tap
	{
		gamma = 0.0409;
		a = (4.0 - 12.0 * gamma) / (6.0 * gamma + 1.0) ;
		pole[0] = (-a + sqrt(a * a - 4)) / 2.0;
	}
	else if (Interpolation_Algorithm == 4) // optimized 4-tap
	{
		nPoles = 1;
		pole[0] = (-13.0 + sqrt(105.0)) / 8.0;
	}
	else if (Interpolation_Algorithm == 5) // optimized 6-tap
	{
		nPoles = 2;
		pole[0] = -0.410549185795627524168;
		pole[1] = -0.0316849091024414351363;
	}
	else if (Interpolation_Algorithm == 6) // 8-tap
	{
		nPoles = 2;
		pole[0] = sqrt(135.0 / 2.0 - sqrt(17745.0 / 4.0)) + sqrt(105.0 / 4.0) - 13.0 / 2.0;
		pole[1] = sqrt(135.0 / 2.0 + sqrt(17745.0 / 4.0)) - sqrt(105.0 / 4.0) - 13.0 / 2.0;
	}

	//Perform the 1D prefiltering along the rows
	double *LineWidth = new double[width];
	for (i = 0; i < height; i++) 
	{
		//Prefiltering each row
		for (j = 0; j < width;j++)
		{
			*(LineWidth + j) = *(Image + i * width + j);
		}
		if (Interpolation_Algorithm == 3)
			Prefilter_1Dm(LineWidth, width, pole, tolerance, gamma);
		else
			Prefilter_1D(LineWidth, width, pole, tolerance, nPoles);

		// Put the prefiltered coeffiecients into Para array
		for (j = 0; j < width; j++)
		{
			*(Para + i * width + j) = (*(LineWidth + j));
		}
	}
	delete []LineWidth;

	//Perform the 1D prefiltering along the columns
	double *LineHeight = new double[height];
	for (i = 0; i < width; i++)
	{
		//Prefiltering each comlumn
		for (j = 0; j < height;j++)
		{
			*(LineHeight + j) = (*(Para + j * width + i));
		}
		if (Interpolation_Algorithm == 3)
			Prefilter_1Dm(LineHeight,height, pole, tolerance, gamma);
		else
			Prefilter_1D(LineHeight,height, pole, tolerance, nPoles);

		//Put the prefilterd coefficients into the Para array
		for (j = 0; j < height; j++)
		{
			*(Para + j * width + i) = (*(LineHeight + j));
		}
	}
	delete []LineHeight;
	return;
}
void Generate_Para_Spline(char *Image, double *Para, int width, int height, int Interpolation_Algorithm)
{
	int i, length = width*height;
	double *Image_2 = new double[length];
	for(i=0; i<length; i++)
		*(Image_2+i) = (double)((int)((unsigned char)(*(Image+i))));

	Generate_Para_Spline(Image_2, Para, width, height, Interpolation_Algorithm);

	delete []Image_2;
	return;
}
void Generate_Para_Spline(unsigned char *Image, double *Para, int width, int height, int Interpolation_Algorithm)
{
	int i, length = width*height;
	double *Image_2 = new double[length];
	for(i=0; i<length; i++)
		*(Image_2+i) = (double)((int)(*(Image+i)));

	Generate_Para_Spline(Image_2, Para, width, height, Interpolation_Algorithm);

	delete []Image_2;
	return;
}
void Generate_Para_Spline(int *Image, double *Para, int width, int height, int Interpolation_Algorithm)
{
	int i, length = width*height;
	double *Image_2 = new double[length];
	for(i=0; i<length; i++)
		*(Image_2+i) = (double)(*(Image+i));

	Generate_Para_Spline(Image_2, Para, width, height, Interpolation_Algorithm);

	delete []Image_2;
	return;
}
void Get_Value_Spline(double *Para, int width,int height, double X, double Y, double *S, int S_Flag, int Interpolation_Algorithm)
{
	int i, j, width2, height2, xIndex[6], yIndex[6];
	double Para_Value, xWeight[6], yWeight[6], xWeightGradient[6], yWeightGradient[6], w, w2, w3, w4, t, t0, t1, gamma;
	double oneSix = 1.0/6.0;

	width2 = 2 * width - 2;
	height2 = 2 * height - 2;

	if (Interpolation_Algorithm == 6)
	{
		xIndex[0] = int(X) - 2;
		yIndex[0] = int(Y) - 2;
		for (i = 1; i < 6; i++)
		{
			xIndex[i] = xIndex[i-1] + 1;
			yIndex[i] = yIndex[i-1] + 1;
		}
	}
	else if ((Interpolation_Algorithm == 2) || (Interpolation_Algorithm == 5))
	{
		xIndex[0] = int(X + 0.5) - 2;
		yIndex[0] = int(Y + 0.5) - 2;
		for (i = 1; i < 5; i++)
		{
			xIndex[i] = xIndex[i-1] + 1;
			yIndex[i] = yIndex[i-1] + 1;
		}
	}
	else
	{
		xIndex[0] = int(X ) - 1;
		yIndex[0] = int(Y ) - 1;
		for (i = 1; i < 4; i++)
		{
			xIndex[i] = xIndex[i-1] + 1;
			yIndex[i] = yIndex[i-1] + 1;
		}
	}

	//Calculate the weights of x,y and their derivatives
	if (Interpolation_Algorithm == 1)
	{
		w = X - (double)xIndex[1];
		w2 = w*w; w3 = w2*w;
		xWeight[3] = oneSix * w3;
		xWeight[0] = oneSix + 0.5 * (w2 - w) - xWeight[3];
		xWeight[2] = w + xWeight[0] - 2.0 * xWeight[3];
		xWeight[1] = 1.0 - xWeight[0] - xWeight[2] - xWeight[3];

		if(S_Flag >-1)
		{
			xWeightGradient[3] = w2 / 2.0;
			xWeightGradient[0] = w - 0.5 - xWeightGradient[3];
			xWeightGradient[2] = 1.0 + xWeightGradient[0] - 2.0 * xWeightGradient[3];
			xWeightGradient[1] = - xWeightGradient[0] - xWeightGradient[2] - xWeightGradient[3];
		}

		/* y */
		w = Y - (double)yIndex[1];
		w2 = w*w; w3 = w2*w;
		yWeight[3] = oneSix * w3;
		yWeight[0] = oneSix + 0.5 * (w2 - w) - yWeight[3];
		yWeight[2] = w + yWeight[0] - 2.0 * yWeight[3];
		yWeight[1] = 1.0 - yWeight[0] - yWeight[2] - yWeight[3];

		if(S_Flag >-1)
		{
			yWeightGradient[3] = w2 / 2.0;
			yWeightGradient[0] = w - 0.5 - yWeightGradient[3];
			yWeightGradient[2] = 1.0 + yWeightGradient[0] - 2.0 * yWeightGradient[3];
			yWeightGradient[1] = - yWeightGradient[0] - yWeightGradient[2] - yWeightGradient[3];
		}
	}
	else if (Interpolation_Algorithm == 2)
	{
		w = X - (double)xIndex[2];
		w2 = w * w;
		t = (1.0 / 6.0) * w2;
		xWeight[0] = 1.0 / 2.0 - w;
		xWeight[0] *= xWeight[0];
		xWeight[0] *= (1.0 / 24.0) * xWeight[0];
		t0 = w * (t - 11.0 / 24.0);
		t1 = 19.0 / 96.0 + w2 * (1.0 / 4.0 - t);
		xWeight[1] = t1 + t0;
		xWeight[3] = t1 - t0;
		xWeight[4] = xWeight[0] + t0 + (1.0 / 2.0) * w;
		xWeight[2] = 1.0 - xWeight[0] - xWeight[1] - xWeight[3] - xWeight[4];

		xWeightGradient[0] = -(1.0 / 2.0 - w) * (1.0 / 2.0 - w) * (1.0 / 2.0 - w) / 6.0;
		xWeightGradient[1] = w * w / 2 - 11.0 / 24.0 + w / 2.0  - 2.0 * w * w * w / 3.0 ;
		xWeightGradient[3] = -w * w / 2 + 11.0 / 24.0 + w / 2.0  - 2.0 * w * w * w / 3.0 ;
		xWeightGradient[4] = xWeightGradient[0] + w * w / 2.0  + 1.0 / 24.0;
		xWeightGradient[2] = -xWeightGradient[0] - xWeightGradient[1] - xWeightGradient[3] - xWeightGradient[4];

		/* y */
		w = Y - (double)yIndex[2];
		w2 = w * w;
		t = (1.0 / 6.0) * w2;
		yWeight[0] = 1.0 / 2.0 - w;
		yWeight[0] *= yWeight[0];
		yWeight[0] *= (1.0 / 24.0) * yWeight[0];
		t0 = w * (t - 11.0 / 24.0);
		t1 = 19.0 / 96.0 + w2 * (1.0 / 4.0 - t);
		yWeight[1] = t1 + t0;
		yWeight[3] = t1 - t0;
		yWeight[4] = yWeight[0] + t0 + (1.0 / 2.0) * w;
		yWeight[2] = 1.0 - yWeight[0] - yWeight[1] - yWeight[3] - yWeight[4];

		yWeightGradient[0] = -(1.0 / 2.0 - w) * (1.0 / 2.0 - w) * (1.0 / 2.0 - w) / 6.0;
		yWeightGradient[1] = w * w / 2 - 11.0 / 24.0 + w / 2.0  - 2.0 * w * w * w / 3.0 ;
		yWeightGradient[3] = -w * w / 2 + 11.0 / 24.0 + w / 2.0  - 2.0 * w * w * w / 3.0 ;
		yWeightGradient[4] = yWeightGradient[0] + w * w / 2.0  + 1.0 / 24.0;
		yWeightGradient[2] = -yWeightGradient[0] - yWeightGradient[1] - yWeightGradient[3] - yWeightGradient[4];
	}
	else if (Interpolation_Algorithm == 3)
	{
		gamma = 0.0409;
		w = X - (double)xIndex[1];
		xWeight[0] = - w * w * w / 6.0 + w * w / 2.0 - (gamma + 0.5) * w + 1.0 / 6.0 + gamma;
		xWeight[1] = w * w * w / 2.0 - w * w + 3 * gamma * w + 2.0 / 3.0 - 2.0 * gamma;
		xWeight[2] = -w * w * w / 2.0 +  w * w / 2.0 + (1.0 / 2.0 - 3.0 * gamma) * w + gamma + 1.0 / 6.0;
		xWeight[3] = w * w * w / 6.0 + gamma * w ;

		xWeightGradient[0] = - w * w / 2.0 + w - gamma - 0.5;
		xWeightGradient[1] = 3.0 * w * w / 2.0 - 2.0 * w + 3.0 * gamma;
		xWeightGradient[2] = -3.0 * w * w / 2.0 + w + 1.0 / 2.0 - 3.0 * gamma;
		xWeightGradient[3] = w * w /2.0 + gamma;

		/* y */
		w = Y - (double)yIndex[1];
		yWeight[0] = - w * w * w / 6.0 + w * w / 2.0 - (gamma + 0.5) * w + 1.0 / 6.0 + gamma;
		yWeight[1] = w * w * w / 2.0 - w * w + 3 * gamma * w + 2.0 / 3.0 - 2.0 * gamma;
		yWeight[2] = -w * w * w / 2.0 +  w * w / 2.0 + (1.0 / 2.0 - 3.0 * gamma) * w + gamma + 1.0 / 6.0;
		yWeight[3] = w * w * w / 6.0 + gamma * w ;

		yWeightGradient[0] = - w * w / 2.0 + w - gamma - 0.5;
		yWeightGradient[1] = 3.0 * w * w / 2.0 - 2.0 * w + 3.0 * gamma;
		yWeightGradient[2] = -3.0 * w * w / 2.0 + w + 1.0 / 2.0 - 3.0 * gamma;
		yWeightGradient[3] = w * w /2.0 + gamma;
	}
	else if (Interpolation_Algorithm == 4)
	{
		w = X - (double)xIndex[1];
		xWeight[0] = - w * w * w / 6.0 + w * w / 2.0 - 11.0 * w / 21.0 + 4.0 / 21.0;
		xWeight[1] = w * w * w / 2.0 - w * w + 3.0 * w / 42.0 + 13.0 / 21.0;
		xWeight[2] = - w * w * w / 2.0 + w * w / 2.0 + 3.0 * w / 7.0 + 4.0 / 21.0;
		xWeight[3] = w * w * w / 6.0 + w / 42.0;

		xWeightGradient[0] = - w * w / 2.0 + w - 11.0 / 21.0;
		xWeightGradient[1] = 3.0 * w * w / 2.0 - 2.0 * w + 3.0 /42.0;
		xWeightGradient[2] = -3.0 * w * w / 2.0 + w + 3.0 /7.0;
		xWeightGradient[3] = w * w / 2.0 + 1.0 / 42.0;

		/* y */
		w = Y - (double)yIndex[1];
		yWeight[0] = - w * w * w / 6.0 + w * w / 2.0 - 11.0 * w / 21.0 + 4.0 / 21.0;
		yWeight[1] = w * w * w / 2.0 - w * w + 3.0 * w / 42.0 + 13.0 / 21.0;
		yWeight[2] = - w * w * w / 2.0 + w * w / 2.0 + 3.0 * w / 7.0 + 4.0 / 21.0;
		yWeight[3] = w * w * w / 6.0 + w / 42.0;

		yWeightGradient[0] = - w * w / 2.0 + w - 11.0 / 21.0;
		yWeightGradient[1] = 3.0 * w * w / 2.0 - 2.0 * w + 3.0 /42.0;
		yWeightGradient[2] = -3.0 * w * w / 2.0 + w + 3.0 /7.0;
		yWeightGradient[3] = w * w / 2.0 + 1.0 / 42.0;
	}
	else if (Interpolation_Algorithm == 5)
	{
		w = X - (double)xIndex[2];
		w2 = w*w; w3 = w2*w; w4 = w2*w2;
		double coeff1 = 743.0 / 120960.0, coeff2 = 6397.0 / 30240.0, coeff3 =  5.0 / 144.0, coeff4 = 31.0 / 72.0, coeff5 = 11383.0 / 20160.0;
		double coeff6 = 11.0 / 144.0, coeff7 = 5.0/ 144.0, coeff8 = 7.0 / 36.0, coeff9 = 31.0/ 72.0, coeff10 = 13.0/24.0, coeff11 = 11.0/ 72.0, coeff12 = 7.0 / 18.0;
		xWeight[0] = w4 / 24.0 - w3/ 12.0 + w2 * coeff6 - w *coeff7 + coeff1;
		xWeight[1] = -w4 /6.0 + w3/6.0 + w2*coeff8 - w*coeff9 + coeff2;
		xWeight[2] = w4 / 4.0 -  w2 *coeff10 + coeff5;
		xWeight[3] = -w4 /6.0 - w3/6.0 + w2*coeff8 + w*coeff9 + coeff2;
		xWeight[4] = w4 / 24.0 + w3/ 12.0 + w2 * coeff6 + w *coeff7 + coeff1;
		//xWeight[2] = 1.0 - xWeight[0] - xWeight[1] - xWeight[3] - xWeight[4]; 

		if(S_Flag >-1)
		{
			xWeightGradient[0] =  w3/ 6.0 - w2 /4.0 +  w *coeff11 - coeff3;
			xWeightGradient[1] = -2.0 * w3/ 3.0 + w2 / 2.0 +  w*coeff12 - coeff4;
			xWeightGradient[2] = w3- 13.0 * w / 12.0;
			xWeightGradient[3] = -2.0 * w3/ 3.0 - w2 / 2.0 +  w*coeff12 + coeff4;
			xWeightGradient[4] = w3/ 6.0 + w2 /4.0 +  w *coeff11 + coeff3;
		}

		/* y */
		w = Y - (double)yIndex[2];
		w2 = w*w; w3 = w2*w; w4 = w2*w2 ;
		yWeight[0] = w4 / 24.0 - w3/ 12.0 + w2 * coeff6 - w *coeff7 + coeff1;
		yWeight[1] = -w4 /6.0 + w3/6.0 + w2*coeff8 - w*coeff9 + coeff2;
		yWeight[2] = w4 / 4.0 -  w2 *coeff10 + coeff5;
		yWeight[3] = -w4 /6.0 - w3/6.0 + w2*coeff8 + w*coeff9 + coeff2;
		yWeight[4] = w4 / 24.0 + w3/ 12.0 + w2 * coeff6 + w *coeff7 + coeff1;
		//yWeight[2] = 1.0 - xWeight[0] - xWeight[1] - xWeight[3] - xWeight[4]; 

		if(S_Flag >-1)
		{
			yWeightGradient[0] =  w3/ 6.0 - w2 /4.0 +  w *coeff11 - coeff3;
			yWeightGradient[1] = -2.0 * w3/ 3.0 + w2 / 2.0 +  w*coeff12 - coeff4;
			yWeightGradient[2] = w3- 13.0 * w / 12.0;
			yWeightGradient[3] = -2.0 * w3/ 3.0 - w2 / 2.0 +  w*coeff12 + coeff4;
			yWeightGradient[4] = w3/ 6.0 + w2 /4.0 +  w *coeff11 + coeff3;
		}
	}
	else if (Interpolation_Algorithm == 6)
	{
		w = X - (double)xIndex[2];
		w2 = w * w;
		xWeight[5] = (1.0 / 120.0) * w * w2 * w2;
		w2 -= w;
		w4 = w2 * w2;
		w -= 1.0 / 2.0;
		t = w2 * (w2 - 3.0);
		xWeight[0] = (1.0 / 24.0) * (1.0 / 5.0 + w2 + w4) - xWeight[5];
		t0 = (1.0 / 24.0) * (w2 * (w2 - 5.0) + 46.0 / 5.0);
		t1 = (-1.0 / 12.0) * w * (t + 4.0);
		xWeight[2] = t0 + t1;
		xWeight[3] = t0 - t1;
		t0 = (1.0 / 16.0) * (9.0 / 5.0 - t);
		t1 = (1.0 / 24.0) * w * (w4 - w2 - 5.0);
		xWeight[1] = t0 + t1;
		xWeight[4] = t0 - t1;

		xWeightGradient[5] = w * w * w * w / 24.0;
		xWeightGradient[0] = (4 * w * w * w - 6 * w * w + 4 * w - 1) / 24.0 - xWeightGradient[5];
		t0 = (4.0 * w * w * w - 6.0 * w * w - 8.0 * w + 5.0) / 24.0;
		t1 = -(5.0 * w * w * w * w - 10.0 * w * w * w - 3.0 * w * w + 8.0 * w + 5.0 /2.0) / 12.0;
		xWeightGradient[2] = t0 + t1;
		xWeightGradient[3] = t0 - t1;
		t0 = (-4.0 * w * w * w + 6.0 * w * w + 4.0 * w - 3) / 16.0;
		t1 = (5.0 * w * w * w * w - 10.0 * w * w * w + 3.0 * w * w + 2 * w - 11.0 / 2.0) / 24.0;
		xWeightGradient[1] = t0 + t1;
		xWeightGradient[4] = t0 - t1;

		/* y */
		w = Y - (double)yIndex[2];
		w2 = w * w;
		yWeight[5] = (1.0 / 120.0) * w * w2 * w2;
		w2 -= w;
		w4 = w2 * w2;
		w -= 1.0 / 2.0;
		t = w2 * (w2 - 3.0);
		yWeight[0] = (1.0 / 24.0) * (1.0 / 5.0 + w2 + w4) - yWeight[5];
		t0 = (1.0 / 24.0) * (w2 * (w2 - 5.0) + 46.0 / 5.0);
		t1 = (-1.0 / 12.0) * w * (t + 4.0);
		yWeight[2] = t0 + t1;
		yWeight[3] = t0 - t1;
		t0 = (1.0 / 16.0) * (9.0 / 5.0 - t);
		t1 = (1.0 / 24.0) * w * (w4 - w2 - 5.0);
		yWeight[1] = t0 + t1;
		yWeight[4] = t0 - t1;

		yWeightGradient[5] = w * w * w * w / 24.0;
		yWeightGradient[0] = (4 * w * w * w - 6 * w * w + 4 * w - 1) / 24.0 - yWeightGradient[5];
		t0 = (4.0 * w * w * w - 6.0 * w * w - 8.0 * w + 5.0) / 24.0;
		t1 = -(5.0 * w * w * w * w - 10.0 * w * w * w - 3.0 * w * w + 8.0 * w + 5.0 /2.0) / 12.0;
		yWeightGradient[2] = t0 + t1;
		yWeightGradient[3] = t0 - t1;
		t0 = (-4.0 * w * w * w + 6.0 * w * w + 4.0 * w - 3) / 16.0;
		t1 = (5.0 * w * w * w * w - 10.0 * w * w * w + 3.0 * w * w + 2 * w - 11.0 / 2.0) / 24.0;
		yWeightGradient[1] = t0 + t1;
		yWeightGradient[4] = t0 - t1;
	}
	//***********************************

	/* apply the mirror boundary conditions and calculate the interpolated values */
	S[0] = 0;
	S[1] = 0;
	S[2] = 0;

	if (Interpolation_Algorithm == 6)
	{
		for (i = 0; i < 6; i++)
		{
			xIndex[i] = (width == 1) ? (0) : ((xIndex[i] < 0) ?
				(-xIndex[i] - width2 * ((-xIndex[i]) / width2))
				: (xIndex[i] - width2 * (xIndex[i] / width2)));
			if (width <= xIndex[i]) {
				xIndex[i] = width2 - xIndex[i];
			}
			yIndex[i] = (height == 1) ? (0) : ((yIndex[i] < 0) ?
				(-yIndex[i] - height2 * ((-yIndex[i]) / height2))
				: (yIndex[i] - height2 * (yIndex[i] / height2)));
			if (height <= yIndex[i]) {
				yIndex[i] = height2 - yIndex[i];
			}
		}

		for(i = 0; i < 6; i++)
		{
			for(j = 0; j < 6;j++)
			{
				Para_Value = (*(Para + width * yIndex[i] + xIndex[j] ));
				S[0] = S[0] + Para_Value * xWeight[j] * yWeight[i];
				if(S_Flag >-1)
				{
					S[1] = S[1] + Para_Value * xWeightGradient[j] * yWeight[i];
					S[2] = S[2] + Para_Value * xWeight[j] * yWeightGradient[i];
				}
			}
		}
	}
	else if ((Interpolation_Algorithm == 2) || (Interpolation_Algorithm == 5))
	{
		for (i = 0; i < 5; i++)
		{
			xIndex[i] = (width == 1) ? (0) : ((xIndex[i] < 0) ?
				(-xIndex[i] - width2 * ((-xIndex[i]) / width2))
				: (xIndex[i] - width2 * (xIndex[i] / width2)));
			if (width <= xIndex[i]) {
				xIndex[i] = width2 - xIndex[i];
			}
			yIndex[i] = (height == 1) ? (0) : ((yIndex[i] < 0) ?
				(-yIndex[i] - height2 * ((-yIndex[i]) / height2))
				: (yIndex[i] - height2 * (yIndex[i] / height2)));
			if (height <= yIndex[i]) {
				yIndex[i] = height2 - yIndex[i];
			}
		}
		for(i = 0; i < 5; i++)
		{
			for(j = 0; j < 5;j++)
			{
				Para_Value = (*(Para + width * yIndex[i] + xIndex[j] ));
				S[0] = S[0] + Para_Value * xWeight[j] * yWeight[i];
				if(S_Flag >-1)
				{
					S[1] = S[1] + Para_Value * xWeightGradient[j] * yWeight[i];
					S[2] = S[2] + Para_Value * xWeight[j] * yWeightGradient[i];
				}
			}
		}
	}
	else
	{
		for (i = 0; i < 4; i++)
		{
			xIndex[i] = (width == 1) ? (0) : ((xIndex[i] < 0) ?
				(-xIndex[i] - width2 * ((-xIndex[i]) / width2))
				: (xIndex[i] - width2 * (xIndex[i] / width2)));
			if (width <= xIndex[i]) {
				xIndex[i] = width2 - xIndex[i];
			}
			yIndex[i] = (height == 1) ? (0) : ((yIndex[i] < 0) ?
				(-yIndex[i] - height2 * ((-yIndex[i]) / height2))
				: (yIndex[i] - height2 * (yIndex[i] / height2)));
			if (height <= yIndex[i]) {
				yIndex[i] = height2 - yIndex[i];
			}
		}
		for(i = 0; i < 4; i++)
		{
			for(j = 0; j < 4;j++)
			{
				Para_Value = (*(Para + width * yIndex[i] + xIndex[j] ));
				S[0] = S[0] + Para_Value * xWeight[j] * yWeight[i];
				if(S_Flag >-1)
				{
					S[1] = S[1] + Para_Value * xWeightGradient[j] * yWeight[i];
					S[2] = S[2] + Para_Value * xWeight[j] * yWeightGradient[i];
				}
			}
		}
	}

	return;
}

//Float is to save memory, so, its input and output are a bit different
void Generate_Para_Spline(float *Image, float *Para, int width, int height, int Interpolation_Algorithm)
{
	double tolerance;
	int i, j, nPoles;
	int length = width * height;
	double pole[2], a, gamma;
	tolerance = 1e-4;
	if (Interpolation_Algorithm == 1) // 4-tap
	{
		nPoles = 1;
		pole[0] = sqrt(3.0) - 2.0;
	}
	else if (Interpolation_Algorithm == 2) // 6-tap
	{
		nPoles = 2;
		pole[0] = sqrt(664.0 - sqrt(438976.0)) + sqrt(304.0) - 19.0;
		pole[1] = sqrt(664.0 + sqrt(438976.0)) - sqrt(304.0) - 19.0;
	}
	else if (Interpolation_Algorithm == 3) // modified 4-tap
	{
		gamma = 0.0409;
		a = (4.0 - 12.0 * gamma) / (6.0 * gamma + 1.0) ;
		pole[0] = (-a + sqrt(a * a - 4)) / 2.0;
	}
	else if (Interpolation_Algorithm == 4) // optimized 4-tap
	{
		nPoles = 1;
		pole[0] = (-13.0 + sqrt(105.0)) / 8.0;
	}
	else if (Interpolation_Algorithm == 5) // optimized 6-tap
	{
		nPoles = 2;
		pole[0] = -0.410549185795627524168;
		pole[1] = -0.0316849091024414351363;
	}
	else if (Interpolation_Algorithm == 6) // 8-tap
	{
		nPoles = 2;
		pole[0] = sqrt(135.0 / 2.0 - sqrt(17745.0 / 4.0)) + sqrt(105.0 / 4.0) - 13.0 / 2.0;
		pole[1] = sqrt(135.0 / 2.0 + sqrt(17745.0 / 4.0)) - sqrt(105.0 / 4.0) - 13.0 / 2.0;
	}

	//Perform the 1D prefiltering along the rows
	double *LineWidth = new double[width];
	for (i = 0; i < height; i++) 
	{
		//Prefiltering each row
		for (j = 0; j < width;j++)
		{
			*(LineWidth + j) = *(Image + i * width + j);
		}
		if (Interpolation_Algorithm == 3)
			Prefilter_1Dm(LineWidth, width, pole, tolerance, gamma);
		else
			Prefilter_1D(LineWidth, width, pole, tolerance, nPoles);

		// Put the prefiltered coeffiecients into Para array
		for (j = 0; j < width; j++)
			Para[i * width + j] = (float)LineWidth[ j];
	}
	delete []LineWidth;

	//Perform the 1D prefiltering along the columns
	double *LineHeight = new double[height];
	for (i = 0; i < width; i++)
	{
		//Prefiltering each comlumn
		for (j = 0; j < height;j++)
		{
			*(LineHeight + j) = (*(Para + j * width + i));
		}
		if (Interpolation_Algorithm == 3)
			Prefilter_1Dm(LineHeight,height, pole, tolerance, gamma);
		else
			Prefilter_1D(LineHeight,height, pole, tolerance, nPoles);

		//Put the prefilterd coefficients into the Para array
		for (j = 0; j < height; j++)
			Para[j * width + i] = (float)LineHeight[j];
	}
	delete []LineHeight;

	return;
}
void Get_Value_Spline(float *Para, int width,int height, double X, double Y, double *S, int S_Flag, int Interpolation_Algorithm)
{	
	int i, j, width2, height2, xIndex[6], yIndex[6];
	double Para_Value, xWeight[6], yWeight[6], xWeightGradient[6], yWeightGradient[6], w, w2, w3, w4, t, t0, t1, gamma;
	double oneSix = 1.0/6.0;

	width2 = 2 * width - 2;
	height2 = 2 * height - 2;

	if (Interpolation_Algorithm == 6)
	{
		xIndex[0] = int(X) - 2;
		yIndex[0] = int(Y) - 2;
		for (i = 1; i < 6; i++)
		{
			xIndex[i] = xIndex[i-1] + 1;
			yIndex[i] = yIndex[i-1] + 1;
		}
	}
	else if ((Interpolation_Algorithm == 2) || (Interpolation_Algorithm == 5))
	{
		xIndex[0] = int(X + 0.5) - 2;
		yIndex[0] = int(Y + 0.5) - 2;
		for (i = 1; i < 5; i++)
		{
			xIndex[i] = xIndex[i-1] + 1;
			yIndex[i] = yIndex[i-1] + 1;
		}
	}
	else
	{
		xIndex[0] = int(X ) - 1;
		yIndex[0] = int(Y ) - 1;
		for (i = 1; i < 4; i++)
		{
			xIndex[i] = xIndex[i-1] + 1;
			yIndex[i] = yIndex[i-1] + 1;
		}
	}

	//Calculate the weights of x,y and their derivatives
	if (Interpolation_Algorithm == 1)
	{
		w = X - (double)xIndex[1];
		w2 = w*w; w3 = w2*w;
		xWeight[3] = oneSix * w3;
		xWeight[0] = oneSix + 0.5 * (w2 - w) - xWeight[3];
		xWeight[2] = w + xWeight[0] - 2.0 * xWeight[3];
		xWeight[1] = 1.0 - xWeight[0] - xWeight[2] - xWeight[3];

		xWeightGradient[3] = w2 / 2.0;
		xWeightGradient[0] = w - 0.5 - xWeightGradient[3];
		xWeightGradient[2] = 1.0 + xWeightGradient[0] - 2.0 * xWeightGradient[3];
		xWeightGradient[1] = - xWeightGradient[0] - xWeightGradient[2] - xWeightGradient[3];

		/* y */
		w = Y - (double)yIndex[1];
		w2 = w*w; w3 = w2*w;
		yWeight[3] = oneSix * w3;
		yWeight[0] = oneSix + 0.5 * (w2 - w) - yWeight[3];
		yWeight[2] = w + yWeight[0] - 2.0 * yWeight[3];
		yWeight[1] = 1.0 - yWeight[0] - yWeight[2] - yWeight[3];

		yWeightGradient[3] = w2 / 2.0;
		yWeightGradient[0] = w - 0.5 - yWeightGradient[3];
		yWeightGradient[2] = 1.0 + yWeightGradient[0] - 2.0 * yWeightGradient[3];
		yWeightGradient[1] = - yWeightGradient[0] - yWeightGradient[2] - yWeightGradient[3];
	}
	else if (Interpolation_Algorithm == 2)
	{
		w = X - (double)xIndex[2];
		w2 = w * w;
		t = (1.0 / 6.0) * w2;
		xWeight[0] = 1.0 / 2.0 - w;
		xWeight[0] *= xWeight[0];
		xWeight[0] *= (1.0 / 24.0) * xWeight[0];
		t0 = w * (t - 11.0 / 24.0);
		t1 = 19.0 / 96.0 + w2 * (1.0 / 4.0 - t);
		xWeight[1] = t1 + t0;
		xWeight[3] = t1 - t0;
		xWeight[4] = xWeight[0] + t0 + (1.0 / 2.0) * w;
		xWeight[2] = 1.0 - xWeight[0] - xWeight[1] - xWeight[3] - xWeight[4];

		xWeightGradient[0] = -(1.0 / 2.0 - w) * (1.0 / 2.0 - w) * (1.0 / 2.0 - w) / 6.0;
		xWeightGradient[1] = w * w / 2 - 11.0 / 24.0 + w / 2.0  - 2.0 * w * w * w / 3.0 ;
		xWeightGradient[3] = -w * w / 2 + 11.0 / 24.0 + w / 2.0  - 2.0 * w * w * w / 3.0 ;
		xWeightGradient[4] = xWeightGradient[0] + w * w / 2.0  + 1.0 / 24.0;
		xWeightGradient[2] = -xWeightGradient[0] - xWeightGradient[1] - xWeightGradient[3] - xWeightGradient[4];

		/* y */
		w = Y - (double)yIndex[2];
		w2 = w * w;
		t = (1.0 / 6.0) * w2;
		yWeight[0] = 1.0 / 2.0 - w;
		yWeight[0] *= yWeight[0];
		yWeight[0] *= (1.0 / 24.0) * yWeight[0];
		t0 = w * (t - 11.0 / 24.0);
		t1 = 19.0 / 96.0 + w2 * (1.0 / 4.0 - t);
		yWeight[1] = t1 + t0;
		yWeight[3] = t1 - t0;
		yWeight[4] = yWeight[0] + t0 + (1.0 / 2.0) * w;
		yWeight[2] = 1.0 - yWeight[0] - yWeight[1] - yWeight[3] - yWeight[4];

		yWeightGradient[0] = -(1.0 / 2.0 - w) * (1.0 / 2.0 - w) * (1.0 / 2.0 - w) / 6.0;
		yWeightGradient[1] = w * w / 2 - 11.0 / 24.0 + w / 2.0  - 2.0 * w * w * w / 3.0 ;
		yWeightGradient[3] = -w * w / 2 + 11.0 / 24.0 + w / 2.0  - 2.0 * w * w * w / 3.0 ;
		yWeightGradient[4] = yWeightGradient[0] + w * w / 2.0  + 1.0 / 24.0;
		yWeightGradient[2] = -yWeightGradient[0] - yWeightGradient[1] - yWeightGradient[3] - yWeightGradient[4];
	}
	else if (Interpolation_Algorithm == 3)
	{
		gamma = 0.0409;
		w = X - (double)xIndex[1];
		xWeight[0] = - w * w * w / 6.0 + w * w / 2.0 - (gamma + 0.5) * w + 1.0 / 6.0 + gamma;
		xWeight[1] = w * w * w / 2.0 - w * w + 3 * gamma * w + 2.0 / 3.0 - 2.0 * gamma;
		xWeight[2] = -w * w * w / 2.0 +  w * w / 2.0 + (1.0 / 2.0 - 3.0 * gamma) * w + gamma + 1.0 / 6.0;
		xWeight[3] = w * w * w / 6.0 + gamma * w ;

		xWeightGradient[0] = - w * w / 2.0 + w - gamma - 0.5;
		xWeightGradient[1] = 3.0 * w * w / 2.0 - 2.0 * w + 3.0 * gamma;
		xWeightGradient[2] = -3.0 * w * w / 2.0 + w + 1.0 / 2.0 - 3.0 * gamma;
		xWeightGradient[3] = w * w /2.0 + gamma;

		/* y */
		w = Y - (double)yIndex[1];
		yWeight[0] = - w * w * w / 6.0 + w * w / 2.0 - (gamma + 0.5) * w + 1.0 / 6.0 + gamma;
		yWeight[1] = w * w * w / 2.0 - w * w + 3 * gamma * w + 2.0 / 3.0 - 2.0 * gamma;
		yWeight[2] = -w * w * w / 2.0 +  w * w / 2.0 + (1.0 / 2.0 - 3.0 * gamma) * w + gamma + 1.0 / 6.0;
		yWeight[3] = w * w * w / 6.0 + gamma * w ;

		yWeightGradient[0] = - w * w / 2.0 + w - gamma - 0.5;
		yWeightGradient[1] = 3.0 * w * w / 2.0 - 2.0 * w + 3.0 * gamma;
		yWeightGradient[2] = -3.0 * w * w / 2.0 + w + 1.0 / 2.0 - 3.0 * gamma;
		yWeightGradient[3] = w * w /2.0 + gamma;
	}
	else if (Interpolation_Algorithm == 4)
	{
		w = X - (double)xIndex[1];
		xWeight[0] = - w * w * w / 6.0 + w * w / 2.0 - 11.0 * w / 21.0 + 4.0 / 21.0;
		xWeight[1] = w * w * w / 2.0 - w * w + 3.0 * w / 42.0 + 13.0 / 21.0;
		xWeight[2] = - w * w * w / 2.0 + w * w / 2.0 + 3.0 * w / 7.0 + 4.0 / 21.0;
		xWeight[3] = w * w * w / 6.0 + w / 42.0;

		xWeightGradient[0] = - w * w / 2.0 + w - 11.0 / 21.0;
		xWeightGradient[1] = 3.0 * w * w / 2.0 - 2.0 * w + 3.0 /42.0;
		xWeightGradient[2] = -3.0 * w * w / 2.0 + w + 3.0 /7.0;
		xWeightGradient[3] = w * w / 2.0 + 1.0 / 42.0;

		/* y */
		w = Y - (double)yIndex[1];
		yWeight[0] = - w * w * w / 6.0 + w * w / 2.0 - 11.0 * w / 21.0 + 4.0 / 21.0;
		yWeight[1] = w * w * w / 2.0 - w * w + 3.0 * w / 42.0 + 13.0 / 21.0;
		yWeight[2] = - w * w * w / 2.0 + w * w / 2.0 + 3.0 * w / 7.0 + 4.0 / 21.0;
		yWeight[3] = w * w * w / 6.0 + w / 42.0;

		yWeightGradient[0] = - w * w / 2.0 + w - 11.0 / 21.0;
		yWeightGradient[1] = 3.0 * w * w / 2.0 - 2.0 * w + 3.0 /42.0;
		yWeightGradient[2] = -3.0 * w * w / 2.0 + w + 3.0 /7.0;
		yWeightGradient[3] = w * w / 2.0 + 1.0 / 42.0;
	}
	else if (Interpolation_Algorithm == 5)
	{
		w = X - (double)xIndex[2];
		w2 = w*w; w3 = w2*w; w4 = w2*w2;
		double coeff1 = 743.0 / 120960.0, coeff2 = 6397.0 / 30240.0, coeff3 =  5.0 / 144.0, coeff4 = 31.0 / 72.0, coeff5 = 11383.0 / 20160.0;
		double coeff6 = 11.0 / 144.0, coeff7 = 5.0/ 144.0, coeff8 = 7.0 / 36.0, coeff9 = 31.0/ 72.0, coeff10 = 13.0/24.0, coeff11 = 11.0/ 72.0, coeff12 = 7.0 / 18.0;
		xWeight[0] = w4 / 24.0 - w3/ 12.0 + w2 * coeff6 - w *coeff7 + coeff1;
		xWeight[1] = -w4 /6.0 + w3/6.0 + w2*coeff8 - w*coeff9 + coeff2;
		xWeight[2] = w4 / 4.0 -  w2 *coeff10 + coeff5;
		xWeight[3] = -w4 /6.0 - w3/6.0 + w2*coeff8 + w*coeff9 + coeff2;
		xWeight[4] = w4 / 24.0 + w3/ 12.0 + w2 * coeff6 + w *coeff7 + coeff1;
		//xWeight[2] = 1.0 - xWeight[0] - xWeight[1] - xWeight[3] - xWeight[4]; 

		xWeightGradient[0] =  w3/ 6.0 - w2 /4.0 +  w *coeff11 - coeff3;
		xWeightGradient[1] = -2.0 * w3/ 3.0 + w2 / 2.0 +  w*coeff12 - coeff4;
		xWeightGradient[2] = w3- 13.0 * w / 12.0;
		xWeightGradient[3] = -2.0 * w3/ 3.0 - w2 / 2.0 +  w*coeff12 + coeff4;
		xWeightGradient[4] = w3/ 6.0 + w2 /4.0 +  w *coeff11 + coeff3;

		/* y */
		w = Y - (double)yIndex[2];
		w2 = w*w; w3 = w2*w; w4 = w2*w2 ;
		yWeight[0] = w4 / 24.0 - w3/ 12.0 + w2 * coeff6 - w *coeff7 + coeff1;
		yWeight[1] = -w4 /6.0 + w3/6.0 + w2*coeff8 - w*coeff9 + coeff2;
		yWeight[2] = w4 / 4.0 -  w2 *coeff10 + coeff5;
		yWeight[3] = -w4 /6.0 - w3/6.0 + w2*coeff8 + w*coeff9 + coeff2;
		yWeight[4] = w4 / 24.0 + w3/ 12.0 + w2 * coeff6 + w *coeff7 + coeff1;
		//yWeight[2] = 1.0 - xWeight[0] - xWeight[1] - xWeight[3] - xWeight[4]; 

		yWeightGradient[0] =  w3/ 6.0 - w2 /4.0 +  w *coeff11 - coeff3;
		yWeightGradient[1] = -2.0 * w3/ 3.0 + w2 / 2.0 +  w*coeff12 - coeff4;
		yWeightGradient[2] = w3- 13.0 * w / 12.0;
		yWeightGradient[3] = -2.0 * w3/ 3.0 - w2 / 2.0 +  w*coeff12 + coeff4;
		yWeightGradient[4] = w3/ 6.0 + w2 /4.0 +  w *coeff11 + coeff3;
	}
	else if (Interpolation_Algorithm == 6)
	{
		w = X - (double)xIndex[2];
		w2 = w * w;
		xWeight[5] = (1.0 / 120.0) * w * w2 * w2;
		w2 -= w;
		w4 = w2 * w2;
		w -= 1.0 / 2.0;
		t = w2 * (w2 - 3.0);
		xWeight[0] = (1.0 / 24.0) * (1.0 / 5.0 + w2 + w4) - xWeight[5];
		t0 = (1.0 / 24.0) * (w2 * (w2 - 5.0) + 46.0 / 5.0);
		t1 = (-1.0 / 12.0) * w * (t + 4.0);
		xWeight[2] = t0 + t1;
		xWeight[3] = t0 - t1;
		t0 = (1.0 / 16.0) * (9.0 / 5.0 - t);
		t1 = (1.0 / 24.0) * w * (w4 - w2 - 5.0);
		xWeight[1] = t0 + t1;
		xWeight[4] = t0 - t1;

		xWeightGradient[5] = w * w * w * w / 24.0;
		xWeightGradient[0] = (4 * w * w * w - 6 * w * w + 4 * w - 1) / 24.0 - xWeightGradient[5];
		t0 = (4.0 * w * w * w - 6.0 * w * w - 8.0 * w + 5.0) / 24.0;
		t1 = -(5.0 * w * w * w * w - 10.0 * w * w * w - 3.0 * w * w + 8.0 * w + 5.0 /2.0) / 12.0;
		xWeightGradient[2] = t0 + t1;
		xWeightGradient[3] = t0 - t1;
		t0 = (-4.0 * w * w * w + 6.0 * w * w + 4.0 * w - 3) / 16.0;
		t1 = (5.0 * w * w * w * w - 10.0 * w * w * w + 3.0 * w * w + 2 * w - 11.0 / 2.0) / 24.0;
		xWeightGradient[1] = t0 + t1;
		xWeightGradient[4] = t0 - t1;

		/* y */
		w = Y - (double)yIndex[2];
		w2 = w * w;
		yWeight[5] = (1.0 / 120.0) * w * w2 * w2;
		w2 -= w;
		w4 = w2 * w2;
		w -= 1.0 / 2.0;
		t = w2 * (w2 - 3.0);
		yWeight[0] = (1.0 / 24.0) * (1.0 / 5.0 + w2 + w4) - yWeight[5];
		t0 = (1.0 / 24.0) * (w2 * (w2 - 5.0) + 46.0 / 5.0);
		t1 = (-1.0 / 12.0) * w * (t + 4.0);
		yWeight[2] = t0 + t1;
		yWeight[3] = t0 - t1;
		t0 = (1.0 / 16.0) * (9.0 / 5.0 - t);
		t1 = (1.0 / 24.0) * w * (w4 - w2 - 5.0);
		yWeight[1] = t0 + t1;
		yWeight[4] = t0 - t1;

		yWeightGradient[5] = w * w * w * w / 24.0;
		yWeightGradient[0] = (4 * w * w * w - 6 * w * w + 4 * w - 1) / 24.0 - yWeightGradient[5];
		t0 = (4.0 * w * w * w - 6.0 * w * w - 8.0 * w + 5.0) / 24.0;
		t1 = -(5.0 * w * w * w * w - 10.0 * w * w * w - 3.0 * w * w + 8.0 * w + 5.0 /2.0) / 12.0;
		yWeightGradient[2] = t0 + t1;
		yWeightGradient[3] = t0 - t1;
		t0 = (-4.0 * w * w * w + 6.0 * w * w + 4.0 * w - 3) / 16.0;
		t1 = (5.0 * w * w * w * w - 10.0 * w * w * w + 3.0 * w * w + 2 * w - 11.0 / 2.0) / 24.0;
		yWeightGradient[1] = t0 + t1;
		yWeightGradient[4] = t0 - t1;
	}
	//***********************************

	/* apply the mirror boundary conditions and calculate the interpolated values */
	S[0] = 0;
	S[1] = 0;
	S[2] = 0;

	if (Interpolation_Algorithm == 6)
	{
		for (i = 0; i < 6; i++)
		{
			xIndex[i] = (width == 1) ? (0) : ((xIndex[i] < 0) ?
				(-xIndex[i] - width2 * ((-xIndex[i]) / width2))
				: (xIndex[i] - width2 * (xIndex[i] / width2)));
			if (width <= xIndex[i]) {
				xIndex[i] = width2 - xIndex[i];
			}
			yIndex[i] = (height == 1) ? (0) : ((yIndex[i] < 0) ?
				(-yIndex[i] - height2 * ((-yIndex[i]) / height2))
				: (yIndex[i] - height2 * (yIndex[i] / height2)));
			if (height <= yIndex[i]) {
				yIndex[i] = height2 - yIndex[i];
			}
		}

		for(i = 0; i < 6; i++)
		{
			for(j = 0; j < 6;j++)
			{
				Para_Value = (*(Para + width * yIndex[i] + xIndex[j] ));
				S[0] = S[0] + Para_Value * xWeight[j] * yWeight[i];
				if(S_Flag >-1)
				{
					S[1] = S[1] + Para_Value * xWeightGradient[j] * yWeight[i];
					S[2] = S[2] + Para_Value * xWeight[j] * yWeightGradient[i];
				}
			}
		}
	}
	else if ((Interpolation_Algorithm == 2) || (Interpolation_Algorithm == 5))
	{
		for (i = 0; i < 5; i++)
		{
			xIndex[i] = (width == 1) ? (0) : ((xIndex[i] < 0) ?
				(-xIndex[i] - width2 * ((-xIndex[i]) / width2))
				: (xIndex[i] - width2 * (xIndex[i] / width2)));
			if (width <= xIndex[i]) {
				xIndex[i] = width2 - xIndex[i];
			}
			yIndex[i] = (height == 1) ? (0) : ((yIndex[i] < 0) ?
				(-yIndex[i] - height2 * ((-yIndex[i]) / height2))
				: (yIndex[i] - height2 * (yIndex[i] / height2)));
			if (height <= yIndex[i]) {
				yIndex[i] = height2 - yIndex[i];
			}
		}
		for(i = 0; i < 5; i++)
		{
			for(j = 0; j < 5;j++)
			{
				Para_Value = (*(Para + width * yIndex[i] + xIndex[j] ));
				S[0] = S[0] + Para_Value * xWeight[j] * yWeight[i];
				if(S_Flag >-1)
				{
					S[1] = S[1] + Para_Value * xWeightGradient[j] * yWeight[i];
					S[2] = S[2] + Para_Value * xWeight[j] * yWeightGradient[i];
				}
			}
		}
	}
	else
	{
		for (i = 0; i < 4; i++)
		{
			xIndex[i] = (width == 1) ? (0) : ((xIndex[i] < 0) ?
				(-xIndex[i] - width2 * ((-xIndex[i]) / width2))
				: (xIndex[i] - width2 * (xIndex[i] / width2)));
			if (width <= xIndex[i]) {
				xIndex[i] = width2 - xIndex[i];
			}
			yIndex[i] = (height == 1) ? (0) : ((yIndex[i] < 0) ?
				(-yIndex[i] - height2 * ((-yIndex[i]) / height2))
				: (yIndex[i] - height2 * (yIndex[i] / height2)));
			if (height <= yIndex[i]) {
				yIndex[i] = height2 - yIndex[i];
			}
		}
		for(i = 0; i < 4; i++)
		{
			for(j = 0; j < 4;j++)
			{
				Para_Value = (*(Para + width * yIndex[i] + xIndex[j] ));
				S[0] = S[0] + Para_Value * xWeight[j] * yWeight[i];
				if(S_Flag >-1)
				{
					S[1] = S[1] + Para_Value * xWeightGradient[j] * yWeight[i];
					S[2] = S[2] + Para_Value * xWeight[j] * yWeightGradient[i];
				}
			}
		}
	}

	return;
}

//Bicubic spline
void Generate_Para_BiCubic_Spline_Double(double *Image, double *Para, int width, int height)
{
	int i, j;
	int width_ex = width + 2;
	int height_ex = height + 2;

	for(j=0; j<height; j++)
	{
		for(i=0; i<width; i++)
		{
			*(Para+(j+1)*width_ex+i+1) =  *(Image+j*width+i);
		}
	}

	// bottom
	for(i=0; i<width; i++)
		*(Para+i+1) =  *(Image+i)*2 - *(Image+width+i);

	// top
	for(i=0; i<width; i++)
		*(Para+(height_ex-1)*width_ex+i+1) =  *(Image+(height-1)*width+i)*2 - *(Image+(height-2)*width+i);

	// left
	for(j=0; j<height; j++)
		*(Para+(j+1)*width_ex) =  *(Image+j*width)*2 - *(Image+j*width+1);

	// right
	for(j=0; j<height; j++)
		*(Para+(j+1)*width_ex+width_ex-1) =  *(Image+j*width+width-1)*2 - *(Image+j*width+width-2);

	//l-b corner
	*(Para) =  *(Image)*4 - *(Image+1)*2 - *(Image+width)*2 + *(Image+width+1);

	//r-b corner
	*(Para+width_ex-1) = *(Image+width-1)*4 - *(Image+width-2)*2 - *(Image+width+width-1)*2 + *(Image+width+width-2) ;

	//l-t corner
	*(Para+(height_ex-1)*width_ex) = *(Image+(height-1)*width)*4 - *(Image+(height-1)*width+1)*2 - *(Image+(height-2)*width)*2 + *(Image+(height-2)*width+1);

	//r-t corner
	*(Para+(height_ex-1)*width_ex+width_ex-1) = *(Image+(height-1)*width+width-1)*4 - *(Image+(height-1)*width+width-2)*2 - *(Image+(height-2)*width+width-1)*2 + *(Image+(height-2)*width+width-2);

	return;
}
void Get_Value_BiCubic_Spline(double *Para, int width_ex, int height_ex, double X, double Y, double *S, int S_Flag)
{
	int i, j, xIndex[4], yIndex[4];
	double xWeight[4], yWeight[4], xWeightGradient[4], yWeightGradient[4], w, a, Image_value, w3, w2;
	a = -0.5;

	xIndex[0] = int(X);
	yIndex[0] = int(Y);
	for (i = 1; i < 4; i++)
	{
		xIndex[i] = xIndex[i-1] + 1;
		yIndex[i] = yIndex[i-1] + 1;
	}
	//Another method********************
	w = X - (double)xIndex[0];
	w2 = w*w; w3 = w2*w;
	xWeight[0] = a * (w3 - 2.0 * w2+ w);
	xWeight[1] = (a + 2.0) * w3 - (a + 3.0) * w2+ 1;
	xWeight[2] = -(a + 2.0) * w3 + (2.0 * a + 3.0) * w2- a * w;
	xWeight[3] = a * (-w3 + w * w);

	if(S_Flag>-1)
	{
		xWeightGradient[0] = a * (3.0 * w2- 4.0 * w + 1);
		xWeightGradient[1] = 3.0 * (a + 2.0) * w2- 2.0 * (a + 3.0) * w;
		xWeightGradient[2] = -3.0 * (a + 2.0) * w2+ 2.0 * (2.0 * a + 3.0) * w - a;
		xWeightGradient[3] = a * (-3.0 * w2+ 2.0 * w);
	}

	w = Y - (double)yIndex[0];
	w2 = w*w; w3 = w2*w;
	yWeight[0] = a * (w3 - 2.0 * w2+ w);
	yWeight[1] = (a + 2.0) * w3 - (a + 3.0) * w2+ 1;
	yWeight[2] = -(a + 2.0) * w3 + (2.0 * a + 3.0) * w2- a * w;
	yWeight[3] = a * (-w3 + w * w);

	if(S_Flag>-1)
	{
		yWeightGradient[0] = a * (3.0 * w2- 4.0 * w + 1);
		yWeightGradient[1] = 3.0 * (a + 2.0) * w2- 2.0 * (a + 3.0) * w;
		yWeightGradient[2] = -3.0 * (a + 2.0) * w2+ 2.0 * (2.0 * a + 3.0) * w - a;
		yWeightGradient[3] = a * (-3.0 * w2+ 2.0 * w);
	}
	//***********************************

	S[0] = 0;
	if(S_Flag>-1)
	{
		S[1] = 0;
		S[2] = 0;
	}

	for(i = 0; i < 4; i++)
	{
		for(j = 0; j < 4;j++)
		{
			Image_value = (*(Para + width_ex * yIndex[i] + xIndex[j] ));
			S[0] = S[0] + Image_value * xWeight[j] * yWeight[i];
			if(S_Flag>-1)
			{
				S[1] = S[1] + Image_value * xWeightGradient[j] * yWeight[i];
				S[2] = S[2] + Image_value * xWeight[j] * yWeightGradient[i];
			}
		}
	}

	return;
}

int LinearInterp(int *data, int width, int height, double u, double v)
{
	int ul = (int)(u), uh = (int) (u+1), vl = (int)v, vh = (int)(v+1);
	double ufrac = u - 1.0*ul, vfrac = v-1.0*vl;

	int f00 = data[ul+vl*width];
	int f10= data[uh+vl*width];
	int f01 = data[ul+vh*width];
	int f11 = data[uh+vh*width];

	double res= (1.0-ufrac)*(1.0-vfrac)*f00 + ufrac*(1.0-vfrac)*f10 + (1.0-ufrac)*vfrac*f01 + ufrac*vfrac*f11;
	return (int) (res+0.5);
}
double BilinearInterp(double *data, int width, int height, double x, double y)
{
	if(x<0 || x>width-2 || y<0 || y>height-2)
		return 255; //Make it white

	int xiD = (int)(x), yiD = (int)(y);
	int xiU = xiD+1, yiU = yiD+1;

	double f00 = data[xiD + yiD*width ];
	double f01 = data[xiU + yiD*width];
	double f10 = data[xiD + yiU*width];
	double  f11 = data[xiU + yiU*width];
	double res = (f01-f00)*(x-xiD) + (f10-f00)*(y-yiD) + (f11-f01-f10+f00)*(x-xiD)*(y-yiD) + f00; 

	return res;
}
void Average_Filtering_All(char *lpD, int width, int height, int ni, int HSize, int VSize)
{
	int length = width*height;
	int i, j, k, m, n, ii, jj, s;

	char *T = new char[length];

	for(k=0; k<ni; k++)
	{
		/// Testing shows that memcpy() is NOT faster than the conventional way
		// memcpy(T, lpD+k*length, length);
		for(n=0; n<length; n++)
			*(T+n) = *(lpD+k*length+n);

		/*
		for(j=0; j<height; j++)
		{
		for(i=0; i<width; i++)
		{
		s = 0;
		m = 0;
		for(jj=-VSize; jj<=VSize; jj++)
		{
		for(ii=-HSize; ii<=HSize; ii++)
		{
		if( (j+jj)>=0 && (j+jj)<height && (i+ii)>=0 && (i+ii)<width )
		{
		s +=  (int)((unsigned char)*(lpD+k*length+(j+jj)*width+i+ii));
		m++;
		}
		}
		}
		s = int( (double)s/m + 0.5 );
		*(T+j*width+i) = (char)s;
		}
		}*/

		/// Do not consider the boundary issue to increase the processing speed
		m = (2*HSize+1)*(2*VSize+1);
		for(j=VSize; j<height-VSize; j++)
		{
			for(i=HSize; i<width-HSize; i++)
			{
				s = 0;
				for(jj=-VSize; jj<=VSize; jj++)
				{
					for(ii=-HSize; ii<=HSize; ii++)
					{
						s +=  (int)((unsigned char)*(lpD+k*length+(j+jj)*width+i+ii));
					}
				}
				s = int( (double)s/m + 0.5 );
				*(T+j*width+i) = (char)s;
			}
		}

		for(n=0; n<length; n++)
			*(lpD+k*length+n) = *(T+n);
	}

	delete []T;

	return;
}

void MConventional_PhaseShifting(char *lpD, char *lpPBM, double* lpFO, int nipf, int length, int Mask_Threshold, double *f_atan2)
{
	int n, ni, nj;
	int I1, I2, I3, I4, I5, f1, f2;
	double q;

	if(nipf==3)
	{
		ni = 510;
		nj = 255;
		for(n=0;n<length;n++)
		{
			I1=(int)((unsigned char)*(lpD+n));
			I2=(int)((unsigned char)*(lpD+length+n));
			I3=(int)((unsigned char)*(lpD+2*length+n));
			if( abs(I1-I2)<Mask_Threshold && abs(I2-I3)<Mask_Threshold && abs(I3-I1)<Mask_Threshold)
				*(lpPBM+n)=(char)0;

			f1=2*I1-I2-I3;
			f2=I3-I2;
			q =f_atan2[(f2+nj)*(2*ni+1) + f1+ni];

			if(q<0.0)
				q=q+2*Pi;
			*(lpFO+n) = q/(2*Pi);
		}
	}
	else if(nipf==4)
	{
		ni = 255;
		nj = 255;

		for(n=0;n<length;n++)
		{
			I1=(int)((unsigned char)*(lpD+n));
			I2=(int)((unsigned char)*(lpD+length+n));
			I3=(int)((unsigned char)*(lpD+2*length+n));
			I4=(int)((unsigned char)*(lpD+3*length+n));
			if( abs(I1-I2)<Mask_Threshold && abs(I2-I3)<Mask_Threshold && abs(I3-I4)<Mask_Threshold && abs(I4-I1)<Mask_Threshold )
			{
				*(lpPBM+n)=(char)0;
			}

			f1=I1-I3;
			f2=I4-I2;
			q = *(f_atan2 + (f2+nj)*(2*ni+1) + f1+ni);
			if(q<0.0)
				q=q+2*Pi;
			*(lpFO+n) = q/(2*Pi);
		}
	}
	else if(nipf==5)
	{
		ni = 510;
		nj = 510;

		for(n=0;n<length;n++)
		{
			I1=(int)((unsigned char)*(lpD+n));
			I2=(int)((unsigned char)*(lpD+length+n));
			I3=(int)((unsigned char)*(lpD+2*length+n));
			I4=(int)((unsigned char)*(lpD+3*length+n));
			I5=(int)((unsigned char)*(lpD+4*length+n));
			if( abs(I1-I2)<Mask_Threshold && abs(I2-I3)<Mask_Threshold && abs(I3-I4)<Mask_Threshold && abs(I4-I5)<Mask_Threshold && abs(I5-I1)<Mask_Threshold )
			{
				*(lpPBM+n)=(char)0;
			}

			f1=I1+I5-2*I3;
			f2=2*(I4-I2);
			q = *(f_atan2 + (f2+nj)*(2*ni+1) + f1+ni);
			if(q<0.0)
				q=q+2*Pi;
			*(lpFO+n) = q/(2*Pi);
		}
	}
	else
	{
		int I_max, I_min, i;
		double ff1, ff2;
		I_max=0;
		I_min=255;
		int *I = new int[nipf];

		for(n=0;n<length;n++)
		{
			for(i=0;i<nipf;i++)
			{
				I[i]=(int)((unsigned char)*(lpD+i*length+n));
				if(I[i]>I_max)
					I_max=I[i];
				if(I[i]<I_min)
					I_min=I[i];
			}
			if( (I_max-I_min)<Mask_Threshold )
				*(lpPBM+n)=(char)0;

			ff1=0.0;
			ff2=0.0;
			for(i=0;i<nipf;i++)
			{
				ff1 = ff1 + I[i]*cos(2.0*i*Pi/nipf);
				ff2 = ff2 + I[i]*sin(2.0*i*Pi/nipf);
			}
			q=atan2(-ff2, ff1);
			if(q<0.0)
				q=q+2*Pi;
			*(lpFO+n) = q/(2.0*Pi);
		}
		delete []I;
	}

	return;
}
void DecodePhaseShift2(char *Image, char *PBM, double *PhaseUW, int width, int height, int *frequency, int nfrequency, int sstep, int LFstep, int half_filter_size, int m_mask)
{
	int ii, jj, k, m, n, length = width*height;
	int ni = sstep*(nfrequency-1)+LFstep;
	double *PhaseW = new double[nfrequency*length];

	double s;
	if(sstep == 3)
	{
		m = 510; n = 255; s = sqrt(3.0);	
	}
	else if(sstep==4)
	{
		m = 255; n = 255; s = 1.0;
	}
	else if(sstep==5)
	{
		m = 510; n = 510; s = 1.0;
	}
	else
	{
		m=1; n=1;
	}

	double *f_atan2 = new double[(2*m+1)*(2*n+1)];
	for(jj=-n; jj<=n; jj++)
	{
		for(ii=-m; ii<=m;ii++)
		{
			f_atan2[(jj+n)*(2*m+1) + ii+m] = atan2( s*jj, 1.0*ii );
		}
	}

	// Filtering all images first
	if( half_filter_size>=1 )
		Average_Filtering_All(Image, width, height, ni, half_filter_size, half_filter_size);

	m = nfrequency*length;
	for(n=0; n<m; n++)
		*(PBM+n) = (char)255;

	//Phase warpping
	for(k=0; k<nfrequency; k++)
	{		
		if(k<nfrequency-1)
			MConventional_PhaseShifting(Image+k*sstep*length, PBM+k*length, PhaseW+k*length, sstep, length, m_mask, f_atan2);
		else
			MConventional_PhaseShifting(Image+k*sstep*length, PBM+k*length, PhaseW+k*length, LFstep, length, m_mask, f_atan2);
	}

	//Incremental phase unwarpped
	double m_s;
	for(k=1; k<nfrequency; k++)
	{
		for(n=0; n<length; n++)
		{
			if(PBM[(k-1)*length+n]==(char)0)
			{
				PBM[k*length+n] = (char)0;
				PhaseW[k*length+n]=1000.0f;
			}
			else
			{
				m_s = PhaseW[(k-1)*length+n]*frequency[k]/frequency[k-1]-PhaseW[k*length+n];
				if(m_s>=0.0)
					PhaseW[k*length+n] += (int)(m_s+0.5);
				else
					PhaseW[k*length+n] += (int)(m_s-0.5);
			}
		}
	}

	for(n=0; n<length; n++)
		PhaseUW[n] = PhaseW[(nfrequency-1)*length+n];

	delete []PhaseW;
	delete []f_atan2;
	return;
}

void RemoveNoiseMedianFilter(float *data, int width, int height, int ksize, float thresh)
{
	Mat src = Mat(height, width, CV_32F, data);
	Mat dst = Mat(10, 10, CV_32F, Scalar(0));

	if(ksize>5)
		ksize = 5;

	medianBlur ( src, dst, ksize);

	for(int jj=0; jj<height; jj++)
		for(int ii=0; ii<width; ii++)
			if(abs(data[ii+jj*width] - dst.at<float>(jj, ii))>thresh)
				data[ii+jj*width] = 0.0;

	return;
}
void RemoveNoiseMedianFilter(double *data, int width, int height, int ksize, float thresh, float *fdata = 0)
{
	bool createMem = false;
	if(fdata == NULL)
	{
		createMem = true;
		fdata = new float[width*height];
	}

	for(int ii=0; ii<width*height; ii++)
		fdata[ii] = (float)data[ii];

	Mat src = Mat(height, width, CV_32F, fdata);
	Mat dst = Mat(10, 10, CV_32F, Scalar(0));

	if(ksize>5)
		ksize = 5;

	medianBlur ( src, dst, ksize);

	for(int jj=0; jj<height; jj++)
		for(int ii=0; ii<width; ii++)
			if(abs(data[ii+jj*width] - dst.at<float>(jj, ii))>thresh)
				data[ii+jj*width] = 0.0;

	if(createMem)
		delete []fdata;
	return;
}

// Author: Xiaotao Duan
//
// This library contains image processing method to detect
// image blurriness.
//
// This library is *not* thread safe because static memory is
// used for performance.
//
// A method to detect whether a given image is blurred or not.
// The algorithm is based on H. Tong, M. Li, H. Zhang, J. He,
// and C. Zhang. "Blur detection for digital images using wavelet
// transform".
//
// To achieve better performance on client side, the method
// is running on four 128x128 portions which compose the 256x256
// central area of the given image. On Nexus One, average time
// to process a single image is ~5 milliseconds.
static const int kDecomposition = 3;
static const int kThreshold = 35;

static const int kMaximumWidth = 2048;
static const int kMaximumHeight = 1536;

static int32_t _smatrix[kMaximumWidth * kMaximumHeight];
static int32_t _arow[kMaximumWidth > kMaximumHeight ? kMaximumWidth : kMaximumHeight];

// Does Haar Wavelet Transformation in place on a given row of a matrix. The matrix is in size of matrix_height * matrix_width and represented in a linear array. 
//Parameter offset_row indicates transformation is performed on which row. offset_column and num_columns indicate column range of the given row.
inline void Haar1DX(int* matrix, int matrix_height, int matrix_width, int offset_row, int offset_column, int num_columns) {
	int32_t* ptr_a = _arow;
	int32_t* ptr_matrix = matrix + offset_row * matrix_width + offset_column;
	int half_num_columns = num_columns / 2;

	int32_t* a_tmp = ptr_a;
	int32_t* matrix_tmp = ptr_matrix;
	for (int j = 0; j < half_num_columns; ++j) {
		*a_tmp++ = (matrix_tmp[0] + matrix_tmp[1]) / 2;
		matrix_tmp += 2;
	}

	int32_t* average = ptr_a;
	a_tmp = ptr_a + half_num_columns;
	matrix_tmp = ptr_matrix;
	for (int j = 0; j < half_num_columns; ++j) {
		*a_tmp++ = *matrix_tmp - *average++;
		matrix_tmp += 2;
	}

	memcpy(ptr_matrix, ptr_a, sizeof(int32_t) * num_columns);
}
// Does Haar Wavelet Transformation in place on a given column of a matrix.
inline void Haar1DY(int* matrix, int matrix_height, int matrix_width, int offset_column, int offset_row, int num_rows) {
	int32_t* ptr_a = _arow;
	int32_t* ptr_matrix = matrix + offset_row * matrix_width + offset_column;
	int half_num_rows = num_rows / 2;
	int two_line_width = matrix_width * 2;

	int32_t* a_tmp = ptr_a;
	int32_t* matrix_tmp = ptr_matrix;
	for (int j = 0; j < half_num_rows; ++j) {
		*a_tmp++ = (matrix_tmp[matrix_width] + matrix_tmp[0]) / 2;
		matrix_tmp += two_line_width;
	}

	int32_t* average = ptr_a;
	a_tmp = ptr_a + half_num_rows;
	matrix_tmp = ptr_matrix;
	for (int j = 0; j < num_rows; j += 2) {
		*a_tmp++ = *matrix_tmp - *average++;
		matrix_tmp += two_line_width;
	}

	for (int j = 0; j < num_rows; ++j) {
		*ptr_matrix = *ptr_a++;
		ptr_matrix += matrix_width;
	}
}
// Does Haar Wavelet Transformation in place for a specified area of a matrix. The matrix size is specified by matrix_width and matrix_height.
// The area on which the transformation is performed is specified by offset_column, num_columns, offset_row and num_rows.
void Haar2D(int* matrix, int matrix_height, int matrix_width, int offset_column, int num_columns, int offset_row, int num_rows) {
	for (int i = offset_row; i < offset_row + num_rows; ++i) {
		Haar1DX(matrix, matrix_height, matrix_width, i, offset_column, num_columns);
	}

	for (int i = offset_column; i < offset_column + num_columns; ++i){
		Haar1DY(matrix, matrix_height, matrix_width, i, offset_row, num_rows);
	}
}
// Reads in a given matrix, does first round HWT and outputs result matrix into target array. This function is used for optimization by avoiding a memory copy. 
//The input matrix has height rows and width columns. The transformation is performed on the given area specified by offset_column, num_columns, offset_row, num_rows. 
// After transformation, the output matrix has num_columns columns and num_rows rows.
void HwtFirstRound(const unsigned char* const data, int height, int width, int offset_column, int num_columns, int offset_row, int num_rows, int32_t* matrix)
{
	int32_t* ptr_a = _arow;
	const unsigned char* ptr_data = data + offset_row * width + offset_column;
	int half_num_columns = num_columns / 2;

	for (int i = 0; i < num_rows; ++i)
	{
		int32_t* a_tmp = ptr_a;
		const unsigned char* data_tmp = ptr_data;
		for (int j = 0; j < half_num_columns; ++j)
		{
			*a_tmp++ = (int32_t)((data_tmp[0] + data_tmp[1]) / 2);
			data_tmp += 2;
		}

		int32_t* average = ptr_a;
		a_tmp = ptr_a + half_num_columns;
		data_tmp = ptr_data;
		for (int j = 0; j < half_num_columns; ++j)
		{
			*a_tmp++ = *data_tmp - *average++;
			data_tmp += 2;
		}

		int32_t* ptr_matrix = matrix + i * num_columns;
		a_tmp = ptr_a;
		for (int j = 0; j < num_columns; ++j)
		{
			*ptr_matrix++ = *a_tmp++;
		}

		ptr_data += width;
	}

	// Column transformation does not involve input data.
	for (int i = 0; i < num_columns; ++i)
		Haar1DY(matrix, num_rows, num_columns, i, 0, num_rows);
}
// Returns the weight of a given point in a certain scale of a matrix after wavelet transformation.
// The point is specified by k and l which are y and x coordinate respectively. Parameter scale tells in which scale the weight is computed, must be 1, 2 or 3 which stands respectively for 1/2, 1/4, and 1/8 of original size.
int ComputeEdgePointWeight(int* matrix, int width, int height, int k, int l, int scale) {
	int r = k >> scale;
	int c = l >> scale;
	int window_row = height >> scale;
	int window_column = width >> scale;

	int v_top_right = pow(matrix[r * width + c + window_column], 2);
	int v_bot_left = pow(matrix[(r + window_row) * width + c], 2);
	int v_bot_right = pow(matrix[(r + window_row) * width + c + window_column], 2);

	int v = sqrt(v_top_right + v_bot_left + v_bot_right);
	return v;
}
// Computes point with maximum weight for a given local window for a given scale. Parameter scaled_width and scaled_height define scaled image size of a certain decomposition level. 
//The window size is defined by window_size. Output value k and l store row (y coordinate) and column (x coordinate) respectively of the point with maximum weight. The maximum weight is returned.
int ComputeLocalMaximum(int* matrix, int width, int height, int scaled_width, int scaled_height, int top, int left, int window_size, int* k, int* l) {
	int max = -1;
	*k = top;
	*l = left;

	for (int i = 0; i < window_size; ++i) {
		for (int j = 0; j < window_size; ++j) {
			int r = top + i;
			int c = left + j;

			int v_top_right = abs(matrix[r * width + c + scaled_width]);
			int v_bot_left = abs(matrix[(r + scaled_height) * width + c]);
			int v_bot_right =
				abs(matrix[(r + scaled_height) * width + c + scaled_width]);
			int v = v_top_right + v_bot_left + v_bot_right;

			if (v > max) {
				max = v;
				*k = r;
				*l = c;
			}
		}
	}

	int r = *k;
	int c = *l;
	int v_top_right = pow(matrix[r * width + c + scaled_width], 2);
	int v_bot_left = pow(matrix[(r + scaled_height) * width + c], 2);
	int v_bot_right = pow(matrix[(r + scaled_height) * width + c + scaled_width], 2);
	int v = sqrt(v_top_right + v_bot_left + v_bot_right);

	return v;
}
// Detects blurriness of a transformed matrix. Blur confidence and extent will be returned through blur_conf and blur_extent. 1 is returned while input matrix is blurred.
int DetectBlur(int* matrix, int width, int height, float* blur_conf, float* blur_extent, float blurThresh) {
	int nedge = 0;
	int nda = 0;
	int nrg = 0;
	int nbrg = 0;

	// For each scale
	for (int current_scale = kDecomposition; current_scale > 0; --current_scale) {
		int scaled_width = width >> current_scale;
		int scaled_height = height >> current_scale;
		int window_size = 16 >> current_scale;  // 2, 4, 8
		// For each window
		for (int r = 0; r + window_size < scaled_height; r += window_size) {
			for (int c = 0; c + window_size < scaled_width; c += window_size) {
				int k, l;
				int emax = ComputeLocalMaximum(matrix, width, height,
					scaled_width, scaled_height, r, c, window_size, &k, &l);
				if (emax > kThreshold) {
					int emax1, emax2, emax3;
					switch (current_scale) {
					case 1:
						emax1 = emax;
						emax2 = ComputeEdgePointWeight(matrix, width, height,
							k << current_scale, l << current_scale, 2);
						emax3 = ComputeEdgePointWeight(matrix, width, height,
							k << current_scale, l << current_scale, 3);
						break;
					case 2:
						emax1 = ComputeEdgePointWeight(matrix, width, height,
							k << current_scale, l << current_scale, 1);
						emax2 = emax;
						emax3 = ComputeEdgePointWeight(matrix, width, height,
							k << current_scale, l << current_scale, 3);
						break;
					case 3:
						emax1 = ComputeEdgePointWeight(matrix, width, height,
							k << current_scale, l << current_scale, 1);
						emax2 = ComputeEdgePointWeight(matrix, width, height,
							k << current_scale, l << current_scale, 2);
						emax3 = emax;
						break;
					}

					nedge++;
					if (emax1 > emax2 && emax2 > emax3) {
						nda++;
					}
					if (emax1 < emax2 && emax2 < emax3) {
						nrg++;
						if (emax1 < kThreshold) {
							nbrg++;
						}
					}
					if (emax2 > emax1 && emax2 > emax3) {
						nrg++;
						if (emax1 < kThreshold) {
							nbrg++;
						}
					}
				}
			}
		}
	}

	// TODO(xiaotao): No edge point at all, blurred or not?
	float per = nedge == 0 ? 0 : (float)nda / nedge;

	*blur_conf = per;
	*blur_extent = (float)nbrg / nrg;

	return per < blurThresh;
}
// Detects blurriness of a given portion of a luminance matrix.
int IsBlurredInner(const unsigned char* const luminance, const int width, const int height, const int left, const int top, const int width_wanted, const int height_wanted, float* const blur, float* const extent, float blurThresh) {
	int32_t* matrix = _smatrix;

	HwtFirstRound(luminance, height, width, left, width_wanted, top, height_wanted, matrix);
	Haar2D(matrix, height_wanted, width_wanted, 0, width_wanted >> 1, 0, height_wanted >> 1);
	Haar2D(matrix, height_wanted, width_wanted, 0, width_wanted >> 2, 0, height_wanted >> 2);

	int blurred = DetectBlur(matrix, width_wanted, height_wanted, blur, extent, blurThresh);

	return blurred;
}
int IsBlurred(const unsigned char* const luminance, const int width, const int height, float &blur, float &extent, float blurThresh) {

	int desired_width = min(kMaximumWidth, width);
	int desired_height = min(kMaximumHeight, height);
	int left = (width - desired_width) >> 1;
	int top = (height - desired_height) >> 1;

	float conf1, extent1;
	int blur1 = IsBlurredInner(luminance, width, height, left, top, desired_width >> 1, desired_height >> 1, &conf1, &extent1, blurThresh);
	float conf2, extent2;
	int blur2 = IsBlurredInner(luminance, width, height, left + (desired_width >> 1), top, desired_width >> 1, desired_height >> 1, &conf2, &extent2, blurThresh);
	float conf3, extent3;
	int blur3 = IsBlurredInner(luminance, width, height, left, top + (desired_height >> 1), desired_width >> 1, desired_height >> 1, &conf3, &extent3, blurThresh);
	float conf4, extent4;
	int blur4 = IsBlurredInner(luminance, width, height, left + (desired_width >> 1), top + (desired_height >> 1), desired_width >> 1, desired_height >> 1, &conf4, &extent4, blurThresh);

	blur = (conf1 + conf2 + conf3 + conf4) / 4;
	extent = (extent1 + extent2 + extent3 + extent4) / 4;
	return blur < blurThresh;
}

double TMatchingFine_ZNCC(double *Pattern, int pattern_size, int hsubset, double *Para, int width, int height, Point2d &POI, int advanced_tech, int Convergence_Criteria, double ZNCCthresh, int InterpAlgo, double *Znssd_reqd)
{
	int i, j, k, m, ii, jj, iii, jjj, iii2, jjj2;
	double II, JJ, iii_n, jjj_n, gx, gy, DIC_Coeff, DIC_Coeff_min, t_1, t_2, t_3, t_4, t_5, t_6, m_F, m_G, t_f, t_ff, t_g, S[6];
	double conv_crit_1 = pow(10.0, -Convergence_Criteria - 2);
	double conv_crit_2 = conv_crit_1*0.1;
	int NN[] = { 6, 12 }, P_Jump_Incr[] = { 1, 1 };
	int nn = NN[advanced_tech], _iter = 0, Iter_Max = 50;
	int p_jump, p_jump_0 = 1, p_jump_incr = P_Jump_Incr[advanced_tech];

	double AA[144], BB[12], CC[12];

	bool createMem = false;
	if (Znssd_reqd == NULL)
	{
		createMem = true;
		Znssd_reqd = new double[6 * (2 * hsubset + 1)*(2 * hsubset + 1)];
	}

	int Pattern_cen_x = pattern_size / 2;
	int Pattern_cen_y = pattern_size / 2;

	double p[12], p_best[12];
	for (i = 0; i < 12; i++)
		p[i] = 0.0;

	nn = NN[advanced_tech];
	int pixel_increment_in_subset[] = { 1, 2, 2, 3 };

	bool printout = false;
	FILE *fp1 = 0, *fp2 = 0;

	/// Iteration: Begin
	bool Break_Flag = false;
	DIC_Coeff_min = 4.0;
	for (p_jump = p_jump_0; p_jump > 0; p_jump -= p_jump_incr)
	{
		for (k = 0; k < Iter_Max; k++)
		{
			m = -1;
			t_1 = 0.0, t_2 = 0.0;
			for (iii = 0; iii < 144; iii++)
				AA[iii] = 0.0;
			for (iii = 0; iii < 12; iii++)
				BB[iii] = 0.0;

			if (printout)
				fp1 = fopen("C:/temp/src.txt", "w+"), fp2 = fopen("C:/temp/tar.txt", "w+");

			for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
			{
				for (iii = -hsubset; iii <= hsubset; iii += p_jump)
				{
					ii = Pattern_cen_x + iii, jj = Pattern_cen_y + jjj;

					if (ii<0 || ii>(width - 1) || jj<0 || jj>(height - 1))
						continue;

					iii2 = iii*iii, jjj2 = jjj*jjj;
					if (advanced_tech == 0)
						II = POI.x + iii + p[0] + p[2] * iii + p[3] * jjj, JJ = POI.y + jjj + p[1] + p[4] * iii + p[5] * jjj;
					else if (advanced_tech == 1)
					{
						II = POI.x + iii + p[0] + p[2] * iii + p[3] * jjj + p[6] * iii2*0.5 + p[7] * jjj2*0.5 + p[8] * iii*jjj;
						JJ = POI.y + jjj + p[1] + p[4] * iii + p[5] * jjj + p[9] * iii2*0.5 + p[10] * jjj2*0.5 + p[11] * iii*jjj;
					}

					if (II<0.0 || II>(double)(width - 1) - (1e-10) || JJ<0.0 || JJ>(double)(height - 1) - (1e-10))
						continue;

					Get_Value_Spline(Para, width, height, II, JJ, S, 0, InterpAlgo);
					m_F = Pattern[ii + jj*pattern_size];
					m_G = S[0], gx = S[1], gy = S[2];
					m++;

					Znssd_reqd[6 * m + 0] = m_F, Znssd_reqd[6 * m + 1] = m_G;
					Znssd_reqd[6 * m + 2] = gx, Znssd_reqd[6 * m + 3] = gy;
					Znssd_reqd[6 * m + 4] = (double)iii, Znssd_reqd[6 * m + 5] = (double)jjj;
					t_1 += m_F, t_2 += m_G;

					if (printout)
						fprintf(fp1, "%e ", m_F), fprintf(fp2, "%e ", m_G);
				}
				if (printout)
					fprintf(fp1, "\n"), fprintf(fp2, "\n");
			}
			if (printout)
				fclose(fp1), fclose(fp2);

			if (k == 0)
			{
				t_f = t_1 / (m + 1);
				t_1 = 0.0;
				for (iii = 0; iii <= m; iii++)
				{
					t_4 = Znssd_reqd[6 * iii + 0] - t_f;
					t_1 += t_4*t_4;
				}
				t_ff = sqrt(t_1);
			}

			t_g = t_2 / (m + 1);
			t_2 = 0.0;
			for (iii = 0; iii <= m; iii++)
			{
				t_5 = Znssd_reqd[6 * iii + 1] - t_g;
				t_2 += t_5*t_5;
			}
			t_2 = sqrt(t_2);

			DIC_Coeff = 0.0;
			for (iii = 0; iii <= m; iii++)
			{
				t_4 = Znssd_reqd[6 * iii + 0] - t_f;
				t_5 = Znssd_reqd[6 * iii + 1] - t_g;
				t_6 = t_5 / t_2 - t_4 / t_ff;
				t_3 = t_6 / t_2;
				gx = Znssd_reqd[6 * iii + 2], gy = Znssd_reqd[6 * iii + 3];
				iii_n = Znssd_reqd[6 * iii + 4], jjj_n = Znssd_reqd[6 * iii + 5];
				CC[0] = gx, CC[1] = gy;
				CC[2] = gx*iii_n, CC[3] = gx*jjj_n;
				CC[4] = gy*iii_n, CC[5] = gy*jjj_n;
				if (advanced_tech == 1)
				{
					CC[6] = gx*iii_n*iii_n*0.5, CC[7] = gx*jjj_n*jjj_n*0.5, CC[8] = gx*iii_n*jjj_n;
					CC[9] = gy*iii_n*iii_n*0.5, CC[10] = gy*jjj_n*jjj_n*0.5, CC[11] = gy*iii_n*jjj_n;
				}
				for (j = 0; j < nn; j++)
				{
					BB[j] += t_3*CC[j];
					for (i = 0; i < nn; i++)
						AA[j*nn + i] += CC[i] * CC[j] / (t_2*t_2);
				}

				DIC_Coeff += t_6*t_6;
			}

			QR_Solution_Double(AA, BB, nn, nn);
			for (iii = 0; iii < nn; iii++)
				p[iii] -= BB[iii];

			if (!IsNumber(p[0]) || abs(p[0]) > hsubset || abs(p[1]) > hsubset)
			{
				if (createMem)
					delete[]Znssd_reqd;
				return false;
			}

			if (DIC_Coeff < DIC_Coeff_min)	// If the iteration does not converge, this can be helpful
			{
				DIC_Coeff_min = DIC_Coeff;
				for (iii = 0; iii < nn; iii++)
					p_best[iii] = p[iii];
			}

			if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
			{
				for (iii = 2; iii < nn; iii++)
				{
					if (fabs(BB[iii]) > conv_crit_2)
						break;
				}
				if (iii == nn)
					Break_Flag = true;
			}

			if (Break_Flag)
				break;
		}
		// In case the iteration converges to "wrong" points, always use the data that lead to the least-square value.
		for (iii = 0; iii < nn; iii++)
			p[iii] = p_best[iii];
	}
	/// Iteration: End

	if (createMem)
		delete[]Znssd_reqd;
	if (abs(p[0]) > hsubset || abs(p[1]) > hsubset || p[0] != p[0] || p[1] != p[1] || DIC_Coeff_min > 1.0 - ZNCCthresh)
		return false;

	POI.x += p[0], POI.y += p[1];

	return 1.0 - DIC_Coeff_min;
}
double TrackingByLK(double *RefPara, double *TarPara, int hsubset, int widthRef, int heightRef, int widthTar, int heightTar, int nchannels, Point2d PR, Point2d PT, int advanced_tech, int Convergence_Criteria, double ZNCCThreshold, int Iter_Max, int InterpAlgo, double *fufv, bool greedySearch, double *ShapePara, double *oPara, double *Timg, double *T, double *Znssd_reqd)
{
	//Also a fine ImgRef matching,.... some differences in the input as compared to TMatchingFine though
	// NOTE: initial guess is of the form of the homography

	int i, j, k, m, ii, kk, iii, jjj, iii_n, jjj_n, iii2, jjj2, ij;
	double II, JJ, a, b, gx, gy, DIC_Coeff, DIC_Coeff_min, t_1, t_2, t_3, t_4, t_5, t_6, t_f, t_ff, t_g, m_F, m_G, S[6];
	double conv_crit_1 = pow(10.0, -Convergence_Criteria - 2);
	double conv_crit_2 = conv_crit_1*0.01;
	int NN[] = { 8, 14, 6, 12 };
	int nn = NN[advanced_tech - 1], nExtraParas = advanced_tech > 2 ? 0 : 2, _iter = 0;
	int p_jump, p_jump_0 = 1, p_jump_incr = 1;
	int TimgS = 2 * hsubset + 1, Tlength = TimgS*TimgS, RefLength = widthRef*heightRef, TarLength = widthTar*heightTar;

	double 	AA[196 * 196], BB[14], CC[14], p[14], ip[14], p_best[14];
	if (ShapePara == NULL)
	{
		for (ii = 0; ii < nn; ii++)
			p[ii] = (ii == nn - nExtraParas ? 1.0 : 0.0);
	}
	else
	{
		if (advanced_tech == 1) //These are basically taylor approximation of the denumerator
		{
			p[0] = ShapePara[2] - PT.x, p[1] = ShapePara[5] - PT.y;
			p[2] = ShapePara[0] - ShapePara[2] * ShapePara[6] - 1.0;
			p[3] = ShapePara[1] - ShapePara[2] * ShapePara[7];
			p[4] = ShapePara[3] - ShapePara[5] * ShapePara[6];
			p[5] = ShapePara[4] - ShapePara[5] * ShapePara[7] - 1.0;
			p[6] = 1.0, p[7] = 0.0;
		}
		else
		{
			p[0] = ShapePara[2] - PT.x, p[1] = ShapePara[5] - PT.y;
			p[2] = ShapePara[0] - ShapePara[2] * ShapePara[6] - 1.0;
			p[3] = ShapePara[1] - ShapePara[2] * ShapePara[7];
			p[4] = ShapePara[3] - ShapePara[5] * ShapePara[6];
			p[5] = ShapePara[4] - ShapePara[5] * ShapePara[7] - 1.0;
			p[6] = -0.5*ShapePara[0] * ShapePara[6];
			p[7] = -0.5*ShapePara[1] * ShapePara[7];
			p[8] = -(ShapePara[0] * ShapePara[7] + ShapePara[1] * ShapePara[6]);
			p[9] = -0.5*ShapePara[3] * ShapePara[6];
			p[10] = -0.5*ShapePara[4] * ShapePara[7];
			p[11] = -(ShapePara[3] * ShapePara[7] + ShapePara[4] * ShapePara[6]);
			p[12] = 1.0, p[13] = 0.0;
		}
	}
	for (i = 0; i < nn; i++)
		ip[i] = p[i];

	bool createMem = false;
	if (Timg == NULL)
	{
		Timg = new double[Tlength*nchannels];
		T = new double[2 * Tlength*nchannels];
		Znssd_reqd = new double[6 * Tlength];
		createMem = true;
	}

	for (jjj = -hsubset; jjj <= hsubset; jjj++)
	{
		for (iii = -hsubset; iii <= hsubset; iii++)
		{
			II = PR.x + iii, JJ = PR.y + jjj;
			for (kk = 0; kk < nchannels; kk++)
			{
				Get_Value_Spline(RefPara + kk*RefLength, widthRef, heightRef, II, JJ, S, -1, InterpAlgo);
				Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength] = S[0];
			}
		}
	}

	bool printout = false; FILE *fp = 0;
	if (printout)
	{
		fp = fopen("C:/temp/src.txt", "w+");
		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
				for (kk = 0; kk < nchannels; kk++)
					fprintf(fp, "%.2f ", Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	if (greedySearch)
	{
		/// Let's start with only translation and only match the at the highest level of the pyramid
		double zncc, znccMin;
		for (p_jump = p_jump_0; p_jump > 0; p_jump -= (advanced_tech == 0 ? 1 : 2))
		{
			znccMin = 1e10;
			for (k = 0; k < Iter_Max; k++)
			{
				t_1 = 0.0;
				t_2 = 0.0;
				for (i = 0; i < 4; i++)
					AA[i] = 0.0;
				for (i = 0; i < 2; i++)
					BB[i] = 0.0;

				for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
				{
					for (iii = -hsubset; iii <= hsubset; iii += p_jump)
					{
						II = PT.x + iii + p[0], JJ = PT.y + jjj + p[1];
						if (II<0.0 || II>(double)(widthTar - 2) || JJ<0.0 || JJ>(double)(heightTar - 2))
							continue;

						for (kk = 0; kk < nchannels; kk++)
						{
							Get_Value_Spline(TarPara + kk*TarLength, widthTar, heightTar, II, JJ, S, 0, InterpAlgo);

							m_F = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength];
							m_G = S[0];

							t_3 = m_G - m_F;
							CC[0] = S[1], CC[1] = S[2];

							for (i = 0; i < 2; i++)
								BB[i] += t_3*CC[i];

							for (j = 0; j < 2; j++)
								for (i = 0; i < 2; i++)
									AA[j * 2 + i] += CC[i] * CC[j];

							t_1 += t_3*t_3, t_2 += m_F*m_F;
						}
					}
				}
				zncc = t_1 / t_2;

				QR_Solution_Double(AA, BB, 2, 2);
				for (i = 0; i < 2; i++)
					p[i] -= BB[i];

				if (abs(p[0]) > 0.005*widthTar || abs(p[1]) > 0.005*widthTar || !IsFiniteNumber(p[0]))
				{
					if (createMem)
					{
						delete[]Timg;
						delete[]T;
					}
					return 0.0;
				}

				if (zncc < znccMin)	// If the iteration does not converge, this can be helpful
				{
					znccMin = zncc;
					p_best[0] = p[0], p_best[1] = p[1];
				}

				if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
					break;
			}
		}
		p[0] = p_best[0], p[1] = p_best[1];
	}

	/// DIC Iteration: Begin
	bool Break_Flag;
	DIC_Coeff_min = 1e10;
	for (p_jump = p_jump_0; p_jump > 0; p_jump -= p_jump_incr)
	{
		Break_Flag = false;
		for (k = 0; k < Iter_Max; k++)
		{
			m = -1, t_1 = 0.0, t_2 = 0.0;
			for (iii = 0; iii < nn*nn; iii++)
				AA[iii] = 0.0;
			for (iii = 0; iii < nn; iii++)
				BB[iii] = 0.0;

			a = p[nn - 2], b = p[nn - 1];
			if (printout)
				fp = fopen("C:/temp/tar.txt", "w+");

			for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
			{
				for (iii = -hsubset; iii <= hsubset; iii += p_jump)
				{
					if (advanced_tech % 2 == 1)
						II = PT.x + iii + p[0] + p[2] * iii + p[3] * jjj, JJ = PT.y + jjj + p[1] + p[4] * iii + p[5] * jjj;
					else if (advanced_tech == 0)
					{
						iii2 = iii*iii, jjj2 = jjj*jjj, ij = iii*jjj;
						II = PT.x + iii + p[0] + p[2] * iii + p[3] * jjj + p[6] * iii2*0.5 + p[7] * jjj2*0.5 + p[8] * ij;
						JJ = PT.y + jjj + p[1] + p[4] * iii + p[5] * jjj + p[9] * iii2*0.5 + p[10] * jjj2*0.5 + p[11] * ij;
					}

					if (II<5.0 || II>(double)(widthTar - 5) || JJ<5.0 || JJ>(double)(heightTar - 5))
						continue;

					for (kk = 0; kk < nchannels; kk++)
					{
						m_F = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength];
						Get_Value_Spline(TarPara + kk*TarLength, widthTar, heightTar, II, JJ, S, 0, InterpAlgo);
						m_G = S[0], gx = S[1], gy = S[2];
						m++;

						if (printout)
							fprintf(fp, "%.2f ", m_G);

						if (advanced_tech < 2)
						{
							t_3 = a*m_G + b - m_F, t_4 = a;

							t_5 = t_4*gx, t_6 = t_4*gy;
							CC[0] = t_5, CC[1] = t_6;
							CC[2] = t_5*iii, CC[3] = t_5*jjj;
							CC[4] = t_6*iii, CC[5] = t_6*jjj;
							CC[6] = m_G, CC[7] = 1.0;

							for (j = 0; j < nn; j++)
								BB[j] += t_3*CC[j];

							for (j = 0; j < nn; j++)
								for (i = 0; i < nn; i++)
									AA[j*nn + i] += CC[i] * CC[j];

							t_1 += t_3*t_3, t_2 += m_F*m_F;
						}
						else
						{
							Znssd_reqd[6 * m + 0] = m_F, Znssd_reqd[6 * m + 1] = m_G;
							Znssd_reqd[6 * m + 2] = gx, Znssd_reqd[6 * m + 3] = gy;
							Znssd_reqd[6 * m + 4] = (double)iii, Znssd_reqd[6 * m + 5] = (double)jjj;
							t_1 += m_F, t_2 += m_G;
						}
					}
				}
				if (printout)
					fprintf(fp, "\n");
			}
			if (printout)
				fclose(fp);

			if (advanced_tech < 3)
			{
				DIC_Coeff = t_1 / t_2;
				if (t_2 < 10.0e-9)
					break;
			}
			else
			{
				if (k == 0)
				{
					t_f = t_1 / (m + 1);
					t_1 = 0.0;
					for (iii = 0; iii <= m; iii++)
					{
						t_4 = Znssd_reqd[6 * iii + 0] - t_f;
						t_1 += t_4*t_4;
					}
					t_ff = sqrt(t_1);
				}
				t_g = t_2 / (m + 1);
				t_2 = 0.0;
				for (iii = 0; iii <= m; iii++)
				{
					t_5 = Znssd_reqd[6 * iii + 1] - t_g;
					t_2 += t_5*t_5;
				}
				t_2 = sqrt(t_2);

				DIC_Coeff = 0.0;
				for (iii = 0; iii <= m; iii++)
				{
					t_4 = Znssd_reqd[6 * iii + 0] - t_f;
					t_5 = Znssd_reqd[6 * iii + 1] - t_g;
					t_6 = t_5 / t_2 - t_4 / t_ff;
					t_3 = t_6 / t_2;
					gx = Znssd_reqd[6 * iii + 2], gy = Znssd_reqd[6 * iii + 3];
					iii_n = Znssd_reqd[6 * iii + 4], jjj_n = Znssd_reqd[6 * iii + 5];
					CC[0] = gx, CC[1] = gy;
					CC[2] = gx*iii_n, CC[3] = gx*jjj_n;
					CC[4] = gy*iii_n, CC[5] = gy*jjj_n;
					if (advanced_tech == 4)
					{
						CC[6] = gx*iii_n*iii_n*0.5, CC[7] = gx*jjj_n*jjj_n*0.5, CC[8] = gx*iii_n*jjj_n;
						CC[9] = gy*iii_n*iii_n*0.5, CC[10] = gy*jjj_n*jjj_n*0.5, CC[11] = gy*iii_n*jjj_n;
					}
					for (j = 0; j < nn; j++)
					{
						BB[j] += t_3*CC[j];
						for (i = 0; i < nn; i++)
							AA[j*nn + i] += CC[i] * CC[j] / (t_2*t_2);
					}

					DIC_Coeff += t_6*t_6;
				}
				if (!IsNumber(DIC_Coeff))
					return 9e9;
				if (!IsFiniteNumber(DIC_Coeff))
					return 9e9;
			}

			QR_Solution_Double(AA, BB, nn, nn);
			for (iii = 0; iii < nn; iii++)
				p[iii] -= BB[iii];

			if (abs(p[0]) > 0.005*widthTar || abs(p[1]) > 0.005*widthTar || !IsFiniteNumber(p[0]))
			{
				if (createMem)
				{
					delete[]Timg;
					delete[]T;
				}
				return 0.0;
			}

			if (DIC_Coeff < DIC_Coeff_min)	// If the iteration does not converge, this can be helpful
			{
				DIC_Coeff_min = DIC_Coeff;
				for (iii = 0; iii < nn; iii++)
					p_best[iii] = p[iii];
				if (!IsNumber(p[0]) || !IsNumber(p[1]))
					return 9e9;
			}

			if (advanced_tech < 3)
			{
				if (abs(p[0] - ip[0]) > hsubset || abs(p[1] - ip[1]) > hsubset)
					return 9e9;
				if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
				{
					for (iii = 2; iii < nn - nExtraParas; iii++)
					{
						if (fabs(BB[iii]) > conv_crit_2)
							break;
					}
					if (iii == nn - nExtraParas)
						Break_Flag = true;
				}
			}
			else if (advanced_tech == 3)
			{
				if (abs(p[0] - ip[0]) > hsubset || abs(p[1] - ip[1]) > hsubset)
					return 9e9;
				if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
				{
					for (iii = 2; iii < nn - nExtraParas; iii++)
						if (fabs(BB[iii]) > conv_crit_2)
							break;
					if (iii == nn - nExtraParas)
						Break_Flag = true;
				}
			}
			else
			{
				if (abs(p[0] - ip[0]) > hsubset || abs(p[1] - ip[1]) > hsubset)
					return 9e9;
				if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
				{
					for (iii = 2; iii < nn; iii++)
						if (fabs(BB[iii]) > conv_crit_2)
							break;
					if (iii == nn)
						Break_Flag = true;
				}
			}
			if (Break_Flag)
				break;
		}
		_iter += k;

		// In case the iteration converges to "wrong" points, always use the data that lead to the least-square value.
		for (iii = 0; iii < nn; iii++)
			p[iii] = p_best[iii];
	}
	/// DIC Iteration: End

	//Now, dont really trust the pssad error too much, compute zncc score instead! 
	//They are usually close when the convergence goes smothly, but in case of trouble, zncc is more reliable.
	double ZNCC;
	if (advanced_tech < 3)
	{
		int m = 0;
		double t_1, t_2, t_3, t_4, t_5, t_f = 0.0, t_g = 0.0;
		if (printout)
			fp = fopen("C:/temp/tar.txt", "w+");
		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
			{
				II = PT.x + iii + p[0] + p[2] * iii + p[3] * jjj;
				JJ = PT.y + jjj + p[1] + p[4] * iii + p[5] * jjj;

				if (II<0.0 || II>(double)(widthTar - 1) || JJ<0.0 || JJ>(double)(heightTar - 1))
					continue;

				for (kk = 0; kk < nchannels; kk++)
				{
					Get_Value_Spline(TarPara + kk*TarLength, widthTar, heightTar, II, JJ, S, -1, InterpAlgo);
					if (printout)
						fprintf(fp, "%.4f ", S[0]);

					T[2 * m] = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength];
					T[2 * m + 1] = S[0];
					t_f += T[2 * m];
					t_g += T[2 * m + 1];
					m++;
				}
			}
			if (printout)
				fprintf(fp, "\n");
		}
		if (printout)
			fclose(fp);

		t_f = t_f / m;
		t_g = t_g / m;
		t_1 = 0.0, t_2 = 0.0, t_3 = 0.0;
		for (i = 0; i < m; i++)
		{
			t_4 = T[2 * i] - t_f;
			t_5 = T[2 * i + 1] - t_g;
			t_1 += 1.0*t_4*t_5;
			t_2 += 1.0*t_4*t_4;
			t_3 += 1.0*t_5*t_5;
		}

		t_2 = sqrt(t_2*t_3);
		if (t_2 < 1e-10)
			t_2 = 1e-10;

		ZNCC = t_1 / t_2; //This is the zncc score
		if (abs(ZNCC) > 1.0)
			ZNCC = 0.0;
	}
	else
		ZNCC = 1.0 - 0.5*DIC_Coeff_min; //from ZNSSD to ZNCC

	if (abs(p[0]) > 0.005*widthTar || abs(p[1]) > 0.005*widthTar || p[0] != p[0] || p[1] != p[1] || ZNCC < ZNCCThreshold)
	{
		if (createMem)
		{
			delete[]Timg;
			delete[]T;
		}
		return 0.0;
	}

	fufv[0] = p_best[0], fufv[1] = p_best[1];
	if (oPara != NULL)
		for (ii = 0; ii < 8; ii++)
			oPara[ii] = p_best[ii];

	if (createMem)
	{
		delete[]Timg;
		delete[]T;
	}
	return ZNCC;
}
double TrackingByLK(float *RefPara, float *TarPara, int hsubset, int widthRef, int heightRef, int widthTar, int heightTar, int nchannels, Point2d PR, Point2d PT, int advanced_tech, int Convergence_Criteria, double ZNCCThreshold, int Iter_Max, int InterpAlgo, double *fufv, bool greedySearch, double *ShapePara, double *oPara, double *Timg, double *T, double *Znssd_reqd)
{
	//Also a fine ImgRef matching,.... some differences in the input as compared to TMatchingFine though
	// NOTE: initial guess is of the form of the homography

	int i, j, k, m, ii, kk, iii, jjj, iii_n, jjj_n, iii2, jjj2, ij;
	double II, JJ, a, b, gx, gy, DIC_Coeff, DIC_Coeff_min, t_1, t_2, t_3, t_4, t_5, t_6, t_f, t_ff, t_g, m_F, m_G, S[6];
	double conv_crit_1 = pow(10.0, -Convergence_Criteria - 2);
	double conv_crit_2 = conv_crit_1*0.01;
	int NN[] = { 8, 14, 6, 12 };
	int nn = NN[advanced_tech - 1], nExtraParas = advanced_tech > 2 ? 0 : 2, _iter = 0;
	int p_jump, p_jump_0 = 1, p_jump_incr = 1;
	int TimgS = 2 * hsubset + 1, Tlength = TimgS*TimgS, RefLength = widthRef*heightRef, TarLength = widthTar*heightTar;

	double 	AA[196 * 196], BB[14], CC[14], p[14], ip[14], p_best[14];
	if (ShapePara == NULL)
	{
		for (ii = 0; ii < nn; ii++)
			p[ii] = (ii == nn - nExtraParas ? 1.0 : 0.0);
	}
	else
	{
		if (advanced_tech == 1) //These are basically taylor approximation of the denumerator
		{
			p[0] = ShapePara[2] - PT.x, p[1] = ShapePara[5] - PT.y;
			p[2] = ShapePara[0] - ShapePara[2] * ShapePara[6] - 1.0;
			p[3] = ShapePara[1] - ShapePara[2] * ShapePara[7];
			p[4] = ShapePara[3] - ShapePara[5] * ShapePara[6];
			p[5] = ShapePara[4] - ShapePara[5] * ShapePara[7] - 1.0;
			p[6] = 1.0, p[7] = 0.0;
		}
		else
		{
			p[0] = ShapePara[2] - PT.x, p[1] = ShapePara[5] - PT.y;
			p[2] = ShapePara[0] - ShapePara[2] * ShapePara[6] - 1.0;
			p[3] = ShapePara[1] - ShapePara[2] * ShapePara[7];
			p[4] = ShapePara[3] - ShapePara[5] * ShapePara[6];
			p[5] = ShapePara[4] - ShapePara[5] * ShapePara[7] - 1.0;
			p[6] = -0.5*ShapePara[0] * ShapePara[6];
			p[7] = -0.5*ShapePara[1] * ShapePara[7];
			p[8] = -(ShapePara[0] * ShapePara[7] + ShapePara[1] * ShapePara[6]);
			p[9] = -0.5*ShapePara[3] * ShapePara[6];
			p[10] = -0.5*ShapePara[4] * ShapePara[7];
			p[11] = -(ShapePara[3] * ShapePara[7] + ShapePara[4] * ShapePara[6]);
			p[12] = 1.0, p[13] = 0.0;
		}
	}
	for (i = 0; i < nn; i++)
		ip[i] = p[i];

	bool createMem = false;
	if (Timg == NULL)
	{
		Timg = new double[Tlength*nchannels];
		T = new double[2 * Tlength*nchannels];
		Znssd_reqd = new double[6 * Tlength];
		createMem = true;
	}

	for (jjj = -hsubset; jjj <= hsubset; jjj++)
	{
		for (iii = -hsubset; iii <= hsubset; iii++)
		{
			II = PR.x + iii, JJ = PR.y + jjj;
			for (kk = 0; kk < nchannels; kk++)
			{
				Get_Value_Spline(RefPara + kk*RefLength, widthRef, heightRef, II, JJ, S, -1, InterpAlgo);
				Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength] = S[0];
			}
		}
	}

	bool printout = false; FILE *fp = 0;
	if (printout)
	{
		fp = fopen("C:/temp/src.txt", "w+");
		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
				for (kk = 0; kk < nchannels; kk++)
					fprintf(fp, "%.2f ", Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	if (greedySearch)
	{
		/// Let's start with only translation and only match the at the highest level of the pyramid
		double zncc, znccMin;
		for (p_jump = p_jump_0; p_jump > 0; p_jump -= (advanced_tech == 0 ? 1 : 2))
		{
			znccMin = 1e10;
			for (k = 0; k < Iter_Max; k++)
			{
				t_1 = 0.0;
				t_2 = 0.0;
				for (i = 0; i < 4; i++)
					AA[i] = 0.0;
				for (i = 0; i < 2; i++)
					BB[i] = 0.0;

				for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
				{
					for (iii = -hsubset; iii <= hsubset; iii += p_jump)
					{
						II = PT.x + iii + p[0], JJ = PT.y + jjj + p[1];
						if (II<0.0 || II>(double)(widthTar - 2) || JJ<0.0 || JJ>(double)(heightTar - 2))
							continue;

						for (kk = 0; kk < nchannels; kk++)
						{
							Get_Value_Spline(TarPara + kk*TarLength, widthTar, heightTar, II, JJ, S, 0, InterpAlgo);

							m_F = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength];
							m_G = S[0];

							t_3 = m_G - m_F;
							CC[0] = S[1], CC[1] = S[2];

							for (i = 0; i < 2; i++)
								BB[i] += t_3*CC[i];

							for (j = 0; j < 2; j++)
								for (i = 0; i < 2; i++)
									AA[j * 2 + i] += CC[i] * CC[j];

							t_1 += t_3*t_3, t_2 += m_F*m_F;
						}
					}
				}
				zncc = t_1 / t_2;

				QR_Solution_Double(AA, BB, 2, 2);
				for (i = 0; i < 2; i++)
					p[i] -= BB[i];

				if (abs(p[0]) > 0.005*widthTar || abs(p[1]) > 0.005*widthTar || !IsFiniteNumber(p[0]))
				{
					if (createMem)
					{
						delete[]Timg;
						delete[]T;
					}
					return 0.0;
				}

				if (zncc < znccMin)	// If the iteration does not converge, this can be helpful
				{
					znccMin = zncc;
					p_best[0] = p[0], p_best[1] = p[1];
				}

				if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
					break;
			}
		}
		p[0] = p_best[0], p[1] = p_best[1];
	}

	/// DIC Iteration: Begin
	bool Break_Flag;
	DIC_Coeff_min = 1e10;
	for (p_jump = p_jump_0; p_jump > 0; p_jump -= p_jump_incr)
	{
		Break_Flag = false;
		for (k = 0; k < Iter_Max; k++)
		{
			m = -1, t_1 = 0.0, t_2 = 0.0;
			for (iii = 0; iii < nn*nn; iii++)
				AA[iii] = 0.0;
			for (iii = 0; iii < nn; iii++)
				BB[iii] = 0.0;

			a = p[nn - 2], b = p[nn - 1];
			if (printout)
				fp = fopen("C:/temp/tar.txt", "w+");

			for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
			{
				for (iii = -hsubset; iii <= hsubset; iii += p_jump)
				{
					if (advanced_tech % 2 == 1)
						II = PT.x + iii + p[0] + p[2] * iii + p[3] * jjj, JJ = PT.y + jjj + p[1] + p[4] * iii + p[5] * jjj;
					else if (advanced_tech == 0)
					{
						iii2 = iii*iii, jjj2 = jjj*jjj, ij = iii*jjj;
						II = PT.x + iii + p[0] + p[2] * iii + p[3] * jjj + p[6] * iii2*0.5 + p[7] * jjj2*0.5 + p[8] * ij;
						JJ = PT.y + jjj + p[1] + p[4] * iii + p[5] * jjj + p[9] * iii2*0.5 + p[10] * jjj2*0.5 + p[11] * ij;
					}

					if (II<5.0 || II>(double)(widthTar - 5) || JJ<5.0 || JJ>(double)(heightTar - 5))
						continue;

					for (kk = 0; kk < nchannels; kk++)
					{
						m_F = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength];
						Get_Value_Spline(TarPara + kk*TarLength, widthTar, heightTar, II, JJ, S, 0, InterpAlgo);
						m_G = S[0], gx = S[1], gy = S[2];
						m++;

						if (printout)
							fprintf(fp, "%.2f ", m_G);

						if (advanced_tech < 2)
						{
							t_3 = a*m_G + b - m_F, t_4 = a;

							t_5 = t_4*gx, t_6 = t_4*gy;
							CC[0] = t_5, CC[1] = t_6;
							CC[2] = t_5*iii, CC[3] = t_5*jjj;
							CC[4] = t_6*iii, CC[5] = t_6*jjj;
							CC[6] = m_G, CC[7] = 1.0;

							for (j = 0; j < nn; j++)
								BB[j] += t_3*CC[j];

							for (j = 0; j < nn; j++)
								for (i = 0; i < nn; i++)
									AA[j*nn + i] += CC[i] * CC[j];

							t_1 += t_3*t_3, t_2 += m_F*m_F;
						}
						else
						{
							Znssd_reqd[6 * m + 0] = m_F, Znssd_reqd[6 * m + 1] = m_G;
							Znssd_reqd[6 * m + 2] = gx, Znssd_reqd[6 * m + 3] = gy;
							Znssd_reqd[6 * m + 4] = (double)iii, Znssd_reqd[6 * m + 5] = (double)jjj;
							t_1 += m_F, t_2 += m_G;
						}
					}
				}
				if (printout)
					fprintf(fp, "\n");
			}
			if (printout)
				fclose(fp);

			if (advanced_tech < 3)
			{
				DIC_Coeff = t_1 / t_2;
				if (t_2 < 10.0e-9)
					break;
			}
			else
			{
				if (k == 0)
				{
					t_f = t_1 / (m + 1);
					t_1 = 0.0;
					for (iii = 0; iii <= m; iii++)
					{
						t_4 = Znssd_reqd[6 * iii + 0] - t_f;
						t_1 += t_4*t_4;
					}
					t_ff = sqrt(t_1);
				}
				t_g = t_2 / (m + 1);
				t_2 = 0.0;
				for (iii = 0; iii <= m; iii++)
				{
					t_5 = Znssd_reqd[6 * iii + 1] - t_g;
					t_2 += t_5*t_5;
				}
				t_2 = sqrt(t_2);

				DIC_Coeff = 0.0;
				for (iii = 0; iii <= m; iii++)
				{
					t_4 = Znssd_reqd[6 * iii + 0] - t_f;
					t_5 = Znssd_reqd[6 * iii + 1] - t_g;
					t_6 = t_5 / t_2 - t_4 / t_ff;
					t_3 = t_6 / t_2;
					gx = Znssd_reqd[6 * iii + 2], gy = Znssd_reqd[6 * iii + 3];
					iii_n = Znssd_reqd[6 * iii + 4], jjj_n = Znssd_reqd[6 * iii + 5];
					CC[0] = gx, CC[1] = gy;
					CC[2] = gx*iii_n, CC[3] = gx*jjj_n;
					CC[4] = gy*iii_n, CC[5] = gy*jjj_n;
					if (advanced_tech == 4)
					{
						CC[6] = gx*iii_n*iii_n*0.5, CC[7] = gx*jjj_n*jjj_n*0.5, CC[8] = gx*iii_n*jjj_n;
						CC[9] = gy*iii_n*iii_n*0.5, CC[10] = gy*jjj_n*jjj_n*0.5, CC[11] = gy*iii_n*jjj_n;
					}
					for (j = 0; j < nn; j++)
					{
						BB[j] += t_3*CC[j];
						for (i = 0; i < nn; i++)
							AA[j*nn + i] += CC[i] * CC[j] / (t_2*t_2);
					}

					DIC_Coeff += t_6*t_6;
				}
				if (!IsNumber(DIC_Coeff))
					return 9e9;
				if (!IsFiniteNumber(DIC_Coeff))
					return 9e9;
			}

			QR_Solution_Double(AA, BB, nn, nn);
			for (iii = 0; iii < nn; iii++)
				p[iii] -= BB[iii];

			if (abs(p[0]) > 0.005*widthTar || abs(p[1]) > 0.005*widthTar || !IsFiniteNumber(p[0]))
			{
				if (createMem)
				{
					delete[]Timg;
					delete[]T;
				}
				return 0.0;
			}

			if (DIC_Coeff < DIC_Coeff_min)	// If the iteration does not converge, this can be helpful
			{
				DIC_Coeff_min = DIC_Coeff;
				for (iii = 0; iii < nn; iii++)
					p_best[iii] = p[iii];
				if (!IsNumber(p[0]) || !IsNumber(p[1]))
					return 0.0;
			}

			if (advanced_tech < 3)
			{
				if (abs(p[0] - ip[0]) > hsubset || abs(p[1] - ip[1]) > hsubset)
					return 0.0;
				if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
				{
					for (iii = 2; iii < nn - nExtraParas; iii++)
					{
						if (fabs(BB[iii]) > conv_crit_2)
							break;
					}
					if (iii == nn - nExtraParas)
						Break_Flag = true;
				}
			}
			else if (advanced_tech == 3)
			{
				if (abs(p[0] - ip[0]) > hsubset || abs(p[1] - ip[1]) > hsubset)
					return 0.0;
				if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
				{
					for (iii = 2; iii < nn - nExtraParas; iii++)
						if (fabs(BB[iii]) > conv_crit_2)
							break;
					if (iii == nn - nExtraParas)
						Break_Flag = true;
				}
			}
			else
			{
				if (abs(p[0] - ip[0]) > hsubset || abs(p[1] - ip[1]) > hsubset)
					return 0.0;
				if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
				{
					for (iii = 2; iii < nn; iii++)
						if (fabs(BB[iii]) > conv_crit_2)
							break;
					if (iii == nn)
						Break_Flag = true;
				}
			}
			if (Break_Flag)
				break;
		}
		_iter += k;

		// In case the iteration converges to "wrong" points, always use the data that lead to the least-square value.
		for (iii = 0; iii < nn; iii++)
			p[iii] = p_best[iii];
	}
	/// DIC Iteration: End

	//Now, dont really trust the pssad error too much, compute zncc score instead! 
	//They are usually close when the convergence goes smothly, but in case of trouble, zncc is more reliable.
	double ZNCC;
	if (advanced_tech < 3)
	{
		int m = 0;
		double t_1, t_2, t_3, t_4, t_5, t_f = 0.0, t_g = 0.0;
		if (printout)
			fp = fopen("C:/temp/tar.txt", "w+");
		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
			{
				II = PT.x + iii + p[0] + p[2] * iii + p[3] * jjj;
				JJ = PT.y + jjj + p[1] + p[4] * iii + p[5] * jjj;

				if (II<0.0 || II>(double)(widthTar - 1) || JJ<0.0 || JJ>(double)(heightTar - 1))
					continue;

				for (kk = 0; kk < nchannels; kk++)
				{
					Get_Value_Spline(TarPara + kk*TarLength, widthTar, heightTar, II, JJ, S, -1, InterpAlgo);
					if (printout)
						fprintf(fp, "%.4f ", S[0]);

					T[2 * m] = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength];
					T[2 * m + 1] = S[0];
					t_f += T[2 * m];
					t_g += T[2 * m + 1];
					m++;
				}
			}
			if (printout)
				fprintf(fp, "\n");
		}
		if (printout)
			fclose(fp);

		t_f = t_f / m;
		t_g = t_g / m;
		t_1 = 0.0, t_2 = 0.0, t_3 = 0.0;
		for (i = 0; i < m; i++)
		{
			t_4 = T[2 * i] - t_f;
			t_5 = T[2 * i + 1] - t_g;
			t_1 += 1.0*t_4*t_5;
			t_2 += 1.0*t_4*t_4;
			t_3 += 1.0*t_5*t_5;
		}

		t_2 = sqrt(t_2*t_3);
		if (t_2 < 1e-10)
			t_2 = 1e-10;

		ZNCC = t_1 / t_2; //This is the zncc score
		if (abs(ZNCC) > 1.0)
			ZNCC = 0.0;
	}
	else
		ZNCC = 1.0 - 0.5*DIC_Coeff_min; //from ZNSSD to ZNCC

	if (abs(p[0]) > 0.005*widthTar || abs(p[1]) > 0.005*widthTar || p[0] != p[0] || p[1] != p[1] || ZNCC < ZNCCThreshold)
	{
		if (createMem)
		{
			delete[]Timg;
			delete[]T;
		}
		return 0.0;
	}

	fufv[0] = p_best[0], fufv[1] = p_best[1];
	if (oPara != NULL)
		for (ii = 0; ii < 8; ii++)
			oPara[ii] = p_best[ii];

	if (createMem)
	{
		delete[]Timg;
		delete[]T;
	}
	return ZNCC;
}