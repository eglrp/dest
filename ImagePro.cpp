#include "ImagePro.h"
#include "Ultility.h"

void filter1D_row_Double(double *kernel, int k_size, double *in, double *out, int width, int height)
{
	int ii, jj, i_in, iKernel, k_mid = k_size / 2;
	double t_value;

	for (ii = 0; ii < width*height; ii++)
		out[ii] = 0.0;

	for (jj = 0; jj < height; jj++)
	{
		for (ii = k_mid; ii < width - k_mid; ii++)
		{
			t_value = 0.0;
			for (iKernel = 0; iKernel < k_size; iKernel++)
			{
				i_in = ii + (iKernel - k_mid);
				t_value += kernel[iKernel] * in[i_in + jj*width];
			}
			out[ii + jj*width] = t_value;
		}
	}

	return;
}
void filter1D_row(double *kernel, int k_size, unsigned char *in, float *out, int width, int height)
{
	int ii, jj, i_in, iKernel, k_mid = k_size / 2;
	float t_value;

	for (ii = 0; ii < width*height; ii++)
		out[ii] = 0.0;

	for (jj = 0; jj < height; jj++)
	{
		for (ii = k_mid; ii < width - k_mid; ii++)
		{
			t_value = 0.0;
			for (iKernel = 0; iKernel < k_size; iKernel++)
			{
				i_in = ii + (iKernel - k_mid);
				t_value += kernel[iKernel] * ((int)((in[i_in + jj*width])));
			}
			out[ii + jj*width] = t_value;
		}
	}

	return;
}
void filter1D_row(double *kernel, int k_size, char *in, float *out, int width, int height)
{
	int ii, jj, i_in, iKernel, k_mid = k_size/ 2;
	float t_value;

	for (ii = 0; ii < width*height; ii++)
		out[ii] = 0.0;

	for (jj = 0; jj < height; jj++)
	{
		for (ii = k_mid; ii < width - k_mid; ii++)
		{
			t_value = 0.0;
			for (iKernel = 0; iKernel < k_size; iKernel++)
			{
				i_in = ii + (iKernel - k_mid);
				t_value += kernel[iKernel] * ((int)((unsigned char)(in[i_in + jj*width])));
			}
			out[ii + jj*width] = t_value;
		}
	}

	return;
}
void filter1D_col(double *kernel, int k_size, float *in, float *out, int width, int height, double &i_max)
{
	int ii, jj, j_in, jKernel, k_mid = (k_size - 1) / 2;
	float t_value;

	for (ii = 0; ii < width*height; ii++)
		out[ii] = 0.0;

	i_max = 1.0;
	for (ii = k_mid; ii < width - k_mid; ii++)
	{
		for (jj = k_mid; jj < height - k_mid; jj++)
		{
			t_value = 0;
			for (jKernel = 0; jKernel < k_size; jKernel++)
			{
				j_in = jj + (jKernel - k_mid);
				t_value += kernel[jKernel] * in[ii + j_in*width];
			}
			out[ii + jj*width] = t_value;
			if (t_value > i_max)
				i_max = t_value;
		}
	}

	return;
}
void filter1D_row(double *kernel, int k_size, unsigned char *in, double *out, int width, int height)
{
	int ii, jj, i_in, iKernel, k_mid = (k_size - 1) / 2;
	double t_value;

	for (ii = 0; ii < width*height; ii++)
		out[ii] = 0.0;

	for (jj = 0; jj < height; jj++)
	{
		for (ii = k_mid; ii < width - k_mid; ii++)
		{
			t_value = 0.0;
			for (iKernel = 0; iKernel < k_size; iKernel++)
			{
				i_in = ii + (iKernel - k_mid);
				t_value += kernel[iKernel] * ((int)((in[i_in + jj*width])));
			}
			out[ii + jj*width] = t_value;
		}
	}

	return;
}
void filter1D_row(double *kernel, int k_size, char *in, double *out, int width, int height)
{


	int ii, jj, i_in, iKernel, k_mid = (k_size - 1) / 2;
	double t_value;

	for (ii = 0; ii < width*height; ii++)
		out[ii] = 0.0;

	for (jj = 0; jj < height; jj++)
	{
		for (ii = k_mid; ii < width - k_mid; ii++)
		{
			t_value = 0.0;
			for (iKernel = 0; iKernel < k_size; iKernel++)
			{
				i_in = ii + (iKernel - k_mid);
				t_value += kernel[iKernel] * ((int)((unsigned char)(in[i_in + jj*width])));
			}
			out[ii + jj*width] = t_value;
		}
	}

	return;
}
void filter1D_col(double *kernel, int k_size, double *in, double *out, int width, int height, double &i_max)
{


	int ii, jj, j_in, jKernel, k_mid = (k_size - 1) / 2;
	double t_value;

	for (ii = 0; ii < width*height; ii++)
		out[ii] = 0.0;

	i_max = 1.0;
	for (ii = k_mid; ii < width - k_mid; ii++)
	{
		for (jj = k_mid; jj < height - k_mid; jj++)
		{
			t_value = 0;
			for (jKernel = 0; jKernel < k_size; jKernel++)
			{
				j_in = jj + (jKernel - k_mid);
				t_value += kernel[jKernel] * in[ii + j_in*width];
			}
			out[ii + jj*width] = t_value;
			if (t_value > i_max)
				i_max = t_value;
		}
	}

	return;
}
void Gaussian_smooth(unsigned char* data, float* out_data, int height, int width, double max_i, double sigma)
{
	int ii, jj, size = MyFtoI(6.0*sigma + 1) / 2 * 2 + 1;
	double max_filter, sigma2 = 2.0*sigma*sigma, sqrt2Pi_sigma = sqrt(2.0*Pi)*sigma;
	double *kernel = new double[size];
	float *temp = new float[width*height];
	int kk = (size - 1) / 2;

	for (ii = -kk; ii <= kk; ii++)
		kernel[ii + kk] = exp(-(ii*ii) / sigma2) / sqrt2Pi_sigma;

	if (abs(max_i - 255.0) < 1.0)
	{
		for (ii = 0; ii<width*height; ii++)
			if (data[ii] > max_i)
				max_i = (unsigned char)data[ii];
	}

	filter1D_row(kernel, size, data, temp, width, height);
	filter1D_col(kernel, size, temp, out_data, width, height, max_filter);

	for (jj = kk; jj < height - kk; jj++)
		for (ii = kk; ii < width - kk; ii++)
			out_data[ii + jj*width] = out_data[ii + jj*width] * max_i / max_filter;

	for (ii = 0; ii < width; ii++)
	{
		for (jj = 0; jj < kk; jj++)
			out_data[ii + jj*width] = (float)((int)((unsigned char)(data[ii + jj*width])));
		for (jj = height - kk; jj < height; jj++)
			out_data[ii + jj*width] = (float)((int)((unsigned char)(data[ii + jj*width])));
	}

	for (jj = kk; jj < height - kk; jj++)
	{
		for (ii = 0; ii < kk; ii++)
			out_data[ii + jj*width] = (float)((int)((unsigned char)(data[ii + jj*width])));
		for (ii = width - kk; ii < width; ii++)
			out_data[ii + jj*width] = (float)((int)((unsigned char)(data[ii + jj*width])));
	}

	delete[]temp;
	delete[]kernel;
	return;
}
void Gaussian_smooth(char* data, float* out_data, int height, int width, double max_i, double sigma)
{
	int ii, jj, size = MyFtoI(6.0*sigma + 1) / 2 * 2 + 1;
	double max_filter, sigma2 = 2.0*sigma*sigma, sqrt2Pi_sigma = sqrt(2.0*Pi)*sigma;
	double *kernel = new double[size];
	float *temp = new float[width*height];
	int kk = (size - 1) / 2;

	for (ii = -kk; ii <= kk; ii++)
		kernel[ii + kk] = exp(-(ii*ii) / sigma2) / sqrt2Pi_sigma;

	if (abs(max_i - 255.0) < 1.0)
	{
		for (ii = 0; ii<width*height; ii++)
			if (data[ii] > max_i)
				max_i = (unsigned char)data[ii];
	}

	filter1D_row(kernel, size, data, temp, width, height);
	filter1D_col(kernel, size, temp, out_data, width, height, max_filter);

	for (jj = kk; jj < height - kk; jj++)
		for (ii = kk; ii < width - kk; ii++)
			out_data[ii + jj*width] = out_data[ii + jj*width] * max_i / max_filter;

	for (ii = 0; ii < width; ii++)
	{
		for (jj = 0; jj < kk; jj++)
			out_data[ii + jj*width] = (float)((int)((unsigned char)(data[ii + jj*width])));
		for (jj = height - kk; jj < height; jj++)
			out_data[ii + jj*width] = (float)((int)((unsigned char)(data[ii + jj*width])));
	}

	for (jj = kk; jj < height - kk; jj++)
	{
		for (ii = 0; ii < kk; ii++)
			out_data[ii + jj*width] = (float)((int)((unsigned char)(data[ii + jj*width])));
		for (ii = width - kk; ii < width; ii++)
			out_data[ii + jj*width] = (float)((int)((unsigned char)(data[ii + jj*width])));
	}

	delete[]temp;
	delete[]kernel;
	return;
}
void Gaussian_smooth(unsigned char* data, double* out_data, int height, int width, double max_i, double sigma)
{
	int ii, jj, size = MyFtoI(6.0*sigma + 1) / 2 * 2 + 1;
	double max_filter, sigma2 = 2.0*sigma*sigma, sqrt2Pi_sigma = sqrt(2.0*Pi)*sigma;
	double *kernel = new double[size];
	double *temp = new double[width*height];
	int kk = (size - 1) / 2;

	for (ii = -kk; ii <= kk; ii++)
		kernel[ii + kk] = exp(-(ii*ii) / sigma2) / sqrt2Pi_sigma;

	if (abs(max_i - 255.0) < 1.0)
	{
		for (ii = 0; ii<width*height; ii++)
			if (data[ii] > max_i)
				max_i = (unsigned char)data[ii];
	}

	filter1D_row(kernel, size, data, temp, width, height);
	filter1D_col(kernel, size, temp, out_data, width, height, max_filter);

	for (jj = kk; jj < height - kk; jj++)
		for (ii = kk; ii < width - kk; ii++)
			out_data[ii + jj*width] = out_data[ii + jj*width] * max_i / max_filter;

	for (ii = 0; ii < width; ii++)
	{
		for (jj = 0; jj < kk; jj++)
			out_data[ii + jj*width] = (double)((int)((unsigned char)(data[ii + jj*width])));
		for (jj = height - kk; jj < height; jj++)
			out_data[ii + jj*width] = (double)((int)((unsigned char)(data[ii + jj*width])));
	}

	for (jj = kk; jj < height - kk; jj++)
	{
		for (ii = 0; ii < kk; ii++)
			out_data[ii + jj*width] = (double)((int)((unsigned char)(data[ii + jj*width])));
		for (ii = width - kk; ii < width; ii++)
			out_data[ii + jj*width] = (double)((int)((unsigned char)(data[ii + jj*width])));
	}

	delete[]temp;
	delete[]kernel;
	return;
}
void Gaussian_smooth(char* data, double* out_data, int height, int width, double max_i, double sigma)
{
	int ii, jj, size = MyFtoI(6.0*sigma + 1) / 2 * 2 + 1;
	double max_filter, sigma2 = 2.0*sigma*sigma, sqrt2Pi_sigma = sqrt(2.0*Pi)*sigma;
	double *kernel = new double[size];
	double *temp = new double[width*height];
	int kk = (size - 1) / 2;

	for (ii = -kk; ii <= kk; ii++)
		kernel[ii + kk] = exp(-(ii*ii) / sigma2) / sqrt2Pi_sigma;

	if (abs(max_i - 255.0) < 1.0)
	{
		for (ii = 0; ii<width*height; ii++)
			if (data[ii] > max_i)
				max_i = (unsigned char)data[ii];
	}

	filter1D_row(kernel, size, data, temp, width, height);
	filter1D_col(kernel, size, temp, out_data, width, height, max_filter);

	for (jj = kk; jj < height - kk; jj++)
		for (ii = kk; ii < width - kk; ii++)
			out_data[ii + jj*width] = out_data[ii + jj*width] * max_i / max_filter;

	for (ii = 0; ii < width; ii++)
	{
		for (jj = 0; jj < kk; jj++)
			out_data[ii + jj*width] = (double)((int)((unsigned char)(data[ii + jj*width])));
		for (jj = height - kk; jj < height; jj++)
			out_data[ii + jj*width] = (double)((int)((unsigned char)(data[ii + jj*width])));
	}

	for (jj = kk; jj < height - kk; jj++)
	{
		for (ii = 0; ii < kk; ii++)
			out_data[ii + jj*width] = (double)((int)((unsigned char)(data[ii + jj*width])));
		for (ii = width - kk; ii < width; ii++)
			out_data[ii + jj*width] = (double)((int)((unsigned char)(data[ii + jj*width])));
	}

	delete[]temp;
	delete[]kernel;
	return;
}
void Gaussian_smooth(double* data, double* out_data, int height, int width, double max_i, double sigma)
{
	int ii, jj, size = MyFtoI(6.0*sigma + 1) / 2 * 2 + 1;
	double max_filter, sigma2 = 2.0*sigma*sigma, sqrt2Pi_sigma = sqrt(2.0*Pi)*sigma;
	double *kernel = new double[size];
	double *temp = new double[width*height];
	int kk = (size - 1) / 2;

	for (ii = -kk; ii <= kk; ii++)
		kernel[ii + kk] = exp(-(ii*ii) / sigma2) / sqrt2Pi_sigma;

	if (abs(max_i - 255.0) < 1.0)
	{
		for (ii = 0; ii<width*height; ii++)
			if (data[ii] > max_i)
				max_i = data[ii];
	}

	filter1D_row_Double(kernel, size, data, temp, width, height);
	filter1D_col(kernel, size, temp, out_data, width, height, max_filter);

	for (jj = kk; jj < height - kk; jj++)
		for (ii = kk; ii < width - kk; ii++)
			out_data[ii + jj*width] = out_data[ii + jj*width] * max_i / max_filter;

	for (ii = 0; ii < width; ii++)
	{
		for (jj = 0; jj < kk; jj++)
			out_data[ii + jj*width] = data[ii + jj*width];
		for (jj = height - kk; jj < height; jj++)
			out_data[ii + jj*width] = data[ii + jj*width];
	}

	for (jj = kk; jj < height - kk; jj++)
	{
		for (ii = 0; ii < kk; ii++)
			out_data[ii + jj*width] = data[ii + jj*width];
		for (ii = width - kk; ii < width; ii++)
			out_data[ii + jj*width] = data[ii + jj*width];
	}

	delete[]temp;
	delete[]kernel;
	return;
}
double InitialCausalCoefficient(double *sample, int length, double pole, double tolerance)
{
	double zn, iz, z2n;
	double FirstCausalCoef;
	int n, horizon;
	horizon = (int)(ceil(log(tolerance) / log(fabs(pole))) + 0.01);
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
double InitialAnticausalCoefficient(double *CausalCoef, int length, double pole)
{
	return((pole / (pole * pole - 1.0)) * (pole * CausalCoef[length - 2] + CausalCoef[length - 1]));
}
// prefilter for 4-tap, 6-tap, 8-tap, optimized 4-tap, and optimized 6-tap
void Prefilter_1D(double *coefficient, int length, double *pole, double tolerance, int nPoles)
{
	int i, n, k;
	double Lambda;
	Lambda = 1;
	if (length == 1)
		return;
	/* compute the overall gain */
	for (k = 0; k < nPoles; k++)
		Lambda = Lambda * (1.0 - pole[k]) * (1.0 - 1.0 / pole[k]);

	// Applying the gain to original image
	for (i = 0; i < length; i++)
		*(coefficient + i) = (*(coefficient + i)) * Lambda;

	for (k = 0; k < nPoles; k++)
	{
		// Compute the first causal coefficient
		*(coefficient) = InitialCausalCoefficient(coefficient, length, pole[k], tolerance);

		// Causal prefilter
		for (n = 1; n < length; n++)
			coefficient[n] += pole[k] * coefficient[n - 1];

		//Compute the first anticausal coefficient
		*(coefficient + length - 1) = InitialAnticausalCoefficient(coefficient, length, pole[k]);

		//Anticausal prefilter
		for (n = length - 2; n >= 0; n--)
			coefficient[n] = pole[k] * (coefficient[n + 1] - coefficient[n]);
	}
}
// Prefilter for modified 4-tap
void Prefilter_1Dm(double *coefficient, int length, double *pole, double tolerance, double gamma)
{
	int i, n, k;
	double Lambda;
	Lambda = 6.0 / (6.0 * gamma + 1.0);
	if (length == 1)
		return;

	// Applying the gain to original image
	for (i = 0; i < length; i++)
		*(coefficient + i) = (*(coefficient + i)) * Lambda;

	for (k = 0; k < 1; k++)
	{
		// Compute the first causal coefficient
		*(coefficient) = InitialCausalCoefficient(coefficient, length, pole[k], tolerance);

		// Causal prefilter
		for (n = 1; n < length; n++)
			coefficient[n] += pole[k] * coefficient[n - 1];

		//Compute the first anticausal coefficient
		*(coefficient + length - 1) = InitialAnticausalCoefficient(coefficient, length, pole[k]);

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
		a = (4.0 - 12.0 * gamma) / (6.0 * gamma + 1.0);
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
		for (j = 0; j < width; j++)
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
	delete[]LineWidth;

	//Perform the 1D prefiltering along the columns
	double *LineHeight = new double[height];
	for (i = 0; i < width; i++)
	{
		//Prefiltering each comlumn
		for (j = 0; j < height; j++)
		{
			*(LineHeight + j) = (*(Para + j * width + i));
		}
		if (Interpolation_Algorithm == 3)
			Prefilter_1Dm(LineHeight, height, pole, tolerance, gamma);
		else
			Prefilter_1D(LineHeight, height, pole, tolerance, nPoles);

		//Put the prefilterd coefficients into the Para array
		for (j = 0; j < height; j++)
		{
			*(Para + j * width + i) = (*(LineHeight + j));
		}
	}
	delete[]LineHeight;
	return;
}
void Generate_Para_Spline(char *Image, double *Para, int width, int height, int Interpolation_Algorithm)
{
	int i, length = width*height;
	double *Image_2 = new double[length];
	for (i = 0; i < length; i++)
		*(Image_2 + i) = (double)((int)((unsigned char)(*(Image + i))));

	Generate_Para_Spline(Image_2, Para, width, height, Interpolation_Algorithm);

	delete[]Image_2;
	return;
}
void Generate_Para_Spline(unsigned char *Image, double *Para, int width, int height, int Interpolation_Algorithm)
{
	int i, length = width*height;
	double *Image_2 = new double[length];
	for (i = 0; i < length; i++)
		*(Image_2 + i) = (double)((int)(*(Image + i)));

	Generate_Para_Spline(Image_2, Para, width, height, Interpolation_Algorithm);

	delete[]Image_2;
	return;
}
void Generate_Para_Spline(int *Image, double *Para, int width, int height, int Interpolation_Algorithm)
{
	int i, length = width*height;
	double *Image_2 = new double[length];
	for (i = 0; i < length; i++)
		*(Image_2 + i) = (double)(*(Image + i));

	Generate_Para_Spline(Image_2, Para, width, height, Interpolation_Algorithm);

	delete[]Image_2;
	return;
}
void Get_Value_Spline(double *Para, int width, int height, double X, double Y, double *S, int S_Flag, int Interpolation_Algorithm)
{
	int i, j, width2, height2, xIndex[6], yIndex[6];
	double Para_Value, xWeight[6], yWeight[6], xWeightGradient[6], yWeightGradient[6], w, w2, w3, w4, t, t0, t1, gamma;
	double oneSix = 1.0 / 6.0;

	width2 = 2 * width - 2;
	height2 = 2 * height - 2;

	if (Interpolation_Algorithm == 6)
	{
		xIndex[0] = int(X) - 2;
		yIndex[0] = int(Y) - 2;
		for (i = 1; i < 6; i++)
		{
			xIndex[i] = xIndex[i - 1] + 1;
			yIndex[i] = yIndex[i - 1] + 1;
		}
	}
	else if ((Interpolation_Algorithm == 2) || (Interpolation_Algorithm == 5))
	{
		xIndex[0] = int(X + 0.5) - 2;
		yIndex[0] = int(Y + 0.5) - 2;
		for (i = 1; i < 5; i++)
		{
			xIndex[i] = xIndex[i - 1] + 1;
			yIndex[i] = yIndex[i - 1] + 1;
		}
	}
	else
	{
		xIndex[0] = int(X) - 1;
		yIndex[0] = int(Y) - 1;
		for (i = 1; i < 4; i++)
		{
			xIndex[i] = xIndex[i - 1] + 1;
			yIndex[i] = yIndex[i - 1] + 1;
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

		if (S_Flag > -1)
		{
			xWeightGradient[3] = w2 / 2.0;
			xWeightGradient[0] = w - 0.5 - xWeightGradient[3];
			xWeightGradient[2] = 1.0 + xWeightGradient[0] - 2.0 * xWeightGradient[3];
			xWeightGradient[1] = -xWeightGradient[0] - xWeightGradient[2] - xWeightGradient[3];
		}

		/* y */
		w = Y - (double)yIndex[1];
		w2 = w*w; w3 = w2*w;
		yWeight[3] = oneSix * w3;
		yWeight[0] = oneSix + 0.5 * (w2 - w) - yWeight[3];
		yWeight[2] = w + yWeight[0] - 2.0 * yWeight[3];
		yWeight[1] = 1.0 - yWeight[0] - yWeight[2] - yWeight[3];

		if (S_Flag > -1)
		{
			yWeightGradient[3] = w2 / 2.0;
			yWeightGradient[0] = w - 0.5 - yWeightGradient[3];
			yWeightGradient[2] = 1.0 + yWeightGradient[0] - 2.0 * yWeightGradient[3];
			yWeightGradient[1] = -yWeightGradient[0] - yWeightGradient[2] - yWeightGradient[3];
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
		xWeightGradient[1] = w * w / 2 - 11.0 / 24.0 + w / 2.0 - 2.0 * w * w * w / 3.0;
		xWeightGradient[3] = -w * w / 2 + 11.0 / 24.0 + w / 2.0 - 2.0 * w * w * w / 3.0;
		xWeightGradient[4] = xWeightGradient[0] + w * w / 2.0 + 1.0 / 24.0;
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
		yWeightGradient[1] = w * w / 2 - 11.0 / 24.0 + w / 2.0 - 2.0 * w * w * w / 3.0;
		yWeightGradient[3] = -w * w / 2 + 11.0 / 24.0 + w / 2.0 - 2.0 * w * w * w / 3.0;
		yWeightGradient[4] = yWeightGradient[0] + w * w / 2.0 + 1.0 / 24.0;
		yWeightGradient[2] = -yWeightGradient[0] - yWeightGradient[1] - yWeightGradient[3] - yWeightGradient[4];
	}
	else if (Interpolation_Algorithm == 3)
	{
		gamma = 0.0409;
		w = X - (double)xIndex[1];
		xWeight[0] = -w * w * w / 6.0 + w * w / 2.0 - (gamma + 0.5) * w + 1.0 / 6.0 + gamma;
		xWeight[1] = w * w * w / 2.0 - w * w + 3 * gamma * w + 2.0 / 3.0 - 2.0 * gamma;
		xWeight[2] = -w * w * w / 2.0 + w * w / 2.0 + (1.0 / 2.0 - 3.0 * gamma) * w + gamma + 1.0 / 6.0;
		xWeight[3] = w * w * w / 6.0 + gamma * w;

		xWeightGradient[0] = -w * w / 2.0 + w - gamma - 0.5;
		xWeightGradient[1] = 3.0 * w * w / 2.0 - 2.0 * w + 3.0 * gamma;
		xWeightGradient[2] = -3.0 * w * w / 2.0 + w + 1.0 / 2.0 - 3.0 * gamma;
		xWeightGradient[3] = w * w / 2.0 + gamma;

		/* y */
		w = Y - (double)yIndex[1];
		yWeight[0] = -w * w * w / 6.0 + w * w / 2.0 - (gamma + 0.5) * w + 1.0 / 6.0 + gamma;
		yWeight[1] = w * w * w / 2.0 - w * w + 3 * gamma * w + 2.0 / 3.0 - 2.0 * gamma;
		yWeight[2] = -w * w * w / 2.0 + w * w / 2.0 + (1.0 / 2.0 - 3.0 * gamma) * w + gamma + 1.0 / 6.0;
		yWeight[3] = w * w * w / 6.0 + gamma * w;

		yWeightGradient[0] = -w * w / 2.0 + w - gamma - 0.5;
		yWeightGradient[1] = 3.0 * w * w / 2.0 - 2.0 * w + 3.0 * gamma;
		yWeightGradient[2] = -3.0 * w * w / 2.0 + w + 1.0 / 2.0 - 3.0 * gamma;
		yWeightGradient[3] = w * w / 2.0 + gamma;
	}
	else if (Interpolation_Algorithm == 4)
	{
		w = X - (double)xIndex[1];
		xWeight[0] = -w * w * w / 6.0 + w * w / 2.0 - 11.0 * w / 21.0 + 4.0 / 21.0;
		xWeight[1] = w * w * w / 2.0 - w * w + 3.0 * w / 42.0 + 13.0 / 21.0;
		xWeight[2] = -w * w * w / 2.0 + w * w / 2.0 + 3.0 * w / 7.0 + 4.0 / 21.0;
		xWeight[3] = w * w * w / 6.0 + w / 42.0;

		xWeightGradient[0] = -w * w / 2.0 + w - 11.0 / 21.0;
		xWeightGradient[1] = 3.0 * w * w / 2.0 - 2.0 * w + 3.0 / 42.0;
		xWeightGradient[2] = -3.0 * w * w / 2.0 + w + 3.0 / 7.0;
		xWeightGradient[3] = w * w / 2.0 + 1.0 / 42.0;

		/* y */
		w = Y - (double)yIndex[1];
		yWeight[0] = -w * w * w / 6.0 + w * w / 2.0 - 11.0 * w / 21.0 + 4.0 / 21.0;
		yWeight[1] = w * w * w / 2.0 - w * w + 3.0 * w / 42.0 + 13.0 / 21.0;
		yWeight[2] = -w * w * w / 2.0 + w * w / 2.0 + 3.0 * w / 7.0 + 4.0 / 21.0;
		yWeight[3] = w * w * w / 6.0 + w / 42.0;

		yWeightGradient[0] = -w * w / 2.0 + w - 11.0 / 21.0;
		yWeightGradient[1] = 3.0 * w * w / 2.0 - 2.0 * w + 3.0 / 42.0;
		yWeightGradient[2] = -3.0 * w * w / 2.0 + w + 3.0 / 7.0;
		yWeightGradient[3] = w * w / 2.0 + 1.0 / 42.0;
	}
	else if (Interpolation_Algorithm == 5)
	{
		w = X - (double)xIndex[2];
		w2 = w*w; w3 = w2*w; w4 = w2*w2;
		double coeff1 = 743.0 / 120960.0, coeff2 = 6397.0 / 30240.0, coeff3 = 5.0 / 144.0, coeff4 = 31.0 / 72.0, coeff5 = 11383.0 / 20160.0;
		double coeff6 = 11.0 / 144.0, coeff7 = 5.0 / 144.0, coeff8 = 7.0 / 36.0, coeff9 = 31.0 / 72.0, coeff10 = 13.0 / 24.0, coeff11 = 11.0 / 72.0, coeff12 = 7.0 / 18.0;
		xWeight[0] = w4 / 24.0 - w3 / 12.0 + w2 * coeff6 - w *coeff7 + coeff1;
		xWeight[1] = -w4 / 6.0 + w3 / 6.0 + w2*coeff8 - w*coeff9 + coeff2;
		xWeight[2] = w4 / 4.0 - w2 *coeff10 + coeff5;
		xWeight[3] = -w4 / 6.0 - w3 / 6.0 + w2*coeff8 + w*coeff9 + coeff2;
		xWeight[4] = w4 / 24.0 + w3 / 12.0 + w2 * coeff6 + w *coeff7 + coeff1;
		//xWeight[2] = 1.0 - xWeight[0] - xWeight[1] - xWeight[3] - xWeight[4]; 

		if (S_Flag > -1)
		{
			xWeightGradient[0] = w3 / 6.0 - w2 / 4.0 + w *coeff11 - coeff3;
			xWeightGradient[1] = -2.0 * w3 / 3.0 + w2 / 2.0 + w*coeff12 - coeff4;
			xWeightGradient[2] = w3 - 13.0 * w / 12.0;
			xWeightGradient[3] = -2.0 * w3 / 3.0 - w2 / 2.0 + w*coeff12 + coeff4;
			xWeightGradient[4] = w3 / 6.0 + w2 / 4.0 + w *coeff11 + coeff3;
		}

		/* y */
		w = Y - (double)yIndex[2];
		w2 = w*w; w3 = w2*w; w4 = w2*w2;
		yWeight[0] = w4 / 24.0 - w3 / 12.0 + w2 * coeff6 - w *coeff7 + coeff1;
		yWeight[1] = -w4 / 6.0 + w3 / 6.0 + w2*coeff8 - w*coeff9 + coeff2;
		yWeight[2] = w4 / 4.0 - w2 *coeff10 + coeff5;
		yWeight[3] = -w4 / 6.0 - w3 / 6.0 + w2*coeff8 + w*coeff9 + coeff2;
		yWeight[4] = w4 / 24.0 + w3 / 12.0 + w2 * coeff6 + w *coeff7 + coeff1;
		//yWeight[2] = 1.0 - xWeight[0] - xWeight[1] - xWeight[3] - xWeight[4]; 

		if (S_Flag > -1)
		{
			yWeightGradient[0] = w3 / 6.0 - w2 / 4.0 + w *coeff11 - coeff3;
			yWeightGradient[1] = -2.0 * w3 / 3.0 + w2 / 2.0 + w*coeff12 - coeff4;
			yWeightGradient[2] = w3 - 13.0 * w / 12.0;
			yWeightGradient[3] = -2.0 * w3 / 3.0 - w2 / 2.0 + w*coeff12 + coeff4;
			yWeightGradient[4] = w3 / 6.0 + w2 / 4.0 + w *coeff11 + coeff3;
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
		t1 = -(5.0 * w * w * w * w - 10.0 * w * w * w - 3.0 * w * w + 8.0 * w + 5.0 / 2.0) / 12.0;
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
		t1 = -(5.0 * w * w * w * w - 10.0 * w * w * w - 3.0 * w * w + 8.0 * w + 5.0 / 2.0) / 12.0;
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
	if (S_Flag > -1)
		S[1] = 0, S[2] = 0;

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

		for (i = 0; i < 6; i++)
		{
			for (j = 0; j < 6; j++)
			{
				Para_Value = (*(Para + width * yIndex[i] + xIndex[j]));
				S[0] = S[0] + Para_Value * xWeight[j] * yWeight[i];
				if (S_Flag > -1)
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
		for (i = 0; i < 5; i++)
		{
			for (j = 0; j < 5; j++)
			{
				Para_Value = (*(Para + width * yIndex[i] + xIndex[j]));
				S[0] = S[0] + Para_Value * xWeight[j] * yWeight[i];
				if (S_Flag > -1)
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
		for (i = 0; i < 4; i++)
		{
			for (j = 0; j < 4; j++)
			{
				Para_Value = (*(Para + width * yIndex[i] + xIndex[j]));
				S[0] = S[0] + Para_Value * xWeight[j] * yWeight[i];
				if (S_Flag > -1)
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
void Generate_Para_Spline(char *Image, float *Para, int width, int height, int Interpolation_Algorithm)
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
		a = (4.0 - 12.0 * gamma) / (6.0 * gamma + 1.0);
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
		for (j = 0; j < width; j++)
		{
			*(LineWidth + j) = (float)(int)(unsigned char)Image[i * width + j];
		}
		if (Interpolation_Algorithm == 3)
			Prefilter_1Dm(LineWidth, width, pole, tolerance, gamma);
		else
			Prefilter_1D(LineWidth, width, pole, tolerance, nPoles);

		// Put the prefiltered coeffiecients into Para array
		for (j = 0; j < width; j++)
			Para[i * width + j] = (float)LineWidth[j];
	}
	delete[]LineWidth;

	//Perform the 1D prefiltering along the columns
	double *LineHeight = new double[height];
	for (i = 0; i < width; i++)
	{
		//Prefiltering each comlumn
		for (j = 0; j < height; j++)
		{
			*(LineHeight + j) = (*(Para + j * width + i));
		}
		if (Interpolation_Algorithm == 3)
			Prefilter_1Dm(LineHeight, height, pole, tolerance, gamma);
		else
			Prefilter_1D(LineHeight, height, pole, tolerance, nPoles);

		//Put the prefilterd coefficients into the Para array
		for (j = 0; j < height; j++)
			Para[j * width + i] = (float)LineHeight[j];
	}
	delete[]LineHeight;

	return;
}
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
		a = (4.0 - 12.0 * gamma) / (6.0 * gamma + 1.0);
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
		for (j = 0; j < width; j++)
		{
			*(LineWidth + j) = *(Image + i * width + j);
		}
		if (Interpolation_Algorithm == 3)
			Prefilter_1Dm(LineWidth, width, pole, tolerance, gamma);
		else
			Prefilter_1D(LineWidth, width, pole, tolerance, nPoles);

		// Put the prefiltered coeffiecients into Para array
		for (j = 0; j < width; j++)
			Para[i * width + j] = (float)LineWidth[j];
	}
	delete[]LineWidth;

	//Perform the 1D prefiltering along the columns
	double *LineHeight = new double[height];
	for (i = 0; i < width; i++)
	{
		//Prefiltering each comlumn
		for (j = 0; j < height; j++)
		{
			*(LineHeight + j) = (*(Para + j * width + i));
		}
		if (Interpolation_Algorithm == 3)
			Prefilter_1Dm(LineHeight, height, pole, tolerance, gamma);
		else
			Prefilter_1D(LineHeight, height, pole, tolerance, nPoles);

		//Put the prefilterd coefficients into the Para array
		for (j = 0; j < height; j++)
			Para[j * width + i] = (float)LineHeight[j];
	}
	delete[]LineHeight;

	return;
}
void Get_Value_Spline(float *Para, int width, int height, double X, double Y, double *S, int S_Flag, int Interpolation_Algorithm)
{
	int i, j, width2, height2, xIndex[6], yIndex[6];
	double Para_Value, xWeight[6], yWeight[6], xWeightGradient[6], yWeightGradient[6], w, w2, w3, w4, t, t0, t1, gamma;
	double oneSix = 1.0 / 6.0;

	width2 = 2 * width - 2;
	height2 = 2 * height - 2;

	if (Interpolation_Algorithm == 6)
	{
		xIndex[0] = int(X) - 2;
		yIndex[0] = int(Y) - 2;
		for (i = 1; i < 6; i++)
		{
			xIndex[i] = xIndex[i - 1] + 1;
			yIndex[i] = yIndex[i - 1] + 1;
		}
	}
	else if ((Interpolation_Algorithm == 2) || (Interpolation_Algorithm == 5))
	{
		xIndex[0] = int(X + 0.5) - 2;
		yIndex[0] = int(Y + 0.5) - 2;
		for (i = 1; i < 5; i++)
		{
			xIndex[i] = xIndex[i - 1] + 1;
			yIndex[i] = yIndex[i - 1] + 1;
		}
	}
	else
	{
		xIndex[0] = int(X) - 1;
		yIndex[0] = int(Y) - 1;
		for (i = 1; i < 4; i++)
		{
			xIndex[i] = xIndex[i - 1] + 1;
			yIndex[i] = yIndex[i - 1] + 1;
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
		xWeightGradient[1] = -xWeightGradient[0] - xWeightGradient[2] - xWeightGradient[3];

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
		yWeightGradient[1] = -yWeightGradient[0] - yWeightGradient[2] - yWeightGradient[3];
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
		xWeightGradient[1] = w * w / 2 - 11.0 / 24.0 + w / 2.0 - 2.0 * w * w * w / 3.0;
		xWeightGradient[3] = -w * w / 2 + 11.0 / 24.0 + w / 2.0 - 2.0 * w * w * w / 3.0;
		xWeightGradient[4] = xWeightGradient[0] + w * w / 2.0 + 1.0 / 24.0;
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
		yWeightGradient[1] = w * w / 2 - 11.0 / 24.0 + w / 2.0 - 2.0 * w * w * w / 3.0;
		yWeightGradient[3] = -w * w / 2 + 11.0 / 24.0 + w / 2.0 - 2.0 * w * w * w / 3.0;
		yWeightGradient[4] = yWeightGradient[0] + w * w / 2.0 + 1.0 / 24.0;
		yWeightGradient[2] = -yWeightGradient[0] - yWeightGradient[1] - yWeightGradient[3] - yWeightGradient[4];
	}
	else if (Interpolation_Algorithm == 3)
	{
		gamma = 0.0409;
		w = X - (double)xIndex[1];
		xWeight[0] = -w * w * w / 6.0 + w * w / 2.0 - (gamma + 0.5) * w + 1.0 / 6.0 + gamma;
		xWeight[1] = w * w * w / 2.0 - w * w + 3 * gamma * w + 2.0 / 3.0 - 2.0 * gamma;
		xWeight[2] = -w * w * w / 2.0 + w * w / 2.0 + (1.0 / 2.0 - 3.0 * gamma) * w + gamma + 1.0 / 6.0;
		xWeight[3] = w * w * w / 6.0 + gamma * w;

		xWeightGradient[0] = -w * w / 2.0 + w - gamma - 0.5;
		xWeightGradient[1] = 3.0 * w * w / 2.0 - 2.0 * w + 3.0 * gamma;
		xWeightGradient[2] = -3.0 * w * w / 2.0 + w + 1.0 / 2.0 - 3.0 * gamma;
		xWeightGradient[3] = w * w / 2.0 + gamma;

		/* y */
		w = Y - (double)yIndex[1];
		yWeight[0] = -w * w * w / 6.0 + w * w / 2.0 - (gamma + 0.5) * w + 1.0 / 6.0 + gamma;
		yWeight[1] = w * w * w / 2.0 - w * w + 3 * gamma * w + 2.0 / 3.0 - 2.0 * gamma;
		yWeight[2] = -w * w * w / 2.0 + w * w / 2.0 + (1.0 / 2.0 - 3.0 * gamma) * w + gamma + 1.0 / 6.0;
		yWeight[3] = w * w * w / 6.0 + gamma * w;

		yWeightGradient[0] = -w * w / 2.0 + w - gamma - 0.5;
		yWeightGradient[1] = 3.0 * w * w / 2.0 - 2.0 * w + 3.0 * gamma;
		yWeightGradient[2] = -3.0 * w * w / 2.0 + w + 1.0 / 2.0 - 3.0 * gamma;
		yWeightGradient[3] = w * w / 2.0 + gamma;
	}
	else if (Interpolation_Algorithm == 4)
	{
		w = X - (double)xIndex[1];
		xWeight[0] = -w * w * w / 6.0 + w * w / 2.0 - 11.0 * w / 21.0 + 4.0 / 21.0;
		xWeight[1] = w * w * w / 2.0 - w * w + 3.0 * w / 42.0 + 13.0 / 21.0;
		xWeight[2] = -w * w * w / 2.0 + w * w / 2.0 + 3.0 * w / 7.0 + 4.0 / 21.0;
		xWeight[3] = w * w * w / 6.0 + w / 42.0;

		xWeightGradient[0] = -w * w / 2.0 + w - 11.0 / 21.0;
		xWeightGradient[1] = 3.0 * w * w / 2.0 - 2.0 * w + 3.0 / 42.0;
		xWeightGradient[2] = -3.0 * w * w / 2.0 + w + 3.0 / 7.0;
		xWeightGradient[3] = w * w / 2.0 + 1.0 / 42.0;

		/* y */
		w = Y - (double)yIndex[1];
		yWeight[0] = -w * w * w / 6.0 + w * w / 2.0 - 11.0 * w / 21.0 + 4.0 / 21.0;
		yWeight[1] = w * w * w / 2.0 - w * w + 3.0 * w / 42.0 + 13.0 / 21.0;
		yWeight[2] = -w * w * w / 2.0 + w * w / 2.0 + 3.0 * w / 7.0 + 4.0 / 21.0;
		yWeight[3] = w * w * w / 6.0 + w / 42.0;

		yWeightGradient[0] = -w * w / 2.0 + w - 11.0 / 21.0;
		yWeightGradient[1] = 3.0 * w * w / 2.0 - 2.0 * w + 3.0 / 42.0;
		yWeightGradient[2] = -3.0 * w * w / 2.0 + w + 3.0 / 7.0;
		yWeightGradient[3] = w * w / 2.0 + 1.0 / 42.0;
	}
	else if (Interpolation_Algorithm == 5)
	{
		w = X - (double)xIndex[2];
		w2 = w*w; w3 = w2*w; w4 = w2*w2;
		double coeff1 = 743.0 / 120960.0, coeff2 = 6397.0 / 30240.0, coeff3 = 5.0 / 144.0, coeff4 = 31.0 / 72.0, coeff5 = 11383.0 / 20160.0;
		double coeff6 = 11.0 / 144.0, coeff7 = 5.0 / 144.0, coeff8 = 7.0 / 36.0, coeff9 = 31.0 / 72.0, coeff10 = 13.0 / 24.0, coeff11 = 11.0 / 72.0, coeff12 = 7.0 / 18.0;
		xWeight[0] = w4 / 24.0 - w3 / 12.0 + w2 * coeff6 - w *coeff7 + coeff1;
		xWeight[1] = -w4 / 6.0 + w3 / 6.0 + w2*coeff8 - w*coeff9 + coeff2;
		xWeight[2] = w4 / 4.0 - w2 *coeff10 + coeff5;
		xWeight[3] = -w4 / 6.0 - w3 / 6.0 + w2*coeff8 + w*coeff9 + coeff2;
		xWeight[4] = w4 / 24.0 + w3 / 12.0 + w2 * coeff6 + w *coeff7 + coeff1;
		//xWeight[2] = 1.0 - xWeight[0] - xWeight[1] - xWeight[3] - xWeight[4]; 

		xWeightGradient[0] = w3 / 6.0 - w2 / 4.0 + w *coeff11 - coeff3;
		xWeightGradient[1] = -2.0 * w3 / 3.0 + w2 / 2.0 + w*coeff12 - coeff4;
		xWeightGradient[2] = w3 - 13.0 * w / 12.0;
		xWeightGradient[3] = -2.0 * w3 / 3.0 - w2 / 2.0 + w*coeff12 + coeff4;
		xWeightGradient[4] = w3 / 6.0 + w2 / 4.0 + w *coeff11 + coeff3;

		/* y */
		w = Y - (double)yIndex[2];
		w2 = w*w; w3 = w2*w; w4 = w2*w2;
		yWeight[0] = w4 / 24.0 - w3 / 12.0 + w2 * coeff6 - w *coeff7 + coeff1;
		yWeight[1] = -w4 / 6.0 + w3 / 6.0 + w2*coeff8 - w*coeff9 + coeff2;
		yWeight[2] = w4 / 4.0 - w2 *coeff10 + coeff5;
		yWeight[3] = -w4 / 6.0 - w3 / 6.0 + w2*coeff8 + w*coeff9 + coeff2;
		yWeight[4] = w4 / 24.0 + w3 / 12.0 + w2 * coeff6 + w *coeff7 + coeff1;
		//yWeight[2] = 1.0 - xWeight[0] - xWeight[1] - xWeight[3] - xWeight[4]; 

		yWeightGradient[0] = w3 / 6.0 - w2 / 4.0 + w *coeff11 - coeff3;
		yWeightGradient[1] = -2.0 * w3 / 3.0 + w2 / 2.0 + w*coeff12 - coeff4;
		yWeightGradient[2] = w3 - 13.0 * w / 12.0;
		yWeightGradient[3] = -2.0 * w3 / 3.0 - w2 / 2.0 + w*coeff12 + coeff4;
		yWeightGradient[4] = w3 / 6.0 + w2 / 4.0 + w *coeff11 + coeff3;
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
		t1 = -(5.0 * w * w * w * w - 10.0 * w * w * w - 3.0 * w * w + 8.0 * w + 5.0 / 2.0) / 12.0;
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
		t1 = -(5.0 * w * w * w * w - 10.0 * w * w * w - 3.0 * w * w + 8.0 * w + 5.0 / 2.0) / 12.0;
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

		for (i = 0; i < 6; i++)
		{
			for (j = 0; j < 6; j++)
			{
				Para_Value = (*(Para + width * yIndex[i] + xIndex[j]));
				S[0] = S[0] + Para_Value * xWeight[j] * yWeight[i];
				if (S_Flag > -1)
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
		for (i = 0; i < 5; i++)
		{
			for (j = 0; j < 5; j++)
			{
				Para_Value = (*(Para + width * yIndex[i] + xIndex[j]));
				S[0] = S[0] + Para_Value * xWeight[j] * yWeight[i];
				if (S_Flag > -1)
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
		for (i = 0; i < 4; i++)
		{
			for (j = 0; j < 4; j++)
			{
				Para_Value = (*(Para + width * yIndex[i] + xIndex[j]));
				S[0] = S[0] + Para_Value * xWeight[j] * yWeight[i];
				if (S_Flag > -1)
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

	for (j = 0; j < height; j++)
	{
		for (i = 0; i < width; i++)
		{
			*(Para + (j + 1)*width_ex + i + 1) = *(Image + j*width + i);
		}
	}

	// bottom
	for (i = 0; i < width; i++)
		*(Para + i + 1) = *(Image + i) * 2 - *(Image + width + i);

	// top
	for (i = 0; i < width; i++)
		*(Para + (height_ex - 1)*width_ex + i + 1) = *(Image + (height - 1)*width + i) * 2 - *(Image + (height - 2)*width + i);

	// left
	for (j = 0; j < height; j++)
		*(Para + (j + 1)*width_ex) = *(Image + j*width) * 2 - *(Image + j*width + 1);

	// right
	for (j = 0; j < height; j++)
		*(Para + (j + 1)*width_ex + width_ex - 1) = *(Image + j*width + width - 1) * 2 - *(Image + j*width + width - 2);

	//l-b corner
	*(Para) = *(Image)* 4 - *(Image + 1) * 2 - *(Image + width) * 2 + *(Image + width + 1);

	//r-b corner
	*(Para + width_ex - 1) = *(Image + width - 1) * 4 - *(Image + width - 2) * 2 - *(Image + width + width - 1) * 2 + *(Image + width + width - 2);

	//l-t corner
	*(Para + (height_ex - 1)*width_ex) = *(Image + (height - 1)*width) * 4 - *(Image + (height - 1)*width + 1) * 2 - *(Image + (height - 2)*width) * 2 + *(Image + (height - 2)*width + 1);

	//r-t corner
	*(Para + (height_ex - 1)*width_ex + width_ex - 1) = *(Image + (height - 1)*width + width - 1) * 4 - *(Image + (height - 1)*width + width - 2) * 2 - *(Image + (height - 2)*width + width - 1) * 2 + *(Image + (height - 2)*width + width - 2);

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
		xIndex[i] = xIndex[i - 1] + 1;
		yIndex[i] = yIndex[i - 1] + 1;
	}
	//Another method********************
	w = X - (double)xIndex[0];
	w2 = w*w; w3 = w2*w;
	xWeight[0] = a * (w3 - 2.0 * w2 + w);
	xWeight[1] = (a + 2.0) * w3 - (a + 3.0) * w2 + 1;
	xWeight[2] = -(a + 2.0) * w3 + (2.0 * a + 3.0) * w2 - a * w;
	xWeight[3] = a * (-w3 + w * w);

	if (S_Flag > -1)
	{
		xWeightGradient[0] = a * (3.0 * w2 - 4.0 * w + 1);
		xWeightGradient[1] = 3.0 * (a + 2.0) * w2 - 2.0 * (a + 3.0) * w;
		xWeightGradient[2] = -3.0 * (a + 2.0) * w2 + 2.0 * (2.0 * a + 3.0) * w - a;
		xWeightGradient[3] = a * (-3.0 * w2 + 2.0 * w);
	}

	w = Y - (double)yIndex[0];
	w2 = w*w; w3 = w2*w;
	yWeight[0] = a * (w3 - 2.0 * w2 + w);
	yWeight[1] = (a + 2.0) * w3 - (a + 3.0) * w2 + 1;
	yWeight[2] = -(a + 2.0) * w3 + (2.0 * a + 3.0) * w2 - a * w;
	yWeight[3] = a * (-w3 + w * w);

	if (S_Flag > -1)
	{
		yWeightGradient[0] = a * (3.0 * w2 - 4.0 * w + 1);
		yWeightGradient[1] = 3.0 * (a + 2.0) * w2 - 2.0 * (a + 3.0) * w;
		yWeightGradient[2] = -3.0 * (a + 2.0) * w2 + 2.0 * (2.0 * a + 3.0) * w - a;
		yWeightGradient[3] = a * (-3.0 * w2 + 2.0 * w);
	}
	//***********************************

	S[0] = 0;
	if (S_Flag > -1)
	{
		S[1] = 0;
		S[2] = 0;
	}

	for (i = 0; i < 4; i++)
	{
		for (j = 0; j < 4; j++)
		{
			Image_value = (*(Para + width_ex * yIndex[i] + xIndex[j]));
			S[0] = S[0] + Image_value * xWeight[j] * yWeight[i];
			if (S_Flag > -1)
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
	int ul = (int)(u), uh = (int)(u + 1), vl = (int)v, vh = (int)(v + 1);
	double ufrac = u - 1.0*ul, vfrac = v - 1.0*vl;

	int f00 = data[ul + vl*width];
	int f10 = data[uh + vl*width];
	int f01 = data[ul + vh*width];
	int f11 = data[uh + vh*width];

	double res = (1.0 - ufrac)*(1.0 - vfrac)*f00 + ufrac*(1.0 - vfrac)*f10 + (1.0 - ufrac)*vfrac*f01 + ufrac*vfrac*f11;
	return (int)(res + 0.5);
}
double BilinearInterp(double *data, int width, int height, double x, double y)
{
	if (x<0 || x>width - 2 || y<0 || y>height - 2)
		return 255; //Make it white

	int xiD = (int)(x), yiD = (int)(y);
	int xiU = xiD + 1, yiU = yiD + 1;

	double f00 = data[xiD + yiD*width];
	double f01 = data[xiU + yiD*width];
	double f10 = data[xiD + yiU*width];
	double  f11 = data[xiU + yiU*width];
	double res = (f01 - f00)*(x - xiD) + (f10 - f00)*(y - yiD) + (f11 - f01 - f10 + f00)*(x - xiD)*(y - yiD) + f00;

	return res;
}