#include <cmath>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#if !defined(IMAGEPRO_H )
#define IMAGEPRO_H
using namespace cv;

void filter1D_row_Double(double *kernel, int k_size, double *in, double *out, int width, int height);
void filter1D_row(double *kernel, int k_size, char *in, double *out, int width, int height);
void filter1D_col(double *kernel, int k_size, double *in, double *out, int width, int height, double &i_max);
void filter1D_row(double *kernel, int k_size, unsigned char *in, float *out, int width, int height);
void filter1D_row(double *kernel, int k_size, char *in, float *out, int width, int height);
void filter1D_col(double *kernel, int k_size, float *in, float *out, int width, int height, double &i_max);
void Gaussian_smooth(unsigned char* data, double* out_data, int height, int width, double max_i, double sigma);
void Gaussian_smooth(unsigned char* data, float* out_data, int height, int width, double max_i, double sigma);
void Gaussian_smooth(char* data, double* out_data, int height, int width, double max_i , double sigma);
void Gaussian_smooth(char* data, float* out_data, int height, int width, double max_i, double sigma);
void Gaussian_smooth(double* data, double* out_data, int height, int width, double max_i, double sigma);

void Generate_Para_Spline(double *Image, double *Para, int width, int height, int Interpolation_Algorithm);
void Generate_Para_Spline(char *Image, float *Para, int width, int height, int Interpolation_Algorithm);
void Generate_Para_Spline(float *Image, float *Para, int width, int height, int Interpolation_Algorithm);
void Generate_Para_Spline(char *Image, double *Para, int width, int height, int Interpolation_Algorithm);
void Generate_Para_Spline(unsigned char *Image, double *Para, int width, int height, int Interpolation_Algorithm);
void Generate_Para_Spline(int *Image, double *Para, int width, int height, int Interpolation_Algorithm);
void Get_Value_Spline(float *Para, int width,int height, double X, double Y, double *S, int S_Flag, int Interpolation_Algorithm);
void Get_Value_Spline(double *Para, int width,int height, double X, double Y, double *S, int S_Flag, int Interpolation_Algorithm);

//Bicubic Spline
void Generate_Para_BiCubic_Spline_Double(double *Image, double *Para, int width, int height);
void Get_Value_BiCubic_Spline(double *Para, int width_ex, int height_ex, double X, double Y, double *S, int S_Flag);

//Linear interpolation
int LinearInterp(int *data, int width, int height, double u, double v);
double BilinearInterp(double *data, int width, int height, double x, double y);

#endif
