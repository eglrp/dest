// Copyright (C) 1991 - 1999 Rational Software Corporation

#pragma once

#ifndef _INC_MATRIX_3CD04AE9004A_INCLUDED
#define _INC_MATRIX_3CD04AE9004A_INCLUDED

#include "stdio.h"
#include <iostream>
#include <math.h>
#include <iomanip>


//matrix class provides matrix creation, access and matrix 
//operation methods
//##ModelId=3CD04AE9004A
class Matrix 
{
public:
	//##ModelId=3CD053F201FE
	Matrix();
	explicit Matrix(unsigned int size);
	Matrix(unsigned int r, unsigned int c);
	Matrix(const Matrix &m);


	Matrix operator-(const Matrix &Term) const;
	Matrix operator-(double Scalar) const;
	Matrix operator+(const Matrix &Term) const;
	Matrix operator+(double Scalar) const;
	Matrix operator*(const Matrix &Term) const;
	Matrix operator*(int Scalar) const;
	Matrix operator*(double Scalar) const;
	Matrix operator/(int Scalar) const;
	Matrix operator/(double Scalar) const;
	bool operator==(const Matrix& RHS) const;
	bool operator!=(const Matrix& RHS) const;
	double*operator() (int	index) const;

	void Set_Sub_Mat(Matrix A, unsigned int s_r,unsigned int s_c);
	Matrix Get_Sub_Mat(unsigned int s_r,unsigned int s_c,unsigned int g_r,unsigned int g_c) const;
	Matrix Get_Row(unsigned int I) const;
	Matrix Get_Column(unsigned int J) const;
	unsigned int Get_Num_Rows() const;
	unsigned int Get_Num_Columns() const;

	double& Cell(unsigned int I, unsigned int J) const;		// Index accessor
	friend std::ostream& operator<<(std::ostream& Output, const Matrix& Matrix_To_Display);

	//##ModelId=3CD053F2023B
	~Matrix();

	// make it self identity
	void Identity();
	void Set_Zero();
	void Matrix_Init(double* data);
	double& operator[](unsigned int i) const;
	Matrix& operator=(const Matrix &a);

	//addition of two metrices: the input metrix and the 
	//matrix object tiself.  The sum of two matrix replace 
	//the object itself.
	//If the size of the matrix object is different from the 
	//size of input matrix, return false.
	//##ModelId=3CD04DCB02A4
	bool addition(Matrix &);

	//the input matrix subtract is subtracted from the matrix 
	//object and the reslur matrix is returned to the object 
	//itself.  If the object matrix is different from the 
	//size of the input matrix, return false.
	//##ModelId=3CD04DD002DD
	bool subtraction(Matrix &);

	//the matrix object is multiplied with the input matrix 
	//and the result of the multiplication replace the origin 
	//object. If the column number of the matrix object is 
	//different from the row number of the input matrix, 
	//returns false.
	//##ModelId=3CD04DD3011F
	bool multiplication(Matrix &);

	//the object is replaced by a vector containing the 
	//eigenvalues of the matrix object, if the object is not 
	//a square matrix, returns false.
	bool Eigen (Matrix &eval,Matrix &evec); 

	//to calculate the sum of diagonal elements of the matrix 
	//object, if the object is not a square matrix, return 
	//false.
	//##ModelId=3CD04F3E0366
	double trace() const;

	double determinant ( void ) ;

	Matrix Inv_3x3(void);
	Matrix Inversion(bool symetric = false, bool positive_definite = false);

	Matrix Pinv();
	double radius(double u, double v);
	void SVDcmp(int nrow, int ncol, double *u,double *w, double *v,int flags=0);
	void svdcmp(double **a, int m, int n, double *w, double **v);

	bool SolveCubic(double *res, int &n);
	void SolvePoly(double *res);
	Matrix Closest_RankN_Approx(int rank_approx, int ncol, int nrow);

	double L2_norm(void);
	double Linf_norm(void);
	
	Matrix Normalize();
	Matrix Transpose() const;
	
	// to write and read the matrix in text form
	void WriteTxt (FILE *fptr);
	void ReadTxt (FILE *fptr);
	
	Matrix Minor(unsigned int Row, unsigned int Column) const;
	void Imprint(const Matrix& Original, int Row_Offset = 0, int Column_Offset = 0);

	void ConvertedFromVector(double *V, int dimension);
	void ConvertedToVector(double *V, int dimension);

	void ReleaseMatrix();

private:
	//number of row in the matrix object
	//##ModelId=3CD04CED0255
	unsigned int row;

	//number of columns in the matrix object
	//##ModelId=3CD04D5200C9
	unsigned int column;

	int Mem_Flag;

	//metrix elements
	//##ModelId=3CD04D710218
public :
	double *matrix ;

private :
	bool triangulate();
	void scale(double x);
	void matrix_of_cofactors();
	double cofactor(int i, int j);
	bool my_minor(int i, int j);
	void transpose ( void );
};

Matrix operator*(int Coefficient, const Matrix& Term);
Matrix operator*(double Coefficient, const Matrix& Term);

#endif /* _INC_MATRIX_3CD04AE9004A_INCLUDED */
