#include "Matrix.h"
#pragma once
#include <iostream>
#include <fstream>
#include <omp.h>
#include <vector>
#include <opencv2/opencv.hpp>


#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#ifndef __max
#define __max(a,b) (((a) > (b)) ? (a) : (b))
#endif

using namespace cv;

// constructor
Matrix::Matrix()
{
	column = 1;
	row = 1;
	matrix = new double[1];
	matrix[0] = 0.0;
	Mem_Flag = 12345678;
}

Matrix Matrix::Minor(unsigned int I, unsigned int J) const
{
	if(row <= 1 || column <= 1) // Make sure the matrix minor will have at least 1 row and 1 column.
		return Matrix();

	Matrix Result(row - 1, column - 1);

	unsigned int Row_Offset = 0;
	unsigned int Column_Offset = 0;

	for(unsigned int Current_Column = 0; Current_Column < column; Current_Column++)
	{
		Row_Offset = 0 ;
		if(Current_Column == J)
			continue;
		for(unsigned int Current_Row = 0; Current_Row < row; Current_Row++)
		{
			if(Current_Row == I)
				continue;
			Result.Cell(Row_Offset, Column_Offset) = Cell(Row_Offset, Column_Offset);
			Row_Offset++;
		}
		Column_Offset++;
	}

	return Result;
}

void Matrix::Imprint(const Matrix& Original, int Row_Offset, int Column_Offset)
{
	if(Original.column + Column_Offset > column || Original.row + Row_Offset > row)
		return;

	for(unsigned int i = 0; i < Original.row; i++)
		for(unsigned int j = 0; j < Original.row; j++)
			Cell(i + Row_Offset, j + Column_Offset) = Original.Cell(i, j);
}

// constructor - given size value
Matrix::Matrix(unsigned int size)
{
  if(size == 0)
    size = 1;
  column = row = size ;
  matrix = new double[row*column] ; //(double *)malloc ( column*row*sizeof(double) ) ;
  memset ( matrix, 0, row*column*sizeof(double) ) ;
  Mem_Flag = 12345678;
}

// constructor - given row and column value
Matrix::Matrix ( unsigned int r, unsigned int c )
{
  column = c, row = r ;
  matrix = new double [row*column] ; //(double *)malloc ( column*row*sizeof(double) ) ;
  memset ( matrix, 0, row*column*sizeof(double) ) ;
  Mem_Flag = 12345678;
}

// copy constructor
Matrix::Matrix(const Matrix &a) {
  column = a.column, row = a.row ;
  matrix = new double [row * column] ;
  memcpy( matrix, a.matrix, row * column * sizeof(double) );
  Mem_Flag = 12345678;
}

// destructor
Matrix::~Matrix()
{
  ReleaseMatrix();
}

void Matrix::ReleaseMatrix()
{
	if (Mem_Flag == 12345678)
	{
		delete [] matrix;
		Mem_Flag = 0;
	}
}

void Matrix::Set_Sub_Mat(Matrix A, unsigned int s_r,unsigned int s_c)
{
	unsigned int ii,jj, g_r = A.row, g_c = A.column;
	
	for(jj=s_r;jj<g_r+s_r;jj++)
		for(ii=s_c;ii<g_c+s_c;ii++)
			Cell(jj,ii) = A[ii-s_c+(jj-s_r)*g_c];

	return;
}

Matrix Matrix::Get_Sub_Mat(unsigned int s_r,unsigned int s_c,unsigned int g_r,unsigned int g_c) const
{
	unsigned int ii,jj;
	Matrix sub(g_r,g_c);
	
	for(jj=s_r;jj<g_r+s_r;jj++)
		for(ii=s_c;ii<g_c+s_c;ii++)
			sub[ii-s_c+(jj-s_r)*g_c]=Cell(jj,ii);

	return sub;
}

Matrix Matrix::Get_Row(unsigned int I) const
{
	if(I > row)
		return Matrix();
	Matrix The_Row(1, column);
	for(unsigned int Count = 0; Count < column; Count++)
		The_Row.Cell(0, Count) = Cell(I, Count);
	return The_Row;
}

Matrix Matrix::Get_Column(unsigned int J) const
{
	if(J > column)
		return Matrix();

	Matrix The_Column(row, 1);

	for(unsigned int Count = 0; Count < row; Count++)
		The_Column.Cell(Count, 0) = Cell(Count, J);

	return The_Column;
}

unsigned int Matrix::Get_Num_Rows() const
{
	return row;
}

unsigned int Matrix::Get_Num_Columns() const
{
	return column;
}

double& Matrix::Cell(unsigned int I, unsigned int J) const
{
#ifdef DEBUG
	if(I >= row || J >= column)
		std::cout << "Bounds error!\n";
#endif
	return matrix[I*column+J];
}

void Matrix:: Matrix_Init(double* data)
{
	unsigned int ii,jj;
	
	for(jj=0;jj<row;jj++)
		for(ii=0;ii<column;ii++)
			matrix [ii+jj*column]=data[ii+jj*column];
	return;
}

// make it self identity
void Matrix::Identity()
{
	for (unsigned int i = 0; i < row; i++)
		for (unsigned int j = 0; j < column; j++)
			if (i == j)
				matrix[i * column + j] = 1;
			else
				matrix[i * column + j] = 0;
}

void Matrix::Set_Zero()
{
	for (unsigned int i = 0; i < row; i++)
		for (unsigned int j = 0; j < column; j++)
				matrix[i * column + j] = 0.0;
}

// access operator
double& Matrix::operator[](unsigned int i) const
{  

	return matrix[i];
}

// operator =
Matrix& Matrix::operator = (const Matrix &a) 
{
  if(&a == this) // check for self assignment
    return *this;

  delete [] matrix ;
  column = a.column, row = a.row ;
  matrix = new double [row * column] ;
  memcpy ( matrix, a.matrix, row * column * sizeof(double) ) ;
  return *this;
}

// Equality comparision operator - checks to see if two matricies are the same.
bool Matrix::operator==(const Matrix& RHS) const
{
	if(&RHS == this) // The matrix is always equal to itself.
		return true;
	if((row != RHS.row) || (column != RHS.column)) // Dimension difference imples inequality.
		return false;
	unsigned int Number_Of_Elements = column * row;
	for(unsigned int Index = 0; Index < Number_Of_Elements; Index++) // Compare each element of both matricies.
		if(matrix[Index] != RHS.matrix[Index])
			return false;
	return true; // Dimensions of the matricies and all elements are the same, return success.
}

// Inquality comparision operator - checks to see if two matricies are different.
bool Matrix::operator!=(const Matrix& RHS) const
{
	if(&RHS == this) // The matrix is never not equal to itself.
		return false;
	return !(*this == RHS);
}



//addition of two metrices: the input metrix and the matrix 
//object tiself.  The sum of two matrix replace the object 
//itself.
//If the size of the matrix object is different from the size 
//of input matrix, return false.
//##ModelId=3CD04DCB02A4
bool Matrix::addition(Matrix &a)
{
  if ( row != a.row || column != a.column )
    return false ;

  int total = row * column ;

  for ( int i = 0 ; i < total ; i++ )
    matrix[i] += a.matrix[i] ;

  return true ;
}

Matrix Matrix::operator+(double Scalar) const
{
	Matrix Result(row,column);
	unsigned int Total = row * column ;
	for(unsigned int Index = 0; Index < Total; Index++)
		Result.matrix[Index] = matrix[Index] + Scalar;
	return Result;
}
// Addition of two matrices - example in a simple assignment statement:
// Return value    *this       Term
// [ 1  2  3 ]     [ 0 1 2 ]   [ 1 1 1 ]
// [ 5  6  7 ]  =  [ 3 4 5 ] + [ 2 2 2 ]
// [ 9 10 11 ]     [ 6 7 8 ]   [ 3 3 3 ]
// Note: If the size of the matrix object is different
// from the size of input matrix then the function fails.
Matrix Matrix::operator+(const Matrix &Term) const
{
	if(row != Term.row || column != Term.column)
		return Matrix(); // TODO: Dimension mismatch. Throw an exception here?
	Matrix Result(row, column);
	unsigned int Total = row * column ;
	for(unsigned int Index = 0; Index < Total; Index++)
		Result.matrix[Index] = matrix[Index] + Term.matrix[Index];
	return Result;
}

//the input matrix subtract is subtracted from the matrix 
//object and the reslur matrix is returned to the object 
//itself.  If the object matrix is different from the size of 
//the input matrix, return false.
//##ModelId=3CD04DD002DD
bool Matrix::subtraction(Matrix &a)
{
  if ( row != a.row || column != a.column )
    return false ;

  int total = row * column ;

  for ( int i = 0 ; i < total ; i++ )
    matrix[i] -= a.matrix[i] ;

  return true ;
}

// Subtraction of two matrices - example in a simple assignment statement:
// Return value     *this       Term
// [ -1  0  1 ]     [ 0 1 2 ]   [ 1 1 1 ]
// [  1  2  3 ]  =  [ 3 4 5 ] - [ 2 2 2 ]
// [  3  4  5 ]     [ 6 7 8 ]   [ 3 3 3 ]
// Note: If the size of the matrix object is different
// from the size of input matrix then the function fails.
Matrix Matrix::operator-(const Matrix &Term) const
{
	if(row != Term.row || column != Term.column)
		return Matrix(); // TODO: Dimension mismatch. Throw an exception here?
	Matrix Result(row, column);
	unsigned int Total = row * column ;
	for(unsigned int Index = 0; Index < Total; Index++)
		Result.matrix[Index] = matrix[Index] - Term.matrix[Index];
	return Result;
}
Matrix Matrix::operator-(double Scalar) const
{
	Matrix Result(row, column);
	unsigned int Total = row * column ;
	for(unsigned int Index = 0; Index < Total; Index++)
		Result.matrix[Index] = matrix[Index] - Scalar;
	return Result;
}

//the matrix object is multiplied with the input matrix and 
//the result of the multiplication replace the origin object. 
//If the column number of the matrix object is different from 
//the row number of the input matrix, returns false.
//##ModelId=3CD04DD3011F
bool Matrix::multiplication(Matrix &a)
{
  if ( column != a.row )
    return false ;

  int new_row = row, new_col = a.column ;
  double *new_matrix = new double[new_row*new_col] ;

  for ( unsigned int i = 0 ; i < row ; i++ ) {
    for ( unsigned int j = 0 ; j < a.column ; j++ ) {
      new_matrix[i*new_col+j] = 0 ;
      for ( unsigned int k = 0 ; k < column; k++ )
        new_matrix[i*new_col+j] += matrix[i*column+k] * a.matrix[k*a.column+j] ;
    }
  }

  delete [] matrix;
  column = new_col, row = new_row ;
  matrix = new_matrix ;

  return true ;
}

// Multiplication of two matrices - example in a simple assignment statement:
// Return value     *this       Term
// [  8  8  8 ]     [ 0 1 2 ]   [ 1 1 1 ]
// [ 26 26 26 ]  =  [ 3 4 5 ] * [ 2 2 2 ]
// [ 44 44 44 ]     [ 6 7 8 ]   [ 3 3 3 ]
// Note: If the column number of the matrix is different from 
// the row number of the input matrix then the function fails.
Matrix Matrix::operator*(const Matrix &Term) const
{
	if(column != Term.row )
		return Matrix(); // TODO: Dimension mismatch. Throw an exception here?

	Matrix Result(row, Term.column);

	for(unsigned int I = 0; I < row; I++) {
		for(unsigned int J = 0 ; J < Term.column; J++) {
			Result.matrix[I*Term.column+J] = 0;
			for(unsigned int K = 0 ; K < column; K++)
				Result.matrix[I*Term.column+J] += matrix[I*column+K] * Term.matrix[K*Term.column+J];
		}
	}

	return Result;
}

Matrix Matrix::operator*(double Scalar) const
{
	Matrix Result(row, column);
	for(unsigned int I = 0; I < row; I++)
		for(unsigned int J = 0 ; J < column; J++)
			Result.Cell(I, J) = Cell(I, J) * Scalar;
	return Result;
}

Matrix Matrix::operator*(int Scalar) const
{
	Matrix Result(row, column);
	unsigned int Total = row * column ;
	for(unsigned int Index = 0; Index < Total; Index++)
		Result.matrix[Index] = matrix[Index] * Scalar;
	return Result;
}

Matrix Matrix::operator /(double Scalar) const
{
	Matrix Result(row, column);
	for(unsigned int I = 0; I < row; I++)
		for(unsigned int J = 0 ; J < column; J++)
			Result.Cell(I, J) = Cell(I, J) /Scalar;
	return Result;
}

Matrix Matrix::operator /(int Scalar) const
{
	Matrix Result(row, column);
	for(unsigned int I = 0; I < row; I++)
		for(unsigned int J = 0 ; J < column; J++)
			Result.Cell(I, J) = Cell(I, J) /Scalar;
	return Result;
}
//the object is replaced by a vector containing the 
//eigenvalues of the matrix object, if the object is not a 
//square matrix, returns false.
//##ModelId=3CD04F38010E
bool Matrix::Eigen(Matrix &EVAL, Matrix &EVEC)
{
	if(column==row)
	{
		int ii,jj,size=column;
		CvMat *A = cvCreateMat( size,size, CV_64F );
		CvMat *evec = cvCreateMat( size,size, CV_64F );
		CvMat *eval  = cvCreateMat( 1,size, CV_64F );
		
		Matrix a=*this;

		for(jj=0;jj<size;jj++)
			for(ii=0;ii<size;ii++)
				A->data.db[ii+jj*size]=a[ii+jj*size];

		cvEigenVV(A,evec,eval);

		for(jj=0;jj<size;jj++)
			for(ii=0;ii<size;ii++)
				EVEC[ii+jj*size]=evec->data.db[ii+jj*size];

		for(jj=0;jj<size;jj++)
			EVAL[jj]=eval->data.db[jj];

		cvReleaseMat(&A);
		cvReleaseMat(&evec);
		cvReleaseMat(&eval);

		return true;
	}
	else
		return false;

}

//to calculate the sum of diagonal elements of the matrix 
//object, if the object is not a square matrix, return false.
// how can i return false if return type is double? (andrey)
//##ModelId=3CD04F3E0366
double Matrix::trace() const
{
	if(column != row)
		return 0.0;
	double Result = 0.0;
	for(unsigned int i = 0; i < column; i++)
		Result += Cell(i, i);
	return Result;
}

Matrix Matrix::Transpose() const
{
	Matrix Result(column, row);
	for(unsigned int i = 0; i < row; i++)
		for(unsigned int j = 0; j < column; j++)
			Result.Cell(j, i) = Cell(i, j);
	return Result;
}


///////////// private sector ////////////
double Matrix::determinant ( void )
{
  unsigned int ii,jj;
  double det = 0 ;
  Matrix t ( column ) ;
  t = *this ;

  Mat mat(row,column, CV_64F);
  for(jj=0;jj<row;jj++)
	  for(ii=0;ii<column;ii++)
		  mat.at<double>(jj, ii)=t.matrix[ii+jj*column];
	
  det = cv::determinant(mat); 
  return det ;
}

bool Matrix::triangulate ( void )
{
  int chngcnt = 0 ;
  unsigned int i, j, k ;
  double c, tmp ;

  // for all rows
  for ( i = 0 ; i < row ; i++ ) {

    if ( matrix[i*column+i] == 0 ) {

      for ( k = i + 1 ; k < row ; k++ )
        if ( matrix[k*column+i] )
          break ;

      if ( k == column )
        return false ;

      for ( j = i ; j < column ; j++ )
        tmp = matrix[i*column+j], matrix[i*column+j] = matrix[k*column+j], matrix[k*column+j] = tmp ;

      chngcnt++ ;
    }

    for ( k = i+1 ; k < row ; k++ ) {
      c = -matrix[k*column+i]/matrix[i*column+i] ;

      for ( j = i ; j < column ; j++ )
        matrix[k*column+j] = matrix[k*column+j] + c*matrix[i*column+j] ;  
    }
  }

  if ( chngcnt & 1 )
    scale(-1) ;

  return true ;
}

void Matrix::scale ( double value )
{
  int total = column * row ;
  for ( int i = 0 ; i < total ; i++ )
    matrix[i] *= value ;
}

void Matrix::transpose ( void )
{
  unsigned int i, j ;
  double *new_matrix = new double[column * row] ;

  for ( i = 0 ; i < column ; i++ )
    for ( j = 0 ; j < row ; j++ )
      new_matrix[i*row+j] = matrix[j*column+i] ;

  int temp = row;
  row = column;
  column = temp;
  delete [] matrix;
  matrix = new_matrix;
}

// to write out the matrix in text form
void Matrix::WriteTxt (FILE *fptr)
{
	//fprintf (fptr, "Row: %d\n", row);
	//fprintf (fptr, "Column: %d\n", column);

	for (unsigned int i = 0; i < row; i++)
	{
		for (unsigned int j = 0; j < column; j++)
		{
			fprintf (fptr, "%.16e ", matrix[i*column+j]);
		}
		fprintf (fptr, "\n");
	}
}

// to read in the matrix in text form
void Matrix::ReadTxt (FILE *fptr)
{
	fscanf (fptr, "Row: %d\n", &row);
	fscanf (fptr, "Column: %d\n", &column);

	if (matrix)
		delete [] matrix;
	matrix = new double[row * column];

	for (unsigned int i = 0; i < row; i++)
	{
		for (unsigned int j = 0; j < column; j++)
		{
			fscanf (fptr, "%lf ", &matrix[i*row+j]);
		}
		fscanf (fptr, "\n");
	}
}

/****************************************** PRIVATE FUNCTION **************************************/

void Matrix::matrix_of_cofactors ( void )
{
  unsigned int i;
  unsigned int j ;
  Matrix t(column,row) ;

  t = *this ;

  for ( i = 0 ; i < row ; i++ )
    for ( j = 0 ; j < column ; j++ )
      matrix[i*column+j] = t.cofactor(i,j) ;
}

double Matrix::cofactor ( int i, int j )
{
  Matrix m ( column, row ) ;

  m = *this ;
  m.my_minor ( i, j ) ;

  double det = m.determinant() ;

  if ( ( i + j ) & 1 )
    det = -det ;

  return det ;
}

bool Matrix::my_minor ( int i, int j )
{
  int new_col = column - 1, new_row = row - 1 ;

  if ( new_col <= 0 || new_row <= 0 )
    return false ;

  double *new_matrix = new double [ new_col * new_row ] ;

  unsigned int di = 0, dj = 0, ci, cj ;

  for ( cj = 0 ; cj < column ; cj++ ) {
    di = 0 ;
    if ( cj == j )
      continue ;

    for ( ci = 0 ; ci < row ; ci++ ) {
      if ( ci == i )
        continue ;
      new_matrix[di*new_col+dj] = matrix[ci*column+cj] ;
      di++ ;
    }

    dj++ ;
  }

  row = new_row, column = new_col ;
  delete [] matrix ;
  matrix = new_matrix ;

  return true ;
}

std::ostream& operator<<(std::ostream& Output, const Matrix& Matrix_To_Display)
{
	Output << Matrix_To_Display.row << '\n';
	Output << Matrix_To_Display.column << '\n';
	unsigned int Total = Matrix_To_Display.row * Matrix_To_Display.column;
	for(unsigned int Index = 0; Index < Total; Index++)
	{
		if((Index != 0) && (Index % Matrix_To_Display.column == 0))
			Output << '\n';
		Output << std::setw(15) << Matrix_To_Display.matrix[Index];
	}
	return Output;
}

Matrix Matrix::Inv_3x3(void )
{
	//Taken from IPL1
	Matrix M = *this;
	Matrix IM(3, 3);
	
	IM[0] = M[4]*M[8] - M[5]*M[7];
	IM[1] = M[2]*M[7] - M[1]*M[8];
	IM[2] = M[1]*M[5] - M[2]*M[4];
	IM[3] = M[5]*M[6] - M[3]*M[8];
	IM[4] = M[0]*M[8] - M[2]*M[6];
	IM[5] = M[2]*M[3] - M[0]*M[5];
	IM[6] = M[3]*M[7] - M[4]*M[6];
	IM[7] = M[1]*M[6] - M[0]*M[7];
	IM[8] = M[0]*M[4] - M[1]*M[3];
	double det = M[0]*IM[0] + M[1]*IM[3] + M[2]*IM[6];
	for(int i=0; i<9; i++)
		IM[i] = IM[i]/det;

	return IM;
}

Matrix Matrix::Inversion(bool symmetric, bool positive_definite)
{
	unsigned int ii,jj;
	Matrix in= *this;
	Matrix out(column);

	CvMat* Mat = cvCreateMat(row,column,CV_64FC1);
	CvMat* Mat_r = cvCreateMat(row,column,CV_64FC1);
	for(jj=0;jj<row;jj++)
		for(ii=0;ii<column;ii++)
			Mat->data.db[ii+jj*column]=in[ii+jj*column];

	if(positive_definite == 1 && symmetric == 1)
		cvInvert(Mat, Mat_r, 3);
	else if(symmetric==1)
		cvInvert(Mat,Mat_r, 2);
	else
		cvInvert(Mat,Mat_r, 1);
	
	for(jj=0;jj<row;jj++)
		for(ii=0;ii<column;ii++)
			out[ii+jj*column]=Mat_r->data.db[ii+jj*column];

	cvReleaseMat(&Mat);
	cvReleaseMat(&Mat_r);

	return out;
}

//Pseudo inverse
Matrix Matrix::Pinv()
{
	Matrix A = *this;
	int m,n,i,j,temp,flag;

	flag=0;

	m = this->Get_Num_Rows();
	n = this->Get_Num_Columns();

	if(m<n)
	{
		flag=1;
		temp=m;
		m=n;
		n=temp;
		A.transpose();
	}

	Matrix T(n, m);
	Matrix U(m, n);
	Matrix W(n, n);
	Matrix V(n, n);

	double **a;
	double *w = new double[n];
	double **v;

	a = (double **)malloc(m*sizeof(double *)); 
	v = (double **)malloc(n*sizeof(double *)); 
	for(i=0;i<m;i++)
		a[i]=(double *)malloc(n*sizeof(double));
	for(i=0;i<n;i++)
		v[i]=(double *)malloc(n*sizeof(double));

	for(i=0;i<m;i++)
	{
		for(j=0;j<n;j++)
		{
			a[i][j] = A.Cell(i,j);
		}
	}

	svdcmp(a, m, n, w, v);

	for(i=0;i<m;i++)
	{
		for(j=0;j<n;j++)
		{
			U.Cell(i,j)=a[i][j];
		}
	}

	for(i=0;i<n;i++)
	{
		W.Cell(i,i)=1.0/w[i];
		if(w[i]<1E-8)
			W.Cell(i,i)=0.0;
	}

	for(i=0;i<n;i++)
	{
		for(j=0;j<n;j++)
		{
			V.Cell(i,j)=v[i][j];
		}
	}

	T=V*W*U.Transpose();

	if(flag==1)
		T.transpose();

	free(v);
	free(a);
	delete []w;

	return T;
}

//Sign and radius are used in svdcmp()
#define Sign(u,v)	( (v)>=0.0 ? fabs(u) : -fabs(u) )

double Matrix::radius(double u, double v)
{
	double  Au, Av, Aw;
	
	Au = fabs(u);
	Av = fabs(v);
	
	if (Au > Av)
	{
		Aw = Av / Au;
		return (Au * sqrt(1. + Aw * Aw));
	}
	else
	{
		if (Av>0.0)
		{
			Aw = Au / Av;
			return (Av * sqrt(1. + Aw * Aw));
		}
		else
		{
			return 0.0;
		}
	}
}

/*************************** SVDcmp *****************************************
 * Given matrix A[m][n], m>=n, using svd decomposition A = U W V' to get     *
 * U[m][m], W[m][n] and V[n][n], where U occupies the position of A.         *
 * NOTE: if m<n, A should be filled up to square with zero rows.             *
 * A =U*W*VT																				*
 * flag=CV_SVD_MODIFY_A: Allows modifi cation of matrix A
 * flag=CV_SVD_U_T: Return UT instead of U
 *flag= CV_SVD_V_T: Return VT instead of V
 *ex: xx.SVDcmp(4,4,U,W,V,CV_SVD_U_T + CV_SVD_V_T);*
 ****************************************************************************/

void Matrix::SVDcmp( int nrow, int ncol, double *u,double *w, double *v,int flags)
{
	int ii,jj;
	CvMat *A = cvCreateMat( nrow,ncol, CV_64F );
	CvMat *U = cvCreateMat( nrow,nrow, CV_64F );
	CvMat *W =cvCreateMat( nrow,ncol, CV_64F );
	CvMat *V =cvCreateMat( ncol,ncol, CV_64F );
	
	Matrix a=*this;

	for(jj=0;jj<nrow;jj++)
		for(ii=0;ii<ncol;ii++)
			A->data.db[ii+jj*ncol]=a[ii+jj*ncol];

	cvSVD(A,W,U,V,flags);
	
	for(jj=0;jj<nrow;jj++)
		for(ii=0;ii<nrow;ii++)
			u[ii+jj*nrow]=U->data.db[ii+jj*nrow];

	for(jj=0;jj<nrow;jj++)
		for(ii=0;ii<ncol;ii++)
			w[ii+jj*ncol]=W->data.db[ii+jj*ncol];

	for(jj=0;jj<ncol;jj++)
		for(ii=0;ii<ncol;ii++)
			v[ii+jj*ncol]=V->data.db[ii+jj*ncol];
	
	cvReleaseMat(&A);
	cvReleaseMat(&U);
	cvReleaseMat(&W);
	cvReleaseMat(&V);
	return;
}

/*************************** SVDcmp *****************************************
 * Given matrix A[m][n], m>=n, using svd decomposition A = U W V' to get     *
 * U[m][n], W[n][n] and V[n][n], where U occupies the position of A.         *
 * NOTE: if m<n, A should be filled up to square with zero rows.             *
 *       A[m][n] has been destroyed by U[m][n] after the decomposition.      *
 ****************************************************************************/
void Matrix::svdcmp(double **a, int m, int n, double *w, double **v)
{

	int flag, i, its, j, jj, k, l, nm, nm1 = n - 1, mm1 = m - 1;
	double  c, f, h, s, x, y, z;
	double  anorm = 0.0, g = 0.0, scale = 0.0;
	double *rv1;
	
	rv1 = (double *) malloc((unsigned) n * sizeof(double));

	/* Householder reduction to bidigonal form */
	for (i = 0; i < n; i++)
	{
		l = i + 1;
		rv1[i] = scale * g;
		g = s = scale = 0.0;
		if (i < m)
		{

			for (k = i; k < m; k++)
				scale += fabs(a[k][i]);
			if (scale)
			{
				for (k = i; k < m; k++)
				{
					a[k][i] /= scale;
					s += a[k][i] * a[k][i];
				}

				f = a[i][i];
				g = -Sign(sqrt(s), f);
				h = f * g - s;
				a[i][i] = f - g;
				if (i != nm1)
				{
					for (j = l; j < n; j++)
					{
						for (s = 0.0, k = i; k < m; k++)
							s += a[k][i] * a[k][j];
						f = s / h;
						for (k = i; k < m; k++)
							a[k][j] += f * a[k][i];
					}
				}

				for (k = i; k < m; k++)
					a[k][i] *= scale;
			}
		}
		w[i] = scale * g;
		g = s = scale = 0.0;
		if (i < m && i != nm1)
		{
			for (k = l; k < n; k++)
				scale += fabs(a[i][k]);
			if (scale)
			{
				for (k = l; k < n; k++)
				{
					a[i][k] /= scale;
					s += a[i][k] * a[i][k];
				}
                 
				f = a[i][l];
				g = -Sign(sqrt(s), f);
				h = f * g - s;
				a[i][l] = f - g;
				for (k = l; k < n; k++)
					rv1[k] = a[i][k] / h;
				if (i != mm1)
				{
					for (j = l; j < m; j++)
					{
						for (s = 0.0, k = l; k < n; k++)
							s += a[j][k] * a[i][k];
						for (k = l; k < n; k++)
							a[j][k] += s * rv1[k];
					}
				}

				for (k = l; k < n; k++)
					a[i][k] *= scale;
			}
		}
		anorm = __max(anorm, (fabs(w[i]) + fabs(rv1[i])));
	}
 
	/* Accumulation of right-hand transformations */
	for (i = n - 1; i >= 0; i--)
	{
		if (i < nm1)
		{
			if (g)
			{
				/* double division to avoid possible underflow */
				for (j = l; j < n; j++)
					v[j][i] = (a[i][j] / a[i][l]) / g;
				for (j = l; j < n; j++)
				{
					for (s = 0.0, k = l; k < n; k++)
						s += a[i][k] * v[k][j];
					for (k = l; k < n; k++)
						v[k][j] += s * v[k][i];
				}
			}
			for (j = l; j < n; j++)
				v[i][j] = v[j][i] = 0.0;
		}
		v[i][i] = 1.0;
		g = rv1[i];
		l = i;
	}

	/* Accumulation of left-hand transformations */
	for (i = n - 1; i >= 0; i--)
	{
		l = i + 1;
		g = w[i];
		if (i < nm1)
			for (j = l; j < n; j++)
				a[i][j] = 0.0;
		if (g)
		{
			g = 1.0 / g;
			if (i != nm1)
			{
				for (j = l; j < n; j++)
				{
					for (s = 0.0, k = l; k < m; k++)
						s += a[k][i] * a[k][j];
					f = (s / a[i][i]) * g;
					for (k = i; k < m; k++)
						a[k][j] += f * a[k][i];
				}
			}
			for (j = i; j < m; j++)
				a[j][i] *= g;
		}
		else
			for (j = i; j < m; j++)
				a[j][i] = 0.0;
		++a[i][i];
	}
	
	/* diagonalization of the bidigonal form */
	for (k = n - 1; k >= 0; k--)
	{                           /* loop over singlar values */
		for (its = 0; its < 30; its++)
		{                       /* loop over allowed iterations */
			flag = 1;
			for (l = k; l >= 0; l--)
			{                   /* test for splitting */
				nm = l - 1;     /* note that rv1[l] is always zero */
				if (fabs(rv1[l]) + anorm == anorm)
				{
					flag = 0;
					break;
				}
				if (fabs(w[nm]) + anorm == anorm)
					break;
			}
			if (flag)
			{
				c = 0.0;        /* cancellation of rv1[l], if l>1 */
				s = 1.0;
				for (i = l; i <= k; i++)
				{
					f = s * rv1[i];
					if (fabs(f) + anorm != anorm)
					{
						g = w[i];
						h = radius(f, g);
						w[i] = h;
						h = 1.0 / h;
						c = g * h;
						s = (-f * h);
						for (j = 0; j < m; j++)
						{
							y = a[j][nm];
							z = a[j][i];
							a[j][nm] = y * c + z * s;
							a[j][i] = z * c - y * s;
						}
					}
				}
			}
			z = w[k];
			if (l == k)
			{                   /* convergence */
				if (z < 0.0)
				{
					w[k] = -z;
					for (j = 0; j < n; j++)
						v[j][k] = (-v[j][k]);
				}
				break;
			}
			
			x = w[l];           /* shift from bottom 2-by-2 minor */
			nm = k - 1;
			y = w[nm];
			g = rv1[nm];
			h = rv1[k];
			f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
			g = radius(f, 1.0);

			/* next QR transformation */
			f = ((x - z) * (x + z) + h * ((y / (f + Sign(g, f))) - h)) / x;
			c = s = 1.0;
			for (j = l; j <= nm; j++)
			{
				i = j + 1;
				g = rv1[i];
				y = w[i];
				h = s * g;
				g = c * g;
				z = radius(f, h);
				rv1[j] = z;
				c = f / z;
				s = h / z;
				f = x * c + g * s;
				g = g * c - x * s;
				h = y * s;
				y = y * c;
				for (jj = 0; jj < n; jj++)
				{
					x = v[jj][j];
					z = v[jj][i];
					v[jj][j] = x * c + z * s;
					v[jj][i] = z * c - x * s;
				}
				z = radius(f, h);
				w[j] = z;       /* rotation can be arbitrary id z=0 */
				if (z)
				{
					z = 1.0 / z;
					c = f * z;
					s = h * z;
				}
				f = (c * g) + (s * y);
				x = (c * y) - (s * g);
				for (jj = 0; jj < m; jj++)
				{
					y = a[jj][j];
					z = a[jj][i];
					a[jj][j] = y * c + z * s;
					a[jj][i] = z * c - y * s;
				}
			}
			rv1[l] = 0.0;
			rv1[k] = f;
			w[k] = x;
		}
	}
	free((void *) rv1);
}
bool Matrix::SolveCubic(double *res, int &n)
{
	// This function only returns real roots.
	int ii;
	CvMat *coeffs = cvCreateMat(1, 4, CV_64F);
	CvMat *roots = cvCreateMat(1, 3, CV_64F);
	
	Matrix coeff = *this;
	for(ii=0; ii<4; ii++)
		coeffs->data.db[ii] = coeff[ii];

	n = cvSolveCubic(coeffs, roots);
	
	if(n<1 || n>3)
	{
		cvReleaseMat(&coeffs);
		cvReleaseMat(&roots);
		return false;
	}
	else
	{
		for(ii=0; ii<n; ii++)
			res[ii] = roots->data.db[ii];

		cvReleaseMat(&coeffs);
		cvReleaseMat(&roots);

		return true;
	}
}

void Matrix::SolvePoly(double *res)
{
	int ii;
	Matrix coeff = *this;
	int poly_th = (coeff.row)*(coeff.column)-1;

	CvMat *coeffs = cvCreateMat(poly_th+1, 1, CV_64F);
	CvMat *roots = cvCreateMat(poly_th, 1, CV_64FC2);
	
	for(ii=0; ii<poly_th+1; ii++)
		coeffs->data.db[poly_th-ii] = coeff[ii];

	cvSolvePoly(coeffs, roots);
	
	for(ii=0; ii<poly_th; ii++)
	{
		res[2*ii] = roots->data.db[2*ii]; // real part
		res[2*ii+1] = roots->data.db[2*ii+1]; //ima part
	}

	cvReleaseMat(&roots);

	return;
}
Matrix Matrix::Closest_RankN_Approx(int rank_approx, int nrow, int ncol)
{
	int ii, jj;
	double *u = new double [nrow*nrow];
	double *w = new double [nrow*ncol];
	double *v = new double [ncol*ncol];

	Matrix A=*this;
	Matrix U(nrow, nrow), Vt(nrow, ncol), W(ncol, ncol), Approx(nrow, ncol);

	A.SVDcmp(nrow, ncol, u, w, v, CV_SVD_MODIFY_A + CV_SVD_V_T );

	int rank_eliminate = nrow - rank_approx;
	for(ii=0; ii<rank_approx; ii++)
		W[ii+ii*nrow] = w[ii+ii*nrow];

	for(jj=0; jj<nrow; jj++)
		for(ii=0; ii<nrow; ii++)
			U[ii+jj*nrow] = u[ii+jj*nrow];

	for(jj=0; jj<ncol; jj++)
		for(ii=0; ii<ncol; ii++)
			Vt[ii+jj*ncol] = v[ii+jj*ncol];

	Approx = U*W*Vt;

	delete []u;
	delete []w;
	delete []v;

	return Approx;
}

double  Matrix::L2_norm()
{
	double norm = 0.0;
	unsigned int ii,jj;

	Matrix A=*this;

	for(jj=0; jj<row; jj++)
		for(ii=0; ii<column; ii++)
			norm += A[ii+jj*column]*A[ii+jj*column];

	return sqrt(norm);
}

double  Matrix::Linf_norm()
{
	double norm = 0.0;
	unsigned int ii,jj;

	Matrix A=*this;

	for(jj=0; jj<row; jj++)
		for(ii=0; ii<column; ii++)
			if (norm < A[ii+jj*column])
				norm = A[ii+jj*column];

	return norm;
}

Matrix Matrix::Normalize(void)
{
	unsigned int ii,jj;
	double norm=0.0;

	Matrix a=*this;

	for(ii=0;ii<row;ii++)
		for(jj=0;jj<column;jj++)
			norm+=a[ii+jj*column]*a[ii+jj*column];

	norm=sqrt(norm);

	for(ii=0;ii<row;ii++)
		for(jj=0;jj<column;jj++)
			a[ii+jj*column]/=norm;

	return a;
}
// Multiplication by a scalar - multiplies each element
// of the matrix by the Coefficient.
Matrix operator*(int Coefficient, const Matrix& Term)
{
	return (Term * Coefficient);
}

// Multiplication by a scalar - multiplies each element
// of the matrix by the Coefficient.
Matrix operator*(double Coefficient, const Matrix& Term)
{
	return (Term * Coefficient);
}

void Matrix::ConvertedFromVector(double *V, int dimension)
{
	for(int i=0; i<dimension; i++)
	{
		*(matrix+i) = *(V+i);
	}
	return;
}

void Matrix::ConvertedToVector(double *V, int dimension)
{
	for(int i=0; i<dimension; i++)
	{
		*(V+i) = *(matrix+i);
	}
	return;
}
