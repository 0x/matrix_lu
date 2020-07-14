//
//  matrix_lu_block.cpp
//  matrix_lu
//

#include <iostream>
#include "dpc_common.hpp"
 
using namespace std;
using namespace sycl;
 
// in: a[n][n] : lu-decomposition
void proc_lu(const int n, double *a1d)
{
  auto a = (double(*)[n][n]) a1d;
  
  for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++)
    {
      (*a)[j][i] /= (*a)[i][i];
      for (int k = i + 1; k < n; k++) 
      	(*a)[j][k] -= (*a)[j][i] * (*a)[i][k];
    }
}

 // in: (*a)[ny][ny], inout: au[ny][nx]
void proc_u(const int ny, const int nx, double *a1d, double *au1d)
{
  auto a = (double(*)[ny][ny]) a1d;
  auto au = (double(*)[ny][nx]) au1d;
  
  for (int i = 0; i < ny; i++)
    for (int j = i + 1; j < ny; j++)
      for (int k = 0; k < nx; k++)
        (*au)[j][k] -= (*a)[j][i] * (*au)[i][k];
}

// in: (*a)[nx][nx], inout: al[ny][nx]
void proc_l(const int ny, const int nx, double *a1d, double *al1d)
{
  auto a = (double(*)[nx][nx]) a1d;
  auto al = (double(*)[ny][nx]) al1d;
  
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
    {
      (*al)[j][i] /= (*a)[i][i];
      for (int k = i + 1; k < nx; k++) 
      	(*al)[j][k] -=  (*al)[j][i] * (*a)[i][k];
    }
}

// in: al[ny][n],au[n][nx], inout: ag[ny][nx] : ag -= al*au
void proc_g(const int ny, const int nx, const int n, double *al1d, double *au1d, double *ag1d)
{
  auto al = (double(*)[ny][n ]) al1d;
  auto au = (double(*)[n ][nx]) au1d;
  auto ag = (double(*)[ny][nx]) ag1d;
  
  for (int j = 0; j < ny; j++)
    for (int i = 0; i < n; i++)
      for (int k = 0; k < nx; k++) 
      	(*ag)[j][k] -= (*al)[j][i] * (*au)[i][k];
}

// in: a[ny][n],b[n][nx], inout: c[ny][nx] : c += a*b
void proc_mm(int ny, int nx, int n, double *a1d, double *b1d, double *c1d)
{
  auto a = (double(*)[ny][n ]) a1d;
  auto b = (double(*)[n ][nx]) b1d;
  auto c = (double(*)[ny][nx]) c1d;
  for (int j = 0; j < ny; j++)
    for (int i = 0; i < n; i++)
      for (int k = 0; k < nx; k++) 
      	(*c)[j][k] += (*a)[j][i] * (*b)[i][k];
}

// out: a[ny][nx] : a = value (fill with value)
void proc_fill(int ny, int nx, double *a, double value)
{
  for (int i = 0; i < nx * ny; i++) 
  	a[i] = value;
}

// in: a[ny][nx], out: b[ny][nx] : b = a (copy matrix)
void proc_copy(int ny, int nx, double *a, double *b)
{
  for (int i = 0; i < nx * ny; i++)
  	b[i] = a[i];
}

// in: a[ny][nx], out: b[ny][nx] : b = lower(a) : get lower triangular of square matrix
void proc_get_lower(int n, double *a1d, double *b1d)
{
  auto a = (double(*)[n][n]) a1d;
  auto b = (double(*)[n][n]) b1d;
  
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j <= i; j++)
    	(*b)[i][j] = (*a)[i][j];
    	
    (*b)[i][i] = 1.0;
    for (int j = i + 1; j < n; j++) 
    	(*b)[i][j] = 0.0;
  }
}

// in: a[ny][nx], out: b[ny][nx] : b = upper_with_diagonal(a) : get upper triangular of square matrix
void proc_get_upper(int n, double *a1d, double *b1d)
{
  auto a = (double(*)[n][n]) a1d;
  auto b = (double(*)[n][n]) b1d;
  
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < i; j++) 
    	(*b)[i][j] = 0.0;
    for (int j = i; j < n; j++) 
    	(*b)[i][j] = (*a)[i][j];
  }
}

double proc_delta(int ny, int nx, double *a, double *b)
{
  double delta = 0.0;
  
  for (int i = 0; i < ny * nx; i++)
  {
    double d = fabs(a[i] - b[i]);
    delta = delta >= d ? delta : d;
  }
  return delta;
}
 
/**
 * Operations with square blocked matrices
 */
// allocate 2D array (nb*nb) of 2D blocks (bs*bs) of double
double **allocate_blocked_matrix(int nb, int bs)
{
  double **a = (double**)malloc(nb*nb*sizeof(double *));
  double *data = (double*)malloc(nb*nb*bs*bs*sizeof(double));
  
  for (int i = 0; i < nb; i++)
    for (int j = 0; j < nb; j++) {
      int k = i * nb + j;
      a[k] = &data[k * bs * bs];
    }
  return a;
}

// free blocked array
void free_blocked_matrix(double **a)
{
  free(*a);
  free(a);
}

// LU-decompositiob of square blocked matrix
void LU_decomposition(int nb, int bs, double **a1d)
{
  auto a = (double*(*)[nb][nb]) a1d;

  for (int i = 0; i < nb; i++)
  {
  	//
    	proc_lu(bs, (*a)[i][i]);
   
    for (int k = i + 1; k < nb; k++) 
    	// 
    	proc_u(bs, bs, (*a)[i][i], (*a)[i][k]);
    
    for (int j = i + 1; j < nb; j++)
    {
    	//
      	proc_l(bs, bs, (*a)[i][i], (*a)[j][i]);
      
      for (int k = i + 1; k < nb; k++) 
	// 
      	proc_g(bs, bs, bs, (*a)[j][i], (*a)[i][k], (*a)[j][k]);
    }
  }
}

// multiplication of square blocked matrices: c += a*b
void matrix_multiplication(int nb, int bs, double **a1d, double **b1d, double **c1d)
{
  auto a = (double*(*)[nb][nb]) a1d;
  auto b = (double*(*)[nb][nb]) b1d;
  auto c = (double*(*)[nb][nb]) c1d;
  
  for (int i = 0; i < nb; i++)
    for (int k = 0; k < nb; k++)
      for (int j = 0; j < nb; j++) 
      	proc_mm(bs, bs, bs, (*a)[i][k], (*b)[k][j], (*c)[i][j]);
}
 
// get lower triangle of blocked square matrix: al = lower(a)
void get_lower_matrix(int nb, int bs, double **a1d, double **al1d)
{
  auto a = (double*(*)[nb][nb]) a1d;
  auto al = (double*(*)[nb][nb]) al1d;
  
  for (int i = 0; i < nb; i++)
  {
    for (int j = 0; j < i; j++) 
    	proc_copy(bs, bs, (*a)[i][j], (*al)[i][j]);
  
    proc_get_lower(bs, (*a)[i][i], (*al)[i][i]);
    for (int j = i + 1; j < nb; j++) 
    	proc_fill(bs, bs, (*al)[i][j], 0.0);
  }
}
 
// get upper triangle with diagonal of blocked square matrix: al = upper_with_diagonal(a)
void get_upper_matrix(int nb, int bs, double **a1d, double **au1d)
{
  auto a = (double*(*)[nb][nb]) a1d;
  auto au = (double*(*)[nb][nb]) au1d;
  
  for (int i = 0; i < nb; i++) {
    for (int j = 0; j < i; j++)
      proc_fill(bs, bs, (*au)[i][j], 0.0);
      
    proc_get_upper(bs, (*a)[i][i], (*au)[i][i]);
    for (int j = i + 1; j < nb; j++)
      proc_copy(bs, bs, (*a)[i][j], (*au)[i][j]);
  }
}

// fill square blocked matrix with given value
void fill_matrix(int nb, int bs, double **a, double value)
{
  for (int i = 0; i < nb * nb; i++)
    proc_fill(bs, bs, a[i], value);
}

// copy square blocked matrix : b = a
void copy_matrix(int nb, int bs, double **a, double **b)
{
  for (int i = 0; i < nb * nb; i++)
    proc_copy(bs, bs, a[i], b[i]);
}
 
double matrix_delta(int nb, int bs, double **a, double **b)
{
  double delta = 0.0;
  for (int i = 0; i < nb * nb; i++)
  {
    double d = proc_delta(bs, bs, a[i], b[i]);
    delta = delta >= d ? delta : d;
  }
  return delta;
}
 
void print_matrix(int nb, int bs, double **a)
{
  for (int ib = 0; ib < nb; ib++)
    for (int i = 0; i < bs; i++)
    {
      for (int jb = 0; jb < nb; jb++)
        for (int j = 0; j < bs; j++)
          cout << a[ib*nb+jb][i*bs+j] << " ";
      cout << "\n";
    }
}
 

int main()
{
  const int nb = 3;  // Number of blocks: nb*nb
  const int bs = 3;  // Block size: bs*bs
  
  cout << "Problem size: S(" << nb * bs << "," << nb * bs << ")\n";
  cout << "Block size: (" << bs << "," << bs << ")\n";
  
  double **aS = allocate_blocked_matrix(nb, bs);  // Source matrix
  double **aLU= allocate_blocked_matrix(nb, bs);  // LU matrix
  double **aL = allocate_blocked_matrix(nb, bs);  // L matrix
  double **aU = allocate_blocked_matrix(nb, bs);  // U matrix
  double **aM = allocate_blocked_matrix(nb, bs);  // Matrix after multiplication
 
  // Set initial matrix
  fill_matrix(nb, bs, aS, 1.0);
  for (int i = 0; i < nb; i++)
    for (int j = 0; j < bs; j++)
      aS[i * nb + i][j * bs + j] = nb * bs;
 
  copy_matrix(nb, bs, aS, aLU);
 
  dpc_common::TimeInterval matrixLUBlock;
  LU_decomposition(nb, bs, aLU);
  cout << "Time matrixLUBlock: " << matrixLUBlock.Elapsed() << std::endl;
 
  get_lower_matrix(nb, bs, aLU, aL);
  get_upper_matrix(nb, bs, aLU, aU);
 
  dpc_common::TimeInterval matrixMul;
  matrix_multiplication(nb, bs, aL, aU, aM);
  cout << "Time matrixMul: " << matrixMul.Elapsed() << std::endl;
 
  //cout << "Source\n"; print_matrix(nb, bs, aS);
  cout << "LU\n"; print_matrix(nb, bs, aLU);
  //cout << "L\n"; print_matrix(nb, bs, aL);
  //cout << "U\n"; print_matrix(nb, bs, aU);
  //cout << "L*U\n"; print_matrix(nb, bs, aM);
 
  free_blocked_matrix(aS);
  free_blocked_matrix(aLU);
  free_blocked_matrix(aL);
  free_blocked_matrix(aU);
  free_blocked_matrix(aM);
 
  return 0;
}
