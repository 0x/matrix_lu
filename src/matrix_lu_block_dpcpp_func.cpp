//
//  matrix_lu_block_dpcpp.cpp
//  matrix_lu
//
//  Created by Vitaly Koynov on 02/09/20.
//  Copyright © 2020 Sergey Kireev, Vitaly Koynov. All rights reserved.
//  SPDX-License-Identifier: MIT
//

#include <CL/sycl.hpp>
#include <iostream>
#include <limits>
#include "dpc_common.hpp"
 
using namespace std;
using namespace sycl;

//CustomDeviceSelector d_selector;  // Select default device (must be cpu)
cpu_selector d_selector;  // Select default cpu device
//gpu_selector d_selector;   // Select default gpu device
 
// Initialize the device queue with the default selector, cpu_selector or gpu_selector. 
// The device queue is used to enqueue kernels. 
// It encapsulates all states needed for execution.
queue q(d_selector, dpc_common::exception_handler);
 
// in: a[n][n] : lu-decomposition
void proc_lu(const int n, double *a1d)
{
  buffer a(a1d, range(n, n));
  
  for (int i = 0; i < n; i++)
  {
    // Division step 
    // Submit command group to queue to division step LU-decomposition matrix S (source)
    q.submit([&](handler &h) {

      auto A = a.get_access<access::mode::read_write>(h);
      
      h.parallel_for(range(n - i - 1), [=](id<1> index) {
        int j = i + 1 + index;
      	A[j][i] /= A[i][i];
      });
    });
    
    // Elimination step
    // Submit command group to queue to elimination step LU-decomposition matrix S (source)
    q.submit([&](handler &h) {

      auto A = a.get_access<access::mode::read_write>(h);

      h.parallel_for(range(n - i - 1), [=](id<1> index) {
        int j = i + 1 + index;
        for (int k = i + 1; k < n; k++)
        {
          A[j][k] -= A[j][i] * A[i][k];
        }
      });
    });
  }
}

 // in: (*a)[ny][ny], inout: au[ny][nx]
void proc_u(const int ny, const int nx, double *a1d, double *au1d)
{
  buffer a(a1d, range(ny, ny));
  buffer au(au1d, range(ny, nx));

  q.submit([&](handler &h) {
    // Read from al and au, write to ag
    auto A = a.get_access<access::mode::read>(h);
    auto AU = au.get_access<access::mode::read_write>(h);
    
	  // Execute kernel.
    h.parallel_for(range(ny, nx), [=](id<2> index) {
      // Get global position in Y direction.
      int row = index[0];
      // Get global position in X direction.
      int col = index[1];

      // Compute the result of one element of AL
      for (int j = row + 1; j < ny; j++) {
	      AU[j][col] -=  A[j][row] * A[row][col];
	    }
    });
  });
}

// in: (*a)[nx][nx], inout: al[ny][nx]
void proc_l(const int ny, const int nx, double *a1d, double *al1d)
{
	buffer a(a1d, range(nx, nx));
	buffer al(al1d, range(ny, nx));
  
  q.submit([&](handler &h) {
    // Read from al and au, write to ag
    auto A = a.get_access<access::mode::read>(h);
    auto AL = al.get_access<access::mode::read_write>(h);
    
	  // Execute kernel.
    h.parallel_for(range(nx, ny), [=](id<2> index) {
      // Get global position in Y direction.
      int row = index[0];
      // Get global position in X direction.
      int col = index[1];

      // Compute the result of one element of AL
      AL[col][row] /= A[row][row];
    });
  });
  q.submit([&](handler &h) {
    // Read from al and au, write to ag
    auto A = a.get_access<access::mode::read>(h);
    auto AL = al.get_access<access::mode::read_write>(h);
    
	  // Execute kernel.
    h.parallel_for(range(nx, ny), [=](id<2> index) {
      // Get global position in Y direction.
      int row = index[0];
      // Get global position in X direction.
      int col = index[1];

      // Compute the result of one element of AL
      for (int k = row + 1; k < nx; k++) {
	      AL[col][k] -=  AL[col][row] * A[row][k];
	    }
    });
  });
}

// in: al[ny][n],au[n][nx], inout: ag[ny][nx] : ag -= al*au
void proc_g(const int ny, const int nx, const int n, double *al1d, double *au1d, double *ag1d)
{
  buffer al((al1d), range(ny, n));
  buffer au((au1d), range(n, nx));
  buffer ag((ag1d), range(ny, nx));
  
  q.submit([&](handler &h) {
    // Read from al and au, write to ag
    auto AL = al.get_access<access::mode::read>(h);
    auto AU = au.get_access<access::mode::read>(h);
    auto AG = ag.get_access<access::mode::read_write>(h);

    // Execute kernel.
    h.parallel_for(range(ny, nx), [=](id<2> index) {
      // Get global position in Y direction.
      int row = index[0];
      // Get global position in X direction.
      int col = index[1];

      // Compute the result of one element of AG
      for (int k = 0; k < n; k++) {
        AG[row][col] -= AL[row][k] * AU[k][col];
      }
    });
  });
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
    delta = (delta >= d) ? delta : d;
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
    delta = (delta >= d) ? delta : d;
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
  const int nb = 16;  // Number of blocks: nb*nb
  const int bs = 128;  // Block size: bs*bs
 
  cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";
    
  double **aS = allocate_blocked_matrix(nb, bs);  // Source matrix
  double **aLU = allocate_blocked_matrix(nb, bs);
  
  cout << "Problem size: S(" << nb * bs << "," << nb * bs << ")\n";
  cout << "Block size: (" << bs << "," << bs << ")\n";
  
  // Set initial matrix
  fill_matrix(nb, bs, aS, 1.0);
  for (int i = 0; i < nb; i++)
      for (int j = 0; j < bs; j++)
          aS[i * nb + i][j * bs + j] = nb * bs;

  copy_matrix(nb, bs, aS, aLU);
  
  dpc_common::TimeInterval matrixLUBlock;
  LU_decomposition(nb, bs, aLU);
  cout << "Time matrixLUBlock: " << matrixLUBlock.Elapsed() << std::endl;
  
  cout <<"delta: " << matrix_delta(nb, bs, aLU, aS) << std::endl;
  
  int result;
  cout << "Result of matrix LU-decomposition using DPC++: "; 
  //result = VerifyResult(nb, bs, aLU);
  
  free_blocked_matrix(aS);
  free_blocked_matrix(aLU);

  return 0;
}
