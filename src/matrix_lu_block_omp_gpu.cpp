//
//  matrix_lu_block_omp.cpp
//  matrix_lu
//
//  Created by Vitaly Koynov on 02/09/20.
//  Copyright © 2020 Sergey Kireev, Vitaly Koynov. All rights reserved.
//  SPDX-License-Identifier: MIT
//
 
#include <math.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include <iostream>
 
 
void time_start(struct timeval *tv) { gettimeofday(tv, NULL); }
double time_stop_start(struct timeval *tv)
{
  struct timeval tv0 = *tv;
  gettimeofday(tv, NULL);
  return (*tv).tv_sec - tv0.tv_sec + 1e-6*((*tv).tv_usec - tv0.tv_usec);
}
double time_stop(struct timeval *tv)
{
  struct timeval tv1;
  gettimeofday(&tv1, NULL);
  return tv1.tv_sec - (*tv).tv_sec + 1e-6*(tv1.tv_usec - (*tv).tv_usec);
}
 
//------------------------------------------
// operations with matrix blocks
//------------------------------------------
 
// in: a[n][n] : lu-decomposition
void proc_lu(const int n,double *a1d)
{
//#pragma omp task depend(inout:a1d)
// Parallelize on target device.
#pragma omp target map(tofrom : a1d[0:n*n])
{
#pragma omp parallel for
  for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++) {
      a1d[j*n+i] /= a1d[i*n+i];
        for (int k = i + 1; k < n; k++)
          a1d[j*n+k] -= a1d[j*n+i]* a1d[i*n+k];
    }
}
}
// in: (*a)[ny][ny], inout: au[ny][nx]
void proc_u(const int ny,const int nx,double *a1d,double *au1d)
{
// Parallelize on target device.
#pragma omp target teams distribute parallel for map(to : a1d[0:ny*nx]) \
  map(tofrom : au1d[0:ny*nx]) thread_limit(128)  
    {
        
    for (int i = 0; i < ny; i++)
      for (int j = i + 1; j < ny; j++)
        for (int k = 0; k < nx; k++)
          au1d[j * ny + k] -= a1d[j * ny + i] * au1d[i * nx + k];
}
}

// in: (*a)[nx][nx], inout: al[ny][nx]
void proc_l(const int ny,const int nx,double *a1d,double *al1d)
{
// Parallelize on target device.
#pragma omp target teams distribute parallel for map(to : a1d[0:ny*nx]) \
  map(tofrom : al1d[0:ny*nx]) thread_limit(128)  
  {
    for (int i = 0; i < nx; i++)
      for (int j = 0; j < ny; j++) {
        al1d[j*ny + i] /= a1d[i*nx+i];
          for (int k = i + 1; k < nx; k++) al1d[j*ny+k] -= al1d[j*ny+i] * a1d[i*nx +k];
      }
}
}

// in: al[ny][n],au[n][nx], inout: ag[ny][nx] : ag -= al*au
void proc_g(const int ny,const int nx,const int n,double *al1d,double *au1d,double *ag1d)
{
// Parallelize on target device.
#pragma omp target teams distribute parallel for map(to : al1d[0:ny*n], au1d[0:n*nx]) \
  map(tofrom : ag1d[0:ny*nx]) thread_limit(128)
  {
    for (int j = 0; j < ny; j++)
      for (int i = 0; i < n; i++)
        for (int k = 0; k < nx; k++) 
        {
          ag1d[j*ny+k] -= al1d[j*ny+i] * au1d[i*nx+k];
        }
  }
}

// in: a[ny][n],b[n][nx], inout: c[ny][nx] : c += a*b
void proc_mm(const int ny, const int nx,const int n,double *a1d,double *b1d,double *c1d)
{
#pragma omp task depend(in:a1d,b1d) depend(inout:c1d)
{
  auto a = (double (*)[ny][n]) a1d;
  auto b = (double (*)[n][nx]) b1d;
  auto c = (double (*)[ny][nx]) c1d;
  for (int j = 0; j < ny; j++)
    for (int i = 0; i < n; i++)
      for (int k = 0; k < nx; k++) 
        (*c)[j][k] += (*a)[j][i] * (*b)[i][k];
}
}

// out: a[ny][nx] : a = value (fill with value)
void proc_fill(int ny,int nx,double *a,double value)
{
  for (int i = 0; i < nx * ny; i++)
    a[i] = value;
}

// in: a[ny][nx], out: b[ny][nx] : b = a (copy matrix)
void proc_copy(const int ny,const int nx,double *a,double *b)
{
#pragma omp task depend(in:a) depend(out:b)
  {
    for (int i = 0; i < nx * ny; i++) 
      b[i] = a[i];
  }
}

// in: a[ny][nx], out: b[ny][nx] : b = lower(a) : get lower triangular of square matrix
void proc_get_lower(const int n,double *a1d,double *b1d)
{
#pragma omp task depend(in:a1d) depend(out:b1d)
{
  auto a = (double (*)[n][n]) a1d;
  auto b = (double (*)[n][n]) b1d;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++) 
      (*b)[i][j] = (*a)[i][j];
    
    (*b)[i][i] = 1.0;
    for (int j = i + 1; j < n; j++) 
      (*b)[i][j] = 0.0;
  }
}
}

// in: a[ny][nx], out: b[ny][nx] : b = upper_with_diagonal(a) : get upper triangular of square matrix
void proc_get_upper(const int n,double *a1d,double *b1d)
{
#pragma omp task depend(in:a1d) depend(out:b1d)
{
  auto a = (double (*)[n][n]) a1d;
  auto b = (double (*)[n][n]) b1d;
  
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < i; j++) (*b)[i][j] = 0.0;
      for (int j = i; j < n; j++) (*b)[i][j] = (*a)[i][j];
  }
}
}

double proc_delta(const int ny,const int nx,double *a,double *b)
{
  double delta;
  delta = 0.0;
    for (int i = 0; i < ny * nx; i++) {
      double d = fabs(a[i] - b[i]);
      delta = delta >= d ? delta : d;
    }
  return delta;
}
 
//------------------------------------------
// operations with square blocked matrices
//------------------------------------------
 
// allocate 2D array (nb*nb) of 2D blocks (bs*bs) of double
double **allocate_blocked_matrix(int nb,int bs)
{
  double **a = new double*[nb*nb];//(double**)malloc(nb*nb*sizeof(double *));
  double *data = new double[nb*nb*bs*bs];
  for (int i=0;i<nb;i++)
    for (int j=0;j<nb;j++) {
      int k = i*nb + j;
      a[k] = &data[k * bs*bs];
    }
  return a;
}

// free blocked array
void free_blocked_matrix(double **a)
{
  delete (*a);
  delete (a);
}
 
// LU-decomposition of square blocked matrix
void LU_decomposition(const int nb, const int bs,double **a1d)
{
  auto a = (double*(*)[nb][nb]) a1d;

  for (int i=0;i<nb;i++)
  {
    proc_lu(bs,(*a)[i][i]);
    for (int k=i+1;k<nb;k++) proc_u(bs,bs,(*a)[i][i],(*a)[i][k]);
    for (int j=i+1;j<nb;j++)
    {
      proc_l(bs,bs,(*a)[i][i],(*a)[j][i]);
      for (int k=i+1;k<nb;k++) proc_g(bs,bs,bs,(*a)[j][i],(*a)[i][k],(*a)[j][k]);
    }
  }
}
 
 
// multiplication of square blocked matrices: c += a*b
void matrix_multiplication(int nb,int bs,double **a1d,double **b1d,double **c1d)
{
  auto a = (double *(*)[nb][nb]) a1d;
  auto b = (double *(*)[nb][nb]) b1d;
  auto c = (double *(*)[nb][nb]) c1d;

  for (int i = 0; i < nb; i++)
    for (int k = 0; k < nb; k++)
      for (int j = 0; j < nb; j++) 
        proc_mm(bs, bs, bs, (*a)[i][k], (*b)[k][j], (*c)[i][j]);
}
 
// get lower triangle of blocked square matrix: al = lower(a)
void get_lower_matrix(int nb,int bs,double **a1d,double **al1d)
{
  auto a = (double*(*)[nb][nb]) a1d;
  auto al = (double*(*)[nb][nb]) al1d;
  for (int i=0;i<nb;i++)
  {
    for (int j=0;j<i;j++) proc_copy(bs,bs,(*a)[i][j],(*al)[i][j]);
    proc_get_lower(bs,(*a)[i][i],(*al)[i][i]);
    for (int j=i+1;j<nb;j++) proc_fill(bs,bs,(*al)[i][j],0.0);
  }
}
 
// get upper triangle with diagonal of blocked square matrix: al = upper_with_diagonal(a)
void get_upper_matrix(int nb,int bs,double **a1d,double **au1d)
{
  auto a= (double*(*)[nb][nb]) a1d;
  auto au = (double*(*)[nb][nb]) au1d;
  for (int i=0;i<nb;i++) {
    for (int j=0;j<i;j++)
      proc_fill(bs,bs,(*au)[i][j],0.0);

    proc_get_upper(bs,(*a)[i][i],(*au)[i][i]);
    for (int j=i+1;j<nb;j++)
      proc_copy(bs,bs,(*a)[i][j],(*au)[i][j]);
  }
}

// fill square blocked matrix with given value
void fill_matrix(int nb,int bs,double **a,double value)
{
  for (int i=0;i<nb*nb;i++)
    proc_fill(bs,bs,a[i],value);
}

// copy square blocked matrix : b = a
void copy_matrix(int nb,int bs,double **a,double **b)
{
  for (int i=0;i<nb*nb;i++)
    proc_copy(bs,bs,a[i],b[i]);
}
 
double matrix_delta(int nb,int bs,double **a,double **b)
{
  double delta = 0.0;
 
#pragma omp parallel for reduction(max:delta)
  for (int i = 0; i < nb * nb; i++)
  {
    double d = proc_delta(bs, bs, a[i], b[i]);
    delta = delta >= d ? delta : d;
  }
  return delta;
}
 
void print_matrix(int nb,int bs,double **a)
{
  for (int ib=0;ib<nb;ib++)
    for (int i=0;i<bs;i++)
    {
      for (int jb=0;jb<nb;jb++)
        for (int j=0;j<bs;j++)
          std::cout << a[ib*nb+jb][i*bs+j] << " ";
      printf("\n");
    }
}
 
int main()
{
  const int nb = 1; // number of blocks: nb*nb
  const int bs = 64; // block size: bs*bs
  
  std::cout << "bs: " << bs << std::endl;
  
  double **aS = allocate_blocked_matrix(nb, bs); // source matrix
  double **aLU = allocate_blocked_matrix(nb, bs); // LU matrix
  
  // set initial matrix
  fill_matrix(nb, bs, aS, 1.0);
  for (int i = 0; i < nb; i++)
  for (int j = 0; j < bs; j++)
  aS[i * nb + i][j * bs + j] = nb * bs;

  copy_matrix(nb, bs, aS, aLU);

  double t1;
  double delta = 0;
  struct timeval tv;
  
  time_start(&tv);
  std::cout << "omp_get_num_devices(): " << omp_get_num_devices() << std::endl;
  LU_decomposition(nb, bs, aLU);
  t1 = time_stop_start(&tv);
  
  delta = matrix_delta(nb, bs, aLU, aS);

  std::cout << "Time: " << t1 << std::endl;
  std::cout << "Delta: " << delta << std::endl;
  
  free_blocked_matrix(aS);
  free_blocked_matrix(aLU);

  return 0;
}
