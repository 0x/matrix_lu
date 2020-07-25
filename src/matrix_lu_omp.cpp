//
//  matrix_lu_omp.cpp
//  matrix_lu
//
//  Created by Vitaly Koynov on 02/09/20.
//  Copyright © 2020 Vitaly Koynov. All rights reserved.
//  SPDX-License-Identifier: MIT
//

#include <math.h>
#include <omp.h>
#include <iostream>
#include <limits>
#include <chrono>
#include <stdlib.h>
#include "dpc_common.hpp"

using namespace std;

// Matrix size constants.
constexpr int N = 8192;

/**
 * Each element of the product matrix c[i][j] is computed from a unique row and
 * column of the factor matrices, a[i][k] and b[k][j]
 */
double S[N][N];

/**
 * Perform matrix multiplication on CPU with OpenMP.
 */
void MatrixLuOpenMpCpu(double (*a)[N]);

/**
 * Perform matrix multiplication on host to verify results from OpenMP.
 */
int VerifyResult(double (*LU_back)[N]);

int main(void) {
  int Result1;

  cout << "Problem size: S(" << N << "," << N << ")\n";

  cout << "Running on " << omp_get_num_devices() << " device(s)\n";
  cout << "The default device id: " << omp_get_default_device() << "\n";

  dpc_common::TimeInterval matrixLuOpenMpCpu;
  MatrixLuOpenMpCpu(S);
  cout << "Time matrixMulOpenMpCpu: " << matrixLuOpenMpCpu.Elapsed() << std::endl;
  
  cout << "Result of matrix lu-decomposition using OpenMP: ";
  Result1 = VerifyResult(S);

  return Result1;
}

void MatrixLuOpenMpCpu(double (*a)[N]) {

  // Each element of matrix a is 1.
  // Diag elements is N.
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      a[i][j] = (i== j)?N:1.0;
    }
  }
  
  // LU-decomposition
  for (int i = 0; i < N; i++)
  {
  // Parallelize by row. The threads don't need to synchronize at
  // loop end, so "nowait" can be used.
#pragma omp parallel for
    for (int j = i + 1; j < N; j++)
    {
      a[j][i] /= a[i][i];
    }
#pragma omp parallel for
    for (int j = i + 1; j < N; j++)
    {
      for (int k = i + 1; k < N; k++)
      {
        a[j][k] -= a[j][i] * a[i][k];
      }
    }
  }
}



bool ValueSame(double a, double b) {
  return fabs(a - b) < numeric_limits<double>::epsilon();
}

int VerifyResult(double (*c_back)[N]) {
  // Check that the results are correct by comparing with host computing.

  // 2D arrays on host side.
  double(*s_host)[N] = new double[N][N];

  // Each element of matrix a is 1.
  // Diag elements is N.
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      s_host[i][j] = (i== j)?N:1.0;
    }
  }
  
  // LU-decomposition
  for (int i = 0; i < N; i++)
  {
    for (int j = i + 1; j < N; j++)
    {
      s_host[j][i] /= s_host[i][i];
    }
    for (int j = i + 1; j < N; j++)
    {
      for (int k = i + 1; k < N; k++)
      {
        s_host[j][k] -= s_host[j][i] * s_host[i][k];
      }
    }
  }

  bool mismatch_found = false;

  // Compare host side results with the result buffer from device side: print
  // mismatched data 5 times only.
  int print_count = 0;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (!ValueSame(c_back[i][j], s_host[i][j])) {
        cout << "Fail - The result is incorrect for element: [" << i << ", "
             << j << "], expected: " << s_host[i][j]
             << ", but found: " << c_back[i][j] << "\n";
        mismatch_found = true;
        print_count++;
        if (print_count == 5) break;
      }
    }

    if (print_count == 5) break;
  }

  delete[] s_host;

  if (!mismatch_found) {
    cout << "Success - The results are correct!\n";
    return 0;
  } else {
    cout << "Fail - The results mismatch!\n";
    return -1;
  }
}
