//
//  matrix_lu_dpcpp.cpp
//  matrix_lu
//
//  Created by Vitaly Koynov on 02/09/20.
//  Copyright © 2020 Vitaly Koynov. All rights reserved.
//  SPDX-License-Identifier: MIT
//

#include <CL/sycl.hpp>
#include <iostream>
#include <limits>
#include <iomanip>
#include "dpc_common.hpp"

using namespace std;
using namespace sycl;

/**
 * Each element of the product matrix c[i][j] is computed from a unique row and
 * column of the factor matrices, a[i][k] and b[k][j]
 */

// Matrix size constants.
constexpr int N = 8192;

/**
 * Perform matrix multiplication on host to verify results from device.
 */
int VerifyResult(double (*LU_back)[N]);

int main() {
  // Host memory buffer that device will write data back before destruction.
  double(*LU_back)[N] = new double[N][N];  // LU matrix

  // Intialize LU_back
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) LU_back[i][j] = 0.0;

  dpc_common::TimeInterval matrixLUDPCPP;

  
  //CustomDeviceSelector d_selector;  // Select default device (must be cpu)
  cpu_selector d_selector;  // Select default cpu device
  //gpu_selector d_selector;   // Select default gpu device
  
  // Initialize the device queue with the default selector, cpu_selector or gpu_selector. 
  // The device queue is used to enqueue kernels. 
  // It encapsulates all states needed for execution.
  try {
    
    queue q(d_selector, dpc_common::exception_handler);

    cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";

    // Create 2D buffers for matrices, buffer LU is bound with host memory LU_back
    buffer S(reinterpret_cast<double *>(LU_back), range(N, N));

    cout << "Problem size: S(" << N << "," << N << ")\n";

    // Submit command group to queue to initialize matrix S (source)
    q.submit([&](handler &h) {
      // Get write only access to the buffer on a device.
      auto accessor = S.get_access<access::mode::write>(h);
      // Execute kernel.
      h.parallel_for(range(N, N), [=](id<2> index) {
        // Each element of matrix S is 1.
        // Diag elements is N.
        accessor[index] = (index[1] == index[0])?N:1.0;
      });
    });
    
    // LU-decomposition
    for (int i = 0; i < N; i++)
    {
      // Division step 
      // Submit command group to queue to division step LU-decomposition matrix S (source)
      q.submit([&](handler &h) {

      auto accessor = S.get_access<access::mode::write>(h);
      
      h.parallel_for(range(N - i - 1), [=](id<1> idx) {
        int j = i + 1 + idx;
      	accessor[j][i] /= accessor[i][i];
      });
      });
      
      // Elimination step
      // Submit command group to queue to elimination step LU-decomposition matrix S (source)
      q.submit([&](handler &h) {

      auto accessor = S.get_access<access::mode::write>(h);

      h.parallel_for(range(N - i - 1), [=](id<1> idx) {
        int j = i + 1 + idx;
        for (int k = i + 1; k < N; k++)
        {
          accessor[j][k] -= accessor[j][i] * accessor[i][k];
        }
      });
      });
    }
    
  } catch (sycl::exception const &e) {
    cout << "An exception is caught while multiplying matrices.\n";
    terminate();
  }

  int result;
  cout << "Result of matrix LU-decomposition using DPC++: ";
  result = VerifyResult(LU_back);
  
  cout << "Time matrixLUDPCPP: "  << matrixLUDPCPP.Elapsed() << std::endl;
  
  delete[] LU_back;
  return result;
}

bool ValueSame(double a, double b) {
  return fabs(a - b) < numeric_limits<double>::epsilon();
}

int VerifyResult(double (*c_back)[N]) {
  // Check that the results are correct by comparing with host computing.

  // 2D arrays on host side.
  double(*s_host)[N] = new double[N][N];

  dpc_common::TimeInterval matrixLU;
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
    // Division step 
    for (int j = i + 1; j < N; j++)
    {
      s_host[j][i] /= s_host[i][i];
    }
    
    // Elimination step
    for (int j = i + 1; j < N; j++)
    {
      for (int k = i + 1; k < N; k++)
      {
        s_host[j][k] -= s_host[j][i] * s_host[i][k];
      }
    }
  }
  cout << "Time matrixLU: " << matrixLU.Elapsed() << std::endl;
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

