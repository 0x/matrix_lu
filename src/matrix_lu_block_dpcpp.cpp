//
//  matrix_lu_block_dpcpp.cpp
//  matrix_lu
//

#include <CL/sycl.hpp>
#include <iostream>
#include <limits>
#include "dpc_common.hpp"
 
using namespace std;
using namespace sycl;
 
constexpr int N = 9;

// in: a[n][n] : lu-decomposition
void proc_lu(const int n, double *a1d)
{
  auto a = (double(*)[n][n]) a1d;
  // 
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
 
bool ValueSame(double a, double b) {
  return fabs(a - b) < numeric_limits<double>::epsilon();
}

int VerifyResult(int nb, int bs, double **c_back) {
  // Check that the results are correct by comparing with host computing.

  // 2D arrays on host side.
  double(*s_host)[N] = new double[N][N];

  // Each element of matrix a is 1.
  // Diag elements is N.
  for (int i = 0; i < nb*bs; i++)
  {
    for (int j = 0; j < nb*bs; j++)
    {
      s_host[i][j] = (i== j)?(nb*bs):1.0;
    }
  }
  
  // LU-decomposition
  for (int i = 0; i < nb*bs; i++)
  {
    // Division step 
    for (int j = i + 1; j < nb*bs; j++)
    {
      s_host[j][i] /= s_host[i][i];
    }
    
    // Elimination step
    for (int j = i + 1; j < nb*bs; j++)
    {
      for (int k = i + 1; k < nb*bs; k++)
      {
        s_host[j][k] -= s_host[j][i] * s_host[i][k];
      }
    }
  }

  bool mismatch_found = false;

  // Compare host side results with the result buffer from device side: print
  // mismatched data 5 times only.
  int print_count = 0;
  for (int ib = 0; ib < nb; ib++){
    for (int i = 0; i < bs; i++)
    {
      for (int jb = 0; jb < nb; jb++)
        for (int j = 0; j < bs; j++) {
          if (ValueSame(c_back[ib*nb+jb][i*bs+j], s_host[ib*nb+jb][i*bs+j])) {
        	cout << "Fail - The result is incorrect for element: [" << i << ", "
             	<< j << "], expected: " << s_host[ib*nb+jb][i*bs+j]
             	<< ", but found: " << c_back[ib*nb+jb][i*bs+j] << "\n";
       		mismatch_found = true;
        	print_count++;
        	if (print_count == 5) break;
      		}  
        }
        if (print_count == 5) break;
    }

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

int main()
{
  const int nb = 3;  // Number of blocks: nb*nb
  const int bs = 3;  // Block size: bs*bs
  
  queue q(default_selector{}, dpc_common::exception_handler);
  cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";
    
  double **aS = allocate_blocked_matrix(nb, bs);  // Source matrix
    
  cout << "Problem size: S(" << nb * bs << "," << nb * bs << ")\n";
  cout << "Block size: (" << bs << "," << bs << ")\n";
  
  dpc_common::TimeInterval matrixLUBlock;
  
  auto a = (double*(*)[nb][nb]) aS;
  
  // init
  for (int i = 0; i < nb * nb; i++)
  {
    auto a = aS[i];
    
    for (int i = 0; i < bs * bs; i++) 
  	  a[i] = 1.0;
  }
     
  for (int i = 0; i < nb; i++)
    for (int j = 0; j < bs; j++)
      aS[i * nb + i][j * bs + j] = nb * bs;
 

  


  for (int i = 0; i < nb; i++)
  {
  	//proc_lu(bs, (*a)[i][i]);
    buffer aa(reinterpret_cast<double *>((*a)[i][i]), range(bs, bs));
    
    for (int i = 0; i < bs; i++)
    {
      // Division step 
      // Submit command group to queue to division step LU-decomposition matrix S (source)
      q.submit([&](handler &h) {

        auto accessor = aa.get_access<access::mode::read_write>(h);
        
        h.parallel_for(range(bs - i + 1), [=](id<1> idx) {
          int j = i + 1 + idx;
        	accessor[j][i] /= accessor[i][i];
        });
      });
      
      // Elimination step
      // Submit command group to queue to elimination step LU-decomposition matrix S (source)
      q.submit([&](handler &h) {

        auto accessor = aa.get_access<access::mode::read_write>(h);

        h.parallel_for(range(bs - i + 1), [=](id<1> idx) {
          int j = i + 1 + idx;
          for (int k = i + 1; k < bs; k++)
          {
            accessor[j][k] -= accessor[j][i] * accessor[i][k];
          }
        });
      });
    }
    /*
       auto aa = (double(*)[bs][bs]) (*a)[i][i];
       for (int i = 0; i < bs; i++)
       {
        for (int j = i + 1; j < bs; j++)
          (*aa)[j][i] /= (*aa)[i][i];
 
       for (int j = i + 1; j < bs; j++)
          for (int k = i + 1; k < bs; k++) 
        	  (*aa)[j][k] -= (*aa)[j][i] * (*aa)[i][k];
       } 
  */
       
    for (int k = i + 1; k < nb; k++) 
    {
    	// proc_u(bs, bs, (*a)[i][i], (*a)[j][i]);
      //buffer aa(reinterpret_cast<double *>((*a)[i][i]), range(bs, bs));
      buffer au(reinterpret_cast<double *>((*a)[i][k]), range(bs, bs));
  
      q.submit([&](handler &h) {
        // Read from al and au, write to ag
        auto AA = aa.get_access<access::mode::read>(h);
        auto AU = au.get_access<access::mode::read_write>(h);
        
        int width_a = AA.get_range()[1];
    	
    	  // Execute kernel.
        h.parallel_for(range(bs, bs), [=](id<2> index) {
          // Get global position in Y direction.
          int i = index[0];
          // Get global position in X direction.
          int k = index[1];

          // Compute the result of one element of AL
          for (int j = i + 1; j < width_a; j++) {
    	      AU[j][k] -=  AA[j][i] * AA[i][k];
    	    }
        });
      });
      /*
      auto aa = (double(*)[bs][bs]) (*a)[i][i];
      auto au = (double(*)[bs][bs]) (*a)[i][k];
  
      for (int i = 0; i < bs; i++)
        for (int j = i + 1; j < bs; j++)
          for (int k = 0; k < bs; k++)
            (*au)[j][k] -= (*aa)[j][i] * (*au)[i][k];
      */
    }     
            
    for (int j = i + 1; j < nb; j++)
    {
    	// proc_l(bs, bs, (*a)[i][i], (*a)[j][i]);
    	//buffer aa(reinterpret_cast<double *>((*a)[i][i]), range(bs, bs));
    	buffer al(reinterpret_cast<double *>((*a)[j][i]), range(bs, bs));
	    
      q.submit([&](handler &h) {
        // Read from al and au, write to ag
        auto AA = aa.get_access<access::mode::read>(h);
        auto AL = al.get_access<access::mode::read_write>(h);
        
        int width_a = AL.get_range()[1];
    	
    	  // Execute kernel.
        h.parallel_for(range(bs, bs), [=](id<2> index) {
          // Get global position in Y direction.
          int j = index[0];
          // Get global position in X direction.
          int i = index[1];

          // Compute the result of one element of AG
          AL[j][i] /= AA[i][i];
          for (int k = i + 1; k < width_a; k++) {
    	      AL[j][k] -=  AL[j][i] * AA[i][k];
    	    }
        });
      });
      
      /*
    	auto aa = (double(*)[bs][bs]) (*a)[i][i];
      auto al = (double(*)[bs][bs]) (*a)[j][i];

      for (int i = 0; i < bs; i++)
        for (int j = 0; j < bs; j++)
        {
          (*al)[j][i] /= (*aa)[i][i];
          for (int k = i + 1; k < bs; k++) 
    	      (*al)[j][k] -=  (*al)[j][i] * (*aa)[i][k];
        }
      */
      
      for (int k = i + 1; k < nb; k++) 
	    {
	      // proc_g(bs, bs, bs, (*a)[j][i], (*a)[i][k], (*a)[j][k]);
	      // It works
	      //buffer al(reinterpret_cast<double *>((*a)[j][i]), range(bs, bs));
	      buffer au(reinterpret_cast<double *>((*a)[i][k]), range(bs, bs));
	      buffer ag(reinterpret_cast<double *>((*a)[j][k]), range(bs, bs));
	      
	      q.submit([&](handler &h) {
          // Read from al and au, write to ag
          auto AL = al.get_access<access::mode::read>(h);
          auto AU = au.get_access<access::mode::read>(h);
          auto AG = ag.get_access<access::mode::write>(h);

          int width_a = AL.get_range()[1];

          // Execute kernel.
          h.parallel_for(range(bs, bs), [=](id<2> index) {
            // Get global position in Y direction.
            int row = index[0];
            // Get global position in X direction.
            int col = index[1];

            // Compute the result of one element of AG
            for (int k = 0; k < width_a; k++) {
              AG[row][col] -= AL[row][k] * AU[k][col];
            }
          });
        });
      }
    }
  }
  
  cout << "Time matrixLUBlock: " << matrixLUBlock.Elapsed() << std::endl;
  
  int result;
  cout << "Result of matrix LU-decomposition using DPC++: "; 
  // result = VerifyResult(nb, bs, aLU);
   
  cout << "Source\n"; print_matrix(nb, bs, aS);
 
  free_blocked_matrix(aS);
 
  return result;
}
