DPCPP_CXX = dpcpp
DPCPP_CXXFLAGS = -std=c++17 -g -o 
DPCPP_LDFLAGS = 
DPCPP_EXE_NAME = matrix_lu_dpc
DPCPP_SOURCES = src/matrix_lu_dpcpp.cpp

BLOCK_EXE_NAME = matrix_lu_block
BLOCK_SOURCES = src/matrix_lu_block.cpp

TIME_EXE_NAME = time_test
TIME_SOURCES = src/time_test.cpp

BLOCK_DPCPP_EXE_NAME = matrix_lu_block_dpc
BLOCK_DPCPP_SOURCES = src/matrix_lu_block_dpcpp.cpp

BLOCK_DPCPP_FUNC_EXE_NAME = matrix_lu_block_dpc_func
BLOCK_DPCPP_FUNC_SOURCES = src/matrix_lu_block_dpcpp_func.cpp

CXX = icc 
CXXFLAGS_REPORT = -qopt-report-stdout -qopt-report-format=vs -qopt-report=3 -qopt-report-phase=loop,vec  
OMP_CXXFLAGS = -qnextgen -fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__  -g -o
OMP_LDFLAGS = 
OMP_EXE_NAME = matrix_lu_omp
OMP_SOURCES = src/matrix_lu_omp.cpp

BLOCK_OMP_EXE_NAME = matrix_lu_block_omp
BLOCK_OMP_SOURCES = src/matrix_lu_block_omp.cpp

BLOCK_OMP_GPU_EXE_NAME = matrix_lu_block_omp_gpu
BLOCK_OMP_GPU_SOURCES = src/matrix_lu_block_omp_gpu.cpp

all:
	$(DPCPP_CXX) $(DPCPP_CXXFLAGS) $(DPCPP_EXE_NAME) $(DPCPP_SOURCES) $(DPCPP_LDFLAGS)

build_dpcpp:
	$(DPCPP_CXX) $(DPCPP_CXXFLAGS) $(DPCPP_EXE_NAME) $(DPCPP_SOURCES) $(DPCPP_LDFLAGS)
	
build_time:
	$(DPCPP_CXX) $(DPCPP_CXXFLAGS) $(TIME_EXE_NAME) $(TIME_SOURCES) $(DPCPP_LDFLAGS)

build_block:
	$(DPCPP_CXX) $(DPCPP_CXXFLAGS) $(BLOCK_EXE_NAME) $(BLOCK_SOURCES) $(DPCPP_LDFLAGS)

build_block_dpcpp:
	$(DPCPP_CXX) $(DPCPP_CXXFLAGS) $(BLOCK_DPCPP_EXE_NAME) $(BLOCK_DPCPP_SOURCES) $(DPCPP_LDFLAGS)
	
build_block_dpcpp_func:
	$(DPCPP_CXX) $(DPCPP_CXXFLAGS) $(BLOCK_DPCPP_FUNC_EXE_NAME) $(BLOCK_DPCPP_FUNC_SOURCES) $(DPCPP_LDFLAGS)
	
build_omp:
	$(CXX) $(OMP_CXXFLAGS) $(OMP_EXE_NAME) $(OMP_SOURCES) $(OMP_LDFLAGS)
	
build_block_omp:
	$(CXX) $(OMP_CXXFLAGS) $(BLOCK_OMP_EXE_NAME) $(BLOCK_OMP_SOURCES) $(OMP_LDFLAGS)
	
build_block_omp_gpu:
	$(CXX) $(CXXFLAGS_REPORT) $(OMP_CXXFLAGS) $(BLOCK_OMP_GPU_EXE_NAME) $(BLOCK_OMP_GPU_SOURCES) $(OMP_LDFLAGS)


run:
	./$(DPCPP_EXE_NAME)

run_dpcpp:
	./$(DPCPP_EXE_NAME)

run_omp:
	./$(OMP_EXE_NAME)


clean: 
	rm -rf $(DPCPP_EXE_NAME) $(OMP_EXE_NAME)



