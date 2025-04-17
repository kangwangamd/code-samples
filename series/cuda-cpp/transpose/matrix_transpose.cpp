#include <stdio.h>
#include <assert.h>
#include <hip/hip_runtime.h>

// Convenience function for checking HIP runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline hipError_t checkHip(hipError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != hipSuccess) {
    fprintf(stderr, "HIP Runtime Error: %s\n", hipGetErrorString(result));
    assert(result == hipSuccess);
  }
#endif
  return result;
}

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;
const int NUM_REPS = 1;

// Check errors and print GB/s
void postprocess(const float *ref, const float *res, int n, float ms)
{
  bool passed = true;
  for (int i = 0; i < n; i++)
    if (res[i] != ref[i]) {
      printf("%d %f %f\n", i, res[i], ref[i]);
      printf("%25s\n", "*** FAILED ***");
      passed = false;
      break;
    }
  if (passed)
    printf("%20.2f\n", 2 * n * sizeof(float) * 1e-6 * NUM_REPS / ms );
//   printf("ms=%f\n", ms);
}

// simple copy kernel
__global__ void copy(float *odata, const float *idata)
{
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;
  if (threadIdx.x == 1 && threadIdx.y == 1 && blockIdx.x == 0 && blockIdx.y == 0)
      printf("x: %d, y: %d, threadIdx.x: %d , threadIdx.y %d, blockIdx.x %d, blockIdx.y %d \n", x, y, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);

  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
  {
      if (x == 1 && y == 1){
	  // printf("blockIdx.x: %d, threadIdx.x: %d, blockIdx.y %d, threadIdx.y %d \n", blockIdx.x, threadIdx.x, blockIdx.y, threadIdx.y);
	  printf("j: %d, (y+j)*width + x: %d \n", j, (y+j)*width + x);
      }
      odata[(y+j)*width + x] = idata[(y+j)*width + x];
  }
}

// copy kernel using shared memory
__global__ void copySharedMem(float *odata, const float *idata)
{
  __shared__ float tile[TILE_DIM * TILE_DIM];
  
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS){
      if (x == 31 && y == 7)
	  printf("j: %d, (y+j)*width + x: %d, (threadIdx.y+j)*TILE_DIM + threadIdx.x %d. \n", j, (y+j)*width + x, (threadIdx.y+j)*TILE_DIM + threadIdx.x);
     tile[(threadIdx.y+j)*TILE_DIM + threadIdx.x] = idata[(y+j)*width + x];
  }
  __syncthreads();

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(y+j)*width + x] = tile[(threadIdx.y+j)*TILE_DIM + threadIdx.x];          
}

// naive transpose
__global__ void transposeNaive(float *odata, const float *idata)
{
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS){
      if (x == 31 && y == 1)
	  printf("j: %d, (y+j)*width + x: %d, x*width + (y+j) %d. \n", j, (y+j)*width + x, x*width + (y+j));
    odata[x*width + (y+j)] = idata[(y+j)*width + x] * 10;
  }
}

// coalesced transpose
__global__ void transposeCoalesced(float *odata, const float *idata)
{
  __shared__ float tile[TILE_DIM][TILE_DIM];
    
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS){
      if (x == 31 && y == 1)
	  printf("j: %d, (y+j)*width + x: %d, threadIdx.y+j %d, threadIdx.x %d. \n", j, (y+j)*width + x, threadIdx.y+j, threadIdx.x);
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];
  }

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j]* 10;
}
   
// No bank-conflict transpose
__global__ void transposeNoBankConflicts(float *odata, const float *idata)
{
  __shared__ float tile[TILE_DIM][TILE_DIM+1];
    
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS){
      if (x == 31 && y == 0)
	  printf("TILE: j: %d, (y+j)*width + x: %d, threadIdx.y+j %d, threadIdx.x %d. \n", j, (y+j)*width + x, threadIdx.y+j, threadIdx.x);
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];
  }

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS){
      if (x == 31 && y == 0)
	  printf("TILE OUTPUT: j: %d, (y+j)*width + x: %d, threadIdx.x %d, threadIdx.y + j %d. \n", j, (y+j)*width + x, threadIdx.x, threadIdx.y + j);
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j]* 10;
  }
}



int main(int argc, char **argv)
{
  const int nx = 1024;
  const int ny = 1024;
  const int mem_size = nx*ny*sizeof(float);

  dim3 dimGrid(nx/TILE_DIM, ny/TILE_DIM, 1);
  dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

  int devId = 0;
  if (argc > 1) devId = atoi(argv[1]);

  hipDeviceProp_t prop;
  checkHip(hipGetDeviceProperties(&prop, devId));
  printf("\nDevice : %s\n", prop.name);
  printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n", 
         nx, ny, TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
  printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n",
         dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
  
  checkHip(hipSetDevice(devId));

  float *h_idata = (float*)malloc(mem_size);
  float *h_cdata = (float*)malloc(mem_size);
  float *h_tdata = (float*)malloc(mem_size);
  float *gold    = (float*)malloc(mem_size);
  
  float *d_idata, *d_cdata, *d_tdata;
  checkHip(hipMalloc(&d_idata, mem_size));
  checkHip(hipMalloc(&d_cdata, mem_size));
  checkHip(hipMalloc(&d_tdata, mem_size));

  // check parameters and calculate execution configuration
  if (nx % TILE_DIM || ny % TILE_DIM) {
    printf("nx and ny must be a multiple of TILE_DIM\n");
    goto error_exit;
  }

  if (TILE_DIM % BLOCK_ROWS) {
    printf("TILE_DIM must be a multiple of BLOCK_ROWS\n");
    goto error_exit;
  }
    
  // host
  for (int j = 0; j < ny; j++)
    for (int i = 0; i < nx; i++)
      h_idata[j*nx + i] = j*nx + i;

  // correct result for error checking
  for (int j = 0; j < ny; j++)
    for (int i = 0; i < nx; i++)
      gold[j*nx + i] = h_idata[i*nx + j]* 10;
  
  // device
  checkHip(hipMemcpy(d_idata, h_idata, mem_size, hipMemcpyHostToDevice));
  
  // events for timing
  hipEvent_t startEvent, stopEvent;
  checkHip(hipEventCreate(&startEvent));
  checkHip(hipEventCreate(&stopEvent));
  float ms;

  // ------------ 
  // time kernels
  // ------------ 
  printf("%25s%25s\n", "Routine", "Bandwidth (GB/s)");

  // Device spec
  printf("%25s %19.2f\n", "Peak Memory Bandwidth", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  
  // ----
  // copy 
  // ----
  printf("%25s", "copy \n");
  checkHip(hipMemset(d_cdata, 0, mem_size));
  // warm up
  copy<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
  checkHip(hipEventRecord(startEvent, 0));
  for (int i = 0; i < NUM_REPS; i++)
     copy<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
  checkHip(hipEventRecord(stopEvent, 0));
  checkHip(hipEventSynchronize(stopEvent));
  checkHip(hipEventElapsedTime(&ms, startEvent, stopEvent));
  checkHip(hipMemcpy(h_cdata, d_cdata, mem_size, hipMemcpyDeviceToHost));
  postprocess(h_idata, h_cdata, nx*ny, ms);
  

  // ------------- 
  // copySharedMem 
  // ------------- 
  printf("%25s", "shared memory copy \n");
  checkHip(hipMemset(d_cdata, 0, mem_size));
  // warm up
  copySharedMem<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
  checkHip(hipEventRecord(startEvent, 0));
  for (int i = 0; i < NUM_REPS; i++)
     copySharedMem<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
  checkHip(hipEventRecord(stopEvent, 0));
  checkHip(hipEventSynchronize(stopEvent));
  checkHip(hipEventElapsedTime(&ms, startEvent, stopEvent));
  checkHip(hipMemcpy(h_cdata, d_cdata, mem_size, hipMemcpyDeviceToHost));
  postprocess(h_idata, h_cdata, nx * ny, ms);

printf("--------------------------------------------------------\n");
  // -------------- 
  // 1. transposeNaive 
  // -------------- 
  printf("%25s", "naive transpose \n");
  checkHip(hipMemset(d_tdata, 0, mem_size));
  // warmup
  transposeNaive<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
  checkHip(hipEventRecord(startEvent, 0));
  for (int i = 0; i < NUM_REPS; i++)
     transposeNaive<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
  checkHip(hipEventRecord(stopEvent, 0));
  checkHip(hipEventSynchronize(stopEvent));
  checkHip(hipEventElapsedTime(&ms, startEvent, stopEvent));
  checkHip(hipMemcpy(h_tdata, d_tdata, mem_size, hipMemcpyDeviceToHost));
  postprocess(gold, h_tdata, nx * ny, ms);

  // ------------------ 
  // 2. transposeCoalesced 
  // ------------------ 
  printf("%25s", "coalesced transpose \n");
  checkHip(hipMemset(d_tdata, 0, mem_size));
  // warmup
  transposeCoalesced<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
  checkHip(hipEventRecord(startEvent, 0));
  for (int i = 0; i < NUM_REPS; i++)
     transposeCoalesced<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
  checkHip(hipEventRecord(stopEvent, 0));
  checkHip(hipEventSynchronize(stopEvent));
  checkHip(hipEventElapsedTime(&ms, startEvent, stopEvent));
  checkHip(hipMemcpy(h_tdata, d_tdata, mem_size, hipMemcpyDeviceToHost));
  postprocess(gold, h_tdata, nx * ny, ms);


  // ------------------------ 
  // 3. transposeNoBankConflicts 
  // ------------------------ 
  printf("%25s", "conflict-free transpose \n");
  checkHip(hipMemset(d_tdata, 0, mem_size));
  // warmup
  transposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
  checkHip(hipEventRecord(startEvent, 0));
  for (int i = 0; i < NUM_REPS; i++)
     transposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
  checkHip(hipEventRecord(stopEvent, 0));
  checkHip(hipEventSynchronize(stopEvent));
  checkHip(hipEventElapsedTime(&ms, startEvent, stopEvent));
  checkHip(hipMemcpy(h_tdata, d_tdata, mem_size, hipMemcpyDeviceToHost));
  postprocess(gold, h_tdata, nx * ny, ms);



error_exit:
  // cleanup
  checkHip(hipEventDestroy(startEvent));
  checkHip(hipEventDestroy(stopEvent));
  checkHip(hipFree(d_tdata));
  checkHip(hipFree(d_cdata));
  checkHip(hipFree(d_idata));
  free(h_idata);
  free(h_tdata);
  free(h_cdata);
  free(gold);
}

