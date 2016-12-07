#include "config.hpp"

#ifdef LB_ADAPTIVE_GPU
#include <cuda.h>

#include "cuda_interface.hpp"
#include "cuda_utils.hpp"
#include "lb-adaptive-gpu.hpp"

__global__ void simple_kernel(test_grid_t *a) {
  a->thread_idx[threadIdx.x][threadIdx.y][threadIdx.z] =
      (lb_float)LBADAPT_PATCHSIZE_HALO * (lb_float)LBADAPT_PATCHSIZE_HALO *
          (lb_float)threadIdx.z +
      (lb_float)LBADAPT_PATCHSIZE_HALO * (lb_float)threadIdx.y +
      (lb_float)threadIdx.x;
  a->block_idx[threadIdx.x][threadIdx.y][threadIdx.z] =
      (lb_float)LBADAPT_PATCHSIZE_HALO * (lb_float)LBADAPT_PATCHSIZE_HALO *
          (lb_float)blockIdx.z +
      (lb_float)LBADAPT_PATCHSIZE_HALO * (lb_float)blockIdx.y +
      (lb_float)blockIdx.x;
}

void test(test_grid_t *data_host) {
  test_grid_t *data_dev;
  size_t data_size = sizeof(test_grid_t) * local_num_quadrants;

  cudaMalloc(&data_dev, data_size);

  dim3 blocks_per_grid(local_num_quadrants);
  dim3 threads_per_block(LBADAPT_PATCHSIZE_HALO, LBADAPT_PATCHSIZE_HALO,
                         LBADAPT_PATCHSIZE_HALO);

  simple_kernel<<<blocks_per_grid, threads_per_block>>>(data_dev);

  cudaMemcpy(data_host, data_dev, data_size, cudaMemcpyDeviceToHost);

  cudaFree(data_dev);
}
#endif // LB_ADAPTIVE_GPU
