#include "config.hpp"

#ifdef LB_ADAPTIVE_GPU
#include <cuda.h>
#include <stdio.h>

#include "cuda_interface.hpp"
#include "cuda_utils.hpp"
#include "lb-adaptive-gpu.hpp"

__global__ void simple_kernel(test_grid_t *a) {
  a[blockIdx.x].thread_idx[threadIdx.x][threadIdx.y][threadIdx.z] =
      LBADAPT_PATCHSIZE_HALO * LBADAPT_PATCHSIZE_HALO * threadIdx.z +
      LBADAPT_PATCHSIZE_HALO * threadIdx.y +
      threadIdx.x;
  a[blockIdx.x].block_idx[threadIdx.x][threadIdx.y][threadIdx.z] =
      LBADAPT_PATCHSIZE_HALO * LBADAPT_PATCHSIZE_HALO * blockIdx.z +
      LBADAPT_PATCHSIZE_HALO * blockIdx.y +
      blockIdx.x;
  printf("block: %i, %i, %i => %f; thread %i, %i, %i => %f\n",
         blockIdx.x, blockIdx.y, blockIdx.z,
         a[blockIdx.x].block_idx[threadIdx.x][threadIdx.y][threadIdx.z],
         threadIdx.x, threadIdx.y, threadIdx.z,
         a[blockIdx.x].thread_idx[threadIdx.x][threadIdx.y][threadIdx.z]);
}

void test(test_grid_t *data_host) {
  test_grid_t *data_dev;
  size_t data_size = sizeof(test_grid_t) * local_num_quadrants;

  cudaMalloc(&data_dev, data_size);

  dim3 blocks_per_grid(local_num_quadrants);
  dim3 threads_per_block(LBADAPT_PATCHSIZE_HALO, LBADAPT_PATCHSIZE_HALO,
                         LBADAPT_PATCHSIZE_HALO);

  printf ("blocks: %i, %i, %i\n", blocks_per_grid.x, blocks_per_grid.y, blocks_per_grid.z);
  printf ("threads: %i, %i, %i\n", threads_per_block.x, threads_per_block.y, threads_per_block.z);
  simple_kernel<<<blocks_per_grid, threads_per_block>>>(data_dev);

  cudaMemcpy(data_host, data_dev, data_size, cudaMemcpyDeviceToHost);

  cudaFree(data_dev);
}
#endif // LB_ADAPTIVE_GPU
