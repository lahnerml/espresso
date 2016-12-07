#include "config.hpp"

#ifdef LB_ADAPTIVE_GPU
#include <cuda.h>
#include <stdio.h>

#include "cuda_interface.hpp"
#include "cuda_utils.hpp"
#include "lb-adaptive-gpu.hpp"

__global__ void simple_kernel(thread_block_container_t *a) {
  a[blockIdx.x].thread_idx[threadIdx.x][threadIdx.y][threadIdx.z] =
      LBADAPT_PATCHSIZE_HALO * LBADAPT_PATCHSIZE_HALO * threadIdx.z +
      LBADAPT_PATCHSIZE_HALO * threadIdx.y + threadIdx.x;
  a[blockIdx.x].block_idx[threadIdx.x][threadIdx.y][threadIdx.z] =
      LBADAPT_PATCHSIZE_HALO * LBADAPT_PATCHSIZE_HALO * blockIdx.z +
      LBADAPT_PATCHSIZE_HALO * blockIdx.y + blockIdx.x;
}

void show_blocks_threads(thread_block_container_t *data_host) {
  thread_block_container_t *data_dev;
  size_t data_size = sizeof(thread_block_container_t) * local_num_quadrants;

  cudaMalloc(&data_dev, data_size);

  dim3 blocks_per_grid(local_num_quadrants);
  dim3 threads_per_block(LBADAPT_PATCHSIZE_HALO, LBADAPT_PATCHSIZE_HALO,
                         LBADAPT_PATCHSIZE_HALO);

#if 0
  printf("blocks: %i, %i, %i\n", blocks_per_grid.x, blocks_per_grid.y,
         blocks_per_grid.z);
  printf("threads: %i, %i, %i\n", threads_per_block.x, threads_per_block.y,
         threads_per_block.z);
#endif // 0
  simple_kernel<<<blocks_per_grid, threads_per_block>>>(data_dev);

  cudaMemcpy(data_host, data_dev, data_size, cudaMemcpyDeviceToHost);

  cudaFree(data_dev);
}
#endif // LB_ADAPTIVE_GPU
