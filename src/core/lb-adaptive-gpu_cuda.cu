#include "config.hpp"

#ifdef LB_ADAPTIVE_GPU

#include <assert.h>
#include <cuda.h>
#include <stdio.h>

#include "cuda_interface.hpp"
#include "cuda_utils.hpp"
#include "lb-adaptive-gpu.hpp"
#include "utils.hpp"

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

  CUDA_CALL(cudaMalloc(&data_dev, data_size));

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

  CUDA_CALL(cudaMemcpy(data_host, data_dev, data_size, cudaMemcpyDeviceToHost));

  CUDA_CALL(cudaFree(data_dev));
}

void allocate_device_memory_gpu() {
  assert (dev_local_real_quadrants == NULL);
  assert (dev_local_virt_quadrants == NULL);
  dev_local_real_quadrants = (lbadapt_payload_t**) malloc (sizeof(lbadapt_payload_t*) * P8EST_MAXLEVEL);
  dev_local_virt_quadrants = (lbadapt_payload_t**) malloc (sizeof(lbadapt_payload_t*) * P8EST_MAXLEVEL);
  for (int l = 0; l < P8EST_MAXLEVEL; ++l) {
    CUDA_CALL(cudaMalloc(&dev_local_real_quadrants[l],
                         local_num_real_quadrants_level[l] *
                             sizeof(lbadapt_payload_t)));
    CUDA_CALL(cudaMalloc(&dev_local_virt_quadrants[l],
                         local_num_virt_quadrants_level[l] *
                             sizeof(lbadapt_payload_t)));
  }
}

void deallocate_device_memory_gpu() {
  if (dev_local_real_quadrants == NULL) {
    return;
  }
  int device;
  CUDA_CALL(cudaGetDevice(&device));
  printf ("[rank %i] free memory on device %i\n", this_node, device);
  //CUDA_CALL(cudaThreadSynchronize());
  for (int l = 0; l < P8EST_MAXLEVEL; ++l) {
    //printf("[rank %i] free real of level %i (%i)\n", this_node, l, dev_local_real_quadrants[l]);
    CUDA_CALL(cudaFree(dev_local_real_quadrants[l]));
    //printf("[rank %i] success\n", this_node);
    //printf("[rank %i] free virt of level %i\n", this_node, l);
    CUDA_CALL(cudaFree(dev_local_virt_quadrants[l]));
    //printf("[rank %i] success\n", this_node);
  }
  free (dev_local_real_quadrants);
  free (dev_local_virt_quadrants);
  dev_local_real_quadrants = NULL;
  dev_local_virt_quadrants = NULL;
}

void copy_data_to_device(lbadapt_payload_t *source_real,
                         lbadapt_payload_t *source_virt, int level) {
  if (source_real) {
    CUDA_CALL(cudaMemcpy(dev_local_real_quadrants[level], source_real,
                         sizeof(lbadapt_payload_t) *
                             local_num_real_quadrants_level[level],
                         cudaMemcpyHostToDevice));
  }
  if (source_virt) {
    CUDA_CALL(cudaMemcpy(dev_local_virt_quadrants[level], source_virt,
                         sizeof(lbadapt_payload_t) *
                             local_num_virt_quadrants_level[level],
                         cudaMemcpyHostToDevice));
  }
}

void copy_data_from_device(lbadapt_payload_t *dest_real,
                           lbadapt_payload_t *dest_virt, int level) {
  if (dest_real) {
    CUDA_CALL(cudaMemcpy(dest_real, dev_local_real_quadrants[level],
                         sizeof(lbadapt_payload_t) *
                             local_num_real_quadrants_level[level],
                         cudaMemcpyDeviceToHost));
  }
  if (dest_virt) {
    CUDA_CALL(cudaMemcpy(dest_virt, dev_local_virt_quadrants[level],
                         sizeof(lbadapt_payload_t) *
                             local_num_virt_quadrants_level[level],
                         cudaMemcpyDeviceToHost));
  }
}
#endif // LB_ADAPTIVE_GPU
