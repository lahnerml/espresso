#include "config.hpp"

#ifdef LB_ADAPTIVE_GPU

#include <assert.h>
#include <cuda.h>
#include <stdio.h>

#include "cuda_interface.hpp"
#include "cuda_utils.hpp"
#include "lb-adaptive-gpu.hpp"
#include "utils.hpp"

void lbadapt_gpu_allocate_device_memory() {
  assert(dev_local_real_quadrants == NULL);
  assert(dev_local_virt_quadrants == NULL);
  dev_local_real_quadrants = (lbadapt_payload_t **)malloc(
      sizeof(lbadapt_payload_t *) * P8EST_MAXLEVEL);
  dev_local_virt_quadrants = (lbadapt_payload_t **)malloc(
      sizeof(lbadapt_payload_t *) * P8EST_MAXLEVEL);
  for (int l = 0; l < P8EST_MAXLEVEL; ++l) {
    CUDA_CALL(cudaMalloc(&dev_local_real_quadrants[l],
                         local_num_real_quadrants_level[l] *
                             sizeof(lbadapt_payload_t)));
    CUDA_CALL(cudaMalloc(&dev_local_virt_quadrants[l],
                         local_num_virt_quadrants_level[l] *
                             sizeof(lbadapt_payload_t)));
  }
}

void lbadapt_gpu_deallocate_device_memory() {
  if (dev_local_real_quadrants == NULL) {
    return;
  }
  for (int l = 0; l < P8EST_MAXLEVEL; ++l) {
    CUDA_CALL(cudaFree(dev_local_real_quadrants[l]));
    CUDA_CALL(cudaFree(dev_local_virt_quadrants[l]));
  }
  free(dev_local_real_quadrants);
  free(dev_local_virt_quadrants);
  dev_local_real_quadrants = NULL;
  dev_local_virt_quadrants = NULL;
}

// TODO: Use asynchronous memcpy
void lbadapt_gpu_copy_data_to_device(lbadapt_payload_t *source_real,
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

// TODO: Use asynchronous memcpy
void lbadapt_gpu_copy_data_from_device(lbadapt_payload_t *dest_real,
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

__global__ void lbadapt_gpu_collide_calc_modes(lbadapt_payload_t *quad_data) {}

__global__ void lbadapt_gpu_collide_relax_modes(lbadapt_payload_t *quad_data) {}

__global__ void
lbadapt_gpu_collide_thermalize_modes(lbadapt_payload_t *quad_data) {}

__global__ void lbadapt_gpu_collide_apply_forces(lbadapt_payload_t *quad_data) {

}

void execute_collision_kernel(int level) {
  dim3 blocks_per_grid(local_num_real_quadrants_level[level]);
  dim3 threads_per_block(LBADAPT_PATCHSIZE_HALO, LBADAPT_PATCHSIZE_HALO,
                         LBADAPT_PATCHSIZE_HALO);

  // call kernels: calc modes, relax modes, thermalize modes, apply forces
  // TODO: smarter to put into a single kernel?
  lbadapt_gpu_collide_calc_modes<<<blocks_per_grid, threads_per_block>>>(
      dev_local_real_quadrants[level]);
  lbadapt_gpu_collide_relax_modes<<<blocks_per_grid, threads_per_block>>>(
      dev_local_real_quadrants[level]);
  lbadapt_gpu_collide_thermalize_modes<<<blocks_per_grid, threads_per_block>>>(
      dev_local_real_quadrants[level]); // stub only
  lbadapt_gpu_collide_apply_forces<<<blocks_per_grid, threads_per_block>>>(
      dev_local_real_quadrants[level]);
}

// NOT LB-specific; visualize utilization of thread and block ids in vtk format
__global__ void visualize_threads_blocks(thread_block_container_t *a) {
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

  visualize_threads_blocks<<<blocks_per_grid, threads_per_block>>>(data_dev);

  CUDA_CALL(cudaMemcpy(data_host, data_dev, data_size, cudaMemcpyDeviceToHost));

  CUDA_CALL(cudaFree(data_dev));
}
#endif // LB_ADAPTIVE_GPU
