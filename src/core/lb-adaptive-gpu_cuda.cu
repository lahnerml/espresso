#include "config.hpp"

#ifdef LB_ADAPTIVE_GPU

#include <assert.h>
#include <cuda.h>
#include <stdio.h>

#include "cuda_interface.hpp"
#include "cuda_utils.hpp"
#include "lb-adaptive-gpu.hpp"
#include "lb-d3q19.hpp"
#include "utils.hpp"

LB_Parameters d_lbpar;
LB_Model d_lbmodel = {19,      d3q19_lattice, d3q19_coefficients,
                      d3q19_w, NULL,          1. / 3.};

void lbadapt_gpu_init() {
  lbpar.agrid = (lb_float)P8EST_QUADRANT_LEN(lbpar.max_refinement_level) /
                ((lb_float)LBADAPT_PATCHSIZE * (lb_float)P8EST_ROOT_LEN);
  CUDA_CALL(cudaMemcpy(&d_lbpar, &lbpar, sizeof(LB_Parameters),
                       cudaMemcpyHostToDevice));
}

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
                                     lbadapt_payload_t *source_virt,
                                     int level) {
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
                                       lbadapt_payload_t *dest_virt,
                                       int level) {
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

__global__ void lbadapt_gpu_collide_calc_modes(lbadapt_payload_t *quad_data) {
  // clang-format off
  // mass mode
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[0] =
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 0] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 1] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 2] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 3] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 4] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 5] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 6] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 7] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 8] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 9] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][10] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][11] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][12] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][13] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][14] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][15] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][16] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][17] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][18];

  // kinetic modes
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[1] =
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 1] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 2]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 7] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 8]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 9] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][10]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][11] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][12]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][13] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][14]);

  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[2] =
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 3] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 4]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 7] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 8]) -
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 9] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][10]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][15] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][16]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][17] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][18]);

  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[3] =
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 5] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 6]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][11] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][12]) -
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][13] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][14]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][15] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][16]) -
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][17] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][18]);

  // stress modes
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[4] =
     -quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 0] +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 7] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 8]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 9] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][10]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][11] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][12]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][13] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][14]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][15] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][16]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][17] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][18]);

  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[5] =
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 1] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 2]) -
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 3] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 4]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][11] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][12]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][13] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][14]) -
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][15] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][16]) -
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][17] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][18]);

  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[6] =
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 1] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 2]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 3] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 4]) -
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][11] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][12]) -
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][13] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][14]) -
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][15] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][16]) -
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][17] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][18]) -
    2.0f * ((quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 5] +
             quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 6]) -
            (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 7] +
             quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 8]) -
            (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 9] +
             quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][10]));

  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[7] =
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 7] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 8]) -
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 9] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][10]);

  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[8] =
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][11] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][12]) -
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][13] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][14]);

  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[9] =
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][15] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][16]) -
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][17] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][18]);

  // kinetic modes
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[10] =
    -2.0f * (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 1] -
             quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 2]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 7] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 8]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 9] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][10]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][11] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][12]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][13] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][14]);

  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[11] =
    -2.0f * (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 3] -
             quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 4]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 7] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 8]) -
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 9] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][10]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][15] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][16]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][17] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][18]);

  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[12] =
    -2.0f * (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 5] -
             quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 6]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][11] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][12]) -
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][13] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][14]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][15] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][16]) -
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][17] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][18]);

  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[13] =
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 7] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 8]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 9] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][10]) -
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][11] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][12]) -
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][13] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][14]);

  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[14] =
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 7] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 8]) -
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 9] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][10]) -
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][15] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][16]) -
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][17] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][18]);

  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[15] =
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][11] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][12]) -
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][13] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][14]) -
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][15] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][16]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][17] -
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][18]);

  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[16] =
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 0] +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 7] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 8]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 9] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][10]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][11] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][12]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][13] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][14]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][15] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][16]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][17] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][18]) -
    2.0f * ((quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 1] +
             quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 2]) +
            (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 3] +
             quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 4]) +
            (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 5] +
             quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 6]));

  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[17] =
    -(quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 1] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 2]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 3] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 4]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][11] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][12]) +
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][13] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][14]) -
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][15] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][16]) -
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][17] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][18]);

  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[18] =
    -(quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 1] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 2]) -
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 3] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 4]) -
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][11] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][12]) -
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][13] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][14]) -
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][15] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][16]) -
     (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][17] +
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][18]) +
    2.0f * ((quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 5] +
             quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 6]) +
            (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 7] +
             quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 8]) +
            (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][ 9] +
             quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].lbfluid[0][10]));
  // clang-format on
}

__global__ void lbadapt_gpu_collide_relax_modes(lbadapt_payload_t *quad_data) {
  lb_float rho, j[3], pi[6];
  rho = quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[0];
}

__global__ void
lbadapt_gpu_collide_thermalize_modes(lbadapt_payload_t *quad_data) {}

__global__ void lbadapt_gpu_collide_apply_forces(lbadapt_payload_t *quad_data) {

}

void lbadapt_gpu_execute_collision_kernel(int level) {
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

void lbadapt_gpu_execute_populate_virtuals_kernel(int level) {}

void lbadapt_gpu_execute_update_from_virtuals_kernel(int level) {}

__global__ void lbadapt_gpu_stream(lbadapt_payload_t *quad_data) {}

__global__ void lbadapt_gpu_stream_virtuals(lbadapt_payload_t *quad_data) {}

void lbadapt_gpu_execute_streaming_kernel(int level) {}

void lbadapt_gpu_execute_bounce_back_kernel(int level) {}

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
