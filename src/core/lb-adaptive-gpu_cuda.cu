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

__global__ void lbadapt_gpu_collide_relax_modes(lbadapt_payload_t *quad_data, int level,
                                                lb_float h_max) {
  lb_float rho, j[3], pi_eq[6];

  /** reconstruct real density */
  rho = quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[0] +
        lbpar.rho[0] * h_max * h_max * h_max;

  /** momentum density is redefined to include half-step of force action */
  j[0] = quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[1];
  j[1] = quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[2];
  j[2] = quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[3];

  j[0] += 0.5 * quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].force[0];
  j[1] += 0.5 * quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].force[1];
  j[2] += 0.5 * quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].force[2];

  /** calculate equilibrium part of stress modes */
  pi_eq[0] = ((j[0] * j[0]) + (j[1] * j[1]) + (j[2] * j[2])) / rho;
  pi_eq[1] = ((j[0] * j[0]) - (j[1]) * j[1])) / rho;
  pi_eq[2] =
      ((j[0] * j[0]) + (j[1] * j[1]) + (j[2] * j[2]) - 3.0f * (j[2] * j[2])) /
      rho;
  pi_eq[3] = (j[0] * j[1]) / rho;
  pi_eq[4] = (j[0] * j[2]) / rho;
  pi_eq[5] = (j[1] * j[2]) / rho;

  /** relax stress modes toward equilibrium */
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[4] =
      pi_eq[0] +
      lbpar.gamma_bulk[level] *
          (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[4] -
           pi_eq[0]);
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[5] =
      pi_eq[1] +
      lbpar.gamma_shear[level] *
          (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[5] -
           pi_eq[1]);
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[6] =
      pi_eq[2] +
      lbpar.gamma_shear[level] *
          (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[6] -
           pi_eq[2]);
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[7] =
      pi_eq[3] +
      lbpar.gamma_shear[level] *
          (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[7] -
           pi_eq[3]);
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[8] =
      pi_eq[4] +
      lbpar.gamma_shear[level] *
          (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[8] -
           pi_eq[4]);
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[9] =
      pi_eq[5] +
      lbpar.gamma_shear[level] *
          (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[9] -
           pi_eq[5]);

  /** relax ghost modes */
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[10] *=
      gamma_odd;
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[11] *=
      gamma_odd;
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[12] *=
      gamma_odd;
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[13] *=
      gamma_odd;
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[14] *=
      gamma_odd;
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[15] *=
      gamma_odd;
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[16] *=
      gamma_even;
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[17] *=
      gamma_even;
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[18] *=
      gamma_even;
}

// TODO: Implement
__global__ void
lbadapt_gpu_collide_thermalize_modes(lbadapt_payload_t *quad_data) {}

__global__ void lbadapt_gpu_collide_apply_forces(lbadapt_payload_t *quad_data,
                                                 int level,
                                                 double h_max) {
  lb_float rho, u[3], C[6];

  /** reconstruct density */
  rho = quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[0] +
        lbpar.rho[0] * h_max * h_max * h_max;

  /** momentum density is redefined in case of external forces */
  j[0] = (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[1] *
         0.5f * quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].force[0];
  j[1] = (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[2] *
         0.5f * quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].force[1];
  j[2] = (quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[3] *
         0.5f * quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].force[2];

  C[0] = (1. + gamma_bulk[level]) * u[0] * f[0] +
         1. / 3. * (gamma_bulk[level] - gamma_shear[level]) * scalar(u, f);
  C[2] = (1. + gamma_bulk[level]) * u[1] * f[1] +
         1. / 3. * (gamma_bulk[level] - gamma_shear[level]) * scalar(u, f);
  C[5] = (1. + gamma_bulk[level]) * u[2] * f[2] +
         1. / 3. * (gamma_bulk[level] - gamma_shear[level]) * scalar(u, f);
  C[1] = 0.5 * (1. + gamma_shear[level]) * (u[0] * f[1] + u[1] * f[0]);
  C[3] = 0.5 * (1. + gamma_shear[level]) * (u[0] * f[2] + u[2] * f[0]);
  C[4] = 0.5 * (1. + gamma_shear[level]) * (u[1] * f[2] + u[2] * f[1]);

  /** update momentum modes */
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[1] +=
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].force[0];
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[2] +=
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].force[1];
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[3] +=
      quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].force[2];

  /** update stress modes */
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[4] +=
      C[0] + C[2] + C[5];
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[5] +=
      C[0] - C[2];
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[6] +=
      C[0] + C[2] - 2.0f * C[5];
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[7] += C[1];
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[8] += C[3];
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].modes[9] += C[4];

  /** reset external force */
#ifdef EXTERNAL_FORCES
  // unit conversion: force density
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].force[0] =
      prefactors[level] * lbpar.ext_force[0] * SQR(h_max) * SQR(lbpar.tau);
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].force[1] =
      prefactors[level] * lbpar.ext_force[1] * SQR(h_max) * SQR(lbpar.tau);
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].force[2] =
      prefactors[level] * lbpar.ext_force[2] * SQR(h_max) * SQR(lbpar.tau);
#else  // EXTERNAL_FORCES
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].force[0] =
      (lb_float) 0.0;
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].force[1] =
      (lb_float) 0.0;
  quad_data->patch[threadIdx.x][threadIdx.y][threadIdx.z].force[2] =
      (lb_float) 0.0;
#endif // EXTERNAL_FORCES
}

void lbadapt_gpu_execute_collision_kernel(int level) {
  dim3 blocks_per_grid(local_num_real_quadrants_level[level]);
  dim3 threads_per_block(LBADAPT_PATCHSIZE_HALO, LBADAPT_PATCHSIZE_HALO,
                         LBADAPT_PATCHSIZE_HALO);

  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(lbpar.max_refinement_level) /
                   ((lb_float)LBADAPT_PATCHSIZE * (lb_float)P8EST_ROOT_LEN);
  lb_float h = (lb_float)P8EST_QUADRANT_LEN(level) /
               ((lb_float)LBADAPT_PATCHSIZE * (lb_float)P8EST_ROOT_LEN);

  // call kernels: calc modes, relax modes, thermalize modes, apply forces
  // TODO: smarter to put into a single kernel?
  lbadapt_gpu_collide_calc_modes<<<blocks_per_grid, threads_per_block>>>(
      dev_local_real_quadrants[level]);
  lbadapt_gpu_collide_relax_modes<<<blocks_per_grid, threads_per_block>>>(
      dev_local_real_quadrants[level], level, h_max);
  lbadapt_gpu_collide_thermalize_modes<<<blocks_per_grid, threads_per_block>>>(
      dev_local_real_quadrants[level]); // stub only
  lbadapt_gpu_collide_apply_forces<<<blocks_per_grid, threads_per_block>>>(
      dev_local_real_quadrants[level], level, h_max);
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
