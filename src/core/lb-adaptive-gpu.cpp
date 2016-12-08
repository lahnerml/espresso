#include "config.hpp"
#include "lb-adaptive-gpu.hpp"
#include "lb-adaptive.hpp"
#include "lb-d3q19.hpp"

#include <assert.h>
#include <mpi.h>

int local_num_quadrants = 0;

// clang-format off
int local_num_real_quadrants_level[P8EST_MAXLEVEL] =
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
int local_num_virt_quadrants_level[P8EST_MAXLEVEL] =
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
// clang-format on

LB_Parameters lbpar = {
    // rho
    {0.0},
    // viscosity
    {0.0},
    // bulk_viscosity
    {-1.0},
    // agrid
    -1.0,
    // tau
    -1.0,
    // base level for calculation of tau
    -1,
    // max level
    P8EST_MAXLEVEL,
    // friction
    {0.0},
    // ext_force
    {0.0, 0.0, 0.0},
    // rho_lb_units
    {0.},
    // gamma_odd
    {0.},
    // gamma_even
    {0.},
    // is_TRT
    false,
    // resend_halo
    0};

LB_Model lbmodel = {19,      d3q19_lattice, d3q19_coefficients,
                    d3q19_w, NULL,          1. / 3.};

lbadapt_payload_t *dev_local_real_quadrants[P8EST_MAXLEVEL] =
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
lbadapt_payload_t *dev_local_virt_quadrants[P8EST_MAXLEVEL] =
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#if (!defined(FLATNOISE) && !defined(GAUSSRANDOMCUT) && !defined(GAUSSRANDOM))
#define FLATNOISE
#endif // (!defined(FLATNOISE) && !defined(GAUSSRANDOMCUT) &&
       // !defined(GAUSSRANDOM))

/** Flag indicating momentum exchange between particles and fluid */
int transfer_momentum = 0;

/** flag indicating if there is brownian motion */
int fluct;

double prefactors[P8EST_MAXLEVEL] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                     0., 0., 0., 0., 0., 0., 0., 0., 0.};
double gamma_shear[P8EST_MAXLEVEL] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0.};
double gamma_bulk[P8EST_MAXLEVEL] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                     0., 0., 0., 0., 0., 0., 0., 0., 0.};
/** relaxation of the odd kinetic modes */
double gamma_odd = 0.0;
/** relaxation of the even kinetic modes */
double gamma_even = 0.0;
/** amplitudes of the fluctuations of the modes */
double lb_phi[19];
/** amplitude of the fluctuations in the viscous coupling */
double lb_coupl_pref = 0.0;
/** amplitude of the fluctuations in the viscous coupling with gaussian random
 * numbers */
double lb_coupl_pref2 = 0.0;

int lbadapt_print_gpu_utilization(char *filename) {
  int len;
  /* strip file ending from filename (if given) */
  char *pos_file_ending = strpbrk(filename, ".");
  if (pos_file_ending != 0) {
    *pos_file_ending = '\0';
  } else {
    pos_file_ending = strpbrk(filename, "\0");
  }

  /* this is parallel io, i.e. we have to communicate the filename to all
   * other processes. */
  len = pos_file_ending - filename + 1;

  /* call mpi printing routine on all slaves and communicate the filename */
  mpi_call(mpi_lbadapt_vtk_print_gpu_utilization, -1, len);
  MPI_Bcast(filename, len, MPI_CHAR, 0, comm_cart);

  thread_block_container_t *a;
  a = P4EST_ALLOC(thread_block_container_t, p8est->local_num_quadrants);

  show_blocks_threads(a);
  lbadapt_vtk_context_t *c;
  p4est_locidx_t cells_per_patch =
      LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE;
  p4est_locidx_t num_cells = cells_per_patch * p8est->local_num_quadrants;
  sc_array_t *values_thread, *values_block;
  values_thread = sc_array_new_size(sizeof(double), num_cells);
  values_block = sc_array_new_size(sizeof(double), num_cells);

  double *block_ptr, *thread_ptr;
  for (int i = 0; i < p8est->local_num_quadrants; ++i) {
    block_ptr = (double *)sc_array_index(values_block, cells_per_patch * i);
    thread_ptr = (double *)sc_array_index(values_thread, cells_per_patch * i);
    int patch_count = 0;
    for (int patch_z = 1; patch_z <= LBADAPT_PATCHSIZE; ++patch_z) {
      for (int patch_y = 1; patch_y <= LBADAPT_PATCHSIZE; ++patch_y) {
        for (int patch_x = 1; patch_x <= LBADAPT_PATCHSIZE; ++patch_x) {
          block_ptr[patch_count] =
              (double)a[i].block_idx[patch_x][patch_y][patch_z];
          thread_ptr[patch_count] =
              (double)a[i].thread_idx[patch_x][patch_y][patch_z];
          ++patch_count;
        }
      }
    }
  }

  c = lbadapt_vtk_context_new(filename);
  c = lbadapt_vtk_write_header(c);
  c = lbadapt_vtk_write_cell_dataf(c, 1, 1, 1, 0, 1, 2, 0, "block",
                                   values_block, "thread", values_thread, c);
  lbadapt_vtk_write_footer(c);

  sc_array_destroy(values_thread);
  sc_array_destroy(values_block);
  P4EST_FREE(a);

  return 0;
}

lbadapt_vtk_context_t *lbadapt_vtk_context_new(const char *filename) {
  lbadapt_vtk_context_t *cont;

  assert(filename != NULL);

  /* Allocate, initialize the vtk context.  Important to zero all fields. */
  cont = P4EST_ALLOC_ZERO(lbadapt_vtk_context_t, 1);

  cont->filename = P4EST_STRDUP(filename);

  strcpy(cont->vtk_float_name, "Float64");

  return cont;
}

void lbadapt_vtk_context_destroy(lbadapt_vtk_context_t *context) {
  P4EST_ASSERT(context != NULL);

  /* since this function is called inside write_header and write_footer,
   * we cannot assume a consistent state of all member variables */

  P4EST_ASSERT(context->filename != NULL);
  P4EST_FREE(context->filename);

  /* deallocate node storage */
  P4EST_FREE(context->node_to_corner);

  /* Close all file pointers. */
  if (context->vtufile != NULL) {
    if (fclose(context->vtufile)) {
      P4EST_LERRORF(P8EST_STRING "_vtk: Error closing <%s>.\n",
                    context->vtufilename);
    }
    context->vtufile = NULL;
  }

  /* Close paraview master file */
  if (context->pvtufile != NULL) {
    /* Only the root process opens/closes these files. */
    P4EST_ASSERT(p8est->mpirank == 0);
    if (fclose(context->pvtufile)) {
      P4EST_LERRORF(P8EST_STRING "_vtk: Error closing <%s>.\n",
                    context->pvtufilename);
    }
    context->pvtufile = NULL;
  }

  /* Close visit master file */
  if (context->visitfile != NULL) {
    /* Only the root process opens/closes these files. */
    P4EST_ASSERT(p8est->mpirank == 0);
    if (fclose(context->visitfile)) {
      P4EST_LERRORF(P8EST_STRING "_vtk: Error closing <%s>.\n",
                    context->visitfilename);
    }
    context->visitfile = NULL;
  }

  /* Free context structure. */
  P4EST_FREE(context);
}

lbadapt_vtk_context_t *lbadapt_vtk_write_header(lbadapt_vtk_context_t *cont) {
  const lb_float intsize = 1.0 / P8EST_ROOT_LEN;
  int mpirank;
  const char *filename;
  const lb_float *v;
  const p4est_topidx_t *tree_to_vertex;
  p4est_topidx_t first_local_tree, last_local_tree;
  const p4est_locidx_t cells_per_patch =
      LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE;
  p4est_locidx_t Ncells, Ncorners;
  p8est_t *p4est;
  p8est_connectivity_t *connectivity;
  int retval;
  uint8_t *uint8_data;
  p4est_locidx_t *locidx_data;
  int xi, yi, j, k;
  int zi;
  double xyz[3]; /* 3 not P4EST_DIM */
  size_t num_quads, zz;
  p4est_topidx_t jt;
  p4est_topidx_t vt[P8EST_CHILDREN];
  p4est_locidx_t patch_count, quad_count, Npoints;
  p4est_locidx_t il, *ntc;
  double *float_data;
  sc_array_t *quadrants, *indeps;
  sc_array_t *trees;
  p8est_tree_t *tree;
  p8est_quadrant_t *quad;

  /* check a whole bunch of assertions, here and below */
  P4EST_ASSERT(cont != NULL);
  P4EST_ASSERT(!cont->writing);

  /* avoid uninitialized warning */
  for (k = 0; k < P8EST_CHILDREN; ++k) {
    vt[k] = -(k + 1);
  }

  /* from now on this context is officially in use for writing */
  cont->writing = 1;

  /* grab context variables */
  p4est = p8est;
  filename = cont->filename;
  P4EST_ASSERT(filename != NULL);

  /* grab details from the forest */
  P4EST_ASSERT(p4est != NULL);
  mpirank = p4est->mpirank;
  connectivity = p4est->connectivity;
  P4EST_ASSERT(connectivity != NULL);
  v = (lb_float *)connectivity->vertices;
  tree_to_vertex = connectivity->tree_to_vertex;

  SC_CHECK_ABORT(connectivity->num_vertices > 0,
                 "Must provide connectivity with vertex information");
  P4EST_ASSERT(v != NULL && tree_to_vertex != NULL);

  trees = p4est->trees;
  first_local_tree = p4est->first_local_tree;
  last_local_tree = p4est->last_local_tree;
  Ncells = cells_per_patch * p4est->local_num_quadrants;

  cont->num_corners = Ncorners = P8EST_CHILDREN * Ncells;
  cont->num_points = Npoints = Ncorners;
  cont->node_to_corner = ntc = NULL;
  indeps = NULL;

  /* Have each proc write to its own file */
  snprintf(cont->vtufilename, BUFSIZ, "%s_%04d.vtu", filename, mpirank);

  /* Use "w" for writing the initial part of the file.
   * For further parts, use "r+" and fseek so write_compressed succeeds.
   */
  cont->vtufile = fopen(cont->vtufilename, "wb");
  if (cont->vtufile == NULL) {
    P4EST_LERRORF("Could not open %s for output\n", cont->vtufilename);
    lbadapt_vtk_context_destroy(cont);
    return NULL;
  }

  fprintf(cont->vtufile, "<?xml version=\"1.0\"?>\n");
  fprintf(cont->vtufile, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\"");
  fprintf(cont->vtufile, " compressor=\"vtkZLibDataCompressor\"");
#ifdef SC_IS_BIGENDIAN
  fprintf(cont->vtufile, " byte_order=\"BigEndian\">\n");
#else
  fprintf(cont->vtufile, " byte_order=\"LittleEndian\">\n");
#endif
  fprintf(cont->vtufile, "  <UnstructuredGrid>\n");
  fprintf(cont->vtufile,
          "    <Piece NumberOfPoints=\"%lld\" NumberOfCells=\"%lld\">\n",
          (long long)Npoints, (long long)Ncells);
  fprintf(cont->vtufile, "      <Points>\n");

  float_data = P4EST_ALLOC(double, 3 * Npoints);

  /* write point position data */
  fprintf(cont->vtufile, "        <DataArray type=\"%s\" Name=\"Position\""
                         " NumberOfComponents=\"3\" format=\"%s\">\n",
          cont->vtk_float_name, "binary");
  int base;
  int root = P8EST_ROOT_LEN;

  double xyz_quad[3], xyz_patch[3];
  double patch_offset;
  /* loop over the trees */
  for (jt = first_local_tree, quad_count = 0; jt <= last_local_tree; ++jt) {
    tree = p8est_tree_array_index(trees, jt);
    quadrants = &tree->quadrants;
    num_quads = quadrants->elem_count;

    /* loop over the elements in tree and calculate vertex coordinates */
    for (zz = 0; zz < num_quads; ++zz, ++quad_count) {
      quad = p8est_quadrant_array_index(quadrants, zz);
      base = P8EST_QUADRANT_LEN(quad->level);
      patch_offset = ((double)base / (LBADAPT_PATCHSIZE * (double)root));

      // lower left corner of quadrant
      lbadapt_get_front_lower_left(p8est, jt, quad, xyz_quad);

      patch_count = 0;
      for (int patch_z = 0; patch_z < LBADAPT_PATCHSIZE; ++patch_z) {
        for (int patch_y = 0; patch_y < LBADAPT_PATCHSIZE; ++patch_y) {
          for (int patch_x = 0; patch_x < LBADAPT_PATCHSIZE; ++patch_x) {
            // lower left corner of patch
            xyz_patch[0] = xyz_quad[0] + patch_x * patch_offset;
            xyz_patch[1] = xyz_quad[1] + patch_y * patch_offset;
            xyz_patch[2] = xyz_quad[2] + patch_z * patch_offset;
            k = 0;
            // calculate remaining coordinates
            for (zi = 0; zi < 2; ++zi) {
              for (yi = 0; yi < 2; ++yi) {
                for (xi = 0; xi < 2; ++xi) {
                  xyz[0] = xyz_patch[0] + xi * patch_offset;
                  xyz[1] = xyz_patch[1] + yi * patch_offset;
                  xyz[2] = xyz_patch[2] + zi * patch_offset;

                  for (j = 0; j < 3; ++j) {
                    float_data[j +
                               3 * (k +
                                    P8EST_CHILDREN *
                                        (patch_count +
                                         cells_per_patch * quad_count))] =
                        xyz[j];
                  }
                  ++k; // count coordinates written up to now
                }
              }
            }
            ++patch_count; // count patches written up to now
          }
        }
      }
    }
    assert(k == P8EST_CHILDREN);
    assert(patch_count == cells_per_patch);
  }
  assert(P8EST_CHILDREN * cells_per_patch * quad_count == Npoints);

  fprintf(cont->vtufile, "          ");
  /* TODO: Don't allocate the full size of the array, only allocate
   * the chunk that will be passed to zlib and do this a chunk
   * at a time.
   */
  retval = sc_vtk_write_compressed(cont->vtufile, (char *)float_data,
                                   sizeof(*float_data) * 3 * Npoints);
  fprintf(cont->vtufile, "\n");
  if (retval) {
    P4EST_LERROR(P8EST_STRING "_vtk: Error encoding points\n");
    lbadapt_vtk_context_destroy(cont);
    P4EST_FREE(float_data);
    return NULL;
  }
  P4EST_FREE(float_data);

  fprintf(cont->vtufile, "        </DataArray>\n");
  fprintf(cont->vtufile, "      </Points>\n");
  fprintf(cont->vtufile, "      <Cells>\n");

  /* write connectivity data */
  fprintf(cont->vtufile, "        <DataArray type=\"%s\" Name=\"connectivity\""
                         " format=\"%s\">\n",
          P4EST_VTK_LOCIDX, "binary");

  fprintf(cont->vtufile, "          ");

  locidx_data = P4EST_ALLOC(p4est_locidx_t, Ncorners);
  for (il = 0; il < Ncorners; ++il) {
    locidx_data[il] = il;
  }
  retval = sc_vtk_write_compressed(cont->vtufile, (char *)locidx_data,
                                   sizeof(p4est_locidx_t) * Ncorners);
  P4EST_FREE(locidx_data);

  fprintf(cont->vtufile, "\n");
  if (retval) {
    P4EST_LERROR(P8EST_STRING "_vtk: Error encoding connectivity\n");
    lbadapt_vtk_context_destroy(cont);
    return NULL;
  }
  fprintf(cont->vtufile, "        </DataArray>\n");

  /* write offset data */
  fprintf(cont->vtufile, "        <DataArray type=\"%s\" Name=\"offsets\""
                         " format=\"%s\">\n",
          P4EST_VTK_LOCIDX, "binary");
  locidx_data = P4EST_ALLOC(p4est_locidx_t, Ncells);
  for (il = 1; il <= Ncells; ++il) {
    locidx_data[il - 1] = P8EST_CHILDREN * il;
  }

  fprintf(cont->vtufile, "          ");
  retval = sc_vtk_write_compressed(cont->vtufile, (char *)locidx_data,
                                   sizeof(p4est_locidx_t) * Ncells);
  fprintf(cont->vtufile, "\n");

  P4EST_FREE(locidx_data);

  if (retval) {
    P4EST_LERROR(P8EST_STRING "_vtk: Error encoding offsets\n");
    lbadapt_vtk_context_destroy(cont);
    return NULL;
  }
  fprintf(cont->vtufile, "        </DataArray>\n");

  /* write type data */
  fprintf(cont->vtufile, "        <DataArray type=\"UInt8\" Name=\"types\""
                         " format=\"%s\">\n",
          "binary");
  uint8_data = P4EST_ALLOC(uint8_t, Ncells);
  for (il = 0; il < Ncells; ++il) {
    uint8_data[il] = P4EST_VTK_CELL_TYPE;
  }

  fprintf(cont->vtufile, "          ");
  retval = sc_vtk_write_compressed(cont->vtufile, (char *)uint8_data,
                                   sizeof(*uint8_data) * Ncells);
  fprintf(cont->vtufile, "\n");

  P4EST_FREE(uint8_data);

  if (retval) {
    P4EST_LERROR(P8EST_STRING "_vtk: Error encoding types\n");
    lbadapt_vtk_context_destroy(cont);
    return NULL;
  }
  fprintf(cont->vtufile, "        </DataArray>\n");
  fprintf(cont->vtufile, "      </Cells>\n");

  if (ferror(cont->vtufile)) {
    P4EST_LERROR(P8EST_STRING "_vtk: Error writing header\n");
    lbadapt_vtk_context_destroy(cont);
    return NULL;
  }

  /* Only have the root write to the parallel vtk file */
  if (mpirank == 0) {
    snprintf(cont->pvtufilename, BUFSIZ, "%s.pvtu", filename);

    cont->pvtufile = fopen(cont->pvtufilename, "wb");
    if (!cont->pvtufile) {
      P4EST_LERRORF("Could not open %s for output\n", cont->pvtufilename);
      lbadapt_vtk_context_destroy(cont);
      return NULL;
    }

    fprintf(cont->pvtufile, "<?xml version=\"1.0\"?>\n");
    fprintf(cont->pvtufile,
            "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\"");
    fprintf(cont->pvtufile, " compressor=\"vtkZLibDataCompressor\"");
#ifdef SC_IS_BIGENDIAN
    fprintf(cont->pvtufile, " byte_order=\"BigEndian\">\n");
#else
    fprintf(cont->pvtufile, " byte_order=\"LittleEndian\">\n");
#endif

    fprintf(cont->pvtufile, "  <PUnstructuredGrid GhostLevel=\"0\">\n");
    fprintf(cont->pvtufile, "    <PPoints>\n");
    fprintf(cont->pvtufile, "      <PDataArray type=\"%s\" Name=\"Position\""
                            " NumberOfComponents=\"3\" format=\"%s\"/>\n",
            cont->vtk_float_name, "binary");
    fprintf(cont->pvtufile, "    </PPoints>\n");

    if (ferror(cont->pvtufile)) {
      P4EST_LERROR(P8EST_STRING "_vtk: Error writing parallel header\n");
      lbadapt_vtk_context_destroy(cont);
      return NULL;
    }

    /* Create a master file for visualization in Visit; this will be used
     * only in p4est_vtk_write_footer().
     */
    snprintf(cont->visitfilename, BUFSIZ, "%s.visit", filename);
    cont->visitfile = fopen(cont->visitfilename, "wb");
    if (!cont->visitfile) {
      P4EST_LERRORF("Could not open %s for output\n", cont->visitfilename);
      lbadapt_vtk_context_destroy(cont);
      return NULL;
    }
  }

  return cont;
}

lbadapt_vtk_context_t *
lbadapt_vtk_write_cell_scalar(lbadapt_vtk_context_t *cont,
                              const char *scalar_name, sc_array_t *values) {
  const p4est_locidx_t cells_per_patch =
      LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE;
  const p4est_locidx_t Ncells = cells_per_patch * p8est->local_num_quadrants;
  p4est_locidx_t il;
  int retval;
  double *float_data;

  P4EST_ASSERT(cont != NULL && cont->writing);

  /* Write cell data. */
  fprintf(cont->vtufile, "        <DataArray type=\"%s\" Name=\"%s\""
                         " format=\"%s\">\n",
          cont->vtk_float_name, scalar_name, "binary");

  float_data = P4EST_ALLOC(double, Ncells);
  for (il = 0; il < Ncells; ++il) {
    float_data[il] = (double)*((double *)sc_array_index(values, il));
  }

  fprintf(cont->vtufile, "          ");
  /* TODO: Don't allocate the full size of the array, only allocate
   * the chunk that will be passed to zlib and do this a chunk
   * at a time.
   */
  retval = sc_vtk_write_compressed(cont->vtufile, (char *)float_data,
                                   sizeof(*float_data) * Ncells);
  fprintf(cont->vtufile, "\n");

  P4EST_FREE(float_data);

  if (retval) {
    P4EST_LERROR(P8EST_STRING "_vtk: Error encoding scalar cell data\n");
    lbadapt_vtk_context_destroy(cont);
    return NULL;
  }
  fprintf(cont->vtufile, "        </DataArray>\n");

  if (ferror(cont->vtufile)) {
    P4EST_LERROR(P8EST_STRING "_vtk: Error writing cell scalar file\n");
    lbadapt_vtk_context_destroy(cont);
    return NULL;
  }

  return cont;
}

lbadapt_vtk_context_t *
lbadapt_vtk_write_cell_vector(lbadapt_vtk_context_t *cont,
                              const char *vector_name, sc_array_t *values) {
  const p4est_locidx_t cells_per_patch =
      LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE;
  const p4est_locidx_t Ncells = cells_per_patch * p8est->local_num_quadrants;
  p4est_locidx_t il;
  int retval;
  double *float_data;

  P4EST_ASSERT(cont != NULL && cont->writing);

  /* Write cell data. */
  fprintf(cont->vtufile, "        <DataArray type=\"%s\" Name=\"%s\""
                         " NumberOfComponents=\"3\" format=\"%s\">\n",
          cont->vtk_float_name, vector_name, "binary");

  float_data = P4EST_ALLOC(double, 3 * Ncells);
  for (il = 0; il < (3 * Ncells); ++il) {
    float_data[il] = (double)*((double *)sc_array_index(values, il));
  }

  fprintf(cont->vtufile, "          ");
  /* TODO: Don't allocate the full size of the array, only allocate
   * the chunk that will be passed to zlib and do this a chunk
   * at a time.
   */
  retval = sc_vtk_write_compressed(cont->vtufile, (char *)float_data,
                                   sizeof(*float_data) * 3 * Ncells);
  fprintf(cont->vtufile, "\n");

  P4EST_FREE(float_data);

  if (retval) {
    P4EST_LERROR(P8EST_STRING "_vtk: Error encoding scalar cell data\n");
    lbadapt_vtk_context_destroy(cont);
    return NULL;
  }
  fprintf(cont->vtufile, "        </DataArray>\n");

  if (ferror(cont->vtufile)) {
    P4EST_LERROR(P8EST_STRING "_vtk: Error writing cell scalar file\n");
    lbadapt_vtk_context_destroy(cont);
    return NULL;
  }

  return cont;
}

/** Write VTK cell data.
 *
 * This function exports custom cell data to the vtk file; it is functionally
 * the same as \b p4est_vtk_write_cell_dataf with the only difference being
 * that instead of a variable argument list, an initialized \a va_list is
 * passed as the last argument. The \a va_list is initialized from the variable
 * argument list of the calling function.
 *
 * \note This function is actually called from \b p4est_vtk_write_cell_dataf
 * and does all of the work.
 *
 * \param [in,out] cont    A vtk context created by \ref p4est_vtk_context_new.
 * \param [in] num_cell_scalars Number of point scalar datasets to output.
 * \param [in] num_cell_vectors Number of point vector datasets to output.
 * \param [in,out] ap      An initialized va_list used to access the
 *                         scalar/vector data.
 *
 * \return          On success, the context that has been passed in.
 *                  On failure, returns NULL and deallocates the context.
 */
static lbadapt_vtk_context_t *
lbadapt_vtk_write_cell_datav(lbadapt_vtk_context_t *cont, int write_tree,
                             int write_level, int write_rank, int wrap_rank,
                             int write_qid, int num_cell_scalars,
                             int num_cell_vectors, va_list ap) {
  /* This function needs to do nothing if there is no data. */
  if (!(write_tree || write_level || write_rank || wrap_rank ||
        num_cell_vectors || num_cell_vectors))
    return cont;

  const int mpirank = p8est->mpirank;
  int retval;
  int i, all = 0;
  int scalar_strlen, vector_strlen;

  sc_array_t *trees = p8est->trees;
  p8est_tree_t *tree;
  const p4est_topidx_t first_local_tree = p8est->first_local_tree;
  const p4est_topidx_t last_local_tree = p8est->last_local_tree;

  const p4est_locidx_t cells_per_patch =
      LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE;
  const p4est_locidx_t Ncells = cells_per_patch * p8est->local_num_quadrants;

  char cell_scalars[BUFSIZ], cell_vectors[BUFSIZ];
  const char *name, **names;
  sc_array_t **values;
  size_t num_quads, zz;
  sc_array_t *quadrants;
  p8est_quadrant_t *quad;
  uint8_t *uint8_data;
  p4est_locidx_t *locidx_data;
  p4est_topidx_t jt;
  p4est_locidx_t il;

  P4EST_ASSERT(cont != NULL && cont->writing);
  P4EST_ASSERT(wrap_rank >= 0);

  values = P4EST_ALLOC(sc_array_t *, num_cell_scalars + num_cell_vectors);
  names = P4EST_ALLOC(const char *, num_cell_scalars + num_cell_vectors);

  /* Gather cell data. */
  scalar_strlen = 0;
  cell_scalars[0] = '\0';
  for (i = 0; i < num_cell_scalars; ++all, ++i) {
    name = names[all] = va_arg(ap, const char *);
    retval = snprintf(cell_scalars + scalar_strlen, BUFSIZ - scalar_strlen,
                      "%s%s", i == 0 ? "" : ",", name);
    SC_CHECK_ABORT(retval > 0,
                   P8EST_STRING "_vtk: Error collecting cell scalars");
    scalar_strlen += retval;
    values[all] = va_arg(ap, sc_array_t *);

    /* Validate input. */
    SC_CHECK_ABORT(values[all]->elem_size == sizeof(double),
                   P8EST_STRING "_vtk: Error: incorrect cell scalar data type; "
                                "scalar data must contain lb_floats.");
    SC_CHECK_ABORT(values[all]->elem_count == (size_t)Ncells,
                   P8EST_STRING "_vtk: Error: incorrect cell scalar data "
                                "count; scalar data must contain exactly "
                                "p4est->local_num_quadrants lb_floats.");
  }

  vector_strlen = 0;
  cell_vectors[0] = '\0';
  for (i = 0; i < num_cell_vectors; ++all, ++i) {
    name = names[all] = va_arg(ap, const char *);
    retval = snprintf(cell_vectors + vector_strlen, BUFSIZ - vector_strlen,
                      "%s%s", i == 0 ? "" : ",", name);
    SC_CHECK_ABORT(retval > 0,
                   P8EST_STRING "_vtk: Error collecting cell vectors");
    vector_strlen += retval;
    values[all] = va_arg(ap, sc_array_t *);

    /* Validate input. */
    SC_CHECK_ABORT(values[all]->elem_size == sizeof(double),
                   P8EST_STRING "_vtk: Error: incorrect cell vector data type; "
                                "vector data must contain lb_floats.");
    SC_CHECK_ABORT(values[all]->elem_count == (size_t)Ncells * 3,
                   P8EST_STRING "_vtk: Error: incorrect cell vector data "
                                "count; vector data must contain exactly "
                                "3*p4est->local_num_quadrants lb_floats.");
  }

  /* Check for pointer variable marking the end of variable data input. */
  lbadapt_vtk_context_t *end = va_arg(ap, lbadapt_vtk_context_t *);
  SC_CHECK_ABORT(
      end == cont, P8EST_STRING
      "_vtk Error: the end of variable "
      "data must be specified by passing, as the last argument, the current "
      "lbadapt_vtk_context_t struct.");

  char vtkCellDataString[BUFSIZ] = "";
  int printed = 0;

  if (write_tree)
    printed +=
        snprintf(vtkCellDataString + printed, BUFSIZ - printed, "treeid");

  if (write_level)
    printed += snprintf(vtkCellDataString + printed, BUFSIZ - printed,
                        printed > 0 ? ",level" : "level");

  if (write_rank)
    printed += snprintf(vtkCellDataString + printed, BUFSIZ - printed,
                        printed > 0 ? ",mpirank" : "mpirank");

  if (write_qid)
    printed += snprintf(vtkCellDataString + printed, BUFSIZ - printed,
                        printed > 0 ? ",qid" : "qid");

  if (num_cell_scalars)
    printed += snprintf(vtkCellDataString + printed, BUFSIZ - printed,
                        printed > 0 ? ",%s" : "%s", cell_scalars);

  if (num_cell_vectors)
    printed += snprintf(vtkCellDataString + printed, BUFSIZ - printed,
                        printed > 0 ? ",%s" : "%s", cell_vectors);

  fprintf(cont->vtufile, "      <CellData Scalars=\"%s\">\n",
          vtkCellDataString);

  locidx_data = P4EST_ALLOC(p4est_locidx_t, Ncells);
  uint8_data = P4EST_ALLOC(uint8_t, Ncells);

  if (write_tree) {
    fprintf(cont->vtufile, "        <DataArray type=\"%s\" Name=\"treeid\""
                           " format=\"%s\">\n",
            P4EST_VTK_LOCIDX, "binary");
    for (il = 0, jt = first_local_tree; jt <= last_local_tree; ++jt) {
      tree = p8est_tree_array_index(trees, jt);
      num_quads = tree->quadrants.elem_count;
      for (zz = 0; zz < num_quads; ++zz) {
        for (int p = 0; p < cells_per_patch; ++p, ++il) {
          locidx_data[il] = (p4est_locidx_t)jt;
        }
      }
    }
    fprintf(cont->vtufile, "          ");
    retval = sc_vtk_write_compressed(cont->vtufile, (char *)locidx_data,
                                     sizeof(*locidx_data) * Ncells);
    fprintf(cont->vtufile, "\n");
    if (retval) {
      P4EST_LERROR(P8EST_STRING "_vtk: Error encoding types\n");
      lbadapt_vtk_context_destroy(cont);

      P4EST_FREE(values);
      P4EST_FREE(names);
      P4EST_FREE(locidx_data);
      P4EST_FREE(uint8_data);

      return NULL;
    }
    fprintf(cont->vtufile, "        </DataArray>\n");
    P4EST_ASSERT(il == Ncells);
  }

  if (write_level) {
    fprintf(cont->vtufile, "        <DataArray type=\"%s\" Name=\"level\""
                           " format=\"%s\">\n",
            "UInt8", "binary");
    for (il = 0, jt = first_local_tree; jt <= last_local_tree; ++jt) {
      tree = p8est_tree_array_index(trees, jt);
      quadrants = &tree->quadrants;
      num_quads = quadrants->elem_count;
      for (zz = 0; zz < num_quads; ++zz) {
        quad = p8est_quadrant_array_index(quadrants, zz);
        for (int p = 0; p < cells_per_patch; ++p, ++il) {
          uint8_data[il] = (uint8_t)quad->level;
        }
      }
    }

    fprintf(cont->vtufile, "          ");
    retval = sc_vtk_write_compressed(cont->vtufile, (char *)uint8_data,
                                     sizeof(*uint8_data) * Ncells);
    fprintf(cont->vtufile, "\n");

    P4EST_FREE(uint8_data);

    if (retval) {
      P4EST_LERROR(P8EST_STRING "_vtk: Error encoding types\n");
      lbadapt_vtk_context_destroy(cont);

      P4EST_FREE(values);
      P4EST_FREE(names);
      P4EST_FREE(locidx_data);

      return NULL;
    }
    fprintf(cont->vtufile, "        </DataArray>\n");
  }

  if (write_rank) {
    const int wrapped_rank = wrap_rank > 0 ? mpirank % wrap_rank : mpirank;

    fprintf(cont->vtufile, "        <DataArray type=\"%s\" Name=\"mpirank\""
                           " format=\"%s\">\n",
            P4EST_VTK_LOCIDX, "binary");
    for (il = 0; il < Ncells; ++il)
      locidx_data[il] = (p4est_locidx_t)wrapped_rank;

    fprintf(cont->vtufile, "          ");
    retval = sc_vtk_write_compressed(cont->vtufile, (char *)locidx_data,
                                     sizeof(*locidx_data) * Ncells);
    fprintf(cont->vtufile, "\n");

    if (retval) {
      P4EST_LERROR(P8EST_STRING "_vtk: Error encoding types\n");
      lbadapt_vtk_context_destroy(cont);

      P4EST_FREE(values);
      P4EST_FREE(names);

      return NULL;
    }
    fprintf(cont->vtufile, "        </DataArray>\n");
  }

  if (write_qid) {
    fprintf(cont->vtufile, "        <DataArray type=\"%s\" Name=\"qid\""
                           " format=\"%s\">\n",
            P4EST_VTK_LOCIDX, "binary");
    int offset = p8est->global_first_quadrant[mpirank];
    for (il = 0, jt = first_local_tree; jt <= last_local_tree; ++jt) {
      tree = p8est_tree_array_index(trees, jt);
      num_quads = tree->quadrants.elem_count;
      for (zz = 0; zz < num_quads; ++zz) {
        quad = p8est_quadrant_array_index(quadrants, zz);
        for (int p = 0; p < cells_per_patch; ++p, ++il) {
          locidx_data[il] =
              offset + (p4est_locidx_t)zz + tree->quadrants_offset;
        }
      }
    }
    fprintf(cont->vtufile, "          ");
    retval = sc_vtk_write_compressed(cont->vtufile, (char *)locidx_data,
                                     sizeof(*locidx_data) * Ncells);
    fprintf(cont->vtufile, "\n");

    P4EST_FREE(locidx_data);

    if (retval) {
      P4EST_LERROR(P8EST_STRING "_vtk: Error encoding types\n");
      lbadapt_vtk_context_destroy(cont);

      P4EST_FREE(values);
      P4EST_FREE(names);
      P4EST_FREE(locidx_data);
      P4EST_FREE(uint8_data);

      return NULL;
    }
    fprintf(cont->vtufile, "        </DataArray>\n");
    P4EST_ASSERT(il == Ncells);
  }

  if (ferror(cont->vtufile)) {
    P4EST_LERRORF(P8EST_STRING "_vtk: Error writing %s\n", cont->vtufilename);
    lbadapt_vtk_context_destroy(cont);

    P4EST_FREE(values);
    P4EST_FREE(names);

    return NULL;
  }

  all = 0;
  for (i = 0; i < num_cell_scalars; ++all, ++i) {
    cont = lbadapt_vtk_write_cell_scalar(cont, names[all], values[all]);
    SC_CHECK_ABORT(cont != NULL,
                   P8EST_STRING "_vtk: Error writing cell scalars");
  }

  for (i = 0; i < num_cell_vectors; ++all, ++i) {
    cont = lbadapt_vtk_write_cell_vector(cont, names[all], values[all]);
    SC_CHECK_ABORT(cont != NULL,
                   P8EST_STRING "_vtk: Error writing cell vectors");
  }

  fprintf(cont->vtufile, "      </CellData>\n");

  P4EST_FREE(values);

  if (ferror(cont->vtufile)) {
    P4EST_LERRORF(P8EST_STRING "_vtk: Error writing %s\n", cont->vtufilename);
    lbadapt_vtk_context_destroy(cont);

    P4EST_FREE(names);

    return NULL;
  }

  /* Only have the root write to the parallel vtk file */
  if (mpirank == 0) {
    fprintf(cont->pvtufile, "    <PCellData Scalars=\"%s\">\n",
            vtkCellDataString);

    if (write_tree)
      fprintf(cont->pvtufile,
              "      "
              "<PDataArray type=\"%s\" Name=\"treeid\" format=\"%s\"/>\n",
              P4EST_VTK_LOCIDX, "binary");

    if (write_level)
      fprintf(cont->pvtufile,
              "      "
              "<PDataArray type=\"%s\" Name=\"level\" format=\"%s\"/>\n",
              "UInt8", "binary");

    if (write_rank)
      fprintf(cont->pvtufile,
              "      "
              "<PDataArray type=\"%s\" Name=\"mpirank\" format=\"%s\"/>\n",
              P4EST_VTK_LOCIDX, "binary");

    if (write_qid)
      fprintf(cont->pvtufile,
              "      "
              "<PDataArray type=\"%s\" Name=\"qid\" format=\"%s\"/>\n",
              P4EST_VTK_LOCIDX, "binary");

    all = 0;
    for (i = 0; i < num_cell_scalars; ++all, i++)
      fprintf(cont->pvtufile,
              "      "
              "<PDataArray type=\"%s\" Name=\"%s\" format=\"%s\"/>\n",
              cont->vtk_float_name, names[all], "binary");

    for (i = 0; i < num_cell_vectors; ++all, i++)
      fprintf(cont->pvtufile,
              "      "
              "<PDataArray type=\"%s\" Name=\"%s\" NumberOfComponents=\"3\""
              " format=\"%s\"/>\n",
              cont->vtk_float_name, names[all], "binary");

    fprintf(cont->pvtufile, "    </PCellData>\n");

    if (ferror(cont->pvtufile)) {
      P4EST_LERROR(P8EST_STRING "_vtk: Error writing parallel header\n");
      lbadapt_vtk_context_destroy(cont);

      P4EST_FREE(names);

      return NULL;
    }
  }

  P4EST_FREE(names);

  return cont;
}

lbadapt_vtk_context_t *
lbadapt_vtk_write_cell_dataf(lbadapt_vtk_context_t *cont, int write_tree,
                             int write_level, int write_rank, int wrap_rank,
                             int write_qid, int num_cell_scalars,
                             int num_cell_vectors, ...) {
  va_list ap;

  P4EST_ASSERT(cont != NULL && cont->writing);
  P4EST_ASSERT(num_cell_scalars >= 0 && num_cell_vectors >= 0);

  va_start(ap, num_cell_vectors);
  cont = lbadapt_vtk_write_cell_datav(cont, write_tree, write_level, write_rank,
                                      wrap_rank, write_qid, num_cell_scalars,
                                      num_cell_vectors, ap);
  va_end(ap);

  return cont;
}

int lbadapt_vtk_write_footer(lbadapt_vtk_context_t *cont) {
  int p;
  int procRank = p8est->mpirank;
  int numProcs = p8est->mpisize;

  P4EST_ASSERT(cont != NULL && cont->writing);

  fprintf(cont->vtufile, "    </Piece>\n");
  fprintf(cont->vtufile, "  </UnstructuredGrid>\n");
  fprintf(cont->vtufile, "</VTKFile>\n");

  if (ferror(cont->vtufile)) {
    P4EST_LERROR(P8EST_STRING "_vtk: Error writing footer\n");
    lbadapt_vtk_context_destroy(cont);
    return -1;
  }

  /* Only have the root write to the parallel vtk file */
  if (procRank == 0) {
    fprintf(cont->visitfile, "!NBLOCKS %d\n", numProcs);

    /* Do not write paths to parallel vtk file, because they will be in the same
     * directory as the vtk files written by each process */
    char *file_basename;
    file_basename = strrchr(cont->filename, '/');
    if (file_basename != NULL) {
      ++file_basename;
    } else {
      file_basename = cont->filename;
    }

    /* Write data about the parallel pieces into both files */
    for (p = 0; p < numProcs; ++p) {
      fprintf(cont->pvtufile, "    <Piece Source=\"%s_%04d.vtu\"/>\n",
              file_basename, p);
      fprintf(cont->visitfile, "%s_%04d.vtu\n", file_basename, p);
    }
    fprintf(cont->pvtufile, "  </PUnstructuredGrid>\n");
    fprintf(cont->pvtufile, "</VTKFile>\n");

    if (ferror(cont->pvtufile)) {
      P4EST_LERROR(P8EST_STRING "_vtk: Error writing parallel footer\n");
      lbadapt_vtk_context_destroy(cont);
      return -1;
    }

    if (ferror(cont->visitfile)) {
      P4EST_LERROR(P8EST_STRING "_vtk: Error writing parallel footer\n");
      lbadapt_vtk_context_destroy(cont);
      return -1;
    }
  }

  /* Destroy context structure. */
  lbadapt_vtk_context_destroy(cont);

  return 0;
}
