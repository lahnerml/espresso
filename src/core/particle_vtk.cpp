
#include <cstdio>
#include <cstring>
#include <libgen.h>
#include <mpi.h>
#include "particle_vtk.hpp"
#include "integrate.hpp"
#include "communication.hpp"
#include "cells.hpp"
#include "p4est_utils.hpp"

void write_parallel_particle_vtk(char *filename) {
  /* strip file ending from filename (if given) */
  char *pos_file_ending;
  pos_file_ending = strrchr(filename, '.');
  if (pos_file_ending != nullptr) {
    *pos_file_ending = '\0';
  }
  int len = static_cast<int>(strlen(filename));
  ++len;

  /* call mpi printing routine on all slaves and communicate the filename */
  mpi_call(mpi_write_particle_vtk, -1, len);

  MPI_Bcast(filename, len, MPI_CHAR, 0, comm_cart);
  write_particle_vtk(filename);
}

void write_particle_vtk(char *filename) {
  char fname[1024];
  char filename_copy[1024];
  // node 0 writes the header file
  if (!this_node) {
    sprintf(fname, "%s.pvtp", filename);
    FILE *h = fopen(fname, "w");
    fprintf(h, "<?xml version=\"1.0\"?>\n");
    fprintf(h, "<VTKFile type=\"PPolyData\" version=\"0.1\" "
               "byte_order=\"LittleEndian\">\n\t");
    fprintf(h, "<PPolyData GhostLevel=\"0\">\n\t\t<PPoints>\n\t\t\t");
    fprintf(h, "<PDataArray type=\"Float32\" Name=\"Position\" "
               "NumberOfComponents=\"3\" format=\"ascii\"/>\n");
    fprintf(h, "\t\t</PPoints>\n\t\t");
    fprintf(h, "<PPointData Scalars=\"mpirank,part_id,cell_id\" "
               "Vectors=\"velocity\">\n\t\t\t");
    fprintf(h, "<PDataArray type=\"Int32\" Name=\"mpirank\" "
               "format=\"ascii\"/>\n\t\t\t");
    fprintf(h, "<PDataArray type=\"Int32\" Name=\"part_id\" "
               "format=\"ascii\"/>\n\t\t\t");
    fprintf(h, "<PDataArray type=\"Int32\" Name=\"cell_id\" "
               "format=\"ascii\"/>\n\t\t\t");
    fprintf(h, "<PDataArray type=\"Float32\" Name=\"velocity\" "
               "NumberOfComponents=\"3\" format=\"ascii\"/>\n\t\t");
    fprintf(h, "</PPointData>\n");
    for (int p = 0; p < n_nodes; ++p) {
      /* We want to write the basename of each processes file, since
       * the basename function could modify its argument, we create a
       * temporary copy. */
      sprintf(filename_copy, "%s", filename);
      char *filename_basename = basename(filename_copy);
      fprintf(h, "\t\t<Piece Source=\"%s_%04i.vtp\"/>\n", filename_basename, p);
    }
    fprintf(h, "\t</PPolyData>\n</VTKFile>\n");
    fclose(h);
  }
  // write the actual parallel particle files
  sprintf(fname, "%s_%04i.vtp", filename, this_node);
  int num_p = 0;
  for (int c = 0; c < local_cells.n; c++) {
    num_p += local_cells.cell[c]->n;
  }
  FILE *h = fopen(fname, "w");
  fprintf(h, "<?xml version=\"1.0\"?>\n");
  fprintf(h, "<VTKFile type=\"PolyData\" version=\"0.1\" "
             "byte_order=\"LittleEndian\">\n\t");
  fprintf(h,
          "<PolyData>\n\t\t<Piece NumberOfPoints=\"%i\" NumberOfVerts=\"0\" ",
          num_p);
  fprintf(h, "NumberOfLines=\"0\" NumberOfStrips=\"0\" "
             "NumberOfPolys=\"0\">\n\t\t\t<Points>\n\t\t\t\t");
  fprintf(h, "<DataArray type=\"Float32\" Name=\"Position\" "
             "NumberOfComponents=\"3\" format=\"ascii\">\n");
  for (int c = 0; c < local_cells.n; c++) {
    int np = local_cells.cell[c]->n;
    Particle *part = local_cells.cell[c]->part;
    for (int p = 0; p < np; ++p) {
#ifdef DD_P4EST
      std::array <double, 3> t_pos = boxl_to_treecoords_copy(part[p].r.p.data());
#else
      std::array <double, 3> t_pos = part[p].r.p;
#endif
      fprintf(h, "\t\t\t\t\t%le %le %le\n", t_pos[0], t_pos[1], t_pos[2]);
    }
  }
  fprintf(h, "\t\t\t\t</DataArray>\n\t\t\t</Points>\n\t\t\t");
  fprintf(h, "<PointData Scalars=\"mpirank,part_id,cell_id\" "
             "Vectors=\"velocity\">\n\t\t\t\t");
  fprintf(h, "<DataArray type=\"Int32\" Name=\"mpirank\" "
             "format=\"ascii\">\n\t\t\t\t\t");
  for (int c = 0; c < local_cells.n; c++) {
    int np = local_cells.cell[c]->n;
    for (int p = 0; p < np; ++p) {
      fprintf(h, "%i ", this_node);
    }
  }
  fprintf(h, "\n\t\t\t\t</DataArray>\n\t\t\t\t<DataArray type=\"Int32\" "
             "Name=\"part_id\" format=\"ascii\">\n\t\t\t\t\t");
  for (int c = 0; c < local_cells.n; c++) {
    int np = local_cells.cell[c]->n;
    Particle *part = local_cells.cell[c]->part;
    for (int p = 0; p < np; ++p) {
      fprintf(h, "%i ", part[p].p.identity);
    }
  }
  fprintf(h, "\n\t\t\t\t</DataArray>\n\t\t\t\t<DataArray type=\"Int32\" "
             "Name=\"cell_id\" format=\"ascii\">\n\t\t\t\t\t");
  for (int c = 0; c < local_cells.n; c++) {
    int np = local_cells.cell[c]->n;
    for (int p = 0; p < np; ++p) {
      fprintf(h, "%i ", c);
    }
  }
  fprintf(h, "\n\t\t\t\t</DataArray>\n\t\t\t\t<DataArray type=\"Float32\" "
             "Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n");
  for (int c = 0; c < local_cells.n; c++) {
    int np = local_cells.cell[c]->n;
    Particle *part = local_cells.cell[c]->part;
    for (int p = 0; p < np; ++p) {
      fprintf(h, "\t\t\t\t\t%le %le %le\n", part[p].m.v[0] / time_step,
              part[p].m.v[1] / time_step, part[p].m.v[2] / time_step);
    }
  }
  fprintf(h, "\t\t\t\t</DataArray>\n\t\t\t</PointData>\n");
  fprintf(h, "\t\t</Piece>\n\t</PolyData>\n</VTKFile>\n");
  fclose(h);
}

