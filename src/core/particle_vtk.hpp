
#ifndef PARTICLE_VTK_INCLUDED
#define PARTICLE_VTK_INCLUDED

void write_parallel_particle_vtk(char* filename);

/** Generate vtk files for particles.
 *
 * @param filename     Filename for output-files
 */
void write_particle_vtk(char* filename);

#endif
