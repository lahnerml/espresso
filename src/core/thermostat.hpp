/*
  Copyright (C) 2010,2011,2012,2013,2014,2015,2016 The ESPResSo project
  Copyright (C) 2002,2003,2004,2005,2006,2007,2008,2009,2010 
    Max-Planck-Institute for Polymer Research, Theory Group
  
  This file is part of ESPResSo.
  
  ESPResSo is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  
  ESPResSo is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>. 
*/
#ifndef _THERMOSTAT_H
#define _THERMOSTAT_H
/** \file thermostat.hpp 

*/

#include <cmath>
#include "utils.hpp"
#include "debug.hpp"
#include "particle_data.hpp"
#include "random.hpp"
#include "global.hpp"
#include "integrate.hpp"
#include "cells.hpp"
#include "lb.hpp"
#include "dpd.hpp"
#include "virtual_sites.hpp"

/** \name Thermostat switches*/
/************************************************************/
/*@{*/

#define THERMO_OFF        0
#define THERMO_LANGEVIN   1
#define THERMO_DPD        2
#define THERMO_NPT_ISO    4
#define THERMO_LB         8
#define THERMO_INTER_DPD  16
#define THERMO_GHMC       32
#define THERMO_CPU        64
#define THERMO_SD         128
#define THERMO_BD         256
/*@}*/

// Handle switching of noise function flat vs Gaussian
#if (!defined(FLATNOISE) && !defined(GAUSSRANDOMCUT) && !defined(GAUSSRANDOM))
#define FLATNOISE
#endif

#if defined (FLATNOISE)
  #define noise (d_random() -0.5)
#elif defined (GAUSSRANDOMCUT)
  #define noise gaussian_random_cut()
#elif defined (GAUSSRANDOM)
  #define noise gaussian_random()
#else
 #error "No noise function defined"
#endif




/************************************************
 * exported variables
 ************************************************/

/** Switch determining which thermostat to use. This is a or'd value
    of the different possible thermostats (defines: \ref THERMO_OFF,
    \ref THERMO_LANGEVIN, \ref THERMO_DPD \ref THERMO_NPT_ISO). If it
    is zero all thermostats are switched off and the temperature is
    set to zero.  */
extern int thermo_switch;

/** temperature. */
extern double temperature;

/** Langevin friction coefficient gamma. */
extern double langevin_gamma;

/** Langevin friction coefficient gamma. */
#ifndef ROTATIONAL_INERTIA
extern double langevin_gamma_rotation;
#else
extern double langevin_gamma_rotation[3];
#endif

/** Langevin for translations */
extern bool langevin_trans;

/** Langevin for rotations */
extern bool langevin_rotate;

/** Friction coefficient for nptiso-thermostat's inline-function friction_therm0_nptiso */
extern double nptiso_gamma0;
/** Friction coefficient for nptiso-thermostat's inline-function friction_thermV_nptiso */
extern double nptiso_gammav;

/** Number of NVE-MD steps in GHMC Cycle*/
extern int ghmc_nmd;
/** Phi parameter for GHMC partial momenum update step */
extern double ghmc_phi;

#ifdef USE_FLOWFIELD
/** Flow field langevin parameter */
extern double langevin_pref3;

/** Flow field itself */
extern std::vector<double> velu, velv, velw;

#define FLOWFIELD_SIZE 65
// Overflow is unlikely (who has that much memory). But prevent silent
// execution or segmentation faults because of accidentally wrongly set
// flowfield sizes.
#define CBRT_SIZE_MAX 2642245UL
#if (FLOWFIELD_SIZE > CBRT_SIZE_MAX)
#  error Flowfield size too great for size_t
#endif

#define veluu(i, j, k) (velu[FLOWFIELD_SIZE * FLOWFIELD_SIZE * i + FLOWFIELD_SIZE * j + k])
#define velvv(i, j, k) (velv[FLOWFIELD_SIZE * FLOWFIELD_SIZE * i + FLOWFIELD_SIZE * j + k])
#define velww(i, j, k) (velw[FLOWFIELD_SIZE * FLOWFIELD_SIZE * i + FLOWFIELD_SIZE * j + k])
#endif // USE_FLOWFIELD


/************************************************
 * functions
 ************************************************/


/** initialize constants of the thermostat on
    start of integration */
void thermo_init();

/** very nasty: if we recalculate force when leaving/reentering the integrator,
    a(t) and a((t-dt)+dt) are NOT equal in the vv algorithm. The random
    numbers are drawn twice, resulting in a different variance of the random force.
    This is corrected by additional heat when restarting the integrator here.
    Currently only works for the Langevin thermostat, although probably also others
    are affected.
*/
void thermo_heat_up();

/** pendant to \ref thermo_heat_up */
void thermo_cool_down();
/** Get current temperature for CPU thermostat */
int get_cpu_temp();

/** Start the CPU thermostat */
void set_cpu_temp(int temp);

/** locally defined funcion to find Vx. In case of LEES_EDWARDS, that is relative to the LE shear frame
    @param i      coordinate index
    @param vel    velocity vector
    @param pos    position vector
    @return       adjusted (or not) i^th velocity coordinate */
inline double le_frameV(int i, double *vel, double *pos)
{
#ifdef LEES_EDWARDS

   if( i == 0 ){
       double relY  = pos[1] * box_l_i[1] - 0.5;
       return( vel[0] - relY * lees_edwards_rate );
   }

#endif

   return vel[i];
}

#ifdef NPT
/** add velocity-dependend noise and friction for NpT-sims to the particle's velocity 
    @param dt_vj  j-component of the velocity scaled by time_step dt 
    @return       j-component of the noise added to the velocity, also scaled by dt (contained in prefactors) */
inline double friction_therm0_nptiso(double dt_vj) {
  extern double nptiso_pref1, nptiso_pref2;
  if(thermo_switch & THERMO_NPT_ISO)
    return ( nptiso_pref1*dt_vj + nptiso_pref2*noise );
  
  return 0.0;
}

/** add p_diff-dependend noise and friction for NpT-sims to \ref nptiso_struct::p_diff */
inline double friction_thermV_nptiso(double p_diff) {
  extern double nptiso_pref3, nptiso_pref4;
  if(thermo_switch & THERMO_NPT_ISO)   
    return ( nptiso_pref3*p_diff + nptiso_pref4*noise );
  return 0.0;
}
#endif


#ifdef USE_FLOWFIELD
inline void fluid_velocity(double pos[3], double vfxyz[3])
{
  // Velocities at the corners of the grid cell particle p is in.
  double u000, u100, u010, u001, u101, u011, u110, u111;
  double v000, v100, v010, v001, v101, v011, v110, v111;
  double w000, w100, w010, w001, w101, w011, w110, w111;

  // Flowfield is defined on vertices, i.e. has FLOWFIELD_SIZE - 1 cells.
  std::array<double, 3> ff_cellsize = {{ box_l[0] / (FLOWFIELD_SIZE - 1),
                                         box_l[1] / (FLOWFIELD_SIZE - 1),
                                         box_l[2] / (FLOWFIELD_SIZE - 1)}};
  // Cell index and upper cell index
  // In combination 000, 001, 010, ... all 8 neighboring grid points of p
  int x0 = pos[0] / ff_cellsize[0];
  int x1 = x0 + 1;

  int y0 = pos[1] / ff_cellsize[1];
  int y1 = y0 + 1;

  int z0 = pos[2] / ff_cellsize[2];
  int z1 = z0 + 1;

  // Interpolation weights
  double wx = (pos[0] - (x0 * ff_cellsize[0])) / ff_cellsize[0];
  double wy = (pos[1] - (y0 * ff_cellsize[1])) / ff_cellsize[1];
  double wz = (pos[2] - (z0 * ff_cellsize[2])) / ff_cellsize[2];

  // Correct for oob particles (only correct for slightly oob particles).
  // In parallel they cannot be far oob because otherwise they would be
  // transferred to the corresponding process.
  if (pos[0] >= box_l[0]) {
    x0 = 0;
    x1 = 1;
  } else if (pos[0] < 0) {
    x0 = FLOWFIELD_SIZE - 2;
    x1 = FLOWFIELD_SIZE - 1;
  }

  if (pos[1] >= box_l[1]) {
    y0 = 0;
    y1 = 1;
  } else if (pos[1] < 0) {
    y0 = FLOWFIELD_SIZE - 2;
    y1 = FLOWFIELD_SIZE - 1;
  }

  if (pos[2] >= box_l[2]) {
    z0 = 0;
    z1 = 1;
  } else if (pos[2] < 0) {
    z0 = FLOWFIELD_SIZE - 2;
    z1 = FLOWFIELD_SIZE - 1;
  }

  // Look up the values of the 8 points of the grid

  // HERE I,J,K VALUES ARE IN REVERSE ORDER TO MAINTAIN THE CORRECT
  // CONFIGURATION !!!
  // DOUBLE CHECK WHEN THE DNS DATASET CHANGED
  u000 = veluu(z0, x0, y0);
  u100 = veluu(z1, x0, y0);
  u010 = veluu(z0, x1, y0);
  u001 = veluu(z0, x0, y1);
  u101 = veluu(z1, x0, y1);
  u011 = veluu(z0, x1, y1);
  u110 = veluu(z1, x1, y0);
  u111 = veluu(z1, x1, y1);

  v000 = velvv(z0, x0, y0);
  v100 = velvv(z1, x0, y0);
  v010 = velvv(z0, x1, y0);
  v001 = velvv(z0, x0, y1);
  v101 = velvv(z1, x0, y1);
  v011 = velvv(z0, x1, y1);
  v110 = velvv(z1, x1, y0);
  v111 = velvv(z1, x1, y1);

  w000 = velww(z0, x0, y0);
  w100 = velww(z1, x0, y0);
  w010 = velww(z0, x1, y0);
  w001 = velww(z0, x0, y1);
  w101 = velww(z1, x0, y1);
  w011 = velww(z0, x1, y1);
  w110 = velww(z1, x1, y0);
  w111 = velww(z1, x1, y1);

  // Compute the velocity components u, v, w at point
  vfxyz[0] = u000 * (1 - wx) * (1 - wy) * (1 - wz) +
             u100 * wx * (1 - wy) * (1 - wz) + u010 * (1 - wx) * wy * (1 - wz) +
             u001 * (1 - wx) * (1 - wy) * wz + u101 * wx * (1 - wy) * wz +
             u011 * (1 - wx) * wy * wz + u110 * wx * wy * (1 - wz) +
             u111 * wx * wy * wz;
  vfxyz[1] = v000 * (1 - wx) * (1 - wy) * (1 - wz) +
             v100 * wx * (1 - wy) * (1 - wz) + v010 * (1 - wx) * wy * (1 - wz) +
             v001 * (1 - wx) * (1 - wy) * wz + v101 * wx * (1 - wy) * wz +
             v011 * (1 - wx) * wy * wz + v110 * wx * wy * (1 - wz) +
             v111 * wx * wy * wz;
  vfxyz[2] = w000 * (1 - wx) * (1 - wy) * (1 - wz) +
             w100 * wx * (1 - wy) * (1 - wz) + w010 * (1 - wx) * wy * (1 - wz) +
             w001 * (1 - wx) * (1 - wy) * wz + w101 * wx * (1 - wy) * wz +
             w011 * (1 - wx) * wy * wz + w110 * wx * wy * (1 - wz) +
             w111 * wx * wy * wz;
}
#endif // USE_FLOWFIELD

/** overwrite the forces of a particle with
    the friction term, i.e. \f$ F_i= -\gamma v_i + \xi_i\f$.
*/
inline void friction_thermo_langevin(Particle *p)
{
  extern double langevin_pref1, langevin_pref2;
  
  double langevin_pref1_temp, langevin_pref2_temp, langevin_temp_coeff;

#ifdef MULTI_TIMESTEP
  extern double langevin_pref1_small;
#ifndef LANGEVIN_PER_PARTICLE
  extern double langevin_pref2_small;
#endif /* LANGEVIN_PER_PARTICLE */
#endif /* MULTI_TIMESTEP */

  int j;
  double switch_trans = 1.0;
  if ( langevin_trans == false )
  {
    switch_trans = 0.0;
  }

  // Virtual sites related decision making
#ifdef VIRTUAL_SITES
#ifndef VIRTUAL_SITES_THERMOSTAT
      // In this case, virtual sites are NOT thermostated 
  if (ifParticleIsVirtual(p))
    {
      for (j=0;j<3;j++)
        p->f.f[j]=0;
  
      return;
    }
#endif /* VIRTUAL_SITES_THERMOSTAT */
#ifdef THERMOSTAT_IGNORE_NON_VIRTUAL
      // In this case NON-virtual particles are NOT thermostated
  if (!ifParticleIsVirtual(p))
    {
      for (j=0;j<3;j++)
        p->f.f[j]=0;
  
      return;
    }
#endif /* THERMOSTAT_IGNORE_NON_VIRTUAL */
#endif /* VIRTUAL_SITES */

  // Get velocity effective in the thermostatting
  double velocity[3];
  for (int i = 0; i < 3; i++) {
    // Particle velocity
    velocity[i] = p->m.v[i];
#ifdef USE_FLOWFIELD
    double vfxyz[3];
    // Fluid velocity
    fluid_velocity(p->r.p, vfxyz);
    velocity[i] -= langevin_pref3 * vfxyz[i];
#endif // USE_FLOWFIELD

    #ifdef ENGINE
      // In case of the engine feature, the velocity is relaxed
      // towards a swimming velocity oriented parallel to the
      // particles director
      velocity[i] -= (p->swim.v_swim*time_step)*p->r.quatu[i];
    #endif

    // Local effective velocity for leeds-edwards boundary conditions
    velocity[i]=le_frameV(i,velocity,p->r.p);
  } // for
  
  // Determine prefactors for the friction and the noise term 

  // first, set defaults
  langevin_pref1_temp = langevin_pref1;
  langevin_pref2_temp = langevin_pref2;

  // Override defaults if per-particle values for T and gamma are given 
#ifdef LANGEVIN_PER_PARTICLE  
    // If a particle-specific gamma is given
#if defined (FLATNOISE)
  langevin_temp_coeff = 24.0;
#elif defined (GAUSSRANDOMCUT) || defined (GAUSSRANDOM)
  langevin_temp_coeff = 2.0;
#else
#error No Noise defined
#endif

    if(p->p.gamma >= 0.) 
    {
      langevin_pref1_temp = -p->p.gamma/time_step;
      // Is a particle-specific temperature also specified?
      if(p->p.T >= 0.)
        langevin_pref2_temp = sqrt(langevin_temp_coeff*p->p.T*p->p.gamma/time_step);
      else
        // Default temperature but particle-specific gamma
        langevin_pref2_temp = sqrt(langevin_temp_coeff*temperature*p->p.gamma/time_step);

    } // particle specific gamma
    else 
    {
      langevin_pref1_temp = -langevin_gamma/time_step;
      // No particle-specific gamma, but is there particle-specific temperature
      if(p->p.T >= 0.)
        langevin_pref2_temp = sqrt(langevin_temp_coeff*p->p.T*langevin_gamma/time_step);
      else
        // Defaut values for both
        langevin_pref2_temp = langevin_pref2;
    }
#endif /* LANGEVIN_PER_PARTICLE */

  // Multi-timestep handling
  // This has to be last, as it may set the prefactors to 0.
  #ifdef MULTI_TIMESTEP
    if (smaller_time_step > 0.) {
      langevin_pref1_temp *= time_step/smaller_time_step;
      if (p->p.smaller_timestep==1 && current_time_step_is_small==1) 
        langevin_pref2_temp *= sqrt(time_step/smaller_time_step);
      else if (p->p.smaller_timestep != current_time_step_is_small) {
        langevin_pref1_temp  = 0.;
        langevin_pref2_temp  = 0.;
      }
    }
#endif /* MULTI_TIMESTEP */

  
  // Do the actual thermostatting
  for ( j = 0 ; j < 3 ; j++) 
  {
    #ifdef EXTERNAL_FORCES
      // If individual coordinates are fixed, set force to 0.
      if ((p->p.ext_flag & COORD_FIXED(j)))
        p->f.f[j] = 0;
      else	
    #endif
    {
      // Apply the force
      p->f.f[j] = langevin_pref1_temp*velocity[j] + switch_trans*langevin_pref2_temp*noise;
    }
  } // END LOOP OVER ALL COMPONENTS


  // printf("%d: %e %e %e %e %e %e\n",p->p.identity, p->f.f[0],p->f.f[1],p->f.f[2], p->m.v[0],p->m.v[1],p->m.v[2]);
  ONEPART_TRACE(if(p->p.identity==check_id) fprintf(stderr,"%d: OPT: LANG f = (%.3e,%.3e,%.3e)\n",this_node,p->f.f[0],p->f.f[1],p->f.f[2]));
  THERMO_TRACE(fprintf(stderr,"%d: Thermo: P %d: force=(%.3e,%.3e,%.3e)\n",this_node,p->p.identity,p->f.f[0],p->f.f[1],p->f.f[2]));
}

#ifdef ROTATION
/** set the particle torques to the friction term, i.e. \f$\tau_i=-\gamma w_i + \xi_i\f$.
    The same friction coefficient \f$\gamma\f$ is used as that for translation.
*/
inline void friction_thermo_langevin_rotation(Particle *p)
{
#ifndef ROTATIONAL_INERTIA
  extern double langevin_pref2_rotation;
  double langevin_pref1_temp, langevin_pref2_temp;
#else
  extern double langevin_pref2_rotation[3];
  double langevin_pref1_temp[3], langevin_pref2_temp[3];
#endif
  double langevin_temp_coeff;

  int j;
  double switch_rotate = 1.0;
  if ( langevin_rotate == false )
  {
    switch_rotate = 0.0;
  }

  // first, set defaults
#ifndef ROTATIONAL_INERTIA
  langevin_pref1_temp = langevin_gamma_rotation;
  langevin_pref2_temp = langevin_pref2_rotation;
#else
  for ( j = 0 ; j < 3 ; j++)
  {
	  langevin_pref1_temp[j] = langevin_gamma_rotation[j];
	  langevin_pref2_temp[j] = langevin_pref2_rotation[j];
  }
#endif

  // Override defaults if per-particle values for T and gamma are given
#ifdef LANGEVIN_PER_PARTICLE
    // If a particle-specific gamma is given
#if defined (FLATNOISE)
  langevin_temp_coeff = 24.0;
#elif defined (GAUSSRANDOMCUT) || defined (GAUSSRANDOM)
  langevin_temp_coeff = 2.0;
#else
#error No Noise defined
#endif
#ifndef ROTATIONAL_INERTIA
    if(p->p.gamma_rot >= 0.)
    {
      langevin_pref1_temp = p->p.gamma_rot;
      // Is a particle-specific temperature also specified?
      if(p->p.T >= 0.)
        langevin_pref2_temp = sqrt(langevin_temp_coeff*p->p.T*p->p.gamma_rot/time_step);
      else
        // Default temperature but particle-specific gamma
        langevin_pref2_temp = sqrt(langevin_temp_coeff*temperature*p->p.gamma_rot/time_step);

    } // particle specific gamma
    else
    {
      langevin_pref1_temp = langevin_gamma_rotation;
      // No particle-specific gamma, but is there particle-specific temperature
      if(p->p.T >= 0.)
        langevin_pref2_temp = sqrt(langevin_temp_coeff*p->p.T*langevin_gamma_rotation/time_step);
      else
        // Default values for both
        langevin_pref2_temp = langevin_pref2_rotation;
    }
#else
    for ( j = 0 ; j < 3 ; j++)
    if(p->p.gamma_rot[j] >= 0.)
    {
      langevin_pref1_temp[j] = p->p.gamma_rot[j];
      // Is a particle-specific temperature also specified?
      if(p->p.T >= 0.)
        langevin_pref2_temp[j] = sqrt(langevin_temp_coeff*p->p.T*p->p.gamma_rot[j]/time_step);
      else
        // Default temperature but particle-specific gamma
        langevin_pref2_temp[j] = sqrt(langevin_temp_coeff*temperature*p->p.gamma_rot[j]/time_step);

    } // particle specific gamma
    else
    {
      langevin_pref1_temp[j] = langevin_gamma_rotation[j];
      // No particle-specific gamma, but is there particle-specific temperature
      if(p->p.T >= 0.)
        langevin_pref2_temp[j] = sqrt(langevin_temp_coeff*p->p.T*langevin_gamma_rotation[j]/time_step);
      else
        // Default values for both
        langevin_pref2_temp[j] = langevin_pref2_rotation[j];
    }
#endif // ROTATIONAL_INERTIA
#endif /* LANGEVIN_PER_PARTICLE */


  // Rotational degrees of virtual sites are thermostatted,
  // so no switching here


  // Here the thermostats happens
  for ( j = 0 ; j < 3 ; j++) 
  {
#ifdef ROTATIONAL_INERTIA
    p->f.torque[j] = -langevin_pref1_temp[j]*p->m.omega[j] + switch_rotate*langevin_pref2_temp[j]*noise;
#else
    p->f.torque[j] = -langevin_pref1_temp*p->m.omega[j] + switch_rotate*langevin_pref2_temp*noise;
#endif
  }

  ONEPART_TRACE(if(p->p.identity==check_id) fprintf(stderr,"%d: OPT: LANG f = (%.3e,%.3e,%.3e)\n",this_node,p->f.f[0],p->f.f[1],p->f.f[2]));
  THERMO_TRACE(fprintf(stderr,"%d: Thermo: P %d: force=(%.3e,%.3e,%.3e)\n",this_node,p->p.identity,p->f.f[0],p->f.f[1],p->f.f[2]));
}


#endif // ROTATION


#undef noise
#endif
