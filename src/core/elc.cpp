/*
  Copyright (C) 2010,2012,2013,2014 The ESPResSo project
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
/** \file elc.cpp
 *
 *  For more information about ELC, see \ref elc.hpp "elc.hpp".
 */
#include <cmath>
#include <mpi.h>
#include "utils.hpp"
#include "communication.hpp"
#include "particle_data.hpp"
#include "interaction_data.hpp"
#include "cells.hpp"
#include "elc.hpp"
#include "mmm-common.hpp"
#include "pressure.hpp"
#include "p3m.hpp"
#include "errorhandling.hpp"

#ifdef P3M

// #define CHECKPOINTS
// #define LOG_FORCES

/****************************************
 * LOCAL DEFINES
 ****************************************/

/** Largest reasonable cutoff for far formula */
#define MAXIMAL_FAR_CUT 50

/****************************************
 * LOCAL VARIABLES
 ****************************************/

/** \name Inverse box dimensions and derived constants */
/*@{*/
static double ux, ux2, uy, uy2, uz, height_inverse;
/*@}*/

ELC_struct elc_params = { 1e100, 10, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0 };

/****************************************
 * LOCAL ARRAYS
 ****************************************/

/** \name Product decomposition data organization
    For the cell blocks
    it is assumed that the lower blocks part is in the lower half.
    This has to have positive sign, so that has to be first. */
/*@{*/
#define POQESP 0
#define POQECP 1
#define POQESM 2
#define POQECM 3

#define PQESSP 0
#define PQESCP 1
#define PQECSP 2
#define PQECCP 3
#define PQESSM 4
#define PQESCM 5
#define PQECSM 6
#define PQECCM 7
/*@}*/

/** number of local particles, equals the size of \ref elc::partblk. */
static int n_localpart = 0;

/** temporary buffers for product decomposition */
static double *partblk = NULL;
/** collected data from the other cells */
static double gblcblk[8];

/** structure for storing of sin and cos values */
typedef struct {
  double s, c;
} SCCache;

/** \name sin/cos caching */ 
/*@{*/
static SCCache *scxcache = NULL;
static int    n_scxcache;  
static SCCache *scycache = NULL;
static int    n_scycache;  
/*@}*/

/****************************************
 * LOCAL FUNCTIONS
 ****************************************/

/** \name sin/cos storage */
/*@{*/
static void prepare_scx_cache();
static void prepare_scy_cache();
/*@}*/
/** \name common code */
/*@{*/
static void distribute(int size);
/*@}*/
/** \name p=0 per frequency code */
/*@{*/
static void setup_P(int p, double omega);
static void add_P_force();
static double   P_energy(double omega);
/*@}*/
/** \name q=0 per frequency code */
/*@{*/
static void setup_Q(int q, double omega);
static void add_Q_force();
static double   Q_energy(double omega);
/*@}*/
/** \name p,q <> 0 per frequency code */
/*@{*/
static void setup_PQ(int p, int q, double omega);
static void add_PQ_force(int p, int q, double omega);
static double   PQ_energy(double omega);
static void add_dipole_force();
static double dipole_energy();
static double z_energy();
static void add_z_force();
/*@}*/

/* COMMON */
/**********/

void ELC_setup_constants()
{
  ux  = 1/box_l[0];
  ux2 = ux*ux;
  uy  = 1/box_l[1];
  uy2 = uy*uy;  
  uz  = 1/box_l[2];

  height_inverse = 1/elc_params.h;
}

/* SC Cache */
/************/
static void prepare_scx_cache()
{
  int np, c, i, ic, freq, o;
  double pref, arg;
  Particle *part;
  
  for (freq = 1; freq <= n_scxcache; freq++) {
    pref = C_2PI*ux*freq;
    o = (freq-1)*n_localpart;
    ic = 0;
    for (c = 0; c < local_cells.n; c++) {
      np   = local_cells.cell[c]->n;
      part = local_cells.cell[c]->part;
      for (i = 0; i < np; i++) {
	arg = pref*part[i].r.p[0];
	scxcache[o + ic].s = sin(arg);
	scxcache[o + ic].c = cos(arg);
	ic++;
      }
    }
  }
}

static void prepare_scy_cache()
{
  int np, c, i, ic, freq, o;
  double pref, arg;
  Particle *part;
  
  for (freq = 1; freq <= n_scycache; freq++) {
    pref = C_2PI*uy*freq;
    o = (freq-1)*n_localpart;
    ic = 0;
    for (c = 0; c < local_cells.n; c++) {
      np   = local_cells.cell[c]->n;
      part = local_cells.cell[c]->part;
      for (i = 0; i < np; i++) {
	arg = pref*part[i].r.p[1];
	scycache[o + ic].s = sin(arg);
	scycache[o + ic].c = cos(arg);
	ic++;
      }
    }
  }
}

/*****************************************************************/
/* data distribution */
/*****************************************************************/

inline void clear_vec(double *pdc, int size)
{
  int i;
  for (i = 0; i < size; i++)
    pdc[i] = 0;
}

inline void copy_vec(double *pdc_d, double *pdc_s, int size)
{
  int i;
  for (i = 0; i < size; i++)
    pdc_d[i] = pdc_s[i];
}

inline void add_vec(double *pdc_d, double *pdc_s1, double *pdc_s2, int size)
{
  int i;
  for (i = 0; i < size; i++)
    pdc_d[i] = pdc_s1[i] + pdc_s2[i];
}

inline void addscale_vec(double *pdc_d, double scale, double *pdc_s1, double *pdc_s2, int size)
{
  int i;
  for (i = 0; i < size; i++)
    pdc_d[i] = scale*pdc_s1[i] + pdc_s2[i];
}

inline void scale_vec(double scale, double *pdc, int size)
{
  int i;
  for (i = 0; i < size; i++)
    pdc[i] *= scale;
}

inline double *block(double *p, int index, int size)
{
  return &p[index*size];
}

void distribute(int size)
{
  double send_buf[8];
  copy_vec(send_buf, gblcblk, size);
  MPI_Allreduce(send_buf, gblcblk, size, MPI_DOUBLE, MPI_SUM, comm_cart);
}

#ifdef CHECKPOINTS
static void checkpoint(char *text, int p, int q, int e_size)
{
  int c, i;
  fprintf(stderr, "%d: %s %d %d\n", this_node, text, p, q);

  fprintf(stderr, "partblk\n");
  for (c = 0; c < n_localpart; c++) {
    fprintf(stderr, "%d", c);    
    for (i = 0; i < e_size; i++)
      fprintf(stderr, " %10.3g", block(partblk, c, 2*e_size)[i]);
    fprintf(stderr, " m");
    for (i = 0; i < e_size; i++)
      fprintf(stderr, " %10.3g", block(partblk, c, 2*e_size)[i + e_size]);
    fprintf(stderr, "\n");
  }
  fprintf(stderr, "\n");

  fprintf(stderr, "gblcblk\n");
  for (i = 0; i < e_size; i++)
    fprintf(stderr, " %10.3g", gblcblk[i]);
  fprintf(stderr, " m");
  for (i = 0; i < e_size; i++)
    fprintf(stderr, " %10.3g", gblcblk[i + e_size]);
  fprintf(stderr, "\n");
}

#else
#define checkpoint(text,p,q,size)
#endif

#ifdef LOG_FORCES
static void clear_log_forces(char *where)
{
  int np, c, i, j;
  Particle *part;

  fprintf(stderr, "%s\n", where);
  for (c = 0; c < local_cells.n; c++) {
    np   = local_cells.cell[c]->n;
    part = local_cells.cell[c]->part;
    for (i = 0; i < np; i++) {
      fprintf(stderr, "%d %g %g %g\n", part[i].p.identity,
	      part[i].f.f[0], part[i].f.f[1], part[i].f.f[2]);
      for (j = 0; j < 3; j++)
	part[i].f.f[j] = 0;
    }
  }
}
#else
#define clear_log_forces(w)
#endif

/*****************************************************************/
/* dipole terms */
/*****************************************************************/

static void add_dipole_force()
{
  double pref = coulomb.prefactor*4*M_PI*ux*uy*uz;
  int size = 3;

  /* for nonneutral systems, this shift gives the background contribution
     (rsp. for this shift, the DM of the background is zero) */
  double shift = 0.5*box_l[2];
  double field_tot = 0;

  // collect moments

  gblcblk[0] = 0; // sum q_i (z_i - L/2)
  gblcblk[1] = 0; // sum q_i z_i
  gblcblk[2] = 0; // sum q_i

  for (int c = 0; c < local_cells.n; c++) {
    int np = local_cells.cell[c]->n;
    Particle *part = local_cells.cell[c]->part;
    for (int i = 0; i < np; i++) {
      gblcblk[0] += part[i].p.q*(part[i].r.p[2] - shift);
      gblcblk[1] += part[i].p.q*part[i].r.p[2];
      gblcblk[2] += part[i].p.q;

      if(elc_params.dielectric_contrast_on) {
	if(part[i].r.p[2]<elc_params.space_layer) {
	  gblcblk[0] +=elc_params.di_mid_bot*part[i].p.q*(-part[i].r.p[2] - shift);
          gblcblk[2] +=elc_params.di_mid_bot*part[i].p.q;
        }  
	if(part[i].r.p[2]>(elc_params.h-elc_params.space_layer)) {
	  gblcblk[0] +=elc_params.di_mid_top*part[i].p.q*(2*elc_params.h-part[i].r.p[2] - shift);
          gblcblk[2] +=elc_params.di_mid_top*part[i].p.q;
        }
      }
    }
  }
  gblcblk[0] *= pref;
  gblcblk[1] *= pref*height_inverse/uz;
  gblcblk[2] *= pref;
  
  distribute(size);

  // Yeh+Berkowitz dipole term
  field_tot = gblcblk[0];

  // Const. potential contribution
  if (elc_params.const_pot_on) {
    field_induced = gblcblk[1];
    field_applied = elc_params.pot_diff * height_inverse;
    field_tot -= field_applied + field_induced;
  }

  for (int c = 0; c < local_cells.n; c++) {
    int np = local_cells.cell[c]->n;
    Particle *part = local_cells.cell[c]->part;
    for (int i = 0; i < np; i++) {
      part[i].f.f[2] -= field_tot*part[i].p.q;

      if (!elc_params.neutralize) {
        // SUBTRACT the forces of the P3M homogeneous neutralizing background
	part[i].f.f[2] += gblcblk[2]*part[i].p.q*(part[i].r.p[2] - shift);
      }
    }
  }
}

static double dipole_energy()
{
  double pref = coulomb.prefactor*2*M_PI*ux*uy*uz;
  int size = 7;
  /* for nonneutral systems, this shift gives the background contribution
     (rsp. for this shift, the DM of the background is zero) */
  double shift = 0.5*box_l[2];

  // collect moments

  gblcblk[0] = 0; // sum q_i               primary box
  gblcblk[1] = 0; // sum q_i               boundary layers
  gblcblk[2] = 0; // sum q_i (z_i - L/2)   primary box
  gblcblk[3] = 0; // sum q_i (z_i - L/2)   boundary layers
  gblcblk[4] = 0; // sum q_i (z_i - L/2)^2 primary box
  gblcblk[5] = 0; // sum q_i (z_i - L/2)^2 boundary layers
  gblcblk[6] = 0; // sum q_i z_i           primary box
  
  for (int c = 0; c < local_cells.n; c++) {
    int np = local_cells.cell[c]->n;
    Particle *part = local_cells.cell[c]->part;
    for (int i = 0; i < np; i++) {
      gblcblk[0] += part[i].p.q;
      gblcblk[2] += part[i].p.q*(part[i].r.p[2] - shift);
      gblcblk[4] += part[i].p.q*(SQR(part[i].r.p[2]- shift));
      gblcblk[6] += part[i].p.q*part[i].r.p[2];
      
      if(elc_params.dielectric_contrast_on) {
	if(part[i].r.p[2]<elc_params.space_layer) {
	  gblcblk[1] +=elc_params.di_mid_bot*part[i].p.q;
	  gblcblk[3] +=elc_params.di_mid_bot*part[i].p.q*(-part[i].r.p[2] - shift);
	  gblcblk[5] +=elc_params.di_mid_bot*part[i].p.q*(SQR(-part[i].r.p[2] - shift));
        }
	if(part[i].r.p[2]>(elc_params.h-elc_params.space_layer)) {
	  gblcblk[1] +=elc_params.di_mid_top*part[i].p.q;
	  gblcblk[3] +=elc_params.di_mid_top*part[i].p.q*(2*elc_params.h-part[i].r.p[2] - shift);
	  gblcblk[5] +=elc_params.di_mid_top*part[i].p.q*(SQR(2*elc_params.h-part[i].r.p[2] - shift));
        }
      }
    }
  }

  distribute(size);

  // Yeh + Berkowitz term
  double eng = 2*pref*(SQR(gblcblk[2])+gblcblk[2]*gblcblk[3]);
  
  if (!elc_params.neutralize) {
    // SUBTRACT the energy of the P3M homogeneous neutralizing background
    eng += 2*pref*(-gblcblk[0]*gblcblk[4] - (.25 - .5/3.)*SQR(gblcblk[0]*box_l[2]));
  }
  
  if (elc_params.dielectric_contrast_on) {
    if (elc_params.const_pot_on) {
      // zero potential difference contribution
      eng += pref*height_inverse/uz*SQR(gblcblk[6]);
      // external potential shift contribution
      eng += elc_params.pot_diff * height_inverse * gblcblk[6];
    }

    /* counter the P3M homogeneous background contribution to the
       boundaries.  We never need that, since a homogeneous background
       spanning the artifical boundary layers is aphysical. */
    eng += pref*(-(gblcblk[1]*gblcblk[4] + gblcblk[0]*gblcblk[5])
                 - (1. - 2./3.)*gblcblk[0]*gblcblk[1]*SQR(box_l[2]));
  }

  return this_node == 0 ? eng : 0;
}

/*****************************************************************/

inline double image_sum_b(double q, double z) 
{
  double shift = 0.5*box_l[2];
  double fac=elc_params.di_mid_top*elc_params.di_mid_bot;
  double image_sum=(q/(1.0-fac)*(z-2.0*fac*box_l[2]/(1.0-fac)))-q*shift/(1-fac);
  return image_sum;
}

inline double image_sum_t(double q, double z) 
{
  double shift = 0.5*box_l[2];
  double fac=elc_params.di_mid_top*elc_params.di_mid_bot;
  double image_sum=(q/(1.0-fac)*(z+2.0*fac*box_l[2]/(1.0-fac)))-q*shift/(1-fac);
  return image_sum;
}

/*****************************************************************/
static double z_energy() 
{
  double pref = coulomb.prefactor*2*M_PI*ux*uy;
  int size = 4;

  double eng=0;
  /* for nonneutral systems, this shift gives the background contribution
     (rsp. for this shift, the DM of the background is zero) */
  double shift = 0.5*box_l[2];
  
  if(elc_params.dielectric_contrast_on) {
    if(elc_params.const_pot_on) {
      clear_vec(gblcblk, size);
      for (int c = 0; c < local_cells.n; c++) {
        int np = local_cells.cell[c]->n;
        Particle *part = local_cells.cell[c]->part;
        for (int i = 0; i < np; i++) {
          gblcblk[0] += part[i].p.q; 
          gblcblk[1] += part[i].p.q*(part[i].r.p[2] - shift);
          if(elc_params.dielectric_contrast_on) {
            if(part[i].r.p[2]<elc_params.space_layer) {
              gblcblk[2] -= elc_params.di_mid_bot*part[i].p.q;
              gblcblk[3] -= elc_params.di_mid_bot*part[i].p.q*(-part[i].r.p[2] - shift);
            }
            if (part[i].r.p[2]>(elc_params.h-elc_params.space_layer)) {
              gblcblk[2] += elc_params.di_mid_top*part[i].p.q;
              gblcblk[3] += elc_params.di_mid_top*part[i].p.q*(2*elc_params.h - part[i].r.p[2] - shift);
            }
          }      
        }
      }
    }
    else {
      double delta = elc_params.di_mid_top*elc_params.di_mid_bot;
      double fac_delta_mid_bot = elc_params.di_mid_bot /(1 - delta);
      double fac_delta_mid_top = elc_params.di_mid_top /(1 - delta);
      double fac_delta         = delta                 /(1 - delta);

      clear_vec(gblcblk, size);
      for (int c = 0; c < local_cells.n; c++) {
        int np   = local_cells.cell[c]->n;
        Particle *part = local_cells.cell[c]->part;
        for (int i = 0; i < np; i++) {
          gblcblk[0] += part[i].p.q;
          gblcblk[1] += part[i].p.q*(part[i].r.p[2] - shift);
          if(elc_params.dielectric_contrast_on) {
            if(part[i].r.p[2]<elc_params.space_layer) {
              gblcblk[2] += fac_delta*(elc_params.di_mid_bot+1)*part[i].p.q; 
              gblcblk[3] += part[i].p.q*(image_sum_b(elc_params.di_mid_bot*delta,-(2*elc_params.h+part[i].r.p[2]))
                                         +image_sum_b(delta,-(2*elc_params.h-part[i].r.p[2])));
            } else {
              gblcblk[2] += fac_delta_mid_bot*(1+elc_params.di_mid_top)*part[i].p.q; 
              gblcblk[3] += part[i].p.q*(image_sum_b(elc_params.di_mid_bot,-part[i].r.p[2])
                                         +image_sum_b(delta,-(2*elc_params.h-part[i].r.p[2])));
            }
            if(part[i].r.p[2]>(elc_params.h-elc_params.space_layer)) {
              //note the minus sign here which is required due to |z_i-z_j|
              gblcblk[2] -= fac_delta*(elc_params.di_mid_top+1)*part[i].p.q; 
              gblcblk[3] -= part[i].p.q*(image_sum_t(elc_params.di_mid_top*delta,4*elc_params.h-part[i].r.p[2])
                                         +image_sum_t(delta,2*elc_params.h+part[i].r.p[2]));
            } else {
              //note the minus sign here which is required due to |z_i-z_j|
              gblcblk[2] -= fac_delta_mid_top*(1+elc_params.di_mid_bot)*part[i].p.q; 	  
              gblcblk[3] -= part[i].p.q*(image_sum_t(elc_params.di_mid_top,2*elc_params.h-part[i].r.p[2])
                                         +image_sum_t(delta,2*elc_params.h+part[i].r.p[2]));
            }
          }    
        }
      }
    }
  }
  distribute(size);

  if (this_node == 0)
    eng -= pref*(gblcblk[1]*gblcblk[2]-gblcblk[0]*gblcblk[3]); 

  return eng;
}

/*****************************************************************/
static void add_z_force() 
{
  double pref = coulomb.prefactor*2*M_PI*ux*uy;
  int size = 1;

  if(elc_params.dielectric_contrast_on) {
    if (elc_params.const_pot_on) {
      clear_vec(gblcblk, size);
      for (int c = 0; c < local_cells.n; c++) {
        int np = local_cells.cell[c]->n;
        Particle *part = local_cells.cell[c]->part;
        /* just counter the 2 pi |z| contribution stemming from P3M */
        for (int i = 0; i < np; i++) {
          if(part[i].r.p[2]<elc_params.space_layer)
            gblcblk[0] -= elc_params.di_mid_bot*part[i].p.q;
          if(part[i].r.p[2]>(elc_params.h-elc_params.space_layer))
            gblcblk[0] += elc_params.di_mid_top*part[i].p.q;
        }
      }
    }
    else {
      double delta = elc_params.di_mid_top*elc_params.di_mid_bot;
      double fac_delta_mid_bot = elc_params.di_mid_bot /(1 - delta);
      double fac_delta_mid_top = elc_params.di_mid_top /(1 - delta);
      double fac_delta         = delta                 /(1 - delta);
        
      clear_vec(gblcblk, size);
      for (int c = 0; c < local_cells.n; c++) {
        int np = local_cells.cell[c]->n;
        Particle *part = local_cells.cell[c]->part;
        for (int i = 0; i < np; i++) {
          if(part[i].r.p[2]<elc_params.space_layer) {
            gblcblk[0] += fac_delta*(elc_params.di_mid_bot+1)*part[i].p.q;
          } else {
            gblcblk[0] += fac_delta_mid_bot*(1+elc_params.di_mid_top)*part[i].p.q;
          }
          
          if(part[i].r.p[2]>(elc_params.h-elc_params.space_layer)) {
            //note the minus sign here which is required due to |z_i-z_j|
            gblcblk[0] -= fac_delta*(elc_params.di_mid_top + 1)*part[i].p.q;
          } else {
            //note the minus sign here which is required due to |z_i-z_j|
            gblcblk[0] -= fac_delta_mid_top*(1+elc_params.di_mid_bot)*part[i].p.q;
          }
        }
      }
    }
    
    gblcblk[0] *= pref;
    
    distribute(size);
    
    for (int c = 0; c < local_cells.n; c++) {
      int np = local_cells.cell[c]->n;
      Particle *part = local_cells.cell[c]->part;
      for (int i = 0; i < np; i++) {
        part[i].f.f[2] += gblcblk[0]*part[i].p.q; 
      }
    }
  }
}

/*****************************************************************/
/* PoQ exp sum */
/*****************************************************************/

static void setup_P(int p, double omega)
{
  int np, c, i, ic, o = (p-1)*n_localpart;
  Particle *part;
  double pref = -coulomb.prefactor*4*M_PI*ux*uy/(exp(omega*box_l[2]) - 1);
  double pref_di = coulomb.prefactor*4*M_PI*ux*uy; 
  double e;
  int size = 4;
  double lclimgebot[4],lclimgetop[4],lclimge[4];  
  double fac_delta_mid_bot=1,fac_delta_mid_top=1,fac_delta=1;
  double scale=1;

  if(elc_params.dielectric_contrast_on) {
    double fac_elc=1.0/(1-elc_params.di_mid_top*elc_params.di_mid_bot*exp(-omega*2*elc_params.h));
    fac_delta_mid_bot=elc_params.di_mid_bot*fac_elc;
    fac_delta_mid_top=elc_params.di_mid_top*fac_elc; 
    fac_delta=fac_delta_mid_bot*elc_params.di_mid_top;
  }

  clear_vec(lclimge, size); 
  clear_vec(gblcblk, size);

  ic = 0;
  for (c = 0; c < local_cells.n; c++) {
    np   = local_cells.cell[c]->n;
    part = local_cells.cell[c]->part;
    for (i = 0; i < np; i++) {
      e = exp(omega*part[i].r.p[2]);

      partblk[size*ic + POQESM] = part[i].p.q*scxcache[o + ic].s/e;
      partblk[size*ic + POQESP] = part[i].p.q*scxcache[o + ic].s*e;
      partblk[size*ic + POQECM] = part[i].p.q*scxcache[o + ic].c/e;
      partblk[size*ic + POQECP] = part[i].p.q*scxcache[o + ic].c*e;
      
      add_vec(gblcblk, gblcblk, block(partblk, ic, size), size);
      
      if(elc_params.dielectric_contrast_on) {
	if(part[i].r.p[2]<elc_params.space_layer) { //handle the lower case first
	  //negative sign is okay here as the image is located at -part[i].r.p[2]
	  
	  e= exp(-omega*part[i].r.p[2]);
	  
	  scale = part[i].p.q*elc_params.di_mid_bot;
	  
	  lclimgebot[POQESM]=scxcache[o + ic].s/e;
	  lclimgebot[POQESP]=scxcache[o + ic].s*e;	
	  lclimgebot[POQECM]=scxcache[o + ic].c/e;
	  lclimgebot[POQECP]=scxcache[o + ic].c*e;
	  
	  addscale_vec(gblcblk, scale, lclimgebot, gblcblk, size);
	  
	  e = ( exp(omega*(-part[i].r.p[2] - 2*elc_params.h  ))*elc_params.di_mid_bot +
		exp(omega*( part[i].r.p[2] - 2*elc_params.h )) )*fac_delta;  
	  
	} else {
	  
	  e =( exp(omega*(-part[i].r.p[2] )) +
	       exp(omega*( part[i].r.p[2] - 2*elc_params.h ))*elc_params.di_mid_top )*fac_delta_mid_bot;    
	}
	
	lclimge[POQESP]+= part[i].p.q*scxcache[o + ic].s*e;
	lclimge[POQECP]+= part[i].p.q*scxcache[o + ic].c*e;
	
	
	if(part[i].r.p[2]>(elc_params.h-elc_params.space_layer)) { //handle the upper case now
	  
	  e=exp(omega*(2*elc_params.h-part[i].r.p[2]));
      
	  scale =part[i].p.q*elc_params.di_mid_top;      
	  
	  lclimgetop[POQESM]=scxcache[o + ic].s/e;
	  lclimgetop[POQESP]=scxcache[o + ic].s*e;	
	  lclimgetop[POQECM]=scxcache[o + ic].c/e;
	  lclimgetop[POQECP]=scxcache[o + ic].c*e;
	  
	  addscale_vec(gblcblk, scale, lclimgetop, gblcblk, size);
	  
	  e = ( exp(omega*( part[i].r.p[2] -4*elc_params.h ))*elc_params.di_mid_top +
		exp(omega*(-part[i].r.p[2] -2*elc_params.h )) )*fac_delta; 
	  
	} else {
	  
	  e = ( exp(omega*( +part[i].r.p[2] -2*elc_params.h )) +
		exp(omega*( -part[i].r.p[2] -2*elc_params.h ))*elc_params.di_mid_bot )*fac_delta_mid_top;
	}
	
	lclimge[POQESM]+= part[i].p.q*scxcache[o + ic].s*e;
	lclimge[POQECM]+= part[i].p.q*scxcache[o + ic].c*e;
      }
      
      ic++;
    }
  }
 
  scale_vec(pref, gblcblk, size);
 
  if(elc_params.dielectric_contrast_on) {
    scale_vec(pref_di, lclimge, size);
    add_vec(gblcblk, gblcblk, lclimge, size);
  }
}

static void setup_Q(int q, double omega)
{
  int np, c, i, ic, o = (q-1)*n_localpart;
  Particle *part;
  double pref = -coulomb.prefactor*4*M_PI*ux*uy/(exp(omega*box_l[2]) - 1);
  double pref_di = coulomb.prefactor*4*M_PI*ux*uy; 
  double e;
  int size = 4;
  double lclimgebot[4],lclimgetop[4],lclimge[4];
  double fac_delta_mid_bot=1,fac_delta_mid_top=1,fac_delta=1;
  double scale=1;

  if(elc_params.dielectric_contrast_on) {
    double fac_elc=1.0/(1-elc_params.di_mid_top*elc_params.di_mid_bot*exp(-omega*2*elc_params.h)); 
    fac_delta_mid_bot=elc_params.di_mid_bot*fac_elc; 
    fac_delta_mid_top=elc_params.di_mid_top*fac_elc; 
    fac_delta=fac_delta_mid_bot*elc_params.di_mid_top;
  }

  clear_vec(lclimge, size); 
  clear_vec(gblcblk, size);
  ic = 0;
  for (c = 0; c < local_cells.n; c++) {
    np   = local_cells.cell[c]->n;
    part = local_cells.cell[c]->part;
    for (i = 0; i < np; i++) {
      e = exp(omega*part[i].r.p[2]);

      partblk[size*ic + POQESM] = part[i].p.q*scycache[o + ic].s/e;
      partblk[size*ic + POQESP] = part[i].p.q*scycache[o + ic].s*e;
      partblk[size*ic + POQECM] = part[i].p.q*scycache[o + ic].c/e;
      partblk[size*ic + POQECP] = part[i].p.q*scycache[o + ic].c*e;
      
      add_vec(gblcblk, gblcblk, block(partblk, ic, size), size);
      
      if(elc_params.dielectric_contrast_on) {
	if(part[i].r.p[2]<elc_params.space_layer) { //handle the lower case first
	  //negative sign before omega is okay here as the image is located at -part[i].r.p[2]
	  
	  e= exp(-omega*part[i].r.p[2]);

	  scale = part[i].p.q*elc_params.di_mid_bot;

	  lclimgebot[POQESM]=scycache[o + ic].s/e;
	  lclimgebot[POQESP]=scycache[o + ic].s*e;	
	  lclimgebot[POQECM]=scycache[o + ic].c/e;
	  lclimgebot[POQECP]=scycache[o + ic].c*e;
	  
	  addscale_vec(gblcblk, scale, lclimgebot, gblcblk, size);
	  
	  e = ( exp(omega*(-part[i].r.p[2] - 2*elc_params.h  ))*elc_params.di_mid_bot +
		exp(omega*( part[i].r.p[2] - 2*elc_params.h )) )*fac_delta;  
	  
	} else {
	  
	  e= ( exp(omega*(-part[i].r.p[2] )) +
	       exp(omega*( part[i].r.p[2] - 2*elc_params.h ))*elc_params.di_mid_top )*fac_delta_mid_bot;    
	}
	
	lclimge[POQESP]+= part[i].p.q*scycache[o + ic].s*e;
	lclimge[POQECP]+= part[i].p.q*scycache[o + ic].c*e;
	
	
	if(part[i].r.p[2]>(elc_params.h-elc_params.space_layer)) { //handle the upper case now
	  
	  e=exp(omega*(2*elc_params.h-part[i].r.p[2]));

	  scale = part[i].p.q*elc_params.di_mid_top;

	  lclimgetop[POQESM]=scycache[o + ic].s/e;
	  lclimgetop[POQESP]=scycache[o + ic].s*e;	
	  lclimgetop[POQECM]=scycache[o + ic].c/e;
	  lclimgetop[POQECP]=scycache[o + ic].c*e;
	  
	  addscale_vec(gblcblk, scale, lclimgetop, gblcblk, size); 
	  
	  e = ( exp(omega*( part[i].r.p[2] -4*elc_params.h ))*elc_params.di_mid_top +
		exp(omega*(-part[i].r.p[2] -2*elc_params.h )) )*fac_delta; 
	  
	} else {
	  
	  e = ( exp(omega*( part[i].r.p[2] - 2*elc_params.h )) +
		exp(omega*(-part[i].r.p[2] -2*elc_params.h ))*elc_params.di_mid_bot )*fac_delta_mid_top;
	}
	
	lclimge[POQESM]+= part[i].p.q*scycache[o + ic].s*e;
	lclimge[POQECM]+= part[i].p.q*scycache[o + ic].c*e;
      }
      
      ic++;
    }
  }
  scale_vec(pref, gblcblk, size);

  if(elc_params.dielectric_contrast_on) {
    scale_vec(pref_di, lclimge, size);
    add_vec(gblcblk, gblcblk, lclimge, size);
  }

}

static void add_P_force()
{
  int np, c, i, ic;
  Particle *part;
  int size = 4;

  ic = 0;
  for (c = 0; c < local_cells.n; c++) {
    np   = local_cells.cell[c]->n;
    part = local_cells.cell[c]->part;
    for (i = 0; i < np; i++) {
      part[i].f.f[0] +=
	partblk[size*ic + POQESM]*gblcblk[POQECP] - partblk[size*ic + POQECM]*gblcblk[POQESP] +
	partblk[size*ic + POQESP]*gblcblk[POQECM] - partblk[size*ic + POQECP]*gblcblk[POQESM];
      part[i].f.f[2] +=
	partblk[size*ic + POQECM]*gblcblk[POQECP] + partblk[size*ic + POQESM]*gblcblk[POQESP] -
	partblk[size*ic + POQECP]*gblcblk[POQECM] - partblk[size*ic + POQESP]*gblcblk[POQESM];
      ic++;
    }
  }
}

static double P_energy(double omega)
{
  int np, c, i, ic;
  int size = 4;
  double eng = 0;
  double pref = 1/omega;

  ic = 0;
  for (c = 0; c < local_cells.n; c++) {
    np   = local_cells.cell[c]->n;
    for (i = 0; i < np; i++) {
      eng += pref*(partblk[size*ic + POQECM]*gblcblk[POQECP] + partblk[size*ic + POQESM]*gblcblk[POQESP] +
		   partblk[size*ic + POQECP]*gblcblk[POQECM] + partblk[size*ic + POQESP]*gblcblk[POQESM]);
      ic++;
    }
  }
  return eng;
}

static void add_Q_force()
{
  int np, c, i, ic;
  Particle *part;
  int size = 4;

  ic = 0;
  for (c = 0; c < local_cells.n; c++) {
    np   = local_cells.cell[c]->n;
    part = local_cells.cell[c]->part;
    for (i = 0; i < np; i++) {
      part[i].f.f[1] +=
	partblk[size*ic + POQESM]*gblcblk[POQECP] - partblk[size*ic + POQECM]*gblcblk[POQESP] +
	partblk[size*ic + POQESP]*gblcblk[POQECM] - partblk[size*ic + POQECP]*gblcblk[POQESM];
      part[i].f.f[2] +=
	partblk[size*ic + POQECM]*gblcblk[POQECP] + partblk[size*ic + POQESM]*gblcblk[POQESP] -
	partblk[size*ic + POQECP]*gblcblk[POQECM] - partblk[size*ic + POQESP]*gblcblk[POQESM];
      ic++;
    }
  }
}

static double Q_energy(double omega)
{
  int np, c, i, ic;
  int size = 4;
  double eng = 0;
  double pref = 1/omega;

  ic = 0;
  for (c = 0; c < local_cells.n; c++) {
    np   = local_cells.cell[c]->n;
    for (i = 0; i < np; i++) {
      eng += pref*(partblk[size*ic + POQECM]*gblcblk[POQECP] + partblk[size*ic + POQESM]*gblcblk[POQESP] +
		   partblk[size*ic + POQECP]*gblcblk[POQECM] + partblk[size*ic + POQESP]*gblcblk[POQESM]);
      ic++;
    }
  }
  return eng;
}

/*****************************************************************/
/* PQ particle blocks */
/*****************************************************************/

static void setup_PQ(int p, int q, double omega)
{
  int np, c, i, ic, ox = (p - 1)*n_localpart, oy = (q - 1)*n_localpart;
  Particle *part;
  double pref = -coulomb.prefactor*8*M_PI*ux*uy/(exp(omega*box_l[2]) - 1);
  double pref_di = coulomb.prefactor*8*M_PI*ux*uy; 
  double e;
  int size = 8;
  double lclimgebot[8],lclimgetop[8],lclimge[8];
  double fac_delta_mid_bot=1,fac_delta_mid_top=1,fac_delta=1;
  double scale=1;
  if(elc_params.dielectric_contrast_on) {
    double fac_elc=1.0/(1-elc_params.di_mid_top*elc_params.di_mid_bot*exp(-omega*2*elc_params.h)); 
    fac_delta_mid_bot=elc_params.di_mid_bot*fac_elc;
    fac_delta_mid_top=elc_params.di_mid_top*fac_elc; 
    fac_delta=fac_delta_mid_bot*elc_params.di_mid_top;
  }

  clear_vec(lclimge, size); 
  clear_vec(gblcblk, size);

  ic = 0;
  for (c = 0; c < local_cells.n; c++) {
    np   = local_cells.cell[c]->n;
    part = local_cells.cell[c]->part;
    for (i = 0; i < np; i++) {
      e = exp(omega*part[i].r.p[2]);

      partblk[size*ic + PQESSM] = scxcache[ox + ic].s*scycache[oy + ic].s*part[i].p.q/e;
      partblk[size*ic + PQESCM] = scxcache[ox + ic].s*scycache[oy + ic].c*part[i].p.q/e;
      partblk[size*ic + PQECSM] = scxcache[ox + ic].c*scycache[oy + ic].s*part[i].p.q/e;
      partblk[size*ic + PQECCM] = scxcache[ox + ic].c*scycache[oy + ic].c*part[i].p.q/e;

      partblk[size*ic + PQESSP] = scxcache[ox + ic].s*scycache[oy + ic].s*part[i].p.q*e;
      partblk[size*ic + PQESCP] = scxcache[ox + ic].s*scycache[oy + ic].c*part[i].p.q*e;
      partblk[size*ic + PQECSP] = scxcache[ox + ic].c*scycache[oy + ic].s*part[i].p.q*e;
      partblk[size*ic + PQECCP] = scxcache[ox + ic].c*scycache[oy + ic].c*part[i].p.q*e;

      add_vec(gblcblk, gblcblk, block(partblk, ic, size), size);
      
      if(elc_params.dielectric_contrast_on) {
	if(part[i].r.p[2]<elc_params.space_layer) { //handle the lower case first
	  //change e to take into account the z position of the images
	  
	  e= exp(-omega*part[i].r.p[2]);
	  scale = part[i].p.q*elc_params.di_mid_bot;
	  
	  lclimgebot[PQESSM] = scxcache[ox + ic].s*scycache[oy + ic].s/e;
	  lclimgebot[PQESCM] = scxcache[ox + ic].s*scycache[oy + ic].c/e;
	  lclimgebot[PQECSM] = scxcache[ox + ic].c*scycache[oy + ic].s/e;
	  lclimgebot[PQECCM] = scxcache[ox + ic].c*scycache[oy + ic].c/e;
	  
	  lclimgebot[PQESSP] = scxcache[ox + ic].s*scycache[oy + ic].s*e;
	  lclimgebot[PQESCP] = scxcache[ox + ic].s*scycache[oy + ic].c*e;
	  lclimgebot[PQECSP] = scxcache[ox + ic].c*scycache[oy + ic].s*e;
	  lclimgebot[PQECCP] = scxcache[ox + ic].c*scycache[oy + ic].c*e;
	  
	  addscale_vec(gblcblk, scale, lclimgebot, gblcblk, size);
	  
	  e = ( exp(omega*(-part[i].r.p[2] - 2*elc_params.h  ))*elc_params.di_mid_bot +
		exp(omega*( part[i].r.p[2] - 2*elc_params.h )) )*fac_delta*part[i].p.q;  
	  
	} else {
	  
	  e = ( exp(omega*(-part[i].r.p[2]     )) +
		exp(omega*( part[i].r.p[2] - 2*elc_params.h ))*elc_params.di_mid_top )*fac_delta_mid_bot*part[i].p.q;    
	} 
	
	lclimge[PQESSP]+= scxcache[ox + ic].s*scycache[oy + ic].s*e;
	lclimge[PQESCP]+= scxcache[ox + ic].s*scycache[oy + ic].c*e;
	lclimge[PQECSP]+= scxcache[ox + ic].c*scycache[oy + ic].s*e;
	lclimge[PQECCP]+= scxcache[ox + ic].c*scycache[oy + ic].c*e;
	
	
	if(part[i].r.p[2]>(elc_params.h-elc_params.space_layer)) { //handle the upper case now
	  
	  e=exp(omega*(2*elc_params.h-part[i].r.p[2]));
	  scale = part[i].p.q*elc_params.di_mid_top;
	  
	  lclimgetop[PQESSM] = scxcache[ox + ic].s*scycache[oy + ic].s/e;
	  lclimgetop[PQESCM] = scxcache[ox + ic].s*scycache[oy + ic].c/e;
	  lclimgetop[PQECSM] = scxcache[ox + ic].c*scycache[oy + ic].s/e;
	  lclimgetop[PQECCM] = scxcache[ox + ic].c*scycache[oy + ic].c/e;
	  
	  lclimgetop[PQESSP] = scxcache[ox + ic].s*scycache[oy + ic].s*e;
	  lclimgetop[PQESCP] = scxcache[ox + ic].s*scycache[oy + ic].c*e;
	  lclimgetop[PQECSP] = scxcache[ox + ic].c*scycache[oy + ic].s*e;
	  lclimgetop[PQECCP] = scxcache[ox + ic].c*scycache[oy + ic].c*e;
	  
	  addscale_vec(gblcblk,scale, lclimgetop, gblcblk, size); 
	  
	  e = ( exp(omega*( part[i].r.p[2] -4*elc_params.h ))*elc_params.di_mid_top +
		exp(omega*(-part[i].r.p[2] -2*elc_params.h )) )*fac_delta*part[i].p.q; 
	  
	} else {
	  
	  e = ( exp(omega*( part[i].r.p[2] -2*elc_params.h )) +
		exp(omega*(-part[i].r.p[2] -2*elc_params.h ))*elc_params.di_mid_bot )*fac_delta_mid_top*part[i].p.q;  
	}
	
	lclimge[PQESSM]+= scxcache[ox + ic].s*scycache[oy + ic].s*e;
	lclimge[PQESCM]+= scxcache[ox + ic].s*scycache[oy + ic].c*e;
	lclimge[PQECSM]+= scxcache[ox + ic].c*scycache[oy + ic].s*e;
	lclimge[PQECCM]+= scxcache[ox + ic].c*scycache[oy + ic].c*e;
	
      }
      
      ic++;
    }
  }

  scale_vec(pref, gblcblk, size);
  if(elc_params.dielectric_contrast_on)	{
    scale_vec(pref_di, lclimge, size);
    add_vec(gblcblk, gblcblk, lclimge, size);
  }
 }

static void add_PQ_force(int p, int q, double omega)
{
  int np, c, i, ic;
  Particle *part;
  double pref_x = C_2PI*ux*p/omega;
  double pref_y = C_2PI*uy*q/omega;
  int size = 8;

  ic = 0;
  for (c = 0; c < local_cells.n; c++) {
    np   = local_cells.cell[c]->n;
    part = local_cells.cell[c]->part;
    for (i = 0; i < np; i++) {
      part[i].f.f[0] +=
	pref_x*(partblk[size*ic + PQESCM]*gblcblk[PQECCP] + partblk[size*ic + PQESSM]*gblcblk[PQECSP] -
		partblk[size*ic + PQECCM]*gblcblk[PQESCP] - partblk[size*ic + PQECSM]*gblcblk[PQESSP] +
		partblk[size*ic + PQESCP]*gblcblk[PQECCM] + partblk[size*ic + PQESSP]*gblcblk[PQECSM] -
		partblk[size*ic + PQECCP]*gblcblk[PQESCM] - partblk[size*ic + PQECSP]*gblcblk[PQESSM]);
      part[i].f.f[1] +=
	pref_y*(partblk[size*ic + PQECSM]*gblcblk[PQECCP] + partblk[size*ic + PQESSM]*gblcblk[PQESCP] -
		partblk[size*ic + PQECCM]*gblcblk[PQECSP] - partblk[size*ic + PQESCM]*gblcblk[PQESSP] +
		partblk[size*ic + PQECSP]*gblcblk[PQECCM] + partblk[size*ic + PQESSP]*gblcblk[PQESCM] -
		partblk[size*ic + PQECCP]*gblcblk[PQECSM] - partblk[size*ic + PQESCP]*gblcblk[PQESSM]);
      part[i].f.f[2] +=
	       (partblk[size*ic + PQECCM]*gblcblk[PQECCP] + partblk[size*ic + PQECSM]*gblcblk[PQECSP] +
	        partblk[size*ic + PQESCM]*gblcblk[PQESCP] + partblk[size*ic + PQESSM]*gblcblk[PQESSP] -
	        partblk[size*ic + PQECCP]*gblcblk[PQECCM] - partblk[size*ic + PQECSP]*gblcblk[PQECSM] -
	        partblk[size*ic + PQESCP]*gblcblk[PQESCM] - partblk[size*ic + PQESSP]*gblcblk[PQESSM]);
      ic++;
    }
  }
}

static double PQ_energy(double omega)
{
  int np, c, i, ic;
  int size = 8;
  double eng = 0;
  double pref = 1/omega;

  ic = 0;
  for (c = 0; c < local_cells.n; c++) {
    np   = local_cells.cell[c]->n;
    for (i = 0; i < np; i++) {
      eng += pref*(partblk[size*ic + PQECCM]*gblcblk[PQECCP] + partblk[size*ic + PQECSM]*gblcblk[PQECSP] +
		   partblk[size*ic + PQESCM]*gblcblk[PQESCP] + partblk[size*ic + PQESSM]*gblcblk[PQESSP] +
		   partblk[size*ic + PQECCP]*gblcblk[PQECCM] + partblk[size*ic + PQECSP]*gblcblk[PQECSM] +
		   partblk[size*ic + PQESCP]*gblcblk[PQESCM] + partblk[size*ic + PQESSP]*gblcblk[PQESSM]);
      ic++;
    }
  }
  return eng;
}

/*****************************************************************/
/* main loops */
/*****************************************************************/

void ELC_add_force()
{
  int p, q;
  double omega;

  prepare_scx_cache();
  prepare_scy_cache();

  clear_log_forces("start");

  add_dipole_force();  

  clear_log_forces("dipole");

  add_z_force();

  clear_log_forces("z_force");

  /* the second condition is just for the case of numerical accident */
  for (p = 1; ux*(p - 1) < elc_params.far_cut && p <= n_scxcache; p++) {
    omega = C_2PI*ux*p;
    setup_P(p, omega);
    distribute(4);
    add_P_force(); 
    checkpoint("************distri p", p, 0, 2);
  }

  for (q = 1; uy*(q - 1) < elc_params.far_cut && q <= n_scycache; q++) {
    omega = C_2PI*uy*q;
    setup_Q(q, omega);
    distribute(4); 
    add_Q_force();
    checkpoint("************distri q", 0, q, 2);
  }

  for (p = 1; ux*(p - 1) < elc_params.far_cut  && p <= n_scxcache ; p++) {
    for (q = 1; SQR(ux*(p - 1)) + SQR(uy*(q - 1)) < elc_params.far_cut2 && q <= n_scycache; q++) {
      omega = C_2PI*sqrt(SQR(ux*p) + SQR(uy*q));
      setup_PQ(p, q, omega);
      distribute(8);
      add_PQ_force(p, q, omega); 
      checkpoint("************distri pq", p, q, 4);
    }
  }

  clear_log_forces("end");
}

double ELC_energy()
{
  double eng;
  int p, q;
  double omega;

  eng = dipole_energy(); 
  eng += z_energy();
  prepare_scx_cache();
  prepare_scy_cache();

  /* the second condition is just for the case of numerical accident */
  for (p = 1; ux*(p - 1) < elc_params.far_cut && p <= n_scxcache; p++) {
    omega = C_2PI*ux*p;
    setup_P(p, omega);
    distribute(4);
    eng += P_energy(omega);
    checkpoint("E************distri p", p, 0, 2);
  }
  for (q = 1; uy*(q - 1) < elc_params.far_cut && q <= n_scycache; q++) {
    omega = C_2PI*uy*q;
    setup_Q(q, omega);
    distribute(4);
    eng += Q_energy(omega);
    checkpoint("E************distri q", 0, q, 2);
  }
  for (p = 1; ux*(p - 1) < elc_params.far_cut  && p <= n_scxcache ; p++) {
    for (q = 1; SQR(ux*(p - 1)) + SQR(uy*(q - 1)) < elc_params.far_cut2 && q <= n_scycache; q++) {
      omega = C_2PI*sqrt(SQR(ux*p) + SQR(uy*q));
      setup_PQ(p, q, omega);
      distribute(8);
      eng += PQ_energy(omega);
      checkpoint("E************distri pq", p, q, 4);
    }
  }
  /* we count both i<->j and j<->i, so return just half of it */
  return 0.5*eng;
}

int ELC_tune(double error)
{
  double err;
  double h = elc_params.h, lz = box_l[2];
  double min_inv_boxl = dmin(ux, uy);
  
  if (elc_params.dielectric_contrast_on) {
    // adjust lz according to dielectric layer method
    lz = elc_params.h + elc_params.space_layer;
  }

  if (h < 0)
    return ES_ERROR;

  elc_params.far_cut = min_inv_boxl;
  do {
    err = 0.5*(exp(2*M_PI*elc_params.far_cut*h)/(lz - h)*
	       (C_2PI*elc_params.far_cut + 2*(ux + uy) + 1/(lz - h))/
	       (exp(2*M_PI*elc_params.far_cut*lz)- 1) +
	       exp(-2*M_PI*elc_params.far_cut*h)/(lz + 1)*
	       (C_2PI*elc_params.far_cut + 2*(ux + uy) + 1/(lz + h))/
	       (exp(2*M_PI*elc_params.far_cut*lz)- 1));

    elc_params.far_cut += min_inv_boxl;
  }
  while (err > error && elc_params.far_cut < MAXIMAL_FAR_CUT);
  if (elc_params.far_cut >= MAXIMAL_FAR_CUT)
    return ES_ERROR;
  elc_params.far_cut -= min_inv_boxl;
  elc_params.far_cut2 = SQR(elc_params.far_cut);

  return ES_OK;
}

/****************************************
 * COMMON PARTS
 ****************************************/

int ELC_sanity_checks()
{
  if (!PERIODIC(0) || !PERIODIC(1) || !PERIODIC(2)) {
      ostringstream msg;
      msg <<"ELC requires periodicity 1 1 1";
      runtimeError(msg);
    return 1;
  }
  /* The product of the two dielectric contrasts should be < 1 for ELC to work. This is not the case for
     two parallel boundaries, which can only be treated by the constant potential code */
  if (elc_params.dielectric_contrast_on && (fabs(1.0 - elc_params.di_mid_top*elc_params.di_mid_bot) < ROUND_ERROR_PREC)
      && !elc_params.const_pot_on) {
      ostringstream msg;
      msg <<"ELC with two parallel metallic boundaries requires the capacitor option";
      runtimeError(msg);
    return 1;
  }

  return 0;
}

void ELC_init()
{
  double maxsl;

  ELC_setup_constants();

  if (coulomb.method == COULOMB_ELC_P3M && elc_params.dielectric_contrast_on) {
    // recalculate the space layer size
    // set the space_layer to be 1/3 of the gap size, so that box = layer
    elc_params.space_layer = (1./3.)*elc_params.gap_size;
    // but make sure we leave enough space to not have to bother with overlapping
    // realspace P3M
    maxsl = elc_params.gap_size - p3m.params.r_cut;
    // and make sure the space layer is not bigger than half the actual simulation box,
    // to avoid overlaps
    if (maxsl > .5*elc_params.h) maxsl = .5*elc_params.h;
    if (elc_params.space_layer > maxsl) {
      if (maxsl <= 0) {
      ostringstream msg;
      msg <<"P3M real space cutoff too large for ELC w/ dielectric contrast";
      runtimeError(msg);
      }
      else
	elc_params.space_layer = maxsl;
    }

    // set the space_box 
    elc_params.space_box = elc_params.gap_size - 2*elc_params.space_layer;
    // reset minimal_dist for tuning
    elc_params.minimal_dist = dmin(elc_params.space_box, elc_params.space_layer);
  }

  if (elc_params.far_calculated &&
      (coulomb.method == COULOMB_ELC_P3M && elc_params.dielectric_contrast_on)) {
    if (ELC_tune(elc_params.maxPWerror) == ES_ERROR) {
        ostringstream msg;
        msg <<"ELC auto-retuning failed, gap size too small";
        runtimeError(msg);
    }
  }
  if (coulomb.method == COULOMB_ELC_P3M && elc_params.dielectric_contrast_on) {
    p3m.params.additional_mesh[0] = p3m.params.additional_mesh[1] = 0;
    p3m.params.additional_mesh[2] = elc_params.space_layer; 
  }
}

void ELC_on_resort_particles()
{
  n_localpart = cells_get_n_particles();
  n_scxcache = (int)(ceil(elc_params.far_cut/ux) + 1);
  n_scycache = (int)(ceil(elc_params.far_cut/uy) + 1);
  scxcache = (SCCache*)Utils::realloc(scxcache, n_scxcache*n_localpart*sizeof(SCCache));
  scycache = (SCCache*)Utils::realloc(scycache, n_scycache*n_localpart*sizeof(SCCache));
    
  partblk   = (double*)Utils::realloc(partblk,  n_localpart*8*sizeof(double));
}

int ELC_set_params(double maxPWerror, double gap_size, double far_cut, int neutralize,
		   double delta_top, double delta_bot, int const_pot_on, double pot_diff)
{
  elc_params.maxPWerror = maxPWerror;
  elc_params.gap_size = gap_size;
  elc_params.h = box_l[2] - gap_size;

  if (delta_top != 0.0 || delta_bot != 0.0) {
    elc_params.dielectric_contrast_on = 1;

    elc_params.di_mid_top = delta_top;
    elc_params.di_mid_bot = delta_bot;   

    // neutralize is automatical with dielectric contrast
    elc_params.neutralize = 0;
    // initial setup of parameters, may change later when P3M is finally tuned
    // set the space_layer to be 1/3 of the gap size, so that box = layer
    elc_params.space_layer = (1./3.)*gap_size;
    // set the space_box 
    elc_params.space_box = gap_size - 2*elc_params.space_layer;
    // reset minimal_dist for tuning
    elc_params.minimal_dist = dmin(elc_params.space_box, elc_params.space_layer);

    //Constant potential parameter setup
    if (const_pot_on)
    {
       elc_params.const_pot_on = 1;
       elc_params.pot_diff = pot_diff;	
    }
  }
  else {
    // setup without dielectric contrast
    elc_params.dielectric_contrast_on = 0;
    elc_params.const_pot_on = 0;
    elc_params.di_mid_top = 0;
    elc_params.di_mid_bot = 0;
    elc_params.neutralize = neutralize;
    elc_params.space_layer=0;
    elc_params.space_box = elc_params.minimal_dist = gap_size;
  }
  
  ELC_setup_constants();


  switch (coulomb.method) {
  case COULOMB_P3M_GPU:
  {
      ostringstream msg;
      msg <<"ELC tuning failed, ELC is not set up to work with the GPU P3M";
      runtimeError(msg);
      return ES_ERROR;
  }
  case COULOMB_ELC_P3M:
    
  case COULOMB_P3M:
    p3m.params.epsilon = P3M_EPSILON_METALLIC;
    coulomb.method = COULOMB_ELC_P3M;
    break;
  default:
    return ES_ERROR;
  }

  elc_params.far_cut = far_cut;
  if (far_cut != -1) {
    elc_params.far_cut2 = SQR(far_cut);
    elc_params.far_calculated = 0;
  }
  else {
    elc_params.far_calculated = 1;
    if (ELC_tune(elc_params.maxPWerror) == ES_ERROR) {
        ostringstream msg;
        msg <<"ELC tuning failed, gap size too small";
        runtimeError(msg);
    }
  }
  mpi_bcast_coulomb_params();

  return ES_OK;
}

////////////////////////////////////////////////////////////////////////////////////

void ELC_P3M_self_forces()
{
  Cell *cell;
  Particle *p;
  int i,c,np;
  double pos[3];
  double q, d[3], dist,dist2;

  for (c = 0; c < local_cells.n; c++) {
    cell = local_cells.cell[c];
    p  = cell->part;
    np = cell->n;
    for(i = 0; i < np; i++) {
      if(p[i].r.p[2]<elc_params.space_layer) {
	q=elc_params.di_mid_bot*p[i].p.q*p[i].p.q;
	pos[0]=p[i].r.p[0]; pos[1]=p[i].r.p[1]; pos[2]=-p[i].r.p[2];
	get_mi_vector(d, p[i].r.p, pos);
	dist2 = sqrlen(d);
	dist = sqrt(dist2);
	p3m_add_pair_force(q,d,dist2,dist,p[i].f.f);
      }
      if(p[i].r.p[2]>(elc_params.h-elc_params.space_layer)) {
	q=elc_params.di_mid_top*p[i].p.q*p[i].p.q;
	pos[0]=p[i].r.p[0]; pos[1]=p[i].r.p[1]; pos[2]=2*elc_params.h-p[i].r.p[2];
	get_mi_vector(d, p[i].r.p, pos);
	dist2 = sqrlen(d);
	dist = sqrt(dist2);
	p3m_add_pair_force(q,d,dist2,dist,p[i].f.f);
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////

void ELC_p3m_charge_assign_both()
{
  Cell *cell;
  Particle *p;
  double pos[3];
  int i,c,np;
  /* charged particle counter, charge fraction counter */
  int cp_cnt=0;
  /* prepare local FFT mesh */
  for(i=0; i<p3m.local_mesh.size; i++) p3m.rs_mesh[i] = 0.0;

  for (c = 0; c < local_cells.n; c++) {
    cell = local_cells.cell[c];
    p  = cell->part;
    np = cell->n;
    for(i = 0; i < np; i++) {
      if( p[i].p.q != 0.0 ) {
	p3m_assign_charge(p[i].p.q, p[i].r.p,cp_cnt);

	if(p[i].r.p[2]<elc_params.space_layer) {
	  double q=elc_params.di_mid_bot*p[i].p.q;
	  pos[0]=p[i].r.p[0]; pos[1]=p[i].r.p[1]; pos[2]=-p[i].r.p[2];
	  p3m_assign_charge(q, pos, -1);
	}
	
	if(p[i].r.p[2]>(elc_params.h-elc_params.space_layer)) {
	  double q=elc_params.di_mid_top*p[i].p.q;
	  pos[0]=p[i].r.p[0]; pos[1]=p[i].r.p[1]; pos[2]=2*elc_params.h-p[i].r.p[2];
	  p3m_assign_charge(q, pos,-1);
	}

	cp_cnt++;
      }
    }
  }
#ifdef P3M_STORE_CA_FRAC
  p3m_shrink_wrap_charge_grid(cp_cnt);
#endif
}

void ELC_p3m_charge_assign_image() 
{
  Cell *cell;
  Particle *p;
  double pos[3];
  int i,c,np;
  /* prepare local FFT mesh */
  for(i=0; i<p3m.local_mesh.size; i++) p3m.rs_mesh[i] = 0.0;

  for (c = 0; c < local_cells.n; c++) {
    cell = local_cells.cell[c];
    p  = cell->part;
    np = cell->n;
    for(i = 0; i < np; i++) {
      if( p[i].p.q != 0.0 ) {

	if(p[i].r.p[2]<elc_params.space_layer) {
	  double q=elc_params.di_mid_bot*p[i].p.q;
	  pos[0]=p[i].r.p[0]; pos[1]=p[i].r.p[1]; pos[2]=-p[i].r.p[2];
	  p3m_assign_charge(q, pos,-1);
	}
	
	if(p[i].r.p[2]>(elc_params.h-elc_params.space_layer)) {
	  double q=elc_params.di_mid_top*p[i].p.q;
	  pos[0]=p[i].r.p[0]; pos[1]=p[i].r.p[1]; pos[2]=2*elc_params.h-p[i].r.p[2];
	  p3m_assign_charge(q, pos, -1);
	}
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////

void ELC_P3M_dielectric_layers_force_contribution(Particle *p1, Particle *p2, double force1[3], double force2[3])
{
  double dist, dist2, d[3];
  double pos[3],q;
  double tp2;
  
  tp2=p2->r.p[2];
  
  if(p1->r.p[2]<elc_params.space_layer) {
    q=elc_params.di_mid_bot*p1->p.q*p2->p.q;
    pos[0]=p1->r.p[0]; pos[1]=p1->r.p[1]; pos[2]=-p1->r.p[2];
    get_mi_vector(d, p2->r.p, pos);
    dist2 = sqrlen(d);
    dist = sqrt(dist2);
    p3m_add_pair_force(q,d,dist2,dist,force2);
  } 
  
  if(p1->r.p[2]>(elc_params.h-elc_params.space_layer)) {
    q=elc_params.di_mid_top*p1->p.q*p2->p.q;
    pos[0]=p1->r.p[0]; pos[1]=p1->r.p[1]; pos[2]=2*elc_params.h-p1->r.p[2];
    get_mi_vector(d, p2->r.p, pos);
    dist2 = sqrlen(d);
    dist = sqrt(dist2);
    p3m_add_pair_force(q,d,dist2,dist,force2);
  } 
  
  if(tp2<elc_params.space_layer) {
    q=elc_params.di_mid_bot*p1->p.q*p2->p.q;
    pos[0]=p2->r.p[0]; pos[1]=p2->r.p[1]; pos[2]=-tp2;
    get_mi_vector(d, p1->r.p, pos);
    dist2 = sqrlen(d);
    dist = sqrt(dist2);
    p3m_add_pair_force(q,d,dist2,dist,force1);
  }
  
  if(tp2>(elc_params.h-elc_params.space_layer)) {	  
    q=elc_params.di_mid_top*p1->p.q*p2->p.q;
    pos[0]=p2->r.p[0]; pos[1]=p2->r.p[1]; pos[2]=2*elc_params.h-tp2;
    get_mi_vector(d, p1->r.p, pos);
    dist2 = sqrlen(d);
    dist = sqrt(dist2);
    p3m_add_pair_force(q,d,dist2,dist,force1);
  }
}

/////////////////////////////////////////////////////////////////////////////////////	  

double ELC_P3M_dielectric_layers_energy_contribution(Particle *p1, Particle *p2) 
{
  double pos[3],q;
  double dist, dist2, d[3];
  double tp2;
  double eng=0.0;
  
	
  tp2=p2->r.p[2];
  
  if(p1->r.p[2]<elc_params.space_layer) {
    q=elc_params.di_mid_bot*p1->p.q*p2->p.q;
    pos[0]=p1->r.p[0]; pos[1]=p1->r.p[1]; pos[2]=-p1->r.p[2];
    get_mi_vector(d, p2->r.p, pos);
    dist2 = sqrlen(d);
    dist = sqrt(dist2);
    eng+=p3m_pair_energy(q,d,dist2,dist);
    
  } 
  
  if(p1->r.p[2]>(elc_params.h-elc_params.space_layer)) {
    q=elc_params.di_mid_top*p1->p.q*p2->p.q;
    pos[0]=p1->r.p[0]; pos[1]=p1->r.p[1]; pos[2]=2*elc_params.h-p1->r.p[2];
    get_mi_vector(d, p2->r.p, pos);
    dist2 = sqrlen(d);
    dist = sqrt(dist2);
    eng+=p3m_pair_energy(q,d,dist2,dist);
  } 
  
  if(tp2<elc_params.space_layer) {
    q=elc_params.di_mid_bot*p1->p.q*p2->p.q;
    pos[0]=p2->r.p[0]; pos[1]=p2->r.p[1]; pos[2]=-tp2;
    get_mi_vector(d, p1->r.p, pos);
    dist2 = sqrlen(d);
    dist = sqrt(dist2);
    eng+=p3m_pair_energy(q,d,dist2,dist);
  }
  
  if(tp2>(elc_params.h-elc_params.space_layer)) {
    q=elc_params.di_mid_top*p1->p.q*p2->p.q;
    pos[0]=p2->r.p[0]; pos[1]=p2->r.p[1]; pos[2]=2*elc_params.h-tp2;
    get_mi_vector(d, p1->r.p, pos);
    dist2 = sqrlen(d);
    dist = sqrt(dist2);
    eng+=p3m_pair_energy(q,d,dist2,dist);
  }

  // fprintf(stderr,"energy is %f\n",eng);
  return (eng);
}

//////////////////////////////////////////////////////////////////////////////////

double ELC_P3M_dielectric_layers_energy_self() {
  int c, np1, i;
  Cell *cell;
  Particle *p1;
  double pos[3],q;
  double dist, dist2, d[3];
  double eng=0.0;
  for (c = 0; c < local_cells.n; c++) {
    cell = local_cells.cell[c];
    p1   = cell->part;
    np1  = cell->n;
    
    // Loop cell neighbors 
    for(i=0; i < np1; i++) {
      // Loop neighbor cell particles 
      
      if(p1[i].r.p[2]<elc_params.space_layer) {
	q=elc_params.di_mid_bot*p1[i].p.q*p1[i].p.q;
	pos[0]=p1[i].r.p[0]; pos[1]=p1[i].r.p[1]; pos[2]=-p1[i].r.p[2];
	get_mi_vector(d, p1[i].r.p, pos);
	dist2 = sqrlen(d);
	dist = sqrt(dist2);
	eng+=p3m_pair_energy(q,d,dist2,dist);
	//	fprintf(stderr,"energy is %f\n",eng);
      }
	
      if(p1[i].r.p[2]>(elc_params.h-elc_params.space_layer)) {
	q=elc_params.di_mid_top*p1[i].p.q*p1[i].p.q;
	pos[0]=p1[i].r.p[0]; pos[1]=p1[i].r.p[1]; pos[2]=2*elc_params.h-p1[i].r.p[2];
	get_mi_vector(d, p1[i].r.p, pos);
	dist2 = sqrlen(d);
	dist = sqrt(dist2);
       	eng+=p3m_pair_energy(q,d,dist2,dist);
	//	fprintf(stderr,"energy is %f\n",eng);
      }
    }
  }
  return (eng);
 } 

/////////////////////////////////////////////////////////////////////////////////

void  ELC_P3M_modify_p3m_sums_both()
{
  Cell *cell;
  Particle *part;
  int i,c,np;
  double node_sums[3], tot_sums[3];

  for(i=0;i<3;i++) { node_sums[i]=0.0; tot_sums[i]=0.0;}

  for (c = 0; c < local_cells.n; c++) {
    cell = local_cells.cell[c];
    part = cell->part;
    np   = cell->n;
    for(i=0;i<np;i++) {
      if( part[i].p.q != 0.0 ) {
	
        node_sums[0] += 1.0;
        node_sums[1] += SQR(part[i].p.q);
        node_sums[2] += part[i].p.q;

	if(part[i].r.p[2]<elc_params.space_layer) {

	  node_sums[0] += 1.0;
	  node_sums[1] += SQR(elc_params.di_mid_bot*part[i].p.q);
	  node_sums[2] += elc_params.di_mid_bot*part[i].p.q;
	  
	}
	if(part[i].r.p[2]>(elc_params.h-elc_params.space_layer)) {

	  node_sums[0] += 1.0;
	  node_sums[1] += SQR(elc_params.di_mid_top*part[i].p.q);
	  node_sums[2] += elc_params.di_mid_top*part[i].p.q;
	  
	}
	
      }
    }
  }
  
  MPI_Allreduce(node_sums, tot_sums, 3, MPI_DOUBLE, MPI_SUM, comm_cart);
  p3m.sum_qpart    = (int)(tot_sums[0]+0.1);
  p3m.sum_q2       = tot_sums[1];
  p3m.square_sum_q = SQR(tot_sums[2]);
}

void  ELC_P3M_modify_p3m_sums_image()
{
  Cell *cell;
  Particle *part;
  int i,c,np;
  double node_sums[3], tot_sums[3];

  for(i=0;i<3;i++) { node_sums[i]=0.0; tot_sums[i]=0.0;}

  for (c = 0; c < local_cells.n; c++) {
    cell = local_cells.cell[c];
    part = cell->part;
    np   = cell->n;
    for(i=0;i<np;i++) {
      if( part[i].p.q != 0.0 ) {
	
	if(part[i].r.p[2]<elc_params.space_layer) {

	  node_sums[0] += 1.0;
	  node_sums[1] += SQR(elc_params.di_mid_bot*part[i].p.q);
	  node_sums[2] += elc_params.di_mid_bot*part[i].p.q;
	  
	}
	if(part[i].r.p[2]>(elc_params.h-elc_params.space_layer)) {

	  node_sums[0] += 1.0;
	  node_sums[1] += SQR(elc_params.di_mid_top*part[i].p.q);
	  node_sums[2] += elc_params.di_mid_top*part[i].p.q;
	  
	}
	
      }
    }
  }
  
  MPI_Allreduce(node_sums, tot_sums, 3, MPI_DOUBLE, MPI_SUM, comm_cart);

  p3m.sum_qpart    = (int)(tot_sums[0]+0.1);
  p3m.sum_q2       = tot_sums[1];
  p3m.square_sum_q = SQR(tot_sums[2]);
}

// this function is required in force.cpp for energy evaluation
void  ELC_P3M_restore_p3m_sums()
{
  Cell *cell;
  Particle *part;
  int i,c,np;
  double node_sums[3], tot_sums[3];

  for(i=0;i<3;i++) { node_sums[i]=0.0; tot_sums[i]=0.0;}

  for (c = 0; c < local_cells.n; c++) {
    cell = local_cells.cell[c];
    part = cell->part;
    np   = cell->n;
    for(i=0;i<np;i++) {
      if( part[i].p.q != 0.0 ) {

        node_sums[0] += 1.0;
        node_sums[1] += SQR(part[i].p.q);
        node_sums[2] += part[i].p.q;
      }
    }
  }
  
  MPI_Allreduce(node_sums, tot_sums, 3, MPI_DOUBLE, MPI_SUM, comm_cart);

  p3m.sum_qpart    = (int)(tot_sums[0]+0.1);
  p3m.sum_q2       = tot_sums[1];
  p3m.square_sum_q = SQR(tot_sums[2]);
}

#endif

