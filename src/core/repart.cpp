
#include "repart.hpp"
#include <chrono>
#include <stdexcept>
#include <boost/iterator/iterator_facade.hpp>
#include "cells.hpp"
#include "domain_decomposition.hpp"
#include "mol_cut.hpp"
#include "interaction_data.hpp"

// Iterator for iterating over integers (enumerating things while using
// std::transform)
struct IotaIter
  : public boost::iterator_facade<IotaIter,
                                  const int,
                                  boost::forward_traversal_tag>
{
  IotaIter(): i(0) {}
  IotaIter(int i): i(i) {}
  IotaIter(const IotaIter& other): i(other.i) {}

private:
  int i;
  friend class boost::iterator_core_access;

  void increment() { i++; }
  bool equal(const IotaIter& other) const { return i == other.i; }
  const int& dereference() const { return i; }
};

template <typename T>
struct Averager {
    void operator()(T val) { sum += val; count++; }
    T operator*() { return sum / count; }

private:
    T sum = T{0};
    int count = 0;
};

template <typename T, typename It>
T average(It first, It last)
{
    return *std::for_each(first, last, Averager<T>());
}

template <typename It>
typename It::value_type average(It first, It last)
{
    return average<typename It::value_type>(first, last);
}


void
repart::print_cell_info(const std::string& prefix, const std::string& method)
{
  int nc = local_cells.n;
  int npart = std::accumulate(local_cells.cell,
                              local_cells.cell + nc,
                              0,
                              [](int acc, Cell* c) { return acc + c->n; });

  int pmax, pmin, cmax, cmin;
  MPI_Reduce(&npart, &pmax, 1, MPI_INT, MPI_MAX, 0, comm_cart);
  MPI_Reduce(&npart, &pmin, 1, MPI_INT, MPI_MIN, 0, comm_cart);
  MPI_Reduce(&nc, &cmax, 1, MPI_INT, MPI_MAX, 0, comm_cart);
  MPI_Reduce(&nc, &cmin, 1, MPI_INT, MPI_MIN, 0, comm_cart);

  if (this_node == 0)
    printf("%s (%s): #Particle (#Cells) max: %i (%i), min: %i (%i)\n",
           prefix.c_str(), method.c_str(), pmax, cmax, pmin, cmin);
}


// Fills metric with a constant.
static void
metric_ncells(std::vector<double>& metric)
{
  std::fill(metric.begin(), metric.end(), 1.0);
}

// Fills metric with the number of particles per cell.
static void
metric_npart(std::vector<double>& metric)
{
  std::transform(local_cells.cell,
                 local_cells.cell + local_cells.n,
                 metric.begin(),
                 [](const Cell *c) { return c->n; });
}

int cell_ndistpairs(Cell *c, int i)
{
  int nnp = std::accumulate(dd.cell_inter[i].nList,
                            dd.cell_inter[i].nList
                              + dd.cell_inter[i].n_neighbors,
                            0,
                            [](int acc, const IA_Neighbor& neigh){
                              return acc + neigh.pList->n;
                            });
  return c->n * nnp;
}

// Fills metric with the number of distance pairs per cell.
static void
metric_ndistpairs(std::vector<double>& metric)
{
  // Cell number can't be easily deduced from a copied Cell*/**
  // Therefore, we use std::transform with two input iterators
  // and one is an IotaIter.
  std::transform(local_cells.cell,
                 local_cells.cell + local_cells.n,
                 IotaIter(),
                 metric.begin(),
                 cell_ndistpairs);
}

int cell_nforcepairs(Cell *cell, int c)
{
  int npairs = 0;
  for (int n = 0; n < dd.cell_inter[c].n_neighbors; n++) {
    IA_Neighbor *neigh = &dd.cell_inter[c].nList[n];
    for (Particle *p1 = cell->part; p1 < cell->part + cell->n; p1++) {
      for (Particle *p2 = neigh->pList->part + (n == 0? p1 - cell->part + 1: 0); /* Half shell within a cell itself. */
          p2 < neigh->pList->part + neigh->pList->n; p2++) {
#ifdef EXCLUSIONS
        if(do_nonbonded(p1, p2))
#endif
        {
          double vec21[3];
          double dist = sqrt(distance2vec(p1->r.p, p2->r.p, vec21));
          IA_parameters *ia_params = get_ia_param(p1->p.type, p2->p.type);
          if (CUTOFF_CHECK(dist < ia_params->LJ_cut+ia_params->LJ_offset) &&
              CUTOFF_CHECK(dist > ia_params->LJ_min+ia_params->LJ_offset))
            npairs++;
        }
      }
    }
  }
  return npairs;
}

static void
metric_nforcepairs(std::vector<double>& metric)
{
  std::transform(local_cells.cell,
                 local_cells.cell + local_cells.n,
                 IotaIter(),
                 metric.begin(),
                 cell_nforcepairs);
}

int cell_nbondedia(Cell *cell, int c)
{
  int nbondedia = 0;
  for (Particle *p = cell->part; p < cell->part + cell->n; p++) {
    for (int i = 0; i < p->bl.n; ) {
      int type_num = p->bl.e[i++];
      Bonded_ia_parameters *iaparams = &bonded_ia_params[type_num];
      //int type = iaparams->type;

      // This could be incremented conditionally if "type" has a specific value
      // to only count bonded_ia of a certain type.
      nbondedia++;
      i += iaparams->num; // Skip the all partner particle numbers
    }
  }
  return nbondedia;
}

static void
metric_nbondedia(std::vector<double>& metric)
{
  std::transform(local_cells.cell,
                 local_cells.cell + local_cells.n,
                 IotaIter(),
                 metric.begin(),
                 cell_nforcepairs);
}

static void
metric_runtime(std::vector<double>& metric)
{
    std::vector<double> ts(local_cells.n, 0.0);
    calc_link_cell_runtime(10, ts);
    // Factor to make entries larger than 1.
    double f = 10.0 / average(ts.begin(), ts.end());
    std::transform(ts.begin(), ts.end(),
                   metric.begin(),
                   [f](double d){ return f * d;});
}

// Generator for random integers
struct Randintgen {
  Randintgen():
    mt(std::chrono::high_resolution_clock::now().time_since_epoch().count()),
    d(1, 1000) {}
  Randintgen(Randintgen&& other): mt(std::move(other.mt)),
                                  d(std::move(other.d)) {}
  int operator()() { return d(mt); }
private:
  std::mt19937 mt;
  std::uniform_int_distribution<int> d;
};

// Fills metric with random integers
static void
metric_rand(std::vector<double>& metric)
{
  std::generate(metric.begin(), metric.end(), Randintgen());
}


std::function<void(std::vector<double>&)>
repart::get_metric_func(const std::string& desc)
{
  using repart_func = std::function<void(std::vector<double>&)>;
  static const std::vector<std::pair<std::string, repart_func>> mets = {
    { "ncells"     , metric_ncells },
    { "npart"      , metric_npart },
    { "ndistpairs" , metric_ndistpairs },
    { "nforcepairs", metric_nforcepairs },
    { "nbondedia"  , metric_nbondedia },
    { "runtime"    , metric_runtime },
    { "rand"       , metric_rand }
  };

  for (const auto& t: mets) {
    if (desc == std::get<0>(t)) {
      return std::get<1>(t);
    }
  }

  throw std::invalid_argument(std::string("No such metric available: ") + desc);
}

