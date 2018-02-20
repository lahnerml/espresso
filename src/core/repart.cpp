
#include "repart.hpp"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <stdexcept>
#include "cells.hpp"
#include "domain_decomposition.hpp"
#include "interaction_data.hpp"
#include "short_range_loop.hpp"

// Fills weights with a constant.
static void metric_ncells(std::vector<double> &weights) {
  std::fill(weights.begin(), weights.end(), 1.0);
}

// Fills weights with the number of particles per cell.
static void metric_npart(std::vector<double> &weights) {
  std::transform(local_cells.cell, local_cells.cell + local_cells.n,
                 weights.begin(), [](const Cell *c) { return c->n; });
}

int cell_ndistpairs(Cell *c) {
  int nnp =
      std::accumulate(std::begin(c->m_neighbors), std::end(c->m_neighbors), 0,
                      [](int acc, const Cell &neigh) { return acc + neigh.n; });
  return c->n * nnp;
}

// Fills weights with the number of distance pairs per cell.
static void metric_ndistpairs(std::vector<double> &weights) {
  std::transform(local_cells.begin(), local_cells.end(), weights.begin(),
                 cell_ndistpairs);
}

static void metric_nforcepairs(std::vector<double> &weights) {
  // Ugly hack: Use pairkernel to advance the current cell number
  int cellno = 0;
  for(;cellno < (local_cells.n - 1) && local_cells.cell[cellno]->n == 0; cellno++);

  auto particlekernel = [&cellno](Particle &p) {
    const Cell *c = local_cells.cell[cellno];
    if (p.p.identity == c->part[c->n - 1].p.identity) {
        if (cellno < (local_cells.n - 1))
            cellno++;
        while (cellno < (local_cells.n - 1) && local_cells.cell[cellno]->n == 0)
            cellno++;
    }
  };

  auto pairkernel = [&weights, &cellno](Particle &p1, Particle &p2,
                                       Distance &d) {
#ifdef EXCLUSIONS
    if (do_nonbonded(&p1, &p2))
#endif
    {
      IA_parameters *ia_params = get_ia_param(p1.p.type, p2.p.type);
      double dist = std::sqrt(d.dist2);
      if (dist < ia_params->LJ_cut + ia_params->LJ_offset &&
          dist > ia_params->LJ_min + ia_params->LJ_offset)
        weights[cellno]++;
    }
  };

  short_range_loop(particlekernel, pairkernel);
}

int cell_nbondedia(Cell *cell) {
  int nbondedia = 0;
  for (Particle *p = cell->part; p < cell->part + cell->n; p++) {
    for (int i = 0; i < p->bl.n;) {
      int type_num = p->bl.e[i++];
      Bonded_ia_parameters *iaparams = &bonded_ia_params[type_num];
      // int type = iaparams->type;

      // This could be incremented conditionally if "type" has a specific value
      // to only count bonded_ia of a certain type.
      nbondedia++;
      i += iaparams->num; // Skip the all partner particle numbers
    }
  }
  return nbondedia;
}

static void metric_nbondedia(std::vector<double> &weights) {
  std::transform(local_cells.begin(), local_cells.end(), weights.begin(),
                 cell_nbondedia);
}

static void metric_nghostcells(std::vector<double> &weights) {
  // Reminder: Weights is a vector over local cells.
  // We simply count the number of ghost cells and
  // add this cost in equal parts to all local cells
  double nghostfrac = static_cast<double>(ghost_cells.n) / local_cells.n;
  std::fill(weights.begin(), weights.end(), nghostfrac);
}

static void metric_nghostpart(std::vector<double> &weights) {
  // Reminder: Weights is a vector over local cells.
  // We simply count the number of ghost particles and
  // add this cost in equal parts to all local cells
  int nghostpart =
      std::accumulate(ghost_cells.begin(), ghost_cells.end(), 0,
                      [](int acc, const Cell *c) { return acc + c->n; });
  double nghostfrac = static_cast<double>(nghostpart) / local_cells.n;
  std::fill(weights.begin(), weights.end(), nghostfrac);
}

// Generator for random integers
struct Randintgen {
  Randintgen()
      : mt(std::chrono::high_resolution_clock::now()
               .time_since_epoch()
               .count()),
        d(1, 1000) {}
  Randintgen(Randintgen &&other)
      : mt(std::move(other.mt)), d(std::move(other.d)) {}
  int operator()() { return d(mt); }

private:
  std::mt19937 mt;
  std::uniform_int_distribution<int> d;
};

// Fills weights with random integers
static void metric_rand(std::vector<double> &weights) {
  std::generate(weights.begin(), weights.end(), Randintgen());
}


// Get the appropriate metric function described in string "desc".
// Throws a std::invalid_argument exception if no such metric is available
static repart::metric::metric_func
get_single_metric_func(const std::string& desc)
{
  using desc_func_pair = std::pair<std::string, repart::metric::metric_func>;
  static const std::vector<desc_func_pair> mets = {
    { "ncells"     , metric_ncells },
    { "npart"      , metric_npart },
    { "ndistpairs" , metric_ndistpairs },
    { "nforcepairs", metric_nforcepairs },
    { "nbondedia"  , metric_nbondedia },
    { "nghostcells", metric_nghostcells },
    { "nghostpart" , metric_nghostpart },
    { "rand"       , metric_rand }
  };

  for (const auto& t: mets) {
    if (desc == std::get<0>(t)) {
      return std::get<1>(t);
    }
  }

  throw std::invalid_argument(std::string("No such metric available: ") + desc);
}

repart::metric::metric(const std::string& desc) {
  parse_metric_desc(desc);
}

void repart::metric::parse_metric_desc(const std::string& desc) {
  std::istringstream is(desc);
  bool parseadd = desc[0] == '+' || desc[0] == '-';

  // Single metric case
  if (std::all_of(desc.begin(), desc.end(), [](char c){ return ::isalpha(c) || c == '_';})) {
    mdesc.emplace_back(1.0, get_single_metric_func(desc));
    return;
  }

  while (is.good()) {
    double factor;
    char add = '+';
    char mult;
    std::string method;

    if (parseadd)
      is >> add; // Skip this conditionally during the first iteration since
                 // positive factor does not need to have '+' in fist addend.
    else
      parseadd = true;
    if (!is || (add != '+' && add != '-'))
      throw std::invalid_argument("Could not parse metric string: Expected +/-.");

    is >> factor;
    if (!is)
      throw std::invalid_argument("Could not parse metric string: Expected factor.");

    if (add == '-')
      factor *= -1;

    is >> mult; // '*'
    if (!is || mult != '*')
      throw std::invalid_argument("Could not parse metric string: Expected '*'.");

    // FIXME: Needs space after method name.
    is >> method;
    if (!is)
      throw std::invalid_argument("Could not parse metric string: Expected method name. Space after name is mandatory.");

    mdesc.emplace_back(factor, get_single_metric_func(method));
  }
}

std::vector<double> repart::metric::operator()() const {
  std::vector<double> w(local_cells.n, 0.0), tmp(local_cells.n);

  for (const auto& t: mdesc) {
    double factor = std::get<0>(t);
    repart::metric::metric_func func = std::get<1>(t);

    func(tmp);
    // Multiply add to w.
    std::transform(w.begin(), w.end(), tmp.begin(), w.begin(),
                   [factor](double acc, double val){ return acc + factor * val; });
  }
  return w;
}

