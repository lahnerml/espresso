
#include <cstdio>
#include <vector>
#include <string>

#include "lbmd_repart.hpp"
#include "repart.hpp"
#include "p4est_utils.hpp"
#include "cells.hpp"

#if (defined(LB_ADAPTIVE) || defined(DD_P4EST))
#include <p8est.h>

namespace __md_detail {

static std::vector<double> weights(const std::string& metric_desc)
{
    return repart::metric{metric_desc}();
}

static void preprocess()
{
    // Saves "old" nquads for later use to resort along the
    // space-filling curve.
#ifdef DD_P4EST
    p4est_dd_repart_preprocessing();
#endif
#ifdef LB_ADAPTIVE

#endif
}

static void postprocess()
{
    // This is save to do as dd_p4est_create_grid does not repartition
    // (as it normally would) if LB_ADAPTIVE is defined.
    cells_re_init(CELL_STRUCTURE_CURRENT, true, true);
}

}

namespace __lbm_detail {

static std::vector<double> weights(const std::string& metric_desc)
{
    // TODO
    return std::vector<double>{};
}

static void preprocess()
{
    // TODO
}

static void postprocess()
{
    // TODO
}

}


struct repart_info {
    forest_order fo;
    std::vector<double> (*weights)(const std::string&);
    void (*preprocess)();
    void (*postprocess)();
};

std::vector<repart_info> repart_infos = {
#ifdef DD_P4EST
    { forest_order::short_range, __md_detail::weights, __md_detail::preprocess, __md_detail::postprocess },
#endif
#ifdef LB_ADAPTIVE
    { forest_order::adaptive_LB, __lbm_detail::weights, __lbm_detail::preprocess, __lbm_detail::postprocess }
#endif
};

static p8est_t* tree_of(const repart_info& ri)
{
    return p4est_utils_get_forest_info(ri.fo).p4est;
}

namespace lbmd {

void repart_all(const std::vector<std::string>& metrics, const std::vector<double>& alphas)
{
    std::vector<std::vector<double>> ws;

    if (metrics.size() != repart_infos.size() || metrics.size() != alphas.size()) {
        std::fprintf(stderr, "Error: Have %zu metric strings but only %zu forests to repart.\n",
                     metrics.size(), repart_infos.size());
        return;
    }

    if (repart_infos.size() != 2) {
        std::fprintf(stderr, "Error: Currently, repartitioning is only implemented for exactly two forests.\n");
        return;
    }

    for (const auto& ri: repart_infos)
        ri.preprocess();

    ws.reserve(repart_infos.size());
    std::transform(std::begin(repart_infos), std::end(repart_infos),
                   std::begin(metrics), std::back_inserter(ws),
                   [](const repart_info& ri, const std::string& m){
        return ri.weights(m);
    });

    p4est_utils_weighted_partition(tree_of(repart_infos[0]),
                                   ws[0], alphas[0],
                                   tree_of(repart_infos[1]),
                                   ws[1], alphas[1]);

    for (const auto& ri: repart_infos)
        ri.postprocess();
}

}

#endif

