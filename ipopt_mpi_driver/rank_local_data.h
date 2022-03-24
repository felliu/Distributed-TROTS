#ifndef RANK_LOCAL_DATA_H
#define RANK_LOCAL_DATA_H

#include <memory>
#include <vector>

#include "SparseMat.h"
#include "TROTSEntry.h"

struct LocalData {
    std::vector<std::unique_ptr<SparseMatrix<double>>> matrices;
    std::vector<int> matrix_ids;

    std::vector<std::vector<double>> mean_vecs;
    std::vector<int> mean_vec_ids;

    std::vector<TROTSEntry> trots_entries;
};

#endif