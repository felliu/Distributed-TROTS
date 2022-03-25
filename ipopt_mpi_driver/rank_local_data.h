#ifndef RANK_LOCAL_DATA_H
#define RANK_LOCAL_DATA_H

#include <memory>
#include <vector>
#include <unordered_map>

#include "SparseMat.h"
#include "TROTSEntry.h"

struct LocalData {
    //Map from dataID to dose matrix
    std::unordered_map<int, std::unique_ptr<SparseMatrix<double>>> matrices;
    std::unordered_map<int, std::vector<double>> mean_vecs;

    std::vector<TROTSEntry> trots_entries;
};

void init_local_data(LocalData& data);

#endif