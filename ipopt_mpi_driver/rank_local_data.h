#ifndef RANK_LOCAL_DATA_H
#define RANK_LOCAL_DATA_H

#include <memory>
#include <vector>
#include <unordered_map>

#include "SparseMat.h"
#include "TROTSEntry.h"

//Data each MPI rank has for computing its terms in the objective function
struct LocalData {
    //Map from dataID to dose matrix
    std::unordered_map<int, std::unique_ptr<SparseMatrix<double>>> matrices;
    std::unordered_map<int, std::vector<double>> mean_vecs;

    std::vector<TROTSEntry> obj_entries;
    std::vector<TROTSEntry> cons_entries;

    int local_jac_nnz;

    std::vector<double> x_buffer;
    std::vector<double> grad_tmp;
    int num_vars;
};



void init_local_data(LocalData& data);

#endif