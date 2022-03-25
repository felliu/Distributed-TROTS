#ifndef SPARSE_MATRIX_TRANSFERS_H
#define SPARSE_MATRIX_TRANSFERS_H

enum MPIMessageTags {
    NUM_MATS_TAG,
    DATA_ID_TAG,
    VEC_FLAG_TAG,
    VEC_DATA_TAG,
    CSR_DATA_TAG,
    CSR_COL_INDS_TAG,
    CSR_ROW_PTRS_TAG,
    CSR_NUM_COLS_TAG
};

class TROTSProblem;
class LocalData;

void distribute_sparse_matrices_send(
    TROTSProblem& trots_problem,
    const std::vector<std::vector<int>>& rank_distrib_obj,
    const std::vector<std::vector<int>>& rank_distrib_cons);
void receive_sparse_matrices(LocalData& local_data);
#endif