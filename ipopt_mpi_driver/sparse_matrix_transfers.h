#ifndef SPARSE_MATRIX_TRANSFERS_H
#define SPARSE_MATRIX_TRANSFERS_H


class TROTSProblem;
class LocalData;

void distribute_sparse_matrices_send(
    TROTSProblem& trots_problem,
    const std::vector<std::vector<int>>& rank_distrib_obj,
    const std::vector<std::vector<int>>& rank_distrib_cons);
void receive_sparse_matrices(LocalData& local_data);
#endif