#ifndef DATA_DISTRIBUTION_H
#define DATA_DISTRIBUTION_H

#include <vector>
#include <tuple>

#include <mpi.h>

#include "trots.h"

//Distributes the terms of the TROTSProblem (roughly) evenly between MPI ranks so that
//the workload is even.
//Return value: map from MPI rank to list of indexes of the terms in the TROTSProblem belonging to that rank.
std::vector<std::vector<int>>
get_rank_distribution(const std::vector<TROTSEntry>& entries, int num_ranks);

std::tuple<std::vector<int>, std::vector<int>>
get_obj_cons_rank_idxs(const TROTSProblem& problem);

std::tuple<MPI_Comm, MPI_Comm>
split_obj_cons_comm(const std::vector<int>& obj_ranks, const std::vector<int>& cons_ranks);
#endif
