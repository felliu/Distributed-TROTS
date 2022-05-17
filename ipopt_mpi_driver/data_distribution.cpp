#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <tuple>

#include <mpi.h>

#include "data_distribution.h"
#include "util.h"

std::tuple<MPI_Comm, MPI_Comm>
split_obj_cons_comm(const std::vector<int>& obj_ranks, const std::vector<int>& cons_ranks) {
        MPI_Group world_group;
        MPI_Comm_group(MPI_COMM_WORLD, &world_group);

        MPI_Group cons_group;
        MPI_Group_incl(world_group, cons_ranks.size(), &cons_ranks[0], &cons_group);

        MPI_Group obj_group;
        MPI_Group_incl(world_group, obj_ranks.size(), &obj_ranks[0], &obj_group);

        MPI_Comm obj_comm, cons_comm;

        MPI_Comm_create(MPI_COMM_WORLD, obj_group, &obj_comm);
        MPI_Comm_create(MPI_COMM_WORLD, cons_group, &cons_comm);
        return std::make_tuple(obj_comm, cons_comm);
}

std::tuple<std::vector<int>, std::vector<int>>
get_obj_cons_rank_idxs(const TROTSProblem& problem) {
    const auto num_constraints = problem.constraint_entries.size();
    const auto num_objectives = problem.objective_entries.size();
    int num_ranks_world = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks_world);

    int nnz_cons = 0;
    int nnz_obj = 0;
    for (const TROTSEntry& cons_entry : problem.constraint_entries) {
        nnz_cons += cons_entry.get_nnz();
    }
    for (const TROTSEntry& obj_entry : problem.objective_entries) {
        nnz_obj += obj_entry.get_nnz();
    }
    const int nnz_total = nnz_cons + nnz_obj;
    const double cons_rank_frac = nnz_cons / static_cast<double>(nnz_total);
    int num_cons_ranks = static_cast<int>(std::round(cons_rank_frac * num_ranks_world));
    num_cons_ranks = std::min(num_cons_ranks, num_ranks_world - 2);
    int num_obj_ranks = num_ranks_world - num_cons_ranks + 1;

    std::vector<int> cons_ranks;
    for (int i = 0; i < num_cons_ranks; ++i)
        cons_ranks.push_back(i);

    std::vector<int> obj_ranks = {0};
    for (int i = num_cons_ranks; i < num_ranks_world; ++i)
        obj_ranks.push_back(i);
    print_vector(cons_ranks);
    print_vector(obj_ranks);

    return std::make_tuple(obj_ranks, cons_ranks);
}

//The return value is a partitioning of TROTSEntries of roughly equal size.
std::vector<std::vector<int>>
get_rank_distribution(const std::vector<TROTSEntry>& entries, int num_ranks) {
    std::vector<std::vector<int>> buckets;
    for (int i = 0; i < num_ranks; ++i)
        buckets.emplace_back();

    std::vector<int> entry_idxs(entries.size());
    std::iota(entry_idxs.begin(), entry_idxs.end(), 0);

    const auto comparison_func = [&entries](int entry_a, int entry_b) -> bool {
        return entries[entry_a].get_nnz() > entries[entry_b].get_nnz();
    };

    //Sort the entry_idx array by the nnz of the corresponding TROTSEntry
    //(In descending order)
    std::sort(entry_idxs.begin(), entry_idxs.end(), comparison_func);

    const auto compare_nnz_sums =
        [&entries](const std::vector<int>& a,
                   const std::vector<int>& b) -> bool {
        int a_nnz_sum = 0;
        for (int i : a)
            a_nnz_sum += entries[i].get_nnz();
        int b_nnz_sum = 0;
        for (int i : b)
            b_nnz_sum += entries[i].get_nnz();

        return a_nnz_sum < b_nnz_sum;
    };

    //greedy distribution: loop over all indexes, except the first one (belonging to rank 0) and put the next entry into the currently
    //smallest bucket
    for (int idx : entry_idxs) {
        auto min_elem_it = std::min_element(buckets.begin() + 1, buckets.end(), compare_nnz_sums);
        min_elem_it->push_back(idx);
    }

    return buckets;
}
