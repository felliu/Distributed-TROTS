#include <mpi.h>

#include <iostream>
#include <filesystem>
<<<<<<< HEAD
#include <unordered_set>
=======
>>>>>>> c68b32435f1d34ff87c68ee86b8b1eb27a365bbf

#include "data_distribution.h"
#include "trots.h"

namespace {
    void print_distribution_info(const std::vector<std::vector<int>>& buckets,
                                 const TROTSProblem& problem) {
        int idx = 0;
        for (const std::vector<int>& vec : buckets) {
            int nnz_sum = 0;
            std::cout << "Bucket number " << idx << " elements:\n";
            for (int idx : vec) {
                std::cout << idx << ",";
                nnz_sum += problem.objective_entries[idx].get_nnz();
            }
            std::cout << " nnz sum: " << nnz_sum << "\n\n";
            ++idx;
        }
    }

    MPI_Comm obj_ranks_comm;
    MPI_Comm cons_ranks_comm;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int num_ranks = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    std::vector<int> obj_ranks;
    std::vector<int> cons_ranks;
    TROTSProblem trots_problem;
    if (world_rank == 0) {
        if (argc < 2 || argc > 3) {
            std::cerr << "Usage: ./program <mat_file>\n"
                      << "\t./program <mat_file> <max_iters>\n";
        }

        std::string path_str{argv[1]};
        std::filesystem::path path{path_str};

        trots_problem = std::move(TROTSProblem{TROTSMatFileData{path}});

        //Get the distribution between ranks to calculate terms of the objective
        //and constraints
        std::tie(obj_ranks, cons_ranks) = get_obj_cons_rank_idxs(trots_problem);


        //Create the communicators:
        //Broadcast the rank distribution info to all other ranks
        int sizes[] = {static_cast<int>(obj_ranks.size()),
                       static_cast<int>(cons_ranks.size())};
        //First send the sizes of the buffers
        MPI_Bcast(&sizes[0], 2, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&obj_ranks[0], obj_ranks.size(), MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&cons_ranks[0], cons_ranks.size(), MPI_INT, 0, MPI_COMM_WORLD);

    } else {
        //First involvement of other ranks: get the rank distribution info from rank 0
        int sizes[2];
        MPI_Bcast(&sizes[0], 2, MPI_INT, 0, MPI_COMM_WORLD);
        obj_ranks.resize(sizes[0]);
        cons_ranks.resize(sizes[1]);

        MPI_Bcast(&obj_ranks[0], sizes[0], MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&cons_ranks[0], sizes[1], MPI_INT, 0, MPI_COMM_WORLD);
    }

    std::tie(obj_ranks_comm, cons_ranks_comm) = split_obj_cons_comm(obj_ranks, cons_ranks);

    if (world_rank == 0) {
        std::vector<std::vector<int>> rank_distrib_obj =
            get_rank_distribution(trots_problem.objective_entries, obj_ranks.size());

        std::vector<std::vector<int>> rank_distrib_cons =
            get_rank_distribution(trots_problem.constraint_entries, cons_ranks.size());

        //Check that rank 0 is still rank 0 in the other comms:
        int obj_comm_rank = 1;
        MPI_Comm_rank(obj_ranks_comm, &obj_comm_rank);
        int cons_comm_rank = 1;
        MPI_Comm_rank(cons_ranks_comm, &cons_comm_rank);

        assert(obj_comm_rank != 0);
        assert(cons_comm_rank == 0);

        //Step 1: post the sends for the data matrices to the correct ranks.
        //Figure out which matrix goes where
        std::vector<std::unordered_set<int>> data_id_buckets_obj_ranks(rank_distrib_obj.size());
        std::vector<std::unordered_set<int>> data_id_buckets_cons_ranks(rank_distrib_cons.size());
        for (int i = 0; i < rank_distrib_obj.size(); ++i) {
            const std::vector<int>& entry_idxs = rank_distrib_obj[i];
            for (const int entry_idx : entry_idxs) {

            }
        }

    }

    MPI_Finalize();
    return 0;
}