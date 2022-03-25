#include <mpi.h>

#include <iostream>
#include <filesystem>
#include <unordered_set>

#include "data_distribution.h"
#include "globals.h"
#include "rank_local_data.h"
#include "sparse_matrix_transfers.h"
#include "trots.h"
#include "debug_utils.h"

MPI_Comm obj_ranks_comm = MPI_COMM_NULL;
MPI_Comm cons_ranks_comm = MPI_COMM_NULL;

namespace {


    void show_rank_local_data(int rank, const LocalData& data) {
        int my_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        if (my_rank != rank)
            return;

        std::cout << "Local data for rank: " << rank << "\n";
        std::cout << "Matrix ids: ";
        print_vector(data.matrix_ids);

        std::cout << "Vec ids: ";
        print_vector(data.mean_vec_ids);

        int total_nnz = 0;
        for (const auto& mat_ptr : data.matrices) {
            total_nnz += mat_ptr->get_nnz();
        }

        for (const auto& mean_vec : data.mean_vecs) {
            total_nnz += mean_vec.size();
        }

        std::cout << "Total nnz: " << total_nnz << std::endl;
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int num_ranks = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    std::vector<int> obj_ranks;
    std::vector<int> cons_ranks;
    std::vector<std::vector<int>> rank_distrib_obj;
    std::vector<std::vector<int>> rank_distrib_cons;
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

    std::cerr << "communicators split!\n";

    if (world_rank == 0) {
        int num_obj_ranks = 0;
        MPI_Comm_size(obj_ranks_comm, &num_obj_ranks);
        int num_cons_ranks = 0;
        MPI_Comm_size(cons_ranks_comm, &num_cons_ranks);

        //Get a roughly even distribution of the matrices between the ranks, excluding rank 0
        rank_distrib_obj = get_rank_distribution(trots_problem.objective_entries, num_obj_ranks - 1);
        rank_distrib_cons = get_rank_distribution(trots_problem.constraint_entries, num_cons_ranks - 1);

        distribute_sparse_matrices_send(trots_problem, rank_distrib_obj, rank_distrib_cons);
    }

    LocalData rank_local_data;
    if (world_rank != 0)
        receive_sparse_matrices(rank_local_data);

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        std::cout << "Finished transfers" << std::endl;
    }

    for (int i = 0; i < num_ranks; ++i) {
        show_rank_local_data(i, rank_local_data);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}