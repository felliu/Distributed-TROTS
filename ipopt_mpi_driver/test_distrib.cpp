#include "test_distrib.h"
#include "TROTSEntry.h"
#include "trots.h"

#include "util.h"

#include <mpi.h>

#include <iostream>
#include <fstream>

void test_trotsentry_distrib(const LocalData& local_data,
                             const std::vector<std::vector<int>>* distrib_ptr,
                             const std::vector<TROTSEntry>* entries,
                             MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    int size;
    MPI_Comm_size(comm, &size);
    std::vector<int> data_ids;
    if (rank == 0) {
        assert(distrib_ptr != nullptr && entries != nullptr);
        std::vector<std::vector<int>> distrib = *distrib_ptr;
        assert(size == static_cast<int>(distrib.size()));
        for (int other_rank = 1; other_rank < distrib.size(); ++other_rank) {
            const std::vector<int>& idxs = distrib[other_rank];
            for (int idx : idxs) {
                data_ids.push_back((*entries)[idx].get_id());
            }
            std::cout << "data_ids: ";
            print_vector(data_ids);
            MPI_Send(&data_ids[0], data_ids.size(), MPI_INT, other_rank, 0, comm);
            data_ids.clear();
        }
    }
    else {
        std::vector<int> local_data_ids;
        for (int i = 0; i < local_data.obj_entries.size(); ++i) {
            local_data_ids.push_back(local_data.obj_entries[i].get_id());
        }
        data_ids.resize(local_data_ids.size());
        MPI_Recv(&data_ids[0], data_ids.size(), MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
        for (int i = 0; i < local_data_ids.size(); ++i) {
            assert(data_ids[i] == local_data_ids[i]);
        }
    }
}

void print_local_nnz_count(const LocalData& local_data) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int local_nnz = 0;
    for (auto&& [id, mat] : local_data.matrices) {
        local_nnz += mat->get_nnz();
    }
    for (auto&& [id, vec] : local_data.mean_vecs) {
        local_nnz += vec.size();
    }
    std::cout << "World rank: " << rank << " nnz: " << local_nnz << std::endl;
}

void dump_distrib_data_to_file(const std::vector<std::vector<int>>& obj_distrib,
                               const std::vector<std::vector<int>>& cons_distrib,
                               const TROTSProblem& problem) {
    int num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    const std::string file_name("rank_distrib_" + std::to_string(num_ranks) + "_ranks.txt");
    std::ofstream out_file(file_name);

    std::vector<std::vector<int>> obj_nnz_distrib;
    std::vector<std::vector<int>> cons_nnz_distrib;

    assert(obj_distrib.size() == cons_distrib.size());
    for (int i = 0; i < obj_distrib.size(); ++i) {
        const std::vector<int>& obj_distrib_rank = obj_distrib[i];
        std::vector<int>& obj_nnz_rank = obj_nnz_distrib.emplace_back();
        for (int obj_idx : obj_distrib_rank) {
            obj_nnz_rank.push_back(problem.objective_entries[obj_idx].get_nnz());
        }
        const std::vector<int>& cons_distrib_rank = cons_distrib[i];
        std::vector<int>& cons_nnz_rank = cons_nnz_distrib.emplace_back();
        for (int cons_idx : cons_distrib_rank) {
            cons_nnz_rank.push_back(problem.constraint_entries[cons_idx].get_nnz());
        }
    }

    out_file << "Objective distrib nnz\n";
    for (int i = 1; i < obj_nnz_distrib.size(); ++i) {
        out_file << "Rank " << i << " nnz counts: ";
        const std::vector<int>& nnz_cnts = obj_nnz_distrib[i];
        for (int nnz : nnz_cnts) {
            out_file << nnz << ",";
        }
        out_file << "\n\n";
    }

    out_file << "Cons distrib nnz\n";
    for (int i = 1; i < cons_nnz_distrib.size(); ++i) {
        out_file << "Rank " << i << " nnz counts: ";
        const std::vector<int>& nnz_cnts = cons_nnz_distrib[i];
        for (int nnz : nnz_cnts) {
            out_file << nnz << ",";
        }
        out_file << "\n\n";
    }
}