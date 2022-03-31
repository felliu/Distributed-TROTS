#include "test_distrib.h"
#include "TROTSEntry.h"

#include "util.h"

#include <iostream>

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
        for (int i = 0; i < local_data.trots_entries.size(); ++i) {
            local_data_ids.push_back(local_data.trots_entries[i].get_id());
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