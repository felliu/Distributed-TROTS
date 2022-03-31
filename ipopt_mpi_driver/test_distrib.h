#ifndef TEST_DISTRIB_H
#define TEST_DISTRIB_H

#include <vector>
#include <mpi.h>
#include "rank_local_data.h"

void test_trotsentry_distrib(const LocalData& local_data,
                             const std::vector<std::vector<int>>* entry_distrib,
                             const std::vector<TROTSEntry>* entries,
                             MPI_Comm comm);

void print_local_nnz_count(const LocalData& local_data);
#endif