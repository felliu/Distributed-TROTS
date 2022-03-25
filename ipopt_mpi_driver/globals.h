#ifndef GLOBALS_H
#define GLOBALS_H

#include <mpi.h>

extern MPI_Comm obj_ranks_comm;
extern MPI_Comm cons_ranks_comm;

enum MPIMessageTags {
    NUM_MATS_TAG,
    DATA_ID_TAG,
    VEC_FLAG_TAG,
    VEC_DATA_TAG,
    CSR_DATA_TAG,
    CSR_COL_INDS_TAG,
    CSR_ROW_PTRS_TAG,
    CSR_NUM_COLS_TAG,
    TROTS_ENTRY_TAG,
    NUM_ENTRIES_TAG
};

#endif