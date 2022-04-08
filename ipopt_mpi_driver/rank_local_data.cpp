#include "rank_local_data.h"

#include <cassert>

#include "TROTSEntry.h"

namespace {
    void set_matrix_reference(TROTSEntry& entry, LocalData& data) {
        bool found_id = false;
        const int data_id = entry.get_id();
        auto mat_it = data.matrices.find(data_id);
        if (mat_it != data.matrices.end()) {
            found_id = true;
            entry.set_matrix_ptr(mat_it->second.get());
        }

        auto vec_it = data.mean_vecs.find(data_id);
        if (vec_it != data.mean_vecs.end()) {
            found_id = true;
            entry.set_mean_vec_ptr(&(vec_it->second));
        }

        //If there is no corresponding entry with the right ID, something is wrong.
        assert(found_id);
    }
}

void init_local_data(LocalData& data) {
    data.x_buffer.resize(data.num_vars);
    data.grad_tmp.resize(data.num_vars);
    data.local_jac_nnz = 0;
    for (TROTSEntry& entry : data.obj_entries) {
        set_matrix_reference(entry, data);
    }

    for (TROTSEntry& entry : data.cons_entries) {
        data.local_jac_nnz += entry.get_grad_nnz();
        set_matrix_reference(entry, data);
    }
}

