#include "rank_local_data.h"

#include <cassert>

#include "TROTSEntry.h"

void init_local_data(LocalData& data) {
    for (TROTSEntry& entry : data.trots_entries) {
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

