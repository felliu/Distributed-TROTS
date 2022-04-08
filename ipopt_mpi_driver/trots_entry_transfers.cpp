#include <sstream>

#include <iostream>

#include <boost/archive/basic_binary_oarchive.hpp>

#include "rank_local_data.h"
#include "TROTSEntry.h"
#include "trots_entry_transfers.h"
#include "globals.h"

namespace {
    int probe_message_size(enum MPIMessageTags tag, MPI_Comm communicator, MPI_Datatype type, int rank) {
        MPI_Status status;
        MPI_Probe(rank, tag, communicator, &status);
        int size;
        MPI_Get_count(&status, type, &size);
        return size;
    }

    void distribute_trots_entries_comm(
            const std::vector<TROTSEntry>& entries,
            const std::vector<std::vector<int>>& rank_distrib,
            MPI_Comm communicator) {
        for (int rank = 1; rank < rank_distrib.size(); ++rank) {
            const std::vector<int>& entries_for_rank = rank_distrib[rank];
            const int num_entries = static_cast<int>(entries_for_rank.size());
            MPI_Send(&num_entries, 1, MPI_INT, rank, NUM_ENTRIES_TAG, communicator);
            for (int entry_idx : entries_for_rank) {
                std::ostringstream stream;
                boost::archive::binary_oarchive o_archive(stream);
                o_archive << entries[entry_idx];
                const std::string str = stream.str();
                //Avoid problems with null-termination by copying the characters of the string
                std::vector<char> char_buf(str.begin(), str.end());
                MPI_Send(static_cast<const void*>(&char_buf[0]), char_buf.size(),
                         MPI_CHAR, rank, TROTS_ENTRY_TAG, communicator);
            }
        }
    }

    void recv_entries_for_comm(LocalData& data, MPI_Comm comm, bool objective_entries) {
        int num_entries = 0;
        MPI_Recv(&num_entries, 1, MPI_INT, 0, NUM_ENTRIES_TAG, comm, MPI_STATUS_IGNORE);
        for (int i = 0; i < num_entries; ++i) {
            int num_bytes = probe_message_size(TROTS_ENTRY_TAG, comm, MPI_CHAR, 0);
            std::vector<char> buf(num_bytes);
            MPI_Recv(&buf[0], num_bytes, MPI_CHAR, 0, TROTS_ENTRY_TAG, comm, MPI_STATUS_IGNORE);

            std::string str_buf(buf.cbegin(), buf.cend());
            std::istringstream istream(std::move(str_buf));
            boost::archive::binary_iarchive iarchive(istream);
            TROTSEntry new_entry;
            iarchive >> new_entry;

            if (objective_entries)
                data.obj_entries.push_back(new_entry);
            else
                data.cons_entries.push_back(new_entry);
        }
    }
}


void distribute_trots_entries_send(const std::vector<TROTSEntry>& obj_entries,
                                   const std::vector<TROTSEntry>& cons_entries,
                                   const std::vector<std::vector<int>>& rank_distrib_obj,
                                   const std::vector<std::vector<int>>& rank_distrib_cons) {

    //When this function is called,
    //the objective and constraint rank communicators should have been initialized already

    //This function should only be called by rank 0
    //Check that we're rank zero on all communicators
    int world_rank = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    assert(world_rank == 0);

    distribute_trots_entries_comm(obj_entries, rank_distrib_obj, MPI_COMM_WORLD);
    distribute_trots_entries_comm(cons_entries, rank_distrib_cons, MPI_COMM_WORLD);
}

void recv_trots_entries(LocalData& data) {
    //We need to be part of at least one of the objective or constraints communicators
    recv_entries_for_comm(data, MPI_COMM_WORLD, true);
    recv_entries_for_comm(data, MPI_COMM_WORLD, false);
}

