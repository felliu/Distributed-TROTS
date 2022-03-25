#ifndef TROTS_ENTRY_TRANSFERS_H
#define TROTS_ENTRY_TRANSFERS_H

#include "TROTSEntry.h"

struct LocalData;

void distribute_trots_entries_send(const std::vector<TROTSEntry>& obj_entries,
                                   const std::vector<TROTSEntry>& cons_entries,
                                   const std::vector<std::vector<int>>& rank_distrib_obj,
                                   const std::vector<std::vector<int>>& rank_distrib_cons);

void recv_trots_entries(LocalData& data);

#endif