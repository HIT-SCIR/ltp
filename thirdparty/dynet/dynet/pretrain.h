#ifndef DYNET_PRETRAIN_H
#define DYNET_PRETRAIN_H

#include <string>
#include <vector>
#include <unordered_map>
#include "dynet/dict.h"
#include "dynet/model.h"

namespace dynet {

void save_pretrained_embeddings(const std::string& fname,
    const Dict& d,
    const LookupParameter& lp);

void read_pretrained_embeddings(const std::string& fname,
    Dict& d,
    std::unordered_map<int, std::vector<float>>& vectors);

} // namespace dynet

#endif
