#include "dynet/pretrain.h"

#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <fstream>
#include "dynet/dict.h"
#include "dynet/model.h"

using namespace std;

namespace dynet {

void save_pretrained_embeddings(const std::string& fname,
    const Dict& d,
    const LookupParameter& lp) {
  cerr << "Writing word vectors to " << fname << " ...\n";
  ofstream out(fname);
  if(!out)
    DYNET_INVALID_ARG("Could not save embeddings to " << fname);
  auto& m = *lp.get();
  for (unsigned i = 0; i < d.size(); ++i) {
    out << d.convert(i) << ' ' << (*m.values[i]).transpose() << endl;
  }
}

void read_pretrained_embeddings(const std::string& fname,
    Dict& d,
    std::unordered_map<int, std::vector<float>>& vectors) {
  int unk = -1;
  if (d.is_frozen()) unk = d.get_unk_id();
  cerr << "Loading word vectors from " << fname << " ...\n";
  ifstream in(fname);
  if(!in)
    DYNET_INVALID_ARG("Could not load embeddings from " << fname);
  string line;
  string word;
  vector<float> v;
  getline(in, line);
  istringstream lin(line);
  lin >> word;
  while(lin) {
    float x;
    lin >> x;
    if (!lin) break;
    v.push_back(x);
  }
  unsigned vec_size = v.size();
  int wid = d.convert(word);
  if (wid != unk) vectors[wid] = v;
  while(getline(in, line)) {
    istringstream lin(line);
    lin >> word;
    int w = d.convert(word);
    if (w != unk) {
      for (unsigned i = 0; i < vec_size; ++i) lin >> v[i];
      vectors[w] = v;
    }
  }
}

} // dynet
