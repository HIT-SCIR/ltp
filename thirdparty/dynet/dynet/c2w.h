#ifndef DYNET_C2W_H_
#define DYNET_C2W_H_

#include <vector>
#include <map>

#include "dynet/dynet.h"
#include "dynet/model.h"
#include "dynet/lstm.h"

namespace dynet {

// computes a representation of a word by reading characters
// one at a time
struct C2WBuilder {
  LSTMBuilder fc2w;
  LSTMBuilder rc2w;
  LookupParameter p_lookup;
  std::vector<VariableIndex> words;
  std::map<int, VariableIndex> wordid2vi;
  explicit C2WBuilder(int vocab_size,
                      unsigned layers,
                      unsigned input_dim,
                      unsigned hidden_dim,
                      Model* m) :
      fc2w(layers, input_dim, hidden_dim, m),
      rc2w(layers, input_dim, hidden_dim, m),
      p_lookup(m->add_lookup_parameters(vocab_size, {input_dim})) {
  }
  void new_graph(ComputationGraph* cg) {
    words.clear();
    fc2w.new_graph(cg);
    rc2w.new_graph(cg);
  }
  // compute a composed representation of a word out of characters
  // wordid should be a unique index for each word *type* in the graph being built
  VariableIndex add_word(int word_id, const std::vector<int>& chars, ComputationGraph* cg) {
    auto it = wordid2vi.find(word_id);
    if (it == wordid2vi.end()) {
      fc2w.start_new_sequence(cg);
      rc2w.start_new_sequence(cg);
      std::vector<VariableIndex> ins(chars.size());
      std::map<int, VariableIndex> c2i;
      for (unsigned i = 0; i < ins.size(); ++i) {
        VariableIndex& v = c2i[chars[i]];
        if (!v) v = cg->add_lookup(p_lookup, chars[i]);
        ins[i] = v;
        fc2w.add_input(v, cg);
      }
      for (int i = ins.size() - 1; i >= 0; --i)
        rc2w.add_input(ins[i], cg);
      VariableIndex i_concat = cg->add_function<Concatenate>({fc2w.back(), rc2w.back()});
      it = wordid2vi.insert(std::make_pair(word_id, i_concat)).first;
    }
    return it->second;
  }
};

} // namespace dynet

#endif
