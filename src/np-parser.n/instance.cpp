#include "parser.n/instance.h"
#include "utils/strutils.hpp"
#include "utils/codecs.hpp"

namespace ltp {
namespace depparser {

using strutils::codecs::is_unicode_punctuation;

Instance::Instance() {}
Instance::~Instance() {}

size_t Instance::size() const { return forms.size(); }

size_t Instance::num_heads(bool ignore) const {
  size_t ret = 0;
  for (size_t i = 0; i < raw_forms.size(); ++ i) {
    if (ignore && is_unicode_punctuation(raw_forms[i])) { continue; }
    ++ ret;
  }
  return ret;
}

size_t Instance::num_recalled_heads(bool ignore) const {
  size_t ret = 0;
  for (size_t i = 0; i < raw_forms.size(); ++ i) {
    if (ignore && is_unicode_punctuation(raw_forms[i])) { continue; }
    if (heads[i] == predict_heads[i]) {
      ++ ret;
    }
  }
  return ret;
}

size_t Instance::num_recalled_deprels(bool ignore) const {
  size_t ret= 0;
  for (size_t i = 0; i < raw_forms.size(); ++ i) {
    if (ignore && is_unicode_punctuation(raw_forms[i])) { continue; }
    if (heads[i] == predict_heads[i] && deprels[i] == predict_deprels[i]) {
      ++ ret;
    }
  }
  return ret;
}

bool Instance::is_tree(const std::vector<int>& heads) const {
  tree_t tree(heads.size());
  int root = -1;
  for (int modifier = 0; modifier < heads.size(); ++ modifier) {
    int head = heads[modifier];
    if (head == -1) { root = modifier; } else { tree[head].push_back(modifier); }
  }
  std::vector<bool> visited(heads.size(), false);
  if (!is_tree_travel(root, tree, visited)) { return false; }
  for (size_t i = 0; i < visited.size(); ++ i) {
    bool visit = visited[i];
    if (!visit) { return false; }
  }
  return true;
}

bool Instance::is_tree_travel(int now, const tree_t& tree,
    std::vector<bool>& visited) const{
  if (visited[now]) { return false; }
  visited[now] = true;
  for (int c = 0; c < tree[now].size(); ++ c) {
    int next = tree[now][c];
    if (!is_tree_travel(next, tree, visited)) { return false; }
  }
  return true;
}

bool Instance::is_non_projective(const std::vector<int>& heads) const {
  for (int modifier = 0; modifier < heads.size(); ++ modifier) {
    int head = heads[modifier];
    if (head < modifier) {
      for (int from = head+ 1; from < modifier; ++ from) {
        int to = heads[from];
        if (to < head || to > modifier) { return true; }
      }
    } else {
      for (int from = modifier+ 1; from < head; ++ from) {
        int to = heads[from];
        if (to < modifier || to > head) { return true; }
      }
    }
  }
  return false;
}

bool Instance::is_projective(const std::vector<int>& heads) const {
  return !is_non_projective(heads);
}

bool Instance::is_tree() const { return is_tree(heads); }
bool Instance::is_projective() const { return is_projective(heads); }
bool Instance::is_non_projective() const { return is_non_projective(heads); }

size_t Dependency::size() const { return forms.size(); }
} //  namespace depparser
} //  namespace ltp
