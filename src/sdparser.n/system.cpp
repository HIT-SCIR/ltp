#include "parser.n/system.h"
#include "utils/math/mat.h"
#include "utils/logging.hpp"
#include <algorithm>

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

namespace ltp {
namespace depparser {

using math::Mat3;

Action ActionFactory::make_shift() {
  return Action(Action::kShift, 0);
}

Action ActionFactory::make_left_arc(const int& rel) {
  return Action(Action::kLeftArc, rel);
}

Action ActionFactory::make_right_arc(const int& rel) {
  return Action(Action::kRightArc, rel);
}

Action ActionFactory::make_swap() {
  return Action(Action::kSwap, 0);
}

std::ostream& operator<<(std::ostream& os, const Action& act) {
  if (act.name() == Action::kShift) {
    os << "SH";
  } else if (act.name() == Action::kLeftArc) {
    os << "LA~" << act.rel();
  } else if (act.name() == Action::kRightArc) {
    os << "RA~" << act.rel();
  } else if (act.name() == Action::kSwap) {
    os << "SW";
  }else if (act.name() == Action::kNone) {
    os << "NO";
  } else {
    WARNING_LOG("unknown action");
  }
  return os;
}

bool ActionUtils::is_shift(const Action& act) {
  return (act.name() == Action::kShift);
}

bool ActionUtils::is_left_arc(const Action& act, int& deprel) {
  if (act.name() == Action::kLeftArc) { deprel = act.rel(); return true;  }
  deprel = 0;
  return false;
}

bool ActionUtils::is_right_arc(const Action& act, int& deprel) {
  if (act.name() == Action::kRightArc) { deprel = act.rel(); return true; }
  deprel = 0;
  return false;
}

bool ActionUtils::is_swap(const Action& act) {
  return (act.name() == Action::kSwap);
}

void ActionUtils::get_oracle_actions(const std::vector<int>& heads,
    const std::vector<int>& deprels,
    std::vector<Action>& actions) {
  // The oracle finding algorithm for arcstandard is using a in-order tree
  // searching.
  int N = heads.size();
  int root = -1;
  tree_t tree(N);

  actions.clear();
  for (int i = 0; i < N; ++ i) {
    int head = heads[i];
    if (head == -1) {
      if (root == -1)
        WARNING_LOG("error: there should be only one root.");
      root = i;
    } else {
      tree[head].push_back(i);
    }
  }

  get_oracle_actions_travel(root, heads, deprels, tree, actions);
}

void ActionUtils::get_oracle_actions(const Dependency& instance,
    std::vector<Action>& actions) {
  get_oracle_actions(instance.heads, instance.deprels, actions);
}

void ActionUtils::get_oracle_actions_travel(int root,
    const std::vector<int>& heads,
    const std::vector<int>& deprels,
    const tree_t& tree,
    std::vector<Action>& actions) {
  const std::vector<int>& children = tree[root];

  int i;
  for (i = 0; i < children.size() && children[i] < root; ++ i) {
    get_oracle_actions_travel(children[i], heads, deprels, tree, actions);
  }

  actions.push_back(ActionFactory::make_shift());

  for (int j = i; j < children.size(); ++ j) {
    int child = children[j];
    get_oracle_actions_travel(child, heads, deprels, tree, actions);
    actions.push_back(ActionFactory::make_right_arc(deprels[child]));
  }

  for (int j = i - 1; j >= 0; -- j) {
    int child = children[j];
    actions.push_back(ActionFactory::make_left_arc(deprels[child]));
  }
}

void ActionUtils::get_oracle_actions2(const Dependency& instance,
    std::vector<Action>& actions) {
  get_oracle_actions2(instance.heads, instance.deprels, actions);
}

void ActionUtils::get_oracle_actions2(const std::vector<int>& heads,
    const std::vector<int>& deprels,
    std::vector<Action>& actions) {
  //! represent the tree structure for projective order calculation
  int N = heads.size();
  int root = -1;
  tree_t tree(N);
  for (int i = 0; i < N; ++ i) {
    int head = heads[i];
    if (head == -1) {
      if (root != -1)
        WARNING_LOG("error: there should be only one root.");
      root = i;
    } else {
      tree[head].push_back(i);
    }
  }
  //! calculate the projective order
  int timestamp = 0;//!count for the order number
  std::vector<int> orders(N, -1);
  get_oracle_actions_calculate_orders(root,tree,orders,timestamp);
  std::vector<int> MPC(N, 0);
  get_oracle_actions_calculate_mpc(root,tree,MPC);
  actions.clear();
  size_t len = heads.size();
  std::vector<int> sigma;
  std::vector<int> beta;
  std::vector<int> heads_rec(N, -1);
  //std::vector<int> output(len, -1);

  //int step = 0;
  for (int i = N - 1; i >= 0; -- i) { beta.push_back(i); }
  while (!(sigma.size() ==1 && beta.empty())) {
    get_oracle_actions_onestep(heads, deprels, tree, heads_rec, sigma, beta, actions,orders,MPC);
  }
}

void ActionUtils::get_oracle_actions_onestep(const std::vector<int>& heads,
    const std::vector<int>& deprels,
    const tree_t& tree,
    std::vector<int>& heads_rec,
    std::vector<int>& sigma,
    std::vector<int>& beta,
    //std::vector<int>& output,
    std::vector<Action>& actions,
    const std::vector<int>& orders,
    const std::vector<int>& MPC) {
  //! the head will be saved in heads record after been reduced
  
  if (sigma.size() < 2) {
    actions.push_back(ActionFactory::make_shift());
    sigma.push_back(beta.back()); beta.pop_back();
    return;
  }

  int top0 = sigma.back();
  int top1 = sigma[sigma.size() - 2];

  //INFO_LOG("step1 %d %d %d",top1,top0,beta.back());
  if (heads[top1] == top0) {
    bool all_found = true;
    for (int c: tree[top1]) { if (heads_rec[c] == -1) { all_found = false; } }
    if (all_found) {
      actions.push_back(ActionFactory::make_left_arc(deprels[top1]));
      sigma.pop_back(); sigma.back() = top0; heads_rec[top1] = top0;
      return;
    }
  }
  if (heads[top0] == top1) {
    bool all_found = true;
    for (int c: tree[top0]) { if (heads_rec[c] == -1) { all_found = false; } }
    if (all_found) {
      actions.push_back(ActionFactory::make_right_arc(deprels[top0]));
      sigma.pop_back(); heads_rec[top0] = top1;
      return;
    }
  }
  int k = beta.empty() ? -1 : beta.back();
  if ((orders[top0] < orders[top1]) &&
      (k == -1 || MPC[top0] != MPC[k])) {
    actions.push_back(ActionFactory::make_swap());
    sigma.pop_back(); sigma.back() = top0; beta.push_back(top1);
  } else {
    actions.push_back(ActionFactory::make_shift());
    sigma.push_back(beta.back()); beta.pop_back();
  }

  /*
  int top0 = (sigma.size() > 0 ? sigma.back(): -1);
  int top1 = (sigma.size() > 1 ? sigma[sigma.size()- 2]: -1);

  bool all_descendents_reduced = true;
  if (top0 >= 0) {
    for (size_t i = 0; i < heads.size(); ++ i) {
      if (heads[i] == top0 && output[i] != top0) {
        // _INFO << i << " " << output[i];
        all_descendents_reduced = false; break; }
    }
  }

  if (top1 >= 0 && heads[top1] == top0) {
    actions.push_back(ActionFactory::make_left_arc(deprels[top1]));
    output[top1] = top0;
    sigma.pop_back();
    sigma.back() = top0;
  } else if (top1 >= 0 && heads[top0] == top1 && all_descendents_reduced) {
    actions.push_back(ActionFactory::make_right_arc(deprels[top0]));
    output[top0] = top1;
    sigma.pop_back();
  } else if (beta < heads.size()) {
    actions.push_back(ActionFactory::make_shift());
    sigma.push_back(beta);
    ++ beta;
  }
  */
}

void ActionUtils::get_oracle_actions_calculate_orders(int root,
    const tree_t& tree,
    std::vector<int>& orders,
    int& timestamp) {
  const std::vector<int>& children = tree[root];
  if (children.size() == 0) {
    orders[root] = timestamp;
    timestamp += 1;
    return;
  }

  int i;
  for (i = 0; i < children.size() && children[i] < root; ++ i) {
    int child = children[i];
    get_oracle_actions_calculate_orders(child, tree, orders, timestamp);
  }

  orders[root] = timestamp;
  timestamp += 1;

  for (; i < children.size(); ++ i) {
    int child = children[i];
    get_oracle_actions_calculate_orders(child, tree, orders, timestamp);
  }
}

ActionUtils::mpc_result_t ActionUtils::get_oracle_actions_calculate_mpc(int root,
    const tree_t& tree,
    std::vector<int>& MPC) {
  const std::vector<int>& children = tree[root];
  if (children.size() == 0) {
    MPC[root] = root;
    return std::make_tuple(true, root, root);
  }

  int left = root, right = root;
  bool overall = true;

  int pivot = -1;
  for (pivot = 0; pivot < children.size() && children[pivot] < root; ++ pivot);

  for (int i = pivot - 1; i >= 0; -- i) {
    int child = children[i];
    ActionUtils::mpc_result_t result =
      get_oracle_actions_calculate_mpc(child, tree, MPC);
    overall = overall && std::get<0>(result);
    if (std::get<0>(result) == true && std::get<2>(result) + 1 == left) {
      left = std::get<1>(result);
    } else {
      overall = false;
    }
  }

  for (int i = pivot; i < children.size(); ++ i) {
    int child = children[i];
    mpc_result_t result = get_oracle_actions_calculate_mpc(child, tree, MPC);
    overall = overall && std::get<0>(result);
    if (std::get<0>(result) == true && right + 1 == std::get<1>(result)) {
      right = std::get<2>(result);
    } else {
      overall = false;
    }
  }

  for (int i = left; i <= right; ++ i) { MPC[i] = root; }

  return std::make_tuple(overall, left, right);
}


State::State(): ref(0) { clear(); }
State::State(const Dependency* r): ref(r) {
  clear();
  size_t L = r->size();
  for(int i=L-1;i>=0;i--)
    buffer.push_back(i);
  heads.resize(L, -1);
  deprels.resize(L, 0);
  nr_left_children.resize(L, 0);
  nr_right_children.resize(L, 0);
  left_most_child.resize(L, -1);
  right_most_child.resize(L, -1);
  left_2nd_most_child.resize(L, -1);
  right_2nd_most_child.resize(L, -1);
}

bool State::can_shift() const     { return !buffer_empty(); }
bool State::can_left_arc() const  { return stack_size() >= 2; }
bool State::can_right_arc() const { return stack_size() >= 2; }
bool State::can_swap() const     { return stack_size() > 2; }
bool State::is_complete() const { return stack.size() == 1 && buffer.empty(); }

void State::copy(const State& source) {
  this->ref = source.ref;
  this->score = source.score;
  this->previous = source.previous;
  this->buffer= source.buffer;
  this->top0 = source.top0;
  this->top1 = source.top1;
  this->stack = source.stack;
  this->last_action = source.last_action;
  this->heads = source.heads;
  this->deprels = source.deprels;
  this->left_most_child = source.left_most_child;
  this->right_most_child = source.right_most_child;
  this->left_2nd_most_child = source.left_2nd_most_child;
  this->right_2nd_most_child = source.right_2nd_most_child;
  this->nr_left_children = source.nr_left_children;
  this->nr_right_children = source.nr_right_children;
}

void State::clear() {
  this->score = 0;
  this->previous = 0;
  this->top0 = -1;
  this->top1 = -1;
  buffer.clear();
  stack.clear();
  std::fill(heads.begin(), heads.end(), -1);
  std::fill(deprels.begin(), deprels.end(), 0);
  std::fill(nr_left_children.begin(), nr_left_children.end(), 0);
  std::fill(nr_right_children.begin(), nr_right_children.end(), 0);
  std::fill(left_most_child.begin(), left_most_child.end(), -1);
  std::fill(right_most_child.begin(), right_most_child.end(), -1);
  std::fill(left_2nd_most_child.begin(), left_2nd_most_child.end(), -1);
  std::fill(right_2nd_most_child.begin(), right_2nd_most_child.end(), -1);
}

void State::refresh_stack_information() {
  size_t sz = stack.size();
  if (0 == sz) {
    top0 = -1;
    top1 = -1;
  } else if (1 == sz) {
    top0 = stack.at(sz - 1);
    top1 = -1;
  } else {
    top0 = stack.at(sz - 1);
    top1 = stack.at(sz - 2);
  }
}

bool State::shift(const State& source) {
  if (!source.can_shift()) { return false; }

  this->copy(source);
  stack.push_back(buffer.back()); buffer.pop_back();
  refresh_stack_information();

  this->last_action = ActionFactory::make_shift();
  this->previous = &source;
  return true;
}

bool State::left_arc(const State& source, int deprel) {
  if (!source.can_left_arc()) { return false; }

  this->copy(source);
  stack.pop_back();
  stack.back() = top0;

  heads[top1] = top0;
  deprels[top1] = deprel;

  if (-1 == left_most_child[top0]) {
    // TP0 is left-isolate node.
    left_most_child[top0] = top1;
  } else if (top1 < left_most_child[top0]) {
    // (TP1, LM0, TP0)
    left_2nd_most_child[top0] = left_most_child[top0];
    left_most_child[top0] = top1;
  } else if (top1 < left_2nd_most_child[top0]) {
    // (LM0, TP1, TP0)
    left_2nd_most_child[top0] = top1;
  }

  ++ nr_left_children[top0];
  refresh_stack_information();
  this->last_action = ActionFactory::make_left_arc(deprel);
  this->previous = &source;
  return true;
}

bool State::right_arc(const State& source, int deprel) {
  if (!source.can_right_arc()) { return false; }

  this->copy(source);
  stack.pop_back();
  heads[top0] = top1;
  deprels[top0] = deprel;

  if (-1 == right_most_child[top1]) {
    // TP1 is right-isolate node.
    right_most_child[top1] = top0;
  } else if (right_most_child[top1] < top0) {
    right_2nd_most_child[top1] = right_most_child[top1];
    right_most_child[top1] = top0;
  } else if (right_2nd_most_child[top1] < top0) {
    right_2nd_most_child[top1] = top0;
  }
  ++ nr_right_children[top1];
  refresh_stack_information();
  this->last_action = ActionFactory::make_right_arc(deprel);
  this->previous = &source;
  return true;
}

bool State::swap(const State& source) {
  if (!source.can_swap()) { return false; }

  this->copy(source);
  stack.pop_back(); stack.back() = top0; buffer.push_back(top1);
  refresh_stack_information();
  this->last_action = ActionFactory::make_swap();
  this->previous = &source;
  return true;
}

int State::cost(const std::vector<int>& gold_heads,
    const std::vector<int>& gold_deprels) {
  std::vector< std::vector<int> > tree(gold_heads.size());
  for (size_t i = 0; i < gold_heads.size(); ++ i) {
    int h = gold_heads[i]; if (h >= 0) { tree[h].push_back(i); }
  }

  const std::vector<int>& sigma_l = stack;
  std::vector<int> sigma_r; sigma_r.push_back(stack.back());

  std::vector<bool> sigma_l_mask(gold_heads.size(), false);
  std::vector<bool> sigma_r_mask(gold_heads.size(), false);
  for (size_t s = 0; s < sigma_l.size(); ++ s) { sigma_l_mask[sigma_l[s]]= true; }

  for (int i = buffer.back(); i < ref->size(); ++ i) {
    if (gold_heads[i] < buffer.back()) {
      sigma_r.push_back(i);
      sigma_r_mask[i] = true;
      continue;
    }

    const std::vector<int>& node = tree[i];
    for (size_t d = 0; d < node.size(); ++ d) {
      if (sigma_l_mask[node[d]] || sigma_r_mask[node[d]]) {
        sigma_r.push_back(i);
        sigma_r_mask[i] = true;
        break;
      }
    }
  }

  int len_l = sigma_l.size();
  int len_r = sigma_r.size();

  // typedef boost::multi_array<int, 3> array_t;
  // array_t T(boost::extents[len_l][len_r][len_l+len_r-1]);
  // std::fill( T.origin(), T.origin()+ T.num_elements(), 1024);
  Mat3<int> T(len_l, len_r, len_l+len_r-1);
  T = 1024;

  T[0][0][len_l-1]= 0;
  for (int d = 0; d < len_l+len_r- 1; ++ d) {
    for (int j = std::max(0, d-len_l+1); j < std::min(d+1, len_r); ++ j) {
      int i = d-j;
      if (i < len_l-1) {
        int i_1 = sigma_l[len_l-i-2];
        int i_1_rank = len_l-i-2;
        for (int rank = len_l-i-1; rank < len_l; ++ rank) {
          int h = sigma_l[rank];
          int h_rank = rank;
          T[i+1][j][h_rank] = std::min(T[i+1][j][h_rank],
              T[i][j][h_rank] + (gold_heads[i_1] == h ? 0: 2));
          T[i+1][j][i_1_rank] = std::min(T[i+1][j][i_1_rank],
              T[i][j][h_rank] + (gold_heads[h] == i_1 ? 0: 2));
        }
        for (int rank = 1; rank < j+1; ++ rank) {
          int h =sigma_r[rank];
          int h_rank = len_l+rank-1;
          T[i+1][j][h_rank] = std::min(T[i+1][j][h_rank],
              T[i][j][h_rank] + (gold_heads[i_1] == h ? 0: 2));
          T[i+1][j][i_1_rank] = std::min(T[i+1][j][i_1_rank],
              T[i][j][h_rank] + (gold_heads[h] == i_1 ? 0: 2));
        }
      }
      if (j < len_r-1) {
        int j_1 = sigma_r[j+1];
        int j_1_rank = len_l+j;
        for (int rank = len_l-i-1; rank < len_l; ++ rank) {
          int h = sigma_l[rank];
          int h_rank = rank;
          T[i][j+1][h_rank] = std::min(T[i][j+1][h_rank],
              T[i][j][h_rank] + (gold_heads[j_1] == h ? 0: 2));
          T[i][j+1][j_1_rank] = std::min(T[i][j+1][j_1_rank],
              T[i][j][h_rank] + (gold_heads[h] == j_1 ? 0: 2));
        }
        for (int rank = 1; rank < j+1; ++ rank) {
          int h =sigma_r[rank];
          int h_rank = len_l+rank-1;
          T[i][j+1][h_rank] = std::min(T[i][j+1][h_rank],
              T[i][j][h_rank] + (gold_heads[j_1] == h ? 0: 2));
          T[i][j+1][j_1_rank] = std::min(T[i][j+1][j_1_rank],
              T[i][j][h_rank] + (gold_heads[h] == j_1 ? 0: 2));
        }
      }
    }
  }
  int penalty = 0;
  for (int i = 0; i < buffer.back(); ++ i) {
    if (heads[i] != -1) {
      if (heads[i] != gold_heads[i]) { penalty += 2; }
      else if (deprels[i] != gold_deprels[i]) { penalty += 1; }
    }
  }
  return T[len_l-1][len_r-1][0]+ penalty;
}

bool State::buffer_empty() const { return (this->buffer.size()==0); }
size_t State::stack_size() const { return (this->stack.size()); }

TransitionSystem::TransitionSystem(): L(0), R(-1), D(-1) {}

void TransitionSystem::set_dummy_relation(int d) { D = d; }

void TransitionSystem::set_root_relation(int r) { R = r; }

// l should exclude Special::ROOT and Special::NIL, but the root, which is linked
// to zero should be included.
void TransitionSystem::set_number_of_relations(size_t l) { L = l; }

void TransitionSystem::get_possible_actions(const State& source,
    std::vector<Action>& actions) {
  if (0 == L || -1 == R) {
    WARNING_LOG("decoder: not initialized,"
        " please check if the root dependency relation is correct set by --root.");
    return;
  }
  actions.clear();
 // INFO_LOG("buffer len %d",source.buffer.size());
  if (!source.buffer_empty()) {
    actions.push_back(ActionFactory::make_shift());
  }

  if (source.stack_size() == 2) {
    if (source.buffer_empty()) {
      actions.push_back(ActionFactory::make_right_arc(R));
    }
  } else if (source.stack_size() > 2) {
    for (size_t l = 0; l < L; ++ l) {
      if (l == R) { continue; }
      actions.push_back(ActionFactory::make_left_arc(l));
      actions.push_back(ActionFactory::make_right_arc(l));
    }
    if (source.top1<source.top0){
      actions.push_back(ActionFactory::make_swap());
    }
  }
}

void TransitionSystem::transit(const State& source, const Action& act, State* target) {
  int deprel;
  if (ActionUtils::is_shift(act)) {
    target->shift(source);
  } else if (ActionUtils::is_left_arc(act, deprel)) {
    target->left_arc(source, deprel);
  } else if (ActionUtils::is_right_arc(act, deprel)) {
    target->right_arc(source, deprel);
  } else if (ActionUtils::is_swap(act)) {
    target->swap(source);
  } else {
    WARNING_LOG("unknown transition in transit: %d-%d", act.name(), act.rel());
  }
}

std::vector<int> TransitionSystem::transform(const std::vector<Action>& actions) {
  std::vector<int> classes;
  transform(actions, classes);
  return classes;
}

void TransitionSystem::transform(const std::vector<Action>& actions,
    std::vector<int>& classes) {
  classes.clear();
  for (size_t i = 0; i < actions.size(); ++ i) {
    classes.push_back( transform(actions[i]) );
  }
}

int TransitionSystem::transform(const Action& act) {
  int deprel;
  if (ActionUtils::is_shift(act)) { return 0; }
  else if (ActionUtils::is_left_arc(act, deprel))  { return 1+ deprel; }
  else if (ActionUtils::is_right_arc(act, deprel)) { return L+ 1+ deprel; }
  else if (ActionUtils::is_swap(act)) { return 2*L+1; }
  else {
    WARNING_LOG("unknown transition in transform(Action&): %d-%d", act.name(), act.rel()); }
  return -1;
}

Action TransitionSystem::transform(int act) {
  if (act == 0) { return ActionFactory::make_shift(); }
  else if (act < 1+L) { return ActionFactory::make_left_arc(act- 1); }
  else if (act < 1+2*L) { return ActionFactory::make_right_arc(act- 1- L); }
  else if (act == 1+2*L) { return ActionFactory::make_swap(); }
  else { WARNING_LOG("unknown transition in transform(int&): %d", act); }
  return Action();
}

size_t TransitionSystem::number_of_transitions() const { return L*2+2; }


} //  namespace depparser
} //  namespace ltp

