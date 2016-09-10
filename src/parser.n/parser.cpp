#include <fstream>
#include "parser.n/parser.h"
#include "utils/logging.hpp"
#include "utils/math/fast_binned.h"

namespace ltp {
namespace depparser {

NeuralNetworkParser::NeuralNetworkParser()
  : classifier(W1, W2, E, b1, saved, precomputation_id_encoder),
  use_cluster(false), use_distance(false), use_valency(false) {}

const std::string NeuralNetworkParser::model_header = "nnparser";

void NeuralNetworkParser::predict(const Instance& data, std::vector<int>& heads,
    std::vector<std::string>& deprels) {
  Dependency dependency;
  std::vector<int> cluster, cluster4, cluster6;
  transduce_instance_to_dependency(data, &dependency, false);
  get_cluster_from_dependency(dependency, cluster4, cluster6, cluster);

  size_t L = data.forms.size();
  std::vector<State> states(L*2);
  states[0].copy(State(&dependency));
  system.transit(states[0], ActionFactory::make_shift(), &states[1]);
  for (size_t step = 1; step < L*2-1; ++ step) {
    std::vector<int> attributes;
    if (use_cluster) {
      get_features(states[step], cluster4, cluster6, cluster, attributes);
    } else {
      get_features(states[step], attributes);
    }

    std::vector<double> scores(system.number_of_transitions(), 0);
    classifier.score(attributes, scores);

    std::vector<Action> possible_actions;
    system.get_possible_actions(states[step], possible_actions);

    size_t best = -1;
    for (size_t j = 0; j < possible_actions.size(); ++ j) {
      int l = system.transform(possible_actions[j]);
      if (best == -1 || scores[best] < scores[l]) { best = l; }
    }

    Action act = system.transform(best);
    system.transit(states[step], act, &states[step+ 1]);
  }

  heads.resize(L);
  deprels.resize(L);
  for (size_t i = 0; i < L; ++ i) {
    heads[i] = states[L*2-1].heads[i];
    deprels[i] = deprels_alphabet.at(states[L*2-1].deprels[i]);
  }
}


void NeuralNetworkParser::get_context(const State& s, Context* ctx) {
  ctx->S0 = (s.stack.size() > 0 ? s.stack[s.stack.size() - 1]: -1);
  ctx->S1 = (s.stack.size() > 1 ? s.stack[s.stack.size() - 2]: -1);
  ctx->S2 = (s.stack.size() > 2 ? s.stack[s.stack.size() - 3]: -1);
  ctx->N0 = (s.buffer < s.ref->size()    ? s.buffer:    -1);
  ctx->N1 = (s.buffer+ 1 < s.ref->size() ? s.buffer+ 1: -1);
  ctx->N2 = (s.buffer+ 2 < s.ref->size() ? s.buffer+ 2: -1);

  ctx->S0L  = (ctx->S0 >= 0  ? s.left_most_child[ctx->S0]:  -1);
  ctx->S0R  = (ctx->S0 >= 0  ? s.right_most_child[ctx->S0]: -1);
  ctx->S0L2 = (ctx->S0 >= 0  ? s.left_2nd_most_child[ctx->S0]:  -1);
  ctx->S0R2 = (ctx->S0 >= 0  ? s.right_2nd_most_child[ctx->S0]: -1);
  ctx->S0LL = (ctx->S0L >= 0 ? s.left_most_child[ctx->S0L]:  -1);
  ctx->S0RR = (ctx->S0R >= 0 ? s.right_most_child[ctx->S0R]: -1);

  ctx->S1L  = (ctx->S1 >= 0  ? s.left_most_child[ctx->S1]:  -1);
  ctx->S1R  = (ctx->S1 >= 0  ? s.right_most_child[ctx->S1]: -1);
  ctx->S1L2 = (ctx->S1 >= 0  ? s.left_2nd_most_child[ctx->S1]:  -1);
  ctx->S1R2 = (ctx->S1 >= 0  ? s.right_2nd_most_child[ctx->S1]: -1);
  ctx->S1LL = (ctx->S1L >= 0 ? s.left_most_child[ctx->S1L]: -1);
  ctx->S1RR = (ctx->S1R >= 0 ? s.right_most_child[ctx->S1R]: -1);
}

void NeuralNetworkParser::get_features(const State& s,
    std::vector<int>& features) {
  Context ctx;
  get_context(s, &ctx);
  get_basic_features(ctx, s.ref->forms, s.ref->postags, s.deprels, features);
  get_distance_features(ctx, features);
  get_valency_features(ctx, s.nr_left_children, s.nr_right_children, features);
}

void NeuralNetworkParser::get_features(const State& s,
    const std::vector<int>& cluster4,
    const std::vector<int>& cluster6,
    const std::vector<int>& cluster,
    std::vector<int>& features) {
  Context ctx;
  get_context(s, &ctx);
  get_basic_features(ctx, s.ref->forms, s.ref->postags, s.deprels, features);
  get_distance_features(ctx, features);
  get_valency_features(ctx, s.nr_left_children, s.nr_right_children, features);
  get_cluster_features(ctx, cluster4, cluster6, cluster, features);
}

void NeuralNetworkParser::get_basic_features(const Context& ctx,
    const std::vector<int>& forms,
    const std::vector<int>& postags,
    const std::vector<int>& deprels,
    std::vector<int>& features) {
#define FORM(id)    ((ctx.id != -1) ? (forms[ctx.id]): kNilForm)
#define POSTAG(id)  ((ctx.id != -1) ? (postags[ctx.id]+ kPostagInFeaturespace): kNilPostag)
#define DEPREL(id)  ((ctx.id != -1) ? (deprels[ctx.id]+ kDeprelInFeaturespace): kNilDeprel)
#define PUSH(feat)  do { features.push_back( feat ); } while (0);
  PUSH( FORM(S0) );   PUSH( POSTAG(S0) );
  PUSH( FORM(S1) );   PUSH( POSTAG(S1) );
  PUSH( FORM(S2) );   PUSH( POSTAG(S2) );
  PUSH( FORM(N0) );   PUSH( POSTAG(N0) );
  PUSH( FORM(N1) );   PUSH( POSTAG(N1) );
  PUSH( FORM(N2) );   PUSH( POSTAG(N2) );
  PUSH( FORM(S0L) );  PUSH( POSTAG(S0L) );  PUSH( DEPREL(S0L) );
  PUSH( FORM(S0R) );  PUSH( POSTAG(S0R) );  PUSH( DEPREL(S0R) );
  PUSH( FORM(S0L2) ); PUSH( POSTAG(S0L2) ); PUSH( DEPREL(S0L2) );
  PUSH( FORM(S0R2) ); PUSH( POSTAG(S0R2) ); PUSH( DEPREL(S0R2) );
  PUSH( FORM(S0LL) ); PUSH( POSTAG(S0LL) ); PUSH( DEPREL(S0LL) );
  PUSH( FORM(S0RR) ); PUSH( POSTAG(S0RR) ); PUSH( DEPREL(S0RR) );
  PUSH( FORM(S1L) );  PUSH( POSTAG(S1L) );  PUSH( DEPREL(S1L) );
  PUSH( FORM(S1R) );  PUSH( POSTAG(S1R) );  PUSH( DEPREL(S1R) );
  PUSH( FORM(S1L2) ); PUSH( POSTAG(S1L2) ); PUSH( DEPREL(S1L2) );
  PUSH( FORM(S1R2) ); PUSH( POSTAG(S1R2) ); PUSH( DEPREL(S1R2) );
  PUSH( FORM(S1LL) ); PUSH( POSTAG(S1LL) ); PUSH( DEPREL(S1LL) );
  PUSH( FORM(S1RR) ); PUSH( POSTAG(S1RR) ); PUSH( DEPREL(S1RR) );
#undef FORM
#undef POSTAG
#undef DEPREL
#undef PUSH
}

void NeuralNetworkParser::get_distance_features(const Context& ctx,
    std::vector<int>& features) {
  if (!use_distance) { return; }

  size_t dist = 8;
  if (ctx.S0 >= 0 && ctx.S1 >= 0) {
    dist = math::binned_1_2_3_4_5_6_10[ctx.S0-ctx.S1];
    if (dist == 10) { dist = 7; }
  }
  features.push_back(dist+ kDistanceInFeaturespace);
}

void NeuralNetworkParser::get_valency_features(const Context& ctx,
    const std::vector<int>& nr_left_children,
    const std::vector<int>& nr_right_children,
    std::vector<int>& features) {
  if (!use_valency) { return; }

  size_t lvc = 8;
  size_t rvc = 8;
  if (ctx.S0 >= 0) {
    lvc = math::binned_1_2_3_4_5_6_10[nr_left_children[ctx.S0]];
    rvc = math::binned_1_2_3_4_5_6_10[nr_right_children[ctx.S0]];
    if (lvc == 10) { lvc = 7; }
    if (rvc == 10) { rvc = 7; }
  }
  features.push_back(lvc+kValencyInFeaturespace);
  features.push_back(rvc+kValencyInFeaturespace);

  lvc = 8;
  rvc = 8;
  if (ctx.S1 >= 0) {
    lvc = math::binned_1_2_3_4_5_6_10[nr_left_children[ctx.S1]];
    rvc = math::binned_1_2_3_4_5_6_10[nr_right_children[ctx.S1]];
    if (lvc == 10) { lvc = 7; }
    if (rvc == 10) { rvc = 7; }
  }
  features.push_back(lvc+kValencyInFeaturespace);
  features.push_back(rvc+kValencyInFeaturespace);
}

void NeuralNetworkParser::get_cluster_features(const Context& ctx,
    const std::vector<int>& cluster4,
    const std::vector<int>& cluster6,
    const std::vector<int>& cluster,
    std::vector<int>& features) {
  if (!use_cluster) { return; }

#define CLUSTER(id)  (ctx.id >= 0 ? (cluster[ctx.id]+kClusterInFeaturespace): kNilCluster)
#define CLUSTER4(id) (ctx.id >= 0 ? (cluster4[ctx.id]+kCluster4InFeaturespace): kNilCluster4)
#define CLUSTER6(id) (ctx.id >= 0 ? (cluster6[ctx.id]+kCluster6InFeaturespace): kNilCluster6)
#define PUSH(feat)  do { features.push_back( feat ); } while (0);
  PUSH( CLUSTER(S0) );    PUSH( CLUSTER4(S0) );     PUSH( CLUSTER6(S0) );
  PUSH( CLUSTER(S1) );
  PUSH( CLUSTER(S2) );
  PUSH( CLUSTER(N0) );    PUSH( CLUSTER4(N0) );     PUSH( CLUSTER6(N0) );
  PUSH( CLUSTER(N1) );
  PUSH( CLUSTER(N2) );
  PUSH( CLUSTER(S0L) );
  PUSH( CLUSTER(S0R) );
  PUSH( CLUSTER(S0L2) );
  PUSH( CLUSTER(S0R2) );
  PUSH( CLUSTER(S0LL) );
  PUSH( CLUSTER(S0RR) );
  PUSH( CLUSTER(S1L) );
  PUSH( CLUSTER(S1R) );
  PUSH( CLUSTER(S1L2) );
  PUSH( CLUSTER(S1R2) );
  PUSH( CLUSTER(S1LL) );
  PUSH( CLUSTER(S1RR) );
#undef CLUSTER
#undef CLUSTER4
#undef CLUSTER6
#undef PUSH
}

void NeuralNetworkParser::report() {
  INFO_LOG("#: loaded %d forms", forms_alphabet.size());
  INFO_LOG("#: loaded %d postags", postags_alphabet.size());
  INFO_LOG("#: loaded %d deprels", deprels_alphabet.size());

  if (use_cluster) {
    INFO_LOG("#: loaded %d cluster(4)", cluster4_types_alphabet.size());
    INFO_LOG("#: loaded %d cluster(6)", cluster6_types_alphabet.size());
    INFO_LOG("#: loaded %d cluster", cluster_types_alphabet.size());
  }

  INFO_LOG("report: form located at: [%d ... %d]", kFormInFeaturespace,
      kPostagInFeaturespace- 1);
  INFO_LOG("report: postags located at: [%d ... %d]", kPostagInFeaturespace,
      kDeprelInFeaturespace- 1);
  INFO_LOG("report: deprels located at: [%d ... %d]", kDeprelInFeaturespace,
      kDistanceInFeaturespace- 1);
  if (use_distance) {
    INFO_LOG("report: distance located at: [%d ... %d]", kDistanceInFeaturespace,
        kValencyInFeaturespace- 1);
  }
  if (use_valency) {
    INFO_LOG("report: valency located at: [%d ... %d]", kValencyInFeaturespace,
        kCluster4InFeaturespace- 1);
  }
  if (use_cluster) {
    INFO_LOG("report: cluster4 located at: [%d ... %d]", kCluster4InFeaturespace,
        kCluster6InFeaturespace- 1);
    INFO_LOG("report: cluster6 located at: [%d ... %d]", kCluster6InFeaturespace,
        kClusterInFeaturespace- 1);
    INFO_LOG("report: cluster located at: [%d ... %d]", kClusterInFeaturespace,
        kFeatureSpaceEnd- 1);
  }
  INFO_LOG("report: nil form (in f.s.) =%d", kNilForm);
  INFO_LOG("report: nil postag (in f.s.) =%d", kNilPostag);
  INFO_LOG("report: nil deprel (in f.s.) =%d", kNilDeprel);
  if (use_distance) {
    INFO_LOG("report: nil distance (in f.s.) =%d", kNilDistance);
  }
  if (use_valency) {
    INFO_LOG("report: nil valency (in f.s.) =%d", kNilValency);
  }
  if (use_cluster) {
    INFO_LOG("report: nil cluster4 (in f.s.) =%d", kNilCluster4);
    INFO_LOG("report: nil cluster6 (in f.s.) =%d", kNilCluster6);
    INFO_LOG("report: nil cluster (in f.s.) =%d", kNilCluster);
  }
}

void NeuralNetworkParser::transduce_instance_to_dependency(const Instance& data,
    Dependency* dependency, bool with_dependencies) {
  size_t L = data.size();
  for (size_t i = 0; i < L; ++ i) {
    int form = forms_alphabet.index(data.forms[i]);
    if (form == -1) { form = forms_alphabet.index(SpecialOption::UNKNOWN); }
    int postag = postags_alphabet.index(data.postags[i]);
    int head = with_dependencies? data.heads[i]: -1;
    int deprel = (with_dependencies ? deprels_alphabet.index(data.deprels[i]): -1);

    dependency->forms.push_back(form);
    dependency->postags.push_back(postag);
    dependency->heads.push_back(head);
    dependency->deprels.push_back(deprel);
  }
}

void NeuralNetworkParser::get_cluster_from_dependency(const Dependency& data,
    std::vector<int>& cluster4,
    std::vector<int>& cluster6,
    std::vector<int>& cluster) {
  if (use_cluster) {
    size_t L = data.size();
    for (size_t i = 0; i < L; ++ i) {
      int form = data.forms[i];
      cluster4.push_back(i == 0?
          cluster4_types_alphabet.index(SpecialOption::ROOT): form_to_cluster4[form]);
      cluster6.push_back(i == 0?
          cluster6_types_alphabet.index(SpecialOption::ROOT): form_to_cluster6[form]);
      cluster.push_back(i == 0?
          cluster_types_alphabet.index(SpecialOption::ROOT): form_to_cluster[form]);
    }
  }
}

template<class Matrix> void NeuralNetworkParser::write_matrix(std::ostream& os, const Matrix& mat) {
  unsigned long long rows = mat.rows(), cols = mat.cols();
  os.write((char*) (&rows), sizeof(unsigned long long));
  os.write((char*) (&cols), sizeof(unsigned long long));
  os.write((char*) mat.data(), rows * cols * sizeof(typename Matrix::Scalar) );
}

template<class Matrix> void NeuralNetworkParser::read_matrix(std::istream& is, Matrix& mat) {
  unsigned long long rows=0, cols=0;
  is.read((char*) (&rows),sizeof(unsigned long long));
  is.read((char*) (&cols),sizeof(unsigned long long));
  mat.resize(rows, cols);
  is.read((char *) mat.data() , rows * cols * sizeof(typename Matrix::Scalar));
}

template<class Vector> void NeuralNetworkParser::write_vector(std::ostream& os, const Vector& vec) {
  unsigned long long rows = vec.rows();
  os.write((char*) (&rows), sizeof(unsigned long long));
  os.write((char*) vec.data(), rows * sizeof(typename Vector::Scalar) );
}

template<class Vector> void NeuralNetworkParser::read_vector(std::istream& is, Vector& vec) {
  unsigned long long rows = 0;
  is.read((char*) (&rows), sizeof(unsigned long long));
  vec.resize(rows);
  is.read((char *) vec.data() , rows * sizeof(typename Vector::Scalar));
}

void NeuralNetworkParser::save(const std::string& filename) {
  std::ofstream ofs(filename.c_str(), std::ofstream::binary);
  char chunk[128];

  memset(chunk, 0, sizeof(chunk));
  strncpy(chunk, model_header.c_str(), 128);
  ofs.write(chunk, 128);

  memset(chunk, 0, sizeof(chunk));
  strncpy(chunk, root.c_str(), 128);
  ofs.write(chunk, 128);

  ofs.write(reinterpret_cast<const char*>(&use_distance), sizeof(bool));
  ofs.write(reinterpret_cast<const char*>(&use_valency), sizeof(bool));
  ofs.write(reinterpret_cast<const char*>(&use_cluster), sizeof(bool));

  /*W1.save(ofs);
  W2.save(ofs);
  E.save(ofs);
  b1.save(ofs);
  saved.save(ofs);*/
  write_matrix(ofs, W1);
  write_matrix(ofs, W2);
  write_matrix(ofs, E);
  write_vector(ofs, b1);
  write_matrix(ofs, saved);

  forms_alphabet.dump(ofs);
  postags_alphabet.dump(ofs);
  deprels_alphabet.dump(ofs);

  int* payload = 0;
  size_t now = 0;
  payload = new int[precomputation_id_encoder.size()* 2];
  for (std::unordered_map<int, size_t>::const_iterator rep = precomputation_id_encoder.begin();
      rep != precomputation_id_encoder.end(); ++ rep) {
    payload[now++] = rep->first;
    payload[now++] = rep->second;
  }
  write_uint(ofs, now);
  ofs.write(reinterpret_cast<const char*>(payload), sizeof(int)*now);
  delete payload;

  if (use_cluster) {
    cluster4_types_alphabet.dump(ofs);
    cluster6_types_alphabet.dump(ofs);
    cluster_types_alphabet.dump(ofs);

    payload = new int[form_to_cluster.size()* 2];

    now = 0;
    for (std::unordered_map<int, int>::const_iterator rep = form_to_cluster4.begin();
        rep != form_to_cluster4.end(); ++ rep) {
      payload[now++] = rep->first;
      payload[now++] = rep->second;
    }
    write_uint(ofs, now);
    ofs.write(reinterpret_cast<const char*>(payload), sizeof(int)* now);

    now = 0;
    for (std::unordered_map<int, int>::const_iterator rep = form_to_cluster6.begin();
        rep != form_to_cluster6.end(); ++ rep) {
      payload[now++] = rep->first;
      payload[now++] = rep->second;
    }
    write_uint(ofs, now);
    ofs.write(reinterpret_cast<const char*>(payload), sizeof(int)* now);

    now = 0;
    for (std::unordered_map<int, int>::const_iterator rep = form_to_cluster.begin();
        rep != form_to_cluster.end(); ++ rep) {
      payload[now++] = rep->first;
      payload[now++] = rep->second;
    }
    write_uint(ofs, now);
    ofs.write(reinterpret_cast<const char*>(payload), sizeof(int)* now);
    delete payload;
  }
}

bool NeuralNetworkParser::load(const std::string& filename) {
  std::ifstream ifs(filename.c_str(), std::ifstream::binary);
  if (!ifs.good()) {
    return false;
  }
  char chunk[128];
  ifs.read(chunk, 128);
  if (strcmp(chunk, model_header.c_str())) {
    return false;
  }

  ifs.read(chunk, 128);
  root = chunk;

  ifs.read(reinterpret_cast<char*>(&use_distance), sizeof(bool));
  ifs.read(reinterpret_cast<char*>(&use_valency), sizeof(bool));
  ifs.read(reinterpret_cast<char*>(&use_cluster), sizeof(bool));

  /*W1.load(ifs);
  W2.load(ifs);
  E.load(ifs);
  b1.load(ifs);
  saved.load(ifs);*/

  read_matrix(ifs, W1);
  read_matrix(ifs, W2);
  read_matrix(ifs, E);
  read_vector(ifs, b1);
  read_matrix(ifs, saved);

  forms_alphabet.load(ifs);
  postags_alphabet.load(ifs);
  deprels_alphabet.load(ifs);

  int* payload = NULL;
  size_t now = 0;
  now= read_uint(ifs);
  payload = new int[now];
  ifs.read(reinterpret_cast<char*>(payload), now*sizeof(int));
  for (size_t i = 0; i < now; i += 2) {
    precomputation_id_encoder[payload[i]] = payload[i+1];
  }
  delete [] payload;

  if (use_cluster) {
    cluster4_types_alphabet.load(ifs);
    cluster6_types_alphabet.load(ifs);
    cluster_types_alphabet.load(ifs);

    now= read_uint(ifs);
    payload = new int[now];
    ifs.read(reinterpret_cast<char*>(payload), now*sizeof(int));
    for (size_t i = 0; i < now; i += 2) {
      form_to_cluster4[payload[i]] = payload[i+1];
    }
    delete [] payload;

    now= read_uint(ifs);
    payload = new int[now];
    ifs.read(reinterpret_cast<char*>(payload), now*sizeof(int));
    for (size_t i = 0; i < now; i += 2) {
      form_to_cluster6[payload[i]] = payload[i+1];
    }
    delete [] payload;

    now= read_uint(ifs);
    payload = new int[now];
    ifs.read(reinterpret_cast<char*>(payload), now*sizeof(int));
    for (size_t i = 0; i < now; i += 2) {
      form_to_cluster[payload[i]] = payload[i+1];
    }
    delete [] payload;
  }

  classifier.canonical();
  return true;
}

void NeuralNetworkParser::setup_system() {
  // system.set_dummy_relation(deprels_alphabet.index(SpecialOption::ROOT));
  system.set_root_relation(deprels_alphabet.index(root));
  system.set_number_of_relations(deprels_alphabet.size()- 2);
}

void NeuralNetworkParser::build_feature_space() {
  kFormInFeaturespace = 0;
  kNilForm = forms_alphabet.index(SpecialOption::NIL);
  kFeatureSpaceEnd = forms_alphabet.size();

  kPostagInFeaturespace = kFeatureSpaceEnd;
  kNilPostag = kFeatureSpaceEnd+ postags_alphabet.index(SpecialOption::NIL);
  kFeatureSpaceEnd += postags_alphabet.size();

  kDeprelInFeaturespace = kFeatureSpaceEnd;
  kNilDeprel = kFeatureSpaceEnd+ deprels_alphabet.index(SpecialOption::NIL);
  kFeatureSpaceEnd += deprels_alphabet.size();

  kDistanceInFeaturespace = kFeatureSpaceEnd;
  kNilDistance = kFeatureSpaceEnd+ (use_distance ? 8: 0);
  kFeatureSpaceEnd += (use_distance? 9: 0);

  kValencyInFeaturespace = kFeatureSpaceEnd;
  kNilValency = kFeatureSpaceEnd+ (use_valency? 8: 0);
  kFeatureSpaceEnd += (use_valency? 9: 0);

  kCluster4InFeaturespace = kFeatureSpaceEnd;
  if (use_cluster) {
    kNilCluster4 = kFeatureSpaceEnd+ cluster4_types_alphabet.index(SpecialOption::NIL);
    kFeatureSpaceEnd += cluster4_types_alphabet.size();
  } else { kNilCluster4 = kFeatureSpaceEnd; }

  kCluster6InFeaturespace = kFeatureSpaceEnd;
  if (use_cluster) {
    kNilCluster6 = kFeatureSpaceEnd+ cluster6_types_alphabet.index(SpecialOption::NIL);
    kFeatureSpaceEnd += cluster6_types_alphabet.size();
  } else { kNilCluster6 = kFeatureSpaceEnd; }

  kClusterInFeaturespace = kFeatureSpaceEnd;
  if (use_cluster) {
    kNilCluster = kFeatureSpaceEnd+ cluster_types_alphabet.index(SpecialOption::NIL);
    kFeatureSpaceEnd += cluster_types_alphabet.size();
  } else { kNilCluster = kFeatureSpaceEnd; }

}

} //  namespace depparser
} //  namespace ltp
