#include "parser.n/parser.h"
#include "utils/logging.hpp"
#include "utils/math/fast_binned.h"

namespace ltp {
namespace depparser {

NeuralNetworkParser::NeuralNetworkParser() {}

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
  TRACE_LOG("report: form located at: [%d ... %d]", kFormInFeaturespace,
      kPostagInFeaturespace- 1);
  TRACE_LOG("report: postags located at: [%d ... %d]", kPostagInFeaturespace,
      kDeprelInFeaturespace- 1);
  TRACE_LOG("report: deprels located at: [%d ... %d]", kDeprelInFeaturespace,
      kDistanceInFeaturespace- 1);
  if (use_distance) {
    TRACE_LOG("report: distance located at: [%d ... %d]", kDistanceInFeaturespace,
        kValencyInFeaturespace- 1);
  }
  if (use_valency) {
    TRACE_LOG("report: valency located at: [%d ... %d]", kValencyInFeaturespace,
        kCluster4InFeaturespace- 1);
  }
  if (use_cluster) {
    TRACE_LOG("report: cluster4 located at: [%d ... %d]", kCluster4InFeaturespace,
        kCluster6InFeaturespace- 1);
    TRACE_LOG("report: cluster6 located at: [%d ... %d]", kCluster6InFeaturespace,
        kClusterInFeaturespace- 1);
    TRACE_LOG("report: cluster located at: [%d ... %d]", kClusterInFeaturespace,
        kFeatureSpaceEnd- 1);
  }
  TRACE_LOG("report: nil form (in f.s.) =%d", kNilForm);
  TRACE_LOG("report: nil postag (in f.s.) =%d", kNilPostag);
  TRACE_LOG("report: nil deprel (in f.s.) =%d", kNilDeprel);
  if (use_distance) {
    TRACE_LOG("report: nil distance (in f.s.) =%d", kNilDistance);
  }
  if (use_valency) {
    TRACE_LOG("report: nil valency (in f.s.) =%d", kNilValency);
  }
  if (use_cluster) {
    TRACE_LOG("report: nil cluster4 (in f.s.) =%d", kNilCluster4);
    TRACE_LOG("report: nil cluster6 (in f.s.) =%d", kNilCluster6);
    TRACE_LOG("report: nil cluster (in f.s.) =%d", kNilCluster);
  }
}

void NeuralNetworkParser::transduce_instance_to_dependency(const Instance& data,
    Dependency* dependency, bool with_dependencies) {
  size_t L = data.forms.size();
  for (size_t i = 0; i < L; ++ i) {
    int form = forms_alphabet.index(data.forms[i]);
    if (form == -1) { form = forms_alphabet.index(SpecialOption::UNKNOWN); }
    int postag = postags_alphabet.index(data.postags[i]);
    int deprels = deprels_alphabet.index(data.deprels[i]);

    dependency->forms.push_back(form);
    dependency->postags.push_back(postag);
    dependency->heads.push_back(with_dependencies? data.heads[i]: -1);
    dependency->deprels.push_back(with_dependencies? deprels: -1);
  }
}

void NeuralNetworkParser::get_cluster_from_dependency(const Dependency& data,
    std::vector<int>& cluster4,
    std::vector<int>& cluster6,
    std::vector<int>& cluster) {
  if (use_cluster) {
    size_t L = data.forms.size();
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

} //  namespace depparser
} //  namespace ltp
