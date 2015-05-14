#include <fstream>
#include "parser.n/parser_frontend.h"
#include "parser.n/io.h"
#include "utils/logging.hpp"
#include "utils/strutils.hpp"
#include "utils/time.hpp"

namespace ltp {
namespace depparser {

using framework::LineCountsReader;
using strutils::chomp;
using strutils::split;
using strutils::to_double;
using strutils::is_unicode_punctuation;
using utility::timer;

NeuralNetworkParserFrontend::NeuralNetworkParserFrontend(
    const LearnOption& opt): learn_opt(&opt), test_opt(NULL) {
  use_distance = opt.use_distance;
  use_valency = opt.use_valency;
  use_cluster = opt.use_cluster;
}

NeuralNetworkParserFrontend::NeuralNetworkParserFrontend(
    const TestOption& opt): learn_opt(NULL), test_opt(&opt) {}

NeuralNetworkParserFrontend::~NeuralNetworkParserFrontend() {
  for (size_t i = 0; i < train_dat.size(); ++ i) {
    if (train_dat[i]) { delete train_dat[i]; train_dat[i] = 0; }
  }
  for (size_t i = 0; i < devel_dat.size(); ++ i) {
    if (devel_dat[i]) { delete devel_dat[i]; devel_dat[i] = 0; }
  }
}

void NeuralNetworkParserFrontend::check_dataset(
    const std::vector<Instance*>& dataset) {
  size_t nr_non_trees = 0;
  size_t nr_non_projective_trees = 0;
  for (size_t i = 0; i < dataset.size(); ++ i) {
    if (!dataset[i]->is_tree()) {
      ++ nr_non_trees;
    } else if (!dataset[i]->is_projective()) {
      ++ nr_non_projective_trees;
    }
  }
  TRACE_LOG("report: %d tree(s) are illegal.", nr_non_trees);
  TRACE_LOG("report: %d tree(s) is legal but not projective.", nr_non_projective_trees);
}

bool NeuralNetworkParserFrontend::read_training_data(void) {
  train_dat.clear();
  devel_dat.clear();

  std::ifstream ifs(learn_opt->reference_file.c_str());
  if (!ifs.good()) {
    ERROR_LOG("#: failed to open reference file, training halted.");
    return false;
  }

  TRACE_LOG("#: start loading dataset from reference file ...");
  CoNLLReader reader(ifs, true);
  Instance* inst = NULL;
  while ((inst = reader.next())) { train_dat.push_back(inst); }

  TRACE_LOG("report: %d training instance(s) is loaded.", train_dat.size());
  check_dataset(train_dat);

  std::ifstream ifs2(learn_opt->devel_file.c_str());
  if (!ifs2.good()) {
    WARNING_LOG("#: development file is not loaded.");
  } else {
    CoNLLReader reader2(ifs2, false);
    while ((inst = reader2.next())) { devel_dat.push_back(inst); }
    TRACE_LOG("report: %d developing instance(s) is loaded.", devel_dat.size());
    check_dataset(devel_dat);
  }
  return true;
}

void NeuralNetworkParserFrontend::build_alphabet(void) {
  typedef std::unordered_map<std::string, int> map_t;
  map_t frequencies;
  for (size_t i = 0; i < train_dat.size(); ++ i) {
    const Instance* inst = train_dat[i];
    for (size_t j = 0; j < inst->size(); ++ j) {
      postags_alphabet.push(inst->postags[j]);
      deprels_alphabet.push(inst->deprels[j]);
      frequencies[inst->forms[j]] += 1;
    }
  }

  for (map_t::const_iterator entry= frequencies.begin();
      entry != frequencies.end(); ++ entry) {
    if (entry->second >= learn_opt->word_cutoff) { forms_alphabet.push(entry->first); }
  }

  forms_alphabet.push(SpecialOption::UNKNOWN);
  kNilForm = forms_alphabet.push(SpecialOption::NIL);
  forms_alphabet.push(SpecialOption::ROOT);

  postags_alphabet.push(SpecialOption::UNKNOWN);
  kNilPostag = postags_alphabet.push(SpecialOption::NIL);
  postags_alphabet.push(SpecialOption::ROOT);

  kNilDeprel = deprels_alphabet.push(SpecialOption::NIL);

  kNilDistance = (learn_opt->use_distance? 8: 0);
  kNilValency = (learn_opt->use_valency? 8: 0);

  system.set_root_relation(deprels_alphabet.index(learn_opt->root));
  system.set_number_of_relations(deprels_alphabet.size()- 1);
}

void NeuralNetworkParserFrontend::build_cluster(void) {
  if (forms_alphabet.size() == 0) {
    ERROR_LOG("#: should not load cluster before constructing forms alphabets.");
    return;
  }

  std::ifstream ifs(learn_opt->cluster_file.c_str());
  if (!ifs.good()) {
    ERROR_LOG("#: cluster file open failed, cluster is not loaded.");
    return;
  }

  std::string line;
  size_t interval = LineCountsReader(ifs).number_of_lines() / 10;
  size_t nr_lines = 1;

  while (std::getline(ifs, line)) {
    if (nr_lines++ % interval == 0) {
      TRACE_LOG("#: loaded %lf0%% cluster.", nr_lines / interval);
    }
    line = chomp(line);
    if (line.size() == 0) { continue; }
    std::vector<std::string> items = split(line);
    int form = forms_alphabet.index(items[0]);
    // TODO to lower fails with utf8 input
    // int transformed_form = forms_alphabet.encode(items[0].tolower());
    if (form == -1) {
      // _TRACE << "report: form\'" << items[0] << "\' not occur in training data, ignore.";
      continue;
    }
    if (items.size() < 2) {
      WARNING_LOG("cluster file in ill format.");
      continue;
    }

    form_to_cluster4[form] = cluster4_types_alphabet.push(items[1].substr(4));
    form_to_cluster6[form] = cluster6_types_alphabet.push(items[1].substr(6));
    form_to_cluster[form] = cluster_types_alphabet.push(items[1]);
  }
  // push nil cluster.
  cluster4_types_alphabet.push(SpecialOption::UNKNOWN);
  kNilCluster4 = cluster4_types_alphabet.push(SpecialOption::NIL);
  cluster4_types_alphabet.push(SpecialOption::ROOT);

  cluster6_types_alphabet.push(SpecialOption::UNKNOWN);
  kNilCluster6 = cluster6_types_alphabet.push(SpecialOption::NIL);
  cluster6_types_alphabet.push(SpecialOption::ROOT);

  cluster_types_alphabet.push(SpecialOption::UNKNOWN);
  kNilCluster = cluster_types_alphabet.push(SpecialOption::NIL);
  cluster6_types_alphabet.push(SpecialOption::ROOT);

  TRACE_LOG("#: loaded %d cluster(4)", cluster4_types_alphabet.size());
  TRACE_LOG("#: loaded %d cluster(6)", cluster6_types_alphabet.size());
  TRACE_LOG("#: loaded %d cluster", cluster_types_alphabet.size());
}

void NeuralNetworkParserFrontend::build_feature_space() {
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
    kFeatureSpaceEnd += cluster4_types_alphabet.size();
  } else { kNilCluster6 = kFeatureSpaceEnd; }

  kClusterInFeaturespace = kFeatureSpaceEnd;
  if (use_cluster) {
    kNilCluster = kFeatureSpaceEnd+ cluster_types_alphabet.index(SpecialOption::NIL);
    kFeatureSpaceEnd += cluster_types_alphabet.size();
  } else { kNilCluster = kFeatureSpaceEnd; }

  report();
}

void NeuralNetworkParserFrontend::collect_precomputed_features() {
  std::unordered_map<int, int> features_frequencies;

  size_t nr_processed = 0;
  nr_feature_types = 0;
  size_t interval = train_dat.size() / 10; if (interval == 0) { interval = 10; }
  for (size_t d = 0; d < train_dat.size(); ++ d) {
    Instance* inst = train_dat[d];
    if (!inst->is_tree() || !inst->is_projective()) { continue; }

    Dependency dependency;
    std::vector<int> cluster4, cluster6, cluster;
    transduce_instance_to_dependency((*inst), &dependency, true);
    get_cluster_from_dependency(dependency, cluster4, cluster6, cluster);

    std::vector<Action> oracle_actions;
    ActionUtils::get_oracle_actions2(dependency, oracle_actions);

    std::vector<State> states(oracle_actions.size()+ 1);
    states[0].copy(State(&dependency));
    system.transit(states[0], ActionFactory::make_shift(), &states[1]);
    for (size_t step = 1; step < oracle_actions.size(); ++ step) {
      const Action& oracle_action = oracle_actions[step];
      std::vector<int> attributes;
      if (use_cluster) {
        get_features(states[step], cluster4, cluster6, cluster, attributes);
      } else {
        get_features(states[step], attributes);
      }

      if (nr_feature_types == 0) {
        nr_feature_types = attributes.size();
      } else if (attributes.size() != nr_feature_types) {
        TRACE_LOG("#: number of feature types unequal to configed number");
      }

      if (learn_opt->oracle == "static") {
        // If not using the dynamic oracle mode, caching all the training instances
        // in the dataset
        std::vector<Action> possible_actions;
        system.get_possible_actions(states[step], possible_actions);
        std::vector<double> classes(system.number_of_transitions(), -1.);
        for (size_t j = 0; j < possible_actions.size(); ++ j) {
          size_t l = system.transform(possible_actions[j]);
          if (possible_actions[j] == oracle_action) { classes[l] = 1.; }
          else { classes[l] = 0.; }
        }
        dataset.push_back(Sample(attributes, classes));
      }

      for (size_t j = 0; j < attributes.size(); ++ j) {
        size_t fid = attributes[j] * attributes.size() + j;
        features_frequencies[fid] += 1;
      }
      system.transit(states[step], oracle_action, &states[step+ 1]);
    }

    if (++ nr_processed % interval == 0) {
      TRACE_LOG("#: generated training samples for %d0%% sentences.", nr_processed / interval);
    }
  }
  TRACE_LOG("#: generated %d training samples.", dataset.size());

  std::vector<std::pair<int, int> > top;
  if (features_frequencies.size() < learn_opt->nr_precomputed) {
    for (std::unordered_map<int, int>::const_iterator rep = features_frequencies.begin();
        rep != features_frequencies.end(); ++ rep) {
      top.push_back((*rep));
    }
  } else {
    top.resize(learn_opt->nr_precomputed);
    std::partial_sort_copy(features_frequencies.begin(),
        features_frequencies.end(), top.begin(), top.end(), PairGreaterBySecond());
  }

  for (size_t t = 0; t < top.size(); ++ t) { precomputed_features.push_back(top[t].first); }
}

void NeuralNetworkParserFrontend::initialize_classifier() {
  TRACE_LOG("#: start to load embedding ...");
  std::vector< std::vector<double> > embeddings;
  std::ifstream ifs(learn_opt->embedding_file.c_str());
  if (!ifs.good()) {
    WARNING_LOG("#: failed to open embedding file, embedding will be randomly initialized.");
  } else {
    std::string line;
    size_t interval = LineCountsReader(ifs).number_of_lines() / 10;
    size_t nr_lines = 1;

    while (std::getline(ifs, line)) {
      if (nr_lines++ % interval == 0) {
        TRACE_LOG("#: loaded %d0%% embeddings.", nr_lines / interval);
      }

      line = chomp(line);
      if (line.size() == 0) { continue; }
      std::vector<std::string> items = split(line);

      int form = forms_alphabet.index(items[0]);
      if (form == -1) { continue; }

      if (items.size() != learn_opt->embedding_size + 1) {
        WARNING_LOG("report: embedding dimension not match to configuration.");
        continue;
      }

      std::vector<double> embedding;
      embedding.push_back(form);
      for (size_t i = 1; i < items.size(); ++ i) {
        embedding.push_back( to_double(items[i]) );
      }
      embeddings.push_back( embedding );
    }
  }
  TRACE_LOG("report: %d embedding is loaded.", embeddings.size());

  classifier.initialize(kFeatureSpaceEnd,
      deprels_alphabet.size()*2-1,
      nr_feature_types,
      (*learn_opt),
      embeddings,
      precomputed_features
      );
  TRACE_LOG("report: classifier is initialized.");
}

void NeuralNetworkParserFrontend::generate_training_samples_one_batch(
    std::vector<Sample>::const_iterator& begin,
    std::vector<Sample>::const_iterator& end) {

  if (learn_opt->oracle == "static") {
    begin = dataset.begin() + cursor;
    end = dataset.end();
    if (cursor + learn_opt->batch_size < dataset.size()) {
      end = dataset.begin() + cursor + learn_opt->batch_size;
      cursor += learn_opt->batch_size;
    } else {
      cursor = 0;
    }
  } else {
    dataset.clear();

    while (dataset.size() < learn_opt->batch_size) {
      const Instance* data = train_dat[cursor ++];
      if (cursor == train_dat.size()) { cursor = 0; }
      if (!data->is_tree() || !data->is_projective()) { continue; }

      size_t L = data->size();
      Dependency dependency;
      std::vector<int> cluster, cluster4, cluster6;
      transduce_instance_to_dependency((*data), &dependency, true);
      get_cluster_from_dependency(dependency, cluster4, cluster6, cluster);

      double prev_cost = 0;
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
        if (attributes.size() != nr_feature_types) {
          TRACE_LOG("#: number of feature types unequal to configed number");
        }
        std::vector<double> scores;
        classifier.score(attributes, scores);

        std::vector<Action> possible_actions;
        system.get_possible_actions(states[step], possible_actions);

        std::vector<double> classes(system.number_of_transitions(), -1.);
        std::vector<int> costs(system.number_of_transitions(), 1024);

        int oracle = -1;
        int best_trans = -1;
        for (size_t j = 0; j < possible_actions.size(); ++ j) {
          const Action& act = possible_actions[j];
          int l = system.transform(act);
          classes[l] = 0.;

          system.transit(states[step], act, &states[step+1]);
          costs[l] = states[step+1].cost(dependency.heads, dependency.deprels);
          if (costs[l] - prev_cost == 0 &&
              (oracle == -1 || scores[oracle] < scores[l])) {
            oracle = l;
          }

          if (best_trans == -1 || scores[best_trans] < scores[l]) {
            best_trans = l;
          }
        }

        if (oracle == -1) { ERROR_LOG("# unexpected error occurs!"); }
        classes[oracle] = 1.;

        dataset.push_back(Sample(attributes, classes));
        if (learn_opt->oracle == "nondet") {
          system.transit(states[step], system.transform(oracle), &states[step+ 1]);
          prev_cost = costs[oracle];
        } else if (learn_opt->oracle == "explore") {
          system.transit(states[step], system.transform(best_trans), &states[step+ 1]);
          prev_cost = costs[best_trans];
        }
      }
    }
    begin = dataset.begin();
    end = dataset.end();
  }
}

void NeuralNetworkParserFrontend::predict(const Instance& data, std::vector<int>& heads,
    std::vector<std::string>& deprels) {
  Dependency dependency;
  std::vector<int> cluster, cluster4, cluster6;
  transduce_instance_to_dependency(data, &dependency, false);
  get_cluster_from_dependency(dependency, cluster, cluster4, cluster6);

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


void NeuralNetworkParserFrontend::train(void) {
  if (!read_training_data()) { return; }
  build_alphabet();
  if (learn_opt->use_cluster) {
    build_cluster();
  }
  build_feature_space();
  collect_precomputed_features();
  initialize_classifier();

  cursor = 0;
  double best_uas = -1;
  for (size_t iter = 0; iter < learn_opt->max_iter; ++ iter) {
    timer t;

    std::vector<Sample>::const_iterator begin;
    std::vector<Sample>::const_iterator end;
    generate_training_samples_one_batch(begin, end);

    classifier.compute_ada_gradient_step(begin, end);
    TRACE_LOG("pipe (iter#%d): cost=%lf, accuracy(%)=%lf (%lf)", (iter+1),
       classifier.get_cost(), classifier.get_accuracy(), t.elapsed()); 
    classifier.take_ada_gradient_step();

    if (devel_dat.size() > 0 && (iter+1) % learn_opt->evaluation_stops == 0) {
      TRACE_LOG("eval: start evaluating ...");
      classifier.precomputing();

      std::vector<int> heads;
      std::vector<std::string> deprels;
      size_t corr_heads = 0, corr_deprels = 0, nr_tokens = 0;

      t.restart();
      for (size_t i = 0; i < devel_dat.size(); ++ i) {
        Instance* data = devel_dat[i];
        predict((*data), heads, deprels);

        size_t L = heads.size();
        for (size_t i = 1; i < L; ++ i) { // ignore dummy root
          if (is_unicode_punctuation(data->forms[i])) {
            continue;
          }
          ++ nr_tokens;
          if (heads[i] == data->heads[i]) {
            ++ corr_heads;
            if (deprels[i] == data->deprels[i]) { ++ corr_deprels; }
          }
        }
      }
      double uas = (double)corr_heads/nr_tokens;
      double las = (double)corr_deprels/nr_tokens;
      TRACE_LOG("eval: evaluating done. UAS=%lf LAS=%lf (%lf)", uas, las, t.elapsed());

      if (best_uas < uas && learn_opt->save_intermediate) {
        best_uas = uas;
        // save_model(model_file);
        TRACE_LOG("report: model saved to %s", learn_opt->model_file.c_str());
      }
    }
  }
}

void NeuralNetworkParserFrontend::test(void) {
}


}   //  end for namespace depparser
}   //  end for namespace ltp
