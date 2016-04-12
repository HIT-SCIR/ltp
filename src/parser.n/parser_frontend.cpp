#include <fstream>
#include "parser.n/parser_frontend.h"
#include "parser.n/io.h"
#include "utils/logging.hpp"
#include "utils/strutils.hpp"
#include "utils/codecs.hpp"
#include "utils/time.hpp"

namespace ltp {
namespace depparser {

using framework::LineCountsReader;
using strutils::trim;
using strutils::split;
using strutils::to_double;
using strutils::codecs::is_unicode_punctuation;
using utility::timer;

NeuralNetworkParserFrontend::NeuralNetworkParserFrontend(
    const LearnOption& opt): learn_opt(&opt), test_opt(NULL) {
  use_distance = opt.use_distance;
  use_valency = opt.use_valency;
  use_cluster = opt.use_cluster;
  root = opt.root;
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
  INFO_LOG("report: %d tree(s) are illegal.", nr_non_trees);
  INFO_LOG("report: %d tree(s) is legal but not projective.", nr_non_projective_trees);
}

bool NeuralNetworkParserFrontend::read_training_data(void) {
  train_dat.clear();
  devel_dat.clear();

  std::ifstream ifs(learn_opt->reference_file.c_str());
  if (!ifs.good()) {
    ERROR_LOG("#: failed to open reference file, training halted.");
    return false;
  }

  INFO_LOG("#: start loading dataset from reference file ...");
  CoNLLReader reader(ifs, true);
  Instance* inst = NULL;
  while ((inst = reader.next())) { train_dat.push_back(inst); }

  INFO_LOG("report: %d training instance(s) is loaded.", train_dat.size());
  check_dataset(train_dat);

  std::ifstream ifs2(learn_opt->devel_file.c_str());
  if (!ifs2.good()) {
    WARNING_LOG("#: development file is not loaded.");
  } else {
    CoNLLReader reader2(ifs2, false);
    while ((inst = reader2.next())) { devel_dat.push_back(inst); }
    INFO_LOG("report: %d developing instance(s) is loaded.", devel_dat.size());
    check_dataset(devel_dat);
  }
  return true;
}

void NeuralNetworkParserFrontend::build_alphabet(void) {
  typedef std::unordered_map<std::string, int> map_t;
  map_t frequencies;
  for (size_t i = 0; i < train_dat.size(); ++ i) {
    const Instance* inst = train_dat[i];
    for (size_t j = 1; j < inst->size(); ++ j) {
      // Starting from first effective word. Leave ROOT to the final stage in
      // this function.
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
  deprels_alphabet.push(SpecialOption::ROOT);

  kNilDistance = (learn_opt->use_distance? 8: 0);
  kNilValency = (learn_opt->use_valency? 8: 0);
}

void NeuralNetworkParserFrontend::build_cluster(void) {
  if (forms_alphabet.size() == 0) {
    ERROR_LOG("#: should not load cluster before constructing forms alphabets.");
    return;
  }

  std::ifstream ifs(learn_opt->cluster_file.c_str());
  if (!ifs.good()) {
    ERROR_LOG("#: cluster file open failed, cluster is not loaded.");
    use_cluster = false;
    return;
  }

  std::string line;
  //TODO 构造函数 LineCountsReader 会计算 interval，此处的计算重复了.
  size_t interval = LineCountsReader(ifs).number_of_lines() / 10;
  if (interval == 0) { interval = 1; }
  size_t nr_lines = 1;

  while (std::getline(ifs, line)) {
    if (nr_lines++ % interval == 0) {
      INFO_LOG("#: loaded %d0%% cluster.", nr_lines / interval);
    }
    trim(line);
    if (line.size() == 0) { continue; }
    std::vector<std::string> items = split(line);

    if (items.size() < 2) {
      WARNING_LOG("cluster file in ill format.");
      continue;
    }

    int form = forms_alphabet.index(items[1]);
    // TODO to lower fails with utf8 input
    // int transformed_form = forms_alphabet.encode(items[0].tolower());
    if (form == -1) {
      // _TRACE << "report: form\'" << items[0] << "\' not occur in training data, ignore.";
      continue;
    }

    form_to_cluster4[form] = cluster4_types_alphabet.push(items[0].substr(0, 4));
    form_to_cluster6[form] = cluster6_types_alphabet.push(items[0].substr(0, 6));
    form_to_cluster[form] = cluster_types_alphabet.push(items[0]);
  }
  // push nil cluster.
  form_to_cluster4[-1] = cluster4_types_alphabet.push(SpecialOption::UNKNOWN);
  kNilCluster4 = cluster4_types_alphabet.push(SpecialOption::NIL);
  cluster4_types_alphabet.push(SpecialOption::ROOT);

  form_to_cluster6[-1] = cluster6_types_alphabet.push(SpecialOption::UNKNOWN);
  kNilCluster6 = cluster6_types_alphabet.push(SpecialOption::NIL);
  cluster6_types_alphabet.push(SpecialOption::ROOT);

  form_to_cluster[-1] = cluster_types_alphabet.push(SpecialOption::UNKNOWN);
  kNilCluster = cluster_types_alphabet.push(SpecialOption::NIL);
  cluster_types_alphabet.push(SpecialOption::ROOT);
}

void NeuralNetworkParserFrontend::collect_precomputed_features() {
  std::unordered_map<int, int> features_frequencies;

  size_t nr_processed = 0;
  nr_feature_types = 0;
  size_t interval = train_dat.size() / 10; if (interval == 0) { interval = 1; }
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
        INFO_LOG("#: number of feature types unequal to configed number");
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
      INFO_LOG("#: generated training samples for %d0%% sentences.", nr_processed / interval);
    }
  }
  INFO_LOG("#: generated %d training samples.", dataset.size());

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
  INFO_LOG("#: start to load embedding ...");
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
        INFO_LOG("#: loaded %d0%% embeddings.", nr_lines / interval);
      }

      trim(line);
      if (line.size() == 0) { continue; }
      std::vector<std::string> items = split(line);

      int form = forms_alphabet.index(items[0]);
      if (form == -1) { continue; }

      if (items.size() != learn_opt->embedding_size + 1) {
        WARNING_LOG("report: line %d, embedding dimension(%d) not match to configuration.", nr_lines, items.size()-1);
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
  INFO_LOG("report: %d embedding is loaded.", embeddings.size());

  classifier.initialize(kFeatureSpaceEnd,
      system.number_of_transitions(),
      nr_feature_types,
      (*learn_opt),
      embeddings,
      precomputed_features
      );
  INFO_LOG("report: classifier is initialized.");
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
          INFO_LOG("#: number of feature types unequal to configed number");
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

        if (oracle == -1) {
          ERROR_LOG("# unexpected error occurs!");
          break;
        }
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

void NeuralNetworkParserFrontend::evaluate_one_instance(const Instance& data,
    const std::vector<int>& heads, const std::vector<std::string>& deprels,
    size_t& corr_heads, size_t& corr_deprels, size_t& nr_tokens) {
  size_t L = heads.size();
  for (size_t i = 1; i < L; ++ i) { // ignore dummy root
    if (is_unicode_punctuation(data.raw_forms[i])) {
      continue;
    }
    ++ nr_tokens;
    if (heads[i] == data.heads[i]) {
      ++ corr_heads;
      if (deprels[i] == data.deprels[i]) { ++ corr_deprels; }
    }
  }
}

void NeuralNetworkParserFrontend::evaluate(double& uas, double& las) {
  INFO_LOG("eval: start evaluating ...");
  classifier.precomputing();

  std::vector<int> heads;
  std::vector<std::string> deprels;
  size_t corr_heads = 0, corr_deprels = 0, nr_tokens = 0;

  for (size_t i = 0; i < devel_dat.size(); ++ i) {
    Instance* data = devel_dat[i];
    predict((*data), heads, deprels);
    evaluate_one_instance((*data), heads, deprels, corr_heads,
        corr_deprels, nr_tokens);
  }

  uas = (double)corr_heads/nr_tokens;
  las = (double)corr_deprels/nr_tokens;
}

void NeuralNetworkParserFrontend::train(void) {
  if (!read_training_data()) { return; }
  build_alphabet();
  setup_system();
  if (learn_opt->use_cluster) {
    build_cluster();
  }
  build_feature_space();
  report();
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
    INFO_LOG("pipe (iter#%d): cost=%lf, accuracy(%)=%lf (%lf)", (iter+1),
       classifier.get_cost(), classifier.get_accuracy(), t.elapsed()); 
    classifier.take_ada_gradient_step();

    if (devel_dat.size() > 0 && (iter+1) % learn_opt->evaluation_stops == 0) {
      t.restart();
      double uas, las;
      evaluate(uas, las);
      INFO_LOG("eval: evaluating done. UAS=%lf LAS=%lf (%lf)", uas, las, t.elapsed());

      if (best_uas < uas && learn_opt->save_intermediate) {
        best_uas = uas;
        save(learn_opt->model_file);
        INFO_LOG("report: model saved to %s", learn_opt->model_file.c_str());
      }
    }
  }

  if (devel_dat.size() > 0) {
    timer t;
    double uas, las;
    evaluate(uas, las);
    INFO_LOG("eval: evaluating done. UAS=%lf LAS=%lf (%lf)", uas, las, t.elapsed());

    if (best_uas < uas && learn_opt->save_intermediate) {
      best_uas = uas;
      save(learn_opt->model_file);
      INFO_LOG("report: model saved to %s", learn_opt->model_file.c_str());
    }
  } else {
    save(learn_opt->model_file);
    INFO_LOG("report: model saved to %s", learn_opt->model_file.c_str());
  }
}

void NeuralNetworkParserFrontend::test(void) {
  INFO_LOG("report: loading model from %s.", test_opt->model_file.c_str());
  if (!load(test_opt->model_file)) {
    WARNING_LOG("failed to load model file.");
    return;
  }
  setup_system();
  build_feature_space();
  report();
  classifier.info();

  std::ifstream ifs(test_opt->input_file.c_str());
  if (!ifs.good()) {
    ERROR_LOG("failed to open input file, testing halted.");
    return;
  }

  CoNLLReader reader(ifs, false);
  CoNLLWriter writer(std::cout);
  Instance* inst= NULL;
  std::vector<int> heads;
  std::vector<std::string> deprels;
  size_t corr_heads = 0, corr_deprels = 0, nr_tokens = 0;

  timer t;
  while (true) {
    inst = reader.next();
    if (inst == NULL) { break; }
    predict((*inst), heads, deprels);
    writer.write(*inst, heads, deprels);

    if (test_opt->evaluate) {
      evaluate_one_instance((*inst), heads, deprels, corr_heads,
          corr_deprels, nr_tokens);
    }
    delete inst;
  }

  if (test_opt->evaluate) {
    double uas = (double)corr_heads/nr_tokens;
    double las = (double)corr_deprels/nr_tokens;
    INFO_LOG("eval: evaluating done. UAS=%lf LAS=%lf", uas, las);
  }
  INFO_LOG("elapsed time: %lf", t.elapsed());
}

}   //  end for namespace depparser
}   //  end for namespace ltp
