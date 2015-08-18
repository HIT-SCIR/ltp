#include <fstream>
#include "segmentor/segmentor_frontend.h"
#include "segmentor/settings.h"
#include "segmentor/extractor.h"
#include "segmentor/io.h"
#include "segmentor/partial_segmentation.h"
#include "utils/smartmap.hpp"
#include "utils/strutils.hpp"
#include "utils/sbcdbc.hpp"
#include "utils/time.hpp"

namespace ltp {
namespace segmentor {

using framework::Frontend;
using framework::Parameters;
using framework::ViterbiDecoder;
using framework::ViterbiFeatureContext;
using framework::ViterbiScoreMatrix;
using framework::kLearn;
using framework::kTest;
using framework::kDump;
using math::Mat;
using math::SparseVec;
using math::FeatureVector;
using strutils::trim;
using utility::SmartMap;
using utility::timer;

SegmentorFrontend::SegmentorFrontend(const std::string& reference_file,
    const std::string& holdout_file,
    const std::string& model_name,
    const std::string& algorithm,
    const size_t& max_iter,
    const size_t& rare_feature_threshold,
    bool dump_model_details)
  : timestamp(0), Frontend(kLearn) {
  train_opt.train_file = reference_file;
  train_opt.holdout_file = holdout_file;
  train_opt.algorithm = algorithm;
  train_opt.model_name = model_name;
  train_opt.max_iter = max_iter;
  train_opt.rare_feature_threshold = rare_feature_threshold;
  train_opt.dump_model_details = dump_model_details;

  INFO_LOG("||| ltp segmentor, training ...");
  INFO_LOG("report: reference file = %s", train_opt.train_file.c_str());
  INFO_LOG("report: holdout file = %s", train_opt.holdout_file.c_str());
  INFO_LOG("report: algorithm = %s", train_opt.algorithm.c_str());
  INFO_LOG("report: model name = %s", train_opt.model_name.c_str());
  INFO_LOG("report: maximum iteration = %d", train_opt.max_iter);
  INFO_LOG("report: rare threshold = %d", train_opt.rare_feature_threshold);
  INFO_LOG("report: dump model details = %s",
      (train_opt.dump_model_details? "true": "false"));
}

SegmentorFrontend::SegmentorFrontend(const std::string& input_file,
    const std::string& model_file,
    bool evaluate,
    bool sequence_prob,
    bool marginal_prob)
  : timestamp(0), Frontend(kTest) {
  test_opt.test_file = input_file;
  test_opt.model_file = model_file;
  test_opt.evaluate = evaluate;
  test_opt.sequence_prob = sequence_prob;
  test_opt.marginal_prob = marginal_prob;

  INFO_LOG("||| ltp segmentor, testing ...");
  INFO_LOG("report: input file = %s", test_opt.test_file.c_str());
  INFO_LOG("report: model file = %s", test_opt.model_file.c_str());
  INFO_LOG("report: evaluate = %s", (test_opt.evaluate? "true": "false"));
  INFO_LOG("report: sequence probability = %s", (test_opt.sequence_prob? "true": "false"));
  INFO_LOG("report: marginal probability = %s", (test_opt.marginal_prob? "true":"false"));
}

SegmentorFrontend::SegmentorFrontend(const std::string& model_file)
  : timestamp(0), Frontend(kDump) {
  dump_opt.model_file = model_file;

  INFO_LOG("||| ltp segmentor, dumpping ...");
  INFO_LOG("report: model file = %s", model_file.c_str());
}

SegmentorFrontend::~SegmentorFrontend() {
  for (size_t i = 0; i < train_data_input.size(); ++ i) {
    if (train_data_input[i]) {
      delete train_data_input[i];
      train_data_input[i] = 0;
    }
  }
}

size_t SegmentorFrontend::get_timestamp() const {
  return timestamp;
}

void SegmentorFrontend::increase_timestamp() {
  ++ timestamp;
}

void SegmentorFrontend::build_configuration(void) {
  for (size_t i = 0; i < __num_pos_types__; ++ i) { model->labels.push(__pos_types__[i]); }
  INFO_LOG("Label sets is built");

  SmartMap<bool> wordfreq;
  unsigned long long total_freq = 0;
  for (size_t i = 0; i < train_data_output.size(); ++ i) {
    const std::vector<std::string>& words = train_data_output[i].words;
    if (train_data_output[i].is_partial) {
      for (size_t j = 0; j < words.size(); ++ j) {
        if (PartialSegmentationUtils::is_partial_tagged_word(words[j])) {
          std::string actual_word;
          PartialSegmentationUtils::trim_partial_tag(words[j], actual_word);
          wordfreq.set(actual_word.c_str(), true);
        }
      }
    } else {
      for (size_t j = 0; j < words.size(); ++ j) {
        wordfreq.set(words[j].c_str(), true);
      }
    }
    total_freq += words.size();
  }

  // count words frequency
  std::vector<size_t> freqs;
  for (SmartMap<bool>::const_iterator itx = wordfreq.begin();
      itx != wordfreq.end(); ++ itx) {
    freqs.push_back(itx.frequency());
  }

  // filter words based on frequency
  unsigned long long accumulate_freq = 0;
  std::sort(freqs.begin(), freqs.end(), std::greater<int>());
  size_t target_freq = freqs[int(freqs.size() * 0.2)];
  for (size_t i = 0; i < freqs.size(); ++ i) {
    accumulate_freq += freqs[i];
    if (accumulate_freq > total_freq * 0.9) {
      target_freq = freqs[i];
      break;
    }
  }

  // build words dictionary
  for (SmartMap<bool>::const_iterator itx = wordfreq.begin();
      itx != wordfreq.end(); ++ itx) {
    if (itx.frequency() >= target_freq && strutils::codecs::length(itx.key()) > 1) {
      model->internal_lexicon.set(itx.key(), true);
    }
  }
  setup_lexicons();

  INFO_LOG("report: collecting interanl lexicon is done.");
  INFO_LOG("report: total word frequency : %ld", total_freq);
  INFO_LOG("report: vocabulary size: %d", wordfreq.size());
  INFO_LOG("report: trancation word frequency : %d", target_freq);
  INFO_LOG("report: internal lexicon size : %d", model->internal_lexicon.size());
}

void SegmentorFrontend::extract_features(const Instance& inst, bool create) {
  Segmentor::extract_features(inst, model, &ctx, create);
}

void SegmentorFrontend::extract_features(const Instance& inst) {
  Segmentor::extract_features(inst, model, NULL, true);
}

void SegmentorFrontend::calculate_scores(const Instance& inst, bool avg) {
  Segmentor::calculate_scores(inst, (*model), ctx, avg, &scm);
}

void SegmentorFrontend::collect_features(const Instance& inst) {
  Frontend::collect_features(model, ctx.uni_features, inst.tagsidx, ctx.correct_features);
  Frontend::collect_features(model, ctx.uni_features, inst.predict_tagsidx, ctx.predict_features);
}

void SegmentorFrontend::clear_context() {
  ctx.clear();
}

void SegmentorFrontend::build_feature_space(void) {
  // build feature space, it is a wrapper for featurespace.build_feature_space
  Extractor::num_templates();
  size_t L = model->num_labels();
  model->space.set_num_labels(L);

  size_t interval = train_data_input.size() / 10;
  if (0 == interval) { interval = 1; }

  for (size_t i = 0; i < train_data_input.size(); ++ i) {
    build_lexicon_match_state(lexicons, train_data_input[i]);
    extract_features((*train_data_input[i]), true);

    if ((i+1) % interval == 0) {
      INFO_LOG("build-featurespace: %d0%% instances is extracted.", (i+1) / interval);
    }
  }
  INFO_LOG("trace: feature space is built for %d instances.", train_data_input.size());
}

bool SegmentorFrontend::read_instance(const char* train_file) {
  std::ifstream ifs(train_file);

  if (!ifs) { return false; }

  std::string line;
  size_t nr_lines = 0, nr_fully_segmented = 0, nr_partially_segmented = 0;

  train_data_input.clear();
  train_data_output.clear();
  while (std::getline(ifs, line)) {
    ++ nr_lines;

    Segmentation output;
    trim(line);

    if (line.length() == 0) {
      WARNING_LOG("read inst: line #%d, is empty", nr_lines);
      continue;
    }

    int ret = PartialSegmentationUtils::split_by_partial_tag(line, output.words);
    if (ret == -1) {
      WARNING_LOG("read inst: line #%d, partial segmented data is in ill-format.");
      continue;
    }
    output.is_partial = (ret == 1);
    if (output.is_partial) { nr_partially_segmented ++; } else { nr_fully_segmented ++; }

    Instance* input = new Instance;
    if (!output.is_partial) {
      for (size_t i = 0; i < output.words.size(); ++ i) {
        int num_chars = preprocessor.preprocess(output.words[i], input->raw_forms,
            input->forms, input->chartypes);

        for (size_t j = 0; j < num_chars; ++ j) {
          if (1 == num_chars) { output.con.append( 1 << __s_id__ ); }
          else {
            if (0 == j) { output.con.append( 1 << __b_id__ ); }
            else if (num_chars - 1 == j) { output.con.append( 1 << __e_id__ ); }
            else { output.con.append( 1 << __i_id__ ); }
          }
        }
      }
    } else {
      for (size_t i = 0; i < output.words.size(); ++ i) {
        std::string word;
        bool is_word = PartialSegmentationUtils::is_partial_tagged_word(output.words[i]);
        PartialSegmentationUtils::trim_partial_tag(output.words[i], word);

        int num_chars = preprocessor.preprocess(word, input->raw_forms,
            input->forms, input->chartypes);

        if (is_word) {
          for (size_t j = 0; j < num_chars; ++ j) {
            if (1 == num_chars) { output.con.append( 1 << __s_id__ ); }
            else {
              if (0 == j) { output.con.append( 1 << __b_id__ ); }
              else if (num_chars - 1 == j) { output.con.append( 1 << __e_id__ ); }
              else { output.con.append( 1 << __i_id__ ); }
            }
          }
        } else {
          for (size_t j = 0; j < num_chars; ++ j) {
            if (1 == num_chars) { output.con.append( 1 << __s_id__ ); }
            else {
              if (0 == j) {
                output.con.append( (1<<__b_id__)|(1<<__s_id__) );
              } else if (num_chars - 1 == j) {
                output.con.append( (1<<__e_id__)|(1<<__s_id__) );
              } else {
                output.con.append( (1<<__b_id__)|(1<<__i_id__)|(1<<__e_id__)|(1<<__s_id__) );
              }
            }
          }
        }
      }
    }
    train_data_input.push_back(input);
    train_data_output.push_back(output);
  }

  INFO_LOG("report: loaded %d fully segmented data.", nr_fully_segmented);
  INFO_LOG("report: loaded %d partially segmented data.", nr_partially_segmented);
  return true;
}

void SegmentorFrontend::update(const Instance& inst, SparseVec& updated_features) {
  updated_features.add(ctx.correct_features, 1.);
  updated_features.add(ctx.predict_features, -1.);

  learn(train_opt.algorithm, updated_features, get_timestamp(),
      InstanceUtils::num_errors(inst.tagsidx, inst.predict_tagsidx), model);
}

void SegmentorFrontend::setup_lexicons() {
  lexicons.push_back(&(model->internal_lexicon));
  lexicons.push_back(&(model->external_lexicon));
}

void SegmentorFrontend::train(void) {
  // if (!train_setup()) { return; }
  // read in training instance
  INFO_LOG("trace: reading reference dataset ...");
  if (!read_instance(train_opt.train_file.c_str())) {
    ERROR_LOG("Training file not exist.");
    return;
  }
  INFO_LOG("trace: %d sentence is loaded.", train_data_input.size());

  model = new Model;

  // build tag dictionary, map string tag to index
  INFO_LOG("report: start build configuration ...");
  build_configuration();
  INFO_LOG("report: build configuration is done.");
  INFO_LOG("report: number of labels: [%d]", model->labels.size());

  // build feature space from the training instance
  INFO_LOG("report: start building feature space ...");
  build_feature_space();
  INFO_LOG("report: building feature space is done.");
  INFO_LOG("report: number of features: %d", model->space.num_features());

  model->param.realloc(model->space.dim());
  INFO_LOG("report: allocate %d dimensition parameter.", model->space.dim());

  int nr_groups = model->space.num_groups();
  std::vector<size_t> update_counts;

  // If the rare feature threshold is used, allocate memory for the
  // feature group updated time.
  if (train_opt.rare_feature_threshold > 0) {
    update_counts.resize(nr_groups, 0);
    INFO_LOG("report: allocate %d update-time counters", nr_groups);
  } else {
    INFO_LOG("report: model truncation is inactived.");
  }

  int best_iteration = -1;
  double best_p = -1., best_r = -1., best_f = -1.;

  for (size_t iter = 0; iter < train_opt.max_iter; ++ iter) {
    INFO_LOG("Training iteration #%d", (iter + 1));

    size_t interval = train_data_input.size()/ 10;
    if (interval == 0) { interval = 1; }
    for (size_t i = 0; i < train_data_input.size(); ++ i) {
      increase_timestamp();

      Instance* input = train_data_input[i];
      extract_features((*input), false);
      calculate_scores((*input), false);

      Segmentation& output = train_data_output[i];

      output.con.regist(&(input->chartypes));
      con.regist(&(input->chartypes));

      decoder.decode(scm, output.con, input->tagsidx);
      decoder.decode(scm, con, input->predict_tagsidx);

      collect_features((*input));

      SparseVec updated_features;
      update((*input), updated_features);
      clear_context();

      if (train_opt.rare_feature_threshold > 0) {
        increase_groupwise_update_counts(model, updated_features, update_counts);
      }
      if ((i+1) % interval == 0) {
        INFO_LOG("training: %d0%% (%d) instances is trained.", ((i+1)/interval), i+1);
      }
    }
    INFO_LOG("trace: %d instances is trained.", train_data_input.size());

    model->param.flush(get_timestamp());

    Model* new_model = new Model;
    erase_rare_features(model, new_model,
        train_opt.rare_feature_threshold, update_counts);

    for (Model::lexicon_t::const_iterator itx = model->internal_lexicon.begin();
        itx != model->internal_lexicon.end();
        ++ itx) {
      new_model->internal_lexicon.set(itx.key(), true);
    }

    std::swap(model, new_model);
    double p, r, f;
    evaluate(p,r,f);

    std::swap(model, new_model);

    if (f > best_f) {
      best_p = p; best_r = r; best_f = f;
      best_iteration = iter;

      std::ofstream ofs(train_opt.model_name.c_str(), std::ofstream::binary);
      new_model->save(model_header.c_str(),
          (train_opt.dump_model_details? Parameters::kDumpDetails: Parameters::kDumpAveraged),
          ofs);

      INFO_LOG("Model for iteration #%d is saved to [%s]", iter+1, train_opt.model_name.c_str());
    }
    delete new_model;
  }

  INFO_LOG("Best result (iteratin = %d) P = %lf | R = %lf | F = %lf",
      best_iteration, best_p, best_r, best_f);
}

void SegmentorFrontend::evaluate(double &p, double &r, double &f) {
  const char* holdout_file = train_opt.holdout_file.c_str();

  std::ifstream ifs(holdout_file);

  if (!ifs) {
    ERROR_LOG("Failed to open holdout file.");
    return;
  }

  SegmentReader reader(ifs, preprocessor, true, false);
  Instance* inst = NULL;

  size_t num_recalled_words = 0;
  size_t num_predicted_words = 0;
  size_t num_gold_words = 0;

  while ((inst = reader.next())) {
    size_t len = inst->size();
    inst->tagsidx.resize(len);
    for (size_t i = 0; i < len; ++ i) {
      inst->tagsidx[i] = model->labels.index(inst->tags[i]);
    }

    build_lexicon_match_state(lexicons, inst);
    extract_features((*inst), false);
    calculate_scores((*inst), true);

    con.regist(&(inst->chartypes));
    decoder.decode(scm, con, inst->predict_tagsidx);
    ctx.clear();

    std::vector<std::string> answer_words;
    std::vector<std::string> predict_words;

    build_words(inst->raw_forms, inst->tagsidx, answer_words);
    build_words(inst->raw_forms, inst->predict_tagsidx, predict_words);

    num_recalled_words += InstanceUtils::num_recalled_words(answer_words, predict_words);
    num_predicted_words += predict_words.size();
    num_gold_words += answer_words.size();

    delete inst;
  }

  p = (double)num_recalled_words / num_predicted_words;
  r = (double)num_recalled_words / num_gold_words;
  f = 2 * p * r / (p + r);

  INFO_LOG("P: %lf ( %d / %d )", p, num_recalled_words, num_predicted_words);
  INFO_LOG("R: %lf ( %d / %d )", r, num_recalled_words, num_gold_words);
  INFO_LOG("F: %lf" , f);
  return;
}

void SegmentorFrontend::test(void) {
  const char* model_file = test_opt.model_file.c_str();
  std::ifstream mfs(model_file, std::ifstream::binary);

  if (!mfs) {
    ERROR_LOG("Failed to load model");
    return;
  }

  model = new Model;
  if (!model->load(model_header.c_str(), mfs)) {
    ERROR_LOG("Failed to load model");
    return;
  }

  INFO_LOG("report: number of labels = %d", model->num_labels());
  INFO_LOG("report: number of features = %d", model->space.num_features());
  INFO_LOG("report: number of dimension = %d", model->space.dim());

  size_t num_recalled_words = 0;
  size_t num_predicted_words = 0;
  size_t num_gold_words = 0;

  // load exteranl lexicon
  const char* lexicon_file = test_opt.lexicon_file.c_str();
  load_lexicon(lexicon_file, &(model->external_lexicon));

  const char* test_file = test_opt.test_file.c_str();
  std::ifstream ifs(test_file);

  if (!ifs) {
    ERROR_LOG("Failed to open test file.");
    return;
  }

  SegmentWriter writer(std::cout, test_opt.sequence_prob, test_opt.marginal_prob);
  SegmentReader reader(ifs, preprocessor, test_opt.evaluate, false);

  setup_lexicons();

  decoder.set_sequence_prob(test_opt.sequence_prob);
  decoder.set_marginal_prob(test_opt.marginal_prob);

  Instance* inst = NULL;
  timer t;
  while ((inst = reader.next())) {
    int len = inst->size();
    if (test_opt.evaluate) {
      inst->tagsidx.resize(len);
      for (int i = 0; i < len; ++ i) {
        inst->tagsidx[i] = model->labels.index(inst->tags[i]);
      }
    }
    build_lexicon_match_state(lexicons, inst);
    extract_features((*inst), false);
    calculate_scores((*inst), true);

    con.regist(&(inst->chartypes));
    if (test_opt.sequence_prob || test_opt.marginal_prob) {
      decoder.decode(scm,
              con,
              inst->predict_tagsidx,
              inst->sequence_probability,
              inst->point_probabilities,
              inst->partial_probabilities,
              inst->partial_idx,
              true,
              model->param._last_timestamp);
    } else {
      decoder.decode(scm, con, inst->predict_tagsidx);
    }
    ctx.clear();

    build_words(inst->raw_forms, inst->predict_tagsidx, inst->predict_words);

    if (test_opt.evaluate) {
      std::vector<std::string> answer_words;
      build_words(inst->raw_forms, inst->tagsidx, answer_words);

      num_recalled_words += InstanceUtils::num_recalled_words(answer_words, inst->predict_words);
      num_predicted_words += inst->predict_words.size();
      num_gold_words += answer_words.size();
    }
    writer.write(inst);
    delete inst;
  }

  if (test_opt.evaluate) {
    double p = (double)num_recalled_words / num_predicted_words;
    double r = (double)num_recalled_words / num_gold_words;
    double f = 2 * p * r / (p + r);

    INFO_LOG("P: %lf ( %d / %d )", p, num_recalled_words, num_predicted_words);
    INFO_LOG("R: %lf ( %d / %d )", r, num_recalled_words, num_gold_words);
    INFO_LOG("F: %lf" , f);
  }
  INFO_LOG("Elapsed time %lf", t.elapsed());
  return;
}


void SegmentorFrontend::dump(void) {
  // load model
  const char* model_file = dump_opt.model_file.c_str();
  std::ifstream mfs(model_file, std::ifstream::binary);

  if (!mfs) {
    ERROR_LOG("Failed to load model");
    return;
  }

  model = new Model;
  if (!model->load(model_header.c_str(), mfs)) {
    ERROR_LOG("Failed to load model");
    return;
  }

  int L = model->num_labels();
  INFO_LOG("Number of labels %d", model->num_labels());
  INFO_LOG("Number of features %d", model->space.num_features());
  INFO_LOG("Number of dimension %d", model->space.dim());

  for (framework::FeatureSpaceIterator itx = model->space.begin();
       itx != model->space.end(); ++ itx) {
    const char* key = itx.key();
    int tid = itx.tid();
    int id = model->space.index(tid, key);

    for (int l = 0; l < L; ++ l) {
      std::cout << key << " ( " << id + l << " ) "
                << " --> " << model->param.dot(id + l)
                << std::endl;
    }
  }

  for (int pl = 0; pl < L; ++ pl) {
    for (int l = 0; l < L; ++ l) {
      int id = model->space.index(pl, l);
      std::cout << pl << " --> " << l << " " << model->param.dot(id) << std::endl;
    }
  }
}

} //  namespace segmentor
} //  namespace ltp
