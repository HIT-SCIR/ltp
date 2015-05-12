#include <fstream>
#include <sstream>
#include "ner/ner_frontend.h"
#include "ner/settings.h"
#include "ner/io.h"
#include "ner/extractor.h"
#include "utils/time.hpp"
#include "utils/logging.hpp"
#include "utils/strutils.hpp"

namespace ltp {
namespace ner {

using framework::Frontend;
using framework::Parameters;
using framework::Model;
using framework::kLearn;
using framework::kTest;
using framework::kDump;
using math::Mat;
using math::FeatureVector;
using math::SparseVec;
using utility::get_time;
using strutils::to_str;

const std::string NamedEntityRecognizerFrontend::model_header = "otner";

NamedEntityRecognizerFrontend::NamedEntityRecognizerFrontend(
    const std::string& reference_file,
    const std::string& holdout_file,
    const std::string& model_name,
    const std::string& algorithm,
    const int max_iter,
    const int rare_feature_threshold,
    const std::string& delim)
  : decoder(0), glob_con(0), ctx(0), scm(0), delimiter(delim), Frontend(kLearn) {
  train_opt.train_file = reference_file;
  train_opt.holdout_file = holdout_file;
  train_opt.algorithm = algorithm;
  train_opt.model_name = model_name;
  train_opt.max_iter = max_iter;
  train_opt.rare_feature_threshold = rare_feature_threshold;

  TRACE_LOG("||| ltp ner, trainig ...");
  TRACE_LOG("report: reference file = %s", train_opt.train_file.c_str());
  TRACE_LOG("report: holdout file = %s", train_opt.holdout_file.c_str());
  TRACE_LOG("report: algorith = %s", train_opt.algorithm.c_str());
  TRACE_LOG("report: model name = %s", train_opt.model_name.c_str());
  TRACE_LOG("report: maximum iteration = %d", train_opt.max_iter);
  TRACE_LOG("report: rare threshold = %d", train_opt.rare_feature_threshold);
}

NamedEntityRecognizerFrontend::NamedEntityRecognizerFrontend(
    const std::string& input_file,
    const std::string& model_file,
    bool evaluate,
    const std::string& delim)
  : decoder(0), glob_con(0), ctx(0), scm(0), delimiter(delim), Frontend(kTest) {
  test_opt.test_file = input_file;
  test_opt.model_file = model_file;
  test_opt.evaluate = evaluate;

  TRACE_LOG("||| ltp ner, testing ...");
  TRACE_LOG("report: input file = %s", test_opt.test_file.c_str());
  TRACE_LOG("report: model file = %s", test_opt.model_file.c_str());
  TRACE_LOG("report: evaluate = %s", (test_opt.evaluate? "true": "false"));
}

NamedEntityRecognizerFrontend::NamedEntityRecognizerFrontend(
    const std::string& model_file)
  : decoder(0), glob_con(0), ctx(0), scm(0), Frontend(kDump) {
  dump_opt.model_file = model_file;

  TRACE_LOG("||| ltp ner, dumpping ...");
  TRACE_LOG("report: model file = %s", model_file.c_str());
}

NamedEntityRecognizerFrontend::~NamedEntityRecognizerFrontend() {
  if (decoder)  { delete decoder;   decoder = 0; }
  if (glob_con) { delete glob_con;  glob_con = 0; }
  if (ctx)      { delete ctx;       ctx = 0; }
  if (scm)      { delete scm;       scm = 0; }

  for (size_t i = 0; i < train_dat.size(); ++ i) {
    if (train_dat[i]) { delete train_dat[i];  train_dat[i] = 0; }
  }
}

bool NamedEntityRecognizerFrontend::read_instance(const std::string& train_file) {
  std::ifstream ifs(train_file.c_str());
  if (!ifs) { return false; }

  NERReader reader(ifs, true, true);
  train_dat.clear();

  Instance* inst = NULL;
  while ((inst = reader.next())) { train_dat.push_back(inst); }

  return true;
}

void NamedEntityRecognizerFrontend::build_configuration(void) {
  // tag set is some kind of hard coded into the source
  set_t ne_types;
  for (size_t i = 0; i < train_dat.size(); ++ i) {
    Instance* inst = train_dat[i];
    for (size_t j = 0; j < inst->size(); ++ j) {
      const std::string& tag = inst->tags[j];
      size_t found = tag.find_last_of(delimiter);
      if (found == std::string::npos) {
        if (tag != OTHER) {
          WARNING_LOG("build-config: "
              "a error is detected at instance (%d,%d) %s.", i, j, tag.c_str());
          WARNING_LOG("build-config: look the data format by example option.");
          continue;
        }
      } else {
        std::string position = tag.substr(0, found);
        std::string ne_type = tag.substr(found+ 1);
        if (position != "S" && position != "B" && position != "I" && position != "E") {
          WARNING_LOG("build-config: "
              "a error is detected at instance (%d,%d) %s.", i, j, tag.c_str());
          WARNING_LOG("build-config: look the data format by example option.");
          continue;
        }

        ne_types.insert(ne_type);
      }
    }
  }

  TRACE_LOG("build-config: detect %d entity types.", ne_types.size());

  std::stringstream S;
  model->labels.push(OTHER);
  for (set_t::const_iterator j = ne_types.begin(); j != ne_types.end(); ++ j) {
    for (size_t i = 0; i < __num_pos_types__; ++ i) {
      S.str(std::string()); S << __pos_types__[i] << delimiter << (*j);
      model->labels.push(S.str());
    }
  }

  for (size_t i = 0; i < train_dat.size(); ++ i) {
    Instance* inst = train_dat[i];
    size_t len = inst->size();

    inst->tagsidx.resize(len);
    for (size_t j = 0; j < len; ++ j) {
      // build labels dictionary
      inst->tagsidx[j] = model->labels.index( inst->tags[j] );
      if (inst->tagsidx[j] < 0) {
        WARNING_LOG("Found illegal tag %s", inst->tags[j].c_str());
      }
    }
  }

  build_glob_tran_cons(ne_types);
}

void NamedEntityRecognizerFrontend::build_glob_tran_cons(const set_t& ne_types) {
  if (glob_con != NULL) {
    WARNING_LOG("Transition constrain should not be double allocated.");
  }

  const std::string& _ = delimiter;
  std::vector<std::string> includes;
  includes.push_back("O -> O");

  std::stringstream S;
  for (set_t::const_iterator i = ne_types.begin(); i != ne_types.end(); ++ i) {
    S.str(""); S << "O -> S" << _ << (*i); includes.push_back(S.str());
    S.str(""); S << "O -> B" << _ << (*i); includes.push_back(S.str());
    S.str(""); S << "S" << _ << (*i) << " -> O"; includes.push_back(S.str());
    S.str(""); S << "E" << _ << (*i) << " -> O"; includes.push_back(S.str());
    S.str(""); S << "B" << _ << (*i) << " -> I" << _ << (*i); includes.push_back(S.str());
    S.str(""); S << "B" << _ << (*i) << " -> E" << _ << (*i); includes.push_back(S.str());
    S.str(""); S << "I" << _ << (*i) << " -> I" << _ << (*i); includes.push_back(S.str());
    S.str(""); S << "I" << _ << (*i) << " -> E" << _ << (*i); includes.push_back(S.str());

    for (set_t::const_iterator j = ne_types.begin(); j != ne_types.end(); ++ j) {
      S.str(""); S << "S" << _ << (*i) << " -> S" << _ << (*j); includes.push_back(S.str());
      S.str(""); S << "S" << _ << (*i) << " -> B" << _ << (*j); includes.push_back(S.str());
      S.str(""); S << "E" << _ << (*i) << " -> S" << _ << (*j); includes.push_back(S.str());
      S.str(""); S << "E" << _ << (*i) << " -> B" << _ << (*j); includes.push_back(S.str());
    }
  }

  TRACE_LOG("build-config: add %d constrains.", includes.size());
  glob_con = new NERTransitionConstrain(model->labels, includes);
}

void NamedEntityRecognizerFrontend::build_feature_space(void) {
  // build feature space, it a wrapper for featurespace.build_feature_space
  Extractor::num_templates();
  int L = model->num_labels();
  model->space.set_num_labels(L);

  size_t interval = train_dat.size() / 10;
  for (size_t i = 0; i < train_dat.size(); ++ i) {
    NamedEntityRecognizer::extract_features((*train_dat[i]), NULL, true);
    if ((i+ 1) % interval == 0) {
      TRACE_LOG("build-featurespace: %d0%% instances is extracted.", (i+1) / interval);
    }
  }
  TRACE_LOG("trace: feature space is built for %d instances.", train_dat.size());
}

void NamedEntityRecognizerFrontend::train(void) {
  // read in training instance
  TRACE_LOG("trace: reading reference dataset ...");
  if (!read_instance(train_opt.train_file)) {
    ERROR_LOG("Training file doesn't exist");
  }
  TRACE_LOG("trace: %d sentences is loaded.", train_dat.size());

  model = new Model(Extractor::num_templates());
  // build tag dictionary, map string tag to index
  TRACE_LOG("report: start building configuration ...");
  build_configuration();
  TRACE_LOG("report: build configuration is done.");
  TRACE_LOG("report: number of labels (POS and NE tag coupled): %d", model->labels.size());

  // build feature space from the training instance
  TRACE_LOG("report: start building feature space ...");
  build_feature_space();
  TRACE_LOG("report: building feature space is done ...");
  TRACE_LOG("report: number of features: %d", model->space.num_features());

  model->param.realloc(model->space.dim());
  TRACE_LOG("report: allocate %d dimensition parameter.", model->space.dim());

  int nr_groups = model->space.num_groups();
  std::vector<int> groupwise_update_counters;
  if (train_opt.rare_feature_threshold > 0) {
    groupwise_update_counters.resize(nr_groups, 0);
    TRACE_LOG("report: allocate %d update-time counters", nr_groups);
  } else {
    TRACE_LOG("report: model truncation is inactived.");
  }

  // use pa or average perceptron algorithm
  ctx = new ViterbiFeatureContext;
  scm = new ViterbiScoreMatrix;
  decoder = new ViterbiDecoder;
  TRACE_LOG("report: allocated plain decoder");

  int best_iteration = -1;
  double best_f_score = -1.;

  std::vector<size_t> update_counts;
  if (train_opt.rare_feature_threshold > 0) {
    update_counts.resize(nr_groups, 0);
    TRACE_LOG("report: allocate %d update-time counters", nr_groups);
  } else {
    TRACE_LOG("report: model truncation is inactived.");
  }

  for (int iter = 0; iter < train_opt.max_iter; ++ iter) {
    TRACE_LOG("Training iteration #%d", (iter + 1));

    size_t interval = train_dat.size() / 10;
    for (size_t i = 0; i < train_dat.size(); ++ i) {
      Instance* inst = train_dat[i];
      NamedEntityRecognizer::extract_features((*inst), ctx, false);
      NamedEntityRecognizer::calculate_scores((*inst), (*ctx), false, scm);
      decoder->decode((*scm), (*glob_con), inst->predict_tagsidx);

      Frontend::collect_features(model, ctx->uni_features,
          inst->tagsidx, ctx->correct_features);
      Frontend::collect_features(model, ctx->uni_features,
          inst->predict_tagsidx, ctx->predict_features);

      SparseVec update_features;
      Frontend::learn(train_opt.algorithm,
          ctx->correct_features,
          ctx->predict_features,
          iter* train_dat.size()+ i+ 1,
          inst->num_errors(),
          model,
          update_features);

      if (train_opt.rare_feature_threshold > 0) {
        Frontend::increase_groupwise_update_counts(model, update_features, update_counts);
      }

      ctx->clear();
      if ((i+1) % interval == 0) {
        TRACE_LOG("training: %d0%% (%d) instances is trained.", ((i+1)/interval), i+1);
      }
    }
    TRACE_LOG("trace: %d instances is trained.", train_dat.size());
    model->param.flush( train_dat.size()*(iter+1) );

    Model* new_model = Frontend::erase_rare_features(model,
        train_opt.rare_feature_threshold, update_counts);

    std::swap(model, new_model);
    double f_score;
    evaluate(f_score);

    if(f_score > best_f_score){
      best_f_score = f_score;
      best_iteration = iter;
    }

    std::string saved_model_file = (train_opt.model_name+ "."+ to_str(iter));
    std::ofstream ofs(saved_model_file.c_str(), std::ofstream::binary);
    std::swap(model, new_model);
    new_model->save(model_header.c_str(), Parameters::kDumpAveraged, ofs);
    delete new_model;

    TRACE_LOG("trace: model for iteration #%d is saved to %s", iter+1, saved_model_file.c_str());
  }
  TRACE_LOG("Best result (iteration = %d) : F-score = %lf", best_iteration, best_f_score);
}

void NamedEntityRecognizerFrontend::evaluate(double& f_score) {
  const char* holdout_file = train_opt.holdout_file.c_str();

  std::ifstream ifs(holdout_file);

  if (!ifs) {
    ERROR_LOG("Failed to open holdout file.");
    return;
  }

  NERReader reader(ifs, true);
  NERWriter writer(std::cout);
  Instance* inst = NULL;

  int num_recalled_entities = 0;
  int num_predict_entities = 0;
  int num_gold_entities = 0;

  while ((inst = reader.next())) {
    int len = inst->size();
    inst->tagsidx.resize(len);
    for (int i = 0; i < len; ++ i) {
      inst->tagsidx[i] = model->labels.index(inst->tags[i]);
    }

    NamedEntityRecognizer::extract_features((*inst), ctx, false);
    NamedEntityRecognizer::calculate_scores((*inst), (*ctx), true, scm);
    decoder->decode((*scm), (*glob_con), inst->predict_tagsidx);
    ctx->clear();

    build_entities(inst, inst->tagsidx, inst->entities,
        inst->entities_tags);
    build_entities(inst, inst->predict_tagsidx, inst->predict_entities,
        inst->predict_entities_tags);

    num_recalled_entities += inst->num_recalled_entites();
    num_predict_entities += inst->num_predict_entities();
    num_gold_entities += inst->num_gold_entities();

    delete inst;
  }

  double p = (double)num_recalled_entities / num_predict_entities;
  double r = (double)num_recalled_entities / num_gold_entities;
  f_score = 2*p*r / (p + r);

  TRACE_LOG("P: %lf ( %d / %d )", p, num_recalled_entities, num_predict_entities);
  TRACE_LOG("R: %lf ( %d / %d )", r, num_recalled_entities, num_gold_entities);
  TRACE_LOG("F: %lf" , f_score);
  return;
}

void NamedEntityRecognizerFrontend::test(void) {
  // load model
  const char * model_file = test_opt.model_file.c_str();
  std::ifstream mfs(model_file, std::ifstream::binary);

  if (!mfs) {
    ERROR_LOG("Failed to load model");
    return;
  }

  model = new framework::Model(Extractor::num_templates());
  if (!model->load(model_header.c_str(), mfs)) {
    ERROR_LOG("Failed to load model");
    return;
  }

  std::unordered_set<std::string> ne_types;
  for (size_t i = 0; i < model->num_labels(); ++ i) {
    std::string tag = model->labels.at(i);
    if (tag == OTHER) { continue; }
    ne_types.insert(tag.substr(1+delimiter.size()));
  }
  build_glob_tran_cons(ne_types);

  TRACE_LOG("report: number of labels %d", model->num_labels());
  TRACE_LOG("report: number of features %d", model->space.num_features());
  TRACE_LOG("report: number of dimension %d", model->space.dim());

  const char* test_file = test_opt.test_file.c_str();
  std::ifstream ifs(test_file);

  if (!ifs) {
    ERROR_LOG("Failed to open test file.");
    return;
  }

  ctx = new ViterbiFeatureContext;
  scm = new ViterbiScoreMatrix;
  decoder = new framework::ViterbiDecoder;

  NERWriter writer(std::cout);
  NERReader reader(ifs, test_opt.evaluate);
  Instance* inst = NULL;

  double before = get_time();
  TRACE_LOG("report: start testing ...");
  while ((inst = reader.next())) {
    size_t len = inst->size();
    if (test_opt.evaluate) {
      inst->tagsidx.resize(len);
      for (size_t i = 0; i < len; ++ i) {
        inst->tagsidx[i] = model->labels.index(inst->tags[i]);
      }
    }

    NamedEntityRecognizer::extract_features((*inst), ctx, false);
    NamedEntityRecognizer::calculate_scores((*inst), (*ctx), true, scm);
    decoder->decode((*scm), (*glob_con), inst->predict_tagsidx);
    ctx->clear();

    inst->predict_tags.resize(len);
    for(size_t i = 0; i < len; ++i) {
      inst->predict_tags[i] = model->labels.at(inst->predict_tagsidx[i]);
    }
    writer.write(inst);
    delete inst;
  }

  double after = get_time();
  TRACE_LOG("Elapsed time %lf", after - before);
  return;
}

void NamedEntityRecognizerFrontend::dump() {
  // load model
  const char * model_file = dump_opt.model_file.c_str();
  std::ifstream mfs(model_file, std::ifstream::binary);

  if (!mfs) {
    ERROR_LOG("Failed to load model");
    return;
  }

  model = new Model(Extractor::num_templates());
  if (!model->load(model_header.c_str(), mfs)) {
    ERROR_LOG("Failed to load model");
    return;
  }

  int L = model->num_labels();
  TRACE_LOG("report: number of labels %d", model->num_labels());
  TRACE_LOG("report: number of features %d", model->space.num_features());
  TRACE_LOG("report: number of dimension %d", model->space.dim());

  for (framework::FeatureSpaceIterator itx = model->space.begin();
       itx != model->space.end();
       ++ itx) {
    const char * key = itx.key();
    int tid = itx.tid();
    int id = model->space.index(tid, key);

    for (int l = 0; l < L; ++ l) {
      std::cout << key << " ( " << id + l << " ) "
                << " --> "
                << model->param.dot(id + l)
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

} //  namespace ner
} //  namespace ltp
