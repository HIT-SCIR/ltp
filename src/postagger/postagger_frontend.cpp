#include <fstream>
#include "postagger/postagger_frontend.h"
#include "postagger/io.h"
#include "postagger/extractor.h"
#include "postagger/decoder.h"
#include "utils/time.hpp"
#include "utils/logging.hpp"
#include "utils/strutils.hpp"

namespace ltp {
namespace postagger {

using framework::Frontend;
using framework::Parameters;
using framework::Model;
using framework::kLearn;
using framework::kTest;
using framework::kDump;
using math::Mat;
using math::SparseVec;
using math::FeatureVector;
using strutils::to_str;
using utility::timer;

PostaggerFrontend::PostaggerFrontend(const std::string& reference_file,
    const std::string& holdout_file,
    const std::string& model_name,
    const std::string& algorithm,
    const size_t& max_iter,
    const size_t& rare_feature_threshold)
  : Frontend(kLearn) {
  train_opt.train_file = reference_file;
  train_opt.holdout_file = holdout_file;
  train_opt.algorithm = algorithm;
  train_opt.model_name = model_name;
  train_opt.max_iter = max_iter;
  train_opt.rare_feature_threshold = rare_feature_threshold;

  INFO_LOG("||| ltp postagger, training ...");
  INFO_LOG("report: reference file = %s", train_opt.train_file.c_str());
  INFO_LOG("report: holdout file = %s", train_opt.holdout_file.c_str());
  INFO_LOG("report: algorithm = %s", train_opt.algorithm.c_str());
  INFO_LOG("report: model name = %s", train_opt.model_name.c_str());
  INFO_LOG("report: maximum iteration = %d", train_opt.max_iter);
  INFO_LOG("report: rare threshold = %d", train_opt.rare_feature_threshold);
}

PostaggerFrontend::PostaggerFrontend(const std::string& input_file,
    const std::string& model_file,
    const std::string& lexicon_file,
    bool evaluate,
    bool sequence_prob,
    bool marginal_prob)
  : Frontend(kTest) {
  test_opt.test_file = input_file;
  test_opt.model_file = model_file;
  test_opt.lexicon_file = lexicon_file;
  test_opt.evaluate = evaluate;
  test_opt.sequence_prob = sequence_prob;
  test_opt.marginal_prob = marginal_prob;


  INFO_LOG("||| ltp postagger, testing ...");
  INFO_LOG("report: input file = %s", test_opt.test_file.c_str());
  INFO_LOG("report: model file = %s", test_opt.model_file.c_str());
  INFO_LOG("report: lexicon file = %s", test_opt.lexicon_file.c_str());
  INFO_LOG("report: evaluate = %s", (test_opt.evaluate? "true": "false"));
  INFO_LOG("report: sequence probability = %s", (test_opt.sequence_prob? "true": "false"));
  INFO_LOG("report: marginal probability = %s", (test_opt.marginal_prob? "true":"false"));

}

PostaggerFrontend::PostaggerFrontend(const std::string& model_file)
  : Frontend(kDump) {
  dump_opt.model_file = model_file;

  INFO_LOG("||| ltp postagger, dumpping ...");
  INFO_LOG("report: model file = %s", model_file.c_str());
}

PostaggerFrontend::~PostaggerFrontend() {
  for (size_t i = 0; i < train_dat.size(); ++ i) {
    if (train_dat[i]) { delete train_dat[i]; train_dat[i] = 0; }
  }
}

bool PostaggerFrontend::read_instances(const char* train_file) {
  std::ifstream ifs(train_file);

  if (!ifs) {
    return false;
  }

  PostaggerReader reader(ifs, "_", true, true);
  train_dat.clear();

  Instance* inst = NULL;
  while ((inst = reader.next())) { train_dat.push_back(inst); }

  return true;
}

void PostaggerFrontend::build_configuration(void) {
  for (size_t i = 0; i < train_dat.size(); ++ i) {
    Instance * inst = train_dat[i];
    size_t len = inst->size();
    inst->tagsidx.resize(len);
    inst->postag_constrain.resize(len);
    for (size_t j = 0; j < len; ++ j) {
      inst->tagsidx[j] = model->labels.push( inst->tags[j] );
    }
  }
}

void PostaggerFrontend::build_feature_space(void) {
  // build feature space, it a wrapper for featurespace.build_feature_space
  Extractor::num_templates();
  int L = model->num_labels();
  model->space.set_num_labels(L);

  size_t interval = train_dat.size() / 10;
  if (interval == 0) { interval = 1; }
  for (size_t i = 0; i < train_dat.size(); ++ i) {
    Postagger::extract_features((*train_dat[i]), NULL, true);
    if ((i+ 1) % interval == 0) {
      INFO_LOG("build-featurespace: %d0%% instances is extracted.", (i+1) / interval);
    }
  }
  INFO_LOG("trace: feature space is built for %d instances.", train_dat.size());
}

void PostaggerFrontend::train(void) {
  // read in training instance
  INFO_LOG("trace: reading reference dataset ...");
  if (!read_instances(train_opt.train_file.c_str())) {
    ERROR_LOG("Training file doesn't exist.");
  }
  INFO_LOG("trace: %d sentences is loaded.", train_dat.size());

  model = new Model(Extractor::num_templates());
  // build tag dictionary, map string tag to index
  INFO_LOG("report: start building configuration ...");
  build_configuration();
  INFO_LOG("report: build configuration is done.");
  INFO_LOG("report: number of postags: %d", model->labels.size());

  // build feature space from the training instance
  INFO_LOG("report: start building feature space ...");
  build_feature_space();
  INFO_LOG("report: building feature space is done.");
  INFO_LOG("report: number of features: %d", model->space.num_features());

  model->param.realloc(model->space.dim());
  INFO_LOG("report: allocate %d dimensition parameter.", model->space.dim());

  int nr_groups = model->space.num_groups();
  std::vector<size_t> update_counts;

  if (train_opt.rare_feature_threshold > 0) {
    update_counts.resize(nr_groups, 0);
    INFO_LOG("report: allocate %d update-time counters", nr_groups);
  } else {
    INFO_LOG("report: model truncation is inactived.");
  }

  int best_iteration = -1;
  double best_p = -1.;


  for (int iter = 0; iter < train_opt.max_iter; ++ iter) {
    INFO_LOG("Training iteraition #%d", (iter + 1));

    size_t interval= train_dat.size() / 10;
    if (interval == 0) { interval = 1; }
    for (size_t i = 0; i < train_dat.size(); ++ i) {
      Instance* inst = train_dat[i];
      extract_features((*inst), &ctx, false);
      calculate_scores((*inst), ctx, false, &scm);
      decoder.decode(scm, inst->predict_tagsidx);

      collect_features(model, ctx.uni_features, inst->tagsidx, ctx.correct_features);
      collect_features(model, ctx.uni_features, inst->predict_tagsidx, ctx.predict_features);

      SparseVec updated_features;
      updated_features.add(ctx.correct_features, 1.);
      updated_features.add(ctx.predict_features, -1.);

      learn(train_opt.algorithm, updated_features,
        iter*train_dat.size() + i + 1, inst->num_errors(), model);


      if (train_opt.rare_feature_threshold > 0) {
        increase_groupwise_update_counts(model, updated_features, update_counts);
      }

      ctx.clear();
      if ((i+1) % interval == 0) {
        INFO_LOG("training: %d0%% (%d) instances is trained.", ((i+1)/interval), i+1);
      }
    }
    INFO_LOG("trace: %d instances is trained.", train_dat.size());
    model->param.flush( train_dat.size() * (iter + 1) );

    Model* new_model = new Model(Extractor::num_templates());
    erase_rare_features(model, new_model, train_opt.rare_feature_threshold,
        update_counts);

    std::swap(model, new_model);
    double p;
    evaluate(p);
    std::swap(model, new_model);

    if(p > best_p){
      best_p = p;
      best_iteration = iter;

      std::ofstream ofs(train_opt.model_name.c_str(), std::ofstream::binary);
      new_model->save(model_header, Parameters::kDumpAveraged, ofs);

      INFO_LOG("trace: model for iteration #%d is saved to %s",
          iter+1, train_opt.model_name.c_str());
    }

    delete new_model;
  }

  INFO_LOG("Best result (iteration = %d) : P = %lf", best_iteration, best_p);
}


void PostaggerFrontend::evaluate(double &p) {
  const char * holdout_file = train_opt.holdout_file.c_str();

  std::ifstream ifs(holdout_file);

  if (!ifs) {
    ERROR_LOG("Failed to open holdout file.");
    return;
  }

  PostaggerReader reader(ifs, "_", true, false);
  Instance* inst = NULL;

  int num_recalled_tags = 0;
  int num_tags = 0;

  while ((inst = reader.next())) {
    int len = inst->size();
    inst->tagsidx.resize(len);
    // inst->postag_constrain.resize(len);
    for (int i = 0; i < len; ++ i) {
      inst->tagsidx[i] = model->labels.index(inst->tags[i]);
      // inst->postag_constrain[i].allsetones();
    }

    Postagger::extract_features((*inst), &ctx, false);
    Postagger::calculate_scores((*inst), ctx, true, &scm);

    decoder.decode(scm, inst->predict_tagsidx);
    ctx.clear();

    num_recalled_tags += inst->num_corrected_predict_tags();
    num_tags += inst->size();

    delete inst;
  }

  p = (double)num_recalled_tags / num_tags;

  INFO_LOG("P: %lf ( %d / %d )", p, num_recalled_tags, num_tags);
  return;
}

void PostaggerFrontend::test(void) {
  const char * model_file = test_opt.model_file.c_str();
  std::ifstream mfs(model_file, std::ifstream::binary);

  if (!mfs) {
    ERROR_LOG("Failed to load model");
    return;
  }

  model = new Model(Extractor::num_templates());
  if (!model->load(model_header, mfs)) {
    ERROR_LOG("Failed to load model");
    return;
  }

  INFO_LOG("report: number of labels = %d", model->num_labels());
  INFO_LOG("report: number of features = %d", model->space.num_features());
  INFO_LOG("report: number of dimension = %d", model->space.dim());

  // load exteranl lexicon
  // const char * lexicon_file = test_opt.lexicon_file.c_str();
  // load_constrain(model, lexicon_file);
  const char * test_file = test_opt.test_file.c_str();
  std::ifstream ifs(test_file);

  if (!ifs) {
    ERROR_LOG("Failed to open test file.");
    return;
  }

  PostaggerWriter writer(std::cout, test_opt.sequence_prob, test_opt.marginal_prob);
  PostaggerReader reader(ifs, "_", test_opt.evaluate, false);

  decoder.set_sequence_prob(test_opt.sequence_prob);
  decoder.set_marginal_prob(test_opt.marginal_prob);

  PostaggerLexicon lex;
  std::ifstream lfs(test_opt.lexicon_file.c_str());
  if (lfs.good()) { lex.load(lfs, model->labels); }

  Instance* inst = NULL;
  size_t num_recalled_tags = 0;
  size_t num_tags = 0;
  timer t;
  while ((inst = reader.next())) {
    int len = inst->size();
    if (test_opt.evaluate) {
      inst->tagsidx.resize(len);
      for (int i = 0; i < len; ++ i) {
        inst->tagsidx[i] = model->labels.index(inst->tags[i]);
      }
    }
    Postagger::extract_features((*inst), &ctx, false);
    Postagger::calculate_scores((*inst), ctx, true, &scm);

    if (lex.success()) {
      PostaggerLexiconConstrain con = lex.get_con(inst->forms);
      if (test_opt.sequence_prob || test_opt.marginal_prob) {
        decoder.decode(scm,
                con,
                inst->predict_tagsidx,
                inst->sequence_probability,
                inst->point_probabilities,
                true,
                model->param._last_timestamp);
      } else {
        decoder.decode(scm, con, inst->predict_tagsidx);
      }
    } else {
      if (test_opt.sequence_prob || test_opt.marginal_prob) {
        decoder.decode(scm,
                inst->predict_tagsidx,
                inst->sequence_probability,
                inst->point_probabilities,
                true,
                model->param._last_timestamp);
      } else {
        decoder.decode(scm, inst->predict_tagsidx);
      }
    }
    ctx.clear();

    build_labels((*inst), inst->predict_tags);
    if (test_opt.evaluate) {
      num_recalled_tags += inst->num_corrected_predict_tags();
      num_tags += inst->size();
    }
    writer.write(inst);
    delete inst;
  }

  if (test_opt.evaluate) {
    double p = (double)num_recalled_tags / num_tags;
    INFO_LOG("P: %lf ( %d / %d )", p, num_recalled_tags, num_tags);
  }

  INFO_LOG("Elapsed time %lf", t.elapsed());

  //sleep(1000000);
  return;
}

void PostaggerFrontend::dump(void) {
  // load model
  const char* model_file = dump_opt.model_file.c_str();
  std::ifstream mfs(model_file, std::ifstream::binary);

  if (!mfs) {
    ERROR_LOG("Failed to load model");
    return;
  }

  model = new Model(Extractor::num_templates());
  if (!model->load(model_header, mfs)) {
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

} //  namespace postagger
} //  namespace ltp
