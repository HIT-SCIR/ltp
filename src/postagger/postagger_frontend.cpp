#include "postagger/postagger_frontend.h"
#include "postagger/io.h"
#include "postagger/extractor.h"
#include "utils/time.hpp"
#include "utils/logging.hpp"
#include "utils/codecs.hpp"

namespace ltp {
namespace postagger {

using framework::Frontend;
using framework::kLearn;
using framework::kTest;
using framework::kDump;

PostaggerFrontend::PostaggerFrontend(const std::string& reference_file,
    const std::string& holdout_file,
    const std::string& model_name,
    const std::string& algorithm,
    const int max_iter,
    const int rare_feature_threshold)
  : decoder(0), decode_context(0), score_matrix(0), Frontend(kLearn) {
  train_opt.train_file             = reference_file;
  train_opt.holdout_file           = holdout_file;
  train_opt.algorithm              = algorithm;
  train_opt.model_name             = model_name;
  train_opt.max_iter               = max_iter;
  train_opt.rare_feature_threshold = rare_feature_threshold;

  TRACE_LOG("||| ltp postagger, training ...");
  TRACE_LOG("report: reference file = %s", train_opt.train_file.c_str());
  TRACE_LOG("report: holdout file = %s", train_opt.holdout_file.c_str());
  TRACE_LOG("report: algorith = %s", train_opt.algorithm.c_str());
  TRACE_LOG("report: model name = %s", train_opt.model_name.c_str());
  TRACE_LOG("report: maximum iteration = %d", train_opt.max_iter);
  TRACE_LOG("report: rare threshold = %d", train_opt.rare_feature_threshold);
}

PostaggerFrontend::PostaggerFrontend(const std::string& model_file,
    const std::string& input_file,
    const std::string& lexicon_file,
    bool evaluate)
  : decoder(0), decode_context(0), score_matrix(0), Frontend(kTest) {
  test_opt.test_file = input_file;
  test_opt.model_file = model_file;
  test_opt.lexicon_file = lexicon_file;
  test_opt.evaluate = evaluate;

  TRACE_LOG("||| ltp postagger, testing ...");
  TRACE_LOG("report: input file = %s", test_opt.test_file.c_str());
  TRACE_LOG("report: model file = %s", test_opt.model_file.c_str());
  TRACE_LOG("report: lexicon file = %s", test_opt.lexicon_file.c_str());
  TRACE_LOG("report: evaluate = %s", (test_opt.evaluate? "true": "false"));
}

PostaggerFrontend::PostaggerFrontend(const std::string& model_file)
  : decoder(0), decode_context(0), score_matrix(0), Frontend(kDump) {
  dump_opt.model_file = "";

  TRACE_LOG("||| ltp postagger, dumpping ...");
  TRACE_LOG("report: model file = %s", model_file.c_str());
}

PostaggerFrontend::~PostaggerFrontend() {
  if (decoder)        { delete decoder;         decoder = 0; }
  if (decode_context) { delete decode_context;  decode_context = 0; }
  if (score_matrix)   { delete score_matrix;    score_matrix = 0; }

  for (size_t i = 0; i < train_dat.size(); ++ i) {
    if (train_dat[i]) {
      delete train_dat[i];
      train_dat[i] = 0;
    }
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

  while ((inst = reader.next())) {
    train_dat.push_back(inst);
  }

  return true;
}

void PostaggerFrontend::build_configuration(void) {
  // model->labels.push( __dummy__ );

  for (int i = 0; i < train_dat.size(); ++ i) {
    Instance * inst = train_dat[i];
    int len = inst->size();
    inst->tagsidx.resize(len);
    inst->postag_constrain.resize(len);
    for (int j = 0; j < len; ++ j) {
      inst->tagsidx[j] = model->labels.push( inst->tags[j] );
      inst->postag_constrain[j].allsetones();
    }
  }
}

void PostaggerFrontend::extract_features(const Instance* inst, bool create) {
  Postagger::extract_features(inst, decode_context, create);
}

void PostaggerFrontend::build_feature_space(void) {
  // build feature space, it a wrapper for featurespace.build_feature_space
  Extractor::num_templates();
  int L = model->num_labels();
  model->space.set_num_labels(L);

  int interval = train_dat.size() / 10;
  for (int i = 0; i < train_dat.size(); ++ i) {
    Postagger::extract_features(train_dat[i], NULL, true);
    // cleanup_decode_context();

    if ((i+ 1) % interval == 0) {
      TRACE_LOG("build-featurespace: %d0%% instances is extracted.", (i+1) / interval);
    }
  }

  TRACE_LOG("trace: feature space is built for %d instances.", train_dat.size());
}

void PostaggerFrontend::calculate_scores(const Instance* inst, bool use_avg) {
  Postagger::calculate_scores(inst, decode_context, use_avg, score_matrix);
}

void PostaggerFrontend::collect_features(const math::Mat< math::FeatureVector* >& features,
    const std::vector<int> & tagsidx, math::SparseVec & vec) {
  int len = tagsidx.size();
  vec.zero();
  for (int i = 0; i < len; ++ i) {
    int l = tagsidx[i];
    const math::FeatureVector * fv = features[i][l];

    if (!fv) {
      continue;
    }

    vec.add(fv->idx, fv->val, fv->n, fv->loff, 1.);

    if (i > 0) {
      int pl = tagsidx[i-1];
      int idx = model->space.index(pl, l);
      vec.add(idx, 1.);
    }
  }
}

void PostaggerFrontend::increase_group_updated_time(const math::SparseVec & vec,
    int * feature_group_updated_time) {
  int L = model->num_labels();
  for (math::SparseVec::const_iterator itx = vec.begin();
      itx != vec.end();
      ++ itx) {

    int idx = itx->first;
    if (itx->second != 0.0) {
      ++ feature_group_updated_time[idx / L];
    }
  }
}

Model * PostaggerFrontend::erase_rare_features(int * feature_group_updated_time) {
  Model * new_model = new Model;
  // copy the label indexable map to the new model
  for (int i = 0; i < model->labels.size(); ++ i) {
    const char * key = model->labels.at(i);
    new_model->labels.push(key);
  }
  TRACE_LOG("trunc-model: building labels map for truncated model is done.");

  int L = new_model->num_labels();
  new_model->space.set_num_labels(L);

  for (FeatureSpaceIterator itx = model->space.begin();
      itx != model->space.end();
      ++ itx) {
    const char * key = itx.key();
    int tid = itx.tid();
    int id = model->space.index(tid, key);
    bool flag = false;

    for (int l = 0; l < L; ++ l) {
      double p = model->param.dot(id + l);
      if (p != 0.) {
        flag = true;
      }
    }

    if (!flag) {
      continue;
    }

    int idx = model->space.retrieve(tid, key, false);
    if (feature_group_updated_time
        && (feature_group_updated_time[idx] < train_opt.rare_feature_threshold)) {
      continue;
    }

    new_model->space.retrieve(tid, key, true);
  }

  TRACE_LOG("trunc-model: building new feature space for truncated model is done.");
  new_model->param.realloc(new_model->space.dim());
  TRACE_LOG("trunc-model: parameter dimension of new model is %d.", new_model->space.dim());

  for (FeatureSpaceIterator itx = new_model->space.begin();
      itx != new_model->space.end();
      ++ itx) {
    const char * key = itx.key();

    int tid = itx.tid();

    int old_id = model->space.index(tid, key);
    int new_id = new_model->space.index(tid, key);

    for (int l = 0; l < L; ++ l) {
      // pay attention to this place, use average should be set true
      // some dirty code
      new_model->param._W[new_id + l]      = model->param._W[old_id + l];
      new_model->param._W_sum[new_id + l]  = model->param._W_sum[old_id + l];
      new_model->param._W_time[new_id + l] = model->param._W_time[old_id + l];
    }
  }

  for (int pl = 0; pl < L; ++ pl) {
    for (int l = 0; l < L; ++ l) {
      int old_id = model->space.index(pl, l);
      int new_id = new_model->space.index(pl, l);

      new_model->param._W[new_id]      = model->param._W[old_id];
      new_model->param._W_sum[new_id]  = model->param._W_sum[old_id];
      new_model->param._W_time[new_id] = model->param._W_time[old_id];
    }
  }
  TRACE_LOG("trunc-model: building parameter for new model is done.");
  TRACE_LOG("trunc-model: building new model is done.");
  return new_model;
}

void PostaggerFrontend::cleanup_decode_context() { decode_context->clear(); }

void PostaggerFrontend::train(void) {
  // read in training instance
  TRACE_LOG("trace: reading reference dataset ...");
  if (!read_instances(train_opt.train_file.c_str())) {
    ERROR_LOG("Training file doesn't exist.");
  }
  TRACE_LOG("trace: %d sentences is loaded.", train_dat.size());

  model = new Model;
  decode_context = new DecodeContext;
  score_matrix = new ScoreMatrix;
  // build tag dictionary, map string tag to index
  TRACE_LOG("report: start build configuration ...");
  build_configuration();
  TRACE_LOG("report: build configuration is done.");
  TRACE_LOG("report: number of postags: %d", model->labels.size());

  // build feature space from the training instance
  TRACE_LOG("report: start building feature space ...");
  build_feature_space();
  TRACE_LOG("report: building feature space is done.");
  TRACE_LOG("report: number of features: %d", model->space.num_features());

  model->param.realloc(model->space.dim());
  TRACE_LOG("report: allocate %d dimensition parameter.", model->space.dim());

  int nr_feature_groups = model->space.num_feature_groups();
  int* feature_group_updated_times = NULL;

  if (train_opt.rare_feature_threshold > 0) {
    feature_group_updated_times = new int[nr_feature_groups];
    for (int i = 0; i < nr_feature_groups; ++ i) {
      feature_group_updated_times[i] = 0;
    }
    TRACE_LOG("report: allocate %d update-time counters", nr_feature_groups);
  } else {
    TRACE_LOG("report: model truncation is inactived.");
  }

  PostaggerWriter writer(std::cout);

  // use pa or average perceptron algorithm
  decoder = new Decoder(model->num_labels());
  TRACE_LOG("trace: allocated plain decoder");

  int best_iteration = -1;
  double best_p = -1.;

  for (int iter = 0; iter < train_opt.max_iter; ++ iter) {
    TRACE_LOG("Training iteraition #%d", (iter + 1));

    int interval= train_dat.size() / 10;
    for (int i = 0; i < train_dat.size(); ++ i) {
      // extract_features(train_dat[i]);
      Instance * inst = train_dat[i];
      extract_features(inst, false);
      calculate_scores(inst, false);
      decoder->decode(inst, score_matrix);

      collect_features(decode_context->uni_features,
          inst->tagsidx, decode_context->correct_features);
      collect_features(decode_context->uni_features,
          inst->predicted_tagsidx, decode_context->predicted_features);

      if (train_opt.algorithm == "pa") {
        decode_context->updated_features.zero();
        decode_context->updated_features.add(decode_context->correct_features, 1.);
        decode_context->updated_features.add(decode_context->predicted_features, -1.);

        if (feature_group_updated_times) {
          increase_group_updated_time(decode_context->updated_features,
              feature_group_updated_times);
        }

        double error = train_dat[i]->num_errors();
        double score = model->param.dot(decode_context->updated_features, false);
        double norm = decode_context->updated_features.L2();

        double step = 0.;
        if (norm < EPS) {
          step = 0;
        } else {
          step = (error - score) / norm;
        }

        model->param.add(decode_context->updated_features,
            iter * train_dat.size() + i + 1,
            step);
      } else if (train_opt.algorithm == "ap") {
        decode_context->updated_features.zero();
        decode_context->updated_features.add(decode_context->correct_features, 1.);
        decode_context->updated_features.add(decode_context->predicted_features, -1.);

        if (feature_group_updated_times) {
          increase_group_updated_time(decode_context->updated_features,
              feature_group_updated_times);
        }
        model->param.add(decode_context->updated_features,
            iter * train_dat.size() + i + 1,
            1.);
      }

      cleanup_decode_context();
      if ((i+1) % interval == 0) {
        TRACE_LOG("training: %d0%% (%d) instances is trained.", ((i+1)/interval), i+1);
      }
    }
    TRACE_LOG("trace: %d instances is trained.", train_dat.size());
    model->param.flush( train_dat.size() * (iter + 1) );
    Model* new_model = NULL;

    new_model = erase_rare_features(feature_group_updated_times);
    std::swap(model, new_model);

    double p;
    evaluate(p);

    if(p > best_p){
      best_p = p;
      best_iteration = iter;
    }

    std::string saved_model_file = (train_opt.model_name
        + "." + strutils::to_str(iter) + ".model");
    std::ofstream ofs(saved_model_file.c_str(), std::ofstream::binary);
    std::swap(model, new_model);
    new_model->save(ofs);
    delete new_model;

    TRACE_LOG("trace: model for iteration #%d is saved to %s", iter+1, saved_model_file.c_str());
  }

  if (feature_group_updated_times) {
    delete [](feature_group_updated_times);
  }

  TRACE_LOG("Best result (iteration = %d) : P = %lf", best_iteration, best_p);
}


void PostaggerFrontend::evaluate(double &p) {
  const char * holdout_file = train_opt.holdout_file.c_str();

  std::ifstream ifs(holdout_file);

  if (!ifs) {
    ERROR_LOG("Failed to open holdout file.");
    return;
  }

  PostaggerReader reader(ifs, "_", true, false);
  Instance * inst = NULL;

  int num_recalled_tags = 0;
  int num_tags = 0;

  while ((inst = reader.next())) {
    int len = inst->size();
    inst->tagsidx.resize(len);
    inst->postag_constrain.resize(len);
    for (int i = 0; i < len; ++ i) {
      inst->tagsidx[i] = model->labels.index(inst->tags[i]);
      inst->postag_constrain[i].allsetones();
    }

    extract_features(inst, false);
    calculate_scores(inst, true);

    decoder->decode(inst, score_matrix);
    cleanup_decode_context();

    num_recalled_tags += inst->num_corrected_predicted_tags();
    num_tags += inst->size();

    delete inst;
  }

  p = (double)num_recalled_tags / num_tags;

  TRACE_LOG("P: %lf ( %d / %d )", p, num_recalled_tags, num_tags);
  return;
}

void PostaggerFrontend::test(void) {
  const char * model_file = test_opt.model_file.c_str();
  std::ifstream mfs(model_file, std::ifstream::binary);

  if (!mfs) {
    ERROR_LOG("Failed to load model");
    return;
  }

  model = new Model;
  if (!model->load(mfs)) {
    ERROR_LOG("Failed to load model");
    return;
  }

  TRACE_LOG("report: number of labels = %d", model->num_labels());
  TRACE_LOG("report: number of features = %d", model->space.num_features());
  TRACE_LOG("report: number of dimension = %d", model->space.dim());

  // load exteranl lexicon
  const char * lexicon_file = test_opt.lexicon_file.c_str();
  load_constrain(model, lexicon_file);
  const char * test_file = test_opt.test_file.c_str();

  std::ifstream ifs(test_file);

  if (!ifs) {
    ERROR_LOG("Failed to open test file.");
    return;
  }

  decoder = new Decoder(model->num_labels());
  score_matrix = new ScoreMatrix;
  decode_context = new DecodeContext;

  PostaggerWriter writer(std::cout);
  PostaggerReader reader(ifs, "_", test_opt.evaluate, false);

  Instance * inst = NULL;
  int num_recalled_tags = 0;
  int num_tags = 0;
  double before = get_time();
  while ((inst = reader.next())) {
    int len = inst->size();
    if (test_opt.evaluate) {
      inst->tagsidx.resize(len);
      for (int i = 0; i < len; ++ i) {
        inst->tagsidx[i] = model->labels.index(inst->tags[i]);
      }
    }
    inst->postag_constrain.resize(len);
    if (model->external_lexicon.size() != 0) {
      for (int i = 0; i < len; ++ i) {
        Bitset * mask = model->external_lexicon.get((inst->forms[i]).c_str());
        if (NULL != mask) {
          inst->postag_constrain[i].merge((*mask));
        } else {
          inst->postag_constrain[i].allsetones();
        }
      }
    } else {
      for (int i = 0; i < len; ++ i) {
        inst->postag_constrain[i].allsetones();
      }
    }
    extract_features(inst, false);
    calculate_scores(inst, true);
    decoder->decode(inst, score_matrix);
    cleanup_decode_context();
    build_labels(inst, inst->predicted_tags);
    writer.write(inst);
    if (test_opt.evaluate) {
      num_recalled_tags += inst->num_corrected_predicted_tags();
      num_tags += inst->size();
    }

    delete inst;
  }

  double after = get_time();
  double p = (double)num_recalled_tags / num_tags;
  if (test_opt.evaluate) {
    TRACE_LOG("P: %lf ( %d / %d )", p, num_recalled_tags, num_tags);
  }
  TRACE_LOG("Elapsed time %lf", after - before);

  //sleep(1000000);
  return;
}

void PostaggerFrontend::dump() {
  // load model
  const char * model_file = dump_opt.model_file.c_str();
  std::ifstream mfs(model_file, std::ifstream::binary);

  if (!mfs) {
    ERROR_LOG("Failed to load model");
    return;
  }

  model = new Model;
  if (!model->load(mfs)) {
    ERROR_LOG("Failed to load model");
    return;
  }

  int L = model->num_labels();
  TRACE_LOG("Number of labels         [%d]", model->num_labels());
  TRACE_LOG("Number of features       [%d]", model->space.num_features());
  TRACE_LOG("Number of dimension      [%d]", model->space.dim());

  for (FeatureSpaceIterator itx = model->space.begin();
       itx != model->space.end();
       ++ itx) {
    const char * key = itx.key();
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
