#ifndef __LTP_PARSER_OPTIONS_H__
#define __LTP_PARSER_OPTIONS_H__

#include <iostream>

namespace ltp {
namespace parser {

using namespace std;

struct TrainOptions {
  string  train_file;             /*< the training file */
  string  holdout_file;           /*< the develop file */
  string  algorithm;              /*< the algorithm */
  string  model_name;             /*< the model name */
  int     rare_feature_threshold; /*< specify the max number of rare feature */
  int     max_iter;               /*< the iteration number */
};

struct TestOptions {
  string test_file;         /*< test file path */
  string model_file;        /*< model file path. in test mode, config is
                             *< writen in model */
};

struct FeatureOptions {
  bool use_postag;            /*< use postag feature, not implemented */
  bool use_postag_unigram;    /*< use postag unigram feature, not implemented */
  bool use_postag_bigram;     /*< use postag bigram feature, not implemented */
  bool use_postag_chars;      // template: pos+chars

  // dependency feature group
  bool use_dependency;            /*< use dependency feature */
  bool use_dependency_unigram;    /*< use dependency unigram feature */
  bool use_dependency_bigram;     /*< use dependency bigram feature */
  bool use_dependency_surrounding;/*< use dependency surrounding feature */
  bool use_dependency_between;    /*< use dependency between features */

  bool use_sibling;               /*< use sibling feature */
  bool use_sibling_basic;         /*< use sibling basic feature */
  bool use_sibling_linear;        /*< use sibling linear feature */

  bool use_grand;                 /*< use grand features, not implemented */
  bool use_grand_basic;
  bool use_grand_linear;

  // sth weired
  bool use_last_sibling;
  bool use_no_grand;

  // automaticall calculate
  bool use_distance_in_features;  /*< use distance, always true */
  bool use_unlabeled_dependency;  /*< equals to !model.labeled and use_dependency */
  bool use_labeled_dependency;    /*< equals to model.labeled and use_dependency */
  bool use_unlabeled_sibling;     /*< equals to !model.labeled and use_sibling */
  bool use_labeled_sibling;       /*< equals to model.labeled and use_sibling */
  bool use_unlabeled_grand;       /*< equals to !model.labeled and use_grand */
  bool use_labeled_grand;         /*< equals to model.labeled and use_grand */

  bool use_lemma;
  bool use_coarse_postag;

};

struct ModelOptions {
  bool    labeled;            /*< specify use label */
  string  decoder_name;       /*< the training order */
  int     display_interval;   /*< the display interval */
};

// declear the global options
extern ModelOptions   model_opt;
extern TrainOptions   train_opt;
extern TestOptions    test_opt;
extern FeatureOptions   feat_opt;

}     //  end for namespace parser
}     //  end for namespace ltp

#endif  //  end for __LTP_PARSER_OPTIONS_H__
