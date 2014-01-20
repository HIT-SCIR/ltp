#ifndef __LTP_NER_OPTIONS_H__
#define __LTP_NER_OPTIONS_H__

#include <iostream>

namespace ltp {
namespace ner {

struct ModelOptions {
  std::string   model_file;
};

struct TrainOptions {
  std::string   train_file;
  std::string   holdout_file;
  std::string   model_name;
  std::string   algorithm;
  int           max_iter;
  int           display_interval;
};

struct TestOptions {
  std::string   test_file;
  std::string   model_file;
  std::string   lexicon_file;
};

struct DumpOptions {
  std::string   model_file;
};

extern ModelOptions model_opt;
extern TrainOptions train_opt;
extern TestOptions  test_opt;
extern DumpOptions  dump_opt;

}       //  end for namespace ner
}       //  end for namespace ltp

#endif    //  end for __LTP_NER_OPTIONS_H__
