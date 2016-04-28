#define private   public
#define protected public
#include <iostream>
#include <fstream>
#include "config.h"
#include "boost/program_options.hpp"
#include "utils/logging.hpp"
#include "framework/model.h"
#include "segmentor/model.h"
#include "segmentor/segmentor.h"
#include "postagger/extractor.h"
#include "postagger/postagger.h"
#include "ner/extractor.h"
#include "ner/ner.h"
#include "parser.n/parser.h"

using boost::program_options::options_description;
using boost::program_options::value;
using boost::program_options::variables_map;
using boost::program_options::store;
using boost::program_options::parse_command_line;

const int kNumberOfSegmentTags = 4;
const char* kSegmentTags[] = {"b", "i", "e", "s"};

bool test_segmentor_model(const std::string& filename) {
  std::ifstream mfs(filename.c_str());
  if (!mfs) {
    WARNING_LOG("testing segment model: model not exist.");
    return false;
  }

  ltp::segmentor::Model* model = new ltp::segmentor::Model();
  if (!model->load(ltp::segmentor::Segmentor::model_header.c_str(), mfs)) {
    WARNING_LOG("testing segment model: failed to load model.");
    return false;
  }

  if (kNumberOfSegmentTags != model->labels.size()) {
    WARNING_LOG("testing segment model: number of tags mismatch.");
    return false;
  }

  for (size_t i = 0; i < kNumberOfSegmentTags; ++ i) {
    if (model->labels.index(kSegmentTags[i]) != -1) {
      if (strcmp(model->labels.at(i), kSegmentTags[i])) {
        WARNING_LOG("testing segment model: tag(%d) in wrong position.", i);
        return false;
      }
    } else {
      WARNING_LOG("testing segment model: tag(%s) not found.", kSegmentTags[i]);
      return false;
    }
  }

  if (model->param.is_wrapper()) {
    WARNING_LOG("testing segment model: release model should be in dump details mode.");
    return false;
  }
  delete model;
  return true;
}

const int kNumberOfPOSTags = 27;
const char * kPostags[] = {"a",  "b",  "c",
  "d",  "e",  "h",  "i",  "j",  "k",  "m",
  "n",  "nd", "nh", "ni", "nl", "ns", "nt",
  "nz", "o",  "p",  "q",  "r",  "u",  "v",
  "wp", "ws", "z"};

bool test_postagger_model(const std::string& filename) {
  std::ifstream mfs(filename.c_str());
  if (!mfs) {
    WARNING_LOG("testing postag model: model not exist.");
    return false;
  }

  ltp::framework::Model* model =
    new ltp::framework::Model(ltp::postagger::Extractor::num_templates());

  if (!model->load(ltp::postagger::Postagger::model_header, mfs)) {
    WARNING_LOG("testing postag model: failed to load model.");
    return false;
  }

  if (kNumberOfPOSTags != model->labels.size()) {
    WARNING_LOG("testing postag model: number of tags mismatch.");
    INFO_LOG("testing postag model: model gives %d tags, which is expected to be %d",
        model->labels.size(), kNumberOfPOSTags);
    for (size_t i = 0; i < model->labels.size(); ++ i) {
      INFO_LOG("testing postag model: model gives %d, %s", i, model->labels.at(i));
    }
    return false;
  }

  for (size_t i = 0; i < kNumberOfPOSTags; ++ i) {
    if (model->labels.index(kPostags[i]) == -1) {
      WARNING_LOG("testing postag model: tag(%s) not found.", kPostags[i]);
      return false;
    }
  }

  if (!model->param.is_wrapper()) {
    WARNING_LOG("testing postag model: release model should be in simplified mode.");
    return false;
  }
  delete model;
  return true;
}

bool test_ner_model(const std::string& filename) {
  std::ifstream mfs(filename.c_str());
  if (!mfs) {
    WARNING_LOG("testing ne model: model not exist.");
    return false;
  }

  ltp::framework::Model* model =
    new ltp::framework::Model(ltp::ner::Extractor::num_templates());

  if (!model->load(ltp::ner::NamedEntityRecognizer::model_header, mfs)) {
    WARNING_LOG("testing ne model: failed to load model.");
    return false;
  }

  if (!model->param.is_wrapper()) {
    WARNING_LOG("testing postag model: release model should be in simplified mode.");
    return false;
  }
  delete model;
  return true;
}

const int kNumberOfDeprels = 14;
const char * kDeprels[] = {"SBV", "VOB",
  "IOB", "FOB", "DBL", "ATT", "ADV",
  "CMP", "COO", "POB", "LAD", "RAD",
  "IS", "HED",};

bool test_parser_model(const std::string& filename) {
  bool ret = true;
  ltp::depparser::NeuralNetworkParser parser;
  if (!parser.load(filename)) {
    WARNING_LOG("testing parser model: failed to load model.");
    return false;
  }

  if (kNumberOfPOSTags+2 != parser.postags_alphabet.size()) {
    // with root, unk, nil, without z
    WARNING_LOG("testing parser model: number of postag tags mismatch.");
    INFO_LOG("testing parser model: model gives %d tags, which is expected to be %d",
        parser.postags_alphabet.size(), kNumberOfPOSTags+2);
    for (size_t i = 0; i < parser.postags_alphabet.size(); ++ i) {
      INFO_LOG("testing parser model: model gives %d, %s", i, parser.postags_alphabet.at(i));
    }
    ret = false;
  }

  for (size_t i = 0; i < kNumberOfPOSTags; ++ i) {
    if (parser.postags_alphabet.index(kPostags[i]) == -1) {
      WARNING_LOG("testing parser model: postag tag(%s) not found.", kPostags[i]);
    }
    ret = false;
  }

  if (kNumberOfDeprels+2 != parser.deprels_alphabet.size()) {
    // with root, nil
    WARNING_LOG("testing parser model: number of deprels mismatch.");
    INFO_LOG("testing parser model: model gives %d relations, which is expected to be %d",
        parser.deprels_alphabet.size(), kNumberOfDeprels+2);
    for (size_t i = 0; i < parser.deprels_alphabet.size(); ++ i) {
      INFO_LOG("testing parser model: model gives %d, %s", i, parser.deprels_alphabet.at(i));
    }
    ret = false;
  }

  return ret;
}

#define EXECUTABLE "model_validation"
#define DESCRIPTION "model validation suite."

int main(int argc, char* argv[]) {
  std::string usage = EXECUTABLE " in LTP " LTP_VERSION " - " LTP_COPYRIGHT "\n";
  usage += DESCRIPTION "\n\n";
  usage += "usage: ./" EXECUTABLE " <options>\n\n";
  usage += "options";

  options_description optparser = options_description(usage);
  optparser.add_options()
    ("segmentor-model", value<std::string>(),
     "The path to the segment model [default=ltp_data/cws.model].")
    ("postagger-model", value<std::string>(),
     "The path to the postag model [default=ltp_data/pos.model].")
    ("ner-model", value<std::string>(),
     "The path to the NER model [default=ltp_data/ner.model].")
    ("parser-model", value<std::string>(),
     "The path to the parser model [default=ltp_data/parser.model].")
    ("srl-data", value<std::string>(),
     "The path to the SRL model directory [default=ltp_data/srl_data/].")
    ("help,h", "Show help information");

  if (argc == 1) {
    std::cerr << optparser << std::endl;
    return 1;
  }

  variables_map vm;
  store(parse_command_line(argc, argv, optparser), vm);

  std::string segmentor_model = "ltp_data/cws.model";
  if (vm.count("segmentor-model")) {
    segmentor_model = vm["segmentor-model"].as<std::string>();
  }

  std::string segmentor_lexicon = "";
  if (vm.count("segmentor-lexicon")) {
    segmentor_lexicon= vm["segmentor-lexicon"].as<std::string>();
  }

  std::string postagger_model = "ltp_data/pos.model";
  if (vm.count("postagger-model")) {
    postagger_model= vm["segmentor-model"].as<std::string>();
  }

  std::string postagger_lexcion = "";
  if (vm.count("postagger-lexicon")) {
    postagger_lexcion= vm["postagger-lexicon"].as<std::string>();
  }

  std::string ner_model = "ltp_data/ner.model";
  if (vm.count("ner-model")) {
    ner_model= vm["ner-model"].as<std::string>();
  }

  std::string parser_model = "ltp_data/parser.model";
  if (vm.count("parser-model")) {
    parser_model= vm["parser-model"].as<std::string>();
  }

  if (!test_segmentor_model(segmentor_model)) {
    WARNING_LOG("testing segment model: failed.");
  } else {
    INFO_LOG("testing segment model: success.");
  }

  if (!test_postagger_model(postagger_model)) {
    WARNING_LOG("testing postag model: failed.");
  } else {
    INFO_LOG("testing postag model: success.");
  }

  if (!test_ner_model(ner_model)) {
    WARNING_LOG("testing ne model: failed.");
  } else {
    INFO_LOG("testing ne model: success.");
  }

  if (!test_parser_model(parser_model)) {
    WARNING_LOG("testing parser model: failed.");
  } else {
    INFO_LOG("testing parser model: success.");
  }

  return 0;
}
