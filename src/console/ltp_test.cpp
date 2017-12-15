// Defines the entry point for the console application.
#include <iostream>
#include <vector>
#include <list>
#include "config.h"
#include "xml4nlp/Xml4nlp.h"
#include "ltp/Ltp.h"
#include "utils/strutils.hpp"
#include "utils/time.hpp"
#include "utils/xml4nlp_helper.h"
#include "console/dispatcher.h"
#include "boost/program_options.hpp"

#define EXECUTABLE "ltp_test"
#define DESCRIPTION "The console application for Language Technology Platform."

using boost::program_options::options_description;
using boost::program_options::value;
using boost::program_options::variables_map;
using boost::program_options::store;
using boost::program_options::parse_command_line;
using ltp::strutils::trim;

std::string type;
std::string format;

void multithreaded_ltp( void * args) {
  std::string sentence;
  Dispatcher * dispatcher = (Dispatcher *)args;
  LTP* engine = (LTP*)dispatcher->get_engine();

  while (true) {
    int ret = dispatcher->next(sentence);
    if (ret < 0)
      break;

    XML4NLP xml4nlp;
    xml4nlp.CreateDOMFromString(sentence);

    /*if (type == LTP_SERVICE_NAME_SPLITSENT) {
      engine->splitSentence_dummy(xml4nlp);
    } else*/ if(type == LTP_SERVICE_NAME_SEGMENT) {
      engine->wordseg(xml4nlp);
    } else if(type == LTP_SERVICE_NAME_POSTAG) {
      engine->postag(xml4nlp);
    } else if(type == LTP_SERVICE_NAME_NER) {
      engine->ner(xml4nlp);
    } else if(type == LTP_SERVICE_NAME_DEPPARSE) {
      engine->parser(xml4nlp);
    } else if(type == LTP_SERVICE_NAME_SRL) {
      engine->srl(xml4nlp);
    } else {
      engine->srl(xml4nlp);
    }

    std::string result;
    if (format == LTP_SERVICE_OUTPUT_FORMAT_JSON) {
      result = ltp::utility::xml2jsonstr(xml4nlp, type);
    } else { //xml
      xml4nlp.SaveDOM(result);
    }
    dispatcher->output(ret, result);
    xml4nlp.ClearDOM();
  }
  return;
}

int main(int argc, char *argv[]) {
  std::string usage = EXECUTABLE " in LTP " LTP_VERSION " - " LTP_COPYRIGHT "\n";
  usage += DESCRIPTION "\n\n";
  usage += "usage: ./" EXECUTABLE " <options>\n\n";
  usage += "options";

  options_description optparser = options_description(usage);
  optparser.add_options()
    ("threads", value<int>(), "The number of threads [default=1].")
    ("last-stage", value<std::string>(),
     "The last stage of analysis. This option can be used when the user only"
     "wants to perform early stage analysis, like only segment without postagging."
     "value includes:\n"
     "- " LTP_SERVICE_NAME_SEGMENT ": Chinese word segmentation\n"
     "- " LTP_SERVICE_NAME_POSTAG ": Part of speech tagging\n"
     "- " LTP_SERVICE_NAME_NER ": Named entity recognization\n"
     "- " LTP_SERVICE_NAME_DEPPARSE ": Dependency parsing\n"
     "- " LTP_SERVICE_NAME_SRL ": Semantic role labeling (equals to all)\n"
     "- " LTP_SERVICE_NAME_ALL ": The whole pipeline [default]")
    ("format", value<std::string>(), "Ouput format\n"
     "- " LTP_SERVICE_OUTPUT_FORMAT_XML " [default]\n"
     "- " LTP_SERVICE_OUTPUT_FORMAT_JSON)
    ("input", value<std::string>(), "The path to the input file.")
    ("segmentor-model", value<std::string>(),
     "The path to the segment model [default=ltp_data/cws.model].")
    ("segmentor-lexicon", value<std::string>(),
     "The path to the external lexicon in segmentor [optional].")
    ("postagger-model", value<std::string>(),
     "The path to the postag model [default=ltp_data/pos.model].")
    ("postagger-lexicon", value<std::string>(),
     "The path to the external lexicon in postagger [optional].")
    ("ner-model", value<std::string>(),
     "The path to the NER model [default=ltp_data/ner.model].")
    ("parser-model", value<std::string>(),
     "The path to the parser model [default=ltp_data/parser.model].")
    ("srl-model", value<std::string>(),
     "The path to the SRL model [default=ltp_data/pisrl.model].")
    ("debug-level", value<int>(), "The debug level.")
    ("help,h", "Show help information");

  if (argc == 1) {
    std::cerr << optparser << std::endl;
    return 1;
  }

  variables_map vm;
  store(parse_command_line(argc, argv, optparser), vm);

  if (vm.count("help")) {
    std::cerr << optparser << std::endl;
    return 0;
  }

  int threads = 1;
  if (vm.count("threads")) {
    threads = vm["threads"].as<int>();
    if (threads < 0) {
      std::cerr << "number of threads should not less than 0, reset to 1." << std::endl;
      threads = 1;
    }
  }

  std::string last_stage = "all";
  if (vm.count("last-stage")) {
    last_stage = vm["last-stage"].as<std::string>();
    vector<string> stages = ltp::strutils::split_by_sep(last_stage, "|");

    for (int j = 0; j < stages.size(); ++j) {
      if (stages[j] != LTP_SERVICE_NAME_SEGMENT
          && stages[j] != LTP_SERVICE_NAME_POSTAG
          && stages[j] != LTP_SERVICE_NAME_NER
          && stages[j] != LTP_SERVICE_NAME_DEPPARSE
          && stages[j] != LTP_SERVICE_NAME_SRL
          && stages[j] != "all") {
        std::cerr << "Unknown stage name:" << last_stage << ", reset to 'all'" << std::endl;
        last_stage = "all";
        break;
      }
    }
  }

  format = LTP_SERVICE_OUTPUT_FORMAT_DEFAULT;
  if (vm.count("format")) {
    format = vm["format"].as<std::string>();
    if (format != LTP_SERVICE_OUTPUT_FORMAT_XML
        && format != LTP_SERVICE_OUTPUT_FORMAT_JSON) {
      std::cerr << "Unknown format:" << last_stage << ", reset to '" LTP_SERVICE_OUTPUT_FORMAT_DEFAULT "'" << std::endl;
      format = LTP_SERVICE_OUTPUT_FORMAT_DEFAULT;
    }
  }

  std::string input = "";
  if (vm.count("input")) { input = vm["input"].as<std::string>(); }

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
    postagger_model= vm["postagger-model"].as<std::string>();
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

  std::string srl_model= "ltp_data/pisrl.model";
  if (vm.count("srl-data")) {
    srl_model = vm["srl-data"].as<std::string>();
  }

  LTP engine(last_stage, segmentor_model, segmentor_lexicon, postagger_model,
      postagger_lexcion, ner_model, parser_model, srl_model);

  if (!engine.loaded()) {
    std::cerr << "Failed to load LTP" << std::endl;
    return 1;
  }

  type = last_stage;
  std::ifstream ifs(input.c_str());
  std::istream* is = NULL;

  if (!ifs.good()) {
    std::cerr << "WARN: Cann't open file! use stdin instead." << std::endl;
    is = (&std::cin);
  } else {
    is = (&ifs);
  }

  ltp::utility::WallClockTimer t;
  Dispatcher* dispatcher = new Dispatcher(&engine, (*is), std::cout);
  int num_threads = threads;
  std::cerr << "TRACE: LTP is built" << std::endl;
  std::cerr << "TRACE: Running " << num_threads << " thread(s)" << std::endl;

  std::list<tthread::thread *> thread_list;
  for (size_t i = 0; i < num_threads; ++ i) {
    tthread::thread * t = new tthread::thread(multithreaded_ltp, (void *)dispatcher );
    thread_list.push_back( t );
  }

  for (std::list<tthread::thread *>::iterator i = thread_list.begin();
      i != thread_list.end(); ++ i) {
    tthread::thread * t = *i;
    t->join();
    delete t;
  }
  std::cerr << "TRACE: consume " << t.elapsed() << " seconds." << std::endl;
  delete dispatcher;
  return 0;
}
