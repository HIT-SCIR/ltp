#include <iostream>
#include <fstream>
#include <sstream>
#include <list>
#include "config.h"
#include "ner/ner_dll.h"
#include "console/dispatcher.h"
#include "boost/program_options.hpp"
#include "utils/strutils.hpp"
#include "utils/time.hpp"

#define EXECUTABLE "ner_cmdline"
#define DESCRIPTION "The console application for named entity recognization."

using boost::program_options::options_description;
using boost::program_options::value;
using boost::program_options::variables_map;
using boost::program_options::store;
using boost::program_options::parse_command_line;
using ltp::utility::WallClockTimer;
using ltp::strutils::split;

void multithreaded_recognize( void * args) {
  std::vector<std::string> words;
  std::vector<std::string> postags;
  std::vector<std::string> tags;

  Dispatcher * dispatcher = (Dispatcher *)args;
  void * model = dispatcher->get_engine();
  std::string buffer;
  std::string token;
  while (true) {
    int ret = dispatcher->next(buffer);
    if (ret < 0)
      break;

    std::stringstream S(buffer);
    words.clear();
    postags.clear();
    while (S >> token) {
      size_t npos = token.find_last_of("_");
      words.push_back(token.substr(0, npos));
      postags.push_back(token.substr(npos+ 1));
    }

    tags.clear();
    ner_recognize(model, words, postags, tags);
    std::string output;

    S.clear(); S.str("");
    for (size_t i = 0; i < tags.size(); ++ i) {
      if (i > 0) { S << "\t"; }
      S << words[i] << "/" << postags[i] << "#" << tags[i];
    }
    dispatcher->output(ret, S.str());
  }

  return;
}

int main(int argc, char * argv[]) {
  std::string usage = EXECUTABLE " in LTP " LTP_VERSION " - " LTP_COPYRIGHT "\n";
  usage += DESCRIPTION "\n\n";
  usage += "usage: ./" EXECUTABLE " <options>\n\n";
  usage += "options";

  options_description optparser = options_description(usage);
  optparser.add_options()
    ("threads", value<int>(), "The number of threads [default=1].")
    ("input", value<std::string>(), "The path to the input file. "
     "Input data should contain one sentence each line. "
     "Words should be separated by space with POS tag appended by "
     "'_' (e.g. \"w1_p1 w2_p2 w3_p3 w4_p4\").")
    ("ner-model", value<std::string>(),
     "The path to the postag model [default=ltp_data/ner.model].")
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

  std::string input = "";
  if (vm.count("input")) { input = vm["input"].as<std::string>(); }

  std::string ner_model = "ltp_data/ner.model";
  if (vm.count("ner-model")) {
    ner_model = vm["ner-model"].as<std::string>();
  }

  void *engine = ner_create_recognizer(ner_model.c_str());
  if (!engine) {
    return 1;
  }

  std::cerr << "TRACE: Model is loaded" << std::endl;
  std::cerr << "TRACE: Running " << threads << " thread(s)" << std::endl;

  std::ifstream ifs(input.c_str());
  std::istream* is = NULL;

  if (!ifs.good()) {
    std::cerr << "WARN: Cann't open file! use stdin instead." << std::endl;
    is = (&std::cin);
  } else {
    is = (&ifs);
  }

  Dispatcher * dispatcher = new Dispatcher( engine, (*is), std::cout );
  WallClockTimer t;
  std::list<tthread::thread *> thread_list;
  for (int i = 0; i < threads; ++ i) {
    tthread::thread * t = new tthread::thread( multithreaded_recognize, (void *)dispatcher );
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
  ner_release_recognizer(engine);
  return 0;
}

