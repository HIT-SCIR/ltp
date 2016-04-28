/*
 * Multi-threaded postagger test program. The user input a line
 * of Chinese sentence an the program will output its segment
 * result.
 *
 *  @dependency package: tinythread - a portable c++ wrapper for
 *                       multi-thread library.
 *  @author:             LIU, Yijia
 *  @data:               2013-09-24
 */
#include <iostream>
#include <fstream>
#include <list>
#include "config.h"
#include "postagger/postag_dll.h"
#include "console/dispatcher.h"
#include "boost/program_options.hpp"
#include "utils/strutils.hpp"
#include "utils/time.hpp"

#define EXECUTABLE "pos_cmdline"
#define DESCRIPTION "The console application for Part of speech tagging."

using boost::program_options::options_description;
using boost::program_options::value;
using boost::program_options::variables_map;
using boost::program_options::store;
using boost::program_options::parse_command_line;
using ltp::utility::WallClockTimer;
using ltp::strutils::split;

void multithreaded_postag( void * args) {
  std::vector<std::string> words;
  std::vector<std::string> postags;

  Dispatcher * dispatcher = (Dispatcher *)args;
  void * model = dispatcher->get_engine();
  std::string buffer;
  while (true) {
    int ret = dispatcher->next(buffer);
    if (ret < 0)
      break;

    split(buffer, words);
    postags.clear();
    postagger_postag(model, words, postags);
    std::string output;
    for (size_t i = 0; i < words.size(); ++ i) {
      if (i > 0) { output.append("\t"); }
      output.append(words[i]);
      output.append("_");
      output.append(postags[i]);
    }
    dispatcher->output(ret, output);
  }

  return;
}

int main(int argc, char ** argv) {
  std::string usage = EXECUTABLE " in LTP " LTP_VERSION " - " LTP_COPYRIGHT "\n";
  usage += DESCRIPTION "\n\n";
  usage += "usage: ./" EXECUTABLE " <options>\n\n";
  usage += "options";

  options_description optparser = options_description(usage);
  optparser.add_options()
    ("threads", value<int>(), "The number of threads [default=1].")
    ("input", value<std::string>(), "The path to the input file. "
     "Input data should contain one sentence each line. "
     "Words should be separated by space (e.g. \"w1 w2 w3 w4\").")
    ("postagger-model", value<std::string>(),
     "The path to the postag model [default=ltp_data/pos.model].")
    ("postagger-lexicon", value<std::string>(),
     "The path to the external lexicon in postagger [optional].")
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

  std::string postagger_model = "ltp_data/pos.model";
  if (vm.count("postagger-model")) {
    postagger_model= vm["postagger-model"].as<std::string>();
  }

  std::string postagger_lexcion = "";
  if (vm.count("postagger-lexicon")) {
    postagger_lexcion= vm["postagger-lexicon"].as<std::string>();
  }

  void *engine = NULL;
  if (postagger_lexcion == "") {
    engine = postagger_create_postagger(postagger_model.c_str());
  } else {
    engine = postagger_create_postagger(postagger_model.c_str(), postagger_lexcion.c_str());
  }

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
    tthread::thread * t = new tthread::thread( multithreaded_postag, (void *)dispatcher );
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
  postagger_release_postagger(engine);
  return 0;
}

