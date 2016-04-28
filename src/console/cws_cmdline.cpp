/**
 * Multi-threaded segmentor test program. The user input a line
 * of Chinese sentence an the program will output its segment
 * result.
 *
 *  @dependency package: tinythread - a portable c++ wrapper for
 *                       multi-thread library.
 *  @author:             LIU, Yijia
 *  @date:               2013-09-24
 */
#include <iostream>
#include <fstream>
#include <list>
#include "config.h"
#include "segmentor/segment_dll.h"
#include "console/dispatcher.h"
#include "boost/program_options.hpp"
#include "utils/strutils.hpp"
#include "utils/time.hpp"

#define EXECUTABLE "cws_cmdline"
#define DESCRIPTION "The console application for Chinese word segmentation."

using boost::program_options::options_description;
using boost::program_options::value;
using boost::program_options::variables_map;
using boost::program_options::store;
using boost::program_options::parse_command_line;
using ltp::strutils::join;
using ltp::utility::WallClockTimer;

void multithreaded_segment( void * args) {
  std::string sentence;
  std::vector<std::string> result;

  Dispatcher * dispatcher = (Dispatcher *)args;
  void* model = dispatcher->get_engine();

  while (true) {
    int ret = dispatcher->next(sentence);
    if (ret < 0)
      break;

    result.clear();
    segmentor_segment(model, sentence, result);
    dispatcher->output(ret, join(result, "\t"));
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
     "Input data should contain one raw sentence each line.")
    ("segmentor-model", value<std::string>(),
     "The path to the segment model [default=ltp_data/cws.model].")
    ("segmentor-lexicon", value<std::string>(),
     "The path to the external lexicon in segmentor [optional].")
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

  std::string segmentor_model = "ltp_data/cws.model";
  if (vm.count("segmentor-model")) {
    segmentor_model = vm["segmentor-model"].as<std::string>();
  }

  std::string segmentor_lexicon = "";
  if (vm.count("segmentor-lexicon")) {
    segmentor_lexicon= vm["segmentor-lexicon"].as<std::string>();
  }

  void* engine = 0;
  if (segmentor_lexicon == "") {
    engine = segmentor_create_segmentor(segmentor_model.c_str());
  } else {
    engine = segmentor_create_segmentor(segmentor_model.c_str(), segmentor_lexicon.c_str());
  }

  if (!engine) {
    return -1;
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

  Dispatcher* dispatcher = new Dispatcher( engine, (*is), std::cout );
  WallClockTimer t;
  std::list<tthread::thread *> thread_list;
  for (int i = 0; i < threads; ++ i) {
    tthread::thread* t = new tthread::thread( multithreaded_segment, (void *)dispatcher );
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
  segmentor_release_segmentor(engine);
  return 0;
}

