//
// Created by yliu on 2017/5/24.
//

#define EXECUTABLE "srl_cmdline"
#define DESCRIPTION "The console application for Semantic Role Labelling."

#include <iostream>
#include <fstream>
#include <list>
#include "config.h"
#include "srl/SRL_DLL.h"
#include "console/dispatcher.h"
#include "boost/program_options.hpp"
#include "utils/strutils.hpp"
#include "utils/time.hpp"

using boost::program_options::options_description;
using boost::program_options::value;
using boost::program_options::variables_map;
using boost::program_options::store;
using boost::program_options::parse_command_line;
using ltp::utility::WallClockTimer;
using ltp::strutils::split;


inline int findRoot(vector<pair<int, string> > & parse, pair<int, int> edge) {
  int begin = edge.first;
  int end = edge.second;
  for (int j = begin; j <= end; ++j) {
    if (parse[j].first < begin || parse[j].first > end) {
      return j;
    }
  }
  return begin;
}

void multithreaded_srl(void *args) {

  Dispatcher * dispatcher = (Dispatcher *)args;

  while (true) {
    vector<std::string> buffer;
    int ret = dispatcher->next_block(buffer);
    if (ret < 0)
      break;
    if (!buffer.size()){
      continue;
    }

    std::vector<std::string> words;
    std::vector<std::string> postags;
    vector<pair<int, string> > parse;
    vector<pair<int, vector<pair<string, pair<int, int> > > > > vecSRLResult;
    for (int j = 0; j < buffer.size(); ++j) {
      std::stringstream S(buffer[j]);
      string str; int parent;
      S >> str; words.push_back(str);
      S >> str; postags.push_back(str);
      S >> parent; S >> str; parse.push_back(make_pair(parent, str));
    }

    srl_dosrl(words, postags, parse, vecSRLResult);

    vector<vector<string> > arg(words.size(), vector<string>(vecSRLResult.size(), "_"));
    vector<bool> is_pred(words.size(), false);
    for (int k = 0; k < vecSRLResult.size(); ++k) {
      is_pred[vecSRLResult[k].first] = true;
      for (int j = 0; j < vecSRLResult[k].second.size(); ++j) {
        arg[findRoot(parse, vecSRLResult[k].second[j].second)][k] = vecSRLResult[k].second[j].first;
      }
    }

    std::stringstream S; S.clear(); S.str("");
    for (size_t i = 0; i < words.size(); ++ i) {
      S << i << "\t" << words[i] << "\t" << postags[i] << "\t" << parse[i].first << "\t" << parse[i].second;
      S << "\t" << (is_pred[i] ? "Y" : "_");
      for (int j = 0; j < arg[i].size(); ++j) {
        S << "\t" << arg[i][j];
      }

      S << std::endl;
    }
    dispatcher->output(ret, S.str());
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
                  "Input data should contain one word each line. "
                  "Sentence should be separated by a blank line. "
                  "! Note that index start at 0, and index(ROOT)=-1 in the thrid column. "
                  "(e.g. \"中国    ns      2       ATT\").")
          ("pisrl-model", value<std::string>(),
           "The path to the pi-srl joint model [default=ltp_data/pisrl.model].")
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

  std::string srl_model = "ltp_data/pisrl.model";
  if (vm.count("pisrl-model")) {
    srl_model = vm["pisrl-model"].as<std::string>();
  }

  std::string postagger_lexcion = "";
  if (vm.count("postagger-lexicon")) {
    postagger_lexcion = vm["postagger-lexicon"].as<std::string>();
  }

  if (srl_load_resource(srl_model)) {
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

  Dispatcher * dispatcher = new Dispatcher( NULL, (*is), std::cout );
  WallClockTimer t;
  std::list<tthread::thread *> thread_list;
  for (int i = 0; i < threads; ++ i) {
    tthread::thread * t = new tthread::thread( multithreaded_srl, (void *)dispatcher );
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
  srl_release_resource();
  return 0;
}

