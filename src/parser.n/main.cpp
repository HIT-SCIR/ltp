#include <iostream>
#include <cstring>
#include "config.h"
#include "utils/logging.hpp"
#include "boost/program_options.hpp"
#include "parser.n/options.h"
#include "parser.n/parser_frontend.h"

#define DESCRIPTION "Neural Network Parser"
#define EXECUTABLE "nndepparser"

using boost::program_options::options_description;
using boost::program_options::value;
using boost::program_options::variables_map;
using boost::program_options::store;
using boost::program_options::parse_command_line;
using ltp::depparser::NeuralNetworkParserFrontend;
using ltp::depparser::LearnOption;
using ltp::depparser::TestOption;

int test(int argc, char** argv) {
  std::string usage = EXECUTABLE "(test) in LTP " LTP_VERSION " - " LTP_COPYRIGHT "\n";
  usage += "Testing suite for " DESCRIPTION "\n\n";
  usage += "usage: ./" EXECUTABLE " test <options>\n\n";
  usage += "options:";
  TestOption opt;
  options_description optparser(usage);
  optparser.add_options()
    ("model", value<std::string>(&opt.model_file)->required(), "The path to the model.")
    ("input", value<std::string>(&opt.input_file)->required(), "The path to the reference.")
    ("evaluate", value<bool>(&opt.evaluate)->default_value(false),
     "if configured, perform evaluation, heads and deprels columns should be filled.")
    ("help,h", "Show help information");

  if (argc == 1) { std::cerr << optparser << std::endl; return 1; }

  variables_map vm;
  try {
      store(parse_command_line(argc, argv, optparser), vm);
      if (vm.count("help")) {
        std::cerr << optparser << std::endl;
        return 1;
      }

      boost::program_options::notify(vm);
  } catch(const boost::program_options::error &e) {
      std::cerr << e.what() << std::endl;
      std::cout << optparser << std::endl;
  }

  NeuralNetworkParserFrontend frontend(opt);
  frontend.test();
  return 0;
}

int learn(int argc, char** argv) {
  std::string usage = EXECUTABLE "(learn) in LTP " LTP_VERSION " - " LTP_COPYRIGHT "\n";
  usage += "Training suite for " DESCRIPTION "\n";
  usage += "usage: ./" EXECUTABLE " learn <options>\n\n";
  usage += "options:";
  LearnOption opt;
  options_description optparser(usage);
  optparser.add_options()
    ("model",     value<std::string>(&opt.model_file)->required(), "The path to the model.")
    ("embedding", value<std::string>(&opt.embedding_file)->required(), "The path to the embedding file.")
    ("reference", value<std::string>(&opt.reference_file)->required(), "The path to the reference file.")
    ("development", value<std::string>(&opt.devel_file), "The path to the development file.\n")
    ("init-range", value<double>(&opt.init_range)->default_value(0.01), "The initialization range. [default=0.01]")
    ("word-cutoff", value<int>(&opt.word_cutoff)->default_value(1), "The frequency of rare word. Word with frequency less than cutoff will be considered unknown")
    ("max-iter", value<int>(&opt.max_iter)->default_value(20000), "The number of max iteration. [default=20000]")
    ("batch-size", value<int>(&opt.batch_size)->default_value(10000), "The size of batch. [default=10000]")
    ("hidden-size", value<int>(&opt.hidden_layer_size)->default_value(200), "The size of hidden layer. [default=200]")
    ("embedding-size", value<int>(&opt.embedding_size)->default_value(50), "The size of embedding. [default=50]")
    //("features-number", value<int>(), "The number of features. [default=48]")
    ("precomputed-number", value<int>(&opt.nr_precomputed)->default_value(100000), "The number of precomputed. [default=100000]")
    ("evaluation-stops", value<int>(&opt.evaluation_stops)->default_value(100), "Evaluation on per-iteration. [default=100]")
    ("ada-eps", value<double>(&opt.ada_eps)->default_value(1e-6), "The EPS in AdaGrad. [defautl=1e-6]")
    ("ada-alpha", value<double>(&opt.ada_alpha)->default_value(0.01), "The Alpha in AdaGrad. [default=0.01]")
    ("lambda", value<double>(&opt.lambda)->default_value(1e-8), "The regularizer parameter. [default=1e-8]")
    ("dropout-probability", value<double>(&opt.dropout_probability)->default_value(0.5), "The probability for dropout. [default=0.5]")
    ("oracle", value<std::string>(&opt.oracle)->default_value("static"),
     "The oracle type\n"
     " - static: The static oracle [default]\n"
     " - nondet: The non-deterministic oracle\n"
     " - explore: The explore oracle.")
    ("save-intermediate", value<bool>(&opt.save_intermediate)->default_value(true), "Save the intermediate. [default=true]")
    ("fix-embeddings", value<bool>(&opt.fix_embeddings)->default_value(false), "Fix the embeddings. [default=false]")
    ("use-distance", value<bool>(&opt.use_distance)->default_value(false), "Specify to use distance feature. [default=false]")
    ("use-valency", value<bool>(&opt.use_valency)->default_value(false), "Specify to use valency feature. [default=false]")
    ("use-cluster", value<bool>(&opt.use_cluster)->default_value(false), "Specify to use cluster feature. [default=false]")
    ("cluster", value<std::string>(&opt.cluster_file), "Specify the path to the cluster file.")
    ("root", value<std::string>(&opt.root)->default_value("ROOT"), "The root tag. [default=ROOT]")
    ("verbose", "Logging more details.")
    ("help,h", "Show help information.");

  if (argc == 1) { std::cerr << optparser << std::endl; return 1; }

  variables_map vm;
  try {
      store(parse_command_line(argc, argv, optparser), vm);
      if (vm.count("help")) {
        std::cerr << optparser << std::endl;
        return 1;
      }

      boost::program_options::notify(vm);
  } catch (const boost::program_options::error &e) {
      std::cerr << e.what() << std::endl;
      std::cout << optparser << std::endl;
  }

//  opt.clear_gradient_per_iter = 0;
//  if (vm.count("clear-gradient-per-iter")) {
//    opt.clear_gradient_per_iter = vm["clear-gradient-per-iter"].as<int>();
//  }

  //some extra validate of opt
  if (opt.oracle != "static" && opt.oracle != "nondet" && opt.oracle != "explore") {
    ERROR_LOG("parse opt [--oracle]: oracle value error, allowed values are static|nondet|explore.");
    opt.oracle = "static";
  }

  if (opt.use_cluster && vm.count("cluster") == 0) {
      ERROR_LOG("parse opt [--cluster]: cluster file should be specified when using cluster feature.");
      return 1;
  }

  NeuralNetworkParserFrontend frontend(opt);
  frontend.train();

  return 0;
}

int main(int argc, char** argv) {
  std::string usage = EXECUTABLE " in LTP " LTP_VERSION " - " LTP_COPYRIGHT "\n";
  usage += "Training and testing suite for " DESCRIPTION "\n\n";
  usage += "usage: ./" EXECUTABLE " [learn|test] <options>";

  if (argc == 1) {
    std::cerr << usage << std::endl;
    return 1;
  } else if (strcmp(argv[1], "learn") == 0) {
    return learn(argc- 1, argv+ 1);
  } else if (strcmp(argv[1], "test") == 0) {
    return test(argc- 1, argv+ 1);
  } else {
    std::cerr << "unknown mode: " << argv[1] << std::endl;
    std::cerr << usage << std::endl;
    return 1;
  }
  return 0;
}
