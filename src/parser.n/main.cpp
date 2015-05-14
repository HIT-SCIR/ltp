#include <iostream>
#include <cstring>
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
  std::string usage = EXECUTABLE "(test) - Testing suite for " DESCRIPTION ".\n";
  usage += "Copyright (C) 2012-2015 HIT-SCIR\n\n";
  usage += "usage: ./" EXECUTABLE " test <options>\n\n";
  usage += "options:";
  options_description optparser = options_description(usage);
  optparser.add_options()
    ("model", value<std::string>(), "The path to the model.")
    ("input", value<std::string>(), "The path to the reference.")
    ("output", value<std::string>(), "The path to the output file.");

  TestOption opt;
  NeuralNetworkParserFrontend frontend(opt);
  frontend.test();
  return 0;
}

int learn(int argc, char** argv) {
  std::string usage = EXECUTABLE "(learn) - Learning suite for " DESCRIPTION ".\n";
  usage += "Copyright (C) 2012-2015 HIT-SCIR\n\n";
  usage += "usage: ./" EXECUTABLE " learn <options>\n\n";
  usage += "options:";
  options_description optparser = options_description(usage);
  optparser.add_options()
    ("model",     value<std::string>(), "The path to the model.")
    ("embedding", value<std::string>(), "The path to the embedding file.")
    ("reference", value<std::string>(), "The path to the reference file.")
    ("development", value<std::string>(), "The path to the development file.\n")
    ("init-range", value<double>(), "The initialization range. [default=0.01]")
    ("word-cutoff", value<int>(), "The frequency of rare word. [default=1]")
    ("max-iter", value<int>(), "The number of max iteration. [default=20000]")
    ("batch-size", value<int>(), "The size of batch. [default=10000]")
    ("hidden-size", value<int>(), "The size of hidden layer. [default=200]")
    ("embedding-size", value<int>(), "The size of embedding. [default=50]")
    ("features-number", value<int>(), "The number of features. [default=48]")
    ("precomputed-number", value<int>(), "The number of precomputed. [default=100000]")
    ("evaluation-stops", value<int>(), "Evaluation on per-iteration. [default=100]")
    ("ada-eps", value<double>(), "The EPS in AdaGrad. [defautl=1e-6]")
    ("ada-alpha", value<double>(), "The Alpha in AdaGrad. [default=0.01]")
    ("lambda", value<double>(), "The regularizer parameter. [default=1e-8]")
    ("dropout-probability", value<double>(), "The probability for dropout. [default=0.5]")
    ("oracle", value<std::string>(),
     "The oracle type\n"
     " - static: The static oracle [default]\n"
     " - nondet: The non-deterministic oracle\n"
     " - explore: The explore oracle.")
    ("save-intermediate", value<bool>(), "Save the intermediate. [default=true]")
    ("fix-embeddings", value<bool>(), "Fix the embeddings. [default=false]")
    ("use-distance", value<bool>(), "Specify to use distance feature. [default=false]")
    ("use-valency", value<bool>(), "Specify to use valency feature. [default=false]")
    ("use-cluster", value<bool>(), "Specify to use cluster feature. [default=false]")
    ("cluster", value<std::string>(), "Specify the path to the cluster file.")
    ("root", value<std::string>(), "The root tag. [default=ROOT]")
    ("verbose", "Logging more details.")
    ("help,h", "Show help information.");

  variables_map vm;
  store(parse_command_line(argc, argv, optparser), vm);

  if (vm.count("help")) {
    std::cerr << optparser << std::endl;
    return 1;
  }

  LearnOption opt;

  opt.model_file = "";
  if (!vm.count("model")) {
    ERROR_LOG("parse opt: model file must be specified [--model].");
    return 1;
  } else {
    opt.model_file= vm["model"].as<std::string>();
  }

  opt.embedding_file = "";
  if (!vm.count("embedding")) {
    ERROR_LOG("parse opt: embedding file must be specified [--embedding].");
    return 1;
  } else {
    opt.embedding_file = vm["embedding"].as<std::string>();
  }

  opt.reference_file = "";
  if (!vm.count("reference")) {
    ERROR_LOG("parse opt: reference file must be specified [--reference].");
    return false;
  } else {
    opt.reference_file = vm["reference"].as<std::string>();
  }

  opt.devel_file = "";
  if (vm.count("development")) {
    opt.devel_file = vm["development"].as<std::string>();
  }

  opt.ada_eps = 1e-6;
  if (vm.count("ada-eps")) {
    opt.ada_eps = vm["ada-eps"].as<double>();
  }

  opt.ada_alpha = 0.01;
  if (vm.count("ada-alpha")) {
    opt.ada_alpha = vm["ada-alpha"].as<double>();
  }

  opt.lambda = 1e-8;
  if (vm.count("lambda")) {
    opt.lambda = vm["lambda"].as<double>();
  }

  opt.dropout_probability = 0.5;
  if (vm.count("dropout-probability")) {
    opt.dropout_probability = vm["dropout-probability"].as<double>();
  }

  opt.hidden_layer_size = 200;
  if (vm.count("hidden-size")) {
    opt.hidden_layer_size = vm["hidden-size"].as<int>();
  }

  opt.embedding_size = 50;
  if (vm.count("embedding-size")) {
    opt.embedding_size = vm["embedding-size"].as<int>();
  }

  opt.max_iter = 20000;
  if (vm.count("max-iter")) {
    opt.max_iter = vm["max-iter"].as<int>();
  }

  opt.init_range = .01;
  if (vm.count("init-range")) {
    opt.init_range = vm["init-range"].as<double>();
  }

  opt.word_cutoff = 1;
  if (vm.count("word-cutoff")) {
    opt.word_cutoff = vm["word-cutoff"].as<int>();
  }

  opt.batch_size = 10000;
  if (vm.count("batch-size")) {
    opt.batch_size = vm["batch-size"].as<int>();
  }

  opt.nr_precomputed = 100000;
  if (vm.count("precomputed-number")) {
    opt.nr_precomputed = vm["precomputed-number"].as<int>();
  }

  opt.evaluation_stops = 100;
  if (vm.count("evaluation-stops")) {
    opt.evaluation_stops = vm["evaluation-stops"].as<int>();
  }

  opt.clear_gradient_per_iter = 0;
  if (vm.count("clear-gradient-per-iter")) {
    opt.clear_gradient_per_iter = vm["clear-gradient-per-iter"].as<int>();
  }

  opt.oracle = "static";
  if (vm.count("oracle")) {
    opt.oracle = vm["oracle"].as<std::string>();
    if (opt.oracle != "static" && opt.oracle != "nondet" && opt.oracle != "explore") {
      opt.oracle = "static";
    }
  }

  opt.save_intermediate = true;
  if (vm.count("save-intermediate")) {
    opt.save_intermediate = vm["save-intermediate"].as<bool>();
  }

  opt.fix_embeddings = false;
  if (vm.count("fix-embeddings")) {
    opt.fix_embeddings = vm["fix-embeddings"].as<bool>();
  }

  opt.root = "ROOT";
  if (vm.count("root")) {
    opt.root = vm["root"].as<std::string>();
  }

  opt.use_distance = false;
  if (vm.count("use-distance")) { opt.use_distance = vm["use-distance"].as<bool>(); }

  opt.use_valency = false;
  if (vm.count("use-valency")) { opt.use_valency = vm["use-valency"].as<bool>(); }

  opt.use_cluster = false;
  if (vm.count("use-cluster")) { opt.use_cluster = vm["use-cluster"].as<bool>(); }

  opt.cluster_file = "";
  if (opt.use_cluster) {
    if (vm.count("cluster")) {
      opt.cluster_file = vm["cluster"].as<std::string>();
    } else {
      ERROR_LOG("cluster file should be specified when using cluster feature.");
    }
  }

  NeuralNetworkParserFrontend frontend(opt);
  frontend.train();
  return 0;
}

int main(int argc, char** argv) {
  std::string usage = EXECUTABLE " - Training and testing suite for " DESCRIPTION ".\n";
  usage += "Copyright (C) 2012-2015 HIT-SCIR\n\n";
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
