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
  std::string usage = EXECUTABLE "(test) in LTP " LTP_VERSION " - (C) 2012-2015 HIT-SCIR\n";
  usage += "Testing suite for " DESCRIPTION "\n\n";
  usage += "usage: ./" EXECUTABLE " test <options>\n\n";
  usage += "options:";
  options_description optparser = options_description(usage);
  optparser.add_options()
    ("model", value<std::string>(), "The path to the model.")
    ("input", value<std::string>(), "The path to the reference.")
    ("evaluate", value<bool>()->default_value(false),
     "if configured, perform evaluation, heads and deprels columns should be filled.")
    ("help,h", "Show help information");

  if (argc == 1) { std::cerr << optparser << std::endl; return 1; }

  variables_map vm;
  store(parse_command_line(argc, argv, optparser), vm);

  if (vm.count("help")) {
    std::cerr << optparser << std::endl;
    return 1;
  }

  TestOption opt;
  opt.model_file = "";
  if (!vm.count("model")) {
    ERROR_LOG("model path should be specified [--model].");
    return 1;
  } else {
    opt.model_file = vm["model"].as<std::string>();
  }

  opt.input_file = "";
  if (!vm.count("input")) {
    ERROR_LOG("input file should be specified [--input].");
    return 1;
  } else {
    opt.input_file = vm["input"].as<std::string>();
  }

  opt.evaluate = vm["evaluate"].as<bool>();

  NeuralNetworkParserFrontend frontend(opt);
  frontend.test();
  return 0;
}

int learn(int argc, char** argv) {
  std::string usage = EXECUTABLE "(learn) in LTP " LTP_VERSION " - (C) 2012-2015 HIT-SCIR\n";
  usage += "Training suite for " DESCRIPTION "\n";
  usage += "usage: ./" EXECUTABLE " learn <options>\n\n";
  usage += "options:";
  options_description optparser = options_description(usage);
  optparser.add_options()
    ("model",     value<std::string>(), "The path to the model.")
    ("embedding", value<std::string>(), "The path to the embedding file.")
    ("reference", value<std::string>(), "The path to the reference file.")
    ("development", value<std::string>(), "The path to the development file.\n")
    ("init-range", value<double>()->default_value(0.01), "The initialization range. [default=0.01]")
    ("word-cutoff", value<int>()->default_value(1), "The frequency of rare word. [default=1]")
    ("max-iter", value<int>()->default_value(20000), "The number of max iteration. [default=20000]")
    ("batch-size", value<int>()->default_value(10000), "The size of batch. [default=10000]")
    ("hidden-size", value<int>()->default_value(200), "The size of hidden layer. [default=200]")
    ("embedding-size", value<int>()->default_value(50), "The size of embedding. [default=50]")
    ("features-number", value<int>()->default_value(48), "The number of features. [default=48]")
    ("precomputed-number", value<int>()->default_value(100000), "The number of precomputed. [default=100000]")
    ("evaluation-stops", value<int>()->default_value(100), "Evaluation on per-iteration. [default=100]")
    ("ada-eps", value<double>()->default_value(1e-6), "The EPS in AdaGrad. [defautl=1e-6]")
    ("ada-alpha", value<double>()->default_value(0.01), "The Alpha in AdaGrad. [default=0.01]")
    ("lambda", value<double>()->default_value(1e-8), "The regularizer parameter. [default=1e-8]")
    ("dropout-probability", value<double>()->default_value(0.5), "The probability for dropout. [default=0.5]")
    ("oracle", value<std::string>()->default_value("static"),
     "The oracle type\n"
     " - static: The static oracle [default]\n"
     " - nondet: The non-deterministic oracle\n"
     " - explore: The explore oracle.")
    ("save-intermediate", value<bool>()->default_value(true), "Save the intermediate. [default=true]")
    ("fix-embeddings", value<bool>()->default_value(false), "Fix the embeddings. [default=false]")
    ("use-distance", value<bool>()->default_value(false), "Specify to use distance feature. [default=false]")
    ("use-valency", value<bool>()->default_value(false), "Specify to use valency feature. [default=false]")
    ("use-cluster", value<bool>()->default_value(false), "Specify to use cluster feature. [default=false]")
    ("cluster", value<std::string>(), "Specify the path to the cluster file.")
    ("root", value<std::string>()->default_value("ROOT"), "The root tag. [default=ROOT]")
    ("verbose", "Logging more details.")
    ("help,h", "Show help information.");

  if (argc == 1) { std::cerr << optparser << std::endl; return 1; }

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

  opt.ada_eps = vm["ada-eps"].as<double>();
  opt.ada_alpha = vm["ada-alpha"].as<double>();
  opt.lambda = vm["lambda"].as<double>();
  opt.dropout_probability = vm["dropout-probability"].as<double>();
  opt.hidden_layer_size = vm["hidden-size"].as<int>();
  opt.embedding_size = vm["embedding-size"].as<int>();
  opt.max_iter = vm["max-iter"].as<int>();
  opt.init_range = vm["init-range"].as<double>();
  opt.word_cutoff = vm["word-cutoff"].as<int>();
  opt.batch_size = vm["batch-size"].as<int>();
  opt.nr_precomputed = vm["precomputed-number"].as<int>();
  opt.evaluation_stops = vm["evaluation-stops"].as<int>();
  opt.clear_gradient_per_iter = 0;
  opt.oracle = vm["oracle"].as<std::string>();
  if (opt.oracle != "static" && opt.oracle != "nondet" && opt.oracle != "explore") {
    opt.oracle = "static";
  }

  opt.save_intermediate = vm["save-intermediate"].as<bool>();
  opt.fix_embeddings = vm["fix-embeddings"].as<bool>();
  opt.root = vm["root"].as<std::string>();
  opt.use_distance = vm["use-distance"].as<bool>();
  opt.use_valency = vm["use-valency"].as<bool>();
  opt.use_cluster = vm["use-cluster"].as<bool>();

  opt.cluster_file = "";
  if (opt.use_cluster) {
    if (vm.count("cluster")) {
      opt.cluster_file = vm["cluster"].as<std::string>();
    } else {
      ERROR_LOG("cluster file should be specified when using cluster feature.");
      return 1;
    }
  }

  NeuralNetworkParserFrontend frontend(opt);
  frontend.train();
  return 0;
}

int main(int argc, char** argv) {
  std::string usage = EXECUTABLE " in LTP " LTP_VERSION " - (C) 2012-2015 HIT-SCIR\n";
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
