#include <iostream>
#include "config.h"
#include "boost/program_options.hpp"
#include "utils/logging.hpp"
#include "segmentor/segmentor_frontend.h"
#include "segmentor/customized_segmentor_frontend.h"

#define DESCRIPTION "Chinese word segmentation"
#define EXECUTABLE "otcws"

using boost::program_options::options_description;
using boost::program_options::value;
using boost::program_options::variables_map;
using boost::program_options::store;
using boost::program_options::parse_command_line;
using ltp::segmentor::SegmentorFrontend;
using ltp::segmentor::CustomizedSegmentorFrontend;

int learn(int argc, const char* argv[]) {
  std::string usage = EXECUTABLE "(learn) in LTP " LTP_VERSION " - " LTP_COPYRIGHT "\n";
  usage += "Training suite for " DESCRIPTION "\n\n";
  usage += "usage: ./" EXECUTABLE " learn <options>\n\n";
  usage += "options";

  options_description optparser = options_description(usage);
  optparser.add_options()
    ("model", value<std::string>(),
     "The prefix of the model file, model will be stored as model.$iter.")
    ("reference", value<std::string>(),  "The path to the reference file.")
    ("development", value<std::string>(), "The path to the development file.")
    ("algorithm", value<std::string>()->default_value("pa"), "The learning algorithm\n"
     " - ap: averaged perceptron\n"
     " - pa: passive aggressive [default]")
    ("max-iter", value<int>()->default_value(10), "The number of iteration [default=10].")
    ("rare-feature-threshold", value<int>()->default_value(0),
     "The threshold for rare feature, used in model truncation. [default=0]")
    ("dump-details", value<bool>()->default_value(false),
     "Save the detailed model, used in incremental training. [default=false]")
    ("help,h", "Show help information");

  if (argc == 1) { std::cerr << optparser << std::endl;  return 1; }

  variables_map vm;
  store(parse_command_line(argc, argv, optparser), vm);

  if (vm.count("help")) {
    std::cerr << optparser << std::endl;
    return 0;
  }

  std::string reference = "";
  if (!vm.count("reference")) {
    ERROR_LOG("reference file should be specified [--reference].");
    return 1;
  } else {
    reference = vm["reference"].as<std::string>();
  }

  std::string model_name = "";
  if (!vm.count("model")) {
    ERROR_LOG("model prefix should be specified [--model].");
    return 1;
  } else {
    model_name = vm["model"].as<std::string>();
  }

  std::string development = "";
  if (!vm.count("development")) {
    WARNING_LOG("development file is not configed, evaluation will not be performed.");
  } else {
    development = vm["development"].as<std::string>();
  }

  std::string algorithm = vm["algorithm"].as<std::string>();
  if (algorithm != "pa" && algorithm != "ap") {
    WARNING_LOG("algorithm should either be ap or pa, set as default [pa].");
    algorithm = "pa";
  }

  int max_iter = vm["max-iter"].as<int>();
  int rare_feature_threshold= vm["rare-feature-threshold"].as<int>();
  bool dump_model_details = dump_model_details = vm["dump-details"].as<bool>();

  SegmentorFrontend frontend(reference, development, model_name,
      algorithm, max_iter, rare_feature_threshold, dump_model_details);
  frontend.train();
  return 0;
}

int test(int argc, const char* argv[]) {
  std::string usage = EXECUTABLE "(test) in LTP " LTP_VERSION " - " LTP_COPYRIGHT "\n";
  usage += "Testing suite for " DESCRIPTION "\n\n";
  usage += "usage: ./" EXECUTABLE " test <options>\n\n";
  usage += "options";

  options_description optparser = options_description(usage);
  optparser.add_options()
    ("model", value<std::string>(), "The path to the model file.")
    ("lexicon", value<std::string>(),
     "The lexicon file, (optional, if configured, constrained decoding will be performed).")
    ("input", value<std::string>(), "The path to the reference file.")
    ("evaluate", value<bool>()->default_value(false),
     "if configured, perform evaluation, input words in sentence should be separated by space [default=false].")
    ("sequence", value<bool>()->default_value(false), "Output the probability of the label sequences")
    ("marginal", value<bool>()->default_value(false), "Output the marginal probabilities of tags")
    ("help,h", "Show help information");

  if (argc == 1) { std::cerr << optparser << std::endl;  return 1; }

  variables_map vm;
  store(parse_command_line(argc, argv, optparser), vm);

  if (vm.count("help")) {
    std::cerr << optparser << std::endl;
    return 1;
  }

  std::string model_file = "";
  if (!vm.count("model")) {
    ERROR_LOG("model prefix should be specified [--model].");
    return 1;
  } else {
    model_file = vm["model"].as<std::string>();
  }

  std::string input_file = "";
  if (!vm.count("input")) {
    ERROR_LOG("input file should be specified [--input].");
    return 1;
  } else {
    input_file = vm["input"].as<std::string>();
  }

  std::string lexicon_file = "";
  if (vm.count("lexicon")) { lexicon_file = vm["lexicon"].as<std::string>(); }

  std::string output_file = "";
  if (vm.count("output")) { output_file = vm["output"].as<std::string>(); }

  bool evaluate = vm["evaluate"].as<bool>();
  bool sequence_prob = vm["sequence"].as<bool>();
  bool marginal_prob = vm["marginal"].as<bool>();

  SegmentorFrontend frontend(input_file, model_file, evaluate, sequence_prob, marginal_prob);
  frontend.test();
  return 0;
}

int dump(int argc, const char* argv[]) {
  std::string usage = EXECUTABLE "(dump) in LTP " LTP_VERSION " - " LTP_COPYRIGHT "\n";
  usage += "Model visualization suite for " DESCRIPTION "\n\n";
  usage += "usage: ./" EXECUTABLE " dump <options>\n\n";
  usage += "options";

  options_description optparser = options_description(usage);
  optparser.add_options()
    ("model", value<std::string>(), "The path to the model file.")
    ("help,h", "Show help information");

  if (argc == 1) { std::cerr << optparser << std::endl;  return 1; }

  variables_map vm;
  store(parse_command_line(argc, argv, optparser), vm);

  if (vm.count("help")) {
    std::cerr << optparser << std::endl;
    return 1;
  }

  std::string model_name = "";
  if (!vm.count("model")) {
    ERROR_LOG("model prefix should be specified [--model].");
    return 1;
  } else {
    model_name = vm["model"].as<std::string>();
  }

  SegmentorFrontend frontend(model_name);
  frontend.dump();
  return 0;
}

int customized_learn(int argc, const char* argv[]) {
  std::string usage = EXECUTABLE "(customized-learn) in LTP " LTP_VERSION " - " LTP_COPYRIGHT "\n";
  usage += "Customized training suite for " DESCRIPTION "\n\n";
  usage += "usage: ./" EXECUTABLE " learn <options>\n\n";
  usage += "options";

  options_description optparser = options_description(usage);
  optparser.add_options()
    ("baseline-model", value<std::string>(),
    "The baseline model, which should be saved with --dump-details options.")
    ("model", value<std::string>(),
    "The prefix of the model file, model will be stored as model.$iter.")
    ("reference", value<std::string>(), "The path to the reference file.")
    ("development", value<std::string>(), "The path to the development file.")
    ("algorithm", value<std::string>()->default_value("pa"), "The learning algorithm\n"
    " - ap: averaged perceptron\n"
    " - pa: passive aggressive [default]")
    ("max-iter", value<int>()->default_value(10), "The number of iteration [default=10].")
    ("rare-feature-threshold", value<int>()->default_value(0),
    "The threshold for rare feature, used in model truncation. [default=0]")
    ("help,h", "Show help information");

  if (argc == 1) { std::cerr << optparser << std::endl;  return 1; }

  variables_map vm;
  store(parse_command_line(argc, argv, optparser), vm);

  if (vm.count("help")) {
    std::cerr << optparser << std::endl;
    return 0;
  }

  std::string reference = "";
  if (!vm.count("reference")) {
    ERROR_LOG("reference file should be specified [--reference].");
    return 1;
  } else {
    reference = vm["reference"].as<std::string>();
  }

  std::string baseline_model_file = "";
  if (!vm.count("baseline-model")) {
    ERROR_LOG("baseline model should be specified [--baseline-model].");
    return 1;
  } else {
    baseline_model_file = vm["baseline-model"].as<std::string>();
  }

  std::string model_name = "";
  if (!vm.count("model")) {
    ERROR_LOG("model prefix should be specified [--model].");
    return 1;
  } else {
    model_name = vm["model"].as<std::string>();
  }

  std::string development = "";
  if (!vm.count("development")) {
    WARNING_LOG("development file is not configed, evaluation will not be performed.");
  } else {
    development = vm["development"].as<std::string>();
  }

  std::string algorithm = vm["algorithm"].as<std::string>();
  if (algorithm != "pa" && algorithm != "ap") {
    WARNING_LOG("algorithm should either be ap or pa, set as default [pa].");
    algorithm = "pa";
  }

  int max_iter = vm["max-iter"].as<int>();
  int rare_feature_threshold = vm["rare-feature-threshold"].as<int>();

  CustomizedSegmentorFrontend frontend(reference, development, model_name,
    baseline_model_file, algorithm, max_iter, rare_feature_threshold);
  frontend.train();
  return 0;
}

int customized_test(int argc, const char* argv[]) {
  std::string usage = EXECUTABLE "(customized-test) in LTP " LTP_VERSION
    " - " LTP_COPYRIGHT "\n";
  usage += "Customized testing suite for " DESCRIPTION "\n\n";
  usage += "usage: ./" EXECUTABLE " test <options>\n\n";
  usage += "options";

  options_description optparser = options_description(usage);
  optparser.add_options()
    ("baseline-model", value<std::string>(), "The path to the baseline model.")
    ("model", value<std::string>(), "The path to the model file.")
    ("lexicon", value<std::string>(),
    "The lexicon file, (optional, if configured, constrained decoding will be performed).")
    ("input", value<std::string>(), "The path to the reference file.")
    ("evaluate", value<bool>()->default_value(false),
    "if configured, perform evaluation, input words in sentence should be separated by space [default=false].")
    ("help,h", "Show help information");

  if (argc == 1) { std::cerr << optparser << std::endl;  return 1; }

  variables_map vm;
  store(parse_command_line(argc, argv, optparser), vm);

  if (vm.count("help")) {
    std::cerr << optparser << std::endl;
    return 1;
  }

  std::string baseline_model_file = "";
  if (!vm.count("baseline-model")) {
    ERROR_LOG("baseline model should be specified [--baseline-model].");
    return 1;
  } else {
    baseline_model_file = vm["baseline-model"].as<std::string>();
  }

  std::string model_file = "";
  if (!vm.count("model")) {
    ERROR_LOG("model prefix should be specified [--model].");
    return 1;
  } else {
    model_file = vm["model"].as<std::string>();
  }

  std::string input_file = "";
  if (!vm.count("input")) {
    ERROR_LOG("input file should be specified [--input].");
    return 1;
  } else {
    input_file = vm["input"].as<std::string>();
  }

  std::string lexicon_file = "";
  if (vm.count("lexicon")) { lexicon_file = vm["lexicon"].as<std::string>(); }

  std::string output_file = "";
  if (vm.count("output")) { output_file = vm["output"].as<std::string>(); }

  bool evaluate = vm["evaluate"].as<bool>();

  CustomizedSegmentorFrontend frontend(input_file, model_file,
      baseline_model_file, evaluate);
  frontend.test();
  return 0;
}

int main(int argc, const char* argv[]) {
  std::string usage = EXECUTABLE " in LTP " LTP_VERSION " - " LTP_COPYRIGHT "\n";
  usage += "Training and testing suite for " DESCRIPTION "\n\n";
  usage += "usage: ./" EXECUTABLE;
  usage += " [learn|customized-learn|test|customized-test|dump] <options>";

  if (argc == 1) {
    std::cerr << usage << std::endl;
    return 1;
  } else if (std::string(argv[1]) == "learn") {
    return learn(argc - 1, argv + 1);
  } else if (std::string(argv[1]) == "customized-learn") {
    return customized_learn(argc - 1, argv + 1);
  } else if (std::string(argv[1]) == "test") {
    return test(argc - 1, argv + 1);
  } else if (std::string(argv[1]) == "customized-test") {
    return customized_test(argc - 1, argv + 1);
  } else if (std::string(argv[1]) == "dump") {
    return dump(argc- 1, argv+ 1);
  } else {
    std::cerr << "unknown mode: " << argv[1] << std::endl;
    std::cerr << usage << std::endl;
    return 1;
  }
  return 0;
}

