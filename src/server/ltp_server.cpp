// Defines the entry point for the Web Service application.
#include <sys/wait.h>
#include <unistd.h>       /* For pause() */
#include <stdlib.h>
#include <signal.h>
#include <iostream>
#include "config.h"
#include "ltp/Ltp.h"
#include "server/mongoose.h"
#include "boost/program_options.hpp"
#include "utils/strutils.hpp"
#include "utils/logging.hpp"
#include "utils/codecs.hpp"

#define POST_LEN 1024
#define EXECUTABLE "ltp_server"
#define DESCRIPTION "The HTTP server frontend for Language Technology Platform."

using boost::program_options::options_description;
using boost::program_options::value;
using boost::program_options::variables_map;
using boost::program_options::store;
using boost::program_options::parse_command_line;
using ltp::strutils::to_str;
using ltp::strutils::codecs::isclear;

static LTP * engine = NULL;
static int exit_flag;
static int Service(struct mg_connection* conn);
static void ErrorResponse(struct mg_connection* conn,
    const enum ErrorCodes& code);

static void signal_handler(int sig_num) {
  exit_flag = sig_num;
}

int main(int argc, char *argv[]) {
  std::string usage = EXECUTABLE " in LTP " LTP_VERSION " - (C) 2012-2015 HIT-SCIR\n";
  usage += DESCRIPTION "\n\n";
  usage += "usage: ./" EXECUTABLE " <options>\n\n";
  usage += "options";

  options_description optparser = options_description(usage);
  optparser.add_options()
    ("port", value<int>(), "The port number [default=12345].")
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
     "- all: The whole pipeline [default]")
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
    ("srl-data", value<std::string>(),
     "The path to the SRL model directory [default=ltp_data/srl_data/].")
    ("log-level", value<int>(), "The log level:\n"
     "- 0: TRACE level\n"
     "- 1: DEBUG level\n"
     "- 2: INFO level\n")
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

  int port = 12345;
  if (vm.count("port")) {
    port = vm["port"].as<int>();
    if (port < 80 || port > 65535) {
      ERROR_LOG("port number %d is not in legal range (80 .. 65535)");
      return 1;
    }
  }

  int threads = 1;
  if (vm.count("threads")) {
    threads = vm["threads"].as<int>();
    if (threads < 0) {
      WARNING_LOG("number of threads should not less than 0, reset to 1.");
      threads = 1;
    }
  }

  std::string last_stage = "all";
  if (vm.count("last-stage")) {
    last_stage = vm["last-stage"].as<std::string>();
    if (last_stage != LTP_SERVICE_NAME_SEGMENT
        && last_stage != LTP_SERVICE_NAME_POSTAG
        && last_stage != LTP_SERVICE_NAME_NER
        && last_stage != LTP_SERVICE_NAME_DEPPARSE
        && last_stage != LTP_SERVICE_NAME_SRL
        && last_stage != "all") {
      std::cerr << "Unknown stage name:" << last_stage << ", reset to 'all'" << std::endl;
      last_stage = "all";
    }
  }

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

  std::string srl_data= "ltp_data/srl/";
  if (vm.count("srl-data")) {
    srl_data = vm["srl-data"].as<std::string>();
  }

  int log_level = LTP_LOG_INFO;
  if (vm.count("log-level")) {
    log_level = vm["log-level"].as<int>();
    if (log_level == 0) {
      ltp::utility::Logger<void>::get_logger()->set_lvl(LTP_LOG_TRACE);
    } else if (log_level == 1) {
      ltp::utility::Logger<void>::get_logger()->set_lvl(LTP_LOG_DEBUG);
    }
  }

  engine = new LTP(last_stage, segmentor_model, segmentor_lexicon, postagger_model,
      postagger_lexcion, ner_model, parser_model, srl_data);

  if (!engine->loaded()) {
    ERROR_LOG("Failed to setup LTP engine.");
    return 1;
  }

  std::string port_str = to_str(port);
  std::string threads_str = to_str(threads);

  signal(SIGTERM, signal_handler);
  signal(SIGINT, signal_handler);
  struct mg_context *ctx;
  const char *options[] = {"listening_ports", port_str.c_str(),
    "num_threads", threads_str.c_str(), NULL};
  struct mg_callbacks callbacks;

  memset(&callbacks, 0, sizeof(callbacks));
  callbacks.begin_request = Service;

  if ((ctx = mg_start(&callbacks, NULL, options)) == NULL) {
    ERROR_LOG("Cannot initialize Mongoose context");
    INFO_LOG("please check your network configuration.");
    exit(EXIT_FAILURE);
  }

  // getchar();
  while (exit_flag == 0) {
    sleep(100000);
  }
  mg_stop(ctx);

  return 0;
}

static void ErrorResponse(struct mg_connection* conn,
                          const enum ErrorCodes& code) {
  switch (code) {
    case kEmptyStringError:
      {
        std::string response = "HTTP/1.1 400 EMPTY SENTENCE\r\n\r\n";
        mg_printf(conn, "%s", response.c_str());
        break;
      }
    case kEncodingError:
      {
        // Input sentence is not clear
        std::string response = "HTTP/1.1 400 ENCODING NOT IN UTF8\r\n\r\n";
        mg_printf(conn, "%s", response.c_str());
        break;
      }
    case kXmlParseError :
      {
        // Failed the xml validation check
        std::string response = "HTTP/1.1 400 BAD XML FORMAT\r\n\r\n";
        response += "Failed to load custom xml";
        mg_printf(conn, "%s", response.c_str());
        break;
      }
    case kSentenceTooLongError:
      {
        std::string response = "HTTP/1.1 400 SENTENCE TOO LONG\r\n\r\n";
        response += "Input sentence exceed 300 characters or 70 words";
        mg_printf(conn, "%s", response.c_str());
        break;
      }
    case kSplitSentenceError:
    case kWordsegError:
    case kPostagError:
    case kParserError:
    case kNERError:
    case kSRLError:
      {
        std::string response = "HTTP/1.1 400 ANALYSIS FAILED\r\n\r\n";
        mg_printf(conn, "%s", response.c_str());
        break;
      }
    default:
      {
        ERROR_LOG("Non-sence error catch");
        break;
      }
  }
}

static int Service(struct mg_connection *conn) {
  char *sentence;
  char type[10];
  char xml[10];

  std::string str_post_data;
  std::string str_type;
  std::string str_xml;

  const struct mg_request_info *ri = mg_get_request_info(conn);

  if (!strcmp(ri->uri, "/ltp")) {
    int len;
    char buffer[POST_LEN];

    while((len = mg_read(conn, buffer, sizeof(buffer) - 1)) > 0){
      buffer[len] = 0;
      str_post_data += buffer;
    }

    DEBUG_LOG("CDATA: %s (length=%d)", str_post_data.c_str(), str_post_data.size());
    if (str_post_data.size() == 0) {
      WARNING_LOG("Input request is empty");
      ErrorResponse(conn, kEmptyStringError);
      return 0;
    }

    sentence = new char[str_post_data.size() + 1];
    mg_get_var(str_post_data.c_str(),
               str_post_data.size(),
               "s",
               sentence,
               str_post_data.size());

    mg_get_var(str_post_data.c_str(),
               str_post_data.size(),
               "t",
               type,
               sizeof(type) - 1);

    mg_get_var(str_post_data.c_str(),
               str_post_data.size(),
               "x",
               xml,
               sizeof(xml) - 1);

    string strSentence = sentence;

    // validation check
    if (strlen(sentence) == 0) {
      WARNING_LOG("Input sentence is empty");
      ErrorResponse(conn, kEmptyStringError);
      return 0;
    }

    if (!isclear(strSentence)) {
      WARNING_LOG("Failed string validation check");
      ErrorResponse(conn, kEncodingError);
      return 0;
    }

    if(strlen(type) == 0) {
      str_type = "";
    } else {
      str_type = type;
    }

    if(strlen(xml) == 0) {
      str_xml = "";
    } else {
      str_xml = xml;
    }

    delete []sentence;
    DEBUG_LOG("Input sentence is: %s", strSentence.c_str());

    //Get a XML4NLP instance here.
    XML4NLP  xml4nlp;

    if(str_xml == "y") {
      if (-1 == xml4nlp.LoadXMLFromString(strSentence)) {
        ErrorResponse(conn, kXmlParseError);
        return 0;
      }
      // move sentence validation check into each module
    } else {
      xml4nlp.CreateDOMFromString(strSentence);
    }

    if (str_type == LTP_SERVICE_NAME_SEGMENT){
      int ret = engine->wordseg(xml4nlp);
      if (0 != ret) {
        ErrorResponse(conn, static_cast<ErrorCodes>(ret));
        return 0;
      }
    } else if (str_type == LTP_SERVICE_NAME_POSTAG){
      int ret = engine->postag(xml4nlp);
      if (0 != ret) {
        ErrorResponse(conn, static_cast<ErrorCodes>(ret));
        return 0;
      }
    } else if (str_type == LTP_SERVICE_NAME_NER) {
      int ret = engine->ner(xml4nlp);
      if (0 != ret) {
        ErrorResponse(conn, static_cast<ErrorCodes>(ret));
        return 0;
      }
    } else if (str_type == LTP_SERVICE_NAME_DEPPARSE){
      int ret = engine->parser(xml4nlp);
      if (0 != ret) {
        ErrorResponse(conn, static_cast<ErrorCodes>(ret));
        return 0;
      }
    } else { // srl or all
      int ret = engine->srl(xml4nlp);
      if (0 != ret) {
        ErrorResponse(conn, static_cast<ErrorCodes>(ret));
        return 0;
      }
    }

    TRACE_LOG("Analysis is done.");

    std::string strResult;
    xml4nlp.SaveDOM(strResult);

    strResult = "HTTP/1.1 200 OK\r\n\r\n" + strResult;
    mg_printf(conn, "%s", strResult.c_str());

    xml4nlp.ClearDOM();
  }
  return 1;
}

