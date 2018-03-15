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
#include "json/json.h"

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
  std::string usage = EXECUTABLE " in LTP " LTP_VERSION " - " LTP_COPYRIGHT "\n";
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
     "- " LTP_SERVICE_NAME_SRL ": Semantic role labeling\n"
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
    ("srl-model", value<std::string>(),
     "The path to the srl model [default=ltp_data/pisrl.model].")
    ("log-level", value<int>(), "The log level:\n"
     "- 0: TRACE level\n"
     "- 1: DEBUG level\n"
     "- 2: INFO level [default]\n")
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
  //INFO_LOG("parser model after vm :\"%s\"", parser_model.c_str());
  std::string srl_model= "ltp_data/pisrl.model";
  if (vm.count("srl-model")) {
    srl_model = vm["srl-model"].as<std::string>();
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
      postagger_lexcion, ner_model, parser_model, srl_model);

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

  INFO_LOG("Start listening on port [%s]...", port_str.c_str());


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

static std::string xml2jsonstr(const XML4NLP & xml, std::string str_type) {
  Json::Value root;

  int paragraphNum = xml.CountParagraphInDocument();

  for (int pid = 0; pid < paragraphNum; ++ pid) {
    Json::Value paragraph;

    int stnsNum = xml.CountSentenceInParagraph(pid);
    for (int sid = 0; sid < stnsNum; ++sid) {
      Json::Value sentence;

      std::vector<std::string> vecWord;
      std::vector<std::string> vecPOS;
      std::vector<std::string> vecNETag;
      std::vector<std::pair<int, std::string>> vecParse;
      //std::vector<std::vector<std::string>> vecSemResult;
      std::vector<std::vector<std::pair<int, std::string>>> vecSemResult;
      std::vector<std::pair<int, std::vector<std::pair<const char *, std::pair< int, int > > > > > vecSRLResult;

      // seg
      xml.GetWordsFromSentence(vecWord, pid, sid);

      // postag
      if (str_type == LTP_SERVICE_NAME_POSTAG
          || str_type == LTP_SERVICE_NAME_NER
          || str_type == LTP_SERVICE_NAME_DEPPARSE
          || str_type == LTP_SERVICE_NAME_SRL
          || str_type == LTP_SERVICE_NAME_ALL) {
        xml.GetPOSsFromSentence(vecPOS, pid, sid);
      }

      // ner
      if (str_type == LTP_SERVICE_NAME_NER
          || str_type == LTP_SERVICE_NAME_SRL
          || str_type == LTP_SERVICE_NAME_ALL) {
        xml.GetNEsFromSentence(vecNETag, pid, sid);
      }

      // dp
      if (str_type == LTP_SERVICE_NAME_DEPPARSE
          || str_type == LTP_SERVICE_NAME_SRL
          || str_type == LTP_SERVICE_NAME_ALL) {
        xml.GetParsesFromSentence(vecParse, pid, sid);
      }

      // srl
      if (str_type == LTP_SERVICE_NAME_SRL
          || str_type == LTP_SERVICE_NAME_ALL) {
        // get by word
      }

      for (int wid = 0; wid < vecWord.size(); ++wid) {
        Json::Value word;
        word["id"] = wid;
        word["cont"] = vecWord[wid];

        // postag
        if (str_type == LTP_SERVICE_NAME_POSTAG
            || str_type == LTP_SERVICE_NAME_NER
            || str_type == LTP_SERVICE_NAME_DEPPARSE
            || str_type == LTP_SERVICE_NAME_SRL
            || str_type == LTP_SERVICE_NAME_ALL) {
          word["pos"] = vecPOS[wid];

        }

        // ner
        if (str_type == LTP_SERVICE_NAME_NER
            || str_type == LTP_SERVICE_NAME_SRL
            || str_type == LTP_SERVICE_NAME_ALL) {
          word["ne"] = vecNETag[wid];
        }

        // dp
        if (str_type == LTP_SERVICE_NAME_DEPPARSE
            || str_type == LTP_SERVICE_NAME_SRL
            || str_type == LTP_SERVICE_NAME_ALL) {
          word["parent"] = vecParse[wid].first;
          word["relate"] = vecParse[wid].second;
        }

        // srl
        if (str_type == LTP_SERVICE_NAME_SRL
            || str_type == LTP_SERVICE_NAME_ALL) {
          Json::Value args;
          std::vector<std::string> vecType;
          std::vector<std::pair<int, int>> vecBegEnd;
          xml.GetPredArgToWord(pid, sid, wid, vecType, vecBegEnd);
          if (vecType.size() != 0) {
            for (int arg_id = 0; arg_id < vecType.size(); ++arg_id) {
              Json::Value arg;
              arg["id"] = arg_id;
              arg["type"] = vecType[arg_id];
              arg["beg"] = vecBegEnd[arg_id].first;
              arg["end"] = vecBegEnd[arg_id].second;
              args.append(arg);
            }
          } else {
            args.resize(0);
          }
          word["arg"] = args;
        }

        sentence.append(word);
      }

      paragraph.append(sentence);
    } // sentence
    root.append(paragraph);
  } // paragraph
  return root.toStyledString();
}

static int Service(struct mg_connection *conn) {
  char *sentence;
  char type[10];
  char xml[10];
  char format[10];

  std::string str_post_data;
  std::string str_type;
  std::string str_xml;
  std::string str_format;

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

    mg_get_var(str_post_data.c_str(),
               str_post_data.size(),
               "f",
               format,
               sizeof(format) - 1);

    string strSentence = sentence;

    // validation check
    if (strlen(sentence) == 0) {
      WARNING_LOG("Input sentence is empty");
      ErrorResponse(conn, kEmptyStringError);
      delete[] sentence;
      return 0;
    }

    if (!isclear(strSentence)) {
      WARNING_LOG("Failed string validation check");
      ErrorResponse(conn, kEncodingError);
      delete[] sentence;
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

    if(strlen(format) == 0) {
      str_format = "";
    } else {
      str_format = format;
    }

    DEBUG_LOG("Input sentence is: %s", strSentence.c_str());

    //Get a XML4NLP instance here.
    XML4NLP  xml4nlp;

    if(str_xml == "y") {
      if (-1 == xml4nlp.LoadXMLFromString(strSentence)) {
        ErrorResponse(conn, kXmlParseError);
        delete[] sentence;
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
        delete[] sentence;
        return 0;
      }
    } else if (str_type == LTP_SERVICE_NAME_POSTAG){
      int ret = engine->postag(xml4nlp);
      if (0 != ret) {
        ErrorResponse(conn, static_cast<ErrorCodes>(ret));
        delete[] sentence;
        return 0;
      }
    } else if (str_type == LTP_SERVICE_NAME_NER) {
      int ret = engine->ner(xml4nlp);
      if (0 != ret) {
        ErrorResponse(conn, static_cast<ErrorCodes>(ret));
        delete[] sentence;
        return 0;
      }
    } else if (str_type == LTP_SERVICE_NAME_DEPPARSE){
      int ret = engine->parser(xml4nlp);
      if (0 != ret) {
        ErrorResponse(conn, static_cast<ErrorCodes>(ret));
        delete[] sentence;
        return 0;
      }
    } else if (str_type == LTP_SERVICE_NAME_SRL){ // srl
      int ret = engine->srl(xml4nlp);
      if (0 != ret) {
        ErrorResponse(conn, static_cast<ErrorCodes>(ret));
        delete[] sentence;
        return 0;
      }
    } else {   // all
      str_type = LTP_SERVICE_NAME_ALL;
      int ret = engine->srl(xml4nlp);
      if (0 != ret) {
        ErrorResponse(conn, static_cast<ErrorCodes>(ret));
        delete[] sentence;
        return 0;
      }
    }

    TRACE_LOG("Analysis is done.");

    std::string strResult;
    if (str_format == "xml") { //xml
      xml4nlp.SaveDOM(strResult);
    } else if (str_format == "json") { //json
      strResult = xml2jsonstr(xml4nlp, str_type);
    } else {  // if str_format not set, or is invalid, use xml
      xml4nlp.SaveDOM(strResult);
    }


    strResult = "HTTP/1.1 200 OK\r\nContent-Length: " \
      + std::to_string(strResult.length()) + "\r\n\r\n" + strResult;
    // TRACE_LOG(strResult.c_str());
    mg_printf(conn, "%s", strResult.c_str());

    xml4nlp.ClearDOM();
    delete[] sentence;
  }
  return 1;
}

