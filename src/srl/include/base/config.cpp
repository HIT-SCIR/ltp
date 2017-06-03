//
// Created by liu on 2016/12/19.
//

#include "config.h"
#include "iostream"
#include <string>
#include <boost/program_options.hpp>
#include <fstream>

using namespace std;

const boost::program_options::variables_map &base::config::getConf() const {
  return conf;
}

string base::config::toString(string outerSep, string innerSep) {
  map<string, pair<Type, void*> >::iterator iter;
  std::ostringstream stream;
  for (iter = confMap.begin(); iter != confMap.end(); iter++) {
    stream << outerSep << iter->first << innerSep;
    switch (iter->second.first) {
      case (INT): stream << *((int*)(iter->second.second)); break;
      case (UNSIGNED): stream << *((unsigned*)(iter->second.second)); break;
      case (FLOAT): stream << *((float*)(iter->second.second)); break;
      case (STRING): stream << *((string*)(iter->second.second)); break;
      case (BOOL): stream << *((bool*)(iter->second.second)); break;
    }

  }
  return stream.str();
}


void base::config::init(int argc, char **argv)  {
  /* 当传入 @configFile 时要正确解析 */
  /* 覆盖策略：
   *   前面定义的覆盖后面的 // 命令行必须在 配置文件之前
   *   --a b @config(其中a=c) // a最终为b
   *   @config(其中a=b) @config(其中a=c) // a最终为b
   * */
  try {

    if (argc >=2) {
      if (argv[1][0] == '@' && argc == 2) { // @config
        init(argv[1] + 1);
        return;
      } else if (argv[1][0] == '@') { // @config1 @config2
        ifstream f(argv[1] + 1); if (!f) { cerr << "config file '" << (char *)(argv[1] + 1) << "' not found!"; exit(1);}
        po::store(po::parse_config_file(f,optionDescription), conf);
        f.close();
        init(argc - 1, argv + 1);
        return;
      } else {
        int firstConfigFile = 1;
        for (; firstConfigFile < argc && argv[firstConfigFile][0] != '@' ; firstConfigFile += 2);
        if (firstConfigFile >= argc) {// --a b
          po::store(po::parse_command_line(argc, (const char *const *) argv, optionDescription), conf);
        } else { // --a b @config
          po::store(po::parse_command_line(firstConfigFile, (const char *const *) argv, optionDescription), conf);
          init(argc - firstConfigFile + 1, argv + firstConfigFile - 1);
          return;
        }
      }
    } else {
      po::store(po::parse_command_line(argc, (const char *const *) argv, optionDescription), conf);
    }
    if (conf.count("help")) {
      cerr << optionDescription << endl;
      exit(1);
    }
    po::notify(conf);
    extractBool();
  } catch ( boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::program_options::required_option> > & e) {
    cerr << endl  << "[error] " << e.get_option_name() << " must be set!" << endl << endl;
    cerr << optionDescription << endl;
    exit(1);
  } catch (boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::program_options::unknown_option> > & e) {
    cerr << endl << "[error]  unrecognised option '" << e.get_option_name() << "'." << endl << endl;
    cerr << optionDescription << endl;
    exit(1);
  }

}

void base::config::init(string configFile) {
  try {
    ifstream f(configFile);
    if (!f) { cerr << "config file '" << configFile << "' not found!"; exit(1);}
    po::store(po::parse_config_file(f,optionDescription), conf);
    if (conf.count("help")) {
      cerr << optionDescription << endl;
      exit(1);
    }
    po::notify(conf);
    extractBool();
    f.close();
  } catch ( boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::program_options::required_option> > & e) {
    cerr << endl  << "[error] " << e.get_option_name() << " must be set!" << endl << endl;
    cerr << optionDescription << endl;
    exit(1);
  } catch (boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::program_options::unknown_option> > & e) {
    cerr << endl << "[error]  unrecognised option '" << e.get_option_name() << "'." << endl << endl;
    cerr << optionDescription << endl;
    exit(1);
  }
}

void base::config::extractBool() {
  map<string, pair<Type, void*> >::iterator iter;
  for (iter = confMap.begin(); iter != confMap.end(); iter++) {
    if (iter->second.first == BOOL) {
      *(bool*)(iter->second.second) = (bool) conf.count(iter->first);
    }
  }
}
