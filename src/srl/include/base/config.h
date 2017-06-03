//
// Created by liu on 2016/12/19.
//

#ifndef PROJECT_CONFIG_H
#define PROJECT_CONFIG_H

#include <boost/program_options.hpp>
#include <boost/any.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <map>
using namespace std;
using boost::any_cast;

namespace po = boost::program_options;
namespace base {

  class config {
  public:
    enum Type{
      INT,
      UNSIGNED,
      FLOAT,
      STRING,
      BOOL,
    };
    po::variables_map conf;
    map<string, pair<Type, void*> > confMap;
    po::options_description optionDescription;
    po::options_description_easy_init addOpt;

  public:
    config(string confName = "Configuration"):
            optionDescription(confName),
            addOpt(optionDescription.add_options())
    {
      addOpt = addOpt("help,h", "Help");
    };

    string toString(string outerSep = "_", string innerSep = "_");

    template <class T>
    void registerConf(const char * name, base::config::Type type, T& arg, const char * comment);
    template <class T>
    void registerConf(const char * name, base::config::Type type, T& arg, const char * comment, T def);

    void extractBool();

    const boost::program_options::variables_map &getConf() const;
    virtual void init(int argc, char * argv[]);
    virtual void init(string configFile);
  };
}

template <class T>
void base::config::registerConf(const char * name, base::config::Type type, T& arg, const char * comment) {
  confMap[name] = make_pair(type, &arg);
  if (type == BOOL) {
    addOpt = addOpt(name, comment);
  } else {
    addOpt = addOpt(name, po::value<T>(&arg)->required(), comment);
  }
}

template <class T>
void base::config::registerConf(const char * name, base::config::Type type, T& arg, const char * comment, T def) {
  confMap[name] = make_pair(type, &arg);
  addOpt = addOpt(name, po::value<T>(&arg)->default_value(def), comment);
}

#endif //PROJECT_CONFIG_H
