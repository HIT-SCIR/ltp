#ifndef __LTP_UTILS_CONFIG_PARSER_H__
#define __LTP_UTILS_CONFIG_PARSER_H__

#include <iostream>
#include <fstream>
#include <vector>
#include "hasher.hpp"
#include "unordered_map.hpp"
#include "strutils.hpp"

namespace ltp { //LTP_NAMESPACE_BEGIN
namespace utility { //LTP_UTILITY_NAMESPACE_BEGIN

using namespace ltp::strutils;

class ConfigParser {
private:
  const static int MAX_ENTRIES = 50;
  int _num_entries;
  bool _valid;

  typedef std::unordered_map<std::string, std::string,
          __Default_String_HashFunction> internal_entries_t;
  typedef std::unordered_map<std::string, internal_entries_t,
          __Default_String_HashFunction> internal_sections_t;

  internal_sections_t sec;

public:
  /**
   * constructor function for ConfigParser,
   * construct config items from file
   *
   *  @param[in]  filename    the filename
   */
  ConfigParser(const char * filename) : _valid(false) {
    std::ifstream f( filename );
    if ( f.fail() ) {
      _valid = false;
    } else {
      std::string line;
      std::string section_name("__&_global_X__");
      internal_entries_t * section = NULL;

      sec[section_name] = internal_entries_t();
      section = &sec[section_name];

      _num_entries = 0;
      _valid = true;

      while ( !f.eof() ) {
        getline( f, line );

        // handle following case:
        // x = y # comments
        line = strutils::cutoff(line, "#");
        if (line.size() == 0) {
          continue;
        }
        //  section name
        if (strutils::startswith(line, "[") &&
            strutils::endswith(line, "]") ) {
          int len = line.length();
          section_name = line.substr(1, len - 2);
          sec[section_name] = internal_entries_t();
          section = &sec[section_name];
        }

        std::vector<std::string> sep = strutils::split_by_sep(line, "=");
        if (sep.size() != 2) {
          continue;
        }

        sep[0] = strutils::trim_copy(sep[0]);
        sep[1] = strutils::trim_copy(sep[1]);

        if (!section) {
          _valid = false;
          break;
        }

        (*section)[sep[0]] = sep[1];
        _num_entries ++;

        if (_num_entries > MAX_ENTRIES) {
          break;
        }
      }
    }
  }

  /**
   * string wrapper for constructor
   *
   *  @param[in]  filename    the filename
   */
  ConfigParser(const std::string& filename) {
    ConfigParser(filename.c_str());
  }

  ~ConfigParser() {
  }

  bool operator! () const {
    return (_valid == false);
  }

  bool has_section(const std::string& section) {
    return (sec.find(section) != sec.end());
  }

  /**
   * Get the configuration in global section
   *
   *  @param[in]  name  The item name.
   *  @param[out] val   The item value.
   *  @return
   */
  bool get(const std::string & name, std::string & val) {
    bool ret = false;
    std::string section("__&_global_X__");
    return get(section, name, val);
  }

  bool get(const std::string& section, const std::string& name, std::string& val) {
    bool ret = false;
    if (sec.find(section) != sec.end()) {
      if (sec[section].find(name) != sec[section].end()) {
        val = sec[section][name];
        ret = true;
      }
    }
    return ret;
  }

  bool set(const std::string& section, const std::string &name, const std::string& val) {
    bool ret = false;
    if (sec.find(section) != sec.end()) {
      sec[section][name] = val;
      ret = true;
    }
    return ret;
  }

  bool get_integer(const std::string& name, int& intval) {
    std::string strval;
    int ret = get(name, strval);
    if (!ret) {
      return false;
    }

    if (strutils::is_int(strval)) {
      intval = strutils::to_int(strval);
      return true;
    } else {
      return false;
    }
    return false;
  }

  bool get_integer(const std::string& section, const std::string& name, int& intval) {
    std::string strval;
    int ret = get(section, name, strval);
    if (!ret) {
      return false;
    }

    if (strutils::is_int(strval)) {
      intval = strutils::to_int(strval);
      return true;
    } else {
      return false;
    }

    return false;
  }

  bool get_float(const std::string& name, double& dblval) {
    std::string strval;
    int ret = get(name, strval);
    if (!ret) {
      return false;
    }
    if (is_double(strval)) {
      dblval = to_double(strval);
      return true;
    } else {
      return false;
    }
    return false;
  }

  bool get_float(const std::string& section, const std::string& name, double& dblval) {
    std::string strval;
    int ret = get(section, name, strval);
    if (!ret) {
      return false;
    }
    if (is_double(strval)) {
      dblval = to_double(strval);
      return true;
    } else {
      return false;
    }
    return false;
  }

  void display(std::ostream & out) {
    for (internal_sections_t::const_iterator itx = sec.begin();
        itx != sec.end(); ++ itx) {
      out << "[" << itx->first << "]" << std::endl;
      for (internal_entries_t::const_iterator j = itx->second.begin();
          j != itx->second.end(); ++ j) {
        out << j->first << " = " << j->second << std::endl;
      }
    }
  }

};

} // LTP_UTILITY_NAMESPACE_END
} // LTP_NAMESPACE_END

#endif  // end for __CONFIGURE_H__
