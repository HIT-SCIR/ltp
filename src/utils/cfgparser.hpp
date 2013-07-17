#ifndef __CONFIG_PARSER_H__
#define __CONFIG_PARSER_H__

#include <iostream>
#include <fstream>
#include <vector>

#ifdef _WIN32
#include <hash_map>
#else
#include <tr1/unordered_map>
struct StringHashFunc{
    size_t operator()(const std::string& s) const {
        unsigned int _seed = 131; // 31 131 1313 13131 131313 etc..
        unsigned int _hash = 0;
        for(std::size_t i = 0; i < s.size(); i++) {
            _hash = (_hash * _seed) + s[i];
        }
        return size_t(_hash);
    }
};
#endif

#include "strutils.hpp"

namespace ltp { //LTP_NAMESPACE_BEGIN
namespace utility { //LTP_UTILITY_NAMESPACE_BEGIN

using namespace std;
using namespace ltp::strutils;

class ConfigParser {

private:
    const static int MAX_ENTRIES = 50;
    int _num_entries;
    bool _valid;

#ifdef _WIN32
    typedef stdext::hash_map<string, string>                internal_entries_t;
    typedef stdext::hash_map<string, internal_entries_t>    internal_sections_t;
#else
    typedef std::tr1::unordered_map<string, string, StringHashFunc>             internal_entries_t;
    typedef std::tr1::unordered_map<string, internal_entries_t, StringHashFunc> internal_sections_t;
#endif  //  end for _WIN32
    internal_sections_t sec;

public:
    /*
     * constructor function for ConfigParser,
     * construct config items from file
     *
     *  @param[in]  filename    the filename
     */
    ConfigParser(const char * filename) : _valid(false) {
        ifstream f( filename );
        if ( f.fail() ) {
            _valid = false;
        } else {
            string line;
            string section_name;
            internal_entries_t * section = NULL;

            _num_entries = 0;
            _valid = true;

            while ( !f.eof() ) {
                getline( f, line );

                /*
                 * handle following case:
                 *  x = y # comments
                 */
                line = cutoff(line, "#");
                if (line.size() == 0) {
                    continue;
                }

                //  section name
                if ( startswith(line, "[") && endswith(line, "]") ) {
                    int len = line.length();
                    section_name = line.substr(1, len - 2);
                    sec[section_name] = internal_entries_t();
                    section = &sec[section_name];
                }

                vector<string> sep = split_by_sep(line, "=");
                if (sep.size() != 2) {
                    continue;
                }

                sep[0] = chomp(sep[0]);
                sep[1] = chomp(sep[1]);

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

    /*
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

    bool has_section(const string& section) {
        return (sec.find(section) != sec.end());
    }

    bool get(const string& section, const string& name, string& val) {
        bool ret = false;
        if (sec.find(section) != sec.end()) {
            if (sec[section].find(name) != sec[section].end()) {
                val = sec[section][name];
                ret = true;
            }
        }
        return ret;
    }

    bool get_integer(const string& section, const string& name, int& intval) {
        string strval;
        int ret = get(section, name, strval);
        if (!ret) {
            return ret;
        }
        if (is_int(strval)) {
            intval = to_int(strval);
            return true;
        } else {
            return false;
        }

        return false;
    }

    bool get_float(const string& section, const string& name, double& dblval) {
        string strval;
        int ret = get(section, name, strval);
        if (!ret) {
            return ret;
        }
        if (is_double(strval)) {
            dblval = to_double(strval);
            return true;
        } else {
            return false;
        }

        return false;
    }

    void display(ostream & out) {
        for (internal_sections_t::const_iterator itx = sec.begin();
                itx != sec.end(); ++ itx) {
            out << "[" << itx->first << "]" << endl;
            for (internal_entries_t::const_iterator j = itx->second.begin();
                    j != itx->second.end(); ++ j) {
                out << j->first << " = " << j->second << endl;
            }
        }
    }

};

} // LTP_UTILITY_NAMESPACE_END
} // LTP_NAMESPACE_END

#endif  // end for __CONFIGURE_H__
