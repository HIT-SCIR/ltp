#ifndef __LTP_CONSOLE_DISPATCHER__
#define __LTP_CONSOLE_DISPATCHER__

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "tinythread.h"

using namespace std;

class Dispatcher {
public:
  Dispatcher(void* engine, std::istream& is, std::ostream& os):
    _engine(engine),
    _max_idx(0),
    _idx(0),
    _is(is),
    _os(os) {}

  int next(std::string& sentence) {
    sentence = "";
    tthread::lock_guard<tthread::mutex> guard(_mutex);
    if (!std::getline(_is, sentence, '\n')) {
      return -1;
    }
    return _max_idx ++;
  }

  int next_block(vector<std::string>& block) {
    block.clear();
    tthread::lock_guard<tthread::mutex> guard(_mutex);
    std::string line;
    while (std::getline(_is, line, '\n')) {
      if (line != "") {
        block.push_back(line);
      } else {
        return _max_idx ++;
      }
    }
    if (block.size()) return _max_idx++;
    return -1;
  }

  void output(const size_t& idx, const std::string& result) {
    tthread::lock_guard<tthread::mutex> guard(_mutex);
    if (idx > _idx) {
      _back[idx] = result;
    } else if (idx == _idx) {
      _os << result << std::endl;
      ++ _idx;

      std::map<size_t, std::string>::iterator itx;
      itx = _back.find(_idx);

      while (itx != _back.end()) {
        _os << itx->second << std::endl;
        _back.erase(itx);
        ++ _idx;
        itx = _back.find(_idx);
      }
    }
    return;
  }

  void* get_engine() {
    return _engine;
  }

private:
  tthread::mutex  _mutex;
  void* _engine;
  size_t _max_idx;
  size_t _idx;
  std::istream& _is;
  std::ostream& _os;
  std::map<size_t, std::string> _back;
};

#endif  //  end for __LTP_CONSOLE_DISPATCHER__
