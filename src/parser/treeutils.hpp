#ifndef __LTP_PARSER_TREE_UTILS_HPP__
#define __LTP_PARSER_TREE_UTILS_HPP__

#include <iostream>
#include <vector>
#include <list>

#include <string.h>

namespace ltp {
namespace parser {
namespace treeutils {

/*
 * Get children given certain heads, complexity of is O(n)
 *
 *  @param[in]  heads       the heads
 *  @param[out] children_left   result for left children
 *  @param[out] children_right  result for right children
 */
inline int get_children( const std::vector<int> & heads,
    std::vector< std::list<int> > & children_left,
    std::vector< std::list<int> > & children_right ) {
  int ret = 0;
  int len = heads.size();

  children_left.resize(len);
  children_right.resize(len);

  // clear each element in the output vector
  for (int i = 0; i < len; ++ i) {
    children_left[i].clear();
    children_right[i].clear();
  }

  for (int i = 1; i < len; ++ i) {
    int hid = heads[i];

    if (i < hid) {
      ret ++;
      children_left[hid].push_front(i);
    } else {
      children_right[hid].push_back(i);
    }
  }

  return ret;
}

// Generate all the tree space in dependency
// This class is a Python `yield` like generator
// detail for implement can refer to
// http://www.chiark.greenend.org.uk/~sgtatham/coroutines.html
/*
 * dependency tree space iterator, enumerate all posibly feature
 * 2-tuple of a tree. for example, for a tree like:
 *
 *  [0] -> ROOT; [1] -> [2]; [2] -> [0]
 *
 * it will generate:
 *  (0, 1), (0, 2), (1, 2), (2, 1)
 */
class DEPTreeSpaceIterator {
public:
  DEPTreeSpaceIterator(int len) :
    _len(len),
    _hid(0),
    _cid(0),
    _state(0) {
    ++ (*this);
  }

  inline int hid(void) {
    return _hid;
  }

  inline int cid(void) {
    return _cid;
  }

  inline bool end(void) {
    return _hid >= _len;
  }

  void operator ++(void) {
    switch (_state) {
      case 0:
        for (_hid = 0; _hid < _len; ++ _hid) {
          for (_cid = 0; _cid < _len; ++ _cid) {
            if (_hid == _cid) {
              continue;
            }
            _state = 1;
            return;
      case 1:;
          }
        }
    }
  }
private:
  int _len;
  int _hid;
  int _cid;
  int _state;
};    //  end for DEPIterator

/*
 * sibling tree space iterator, enumerate all possible feature
 * 3-tuple of a tree. for example, for a tree like:
 *
 *  [0] -> ROOT; [1] -> [2]; [2] -> [0]; [3] -> [2]
 *
 * it will generate:
 *
 *  (0,1,0), (0,2,0), (0,3,0), (0,2,1), (0,2,2) (if last sibling is
 *  configed.) ...
 */
class SIBTreeSpaceIterator {
public:
  SIBTreeSpaceIterator(int len, bool last_sibling = true) :
    _len(len),
    _last_sibling(last_sibling),
    _hid(0),
    _cid(0),
    _sid(0),
    _step(0),
    _end(0),
    _state(0) {
    ++ (*this);
  }

  inline int hid() {
    return _hid;
  }

  inline int cid() {
    return _cid;
  }

  inline int sid() {
    return _sid;
  }

  inline bool end() {
    return _hid >= _len;
  }

  void operator ++(void) {
    switch (_state) {
      case 0:
        for (_hid = 0; _hid < _len; ++ _hid) {
          for (_cid = 0; _cid < _len; ++ _cid) {
            if (_hid == _cid) {
              continue;
            }

            _step = (_hid < _cid ? 1 : -1);
            _end = (_last_sibling ? _cid + _step : _cid);
            for (_sid = _hid; _sid != _end; _sid += _step) {
              _state = 1;
              return;
      case 1:;
            }
          }
        }
    }
  }
private:
  int _len;
  int _hid;
  int _cid;
  int _sid;
  int _step;
  int _end;
  int _state;
  bool _last_sibling;
};

/*
 * grand tree space iterator, enumerate all possible feature
 * 3-tuple of a tree. for example, for a tree like:
 *
 *  [0] -> ROOT; [1] -> [2]; [2] -> [0]; [3] -> [2]
 *
 * it will generate:
 *
 *  (0,1,0), (0,2,0), (0,3,0), (0,2,1), (0,2,2)  ...
 */
class GRDTreeSpaceIterator {
public:
  GRDTreeSpaceIterator(int len, bool no_grand = true) :
    _hid(0),
    _step(0),
    _end(0),
    _len(len),
    _state(0),
    _no_grand(no_grand) {
    ++ (*this);
  }

  inline int hid() {
    return _hid;
  }

  inline int cid() {
    return _cid;
  }

  inline int gid() {
    return _gid;
  }

  bool end() {
    return _hid >= _len;
  }

  void operator ++(void) {
    switch(_state) {
      case 0:
        for (_hid = 0; _hid < _len; ++ _hid) {
          for (_cid = 1; _cid < _len; ++ _cid) {
            if (_cid == _hid) {
              continue;
            }
            _step = (_hid < _cid ? 1 : -1);
            _end = (_hid < _cid ? _len : 0);

            for (_gid = _hid; _gid != _end; _gid += _step) {
              if ((_gid == _hid || _gid == _cid) && !_no_grand) {
                continue;
              }
              _state = 1;
              return;
      case 1:;
            }
          }
        }
    }
  }
private:
  int _len;
  int _hid;
  int _cid;
  int _gid;
  int _step;
  int _end;
  int _state;
  bool _no_grand;
};

/*
 * dependency tree iterator, enumerate all possible features
 * 2-tuple according a tree. for example, for a tree like:
 *
 *  [0] -> ROOT; [1] -> [2]; [2] -> [0]; [3] -> [2]
 *
 * it will generate:
 *
 *  (0, 2), (2, 1), (2, 3)
 */
class DEPIterator {
public:
  DEPIterator(const std::vector<int> & heads) :
    _cid(1),
    _len(heads.size()),
    _heads(heads) {}

  inline int hid() {
    return _heads[_cid];
  }

  inline int cid() {
    return _cid;
  }

  inline bool end() {
    return _cid >= _len;
  }

  void operator ++(void) {
    ++ _cid;
  }
private:
  int _len;
  int _cid;
  const std::vector<int> & _heads;
};

/*
 * sibling tree space iterator, enumerate all possible feature
 * 3-tuple of a tree. for example, for a tree like:
 *
 *  [0] -> ROOT; [1] -> [0]; [2] -> [0]; [3] -> [0]
 *
 * it will generate:
 *
 *  (0,1,0), (0,2,1), (0,3,2), (0,3,3) (if last sibling is configed.) ...
 */
class SIBIterator {
public:
  SIBIterator(const std::vector<int> & heads, bool last_sibling = true) :
    _hid(0),
    _state(0),
    _last_sibling(last_sibling),
    _len(heads.size()),
    _heads(heads) {

    for (int dir = 0; dir < 2; ++ dir) {
      _children[dir]  = new int *[_len];
      _num_children[dir] = new int[_len];

      memset(_num_children[dir], 0, sizeof(int) * _len);

      for (int i = 0; i < _len; ++ i) {
        _children[dir][i]  = new int[_len];
        _children[dir][i][_num_children[dir][i] ++] = i;
      }
    }

    for (int i = _len - 1; i > 0; -- i) {
      int hid = _heads[i];
      int * children = _children[0][hid];
      if (i < hid) {
        children[_num_children[0][hid] ++] = i;
      }
    }

    for (int i = 1; i < _len; ++ i) {
      int hid = _heads[i];
      int * children = _children[1][hid];
      if (i > hid) {
        children[_num_children[1][hid] ++] = i;
      }
    }

    if (_last_sibling) {
      for (int i = 0; i < _len; ++ i) {
        for (int dir = 0; dir < 2; ++ dir) {
          if (_num_children[dir][i] > 1) {
            _children[dir][i][_num_children[dir][i]] = _children[dir][i][_num_children[dir][i] - 1];
            _num_children[dir][i] ++;
          }
        }
      }
    }

    ++ (*this);
  }

  ~SIBIterator() {
    for (int i = 0; i < _len; ++ i) {
      delete [](_children[0][i]);
      delete [](_children[1][i]);
    }

    delete [](_children[0]);
    delete [](_children[1]);

    delete [](_num_children[0]);
    delete [](_num_children[1]);
  }

  inline int hid(void) {
    return _hid;
  }

  inline int cid(void) {
    return _cid;
  }

  inline int sid(void) {
    return _sid;
  }

  inline bool end(void) {
    return _hid >= _len;
  }

  void operator ++(void) {

    switch (_state) {
      case 0:
        for (_hid = 0; _hid < _len; ++ _hid) {
          for (_dir = 0; _dir < 2; ++ _dir) {
            for (_idx = 1; _idx < _num_children[_dir][_hid]; ++ _idx) {
              _cid = _children[_dir][_hid][_idx];
              _sid = _children[_dir][_hid][_idx - 1];
              _state = 1;
              return;
      case 1:;
            }
          }
        }
    }
  }
private:
  void debug(void) {
    for (int i = 0; i < _len; ++ i) {
      std::cerr << "[" << i << "] --> (";
      for (int j = 1; j < _num_children[0][i]; ++ j) {
        std::cerr << _children[0][i][j] << ",";
      }
      std::cerr << "), (";
      for (int j = 1; j < _num_children[1][i]; ++ j) {
        std::cerr << _children[1][i][j] << ",";
      }
      std::cerr << ")" << std::endl;
    }
  }

private:
  int _hid;
  int _cid;
  int _sid;
  int _dir;
  int _len;
  int _idx;
  int _state;
  bool _last_sibling;
  const std::vector<int> & _heads;

  int ** _children[2];
  int *  _num_children[2];
};

class GRDIterator {
public:
  GRDIterator(const std::vector<int> & heads,
              bool no_grand = true,
              bool outmost_grand = true)
    : _hid(0),
      _state(0),
      _no_grand(no_grand),
      _outmost_grand(outmost_grand),
      _len(heads.size()),
      _heads(heads) {
    for (int dir = 0; dir < 2; ++ dir) {
      _children[dir]  = new int *[_len];
      _num_children[dir] = new int[_len];

      memset(_num_children[dir], 0, sizeof(int) * _len);

      for (int i = 0; i < _len; ++ i) {
        _children[dir][i]  = new int[_len];
      }
    }

    for (int i = _len - 1; i > 0; -- i) {
      int hid = _heads[i];
      int * children = _children[1][hid];
      if (i > hid) {
        if (_num_children[1][hid] > 0 && outmost_grand) {
          continue;
        }

        children[_num_children[1][hid] ++] = i;
      }
    }

    for (int i = 1; i < _len; ++ i) {
      int hid = _heads[i];
      int * children = _children[0][hid];
      if (i < hid) {
        if (_num_children[0][hid] > 0 && outmost_grand) {
          continue;
        }

        children[_num_children[0][hid] ++] = i;
      }
    }

    if (_no_grand) {
      for (int cid = 1; cid < _len; ++ cid) {
        int hid = _heads[cid];
        for (int dir = 0; dir < 2; ++ dir) {
          if (_num_children[dir][cid] == 0) {
            _children[dir][cid][_num_children[dir][cid] ++] = (dir ? hid : cid);
          }
        }
      }
    }

    ++ (*this);
  }

  ~GRDIterator() {
    for (int i = 0; i < _len; ++ i) {
      delete [](_children[0][i]);
      delete [](_children[1][i]);
    }

    delete [](_children[0]);
    delete [](_children[1]);

    delete [](_num_children[0]);
    delete [](_num_children[1]);
  }

  inline int hid(void) {
    return _hid;
  }

  inline int cid(void) {
    return _cid;
  }

  inline int gid(void) {
    return _gid;
  }

  inline bool end(void) {
    return _cid >= _len;
  }

  void operator ++(void) {
    switch (_state) {
      case 0:
        for (_cid = 1; _cid < _len; ++ _cid) {
          _hid = _heads[_cid];
          for (_dir = 0; _dir < 2; ++ _dir) {
            for (_idx = 0; _idx < _num_children[_dir][_cid]; ++ _idx) {
              _gid = _children[_dir][_cid][_idx];
              _state = 1;
              return;
      case 1:;
            }
          }
        }
    }
  }

private:
  int _hid;
  int _cid;
  int _gid;
  int _len;
  int _dir;
  int _state;
  int _idx;
  bool _no_grand;
  bool _outmost_grand;
  const std::vector<int> & _heads;

  int ** _children[2];
  int * _num_children[2];
};

}     //  end for namespace treeutils
}     //  end for namespace parser
}     //  end for namespace ltp

#endif  //  end for __LTP_PARSER_TREE_UTILS_HPP__
