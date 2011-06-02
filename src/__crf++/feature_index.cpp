//
//  CRF++ -- Yet Another CRF toolkit
//
//  $Id: feature_index.cpp 1587 2007-02-12 09:00:36Z taku $;
//
//  Copyright(C) 2005-2007 Taku Kudo <taku@chasen.org>
//
#include <iostream>
#include <fstream>
#include <cstring>
#include <set>
#include "common.h"
#include "feature_index.h"

namespace CRFPP {

  static inline char *read_ptr(char **ptr, size_t size) {
    char *r = *ptr;
    *ptr += size;
    return r;
  }

  template <class T> static inline void read_static(char **ptr,
                                                    T *value) {
    char *r = read_ptr(ptr, sizeof(T));
    memcpy(value, r, sizeof(T));
  }

  int DecoderFeatureIndex::getID(const char *key) {
    return da_.exactMatchSearch<Darts::DoubleArray::result_type>(key);
  }

  int EncoderFeatureIndex::getID(const char *key) {
    std::map <std::string, std::pair<int, unsigned int> >::iterator
      it = dic_.find(key);
    if (it == dic_.end()) {
      dic_.insert(std::make_pair<std::string, std::pair<int, unsigned int> >
                  (key, std::make_pair<int, unsigned int>(maxid_, 1)));
      int n = maxid_;
      maxid_ += (key[0] == 'U' ? y_.size() : y_.size() * y_.size());
      return n;
    } else {
      it->second.second++;
      return it->second.first;
    }
    return -1;
  }

  bool EncoderFeatureIndex::open(const char *filename1,
                                 const char *filename2) {
    return openTemplate(filename1) && openTagSet(filename2);
  }

  bool EncoderFeatureIndex::openTemplate(const char *filename) {
    std::ifstream ifs(filename);
    CHECK_FALSE(ifs) << "open failed: "  << filename;

    std::string line;
    while (std::getline(ifs, line)) {
      if (!line[0] || line[0] == '#') continue;
      if (line[0] == 'U') {
        unigram_templs_.push_back(this->strdup(line.c_str()));
      } else if (line[0] == 'B') {
        bigram_templs_.push_back(this->strdup(line.c_str()));
      } else {
        CHECK_FALSE(true) << "unknown type: " << line << " " << filename;
      }
    }

    return true;
  }

  bool EncoderFeatureIndex::openTagSet(const char *file) {
    std::ifstream ifs(file);
    CHECK_FALSE(ifs) << "no such file or directory: " << file ;

    char  line[8192];
    char* column[1024];
    size_t max_size = 0;
    std::set<std::string> candset;

    while (ifs.getline(line, sizeof(line))) {
      if (line[0] == '\0' || line[0] == ' ' || line[0] == '\t') continue;
      size_t size = tokenize2(line, "\t ", column, 1024);
      if (max_size == 0) max_size = size;
      CHECK_FALSE(max_size == size)
        << "inconsistent column size: "
        << max_size << " " << size << " " << file;
      xsize_ = size - 1;
      candset.insert(column[max_size-1]);
    }

    y_.clear();
    for (std::set<std::string>::iterator it = candset.begin();
         it != candset.end(); ++it)
      y_.push_back(this->strdup(it->c_str()));

    ifs.close();

    return true;
  }

  bool DecoderFeatureIndex::open(const char *filename1,
                                 const char *filename2) {
    CHECK_FALSE(mmap_.open(filename1)) << mmap_.what();

    char *ptr = mmap_.begin();
    unsigned int version_ = 0;

    read_static<unsigned int>(&ptr, &version_);

    CHECK_FALSE(version_ / 100 == version / 100)
      << "model version is different: " << version_
      << " vs " << version << " : " << filename1;
    int type = 0;
    read_static<int>(&ptr, &type);
    read_static<double>(&ptr, &cost_factor_);
    read_static<unsigned int>(&ptr, &maxid_);
    read_static<unsigned int>(&ptr, &xsize_);

    unsigned int dsize = 0;
    read_static<unsigned int>(&ptr, &dsize);

    unsigned int y_str_size;
    read_static<unsigned int>(&ptr, &y_str_size);
    char *y_str = read_ptr(&ptr, y_str_size);
    size_t pos = 0;
    while (pos < y_str_size) {
      y_.push_back(y_str + pos);
      while (y_str[pos++] != '\0') {}
    }

    unsigned int tmpl_str_size;
    read_static<unsigned int>(&ptr, &tmpl_str_size);
    char *tmpl_str = read_ptr(&ptr, tmpl_str_size);
    pos = 0;
    while (pos < tmpl_str_size) {
      char *v = tmpl_str + pos;
      if (v[0] == '\0') {
        ++pos;
      } else if (v[0] == 'U') {
        unigram_templs_.push_back(v);
      } else if (v[0] == 'B') {
        bigram_templs_.push_back(v);
      } else {
        CHECK_FALSE(true) << "unknown type: " << v;
      }
      while (tmpl_str[pos++] != '\0') {}
    }

    da_.set_array(ptr);
    ptr += dsize;

    alpha_float_ = reinterpret_cast<float *>(ptr);
    ptr += sizeof(alpha_float_[0]) * maxid_;

    CHECK_FALSE(ptr == mmap_.end()) <<
      "model file is broken: " << filename1;

    return true;
  }

  void EncoderFeatureIndex::shrink(size_t freq) {
    if (freq <= 1) return;

    std::map<int, int> old2new;
    int new_maxid = 0;

    for (std::map<std::string, std::pair<int, unsigned int> >::iterator
           it = dic_.begin(); it != dic_.end();) {
      const std::string &key = it->first;

      if (it->second.second >= freq) {
        old2new.insert(std::make_pair<int, int>(it->second.first, new_maxid));
        it->second.first = new_maxid;
        new_maxid += (key[0] == 'U' ? y_.size() : y_.size() * y_.size());
        ++it;
      } else {
        dic_.erase(it++);
      }
    }

    feature_cache_.shrink(&old2new);

    maxid_ = new_maxid;

    return;
  }

  void DecoderFeatureIndex::clear() {
    char_freelist_.free();
    feature_cache_.clear();
    for (size_t i = 0; i < thread_num_; ++i) {
      node_freelist_[i].free();
      path_freelist_[i].free();
    }
  }

  void EncoderFeatureIndex::clear() {}

  bool EncoderFeatureIndex::convert(const char *filename1,
                                    const char *filename2) {
    std::ifstream ifs(filename1);

    y_.clear();
    dic_.clear();
    unigram_templs_.clear();
    bigram_templs_.clear();
    xsize_ = 0;
    maxid_ = 0;

    CHECK_FALSE(ifs) << "open failed: " << filename1;

    char line[8192];
    char *column[8];

    // read header
    while (true) {
      CHECK_FALSE(ifs.getline(line, sizeof(line)))
        << " format error: " << filename1;

      if (std::strlen(line) == 0) break;

      size_t size = tokenize(line, "\t ", column, 2);

      CHECK_FALSE(size == 2) << "format error: " << filename1;

      if (std::strcmp(column[0], "xsize:") == 0)
        xsize_ = std::atoi(column[1]);

      if (std::strcmp(column[0], "maxid:") == 0)
        maxid_ = std::atoi(column[1]);
    }
     
    CHECK_FALSE(maxid_ > 0) << "maxid is not defined: " << filename1;

    CHECK_FALSE(xsize_ > 0) << "xsize is not defined: " << filename1;

    while (true) {
      CHECK_FALSE(ifs.getline(line, sizeof(line)))
        << "format error: " << filename1;
      if (std::strlen(line) == 0) break;
      y_.push_back(this->strdup(line));
    }

    while (true) {
      CHECK_FALSE(ifs.getline(line, sizeof(line)))
        << "format error: " << filename1;
      if (std::strlen(line) == 0) break;
      if (line[0] == 'U') {
        unigram_templs_.push_back(this->strdup(line));
      } else if (line[0] == 'B') {
        bigram_templs_.push_back(this->strdup(line));
      } else {
        CHECK_FALSE(true) << "unknown type: " << line << " " << filename1;
      }
    }

    while (true) {
      CHECK_FALSE(ifs.getline(line, sizeof(line)))
        << "format error: " << filename1;
      if (std::strlen(line) == 0) break;

      size_t size = tokenize(line, "\t ", column, 2);
      CHECK_FALSE(size == 2) << "format error: " << filename1;

      dic_.insert(std::make_pair<std::string, std::pair<int, unsigned int> >
                  (column[1],
                   std::make_pair<int, unsigned int>(std::atoi(column[0]), 1)));
    }

    std::vector<double> alpha;
    while (ifs.getline(line, sizeof(line)))
      alpha.push_back(std::atof(line));

    alpha_ = &alpha[0];

    CHECK_FALSE(alpha.size() == maxid_) << " file is broken: "  << filename1;

    return save(filename2, false);
  }


  bool EncoderFeatureIndex::save(const char *filename, bool textmodelfile) {
    std::vector <char *> key;
    std::vector <int>    val;

    std::string y_str;
    for (size_t i = 0; i < y_.size(); ++i) {
      y_str += std::string(y_[i]);
      y_str += '\0';
    }

    std::string templ_str;
    for (size_t i = 0; i < unigram_templs_.size(); ++i) {
      templ_str += std::string(unigram_templs_[i]);
      templ_str += '\0';
    }

    for (size_t i = 0; i < bigram_templs_.size(); ++i) {
      templ_str += std::string(bigram_templs_[i]);
      templ_str += '\0';
    }

    while ((y_str.size() + templ_str.size()) % 4 != 0)
      templ_str += '\0';

    for (std::map<std::string, std::pair<int, unsigned int> >::iterator
           it = dic_.begin();
         it != dic_.end(); ++it) {
      key.push_back(const_cast<char *>(it->first.c_str()));
      val.push_back(it->second.first);
    }

    Darts::DoubleArray da;

    CHECK_FALSE(da.build(key.size(), &key[0], 0, &val[0]) == 0)
      << "cannot build double-array";

    std::ofstream bofs;
    bofs.open(filename, OUTPUT_MODE);

    CHECK_FALSE(bofs) << "open failed: " << filename;

    unsigned int version_ = version;
    bofs.write(reinterpret_cast<char *>(&version_), sizeof(unsigned int));

    int type = 0;
    bofs.write(reinterpret_cast<char *>(&type), sizeof(type));
    bofs.write(reinterpret_cast<char *>(&cost_factor_), sizeof(cost_factor_));
    bofs.write(reinterpret_cast<char *>(&maxid_), sizeof(maxid_));

    if (max_xsize_ > 0) {
      xsize_ = _min(xsize_, max_xsize_);
    }
    bofs.write(reinterpret_cast<char *>(&xsize_), sizeof(xsize_));
    unsigned int dsize = da.unit_size() * da.size();
    bofs.write(reinterpret_cast<char *>(&dsize), sizeof(dsize));
    unsigned int size = y_str.size();
    bofs.write(reinterpret_cast<char *>(&size),  sizeof(size));
    bofs.write(const_cast<char *>(y_str.data()), y_str.size());
    size = templ_str.size();
    bofs.write(reinterpret_cast<char *>(&size),  sizeof(size));
    bofs.write(const_cast<char *>(templ_str.data()), templ_str.size());
    bofs.write(reinterpret_cast<const char *>(da.array()), dsize);

    for (size_t i  = 0; i < maxid_; ++i) {
      float alpha = static_cast<float>(alpha_[i]);
      bofs.write(reinterpret_cast<char *>(&alpha), sizeof(alpha));
    }

    bofs.close();

    if (textmodelfile) {
      std::string filename2 = filename;
      filename2 += ".txt";

      std::ofstream tofs(filename2.c_str());

      CHECK_FALSE(tofs) << " no such file or directory: " << filename2;

      // header
      tofs << "version: "     << version_ << std::endl;
      tofs << "cost-factor: " << cost_factor_ << std::endl;
      tofs << "maxid: "       << maxid_ << std::endl;
      tofs << "xsize: "       << xsize_ << std::endl;

      tofs << std::endl;

      // y
      for (size_t i = 0; i < y_.size(); ++i)
        tofs << y_[i] << std::endl;

      tofs << std::endl;

      // template
      for (size_t i = 0; i < unigram_templs_.size(); ++i)
        tofs << unigram_templs_[i] << std::endl;

      for (size_t i = 0; i < bigram_templs_.size(); ++i)
        tofs << bigram_templs_[i] << std::endl;

      tofs << std::endl;

      // dic
      for (std::map<std::string, std::pair<int, unsigned int> >::iterator
             it = dic_.begin();
           it != dic_.end(); ++it) {
        tofs << it->second.first << " " << it->first << std::endl;
      }

      tofs << std::endl;

      tofs.setf(std::ios::fixed, std::ios::floatfield);
      tofs.precision(16);

      for (size_t i  = 0; i < maxid_; ++i)
        tofs << alpha_[i] << std::endl;
    }

    return true;
  }

  char *FeatureIndex::strdup(const char *p) {
    size_t len = std::strlen(p);
    char *q = char_freelist_.alloc(len+1);
    std::strcpy(q, p);
    return q;
  }

  void FeatureIndex::calcCost(Node *n) {
    n->cost = 0.0;

#define ADD_COST(T, A) \
      { T c = 0; \
        for (int *f = n->fvector; *f != -1; ++f) c += (A)[*f + n->y]; \
        n->cost =cost_factor_ *(T)c; }

    if (alpha_float_) ADD_COST(float,  alpha_float_)
                        else             ADD_COST(double, alpha_);
#undef ADD_COST
  }

  void FeatureIndex::calcCost(Path *p) {
    p->cost = 0.0;

#define ADD_COST(T, A) \
      { T c = 0.0; \
        for (int *f = p->fvector; *f != -1; ++f) \
          c += (A)[*f + p->lnode->y * y_.size() + p->rnode->y]; \
        p->cost =cost_factor_*(T)c; }

    if (alpha_float_) ADD_COST(float,  alpha_float_)
                        else             ADD_COST(double, alpha_);
  }
#undef ADD_COST
}
