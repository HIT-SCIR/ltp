//
//  CRF++ -- Yet Another CRF toolkit
//
//  $Id: feature_index.h 1588 2007-02-12 09:03:39Z taku $;
//
//  Copyright(C) 2005-2007 Taku Kudo <taku@chasen.org>
//
#ifndef CRFPP_FEATURE_INDEX_H__
#define CRFPP_FEATURE_INDEX_H__

#include <vector>
#include <map>
#include <iostream>
#include "common.h"
#include "scoped_ptr.h"
#include "feature_cache.h"
#include "path.h"
#include "node.h"
#include "freelist.h"
#include "mmap.h"
#include "darts.h"

namespace CRFPP {
  class TaggerImpl;

  class FeatureIndex {
  protected:
    unsigned int        maxid_;
    double             *alpha_;
    float              *alpha_float_;
    double              cost_factor_;
    unsigned int        xsize_;
    unsigned int        max_xsize_;
    size_t              thread_num_;
    FeatureCache        feature_cache_;
    std::vector<char*>  unigram_templs_;
    std::vector<char*>  bigram_templs_;
    std::vector<char*>  y_;
    FreeList<char>      char_freelist_;
    scoped_array< FreeList<Path> > path_freelist_;
    scoped_array< FreeList<Node> > node_freelist_;
    whatlog             what_;

    virtual int getID(const char *) = 0;

    const char *get_index(char *&, size_t, const TaggerImpl &);
    bool apply_rule(string_buffer *,
                    char *,
                    size_t, const TaggerImpl &);

  public:
    static const unsigned int version = MODEL_VERSION;

    size_t size() const  { return maxid_; }
    size_t xsize() const { return xsize_; }
    size_t ysize() const { return y_.size(); }
    const char* y(size_t i) const { return y_[i]; }
    void   set_alpha(double *alpha) { alpha_ = alpha; }
    const float *alpha_float() { return const_cast<float *>(alpha_float_); }
    const double *alpha() { return const_cast<double *>(alpha_); }
    void set_cost_factor(double cost_factor) { cost_factor_ = cost_factor; }
    double cost_factor() { return cost_factor_; }
    char *strdup(const char *);
    void calcCost(Node *);
    void calcCost(Path *);

    bool buildFeatures(TaggerImpl *);
    void rebuildFeatures(TaggerImpl *);

    const char* what() { return what_.str(); }

    virtual bool open(const char*, const char*) = 0;
    virtual void clear() = 0;

    void init() {
      path_freelist_.reset(new FreeList<Path> [thread_num_]);
      node_freelist_.reset(new FreeList<Node> [thread_num_]);
      for (size_t i = 0; i < thread_num_; ++i) {
        path_freelist_[i].set_size(8192 * 16);
        node_freelist_[i].set_size(8192);
      }
    }

    explicit FeatureIndex(): maxid_(0), alpha_(0), alpha_float_(0),
      cost_factor_(1.0), xsize_(0), max_xsize_(0),
      thread_num_(1), char_freelist_(8192) {}
    virtual ~FeatureIndex() {}
  };

  class EncoderFeatureIndex: public FeatureIndex {
  private:
    std::map <std::string, std::pair<int, unsigned int> > dic_;
    int getID(const char *);
    bool openTemplate(const char *);
    bool openTagSet(const char *);
  public:
    explicit EncoderFeatureIndex(size_t n) {
      thread_num_ = n;
      init();
    }
    bool open(const char*, const char*);
    bool save(const char *, bool);
    bool convert(const char *, const char*);
    void clear();
    void shrink(size_t) ;
  };

  class DecoderFeatureIndex: public FeatureIndex {
  private:
    Mmap <char> mmap_;
    Darts::DoubleArray da_;
    int getID(const char *);
  public:
    explicit DecoderFeatureIndex() { init(); }
    bool open(const char *, const char *);
    void clear();
  };
}

#endif
