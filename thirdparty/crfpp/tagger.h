//
//  CRF++ -- Yet Another CRF toolkit
//
//  $Id: tagger.h 1588 2007-02-12 09:03:39Z taku $;
//
//  Copyright(C) 2005-2007 Taku Kudo <taku@chasen.org>
//
#ifndef CRFPP_TAGGER_H__
#define CRFPP_TAGGER_H__

#include <iostream>
#include <vector>
#include <queue>
#include "param.h"
#include "crfpp.h"
#include "scoped_ptr.h"
#include "feature_index.h"

namespace CRFPP {

  static inline double toprob(Node *n, double Z) {
    return std::exp(n->alpha + n->beta - n->cost - Z);
  }

  class TaggerImpl : public Tagger {
//  private:
  protected:

    struct QueueElement {
      Node *node;
      QueueElement *next;
      double fx;
      double gx;
    };

    class QueueElementComp {
    public:
      const bool operator()(QueueElement *q1,
                            QueueElement *q2)
        { return(q1->fx > q2->fx); }
    };

    enum { TEST, LEARN };
    unsigned int mode_   : 2;
    unsigned int vlevel_ : 3;
    unsigned int nbest_  : 11;
    size_t                            ysize_;
    double                            cost_;
    double                            Z_;
    size_t                            feature_id_;
    unsigned short                    thread_id_;
    FeatureIndex                     *feature_index_;
    std::vector <std::vector <const char *> > x_;
    std::vector <std::vector <Node *> > node_;
    std::vector <unsigned short int>  answer_;
    std::vector <unsigned short int>  result_;
    whatlog what_;
    string_buffer os_;

    scoped_ptr<std::priority_queue <QueueElement*, std::vector <QueueElement *>,
      QueueElementComp> > agenda_;
    scoped_ptr<FreeList <QueueElement> > nbest_freelist_;

    void forwardbackward();
    void viterbi();
    void buildLattice();
    bool initNbest();
    bool add2(size_t, const char **, bool);

  public:
    explicit TaggerImpl(): mode_(TEST), vlevel_(0), nbest_(0),
      ysize_(0), Z_(0), feature_id_(0),
      thread_id_(0), feature_index_(0) {}
    virtual ~TaggerImpl() { close(); }

    void   set_feature_id(size_t id) { feature_id_  = id; }
    size_t feature_id() const { return feature_id_; }
    void   set_thread_id(unsigned short id) { thread_id_ = id; }
    unsigned short thread_id() { return thread_id_; }
    Node  *node(size_t i, size_t j) { return node_[i][j]; }
    void   set_node(Node *n, size_t i, size_t j) { node_[i][j] = n; }

    int          eval();
    double       gradient(double *);
    double       collins(double *);
    bool         shrink();
    bool         parse_stream(std::istream *, std::ostream *);
    bool         read(std::istream *);
    bool         open(Param *);
    bool         open(FeatureIndex *);
    bool         open(const char*);
    bool         open(int, char **);
    void         close();
    bool         add(size_t, const char **);
    bool         add(const char*);
    size_t       size() const { return x_.size(); }
    size_t       xsize() const { return feature_index_->xsize(); }
    size_t       dsize() const { return feature_index_->size(); }
    const float *weight_vector() const { return feature_index_->alpha_float(); }
    bool         empty() const { return x_.empty(); }
    size_t ysize() const { return ysize_; }
    double cost() const { return cost_; }
    double Z() const { return Z_; }
    double       prob() const { return std::exp(- cost_ - Z_); }
    double       prob(size_t i, size_t j) const {
      return toprob(node_[i][j], Z_);
    }
    double       prob(size_t i) const {
      return toprob(node_[i][result_[i]], Z_);
    }
    double alpha(size_t i, size_t j) const { return node_[i][j]->alpha; }
    double beta(size_t i, size_t j) const { return node_[i][j]->beta; }
    double emission_cost(size_t i, size_t j) const { return node_[i][j]->cost; }
    double next_transition_cost(size_t i, size_t j, size_t k) const {
      return node_[i][j]->rpath[k]->cost;
    }
    double prev_transition_cost(size_t i, size_t j, size_t k) const {
      return node_[i][j]->lpath[k]->cost;
    }
    double best_cost(size_t i, size_t j) const {
      return node_[i][j]->bestCost;
    }
    const int *emission_vector(size_t i, size_t j) const {
      return const_cast<int *>(node_[i][j]->fvector);
    }
    const int* next_transition_vector(size_t i, size_t j, size_t k) const {
      return node_[i][j]->rpath[k]->fvector;
    }
    const int* prev_transition_vector(size_t i, size_t j, size_t k) const {
      return node_[i][j]->lpath[k]->fvector;
    }
    size_t answer(size_t i) const { return answer_[i]; }
    size_t result(size_t i) const { return result_[i]; }
    size_t y(size_t i)  const    { return result_[i]; }
    const char* yname(size_t i) const    { return feature_index_->y(i); }
    const char* y2(size_t i) const      { return yname(result_[i]); }
    const char*  x(size_t i, size_t j) const { return x_[i][j]; }
    const char** x(size_t i) const {
      return const_cast<const char **>(&x_[i][0]);
    }
    const char* toString();
    const char* toString(char *, size_t);
    const char* parse(const char*);
    const char* parse(const char*, size_t);
    const char* parse(const char*, size_t, char*, size_t);
    bool parse();
    bool clear();
    bool next();

    unsigned int vlevel() const { return vlevel_; }

    float cost_factor() const {
      return (float)feature_index_->cost_factor();
    }

    size_t nbest() const { return nbest_; }

    void set_vlevel(unsigned int vlevel) {
      vlevel_ = vlevel;
    }

    void set_cost_factor(float cost_factor) {
      if (cost_factor > 0)
        feature_index_->set_cost_factor(cost_factor);
    }

    void set_nbest(size_t nbest) {
      nbest_ = (unsigned int)nbest;
    }

    const char* what() { return what_.str(); }
  };
}

#endif
