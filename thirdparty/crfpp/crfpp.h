/*
  CRF++ -- Yet Another CRF toolkit

  $Id: crfpp.h 1592 2007-02-12 09:40:53Z taku $;

  Copyright(C) 2005-2007 Taku Kudo <taku@chasen.org>
*/
#ifndef CRFPP_CRFPP_H__
#define CRFPP_CRFPP_H__

/* C interface  */
#ifdef __cplusplus
#include <cstdio>
#else
#include <stdio.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#include <windows.h>
#  ifdef DLL_EXPORT
#    define CRFPP_DLL_EXTERN  __declspec(dllexport)
#  else
//#    define CRFPP_DLL_EXTERN  __declspec(dllimport)
# define CRFPP_DLL_EXTERN extern
#  endif
#endif

#ifndef CRFPP_DLL_EXTERN
#  define CRFPP_DLL_EXTERN extern
#endif

#ifndef SWIG
  typedef struct crfpp_t crfpp_t;

  /* C interface */
  CRFPP_DLL_EXTERN crfpp_t* crfpp_new(int,  char**);
  CRFPP_DLL_EXTERN crfpp_t* crfpp_new2(const char*);
  CRFPP_DLL_EXTERN void     crfpp_destroy(crfpp_t*);
  CRFPP_DLL_EXTERN bool     crfpp_add2(crfpp_t*, size_t, const char **);
  CRFPP_DLL_EXTERN bool     crfpp_add(crfpp_t*, const char*);
  CRFPP_DLL_EXTERN size_t   crfpp_size(crfpp_t*);
  CRFPP_DLL_EXTERN size_t   crfpp_xsize(crfpp_t*);
  CRFPP_DLL_EXTERN size_t   crfpp_dsize(crfpp_t*);
  CRFPP_DLL_EXTERN const float* crfpp_weight_vector(crfpp_t*);
  CRFPP_DLL_EXTERN size_t   crfpp_result(crfpp_t*, size_t);
  CRFPP_DLL_EXTERN size_t   crfpp_answer(crfpp_t*, size_t);
  CRFPP_DLL_EXTERN size_t   crfpp_y(crfpp_t*, size_t);
  CRFPP_DLL_EXTERN size_t   crfpp_ysize(crfpp_t*);
  CRFPP_DLL_EXTERN double   crfpp_prob(crfpp_t*, size_t, size_t);
  CRFPP_DLL_EXTERN double   crfpp_prob2(crfpp_t*, size_t);
  CRFPP_DLL_EXTERN double   crfpp_prob3(crfpp_t*);
  CRFPP_DLL_EXTERN double   crfpp_alpha(crfpp_t*, size_t, size_t);
  CRFPP_DLL_EXTERN double   crfpp_beta(crfpp_t*, size_t, size_t);
  CRFPP_DLL_EXTERN double   crfpp_emisstion_cost(crfpp_t*, size_t, size_t);
  CRFPP_DLL_EXTERN double   crfpp_next_transition_cost(crfpp_t*, size_t,
                                                       size_t, size_t);
  CRFPP_DLL_EXTERN double   crfpp_prev_transition_cost(crfpp_t*, size_t,
                                                       size_t, size_t);
  CRFPP_DLL_EXTERN double   crfpp_best_cost(crfpp_t*, size_t, size_t);
  CRFPP_DLL_EXTERN const int* crfpp_emittion_vector(crfpp_t*, size_t, size_t);
  CRFPP_DLL_EXTERN const int* crfpp_next_transition_vector(crfpp_t*, size_t,
                                                           size_t, size_t);
  CRFPP_DLL_EXTERN const int* crfpp_prev_transition_vector(crfpp_t*, size_t,
                                                           size_t, size_t);
  CRFPP_DLL_EXTERN double   crfpp_Z(crfpp_t*);
  CRFPP_DLL_EXTERN bool     crfpp_parse(crfpp_t*);
  CRFPP_DLL_EXTERN bool     crfpp_empty(crfpp_t*);
  CRFPP_DLL_EXTERN bool     crfpp_clear(crfpp_t*);
  CRFPP_DLL_EXTERN bool     crfpp_next(crfpp_t*);
  CRFPP_DLL_EXTERN int      crfpp_test(int, char **);
  CRFPP_DLL_EXTERN int      crfpp_learn(int, char **);
  CRFPP_DLL_EXTERN const char*  crfpp_strerror(crfpp_t*);
  CRFPP_DLL_EXTERN const char*  crfpp_yname(crfpp_t*, size_t);
  CRFPP_DLL_EXTERN const char*  crfpp_y2(crfpp_t*, size_t);
  CRFPP_DLL_EXTERN const char*  crfpp_x(crfpp_t*, size_t, size_t);
  CRFPP_DLL_EXTERN const char** crfpp_x2(crfpp_t*, size_t);
  CRFPP_DLL_EXTERN const char*  crfpp_parse_tostr(crfpp_t*, const char*);
  CRFPP_DLL_EXTERN const char*  crfpp_parse_tostr2(crfpp_t*,
                                                   const char*, size_t);
  CRFPP_DLL_EXTERN const char*  crfpp_parse_tostr3(crfpp_t*, const char*,
                                                   size_t, char *, size_t);
  CRFPP_DLL_EXTERN const char*  crfpp_tostr(crfpp_t*);
  CRFPP_DLL_EXTERN const char*  crfpp_tostr2(crfpp_t*, char *, size_t);

  CRFPP_DLL_EXTERN void crfpp_set_vlevel(crfpp_t *, unsigned int);
  CRFPP_DLL_EXTERN unsigned int crfpp_vlevel(crfpp_t *);
  CRFPP_DLL_EXTERN void crfpp_set_cost_factor(crfpp_t *, float);
  CRFPP_DLL_EXTERN float crfpp_cost_factor(crfpp_t *);
  CRFPP_DLL_EXTERN void crfpp_set_nbest(crfpp_t *, size_t);
#endif

#ifdef __cplusplus
}
#endif

/* C++ interface */
#ifdef __cplusplus

namespace CRFPP {

  class Tagger {
  public:
#ifndef SWIG
    // open model with parameters in argv[]
    // e.g, argv[] = {"CRF++", "-m", "model", "-v3"};
    virtual bool open(int argc,  char** argv) = 0;

    // open model with parameter arg, e.g. arg = "-m model -v3";
    virtual bool open(const char* arg) = 0;

    // add str[] as tokens to the current context
    virtual bool add(size_t size, const char **str) = 0;

    // close the current model
    virtual void close() = 0;

    // return parameter vector. the size should be dsize();
    virtual const float *weight_vector() const = 0;
#endif

    // set vlevel
    virtual void set_vlevel(unsigned int vlevel) = 0;

    // get vlevel
    virtual unsigned int vlevel() const = 0;

    // set cost factor
    virtual void set_cost_factor(float cost_factor) = 0;

    // get cost factor
    virtual float cost_factor() const = 0;

    // set nbest
    virtual void set_nbest(size_t nbest) = 0;

    // get nbest
    virtual size_t nbest() const = 0;

    // add one line to the current context
    virtual bool add(const char* str) = 0;

    // return size of tokens(lines)
    virtual size_t size() const = 0;

    // return size of column
    virtual size_t xsize() const = 0;

    // return size of features
    virtual size_t dsize() const = 0;

    // return output tag-id of i-th token
    virtual size_t result(size_t i) const = 0;

    // return answer tag-id of i-th token if it is available
    virtual size_t answer(size_t i) const = 0;

    // alias of result(i)
    virtual size_t y(size_t i) const = 0;

    // return output tag of i-th token as string
    virtual const char*   y2(size_t i) const = 0;

    // return i-th tag-id as string
    virtual const char*   yname(size_t i) const = 0;

    // return token at [i,j] as string(i:token j:column)
    virtual const char*   x(size_t i, size_t j) const = 0;

#ifndef SWIG
    // return an array of strings at i-th tokens
    virtual const char**  x(size_t) const = 0;
#endif

    // return size of output tags
    virtual size_t ysize() const = 0;

    // return marginal probability of j-th tag id at i-th token
    virtual double prob(size_t i, size_t j) const = 0;

    // return marginal probability of output tag at i-th token
    // same as prob(i, tagger->y(i));
    virtual double prob(size_t i) const = 0;

    // return conditional probability of enter output
    virtual double prob() const = 0;

    // return forward log-prob of the j-th tag at i-th token
    virtual double alpha(size_t i, size_t j) const = 0;

    // return backward log-prob of the j-th tag at i-th token
    virtual double beta(size_t i, size_t j) const = 0;

    // return emission cost of the j-th tag at i-th token
    virtual double emission_cost(size_t i, size_t j) const = 0;

    // return transition cost of [j-th tag at i-th token] to
    // [k-th tag at(i+1)-th token]
    virtual double next_transition_cost(size_t i,
                                        size_t j, size_t k) const = 0;

    // return transition cost of [j-th tag at i-th token] to
    // [k-th tag at(i-1)-th token]
    virtual double prev_transition_cost(size_t i,
                                        size_t j, size_t k) const = 0;

    //  return the best accumulative cost to the j-th tag at i-th token
    // used in viterbi search
    virtual double best_cost(size_t i, size_t j) const = 0;

#ifndef SWIG
    // return emission feature vector of the j-th tag at i-th token
    virtual const int* emission_vector(size_t i, size_t j) const = 0;

    // return transition feature vector of [j-th tag at i-th token] to
    // [k-th tag at(i+1)-th token]
    virtual const int* next_transition_vector(size_t i,
                                              size_t j, size_t k) const = 0;

    // return transition feature vector of [j-th tag at i-th token] to
    // [k-th tag at(i-1)-th token]
    virtual const int* prev_transition_vector(size_t i,
                                              size_t j, size_t k) const = 0;
#endif

    // normalizing factor(log-prob)
    virtual double Z() const = 0;

    // do parse and change the internal status, if failed, returns false
    virtual bool parse() = 0;

    // return true if the context is empty
    virtual bool empty() const = 0;

    // clear all context
    virtual bool clear() = 0;

    // change the internal state to output next-optimal output.
    // calling it n-th times, can get n-best results,
    // Neeed to specify -nN option to use this function, where
    // N>=2
    virtual bool next() = 0;

    // parse 'str' and return result as string
    // 'str' must be written in CRF++'s input format
    virtual const char* parse(const char* str) = 0;

#ifndef SWIG
    // return parsed result as string
    virtual const char* toString() = 0;

    // return parsed result as string.
    // Result is saved in the buffer 'result', 'size' is the
    // size of the buffer. if failed, return NULL
    virtual const char* toString(char* result , size_t size) = 0;

    // parse 'str' and return parsed result.
    // You don't need to delete return value, but the buffer
    // is rewritten whenever you call parse method.
    // if failed, return NULL
    virtual const char* parse(const char *str, size_t size) = 0;

    // parse 'str' and return parsed result.
    // The result is stored in the buffer 'result'.
    // 'size2' is the size of the buffer. if failed, return NULL
    virtual const char* parse(const char *str, size_t size1,
                              char *result, size_t size2) = 0;
#endif
    // return internal error code as string
    virtual const char* what() = 0;

    virtual ~Tagger() {}
  };

  /* factory method */

  // create CRFPP::Tagger instance with parameters in argv[]
  // e.g, argv[] = {"CRF++", "-m", "model", "-v3"};

  CRFPP_DLL_EXTERN Tagger *createTagger(int argc, char **argv);

  // create CRFPP::Tagger instance with parameter in arg
  // e.g. arg = "-m model -v3";
  CRFPP_DLL_EXTERN Tagger *createTagger(const char *arg);

  // return error code of createTagger();
  CRFPP_DLL_EXTERN const char* getTaggerError();
}

#endif
#endif
