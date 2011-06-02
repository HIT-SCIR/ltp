//
//  CRF++ -- Yet Another CRF toolkit
//
//  $Id: libcrfpp.cpp 1587 2007-02-12 09:00:36Z taku $;
//
//  Copyright(C) 2005-2007 Taku Kudo <taku@chasen.org>
//
#ifdef HAVE_CONFIG_H
#ifdef WIN32
#include "config-win32.h"
#else
#include "config.h"
#endif
#endif

#include "crfpp.h"
#include <string>

#if defined(_WIN32) && !defined(__CYGWIN__)
#include <windows.h>
#endif

#define LIBCRFPP_ID 113212

namespace {
  std::string errorStr;
}

struct crfpp_t {
  int allocated;
  CRFPP::Tagger* ptr;
};

#if defined(_WIN32) && !defined(__CYGWIN__)
BOOL __stdcall DllMain(HINSTANCE hinst, DWORD dwReason, void*) {
  return TRUE;
}
#endif

crfpp_t* crfpp_new(int argc, char **argv) {
  crfpp_t *c = new crfpp_t;
  CRFPP::Tagger *ptr = CRFPP::createTagger(argc, argv);
  if (!c || !ptr) {
    delete c;
    delete ptr;
    errorStr = CRFPP::getTaggerError();
    return 0;
  }
  c->ptr = ptr;
  c->allocated = LIBCRFPP_ID;
  return c;
}

crfpp_t* crfpp_new2(char *arg) {
  crfpp_t *c = new crfpp_t;
  CRFPP::Tagger *ptr = CRFPP::createTagger(arg);
  if (!c || !ptr) {
    delete c;
    delete ptr;
    errorStr = CRFPP::getTaggerError();
    return 0;
  }
  c->ptr = ptr;
  c->allocated = LIBCRFPP_ID;
  return c;
}

const char* crfpp_strerror(crfpp_t *c) {
  if (!c || !c->allocated)
    return const_cast<char *> (errorStr.c_str());
  return c->ptr->what();
}

void crfpp_destroy(crfpp_t *c) {
  if (c && c->allocated) {
    delete c->ptr;
    delete c;
  }
  c = 0;
}

#define CRFPP_CHECK_FIRST_ARG(c, t)  \
if (!(c) || (c)->allocated != LIBCRFPP_ID) { \
  errorStr = "first argment seems to be invalid"; \
  return 0; \
} CRFPP::Tagger *(t) = (c)->ptr;

#define CRFPP_CHECK_FIRST_ARG_VOID(c, t)  \
if (!(c) || (c)->allocated != LIBCRFPP_ID) { \
  errorStr = "first argment seems to be invalid"; \
  return; \
} CRFPP::Tagger *(t) = (c)->ptr;

bool     crfpp_add2(crfpp_t* c, size_t s, const char **line) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->add(s, line);
}

bool     crfpp_add(crfpp_t* c, const char*s) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->add(s);
}

size_t   crfpp_size(crfpp_t* c) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->size();
}

size_t   crfpp_xsize(crfpp_t* c) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->xsize();
}

size_t   crfpp_result(crfpp_t* c, size_t i) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->result(i);
}

size_t   crfpp_answer(crfpp_t* c, size_t i) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->answer(i);
}

size_t   crfpp_y(crfpp_t* c, size_t i) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->y(i);
}

size_t   crfpp_ysize(crfpp_t* c) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->ysize();
}

double   crfpp_prob(crfpp_t* c, size_t i, size_t j) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->prob(i, j);
}

double   crfpp_prob2(crfpp_t* c, size_t i) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->prob(i);
}

double   crfpp_prob3(crfpp_t* c) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->prob();
}

double   crfpp_alpha(crfpp_t* c, size_t i, size_t j) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->alpha(i, j);
}

double   crfpp_beta(crfpp_t* c, size_t i, size_t j) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->beta(i, j);
}

double   crfpp_best_cost(crfpp_t* c, size_t i, size_t j) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->best_cost(i, j);
}

double   crfpp_emisstion_cost(crfpp_t* c, size_t i, size_t j) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->emission_cost(i, j);
}

const int* crfpp_emisstion_vector(crfpp_t* c, size_t i, size_t j) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->emission_vector(i, j);
}

double   crfpp_next_transition_cost(crfpp_t* c, size_t i, size_t j, size_t k) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->next_transition_cost(i, j, k);
}

double   crfpp_prev_transition_cost(crfpp_t* c, size_t i, size_t j, size_t k) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->next_transition_cost(i, j, k);
}

const  int* crfpp_next_transition_vector(crfpp_t* c, size_t i,
                                         size_t j, size_t k) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->next_transition_vector(i, j, k);
}

const int* crfpp_prev_transition_vector(crfpp_t* c, size_t i,
                                        size_t j, size_t k) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->next_transition_vector(i, j, k);
}

size_t crfpp_dsize(crfpp_t* c) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->dsize();
}

const float* crfpp_weight_vector(crfpp_t* c) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->weight_vector();
}

double   crfpp_Z(crfpp_t* c) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->Z();
}

bool     crfpp_parse(crfpp_t* c) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->parse();
}

bool     crfpp_empty(crfpp_t* c) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->empty();
}

bool     crfpp_clear(crfpp_t* c) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->clear();
}

bool     crfpp_next(crfpp_t* c) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->next();
}

const char*   crfpp_yname(crfpp_t* c, size_t i) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->yname(i);
}

const char*   crfpp_y2(crfpp_t* c, size_t i) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->y2(i);
}

const char*   crfpp_x(crfpp_t* c, size_t i, size_t j) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->x(i, j);
}

const char**  crfpp_x2(crfpp_t* c, size_t i) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->x(i);
}

const char*  crfpp_parse_tostr(crfpp_t* c, const char* str) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->parse(str);
}

const char*  crfpp_parse_tostr2(crfpp_t* c, const char* str, size_t len) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->parse(str, len);
}

const char*  crfpp_parse_tostr3(crfpp_t* c, const char* str,
                                size_t len, char *ostr, size_t len2) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->parse(str, len, ostr, len2);
}

const char*  crfpp_tostr(crfpp_t* c) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->toString();
}

const char*  crfpp_tostr2(crfpp_t* c, char *ostr, size_t len) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->toString(ostr, len);
}

void crfpp_set_vlevel(crfpp_t *c, unsigned int vlevel) {
  CRFPP_CHECK_FIRST_ARG_VOID(c, t);
  t->set_vlevel(vlevel);
}

unsigned int crfpp_vlevel(crfpp_t *c) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->vlevel();
}

void crfpp_set_cost_factor(crfpp_t *c, float cost_factor) {
  CRFPP_CHECK_FIRST_ARG_VOID(c, t);
  t->set_cost_factor(cost_factor);
}

float crfpp_cost_factor(crfpp_t *c) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->cost_factor();
}

void crfpp_set_nbest(crfpp_t *c, size_t nbest) {
  CRFPP_CHECK_FIRST_ARG_VOID(c, t);
  t->set_nbest(nbest);
}

size_t crfpp_nbest(crfpp_t *c) {
  CRFPP_CHECK_FIRST_ARG(c, t);
  return t->nbest();
}
