//
//  CRF++ -- Yet Another CRF toolkit
//
//  $Id: encoder.h 1588 2007-02-12 09:03:39Z taku $;
//
//  Copyright(C) 2005-2007 Taku Kudo <taku@chasen.org>
//
#ifndef CRFPP_ENCODER_H__
#define CRFPP_ENCODER_H__

#include "common.h"

namespace CRFPP {
  class Encoder {
  private:
    whatlog what_;
  public:
    enum { CRF_L2, CRF_L1, MIRA };
    bool learn(const char *, const char *, const char *, bool, size_t, size_t,
               double, double, unsigned short, unsigned short, int);
    bool convert(const char *, const char*);
    const char* what() { return what_.str(); }
  };
}
#endif
