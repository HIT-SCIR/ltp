#ifndef __LTP_PARSER_SETTINGS_H__
#define __LTP_PARSER_SETTINGS_H__

#include <iostream>

namespace ltp {
namespace parser {

// define constant
const std::string ROOT_FORM     =   "RT#";
const std::string ROOT_LEMMA    =   "RT#";
const std::string ROOT_CPOSTAG  =   "RT#";
const std::string ROOT_POSTAG   =   "RT#";
const std::string ROOT_FEAT     =   "RT#";
const std::string ROOT_DEPREL   =   "RT#";

// weired things.
const std::string PRP           =   "PRP";
const std::string PRP2          =   "PRP2";
const std::string OOV           =   "-OOV-";

// none const
const std::string NONE_FORM     =   "##";
const std::string NONE_LEMMA    =   "##";
const std::string NONE_CPOSTAG  =   "##";
const std::string NONE_POSTAG   =   "##";
const std::string NONE_FEAT     =   "##";

const std::string FSEP          =   "-";

const double DOUBLE_POS_INF     =   1e20;
const double DOUBLE_NEG_INF     =   -1e20;
const double EPS                =   1e-10;

// span type const
const size_t CMP                =   0;
const size_t INCMP              =   1;
const size_t SIBSP              =   2;

enum {
  DEPU,       //  Unlabeled Standard Features
  DEPL,       //  Labeled Standard Features
  SIBU,       //  Unlabeled Sibling Features
  SIBL,       //  Labeled Sibling Features
  GRDU,       //  Unlabeled Grandchild Feature
  GRDL,       //  Labeled Grandchild Feature
  GRDSIBU,    //  Unlabeled Grand Sibling Features
  GRDSIBL,
  POSU,
  POSB,
};

}       //  end for namespace parser
}       //  end for namespace ltp
#endif  //  end for __LTP_PARSER_SETTINGS_H__

