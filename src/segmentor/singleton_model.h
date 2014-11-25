#ifndef __LTP_SEGMENGOR_SINGLETON_MODEL_H__
#define __LTP_SEGMENGOR_SINGLETON_MODEL_H__

#include "segmentor/model.h"

namespace ltp {
namespace segmentor {
class SingletonModel {
public:
  static Model * get_model();
  static bool create_model(const char * model_file);
private:
  SingletonModel();
private:
  static Model * model;
};
}    //  end for namespace segmentor
}    //  end for namespace  ltp

#endif  //  end for __LTP_SEGMENGOR_SINGLETON_MODEL_H__
