#include "segmentor/singleton_model.h"
#include <fstream>

namespace ltp {
namespace segmentor {

Model * SingletonModel::model = NULL;
Model *
SingletonModel::get_model() {
  return model;
}
bool
SingletonModel::create_model(const char * model_file) {
  std::ifstream mfs(model_file, std::ifstream::binary);

  if (!mfs) {
    return false;
  }

  model = new Model;

  if (!model->load(mfs)) {
    delete model;
    model = 0;
    return false;
  }

  return true;
}
}    //  end for namespace segmentor
}    //  end for namespace ltp
