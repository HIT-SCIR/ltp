#ifndef __LTP_FRAMEWORK_MODEL_H__
#define __LTP_FRAMEWORK_MODEL_H__

#include "framework/serializable.h"
#include "framework/parameter.h"
#include "framework/featurespace.h"
#include "utils/smartmap.hpp"

namespace ltp {
namespace framework {

class Model: public Serializable {
public:
  Model(const size_t& nr_feature_types): space(nr_feature_types) {}
  ~Model() {}

  /**
   * get number of labels;
   *
   *  @return   int   the number of labels
   */
  size_t num_labels(void) const { return labels.size(); }

  void save(const std::string& model_name, const Parameters::DumpOption& opt,
      std::ostream & ofs) const {
    save(model_name.c_str(), opt, ofs);
  }

  /**
   * save the model to a output stream
   *
   *  @param[in]  model_name  The name for the model.
   *  @param[in]  opt         The Parameters::DumpOption opt
   *  @param[out] ofs         The output stream.
   */
  void save(const char* model_name, const Parameters::DumpOption& opt,
      std::ostream & ofs) const {
    // write a signature into the file
    char chunk[128];
    strncpy(chunk, model_name, 128);

    ofs.write(chunk, 128);
    labels.dump(ofs);
    space.dump(ofs);
    param.dump(ofs, opt);
  }

  bool load(const std::string& model_name, std::istream& ifs) {
    return load(model_name.c_str(), ifs);
  }

  /**
   * load the model from an input stream
   *
   *  @param[in]  ifs   the input stream
   */
  bool load(const char* model_name, std::istream& ifs) {
    char chunk[128];
    ifs.read(chunk, 128);

    if (strcmp(chunk, model_name)) {
      return false;
    }
    if (!labels.load(ifs)) {
      return false;
    }
    if (!space.load(ifs)) {
      return false;
    }
    space.set_num_labels(labels.size());
    if (!param.load(ifs)) {
      return false;
    }

    return true;
  }
public:
  utility::IndexableSmartMap labels;  //! The labels.
  ViterbiFeatureSpace space;          //! The feature space.
  Parameters param;                   //! The parameters.
};


} //  namespace framework
} //  namespace ltp

#endif  //  end for __LTP_FRAMEWORK_MODEL_H__
