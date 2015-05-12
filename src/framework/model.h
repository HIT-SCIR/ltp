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
  inline int num_labels(void) const {
    return labels.size();
  }

  /**
   * save the model to a output stream
   *
   *  @param[in]  model_name  The name for the model.
   *  @param[in]  opt         The Parameters::DumpOption opt
   *  @param[out] ofs         The output stream.
   */
  void save(const char* model_name, const Parameters::DumpOption& opt,
      std::ostream & ofs) {
    // write a signature into the file
    char chunk[128];
    strncpy(chunk, model_name, 128);

    ofs.write(chunk, 128);
    int off = ofs.tellp();
    size_t labels_offset    = 0;
    size_t feature_offset   = 0;
    size_t parameter_offset = 0;

    write_uint(ofs, 0); //  the label offset
    write_uint(ofs, 0); //  the features offset
    write_uint(ofs, 0); //  the parameter offset

    labels_offset = ofs.tellp();
    labels.dump(ofs);

    feature_offset = ofs.tellp();
    space.dump(ofs);

    parameter_offset = ofs.tellp();
    param.dump(ofs, opt);

    ofs.seekp(off);
    write_uint(ofs, labels_offset);
    write_uint(ofs, feature_offset);
    write_uint(ofs, parameter_offset);
  }

  /**
   * load the model from an input stream
   *
   *  @param[in]  ifs   the input stream
   */
  bool load(const char* model_name, std::istream & ifs) {
    char chunk[128];
    ifs.read(chunk, 128);

    if (strcmp(chunk, model_name)) {
      return false;
    }

    size_t labels_offset = read_uint(ifs);
    size_t feature_offset = read_uint(ifs);
    size_t parameter_offset = read_uint(ifs);

    ifs.seekg(labels_offset);
    if (!labels.load(ifs)) {
      return false;
    }

    ifs.seekg(feature_offset);
    if (!space.load(ifs)) {
      return false;
    }
    space.set_num_labels(labels.size());

    ifs.seekg(parameter_offset);
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
