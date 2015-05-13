#ifndef __LTP_SEGMENTOR_SEGMENTOR_FRONTEND_H__
#define __LTP_SEGMENTOR_SEGMENTOR_FRONTEND_H__

#include "framework/frontend.h"
#include "framework/decoder.h"
#include "segmentor/segmentor.h"
#include "segmentor/instance.h"
#include "segmentor/options.h"

namespace ltp {
namespace segmentor {

class SegmentorFrontend: public Segmentor, framework::Frontend {
private:
  framework::ViterbiDecoder* decoder;     //! The decoder.
  framework::ViterbiFeatureContext* ctx;  //! The decode context
  framework::ViterbiScoreMatrix* scm;     //! The score matrix
  std::vector<Instance *> train_dat;      //! The training data.

  TrainOptions train_opt;
  TestOptions  test_opt;
  DumpOptions  dump_opt;

  static const std::string model_header;

  size_t timestamp;
public:
  SegmentorFrontend(const std::string& reference_file,
      const std::string& holdout_file,
      const std::string& model_name,
      const std::string& algorithm,
      const size_t& max_iter,
      const size_t& rare_feature_threshold,
      bool dump_model_details);

  SegmentorFrontend(const std::string& input_file,
      const std::string& model_file,
      bool evaluate);

  SegmentorFrontend(const std::string& model_file);

  ~SegmentorFrontend();

  void train(void);
  void test(void);
  void dump(void);

private:

  /**
   * Read instances from file and store them in train_dat
   *
   *  @param[in]  file_name   the filename
   *  @return     bool        true on success, otherwise false
   */
  bool read_instance( const char * file_name );


  /**
   * Build configuration before model training. Three things are done
   * during the building configuration pharse:
   *
   *  1. Build tag sets;
   *  2. Collect internal word map;
   *  3. Record word frequency.
   */
  void build_configuration(void);


  /**
   * Build feature space.
   */
  void build_feature_space(void);


  /**
   * The main evaluating process.
   *
   *  @param[out]  p   The precise
   *  @param[out]  r   The recall
   *  @param[out]  f   The F-score
   */
  virtual void evaluate(double &p, double &r, double &f);

  void set_timestamp(const size_t& ts);

  size_t get_timestamp() const;

  void increase_timestamp();
};

} //  namespace segmentor
} //  namespace ltp

#endif  //  end for __LTP_SEGMENTOR_SEGMENTOR_FRONTEND_H__
