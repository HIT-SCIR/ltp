#ifndef __LTP_SEGMENTOR_SEGMENTOR_FRONTEND_H__
#define __LTP_SEGMENTOR_SEGMENTOR_FRONTEND_H__

#include "framework/frontend.h"
#include "framework/decoder.h"
#include "segmentor/segmentor.h"
#include "segmentor/instance.h"
#include "segmentor/options.h"
#include "segmentor/decoder.h"

namespace ltp {
namespace segmentor {

class SegmentorFrontend: public Segmentor, public framework::Frontend {
protected:
  SegmentationViterbiDecoderWithMarginal decoder;     //! The decoder.
  framework::ViterbiFeatureContext ctx;  //! The decode context
  framework::ViterbiScoreMatrix scm;     //! The score matrix
  std::vector<const Model::lexicon_t*> lexicons;

  TrainOptions train_opt;
  TestOptions  test_opt;
  DumpOptions  dump_opt;

  size_t timestamp;

protected:
  struct Segmentation {
    bool is_partial;
    std::vector<std::string> words;
    PartialSegmentationConstrain con;
  };

  std::vector<Instance *> train_data_input;     //! The training data input.
  std::vector<Segmentation> train_data_output;  //! The training data output.

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
      bool evaluate,
      bool sequence_prob = false,
      bool marginal_prob = false);

  SegmentorFrontend(const std::string& model_file);

  ~SegmentorFrontend();

  void train(void);
  void test(void);
  void dump(void);

protected:

  virtual void extract_features(const Instance& inst, bool create);
  virtual void extract_features(const Instance& inst);
  virtual void calculate_scores(const Instance& inst, bool avg);
  virtual void collect_features(const Instance& inst);
  virtual void update(const Instance& inst, math::SparseVec& updated_features);

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
  void evaluate(double &p, double &r, double &f);

  virtual size_t get_timestamp() const;

  void increase_timestamp();

  virtual void setup_lexicons();

  virtual void clear_context();
};

} //  namespace segmentor
} //  namespace ltp

#endif  //  end for __LTP_SEGMENTOR_SEGMENTOR_FRONTEND_H__
