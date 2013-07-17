#ifndef __LTP_SEGMENTOR_SEGMENTOR_H__
#define __LTP_SEGMENTOR_SEGMENTOR_H__

#include "cfgparser.hpp"
#include "model.h"
#include "decoder.h"

namespace ltp {
namespace segmentor {

class Segmentor {
public:
    Segmentor(ltp::utility::ConfigParser & cfg);
    ~Segmentor() {}

    void run();

private:
    bool parse_cfg(ltp::utility::ConfigParser & cfg);
    bool read_instance( const char * file_name );
    void build_configuration(void);
    void build_feature_space(void);
 
    void extract_features(Instance * inst, bool create = false);
    /*
     * build words from tags for certain instance
     *
     *  @param[in/out]  inst    the instance
     *  @param[out]     words   the output words
     *  @param[in]      tagsidx the index of tags
     *  @param[in]      begtag0 first of the word begin tag
     *  @param[in]      begtag1 second of the word begin tag
     */
    void build_words(Instance * inst, 
            const std::vector<int> & tagsidx,
            std::vector<std::string> & words,
            int beg_tag0,
            int beg_tag1 = -1);

    /*
     * cache all the score for the certain instance.
     *
     *  @param[in/out]  inst    the instance
     *  @param[in]      use_avg use to specify use average parameter
     */
    void calculate_scores(Instance * inst, bool use_avg);

    /*
     * collect feature when given the tags index
     *
     *  @param[in]      inst    the instance
     *  @param[in]      tagsidx the tags index
     *  @param[out]     vec     the output sparse vector
     */
    void collect_features(Instance * inst, 
            const std::vector<int> & tagsidx, 
            ltp::math::SparseVec & vec);

    /*
     * the training process
     */
    void train(void);

    /*
     * the evaluating process
     */
    void evaluate(void);

    /*
     * the testing process
     */
    void test(void);
private:
    bool    __TRAIN__;
    bool    __TEST__;

private:
    std::vector< Instance * > train_dat;
    Model * model;
    Decoder * decoder;
};

}       //  end for namespace segmentor
}       //  end for namespace ltp

#endif  //  end for __LTP_SEGMENTOR_SEGMENTOR_H__
