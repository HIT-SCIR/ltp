#ifndef __LTP_POSTAGGER_POSTAGGER_H__
#define __LTP_POSTAGGER_POSTAGGER_H__

#include "cfgparser.hpp"
#include "model.h"
#include "decoder.h"

namespace ltp {
namespace postagger {

class Postagger {
public:
    Postagger();
    Postagger(ltp::utility::ConfigParser & cfg);
    ~Postagger();

    /*
     * main function of running postagging.
     */
    void run();

private:
    bool parse_cfg(ltp::utility::ConfigParser & cfg);

    /*
     * read instance from the file, and store the instances
     * in train data.
     *
     *  @param[in]  file_name   the file name
     */
    bool read_instance( const char * file_name );

    /*
     * build postags dictionary
     */
    void build_configuration(void);

    /*
     * build the feature space
     */
    void build_feature_space(void);

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

    /*
     * the dump model process
     */
    void dump(void);

    /*
     * do feature trauncation on the model. create a model duplation
     * on the model and return their
     *
     *  @return Model the duplication of the model
     */
    Model * truncate(void);
protected:
    /*
     * build labels string for the instance
     *
     *  @param[in]  inst        the instance
     *  @param[out] tags        the list of tags string
     */
    void build_labels(Instance * inst, std::vector<std::string> & tags);

    /*
     * extract feature from the instance. If create handler is configured,
     *
     *  @param[in]  inst        the instance
     *  @param[in]  create      use to specify create process
     */
    void extract_features(Instance * inst, bool create = false);

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

private:
    bool    __TRAIN__;
    bool    __TEST__;
    bool    __DUMP__;

private:
    std::vector< Instance * > train_dat;

protected:
    Model * model;
    Decoder * decoder;
};

}       //  end for namespace postagger
}       //  end for namespace ltp

#endif  //  end for __LTP_POSTAGGER_SEGMENTOR_H__
