#ifndef __PARSER_H__
#define __PARSER_H__

#include <iostream>

#include "instance.h"
#include "model.h"
#include "extractor.h"
#include "decoder.h"

#include "cfgparser.hpp"
#include "logging.hpp"
#include "time.hpp"

#include "debug.h"

using namespace std;
using namespace ltp::utility;
using namespace ltp::strutils;

namespace ltp {
namespace parser {

class Parser{

/* Parser Options */
private:
    bool            __TRAIN__;
    bool            __TEST__;

public:
    Parser();

    Parser( ConfigParser& cfg );

    ~Parser();

    bool operator! () const {
        return _valid;
    }

    void run() {
        /* running train process */
        if (__TRAIN__) {
            train();
        }

        /* running test process */
        if (__TEST__) {
            test();
        }
    }

private:
    bool _valid; /* indicating if the parser is valid */
    vector<Instance *> train_dat;

protected:
    Model * model;
    Decoder * decoder;
private:
    void init_opt();

    bool parse_cfg(ConfigParser& cfg);

    bool read_instances(const char * filename, vector<Instance *>& dat);

    void build_feature_space(void);

    void build_feature_space_truncate(Model * m);

    void build_configuration(void);

    void extract_features(vector<Instance *>& dat);

    void build_gold_features(void);

    void train(void);

    void evaluate(void);

    void test(void);

    void collect_unlabeled_features_of_one_instance(Instance * inst,
            const vector<int> & heads,
            SparseVec & vec);

    void collect_labeled_features_of_one_instance(Instance * inst,
            const vector<int> & heads,
            const vector<int> & deprelsidx,
            SparseVec & vec);

    void collect_features_of_one_instance(Instance * inst, 
            bool gold = false);

    /*
     * perform the feature space truncation
     *
     *  @param[out]     new_model   the pointer to the new model
     *  @param[in]      gid         the index of the group.
     */
    void copy_featurespace(Model * new_model, int gid);

    /*
     * perform the parameter truncation.
     *
     *  @param[in/out]  new_model   the pointer to the output model
     *  @param[in]      gid         the index of the group.
     */
    void copy_parameters(Model * new_model, int gid);

    /*
     * perform model truncation and return a new model
     *
     *  @return     Model *     pointer to the new model
     */
    Model * truncate();

protected:
    Decoder * build_decoder(void);
    void extract_features(Instance * inst);

    void calculate_score(Instance * inst, const Parameters& param, bool use_avg = false);

};  //  end for class Parser
}   //  end for namespace parser
}   //  end for namespace ltp

#endif  // end for __PARSER_H__
