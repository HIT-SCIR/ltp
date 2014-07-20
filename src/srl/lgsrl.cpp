/**
 * Training and testing suite for Semantic Role Labeling
 *
 * Feature:
 *  -> Train PRG model (predicate recognition)
 *  -> Train SRL model (semantic role labeling)
 *  -> Test PRG+SRL    (pipeline)
 *
 * Author: jiangfeng
 * Date  : 2013.8.23
 *
 */


#include <vector>
#include <string>
#include <fstream>

#include "Corpus.h"
#include "Configuration.h"
#include "Sentence.h"
#include "FeatureExtractor.h"
#include "GetInstance.h"
#include "maxent.h"
#include "options.h"
#include "cfgparser.hpp"
#include "logging.hpp"
#include "strutils.hpp"
#include "SRL_DLL.h"

using namespace std;
using namespace ltp::utility;
using namespace ltp::strutils;
using namespace maxent;

TrainOptions train_opt;
TestOptions  test_opt;

ME_Parameter me_prg_param;
ME_Parameter me_srl_param;

bool __TRAIN_PRG__ = false;
bool __TRAIN_SRL__ = false;
bool __TEST__      = false;

void usage(void) {
    cerr << "srltrain - Training suite for semantic role labeling" << endl;
    cerr << "Copyright (C) 2012-2014 HIT-SCIR" << endl;
    cerr << endl;
    cerr << "usage: ./srltrain <config_file>" << endl;
    cerr << endl;
}

bool parse_cfg(ConfigParser & cfg)
{
    string strbuf;
    int    intbuf;
    double dblbuf;

    if (cfg.has_section("train-srl")) {
        TRACE_LOG("SRL training mode specified");

        __TRAIN_SRL__ = true;

        if (cfg.get("train-srl", "srl-train-file", strbuf)) {
            train_opt.srl_train_file = strbuf;
        } else {
            ERROR_LOG("srl-train-file config item is not found");
            return false;
        }

        if (cfg.get("train-srl", "core-config", strbuf)) {
            train_opt.core_config = strbuf;
        } else {
            ERROR_LOG("core-config config item is not found");
            return false;
        }

        if (cfg.get("train-srl", "srl-config", strbuf)) {
            train_opt.srl_config = strbuf;
        } else {
            ERROR_LOG("srl-config config item is not found");
            return false;
        }

        if (cfg.get("train-srl", "srl-feature-dir", strbuf)) {
            train_opt.srl_feature_dir = strbuf;
        } else {
            ERROR_LOG("[SRL] srl-feature-dir config item is not found");
            return false;
        }

        if (cfg.get("train-srl", "srl-instance-file", strbuf)) {
            train_opt.srl_instance_file = strbuf;
        } else {
            ERROR_LOG("[SRL] srl-instance-file config item is not found");
            return false;
        }

        if (cfg.get("train-srl", "srl-model-file", strbuf)) {
            train_opt.srl_model_file = strbuf;
        } else {
            ERROR_LOG("[SRL] srl-model-file config item is not found");
            return false;
        }

        if (cfg.get("train-srl", "dst-config-dir", strbuf)) {
            train_opt.dst_config_dir = strbuf;
        } else {
            ERROR_LOG("[SRL] dst_config_dir config item is not found");
            return false;
        }

        if (cfg.get_integer("train-srl", "solver-type", intbuf)) {
            switch (intbuf) {
                case 0: me_srl_param.solver_type = L1_OWLQN; break;
                case 1: me_srl_param.solver_type = L1_SGD;   break;
                case 2: me_srl_param.solver_type = L2_LBFGS; break;
                default:
                    ERROR_LOG("Unsupported solver [%d]", intbuf);
                    break;
            }
        }

        if (cfg.get_float("train-srl", "l1-reg", dblbuf)) {
            me_srl_param.l1_reg = dblbuf;
        }

        if (cfg.get_float("train-srl", "l2-reg", dblbuf)) {
            me_srl_param.l2_reg = dblbuf;
        }

        if (cfg.get_integer("train-srl", "sgd-iter", intbuf)) {
            me_srl_param.sgd_iter = intbuf;
        }

        if (cfg.get_float("train-srl", "sgd-eta0", dblbuf)) {
            me_srl_param.sgd_eta0 = dblbuf;
        }

        if (cfg.get_float("train-srl", "sgd-alpha", dblbuf)) {
            me_srl_param.sgd_alpha = dblbuf;
        }

        if (cfg.get_integer("train-srl", "nheldout", intbuf)) {
            me_srl_param.nheldout = intbuf;
        }
    }

    if (cfg.has_section("train-prg")) {
        TRACE_LOG("PRG training model specified");

        __TRAIN_PRG__ = true;

        if (cfg.get("train-prg", "prg-train-file", strbuf)) {
            train_opt.prg_train_file = strbuf;
        } else {
            ERROR_LOG("prg-train-file config item is not found");
            return false;
        }

        if (cfg.get("train-prg", "core-config", strbuf)) {
            train_opt.core_config = strbuf;
        } else {
            ERROR_LOG("core-config config item is not found");
            return false;
        }

        if (cfg.get("train-prg", "prg-instance-file", strbuf)) {
            train_opt.prg_instance_file = strbuf;
        } else {
            ERROR_LOG("[PRG] prg-instance-file config item is not found");
            return false;
        }

        if (cfg.get("train-prg", "prg-model-file", strbuf)) {
            train_opt.prg_model_file = strbuf;
        } else {
            ERROR_LOG("[PRG] prg-model-file config item is not found");
            return false;
        }

        if (cfg.get("train-prg", "dst-config-dir", strbuf)) {
            train_opt.dst_config_dir = strbuf;
        } else {
            ERROR_LOG("[PRG] dst_config_dir config item is not found");
            return false;
        }

        if (cfg.get_integer("train-prg", "solver-type", intbuf)) {
            switch (intbuf) {
                case 0: me_prg_param.solver_type = L1_OWLQN; break;
                case 1: me_prg_param.solver_type = L1_SGD;   break;
                case 2: me_prg_param.solver_type = L2_LBFGS; break;
                default:
                    ERROR_LOG("Unsupported solver [%d]", intbuf);
                    break;
            }

        }

        if (cfg.get_float("train-prg", "l1-reg", dblbuf)) {
            me_prg_param.l1_reg = dblbuf;
        }

        if (cfg.get_float("train-prg", "l2-reg", dblbuf)) {
            me_prg_param.l2_reg = dblbuf;
        }

        if (cfg.get_integer("train-prg", "sgd-iter", intbuf)) {
            me_prg_param.sgd_iter = intbuf;
        }

        if (cfg.get_float("train-prg", "sgd-eta0", dblbuf)) {
            me_prg_param.sgd_eta0 = dblbuf;
        }

        if (cfg.get_float("train-prg", "sgd-alpha", dblbuf)) {
            me_prg_param.sgd_alpha = dblbuf;
        }

        if (cfg.get_integer("train-prg", "nheldout", intbuf)) {
            me_prg_param.nheldout = intbuf;
        }
    }

    if (cfg.has_section("test")) {
        TRACE_LOG("PRG-SRL testing specified");

        __TEST__ = true;

        if (cfg.get("test", "test-file", strbuf)) {
            test_opt.test_file = strbuf;
        } else {
            ERROR_LOG("test-file config item is not found");
            return false;
        }

        if (cfg.get("test", "config-dir", strbuf)) {
            test_opt.config_dir = strbuf;
        } else {
            ERROR_LOG("config-dir config item is not found");
            return false;
        }

        if (cfg.get("test", "output-file", strbuf)) {
            test_opt.output_file = strbuf;
        } else {
            ERROR_LOG("output-file config item is not found");
            return false;
        }
    }

    return true;
}

bool copy_cfg(const string & src_cfg,
            const string & dst_cfg)
{
    ifstream fsrc(src_cfg.c_str());
    ofstream fdst(dst_cfg.c_str());

    if (!fdst)
    {
        ERROR_LOG("Cannot open [%s]", dst_cfg.c_str());
        return false;
    }

    string line;
    while (getline(fsrc, line))
        fdst << line << endl;

    fsrc.close();
    fdst.close();

    return true;
}

bool collect_prg_instances()
{
    Configuration       configuration(train_opt.core_config);
    FeatureExtractor    feature_extractor(configuration);
    FeatureCollection   feature_collection;
    vector<int>         feature_numbers;
    vector<string>      feature_prefixes;

    ofstream inst_stream(train_opt.prg_instance_file.c_str());
    if (!inst_stream) {
        ERROR_LOG("[PRG] cannot open instance file:[%s] for writing",
                train_opt.prg_instance_file.c_str());
        return false;
    }

    const vector<string> & feat_set =
        configuration.get_pred_recog_config().get_feature_names();

    for (size_t i = 0; i < feat_set.size(); ++i) {
        const string& feature_name = feat_set[i];
        const int feature_number =
            feature_collection.get_feature_number(feature_name);
        const string& feature_prefix =
            feature_collection.get_feature_prefix(feature_number);

        feature_numbers.push_back(feature_number);
        feature_prefixes.push_back(feature_prefix);
    }

    Corpus corpus(train_opt.prg_train_file);
    vector<string> lines;
    Sentence sentence;

    size_t sentence_count = 0;
    while (corpus.get_next_block(lines)) {
        ++sentence_count;

        sentence.from_corpus_block(lines);
        const size_t row_count = sentence.get_row_count();

        feature_extractor.set_target_sentence(sentence);
        feature_extractor.calc_node_features();

        vector<vector<string> > vct_feature_values;
        for (size_t i = 0; i < feature_numbers.size(); ++i) {
            vector<string> feature_values;

            const int feature_number = feature_numbers[i];
            const string& feature_prefix = feature_prefixes[i];
            bool feature_empty_flag = false;
            try {
                feature_extractor.get_feature_for_rows(
                        feature_number, feature_values);
            } catch (...) {
                feature_empty_flag = true;
            }

            if (feature_empty_flag) {
                feature_values.clear();
                for (size_t row = 1; row <= row_count; ++row)
                    feature_values.push_back("");
            }
            vct_feature_values.push_back(feature_values);
        }

        for (size_t row = 1; row <= row_count; ++row) {
            inst_stream << ((sentence.get_FILLPRED(row) == "Y") ? 'Y' : 'N');
            for (size_t i = 0; i < feature_numbers.size(); ++i) {
                inst_stream << " " << feature_prefixes[i]
                    << "@" << vct_feature_values[i][row];
            }
            inst_stream << endl;
        }
    }

    inst_stream.close();
    return true;
}

bool collect_srl_instances()
{
    Configuration       configuration(train_opt.core_config);
    FeatureExtractor    feature_extractor(configuration);
    FeatureCollection   feature_collection;
    vector<int>         feature_numbers;
    vector<string>      feature_prefixes;

    ofstream output_streams[TOTAL_FEATURE];
    ofstream label_stream;

    const vector<string> & feat_set =
        configuration.get_argu_config().get_feature_names();
    feature_numbers.clear();
    feature_prefixes.clear();

    for (size_t i = 0; i < feat_set.size(); ++i) {
        const string& feature_name = feat_set[i];
        const int feature_number
            = feature_collection.get_feature_number(feature_name);
        const string& feature_prefix
            = feature_collection.get_feature_prefix(feature_number);

        feature_numbers.push_back(feature_number);
        feature_prefixes.push_back(feature_prefix);

        string filename = train_opt.srl_feature_dir + "/" + feature_name;
        output_streams[feature_number].open(filename.c_str());

        if (!output_streams[feature_number]) {
            ERROR_LOG("cannot open feature output file: [%s]", feature_name.c_str());
            return false;
        }
    }

    string label_filename = train_opt.srl_feature_dir + "/labels";
    label_stream.open(label_filename.c_str());
    if (!label_stream) {
        ERROR_LOG("can't open labels file");
        return false;
    }

    Corpus corpus(train_opt.srl_train_file);
    vector<string> lines;
    Sentence sentence;

    size_t sentence_count = 0;
    while (corpus.get_next_block(lines))
    {
        ++sentence_count;

        sentence.from_corpus_block(lines);
        const size_t predicate_count = sentence.get_predicates().size();
        const size_t row_count       = sentence.get_row_count();

        feature_extractor.set_target_sentence(sentence);
        vector<string> feature_values;

        for (size_t predicate_index = 0; predicate_index < predicate_count;
                ++predicate_index) {   // loop for each predicate
            feature_extractor.calc_features(predicate_index);

            for (size_t i = 0; i < feature_numbers.size(); ++i) {
                const int feature_number     = feature_numbers[i];
                const string& feature_prefix = feature_prefixes[i];
                bool feature_empty_flag      = false;
                try {
                    feature_extractor.get_feature_for_rows(
                            feature_number, feature_values);
                }
                catch(...) {
                    feature_empty_flag = true;
                }

                if (feature_empty_flag) {
                    for (size_t row = 1; row <= row_count; ++row)
                        output_streams[feature_number]<<endl;
                }
                else {
                    for (size_t row = 1; row <= row_count; ++row) {
                        if (feature_prefix == "PFEATNULL"
                                && feature_values[row] == "")
                            output_streams[feature_number] << endl;
                        else
                            output_streams[feature_number]
                                << feature_prefix
                                << "@"
                                << feature_values[row]
                                << endl;
                    }
                }
                output_streams[feature_number]<<endl;
            }

            for (size_t row = 1; row <= row_count; ++row) {
                const string &argument =
                    sentence.get_argument(predicate_index, row);
                if (argument.empty())
                    label_stream << "NULL" << endl;
                else
                    label_stream<< argument << endl;
            }
            label_stream<<endl;
        }
    }

    for (size_t i = 0; i < TOTAL_FEATURE; ++i)
        output_streams[i].close();
    label_stream.close();

    GetInstance get_instance(configuration);
    get_instance.generate_argu_instance(
            train_opt.srl_feature_dir,
            train_opt.srl_config,
            train_opt.srl_instance_file);

    return true;
}

bool train(ME_Model & model,
        const string & input,
        const string & model_path)
{
    ifstream ifile(input.c_str());

    if (!ifile) {
        ERROR_LOG("Cannot open [%s]", input.c_str());
        return false;
    }

    string line;
    while (getline(ifile, line)) {
        vector<string> vs = split(line);
        ME_Sample mes(vs, true);
        model.add_training_sample(mes);
    }

    model.train();
    model.save(model_path);

    return true;
}

// unused
bool prg_predict()
{
    string core_config = test_opt.config_dir + "./Chinese.xml";
    string model_file  = test_opt.config_dir + "./prg.model";
    Configuration   configuration(core_config);
    ME_Model        prg_model(model_file);
    Corpus          corpus(test_opt.test_file);
    ofstream        output(test_opt.output_file.c_str());

    FeatureExtractor  feature_extractor(configuration);
    FeatureCollection feature_collection;
    vector<int>       feature_numbers;
    vector<string>    feature_prefixes;

    const vector<string>& feat_set =
        configuration.get_pred_recog_config().get_feature_names();
    for (size_t i = 0; i < feat_set.size(); ++i) {
        const string& feature_name = feat_set[i];
        const int feature_number =
            feature_collection.get_feature_number(feature_name);
        const string& feature_prefix =
            feature_collection.get_feature_prefix(feature_number);

        feature_numbers.push_back(feature_number);
        feature_prefixes.push_back(feature_prefix);
    }

    if (!output) {
        ERROR_LOG("Cannot open [%s]", test_opt.output_file.c_str());
        return false;
    }

    vector<string> corpus_lines;
    Sentence sentence;

    // for each sentence
    size_t sentence_count = 0;
    while (corpus.get_next_block(corpus_lines)) {
        ++sentence_count;

        sentence.from_corpus_block(corpus_lines);
        const size_t row_count = sentence.get_row_count();

        feature_extractor.set_target_sentence(sentence);
        feature_extractor.calc_node_features();

        vector< vector<string> > vct_feature_values;
        for (size_t i = 0; i < feature_numbers.size(); ++i) {
            vector<string> feature_values;

            const int feature_number = feature_numbers[i];
            const string& feature_prefix = feature_prefixes[i];
            bool feature_empty_flag = false;
            try {
                feature_extractor.get_feature_for_rows(
                        feature_number, feature_values);
            } catch (...) {
                feature_empty_flag = true;
            }

            if (feature_empty_flag) {
                feature_values.clear();
                for (size_t row = 1; row <= row_count; ++row) {
                    feature_values.push_back("");
                }
            }
            vct_feature_values.push_back(feature_values);
        }

        vector<size_t> predicate_rows;
        for (size_t row = 1; row <= row_count; ++row) {
            vector< pair<string, double> > outcome;
            vector<string> instance;

            for (size_t i = 0; i < feature_numbers.size(); ++i) {
                string feature =
                    feature_prefixes[i]
                    + "@"
                    + vct_feature_values[i][row];
                instance.push_back(feature);
            }

            ME_Sample mes(instance);
            prg_model.predict(mes, outcome);
            if (outcome[0].first == "Y")
                predicate_rows.push_back(row);
        }
        sentence.set_predicates(predicate_rows);

        output << sentence.to_corpus_block() << endl;
    }

    return true;
}

bool predict()
{
    typedef pair<const char*, pair<int, int> > Argu;
    typedef pair<int, vector<Argu> > ArgusForOnePredicate;
    typedef vector<ArgusForOnePredicate> ArgusForPredicates;

    SRL_LoadResource(test_opt.config_dir);

    ofstream output(test_opt.output_file.c_str());
    if (!output) {
        ERROR_LOG("Failed to open [%s]",
                test_opt.output_file.c_str());
        return false;
    }

    Corpus corpus(test_opt.test_file);
    vector<string> corpus_lines;
    Sentence sentence;

    while (corpus.get_next_block(corpus_lines)) {
        sentence.from_corpus_block(corpus_lines);
        const size_t row_count = sentence.get_row_count();

        vector<string> words, poss, nes;
        vector< pair<int, string> > parses;
        for (size_t i = 1; i <= row_count; ++i) {

            words.push_back(sentence.get_FORM(i));
            poss.push_back(sentence.get_PPOS(i));
            nes.push_back("O"); // unused feature

            int phead = sentence.get_PHEAD(i);
            string pdeprel = sentence.get_PDEPREL(i);
            parses.push_back(make_pair(phead-1, pdeprel));
        }

        ArgusForPredicates srl_result;
        SRL(words, poss, nes, parses, srl_result);

        vector<size_t> predicate_rows;
        for (size_t i = 0; i < srl_result.size(); ++i) {

            ArgusForOnePredicate argus = srl_result[i];
            size_t predicate_row = argus.first + 1; // starts from 1
            predicate_rows.push_back(predicate_row);

        }
        sentence.set_predicates(predicate_rows);    // make room for arguments

        for (size_t i = 0; i < srl_result.size(); ++i) {

            ArgusForOnePredicate argus = srl_result[i];

            for (size_t j = 0; j < argus.second.size(); ++j) {
                Argu argu = argus.second[j];
                sentence.set_argument(i, argu.second.first+1, argu.first);
            }
        }

        output << sentence.to_corpus_block() << endl;
    }

    output.close();
    SRL_ReleaseResource();

    return true;
}

int main(int argc, char *argv[])
{
    /*
     * All params are defined in config file
     *
     * params: path-Chinese.xml
     * params: path-srl.cfg
     * params: path-corpus
     * params: path-feature folder
     * params: path-instances
     *
     * params: option-maxent-solver_type
     * params: option-maxent-reg_coefficient
     * params: option-maxent-sgd_iter (recommend: default)
     * params: option-maxent-sgd_eta0 (recommend: default)
     * params: option-maxent-sgd_alpha(recommend: default)
     * params: option-maxent-heldout
     *
     */
    if (argc < 2) {
        usage(); return -1;
    }

    ConfigParser cfg(argv[1]);

    if (!cfg) {
        ERROR_LOG("Failed to parse config file");
        return -1;
    }

    parse_cfg(cfg);

    if (__TRAIN_PRG__) {

        // collect training instances for PRG training
        TRACE_LOG("Collecting instances for PRG training");
        if (!collect_prg_instances()) {
            ERROR_LOG("Failed collect prg instances");
            return -1;
        }

        // training PRG model
        ME_Model prg_model(me_prg_param);
        TRACE_LOG("Training PRG Model");
        train(prg_model,
                train_opt.prg_instance_file,
                train_opt.prg_model_file);
        copy_cfg(train_opt.core_config,
                train_opt.dst_config_dir + "/Chinese.xml");
    }

    if (__TRAIN_SRL__) {
        // collect training instances for SRL training
        TRACE_LOG("Collecting instances for SRL training");
        if (!collect_srl_instances()) {
            ERROR_LOG("Failed collect srl instances");
            return -1;
        }

        // training SRL model
        ME_Model srl_model(me_srl_param);
        TRACE_LOG("Training SRL Model");
        train(srl_model,
                train_opt.srl_instance_file,
                train_opt.srl_model_file);
        copy_cfg(train_opt.srl_config,
                train_opt.dst_config_dir + "/srl.cfg");
    }

    if (__TEST__) {
        TRACE_LOG("Predicting [%s]", test_opt.test_file.c_str());
        if (!predict()) {
            TRACE_LOG("Failed predicting [%s]",
                    test_opt.test_file.c_str());
            return -1;
        }
        TRACE_LOG("Output to [%s]", test_opt.output_file.c_str());
    }

    return 0;
}

