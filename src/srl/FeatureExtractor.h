/*
 * File Name     : FeatureExtractor.h
 * Author        : msmouse
 * Create Time   : 2006-12-31
 * Project Name  : NewSRLBaseLine
 *
 * Updated by    : jiangfeng
 * Update Time   : 2013-08-21
 */


#ifndef _FEATURE_EXTRACTOR_H_
#define _FEATURE_EXTRACTOR_H_

#include "boost/function.hpp"
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <bitset>
#include <tree.hh>
#include <set>
#include <map>
#include "Sentence.h"
#include "Configuration.h"

class FeatureExtractor;

/* a boost::function is a wraper for either a function pointer or function 
 *   object with the specified interface
 * a FeatureFunction is a member function of FeatureExtractor, with a parameter
 *   of the type size_t (the row number in a sentence)
 */
typedef boost::function<void (FeatureExtractor*, size_t)> FeatureFunction;


// type of a feature
enum FEAT_TYPE
{
    FEAT_TYPE_PRED,         // predicate feature (related to the predicate itself only)
    FEAT_TYPE_NODE,         // predicate-independent feature (related to a node only)
    FEAT_TYPE_NODE_VS_PRED, /* predicate-dependent feature
                             * (related to the relationship between the node and the predicate)
                             */
    FEAT_TYPE_UNKNOWN       // unknown feature type, usually not used, usually causing a exception
};


// feature numbers, each relating to a feature name, used internally, for the sake of efficiency
enum FEAT_NUM 
{
    FEAT_DEPREL,              // the dep-relation name
    FEAT_HEADWORD_POS,        // head word POS
    FEAT_DEPWORD_POS,         // dep word POS
    FEAT_HEADWORD,            // headword
    FEAT_DEPWORD,             // depword
    FEAT_HEADWORD_LEMMA,      // head word lemma
    FEAT_DEPWORD_LEMMA,       // dep word lemma
    FEAT_FIRST_WORD,          // first word in the subtree
    FEAT_FIRST_POS,           // first POS in the subtree
    FEAT_FIRST_LEMMA,         // first word lemma
    FEAT_LAST_WORD,           // last word in the subtree
    FEAT_LAST_POS,            // last POS in the subtree
    FEAT_LAST_LEMMA,          // last word lemma
    FEAT_POS_PATTERN,
    // first-pos + inner POS's (duplicated reduced) + last-pos
    // see hjliu's BegEndPosPattern in the paper
    FEAT_CHD_POS,             // pos pattern for children
    FEAT_CHD_POS_NDUP,        // (no duplicate)
    FEAT_CHD_REL,             // relation pattern for children
    FEAT_CHD_REL_NDUP,        // (no duplicate)
    FEAT_SIB_POS,             // pos pattern for siblings 
    FEAT_SIB_POS_NDUP,        // (no duplicate)
    FEAT_SIB_REL,             // relation pattern for siblings
    FEAT_SIB_REL_NDUP,        // (no duplicate)

    FEAT_HAS_SV,              // whether has a Support Verb
    FEAT_PRED_CHD_POS,        // pos pattern for predicate children
    FEAT_PRED_CHD_POS_NDUP,   // (no duplicate)
    FEAT_PRED_CHD_REL,        // relation pattern for predicate children
    FEAT_PRED_CHD_REL_NDUP,   // (no duplicate)
    FEAT_PRED_SIB_POS,        // pos pattern for predicate siblings 
    FEAT_PRED_SIB_POS_NDUP,   // (no duplicate)
    FEAT_PRED_SIB_REL,        // relation pattern for predicate siblings
    FEAT_PRED_SIB_REL_NDUP,   // (no duplicate)
    FEAT_PRED_LEMMA,          // predicate lemma
    FEAT_PREDICATE,           // predicate itself
    FEAT_PRED_SENSE,          //  predicate lemma + sense


    FEAT_PATH,                // the path from the node to the predicate
    FEAT_UP_PATH,             // the path from node to common parent
    FEAT_REL_PATH,            // relations along the path
    FEAT_UP_REL_PATH,         // relations along the half path
    FEAT_PATH_LENGTH,         // length of the feature "path"
    FEAT_UP_PATH_LEN,         //
    FEAT_DOWN_PATH_LEN,       //
    FEAT_DESC_OF_PD,          // whether is a descendant of the predicate
    FEAT_POSITION,            // before or after the predicate
    FEAT_PRED_FAMILYSHIP,     // parent/child/sibling of the predicate

    // new features for predicate sense
    FEAT_BAG_OF_WORD,         // all words in the sentence (multiple features)
    FEAT_BAG_OF_WORD_O,       // all words with left/target/right suffix
    FEAT_BAG_OF_POS_O,        // all POS's with numbered suffix
    FEAT_BAG_OF_POS_N,        // all POS's with left/target/right suffix
    FEAT_WIND5_BIGRAM,        // bigrams in the context window (5 word each side)
    FEAT_WIND5_BIGRAM_POS,
    FEAT_BAG_OF_POS_WIND5,
    FEAT_BAG_OF_POS_O_W5,
    FEAT_BAG_OF_POS_N_W5,
    FEAT_BAG_OF_WORD_IS_DES_O_PRED,

    FEAT_VERB_VOICE_EN,
    FEAT_PRED_VOICE_EN,

    FEAT_SUB_POS,
    FEAT_PFEAT_COLUMN,
    FEAT_PFEAT_EXC_NULL,
    FEAT_PFEAT,

    FEAT_NODE_V_PRED,         // there's a verb between node and predicate

    /* FEAT_VERB_VOICE,       // verb voice (for nouns are "NONVERB")
     * FEAT_PRED_VOICE,       // the voice of verb predicate (for PRED_NOUN's are "NONVERB")
    */

    TOTAL_FEATURE,            // total feature number
};

/* Auxiliary class for FeatureExtractor, holding information for the features
 * all FeatureExtractor objects hold one common non-static FeatureCollection, for 
 * looking up feature informations (such as feature names, feature prefix, etc)
 */
class FeatureCollection
{
    public:
        /* constructor, register features, record their feature number,
         * feature name, feature prefix, feature type, etc for later looking up
         */
        FeatureCollection();

        /* get the feature number for a given feature name
         */
        int get_feature_number(const std::string &feature_name);

        /* get the type of a given feature number
         */
        int get_feature_type(int feature_number);

        /* get the feature extraction function object of a given feature number
         */
        const FeatureFunction& get_feature_function(int feature_number);

        /* get the feature prefix for output of a given feature number
         */
        const std::string get_feature_prefix(int feature_number);

        /* get predicate feature number list
         */
        const std::vector<FEAT_NUM>& get_predicate_features()
        {
            return m_predicate_features;
        }

        /* get predicate feature number list
         */
        const std::vector<FEAT_NUM>& get_node_vs_predicate_features()
        {
            return m_node_vs_predicate_features;
        }

    private:
        struct FeatureInfo
        {
            std::string     name;
            std::string     prefix;
            FEAT_TYPE       type;
            FeatureFunction getter; // see FeatureFunction typedef
        };

    private:
        /* register informations for a feature, invoked in the constructor
         */
        void add_feature_(
                FEAT_NUM feature_number,
                FEAT_TYPE type,
                const std::string& name,
                const std::string& prefix,
                const FeatureFunction& getter);

    private:
        std::vector<FeatureInfo> m_feature_infos;
        std::vector<FEAT_NUM>    m_predicate_features;
        std::vector<FEAT_NUM>    m_node_vs_predicate_features;

};

struct FeatureSet
{
    std::vector<int> for_predicate;
    std::vector<int> for_node;
    std::vector<int> for_node_vs_predicate;

    void clear()
    {
        for_predicate.clear();
        for_node.clear();
        for_node_vs_predicate.clear();
    }
};

class FeatureExtractor
{
    public:
        explicit FeatureExtractor(const Configuration& config)
        {
            set_feature_set(config.get_argu_config().get_feature_names());
            m_configuration = config;
        }

        /* set the sentence from which features are extracted
         */
        void set_target_sentence(const Sentence &sentence);

        /* calculate all features in the feature set
         */
        void calc_features(const size_t predicate_index);
        void calc_node_features();

        void get_feature_for_rows(
                int feature_number,
                std::vector<std::string>& features_for_rows); 

        void set_feature_set(const std::vector<std::string>& feature_set_str);

        void clear_features();

        /* used for predicate sense
         */
        void set_feature_set_by_file(
                const std::string& config_file,
                const Configuration& configuration,
                std::vector<std::vector<std::string> >& com_features);

        void get_feature_string_for_row(
                const size_t predicate_row,
                std::string &result,
                const std::vector<std::vector<std::string> >& m_vct_vct_feature_names);
        //new function
        int get_feature_number_for_extractor(const std::string &feature_name);
        int get_feature_type_for_extractor(int feature_number);
        const FeatureFunction& get_feature_function_for_extractor(int feature_number);
        const std::vector<FEAT_NUM>& get_predicate_features_for_extractor();
        const std::vector<FEAT_NUM>& get_node_vs_predicate_features_for_extractor();
        const std::string get_feature_prefix_for_extractor(int feature_number);


    private:
        /* Get single feature for specific row
         * if not yet calculated, do it immediately
         */
        const std::string& get_feature_value_(const int feature_number, const size_t row);

        void set_feature_value_(const int feature_number, const size_t row, const std::string& feature_value);

        /* whether a specified feature for specified row is empty
         */
        bool is_feature_empty_(const int feature_number, const size_t row);

        void set_feature_empty_(const int feature_number, const size_t row, const bool empty);

        void set_feature_set_(
                const std::vector<std::string>& feature_set_str,
                FeatureSet& feature_set);

        std::string& get_feature_storage_(const int feature_number, const size_t row);

        void calc_features_(const FeatureSet& feature_set);

        void calc_node_features_(const std::vector<int>& node_features);

        void calc_predicate_features_(const std::vector<int>& predicate_features);

        void calc_node_vs_predicate_features_(const std::vector<int>& node_vs_predicate_features);

        void clear_predicate_features_();
        void clear_node_vs_predicate_features_();

        int string2int(const std::string& str)
        {
            std::istringstream in_stream(str);
            size_t res;
            in_stream>>res;
            return res;
        }

        std::string int2string(const int num)
        {
            std::ostringstream out_stream;
            out_stream<<num;
            return out_stream.str();
        }

        std::vector<std::string> split_(std::string line, char s='+')
        {
            replace(line.begin(), line.end(), s, ' ');
            std::istringstream istr(line);
            std::vector<std::string> res;
            std::string tmp_str;
            while (istr>>tmp_str)
            {
                res.push_back(tmp_str);
            }
            return res;
        }

        std::map<std::string, std::string> split_feat_(std::string line)
        {
            replace(line.begin(), line.end(), '|', ' ');
            std::istringstream istr(line);
            std::map<std::string, std::string> res;
            std::string tmp_str;
            while (istr>>tmp_str)
            {
                size_t find = tmp_str.find("=");
                assert(std::string::npos != find);
                std::string word  = tmp_str.substr(0, find);
                std::string value = tmp_str.substr(find+1); 
                res[word] = value;
            }
            return res;
        }

        void check_feature_exist(
                const std::vector<std::vector<std::string> >& com_features,
                const std::vector<std::string>& feature_set)
        {
            for (size_t i=0; i<com_features.size();++i)
                for (size_t j=0;j<com_features[i].size();++j)
                {
                    if (find(feature_set.begin(), feature_set.end(),
                                com_features[i][j]) == feature_set.end())
                    {
                        throw std::runtime_error(com_features[i][j]+" is not in predicate sense configuration");
                    }
                }
        }

        std::vector<std::string> vct_vct_string2_vct_string(
                const std::vector<std::vector<std::string> >& feature_set)
        {
            std::vector<std::string> res;
            for (size_t i=0; i<feature_set.size(); ++i)
            {
                for (size_t j=0; j<feature_set[i].size(); ++j)
                {
                    const std::string& feature = feature_set[i][j];
                    if (find (res.begin(), res.end(), feature) == res.end())
                    {
                        res.push_back(feature);
                    }
                }
            }
            return res;
        }

    private:
        // the sentence from which freatures are extracted
        const Sentence* mp_sentence;

        friend class FeatureCollection;

        FeatureSet m_feature_set;

        // the current extracting predicate_index
        size_t m_predicate_row;

        bool m_node_features_extracted_flag;

        // turn static assistant class into common member variable
        FeatureCollection ms_feature_collection;

        // storage the feature value
        std::vector<std::vector<std::string> > m_feature_values;

        // flag for whether a feature is already calculated for specific row
        std::vector<std::bitset<TOTAL_FEATURE> > m_feature_extracted_flags;

        // Configuration
        Configuration m_configuration;

    private:
        void fg_basic_info_(const size_t row);
        void fg_constituent_(const size_t row);
        void fg_children_pattern_(const size_t row);
        void fg_siblings_pattern_(const size_t row);
        // void fg_has_support_verb_(const size_t row);
        void fg_predicate_children_pattern_(const size_t row);
        void fg_predicate_siblings_pattern_(const size_t row);
        void fg_predicate_basic_(const size_t row);
        void fg_path_(const size_t row);
        void fg_path_length_(const size_t row);
        void fg_descendant_of_predicate_(const size_t row);
        void fg_position_(const size_t row);
        void fg_predicate_familyship_(const size_t row);
        void fg_predicate_bag_of_words_(const size_t row);
        void fg_predicate_bag_of_words_ordered_(const size_t row);
        void fg_predicate_bag_of_POSs_ordered_(const size_t row);
        void fg_predicate_bag_of_POSs_numbered_(const size_t row);
        void fg_predicate_window5_bigram_(const size_t row);

        void fg_verb_voice_en_(const size_t row);
        void fg_predicate_voice_en_(const size_t row);
        void fg_feat_column(const size_t row);
        void fg_predicate_bag_of_POSs_window5_(const size_t row);
        void fg_pfeat_column_(const size_t row);
        void fg_pfeat_(const size_t row);

};


#endif

