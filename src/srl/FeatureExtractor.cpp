/*
 * File Name     : FeatureExtractor.cpp
 * Author        : msmouse
 * Create Time   : 2006-12-31
 * Project Name  : NewSRLBaseLine
 *
 * Updated by    : jiangfeng
 * Update Time   : 2013-08-21
 */


#include "FeatureExtractor.h"

#include <iostream>

using namespace std;

// implementation for FeatureNameFunctionMap

FeatureCollection::FeatureCollection()
{
    // make room for all the features
    m_feature_infos.clear();
    m_feature_infos.resize(TOTAL_FEATURE);

    // add feature functions
    //   node features
    //           feature_number,         type,                        name,                            prefix,            getter_function
    add_feature_(FEAT_DEPREL,            FEAT_TYPE_NODE,              "DepRelation",                   "DEPREL",     &FeatureExtractor::fg_basic_info_);
    add_feature_(FEAT_HEADWORD_POS,      FEAT_TYPE_NODE,              "HeadwordPOS",                   "HEAD_POS",   &FeatureExtractor::fg_basic_info_);
    add_feature_(FEAT_DEPWORD_POS,       FEAT_TYPE_NODE,              "DepwordPOS",                    "DEP_POS",    &FeatureExtractor::fg_basic_info_);
    add_feature_(FEAT_HEADWORD,          FEAT_TYPE_NODE,              "Headword",                      "HEADWORD",   &FeatureExtractor::fg_basic_info_);
    add_feature_(FEAT_DEPWORD,           FEAT_TYPE_NODE,              "Depword",                       "DEPWORD",    &FeatureExtractor::fg_basic_info_);
    add_feature_(FEAT_HEADWORD_LEMMA,    FEAT_TYPE_NODE,              "HeadwordLemma",                 "HEDLEMMA",   &FeatureExtractor::fg_basic_info_);
    add_feature_(FEAT_DEPWORD_LEMMA,     FEAT_TYPE_NODE,              "DepwordLemma",                  "DEPLEMMA",   &FeatureExtractor::fg_basic_info_);

    add_feature_(FEAT_FIRST_WORD,        FEAT_TYPE_NODE,              "FirstWord",                     "FIRST_WD",   &FeatureExtractor::fg_constituent_);
    add_feature_(FEAT_LAST_WORD,         FEAT_TYPE_NODE,              "LastWord",                      "LAST_WD",    &FeatureExtractor::fg_constituent_);
    add_feature_(FEAT_FIRST_POS,         FEAT_TYPE_NODE,              "FirstPOS",                      "FIRST_POS",  &FeatureExtractor::fg_constituent_);
    add_feature_(FEAT_LAST_POS,          FEAT_TYPE_NODE,              "LastPOS",                       "LAST_POS",   &FeatureExtractor::fg_constituent_);
    add_feature_(FEAT_POS_PATTERN,       FEAT_TYPE_NODE,              "ConstituentPOSPattern",         "POS_PAT",    &FeatureExtractor::fg_constituent_);
    add_feature_(FEAT_FIRST_LEMMA,       FEAT_TYPE_NODE,              "FirstLemma",                    "FIRST_LEM",  &FeatureExtractor::fg_constituent_);
    add_feature_(FEAT_LAST_LEMMA,        FEAT_TYPE_NODE,              "LastLemma",                     "LAST_LEM",   &FeatureExtractor::fg_constituent_);

    add_feature_(FEAT_CHD_POS,           FEAT_TYPE_NODE,              "ChildrenPOS",                   "CH_POS",     &FeatureExtractor::fg_children_pattern_);
    add_feature_(FEAT_CHD_POS_NDUP,      FEAT_TYPE_NODE,              "ChildrenPOSNoDup",              "CH_POS2",    &FeatureExtractor::fg_children_pattern_);
    add_feature_(FEAT_CHD_REL,           FEAT_TYPE_NODE,              "ChildrenREL",                   "CH_REL",     &FeatureExtractor::fg_children_pattern_);
    add_feature_(FEAT_CHD_REL_NDUP,      FEAT_TYPE_NODE,              "ChildrenRELNoDup",              "CH_REL2",    &FeatureExtractor::fg_children_pattern_);

    add_feature_(FEAT_SIB_POS,           FEAT_TYPE_NODE,              "SiblingsPOS",                   "SB_POS",     &FeatureExtractor::fg_siblings_pattern_);
    add_feature_(FEAT_SIB_POS_NDUP,      FEAT_TYPE_NODE,              "SiblingsPOSNoDup",              "SB_POS2",    &FeatureExtractor::fg_siblings_pattern_);
    add_feature_(FEAT_SIB_REL,           FEAT_TYPE_NODE,              "SiblingsREL",                   "SB_REL",     &FeatureExtractor::fg_siblings_pattern_);
    add_feature_(FEAT_SIB_REL_NDUP,      FEAT_TYPE_NODE,              "SiblingsRELNoDup",              "SB_REL2",    &FeatureExtractor::fg_siblings_pattern_);

    // Predicate features

    add_feature_(FEAT_PRED_CHD_POS,      FEAT_TYPE_PRED,              "PredicateChildrenPOS",          "P_CH_POS",   &FeatureExtractor::fg_predicate_children_pattern_);
    add_feature_(FEAT_PRED_CHD_POS_NDUP, FEAT_TYPE_PRED,              "PredicateChildrenPOSNoDup",     "P_CH_POS2",  &FeatureExtractor::fg_predicate_children_pattern_);
    add_feature_(FEAT_PRED_CHD_REL,      FEAT_TYPE_PRED,              "PredicateChildrenREL",          "P_CH_REL",   &FeatureExtractor::fg_predicate_children_pattern_);
    add_feature_(FEAT_PRED_CHD_REL_NDUP, FEAT_TYPE_PRED,              "PredicateChildrenRELNoDup",     "P_CH_REL2",  &FeatureExtractor::fg_predicate_children_pattern_);

    add_feature_(FEAT_PRED_SIB_POS,      FEAT_TYPE_PRED,              "PredicateSiblingsPOS",          "P_SB_POS",   &FeatureExtractor::fg_predicate_siblings_pattern_);
    add_feature_(FEAT_PRED_SIB_POS_NDUP, FEAT_TYPE_PRED,              "PredicateSiblingsPOSNoDup",     "P_SB_POS2",  &FeatureExtractor::fg_predicate_siblings_pattern_);
    add_feature_(FEAT_PRED_SIB_REL,      FEAT_TYPE_PRED,              "PredicateSiblingsREL",          "P_SB_REL",   &FeatureExtractor::fg_predicate_siblings_pattern_);
    add_feature_(FEAT_PRED_SIB_REL_NDUP, FEAT_TYPE_PRED,              "PredicateSiblingsRELNoDup",     "P_SB_REL2",  &FeatureExtractor::fg_predicate_siblings_pattern_);
    
    add_feature_(FEAT_PRED_LEMMA,        FEAT_TYPE_PRED,              "PredicateLemma",                "P_LEMMA",    &FeatureExtractor::fg_predicate_basic_);
    add_feature_(FEAT_PREDICATE,         FEAT_TYPE_PRED,              "Predicate",                     "PRED",       &FeatureExtractor::fg_predicate_basic_);
    add_feature_(FEAT_PRED_SENSE,        FEAT_TYPE_PRED,              "PredicateSense",                "P_SENSE",    &FeatureExtractor::fg_predicate_basic_);

    // node_vs_predicate features

    add_feature_(FEAT_PATH,              FEAT_TYPE_NODE_VS_PRED,      "Path",                          "PATH",       &FeatureExtractor::fg_path_);
    add_feature_(FEAT_UP_PATH,           FEAT_TYPE_NODE_VS_PRED,      "UpPath",                        "UP_PTH",     &FeatureExtractor::fg_path_);
    add_feature_(FEAT_REL_PATH,          FEAT_TYPE_NODE_VS_PRED,      "RelationPath",                  "REL_PATH",   &FeatureExtractor::fg_path_);
    add_feature_(FEAT_UP_REL_PATH,       FEAT_TYPE_NODE_VS_PRED,      "UpRelationPath",                "UP_REL_PT",  &FeatureExtractor::fg_path_);
    
    add_feature_(FEAT_PATH_LENGTH,       FEAT_TYPE_NODE_VS_PRED,      "PathLength",                    "PATH_LEN",   &FeatureExtractor::fg_path_length_);
    add_feature_(FEAT_UP_PATH_LEN,       FEAT_TYPE_NODE_VS_PRED,      "UpPathLength",                  "UP_PT_LEN",  &FeatureExtractor::fg_path_length_);
    add_feature_(FEAT_DOWN_PATH_LEN,     FEAT_TYPE_NODE_VS_PRED,      "DownPathLength",                "DN_PT_LEN",  &FeatureExtractor::fg_path_length_);
    
    add_feature_(FEAT_DESC_OF_PD,        FEAT_TYPE_NODE_VS_PRED,      "DescendantOfPredicate",         "D_OF_PRD",   &FeatureExtractor::fg_descendant_of_predicate_);

    add_feature_(FEAT_POSITION,          FEAT_TYPE_NODE_VS_PRED,      "Position",                      "POSITION",   &FeatureExtractor::fg_position_);

    add_feature_(FEAT_PRED_FAMILYSHIP,   FEAT_TYPE_NODE_VS_PRED,      "PredicateFamilyship",           "PRD_FAMIL",  &FeatureExtractor::fg_predicate_familyship_);


    // not addd  verb_voice
    // add_feature_(FEAT_VERB_VOICE,       FEAT_TYPE_NODE,             "VerbVoice",                "VOICE",    &FeatureExtractor::fg_verb_voice_);
    // add_feature_(FEAT_PRED_VOICE,       FEAT_TYPE_PRED,             "PredicateVoice",           "PREDVOICE",&FeatureExtractor::fg_predicate_voice_);
    // add_feature_(FEAT_NODE_V_PRED,       FEAT_TYPE_NODE_VS_PRED,      "VerbBetweenPredicate",       "N_V_PRED",  &FeatureExtractor::fg_has_verb_between_predicate_);
    // add_feature_(FEAT_HAS_SV,            FEAT_TYPE_PRED,              "HasSupportVerb",        "HAS_SV",     &FeatureExtractor::fg_has_support_verb_); // problem
    

    // new features for predicate sense recognition
    add_feature_(FEAT_BAG_OF_WORD,       FEAT_TYPE_PRED,              "PredicateBagOfWords",           "P_BOW",      &FeatureExtractor::fg_predicate_bag_of_words_);
    add_feature_(FEAT_BAG_OF_WORD_O,     FEAT_TYPE_PRED,              "PredicateBagOfWordsOrdered",    "P_BOWO",     &FeatureExtractor::fg_predicate_bag_of_words_ordered_);
    add_feature_(FEAT_BAG_OF_POS_O,      FEAT_TYPE_PRED,              "PredicateBagOfPOSOrdered",      "P_BOPO",     &FeatureExtractor::fg_predicate_bag_of_POSs_ordered_);
    add_feature_(FEAT_BAG_OF_POS_N,      FEAT_TYPE_PRED,              "PredicateBagOfPOSNumbered",     "P_BOPN",     &FeatureExtractor::fg_predicate_bag_of_POSs_numbered_);
    add_feature_(FEAT_WIND5_BIGRAM,      FEAT_TYPE_PRED,              "PredicateWindow5Bigram",        "P_W5BGRM",   &FeatureExtractor::fg_predicate_window5_bigram_);
    add_feature_(FEAT_WIND5_BIGRAM_POS,  FEAT_TYPE_PRED,              "PredicateWindow5BigramPOS",     "P_W5BGPOS",  &FeatureExtractor::fg_predicate_window5_bigram_);
    add_feature_(FEAT_BAG_OF_POS_WIND5,  FEAT_TYPE_PRED,              "PredicateBagOfPOSWindow5",      "P_BOPW5",    &FeatureExtractor::fg_predicate_bag_of_POSs_window5_);
    add_feature_(FEAT_BAG_OF_POS_O_W5,   FEAT_TYPE_PRED,              "PredicateBagOfPOSorderedWindow5", "P_BOPOW5", &FeatureExtractor::fg_predicate_bag_of_POSs_ordered_);
    add_feature_(FEAT_BAG_OF_POS_N_W5,   FEAT_TYPE_PRED,              "PredicateBagOfPOSNumberedWindow5", "P_POSNW5", &FeatureExtractor::fg_predicate_bag_of_POSs_numbered_);
    add_feature_(FEAT_BAG_OF_WORD_IS_DES_O_PRED, FEAT_TYPE_PRED,      "PredicateBagOfWordsAndIsDesOfPRED", "P_BOWDP", &FeatureExtractor::fg_predicate_bag_of_words_);

    // special features
    // for English
    add_feature_(FEAT_VERB_VOICE_EN,    FEAT_TYPE_NODE,               "VerbVoiceEn",                   "VOICE_EN",     &FeatureExtractor::fg_verb_voice_en_);
    add_feature_(FEAT_PRED_VOICE_EN,    FEAT_TYPE_PRED,               "PredicateVoiceEn",              "PREDVOICE_EN", &FeatureExtractor::fg_predicate_voice_en_);
    // for Chinese
    // for Spanish
    // for Catalan
    // for German
    // for Czech
    // for Japanese
    // for Spanish Catalan German Czech Japanese
    add_feature_(FEAT_SUB_POS,          FEAT_TYPE_NODE,               "SubPOS",                        "SUBPOS",       &FeatureExtractor::fg_feat_column);
    add_feature_(FEAT_PFEAT_COLUMN,     FEAT_TYPE_NODE,               "PFEATColumn",                   "PFEATC",       &FeatureExtractor::fg_pfeat_column_);
    add_feature_(FEAT_PFEAT_EXC_NULL,   FEAT_TYPE_NODE,               "PFEATExceptNull",               "PFEATNULL",    &FeatureExtractor::fg_pfeat_column_);
    add_feature_(FEAT_PFEAT,            FEAT_TYPE_NODE,               "PFEAT",                         "PFEAT",        &FeatureExtractor::fg_pfeat_);
}

void FeatureCollection::add_feature_(
        FEAT_NUM feature_number,
        FEAT_TYPE type,
        const std::string& name,
        const std::string& prefix,
        const FeatureFunction& getter)
{
    m_feature_infos[feature_number].name   = name;
    m_feature_infos[feature_number].prefix = prefix;
    m_feature_infos[feature_number].type   = type;
    m_feature_infos[feature_number].getter = getter;

    switch (type) 
    {
        case FEAT_TYPE_PRED:
            m_predicate_features.push_back(feature_number);
            break;
        case FEAT_TYPE_NODE_VS_PRED:
            m_node_vs_predicate_features.push_back(feature_number);
            break;
        default:
            break;
    }
}

int FeatureCollection::get_feature_number(const string &feature_name)
{
    // linear search for the given feature name
    size_t feature_idx;
    for (feature_idx=0; feature_idx<TOTAL_FEATURE; ++feature_idx)
    {
        if (m_feature_infos[feature_idx].name == feature_name)
        {
            break;
        }
    }
    if (feature_idx < TOTAL_FEATURE)  // found
    {
        return static_cast<int>(feature_idx);
    }
    else
    {
        throw runtime_error("Unknown feature name: " + feature_name);
    }
}

int FeatureCollection::get_feature_type(int feature_number)
{
    return m_feature_infos[feature_number].type;
}

const FeatureFunction& FeatureCollection::get_feature_function(int feature_number)
{
    return m_feature_infos[feature_number].getter;
}

const string FeatureCollection::get_feature_prefix(int feature_number)
{
    return m_feature_infos[feature_number].prefix;
}

// impolementation for FeatureExtractor

//new function
int FeatureExtractor::get_feature_number_for_extractor(const std::string &feature_name)
{
    return ms_feature_collection.get_feature_number(feature_name);
}
int FeatureExtractor::get_feature_type_for_extractor(int feature_number)
{
     return ms_feature_collection.get_feature_type(feature_number);
}
const FeatureFunction&  FeatureExtractor::get_feature_function_for_extractor(int feature_number)
{
    return ms_feature_collection.get_feature_function(feature_number);
}
const std::vector<FEAT_NUM>& FeatureExtractor::get_predicate_features_for_extractor()
{
    return ms_feature_collection.get_predicate_features();
}
const std::vector<FEAT_NUM>& FeatureExtractor::get_node_vs_predicate_features_for_extractor()
{
    return ms_feature_collection.get_node_vs_predicate_features();
}
const std::string FeatureExtractor::get_feature_prefix_for_extractor(int feature_number)
{
    return ms_feature_collection.get_feature_prefix(feature_number);
}
void FeatureExtractor::clear_features()
{
    m_feature_extracted_flags.clear();
    m_feature_values.clear();
    m_feature_values.resize(TOTAL_FEATURE);

    m_node_features_extracted_flag = false;
}

void FeatureExtractor::set_target_sentence(const Sentence &sentence)
{
    clear_features();
    mp_sentence = &sentence;

    size_t row_count = sentence.get_row_count();
    m_feature_extracted_flags.resize(row_count+1);
}

void FeatureExtractor::set_feature_set_(
    const std::vector<std::string>& feature_set_str,
    FeatureSet& feature_set)
{
    feature_set.clear();

    set<int> predicate_features;
    set<int> node_features;
    set<int> node_vs_predicate_features;

    for (size_t i=0; i<feature_set_str.size(); ++i)
    {
        const string& feat_tmp = feature_set_str[i];
        int feature_number = get_feature_number_for_extractor(feat_tmp);
        int feature_type = get_feature_type_for_extractor(feature_number);

        switch (feature_type)
        {
            case FEAT_TYPE_PRED:
                predicate_features.insert(feature_number);
                break;
            case FEAT_TYPE_NODE:
                node_features.insert(feature_number);
                break;
            case FEAT_TYPE_NODE_VS_PRED:
                node_vs_predicate_features.insert(feature_number);
                break;
            default:
                std::stringstream S; S << feature_number;
                throw runtime_error("Unknown feature type for" + S.str());
        }
    }

    // store single features
    feature_set.for_predicate.assign(
        predicate_features.begin(),
        predicate_features.end());

    feature_set.for_node.assign(
        node_features.begin(),
        node_features.end());

    feature_set.for_node_vs_predicate.assign(
        node_vs_predicate_features.begin(),
        node_vs_predicate_features.end());
}

void FeatureExtractor::set_feature_set(const vector<string> &feature_set_str)
{
    set_feature_set_(feature_set_str, m_feature_set);
}

const std::string& FeatureExtractor::get_feature_value_(
    const int feature_number,
    const size_t row)
{
    if (is_feature_empty_(feature_number, row))
    {
        FeatureFunction function 
            = get_feature_function_for_extractor(feature_number);

        function(this, row);
    }

    return get_feature_storage_(feature_number, row);
}

void FeatureExtractor::set_feature_value_(
    const int    feature_number,
    const size_t row,
    const        string& feature_value)
{
    get_feature_storage_(feature_number, row) = feature_value;
    set_feature_empty_(feature_number, row, false);
}

bool FeatureExtractor::is_feature_empty_(const int feature_number, const size_t row)
{
    int feature_type
        = get_feature_type_for_extractor(feature_number);
    
    if (FEAT_TYPE_PRED == feature_type)
    {
        return !m_feature_extracted_flags[m_predicate_row][feature_number];
    }
    else
    {
        return !m_feature_extracted_flags[row][feature_number];
    }
}

void FeatureExtractor::set_feature_empty_(
    const int     feature_number,
    const size_t  row,
    const bool    empty)
{
    int feature_type
        = get_feature_type_for_extractor(feature_number);

    if (FEAT_TYPE_PRED == feature_type)
    {
        m_feature_extracted_flags[m_predicate_row][feature_number] = !empty;
    }
    else
    {
        m_feature_extracted_flags[row][feature_number] = !empty;
    }
}

string& FeatureExtractor::get_feature_storage_(
    const int feature_number,
    const size_t row)
{
    const int feature_type
        = get_feature_type_for_extractor(feature_number);

    switch (feature_type)
    {
        case FEAT_TYPE_PRED:
            if (m_feature_values[feature_number].empty())
            {
//                std::cout<<"hello"<<std::endl;
                m_feature_values[feature_number].resize(1);
            }
            return m_feature_values[feature_number][0];
         case FEAT_TYPE_UNKNOWN:
            throw runtime_error("Unknown feature number:");


         default:
            if (m_feature_values[feature_number].empty())
            {
                size_t row_count = mp_sentence->get_row_count();
                m_feature_values[feature_number].resize(row_count+1);
            }
            return m_feature_values[feature_number][row];
    }
}

void FeatureExtractor::calc_features(const size_t predicate_index)
{
    const Predicate &predicate
        = mp_sentence->get_predicates()[predicate_index];

    m_predicate_row  = predicate.row;

    calc_features_(m_feature_set);
}

void FeatureExtractor::calc_features_(const FeatureSet& feature_set)
{
    calc_predicate_features_(feature_set.for_predicate);
    calc_node_vs_predicate_features_(feature_set.for_node_vs_predicate);
    calc_node_features_(feature_set.for_node);
}

void FeatureExtractor::calc_node_features()
{
    calc_node_features_(m_feature_set.for_node);
}

void FeatureExtractor::calc_node_features_(const vector<int>& node_features)
{
    if (m_node_features_extracted_flag)
    {
        return;
    }

    const SRLTree& parse_tree = mp_sentence->get_parse_tree();
    typedef SRLTree::post_order_iterator PostIter;
    for (PostIter node_iter = parse_tree.begin_post();
         node_iter != --parse_tree.end_post();
         ++node_iter)
    {
        for (size_t i=0; i<node_features.size(); ++i)
        {
            const int &feature_number = node_features[i];
            get_feature_value_(feature_number, *node_iter);
        }
    }
    m_node_features_extracted_flag = true;

}

void FeatureExtractor::calc_predicate_features_(const vector<int>& predicate_features)
{
    clear_predicate_features_();

    for (size_t i = 0; i < predicate_features.size(); ++ i) {
      int feature_number = predicate_features[i];
        get_feature_value_(feature_number, m_predicate_row);
    }
}

void FeatureExtractor::calc_node_vs_predicate_features_(const vector<int>& node_vs_predicate_features)
{
    clear_node_vs_predicate_features_();

    // prepare constants
    const SRLTree& parse_tree = mp_sentence->get_parse_tree();
    const size_t   row_count  = mp_sentence->get_row_count();

    // prepare for path calculation algorithm
    get_feature_storage_(FEAT_PATH,        m_predicate_row)
        = mp_sentence->get_PPOS(m_predicate_row);
    get_feature_storage_(FEAT_UP_PATH,     m_predicate_row)
        = string();
    get_feature_storage_(FEAT_REL_PATH,    m_predicate_row)
        = string();
    get_feature_storage_(FEAT_UP_REL_PATH, m_predicate_row)
        = string();

    vector<bool> node_visited_flags(row_count+1);

    // traversal begins at the predicate
    queue<SRLTree::iterator> nodes_queue;
    SRLTree::iterator
        node_iter = mp_sentence->get_node_of_row(m_predicate_row);
    nodes_queue.push(node_iter);

    // traverse
    while (!nodes_queue.empty())
    {
        // fetch a node from the queue
        node_iter = nodes_queue.front();
        nodes_queue.pop();

        for (size_t i = 0; i < node_vs_predicate_features.size(); ++ i) {
          int feature_number = node_vs_predicate_features[i];
            get_feature_value_(feature_number, *node_iter);
        }

        node_visited_flags[*node_iter] = true; // visit;

        // add children to the queue
        typedef SRLTree::sibling_iterator SiblingIter;
        for (SiblingIter child_iter = node_iter.begin();
             child_iter != node_iter.end();
             ++child_iter)
        {
            if (!node_visited_flags[*child_iter])
            {
                nodes_queue.push(child_iter);
            }
        }

        // add parent to queue
        SRLTree::iterator parent = parse_tree.parent(node_iter);
        if (parse_tree.is_valid(parent) && !node_visited_flags[*parent])
        {
            nodes_queue.push(parent);
        }
    }
}

void FeatureExtractor::clear_predicate_features_()
{
  const std::vector<FEAT_NUM>& payload = get_predicate_features_for_extractor();
  for (size_t i = 0; i < payload.size(); ++ i) {
    int feature_number = payload[i];
    m_feature_extracted_flags[m_predicate_row][feature_number] = false;
    m_feature_values[feature_number].clear();
  }
}

void FeatureExtractor::clear_node_vs_predicate_features_()
{
    // clear empty flags
    for (size_t row=1; row<=mp_sentence->get_row_count(); ++row)
    {
      const std::vector<FEAT_NUM>& payload = get_node_vs_predicate_features_for_extractor();
      for (size_t i = 0; i < payload.size(); ++ i) {
        int feature_number = payload[i];
        m_feature_extracted_flags[row][feature_number] = false;
      }
    }

    // clear feature values
    const std::vector<FEAT_NUM>& payload = get_node_vs_predicate_features_for_extractor();
    for (size_t i = 0; i < payload.size(); ++ i) {
      int feature_number = payload[i];
        m_feature_values[feature_number].clear();
    }
}

void FeatureExtractor::set_feature_set_by_file(
    const string& config_file,
    const Configuration &configuration,
    vector<vector<string> >& com_features)
{
    ifstream config_stream(config_file.c_str());
    if (!config_stream) 
    {
        throw runtime_error("FeatureExtractor: Error opening config file.");
    }

    string line;
    com_features.clear();
    vector<vector<string> >* p_features;
    p_features = &com_features;

    while (getline(config_stream, line))
    {
        if ('#' != line[0])
            p_features->push_back(split_(line));
    }

    // check features in config file belongs language configuration
    const vector<string>& features = configuration.get_pred_class_config().get_feature_names();
    check_feature_exist(com_features, features);
    set_feature_set(
        vct_vct_string2_vct_string(com_features)
    );
}

void FeatureExtractor::get_feature_string_for_row(
    const size_t predicate_row,
    string &result,
    const vector<vector<string> >& vct_vct_feature_names)
{
    stringstream row_features_stream;
    for (size_t i=0; i<vct_vct_feature_names.size(); ++i)
    {
        const vector<string> & com_feature_names = vct_vct_feature_names[i];

        bool first_part_flag = true;
        for (size_t j=0; j<com_feature_names.size(); ++j)
        {
            const string& feature_name = com_feature_names[j];
            const size_t feature_number = get_feature_number_for_extractor(feature_name);
            if (first_part_flag)
            {
                first_part_flag = false;
            }
            else
            {
                row_features_stream<<'+';
            }
            string feature_prefix =get_feature_prefix_for_extractor(feature_number);
            string feature_result = get_feature_storage_(feature_number, predicate_row);
            if (feature_prefix == "PFEATNULL" && feature_result == "")
            {
                continue;
            }
            row_features_stream
                <<get_feature_prefix_for_extractor(feature_number)
                <<"@"
                <<get_feature_storage_(feature_number, predicate_row);
        }
        row_features_stream<<' ';
    }
    result = row_features_stream.str();
}

void FeatureExtractor::get_feature_for_rows(
    int feature_number,
    vector<string>& features_for_rows)
{
    features_for_rows.clear();
    features_for_rows.push_back(get_feature_storage_(feature_number, 0));

    const size_t row_count = mp_sentence->get_row_count();
    for (size_t row=1; row<=row_count; ++row) // row id start at 1
    {
        if (is_feature_empty_(feature_number, row))
        {
            throw runtime_error("Specified feature_number is empty for row");
        }

        features_for_rows.push_back(get_feature_storage_(feature_number, row));
    }
}

void FeatureExtractor::fg_basic_info_(const size_t row)
{
    const size_t headword_row = mp_sentence->get_PHEAD(row);

    // set feature values;
    set_feature_value_(FEAT_DEPREL,        row,      mp_sentence->get_PDEPREL(row));
    set_feature_value_(FEAT_HEADWORD,      row,      mp_sentence->get_FORM(headword_row));
    set_feature_value_(FEAT_DEPWORD,       row,      mp_sentence->get_FORM(row));
    set_feature_value_(FEAT_HEADWORD_POS,  row,      mp_sentence->get_PPOS(headword_row));
    set_feature_value_(FEAT_DEPWORD_POS,   row,      mp_sentence->get_PPOS(row));
    set_feature_value_(FEAT_HEADWORD_LEMMA,row,      mp_sentence->get_PLEMMA(headword_row));
    set_feature_value_(FEAT_DEPWORD_LEMMA, row,      mp_sentence->get_PLEMMA(row));

}

void FeatureExtractor::fg_constituent_(const size_t row)
{
    const SRLTree& parse_tree = mp_sentence->get_parse_tree();

    typedef SRLTree::iterator Iter;
    const Iter& node = mp_sentence->get_node_of_row(row);

    if (parse_tree.number_of_children(node))
    {
        size_t begin = row, end = row;
        for (Iter child = node.begin(); child != node.end(); ++child)
        {
            if (*child < begin)
            {
                begin = *child;
            }
            if (*child > end)
            {
                end = *child;
            }
        }

        const string& first_FORM   = mp_sentence->get_FORM(begin);
        const string& first_POS    = mp_sentence->get_PPOS(begin);
        const string& first_LEMMA  = mp_sentence->get_PLEMMA(begin);
        const string& last_FORM    = mp_sentence->get_FORM(end);
        const string& last_POS     = mp_sentence->get_PPOS(end);
        const string& last_LEMMA   = mp_sentence->get_PLEMMA(end);

        set_feature_value_(FEAT_FIRST_WORD,   row,   first_FORM);
        set_feature_value_(FEAT_FIRST_POS,    row,   first_POS);
        set_feature_value_(FEAT_FIRST_LEMMA,  row,   first_LEMMA);
        set_feature_value_(FEAT_LAST_WORD,    row,   last_FORM);
        set_feature_value_(FEAT_LAST_POS,     row,   last_POS);
        set_feature_value_(FEAT_LAST_LEMMA,   row,   last_LEMMA);

        if (begin == end)
        {
            set_feature_value_(FEAT_POS_PATTERN, row, first_POS);
            throw runtime_error("Only leaf's begin == end");
        }
        else
        {
            string POS_pattern;
            POS_pattern = first_POS;
            set<string> inner_POS;
            for (size_t i=begin+1; i < end; ++i)
            {
                inner_POS.insert(mp_sentence->get_PPOS(i));
            }
            for (set<string>::iterator iter = inner_POS.begin();
                 iter != inner_POS.end();
                 ++iter)
            {
                POS_pattern += "-";
                POS_pattern += *iter;
            }
            POS_pattern += "-";
            POS_pattern += last_POS;
            set_feature_value_(FEAT_POS_PATTERN, row, POS_pattern);
        }
    }
    else // leaf
    {
        const string& FORM  = mp_sentence->get_FORM(row);
        const string& POS   = mp_sentence->get_PPOS(row);
        const string& LEMMA = mp_sentence->get_PLEMMA(row);

        set_feature_value_(FEAT_FIRST_WORD,   row,  FORM);
        set_feature_value_(FEAT_FIRST_POS,    row,  POS);
        set_feature_value_(FEAT_FIRST_LEMMA,  row,  LEMMA);
        set_feature_value_(FEAT_LAST_WORD,    row,  FORM);
        set_feature_value_(FEAT_LAST_POS,     row,  POS);
        set_feature_value_(FEAT_LAST_LEMMA,   row,  LEMMA);
        set_feature_value_(FEAT_POS_PATTERN,  row,  POS);
    }
}

void FeatureExtractor::fg_children_pattern_(const size_t row)
{
    typedef SRLTree::sibling_iterator Iter;
    Iter node_iter = mp_sentence->get_node_of_row(row);

    string children_pos;
    string children_rel;
    string children_pos_ndup;
    string children_rel_ndup;

    string child_pos;
    string child_rel;
    string old_child_pos;
    string old_child_rel;

    for (Iter child = node_iter.begin();
         child != node_iter.end();
         ++child)
    {
        child_pos = mp_sentence->get_PPOS(*child);
        child_rel = mp_sentence->get_PDEPREL(*child);

        children_pos.append(child_pos);
        children_pos.append("-");
        children_rel.append(child_rel);
        children_rel.append("-");

        if (child_pos != old_child_pos) 
        {
            children_pos_ndup.append(child_pos);
            children_pos_ndup.append("-");
            old_child_pos = child_pos;
        }
        if (child_rel != old_child_rel) 
        {
            children_rel_ndup.append(child_rel);
            children_rel_ndup.append("-");
            old_child_rel = child_rel;
        }
    }

    set_feature_value_(FEAT_CHD_POS, row, children_pos);
    set_feature_value_(FEAT_CHD_REL, row, children_rel);
    set_feature_value_(FEAT_CHD_POS_NDUP, row, children_pos_ndup);
    set_feature_value_(FEAT_CHD_REL_NDUP, row, children_rel_ndup);
}

void FeatureExtractor::fg_siblings_pattern_( const size_t row )
{
    typedef SRLTree::sibling_iterator Iter;
    const size_t parent_row = mp_sentence->get_PHEAD(row);
    Iter parent_node = mp_sentence->get_node_of_row(parent_row);

    string siblings_pos;
    string siblings_rel;
    string siblings_pos_ndup;
    string siblings_rel_ndup;

    string sibling_pos;
    string sibling_rel;
    string old_sibling_pos;
    string old_sibling_rel;

    for (Iter sib = parent_node.begin();
        sib != parent_node.end();
        ++sib)
    {
        sibling_pos = mp_sentence->get_PPOS(*sib);
        sibling_rel = mp_sentence->get_PDEPREL(*sib);
        siblings_pos.append(sibling_pos);
        siblings_pos.append("-");
        siblings_rel.append(sibling_rel);
        siblings_rel.append("-");

        if (sibling_pos != old_sibling_pos) {
            siblings_pos_ndup.append(sibling_pos);
            siblings_pos_ndup.append("-");
            old_sibling_pos = sibling_pos;
        }
        if (sibling_rel != old_sibling_rel) {
            siblings_rel_ndup.append(sibling_rel);
            siblings_rel_ndup.append("-");
            old_sibling_rel = sibling_rel;
        }
    }

    set_feature_value_(FEAT_SIB_POS, row, siblings_pos);
    set_feature_value_(FEAT_SIB_REL, row, siblings_rel);
    set_feature_value_(FEAT_SIB_POS_NDUP, row, siblings_pos_ndup);
    set_feature_value_(FEAT_SIB_REL_NDUP, row, siblings_rel_ndup);
}

void FeatureExtractor::fg_predicate_children_pattern_( const size_t row )
{
    typedef SRLTree::sibling_iterator Iter;
    Iter   predicate_node = mp_sentence->get_node_of_row(m_predicate_row);

    string children_pos;
    string children_rel;
    string children_pos_ndup;
    string children_rel_ndup;

    string child_pos;
    string child_rel;
    string old_child_pos;
    string old_child_rel;

    for (Iter child = predicate_node.begin();
        child != predicate_node.end();
        ++child)
    {
        child_pos = mp_sentence->get_PPOS(*child);
        child_rel = mp_sentence->get_PDEPREL(*child);

        children_pos.append(child_pos);
        children_pos.append("-");
        children_rel.append(child_rel);
        children_rel.append("-");

        if (child_pos != old_child_pos) 
        {
            children_pos_ndup.append(child_pos);
            children_pos_ndup.append("-");
            old_child_pos = child_pos;
        }
        if (child_rel != old_child_rel) 
        {
            children_rel_ndup.append(child_rel);
            children_rel_ndup.append("-");
            old_child_rel = child_rel;
        }
    }

    set_feature_value_(FEAT_PRED_CHD_POS, m_predicate_row, children_pos);
    set_feature_value_(FEAT_PRED_CHD_REL, m_predicate_row, children_rel);
    set_feature_value_(FEAT_PRED_CHD_POS_NDUP, m_predicate_row, children_pos_ndup);
    set_feature_value_(FEAT_PRED_CHD_REL_NDUP, m_predicate_row, children_rel_ndup);
}

void FeatureExtractor::fg_predicate_siblings_pattern_(const size_t row)
{
    typedef SRLTree::sibling_iterator Iter;
    const size_t parent_row = mp_sentence->get_PHEAD(m_predicate_row);
    Iter parent_node = mp_sentence->get_node_of_row(parent_row);

    string siblings_pos;
    string siblings_rel;
    string siblings_pos_ndup;
    string siblings_rel_ndup;

    string sibling_pos;
    string sibling_rel;
    string old_sibling_pos;
    string old_sibling_rel;

    for (Iter sib = parent_node.begin();
        sib != parent_node.end();
        ++sib)
    {
        sibling_pos = mp_sentence->get_PPOS(*sib);
        sibling_rel = mp_sentence->get_PDEPREL(*sib);
        siblings_pos.append(sibling_pos);
        siblings_pos.append("-");
        siblings_rel.append(sibling_rel);
        siblings_rel.append("-");

        if (sibling_pos != old_sibling_pos) 
        {
            siblings_pos_ndup.append(sibling_pos);
            siblings_pos_ndup.append("-");
            old_sibling_pos = sibling_pos;
        }
        if (sibling_rel != old_sibling_rel) 
        {
            siblings_rel_ndup.append(sibling_rel);
            siblings_rel_ndup.append("-");
            old_sibling_rel = sibling_rel;
        }
    }

    set_feature_value_(FEAT_PRED_SIB_POS, m_predicate_row, siblings_pos);
    set_feature_value_(FEAT_PRED_SIB_REL, m_predicate_row, siblings_rel);
    set_feature_value_(FEAT_PRED_SIB_POS_NDUP, m_predicate_row, siblings_pos_ndup);
    set_feature_value_(FEAT_PRED_SIB_REL_NDUP, m_predicate_row, siblings_rel_ndup);
}

void FeatureExtractor::fg_predicate_basic_( const size_t row )
{
    set_feature_value_(
        FEAT_PREDICATE, 
        m_predicate_row, 
        mp_sentence->get_FORM(m_predicate_row)
        );

    set_feature_value_(
        FEAT_PRED_LEMMA, 
        m_predicate_row, 
        mp_sentence->get_PLEMMA(m_predicate_row)
        );

    set_feature_value_(
        FEAT_PRED_SENSE,
        m_predicate_row,
        mp_sentence->get_PRED(m_predicate_row)
        );
}

void FeatureExtractor::fg_path_(const size_t row)
{
    const SRLTree&    parse_tree = mp_sentence->get_parse_tree();
    SRLTree::iterator node_iter  = mp_sentence->get_node_of_row(row);
    SRLTree::iterator parent     = parse_tree.parent(node_iter);

    if (row) // skip ROOT (0 == row)
    { 
        // HACK: detect whether the path feature of the parent node is set
        const string &path = get_feature_storage_(FEAT_PATH, *parent);
        if ("" == path) // parent not yet done, this node knows how to get to the predicate
        { 			
            if( row < *parent )//Left
			{
				 get_feature_storage_(FEAT_PATH, *parent) 
                    = mp_sentence->get_PPOS(*parent)
                    + "<L#"
                    + get_feature_storage_(FEAT_PATH, row);

			     get_feature_storage_(FEAT_REL_PATH, *parent) 
                    = "<L#"
                    + mp_sentence->get_PDEPREL(row)
                    + get_feature_storage_(FEAT_REL_PATH, row);
			}
			else//Right
			{
			     get_feature_storage_(FEAT_PATH, *parent) 
                    = mp_sentence->get_PPOS(*parent)
                    + "<R#"
                    + get_feature_storage_(FEAT_PATH, row);

			     get_feature_storage_(FEAT_REL_PATH, *parent) 
                    = "<R#"
                    + mp_sentence->get_PDEPREL(row)
                    + get_feature_storage_(FEAT_REL_PATH, row);
			}
            get_feature_storage_(FEAT_UP_PATH, *parent) 
                = get_feature_storage_(FEAT_UP_PATH, row);

            get_feature_storage_(FEAT_UP_REL_PATH, *parent) 
                = get_feature_storage_(FEAT_UP_REL_PATH, row);
        }
        else
        { // parent path already got (parent knows the path to the predicate)
			if(row < *parent)//Left
			{
                get_feature_storage_(FEAT_PATH, row) 
                    = mp_sentence->get_PPOS(row) 
				    + ">L#"
                    + get_feature_storage_(FEAT_PATH, *parent);

                get_feature_storage_(FEAT_UP_PATH, row) 
                    = mp_sentence->get_PPOS(row) 
				    + ">L#"
                    + get_feature_storage_(FEAT_UP_PATH, *parent);

                get_feature_storage_(FEAT_REL_PATH, row) 
                    = mp_sentence->get_PDEPREL(row)
				    + ">L#"
                    + get_feature_storage_(FEAT_REL_PATH, *parent);

                get_feature_storage_(FEAT_UP_REL_PATH, row) 
                    = mp_sentence->get_PDEPREL(row)
				    + ">L#"
                    + get_feature_storage_(FEAT_UP_REL_PATH, *parent);
			}
			else//Right
			{
                get_feature_storage_(FEAT_PATH, row) 
                    = mp_sentence->get_PPOS(row) 
				    + ">R#"
                    + get_feature_storage_(FEAT_PATH, *parent);		

                get_feature_storage_(FEAT_UP_PATH, row) 
                    = mp_sentence->get_PPOS(row) 
				    + ">R#"
                    + get_feature_storage_(FEAT_UP_PATH, *parent);

                get_feature_storage_(FEAT_REL_PATH, row) 
                    = mp_sentence->get_PDEPREL(row)
				    + ">R#"
                    + get_feature_storage_(FEAT_REL_PATH, *parent);

                get_feature_storage_(FEAT_UP_REL_PATH, row) 
                    = mp_sentence->get_PDEPREL(row)
				    + ">R#"
                    + get_feature_storage_(FEAT_UP_REL_PATH, *parent);
			}

        }
    }

    set_feature_empty_(FEAT_PATH,          row, false);
    set_feature_empty_(FEAT_UP_PATH,       row, false);
    set_feature_empty_(FEAT_REL_PATH,      row, false);
    set_feature_empty_(FEAT_UP_REL_PATH,   row, false);
}

void FeatureExtractor::fg_path_length_(const size_t row)
{
    const std::string& path = get_feature_value_(FEAT_PATH, row);
    const std::string& up_path = get_feature_value_(FEAT_UP_PATH, row);

    int up_path_len   = std::count(path.begin(), path.end(), '>');
    int down_path_len = std::count(path.begin(), path.end(), '<');
    int path_length   = up_path_len + down_path_len;

    get_feature_storage_(FEAT_PATH_LENGTH, row)   = int2string(path_length);
    get_feature_storage_(FEAT_UP_PATH_LEN, row)   = int2string(up_path_len);
    get_feature_storage_(FEAT_DOWN_PATH_LEN, row) = int2string(down_path_len);

    set_feature_empty_(FEAT_PATH_LENGTH,   row, false);
    set_feature_empty_(FEAT_UP_PATH_LEN,   row, false);
    set_feature_empty_(FEAT_DOWN_PATH_LEN, row, false);


/*    if (row) // skip ROOT (0 == row)
    {
        const int parent_path_length
            = string2int(get_feature_storage_(FEAT_PATH_LENGTH, *parent));

        if ( parent_path_length == 0 && *parent != m_predicate_row) // parent not yet done, this node knows how to get to the predicate
        { 
            get_feature_storage_(FEAT_PATH_LENGTH, *parent) =
                int2string(
                    string2int(get_feature_storage_(FEAT_PATH_LENGTH, row))+1);

            get_feature_storage_(FEAT_UP_PATH_LEN, *parent) =
                get_feature_storage_(FEAT_UP_PATH_LEN, row);

            get_feature_storage_(FEAT_DOWN_PATH_LEN, *parent) =
                int2string(
                string2int(get_feature_storage_(FEAT_DOWN_PATH_LEN, row)) + 1);
        }
        else // parent path length already got (parent knows the path length to the predicate)
        { 
             get_feature_storage_(FEAT_PATH_LENGTH, row) =
                int2string(
                string2int(get_feature_storage_(FEAT_PATH_LENGTH, *parent))+1);

            get_feature_storage_(FEAT_UP_PATH_LEN, row) =
                int2string(
                string2int(get_feature_storage_(FEAT_UP_PATH_LEN, *parent))+1);

            get_feature_storage_(FEAT_DOWN_PATH_LEN, row) =
                get_feature_storage_(FEAT_DOWN_PATH_LEN, *parent);
        }
    }
    */
}

void FeatureExtractor::fg_descendant_of_predicate_( const size_t row )
{
    const string&  up_path_length
        = get_feature_value_(FEAT_UP_PATH_LEN, row);
    const string& down_path_length
        = get_feature_value_(FEAT_DOWN_PATH_LEN, row);

    if ("0" == down_path_length && "0" != up_path_length) 
    {
        set_feature_value_(FEAT_DESC_OF_PD, row, "1");    
    }
    else 
    {
        set_feature_value_(FEAT_DESC_OF_PD, row, "0");
    }
}

void FeatureExtractor::fg_position_(const size_t row)
{
    if (row <= m_predicate_row)
    {
        set_feature_value_(FEAT_POSITION, row, "before");
    }
    else
    {
        set_feature_value_(FEAT_POSITION, row, "after");
    }
}

void FeatureExtractor::fg_predicate_familyship_( const size_t row )
{
    const string& up_path_length
        = get_feature_value_(FEAT_UP_PATH_LEN, row);
    const string& down_path_length
        = get_feature_value_(FEAT_DOWN_PATH_LEN, row);

    string familyship;

    if ("0" == down_path_length)
    {
        if ("0" == up_path_length)
        {
            familyship = "self";
        }
        else if ("1" == up_path_length)
        {
            familyship = "child";
        }
        else
        {
            familyship = "descendant";
        }
    }
    else if ("0" == up_path_length)
    {
        if ("1" == down_path_length)
        {
            familyship = "parent";
        }
        else
        {
            familyship = "ancestor";
        }
    }
    else if ("1" == up_path_length && "1" == down_path_length)
    {
        familyship = "sibling";
    }
    else
    {
        familyship = "not-relative";
    }

    set_feature_value_(FEAT_PRED_FAMILYSHIP, row, familyship);

}

void FeatureExtractor::fg_predicate_bag_of_words_(const size_t row) 
{
    const string& prefix   = get_feature_prefix_for_extractor(FEAT_BAG_OF_WORD)+"@";
    const size_t row_count = mp_sentence->get_row_count();

    string bag_of_words = "NONSENSE";

    for (size_t i=1; i<m_predicate_row; ++i) 
    {
        bag_of_words += " ";
        bag_of_words += prefix;
        bag_of_words += mp_sentence->get_FORM(i);
    }
    bag_of_words += " ";
    bag_of_words += prefix;
    bag_of_words += mp_sentence->get_FORM(m_predicate_row);
    for (size_t i=m_predicate_row+1; i<=row_count; ++i) {
        bag_of_words += " ";
        bag_of_words += prefix;
        bag_of_words += mp_sentence->get_FORM(i);
    }

    set_feature_value_(FEAT_BAG_OF_WORD, row, bag_of_words);

    string bag_of_words_add_des_of_pred = "";
    const string& new_prefix = get_feature_prefix_for_extractor(FEAT_BAG_OF_WORD_IS_DES_O_PRED)+"@";

    for (size_t i=1; i<=row_count; ++i)
    {
        if (bag_of_words_add_des_of_pred != "")
        {
            bag_of_words_add_des_of_pred += " ";
            bag_of_words_add_des_of_pred += new_prefix;
        }
        bag_of_words_add_des_of_pred += mp_sentence->get_FORM(i);
        bag_of_words_add_des_of_pred += "_";
        bag_of_words_add_des_of_pred += get_feature_value_(FEAT_DESC_OF_PD, i);
    }

    set_feature_value_(FEAT_BAG_OF_WORD_IS_DES_O_PRED, row, bag_of_words_add_des_of_pred);
}

void FeatureExtractor::fg_predicate_bag_of_words_ordered_(const size_t row) 
{
    const string& prefix   =get_feature_prefix_for_extractor(FEAT_BAG_OF_WORD_O)+"@";
    const size_t row_count = mp_sentence->get_row_count();

    string bag_of_words_o = "NONSENSE";

    for (size_t i=1; i<m_predicate_row; ++i) {
        bag_of_words_o += " ";
        bag_of_words_o += prefix;
        bag_of_words_o += mp_sentence->get_FORM(i);
        bag_of_words_o += "_l";
    }
    bag_of_words_o += " ";
    bag_of_words_o += prefix;
    bag_of_words_o += mp_sentence->get_FORM(m_predicate_row);
    bag_of_words_o += "_t";

    for (size_t i=m_predicate_row+1; i<=row_count; ++i) {
        bag_of_words_o += " ";
        bag_of_words_o += prefix;
        bag_of_words_o += mp_sentence->get_FORM(i);
        bag_of_words_o += "_r";
    }

    set_feature_value_(FEAT_BAG_OF_WORD_O, m_predicate_row, bag_of_words_o);
}

void FeatureExtractor::fg_predicate_bag_of_POSs_ordered_(const size_t row) 
{
    const string& prefix   = get_feature_prefix_for_extractor(FEAT_BAG_OF_POS_O)+"@";
    const size_t row_count = mp_sentence->get_row_count();

    string bag_of_POSs_o = "NONSENSE";

    for (size_t i=1; i<m_predicate_row; ++i) {
        bag_of_POSs_o += " ";
        bag_of_POSs_o += prefix;
        bag_of_POSs_o += mp_sentence->get_PPOS(i);
        bag_of_POSs_o += "_l";
    }
    bag_of_POSs_o += " ";
    bag_of_POSs_o += prefix;
    bag_of_POSs_o += mp_sentence->get_PPOS(m_predicate_row);
    bag_of_POSs_o += "_t";

    for (size_t i=m_predicate_row+1; i<=row_count; ++i) {
        bag_of_POSs_o += " ";
        bag_of_POSs_o += prefix;
        bag_of_POSs_o += mp_sentence->get_PPOS(i);
        bag_of_POSs_o += "_r";
    }

    set_feature_value_(FEAT_BAG_OF_POS_O, m_predicate_row, bag_of_POSs_o);

    string bag_of_POSs_o_w5 = "";
    const string& w5_prefix = get_feature_prefix_for_extractor(FEAT_BAG_OF_POS_O_W5) + "@";
    const size_t wind_begin = (m_predicate_row-5>1         ? m_predicate_row-5 : 1);
    const size_t wind_end   = (m_predicate_row+5<row_count ? m_predicate_row+5 : row_count);

    for (size_t i=wind_begin; i<m_predicate_row; ++i)
    {
        if (bag_of_POSs_o_w5 != "")
        {
            bag_of_POSs_o_w5 += " ";
            bag_of_POSs_o_w5 += w5_prefix;
        }
        bag_of_POSs_o_w5 += mp_sentence->get_PPOS(i);
        bag_of_POSs_o_w5 += "_l";
    }
    if (bag_of_POSs_o_w5!= "")
    {
        bag_of_POSs_o_w5 += " ";
        bag_of_POSs_o_w5 += w5_prefix;
    }
    bag_of_POSs_o_w5 += mp_sentence->get_PPOS(m_predicate_row);
    bag_of_POSs_o_w5 += "_t";

    for (size_t i=m_predicate_row+1; i<=wind_end; ++i)
    {
        bag_of_POSs_o_w5 += " ";
        bag_of_POSs_o_w5 += w5_prefix;
        bag_of_POSs_o_w5 += mp_sentence->get_PPOS(i);
        bag_of_POSs_o_w5 += "_r";
    }
    set_feature_value_(FEAT_BAG_OF_POS_O_W5, m_predicate_row, bag_of_POSs_o_w5);

}
void FeatureExtractor::fg_predicate_bag_of_POSs_window5_(const size_t row)
{
    const string& prefix = get_feature_prefix_for_extractor(FEAT_BAG_OF_POS_WIND5)+ "@";
    const size_t row_count = mp_sentence->get_row_count();

    string bag_of_POSs_window5 = "";
    const size_t wind_begin = (m_predicate_row-5>1         ? m_predicate_row-5 : 1);
    const size_t wind_end   = (m_predicate_row+5<row_count ? m_predicate_row+5 : row_count);

    for (size_t i=wind_begin; i<=wind_end; ++i)
    {
        if (i != wind_begin)
        {
            bag_of_POSs_window5 += " ";
            bag_of_POSs_window5 += prefix;
        }
        bag_of_POSs_window5 += mp_sentence->get_PPOS(i);
    }
    set_feature_value_(FEAT_BAG_OF_POS_WIND5, m_predicate_row, bag_of_POSs_window5);
}

void FeatureExtractor::fg_predicate_bag_of_POSs_numbered_(const size_t row) 
{
    const string& prefix   = get_feature_prefix_for_extractor(FEAT_BAG_OF_POS_N)+"@";
    const size_t row_count = mp_sentence->get_row_count();

    stringstream bag_of_POSs_n;
    bag_of_POSs_n<<"NONSENSE";

    for (size_t i=m_predicate_row-1; i>=1; --i) {
        const int distance = int(i - m_predicate_row);
        bag_of_POSs_n
            <<" "
            <<prefix
            <<mp_sentence->get_PPOS(i)
            <<"_"
            <<distance;
    }
    bag_of_POSs_n
        <<" "
        <<prefix
        <<mp_sentence->get_PPOS(m_predicate_row)
        <<"_"
        <<0;
    for (size_t i=m_predicate_row+1; i<=row_count; ++i) {
        const int distance = int(i - m_predicate_row);
        bag_of_POSs_n
            <<" "
            <<prefix
            <<mp_sentence->get_PPOS(i)
            <<"_"
            <<distance;
    }

    set_feature_value_(FEAT_BAG_OF_POS_N, m_predicate_row, bag_of_POSs_n.str());

    const string& w5_prefix = get_feature_prefix_for_extractor(FEAT_BAG_OF_POS_N_W5)+"@";

    stringstream bag_of_POSs_n_w5;
    bool visit = false;
    const size_t wind_begin = (m_predicate_row-5>1         ? m_predicate_row-5 : 1);
    const size_t wind_end   = (m_predicate_row+5<row_count ? m_predicate_row+5 : row_count);

    for (size_t i=m_predicate_row-1; i>= wind_begin; --i)
    {
        const int distance = int(i-m_predicate_row);
        if (visit)
        {
            bag_of_POSs_n_w5
            <<" "<<w5_prefix;
        }
        else { visit = true;}
        bag_of_POSs_n_w5<<mp_sentence->get_PPOS(i)<<"_"<<distance;
    }
    if (visit)
    {
        bag_of_POSs_n_w5
        <<" "<<w5_prefix;
    }
    else { visit = true;}
    bag_of_POSs_n_w5<<mp_sentence->get_PPOS(m_predicate_row)<<"_"<<0;

    for (size_t i=m_predicate_row+1; i<=wind_end; ++i)
    {
        const int distance = int(i-m_predicate_row);
        bag_of_POSs_n_w5<<" "<<w5_prefix<<mp_sentence->get_PPOS(i)<<"_"<<distance;
    }
    set_feature_value_(FEAT_BAG_OF_POS_N_W5, m_predicate_row, bag_of_POSs_n_w5.str());

}

void FeatureExtractor::fg_predicate_window5_bigram_(const size_t row) 
{
    const string& prefix   = get_feature_prefix_for_extractor(FEAT_WIND5_BIGRAM)+"@";
    const size_t row_count = mp_sentence->get_row_count();

    string wind5_bigram = "NONSENSE";

    const size_t wind_begin = (m_predicate_row-5>1         ? m_predicate_row-5 : 1);
    const size_t wind_end   = (m_predicate_row+5<row_count ? m_predicate_row+5 : row_count);

    for (size_t i=wind_begin; i<wind_end; ++i) {
        wind5_bigram += " ";
        wind5_bigram += prefix;
        wind5_bigram += mp_sentence->get_FORM(i);
        wind5_bigram += "_";
        wind5_bigram += mp_sentence->get_FORM(i+1);
    }

    set_feature_value_(FEAT_WIND5_BIGRAM, m_predicate_row, wind5_bigram);

    const string& pos_prefix = get_feature_prefix_for_extractor(FEAT_WIND5_BIGRAM_POS)+"@";
    string wind5_bigram_pos = "";
    for (size_t i=wind_begin; i<wind_end; ++i)
    {
        if (i != wind_begin)
        {
            wind5_bigram_pos+=" ";
            wind5_bigram_pos +=pos_prefix;
        }
        wind5_bigram_pos +=mp_sentence->get_PPOS(i);
        wind5_bigram_pos +="_";
        wind5_bigram_pos +=mp_sentence->get_PPOS(i+1);
    }

    set_feature_value_(FEAT_WIND5_BIGRAM_POS, m_predicate_row, wind5_bigram_pos);
}


void FeatureExtractor::fg_verb_voice_en_(const size_t row)
{
    const string& PPOS  = mp_sentence->get_PPOS(row);
    const string& LEMMA = mp_sentence->get_PLEMMA(row);

    if (!m_configuration.is_verbPOS(PPOS))
    {
        set_feature_value_(FEAT_VERB_VOICE_EN, row, "NON_VERB");
    }
    else if ( ("VBN" == PPOS || "VBD" == PPOS)
        &&
        ("be" == get_feature_value_(FEAT_HEADWORD_LEMMA, row)
        || "get" == get_feature_value_(FEAT_HEADWORD_LEMMA, row)
        || "APPO" == get_feature_value_(FEAT_DEPREL, row))
        )
    {
        set_feature_value_(FEAT_VERB_VOICE_EN, row, "PASSIVE");
    }
    else
    {
        set_feature_value_(FEAT_VERB_VOICE_EN, row, "ACTIVE");
    }
}

void FeatureExtractor::fg_predicate_voice_en_(const size_t row)
{
    set_feature_value_(
        FEAT_PRED_VOICE_EN,
        row,
        get_feature_value_(FEAT_VERB_VOICE_EN, m_predicate_row)
    );
}

void FeatureExtractor::fg_feat_column(const size_t row)
{
    const string& pfeat = mp_sentence->get_PFEAT(row);
    if (pfeat == "_")
    {
        throw runtime_error("feat_column function cannot calc the pfeat column is empty");
    }
    map<string, string> feat_res = split_feat_(pfeat);

    if (feat_res.find("SubPOS") != feat_res.end())
    {
        set_feature_value_(FEAT_SUB_POS, row, feat_res["SubPOS"]);    
    }
    else
    {
        set_feature_value_(FEAT_SUB_POS, row, "");    
    }
}

void FeatureExtractor::fg_pfeat_column_(const size_t row)
{
    const string& pfeat = mp_sentence->get_PFEAT(row);
    if ("_" == pfeat)
    {
        set_feature_value_(FEAT_PFEAT_COLUMN, row, "");
        set_feature_value_(FEAT_PFEAT_EXC_NULL, row, "");
        return;
    }
    string prefix = get_feature_prefix_for_extractor(FEAT_PFEAT_COLUMN)+"@";

    string prefix_exc_null = get_feature_prefix_for_extractor(FEAT_PFEAT_EXC_NULL)+"@";

    vector<string> result = split_(pfeat, '|');
    sort(result.begin(), result.end());
    string pfeat_str = "";
    string pfeat_exc_null = "";

    string last_res;
    if (result.size() > 0)
    {
        last_res = result[0];
        pfeat_str+=last_res;
        pfeat_exc_null+=last_res;
    }
    for (size_t i=1; i<result.size(); ++i)
    {
        if (result[i] != last_res)
        {
            pfeat_str+=" ";
            pfeat_str+=prefix;
            pfeat_str+=result[i];

            pfeat_exc_null += " ";
            pfeat_exc_null += prefix_exc_null;
            pfeat_exc_null += result[i];

            last_res = result[i];
        }
    }
    set_feature_value_(FEAT_PFEAT_COLUMN, row, pfeat_str);
    set_feature_value_(FEAT_PFEAT_EXC_NULL, row, pfeat_exc_null);
}

void FeatureExtractor::fg_pfeat_(const size_t row)
{
    const string& pfeat = mp_sentence->get_PFEAT(row);
    set_feature_value_(FEAT_PFEAT, row, pfeat);
}

