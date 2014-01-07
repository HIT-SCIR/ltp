/*
 * File Name     : SRLBaselineExt.cpp
 * Author        : msmouse
 *
 * Updated by    : jiangfeng
 * Update Time   : 2013-8-21
 *
 */

#include "SRLBaselineExt.h"
#include "Configuration.h"
#include "FeatureExtractor.h"
#include <iostream>

SRLBaselineExt::SRLBaselineExt(string configXml, string selectFeats)
    :SRLBaseline(configXml, selectFeats)
{
    m_configuration.load_xml(configXml);
    m_featureExtractor = new FeatureExtractor(m_configuration);
    m_featureCollection = new FeatureCollection();

    m_srlFeatureNumbers.clear();
    m_srlFeaturePrefixes.clear();
    m_prgFeatureNumbers.clear();
    m_prgFeaturePrefixes.clear();

    get_feature_config();
    open_select_config(selectFeats);
}

SRLBaselineExt::~SRLBaselineExt()
{
}

void SRLBaselineExt::ExtractPrgFeatures(vector< vector<string> >& vecPrgFeatures) const
{
    vecPrgFeatures.clear();

    Sentence sentence;

    vector<string> vecRows;
    convert2ConllFormat(vecRows);

    sentence.from_corpus_block(vecRows);
    const size_t row_count = sentence.get_row_count();

    m_featureExtractor->set_target_sentence(sentence);

    m_featureExtractor->calc_node_features();

    vector< vector<string> > vec_feature_values;
    for (size_t i = 0; i < m_prgFeatureNumbers.size(); ++i)
    {
        vector<string> feature_values;

        const int feature_number = m_prgFeatureNumbers[i];
        const string& feature_prefix = m_prgFeaturePrefixes[i];
        bool feature_empty_flag = false;
        try {
            m_featureExtractor->get_feature_for_rows(feature_number, feature_values);
        } catch (...) {
            feature_empty_flag = true;
        }

        if (feature_empty_flag)
        {
            feature_values.clear();
            for (size_t row = 0; row <= row_count; ++row)
            {
                feature_values.push_back("");
            }
        }

        vec_feature_values.push_back(feature_values);
    }

    for (size_t row = 1; row <= row_count; ++row)
    {
        vector<string> instance;
        for (size_t i = 0; i < m_prgFeatureNumbers.size(); ++i)
        {
            string feature = m_prgFeaturePrefixes[i] + "@"
                + vec_feature_values[i][row];
            instance.push_back(feature);
        }
        vecPrgFeatures.push_back(instance);
    }
}

void SRLBaselineExt::ExtractSrlFeatures(
        VecFeatForSent& vecAllFeatures,
        VecPosForSent& vecAllPos) const
{
    vecAllFeatures.clear();
    vecAllPos.clear();

    Sentence sentence;

    map<int, int> feat_number_index;
    feat_number_index.clear();

    for (size_t k = 0; k < m_srlFeatureNumbers.size(); ++k)
    {
        feat_number_index[m_srlFeatureNumbers[k]] = k;
    }

    vector<string> vecRows;
    convert2ConllFormat(vecRows);

    sentence.from_corpus_block(vecRows);
    const size_t predicate_count = sentence.get_predicates().size();
    const size_t row_count       = sentence.get_row_count();

    //feature_extractor.set_target_sentence(sentence);
    m_featureExtractor->set_target_sentence(sentence);
    vector<string> feature_values;
    vector< vector<string> > all_feature_values;

    // loop for each predicate
    for (size_t predicate_index = 0; predicate_index < predicate_count; ++predicate_index)
    {
        VecFeatForVerb vecFeatAllCons;
        VecFeatForCons vecForCons;
        VecPosForVerb vecPosVerb;

        int predID = m_vecPredicate[predicate_index];
        all_feature_values.clear();

        // calculate features
        //feature_extractor.calc_features(predicate_index);
        m_featureExtractor->calc_features(predicate_index);

        // loop for each feature
        for (size_t i = 0; i < m_srlFeatureNumbers.size(); ++i)
        {
            const int feature_number     = m_srlFeatureNumbers[i];
            const string& feature_prefix = m_srlFeaturePrefixes[i];
            bool feature_empty_flag = false;
            try
            {
                m_featureExtractor->get_feature_for_rows(feature_number, feature_values);
            }
            catch (...)
            {
                feature_empty_flag = true;
            }

            if (feature_empty_flag)
            {
                feature_values.clear();
                // loop for each row
                for (size_t row = 1; row <= row_count; ++row)
                {
                    feature_values.push_back("");
                }
            }

            all_feature_values.push_back(feature_values);
        }

        for (size_t row = 1; row <= row_count; ++row)
        {
            vecForCons.clear();
            if (IsFilter(row-1, predID))
                continue;
            for (size_t i = 0; i < m_srlSelectFeatures.size(); ++i)
            {
                string select_feature;
                select_feature.clear();
                for (size_t j = 0; j < m_srlSelectFeatures[i].size(); ++j)
                {
                    string feat_name = m_srlSelectFeatures[i][j];
                    int feat_number = m_featureCollection->get_feature_number(feat_name);
                    int value_index = feat_number_index[feat_number];
                    if (j == m_srlSelectFeatures[i].size()-1)
                        select_feature += m_srlFeaturePrefixes[value_index] + "@" + all_feature_values[value_index][row];
                    else
                        select_feature += m_srlFeaturePrefixes[value_index] + "@" + all_feature_values[value_index][row] + "+";
                }
                vecForCons.push_back(select_feature);
            }
            vecFeatAllCons.push_back(vecForCons);
        }

        vecAllFeatures.push_back(vecFeatAllCons);

        for (int nodeID = 0; nodeID < m_dataPreProc->m_intItemNum; nodeID++)
        {
            int predID = m_vecPredicate[predicate_index];
            if (!IsFilter(nodeID, predID))
            {
                //get position of unFiltered nodes, and push_back to vecPosVerb
                DepNode curNode;
                m_dataPreProc->m_myTree->GetNodeValue(curNode, nodeID);
                vecPosVerb.push_back(curNode.constituent);
            }
        }
        vecAllPos.push_back(vecPosVerb);
    }
}

void SRLBaselineExt::convert2ConllFormat(vector<string>& vecRows) const
{
    size_t row_count = m_dataPreProc->m_ltpData->vecWord.size();
    size_t predicate_count = m_vecPredicate.size();

    for (size_t id = 1; id <= row_count; ++id)
    {
        ostringstream row;
        row.str("");
        /*construct a line with element: word, pos, relation, .etc*/
        row << id; // first column: id
        row << " " << m_dataPreProc->m_ltpData->vecWord[id-1]; // second column: form
        row << " " << m_dataPreProc->m_ltpData->vecWord[id-1]; // third column: lemma, same with form
        row << " " << m_dataPreProc->m_ltpData->vecWord[id-1]; // forth column: plemma, same with lemma
        row << " " << m_dataPreProc->m_ltpData->vecPos[id-1]; // fifth column: pos
        row << " " << m_dataPreProc->m_ltpData->vecPos[id-1]; // sixth column: ppos, same with ppos
        row << " " << "_"; // 7th column: feat: null
        row << " " << "_"; // 8th column: pfeat: null

        if (m_dataPreProc->m_ltpData->vecParent[id-1] == -2)
        {
            row << " " << 0;
            row << " " << 0;
        }
        else
        {
            row << " " << m_dataPreProc->m_ltpData->vecParent[id-1] + 1;
            row << " " << m_dataPreProc->m_ltpData->vecParent[id-1] + 1;
        }

        row << " " << m_dataPreProc->m_ltpData->vecRelation[id-1]; //deprel
        row << " " << m_dataPreProc->m_ltpData->vecRelation[id-1]; //pdeprel

        if (count(m_vecPredicate.begin(), m_vecPredicate.end(), id - 1) != 0) // fillpred
        {
            row << " " << "Y";
            row << " " << "Y";
        }
        else
        {
            row << " " << "_";
            row << " " << "_";
        }

        for (size_t args = 0; args < predicate_count; ++args) // make room for args
            row << " " << "_";

        /*finish construct a line*/
        vecRows.push_back(row.str());
    }
}

void SRLBaselineExt::get_feature_config()
{
    /* feature set for role labeling */
    const vector<string>& argu_feat_set = m_configuration.get_argu_config().get_feature_names();

    /* feature set for predicate recognization */
    const vector<string>& prg_feat_set  = m_configuration.get_pred_recog_config().get_feature_names();

    m_srlFeatureNumbers.clear();
    m_srlFeaturePrefixes.clear();
    for (size_t i=0; i<argu_feat_set.size(); ++i)
    {
        const string& feature_name = argu_feat_set[i];
        const int feature_number 
            = m_featureCollection->get_feature_number(feature_name);
        const string& feature_prefix
            = m_featureCollection->get_feature_prefix(feature_number);

        if ( (find(m_srlFeatureNumbers.begin(),
                        m_srlFeatureNumbers.end(),
                        feature_number))
                == m_srlFeatureNumbers.end()) // not find
        {
            m_srlFeatureNumbers.push_back(feature_number);
            m_srlFeaturePrefixes.push_back(feature_prefix);
        }
    }

    m_prgFeatureNumbers.clear();
    m_prgFeaturePrefixes.clear();
    for (size_t i=0; i<prg_feat_set.size(); ++i)
    {
        const string& feature_name = prg_feat_set[i];
        const int feature_number
            = m_featureCollection->get_feature_number(feature_name);
        const string& feature_prefix
            = m_featureCollection->get_feature_prefix(feature_number);

        if ( (find(m_prgFeatureNumbers.begin(),
                        m_prgFeatureNumbers.end(),
                        feature_number)) == m_prgFeatureNumbers.end()) // not find
        {
            m_prgFeatureNumbers.push_back(feature_number);
            m_prgFeaturePrefixes.push_back(feature_prefix);
        }
    }
}

void SRLBaselineExt::open_select_config(string selectConfig)
{
    ifstream conf_input(selectConfig.c_str());
    if (!conf_input)
    {
        throw runtime_error("select_config file cannot open!");
    }
    m_srlSelectFeatures.clear();
    string line;
    while (getline(conf_input, line))
    {
        if ("" != line)
        {
            if ('#' == line[0])
            {
                continue;
            }
            vector<string> vec_str;
            replace(line.begin(), line.end(), '+', ' ');
            istringstream istr(line);
            string temp_str;
            while (istr >> temp_str)
            {
                vec_str.push_back(temp_str);
            }
            m_srlSelectFeatures.push_back(vec_str);
        }
    }
    conf_input.close();
}

bool SRLBaselineExt::IsFilter(int nodeID, int intCurPd) const
{
    DepNode depNode;
    m_dataPreProc->m_myTree->GetNodeValue(depNode, nodeID);

    //the punctuation nodes, current predicate node
    //changed for PTBtoDep, only filter the current predicate
    if( (nodeID == intCurPd) ||
            (depNode.parent < 0) ||
            ( (depNode.constituent.first <= intCurPd) &&
              (depNode.constituent.second >= intCurPd) ) )
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

