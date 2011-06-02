#include "SRLBaselineExt.h"
#include "Configuration.h"
#include "FeatureExtractor.h"

//////////////////////////////////////////////////////////////////////////
// constructor and destructor
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
SRLBaselineExt::SRLBaselineExt(string configXml, string selectFeats)
:SRLBaseline(configXml, selectFeats)
{
	m_configuration.load_xml(configXml);
	m_featureExtractor = new FeatureExtractor(m_configuration);
	m_featureCollection = new FeatureCollection();

	m_featureNumbers.clear();
	m_featurePrefixes.clear();

	get_feature_config();
	open_select_config(selectFeats);
}

//////////////////////////////////////////////////////////////////////////
SRLBaselineExt::~SRLBaselineExt()
{

}


//////////////////////////////////////////////////////////////////////////
// method
//////////////////////////////////////////////////////////////////////////

//Feature extracting method used in CoNLL2009. 
//add by jiangfeng. 2010.1.31
void SRLBaselineExt::ExtractFeatures(VecFeatForSent& vecAllFeatures, VecPosForSent& vecAllPos) const
{
	vecAllFeatures.clear();
	vecAllPos.clear();

	Sentence sentence;

	map<int, int> feat_number_index;
	feat_number_index.clear();

	for (size_t k = 0; k < m_featureNumbers.size(); ++k)
	{
		feat_number_index[m_featureNumbers[k]] = k;
	}

	vector<string> vecRows;
	convert2ConllFormat(vecRows);
	
	sentence.from_corpus_block(vecRows, m_configuration);
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
		for (size_t i = 0; i < m_featureNumbers.size(); ++i)
		{
			const int feature_number     = m_featureNumbers[i];
			const string& feature_prefix = m_featurePrefixes[i];
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
			for (size_t i = 0; i < m_selectFeatures.size(); ++i)
			{
				string select_feature;
				select_feature.clear();
				for (size_t j = 0; j < m_selectFeatures[i].size(); ++j)
				{
					string feat_name = m_selectFeatures[i][j];
					int feat_number = m_featureCollection->get_feature_number(feat_name);
					int value_index = feat_number_index[feat_number];
                    if (j == m_selectFeatures[i].size()-1)
    					select_feature += m_featurePrefixes[value_index] + "@" + all_feature_values[value_index][row];
                    else
    					select_feature += m_featurePrefixes[value_index] + "@" + all_feature_values[value_index][row] + "+";
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

//to construct a line of CoNLL2009 corpus format from ltpData.
//add by jiangfeng. 2010.1.31
void SRLBaselineExt::convert2ConllFormat(vector<string>& vecRows) const
{
	size_t row_count = m_dataPreProc->m_ltpData->vecWord.size();
	size_t predicate_count = m_vecPredicate.size();

	for (size_t id = 1; id <= row_count; ++id)
	{
		ostringstream row;
		row.str("");
		/*construct a line with element: word, pos, relation, .etc*/
		row << id << " "; // first column: id
		row << m_dataPreProc->m_ltpData->vecWord[id-1] << " "; // second column: form
		row << m_dataPreProc->m_ltpData->vecWord[id-1] << " "; // third column: lemma, same with form
		row << m_dataPreProc->m_ltpData->vecWord[id-1] << " "; // forth column: plemma, same with lemma
		row << m_dataPreProc->m_ltpData->vecPos[id-1] << " "; // fifth column: pos
		row << m_dataPreProc->m_ltpData->vecPos[id-1] << " "; // sixth column: ppos, same with ppos
		row << "_" << " "; // 7th column: feat: null
		row << "_" << " "; // 8th column: pfeat: null
		
        if (m_dataPreProc->m_ltpData->vecParent[id-1] == -2)
        {
            row << 0 << " ";
            row << 0 << " ";
        }
        else
        {
			row << m_dataPreProc->m_ltpData->vecParent[id-1] + 1 << " ";
			row << m_dataPreProc->m_ltpData->vecParent[id-1] + 1 << " ";
        }

		row << m_dataPreProc->m_ltpData->vecRelation[id-1] << " "; //deprel
		row << m_dataPreProc->m_ltpData->vecRelation[id-1] << " "; //pdeprel

		if (count(m_vecPredicate.begin(), m_vecPredicate.end(), id - 1) != 0) // fillpred
		{
			row << "Y" << " ";
            row << "Y" << " ";
		}
		else
        {
			row << "_" << " ";
            row << "_" << " ";
        }

		for (size_t args = 0; args < predicate_count - 1; ++args) // make room for args
			row << "_" << " ";
		row << "_";

		/*finish construct a line*/
		vecRows.push_back(row.str());
	}
}

//features need to be extracted are defined in the configure file "Chinese.xml"
//add by jiangfeng. 2010.1.31
void SRLBaselineExt::get_feature_config()
{
	const vector<string> & noun_set = m_configuration.get_argu_config().get_noun_feature_names();
    const vector<string> & verb_set = m_configuration.get_argu_config().get_verb_feature_names();
	
	m_featureNumbers.clear();
    m_featurePrefixes.clear();
    for (size_t i=0; i<noun_set.size(); ++i)
    {
        const string& feature_name = noun_set[i];
        const int feature_number 
            = m_featureCollection->get_feature_number(feature_name);
        const string& feature_prefix
            = m_featureCollection->get_feature_prefix(feature_number);

        m_featureNumbers.push_back(feature_number);
        m_featurePrefixes.push_back(feature_prefix);
    }

    for (size_t i=0; i<verb_set.size(); ++i)
    {
        const string& feature_name = noun_set[i];
        const int feature_number 
            = m_featureCollection->get_feature_number(feature_name);
        const string& feature_prefix
            = m_featureCollection->get_feature_prefix(feature_number);

        if ( (find(m_featureNumbers.begin(), 
                   m_featureNumbers.end(),
                   feature_number)) == m_featureNumbers.end()) // not find
        {
            m_featureNumbers.push_back(feature_number);
            m_featurePrefixes.push_back(feature_prefix);
        }
    }
}

//features to be used is defined in the configure file "conll2009-arg.conf"
//add by jiangfeng. 2010.1.31
void SRLBaselineExt::open_select_config(string selectConfig)
{
	ifstream conf_input(selectConfig.c_str());
	if (!conf_input)
	{
		throw runtime_error("select_config file cannot open!");
	}
	m_selectFeatures.clear();
	string line;
	while (getline(conf_input, line))
	{
		if (line == "[VERB]")
		{
			continue;
		}
		else if ("" != line)
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
			m_selectFeatures.push_back(vec_str);
		}
	}
	conf_input.close();
}

//////////////////////////////////////////////////////////////////////////
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
