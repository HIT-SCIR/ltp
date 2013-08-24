/*
 * File Name     : GetInstance.cpp
 * Author        : msmouse
 * Create Time   : 2006-12-31
 * Project Name  : NewSRLBaseLine
 *
 * Updated by    : jiangfeng
 * Update Time   : 2013-08-21
 */


#include "GetInstance.h"

using namespace std;

void GetInstance::generate_argu_instance(
    const string& feature_folder,
    const string& select_config,
    const string& instance_file,
    bool is_devel)
{
    open_select_config(select_config);
    close_();
    test_feature_set_(feature_folder);

    // open
    ofstream inst_stream(instance_file.c_str());

    if (!inst_stream)
    {
        throw runtime_error("instance file cannot open");
    }

    vector<string> values;

    string tmp;
    tmp = feature_folder + "/labels";
    m_label_stream.open(tmp.c_str());
    if (! m_label_stream )
    {
        throw runtime_error(feature_folder+"/labels cannot open");
    }

    // output
    while ( getline(m_label_stream, tmp) )
    {
        if (tmp == "")
        {
            read_line_(values);
        }
        else
        {
            read_line_(values);
            if (! is_devel)
            {
                inst_stream << tmp << " ";
            }
            output_(inst_stream, values, m_select_features);
        }
    }

    inst_stream.close();
}

void GetInstance::output_(ofstream& out_stream,
        const vector<string>& values,
        const vector<vector<string> >& select_features)
{
    for (size_t i=0; i<select_features.size(); i++)
    {
        const vector<string>& com_feature = select_features[i];
        if (0 != i)
        {
            out_stream<<" ";
        }
        for (size_t j=0; j<com_feature.size(); ++j)
        {
            if (0 != j)
            {
                out_stream<<"+";
            }
            int feature_number = m_feature_collection.get_feature_number(com_feature[j]);
            out_stream<<values[feature_number];
        }
    }
    out_stream<<endl;
}

void GetInstance::open_select_config(const string& select_config)
{
    ifstream conf_input(select_config.c_str());
    if (!conf_input)
    {
        throw runtime_error("Select_config file cannot open!");
    }

    m_select_features.clear();
    string line;

    while (getline(conf_input, line))
    {
        if ("" != line)
        {
            if (line[0] == '#')
                continue;

            vector<string> vec_str;
            replace(line.begin(), line.end(), '+', ' ');
            istringstream istr(line);
            string temp_str;
            while (istr>>temp_str)
            {
                vec_str.push_back(temp_str);
            }

            m_select_features.push_back(vec_str);
        }
    }
    conf_input.close();
}

void GetInstance::close_()
{
    for (size_t i=0; i<TOTAL_FEATURE; ++i)
    {
        m_input_streams[i].close();
        m_input_streams[i].clear();
    }
    m_label_stream.close();
    m_label_stream.clear();
}

void GetInstance::read_line_(vector<string> &values)
{
    if (values.size() < TOTAL_FEATURE)
    {
        values.resize(TOTAL_FEATURE);
    }

    for (size_t feature_number =0; feature_number<TOTAL_FEATURE; feature_number++)
    {
        if (m_opened_flags[feature_number])
        {
            getline(m_input_streams[feature_number], values[feature_number]);
        }
    }
}

void GetInstance::test_and_open_(const vector<vector<string> >& select_features,
        const vector<string>& features,
        const string& feature_folder)
{
    // test the feature in select_config file exist in the language configruation
    m_opened_flags.resize(TOTAL_FEATURE, false);
    for (size_t i=0; i<select_features.size(); ++i)
    {
        const vector<string> & com_feature = select_features[i];

        for (size_t j=0; j<com_feature.size(); ++j)
        {
            const string& test_feature = com_feature[j];
            int feature_number = m_feature_collection.get_feature_number(test_feature);
            if (!m_opened_flags[feature_number])
            {
                string tmp = feature_folder+"/"+test_feature;
                m_input_streams[feature_number].open(tmp.c_str());
                if (!m_input_streams)
                {
                    throw runtime_error(tmp + " cannot open");
                }
                m_opened_flags[feature_number] = true;
            }

            if ( find(features.begin(), features.end(), test_feature) == features.end() ) 
            {
                throw runtime_error(test_feature+" is not in configuration");
            }
        }
    }
}

