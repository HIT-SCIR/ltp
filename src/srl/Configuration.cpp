/*
 * File Name     : Configuration.cpp
 * Author        : msmouse
 *
 * Updated by    : jiangfeng
 * Update Time   : 2013-08-21
 *
 */


#include "Configuration.h"
#include <stdexcept>
#include <fstream>
#include <iostream>

using namespace std;

void Configuration::load_xml(const string& filename)
{
    ifstream xml_file(filename.c_str());
    if (!xml_file)
    {
        throw runtime_error("Can't open the configuration file\n");
    }
    vector<string> lines;
    lines.clear();
    string line;
    while (getline(xml_file, line))
    {
        trim(line);
        lines.push_back(line);
    }

    parse(lines);
}

void Configuration::trim(string& line)
{
    size_t begin, end;
    begin = line.find_first_not_of(" \t\n");
    end   = line.find_last_not_of(" \t\n");
    line  = line.substr(begin,end+1-begin);
}

size_t Configuration::find(
    const vector<string>& lines,
    const string& tag) const
{
    for (size_t i=0; i<lines.size(); ++i)
    {
        if (tag == lines[i])
            return i;
    }
    throw runtime_error("Unknown tag in xml file");
}

void Configuration::parse(const vector<string>& lines)
{
    size_t language_begin  = find(lines, "<language>");
    size_t language_end    = find(lines, "</language>");

    size_t pred_rg_begin   = find(lines, "<features_pred_rg>");
    size_t pred_rg_end     = find(lines, "</features_pred_rg>");
    size_t pred_cl_begin   = find(lines, "<features_pred_cl>");
    size_t pred_cl_end     = find(lines, "</features_pred_cl>");

    size_t feat_begin      = find(lines, "<features_role_cl>");
    size_t feat_end        = find(lines, "</features_role_cl>");

    size_t noun_POS_begin  = find(lines, "<noun>");
    size_t noun_POS_end    = find(lines, "</noun>");
    size_t verb_POS_begin  = find(lines, "<verb>");
    size_t verb_POS_end    = find(lines, "</verb>");

    m_language = lines[language_begin+1];

    vector<string> vec;
    vec.clear();
    for (size_t i = feat_begin+1; i < feat_end; i++)
    {
        vec.push_back(lines[i]);
    }
    m_argu_config.set_feature_names(vec);

    vec.clear();
    for (size_t i = pred_rg_begin+1; i < pred_rg_end; i++)
    {
        vec.push_back(lines[i]);
    }
    m_pred_recog_config.set_feature_names(vec);

    vec.clear();
    for (size_t i = pred_cl_begin+1; i < pred_cl_end; i++)
    {
        vec.push_back(lines[i]);
    }
    m_pred_class_config.set_feature_names(vec);

    m_noun_POS.clear();
    for (size_t i=noun_POS_begin+1; i<noun_POS_end; i++)
    {
        m_noun_POS.push_back(lines[i]);
    }
    m_verb_POS.clear();
    for (size_t i=verb_POS_begin+1; i<verb_POS_end; i++)
    {
        m_verb_POS.push_back(lines[i]);
    }
}

bool Configuration::is_verbPOS(const string& POS) const
{
    for (size_t i=0; i<m_verb_POS.size(); ++i)
    {
        if (POS == m_verb_POS[i])
        {
            return true;
        }
    }
    return false;
}

bool Configuration::is_nounPOS(const string& POS) const
{
    for (size_t i=0; i<m_noun_POS.size(); ++i)
    {
        if (POS == m_noun_POS[i])
        {
            return true;
        }
    }
    return false;
}

