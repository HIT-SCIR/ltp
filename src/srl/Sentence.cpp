/*
 * File Name     : Sentence.cpp
 * Author        : msmouse
 * Create Time   : 2006-12-31
 * Project Name  : NewSRLBaseLine
 *
 * Updated by    : jiangfeng
 * Update Time   : 2013-08-21
 */


#include "Sentence.h"
#include <queue>
#include <sstream>
#include "boost/lexical_cast.hpp"


#define _DEBUG_
#ifdef _DEBUG_
#include <iostream>
#endif

using namespace std;

void Sentence::from_corpus_block(
    const std::vector<std::string> &corpus_block)
    // const Configuration& config)
{
    size_t row_count = corpus_block.size();

    // make room for data storage
    resize_(row_count);

    vector<vector<size_t> > children_of_node(row_count+1);

    // loop for each line
    for (size_t row=1; row <= row_count; ++row)
    {
        istringstream line_stream(corpus_block[row-1]); // row ID starts at 1

        size_t ID;
        line_stream>>ID;
        assert(row == ID);

        // get other fields;
        for (size_t field = FIELD_FORM; field < FIELD_NUMBER; ++field)
        {
            line_stream>>m_fields[row][field];
        }

        // get arguments
        size_t predicate_number = 0;
        string argument;
        while (line_stream>>argument)
        {
            ++predicate_number;

            if (predicate_number > m_argument_columns.size())
            {
                m_argument_columns.resize(predicate_number);
                m_argument_columns[predicate_number-1].push_back(string());
                // row starts at 1
            }

            if ("_" == argument)
            {
                m_argument_columns[predicate_number-1].push_back(string());
            }
            else
            {
                m_argument_columns[predicate_number-1].push_back(argument);
            }
        }

        // predicate
        if ("Y" == m_fields[row][FIELD_FILLPRED])
        {
            // m_predicates.push_back(Predicate(row, get_predicate_type_try_hard_(config,row)));
            m_predicates.push_back(Predicate(row));
        }

        // parent and child relationship
        size_t parent = boost::lexical_cast<size_t>(m_fields[row][FIELD_HEAD]);
        m_HEADs.push_back(parent);
        children_of_node[parent].push_back(row);
    }

    if (m_predicates.size() != m_argument_columns.size())
    {
        m_argument_columns.resize(m_predicates.size()); //proinsight
        // cout<<m_fields[1][FIELD_FORM]<<endl;
    }

    // assert(m_predicates.size() == m_argument_columns.size());

    // build parse_tree
    SRLTree::iterator node_iter;
    node_iter = m_parse_tree.set_head(0);

    m_node_of_row.resize(row_count+1);
    m_node_of_row[0] = node_iter; // store the ROOT

    queue<size_t> node_queue;
    node_queue.push(0);

    while (!node_queue.empty())
    {
        size_t node = node_queue.front();
        node_queue.pop();
        node_iter = m_node_of_row[node];

        for (int i = 0; i < children_of_node[node].size(); ++ i) {
          size_t child = children_of_node[node][i];
            m_node_of_row[child] = m_parse_tree.append_child(node_iter, child);
            node_queue.push(child);
        }
    }

}

const std::string Sentence::to_corpus_block() const
{
    ostringstream output_stream;

    size_t row_count = m_fields.size()-1;

    for (size_t row=1; row<=row_count; ++row)
    {
        // row ID
        output_stream<<row<<"\t";
        
        //for each field
        bool first_column = true;
        for(int field=FIELD_FORM; field<FIELD_NUMBER; ++field)
        {
            if (first_column)
            {
                first_column = false;
            }
            else
            {
                output_stream<<"\t";
            }
            output_stream<<get_field(row, field);
        }

        for (size_t predicate_index=0;
            predicate_index<m_predicates.size();
            ++predicate_index)
        {
            const string& argument = get_argument(predicate_index, row);
            if (argument.empty()) 
            {
                output_stream<<"\t"<<"_";
            }
            else
            {
                output_stream<<"\t"<<argument;
            }
        }
        output_stream<<endl;
    }
    return output_stream.str();
}

void Sentence::set_predicates(const std::vector<size_t> &predicate_rows)
{
    m_predicates.clear();

    for (size_t row=1; row<=m_row_count; ++row)
    {
        m_fields[row][FIELD_PRED] = "_";
        m_fields[row][FIELD_FILLPRED] = "_";
    }
    for (size_t i=0; i<predicate_rows.size(); ++i)
    {
        const size_t row = predicate_rows[i];
        m_predicates.push_back(Predicate(row));

        m_fields[row][FIELD_FILLPRED] = "Y";
        m_fields[row][FIELD_PRED] = get_PLEMMA(row)+".01";
    }

    m_argument_columns.clear();
    m_argument_columns.resize(predicate_rows.size());
}


void Sentence::set_PRED(const size_t row, const std::string &PRED) // proinsght
{
    m_fields[row][FIELD_PRED] = PRED;
}

void Sentence::clear()
{
    m_fields.resize(boost::extents[0][0]);
    m_predicates.clear();
    m_argument_columns.clear();
    m_HEADs.clear();

    m_parse_tree.clear();
    m_node_of_row.clear();

    m_row_count = 0;
}

const std::string& Sentence::get_argument(const size_t predicate_index, const size_t row) const
{
    //very useful ,will not infulence the multi-thread
    static string empty = "";
    if (m_argument_columns[predicate_index].size())
    {
        return m_argument_columns[predicate_index][row];
    }
    else
    {
        return empty;
    }
}

void Sentence::set_argument(
        const size_t predicate_index,
        const size_t row,
        const std::string& argument_name)
{
    if (!m_argument_columns[predicate_index].size())
    {
        m_argument_columns[predicate_index].resize(m_row_count+1);
    }
    m_argument_columns[predicate_index][row] = argument_name;
}

// resize the sentence to row_count rows, all data will be erased
void Sentence::resize_(const size_t row_count)
{
    // make the sentence empty
    clear();

    // row ID starts at 1
    m_fields.resize(boost::extents[row_count+1][FIELD_NUMBER]);
    m_HEADs.push_back(static_cast<size_t>(-1));

    m_row_count = row_count;
}

