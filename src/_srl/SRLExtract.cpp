#include "Corpus.h"
#include "Configuration.h"
#include "Sentence.h"
#include "FeatureExtractor.h"
#include <vector>
#include <string>
#include <fstream>

using namespace std;

void print_usage(char* exe_path)
{
    string exe_name(exe_path);
    size_t find = exe_name.find_last_of("\\/");

    if (string::npos != find)
    {
        exe_name = exe_name.substr(find+1);
    }
    cout<<"This program extract all the features from CoNLL2009"<<endl
        <<"closed corpus and output each feature to a separate file."<<endl
        <<"Usage:"<<endl
        <<"      "<<exe_name<<" [config.xml/IN] [corpus/IN] [feature folder/OUT]"<<endl;
}

void open_files(const char* output_path_c,
                const Configuration& configuration,
                FeatureCollection &feature_collection,
                vector<int>& feature_numbers,
                vector<string>& feature_prefixes,
                ofstream* output_streams,
                ofstream& label_stream )
{
    string output_path(output_path_c);

    const vector<string> & noun_set = configuration.get_argu_config().get_noun_feature_names();
    const vector<string> & verb_set = configuration.get_argu_config().get_verb_feature_names();

    feature_numbers.clear();
    feature_prefixes.clear();
    for (size_t i=0; i<noun_set.size(); ++i)
    {
        const string& feature_name = noun_set[i];
        const int feature_number 
            = feature_collection.get_feature_number(feature_name);
        const string& feature_prefix
            = feature_collection.get_feature_prefix(feature_number);

        feature_numbers.push_back(feature_number);
        feature_prefixes.push_back(feature_prefix);

        string filename = output_path + "/" + feature_name;

        output_streams[feature_number].open(filename.c_str());
        if (!output_streams[feature_number])
        {
            throw runtime_error("can't open feature output file " + feature_name);
        }
    }
    for (size_t i=0; i<verb_set.size(); ++i)
    {
        const string& feature_name = verb_set[i];
        const int feature_number 
            = feature_collection.get_feature_number(feature_name);
        const string& feature_prefix
            = feature_collection.get_feature_prefix(feature_number);

        if ( (find(feature_numbers.begin(), 
                   feature_numbers.end(),
                   feature_number)) == feature_numbers.end()) // not find
        {
            feature_numbers.push_back(feature_number);
            feature_prefixes.push_back(feature_prefix);

            string filename = output_path + "/" + feature_name;

            output_streams[feature_number].open(filename.c_str());
            if (!output_streams[feature_number])
            {
                throw runtime_error("can't open feature output file " + feature_name);
            }
        }
    }

    string label_filename = output_path + "/labels";
    label_stream.open(label_filename.c_str());
    if (!label_stream)
    {
        throw runtime_error("can't open labels file");
    }

}

void extract_and_output(const char*          corpus_path,
                        const Configuration &configuration,
                        FeatureExtractor    &feature_extractor,
                        vector<int>         &feature_numbers,
                        vector<string>      &feature_prefixes,
                        ofstream            *output_streams,
                        ofstream            &label_stream)
{
    // extract features and output
    Corpus corpus(corpus_path);
    vector<string> lines;
    Sentence sentence;

    size_t sentence_count = 0;
    while (corpus.get_next_block(lines))
    {
        cout<<++sentence_count<<endl;

        sentence.from_corpus_block(lines, configuration);
        const size_t predicate_count = sentence.get_predicates().size();
        const size_t row_count       = sentence.get_row_count();

        feature_extractor.set_target_sentence(sentence);
        vector<string> feature_values;

        // loop for each predicate
        for (size_t predicate_index=0; predicate_index<predicate_count; ++predicate_index)
        {
            // calculate features
            feature_extractor.calc_features(predicate_index);

            // output 
            for (size_t i=0; i<feature_numbers.size(); ++i)
            {
                const int feature_number     = feature_numbers[i];
                const string& feature_prefix = feature_prefixes[i];
                bool feature_empty_flag = false;
                try{
                    feature_extractor.get_feature_for_rows(feature_number, feature_values);
                }
                catch(...)
                {
                    feature_empty_flag = true;
                }

                if (feature_empty_flag)  // empty
                {
                    for (size_t row=1; row<=row_count; ++row)
                    {
                        output_streams[feature_number]<<endl;
                    }
                }
                else
                {
                    for (size_t row=1; row<=row_count; ++row)
                    {
                        if (feature_prefix == "PFEATNULL" && feature_values[row]=="")
                        {
                            output_streams[feature_number]<<endl;
                        }
                        else
                        {
                        output_streams[feature_number]
                            <<feature_prefix
                            <<"@"
                            <<feature_values[row]
                            <<endl;
                        }
                    }
                }
                output_streams[feature_number]<<endl;
            }

            // output labels
            const int predicate_type = sentence.get_predicates()[predicate_index].type;
            if (Predicate::PRED_NOUN == predicate_type)
            {
                label_stream<<"[NOUN]"<<endl;
            }
            else
            {
                label_stream<<"[VERB]"<<endl;
            }

            for (size_t row=1; row<=row_count; ++row)
            {
                const string &argument = sentence.get_argument(predicate_index, row);
                if (argument.empty()) 
                {
                    label_stream<<"NULL"<<endl;
                }
                else
                {
                    label_stream<<argument<<endl;
                }
            }
            label_stream<<endl;
        }
    }
}

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        print_usage(argv[0]);
        return 1;
    }

    Configuration     configuration(argv[1]);
    FeatureExtractor  feature_extractor(configuration);
    FeatureCollection feature_collection;
    vector<int>       feature_numbers;
    vector<string>    feature_prefixes;
    
    ofstream output_streams[TOTAL_FEATURE];
    ofstream label_stream;

    open_files(argv[3], 
               configuration, 
               feature_collection, 
               feature_numbers, 
               feature_prefixes,
               output_streams,
               label_stream);
    extract_and_output(argv[2],
                       configuration,
                       feature_extractor,
                       feature_numbers,
                       feature_prefixes,
                       output_streams,
                       label_stream);
    for (size_t i=0; i<TOTAL_FEATURE; ++i)
        output_streams[i].close();
    label_stream.close();
    return 0;
}
