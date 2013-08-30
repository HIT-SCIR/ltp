/*
 * File Name     : GetInstance.h
 * Author        : msmouse
 * Create Time   : 2006-12-31
 * Project Name  : NewSRLBaseLine
 *
 * Updated by    : jiangfeng
 * Update Time   : 2013-08-21
 */

#ifndef _GET_INSTANCE_
#define _GET_INSTANCE_

#include <vector>
#include <string>
#include <fstream>
#include "Configuration.h"
#include "FeatureExtractor.h"

class GetInstance
{
    public:
        explicit GetInstance(const Configuration& configuration)
            : m_configuration(configuration)
        {
            m_opened_flags.resize(TOTAL_FEATURE, false);
        }


        void generate_argu_instance(
                const std::string& feature_folder,
                const std::string& select_config,
                const std::string& instance_file,
                bool is_devel=false);

    private:
        void open_select_config(const std::string& select_config);
        void close_();
        void read_line_(std::vector<std::string>& values);
        void test_and_open_(
                const std::vector<std::vector<std::string> >& select_features,
                const std::vector<std::string>& features,
                const std::string& feature_folder);

        void test_feature_set_(const std::string& feature_folder)
        {
            test_and_open_(
                    m_select_features,
                    m_configuration.get_argu_config().get_feature_names(),
                    feature_folder);
        }

        void output_(
                std::ofstream& out_stream,
                const std::vector<std::string> &values,
                const std::vector<std::vector<std::string> >& select_features);

    private:
        GetInstance(const GetInstance &);
        GetInstance & operator=(const GetInstance &);

    private:
        Configuration      m_configuration;
        std::ifstream      m_input_streams[TOTAL_FEATURE];
        std::ifstream      m_label_stream;
        FeatureCollection  m_feature_collection;
        std::vector<bool>  m_opened_flags;
        std::vector<std::vector<std::string> >  m_select_features;
};

#endif

