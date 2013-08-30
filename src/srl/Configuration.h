/*
 * File Name     : Configuration.h
 * Author        : msmouse
 *
 * Updated by    : jiangfeng
 * Update Time   : 2013-08-21
 *
 */


#ifndef _CONFIGURATION_H_
#define _CONFIGURATION_H_

#include <string>
#include <vector>

class PredClassConfig
{
    /* config for predicate classifier */
    public:
        PredClassConfig() {};

        void set_feature_names(const std::vector<std::string>& _features)
        {
            m_feature_names = _features;
        }

        const std::vector<std::string>& get_feature_names() const
        {
            return m_feature_names;
        }

    private:
        std::vector<std::string> m_feature_names;
};

class PredRecogConfig
{
    /* config for predicate recognizer */
    public:
        PredRecogConfig() {};

        void set_feature_names(const std::vector<std::string>& _features)
        {
            m_feature_names = _features;
        }

        const std::vector<std::string>& get_feature_names() const
        {
            return m_feature_names;
        }

    private:
        std::vector<std::string> m_feature_names;
};

class ArguConfig
{
    /* config for semantic role classifer */
    public:
        ArguConfig(){};

        void set_feature_names(const std::vector<std::string>& _features)
        {
            m_feature_names = _features;
        }

        const std::vector<std::string>& get_feature_names() const
        {
            return m_feature_names;
        }

    private:
        std::vector<std::string> m_feature_names;
};

class Configuration
{
    public:
        Configuration(){};

        explicit Configuration(const std::string filename)
        {
            load_xml(filename);
        }

        void load_xml(const std::string& filename);

        std::string get_language() const
        {
            return m_language;
        }

        PredClassConfig& get_pred_class_config()
        {
            return m_pred_class_config;
        }

        PredRecogConfig& get_pred_recog_config()
        {
            return m_pred_recog_config;
        }

        ArguConfig& get_argu_config()
        {
            return m_argu_config;
        }

        const PredClassConfig& get_pred_class_config() const
        {
            return m_pred_class_config;
        }

        const PredRecogConfig& get_pred_recog_config() const
        {
            return m_pred_recog_config;
        }

        const ArguConfig& get_argu_config() const
        {
            return m_argu_config;
        }

        bool is_verbPOS(const std::string& POS) const;
        bool is_nounPOS(const std::string& POS) const;

    private:
        /* parse the xml file (simple version) */
        void parse(const std::vector<std::string>& lines);

        /* trim */
        void trim(std::string& line);

        size_t find(const std::vector<std::string>& lines, const std::string& tag) const;

    private:
        PredClassConfig m_pred_class_config;
        PredRecogConfig m_pred_recog_config;
        ArguConfig      m_argu_config;

        std::string     m_language;

        std::vector<std::string> m_noun_POS;
        std::vector<std::string> m_verb_POS;
};

#endif 
