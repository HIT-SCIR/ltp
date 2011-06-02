#ifndef _CONFIGURATION_H_
#define _CONFIGURATION_H_

#include <string>
#include <vector>

// predicate class configuration
class PredClassConfig
{
public:
    PredClassConfig(){};
    
    // set features' name
    void set_feature_names(const std::vector<std::string>& _features)
    {
        m_feature_names = _features;
    }
    // get features' name
    const std::vector<std::string>& get_feature_names() const
    {
        return m_feature_names;
    }


private:
    // storage features' name
    std::vector<std::string> m_feature_names;
};

// Argument class configuration
class ArguConfig
{
public:
    ArguConfig(){};

    void set_noun_feature_names(const std::vector<std::string>& _features)
    {
        m_noun_feature_names = _features;
    }
    
    void set_verb_feature_names(const std::vector<std::string>& _features)
    {
        m_verb_feature_names = _features;
    }

    const std::vector<std::string>& get_noun_feature_names() const
    {
        return m_noun_feature_names;
    }

    const std::vector<std::string>& get_verb_feature_names() const
    {
        return m_verb_feature_names;
    }

private:
    // storage feature's name for noun predicate
    std::vector<std::string> m_noun_feature_names;
    // storage feature's name for verb predicate
    std::vector<std::string> m_verb_feature_names;
};

// PredClassConfig + ArguConfig + something
class Configuration
{
public:
    Configuration(){};

    explicit Configuration(const std::string filename)
    {
        load_xml(filename);
    }

    // loading configuration file
    void load_xml(const std::string& filename);

    // get the language of current configuration file
    std::string get_language() const
    {
        return m_language;
    }

    PredClassConfig& get_pred_class_config()
    {
        return m_pred_class_config;
    }
    ArguConfig& get_argu_config()
    {
        return m_argu_config;
    }
    const PredClassConfig& get_pred_class_config() const
    {
        return m_pred_class_config;
    }
    const ArguConfig& get_argu_config() const
    {
        return m_argu_config;
    }

    // whether the POS belong verb( or noun) preidcate
    bool is_verbPOS(const std::string& POS) const;
    bool is_nounPOS(const std::string& POS) const;

private:
    // parse the xml file (simple version)
    void parse(const std::vector<std::string>& lines);
    // trim 
    void trim(std::string& line);

    // find tag in lines and return the line index
    size_t find(const std::vector<std::string>& lines, const std::string& tag) const;

private:
    PredClassConfig m_pred_class_config;
    ArguConfig      m_argu_config;
    std::string     m_language;
    std::vector<std::string> m_noun_POS;
    std::vector<std::string> m_verb_POS;
};

#endif 
