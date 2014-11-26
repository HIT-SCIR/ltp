#ifndef __LTP_RESOURCE_H__
#define __LTP_RESOURCE_H__

#include <string>

// extern ofstream ltp_log_file;

class LTPResource {
public:
    LTPResource();
    ~LTPResource();

    /*
     * Load segmentor resource from model file. Return 0 on success,
     * otherwise -1.
     *
     *  @param[in]  model_file      the model_file
     *  @return     int             0 on success, otherwise -1
     */
    int LoadSegmentorResource(const char * model_file);

    /*
     * std::string wrapper for LoadSegmentResource(const char *);
     */
    int LoadSegmentorResource(const std::string & model_file);

    /*
     * load postagger resource from model file. Return 0 on success,
     * otherwise -1.
     *
     *  @param[in]  model_file
     *  @return     int             0 on success, otherwise -1
     */
    int LoadPostaggerResource(const char * model_file);

    /*
     * std::string wrapper for LoadSegmentResouce(const char *);
     */
    int LoadPostaggerResource(const std::string & model_file);

    /*
     * load parser resource from model file. Return 0 on success,
     * otherwise -1.
     *
     *  @param[in]  model_file
     *  @return     int             0 on success, otherwise -1
     */
    int LoadNEResource(const char * model_file);

    /*
     * std::string wrapper for LoadNEResource(const char *)
     */
    int LoadNEResource(const std::string & model_file);

    /*
     * load parser resource from model file. Return 0 on success,
     * otherwise -1.
     *
     *  @param[in]  model_file
     *  @return     int             0 on success, otherwise -1
     */
    int LoadParserResource(const char * model_file);

    /*
     * std::string wrapper for LoadParserResource(const char *)
     */
    int LoadParserResource(const std::string & data_folder);

    /*
     * load srl resource from model file. Return 0 on success,
     * otherwise -1.
     *
     *  @param[in]  model_file
     *  @return     int             0 on success, otherwise -1
     */
    int LoadSRLResource(const char * data_folder);

    /*
     * std::string wrapper for LoadParserResource(const char *)
     */
    int LoadSRLResource(const std::string & data_folder);

    /*
     * release segmentor resource
     */
    void ReleaseSegmentorResource(void);

    /*
     * release postagger resource
     */
    void ReleasePostaggerResource(void);

    /*
     * release ner resource
     */
    void ReleaseNEResource(void);

    /*
     * release parser resource
     */
    void ReleaseParserResource(void);

    /*
     * release srl resource
     */
    void ReleaseSRLResource(void);

    /*
     * Get the segmentor
     *
     *  @return void *  pointer to the segmentor
     */
    void * GetSegmentor();

    /*
     * Get the postagger
     *
     *  @return void *  pointer to the postagger
     */
    void * GetPostagger();

    /*
     * Get the parser;
     *
     *  @return void *  pointer to the parser
     */
    void * GetParser();

    /*
     * Get the ner
     *
     *  @return void *  pointer to the ner
     */
    void * GetNER();
private:
    void * m_segmentor;
    void * m_postagger;
    void * m_parser;
    void * m_ner;
private:
        // copy operator and assign operator is not allowed.

private:
    bool    m_isSegmentorResourceLoaded;
    bool    m_isPostaggerResourceLoaded;
    bool    m_isNEResourceLoaded;
    bool    m_isParserResourceLoaded;
    bool    m_isSRLResourceLoaded;
};

#endif      //  end for __LTP_RESOURCE_H__
