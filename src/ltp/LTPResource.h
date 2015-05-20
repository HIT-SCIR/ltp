#ifndef __LTP_RESOURCE_H__
#define __LTP_RESOURCE_H__

#include <string>

class LTPResource {
public:
  LTPResource();
  ~LTPResource();

  /**
   * Load segmentor resource from model file. Return 0 on success,
   * otherwise -1.
   *
   *  @param[in]  model_file      the model_file
   *  @return     int             0 on success, otherwise -1
   */
  int LoadSegmentorResource(const char* model_file);
  int LoadSegmentorResource(const char* model_file, const char* lexicon);
  int LoadSegmentorResource(const std::string& model_file);
  int LoadSegmentorResource(const std::string& model_file, const std::string& lexicon);

  /**
   * load postagger resource from model file. Return 0 on success,
   * otherwise -1.
   *
   *  @param[in]  model_file
   *  @return     int             0 on success, otherwise -1
   */
  int LoadPostaggerResource(const char* model_file);
  int LoadPostaggerResource(const char* model_file, const char* lexicon);
  int LoadPostaggerResource(const std::string& model_file);
  int LoadPostaggerResource(const std::string& model_file, const std::string& lexicon);

  /**
   * load parser resource from model file. Return 0 on success,
   * otherwise -1.
   *
   *  @param[in]  model_file
   *  @return     int             0 on success, otherwise -1
   */
  int LoadNEResource(const char * model_file);
  int LoadNEResource(const std::string & model_file);

  /**
   * load parser resource from model file. Return 0 on success,
   * otherwise -1.
   *
   *  @param[in]  model_file
   *  @return     int             0 on success, otherwise -1
   */
  int LoadParserResource(const char* model_file);
  int LoadParserResource(const std::string& model_file);

  /**
   * load srl resource from model file. Return 0 on success,
   * otherwise -1.
   *
   *  @param[in]  model_file
   *  @return     int             0 on success, otherwise -1
   */
  int LoadSRLResource(const char* data_folder);
  int LoadSRLResource(const std::string& data_folder);

  void ReleaseSegmentorResource(void);
  void ReleasePostaggerResource(void);
  void ReleaseNEResource(void);
  void ReleaseParserResource(void);
  void ReleaseSRLResource(void);

  void* GetSegmentor();   // access the segmentor.
  void* GetPostagger();   // access the postagger.
  void* GetParser();      // access the parser.
  void* GetNER();         // access the ner.
private:
  void* m_segmentor;
  void* m_postagger;
  void* m_parser;
  void* m_ner;
private:
  bool m_isSegmentorResourceLoaded;
  bool m_isPostaggerResourceLoaded;
  bool m_isNEResourceLoaded;
  bool m_isParserResourceLoaded;
  bool m_isSRLResourceLoaded;
};

#endif      //  end for __LTP_RESOURCE_H__
