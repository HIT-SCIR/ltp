#include "LTPResource.h"
#include "xml4nlp/Xml4nlp.h"
#include "splitsnt/SplitSentence.h"
#include "segmentor/segment_dll.h"
#include "postagger/postag_dll.h"
#include "parser.n/parser_dll.h"
#include "ner/ner_dll.h"
#include "srl/SRL_DLL.h"
#include "utils/logging.hpp"

// #if _WIN32
// #pragma warning(disable: 4786 4284)
// #pragma comment(lib, "segmentor.lib")
// #pragma comment(lib, "postagger.lib")
// #pragma comment(lib, "parser.lib")
// #pragma comment(lib, "ner.lib")
// #pragma comment(lib, "srl.lib")
// #endif

LTPResource::LTPResource() :
  m_segmentor(NULL),
  m_postagger(NULL),
  m_ner(NULL),
  m_parser(NULL),
  m_isSegmentorResourceLoaded(false),
  m_isPostaggerResourceLoaded(false),
  m_isNEResourceLoaded(false),
  m_isParserResourceLoaded(false),
  m_isSRLResourceLoaded(false) {
}


LTPResource::~LTPResource() {
  ReleaseSegmentorResource();
  ReleasePostaggerResource();
  ReleaseNEResource();
  ReleaseParserResource();
  ReleaseSRLResource();
}

/* ======================================================== *
 * Segmentor related resource management                    *
 * ======================================================== */
int LTPResource::LoadSegmentorResource(const char * model_file) {
  if (m_isSegmentorResourceLoaded) { return 0; }

  INFO_LOG("Loading segmentor model from \"%s\" ...", model_file);
  m_segmentor = segmentor_create_segmentor(model_file);
  if (0 == m_segmentor) {
    ERROR_LOG("Failed to load segmentor model");
    return -1;
  }

  m_isSegmentorResourceLoaded = true;
  INFO_LOG("segmentor model is loaded.");
  return 0;
}

int LTPResource::LoadSegmentorResource(const char* model_file, const char* lexicon) {
  if (m_isSegmentorResourceLoaded) { return 0; }

  INFO_LOG("Loading segmentor model from \"%s\", \"%s\" ...", model_file, lexicon);
  m_segmentor = segmentor_create_segmentor(model_file, lexicon);
  if (0 == m_segmentor) {
    ERROR_LOG("Failed to load segmentor model");
    return -1;
  }

  m_isSegmentorResourceLoaded = true;
  INFO_LOG("segmentor model is loaded.");
  return 0;
}

int LTPResource::LoadSegmentorResource(const std::string& model_file) {
  return LoadSegmentorResource(model_file.c_str());
}

int LTPResource::LoadSegmentorResource(const std::string& model_file,
    const std::string& lexicon) {
  return LoadSegmentorResource(model_file.c_str(), lexicon.c_str());
}

void LTPResource::ReleaseSegmentorResource() {
  if (!m_isSegmentorResourceLoaded) { return; }

  segmentor_release_segmentor(m_segmentor);
  INFO_LOG("segmentor model is released.");
  m_segmentor = 0;
  m_isSegmentorResourceLoaded = false;
}

void* LTPResource::GetSegmentor() { return m_segmentor; }

/* ======================================================== *
 * Postagger related resource management                    *
 * ======================================================== */
int LTPResource::LoadPostaggerResource(const char * model_file) {
  if (m_isPostaggerResourceLoaded) { return 0; }

  INFO_LOG("Loading postagger model from \"%s\" ...", model_file);
  m_postagger = postagger_create_postagger(model_file);
  if (0 == m_postagger) {
    ERROR_LOG("Failed to load postagger model");
    return -1;
  }

  m_isPostaggerResourceLoaded = true;
  INFO_LOG("postagger model is loaded");
  return 0;
}


int LTPResource::LoadPostaggerResource(const char* model_file, const char* lexicon) {
  if (m_isPostaggerResourceLoaded) { return 0; }

  INFO_LOG("Loading postagger model from \"%s\" ...", model_file);
  m_postagger = postagger_create_postagger(model_file);
  if (0 == m_postagger) {
    ERROR_LOG("Failed to load postagger model");
    return -1;
  }

  m_isPostaggerResourceLoaded = true;
  INFO_LOG("postagger model is loaded");
  return 0;
}

int LTPResource::LoadPostaggerResource(const std::string& model_file) {
  return LoadPostaggerResource(model_file.c_str());
}

int LTPResource::LoadPostaggerResource(const std::string& model_file,
    const std::string& lexicon) {
  return LoadPostaggerResource(model_file.c_str(), lexicon.c_str());
}

void LTPResource::ReleasePostaggerResource() {
  if (!m_isPostaggerResourceLoaded) { return; }
  postagger_release_postagger(m_postagger);
  m_postagger = 0;
  m_isPostaggerResourceLoaded = false;
  INFO_LOG("postagger resource is released");
}

void * LTPResource::GetPostagger() { return m_postagger; }

/* ======================================================== *
 * NER related resource management                          *
 * ======================================================== */
int LTPResource::LoadNEResource(const char * model_file) {
  if (m_isNEResourceLoaded) {
    return 0;
  }

  INFO_LOG("Loading NER resource from \"%s\"", model_file);

  m_ner = ner_create_recognizer(model_file);

  if (0 == m_ner) {
    ERROR_LOG("Failed to load ner model");
    return -1;
  }

  m_isNEResourceLoaded = true;
  INFO_LOG("NER resource is loaded.");
  return 0;
}

int LTPResource::LoadNEResource(const std::string & model_file) {
  return LoadNEResource(model_file.c_str());
}

void LTPResource::ReleaseNEResource() {
  if (!m_isNEResourceLoaded) {
    return;
  }

  ner_release_recognizer(m_ner);

  m_ner = NULL;
  m_isNEResourceLoaded = false;
  INFO_LOG("NER resource is released");
}

void * LTPResource::GetNER() {
  return m_ner;
}

/* ====================================================== *
 * Parser related resource                                *
 * ====================================================== */
int LTPResource::LoadParserResource(const char * model_file) {
  if (m_isParserResourceLoaded) {
    return 0;
  }

  INFO_LOG("Loading parser resource from \"%s\"", model_file);

  m_parser = parser_create_parser(model_file);
  if (!m_parser) {
    ERROR_LOG("Failed to create parser");
    return -1;
  }

  INFO_LOG("parser is loaded.");

  m_isParserResourceLoaded = true;
  return 0;
}

int LTPResource::LoadParserResource(const std::string & model_file) {
  return LoadParserResource(model_file.c_str());
}

void LTPResource::ReleaseParserResource() {
  if (!m_isParserResourceLoaded) {
    return;
  }

  parser_release_parser(m_parser);
  INFO_LOG("Parser is released");

  m_parser = NULL;
  m_isParserResourceLoaded = false;
}

void * LTPResource::GetParser() {
  return m_parser;
}

/* ======================================================== *
 * srl related resource management                          *
 * ======================================================== */
int LTPResource::LoadSRLResource(const char *data_folder) {
  if (m_isSRLResourceLoaded) {
    return 0;
  }

  INFO_LOG("Loading srl resource from \"%s\"", data_folder);

  if (0 != srl_load_resource(string(data_folder))) {
    ERROR_LOG("Failed to load srl resource.");
    return -1;
  }

  INFO_LOG("srl resource is loaded.");
  m_isSRLResourceLoaded = true;
  return 0;
}

int LTPResource::LoadSRLResource(const std::string & data_folder) {
  return LoadSRLResource(data_folder.c_str());
}

void LTPResource::ReleaseSRLResource() {
  if (!m_isSRLResourceLoaded) {
    return;
  }

  if (0 != srl_release_resource()) {
    ERROR_LOG("Failed to release srl resource");
    return;
  }

  INFO_LOG("srl is released");

  m_isSRLResourceLoaded = false;
  return;
}
