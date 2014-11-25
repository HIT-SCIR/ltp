#include "LTPResource.h"

#include "MyLib.h"
#include "Xml4nlp.h"
#include "SplitSentence.h"
#include "segmentor/segment_dll.h"
#include "segmentor/singleton_model.h"
#include "segmentor/customized_segment_dll.h"
#include "postag_dll.h"
#include "parser_dll.h"
#include "ner_dll.h"
#include "SRL_DLL.h"

#if _WIN32
#pragma warning(disable: 4786 4284)
#pragma comment(lib, "segmentor.lib")
#pragma comment(lib, "postagger.lib")
#pragma comment(lib, "parser.lib")
#pragma comment(lib, "ner.lib")
#pragma comment(lib, "srl.lib")
#endif

#include "logging.hpp"

namespace seg = ltp::segmentor;

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
// function wrapper of segmentor_create_segmentor
int LTPResource::LoadSegmentorResource(const char * model_file) {
  //  resource has be loaded.
  if (m_isSegmentorResourceLoaded) {
    return 0;
  }

  TRACE_LOG("Loading segmentor model from \"%s\" ...", model_file);

  if (!seg::SingletonModel::create_model(model_file)) {
    ERROR_LOG("Failed to load segmentor model");
  }

  m_segmentor = segmentor_create_segmentor();

  m_customized_segmentor = customized_segmentor_create_segmentor();

  m_isSegmentorResourceLoaded = true;
  TRACE_LOG("segmentor model is loaded.");
  return 0;
}

int LTPResource::LoadSegmentorResource(const std::string & model_file) {
  return LoadSegmentorResource(model_file.c_str());
}

void LTPResource::ReleaseSegmentorResource() {
  if (!m_isSegmentorResourceLoaded) {
    return;
  }

  segmentor_release_segmentor(m_segmentor);

  TRACE_LOG("segmentor model is released.");

  m_segmentor = 0;
  m_isSegmentorResourceLoaded = false;
}

void * LTPResource::GetSegmentor() {
  return m_segmentor;
}

void * LTPResource::GetCustomizedSegmentor() {
  return m_customized_segmentor;
}
/* ======================================================== *
 * Postagger related resource management                    *
 * ======================================================== */
int LTPResource::LoadPostaggerResource(const char * model_file) {
  if (m_isPostaggerResourceLoaded) {
    return 0;
  }

  TRACE_LOG("Loading postagger model from \"%s\" ...", model_file);

  m_postagger = postagger_create_postagger(model_file);

  if (0 == m_postagger) {
    ERROR_LOG("Failed to load postagger model");
    return -1;
  }

  m_isPostaggerResourceLoaded = true;
  TRACE_LOG("postagger model is loaded");
  return 0;
}

int LTPResource::LoadPostaggerResource(const std::string & model_file) {
  return LoadPostaggerResource(model_file.c_str());
}

void LTPResource::ReleasePostaggerResource() {
  if (!m_isPostaggerResourceLoaded) {
    return;
  }

  postagger_release_postagger(m_postagger);

  m_postagger = 0;
  m_isPostaggerResourceLoaded = false;
}

void * LTPResource::GetPostagger() {
  return m_postagger;
}

/* ======================================================== *
 * NER related resource management                          *
 * ======================================================== */
int LTPResource::LoadNEResource(const char * model_file) {
  if (m_isNEResourceLoaded) {
    return 0;
  }

  TRACE_LOG("Loading NER resource from \"%s\"", model_file);

  m_ner = ner_create_recognizer(model_file);

  if (0 == m_ner) {
    ERROR_LOG("Failed to load ner model");
    return -1;
  }

  m_isNEResourceLoaded = true;
  TRACE_LOG("NER resource is loaded.");
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
  TRACE_LOG("NER resource is released");
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

  TRACE_LOG("Loading parser resource from \"%s\"", model_file);

  m_parser = parser_create_parser(model_file);
  if (!m_parser) {
    ERROR_LOG("Failed to create parser");
    return -1;
  }

  TRACE_LOG("Parser is loaded.");

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
  TRACE_LOG("Parser is released");

  m_parser = NULL;
  m_isParserResourceLoaded = false;
}

void * LTPResource::GetParser() {
  return m_parser;
}

/* ======================================================== *
 * SRL related resource management                          *
 * ======================================================== */
int LTPResource::LoadSRLResource(const char *data_folder) {
  if (m_isSRLResourceLoaded) {
    return 0;
  }

  TRACE_LOG("Loading SRL resource from \"%s\"", data_folder);

  if (0 != SRL_LoadResource(string(data_folder))) {
    ERROR_LOG("Failed to load SRL resource.");
    return -1;
  }

  TRACE_LOG("SRL resource is loaded.");
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

  if (0 != SRL_ReleaseResource()) {
    ERROR_LOG("Failed to release SRL resource");
    return;
  }

  TRACE_LOG("SRL is released");

  m_isSRLResourceLoaded = false;
  return;
}

