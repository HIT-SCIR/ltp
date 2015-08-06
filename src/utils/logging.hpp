#ifndef __LTP_UTILS_LOGGING_HPP__
#define __LTP_UTILS_LOGGING_HPP__

#include "spthread.hpp"

#include <stdarg.h> // 
#include <stdio.h>  // for vfprintf
#include <stdlib.h> // for calloc
#include <string.h> // for strlen
#include <time.h>   // for time

#define LTP_LOG_TRACE      10000
#define LTP_LOG_DEBUG      20000
#define LTP_LOG_INFO       30000
#define LTP_LOG_WARNING    40000
#define LTP_LOG_ERROR      50000
#define LTP_LOG_FATAL      60000

namespace ltp {
namespace utility {
 
//! Trick, to make it header-only, a template class is needed.
template<typename T>
class Logger {
public:
  /**
   * Get instance of logger
   *
   *  @return     ul_logger   the logger instance
   */
  static Logger * get_logger() {
    if (_instance == NULL) {
      spthread_mutex_init(&mutex, NULL);
      spthread_mutex_lock(&mutex);

      if (_instance == NULL) {
        _instance = new Logger(NULL, LTP_LOG_INFO);
      }
      spthread_mutex_unlock(&mutex);
    }
    return _instance;
  }

  /**
   * If the logger haven't been instantiatd, instantiate with log file
   * path and write log level, else return
   *
   *  @param[in]  _filename   the filename
   *  @param[in]  _lvl        the log level
   *  @return     int         0 on success, otherwise -1
   */
  static inline int config(const char* _filename,
    int _lvl) {
    if (_instance != NULL) {
      return -1;
    }

    spthread_mutex_init(&mutex, NULL);
    spthread_mutex_lock(&mutex);
    if (_instance == NULL) {
      _instance = new Logger(_filename, _lvl);
    }
    spthread_mutex_unlock(&mutex);
    return 0;
  }
protected:
  /**
   * method for allocate a logger
   *
   *  @param  filename    the filename for log file
   *  @param  lvl         mininum log level
   */
  Logger(const char* _filename = NULL,
    int _lvl = 0) {

    if (!_filename || !(log_fpo = fopen(_filename, "w"))) {
      log_fpo = stderr;
    }

    log_lvl = _lvl;

    num_lvl_name_entries = 0;
    lvl_name_entries = new ul_logger_lvl_name_t[max_lvl_name_entries];

    ul_logger_lvl_name_t* entry = lvl_name_entries;
    entry->lvl = LTP_LOG_TRACE;
    entry->name = new char[strlen("TRACE") + 1]; strcpy(entry->name, "TRACE");
    entry++; ++num_lvl_name_entries;

    entry->lvl = LTP_LOG_DEBUG;
    entry->name = new char[strlen("DEBUG") + 1]; strcpy(entry->name, "DEBUG");
    entry++; ++num_lvl_name_entries;

    entry->lvl = LTP_LOG_INFO;
    entry->name = new char[strlen("INFO") + 1]; strcpy(entry->name, "INFO");
    entry++; ++num_lvl_name_entries;

    entry->lvl = LTP_LOG_WARNING;
    entry->name = new char[strlen("WARNING") + 1]; strcpy(entry->name, "WARNING");
    entry++; ++num_lvl_name_entries;

    entry->lvl = LTP_LOG_ERROR;
    entry->name = new char[strlen("ERROR") + 1]; strcpy(entry->name, "ERROR");
    entry++; ++num_lvl_name_entries;
    
    entry->lvl = LTP_LOG_FATAL;
    entry->name = new char[strlen("FATAL") + 1]; strcpy(entry->name, "FATAL");
    entry++; ++num_lvl_name_entries;
  }
public:
  /**
   * function for writing log
   *
   *  @param  lvl     log level
   *  @param  fmt     the format string
   *  @param  va_arg
   */
  inline void write_log(int lvl, const char * fmt, ...) {
    if (lvl < log_lvl) {
      return;
    }
    char buffer[80];
    
    time_t rawtime;
    struct tm * timeinfo;
    
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer, 80,
      "%Y-%m-%d %H:%M:%S",
      timeinfo);

    int i;
    for (i = 0; i < num_lvl_name_entries; ++i) {
      if (lvl_name_entries[i].lvl == lvl) {
        break;
      }
    }

    va_list lst;
    va_start(lst, fmt);

    spthread_mutex_lock(&mutex);

    fprintf(log_fpo, "[%s] %s ",
      (i < num_lvl_name_entries ? lvl_name_entries[i].name : "UNKNOWN"),
      buffer);

    vfprintf(log_fpo, fmt, lst);
    fprintf(log_fpo, "\n");
    fflush(log_fpo) ; // clear output buffer

    spthread_mutex_unlock(&mutex);

    va_end(lst);
    return;
  }


  void set_lvl(int lvl) {
    log_lvl = lvl;
  }

  /**
   *
   *
   *
   *
   */
  inline void regist_lvl(int lvl, const char * lvl_name) {
    if (num_lvl_name_entries >= max_lvl_name_entries) {
      return;
    }

    for (int i = 0; i < num_lvl_name_entries; ++i) {
      if ((lvl_name_entries + i)->lvl == lvl) {
        return;
      }
    }

    spthread_mutex_lock(&mutex);

    ul_logger_lvl_name_t * entry = lvl_name_entries + num_lvl_name_entries;
    entry->lvl = lvl;
    entry->name = new char[strlen(lvl_name) + 1];
    strcpy(entry->name, lvl_name);
    
    ++num_lvl_name_entries;
    spthread_mutex_unlock(&mutex);
    return;
  }
private:
  struct ul_logger_lvl_name_t {
    char *  name;
    int     lvl;
  };
private:
  static Logger* _instance;
  FILE * log_fpo;
  char * log_template;
  int    log_lvl;

  static const int max_lvl_name_entries = 20;
  int num_lvl_name_entries;
  ul_logger_lvl_name_t * lvl_name_entries;
  
  static spthread_mutex_t mutex;
};

}
}

#define TRACE_LOG(msg, ...) do { \
  ltp::utility::Logger<void>::get_logger()->write_log(LTP_LOG_TRACE, msg, ##__VA_ARGS__); \
} while (0);

#define INFO_LOG(msg, ...) do { \
  ltp::utility::Logger<void>::get_logger()->write_log(LTP_LOG_INFO, msg, ##__VA_ARGS__); \
} while (0);

#define DEBUG_LOG(msg, ...) do { \
  ltp::utility::Logger<void>::get_logger()->write_log(LTP_LOG_DEBUG, msg, ##__VA_ARGS__); \
} while (0);

#define WARNING_LOG(msg, ...) do { \
  ltp::utility::Logger<void>::get_logger()->write_log(LTP_LOG_WARNING, msg, ##__VA_ARGS__); \
} while (0);

#define ERROR_LOG(msg, ...) do { \
  ltp::utility::Logger<void>::get_logger()->write_log(LTP_LOG_ERROR, "%s: line %d: %s(): " msg, __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__); \
} while (0);

#define FATAL_LOG(msg, ...) do { \
  ltp::utility::Logger<void>::get_logger()->write_log(LTP_LOG_FATAL, "%s: line %d: %s(): " msg, __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__); \
} while (0);

template<typename T> ltp::utility::Logger<T>* ltp::utility::Logger<T>::_instance = NULL;
template<typename T> spthread_mutex_t ltp::utility::Logger<T>::mutex;

#endif  // end for __LTP_UTILS_LOGGING_HPP__

