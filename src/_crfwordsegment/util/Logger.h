// Judian Search logger (common)
// Created by Victor, 2007.1
///////////////////////////////////////////////////////////////

#ifndef _LOGGER_H_
#define _LOGGER_H_

#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef _LINUX
#include <semaphore.h>
#include <stdarg.h>		// for va_start(...)
#include <sys/types.h>	// for mkdir()
#include <sys/stat.h>
#include <limits.h>		// for MAX_PATH
#include <unistd.h>
#define MAX_PATH 1024
#endif


#ifdef _WIN32
typedef CRITICAL_SECTION	semaphore;	// type of semaphore in Win32
#endif
#ifdef _LINUX
typedef sem_t	semaphore;	// type of semaphore in Linux
#endif

class CLogger
{
public:
	CLogger(const char *pszLogPath, int DefLogLevel = 0x7FFFFFFF);
	~CLogger(void);

	int inline SetOutputLevel(int Level){
		int OldLevel = m_OutputLevel;
		m_OutputLevel = Level;
		return OldLevel;
	};

	void inline Log(const char* MsgFmt, ...) {
		va_list argptr; va_start(argptr, MsgFmt);
		Log(m_DefLogLevel, MsgFmt, argptr);
	};
	void inline Log(int LogLevel, const char* MsgFmt, ...) {
		va_list argptr; va_start(argptr, MsgFmt);
		Log(LogLevel, MsgFmt, argptr);
	}

private:
	void Log(int LogLevel, const char* MsgFmt, va_list argptr);

	struct {
		unsigned int Year;
		unsigned int Month;
		unsigned int Day;
	} m_CurDate;
	std::ofstream m_LogFile;
	char m_OutputBuf[MAX_PATH];

	int m_MsgId;
	int m_DefLogLevel, m_OutputLevel;

	semaphore m_csLogger;
	inline static void InitializeCriticalSection(semaphore *s);
	inline static void EnterCriticalSection(semaphore *s);
	inline static void LeaveCriticalSection(semaphore *s);
	inline static int GetCurrentThreadId();

	inline static bool FileExists(const char *szFileName);

	std::string _strLogPath;
};

#endif //_LOGGER_H_
