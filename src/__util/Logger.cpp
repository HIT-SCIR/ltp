// common logger
// Created by Gu Xu, 2006.6
// Modifief by Victor Gao, 2006.12
///////////////////////////////////////////////////////////////

#include <sys/timeb.h>
#include <time.h>
#include <string.h>

#include "Logger.h"

CLogger::CLogger(const char *pszLogPath, int DefLogLevel)
: m_OutputLevel(0), m_DefLogLevel(DefLogLevel), m_MsgId(0),
  _strLogPath(pszLogPath)
{
	InitializeCriticalSection(&m_csLogger);
#ifdef _WIN32
	::CreateDirectoryA(pszLogPath, NULL);
#endif
#ifdef _LINUX
	mkdir(pszLogPath, 0777);
#endif
}

CLogger::~CLogger(void)
{
#ifdef _WIN32
	::DeleteCriticalSection(&m_csLogger);
#endif
}

void CLogger::InitializeCriticalSection(semaphore *s)
{
#ifdef _WIN32
	::InitializeCriticalSection(s);
#endif
#ifdef _LINUX
	sem_init( s, 0, 1 /*nInitValue*/);
#endif
}

void CLogger::EnterCriticalSection(semaphore *s)
{
#ifdef _WIN32
	::EnterCriticalSection(s);
#endif
#ifdef _LINUX
	sem_wait(s);
#endif
}

bool CLogger::FileExists(const char *szFileName)
{
#ifdef _WIN32
	WIN32_FIND_DATAA findFileData;
	HANDLE hFind = FindFirstFileA(szFileName, &findFileData);
	return hFind != INVALID_HANDLE_VALUE;
#else
	FILE *fp = fopen(szFileName, "rb");
	if( fp == NULL ) return false;
	fclose(fp);
	return true;
#endif
}

void CLogger::LeaveCriticalSection(semaphore *s)
{
#ifdef _WIN32
	::LeaveCriticalSection(s);
#endif
#ifdef _LINUX
	sem_post(s);
#endif
}

int CLogger::GetCurrentThreadId()
{
#ifdef _WIN32
	return ::GetCurrentThreadId();
#endif
#ifdef _LINUX
	return pthread_self();
#endif
}

void CLogger::Log(int LogLevel, const char* MsgFmt, va_list argptr)
{
	// null output
	if (LogLevel < m_OutputLevel || !MsgFmt) return;

	// get current time
	time_t RawCurTime;
	time(&RawCurTime);
	struct tm* CurTime;
	CurTime = localtime( &RawCurTime );
	CurTime->tm_year += 1900;
	CurTime->tm_mon++; // adjust

	CLogger::EnterCriticalSection(&m_csLogger);

	// check current log file
	if (CurTime->tm_year != m_CurDate.Year || CurTime->tm_mon != m_CurDate.Month || CurTime->tm_mday != m_CurDate.Day)
	{
		// to do: a new file stream
		if (m_LogFile.is_open()) m_LogFile.close();

		char *StrPtr, *DesPtr;
		DesPtr = m_OutputBuf;
		StrPtr = const_cast<char*>(_strLogPath.c_str() );
		while (*StrPtr) *DesPtr++ = *StrPtr++;
		if (DesPtr[-1] != '\\' || DesPtr[-1] != '/') *DesPtr++ = '/';

		sprintf(DesPtr, "%04d-%02d-%02d.log", CurTime->tm_year, CurTime->tm_mon, CurTime->tm_mday);
		m_CurDate.Year = CurTime->tm_year; m_CurDate.Month = CurTime->tm_mon; m_CurDate.Day = CurTime->tm_mday;

		bool bAppend = CLogger::FileExists(m_OutputBuf);
		m_LogFile.open(m_OutputBuf,  std::ios::binary | std::ios_base::app); //std::ios_base::app | std::ios_base::out |
		//m_LogFile.seekp(0, std::ios_base::end);
		if( !m_LogFile )
		{
			printf("Can't create the log file %s!\n", m_OutputBuf);
			return;
		}
		if (bAppend)
		{
			m_LogFile.put('\r');
			m_LogFile.put('\n');
		}
		int StrLen = sprintf(m_OutputBuf, "-------- logged at %04d/%02d/%02d %02d:%02d:%02d --------",
			CurTime->tm_year, CurTime->tm_mon, CurTime->tm_mday, CurTime->tm_hour, CurTime->tm_min, CurTime->tm_sec);
		m_LogFile.write(m_OutputBuf, StrLen);
	}

	// output
	bool isNewLine = true;
	char *ptrTmpFmt = 0, *ptrStr = const_cast<char*>(MsgFmt);
	while (*ptrStr)
	{
		if (isNewLine)
		{
			int StrLen = sprintf(m_OutputBuf, "\r\n[%07d][%02d:%02d:%02d][%08X] ", m_MsgId, CurTime->tm_hour, CurTime->tm_min, CurTime->tm_sec, CLogger::GetCurrentThreadId());
			m_LogFile.write(m_OutputBuf, StrLen); isNewLine = false;
		}

		char ch = *ptrStr++;
		if (ptrTmpFmt)
		{
			*ptrTmpFmt++ = ch;

			int StrLen;
			bool bFlush = false;
			ch = tolower(ch) - 'a';
			if (ch < 0 || ch > 26)
			{
				if (ch == '%' || ch == '\\')
				{
					bFlush = true;
					StrLen = ptrTmpFmt - m_OutputBuf;
					ptrTmpFmt = m_OutputBuf;
				}
			}
			else
			{
				const unsigned int TypeMask = 0x94E17D;
				if (TypeMask & (1 << ch))
				{
					bFlush = true;
					if (ch == ('s' - 'a'))
					{
						ptrTmpFmt = va_arg(argptr, char*);
						StrLen = strlen(ptrTmpFmt);
					}
					else
					{
						*ptrTmpFmt++ = '\0';
						StrLen = sprintf(ptrTmpFmt, m_OutputBuf, va_arg(argptr, int));
					}
				}
			}
			if (bFlush)
			{
				m_LogFile.write(ptrTmpFmt, StrLen);
				ptrTmpFmt = 0;
			}
		}
		else
		{
			if (ch == '%')
			{
				ptrTmpFmt = m_OutputBuf;
				*ptrTmpFmt++ = ch;
			}
			else if (ch == '\n') isNewLine = true;
			else if (ch != '\r') m_LogFile.put(ch);
		}
	}
	m_LogFile.flush();
	m_MsgId ++;

	CLogger::LeaveCriticalSection(&m_csLogger);
}
