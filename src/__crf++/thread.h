//
//  CRF++ -- Yet Another CRF toolkit
//
//  $Id: thread.h 1588 2007-02-12 09:03:39Z taku $;
//
//  Copyright(C) 2005-2007 Taku Kudo <taku@chasen.org>
//
#ifndef CRFPP_THREAD_H__
#define CRFPP_THREAD_H__

#ifdef HAVE_PTHREAD_H
#include <pthread.h>
#else
#ifdef _WIN32
#include <windows.h>
#include <process.h>
#endif
#endif

#if defined HAVE_PTHREAD_H
#define CRFPP_USE_THREAD 1
#endif

#if(defined(_WIN32) && !defined (__CYGWIN__))
#define CRFPP_USE_THREAD 1
#define BEGINTHREAD(src, stack, func, arg, flag, id) \
     (HANDLE)_beginthreadex((void *)(src), (unsigned)(stack), \
                       (unsigned(_stdcall *)(void *))(func), (void *)(arg), \
                       (unsigned)(flag), (unsigned *)(id))
#endif

namespace CRFPP {

  class thread {
  private:
#ifdef HAVE_PTHREAD_H
    pthread_t hnd_;
#else
#ifdef _WIN32
    HANDLE  hnd_;
#endif
#endif

  public:
    static void* wrapper(void *ptr) {
      thread *p = static_cast<thread *>(ptr);
      p->run();
      return 0;
    }

    virtual void run() {}

    void start() {
#ifdef HAVE_PTHREAD_H
      pthread_create(&hnd_, 0, &thread::wrapper,
                     static_cast<void *>(this));

#else
#ifdef _WIN32
      DWORD id;
      hnd_ = BEGINTHREAD(0, 0, &thread::wrapper, this, 0, &id);
#else
      run();
#endif
#endif
    }

    void join() {
#ifdef HAVE_PTHREAD_H
      pthread_join(hnd_, 0);
#else
#ifdef _WIN32
      WaitForSingleObject(hnd_, INFINITE);
      CloseHandle(hnd_);
#endif
#endif
    }

    virtual ~thread() {}
  };
}

#endif
