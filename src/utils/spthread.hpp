/*
 * a portable thread library.
 *
 *  @author: CHEN, Xin {xchen@ir.hit.edu.cn}
 *  @author: LI, Zhenghua {lzh@ir.hit.edu.cn}
 *  @modifier:
 */
#ifndef __SPTHREAD_HPP__
#define __SPTHREAD_HPP__

/*
 + ======================================== +
 | The posix thread case                    |
 |                                          |
 | Used in linux and mac os                 |
 + ======================================== +
 */
#if !defined(_WIN32) && !defined(_WIN64)

#include <pthread.h>
#include <unistd.h>

typedef void * spthread_result_t;
typedef pthread_mutex_t spthread_mutex_t;
typedef pthread_cond_t  spthread_cond_t;
typedef pthread_t       spthread_t;
typedef pthread_attr_t  spthread_attr_t;

#define spthread_mutex_init(m,a)   pthread_mutex_init(m,a)
#define spthread_mutex_destroy(m)  pthread_mutex_destroy(m)
#define spthread_mutex_lock(m)     pthread_mutex_lock(m)
#define spthread_mutex_unlock(m)   pthread_mutex_unlock(m)

#define spthread_cond_init(c,a)    pthread_cond_init(c,a)
#define spthread_cond_destroy(c)   pthread_cond_destroy(c)
#define spthread_cond_wait(c,m)    pthread_cond_wait(c,m)
#define spthread_cond_signal(c)    pthread_cond_signal(c)

#define spthread_attr_init(a)        pthread_attr_init(a)
#define spthread_attr_setdetachstate pthread_attr_setdetachstate
#define SPTHREAD_CREATE_DETACHED     PTHREAD_CREATE_DETACHED

#define spthread_self    pthread_self
#define spthread_create  pthread_create
#define spthread_exit    pthread_exit

#define SP_THREAD_CALL
typedef spthread_result_t ( * spthread_func_t )( void * args );

#define spsleep(x) sleep(x)

#else   // ifndef WIN32
/*
 + ======================================== +
 | The win32 thread case                    |
 |                                          |
 | Used in windows                          |
 + ======================================== +
 */
#include <winsock2.h>
#include <process.h>

typedef unsigned spthread_t;

typedef unsigned spthread_result_t;
#define SP_THREAD_CALL __stdcall
typedef spthread_result_t ( __stdcall * sp_thread_func_t )( void * args );

typedef HANDLE  spthread_mutex_t;
typedef HANDLE  spthread_cond_t;
typedef DWORD   spthread_attr_t;

#define SP_THREAD_CREATE_DETACHED 1
#define spsleep(x) Sleep(1000*x)

inline int spthread_mutex_init( spthread_mutex_t * mutex, void * attr ) {
    *mutex = CreateMutex( NULL, FALSE, NULL );
    return NULL == * mutex ? GetLastError() : 0;
}

inline int spthread_mutex_destroy( spthread_mutex_t * mutex ){
    int ret = CloseHandle( *mutex );
    return 0 == ret ? GetLastError() : 0;
}

inline int spthread_mutex_lock( spthread_mutex_t * mutex ) {
    int ret = WaitForSingleObject( *mutex, INFINITE );
    return WAIT_OBJECT_0 == ret ? 0 : GetLastError();
}

inline int spthread_mutex_unlock( spthread_mutex_t * mutex ) {
    int ret = ReleaseMutex( *mutex );
    return 0 != ret ? 0 : GetLastError();
}

inline int spthread_cond_init( spthread_cond_t * cond, void * attr ) {
    *cond = CreateEvent( NULL, FALSE, FALSE, NULL );
    return NULL == *cond ? GetLastError() : 0;
}

inline int spthread_cond_destroy( spthread_cond_t * cond ) {
    int ret = CloseHandle( *cond );
    return 0 == ret ? GetLastError() : 0;
}

/*
   Caller MUST be holding the mutex lock; the
   lock is released and the caller is blocked waiting
   on 'cond'. When 'cond' is signaled, the mutex
   is re-acquired before returning to the caller.
   */
inline int spthread_cond_wait( spthread_cond_t * cond, spthread_mutex_t * mutex ) {
    int ret = 0;

    spthread_mutex_unlock( mutex );
    ret = WaitForSingleObject( *cond, INFINITE );
    spthread_mutex_lock( mutex );
    return WAIT_OBJECT_0 == ret ? 0 : GetLastError();
}


inline int spthread_cond_signal( spthread_cond_t * cond ) {
    int ret = SetEvent( *cond );
    return 0 == ret ? GetLastError() : 0;
}

inline spthread_t spthread_self() {
    return GetCurrentThreadId();
}

inline int spthread_attr_init( spthread_attr_t * attr ) {
    *attr = 0;
    return 0;
}

inline int spthread_attr_setdetachstate( spthread_attr_t * attr, int detachstate ) {
    *attr |= detachstate;
    return 0;
}

inline int spthread_create( spthread_t * thread, spthread_attr_t * attr,
        sp_thread_func_t myfunc, void * args ) {
    // _beginthreadex returns 0 on an error
    HANDLE h = (HANDLE)_beginthreadex( NULL, 0, myfunc, args, 0, thread );
    return h > 0 ? 0 : GetLastError();
}


#endif  // ifndef WIN32
#endif  // ifndef __SPTHREAD_HPP__
