/*
 * vi:ts=4:tw=78:shiftwidth=4:expandtab
 * vim600:fdm=marker
 *
 * mmapfile.h  -  A platform independent mmap wrapper
 *
 * Copyright (C) 2004 by Zhang Le <ejoy@users.sourceforge.net>
 * Begin       : 29-May-2004
 * Last Change : 22-May-2005.
 * 
 * USEAGE:
 *   if mmap support is identified on the system on which this file is compiled,
 *   a macro HAVE_SYSTEM_MMAP will be defined. Then you can write your mmap
 *   supporting code like:
 *
 *   #if defined (HAVE_SYSTEM_MMAP)
 *      int rc;
 *      mmap_info mi;
 *      rc = mmap_open(&mi, file, mode, flags); // create a file mapping
 *      if (rc)
 *          ...  // mmap failed
 *      ...
 *
 *      `mi.addr' now points to the mapped address of the whole file
 *      `mi.size' is the size of the mmapped memory in bytes
 *      ...
 *
 *      rc = mmap_close(&mi);  // close the file mapping
 *      if (rc)
 *          ...  // unmmap failed
 *   #endif
 *
 *   Currently only mapping with shared reading is implemented. Do not attempt to
 *   write to the mapped address `addr'. The code should work on both unix
 *   platform that supports mmap(2) call or Win32 platform (mingw & cygwin). 
 *
 *   This file should work out of the box on Linux/FreeBSD/Win32. However on
 *   system unknown to the file, you may want to use autoconf macro
 *   AC_FUNC_MMAP to check the existence of working mmap(2) first. Otherwise
 *   these routines will not be available.
 *
 *   A C++ wrapper class is provided in `mmapfile.hpp'.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef MMAPFILE_H
#define MMAPFILE_H

/* checking for the existence of mmap call {{{*/
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#if defined(HAVE_MMAP) /* autoconf checks it for us */
    #define  USE_POSIX_MMAP
#else
    #if defined(__FreeBSD__) || defined(__linux__)
        #define USE_POSIX_MMAP
    #elif defined(WIN32) || defined(__CYGWIN__)
        #define USE_WIN32_MMAP
        #include <windows.h>
    #else
        #warning "Unknown platform for mmap wrapper"
    #endif
#endif

#if defined(USE_POSIX_MMAP) || defined(USE_WIN32_MMAP)
    #define HAVE_SYSTEM_MMAP 1
#else
    #undef HAVE_SYSTEM_MMAP
#endif
/*}}}*/

#if defined(HAVE_SYSTEM_MMAP)

#include <stdio.h>

#if defined(__cplusplus)
extern "C" {
#endif

typedef struct {
    unsigned long  size;
    int            fd;
    FILE          *fp;
    void          *addr;
    int            flags;

#if defined(USE_WIN32_MMAP)
    HANDLE win32_fh;
    HANDLE win32_view;
#endif
} mmap_info;

int mmap_open(mmap_info* info, const char* file, const char* mode, int flags);
int mmap_close(mmap_info* info);

#if defined(__cplusplus)
}
#endif

#endif /* ifndef HAVE_SYSTEM_MMAP */

#endif /* ifndef MMAPFILE_H */

