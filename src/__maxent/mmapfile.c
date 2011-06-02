/*
 * vi:ts=4:tw=78:shiftwidth=4:expandtab
 * vim600:fdm=marker
 *
 * mmapfile.c  -  A platform independent mmap wrapper
 *
 * Copyright (C) 2004 by Zhang Le <ejoy@users.sourceforge.net>
 * Begin       : 29-May-2004
 * Last Change : 22-May-2005.
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

/*
#include <stdio.h>
#include <assert.h>
*/

#include "mmapfile.h"

#if defined(USE_POSIX_MMAP) /* {{{ */

#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>


int mmap_open(mmap_info* info, const char* file, const char* mode, int flags) {
    struct stat st;
    if (stat(file, &st)) {
        perror("can not stat file size");
        return 1;
    } else {
        info->size = st.st_size; 
    }

    info->fp = fopen(file, mode);
    if (info->fp == NULL) {
        perror("fail to open file");
        return 1;
    }
    info->fd = fileno(info->fp);

    info->addr = mmap(NULL, info->size, PROT_READ, MAP_SHARED, info->fd, 0);
    if (info->addr == MAP_FAILED) {
        perror("fail to mmap file");
        return 2;
    }

    return 0;
}

int mmap_close(mmap_info* info) {
    if (munmap(info->addr, info->size)) {
        perror("fail to munmap file");
        return 2;
    }

    if (fclose(info->fp)) {
        perror("fail to close file");
        return 1;
    }

    return 0;
}

#endif /* USE_POSIX_MMAP }}}*/

#if defined(USE_WIN32_MMAP) /* {{{ */
#include <sys/types.h>
#include <sys/stat.h>
#include <windows.h>

int mmap_open(mmap_info* info, const char* file, const char* mode, int flags) {
    DWORD access_mode = GENERIC_READ;
    DWORD map_mode    = PAGE_READONLY;
    DWORD view_mode   = FILE_MAP_READ;
    HANDLE fh;
    HANDLE view;

    struct stat st;
    if (stat(file, &st)) {
        perror("can not stat file size");
        return 1;
    } else {
        info->size = st.st_size; 
    }

    fh = CreateFile(file, access_mode, FILE_SHARE_READ, NULL,
            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL|FILE_FLAG_RANDOM_ACCESS,NULL);
    if (fh == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "fail to open file %s for mapping\n", file);
        return 1;
    }
    info->addr = 0;

    view = CreateFileMapping(fh, NULL ,
            map_mode, 0 /* len >> 32 */ ,
            0 /* len & 0xFFFFFFFF */ ,     /* low-order DWORD of size */
            0);

    if (view) {
        info->addr = MapViewOfFile(view, view_mode, 0, 0, info->size);
    }

    /* check if mapping succeded and is usable */
    if (info->addr == 0) {
        CloseHandle(fh);
        if (view)
            CloseHandle(view);
        fprintf(stderr, "fail to mmap file %s\n", file);
        return 2;
    }

    info->win32_fh = fh;
    info->win32_view = view;

    return 0;
}

int mmap_close(mmap_info* info) {
    if (!(info->addr))
        return 1;

    if (!UnmapViewOfFile(info->addr)) {
        perror("fail to unmap file view");
        return 2;
    }

    if (!CloseHandle(info->win32_view)) {
        perror("fail to close file view");
        return 3;
    }

    if (!CloseHandle(info->win32_fh)) {
        perror("fail to close file handle");
        return 4;
    }

    return 0;
}

#endif /* USE_WIN32_MMAP }}}*/
