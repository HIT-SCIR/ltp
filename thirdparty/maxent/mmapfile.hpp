/*
 * vi:ts=4:tw=78:shiftwidth=4:expandtab
 * vim600:fdm=marker
 *
 * mmapfile.hpp  -  C++ wrapper for a mmap wrapper in C. See file `mmapfile.h'
 * for detail.
 *
 * Copyright (C) 2004 by Zhang Le <ejoy@users.sourceforge.net>
 * Begin       : 29-May-2004
 * Last Change : 30-May-2004.
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

#ifndef MMAPFILE_HPP
#define MMAPFILE_HPP

#include "mmapfile.h"

#if defined(HAVE_SYSTEM_MMAP)

#include <string.h>

class MmapFile {
    public:
        MmapFile(const char* file, const char* mode = "r", int flags = 0) {
            opened_      = false;
            file_        = strdup(file);
            mode_        = strdup(mode);
            info_.flags = flags;
        }

//XXX: free strdup?
        ~MmapFile() {
            if (opened_) close();
        }

        void* addr() { return info_.addr; }

        unsigned long size() const { return info_.size; }

        bool open() {
            int rc = mmap_open(&info_, file_, mode_, info_.flags);
            if (rc == 0) {
                opened_ = true;
                return true;
            } else
                return false;
        }

        bool close() { 
            int rc =  mmap_close(&info_);
            if (rc == 0) {
                opened_ = false;
                return true;
            } else
                return false;
        }

    private:

        char      *file_;
        char      *mode_;
        mmap_info  info_;
        bool       opened_;

};

#endif

#endif /* ifndef MMAPFILE_H */

