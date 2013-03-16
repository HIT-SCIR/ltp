/*
 * vi:ts=4:tw=78:shiftwidth=4:expandtab
 * vim600:fdm=marker
 *
 * lbfgs.h  -  wrapper code for the Fortran L-BFGS optimization routine
 *
 * You can get the Fortran lbfgs routine from:
 * http://www.netlib.org/opt/lbfgs_um.shar
 *
 * For detail usage, please consult the comment in the beginning of lbfgs.f.
 *
 * sdriver.c provides a simple use of the C interface.
 *
 * Copyright (C) 2004 by Zhang Le <ejoy@users.sourceforge.net>
 * Begin       : 17-Nov-2004
 * Last Change : 26-Apr-2005.
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

#ifndef _LBFGS_H
#define _LBFGS_H

#if defined(__cplusplus)
extern "C" {
#endif

typedef struct {
    int     n;
    int     m;
    int     niter;       /* number of iterations so far                        */
    int     nfuns;       /* number of function evaluations so far              */
    int     iflag;
    int     diagco;
    int     iprint[2];   /* see the comment in lbfgs.f for usage of this field */
    double  eps;
    double  xtol;
    double *diag;
    double *w;
} lbfgs_t;

lbfgs_t* lbfgs_create(int n, int m, double eps);
int lbfgs_run(lbfgs_t* obj, double* x, double* f, double* g);
void lbfgs_destory(lbfgs_t* obj);

#if defined(__cplusplus)
}
#endif

#endif /* ifndef _LBFGS_H */

