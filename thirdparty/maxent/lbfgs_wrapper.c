/*
 * vi:ts=4:tw=78:shiftwidth=4:expandtab
 * vim600:fdm=marker
 *
 * lbfgs_wrapper.c  - wrapper code for Fortran L-BFGS optimization routine
 *
 * You can get the Fortran lbfgs routine from:
 * http:// www.netlib.org/opt/lbfgs_um.shar
 *
 * For detail usage, please consult the comment in the beginning of lbfgs.f.
 *
 * Copyright (C) 2004 by Zhang Le <ejoy@users.sourceforge.net>
 * Begin       : 17-Nov-2004
 * Last Change : 17-Nov-2004.
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

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdlib.h>
#include <float.h>
#include "lbfgs.h"

#if defined(WIN32) && !defined(__GNUC__) /* assume use Intel Fortran Compiler on win32 platform if not compiled with GCC */
#    define LBFGS_FUN LBFGS
#else
#    define LBFGS_FUN lbfgs_ /* assume use GNU f77 otherwise */
#endif
    /* Fortran lbfgs interface */
#if defined(__cplusplus)
extern "C" {
#endif
    extern void LBFGS_FUN(int* n, int* m, double* x, double* f, double* g,
            int* diagco,double* diag, int* iprint, double* eps,
            double* xtol, double* w, int* iflag, int* niter, int* nfuns);
#if defined(__cplusplus)
}
#endif

/* create an opt object
 * n is the dimension of the variable
 * m is the number of the corrections used in BFGS update
 * eps is used to determines the terninating accuracy
 */
lbfgs_t* lbfgs_create(int n, int m, double eps) {
    lbfgs_t* opt = (lbfgs_t*)malloc(sizeof(lbfgs_t));

    if (!opt)
        return 0;
    opt->w = (double*)malloc(sizeof(double) * (n*( 2*m+1) + 2*m));
    if (!opt->w) {
        free(opt);
        return 0;
    }
    opt->diag = (double*)malloc(sizeof(double) * n);
    if (!opt->diag) {
        free(opt->w);
        free(opt);
        return 0;
    }

    opt->n         = n;
    opt->m         = m;
    opt->eps       = eps;
    opt->xtol      = DBL_EPSILON;
    opt->diagco    = 0;           /* by default we do not provide diagonal matrices */
    opt->iflag     = 0;
    opt->iprint[0] = -1;          /* by default print nothing                       */
    opt->iprint[1] = 0;
    opt->niter     = 0;
    opt->nfuns     = 0;

    return opt;
}

/* free all the memory used by the optimizer */
void lbfgs_destory(lbfgs_t* opt) {
    free(opt->diag);
    free(opt->w);
    free(opt);
}


/* x is the n-dimension variable
 * f is the current value of objective function
 * g is the n-dimension gradient of f
 *
 * return value:
 *       = 0: success
 *       < 0: some error occur, see comment in lbfgs.f for detail
 *       = 1: user must evaluate F and G
 *       = 2: user must provide the diagonal matrix Hk0.
 */
int lbfgs_run(lbfgs_t* opt, double* x, double* f, double* g) {
    LBFGS_FUN(&opt->n, &opt->m, x, f, g, &opt->diagco, opt->diag,
            opt->iprint, &opt->eps, &opt->xtol, opt->w, &opt->iflag,
            &opt->niter, &opt->nfuns);
    return opt->iflag;
}

