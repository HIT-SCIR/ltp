#ifndef _FINITE_H
#define _FINITE_H

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <math.h>

#if defined(HAVE_IEEEFP_H)
    #include <ieeefp.h> /* for sun os */
#endif

#if defined(_MSC_VER) || defined(__BORLANDC__)
inline int finite(double x) { return _finite(x); }
#endif

#endif /* _FINITE_H */
