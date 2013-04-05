/*
 * vi:ts=4:tw=78:shiftwidth=4:expandtab
 * vim600:fdm=marker
 *
 * fc_dummy_main.cpp  -  dummy main definition for linking with fortran lib
 *
 * Copyright (C) 2004 by Zhang Le <ejoy@users.sourceforge.net>
 * Begin       : 24-Apr-2004
 * Last Change : 04-Oct-2006.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser GPL (LGPL) as published by
 * the Free Software Foundation; either version 2.1 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program.
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef FC_DUMMY_MAIN
#  ifdef __cplusplus
extern "C"
#  endif
int FC_DUMMY_MAIN() { return 1; }
#endif

