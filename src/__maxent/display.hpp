/*
 * vi:ts=4:tw=78:shiftwidth=4:expandtab
 * vim600:fdm=marker
 *
 * display.hpp  -  a handy printf like wrapper routine with controlled output
 *
 * Copyright (C) 2003 by Zhang Le <ejoy@users.sourceforge.net>
 * Begin       : 01-Jun-2003
 * Last Change : 04-Jul-2004.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */

#ifndef DISPLAY_H
#define DISPLAY_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

namespace maxent {

/**
 * verbose flag
 *
 * If set to 1 (default) various verbose information will be printed on
 * stdout. Set this flag to 0 to restrain verbose output.
 */
extern int verbose;
void display(const char *msg, ... ); // with newline
void displayA(const char *msg, ... );// without newline

}

#endif /* ifndef DISPLAY_H */

