/*
 * Copyright (C) 2004 Jesus Gimenez, Lluis Marquez and Senen Moya
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#ifndef ER_H
#define ER_H

#include <sys/types.h>
//#include "regex.h"
#include <boost/cregex.hpp>
using namespace boost;

/*****************************************************************
 * Regular expression
 *****************************************************************/

#define CARD		100
#define CARDPUNCT	101
#define CARDSEPS	102
#define CARDSUFFIX	103

extern regex_t  erCard,erCardPunct,erCardSeps,erCardSuffix;
extern regex_t  erMultiWord,erContainNum,erStartCap,erStartLower,erStartNumber,
erAllUp,erAllLow,erContainCap,erContainCaps,erContainPeriod,erContainComma;


void erCompRegExp();
void erFreeRegExp();
int erLookRegExp2(void *er,char * str);
int erLookRegExp(char *m);


#endif
