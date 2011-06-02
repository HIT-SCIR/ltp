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

//#include <stdio.h>
#include <boost/cregex.hpp>
using namespace boost;
#include "er.h"


/*****************************************************************
* Regular expressions
*****************************************************************/

regex_t  erCard,erCardPunct,erCardSeps,erCardSuffix;
regex_t  erMultiWord,erContainNum,erStartCap,erStartLower,erStartNumber,
         erAllUp,erAllLow,erContainCap,erContainCaps,erContainPeriod,erContainComma;

/*#define ER_STARTCAP	"^[A-Z«—¡…Õ”⁄¿»Ã“ŸƒÀœ÷‹].*$"
#define ER_STARTLOWER	"^[a-zÁÒ·ÈÌÛ˙‡ËÏÚ˘‰ÎÔˆ¸].*$"
#define ER_STARTNUMBER	"^[0-9].*$"
#define ER_ALLUP	"^[A-Z«—¡…Õ”⁄¿»Ã“ŸƒÀœ÷‹]+$"
#define ER_ALLLOW	"^[a-zÁÒ·ÈÌÛ˙‡ËÏÚ˘‰ÎÔˆ¸]+$"
#define ER_CONTAINCAP	"^.+[A-Z«—¡…Õ”⁄¿»Ã“ŸƒÀœ÷‹].*$"
#define ER_CONTAINCAPS	"^.*[A-Z«—¡…Õ”⁄¿»Ã“ŸƒÀœ÷‹].*[A-Z«—¡…Õ”⁄¿»Ã“ŸƒÀœ÷‹].*$"*/
#define ER_STARTCAP	"^[A-Z].*$"
#define ER_STARTLOWER	"^[a-z].*$"
#define ER_STARTNUMBER	"^[0-9].*$"
#define ER_ALLUP	"^[A-Z]+$"
#define ER_ALLLOW	"^[a-z]+$"
#define ER_CONTAINCAP	"^.+[A-Z].*$"
#define ER_CONTAINCAPS	"^.*[A-Z].*[A-Z].*$"
#define ER_CONTAINPERIOD "^.*[.].*$"
#define ER_CONTAINCOMMA "^.*[,].*$"
#define ER_CONTAINNUM	"^.*[0-9].*$"
#define ER_MULTIWORD	"^.*[-].*$"
#define ER_CARD		"^[0-9]+$"
#define ER_CARDPUNCT	"^[0-9]+[,!?:.]+$"
#define ER_CARDSEPS	"^[0-9]+[-,:\\/.][0-9,:\\/.-]+$"
#define ER_CARDSUFFIX	"^[0-9]+[^0-9]+.*$"


void erCompRegExp()
{
	regcomp (&erCard,ER_CARD,REG_EXTENDED);
	regcomp (&erCardPunct,ER_CARDPUNCT,REG_EXTENDED);
	regcomp (&erCardSeps,ER_CARDSEPS,REG_EXTENDED);
	regcomp (&erCardSuffix,ER_CARDSUFFIX,REG_EXTENDED);

	regcomp (&erStartCap,ER_STARTCAP,REG_EXTENDED);
	regcomp (&erStartNumber,ER_STARTNUMBER,REG_EXTENDED);
	regcomp (&erStartLower,ER_STARTLOWER,REG_EXTENDED);
	regcomp (&erAllUp,ER_ALLUP,REG_EXTENDED);
	regcomp (&erAllLow,ER_ALLLOW,REG_EXTENDED);
	regcomp (&erContainCap,ER_CONTAINCAP,REG_EXTENDED);
	regcomp (&erContainCaps,ER_CONTAINCAPS,REG_EXTENDED);
	regcomp (&erContainPeriod,ER_CONTAINPERIOD,REG_EXTENDED);
	regcomp (&erContainComma,ER_CONTAINCOMMA,REG_EXTENDED);
	regcomp (&erContainNum,ER_CONTAINNUM,REG_EXTENDED);
	regcomp (&erMultiWord,ER_MULTIWORD,REG_EXTENDED);
}

void erFreeRegExp()
{
	regfree(&erCard);
	regfree(&erCardSuffix);
	regfree(&erCardSeps);
	regfree(&erCardPunct);

	regfree(&erStartCap);
	regfree(&erStartLower);
	regfree(&erStartNumber);
	regfree(&erAllUp);
	regfree(&erAllLow);
	regfree(&erContainCap);
	regfree(&erContainCaps);
	regfree(&erContainComma);
	regfree(&erContainPeriod);
	regfree(&erContainNum);
	regfree(&erMultiWord);
}


/*
 * return 1 if str is like the regular expression
 * in other case return 0
 */
int erLookRegExp2(void *er,char * str)
{
	int ret=0;

	if (!regexec ((regex_t *)er,str,0,NULL,0)) return  1;

	return 0;
}



int erLookRegExp(char *m)
{
	int ret=-1;

	if (!regexec (&erCardPunct,m,0,NULL,0)) ret=CARDPUNCT;
	else if (!regexec (&erCardSeps,m,0,NULL,0)) ret=CARDSEPS;
	else if (!regexec (&erCardSuffix,m,0,NULL,0)) ret=CARDSUFFIX;
	else if (!regexec (&erCard,m,0,NULL,0))  ret=CARD;

	return ret;
}
