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

#ifndef MARKS_H

#define SLASTW "Swn" //Last Word
#define WMARK  "w"   //Words
#define PMARK  "p"   //POS
#define KMARK  "k"   //Ambiguity Classes
#define MMARK  "m"   //Maybe 
#define MFTMARK "f"   //Most Frequent Tag --> f(-1) --> f-1:NN
#define PREFIX_MARK  "a"   //prefixes
#define SUFFIX_MARK  "z"   //Suffixes
#define CHAR_A_MARK "ca"  //Character, counting from the beggining of the begining of the token (starting at 1)
#define CHAR_Z_MARK "cz"  //Character, counting from the end of the begining of the token (starting at 1)
#define LENGTH_MARK "L"  //token length
#define START_CAPITAL_MARK "SA" //start with upper case
#define START_LOWER_MARK   "sa" //start with lower case
#define START_NUMBER_MARK  "SN" //start with number
#define ALL_UPPER_MARK "AA" //all upper case
#define ALL_LOWER_MARK "aa" //all lower case
#define CONTAIN_CAP_MARK "CA" //contains a capital letter
#define CONTAIN_CAPS_MARK "CAA" //contains several capital letters
#define CONTAIN_PERIOD_MARK "CP" //contains period
#define CONTAIN_COMMA_MARK "CC" //contains comma
#define CONTAIN_NUMBER_MARK "CN" //contains number
#define MULTIWORD_MARK "MW" //contains underscores (multiword)
#define PRE_BUSHOU "bsa"   //部首前缀化特征
#define SUF_BUSHOU "bsz"   //部首后缀化特征
#define DOU "DOU"          //重叠特征

#define MARKS_H
#endif
