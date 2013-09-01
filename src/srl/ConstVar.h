/*
 * File Name     : ConstVar.h
 * Author        : Frumes
 * Create Time   : 2006Äê12ÔÂ31ÈÕ
 * Project Name  : NewSRLBaseLine
 * Remark        : define the constant variable used in the project,
 *                 the variable is classified as char, char* and int.
 */


#ifndef  _CONST_VAR_
#define  _CONST_VAR_

/*------------  const char type variable begin --------------*/
//the B-I-E-S-O tag for name entity
const char C_NE_SINGLE  = 'S';
const char C_NE_BEGIN   = 'B';
const char C_NE_END     = 'E';
const char C_NE_IN      = 'I';
const char C_NE_OUT     = 'O';

const char C_NE_SEP     = '-';   // separate tag
const char C_END_CHAR   = '\0';  // the end character of c type string
const char C_COMMENT_CHAR = '#'; // the comment character in configuration file
const char C_FEATTYPE_COMMENT = '$';
const char C_UP   = '>';
const char C_DOWN = '<';
const char C_ADD  = '+';
const char C_TAB  = '\t';
/*------------  const char type variable end --------------*/

/*--------  const char* const type variable  begin --------*/
const char* const S_ROOT_REL    = "HED";  // root relation tag in dependency tree
const char* const S_NULL_REL    = "NREL"; // relation of null node

// family members relationship of tow tree node
const char* const S_FMS_PARENT  = "FMSP";
const char* const S_FMS_CHILD   = "FMSC";
const char* const S_FMS_SIBLING = "FMSS";
const char* const S_FMS_ANCESTOR  = "FMSAC";
const char* const S_FMS_POSTERITY = "FMSPT";
const char* const S_FMS_OTHER   = "FMSO";

// position tag related to the predicate
const char* const S_PS_BEFORE  = "PSB";
const char* const S_PS_AFTER   = "PSA";
const char* const S_PS_PD      = "PSP";

// the path feature related string
const char* const S_PATH_PD     = "PD";
const char* const S_PATH_UP     =">";
const char* const S_PATH_DOWN   = "<";

// some null type tag
const char* const S_NULL_NE     = "NNE";
const char* const S_NULL_ARG    = "NULL";
const char* const S_NULL_WORD   = "NWD";
const char* const S_NULL_POS    = "NPOS";
const char* const S_NULL_STR    = "";
const char* const S_NULL_PD     = "N-P";

const char* const S_VERB_POS       = "v"; //the POS tag of verb
const char* const S_HYPHEN_TAG     = "-"; //the hyphenation tag
const char* const S_STAR           = "*";
const char* const S_LEFT_BRACKET   = "(";
const char* const S_RIGHT_BRACKET  = ")";

// the null pattern features of predicate
const char* const S_NULL_POSPAT_PDCHR  = "NPPPC";
const char* const S_NULL_RELPAT_PDCHR  = "NRPPC";
const char* const S_NULL_POSPAT_PDSIBS = "NPPPS";
const char* const S_NULL_RELPAT_PDSIBS = "NRPPS";

const char* const S_NULL_PD_CLASS  = "NPDC"; //the predicate which can not find in dict
const char* const S_PD_ARG         = "rel";  //the predicate arg label
/*------00-  const char* const type variable  end --------*/

/*------------  const int type variable begin ------------*/
const int I_NULL_ID     = -1;       //the ID of null node
const int I_NULL_RIGHT  = 10000;    //the default null right ID
const int I_NULL_RCP    = -1;       //the tow node have no recent common parent

//some const int number about name entity
const int I_NE_LENGTH  = 4;
const int I_NE_FIRSTPS = 0;
const int I_NE_SEPPS   = 1;
const int I_NE_BEGINPS = 2;
const int I_NE_SIZE    = 2;

const int I_PUN_PARENT_ID = -2; //the parent ID of punctuation character
const int I_HED_PARENT_ID = -1; // note: changed for PTBtoDep
const int I_NUMEXC = 1;         //used for gold args file //changed for PTBtoDep
const int I_RADIX  = 10;        //the radix parameter of function: atoi

const int I_FEATSEL_NUM = 64;   //the number of features in the features-select configuration file
const int I_FEATCOMB_NUM = 32;  //the number of features in the features-combine configuration file
const int I_WORD_LEN = 1024;    //the length of a word
/*------------  const int type variable end ------------*/

/*------ some const variable for srl result combine ----*/
const int I_SENT_IDX   = 0;
const int I_PD_IDX     = 1;
const int I_PS_BEG_IDX = 2;
const int I_PS_END_IDX = 3;
const int I_SENT_NUM   = 1500;

const double I_ARG_THRESHOLD_VAL = 0.5;

const char C_PATTERN_SEP = '|';

const char* const S_QTY_ARG = "QTY";
const char* const S_PSE_ARG = "PSE";
const char* const S_PSR_ARG = "PSR";
const char* const S_QTY_POS_PAT = "AD|CD|M|Q";
const char* const S_ARG0_TYPE = "ARG0";
/*------ some const variable for srl result combine ----*/


#endif

