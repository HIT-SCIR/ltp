#ifndef __LAS_BASE__
#define __LAS_BASE__

#define LAS_NS_BEG namespace las {
#define LAS_NS_END }; // end of namespace las

LAS_NS_BEG

const int MaxLookupAtomNum = 4; //the maximum atom number to look up
///下面的常量定义与graph相关
const int MaxGraphMatrixSize = 1024; ///最大矩阵大小为100×100，当矩阵大小不足是将会发生重新分配
const int MaxWordArrayLength = 2048;      ///声明预分配的词的最大长度，超过该长度的将会导致内存重新分配
const int MaxWordPathMatrixSize = 2048;
const int MaxWordPathLinkLength = 64;
const int MaxLinkLength = 64;

LAS_NS_END

#endif
