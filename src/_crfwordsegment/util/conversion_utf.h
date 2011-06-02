

/// 实现的是从从一个字符到一个字符的转换
/// 你再加上多个字符的处理就可以了


/*-------------------------------------------------------------------------------------
wchar_t (UNICODE 2bit)->char (UTF-8)(multi bit )
它通过简单的码位析取与分配即可完成.

本函数提供这一实现.
dest_str:
宽字节字符转换为UTF-8编码字符的目标地址.
src_wchar:被转换的宽字节源字符.
返回值:
返回实际转换后的字符的字节数. 若遇到错误或检测到非法字节序列, 则返回-1.

注意! 传递进来的宽字符应是能被合法转换为UTF-8编码的字符.
--------------------------------------------------------------------------------------
*/

size_t g_f_wctou8(char * dest_str, const wchar_t src_wchar)
{
	int count_bytes = 0;
	wchar_t byte_one = 0, byte_other = 0x3f; // 用于位与运算以提取位值 0x3f--->00111111
	unsigned char utf_one = 0, utf_other = 0x80; // 用于"位或"置标UTF-8编码 0x80--->1000000
	wchar_t tmp_wchar = L'0'; // 用于宽字符位置析取和位移(右移6位)
	unsigned char tmp_char = '0';

	if (!src_wchar)//
		return (size_t)-1;

	for (;;) // 检测字节序列长度
	{
		if (src_wchar <= 0x7f){ // <=01111111
			count_bytes = 1; // ASCII字符: 0xxxxxxx( ~ 01111111)
			byte_one = 0x7f; // 用于位与运算, 提取有效位值, 下同
			utf_one = 0x0;
			break;
		}
		if ( (src_wchar > 0x7f) && (src_wchar <= 0x7ff) ){ // <=0111,11111111
			count_bytes = 2; // 110xxxxx 10xxxxxx[1](最多11个1位, 简写为11*1)
			byte_one = 0x1f; // 00011111, 下类推(1位的数量递减)
			utf_one = 0xc0; // 11000000
			break;
		}
		if ( (src_wchar > 0x7ff) && (src_wchar <= 0xffff) ){ //0111,11111111<=11111111,11111111
			count_bytes = 3; // 1110xxxx 10xxxxxx[2](MaxBits: 16*1)
			byte_one = 0xf; // 00001111
			utf_one = 0xe0; // 11100000
			break;
		}
		if ( (src_wchar > 0xffff) && (src_wchar <= 0x1fffff) ){ //对UCS-4的支持..
			count_bytes = 4; // 11110xxx 10xxxxxx[3](MaxBits: 21*1)
			byte_one = 0x7; // 00000111
			utf_one = 0xf0; // 11110000
			break;
		}
		if ( (src_wchar > 0x1fffff) && (src_wchar <= 0x3ffffff) ){
			count_bytes = 5; // 111110xx 10xxxxxx[4](MaxBits: 26*1)
			byte_one = 0x3; // 00000011
			utf_one = 0xf8; // 11111000
			break;
		}
		if ( (src_wchar > 0x3ffffff) && (src_wchar <= 0x7fffffff) ){
			count_bytes = 6; // 1111110x 10xxxxxx[5](MaxBits: 31*1)
			byte_one = 0x1; // 00000001
			utf_one = 0xfc; // 11111100
			break;
		}
		return (size_t)-1; // 以上皆不满足则为非法序列
	}
	// 以下几行析取宽字节中的相应位, 并分组为UTF-8编码的各个字节
	tmp_wchar = src_wchar;
	for (int i = count_bytes; i > 1; i--)
	{ // 一个宽字符的多字节降序赋值
		tmp_char = (unsigned char)(tmp_wchar & byte_other);///后6位与byte_other 00111111
		dest_str[i - 1] = (tmp_char | utf_other);/// 在前面加10----跟10000000或
		tmp_wchar >>= 6;//右移6位
	}
	//这个时候i=1
	//对UTF-8第一个字节位处理，
	//第一个字节的开头"1"的数目就是整个串中字节的数目
	tmp_char = (unsigned char)(tmp_wchar & byte_one);//根据上面附值得来，有效位个数
	dest_str[0] = (tmp_char | utf_one);//根据上面附值得来 1的个数
	// 位值析取分组__End!
	return count_bytes;
}

/*-----------------------------------------------------------------------------
char *-->wchar_t
它通过简单的码位截取与合成即可完成.
本函数提供这一实现.
dest_wchar:
保存转换后的宽字节字符目标地址.
src_str:
被转换的UTF-8编码源字符的多字节序列.
返回值:
返回被转换的字符的字节数. 若遇到错误或检测到非法字节序列, 则返回-1.

注意! 传递进来的宽字符应是能被合法转换为UTF-8编码的字符.
------------------------------------------------------------------------------*/
size_t g_f_u8towc(wchar_t &dest_wchar, const unsigned char * src_str)
{
	int count_bytes = 0;
	unsigned char byte_one = 0, byte_other = 0x3f; // 用于位与运算以提取位值 0x3f-->00111111
	wchar_t tmp_wchar = L'0';

	if (!src_str)
		return (size_t)-1;

	for (;;) // 检测字节序列长度,根据第一个字节头的1个个数
	{
		if (src_str[0] <= 0x7f){
			count_bytes = 1; // ASCII字符: 0xxxxxxx( ~ 01111111)
			byte_one = 0x7f; // 用于位与运算, 提取有效位值, 下同 01111111
			break;
		}
		if ( (src_str[0] >= 0xc0) && (src_str[0] <= 0xdf) ){
			count_bytes = 2; // 110xxxxx(110 00000 ~ 110 111111)
			byte_one = 0x1f; //00011111 第一字节有效位的个数
			break;
		}
		if ( (src_str[0] >= 0xe0) && (src_str[0] <= 0xef) ){
			count_bytes = 3; // 1110xxxx(1110 0000 ~ 1110 1111)
			byte_one = 0xf; //00001111
			break;
		}
		if ( (src_str[0] >= 0xf0) && (src_str[0] <= 0xf7) ){
			count_bytes = 4; // 11110xxx(11110 000 ~ 11110 111)
			byte_one = 0x7;
			break;
		}
		if ( (src_str[0] >= 0xf8) && (src_str[0] <= 0xfb) ){
			count_bytes = 5; // 111110xx(111110 00 ~ 111110 11)
			byte_one = 0x3;
			break;
		}
		if ( (src_str[0] >= 0xfc) && (src_str[0] <= 0xfd) ){
			count_bytes = 6; // 1111110x(1111110 0 ~ 1111110 1)
			byte_one = 0x1;
			break;
		}
		return (size_t)-1; // 以上皆不满足则为非法序列
	}
	// 以下几行析取UTF-8编码字符各个字节的有效位值
	//先得到第一个字节的有效位数据
	tmp_wchar = src_str[0] & byte_one;
	for (int i=1; i<count_bytes; i++)
	{
		tmp_wchar <<= 6; // 左移6位后与后续字节的有效位值"位或"赋值
		tmp_wchar = tmp_wchar | (src_str[i] & byte_other);//先与后或
	}
	// 位值析取__End!
	dest_wchar = tmp_wchar;
	return count_bytes;
}
