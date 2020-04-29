#include "util.h"

size_t D2D::UTF8toUTF16(const char* FG_RESTRICT input, ptrdiff_t srclen, wchar_t* FG_RESTRICT output, size_t buflen)
{
  return (size_t)MultiByteToWideChar(CP_UTF8, 0, input, (int)srclen, output, (int)(!output ? 0 : buflen));
}
size_t D2D::UTF16toUTF8(const wchar_t* FG_RESTRICT input, ptrdiff_t srclen, char* FG_RESTRICT output, size_t buflen)
{
  return (size_t)WideCharToMultiByte(CP_UTF8, 0, input, (int)srclen, output, (int)(!output ? 0 : buflen), NULL, NULL);
}