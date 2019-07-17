// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#include "feather/backend.h"
#include <filesystem>
#include <stdarg.h>

namespace fs = std::filesystem;

#ifdef FG_PLATFORM_WIN32
#pragma pack(push)
#pragma pack(8)
#define WINVER 0x0501 //_WIN32_WINNT_WINXP   
#define _WIN32_WINNT 0x0501
#define NTDDI_VERSION 0x05010300 //NTDDI_WINXPSP3 
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX // Some compilers enable this by default
#define NOMINMAX
#endif
#define NODRAWTEXT
#define NOBITMAP
#define NOMCX
#define NOSERVICE
#define NOHELP
#include <windows.h>
#pragma pack(pop)

void FreeDLL(void* dll) { FreeLibrary((HMODULE)dll); }
#else
#include <dlfcn.h>
#include <iconv.h>

void FreeDLL(void* dll) { dlclose(dll); }
#endif

extern "C" fgInitBackend fgLoadBackend(const char* path, void** library, const char* name)
{
  fs::path p = fs::u8path(path);
#ifdef FG_PLATFORM_WIN32
  HMODULE l = LoadLibraryW(p.wstring().c_str());
#else
  void* l = dlopen(p.u8string().c_str(), RTLD_NOW);
#endif

  if(!l)
    return nullptr;

  std::string derived;
  if(!name)
  {
    derived = p.stem().u8string();
    name = derived.c_str();
  }

#ifdef FG_PLATFORM_WIN32
  FARPROC f = GetProcAddress(l, name);
#else
  void* f = dlsym(l, name);
#endif
  if(!f)
  {
    FreeDLL(l);
    return nullptr;
  }

  if(library)
    * library = l;

  return reinterpret_cast<fgInitBackend>(f);
}

extern "C" void fgFreeBackend(void* library)
{
  FreeDLL(library);
}

#ifdef FG_PLATFORM_WIN32
extern "C" size_t fgUTF8toUTF16(const char* FG_RESTRICT input, ptrdiff_t srclen, wchar_t* FG_RESTRICT output, size_t buflen)
{
  return (size_t)MultiByteToWideChar(CP_UTF8, 0, input, (int)srclen, output, (int)(!output ? 0 : buflen));
}

extern "C" size_t fgUTF16toUTF8(const wchar_t* FG_RESTRICT input, ptrdiff_t srclen, char* FG_RESTRICT output, size_t buflen)
{
  return (size_t)WideCharToMultiByte(CP_UTF8, 0, input, (int)srclen, output, (int)(!output ? 0 : buflen), NULL, NULL);
}
#else
size_t fgUTF8toUTF16(const char* FG_RESTRICT input, ptrdiff_t srclen, wchar_t* FG_RESTRICT output, size_t buflen)
{
  static iconv_t iconv_utf8to16 = 0;
  size_t len = srclen < 0 ? strlen(input) : srclen;
  char* out = (char*)output;
  if(!output) return (len * 4) + 1;
  len += 1; // include null terminator
  if(!iconv_utf8to16) iconv_utf8to16 = iconv_open("UTF-8", "UTF-16");
  char* in = (char*)input; // Linux is stupid
  return iconv(iconv_utf8to16, &in, &len, &out, &buflen);
}

extern size_t fgUTF16toUTF8(const wchar_t* FG_RESTRICT input, ptrdiff_t srclen, char* FG_RESTRICT output, size_t buflen)
{
  static iconv_t iconv_utf16to8 = 0;
  size_t len = (srclen < 0 ? wcslen(input) : srclen) * 2;
  char* in = (char*)input;
  if(!output) return (len * 2) + 1;
  len += 2; // include null terminator (which is 2 bytes wide here)
  if(!iconv_utf16to8) iconv_utf16to8 = iconv_open("UTF-16", "UTF-8");
  return iconv(iconv_utf16to8, &in, &len, (char**)& output, &buflen);
}
#endif
