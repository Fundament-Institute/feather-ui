// Copyright (c)2020 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#ifndef D2D__WIN32_INCLUDES_H
#define D2D__WIN32_INCLUDES_H

#pragma pack(push)
#pragma pack(8)
#define WINVER        0x0601 //_WIN32_WINNT_WIN7
#define _WIN32_WINNT  0x0601
#define NTDDI_VERSION 0x06010000 // NTDDI_WIN7
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX // Some compilers enable this by default
  #define NOMINMAX
#endif
#define NOBITMAP
#define NOMCX
#define NOSERVICE
#define NOHELP
#include <windows.h>
#pragma pack(pop)

#endif
