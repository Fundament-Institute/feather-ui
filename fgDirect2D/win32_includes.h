#ifndef __D2D_WIN32_INCLUDES_H__
#define __D2D_WIN32_INCLUDES_H__
#pragma pack(push)
#pragma pack(8)
#define WINVER 0x0601 //_WIN32_WINNT_WIN7   
#define _WIN32_WINNT 0x0601
#define NTDDI_VERSION 0x06010000 //NTDDI_WIN7 
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