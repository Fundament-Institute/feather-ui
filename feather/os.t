local ffi = require("ffi")

local C
local S = {}

if ffi.os == "Windows" then
  C = terralib.includecstring [[
#include <intrin.h>
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
// this is due to a conflict in clang, so we force windows to discard it's version
#define _m_prefetchw _m_prefetchw_UNUSED
#include <windows.h>
#pragma pack(pop)

void FreeDLL(void* dll) { FreeLibrary((HMODULE)dll); }
char* LoadDLLError()
{
  char* p;
  FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr,
                  ::GetLastError(), 0, (LPTSTR)&p, 0, nullptr);
  return p;
}

size_t fgUTF8toUTF16(const char* restrict input, ptrdiff_t srclen, wchar_t* restrict output, size_t buflen) {
  return (size_t)MultiByteToWideChar(CP_UTF8, 0, input, (int)srclen, output, (int)(!output ? 0 : buflen));
}
  ]]
  
  terra S.LoadLibrary(path : rawstring) : &opaque
    var len = C.fgUTF8toUTF16(path, -1, nil, 0);
    var wpath = [&uint16](C.malloc(sizeof(uint16) * len))
    C.fgUTF8toUTF16(path, -1, wpath, len);

    var l : &opaque = C.LoadLibraryW(wpath)
    C.free(wpath)
    return l
  end

  terra S.LoadFunction(ptr : &opaque, name : rawstring) : &opaque
    return C.GetProcAddress([C.HMODULE](ptr), name)
  end

  terra S.LoadDLLError() : rawstring
    return C.LoadDLLError()
  end

else
  C = terralib.includecstring [[
#include <dlfcn.h>
#include <iconv.h>

  void FreeDLL(void* dll) { dlclose(dll); }

  ]]

  terra S.LoadLibrary(path : rawstring) : &opaque
    return C.dlopen(path, C.RTLD_NOW)
  end

  terra S.LoadFunction(ptr : &opaque, name : rawstring) : &opaque
    return C.dlsym(ptr, name)
  end
end

terra S.FreeLibrary(ptr : &opaque) : {}
  C.FreeDLL(ptr)
end

terra S.LoadDLLError() : rawstring
  return C.dlerror()
end

return S
