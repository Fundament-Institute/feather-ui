// Copyright ©2012 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "bss-util/bss_defines.h"

#ifdef BSS_PLATFORM_WIN32
#include "fgWinAPI.h"
#include "win32_includes.h"

extern WinAPIfgRoot* _fgroot;

fgStatic* fgLoadImage(const char* path)
{
  return 0;
}
fgStatic* fgLoadImageData(const void* data, size_t length)
{
  return 0;
}

void WinAPIfgImage_Destroy(WinAPIfgImage* self)
{
  switch(self->st.render.element.flags&0x18) // We store what type it is here
  {
  case 8:
    DestroyCursor(self->imghandle);
    break;
  case 16:
    DestroyIcon(self->imghandle);
    break;
  default:
    DeleteObject(self->imghandle);
    break;
  }
  fgStatic_Destroy((fgStatic*)self);
}

#endif