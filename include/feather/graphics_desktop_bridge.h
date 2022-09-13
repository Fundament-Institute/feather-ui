
#include "desktop_interface.h"
#include "graphics_interface.h"

struct FG_GraphicsDesktopBridge
{
  FG_Window* (*emplaceContext)(struct FG_GraphicsDesktopBridge* self, FG_Window* window, enum FG_PixelFormat backbuffer);
  FG_Window* (*attachContext)(struct FG_GraphicsDesktopBridge* self, FG_Context* context, FG_Window* window);
  int (*beginDraw)(struct FG_GraphicsDesktopBridge* self, FG_Window* window, FG_Rect* area);
  int (*endDraw)(struct FG_GraphicsDesktopBridge* self, FG_Window* window);
  int (*destroy)(struct FG_GraphicsDesktopBridge* self);
};

typedef struct FG_GraphicsDesktopBridge* (*FG_InitGraphicsDesktopBridge)(FG_GraphicsInterface*, void*, FG_Log);