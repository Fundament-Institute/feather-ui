/* graphics_desktop_bridge.h - Standard C bridge between feather graphics and desktop backends
Copyright (c)2022 Fundament Software

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef GRAPHICS_DESKTOP_BRIDGE_H
#define GRAPHICS_DESKTOP_BRIDGE_H

#include "desktop_interface.h"
#include "graphics_interface.h"

struct FG_GraphicsDesktopBridge
{
  int (*emplaceContext)(struct FG_GraphicsDesktopBridge* self, FG_Window* window, enum FG_PixelFormat backbuffer);
  int (*attachContext)(struct FG_GraphicsDesktopBridge* self, FG_Context* context, FG_Window* window);
  int (*beginDraw)(struct FG_GraphicsDesktopBridge* self, FG_Window* window, FG_Rect* area);
  int (*endDraw)(struct FG_GraphicsDesktopBridge* self, FG_Window* window);
  int (*destroy)(struct FG_GraphicsDesktopBridge* self);
};

typedef struct FG_GraphicsDesktopBridge* (*FG_InitGraphicsDesktopBridge)(struct FG_GraphicsInterface*, void*, FG_Log);

#endif