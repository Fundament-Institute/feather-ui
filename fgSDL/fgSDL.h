// Copyright ©2016 Black Sphere Studios
// fgSDL - An SDL impmlementation of featherGUI
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_SDL_H__
#define __FG_SDL_H__

#include "fgRoot.h"
#include "SDL\SDL.h"

#ifdef  __cplusplus
extern "C" {
#endif

FG_EXTERN void (FG_FASTCALL *debugmsghook)(fgWindow* self, const FG_Msg* msg);

typedef struct __FG_ROOT_SDL
{
  fgRoot root;
  SDL_Window* window;
  SDL_Renderer* render;
  SDL_Rect* clipstack;
  unsigned int clipcapacity;
  unsigned int cliplength;
} fgRootSDL;

#ifdef  __cplusplus
}
#endif

#endif
