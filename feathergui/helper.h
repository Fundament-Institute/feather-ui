// Copyright ©2015 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_HELPER_H__
#define __FG_HELPER_H__

#define fgVector_AddObject(self, type, init, ...) \
  fgVector_CheckSize((self), sizeof(type)); \
  FG_UINT r = (self)->l++; \
  init(((type*)(self)->p) + r, __VA_ARGS__); \
  return r;

#endif
