/* fgDirect2D - Direct2D Backend for Feather GUI
Copyright ©2017 Black Sphere Studios

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

#ifndef __FG_DIRECT2D_H__
#define __FG_DIRECT2D_H__

#include "fgRoot.h"
#include "fgContext.h"

struct ID2D1Factory1;
struct IWICImagingFactory;
struct IDWriteFactory1;
struct IWICFormatConverter;
struct fgWindowD2D;

struct fgDirect2D
{
  fgRoot root;
  fgContext context;
  fgContext debugcontext;
  HWND__* debughwnd;
  fgWindowD2D* debugtarget;
  ID2D1Factory1* factory;
  IWICImagingFactory* wicfactory;
  IDWriteFactory1* writefactory;
  HWND__* tophwnd;

  static longptr_t __stdcall WndProc(HWND__* hWnd, unsigned int message, size_t wParam, longptr_t lParam);
  static longptr_t __stdcall DebugWndProc(HWND__* hWnd, unsigned int message, size_t wParam, longptr_t lParam);

  static fgDirect2D* instance;
};

fgWindowD2D* GetElementWindow(fgElement* cur);

#endif
