/* fgDirect2D - Direct2D Backend for Feather GUI
Copyright ©2018 Black Sphere Studios

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

#ifndef FG__DIRECT2D_H
#define FG__DIRECT2D_H

#include "feather/root.h"
#include "Context.h"
#include <vector>

struct ID2D1Factory1;
struct IWICImagingFactory;
struct IDWriteFactory1;
struct IWICFormatConverter;
struct _DWM_BLURBEHIND;
struct HMONITOR__;

typedef long(__stdcall* DWMCOMPENABLE)(int*);
typedef long(__stdcall* DWMBLURBEHIND)(struct HWND__*, const struct _DWM_BLURBEHIND*);

namespace D2D {
  struct Direct2D : fgBackend
  {
    ~Direct2D();
    void RefreshMonitors();
    void WipeMonitors();
    fgOutlineNode* GetDisplay(struct HMONITOR__* hWnd);
    fgAsset* LoadAsset(const char* data, size_t count);

    ID2D1Factory1* factory;
    IWICImagingFactory* wicfactory;
    IDWriteFactory1* writefactory;
    HWND__* tophwnd;
    DWMBLURBEHIND dwmblurbehind;
    struct FG__ROOT* root;
    long(__stdcall* getDpiForMonitor)(struct HMONITOR__*, int, unsigned int*, unsigned int*);
    long(__stdcall* getScaleFactorForMonitor)(struct HMONITOR__*, int*);
    
    static const unsigned char VALID_DISPLAY = 0x40;
  };
}

#endif
