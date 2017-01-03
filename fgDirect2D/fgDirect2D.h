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
#include "fgWindowD2D.h"

struct ID2D1Factory;
struct IWICImagingFactory;
struct IDWriteFactory;
struct IWICFormatConverter;

struct fgDirect2D
{
  fgRoot root;
  ID2D1Factory* factory;
  IWICImagingFactory* wicfactory;
  IDWriteFactory* writefactory;

  static fgDirect2D* instance;
};

fgWindowD2D* GetElementWindow(fgElement* cur);

#endif
