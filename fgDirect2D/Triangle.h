// Copyright (c)2020 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#ifndef D2D__TRIANGLE_H
#define D2D__TRIANGLE_H

#include "EffectBase.h"

namespace D2D {
  DEFINE_GUID(CLSID_Triangle, 0xEAB57D83, 0x9556, 0x4CD2, 0xA0, 0x66, 0xE4, 0xF1, 0x60, 0x3E, 0x90, 0xC7);
  DEFINE_GUID(CLSID_TrianglePixelShader, 0x692EB775, 0x3EC4, 0x4B55, 0x93, 0xB6, 0x44, 0x78, 0x07, 0x6D, 0xE0, 0xD3);

  class Triangle : public EffectBase
  {
  public:
    IFACEMETHODIMP Initialize(_In_ ID2D1EffectContext* pContextInternal, _In_ ID2D1TransformGraph* pTransformGraph);
    static HRESULT Register(_In_ ID2D1Factory1* pFactory);
    static HRESULT __stdcall CreateEffect(_Outptr_ IUnknown** ppEffectImpl);

    IFACEMETHODIMP SetDrawInfo(_In_ ID2D1DrawInfo* pRenderInfo);

  private:
    Triangle();
    ~Triangle();
  };
}

#endif
