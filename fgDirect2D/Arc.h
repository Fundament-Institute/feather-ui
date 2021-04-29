// Copyright (c)2021 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#ifndef D2D__ARC_H
#define D2D__ARC_H

#include "EffectBase.h"

namespace D2D {
  DEFINE_GUID(CLSID_Arc, 0x5E6C30D3, 0xFB29, 0x49B3, 0x95, 0x39, 0x54, 0x29, 0xD5, 0xF7, 0x75, 0xED);
  DEFINE_GUID(CLSID_ArcPixelShader, 0x92D9905B, 0xBC88, 0x4658, 0xAB, 0x4A, 0xDE, 0x11, 0xA8, 0x61, 0x33, 0xE5);

  class Arc : public EffectBase
  {
  public:
    IFACEMETHODIMP Initialize(_In_ ID2D1EffectContext* pContextInternal, _In_ ID2D1TransformGraph* pTransformGraph);
    static HRESULT Register(_In_ ID2D1Factory1* pFactory);
    static HRESULT __stdcall CreateEffect(_Outptr_ IUnknown** ppEffectImpl);
    HRESULT SetArcs(D2D_VECTOR_2F arcs);
    D2D_VECTOR_2F GetArcs() const;
    HRESULT SetInnerRadius(FLOAT r);
    FLOAT GetInnerRadius() const;

    IFACEMETHODIMP SetDrawInfo(_In_ ID2D1DrawInfo* pRenderInfo);

  private:
    Arc();
    ~Arc();
  };
}

#endif
