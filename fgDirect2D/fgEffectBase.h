// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#ifndef __FG_EFFECTBASE_D2D_H__
#define __FG_EFFECTBASE_D2D_H__

#include <d2d1effectauthor.h>

class fgEffectBase : public ID2D1EffectImpl, public ID2D1DrawTransform
{
public:
  IFACEMETHODIMP PrepareForRender(D2D1_CHANGE_TYPE changeType);
  IFACEMETHODIMP SetGraph(_In_ ID2D1TransformGraph* pGraph);
  IFACEMETHODIMP_(ULONG) AddRef();
  IFACEMETHODIMP_(ULONG) Release();
  IFACEMETHODIMP QueryInterface(_In_ REFIID riid, _Outptr_ void** ppOutput);

  HRESULT SetRect(D2D_VECTOR_4F rect);
  D2D_VECTOR_4F GetRect() const;

  HRESULT SetCorners(D2D_VECTOR_4F corners);
  D2D_VECTOR_4F GetCorners() const;

  HRESULT SetColor(UINT color);
  UINT GetColor() const;

  HRESULT SetOutlineColor(UINT color);
  UINT GetOutlineColor() const;

  HRESULT SetOutline(FLOAT outline);
  FLOAT GetOutline() const;

  IFACEMETHODIMP MapOutputRectToInputRects(_In_ const D2D1_RECT_L* pOutputRect, _Out_writes_(inputRectCount) D2D1_RECT_L* pInputRects, UINT32 inputRectCount) const;
  IFACEMETHODIMP MapInputRectsToOutputRect(_In_reads_(inputRectCount) CONST D2D1_RECT_L* pInputRects, _In_reads_(inputRectCount) CONST D2D1_RECT_L* pInputOpaqueSubRects, UINT32 inputRectCount, _Out_ D2D1_RECT_L* pOutputRect, _Out_ D2D1_RECT_L* pOutputOpaqueSubRect);
  IFACEMETHODIMP MapInvalidRect(UINT32 inputIndex, D2D1_RECT_L invalidInputRect, _Out_ D2D1_RECT_L* pInvalidOutputRect) const;
  IFACEMETHODIMP_(UINT32) GetInputCount() const;

protected:
  fgEffectBase();
  ~fgEffectBase();
  HRESULT UpdateConstants();

  ID2D1DrawInfo* _drawInfo;
  D2D1_RECT_L _rect;
  LONG _ref;
  UINT _color;
  UINT _outlinecolor;

  struct
  {
    D2D_VECTOR_4F rect;
    D2D_VECTOR_4F corners;
    D2D1_COLOR_F color;
    D2D1_COLOR_F outlinecolor;
    float outline;
  } _constants;
};

#endif#pragma once
