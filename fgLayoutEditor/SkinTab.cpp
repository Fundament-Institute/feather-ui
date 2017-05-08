// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "SkinTab.h"
#include "bss-util/bss_util.h"

using namespace bss;

SkinTab::SkinTab()
{
  bssFill(*this, 0);
}
SkinTab::~SkinTab()
{
  Clear();
}
void SkinTab::Init(EditorBase* base)
{
  _base = base;
}
void SkinTab::Clear()
{

}

