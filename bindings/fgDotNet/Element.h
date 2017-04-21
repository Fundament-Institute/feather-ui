// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#pragma once
#include "fgElement.h"
#include <vcclr.h>

#define TOCHARSTR(str, pstr) pin_ptr<const unsigned char> pstr = !str->Length?(unsigned char*)"":&System::Text::Encoding::UTF8->GetBytes(str)[0]
#define TOCHAR(str) TOCHARSTR(str, pstr)

template<typename T, typename P>
inline static T^ GenNewManagedPtr(P* p) { return !p ? (T^)nullptr : gcnew T(p); }

namespace fgDotNet {
  public value struct UnifiedCoord
  {
    inline UnifiedCoord(const Coord& c) : abs(c.abs), rel(c.rel) {}
    FABS abs;
    FREL rel;

    inline operator Coord() { return Coord{ abs, rel }; }
    inline static operator Coord(UnifiedCoord^ c) { return c->operator Coord(); }
  };

  public value struct UnifiedVec
  {
    inline UnifiedVec(const CVec& v) : x(v.x), y(v.y) {}
    UnifiedCoord x;
    UnifiedCoord y;
    inline operator CVec() { return CVec{ Coord{ x.abs, x.rel },Coord{ y.abs, y.rel }}; }
    inline static operator CVec(UnifiedVec^ v) { return v->operator CVec(); }
  };

  public value struct UnifiedRect
  {
    inline UnifiedRect(const CRect& r) : left(r.left), top(r.top), right(r.right), bottom(r.bottom) {}
    UnifiedCoord left;
    UnifiedCoord top;
    UnifiedCoord right;
    UnifiedCoord bottom;
    inline operator CRect() { return CRect{ Coord{ left.abs, left.rel },Coord{ top.abs, top.rel },Coord{right.abs, right.rel },Coord{ bottom.abs, bottom.rel }, }; }
    inline static operator CRect(UnifiedRect^ r) { return r->operator CRect(); }
  };

  public value struct UnifiedTransform
  {
    inline UnifiedTransform(const fgTransform& t) : area(t.area), rotation(t.rotation), center(t.center) {}
    UnifiedRect area;
    FABS rotation;
    UnifiedVec center;
    inline operator fgTransform() { return fgTransform{ (CRect)area, rotation, (CVec)center }; }
    inline static operator fgTransform(UnifiedTransform^ r) { return r->operator fgTransform(); }
  };

  public ref struct DrawAuxData {
    System::Drawing::Point dpi;
    System::Drawing::PointF scale;
    System::Drawing::PointF scalecenter;

    inline operator fgDrawAuxData() { return fgDrawAuxData{ sizeof(fgDrawAuxData), {dpi.X, dpi.Y}, {scale.X, scale.Y}, {scalecenter.X, scalecenter.Y} }; }
    inline static operator fgDrawAuxData(DrawAuxData^ r) { return r->operator fgDrawAuxData(); }
  };

  ref class Asset;
  ref class Font;
  ref class Layout;
  ref class Skin;
  ref struct Style;
  
  public ref class Element
  {
  public:
    Element(fgElement* p);
    Element(Element^ parent, Element^ next, System::String^ name, fgFlag flags, UnifiedTransform^ transform, unsigned short units);
    ~Element();
    !Element();
    void Move(unsigned short subtype, Element^ child, size_t diff);
    size_t SetAlpha(float alpha);
    size_t SetArea(UnifiedRect^ area);
    size_t SetTransform(UnifiedTransform^ transform);
    void SetFlag(fgFlag flag, bool value);
    void SetFlags(fgFlag flags);
    size_t SetMargin(System::Drawing::RectangleF^ margin);
    size_t SetPadding(System::Drawing::RectangleF^ padding);
    void SetParent(Element^ parent);
    void SetParent(Element^ parent, Element^ next);
    size_t AddChild(Element^ child);
    size_t AddChild(Element^ child, Element^ next);
    //Element^ AddItem(void* item, size_t index = (size_t)~0);
    Element^ AddItemText(System::String^ item);
    Element^ AddItemElement(Element^ item);
    size_t RemoveChild(Element^ child);
    size_t RemoveItem(size_t item);
    void LayoutChange(unsigned short subtype, Element^ target, Element^ old);
    size_t LayoutFunction(const FG_Msg& msg, UnifiedRect^ area, bool scrollbar);
    Element^ LayoutLoad(Layout^ layout);
    size_t DragOver(int x, int y);
    size_t Drop(int x, int y, unsigned char allbtn);
    void Draw(System::Drawing::RectangleF^ area, DrawAuxData^ aux);
    size_t Inject(const FG_Msg* msg, System::Drawing::RectangleF^ area);
    size_t SetSkin(Skin^ skin);
    Skin^ GetSkin();
    Skin^ GetSkin(Element^ child);
    size_t SetStyle(System::String^ name, FG_UINT mask);
    size_t SetStyle(Style^ style);
    size_t SetStyle(FG_UINT index, FG_UINT mask);
    Style^ GetStyle();
    System::Drawing::Point^ GetDPI();
    void SetDPI(int x, int y);
    System::String^ GetClassName();
    size_t GetUserdata();
    size_t GetUserdata(System::String^ name);
    void SetUserdata(size_t data);
    void SetUserdata(size_t data, System::String^ name);
    size_t MouseDown(int x, int y, unsigned char button, unsigned char allbtn);
    size_t MouseDblClick(int x, int y, unsigned char button, unsigned char allbtn);
    size_t MouseUp(int x, int y, unsigned char button, unsigned char allbtn);
    size_t MouseOn(int x, int y);
    size_t MouseOff(int x, int y);
    size_t MouseMove(int x, int y);
    size_t MouseScroll(int x, int y, unsigned short delta, unsigned short hdelta);
    size_t KeyUp(unsigned char keycode, char sigkeys);
    size_t KeyDown(unsigned char keycode, char sigkeys);
    size_t KeyChar(int keychar, char sigkeys);
    size_t JoyButtonDown(short joybutton);
    size_t JoyButtonUp(short joybutton);
    size_t JoyAxis(float joyvalue, short joyaxis);
    size_t GotFocus();
    void LostFocus();
    size_t SetName(System::String^ name);
    System::String^ GetName();
    void SetContextMenu(Element^ menu);
    Element^ GetContextMenu();
    void Neutral();
    void Hover();
    void Active();
    void Action();
    void SetMaxDim(float x, float y);
    void SetMinDim(float x, float y);
    System::Drawing::PointF^ GetMaxDim();
    System::Drawing::PointF^ GetMinDim();
    Element^ GetItem(size_t index);
    Element^ GetItemAt(int x, int y);
    size_t GetNumItems();
    Element^ GetSelectedItem();
    Element^ GetSelectedItem(size_t index);
    size_t GetValue(ptrdiff_t aux);
    float GetValueF(ptrdiff_t aux);
    size_t SetValue(ptrdiff_t value);
    size_t SetValueF(float value);
    size_t SetAsset(Asset^ res);
    size_t SetUV(UnifiedRect^ uv);
    size_t SetColor(unsigned int color, FGSETCOLOR index);
    size_t SetOutline(float outline);
    size_t SetFont(Font^ font);
    size_t SetLineHeight(float lineheight);
    size_t SetLetterSpacing(float letterspacing);
    size_t SetText(System::String^ text);
    size_t SetPlaceholder(System::String^ text);
    size_t SetMask(wchar_t mask);
    Asset^ GetAsset();
    UnifiedRect^ GetUV();
    unsigned int GetColor(FGSETCOLOR index);
    float GetOutline();
    Font^ GetFont();
    float GetLineHeight();
    float GetLetterSpacing();
    System::String^ GetText();
    System::String^ GetPlaceholder();
    wchar_t GetMask();
    void AddListener(unsigned short type, fgListener listener);
    Element^ GetChildByName(System::String^ name);

    static operator fgElement*(Element^ e);
    static AbsRect From(System::Drawing::RectangleF^ r);
    static AbsVec From(System::Drawing::PointF^ p);
    static fgIntVec From(System::Drawing::Point^ p);

  protected:
    fgElement* _p;
    bool _owner;
  };
}