// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __EDITOR_BASE_H__
#define __EDITOR_BASE_H__

#include "fgAll.h"
#include "bss-util/Str.h"

class EditorBase
{
public:
  enum PROPERTIES : FG_UINT {
    PROP_USERID = 1,
    PROP_USERINFO,
    PROP_TEXT,
    PROP_PLACEHOLDER,
    PROP_FONT,
    PROP_LINEHEIGHT,
    PROP_LETTERSPACING,
    PROP_COLOR,
    PROP_PLACECOLOR,
    PROP_CURSORCOLOR,
    PROP_SELECTCOLOR,
    PROP_HOVERCOLOR,
    PROP_DRAGCOLOR,
    PROP_EDGECOLOR,
    PROP_DIVIDERCOLOR,
    PROP_COLUMNDIVIDERCOLOR,
    PROP_ROWEVENCOLOR,
    PROP_VALUEI,
    PROP_VALUEF,
    PROP_RANGE,
    PROP_UV,
    PROP_ASSET,
    PROP_OUTLINE,
    PROP_CONTEXTMENU,
    PROP_TOOLTIP,
    PROP_TOTALPLUSONE, // End of mutable properties
    PROP_ID,
    PROP_NAME,
    PROP_SKIN,
    PROP_AREA,
    PROP_ROTATION,
    PROP_CENTER,
    PROP_MARGIN,
    PROP_PADDING,
    PROP_MINDIM,
    PROP_MAXDIM,
    PROP_SCALING,
    PROP_STYLE,
    PROP_ORDER,
    PROP_FLAGS,
    PROP_ALPHA,
  };

  enum TARGETTYPE {
    TYPE_LAYOUT = 1,
    TYPE_CLASSLAYOUT,
    TYPE_SKINLAYOUT,
    TYPE_SKIN,
    TYPE_STYLE,
  };
  explicit EditorBase(fgLayout* layout);
  ~EditorBase();
  fgElement* AddProp(fgGrid& e, const char* name, const char* type = "Text", FG_UINT userid = 0, fgMessage fn = 0, fgFlag flags = FGELEMENT_EXPANDY);
  void SetProps(fgGrid& g, fgClassLayout* layout, fgSkinElement* element, const fgSkin* skin, const char* name, fgStyle& style);
  void AddMutableProp(fgGrid& g, PROPERTIES id, const char* type, std::function<void(fgElement*, const char*)>& f, fgFlag flags = FGTEXTBOX_ACTION | FGTEXTBOX_SINGLELINE | FGELEMENT_EXPANDY);
  void ClearProps(fgGrid& g);
  void LoadProps(fgGrid& g, const char* type, fgClassLayout* layout, fgSkinElement* element, const fgSkin* skin, const char* name, fgStyle& style, std::function<void(fgElement*, const char*)>& f);
  void ParseStyleMsg(fgStyle& target, fgElement* instance, fgSkinElement* element, fgClassLayout* layout, fgSkin** skin, const char** name, PROPERTIES id, const char* s);
  uint32_t ParseColor(const char* s);
  uint16_t GetTransformMsg(const fgStyle& target, fgTransform& out);
  fgElement* FindProp(fgGrid& g, PROPERTIES prop);
  virtual void Destroy() = 0;
  virtual void DisplayLayout(fgLayout* layout) = 0;
  void WindowOnDestroy(struct _FG_ELEMENT*, const FG_Msg*);
  virtual void ReapplySkin(fgSkin* skin) = 0;
  virtual void SetNeedSave(bool needsave) = 0;
  virtual void AddUndo() = 0;
  virtual void SetCurElement(fgElement* cur) = 0;
  void AddFlagCheckboxes(fgElement* parent, const char* type, fgFlag& flags);
  fgFlag GetFlagCheckboxes(fgElement* box);
  void ParseFlagMsg(fgStyle& target, fgElement* instance, fgSkinElement* element, fgFlag flags);

  template<typename T, size_t(*F)(char*, size_t, const T*, fgMsgType)>
  inline bss::Str WrapWrite(const T& v, fgMsgType u)
  {
    bss::Str s;
    s.resize(F(0, 0, &v, u));
    s.resize(F(s.UnsafeString(), s.size(), &v, u));
    return s;
  }

  static void RemoveStyleMsg(fgStyle& s, uint16_t type, uint16_t subtype = (uint16_t)~0);
  static void AddMenuControls(const char* id);

  fgLayout curlayout; // Currently loaded root layout
  fgElement* hoverelement;

protected:
  fgElement* _window;
};

#endif
