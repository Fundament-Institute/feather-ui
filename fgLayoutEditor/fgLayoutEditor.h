// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_LAYOUT_EDITOR_H__
#define __FG_LAYOUT_EDITOR_H__

#include "fgAll.h"
#include "bss-util/cTOML.h"

typedef struct FG_LAYOUT_ACTION
{
  fgElement* self;
  FG_Msg msg;
  struct FG_LAYOUT_ACTION* next;
} fgLayoutAction;

struct EditorSettings
{
  bool showgrid;
  bool showcrosshairs;
  bool showrulers;
  bool showcursors;
  bool snaptogrid;
  bool snapnear;
  bool showwireframe;
  std::vector<std::string> recent;

  template<typename Engine>
  void Serialize(bss::cSerializer<Engine>& e)
  {
    e.template EvaluateType<EditorSettings>(
      bss::GenPair("showgrid", showgrid),
      bss::GenPair("showcrosshairs", showcrosshairs),
      bss::GenPair("showrulers", showrulers),
      bss::GenPair("showcursors", showcursors),
      bss::GenPair("snaptogrid", snaptogrid),
      bss::GenPair("snapnear", snapnear),
      bss::GenPair("showwireframe", showwireframe),
      bss::GenPair("recent", recent)
      );
  }
};

class fgLayoutEditor
{
public:
  fgLayoutEditor(fgLayout* layout, EditorSettings& settings);
  ~fgLayoutEditor();
  void OpenLayout(fgLayout* layout);

  static void MenuFile(struct _FG_ELEMENT*, const FG_Msg*);
  static void MenuRecent(struct _FG_ELEMENT*, const FG_Msg*);
  static void MenuEdit(struct _FG_ELEMENT*, const FG_Msg*);
  static void MenuView(struct _FG_ELEMENT*, const FG_Msg*);
  static void MenuHelp(struct _FG_ELEMENT*, const FG_Msg*);
  static void ExplorerOnFocus(struct _FG_ELEMENT*, const FG_Msg*);

  static fgLayoutEditor* Instance;

  enum MUTABLE_PROPERTIES : FG_UINT {
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
    PROP_SPLITTER,
    PROP_CONTEXTMENU,
    PROP_TOTALPLUSONE,
  };
  
  fgTextbox EditBox;

protected:
  void _openlayout(fgElement* root, const fgVectorClassLayout& layout);
  void _addprop(fgGrid& e, const char* name, const char* type = "Text", FG_UINT userid = 0);
  void _setprops(fgGrid& g, fgClassLayout& layout);
  void _addmutableprop(fgGrid& g, MUTABLE_PROPERTIES id, const char* type);
  void _clearprops(fgGrid& g, fgClassLayout& layout);
  fgElement* FindProp(fgGrid& g, MUTABLE_PROPERTIES prop);

  static size_t _inject(fgRoot* self, const FG_Msg* msg);
  static size_t _propertyMessage(fgText* self, const FG_Msg* msg);

  fgWindow* _mainwindow;
  fgTreeview* _explorer;
  fgGrid* _properties;
  fgTreeview* _skinexplorer;
  fgGrid* _skinprops;
  fgWorkspace* _workspace;

  fgLayout* _layout;
  fgElement* _selected;
  fgLayoutAction* _undo;
  fgLayoutAction* _redo;
  EditorSettings _settings;
};

#endif