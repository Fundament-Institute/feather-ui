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
  void Serialize(bss_util::cSerializer<Engine>& e)
  {
    e.template EvaluateType<EditorSettings>(
      bss_util::GenPair("showgrid", showgrid),
      bss_util::GenPair("showcrosshairs", showcrosshairs),
      bss_util::GenPair("showrulers", showrulers),
      bss_util::GenPair("showcursors", showcursors),
      bss_util::GenPair("snaptogrid", snaptogrid),
      bss_util::GenPair("snapnear", snapnear),
      bss_util::GenPair("showwireframe", showwireframe),
      bss_util::GenPair("recent", recent)
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

protected:
  void _openlayout(fgElement* root, const fgVectorClassLayout& layout);
  void _addprop(fgGrid* e, const char* name);
  void _setprops(fgGrid* e, fgClassLayout& layout);
  static size_t _inject(fgRoot* self, const FG_Msg* msg);

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