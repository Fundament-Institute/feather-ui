// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_LAYOUT_EDITOR_H__
#define __FG_LAYOUT_EDITOR_H__

#include "LayoutTab.h"
#include "SkinTab.h"
#include "bss-util/TOML.h"

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
  void Serialize(bss::Serializer<Engine>& e)
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

class fgLayoutEditor : public EditorBase
{
public:
  fgLayoutEditor(fgLayout* layout, EditorSettings& settings);
  ~fgLayoutEditor();
  inline LayoutTab& Layout() { return _layout; }
  inline SkinTab& Skin() { return _skin; }
  virtual void DisplayLayout(fgLayout* layout) override;
  void OpenLayout();
  void LoadFile(const char* file);
  void SaveFile();
  void NewFile();
  void Close();
  void CheckSave();
  void ShowDialog(const char* text, const char* button1, const char* button2 = 0, const char* button3 = 0);
  virtual void Destroy() override;
  void MenuFile(struct _FG_ELEMENT*, const FG_Msg*);
  void MenuRecent(struct _FG_ELEMENT*, const FG_Msg*);
  void MenuEdit(struct _FG_ELEMENT*, const FG_Msg*);
  void MenuView(struct _FG_ELEMENT*, const FG_Msg*);
  void MenuHelp(struct _FG_ELEMENT*, const FG_Msg*);
  void SaveSettings();

  static size_t WorkspaceMessage(fgWorkspace* e, const FG_Msg* m);
  static fgElement* LoadLayout(fgElement* parent, fgElement* next, fgClassLayout* layout);
  static std::string FileDialog(bool open, unsigned long flags, const char* file, const wchar_t* filter, const char* initdir, const char* defext);
  
  static fgLayoutEditor* Instance;
  static const int MAX_OPEN_HISTORY = 10;

protected:
  static size_t _inject(fgRoot* self, const FG_Msg* msg);

  fgWindow* _mainwindow;
  fgWorkspace* _workspace;
  fgLayout* displaylayout; // Currently displayed layout or sublayout

  LayoutTab _layout;
  SkinTab _skin;
  fgLayoutAction* _undo;
  fgLayoutAction* _redo;
  EditorSettings _settings;
  std::string _path;
};

#endif