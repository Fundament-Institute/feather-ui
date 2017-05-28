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
  bool Close();
  bool CheckSave();
  char ShowDialog(const char* text, const char* button1, const char* button2 = 0, const char* button3 = 0);
  virtual void Destroy() override;
  void MenuFile(struct _FG_ELEMENT*, const FG_Msg*);
  void MenuRecent(struct _FG_ELEMENT*, const FG_Msg*);
  void MenuEdit(struct _FG_ELEMENT*, const FG_Msg*);
  void MenuView(struct _FG_ELEMENT*, const FG_Msg*);
  void MenuHelp(struct _FG_ELEMENT*, const FG_Msg*);
  void ToolAction(struct _FG_ELEMENT*, const FG_Msg*);
  void SaveSettings();
  virtual void ReapplySkin(fgSkin* skin) override;
  void FillRecentList();
  void ProcessEvent(int id);
  virtual void SetNeedSave(bool needsave) override;

  static size_t WorkspaceRootMessage(fgElement* e, const FG_Msg* m);
  static fgElement* LoadLayout(fgElement* parent, fgElement* next, fgClassLayout* layout);
  static std::string FileDialog(bool open, unsigned long flags, const char* file, const wchar_t* filter, const char* initdir, const char* defext);
  static uint8_t HitElement(fgElement* target, const FG_Msg* m);
    
  static fgLayoutEditor* Instance;
  static const int MAX_OPEN_HISTORY = 10;

  enum EVENTS {
    EVENT_NEW = 1,
    EVENT_OPEN,
    EVENT_SAVE,
    EVENT_SAVEAS,
    EVENT_CLOSE,
    EVENT_EXIT,
    EVENT_CUT,
    EVENT_COPY,
    EVENT_PASTE,
    EVENT_DELETE,
    EVENT_UNDO,
    EVENT_REDO,
    EVENT_HELP_MANUAL,
    EVENT_HELP_DOCS,
    EVENT_HELP_GITHUB,
    EVENT_HELP_ABOUT,
  };
  enum CORNERHIT {
    HIT_TOP = 1,
    HIT_BOTTOM = 2,
    HIT_LEFT = 4,
    HIT_RIGHT = 8,
    HIT_NONE = 15,
  };

protected:
  static size_t _inject(fgRoot* self, const FG_Msg* msg);

  fgWorkspace* _workspace;
  fgLayout* displaylayout; // Currently displayed layout or sublayout
  fgElement* _workspaceroot;
  fgElement* _toolbar;
  fgElement* _recentmenu;
  fgSkin* _resizebox;
  
  bool _needsave;
  LayoutTab _layout;
  SkinTab _skin;
  fgLayoutAction* _undo;
  fgLayoutAction* _redo;
  EditorSettings _settings;
  std::string _path;
  uint8_t _sizing;
  AbsVec _lastmouse;
  AbsVec _anchor;
};

#endif