// Copyright ©2018 Black Sphere Studios
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
  void Serialize(bss::Serializer<Engine>& e, const char*)
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
  void NewFile(std::function<void()> fn);
  void Close(std::function<void()> fn);
  void CheckSave(std::function<void()> fn);
  void ShowDialog(const char* title, const char* text, std::function<void(char)> fn, const char* button1, const char* button2 = 0, const char* button3 = 0);
  virtual void Destroy() override;
  void MenuAction(struct _FG_ELEMENT*, const FG_Msg*);
  void MenuRecent(struct _FG_ELEMENT*, const FG_Msg*);
  void ToolAction(struct _FG_ELEMENT*, const FG_Msg*);
  void ControlAction(struct _FG_ELEMENT*, const FG_Msg*);
  void SaveSettings();
  virtual void ReapplySkin(fgSkin* skin) override;
  void FillRecentList();
  void ProcessEvent(int id);
  virtual void SetNeedSave(bool needsave) override;
  virtual void AddUndo() override;
  void ApplyLayoutCopy(fgLayout* layout);
  void EnableEvent(FG_UINT id, bool enable);
  virtual void SetCurElement(fgElement* cur) override { _curelement = cur; EnableEvent(EVENT_DELETE, cur != 0); }

  static size_t WorkspaceRootMessage(fgElement* e, const FG_Msg* m);
  static fgElement* LoadLayout(fgElement* parent, fgElement* next, fgClassLayout* layout);
  static std::string FileDialog(bool open, unsigned long flags, const char* file, const wchar_t* filter, const char* initdir, const char* defext);
  static uint8_t HitElement(fgElement* target, const FG_Msg* m);
  static void ClearLayoutVector(std::vector<fgLayout*>& v);
  static fgElement* FindUserID(fgElement* parent, FG_UINT userid);

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
    EVENT_NUM,
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
  fgElement* _toolbarcontrols;
  fgElement* _menu;
  fgElement* _recentmenu;
  fgSkin* _resizebox;
  fgElement* _toolbarbuttons[EVENT_NUM];
  fgElement* _menubuttons[EVENT_NUM];

  const char* _insert;
  fgElement* _insertdest;
  AbsVec _insertbegin;
  AbsVec _insertend;
  bool _needsave;
  LayoutTab _layout;
  SkinTab _skin;
  std::vector<fgLayout*> _history;
  ptrdiff_t _historypos; // location of where we are in our history (can be -1 to represent the current state if it hasn't been added to the history stack)
  EditorSettings _settings;
  std::string _path;
  uint8_t _sizing;
  AbsVec _lastmouse;
  AbsVec _anchor;
  fgLayout* _dialog;
  fgElement* _curelement; // currently selected element in the workspace
};

#endif