/* graphics_interface.h - Standard C interface for feather desktop interfaces
Copyright (c)2022 Fundament Software

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef DESKTOP_INTERFACE_H
#define DESKTOP_INTERFACE_H

#include "shared_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FG_Display__
{
  FG_Vec2i size;
  FG_Vec2i offset;
  FG_Vec2 dpi;
  float scale;
  uintptr_t handle;
  bool primary;
} FG_Display;

enum FG_Clipboard
{
  FG_Clipboard_None    = 0,
  FG_Clipboard_Text    = 1,
  FG_Clipboard_Wave    = 2,
  FG_Clipboard_Bitmap  = 3,
  FG_Clipboard_File    = 4,
  FG_Clipboard_Element = 5,
  FG_Clipboard_Custom  = 6,
  FG_Clipboard_All     = 7,
};

enum FG_Cursor
{
  FG_Cursor_None       = 0,
  FG_Cursor_Arrow      = 1,
  FG_Cursor_IBeam      = 2,
  FG_Cursor_Cross      = 3,
  FG_Cursor_Wait       = 4,
  FG_Cursor_Hand       = 5,
  FG_Cursor_ResizeNS   = 6,
  FG_Cursor_ResizeWE   = 7,
  FG_Cursor_ResizeNWSE = 8,
  FG_Cursor_ResizeNESW = 9,
  FG_Cursor_ResizeALL  = 10,
  FG_Cursor_No         = 11,
  FG_Cursor_Help       = 12,
  FG_Cursor_Drag       = 13,
  FG_Cursor_Custom     = 14,
};

enum FG_Event_Kind
{
  FG_Event_Kind_Action         = 0,
  FG_Event_Kind_Draw           = 1,
  FG_Event_Kind_Drop           = 2,
  FG_Event_Kind_GetWindowFlags = 3,
  FG_Event_Kind_GotFocus       = 4,
  FG_Event_Kind_JoyAxis        = 5,
  FG_Event_Kind_JoyButtonDown  = 6,
  FG_Event_Kind_JoyButtonUp    = 7,
  FG_Event_Kind_JoyOrientation = 8,
  FG_Event_Kind_KeyChar        = 9,
  FG_Event_Kind_KeyDown        = 10,
  FG_Event_Kind_KeyUp          = 11,
  FG_Event_Kind_LostFocus      = 12,
  FG_Event_Kind_MouseDblClick  = 13,
  FG_Event_Kind_MouseDown      = 14,
  FG_Event_Kind_MouseMove      = 15,
  FG_Event_Kind_MouseOff       = 16,
  FG_Event_Kind_MouseOn        = 17,
  FG_Event_Kind_MouseScroll    = 18,
  FG_Event_Kind_MouseUp        = 19,
  FG_Event_Kind_SetWindowFlags = 20,
  FG_Event_Kind_SetWindowRect  = 21,
  FG_Event_Kind_TouchBegin     = 22,
  FG_Event_Kind_TouchEnd       = 23,
  FG_Event_Kind_TouchMove      = 24,
};

enum FG_Keys
{
  FG_Keys_4               = 52,
  FG_Keys_S               = 83,
  FG_Keys_PAGEDOWN        = 34,
  FG_Keys_EXECUTE         = 43,
  FG_Keys_F7              = 118,
  FG_Keys_PAGEUP          = 33,
  FG_Keys_8               = 56,
  FG_Keys_W               = 87,
  FG_Keys_NUMPAD_EQUAL    = 146,
  FG_Keys_6               = 54,
  FG_Keys_F16             = 127,
  FG_Keys_NUMPAD5         = 101,
  FG_Keys_Y               = 89,
  FG_Keys_RETURN          = 13,
  FG_Keys_F9              = 120,
  FG_Keys_LSUPER          = 91,
  FG_Keys_KANA            = 21,
  FG_Keys_BACK            = 8,
  FG_Keys_GRAVE           = 192,
  FG_Keys_DELETE          = 46,
  FG_Keys_LSHIFT          = 160,
  FG_Keys_F14             = 125,
  FG_Keys_JUNJA           = 23,
  FG_Keys_RSUPER          = 92,
  FG_Keys_FINAL           = 24,
  FG_Keys_B               = 66,
  FG_Keys_LCONTROL        = 162,
  FG_Keys_INSERT          = 45,
  FG_Keys_F15             = 126,
  FG_Keys_APPS            = 93,
  FG_Keys_XBUTTON1        = 5,
  FG_Keys_SELECT          = 41,
  FG_Keys_H               = 72,
  FG_Keys_F               = 70,
  FG_Keys_F21             = 132,
  FG_Keys_J               = 74,
  FG_Keys_L               = 76,
  FG_Keys_NUMLOCK         = 144,
  FG_Keys_RSHIFT          = 161,
  FG_Keys_COMMA           = 188,
  FG_Keys_F20             = 131,
  FG_Keys_NUMPAD1         = 97,
  FG_Keys_LEFT_BRACKET    = 219,
  FG_Keys_SPACE           = 32,
  FG_Keys_F18             = 129,
  FG_Keys_F23             = 134,
  FG_Keys_SEMICOLON       = 186,
  FG_Keys_MODECHANGE      = 31,
  FG_Keys_MENU            = 18,
  FG_Keys_NUMPAD9         = 105,
  FG_Keys_5               = 53,
  FG_Keys_R               = 82,
  FG_Keys_F19             = 130,
  FG_Keys_F6              = 117,
  FG_Keys_T               = 84,
  FG_Keys_NUMPAD_MULTIPLY = 106,
  FG_Keys_ACCEPT          = 30,
  FG_Keys_F22             = 133,
  FG_Keys_UP              = 38,
  FG_Keys_NUMPAD2         = 98,
  FG_Keys_CANCEL          = 3,
  FG_Keys_9               = 57,
  FG_Keys_X               = 88,
  FG_Keys_F25             = 136,
  FG_Keys_NUMPAD0         = 96,
  FG_Keys_V               = 86,
  FG_Keys_DOWN            = 40,
  FG_Keys_SNAPSHOT        = 44,
  FG_Keys_NUMPAD3         = 99,
  FG_Keys_F24             = 135,
  FG_Keys_NUMPAD8         = 104,
  FG_Keys_APOSTROPHE      = 222,
  FG_Keys_A               = 65,
  FG_Keys_NUMPAD6         = 102,
  FG_Keys_TAB             = 9,
  FG_Keys_LBUTTON         = 1,
  FG_Keys_PLUS            = 187,
  FG_Keys_C               = 67,
  FG_Keys_RIGHT_BRACKET   = 221,
  FG_Keys_BACKSLASH       = 220,
  FG_Keys_SLASH           = 191,
  FG_Keys_PERIOD          = 190,
  FG_Keys_HOME            = 36,
  FG_Keys_NUMPAD4         = 100,
  FG_Keys_F1              = 112,
  FG_Keys_LMENU           = 164,
  FG_Keys_E               = 69,
  FG_Keys_RCONTROL        = 163,
  FG_Keys_D               = 68,
  FG_Keys_NONCONVERT      = 29,
  FG_Keys_F11             = 122,
  FG_Keys_F8              = 119,
  FG_Keys_7               = 55,
  FG_Keys_NUMPAD7         = 103,
  FG_Keys_G               = 71,
  FG_Keys_3               = 51,
  FG_Keys_F4              = 115,
  FG_Keys_RIGHT           = 39,
  FG_Keys_RMENU           = 165,
  FG_Keys_XBUTTON2        = 6,
  FG_Keys_SCROLL          = 145,
  FG_Keys_CAPITAL         = 20,
  FG_Keys_F12             = 123,
  FG_Keys_I               = 73,
  FG_Keys_RBUTTON         = 2,
  FG_Keys_1               = 49,
  FG_Keys_OEM_8           = 223,
  FG_Keys_P               = 80,
  FG_Keys_U               = 85,
  FG_Keys_Z               = 90,
  FG_Keys_NULL            = 0,
  FG_Keys_CONTROL         = 17,
  FG_Keys_NUMPAD_DECIMAL  = 110,
  FG_Keys_K               = 75,
  FG_Keys_CLEAR           = 12,
  FG_Keys_M               = 77,
  FG_Keys_F2              = 113,
  FG_Keys_KANJI           = 25,
  FG_Keys_ESCAPE          = 27,
  FG_Keys_SHIFT           = 16,
  FG_Keys_F13             = 124,
  FG_Keys_MBUTTON         = 4,
  FG_Keys_LEFT            = 37,
  FG_Keys_HELP            = 47,
  FG_Keys_MINUS           = 189,
  FG_Keys_N               = 78,
  FG_Keys_0               = 48,
  FG_Keys_CONVERT         = 28,
  FG_Keys_O               = 79,
  FG_Keys_PRINT           = 42,
  FG_Keys_SLEEP           = 95,
  FG_Keys_NUMPAD_ADD      = 107,
  FG_Keys_NUMPAD_DIVIDE   = 111,
  FG_Keys_END             = 35,
  FG_Keys_F10             = 121,
  FG_Keys_NUMPAD_SUBTRACT = 109,
  FG_Keys_SEPARATOR       = 108,
  FG_Keys_Q               = 81,
  FG_Keys_F5              = 116,
  FG_Keys_2               = 50,
  FG_Keys_F3              = 114,
  FG_Keys_PAUSE           = 19,
  FG_Keys_F17             = 128
};

enum FG_ModKey
{
  FG_ModKey_Shift    = 1,
  FG_ModKey_Control  = 2,
  FG_ModKey_Alt      = 4,
  FG_ModKey_Super    = 8,
  FG_ModKey_Capslock = 16,
  FG_ModKey_Numlock  = 32,
  FG_ModKey_Held     = 64,
};

enum FG_MouseButton
{
  FG_MouseButton_L  = 1,
  FG_MouseButton_R  = 2,
  FG_MouseButton_M  = 4,
  FG_MouseButton_X1 = 8,
  FG_MouseButton_X2 = 16,
  FG_MouseButton_X3 = 32,
  FG_MouseButton_X4 = 64,
  FG_MouseButton_X5 = 128,
};

enum FG_JoyAxis
{
  FG_JoyAxis_X       = 0,
  FG_JoyAxis_Y       = 1,
  FG_JoyAxis_Z       = 2,
  FG_JoyAxis_R       = 3,
  FG_JoyAxis_U       = 4,
  FG_JoyAxis_V       = 5,
  FG_JoyAxis_Invalid = 65535,
};

enum FG_Joy
{
  FG_Joy_Button4  = 3,
  FG_Joy_Button21 = 20,
  FG_Joy_Button11 = 10,
  FG_Joy_Button12 = 11,
  FG_Joy_Button25 = 24,
  FG_Joy_Button3  = 2,
  FG_Joy_Button15 = 14,
  FG_Joy_ID2      = 256,
  FG_Joy_Button31 = 30,
  FG_Joy_Button32 = 31,
  FG_Joy_Button2  = 1,
  FG_Joy_Button28 = 27,
  FG_Joy_Button18 = 17,
  FG_Joy_ID7      = 1536,
  FG_Joy_ID15     = 3584,
  FG_Joy_ID14     = 3328,
  FG_Joy_Button22 = 21,
  FG_Joy_Button9  = 8,
  FG_Joy_ID12     = 2816,
  FG_Joy_ID3      = 512,
  FG_Joy_ID11     = 2560,
  FG_Joy_ID10     = 2304,
  FG_Joy_ID9      = 2048,
  FG_Joy_ID1      = 0,
  FG_Joy_ID8      = 1792,
  FG_Joy_Button20 = 19,
  FG_Joy_ID16     = 3840,
  FG_Joy_Button8  = 7,
  FG_Joy_ID6      = 1280,
  FG_Joy_Button26 = 25,
  FG_Joy_ID5      = 1024,
  FG_Joy_Button23 = 22,
  FG_Joy_Button1  = 0,
  FG_Joy_Button24 = 23,
  FG_Joy_ID4      = 768,
  FG_Joy_Button29 = 28,
  FG_Joy_Button16 = 15,
  FG_Joy_Button7  = 6,
  FG_Joy_Button10 = 9,
  FG_Joy_Button27 = 26,
  FG_Joy_Button30 = 29,
  FG_Joy_Button5  = 4,
  FG_Joy_Button17 = 16,
  FG_Joy_Button19 = 18,
  FG_Joy_ID13     = 3072,
  FG_Joy_Button14 = 13,
  FG_Joy_Button13 = 12,
  FG_Joy_Button6  = 5
};

struct FG_Msg_TouchMove
{
  float x;
  float y;
  float z;
  float r;
  float pressure;
  uint16_t index;
  uint8_t flags;
  uint8_t modkeys;
};
struct FG_Msg_MouseScroll
{
  float x;
  float y;
  float delta;
  float hdelta;
};
struct FG_Msg_MouseOn
{
  float x;
  float y;
  uint8_t all;
  uint8_t modkeys;
};
struct FG_Msg_JoyButtonUp
{
  uint16_t index;
  uint16_t button;
  uint8_t modkeys;
};
struct FG_Msg_MouseOff
{
  float x;
  float y;
  uint8_t all;
  uint8_t modkeys;
};
struct FG_Msg_JoyButtonDown
{
  uint16_t index;
  uint16_t button;
  uint8_t modkeys;
};
struct FG_Msg_GetWindowFlags
{
  char __padding;
};
struct FG_Msg_KeyDown
{
  uint8_t key;
  uint8_t modkeys;
  uint16_t scancode;
};
struct FG_Msg_Action
{
  int32_t subkind;
};
struct FG_Msg_KeyChar
{
  int32_t unicode;
  uint8_t modkeys;
};
struct FG_Msg_KeyUp
{
  uint8_t key;
  uint8_t modkeys;
  uint16_t scancode;
};
struct FG_Msg_TouchEnd
{
  float x;
  float y;
  float z;
  float r;
  float pressure;
  uint16_t index;
  uint8_t flags;
  uint8_t modkeys;
};
struct FG_Msg_TouchBegin
{
  float x;
  float y;
  float z;
  float r;
  float pressure;
  uint16_t index;
  uint8_t flags;
  uint8_t modkeys;
};
struct FG_Msg_SetWindowFlags
{
  uint32_t flags;
};
struct FG_Msg_MouseUp
{
  float x;
  float y;
  uint8_t all;
  uint8_t modkeys;
  uint8_t button;
};
struct FG_Msg_MouseDblClick
{
  float x;
  float y;
  uint8_t all;
  uint8_t modkeys;
  uint8_t button;
};
struct FG_Msg_JoyAxis
{
  uint16_t index;
  float value;
  uint16_t axis;
  uint8_t modkeys;
};
struct FG_Msg_LostFocus
{
  char __padding;
};
struct FG_Msg_MouseDown
{
  float x;
  float y;
  uint8_t all;
  uint8_t modkeys;
  uint8_t button;
};
struct FG_Msg_JoyOrientation
{
  uint16_t index;
  FG_Vec3 velocity;
  FG_Vec3 rotation;
};
struct FG_Msg_Draw
{
  FG_Rect area;
};
struct FG_Msg_Drop
{
  int32_t kind;
  void* target;
  uint32_t count;
};
struct FG_Msg_SetWindowRect
{
  FG_Rect rect;
};
struct FG_Msg_GotFocus
{
  char __padding;
};
struct FG_Msg_MouseMove
{
  float x;
  float y;
  uint8_t all;
  uint8_t modkeys;
};

typedef struct FG_Msg__
{
  uint16_t kind;
  union
  {
    struct FG_Msg_TouchMove touchMove;
    struct FG_Msg_MouseScroll mouseScroll;
    struct FG_Msg_MouseOn mouseOn;
    struct FG_Msg_JoyButtonUp joyButtonUp;
    struct FG_Msg_MouseOff mouseOff;
    struct FG_Msg_JoyButtonDown joyButtonDown;
    struct FG_Msg_GetWindowFlags getWindowFlags;
    struct FG_Msg_KeyDown keyDown;
    struct FG_Msg_Action action;
    struct FG_Msg_KeyChar keyChar;
    struct FG_Msg_KeyUp keyUp;
    struct FG_Msg_TouchEnd touchEnd;
    struct FG_Msg_TouchBegin touchBegin;
    struct FG_Msg_SetWindowFlags setWindowFlags;
    struct FG_Msg_MouseUp mouseUp;
    struct FG_Msg_MouseDblClick mouseDblClick;
    struct FG_Msg_JoyAxis joyAxis;
    struct FG_Msg_LostFocus lostFocus;
    struct FG_Msg_MouseDown mouseDown;
    struct FG_Msg_JoyOrientation joyOrientation;
    struct FG_Msg_Draw draw;
    struct FG_Msg_Drop drop;
    struct FG_Msg_SetWindowRect setWindowRect;
    struct FG_Msg_GotFocus gotFocus;
    struct FG_Msg_MouseMove mouseMove;
  };
} FG_Msg;

typedef union FG_Result__
{
  int32_t touchMove;
  int32_t mouseScroll;
  int32_t mouseOn;
  int32_t joyButtonUp;
  int32_t mouseOff;
  int32_t joyButtonDown;
  uint32_t getWindowFlags;
  int32_t keyDown;
  int32_t action;
  int32_t keyChar;
  int32_t keyUp;
  int32_t touchEnd;
  int32_t touchBegin;
  int32_t setWindowFlags;
  int32_t mouseUp;
  int32_t mouseDblClick;
  int32_t joyAxis;
  int32_t lostFocus;
  int32_t mouseDown;
  int32_t joyOrientation;
  int32_t draw;
  uintptr_t drop;
  int32_t setWindowRect;
  int32_t gotFocus;
  int32_t mouseMove;
} FG_Result;

enum FG_WindowFlag
{
  FG_WindowFlag_Minimizable = 1,
  FG_WindowFlag_Maximizable = 2,
  FG_WindowFlag_Resizable   = 4,
  FG_WindowFlag_No_Caption  = 8,
  FG_WindowFlag_No_Border   = 16,
  FG_WindowFlag_Minimized   = 32,
  FG_WindowFlag_Maximized   = 64,
  FG_WindowFlag_Closed      = 128,
  FG_WindowFlag_Fullscreen  = 256,
};

typedef struct FG_Window__
{
  uintptr_t handle;
  FG_Context* context;
  uintptr_t window_id;
} FG_Window;

typedef FG_Result (*FG_Behavior)(FG_Window*, FG_Msg*, void*, uintptr_t);

struct FG_DesktopInterface
{
  FG_Window* (*createWindow)(struct FG_DesktopInterface* self, uintptr_t window_id, FG_Display* display, FG_Vec2* pos,
                            FG_Vec2* dim, const char* caption, uint64_t flags);
  int (*setWindow)(struct FG_DesktopInterface* self, FG_Window* window, FG_Display* display, FG_Vec2* pos, FG_Vec2* dim,
                   const char* caption, uint64_t flags);
  int (*destroyWindow)(struct FG_DesktopInterface* self, FG_Window* window);
  int (*invalidateWindow)(struct FG_DesktopInterface* self, FG_Window* window, FG_Rect* optarea);
  int (*putClipboard)(struct FG_DesktopInterface* self, FG_Window* window, enum FG_Clipboard kind, const char* data,
                      uint32_t count);
  uint32_t (*getClipboard)(struct FG_DesktopInterface* self, FG_Window* window, enum FG_Clipboard kind, void* target,
                           uint32_t count);
  bool (*checkClipboard)(struct FG_DesktopInterface* self, FG_Window* window, enum FG_Clipboard kind);
  int (*clearClipboard)(struct FG_DesktopInterface* self, FG_Window* window, enum FG_Clipboard kind);
  int (*processMessages)(struct FG_DesktopInterface* self, FG_Window* window, void* ui_state);
  int (*getMessageSyncObject)(struct FG_DesktopInterface* self, FG_Window* window);
  int (*setCursor)(struct FG_DesktopInterface* self, FG_Window* window, enum FG_Cursor cursor);
  int (*getDisplayIndex)(struct FG_DesktopInterface* self, unsigned int index, FG_Display* out);
  int (*getDisplay)(struct FG_DesktopInterface* self, uintptr_t handle, FG_Display* out);
  int (*getDisplayWindow)(struct FG_DesktopInterface* self, FG_Window* window, FG_Display* out);
  int (*createSystemControl)(struct FG_DesktopInterface* self, FG_Window* window, const char* id, FG_Rect* area, ...);
  int (*setSystemControl)(struct FG_DesktopInterface* self, FG_Window* window, void* control, FG_Rect* area, ...);
  int (*destroySystemControl)(struct FG_DesktopInterface* self, FG_Window* window, void* control);
  int (*destroy)(struct FG_DesktopInterface* self);

  int cursorblink;
  int tooltipdelay;
};

typedef struct FG_DesktopInterface* (*FG_InitDesktop)(void*, FG_Log, FG_Behavior);

#ifdef __cplusplus
}
#endif

#endif
