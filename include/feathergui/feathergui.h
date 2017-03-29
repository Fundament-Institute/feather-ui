/* Feather - Lightweight GUI Abstraction Layer
   Copyright ©2017 Black Sphere Studios

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


#ifndef __FEATHER_GUI_H__
#define __FEATHER_GUI_H__

#include <string.h> // memcpy,memset
#include <stddef.h>
#include "bss_compiler.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef float FREL; // Change this to double for double precision (why on earth you would need that for a GUI, however, is beyond me)
typedef float FABS; // We seperate both of these types because then you don't have to guess if a float is relative or absolute
typedef unsigned int FG_UINT; 
typedef unsigned int fgFlag;
typedef void* fgAsset;
typedef void* fgFont;

#define FGUI_VERSION_MAJOR 0
#define FGUI_VERSION_MINOR 1
#define FGUI_VERSION_REVISION 0
#define FG_EXTERN extern BSS_COMPILER_DLLEXPORT
#define FG_ACCEPT 1

#ifndef FG_STATIC_LIB
#ifdef feathergui_EXPORTS // All implementations need to define this to properly export the C++ functions in the feather static library.
#pragma warning(disable:4251)
#define FG_DLLEXPORT BSS_COMPILER_DLLEXPORT
#else
#define FG_DLLEXPORT BSS_COMPILER_DLLIMPORT
#endif
#else
#define FG_DLLEXPORT
#endif

#ifdef BSS_DEBUG // Used for dealing with windows eating assertions inside callback functions
#define fgassert(x) if(!(x)) { *((int*)0) = 0; } 
#else
#define fgassert(x) 
#endif

// A unified coordinate specifies things in terms of absolute and relative positions.
typedef struct {
  FABS abs; // Absolute coordinates are added to relative coordinates, which specify a point from a linear interpolation of the parent's dimensions
  FREL rel;
} Coord;

#define MAKE_VEC(_T,_N) typedef struct { _T x; _T y; } _N

MAKE_VEC(FREL,RelVec);
MAKE_VEC(FABS,AbsVec);
// A coordinate vector specifies a point by unified coordinates
MAKE_VEC(Coord,CVec);
MAKE_VEC(int, fgIntVec);

#define MAKE_RECT(_T,_T2,_N) typedef struct { \
  union { \
    struct { \
      _T left; \
      _T top; \
      _T right; \
      _T bottom; \
    }; \
    struct { \
      _T2 topleft; \
      _T2 bottomright; \
    }; \
  }; \
} _N

MAKE_RECT(FREL,RelVec,RelRect);
MAKE_RECT(FABS,AbsVec,AbsRect);
// A coordinate rect specifies its topleft and bottomright corners in terms of coordinate vectors.
MAKE_RECT(Coord,CVec,CRect);

static BSS_FORCEINLINE FABS fglerp(FABS a, FABS b, FREL amt)
{
	return a+((FABS)((b-a)*amt));
}

#define fgDeclareVector(type, n) struct __VECTOR__ ## n { \
  type* p; \
  size_t s; /* This is the total size of the array in BYTES. */ \
  size_t l; /* This is how much of the array is being used in ELEMENTS. */ \
}

typedef fgDeclareVector(void, void) fgVector;

typedef struct {
  CRect area;
  FABS rotation;
  CVec center;
} fgTransform;

typedef union
{
  unsigned int color;
  unsigned char colors[4];
  struct {
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
  };
} fgColor;

FG_EXTERN const fgColor fgColor_NONE; // Fully transparent
FG_EXTERN const fgColor fgColor_BLACK;
FG_EXTERN const fgColor fgColor_WHITE;
FG_EXTERN const fgTransform fgTransform_DEFAULT;
FG_EXTERN const fgTransform fgTransform_EMPTY;
FG_EXTERN const fgTransform fgTransform_CENTER;
FG_EXTERN const CRect CRect_EMPTY;
FG_EXTERN const AbsVec AbsVec_EMPTY;
FG_EXTERN const fgIntVec fgIntVec_EMPTY;

enum FGMOVE
{
  FGMOVE_PROPAGATE = (1 << 0),
  FGMOVE_RESIZEX = (1 << 1),
  FGMOVE_RESIZEY = (1 << 2),
  FGMOVE_RESIZE = (FGMOVE_RESIZEX | FGMOVE_RESIZEY),
  FGMOVE_MOVEX = (1 << 3),
  FGMOVE_MOVEY = (1 << 4),
  FGMOVE_MOVE = (FGMOVE_MOVEX | FGMOVE_MOVEY),
  FGMOVE_CENTERX = (1 << 5),
  FGMOVE_CENTERY = (1 << 6),
  FGMOVE_CENTER = (FGMOVE_CENTERX | FGMOVE_CENTERY),
  FGMOVE_ROTATION = (1 << 7),
  FGMOVE_PADDING = (1 << 8),
  FGMOVE_MARGIN = (1 << 9),
};

enum FGTEXTFMT
{
  FGTEXTFMT_UTF8 = 0,
  FGTEXTFMT_UTF16 = 1,
  FGTEXTFMT_UTF32 = 2,
  FGTEXTFMT_PLACEHOLDER_UTF8 = 4,
  FGTEXTFMT_PLACEHOLDER_UTF16 = 5,
  FGTEXTFMT_PLACEHOLDER_UTF32 = 6,
  FGTEXTFMT_DYNAMIC_UTF8 = 8,
  FGTEXTFMT_DYNAMIC_UTF16 = 9,
  FGTEXTFMT_DYNAMIC_UTF32 = 10,
  FGTEXTFMT_MASK = 15,
};

enum FGSETSTYLE
{
  FGSETSTYLE_NAME = 0,
  FGSETSTYLE_INDEX,
  FGSETSTYLE_POINTER,
};

enum FGSETCOLOR
{
  FGSETCOLOR_MAIN = 0,
  FGSETCOLOR_PLACEHOLDER,
  FGSETCOLOR_CURSOR,
  FGSETCOLOR_SELECT,
  FGSETCOLOR_HOVER,
  FGSETCOLOR_DRAG,
  FGSETCOLOR_EDGE,
  FGSETCOLOR_DIVIDER,
  FGSETCOLOR_COLUMNDIVIDER,
  FGSETCOLOR_ROWEVEN,
};

enum FGDIM
{
  FGDIM_MIN = 0,
  FGDIM_MAX,
  FGDIM_FIXED,
};

enum FGCHECKED
{
  FGCHECKED_NONE = 0,
  FGCHECKED_CHECKED = 1,
  FGCHECKED_INDETERMINATE = 2,
};

enum FGUNIT
{
  FGUNIT_DP = 0, // By default, all feathergui units are in DPI-scaled units.
  FGUNIT_SP = 1,
  FGUNIT_EM = 2,
  FGUNIT_PX = 3, // Corresponds to exactly one pixel regardless of DPI
  FGUNIT_LEFT_DP = 0,
  FGUNIT_LEFT_SP = (1 << 0),
  FGUNIT_LEFT_EM = (2 << 0),
  FGUNIT_LEFT_PX = (3 << 0),
  FGUNIT_LEFT_MASK = (0b11 << 0),
  FGUNIT_LEFT = 0,
  FGUNIT_TOP_DP = 0,
  FGUNIT_TOP_SP = (1 << 2),
  FGUNIT_TOP_EM = (2 << 2),
  FGUNIT_TOP_PX = (3 << 2),
  FGUNIT_TOP_MASK = (0b11 << 2),
  FGUNIT_TOP = 2,
  FGUNIT_RIGHT_DP = 0,
  FGUNIT_RIGHT_SP = (1 << 4),
  FGUNIT_RIGHT_EM = (2 << 4),
  FGUNIT_RIGHT_PX = (3 << 4),
  FGUNIT_RIGHT_MASK = (0b11 << 4),
  FGUNIT_RIGHT = 4,
  FGUNIT_BOTTOM_DP = 0,
  FGUNIT_BOTTOM_SP = (1 << 6),
  FGUNIT_BOTTOM_EM = (2 << 6),
  FGUNIT_BOTTOM_PX = (3 << 6),
  FGUNIT_BOTTOM_MASK = (0b11 << 6),
  FGUNIT_BOTTOM = 6,
  FGUNIT_X_DP = 0,
  FGUNIT_X_SP = (1 << 8),
  FGUNIT_X_EM = (2 << 8),
  FGUNIT_X_PX = (3 << 8),
  FGUNIT_X_MASK = (0b11 << 8),
  FGUNIT_X = 8,
  FGUNIT_Y_DP = 0,
  FGUNIT_Y_SP = (1 << 10),
  FGUNIT_Y_EM = (2 << 10),
  FGUNIT_Y_PX = (3 << 10),
  FGUNIT_Y_MASK = (0b11 << 10),
  FGUNIT_Y = 10,
  FGUNIT_RIGHT_WIDTH = (1 << 12),
  FGUNIT_BOTTOM_HEIGHT = (1 << 13),
};

enum FGITEM
{
  FGITEM_DEFAULT = 0,
  FGITEM_TEXT, 
  FGITEM_ELEMENT,
  FGITEM_ROW,
  FGITEM_COLUMN,
  FGITEM_LOCATION,
  FGITEM_COUNT,
};

enum FGVALUE
{
  FGVALUE_UNKNOWN = 0,
  FGVALUE_INT64 = 1,
  FGVALUE_FLOAT = 2,
  FGVALUE_POINTER = 3,
};

enum FGSETPARENT
{
  FGSETPARENT_DEFAULT = 0,
  FGSETPARENT_FIRST = 1,
  FGSETPARENT_LAST = 2,
};

enum FG_MSGTYPE
{
  FG_UNKNOWN = 0,
  FG_CONSTRUCT = 1,
  FG_DESTROY, // Notification when the element is being destroyed. Sending this message will not destroy the element or call the destructor.
  FG_CLONE,
  FG_MOVE, // Passed when any change is made to an element. 1: propagating up, 2: x-axis resize, 4: y-axis resize, 8: x-axis move, 16: y-axis move, 32: x center move, 64: y center move, 128: rotation change, 256: padding change
  FG_SETALPHA, // Used so an entire widget can be made to fade in or out. (Support is not guaranteed)
  FG_SETAREA,
  FG_SETTRANSFORM,
  FG_SETFLAG, // Send in the flag in the first int arg, then set it to on or off (1 or 0) in the second argument
  FG_SETFLAGS, // Sets all the flags to the first int arg.
  FG_SETMARGIN,
  FG_SETPADDING,
  FG_SETPARENT, // Adds this element to the parent in the first argument, inserting it after the child in the second argument, or before the root if the second argument is NULL.
  FG_ADDCHILD, // Pass an FG_Msg with this type and set the other pointer to the child that should be added.
  FG_REMOVECHILD, // Verifies child's parent is this, then sets the child's parent to NULL.
  FG_REORDERCHILD, // This is used ONLY when a child cannot be removed and re-added to a parent and must be directly re-ordered within them.
  FG_PARENTCHANGE, // This is ONLY used to notify a child that it's parent has changed, because FG_SETPARENT might not ever be called if FG_ADDCHILD is called directly.
  FG_LAYOUTCHANGE, 
  FG_LAYOUTFUNCTION,
  FG_LAYOUTLOAD, // Loads a layout passed in the first pointer with an optional custom class name resolution function passed into the second pointer of type fgElement* (*)(const char*, fgTransform*, fgFlag)
  FG_DRAGOVER, // Sent to any element a dragged element is hovering over so it can set the cursor icon.
  FG_DROP, // Sent when an element is "dropped" on another element. Whether or not this does anything is up to the control.
  FG_DRAW,
  FG_INJECT,
  FG_SETSKIN, // Sets the skin. If NULL, uses GETSKIN to resolve the skin.
  FG_GETSKIN,
  FG_SETSTYLE, // Sets the style. -1 causes it to call GETSTYLE to try and resolve the style index.
  FG_GETSTYLE,
  FG_GETCLASSNAME, // Returns a unique string identifier for the class
  FG_GETDPI,
  FG_SETDPI,
  FG_SETUSERDATA, // Stores the pointer-sized first argument in a hash using the second argument as a string key. If the key is null, stores it in userdata.
  FG_GETUSERDATA, 
  FG_SETDIM,
  FG_GETDIM,
  FG_SETSCALING,
  FG_GETSCALING,
  // fgControl
  FG_MOUSEDOWN,
  FG_MOUSEDBLCLICK,
  FG_MOUSEUP, 
  FG_MOUSEON,
  FG_MOUSEOFF,
  FG_MOUSEMOVE,
  FG_MOUSESCROLL, 
  FG_TOUCHBEGIN,
  FG_TOUCHEND,
  FG_TOUCHMOVE,
  FG_KEYUP,
  FG_KEYDOWN,
  FG_KEYCHAR, //Passed in conjunction with keydown/up to differentiate a typed character from other keys.
  FG_JOYBUTTONDOWN,
  FG_JOYBUTTONUP,
  FG_JOYAXIS,
  FG_GOTFOCUS,
  FG_LOSTFOCUS,
  FG_SETNAME, // Sets the unique name for this object for skin collection mapping. Can be null.
  FG_GETNAME, // May return a unique string for this object, or will return NULL.
  FG_SETCONTEXTMENU,
  FG_GETCONTEXTMENU,
  // fgButton and others
  FG_NEUTRAL, // Sent when a button or other hover-enabled control switches to it's neutral state
  FG_HOVER, // Sent when a hover-enabled control switches to its hover state
  FG_ACTIVE, // Sent when a hover-enabled control switches to its active state
  FG_ACTION, // Sent when a hover-enabled control recieves a valid click event (a MOUSEUP inside the control while it has focus)
  // fgList, fgMenu, etc.
  FG_GETITEM,
  FG_ADDITEM, // Used for anything involving items (menus, lists, etc)
  FG_REMOVEITEM,
  FG_SETITEM,
  FG_GETSELECTEDITEM, // Used to get the selected item (or items, or text) in a control.
  // fgCheckbox, fgRadiobutton, fgProgressbar, etc.
  FG_GETVALUE, // Gets the on/off state of a checkbox or the current progress on a progress bar
  FG_SETVALUE, // Sets the on/off state or progress
  FG_GETRANGE, 
  FG_SETRANGE,
  // fgResource or fgText
  FG_SETASSET,
  FG_SETUV,
  FG_SETCOLOR,
  FG_SETOUTLINE,
  FG_SETFONT,
  FG_SETLINEHEIGHT,
  FG_SETLETTERSPACING,
  FG_SETTEXT,
  FG_GETASSET,
  FG_GETUV,
  FG_GETCOLOR,
  FG_GETOUTLINE,
  FG_GETFONT,
  FG_GETLINEHEIGHT,
  FG_GETLETTERSPACING,
  FG_GETTEXT,
  FG_CUSTOMEVENT
};

enum FG_KEYS
{
  FG_KEY_NULL = 0, //because its possible to have _lastkey Set to this
  FG_KEY_LBUTTON = 0x01,
  FG_KEY_RBUTTON = 0x02,
  FG_KEY_CANCEL = 0x03,
  FG_KEY_MBUTTON = 0x04,    /* NOT contiguous with L & RBUTTON */
  FG_KEY_XBUTTON1 = 0x05,    /* NOT contiguous with L & RBUTTON */
  FG_KEY_XBUTTON2 = 0x06,    /* NOT contiguous with L & RBUTTON */
  FG_KEY_BACK = 0x08,
  FG_KEY_TAB = 0x09,
  FG_KEY_CLEAR = 0x0C,
  FG_KEY_RETURN = 0x0D,
  FG_KEY_SHIFT = 0x10,
  FG_KEY_CONTROL = 0x11,
  FG_KEY_MENU = 0x12,
  FG_KEY_PAUSE = 0x13,
  FG_KEY_CAPITAL = 0x14,
  FG_KEY_KANA = 0x15,
  FG_KEY_JUNJA = 0x17,
  FG_KEY_FINAL = 0x18,
  FG_KEY_KANJI = 0x19,
  FG_KEY_ESCAPE = 0x1B,
  FG_KEY_CONVERT = 0x1C,
  FG_KEY_NONCONVERT = 0x1D,
  FG_KEY_ACCEPT = 0x1E,
  FG_KEY_MODECHANGE = 0x1F,
  FG_KEY_SPACE = 0x20,
  FG_KEY_PAGEUP = 0x21, // named PRIOR in windows virtual keys
  FG_KEY_PAGEDOWN = 0x22, // named NEXT in windows virtual keys
  FG_KEY_END = 0x23,
  FG_KEY_HOME = 0x24,
  FG_KEY_LEFT = 0x25,
  FG_KEY_UP = 0x26,
  FG_KEY_RIGHT = 0x27,
  FG_KEY_DOWN = 0x28,
  FG_KEY_SELECT = 0x29,
  FG_KEY_PRINT = 0x2A,
  //FG_KEY_EXECUTE      = 0x2B,
  FG_KEY_SNAPSHOT = 0x2C,
  FG_KEY_INSERT = 0x2D,
  FG_KEY_DELETE = 0x2E,
  FG_KEY_HELP = 0x2F,
  FG_KEY_0 = 0x30,
  FG_KEY_1 = 0x31,
  FG_KEY_2 = 0x32,
  FG_KEY_3 = 0x33,
  FG_KEY_4 = 0x34,
  FG_KEY_5 = 0x35,
  FG_KEY_6 = 0x36,
  FG_KEY_7 = 0x37,
  FG_KEY_8 = 0x38,
  FG_KEY_9 = 0x39,
  FG_KEY_A = 0x41,
  FG_KEY_B = 0x42,
  FG_KEY_C = 0x43,
  FG_KEY_D = 0x44,
  FG_KEY_E = 0x45,
  FG_KEY_F = 0x46,
  FG_KEY_G = 0x47,
  FG_KEY_H = 0x48,
  FG_KEY_I = 0x49,
  FG_KEY_J = 0x4A,
  FG_KEY_K = 0x4B,
  FG_KEY_L = 0x4C,
  FG_KEY_M = 0x4D,
  FG_KEY_N = 0x4E,
  FG_KEY_O = 0x4F,
  FG_KEY_P = 0x50,
  FG_KEY_Q = 0x51,
  FG_KEY_R = 0x52,
  FG_KEY_S = 0x53,
  FG_KEY_T = 0x54,
  FG_KEY_U = 0x55,
  FG_KEY_V = 0x56,
  FG_KEY_W = 0x57,
  FG_KEY_X = 0x58,
  FG_KEY_Y = 0x59,
  FG_KEY_Z = 0x5A,
  FG_KEY_LWIN = 0x5B,
  FG_KEY_RWIN = 0x5C,
  FG_KEY_APPS = 0x5D,
  FG_KEY_SLEEP = 0x5F,
  FG_KEY_NUMPAD0 = 0x60,
  FG_KEY_NUMPAD1 = 0x61,
  FG_KEY_NUMPAD2 = 0x62,
  FG_KEY_NUMPAD3 = 0x63,
  FG_KEY_NUMPAD4 = 0x64,
  FG_KEY_NUMPAD5 = 0x65,
  FG_KEY_NUMPAD6 = 0x66,
  FG_KEY_NUMPAD7 = 0x67,
  FG_KEY_NUMPAD8 = 0x68,
  FG_KEY_NUMPAD9 = 0x69,
  FG_KEY_MULTIPLY = 0x6A,
  FG_KEY_ADD = 0x6B,
  FG_KEY_SEPARATOR = 0x6C,
  FG_KEY_SUBTRACT = 0x6D,
  FG_KEY_DECIMAL = 0x6E,
  FG_KEY_DIVIDE = 0x6F,
  FG_KEY_F1 = 0x70,
  FG_KEY_F2 = 0x71,
  FG_KEY_F3 = 0x72,
  FG_KEY_F4 = 0x73,
  FG_KEY_F5 = 0x74,
  FG_KEY_F6 = 0x75,
  FG_KEY_F7 = 0x76,
  FG_KEY_F8 = 0x77,
  FG_KEY_F9 = 0x78,
  FG_KEY_F10 = 0x79,
  FG_KEY_F11 = 0x7A,
  FG_KEY_F12 = 0x7B,
  FG_KEY_F13 = 0x7C,
  FG_KEY_F14 = 0x7D,
  FG_KEY_F15 = 0x7E,
  FG_KEY_F16 = 0x7F,
  FG_KEY_F17 = 0x80,
  FG_KEY_F18 = 0x81,
  FG_KEY_F19 = 0x82,
  FG_KEY_F20 = 0x83,
  FG_KEY_F21 = 0x84,
  FG_KEY_F22 = 0x85,
  FG_KEY_F23 = 0x86,
  FG_KEY_F24 = 0x87,
  FG_KEY_NUMLOCK = 0x90,
  FG_KEY_SCROLL = 0x91,
  FG_KEY_OEM_NEC_EQUAL = 0x92,   // '=' key on numpad
  FG_KEY_LSHIFT = 0xA0,
  FG_KEY_RSHIFT = 0xA1,
  FG_KEY_LCONTROL = 0xA2,
  FG_KEY_RCONTROL = 0xA3,
  FG_KEY_LMENU = 0xA4,
  FG_KEY_RMENU = 0xA5,
  FG_KEY_OEM_1 = 0xBA,   // ';:' for US
  FG_KEY_OEM_PLUS = 0xBB,   // '+' any country
  FG_KEY_OEM_COMMA = 0xBC,   // ',' any country
  FG_KEY_OEM_MINUS = 0xBD,   // '-' any country
  FG_KEY_OEM_PERIOD = 0xBE,   // '.' any country
  FG_KEY_OEM_2 = 0xBF,   // '/?' for US
  FG_KEY_OEM_3 = 0xC0,   // '`virtual ~' for US
  FG_KEY_OEM_4 = 0xDB,  //  '[{' for US
  FG_KEY_OEM_5 = 0xDC,  //  '\|' for US
  FG_KEY_OEM_6 = 0xDD,  //  ']}' for US
  FG_KEY_OEM_7 = 0xDE,  //  ''"' for US
  FG_KEY_OEM_8 = 0xDF
};

//The joystick button enumerator
enum FG_JOYBUTTONS
{
  FG_JOYBUTTON1 = 0,
  FG_JOYBUTTON2 = 1,
  FG_JOYBUTTON3 = 2,
  FG_JOYBUTTON4 = 3,
  FG_JOYBUTTON5 = 4,
  FG_JOYBUTTON6 = 5,
  FG_JOYBUTTON7 = 6,
  FG_JOYBUTTON8 = 7,
  FG_JOYBUTTON9 = 8,
  FG_JOYBUTTON10 = 9,
  FG_JOYBUTTON11 = 10,
  FG_JOYBUTTON12 = 11,
  FG_JOYBUTTON13 = 12,
  FG_JOYBUTTON14 = 13,
  FG_JOYBUTTON15 = 14,
  FG_JOYBUTTON16 = 15,
  FG_JOYBUTTON17 = 16,
  FG_JOYBUTTON18 = 17,
  FG_JOYBUTTON19 = 18,
  FG_JOYBUTTON20 = 19,
  FG_JOYBUTTON21 = 20,
  FG_JOYBUTTON22 = 21,
  FG_JOYBUTTON23 = 22,
  FG_JOYBUTTON24 = 23,
  FG_JOYBUTTON25 = 24,
  FG_JOYBUTTON26 = 25,
  FG_JOYBUTTON27 = 26,
  FG_JOYBUTTON28 = 27,
  FG_JOYBUTTON29 = 28,
  FG_JOYBUTTON30 = 29,
  FG_JOYBUTTON31 = 30,
  FG_JOYBUTTON32 = 31,
  FG_JOYSTICK_ID1 = 0x0000,
  FG_JOYSTICK_ID2 = 0x0100,
  FG_JOYSTICK_ID3 = 0x0200,
  FG_JOYSTICK_ID4 = 0x0300,
  FG_JOYSTICK_ID5 = 0x0400,
  FG_JOYSTICK_ID6 = 0x0500,
  FG_JOYSTICK_ID7 = 0x0600,
  FG_JOYSTICK_ID8 = 0x0700,
  FG_JOYSTICK_ID9 = 0x0800,
  FG_JOYSTICK_ID10 = 0x0900,
  FG_JOYSTICK_ID12 = 0x0A00,
  FG_JOYSTICK_ID13 = 0x0B00,
  FG_JOYSTICK_ID14 = 0x0C00,
  FG_JOYSTICK_ID15 = 0x0D00,
  FG_JOYSTICK_ID16 = 0x0E00,
  FG_JOYSTICK_INVALID = 0xFFFF,
  FG_JOYAXIS_X = 0,
  FG_JOYAXIS_Y = 1,
  FG_JOYAXIS_Z = 2,
  FG_JOYAXIS_R = 3,
  FG_JOYAXIS_U = 4,
  FG_JOYAXIS_V = 5
};

enum FG_MOUSEBUTTON // Used in FG_Msg.button and FG_Msg.allbtn
{
  FG_MOUSELBUTTON=1,
  FG_MOUSERBUTTON=2,
  FG_MOUSEMBUTTON=4,
  FG_MOUSEXBUTTON1=8,
  FG_MOUSEXBUTTON2=16,
};

enum FG_CURSOR
{
  FGCURSOR_NONE = 0,
  FGCURSOR_ARROW,
  FGCURSOR_IBEAM,
  FGCURSOR_CROSS,
  FGCURSOR_WAIT,
  FGCURSOR_HAND,
  FGCURSOR_RESIZENS, // up down
  FGCURSOR_RESIZEWE, // left right
  FGCURSOR_RESIZENWSE, // upper left to lower right
  FGCURSOR_RESIZENESW , // upper right to lower left
  FGCURSOR_RESIZEALL,
  FGCURSOR_NO,
  FGCURSOR_HELP, // contextual menu cursor on mac
  FGCURSOR_DRAG, // Mac has a default drag cursor, but windows doesn't
  FGCURSOR_CUSTOM,
  FGCURSOR_OVERRIDE = 0x40,
};

enum FG_CLIPBOARD
{
  FGCLIPBOARD_NONE = 0,
  FGCLIPBOARD_TEXT,
  FGCLIPBOARD_WAVE,
  FGCLIPBOARD_BITMAP,
  FGCLIPBOARD_FILE,
  FGCLIPBOARD_ELEMENT,
  FGCLIPBOARD_CUSTOM,
  FGCLIPBOARD_ALL,
};

enum FG_MOUSEFLAGS {
  FGMOUSE_INSIDE = 1, // True if the current button was pushed down while inside this control
  FGMOUSE_HOVER = 2,  // True only if the mouse is currently inside the control
  FGMOUSE_DRAG = 4, // True if the last mouse event recieved was dragging something
  FGMOUSE_SEND_MOUSEMOVE = 32, // Used on the root only, schedules another mousemove event to be sent to compensate for a moving control.
};

typedef struct _FG_MOUSESTATE
{
  float x, y; // Last known position of the mouse for this control
  unsigned char buttons; // Last known configuration of mouse buttons recieved by this control
  unsigned char state;
} fgMouseState;

struct _FG_ELEMENT;

// General message structure which contains the message type and then various kinds of information depending on the type.
typedef struct _FG_MSG {
  unsigned short type;
  unsigned short subtype;
  union {
    struct { float x; float y; // Mouse and touch events
      union { 
        struct { unsigned char button; unsigned char allbtn; }; 
        struct { short scrolldelta; short scrollhdelta; }; // MOUSESCROLL
        short touchindex; // Touch events
      };
    }; 
    struct {  // Keys
        int keychar; //Only used by KEYCHAR, represents a utf32 character
        unsigned short keyraw; // In some cases (e.g. games) an application may want to pass around the raw hardware scancode of a key.
        unsigned char keycode; //only used by KEYDOWN/KEYUP, represents an actual keycode in FG_KEYS, not a character
        char sigkeys; // 1: shift, 2: ctrl, 4: alt, 8: held
    };
    struct { float joyvalue; short joyaxis; }; // JOYAXIS
    struct { char joydown; short joybutton; }; // JOYBUTTON
    struct {
      union { void* p; ptrdiff_t i; size_t u; FABS f; struct _FG_ELEMENT* e; };
      union { void* p2; ptrdiff_t i2; size_t u2; FABS f2; struct _FG_ELEMENT* e2; };
    };
  };

#ifdef __cplusplus
  inline bool IsPressed() const { return (button&allbtn) != 0; }
  inline bool IsShiftDown() const { return (1 & sigkeys) != 0; }
  inline bool IsCtrlDown() const { return (2 & sigkeys) != 0; }
  inline bool IsAltDown() const { return (4 & sigkeys) != 0; }
  inline bool IsHeld() const { return (8 & sigkeys) != 0; }
#endif
} FG_Msg;

FG_EXTERN AbsVec ResolveVec(const CVec* v, const AbsRect* last);
FG_EXTERN inline char CompareMargins(const AbsRect* l, const AbsRect* r); // Returns 0 if both are the same or a difference bitset otherwise.
FG_EXTERN inline char CompareCRects(const CRect* l, const CRect* r); // Returns 0 if both are the same or a difference bitset otherwise.
FG_EXTERN inline char CompareTransforms(const fgTransform* l, const fgTransform* r);
FG_EXTERN inline void MoveCRect(FABS x, FABS y, CRect* r);
FG_EXTERN inline char HitAbsRect(const AbsRect* r, FABS x, FABS y);
//FG_EXTERN void ToIntAbsRect(const AbsRect* r, int target[static 4]);
FG_EXTERN inline void ToIntAbsRect(const AbsRect* r, int target[4]);
FG_EXTERN inline void ToLongAbsRect(const AbsRect* r, long target[4]);
FG_EXTERN inline char MsgHitAbsRect(const FG_Msg* msg, const AbsRect* r);
FG_EXTERN const char* fgCopyText(const char* text, const char* file, size_t line);
FG_EXTERN void fgFreeText(const char* text, const char* file, size_t line);
FG_EXTERN inline void fgUpdateMouseState(fgMouseState* state, const FG_Msg* msg);
FG_EXTERN inline char fgRectIntersect(const AbsRect* l, const AbsRect* r); // Returns 1 if the rectangles intersect, or 0 otherwise
FG_EXTERN inline void fgRectIntersection(const AbsRect* BSS_RESTRICT l, const AbsRect* BSS_RESTRICT r, AbsRect* out);
FG_EXTERN inline void fgScaleRectDPI(AbsRect* rect, int dpix, int dpiy);
FG_EXTERN size_t fgUTF32toUTF16(const int*BSS_RESTRICT input, ptrdiff_t srclen, wchar_t*BSS_RESTRICT output, size_t buflen);
FG_EXTERN size_t fgUTF8toUTF16(const char*BSS_RESTRICT input, ptrdiff_t srclen, wchar_t*BSS_RESTRICT output, size_t buflen);
FG_EXTERN size_t fgUTF16toUTF8(const wchar_t*BSS_RESTRICT input, ptrdiff_t srclen, char*BSS_RESTRICT output, size_t buflen);
FG_EXTERN size_t fgUTF8toUTF32(const char*BSS_RESTRICT input, ptrdiff_t srclen, int*BSS_RESTRICT output, size_t buflen);
FG_EXTERN size_t fgUTF32toUTF8(const int*BSS_RESTRICT input, ptrdiff_t srclen, char*BSS_RESTRICT output, size_t buflen);
FG_EXTERN size_t fgUTF16toUTF32(const wchar_t*BSS_RESTRICT input, ptrdiff_t srclen, int*BSS_RESTRICT output, size_t buflen);

typedef struct _FG_DRAW_AUX_DATA {
  size_t fgSZ;
  fgIntVec dpi;
  AbsVec scale;
  AbsVec scalecenter;
} fgDrawAuxData;

#ifdef  __cplusplus
}

template<int I, typename T>
void BSS_FORCEINLINE fgSendMsgArg(FG_Msg&, T);

template<> void BSS_FORCEINLINE fgSendMsgArg<1, const void*>(FG_Msg& msg, const void* p) { msg.p = const_cast<void*>(p); }
template<> void BSS_FORCEINLINE fgSendMsgArg<1, void*>(FG_Msg& msg, void* p) { msg.p = p; }
template<> void BSS_FORCEINLINE fgSendMsgArg<1, float>(FG_Msg& msg, float p) { msg.f = p; }
template<> void BSS_FORCEINLINE fgSendMsgArg<1, ptrdiff_t>(FG_Msg& msg, ptrdiff_t p) { msg.i = p; }
template<> void BSS_FORCEINLINE fgSendMsgArg<1, size_t>(FG_Msg& msg, size_t p) { msg.u = p; }
template<> void BSS_FORCEINLINE fgSendMsgArg<1, const struct _FG_ELEMENT*>(FG_Msg& msg, const struct _FG_ELEMENT* p) { msg.p = const_cast<struct _FG_ELEMENT*>(p); }
template<> void BSS_FORCEINLINE fgSendMsgArg<1, struct _FG_ELEMENT*>(FG_Msg& msg, struct _FG_ELEMENT* p) { msg.p = p; }

template<> void BSS_FORCEINLINE fgSendMsgArg<2, const void*>(FG_Msg& msg, const void* p) { msg.p2 = const_cast<void*>(p); }
template<> void BSS_FORCEINLINE fgSendMsgArg<2, void*>(FG_Msg& msg, void* p) { msg.p2 = p; }
template<> void BSS_FORCEINLINE fgSendMsgArg<2, float>(FG_Msg& msg, float p) { msg.f2 = p; }
template<> void BSS_FORCEINLINE fgSendMsgArg<2, ptrdiff_t>(FG_Msg& msg, ptrdiff_t p) { msg.i2 = p; }
template<> void BSS_FORCEINLINE fgSendMsgArg<2, size_t>(FG_Msg& msg, size_t p) { msg.u2 = p; }
template<> void BSS_FORCEINLINE fgSendMsgArg<2, const struct _FG_ELEMENT*>(FG_Msg& msg, const struct _FG_ELEMENT* p) { msg.p2 = const_cast<struct _FG_ELEMENT*>(p); }
template<> void BSS_FORCEINLINE fgSendMsgArg<2, struct _FG_ELEMENT*>(FG_Msg& msg, struct _FG_ELEMENT* p) { msg.p2 = p; }

template<int I, typename... Args> struct fgSendMsgCall;
template<int I, typename Arg, typename... Args>
struct fgSendMsgCall<I, Arg, Args...> {
  BSS_FORCEINLINE static void F(FG_Msg& msg, Arg arg, Args... args) {
    fgSendMsgArg<I, Arg>(msg, arg);
    fgSendMsgCall<I + 1, Args...>::F(msg, args...);
  }
};
template<int I>
struct fgSendMsgCall<I> { BSS_FORCEINLINE static void F(FG_Msg& msg) {} };
#endif

#endif
