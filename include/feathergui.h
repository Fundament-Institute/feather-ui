/* Feather - Lightweight GUI Abstraction Layer
   Copyright ©2015 Black Sphere Studios

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

#include <assert.h>
#include <string.h> // memcpy,memset
#include <malloc.h> 
#include "bss_compiler.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef float FREL; // Change this to double for double precision (why on earth you would need that for a GUI, however, is beyond me)
typedef float FABS; // We seperate both of these types because then you don't have to guess if a float is relative or absolute
typedef unsigned int FG_UINT; 
typedef unsigned int fgFlag;

#define FGUI_VERSION_MAJOR 0
#define FGUI_VERSION_MINOR 1
#define FGUI_VERSION_REVISION 0
#define FG_FASTCALL BSS_COMPILER_FASTCALL
#define FG_EXTERN extern BSS_COMPILER_DLLEXPORT

// A unified coordinate specifies things in terms of absolute and relative positions.
typedef struct {
  FABS abs; // Absolute coordinates are added to relative coordinates, which specify a point from a linear interpolation of the parent's dimensions
  FREL rel;
} Coord;

#define T_SETBIT(w,b,f) (((w) & (~(b))) | ((-(char)f) & (b)))
#define MAKE_VEC(_T,_N) typedef struct { _T x; _T y; } _N

MAKE_VEC(FREL,RelVec);
MAKE_VEC(FABS,AbsVec);
// A coordinate vector specifies a point by unified coordinates
MAKE_VEC(Coord,CVec);

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

static BSS_FORCEINLINE FG_UINT FG_FASTCALL fbnext(FG_UINT in)
{
  return in + 1 + (in>>1) + (in>>3) - (in>>7);
}
static BSS_FORCEINLINE FABS FG_FASTCALL lerp(FABS a, FABS b, FREL amt)
{
	return a+((FABS)((b-a)*amt));
}

typedef struct __VECTOR {
  void* p;
  FG_UINT s; // This is the total size of the array in BYTES.
  FG_UINT l; // This is how much of the array is being used in ELEMENTS.
} fgVector;

FG_EXTERN void FG_FASTCALL fgVector_Init(fgVector* self); // Zeros everything
FG_EXTERN void FG_FASTCALL fgVector_Destroy(fgVector* self);
FG_EXTERN void FG_FASTCALL fgVector_SetSize(fgVector* self, FG_UINT length, FG_UINT size); // Grows or shrinks the array. self.l is shrunk if necessary.
FG_EXTERN void FG_FASTCALL fgVector_CheckSize(fgVector* self, FG_UINT size); // Ensures the vector is large enough to hold at least one more item.
FG_EXTERN void FG_FASTCALL fgVector_Remove(fgVector* self, FG_UINT index, FG_UINT size); // Removes element at the given index.

#define fgVector_Add(self, item, TYPE) fgVector_CheckSize(&self,sizeof(TYPE)); ((TYPE*)self.p)[self.l++]=item;
#define fgVector_Insert(self, item, index, TYPE) fgVector_CheckSize(&self,sizeof(TYPE)); if(self.l-index>0) { memmove(((TYPE*)self.p)+(index+1),((TYPE*)self.p)+index,((self.l++)-index)*sizeof(TYPE)); } ((TYPE*)self.p)[index]=item;
#define fgVector_Get(self, index, TYPE) ((TYPE*)(self).p)[index]
#define fgVector_GetP(self, index, TYPE) (((TYPE*)(self).p)+(index))

typedef struct {
  CRect area;
  FABS rotation;
  CVec center;
} fgElement;

extern const fgElement fgElement_DEFAULT;
extern const fgElement fgElement_CENTER;

enum FG_MSGTYPE
{
  // fgChild
  FG_MOVE, // Passed when any change is made to an element. 1: propagating up, 2: x-axis resize, 4: y-axis resize, 8: x-axis move, 16: y-axis move, 32: zorder change
  FG_SETALPHA, // Used so an entire widget can be made to fade in or out. (Support is not guaranteed)
  FG_SETAREA,
  FG_SETELEMENT,
  FG_SETFLAG, // Send in the flag in the first int arg, then set it to on or off (1 or 0) in the second argument
  FG_SETFLAGS, // Sets all the flags to the first int arg.
  FG_SETMARGIN,
  FG_SETPADDING,
  FG_SETORDER, // Sets the order of a window in the otherint argument
  FG_SETPARENT,
  FG_ADDCHILD, // Pass an FG_Msg with this type and set the other pointer to the child that should be added.
  FG_REMOVECHILD, // Verifies child's parent is this, then sets the child's parent to NULL.
  FG_LAYOUTRESIZE, // Called when the element is resized
  FG_LAYOUTADD, // Called when any child is added that needs to have the layout applied to it.
  FG_LAYOUTREMOVE, // Called when any child is removed that needs to have the layout applied to it.
  FG_LAYOUTMOVE, // Called when any child is moved so the layout can adjust as necessary.
  FG_LAYOUTREORDER, // Called when any child is reordered
  FG_LAYOUTLOAD, // Loads a layout passed in the first pointer with an optional custom class name resolution function passed into the second pointer of type fgChild* (*)(const char*, fgElement*, fgFlag)
  FG_DRAG, // Sent to initiate a drag&drop
  FG_DRAGGING, // Sent to any element a dragged element is hovering over so it can set the cursor icon.
  FG_DROP, // Sent when an element is "dropped" on another element. Whether or not this does anything is up to the control.
  FG_DRAW,
  FG_CLONE, // Clones the fgChild
  FG_SETSKIN, // Sets the skin. If NULL, uses GETSKIN to resolve the skin.
  FG_GETSKIN,
  FG_SETSTYLE, // Sets the style. -1 causes it to call GETSTYLE to try and resolve the style index.
  FG_GETSTYLE,
  FG_GETCLASSNAME, // Returns a unique string identifier for the class
  // fgWindow
  FG_MOUSEDOWN,
  //FG_MOUSEDBLCLICK,
  FG_MOUSEUP, 
  FG_MOUSEON,
  FG_MOUSEOFF,
  FG_MOUSEMOVE,
  FG_MOUSESCROLL, 
  FG_MOUSELEAVE, // Sent when the mouse leaves the root area, forces a MOUSEOFF message on current hover window.
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
  // fgButton and others
  FG_NUETRAL, // Sent when a button or other hover-enabled control switches to it's nuetral state
  FG_HOVER, // Sent when a hover-enabled control switches to its hover state
  FG_ACTIVE, // Sent when a hover-enabled control switches to its active state
  FG_ACTION, // Sent when a hover-enabled control recieves a valid click event (a MOUSEUP inside the control while it has focus)
  // fgList, fgMenu, etc.
  FG_SETCOLUMNS, // If the second pointer is 0, the number of columns is set to the first int. Otherwise, the first int is treated as a column index number, which has it's width set to the Coord pointer to by the second pointer
  FG_GETITEM,
  FG_GETROW,
  FG_ADDITEM, // Used for anything involving items (menus, lists, etc)
  FG_REMOVEITEM,
  FG_GETSELECTEDITEM, // Used to get the selected item (or items, or text) in a control.
  // fgCheckbox, fgRadioButton, fgProgressbar, etc.
  FG_GETSTATE, // Gets the on/off state of a checkbox or the current progress on a progress bar
  FG_SETSTATE, // Sets the on/off state or progress
  // fgResource or fgText
  FG_SETRESOURCE,
  FG_SETUV,
  FG_SETCOLOR,
  FG_SETFONT,
  FG_SETFONTCOLOR, // split from SETCOLOR so it can be propagated down seperately from setting image colors
  FG_SETTEXT,
  FG_GETRESOURCE,
  FG_GETUV,
  FG_GETCOLOR,
  FG_GETFONT,
  FG_GETFONTCOLOR,
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
  FG_KEY_PRIOR = 0x21,
  FG_KEY_NEXT = 0x22,
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

enum FG_MOUSEBUTTON // Used in FG_Msg.button and FG_Msg.allbtn
{
  FG_MOUSELBUTTON=1,
  FG_MOUSERBUTTON=2,
  FG_MOUSEMBUTTON=4,
  FG_MOUSEXBUTTON1=8,
  FG_MOUSEXBUTTON2=16,
};

// General message structure which contains the message type and then various kinds of information depending on the type.
typedef struct __FG_MSG {
  union {
    struct { int x; int y; unsigned char button; unsigned char allbtn; }; // Mouse events
    struct { short scrolldelta; }; // Mouse scroll
    struct {  // Keys
        int keychar; //Only used by KEYCHAR, represents a utf32 character
        unsigned char keycode; //only used by KEYDOWN/KEYUP, represents an actual keycode, not a character
        char keydown;
        char sigkeys; // 1: shift, 2: ctrl, 4: alt, 8: held
    };
    struct { float joyvalue; short joyaxis; }; // JOYAXIS
    struct { char joydown; short joybutton; }; // JOYBUTTON
    struct { void* other; size_t otheraux; }; // Used by any generic messages (FG_SETPARENT, etc.)
    struct { void* other1; void* other2; }; // Used by anything requiring 2 pointers, possibly for a return value.
    struct { ptrdiff_t otherint; ptrdiff_t otherintaux; }; // Used by any generic message that want an int (FG_SETORDER, etc.)
  };
  unsigned char type;
} FG_Msg;


FG_EXTERN AbsVec FG_FASTCALL ResolveVec(const CVec* v, const AbsRect* last);
FG_EXTERN char FG_FASTCALL CompareAbsRects(const AbsRect* l, const AbsRect* r); // Returns 0 if both are the same or a difference bitset otherwise.
FG_EXTERN char FG_FASTCALL CompareCRects(const CRect* l, const CRect* r); // Returns 0 if both are the same or a difference bitset otherwise.
FG_EXTERN char FG_FASTCALL CompareElements(const fgElement* l, const fgElement* r);
FG_EXTERN void FG_FASTCALL MoveCRect(AbsVec v, CRect* r);
FG_EXTERN void FG_FASTCALL MoveCRectInv(AbsVec v, CRect* r);
FG_EXTERN char FG_FASTCALL HitAbsRect(const AbsRect* r, FABS x, FABS y);
//FG_EXTERN void FG_FASTCALL ToIntAbsRect(const AbsRect* r, int target[static 4]);
FG_EXTERN void FG_FASTCALL ToIntAbsRect(const AbsRect* r, int target[4]);
FG_EXTERN void FG_FASTCALL ToLongAbsRect(const AbsRect* r, long target[4]);
FG_EXTERN char FG_FASTCALL MsgHitAbsRect(const FG_Msg* msg, const AbsRect* r);

#ifdef  __cplusplus
}
#endif

#endif
