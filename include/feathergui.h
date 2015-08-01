/* Feather - Lightweight GUI Abstraction Layer
   Copyright ©2013 Black Sphere Studios

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
typedef unsigned short fgFlag;

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

typedef struct _FG_CHILD {
  fgElement element;
  void (FG_FASTCALL *destroy)(void* self); // virtual destructor
  void (*free)(void* self); // pointer to deallocation function
  struct _FG_CHILD* parent;
  struct _FG_CHILD* root; // children list root
  struct _FG_CHILD* last; // children list last
  struct _FG_CHILD* next;
  struct _FG_CHILD* prev;
  AbsRect margin; // defines the amount of external margin. Bottom and right are not automatically negated, so a 5 pixel margin would be 5,5,-5,-5
  int order; // order relative to other windows or statics
  fgFlag flags;
} fgChild;


enum FG_MSGTYPE
{
  FG_MOUSEDOWN,
  //FG_MOUSEDBLCLICK,
  FG_MOUSEUP, // Sent to focused window
  FG_MOUSEON,
  FG_MOUSEOFF,
  FG_MOUSEMOVE, // Sent to focused window
  FG_MOUSESCROLL, // Sent to focused window
  FG_KEYUP,
  FG_KEYDOWN,
  FG_KEYCHAR, //Passed in conjunction with keydown/up to differentiate a typed character from other keys.
  FG_JOYBUTTONDOWN,
  FG_JOYBUTTONUP,
  FG_JOYAXIS,
  FG_GOTFOCUS,
  FG_LOSTFOCUS,
  FG_MOVE, // Passed when any change is made to an element
  FG_SETPARENT,
  FG_SETFLAG, // Send in the flag in the first int arg, then set it to on or off (1 or 0) in the second argument
  FG_SETFLAGS, // Sets all the flags to the first int arg.
  FG_SETORDER, // Sets the order of a window in the otherint argument
  FG_SETCAPTION, 
  FG_SETALPHA, // Used so an entire widget can be made to fade in or out. (Support is not guaranteed)
  FG_SETMARGIN,
  FG_SETCOLUMNS, // If the second pointer is 0, the number of columns is set to the first int. Otherwise, the first int is treated as a column index number, which has it's width set to the Coord pointer to by the second pointer
  FG_SETSKIN,
  FG_SETNAME, // Sets the unique name for this object for skin collection mapping. Can be null.
  FG_SETSTYLE,
  FG_GETCLASSNAME, // Returns a unique string identifier for the class
  FG_GETNAME, // May return a unique string for this object, or will return NULL.
  FG_GETITEM, // If the control has columns, the column number is specified in the high-order word
  FG_GETROW,
  FG_ADDITEM, // Used for anything involving items (menus, lists, etc)
  FG_ADDCHILD, // Pass an FG_Msg with this type and set the other pointer to the child that should be added. Can be either an fgWindow or an fgStatic
  FG_REMOVECHILD,
  FG_REMOVEITEM,
  FG_SHOW, // Send an otherint equal to 0 to hide, otherwise its made visible
  FG_NUETRAL, // Sent when a button or other hover-enabled control switches to it's nuetral state
  FG_HOVER, // Sent when a hover-enabled control switches to its hover state
  FG_ACTIVE, // Sent when a hover-enabled control switches to its active state
  FG_CLICKED, // Sent when a hover-enabled control recieves a valid click event (a MOUSEUP inside the control while it has focus)
  FG_DESTROY, // Represents a request for the window to be destroyed. This request can be ignored, so it should not be used as a destructor.
  FG_CUSTOMEVENT
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
        char sigkeys;
    };
    struct { float joyvalue; short joyaxis; }; // JOYAXIS
    struct { char joydown; short joybutton; }; // JOYBUTTON
    struct { void* other; size_t otheraux; }; // Used by any generic messages (FG_ADDCHILD, FG_SETPARENT, etc.)
    struct { void* other1; void* other2; }; // Used by anything requiring 2 pointers, possibly for a return value.
    struct { ptrdiff_t otherint; ptrdiff_t otherintaux; }; // Used by any generic message that want an int (FG_SETCLIP, FG_SETCENTERED, etc.)
  };
  unsigned char type;
} FG_Msg;

FG_EXTERN void FG_FASTCALL fgChild_Init(fgChild* self);
FG_EXTERN void FG_FASTCALL fgChild_Destroy(fgChild* self);
FG_EXTERN void FG_FASTCALL fgChild_SetParent(fgChild* BSS_RESTRICT self, fgChild* BSS_RESTRICT parent, fgFlag flag);

FG_EXTERN AbsVec FG_FASTCALL ResolveVec(const CVec* v, const AbsRect* last);
FG_EXTERN void FG_FASTCALL ResolveRect(const fgChild* self, AbsRect* out);
FG_EXTERN void FG_FASTCALL ResolveRectCache(AbsRect* BSS_RESTRICT r, const fgChild* elem, const AbsRect* BSS_RESTRICT last);
FG_EXTERN char FG_FASTCALL CompareCRects(const CRect* l, const CRect* r); // Returns 0 if both are the same or 1 otherwise
//FG_EXTERN char FG_FASTCALL CompChildOrder(const fgChild* l, const fgChild* r);
FG_EXTERN void FG_FASTCALL MoveCRect(AbsVec v, CRect* r);
FG_EXTERN char FG_FASTCALL HitAbsRect(const AbsRect* r, FABS x, FABS y);
//FG_EXTERN void FG_FASTCALL ToIntAbsRect(const AbsRect* r, int target[static 4]);
FG_EXTERN void FG_FASTCALL ToIntAbsRect(const AbsRect* r, int target[4]);
FG_EXTERN void FG_FASTCALL ToLongAbsRect(const AbsRect* r, long target[4]);
FG_EXTERN char FG_FASTCALL MsgHitAbsRect(const FG_Msg* msg, const AbsRect* r);
FG_EXTERN char FG_FASTCALL MsgHitCRect(const FG_Msg* msg, const fgChild* child);
FG_EXTERN void FG_FASTCALL LList_Remove(fgChild* self, fgChild** root, fgChild** last);
FG_EXTERN void FG_FASTCALL LList_Add(fgChild* self, fgChild** root, fgChild** last, fgFlag flag);
FG_EXTERN void FG_FASTCALL LList_Insert(fgChild* self, fgChild* target, fgChild** last);
FG_EXTERN void FG_FASTCALL LList_ChangeOrder(fgChild* self, fgChild** root, fgChild** last, fgFlag flag);
FG_EXTERN void FG_FASTCALL VirtualFreeChild(fgChild* self);

#ifdef  __cplusplus
}
#endif

#endif
