/* Feather - Data Driven UI Layout Engine
   Copyright (c)2019 Black Sphere Studios

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

#ifndef FG__FEATHER_H
#define FG__FEATHER_H

#include "compiler.h"
#include <stdbool.h>

#define FEATHER_VERSION_MAJOR 0
#define FEATHER_VERSION_MINOR 1
#define FEATHER_VERSION_REVISION 1
#define FEATHER_VERSION(v, m, r, b) (((v|0ULL)<<48) | ((m|0ULL)<<32) | ((r|0ULL)<<16) | (b|0ULL))

#ifdef  __cplusplus
extern "C" {
#endif

typedef unsigned int fgFlag;
typedef int fgError;
typedef void(*fgDelegate)(void*);

// A point in 2D space
typedef struct
{
  float x;
  float y;
} fgVec;

// Integer points, for DPI and sizes that cannot be fractional.
typedef struct
{
  int x;
  int y;
} fgVeci;

// A point in 3D space. Used only for layer transformations and 3D projection calculations, not in the layout.
typedef struct
{
  float x;
  float y;
  float z;
} fgVec3D;

// Represents a rectangle by absolute placement of the top-left and bottom-right corners.
typedef union
{
  struct
  {
    float left;
    float top;
    float right;
    float bottom;
  };

  struct
  {
    fgVec topleft;
    fgVec bottomright;
  };

  float ltrb[4];
} fgRect;

// Unified vector that contains both absolute and relative coordinates
typedef struct
{
  fgVec abs;
  fgVec rel;
} UVec;

// Unified rect that contains both absolute and relative coordinates
typedef struct
{
  fgRect abs;
  fgRect rel;
} URect;

// A 32-bit RGBA, 8-bit per channel color used for all rendering.
typedef union
{
  unsigned int color;
  unsigned char colors[4];
  struct
  {
    unsigned char b;
    unsigned char g;
    unsigned char r;
    unsigned char a;
  };
} fgColor;

enum LAYOUT_FLAGS
{
  LAYOUT_ALLOW_TAB = (1 << 0),
  LAYOUT_ALLOW_KEY_EVENTS = (1 << 1),
  LAYOUT_ALLOW_MOUSE_EVENTS = (1 << 2),
  LAYOUT_ALLOW_TOUCH_EVENTS = (1 << 3),
  LAYOUT_ALLOW_JOYSTICK_EVENTS = (1 << 4),
  LAYOUT_CLIP_CHILDREN = (1 << 5),
  LAYOUT_HIDDEN = (1 << 6),
};

// Standard boolean states shared by most behavior components
enum STATE_FLAGS
{
  STATE_HOVER = (1 << 0), // Mouse is hovering over this element
  STATE_ACTIVE = (1 << 1), // Mouse button is being held down on this element
  STATE_TAB_FOCUS = (1 << 2), // Current tabbed item
  STATE_KEY_FOCUS = (1 << 3), // Has keyboard focus (usually the same as tabFocus, but some elements reject the key focus)
  STATE_SELECTED = (1 << 4),
  STATE_DISABLED = (1 << 5),
};

enum FG_UNIT
{
  UNIT_DP = 0, // DPI scaled units by default
  UNIT_SP,
  UNIT_EM, // Scales on font-size
  UNIT_LN,
  UNIT_PT,
  UNIT_MM,
  UNIT_PX, // Exactly this many pixels regardless of DPI
  UNIT_MASK = 7,
};

enum WORD_WRAP
{
  WORD_NOWRAP = 0,
  WORD_WRAP,
  LETTER_WRAP,
};

enum TEXT_DIRECTION
{
  TEXT_LEFT_RIGHT = 0,
  TEXT_RIGHT_LEFT,
  TEXT_TOP_BOTTOM,
};

enum FLEX_JUSTIFY
{
  FLEX_START,
  FLEX_CENTER,
  FLEX_END,
};

enum FLEX_ALIGN
{
  FLEX_ALIGN_STRETCH = 0,
  FLEX_ALIGN_START,
  FLEX_ALIGN_CENTER,
  FLEX_ALIGN_END,
  FLEX_ALIGN_BASELINE,
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

enum FG_LOGLEVEL {
  FGLOG_NONE = -1,
  FGLOG_FATAL = 0,
  FGLOG_ERROR = 1,
  FGLOG_WARNING = 2,
  FGLOG_NOTICE = 3,
  FGLOG_INFO = 4,
  FGLOG_DEBUG = 5,
};

FG_COMPILER_DLLEXPORT void fgScaleRectDPI(fgRect* rect, float dpix, float dpiy);
FG_COMPILER_DLLEXPORT void fgInvScaleRectDPI(fgRect* rect, float dpix, float dpiy);
FG_COMPILER_DLLEXPORT void fgScaleVecDPI(fgVec* v, float dpix, float dpiy);
FG_COMPILER_DLLEXPORT void fgInvScaleVecDPI(fgVec* v, float dpix, float dpiy);
FG_COMPILER_DLLEXPORT void fgMergeRect(fgRect* target, const fgRect* src);

#ifdef  __cplusplus
}
#endif

#endif
