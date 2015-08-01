// Copyright ©2013 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_RENDERABLE_H__
#define __FG_RENDERABLE_H__

#include "feathergui.h"

#ifdef  __cplusplus
extern "C" {
#endif

// Message types for statics only
enum FG_RENDERMSG
{
  FG_RDRAW,
  FG_RMOVE,
  FG_RSHOW,
  FG_RADDCHILD,
  FG_RREMOVECHILD,
  FG_RSETPARENT,
  FG_RSETAREA,
  FG_RSETELEMENT,
  FG_RSETTEXT,
  FG_RSETUV,
  FG_RSETCOLOR,
  FG_RSETFLAGS,
  FG_RSETORDER,
  FG_RSETFONT, //arg should point to a string that contains the name of the font, other should be the size of the font. Optionally, the top 16-bits of other can specify the line-height.
  FG_RCLONE, // Initializes an empty fgStatic passed in as arg as a clone of this static.
  FG_RCUSTOM
};

enum FG_STATICFLAGS
{
  FGSTATIC_CLIP=1, // Causes any type of static to clip its children.
  FGSTATIC_HIDDEN=2, // Hides a static and all its children.
  FGSTATIC_MARKER=(1<<((sizeof(fgFlag)<<3)-1)), // Tells us this is a static and not a window
};

enum FG_TEXTFLAGS
{
  FGTEXT_CHARWRAP=4, // Wraps lines that go past the edge of the container by character
  FGTEXT_WORDWRAP=8, // Wraps lines that go past the edge of the container by word (the definition of a "word" is implementation specific)
  FGTEXT_ELLIPSES=16, // Lines that go past the bounderies of the text object are cut off with an ellipses (...)
  FGTEXT_RTL=32, // Forces right-to-left text rendering.
  FGTEXT_RIGHTALIGN=64,
  FGTEXT_CENTER=128, // Text horizontal centering behaves differently, because it centers each individual line.
  FGTEXT_STRETCH=256, // Stretches the text to fill the area.
};

enum FG_IMAGEFLAGS
{
  FGIMAGE_STRETCHX=4, // Stretches the image instead of tiling it
  FGIMAGE_STRETCHY=8,
};

struct __WINDOW;

// Representation of a static, which is implemented by the GUI implementation
typedef struct __RENDERABLE {
  fgChild element;
  void (FG_FASTCALL *message)(struct __RENDERABLE* self, unsigned char type, void* arg, int other);
  struct __RENDERABLE* (FG_FASTCALL *clone)(struct __RENDERABLE* self);
  void* userdata;
} fgStatic;

FG_EXTERN void FG_FASTCALL fgStatic_Init(fgStatic* self);
FG_EXTERN void FG_FASTCALL fgStatic_Destroy(fgStatic* self);
FG_EXTERN void FG_FASTCALL fgStatic_Message(fgStatic* self, unsigned char type, void* arg, int other);
FG_EXTERN void FG_FASTCALL fgStatic_RemoveParent(fgStatic* self);
FG_EXTERN void FG_FASTCALL fgStatic_NotifyParent(fgStatic* self);
FG_EXTERN void FG_FASTCALL fgStatic_Clone(fgStatic* self, fgStatic* from); // Clones information and all the children from "from" to "self"
FG_EXTERN void FG_FASTCALL fgStatic_SetParent(fgStatic* BSS_RESTRICT self, fgChild* BSS_RESTRICT parent);

FG_EXTERN fgStatic* FG_FASTCALL fgLoadImage(const char* path);
FG_EXTERN fgStatic* FG_FASTCALL fgLoadImageData(const void* data, size_t length, fgFlag flags);
FG_EXTERN void* FG_FASTCALL fgLoadImageDef(fgFlag flags, const char* data, size_t length, unsigned int color, const CRect* uv);
FG_EXTERN fgStatic* FG_FASTCALL fgLoadText(const char* text, fgFlag flags, const char* font, unsigned short fontsize, unsigned short lineheight);
FG_EXTERN void* FG_FASTCALL fgLoadTextDef(fgFlag flags, const char* text, const char* font, unsigned short fontsize, unsigned short lineheight, unsigned int color);
FG_EXTERN fgStatic* FG_FASTCALL fgLoadDefaultText(const char* text);
//FG_EXTERN fgStatic* FG_FASTCALL fgLoadVector(const char* path);
//FG_EXTERN fgStatic* FG_FASTCALL fgLoadVectorData(const void* data, size_t length);
FG_EXTERN fgStatic* FG_FASTCALL fgEmptyStatic(fgFlag flags);
FG_EXTERN fgStatic* FG_FASTCALL fgLoadDef(void* def, const fgElement* element, int order);
FG_EXTERN void* FG_FASTCALL fgCloneDef(void* def);
FG_EXTERN void FG_FASTCALL fgDestroyDef(void* def);

#ifdef  __cplusplus
}
#endif

#endif
