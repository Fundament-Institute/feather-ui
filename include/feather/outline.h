// Copyright (c)2019 Black Sphere Studios
// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#ifndef FG__OUTLINE_NODE_H
#define FG__OUTLINE_NODE_H

#include "document.h"

#ifdef  __cplusplus
extern "C" {
#endif

struct FG__OUTLINE_NODE;
typedef void fgLayoutData;
typedef void fgBehaviorData;
typedef void (*fgGenerator)(struct FG__ROOT* root, struct FG__OUTLINE_NODE*, unsigned int, float, fgVec);
typedef void (*fgLayout)(fgDocumentNode*, const fgRect*, const struct FG__OUTLINE_NODE*, float, fgVec);
typedef unsigned int (*fgResolver)(void*, unsigned int, fgCalcNode*, const char* n);

KHASH_DECLARE(datapair, struct FG__DATA_HOOK*, unsigned int);

// The outline describes all the metadata required to construct the document state.
typedef struct FG__OUTLINE_NODE
{
  URect area;
  UVec center;
  fgRect margin;
  fgRect padding;
  fgVec min;
  fgVec max;
  float lineHeight;
  float fontSize;
  int zindex;
  fgFlag layoutFlags;
  FG__OUTLINE_NODE* parent;
  FG__OUTLINE_NODE* children;
  FG__OUTLINE_NODE* next;
  FG__OUTLINE_NODE* prev;
  fgGenerator generator;
  void* gendata; // Data for the generator delegate
  void* data; // This is the "parent" data node that we are bound to, which all relative data operations use
  fgDocumentNode* node; // instantiated document state node
  fgBehaviorFunction behavior;
  fgBehaviorData* statedata; // In case we have multiple behavior data extensions, this points to the current one matched to our function.
  fgResolver stateresolver; // behavior function layout resolver
  fgLayout layout; // Despite being passed the LayoutNode, the layout function actually operates on the RTNode subtree owned by this outline node.
  fgLayoutData* auxdata; // layout function specific data (like "flex-justify") must be stored after any behavior parameter extensions (like fgLayoutLayerNode). This points to where it is located. The generator function is allowed to change this data, because it might be bound to stateful layout information.
  fgResolver auxresolver; // This is a function that maps the fields in auxdata  to IDs. This represents the fact that an OutlineNode would know how to access all of it's own fields given proper type information.
  kh_event_t* listeners;
  kh_datapair_t* datahooks;
} fgOutlineNode;

// This is a default runtime outline generator function 
FG_COMPILER_DLLEXPORT void fgDefaultOutlineGenerator(struct FG__ROOT* root, fgOutlineNode* node, unsigned int field, float scale, fgVec dpi);
FG_COMPILER_DLLEXPORT void fgDestroyOutlineNode(struct FG__ROOT* root, fgOutlineNode* node, float scale, fgVec dpi);
FG_COMPILER_DLLEXPORT unsigned int fgStandardResolver(void* outline, unsigned int index, fgCalcNode* out, const char* id);
FG_COMPILER_DLLEXPORT void fgStandardListener(struct FG__ROOT* root, void* self, void* target, unsigned int field, void* pfield);
FG_COMPILER_DLLEXPORT void fgStandardDataListener(struct FG__ROOT* root, void* self, const void* data, unsigned int start, unsigned int count);
FG_COMPILER_DLLEXPORT void fgGatherEvents(struct FG__ROOT* root, fgOutlineNode* node, unsigned int srcfield, fgCalcNode* nodes, unsigned int count);
FG_COMPILER_DLLEXPORT bool fgCheckEventTuple(struct FG__ROOT* root, kh_event_t* source, unsigned int srcfield, struct FG__EVENT** dest, unsigned int destfield);
FG_COMPILER_DLLEXPORT unsigned int fgNullResolver(void*, unsigned int, fgCalcNode*, const char* n);

#ifdef  __cplusplus
}
#endif

#endif
