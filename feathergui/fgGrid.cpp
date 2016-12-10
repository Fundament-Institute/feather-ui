// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgGrid.h"
#include "feathercpp.h"
#include "bss-util/cDynArray.h"

void FG_FASTCALL fgGrid_Init(fgGrid* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, units, (fgDestroy)&fgGrid_Destroy, (fgMessage)&fgGrid_Message);
}
void FG_FASTCALL fgGrid_Destroy(fgGrid* self)
{
  fgBox_Destroy(&self->box);
}
size_t FG_FASTCALL fgGrid_Message(fgGrid* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgBox_Message(&self->box, msg);
    fgList_Init(&self->header, *self, 0, "Grid$header", FGELEMENT_BACKGROUND | FGELEMENT_EXPAND | FGBOX_TILEX, &fgTransform_EMPTY, 0);
    memset(&self->selected, 0, sizeof(fgVectorElement));
    self->hover.color = 0x99999999;
    break;
  case FG_ADDITEM:


    break;
  case FG_SETVALUE:
    return fgPassMessage(self->header, msg);
  }

  return fgBox_Message(&self->box, msg);
}

void FG_FASTCALL fgGridRow_Init(fgGridRow* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, units, (fgDestroy)&fgGridRow_Destroy, (fgMessage)&fgGridRow_Message);

}
void FG_FASTCALL fgGridRow_Destroy(fgGridRow* self)
{
  fgBox_Destroy(&self->box);
}

size_t FG_FASTCALL fgGridRow_Message(fgGridRow* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_ADDITEM:
    break;
  }
  return fgBox_Message(&self->box, msg);
}

void fgGrid::InsertColumn(const char* name, size_t column) { _sendsubmsg<FG_ADDITEM, const void*, size_t>(*this, FGITEM_COLUMN, name, column); }
void fgGrid::SetItem(fgElement* item, size_t column, size_t row) { _sendsubmsg<FG_SETITEM, ptrdiff_t, size_t>(*this, FGITEM_ELEMENT, column, row); }
void fgGrid::SetItem(const char* item, size_t column, size_t row) { _sendsubmsg<FG_SETITEM, ptrdiff_t, size_t>(*this, FGITEM_TEXT, column, row); }
fgGridRow* fgGrid::InsertRow(size_t row) { return reinterpret_cast<fgGridRow*>(_sendsubmsg<FG_ADDITEM, ptrdiff_t>(*this, FGITEM_ROW, row)); }
bool fgGrid::RemoveColumn(size_t column) { return _sendsubmsg<FG_REMOVEITEM, ptrdiff_t>(*this, FGITEM_COLUMN, column); }
bool fgGrid::RemoveRow(size_t row) { return _sendsubmsg<FG_REMOVEITEM, ptrdiff_t>(*this, FGITEM_ROW, row); }
bool fgGrid::RemoveItem(size_t column, size_t row) { return _sendsubmsg<FG_REMOVEITEM, ptrdiff_t>(*this, 0, column, row); }
fgElement* fgGrid::GetItem(size_t column, size_t row) { return reinterpret_cast<fgElement*>(_sendsubmsg<FG_GETITEM, ptrdiff_t>(*this, 0, column, row)); }
fgGridRow* fgGrid::GetRow(size_t row) { return reinterpret_cast<fgGridRow*>(_sendsubmsg<FG_GETITEM, ptrdiff_t>(*this, FGITEM_ROW, row)); }

void fgGridRow::InsertItem(fgElement* item, size_t column) { _sendmsg<FG_ADDITEM, void*, size_t>(*this, item, column); }
void fgGridRow::InsertItem(const char* item, size_t column) { _sendmsg<FG_ADDITEM, const void*, size_t>(*this, item, column); }
bool fgGridRow::RemoveItem(size_t column) { return _sendmsg<FG_REMOVEITEM, ptrdiff_t>(*this, column); }
fgElement* fgGridRow::GetItem(size_t column) { return reinterpret_cast<fgElement*>(_sendmsg<FG_GETITEM, ptrdiff_t>(*this, column)); }