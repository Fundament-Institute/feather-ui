// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#ifndef FG__DATA_H
#define FG__DATA_H

#ifdef  __cplusplus
extern "C" {
#endif

// Defines the primitive types (not the complex type) that feather understands
enum FG_DATA_TYPES
{
  FG_DATA_NONE, // error state
  FG_DATA_FLOAT,
  FG_DATA_DOUBLE,
  FG_DATA_INT32,
  FG_DATA_INT64,
  FG_DATA_FUNCTION, // arbitrary function pointer
  FG_DATA_CHAR, // ascii character or partial UTF8 rune
  FG_DATA_BYTE, // Single byte for binary encoded data
  FG_DATA_COMPLEX, // Opaque data type we cannot understand, represented using a pointer
  FG_DATA_OBJECT, // set of named fields
  FG_DATA_SUBARRAY, // Only used to represent an array of an array
  FG_DATA_ARRAY = 0x10, // When this bit is set, the type is an array of something.
  FG_DATA_ELEMENT_MASK = 0x0F, // When this is an array, this determines the array element type.
  FG_DATA_MASK = 0x1f,
  FG_DATA_UNIT_MASK = (UNIT_MASK << 5),
};

union fgDataUnion
{
  char c;
  unsigned char b;
  float f;
  double d;
  int i;
  long long l;
  unsigned long long sz; // length of array object
  fgDelegate func;
  void* ptr; // Used for returning or setting raw arrays
};

typedef struct
{
  const char* type; // Complex program type name
  const char* name;
  void* obj; // Internal data pointer
  int primitive; // primitive C type that forms a tagged union
  union fgDataUnion data;
} fgDataField;

typedef fgDataField(*fgGetField)(void* obj, const char* field); // If field is NULL, just gets the object itself
typedef fgDataField(*fgGetIndex)(void* obj, unsigned int index); // If index is MAX_UINT (~0), returns a raw pointer to the array, which will only work if the element type is a known C primitive.
typedef bool(*fgSetData)(void* obj, enum FG_DATA_TYPES type, union fgDataUnion data);
// If start == end, just inserts the data. If length is 0, deletes everything between start (inclusive) and end (exclusive). Otherwise replaces the data between start and end with the provided array.
typedef bool(*fgSetRange)(void* obj, enum FG_DATA_TYPES element, void* data, unsigned long long length, unsigned int offset, unsigned int count);

#ifdef  __cplusplus
}
#endif

#endif
