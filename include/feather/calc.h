// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#ifndef FG__CALC_H
#define FG__CALC_H

#include "feather.h"
#include "khash.h"
#define MAX_CALC_SIZE 4096

#ifdef  __cplusplus
extern "C" {
#endif

// Defines the types of calculator nodes our script interpreter understands
enum FG_CALC_NODES
{
  FG_CALC_INT,
  FG_CALC_FLOAT,
  FG_CALC_STRING,
  FG_CALC_FUNC,
  FG_CALC_OPERATOR,
  FG_CALC_MASK = 0x7,
  FG_CALC_UNIT_MASK = (UNIT_MASK << 3),
  FG_CALC_DATA = (1 << 6), // data binding flag, merged with the type it returns
  FG_CALC_STATE = (1 << 7), // state binding flag, merged with the type it returns
};

typedef union
{
  long long i;
  double d;
  fgDelegate fn;
  const char* s;
  void* p;
  khiter_t op;
} fgCalcResult;

// The calculator is represented in postfix notation with fixed operand argument counts, allowing it to be efficiently interpreted.
typedef struct
{
  fgCalcResult value;
  int type;
} fgCalcNode;

typedef fgCalcResult (*fgOperator)(fgCalcResult* args);

typedef struct
{
  unsigned char result;
  unsigned char args[6];
  unsigned char n_args;
  fgOperator call;
} fgCalcFunc;

KHASH_DECLARE(function, const char*, fgCalcFunc);

// Verifies the type resolves to the expected type and validates all function calls
FG_COMPILER_DLLEXPORT bool fgVerify(fgCalcNode* nodes, unsigned int n, kh_function_t* functions, FG_CALC_NODES expected);

struct FG__DOCUMENT_NODE;
struct FG__ROOT;

// Evaluates an expression using the given data pointer and internal state source. Assumes fgVerify() was already called.
FG_COMPILER_DLLEXPORT fgCalcNode fgEvaluate(struct FG__ROOT* root, fgCalcNode* nodes, unsigned int n, void* data, struct FG__DOCUMENT_NODE* node, float font, float line, float dpi, float scale);

inline double fgResolveUnit(double x, unsigned int unit, float font, float line, float dpi, float scale)
{
  switch(unit & UNIT_MASK)
  {
  case UNIT_DP:
    return x * (dpi / 96.0);
  case UNIT_SP:
    return x * (dpi / 96.0) * scale;
  case UNIT_EM:
    return x * font;
  case UNIT_LN:
    return x * line;
  case UNIT_PT:
    return x * (dpi / 72.0); // 1 point is 1/72 of an inch
  case UNIT_MM:
    return x * (dpi / 25.4); // 25.4 mm to an inch
  default:
  case UNIT_PX:
    return x;
  }
}

#ifdef  __cplusplus
}
#endif

#endif
