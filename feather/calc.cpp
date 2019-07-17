// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#include "feather/calc.h"
#include "feather/document.h"
#include "feather/root.h"
#include "util.h"
#include <malloc.h>
#include <assert.h>

__KHASH_IMPL(function, extern "C", const char*, fgCalcFunc, 1, kh_str_hash_funcins, kh_str_hash_equal);

extern "C" bool fgVerify(fgCalcNode* nodes, unsigned int n, kh_function_t* functions, FG_CALC_NODES expected)
{
  if(n > MAX_CALC_SIZE) // Sanity check
    return false;

  auto stack = (char*)_alloca(n);
  unsigned int c = 0;

  for(unsigned int i = 0; i < n; ++i)
  {
    switch(nodes[i].type)
    {
    case FG_CALC_OPERATOR:
    {
      // Verify our pre-evaluated iterator is still valid and get the function info
      if(nodes[i].value.op >= kh_end(functions) || !kh_exist(functions, nodes[i].value.op))
        return false;

      fgCalcFunc& fn = kh_val(functions, nodes[i].value.op);

      // If the number of arguments exceeds the stack size this is obviously invalid
      if(fn.n_args > c)
        return false;

      // Pop each argument off the stack and verify it matches the expected type
      for(unsigned int j = 0; j < fn.n_args; ++j)
      {
        if((stack[--c] & (~FG_CALC_UNIT_MASK)) != fn.args[j]) // Ignore units here
          return false;
      }

      // Push the function result type on to the stack
      stack[c++] = fn.result;
      break;
    }
    case FG_CALC_DATA | FG_CALC_OPERATOR:
    case FG_CALC_STATE | FG_CALC_OPERATOR:
      return false;
      break;
    default:
      if((nodes[i].type & FG_CALC_MASK) == FG_CALC_OPERATOR) // Operators can't have units or other flags
        return false;
      if(nodes[i].type & (FG_CALC_DATA | FG_CALC_STATE))
      {
        if(c < 1 || (stack[--c] != FG_CALC_STRING))
          return false; // both data and state accept 1 string argument
      }

      switch(nodes[i].type & FG_CALC_MASK)
      {
      case FG_CALC_STRING:
      case FG_CALC_FUNC:
      case FG_CALC_OPERATOR:
        if(nodes[i].type & FG_CALC_UNIT_MASK) // Not valid to have units on these types
          return false;
        break;
      }

      stack[c++] = nodes[i].type & (FG_CALC_MASK | FG_CALC_UNIT_MASK);
      break;
    }
  }

  if(c != 1) // must always evaluate to exactly 1 result
    return false;

  return (stack[0] & (~FG_CALC_UNIT_MASK)) == expected;
}

extern "C" fgCalcResult fgResolveCalc(fgCalcResult result, unsigned char type, float font, float line, float dpi, float scale)
{
  switch(type & FG_CALC_MASK)
  {
  case FG_CALC_INT:
    result.i = (long long)fgResolveUnit((double)result.i, (type & FG_CALC_UNIT_MASK) >> 3, font, line, dpi, scale);
    break;
  case FG_CALC_FLOAT:
    result.d = fgResolveUnit(result.d, (type & FG_CALC_UNIT_MASK) >> 3, font, line, dpi, scale);
    break;
  default:
    assert(!(type & FG_CALC_UNIT_MASK));
    break;
  }

  return result;
}

extern "C" fgCalcNode fgEvaluate(struct FG__ROOT* root, fgCalcNode* nodes, unsigned int n, void* data, struct FG__DOCUMENT_NODE* node, float font, float line, float dpi, float scale)
{
  assert(n < MAX_CALC_SIZE);

  // This function assumes that fgVerify has been called so it can skip the safety checks.
  auto stack = (fgCalcResult*)_alloca(n * sizeof(fgCalcResult));
  unsigned int c = 0;
  unsigned char last = FG_CALC_NONE;
  kh_function_t* functions = root->operators;

  for(unsigned int i = 0; i < n; ++i)
  {
    if(nodes[i].type == FG_CALC_OPERATOR)
    {
      assert(nodes[i].value.op >= kh_end(functions) || !kh_exist(functions, nodes[i].value.op));
      fgCalcFunc& fn = kh_val(functions, nodes[i].value.op);

      assert(c > fn.n_args);
      c -= fn.n_args;
      stack[c++] = fn.call(stack);
      last = fn.result;
    }
    else if(nodes[i].type & FG_CALC_STATE)
    {
      fgMessage msg = { FG_MSG_STATE_GET, 0 };
      fgCalcNode calc = { FG_CALC_DATA };
      msg.getState = { nodes[i].value.s, &calc };
      stack[c] = calc.value;
      last = nodes[i].type & FG_CALC_MASK;
      assert(last == calc.type);
      fgSendMessage(root, node, &msg);
      stack[c] = fgResolveCalc(stack[c], nodes[i].type & (FG_CALC_MASK | FG_CALC_UNIT_MASK), font, line, dpi, scale);
      c++;
    }
    else if(nodes[i].type & FG_CALC_DATA)
    {
      last = nodes[i].type & FG_CALC_MASK;
      auto field = fgGetData(root, data, nodes[i].value.s);
      fgCalcResult& ref = stack[c++];
      switch((field.primitive & FG_DATA_MASK) | ((int)last << 16))
      {
      case (FG_DATA_FLOAT | (FG_CALC_FLOAT << 16)): ref.d = fgResolveUnit(field.data.f, field.primitive & FG_DATA_UNIT_MASK, font, line, dpi, scale); break;
      case (FG_DATA_DOUBLE | (FG_CALC_FLOAT << 16)): ref.d = fgResolveUnit(field.data.d, field.primitive & FG_DATA_UNIT_MASK, font, line, dpi, scale); break;
      case (FG_DATA_INT32 | (FG_CALC_INT << 16)): ref.i = (long long)fgResolveUnit(field.data.i, field.primitive & FG_DATA_UNIT_MASK, font, line, dpi, scale); break;
      case (FG_DATA_INT64 | (FG_CALC_INT << 16)): ref.i = (long long)fgResolveUnit(field.data.l, field.primitive & FG_DATA_UNIT_MASK, font, line, dpi, scale); break;
      case (FG_DATA_FLOAT | (FG_CALC_INT << 16)): ref.i = (long long)fgResolveUnit(field.data.f, field.primitive & FG_DATA_UNIT_MASK, font, line, dpi, scale); break;
      case (FG_DATA_DOUBLE | (FG_CALC_INT << 16)): ref.i = (long long)fgResolveUnit(field.data.d, field.primitive & FG_DATA_UNIT_MASK, font, line, dpi, scale); break;
      case (FG_DATA_INT32 | (FG_CALC_FLOAT << 16)): ref.d = fgResolveUnit(field.data.i, field.primitive & FG_DATA_UNIT_MASK, font, line, dpi, scale); break;
      case (FG_DATA_INT64 | (FG_CALC_FLOAT << 16)): ref.d = (double)fgResolveUnit(field.data.l, field.primitive & FG_DATA_UNIT_MASK, font, line, dpi, scale); break;
      case (FG_DATA_FUNCTION | (FG_CALC_FUNC << 16)): ref.fn = field.data.func; break; // TODO: encode lambda object in return value
      case (FG_DATA_ARRAY | FG_DATA_CHAR | (FG_CALC_STRING << 16)): // null terminated string
      case (FG_DATA_ARRAY | FG_DATA_BYTE | (FG_CALC_STRING << 16)): ref.s = (const char*)(root->getindex)(field.obj, ~0).data.ptr; break;
      default:
        assert(false);
        break;
      }
    }
    else
    {
      last = nodes[i].type;
      stack[c++] = nodes[i].value;
    }
  }

  assert(c == 1);
  return fgCalcNode{ stack[0], last };
}