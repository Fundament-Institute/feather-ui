// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "Backend.h"

#include "GLError.h"
#include "BackendGL.h"

using namespace GL;

GLenum GLError::log(Backend* backend)
{
  if(_error != GL_INVALID_INDEX)
  {
    _checked();

    if(_error != GL_NO_ERROR)
    {
      (*backend->_log)(backend->_root, FG_Level_ERROR, "%s: 0x%04X", _context, _error);
      return _error;
    }
  }

  return GL_NO_ERROR;
}
