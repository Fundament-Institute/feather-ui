// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "BackendGL.hpp"

#include "GLError.hpp"
#include "BackendGL.hpp"

using namespace GL;

GLenum GLError::log(Backend* backend)
{
  if(_error != GLError::INVALID_ERROR)
  {
    _checked();

    if(_error != GL_NO_ERROR)
    {
      backend->Log(FG_Level_ERROR, _file, _line, "OpenGL Error: ", _callsite, _error);
      return _error;
    }
  }

  return GL_NO_ERROR;
}
