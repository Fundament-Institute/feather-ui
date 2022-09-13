// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "ProviderGL.hpp"

#include "GLError.hpp"
#include "ProviderGL.hpp"

using namespace GL;

GLenum GLError::log(Provider* backend)
{
  if(_error != GLError::INVALID_ERROR)
  {
    _checked();

    if(_error != GL_NO_ERROR)
    {
      backend->Log(FG_Level_Error, _file, _line, "OpenGL Error: ", _callsite, _error);
      return _error;
    }
  }

  return GL_NO_ERROR;
}
