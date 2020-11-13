
#include "BackendGL.h"
#include "glad/gl.h"
#include <string.h>
#include <memory>
#include "VAO.h"

using namespace GL;

#ifndef USE_EMULATED_VAOS

VAO::VAO(Backend* backend, GLuint shader, const FG_ShaderParameter* parameters, size_t n_parameters, GLuint buffer,
         size_t stride, GLuint indices) :
  _backend(backend)
{
  glGenVertexArrays(1, &_vaoID);
  _backend->LogError("glGenVertexArrays");
  glBindVertexArray(_vaoID);
  _backend->LogError("glBindVertexArray");
  glBindBuffer(GL_ARRAY_BUFFER, buffer);
  _backend->LogError("glBindBuffer");

  if(indices)
  {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indices);
    _backend->LogError("glBindBuffer");
  }

  GLuint offset = 0;
  for(size_t i = 0; i < n_parameters; ++i)
  {
    auto loc = glGetAttribLocation(shader, parameters[i].name);
    _backend->LogError("glGetAttribLocation");
    glEnableVertexAttribArray(loc);
    _backend->LogError("glEnableVertexAttribArray");
    size_t sz   = Context::GetMultiCount(parameters[i].length, parameters[i].multi);
    GLenum type = 0;
    switch(parameters->type)
    {
    case FG_ShaderType_FLOAT: type = GL_FLOAT; break;
    case FG_ShaderType_INT: type = GL_INT; break;
    case FG_ShaderType_UINT: type = GL_UNSIGNED_INT; break;
    }

    glVertexAttribPointer(loc, sz, type, GL_FALSE, stride, (void*)offset);
    offset += Context::GetBytes(type) * sz;
    _backend->LogError("glVertexAttribPointer");
  }

  glBindVertexArray(0);
  _backend->LogError("glBindVertexArray");
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  _backend->LogError("glBindBuffer");
  if(indices)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

VAO::~VAO()
{
  glDeleteVertexArrays(1, &_vaoID);
  _backend->LogError("glDeleteVertexArrays");
}

void VAO::Bind()
{
  glBindVertexArray(_vaoID);
  _backend->LogError("glBindVertexArray");
}

void VAO::Unbind()
{
  glBindVertexArray(0);
  _backend->LogError("glBindVertexArray");
}

#else

VAO::VAO(Backend* backend, GLuint shader, const FG_ShaderParameter* parameters, size_t n_parameters, GLuint buffer,
         size_t stride, GLuint indices) :
  _n_attribs(n_parameters), _vertexBuffer(buffer), _stride(stride), _indexBuffer(indices), _backend(backend)
{
  _attribs = new VAO::VertexAttrib[n_parameters];

  for(size_t i = 0; i < n_parameters; ++i)
  {
    _attribs[i].location = glGetAttribLocation(shader, parameters[i].name);
    _backend->LogError("glGetAttribLocation");

    _attribs[i].numElements = Context::GetMultiCount(parameters[i].length, parameters[i].multi);
    switch(parameters->type)
    {
    case FG_ShaderType_FLOAT: _attribs[i].type = GL_FLOAT; break;
    case FG_ShaderType_INT: _attribs[i].type = GL_INT; break;
    case FG_ShaderType_UINT: _attribs[i].type = GL_UNSIGNED_INT; break;
    }
  }
}

VAO::~VAO() { delete[] _attribs; }

void VAO::Bind()
{
  glBindBuffer(GL_ARRAY_BUFFER, _vertexBuffer);
  _backend->LogError("glBindBuffer");
  if(_indexBuffer != 0)
  {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _indexBuffer);
    _backend->LogError("glBindBuffer");
  }

  GLuint offset = 0;
  for(size_t i = 0; i < _n_attribs; ++i)
  {
    glEnableVertexAttribArray(_attribs[i].location);
    _backend->LogError("glEnableVertexAttribArray");

    glVertexAttribPointer(_attribs[i].location, _attribs[i].numElements, _attribs[i].type, GL_FALSE, _stride,
                          (void*)offset);
    _backend->LogError("glVertexAttribPointer");
    offset += Context::GetBytes(_attribs[i].type) * _attribs[i].numElements;
  }
}

void VAO::Unbind()
{
  for(size_t i = 0; i < _n_attribs; ++i)
  {
    glDisableVertexAttribArray(_attribs[i].location);
    _backend->LogError("glDisableVertexAttribArray");
  }

  if(_indexBuffer != 0)
  {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _indexBuffer);
    _backend->LogError("glBindBuffer");
  }
  glBindBuffer(GL_ARRAY_BUFFER, _vertexBuffer);
  _backend->LogError("glBindBuffer");
}
#endif