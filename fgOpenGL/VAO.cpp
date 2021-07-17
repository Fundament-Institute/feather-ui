
#include "BackendGL.h"
#include "glad/glad.h"
#include <string.h>
#include <memory>
#include "VAO.h"

using namespace GL;

#ifndef USE_EMULATED_VAOS

VAO::VAO(Backend* backend, GLuint shader, std::span<FG_ShaderParameter> parameters, GLuint* vbuffers, GLsizei* vstrides,
         size_t n_vbuffers, GLuint indices, GLuint element) :
  _backend(backend), _element(element)
{
  glGenVertexArrays(1, &_vaoID);
  _backend->LogError("glGenVertexArrays");
  glBindVertexArray(_vaoID);
  _backend->LogError("glBindVertexArray");

  if(indices)
  {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indices);
    _backend->LogError("glBindBuffer");
  }

  char* offset = 0;
  for(auto& param : parameters)
  {
    if(param.index >= n_vbuffers)
    {
      (*_backend->_log)(_backend->_root, FG_Level_ERROR, "Parameter index %zu exceeds buffer count: %zu", param.index,
                        n_vbuffers);
      return;
    }

    glBindBuffer(GL_ARRAY_BUFFER, vbuffers[param.index]);
    _backend->LogError("glBindBuffer");
    auto loc = glGetAttribLocation(shader, param.name);
    _backend->LogError("glGetAttribLocation");
    glEnableVertexAttribArray(loc);
    _backend->LogError("glEnableVertexAttribArray");
    int sz      = Context::GetMultiCount(param.length, param.multi);
    GLenum type = Context::GetShaderType(param.type);

    glVertexAttribPointer(loc, sz, type, GL_FALSE, vstrides[param.index], offset);
    offset += Context::GetBytes(type) * sz;
    _backend->LogError("glVertexAttribPointer");
    if(glVertexAttribDivisor)
    {
      glVertexAttribDivisor(loc, param.step);
      _backend->LogError("glVertexAttribDivisor");
    }
    else if(glVertexAttribDivisorARB)
    {
      glVertexAttribDivisorARB(loc, param.step);
      _backend->LogError("glVertexAttribDivisorARB");
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    _backend->LogError("glBindBuffer");
  }

  glBindVertexArray(0);
  _backend->LogError("glBindVertexArray");
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

VAO::VAO(Backend* backend, GLuint shader, std::span<FG_ShaderParameter> parameters, GLuint* vbuffers, GLsizei* vstrides,
    size_t n_vbuffers, GLuint indices, GLuint element) :
  _backend(backend), _element(element), _indexBuffer(indices)
{
  _attribs.reserve(parameters.size());
  _vertexBuffer.reserve(n_vbuffers);

  for(size_t i = 0; i < n_vbuffers; ++i)
  {
    size_t start = _attribs.size();

    for(auto& param : parameters)
    {
      if(param.index == i)
      {
        _attribs.push_back(VertexAttrib{ glGetAttribLocation(shader, param.name),
                                         Context::GetMultiCount(parameters[i].length, parameters[i].multi),
                                         Context::GetShaderType(param.type), param.step });
      }
    }
    
    _vertexBuffer.push_back(
      VertexBuffer{ vbuffers[i], vstrides[i], std::make_span(_attribs).subspan(start, _attribs.size() - start) });
  }
}

VAO::~VAO() {}

void VAO::Bind()
{
  if(_indexBuffer != 0)
  {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _indexBuffer);
    _backend->LogError("glBindBuffer");
  }

  for(auto& buffer : _vertexBuffer)
  {
    glBindBuffer(GL_ARRAY_BUFFER, buffer.id);
    _backend->LogError("glBindBuffer");
    char* offset = 0;
    for(auto& attrib : buffer.attributes)
    {
      glEnableVertexAttribArray(attrib.location);
      _backend->LogError("glEnableVertexAttribArray");

      glVertexAttribPointer(attrib.location, attrib.numElements, attrib.type, GL_FALSE, buffer.stride, offset);
      _backend->LogError("glVertexAttribPointer");
      if(glVertexAttribDivisor)
      {
        glVertexAttribDivisor(attrib.location, attrib.divisor);
        _backend->LogError("glVertexAttribDivisor");
      }
      else if(glVertexAttribDivisorARB)
      {
        glVertexAttribDivisorARB(attrib.location, attrib.divisor);
        _backend->LogError("glVertexAttribDivisorARB");
      }

      offset += Context::GetBytes(attrib.type) * attrib.numElements;
    }
  }

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  _backend->LogError("glBindBuffer");
}

void VAO::Unbind()
{
  if(_indexBuffer != 0)
  {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    _backend->LogError("glBindBuffer");
  }

  for(auto& buffer : _vertexBuffer)
  {
    glBindBuffer(GL_ARRAY_BUFFER, buffer.id);
    _backend->LogError("glBindBuffer");
    GLuint offset = 0;
    for(auto& attrib : buffer.attributes)
    {
      glDisableVertexAttribArray(attrib.location);
      _backend->LogError("glEnableVertexAttribArray");
    }
  }
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  _backend->LogError("glBindBuffer");
}
#endif