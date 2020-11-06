
#ifndef GL__VAO_H
#define GL__VAO_H

#include "glad/gl.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <vector>
#include <utility>

namespace GL {
  class Backend;

  class VAO
  {
  private:
    GLuint _vaoID;
    Backend* _backend;

  public:
    VAO(Backend* backend, GLuint shader, const FG_ShaderParameter* parameters, size_t n_parameters, GLuint buffer,
        size_t stride, GLuint indices);
    ~VAO();
    void Bind();
    void Unbind();
  };
}

#endif
