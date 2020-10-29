TXT(#version 110\n
uniform mat4 MVP;\n
attribute vec4 vPosUV;\n
attribute vec4 vColor;\n
varying vec2 uv;\n
varying vec4 color;\n

void main()\n
{\n
  gl_Position = MVP * vec4(vPosUV.xy, 0, 1);\n
  uv = vPosUV.zw;\n
  color = vColor;\n
}\n
)