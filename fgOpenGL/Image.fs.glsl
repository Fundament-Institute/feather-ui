TXT(
#version 110\n
varying vec2 uv;\n
varying vec4 color;\n

uniform sampler2D texture;\n

void main()\n
{\n
  gl_FragColor = texture2D(texture, uv).rgba * color;\n
}\n
)