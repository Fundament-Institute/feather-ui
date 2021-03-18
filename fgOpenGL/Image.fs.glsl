TXT(#version 110\n
varying vec2 uv;\n
varying vec4 color;\n

uniform sampler2D texture;\n

void main()\n
{\n
  vec4 c = texture2D(texture, uv);
  gl_FragColor = vec4(c.rgb * color.rgb * vec3(color.a,color.a,color.a), c.a*color.a);\n
  //gl_FragColor = vec4(uv.x, uv.y, 0, 1);\n
}\n
)