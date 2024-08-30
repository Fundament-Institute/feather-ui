TXT(#version 110\n
uniform vec4 Color;\n
varying vec2 vLineCenter;\n
void main(void)\n
{\n
      vec4 col = Color;        \n
      float d = length(vLineCenter-gl_FragCoord.xy);\n
      float LineWidth = 1.0;\n
      col.a *= smoothstep(0.0, 1.0, LineWidth-d);\n
      gl_FragColor = vec4(col.rgb*col.a, col.a);\n
};\n
)