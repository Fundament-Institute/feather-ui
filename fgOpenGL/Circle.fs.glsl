TXT(#version 110\n
varying vec2 pos;\n
uniform vec4 DimBorderBlur;\n
uniform vec4 Corners;\n
uniform vec4 Fill;\n
uniform vec4 Outline;\n
const float PI = 3.14159265359;\n
\n
float linearstep(float low, float high, float x) { return clamp((x - low) / (high - low), 0.0, 1.0); }\n
void main()\n
{\n
    float l = (DimBorderBlur.x + DimBorderBlur.y) * 0.5;\n
    vec2 uv = (pos*2.0) - 1.0;\n
    float w1 = (1.0 + DimBorderBlur.w)*fwidth(pos.x); \n
    
    float border = (DimBorderBlur.z / l) * 2.0; // double because UV is in range [-1,1], not [0,1]\n
    float t = 0.50 - (Corners.x / l);\n
    // We have to compensate for needing to do smoothstep starting from 0, which combined with abs()\n
    // acts as a ceil() function, creating one extra half pixel.\n
    float r = 1.0 - t - w1;\n
    
    // SDF for circle\n
    float inner = (Corners.y / l) * 2.0;\n
    float d0 = abs(length(uv) - r + (border*0.5) - (inner*0.5)) - t + (border*0.5) + (inner*0.5);\n
    float d1 = abs(length(uv) - r) - t;\n
    float s = pow(linearstep(w1*2.0, 0.0, d0), 2.2); \n
    float alpha = pow(linearstep(w1*2.0, 0.0, d1), 2.2); \n
  
  // Output to screen\n 
  gl_FragColor = (vec4(Fill.rgb, 1)*Fill.a*s) + (vec4(Outline.rgb, 1)*Outline.a*clamp(alpha - s, 0.0, 1.0));\n
}
)

