TXT(#version 110\n
varying vec2 pos;\n
uniform vec4 DimBorderBlur;\n
uniform vec4 Corners;\n
uniform vec4 Fill;\n
uniform vec4 Outline;\n
const float PI = 3.14159265359;\n
\n
// For angular antialiasing, a linear curve looks better than smoothstep
float linearstep(float low, float high, float x) { return clamp((x - low) / (high - low), 0.0, 1.0); }\n
vec2 rotate(vec2 p, float a) { return vec2(p.x*cos(a) + p.y*sin(a), p.x*sin(a) - p.y*cos(a)); }\n
void main()\n
{\n
    float l = (DimBorderBlur.x + DimBorderBlur.y) * 0.5;\n
    vec2 uv = (pos*2.0) - 1.0;\n
    float blur = 1.0 + (DimBorderBlur.w * 0.5); // halve blur amount because of UV again\n
    float w1 = fwidth(pos.x)*blur; // Don't take fwidth of abs() because it's discontinuous\n
    
    float t = 0.50 - (Corners.z / l);\n
    // We have to compensate for needing to do smoothstep starting from 0, which combined with abs()\n
    // acts as a ceil() function, creating one extra half pixel.\n
    float r = 1.0 - t - w1;\n
    
    // SDF for circle\n
    float outer = (DimBorderBlur.z / l) * 2.0; // double because UV is in range [-1,1], not [0,1]\n
    float inner = (Corners.w / l) * 2.0;\n
    float d0 = abs(length(uv) - r + (outer*0.5) - (inner*0.5)) - t + (outer*0.5) + (inner*0.5);\n
    float d1 = abs(length(uv) - r) - t;\n
    float s1 = linearstep(w1*2.0, 0.0, d0); \n
    float alpha1 = linearstep(w1*2.0, 0.0, d1); \n

    // SDF for lines that make up arc\n   
    float omega1 = -rotate(uv, Corners.x - Corners.y).y;\n
    float omega2 = rotate(uv, Corners.x + Corners.y).y;\n
    float d2 = min(omega1, omega2);\n
    if(Corners.y > PI*0.5) {\n
      d2 = max(omega1, omega2);\n
    }\n
    
    float alpha2 = smoothstep(w1, -w1, d2); \n
    
    // Merge alpha results of both SDFs\n
    float s = s1*(1.0 - alpha2);\n
    float alpha = alpha1*(1.0 - alpha2);\n
  
  // Output to screen\n 
  gl_FragColor = (vec4(Fill.rgb, 1)*Fill.a*s) + (vec4(Outline.rgb, 1)*Outline.a*clamp(alpha - s, 0.0, 1.0));\n
}
)

