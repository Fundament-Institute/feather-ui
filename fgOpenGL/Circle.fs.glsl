TXT(#version 110\n
varying vec2 pos;\n
uniform vec4 DimBorderBlur;\n
uniform vec4 Corners;\n
uniform vec4 Fill;\n
uniform vec4 Outline;\n
const float PI = 3.14159265359;\n
\n
void main()\n
{\n
    float l = (DimBorderBlur.x + DimBorderBlur.y) * 0.5;\n
    vec2 uv = (pos*2.0) - 1.0;\n
    
    float t = 0.50 - (Corners.z / l) * 2.0;\n
    float r = 1.0 - t;\n
    
    vec2 up = vec2(cos(Corners.x), sin(Corners.x));\n
    
    float outer = (DimBorderBlur.z / l) * 2.0; // double because UV is in range [-1,1], not [0,1]\n
    float inner = (Corners.w / l) * 2.0;\n
    float blur = 0.701 + (DimBorderBlur.w * 0.5); // halve blur amount because of UV again\n
    
    // SDF for circle\n
    float w1 = fwidth(length(uv))*blur; // Don't take fwidth of abs() because it's discontinuous\n
    inner -= w1; // Ensure that antialiasing doesn't create a hole in the middle of the circle\n
    float d0 = abs(length(uv) - r + (outer*0.5) - (inner*0.5)) - t + (outer*0.5) + (inner*0.5);\n
    float d1 = abs(length(uv) - r) - t;\n
    float s1 = smoothstep(w1, -w1, d0); \n
    float alpha1 = smoothstep(w1, -w1, d1); \n

    // SDF for arc\n
    float d2 = dot(up, normalize(uv)) - cos(Corners.y);\n
    float w2 = fwidth(d2)*blur;\n
    float alpha2 = smoothstep(w2, -w2, d2); \n
    //float s2 = smoothstep(w2 + border, -w2 + border, d2); // doesn't work\n
    alpha2 *= smoothstep(0.0, w1, abs(Corners.y - PI));\n
    
    // Merge alpha results of both SDFs\n
    float s = s1*(1.0 - alpha2);\n
    float alpha = alpha1*(1.0 - alpha2);\n
  
  // Output to screen\n
  //gl_FragColor = vec4(adist,adist,adist,1);\n    
  gl_FragColor = (vec4(Fill.rgb, 1)*Fill.a*s) + (vec4(Outline.rgb, 1)*Outline.a*clamp(alpha - s, 0.0, 1.0));\n
}
)