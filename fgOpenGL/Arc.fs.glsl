TXT(#version 110\n
varying vec2 pos;\n
uniform vec4 DimBorderBlur;\n
uniform vec4 Corners;\n
uniform vec4 Fill;\n
uniform vec4 Outline;\n
//uniform vec4 AssetSource; // If the source height is negative, it encodes a type of gradient, otherwise it's an image. \n 
//uniform vec4 AssetDest;\n

const float PI = 3.14159265359;\n
\n
float linearstep(float low, float high, float x) { return clamp((x - low) / (high - low), 0.0, 1.0); }\n
vec2 rotate(vec2 p, float a) { return vec2(p.x*cos(a) + p.y*sin(a), p.x*sin(a) - p.y*cos(a)); }\n
vec4 sRGB(vec4 linearRGB) { return vec4(1.055 * pow(linearRGB.rgb, vec3(1.0 / 2.4) - 0.055), linearRGB.a); }\n
vec4 linearRGB(vec4 sRGB) { return vec4(pow((sRGB.rgb + 0.055) / 1.055, vec3(2.4)), sRGB.a); }\n
void main()\n
{\n
    float l = (DimBorderBlur.x + DimBorderBlur.y) * 0.5;\n
    vec2 uv = (pos*2.0) - 1.0;\n
    float width = fwidth(pos.x);\n
    float w1 = (1.0 + DimBorderBlur.w)*width;\n 
    
    float border = (DimBorderBlur.z / l) * 2.0; // double because UV is in range [-1,1], not [0,1]\n
    float t = 0.50 - (Corners.z / l) + w1*1.5;\n
    // We have to compensate for needing to do smoothstep starting from 0, which combined with abs()\n
    // acts as a ceil() function, creating one extra half pixel.\n
    float r = 1.0 - t + w1;\n
    
    // SDF for circle\n
    float d0 = abs(length(uv) - r) - t + border;\n
    float d1 = abs(length(uv) - r) - t;\n

    // SDF for lines that make up arc\n   
    vec2 omega1 = rotate(uv, Corners.x - Corners.y);\n
    vec2 omega2 = rotate(uv, Corners.x + Corners.y);\n
    float d;\n
    
    // TODO: This cannot deal with non-integer circle radii, but it might be generalizable to those cases.
    if(abs(-omega1.y) + abs(omega2.y) < width) {\n
      d = ((Corners.y/PI) - 0.5)*2.0*width;\n
    } else if(Corners.y > PI*0.5) {\n
      d = max(-omega1.y, omega2.y);\n
    } else {\n
      d = min(-omega1.y, omega2.y);\n
    }\n
    
    // Compensate for blur so the circle is still full or empty at 2pi and 0.\n
    d += (clamp(Corners.y/PI, 0.0, 1.0) - 0.5)*2.0*(DimBorderBlur.w * width) + border;\n
    
    float d2 = d - border + w1;\n
    float d3 = min(d, omega1.x + Corners.y) + w1;\n
    
    // Merge results of both SDFs\n
    float s = linearstep(-w1, w1, min(-d0, d2) - w1);\n 
    float alpha = linearstep(-w1, w1, min(-d1, d3) - w1);\n
  
  // Output to screen\n 
  gl_FragColor = vec4(Fill.rgb, 1)*Fill.a*s + vec4(Outline.rgb, 1)*Outline.a*clamp(alpha - s, 0.0, 1.0);\n
}
)

