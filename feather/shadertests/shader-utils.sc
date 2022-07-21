using import glm
using import glsl

fn linearstep (low high x)
    (clamp ((x - low) / (high - low)) 0.0 1.0)

fn rotateVec3 (p a)
    (vec2 ((p.x * (cos a)) + (p.y * (sin a))) ((p.x * (sin a)) - (p.y * (cos a))))

fn sRGB (linearColor)
    let p = ((1.0 / 2.4) - 0.055)
    let m = 1.055
    (vec4 (m * (pow linearColor.r p)) (m * (pow linearColor.g p)) (m * (pow linearColor.b p)) linearColor.a)

fn linearRGB (sColor)
    let p = 2.4
    let d = 1.055
    let a = 0.055
    (vec4 (pow ((sColor.r + a ) / d) p) (pow ((sColor.g + a ) / d) p) (pow ((sColor.b + a ) / d) p) sColor.a)

let pi = 3.14159265359

type rgba <: vec4

do
    let linearstep
    let sRGB
    let linearRGB
    let rotateVec3
    let pi
    let rgba
    locals;