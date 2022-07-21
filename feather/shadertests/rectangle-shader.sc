using import glm
using import glsl
using import .shader-utils

fn rectangle (samplePos halfSize edges)
    local edge = 20.0
    if (samplePos.x > 0.0)
        if (samplePos.y < 0.0)
            edge = edges.y
        else
            edge = edges.z
    else
        if (samplePos.y < 0.0)
            edge = edges.x
        else
            edge = edges.w
    
    let componentWiseEdgeDistance = ((abs samplePos) - halfSize + (vec2 edge))
    let outsideDist = (length (vec2 (max componentWiseEdgeDistance.x 0.0) (max componentWiseEdgeDistance.y 0.0)))
    let insideDist = (min (max componentWiseEdgeDistance.x componentWiseEdgeDistance.y) 0.0)
    (outsideDist + insideDist - edge)

fn... rect-shader (x : f32, y : f32, size : f32, DimBorderBlur : vec4, corners : vec4, fill : vec4, outline : vec4)
    let pos = (vec2 x y)
    let w = (1.0 + DimBorderBlur.w) # Originally uses the derivative of...DimBorderBlur.x + pos.x? That should still come out to 1.0...
    let uv = ((pos * 2) - size)

    let dist = (rectangle uv ((vec2 DimBorderBlur.x DimBorderBlur.y) * 0.5) corners)
    let alpha = (linearstep w (- w) dist)
    let s = (linearstep w (- w) (dist + DimBorderBlur.z))

    let fill = ((vec4 fill.rgb 1) * fill.a * s)
    let outl = ((vec4 outline.rgb 1) * outline.a * (clamp (alpha - s) 0.0 1.0)) 
    return (fill + outl)

do
    let rect-shader
    locals;