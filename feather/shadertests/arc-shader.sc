using import glm
using import glsl
using import .shader-utils

fn arc (x y size DimBorderBlur corners fill outline)
    let l = ((DimBorderBlur.x + DimBorderBlur.y) * 0.5)
    let uv = (((vec2 x y) * 2.0) - size)
    let width = 1.0 # Width of a pixel is always 1.0 on software renderer
    let w1 = ((1.0 + DimBorderBlur.w) * width)
    
    let border = ((DimBorderBlur.z / 1) * 2.0)
    # Dividing the blur value by 1?
    let t = (0.5 - (corners.z / 1) + w1 * 1.5)
    # And corners.z, too???

    let r = (l - t + w1)
    # l used to be literal 1.0, but based on what I learned from circle,
    # it should be l.

    let d0 = ((abs ((length uv) - r)) - t + border)
    let d1 = ((abs ((length uv) - r)) - t)

    let omega1 = (rotateVec3 uv (corners.x - corners.y)) # Is UV supposed to be in the range 0 to 1?
    let omega2 = (rotateVec3 uv (corners.x + corners.y)) # Omega1/Omega2 get *huge* with -256 < uv < 256.

    local d = 0.0

    if ((abs (- omega1.y)) + (abs omega2.y) < width) # This might be assuming 0 < width < 1.
        d = (((corners.y / pi) - 0.5) * 2.0 * width)
    elseif (corners.y > (pi * 0.5))
        d = (max (- omega1.y) omega2.y)
    else
        d = (min (- omega1.y) omega2.y)

    d = (d + (((clamp 0.0 1.0 (corners.y / pi)) - 0.5) * 2.0 * (DimBorderBlur.w * width) + border))
    # Looks like corners.x and corners.y are both in radians.

    let d2 = (d - border + w1)
    let d3 = ((min d (omega1.x + corners.y)) + w1)

    let s = (linearstep w1 (- w1) ((min (- d0) d2) - w1))
    let alpha = (linearstep w1 (- w1) ((min (- d1) d3) - w1))
    # The first two args of these linearstep calls may or may not need to be swapped.
    # Doing w1 (- w1) gives all white, doing (- w1) w1 gives nothing.
    # Erring on the side of getting output until I figure out where the size assumptions are.

    let arcf = ((vec4 fill.rgb 1) * fill.a * s)
    let outl = ((vec4 outline.rgb 1) * outline.a * (clamp (alpha - s) 0.0 1.0))
    let output = (arcf + outl)
    output

do
    let arc
    locals;