using import glm
using import glsl
using import .shader-utils

# Lines that have not been changed are commented as unchanged. Lines that *have* been changed are marked as changed.
fn old-arc (x y size DimBorderBlur corners fill outline)
    # x, y are the current coords (`pos` in the original).
    # size is the length of the canvas (we're working with a square canvas). Not present in original.
    # DimBorderBlur, corners, fill, and outline are vec4s; in the original, they were marked `in` like GLSL globals, but here they're arguments.
    let l = ((DimBorderBlur.x + DimBorderBlur.y) * 0.5) # Unchanged.
    let uv = (((vec2 x y) * 2.0) - size) # Unchanged. Note that this makes the coordinate space have the range of -256 to 256.
    let width = 1.0 # Width of a pixel is always 1.0 on software renderer, so width is 1.0 instead of fwidth(uv.x). Changed.
    let w1 = ((1.0 + DimBorderBlur.w) * width) # Unchanged.
    
    let border = ((DimBorderBlur.z / 1) * 2.0)
    # Dividing the blur value by 1?
    let t = (0.5 - (corners.z / 1) + w1 * 1.5)
    # And corners.z, too?
    # Both of these lines are unchanged.

    let r = (l - t + w1)
    # Changed.
    # l used to be literal 1.0, but based on what I learned from circle it should be l.

    let d0 = ((abs ((length uv) - r)) - t + border)
    let d1 = ((abs ((length uv) - r)) - t)
    # These lines are unchanged.

    let omega1 = (rotateVec3 uv (corners.x - corners.y)) # Is UV supposed to be in the range 0 to 1?
    let omega2 = (rotateVec3 uv (corners.x + corners.y)) # Omega1/Omega2 get *huge* with -256 < uv < 256.
    # These lines are also unchanged.

    local d = 0.0 # Unchanged.

    if ((abs (- omega1.y)) + (abs omega2.y) < width) # This might be assuming 0 < width < 1.
        d = (((corners.y / pi) - 0.5) * 2.0 * width)
    elseif (corners.y > (pi * 0.5))
        d = (max (- omega1.y) omega2.y)
    else
        d = (min (- omega1.y) omega2.y)
    # This if/else chain is unchanged.
    

    d = (d + (((clamp 0.0 1.0 (corners.y / pi)) - 0.5) * 2.0 * (DimBorderBlur.w * width) + border))
    # Looks like corners.x and corners.y are both in radians.
    # Unchanged.

    let d2 = (d - border + w1)
    let d3 = ((min d (omega1.x + corners.y)) + w1)
    # These lines are unchanged.

    let s = (linearstep w1 (- w1) ((min (- d0) d2) - w1))
    let alpha = (linearstep w1 (- w1) ((min (- d1) d3) - w1))
    # These lines are unchanged.
    # The first two args of these linearstep calls may or may not need to be swapped.
    # Doing w1 (- w1) gives all white, doing (- w1) w1 gives nothing.
    # Erring on the side of getting output until I figure out where the size assumptions are.

    let arcf = ((vec4 fill.rgb 1) * fill.a * s)
    let outl = ((vec4 outline.rgb 1) * outline.a * (clamp (alpha - s) 0.0 1.0))
    let output = (arcf + outl)
    output
    # This is an expanded form of the original version, for clarity's sake, but is equivalent to the original. Technically unchanged.

do
    let arc
    locals;