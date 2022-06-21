
using import glm
import .imageio
using import Array
using import enum
using import struct


type texture < Struct

@@ memo
inline mkTex (element)
    struct ("(texture " .. (tostring element) .. ")") < texture
        texels : (@ element)
        size : uvec2

        inline __drop(self)
            free self.texels

        fn to-image (self)
            returning imageio.image
            static-if (element != vec4)
                error "pixel format NYI"

            let img = (imageio.image (@ self.size 0) (@ self.size 1) 4)
            img.pixel_format = imageio.raw.const.SAIL_PIXEL_FORMAT_BPP32_RGBA
            img.gamma = 1
            for i in (range (img.width * img.height))
                # TODO: gamma correction
                let pixel =
                    (@ self.texels i) * (vec4 255)
                (@ (inttoptr (+ (ptrtoint img.pixels intptr) (* i 4)) (mutable@ (vector u8 4))) 0) =
                    vectorof u8
                        (@ pixel 0) as u8
                        (@ pixel 1) as u8
                        (@ pixel 2) as u8
                        (@ pixel 3) as u8
            img

spice lerp (a b x)
    """"linear interpolation between a and b by x
    let aT = ('typeof a)
    let bT = ('typeof b)
    if (aT != bT)
        error@ ('anchor a) "while compiling lerp" "NYI lerp type promotion"
    let f =
        try
            '@ aT '__lerp
        else
            error@ ('anchor a) "while compiling lerp" "type does not implement lerp"
    `(f a b x)

inline default-lerp (a b x)
    (b - a) * x + a

type+ real
    __lerp := default-lerp

type+ vec-type
    __lerp := default-lerp

inline barycentric-2d (p a b c)
    """"computes the barycentric coordinates of a point with respect to a triangle.
        Sourced from https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
    let v0 v1 v2 =
        b - a
        c - a
        p - a
    let den = (v0.x * v1.y - v1.x * v0.y)
    let v = ((v2.x * v1.y - v1.x * v2.y) / den)
    let w = ((v0.x * v2.y - v2.x * v0.y) / den)
    _
        1 - v - w
        v
        w

run-stage;

struct canvas
    pixels : (mutable@ vec3)
    zbuff : (mutable@ f32)
    size : uvec2

    fn clear (self color)
        for i in (range (self.size.x * self.size.y))
            (self.pixels @ i) = color
            (self.zbuff @ i) = 100000000000

    inline _write-pixel (self pos col z)
        let offs = (pos.x + pos.y * self.size.x)
        (self.pixels @ offs) = col
        (self.zbuff @ offs) = z

    inline triangle (self a b c fragment-shader)
        #inline loop-region (y-top y-bottom x-start x-stop slope-start slope-stop)
            for y in (range (round y-top) (round y-bottom))
                for x in (range (round (x-start + slope-start * (y - y-start))) (round (x-stop + slope-stop * (y - y-start))))
                    fragment-shader
        # do this the bad way at first.
        for y in (range self.size.y)
            for x in (range self.size.x)
                let u v w =
                    barycentric-2d (vec2 x y) a.pos.xy b.pos.xy c.pos.xy
                if
                    and
                        u <= 1
                        u >= 0
                        v <= 1
                        v >= 0
                        w <= 1
                        w >= 0
                    let point =
                        lerp
                            lerp
                                a
                                b
                                v / (v + u)
                            c
                            w
                    if ( (@ self.zbuff (x + self.size.x * y)) > point.pos.z)
                        let col = (fragment-shader point)
                        # ('@ self.tex x y) = (vec4 (col imply vec3) point.pos.z)
                        # _write-pixel self (uvec2 x y) (vec3 1 1 1) 0
                        _write-pixel self (uvec2 x y) col point.pos.z
    inline stencil



type+ texture
    fn from-image (img)
        if (img.pixel_format != imageio.raw.const.SAIL_PIXEL_FORMAT_BPP32_RGBA)
            error "pixel format NYI"
        (mkTex vec4)
            size = (uvec2 img.width img.height)
            texels =
                do
                    let data = (malloc-array vec4 (img.width * img.height))
                    for x in (range img.width)
                        for y in (range img.height)
                            let pixel =
                                @ (inttoptr (+ (ptrtoint img.pixels intptr) (* y img.bytes_per_line) (* x 4)) (@ (vector u8 4))) 0
                            let fpixel =
                                vec4
                                    @ pixel 0
                                    @ pixel 1
                                    @ pixel 2
                                    @ pixel 3
                            # TODO: gamma correction
                            (@ data (x + (img.width * y))) =
                                fpixel / (vec4 255)

                    data
    inline from-shader (width height typ f)
        (mkTex typ)
            size = (uvec2 width height)
            texels =
                do
                    let data = (malloc-array typ (width * height))
                    for x in (range width)
                        for y in (range height)
                            (@ data (x + width * y)) = (f x y)
                    data

    inline from-fill (width height element)
        (mkTex (typeof element))
            size = (uvec2 width height)
            texels =
                do
                    let data = (malloc-array (typeof element) (width * height))
                    for i in (range (width * height))
                        (@ data i) = element
                    data

    inline from-drawing (width height typ draw)
        let size = (width * height)
        let pixels = (malloc-array vec3 size)
        let zbuff = (malloc-array f32 size)
        let cnv =
            canvas
                pixels = pixels
                zbuff = zbuff
                size = (uvec2 width height)
        draw cnv
        free zbuff
        (mkTex vec3)
            texels = pixels
            size = (uvec2 width height)
    inline map (self typ f)
        (mkTex typ)
            size = self.size
            texels =
                do
                    let data = (malloc-array typ (self.size.x * self.size.y))
                    for i in (range (self.size.x * self.size.y))
                        (@ data i) = (f (@ self.texels i))
                    data

    fn @ (self x y)
        x as:= u32
        y as:= u32
        assert (x < self.size.x) "x coordinate out of bounds"
        assert (y < self.size.y) "y coordinate out of bounds"
        @ self.texels (x + y * self.size.x)

inline alpha-over (under under-alpha over over-alpha)
    let alpha = (over-alpha + under-alpha * (1 - over-alpha))
    _
        (over * over-alpha + under * under-alpha * (1 - over-alpha)) / alpha
        alpha
do
    let texture lerp

    locals;
