const std = @import("std");
const cstr = std.cstr;
const mem = std.mem;

/// A point in 2D space
const Vec = extern struct {
    x: f32,
    y: f32,
};

/// A point in 3D space. Used only for layer transformations and 3D projection calculations, not in the layout.
const Vec3D = extern struct {
    x: f32,
    y: f32,
    z: f32,
};

/// Represents a rectangle by absolute placement of the top-left and bottom-right corners.
const Rect = extern struct {
    l: f32,
    t: f32,
    r: f32,
    b: f32,
};

/// Unified rect that contains both absolute and relative coordinates
const URect = extern struct {
    abs: Rect,
    rel: Rect,
};

/// Unified vector that contains both absolute and relative coordinates
const UVec = extern struct {
    abs: Vec,
    rel: Vec,
};

/// A 64-bit RGBA, 16-bit per channel color used for all rendering.
const Color = extern union {
    color: i64,
    colors: [4]i16, // rgba order
};

const LayoutFlags = packed struct {
    allowTab: bool,
    allowKeyEvents: bool,
    allowMouseEvents: bool,
    allowTouchEvents: bool,
    allowControllerEvents: bool,
    clipChildren: bool,
};

/// Standard boolean states shared by most components
const StateFlags = packed struct {
    hover: bool, // Mouse is hovering over this element0
    active: bool, // Mouse button is being held down on this element
    tabFocus: bool, // Current tabbed item
    keyFocus: bool, // Has keyboard focus (usually the same as tabFocus, but some elements reject the key focus)
    selected: bool,
    disabled: bool,
};

const WordWrapping = enum {
    NoWrap,
    WordWrap,
    LetterWrap,
};

const TextDirection = enum {
    LeftRight,
    RightLeft,
    TopBottom,
};

const FlowData = packed struct {
    wordwrap: WordWrapping,
    direction: TextDirection,
};

const JustifyFlex = enum {
    Start,
    Center,
    End,
};

const AlignFlex = enum {
    Stretch,
    Start,
    Center,
    End,
    Baseline,
};

const FlexData = packed struct {
    vertical: bool,
    reverse: bool,
    wrap: bool,
    justify: JustifyFlex,
    alignment: AlignFlex,
};

const DisplayData = packed union {
    none: void,
    flow: FlowData,
    flex: FlexData,
    custom: *c_void,
};

/// Represents an evaluated layout definition node
const LayoutNode = extern struct {
    area: URect,
    center: UVec,
    margin: Rect,
    padding: Rect,
    min: Vec,
    max: Vec,
    line_height: f32,
    font_size: f32,
    flags: LayoutFlags,
    state: StateFlags,
    children: []*LayoutNode,
    generator: fn (node: *LayoutNode) void,
    state: ?*ElementNode, // Points to the first instantiated layout node of this, if it exists
    display: fn (layout: *LayoutNode, data: *DisplayData, state: *ElementNode) Rect,
    display_data: *DisplayData,
};

const LayerNode = struct {
    layout: LayoutNode,
    transform: [4][4]f32, // This is a full 3D affine transformation matrix, including translation along x and y, which ignores the layout position
    opacity: f32, // This opacity applies to all child nodes
    mouse: enum {
        Layout, // The default is to ignore the layer's transform entirely and only do mouse collisions on the original layout area
        Transform, // You can change this so that only the transformed rect has mouse collisions (done by applying the matrix to the mouse position)
        Both, // For certain cases (like buttons floating) you may want both the transformed area and the layout area to be used to prevent flickering
    },
};

const Shadow = struct {
    offset: Vec,
    blur: f32,
    color: Color,
};

const Border = struct {
    width: f32,
    color: Color,
};

const BoxNode = struct {
    layout: LayoutNode,
    background: Color,
    border: Border,
    corners: [4]f32, // top-left, top-right, bottom-right, bottom-left
    shadow: Shadow,
};

const TextNode = struct {
    layout: LayoutNode,
    font: *c_void,
    color: Color,
    background: Color,
    shadow: Shadow,
    letter_spacing: f32,
    word_spacing: f32,
};

export const MessageTag = extern enum(u8) {
    Custom,
    Construct,
    Destroy,
    Draw,
    MouseDown,
    MouseDblClick,
    MouseUp,
    MouseOff,
    MouseMove,
    MouseScroll,
    TouchBegin,
    TouchEnd,
    TouchMove,
    KeyUp,
    KeyDown,
    KeyChar,
    JoyButtonDown,
    JoyButtonUp,
    JoyAxis,
    GotFocus,
    LostFocus,
    DragDrop,
};

const Message = union(MessageTag) {
    Custom: void,
    Construct: void,
    Destroy: void,
    Draw: void,
    MouseDown: void,
    MouseDblClick: void,
    MouseUp: void,
    MouseOff: void,
    MouseMove: void,
    MouseScroll: void,
    TouchBegin: void,
    TouchEnd: void,
    TouchMove: void,
    KeyUp: void,
    KeyDown: void,
    KeyChar: void,
    JoyButtonDown: void,
    JoyButtonUp: void,
    JoyAxis: void,
    GotFocus: void,
    LostFocus: void,
    DragDrop: void,
};

// The result is not tagged because you should know the message type you just called (and this should be exactly 64-bits)
const MessageResult = packed union {
    Custom: void,
    Construct: void,
    Destroy: void,
    Draw: void,
    MouseDown: void,
    MouseDblClick: void,
    MouseUp: void,
    MouseOff: void,
    MouseMove: void,
    MouseScroll: void,
    TouchBegin: void,
    TouchEnd: void,
    TouchMove: void,
    KeyUp: void,
    KeyDown: void,
    KeyChar: void,
    JoyButtonDown: void,
    JoyButtonUp: void,
    JoyAxis: void,
    GotFocus: void,
    LostFocus: void,
    DragDrop: void,
};

const ScrollState = struct {};
const TextState = struct {};
const AssetState = struct {};
const ButtonState = struct {};

const BehaviorState = extern union {
    scroll: *ScrollState,
    text: *TextState,
    asset: *AssetState,
    button: *ButtonState,
    custom: *c_void,
};

const BehaviorFunction = fn (msg: Message, state: BehaviorState) MessageResult;

inline fn ButtonBehavior(msg: Message, state: ButtonState) MessageResult {}

fn ButtonBehaviorWrap(msg: Message, state: BehaviorState) MessageResult {
    return ButtonBehavior(*state.button);
}

fn hash_i32(x: i32) u32 {
    return @bitCast(u32, x);
}

fn eql_i32(a: i32, b: i32) bool {
    return a == b;
}

/// Represents the current base state of a layout node in the layout tree.
const ElementNode = struct {
    area: Rect, // This is relative to the parent R-Tree node, and is not a meaningful value
    layout: Rect,
    //clip: Rect, // Might not be necessary because it's equivilant to the RTreeNode's area
    line_height: f32,
    font_size: f32,
    children: []ElementNode,
    sibling: ?*ElementNode, // Used only to track split flow nodes
    node: *RTreeNode, // The corresponding RTreeNode 
    tstate: std.HashMap(i32, u64, hash_i32, eql_i32), // timestamps mapped to a corresponding field for the last time it was changed
    behavior: BehaviorFunction,
    state: BehaviorState,
};

/// A feather GUI Schema is a function that takes a given piece of data as input and returns a layout subtree. It will
/// recursively call other GUI Schema functions to resolve child fields of the data object, such that the returned
/// subtree is a complete layout definition
/// The layout definition tree is used to generate the layout state tree, which is the current evaluated layout.
export const RTreeNode = extern struct {
    area: Rect,
    node: ?*ElementNode,
    sibling: ?*RTreeNode,
    child: ?*RTreeNode,
};

/// A render call is satisfied by iterating through the layout state tree and it's absolute clipping rectangles,
/// culling any children completely outside the render call rectangle and evaluating the rest of the nodes in a
/// pre-order traversal that calls the render function with the evaluated layout rectangle. This render function
/// itself calls various supported draw calls in the rendering engine for the current platform.
export const Cursor = extern enum(u8) {
    None,
    Arrow,
    IBeam,
    Cross,
    Wait,
    Hand,
    ResizeNS,
    ResizeWE,
    ResizeNWSE,
    ResizeNESW,
    ResizeAll,
    No,
    Help,
    Drag,
};

export const AssetFormat = extern enum(u8) {
    BMP,
    JPG,
    PNG,
    GIF,
    TIFF,
    TGA,
    DDS,
    WIC,
    SVG,
    AVI,
    MP4,
    MKV,
    WEBM,
    GET_FROM_FILE,
};

export const TextAntiAliasing = extern enum(u8) {
    None,
    AntiAliased,
    Subpixel_LCD,
    Subpixel_LCD_V,
};

const Display = extern struct {
    area: Rect,
    dpi: Vec,
    index: u32,
    main: bool,
};

export const Font = extern struct {
    internal: *c_void,
    baseline: f32,
    lineheight: f32,
    pt: u32,
    dpi: Vec,
};

export const Asset = extern struct {
    internal: *c_void,
    format: AssetFormat,
    size: Vec,
    dpi: Vec,
};

export const Backend = extern struct {
    features: packed struct {
        AntiAliasedText: bool,
        SubPixelText: bool,
        TextBlur: bool,
        TextAlpha: bool,
        Formats: @IntType(false, @memberCount(AssetFormat) - 1), // Zig won't let us use an array of bools here
        RectCorners: bool,
        RectBorder: bool,
        RectBlur: bool,
        RectAlpha: bool,
        CircleArcs: bool,
        CircleBorder: bool,
        CircleBlur: bool,
        CircleAlpha: bool,
        LinesAlpha: bool,
        CurveStroke: bool,
        CurveFill: bool,
        LayerTransform: bool,
        LayerOpacity: bool,
        BackgroundOpacity: bool,
        ImmediateMode: bool, // Does not indicate a feature, simply signifies whether this tends to redraw every frame or tries to only update dirty regions.
    },

    beginScene: extern fn (self: *Backend, dirty: [*]Rect) i32,
    drawFont: extern fn (self: *Backend, layout: *c_void, area: Rect, color: Color, lineHeight: f32, letterSpacing: f32, blur: f32, aa: TextAntiAliasing) i32, // Should also have a blur value for shadows, might not be supported
    drawAsset: extern fn (self: *Backend, asset: *c_void, area: Rect, time: f32) i32,
    drawRect: extern fn (self: *Backend, rect: Rect, corners: Rect, rectColor: Color, border: f32, borderColor: Color, blur: f32) i32,
    drawCircle: extern fn (self: *Backend, area: Rect, arcs: Rect, fillColor: Color, border: f32, borderColor: Color, blur: f32) i32,
    drawLines: extern fn (self: *Backend, points: [*]Vec, count: usize, color: Color) i32,
    drawCurve: extern fn (self: *Backend, points: [*]Vec, count: usize, stroke: f32, strokeColor: Color, fill: Color) i32,
    //drawShader: extern fn(self: *Backend, shader: *c_void, area: Rect, ...) i32,
    pushLayer: extern fn (self: *Backend, transform: [4][4]f32, opacity: f32) i32,
    popLayer: extern fn (self: *Backend) i32,
    clipRect: extern fn (self: *Backend, area: Rect) i32,
    endScene: extern fn (self: *Backend) i32,
    getDisplays: extern fn (self: *Backend, count: *usize) [*]Display,

    createFont: extern fn (self: *Backend, family: [*]const u8, weight: u16, italic: bool, pt: u32, dpi: Vec) Font,
    destroyFont: extern fn (self: *Backend, font: *c_void) void,
    fontLayout: extern fn (self: *Backend, text: [*]const u8, area: *Rect, lineHeight: f32, letterSpacing: f32, prev: ?*c_void) ?*c_void,
    fontIndex: extern fn (self: *Backend, layout: *c_void, area: Rect, lineHeight: f32, letterSpacing: f32, pos: Vec, cursor: *Vec, dpi: Vec) usize,
    fontPos: extern fn (self: *Backend, layout: *c_void, area: Rect, lineHeight: f32, letterSpacing: f32, index: usize, dpi: Vec) Vec,

    createAsset: extern fn (self: *Backend, data: [*]const u8, count: usize, format: AssetFormat) Asset,
    destroyAsset: extern fn (self: *Backend, asset: *c_void) void,

    putClipboard: extern fn (self: *Backend, kind: u32, data: [*]const u8, count: usize) bool,
    getClipboard: extern fn (self: *Backend, kind: u32, count: *usize) [*]const u8,
    checkClipboard: extern fn (self: *Backend, kind: u32) bool,
    clearClipboard: extern fn (self: *Backend, kind: u32) void,

    processMessages: extern fn (self: *Backend) bool,
    setCursor: extern fn (self: *Backend, cursor: Cursor) bool,
    destroy: extern fn (self: *Backend) void,
};

export const featherBackend = extern fn () ?*Backend;

export fn LoadBackend(path: [*]const u8, init: ?[*]const u8) ?featherBackend {
    const spath = path[0..mem.len(u8, path)];
    var lib = std.DynLib.open(spath) catch |err| return null;

    var initfunc: []const u8 = "test";
    if(init) |unwrap| {
        initfunc = unwrap[0..mem.len(u8, path)];
    }
    const addr = lib.lookup(initfunc) orelse return null;
    return @intToPtr(featherBackend, addr);
}
/// Messages, usually containing user input, are processed differently based on the class of message. Generic messages
/// are only sent to their target node, which can choose to broadcast them to children if necessary.
/// This processes a data change notification. A data change must be evaluated as either changing the layout
/// definition, or only changing the layout state. Either change requires all impacted subtrees to be re-evaluated.
