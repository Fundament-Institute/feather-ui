
local array = require 'feather.dynarray'
local alloc = require 'alloc'
local C = require 'libc'

local M = {}

--TODO: test optimization of storing the data in place of the data pointer if the size of the data is smaller than the size of the pointer.
local struct builder_segment {
    size: intptr
    data: rawstring
}

local terra segment_merge(a: builder_segment, b: builder_segment)
    var res: builder_segment
    res.size = a.size + b.size
    res.data = alloc.alloc(int8, res.size)
    alloc.copy(res.data, a.data, a.size)
    alloc.copy(&res.data[a.size], b.data, b.size)
    return res
end

local struct builder {
    segs: array(builder_segment)
}

M.builder = builder

terra builder:append_segment(seg: builder_segment)
    self.segs:add(seg)
    var idx = self.segs.size - 1
    while idx > 1 and self.segs(idx).size >= self.segs(idx-1).size do
        var a = self.segs:pop()
        var b = self.segs:pop()
        var res = segment_merge(a, b)
        alloc.free(a.data)
        alloc.free(b.data)
        self.segs:push(res)
        idx = idx - 1
    end
    return self
end

--TODO: special case conversions of constants to improve performance.
local function printf_conversion(type, fmt)
    return terra(val: type)
        var len = C.snprintf(nil, 0, fmt, val)
        var buff = [&int8](C.malloc(len+1))
        len = C.snprintf(buff, len+1, fmt, val)

        return [builder_segment]{len, buff}
    end
end

local printf_conversions = {
    [int] = "%d",
    [float] = "%f",
    [rawstring] = "%s",
}

local default_conversions = {
}

for k,v in pairs(printf_conversions) do
    default_conversions[k] = printf_conversion(k,v)
end


local function tostringseg(expr)
    local conversion = default_conversions[expr:gettype()]
        or expr:gettype().metamethods.__tostringseg
    if conversion ~= nil then
        return `conversion(expr)
    end
    error ("no suitable string segment conversion was available for type "..tostring(expr:gettype()))
end

builder.methods.append = macro(function(self, ...)
    local args = table.pack(...)
    for i = 0, args.n do
        self = `[self]:append_segment([tostringseg(args[i])])
    end
    return self
end)

terra builder:torawstring()
    if self.segs.size == 0 then
        var buff = alloc.alloc(int8, 1)
        buff[0] = 0
        return buff
    else
        var size: intptr
        for seg in self.segs do
            size = size + seg.size
        end
        var buffer = alloc.alloc(int8, size + 1)
        var offset: intptr = 0
        for seg in self.segs do
            alloc.copy(&buffer[offset], seg.data, seg.size)
            offset = offset + seg.size
        end
        buffer[offset] = 0
        return buffer
    end
end
