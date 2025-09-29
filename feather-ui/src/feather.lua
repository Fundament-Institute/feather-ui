-- SPDX-License-Identifier: Apache-2.0
-- SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

local function rawdump(t, spaces)
	if t == nil then return "nil" end
	if type(t) ~= "table" then return tostring(t) end

	local mt = getmetatable(t)
	setmetatable(t, nil)
	spaces = spaces or 0
	local s = "" --"metatable: " .. tostring(mt and mt.name)
	for k, v in pairs(t) do
		s = s .. "\n" .. string.rep(" ", spaces) .. "  " .. tostring(k) .. ": "
		if type(v) == "table" then
			s = s .. rawdump(v, spaces + 2)
		else
			s = s .. tostring(v)
		end
	end

	setmetatable(t, mt)
	return s
end

local memo_mt = { __mode = "k" }
local memo_nan_tag = {}
local memo_nil_tag = {}
local memo_end_tag = {}

---@param t table
---@param thismemo table
local function memo_inner(t, thismemo)
	local mt = getmetatable(t) or memo_nil_tag
	local nextmemo = thismemo[mt]
	if not nextmemo then
		nextmemo = setmetatable({}, memo_mt)
		thismemo[mt] = nextmemo
	end
	thismemo = nextmemo

	for k, v in pairs(t) do
		local nextmemo = thismemo[k]
		if not nextmemo then
			nextmemo = setmetatable({}, memo_mt)
			thismemo[k] = nextmemo
		end
		thismemo = nextmemo

		if type(v) == "table" then
			memo_inner(v, thismemo)
		else
			if v ~= v then -- This is a NaN number, so we have to substitute it
				v = memo_nan_tag
			elseif v == nil then
				v = memo_nil_tag
			end
			local nextmemo = thismemo[v]
			if not nextmemo then
				nextmemo = setmetatable({}, memo_mt)
				thismemo[v] = nextmemo
			end
			thismemo = nextmemo
		end
	end
end

---cache a function's outputs to ensure purity with respect to identity
---@generic F: function
---@param fn F
---@return F
local function memoize(fn)
	local memotab = setmetatable({}, memo_mt)

	local function wrapfn(args)
		local thismemo = memotab
		memo_inner(args, thismemo)
		if not thismemo[memo_end_tag] then
			thismemo[memo_end_tag] = table.pack(fn(args))
		else
			print("memoized: ", table.unpack(thismemo[memo_end_tag]), fn(args))
		end
		local values = thismemo[memo_end_tag]
		return table.unpack(values, 1, values.n)
	end
	return wrapfn
end

--- Creates a string representation of a unified point
---@param px number?
---@param dp number?
---@param rel number?
---@return string
local function ucoord(px, dp, rel)
	if not px and not dp and not rel then return "0" end

	local dp_suffix = ""
	if px then dp_suffix = "dp" end
	local output = {}

	if px then table.insert(output, tostring(px) .. "px") end
	if dp then table.insert(output, tostring(dp) .. dp_suffix) end
	if rel then table.insert(output, tostring(rel * 100) .. "%") end

	return table.concat(output, "/")
end

local function uvalue(v, kind)
	if kind == "px" then
		return tostring(v) .. "px"
	elseif kind == "dp" then
		return tostring(v)
	elseif kind == "rel" then
		return tostring(v * 100) .. "%"
	end

	error("Unknown kind " .. tostring(kind))
end

---@class Size : table
---@field w number?
---@field h number?
local size_mt = { name = "size_mt" }

---@class PxSize : Size
local pxsize_mt
---@class AbsSize : Size
local dpsize_mt
---@class RelSize : Size
local relsize_mt

---@class Point : table
---@field x number?
---@field y number?
local point_mt = { name = "point_mt" }

---@class PxPoint : Point
local pxpoint_mt
---@class AbsPoint : Point
local abspoint_mt
---@class RelPoint : Point
local relpoint_mt

---@class Rect : table
---@field left number?
---@field top number?
---@field right number?
---@field bottom number?
local rect_mt = { name = "rect_mt" }

---@class PxRect : Rect
---@field topleft fun(self) : PxPoint
---@field bottomright fun(self) : PxPoint
local pxrect_mt

---@class AbsRect : Rect
---@field topleft fun(self) : AbsPoint
---@field bottomright fun(self) : AbsPoint
local dprect_mt

---@class RelRect : Rect
---@field topleft fun(self) : RelPoint
---@field bottomright fun(self) : RelPoint
local relrect_mt

local rect_lookup

--- Represents a Rust DValue.
---@class Value : table
---@field px number?
---@field dp number?
---@field rel number?
local value_mt = { name = "value_mt" }

---@class Coord : table
---@field px PxPoint
---@field dp AbsPoint
---@field rel RelPoint
local coord_mt = { name = "coord_mt" }

--- Represents a Rust DRect
---@class Area : table
---@field px PxRect
---@field dp AbsRect
---@field rel RelRect
local area_mt = { name = "area_mt" }

---@param self Value
---@return string?, number?
local function value_kind(self)
	if self.px and not self.dp and not self.rel then return "px", self.px end
	if not self.px and self.dp and not self.rel then return "dp", self.dp end
	if not self.px and not self.dp and self.rel then return "rel", self.rel end

	return nil
end

-- Transforms a specialization like RelPoint or PxRect into the unified Coord or Area
local function promote(v)
	local mt = getmetatable(v)
	local t = {}
	t[mt.kind] = v

	if getmetatable(mt) == point_mt then
		return setmetatable(t, coord_mt)
	elseif getmetatable(mt) == rect_mt then
		return setmetatable(t, area_mt)
	elseif mt == coord_mt or mt == area_mt then
		return v
	end

	error("Unable to promote " .. tostring(v))
end
local function table_equality(l, r)
	local keys = {}
	for k, lv in pairs(l) do
		if lv ~= r[k] then return false end
	end

	for k, _ in pairs(r) do
		if l[k] == nil then return false end
	end

	return true
end

-- Creates a new table using negated elements from t
---@param t table
---@return table
local function negate_table(t)
	local keys = { "px", "dp", "rel", "x", "y", "left", "top", "right", "bottom", "w", "h" }
	local out = {}
	for _, key in ipairs(keys) do
		if t[key] then out[key] = -t[key] end
	end
	return setmetatable(out, getmetatable(t))
end

-- Merges two tables together, adding the elements if both exist
---comment
---@param l table
---@param r table
---@return table
local function merge_tables(l, r)
	local keys = { "px", "dp", "rel", "x", "y", "left", "top", "right", "bottom", "w", "h" }
	local t = {}

	for _, key in ipairs(keys) do
		if l[key] ~= nil and r[key] ~= nil then
			t[key] = l[key] + r[key]
		elseif l[key] == nil and r[key] ~= nil then
			t[key] = r[key]
		elseif l[key] ~= nil and r[key] == nil then
			t[key] = l[key]
		end
	end

	return t
end

---Merges together all sensible combinations of numbers, values, coords, and areas.
---@param l number | Value | PxPoint | AbsPoint | RelPoint | Coord | PxRect | AbsRect | RelRect | Area | PxSize | AbsSize | RelSize
---@param r number | Value | PxPoint | AbsPoint | RelPoint | Coord | PxRect | AbsRect | RelRect | Area | PxSize | AbsSize | RelSize
---@param swapped boolean?
---@return number | Value | PxPoint | AbsPoint | RelPoint | Coord | PxRect | AbsRect | RelRect | Area | PxSize | AbsSize | RelSize
local function merge(l, r, swapped)
	if type(l) == "number" and type(r) == "number" then return l + r end

	-- First we check for identical matching kinds for value, point_mt and rect_mt
	local lmt = getmetatable(l)
	local rmt = getmetatable(r)
	if lmt == rmt then
		---@cast l -number
		---@cast r -number
		return setmetatable(merge_tables(l, r), lmt)
	end

	-- All left-handed number combinations
	if type(lmt) == "number" then
		if rmt == value_mt then
			---@cast r Value
			local kind, v = value_kind(r)
			if kind == nil then error("Can't add number to multi-kinded value type!") end
			local t = {}
			t[kind] = v + r
			return setmetatable(t, value_mt)
		elseif getmetatable(rmt) == point_mt then
			return setmetatable({ x = r.x + l, y = r.y + l }, point_mt)
		elseif getmetatable(rmt) == size_mt then
			return setmetatable({ w = r.w + l, h = r.h + l }, size_mt)
		elseif getmetatable(rmt) == rect_mt then
			return setmetatable(
				{ left = r.left + l, top = r.top + l, right = r.right + l, bottom = r.bottom + l },
				rect_mt
			)
		elseif rmt == coord_mt or rmt == area_mt then
			error("Cannot add number to multi-kind type! (Impossible to know if you wanted absolute or relative)")
		end
	end

	-- All left-handed value combinations
	if lmt == value_mt then
		-- rmt being value_mt should have already been taken care of
		---@cast l Value
		local kind, v = value_kind(l)
		if getmetatable(rmt) == point_mt and rmt.kind == kind then
			return merge(setmetatable({ x = v, y = v }, rmt), r)
		elseif getmetatable(rmt) == size_mt and rmt.kind == kind then
			return merge(setmetatable({ w = v, h = v }, rmt), r)
		elseif getmetatable(rmt) == point_mt or rmt == coord_mt then
			local coord = setmetatable({}, coord_mt)
			if l.px then coord.px = setmetatable({ x = l.px, y = l.px }, pxpoint_mt) end
			if l.dp then coord.dp = setmetatable({ x = l.dp, y = l.dp }, abspoint_mt) end
			if l.rel then coord.rel = setmetatable({ x = l.rel, y = l.rel }, relpoint_mt) end

			return merge(coord, promote(r))
		elseif getmetatable(rmt) == rect_mt and rmt.kind == kind then
			return merge(setmetatable({ left = v, top = v, right = v, bottom = v }, rmt), r)
		elseif getmetatable(rmt) == rect_mt or rmt == area_mt then
			local rect = setmetatable({}, area_mt)
			if l.px then rect.px = setmetatable({ left = l.px, top = l.px, right = l.px, bottom = l.px }, pxrect_mt) end
			if l.dp then rect.dp = setmetatable({ left = l.dp, top = l.dp, right = l.dp, bottom = l.dp }, dprect_mt) end
			if l.rel then
				rect.rel = setmetatable({ left = l.rel, top = l.rel, right = l.rel, bottom = l.rel }, relrect_mt)
			end

			return merge(rect, promote(r))
		end
	end

	-- All left-handed point combinations
	if getmetatable(lmt) == point_mt then
		if getmetatable(rmt) == point_mt or rmt == coord_mt then
			return merge(promote(l), promote(r))
		elseif getmetatable(rmt) == size_mt and rmt.kind == lmt.kind then
			return setmetatable({ left = l.x, top = l.y, right = l.x + r.w, bottom = l.y + r.h }, rect_lookup[lmt.kind])
		elseif getmetatable(rmt) == rect_mt and rmt.kind == lmt.kind then
			return merge(setmetatable({ left = l.x, top = l.y, right = l.x, bottom = l.y }, rmt), r)
		elseif getmetatable(rmt) == rect_mt or rmt == area_mt then
			local rect = setmetatable({ left = l.x, top = l.y, right = l.x, bottom = l.y }, rect_lookup[lmt.kind])
			return merge(promote(rect), promote(r))
		end
	end

	-- All left-handed coordinate combinations
	if lmt == coord_mt then
		if getmetatable(rmt) == rect_mt or rmt == area_mt then
			local rect = setmetatable({}, area_mt)
			if l.px then
				rect.px = setmetatable({ left = l.px.x, top = l.px.y, right = l.px.x, bottom = l.px.y }, pxrect_mt)
			end
			if l.dp then
				rect.dp = setmetatable({ left = l.dp.x, top = l.dp.y, right = l.dp.x, bottom = l.dp.y }, dprect_mt)
			end
			if l.rel then
				rect.rel =
					setmetatable({ left = l.rel.x, top = l.rel.y, right = l.rel.x, bottom = l.rel.y }, relrect_mt)
			end

			return merge(rect, promote(r))
		end
	end

	-- Size and rect combination
	if getmetatable(lmt) == size_mt and getmetatable(rmt) == rect_mt then
		return setmetatable({ left = r.left, top = r.top, right = r.right + l.w, bottom = r.bottom + l.h }, rmt)
	end

	-- All left-handed rect or area combinations
	if getmetatable(lmt) == rect_mt or lmt == area_mt then
		if getmetatable(rmt) == rect_mt or rmt == area_mt then return merge(promote(l), promote(r)) end
	end

	-- Now, if we haven't yet tried swapping the arguments, we call merge again with the args swapped and the swap flag set
	if not swapped then return merge(r, l, true) end

	error("Could not merge " .. rawdump(l) .. " with " .. rawdump(r))
end

---@param self Value
---@return string
function value_mt.__tostring(self) return ucoord(self.px, self.dp, self.rel) end

value_mt.__eq = table_equality
value_mt.__add = merge
value_mt.__unm = negate_table

---@param l Value
---@param r number | Value | Point | Rect
function value_mt.__sub(l, r) return merge(l, -r) end

---@param l Value
---@param r number | Value
---@return Value
function value_mt.__mul(l, r)
	local kind, v = value_kind(l)
	local t = {}
	if kind and type(r) == "number" then
		t[kind] = v * r
		return setmetatable(t, value_mt)
	end

	if kind and getmetatable(r) == value_mt then
		---@cast r Value
		local rkind, rvalue = value_kind(r)
		if kind == rkind then
			t[kind] = v * rvalue
			return setmetatable(t, value_mt)
		end
	end

	error("Can only multiply a value by a number or another value of the same kind.")
end

---@param l Value
---@param r number | Value
---@return Value
function value_mt.__div(l, r)
	local kind, v = value_kind(l)
	local t = {}
	if kind and type(r) == "number" then
		t[kind] = v / r
		return setmetatable(t, value_mt)
	end

	if kind and getmetatable(r) == value_mt then
		---@cast r Value
		local rkind, rvalue = value_kind(r)
		if kind == rkind then
			t[kind] = v / rvalue
			return setmetatable(t, value_mt)
		end
	end

	error("Can only divide a value by a number or another value of the same kind.")
end

---@param kind string
---@return table
local function genpoint_mt(kind, name)
	local mt = setmetatable({
		name = name,
		kind = kind,
		__eq = table_equality,
		__add = merge,
		__sub = function(l, r) return merge(l, -r) end,
		__unm = negate_table,
	}, point_mt)

	---@param self Point
	---@return string
	function mt.__tostring(self) return string.format("[%s %s]", uvalue(self.x, kind), uvalue(self.y, kind)) end

	---@param l Point
	---@param r number | Point
	---@return Point
	function mt.__mul(l, r)
		if type(r) == "number" then
			return setmetatable({ x = l.x * r, y = l.y * r }, mt)
		elseif getmetatable(getmetatable(r)) == point_mt and kind == getmetatable(r).kind then
			return setmetatable({ x = l.x * r.x, y = l.y * r.y }, mt)
		else
			error("Can only multiply a point by a number or another point of the same kind.")
		end
	end

	---@param l Point
	---@param r number | Point
	---@return Point
	function mt.__div(l, r)
		if type(r) == "number" then
			return setmetatable({ x = l.x / r, y = l.y / r }, mt)
		elseif getmetatable(getmetatable(r)) == point_mt and kind == getmetatable(r).kind then
			return setmetatable({ x = l.x / r.x, y = l.y / r.y }, mt)
		else
			error("Can only divide a point by a number or another point of the same kind.")
		end
	end

	return mt
end

pxpoint_mt = genpoint_mt("px", "pxpoint_mt")
abspoint_mt = genpoint_mt("dp", "abspoint_mt")
relpoint_mt = genpoint_mt("rel", "relpoint_mt")

local point_lookup = {
	px = pxpoint_mt,
	dp = abspoint_mt,
	rel = relpoint_mt,
}

---@param kind string
---@return table
local function gensize_mt(kind, name)
	local mt = setmetatable({
		name = name,
		kind = kind,
		__eq = table_equality,
		__add = merge,
		__sub = function(l, r) return merge(l, -r) end,
		__unm = negate_table,
	}, size_mt)

	---@param self Point
	---@return string
	function mt.__tostring(self) return string.format("[%s %s]", uvalue(self.x, kind), uvalue(self.y, kind)) end

	---@param l Point
	---@param r number | Point
	---@return Point
	function mt.__mul(l, r)
		if type(r) == "number" then
			return setmetatable({ x = l.x * r, y = l.y * r }, mt)
		elseif getmetatable(getmetatable(r)) == size_mt and kind == getmetatable(r).kind then
			return setmetatable({ x = l.x * r.x, y = l.y * r.y }, mt)
		else
			error("Can only multiply a size by a number or another size of the same kind.")
		end
	end

	---@param l Point
	---@param r number | Point
	---@return Point
	function mt.__div(l, r)
		if type(r) == "number" then
			return setmetatable({ x = l.x / r, y = l.y / r }, mt)
		elseif getmetatable(getmetatable(r)) == size_mt and kind == getmetatable(r).kind then
			return setmetatable({ x = l.x / r.x, y = l.y / r.y }, mt)
		else
			error("Can only divide a size by a number or another size of the same kind.")
		end
	end

	return mt
end

pxsize_mt = gensize_mt("px", "pxsize_mt")
dpsize_mt = gensize_mt("dp", "dpsize_mt")
relsize_mt = gensize_mt("rel", "relsize_mt")

local size_lookup = {
	px = pxsize_mt,
	dp = dpsize_mt,
	rel = relsize_mt,
}

local function genrect_mt(kind, name)
	local mt = setmetatable({
		name = name,
		kind = kind,
		__eq = table_equality,
		__add = merge,
		__sub = function(l, r) return merge(l, -r) end,
		__unm = negate_table,
	}, rect_mt)

	---@param self Rect
	---@return string
	function mt.__tostring(self)
		return string.format(
			"[%s %s %s %s]",
			uvalue(self.left, kind),
			uvalue(self.top, kind),
			uvalue(self.right, kind),
			uvalue(self.bottom, kind)
		)
	end

	---@param l Rect
	---@param r number | Rect
	---@return Rect
	function mt.__mul(l, r)
		if type(r) == "number" then
			return setmetatable({ left = l.left * r, top = l.top * r, right = l.right * r, bottom = l.bottom * r }, mt)
		elseif getmetatable(getmetatable(r)) == rect_mt and kind == getmetatable(r).kind then
			return setmetatable({
				left = l.left * r.left,
				top = l.top * r.top,
				right = l.right * r.right,
				bottom = l.bottom * r.bottom,
			}, mt)
		else
			error("Can only multiply a rect by a number or another rect of the same kind.")
		end
	end

	---@param l Rect
	---@param r number | Rect
	---@return Rect
	function mt.__div(l, r)
		if type(r) == "number" then
			return setmetatable({ left = l.left / r, top = l.top / r, right = l.right / r, bottom = l.bottom / r }, mt)
		elseif getmetatable(getmetatable(r)) == rect_mt and kind == getmetatable(r).kind then
			return setmetatable({
				left = l.left / r.left,
				top = l.top / r.top,
				right = l.right / r.right,
				bottom = l.bottom / r.bottom,
			}, mt)
		else
			error("Can only divide a rect by a number or another rect of the same kind.")
		end
	end

	return mt
end

pxrect_mt = genrect_mt("px", "pxrect_mt")
dprect_mt = genrect_mt("dp", "dprect_mt")
relrect_mt = genrect_mt("rel", "relrect_mt")

rect_lookup = {
	px = pxrect_mt,
	dp = dprect_mt,
	rel = relrect_mt,
}

---@param self Area
---@return string
function area_mt.__tostring(self)
	return string.format(
		"[%s %s %s %s]",
		ucoord(self.px and self.px.left, self.dp and self.dp.left, self.rel and self.rel.left),
		ucoord(self.px and self.px.top, self.dp and self.dp.top, self.rel and self.rel.top),
		ucoord(self.px and self.px.right, self.dp and self.dp.right, self.rel and self.rel.right),
		ucoord(self.px and self.px.bottom, self.dp and self.dp.bottom, self.rel and self.rel.bottom)
	)
end

area_mt.__eq = table_equality
area_mt.__add = merge
area_mt.__sub = function(l, r) return merge(l, -r) end
area_mt.__unm = negate_table

---@param self Coord
---@return string
function coord_mt.__tostring(self)
	return string.format(
		"[%s %s]",
		ucoord(self.px and self.px.x, self.dp and self.dp.x, self.rel and self.rel.x),
		ucoord(self.px and self.px.y, self.dp and self.dp.y, self.rel and self.rel.y)
	)
end

coord_mt.__eq = table_equality
coord_mt.__add = merge
coord_mt.__sub = function(l, r) return merge(l, -r) end
coord_mt.__unm = negate_table

---@class UnifiedPx
---@field left fun(v: number) : PxRect
---@field top fun(v: number) : PxRect
---@field right fun(v: number) : PxRect
---@field bottom fun(v: number) : PxRect
---@field x fun(v: number) : PxPoint
---@field y fun(v: number) : PxPoint
---@field size fun(v: number, v:number) : PxSize
---@overload fun(l: number, t: number?, r: number?, b: number?) : PxRect | PxPoint
---@operator add(Rect): Area
---@operator add(Point): Area
---@operator sub(Rect): Area
---@operator sub(Point): Area

---@class UnifiedAbs
---@field left fun(v: number) : AbsRect
---@field top fun(v: number) : AbsRect
---@field right fun(v: number) : AbsRect
---@field bottom fun(v: number) : AbsRect
---@field x fun(v: number) : AbsPoint
---@field y fun(v: number) : AbsPoint
---@field size fun(v: number, v:number) : AbsSize
---@overload fun(l: number, t: number?, r: number?, b: number?) : AbsRect | AbsPoint
---@operator add(Rect): Area
---@operator add(Point): Area
---@operator sub(Rect): Area
---@operator sub(Point): Area

---@class UnifiedRel
---@field left fun(v: number) : RelRect
---@field top fun(v: number) : RelRect
---@field right fun(v: number) : RelRect
---@field bottom fun(v: number) : RelRect
---@field x fun(v: number) : RelPoint
---@field y fun(v: number) : RelPoint
---@field size fun(v: number, v:number) : RelSize
---@overload fun(l: number, t: number?, r: number?, b: number?) : RelRect | RelPoint
---@operator add(Rect): Area
---@operator add(Point): Area
---@operator sub(Rect): Area
---@operator sub(Point): Area

---@param kind "px" | "dp" | "rel"
---@return UnifiedPx | UnifiedAbs | UnifiedRel
local function gen_unified(kind)
	---@param lx number
	---@param ty number?
	---@param r number?
	---@param b number?
	---@return Point | Rect
	local gen = function(self, lx, ty, r, b)
		if not ty and not r and not b then
			local v = {}
			v[kind] = lx
			return setmetatable(v, value_mt)
		end
		if not r and not b then return setmetatable({ x = lx, y = ty }, point_lookup[kind]) end
		if not lx or not ty or not r or not b then
			error(
				string.format(
					"Invalid arguments passed to gen function. Must be exactly 1, 2, or 4 arguments. Found: %s %s %s %s",
					tostring(lx),
					tostring(ty),
					tostring(r),
					tostring(b)
				)
			)
		end
		return setmetatable({ left = lx, top = ty, right = r, bottom = b }, rect_lookup[kind])
	end

	return setmetatable({
		x = function(x) return gen(nil, x, 0) end,
		y = function(y) return gen(nil, 0, y) end,
		left = function(l) return gen(nil, l, 0, 0, 0) end,
		top = function(t) return gen(nil, 0, t, 0, 0) end,
		right = function(r) return gen(nil, 0, 0, r, 0) end,
		bottom = function(b) return gen(nil, 0, 0, 0, b) end,
		size = function(w, h) return setmetatable({ w = w, h = h or w }, size_lookup[kind]) end,
		zero = gen(nil, 0, 0),
		one = gen(nil, 1, 1),
	}, { __call = gen, kind = kind })
end

local px = gen_unified("px")
local dp = gen_unified("dp")
local rel = gen_unified("rel")
local printTrace = false

local ID = {}
do
	local IdNode
	local IdCount = -1

	function ID.next()
		IdCount = IdCount + 1
		return create_id(IdNode, IdCount) -- will error if no parent ID node is available
	end

	function ID.named(name)
		return create_id(IdNode, name) -- will error if no parent ID node is available
	end

	local function exitscope(popIdNode, popIdCount, ...)
		IdNode, IdCount = popIdNode, popIdCount
		return ...
	end
	function ID.enter(id, body, ...)
		local pushIdNode, pushIdCount = IdNode, IdCount
		IdNode, IdCount = id, -1
		return exitscope(pushIdNode, pushIdCount, body(...)) -- errors within body may skip scope exit handling
	end

	---@generic T1, T2
	---@param body fun(...: T2) : T1
	---@param ... T2
	---@return T1
	function ID.child(body, ...)
		local nextid = ID.next()
		return ID.enter(nextid, body, ...)
	end

	function ID.iter(name, iterator, state, init)
		if type(name) ~= "string" then
			name, iterator, state, init = nil, name, iterator, state
		end

		local pushIdNode, pushIdCount, parentNode
		local function handleLast(id, ...)
			if rawequal(nil, id) then
				IdNode, IdCount = pushIdNode, pushIdCount
			else
				IdNode, IdCount = create_id(parentNode, id), -1
			end
			return id, ...
		end

		-- Just starting loop; enter block
		-- assumes that no id-relevant stuff happens between this function and the iteration.
		pushIdNode, pushIdCount = IdNode, IdCount
		if name then
			IdNode, IdCount = ID.named(name), -1
		else
			IdNode, IdCount = ID.next(), -1
		end
		parentNode = IdNode
		return function(state, id) return handleLast(iterator(state, id)) end, state, init
	end

	function ID.wrap_child(body, name)
		return function(...) return ID.child(body, ...) end
	end

	function ID.push(newnode, newidx)
		local pushIdNode, pushIdCount = IdNode, IdCount
		IdNode, IdCount = newnode, newidx or -1
		return pushIdNode, pushIdCount
	end
	function ID.pop(oldnode, oldidx)
		IdNode, IdCount = oldnode, oldidx
	end
end

-- "required" marker table
local required = setmetatable({}, {})
local optional = setmetatable({}, {})

--- Validates a set of attributes against a definition and returns the args replaced
--- by default values if validation passes, or raises an error if a failure happens.
---@generic T
---@param args T
---@param def table
---@return T
local function validate_def(args, def)
	for k, v in pairs(args) do
		if def[k] == nil then error("Unexpected attribute found in " .. tostring(k)) end
	end

	local attrs = {}
	for k, v in pairs(def) do
		if args[k] == nil then
			if v == required then error("Missing required argument " .. tostring(k)) end
			if v ~= optional then attrs[k] = v end
		else
			attrs[k] = args[k]
		end
	end

	return setmetatable(attrs, {
		__index = function(t, k)
			if def[k] ~= optional then error("Tried to access invalid arg '" .. k .. "'") end
		end,
	})
end

---@alias attr_func fun(attrs: table):table

--- Creates a custom component with a given set of arguments. `def` represents
--- the parameters accepted by the component function, and provides a default
--- value for the parameter if it isn't required. The returned function validates
--- arguments passed to the function, enters a new ID context, then executes the
--- body and returns the result.
---@generic T
---@param def table
---@return fun(gen: fun(body: T) : table) : fun(attrs: T) : table
local function component(def)
	return function(gen)
		return function(attrs) return ID.child(gen, validate_def(attrs, def)) end
	end
end

local function exit_wrapped_fun(name, ...)
	if name and printTrace then print("leaving ID-sensitive function " .. name) end
	return ...
end

local function wrap_create(f, name)
	return function(body)
		if name and printTrace then print("inside ID-sensitive function " .. name) end
		return exit_wrapped_fun(name, f(ID.next(), body))
	end
end

local multicomponent_mt = { kind = "multicomponent_mt" }

---@generic K, V
---@param name string?
---@param f fun(k:K, v:V) : table | fun(i: integer) : table
---@param iternext fun(a: any, b: any) : ...
---@param iterstate any
---@param itermut any
---@return ...
local function each(name, f, iternext, iterstate, itermut)
	if type(name) ~= "string" then
		name, f, iternext, iterstate, itermut = nil, name, f, iternext, iterstate
	end
	---@cast f -nil

	local r = setmetatable({}, multicomponent_mt)

	do
		local _iter, _s, _var = ID.iter(name, iternext, iterstate, itermut)
		while true do
			local var = { _iter(_s, _var) }
			_var = var[1]

			if _var == nil then break end

			local c = f(table.unpack(var))
			if c == nil then error("function passed to each returned nil on " .. tostring(var[1])) end
			table.insert(r, c)
		end
	end

	return r
end

---@param max integer
---@param curr integer
---@return integer?
local function nextint(max, curr)
	if curr >= max then
		return nil
	else
		return curr + 1
	end
end

---@param start integer
---@param stop integer
---@return fun(max: integer, curr: integer) : integer?
---@return integer
---@return integer
local function range(start, stop) return nextint, stop, start - 1 end

local cond_methods = {}
function cond_methods:Elseif(flag)
	if #self.conditions ~= #self.consequences then error("mismatched conditions") end
	table.insert(self.conditions, flag)
	self.pushIdNode, self.pushIdCount = ID.push(ID.next())
	return self
end
function cond_methods:Then(consequent)
	if #self.conditions ~= (#self.consequences + 1) then error("mismatched conditions") end
	table.insert(self.consequences, consequent)
	ID.pop(self.pushIdNode, self.pushIdCount)
	return self
end
function cond_methods:Else(alternate)
	if #self.conditions ~= #self.consequences then error("mismatched conditions") end
	table.insert(self.conditions, true)
	table.insert(self.consequences, alternate)
	ID.pop(self.pushIdNode, self.pushIdCount)
	return self
end
function cond_methods:End()
	if #self.conditions ~= #self.consequences then error("mismatched conditions") end
	ID.pop(self.rootnode, self.rootcount)
	for i = 1, #self.conditions do
		if self.conditions[i] then
			if type(self.consequences[i]) == "function" then
				return self.consequences[i]()
			else
				return self.consequences[i]
			end
		end
	end
end
local cond_mt = { __index = cond_methods }
local function cond(flag)
	local self = setmetatable({ conditions = { flag }, consequences = {} }, cond_mt)
	self.rootnode, self.rootcount = ID.push(ID.next())
	self.pushIdNode, self.pushIdCount = ID.push(ID.next())
	return self
end

---@generic T
---@param name string
---@param f fun(dispatch: integer, state: T) : T, boolean?
local function add_handler(name, f) error("called placeholder function!") end

---@cast px UnifiedPx
---@cast dp UnifiedAbs
---@cast rel UnifiedRel

return {
	px = px,
	abs = dp,
	rel = rel,
	FILL = rel(0, 0, 1, 1),
	NONE = 0 / 0,
	UNSIZED = rel(0, 0, 0 / 0, 0 / 0),
	Wrap = { None = 0, Word = 1, Character = 2, Any = 3 },
	Direction = { LeftToRight = 0, RightToLeft = 1, BottomToTop = 2, TopToBottom = 3 },
	Justify = {
		Start = 0,
		Center = 1,
		End = 2,
		SpaceBetween = 3,
		SpaceAround = 4,
		SpaceFull = 5,
	},
	Align = {
		Left = 0,
		Right = 1,
		Center = 2,
		Justified = 3,
		End = 4,
	},
	FontStyle = { Normal = 0, Italic = 1, Oblique = 2 },
	component = component,
	required = required,
	optional = optional,
	button = wrap_create(create_button, "create_button"),
	domain = {
		line = wrap_create(create_domain_line, "create_domain_line"),
		point = wrap_create(create_domain_point, "create_domain_point"),
	},
	flexbox = wrap_create(create_flexbox, "create_flexbox"),
	gridbox = wrap_create(create_gridbox, "create_gridbox"),
	image = wrap_create(create_image, "create_image"),
	listbox = wrap_create(create_listbox, "create_listbox"),
	mousearea = wrap_create(create_mousearea, "create_mousearea"),
	region = wrap_create(create_region, "create_region"),
	scrollarea = wrap_create(create_scrollarea, "create_scrollarea"),
	shape = {
		line = wrap_create(create_line, "create_line"),
		rect = wrap_create(create_round_rect, "create_round_rect"),
		circle = wrap_create(create_circle, "create_circle"),
		arc = wrap_create(create_arc, "create_arc"),
		triangle = wrap_create(create_triangle, "create_triangle"),
	},
	text = wrap_create(create_text, "create_text"),
	textbox = wrap_create(create_textbox, "create_textbox"),
	window = wrap_create(create_window, "create_window"),
	ID = ID,
	each = each,
	range = range,
	cond = cond,
	add_handler = add_handler,
	multicomponent = function(...) return setmetatable({ ... }, multicomponent_mt) end,
}
