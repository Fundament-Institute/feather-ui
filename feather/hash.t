C = terralib.includecstring [[
	#include <stdlib.h>
  #include <string.h>
]]

local Util = require 'feather.util'

local H = {}

H.IntHash = macro(function(key) return `[uint32](key) end)
H.IntEqual = macro(function(a, b) return `a == b end)

H.Int64Hash = macro(function(key) return `[uint32](((key>>33)^key^key)<<11) end)
H.Int64Equal = macro(function(a, b) return `a == b end)

terra H.StringHash(s : rawstring) : uint32
	var h : uint32 = [uint32](s[0]);
	if h ~= 0 then
    var i = 1
    while s[i] ~= 0 do
       h = (h << 5) - h + [uint32](s[i]);
       i = i + 1
    end
  end

	return h
end

terra H.StringEqual(a : rawstring, b: rawstring) : bool return C.strcmp(a, b) == 0 end

--[[terra H.StringInsHash(s : rawstring) : uint32
  khint_t h = ((*s)>64 && (*s)<91) ? (*s) + 32 : *s;
  if(h)
    for(++s; *s; ++s)
      h = (h << 5) - h + (((*s)>64 && (*s)<91) ? (*s) + 32 : *s);
  return h;
end
terra H.StringInsEqual(a : rawstring, b: rawstring) : bool return C.stricmp(a, b) == 0 end--]]

H.Hash = Util.type_template(function(Key, Value, Hasher, Equality)
  local is_map = Value ~= nil

	local struct s {
    n_buckets : uint32;
    size : uint32;
    n_occupied : uint32;
    upper_bound : uint32;
    flags : &uint8;
		keys : &Key;
	}
  
  if is_map then 
    s.entries[#s.entries + 1] = { field = "vals", type = &Value }
  end

	function s.metamethods.__typename(self)
    if is_map then 
	    return "HashMap("..tostring(Key)..","..tostring(Value)..")"
    else
	    return "HashSet("..tostring(Key)..")"
    end
	end

  local isempty = macro(function(flag, i) return `(flag[i] and 2) end)
  local isdel = macro(function(flag, i) return `(flag[i] and 1) end)
  local iseither = macro(function(flag, i) return `(flag[i] and 3) end)
  local set_isdel_false = macro(function(flag, i) return quote flag[i]=(flag[i] and (not 1)) end end)
  local set_isempty_false = macro(function(flag, i) return quote flag[i]=(flag[i] and (not 2)) end end)
  local set_isboth_false = macro(function(flag, i) return quote flag[i] = 0 end end)
  local set_isdel_true = macro(function(flag, i) return quote flag[i] = (flag[i] or 1) end end)

  terra s:destruct() : {}
    Util.SafeFree(self.keys, self.flags, self)
    escape if is_map then emit(`Util.SafeFree(self.vals)) end end
  end

  terra s:clear() : {}
    if self.flags ~= nil then
      C.memset(self.flags, 2, self.n_buckets)
      self.size = 0
      self.n_occupied = 0
    end
  end

  terra s:get(key : Key) : uint32
		if self.n_buckets > 0 then
			var k, i, last, mask, step : uint32 = 0
			mask = self.n_buckets - 1
			k = Hasher(key)
      i = k and mask
			last = i
			while (not isempty(self.flags, i) and (isdel(self.flags, i) or not Equality(self.keys[i], key))) do
        step = step + 1
				i = (i + step) and mask
				if (i == last) then return self.n_buckets end
			end
			return terralib.select(iseither(self.flags, i), self.n_buckets, i)
		else
      return 0
    end
  end

  terra s:resize(count : uint32) : int
    var new_flags : &uint8 = nil
		var j : uint32 = 1
		do
			kroundup32(new_n_buckets)
			if new_n_buckets < 4 then new_n_buckets = 32 end

			if self.size >= [uint32](new_n_buckets * HASH_UPPER + 0.5) then
        j = 0
			else
				new_flags = [&uint8](C.malloc(new_n_buckets))
				if new_flags == nil then return -1 end

				C.memset(new_flags, 2, new_n_buckets)
				if self.n_buckets < new_n_buckets then	
					var new_keys : &Key = [&Key](C.realloc(self.keys, new_n_buckets * sizeof(Key)))
					if (new_keys == nil) then 
            kfree(new_flags) 
            return -1
          end

					self.keys = new_keys

					escape
            if is_map then
              emit(quote
                var new_vals : &Value = [&Value](C.realloc(self.vals, new_n_buckets * sizeof(Value)))
                if new_vals == nil then
                  kfree(new_flags)
                  return -1
                end
                self.vals = new_vals
              end)
            end
					end
				end 
			end
		end
		if j ~= 0 then 
      for j = 0, self.n_buckets do
				if iseither(self.flags, j) == 0 then
					var key : Key = self.keys[j]
					var new_mask : uint32 = new_n_buckets - 1
					escape
            if is_map then
              emit(quote
                var val : Value
                val = self.vals[j] 
              end)
            end
          end

					set_isdel_true(self.flags, j)
					while true do 
						var step : uint32 = 0
						var k : uint32 = __hash_func(key)
						var i : uint32 = k and new_mask
						
            while (not isempty(new_flags, i)) do
              step = step + 1
              i = (i + step) and new_mask
            end

						set_isempty_false(new_flags, i)
						if i < self.n_buckets and iseither(self.flags, i) == 0 then 
							do 
                var tmp : Key = self.keys[i]
                self.keys[i] = key
                key = tmp
              end

              escape
                if is_map then
                  emit(quote
                    var tmp : Value = self.vals[i]
                    self.vals[i] = val
                    val = tmp
                  end)
                end
              end

							set_isdel_true(self.flags, i)
						else
							self.keys[i] = key
							escape if is_map then emit(quote self.vals[i] = val end) end end
							break
						end
					end
				end
			end
			if self.n_buckets > new_n_buckets then
				self.keys = [&Key](C.realloc(self.keys, new_n_buckets * sizeof(Key)))
				escape if is_map then emit(quote self.vals = [&Value](krealloc(self.vals, new_n_buckets * sizeof(Value))) end) end end
			end
			C.free(self.flags) 
			self.flags = new_flags
			self.n_buckets = new_n_buckets
			self.n_occupied = self.size
			self.upper_bound = [uint32](self.n_buckets * HASH_UPPER + 0.5)
		end
		return 0
  end

  terra s:put(key : Key) : { uint32, int }
		var x : uint32
		if self.n_occupied >= self.upper_bound then
			if self.n_buckets > (self.size<<1) then
				if self:resize(self.n_buckets - 1) < 0 then
					return self.n_buckets, -1
				end
			elseif self:resize(self.n_buckets + 1) < 0 then 
				return self.n_buckets, -1
			end
		end

		do
      var last : uint32
      var mask : uint32 = self.n_buckets - 1
      var step : uint32 = 0
			var site : uint32 = self.n_buckets

      x = site
      var k : uint32 = Hasher(key)
      var i : uint32 = k and mask
			if isempty(self.flags, i) then
        x = i
			else
				last = i
				while (not isempty(self.flags, i) and (isdel(self.flags, i) or not Equality(self.keys[i], key))) do
					if (isdel(self.flags, i)) then site = i end

          step = step + 1
					i = (i + step) and mask
					if i == last then
            x = site
            break
          end
				end
				if x == self.n_buckets then
					if (isempty(self.flags, i) and site ~= self.n_buckets) then
            x = site
					else
            x = i
          end
				end
			end
		end

		if isempty(self.flags, x) then 
			self.keys[x] = key
			set_isboth_false(self.flags, x)
      self.size = self.size + 1
      self.n_occupied = self.n_occupied + 1
			return x, 1
		elseif (isdel(self.flags, x)) then
			self.keys[x] = key
			set_isboth_false(self.flags, x)
			self.size = self.size + 1
			return x, 2
		end

		return x, 0
  end

  terra s:del(x : uint32)
		if x ~= self.n_buckets and not iseither(self.flags, x) then
			set_isdel_true(self.flags, x)
			self.size = self.size - 1
		end
  end

  terra s:key(index : uint32) : Key
    return self.keys[index]
  end
  
  if is_map then
    terra s:value(index : uint32) : Value
      return self.vals[index]
    end
  end

	terra s:add(obj : T) : bool
    if not self:reserve(self.size + 1) then 
      return false
    end
    self.data[self.size] = obj
    self.size = self.size + 1
	end

	terra s:insert(obj : T, index : uint) : bool
    if index >= self.size then
      return add(obj)
    end
    if not self:reserve(self.size + 1) then 
      return false
    end

    C.memmove(self.data + index, self.data + index + 1, (self.size - index) * sizeof(T))
    self.data[index] = obj
    self.size = self.size + 1
    return true
  end

	terra s:free()
    if self.data ~= nil then
		  C.free(self.data)
    end
    self.size = 0
    self.capacity = 0
	end

	s.metamethods.__apply = macro(function(self,idx)
		return `self.data[idx]
	end)

	return s
end)

return H