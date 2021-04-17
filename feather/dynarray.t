local C = require 'feather.libc'
local Util = require 'feather.util'
local Alloc = require 'std.alloc'
local IT = require 'std.iterator'
local Object = require 'std.object'.Object

local DynArray = Util.type_template(function(T)
	local struct s(Object) {
		data : &T;
		size : uint;
    capacity : uint;
	}
  s.elem = T
  
	function s.metamethods.__typename(self)
	    return "DynArray("..tostring(T)..")"
	end

  terra s:init() : {}
    self.data = nil
    self.size = 0
    self.capacity = 0
  end

  terra s:destruct() : {}
    if self.data ~= nil then
      Alloc.free(self.data)
    end
    self:init() -- this zeroes everything to prevent multiple free attempts
  end

	terra s:reserve(n : uint) : bool
    if n > self.capacity then
      var ptr = Alloc.realloc(self.data, self.capacity, n)
      if ptr == nil then
        return false
      end
      
      self.data = [&T](ptr)
      self.capacity = n
    end
    return true
  end
  
  local terra checkcapacity(self : &s, n : uint) : bool 
    if n > self.capacity then return self:reserve(n * 2) end
    return true
  end

  terra s:resize(n : uint) : bool
    if not checkcapacity(self, n) then 
      return false
    end
    -- TODO: if size increases, construct objects, if it decreases, destruct objects
    self.size = n
  end

  terra s:iter() : IT.FromSlice(T)
    return [IT.FromSlice(T)]{self.data, self.data + self.size}
  end

	terra s:add(obj : T) : bool
    if not checkcapacity(self, self.size + 1) then 
      return false
    end
    self.data[self.size] = obj
    self.size = self.size + 1
	end

  s.methods.push = s.methods.add
  terra s:pop()
    self.size = self.size - 1
    return self.data[self.size]
  end

	terra s:insert(obj : T, index : uint) : bool
    if index >= self.size then
      return self:add(obj)
    end
    if not checkcapacity(self, self.size + 1) then 
      return false
    end

    C.memmove(self.data + index + 1, self.data + index, (self.size - index) * sizeof(T))
    self.data[index] = obj
    self.size = self.size + 1
    return true
  end

  terra s:clear() : {}
    self.size = 0
  end

	s.metamethods.__apply = macro(function(self,idx)
		return `self.data[idx]
	end)

  s.metamethods.__for = function(iter, body)
    return quote
      for i = 0, iter.size do
        [body(`iter(i))]
      end
    end
  end

	return s
end)

return DynArray
