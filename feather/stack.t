local IT = require 'std.iterator'
local Object = require 'std.object'.Object
local DynArray = require 'feather.dynarray'

local Stack = Util.type_template(function(T)
	local struct s(Object) {
		base : DynArray(T)
	}
  
	function s.metamethods.__typename(self)
	    return "Stack("..tostring(T)..")"
	end

	terra s:reserve(n : uint) return self.base:reserve(n) end

	terra s:push(obj : T) return self.base:push(obj) end
	terra s:pop() return self.base:pop() end
  terra s:top() return self.base.data[self.base.size - 1] end

	terra s:insert(obj : T, index : uint) return self.base:insert(obj, index) end

  terra s:clear() : {} self.base:clear() end

	s.metamethods.__apply = macro(function(self,idx)
		return `self.base.data[self.base.size - 1 - idx]
	end)

  s.metamethods.__for = function(iter, body)
    return quote
      var it = iter
      for i = 0, it.size do
        body(it(i))
      end
    end
  end

	return s
end)

return Stack
