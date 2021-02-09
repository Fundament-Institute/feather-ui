local IT = require 'std.iterator'
local Object = require 'std.object'.Object
local Array = require 'feather.array'

local Stack = Util.type_template(function(T)
	local struct s(Object) {
		array : Array(T)
	}
  
	function s.metamethods.__typename(self)
	    return "Stack("..tostring(T)..")"
	end

	terra s:reserve(n : uint) return self.array:reserve(n) end

	terra s:push(obj : T) return self.array:push(obj) end
	terra s:pop() return self.array:pop() end
  terra s:top() return self.array.data[self.array.size - 1] end

	terra s:insert(obj : T, index : uint) return self.array:insert(obj, index) end

  terra s:clear() : {} self.array:clear() end

	s.metamethods.__apply = macro(function(self,idx)
		return `self.array.data[self.array.size - 1 - idx]
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
