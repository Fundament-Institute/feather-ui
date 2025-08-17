-- SPDX-License-Identifier: Apache-2.0
-- SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

function sandbox_impl(use_weaktables)
	local proxy_origins, proxy_owners, proxy_refs, proxy_key
	if use_weaktables then
		local weak = { __mode = "k" }
		proxy_origins, proxy_owners, proxy_refs = setmetatable({}, weak), setmetatable({}, weak), setmetatable({}, weak)
	else
		proxy_key = {}
	end


	local error_kill_flag = {}

	local proxy_get

	local root_module

	local module_proxy_of_mt = { __mode = 'v' }

	if use_weaktables then
		root_module = { proxy_of = setmetatable({}, module_proxy_of_mt) }
	else
		-- can't use proxy_of without weak references
		root_module = {}
	end

	local proxy_get_origin
	if use_weaktables then
		function proxy_get_origin(obj)
			return proxy_origins[obj]
		end
	else
		function proxy_get_origin(obj)
			local info = rawget(obj, proxy_key)
			return info and info.origin
		end
	end

	local proxy_get_target
	if use_weaktables then
		function proxy_get_target(obj)
			return proxy_refs[obj]
		end
	else
		function proxy_get_target(obj)
			local info = rawget(obj, proxy_key)
			return info and info.target
		end
	end

	local proxy_get_owner
	if use_weaktables then
		function proxy_get_owner(obj)
			return proxy_owners[obj]
		end
	else
		function proxy_get_owner(obj)
			local info = rawget(obj, proxy_key)
			return info and info.owner
		end
	end

	local function translate(obj, module_src, module_dst) -- translate values across a module boundary to maintain sandboxing
		local t = type(obj)
		if t == 'string' or t == 'number' or t == 'boolean' or t == 'nil' or t == 'userdata' then
			return obj -- immutable primitives and string may be passed directly
		elseif t == "table" then
			local origin = proxy_get_origin(obj)
			if origin then
				if origin == module_dst then
					return proxy_get_target(obj)
				else
					return proxy_get(proxy_get_target(obj), origin, module_dst)
				end
			end
			local mt = getmetatable(t)
			return proxy_get(obj, module_src, module_dst)
		elseif t == "function" then
			return proxy_get(obj, module_src, module_dst)
		else
			error("NYI unsupported translation between modules for " .. t .. ", this needs to be expanded")
		end
	end

	local function translate_args_inner(module_src, module_dst, count, arg, ...)
		if count > 1 then
			return translate(arg, module_src, module_dst), translate_args_inner(module_src, module_dst, count - 1, ...)
		else
			return translate(arg, module_src, module_dst)
		end
	end
	local function translate_args(module_src, module_dst, ...) -- convenience function for translating a list of args all at once
		local count = select('#', ...)
		if count == 0 then return end
		return translate_args_inner(module_src, module_dst, count, ...)
	end

	local function cloneTab(tab)
		if tab == nil then return nil end
		local clone = {}
		for k, v in pairs(tab) do
			clone[k] = v
		end
		return clone
	end

	local function sandboxed_getmetatable(obj)
		if type(obj) == 'string' then --strings have a shared metatable, so this forbids the global mutable state
			return "string"
		else
			return getmetatable(obj)
		end
	end

	local proxy_mt = { -- metatable for proxies
		__metatable = "proxy",
		__index = function(self, k)
			local origin = proxy_get_origin(self)
			local owner = proxy_get_owner(self)
			return translate(proxy_get_target(self)[translate(k, owner, origin)], origin, owner)
		end,
		__newindex = function(self, k, v)
			local origin = proxy_get_origin(self)
			local owner = proxy_get_owner(self)
			proxy_get_target(self)[translate(k, owner, origin)] = translate(v, owner, origin)
		end,
		__add = function(self, other)
			local origin = proxy_get_origin(self)
			local owner = proxy_get_owner(self)
			return
					translate(
						proxy_get_target(self) + translate(other, owner, origin),
						origin, owner
					)
		end,
		__sub = function(self, other)
			local origin = proxy_get_origin(self)
			local owner = proxy_get_owner(self)
			return
					translate(
						proxy_get_target(self) - translate(other, owner, origin),
						origin, owner
					)
		end,
		__mul = function(self, other)
			local origin = proxy_get_origin(self)
			local owner = proxy_get_owner(self)
			return
					translate(
						proxy_get_target(self) * translate(other, owner, origin),
						origin, owner
					)
		end,
		__div = function(self, other)
			local origin = proxy_get_origin(self)
			local owner = proxy_get_owner(self)
			return
					translate(
						proxy_get_target(self) / translate(other, owner, origin),
						origin, owner
					)
		end,
		__mod = function(self, other)
			local origin = proxy_get_origin(self)
			local owner = proxy_get_owner(self)
			return
					translate(
						proxy_get_target(self) % translate(other, owner, origin),
						origin, owner
					)
		end,
		__pow = function(self, other)
			local origin = proxy_get_origin(self)
			local owner = proxy_get_owner(self)
			return
					translate(
						proxy_get_target(self) ^ translate(other, owner, origin),
						origin, owner
					)
		end,
		__unm = function(self)
			local origin = proxy_get_origin(self)
			local owner = proxy_get_owner(self)
			return
					translate(
						-proxy_get_target(self),
						origin, owner
					)
		end,
		__call = function(self, ...)
			local origin = proxy_get_origin(self)
			local owner = proxy_get_owner(self)
			return
					translate_args(origin, owner,
						proxy_get_target(self)(
							translate_args(owner, origin, ...)
						)
					)
		end,
		__pairs = function(self)
			local origin = proxy_get_origin(self)
			local owner = proxy_get_owner(self)
			return translate_args(origin, owner, pairs(proxy_get_target(self)))
		end,
		__len = function(self)
			local origin = proxy_get_origin(self)
			local owner = proxy_get_owner(self)
			return translate(#proxy_get_target(self), origin, owner)
		end,

		__tostring = function(self) return tostring(proxy_get_target(self)) end,
		__eq = function(self, other) return rawequal(proxy_get_origin(self), proxy_get_origin(other)) and
			proxy_get_target(self) == proxy_get_target(other) end
		--[[__uncall = function(self)
    return uncall(proxy_origins[self])
  end,]]
	}
	local proxy_private_mt = { -- metatable for proxies that hide their state
		__metatable = "proxy",
		__index = function(self, k)
			local origin = proxy_get_origin(self)
			local owner = proxy_get_owner(self)
			return translate(getmetatable(proxy_get_target(self)).__index[translate(k, owner, origin)], origin, owner)
		end,
		__newindex = function(self, k, v)
			error "tried to set a field on a protected object"
		end,
		__add = proxy_mt.__add,
		__sub = proxy_mt.__sub,
		__mul = proxy_mt.__mul,
		__div = proxy_mt.__div,
		__mod = proxy_mt.__mod,
		__pow = proxy_mt.__pow,
		__unm = proxy_mt.__unm,
		__call = proxy_mt.__call,
		__pairs = proxy_mt.__pairs,
		__len = proxy_mt.__len,
		__tostring = proxy_mt.__tostring,
		__eq = proxy_mt.__eq,
		__uncall = proxy_mt.__uncall,
	}
	local proxy_opaque_mt = { -- metatable for proxies that block access from external modules
		__metatable = "proxy",
		__index = function(self, k)
			error "tried to get a field on a protected object"
		end,
		__newindex = function(self, k, v)
			error "tried to set a field on a protected object"
		end,
		-- __call = function(self, ...) error "tried to call a protected object" end,
		-- __tostring = proxy_mt.__tostring,
		__eq = proxy_mt.__eq
		-- __uncall = proxy_mt.__uncall,
	}

	local sandboxed_pairs
	local sandboxed_next
	if use_weaktables then
		sandboxed_next = next
	else
		function sandboxed_next(tab, k)
			local newk, v = next(tab, k)
			if rawequal(newk, proxy_key) then
				return next(tab, newk)
			else
				return newk, v
			end
		end
	end

	do
		local _, tab, _ = pairs(setmetatable({ custom = false },
			{ __pairs = function(_) return next, { custom = true }, nil end }))
		if tab.custom then
			if not use_weaktables then
				-- local function __pairs(self)
				--   return sandboxed_next, self, nil
				-- end
				-- proxy_mt.__pairs = __pairs
				-- proxy_private_mt.__pairs = __pairs
				-- proxy_opaque_mt.__pairs = __pairs
			end
			sandboxed_pairs = pairs
		else
			if use_weaktables then
				sandboxed_pairs = pairs
			else
				function sandboxed_pairs(obj)
					return sandboxed_next, obj, nil
				end
			end
		end
	end

	if use_weaktables then
		function proxy_get(object, module_src, module_dst) -- proxy an object from the source module to the dest, reusing a proxy if possible
			if module_dst.proxy_of[object] then return module_dst.proxy_of[object] end
			local proxy
			local ot = type(object)
			if ot == "function" then
				proxy = function(...) return translate_args(module_src, module_dst,
						object(translate_args(module_dst, module_src, ...)))
					end
			else
				local omt = getmetatable(object)
				local mt = proxy_mt
				if omt then
					if omt.__proxy_private == true then
						mt = proxy_private_mt
					end
					if omt.__proxy_opaque == true then
						mt = proxy_opaque_mt
					end
				end
				proxy = setmetatable({}, mt)
			end
			proxy_origins[proxy] = module_src
			proxy_owners[proxy] = module_dst
			proxy_refs[proxy] = object
			module_dst.proxy_of[object] = proxy
			return proxy
		end
	else
		function proxy_get(object, module_src, module_dst) -- proxy an object from the source module to the dest, but without weak tables it can't reuse proxies
			local omt = getmetatable(object)
			local mt = proxy_mt
			if omt then
				if omt.__proxy_private == true then
					mt = proxy_private_mt
				end
				if omt.__proxy_opaque == true then
					mt = proxy_opaque_mt
				end
			end
			local proxy = setmetatable({ [proxy_key] = { origin = module_src, owner = module_dst, target = object } }, mt)
			return proxy
		end
	end

	local function pcall_handler(ok, err, ...) -- Automatically propagate kill codes through error handling to prevent using any protected mode to avoid a kill signal.
		if not ok and rawequal(err, error_kill_flag) then
			error(err)
		end
		return ok, err, ...
	end

	local sandboxed_pcall = function(func, ...)
		return pcall_handler(pcall(func, ...))
	end

	local sandboxed_xpcall = function(func, msgh, ...)
		local function wrapped_handler(err)
			if rawequal(err, error_kill_flag) then
				return err
			else
				return msgh(err)
			end
		end
		return pcall_handler(xpcall(func, wrapped_handler, ...))
	end

	local function env_create(module)
		local env
		env = {
			assert = assert,
			-- collectgarbage is forbidden to prevent messing with memory tracking
			-- dofile is forbidden because there is no default filesystem access
			error = error, -- this will probably need to be sandboxed in the future to allow errors with nonstring values
			-- _G added later
			getmetatable = sandboxed_getmetatable,
			ipairs = ipairs,
			load = function(ld, source, mode, subenv)
				if not source then source = "=(load)" end
				if not subenv then subenv = env end
				mode = "t"
				return load(ld, source, mode, subenv)
			end,
			-- loadfile is forbidden because there is no default filesystem access
			next = sandboxed_next,
			pairs = sandboxed_pairs,
			pcall = sandboxed_pcall,
			print = print,
			rawequal = rawequal,
			rawget = rawget,
			rawlen = rawlen,
			rawset = rawset,
			select = select,
			setmetatable = setmetatable,
			tonumber = tonumber,
			tostring = tostring,
			type = type,
			_VERSION = _VERSION,
			xpcall = sandboxed_xpcall,
			io = { write = io.write, flush = io.flush },

			coroutine = {
				create = coroutine.create,
				-- polyglot, so include this even if it's on the wrong version since it will just be empty
				---@diagnostic disable-next-line:deprecated
				isyieldable = coroutine.isyieldable,
				resume = function(co, ...)
					return translate_args(root_module, module, coroutine.resume(co, translate_args(module, root_module, ...)))
				end,
				running = coroutine.running,
				status = coroutine.status,
				wrap = function(f)
					local function wrapped_f(...)
						return translate_args(module, root_module, f(translate_args(root_module, module, ...)))
					end
					local co = coroutine.create(wrapped_f)
					return function(...)
						return translate_args(root_module, module,
							coroutine.resume(translate_args(module, root_module, ...) --[[@as thread]]))
					end
				end,
				yield = function(...)
					return translate_args(root_module, module, coroutine.yield(translate_args(module, root_module, ...)))
				end,
			},

			-- require must be set up to permit loading whitelisted packages from source and retrieving injected dependencies on other modules
			-- package configuration table is forbidden because it of filesystem and mutable global state

			string = cloneTab(string), -- string library is safe
			---@diagnostic disable-next-line:undefined-global
			utf8 = cloneTab(utf8),   -- if utf8 lib is available, it is fine
			table = cloneTab(table),
			bit = cloneTab(bit),
			math = cloneTab(math),
			-- io is forbidden
			-- os is forbidden
			-- debug is forbidden pending a sandboxed version
			lpeg = cloneTab(lpeg), -- lpeg can't accept proxies of functions so it has to be inside the security boundary
		}
		env._G = env
		return env
	end

	---Create a new module from specified source
	---@param code string
	---@param source string
	---@param ... unknown
	---@return function|nil module the loaded module
	---@return nil|string err the resulting error
	local function module_create(code, source, injected_deps)
		local module = { proxy_of = setmetatable({}, module_proxy_of_mt) }
		local env = env_create(module)
		for k, v in pairs(injected_deps) do env[k] = v end
		local fn, err = env.load(code, source or '=(module_create)')
		return translate_args(module, root_module, fn, err)
	end

	return module_create
end
