local shared = require 'feather.shared'
local backend = require 'feather.backend'
local message = require 'feather.message'

local C = terralib.includecstring [[
#include <stdio.h>
]]

local M = {}

local va_start = terralib.intrinsic("llvm.va_start", {&int8} -> {})
local va_end = terralib.intrinsic("llvm.va_end", {&int8} -> {})
terra M.stdout_log(root : &opaque, level : L.Level, f : shared.conststring, ...) : {}
  if level.val >= 0 then
    C.printf(L.Levels[level.val])
  end
  var vl : C.va_list
  va_start([&int8](&vl))
  C.vprintf(f, vl)
  va_end([&int8](&vl))
  C.printf("\n");
end

function M.simple_application(executable_name, application, ui, flags)
  if not flags then flags = {} end
  if type(flags) ~= "table" then
    error "expected flags to be a table containing a sequence of llvm command line flags"
  end
  if type(executable_name) ~= "string" then
    error "executable name must be a string"
  end
  local terra run_app(dllpath: rawstring, argc: int, argv: &rawstring)
    var u: ui
    var b: &backend.Backend
    var a: application

    b = backend.Backend.new(&u, [message.Behavior](ui.behavior), M.stdout_log, dllpath, nil)._0
    if b == nil then
      return 1
    end
    a:init(argc, argv)
    u:init(&a, [ui.query_store]{}, b)

    u:enter()
    while b:ProcessMessages() ~= 0 do
      a:update()
      u:update()
    end
    u:exit()
    a:exit()

    u:destruct()
    return 0
  end

  local backend_path = os.getenv "FEATHER_BACKEND"
  
  local terra main(argc: int, argv: &rawstring)
    run_app(backend_path, argc, argv)
  end

  local clangargs = { }

  for i,v in ipairs(arg) do
    if v == "-debug" or v == "-g" then
      executable_name = executable_name .. "_d"
      table.insert(clangargs, "-g")
      if jit.os == "Windows" then
        table.insert(clangargs, "/DEBUG")
      end
    end
  end

  if jit.os == "Windows" then
    executable_name = "bin-"..jit.arch.."/".. executable_name .. ".exe"
  end

  if jit.os == "Linux" then
    table.insert(clangargs, "-ldl")
  end

  table.insert(clangargs, "-lm")

  for _, v in ipairs(flags) do
    table.insert(clangargs, flags)
  end

  if terralib.saveobj(executable_name, "executable", {main = main}, clangargs) ~= nil then
    return -1
  end
end


return M
