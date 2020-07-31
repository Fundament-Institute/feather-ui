local Backend = require 'feather.backend'
local Export = require 'feather.c-headers'
local Shared = require 'feather.shared'
local Log = require 'feather.log'
local Element = require 'feather.element'
local Message = require 'feather.message'
local Messages = require 'feather.messages'

local transfer = {"Vec","URect","Rect","Delegate","UVec","Color","Veci","Color16","Vec3D","Err"}
for i,k in ipairs(transfer) do
  Backend[k] = Shared[k]
end
for k,v in pairs(Messages) do
  Backend[k] = Messages[k]
end
Backend.Level = Log.Level
Backend.Msg = Message.Msg
Backend.Result = Message.Result
Backend.Kind = Message.Kind
Backend.Behavior = Message.Behavior
Backend.Element = Element

local f = io.open("include/backend.h", "wb")
f:write(Export({short_name="FG", long_name="BACKEND"}, Backend))

return 0