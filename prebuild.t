local Backend = require 'feather.backend'
local Export = require 'feather.c-headers'
local Shared = require 'feather.shared'
local Log = require 'feather.log'
local Msg = require 'feather.message'
local Messages = require 'feather.messages'

local transfer = {"Vec","URect","Rect","Delegate","UVec","Color","Veci","Color16","Vec3","Err"}
for i,k in ipairs(transfer) do
  Backend[k] = Shared[k]
end
for k,v in pairs(Messages) do
  Backend[k] = Messages[k]
end
Backend.Level = Log.Level
Backend.Msg = Msg.Message
Backend.Result = Msg.Result
Backend.Kind = Msg.Kind
Backend.Behavior = Msg.Behavior
Backend.MsgReceiver = Msg.Receiver
Backend.Window = Msg.Window

local f = io.open("include/backend.h", "wb")
f:write(Export({short_name="FG", long_name="BACKEND"}, Backend))
f:close()

return 0