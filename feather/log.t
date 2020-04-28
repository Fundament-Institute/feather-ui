local Enumset = require 'feather.enumset'

L = {}

L.Level = Enumset{
  "NONE", -1, -- Always shows up in the log but doesn't have a prefix.
  "FATAL", 0, 
  "ERROR", 1,
  "WARNING", 2,
  "NOTICE", 3,
  "DEBUG", 4,
}

return L