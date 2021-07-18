


local M = {}

local predefines = {}
local specialops = {}

local dimnames = {
  x = 0,
  y = 1,
  z = 2,
  w = 3,
  r = 0,
  g = 1,
  b = 2,
  a = 3
}

local constructvec
local genvectype

local function make_vecswizzle(dimension, type, fields)
  local components = {}
  for i = 1, #fields do
    local dim = fields:sub(i, i)
    components[i] = dimnames[dim]
    if not components[i] or components[i] >= dimension then
      error("unknown component "..dim.." in vector "..type.." "..dimension)
    end
  end
  local terra vecswizzle(vec: genvectype(dimension, type))
    return [constructvec(#components, type)](
      escape
        for i, component in ipairs(components) do
          emit `vec.v[component]
        end
      end
    )
  end
  vecswizzle.name = "vec_swizzle_"..tostring(type).."_"..dimension.."_"..fields
  specialops[vecswizzle] = function(emit)
    return function() emit"(" end, function() emit(")."..fields) end
  end
  return vecswizzle
end
make_vecswizzle = terralib.memoize(make_vecswizzle)


local function vec_entrymissing(entryname, obj)
  local vec_t = obj:gettype()
  return `[make_vecswizzle(vec_t.N, vec_t.T, entryname)](obj)
end
vec_entrymissing = macro(vec_entrymissing)

local ops = { "sub","add","mul","div" }
local function genops(s, N, type)
  for _, op in ipairs(ops) do
    local i = symbol(int,"i")
    local function generator(ae,be)
        return quote
            var c : s
            for [i] = 0,N do
                c.v[i] = operator(["__"..op],ae,be)
            end
            return c
        end
    end
    
    local terra vecvec(a : s, b : s) [generator(`a.v[i],`b.v[i])]  end
    local terra scalarvec(a : type, b : s) [generator(`a,`b.v[i])]  end
    local terra vecscalar(a : s, b : type) [generator(`a.v[i],`b)]  end
    vecvec.name = op.."_vecvec"
    scalarvec.name = op.."_scalarvec"
    vecscalar.name = op.."_vecscalar"
    s.metamethods["__"..op] = terralib.overloadedfunction(op,{vecvec,scalarvec,vecscalar})
  end
end

function genvectype(dimension, type)
  local vec = terralib.types.newstruct("Vec"..dimension)
  vec.entries:insert {field = "v", type = type[dimension]}
  vec.T = type
  vec.N = dimension
  vec.metamethods.__entrymissing = vec_entrymissing
  genops(vec, dimension, type)
  return vec
end
genvectype = terralib.memoize(genvectype)

function genvecfunc(params, dimension, element)
  return terra([params]) : genvectype(dimension, element)
    var x : genvectype(dimension, element)
    escape
      print("ARGS:"..#params)
      if #params == 1 and not params[1].type:isstruct() then
        local v = params[1]
        if v.type ~= element then
          error("Can't build Vec"..dimension.." from a different type: "..tostring(element))
        end
        
        for i=1,dimension do
          emit(quote x.v[ [i] ] = [v] end)
        end
      else
        local totaldim = 0
        for i,v in ipairs(params) do
          if v.type == element then
            totaldim = totaldim + 1
            if totaldim > dimension then
              error("Tried to put "..totaldim.." elements into vec"..dimension)
            end
            emit(quote x.v[ [totaldim] ] = v end)
          elseif v.type:isstruct() and v.type.N ~= nil and v.type.T ~= nil then
            local t = v.type
            if t.T ~= element then
              error("Invalid parameter type in vec constructor, expected "..tostring(element).." but got "..tostring(t.T))
            end
            if totaldim + t.N > dimension then
              error("Tried to put "..(totaldim + t.N).." elements into vec"..dimension.." while appending a vec"..t.N)
            end
            for i=1,t.N do
              totaldim = totaldim + 1
              emit(quote x.v[ [totaldim] ] = v.v[ [i] ] end)
            end
          else
            error("Invalid type passed to vec constructor: "..tostring(v.type))
          end
        end
        if totaldim ~= dimension then
          error("Can't build Vec"..dimension.." from "..totaldim.." components")
        end 
      end
    end

    return x
  end
end
genvecfunc = terralib.memoize(genvecfunc)

-- handle all the different ways to construct a given vector
function constructvec(dimension, element)
  return macro(function(...)
    local args = terralib.newlist({...})
    local params = args:map(function(v) return symbol(v:gettype()) end)
    local func = genvecfunc(params, dimension, element)
    return `[func]([args])
  end)
end
constructvec = terralib.memoize(constructvec)

predefines.Vec2 = genvectype(2, float)
predefines.vec2 = constructvec(2, float)
predefines.Vec3 = genvectype(3, float)
predefines.vec3 = constructvec(3, float)
predefines.Vec4 = genvectype(4, float)
predefines.vec4 = constructvec(4, float)

struct predefines.Sampler2D {}
terra predefines.texture2D(texture: predefines.Sampler2D, uv: predefines.Vec2): predefines.Vec4
  return predefines.vec4(0.0f)
end


local function glslstring(toptree,options)
  options = options or {}
  local breaklines = options.breaklines == nil or options.breaklines
  local showsource = options.showsource == true
  local permissive_output = options.permissive == true
  local namer = options.namer or function(obj) return obj.name or "<anon>" end
  local buffer = terralib.newlist() -- list of strings that concat together into the pretty output
  local env = terralib.newenvironment({})
  local indentstack = terralib.newlist{ 0 } -- the depth of each indent level

  local currentlinelength = 0
  local function enterblock()
    indentstack:insert(indentstack[#indentstack] + 4)
  end
  local function enterindenttocurrentline()
    indentstack:insert(currentlinelength)
  end
  local function leaveblock()
    indentstack:remove()
  end
  local function emit(fmt,...)
    local function toformat(x)
      if type(x) ~= "number" and type(x) ~= "string" then
        return tostring(x)
      else
        return x
      end
    end
    local strs = terralib.newlist({...}):map(toformat)
    local r = fmt:format(unpack(strs))
    currentlinelength = currentlinelength + #r
    buffer:insert(r)
  end
  local function pad(str,len)
    if #str > len then return str:sub(-len)
    else return str..(" "):rep(len - #str) end
  end
  local function differentlocation(a,b)
    return (a.linenumber ~= b.linenumber or a.filename ~= b.filename)
  end
  local lastanchor = { linenumber = "", filename = "" }
  local function begin(anchor,...)
    if showsource then
      local fname = differentlocation(lastanchor,anchor) and (anchor.filename..":"..anchor.linenumber..": ")
        or ""
      emit("%s",pad(fname,24))
    end
    currentlinelength = 0
    emit((" "):rep(indentstack[#indentstack]))
    emit(...)
    lastanchor = anchor
  end

  local function invalidContent(message)
    if permissive_output then
      print("WARNING: "..message)
    else
      error(message)
    end
  end

  local function emitList(lst,begin,sep,finish,fn)
    emit(begin)
    for i,k in ipairs(lst) do
      fn(k,i)
      if i ~= #lst then
        emit(sep)
      end
    end
    emit(finish)
  end

  local function emitType(t)
    --TODO: modify type names to fit C source output
    emit("%s",t)
  end

  local function UniqueName(name,key)
    assert(name) assert(key)
    local lenv = env:localenv()
    local assignedname = lenv[key]
    --if we haven't seen this key in this scope yet, assign a name for this key, favoring the non-mangled name
    if not assignedname then
      local basename,i = name,1
      while lenv[name] do
        --TODO: replace name mangling with C source compatible name mangling
        name,i = basename.."$"..tostring(i),i+1
      end
      lenv[name],lenv[key],assignedname = true,name,name
    end
    return assignedname
  end
  local function emitIdent(name,sym)
    assert(name) assert(terralib.issymbol(sym))
    emit("%s",UniqueName(name,sym))
  end
  local luaexpression = "[ <lua exp> ]"
  local function IdentToString(ident)
    if ident.kind == "luaexpression" then
      invalidContent"luaexpression persisted in tree being compiled to GLSL"
      return luaexpression
    else return tostring(ident.value) end
  end
  local function emitParam(p)
    assert(terralib.irtypes.allocvar:isclassof(p) or terralib.irtypes.param:isclassof(p))
    if terralib.irtypes.unevaluatedparam:isclassof(p) then
      invalidContent "luaexpression in param type persisted in tree being compiled to GLSL"
      emit("%s%s",IdentToString(p.name),p.type and " : "..luaexpression or "")
    else
      if p.type then emit("%s ", namer(p.type))
      else invalidContent "untyped Param persisted in tree being compiled to GLSL" end
      emitIdent(p.name,p.symbol)
    end
  end
  local implicitblock = { repeatstat = true, fornum = true, fornumu = true}
  local emitStmt, emitExp,emitParamList,emitLetIn
  local function emitStmtList(lst) --nested Blocks (e.g. from quotes need "do" appended)
    for i,ss in ipairs(lst) do
      if ss:is "block" and not (#ss.statements == 1 and implicitblock[ss.statements[1].kind]) then
        begin(ss,"{\n")
        emitStmt(ss)
        begin(ss,"}\n")
      else
        emitStmt(ss)
      end
    end
  end
  local function emitAttr(a)
    invalidContent"memory access attributes in tree being compiled to GLSL, unhandled case"
    emit("{ nontemporal = %s, align = %s, isvolatile = %s }",a.nontemporal,a.alignment or "native",a.isvolatile)
  end
  function emitStmt(s)
    if s:is "block" then
      enterblock()
      env:enterblock()
      emitStmtList(s.statements)
      env:leaveblock()
      leaveblock()
    elseif s:is "returnstat" then
      begin(s,"return ")
      emitExp(s.expression)
      emit("\n")
    elseif s:is "label" then
      begin(s,"::%s::\n",IdentToString(s.label))
    elseif s:is "gotostat" then
      begin(s,"goto %s\n",IdentToString(s.label))
    elseif s:is "breakstat" then
      begin(s,"break\n")
    elseif s:is "whilestat" then
      begin(s,"while ()")
      emitExp(s.condition)
      emit(" ) {\n")
      emitStmt(s.body)
      begin(s,"}\n")
    elseif s:is "repeatstat" then
      begin(s,"do {\n")
      enterblock()
      emitStmtList(s.statements)
      leaveblock()
      begin(s.condition,"while (!(")
      emitExp(s.condition)
      emit("))\n")
    elseif s:is "fornum"or s:is "fornumu" then
      begin(s,"for ")
      emitParam(s.variable)
      emit(" = ")
      emitExp(s.initial) emit(";") emitIdent(s.variable.name, s.variable.symbol) emit("<") emitExp(s.limit) emit(";")
      if s.step then
        emitIdent(s.variable.name, s.variable.symbol) emit("+=") emitExp(s.step)
      else
        emit("++") emitParam(s.variable)
      end
      emit(") {\n")
      emitStmt(s.body)
      begin(s,"}\n")
    elseif s:is "forlist" then
      invalidContent "range-based for loop not resolved in tree being compiled to GLSL"
      begin(s,"for ")
      emitList(s.variables,"",", ","",emitParam)
      emit(" in ")
      emitExp(s.iterator)
      emit(" do\n")
      emitStmt(s.body)
      begin(s,"end\n")
    elseif s:is "ifstat" then
      for i,b in ipairs(s.branches) do
        if i == 1 then
          begin(b,"if (")
        else
          begin(b,"else if (")
        end
        emitExp(b.condition)
        emit(") {\n")
        emitStmt(b.body)
      end
      if s.orelse then
        begin(s.orelse,"} else {\n")
        emitStmt(s.orelse)
      end
      begin(s,"}\n")
    elseif s:is "defvar" then
      invalidContent "untyped defvar in tree being compiled to GLSL"
      begin(s,"var ")
      emitList(s.variables,"",", ","",emitParam)
      if s.hasinit then
        emit(" = ")
        emitParamList(s.initializers)
      end
      emit("\n")
    elseif s:is "assignment" then
      if #s.lhs > 1 then
        invalidContent "parallel assignment is not yet handled in GLSL"
      end
      begin(s,"")
      emitParamList(s.lhs)
      emit(" = ")
      emitParamList(s.rhs)
      emit("\n")
    elseif s:is "defer" then
      invalidContent "defer is not yet handled in GLSL"
      begin(s,"defer ")
      emitExp(s.expression)
      emit("\n")
    elseif s:is "statlist" then
      emitStmtList(s.statements)
    else
      begin(s,"")
      emitExp(s)
      emit("\n")
    end
  end

  local function makeprectable(...)
    local lst = {...}
    local sz = #lst
    local tbl = {}
    for i = 1,#lst,2 do
      tbl[lst[i]] = lst[i+1]
    end
    return tbl
  end

  local prectable = makeprectable(
    "+",7,"-",7,"*",8,"/",8,"%",8,
    "^",11,"..",6,"<<",4,">>",4,
    "==",3,"<",3,"<=",3,
    "~=",3,">",3,">=",3,
    "and",2,"or",1,
    "@",9,"&",9,"not",9,"select",12)

  local function getprec(e)
    if e:is "operator" then
      if "-" == e.operator and #e.operands == 1 then return 9 --unary minus case
      else return prectable[e.operator] end
    else
      return 12
    end
  end
  local function doparens(ref,e,isrhs)
    local pr, pe = getprec(ref), getprec(e)
    if pr > pe or (isrhs and pr == pe) then
      emit("(")
      emitExp(e)
      emit(")")
    else
      emitExp(e)
    end
  end

  function emitExp(e,maybeastatement)
    if breaklines and differentlocation(lastanchor,e)then
      local ll = currentlinelength
      emit("\n")
      begin(e,"")
      emit((" "):rep(ll - currentlinelength))
      lastanchor = e
    end
    if e:is "var" then
      if e.symbol then emitIdent(e.name,e.symbol)
      else emit("%s",e.name) end
    elseif e:is "globalvalueref" and e.value.kind == "globalvariable" then
      emitIdent(e.name,e.value.symbol)
    elseif e:is "globalvalueref" and e.value.kind == "terrafunction" then
      emit(e.value.name)
    elseif e:is "allocvar" then
      --emit("var ")
      emitParam(e)
    elseif e:is "setter" then
      invalidContent "custom setter behavior is not yet handled in GLSL compiler"
      emit("<setter:") emitExp(e.setter) emit(">")
    elseif e:is "setteru" then emit("<setteru>") invalidContent "untyped custom setter persisted in tree being compiled to GLSL"
    elseif e:is "operator" then
      local op = e.operator
      local function emitOperand(o,isrhs)
        doparens(e,o,isrhs)
      end
      if #e.operands == 1 then
        if op == "@" then
          emit "*"
        elseif op == "not" then
          emit "!"
        else
          emit(op)
        end
        emitOperand(e.operands[1])
      elseif #e.operands == 2 then
        emitOperand(e.operands[1])
        if op == "and" or op == "or" then
          if e.operands[1].type:islogical() then
            emit(" %s ", op == "and" and "&&" or "||")
          else
            emit(" %s ", op == "and" and "&" or "|")
          end
        else
          if op == ".." then
            invalidContent "string concatenation is not supported in GLSL"
          end
          
          emit(" %s ", op)
        end
        
        emitOperand(e.operands[2],true)
      elseif op == "select" then
        emitOperand(e.operands[1])
        emit("?")
        emitOperand(e.operands[2])
        emit(":")
        emitOperand(e.operands[3])
      else
        invalidContent "unknown operator in tree being compiled to GLSL"
        emit("<??operator:"..op.."??>")
      end
    elseif e:is "index" then
      doparens(e,e.value)
      emit("[")
      emitExp(e.index)
      emit("]")
    elseif e:is "literal" then
      if e.type:isintegral() then
        if not e.stringvalue then
          invalidContent "integer literal without string representation"
        end
        emit(e.stringvalue)
      elseif type(e.value) == "string" then
        emit("%s",("%q"):format(e.value):gsub("\\\n","\\n"))
      elseif e.type:isnumeric() then
        emit("%s", tostring(e.value)..(e.value%1 == 0 and ".0" or ""))
      else
        emit("%s",tostring(e.value))
      end
    elseif e:is "cast" then
      emit("(")
      emitType(e.to or e.type)
      emit(")")
      emit("(")
      emitExp(e.expression)
      emit(")")
    elseif e:is "structcast" then
      emitType(e.to or e.type)
      emit("(")
      emitExp(e.expression)
      emit(")")
    elseif e:is "sizeof" then
      emit("sizeof(%s)",e.oftype)
    elseif e:is "apply" then
      doparens(e,e.value)
      emit("(")
      emitParamList(e.arguments)
      emit(")")
    elseif e:is "selectu" or e:is "select" then
      doparens(e,e.value)
      emit(".")
      emit("%s",e.fieldname or IdentToString(e.field))
    elseif e:is "vectorconstructor" then
      invalidContent "vector constructors aren't known in glsl."
      emit("vector(")
      emitParamList(e.expressions)
      emit(")")
    elseif e:is "arrayconstructor" then
      emit("{")
      emitParamList(e.expressions)
      emit("}")
    elseif e:is "constructor" then
      emitList(e.type:getlayout().entries:map(function(e) return e.value end),"(",", ",")",emitExp)
    elseif e:is "constructoru" then
      invalidContent "untyped constructor in tree being compiled to GLSL"
      emit("{")
      local function emitField(r)
        if r.type == "recfield" then
          emit("%s = ",IdentToString(r.key))
        end
        emitExp(r.value)
      end
      emitList(e.records,"",", ","",emitField)
      emit("}")
    elseif e:is "constant" then
      if e.type:isprimitive() then
        local val = tonumber(e.value)
        emit("%s",tostring(val)..(val%1 == 0 and ".0" or ""))
      else
        invalidContent "Constant of non primitive type in tree being compiled to GLSL"
        emit("<constant:"..tostring(e.type)..">")
      end
    elseif e:is "letin" then
      emitLetIn(e)
    elseif e:is "attrload" then
      invalidContent "attributed load in tree being compiled to GLSL"
      emit("attrload(")
      emitExp(e.address)
      emit(", ")
      emitAttr(e.attrs)
      emit(")")
    elseif e:is "attrstore" then
      emit("attrstore(")
      emitExp(e.address)
      emit(", ")
      emitExp(e.value)
      emit(", ")
      emitAttr(e.attrs)
      emit(")")
    elseif e:is "luaobject" then
      invalidContent "unevaluated Luaobject in tree being compiled to GLSL"
      if terralib.types.istype(e.value) then
        emit("[%s]",e.value)
      elseif terralib.ismacro(e.value) then
        emit("<macro>")
      elseif terralib.isoverloadedfunction(e.value) then
        emit("%s",e.name)
      else
        emit("<lua value: %s>",tostring(e.value))
      end
    elseif e:is "method" then
      invalidContent "NYI: Method call in tree being compiled to GLSL"
      doparens(e,e.value)
      emit(":%s",IdentToString(e.name))
      emit("(")
      emitParamList(e.arguments)
      emit(")")
    elseif e:is "debuginfo" then
      emit("/*debuginfo(%q,%d)*/",e.customfilename,e.customlinenumber)
    elseif e:is "inlineasm" then
      invalidContent "inlineasm not allowed in a shader"
      emit("inlineasm(")
      emitType(e.type)
      emit(",%s,%s,%s,",e.asm,tostring(e.volatile),e.constraints)
      emitParamList(e.arguments)
      emit(")")
    elseif e:is "quote" then
      emitExp(e.tree)
    elseif e:is "luaexpression" then return luaexpression
    elseif maybeastatement then
      emitStmt(e)
    else
      invalidContent "unknown expression kind in tree being compiled to GLSL"
      emit("<??"..e.kind.."??>")
      error("??"..tostring(e.kind))
    end
  end
  function emitParamList(pl)
    emitList(pl,"",", ","",emitExp)
  end
  function emitLetIn(pl)
    if pl.hasstatements then
      invalidContent "NYI: let in tree with nested statements being compiled to GLSL"
      enterindenttocurrentline()
      emit("let\n")
      enterblock()
      emitStmtList(pl.statements)
      leaveblock()
      begin(pl,"in\n")
      enterblock()
      begin(pl,"")
    end
    emitList(pl.expressions,"",", ","",emitExp)
    if pl.hasstatements then
      leaveblock()
      emit("\n")
      begin(pl,"end")
      leaveblock()
    end
  end
  if terralib.irtypes.functiondef:isclassof(toptree) or terralib.irtypes.functiondefu:isclassof(toptree) then
    begin(toptree,"%s %s",namer(toptree.type.returntype), options.name)
    emitList(toptree.parameters,"(",",",") ",emitParam)
    -- if terralib.irtypes.functiondef:isclassof(toptree) then
    --   emit(": ") emitType(toptree.type.returntype)
    -- elseif toptree.returntype then
    --   emit(": ")
    --   if terralib.irtypes.Type:isclassof(toptree.returntype) then emitType(toptree.returntype)
    --   else emitExp(toptree.returntype) end
    -- end
    emit("\n")
    emitStmt(toptree.body)
    begin(toptree,"end\n")
  elseif terralib.irtypes.functionextern:isclassof(toptree) then
    begin(toptree,"terra %s :: %s = <extern>\n",toptree.name,toptree.type)
  else
    emitExp(toptree,true)
    emit("\n")
  end
  return buffer:concat()
end  
  

local defaultnames = {
  int = int32,
}

for k,v in pairs(predefines) do
  if terralib.types.istype(v) then
    defaultnames[k] = v
  end
end

function M.compile(namespace, options)
  local assigned_names = {}
  local used_names = {}
  local generated = {}
  for k, v in pairs(defaultnames) do
    assigned_names[v] = k
    used_names[k] = true
    generated[v] = true
  end
  for k, v in pairs(namespace) do
    assigned_names[v] = k
    used_names[k] = true
  end
  
  local circular_dep_check = {}
  local segments = terralib.newlist()
  local generated_names_index = 0
  
  local generate
  
  local function namer(obj)
    print("naming", obj)
    local name
    if assigned_names[obj] then
      name = assigned_names[obj]
      print("existing assigned name", name)
    else
      name = (obj.name or "anon") .. "$" .. generated_names_index
      generated_names_index = generated_names_index + 1
    end
    if not generated[obj] then
      generate(obj.definition, name)
    end
    return name
  end

  function generate(obj, name)
    if circular_dep_check[obj] then
      return -- Generate forward declaration
    end
    circular_dep_check[obj] = true
    segments:insert(glslstring(obj, {namer = namer, name = name, permissive = options.permissive}))
    circular_dep_check[obj] = false
    generated[obj] = true
  end

  for k, v in pairs(namespace) do
    -- terralib.printraw(v.definition)
    generate(v.definition, k)
  end
  
  return segments:concat("\n")
end

for k,v in pairs(predefines) do
  M[k] = v
end

return M
