// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "bss_util.h"

#include "feathercpp.h"
#include "bss-util/XML.h"
#include "bss-util/bss_util.h"
#include <sstream>
#include <fstream>
#include <string>

using namespace bss;

XMLFile::XMLFile(const XMLFile& copy) : XMLNode(copy) {}
XMLFile::XMLFile(XMLFile&& mov) : XMLNode(std::move(mov)) {}
XMLFile::XMLFile(const char* source) { _name = "xml"; Read(source); }
XMLFile::XMLFile(std::istream& stream) { _name = "xml"; Read(stream); }
XMLFile::XMLFile() { _name = "xml"; }

void XMLFile::Read(const char* source)
{
  if(source) 
  { 
    std::istringstream ss(source); 
    Read(ss); 
  }
}

void XMLFile::Read(std::istream& stream)
{
  Str buf; 
  _initialParse(stream, buf); 
  _parseInner(stream, buf);
}

void XMLFile::_initialParse(std::istream& stream, Str& buf)
{
  if(_match(stream, buf, "<?*?>"))
    _parseAttribute(buf);
}

void XMLFile::Write(const char* file, bool pretty) const
{
  std::ofstream fs(file, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
  Write(fs, pretty);
  fs.close();
}
void XMLFile::Write(std::ostream& stream, bool pretty) const
{
  if(_attributes.Length() > 0)
  {
    stream << "<?";
    _writeAttribute(stream);
    stream << "?>";
  }
  for(size_t i = 0; i < _nodes.Length(); ++i)
    _nodes[i]->_write(stream, pretty, 0);
}

XMLNode::XMLNode(const XMLNode& copy) : _nodes(copy._nodes), _nodehash(copy._nodehash), _attributes(copy._attributes), _attrhash(copy._attrhash),
  _value(copy._value), _name(copy._name)
{
  next = 0;
  prev = 0;
}
XMLNode::XMLNode(XMLNode&& mov) : _nodes(std::move(mov._nodes)), _nodehash(std::move(mov._nodehash)), _attributes(std::move(mov._attributes)),
  _attrhash(std::move(mov._attrhash)), _value(std::move(mov._value)), _name(std::move(mov._name))
{
  next = mov.next;
  prev = mov.prev;
  mov.next = 0;
  mov.prev = 0;
}

XMLNode::XMLNode(const char* parse) { if(parse) { Str buf; std::istringstream ss(parse); _parse(ss, buf); } next = 0; prev = 0; }
XMLNode::XMLNode(std::istream& stream) { Str buf; _parse(stream, buf); next = 0; prev = 0; }

XMLNode& XMLNode::operator=(const XMLNode& copy)
{
  _attributes = copy._attributes;
  _attrhash = copy._attrhash;
  _value = copy._value;
  _name = copy._name;
  for(size_t i = 0; i < copy._nodes.Length(); ++i)
    AddNode(*copy._nodes[i].get());
  return *this;
}
XMLNode& XMLNode::operator=(XMLNode&& mov)
{
  _nodes = std::move(mov._nodes);
  _nodehash = std::move(mov._nodehash);
  _attributes = std::move(mov._attributes);
  _attrhash = std::move(mov._attrhash);
  _value = std::move(mov._value);
  _name = std::move(mov._name);
  return *this;
}

XMLNode* XMLNode::AddNode(const XMLNode& node) { return _addNode(std::unique_ptr<XMLNode>(new XMLNode(node))); }
XMLNode* XMLNode::AddNode(const char* name) { std::unique_ptr<XMLNode> node(new XMLNode()); node->_name = name; return _addNode(std::move(node)); }
XMLValue* XMLNode::AddAttribute(const XMLValue& value) { return _addAttribute(XMLValue(value)); }
XMLValue* XMLNode::AddAttribute(const char* name) { XMLValue v; v.Name = name; return _addAttribute(std::move(v)); }
bool XMLNode::RemoveNode(size_t index)
{
  if(index >= _nodes.Length()) 
    return false;
  khiter_t iter = _nodehash.Iterator(_nodes[index]->_name);
  if(_nodes[index]->next)
  {
    for(size_t i = 0; i < _nodes.Length(); ++i)
    {
      if(_nodes[i].get() == _nodes[index]->next)
      {
        _nodes[i]->prev = 0;
        if(_nodehash.ExistsIter(iter))
          _nodehash.SetValue(iter, i);
      }
    }
  }
  else if(_nodehash.ExistsIter(iter))
    _nodehash.RemoveIter(iter);

  _nodes.Remove(index);
  return true;
}
bool XMLNode::RemoveNode(const char* name) { return RemoveNode(_nodehash[name]); }
bool XMLNode::RemoveAttribute(size_t index)
{
  if(index >= _nodes.Length()) 
    return false;
  _attrhash.Remove(_nodes[index]->_name);
  _nodes.Remove(index);
  return true;
}
bool XMLNode::RemoveAttribute(const char* name) { return RemoveAttribute(_attrhash[name]); }
void XMLNode::SetValue(double value) { _value.Float = value; _value.Integer = (int64_t)value; _value.String = std::to_string(value); }
void XMLNode::SetValue(int64_t value) { _value.Integer = value; _value.Float = (double)value; _value.String = std::to_string(value); }
void XMLNode::SetValue(const char* value) { _value.String = value; _evalValue(_value); }
void XMLNode::Clear()
{
  _nodes.Clear();
  _nodehash.Clear();
  _attributes.Clear();
  _attrhash.Clear();
}

XMLNode* XMLNode::_addNode(std::unique_ptr<XMLNode> && n)
{
  khiter_t iter = _nodehash.Iterator(n->_name);

  if(!_nodehash.ExistsIter(iter))
    _nodehash.Insert(n->_name, _nodes.Length());
  else
  {
    XMLNode* target = _nodes[_nodehash.GetValue(iter)].get();
    LLInsert<XMLNode>(n.get(), target);
    _nodehash.SetValue(iter, _nodes.Length());
  }

  _nodes.Add(std::move(n));
  return _nodes.Back().get();
}
XMLValue* XMLNode::_addAttribute(XMLValue && v)
{
  khiter_t iter = _attrhash.Iterator(v.Name);

  if(!_attrhash.ExistsIter(iter))
  {
    _attributes.Add(std::move(v));
    _attrhash.Insert(_attributes.Back().Name, _attributes.Length() - 1);
    return &_attributes.Back();
  }

  size_t i = _attrhash.GetValue(iter);
  _attributes[i] = std::move(v);
  return &_attributes[i];
}

// Instead of including an entire regex library, we just do a dead simple pattern match of the form "<characters>*<characters>" where * is put into out.
bool XMLNode::_match(std::istream& stream, Str& out, const char* pattern, bool reset)
{
  while(stream.peek() != -1 && stream && isspace(stream.peek())) //eat whitespace
    stream.get(); 
  std::streamoff start = stream.tellg(); // get start of sequence
  int i;

  for(i = 0; stream.peek() != -1 && stream && (pattern[i] != 0) && (pattern[i] != '*'); ++i)
  {
    if(stream.get() != pattern[i])
    {
      stream.seekg(start);
      return false;
    }
  }

  if(!pattern[i]) 
  { 
    if(reset) 
      stream.seekg(start); 
    return true; 
  }

  if(stream.peek() == -1 || !stream) 
  {
    stream.seekg(start);
    return false; 
  }

  int end = ++i; // increment by 1 to skip past * marker
  std::streamoff mid = stream.tellg();
  out.clear();

  while(stream.peek() != -1 && stream && pattern[end] != 0)
  {
    end = (stream.peek() == pattern[end]) ? (end + 1) : i;
    out += stream.get();
  }

  out.resize(i - end + out.length()); // trim off ending
  if(!pattern[end])  // If we got through the entire pattern, the match was a success (even if the file ended).
  { 
    if(reset) 
      stream.seekg(mid);
    return true; 
  }

  stream.seekg(start); // otherwise we failed so reset everything
  return false;
}

bool XMLNode::_parse(std::istream& stream, Str& buf)
{
  while(_match(stream, buf, "<!--*-->"));
  if(!_match(stream, buf, "<*>")) return false;
  if(buf.length() > 0 && buf.back() == '/') // this is a self-closing tag
  {
    buf.resize(buf.length() - 1); // chop off the /
    _parseAttribute(buf);
    return true;
  }
  _parseAttribute(buf);
  _parseInner(stream, buf);
  if(stream.peek() == -1 || !stream) return true;
  if(buf.Trim() != _name) // If the end tag is not a legal end tag, someone didn't close something properly, so we need to backtrack
    stream.seekg(-2 - buf.length(), std::ios_base::cur);
  return true;
}

void XMLNode::_parseInner(std::istream& stream, Str& buf)
{
  _value.String.clear();
  bool end = false;
  while(stream.peek() != -1 && stream && !end) // Run through the content, extracting and parsing all child nodes, until we hit an end tag
  {
    switch(stream.peek())
    {
    case '&':
      _parseEntity(stream, _value.String);
      continue;
    case '<':
      if(_match(stream, buf, "</*>")) { end = true; continue; }
      if(_match(stream, buf, "<!--*-->")) continue;
      std::unique_ptr<XMLNode> node(new XMLNode());
      if(node->_parse(stream, buf))
      {
        _addNode(std::move(node));
        continue; // only continue if we successfully parsed SOMETHING. 
      } // Otherwise, the < was malformed, so we just add it as a character.
    }
    _value.String += stream.get();
  }
  _evalValue(_value);
}

void XMLNode::_parseAttribute(Str& buf)
{
  buf = buf.Trim();
  size_t i;
  for(i = 0; i < buf.length() && !isspace(buf[i]); ++i);

  _name.assign(buf, 0, i);
  while(i < buf.length() && isspace(buf[i]))
    ++i;

  if(i >= buf.length()) 
    return;
  const char* c = ((const char*)buf) + i;

  std::istringstream ss(c);
  while(ss.peek() != -1 && ss)
  {
    XMLValue v;
    if(_match(ss, v.Name, "*="))
    {
      v.Name = v.Name.Trim();
      if(!_match(ss, v.String, "\"*\"")) // If we can't get a valid string just eat everything until the next space.
      {
        while(ss.peek() != -1 && ss && !isspace(ss.peek())) 
          v.String += ss.get();
      }

      _evalValue(v);
    }
    else
    { // If we can't find an equals sign, we'll just have to eat the whole thing.
      while(ss.peek() != -1 && ss && !isspace(ss.peek()))
        v.Name += ss.get();
    }

    _addAttribute(std::move(v));
  }
}

void XMLNode::_parseEntity(std::istream& stream, Str& target)
{
  Str buf;
  char* c;
  if(_match(stream, buf, "&lt;")) target += '<';
  else if(_match(stream, buf, "&gt;")) target += '>';
  else if(_match(stream, buf, "&amp;")) target += '&';
  else if(_match(stream, buf, "&apos;")) target += '\'';
  else if(_match(stream, buf, "&quot;")) target += '"';
  else if(_match(stream, buf, "&#x*;")) OutputUnicode(target, strtol(buf, &c, 16));
  else if(_match(stream, buf, "&#*;")) OutputUnicode(target, strtol(buf, &c, 10));
  else target += stream.get();
}

void XMLNode::_evalValue(XMLValue& val)
{
  char* c;
  val.Integer = 0;
  val.Float = 0.0;
  if(val.String.length() > 0)
  {
    if(val.String[0] == '-')
      val.Integer = strtoll(val.String, &c, 10);
    else if(val.String.length() > 1 && val.String[0] == '0' && val.String[1] == 'x')
      val.Integer = (int64_t)strtoull(val.String, &c, 16);
    else if(val.String.length() > 1 && val.String[0] == '0' && val.String[1] == 'b')
      val.Integer = (int64_t)strtoull(val.String, &c, 2);
    else
      val.Integer = (int64_t)strtoull(val.String, &c, 10);

    val.Float = atof(val.String);
  }
}

void XMLNode::_writeAttribute(std::ostream& stream) const
{
  stream << _name;

  for(size_t i = 0; i < _attributes.Length(); ++i)
  {
    stream << ' ' << _attributes[i].Name << "=\"";
    _writeString(stream, _attributes[i].String, true);
    stream.put('"');
  }
}

void XMLNode::_writeString(std::ostream& stream, const char* s, bool attribute)
{
  while(*s)
  {
    if(!attribute)
    {
      switch(*s)
      {
      case '<': stream << "&lt;"; break;
      case '>': stream << "&gt;"; break;
      case '&': stream << "&amp;"; break;
      case '\'': stream << "&apos;"; break;
      case '"': stream << "&quot;"; break;
      default: stream.put(*s); break;
      }
    }
    else if(*s == '"')
      stream << "&quot;";
    else
      stream.put(*s);

    ++s;
  }
}

void XMLNode::_write(std::ostream& stream, bool pretty, int depth) const
{
  if(pretty)
  {
    stream << std::endl;
    for(int i = 0; i < depth; ++i) stream.put('\t');
  }

  stream.put('<');
  _writeAttribute(stream);
  bool content = _value.String.Trim().length() > 0;

  if(!content && !_nodes.Length()) // If there is no content and no nodes, assume this is a self-closing tag.
  {
    stream << "/>";
    return;
  }

  stream.put('>');
  if(content) _writeString(stream, _value.String, false);

  for(size_t i = 0; i < _nodes.Length(); ++i)
    _nodes[i]->_write(stream, pretty, depth + 1);

  if(_nodes.Length() > 0 && pretty)
  {
    stream << std::endl;
    for(int i = 0; i < depth; ++i) stream.put('\t');
  }

  stream << "</" << _name << ">";
}