// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "bss_util.h"

#include "bss-util/cXML.h"
#include "bss-util/bss_util.h"
#include <sstream>
#include <fstream>
#include <string>

using namespace bss_util;

cXML::cXML(const cXML& copy) : cXMLNode(copy) { }
cXML::cXML(cXML&& mov) : cXMLNode(std::move(mov)) { }
cXML::cXML(const char* source) { if(source) { cStr buf; std::istringstream ss(source); _initialparse(ss, buf); _parseinner(ss, buf); } }
cXML::cXML(std::istream& stream) { cStr buf; _initialparse(stream, buf); _parseinner(stream, buf); }

void cXML::_initialparse(std::istream& stream, cStr& buf)
{
  if(_match(stream, buf, "<?*?>"))
    _parseattribute(buf);
}

void cXML::Write(const char* file, bool pretty) const
{
  std::ofstream fs(file, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
  Write(fs, pretty);
  fs.close();
}
void cXML::Write(std::ostream& stream, bool pretty) const
{
  if(_attributes.Length()>0)
  {
    stream << "<?";
    _writeattribute(stream);
    stream << "?>";
  }
  for(size_t i = 0; i < _nodes.Length(); ++i)
    _nodes[i]->_write(stream, pretty, 0);
}

cXMLNode::cXMLNode(const cXMLNode& copy) { next=0; prev=0; }
cXMLNode::cXMLNode(cXMLNode&& mov) { next=mov.next; prev=mov.prev; mov.next=0; mov.prev=0; }
cXMLNode::cXMLNode(const char* parse) { if(parse) { cStr buf; std::istringstream ss(parse); _parse(ss, buf); } next=0; prev=0; }
cXMLNode::cXMLNode(std::istream& stream) { cStr buf; _parse(stream, buf); next=0; prev=0; }

cXMLNode& cXMLNode::operator=(const cXMLNode& copy)
{
  _attributes = copy._attributes;
  _attrhash = copy._attrhash;
  _value = copy._value;
  _name = copy._name;
  for(size_t i = 0; i < copy._nodes.Length(); ++i)
    AddNode(*copy._nodes[i].get());
  return *this;
}
cXMLNode& cXMLNode::operator=(cXMLNode&& mov)
{
  _nodes = std::move(mov._nodes);
  _nodehash = std::move(mov._nodehash);
  _attributes = std::move(mov._attributes);
  _attrhash = std::move(mov._attrhash);
  _value = std::move(mov._value);
  _name = std::move(mov._name);
  return *this;
}

cXMLNode* cXMLNode::AddNode(const cXMLNode& node) { return _addnode(std::unique_ptr<cXMLNode>(new cXMLNode(node))); }
cXMLNode* cXMLNode::AddNode(const char* name) { std::unique_ptr<cXMLNode> node(new cXMLNode()); node->_name = name; return _addnode(std::move(node)); }
cXMLValue* cXMLNode::AddAttribute(const cXMLValue& value) { return _addattribute(std::move(cXMLValue(value))); }
cXMLValue* cXMLNode::AddAttribute(const char* name) { cXMLValue v; v.Name=name; return _addattribute(std::move(v)); }
bool cXMLNode::RemoveNode(size_t index)
{ 
  if(index>= _nodes.Length()) return false;
  khiter_t iter = _nodehash.Iterator(_nodes[index]->_name);
  if(_nodes[index]->next) {
    for(size_t i = 0; i < _nodes.Length(); ++i) {
      if(_nodes[i].get() == _nodes[index]->next)
      {
        _nodes[i]->prev = 0;
        if(_nodehash.ExistsIter(iter))
          _nodehash.SetValue(iter, i);
      }
    }
  } else if(_nodehash.ExistsIter(iter))
    _nodehash.RemoveIter(iter);
  
  _nodes.Remove(index);
  return true;
}
bool cXMLNode::RemoveNode(const char* name) { return RemoveNode(_nodehash[name]); }
bool cXMLNode::RemoveAttribute(size_t index)
{
  if(index>= _nodes.Length()) return false;
  _attrhash.Remove(_nodes[index]->_name);
  _nodes.Remove(index);
  return true;
}
bool cXMLNode::RemoveAttribute(const char* name) { return RemoveAttribute(_attrhash[name]); }
void cXMLNode::SetValue(double value) { _value.Float = value; _value.Integer = (uint64_t)value; _value.String = std::to_string(value); }
void cXMLNode::SetValue(uint64_t value) { _value.Integer = value; _value.Float = (double)value; _value.String = std::to_string(value); }
void cXMLNode::SetValue(const char* value) { _value.String = value; _evalvalue(_value); }

cXMLNode* cXMLNode::_addnode(std::unique_ptr<cXMLNode> && n)
{
  khiter_t iter = _nodehash.Iterator(n->_name);
  if(!_nodehash.ExistsIter(iter))
    _nodehash.Insert(n->_name, _nodes.Length());
  else
  {
    cXMLNode* target = _nodes[_nodehash.GetValue(iter)].get();
    LLInsert<cXMLNode>(n.get(), target);
    _nodehash.SetValue(iter, _nodes.Length());
  }
    
  _nodes.Add(std::move(n));
  return _nodes.Back().get();
}
cXMLValue* cXMLNode::_addattribute(cXMLValue && v)
{
  khiter_t iter = _attrhash.Iterator(v.Name);
  if(!_attrhash.ExistsIter(iter)) {
    _attributes.Add(std::move(v));
    _attrhash.Insert(_attributes.Back().Name, _attributes.Length() - 1);
    return &_attributes.Back();
  }
  size_t i = _attrhash.GetValue(iter);
  _attributes[i] = std::move(v);
  return &_attributes[i];
}

// Instead of including an entire regex library, we just do a dead simple pattern match of the form "<characters>*<characters>" where * is put into out.
bool cXMLNode::_match(std::istream& stream, cStr& out, const char* pattern, bool reset)
{
  while(stream.peek() != -1 && stream && isspace(stream.peek())) stream.get(); //eat whitespace
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

  if(!pattern[i]) { if(reset) stream.seekg(start); return true; }
  if(stream.peek() == -1 || !stream) { stream.seekg(start); return false; }
  int end = ++i; // increment by 1 to skip past * marker
  std::streamoff mid = stream.tellg();
  out.clear();

  while(stream.peek() != -1 && stream && pattern[end]!=0)
  {
    end = (stream.peek() == pattern[end])?(end+1):i;
    out += stream.get();
  }
  out.resize(i-end+out.length()); // trim off ending
  if(!pattern[end]) { if(reset) stream.seekg(mid); return true; } // If we got through the entire pattern, the match was a success (even if the file ended).
  stream.seekg(start); // otherwise we failed so reset everything
  return false;
}

bool cXMLNode::_parse(std::istream& stream, cStr& buf)
{
  while(_match(stream, buf, "<!--*-->"));
  if(!_match(stream, buf, "<*>")) return false;
  if(buf.length() > 0 && buf.back() == '/') // this is a self-closing tag
  { 
    buf.resize(buf.length()-1); // chop off the /
    _parseattribute(buf);
    return true;
  }
  _parseattribute(buf);
  _parseinner(stream, buf);
  if(stream.peek() == -1 || !stream) return true;
  if(buf.Trim() != _name) // If the end tag is not a legal end tag, someone didn't close something properly, so we need to backtrack
    stream.seekg(-2-buf.length(), std::ios_base::cur);
  return true;
}

void cXMLNode::_parseinner(std::istream& stream, cStr& buf)
{
  _value.String.clear();
  bool end = false;
  while(stream.peek() != -1 && stream && !end) // Run through the content, extracting and parsing all child nodes, until we hit an end tag
  {
    switch(stream.peek())
    {
    case '&':
      _parseentity(stream, _value.String);
      continue;
    case '<':
      if(_match(stream, buf, "</*>")) { end = true; continue; }
      if(_match(stream, buf, "<!--*-->")) continue;
      std::unique_ptr<cXMLNode> node(new cXMLNode());
      if(node->_parse(stream, buf))
      {
        _addnode(std::move(node));
        continue; // only continue if we successfully parsed SOMETHING. 
      } // Otherwise, the < was malformed, so we just add it as a character.
    }
    _value.String += stream.get();
  }
  _evalvalue(_value);
}

void cXMLNode::_parseattribute(cStr& buf)
{
  buf = buf.Trim();
  size_t i;
  for(i = 0; i < buf.length() && !isspace(buf[i]); ++i);
  _name.assign(buf, 0, i);
  while(i < buf.length() && isspace(buf[i])) ++i;
  if(i >= buf.length()) return;
  const char* c = ((const char*)buf) + i;

  std::istringstream ss(c);
  while(ss.peek() != -1 && ss) 
  {
    cXMLValue v;
    if(_match(ss, v.Name, "*="))
    {
      v.Name = v.Name.Trim();
      if(!_match(ss, v.String, "\"*\"")) // If we can't get a valid string just eat everything until the next space.
      {
        while(ss.peek() != -1 && ss && !isspace(ss.peek())) v.String += ss.get();
      }

      _evalvalue(v);
    } else { // If we can't find an equals sign, we'll just have to eat the whole thing.
      while(ss.peek() != -1 && ss && !isspace(ss.peek())) v.Name += ss.get();
    }
    _addattribute(std::move(v));
  }
}

void cXMLNode::_parseentity(std::istream& stream, cStr& target)
{
  cStr buf;
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

void cXMLNode::_evalvalue(cXMLValue& val)
{
  char* c;
  val.Integer = strtoull(val.String, &c, 10);
  val.Float = atof(val.String);
}

void cXMLNode::_writeattribute(std::ostream& stream) const
{
  stream << _name;

  for(size_t i = 0; i < _attributes.Length(); ++i)
  {
    stream << ' ' << _attributes[i].Name << "=\"";
    _writestring(stream, _attributes[i].String);
    stream.put('"');
  }
}

void cXMLNode::_writestring(std::ostream& stream, const char* s)
{
  while(*s)
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
    ++s;
  }
}

void cXMLNode::_write(std::ostream& stream, bool pretty, int depth) const
{
  if(pretty)
  {
    stream << std::endl;
    for(int i = 0; i < depth; ++i) stream.put('\t');
  }
  stream.put('<');
  _writeattribute(stream);
  bool content = _value.String.Trim().length() > 0;
  if(!content && !_nodes.Length()) // If there is no content and no nodes, assume this is a self-closing tag.
  {
    stream << "/>";
    return;
  }

  stream.put('>');
  if(content) _writestring(stream, _value.String);

  for(size_t i = 0; i < _nodes.Length(); ++i)
    _nodes[i]->_write(stream, pretty, depth+1);

  if(_nodes.Length()>0 && pretty)
  {
    stream << std::endl;
    for(int i = 0; i < depth; ++i) stream.put('\t');
  }
  
  stream << "</" << _name << ">";
}