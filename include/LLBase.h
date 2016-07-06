// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "bss_util.h"

#ifndef __LL_BASE_H__BSS__
#define __LL_BASE_H__BSS__

#include "bss_defines.h"
#include <iterator>
#include <stddef.h>

namespace bss_util {
  // A base node for a doubly-linked list. The given parameter T may be any class that publically inherits LLBase
  template<typename T>
  struct BSS_COMPILER_DLLEXPORT LLBase
  {
    T* next;
    T* prev;
  };

  // Does a full insert, calling LLInsertAssign and LLInsert
  template<typename T>
  inline void BSS_FASTCALL LLInsert(T* node, T* target, T*& root) noexcept
  {
		node->prev = target->prev;
		node->next = target;
		if(target->prev != 0) target->prev->next = node;
		else root = node;
		target->prev = node;
  }

  // Does a full insert, calling LLInsertAssign and LLInsert, but doesn't re-assign root
  template<typename T>
  inline void BSS_FASTCALL LLInsert(T* node, T* target) noexcept
  {
		node->prev = target->prev;
		node->next = target;
		if(target->prev != 0) target->prev->next = node;
		target->prev = node;
  }
  
  // Does a full insert before the root
  template<typename T>
  inline void BSS_FASTCALL LLAdd(T* node, T*& root) noexcept
  {
    node->prev=0;
    node->next=root;
    if(root) root->prev=node;
    root=node;
  }
  
  template<typename T>
  inline void BSS_FASTCALL LLAdd(T* node, T*& root, T*& last) noexcept
  {
    node->prev=0;
    node->next=root;
    if(root) root->prev=node;
    else last=node;
    root=node;
  }

  // Does a full insert, calling LLInsertAssign and LLInsert
  template<typename T>
  inline void BSS_FASTCALL LLInsertAfter(T* node, T* target, T*& last) noexcept
  {
		node->next = target->next;
		node->prev = target;
		if (target->next != 0) target->next->prev = node;
		else last = node;
		target->next = node;
  }

  // Does a full insert, calling LLInsertAssign and LLInsert, but doesn't re-assign last
  template<typename T>
  inline void BSS_FASTCALL LLInsertAfter(T* node, T* target) noexcept
  {
		node->next = target->next;
		node->prev = target;
		if (target->next != 0) target->next->prev = node;
		target->next = node;
  }

  // Used in the form last = LLAddAfter(node,last); where last must be the last element of a linked list, defined by last->next = 0.
  template<typename T>
  inline T* BSS_FASTCALL LLAddAfter(T* node, T* last) noexcept
  {
		node->next = 0;
		node->prev = last;
    last->next = node;
    return node;
  }

  template<typename T>
  inline void BSS_FASTCALL LLAddAfter(T* node, T*& root, T*& last) noexcept
  {
    node->next = 0;
    node->prev = last;
    if(last) last->next = node;
    else root = node;
    last = node;
  }

  // Removes a node from a list, re-assigning root and last as necessary. Does not zero values on node
  template<typename T>
  inline void BSS_FASTCALL LLRemove(T* node, T*& root, T*& last) noexcept
  {
		if(node->prev != 0) node->prev->next = node->next;
		else if(root == node) root = node->next;
		if(node->next != 0) node->next->prev = node->prev;
		else if(last == node) last = node->prev;
  }

  // Removes a node from a list, re-assigning root as necessary. Does not zero values on node
  template<typename T>
  inline void BSS_FASTCALL LLRemove(T* node, T*& root) noexcept
  {
		if(node->prev != 0) node->prev->next = node->next;
		else if(root == node) root = node->next;
		if(node->next != 0) node->next->prev = node->prev;
  }

  // Removes a node from a list. Does not zero values on node
  template<typename T>
  inline void BSS_FASTCALL LLRemove(T* node) noexcept
  {
		if(node->prev != 0) node->prev->next = node->next;
		if(node->next != 0) node->next->prev = node->prev;
  }

  // Reimplementations of LLAdd and LLRemove generalized to any data structure instead of requiring it to be inherited.
  template<typename T, LLBase<T>& (*GETNODES)(T* n)>
  BSS_FORCEINLINE void BSS_FASTCALL AltLLAdd(T* node, T*& root) noexcept
  {
    GETNODES(node).prev=0;
    GETNODES(node).next=root;
    if(root) GETNODES(root).prev=node;
    root=node;
  }
  template<typename T, LLBase<T>& (*GETNODES)(T* n)>
  inline void BSS_FASTCALL AltLLRemove(T* node, T*& root) noexcept
  {
		if(GETNODES(node).prev != 0) GETNODES(GETNODES(node).prev).next = GETNODES(node).next;
		else if(root == node) root = static_cast<T*>(GETNODES(node).next);
		if(GETNODES(node).next != 0) GETNODES(GETNODES(node).next).prev = GETNODES(node).prev;
  }

  // Iterator for doubly linked list where the item is itself. Does not support remove; use postfix-- or the equivelent
  template<typename T>
  class BSS_COMPILER_DLLEXPORT LLIterator : public std::iterator<std::bidirectional_iterator_tag,T,ptrdiff_t,T*,T*>
	{
  protected:
    typedef typename std::iterator<std::bidirectional_iterator_tag,T,ptrdiff_t,T*,T*>::pointer pointer;
    typedef typename std::iterator<std::bidirectional_iterator_tag,T,ptrdiff_t,T*,T*>::reference reference;
  public:
    inline LLIterator() : cur(0) {}
    inline explicit LLIterator(T* node) : cur(node) { }
    inline reference operator*() const { return cur; }
    inline pointer operator->() const { return cur; }
    inline LLIterator& operator++() { cur=cur->next; return *this; } //prefix
    inline LLIterator operator++(int) { LLIterator r=*this; ++*this; return r; } //postfix
    inline LLIterator& operator--() { cur=cur->prev; return *this; } //prefix
    inline LLIterator operator--(int) { LLIterator r=*this; --*this; return r; } //postfix
    inline bool operator==(const LLIterator& _Right) const { return (cur == _Right.cur); }
	  inline bool operator!=(const LLIterator& _Right) const { return (cur != _Right.cur); }
    inline bool operator!() const { return !cur; }
    inline bool IsValid() { return cur!=0; }

    T* cur;
	};
}



#endif
