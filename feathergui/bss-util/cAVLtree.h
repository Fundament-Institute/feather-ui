// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "bss_util.h"

#ifndef __C_AVLTREE_H__BSS__
#define __C_AVLTREE_H__BSS__

#include "bss_compare.h"
#include "bss_alloc.h"

namespace bss_util {
  template<class KeyData> // If data is non-void, KeyData is std::pair<Key,Data>, otherwise it's just Key
  struct BSS_COMPILER_DLLEXPORT AVL_Node
  {
    inline AVL_Node() : _key(), _left(0), _right(0), _balance(0) {} // Empty constructor is important so we initialize the data as empty if it exists.
#pragma warning(push)
#pragma warning(disable:4251)
    KeyData _key;
#pragma warning(pop)
    AVL_Node<KeyData>* _left;
    AVL_Node<KeyData>* _right;
    int _balance;

  private:
    inline AVL_Node(const AVL_Node&) BSS_DELETEFUNC // This is to keep the compiler from defining a copy or assignment constructor that could conflict with move-only structures, like unique_ptr
      AVL_Node& operator =(const AVL_Node&) BSS_DELETEFUNCOP
  };

  // Adaptive function definitions to allow for an optional Data field
  template<class Key, class Data>
  class BSS_COMPILER_DLLEXPORT _AVL_TREE_DATAFIELD
  {
  public:
    typedef std::pair<Key, Data> KeyData;
    typedef AVL_Node<KeyData> AVLNode;
    typedef Data KEYGET;
    typedef Data DATAGET;

  protected:
    BSS_FORCEINLINE static void BSS_FASTCALL _setdata(AVLNode* BSS_RESTRICT old, AVLNode* BSS_RESTRICT cur) { assert(old != cur); if(cur != 0) cur->_key.second = std::move(old->_key.second); }
    template<typename U>
    BSS_FORCEINLINE static void BSS_FASTCALL _setraw(U && data, AVLNode* cur) { if(cur != 0) cur->_key.second = std::forward<U>(data); }
    BSS_FORCEINLINE static KEYGET& BSS_FASTCALL _getdata(KeyData& cur) { return cur.second; }
    BSS_FORCEINLINE static Key& BSS_FASTCALL _getkey(KeyData& cur) { return cur.first; }
    BSS_FORCEINLINE static void BSS_FASTCALL _swapdata(AVLNode* BSS_RESTRICT retval, AVLNode* BSS_RESTRICT root)
    {
      assert(retval != root);
      if(retval != 0) // We have to actually swap the data so that retval returns the correct data so it can be used in ReplaceKey
      {
        Data h(std::move(retval->_key.second));
        retval->_key.second = std::move(root->_key.second);
        root->_key.second = std::move(h);
      }
    }
  };

  template<class Key>
  class BSS_COMPILER_DLLEXPORT _AVL_TREE_DATAFIELD<Key, void>
  {
  public:
    typedef Key KeyData;
    typedef AVL_Node<KeyData> AVLNode;
    typedef Key KEYGET;
    typedef char DATAGET;

  protected:
    BSS_FORCEINLINE static void BSS_FASTCALL _setdata(AVLNode* BSS_RESTRICT old, AVLNode* BSS_RESTRICT cur) { }
    template<typename U>
    BSS_FORCEINLINE static void BSS_FASTCALL _setraw(U && data, AVLNode* cur) { }
    BSS_FORCEINLINE static KEYGET& BSS_FASTCALL _getdata(KeyData& cur) { return cur; }
    BSS_FORCEINLINE static Key& BSS_FASTCALL _getkey(KeyData& cur) { return cur; }
    BSS_FORCEINLINE static void BSS_FASTCALL _swapdata(AVLNode* BSS_RESTRICT retval, AVLNode* BSS_RESTRICT root) { }
  };

  // AVL Tree implementation
  template<class Key, class Data, char(*CFunc)(const Key&, const Key&) = CompT<Key>, typename Alloc = StandardAllocPolicy<typename _AVL_TREE_DATAFIELD<Key, Data>::AVLNode>>
  class BSS_COMPILER_DLLEXPORT cAVLtree : protected cAllocTracker<Alloc>, public _AVL_TREE_DATAFIELD<Key, Data>
  {
    cAVLtree(const cAVLtree& copy) BSS_DELETEFUNC
      cAVLtree& operator=(const cAVLtree& copy) BSS_DELETEFUNCOP
  protected:
    typedef _AVL_TREE_DATAFIELD<Key, Data> BASE;
    typedef typename BASE::KeyData KeyData;
    typedef typename BASE::AVLNode AVLNode;
    typedef typename BASE::KEYGET KEYGET;
    typedef typename BASE::DATAGET DATAGET;

  public:
    inline cAVLtree(cAVLtree&& mov) : cAllocTracker<Alloc>(std::move(mov)), _root(mov._root) { mov._root = 0; }
    inline cAVLtree(Alloc* allocator = 0) : cAllocTracker<Alloc>(allocator), _root(0) {}
    inline ~cAVLtree() { Clear(); }
    inline void Clear() { _clear(_root); _root = 0; }
    inline AVLNode* GetRoot() { return _root; }
    template<typename F> // std::function<void(std::pair<key,data>)> (we infer _traverse's template argument here, otherwise GCC explodes)
    BSS_FORCEINLINE void Traverse(F lambda) { _traverse(lambda, _root); }
    BSS_FORCEINLINE AVLNode* BSS_FASTCALL Insert(Key key, const DATAGET& data)
    {
      char change = 0;
      AVLNode* cur = _insert(key, &_root, change);
      BASE::template _setraw<const DATAGET&>(data, cur); // WHO COMES UP WITH THIS SYNTAX?!
      return cur;
    }
    BSS_FORCEINLINE AVLNode* BSS_FASTCALL Insert(Key key, DATAGET&& data)
    {
      char change = 0;
      AVLNode* cur = _insert(key, &_root, change);
      BASE::template _setraw<DATAGET&&>(std::move(data), cur);
      return cur;
    }
    BSS_FORCEINLINE AVLNode* BSS_FASTCALL Insert(Key key)
    {
      char change = 0;
      return _insert(key, &_root, change);
    }
    BSS_FORCEINLINE AVLNode* BSS_FASTCALL Near(const Key key) const { return _near(key); }
    BSS_FORCEINLINE KEYGET BSS_FASTCALL Get(const Key key, const KEYGET& INVALID) const
    {
      AVLNode* retval = _find(key);
      return !retval ? INVALID : BASE::_getdata(retval->_key);
    }
    BSS_FORCEINLINE KEYGET* BSS_FASTCALL GetRef(const Key key) const
    {
      AVLNode* retval = _find(key);
      return !retval ? 0 : &BASE::_getdata(retval->_key);
    }
    inline bool BSS_FASTCALL Remove(const Key key)
    {
      char change = 0;
      AVLNode* node = _remove(key, &_root, change);
      if(node != 0)
      {
        node->~AVLNode();
        cAllocTracker<Alloc>::_deallocate(node, 1);
        return true;
      }
      return false;
    }
    inline bool BSS_FASTCALL ReplaceKey(const Key oldkey, const Key newkey)
    {
      char change = 0;
      AVLNode* old = _remove(oldkey, &_root, change);
      AVLNode* cur = _insert(newkey, &_root, change);
      if(old != 0)
      {
        BASE::_setdata(old, cur);
        old->~AVLNode();
        cAllocTracker<Alloc>::_deallocate(old, 1);
      }
      return (cur != 0);
    }

    inline cAVLtree& operator=(cAVLtree&& mov) { Clear(); _root = mov._root; mov._root = 0; return *this; }

  protected:
    template<typename F>
    BSS_FORCEINLINE static void BSS_FASTCALL _traverse(F lambda, AVLNode* node)
    {
      if(!node) return;
      _traverse<F>(lambda, node->_left);
      lambda(node->_key);
      _traverse<F>(lambda, node->_right);
    }

    inline void BSS_FASTCALL _clear(AVLNode* node)
    {
      if(!node) return;
      _clear(node->_left);
      _clear(node->_right);
      node->~AVLNode();
      cAllocTracker<Alloc>::_deallocate(node, 1);
    }

    static void BSS_FASTCALL _leftrotate(AVLNode** pnode)
    {
      AVLNode* node = *pnode;
      AVLNode* r = node->_right;

      node->_right = r->_left;
      r->_left = node;
      *pnode = r;

      r->_left->_balance -= (1 + bssmax(r->_balance, 0));
      r->_balance -= (1 - bssmin(r->_left->_balance, 0));
    }

    static void BSS_FASTCALL _rightrotate(AVLNode** pnode)
    {
      AVLNode* node = *pnode;
      AVLNode* r = node->_left;

      node->_left = r->_right;
      r->_right = node;
      *pnode = r;

      r->_right->_balance += (1 - bssmin(r->_balance, 0));
      r->_balance += (1 + bssmax(r->_right->_balance, 0));
    }
    AVLNode* BSS_FASTCALL _insert(Key key, AVLNode** proot, char& change) //recursive insertion function
    {
      AVLNode* root = *proot;
      if(!root)
      {
        root = cAllocTracker<Alloc>::_allocate(1);
        *proot = root;
        new(root) AVLNode();
        BASE::_getkey(root->_key) = key;
        change = 1;
        return root;
      }

      char result = CFunc(BASE::_getkey(root->_key), key);
      AVLNode* retval = 0;
      if(result<0)
        retval = _insert(key, &root->_left, change);
      else if(result>0)
        retval = _insert(key, &root->_right, change);
      else
      {
        change = 0;
        return 0;
      }
      result *= change;
      root->_balance += result;
      change = (result && root->_balance) ? (1 - _rebalance(proot)) : 0;
      return retval;
    }

    inline AVLNode* BSS_FASTCALL _find(const Key& key) const
    {
      AVLNode* cur = _root;
      while(cur)
      {
        switch(CFunc(BASE::_getkey(cur->_key), key)) //This is faster then if/else statements because FUCK IF I KNOW!
        {
        case -1: cur = cur->_left; break;
        case 1: cur = cur->_right; break;
        default: return cur;
        }
      }

      return 0;
    }

    inline AVLNode* BSS_FASTCALL _near(const Key& key) const
    {
      AVLNode* prev = 0;
      AVLNode* cur = _root;
      while(cur)
      {
        prev = cur;
        switch(CFunc(BASE::_getkey(cur->_key), key)) //This is faster then if/else statements because FUCK IF I KNOW!
        {
        case -1: cur = cur->_left; break;
        case 1: cur = cur->_right; break;
        default: return cur;
        }
      }

      return prev;
    }
    inline static char BSS_FASTCALL _rebalance(AVLNode** root)
    {
      AVLNode* _root = *root;
      char retval = 0;
      if(_root->_balance<-1)
      {
        if(_root->_left->_balance == 1) // LR
        {
          _leftrotate(&_root->_left);
          _rightrotate(root);
          retval = 1;
        }
        else
        { // LL 
          retval = !_root->_left->_balance ? 0 : 1;
          //retval = (_root->_left->_balance<0)+(_root->_left->_balance>0); //Does the same thing as above without branching (useless)
          _rightrotate(root);
        }
      }
      else if(_root->_balance>1)
      {
        if(_root->_right->_balance == -1)
        { //RL
          _rightrotate(&_root->_right);
          _leftrotate(root);
          retval = 1;
        }
        else
        { // RR
          retval = !_root->_right->_balance ? 0 : 1;
          //retval = (_root->_right->_balance<0)+(_root->_right->_balance>0); //Does the same thing as above without branching (useless)
          _leftrotate(root);
        }
      }
      return retval;
    }

    AVLNode* BSS_FASTCALL _remove(const Key& key, AVLNode** proot, char& change) //recursive removal function
    {
      AVLNode* root = *proot;
      if(!root)
      {
        change = 0;
        return 0;
      }

      char result = CFunc(BASE::_getkey(root->_key), key);
      AVLNode* retval = 0;
      if(result<0)
      {
        retval = _remove(key, &root->_left, change);
        result *= change;
      }
      else if(result>0)
      {
        retval = _remove(key, &root->_right, change);
        result *= change;
      }
      else
      {
        if(!root->_left && !root->_right) //leaf node
        {
          *proot = 0;
          change = 1;
          return root;
        }
        else if(!root->_left || !root->_right)
        { //one child
          *proot = !root->_left ? root->_right : root->_left;
          change = 1;
          return root;
        }
        else
        {
          AVLNode* successor = root->_right;
          while(successor->_left)
            successor = successor->_left;

          BASE::_getkey(root->_key) = BASE::_getkey(successor->_key);
          retval = _remove(BASE::_getkey(successor->_key), &root->_right, result); //this works because we're always removing something from the right side, which means we should always subtract 1 or 0.
          BASE::_swapdata(retval, root);

          change = 1;
        }
      }
      root->_balance -= result;

      change = (result) ? ((root->_balance) ? _rebalance(proot) : 1) : 0;
      return retval;
    }

    AVLNode* _root;
  };
}

#endif
