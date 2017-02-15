// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "bss_util.h"

#ifndef __RWLOCK_H__BSS__
#define __RWLOCK_H__BSS__

#include "bss_compiler.h"
#include "lockless.h"
#include <atomic>
#include <assert.h>
#ifdef BSS_DEBUG
#include <thread>
#include <unordered_map>
#endif

namespace bss_util {
  // Write-preferring Readers-Writer lock, although writer starvation is still theoretically possible if an infinite number of new readers attempt to acquire the lock
  class BSS_COMPILER_DLLEXPORT RWLock {
  public:
    static_assert(ATOMIC_POINTER_LOCK_FREE == 2, "This lock does not function properly on this architecture!");
    inline RWLock() : l(0) // Note: if required, the fetch_or behavior can be emulated by checking a write lock, incrementing the read lock, then checking the write lock again.
    {
      assert(std::atomic_is_lock_free(&l));
      assert(l.load(std::memory_order_relaxed) == 0);
#ifdef BSS_DEBUG
      _debuglock.clear(std::memory_order_release);
#endif
    } 
    inline ~RWLock() { }

    // Acquire write lock
    BSS_FORCEINLINE void Lock() noexcept
    {
      //assert(debugemplace());
      
      while(asmbts<size_t>((size_t*)&l, WBIT)); // While the returned value includes the flag bit, another writer is performing an operation
      while((l.load(std::memory_order_relaxed)&WMASK) > 0); // Wait for any remaining readers to flush
    }

    // Attempts to acquire the lock, but if another writer already got the lock, aborts the attempt.
    BSS_FORCEINLINE bool AttemptLock() noexcept
    {
      if(asmbts<size_t>((size_t*)&l, WBIT)) // If another writer already has the lock, give up
        return false;
      //assert(debugemplace());
      while((l.load(std::memory_order_relaxed)&WMASK) > 0); // Wait for any remaining readers to flush
      return true;
    }

    // Attempts to acquire a write lock, but fails if there are any readers or writers
    BSS_FORCEINLINE bool AttemptStrictLock() noexcept
    {
      size_t prev = 0;
      bool r = l.compare_exchange_strong(prev, WFLAG, std::memory_order_release);
      //if(r) assert(debugemplace());
      return r;
    }

    // Release write lock
    BSS_FORCEINLINE void Unlock() noexcept
    { 
      //assert(debugerase() == 1);
      assert(l.load(std::memory_order_relaxed)&WFLAG);
      l.fetch_and(WMASK, std::memory_order_release);
    }

    // Acquire read lock
    BSS_FORCEINLINE void RLock() noexcept
    {
      //assert(debugemplace());
      while(l.fetch_add(ONE_READER, std::memory_order_acquire)&WFLAG) // Oppurtunistically acquire a read lock and check to see if the writer flag is set
      {
        if(l.fetch_sub(ONE_READER, std::memory_order_release)&WFLAG) // If the writer flag is set, release our lock to let the writer through
          while(l.load(std::memory_order_relaxed)&WFLAG); // Wait until the writer flag is no longer set before looping for another attempt
      }
    }

    // Attempt to acquire read lock, but abort if a writer has locked it.
    BSS_FORCEINLINE bool AttemptRLock() noexcept
    {
      if(l.fetch_add(ONE_READER, std::memory_order_acquire)&WFLAG) // Oppurtunistically acquire a read lock and check to see if the writer flag is set
      {
        l.fetch_sub(ONE_READER, std::memory_order_release);
        return false;
      }
      //assert(debugemplace());
      return true;
    }
    BSS_FORCEINLINE size_t RUnlock() noexcept
    { 
      //assert(debugerase() == 1);
      assert((l.load(std::memory_order_relaxed)&WMASK) > 0);
      return l.fetch_sub(ONE_READER, std::memory_order_release);
    }

    // Upgrades a read lock to a write lock without releasing the read lock. You CANNOT release this write lock using Unlock(), you have to use Downgrade() followed by RUnlock().
    BSS_FORCEINLINE void Upgrade() noexcept
    {
      //assert(debugcount() == 1);
      assert((l.load(std::memory_order_relaxed)&WMASK) > 0);
      while(asmbts<size_t>((size_t*)&l, WBIT)); // Acquire the write lock with acquire/release semantics, because this must have come after an initial RLock()
      while((l.load(std::memory_order_relaxed)&WMASK) > ONE_READER); // Only flush to a single read lock, which will be our own
    }

    // Attempts to upgrade a read lock to a write lock, but aborts if an existing writer has already locked it.
    BSS_FORCEINLINE bool AttemptUpgrade() noexcept
    {
      assert((l.load(std::memory_order_relaxed)&WMASK) > 0);
      if(asmbts<size_t>((size_t*)&l, WBIT))
        return false;
      //assert(debugcount() == 1);
      while((l.load(std::memory_order_relaxed)&WMASK) > ONE_READER); // Only flush to a single read lock, which will be our own
      return true;
    }

    // Releases a write lock without releasing the read lock. MUST be called after calling Upgrade()
    BSS_FORCEINLINE void Downgrade() noexcept
    {
      //assert(debugcount() == 1);
      assert(l.load(std::memory_order_relaxed)&WFLAG);
      l.fetch_and(WMASK, std::memory_order_release); // Even though this actually does the same thing as Unlock, we force you to call this function instead so the debug checks are valid.
    }

    BSS_FORCEINLINE size_t ReaderCount() noexcept
    {
      return l.load(std::memory_order_acquire)&WMASK;
    }

    static const size_t WBIT = (sizeof(size_t) << 3) - 1;
    static const size_t WFLAG = (size_t(1) << WBIT);
    static const size_t WMASK = ~WFLAG;
    static const size_t ONE_READER = 1;

    bool IsFree() const { return !l.load(std::memory_order_relaxed); }

  protected:
    std::atomic<size_t> l;

#ifdef BSS_DEBUG
    bool debugemplace() { while(_debuglock.test_and_set(std::memory_order_acquire)); bool r = _debug.emplace(std::this_thread::get_id(), 0).second; _debuglock.clear(std::memory_order_release); return r; }
    size_t debugerase() { while(_debuglock.test_and_set(std::memory_order_acquire)); size_t r = _debug.erase(std::this_thread::get_id()); _debuglock.clear(std::memory_order_release); return r; }
    size_t debugcount() { while(_debuglock.test_and_set(std::memory_order_acquire)); size_t r = _debug.count(std::this_thread::get_id()); _debuglock.clear(std::memory_order_release); return r; }
    std::unordered_map<std::thread::id, int> _debug;
    std::atomic_flag _debuglock;
#endif
  };
}


#endif
