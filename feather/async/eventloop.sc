

using import struct
using import .event

struct Event

let insertionPoint = (mutable@ (mutable@ Event))

struct EventLoop
    _running : bool = false
    _last-runnable-state : bool = false
    _head : (mutable@ Event)
    _tail : insertionPoint
    _depth-first-insert-point : insertionPoint
    _breadth-first-insert-point : insertionPoint
    _currently-firing : (mutable@ Event)

    fn running? (self)
        self._running

    fn turn (self)
        local event = self._head
        if (event == null)
            false
        else
            self._head = event._next
            if (self._head != null)
                event._prev = (& self._head)
            self._depth-first-insert-point = (& self._head)

            if (self._breadth-first-insert-point == event._next)
                self._breadth-first-insert-point = (& self.head)
            if (self._tail == event._next)
                self._tail = (& self._head)

            event._next = null
            event._prev = null

            event._firing = true
            self._currently-firing = event
            'fire event

            event._currently-firing = null
            event._firing = false

            self._depth-first-insert-point = (& self._head)

            true

    fn run (self max-turn-count)
        self._running = true
        for x in (range max-turn-count)
            if (not ('turn self))
                break;

        self._running = false
        'set-runnable self ('runnable? self)

    fn runnable? (self)
        self.head != null

    fn set-runnable (self runnable)
        if (runnable != self._last-runnable-state)
            #TODO: if there is a port, update the port
        self._last-runnable-state = runnable



global thread-local-event-loop : (@ EventLoop) # need to actually make this thread-local at some point


struct Event
    _fire-impl : (Capture (Capture.function (@ this-type)))
    _loop : (mutable@ EventLoop)
    _next : (mutable@ Event)
    _prev : (mutable@ (mutable@ Event))
    _firing : bool = false

    _live : u32 = magic-live-value
    _source-location : Anchor

    let magic-live-value = 0x1e366381:u32

    """"An event waiting to be executed.
        Not for direct use by applications --
        promises use this internally.

    fn fire (self)
        """"Fire the event.
            Possibly returns a pointer to itself,
            which will be discarded by the caller.
            This is the only way that an event can delete itself as a result of firing,
            as doing so from within fire() will throw an exception.
        self._fire-impl (& self)
        drop self._fire-impl
        self._live = 0
        free self

    fn arm-depth-first (self)
        """"Enqueue this event so that `fire()` will be called from the event loop soon.

            Events scheduled in this way are executed in depth-first order:
            if an event callback arms more events,
            those events are placed at the front of the queue (in the order in which they were armed),
            so that they run immediately after the first event's callback returns.

            Depth-first event scheduling is appropriate for events that represent simple continuations of a previous event that should be globbed together for performance.
            Depth-first scheduling can lead to starvation,
            so any long-running task must occasionally yield with `armBreadthFirst()`.
            (Promise::then() uses depth-first whereas evalLater() uses breadth-first.)

            To use breadth-first scheduling instead,
            use `armBreadthFirst()`.
        if (and (ptrcmp!= thread-local-event-loop self._loop) (ptrcmp!= thread-local-event-loop null))
            report "FATAL: Event armed from different thread than it was created in. You must use Executor to queue events cross-thread."
            abort;
        elseif (!= self._live magic-live-value)
            report "ASSERTION FAILURE: tried to arm Event after it was destroyed" # should log event source location?
            abort;
        elseif (ptrcmp== self._prev null)
            self._next = (@ self._loop._depth-first-insert-point)
            self._prev = self._loop._depth-first-insert-point
            (@ self._prev) = (& self)
            if (ptrcmp!= self._next null)
                self._next._prev = (& self._next)

            self._loop._depth-first-insert-point = (& self._next)

            if (ptrcmp== self._loop._breadth-first-insert-point self._prev)
                self._loop._breadth-first-insert-point = (& self._next)
            if (ptrcmp== self._loop._tail self._prev)
                self._loop._tail = (& self._next)

            'set-runnable self._loop true

    fn arm-breadth-first (self)
        """"Like `armDepthFirst()` except that the event is placed at the end of the queue.
        if (and (ptrcmp!= thread-local-event-loop self._loop) (ptrcmp!= thread-local-event-loop null))
            report "FATAL: Event armed from different thread than it was created in. You must use Executor to queue events cross-thread."
            abort;
        elseif (!= self._live magic-live-value)
            report "ASSERTION FAILURE: tried to arm Event after it was destroyed" # should log event source location?
            abort;
        elseif (ptrcmp== self._prev null)
            self._next = (@ self._loop._breadth-first-insert-point)
            self._prev = self._loop._breadth-first-insert-point
            (@ self._prev) = (& self)
            if (ptrcmp!= self._next null)
                self._next._prev = (& self._next)

            self._loop._breadth-first-insert-point = (& self._next)

            if (ptrcmp== self._loop._tail self._prev)
                self._loop._tail = (& self._next)

            'set-runnable self._loop true

    fn arm-last (self)
        """"Enqueues this event to happen after all other events have run to completion and there is really nothing left to do except wait for I/O.
        if (and (ptrcmp!= thread-local-event-loop self._loop) (ptrcmp!= thread-local-event-loop null))
            report "FATAL: Event armed from different thread than it was created in. You must use Executor to queue events cross-thread."
            abort;
        elseif (!= self._live magic-live-value)
            report "ASSERTION FAILURE: tried to arm Event after it was destroyed" # should log event source location?
            abort;
        elseif (ptrcmp== self._prev null)
            self._next = (@ self._loop._breadth-first-insert-point)
            self._prev = self._loop._breadth-first-insert-point
            (@ self._prev) = (& self)
            if (ptrcmp!= self._next null)
                self._next._prev = (& self._next)

            # We don't update self._loop._breadth-first-insert-point because we want further inserts to go *before* this event.

            if (ptrcmp== self._loop._tail self._prev)
                self._loop._tail = (& self._next)

            'set-runnable self._loop true

    fn next? (self)
        """"True if the Event has been armed and is next in line to be fired.
            This can be used after calling PromiseNode::onReady(event) to determine if a promise being waited is immediately ready,
            in which case continuations may be optimistically run without returning to the event loop.
            Note that this optimization is only valid if we know that we would otherwise immediately return to the event loop without running more application code.
            So this turns out to be useful in fairly narrow circumstances,
            chiefly when a coroutine is about to suspend,
            but discovers it doesn't need to.

            Returns false if the event loop is not currently running.
            This ensures that promise
            continuations don't execute except under a call to .wait().
        and ('running? self._loop) (ptrcmp== self._loop._head (& self))

    fn disarm (self)
        """"If the event is armed but hasn't fired,
            cancel it.
            (Destroying the event does this implicitly.)
        if (ptrcmp!= self._prev null)
            if (and (ptrcmp!= thread-local-event-loop self._loop) (ptrcmp!= thread-local-event-loop null))
                report "FATAL: Promise destroyed from a different thread than it was created in."
                abort;
            else
                if (ptrcmp== self._loop._tail (& self._next))
                    self._loop._tail = self._prev
                if (ptrcmp== self._loop._depth-first-insert-point (& self._next))
                    self._loop._depth-first-insert-point = self._prev
                if (ptrcmp== self._loop._breadth-first-insert-point (& self._next))
                    self._loop._breadth-first-insert-point = self._prev

                (& self._prev) = self._next
                if (ptrcmp!= self._next null)
                    self._next._prev = self._prev

                self._prev = null
                self._next = null

    fn build-event-trace (self trace-builder)
        """"Build a trace of the callers leading up to this event.
            `builder` will be populated with "return addresses" of the promise chain waiting on this event.
            The return addresses may actually the addresses of lambdas passed to .then(),
            but in any case,
            feeding them into addr2line should produce useful source code locations.

            `traceEvent()` may be called from an async signal handler while `fire()` is executing.
            It must not allocate nor take locks.

    fn trace-event
        """"Helper that builds a trace and stringifies it.

locals;
