import threading, collections

class QLock(object):

    __speech_lock = None

    def __init__(self):
        self.lock = threading.Lock()
        self.waiters = collections.deque()
        self.count = 0

    def __enter__(self):
        self.acquire()

    def __exit__(self, type, value, traceback):
        self.release()

    def acquire(self):
        self.lock.acquire()
        if self.count:
            new_lock = threading.Lock()
            new_lock.acquire()
            self.waiters.append(new_lock)
            self.lock.release()
            new_lock.acquire()
            self.lock.acquire()
        self.count += 1
        self.lock.release()

    def release(self):
        with self.lock:
            if not self.count:
                raise ValueError("lock not acquired")
            self.count -= 1
            if self.waiters:
                self.waiters.popleft().release()

    def locked(self):
        return self.count > 0

    @classmethod
    def getQLock(cls):
        if cls.__speech_lock is None:
            cls.__speech_lock = QLock()
        return cls.__speech_lock


# if __name__ == '__main__':
#     lk1 = QLock.getQLock()
#     print(lk1)
#     lk2 = QLock.getQLock()
#     print(lk2)
#     print(lk1 == lk2)
#     print(lk1 is lk2)
#
#     #<__main__.QLock object at 0x00000220EF3FB080>
#     #<__main__.QLock object at 0x00000220EF3FB080>
#     #True
#     #True