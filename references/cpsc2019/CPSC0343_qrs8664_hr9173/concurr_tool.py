from multiprocessing import Pool, Manager
from array_tool import queue_sort


def _task(idx, task, params, queue):
    print('task idx=', idx)
    result = task(*params)
    queue.put((idx, result))


class MultiTask:

    def __init__(self, pool_size, queue_size):
        self.pool = Pool(processes=pool_size)
        self.queue = Manager().Queue(maxsize=queue_size)

    def submit(self, idx, task, params):
        self.pool.apply_async(_task, args=(idx, task, params, self.queue))

    '''finish submitting and wait for the result'''
    def subscribe(self):
        self.pool.close()
        self.pool.join()

        result = queue_sort(self.queue)
        return result

