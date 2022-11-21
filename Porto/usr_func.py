import multiprocessing
import time
import numpy as np
import sys


def daemon():
    p = multiprocessing.current_process()
    print("Starting: ", p.name, p.pid)
    sys.stdout.flush()
    time.sleep(10)
    print("Exiting: ", p.name, p.pid)
    sys.stdout.flush()

def non_daemon():
    p = multiprocessing.current_process()
    print("Starting: ", p.name, p.pid)
    sys.stdout.flush()
    print("Exiting: ", p.name, p.pid)
    sys.stdout.flush()


def test(x, y):

    # print(multiprocessing.current_process())
    return x * y
    # return np.linalg.inv(x)

def worker():
    name = multiprocessing.current_process().name
    print(name, "Starting")
    time.sleep(2)
    print(name, "Exiting")

def my_service():
    name = multiprocessing.current_process().name
    print(name, "Starting")
    time.sleep(3)
    print(name, "Exiting")


import os
def foo(q):
    q.put("hello")

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())


def f(name = "China"):
    # info('function f')
    print('hello', name)

import time
import queue
from multiprocessing import Lock, Process, Queue, current_process

def do_job(tasks_to_accomplish, tasks_that_are_done):
    while True:
        try:
            '''

            '''
            task = tasks_to_accomplish.get_nowait()
        except queue.Empty:
            break
        else:
            '''
            '''
            print(task)
            tasks_that_are_done.put(task + " is done by " + current_process().name)
            time.sleep(1)

    return True

import random
def rand_num():
    num = random.random()
    print(num)

def my_func(x):
    print(current_process())
    return x ** x




