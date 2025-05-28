import sys
from contextlib import contextmanager
import os
@contextmanager
def dis_print():
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout = original_stdout
        
if __name__=="__main__":
    with dis_print():
        print("test1")
    print("test2")
