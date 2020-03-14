# My-Usful-Codes

```python
import itertools
[f[i][j] for i,j in itertools.product(range(t), range(int(n)))]
```
```python
import time
from functools import wraps

def time_of_excecution(func):
  @wraps func
  def wrapper (*args, **kwargs):
    t_s=time.time()
    result=func(*args, **kwargs)
    t_e=time.time()
    print(func.__name__,t_e-t_s)
    return result
  return wrapper
@time_of_excecution
def counting(n):
  while n<500000
  n+=1
   
```
