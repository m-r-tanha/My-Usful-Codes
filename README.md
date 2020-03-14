# My-Usful-Codes

```python
import itertools
[f[i][j] for i,j in itertools.product(range(t), range(int(n)))]
```
## Decorative Wrapper

```python
import time
from functools import wraps

def time_of_excecution(func):
  @wraps (func)
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
## Default Function

```python
 def add (a, b, a) -> float: print(a+b+c)
 ```

## Open a file with "with"
#### with help us to close a open whenever we don't want thsi function

```python
with open('path.txt') as f:
  read_file = f.read()
 ```

## Request and Post

```python
import requests
mydata={'name':'Mohammad','email':'mohammad@exlampl.com'} #the require data can be seen in browser inspector
r_g = requests.get("URL")
r_p = request.post("URL") #it is useful for API
print(r_g.text)
f=open("./page.html","w+")
f.write(r_g.text)
 ```
