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

## Enumerate

```python
l1 = ["eat","sleep","repeat"] 
s1 = "geek"
  
# creating enumerate objects 
obj1 = enumerate(l1) 
obj2 = enumerate(s1) 
  
print ("Return type:",type(obj1) )
print (list(enumerate(l1)) )
  
# changing start index to 2 from 0 
print (list(enumerate(s1,2)))
## Output:
Return type: < type 'enumerate' >
[(0, 'eat'), (1, 'sleep'), (2, 'repeat')]
[(2, 'g'), (3, 'e'), (4, 'e'), (5, 'k')]

 ```
