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
  while (n<500000):
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
  
print ("Return type:",type(obj1) )
print (list(enumerate(l1)) )
  
# changing start index to 2 from 0 
print (list(enumerate(s1,2)))

## Output:

Return type: < type 'enumerate' >
[(0, 'eat'), (1, 'sleep'), (2, 'repeat')]
[(2, 'g'), (3, 'e'), (4, 'e'), (5, 'k')]
 ```
## Pivot_Table

```python
Table=pd.pivot_table(data,values=['CSSR','DCR],index=['Province'], aggfunc={'CSSR':[max, min],'DCR': [lambda x: np.percentage(x,90)]}
```
# Print types
```python
print(*range(10),sep ='')
[print('YES' if re.match(r'[789]\d{9}$',input()) else 'NO' for _ in range(int(input))] #To find Phone Number

List=['T','A','N','H','A']
print(''.join(List))

print('%s%s' % (s1,s2))
print("{}{}".format(s1,s2)
print(f'{s1}{s2})

```

# General Code
```python
string = "geeks for geeks geeks geeks geeks" 
print(string.replace("geeks", "GeeksforGeeks", 3)) 
    # output: GeeksforGeeks for GeeksforGeeks GeeksforGeeks geeks geeks
list.remove(element) # Remove the first element in the list
del list[start:end] # Delete a range of elements
```

# Define a function that takes variable number of arguments
```python
def func (*var):
  for i in var:
    print(i)

func(1)
func(2,4,56)
```
# Append, Concatenate, Vstack

![test](https://github.com/m-r-tanha/My-Usful-Codes/blob/master/append.png)

the vstack is the correct answer, 
the concatenate works on two same dimensional array, 
and append doesn't work on array

# Rgression and Classification
![class_regra](https://github.com/m-r-tanha/My-Usful-Codes/blob/master/classification_regression.png)

# Read CSV file from a link
```python
link=https://doc.google.com/...
source=StringIO.StringIO(requests.get(link).content))
data=pd.read_csv(source)
```
