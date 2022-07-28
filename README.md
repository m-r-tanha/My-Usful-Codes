# My-Usful-Codes

## to Creat Latex Formula as Markdown (https://www.codecogs.com/latex/eqneditor.php)

```python
outlier = set(laser_data_cart.index).symmetric_difference(set(inlier_heat.index))
outlier = laser_data_cart[laser_data_cart.index.isin(outlier)]
```

```python
df.loc[df['column_name'] == some_value]
df['column_name'] >= A & df['column_name'] <= B
df.loc[df['column_name'] != some_value]
```

```python
import itertools
[f[i][j] for i,j in itertools.product(range(t), range(int(n)))]

[x for x in range(20) if x%2 == 0]

# Trick to replace two values
a=10
b=5
a,b = b,a

numpy.random.uniform(1,5,(2,3))
#output:
array([[1.7859482 , 1.85782785, 1.12609184],
       [4.97698711, 3.74835621, 1.23437712]])

a=np.random.standard_normal((2,2)) #generate distributed normal
np.mean(b)
np.median(b)
np.var(b) #varians
np.std(b) #standard deviation
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
## Change cell color in Excel file

![Code](https://github.com/m-r-tanha/My-Usful-Codes/blob/master/1.png)
![result](https://github.com/m-r-tanha/My-Usful-Codes/blob/master/0.png)

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
## Print types
```python
print(*range(10),sep ='')
[print('YES' if re.match(r'[789]\d{9}$',input()) else 'NO')] #To find Phone Number

List=['T','A','N','H','A']
print(''.join(List))

print('%s%s' % (s1,s2))
print("{}{}".format(s1,s2)
print(f'{s1}{s2})

print("FORMAT".format(NUMBER))

Number	       Format    	Output	       Description
3.1415926	    {:.2f}      3.14      	Format float 2 decimal places
3.1415926   	{:+.2f}	    +3.14	      Format float 2 decimal places with sign
-1	          {:+.2f}   	-1.00	      Format float 2 decimal places with sign
2.71828	      {:.0f}	     3	        Format float with no decimal places
5	            {:0>2d}    	05	        Pad number with zeros (left padding, width 2)
5	            {:x<4d}	    5xxx	      Pad number with x’s (right padding, width 4)
10	          {:x<4d}   	10xx        Pad number with x’s (right padding, width 4)
1000000	      {:,}	      1,000,000  	Number format with comma separator
0.25	        {:.2%}	    25.00%	    Format percentage
1000000000	  {:.2e}	    1.00e+09	  Exponent notation
13	          {:10d}	    13      	  Right aligned (default, width 10)
13	          {:<10d}	    13	        Left aligned (width 10)
13	          {:^10d}	    13	        Center aligned (width 10)

```

## General Code
```python
string = "geeks for geeks geeks geeks geeks" 
print(string.replace("geeks", "GeeksforGeeks", 3)) 
    # output: GeeksforGeeks for GeeksforGeeks GeeksforGeeks geeks geeks
list.remove(element) # Remove the first element in the list
del list[start:end] # Delete a range of elements
```

## Define a function that takes variable number of arguments
```python
def func (*var):
  for i in var:
    print(i)

func(1)
func(2,4,56)
```
## Append, Concatenate, Vstack

![test](https://github.com/m-r-tanha/My-Usful-Codes/blob/master/append.png)

the vstack is the correct answer, 
the concatenate works on two same dimensional array, 
and append doesn't work on array

## Rgression and Classification
![class_regra](https://github.com/m-r-tanha/My-Usful-Codes/blob/master/classification_regression.png)

## Read CSV file from a link
```python
link=https://doc.google.com/...
source=StringIO.StringIO(requests.get(link).content))
data=pd.read_csv(source)
```
## Sort an array by the (n-1)th column
```python
X=np.array([[1,2,3],[0,5,2],[2,3,4]])
X[X[:,1].argsort()]
#output:
array([[2, 1, 4],
       [1, 2, 3],
       [0, 5, 2]])
```
## Create a Series from a list, numpy array and dictionary
```python
mylist=list('abcdefghijklmnopqrstuvwxyz')
myarr=np.arange(26)
mydict=dict(zip(mylist,myarr))
ser1=pd.Series(mylist)
ser2=pd.Series(myarr)
ser3=pd.Series(mydict)
```
## not common to both series A and Series B
```python
ser1 = pd.Series([1,2,3,4,5])
ser2 = pd.Series([4,5,6,7,8])
ser_u = pd.Series((np.union1d(ser1,ser2))
ser_i = pd.Series((np.intersect1d(ser1,ser2))
ser_u[~ser_u.isin(ser_i)]
0    1
1    2
2    3
5    6
6    7
7    8
>>> print(ser_u)
0    1
1    2
2    3
3    4
4    5
5    6
6    7
7    8
>>> print(ser_i)
0    4
1    5
```
## How to reverese the rows of a data frame
```python
 df = pd.DataFrame(np.arange(25).reshape(5,-1))
  >>> df
    0   1   2   3   4
0   0   1   2   3   4
1   5   6   7   8   9
2  10  11  12  13  14
3  15  16  17  18  19
4  20  21  22  23  24
 
 df.iloc[::-1,:]
   0   1   2   3   4
4  20  21  22  23  24
3  15  16  17  18  19
2  10  11  12  13  14
1   5   6   7   8   9
0   0   1   2   3   4

 ```
 
 ## Spark (Mlib) Kmeans
 
 ```python
from __future__ import print_function
from numpy import array
from math import sqrt
from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans, KMeansModel

if __name__ == "__main__":
    sc = SparkContext(appName="KMeansExample")  # SparkContext

    # Load and parse the data (create RDD)
    data = sc.textFile("E:\Hadoop\spark-3.0.0-preview2-bin-hadoop2.7\data\mllib\kmeans_data.txt")
    parsedData = data.map(lambda line: array([float(x) for x in line.split(' ')]))

    # Build the model (cluster the data) (Create RDD Pipeline)
    clusters = KMeans.train(parsedData, 2, maxIterations=10, initializationMode="random")

    # Evaluate clustering by computing Within Set Sum of Squared Errors
    def error(point):
        center = clusters.centers[clusters.predict(point)]
        return sqrt(sum([x**2 for x in (point - center)]))

    WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    print("Within Set Sum of Squared Error = " + str(WSSSE))

    # Save and load model
    clusters.save(sc, "target/org/apache/spark/PythonKMeansExample/KMeansModel")
    sameModel = KMeansModel.load(sc, "target/org/apache/spark/PythonKMeansExample/KMeansModel")

    sc.stop()
    # Output 
    Within Set Sum of Squared Error = 0.6928203230275529
 ```
 ## Spark (Linear Regression)
```python
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.mllib.linalg import DenseVector
from pyspark import SparkContext

if __name__ == "__main__":
    sc = SparkContext(appName="Regression Metrics Example")

    # Load and parse the data
    def parsePoint(line):
        values = line.split()
        return LabeledPoint(float(values[0]),
                            DenseVector([float(x.split(':')[1]) for x in values[1:]]))

    #(create RDD)
    data = sc.textFile("E:\Hadoop\spark-3.0.0-preview2-bin-hadoop2.7\data\mllib\sample_linear_regression_data.txt")
    #(create RDD Pipeline)
    parsedData = data.map(parsePoint)

    # Build the model
    model = LinearRegressionWithSGD.train(parsedData)

    # Get predictions
    valuesAndPreds = parsedData.map(lambda p: (float(model.predict(p.features)), p.label))

    # Instantiate metrics object
    metrics = RegressionMetrics(valuesAndPreds)

    # Squared Error
    print("MSE = %s" % metrics.meanSquaredError)
    print("RMSE = %s" % metrics.rootMeanSquaredError)

    # R-squared
    print("R-squared = %s" % metrics.r2)

    # Mean absolute error
    print("MAE = %s" % metrics.meanAbsoluteError)

    # Explained variance
    print("Explained variance = %s" % metrics.explainedVariance)
    
    # Output
MSE = 103.30968681818085
RMSE = 10.164137288436281
R-squared = 0.027639110967836777
MAE = 8.148691907953307
Explained variance = 2.888395201717894
```
## Open Zipfile
```python
from unrar import rarfile
rar = rarfile.RarFile('H:\Farafan\data\March\Cell.rar')
rar.namelist()

def Open_rar():
        import os, zipfile, pyunpack
        basis_folder =  r'H:\Farafan\data\March'

        for root, dirs, files in os.walk(basis_folder):
            for filename in files:
                if filename.endswith(".rar") :
                    print('RAR:'+os.path.join(root,filename))
                elif filename.endswith(".zip"):
                    print('ZIP:'+os.path.join(root,filename))
                name = os.path.splitext(os.path.basename(filename))[0]
                if filename.endswith(".rar") or filename.endswith(".zip"):
                    try:
                        arch = pyunpack.Archive(os.path.join(root,filename))
                        # os.mkdir(name)
                        arch.extractall(directory=root)
                        os.remove(os.path.join(root,filename))
                    except Exception as e:
                        print("ERROR: BAD ARCHIVE "+os.path.join(root,filename))
                        print(e)
                        try:
                            # os.path.join(root,filename)os.remove(filename)
                            pass
                        except OSError as e: # this would be "except OSError, e:" before Python 2.6
                            if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
                                raise # re-raise exception if a different error occured
 ```

### magine a machine learning problem where you have 20 classes and about 7000 sparse boolean features. I want to figure out what the 20 most unique features per class are. In other words, features that are used a lot in a specific class but aren't used in other classes, or hardly used. What would be a good feature selection algorithm or heuristic that can do this?

When you train a Logistic Regression multi-class classifier the train model is a num_class x num_feature matrix which is called the model where its [i,j] value is the weight of feature j in class i. The indices of features are the same as your input feature matrix.

In scikit-learn you can access to the parameters of the model If you use scikit-learn classification algorithms you'll be able to find the most important features per class by:
```python
clf = SGDClassifier(loss='log', alpha=regul, penalty='l1', l1_ratio=0.9, learning_rate='optimal', n_iter=10, shuffle=False, n_jobs=3, fit_intercept=True)
clf.fit(X_train, Y_train)
for i in range(0, clf.coef_.shape[0]):
    top20_indices = np.argsort(clf.coef_[i])[-20:]
    print top20_indices
 ```
 clf.coef_ is the matrix containing the weight of each feature in each class so clf.coef_[0][2] is the weight of the third feature in the first class. If when you build your feature matrix you keep track of the index of each feature in a dictionary where dic[id] = feature_name you'll be able to retrieve the name of the top feature using that dictionary.

## OOP (Object-Oriented Programming)
Python, like every other object-oriented language, allows you to define classes to create objects.
What is Encapsulation?
#### Encapsulation
is the process of preventing clients from accessing certain properties, which can only be accessed through specific methods.
Let's introduce a private attribute called __discount in the Book class.

 ```python
class Book:
    def __init__(self, title, quantity, author, price):
        self.title = title
        self.quantity = quantity
        self.author = author
        self.__price = price
        self.__discount = None

    def set_discount(self, discount):
        self.__discount = discount

    def get_price(self):
        if self.__discount:
            return self.__price * (1-self.__discount)
        return self.__price

    def __repr__(self):
        return f"Book: {self.title}, Quantity: {self.quantity}, Author: {self.author}, Price: {self.get_price()}"
```
#### Inheritance?
The subclass or child class is the class that inherits. The superclass or parent class is the class from which methods and/or attributes are inherited.

```python
class Book:
    def __init__(self, title, quantity, author, price):
        self.title = title
        self.quantity = quantity
        self.author = author
        self.__price = price
        self.__discount = None

    def set_discount(self, discount):
        self.__discount = discount

    def get_price(self):
        if self.__discount:
            return self.__price * (1-self.__discount)
        return self.__price

    def __repr__(self):
        return f"Book: {self.title}, Quantity: {self.quantity}, Author: {self.author}, Price: {self.get_price()}"


class Novel(Book):
    def __init__(self, title, quantity, author, price, pages):
        super().__init__(title, quantity, author, price)
        self.pages = pages


class Academic(Book):
    def __init__(self, title, quantity, author, price, branch):
        super().__init__(title, quantity, author, price)
        self.branch = branch
``` 
### Polymorphism?
a subclass can use a method from its superclass as is or modify it as needed.
```python
class Academic(Book):
    def __init__(self, title, quantity, author, price, branch):
        super().__init__(title, quantity, author, price)
        self.branch = branch

    def __repr__(self):
        return f"Book: {self.title}, Branch: {self.branch}, Quantity: {self.quantity}, Author: {self.author}, Price: {self.get_price()}"
```
### map, filter
- map(lambda x: x**2, items)
- filter(lambda x: x < 0, number_list)
- the risult would be something like: <map at 0x0939849348>
- to solve it we should use list like: list(map(lambda x: x**2, items))


### Difference Between Dictionary and JSON
| Dictionary         | JSON     | 
|--------------|-----------|
|Keys can be any hashable object| Keys can be only strings     |
| Keys cannot be repeated      |Keys can be ordered and repeated |
|No such default value is set| Keys has a default value of undefined    |
| Values can be accessed by subscript   |Values can be accessed by using “.”(dot) or “[]”|
|Can use a single or double quote for the string object| The double quotation is necessary for the string object     |
| Returns ‘dict’ object type    |	Return ‘string’ object type |


### Difference between git pull and git fetch
| Git Fetch        | Git Pull    | 
|--------------|-----------|
|Gives the information of a new change from a remote repository without merging into the current branch| Keys can be only strings|Brings the copy of all the changes from a remote repository and merges them into the current branch |

|Repository data is updated in the .git directory| The local repository is updated directly   |
| Review of commits and changes can be done   |Updates the changes to the local repository immediately.|
|No possibility of merge conflicts.|Merge conflicts are possible if the remote and the local repositories have done changes at the same place.    |

### Generator  Yield

```python
def iter_leafs(d):
    for key, val in d.items():
        if isinstance(val, dict):
            yield from iter_leafs(val)
        else:
            yield val

d = {'a':{'a':{'y':2}},'b':{'c':{'a': 5}},'x':{'a':{'m':6, 'l': 9}}}
list(iter_leafs(d))
output: [2, 5, 6, 9]
#-------------------------
def iter_leafs(d, keys=[]):
    for key, val in d.items():
        if isinstance(val, dict):
            yield from iter_leafs(val, keys+ [key] )
        else:
            yield keys + [key], val
l1 =list(iter_leafs(d))
print(l1)
output: [(['a', 'a', 'y'], 2), (['b', 'c', 'a'], 5), (['x', 'a', 'm'], 6), (['x', 'a', 'l'], 9)]
```
## Save and Load JSON and PKL
```python
import pickle
with open('output.json', 'w+') as f:
    json.dump(lista_items, f)
    
with open('input.json') as f:
    df = json.load(f)   
    
with open('input.json', 'rb') as f:
    PKL = pickle.load(f)
    
with open('output.json', 'wb') as f:
    pickle.dump(saved_file, f)
```
### Add a column based on other columns in a data frame
```python
df[['_date', 'cell_n']] = df['Date_Cell'].astype(str).str.split('_', expand = True)
```
