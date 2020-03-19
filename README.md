# My-Usful-Codes

```python
import itertools
[f[i][j] for i,j in itertools.product(range(t), range(int(n)))]

[x for x in range(20) if x%2 == 0]

# To replace two values
a=10
b=5
a,b = b,a
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
# Sort an array by the (n-1)th column
```python
X=np.array([[1,2,3],[0,5,2],[2,3,4]])
X[X[:,1].argsort()]
#output:
array([[2, 1, 4],
       [1, 2, 3],
       [0, 5, 2]])
```
# Create a Series from a list, numpy array and dictionary
```python
mylist=list('abcdefghijklmnopqrstuvwxyz')
myarr=np.arange(26)
mydict=dict(zip(mylist,myarr))
ser1=pd.Series(mylist)
ser2=pd.Series(myarr)
ser3=pd.Series(mydict)
```
# not common to both series A and Series B
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
# How to reverese the rows of a data frame
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
 
 # Spark (Mlib) Kmeans
 
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
 # Spark Linear Regression
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
