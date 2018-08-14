

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

## Inferential Statistics to [Supervised] Machine Learning
As we've seen, we can use sampling techniques and descriptive statistics to learn more about a population. While the overall population itself will undoubtedly differ to some degree from that of our sample, we can quantify the likelihood and scale of the population's differences from the sample across various dimensions or statistical measures. For example, if modelling a dataset that is approximately a normal distribution, we would start by computing the mean and variance for our sample and we could then calculate confidence intervals for those respective measures of the overall population. 

Supervised machine learning applies these same concepts along with additional algorithms in order to mine structure within the data to make predictive models. This always begins with splitting the data into train and test sets so that we can validate our model performance. This process is analagous to if we took multiple samples from a population; assuming our samples are independent and of a sufficient size, we should expect that descriptive measures such as the mean and standard deviation of those samples should be roughly equivalent. Similarly in machine learning, we will train our algorithm to detect and model patterns in the training set. This is typically a random sample of roughly 75% - 80% of the total data available to us. After training a model on this set of data, we can then further test the validity of our model against the remaining hold-out data which (again typically 20-25% of the original data) we intentionally did not train the model on. As you probably have put together, this second hold-out dataset of those observations that we not included in the training is known as the test set.

Implementing a **train-test split** in python is very straightforward using sklearn's built in method. Let's take a look at this in more detail. We start by importing a dataset and choosing X and y values. This is a standard process for all **supervised machine learning** algorithms. A supervised learning algorithm is one in which we feed input examples (X, via the training set) into a model which then attempts to reproduce appropriate output values (Y) associated with those inputs. This can take many forms including regression problems such as, "if I give you a person's height, age, weight, blood pressure, etc. cholestoral level",  to classification problems such as "if I give you details about a plant including color, stem length, and root structure, predict what species it is" or even text processing such as "if I give you a reviewers comments, predict how positive/negative their viewpoint is". All of these problems can initially be formulated as an input output mapping where we are trying to generalize a formula from one space X, to another space y.


```python
#As usual we begin by importing our dataset
import pandas as pd

df = pd.read_csv('health_data.txt', delimiter='\t')
print('Length of Dataset: ', len(df))
print('Column Names:\n', df.columns)
df.head()
```

    Length of Dataset:  20
    Column Names:
     Index(['Pt', 'BP', 'Age', 'Weight', 'BSA', 'Dur', 'Pulse', 'Stress'], dtype='object')





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pt</th>
      <th>BP</th>
      <th>Age</th>
      <th>Weight</th>
      <th>BSA</th>
      <th>Dur</th>
      <th>Pulse</th>
      <th>Stress</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>105</td>
      <td>47</td>
      <td>85.4</td>
      <td>1.75</td>
      <td>5.1</td>
      <td>63</td>
      <td>33</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>115</td>
      <td>49</td>
      <td>94.2</td>
      <td>2.10</td>
      <td>3.8</td>
      <td>70</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>116</td>
      <td>49</td>
      <td>95.3</td>
      <td>1.98</td>
      <td>8.2</td>
      <td>72</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>117</td>
      <td>50</td>
      <td>94.7</td>
      <td>2.01</td>
      <td>5.8</td>
      <td>73</td>
      <td>99</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>112</td>
      <td>51</td>
      <td>89.4</td>
      <td>1.89</td>
      <td>7.0</td>
      <td>72</td>
      <td>95</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Define X and y
X = df[['Pt', 'Age', 'Weight', 'BSA', 'Dur', 'Pulse', 'Stress']]
y = df['BP']
```

### Train Test Split


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```


```python
#A brief preview of our train test split
print(len(X_train), len(X_test), len(y_train), len(y_test))
```

    13 7 13 7


# Machine Learning - Regression
From here, we will apply models in order to predict a given output (y) and compare our error results in the training set to that of the test set. This will help us gauge how generalizable our model is to new data observations.

#### Importing and initializing the model class


```python
from sklearn.linear_model import LinearRegression

#Initialize a regression object
linreg = LinearRegression()
```

#### Fitting the model to the train data


```python
linreg.fit(X_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)



#### Calculating predictions (y_hat)


```python
y_hat_train = linreg.predict(X_train)
y_hat_test = linreg.predict(X_test)
```

#### Calculating Residuals


```python
train_residuals = y_hat_train - y_train
test_residuals = y_hat_test - y_test
```

#### Calculating Mean Squared Error
A good way to compare overall performance is to compare the mean squarred error for the predicted values on the train and test sets.


```python
import numpy as np
```


```python
train_mse = np.mean([x**2 for x in train_residuals])
test_mse = np.mean([x**2 for x in test_residuals])
print('Train Mean Squarred Error:', train_mse)
print('Test Mean Squarred Error:', test_mse)
```

    Train Mean Squarred Error: 0.07603912037773927
    Test Mean Squarred Error: 0.45440881090074065


Notice here that our test error is substantially worse then our train error demonstrating that our model is overfit and may not generalize well to future cases.
