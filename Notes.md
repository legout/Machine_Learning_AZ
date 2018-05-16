# Machine Learning A-Z™: Hands-On Python & R In Data Science‎

## Part 1 - Data Preprocessing

### Importing libraries

#### Python

Importing libraries in Python is done with the `import` statement. This loads the library into its own namespace or one can define a new namespace. It is possible to import the whole library or just parts of it.

```Python
import numpy # Importing the whole numpy library
import numpy as np # Importing numpy in the namespace np
from numpy import array, arange # Importing only array and arange from numpy
```

#### R

Many libraries are selected/imported by default in RStudio. If not, you have to import them using the `library` function.

```R
library(ggplot2) # Importing ggplot2
```

### Importing Datasets

#### Python

There is a very popular library called `pandas` for tabular data in Python. Several data formats can be read into a `pandas.Series` or `pandas.DataFrame`.

```Python
import pandas as pd # Importing pandas into the namespace pd

dataset = pd.read_csv('Data.csv') # Reading Data.csv
```

Depending on the used machine learning algorithm (supervised or unsupervised) the dataset have to be separated into the independent variables (_features_) and dependent variables (_labels_).

```Python
X = dataset.iloc[:, :-1] # All expect the last column
y = dataset.iloc[:, 3] # The last column
```

#### R

Reading _cvs-Files_ can be done with the function `read.csv`.

```R
dataset <- read.csv("Data.csv")
```

### Missing Data

How to handle missing data in the dataset.

1.  **Removing rows** which includes missing data in one or more columns. However, this is not very useful, if we only have a small number of rows/data or if we there are many columns with missing data
2.  Filling missing data with the **mean, minimum, maximum ...** of other rows.

#### Python

This can be done by the function `Imputer` from `sklearn.preprocessing`.

```Python
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X.iloc[:, 1:3]) # There are only missing data in columns 2 and 3[1:3]
X.iloc[:, 1:3] = imputer.transform(X.iloc[:, 1:3])
```

#### R

Here we are using the function `ifelse` to decide wether there are missing data (_na_) or not. If so, we are using the function `ave` to calculate the **mean**.

```R
dataset$Age <- ifelse(is.na(dataset$Age), ave(dataset$Age, FUN = function(x) mean(x,
    na.rm = TRUE)), dataset$Age)
dataset$Salary <- ifelse(is.na(dataset$Salary), ave(dataset$Salary, FUN = function(x) mean(x,
    na.rm = TRUE)), dataset$Salary)
```

### Categorial Data

Categorial data contains a fixed number of categories. Machine learning algorithms are based on mathematical equations and therefore can only handle _numbers_.  This means, if the categorial data are _text_, we have to encode the categories by replacing them with _numbers_.

#### Python

In `sklearn.preprocessing` there are several functions for encoding, like `LabelEncoder` and `OneHotEncoder`.

-   `LabelEncoder`: This encoder replaces every category (_text_) by a _number_.
-   `OneHotEncoder`: This encoder adds dummy variables (columns) for each category. In this columns the value of the row is _1_ for one category and _0_ for each other.

As `LabelEncoder` replaces _text_ with _numbers_, this encoding is not useful, if there is **no logical order** within the categories Instead use `OneHotEncoder` in addition.

_Since scikit-learn>0.20 there is a new Function called `CategoricalEncoder`. This encodes categorial data into a one-hot-encode or ordinal form_.

```Python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Encoding the Independent Variable

labelencoder_X = LabelEncoder()
X.iloc[:, 0] = labelencoder_X.fit_transform(X.iloc[:, 0])

## No logical order within the categories? Use OneHotEncoder in addition
onehotencoder = OneHotEncoder(categorical_features=[0], sparse=False)
X = onehotencoder.fit_transform(X)

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
```

#### R

With the function `factor` we can encode our _text_ categories into _numbers_.

```R
# Encoding categorical data
dataset$Country <- factor(dataset$Country, levels = c("France", "Spain", "Germany"),
    labels = c(1, 2, 3))
dataset$Purchased <- factor(dataset$Purchased, levels = c("No", "Yes"), labels = c(0,
    1))
```

### Splitting the Dataset

It is useful to split the dataset into a _training set_ and a _test set_, to proof if the machine learning model is stable.

If the model performance on the _training set_ is much better than the performance on the _test set_, the model did not generalize well. It rather learned the correlation between the features and the labels including the noise. This is typically called **Overfitting**.

#### Python

There is the function `train_test_split` from `sklearn.model_selection` fot splitting datasets.

```Python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

#### R

The library `caTools`includes the function `sample.split` for splitting the dataset.

```R
library(caTools)
set.seed(123) # seed to get repeatable results
split <- sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)
```

### Feature Scaling

Many machine learning models are based on the _Euclidean distance d_. The _Euclidean distance_ of point 1 with coordinates $(x1, y1)$ and point 2 with coordinates $(x2, y2)$ is:

$$d = \sqrt{(x2-x1)^2+(y2-y1)^2}$$

This means, if the range of the data is not in the same scale, _d_ would be dominated by the data with the largest scale. Therefore the _features_ should be scaled before training a machine learning model.

There are several ways for scaling the data. The two most common are:

-   Standardization: For every observation of a feature the mean of the feature is withdrawn, divided by the standard deviation

$$x_{stand} = \frac{x-\bar{x}}{\sigma(x)}$$

-   Normalization: The minimum of the feature is substraced from the observation, divided by the difference of the maximum and the minimum of the feature.

$$x_{norm} = \frac{x-x_{min}}{x_{max}-x_{min}}$$

#### Python

In `sklearn.preprocessing` there are many functions for feature scaling, e.g. `StandardScaler` and `Normalizer`

```Python
from sklearn.preprocessing import StandardScaler, Normalizer

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) # The scaler is fitted to the training set followed by transforming the training set
X_test = sc_X.transform(X_test) # The testset is transformed with the scaler fitted  to the training set
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1, 1)) # The labels are fitted and transformed using an other scaler
```

#### R

Use `scale` for feature scaling in R.

```R
# Scaling can only be done on numeric data. Country and Purchased are `factors` and therefore not numeric in R
training_set[, 2:3] <- scale(training_set[, 2:3])
test_set[, 2:3] <- scale(test_set[, 2:3])
```

## Part 2 - Regression

### Section 1 - Simple Linear Regression

### Section 2 - Multiple Linear Regression

### Section 3 - Polynomial Regression

### Section 4 - Support Vector Regression (SVR)

### Section 5 - Decission Tree Regression

### Section 6 - Random Forest Regression

### Section 7 - Evatluating Regression Models Performance

### Section 8 - Regularization Methods

### Section 9 - Part Recap
