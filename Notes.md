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

#### Python

How to handle missing data in the dataset.

1.  **Removing rows** which includes missing data in one or more columns. However, this is not very useful, if we only have a small number of rows/data or if we there are many columns with missing data
2.  Filling missing data with the **mean, minimum, maximum ...** of other rows.

This can be done by the function `Imputer` from `sklearn.preprocessing`.

```Python
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X.iloc[:, 1:3]) # There are only missing data in columns 2 and 3[1:3]
X.iloc[:, 1:3] = imputer.transform(X.iloc[:, 1:3])
```

#### R

Here we are using the function `ifelse` to decide wether there are missing data (_na_) or not. If so, we are using the function `ave` calculate the **mean**.

```R
dataset$Age <- ifelse(is.na(dataset$Age), ave(dataset$Age, FUN = function(x) mean(x,
    na.rm = TRUE)), dataset$Age)
dataset$Salary <- ifelse(is.na(dataset$Salary), ave(dataset$Salary, FUN = function(x) mean(x,
    na.rm = TRUE)), dataset$Salary)
```

### Categorial Data

#### Python

Categorial data contains a fixed number of categories. Machine learning algorithms are based on mathematical equations and therefore can only handle _numbers_.  This means, if the categorial data are _text_, we have to encode the categories by replacing them with _numbers_.

In `sklearn.preprocessing` there are several functions for encoding, like `LabelEncoder` and `OneHotEncoder`.

-   `LabelEncoder`: This encoder replaces every category (_text_) by a _number_.
-   `OneHotEncoder`: This encoder adds dummy variables (columns) for every category.

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
