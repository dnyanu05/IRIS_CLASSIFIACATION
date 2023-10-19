import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
data=pd.read_csv('Iris.csv')
data.head()
data.head(20)
data.tail()
data.describe()
data.size
data.shape
data.dtypes
data.columns
data.groupby('PetalWidthCm').size()
data['PetalWidthCm'].unique().tolist()
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('iris.csv')
sns.boxplot(x='SepalWidthCm' , data=df)
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('iris.csv')
sns.boxplot(x='PetalWidthCm' , data=df)
import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(x='PetalLengthCm',  y='PetalWidthCm' , hue='Species' , data=df,)
plt.legend(bbox_to_anchor=(1,1), loc=2)
plt.show()
data.hist()
sns.pairplot(data, hue='Species')
import seaborn as sns
import matplotlib.pyplot as plt
def graph(y):
    sns.boxplot(x='Species', y=y , data=df)
plt.figure(figsize=(10,10))
plt.subplot(221)
graph('SepalLengthCm')
plt.subplot(222)
graph('SepalWidthCm')
plt.subplot(223)
graph('PetalLengthCm')
plt.subplot(224)
graph('PetalWidthCm')
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(df.drop(['Id'], axis =1),hue='Species',height=2)
import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(x='PetalLengthCm',  y='PetalWidthCm' , hue='Species' , data=df,)
plt.legend(bbox_to_anchor=(1,1), loc=2)
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(x='SepalLengthCm',  y='SepalWidthCm' , hue='Species' , data=df,)
plt.legend(bbox_to_anchor=(1,1), loc=2)
plt.show()
