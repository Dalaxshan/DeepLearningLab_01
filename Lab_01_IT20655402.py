#1
import numpy as np

random_array = np.random.rand(4,4)
print(random_array)

#2
import numpy as np
import matplotlib.pyplot as plt

random_array_exp = np.random.exponential(scale=1.0, size=100000)
random_array_uniform = np.random.rand(100000)
random_array_normal = np.random.normal(loc=0.0, scale=1.0, size=100000)


plt.hist(random_array_exp, density=True, bins=50, histtype="step", color="blue", label="Exponential")
plt.hist(random_array_uniform, density=True, bins=50, histtype="step", color="green", label="Uniform")
plt.hist(random_array_normal, density=True, bins=50, histtype="step", color="red", label="Normal")

plt.legend(loc="upper right")
plt.title("Random Distributions")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()

#3
np.random.exponential(1000,2)

#4
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


X = np.linspace(-5, 5,100)
Y = np.linspace(-5, 5,100)
X, Y = np.meshgrid(X, Y)


Z = X**2 + Y**2


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Z = X**2 + Y**2')

ax.view_init(elev=30, azim=45)
plt.show()

#5
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
!wget -q https://elitedatascience.com/wp-content/uploads/2022/07/Pokemon.csv
df = pd.read_csv('Pokemon.csv', index_col=0, encoding='latin')
df.head()
sns.lmplot(x='Attack', y='Defense', data=df,
           fit_reg=True, # No regression line
           hue='Stage')   # Color by evolution stage
# Boxplot
plt.figure(figsize=(9,6)) # Set plot dimensions
sns.boxplot(data=df)
# Preprocess DataFrame
stats_df = df.drop(['Total', 'Stage', 'Legendary'], axis=1)

# New boxplot using stats_df
plt.figure(figsize=(9,6)) # Set plot dimensions
sns.boxplot(data=stats_df)
# Violin plot
plt.figure(figsize=(12,8)) # Set plot dimensions
sns.violinplot(x='Type 1', y='Attack', data=df)

# Calculate correlations
corr = stats_df.corr()

corr
# Heatmap
plt.figure(figsize=(9,8))
sns.heatmap(corr)
