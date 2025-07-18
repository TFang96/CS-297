import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

real_df = pd.read_csv('users.csv')
fake_df = pd.read_csv('fusers.csv')

real_df['label'] = 0
fake_df['label'] = 1

df = pd.concat([real_df, fake_df], ignore_index=True)

#print first few columns
print(df.head())

#print shape
print(df.shape)

#data types
print(df.dtypes)

#summary statistics
print(df.describe())

#Histogram of friend count
sns.histplot(data=df, x='friends_count', hue='label', kde=True)
plt.title("Friend Count Distribution")
plt.show()

#Correlation heatmap
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()