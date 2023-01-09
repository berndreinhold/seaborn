"""
paired correlation plot
-------------------

_thumb: .5, .5

"""
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("penguins")
df = df.dropna(axis=0)

category_col = "species"  # three species: two paired correlation plots
fig, ax = sns.paired_corr_plot(df, category_col=category_col, annot=True, fmt=".2f", cmap="YlGnBu")
#plt.show()

category_col = "sex"  # two categories: Female, Male: one paired correlation plot
fig, ax = sns.paired_corr_plot(df, category_col=category_col, annot=True, fmt=".2f", cmap="YlGnBu")
