"""
Scatterplot heatmap
-------------------

_thumb: .5, .5

"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from math import pi
sns.set_theme(style="whitegrid")



def corr_plot(df : pd.DataFrame, category_col : str, corr_columns=None):
    fig, ax = plt.subplots()
    # determine the distinct values of category_col:
    categories = df[category_col].unique().tolist()
    if len(categories)<1 or len(categories)==1 or len(categories)>2:
        if len(categories)<1:
            print(f"Category column {category_col} has no values!")
        elif len(categories)==1:
            print(f"Category column {category_col} has exactly one value: {categories[0]}!")
        elif len(categories)>2:
            print(f"Too many categories {categories} to plot, expected 2!")
        print("just plot the correlation matrix of the whole dataframe")

        mat = df.corr()
        if corr_columns is not None:
            mat = mat[corr_columns]
        ax = sns.heatmap(mat, annot=True, fmt=".2f", cmap="YlGnBu", vmin=-1.0, vmax=1.0)
        ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=18)
        ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=18)
        ax.tick_params(axis='x', rotation=90)
        return fig, ax
        
    elif len(categories)==2:
        # split the dataframe into dataframes for each category
        # and compute the correlation matrix for each
        df_cat, mat = [], []
        if corr_columns is None:
            corr_columns = df.columns
        
        for category in categories:
            df_cat.append(df.loc[df[category_col]==category, corr_columns])
            mat.append(df_cat[len(df_cat)-1].corr())

        # combine the correlation matrices into one
        mat_combined = np.triu(mat[0].to_numpy(), k=1) + np.tril(mat[1].to_numpy())
        df_combined = pd.DataFrame(mat_combined, columns=mat[0].columns, index=mat[0].columns)

        ax = sns.heatmap(df_combined, annot=True, fmt=".2f", cmap="YlGnBu", vmin=-1.0, vmax=1.0)
        ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=18)
        ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=18)
        ax.tick_params(axis='x', rotation=90)

        # get padding around figure
        h, w = ax.bbox.height, ax.bbox.width
        # draw line
        #plt.axline([-0.1, -0.1], [1.1, 1.1], linewidth=2, color='r', clip_on=False)
        ax.text(0, 1-0.01, "----------", horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, fontsize=18, rotation = -np.arctan(h/w)*180/pi)
        ax.text(1, 0+0.01, "----------", horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=18, rotation = -np.arctan(h/w)*180/pi)

        ax.text(0, 0, categories[0], horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=18, fontstyle='italic', rotation = np.arctan(h/w)*180/pi)
        ax.text(1, 1, categories[1], horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, fontsize=18, fontstyle='italic', rotation = np.arctan(h/w)*180/pi)

        return fig, ax


# Load the brain networks dataset, select subset, and collapse the multi-index
df = sns.load_dataset("penguins")
df = df.dropna(axis=0)

#columns_ = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
category_col = "sex"
#fig, ax = corr_plot(df, category_col, columns_)
fig, ax = corr_plot(df, category_col)
plt.show()

#sns.pairplot(df, hue="species", vars=columns_, diag_kind="kde", kind="reg", plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.3}})
#plt.show()
