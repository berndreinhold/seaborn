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



def paired_corr_plot(df : pd.DataFrame, category_col : str = None, corr_columns : list = None, **kwargs):
    """
    plot paired correlation matrices of a dataframe in a heatmap
    paired means one dataframe selected for one value of the category_col is shown in the upper triangle and the other in the lower triangle
    different cases are distinguished by the number of distinct values of the category_col:
    Right now, only 2 or 3 categories are supported, Otherwise the correlation matrix of the whole dataframe is plotted
    in case of 2 category values, there is one paired correlation matrix, in case of 3 category values there are two (subplots = category values - 1)

    Parameters:
    -----------
    df : pandas.DataFrame
        dataframe to plot and that contains also the category column
        Right now only support for one category column is implemented.
        Support for more than one is conceivable though.

    category_col : str
        name of the column that contains the category values to distinguish

    corr_columns : list
        list of column names to calculate the correlation matrix from

    **kwargs : dict
        keyword arguments passed to the seaborn.heatmap function
    """

    # determine the distinct values of category_col:
    if category_col is None:
        categories = []
    else:
        categories = df[category_col].unique().tolist()
    
    
    if len(categories)<1 or len(categories)==1 or len(categories)>3:
        fig, ax = plt.subplots(figsize=(9.5,6))

        if len(categories)<1 or category_col is None:
            print(f"Category column ({category_col}) has no values!")
        elif len(categories)==1:
            print(f"Category column ({category_col}) has exactly one value: {categories[0]}!")
        elif len(categories)>3:
            print(f"Too many categories ({categories}) to plot, expected 2 or 3!")
        print("just plot the correlation matrix of the whole dataframe")

        mat = df.corr()
        if corr_columns is not None:
            mat = mat[corr_columns]
        ax = sns.heatmap(mat, **kwargs)
        ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=18)
        ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=18)
        ax.tick_params(axis='x', rotation=90)
        return fig, ax
        
    elif len(categories)==2:
        fig, ax = plt.subplots(figsize=(9.5,6))
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

        ax = sns.heatmap(df_combined, **kwargs)
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

    elif len(categories)==3:
        fig, axes = plt.subplots(1, 2, sharey=True, figsize=(14,6))
        # split the dataframe into dataframes for each category
        # and compute the correlation matrix for each
        df_cat, mat = [], []
        if corr_columns is None:
            corr_columns = df.columns
        
        for category in categories:
            df_cat.append(df.loc[df[category_col]==category, corr_columns])
            mat.append(df_cat[len(df_cat)-1].corr())

        # combine the correlation matrices pairwise (0,1), (0,2):
        mat_combined, df_combined = [], []
        mat_combined.append(np.triu(mat[0].to_numpy(), k=1) + np.tril(mat[1].to_numpy()))
        mat_combined.append(np.triu(mat[0].to_numpy(), k=1) + np.tril(mat[2].to_numpy()))

        index_tuples = (0,1), (0,2)
        for i in range(len(categories)-1):
            df_combined.append(pd.DataFrame(mat_combined[i], columns=mat[0].columns, index=mat[0].columns))
            axes[i] = sns.heatmap(df_combined[i], ax=axes[i], **kwargs)

        for i,ax in enumerate(axes):
            ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=18)
            ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=18)
            ax.tick_params(axis='x', rotation=90)

            # get padding around figure
            h, w = ax.bbox.height, ax.bbox.width
            # draw line
            #plt.axline([-0.1, -0.1], [1.1, 1.1], linewidth=2, color='r', clip_on=False)
            ax.text(0, 1-0.01, "----------", horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, fontsize=18, rotation = -np.arctan(h/w)*180/pi)
            ax.text(1, 0+0.01, "----------", horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=18, rotation = -np.arctan(h/w)*180/pi)

            # text top right
            ax.text(1, 1, categories[index_tuples[i][0]], horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, fontsize=18, fontstyle='italic', rotation = np.arctan(h/w)*180/pi)
            # text bottom left
            ax.text(0, 0, categories[index_tuples[i][1]], horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=18, fontstyle='italic', rotation = np.arctan(h/w)*180/pi)
            

        return fig, axes


df = sns.load_dataset("penguins")
df = df.dropna(axis=0)

category_col = "species"  # three species: two paired correlation plots
#category_col = "sex"  # two categories: Female, Male: one paired correlation plot
fig, ax = paired_corr_plot(df, category_col=category_col, annot=True, fmt=".2f", cmap="YlGnBu")
plt.show()

#sns.pairplot(df, hue="species", vars=columns_, diag_kind="kde", kind="reg", plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.3}})
#plt.show()
