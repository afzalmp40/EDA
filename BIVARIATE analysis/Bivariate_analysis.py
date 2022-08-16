import matplotlib.pyplot as plt
import seaborn as sns

def correlation(df, figsize=(15,10), dpi=100):
    '''
    Plot correlation heatmap for a dataframe.
    Includes both pearson and spearman correlation.
    
    Parameters:
        df: a pandas dataframe
        figsize: default(15,10) set figure size
        dpi: default(100) set figure dpi
        
    Returns:
        None
        
    '''
    
    fig , ax= plt.subplots(1,2, figsize=figsize, dpi=dpi)

    pearson=df.corr()
    ax[0].set_title('pearson')
    sns.heatmap(pearson, cmap='YlGnBu', square=True, annot=True, ax=ax[0])

    spearman=df.corr(method='spearman')
    ax[1].set_title('spearman')
    sns.heatmap(spearman, cmap='YlGnBu', square=True, annot=True, ax=ax[1])

    plt.show()