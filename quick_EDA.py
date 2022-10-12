'''   
Module made for quick EDA.
-Detect and handle outliers using methods like Z score and IQR.
-Outlier handling methods include removing and compressing. 
-Plot correlation heatmap using "correlation" function.

Available functions:

    UNIVARIATE ANALYSIS:
    --------------------
    five_point_summary: Prints five point summary of a feature.
    outliers_z_score: Analyse outliers using Z score.
    outliers_IQR: Analyse outliers using IQR.
    analysis_quant: Analyse quantative features.
    analysis_cate: Analyse categorical features.
    handle_outliers: Handle outliers.
    ______________________________________________________________
    

    BIVARIATE ANALYSIS:
    -------------------
    correlation: Plot correlation heatmap for a dataframe.
    multiplot: Plot multiple plots like correlation heatmap,
               pairwise scatterplot and histogram in single plot.
    mutual_info: Plot mutual info of features with respect to the target.
    ______________________________________________________________
'''
    
    

#############################################################################################
'''                                  UNIVARIATE ANALYSIS                                  '''
#############################################################################################


def five_point_summary(df, columns='all_the_columns'):
    '''
    Prints five point summary of a feature.
    
    Parameters :
    ------------
        df : 
            a pandas dataframe
        
        columns : default('all_the_columns') 
            list of column names.
            (if list of columns is not passed then
            all columns are analysed)
            
    ________________________________
    Returns : 
    ---------
        None 
    '''

    # converting singular value of str to list 
    if type(columns)==str:
        # if list of columns is not passed then all columns are analysed
        if columns=='all_the_columns':
            columns=df.columns
        # else the passed string is converted to a list
        else:
            columns=[columns]

    for column in columns:
        print('5 point summary for:', column)

        # extracting and printing the five point summary from describe function
        print(df[[column]].describe().iloc[3:] )
        print('---------------------------------')
    
    
    locals().clear()
    from gc import collect
    collect()
    del(collect)   
    
#############################################################################################



def outliers_z_score(df, columns='all_the_columns', mode='print'):
    '''
    Analyse outliers using Z score.
    
    Parameters :
    ------------
        df : 
            a pandas dataframe
        
        columns : default('all_the_columns') 
            list of column names.
            (if list of columns is not passed then
            all columns are analysed)
                 
        mode : {'print': 'only prints outliers',
               'return': 'returns outliers dataframe' 
              }
              
    ________________________________
    Returns : 
    ---------
        mode='return' ==> ('upper', 'lower', 'outliers_with_z')
        
        mode='print' ==> None
    '''
    
    from numpy import mean as np_mean ,std as np_std
    from pandas import DataFrame as pd_DataFrame

    # converting singular value of str to list 
    if type(columns)==str:
        # if list of columns is not passed then all columns are analysed
        if columns=='all_the_columns':
            columns=df.columns
        # else the passed string is converted to a list
        else:
            columns=[columns]

    for column in columns:

        ###CALCULATIONS###

        # storing the feature as a series 
        feature=df[column]

        # calculate mean and stdev
        mean = np_mean(feature)
        stdev = np_std(feature)

        # calculate outlier limits via Z score
        lower= -3*stdev + mean
        upper=  3*stdev + mean

        # calculating Z score for features
        Z=(feature-mean)/stdev

        # creating a mask to subset only outlier values( abs(z) > 3 )
        mask=abs( Z )>3

        # a dataframe storing the outliers and their z scores
        outliers_with_z=pd_DataFrame( {
                                'outliers' : feature[mask],
                                'Z-score'  : Z[mask] 
        }).sort_values(by='outliers')

        if mode=='return':
            return upper, lower, outliers_with_z

        else:
            ###PRINTING THE RESULTS###
            print( 'OUTLIERS in ' + column + ' via Z score\n' )
            print('Outlier limits:\nlower limit:', lower, '\nupper limit:', upper)
            print()
            print('Total outliers:', outliers_with_z.shape[0] )

            if outliers_with_z.shape[0]!=0:

                if len(outliers_with_z)>10:
                    print(outliers_with_z[:5],'\n.\n.')
                    print(outliers_with_z[-5:])
                else:
                    print(outliers_with_z)

            print('---------------------------------')
            
            
    locals().clear()
    from gc import collect
    collect()
    del(collect)   
    
#############################################################################################



def outliers_IQR(df, columns='all_the_columns', mode='print'):
    '''
    Analyse outliers using IQR.
    
    Parameters :
    ------------
        df : a pandas dataframe
        
        columns : default('all_the_columns') list of column names.
                 (if list of columns is not passed then
                 all columns are analysed)
                 
        mode : {'print': 'only prints outliers',
               'return': 'returns outliers dataframe' 
              }
              
    ________________________________
    Returns : 
    ---------
        mode='return' ==> ('upper', 'lower', 'outliers_with_IQR') 
        mode='print' ==> None 
    ''' 
    
    from pandas import DataFrame as pd_DataFrame


    # converting singular value of str to list 
    if type(columns)==str:
        # if list of columns is not passed then all columns are analysed
        if columns=='all_the_columns':
            columns=df.columns
        # else the passed string is converted to a list
        else:
            columns=[columns]

    for column in columns:

        ###CALCULATIONS###
        # storing the feature as a series 
        feature=df[column]

        # extracting quartile1, quartile3 from df.describe
        q1,q3=feature.describe().iloc[[4,6]]

        # calculating iqr
        iqr=q3-q1

        # calculate outlier limits using iqr and tukey value of 1.5
        upper= q3 + 1.5*iqr
        lower= q1 - 1.5*iqr

        #creating a mask for filtering
        mask= (feature<lower) | (feature>upper)

        # filter and store feature using outlier limits
        outliers_with_IQR= feature[mask].sort_values()
        outliers_with_IQR.columns='outliers'

        if mode=='return':
            return upper, lower, outliers_with_IQR
        else:
            ###PRINTING THE RESULTS###
            print( 'OUTLIERS in '+ column +' via IQR\n' )
            print('Outlier limits:\nlower limit:',lower,'\nupper limit:',upper)            
            print()
            print('Total outliers:', outliers_with_IQR.shape[0] )

            if outliers_with_IQR.shape[0]!=0:

                if len(outliers_with_IQR)>10:
                    outliers_with_IQR=pd_DataFrame(outliers_with_IQR)
                    print(outliers_with_IQR[:5],'\n.\n.')
                    print(outliers_with_IQR[-5:])
                else:
                    print(outliers_with_IQR)

            print('---------------------------------')
    
    
    locals().clear()
    from gc import collect
    collect()
    del(collect)
    
####################################################################################



def analysis_quant(df, columns='all_the_columns', figsize=(20,2), dpi=120):
    '''
    Analyse quantative features.
    Prints five point summary and outliers via Z score and IQR. 
    Plots boxplot and histogram to visualise outliers.
    
    Parameters :
    ------------
        df : a pandas dataframe
        
        columns : default('all_the_columns') list of column names.
                 (if list of columns is not passed then
                 all columns are analysed)
                 
        figsize : default(20,7) set figure size
        
        dpi : default(120) set figure dpi
        
    ________________________________
    Returns : 
    ---------
        None
    '''

    from matplotlib.pyplot import subplots as plt_subplots, show as plt_show
    from seaborn import histplot as sns_histplot, boxplot as sns_boxplot

    # converting singular value of str to list 
    if type(columns)==str:
        # if list of columns is not passed then all columns are analysed
        if columns=='all_the_columns':
            columns=df.columns
        # else the passed string is converted to a list
        else:
            columns=[columns]

    for column in columns:

        # storing feature as series
        feature=df[column]

        print('\t\t\t\tANALYSIS OF:', column ,'\n')

        if feature.dtype=='object':
            print(f'Feature "{column}" might be categorical.\nPlease use "analysis_cate" function.')
            print('___________________________________________________________________________________________________________')
            continue

        # five point summary
        five_point_summary(df, column)   

        # z score and outliers
        outliers_z_score(df, column)  

        # iqr and outliers
        outliers_IQR(df, column)      

        ###PLOTTING###
        fig, axes = plt_subplots(1, 2, sharex=True, figsize=figsize, dpi=dpi)
        # boxplot
        sns_boxplot(ax=axes[0] , x=feature)  
        # histogram
        sns_histplot(ax=axes[1], data=feature, bins=50)    

        plt_show()
        print('___________________________________________________________________________________________________________')
     

    locals().clear()
    from gc import collect
    collect()
    del(collect)
    
##############################################################################################################



def analysis_cate(df, columns='all_the_columns', figsize=(12,3), dpi=120, force=False):    
    '''
    Analyse categorical features.
    Prints unique values and their counts. 
    Plots barplot and pie chart.
    
    Parameters :
    ------------
        df : a pandas dataframe
        
        columns : default('all_the_columns') list of column names.
                 (if list of columns is not passed then
                 all columns are analysed)
                 
        figsize : default(20,7) set figure size
        
        dpi : default(120) set figure dpi
    
        force : default(False) whether to proceed with a feature that
               might be numerical( !!!MAY CAUSE MEMORY LEAK!!! )
               
    ________________________________
    Returns : 
    ---------
        None 
    '''
    
    from matplotlib.pyplot import subplots as plt_subplots, show as plt_show
    from seaborn import barplot as sns_barplot

    # converting singular value of str to list 
    if type(columns)==str:
        # if list of columns is not passed then all columns are analysed
        if columns=='all_the_columns':
            columns=df.columns
        # else the passed string is converted to a list
        else:
            columns=[columns]

    for column in columns:

        # storing feature as series
        feature=df[column]

        print('\t\t\t\tANALYSIS OF:', column ,'\n')

        # calculate no. of classes in the features and warn that feature might be numerical
        if force==False:
            if feature.nunique()>20:
                print(f'The feature "{column}" might be numerical. Please try the "analysis_quant" function.\nIncase you want to proceed anyways, set "force" parameter to True.\n(Caution!!! May cause memory leak.)')
                print('______________________________________________________________________________________________________')
                continue

        if force==True:
            print(f'The feature "{column}" might be numerical. Proceeding anyways.')

        # calculate and print unique values and their counts
        values=feature.value_counts()
        print('No. of UNIQUE values:')
        print(values)
        print()

        ###PLOTTING###
        fig, axes =  plt_subplots(1, 2, figsize=figsize, dpi=dpi)
        # barplot
        sns_barplot(x=values.index, y=values, ax=axes[0])
        axes[0].set_ylabel('count')
        # pie chart
        axes[1].pie(x=values, labels=values.index )

        plt_show()
        print('_____________________________________________________________________________________________________________________')
     

    locals().clear()
    from gc import collect
    collect()
    del(collect)
    
######################################################################################################



def handle_outliers(df, columns, using='Z', action='compress', custom_intervals=(None,None)):
    '''
    Handle outliers.
    Remove or compress outliers from dataframe(inplace) by using
    either Z score or IQR. Prints the removed/compressed values.

    Parameters :
    ------------
        df : a pandas dataframe
        
        columns : list of column names from which outliers are to be
                 handled
                 
        using : {'Z': Z score,
                'IQR': Inter quartile range
                'custom' : Provide custom lower and upper limits
                           Only works if single column passed
                }
                
        action : {'compress': compresses the outliers to the extreme 
                             values using the chosen method
                 'remove': removes the outliers using the chosen method
                }
        
        custom_intervals : default( (None,None) )
            Supply this when using="custom" 
            Intervals using which outliers will be handled
            Provide custom intervals as a tuple in the format:
            (lower limit, upper limit)
            
    ________________________________
    Returns : 
    ---------
        None
    '''
    
    from pandas import DataFrame as pd_DataFrame, concat as pd_concat
    from matplotlib.pyplot import subplots as plt_subplots, show as plt_show
    from seaborn import histplot as sns_histplot 

    # converting single value to list
    if type(columns)==str:
        columns=[columns]

    for column in columns:
        before=df[column].copy()

        if using.strip().upper()=='CUSTOM': 
            if custom_intervals != (None,None):
                # setting lower and upper limit as the custom_interval 
                lower,upper=custom_intervals
                if lower==None: lower=df[column].min()
                if upper==None: upper=df[column].max()
                #making the outliers dataframe 
                outliers= pd_concat( (df[df[column]<lower], df[df[column]>upper]) )[column].sort_values()
            else:
                using='Z'
                print('Using the z score method as custom intervals were not provided')

        # if IQR method is chosen
        if using.strip().upper()=='IQR':
            # calling 'outliers_IQR_score' function to retrieve limits, outliers
            upper, lower, outliers = outliers_IQR(df, column, mode='return')

        # if Z score method is chosen
        if using.strip().upper()=='Z':
            # calling 'outliers_z_score' function to retrieve limits, outliers
            upper, lower, outliers = outliers_z_score(df, column, mode='return')

        # if remove option is chosen
        if action=='remove':
            # dropping the outliers and printing them as removed
            df.drop(index=outliers.index, inplace=True)
            print('Removed the following outliers in {column}:\n')

        # if compress action is chosen(default)
        if action=='compress':
            # compressing the outliers
            df.loc[ df[column] > upper, column] = upper
            df.loc[ df[column] < lower, column] = lower
            print(f'Compressed the following outliers in {column}:\n')

        print('Total outliers:',len(outliers))
        if len(outliers)>10:
            outliers=pd_DataFrame(outliers)
            print(outliers[:5],'\n.\n.')
            print(outliers[-5:])
        else:
            print(outliers)

        after=df[column]
        # plot the difference after handling outliers
        fig,ax=plt_subplots(1,2, figsize=(20,3), dpi=100)

        #before.hist(bins=50, ax=ax[0])
        sns_histplot(ax=ax[0], data=before, bins=50) 
        ax[0].set_title(f'{column} before')

        #after.hist(bins=50, ax=ax[1])
        sns_histplot(ax=ax[1], data=after, bins=50) 
        ax[1].set_title(f'{column} after')
        plt_show()

        print('_____________________________________________________________________________________________________________________')


    locals().clear()
    from gc import collect
    collect()
    del(collect)

###################################################################################################
'''                                       BIVARIATE ANALYSIS                                    '''
###################################################################################################



def correlation(df, figsize=(15,10), dpi=100):
    '''
    Plot correlation heatmap for a dataframe.
    Includes both pearson and spearman correlation.
    
    Parameters :
    ------------
        df : Default(None)
            a pandas dataframe

        figsize : default(15,10)
            set figure size

        dpi : default(100)
            set figure dpi
        
    ________________________________
    Returns :
    ---------
        None 
    '''
    
    from matplotlib.pyplot import subplots as plt_subplots, show as plt_show
    from seaborn import heatmap as sns_heatmap 

    fig , ax= plt_subplots(1,2, figsize=figsize, dpi=dpi)

    # plotting pearson correlation heatmap
    pearson=df.corr()
    ax[0].set_title('pearson')
    sns_heatmap(pearson, cmap='RdBu', square=True, annot=True, fmt='.2f', vmin=-1, vmax=1, ax=ax[0])

    # plotting spearman correlation heatmap
    spearman=df.corr(method='spearman')
    ax[1].set_title('spearman')
    sns_heatmap(spearman, cmap='RdBu', square=True, annot=True, fmt='.2f', vmin=-1, vmax=1, ax=ax[1])

    plt_show()
    

    locals().clear()
    from gc import collect
    collect()
    del(collect)
    
###################################################################################################



def multiplot(df, line_width=1, line_color='darkblue', point_size=0.5, point_color='darkcyan', alpha=0.5, height=1, dpi=150, aspect=1.5 ):    
    '''
    plot multiple plots like correlation(pearson) heatmap, pairwise scatterplot 
    and histogram in a single plot
    
    Parameters :
    ------------
        df : Default(None)
            a pandas dataframe

        linewidth : default(2)
            width of regression line

        line_color : default('blue')
            color of regression line

        point_size : default(0.5)
            size of scatter points

        point_color : default('darkcyan')
            color of scatter points

        alpha : default(0.5)
            transparency value for scatter points

        height : default(1)
            Height of figure

        dpi : default(100)
            dpi of figure

        aspect : default(1.5)
            aspect ratio of figure
        
    ________________________________
    Returns :
    ---------
        None
    '''
    
    from seaborn import PairGrid as sns_pairgrid, histplot as sns_histplot, despine as sns_despine, regplot as sns_regplot
    from matplotlib.pyplot import gca as plt_gca, Normalize as plt_normalize, get_cmap as plt_get_cmap, show as plt_show
    from scipy.stats import pearsonr 
    import matplotlib.style as mpl_style
    mpl_style.use("default")

    def corrfunc(x, y, **kwds):
        cmap = kwds['cmap']
        norm = kwds['norm']
        ax = plt_gca()
        ax.tick_params(bottom=False, top=False, left=False, right=False)
        sns_despine(ax=ax, bottom=True, top=True, left=True, right=True)
        r, _ = pearsonr(x, y)
        facecolor = cmap(norm(r))
        ax.set_facecolor(facecolor)
        lightness = (max(facecolor[:3]) + min(facecolor[:3]) ) / 2
        ax.annotate(f"r={r:.2f}", xy=(.5, .5), xycoords=ax.transAxes,
                    color='white' if lightness < 0.7 else 'black', size=10, ha='center', va='center')


    g = sns_pairgrid(df, height=height, aspect=aspect)
    # regression plot
    g.map_lower(sns_regplot, ci=0, scatter_kws={'s':point_size, 'alpha':alpha, 'color':point_color}, line_kws={'linewidth':line_width, 'color':line_color})
    # histplot
    g.map_diag(sns_histplot, kde=False)
    # correlation heatmap
    g.map_upper(corrfunc, norm=plt_normalize(vmin=-.5, vmax=.5), cmap=plt_get_cmap('RdBu'))
    
    # equal spacing in both directions
    g.fig.subplots_adjust(wspace=0.06, hspace=0.06) 
    
    g.fig.dpi=dpi
    plt_show()

    mpl_style.use("seaborn-darkgrid")
        

    locals().clear()
    from gc import collect
    collect()
    del(collect)
    
###################################################################################################################



def mutual_info(df, target, n_neighbors=5, limit=10, figsize=(15, 4), dpi=150):
    '''
    Plot mutual info of features with respect to the target.
    Top ten by default.
    
    Parameters :
    ------------
    df : Default(None)
        pandas dataframe
    
    target : Default(None)
        name of target column
    
    n_neighbors : Default(5)
        Number of neighbors to use for MI estimation for continuous variables.
        Higher values reduce variance of the estimation,
        but could introduce a bias.
    
    limit : Default(10)
        limit number of features plotted
        Give argument as "-1" to plot all features
        
    figsize : default(15,10)
        set figure size

    dpi : default(100)
        set figure dpi
    ________________________________
    Returns :
    ---------
        None
    '''
    
    from pandas import Series as pd_series
    from seaborn import barplot as sns_barplot
    from matplotlib.pyplot import title as plt_title, bar_label as plt_bar_label, subplots as plt_subplots, show as plt_show
    from sklearn.feature_selection import mutual_info_regression


    X = df.copy()
    y = X.pop(target)

    # Label encoding for categoricals
    for colname in X.select_dtypes("object"):
        X[colname], _ = X[colname].factorize()

    # All discrete features should now have integer dtypes (double-check this before using MI!)
    discrete_features = X.dtypes == int

    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, n_neighbors=n_neighbors)
    mi_scores = pd_series(mi_scores, name="MI Scores", index=X.columns)
    
    # sort mi scores
    mi_scores = mi_scores.sort_values(ascending=False)
    
    if limit != -1:
        mi_scores = mi_scores[:limit]

    fig, ax= plt_subplots(figsize=figsize, dpi=dpi)
    
    sns_barplot(x=mi_scores, y=mi_scores.index, orient='horizontal', ax=ax, alpha=0.8)
    
    plt_bar_label(ax.containers[-1], fmt='%.3f', label_type='edge')
    plt_title("Mutual Information Scores")
    plt_show()
    

    locals().clear()
    from gc import collect
    collect()
    del(collect)
    
###################################################################################################################



