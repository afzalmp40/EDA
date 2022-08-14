'''   

Module made to detect and handle outliers using methods like Z score and IQR.
Outlier handling methods include removing and compressing. 
Available functions:
    five_point_summary: Function to print five point summary of a feature.
    outliers_z_score: Function to analyse a feature using Z score.
    outliers_IQR: Function to analyse a feature using IQR.
    analysis_quant: Function to analyse quantative features.
    analysis_cate: Function to analyse categorical features.
    handle_outliers: Function to handle outliers.

'''

def five_point_summary(df, columns):
    '''
    Function to print five point summary of a feature.
    
    Parameters:
        df: a pandas dataframe
        columns: list of column names
    
    Returns: 
        None
        
    '''
    
    # converting singular value of str to list 
    if type(columns)==str:
        columns=[columns]
    
    for column in columns:
        print('5 point summary for:', column)
        
        # extracting and printing the five point summary from describe function
        print(df[[column]].describe().iloc[3:] )
        print('---------------------------------')
        
#############################################################################################
def outliers_z_score(df, columns, mode='print'):
    '''
    Function to analyse a feature using Z score.
    
    Parameters:
        df: a pandas dataframe
        columns: list of column names
        mode: {'print': 'only prints outliers',
               'return': 'returns outliers dataframe' 
              }

    Returns: 
        ('upper', 'lower', 'outliers_with_z') when set to 'return'
        None when set to 'print'
        
    '''
    
    # converting singular value of str to list
    if type(columns)==str:
        columns=[columns]
        
    for column in columns:
        
        ###CALCULATIONS###
        
        # storing the feature as a series 
        feature=df[column]
        
        # calculate mean and stdev
        mean=np.mean(feature)
        stdev=np.std(feature)
        
        # calculate outlier limits via Z score
        lower= -3*stdev + mean
        upper=  3*stdev + mean
        
        # calculating Z score for features
        Z=(feature-mean)/stdev
        
        # creating a mask to subset only outlier values( abs(z) > 3 )
        mask=abs( Z )>3
        
        # a dataframe storing the outliers and their z scores
        outliers_with_z=pd.DataFrame( {
                                'outliers' : feature[mask],
                                'Z-score'  : Z[mask] 
        })
        
        if mode=='return':
            return upper, lower, outliers_with_z
        
        else:
            ###PRINTING THE RESULTS###
            print( 'OUTLIERS in ' + column + ' via Z score\n' )
            print('Outlier limits:\nlower limit:', lower, '\nupper limit:', upper)
            print()
            print('Total outliers:', outliers_with_z.shape[0] )
            print( outliers_with_z )
            print('---------------------------------')
            
#############################################################################################
def outliers_IQR(df, columns, mode='print'):
    '''
    Function to analyse a feature using IQR.
    
    Parameters:
        df: a pandas dataframe
        columns: list of column names
        mode: {'print': 'only prints outliers',
               'return': 'returns outliers dataframe' 
              }
    
    Returns: 
        ('upper', 'lower', 'outliers_with_IQR') when set to 'return'
        None when set to 'print' 
        
    '''
    
    # converting singular value of str to list
    if type(columns)==str:
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
        outliers_with_IQR= feature[mask]
        outliers_with_IQR.columns='outliers'

        if mode=='return':
            return upper, lower, outliers_with_IQR
        else:
            ###PRINTING THE RESULTS###
            print( 'OUTLIERS in '+ column +' via IQR\n' )
            print('Outlier limits:\nlower limit:',lower,'\nupper limit:',upper)            
            print()
            print('Total outliers:',outliers_with_IQR.shape[0] )
            print( outliers_with_IQR )
            print('---------------------------------')
    
####################################################################################
def analysis_quant(df, columns):
    '''
    Function to analyse quantative features.
    Prints five point summary and outliers via Z score and IQR. 
    Plots boxplot and histogram to visualise outliers.
    
    Parameters:
        df: a pandas dataframe
        columns: list of column names
    
    Returns: 
        None
        
    '''
    
    # converting singular value of str to list
    if type(columns)==str:
        columns=[columns]
    
    for column in columns:
        
        # storing feature as series
        feature=df[column]
        
        print('\t\t\t\tANALYSIS OF:', column ,'\n')
        
        # five point summary
        five_point_summary(df, column)   
        
        # z score and outliers
        outliers_z_score(df, column)  
        
        # iqr and outliers
        outliers_IQR(df, column)      

        ###PLOTTING###
        fig, axes = plt.subplots(1, 2, sharex=True, figsize=(25,7),dpi=100)
        # boxplot
        sns.boxplot(ax=axes[0] , x=feature)  
        # histogram
        sns.histplot(ax=axes[1], data=feature, bins=25)    
        
        plt.show()
        print('_____________________________________________________________________________________________________________________')
        
##############################################################################################################
def analysis_cate(df, columns):    
    '''
    Function to analyse categorical features.
    Prints unique values and their counts. 
    Plots barplot and pie chart.
    
    Parameters:
        df: a pandas dataframe
        columns: list of column names
    
    Returns: 
        None
        
    '''
    
    # converting singular value of str to list
    if type(columns)==str:
        columns=[columns]
    
    for column in columns:
        
        # storing feature as series
        feature=df[column]
        
        print('\t\t\t\tANALYSIS OF:', column ,'\n')

        # calculate and print unique values and their counts
        values=feature.value_counts()
        print('No. of UNIQUE values:')
        print(values)
        print()

        ###PLOTTING###
        fig, axes =  plt.subplots(1, 2, figsize=(25,5),dpi=100)
        # barplot
        sns.barplot(x=values.index, y=values, ax=axes[0])
        axes[0].set_ylabel('count')
        # pie chart
        axes[1].pie(x=values, labels=values.index )
        
        plt.show()
        print('_____________________________________________________________________________________________________________________')
        
######################################################################################################
def handle_outliers(df, columns, using='Z', action='compress'):
    '''
    Function to handle outliers.
    Remove or compresses outliers from dataframe(inplace) by using either
    Z score or IQR. Prints the removed/compressed values.

    Parameters:
        df: a pandas dataframe
        columns: list of column names from which outliers are to be removed
        using: {'Z': Z score,
                'IQR': Inter quartile range
                }
        action: {'compress': compresses the outliers to the extreme values
                  using the chosen method
                 'remove': removes the outliers using the chosen method
                }

    Returns: 
        None
        
    '''
    
    # converting single value to list
    if type(columns)==str:
        columns=[columns]
    
    for column in columns:
        
        # if IQR method is chosen
        if using.strip().upper()=='IQR':
            # calling 'outliers_z_score' function to retrieve limits, outliers
            upper, lower, outliers = outliers_z_score(df, column, mode='return')

        # if Z score method is chosen
        if using.strip().upper()=='Z':
            # calling 'outliers_IQR' function to retrieve limits, outliers
            upper, lower, outliers = outliers_IQR(df, column, mode='return')
            
        # if remove option is chosen
        if action=='remove':
            # dropping the outliers and printing them as removed
            df.drop(index=outliers.index, inplace=True)
            print('Removed the following outliers:\n',outliers)
        
        # if compress action is chosen(default)
        if action=='compress':
            df.loc[ df[column] > upper, column] = upper
            df.loc[ df[column] < lower, column] = lower
            print('Compressed the following outliers:\n', outliers)
            
        print('_____________________________________________________________________________________________________________________')
        
################################################################################################