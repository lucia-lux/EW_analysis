import scikit_posthocs as sp
#import nltk
from scipy import stats
from textblob import TextBlob
import statsmodels.api as sm
import statsmodels.formula.api as smf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#nltk.download('vader_lexicon')


def kruskal_wallis_func(in_df, group_col, test_col):
    """
    Kruskal Wallis test and
    post-hoc Dunn's.

    Parameters
    ----------
    in_df:  pd DataFrame
        input dataframe
    group_col:  str
        name of group column
    test_col:   str
        name of column containing
        relevant values

    Returns
    -------
    Statistic, pvalue
    """
    data = in_df.pivot(columns = group_col, values = test_col)
    if len(in_df[group_col].unique())>2:
            statistic,pval = stats.kruskal(data.iloc[:,0],data.iloc[:,1],
            data.iloc[:,2],nan_policy = 'omit')
            posthoc = sp.posthoc_dunn(
                        [data.iloc[:,0].dropna(),data.iloc[:,1].dropna(),data.iloc[:,2].dropna()],
                        p_adjust = 'bonferroni'
                        )
            key = [data.columns[0],data.columns[1],data.columns[2]]
    else:
        statistic,pval = stats.kruskal(data.iloc[:,0],data.iloc[:,1],
        nan_policy = 'omit')
        posthoc = None
        key = None
    return statistic,pval, posthoc, key


def get_sentiment(in_df, writing_col):
    """
    Get subjectivity
    and polarity scores.

    Parameters
    ----------
    in_df:  pd DataFrame
        DataFrame to operate on
    writing_col: str
        column holding text data
        to operate on
    
    Returns
    -------
    Input DataFrame with
    subjectivity/polarity columns
    added.

    """
    print("Getting sentiment scores...")
    in_df = in_df.assign(
                        polarity = in_df[writing_col].astype('str').apply(
                        [lambda x: TextBlob(x).sentiment.polarity]),
                        subjectivity = in_df[writing_col].astype('str').apply(
                        [lambda x: TextBlob(x).sentiment.subjectivity])
                        )
    print("Done!")
    return in_df

def run_mixedlm(in_df,group_name,formula, re_intercept):
    """ 
    Run statsmodels LMEM.

    Parameters
    ----------
    in_df:  pandas DataFrame
        input dataframe
    group_name: str
        column to group by
    formula:    str
        patsy formula

    Returns
    -------
    MixedLMResults instance
    """
    if re_intercept:
        model = smf.mixedlm(
                            formula, in_df, groups = group_name,re_formula = re_intercept, missing = 'drop'
                            ).fit()
    else:
        model = smf.mixedlm(formula, in_df, groups = group_name,missing = 'drop').fit()
    return model

def run_gee(in_df,group_name,formula,cov_structure, resp_family):
    """ 
    Run statsmodels GEE.

    Parameters
    ----------
    in_df:  pandas DataFrame
        input dataframe
    group_name: str
        column to group by
    formula:    str
        patsy formula
    cov_structure:  sm covariance structure
        covariance structure (e.g. sm.cov_struct.Independence())
    resp_family:    sm family (e.g. sm.families.Tweedie())
        mean response structure distribution

    Returns
    -------

    """
    model = smf.gee(formula,group_name, in_df, cov_struct = cov_structure, family = resp_family,missing = 'drop').fit()
    
    return model

def flatten_list(list_of_lists):
    """
    Flatten list of lists.

    Parameters
    ----------
    list_of_lists:  list
        list of lists to flatten
    
    Returns
    -------
    Flattened list.
    """
    return [item for sub_list in list_of_lists for item in sub_list]

def add_col(in_df, col_name, var):
    """
    Add column to dataframe.
    Uses pd.df.assign.

    Parameters
    ----------
    in_df:  pd DataFrame
        df to add to
    col_name:   str
        name of column to add
    var:   
        Values to assign
        (accepts callable applied to
        existing columns, see pd.df.assign method)
    Returns
    -------
    Modified dataframe 
    """
    in_df[col_name] = var

    return in_df


def get_polarity(in_df, writing_col, polarity_type: str):
    """
    Get polarity for written statements.
    Using NLTK VADER.

    Parameters
    ----------
    in_df:  pd DataFrame
        input dataframe containing
        written statements
    writing_col:    str
        name of column containing
        statments
    polarity_type:  str
        polarity to extract
        all = get entire polarity_scores
        output as a dict
        pos = positive only
        neg = negative only
        neu = neutral only
        compound = compound
    """
    sid = SentimentIntensityAnalyzer()
    new_col_name = '_'.join(['sentiment',polarity_type, 'vader'])
    if polarity_type == 'all':   
        in_df[new_col_name] = in_df[writing_col].apply(lambda x: sid.polarity_scores(x))
    else:
        in_df[new_col_name] = in_df[writing_col].apply(lambda x: sid.polarity_scores(x)[polarity_type])
    
    return in_df

def scale_features(in_df, col_names, scaler):
    """
    Scale features.
    
    Parameters
    ----------
    in_df:  pd DataFrame
        input datatframe to operate on
    col_names: list[str]
        list of features to scale
    
    Returns
    -------
    in_df with scaled cols added.
    """
    for col in col_names:
        in_df[col+'_scaled'] = scaler.fit_transform(in_df[col].values.reshape(-1,1))
    return in_df

def get_word_count(token_list):
    """
    Get word count after
    preprocessing.

    Parameters
    ----------
    token_list: list[str]
        list of tokens
    Returns
    -------
    token count for each document
    """
    return [len(f) for f in token_list]

def get_word_count_raw(in_df, writing_col):
    """
    Get word count for raw text.

    Parameters
    ----------
    in_df:  pd DataFrame
        input dataframe
    writing_col:    str
        name of column containing statements

    Returns
    -------
    raw word count
    """
    word_count_raw = in_df[writing_col].apply(lambda x: len(x.split()))
    return word_count_raw

def add_word_count(in_df, writing_col, tokens):
    """
    Add word count to dataframe

    Parameters
    ----------
    in_df:  pd DataFrame
        input dataframe
    word_count_tokenized:   list
        tokenized word count
    word_count_raw: list
        raw word count
    
    Returns
    -------
    in_df with word count cols added
    """
    word_count_raw = get_word_count_raw(in_df,writing_col)
    word_count_tokenized = get_word_count(tokens)
    for name,new_name in [(word_count_raw,'word_count_raw'), (word_count_tokenized,'word_count_tokenized')]:
        in_df = add_col(in_df, new_name, name)
    return in_df