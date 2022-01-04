import pandas as pd  
import matplotlib.pyplot as plt
import scikit_posthocs as sp
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import nltk
import numpy as np
from scipy import stats
from scipy import stats
from nltk import FreqDist

def kruskal_dunns_func(in_df, time_col, group_name, time_point,outcome):
    """
    Check for significant differences using
    Kruskal Wallis test and Dunn's post-hoc

    Paramters
    ---------
    in_df:  pandas Dataframe
        Input dataframe
    time_col:   str
        Name of time point column in in_df
    group_name: str
        Groups to compare
    time_point: int
        time point for which to compare data
    outcome:    str
        outcome to compare on
    """
    data = []
    for group in group_name:
        data.append(in_df.loc[(in_df.Group==group)&(in_df[time_col]==time_point),outcome].dropna().values)
    print('Outcome:{},\nstats: {}\n'.format(outcome, stats.kruskal(data[0],data[1],data[2])))
    print('Posthoc:\n1 = {}, 2 = {}, 3 = {}\n {}\n'.format(group_name[0],group_name[1],
            group_name[2], sp.posthoc_dunn(data, p_adjust = 'bonferroni')))

    
def model_checks(data_df, columns_to_use, y_to_use,GEE_res):
    """
    Model diagnostics

    Parameters
    ----------
    data_df:    pandas DataFrame
        input dataframe
    columns_to_use: str
        names of columns containing predictors to be used in the analysis
    y_to_use:   str
        names of outcome column
    GEE_res:    
        GEE model results

    -------
    """
    gee_res_df = data_df.loc[:,columns_to_use]
    drop_inds = gee_res_df.loc[gee_res_df[y_to_use].isna(),:].index.values
    gee_res_df = gee_res_df.drop(labels = drop_inds, axis = 0)
    gee_res_df['resid_dev'] = GEE_res.resid_deviance
    gee_res_df['fitted'] = GEE_res.fittedvalues
    a = GEE_res.resid_deviance
    a.sort()
    fig = plt.figure()
    res = stats.probplot(a,dist = stats.halfnorm,sparams = (-0.18,10), plot=plt)
    plt.show()
    # plotting fitted agains residuals
    g = sns.lmplot(x = "fitted", y = "resid_dev", hue = "Group", data = gee_res_df)
    g = (g.set_axis_labels("Predicted score", "Deviance residuals"))#.set(xlim=(42, 55)))
    g = sns.lmplot(x = "time", y = "resid_dev", hue = "Group", data = gee_res_df)
    g = (g.set_axis_labels("Time [Weeks]", "Deviance residuals").set(xlim=(-1, 14)))


def get_qic_table(data_df, formula, group, family_name):
    """ 
    Compare models with different covariance structures, based on QIC.

    Parameters
    ----------
    data_df:    pandas DataFrame
        input dataframe
    formula:    str
        statsmodels GEE formula (patsy)
    group:  str
        column to group by
    family_name:    str
        mean response structure name
    
    Returns
    -------
        Pandas DataFrame containing qic values for each model.
    """
    model_covi = smf.gee(formula,group, data_df, cov_struct = sm.cov_struct.Independence(), family = family_name,missing = 'drop').fit()
    model_covx = smf.gee(formula,group, data_df, cov_struct = sm.cov_struct.Exchangeable(), family = family_name,missing = 'drop').fit()
    model_covar = smf.gee(formula,group, data_df, cov_struct = sm.cov_struct.Autoregressive(), family = family_name,missing = 'drop').fit()
    out_df = pd.DataFrame({'dependency type': ['Independence','Exchangeable','Autoregressive'], 'QIC': [model_covi.qic()[0], model_covx.qic()[0],model_covar.qic()[0]], 'QICu': [model_covi.qic()[1], model_covx.qic()[1],model_covar.qic()[1]]})
    return out_df


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
    qic_df = get_qic_table(in_df,formula,group_name,resp_family)
    
    return model, qic_df


def model_checks(data_df, columns_to_use, y_to_use,GEE_res):
    """
    Model diagnostics

    Parameters
    ----------
    data_df:    pandas DataFrame
        input dataframe
    columns_to_use: str
        names of columns containing predictors to be used in the analysis
    y_to_use:   str
        names of outcome column
    GEE_res:    
        GEE model results

    -------
    """
    gee_res_df = data_df.loc[:,columns_to_use]
    drop_inds = gee_res_df.loc[gee_res_df[y_to_use].isna(),:].index.values
    gee_res_df = gee_res_df.drop(labels = drop_inds, axis = 0)
    gee_res_df['resid_dev'] = GEE_res.resid_deviance
    gee_res_df['fitted'] = GEE_res.fittedvalues
    a = GEE_res.resid_deviance
    a.sort()
    fig = plt.figure()
    res = stats.probplot(a,dist = stats.halfnorm,sparams = (-0.18,10), plot=plt)
    plt.show()
    # plotting fitted agains residuals
    g = sns.lmplot(x = "fitted", y = "resid_dev", hue = "Group", data = gee_res_df)
    g = (g.set_axis_labels("Predicted score", "Deviance residuals"))#.set(xlim=(42, 55)))
    g = sns.lmplot(x = "time", y = "resid_dev", hue = "Group", data = gee_res_df)
    g = (g.set_axis_labels("Time [Weeks]", "Deviance residuals").set(xlim=(-1, 14)))

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


def get_top_words(day_col, num_day, group_col, group_name, in_df, token_col_name, **pos_tag_type):
    """ 
    Get an ordered list of words in document.

    Parameters
    ----------
    day_col:    str
        Name of df column indicating day of writing
    num_day:    int
        Day of writing (1, 2, 3 or 4)
    group_col:  str
        Name of group column in df
    group_name: str
        The group to process (EW, EWRE or CTR)
    in_df:  pd DataFrame
        input dataframe containing rel data
    token_col_name: str
        Name of column containing tokenized text
    **pos_tag_type: str/list of str
        If processing only nouns/verbs/adjectives
        pass tag to function using kwargs.
        For adjectives, use:
        'JJ'
        For verbs, use:
        'VB'
        For nouns, use:
        'NN'
        If all of the above, pass list:
        ['NN','JJ','VB']
        as kwarg.

    Returns
    -------
    list of words, list of vals
    words = words ordered from most frequent to rare
    vals = corresponding frequency
    """
    token_list = [
                item for sublist in
                [*in_df.loc[
                (in_df[group_col] == group_name) &
                (in_df[day_col] == num_day),
                token_col_name]]
                for item in sublist
                ]

    
    if pos_tag_type:
        selected_list = []
        for tag_type in pos_tag_type.values():
            w_list = [
                    word for (word,pos) 
                    in nltk.pos_tag(token_list)
                    for tag in tag_type
                    if pos[:2] == tag
                    ]
            selected_list.extend(w_list)
    else:
        selected_list = token_list
    selected_list = func_stem(selected_list)
    freqs = FreqDist(selected_list)
    common_tups = freqs.most_common()
    common_words = list(zip(*common_tups))[0]
    common_vals = list(zip(*common_tups))[1]
    return common_words, common_vals

def func_top_words(in_df, pos_tags, visualize):
    """
    Put top 50 words in dataframe,
    with option to visualize using barplots.

    Parameters
    ----------
    in_df:  pd DataFrame
    input dataframe
    pos_tags:   list
        list of pos tags to use
        can be VB, JJ, NN or
        any combination (or all) of these
    visualize:  int
    1 if visualization is needed
    0 otherwise

    Returns
    -------
    Datframe of 50 top words
    and their frequencies for 
    all days and conditions.

    """

    top_50_words = []
    top_50_vals = []
    condition = []
    days = []

    for group_name in in_df.Group.unique():
        for day in in_df.day.unique():
            words,vals= get_top_words('day', day, 'Group', group_name, in_df, 'writing_tokens', pos_tags = pos_tags)
            top_50_words.append(list(words[:50]))
            top_50_vals.append(list(vals[:50]))
            condition.append(np.repeat(group_name,50))
            days.append(np.repeat(day,50))

    data = {
        'words': flatten_list(top_50_words),
        'vals': flatten_list(top_50_vals),
        'day': flatten_list(days),
        'group': flatten_list(condition)
        }
    most_common_words_df = pd.DataFrame(data)

    if visualize:
        for num_day in most_common_words_df.day.unique():
            fig,axes = plt.subplots(3,1,figsize = (30,15),sharey = True)
            for i,group_name in enumerate(most_common_words_df.group.unique()):
                data = most_common_words_df.loc[
                                                (most_common_words_df.day==num_day) &
                                                (most_common_words_df.group == group_name),
                                                ['words','vals']
                                                ]
                sns.barplot(ax=axes[i], x=data.words, y=data.vals)
                axes[i].set_title(f'Condition: {group_name}, Day: {num_day}')

    return most_common_words_df