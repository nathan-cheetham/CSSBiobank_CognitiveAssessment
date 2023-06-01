# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 14:02:04 2022

@author: k2143494

Analysis of CSS Biobank Cognitron data: Factors associated with cognitive testing performance
"""

import numpy as np
import pandas as pd
# https://github.com/EducationalTestingService/factor_analyzer/blob/main/factor_analyzer/factor_analyzer.py
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer import FactorAnalyzer
import statsmodels.api as sm
from statsmodels.genmod.families.links import logit
from statsmodels.genmod.families.links import identity
from statsmodels.tools.eval_measures import rmse
from statsmodels.stats.multitest import fdrcorrection
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn import decomposition
import matplotlib
import matplotlib.pyplot as plt 
plt.rc("font", size=12)
from scipy.stats import chi2_contingency 
from scipy.stats import zscore
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import seaborn as sns
from pySankey.sankey import sankey


export_csv = 0

# Set which stringency limit on logging frequency to use: '7' or '14' 
stringency = '14' 


#%% Create functions
# -----------------------------------------------------------------------------
### For categorical variables, do chi-squared univariate association test and plot cross-tabulation
def categorical_variables_chi_square(data, input_var_categorical, outcome_var, drop_missing, plot_crosstab, print_vars):
    """ Perform chi-square test on cross-tabulation between outcome variable and each categorical variable in input list. Also print cross-tab as proportion in stacked bar chart """
    p_value_list = []
    p_value_missing_dropped_list = []
    var_list_chisquare = []
    for var_cat in input_var_categorical:
        if print_vars == 'yes':
            print(var_cat + ' x ' + outcome_var)
        # Cross tab of frequencies
        crosstab = pd.crosstab(data[var_cat], data[outcome_var], margins = False)            
        # Generate chi-squared statistic
        chi2, p, dof, ex = chi2_contingency(crosstab, correction=False)
        # Save chi-squared results to list
        p_value_list.append(p)
        
        # Drop missing data
        if drop_missing == 'yes':
            crosstab = crosstab.reset_index()
            crosstab = crosstab[~(crosstab[var_cat].isin(missing_data_values))]
            crosstab = crosstab.set_index(var_cat)
                
            # Generate chi-squared statistic
            chi2, p, dof, ex = chi2_contingency(crosstab, correction=False)
            # Save chi-squared results to list
            p_value_missing_dropped_list.append(p)
    
    # Create dataframe showing chi-square test results
    chisquare_results = pd.DataFrame({'Variable':input_var_categorical,
                                      'Chi-squared p-value (with missing)':p_value_list,                                      
                                      })
    if drop_missing == 'yes':
        chisquare_results['Chi-squared p-value (no missing)'] = p_value_missing_dropped_list
    
    return chisquare_results

# -----------------------------------------------------------------------------
### Add dummy variable fields to dataframe generated from un-ordered categoricals
def categorical_to_dummy(df, variable_list_categorical):
    """Create dummy variables from un-ordered categoricals"""
    # Create dummy variables
    dummy_var_list_full = []
    for var in variable_list_categorical:
        df[var] = df[var].fillna('NaN') # fill NaN with 'No data' so missing data can be distinguished from 0 results
        cat_list ='var'+'_'+var # variable name
        cat_list = pd.get_dummies(df[var], prefix=var) # create binary variable of category value
        df = df.join(cat_list) # join new column to dataframe
    
    return df

# -----------------------------------------------------------------------------
### Generate list of categorical dummy variables from original fieldname, deleting original fieldname and deleting reference variable using reference dummy variable list
def generate_dummy_list(original_fieldname_list, full_fieldname_list, reference_fieldname_list, delete_reference):
    """ Generate list of categorical dummy variables from original fieldname, deleting original fieldname. Option to also delete reference variable using reference dummy variable list (delete_reference = 'yes') """
    dummy_list = []
    for var in original_fieldname_list:
        print(var)
        var_matching_all = [variable_name for variable_name in full_fieldname_list if var in variable_name]
        # drop original variable
        var_matching_all.remove(var)
        if delete_reference == 'yes':
            # drop reference variable
            var_matching_reference = [variable_name for variable_name in reference_fieldname_list if var in variable_name][0]
            var_matching_all.remove(var_matching_reference)
        # add to overall list
        dummy_list += var_matching_all
    
    return dummy_list

# -----------------------------------------------------------------------------
### Function to summarise statsmodel results in dataframe
# From https://stackoverflow.com/questions/51734180/converting-statsmodels-summary-object-to-pandas-dataframe
def results_summary_to_dataframe(results, round_dp_coeff):
    '''take the result of an statsmodel results table and transforms it into a dataframe'''
    pvals = results.pvalues
    coeff = results.params
    conf_lower = results.conf_int()[0]
    conf_higher = results.conf_int()[1]

    results_df = pd.DataFrame({"pvals":pvals,
                               "coeff":coeff,
                               "conf_lower":conf_lower,
                               "conf_higher":conf_higher
                                })
    
    # Reorder columns
    results_df = results_df[["coeff","conf_lower","conf_higher","pvals"]]
        
    # Highlight variables where confidence intervals are both below 1 or both above 1
    results_df.loc[(results_df['pvals'] < 0.05)
                        ,'Significance'] = 'Significant, *, p < 0.05'
    results_df.loc[(results_df['pvals'] < 0.01)
                        ,'Significance'] = 'Significant, **, p < 0.01'
    results_df.loc[(results_df['pvals'] < 0.001)
                        ,'Significance'] = 'Significant, ***, p < 0.001'
          
    results_df['tidy_string'] = results_df['coeff'].round(round_dp_coeff).astype(str) + ' (' + results_df['conf_lower'].round(round_dp_coeff).astype(str) + ', ' + results_df['conf_higher'].round(round_dp_coeff).astype(str) + '),' 
    
    results_df.loc[(results_df['pvals'] >= 0.05)
                   ,'tidy_string'] = results_df['tidy_string'] + ' p = ' + results_df['pvals'].round(2).astype(str)
    results_df.loc[(results_df['pvals'] < 0.05) &
                   (results_df['pvals'] >= 0.01)
                   ,'tidy_string'] = results_df['tidy_string'] + ' p = ' + results_df['pvals'].round(2).astype(str) + '*'
    results_df.loc[(results_df['pvals'] < 0.01) &
                   (results_df['pvals'] >= 0.001)
                   ,'tidy_string'] = results_df['tidy_string'] + ' p = ' + results_df['pvals'].round(3).astype(str) + '**'
    results_df.loc[(results_df['pvals'] < 0.001) &
                   (results_df['pvals'] >= 0.0001)
                   ,'tidy_string'] = results_df['tidy_string'] + ' p = ' + results_df['pvals'].round(4).astype(str) + '***'
    results_df.loc[(results_df['pvals'] < 0.0001)
                   ,'tidy_string'] = results_df['tidy_string'] + ' p < 0.0001***'
    
    results_df = results_df.reset_index()
    results_df = results_df.rename(columns = {'index':'variable'})
    
    return results_df


# -----------------------------------------------------------------------------
### Run logistic regression model with HC3 robust error, producing summary dataframe  
def sm_logreg_simple_HC3(x_data, y_data, CI_alpha, do_robust_se, use_weights, weight_data):
    """ Run logistic regression model with HC3 robust error, producing summary dataframe """
    # Add constant - default for sklearn but not statsmodels
    x_data = sm.add_constant(x_data) 
    
    # Add weight data to x_data if weights specified
    if use_weights == 'yes':
        x_data['weight'] = weight_data
        
    # Set model parameters
    max_iterations = 2000
    solver_method = 'newton' # use default
    
    # Also run model on test and train split to assess predictive power
    # Generate test and train split
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, stratify = y_data, test_size = 0.25, random_state = 0)
    
    # drop dummy columns where sum is < 1 - i.e. no observations. Inclusion of these empty columns causes model fit to fail
    x_data_sum_before = x_data.sum()
    for col in x_train: 
        if x_train[col].sum() <= 1:
            x_train = x_train.drop(columns = col)
            x_test = x_test.drop(columns = col)
            x_data = x_data.drop(columns = col)
    x_data_sum_after = x_data.sum().reset_index()
    x_train_sum_after = x_train.sum().reset_index()
    
    # Save weight data in x_train and the drop weight data 
    if use_weights == 'yes':
        weight_data_train = np.asarray(x_train['weight'].copy())
        # drop weight columns
        x_data = x_data.drop(columns = ['weight'])
        x_train = x_train.drop(columns = ['weight'])
        x_test = x_test.drop(columns = ['weight'])
    
    # Set up overall and test-train models
    if use_weights == 'yes':
        model = sm.GLM(y_data, x_data, 
                       var_weights = weight_data, 
                       family = sm.families.Binomial(),
                       link=logit)
        model_testtrain = sm.GLM(y_train, x_train, 
                       var_weights = weight_data_train, 
                       family = sm.families.Binomial(),
                       link=logit)
    else:
        model = sm.GLM(y_data, x_data, 
                       family = sm.families.Binomial(), 
                       link=logit)
        model_testtrain = sm.GLM(y_train, x_train,
                       family = sm.families.Binomial(),
                       link=logit)
    # Fit model
    if do_robust_se == 'HC3':
        model_fit = model.fit(maxiter = max_iterations, 
                          method = solver_method,  
                          cov_type='HC3',
                          use_t=True)
        model_testtrain_fit = model_testtrain.fit(maxiter = max_iterations, 
                          method = solver_method,  
                          cov_type='HC3',
                          use_t=True)
    else:
        model_fit = model.fit(maxiter = max_iterations, 
                              method = solver_method, 
                              use_t=True)
        model_testtrain_fit = model_testtrain.fit(maxiter = max_iterations, 
                              method = solver_method, 
                              use_t=True) 
    
    # print(model_fit.summary())
   
    # Calculate AUC of model
    y_prob = model_testtrain_fit.predict(x_test)
    if np.isnan(np.min(y_prob)) == False:
        model_auc = roc_auc_score(y_test, y_prob)
    else:
        print(y_prob)
        model_auc = 0 # for when AUC failed - e.g. due to non-convergence of model
    
    # Extract coefficients and convert to Odds Ratios
    sm_coeff = model_fit.params
    sm_se = model_fit.bse
    sm_pvalue = model_fit.pvalues
    sm_coeff_CI = model_fit.conf_int(alpha=CI_alpha)
    sm_OR = np.exp(sm_coeff)
    sm_OR_CI = np.exp(sm_coeff_CI)
    
    # Create dataframe summarising results
    sm_summary = pd.DataFrame({'Variable': sm_coeff.index,
                               'Coefficients': sm_coeff,
                               'Standard Error': sm_se,
                               'P-value': sm_pvalue,
                               'Coefficient C.I. (lower)': sm_coeff_CI[0],
                               'Coefficient C.I. (upper)': sm_coeff_CI[1],
                               'Odds ratio': sm_OR,
                               'OR C.I. (lower)': sm_OR_CI[0],
                               'OR C.I. (upper)': sm_OR_CI[1],
                               'OR C.I. error (lower)': np.abs(sm_OR - sm_OR_CI[0]),
                               'OR C.I. error (upper)': np.abs(sm_OR - sm_OR_CI[1]),
                                })
    sm_summary = sm_summary.reset_index(drop = True)
      
    # Add number of observations for given variable in input and outcome datasets
    x_data_count = x_data.sum()
    x_data_count.name = "x_data count"
    sm_summary = pd.merge(sm_summary,x_data_count, how = 'left', left_on = 'Variable', right_index = True)
    
    # join x_data and y_data
    x_y_data = x_data.copy()
    x_y_data['y_data'] = y_data
    # Count observation where y_data = 1
    y_data_count = x_y_data[x_y_data['y_data'] == 1].sum()
    y_data_count.name = "y_data = 1 count"
    sm_summary = pd.merge(sm_summary,y_data_count, how = 'left', left_on = 'Variable', right_index = True)
    
    # Highlight variables where confidence intervals are both below 1 or both above 1
    sm_summary.loc[(sm_summary['OR C.I. (lower)'] > 1.0)
                        & (sm_summary['OR C.I. (upper)'] > 1.0)
                        & (sm_summary['P-value'] < 0.05)
                        ,'Significance'] = 'Significant (OR > 1), *, p < 0.05'
    sm_summary.loc[(sm_summary['OR C.I. (lower)'] > 1.0)
                        & (sm_summary['OR C.I. (upper)'] > 1.0)
                        & (sm_summary['P-value'] < 0.01)
                        ,'Significance'] = 'Significant (OR > 1), **, p < 0.01'
    sm_summary.loc[(sm_summary['OR C.I. (lower)'] > 1.0)
                        & (sm_summary['OR C.I. (upper)'] > 1.0)
                        & (sm_summary['P-value'] < 0.001)
                        ,'Significance'] = 'Significant (OR > 1), ***, p < 0.001'
    
    sm_summary.loc[(sm_summary['OR C.I. (lower)'] < 1.0)
                        & (sm_summary['OR C.I. (upper)'] < 1.0)
                        & (sm_summary['P-value'] < 0.05)
                        ,'Significance'] = 'Significant (OR < 1), *, p < 0.05'
    sm_summary.loc[(sm_summary['OR C.I. (lower)'] < 1.0)
                        & (sm_summary['OR C.I. (upper)'] < 1.0)
                        & (sm_summary['P-value'] < 0.01)
                        ,'Significance'] = 'Significant (OR < 1), **, p < 0.01'
    sm_summary.loc[(sm_summary['OR C.I. (lower)'] < 1.0)
                        & (sm_summary['OR C.I. (upper)'] < 1.0)
                        & (sm_summary['P-value'] < 0.001)
                        ,'Significance'] = 'Significant (OR < 1), ***, p < 0.001'
        
    return sm_summary, model_fit, model_auc


# -----------------------------------------------------------------------------
### Run generalised linear regression model with HC3 robust error, producing summary dataframe  
def sm_OLS_simple_HC3(x_data, y_data, CI_alpha, do_robust_se, use_weights, weight_data):
    """ Run OLS regression model, using GLM class, with option to use weights and HC3 robust errors, producing summary dataframe and goodness of fit metrics """
    # Add constant - default for sklearn but not statsmodels
    x_data = sm.add_constant(x_data) 
    
    # Add weight data to x_data if weights specified
    if use_weights == 'yes':
        x_data['weight'] = weight_data
        
    # Set model parameters
    max_iterations = 2000
    solver_method = 'newton' # use default
    # model = sm.Logit(y_data, x_data, use_t = True) # Previous model - same results. Replaced with more general construction, as GLM allows weights to be included. 
    
    # Also run model on test and train split to assess predictive power
    # Generate test and train split
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.25, random_state = 0)
    
    # drop dummy columns where sum is == 0 - i.e. no observations. Inclusion of these empty columns causes model fit to fail
    x_data_sum_before = x_data.sum()
    for col in x_train: 
        if x_train[col].sum() == 0:
            # print('dropping ' + col + ' because sum == 0, sum = ' + str(x_train[col].sum()))
            x_train = x_train.drop(columns = col)
            x_test = x_test.drop(columns = col)
            x_data = x_data.drop(columns = col)
    x_data_sum_after = x_data.sum().reset_index()
    x_train_sum_after = x_train.sum().reset_index()
    
    # Save weight data in x_train and the drop weight data 
    if use_weights == 'yes':
        weight_data_train = np.asarray(x_train['weight'].copy())
        # drop weight columns
        x_data = x_data.drop(columns = ['weight'])
        x_train = x_train.drop(columns = ['weight'])
        x_test = x_test.drop(columns = ['weight'])
    
    # Set up overall and test-train models
    # Gaussian family + Identity link is equivalent to OLS
    # https://stats.stackexchange.com/questions/211585/how-does-ols-regression-relate-to-generalised-linear-modelling
    # https://stats.stackexchange.com/questions/167588/ols-vs-poisson-glm-with-identity-link?rq=1
    if use_weights == 'yes':
        model = sm.GLM(y_data, x_data, 
                       var_weights = weight_data, 
                       family = sm.families.Gaussian(),
                       link=identity)
        model_testtrain = sm.GLM(y_train, x_train, 
                       var_weights = weight_data_train, 
                       family = sm.families.Gaussian(),
                       link=identity)
    else:
        model = sm.GLM(y_data, x_data, 
                       family = sm.families.Gaussian(), 
                       link=identity)
        model_testtrain = sm.GLM(y_train, x_train,
                       family = sm.families.Gaussian(),
                       link=identity)
    # Fit model
    if do_robust_se == 'HC3':
        model_fit = model.fit(maxiter = max_iterations, 
                          method = solver_method,  
                          cov_type='HC3',
                          use_t=True)
        model_testtrain_fit = model_testtrain.fit(maxiter = max_iterations, 
                          method = solver_method,  
                          cov_type='HC3',
                          use_t=True)
    else:
        model_fit = model.fit(maxiter = max_iterations, 
                              method = solver_method, 
                              use_t=True)
        model_testtrain_fit = model_testtrain.fit(maxiter = max_iterations, 
                              method = solver_method, 
                              use_t=True) 
    
    # print(model_fit.summary())
   
    ### Calculate goodness of fit metrics for main model
    # Adjusted R squared
    # model_adjusted_r_squared = model_fit.rsquared_adj
    
    # R squared
    sst = sum(map(lambda xdata_sfs: np.power(xdata_sfs,2), y_data - np.mean(y_data))) 
    sse = sum(map(lambda xdata_sfs: np.power(xdata_sfs,2), model_fit.resid_response)) 
    model_r_squared = 1.0 - (sse/sst)
    
    # Bayes information criterion
    model_bic = model_fit.bic
    # Akaike information criterion
    model_aic = model_fit.aic 
   
    ### Calculate goodness of fit metrics for test-train model
    # Calculate predicted values
    model_y_pred = model_testtrain_fit.predict(x_test)
    # Root mean square error
    model_testtrain_rmse = rmse(y_test, model_y_pred)
    # Mean absolute error
    model_testtrain_mae = mean_absolute_error(y_test, model_y_pred)
    # R squared of predicted values
    model_testtrain_r_squared = r2_score(y_test, model_y_pred)
   
    model_goodness_list = [model_bic, model_aic, model_r_squared, model_testtrain_r_squared, model_testtrain_rmse, model_testtrain_mae] 
   
    # Create dataframe summarising results
    sm_coeff = model_fit.params
    sm_se = model_fit.bse
    sm_pvalue = model_fit.pvalues
    sm_coeff_CI = model_fit.conf_int(alpha=CI_alpha)
    sm_summary = pd.DataFrame({'Variable': sm_coeff.index,
                               'coeff': sm_coeff,
                               'standard_error': sm_se,
                               'p_value': sm_pvalue,
                               'conf_lower': sm_coeff_CI[0],
                               'conf_upper': sm_coeff_CI[1],
                               'conf_lower_error': np.abs(sm_coeff - sm_coeff_CI[0]),
                               'conf_upper_error': np.abs(sm_coeff - sm_coeff_CI[1]),
                                })
    sm_summary = sm_summary.reset_index(drop = True)
      
    # Add number of observations for given variable in input and outcome datasets
    x_data_count = x_data.sum()
    x_data_count.name = "x_data count"
    sm_summary = pd.merge(sm_summary,x_data_count, how = 'left', left_on = 'Variable', right_index = True)
    
    # join x_data and y_data
    x_y_data = x_data.copy()
    x_y_data['y_data'] = y_data
    # Count observation where y_data = 1
    y_data_count = x_y_data[x_y_data['y_data'] == 1].sum()
    y_data_count.name = "y_data = 1 count"
    sm_summary = pd.merge(sm_summary,y_data_count, how = 'left', left_on = 'Variable', right_index = True)
    
    # Highlight variables where confidence intervals are both below 1 or both above 1
    sm_summary.loc[(sm_summary['conf_lower'] > 0.0)
                        & (sm_summary['conf_upper'] > 0.0)
                        & (sm_summary['p_value'] < 0.05)
                        ,'Significance'] = 'Significant (positive), *, p < 0.05'
    sm_summary.loc[(sm_summary['conf_lower'] > 0.0)
                        & (sm_summary['conf_upper'] > 0.0)
                        & (sm_summary['p_value'] < 0.01)
                        ,'Significance'] = 'Significant (positive), **, p < 0.01'
    sm_summary.loc[(sm_summary['conf_lower'] > 0.0)
                        & (sm_summary['conf_upper'] > 0.0)
                        & (sm_summary['p_value'] < 0.001)
                        ,'Significance'] = 'Significant (positive), ***, p < 0.001'
    
    sm_summary.loc[(sm_summary['conf_lower'] < 0.0)
                        & (sm_summary['conf_upper'] < 0.0)
                        & (sm_summary['p_value'] < 0.05)
                        ,'Significance'] = 'Significant (negative), *, p < 0.05'
    sm_summary.loc[(sm_summary['conf_lower'] < 0.0)
                        & (sm_summary['conf_upper'] < 0.0)
                        & (sm_summary['p_value'] < 0.01)
                        ,'Significance'] = 'Significant (negative), **, p < 0.01'
    sm_summary.loc[(sm_summary['conf_lower'] < 0.0)
                        & (sm_summary['conf_upper'] < 0.0)
                        & (sm_summary['p_value'] < 0.001)
                        ,'Significance'] = 'Significant (negative), ***, p < 0.001'
    
    # Create tidy strings for displaying in tables and graphics
    # Coefficient and confidence intervals
    round_dp_coeff = 2
    sm_summary['coeff_string'] = sm_summary['coeff'].round(round_dp_coeff).astype(str)
    sm_summary['coeff_with_conf_string'] = sm_summary['coeff'].round(round_dp_coeff).astype(str) + ' (' + sm_summary['conf_lower'].round(round_dp_coeff).astype(str) + ', ' + sm_summary['conf_upper'].round(round_dp_coeff).astype(str) + '),' 
    
    # Coefficient only with stars indicating p-value    
    sm_summary.loc[(sm_summary['p_value'] >= 0.05)
                   ,'coeff_with_p_value_stars'] = sm_summary['coeff_string']
    sm_summary.loc[(sm_summary['p_value'] < 0.05) &
                   (sm_summary['p_value'] >= 0.01)
                   ,'coeff_with_p_value_stars'] = sm_summary['coeff_string'] + '*'
    sm_summary.loc[(sm_summary['p_value'] < 0.01) &
                   (sm_summary['p_value'] >= 0.001)
                   ,'coeff_with_p_value_stars'] = sm_summary['coeff_string'] + '**'
    sm_summary.loc[(sm_summary['p_value'] < 0.001)
                   ,'coeff_with_p_value_stars'] = sm_summary['coeff_string'] + '***'
    
    # P-value stars only
    sm_summary.loc[(sm_summary['p_value'] >= 0.05)
                   ,'p_value_stars'] = ''
    sm_summary.loc[(sm_summary['p_value'] < 0.05) &
                   (sm_summary['p_value'] >= 0.01)
                   ,'p_value_stars'] = '*'
    sm_summary.loc[(sm_summary['p_value'] < 0.01) &
                   (sm_summary['p_value'] >= 0.001)
                   ,'p_value_stars'] = '**'
    sm_summary.loc[(sm_summary['p_value'] < 0.001)
                   ,'p_value_stars'] = '***'
    
    
    # Coefficient only with full p-value 
    sm_summary.loc[(sm_summary['p_value'] >= 0.05)
                   ,'coeff_with_p_value_full'] = sm_summary['coeff_string'] + ' p = ' + sm_summary['p_value'].round(2).astype(str)
    sm_summary.loc[(sm_summary['p_value'] < 0.05) &
                   (sm_summary['p_value'] >= 0.01)
                   ,'coeff_with_p_value_full'] = sm_summary['coeff_string'] + ' p = ' + sm_summary['p_value'].round(2).astype(str) + '*'
    sm_summary.loc[(sm_summary['p_value'] < 0.01) &
                   (sm_summary['p_value'] >= 0.001)
                   ,'coeff_with_p_value_full'] = sm_summary['coeff_string'] + ' p = ' + sm_summary['p_value'].round(3).astype(str) + '**'
    sm_summary.loc[(sm_summary['p_value'] < 0.001) &
                   (sm_summary['p_value'] >= 0.0001)
                   ,'coeff_with_p_value_full'] = sm_summary['coeff_string'] + ' p = ' + sm_summary['p_value'].round(4).astype(str) + '***'
    sm_summary.loc[(sm_summary['p_value'] < 0.0001)
                   ,'coeff_with_p_value_full'] = sm_summary['coeff_string'] + ' p < 0.0001***'   
    
    return sm_summary, model_fit, model_goodness_list


# -----------------------------------------------------------------------------
# Function to run a series of OLS linear regression models
def run_OLS_regression_models(data, data_full_col_list, model_var_list, outcome_var, use_weights, weight_var, plot_fig):
    """ Function to run OLS linear regression models, given lists of categorical and continuous input variables """
    # List of missing data strings to identify absence of available data
    missing_data_values = [np.nan, 'NaN','nan', '0.1 Unknown - Answer not provided'] 
    
    model_input_list = []
    model_summary_list = []
    model_fit_list = []
    model_goodness_list_list = []
    for sublist in model_var_list:
        var_continuous = sublist[0]
        var_categorical = sublist[1]
        model_name = sublist[2]
        var_exposure = sublist[3] # identify exposure variable being tested in model
        add_round1 = sublist[4]
        
        # Generate list of dummy fields for complete fields
        var_categorical_dummy = generate_dummy_list(original_fieldname_list = var_categorical, 
                                                         full_fieldname_list = data_full_col_list, 
                                                         reference_fieldname_list = cols_categorical_reference,
                                                         delete_reference = 'yes')
    
        # Set variables to go into model
        if add_round1 == 'addround1': # include round 1 equivalent of current outcome variable to model
            input_var_control_test = var_continuous + var_categorical_dummy + [('round_1_' + outcome_var)]
            model_input = str(var_continuous + var_categorical + [('round_1_' + outcome_var)])
        else:
            input_var_control_test = var_continuous + var_categorical_dummy
            model_input = str(var_continuous + var_categorical)
            
        model_input_list.append(model_input)
        print('model input variables: ' + model_input)
        
        # Filter out missing or excluded data
        print('Individuals before filtering: ' + str(data.shape[0]))
        data_filterformodel = data.copy()
        for col in input_var_control_test:
                data_filterformodel = data_filterformodel[~(data_filterformodel[col].isin(missing_data_values))]
        print('Individuals after filtering: ' + str(data_filterformodel.shape[0]))
        
        # generate x dataset for selected control and test dummy variables only
        reg_data_x = data_filterformodel[input_var_control_test].reset_index(drop=True) # create input variable tables for models             
        # generate y datasets from selected number of vaccinations and outcome of interest
        reg_data_y = data_filterformodel[outcome_var].reset_index(drop=True) # set output variable
        
        if use_weights == 'yes':
            data_weight = data_filterformodel[weight_var].reset_index(drop=True) # filter for weight variable
        
            model_df, model_fit, model_goodness_list = sm_OLS_simple_HC3(x_data = reg_data_x, y_data = reg_data_y, CI_alpha = 0.05, do_robust_se = '', use_weights = use_weights, weight_data = np.asarray(data_weight))   
        else: 
            model_df, model_fit, model_goodness_list = sm_OLS_simple_HC3(x_data = reg_data_x, y_data = reg_data_y, CI_alpha = 0.05, do_robust_se = '', use_weights = '', weight_data = '') 
                
        model_df['outcome_variable'] = outcome_var
        
        model_df['model_input'] = model_input
        model_df['model_name'] = model_name 
        model_df['var_exposure'] = var_exposure
        
        model_summary_list.append(model_df)
        model_fit_list.append(model_fit)
        model_goodness_list_list.append(model_goodness_list)
        
        # Print prediction R squared
        print ('Predicted R squared: ' + str(model_goodness_list[3]))
        
        
    # -----------------------------------------------------------------------------
    # Combine model results tables together
    model_results_summary = pd.concat(model_summary_list)
    
    return model_results_summary, model_goodness_list_list, model_fit_list



# -----------------------------------------------------------------------------
# Function to run a series of logistic regression models
def run_logistic_regression_models(data, data_full_col_list, model_var_list, outcome_var, use_weights, weight_var):
    """ Function to run logistic regression models, given lists of categorical and continuous input variables """
    # List of missing data strings to identify absence of available data
    missing_data_values = [np.nan, 'NaN','nan', '0.1 Unknown - Answer not provided'] 
    
    model_input_list = []
    model_auc_list= []
    model_summary_list = []
    model_fit_list = []
    for sublist in model_var_list:
        var_continuous = sublist[0]
        var_categorical = sublist[1]
        
        # Generate list of dummy fields for complete fields
        var_categorical_dummy = generate_dummy_list(original_fieldname_list = var_categorical, 
                                                         full_fieldname_list = data_full_col_list, 
                                                         reference_fieldname_list = cols_categorical_reference,
                                                         delete_reference = 'yes')
    
        # Set variables to go into model
        input_var_control_test = var_continuous + var_categorical_dummy
        
        model_input = str(var_continuous + var_categorical)
        model_input_list.append(model_input)
        print('model input variables: ' + model_input)
        
        # Filter out missing or excluded data
        print('Individuals before filtering: ' + str(data.shape[0]))
        data_filterformodel = data.copy()
        for col in input_var_control_test:
                data_filterformodel = data_filterformodel[~(data_filterformodel[col].isin(missing_data_values))]
        print('Individuals after filtering: ' + str(data_filterformodel.shape[0]))
        
        # generate x dataset for selected control and test dummy variables only
        logreg_data_x = data_filterformodel[input_var_control_test].reset_index(drop=True) # create input variable tables for models 
        # generate y datasets from selected number of vaccinations and outcome of interest
        logreg_data_y = data_filterformodel[outcome_var].reset_index(drop=True) # set output variable
        
        if use_weights == 'yes':
            logreg_data_weight = data_filterformodel[weight_var].reset_index(drop=True) # filter for weight variable
            # Do logistic regression (stats models) of control + test variables
            sm_summary, model_fit, model_auc = sm_logreg_simple_HC3(x_data = logreg_data_x, y_data = logreg_data_y, 
                                                     CI_alpha = 0.05, do_robust_se = 'HC3',
                                                     use_weights = use_weights, weight_data = np.asarray(logreg_data_weight))
        else:
            sm_summary, model_fit, model_auc = sm_logreg_simple_HC3(x_data = logreg_data_x, y_data = logreg_data_y, 
                                                     CI_alpha = 0.05, do_robust_se = 'HC3',
                                                     use_weights = '', weight_data = '')
            
        sm_summary['model_input'] = model_input
        model_summary_list.append(sm_summary)
        model_fit_list.append(model_fit)
        
        # Print predictive power
        model_auc_list.append(model_auc) 
        print ('AUC: ' + str(model_auc))
        
    
    # -----------------------------------------------------------------------------
    # Combine model results tables together
    model_results_summary = pd.concat(model_summary_list)
    model_auc_summary = pd.DataFrame({'model_input':model_input_list,
                                      'model_auc':model_auc_list,})
    
    return model_results_summary, model_auc_summary, model_fit_list


# -----------------------------------------------------------------------------
### Generate heatmap of correlations
def heatmap(x, y, size):
    """ Generate heatmap of correlations. Adapted from https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec """
    fig, ax = plt.subplots()
    
    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    
    size_scale = 500
    ax.scatter(
        x=x.map(x_to_num), # Use mapping for x
        y=y.map(y_to_num), # Use mapping for y
        s=size * size_scale, # Vector of square sizes, proportional to size parameter
        marker='s' # Use square as scatterplot marker
    )
    
    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)


# -----------------------------------------------------------------------------
# Function to add reference category values into OLS results table
def add_reference_values(data, outcome_var):
    """ Function to add reference category values into OLS results table """
    # Add coeff = 0 for reference variables
    reference_age = {'Variable':'Combined_Age_2021_grouped_decades_4: 50-60', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'Combined_Age_2021_grouped_decades', 'outcome_variable':outcome_var}
    reference_sex = {'Variable':'ZOE_demogs_sex_Female', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'ZOE_demogs_sex', 'outcome_variable':outcome_var}
    reference_ethnicity = {'Variable':'Combined_EthnicityCategory_White', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'Combined_EthnicityCategory', 'outcome_variable':outcome_var}
    reference_education = {'Variable':'educationLevel_cat4_2. Undergraduate degree', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'educationLevel_cat4', 'outcome_variable':outcome_var}
    reference_region = {'Variable':'Region_London', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'Region', 'outcome_variable':outcome_var}
    reference_covid_group = {'Variable':'symptomduration_grouped1_stringencylimit'+stringency+'_N1: Negative COVID-19, asymptomatic (healthy control)', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'symptomduration_grouped1_stringencylimit'+stringency, 'outcome_variable':outcome_var}
    reference_IMD = {'Variable':'Combined_IMD_Quintile_3.0', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'Combined_IMD_Quintile', 'outcome_variable':outcome_var}
    
    reference_BMI = {'Variable':'Combined_BMI_cat5_2: 18.5-25', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'Combined_BMI_cat5', 'outcome_variable':outcome_var}
    reference_hospitalisation = {'Variable':'Flag_InHospitalDuringSpell_stringencylimit'+stringency+'_0.0', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'Flag_InHospitalDuringSpell_stringencylimit'+stringency, 'outcome_variable':outcome_var}
    reference_comorbidities = {'Variable':'ZOE_conditions_condition_count_cat3_0 conditions', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'ZOE_conditions_condition_count_cat3', 'outcome_variable':outcome_var}
    reference_comorbidities_MH = {'Variable':'ZOE_mentalhealth_condition_cat4_0 conditions', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'ZOE_mentalhealth_condition_cat4', 'outcome_variable':outcome_var}
    
    reference_chalder = {'Variable':'q_chalderFatigue_cat2_1. 0-28, below threshold', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'q_chalderFatigue_cat2', 'outcome_variable':outcome_var}
    reference_PHQ4 = {'Variable':'q_PHQ4_cat4_1. 0-2, below threshold', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'q_PHQ4_cat4', 'outcome_variable':outcome_var}
    reference_WSAS = {'Variable':'q_WSAS_cat4_1. 0-9, below threshold', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'q_WSAS_cat4', 'outcome_variable':outcome_var}
    reference_recovery = {'Variable':'Biobank_LCQ_B10_Recovered_NA_covid_negative', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'Biobank_LCQ_B10_Recovered', 'outcome_variable':outcome_var}
    reference_result = {'Variable':'result_stringencylimit'+stringency+'_3.0', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'result_stringencylimit'+stringency, 'outcome_variable':outcome_var}
    
    # Append row to the dataframe
    data = data.append(reference_age, ignore_index=True)
    data = data.append(reference_sex, ignore_index=True)
    data = data.append(reference_ethnicity, ignore_index=True)
    data = data.append(reference_education, ignore_index=True)
    data = data.append(reference_region, ignore_index=True)
    data = data.append(reference_covid_group, ignore_index=True)
    data = data.append(reference_IMD, ignore_index=True)
    data = data.append(reference_BMI, ignore_index=True)
    data = data.append(reference_hospitalisation, ignore_index=True)
    data = data.append(reference_comorbidities, ignore_index=True)
    data = data.append(reference_comorbidities_MH, ignore_index=True)
    data = data.append(reference_chalder, ignore_index=True)
    data = data.append(reference_PHQ4, ignore_index=True)
    data = data.append(reference_WSAS, ignore_index=True)
    data = data.append(reference_recovery, ignore_index=True)
    data = data.append(reference_result, ignore_index=True)
    
    return data


# -----------------------------------------------------------------------------
# PLOT 1 SERIES
def plot_OLS_w_conf_int_1plot(data1, x_fieldname, y_fieldname, conf_int_fieldnames, plot1_label, xlims, ylims, titlelabel, width, height, offset, y_pos_manual, color_list, fontsize, invert_axis, legend_offset):
    plt.rcParams.update({'font.size': fontsize}) # increase font size
    scalar = 1 # scalar factor to increase size of axis
    
    if y_pos_manual == 'yes':
        # introduce offset to be able to separate markers
        data1['x_manual'] = (data1['Variable_order']*scalar) 
    else:
        # introduce offset to be able to separate markers
        data1['x_manual'] = (np.arange(len(data1[x_fieldname]))*scalar)
    
    # Plot 1        
    ax = data1.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "o", color = color_list[0], s = 10, label = plot1_label, figsize=(width,height))
    error_bar1 = ax.errorbar(y = data1['x_manual'], x = data1[y_fieldname], xerr = np.array(data1[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[0])
    if invert_axis == 'yes':
        plt.gca().invert_yaxis()
    
    plt.yticks(data1['x_manual'], data1[x_fieldname]) # set labels manually 
    ax.yaxis.label.set_visible(False) # hide y axis title
    
    if len(ylims) > 0: # if y limits provided
        ax.set_ylim(ylims[0], ylims[1]) # set y axis limits
    if len(xlims) > 0: # if x limits provided
        ax.set_xlim(xlims[0], xlims[1]) # set x axis limits
    
    plt.axvline(x = 0.0, color = 'k', linestyle = 'dashed', linewidth = 0.8) # add line to show coefficient = 0
    
    ax.set_xlabel('Coefficient (units: standard deviations from mean)')
    # ax.set_xscale('log')
    ax.set_title(titlelabel)
    
    # ax.grid(b = True) # add gridlines
    # ax.get_legend().remove() # remove legend
    ax.legend(bbox_to_anchor=(0.5, legend_offset), loc = 'lower center') # move legend out of the way

    return data1

# -----------------------------------------------------------------------------
# 2 SERIES
def plot_OLS_w_conf_int_2plots(data1, data2, x_fieldname, y_fieldname, conf_int_fieldnames, plot1_label, plot2_label, xlims, ylims, titlelabel, width, height, offset, y_pos_manual, color_list, fontsize, legend_offset, invert_axis, y_tick_all_exposures, size1, size2):
    plt.rcParams.update({'font.size': fontsize}) # increase font size
        
    # Alt - try using xticks to create offset https://stackoverflow.com/questions/48157735/plot-multiple-bars-for-categorical-data
    scalar = 1 # scalar factor to increase size of axis
    
    if invert_axis == 'yes':
        offset = -1*offset
    
    if y_pos_manual == 'yes':
        # introduce offset to be able to separate markers
        data1['x_manual'] = (data1['Variable_order']*scalar) + offset 
        data2['x_manual'] = (data2['Variable_order']*scalar) - offset
    else:
        # introduce offset to be able to separate markers
        data1['x_manual'] = (np.arange(len(data1[x_fieldname]))*scalar) + offset 
        data2['x_manual'] = (np.arange(len(data2[x_fieldname]))*scalar) - offset
    
    # Plot 1        
    ax = data1.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "o", color = color_list[0], s = size1, label = plot1_label, figsize=(width,height))
    error_bar1 = ax.errorbar(y = data1['x_manual'], x = data1[y_fieldname], xerr = np.array(data1[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[0])
    # Plot 2
    ax = data2.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "D", color = color_list[1], s = size2, label = plot2_label, ax = ax)
    error_bar2 = ax.errorbar(y = data2['x_manual'], x = data2[y_fieldname], xerr = np.array(data2[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[1])
    
    if invert_axis == 'yes':
        plt.gca().invert_yaxis()
    
    # Set y labels manually
    if y_tick_all_exposures == 'yes':
        # y_var_list = data1['Variable'].append(data2['Variable']).unique()
        y_tick_string = list(dictionary['variable_order'].keys())
        y_tick_string_tidy = list(map(dictionary['variable_tidy'].get, y_tick_string))
        y_tick_string_tidy = list(dict.fromkeys(y_tick_string_tidy)) # remove duplicates
        y_tick_value = list(dictionary['variable_order'].values())
        y_tick_value = list(dict.fromkeys(y_tick_value)) # remove duplicates
        plt.yticks(y_tick_value, y_tick_string_tidy) # set labels manually 
        
    else:
        plt.yticks(data2['x_manual'] + (offset), data2[x_fieldname]) # set labels manually   
    
    ax.yaxis.label.set_visible(False) # hide y axis title
    
    if len(ylims) > 0: # if y limits provided
        ax.set_ylim(ylims[0], ylims[1]) # set y axis limits
    if len(xlims) > 0: # if x limits provided
        ax.set_xlim(xlims[0], xlims[1]) # set x axis limits
    
    plt.axvline(x = 0.0, color = 'k', linestyle = 'dashed', linewidth = 0.8) # add line to show coefficient = 0
    
    ax.set_xlabel('Coefficient (units: standard deviations from mean)')
    # ax.set_xscale('log')
    ax.set_title(titlelabel)
    
    ax.grid(b = True) # add gridlines
    # ax.get_legend().remove() # remove legend
    ax.legend(bbox_to_anchor=(0.5, legend_offset), loc = 'lower center') # move legend out of the way

    return data1, data2

# -----------------------------------------------------------------------------
# 3 SERIES
def plot_OLS_w_conf_int_3plots(data1, data2, data3, x_fieldname, y_fieldname, conf_int_fieldnames, plot1_label, plot2_label, plot3_label, xlims, ylims, titlelabel, width, height, offset, y_pos_manual, color_list, fontsize, bbox_to_anchor_vertical, invert_axis, alpha):       
    plt.rcParams.update({'font.size': fontsize}) # increase font size
    
    # Alt - try using xticks to create offset https://stackoverflow.com/questions/48157735/plot-multiple-bars-for-categorical-data
    scalar = 1 # scalar factor to increase size of axis
    
    if invert_axis == 'yes':
        offset = -1*offset
    
    if y_pos_manual == 'yes':
        # introduce offset to be able to separate markers
        data1['x_manual'] = (data1['Variable_order']*scalar) + offset 
        data2['x_manual'] = (data2['Variable_order']*scalar)
        data3['x_manual'] = (data3['Variable_order']*scalar) - offset
    else:
        # introduce offset to be able to separate markers
        data1['x_manual'] = (np.arange(len(data1[x_fieldname]))*scalar) + offset 
        data2['x_manual'] = (np.arange(len(data2[x_fieldname]))*scalar)
        data3['x_manual'] = (np.arange(len(data3[x_fieldname]))*scalar) - offset
    
    # Plot 1        
    ax = data1.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "s", color = color_list[0], s = 17, label = plot1_label, figsize=(width,height))
    error_bar1 = ax.errorbar(y = data1['x_manual'], x = data1[y_fieldname], xerr = np.array(data1[conf_int_fieldnames].transpose()), alpha = alpha, capsize = 3, label = None, fmt = 'none', color = color_list[0])
    # Plot 2
    ax = data2.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "o", color = color_list[1], s = 22, label = plot2_label, ax = ax)
    error_bar2 = ax.errorbar(y = data2['x_manual'], x = data2[y_fieldname], xerr = np.array(data2[conf_int_fieldnames].transpose()), alpha = alpha, capsize = 3, label = None, fmt = 'none', color = color_list[1])
    # Plot 3
    ax = data3.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "D", color = color_list[2], s = 15, label = plot3_label, ax = ax)
    error_bar3 = ax.errorbar(y = data3['x_manual'], x = data3[y_fieldname], xerr = np.array(data3[conf_int_fieldnames].transpose()), alpha = alpha, capsize = 3, label = None, fmt = 'none', color = color_list[2])
    
    if invert_axis == 'yes':
        plt.gca().invert_yaxis()
    
    plt.yticks(data3['x_manual'] + (offset), data3[x_fieldname]) # set labels manually 
    ax.yaxis.label.set_visible(False) # hide y axis title
    
    if len(ylims) > 0: # if y limits provided
        ax.set_ylim(ylims[0], ylims[1]) # set y axis limits
    if len(xlims) > 0: # if x limits provided
        ax.set_xlim(xlims[0], xlims[1]) # set x axis limits
    
    plt.axvline(x = 0.0, color = 'k', linestyle = 'dashed', linewidth = 0.8) # add line to show coefficient = 0
    
    ax.set_xlabel('Coefficient (units: standard deviations from mean)')
    # ax.set_xscale('log')
    ax.set_title(titlelabel)
    ax.grid(b = True) # add gridlines
    # ax.get_legend().remove() # remove legend
    ax.legend(bbox_to_anchor=(0.5, bbox_to_anchor_vertical), loc = 'lower center') # move legend out of the way

    return data1, data2, data3

# -----------------------------------------------------------------------------
# 4 SERIES
def plot_OLS_w_conf_int_4plots(data1, data2, data3, data4, x_fieldname, y_fieldname, conf_int_fieldnames, plot1_label, plot2_label, plot3_label, plot4_label, xlims, ylims, titlelabel, width, height, offset, y_pos_manual, color_list, fontsize, bbox_to_anchor_vertical, invert_axis, alpha):       
    plt.rcParams.update({'font.size': fontsize}) # increase font size
    
    # Alt - try using xticks to create offset https://stackoverflow.com/questions/48157735/plot-multiple-bars-for-categorical-data
    scalar = 1 # scalar factor to increase size of axis
    
    if invert_axis == 'yes':
        offset = -1*offset
    
    if y_pos_manual == 'yes':
        # introduce offset to be able to separate markers
        data1['x_manual'] = (data1['Variable_order']*scalar) + (1.5*offset)
        data2['x_manual'] = (data2['Variable_order']*scalar) + (0.5*offset)
        data3['x_manual'] = (data3['Variable_order']*scalar) - (0.5*offset)
        data4['x_manual'] = (data4['Variable_order']*scalar) - (1.5*offset)
    else:
        # introduce offset to be able to separate markers
        data1['x_manual'] = (np.arange(len(data1[x_fieldname]))*scalar) + (1.5*offset)
        data2['x_manual'] = (np.arange(len(data2[x_fieldname]))*scalar) + (0.5*offset)
        data3['x_manual'] = (np.arange(len(data3[x_fieldname]))*scalar) - (0.5*offset)
        data4['x_manual'] = (np.arange(len(data4[x_fieldname]))*scalar) - (1.5*offset)
    
    # Plot 1        
    ax = data1.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "s", color = color_list[0], s = 17, label = plot1_label, figsize=(width,height))
    error_bar1 = ax.errorbar(y = data1['x_manual'], x = data1[y_fieldname], xerr = np.array(data1[conf_int_fieldnames].transpose()), alpha = alpha, capsize = 3, label = None, fmt = 'none', color = color_list[0])
    # Plot 2
    ax = data2.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "o", color = color_list[1], s = 22, label = plot2_label, ax = ax)
    error_bar2 = ax.errorbar(y = data2['x_manual'], x = data2[y_fieldname], xerr = np.array(data2[conf_int_fieldnames].transpose()), alpha = alpha, capsize = 3, label = None, fmt = 'none', color = color_list[1])
    # Plot 3
    ax = data3.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "D", color = color_list[2], s = 15, label = plot3_label, ax = ax)
    error_bar3 = ax.errorbar(y = data3['x_manual'], x = data3[y_fieldname], xerr = np.array(data3[conf_int_fieldnames].transpose()), alpha = alpha, capsize = 3, label = None, fmt = 'none', color = color_list[2])
    # Plot 4
    ax = data4.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "^", color = color_list[3], s = 20, label = plot4_label, ax = ax)
    error_bar4 = ax.errorbar(y = data4['x_manual'], x = data4[y_fieldname], xerr = np.array(data4[conf_int_fieldnames].transpose()), alpha = alpha, capsize = 3, label = None, fmt = 'none', color = color_list[3])
    
    if invert_axis == 'yes':
        plt.gca().invert_yaxis()
    
    # get unique fieldnames
    var_fieldname = 'Variable'
    variable_list = data1[var_fieldname].append(data2[var_fieldname])
    variable_list = variable_list.append(data3[var_fieldname])
    variable_list = variable_list.append(data4[var_fieldname])
    
    variable_list_unique = pd.Series(variable_list.unique())
    variable_list_unique_fieldname = variable_list_unique.map(dictionary['variable_tidy'])
    variable_list_unique_value = variable_list_unique.map(dictionary['variable_order'])
    
    plt.yticks(variable_list_unique_value, variable_list_unique_fieldname) # set labels manually 
    # plt.yticks(data3['x_manual'] + (0.5*offset), data3[x_fieldname]) # set labels manually 
    ax.yaxis.label.set_visible(False) # hide y axis title
    
    if len(ylims) > 0: # if y limits provided
        ax.set_ylim(ylims[0], ylims[1]) # set y axis limits
    if len(xlims) > 0: # if x limits provided
        ax.set_xlim(xlims[0], xlims[1]) # set x axis limits
    
    plt.axvline(x = 0.0, color = 'k', linestyle = 'dashed', linewidth = 0.8) # add line to show coefficient = 0
    
    ax.set_xlabel('Coefficient (units: standard deviations from mean)')
    # ax.set_xscale('log')
    ax.set_title(titlelabel)
    ax.grid(b = True) # add gridlines
    # ax.get_legend().remove() # remove legend
    ax.legend(bbox_to_anchor=(0.5, bbox_to_anchor_vertical), loc = 'lower center') # move legend out of the way

    return data1, data2, data3, data4


# -----------------------------------------------------------------------------
# 5 SERIES
def plot_OLS_w_conf_int_5plots(data1, data2, data3, data4, data5, x_fieldname, y_fieldname, conf_int_fieldnames, plot1_label, plot2_label, plot3_label, plot4_label, plot5_label, xlims, ylims, titlelabel, width, height, offset, y_pos_manual, color_list, fontsize, bbox_to_anchor_vertical, invert_axis, alpha):       
    plt.rcParams.update({'font.size': fontsize}) # increase font size
    
    # Alt - try using xticks to create offset https://stackoverflow.com/questions/48157735/plot-multiple-bars-for-categorical-data
    scalar = 1 # scalar factor to increase size of axis
    
    if invert_axis == 'yes':
        offset = -1*offset
    
    if y_pos_manual == 'yes':
        # introduce offset to be able to separate markers
        data1['x_manual'] = (data1['Variable_order']*scalar) + (2*offset)
        data2['x_manual'] = (data2['Variable_order']*scalar) + (1*offset)
        data3['x_manual'] = (data3['Variable_order']*scalar) - (0*offset)
        data4['x_manual'] = (data4['Variable_order']*scalar) - (1*offset)
        data5['x_manual'] = (data5['Variable_order']*scalar) - (2*offset)
    else:
        # introduce offset to be able to separate markers
        data1['x_manual'] = (np.arange(len(data1[x_fieldname]))*scalar) + (2*offset)
        data2['x_manual'] = (np.arange(len(data2[x_fieldname]))*scalar) + (1*offset)
        data3['x_manual'] = (np.arange(len(data3[x_fieldname]))*scalar) - (0*offset)
        data4['x_manual'] = (np.arange(len(data4[x_fieldname]))*scalar) - (1*offset)
        data5['x_manual'] = (np.arange(len(data5[x_fieldname]))*scalar) - (2*offset)
    
    # Plot 1        
    ax = data1.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "s", color = color_list[0], s = 17, label = plot1_label, figsize=(width,height))
    error_bar1 = ax.errorbar(y = data1['x_manual'], x = data1[y_fieldname], xerr = np.array(data1[conf_int_fieldnames].transpose()), alpha = alpha, capsize = 3, label = None, fmt = 'none', color = color_list[0])
    # Plot 2
    ax = data2.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "o", color = color_list[1], s = 22, label = plot2_label, ax = ax)
    error_bar2 = ax.errorbar(y = data2['x_manual'], x = data2[y_fieldname], xerr = np.array(data2[conf_int_fieldnames].transpose()), alpha = alpha, capsize = 3, label = None, fmt = 'none', color = color_list[1])
    # Plot 3
    ax = data3.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "D", color = color_list[2], s = 15, label = plot3_label, ax = ax)
    error_bar3 = ax.errorbar(y = data3['x_manual'], x = data3[y_fieldname], xerr = np.array(data3[conf_int_fieldnames].transpose()), alpha = alpha, capsize = 3, label = None, fmt = 'none', color = color_list[2])
    # Plot 4
    ax = data4.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "^", color = color_list[3], s = 20, label = plot4_label, ax = ax)
    error_bar4 = ax.errorbar(y = data4['x_manual'], x = data4[y_fieldname], xerr = np.array(data4[conf_int_fieldnames].transpose()), alpha = alpha, capsize = 3, label = None, fmt = 'none', color = color_list[3])
    # Plot 5
    ax = data5.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "v", color = color_list[4], s = 20, label = plot5_label, ax = ax)
    error_bar5 = ax.errorbar(y = data5['x_manual'], x = data5[y_fieldname], xerr = np.array(data5[conf_int_fieldnames].transpose()), alpha = alpha, capsize = 3, label = None, fmt = 'none', color = color_list[4])
    
    if invert_axis == 'yes':
        plt.gca().invert_yaxis()
    
    plt.yticks(data3['x_manual'] + (0*offset), data3[x_fieldname]) # set labels manually 
    ax.yaxis.label.set_visible(False) # hide y axis title
    
    if len(ylims) > 0: # if y limits provided
        ax.set_ylim(ylims[0], ylims[1]) # set y axis limits
    if len(xlims) > 0: # if x limits provided
        ax.set_xlim(xlims[0], xlims[1]) # set x axis limits
    
    plt.axvline(x = 0.0, color = 'k', linestyle = 'dashed', linewidth = 0.8) # add line to show coefficient = 0
    
    ax.set_xlabel('Coefficient (units: standard deviations from mean)')
    # ax.set_xscale('log')
    ax.set_title(titlelabel)
    ax.grid(b = True) # add gridlines
    # ax.get_legend().remove() # remove legend
    ax.legend(bbox_to_anchor=(0.5, bbox_to_anchor_vertical), loc = 'lower center') # move legend out of the way

    return data1, data2, data3, data4, data5


# -----------------------------------------------------------------------------
# Add reference categories to results table
def add_reference_values_posthoc(data, outcome_var):
    """ Function to add reference category values into OLS results table """
    # Add coeff = 0 for reference variables
    reference_age = {'Variable':'Combined_Age_2021_grouped_decades_4: 50-60', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'Combined_Age_2021_grouped_decades', 'outcome_variable':outcome_var}
    reference_sex = {'Variable':'ZOE_demogs_sex_Female', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'ZOE_demogs_sex', 'outcome_variable':outcome_var}
    reference_ethnicity = {'Variable':'Combined_EthnicityCategory_White', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'Combined_EthnicityCategory', 'outcome_variable':outcome_var}
    reference_education = {'Variable':'educationLevel_cat4_2. Undergraduate degree', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'educationLevel_cat4', 'outcome_variable':outcome_var}
    reference_region = {'Variable':'Region_London', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'Region', 'outcome_variable':outcome_var}
    reference_IMD = {'Variable':'Combined_IMD_Quintile_3.0', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'Combined_IMD_Quintile', 'outcome_variable':outcome_var}
    reference_BMI = {'Variable':'Combined_BMI_cat5_2: 18.5-25', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'Combined_BMI_cat5', 'outcome_variable':outcome_var}
    reference_comorbidities = {'Variable':'ZOE_conditions_condition_count_cat3_0 conditions', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'ZOE_conditions_condition_count_cat3', 'outcome_variable':outcome_var}
    reference_comorbidities_MH = {'Variable':'ZOE_mentalhealth_condition_cat4_0 conditions', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'ZOE_mentalhealth_condition_cat4', 'outcome_variable':outcome_var}
    
    
    reference_chalder = {'Variable':'q_chalderFatigue_cat2_1. 0-28, below threshold', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'q_chalderFatigue_cat2', 'outcome_variable':outcome_var}
    reference_PHQ4 = {'Variable':'q_PHQ4_cat4_1. 0-2, below threshold', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'q_PHQ4_cat4', 'outcome_variable':outcome_var}
    reference_WSAS = {'Variable':'q_WSAS_cat4_1. 0-9, below threshold', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'q_WSAS_cat4', 'outcome_variable':outcome_var}
    reference_recovery = {'Variable':'Biobank_LCQ_B10_Recovered_NA_covid_negative', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'Biobank_LCQ_B10_Recovered', 'outcome_variable':outcome_var}
    
    reference_result_round1 = {'Variable':'round1_'+'result_stringencylimit'+stringency+'_3.0', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'round1_'+'result_stringencylimit'+stringency, 'outcome_variable':outcome_var}
    reference_hospitalisation_round1 = {'Variable':'round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency+'_0.0', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency, 'outcome_variable':outcome_var}
    reference_covid_group_round1 = {'Variable':'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N1: Negative COVID-19, asymptomatic (healthy control)', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'round1_'+'symptomduration_grouped1_stringencylimit'+stringency, 'outcome_variable':outcome_var}
    
    reference_result_round2 = {'Variable':'round2_'+'result_stringencylimit'+stringency+'_3.0', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'round2_'+'result_stringencylimit'+stringency, 'outcome_variable':outcome_var}
    reference_hospitalisation_round2 = {'Variable':'round2_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency+'_0.0', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'round2_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency, 'outcome_variable':outcome_var}
    reference_covid_group_round2 = {'Variable':'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N1: Negative COVID-19, asymptomatic (healthy control)', 'coeff':0, 'coeff_with_p_value_stars':'0', 'var_exposure':'round2_'+'symptomduration_grouped1_stringencylimit'+stringency, 'outcome_variable':outcome_var}
    
    # Append row to the dataframe
    if 'Combined_Age_2021_grouped_decades' in data['var_exposure'].unique():
        data = data.append(reference_age, ignore_index=True)
    if 'ZOE_demogs_sex' in data['var_exposure'].unique():
        data = data.append(reference_sex, ignore_index=True)
    if 'Combined_EthnicityCategory' in data['var_exposure'].unique():
        data = data.append(reference_ethnicity, ignore_index=True)
    if 'educationLevel_cat4' in data['var_exposure'].unique():
        data = data.append(reference_education, ignore_index=True)
    if 'Region' in data['var_exposure'].unique():
        data = data.append(reference_region, ignore_index=True)
    if 'Combined_IMD_Quintile' in data['var_exposure'].unique():
        data = data.append(reference_IMD, ignore_index=True)
    if 'Combined_BMI_cat5' in data['var_exposure'].unique():
        data = data.append(reference_BMI, ignore_index=True)
    
    if 'ZOE_conditions_condition_count_cat3' in data['var_exposure'].unique():
        data = data.append(reference_comorbidities, ignore_index=True)
    if 'ZOE_mentalhealth_condition_cat4' in data['var_exposure'].unique():
        data = data.append(reference_comorbidities_MH, ignore_index=True)
        
    if 'q_chalderFatigue_cat2' in data['var_exposure'].unique():
        data = data.append(reference_chalder, ignore_index=True)
    if 'q_PHQ4_cat4' in data['var_exposure'].unique():
        data = data.append(reference_PHQ4, ignore_index=True)
    if 'q_WSAS_cat4' in data['var_exposure'].unique():
        data = data.append(reference_WSAS, ignore_index=True)
    if 'Biobank_LCQ_B10_Recovered' in data['var_exposure'].unique():
        data = data.append(reference_recovery, ignore_index=True)
    
    if 'round1_'+'result_stringencylimit'+stringency in data['var_exposure'].unique():
        data = data.append(reference_result_round1, ignore_index=True)
    if 'round1_'+'symptomduration_grouped1_stringencylimit'+stringency in data['var_exposure'].unique():
        data = data.append(reference_covid_group_round1, ignore_index=True)
    if 'round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency in data['var_exposure'].unique():
        data = data.append(reference_hospitalisation_round1, ignore_index=True)   
    
    if 'round2_'+'result_stringencylimit'+stringency in data['var_exposure'].unique():
        data = data.append(reference_result_round2, ignore_index=True)
    if 'round2_'+'symptomduration_grouped1_stringencylimit'+stringency in data['var_exposure'].unique():
        data = data.append(reference_covid_group_round2, ignore_index=True)
    if 'round2_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency in data['var_exposure'].unique():
        data = data.append(reference_hospitalisation_round2, ignore_index=True)
    
    # Create tidy input and outcome variable name columns
    data['Variable_tidy'] = data['Variable'].map(dictionary['variable_tidy'])
    data['outcome_variable_tidy'] = data['outcome_variable'].map(dictionary['outcome_variable_tidy'])
    
    # Add specified variable order 
    data['Variable_order'] = data['Variable'].map(dictionary['variable_order'])
    data['outcome_variable_order'] = data['outcome_variable'].map(dictionary['outcome_variable_order'])
    
    
    return data

# -----------------------------------------------------------------------------
# 1 SERIES AS A BAR CHART
def bar_plot_OLS_w_conf_int_1plot(data1, x_fieldname, y_fieldname, conf_int_fieldnames, plot1_label, xlims, ylims, titlelabel, width, height, y_pos_manual, color_list, fontsize, legend_offset, bar_width, horORver, plot_error, alpha_bar, alpha_error):
    scalar = 1
    # Alt - try using xticks to create offset https://stackoverflow.com/questions/48157735/plot-multiple-bars-for-categorical-data    
    if y_pos_manual == 'yes':
        # introduce offset to be able to separate markers
        data1['x_manual'] = (data1['Variable_order']*scalar)
    else:
        # introduce offset to be able to separate markers
        data1['x_manual'] = (np.arange(len(data1[x_fieldname]))*scalar) 
    
    # Extract arrays needed for plot
    data1_xpos = np.array(data1['x_manual']) 
    data1_yval = np.array(data1[y_fieldname])
        
    xticks = np.array(data1['x_manual'])
    xlabel = np.array(data1[x_fieldname])
    
    if horORver == 'h':
        # Horizontal bar plot
        fig, ax = plt.subplots(figsize=(width,height))
        ax.grid(b = True) # add gridlines
        plt.rcParams.update({'font.size': fontsize}) # increase font size
        if plot_error == 'yes':
            ax.barh(y = data1_xpos, width = data1_yval, height = bar_width, 
                            alpha = alpha_bar, color = color_list[0], label = plot1_label,
                            xerr = np.array(data1[conf_int_fieldnames].transpose()),
                            ecolor = 'black', # color_list[0], #'grey', 
                            capsize = 3, error_kw = {'alpha':alpha_error})
        else:
            ax.barh(y = data1_xpos, width = data1_yval, height = bar_width, 
                    alpha = 1, color = color_list[0], label = plot1_label)
        plt.gca().invert_yaxis()
        ax.set_yticks(xticks)
        plt.yticks(wrap = True, 
                   # rotation=-90
                   )
        ax.set_yticklabels(xlabel)
        
        # ax.get_legend().remove() # remove legend
        ax.legend(bbox_to_anchor=(0.5, legend_offset), loc = 'lower center') # move legend out of the way
        
        if len(ylims) > 0: # if y limits provided
            ax.set_ylim(ylims[0], ylims[1]) # set y axis limits
        if len(xlims) > 0: # if x limits provided
            ax.set_xlim(xlims[0], xlims[1]) # set x axis limits
        
        plt.axvline(x = 0.0, color = 'k', linestyle = 'dashed', linewidth = 0.8) # add line to show coefficient = 0
        # plt.axhline(y = 54.0, color = 'grey', linestyle = 'dashed', linewidth = 1.4) # add line to breaks between models
        # plt.axhline(y = 67.5, color = 'grey', linestyle = 'dashed', linewidth = 1.4) # add line to breaks between models
        
        ax.set_ylabel('Exposure variable')
        ax.set_xlabel('Coefficient (units: standard deviations from mean)')
        ax.set_title(titlelabel)
        
    elif horORver == 'v':
        # Vertical bar plot
        fig, ax = plt.subplots(figsize=(height,width))
        ax.grid(b = True) # add gridlines
        plt.rcParams.update({'font.size': fontsize}) # increase font size
        
        if plot_error == 'yes':
            ax.bar(x = data1_xpos, width = bar_width, height = data1_yval, 
                            alpha = alpha_bar, color = color_list[0], label = plot1_label,
                            yerr = np.array(data1[conf_int_fieldnames].transpose()),
                            ecolor = 'black', #color_list[0], 
                            capsize = 3, error_kw = {'alpha':alpha_error})
        else:
            ax.bar(x = data1_xpos, width = bar_width, height = data1_yval, 
                   alpha = 1, color = color_list[0], label = plot1_label)
        ax.set_xticks(xticks)
        plt.xticks(rotation=90)
        ax.set_xticklabels(xlabel)
        
        # ax.get_legend().remove() # remove legend
        ax.legend(bbox_to_anchor=(legend_offset, 0.5), loc = 'lower center') # move legend out of the way
        
        if len(xlims) > 0: # if y limits provided
            ax.set_ylim(xlims[0], xlims[1]) # set y axis limits
        if len(ylims) > 0: # if x limits provided
            ax.set_xlim(ylims[0], ylims[1]) # set x axis limits
        
        plt.axhline(y = 0.0, color = 'k', linestyle = 'dashed', linewidth = 0.8) # add line to show coefficient = 0
        
        ax.set_xlabel('Exposure variable')
        ax.set_ylabel('Coefficient (units: standard deviations from mean)')
        ax.set_title(titlelabel)
    
    return data1

# -----------------------------------------------------------------------------
# 2 SERIES AS A BAR CHART
def bar_plot_OLS_w_conf_int_2plots(data1, data2, x_fieldname, y_fieldname, conf_int_fieldnames, plot1_label, plot2_label, xlims, ylims, titlelabel, width, height, offset, y_pos_manual, color_list, fontsize, legend_offset, bar_width, horORver, plot_error, alpha_bar, alpha_error):
    scalar = 1
    # Alt - try using xticks to create offset https://stackoverflow.com/questions/48157735/plot-multiple-bars-for-categorical-data    
    if y_pos_manual == 'yes':
        # introduce offset to be able to separate markers
        data1['x_manual'] = (data1['Variable_order']*scalar) - offset 
        data2['x_manual'] = (data2['Variable_order']*scalar) + offset
    else:
        # introduce offset to be able to separate markers
        data1['x_manual'] = (np.arange(len(data1[x_fieldname]))*scalar) - offset 
        data2['x_manual'] = (np.arange(len(data2[x_fieldname]))*scalar) + offset
    
    # Extract arrays needed for plot
    data1_xpos = np.array(data1['x_manual']) 
    data1_yval = np.array(data1[y_fieldname])
    
    data2_xpos = np.array(data2['x_manual']) 
    data2_yval = np.array(data2[y_fieldname])
    
    xticks = np.array(data2['x_manual'] - (offset))
    xlabel = np.array(data2[x_fieldname])
    
    if horORver == 'h':
        # Horizontal bar plot
        fig, ax = plt.subplots(figsize=(width,height))
        ax.grid(b = True) # add gridlines
        plt.rcParams.update({'font.size': fontsize}) # increase font size
        if plot_error == 'yes':
            ax.barh(y = data1_xpos, width = data1_yval, height = bar_width, 
                            alpha = alpha_bar, color = color_list[0], label = plot1_label,
                            xerr = np.array(data1[conf_int_fieldnames].transpose()),
                            ecolor = 'black', # color_list[0], #'grey', 
                            capsize = 3, error_kw = {'alpha':alpha_error})
            ax.barh(y = data2_xpos, width = data2_yval, height = bar_width,
                            alpha = alpha_bar, color = color_list[1], label = plot2_label,
                            xerr = np.array(data2[conf_int_fieldnames].transpose()),
                            ecolor = 'black', #color_list[1], #'grey', 
                            capsize = 3, error_kw = {'alpha':alpha_error})
        else:
            ax.barh(y = data1_xpos, width = data1_yval, height = bar_width, 
                    alpha = 1, color = color_list[0], label = plot1_label)
            ax.barh(y = data2_xpos, width = data2_yval,height = bar_width,
                    alpha = 1, color = color_list[1], label = plot2_label)
        plt.gca().invert_yaxis()
        ax.set_yticks(xticks)
        plt.yticks(wrap = True, 
                   # rotation=-90
                   )
        ax.set_yticklabels(xlabel)
        
        # ax.get_legend().remove() # remove legend
        ax.legend(bbox_to_anchor=(0.5, legend_offset), loc = 'lower center') # move legend out of the way
        
        if len(ylims) > 0: # if y limits provided
            ax.set_ylim(ylims[0], ylims[1]) # set y axis limits
        if len(xlims) > 0: # if x limits provided
            ax.set_xlim(xlims[0], xlims[1]) # set x axis limits
        
        plt.axvline(x = 0.0, color = 'k', linestyle = 'dashed', linewidth = 0.8) # add line to show coefficient = 0
        # plt.axhline(y = 54.0, color = 'grey', linestyle = 'dashed', linewidth = 1.4) # add line to breaks between models
        # plt.axhline(y = 67.5, color = 'grey', linestyle = 'dashed', linewidth = 1.4) # add line to breaks between models
        
        ax.set_ylabel('Exposure variable')
        ax.set_xlabel('Coefficient (units: standard deviations from mean)')
        ax.set_title(titlelabel)
        
    elif horORver == 'v':
        # Vertical bar plot
        fig, ax = plt.subplots(figsize=(height,width))
        ax.grid(b = True) # add gridlines
        plt.rcParams.update({'font.size': fontsize}) # increase font size
        
        if plot_error == 'yes':
            ax.bar(x = data1_xpos, width = bar_width, height = data1_yval, 
                            alpha = alpha_bar, color = color_list[0], label = plot1_label,
                            yerr = np.array(data1[conf_int_fieldnames].transpose()),
                            ecolor = 'black', # color_list[0], 
                            capsize = 3, error_kw = {'alpha':alpha_error})
            ax.bar(x = data2_xpos, width = bar_width, 
                   height = data2_yval,
                            alpha = alpha_bar,
                            color = color_list[1], label = plot2_label,
                            yerr = np.array(data2[conf_int_fieldnames].transpose()),
                            ecolor = 'black', # color_list[1], 
                            capsize = 3, error_kw = {'alpha':alpha_error})
        else:
            ax.bar(x = data1_xpos, width = bar_width, height = data1_yval, 
                   alpha = 1, color = color_list[0], label = plot1_label)
            ax.bar(x = data2_xpos, width = bar_width, height = data2_yval,
                   alpha = 1, color = color_list[1], label = plot2_label,)
        ax.set_xticks(xticks)
        plt.xticks(rotation=90)
        ax.set_xticklabels(xlabel)
        
        # ax.get_legend().remove() # remove legend
        ax.legend(bbox_to_anchor=(legend_offset, 0.5), loc = 'lower center') # move legend out of the way
        
        if len(xlims) > 0: # if y limits provided
            ax.set_ylim(xlims[0], xlims[1]) # set y axis limits
        if len(ylims) > 0: # if x limits provided
            ax.set_xlim(ylims[0], ylims[1]) # set x axis limits
        
        plt.axhline(y = 0.0, color = 'k', linestyle = 'dashed', linewidth = 0.8) # add line to show coefficient = 0
        
        ax.set_xlabel('Exposure variable')
        ax.set_ylabel('Coefficient (units: standard deviations from mean)')
        ax.set_title(titlelabel)
    
    return data1, data2


# -----------------------------------------------------------------------------
# 2 SERIES AS A BAR CHART
def bar_plot_OLS_w_conf_int_3plots(data1, data2, data3, x_fieldname, y_fieldname, conf_int_fieldnames, plot1_label, plot2_label, plot3_label, xlims, ylims, titlelabel, width, height, offset, y_pos_manual, color_list, fontsize, legend_offset, bar_width, horORver, plot_error, alpha_bar, alpha_error):
    scalar = 1
    # Alt - try using xticks to create offset https://stackoverflow.com/questions/48157735/plot-multiple-bars-for-categorical-data    
    if y_pos_manual == 'yes':
        # introduce offset to be able to separate markers
        data1['x_manual'] = (data1['Variable_order']*scalar) - offset 
        data2['x_manual'] = (data2['Variable_order']*scalar)
        data3['x_manual'] = (data3['Variable_order']*scalar) + offset
    else:
        # introduce offset to be able to separate markers
        data1['x_manual'] = (np.arange(len(data1[x_fieldname]))*scalar) - offset 
        data2['x_manual'] = (np.arange(len(data2[x_fieldname]))*scalar)
        data3['x_manual'] = (np.arange(len(data3[x_fieldname]))*scalar) + offset
    
    # Extract arrays needed for plot
    data1_xpos = np.array(data1['x_manual']) 
    data1_yval = np.array(data1[y_fieldname])
    data2_xpos = np.array(data2['x_manual']) 
    data2_yval = np.array(data2[y_fieldname])
    data3_xpos = np.array(data3['x_manual']) 
    data3_yval = np.array(data3[y_fieldname])
    
    xticks = np.array(data3['x_manual'] - (offset))
    xlabel = np.array(data3[x_fieldname])
    
    if horORver == 'h':
        # Horizontal bar plot
        fig, ax = plt.subplots(figsize=(width,height))
        ax.grid(b = True) # add gridlines
        plt.rcParams.update({'font.size': fontsize}) # increase font size
        if plot_error == 'yes':
            ax.barh(y = data1_xpos, width = data1_yval, height = bar_width, 
                            alpha = alpha_bar, color = color_list[0], label = plot1_label,
                            xerr = np.array(data1[conf_int_fieldnames].transpose()),
                            ecolor = 'black', #color_list[0], #'grey', 
                            capsize = 3, error_kw = {'alpha':alpha_error})
            ax.barh(y = data2_xpos, width = data2_yval, height = bar_width,
                            alpha = alpha_bar, color = color_list[1], label = plot2_label,
                            xerr = np.array(data2[conf_int_fieldnames].transpose()),
                            ecolor = 'black', #color_list[1], #'grey', 
                            capsize = 3, error_kw = {'alpha':alpha_error})
            ax.barh(y = data3_xpos, width = data3_yval, height = bar_width,
                            alpha = alpha_bar, color = color_list[2], label = plot3_label,
                            xerr = np.array(data3[conf_int_fieldnames].transpose()),
                            ecolor = 'black', # color_list[2], #'grey', 
                            capsize = 3, error_kw = {'alpha':alpha_error})
        else:
            ax.barh(y = data1_xpos, width = data1_yval, height = bar_width, 
                    alpha = 1, color = color_list[0], label = plot1_label)
            ax.barh(y = data2_xpos, width = data2_yval,height = bar_width,
                    alpha = 1, color = color_list[1], label = plot2_label)
            ax.barh(y = data3_xpos, width = data3_yval,height = bar_width,
                    alpha = 1, color = color_list[2], label = plot3_label)
        plt.gca().invert_yaxis()
        ax.set_yticks(xticks)
        plt.yticks(wrap = True, 
                   # rotation=-90
                   )
        ax.set_yticklabels(xlabel)
        
        ax.legend(bbox_to_anchor=(0.5, legend_offset), loc = 'lower center') # move legend out of the way
        
        if len(ylims) > 0: # if y limits provided
            ax.set_ylim(ylims[0], ylims[1]) # set y axis limits
        if len(xlims) > 0: # if x limits provided
            ax.set_xlim(xlims[0], xlims[1]) # set x axis limits
        
        plt.axvline(x = 0.0, color = 'k', linestyle = 'dashed', linewidth = 0.8) # add line to show coefficient = 0
        # plt.axhline(y = 54.0, color = 'grey', linestyle = 'dashed', linewidth = 1.4) # add line to breaks between models
        # plt.axhline(y = 67.5, color = 'grey', linestyle = 'dashed', linewidth = 1.4) # add line to breaks between models
        
        ax.set_ylabel('Exposure variable')
        ax.set_xlabel('Coefficient (units: standard deviations from mean)')
        ax.set_title(titlelabel)
        
    elif horORver == 'v':
        # Vertical bar plot
        fig, ax = plt.subplots(figsize=(height,width))
        ax.grid(b = True) # add gridlines
        plt.rcParams.update({'font.size': fontsize}) # increase font size
        
        if plot_error == 'yes':
            ax.bar(x = data1_xpos, width = bar_width, height = data1_yval, 
                            alpha = alpha_bar, color = color_list[0], label = plot1_label,
                            yerr = np.array(data1[conf_int_fieldnames].transpose()),
                            ecolor = 'black', # color_list[0], 
                            capsize = 3, error_kw = {'alpha':alpha_error})
            ax.bar(x = data2_xpos, width = bar_width, height = data2_yval,
                            alpha = alpha_bar,color = color_list[1], label = plot2_label,
                            yerr = np.array(data2[conf_int_fieldnames].transpose()),
                            ecolor = 'black', # color_list[1], 
                            capsize = 3, error_kw = {'alpha':alpha_error})
            ax.bar(x = data3_xpos, width = bar_width, height = data3_yval,
                            alpha = alpha_bar,color = color_list[2], label = plot3_label,
                            yerr = np.array(data3[conf_int_fieldnames].transpose()),
                            ecolor = 'black', # color_list[2], 
                            capsize = 3, error_kw = {'alpha':alpha_error})
        else:
            ax.bar(x = data1_xpos, width = bar_width, height = data1_yval, 
                   alpha = 1, color = color_list[0], label = plot1_label)
            ax.bar(x = data2_xpos, width = bar_width, height = data2_yval,
                   alpha = 1, color = color_list[1], label = plot2_label,)
            ax.bar(x = data3_xpos, width = bar_width, height = data3_yval,
                   alpha = 1, color = color_list[2], label = plot3_label,)
        ax.set_xticks(xticks)
        plt.xticks(rotation=90)
        ax.set_xticklabels(xlabel)
        
        # ax.get_legend().remove() # remove legend
        ax.legend(bbox_to_anchor=(legend_offset, 0.5), loc = 'lower center') # move legend out of the way
        
        if len(xlims) > 0: # if y limits provided
            ax.set_ylim(xlims[0], xlims[1]) # set y axis limits
        if len(ylims) > 0: # if x limits provided
            ax.set_xlim(ylims[0], ylims[1]) # set x axis limits
        
        plt.axhline(y = 0.0, color = 'k', linestyle = 'dashed', linewidth = 0.8) # add line to show coefficient = 0
        
        ax.set_xlabel('Exposure variable')
        ax.set_ylabel('Coefficient (units: standard deviations from mean)')
        ax.set_title(titlelabel)
    
    return data1, data2, data3


#%% Load data
# -----------------------------------------------------------------------------
# Round 1 and 2 datasets, which are filtered for invited and participated only. Contains all dummy variables. No imputation
data_cognitron_round1_invited = pd.read_csv(r'cognitron_round1_allinvited_afterparticipationmodelling'+'_stringencylimit'+stringency+'.csv')
data_cognitron_round2_invited = pd.read_csv(r'cognitron_round2_allinvited_afterparticipationmodelling'+'_stringencylimit'+stringency+'.csv')
data_cognitron_round1and2_invited = pd.read_csv(r'cognitron_round1and2_allinvited_afterparticipationmodelling'+'_stringencylimit'+stringency+'.csv')

data_cognitron_round1_invited_cols =data_cognitron_round1_invited.columns.to_list()
data_cognitron_round2_invited_cols =data_cognitron_round2_invited.columns.to_list()
data_cognitron_round1and2_invited_cols =data_cognitron_round1and2_invited.columns.to_list()

# -----------------------------------------------------------------------------
# Inverse probability weights of participation and completion 
weights_join = pd.read_csv(r'cognitron_participation_IPW'+'_stringencylimit'+stringency+'.csv')


#%% Pre-Processing
# -----------------------------------------------------------------------------
# Dictionary of tidy names for input variables
dictionary = {}
dictionary['variable_tidy'] = {'Combined_IMD_Quintile_Data not available': 'IMD: Data not available',
                               'Combined_IMD_Quintile_1.0': 'IMD: Quintile 1 (most 20% deprived areas)',
                               'Combined_IMD_Quintile_2.0': 'IMD: Quintile 2',
                               'Combined_IMD_Quintile_3.0': 'IMD: Quintile 3 (reference)',
                               'Combined_IMD_Quintile_4.0': 'IMD: Quintile 4',
                               'Combined_IMD_Quintile_5.0': 'IMD: Quintile 5 (least 20% deprived areas)',
                               
                               'PRISMA7_score':'PRISMA-7 (units: +1 score, increasing frailty)',
                               
                               'PRISMA7_score_Data not available':'PRISMA-7: Data not available',
                               'PRISMA7_score_0.0':'PRISMA-7: Score 0/7',
                               'PRISMA7_score_1.0':'PRISMA-7: Score 1/7',
                               'PRISMA7_score_2.0':'PRISMA-7: Score 2/7',
                               'PRISMA7_score_3.0':'PRISMA-7: Score 3/7',
                               'PRISMA7_score_4.0':'PRISMA-7: Score 4/7',
                               'PRISMA7_score_5.0':'PRISMA-7: Score 5/7',
                               'PRISMA7_score_6.0':'PRISMA-7: Score 6/7',
                               
                               'InvitationCohort_Data not available':'Recruitment cohort: Data not available',
                               'InvitationCohort_1. October-November 2020 COVID-19 invitation': 'Recruitment cohort: 1. October-November 2020 (reference)',
                               'InvitationCohort_2. May 2021 COVID-19 invitation':'Recruitment cohort: 2a. May 2021 positive COVID-19 top-up',
                               'InvitationCohort_3. May 2021 Healthy control COVID-19 invitation':'Recruitment cohort: 2b. May 2021 negative COVID-19 controls',
                               
                               # Test result
                               # Round 1
                               'round1_'+'result_stringencylimit'+stringency+'_Data not available':'SARS-CoV-2 test result: Data not available',
                               'round1_'+'result_stringencylimit'+stringency+'_3.0':'SARS-CoV-2 test result: Negative (reference)',
                               'round1_'+'result_stringencylimit'+stringency+'_4.0':'SARS-CoV-2 test result: Positive',
                               # Round 2
                               'round2_'+'result_stringencylimit'+stringency+'_Data not available':'SARS-CoV-2 test result: Data not available',
                               'round2_'+'result_stringencylimit'+stringency+'_3.0':'SARS-CoV-2 test result: Negative (reference)',
                               'round2_'+'result_stringencylimit'+stringency+'_4.0':'SARS-CoV-2 test result: Positive',
                               
                               # Covid group
                               # Round 1
                               'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_Data not available':'COVID-19 group: Data not available',
                               'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N1: Negative COVID-19, asymptomatic (healthy control)':'COVID-19 group: SARS-CoV-2 Negative, Asymptomatic (reference)',
                               'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N2&3: Negative COVID-19, 0-4 weeks':'COVID-19 group: SARS-CoV-2 Negative, Symptom duration < 4 weeks',
                               'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N4&5: Negative COVID-19, 4-12 weeks':'COVID-19 group: SARS-CoV-2 Negative, Symptom duration 4-12 weeks',
                               'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N6: Negative COVID-19, 12+ weeks':'COVID-19 group: SARS-CoV-2 Negative, Symptom duration ≥ 12 weeks',
                               'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P1: Positive COVID-19, asymptomatic':'COVID-19 group: SARS-CoV-2 Positive, Asymptomatic',
                               'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P2&3: Positive COVID-19, 0-4 weeks':'COVID-19 group: SARS-CoV-2 Positive, Symptom duration < 4 weeks',
                               'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P4&5: Positive COVID-19, 4-12 weeks':'COVID-19 group: SARS-CoV-2 Positive, Symptom duration 4-12 weeks',
                               'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P6: Positive COVID-19, 12+ weeks':'COVID-19 group: SARS-CoV-2 Positive, Symptom duration ≥ 12 weeks',
                               # Round 2
                               'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_Data not available':'COVID-19 group: Data not available',
                               'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N1: Negative COVID-19, asymptomatic (healthy control)':'COVID-19 group: SARS-CoV-2 Negative, Asymptomatic (reference)',
                               'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N2&3: Negative COVID-19, 0-4 weeks':'COVID-19 group: SARS-CoV-2 Negative, Symptom duration < 4 weeks',
                               'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N4&5: Negative COVID-19, 4-12 weeks':'COVID-19 group: SARS-CoV-2 Negative, Symptom duration 4-12 weeks',
                               'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N6: Negative COVID-19, 12+ weeks':'COVID-19 group: SARS-CoV-2 Negative, Symptom duration ≥ 12 weeks',
                               'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P1: Positive COVID-19, asymptomatic':'COVID-19 group: SARS-CoV-2 Positive, Asymptomatic',
                               'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P2&3: Positive COVID-19, 0-4 weeks':'COVID-19 group: SARS-CoV-2 Positive, Symptom duration < 4 weeks',
                               'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P4&5: Positive COVID-19, 4-12 weeks':'COVID-19 group: SARS-CoV-2 Positive, Symptom duration 4-12 weeks',
                               'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P6: Positive COVID-19, 12+ weeks':'COVID-19 group: SARS-CoV-2 Positive, Symptom duration ≥ 12 weeks',
                               
                               'Combined_Age_2021_grouped_decades_Data not available':'Age: Data not available',
                               'Combined_Age_2021_grouped_decades_1: 0-30': 'Age: 18-30',
                               'Combined_Age_2021_grouped_decades_2: 30-40': 'Age: 30-40',
                               'Combined_Age_2021_grouped_decades_3: 40-50': 'Age: 40-50',
                               'Combined_Age_2021_grouped_decades_4: 50-60': 'Age: 50-60 (reference)',
                               'Combined_Age_2021_grouped_decades_5: 60-70': 'Age: 60-70',
                               'Combined_Age_2021_grouped_decades_6: 70-80': 'Age: 70-80',
                               'Combined_Age_2021_grouped_decades_7: 80+': 'Age: ≥ 80',
                               
                               'ZOE_demogs_sex_Data not available':'Sex: Data not available',
                               'ZOE_demogs_sex_Female': 'Sex: Female (reference)',
                               'ZOE_demogs_sex_Male':'Sex: Male',
                               
                               'Combined_EthnicityCategory_Data not available': 'Ethnicity: Data not available',
                               'Combined_EthnicityCategory_White': 'Ethnicity: White (reference)',
                               'Combined_EthnicityCategory_Any other ethnic group':'Ethnicity: Other',
                               'Combined_EthnicityCategory_Asian or Asian British':'Ethnicity: Asian/Asian British',
                               'Combined_EthnicityCategory_Black or Black British':'Ethnicity: Black/Black British',
                               'Combined_EthnicityCategory_Mixed or multiple ethnic groups':'Ethnicity: Mixed/Multiple',
                               
                               'Region_Data not available':'Region: Data not available',
                               'Region_London':'Region: London (reference)',
                               'Region_East Midlands':'Region: East Midlands',
                               'Region_East of England':'Region: East of England',
                               'Region_North East':'Region: North East',
                               'Region_North West':'Region: North West',
                               'Region_Northern Ireland':'Region: Northern Ireland',
                               'Region_Scotland':'Region: Scotland',
                               'Region_South East':'Region: South East',
                               'Region_South West':'Region: South West',
                               'Region_Wales':'Region: Wales',
                               'Region_West Midlands':'Region: West Midlands',
                               'Region_Yorkshire and The Humber':'Region: Yorkshire and The Humber',
                               
                               'educationLevel_cat4_Data not available':'Education level: Data not available',
                               'educationLevel_cat4_0. Other/ prefer not to say':'Education level: Other/Prefer not to say',
                               'educationLevel_cat4_1. Less than degree level':'Education level: Less than undergraduate degree',
                               'educationLevel_cat4_2. Undergraduate degree':'Education level: Undergraduate degree (reference)',
                               'educationLevel_cat4_3. Postgraduate degree or higher':'Education level: Postgraduate degree or higher',
                               
                               'ZOE_conditions_condition_count_cat3_Data not available':'Physical health conditions: Data not available',
                               'ZOE_conditions_condition_count_cat3_0 conditions':'Physical health conditions: None (reference)',
                               'ZOE_conditions_condition_count_cat3_1 condition':'Physical health conditions: One',
                               'ZOE_conditions_condition_count_cat3_2+ conditions':'Physical health conditions: Two or more',
                               
                               # Number of mental health conditions (from ZOE questionnaire)
                               'ZOE_mentalhealth_condition_cat4_0 conditions': 'Mental health conditions: None (reference)',
                               'ZOE_mentalhealth_condition_cat4_1 condition': 'Mental health conditions: One',
                               'ZOE_mentalhealth_condition_cat4_2 conditions': 'Mental health conditions: Two',
                               'ZOE_mentalhealth_condition_cat4_3+ conditions': 'Mental health conditions: Three or more',
                               'ZOE_mentalhealth_condition_cat4_NaN': 'Mental health conditions: Unknown (non-response)',
                               
                               'Combined_BMI_cat5_Data not available':'BMI: Data not available',
                               'Combined_BMI_cat5_1: 0-18.5':'BMI: < 18.5 kg/m^2',
                               'Combined_BMI_cat5_2: 18.5-25':'BMI: 18.5-25 kg/m^2 (reference)',
                               'Combined_BMI_cat5_3: 25-30':'BMI: 25-30 kg/m^2',
                               'Combined_BMI_cat5_4: 30+':'BMI: ≥ 30 kg/m^2',
                               'Combined_BMI_cat5_NaN':'BMI: Unknown',
                               
                               # Hospitalised 
                               # Round 1
                               'round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency+'_Data not available':'Presentation to hospital during symptomatic period: Data not available',
                               'round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency+'_0.0':'Presentation to hospital during symptomatic period: No (reference)',
                               'round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency+'_1.0':'Presentation to hospital during symptomatic period: Yes',
                               # Round 2
                               'round2_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency+'_Data not available':'Presentation to hospital during symptomatic period: Data not available',
                               'round2_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency+'_0.0':'Presentation to hospital during symptomatic period: No (reference)',
                               'round2_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency+'_1.0':'Presentation to hospital during symptomatic period: Yes',
                               
                                'Biobank_LCQ_B10_Recovered_Data not available': 'Self-perceived COVID-19 recovery: Data not available',
                                'Biobank_LCQ_B10_Recovered_NA_covid_negative': 'Self-perceived COVID-19 recovery: N/A, no self-reported COVID-19 (reference)',
                                'Biobank_LCQ_B10_Recovered_yes': 'Self-perceived COVID-19 recovery: Recovered',
                                'Biobank_LCQ_B10_Recovered_no': 'Self-perceived COVID-19 recovery: Not recovered',
                                'Biobank_LCQ_B10_Recovered_Unknown': 'Self-perceived COVID-19 recovery: Unknown',
                               
                               'q_chalderFatigue_cat2_Data not available':'Chalder Fatigue Scale: Data not available',
                               'q_chalderFatigue_cat2_1. 0-28, below threshold':'Chalder Fatigue Scale: Below threshold, 0-28 (reference)',
                               'q_chalderFatigue_cat2_2. 29-33, above threshold':'Chalder Fatigue Scale: Above threshold, 29-33',
                               
                               'q_PHQ4_cat4_Data not available':'PHQ-4 Scale: Data not available',
                               'q_PHQ4_cat4_1. 0-2, below threshold':'PHQ-4 Scale: Below threshold, 0-2 (reference)',
                               'q_PHQ4_cat4_2. 3-5, mild':'PHQ-4 Scale: Mild, 3-5',
                               'q_PHQ4_cat4_3. 6-8, moderate':'PHQ-4 Scale: Moderate, 6-8',
                               'q_PHQ4_cat4_4. 9-12, severe':'PHQ-4 Scale: Severe, 9-12',
                               
                               'q_WSAS_cat4_Data not available':'WSAS Scale: Data not available',
                               'q_WSAS_cat4_1. 0-9, below threshold':'WSAS Scale: Below threshold, 0-9 (reference)',
                               'q_WSAS_cat4_2. 10-20, mild':'WSAS Scale: Mild, 10-20',
                               'q_WSAS_cat4_3. 21-40, moderate to severe':'WSAS Scale: Moderate to severe, 21-40',
                               
                               'WeeksBetween_Cognitron_SymptomORTest_Data not available':'Weeks between cognitive assessment and symptom start/test date: Data not available',
                               'WeeksBetween_Cognitron_SymptomORTest_':'Weeks between cognitive assessment and symptom start/test date',
                               
                               'WaveAt_SymptomORTest_2020 Q1 Jan-Mar': 'Symptom start date/Test date: Q1 Jan-Mar 2020',
                               'WaveAt_SymptomORTest_2020 Q2 Apr-Jun': 'Symptom start date/Test date: Q2 Apr-Jun 2020',
                               'WaveAt_SymptomORTest_2020 Q3 Jul-Sep': 'Symptom start date/Test date: Q3 Jul-Sep 2020',
                               'WaveAt_SymptomORTest_2020 Q4 Oct-Dec': 'Symptom start date/Test date: Q4 Oct-Dec 2020',
                               'WaveAt_SymptomORTest_2021 Q1 Jan-Mar': 'Symptom start date/Test date: Q1 Jan-Mar 2021',
                               'WaveAt_SymptomORTest_2021 Q2 Apr-Jun': 'Symptom start date/Test date: Q2 Apr-Jun 2021',
                               'WaveAt_SymptomORTest_2021 Q3 Jul-Sep': 'Symptom start date/Test date: Q3 Jul-Sep 2021',
                               'WaveAt_SymptomORTest_2021 Q4 Oct-Dec': 'Symptom start date/Test date: Q4 Oct-Dec 2021',
                               'WaveAt_SymptomORTest_2022 Q1 Jan-Mar': 'Symptom start date/Test date: Q1 Jan-Mar 2022',
                               'WaveAt_SymptomORTest_2022 Q2 Apr-Jun': 'Symptom start date/Test date: Q2 Apr-Jun 2022',
                               'WaveAt_SymptomORTest_2022 Q3 Jul-Sep': 'Symptom start date/Test date: Q3 Jul-Sep 2022',
                               'WaveAt_SymptomORTest_2022 Q4 Oct-Dec': 'Symptom start date/Test date: Q4 Oct-Dec 2022',
                               
                               }

# -----------------------------------------------------------------------------
# Specify relative ordering of input variables
dictionary['variable_order'] = {#
                               'Combined_Age_2021_grouped_decades_1: 0-30': 0,
                               'Combined_Age_2021_grouped_decades_2: 30-40': 1,
                               'Combined_Age_2021_grouped_decades_3: 40-50': 2,
                               'Combined_Age_2021_grouped_decades_4: 50-60': 3,
                               'Combined_Age_2021_grouped_decades_5: 60-70': 4,
                               'Combined_Age_2021_grouped_decades_6: 70-80': 5,
                               'Combined_Age_2021_grouped_decades_7: 80+': 6,
                               
                               'ZOE_demogs_sex_Female': 8,
                               'ZOE_demogs_sex_Male':9,
                               
                               'Combined_EthnicityCategory_White':15,
                               'Combined_EthnicityCategory_Any other ethnic group':14,
                               'Combined_EthnicityCategory_Asian or Asian British':11,
                               'Combined_EthnicityCategory_Black or Black British':12,
                               'Combined_EthnicityCategory_Mixed or multiple ethnic groups':13,
                               
                               'educationLevel_cat4_0. Other/ prefer not to say':17,
                               'educationLevel_cat4_1. Less than degree level':18,
                               'educationLevel_cat4_2. Undergraduate degree':19,
                               'educationLevel_cat4_3. Postgraduate degree or higher':20,
                               
                               'Combined_IMD_Quintile_1.0': 22,
                               'Combined_IMD_Quintile_2.0': 23,
                               'Combined_IMD_Quintile_3.0': 24,
                               'Combined_IMD_Quintile_4.0': 25,
                               'Combined_IMD_Quintile_5.0': 26,
                               
                               'Region_London':30,
                               'Region_East Midlands':28,
                               'Region_East of England':29,
                               'Region_North East':31,
                               'Region_North West':32,
                               'Region_Northern Ireland':33,
                               'Region_Scotland':34,
                               'Region_South East':35,
                               'Region_South West':36,
                               'Region_Wales':37,
                               'Region_West Midlands':38,
                               'Region_Yorkshire and The Humber':39,
                               
                               
                               'PRISMA7_score':41,
                               
                               'Combined_BMI_cat5_1: 0-18.5':43,
                               'Combined_BMI_cat5_2: 18.5-25':44,
                               'Combined_BMI_cat5_3: 25-30':45,
                               'Combined_BMI_cat5_4: 30+':46,
                               
                               'ZOE_conditions_condition_count_cat3_0 conditions':48,
                               'ZOE_conditions_condition_count_cat3_1 condition':49,
                               'ZOE_conditions_condition_count_cat3_2+ conditions':50,
                               
                               # Number of mental health conditions (from ZOE questionnaire)
                               'ZOE_mentalhealth_condition_cat4_0 conditions': 52,
                               'ZOE_mentalhealth_condition_cat4_1 condition': 53,
                               'ZOE_mentalhealth_condition_cat4_2 conditions': 54,
                               'ZOE_mentalhealth_condition_cat4_3+ conditions': 55,
                               'ZOE_mentalhealth_condition_cat4_NaN': 56,
                               
                               
                               # Round 1
                               'round1_'+'result_stringencylimit'+stringency+'_3.0':58,
                               'round1_'+'result_stringencylimit'+stringency+'_4.0':59,
                               # Round 2
                               'round2_'+'result_stringencylimit'+stringency+'_3.0':58,
                               'round2_'+'result_stringencylimit'+stringency+'_4.0':59,
                               
                               # Round 1
                               'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N1: Negative COVID-19, asymptomatic (healthy control)':61,
                               'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N2&3: Negative COVID-19, 0-4 weeks':62,
                               'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N4&5: Negative COVID-19, 4-12 weeks':63,
                               'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N6: Negative COVID-19, 12+ weeks':64,
                               'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P1: Positive COVID-19, asymptomatic':65.5,
                               'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P2&3: Positive COVID-19, 0-4 weeks':66.5,
                               'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P4&5: Positive COVID-19, 4-12 weeks':67.5,
                               'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P6: Positive COVID-19, 12+ weeks':68.5,
                               
                               # Round 2
                               'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N1: Negative COVID-19, asymptomatic (healthy control)':61,
                               'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N2&3: Negative COVID-19, 0-4 weeks':62,
                               'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N4&5: Negative COVID-19, 4-12 weeks':63,
                               'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N6: Negative COVID-19, 12+ weeks':64,
                               'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P1: Positive COVID-19, asymptomatic':65.5,
                               'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P2&3: Positive COVID-19, 0-4 weeks':66.5,
                               'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P4&5: Positive COVID-19, 4-12 weeks':67.5,
                               'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P6: Positive COVID-19, 12+ weeks':68.5,
                               
                               # Round 1
                               'round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency+'_0.0':70.5,
                               'round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency+'_1.0':71.5,
                               # Round 2
                               'round2_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency+'_0.0':70.5,
                               'round2_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency+'_1.0':71.5,
                                                              
                               'q_PHQ4_cat4_1. 0-2, below threshold':73.5,
                               'q_PHQ4_cat4_2. 3-5, mild':74.5,
                               'q_PHQ4_cat4_3. 6-8, moderate':75.5,
                               'q_PHQ4_cat4_4. 9-12, severe':76.5,
                               
                               'q_chalderFatigue_cat2_1. 0-28, below threshold':78.5,
                               'q_chalderFatigue_cat2_2. 29-33, above threshold':79.5,
                               
                               'q_WSAS_cat4_1. 0-9, below threshold':81.5,
                               'q_WSAS_cat4_2. 10-20, mild':82.5,
                               'q_WSAS_cat4_3. 21-40, moderate to severe':83.5,
                               
                                'Biobank_LCQ_B10_Recovered_NA_covid_negative': 85.5,
                                'Biobank_LCQ_B10_Recovered_yes': 86.5,
                                'Biobank_LCQ_B10_Recovered_no': 87.5,
                                'Biobank_LCQ_B10_Recovered_Unknown': 88.5,
                               
                               }

dictionary['variable_order_reverse'] = dict((v, k) for k, v in dictionary['variable_order'].items())

# -----------------------------------------------------------------------------
# Dictionary of tidy names for outcome variables
dictionary['outcome_variable_tidy'] = {#
                    'C1_R1model_accuracy_zscore':'Composite: Principal component 1',
                    'C2_R1model_accuracy_zscore':'Composite: Principal component 2',
                    'C3_R1model_accuracy_zscore': 'Composite: Principal component 3',
                    'C4_R1model_accuracy_zscore': 'Composite: Principal component 4',
                    'primary_accuracy_01_cube_transform_zscore_processed': 'Immediate memory (words)',
                    'primary_accuracy_02_cube_transform_zscore_processed': 'Immediate memory (objects)',
                    'primary_accuracy_03_log_transform_zscore_processed': 'Motor control',
                    'primary_accuracy_04_zscore_processed': '2-D mental manipulations',
                    'primary_accuracy_05_cube_transform_zscore_processed': 'Target detection',
                    'primary_accuracy_06_zscore_processed': 'Spatial span',
                    'primary_accuracy_07_square_transform_zscore_processed': 'Tower of London',
                    'primary_accuracy_08_zscore_processed': 'Verbal analogies',
                    'primary_accuracy_09_cube_transform_zscore_processed': 'Delayed memory (words)',
                    'primary_accuracy_10_cube_transform_zscore_processed': 'Delayed memory (objects)',
                    'primary_accuracy_11_squareroot_transform_zscore_processed': 'Paired-associate learning',
                    'primary_accuracy_12_cube_transform_zscore_processed': 'Cognitive reflection test',
                    
                    'C1_R1model_rt_average_zscore':'Composite: Principal component 1',
                    'C2_R1model_rt_average_zscore':'Composite: Principal component 2',
                    'C3_R1model_rt_average_zscore': 'Composite: Principal component 3',
                    'C4_R1model_rt_average_zscore': 'Composite: Principal component 4',
                    'reaction_time_average_01_log_transform_zscore_processed': 'Immediate memory (words)',
                    'reaction_time_average_02_log_transform_zscore_processed': 'Immediate memory (objects)',
                    'reaction_time_average_03_log_transform_zscore_processed': 'Motor control',
                    'reaction_time_average_04_log_transform_zscore_processed': '2-D mental manipulations',
                    'reaction_time_average_05_zscore_processed': 'Target detection',
                    'reaction_time_average_06_log_transform_zscore_processed': 'Spatial span',
                    'reaction_time_average_07_squareroot_transform_zscore_processed': 'Tower of London',
                    'reaction_time_average_08_squareroot_transform_zscore_processed': 'Verbal analogies',
                    'reaction_time_average_09_log_transform_zscore_processed': 'Delayed memory (words)',
                    'reaction_time_average_10_log_transform_zscore_processed': 'Delayed memory (objects)',
                    'reaction_time_average_11_log_transform_zscore_processed': 'Paired-associate learning',
                    'reaction_time_average_12_squareroot_transform_zscore_processed': 'Cognitive reflection test',
                    
                    'C1_R1model_rt_variation_zscore':'Composite: Principal component 1',
                    'C2_R1model_rt_variation_zscore':'Composite: Principal component 2',
                    'C3_R1model_rt_variation_zscore': 'Composite: Principal component 3',
                    'C4_R1model_rt_variation_zscore': 'Composite: Principal component 4',
                    'reaction_time_variation_01_log_transform_zscore_processed': 'Immediate memory (words)',
                    'reaction_time_variation_02_log_transform_zscore_processed': 'Immediate memory (objects)',
                    'reaction_time_variation_03_log_transform_zscore_processed': 'Motor control',
                    'reaction_time_variation_04_log_transform_zscore_processed': '2-D mental manipulations',
                    'reaction_time_variation_05_zscore_processed': 'Target detection',
                    'reaction_time_variation_06_log_transform_zscore_processed': 'Spatial span',
                    'reaction_time_variation_07_log_transform_zscore_processed': 'Tower of London',
                    'reaction_time_variation_08_log_transform_zscore_processed': 'Verbal analogies',
                    'reaction_time_variation_09_log_transform_zscore_processed': 'Delayed memory (words)',
                    'reaction_time_variation_10_log_transform_zscore_processed': 'Delayed memory (objects)',
                    'reaction_time_variation_11_log_transform_zscore_processed': 'Paired-associate learning',
                    'reaction_time_variation_12_log_transform_zscore_processed': 'Cognitive reflection test',
                               }

# -----------------------------------------------------------------------------
# Specify relative ordering of outcome variables
dictionary['outcome_variable_order'] = {#
                    'C1_R1model_accuracy_zscore':0,
                    'C2_R1model_accuracy_zscore':1,
                    'C3_R1model_accuracy_zscore': 2,
                    'C4_R1model_accuracy_zscore': 3,
                    'primary_accuracy_01_cube_transform_zscore_processed': 4,
                    'primary_accuracy_02_cube_transform_zscore_processed': 5,
                    'primary_accuracy_03_log_transform_zscore_processed': 15,
                    'primary_accuracy_04_zscore_processed': 8,
                    'primary_accuracy_05_cube_transform_zscore_processed': 9,
                    'primary_accuracy_06_zscore_processed': 10,
                    'primary_accuracy_07_square_transform_zscore_processed': 11,
                    'primary_accuracy_08_zscore_processed': 12,
                    'primary_accuracy_09_cube_transform_zscore_processed': 6,
                    'primary_accuracy_10_cube_transform_zscore_processed': 7,
                    'primary_accuracy_11_squareroot_transform_zscore_processed': 13,
                    'primary_accuracy_12_cube_transform_zscore_processed': 14,
                    
                    'C1_R1model_rt_average_zscore':0.1,
                    'C2_R1model_rt_average_zscore':1.1,
                    'C3_R1model_rt_average_zscore':2.1,
                    'C4_R1model_rt_average_zscore': 3.1,
                    'reaction_time_average_01_log_transform_zscore_processed': 4.1,
                    'reaction_time_average_02_log_transform_zscore_processed': 5.1,
                    'reaction_time_average_03_log_transform_zscore_processed': 15.1,
                    'reaction_time_average_04_log_transform_zscore_processed': 8.1,
                    'reaction_time_average_05_zscore_processed': 9.1,
                    'reaction_time_average_06_log_transform_zscore_processed': 10.1,
                    'reaction_time_average_07_squareroot_transform_zscore_processed': 11.1,
                    'reaction_time_average_08_squareroot_transform_zscore_processed': 12.1,
                    'reaction_time_average_09_log_transform_zscore_processed': 6.1,
                    'reaction_time_average_10_log_transform_zscore_processed': 7.1,
                    'reaction_time_average_11_log_transform_zscore_processed': 13.1,
                    'reaction_time_average_12_squareroot_transform_zscore_processed': 14.1,
                    
                    'C1_R1model_rt_variation_zscore':0.2,
                    'C2_R1model_rt_variation_zscore':1.2,
                    'C3_R1model_rt_variation_zscore':2.2,
                    'C4_R1model_rt_variation_zscore':3.2,
                    'reaction_time_variation_01_log_transform_zscore_processed': 4.2,
                    'reaction_time_variation_02_log_transform_zscore_processed':5.2,
                    'reaction_time_variation_03_log_transform_zscore_processed': 15.2,
                    'reaction_time_variation_04_log_transform_zscore_processed': 8.2,
                    'reaction_time_variation_05_zscore_processed': 9.2,
                    'reaction_time_variation_06_log_transform_zscore_processed': 10.2,
                    'reaction_time_variation_07_log_transform_zscore_processed': 11.2,
                    'reaction_time_variation_08_log_transform_zscore_processed': 12.2,
                    'reaction_time_variation_09_log_transform_zscore_processed': 6.2,
                    'reaction_time_variation_10_log_transform_zscore_processed': 7.2,
                    'reaction_time_variation_11_log_transform_zscore_processed':13.2,
                    'reaction_time_variation_12_log_transform_zscore_processed': 14.2,
                               }
dictionary['outcome_variable_order_reverse'] = dict((v, k) for k, v in dictionary['outcome_variable_order'].items())


# -----------------------------------------------------------------------------
# Add time between cognitive assessment completion date and test date
# Round 1 dataset
# Use symptom start date, or if empty (for asymptomatic), use test date
data_cognitron_round1_invited['DaysBetween_Cognitron_SymptomORTest'] = (pd.to_datetime(data_cognitron_round1_invited['endDate'], errors = 'coerce') - pd.to_datetime(data_cognitron_round1_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency], errors = 'coerce')).dt.days
# If covid group is asymptomatic, use test date
data_cognitron_round1_invited.loc[(data_cognitron_round1_invited['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')), 'DaysBetween_Cognitron_SymptomORTest'] = (pd.to_datetime(data_cognitron_round1_invited['endDate'], errors = 'coerce') - pd.to_datetime(data_cognitron_round1_invited['round1_'+'date_effective_test_stringencylimit'+stringency], errors = 'coerce')).dt.days
# Convert to weeks
data_cognitron_round1_invited['WeeksBetween_Cognitron_SymptomORTest'] = (data_cognitron_round1_invited['DaysBetween_Cognitron_SymptomORTest']/7).apply(np.floor) # round down

test = data_cognitron_round1_invited[['endDate',
                                      'round1_'+'symptom_start_date_estimate_stringencylimit'+stringency,
                                      'round1_'+'date_effective_test_stringencylimit'+stringency,
                                      'round1_'+'symptomduration_grouped1_stringencylimit'+stringency,
                                      'DaysBetween_Cognitron_SymptomORTest', 
                                      'WeeksBetween_Cognitron_SymptomORTest']]

# Round 2 dataset
# Use symptom start date, or if empty (for asymptomatic), use test date
data_cognitron_round2_invited['DaysBetween_Cognitron_SymptomORTest'] = (pd.to_datetime(data_cognitron_round2_invited['endDate'], errors = 'coerce') - pd.to_datetime(data_cognitron_round2_invited['round2_'+'symptom_start_date_estimate_stringencylimit'+stringency], errors = 'coerce')).dt.days
# If covid group is asymptomatic, use test date
data_cognitron_round2_invited.loc[(data_cognitron_round2_invited['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')), 'DaysBetween_Cognitron_SymptomORTest'] = (pd.to_datetime(data_cognitron_round2_invited['endDate'], errors = 'coerce') - pd.to_datetime(data_cognitron_round2_invited['round2_'+'date_effective_test_stringencylimit'+stringency], errors = 'coerce')).dt.days
# Convert to weeks
data_cognitron_round2_invited['WeeksBetween_Cognitron_SymptomORTest'] = (data_cognitron_round2_invited['DaysBetween_Cognitron_SymptomORTest']/7).apply(np.floor) # round down

test2 = data_cognitron_round2_invited[['endDate',
                                       'round2_'+'symptom_start_date_estimate_stringencylimit'+stringency,
                                       'round2_'+'date_effective_test_stringencylimit'+stringency,
                                       'round2_'+'symptomduration_grouped1_stringencylimit'+stringency,
                                       'DaysBetween_Cognitron_SymptomORTest', 
                                       'WeeksBetween_Cognitron_SymptomORTest']]

# Round 1 and 2 dataset
# Use symptom start date, or if empty (for asymptomatic), use test date
data_cognitron_round1and2_invited['DaysBetween_Cognitron_SymptomORTest'] = (pd.to_datetime(data_cognitron_round1and2_invited['endDate'], errors = 'coerce') - pd.to_datetime(data_cognitron_round1and2_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency], errors = 'coerce')).dt.days
# If covid group is asymptomatic, use test date
data_cognitron_round1and2_invited.loc[(data_cognitron_round1and2_invited['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')), 'DaysBetween_Cognitron_SymptomORTest'] = (pd.to_datetime(data_cognitron_round1and2_invited['endDate'], errors = 'coerce') - pd.to_datetime(data_cognitron_round1and2_invited['round1_'+'date_effective_test_stringencylimit'+stringency], errors = 'coerce')).dt.days
# Convert to weeks
data_cognitron_round1and2_invited['WeeksBetween_Cognitron_SymptomORTest'] = (data_cognitron_round1and2_invited['DaysBetween_Cognitron_SymptomORTest']/7).apply(np.floor) # round down

test3 = data_cognitron_round1and2_invited[['endDate',
                                      'round1_'+'symptom_start_date_estimate_stringencylimit'+stringency,
                                      'round1_'+'date_effective_test_stringencylimit'+stringency,
                                      'round1_'+'symptomduration_grouped1_stringencylimit'+stringency,
                                      'DaysBetween_Cognitron_SymptomORTest', 
                                      'WeeksBetween_Cognitron_SymptomORTest']]



# -----------------------------------------------------------------------------
# Add year quarter at time of symptom start/test date
# Round 1 dataset
# Use symptom start date, or if empty (for asymptomatic), use test date
data_cognitron_round1_invited.loc[(data_cognitron_round1_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] < '2020-04-01'), 'WaveAt_SymptomORTest'] = '2020 Q1 Jan-Mar'
data_cognitron_round1_invited.loc[(data_cognitron_round1_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] >= '2020-04-01') & (data_cognitron_round1_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] < '2020-07-01'), 'WaveAt_SymptomORTest'] = '2020 Q2 Apr-Jun'
data_cognitron_round1_invited.loc[(data_cognitron_round1_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] >= '2020-07-01') & (data_cognitron_round1_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] < '2020-10-01'), 'WaveAt_SymptomORTest'] = '2020 Q3 Jul-Sep'
data_cognitron_round1_invited.loc[(data_cognitron_round1_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] >= '2020-10-01') & (data_cognitron_round1_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] < '2021-01-01'), 'WaveAt_SymptomORTest'] = '2020 Q4 Oct-Dec'
data_cognitron_round1_invited.loc[(data_cognitron_round1_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] >= '2021-01-01') & (data_cognitron_round1_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] < '2021-04-01'), 'WaveAt_SymptomORTest'] = '2021 Q1 Jan-Mar'
data_cognitron_round1_invited.loc[(data_cognitron_round1_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] >= '2021-04-01') & (data_cognitron_round1_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] < '2021-07-01'), 'WaveAt_SymptomORTest'] = '2021 Q2 Apr-Jun'
data_cognitron_round1_invited.loc[(data_cognitron_round1_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] >= '2021-07-01') & (data_cognitron_round1_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] < '2021-10-01'), 'WaveAt_SymptomORTest'] = '2021 Q3 Jul-Sep'
data_cognitron_round1_invited.loc[(data_cognitron_round1_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] >= '2021-10-01') & (data_cognitron_round1_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] < '2022-01-01'), 'WaveAt_SymptomORTest'] = '2021 Q4 Oct-Dec'
data_cognitron_round1_invited.loc[(data_cognitron_round1_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] >= '2022-01-01') & (data_cognitron_round1_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] < '2022-04-01'), 'WaveAt_SymptomORTest'] = '2022 Q1 Jan-Mar'
data_cognitron_round1_invited.loc[(data_cognitron_round1_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] >= '2022-04-01') & (data_cognitron_round1_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] < '2022-07-01'), 'WaveAt_SymptomORTest'] = '2022 Q2 Apr-Jun'

# If covid group is asymptomatic, use test date
data_cognitron_round1_invited.loc[(data_cognitron_round1_invited['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round1_invited['round1_'+'date_effective_test_stringencylimit'+stringency] < '2020-04-01'), 'WaveAt_SymptomORTest'] = '2020 Q1 Jan-Mar'
data_cognitron_round1_invited.loc[(data_cognitron_round1_invited['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round1_invited['round1_'+'date_effective_test_stringencylimit'+stringency] >= '2020-04-01') & (data_cognitron_round1_invited['round1_'+'date_effective_test_stringencylimit'+stringency] < '2020-07-01'), 'WaveAt_SymptomORTest'] = '2020 Q2 Apr-Jun'
data_cognitron_round1_invited.loc[(data_cognitron_round1_invited['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round1_invited['round1_'+'date_effective_test_stringencylimit'+stringency] >= '2020-07-01') & (data_cognitron_round1_invited['round1_'+'date_effective_test_stringencylimit'+stringency] < '2020-10-01'), 'WaveAt_SymptomORTest'] = '2020 Q3 Jul-Sep'
data_cognitron_round1_invited.loc[(data_cognitron_round1_invited['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round1_invited['round1_'+'date_effective_test_stringencylimit'+stringency] >= '2020-10-01') & (data_cognitron_round1_invited['round1_'+'date_effective_test_stringencylimit'+stringency] < '2021-01-01'), 'WaveAt_SymptomORTest'] = '2020 Q4 Oct-Dec'
data_cognitron_round1_invited.loc[(data_cognitron_round1_invited['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round1_invited['round1_'+'date_effective_test_stringencylimit'+stringency] >= '2021-01-01') & (data_cognitron_round1_invited['round1_'+'date_effective_test_stringencylimit'+stringency] < '2021-04-01'), 'WaveAt_SymptomORTest'] = '2021 Q1 Jan-Mar'
data_cognitron_round1_invited.loc[(data_cognitron_round1_invited['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round1_invited['round1_'+'date_effective_test_stringencylimit'+stringency] >= '2021-04-01') & (data_cognitron_round1_invited['round1_'+'date_effective_test_stringencylimit'+stringency] < '2021-07-01'), 'WaveAt_SymptomORTest'] = '2021 Q2 Apr-Jun'
data_cognitron_round1_invited.loc[(data_cognitron_round1_invited['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round1_invited['round1_'+'date_effective_test_stringencylimit'+stringency] >= '2021-07-01') & (data_cognitron_round1_invited['round1_'+'date_effective_test_stringencylimit'+stringency] < '2021-10-01'), 'WaveAt_SymptomORTest'] = '2021 Q3 Jul-Sep'
data_cognitron_round1_invited.loc[(data_cognitron_round1_invited['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round1_invited['round1_'+'date_effective_test_stringencylimit'+stringency] >= '2021-10-01') & (data_cognitron_round1_invited['round1_'+'date_effective_test_stringencylimit'+stringency] < '2022-01-01'), 'WaveAt_SymptomORTest'] = '2021 Q4 Oct-Dec'
data_cognitron_round1_invited.loc[(data_cognitron_round1_invited['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round1_invited['round1_'+'date_effective_test_stringencylimit'+stringency] >= '2022-01-01') & (data_cognitron_round1_invited['round1_'+'date_effective_test_stringencylimit'+stringency] < '2022-04-01'), 'WaveAt_SymptomORTest'] = '2022 Q1 Jan-Mar'
data_cognitron_round1_invited.loc[(data_cognitron_round1_invited['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round1_invited['round1_'+'date_effective_test_stringencylimit'+stringency] >= '2022-04-01') & (data_cognitron_round1_invited['round1_'+'date_effective_test_stringencylimit'+stringency] < '2022-07-01'), 'WaveAt_SymptomORTest'] = '2022 Q2 Apr-Jun'


# Round 2 dataset
# Use symptom start date, or if empty (for asymptomatic), use test date
data_cognitron_round2_invited.loc[(data_cognitron_round2_invited['round2_'+'symptom_start_date_estimate_stringencylimit'+stringency] < '2020-04-01'), 'WaveAt_SymptomORTest'] = '2020 Q1 Jan-Mar'
data_cognitron_round2_invited.loc[(data_cognitron_round2_invited['round2_'+'symptom_start_date_estimate_stringencylimit'+stringency] >= '2020-04-01') & (data_cognitron_round2_invited['round2_'+'symptom_start_date_estimate_stringencylimit'+stringency] < '2020-07-01'), 'WaveAt_SymptomORTest'] = '2020 Q2 Apr-Jun'
data_cognitron_round2_invited.loc[(data_cognitron_round2_invited['round2_'+'symptom_start_date_estimate_stringencylimit'+stringency] >= '2020-07-01') & (data_cognitron_round2_invited['round2_'+'symptom_start_date_estimate_stringencylimit'+stringency] < '2020-10-01'), 'WaveAt_SymptomORTest'] = '2020 Q3 Jul-Sep'
data_cognitron_round2_invited.loc[(data_cognitron_round2_invited['round2_'+'symptom_start_date_estimate_stringencylimit'+stringency] >= '2020-10-01') & (data_cognitron_round2_invited['round2_'+'symptom_start_date_estimate_stringencylimit'+stringency] < '2021-01-01'), 'WaveAt_SymptomORTest'] = '2020 Q4 Oct-Dec'
data_cognitron_round2_invited.loc[(data_cognitron_round2_invited['round2_'+'symptom_start_date_estimate_stringencylimit'+stringency] >= '2021-01-01') & (data_cognitron_round2_invited['round2_'+'symptom_start_date_estimate_stringencylimit'+stringency] < '2021-04-01'), 'WaveAt_SymptomORTest'] = '2021 Q1 Jan-Mar'
data_cognitron_round2_invited.loc[(data_cognitron_round2_invited['round2_'+'symptom_start_date_estimate_stringencylimit'+stringency] >= '2021-04-01') & (data_cognitron_round2_invited['round2_'+'symptom_start_date_estimate_stringencylimit'+stringency] < '2021-07-01'), 'WaveAt_SymptomORTest'] = '2021 Q2 Apr-Jun'
data_cognitron_round2_invited.loc[(data_cognitron_round2_invited['round2_'+'symptom_start_date_estimate_stringencylimit'+stringency] >= '2021-07-01') & (data_cognitron_round2_invited['round2_'+'symptom_start_date_estimate_stringencylimit'+stringency] < '2021-10-01'), 'WaveAt_SymptomORTest'] = '2021 Q3 Jul-Sep'
data_cognitron_round2_invited.loc[(data_cognitron_round2_invited['round2_'+'symptom_start_date_estimate_stringencylimit'+stringency] >= '2021-10-01') & (data_cognitron_round2_invited['round2_'+'symptom_start_date_estimate_stringencylimit'+stringency] < '2022-01-01'), 'WaveAt_SymptomORTest'] = '2021 Q4 Oct-Dec'
data_cognitron_round2_invited.loc[(data_cognitron_round2_invited['round2_'+'symptom_start_date_estimate_stringencylimit'+stringency] >= '2022-01-01') & (data_cognitron_round2_invited['round2_'+'symptom_start_date_estimate_stringencylimit'+stringency] < '2022-04-01'), 'WaveAt_SymptomORTest'] = '2022 Q1 Jan-Mar'
data_cognitron_round2_invited.loc[(data_cognitron_round2_invited['round2_'+'symptom_start_date_estimate_stringencylimit'+stringency] >= '2022-04-01') & (data_cognitron_round2_invited['round2_'+'symptom_start_date_estimate_stringencylimit'+stringency] < '2022-07-01'), 'WaveAt_SymptomORTest'] = '2022 Q2 Apr-Jun'

# If covid group is asymptomatic, use test date
data_cognitron_round2_invited.loc[(data_cognitron_round2_invited['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round2_invited['round2_'+'date_effective_test_stringencylimit'+stringency] < '2020-04-01'), 'WaveAt_SymptomORTest'] = '2020 Q1 Jan-Mar'
data_cognitron_round2_invited.loc[(data_cognitron_round2_invited['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round2_invited['round2_'+'date_effective_test_stringencylimit'+stringency] >= '2020-04-01') & (data_cognitron_round2_invited['round2_'+'date_effective_test_stringencylimit'+stringency] < '2020-07-01'), 'WaveAt_SymptomORTest'] = '2020 Q2 Apr-Jun'
data_cognitron_round2_invited.loc[(data_cognitron_round2_invited['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round2_invited['round2_'+'date_effective_test_stringencylimit'+stringency] >= '2020-07-01') & (data_cognitron_round2_invited['round2_'+'date_effective_test_stringencylimit'+stringency] < '2020-10-01'), 'WaveAt_SymptomORTest'] = '2020 Q3 Jul-Sep'
data_cognitron_round2_invited.loc[(data_cognitron_round2_invited['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round2_invited['round2_'+'date_effective_test_stringencylimit'+stringency] >= '2020-10-01') & (data_cognitron_round2_invited['round2_'+'date_effective_test_stringencylimit'+stringency] < '2021-01-01'), 'WaveAt_SymptomORTest'] = '2020 Q4 Oct-Dec'
data_cognitron_round2_invited.loc[(data_cognitron_round2_invited['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round2_invited['round2_'+'date_effective_test_stringencylimit'+stringency] >= '2021-01-01') & (data_cognitron_round2_invited['round2_'+'date_effective_test_stringencylimit'+stringency] < '2021-04-01'), 'WaveAt_SymptomORTest'] = '2021 Q1 Jan-Mar'
data_cognitron_round2_invited.loc[(data_cognitron_round2_invited['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round2_invited['round2_'+'date_effective_test_stringencylimit'+stringency] >= '2021-04-01') & (data_cognitron_round2_invited['round2_'+'date_effective_test_stringencylimit'+stringency] < '2021-07-01'), 'WaveAt_SymptomORTest'] = '2021 Q2 Apr-Jun'
data_cognitron_round2_invited.loc[(data_cognitron_round2_invited['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round2_invited['round2_'+'date_effective_test_stringencylimit'+stringency] >= '2021-07-01') & (data_cognitron_round2_invited['round2_'+'date_effective_test_stringencylimit'+stringency] < '2021-10-01'), 'WaveAt_SymptomORTest'] = '2021 Q3 Jul-Sep'
data_cognitron_round2_invited.loc[(data_cognitron_round2_invited['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round2_invited['round2_'+'date_effective_test_stringencylimit'+stringency] >= '2021-10-01') & (data_cognitron_round2_invited['round2_'+'date_effective_test_stringencylimit'+stringency] < '2022-01-01'), 'WaveAt_SymptomORTest'] = '2021 Q4 Oct-Dec'
data_cognitron_round2_invited.loc[(data_cognitron_round2_invited['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round2_invited['round2_'+'date_effective_test_stringencylimit'+stringency] >= '2022-01-01') & (data_cognitron_round2_invited['round2_'+'date_effective_test_stringencylimit'+stringency] < '2022-04-01'), 'WaveAt_SymptomORTest'] = '2022 Q1 Jan-Mar'
data_cognitron_round2_invited.loc[(data_cognitron_round2_invited['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round2_invited['round2_'+'date_effective_test_stringencylimit'+stringency] >= '2022-04-01') & (data_cognitron_round2_invited['round2_'+'date_effective_test_stringencylimit'+stringency] < '2022-07-01'), 'WaveAt_SymptomORTest'] = '2022 Q2 Apr-Jun'


# Round 1 and 2 dataset
# Use symptom start date, or if empty (for asymptomatic), use test date
data_cognitron_round1and2_invited.loc[(data_cognitron_round1and2_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] < '2020-04-01'), 'WaveAt_SymptomORTest'] = '2020 Q1 Jan-Mar'
data_cognitron_round1and2_invited.loc[(data_cognitron_round1and2_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] >= '2020-04-01') & (data_cognitron_round1and2_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] < '2020-07-01'), 'WaveAt_SymptomORTest'] = '2020 Q2 Apr-Jun'
data_cognitron_round1and2_invited.loc[(data_cognitron_round1and2_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] >= '2020-07-01') & (data_cognitron_round1and2_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] < '2020-10-01'), 'WaveAt_SymptomORTest'] = '2020 Q3 Jul-Sep'
data_cognitron_round1and2_invited.loc[(data_cognitron_round1and2_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] >= '2020-10-01') & (data_cognitron_round1and2_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] < '2021-01-01'), 'WaveAt_SymptomORTest'] = '2020 Q4 Oct-Dec'
data_cognitron_round1and2_invited.loc[(data_cognitron_round1and2_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] >= '2021-01-01') & (data_cognitron_round1and2_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] < '2021-04-01'), 'WaveAt_SymptomORTest'] = '2021 Q1 Jan-Mar'
data_cognitron_round1and2_invited.loc[(data_cognitron_round1and2_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] >= '2021-04-01') & (data_cognitron_round1and2_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] < '2021-07-01'), 'WaveAt_SymptomORTest'] = '2021 Q2 Apr-Jun'
data_cognitron_round1and2_invited.loc[(data_cognitron_round1and2_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] >= '2021-07-01') & (data_cognitron_round1and2_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] < '2021-10-01'), 'WaveAt_SymptomORTest'] = '2021 Q3 Jul-Sep'
data_cognitron_round1and2_invited.loc[(data_cognitron_round1and2_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] >= '2021-10-01') & (data_cognitron_round1and2_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] < '2022-01-01'), 'WaveAt_SymptomORTest'] = '2021 Q4 Oct-Dec'
data_cognitron_round1and2_invited.loc[(data_cognitron_round1and2_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] >= '2022-01-01') & (data_cognitron_round1and2_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] < '2022-04-01'), 'WaveAt_SymptomORTest'] = '2022 Q1 Jan-Mar'
data_cognitron_round1and2_invited.loc[(data_cognitron_round1and2_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] >= '2022-04-01') & (data_cognitron_round1and2_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] < '2022-07-01'), 'WaveAt_SymptomORTest'] = '2022 Q2 Apr-Jun'

# If covid group is asymptomatic, use test date
data_cognitron_round1and2_invited.loc[(data_cognitron_round1and2_invited['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round1and2_invited['round1_'+'date_effective_test_stringencylimit'+stringency] < '2020-04-01'), 'WaveAt_SymptomORTest'] = '2020 Q1 Jan-Mar'
data_cognitron_round1and2_invited.loc[(data_cognitron_round1and2_invited['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round1and2_invited['round1_'+'date_effective_test_stringencylimit'+stringency] >= '2020-04-01') & (data_cognitron_round1and2_invited['round1_'+'date_effective_test_stringencylimit'+stringency] < '2020-07-01'), 'WaveAt_SymptomORTest'] = '2020 Q2 Apr-Jun'
data_cognitron_round1and2_invited.loc[(data_cognitron_round1and2_invited['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round1and2_invited['round1_'+'date_effective_test_stringencylimit'+stringency] >= '2020-07-01') & (data_cognitron_round1and2_invited['round1_'+'date_effective_test_stringencylimit'+stringency] < '2020-10-01'), 'WaveAt_SymptomORTest'] = '2020 Q3 Jul-Sep'
data_cognitron_round1and2_invited.loc[(data_cognitron_round1and2_invited['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round1and2_invited['round1_'+'date_effective_test_stringencylimit'+stringency] >= '2020-10-01') & (data_cognitron_round1and2_invited['round1_'+'date_effective_test_stringencylimit'+stringency] < '2021-01-01'), 'WaveAt_SymptomORTest'] = '2020 Q4 Oct-Dec'
data_cognitron_round1and2_invited.loc[(data_cognitron_round1and2_invited['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round1and2_invited['round1_'+'date_effective_test_stringencylimit'+stringency] >= '2021-01-01') & (data_cognitron_round1and2_invited['round1_'+'date_effective_test_stringencylimit'+stringency] < '2021-04-01'), 'WaveAt_SymptomORTest'] = '2021 Q1 Jan-Mar'
data_cognitron_round1and2_invited.loc[(data_cognitron_round1and2_invited['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round1and2_invited['round1_'+'date_effective_test_stringencylimit'+stringency] >= '2021-04-01') & (data_cognitron_round1and2_invited['round1_'+'date_effective_test_stringencylimit'+stringency] < '2021-07-01'), 'WaveAt_SymptomORTest'] = '2021 Q2 Apr-Jun'
data_cognitron_round1and2_invited.loc[(data_cognitron_round1and2_invited['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round1and2_invited['round1_'+'date_effective_test_stringencylimit'+stringency] >= '2021-07-01') & (data_cognitron_round1and2_invited['round1_'+'date_effective_test_stringencylimit'+stringency] < '2021-10-01'), 'WaveAt_SymptomORTest'] = '2021 Q3 Jul-Sep'
data_cognitron_round1and2_invited.loc[(data_cognitron_round1and2_invited['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round1and2_invited['round1_'+'date_effective_test_stringencylimit'+stringency] >= '2021-10-01') & (data_cognitron_round1and2_invited['round1_'+'date_effective_test_stringencylimit'+stringency] < '2022-01-01'), 'WaveAt_SymptomORTest'] = '2021 Q4 Oct-Dec'
data_cognitron_round1and2_invited.loc[(data_cognitron_round1and2_invited['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round1and2_invited['round1_'+'date_effective_test_stringencylimit'+stringency] >= '2022-01-01') & (data_cognitron_round1and2_invited['round1_'+'date_effective_test_stringencylimit'+stringency] < '2022-04-01'), 'WaveAt_SymptomORTest'] = '2022 Q1 Jan-Mar'
data_cognitron_round1and2_invited.loc[(data_cognitron_round1and2_invited['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round1and2_invited['round1_'+'date_effective_test_stringencylimit'+stringency] >= '2022-04-01') & (data_cognitron_round1and2_invited['round1_'+'date_effective_test_stringencylimit'+stringency] < '2022-07-01'), 'WaveAt_SymptomORTest'] = '2022 Q2 Apr-Jun'


# -----------------------------------------------------------------------------
# Add flag to say whether earliest vaccination date is before or after infection date
## Round 1
# Use symptom start date, or if empty (for asymptomatic), use test date
data_cognitron_round1_invited.loc[(data_cognitron_round1_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] <= data_cognitron_round1_invited['ZOE_vaccination_date_earliest_valid_vaccination'])
                                  , 'Flag_VaccinationBeforeInfection'] = '1. Infection before vaccination'
data_cognitron_round1_invited.loc[(data_cognitron_round1_invited['round1_'+'symptom_start_date_estimate_stringencylimit'+stringency] > data_cognitron_round1_invited['ZOE_vaccination_date_earliest_valid_vaccination'])
                                  , 'Flag_VaccinationBeforeInfection'] = '2. Infection after vaccination'
# If covid group is asymptomatic, use test date
data_cognitron_round1_invited.loc[(data_cognitron_round1_invited['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round1_invited['round1_'+'date_effective_test_stringencylimit'+stringency] <= data_cognitron_round1_invited['ZOE_vaccination_date_earliest_valid_vaccination']), 'Flag_VaccinationBeforeInfection'] = '1. Infection before vaccination'
data_cognitron_round1_invited.loc[(data_cognitron_round1_invited['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round1_invited['round1_'+'date_effective_test_stringencylimit'+stringency] > data_cognitron_round1_invited['ZOE_vaccination_date_earliest_valid_vaccination']), 'Flag_VaccinationBeforeInfection'] = '2. Infection after vaccination'


## Round 2
# Use symptom start date, or if empty (for asymptomatic), use test date
data_cognitron_round2_invited.loc[(data_cognitron_round2_invited['round2_'+'symptom_start_date_estimate_stringencylimit'+stringency] <= data_cognitron_round2_invited['ZOE_vaccination_date_earliest_valid_vaccination'])
                                  , 'Flag_VaccinationBeforeInfection'] = '1. Infection before vaccination'
data_cognitron_round2_invited.loc[(data_cognitron_round2_invited['round2_'+'symptom_start_date_estimate_stringencylimit'+stringency] > data_cognitron_round2_invited['ZOE_vaccination_date_earliest_valid_vaccination'])
                                  , 'Flag_VaccinationBeforeInfection'] = '2. Infection after vaccination'
# If covid group is asymptomatic, use test date
data_cognitron_round2_invited.loc[(data_cognitron_round2_invited['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round2_invited['round2_'+'date_effective_test_stringencylimit'+stringency] <= data_cognitron_round2_invited['ZOE_vaccination_date_earliest_valid_vaccination']), 'Flag_VaccinationBeforeInfection'] = '1. Infection before vaccination'
data_cognitron_round2_invited.loc[(data_cognitron_round2_invited['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic')) & (data_cognitron_round2_invited['round2_'+'date_effective_test_stringencylimit'+stringency] > data_cognitron_round2_invited['ZOE_vaccination_date_earliest_valid_vaccination']), 'Flag_VaccinationBeforeInfection'] = '2. Infection after vaccination'


# -----------------------------------------------------------------------------
# Add extra weight columns to main tables
data_cognitron_round1_invited = pd.merge(data_cognitron_round1_invited, weights_join[['cssbiobank_id', 'IPW_Round2_Participation_Any'+'_stringencylimit'+stringency, 'IPW_Round2_Participation_Full'+'_stringencylimit'+stringency, 'IPW_Round1ANDRound2_Participation_Full'+'_stringencylimit'+stringency]], how = 'left', left_on = 'cssbiobank_id', right_on = 'cssbiobank_id')
data_cognitron_round2_invited = pd.merge(data_cognitron_round2_invited, weights_join[['cssbiobank_id', 'IPW_Round1_Participation_Any'+'_stringencylimit'+stringency, 'IPW_Round1_Participation_Full'+'_stringencylimit'+stringency, 'IPW_Round1ANDRound2_Participation_Full'+'_stringencylimit'+stringency]], how = 'left', left_on = 'cssbiobank_id', right_on = 'cssbiobank_id')
data_cognitron_round1and2_invited = pd.merge(data_cognitron_round1and2_invited, weights_join[['cssbiobank_id', 'IPW_Round1_Participation_Any'+'_stringencylimit'+stringency, 'IPW_Round1_Participation_Full'+'_stringencylimit'+stringency, 'IPW_Round2_Participation_Any'+'_stringencylimit'+stringency, 'IPW_Round2_Participation_Full'+'_stringencylimit'+stringency]], how = 'left', left_on = 'cssbiobank_id', right_on = 'cssbiobank_id')

data_cognitron_round1_invited_cols = data_cognitron_round1_invited.columns.to_list()
data_cognitron_round2_invited_cols = data_cognitron_round2_invited.columns.to_list()
data_cognitron_round1and2_invited_cols = data_cognitron_round1and2_invited.columns.to_list()

# -----------------------------------------------------------------------------
# Remap education level values so order is more intuitive
dict_educationLevel_cat4 = {'1. Undergraduate degree':'2. Undergraduate degree', 
                            '2. Less than undergraduate degree level':'1. Less than degree level',
                            '4. Postgraduate degree or higher':'3. Postgraduate degree or higher',
                            '3. Other/ prefer not to say':'0. Other/ prefer not to say',
                            }
data_cognitron_round1_invited['educationLevel_cat4'] = data_cognitron_round1_invited['educationLevel_cat4'].map(dict_educationLevel_cat4)
data_cognitron_round2_invited['educationLevel_cat4'] = data_cognitron_round2_invited['educationLevel_cat4'].map(dict_educationLevel_cat4)
data_cognitron_round1and2_invited['educationLevel_cat4'] = data_cognitron_round1and2_invited['educationLevel_cat4'].map(dict_educationLevel_cat4)


# -----------------------------------------------------------------------------
# Invert task 3 target detection z-score, so higher z-score is better (lower mean - closer to the target), rather than currently higher is worse. Do this so direction is in line with other tasks
data_cognitron_round1_invited['primary_accuracy_03_log_transform_zscore_processed'] = data_cognitron_round1_invited['primary_accuracy_03_log_transform_zscore_processed'] * -1
data_cognitron_round2_invited['primary_accuracy_03_log_transform_zscore_processed'] = data_cognitron_round2_invited['primary_accuracy_03_log_transform_zscore_processed'] * -1
data_cognitron_round1and2_invited['primary_accuracy_03_log_transform_zscore_processed'] = data_cognitron_round1and2_invited['primary_accuracy_03_log_transform_zscore_processed'] * -1

# -----------------------------------------------------------------------------
# Filter for rows with task data only
data_cognitron_round1_participated_any = data_cognitron_round1_invited[(data_cognitron_round1_invited['testingRound'] == 'Round_1')].copy().reset_index(drop=True)
data_cognitron_round1_participated_full = data_cognitron_round1_invited[(data_cognitron_round1_invited['testingRound'] == 'Round_1')
                                                                        & (data_cognitron_round1_invited['status_round_1'] == '3_Completion_Full')].copy().reset_index(drop=True)

data_cognitron_round2_participated_any = data_cognitron_round2_invited[(data_cognitron_round2_invited['testingRound'] == 'Round_2')].copy().reset_index(drop=True)
data_cognitron_round2_participated_full = data_cognitron_round2_invited[(data_cognitron_round2_invited['testingRound'] == 'Round_2')
                                                                        & (data_cognitron_round2_invited['status_round_2'] == '3_Completion_Full')].copy().reset_index(drop=True)


# -----------------------------------------------------------------------------
# Create dummy variables from un-ordered categoricals
# List of categorical input variables to create dummy variables for
variable_list_categorical = [# Cognitron questionnaires
                             'q_GAD7_cat4',
                             'q_GAD7_cat2',
                             'q_PHQ2_cat2',
                             'q_GAD2_cat2',
                             'q_PHQ4_cat4',
                             'q_WSAS_cat4',
                             'q_WSAS_cat2',
                             'q_chalderFatigue_cat2',
                             'educationLevel_cat4',  
                         ]

# Add dummy variables to datasets
data_cognitron_round1_participated_any = categorical_to_dummy(data_cognitron_round1_participated_any, variable_list_categorical)
data_cognitron_round1_participated_full = categorical_to_dummy(data_cognitron_round1_participated_full, variable_list_categorical)

data_cognitron_round2_participated_any = categorical_to_dummy(data_cognitron_round2_participated_any, variable_list_categorical)
data_cognitron_round2_participated_full = categorical_to_dummy(data_cognitron_round2_participated_full, variable_list_categorical)

# save column fieldnames
data_cognitron_round1_participated_any_cols = data_cognitron_round1_participated_any.columns.to_list()
data_cognitron_round1_participated_full_cols = data_cognitron_round1_participated_full.columns.to_list()

data_cognitron_round2_participated_any_cols = data_cognitron_round2_participated_any.columns.to_list()
data_cognitron_round2_participated_full_cols = data_cognitron_round2_participated_full.columns.to_list()



# -----------------------------------------------------------------------------
# Create a wide dataset which combines Round 1 and Round 2 task and questionnaire data for those who participated in both rounds
# Filter for individuals who fully completed both AND Covid group NOT unknown in both rounds
complete_round1andround2_usingRound1 = data_cognitron_round1_participated_full[(data_cognitron_round1_participated_full['status_round_1'] == '3_Completion_Full') & (data_cognitron_round1_participated_full['status_round_2'] == '3_Completion_Full') & ~( (data_cognitron_round1_participated_full['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Unknown|unknown')) |(data_cognitron_round1_participated_full['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Unknown|unknown')) )
                                                                               ]['cssbiobank_id']

complete_round1andround2_usingRound2 = data_cognitron_round2_participated_full[(data_cognitron_round2_participated_full['status_round_1'] == '3_Completion_Full') & (data_cognitron_round2_participated_full['status_round_2'] == '3_Completion_Full') & ~( (data_cognitron_round2_participated_full['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Unknown|unknown')) |(data_cognitron_round2_participated_full['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Unknown|unknown')) )
                                                                               ]['cssbiobank_id']

data_cognitron_round1_participated_full_filter = data_cognitron_round1_participated_full[data_cognitron_round1_participated_full['cssbiobank_id'].isin(complete_round1andround2_usingRound1)].copy()
data_cognitron_round2_participated_full_filter = data_cognitron_round2_participated_full[data_cognitron_round2_participated_full['cssbiobank_id'].isin(complete_round1andround2_usingRound2)].copy()

# Filter out individuals whose status changed between Round 1 and 2
data_cognitron_round1_participated_full_filter = data_cognitron_round1_participated_full_filter[~(data_cognitron_round1_participated_full_filter['round1_'+'symptomduration_grouped1_stringencylimit'+stringency] != data_cognitron_round1_participated_full_filter['round2_'+'symptomduration_grouped1_stringencylimit'+stringency])]

data_cognitron_round2_participated_full_filter = data_cognitron_round2_participated_full_filter[~(data_cognitron_round2_participated_full_filter['round1_'+'symptomduration_grouped1_stringencylimit'+stringency] != data_cognitron_round2_participated_full_filter['round2_'+'symptomduration_grouped1_stringencylimit'+stringency])]



task_accuracy_cols_unprocessed = ['primary_accuracy_01', 'primary_accuracy_02', 'primary_accuracy_03', 'primary_accuracy_04', 'primary_accuracy_05', 'primary_accuracy_06', 'primary_accuracy_07', 'primary_accuracy_08', 'primary_accuracy_09', 'primary_accuracy_10', 'primary_accuracy_11', 'primary_accuracy_12']
task_reaction_average_cols_unprocessed =['reaction_time_average_01', 'reaction_time_average_02', 'reaction_time_average_03', 'reaction_time_average_04', 'reaction_time_average_05', 'reaction_time_average_06', 'reaction_time_average_07', 'reaction_time_average_08', 'reaction_time_average_09', 'reaction_time_average_10', 'reaction_time_average_11', 'reaction_time_average_12']
task_reaction_variation_cols_unprocessed = ['reaction_time_variation_01', 'reaction_time_variation_02', 'reaction_time_variation_03', 'reaction_time_variation_04', 'reaction_time_variation_05', 'reaction_time_variation_06', 'reaction_time_variation_07', 'reaction_time_variation_08', 'reaction_time_variation_09', 'reaction_time_variation_10', 'reaction_time_variation_11', 'reaction_time_variation_12',]

# Rename task metrics and questionnaire columns - add Round Number prefix
task_accuracy_cols = ['primary_accuracy_01_cube_transform_zscore_processed', 'primary_accuracy_02_cube_transform_zscore_processed', 'primary_accuracy_03_log_transform_zscore_processed', 'primary_accuracy_04_zscore_processed', 'primary_accuracy_05_cube_transform_zscore_processed', 'primary_accuracy_06_zscore_processed', 'primary_accuracy_07_square_transform_zscore_processed', 'primary_accuracy_08_zscore_processed', 'primary_accuracy_09_cube_transform_zscore_processed', 'primary_accuracy_10_cube_transform_zscore_processed', 'primary_accuracy_11_squareroot_transform_zscore_processed', 'primary_accuracy_12_cube_transform_zscore_processed',]

task_reaction_average_cols = ['reaction_time_average_01_log_transform_zscore_processed', 'reaction_time_average_02_log_transform_zscore_processed', 'reaction_time_average_03_log_transform_zscore_processed', 'reaction_time_average_04_log_transform_zscore_processed', 'reaction_time_average_05_zscore_processed', 'reaction_time_average_06_log_transform_zscore_processed', 'reaction_time_average_07_squareroot_transform_zscore_processed', 'reaction_time_average_08_squareroot_transform_zscore_processed', 'reaction_time_average_09_log_transform_zscore_processed', 'reaction_time_average_10_log_transform_zscore_processed', 'reaction_time_average_11_log_transform_zscore_processed', 'reaction_time_average_12_squareroot_transform_zscore_processed', ]

task_reaction_variation_cols = ['reaction_time_variation_01_log_transform_zscore_processed', 'reaction_time_variation_02_log_transform_zscore_processed', 'reaction_time_variation_03_log_transform_zscore_processed', 'reaction_time_variation_04_log_transform_zscore_processed', 'reaction_time_variation_05_zscore_processed', 'reaction_time_variation_06_log_transform_zscore_processed', 'reaction_time_variation_07_log_transform_zscore_processed', 'reaction_time_variation_08_log_transform_zscore_processed', 'reaction_time_variation_09_log_transform_zscore_processed', 'reaction_time_variation_10_log_transform_zscore_processed', 'reaction_time_variation_11_log_transform_zscore_processed', 'reaction_time_variation_12_log_transform_zscore_processed',]

task_metrics_cols = task_accuracy_cols + task_reaction_average_cols + task_reaction_variation_cols

questionnaire_cols = ['q_GAD7_score', 'q_PHQ2_score', 'q_WSAS_score', 'q_chalderFatigue_score', 'q_GAD2_score', 'q_PHQ4_score', 'q_GAD7_cat4', 'q_GAD7_cat2', 'q_PHQ2_cat2', 'q_GAD2_cat2', 'q_PHQ4_cat4', 'q_WSAS_cat4', 'q_WSAS_cat2', 'q_chalderFatigue_cat2']

task_metrics_plus_questionnaire_cols = task_metrics_cols + questionnaire_cols

for col in task_metrics_plus_questionnaire_cols:
    data_cognitron_round1_participated_full_filter = data_cognitron_round1_participated_full_filter.rename(columns = {col:'round_1_'+col})
    data_cognitron_round2_participated_full_filter = data_cognitron_round2_participated_full_filter.rename(columns = {col:'round_2_'+col})

task_metrics_plus_questionnaire_cols_round1 = [('round_1_'+task) for task in task_metrics_plus_questionnaire_cols]
task_metrics_plus_questionnaire_cols_round2 = [('round_2_'+task) for task in task_metrics_plus_questionnaire_cols]
task_metrics_plus_questionnaire_cols_round1andround2 = task_metrics_plus_questionnaire_cols_round1 + task_metrics_plus_questionnaire_cols_round2



#%% PRE-PROCESSING FINISHED, FEATURE SELECTION AND REGRESSION ANALYSES BEGIN HERE
# -----------------------------------------------------------------------------
# List of dummy fields to drop from model, to use as reference category
cols_categorical_reference = [# Socio-demographics
                                   'Combined_Age_2021_grouped_decades_4: 50-60', # REFERENCE CATEGORY - Modal
                                   'ZOE_demogs_sex_Female', # REFERENCE CATEGORY - Modal
                                   'Combined_EthnicityCategory_White', # Modal
                                   'Combined_Ethnicity_cat2_White', # Modal
                                   'Combined_BMI_cat5_2: 18.5-25', # REFERENCE CATEGORY - Healthy weight
                                   'Region_London', # REFERENCE CATEGORY - Modal
                                   'Country_England', # REFERENCE CATEGORY - Modal
                                   'ZOE_demogs_healthcare_professional_no', 
                                   'Combined_IMD_cat3_2. Decile 4-7',
                                   'Combined_IMD_Quintile_3.0',
                                   # General health and wellbeing
                                   'Combined_BMI_cat5_2: 18.5-25',
                                   'PRISMA7_cat2_1. 0-2, below threshold',
                                   'ZOE_mentalhealth_ever_diagnosed_with_mental_health_condition_NO',
                                   'MH_BeforeRound1_score_mean_cat_1. 0-3, below threshold',
                                   'MH_BeforeRound2_score_mean_cat_1. 0-3, below threshold',
                                   'ZOE_conditions_condition_count_cat3_0 conditions',
                                   'ZOE_mentalhealth_condition_cat4_0 conditions',
                                   # Illness characteristics
                                   # Round 1
                                   'round1_'+'symptomduration_only_grouped_stringencylimit'+stringency+'_1: Asymptomatic',
                                   'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N1: Negative COVID-19, asymptomatic (healthy control)', # REFERENCE CATEGORY - Healthy control, absence of covid-19 and symptoms
                                   'round1_'+'symptom_count_max_cat4_stringencylimit'+stringency+'_1. 0 symptoms',
                                   'round1_'+'result_stringencylimit'+stringency+'_3.0',
                                   'round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency+'_0.0',
                                   'round1_'+'Flag_BaselineSymptoms_stringencylimit'+stringency+'_1. No regular symptoms between -28 and -14 days',
                                   # Round 2
                                   'round2_'+'symptomduration_only_grouped_stringencylimit'+stringency+'_1: Asymptomatic',
                                   'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N1: Negative COVID-19, asymptomatic (healthy control)', # REFERENCE CATEGORY - Healthy control, absence of covid-19 and symptoms
                                   'round2_'+'symptom_count_max_cat4_stringencylimit'+stringency+'_1. 0 symptoms',
                                   'round2_'+'result_stringencylimit'+stringency+'_3.0',
                                   'round2_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency+'_0.0',
                                   'round2_'+'Flag_BaselineSymptoms_stringencylimit'+stringency+'_1. No regular symptoms between -28 and -14 days',
                                   
                                   'Biobank_LCQ_B10_Recovered_NA_covid_negative',
                                   
                                   # Invitation/admin related
                                   'InvitationCohort_1. October-November 2020 COVID-19 invitation',
                                   'device_category_level_0_Computer',
                                   
                                   # Cognitron questionnaires
                                   'q_GAD7_cat4_1. 0-4, below threshold',
                                   'q_GAD7_cat2_1. 0-9, below threshold',
                                   'q_PHQ2_cat2_1. 0-2, below threshold',
                                   'q_GAD2_cat2_1. 0-2, below threshold',
                                   'q_PHQ4_cat4_1. 0-2, below threshold',
                                   'q_WSAS_cat4_1. 0-9, below threshold',
                                   'q_WSAS_cat2_1. 0-9, below threshold',
                                   'q_chalderFatigue_cat2_1. 0-28, below threshold',
                                   
                                   # Cognitron demographics                           
                                   'educationLevel_cat4_2. Undergraduate degree',
                                   ]


dict_data = {}

dict_data['task_list'] = ['rs_prospectiveMemoryWords_1_immediate',
             'rs_prospectiveMemoryObjects_1_immediate',
             'rs_motorControl',
             'rs_manipulations2D',
             'rs_targetDetection',
             'rs_spatialSpan',
             'rs_TOL',
             'rs_verbalAnalogies',
             'rs_prospectiveMemoryWords_1_delayed',
             'rs_prospectiveMemoryObjects_1_delayed',
             'rs_PAL',
             'rs_CRT',
             ]


#%% Test correlation between task scores
# -----------------------------------------------------------------------------
# Define function
def calculate_task_data_correlation(data, data_cols, corr_thresh, mask_on, dictionary):
    """ Function to calculate and visualise correlation within dataset - designed to test correlation between cognition task metrics """
    # Filter for relevant metrics
    task_data = data[data_cols].copy()
    
    # Calculate correlation between variable values 
    data_correlation = task_data.corr()
    
    data_correlation = data_correlation.rename(index = dictionary)
    data_correlation = data_correlation.rename(columns = dictionary)
    
    # Flatten and filter to identify correlations larger than a threshold and decide which features to eliminate
    data_correlation_flat = data_correlation.stack().reset_index()
    data_correlation_above_thresh = data_correlation_flat[(data_correlation_flat[0].abs() >= corr_thresh)
                                                              & (data_correlation_flat[0].abs() < 1)]
    
    # set up mask so only bottom left part of matrix is shown
    mask = np.zeros_like(data_correlation)
    mask[np.triu_indices_from(mask)] = True
    
    plt.figure()
    if mask_on == 'yes':
        sns.heatmap(data_correlation, cmap = 'YlGnBu', linewidths=.5, mask = mask, center = 0)
    else:
        plt.figure(figsize = (15,12))
        sns.heatmap(data_correlation, 
                    cmap = 'bwr', # bwr YlGnBu
                    linewidths=.5, 
                    annot=True, fmt=".2f",
                    center = 0, vmin = -1, vmax = 1
                    )
    plt.title('Correlation matrix')
        
    return data_correlation, data_correlation_flat

# task_accuracy_cols, task_reaction_average_cols, task_reaction_variation_cols

# Round 1 full
# Accuracy
corr_round1_full_accuracy, corr_flat_round1_full_accuracy = calculate_task_data_correlation(data = data_cognitron_round1_participated_full, data_cols = task_accuracy_cols, corr_thresh = 0.2, mask_on = '', dictionary = dictionary['outcome_variable_tidy'])
# Average reaction time
corr_round1_full_rt_average, corr_flat_round1_full_rt_average = calculate_task_data_correlation(data = data_cognitron_round1_participated_full, data_cols = task_reaction_average_cols, corr_thresh = 0.2, mask_on = '', dictionary = dictionary['outcome_variable_tidy'])
# Variation in reaction time
corr_round1_full_rt_variation, corr_flat_round1_full_rt_variation = calculate_task_data_correlation(data = data_cognitron_round1_participated_full, data_cols = task_reaction_variation_cols, corr_thresh = 0.2, mask_on = '', dictionary = dictionary['outcome_variable_tidy'])

# Round 2 full
# Accuracy
corr_round2_full_accuracy, corr_flat_round2_full_accuracy = calculate_task_data_correlation(data = data_cognitron_round2_participated_full, data_cols = task_accuracy_cols, corr_thresh = 0.2, mask_on = '', dictionary = dictionary['outcome_variable_tidy'])
# Average reaction time
corr_round2_full_rt_average, corr_flat_round2_full_rt_average = calculate_task_data_correlation(data = data_cognitron_round2_participated_full, data_cols = task_reaction_average_cols, corr_thresh = 0.2, mask_on = '', dictionary = dictionary['outcome_variable_tidy'])
# Variation in reaction time
corr_round2_full_rt_variation, corr_flat_round2_full_rt_variation = calculate_task_data_correlation(data = data_cognitron_round2_participated_full, data_cols = task_reaction_variation_cols, corr_thresh = 0.2, mask_on = '', dictionary = dictionary['outcome_variable_tidy'])


# Round 1 full (+ Round 2 full)

# Round 2 full (+ Round 1 full)



#%% Test correlation between mediator variables and COVID exposures
# -----------------------------------------------------------------------------
mediator_var_list_round1 = [# 
                     # 'Flag_InHospitalDuringSpell_stringencylimit'+stringency+'_0.0',
                     'round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency+'_1.0',
                     # 'q_WSAS_cat4_1. 0-9, below threshold', 
                     'q_WSAS_cat4_2. 10-20, mild', 'q_WSAS_cat4_3. 21-40, moderate to severe',
                     # 'q_PHQ4_cat4_1. 0-2, below threshold',
                     'q_PHQ4_cat4_2. 3-5, mild','q_PHQ4_cat4_3. 6-8, moderate','q_PHQ4_cat4_4. 9-12, severe',
                     # 'q_chalderFatigue_cat2_1. 0-28, below threshold', 
                     'q_chalderFatigue_cat2_2. 29-33, above threshold',
                     
                     'Biobank_LCQ_B10_Recovered_NA_covid_negative',
                     'Biobank_LCQ_B10_Recovered_no',
                     'Biobank_LCQ_B10_Recovered_yes',
                     
                     'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N1: Negative COVID-19, asymptomatic (healthy control)',
                     'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N2&3: Negative COVID-19, 0-4 weeks',
                     'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N4&5: Negative COVID-19, 4-12 weeks',
                     'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N6: Negative COVID-19, 12+ weeks',
                     'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P1: Positive COVID-19, asymptomatic',
                     'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P2&3: Positive COVID-19, 0-4 weeks',
                     'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P4&5: Positive COVID-19, 4-12 weeks',
                     'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P6: Positive COVID-19, 12+ weeks',
                     
                     ]

mediator_var_list_round2 = [# 
                     # 'Flag_InHospitalDuringSpell_stringencylimit'+stringency+'_0.0',
                     'round2_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency+'_1.0',
                     # 'q_WSAS_cat4_1. 0-9, below threshold', 
                     'q_WSAS_cat4_2. 10-20, mild', 'q_WSAS_cat4_3. 21-40, moderate to severe',
                     # 'q_PHQ4_cat4_1. 0-2, below threshold',
                     'q_PHQ4_cat4_2. 3-5, mild','q_PHQ4_cat4_3. 6-8, moderate','q_PHQ4_cat4_4. 9-12, severe',
                     # 'q_chalderFatigue_cat2_1. 0-28, below threshold', 
                     'q_chalderFatigue_cat2_2. 29-33, above threshold',
                     
                     'Biobank_LCQ_B10_Recovered_NA_covid_negative',
                     'Biobank_LCQ_B10_Recovered_no',
                     'Biobank_LCQ_B10_Recovered_yes',
                     
                     'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N1: Negative COVID-19, asymptomatic (healthy control)',
                     'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N2&3: Negative COVID-19, 0-4 weeks',
                     'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N4&5: Negative COVID-19, 4-12 weeks',
                     'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N6: Negative COVID-19, 12+ weeks',
                     'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P1: Positive COVID-19, asymptomatic',
                     'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P2&3: Positive COVID-19, 0-4 weeks',
                     'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P4&5: Positive COVID-19, 4-12 weeks',
                     'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P6: Positive COVID-19, 12+ weeks',
                     
                     ]

# Round 1 full
corr_round1_mediator_vars, corr_flat_round1_mediator_vars = calculate_task_data_correlation(data = data_cognitron_round1_participated_full, data_cols = mediator_var_list_round1, corr_thresh = 0.2, mask_on = '', dictionary = dictionary['variable_tidy'])

# Round 2 full (Round 1 groups)
corr_round2_mediator_vars, corr_flat_round2_mediator_vars = calculate_task_data_correlation(data = data_cognitron_round2_participated_full, data_cols = mediator_var_list_round1, corr_thresh = 0.2, mask_on = '', dictionary = dictionary['variable_tidy'])

# Round 2 full (Round 2 groups)
corr_round2_mediator_vars, corr_flat_round2_mediator_vars = calculate_task_data_correlation(data = data_cognitron_round2_participated_full, data_cols = mediator_var_list_round2, corr_thresh = 0.2, mask_on = '', dictionary = dictionary['variable_tidy'])



#%% Do factor analysis to generate composite accuracy and reaction time scores
# -----------------------------------------------------------------------------
# Exploratory Factor Analysis (EFA) or Principal component analysis (PCA) to generate 1st component 'g-factor' which explains largest proportion of variance between task scores
# https://en.wikipedia.org/wiki/G_factor_(psychometrics)
# https://scentellegher.github.io/machine-learning/2020/01/27/pca-loadings-sklearn.html

def do_factor_analysis(data, data_cols, factor_method, num_factors, eigenvalue_thresh):
    """Do factor analysis, choosing either exploratory factor analysis or principal component analysis. Print plots of factor loadings, eigenvalues and variance explained. Summary dataframes produced."""
    # Filter for data to do factor analysis on    
    task_data = data[data_cols].copy()
    
    # Drop rows with missing values
    # task_data = task_data.dropna(axis = 0)
        
    # Adapted from https://www.datacamp.com/tutorial/introduction-factor-analysis
    # KMO test measures the proportion of variance among variables that might be common variance. The higher the proportion, the more suited data is to Factor Analysis. Need > 0.6 as a rule of thumb. https://www.statisticshowto.com/kaiser-meyer-olkin/
    kmo_all, kmo_model = calculate_kmo(task_data)
    print('KMO test result: ' + str(kmo_model))

    if factor_method == 'efa': # exploratory factor analysis
        factor_model = FactorAnalyzer(n_factors = num_factors, rotation = None)
        factor_model_fit = factor_model.fit(task_data)
        
        factor_model_loadings = factor_model.loadings_
        factor_model_explained_variance, factor_model_explained_variance_ratio, factor_model_explained_variance_cumsum = factor_model.get_factor_variance()
        factor_model_eigenvalues, v = factor_model.get_eigenvalues()
               
    elif factor_method == 'pca': # principal component analysis
        factor_model = decomposition.PCA(n_components = 'mle') # Set up PCA
        factor_model_fit = pd.DataFrame(factor_model.fit_transform(task_data)) # Fit model
    
        factor_model_loadings = factor_model.components_.T # Loadings of variables in each component - the coefficient of the variable in linear superposition that makes up the component
        factor_model_explained_variance_ratio = factor_model.explained_variance_ratio_
        factor_model_explained_variance_cumsum = factor_model.explained_variance_ratio_.cumsum()
        factor_model_eigenvalues = factor_model.explained_variance_
        factor_model_correlations = factor_model.components_.T * np.sqrt(factor_model.explained_variance_) # Correlation of variables with component
        factor_model_covariance = factor_model.get_covariance()
        
        # If mean loading weight of first component is negative when analysing scores, multiply by -1 to invert direction for all weighting to make interpretation easier
        component_1_mean_loading = np.mean(factor_model_loadings[:,0])
        if (component_1_mean_loading < 0): # & ('accuracy' in data_cols[0]):
            factor_model_loadings = -1 * factor_model_loadings
        
    # create summary dataframes
    col_names = [('C' + str(n+1)) for n in range(0,factor_model_loadings.shape[1],1)]
    factor_model_loadings_df = pd.DataFrame(data = factor_model_loadings, columns = col_names, index = data_cols)
    if factor_method == 'efa':
        factor_model_variance_df = pd.DataFrame({'factor_eigenvalue':factor_model_eigenvalues[0:num_factors],
                                             'factor_variance_proportion':factor_model_explained_variance_ratio,
                                             'factor_variance_proportion_cumulative':factor_model_explained_variance_cumsum,}
                                            , index = col_names)
    
    elif factor_method == 'pca':
        factor_model_variance_df = pd.DataFrame({'factor_eigenvalue':factor_model_eigenvalues,
                                             'factor_variance_proportion':factor_model_explained_variance_ratio,
                                             'factor_variance_proportion_cumulative':factor_model_explained_variance_cumsum,}
                                            , index = col_names)
        
        # Rename component values dataframe columns
        col_names_fit = [('C' + str(n+1)) for n in range(0,factor_model_loadings.shape[1],1)]
        factor_model_fit.columns = col_names_fit
        
    # calculate how many eigenvalues are above threshold
    num_factors_abovethresh = len(factor_model_eigenvalues[factor_model_eigenvalues > eigenvalue_thresh])
    
    # Print eigenvalues - one rule of thumb for number to select is eigenvalues > 1
    x_axis_val = np.arange(len(factor_model_eigenvalues)) + 1
    plt.figure()
    plt.axhline(y = 1, ls = '--')
    plt.plot(x_axis_val, factor_model_eigenvalues, 'bo-', linewidth=2)
    plt.title('Scree Plot - Eigenvalues')
    plt.ylabel('Eigenvalue')
    plt.show()
        
    # Plot proportion of variance for each factor
    plt.figure()
    plt.plot(col_names, factor_model_explained_variance_ratio, 'ko-', linewidth=2)
    plt.title('Scree Plot - proportion of variance explained')
    plt.xlabel('Component')
    plt.ylabel('Proportion of Variance Explained')
    plt.show()
    
    # Plot cumulative proportion of variance - another rule of thumb is number of components that explain x% of total variance
    plt.figure()
    plt.plot(col_names, factor_model_explained_variance_cumsum, 'ro-', linewidth=2)
    plt.title('Scree Plot - proportion of variance explained')
    plt.xlabel('Component')
    plt.ylabel('Proportion of Variance Explained (cumulative)')
    plt.show()
    
    # Plot loadings for factors with eigenvalue above threshold
    plt.figure()
    factor_model_loadings_df.iloc[:,:num_factors_abovethresh].plot.bar(rot = 90)
    plt.title('Loadings for factors with eigenvalue > ' + str(eigenvalue_thresh))
    plt.ylabel('Loading')
    plt.show()
    
    return factor_model, factor_model_fit, factor_model_loadings_df, factor_model_variance_df, component_1_mean_loading



eigenvalue_thresh = 0.98
### Round 1 full
# Accuracy
round1_accuracy_factor_model, round1_accuracy_factor_fit, round1_accuracy_factor_loading, round1_accuracy_factor_variance, round1_accuracy_component_1_mean_loading = do_factor_analysis(data = data_cognitron_round1_participated_full, data_cols = task_accuracy_cols, factor_method = 'pca', num_factors = 'mle', eigenvalue_thresh = eigenvalue_thresh)
# Average reaction time
round1_rt_average_factor_model, round1_rt_average_factor_fit, round1_rt_average_factor_loading, round1_rt_average_factor_variance, round1_rt_average_component_1_mean_loading = do_factor_analysis(data = data_cognitron_round1_participated_full, data_cols = task_reaction_average_cols, factor_method = 'pca', num_factors = 'mle', eigenvalue_thresh = eigenvalue_thresh)
# task_data_test = data_cognitron_round1_participated_full[task_reaction_average_cols].copy()
# Variation in reaction time
round1_rt_variation_factor_model, round1_rt_variation_factor_fit, round1_rt_variation_factor_loading, round1_rt_variation_factor_variance, round1_rt_variation_component_1_mean_loading = do_factor_analysis(data = data_cognitron_round1_participated_full, data_cols = task_reaction_variation_cols, factor_method = 'pca', num_factors = 'mle', eigenvalue_thresh = eigenvalue_thresh)

### Round 2 full
# Accuracy
round2_accuracy_factor_model, round2_accuracy_factor_fit, round2_accuracy_factor_loading, round2_accuracy_factor_variance, round2_accuracy_component_1_mean_loading = do_factor_analysis(data = data_cognitron_round2_participated_full, data_cols = task_accuracy_cols, factor_method = 'pca', num_factors = 'mle', eigenvalue_thresh = eigenvalue_thresh)
# Average reaction time
round2_rt_average_factor_model, round2_rt_average_factor_fit, round2_rt_average_factor_loading, round2_rt_average_factor_variance, round2_rt_average_component_1_mean_loading = do_factor_analysis(data = data_cognitron_round2_participated_full, data_cols = task_reaction_average_cols, factor_method = 'pca', num_factors = 'mle', eigenvalue_thresh = eigenvalue_thresh)
# Variation in reaction time
round2_rt_variation_factor_model, round2_rt_variation_factor_fit, round2_rt_variation_factor_loading, round2_rt_variation_factor_variance, round2_rt_variation_component_1_mean_loading = do_factor_analysis(data = data_cognitron_round2_participated_full, data_cols = task_reaction_variation_cols, factor_method = 'pca', num_factors = 'mle', eigenvalue_thresh = eigenvalue_thresh)


# Round 1 full (+ Round 2 full)

# Round 2 full (+ Round 1 full)


# -----------------------------------------------------------------------------
### Compare factor loadings of main component between Round 1 and 2 factor analyses
# Accuracy
round12_accuracy_factor = pd.merge(round1_accuracy_factor_loading.add_suffix('_R1'), round2_accuracy_factor_loading.add_suffix('_R2'), how = 'left', left_index = True, right_index = True)

ax = round12_accuracy_factor[['C1_R1', 'C1_R2']].plot.bar(rot = 90)

# Average reaction time
round12_rt_average_factor = pd.merge(round1_rt_average_factor_loading.add_suffix('_R1'), round2_rt_average_factor_loading.add_suffix('_R2'), how = 'left', left_index = True, right_index = True)

ax = round12_rt_average_factor[['C1_R1', 'C1_R2']].plot.bar(rot = 90)

# Variation in reaction time
round12_rt_variation_factor = pd.merge(round1_rt_variation_factor_loading.add_suffix('_R1'), round2_rt_variation_factor_loading.add_suffix('_R2'), how = 'left', left_index = True, right_index = True)

ax = round12_rt_variation_factor[['C1_R1', 'C1_R2']].plot.bar(rot = 90)



task_accuracy_cols = ['primary_accuracy_01_cube_transform_zscore_processed', 'primary_accuracy_02_cube_transform_zscore_processed',
                       'primary_accuracy_03_log_transform_zscore_processed', 
                      'primary_accuracy_04_zscore_processed', 'primary_accuracy_05_cube_transform_zscore_processed', 'primary_accuracy_06_zscore_processed', 'primary_accuracy_07_square_transform_zscore_processed', 'primary_accuracy_08_zscore_processed', 'primary_accuracy_09_cube_transform_zscore_processed', 'primary_accuracy_10_cube_transform_zscore_processed', 'primary_accuracy_11_squareroot_transform_zscore_processed', 'primary_accuracy_12_cube_transform_zscore_processed',]

task_reaction_average_cols = ['reaction_time_average_01_log_transform_zscore_processed', 'reaction_time_average_02_log_transform_zscore_processed',
                               'reaction_time_average_03_log_transform_zscore_processed',
                              'reaction_time_average_04_log_transform_zscore_processed', 'reaction_time_average_05_zscore_processed', 'reaction_time_average_06_log_transform_zscore_processed', 'reaction_time_average_07_squareroot_transform_zscore_processed', 'reaction_time_average_08_squareroot_transform_zscore_processed', 'reaction_time_average_09_log_transform_zscore_processed', 'reaction_time_average_10_log_transform_zscore_processed', 'reaction_time_average_11_log_transform_zscore_processed', 'reaction_time_average_12_squareroot_transform_zscore_processed', ]

task_reaction_variation_cols = ['reaction_time_variation_01_log_transform_zscore_processed', 'reaction_time_variation_02_log_transform_zscore_processed',
                                'reaction_time_variation_03_log_transform_zscore_processed',
                                'reaction_time_variation_04_log_transform_zscore_processed', 'reaction_time_variation_05_zscore_processed', 'reaction_time_variation_06_log_transform_zscore_processed', 'reaction_time_variation_07_log_transform_zscore_processed', 'reaction_time_variation_08_log_transform_zscore_processed', 'reaction_time_variation_09_log_transform_zscore_processed', 'reaction_time_variation_10_log_transform_zscore_processed', 'reaction_time_variation_11_log_transform_zscore_processed', 'reaction_time_variation_12_log_transform_zscore_processed',]

# -----------------------------------------------------------------------------
# Do PCA on subgroups of related tests (either clinically identified domains or based on correlation between test scores) - DECIDED NOT TO DO IN THE END
# DOMAIN 1: IMMEDIATE AND DELAYED MEMORY (1, 2, 9, 10)

# DOMAIN 2: MOTOR CONTROL (3, 12) 

# DOMAIN 3: 

# Round 1 (all who did relevant tests)
# E.g. short term memory etc

# Round 2 (all who did relevant tests)

# Round 1 full (+ Round 2 full)

# Round 2 full (+ Round 1 full)



#%% Generate factor scores for both Round 1 and 2 using loadings from either Round 1 or Round 2 fit, and add to task data dataframes
# -----------------------------------------------------------------------------
# Use model.transform(X) method to apply loadings from Round 1 fit to Round 2 data and vice versa
def generate_factor_values(data, cols, model, suffix_string, number_components, component_1_mean_loading, plot_histogram):
    """ Generate values for factors from factor analysis models, and add to main dataset """
    # -----------------------------------------------------------------------------
    # Generate factor values from model fits
    
    # Slice task data from dataset
    data_slice = data[cols].copy()
    # Apply model fit loadings to generate values for factor components
    factor_values = model.transform(data_slice)
    column_names = [('C' + str(n+1) + suffix_string) for n in range(0,factor_values.shape[1],1)]
    factor_values = pd.DataFrame(data = factor_values, columns = column_names)
    
    # Filter for top number_components
    factor_values = factor_values.iloc[:,0:number_components]
    
    # If mean loading of component 1 is < 0, invert values by multiplying by -1 to make interpretation easier
    if component_1_mean_loading < 0:
        factor_values = factor_values * -1
    
    # -----------------------------------------------------------------------------
    # Convert factor values into z-score and percentiles
    # Generate z-score
    for col in factor_values:
        factor_values[col+'_zscore'] = zscore(np.array(factor_values[col]), nan_policy="omit") 

        # Plot histogram as a check
        if plot_histogram == 'yes':
            ax = plt.figure()
            ax = sns.histplot(data=factor_values, x=col+'_zscore')
            plt.title('Factor value: ' + col+'_zscore')


        # Calculate percentile of z-score
        # Calculate antibody value as a percentile to deal with 25,000 cap
        factor_values[col+'_zscore_percentile'] = pd.qcut(factor_values[col+'_zscore'], 100, labels = False, duplicates= 'drop')
        factor_values[col+'_zscore_percentile'] = factor_values[col+'_zscore_percentile'] + 1 # Add 1 so is 1-n instead of 0-n
        
        # Binary flag to show lowest quartile (percentile 1-25)
        factor_values.loc[(factor_values[col+'_zscore_percentile'] <= 25)
                          , col+'_zscore_bottom_quartile_flag'] = 1
        factor_values[col+'_zscore_bottom_quartile_flag'] = factor_values[col+'_zscore_bottom_quartile_flag'].fillna(0)
    
    # Merge into main dataset
    data = pd.merge(data, factor_values, how = 'left', left_index = True, right_index = True)
    
    return data


num_components = 4 # specify how many components from factor analysis to add
### Round 1 full
# Round 1 accuracy, using Round 1 loading
data_cognitron_round1_participated_full = generate_factor_values(data = data_cognitron_round1_participated_full,
                                                                 cols = task_accuracy_cols, 
                                                                 model = round1_accuracy_factor_model, 
                                                                 suffix_string = '_R1model_accuracy',
                                                                 number_components = num_components,
                                                                 component_1_mean_loading = round1_accuracy_component_1_mean_loading,
                                                                 plot_histogram = '')
# Round 1 accuracy, using Round 2 loading
data_cognitron_round1_participated_full = generate_factor_values(data = data_cognitron_round1_participated_full,
                                                                 cols = task_accuracy_cols, 
                                                                 model = round2_accuracy_factor_model, 
                                                                 suffix_string = '_R2model_accuracy',
                                                                 number_components = num_components,
                                                                 component_1_mean_loading = round2_accuracy_component_1_mean_loading,
                                                                 plot_histogram = '')

# Round 1 reaction time average, using Round 1 loading
data_cognitron_round1_participated_full = generate_factor_values(data = data_cognitron_round1_participated_full,
                                                                 cols = task_reaction_average_cols, 
                                                                 model = round1_rt_average_factor_model, 
                                                                 suffix_string = '_R1model_rt_average',
                                                                 number_components = num_components,
                                                                 component_1_mean_loading = round1_rt_average_component_1_mean_loading,
                                                                 plot_histogram = '')
# Round 1 reaction time average, using Round 2 loading
data_cognitron_round1_participated_full = generate_factor_values(data = data_cognitron_round1_participated_full,
                                                                 cols = task_reaction_average_cols, 
                                                                 model = round2_rt_average_factor_model, 
                                                                 suffix_string = '_R2model_rt_average',
                                                                 number_components = num_components,
                                                                 component_1_mean_loading = round2_rt_average_component_1_mean_loading,
                                                                 plot_histogram = '')

# Round 1 reaction time variation, using Round 1 loading
data_cognitron_round1_participated_full = generate_factor_values(data = data_cognitron_round1_participated_full,
                                                                 cols = task_reaction_variation_cols, 
                                                                 model = round1_rt_variation_factor_model, 
                                                                 suffix_string = '_R1model_rt_variation',
                                                                 number_components = num_components,
                                                                 component_1_mean_loading = round1_rt_variation_component_1_mean_loading,
                                                                 plot_histogram = '')
# Round 1 reaction time variation, using Round 2 loading
data_cognitron_round1_participated_full = generate_factor_values(data = data_cognitron_round1_participated_full,
                                                                 cols = task_reaction_variation_cols, 
                                                                 model = round2_rt_variation_factor_model, 
                                                                 suffix_string = '_R2model_rt_variation',
                                                                 number_components = num_components,
                                                                 component_1_mean_loading = round2_rt_variation_component_1_mean_loading,
                                                                 plot_histogram = '')


### Round 2 full
# Round 2 accuracy, using Round 1 loading
data_cognitron_round2_participated_full = generate_factor_values(data = data_cognitron_round2_participated_full,
                                                                 cols = task_accuracy_cols, 
                                                                 model = round1_accuracy_factor_model, 
                                                                 suffix_string = '_R1model_accuracy',
                                                                 number_components = num_components,
                                                                 component_1_mean_loading = round1_accuracy_component_1_mean_loading,
                                                                 plot_histogram = '')
# Round 2 accuracy, using Round 2 loading
data_cognitron_round2_participated_full = generate_factor_values(data = data_cognitron_round2_participated_full,
                                                                 cols = task_accuracy_cols, 
                                                                 model = round2_accuracy_factor_model, 
                                                                 suffix_string = '_R2model_accuracy',
                                                                 number_components = num_components,
                                                                 component_1_mean_loading = round2_accuracy_component_1_mean_loading,
                                                                 plot_histogram = '')

# Round 2 reaction time average, using Round 1 loading
data_cognitron_round2_participated_full = generate_factor_values(data = data_cognitron_round2_participated_full,
                                                                 cols = task_reaction_average_cols, 
                                                                 model = round1_rt_average_factor_model, 
                                                                 suffix_string = '_R1model_rt_average',
                                                                 number_components = num_components,
                                                                 component_1_mean_loading = round1_rt_average_component_1_mean_loading,
                                                                 plot_histogram = '')
# Round 2 reaction time average, using Round 2 loading
data_cognitron_round2_participated_full = generate_factor_values(data = data_cognitron_round2_participated_full,
                                                                 cols = task_reaction_average_cols, 
                                                                 model = round2_rt_average_factor_model, 
                                                                 suffix_string = '_R2model_rt_average',
                                                                 number_components = num_components,
                                                                 component_1_mean_loading = round2_rt_average_component_1_mean_loading,
                                                                 plot_histogram = '')

# Round 2 reaction time variation, using Round 1 loading
data_cognitron_round2_participated_full = generate_factor_values(data = data_cognitron_round2_participated_full,
                                                                 cols = task_reaction_variation_cols, 
                                                                 model = round1_rt_variation_factor_model, 
                                                                 suffix_string = '_R1model_rt_variation',
                                                                 number_components = num_components,
                                                                 component_1_mean_loading = round1_rt_variation_component_1_mean_loading,
                                                                 plot_histogram = '')
# Round 2 reaction time variation, using Round 2 loading
data_cognitron_round2_participated_full = generate_factor_values(data = data_cognitron_round2_participated_full,
                                                                 cols = task_reaction_variation_cols, 
                                                                 model = round2_rt_variation_factor_model, 
                                                                 suffix_string = '_R2model_rt_variation',
                                                                 number_components = num_components,
                                                                 component_1_mean_loading = round2_rt_variation_component_1_mean_loading,
                                                                 plot_histogram = '')



### Round 1 full (+ Round 2 full)


### Round 2 full (+ Round 1 full)



#%% ANALYSIS. 1.a Test conditional independence between all pairs of variables - Use to inform DAG
missing_data_values = [np.nan, 'NaN','nan', '0.1 Unknown - Answer not provided'] 

do_bivariate = 0
if do_bivariate == 1:
    # -----------------------------------------------------------------------------
    # Set fields for tests
    # List of categorical variables (non-dummy fields)
    input_var_categorical = [
                             # Socio-demographics
                             'Combined_Age_2021_grouped_decades',
                             'ZOE_demogs_sex',
                             'Combined_EthnicityCategory',
                             'Region',
                             'Combined_IMD_Quintile',
                             # General health and wellbeing
                             'Combined_BMI_cat5',
                              'MH_BeforeRound1_score_mean_cat',
                             # 'MH_BeforeRound2_score_mean_cat',
                             'ZOE_conditions_condition_count_cat3',
                              'ZOE_mentalhealth_condition_cat4',
                             # Illness characteristics
                             # Round 1
                             # 'round1_'+'symptomduration_only_grouped_stringencylimit'+stringency,
                             'round1_'+'symptomduration_grouped1_stringencylimit'+stringency,
                             # 'round1_'+'symptom_count_max_cat4_stringencylimit'+stringency,
                             # 'round1_'+'result_stringencylimit'+stringency,
                             'round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,
                             # 'round1_'+'Flag_BaselineSymptoms_stringencylimit'+stringency,   
                             
                              'Biobank_LCQ_B10_Recovered',
                             # Invitation/admin related
                             'InvitationANDRecruitmentGroup',
                             
                             'PRISMA7_score',
                             
                             ### Cognitron questionnaires
                            'q_PHQ4_cat4',
                            'q_WSAS_cat4',
                            'q_chalderFatigue_cat2',
                            'firstLanguage_cat3',
                            'educationLevel_cat4',
                                 
                             ]
    
    
    # -----------------------------------------------------------------------------
    ### Plot and test univariate association between outcome and test variables as method of feature selection
    # Based on https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/
    # For categorical variables, do chi-squared test with participation variable
    var_categorical_stattest_results_list = []
    for outcome_var in ['Combined_IMD_Quintile']: # input_var_categorical status_round_1_3_Completion_Full ZOE_mentalhealth_condition_cat4 MH_BeforeRound1_score_mean_cat
        input_var_categorical_copy = input_var_categorical.copy()
        input_var_categorical_copy.remove(outcome_var) # remove variable being tested as outcome
        var_categorical_stattest_results = categorical_variables_chi_square(data = data_cognitron_round1_invited, # data_1a_anyevidence_symptomatic data_3a_2021LCQ_infection
                                                                        input_var_categorical = input_var_categorical_copy, 
                                                                        outcome_var = outcome_var,
                                                                        drop_missing = 'yes',
                                                                        plot_crosstab = 'yes',
                                                                        print_vars = 'yes')
        
        var_categorical_stattest_results['outcome_var'] = outcome_var
            
        # Apply multiple testing p-value correction to results testing association with same outcome variable
        multiple_test_correction = fdrcorrection(var_categorical_stattest_results['Chi-squared p-value (no missing)'], alpha=0.05, method='indep', is_sorted=False)
        var_categorical_stattest_results['p_value_corrected'] = multiple_test_correction[1]
        
        # Mark p-values < 0.05 before and after correction
        var_categorical_stattest_results.loc[(var_categorical_stattest_results['Chi-squared p-value (no missing)'] < 0.05), 'significance'] = 'Significant, p < 0.05'
        var_categorical_stattest_results.loc[(var_categorical_stattest_results['Chi-squared p-value (no missing)'] >= 0.05), 'significance'] = 'Not significant, p >= 0.05'
        
        var_categorical_stattest_results.loc[(var_categorical_stattest_results['p_value_corrected'] < 0.05), 'significance_afterfdr'] = 'Significant, p < 0.05'
        var_categorical_stattest_results.loc[(var_categorical_stattest_results['p_value_corrected'] >= 0.05), 'significance_afterfdr'] = 'Not significant, p >= 0.05'
                   
        var_categorical_stattest_results_list.append(var_categorical_stattest_results)
        
    var_categorical_stattest_results_summary = pd.concat(var_categorical_stattest_results_list)



#%% ANALYSIS. 1.a Unadjusted univariate analysis of accuracy score
do_univariate = 1
if do_univariate == 1:
    # -----------------------------------------------------------------------------
    # Set fields for tests
    # Set outcome field
    # Individual task e.g. primary_accuracy_04_zscore_processed
    # Component from factor analysis e.g. C1_R1model_accuracy_zscore, C1_R1model_rt_average_zscore, C1_R1model_rt_variation_zscore
    outcome_var = 'C1_R1model_accuracy_zscore' 
    
    # Set input fields
    # List of categorical variables (non-dummy fields)
    input_var_categorical = [### Socio-demographics
                             'Combined_Age_2021_grouped_decades',
                             'ZOE_demogs_sex',
                              'Combined_EthnicityCategory',
                              'Combined_Ethnicity_cat2',
                              # 'Region',
                              'Country',
                             # 'ZOE_demogs_healthcare_professional',
                              'Combined_IMD_cat3',
                              'Combined_IMD_Quintile',
                              
                             ### General health and wellbeing
                             'Combined_BMI_cat5',
                               'PRISMA7_cat2',
                              'ZOE_mentalhealth_ever_diagnosed_with_mental_health_condition',
                             'MH_BeforeRound1_score_mean_cat',
                             'MH_BeforeRound2_score_mean_cat',
                              'ZOE_conditions_condition_count_cat3',
                              'ZOE_mentalhealth_condition_cat4',
                              
                             ### Illness characteristics
                             'round1_'+'symptomduration_only_grouped_stringencylimit'+stringency,
                              'round1_'+'symptomduration_grouped1_stringencylimit'+stringency,
                             # 'round1_'+'symptom_count_max_cat4_stringencylimit'+stringency,
                              'round1_'+'result_stringencylimit'+stringency,
                              'round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,
                             # 'round1_'+'Flag_BaselineSymptoms_stringencylimit'+stringency,     
                            'Biobank_LCQ_B10_Recovered',
                             
                             ### Invitation/admin related
                             'InvitationCohort',
                              
                            ### Cognitron questionnaires
                            'q_GAD7_cat4',
                            'q_GAD7_cat2',
                            'q_PHQ2_cat2',
                            'q_GAD2_cat2',
                            'q_PHQ4_cat4',
                            'q_WSAS_cat4',
                            'q_WSAS_cat2',
                            'q_chalderFatigue_cat2',
                            'educationLevel_cat4',
                             ]
    
    # List of continuous variables
    input_var_continuous = [### Socio-demographics
                            'Combined_Age_2021',
                            # 'Combined_IMD_Decile',
                            
                            ### General health and wellbeing
                            'Combined_BMI_value',
                            'ZOE_conditions_condition_count_max',
                            'ZOE_mentalhealth_condition_count',
                            'PRISMA7_score',
                            'MH_BeforeRound1_score_mean',
                            'MH_BeforeRound1_score_std',
                            'MH_BeforeRound2_score_mean',
                            'MH_BeforeRound2_score_std',
                            'Biobank_LCQ_A1_PrePandemicHealth',
                            'Biobank_LCQ_A2_ShieldingFlag',
                            
                            ### Illness characteristics
                            'round1_'+'symptomduration_weeks_stringencylimit'+stringency,
                            'round1_'+'symptom_count_max_stringencylimit'+stringency,
                            
                            ### Cognitron questionnaires
                            'q_GAD7_score',
                            'q_PHQ2_score',
                            'q_GAD2_score',
                            'q_PHQ4_score',
                            'q_WSAS_score',
                            'q_chalderFatigue_score',                        
                            ]
    
    # -----------------------------------------------------------------------------
    # For categorical variables
    data = data_cognitron_round1_participated_full.copy()
    for var_cat in input_var_categorical:
        # Plot distributions, split by variable of interest
        # Plot unadjusted histogram
        ax = plt.figure()
        ax = sns.histplot(data=data, 
                          x=outcome_var, 
                           hue=var_cat, 
                           element="poly",
                           stat = "probability",
                           fill = False,
                           common_norm = False,
                           # cumulative = True, fill = False, stat = "density",
                           binwidth = 0.5
                           # bins = 50#np.arange(-8, 8, 0.25)
                          )
        sns.move_legend(ax, "center", bbox_to_anchor=(0.5, -0.55)) #ncol=6, title=None, frameon=False, numpoints=2)
        # ax.legend(bbox_to_anchor=(0.5, -0.15), loc = 'lower center') # move legend out of the way
        plt.title(outcome_var + ' x ' + var_cat)
    
        # Plot boxplot for each category
        ax1 = plt.figure()
        ax1 = sns.boxplot(data = data, x = var_cat, y = outcome_var)
        sns.move_legend(ax, "center", bbox_to_anchor=(0.5, -0.55)) #ncol=6, title=None, frameon=False, numpoints=2)
        plt.title(outcome_var + ' x ' + var_cat)
        
    # -----------------------------------------------------------------------------
    # For continuous variables
    # Plot scatter plot with linear fit
    for var_cont in input_var_continuous:
        ax2 = plt.figure()
        sns.lmplot(x = var_cont, y = outcome_var, data = data)



#%% Prepare datasets for multivariable analysis
# -----------------------------------------------------------------------------
# Filter for individuals who fully completed both AND Covid group NOT unknown in both rounds
complete_round1andround2_usingRound1 = data_cognitron_round1_participated_full[(data_cognitron_round1_participated_full['status_round_1'] == '3_Completion_Full') & (data_cognitron_round1_participated_full['status_round_2'] == '3_Completion_Full') & ~( (data_cognitron_round1_participated_full['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Unknown|unknown')) |(data_cognitron_round1_participated_full['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Unknown|unknown')) )
                                                                               ]['cssbiobank_id']

complete_round1andround2_usingRound2 = data_cognitron_round2_participated_full[(data_cognitron_round2_participated_full['status_round_1'] == '3_Completion_Full') & (data_cognitron_round2_participated_full['status_round_2'] == '3_Completion_Full') & ~( (data_cognitron_round2_participated_full['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Unknown|unknown')) |(data_cognitron_round2_participated_full['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Unknown|unknown')) )
                                                                               ]['cssbiobank_id']

data_cognitron_round1_participated_full_filter = data_cognitron_round1_participated_full[data_cognitron_round1_participated_full['cssbiobank_id'].isin(complete_round1andround2_usingRound1)].copy()
data_cognitron_round2_participated_full_filter = data_cognitron_round2_participated_full[data_cognitron_round2_participated_full['cssbiobank_id'].isin(complete_round1andround2_usingRound2)].copy()

# Take a copy for renaming and merging into Round 2 dataset
data_cognitron_round1_participated_full_filter_copy = data_cognitron_round1_participated_full_filter.copy()

# -----------------------------------------------------------------------------
# Join Round 1 individual and composite task performance fields to Round 2 dataset
outcome_list_accuracy_full = ['C1_R1model_accuracy_zscore', 'C2_R1model_accuracy_zscore', 'C3_R1model_accuracy_zscore', 'C4_R1model_accuracy_zscore'] + task_accuracy_cols 
outcome_list_rt_ave_full = ['C1_R1model_rt_average_zscore', 'C2_R1model_rt_average_zscore'] + task_reaction_average_cols 
outcome_list_rt_var_full = ['C1_R1model_rt_variation_zscore', 'C2_R1model_rt_variation_zscore', 'C3_R1model_rt_variation_zscore'] + task_reaction_variation_cols 

outcome_list_PCA = ['C1_R1model_accuracy_zscore', 'C2_R1model_accuracy_zscore', 'C3_R1model_accuracy_zscore', 'C4_R1model_accuracy_zscore'] + ['C1_R1model_rt_average_zscore', 'C2_R1model_rt_average_zscore'] +['C1_R1model_rt_variation_zscore', 'C2_R1model_rt_variation_zscore', 'C3_R1model_rt_variation_zscore'] 

task_metrics_cols = task_accuracy_cols + task_reaction_average_cols + task_reaction_variation_cols

# Add prefix to field name
task_metrics_round1_list = (outcome_list_PCA + task_accuracy_cols + task_reaction_average_cols + task_reaction_variation_cols)
for col in task_metrics_round1_list:
    data_cognitron_round1_participated_full_filter_copy = data_cognitron_round1_participated_full_filter_copy.rename(columns = {col:'round_1_'+col})
task_metrics_round1_list = [('round_1_'+task) for task in task_metrics_round1_list]


# Merge in round 1 task metrics to round 2 dataset
data_cognitron_round2_participated_full_filter = pd.merge(data_cognitron_round2_participated_full_filter, data_cognitron_round1_participated_full_filter_copy[(['cssbiobank_id'] + task_metrics_round1_list)], how = 'left', on = 'cssbiobank_id')

data_cognitron_round2_participated_full_filter_cols = data_cognitron_round2_participated_full_filter.columns.to_list()

test1 = data_cognitron_round1_participated_full_filter['C1_R1model_accuracy_zscore']
test2 = data_cognitron_round2_participated_full_filter['round_1_C1_R1model_accuracy_zscore']


# -----------------------------------------------------------------------------
# Flags to identify how group changed between rounds
### Round 1
# No change in group
data_cognitron_round1_participated_full_filter.loc[(data_cognitron_round1_participated_full_filter['round1_'+'symptomduration_grouped1_stringencylimit'+stringency] == data_cognitron_round1_participated_full_filter['round2_'+'symptomduration_grouped1_stringencylimit'+stringency])
                                                   ,'Flag_ChangeBetweenRounds'] = '1. No change in covid group'
# Change within negative
data_cognitron_round1_participated_full_filter.loc[(data_cognitron_round1_participated_full_filter['round1_'+'symptomduration_grouped1_stringencylimit'+stringency] != data_cognitron_round1_participated_full_filter['round2_'+'symptomduration_grouped1_stringencylimit'+stringency]) # Group has changed
                                                   & (data_cognitron_round1_participated_full_filter['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Negative')) & (data_cognitron_round1_participated_full_filter['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Negative')) # Both rounds negative
                                                   ,'Flag_ChangeBetweenRounds'] = '2. Change within negative covid groups'
# Change within positive
data_cognitron_round1_participated_full_filter.loc[(data_cognitron_round1_participated_full_filter['round1_'+'symptomduration_grouped1_stringencylimit'+stringency] != data_cognitron_round1_participated_full_filter['round2_'+'symptomduration_grouped1_stringencylimit'+stringency]) # Group has changed
                                                   & (data_cognitron_round1_participated_full_filter['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Positive')) & (data_cognitron_round1_participated_full_filter['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Positive')) # Both rounds positive
                                                   ,'Flag_ChangeBetweenRounds'] = '3. Change within positive covid groups'
# Change from negative to positive
data_cognitron_round1_participated_full_filter.loc[(data_cognitron_round1_participated_full_filter['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Negative')) & (data_cognitron_round1_participated_full_filter['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Positive')) # Identify those who changed from Neg to Pos
                                                   ,'Flag_ChangeBetweenRounds'] = '4. Change from negative to positive covid group'

### Round 2
# No change in group
data_cognitron_round2_participated_full_filter.loc[(data_cognitron_round2_participated_full_filter['round1_'+'symptomduration_grouped1_stringencylimit'+stringency] == data_cognitron_round2_participated_full_filter['round2_'+'symptomduration_grouped1_stringencylimit'+stringency])
                                                   ,'Flag_ChangeBetweenRounds'] = '1. No change in covid group'
# Change within negative
data_cognitron_round2_participated_full_filter.loc[(data_cognitron_round2_participated_full_filter['round1_'+'symptomduration_grouped1_stringencylimit'+stringency] != data_cognitron_round2_participated_full_filter['round2_'+'symptomduration_grouped1_stringencylimit'+stringency]) # Group has changed
                                                   & (data_cognitron_round2_participated_full_filter['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Negative')) & (data_cognitron_round2_participated_full_filter['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Negative')) # Both rounds negative
                                                   ,'Flag_ChangeBetweenRounds'] = '2. Change within negative covid groups'
# Change within positive
data_cognitron_round2_participated_full_filter.loc[(data_cognitron_round2_participated_full_filter['round1_'+'symptomduration_grouped1_stringencylimit'+stringency] != data_cognitron_round2_participated_full_filter['round2_'+'symptomduration_grouped1_stringencylimit'+stringency]) # Group has changed
                                                   & (data_cognitron_round2_participated_full_filter['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Positive')) & (data_cognitron_round2_participated_full_filter['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Positive')) # Both rounds positive
                                                   ,'Flag_ChangeBetweenRounds'] = '3. Change within positive covid groups'
# Change from negative to positive
data_cognitron_round2_participated_full_filter.loc[(data_cognitron_round2_participated_full_filter['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Negative')) & (data_cognitron_round2_participated_full_filter['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Positive')) # Identify those who changed from Neg to Pos
                                                   ,'Flag_ChangeBetweenRounds'] = '4. Change from negative to positive covid group'


# -----------------------------------------------------------------------------
# Create datasets stratified by self-perceived recovery
### Round 1 full
# All negative covid groups + positive groups who self-report recovery only
data_cognitron_round1_participated_full_recoverystratify_recovered = data_cognitron_round1_participated_full[((data_cognitron_round1_participated_full['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Positive')) & (data_cognitron_round1_participated_full['Biobank_LCQ_B10_Recovered'] == 'yes')) | (data_cognitron_round1_participated_full['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Negative'))].copy()
# All negative covid groups + positive groups who self-report not recovered only
data_cognitron_round1_participated_full_recoverystratify_notrecovered = data_cognitron_round1_participated_full[((data_cognitron_round1_participated_full['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Positive')) & (data_cognitron_round1_participated_full['Biobank_LCQ_B10_Recovered'] == 'no')) | (data_cognitron_round1_participated_full['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Negative'))].copy()

### Round 2 full
# All negative covid groups + positive groups who self-report recovery only
data_cognitron_round2_participated_full_recoverystratify_recovered = data_cognitron_round2_participated_full[((data_cognitron_round2_participated_full['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Positive')) & (data_cognitron_round2_participated_full['Biobank_LCQ_B10_Recovered'] == 'yes')) | (data_cognitron_round2_participated_full['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Negative'))].copy()
# All negative covid groups + positive groups who self-report not recovered only
data_cognitron_round2_participated_full_recoverystratify_notrecovered = data_cognitron_round2_participated_full[((data_cognitron_round2_participated_full['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Positive')) & (data_cognitron_round2_participated_full['Biobank_LCQ_B10_Recovered'] == 'no')) | (data_cognitron_round2_participated_full['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Negative'))].copy()

# -----------------------------------------------------------------------------
# Create dataset selecting individuals whose status DID NOT CHANGE between Round 1 and 2
data_cognitron_round1_participated_full_filter_NoChange = data_cognitron_round1_participated_full_filter[(data_cognitron_round1_participated_full_filter['Flag_ChangeBetweenRounds'] == '1. No change in covid group')]
data_cognitron_round2_participated_full_filter_NoChange = data_cognitron_round2_participated_full_filter[(data_cognitron_round2_participated_full_filter['Flag_ChangeBetweenRounds'] == '1. No change in covid group')]

# Create dataset selecting individuals whose status went NEGATIVE TO POSITIVE between Round 1 and 2 OR REMAINED WITHIN NEGATIVE GROUPS (so we include healthy control to use as comparison group)
data_cognitron_round1_participated_full_filter_Change = data_cognitron_round1_participated_full_filter[(data_cognitron_round1_participated_full_filter['Flag_ChangeBetweenRounds'] == '4. Change from negative to positive covid group') | (data_cognitron_round1_participated_full_filter['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Negative'))]
data_cognitron_round2_participated_full_filter_Change = data_cognitron_round2_participated_full_filter[(data_cognitron_round2_participated_full_filter['Flag_ChangeBetweenRounds'] == '4. Change from negative to positive covid group') | (data_cognitron_round2_participated_full_filter['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Negative'))]


### Round 1 (with Round 2), no change in group between rounds
# All negative covid groups + positive groups who self-report recovery only
data_cognitron_round1_participated_full_filter_NoChange_recoverystratify_recovered = data_cognitron_round1_participated_full_filter_NoChange[((data_cognitron_round1_participated_full_filter_NoChange['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Positive')) & (data_cognitron_round1_participated_full_filter_NoChange['Biobank_LCQ_B10_Recovered'] == 'yes')) | (data_cognitron_round1_participated_full_filter_NoChange['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Negative'))].copy()
# All negative covid groups + positive groups who self-report not recovered only
data_cognitron_round1_participated_full_filter_NoChange_recoverystratify_notrecovered = data_cognitron_round1_participated_full_filter_NoChange[((data_cognitron_round1_participated_full_filter_NoChange['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Positive')) & (data_cognitron_round1_participated_full_filter_NoChange['Biobank_LCQ_B10_Recovered'] == 'no')) | (data_cognitron_round1_participated_full_filter_NoChange['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Negative'))].copy()

### Round 2 (with Round 1)
# All negative covid groups + positive groups who self-report recovery only
data_cognitron_round2_participated_full_filter_NoChange_recoverystratify_recovered = data_cognitron_round2_participated_full_filter_NoChange[((data_cognitron_round2_participated_full_filter_NoChange['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Positive')) & (data_cognitron_round2_participated_full_filter_NoChange['Biobank_LCQ_B10_Recovered'] == 'yes')) | (data_cognitron_round2_participated_full_filter_NoChange['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Negative'))].copy()
# All negative covid groups + positive groups who self-report not recovered only
data_cognitron_round2_participated_full_filter_NoChange_recoverystratify_notrecovered = data_cognitron_round2_participated_full_filter_NoChange[((data_cognitron_round2_participated_full_filter_NoChange['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Positive')) & (data_cognitron_round2_participated_full_filter_NoChange['Biobank_LCQ_B10_Recovered'] == 'no')) | (data_cognitron_round2_participated_full_filter_NoChange['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Negative'))].copy()


# MEASURING LONGITUDINAL CHANGE
# SUBSET 1: Negative asymptomatic + Remained negative between rounds. I.e. all negative at Round 2
data_cognitron_round2_participated_full_filter_Subset1_RemainedNeg = data_cognitron_round2_participated_full_filter[(data_cognitron_round2_participated_full_filter['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Negative'))].copy()

# SUBSET 2: Negative asymptomatic + Positive at Round 1 & Not recovered in LCQ 2021
data_cognitron_round2_participated_full_filter_Subset2_PosNotRecovered = data_cognitron_round2_participated_full_filter[((data_cognitron_round2_participated_full_filter['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Positive')) & (data_cognitron_round2_participated_full_filter['Biobank_LCQ_B10_Recovered'] == 'no')) | (data_cognitron_round2_participated_full_filter['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('N1: Negative'))].copy()

# SUBSET 3: Negative asymptomatic + Positive at Round 1 & Recovered in LCQ 2021
data_cognitron_round2_participated_full_filter_Subset3_PosRecovered = data_cognitron_round2_participated_full_filter[((data_cognitron_round2_participated_full_filter['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Positive')) & (data_cognitron_round2_participated_full_filter['Biobank_LCQ_B10_Recovered'] == 'yes')) | (data_cognitron_round2_participated_full_filter['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('N1: Negative'))].copy()

# SUBSET 4: Negative asymptomatic + Newly positive (i.e. negative at Round 1)
data_cognitron_round2_participated_full_filter_Subset4_NewPos = data_cognitron_round2_participated_full_filter[(data_cognitron_round2_participated_full_filter['Flag_ChangeBetweenRounds'] == '4. Change from negative to positive covid group') | (data_cognitron_round2_participated_full_filter['round2_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('N1: Negative'))].copy()



#%% Cross-sectional multivariable analysis of cognitive test performance for accuracy and reaction time in Round 1 and 2
# -----------------------------------------------------------------------------
# List models to run. List structure: Continuous input var, Continuous input var, Model name, Exposure variable
### Round 1 list of models to run (for full dataset where education missing for large proportion)
model_var_list_R1_short = [
                    #### EARLY LIFE
                    ### Mutual adjustment
                    # Age
                    [[],['Combined_Age_2021_grouped_decades','Combined_EthnicityCategory','ZOE_demogs_sex'],'0_All_Exposures','Combined_Age_2021_grouped_decades', ''],
                    # Sex
                    [[],['Combined_Age_2021_grouped_decades','Combined_EthnicityCategory','ZOE_demogs_sex'],'0_All_Exposures','ZOE_demogs_sex', ''],
                    # Ethnicity
                    [[],['Combined_Age_2021_grouped_decades','Combined_EthnicityCategory','ZOE_demogs_sex'],'0_All_Exposures','Combined_EthnicityCategory', ''],
                    
                    # Education level - Not available for Round 1, use Region and Deprivation as proxies
                    
                    #### ~ COLLECTED AT APP REGISTRATION
                    ### Adjusted for: Age, BMI, Deprivation, Education, Ethnicity, Frailty (PRISMA-7), Mental health condition count, Physical health condition count, Sex
                    # Region
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','Combined_IMD_Quintile','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','Region','ZOE_demogs_sex'],'0_All_Exposures','Region', ''],
                    # Adjusted for: Age, BMI, Education, Ethnicity, Frailty (PRISMA-7), Mental health condition count, Physical health condition count, Region, Sex
                    # Deprivation
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','Combined_IMD_Quintile','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','Region','ZOE_demogs_sex'],'0_All_Exposures','Combined_IMD_Quintile', ''],
                    
                    # Adjusted for: Age, BMI, Deprivation, Education, Ethnicity, Frailty (PRISMA-7), Mental health condition count, Region, Sex
                    # Number of physical health conditions
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','Combined_IMD_Quintile','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','Region','ZOE_demogs_sex'], '0_All_Exposures','ZOE_conditions_condition_count_cat3', ''],
                    # Adjusted for: Age, BMI, Deprivation, Education, Ethnicity, Frailty (PRISMA-7), Physical health condition count, Region, Sex
                    # Number of mental health conditions
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','Combined_IMD_Quintile','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','Region','ZOE_demogs_sex'], '0_All_Exposures','ZOE_mentalhealth_condition_cat4', ''],
                    # Adjusted for: Age, Deprivation, Education, Ethnicity, Frailty (PRISMA-7), Mental health condition count, Physical health condition count, Region, Sex
                    # Body mass index
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','Combined_IMD_Quintile','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','Region','ZOE_demogs_sex'], '0_All_Exposures','Combined_BMI_cat5', ''],
                    # Adjusted for: Age, BMI, Deprivation, Education, Ethnicity, Mental health condition count, Physical health condition count, Region, Sex
                    # Frailty (PRISMA-7)
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','Combined_IMD_Quintile','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','Region','ZOE_demogs_sex'], '0_All_Exposures','PRISMA7_score', ''],
                    
                    #### ~ AT INVITATION
                    ### Adjusted for: Age, BMI, Deprivation, Education, Ethnicity, Frailty (PRISMA-7), Mental health condition count, Physical health condition count, Presentation to hospital, Region, Sex + SYMPTOM DURATION
                    # Test result only
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','Combined_IMD_Quintile','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'Region','ZOE_demogs_sex', 'round1_'+'symptomduration_only_grouped_stringencylimit'+stringency,'round1_'+'result_stringencylimit'+stringency],'0_All_Exposures','round1_'+'result_stringencylimit'+stringency, ''],                    
                    ### Adjusted for: Age, BMI, Deprivation, Education, Ethnicity, Frailty (PRISMA-7), Mental health condition count, Physical health condition count, Presentation to hospital, Region, Sex
                    # COVID-19 group
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','round1_'+'symptomduration_grouped1_stringencylimit'+stringency,'Combined_IMD_Quintile','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'Region','ZOE_demogs_sex'],'0_All_Exposures','round1_'+'symptomduration_grouped1_stringencylimit'+stringency, ''],
                    ### Adjusted for: Age, BMI, COVID-19 group at invitation, Deprivation, Education, Ethnicity, Frailty (PRISMA-7), Mental health condition count, Physical health condition count, Region, Sex
                    # Presentation to hospital 
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','round1_'+'symptomduration_grouped1_stringencylimit'+stringency,'Combined_IMD_Quintile','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'Region','ZOE_demogs_sex'],'0_All_Exposures','round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency, ''],
                    
                    
                    #### ~ AT COGNITIVE ASSESSMENT
                    ### Adjusted for: Age, BMI, COVID-19 group at invitation, Education, Ethnicity, Frailty (PRISMA-7), Mental health condition count, Physical health condition count, Presentation to hospital, Sex
                    # PHQ-4
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','round1_'+'symptomduration_grouped1_stringencylimit'+stringency,'Combined_IMD_Quintile','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'Region','ZOE_demogs_sex','q_PHQ4_cat4'],'0_All_Exposures','q_PHQ4_cat4', ''],
                    # Fatigue
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','round1_'+'symptomduration_grouped1_stringencylimit'+stringency,'Combined_IMD_Quintile','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'Region','ZOE_demogs_sex','q_chalderFatigue_cat2'],'0_All_Exposures','q_chalderFatigue_cat2', ''],
                    # WSAS
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','round1_'+'symptomduration_grouped1_stringencylimit'+stringency,'Combined_IMD_Quintile','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'Region','ZOE_demogs_sex','q_WSAS_cat4'],'0_All_Exposures','q_WSAS_cat4', ''],
                    
                    # Cross-sectional mediation models
                    ### Adjusted for: Age, BMI, Deprivation, Education, Ethnicity, Frailty (PRISMA-7), Mental health condition count, Physical health condition count, Presentation to hospital, Region, Sex, + Mediator(s)
                    # COVID-19 group (+ PHQ4)
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','round1_'+'symptomduration_grouped1_stringencylimit'+stringency,'Combined_IMD_Quintile','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'Region','ZOE_demogs_sex','q_PHQ4_cat4'],'CovidGroup_Mediation_PHQ4','round1_'+'symptomduration_grouped1_stringencylimit'+stringency, ''],
                    # COVID-19 group (Chalder)
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','round1_'+'symptomduration_grouped1_stringencylimit'+stringency,'Combined_IMD_Quintile','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'Region','ZOE_demogs_sex','q_chalderFatigue_cat2'],'CovidGroup_Mediation_Fatigue','round1_'+'symptomduration_grouped1_stringencylimit'+stringency, ''],
                    # COVID-19 group (WSAS)
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','round1_'+'symptomduration_grouped1_stringencylimit'+stringency,'Combined_IMD_Quintile','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'Region','ZOE_demogs_sex','q_WSAS_cat4'],'CovidGroup_Mediation_WSAS','round1_'+'symptomduration_grouped1_stringencylimit'+stringency, ''],

                    # Covid group + PHQ4 + Chalder + WSAS
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','round1_'+'symptomduration_grouped1_stringencylimit'+stringency,'Combined_IMD_Quintile','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'Region','ZOE_demogs_sex','q_PHQ4_cat4','q_chalderFatigue_cat2','q_WSAS_cat4'],'CovidGroup_Mediation_All','round1_'+'symptomduration_grouped1_stringencylimit'+stringency, ''],

                  ]


### Round 1 list of models to run (for filtered dataset where education data available)
model_var_list_R1_short_wEducation = [
                    #### EARLY LIFE
                    ### Mutual adjustment
                    # Age
                    [[],['Combined_Age_2021_grouped_decades','Combined_EthnicityCategory','ZOE_demogs_sex'],'0_All_Exposures','Combined_Age_2021_grouped_decades', ''],
                    # Sex
                    [[],['Combined_Age_2021_grouped_decades','Combined_EthnicityCategory','ZOE_demogs_sex'],'0_All_Exposures','ZOE_demogs_sex', ''],
                    # Ethnicity
                    [[],['Combined_Age_2021_grouped_decades','Combined_EthnicityCategory','ZOE_demogs_sex'],'0_All_Exposures','Combined_EthnicityCategory', ''],
                    ### Adjusted for: Age
                    # Education level
                    [[],['Combined_Age_2021_grouped_decades','educationLevel_cat4'],'0_All_Exposures','educationLevel_cat4', ''],
                    
                    #### ~ COLLECTED AT APP REGISTRATION
                    ### Adjusted for: Age, BMI, Deprivation, Education, Ethnicity, Frailty (PRISMA-7), Mental health condition count, Physical health condition count, Sex
                    # Region
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','Combined_IMD_Quintile','educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','Region','ZOE_demogs_sex'],'0_All_Exposures','Region', ''],
                    # Adjusted for: Age, BMI, Education, Ethnicity, Frailty (PRISMA-7), Mental health condition count, Physical health condition count, Region, Sex
                    # Deprivation
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','Combined_IMD_Quintile','educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','Region','ZOE_demogs_sex'],'0_All_Exposures','Combined_IMD_Quintile', ''],
                    
                    # Adjusted for: Age, BMI, Deprivation, Education, Ethnicity, Frailty (PRISMA-7), Mental health condition count, Region, Sex
                    # Number of physical health conditions
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','Combined_IMD_Quintile','educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','Region','ZOE_demogs_sex'], '0_All_Exposures','ZOE_conditions_condition_count_cat3', ''],
                    # Adjusted for: Age, BMI, Deprivation, Education, Ethnicity, Frailty (PRISMA-7), Physical health condition count, Region, Sex
                    # Number of mental health conditions
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','Combined_IMD_Quintile','educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','Region','ZOE_demogs_sex'], '0_All_Exposures','ZOE_mentalhealth_condition_cat4', ''],
                    # Adjusted for: Age, Deprivation, Education, Ethnicity, Frailty (PRISMA-7), Mental health condition count, Physical health condition count, Region, Sex
                    # Body mass index
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','Combined_IMD_Quintile','educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','Region','ZOE_demogs_sex'], '0_All_Exposures','Combined_BMI_cat5', ''],
                    # Adjusted for: Age, BMI, Deprivation, Education, Ethnicity, Mental health condition count, Physical health condition count, Region, Sex
                    # Frailty (PRISMA-7)
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','Combined_IMD_Quintile','educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','Region','ZOE_demogs_sex'], '0_All_Exposures','PRISMA7_score', ''],
                    
                    #### ~ AT INVITATION
                    ### Adjusted for: Age, BMI, Deprivation, Education, Ethnicity, Frailty (PRISMA-7), Mental health condition count, Physical health condition count, Presentation to hospital, Region, Sex + SYMPTOM DURATION
                    # Test result only
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','Combined_IMD_Quintile','educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'Region','ZOE_demogs_sex', 'round1_'+'symptomduration_only_grouped_stringencylimit'+stringency,'round1_'+'result_stringencylimit'+stringency],'0_All_Exposures','round1_'+'result_stringencylimit'+stringency, ''],                    
                    ### Adjusted for: Age, BMI, Deprivation, Education, Ethnicity, Frailty (PRISMA-7), Mental health condition count, Physical health condition count, Presentation to hospital, Region, Sex
                    # COVID-19 group
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','round1_'+'symptomduration_grouped1_stringencylimit'+stringency,'Combined_IMD_Quintile','educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'Region','ZOE_demogs_sex'],'0_All_Exposures','round1_'+'symptomduration_grouped1_stringencylimit'+stringency, ''],
                    ### Adjusted for: Age, BMI, COVID-19 group at invitation, Deprivation, Education, Ethnicity, Frailty (PRISMA-7), Mental health condition count, Physical health condition count, Region, Sex
                    # Presentation to hospital 
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','round1_'+'symptomduration_grouped1_stringencylimit'+stringency,'Combined_IMD_Quintile','educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'Region','ZOE_demogs_sex'],'0_All_Exposures','round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency, ''],
                    
                    
                    #### ~ AT COGNITIVE ASSESSMENT
                    ### Adjusted for: Age, BMI, COVID-19 group at invitation, Education, Ethnicity, Frailty (PRISMA-7), Mental health condition count, Physical health condition count, Presentation to hospital, Sex
                    # PHQ-4
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','round1_'+'symptomduration_grouped1_stringencylimit'+stringency,'educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'ZOE_demogs_sex','q_PHQ4_cat4'],'0_All_Exposures','q_PHQ4_cat4', ''],
                    # Fatigue
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','round1_'+'symptomduration_grouped1_stringencylimit'+stringency,'educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'ZOE_demogs_sex','q_chalderFatigue_cat2'],'0_All_Exposures','q_chalderFatigue_cat2', ''],
                    # WSAS
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','round1_'+'symptomduration_grouped1_stringencylimit'+stringency,'educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'ZOE_demogs_sex','q_WSAS_cat4'],'0_All_Exposures','q_WSAS_cat4', ''],
                    
                    # Cross-sectional mediation models
                    ### Adjusted for: Age, BMI, Deprivation, Education, Ethnicity, Frailty (PRISMA-7), Mental health condition count, Physical health condition count, Presentation to hospital, Region, Sex, + Mediator(s)
                    # COVID-19 group (+ PHQ4)
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','round1_'+'symptomduration_grouped1_stringencylimit'+stringency,'Combined_IMD_Quintile','educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'Region','ZOE_demogs_sex','q_PHQ4_cat4'],'CovidGroup_Mediation_PHQ4','round1_'+'symptomduration_grouped1_stringencylimit'+stringency, ''],
                    # COVID-19 group (Chalder)
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','round1_'+'symptomduration_grouped1_stringencylimit'+stringency,'Combined_IMD_Quintile','educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'Region','ZOE_demogs_sex','q_chalderFatigue_cat2'],'CovidGroup_Mediation_Fatigue','round1_'+'symptomduration_grouped1_stringencylimit'+stringency, ''],
                    # COVID-19 group (WSAS)
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','round1_'+'symptomduration_grouped1_stringencylimit'+stringency,'Combined_IMD_Quintile','educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'Region','ZOE_demogs_sex','q_WSAS_cat4'],'CovidGroup_Mediation_WSAS','round1_'+'symptomduration_grouped1_stringencylimit'+stringency, ''],

                    # Covid group + PHQ4 + Chalder + WSAS
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','round1_'+'symptomduration_grouped1_stringencylimit'+stringency,'Combined_IMD_Quintile','educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'Region','ZOE_demogs_sex','q_PHQ4_cat4','q_chalderFatigue_cat2','q_WSAS_cat4'],'CovidGroup_Mediation_All','round1_'+'symptomduration_grouped1_stringencylimit'+stringency, ''],
                    
                  ]


# -----------------------------------------------------------------------------
### Round 2 - all exposures, covid group total, covid group direct (including mediators)
model_var_list_R2_short = [
                    #### EARLY LIFE
                    ### Mutual adjustment
                    # Age
                    [[],['Combined_Age_2021_grouped_decades','Combined_EthnicityCategory','ZOE_demogs_sex'],'0_All_Exposures','Combined_Age_2021_grouped_decades', ''],
                    # Sex
                    [[],['Combined_Age_2021_grouped_decades','Combined_EthnicityCategory','ZOE_demogs_sex'],'0_All_Exposures','ZOE_demogs_sex', ''],
                    # Ethnicity
                    [[],['Combined_Age_2021_grouped_decades','Combined_EthnicityCategory','ZOE_demogs_sex'],'0_All_Exposures','Combined_EthnicityCategory', ''],
                    ### Adjusted for: Age
                    # Education level
                    [[],['Combined_Age_2021_grouped_decades','educationLevel_cat4'],'0_All_Exposures','educationLevel_cat4', ''],
                    
                    #### ~ COLLECTED AT APP REGISTRATION
                    ### Adjusted for: Age, BMI, Deprivation, Education, Ethnicity, Frailty (PRISMA-7), Mental health condition count, Physical health condition count, Sex
                    # Region
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','Combined_IMD_Quintile','educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','Region','ZOE_demogs_sex'],'0_All_Exposures','Region', ''],
                    # Adjusted for: Age, BMI, Education, Ethnicity, Frailty (PRISMA-7), Mental health condition count, Physical health condition count, Region, Sex
                    # Deprivation
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','Combined_IMD_Quintile','educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','Region','ZOE_demogs_sex'],'0_All_Exposures','Combined_IMD_Quintile', ''],
                    
                    # Adjusted for: Age, BMI, Deprivation, Education, Ethnicity, Frailty (PRISMA-7), Mental health condition count, Region, Sex
                    # Number of physical health conditions
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','Combined_IMD_Quintile','educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','Region','ZOE_demogs_sex'], '0_All_Exposures','ZOE_conditions_condition_count_cat3', ''],
                    # Adjusted for: Age, BMI, Deprivation, Education, Ethnicity, Frailty (PRISMA-7), Physical health condition count, Region, Sex
                    # Number of mental health conditions
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','Combined_IMD_Quintile','educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','Region','ZOE_demogs_sex'], '0_All_Exposures','ZOE_mentalhealth_condition_cat4', ''],
                    # Adjusted for: Age, Deprivation, Education, Ethnicity, Frailty (PRISMA-7), Mental health condition count, Physical health condition count, Region, Sex
                    # Body mass index
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','Combined_IMD_Quintile','educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','Region','ZOE_demogs_sex'], '0_All_Exposures','Combined_BMI_cat5', ''],
                    # Adjusted for: Age, BMI, Deprivation, Education, Ethnicity, Mental health condition count, Physical health condition count, Region, Sex
                    # Frailty (PRISMA-7)
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','Combined_IMD_Quintile','educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','Region','ZOE_demogs_sex'], '0_All_Exposures','PRISMA7_score', ''],
                    
                    #### ~ AT INVITATION
                    ### Adjusted for: Age, BMI, Deprivation, Education, Ethnicity, Frailty (PRISMA-7), Mental health condition count, Physical health condition count, Presentation to hospital, Region, Sex + SYMPTOM DURATION
                    # Test result only
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','Combined_IMD_Quintile','educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round2_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'Region','ZOE_demogs_sex', 'round2_'+'symptomduration_only_grouped_stringencylimit'+stringency,'round2_'+'result_stringencylimit'+stringency],'0_All_Exposures','round2_'+'result_stringencylimit'+stringency, ''],                    
                    ### Adjusted for: Age, BMI, Deprivation, Education, Ethnicity, Frailty (PRISMA-7), Mental health condition count, Physical health condition count, Presentation to hospital, Region, Sex
                    # COVID-19 group
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','round2_'+'symptomduration_grouped1_stringencylimit'+stringency,'Combined_IMD_Quintile','educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round2_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'Region','ZOE_demogs_sex'],'0_All_Exposures','round2_'+'symptomduration_grouped1_stringencylimit'+stringency, ''],
                    ### Adjusted for: Age, BMI, COVID-19 group at invitation, Deprivation, Education, Ethnicity, Frailty (PRISMA-7), Mental health condition count, Physical health condition count, Region, Sex
                    # Presentation to hospital 
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','round2_'+'symptomduration_grouped1_stringencylimit'+stringency,'Combined_IMD_Quintile','educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round2_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'Region','ZOE_demogs_sex'],'0_All_Exposures','round2_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency, ''],
                    
                    
                    #### ~ AT COGNITIVE ASSESSMENT
                    ### Adjusted for: Age, BMI, COVID-19 group at invitation, Education, Ethnicity, Frailty (PRISMA-7), Mental health condition count, Physical health condition count, Presentation to hospital, Sex
                    # PHQ-4
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','round2_'+'symptomduration_grouped1_stringencylimit'+stringency,'educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round2_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'ZOE_demogs_sex','q_PHQ4_cat4'],'0_All_Exposures','q_PHQ4_cat4', ''],
                    # Fatigue
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','round2_'+'symptomduration_grouped1_stringencylimit'+stringency,'educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round2_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'ZOE_demogs_sex','q_chalderFatigue_cat2'],'0_All_Exposures','q_chalderFatigue_cat2', ''],
                    # WSAS
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','round2_'+'symptomduration_grouped1_stringencylimit'+stringency,'educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round2_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'ZOE_demogs_sex','q_WSAS_cat4'],'0_All_Exposures','q_WSAS_cat4', ''],
                    
                    # Cross-sectional mediation models
                    ### Adjusted for: Age, BMI, Deprivation, Education, Ethnicity, Frailty (PRISMA-7), Mental health condition count, Physical health condition count, Presentation to hospital, Region, Sex, + Mediator(s)
                    # COVID-19 group (+ PHQ4)
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','round2_'+'symptomduration_grouped1_stringencylimit'+stringency,'Combined_IMD_Quintile','educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round2_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'Region','ZOE_demogs_sex','q_PHQ4_cat4'],'CovidGroup_Mediation_PHQ4','round2_'+'symptomduration_grouped1_stringencylimit'+stringency, ''],
                    # COVID-19 group (Chalder)
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','round2_'+'symptomduration_grouped1_stringencylimit'+stringency,'Combined_IMD_Quintile','educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round2_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'Region','ZOE_demogs_sex','q_chalderFatigue_cat2'],'CovidGroup_Mediation_Fatigue','round2_'+'symptomduration_grouped1_stringencylimit'+stringency, ''],
                    # COVID-19 group (WSAS)
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','round2_'+'symptomduration_grouped1_stringencylimit'+stringency,'Combined_IMD_Quintile','educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round2_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'Region','ZOE_demogs_sex','q_WSAS_cat4'],'CovidGroup_Mediation_WSAS','round2_'+'symptomduration_grouped1_stringencylimit'+stringency, ''],

                    # Covid group + PHQ4 + Chalder + WSAS
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','round2_'+'symptomduration_grouped1_stringencylimit'+stringency,'Combined_IMD_Quintile','educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round2_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'Region','ZOE_demogs_sex','q_PHQ4_cat4','q_chalderFatigue_cat2','q_WSAS_cat4'],'CovidGroup_Mediation_All','round2_'+'symptomduration_grouped1_stringencylimit'+stringency, ''],
                    
                  ]


# Add model which includes round 1 metric as mediator - for Round 2 analysis only
model_var_list_R2_withR1mediator = model_var_list_R2_short + [# Use of round 1 outcome as mediator indicated with 'Round1' variable in categorical variable list to test effect of Round 1 performance as mediator / effect of covid group on change between Round 1 and 2
                    ### Adjusted for: Age, BMI, Deprivation, Education, Ethnicity, Frailty (PRISMA-7), Mental health condition count, Physical health condition count, Presentation to hospital, Region, Sex
                    # COVID-19 group + Round 1 outcome
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','round2_'+'symptomduration_grouped1_stringencylimit'+stringency,'Combined_IMD_Quintile','educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round2_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'Region','ZOE_demogs_sex'],'CovidGroup_Mediation_Round1','round2_'+'symptomduration_grouped1_stringencylimit'+stringency, 'addround1'],
                    
                    ### Adjusted for: Age, BMI, Deprivation, Education, Ethnicity, Frailty (PRISMA-7), Mental health condition count, Physical health condition count, Presentation to hospital, Region, Sex + SYMPTOM DURATION
                    # Test results + Round 1 outcome
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','Combined_IMD_Quintile','educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round2_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'Region','ZOE_demogs_sex', 'round2_'+'symptomduration_only_grouped_stringencylimit'+stringency,'round2_'+'result_stringencylimit'+stringency],'CovidGroup_Mediation_Round1','round2_'+'result_stringencylimit'+stringency, 'addround1'],
                    
                    ### Adjusted for: Age, BMI, COVID-19 group at invitation, Deprivation, Education, Ethnicity, Frailty (PRISMA-7), Mental health condition count, Physical health condition count, Region, Sex
                    # Presentation to hospital + Round 1 outcome
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','round2_'+'symptomduration_grouped1_stringencylimit'+stringency,'Combined_IMD_Quintile','educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round2_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'Region','ZOE_demogs_sex'],'CovidGroup_Mediation_Round1','round2_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency, 'addround1'],
                    
                    
                    ]

model_var_list_R2_withR1mediator_COVIDonly = [
                    ### Adjusted for: Age, BMI, Deprivation, Education, Ethnicity, Frailty (PRISMA-7), Mental health condition count, Physical health condition count, Presentation to hospital, Region, Sex
                    # COVID-19 group + Round 1 outcome
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','round2_'+'symptomduration_grouped1_stringencylimit'+stringency,'Combined_IMD_Quintile','educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round2_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'Region','ZOE_demogs_sex'],'CovidGroup_Mediation_Round1','round2_'+'symptomduration_grouped1_stringencylimit'+stringency, 'addround1'],
                                      
                    ### Adjusted for: Age, BMI, COVID-19 group at invitation, Deprivation, Education, Ethnicity, Frailty (PRISMA-7), Mental health condition count, Physical health condition count, Region, Sex
                    # Presentation to hospital + Round 1 outcome
                    [['PRISMA7_score'],['Combined_Age_2021_grouped_decades','Combined_BMI_cat5','round2_'+'symptomduration_grouped1_stringencylimit'+stringency,'Combined_IMD_Quintile','educationLevel_cat4','Combined_EthnicityCategory','ZOE_mentalhealth_condition_cat4','ZOE_conditions_condition_count_cat3','round2_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,'Region','ZOE_demogs_sex'],'CovidGroup_Mediation_Round1','round2_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency, 'addround1'],
                    
                    ]




# -----------------------------------------------------------------------------
# List groups of dummy variables for plotting in heatmaps
# Confounders: Demographics
var_age = ['Combined_Age_2021_grouped_decades_1: 0-30', 'Combined_Age_2021_grouped_decades_2: 30-40',
              'Combined_Age_2021_grouped_decades_3: 40-50', 'Combined_Age_2021_grouped_decades_5: 60-70',
              'Combined_Age_2021_grouped_decades_6: 70-80', 'Combined_Age_2021_grouped_decades_7: 80+',
              'Combined_Age_2021_grouped_decades_4: 50-60'
              ]
var_sex = ['ZOE_demogs_sex_Male', 'ZOE_demogs_sex_Female']
var_ethnicity = ['Combined_EthnicityCategory_Any other ethnic group', 'Combined_EthnicityCategory_Asian or Asian British', 'Combined_EthnicityCategory_Black or Black British', 'Combined_EthnicityCategory_Mixed or multiple ethnic groups', 'Combined_EthnicityCategory_White']
var_education = ['educationLevel_cat4_0. Other/ prefer not to say', 'educationLevel_cat4_1. Less than degree level', 'educationLevel_cat4_3. Postgraduate degree or higher','educationLevel_cat4_2. Undergraduate degree']
var_imd = ['Combined_IMD_Quintile_1.0', 'Combined_IMD_Quintile_2.0', 'Combined_IMD_Quintile_3.0','Combined_IMD_Quintile_4.0', 'Combined_IMD_Quintile_5.0',]
var_region = ['Region_East Midlands', 'Region_East of England', 'Region_North East', 'Region_North West',
              'Region_Northern Ireland', 'Region_Scotland', 'Region_South East', 'Region_South West',
              'Region_Wales', 'Region_West Midlands', 'Region_Yorkshire and The Humber','Region_London'
              ]

# Confounders: Health related variables
var_comorbidities = ['ZOE_conditions_condition_count_cat3_1 condition', 'ZOE_conditions_condition_count_cat3_2+ conditions', 'ZOE_conditions_condition_count_cat3_0 conditions']
var_comorbidities_MH = ['ZOE_mentalhealth_condition_cat4_1 condition', 'ZOE_mentalhealth_condition_cat4_2 conditions', 'ZOE_mentalhealth_condition_cat4_3+ conditions', 'ZOE_mentalhealth_condition_cat4_0 conditions']
var_frailty = ['PRISMA7_score',]
var_BMI = ['Combined_BMI_cat5_1: 0-18.5', 'Combined_BMI_cat5_2: 18.5-25', 'Combined_BMI_cat5_3: 25-30', 'Combined_BMI_cat5_4: 30+']

# Exposure: Covid infection status and symptom duration group
var_covidgroup_round1 = ['round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N2&3: Negative COVID-19, 0-4 weeks',
                         'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N4&5: Negative COVID-19, 4-12 weeks',
                   'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N6: Negative COVID-19, 12+ weeks',
                  'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P1: Positive COVID-19, asymptomatic',
                  'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P2&3: Positive COVID-19, 0-4 weeks',
                  'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P4&5: Positive COVID-19, 4-12 weeks',
                  'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P6: Positive COVID-19, 12+ weeks',
                  'round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N1: Negative COVID-19, asymptomatic (healthy control)']
var_result_round1 = ['round1_'+'result_stringencylimit'+stringency+'_3.0',
                     'round1_'+'result_stringencylimit'+stringency+'_4.0',]
var_covidseverity_round1 = ['round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency+'_1.0',
                            'round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency+'_0.0']

var_covidgroup_round2 = ['round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N2&3: Negative COVID-19, 0-4 weeks',
                         'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N4&5: Negative COVID-19, 4-12 weeks',
                   'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N6: Negative COVID-19, 12+ weeks',
                  'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P1: Positive COVID-19, asymptomatic',
                  'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P2&3: Positive COVID-19, 0-4 weeks',
                  'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P4&5: Positive COVID-19, 4-12 weeks',
                  'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P6: Positive COVID-19, 12+ weeks',
                  'round2_'+'symptomduration_grouped1_stringencylimit'+stringency+'_N1: Negative COVID-19, asymptomatic (healthy control)']
var_result_round2 = ['round2_'+'result_stringencylimit'+stringency+'_3.0',
                     'round2_'+'result_stringencylimit'+stringency+'_4.0',]
var_covidseverity_round2 = ['round2_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency+'_1.0',
                            'round2_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency+'_0.0']


# Mediators
var_chalder = ['q_chalderFatigue_cat2_2. 29-33, above threshold', 'q_chalderFatigue_cat2_1. 0-28, below threshold']
var_PHQ4 = ['q_PHQ4_cat4_2. 3-5, mild', 'q_PHQ4_cat4_3. 6-8, moderate', 'q_PHQ4_cat4_4. 9-12, severe',
            'q_PHQ4_cat4_1. 0-2, below threshold']
var_WSAS = ['q_WSAS_cat4_2. 10-20, mild', 'q_WSAS_cat4_3. 21-40, moderate to severe',
            'q_WSAS_cat4_1. 0-9, below threshold']
var_recovery = ['Biobank_LCQ_B10_Recovered_yes', 'Biobank_LCQ_B10_Recovered_no', 'Biobank_LCQ_B10_Recovered_NA_covid_negative']


# -----------------------------------------------------------------------------
# Run OLS models testing associated with continuous z-scores
def run_OLS_loop_multiple_outcomes(data, data_full_col_list, outcome_list, model_var_list, heatmap_list, title, use_weights, weight_var, plot_fig, dictionary, add_reference):
    """ Function to run OLS models, combine outputs of individual models and summarise in heatmap of coefficients """
    # -------
    # Part 1: Run models 
    full_df_list = []
    for task in outcome_list:
        model_summary, model_goodness, model_fit = run_OLS_regression_models(data = data, data_full_col_list = data_full_col_list, model_var_list = model_var_list, outcome_var = task, use_weights = use_weights, weight_var = weight_var, plot_fig = plot_fig)
        
        if add_reference == 'yes':
            # Add reference values to results table
            model_summary = add_reference_values(data = model_summary, 
                                                 outcome_var = task)
        
        full_df_list.append(model_summary)
        
    full_df = pd.concat(full_df_list)
    
    # Create tidy input and outcome variable name columns
    full_df['Variable_tidy'] = full_df['Variable'].map(dictionary['variable_tidy'])
    full_df['outcome_variable_tidy'] = full_df['outcome_variable'].map(dictionary['outcome_variable_tidy'])
    
    # Add specified variable order 
    full_df['Variable_order'] = full_df['Variable'].map(dictionary['variable_order'])
    full_df['outcome_variable_order'] = full_df['outcome_variable'].map(dictionary['outcome_variable_order'])
    
    # Drop rows that aren't the exposure variable
    full_df['var_match'] = full_df.apply(lambda x: x.var_exposure in x.Variable, axis = 1)
    full_df = full_df[(full_df['var_match'] == True)].reset_index(drop = True)

    # ---------------------------------------------------------------------
    # Apply multiple testing p-value correction
    # Filter for all exposures testing association with same outcome variable in turn
    outcome_var_list = full_df['outcome_variable'].unique()
    full_df_list = []
    for var in outcome_var_list:
        full_df_slice = full_df[full_df['outcome_variable'] == var].copy()
        multiple_test_correction = fdrcorrection(full_df_slice['p_value'], alpha=0.05, method='indep', is_sorted=False)
        full_df_slice['p_value_corrected'] = multiple_test_correction[1]
        full_df_list.append(full_df_slice)
    full_df = pd.concat(full_df_list)
        
    full_df.loc[(full_df['p_value_corrected'] >= 0.05)
                   ,'coeff_with_p_value_corrected_stars'] = full_df['coeff_with_p_value_stars']
    full_df.loc[(full_df['p_value_corrected'] < 0.05) &
                   (full_df['p_value_corrected'] >= 0.01)
                   ,'coeff_with_p_value_corrected_stars'] = full_df['coeff_with_p_value_stars'] + '^'
    full_df.loc[(full_df['p_value_corrected'] < 0.01) &
                   (full_df['p_value_corrected'] >= 0.001)
                   ,'coeff_with_p_value_corrected_stars'] = full_df['coeff_with_p_value_stars'] + '^'
    full_df.loc[(full_df['p_value_corrected'] < 0.001)
                   ,'coeff_with_p_value_corrected_stars'] = full_df['coeff_with_p_value_stars'] + '^'
    
    
    full_df['p_value_corrected_stars'] = full_df['p_value_stars']
    full_df.loc[(full_df['p_value_corrected'] < 0.05)
                   ,'p_value_corrected_stars'] = full_df['p_value_corrected_stars'] + '^'
    

    # -------------------------------------------------------------------------
    # Part 2: Plot heatmap of coefficients
    for sublist in heatmap_list:
        model = sublist[0]
        var_list = sublist[1]
        model_title = sublist[2]
        # Filter for variables of interest and model of interest
        data_filter = full_df[(full_df['Variable'].isin(var_list))
                            & (full_df['model_name'] == model)
                            ].copy()
        
        # Pivot to get in 2D format for heatmap
        data_filter_pivot = data_filter.pivot(index = 'Variable_order', columns = 'outcome_variable_order', values = 'coeff').reset_index()
        labels = data_filter.pivot(index = 'Variable_order', columns = 'outcome_variable_order', 
                                    values = 'coeff_with_p_value_corrected_stars' # coeff_with_conf_string coeff_with_p_value_stars coeff_with_p_value_corrected_stars
                                    )
        
        # Use dictionaries to convert from variable order to tidy name
        data_filter_pivot['Variable_tidy'] = data_filter_pivot['Variable_order'].map(dictionary['variable_order_reverse'])
        data_filter_pivot['Variable_tidy'] = data_filter_pivot['Variable_tidy'].map(dictionary['variable_tidy'])
        data_filter_pivot = data_filter_pivot.set_index('Variable_tidy')
        data_filter_pivot = data_filter_pivot.drop(columns = ['Variable_order'])
        # Use dictionaries to convert from outcome variable order to tidy name
        data_filter_pivot = data_filter_pivot.rename(columns = dictionary['outcome_variable_order_reverse'])
        data_filter_pivot = data_filter_pivot.rename(columns = dictionary['outcome_variable_tidy'])       
        
        # Plot heatmap of coefficients
        cmap = matplotlib.cm.get_cmap('bwr') # 'bwr', # YlGnBu, bwr, BrBG PiYG, seismic
        cmap_reversed = cmap.reversed()
        
        ax = plt.figure(figsize = (20,14))
        sns.heatmap(data_filter_pivot, linewidths=.5, 
                    cmap = cmap_reversed, 
                    center = 0,
                    annot=labels, fmt = '',
                    # annot = True, fmt=".2f"
                    vmin=-1, vmax=1,
                    cbar_kws={'label': 'Coefficient (units: standard deviations from mean)'}
                    )
        plt.title(title + ', Exposures: ' + model_title)
        # ax.set_title((title + ', Model: ' + model_title), loc='center', wrap=True)
        plt.xlabel('Outcome variable (composite or individual task score)')
        plt.ylabel('Exposure variable')
        
    return full_df
        

outcome_list_accuracy_full = ['C1_R1model_accuracy_zscore', 'C2_R1model_accuracy_zscore', 
                              # 'C3_R1model_accuracy_zscore', 'C4_R1model_accuracy_zscore'
                              ] + task_accuracy_cols 
outcome_list_rt_ave_full = ['C1_R1model_rt_average_zscore', 'C2_R1model_rt_average_zscore'] + task_reaction_average_cols 
outcome_list_rt_var_full = ['C1_R1model_rt_variation_zscore', 'C2_R1model_rt_variation_zscore',
                            # 'C3_R1model_rt_variation_zscore'
                            ] + task_reaction_variation_cols 

outcome_list_PCA = ['C1_R1model_accuracy_zscore', 'C2_R1model_accuracy_zscore', 'C3_R1model_accuracy_zscore', 'C4_R1model_accuracy_zscore'] + ['C1_R1model_rt_average_zscore', 'C2_R1model_rt_average_zscore'] +['C1_R1model_rt_variation_zscore', 'C2_R1model_rt_variation_zscore', 'C3_R1model_rt_variation_zscore']  

# Heatmap list structure - 1. model name, 2. variables to include in heatmap
# List for ROUND 1+2 cross-sectional 
heatmap_list_round1 = [                
                # All exposures on one heatmap - Total causal effects only
                ['0_All_Exposures', (var_age+var_sex+var_ethnicity+var_education+var_comorbidities+var_comorbidities_MH+var_BMI+var_frailty+var_covidgroup_round1+var_covidseverity_round1+var_PHQ4+var_chalder+var_WSAS+var_recovery+var_result_round1+var_region+var_imd), 'All'],
                
                # Covid group and test result only - estimating total causal effects only
                ['0_All_Exposures', (var_covidgroup_round1+var_result_round1), 'SARS-CoV-2 test result and covid group (total causal effect)'],
                
                # Covid group only - estimating total causal effects only
                ['0_All_Exposures', (var_covidgroup_round1), 'Covid group (total causal effect)'],
                
                # Age group and Covid group only - estimating total causal effects only
                ['0_All_Exposures', (var_age+var_covidgroup_round1), 'Age and covid group (total causal effect)'],
                
                # Covid group only - estimating direct causal effects (i.e. with inclusion of potential mediators)
                # PHQ4 as only mediator
                ['CovidGroup_Mediation_PHQ4', (var_covidgroup_round1), 'Covid group (direct causal effect), Mediator: PHQ-4'],
                # Chalder as only mediator
                ['CovidGroup_Mediation_Fatigue', (var_covidgroup_round1), 'Covid group (direct causal effect), Mediator: Chalder fatigue'],
                # WSAS as only mediator
                ['CovidGroup_Mediation_WSAS', (var_covidgroup_round1), 'Covid group (direct causal effect), Mediator: WSAS'],
                
                # PHQ4+Chalder+WSAS as mediators
                ['CovidGroup_Mediation_All', (var_covidgroup_round1), 'Covid group (direct causal effect), Mediators: PHQ-4, Chalder fatigue, WSAS'],
                ]

heatmap_list_round2 = [                
                # All exposures on one heatmap - Total causal effects only
                ['0_All_Exposures', (var_age+var_sex+var_ethnicity+var_education+var_comorbidities+var_comorbidities_MH+var_BMI+var_frailty+var_covidgroup_round2+var_covidseverity_round2+var_PHQ4+var_chalder+var_WSAS+var_recovery+var_result_round2+var_region+var_imd), 'All'],
                
                # Covid group and test result only - estimating total causal effects only
                ['0_All_Exposures', (var_covidgroup_round2+var_result_round2), 'SARS-CoV-2 test result and covid group (total causal effect)'],
                
                # Covid group only - estimating total causal effects only
                ['0_All_Exposures', (var_covidgroup_round2), 'Covid group (total causal effect)'],
                
                # Age group and Covid group only - estimating total causal effects only
                ['0_All_Exposures', (var_age+var_covidgroup_round2), 'Age and covid group (total causal effect)'],
                
                # Covid group only - estimating direct causal effects (i.e. with inclusion of potential mediators)
                # PHQ4 as only mediator
                ['CovidGroup_Mediation_PHQ4', (var_covidgroup_round2), 'Covid group (direct causal effect), Mediator: PHQ-4'],
                # Chalder as only mediator
                ['CovidGroup_Mediation_Fatigue', (var_covidgroup_round2), 'Covid group (direct causal effect), Mediator: Chalder fatigue'],
                # WSAS as only mediator
                ['CovidGroup_Mediation_WSAS', (var_covidgroup_round2), 'Covid group (direct causal effect), Mediator: WSAS'],
                
                # PHQ4+Chalder+WSAS as mediators
                ['CovidGroup_Mediation_All', (var_covidgroup_round2), 'Covid group (direct causal effect), Mediators: PHQ-4, Chalder fatigue, WSAS'],
                ]

heatmap_list_round2 = heatmap_list_round2 + [# Covid group only - estimating direct causal effects through inclusion of Round 1 as mediator
                                   ['CovidGroup_Mediation_Round1', (var_covidgroup_round2), 'Covid group (direct causal effect), Mediator: Round 1 metric'],
                                   # Covid group and test result only - estimating direct causal effects through inclusion of Round 1 as mediator
                                   ['CovidGroup_Mediation_Round1', (var_result_round2+var_covidgroup_round2), 'Test result and covid group (direct causal effect), Mediator: Round 1 metric'],
                                   ['CovidGroup_Mediation_Round1', (var_result_round2+var_covidgroup_round2+var_covidseverity_round2), 'Test result, covid group and hospitalisation (direct causal effect), Mediator: Round 1 metric'],
                                   ]


# -----------------------------------------------------------------------------
# Run various sequences of models and produce heatmaps and tables of results
#### ANALYSES ON FULL DATASETS 
# -----------------------------------------------------------------------------
### ACCURACY
### Round 1 full
overall_round1_accuracy = run_OLS_loop_multiple_outcomes(data = data_cognitron_round1_participated_full, 
                                         data_full_col_list = data_cognitron_round1_participated_full_cols, 
                                         outcome_list = outcome_list_accuracy_full, 
                                         model_var_list = model_var_list_R1_short, 
                                         heatmap_list = heatmap_list_round1, title = 'Round 1, Accuracy',
                                         use_weights = 'yes', weight_var = 'IPW_Round1_Participation_Full'+'_stringencylimit'+stringency, plot_fig = '', dictionary = dictionary, add_reference = '')
### Round 2 full
overall_round2_accuracy = run_OLS_loop_multiple_outcomes(data = data_cognitron_round2_participated_full, 
                                         data_full_col_list = data_cognitron_round2_participated_full_cols, 
                                         outcome_list = outcome_list_accuracy_full, 
                                         model_var_list = model_var_list_R2_short, 
                                         heatmap_list = heatmap_list_round2, title = 'Round 2, Accuracy',
                                         use_weights = 'yes', weight_var = 'IPW_Round2_Participation_Full'+'_stringencylimit'+stringency, plot_fig = '', dictionary = dictionary, add_reference = '')

xx

# Filtered for task breakdown
var_select = 'test result' # 'COVID-19 group' 'hospital' 'SARS-CoV-2'
model_select = '0_All_Exposures'
overall_round1_accuracy_taskcoeff = overall_round1_accuracy[(overall_round1_accuracy['Variable_tidy'].str.contains(var_select))
                        & (overall_round1_accuracy['model_name'].str.contains(model_select))
                        & (overall_round1_accuracy['outcome_variable_order'] >= 4)
                        ].copy()

test_pivot_coeff = overall_round1_accuracy_taskcoeff.pivot_table(index = ['Variable_order','Variable_tidy'], columns = ['outcome_variable_order','outcome_variable_tidy'], values = 'coeff')
test_pivot_pvalue = overall_round1_accuracy_taskcoeff.pivot(index = ['Variable_order','Variable_tidy'], columns = ['outcome_variable_order','outcome_variable_tidy'], values = 'p_value_corrected_stars')

# -----------------------------------------------------------------------------
# Stratify by recovery
### Round 1 full
# Recovered
overall_round1_accuracy_stratify_recovered = run_OLS_loop_multiple_outcomes(data = data_cognitron_round1_participated_full_recoverystratify_recovered, 
                                         data_full_col_list = data_cognitron_round1_participated_full_cols, 
                                         outcome_list = outcome_list_accuracy_full, 
                                         model_var_list = model_var_list_R1_short, 
                                         heatmap_list = heatmap_list_round1, title = 'Round 1, Accuracy (Self-perceived COVID-19 recovery: Recovered',
                                         use_weights = 'yes', weight_var = 'IPW_Round1_Participation_Full'+'_stringencylimit'+stringency, plot_fig = '', dictionary = dictionary, add_reference = '')
# Not recovered
overall_round1_accuracy_stratify_notrecovered = run_OLS_loop_multiple_outcomes(data = data_cognitron_round1_participated_full_recoverystratify_notrecovered, 
                                         data_full_col_list = data_cognitron_round1_participated_full_cols, 
                                         outcome_list = outcome_list_accuracy_full, 
                                         model_var_list = model_var_list_R1_short, 
                                         heatmap_list = heatmap_list_round1, title = 'Round 1, Accuracy (Self-perceived COVID-19 recovery: Not recovered',
                                         use_weights = 'yes', weight_var = 'IPW_Round1_Participation_Full'+'_stringencylimit'+stringency, plot_fig = '', dictionary = dictionary, add_reference = '')


### Round 2 full
# Recovered
overall_round2_accuracy_stratify_recovered = run_OLS_loop_multiple_outcomes(data = data_cognitron_round2_participated_full_recoverystratify_recovered, 
                                         data_full_col_list = data_cognitron_round2_participated_full_cols, 
                                         outcome_list = outcome_list_accuracy_full, 
                                         model_var_list = model_var_list_R2_short, 
                                         heatmap_list = heatmap_list_round2, title = 'Round 2, Accuracy (Self-perceived COVID-19 recovery: Recovered',
                                         use_weights = 'yes', weight_var = 'IPW_Round2_Participation_Full'+'_stringencylimit'+stringency, plot_fig = '', dictionary = dictionary, add_reference = '')
# Not recovered
overall_round2_accuracy_stratify_notrecovered = run_OLS_loop_multiple_outcomes(data = data_cognitron_round2_participated_full_recoverystratify_notrecovered, 
                                         data_full_col_list = data_cognitron_round2_participated_full_cols, 
                                         outcome_list = outcome_list_accuracy_full, 
                                         model_var_list = model_var_list_R2_short, 
                                         heatmap_list = heatmap_list_round2, title = 'Round 2, Accuracy (Self-perceived COVID-19 recovery: Not recovered',
                                         use_weights = 'yes', weight_var = 'IPW_Round2_Participation_Full'+'_stringencylimit'+stringency, plot_fig = '', dictionary = dictionary, add_reference = '')



# -----------------------------------------------------------------------------
# SUBSET BY THOSE WHO COMPLETED BOTH ROUNDS
### ACCURACY 
### Round 1 (for those with Round 2 full)
# Exposures: All. Mediators: None and questionnaires
# USING ROUND 1 COVID GROUP
overall_round1_withround2_accuracy_round1group = run_OLS_loop_multiple_outcomes(data = data_cognitron_round1_participated_full_filter, 
                                         data_full_col_list = data_cognitron_round1_participated_full_cols, 
                                         outcome_list = outcome_list_accuracy_full, 
                                         model_var_list = model_var_list_R1_short_wEducation, 
                                         heatmap_list = heatmap_list_round1, title = 'Round 1, Accuracy [Round 2 also complete], Round 1 groups',
                                         use_weights = 'yes', weight_var = 'IPW_Round1ANDRound2_Participation_Full'+'_stringencylimit'+stringency, plot_fig = '', dictionary = dictionary, add_reference = '')
# USING ROUND 2 COVID GROUP
overall_round1_withround2_accuracy_round2group = run_OLS_loop_multiple_outcomes(data = data_cognitron_round1_participated_full_filter, 
                                         data_full_col_list = data_cognitron_round1_participated_full_cols, 
                                         outcome_list = outcome_list_accuracy_full, 
                                         model_var_list = model_var_list_R2_short, 
                                         heatmap_list = heatmap_list_round2, title = 'Round 1, Accuracy [Round 2 also complete], Round 2 groups',
                                         use_weights = 'yes', weight_var = 'IPW_Round1ANDRound2_Participation_Full'+'_stringencylimit'+stringency, plot_fig = '', dictionary = dictionary, add_reference = '')


# -----------------------------------------------------------------------------
# SUBSET BY THOSE WHO COMPLETED BOTH ROUNDS - SUBSETS TO TEST LONGITUDINAL CHANGE
### ACCURACY 
### Round 1 (for those with Round 2 full)
# Exposures: All. Mediators: None and questionnaires
overall_round2_withround1_accuracy_Subset1_RemainedNeg = run_OLS_loop_multiple_outcomes(data = data_cognitron_round2_participated_full_filter_Subset1_RemainedNeg, 
                                         data_full_col_list = data_cognitron_round2_participated_full_cols, 
                                         outcome_list = ['C1_R1model_accuracy_zscore'], 
                                         model_var_list = model_var_list_R2_withR1mediator_COVIDonly, 
                                         heatmap_list = heatmap_list_round2, title = 'Round 2 [Round 1 also complete], Accuracy, Subset 1: Remained negative',
                                         use_weights = 'yes', weight_var = 'IPW_Round1ANDRound2_Participation_Full'+'_stringencylimit'+stringency, plot_fig = '', dictionary = dictionary, add_reference = '')
overall_round2_withround1_accuracy_Subset2_PosNotRecovered = run_OLS_loop_multiple_outcomes(data = data_cognitron_round2_participated_full_filter_Subset2_PosNotRecovered, 
                                         data_full_col_list = data_cognitron_round2_participated_full_cols, 
                                         outcome_list = ['C1_R1model_accuracy_zscore'], 
                                         model_var_list = model_var_list_R2_withR1mediator_COVIDonly, 
                                         heatmap_list = heatmap_list_round2, title = 'Round 2 [Round 1 also complete], Accuracy, Subset 2: Positive at Round 1, not recovered',
                                         use_weights = 'yes', weight_var = 'IPW_Round1ANDRound2_Participation_Full'+'_stringencylimit'+stringency, plot_fig = '', dictionary = dictionary, add_reference = '')
overall_round2_withround1_accuracy_Subset3_PosRecovered = run_OLS_loop_multiple_outcomes(data = data_cognitron_round2_participated_full_filter_Subset3_PosRecovered, 
                                         data_full_col_list = data_cognitron_round2_participated_full_cols, 
                                         outcome_list = ['C1_R1model_accuracy_zscore'], 
                                         model_var_list = model_var_list_R2_withR1mediator_COVIDonly, 
                                         heatmap_list = heatmap_list_round2, title = 'Round 2 [Round 1 also complete], Accuracy, Subset 3: Positive at Round 1, recovered',
                                         use_weights = 'yes', weight_var = 'IPW_Round1ANDRound2_Participation_Full'+'_stringencylimit'+stringency, plot_fig = '', dictionary = dictionary, add_reference = '')
overall_round2_withround1_accuracy_Subset4_NewPos = run_OLS_loop_multiple_outcomes(data = data_cognitron_round2_participated_full_filter_Subset4_NewPos, 
                                         data_full_col_list = data_cognitron_round2_participated_full_cols, 
                                         outcome_list = ['C1_R1model_accuracy_zscore'], 
                                         model_var_list = model_var_list_R2_withR1mediator_COVIDonly, 
                                         heatmap_list = heatmap_list_round2, title = 'Round 2 [Round 1 also complete], Accuracy, Subset 4: Newly Positive at Round 2',
                                         use_weights = 'yes', weight_var = 'IPW_Round1ANDRound2_Participation_Full'+'_stringencylimit'+stringency, plot_fig = '', dictionary = dictionary, add_reference = '')


XXXXXXXXXXXXXXXXXXXXXXXXXX


### REACTION TIME AVERAGE
### Round 1 (for those with Round 2 full)
# Exposures: All. Mediators: None and questionnaires
overall_round1_withround2_rt_ave = run_OLS_loop_multiple_outcomes(data = data_cognitron_round1_participated_full_filter, 
                                         data_full_col_list = data_cognitron_round1_participated_full_cols, 
                                         outcome_list = outcome_list_rt_ave_full, 
                                         model_var_list = model_var_list_R1_short_wEducation, 
                                         heatmap_list = heatmap_list_round1, title = 'Round 1, Reaction time (average) [Round 2 also complete]',
                                         use_weights = 'yes', weight_var = 'IPW_Round1ANDRound2_Participation_Full'+'_stringencylimit'+stringency, plot_fig = '', dictionary = dictionary, add_reference = '')

### Round 2 (for those with Round 1 full)
# Exposures: All. Mediators: None and questionnaires
overall_round2_withround1_rt_ave = run_OLS_loop_multiple_outcomes(data = data_cognitron_round2_participated_full_filter, 
                                         data_full_col_list = data_cognitron_round2_participated_full_cols, 
                                         outcome_list = outcome_list_rt_ave_full, 
                                         model_var_list = model_var_list_R2_withR1mediator, 
                                         heatmap_list = heatmap_list_round2, title = 'Round 2, Reaction time (average) [Round 1 also complete]',
                                         use_weights = 'yes', weight_var = 'IPW_Round1ANDRound2_Participation_Full'+'_stringencylimit'+stringency, plot_fig = '', dictionary = dictionary, add_reference = '')


### REACTION TIME VARIATION
### Round 1 (for those with Round 2 full)
# Exposures: All. Mediators: None and questionnaires
overall_round1_withround2_rt_var = run_OLS_loop_multiple_outcomes(data = data_cognitron_round1_participated_full_filter, 
                                         data_full_col_list = data_cognitron_round1_participated_full_cols, 
                                         outcome_list = outcome_list_rt_var_full, 
                                         model_var_list = model_var_list_R1_short_wEducation, 
                                         heatmap_list = heatmap_list_round1, title = 'Round 1, Reaction time (variation) [Round 2 also complete]',
                                         use_weights = 'yes', weight_var = 'IPW_Round1ANDRound2_Participation_Full'+'_stringencylimit'+stringency, plot_fig = '', dictionary = dictionary, add_reference = '')

### Round 2 (for those with Round 1 full)
# Exposures: All. Mediators: None and questionnaires
overall_round2_withround1_rt_var = run_OLS_loop_multiple_outcomes(data = data_cognitron_round2_participated_full_filter, 
                                         data_full_col_list = data_cognitron_round2_participated_full_cols, 
                                         outcome_list = outcome_list_rt_var_full, 
                                         model_var_list = model_var_list_R2_withR1mediator, 
                                         heatmap_list = heatmap_list_round2, title = 'Round 2, Reaction time (variation) [Round 1 also complete]',
                                         use_weights = 'yes', weight_var = 'IPW_Round1ANDRound2_Participation_Full'+'_stringencylimit'+stringency, plot_fig = '', dictionary = dictionary, add_reference = '')


### REACTION TIME AVERAGE
### Round 1 full
overall_round1_rt_ave = run_OLS_loop_multiple_outcomes(data = data_cognitron_round1_participated_full, 
                                         data_full_col_list = data_cognitron_round1_participated_full_cols, 
                                         outcome_list = outcome_list_rt_ave_full, 
                                         model_var_list = model_var_list_R1_short, 
                                         heatmap_list = heatmap_list_round1, title = 'Round 1, Reaction time (average)',
                                         use_weights = 'yes', weight_var = 'IPW_Round1_Participation_Full'+'_stringencylimit'+stringency, plot_fig = '', dictionary = dictionary, add_reference = '')
### Round 2 full
overall_round2_rt_ave = run_OLS_loop_multiple_outcomes(data = data_cognitron_round2_participated_full, 
                                         data_full_col_list = data_cognitron_round2_participated_full_cols, 
                                         outcome_list = outcome_list_rt_ave_full, 
                                         model_var_list = model_var_list_R2_short, 
                                         heatmap_list = heatmap_list_round2, title = 'Round 2, Reaction time (average)',
                                         use_weights = 'yes', weight_var = 'IPW_Round2_Participation_Full'+'_stringencylimit'+stringency, plot_fig = '', dictionary = dictionary, add_reference = '')


### REACTION TIME VARIATION
### Round 1 full
overall_round1_rt_var = run_OLS_loop_multiple_outcomes(data = data_cognitron_round1_participated_full, 
                                         data_full_col_list = data_cognitron_round1_participated_full_cols, 
                                         outcome_list = outcome_list_rt_var_full, 
                                         model_var_list = model_var_list_R1_short, 
                                         heatmap_list = heatmap_list_round1, title = 'Round 1, Reaction time (variation)',
                                         use_weights = 'yes', weight_var = 'IPW_Round1_Participation_Full'+'_stringencylimit'+stringency, plot_fig = '', dictionary = dictionary, add_reference = '')
### Round 2 full
overall_round2_rt_var = run_OLS_loop_multiple_outcomes(data = data_cognitron_round2_participated_full, 
                                         data_full_col_list = data_cognitron_round2_participated_full_cols, 
                                         outcome_list = outcome_list_rt_var_full, 
                                         model_var_list = model_var_list_R2_short, 
                                         heatmap_list = heatmap_list_round2, title = 'Round 2, Reaction time (variation)',
                                         use_weights = 'yes', weight_var = 'IPW_Round2_Participation_Full'+'_stringencylimit'+stringency, plot_fig = '', dictionary = dictionary, add_reference = '')


xxx


#%% RESULTS - Generate summary of participant characteristics 'Table 1' type figure
# -----------------------------------------------------------------------------
round_dp = 2
missing_data_values = [np.nan, 'NaN','nan', '0.1 Unknown - Answer not provided'] 

# Choose variable
var_list = ['Combined_Age_2021_grouped_decades', 'Combined_Age_2021_grouped_decades', 'Combined_Age_2021_grouped_decades', 'Combined_Age_2021_grouped_decades', 'Combined_Age_2021_grouped_decades', 'Combined_Age_2021_grouped_decades', 'Combined_Age_2021_grouped_decades',
            'ZOE_demogs_sex', 'ZOE_demogs_sex',
            'Combined_EthnicityCategory', 'Combined_EthnicityCategory', 'Combined_EthnicityCategory', 'Combined_EthnicityCategory', 'Combined_EthnicityCategory',
            
            'InvitationCohort', 'InvitationCohort', 'InvitationCohort',
            
            'educationLevel_cat4', 'educationLevel_cat4', 'educationLevel_cat4', 'educationLevel_cat4',
            'Region', 'Region', 'Region', 'Region', 'Region', 'Region', 'Region', 'Region', 'Region', 'Region', 'Region', 'Region', 
            'Combined_IMD_Quintile', 'Combined_IMD_Quintile', 'Combined_IMD_Quintile', 'Combined_IMD_Quintile', 'Combined_IMD_Quintile',
            
            'Combined_BMI_cat5', 'Combined_BMI_cat5', 'Combined_BMI_cat5', 'Combined_BMI_cat5',
            'ZOE_conditions_condition_count_cat3', 'ZOE_conditions_condition_count_cat3', 'ZOE_conditions_condition_count_cat3',
            'ZOE_mentalhealth_condition_cat4', 'ZOE_mentalhealth_condition_cat4', 'ZOE_mentalhealth_condition_cat4', 'ZOE_mentalhealth_condition_cat4',
            'PRISMA7_score', 'PRISMA7_score', 'PRISMA7_score', 'PRISMA7_score', 'PRISMA7_score', 'PRISMA7_score', 'PRISMA7_score',
            
            'round1_'+'result_stringencylimit'+stringency, 'round1_'+'result_stringencylimit'+stringency,
            'round1_'+'symptomduration_grouped1_stringencylimit'+stringency,
            'round1_'+'symptomduration_grouped1_stringencylimit'+stringency,
            'round1_'+'symptomduration_grouped1_stringencylimit'+stringency,
            'round1_'+'symptomduration_grouped1_stringencylimit'+stringency,
            'round1_'+'symptomduration_grouped1_stringencylimit'+stringency,
            'round1_'+'symptomduration_grouped1_stringencylimit'+stringency,
            'round1_'+'symptomduration_grouped1_stringencylimit'+stringency,
            'round1_'+'symptomduration_grouped1_stringencylimit'+stringency,
            'round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,
            'round1_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,
            
            'round2_'+'result_stringencylimit'+stringency, 'round2_'+'result_stringencylimit'+stringency,
            'round2_'+'symptomduration_grouped1_stringencylimit'+stringency,
            'round2_'+'symptomduration_grouped1_stringencylimit'+stringency,
            'round2_'+'symptomduration_grouped1_stringencylimit'+stringency,
            'round2_'+'symptomduration_grouped1_stringencylimit'+stringency,
            'round2_'+'symptomduration_grouped1_stringencylimit'+stringency,
            'round2_'+'symptomduration_grouped1_stringencylimit'+stringency,
            'round2_'+'symptomduration_grouped1_stringencylimit'+stringency,
            'round2_'+'symptomduration_grouped1_stringencylimit'+stringency,
            'round2_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,
            'round2_'+'Flag_InHospitalDuringSpell_stringencylimit'+stringency,
            
            'Biobank_LCQ_B10_Recovered', 'Biobank_LCQ_B10_Recovered', 'Biobank_LCQ_B10_Recovered', 'Biobank_LCQ_B10_Recovered',
            'WeeksBetween_Cognitron_SymptomORTest',
            'WaveAt_SymptomORTest', 'WaveAt_SymptomORTest', 'WaveAt_SymptomORTest', 'WaveAt_SymptomORTest', 'WaveAt_SymptomORTest', 'WaveAt_SymptomORTest', 'WaveAt_SymptomORTest', 'WaveAt_SymptomORTest', 'WaveAt_SymptomORTest', 'WaveAt_SymptomORTest',
            
            'q_PHQ4_cat4', 'q_PHQ4_cat4', 'q_PHQ4_cat4', 'q_PHQ4_cat4',
            'q_chalderFatigue_cat2', 'q_chalderFatigue_cat2',
            'q_WSAS_cat4', 'q_WSAS_cat4', 'q_WSAS_cat4',
            ]

# Specify category
var_list_category = ['1: 0-30', '2: 30-40', '3: 40-50', '4: 50-60', '5: 60-70', '6: 70-80', '7: 80+',
                     'Female', 'Male', 
                     'Asian or Asian British', 'Black or Black British', 'Mixed or multiple ethnic groups', 'Any other ethnic group', 'White',
                     '1. October-November 2020 COVID-19 invitation', '2. May 2021 COVID-19 invitation', '3. May 2021 Healthy control COVID-19 invitation',
                     '3. Postgraduate degree or higher', '2. Undergraduate degree', '1. Less than degree level', '0. Other/ prefer not to say', 
                     'East Midlands', 'East of England', 'London', 'North East', 'North West', 'Northern Ireland', 'Scotland', 'South East', 'South West', 'Wales', 'West Midlands', 'Yorkshire and The Humber',
                     1.0, 2.0, 3.0, 4.0, 5.0,
                     '1: 0-18.5', '2: 18.5-25', '3: 25-30', '4: 30+', 
                     '0 conditions', '1 condition', '2+ conditions', 
                     '0 conditions', '1 condition', '2 conditions', '3+ conditions', 
                     0.0,1.0,2.0,3.0,4.0,5.0,6.0, # PRISMA
                     3.0, 4.0,
                     'N1: Negative COVID-19, asymptomatic (healthy control)', 
                     'N2&3: Negative COVID-19, 0-4 weeks', 'N4&5: Negative COVID-19, 4-12 weeks',
                     'N6: Negative COVID-19, 12+ weeks', 
                     'P1: Positive COVID-19, asymptomatic', 
                     'P2&3: Positive COVID-19, 0-4 weeks', 'P4&5: Positive COVID-19, 4-12 weeks',
                     'P6: Positive COVID-19, 12+ weeks',
                     0.0, 1.0,
                     
                     3.0, 4.0,
                     'N1: Negative COVID-19, asymptomatic (healthy control)', 
                     'N2&3: Negative COVID-19, 0-4 weeks', 'N4&5: Negative COVID-19, 4-12 weeks',
                     'N6: Negative COVID-19, 12+ weeks', 
                     'P1: Positive COVID-19, asymptomatic', 
                     'P2&3: Positive COVID-19, 0-4 weeks', 'P4&5: Positive COVID-19, 4-12 weeks',
                     'P6: Positive COVID-19, 12+ weeks',
                     0.0, 1.0,
                     
                     'NA_covid_negative', 'no', 'yes', 'Unknown', 
                     '',
                     '2020 Q1 Jan-Mar','2020 Q2 Apr-Jun','2020 Q3 Jul-Sep','2020 Q4 Oct-Dec','2021 Q1 Jan-Mar','2021 Q2 Apr-Jun','2021 Q3 Jul-Sep', '2021 Q4 Oct-Dec','2022 Q1 Jan-Mar','2022 Q2 Apr-Jun',
                     
                     '1. 0-2, below threshold', '2. 3-5, mild', '3. 6-8, moderate', '4. 9-12, severe',
                     '1. 0-28, below threshold', '2. 29-33, above threshold', 
                     '1. 0-9, below threshold', '2. 10-20, mild', '3. 21-40, moderate to severe',
                     ]

var_list_metric = ['%', '%', '%', '%', '%', '%', '%',
                   '%', '%',
                   '%', '%', '%', '%', '%',
                   '%', '%', '%', 
                   '%', '%', '%', '%',
                   '%', '%', '%', '%', '%', '%', '%', '%', '%', '%', '%', '%',
                   '%', '%', '%', '%', '%',
                   '%', '%', '%', '%',
                   '%', '%', '%',
                   '%', '%', '%', '%',
                   '%', '%', '%', '%', '%', '%', '%', # PRISMA
                   
                   '%', '%',
                   '%', '%', '%', '%', '%', '%', '%', '%',
                   '%', '%',
                   
                   '%', '%',
                   '%', '%', '%', '%', '%', '%', '%', '%', 
                   '%', '%',
                   
                   '%', '%', '%', '%',
                   'median',
                   '%', '%', '%', '%', '%', '%', '%', '%', '%', '%',
                   '%', '%', '%', '%',
                   '%', '%',
                   '%', '%', '%',
                   ]

# -----------------------------------------------------------------------------
# List of datasets to create characteristics summary table for
data_list = [# All invited to both rounds
             [data_cognitron_round1_invited, var_list, var_list_category, var_list_metric, 'Invited to Round 1'],
             [data_cognitron_round2_invited, var_list, var_list_category, var_list_metric, 'Invited to Round 2'],
             [data_cognitron_round1_participated_full, var_list, var_list_category, var_list_metric, 'Round 1 (full completion)'],
             [data_cognitron_round2_participated_full, var_list, var_list_category, var_list_metric, 'Round 2 (full completion)'],
             [data_cognitron_round2_participated_full_filter, var_list, var_list_category, var_list_metric, 'Full completion of both Round 1 and 2'],
             [data_cognitron_round2_participated_full_filter_NoChange[(data_cognitron_round2_participated_full_filter_NoChange['round2_result_stringencylimit7'] == 4)], var_list, var_list_category, var_list_metric, 'Full completion of both Round 1 and 2 [No change in group between rounds, SARS-CoV-2 positive only]'],
             
             [data_cognitron_round1_participated_full[data_cognitron_round1_participated_full['round1_symptomduration_grouped1_stringencylimit'+stringency].str.contains('N1')], var_list, var_list_category, var_list_metric, 'Round 1 (full completion), N1'],
             [data_cognitron_round1_participated_full[data_cognitron_round1_participated_full['round1_symptomduration_grouped1_stringencylimit'+stringency].str.contains('N2')], var_list, var_list_category, var_list_metric, 'Round 1 (full completion), N2&3'],
             [data_cognitron_round1_participated_full[data_cognitron_round1_participated_full['round1_symptomduration_grouped1_stringencylimit'+stringency].str.contains('N4')], var_list, var_list_category, var_list_metric, 'Round 1 (full completion), N4&5'],
             [data_cognitron_round1_participated_full[data_cognitron_round1_participated_full['round1_symptomduration_grouped1_stringencylimit'+stringency].str.contains('N6')], var_list, var_list_category, var_list_metric, 'Round 1 (full completion), N6'],
             
             [data_cognitron_round1_participated_full[data_cognitron_round1_participated_full['round1_symptomduration_grouped1_stringencylimit'+stringency].str.contains('P1')], var_list, var_list_category, var_list_metric, 'Round 1 (full completion), P1'],
             [data_cognitron_round1_participated_full[data_cognitron_round1_participated_full['round1_symptomduration_grouped1_stringencylimit'+stringency].str.contains('P2')], var_list, var_list_category, var_list_metric, 'Round 1 (full completion), P2&3'],
             [data_cognitron_round1_participated_full[data_cognitron_round1_participated_full['round1_symptomduration_grouped1_stringencylimit'+stringency].str.contains('P4')], var_list, var_list_category, var_list_metric, 'Round 1 (full completion), P4&5'],
             [data_cognitron_round1_participated_full[data_cognitron_round1_participated_full['round1_symptomduration_grouped1_stringencylimit'+stringency].str.contains('P6')], var_list, var_list_category, var_list_metric, 'Round 1 (full completion), P6'],
             
             
             
             ]

# -----------------------------------------------------------------------------
# Loop through variables and calculate median and IQR for continuous and N and % for categorical
for i in range(0,len(data_list),1):
    data = data_list[i][0].copy()
    var_test_all = data_list[i][1]
    var_test_all_category = data_list[i][2]
    var_test_all_metric = data_list[i][3]
    data_name = data_list[i][4]
    
    var_name_list = []
    var_metric_list = []
    
    var_only_list = []
    cat_only_list = []
    metric_string_list = []
    count_only_list = []
    
    for n in range(0,len(var_test_all),1):
        var = var_test_all[n]
        metric= var_test_all_metric[n]
        selected_cat = var_test_all_category[n]
        # print(selected_cat)
        data_slice = data[var].copy()
           
        # Convert missing data to NaN
        data_slice.loc[data_slice.isin(missing_data_values)] = np.nan
    
        # Count of overall, missing and non-missing values
        count_all = len(data_slice)
        count_non_missing = data_slice.count()
        count_missing = data_slice.isnull().sum() 
        count_missing_pct = ((count_missing/count_all) * 100).round(1)
        
        # tidy_string_missing = str(count_missing) + '/' + str(count_all) + ' (' + str(count_missing_pct) + '%)'
        tidy_string_missing = str(count_missing) + ' (' + str(count_missing_pct) + '%)'
        
        # Append string giving count of missing data, onyl if variable name is different to previous iteration of loop
        if var_test_all[n] != var_test_all[n-1]:
            var_name_list.append(var + ': Data not available, n (%)')
            var_metric_list.append(tidy_string_missing)
            
            var_only_list.append(var)
            metric_string_list.append('N (%)')
            cat_only_list.append('Data not available')
            count_only_list.append(count_missing)
            
        # Calculate frequency of different values 
        count_values = data_slice.value_counts()
        count_values_norm = data_slice.value_counts(normalize = True)
        
        if metric == '%':
            if selected_cat in data_slice.unique():
                selected_cat = var_test_all_category[n]
                count_selected = count_values[selected_cat]
                count_selected_pct = (count_values_norm[selected_cat] * 100).round(1)
                # tidy_string = str(count_selected) + '/' + str(count_non_missing) + ' (' + str(count_selected_pct) + '%)'
                if count_selected < 5:
                    tidy_string = '< 5'
                else:
                    tidy_string = str(count_selected) + ' (' + str(count_selected_pct) + '%)'
            elif selected_cat not in data_slice.unique():
                # tidy_string = str(0) + '/' + str(count_non_missing)
                tidy_string = str(0)
            var_name_list.append(var + ': ' + str(selected_cat) + ', n (%)')
            var_metric_list.append(tidy_string)
            
            var_only_list.append(var)
            cat_only_list.append(str(selected_cat))
            metric_string_list.append('N (%)')
            count_only_list.append(count_selected)
            
        elif metric == 'median':
            # Convert to numeric to remove missing data values
            data_slice = pd.to_numeric(data_slice, errors = 'coerce')
            median_IQR = data_slice.quantile(q = [0.5, 0.25, 0.75])
            tidy_string = str(median_IQR[0.5].round(round_dp)) + ' (' + str(median_IQR[0.25].round(round_dp)) + ', ' + str(median_IQR[0.75].round(round_dp)) + ')'
            
            var_name_list.append(var + ': Median (IQR)')
            var_metric_list.append(tidy_string)
            
            var_only_list.append(var)
            cat_only_list.append('')
            metric_string_list.append('Median (IQR)')
            
            count_only_list.append(count_all)
                        
    
    metric_df = pd.DataFrame(list(zip(var_name_list, var_metric_list, var_only_list, cat_only_list, metric_string_list)),
                             columns =['Variable combined', data_name, 'Variable', 'Category', 'Metric'])
    
    metric_df_count_only = pd.DataFrame(list(zip(var_name_list, count_only_list, var_only_list, cat_only_list)),
                             columns =['Variable combined', data_name, 'Variable', 'Category'])
       
    # Apply dictionary
    if i == 0:
        metric_df_all = metric_df
        metric_df_all_count_only = metric_df_count_only
    if i > 0:
        metric_df_all = pd.merge(metric_df_all, metric_df, how = 'outer', on = ['Variable combined', 'Variable', 'Category', 'Metric'])
        metric_df_all_count_only = pd.merge(metric_df_all_count_only, metric_df_count_only, how = 'outer', on = ['Variable combined', 'Variable', 'Category'])

### Tidy string table
# Combine variable and category name in same way as dummy variables
metric_df_all['Variable_Category'] = metric_df_all['Variable'].astype(str) + '_' + metric_df_all['Category'].astype(str)
metric_df_all['Variable_Category_tidy'] = metric_df_all['Variable_Category'].map(dictionary['variable_tidy'])
metric_df_all['Variable_Category_tidy'] = metric_df_all['Variable_Category_tidy'].fillna(metric_df_all['Variable_Category'])

metric_df_all['Variable_Category_tidy_metric'] = metric_df_all['Variable_Category_tidy'] + ', ' + metric_df_all['Metric']

# Split variable and category
metric_df_all[['Variable','Category']] = metric_df_all['Variable_Category_tidy'].str.split(": ",expand=True,)

metric_df_all_tidy = metric_df_all[['Variable_Category_tidy','Variable_Category_tidy_metric', 'Variable', 'Category', 'Metric', 'Invited to Round 1','Invited to Round 2', 'Round 1 (full completion)','Round 2 (full completion)','Full completion of both Round 1 and 2']]

### Count only table
# Combine variable and category name in same way as dummy variables
metric_df_all_count_only['Variable_Category'] = metric_df_all_count_only['Variable'].astype(str) + '_' + metric_df_all_count_only['Category'].astype(str)
metric_df_all_count_only['Variable_Category_tidy'] = metric_df_all_count_only['Variable_Category'].map(dictionary['variable_tidy'])
metric_df_all_count_only['Variable_Category_tidy'] = metric_df_all_count_only['Variable_Category_tidy'].fillna(metric_df_all_count_only['Variable_Category'])

# Split variable and category
metric_df_all_count_only[['Variable','Category']] = metric_df_all_count_only['Variable_Category_tidy'].str.split(": ",expand=True,)

# Add variable order column
metric_df_all_count_only['Variable_order'] = metric_df_all_count_only['Variable_Category'].map(dictionary['variable_order'])

metric_df_all_count_only_tidy = metric_df_all_count_only[['Variable_order', 'Variable', 'Category', 'Invited to Round 1','Invited to Round 2', 'Round 1 (full completion)','Round 2 (full completion)','Full completion of both Round 1 and 2']]



#%% Plot OLS coefficients on 1D plots
# -----------------------------------------------------------------------------
### Set figure style options
matplotlib.rcParams.update(matplotlib.rcParamsDefault) # Reset to default
plt.rcParams["font.family"] = "Arial" # 'Arial' is seaborn default. Change to nicer font "sans-serif" "cursive"


# -----------------------------------------------------------------------------
### Cognitive accuracy PCA component 1, Round 1 vs Round 2 full completion
##### All exposures
# Round 1 all
data_round1_accuracy_all = overall_round1_accuracy[(overall_round1_accuracy['outcome_variable'] == 'C1_R1model_accuracy_zscore')
                                & ((overall_round1_accuracy['model_name'] == '0_All_Exposures'))
                                ].copy()
# Add reference values
data_round1_accuracy_all = add_reference_values_posthoc(data = data_round1_accuracy_all, 
                             outcome_var = 'C1_R1model_accuracy_zscore')

# Round 1 (also Round 2)
data_round1_withround2_accuracy_round1group = overall_round1_withround2_accuracy_round1group[(overall_round1_withround2_accuracy_round1group['outcome_variable'] == 'C1_R1model_accuracy_zscore')
                                & ((overall_round1_withround2_accuracy_round1group['model_name'] == '0_All_Exposures'))
                                ].copy()
# Add reference values
data_round1_withround2_accuracy_round1group = add_reference_values_posthoc(data = data_round1_withround2_accuracy_round1group, 
                             outcome_var = 'C1_R1model_accuracy_zscore')

# Round 2 all
data_round2_accuracy_all = overall_round2_accuracy[(overall_round2_accuracy['outcome_variable'] == 'C1_R1model_accuracy_zscore')
                                & ((overall_round2_accuracy['model_name'] == '0_All_Exposures'))
                                ].copy()
# Add reference values
data_round2_accuracy_all = add_reference_values_posthoc(data = data_round2_accuracy_all, 
                             outcome_var = 'C1_R1model_accuracy_zscore')

# 1 series as vertical scatter - Round 1 only
data1 = plot_OLS_w_conf_int_1plot(data1 = data_round1_accuracy_all,
                          x_fieldname = 'Variable_tidy',
                          y_fieldname = 'coeff',
                          conf_int_fieldnames = ['conf_lower_error','conf_upper_error'],
                          plot1_label = 'Round 1',
                          xlims = [], 
                          ylims = [],
                          titlelabel = 'Cognitive accuracy: Principal Component 1', 
                          width = 5.5, 
                          height = 16,
                          offset = 0.2,
                          y_pos_manual = 'yes',
                          color_list = ['C2'],
                          fontsize = 12, 
                          legend_offset = -0.11,
                          invert_axis = 'yes', 
                          # y_tick_all_exposures = 'yes',
                          # size1 = 15, 
                          # size2 = 10
                          )

# 2 series as vertical scatter - Round 1 vs 2
data1, data2 = plot_OLS_w_conf_int_2plots(data1 = data_round1_accuracy_all,
                          data2 = data_round2_accuracy_all,
                          x_fieldname = 'Variable_tidy',
                          y_fieldname = 'coeff',
                          conf_int_fieldnames = ['conf_lower_error','conf_upper_error'],
                          plot1_label = 'Round 1',
                          plot2_label = 'Round 2', 
                          xlims = [], 
                          ylims = [],
                          titlelabel = 'Cognitive accuracy: Principal Component 1', 
                          width = 5.5, 
                          height = 17,
                          offset = 0.2,
                          y_pos_manual = 'yes',
                          color_list = ['C2','C1'],
                          fontsize = 12, 
                          legend_offset = -0.1,
                          invert_axis = 'yes', 
                          y_tick_all_exposures = 'yes',
                          size1 = 15, size2 = 10)

# 2 series as vertical scatter - Round 1 (all) vs 1 (also completed Round 2))
data1, data2 = plot_OLS_w_conf_int_2plots(data1 = data_round1_accuracy_all,
                          data2 = data_round1_withround2_accuracy_round1group,
                          x_fieldname = 'Variable_tidy',
                          y_fieldname = 'coeff',
                          conf_int_fieldnames = ['conf_lower_error','conf_upper_error'],
                          plot1_label = 'Round 1',
                          plot2_label = 'Round 1 (Round 2 also complete)', 
                          xlims = [], 
                          ylims = [],
                          titlelabel = 'Cognitive accuracy: Principal Component 1', 
                          width = 5.5, 
                          height = 17,
                          offset = 0.2,
                          y_pos_manual = 'yes',
                          color_list = ['C2','C4'],
                          fontsize = 12, 
                          legend_offset = -0.1,
                          invert_axis = 'yes', 
                          y_tick_all_exposures = 'yes',
                          size1 = 15, size2 = 10)


# -----------------------------------------------------------------------------
### Cognitive accuracy PCA component 1, Round 1 vs Round 2 full completion
##### COVID-19 related exposures only
data_round1_accuracy = overall_round1_accuracy[(overall_round1_accuracy['outcome_variable'] == 'C1_R1model_accuracy_zscore')
                                & ((overall_round1_accuracy['model_name'] == '0_All_Exposures'))
                                & (overall_round1_accuracy['Variable'].isin(var_result_round1+var_covidgroup_round1+var_covidseverity_round1))
                                ].copy()
# Add reference values
data_round1_accuracy = add_reference_values_posthoc(data = data_round1_accuracy, 
                             outcome_var = 'C1_R1model_accuracy_zscore')
data_round2_accuracy = overall_round2_accuracy[(overall_round2_accuracy['outcome_variable'] == 'C1_R1model_accuracy_zscore')
                                & ((overall_round2_accuracy['model_name'] == '0_All_Exposures'))
                                & (overall_round2_accuracy['Variable'].isin(var_result_round2+var_covidgroup_round2+var_covidseverity_round2))
                                ].copy()
# Add reference values
data_round2_accuracy = add_reference_values_posthoc(data = data_round2_accuracy, 
                             outcome_var = 'C1_R1model_accuracy_zscore')

# FIGURE 2
# 2 series as vertical scatter
data1, data2 = plot_OLS_w_conf_int_2plots(data1 = data_round1_accuracy,
                          data2 = data_round2_accuracy,
                          x_fieldname = 'Variable_tidy',
                          y_fieldname = 'coeff',
                          conf_int_fieldnames = ['conf_lower_error','conf_upper_error'],
                          plot1_label = 'Round 1',
                          plot2_label = 'Round 2', 
                          xlims = [-0.5,0.3], 
                          ylims = [],
                          titlelabel = 'Cognitive accuracy: Principal Component 1', 
                          width = 5.2, 
                          height = 7,
                          offset = 0.175,
                          y_pos_manual = 'yes',
                          color_list = ['C2','C1'],
                          fontsize = 12, 
                          legend_offset = -0.22,
                          invert_axis = 'yes',
                          y_tick_all_exposures = '',
                          size1 = 22, size2 = 15)



##-----------------------------------------------------------------------------
### Accuracy PCA and individual tasks, Round 1 vs Round 2 full completion, by Covid group
# EXAMPLE OF WHAT COULD DO - WOULD NEED TO REWORK THE 1 SERIES PLOT FUNCTION A BIT IN SEPARATE PROGRAM
sns.barplot(data = overall_round1_accuracy[((overall_round1_accuracy['model_name'] == '0_All_Exposures'))
                                           & (overall_round1_accuracy['Variable'].isin(['round1_'+'symptomduration_grouped1_stringencylimit'+stringency+'_P6: Positive COVID-19, 12+ weeks']))],
            x = 'outcome_variable', y = 'coeff', hue = 'Variable')


##-----------------------------------------------------------------------------
### Round 1 and 2 mediator models for covid group
# Round 1
data_round1_accuracy_mediators_none = overall_round1_accuracy[(overall_round1_accuracy['outcome_variable'] == 'C1_R1model_accuracy_zscore')
                                & (overall_round1_accuracy['model_name'] == '0_All_Exposures')
                                & (overall_round1_accuracy['Variable'].isin(var_covidgroup_round1))
                                ].copy()
# Add reference values
data_round1_accuracy_mediators_none = add_reference_values_posthoc(data = data_round1_accuracy_mediators_none, 
                             outcome_var = 'C1_R1model_accuracy_zscore')
data_round1_accuracy_mediators_PHQ4 = overall_round1_accuracy[(overall_round1_accuracy['outcome_variable'] == 'C1_R1model_accuracy_zscore')
                                & (overall_round1_accuracy['model_name'] == 'CovidGroup_Mediation_PHQ4')
                                & (overall_round1_accuracy['Variable'].isin(var_covidgroup_round1))
                                ].copy()
# Add reference values
data_round1_accuracy_mediators_PHQ4 = add_reference_values_posthoc(data = data_round1_accuracy_mediators_PHQ4, 
                             outcome_var = 'C1_R1model_accuracy_zscore')
data_round1_accuracy_mediators_Chalder = overall_round1_accuracy[(overall_round1_accuracy['outcome_variable'] == 'C1_R1model_accuracy_zscore')
                                & (overall_round1_accuracy['model_name'] == 'CovidGroup_Mediation_Fatigue')
                                & (overall_round1_accuracy['Variable'].isin(var_covidgroup_round1))
                                ].copy()
# Add reference values
data_round1_accuracy_mediators_Chalder = add_reference_values_posthoc(data = data_round1_accuracy_mediators_Chalder, 
                             outcome_var = 'C1_R1model_accuracy_zscore')
data_round1_accuracy_mediators_WSAS = overall_round1_accuracy[(overall_round1_accuracy['outcome_variable'] == 'C1_R1model_accuracy_zscore')
                                & (overall_round1_accuracy['model_name'] == 'CovidGroup_Mediation_WSAS')
                                & (overall_round1_accuracy['Variable'].isin(var_covidgroup_round1))
                                ].copy()
# Add reference values
data_round1_accuracy_mediators_WSAS = add_reference_values_posthoc(data = data_round1_accuracy_mediators_WSAS, 
                             outcome_var = 'C1_R1model_accuracy_zscore')
data_round1_accuracy_mediators_all = overall_round1_accuracy[(overall_round1_accuracy['outcome_variable'] == 'C1_R1model_accuracy_zscore')
                                & (overall_round1_accuracy['model_name'] == 'CovidGroup_Mediation_All')
                                & (overall_round1_accuracy['Variable'].isin(var_covidgroup_round1))
                                ].copy()
# Add reference values
data_round1_accuracy_mediators_all = add_reference_values_posthoc(data = data_round1_accuracy_mediators_all, 
                             outcome_var = 'C1_R1model_accuracy_zscore')

# 5 series as vertical scatter
data1, data2, data3, data4, data5 = plot_OLS_w_conf_int_5plots(data1 = data_round1_accuracy_mediators_none,
                                              data2 = data_round1_accuracy_mediators_PHQ4,
                          data3 = data_round1_accuracy_mediators_Chalder,
                          data4 = data_round1_accuracy_mediators_WSAS,
                          data5 = data_round1_accuracy_mediators_all,
                          x_fieldname = 'Variable_tidy',
                          y_fieldname = 'coeff',
                          conf_int_fieldnames = ['conf_lower_error','conf_upper_error'],
                          plot1_label = 'Round 1, Mediator: None',
                          plot2_label = 'Round 1, Mediator: PHQ-4', 
                          plot3_label = 'Round 1, Mediator: Chalder fatigue', 
                          plot4_label = 'Round 1, Mediator: WSAS',
                          plot5_label = 'Round 1, Mediator: PHQ-4 + Chalder + WSAS',
                          xlims = [-0.5,0.3], 
                          ylims = [],
                          titlelabel = 'Round 1 accuracy mediation models', 
                          width = 5, 
                          height = 6.5,
                          offset = 0.15,
                          y_pos_manual = 'yes',
                          color_list = ['C2','C3','C4','C9', 'C7'],
                          fontsize = 12,
                          bbox_to_anchor_vertical = -0.4,
                          invert_axis = 'yes',
                          alpha = 0.4
                          )


# Round 2
data_round2_accuracy_mediators_none = overall_round2_accuracy[(overall_round2_accuracy['outcome_variable'] == 'C1_R1model_accuracy_zscore')
                                & (overall_round2_accuracy['model_name'] == '0_All_Exposures')
                                & (overall_round2_accuracy['Variable'].isin(var_covidgroup_round2))
                                ].copy()
# Add reference values
data_round2_accuracy_mediators_none = add_reference_values_posthoc(data = data_round2_accuracy_mediators_none, 
                             outcome_var = 'C1_R1model_accuracy_zscore')
data_round2_accuracy_mediators_PHQ4 = overall_round2_accuracy[(overall_round2_accuracy['outcome_variable'] == 'C1_R1model_accuracy_zscore')
                                & (overall_round2_accuracy['model_name'] == 'CovidGroup_Mediation_PHQ4')
                                & (overall_round2_accuracy['Variable'].isin(var_covidgroup_round2))
                                ].copy()
# Add reference values
data_round2_accuracy_mediators_PHQ4 = add_reference_values_posthoc(data = data_round2_accuracy_mediators_PHQ4, 
                             outcome_var = 'C1_R1model_accuracy_zscore')
data_round2_accuracy_mediators_Chalder = overall_round2_accuracy[(overall_round2_accuracy['outcome_variable'] == 'C1_R1model_accuracy_zscore')
                                & (overall_round2_accuracy['model_name'] == 'CovidGroup_Mediation_Fatigue')
                                & (overall_round2_accuracy['Variable'].isin(var_covidgroup_round2))
                                ].copy()
# Add reference values
data_round2_accuracy_mediators_Chalder = add_reference_values_posthoc(data = data_round2_accuracy_mediators_Chalder, 
                             outcome_var = 'C1_R1model_accuracy_zscore')
data_round2_accuracy_mediators_WSAS = overall_round2_accuracy[(overall_round2_accuracy['outcome_variable'] == 'C1_R1model_accuracy_zscore')
                                & (overall_round2_accuracy['model_name'] == 'CovidGroup_Mediation_WSAS')
                                & (overall_round2_accuracy['Variable'].isin(var_covidgroup_round2))
                                ].copy()
# Add reference values
data_round2_accuracy_mediators_WSAS = add_reference_values_posthoc(data = data_round2_accuracy_mediators_WSAS, 
                             outcome_var = 'C1_R1model_accuracy_zscore')
data_round2_accuracy_mediators_all = overall_round2_accuracy[(overall_round2_accuracy['outcome_variable'] == 'C1_R1model_accuracy_zscore')
                                & (overall_round2_accuracy['model_name'] == 'CovidGroup_Mediation_All')
                                & (overall_round2_accuracy['Variable'].isin(var_covidgroup_round2))
                                ].copy()
# Add reference values
data_round2_accuracy_mediators_all = add_reference_values_posthoc(data = data_round2_accuracy_mediators_all, 
                             outcome_var = 'C1_R1model_accuracy_zscore')

# 5 series as vertical scatter
data1, data2, data3, data4, data5 = plot_OLS_w_conf_int_5plots(data1 = data_round2_accuracy_mediators_none,
                                              data2 = data_round2_accuracy_mediators_PHQ4,
                          data3 = data_round2_accuracy_mediators_Chalder,
                          data4 = data_round2_accuracy_mediators_WSAS,
                          data5 = data_round2_accuracy_mediators_all,
                          x_fieldname = 'Variable_tidy',
                          y_fieldname = 'coeff',
                          conf_int_fieldnames = ['conf_lower_error','conf_upper_error'],
                          plot1_label = 'Round 2, Mediator: None',
                          plot2_label = 'Round 2, Mediator: PHQ-4', 
                          plot3_label = 'Round 2, Mediator: Chalder fatigue', 
                          plot4_label = 'Round 2, Mediator: WSAS',
                          plot5_label = 'Round 2, Mediator: PHQ-4 + Chalder + WSAS',
                          xlims = [-0.5,0.3], 
                          ylims = [],
                          titlelabel = 'Round 2 accuracy mediation models', 
                          width = 5, 
                          height = 6.5,
                          offset = 0.15,
                          y_pos_manual = 'yes',
                          color_list = ['C1','C3','C4','C9', 'C7'],
                          fontsize = 12,
                          bbox_to_anchor_vertical = -0.4,
                          invert_axis = 'yes',
                          alpha = 0.4
                          )


##-----------------------------------------------------------------------------
### Stratifying Round 1 full by self-perceived recovery
data_round1_accuracy_stratify_recovered = overall_round1_accuracy_stratify_recovered[(overall_round1_accuracy_stratify_recovered['outcome_variable'] == 'C1_R1model_accuracy_zscore')
                                & ((overall_round1_accuracy_stratify_recovered['model_name'] == '0_All_Exposures'))
                                & (overall_round1_accuracy_stratify_recovered['Variable'].isin(var_result_round1+var_covidgroup_round1+var_covidseverity_round1))
                                ].copy()
# Add reference values
data_round1_accuracy_stratify_recovered = add_reference_values_posthoc(data = data_round1_accuracy_stratify_recovered, 
                             outcome_var = 'C1_R1model_accuracy_zscore')

data_round1_accuracy_stratify_notrecovered = overall_round1_accuracy_stratify_notrecovered[(overall_round1_accuracy_stratify_notrecovered['outcome_variable'] == 'C1_R1model_accuracy_zscore')
                                & ((overall_round1_accuracy_stratify_notrecovered['model_name'] == '0_All_Exposures'))
                                & (overall_round1_accuracy_stratify_notrecovered['Variable'].isin(var_result_round1+var_covidgroup_round1+var_covidseverity_round1))
                                ].copy()
# Add reference values
data_round1_accuracy_stratify_notrecovered = add_reference_values_posthoc(data = data_round1_accuracy_stratify_notrecovered, 
                             outcome_var = 'C1_R1model_accuracy_zscore')

# FIGURE 3
# 3 series as vertical scatter
data1, data2, data3 = plot_OLS_w_conf_int_3plots(data1 = data_round1_accuracy,
                                              data2 = data_round1_accuracy_stratify_recovered,
                          data3 = data_round1_accuracy_stratify_notrecovered,
                          x_fieldname = 'Variable_tidy',
                          y_fieldname = 'coeff',
                          conf_int_fieldnames = ['conf_lower_error','conf_upper_error'],
                          plot1_label = 'Round 1 (All)',
                          plot2_label = 'Round 1 (Self-perceived COVID-19 recovery: Yes)', 
                          plot3_label = 'Round 1 (Self-perceived COVID-19 recovery: No)', 
                          xlims = [-0.8,0.4], 
                          ylims = [],
                          titlelabel = 'Round 1 accuracy stratified by COVID-19 recovery', 
                          width = 5, 
                          height = 7.5,
                          offset = 0.25,
                          y_pos_manual = 'yes',
                          color_list = ['C2','C6','C8'],
                          fontsize = 12,
                          bbox_to_anchor_vertical = -0.25,
                          invert_axis = 'yes',
                          alpha = 0.4
                          )


##-----------------------------------------------------------------------------
### JAN 23 UPDATED SUBSETS - Models using R1 as mediator to test change between rounds for Round 2 (with Round 1), stratified by change in covid group between rounds
# overall_round2_withround1_accuracy_Subset1_RemainedNeg
# overall_round2_withround1_accuracy_Subset2_PosNotRecovered
# overall_round2_withround1_accuracy_Subset3_PosRecovered
# overall_round2_withround1_accuracy_Subset4_NewPos

# SUBSET 1
data_round2_withround1_accuracy_Subset1_RemainedNeg = overall_round2_withround1_accuracy_Subset1_RemainedNeg[(overall_round2_withround1_accuracy_Subset1_RemainedNeg['outcome_variable'] == 'C1_R1model_accuracy_zscore')
                                & ((overall_round2_withround1_accuracy_Subset1_RemainedNeg['model_name'] == 'CovidGroup_Mediation_Round1'))
                                & (overall_round2_withround1_accuracy_Subset1_RemainedNeg['Variable'].isin(var_covidgroup_round2))
                                ].copy()
# Add reference values
data_round2_withround1_accuracy_Subset1_RemainedNeg = add_reference_values_posthoc(data = data_round2_withround1_accuracy_Subset1_RemainedNeg, 
                             outcome_var = 'C1_R1model_accuracy_zscore')

# SUBSET 2
data_round2_withround1_accuracy_Subset2_PosNotRecovered = overall_round2_withround1_accuracy_Subset2_PosNotRecovered[(overall_round2_withround1_accuracy_Subset2_PosNotRecovered['outcome_variable'] == 'C1_R1model_accuracy_zscore')
                                & ((overall_round2_withround1_accuracy_Subset2_PosNotRecovered['model_name'] == 'CovidGroup_Mediation_Round1'))
                                & (overall_round2_withround1_accuracy_Subset2_PosNotRecovered['Variable'].isin(var_covidgroup_round2))
                                ].copy()
# Add reference values
data_round2_withround1_accuracy_Subset2_PosNotRecovered = add_reference_values_posthoc(data = data_round2_withround1_accuracy_Subset2_PosNotRecovered, 
                             outcome_var = 'C1_R1model_accuracy_zscore')

# SUBSET 3
data_round2_withround1_accuracy_Subset3_PosRecovered = overall_round2_withround1_accuracy_Subset3_PosRecovered[(overall_round2_withround1_accuracy_Subset3_PosRecovered['outcome_variable'] == 'C1_R1model_accuracy_zscore')
                                & ((overall_round2_withround1_accuracy_Subset3_PosRecovered['model_name'] == 'CovidGroup_Mediation_Round1'))
                                & (overall_round2_withround1_accuracy_Subset3_PosRecovered['Variable'].isin(var_covidgroup_round2))
                                ].copy()
# Add reference values
data_round2_withround1_accuracy_Subset3_PosRecovered = add_reference_values_posthoc(data = data_round2_withround1_accuracy_Subset3_PosRecovered, 
                             outcome_var = 'C1_R1model_accuracy_zscore')

# SUBSET 4
data_round2_withround1_accuracy_Subset4_NewPos = overall_round2_withround1_accuracy_Subset4_NewPos[(overall_round2_withround1_accuracy_Subset4_NewPos['outcome_variable'] == 'C1_R1model_accuracy_zscore')
                                & ((overall_round2_withround1_accuracy_Subset4_NewPos['model_name'] == 'CovidGroup_Mediation_Round1'))
                                & (overall_round2_withround1_accuracy_Subset4_NewPos['Variable'].isin(var_covidgroup_round2))
                                ].copy()
# Add reference values
data_round2_withround1_accuracy_Subset4_NewPos = add_reference_values_posthoc(data = data_round2_withround1_accuracy_Subset4_NewPos, 
                             outcome_var = 'C1_R1model_accuracy_zscore')

# FIGURE 4
# 3 series as vertical scatter
data1, data2, data3, data4 = plot_OLS_w_conf_int_4plots(data1 = data_round2_withround1_accuracy_Subset1_RemainedNeg,
                          data2 = data_round2_withround1_accuracy_Subset2_PosNotRecovered,
                          data3 = data_round2_withround1_accuracy_Subset3_PosRecovered,
                          data4 = data_round2_withround1_accuracy_Subset4_NewPos,                          
                          x_fieldname = 'Variable_tidy',
                          y_fieldname = 'coeff',
                          conf_int_fieldnames = ['conf_lower_error','conf_upper_error'],
                          plot1_label = 'Subset 1: Remained SARS-CoV-2 negative between Round 1 and 2',
                          plot2_label = 'Subset 2: SARS-CoV-2 positive before Round 1 (Not recovered at Round 1)', 
                          plot3_label = 'Subset 3: SARS-CoV-2 positive before Round 1 (Recovered at Round 1)', 
                          plot4_label = 'Subset 4: Newly SARS-CoV-2 positive between Round 1 and 2', 
                          xlims = [-0.7,0.7], 
                          ylims = [],
                          titlelabel = 'Change in accuracy between Round 1 and 2', 
                          width = 5, 
                          height = 5,
                          offset = 0.2,
                          y_pos_manual = 'yes',
                          color_list = ['C1','C3','C4','C9'],
                          fontsize = 12,
                          bbox_to_anchor_vertical = -0.45,
                          invert_axis = 'yes',
                          alpha = 0.4
                          )



#%% MISCELLANEOUS Other statistical tests and analysis
# -----------------------------------------------------------------------------
# Correlation between symptom duration and self-reported COVID-19 recovery
### Round 1
# Filter for positive only, and those with self-reported response
data_recovery_filter_round1 = data_cognitron_round1_participated_full[
(data_cognitron_round1_participated_full['round1_'+'symptomduration_grouped1_stringencylimit'+stringency].str.contains('Positive')) & (data_cognitron_round1_participated_full['Biobank_LCQ_B10_Recovered'].isin(['no', 'yes']))
].copy()

# Calculate proportion with recovery
# x = 'round1_'+'symptomduration_grouped1_stringencylimit'+stringency
x = 'round1_'+'symptomduration_weeks_stringencylimit'+stringency
y = 'Biobank_LCQ_B10_Recovered'

test = data_recovery_filter_round1.groupby([x, y]).agg({'cssbiobank_id':'count'}).reset_index()
test_pivot = test.pivot_table(index = x, columns = y, values = 'cssbiobank_id').reset_index()
test_pivot['yes'] = test_pivot['yes'].fillna(0)
test_pivot['total'] = test_pivot['no'] + test_pivot['yes']
test_pivot['prop'] = test_pivot['yes']/test_pivot['total']

# filter out rows with less than 5 
test_pivot_filter = test_pivot[test_pivot['total'] >= 5]

# Do spearman correlation test
spearmanr(test_pivot_filter[x], test_pivot_filter['prop'])