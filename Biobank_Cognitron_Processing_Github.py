# -*- coding: utf-8 -*-
"""
Created on Thu May 19 13:54:15 2022

@author: k2143494

Processing and analysing raw data received from Imperial Cognitron team
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
plt.rc("font", size=12)

export_csv = 0


#%% Define functions

#%% Load data
# -----------------------------------------------------------------------------
# Participant master list - including those who have withdrawn
fname = r"C:\Users\k2143494\OneDrive - King's College London\Documents\CSS Biobank\REDCap extracts\CSSBiobankRegistered-MASTERlistinclwithdr_DATA_2022-01-20_1530.csv"
patientlist = pd.read_csv(fname)
patientlist['cssbiobank_id'] = patientlist['cssbiobank_id'].str.replace("c","C") # make all C capitalised in CSS biobank id

# Withdrawal list - participants who have withdrawn 
fname = r"C:\Users\k2143494\OneDrive - King's College London\Documents\CSS Biobank\REDCap extracts\CSSBiobankRegistered-Withdrawnpii_DATA_2022-10-11_0921.csv"
patientlist_withdrawals = pd.read_csv(fname)
patientlist_withdrawals['cssbiobank_id'] = patientlist_withdrawals['cssbiobank_id'].str.replace("c","C") # make all C capitalised in CSS biobank id

# Identify participants whose data must be deleted. delete all participants apart from those with withdraw_type: 1 = Permission to continue to use previously collected samples and data (2 = Destroy all samples and data CSS Biobank holds, 9 = not specified)
delete_list = patientlist_withdrawals[~(patientlist_withdrawals['withdraw_type'].isin([1,9]))]['cssbiobank_id'].to_list()
# Filter out participants who asked for data to be deleted
patientlist = patientlist[~(patientlist['cssbiobank_id'].isin(delete_list))]

# Select only patient id columns
patientlist = patientlist[['cssbiobank_id','zoepatient_id']]


# -----------------------------------------------------------------------------
# Import file of raw cognitron data
download_date = '2022-07-19' 
data_raw_json = pd.read_json(r"C:\Users\k2143494\OneDrive - King's College London\Documents\CSS Biobank\Cognitron study\Raw data\tmp9j2jptcb.json", lines = True)

# -----------------------------------------------------------------------------
# Import mapping file of CSS Biobank ID and cognitron link IDs
link_id = pd.read_csv(r"C:\Users\k2143494\OneDrive - King's College London\Documents\CSS Biobank\Study participants data\CSSB_D_cognition_full_tracking_with_links_ids_2022-06-06_all_ids.csv")

# -----------------------------------------------------------------------------
# Import tracking file containing information of technical errors
tracking = pd.read_csv(r"C:\Users\k2143494\OneDrive - King's College London\Documents\CSS Biobank\Study participants data\CSSB_D_cognition_full_tracking_with_links_ids_2022-06-06.csv")


# -----------------------------------------------------------------------------
# Import device lookup table
# device_lookup = pd.read_csv(r"C:\Users\k2143494\OneDrive - King's College London\Documents\CSS Biobank\Cognitron study\Imperial scripts\devices_lookup_separated.csv")
# Import device lookup table generated from manual assignment using 2022-06-06 data 
device_lookup = pd.read_csv(r"C:\Users\k2143494\OneDrive - King's College London\Documents\CSS Biobank\Cognitron study\Imperial scripts\devices_lookup_NCversion_dataset_20220606.csv")



#%% Dictionaries containing relevant information and mapping for cognitron tasks
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

# Assign order for each task based on order completed
dict_data['task_number'] = {'rs_prospectiveMemoryWords_1_immediate':1,
                           'rs_prospectiveMemoryObjects_1_immediate':2,
                           'rs_motorControl':3,
                           'rs_manipulations2D':4,
                           'rs_targetDetection':5,
                           'rs_spatialSpan':6,
                           'rs_TOL':7,
                           'rs_verbalAnalogies':8,
                           'rs_prospectiveMemoryWords_1_delayed':9,
                           'rs_prospectiveMemoryObjects_1_delayed':10,
                           'rs_PAL':11,
                           'rs_CRT':12,
                           }

dict_data['task_number_reverse'] = dict((v, k) for k, v in dict_data['task_number'].items())

# Task number as string
dict_data['task_number_string'] = {'rs_prospectiveMemoryWords_1_immediate':'01',
                           'rs_prospectiveMemoryObjects_1_immediate':'02',
                           'rs_motorControl':'03',
                           'rs_manipulations2D':'04',
                           'rs_targetDetection':'05',
                           'rs_spatialSpan':'06',
                           'rs_TOL':'07',
                           'rs_verbalAnalogies':'08',
                           'rs_prospectiveMemoryWords_1_delayed':'09',
                           'rs_prospectiveMemoryObjects_1_delayed':'10',
                           'rs_PAL':'11',
                           'rs_CRT':'12',
                           }

# Map which fields were primary accuracy metric for each task
dict_data['primary_accuracy'] = {'rs_prospectiveMemoryWords_1_immediate':'Scores.totalCorrect',
                                'rs_prospectiveMemoryObjects_1_immediate':'SummaryScore',
                                'rs_motorControl':'Scores.meanDistance',
                                'rs_manipulations2D':'Scores.score',
                                'rs_targetDetection':'Scores.correctResponses',
                                'rs_spatialSpan':'Scores.maxAchieved',
                                'rs_TOL':'Scores.numCorrect',
                                'rs_verbalAnalogies':'SummaryScore',
                                'rs_prospectiveMemoryWords_1_delayed':'Scores.totalCorrect',
                                'rs_prospectiveMemoryObjects_1_delayed':'SummaryScore',
                                'rs_PAL':'Scores.maxAchieved',
                                'rs_CRT':'Scores.numCorrProportion', # Manual
                                }

# Map which fields measure average reaction time for each task
# NOTE: Hampshire et al uses the target correct RT for memory tasks 1, 2, 9, 10 only, and general median RT (from both correct + incorrect) in other tasks. Not clear why this distinction made, so use overall median RT for consistency unless I find out reasonable justification. 
# Replace automatically generated average fields with manually generated, for tasks where automatically generated field contains some nulls but manual value is not null - TOL & CRT
dict_data['reaction_time_average'] = {'rs_prospectiveMemoryWords_1_immediate':'Scores.medianRT.manual', # 'Scores.target_correct_medianRT'
                                      'rs_prospectiveMemoryObjects_1_immediate':'Scores.medianRT.manual', # Manually generated  'Scores.target_correct_medianRT.manual'
                                     'rs_motorControl':'Scores.meanRT',
                                     'rs_manipulations2D':'Scores.medRT',
                                     'rs_targetDetection':'Scores.meanRT.manual', # Previously used the automatically created 'Scores.meanRT' but think this is mislabelled as gives same values as manual median calculation
                                     'rs_spatialSpan':'Scores.medRT',
                                     'rs_TOL':'Scores.medianRT.manual', # 'Scores.medianRT',
                                     'rs_verbalAnalogies':'Scores.medRT',
                                     'rs_prospectiveMemoryWords_1_delayed':'Scores.medianRT.manual', # 'Scores.target_correct_medianRT',
                                     'rs_prospectiveMemoryObjects_1_delayed':'Scores.medianRT.manual', # 'Scores.target_correct_medianRT.manual', # Manually generated 
                                     'rs_PAL':'Scores.medRT',
                                     'rs_CRT':'Scores.medianRT.manual' # 'Scores.medianCRT',
                                     }

# Map which fields measure variation in reaction time metric for each task
# Where reaction time average is mean, use standard deviation, where median, use IQR.
dict_data['reaction_time_variation'] = {'rs_prospectiveMemoryWords_1_immediate':'Scores.RT_IQR', # 'Scores.target_correct_RT_IQR',
                                        'rs_prospectiveMemoryObjects_1_immediate':'Scores.RT_IQR',  # 'Scores.target_correct_RT_IQR', # Manually generated 
                                        'rs_motorControl':'Scores.stdevRT',
                                        'rs_manipulations2D':'Scores.RT_IQR',
                                        'rs_targetDetection':'Scores.stdevRT',
                                        'rs_spatialSpan':'Scores.RT_IQR',
                                        'rs_TOL':'Scores.RT_IQR',
                                        'rs_verbalAnalogies':'Scores.RT_IQR',
                                        'rs_prospectiveMemoryWords_1_delayed':'Scores.RT_IQR', # 'Scores.target_correct_RT_IQR',
                                        'rs_prospectiveMemoryObjects_1_delayed':'Scores.RT_IQR', # 'Scores.target_correct_RT_IQR', # Manually generated 
                                        'rs_PAL':'Scores.RT_IQR',
                                        'rs_CRT':'Scores.RT_IQR',
                                        }


# Dictionary of column headers in Rawdata column of JSON
dict_data['dict_headers_rawdata'] = {'rs_prospectiveMemoryWords_1_immediate':['delayedRecall','Level','word','target','category','correct','Time Resp Enabled','TimeRespMade','RT'],
                        'rs_prospectiveMemoryObjects_1_immediate':['Level', 'targetCategory', 'gridItems', 'gridColours', 'gridOrientations','targetGridIdx', 'responseGridIdx', 'targetOrderId', 'correct', 'Time Resp Enabled', 'TimeRespMade','RT'],
                        'rs_motorControl':['Level','target X','target Y', 'response X','response Y','distance','Time Resp Enabled', 'Time Resp Made', 'RT'],
                        'rs_manipulations2D':['Trial','Num Correct','num Incorrect','score','angles','Target Array','Probe Array 0','Probe Array 1','Probe Array 2','Probe Array 3','Answer','Response','correct','Time Resp Enabled','TimeRespMade','RT'],
                        'rs_targetDetection':['button','X','Y','correct','RT','score','totaldetected','total_targets'],
                        'rs_spatialSpan':['Target','Response', 'correct', 'Time Resp Enabled','TimeRespMade','RT'],
                        'rs_TOL':['Level','Probe Config','Target Config','convolution','N Balls','answer','response','correct','Time Resp Enabled','TimeRespMade','RT'],
                        'rs_verbalAnalogies':['Level','Target','Probe','Code','correctAnswer','Accuracy','Score','Time Resp Enabled','Time Resp Made','RT'],
                        'rs_prospectiveMemoryWords_1_delayed':['delayedRecall','Level','word','target','category','correct','Time Resp Enabled','TimeRespMade','RT'],
                        'rs_prospectiveMemoryObjects_1_delayed':['Level','targetCategory','gridItems','gridPoses','gridColours','gridOrientations','targetGridIdx','responseGridIdx','targetOrderIdx','correct','Time Resp Enabled','TimeRespMade','RT'],
                        'rs_PAL':['Level','Cue Number','Cue Image','answer', 'response', 'correct','Time Resp Enabled','TimeRespMade','RT'],
                        'rs_CRT':['Level','JitterISI','SideCorrect','SideClicked','Correct','Missclick','TimeRespEnabled', 'TimeRespMade', 'RT', 'TimeIsOut'] 
                           }


# Map which fields measure answers to questionnaire and demographics questions
dict_data['questionnaire_fields'] = {'q_PHQ2':'SummaryScore',
                                     'q_GAD7':'SummaryScore',
                                     'q_WSAS':'SummaryScore',
                                     'q_chalderFatigue':'SummaryScore',
                                     }




#%% Processing cognitron data
# -----------------------------------------------------------------------------
# Convert data column of dictionaries into separate columns
data_raw_json = data_raw_json.rename(columns = {'data':'taskData'})

taskData_expand = pd.json_normalize(data_raw_json['taskData']) # slower alternative, less complete expansion: data_raw_json['taskData'].apply(pd.Series)

# Add expanded task data back into original table
data_full = pd.merge(data_raw_json, taskData_expand, how = 'left', left_index = True, right_index = True)
data_full_cols = data_full.columns.to_list()


# -----------------------------------------------------------------------------
# Expand device and operating system data into separate columns 
data_full[['operatingSystem_1','operatingSystem_2','operatingSystem_3']] = pd.DataFrame(data_full['os'].tolist(), index= data_full.index)
data_full[['device_1','device_2','device_3']] = pd.DataFrame(data_full['device'].tolist(), index= data_full.index)
data_full[['webBrowser_1','webBrowser_2','webBrowser_3']] = pd.DataFrame(data_full['browser'].tolist(), index= data_full.index)

# replace None with NaN
data_full['device_2'] = data_full['device_2'].fillna(value=np.nan)

# Use lookup to match based on device and operating system text, adding 2 levels of device categorisation
# Broader category device_category_level_0 - phone, tablet, computer
#  More detailed device_category_level_1 - with operating system
data_full = pd.merge(data_full, device_lookup, how = 'left', on = ['device_2','device_1','operatingSystem_1'])



# -----------------------------------------------------------------------------
# For CRT task, derive fields representing accuracy
# Numerator - Number correct = 60 - Number incorrect - Number time-outs
data_full.loc[(data_full['taskID'] == 'rs_CRT'), 'Scores.numCorr'] = 60 - data_full['Scores.numIncorr'] - data_full['Scores.numTimeOuts']
# Denominator - Number of non-time-outs = 60 - Number time-outs
data_full.loc[(data_full['taskID'] == 'rs_CRT'), 'Scores.numNonTimeOuts'] = 60 - data_full['Scores.numTimeOuts']
# Proportion correct, = (60 - numIncorr - numTimeOuts) / (60 - numTimeOuts) 
data_full.loc[(data_full['taskID'] == 'rs_CRT'), 'Scores.numCorrProportion'] = np.divide(data_full['Scores.numCorr'], data_full['Scores.numNonTimeOuts'])
# Replace infinity caused by dividing by 0 - caused by leaving whole test to time out, i.e. task not completed - as np.nan to identify
data_full['Scores.numCorrProportion'] = data_full['Scores.numCorrProportion'].replace(np.inf, np.nan)

# Delete taskID of rows where individual left whole test to time out - this will mean it is not. Without deleting, it is considered incorrectly as a completion in later groupby of taskID.
data_full.loc[(data_full['Scores.numCorrProportion'].isnull())
              & (data_full['taskID'] == 'rs_CRT'), 'taskID'] = np.nan

# test = data_full[['Scores.numCorr', 'Scores.numTimeOuts', 'Scores.numNonTimeOuts', 'Scores.numCorrProportion','taskID']]


# -----------------------------------------------------------------------------
# Manually generate stats from 'Rawdata' field for tasks where summary not provided already
# List tasks where median reaction time data is missing
tasks_datamissing = ['rs_prospectiveMemoryWords_1_immediate', 'rs_prospectiveMemoryWords_1_delayed',
                     'rs_prospectiveMemoryObjects_1_immediate', 'rs_prospectiveMemoryObjects_1_delayed']

# Identify rows with RT data within Rawdata
data_full.loc[(data_full['Rawdata'].astype(str).str.contains('RT')), 'RT_present'] = 1

# Use new lines to split rows into separate lists
data_full['Rawdata'] = data_full['Rawdata'].str.split("\n")

# Filter for rows with RT in Rawdata only
data_full_filter = data_full[(data_full['RT_present'] == 1)
                             & (data_full['taskID'].isin(dict_data['task_list']))][['Rawdata','RT_present','taskID']].copy()

# Loop over rows and generate stats and save
for idx in data_full_filter['Rawdata'].index:
    print(idx)   
    # Use tabs to split columns into separate lists
    row_data_split = [sub.split("\t") for sub in data_full_filter['Rawdata'][idx]]
    
    # Convert to dataframe
    row_data_split_df = pd.DataFrame(row_data_split)
    
    # Get column names from 2nd row
    row_data_split_df.columns = row_data_split_df.iloc[1]
    
    # Drop first 2 rows and bottom 3 rows
    row_data_split_df = row_data_split_df.iloc[2:-4].reset_index(drop = True)
    
    # Generate median, mean and SD reaction time stats
    row_data_split_df['RT'] = pd.to_numeric(row_data_split_df['RT'], errors = 'coerce')
    
    meanRT = row_data_split_df['RT'].mean()
    medianRT = row_data_split_df['RT'].median()
    stdevRT = row_data_split_df['RT'].std()
    RT_q1 = row_data_split_df['RT'].quantile(0.25)
    RT_q3 = row_data_split_df['RT'].quantile(0.75)
    RT_IQR = RT_q3 - RT_q1
    
    # Save generated stats in main data table
    data_full_filter.loc[idx, 'Scores.meanRT.manual'] = meanRT
    data_full_filter.loc[idx, 'Scores.medianRT.manual'] = medianRT
    data_full_filter.loc[idx, 'Scores.stdevRT'] = stdevRT
    data_full_filter.loc[idx, 'Scores.RT_IQR'] = RT_IQR
    
    # Generate stats for correct responses only for memory tasks
    if data_full_filter['taskID'][idx] in tasks_datamissing:
        target_correct_medianRT = row_data_split_df[(row_data_split_df['correct'] == 'true')]['RT'].median()
        target_correct_stdevRT = row_data_split_df[(row_data_split_df['correct'] == 'true')]['RT'].std()
        target_correct_RT_q1 = row_data_split_df[(row_data_split_df['correct'] == 'true')]['RT'].quantile(0.25)
        target_correct_RT_q3 = row_data_split_df[(row_data_split_df['correct'] == 'true')]['RT'].quantile(0.75)
        target_correct_RT_IQR = target_correct_RT_q3 - target_correct_RT_q1
        
        # Save generated stats in main data table
        data_full_filter.loc[idx, 'Scores.target_correct_medianRT.manual'] = target_correct_medianRT
        data_full_filter.loc[idx, 'Scores.target_correct_stdevRT'] = target_correct_stdevRT
        data_full_filter.loc[idx, 'Scores.target_correct_RT_IQR'] = target_correct_RT_IQR


# Join new stats fields to main table
# Standard deviation and interquartile range for all tasks
data_full = pd.merge(data_full, data_full_filter[['Scores.stdevRT','Scores.RT_IQR']], how = 'left', left_index = True, right_index = True)

# Also join medians and correct ranges for tasks without median data already
data_full_filter_missingdatatasks = data_full_filter[(data_full_filter['taskID'].isin(tasks_datamissing))].copy()
data_full = pd.merge(data_full, data_full_filter[['Scores.meanRT.manual', 'Scores.medianRT.manual', 'Scores.target_correct_medianRT.manual', 'Scores.target_correct_stdevRT', 'Scores.target_correct_RT_IQR']], how = 'left', left_index = True, right_index = True)



#%% Create extra fields summarising stats to later pivot into summary table
# -----------------------------------------------------------------------------
# Create task number field
data_full['taskNumber'] = data_full['taskID'].map(dict_data['task_number'])
data_full['taskNumber_string'] = data_full['taskID'].map(dict_data['task_number_string'])

# Task number and ID together
data_full.loc[(data_full['taskNumber'] > 0)
                     , 'taskNumberID'] = data_full['taskNumber_string'] + '_' + data_full['taskID'].astype(str)


# Create field containing primary accuracy metric
for taskID in dict_data['primary_accuracy']:
    metric_fieldname = dict_data['primary_accuracy'][taskID]
    data_full.loc[(data_full['taskID'] == taskID) 
                         , 'primary_accuracy_fieldname'] = metric_fieldname
    data_full.loc[(data_full['taskID'] == taskID) 
                         , 'primary_accuracy'] = data_full[metric_fieldname]

# Create field containing average reaction time metric
for taskID in dict_data['reaction_time_average']:
    metric_fieldname = dict_data['reaction_time_average'][taskID]
    data_full.loc[(data_full['taskID'] == taskID) 
                         , 'reaction_time_average_fieldname'] = metric_fieldname
    data_full.loc[(data_full['taskID'] == taskID) 
                         , 'reaction_time_average'] = data_full[metric_fieldname]

# Create field containing reaction time variation metric
for taskID in dict_data['reaction_time_variation']:
    metric_fieldname = dict_data['reaction_time_variation'][taskID]
    data_full.loc[(data_full['taskID'] == taskID) 
                         , 'reaction_time_variation_fieldname'] = metric_fieldname
    data_full.loc[(data_full['taskID'] == taskID) 
                         , 'reaction_time_variation'] = data_full[metric_fieldname]


# Convert to numeric
data_full['primary_accuracy'] = pd.to_numeric(data_full['primary_accuracy'], errors = 'coerce')
data_full['reaction_time_average'] = pd.to_numeric(data_full['reaction_time_average'], errors = 'coerce')
data_full['reaction_time_variation'] = pd.to_numeric(data_full['reaction_time_variation'], errors = 'coerce')


# -----------------------------------------------------------------------------
# Delete/replace erroneous values
data_slice_missing_list = []
for task_num in range(1,13,1):
    print(task_num)
    
    # filter for task and describe data
    data_slice = data_full[data_full['taskNumber'] == task_num][['taskID','primary_accuracy','reaction_time_average','reaction_time_variation', 'Scores.target_correct_medianRT', 'Scores.medianRT.manual', 'Scores.target_correct_medianRT.manual', 'Scores.target_correct_stdevRT', 'Scores.target_correct_RT_IQR']]
    
    for col in ['primary_accuracy','reaction_time_average','reaction_time_variation', 'Scores.target_correct_medianRT']:
        data_slice[col] = pd.to_numeric(data_slice[col], errors = 'coerce')
    
    data_slice_describe = data_slice.describe()
        
    # Filter for rows where 1 or more of accuracy, average RT and variation in RT is missing to identify cause of missing data
    data_slice_missing = data_slice[(data_slice['primary_accuracy'].isnull()) 
                                    | (data_slice['reaction_time_average'].isnull()) 
                                    | (data_slice['reaction_time_variation'].isnull()) 
                                    ]
    data_slice_missing_list.append(data_slice_missing)

    # print min values 
    print('accuracy_min: ' + str(data_slice['primary_accuracy'].min()))
    print('reaction_time_average_min: ' + str(data_slice['reaction_time_average'].min()))
    print('reaction_time_variation_min: ' + str(data_slice['reaction_time_variation'].min()))
    
    # For tasks with missing or erroneous data indicating task is incomplete, set taskNumber field at negative value, so rows can be filtered out when creating summary tables of completed task data   
    # If either reaction time average or variation is empty, task not completed so delete taskID 
    data_full.loc[(data_full['taskNumber'] == task_num)
                  & ((data_full['reaction_time_average'].isnull()) 
                    | (data_full['reaction_time_variation'].isnull()))
                  , 'taskNumber'] = -9999 


# Set taskNumber to -9999 where reaction_time_average < 0 for rs_verbalAnalogies task, erroneous entry
data_full.loc[(data_full['taskID'] == 'rs_verbalAnalogies')
              & (data_full['reaction_time_average'] < 0)
              , 'taskNumber'] = -9999
    
# Set taskNumber to -9999 where accuracy < 0 for rs_prospectiveMemoryObjects_1_delayed task, which is only task to have values set to -999999 for incompletes
data_full.loc[(data_full['taskID'] == 'rs_prospectiveMemoryObjects_1_delayed')
              & (data_full['primary_accuracy'] < 0)
              , 'taskNumber'] = -9999
    



#%% Merge cognitron data with other tables to get additional information
# -----------------------------------------------------------------------------
# Join cognitron data to mapping table to get IDs
data_full_merged = pd.merge(link_id, data_full, how = 'left', left_on = 'baseline_lkey', right_on = 'user_id')

# -----------------------------------------------------------------------------
### EXCLUSIONS
# Filter for active participants only
data_full_merged = data_full_merged[data_full_merged['cssbiobank_id'].isin(patientlist['cssbiobank_id'])].copy()

# -----------------------------------------------------------------------------
# Convert start and end time from Unix to datetime format
data_full_merged['startTime_datetime'] = pd.to_datetime(data_full_merged['startTime'],unit='ms')
data_full_merged['endTime_datetime'] = pd.to_datetime(data_full_merged['endTime'],unit='ms')
data_full_merged['endTime_date'] = (data_full_merged['endTime_datetime']).dt.date.astype(str)
data_full_merged['duration_endminusstart_sec'] = (data_full_merged['endTime_datetime'] - data_full_merged['startTime_datetime']).dt.seconds
data_full_merged['duration_sec'] = pd.to_numeric(data_full_merged['duration']) / 1000 # convert from ms to s

# Sort by biobank id, and time row completed
data_full_merged = data_full_merged.sort_values(by = ['cssbiobank_id','endTime_datetime'], ascending = True)


# -----------------------------------------------------------------------------
# Identify first and second round of testing based on date
round_1_date_max = '2021-11-01' # Round 1 final entries in October 2021
data_full_merged.loc[data_full_merged['endTime_date'] < round_1_date_max,'testingRound'] = 'Round_1'
data_full_merged.loc[data_full_merged['endTime_date'] >= round_1_date_max,'testingRound'] = 'Round_2'


# -----------------------------------------------------------------------------
# Merge with tracking spreadsheet errors columns to show dates of technical issues reported for first and second round of testing
data_full_merged = pd.merge(data_full_merged, tracking[['cssbiobank_id','bl_tech_issue_reported','fu_tech_issue_reported']], how = 'left', on = 'cssbiobank_id')

# Create new column which merges date of technical issue into single column
# Round_1 error
data_full_merged.loc[data_full_merged['testingRound'] == 'Round_1', 'tech_issue_date'] = data_full_merged['bl_tech_issue_reported']
# Round_2 error
data_full_merged.loc[data_full_merged['testingRound'] == 'Round_2', 'tech_issue_date'] = data_full_merged['fu_tech_issue_reported']

# Generate list of participants with technical issues and dates in Round_1 and 2
tech_issue_round1 = tracking[~(tracking['bl_tech_issue_reported'].isnull())][['cssbiobank_id','bl_tech_issue_reported']].copy()
tech_issue_round2 = tracking[~(tracking['fu_tech_issue_reported'].isnull())][['cssbiobank_id','fu_tech_issue_reported']].copy()

data_full_merged_cols = data_full_merged.columns.to_list()


# Sort by biobank id, and time row completed
data_full_merged = data_full_merged.sort_values(by = ['cssbiobank_id','endTime_datetime'], ascending = True)




#%% Convert TASK COMPLETION data from long to wide format, grouping by participant ID and testing round and count distinct task numbers to see how many task completed. Use for analysis of participation and completion
# -----------------------------------------------------------------------------
# Filter for rows containing task results
data_tasks_long = data_full_merged[(data_full_merged['taskNumber'] > 0)].copy()

# Group table by participant ID and testing round and count distinct task numbers to see how many task completed
task_completion = data_tasks_long.groupby(['cssbiobank_id', 'testingRound']).agg({'taskID':'nunique',
                                                                             'taskNumber':'max',
                                                                             'taskNumber_string':'max',
                                                                             'cssbiobank_id':'count',
                                                                             'tech_issue_date':'first',
                                                                             'endTime_date':'last'})
task_completion = task_completion.rename(columns = {'taskID':'task_count_unique',
                                          'taskNumber':'task_number_max',
                                          'taskNumber_string':'task_number_string_max',
                                          'cssbiobank_id':'task_count_all',
                                          'endTime_date':'endDate'
                                          }).reset_index()

# Apply mapping to get task name of last task done
task_completion['taskName_max'] = task_completion['task_number_max'].map(dict_data['task_number_reverse'])



#%% Convert TASK PERFORMANCE data from long to wide format, grouping by biobank ID, testing round and task. Pick out relevant metrics for each task  taking the earliest completed (within each testing round)
# -----------------------------------------------------------------------------
# Group table by participant ID, testing round and task number and summarise relevant information
task_metrics = data_tasks_long.groupby(['cssbiobank_id',
                                                   'testingRound', 
                                                   'taskNumber_string']).agg({'taskNumberID':'first',
                                                                       'primary_accuracy_fieldname':'first',
                                                                       'primary_accuracy':'first',
                                                                       'reaction_time_average_fieldname':'first',
                                                                       'reaction_time_average':'first',
                                                                       'reaction_time_variation_fieldname':'first',
                                                                       'reaction_time_variation':'first',
                                                                       'taskID':'count',
                                                                       }).reset_index()
                                                                       
task_metrics = task_metrics.rename(columns = {'taskID':'task_count',
                                                                    })


# -----------------------------------------------------------------------------
# Pivot metrics
task_metrics_wide = task_metrics.pivot(index = ['cssbiobank_id', 'testingRound'], 
                                                columns = ['taskNumber_string'],
                                                values = ['primary_accuracy','reaction_time_average','reaction_time_variation'])

task_metrics_wide.columns = ['_'.join(col).strip() for col in task_metrics_wide.columns.values]

task_metrics_wide = task_metrics_wide.reset_index()


#%% Transformation, calculation of z-score and winsorisation of outliers in task performance data
# -----------------------------------------------------------------------------
# Describe data
task_list = [
             'primary_accuracy_01', 'primary_accuracy_02', 'primary_accuracy_03', 
             'primary_accuracy_04', 'primary_accuracy_05', 'primary_accuracy_06',
             'primary_accuracy_07', 'primary_accuracy_08', 'primary_accuracy_09',
             'primary_accuracy_10', 'primary_accuracy_11', 'primary_accuracy_12',
             
             'reaction_time_average_01', 'reaction_time_average_02', 'reaction_time_average_03',
             'reaction_time_average_04', 'reaction_time_average_05', 'reaction_time_average_06',
             'reaction_time_average_07', 'reaction_time_average_08', 'reaction_time_average_09',
             'reaction_time_average_10', 'reaction_time_average_11', 'reaction_time_average_12', 
             
             'reaction_time_variation_01', 'reaction_time_variation_02', 'reaction_time_variation_03',
             'reaction_time_variation_04', 'reaction_time_variation_05', 'reaction_time_variation_06',
             'reaction_time_variation_07', 'reaction_time_variation_08', 'reaction_time_variation_09',
             'reaction_time_variation_10', 'reaction_time_variation_11', 'reaction_time_variation_12'
             ]

# Convert to numeric
task_metrics_wide[task_list] = task_metrics_wide[task_list].apply(pd.to_numeric, errors = 'coerce')

# Describe data
task_metrics_wide_describe = task_metrics_wide[task_list].describe()


# -----------------------------------------------------------------------------
# Filter for task data and take a copy to do processing on
data = task_metrics_wide[task_list].copy()

# -----------------------------------------------------------------------------
# Delete/replace erroneous values
# primary_accuracy_10 has -999999 values - presume error or sign that task incomplete
data.loc[(data['primary_accuracy_10'] < 0), 'primary_accuracy_10'] = np.nan
data.loc[(data['primary_accuracy_10'] < 0), 'primary_accuracy_10'] = np.nan
data.loc[(data['primary_accuracy_10'] < 0), 'primary_accuracy_10'] = np.nan
# reaction_time_average_08 has one negative value for reaction time - delete
data.loc[(data['reaction_time_average_08'] < 0), 'reaction_time_average_08'] = np.nan

data_describe_error_removed = data.describe()

task_metrics_wide_describe_quantiles = data.min()
task_metrics_wide_describe_quantiles = task_metrics_wide_describe_quantiles.rename('min')
task_metrics_wide_describe_quantiles = pd.DataFrame(task_metrics_wide_describe_quantiles)
task_metrics_wide_describe_quantiles['0.27pct (3 sigma if normal)'] = data.quantile(0.0027)
task_metrics_wide_describe_quantiles['1pct'] = data.quantile(0.01)
task_metrics_wide_describe_quantiles['5pct'] = data.quantile(0.05)
task_metrics_wide_describe_quantiles['10pct'] = data.quantile(0.1)
task_metrics_wide_describe_quantiles['25pct'] = data.quantile(0.25)
task_metrics_wide_describe_quantiles['50pct'] = data.quantile(0.5)
task_metrics_wide_describe_quantiles['75pct'] = data.quantile(0.75)
task_metrics_wide_describe_quantiles['90pct'] = data.quantile(0.9)
task_metrics_wide_describe_quantiles['95pct'] = data.quantile(0.95)
task_metrics_wide_describe_quantiles['99pct'] = data.quantile(0.99)
task_metrics_wide_describe_quantiles['99.73pct (3 sigma if normal)'] = data.quantile(0.9973)
task_metrics_wide_describe_quantiles['max'] = data.max()


# -----------------------------------------------------------------------------
# Winsorisation of outliers
# Performance, average reaction time, variation in reaction time
# Don't winsorise accuracy scores, but do winsorise average and variation in reaction time - more likely to be outliers due to people going really quick (to skip through, deliberate fail) or slow (loss of interest, distraction, not a true record)

# Do manual winsorisation as scipy winsorize function has bugs (treats nan as high number rather than ignoring) which means upper end winsorisation doesn't work
def winsorization(data, winsorization_limits, winsorization_col_list):
    for col in winsorization_col_list:        
        # Calculate percentile
        winsorize_lower = data[col].quantile(winsorization_limits[0])
        winsorize_upper = data[col].quantile(winsorization_limits[1])
        # Replace lower and upper values with limited values
        data.loc[(data[col] < winsorize_lower), col] = winsorize_lower
        data.loc[(data[col] > winsorize_upper), col] = winsorize_upper
    return data

do_winsorisation = 1
if do_winsorisation == 1:
    # Select columns to do winsorization on
    task_winsorization_list = [            
             'reaction_time_average_01', 'reaction_time_average_02', 'reaction_time_average_03',
             'reaction_time_average_04', 'reaction_time_average_05', 'reaction_time_average_06',
             'reaction_time_average_07', 'reaction_time_average_08', 'reaction_time_average_09',
             'reaction_time_average_10', 'reaction_time_average_11', 'reaction_time_average_12', 
             
             'reaction_time_variation_01', 'reaction_time_variation_02', 'reaction_time_variation_03',
             'reaction_time_variation_04', 'reaction_time_variation_05', 'reaction_time_variation_06',
             'reaction_time_variation_07', 'reaction_time_variation_08', 'reaction_time_variation_09',
             'reaction_time_variation_10', 'reaction_time_variation_11', 'reaction_time_variation_12'
             ]
    # Do winsorization
    data = winsorization(data = data, 
                         winsorization_limits = [0.0027,0.9973],
                         winsorization_col_list = task_winsorization_list)


# -----------------------------------------------------------------------------
# Calculate within-sample percentile for each metric (from both rounds, pre-exclusion criteria)
task_centile_cols = [] # list of metrics converted into centile values
for task in task_list: 
    # # Quintile
    data[task+'_quintile'] = pd.qcut(data[task], 5, labels = False, duplicates= 'drop')
    data[task+'_quintile'] = data[task+'_quintile'] + 1 # Add 1 so is 1-n instead of 0-n
    task_centile_cols.append(task+'_quintile')


# -----------------------------------------------------------------------------
# Test skew before and after transformations, plot distributions of accuracy and reaction time for each task
skew_before_list = []
skew_after_log_list = []
skew_after_sqrt_list = []
skew_after_square_list = []
skew_after_cube_list = []
skew_after_exp_list = []
# Inspired by https://towardsdatascience.com/top-3-methods-for-handling-skewed-data-1334e0debf45
for task_col in task_list:
    print(task_col)
    # Add constant to 0 values to avoid log(0) 
    data.loc[~(data[task_col].isnull())
             & (data[task_col] == 0), task_col] = data[task_col] + 0.00001
    
    # Calculate skew before and after transformation
    skew_before = data[task_col].skew()
    skew_before_list.append(skew_before)
    print('Skew before transformation, ' + task_col + ': ' + str(skew_before))
    
    plot_list = [task_col]
    for plot in plot_list:
        # Plot unadjusted histogram
        ax = plt.figure()
        ax = sns.histplot(data=data, 
                          x=plot, 
                          # hue=outcome_var, 
                          # element="poly",
                          # stat = "probability",
                          # common_norm = False
                          )
        plt.title('Task: ' + plot)
    
    # Apply square, cube and exponential transform for negatively skewed data
    skew_threshold = 0 # if skew before is within +- threshold of 0, keep as is
    if skew_before < -skew_threshold:
        skew_after_log_list.append(np.nan)
        skew_after_sqrt_list.append(np.nan)
        
        # Do square transformation
        data[task_col+'_square_transform'] = np.power(data[task_col], 2)
        skew_after_square = data[task_col+'_square_transform'].skew()
        skew_after_square_list.append(skew_after_square)
        print('Skew after square transformation, ' + task_col + ': ' + str(skew_after_square))
        
        # Do cube transformation
        data[task_col+'_cube_transform'] = np.power(data[task_col], 3)
        skew_after_cube = data[task_col+'_cube_transform'].skew()
        skew_after_cube_list.append(skew_after_cube)
        print('Skew after cube transformation, ' + task_col + ': ' + str(skew_after_cube))
        
        # Plot histogram
        plot_list = [task_col+'_square_transform', task_col+'_cube_transform']
        for plot in plot_list:
            ax = plt.figure()
            ax = sns.histplot(data=data, x=plot)
            plt.title('Task: ' + plot)
            
        # Do exponential transformation, if cubed skew is still < -0.33 (moderate skew)
        if skew_after_cube < -0.33:
            data[task_col+'_exp_transform'] = np.exp(data[task_col])
            skew_after_exp = data[task_col+'_exp_transform'].skew()
            skew_after_exp_list.append(skew_after_exp)
            print('Skew after exponential transformation, ' + task_col + ': ' + str(skew_after_exp))
            
            # Plot histogram
            plot_list = [task_col+'_exp_transform']
            for plot in plot_list:
                ax = plt.figure()
                ax = sns.histplot(data=data, x=plot)
                plt.title('Task: ' + plot)
                
        else:
            skew_after_exp_list.append(np.nan)
        
        
    
    # Apply log and square root transform for positively skewed data
    elif skew_before >= skew_threshold:
        skew_after_square_list.append(np.nan)
        skew_after_cube_list.append(np.nan)
        skew_after_exp_list.append(np.nan)
        
        # Do log transformation
        data[task_col+'_log_transform'] = np.log(data[task_col])
        skew_after_log = data[task_col+'_log_transform'].skew()
        skew_after_log_list.append(skew_after_log)
        print('Skew after log transformation, ' + task_col + ': ' + str(skew_after_log))
        
        # Do square root transformation
        data[task_col+'_squareroot_transform'] = np.sqrt(data[task_col])
        skew_after_sqrt = data[task_col+'_squareroot_transform'].skew()
        skew_after_sqrt_list.append(skew_after_sqrt)
        print('Skew after square root transformation, ' + task_col + ': ' + str(skew_after_sqrt))

        # Plot histograms
        plot_list = [task_col+'_log_transform', task_col+'_squareroot_transform']
        for plot in plot_list:
           
            ax = plt.figure()
            ax = sns.histplot(data=data, x=plot)
            plt.title('Task: ' + plot)
    
    else:
        skew_after_log_list.append(np.nan)
        skew_after_sqrt_list.append(np.nan)
        skew_after_square_list.append(np.nan)
        skew_after_cube_list.append(np.nan)
        skew_after_exp_list.append(np.nan)

data_transform_summary = pd.DataFrame(data = {'task':task_list,
                                              'skew_before':skew_before_list,
                                              'skew_after_log':skew_after_log_list,
                                              'skew_after_sqrt':skew_after_sqrt_list,
                                              'skew_after_square':skew_after_square_list,
                                              'skew_after_cube':skew_after_cube_list,
                                              'skew_after_exp':skew_after_exp_list,
                                              })    
data_transform_summary.loc[(data_transform_summary['skew_before'] >= 0), 'skew_direction'] = 'positive (try log or sqrt)'
data_transform_summary.loc[(data_transform_summary['skew_before'] < 0), 'skew_direction'] = 'negative (try exp, square, cube)'

# Find skew value closest to 0
data_transform_summary['skew_min_idx'] = data_transform_summary[['skew_before','skew_after_log','skew_after_sqrt','skew_after_square','skew_after_cube','skew_after_exp']].abs().idxmin(axis = 1)

dict_data['transformation'] = {'skew_before':'',
                               'skew_after_log':'_log_transform',
                               'skew_after_sqrt':'_squareroot_transform',
                               'skew_after_square':'_square_transform',
                               'skew_after_cube':'_cube_transform',
                               'skew_after_exp':'_exp_transform',
                               }
# Column name suffix to use to select best transform
data_transform_summary['transform_select'] = data_transform_summary['skew_min_idx'].map(dict_data['transformation'])
transform_select_list = data_transform_summary['transform_select'].to_list()

# Get list of column names to select best transform for each metric
task_plus_transform_list = []
for n in range(0,len(task_list),1):
    if type(transform_select_list[n]) is str:
        task_plus_transform = task_list[n] + transform_select_list[n]
    else:
        task_plus_transform = task_list[n]
    task_plus_transform_list.append(task_plus_transform)
        
    
# -----------------------------------------------------------------------------
# Transformation to obtain (as close as possible to) normal distributions for each metric
data_with_transform = data[task_plus_transform_list].copy()


# -----------------------------------------------------------------------------
# Convert transformed fields into standard deviations units (z-score)
for task in task_plus_transform_list:
    data_with_transform[task+'_zscore'] = zscore(np.array(data_with_transform[task]), nan_policy="omit") 

    # Plot histogram as a check
    data_with_transform[task+'_zscore']
    
    ax = plt.figure()
    ax = sns.histplot(data=data_with_transform, x=task+'_zscore')
    plt.title('Task: ' + task+'_zscore')

data_with_transform_cols = data_with_transform.columns.to_list()


# -----------------------------------------------------------------------------
# Merge transformed data back into main table
# Add suffix to distinguish between un-processed and processed fields
task_metrics_wide = pd.merge(task_metrics_wide, data_with_transform.add_suffix('_processed'), how = 'left', left_index = True, right_index = True)
task_metrics_wide_cols = task_metrics_wide.columns.to_list()

# -----------------------------------------------------------------------------
# Merge data converted into centile values into main table
task_metrics_wide = pd.merge(task_metrics_wide, data[task_centile_cols], how = 'left', left_index = True, right_index = True)
task_metrics_wide_cols = task_metrics_wide.columns.to_list()


#%% Collect responses to questionnaire and demographics questions and convert to wide format
# -----------------------------------------------------------------------------
# Filter for rows containing questionnaire results
taskID_questionnaire = ['q_PHQ2','q_GAD7','q_WSAS','q_chalderFatigue']
data_questionnaires_long = data_full_merged[(data_full_merged['taskID'].isin(taskID_questionnaire))].copy()

# -----------------------------------------------------------------------------
# Group table by participant ID, testing round and task ID and take maximum value in case of multiple attempts at questionnaire during same testing round
questionnaire_scores = data_questionnaires_long.groupby(['cssbiobank_id',
                                                         'testingRound', 
                                                         'taskID']).agg({'SummaryScore':'max'}).reset_index()


# -----------------------------------------------------------------------------
# Pivot to go from long to wide format
questionnaire_scores_wide = questionnaire_scores.pivot(index = ['cssbiobank_id', 'testingRound'], 
                                                columns = ['taskID'],
                                                values = ['SummaryScore'])

questionnaire_scores_wide.columns = ['_'.join(col).strip() for col in questionnaire_scores_wide.columns.values]

questionnaire_scores_wide = questionnaire_scores_wide.reset_index()

questionnaire_scores_wide = questionnaire_scores_wide.rename(columns = {'SummaryScore_q_PHQ2':'q_PHQ2_score',
                                                                        'SummaryScore_q_GAD7':'q_GAD7_score',
                                                                        'SummaryScore_q_WSAS':'q_WSAS_score',
                                                                        'SummaryScore_q_chalderFatigue':'q_chalderFatigue_score'})


# -----------------------------------------------------------------------------
# Extract responses for GAD-2 questions - so can combine with other GAD-2, PHQ-2, PHQ-4 collected within Biobank sub-studies
# GAD 2 items. Feeling nervous, anxious or on edge - RespObject.Q0.S. Not being able to stop or control worrying - RespObject.Q1.S
gad2_scores = data_questionnaires_long[(data_questionnaires_long['taskID'] == 'q_GAD7')].groupby(['cssbiobank_id',
                                                'testingRound', 
                                                'taskID']).agg({'RespObject.Q0.S':'max',
                                                                'RespObject.Q1.S':'max',}).reset_index()
# Sum columns to get score 
gad2_scores['q_GAD2_score'] = gad2_scores[['RespObject.Q0.S','RespObject.Q1.S']].astype(float).astype('Int64').sum(skipna = True, axis = 1)

# Merge GAD-2 with other questionnaire scores
questionnaire_scores_wide = pd.merge(questionnaire_scores_wide, gad2_scores[['cssbiobank_id','testingRound','q_GAD2_score']], how = 'left', on = ['cssbiobank_id','testingRound'])

# Sum GAD-2 and PHQ-2 to get PHQ-4 score
questionnaire_scores_wide['q_PHQ4_score'] = questionnaire_scores_wide[['q_GAD2_score','q_PHQ2_score']].astype(float).astype('Int64').sum(skipna = True, axis = 1)
# Delete scores for individuals without both PHQ-2 and GAD-2 components
questionnaire_scores_wide.loc[(questionnaire_scores_wide['q_GAD2_score'].isnull()) 
                              | (questionnaire_scores_wide['q_PHQ2_score'].isnull()), 'q_PHQ4_score'] = np.nan

# -----------------------------------------------------------------------------
# Add categorical fields
# GAD-7, 7 item, 0-21
# 4 categories
questionnaire_scores_wide.loc[(questionnaire_scores_wide['q_GAD7_score'] >= 0) & (questionnaire_scores_wide['q_GAD7_score'] <= 4)
                         ,'q_GAD7_cat4'] = '1. 0-4, below threshold'
questionnaire_scores_wide.loc[(questionnaire_scores_wide['q_GAD7_score'] >= 5) & (questionnaire_scores_wide['q_GAD7_score'] <= 9)
                         ,'q_GAD7_cat4'] = '2. 5-9, mild'
questionnaire_scores_wide.loc[(questionnaire_scores_wide['q_GAD7_score'] >= 10) & (questionnaire_scores_wide['q_GAD7_score'] <= 14)
                         ,'q_GAD7_cat4'] = '3. 10-14, moderate'
questionnaire_scores_wide.loc[(questionnaire_scores_wide['q_GAD7_score'] >= 15) & (questionnaire_scores_wide['q_GAD7_score'] <= 21)
                         ,'q_GAD7_cat4'] = '4. 15-21, severe'
# binary 2 categories
questionnaire_scores_wide.loc[(questionnaire_scores_wide['q_GAD7_score'] >= 0) & (questionnaire_scores_wide['q_GAD7_score'] <= 9)
                         ,'q_GAD7_cat2'] = '1. 0-9, below threshold'
questionnaire_scores_wide.loc[(questionnaire_scores_wide['q_GAD7_score'] >= 10) & (questionnaire_scores_wide['q_GAD7_score'] <= 21)
                         ,'q_GAD7_cat2'] = '2. 10-21, above threshold'

# PHQ-2, 2 item, 0-6
# binary 2 categories
questionnaire_scores_wide.loc[(questionnaire_scores_wide['q_PHQ2_score'] >= 0) & (questionnaire_scores_wide['q_PHQ2_score'] <= 2)
                         ,'q_PHQ2_cat2'] = '1. 0-2, below threshold'
questionnaire_scores_wide.loc[(questionnaire_scores_wide['q_PHQ2_score'] >= 3) & (questionnaire_scores_wide['q_PHQ2_score'] <= 6)
                         ,'q_PHQ2_cat2'] = '2. 3-6, above threshold'

# GAD-2, 2 item, 0-6
# binary 2 categories
questionnaire_scores_wide.loc[(questionnaire_scores_wide['q_GAD2_score'] >= 0) & (questionnaire_scores_wide['q_GAD2_score'] <= 2)
                         ,'q_GAD2_cat2'] = '1. 0-2, below threshold'
questionnaire_scores_wide.loc[(questionnaire_scores_wide['q_GAD2_score'] >= 3) & (questionnaire_scores_wide['q_GAD2_score'] <= 6)
                         ,'q_GAD2_cat2'] = '2. 3-6, above threshold'

# PHQ-4, 4 item (PHQ-2 + GAD-2), 0-12
# 4 categories
questionnaire_scores_wide.loc[(questionnaire_scores_wide['q_PHQ4_score'] >= 0) & (questionnaire_scores_wide['q_PHQ4_score'] <= 2)
                         ,'q_PHQ4_cat4'] = '1. 0-2, below threshold'
questionnaire_scores_wide.loc[(questionnaire_scores_wide['q_PHQ4_score'] >= 3) & (questionnaire_scores_wide['q_PHQ4_score'] <= 5)
                         ,'q_PHQ4_cat4'] = '2. 3-5, mild'
questionnaire_scores_wide.loc[(questionnaire_scores_wide['q_PHQ4_score'] >= 6) & (questionnaire_scores_wide['q_PHQ4_score'] <= 8)
                         ,'q_PHQ4_cat4'] = '3. 6-8, moderate'
questionnaire_scores_wide.loc[(questionnaire_scores_wide['q_PHQ4_score'] >= 9) & (questionnaire_scores_wide['q_PHQ4_score'] <= 12)
                         ,'q_PHQ4_cat4'] = '4. 9-12, severe'

# WSAS, 5 item, 0-40
# 3 categories
questionnaire_scores_wide.loc[(questionnaire_scores_wide['q_WSAS_score'] >= 0) & (questionnaire_scores_wide['q_WSAS_score'] <= 9)
                         ,'q_WSAS_cat4'] = '1. 0-9, below threshold'
questionnaire_scores_wide.loc[(questionnaire_scores_wide['q_WSAS_score'] >= 10) & (questionnaire_scores_wide['q_WSAS_score'] <= 20)
                         ,'q_WSAS_cat4'] = '2. 10-20, mild'
questionnaire_scores_wide.loc[(questionnaire_scores_wide['q_WSAS_score'] >= 21) & (questionnaire_scores_wide['q_WSAS_score'] <= 40)
                         ,'q_WSAS_cat4'] = '3. 21-40, moderate to severe'
# binary 2 categories
questionnaire_scores_wide.loc[(questionnaire_scores_wide['q_WSAS_score'] >= 0) & (questionnaire_scores_wide['q_WSAS_score'] <= 9)
                         ,'q_WSAS_cat2'] = '1. 0-9, below threshold'
questionnaire_scores_wide.loc[(questionnaire_scores_wide['q_WSAS_score'] >= 10) & (questionnaire_scores_wide['q_WSAS_score'] <= 40)
                         ,'q_WSAS_cat2'] = '2. 10-40, above threshold'

# Chalder fatigue, 11 item, 0-33
# binary 2 categories
questionnaire_scores_wide.loc[(questionnaire_scores_wide['q_chalderFatigue_score'] >= 0) & (questionnaire_scores_wide['q_chalderFatigue_score'] <= 28)
                         ,'q_chalderFatigue_cat2'] = '1. 0-28, below threshold'
questionnaire_scores_wide.loc[(questionnaire_scores_wide['q_chalderFatigue_score'] >= 28) & (questionnaire_scores_wide['q_chalderFatigue_score'] <= 33)
                         ,'q_chalderFatigue_cat2'] = '2. 29-33, above threshold'


# -----------------------------------------------------------------------------
# Demographics (and device type) questions
# Handedness
fieldname_handedness = 'RespObject.Q0.R'
# Group table by participant ID, testing round and take latest value in case of multiple attempts at questionnaire during same testing round (assuming latest is most reliable)
demogs_handedness = data_full_merged[(data_full_merged['taskID'] == 'q_CSS_intro')].groupby(['cssbiobank_id',
                                              'testingRound']).agg({fieldname_handedness:'last'}).reset_index()

# First language
fieldname_language = 'RespObject.Q1.R' 
# Group table by participant ID, testing round and take latest value in case of multiple attempts at questionnaire during same testing round (assuming latest is most reliable)
demogs_language = data_full_merged[(data_full_merged['taskID'] == 'q_CSS_intro')].groupby(['cssbiobank_id',
                                              'testingRound']).agg({fieldname_language:'last'}).reset_index()

# Education level
fieldname_education = 'RespObject.Q2.R' 
# Group table by participant ID, testing round and take latest value in case of multiple attempts at questionnaire during same testing round (assuming latest is most reliable)
demogs_education= data_full_merged[(data_full_merged['taskID'] == 'q_CSS_intro')].groupby(['cssbiobank_id',
                                              'testingRound']).agg({fieldname_education:'last'}).reset_index()

# Device type
fieldname_device = 'device_category_level_0' 
# Group table by participant ID, testing round and take latest value in case of multiple attempts at questionnaire during same testing round (assuming latest is most reliable)
demogs_device = data_full_merged.groupby(['cssbiobank_id',
                                              'testingRound']).agg({fieldname_device:'last'}).reset_index()
# Pivot to get round 1 and 2 as separate columns
demogs_device_pivot = demogs_device.pivot(index = 'cssbiobank_id', columns = 'testingRound')
# Flatten column name
demogs_device_pivot.columns = ['_'.join(col).strip() for col in demogs_device_pivot.columns.values]

# -----------------------------------------------------------------------------
# Merge together to get demogs in wide format
demogs_wide = pd.merge(demogs_handedness, demogs_language, how = 'outer', left_on = ['cssbiobank_id','testingRound'], right_on = ['cssbiobank_id','testingRound'])
demogs_wide = pd.merge(demogs_wide, demogs_education, how = 'outer', left_on = ['cssbiobank_id','testingRound'], right_on = ['cssbiobank_id','testingRound'])
demogs_wide = demogs_wide.rename(columns = {'RespObject.Q0.R':'handedness',
                                            'RespObject.Q1.R':'firstLanguage',
                                            'RespObject.Q2.R':'educationLevel',
                                            })

# -----------------------------------------------------------------------------
# Add categorical fields
# Language - English vs. Other
demogs_wide.loc[(demogs_wide['firstLanguage'] == 'English')
                         ,'firstLanguage_cat3'] = '1. English'
demogs_wide.loc[~(demogs_wide['firstLanguage'].isin(['English','Prefer not to say']))
                         ,'firstLanguage_cat3'] = '2. Other'
demogs_wide.loc[(demogs_wide['firstLanguage'] == 'Prefer not to say')
                         ,'firstLanguage_cat3'] = np.nan
            
# Education level - Less than degree vs. degree or higher
demogs_wide.loc[(demogs_wide['educationLevel'].isin(['University degree','Postgraduate degree or higher','PhD']))
                         ,'educationLevel_cat3'] = '1. Degree or higher'
demogs_wide.loc[(demogs_wide['educationLevel'].isin(['A levels or advanced GNVQ or equivalent','GCSE or GNVQ or equivalent','Did not complete secondary school']))
                         ,'educationLevel_cat3'] = '2. Less than degree level'
demogs_wide.loc[(demogs_wide['educationLevel'].isin(['Other/ prefer not to say']))
                         ,'educationLevel_cat3'] = '3. Other/ prefer not to say'

# Education level - more detailed
demogs_wide.loc[(demogs_wide['educationLevel'].isin(['University degree']))
                         ,'educationLevel_cat4'] = '1. Undergraduate degree'
demogs_wide.loc[(demogs_wide['educationLevel'].isin(['Postgraduate degree or higher','PhD']))
                         ,'educationLevel_cat4'] = '4. Postgraduate degree or higher'
demogs_wide.loc[(demogs_wide['educationLevel'].isin(['A levels or advanced GNVQ or equivalent','GCSE or GNVQ or equivalent','Did not complete secondary school']))
                         ,'educationLevel_cat4'] = '2. Less than undergraduate degree level'
demogs_wide.loc[(demogs_wide['educationLevel'].isin(['Other/ prefer not to say']))
                         ,'educationLevel_cat4'] = '3. Other/ prefer not to say'

demogs_wide_cols = demogs_wide.columns.to_list()

#%% Join together symptom duration, cognitron participation, task count and task metrics summary tables into one master data table
# -----------------------------------------------------------------------------
# Join together task participation and task metrics tables
task_completion_plus_metrics = pd.merge(task_completion, task_metrics_wide, how = 'left', left_on = ['cssbiobank_id', 'testingRound'], right_on = ['cssbiobank_id', 'testingRound'])

# Subsets for Round_1 and 2 testing only
task_completion_plus_metrics_round1 = task_completion_plus_metrics[task_completion_plus_metrics['testingRound'] == 'Round_1'].copy()
task_completion_plus_metrics_round2 = task_completion_plus_metrics[task_completion_plus_metrics['testingRound'] == 'Round_2'].copy()

# Add invitation status from tracking to task completion and performance metrics table
data_master = pd.merge(tracking[['cssbiobank_id','bl_status','fu_status']], task_completion_plus_metrics,  how = 'left', on = 'cssbiobank_id')

# Add questionnaire results
data_master = pd.merge(data_master, questionnaire_scores_wide, how = 'left', left_on = ['cssbiobank_id', 'testingRound'], right_on = ['cssbiobank_id', 'testingRound'])

# Add demographics responses - merge on id only, not testing round - assuming responses given in round 2 also apply for round 1
col_select = ['cssbiobank_id', 'handedness', 'firstLanguage', 'educationLevel', 'firstLanguage_cat3', 'educationLevel_cat3', 'educationLevel_cat4']
data_master = pd.merge(data_master, demogs_wide[col_select], how = 'left', left_on = 'cssbiobank_id', right_on = 'cssbiobank_id')

# Add device type information
data_master = pd.merge(data_master, demogs_device_pivot, how = 'left', left_on = 'cssbiobank_id', right_index = True)



#%% Additional processing after merging
# -----------------------------------------------------------------------------
# Create new fields summarising participation in cognitron 
### Round_1
# If bl_status any value, mark as invited and later replace based on task completion data
data_master.loc[(data_master['bl_status'].isin(['Did_not_respond',
                                                'Declined_participation',
                                                'Full_completion',
                                                'Partial_completion']))
                               , 'status_round_1'] = '1_Invited'
# If task_count_unique > 0, Partial completion
data_master.loc[(data_master['task_count_unique'] > 0) 
                               & (data_master['testingRound'] == 'Round_1')
                               , 'status_round_1'] = '2_Completion_Partial'
# If task_count_unique == 12, Full completion (overwrites partials)
data_master.loc[(data_master['task_count_unique'] == 12) 
                               & (data_master['testingRound'] == 'Round_1')
                               , 'status_round_1'] = '3_Completion_Full'

# Fill in blanks with 'not invited'
data_master['status_round_1'] = data_master['status_round_1'].fillna('0_Not_Invited')

# Binary field aggregating partial and full
data_master.loc[(data_master['status_round_1'].isin(['2_Completion_Partial', '3_Completion_Full']))
                               , 'status_round_1_binary'] = '2_Completion_FullORPartial'
data_master['status_round_1_binary'] = data_master['status_round_1_binary'].fillna(data_master['status_round_1'])


### Round_2
# If fu_status any value other than 'Email_not_delivered' or 'Not_included', mark as invited and later replace based on task completion data
data_master.loc[(data_master['fu_status'].isin(['Did_not_respond',
                                                 'Declined_participation',
                                                 'Full_completion',
                                                 'Partial_completion',
                                                 'Withdrew_CSSB']))
                               , 'status_round_2'] = '1_Invited'
# If task_count_unique > 0, Partial completion
data_master.loc[(data_master['task_count_unique'] > 0) 
                               & (data_master['testingRound'] == 'Round_2')
                               , 'status_round_2'] = '2_Completion_Partial'
# If task_count_unique == 12, Full completion (overwrites partials)
data_master.loc[(data_master['task_count_unique'] == 12) 
                               & (data_master['testingRound'] == 'Round_2')
                               , 'status_round_2'] = '3_Completion_Full'

# Fill in blanks with 'not invited'
data_master['status_round_2'] = data_master['status_round_2'].fillna('0_Not_Invited')

# Binary field aggregating partial and full
data_master.loc[(data_master['status_round_2'].isin(['2_Completion_Partial', '3_Completion_Full']))
                               , 'status_round_2_binary'] = '2_Completion_FullORPartial'
data_master['status_round_2_binary'] = data_master['status_round_2_binary'].fillna(data_master['status_round_2'])


# -----------------------------------------------------------------------------
### Group by and get max participation status for each round and merge to fill in gaps, so every row shows status for both rounds
data_master_copy = data_master.copy()

### Round 1
data_master_round1_status_grouped = data_master_copy.groupby('cssbiobank_id').agg({'status_round_1':'max',
                                                                              'status_round_1_binary':'max',
                                                                              })
# Drop original columns
data_master = data_master.drop(columns = ['status_round_1','status_round_1_binary'])
# Merge to main table to fill in gaps
data_master = pd.merge(data_master, data_master_round1_status_grouped, how = 'left', left_on = 'cssbiobank_id', right_index = True)

### Round 2
data_master_round2_status_grouped = data_master_copy.groupby('cssbiobank_id').agg({'status_round_2':'max',
                                                                              'status_round_2_binary':'max',
                                                                              })
# Drop original columns
data_master = data_master.drop(columns = ['status_round_2','status_round_2_binary'])
# Merge to main table to fill in gaps
data_master = pd.merge(data_master, data_master_round2_status_grouped, how = 'left', left_on = 'cssbiobank_id', right_index = True)

data_master_cols = data_master.columns.to_list()


#%% Export data
if export_csv == 1:
    data_master.to_csv('cognitron_data_processed.csv', index = False)
