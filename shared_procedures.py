# Project: UoL Thesis 2025
# Aim:     Shared procedures (shared_procedures.py) for all notebooks of this project
# Author:  Stefan Mohr

"""
This Python script contains general procedures and setups used across several Jupyter Notebooks from this project/master thesis. 

Main parts of the routines have been developed in previous courses at the University of London by the same author (Mohr, 2021, 2023, 2024a) and have been developed further to fulfil the needs of the scraping procedure for this MSc thesis. However, the code has been modified to fulfil the latest requirements and package inter-dependencies. Some comments will be added in this Jupyter Notebook and the code has several inline comments. For the project/research itself, see the appropriate document.

References (for this script)
    Mohr, S. (2021) Regional Spatial Clusters of Earthquakes at the Pacific Ring of Fire: Analysing Data from the USGS ANSS
    ComCat and Building Regional Spatial Clusters. DSM020, Python, examined coursework cw1. University of London.*

    Mohr, S. (2023) Clustering of Earthquakes on a Worldwide Scale with the Help of Big Data Machine Learning Methods. 
    DSM010, Big Data, examined coursework cw2. University of London.*

    Mohr, S. (2024a) Comparing Different Tectonic Setups Considering Publicly Available Basic Earthquake’s Data. DSM050,
    Data Visualisation, examined coursework cw1. University of London.*

History:
241018 Creation of this Python script (.py) to hold and shared procedures Add and re-write setup_logging and save_dataset,
       use this at 10_scrape_earthquake_data.ipynb, re-egnineer and move get_data_from_web_api to this collection
250103 Adding load_dataset
250105 Systematically moving all shared procedures used so far to this script
250108 Add load_timeseries, save_timeseries, summarize_time_series
250110 Add get_boolean_columns, get_temporal_columns, analyze_outliers
250112 Add plot_timeseries, plot_acf_pacf_series, perform_stationarity_tests
250113 Add handle_stationarity, calculate_ccf_results, top_max_min_indices_as_dataframe, run_ccf_analysis
250114 Replace run_ccf_analysis by show_crosscorrelation_results, add cross_corr_ci_lag_varying, 
       replace new show_crosscorrelation_results
250116 Add analyze_and_plot_fft, improve stemplots for show_crosscorrelation_results
250118 Add perform_heteroskedasticity_tests, plot_stft, added FFT Highpower Lags capabilities to show_crosscorrelation_results
250121 Add save_notebook_as_html, print_loaded_parameters, list_subdirectories, choose_subdirectory,
       saving figures with show_crosscorrelation_results
250123 Adding missing docstrings to functions, change behaviour of handle_stationarity
250124 Move check_stationarity outside other function, make perform_heteroskedasticity_tests more robust against failing
       one ore more tests and providing results anyway, add meters_to_degrees
250126 Moved all function from 253_spatial_operations_gvp_reclustering
"""

# ----------------------------------------------------------------------------------------------------------------------------

# importing standard libraries
import sys
import os
import warnings
import datetime
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import logging

# importing additional libraries
import requests
from requests.exceptions import HTTPError

from tabulate import tabulate

from sklearn.ensemble import IsolationForest

from matplotlib.dates import DateFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_ccf
from statsmodels.tsa.stattools import acf, pacf, ccf, adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import AutoReg
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, het_arch

from scipy.stats import norm, boxcox
from scipy.signal import correlate, stft

from nbconvert import HTMLExporter
import nbformat

import ruptures as rpt

# ----------------------------------------------------------------------------------------------------------------------------

def setup_logging(logfile_dir, logfile_name, log_level, script_name):
    """
    Sets up logging to both a specified file and the console.
    
    Parameters
    ----------
    logfile_dir : str
        Directory where the log file will be saved.
    logfile_name : str
        Name of the log file.
    log_level : int
        Logging level. E.g., logging.DEBUG, logging.INFO.
    script_name : str
        Name of the calling script.
        
    Logs
    ----
    - Logs its start into the logging system.
    
    Notes
    -----
    - This docstring was generated with the help of AI and proofread by the author.
    """
    
    # ensure logfile dir is already in place
    os.makedirs(logfile_dir, exist_ok=True)

    # set logfile complete path
    logfile_path = os.path.join(logfile_dir, logfile_name)
    
    # clear pre-existing logging handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # setup logging (and logfile if necessary)
    logging.basicConfig(
        level=log_level,                                      # set level
        format='%(asctime)s - %(levelname)s - %(message)s',   # set format
        handlers=[
            logging.FileHandler(logfile_path),                # use log file
            logging.StreamHandler()                           # use console
        ]
    )

    # log the starting of the log from this script (if the loglevel is appropriate)
    logging.info(f"Starting script '{script_name}'.")
    logging.info(f"Set loglevel to {logging.getLevelName(log_level)}.")

# ----------------------------------------------------------------------------------------------------------------------------

def save_dataset(data_file, data_set, append_datetime=True, overwrite_file=False, data_dir=None):
    """
    Save a dataset to a specified directory with a timestamped filename.

    This function ensures that the given directory exists, and saves the dataset to a file in that directory.
    The filename will be suffixed with the current date and time to avoid overwriting existing files.

    Parameters
    ----------
    data_file : str
        The name of the file (with extension) where the data should be saved. 
        The filename will be suffixed with the current date and time.
    data_dir : str
        The directory where the data file should be saved. The directory will be created if it doesn't exist.
    data_set : pandas.DataFrame
        The dataset to save. Should be a pandas DataFrame object.
    append_datetime (bool, optional): Whether to append the current date and time 
        to the filename. Defaults to True.
    overwrite_file (bool, optional): Whether to overwite existing files or not. 
        Defaults to False.

    Raises
    ------
    FileExistsError
        If the file already exists. In this case, the function logs an error and does not overwrite the existing file.
    FileNotFoundError
        If the directory for the file does not exist. Although this should not typically occur since os.makedirs ensures
        the directory's existence, it is handled defensively.
    PermissionError
        If the program lacks necessary permissions to write to the specified directory or file.
    Exception
        Any other exception that might occur during the file writing process is caught and logged.

    Notes
    -----
    - Logs success or any encountered error into the logging system.
    - The date-time appended to the filename is formatted as "yymmdd-HHMMSS".
    - This docstring was generated with the help of AI and proofread by the author.
    """
    
    # ensure data dir is in place
    if(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    # get the current date in the format yymmdd
    current_date_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S")

    # split filename into name and extension
    filename, extension = data_file.rsplit('.', 1)

    # add current date, if wanted, unles leave filename untouched
    if (append_datetime):
        data_file = f"{filename}_{current_date_time}.{extension}"

    # construct full file path and name
    if(data_dir):
        data_file_full_path = os.path.join(data_dir, data_file)
    else:
        data_file_full_path = data_file

    # try to save the data to the specified file
    try:
        # open file in x-mode, fails if the file already exists
        if(overwrite_file):
            data_set.to_csv(data_file_full_path, index=True, sep=',')
        else:
            with open(data_file_full_path, 'x') as f:
                data_set.to_csv(f, index=True, sep=',')

    # handle several different exceptions
    except FileExistsError:
        # file already exists
        logging.error(f"save_dataset: File '{data_file_full_path}' already exists and will not be overwritten!")

    except FileNotFoundError:
        # directory does not exist
        logging.error(f"save_dataset: The directory for the file '{data_file_full_path}' does not exist!")
        logging.error(f"save_dataset: Ensure the '{os.path.dirname(data_file_full_path)}' directory exists relative to the program path!")

    except PermissionError:
        # wrong permissions
        logging.error(f"save_dataset: Permission denied, unable to write to '{data_file_full_path}'!")

    except Exception as e:
        # any other exceptions
        logging.error(f"save_dataset: Unable to save data to file {data_file_full_path}, exception: {e}!")

    # show success
    else:
        # successfully saving data
        logging.info(f"save_dataset: Data saved successfully to '{data_file_full_path}'.")

# ----------------------------------------------------------------------------------------------------------------------------

def load_dataset(data_file, data_dir=None):
    """
    Load a dataset from a specified directory with a given filename and ensures that the given directory exists.

    Parameters
    ----------
    data_file : str
        The name of the file (with extension) to load.
    data_dir : str
        The directory where the data file is stored. Defaults to None.
        
    Returns
    -------
    dataframe : pandas.dataframe
        The loaded CSV data in pandas dataframe format.

    Raises
    ------
    FileNotFoundError
        If the specified file and/or directory  does not exist.
    PermissionError
        If the program lacks necessary permissions to write to the specified directory or file.
    Exception
        Any other exception that might occur during the file writing process is caught and logged.

    Logs
    ----
    - Logs success or any encountered error into the logging system.
    
    Notes
    -----
    - This docstring was generated with the help of AI and proofread by the author.
    """
    
    # construct full file path and name
    data_file_full_path = os.path.join(data_dir, data_file)

    # try to save the data to the specified file
    try:
        # read the data
        loaded_data = pd.read_csv(data_file_full_path, index_col = None, sep = ',', low_memory = False)

    # handle several different exceptions
    except FileNotFoundError:
        # directory does not exist
        logging.error(f"load_dataset: The file file '{data_file_full_path}' does not exist!")

    except PermissionError:
        # wrong permissions
        logging.error(f"load_dataset: Permission denied, unable to load '{data_file_full_path}'!")

    except Exception as e:
        # any other exceptions
        logging.error(f"load_dataset: Unable to load data from file {data_file_full_path}, exception: {e}!")

    # show success
    else:
        # successfully saving data
        logging.info(f"load_dataset: Data loaded successfully from '{data_file_full_path}'.")
        
        # return loaded data
        return loaded_data

# ----------------------------------------------------------------------------------------------------------------------------

def get_data_from_web_api(url, query_parameters, verbosity=0):
    """
    Retrieve data from a web based API with robust error handling.
    
    This function attempts to fetch data from a specified web API endpoint using provided
    query parameters. It includes error handling for HTTP and other exceptions, and logs detailed error
    messages depending on the verbosity level.
    
    Parameters:
    -----------
    url : str
        The URL of the web API endpoint.
        
    query_parameters : dict
        A dictionary of query parameters to include in the API request.
        
    verbosity : int, optional
        The verbosity level of the function's output. 
        0 means silent mode (default), 
        1 means output informational messages.
        
    Returns:
    --------
    tuple:
        A tuple consisting of two elements:
        - bool: Indicates the success status of the API request (True if successful, False otherwise).
        - Response or None: The response object from the API request if successful, otherwise None.

    Raises
    ------
    HTTPError
        If an HTTP error occurs during the request.
    Exception
        For any other errors that occur during the request.

    Logs and Prints
    ---------------
    - Logs errors into the logging system and prints status messages if verbosity is greater than 0.

    Notes:
    ------
    - If verbosity is greater than 0, the function will print some informational messages and errors.
    - The function logs errors using the `logging` module for better traceability and debugging.
    - This docstring was generated with the help of AI and proofread by the author.
    """    

    # str for url, dict for query_parameters, int for verbosity
    if (type(url) is str 
        and type(query_parameters) is dict 
        and type(verbosity) is int
       ):
    
        # try to get the response
        try:
            response = requests.get(url, params = query_parameters)
            if(verbosity > 0):
                print("TRY in get_data_from_web_api")
            response.raise_for_status()

        # exception for http errors
        except HTTPError as http_err:
            logging.error(f"get_data_from_web_api: HTTP error occurred, {http_err}!")
            return(False, None)

        # exception for other errors
        except Exception as err:
            logging.error(f"get_data_from_web_api: Other error occurred, {err}!")
            return(False, None)

        # exception for undefined error and/or any other error
        except:
            logging.error(f"get_data_from_web_api: Unexpected error, {sys.exc_info()[0]}!")

        # successfully got the data from the url
        else:
            if(verbosity > 0):
                print("SUCCESS in get_data_from_web_api, queried url with parameters sucessfully")
                print("   Url:", url)
                print("   Parameters:", str(query_parameters))
            return(True, response)
        
    # wrong parameter types
    else:
            if(verbosity > 0):
                print("ERROR with wrong parameter type(s)")
                logging.error(f"get_data_from_web_api: Wrong parameter type(s)!")
            return(False, None)
        
# ----------------------------------------------------------------------------------------------------------------------------

def create_ascii_table(df, feature):
    """
    Create an ASCII table of distinct values, their counts, total count, 
    and percentage for a given feature in a dataframe.
    
    Parameters
    ----------
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the feature/column.
    
    Returns
    -------
        ascii_table (str): The formatted ASCII table as a string.
        total_count (int): The total count of items.
        number_of_distinct_rows (int): The number of distinct rows.
        
    Notes
    -----
    - This docstring was generated with the help of AI and proofread by the author.
    """
    
    # get the count of unique values and the total
    value_counts = df[feature].value_counts()
    total_count = value_counts.sum()

    # calculate percentages
    percentages = (value_counts / total_count) * 100

    # convert to a list of tuples
    value_counts_list = [(value, count, f"{percentage:.2f}%") for value, count, percentage in zip(value_counts.index, value_counts.values, percentages)]

    # get number of distinct rows
    number_of_distinct_rows = len(value_counts_list)

    # create headers
    headers = [feature, 'Count', 'Percentage']
    
    # generate the ASCII table without horizontal lines or delimiters
    ascii_table = tabulate(value_counts_list, headers=headers, tablefmt="simple")
    
    # return results
    return ascii_table, total_count, number_of_distinct_rows

# ----------------------------------------------------------------------------------------------------------------------------

def show_duplicates(df):
    """
    Display information about duplicate rows in a DataFrame.

    Parameters:
        df (pandas.DataFrame): 
            The input DataFrame to check for duplicates.

    Returns:
        None: Outputs information about duplicates directly via print statements.
        
    Notes:
        This docstring was generated with the help of AI and proofread by the author.
    """
    
    duplicates = df.duplicated(keep='first')
    if duplicates.any():
        num_duplicates = int(duplicates.sum())
        print(f"\n {num_duplicates} duplicate rows.")
    else:
        print("No duplicates found.")
        
# ----------------------------------------------------------------------------------------------------------------------------

def get_data_errors(df):
    """
    Check a given dataframe for any dataerrors which are defined
    in the list *missing_values* from this method.
        
    Parameters:
        df (pd.DataFrame): Dataframe to check for data errors.
        
    Returns:
        results (pd.DataFrame): Dataframe with type and count of errors found.
        
    Notes:
        This docstring was generated with the help of AI and proofread by the author.
    """
    
    # list of  possible data errors
    missing_values = [-1, 0, '-1', np.nan, '', 'nan', 'NA', 'N/A', 'na', 'n/a', 'null', 'NULL', 'Null', 'NaN', 'None', '?', '-', '--', '---', 'unnamed']
    
    # initialise the results
    results = {}
    
    # analyse every column
    for column in df.columns:
        column_failures = {}
        for missing_value in missing_values:
            if pd.isna(missing_value):
                count = int(df[column].isna().sum(axis=0))
            else:
                count = int((df[column] == missing_value).sum(axis=0))
            if count > 0:
                column_failures[str(missing_value)] = count
        if column_failures:
            results[column] = column_failures

    # convert to a dataframe
    results = pd.DataFrame(results)
            
    # get back final results
    return results

# ----------------------------------------------------------------------------------------------------------------------------

def calculate_geo_x(longitude):
    """
    Converts longitude values to a continuous range from 0° to 360°, 
    avoiding issues at the 180°/-180° datum line.

    Parameters:
        longitude (int, float, numpy.int64, numpy.float64): 
            A longitude value.

    Returns:
        float: The longitude adjusted to the range of 0° to 360°,
               None, if the input value is out of bounds or of an unsupported type.
               
    Notes:
        This docstring was generated with the help of AI and proofread by the author.
    """

    # int, numpy.int64, float and numpy.float64 will be accepted
    if (type(longitude) is int
        or type(longitude) is np.int64
        or type(longitude) is float
        or type(longitude) is np.float64
       ):

        # set the boudaries (360° for one roundturn which is already converted)
        if (longitude <= 360 and longitude >= -180):

            #decide whether the value is from a eastern or western longitude
            if(longitude < 0):
                # western part, correct it
                val = float(longitude + 360)
                return(val)

            else:
                # eastern part, leave it unchanged
                val = float(longitude)
                return(val)
        
        # wrong boundaries
        else:
            return(None)
    
    # undefined input values
    else:
        return(None)  
    
# ----------------------------------------------------------------------------------------------------------------------------

def plot_categorical_distribution(df, category_column, title=None, figsize=(8, 6)):
    """
    Plot the distribution of a categorical variable in a DataFrame.

    This function creates a horizontal bar plot showing the distribution 
    of values for a specified categorical column.

    Parameters:
        df (pandas.DataFrame): 
            The DataFrame containing the data.
        category_column (str): 
            The name of the column containing categorical data.
        title (str, optional): 
            The title of the plot. If None, a default title is used 
            (e.g., 'Distribution of {category_column}').
        figsize (tuple, optional): 
            The size of the figure. Default is (8, 6).

    Returns:
        matplotlib.figure.Figure: 
            The created figure object for the plot.
    Notes:
        This docstring was generated with the help of AI and proofread by the author.
    """
    
    # set the style to a more scientific look
    plt.style.use('seaborn-whitegrid')
    
    # prepare some data
    value_counts = df[category_column].value_counts().reset_index()
    value_counts.columns = [category_column, 'count']
    max_count = value_counts['count'].max()

    # set up the figure
    fig, ax = plt.subplots(figsize=figsize)

    # create a horizontal bar plot
    sns.barplot(x='count',
                y=category_column, 
                data=value_counts,
                color='lightgray',
                edgecolor='darkgray',
               )

    # customize the plot
    ax.set_xlabel('Count', fontsize=10)
    ax.set_ylabel(category_column, fontsize=10)
    if title is None:
        title = f'Categorical distribution of {category_column}'
    ax.set_title(title, fontsize=14)

    # set tick label font sizes
    ax.tick_params(axis='x', which='major', labelsize=10)
    ax.tick_params(axis='y', which='major', labelsize=10)
    
    # add value labels to end of each bar
    for i, v in enumerate(value_counts['count']):
        ax.text(v + max_count*0.02, i, str(v), va='center', fontsize=10)

    # remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # adjust layout
    plt.tight_layout()
    
    # returs figure object
    return fig

# ----------------------------------------------------------------------------------------------------------------------------

def plot_numerical_distribution(df, numerical_column, title=None, kde=True, bins='auto'):
    """
    Plot the distribution of a numerical variable in a DataFrame.

    This function creates a vertical histogram with optional Kernel Density 
    Estimation (KDE) for a specified numerical column.

    Parameters:
        df (pandas.DataFrame): 
            The DataFrame containing the data.
        numerical_column (str): 
            The name of the column containing numerical data.
        title (str, optional): 
            The title of the plot. If None, a default title is used 
            (e.g., 'Histogram of {numerical_column}').
        kde (bool, optional): 
            Whether to overlay a KDE plot on the histogram. Default is True.
        bins (int, str, or sequence, optional): 
            The number or method for calculating bins. Default is 'auto'.

    Returns:
        matplotlib.figure.Figure: 
            The created figure object for the plot.
            
    Notes:
        This docstring was generated with the help of AI and proofread by the author.
    """
    
    # Set the style to a more scientific look
    plt.style.use('seaborn-whitegrid')

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(5, 3))

    # Create a vertical histogram
    sns.histplot(data=df,
                 x=numerical_column, 
                 color='lightgray',
                 edgecolor='darkgray',
                 kde=kde,
                 bins=bins
                )
    # kde drawn?
    if kde:
        # set color for kde line
        ax.lines[0].set_color('#666666')
    
    # Customize the plot
    ax.set_xlabel(numerical_column, fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    if title is None:
        title = f'Histogram of {numerical_column}'
    ax.set_title(title, fontsize=11)

    # Set tick label font sizes
    ax.tick_params(axis='x', which='major', labelsize=8)
    ax.tick_params(axis='y', which='major', labelsize=8)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adjust layout
    plt.tight_layout()
    
    # returs figure object
    return fig

# ----------------------------------------------------------------------------------------------------------------------------

def get_statistical_values(df, column_name):
    """
    Generate descriptive statistics for a specified column in a DataFrame.

    This function calculates and returns descriptive statistics, such as 
    count, mean, standard deviation, minimum, maximum, and percentiles, 
    for a specified column in the input DataFrame.

    Parameters:
        df (pandas.DataFrame): 
            The input DataFrame containing the data.
        column_name (str): 
            The name of the column to compute statistics for.

    Returns:
        pandas.DataFrame: 
            A DataFrame containing the descriptive statistics for the specified column.
            
    Notes:
        This docstring was generated with the help of AI and proofread by the author.
    """
    
    # get the descriptive statistics of the specified column and transpose the result
    description = df[column_name].describe().transpose()
    
    # convert the description to a dataframe
    description_df = pd.DataFrame(description)
    
    # returns statistical description
    return description_df

# ----------------------------------------------------------------------------------------------------------------------------

def check_data_types_of_dataframe(df, expected_types):
    """
    Check the data types of DataFrame columns against a given set of expected types.

    This function iterates over the columns of a DataFrame and compares their 
    actual data types with the expected data types provided. If a column's data 
    type does not match the expected type, a message is printed showing the 
    discrepancy.

    Parameters:
        df (pandas.DataFrame): 
            The input DataFrame to check.
        expected_types (dict): 
            A dictionary mapping column names to their expected data types 
            (e.g., `{'column_name': 'expected_dtype'}`).

    Returns:
        None: Outputs discrepancies directly via print statements.
        
    Notes:
        This docstring was generated with the help of AI and proofread by the author.
    """
    
    print("Checking datatypes (no output means nothing to correct):\n")
    for column, expected_type in expected_types.items():
        actual_type = df[column].dtype

        # Create a readable version of numpy dtype for reporting
        readable_type = np.dtype(actual_type).name
        if not np.issubdtype(actual_type, np.dtype(expected_type).type):
            print(f"{column}: {readable_type} --> {expected_type}")
            
# ----------------------------------------------------------------------------------------------------------------------------

def convert_data_types_of_dataframe(df, columns_dtypes):
    """
    Convert the data types of specified columns in a DataFrame.

    This function attempts to convert the data types of the given columns in 
    the DataFrame to the specified data types. If the conversion is successful, 
    a message is printed. If an error occurs, it prints an error message 
    indicating the issue.

    Parameters:
        df (pandas.DataFrame): 
            The input DataFrame with columns to convert.
        columns_dtypes (dict): 
            A dictionary mapping column names to their target data types 
            (e.g., `{'column_name': 'target_dtype'}`).

    Returns:
        pandas.DataFrame: 
            The DataFrame with converted column data types.
            
    Notes:
        This docstring was generated with the help of AI and proofread by the author.
    """
    
    for column_name, new_dtype in columns_dtypes.items():
        try:
            df[column_name] = df[column_name].astype(new_dtype)
            print(f"Column '{column_name}' converted to {new_dtype}.")
        except Exception as e:
            print(f"Error converting column '{column_name}': {e}")
    return df

# ----------------------------------------------------------------------------------------------------------------------------

def get_categorical_columns(df):
    """
    Retrieve the categorical columns from a DataFrame.

    Parameters:
        df (pandas.DataFrame): 
            The input DataFrame to analyze.

    Returns:
        list: A list of column names that are of categorical data types (`object` or `category`).
        
    Notes:
        This docstring was generated with the help of AI and proofread by the author.
    """ 
    
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return categorical_columns

# ----------------------------------------------------------------------------------------------------------------------------

def get_numerical_columns(df):
    """
    Retrieve the numerical columns from a DataFrame.

    This function identifies and returns a list of columns in the DataFrame 
    that are of numerical data types (`int` or `float`).

    Parameters:
        df (pandas.DataFrame): 
            The input DataFrame to analyze.

    Returns:
        list: A list of column names that are of numerical data types (`int` or `float`).
        
    Notes:
        This docstring was generated with the help of AI and proofread by the author.
    """
    
    numerical_columns = df.select_dtypes(include=['int', 'float']).columns.tolist()
    return numerical_columns

# ----------------------------------------------------------------------------------------------------------------------------

def get_boolean_columns(df):
    """
    Retrieve the boolean columns from a DataFrame.

    Parameters:
        df (pandas.DataFrame): 
            The input DataFrame to analyze.

    Returns:
        list: A list of column names that are of categorical data types ('boolean').
        
    Notes:
        This docstring was generated with the help of AI and proofread by the author.
    """ 
    
    boolean_columns = df.select_dtypes(include=['boolean']).columns.tolist()
    return boolean_columns

# ----------------------------------------------------------------------------------------------------------------------------

def get_temporal_columns(df):
    """
    Retrieve the temporal columns from a DataFrame.

    Parameters:
        df (pandas.DataFrame): 
            The input DataFrame to analyze.

    Returns:
        list: A list of column names that are of temporal data types ('datetime64[ns]').
        
    Notes:
        This docstring was generated with the help of AI and proofread by the author.
    """ 
    
    temporal_columns = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()
    return temporal_columns

# ----------------------------------------------------------------------------------------------------------------------------

def reset_random_seeds(seed=654321, script_name="NOT SET"):
    """
    Reset random seeds for reproducibility across multiple libraries.
    This is useful when consistent results are needed in machine learning experiments.

    Parameters:
        seed (int): The seed value to set. Default is 654321.
        
    Notes:
        This docstring was generated with the help of AI and proofread by the author.
    """

    # set seeds for several random generators
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # log this
    logging.info(f"{script_name}: Seeding random generators with seed={seed}.")

# ----------------------------------------------------------------------------------------------------------------------------

def save_timeseries(data_file, data_set, append_datetime=True, data_dir=None, overwrite_file=False):
    """
    Save a time series dataset to a specified file.

    This function saves the given time series data as a CSV file. Optionally, 
    it can append the current date and time to the filename. The function 
    ensures the specified directory exists before saving.

    Parameters:
        data_file (str): The name of the file to save the dataset.
        data_set (pandas.DataFrame or pandas.Series): The time series data to be saved.
        append_datetime (bool, optional): Whether to append the current date and time 
            to the filename. Defaults to True.
        data_dir (str, optional): The directory to save the file in. If None, the file 
            is saved in the current working directory. Defaults to None.
        overwrite_file (bool, optional): Whether to overwite existing files or not. 
            Defaults to False.

    Returns:
        None

    Raises:
        FileExistsError: If the target file already exists.
        FileNotFoundError: If the specified directory does not exist.
        PermissionError: If there are insufficient permissions to write to the file.
        Exception: For any other unexpected errors during the saving process.

    Notes:
        - The function logs errors instead of re-raising exceptions, which may require
          downstream handling for some applications.
        - The date-time appended to the filename is formatted as "yymmdd-HHMMSS".
        - This docstring was generated with the help of AI and proofread by the author.
    """
    
    # ensure data dir is in place
    if(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    # get the current date in the format yymmdd
    current_date_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S")

    # split filename into name and extension
    filename, extension = data_file.rsplit('.', 1)

    # add current date, if wanted, unles leave filename untouched
    if (append_datetime):
        data_file = f"{filename}_{current_date_time}.{extension}"

    # construct full file path and name
    if(data_dir):
        data_file_full_path = os.path.join(data_dir, data_file)
    else:
        data_file_full_path = data_file

    # try to save the data to the specified file
    try:
        # open file in x-mode, fails if the file already exists
        if(overwrite_file):
            data_set.to_csv(data_file_full_path, index=True, sep=',')
        else:
            with open(data_file_full_path, 'x') as f:
                data_set.to_csv(f, index=True, sep=',')

    # handle several different exceptions
    except FileExistsError:
        # file already exists
        logging.error(f"save_timeseries: Timeseries '{data_file_full_path}' already exists and will not be overwritten!")

    except FileNotFoundError:
        # directory does not exist
        logging.error(f"save_timeseries: The directory for the timeseries '{data_file_full_path}' does not exist!")
        logging.error(f"save_timeseries: Ensure the '{os.path.dirname(data_file_full_path)}' directory exists relative to the program path!")

    except PermissionError:
        # wrong permissions
        logging.error(f"save_timeseries: Permission denied, unable to write to timeseries '{data_file_full_path}'!")

    except Exception as e:
        # any other exceptions
        logging.error(f"save_timeseries: Unable to save data to timeseries {data_file_full_path}, exception: {e}!")

    # show success
    else:
        # successfully saving data
        logging.info(f"save_dataset: Timeseries saved successfully to '{data_file_full_path}'.")

# ----------------------------------------------------------------------------------------------------------------------------

def load_timeseries(data_file, data_dir=None):
    """
    Load a time series dataset from a specified file.

    This function attempts to read a CSV file containing time series data, returning 
    the loaded data as a pandas DataFrame or Series. If an error occurs during the 
    loading process, it logs an appropriate error message.

    Parameters:
        data_file (str): The name of the file containing the time series data.
        data_dir (str, optional): The directory where the file is located. Defaults to None.

    Returns:
        pandas.DataFrame or pandas.Series: The loaded time series data if successful.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        PermissionError: If there are insufficient permissions to read the file.
        Exception: For any other unexpected errors during the loading process.

    Notes:
        - The function logs errors instead of re-raising exceptions, which may require
          downstream handling for some applications.
        - This docstring was generated with the help of AI and proofread by the author.
    """
    
    # construct full file path and name
    data_file_full_path = os.path.join(data_dir, data_file)

    # try to save the data to the specified file
    try:
        # read the data
        loaded_data = pd.read_csv(data_file_full_path, index_col = 0, sep = ',', squeeze=True, low_memory = False)

    # handle several different exceptions
    except FileNotFoundError:
        # directory does not exist
        logging.error(f"load_timeseries: The timeseries '{data_file_full_path}' does not exist!")

    except PermissionError:
        # wrong permissions
        logging.error(f"load_timeseries: Permission denied, unable to load timeseries '{data_file_full_path}'!")

    except Exception as e:
        # any other exceptions
        logging.error(f"load_timeseries: Unable to load timeseries {data_file_full_path}, exception: {e}!")

    # show success
    else:
        # successfully saving data
        logging.info(f"load_timeseries: Timeseries assigned from '{data_file_full_path}'.")
        
        # return loaded data
        return loaded_data    

# ----------------------------------------------------------------------------------------------------------------------------

def summarize_time_series(ts):
    """
    Summarize a time series by returning the number of entries, the first date, and the last date.
    
    Parameters:
    ts (pd.Series): A pandas Series with a datetime index.
    
    Returns:
    dict: A dictionary containing the number of entries, the first date, and the last date.

    Notes
    This docstring was generated with the help of AI and proofread by the author.
    """

    if ts.index.is_monotonic_increasing or ts.index.is_monotonic_decreasing:
        first_date = ts.index[0]
        last_date = ts.index[-1]
    else:
        first_date = ts.index.min()
        last_date = ts.index.max()

    ts_type = type(ts)
    ts_freq_inferred = pd.infer_freq(ts.index)
    
    try:
        freq_index = ts.index.freq or ts.index.inferred_freq
    except Exception as e:
        freq_index = None

    return {
        "number_of_entries": len(ts),
        "first_date": first_date,
        "last_date": last_date,
        "ts_type": ts_type,
        "ts_freq_index": freq_index,
        "ts_freq_inferred": ts_freq_inferred
    }

# ----------------------------------------------------------------------------------------------------------------------------

def analyze_outliers(ts, column_to_analyze, outliers_fraction, log_axis=False):
    """
    Analyze outliers in a time series for a specific column using Isolation Forest.

    Parameters:
        ts (pd.DataFrame): The time series data as a DataFrame.
        column_to_analyze (array-like): The column in the DataFrame to analyze for outliers.
        outliers_fraction (float): The proportion of outliers expected in the data.
        log_axis (bool, optional): If True, the y-axis of the plot is scaled logarithmically. Default is False.

    Returns:
        None: The function displays a plot and prints information about detected outliers.

    Raises:
        ValueError: If the length of `column_to_analyze` is not 1.
    
    Notes:
        This docstring was generated with the help of AI and proofread by the author.
    """

    # get mean value and impute any NAN for this column (and make a deep copy!)
    ts = ts.copy()
    ts_mean = ts[column_to_analyze].mean()
    ts[column_to_analyze] = ts[column_to_analyze].replace(np.nan, ts_mean)

    # calculate isolation forest and mark possible outliers
    clf = IsolationForest(contamination=outliers_fraction, random_state=0)
    ts['is_outlier'] = clf.fit_predict(ts[column_to_analyze])

    # get all possible outliers and show them
    outlier_rows = ts[ts['is_outlier'] == -1]
    num_outliers = outlier_rows.shape[0]
    print(f"Outliers Detection of Feature '{column_to_analyze[0]}' and frac={outliers_fraction}")
    print(f"There are {num_outliers} possible outliers.")
    if(num_outliers>0):
        display(outlier_rows)

    # get inferred freq
    freq = pd.infer_freq(ts.index)
        
    # show plot
    plt.figure(figsize=(16, 4))
    filtered_data = ts[ts[column_to_analyze] != ts_mean]
    
    # datapoints
    plt.scatter(filtered_data.index,
                filtered_data[column_to_analyze],
                label=f"{column_to_analyze[0]}", marker='o', s=5)

    # outliers
    plt.scatter(ts.index[ts['is_outlier'] == -1],
                ts[column_to_analyze][ts['is_outlier'] == -1], 
                color='red', label='Possible Outlier', s=50)
    
    if(log_axis):
        plt.yscale('log')
        
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.title(f"Timeseries ({freq}) with possible Outliers of Feature '{column_to_analyze[0]}' and frac={outliers_fraction}", fontsize=16)
    plt.xlabel(f"Date ({freq})")
    plt.ylabel(f"{column_to_analyze[0]}")
    plt.tight_layout()
    plt.show()
    
    # make line
    print("-"*120)

# ----------------------------------------------------------------------------------------------------------------------------

def plot_timeseries(ts, feature_name, plot_lines=True, logarithmic=True, dot_size=15, topic=None, figsize=(14, 3)):
    """
    Plot a scientific-style time series with customizable options.

    Parameters:
        ts (pd.Series): The time series to plot, indexed by DatetimeIndex.
        feature_name (str): Name of the feature being plotted, used in labels.
        plot_lines (bool, optional): If True, connects points with dashed lines. Default is True.
        logarithmic (bool, optional): If True, applies a logarithmic scale to the y-axis. Default is True.
        dot_size (int, optional): Size of the dots in the scatter plot. Default is 15.
        topic (str, optional): Additional topic information included in the plot title, if provided.
        figsize (tuple, optional): Size of the figure as (width, height). Default is (14, 3).

    Returns:
        None: Displays a plot with the specified time series data.

    Notes:
        This docstring was generated with the help of AI and proofread by the author.
    """

    # Infer frequency from time series
    freq = pd.infer_freq(ts.index)

    # Initialize plot
    fig, ax = plt.subplots(figsize=figsize)

    # Set the style to a scientific look
    plt.style.use('seaborn-whitegrid')

    # Plot the time series
    ax.scatter(ts.index, ts.values, label=None, color="#3333dd", s=dot_size)
    if plot_lines:
        ax.plot(ts.index, ts.values, label=None, color="#3333dd", linestyle='--', linewidth=0.5)

    # Set title and axis labels
    if(topic):
        title=(f"Timeseries ({freq}) for '{feature_name}' from {topic}")
    else:
        title=(f"Timeseries ({freq}) for '{feature_name}'")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date", fontsize=10)
    ylabel = f"log({feature_name})" if logarithmic else feature_name
    ax.set_ylabel(ylabel, fontsize=10)

    # Format the x-axis for dates
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))

    # Add gridlines
    ax.grid(which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    ax.grid(which='minor', linestyle=':', linewidth=0.5, color='gray', alpha=0.5)

    # Customize tick parameters
    ax.tick_params(axis='x', which='major', labelsize=10)
    ax.tick_params(axis='y', which='major', labelsize=10)
    ax.tick_params(axis='x', rotation=30)

    # Set y-axis to log scale if needed
    if logarithmic:
        ax.set_yscale('log')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Tight layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------------

def plot_acf_pacf_series(ts, feature_name, lags=10, title=None, xlabel=None, figsize=(14, 4)):
    """
    Plot the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) for a time series.

    Parameters:
        ts (pd.DataFrame): The time series data containing the feature to be analyzed.
        feature_name (str): The name of the feature in the DataFrame to analyze for ACF and PACF.
        lags (int, optional): The number of lags to include in the ACF and PACF plots. Default is 10.
        title (str, optional): An optional title for the plots.
        xlabel (str, optional): The label for the x-axis. Defaults to "Lags" if not provided.
        figsize (tuple, optional): The size of the figure as (width, height). Default is (14, 4).

    Returns:
        None: Displays the ACF and PACF plots for the given time series feature.

    Notes:
        This docstring was generated with the help of AI and proofread by the author.
    """

    # Initialize the figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # Plot the ACF
    plot_acf(ts[feature_name], ax=axs[0], lags=lags, zero=False)
    axs[0].set_title(f"ACF", fontsize=14)
    if(xlabel):
        axs[0].set_xlabel(xlabel, fontsize=12)
    else:    
        axs[0].set_xlabel("Lags", fontsize=12)
    axs[0].set_ylabel(feature_name, fontsize=12)
    axs[0].set_xticks(range(0, lags + 1, 2))
    axs[0].grid(which='major', axis='x', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)    
    

    # Plot the PACF
    plot_pacf(ts[feature_name], ax=axs[1], lags=lags, zero=False, method='ywm')
    axs[1].set_title(f"PACF", fontsize=14)
    if(xlabel):
        axs[1].set_xlabel(xlabel, fontsize=12)
    else:    
        axs[1].set_xlabel("Lags", fontsize=12)
    axs[1].set_xticks(range(0, lags + 1, 2))
    axs[1].grid(which='major', axis='x', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    # Set a common title if provided
    if title:
        fig.suptitle(f"{title} ({xlabel})", fontsize=16)
        fig.subplots_adjust(top=0.85)

    # Tight layout for better spacing
    plt.tight_layout()

    # Show the plots
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------------

def perform_stationarity_tests(ts):
    """
    Perform stationarity tests on a time series using the Augmented Dickey-Fuller (ADF) test and the
    Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test.

    Parameters:
        ts (array-like): The time series data to be tested for stationarity.

    Returns:
        pd.DataFrame: A DataFrame containing the results of the ADF and KPSS tests, including:
            - Statistic: Test statistic value.
            - p-value: P-value of the test.
            - Number of Lags: Number of lags used in the test.
            - Hypothesis: Indicates if the null hypothesis is rejected.
            - Stationarity: Conclusion on stationarity ("yes" or "no").

    Raises:
        ValueError: If the input series `ts` is not suitable for either test.

    Notes:
        This docstring was generated with the help of AI and proofread by the author.
    """

    # Perform ADF test
    adf_result = adfuller(ts)
    adf_stat, adf_p_value, adf_n_lags, _ = adf_result[:4]
    adf_hypothesis = "Reject H0" if adf_p_value < 0.05 else "FAIL to reject H0"
    adf_conclusion = "yes" if adf_p_value < 0.05 else "no"

    # Perform KPSS test
    kpss_result = kpss(ts, regression='c')
    kpss_stat, kpss_p_value, kpss_n_lags, _ = kpss_result[:4]
    kpss_hypothesis = "Reject H0" if kpss_p_value < 0.05 else "FAIL to reject H0"
    kpss_conclusion = "no" if kpss_p_value < 0.05 else "yes"

    # Combine results into a dictionary
    results = {
        "ADF": [adf_stat, adf_p_value, adf_n_lags, adf_hypothesis, adf_conclusion],
        "KPSS": [kpss_stat, kpss_p_value, kpss_n_lags, kpss_hypothesis, kpss_conclusion]
    }

    # Create DataFrame and transpose it
    results_df = pd.DataFrame(
        results,
        index=["Statistic", "p-value", "Number of Lags", "Hypothesis", "Stationarity"]
    ).T

    return results_df

# ----------------------------------------------------------------------------------------------------------------------------

def impute_timeseries(ts, identification):
    """
    Impute missing values in a time series with the mean of the series.

    Parameters:
        ts (pd.Series): The time series data that may contain NaN values.
        identification (str): An identifier for the time series, used in printed messages.

    Returns:
        pd.Series: The time series with NaN values replaced by the mean of the series.

    Notes:
        Prints the number of NaN values found and indicates whether imputation was necessary.
        This docstring was generated with the help of AI and proofread by the author.
    """

    nan_count = ts.isna().sum()
    if nan_count > 0:
        if(identification):
            print(f"Number of NaN {nan_count}, imputing ({identification}) with mean value!")
        else:
            print(f"Number of NaN {nan_count}, imputing with mean value!")
        ts_mean_value = ts.mean()
        ts = ts.fillna(ts_mean_value)
    else:
        if(identification):
            print(f"No NaN found in ({identification})!")
        else:
            print("No NaN found!")
    
    return ts

# ----------------------------------------------------------------------------------------------------------------------------

def check_stationarity(ts, identification):
    """
    Tests the stationarity of a time series using statistical tests and displays the results.

    Parameters:
        ts (pd.Series): The time series data to be tested for stationarity.
        identification (str): A string identifier for the time series being tested, 
                              used for display purposes.

    Returns:
        tuple:
            - bool: True if the time series is stationary according to both ADF and KPSS tests,
                    False otherwise.
            - pd.DataFrame: A DataFrame containing the results of the stationarity tests.

    Exceptions:
        Any exceptions raised by `perform_stationarity_tests` or `display` will propagate.

    Note:
        This function assumes the existence of a `perform_stationarity_tests` function
        that performs the required statistical tests and returns a DataFrame with 
        stationarity results. The DataFrame is expected to have a structure where the 
        index includes 'ADF' and 'KPSS' rows, and a column named 'Stationarity' indicates 
        the stationarity status ('yes' or 'no').
    """    
    print(f"Testing stationarity for {identification}")
    test_results = perform_stationarity_tests(ts)
    display(test_results)
    adf_stationary = test_results.loc['ADF', 'Stationarity'] == 'yes'
    kpss_stationary = test_results.loc['KPSS', 'Stationarity'] == 'yes'
    return adf_stationary and kpss_stationary, test_results

# ----------------------------------------------------------------------------------------------------------------------------

def handle_stationarity(ts1, ts2, ts1_identification, ts2_identification):
    """
    Check and handle stationarity for two time series by assessing them with
    Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests.
    Apply differencing if non-stationarity is detected.

    Parameters:
        ts1 (pd.Series): The first time series to check for stationarity.
        ts2 (pd.Series): The second time series to check for stationarity.
        ts1_identification (str): An identifier for the first time series, used in printed messages.
        ts2_identification (str): An identifier for the second time series, used in printed messages.

    Returns:
        tuple: A tuple containing two pd.Series:
            - The first time series after potential differencing.
            - The second time series after potential differencing.

    Notes:
        This function will display detailed results of the stationarity tests and whether
        differencing was applied.
        This docstring was generated with the help of AI and proofread by the author.
    """

    # First check stationarity for ts1
    print(f"Initial stationarity check for {ts1_identification}:")
    ts1_stationary, ts1_results = check_stationarity(ts1, ts1_identification)
    print()

    # First check stationarity for ts2
    print(f"Initial stationarity check for {ts2_identification}:")
    ts2_stationary, ts2_results = check_stationarity(ts2, ts2_identification)
    print()

    # Determine if either ts1 or ts2 is not stationary
    if not ts1_stationary or not ts2_stationary:
        print("At least one time series is not stationary. Applying differencing to both time series.")
        ts1_result = ts1.diff().dropna()
        ts2_result = ts2.diff().dropna()
        
        # Second check stationarity for ts1
        print(f"Second stationarity check for {ts1_identification}:")
        ts1_stationary_after, ts1_results_after = check_stationarity(ts1_result, ts1_identification)
        if ts1_stationary_after:
            print(f"{ts1_identification} is now stationary after differencing.")
        else:
            print(f"{ts1_identification} is still not stationary after differencing.")

        print()

        # Second check stationarity for ts2
        print(f"Second stationarity check for {ts2_identification}:")
        ts2_stationary_after, ts2_results_after = check_stationarity(ts2_result, ts2_identification)
        if ts2_stationary_after:
            print(f"{ts2_identification} is now stationary after differencing.")
        else:
            print(f"{ts2_identification} is still not stationary after differencing.")

        print("=" * 80)
        print("Final Results:")
        print(f"{ts1_identification} Stationary: {'Yes' if ts1_stationary_after else 'No'}")
        print(f"{ts2_identification} Stationary: {'Yes' if ts2_stationary_after else 'No'}")
    
    else:
        print("Both time series are stationary. No differencing applied.")
        ts1_result = ts1
        ts2_result = ts2

        print("=" * 80)
        print("Final Results:")
        print(f"{ts1_identification} Stationary: {'Yes' if ts1_stationary else 'No'}")
        print(f"{ts2_identification} Stationary: {'Yes' if ts2_stationary else 'No'}")

        
    print("=" * 80)

    return ts1_result, ts2_result

# ----------------------------------------------------------------------------------------------------------------------------

def calculate_ccf_results(ts1, ts2, period_count, confidence_level=0.95, smoothing_lags=1):
    """
    Calculate cross-correlation function (CCF) results for two time series, including confidence intervals.

    Parameters:
        ts1 (array-like): The first time series data.
        ts2 (array-like): The second time series data.
        period_count (int): The number of observations, typically representing the sample size.
        confidence_level (float, optional): The confidence level for the cross-correlation intervals. Default is 0.95.
        smoothing_lags (int, optional): The number of lags to use for smoothing the confidence intervals. Default is 1.

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - "Lag": The lag value corresponding to each cross-correlation value.
            - "ccf12": Cross-correlation values from ts1 to ts2.
            - "ci12lower": Lower bound of the confidence interval for the ccf12 values.
            - "ci12upper": Upper bound of the confidence interval for the ccf12 values.
            - "ccf21": Cross-correlation values from ts2 to ts1.
            - "ci21lower": Lower bound of the confidence interval for the ccf21 values.
            - "ci21upper": Upper bound of the confidence interval for the ccf21 values.

    Raises:
        ValueError: If `period_count` is less than or equal to 3, making the standard error undefined.
        Warning: If `smoothing_lags` is greater than the number of lags, results may be less reliable.

    Notes:
        The function uses the Fisher Z-transformation to calculate confidence intervals, which are then
        transformed back to the correlation scale. Smoothing of the confidence intervals can be applied
        using a centered rolling mean over the specified number of lags.
        This docstring was generated with the help of AI and proofread by the author.
    """
    
    # Calculate cross-correlation in both directions
    ccf_ts1_ts2 = ccf(ts1, ts2)
    ccf_ts2_ts1 = ccf(ts2, ts1)
    
    # Confidence interval calculation function
    def cross_correlation_confidence_intervals(cross_corr_values, n, confidence_level, smoothing_lags):
        """
        Calculates confidence intervals for cross-correlation values based on the Fisher Z-transformation.

        Parameters:
            cross_corr_values (list or np.ndarray): The cross-correlation values for which confidence intervals
                need to be computed.
            n (int): The number of observations used in calculating the cross-correlation.
            confidence_level (float): The desired confidence level for the intervals (e.g., 0.95 for 95% confidence).
            smoothing_lags (int): The number of lags to use for smoothing the confidence intervals.
                If greater than 1, a rolling mean is applied.

        Returns:
            pd.DataFrame: A DataFrame containing the following columns:
                - "Lag": The lag value corresponding to each cross-correlation value.
                - "Lower Bound": The lower bound of the confidence interval for the cross-correlation at the lag.
                - "Upper Bound": The upper bound of the confidence interval for the cross-correlation at the lag.

        Raises:
            ValueError: If `n` is less than or equal to 3, as this would make the standard error undefined.
            Warning: If `smoothing_lags` is greater than the number of lags, results may be less reliable.

        Notes:
            - The Fisher Z-transformation is used to calculate the confidence intervals, which are then
              transformed back to the correlation scale.
            - Smoothing is applied using a centered rolling mean over the specified number of lags.
            - If no smoothing is required, set `smoothing_lags` to 1.

        """        
        z_critical = norm.ppf(1 - (1 - confidence_level) / 2)
        se = 1 / np.sqrt(n - 3)
        confidence_intervals = []

        for lag, r in enumerate(cross_corr_values):
            z = 0.5 * np.log((1 + r) / (1 - r))
            z_lower = z - z_critical * se
            z_upper = z + z_critical * se
            r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
            r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
            confidence_intervals.append((lag, r_lower, r_upper))

        ci_df = pd.DataFrame(confidence_intervals, columns=["Lag", "Lower Bound", "Upper Bound"])

        if smoothing_lags > 1:
            ci_df["Lower Bound"] = ci_df["Lower Bound"].rolling(window=smoothing_lags, center=True).mean()
            ci_df["Upper Bound"] = ci_df["Upper Bound"].rolling(window=smoothing_lags, center=True).mean()

        return ci_df

    # Calculate confidence intervals for both directions
    ci_ts1_ts2 = cross_correlation_confidence_intervals(ccf_ts1_ts2, period_count, confidence_level, smoothing_lags)
    ci_ts2_ts1 = cross_correlation_confidence_intervals(ccf_ts2_ts1, period_count, confidence_level, smoothing_lags)

    ccf_results = pd.DataFrame({
        "Lag": np.arange(len(ccf_ts1_ts2)),
        "ccf12": ccf_ts1_ts2,
        "ci12lower": ci_ts1_ts2["Lower Bound"],
        "ci12upper": ci_ts1_ts2["Upper Bound"],
        "ccf21": ccf_ts2_ts1,
        "ci21lower": ci_ts2_ts1["Lower Bound"],
        "ci21upper": ci_ts2_ts1["Upper Bound"]
    })

    return ccf_results

# ----------------------------------------------------------------------------------------------------------------------------

# def top_max_min_indices_as_dataframe(values, top_n=2, range_limit=None):
#     """
#     Extracts the indices of the top N maximum and minimum values within a specified range
#     and returns them as a DataFrame.

#     Parameters:
#         values (list or np.ndarray): A list or array of numerical values to analyze.
#         top_n (int, optional): The number of top maximum and minimum values to extract.
#             Default is 2.
#         range_limit (int, optional): The range within the values to consider for the analysis.
#             If None, the entire range of `values` is used. Default is None.

#     Returns:
#         pd.DataFrame: A DataFrame containing the following columns:
#             - "Type": Indicates whether the value is a "Max" or "Min".
#             - "Order": The rank of the value (1 for highest/lowest, 2 for second highest/lowest, etc.).
#             - "Value": The numerical value of the element.
#             - "Index": The index of the value in the original `values` list or array.
    
#     Raises:
#         ValueError: If `top_n` is less than 1 or if `range_limit` is invalid.
#     """
    
#     if range_limit is None:
#         range_limit = len(values)
    
#     restricted_values = values[:range_limit]
#     max_indices = sorted(range(len(restricted_values)), key=lambda i: restricted_values[i], reverse=True)[:top_n]
#     max_values = [(restricted_values[i], i, "Max", rank + 1) for rank, i in enumerate(max_indices)]
#     min_indices = sorted(range(len(restricted_values)), key=lambda i: restricted_values[i])[:top_n]
#     min_values = [(restricted_values[i], i, "Min", rank + 1) for rank, i in enumerate(min_indices)]
#     combined = max_values + min_values

#     df = pd.DataFrame(combined, columns=["Value", "Index", "Type", "Order"])
#     df = df[["Type", "Order", "Value", "Index"]]
#     return df

# ----------------------------------------------------------------------------------------------------------------------------

def cross_corr_ci_lag_varying(n, lags, confidence=0.95):
    """
    Calculate lag-varying confidence intervals for cross-correlation values based on the Fisher Z-transformation.

    Parameters:
        n (int): The total number of observations in the time series.
        lags (int): The number of lags for which to compute confidence intervals.
        confidence (float, optional): The desired confidence level for the intervals (e.g., 0.95 for 95% confidence).
                                      Default is 0.95.

    Returns:
        dict: A dictionary where keys are lag values (int) and values are tuples of the form (lower_bound, upper_bound).
              For lags with insufficient data, the value is (None, None).

    Notes:
        The effective sample size decreases with lag, so the standard error and confidence interval adjust
        accordingly. If `n - lag` is less than or equal to zero, indicating insufficient data, the confidence
        interval for that lag will be set to (None, None).
        This docstring was generated with the help of AI and proofread by the author.
    """

    z_critical = norm.ppf(1 - (1 - confidence) / 2)
    ci_bands = {}

    for lag in range(lags):
        # Effective sample size decreases with lag
        n_effective = n - lag
        if n_effective > 0:
            se = 1 / np.sqrt(n_effective)
            ci_band = z_critical * se
            ci_bands[lag] = (-ci_band, ci_band)
        else:
            # Not enough data for this lag
            ci_bands[lag] = (None, None)

    return ci_bands

# ----------------------------------------------------------------------------------------------------------------------------

def show_crosscorrelation_results(
    ts1, ts2, ts1_id, ts2_id, top_n=10, figsize=(18, 5), legend_cols=3,
    lag_limits=None, show_min_max_table=True, order_min_max_table=True, plot_type='line', plotwidth=2, plotsize=3,
    fft_highpower_lags=None, show_fft_highpower_lags_analysis=False,
    clustername='Not set', scope='Not set', figure_save_dir='.', script_name=None
):
    """
    Display and analyze cross-correlation results between two time series, with options for plotting and analysis details.

    Parameters:
        ts1 (pd.Series): The first time series data.
        ts2 (pd.Series): The second time series data.
        ts1_id (str): Identifier for the first time series, used in plots and messages.
        ts2_id (str): Identifier for the second time series, used in plots and messages.
        top_n (int, optional): Number of top positive and negative cross-correlation values to highlight. Default is 10.
        figsize (tuple, optional): The size of the figure as (width, height). Default is (18, 5).
        legend_cols (int, optional): Number of columns in the legend. Default is 3.
        lag_limits (tuple, optional): The range of lags to display, specified as (min_lag, max_lag).
        show_min_max_table (bool, optional): Whether to display a table with min/max cross-correlation values. Default is True.
        order_min_max_table (bool, optional): Whether to order the min/max table by lag. Default is True.
        plot_type (str, optional): Type of plot for cross-correlation coefficients. Options: 'line', 'scatter', 'stem'. Default is 'line'.
        plotwidth (int, optional): Line width for the plot. Default is 2.
        plotsize (int, optional): Marker size for plots. Default is 3.
        fft_highpower_lags (list, optional): Lags corresponding to high power in FFT analysis.
        show_fft_highpower_lags_analysis (bool, optional): Whether to display FFT high power lags analysis. Default is False.
        clustername (str, optional): Name of the cluster for context in titles. Default is 'Not set'.
        scope (str, optional): Scope of the analysis for context in titles. Default is 'Not set'.
        figure_save_dir (str, optional): Directory to save the figure. Default is '.'.
        script_name (str, optional): Name of the script for logging purposes.

    Returns:
        None: The function displays plots and outputs additional analysis tables if specified.

    Notes:
        The function computes cross-correlation coefficients using convolution and adjusts for variance.
        It highlights significant and non-significant correlations, saves the figure, and optionally analyzes
        FFT high power lags.
        This docstring was generated with the help of AI and proofread by the author.
    """
   
    # get timeseries frequency
    period_name = pd.infer_freq(ts1.index)

    # create title
    title = f"CCF of (ts1,ts2) for freq={period_name}, lag-range={lag_limits}, top_n={top_n}\nts1 = {ts1_id}, ts2 = {ts2_id}\nScope = {scope}, Cluster = {clustername}"
    
    # compute cross-correlation
    ccf_coeff = correlate(ts1, ts2, mode='full', method='auto') / (np.std(ts1) * np.std(ts2) * len(ts1))
    
    # lags
    lags = np.arange(-len(ts1) + 1, len(ts2))
   
    # calculate ci bands
    ci_bands = pd.DataFrame(cross_corr_ci_lag_varying(len(ts1), len(ts1)-1, confidence=0.95)).T

    # make series and dataframe with appropriate length and 0 at the center
    lower_band = ci_bands.iloc[:, 0]
    upper_band = ci_bands.iloc[:, 1]
    lower_band_series = np.concatenate((lower_band[::-1], [lower_band[0]], lower_band))
    upper_band_series = np.concatenate((upper_band[::-1], [upper_band[0]], upper_band))
    ci_bands_df = pd.DataFrame({
        'ci_lower': lower_band_series,
        'ci_upper': upper_band_series
    })
    
    # build final dataframe with all information
    ccf_all_data = pd.DataFrame({
        "lag": lags,
        "ccf_coeff": ccf_coeff
    })
    ccf_all_data = pd.concat([ccf_all_data, ci_bands_df], axis=1)
    
    # add a 'significant' column based on the condition
    ccf_all_data['sig'] = ccf_all_data.apply(
        lambda row: 'sig' if row['ccf_coeff'] < row['ci_lower'] or row['ccf_coeff'] > row['ci_upper'] else '',
        axis=1
    )
    
    # get the rows with the lowest ccf_coefficient, add a rank, and label as 'min'
    lowest_rows = ccf_all_data.nsmallest(top_n, 'ccf_coeff').copy()
    lowest_rows['rank'] = range(1, len(lowest_rows) + 1)
    lowest_rows['label'] = 'min'

    # get the rows with the highest ccf_coefficient, add a rank, and label as 'max'
    highest_rows = ccf_all_data.nlargest(top_n, 'ccf_coeff').copy()
    highest_rows['rank'] = range(1, len(highest_rows) + 1)
    highest_rows['label'] = 'max'
    
    # combine the lowest and highest rows into a new dataframe
    ccf_all_data_min_max = pd.concat([lowest_rows, highest_rows]).reset_index(drop=True)
    
    # sort the table if wanted
    if order_min_max_table:
        ccf_all_data_min_max = ccf_all_data_min_max.sort_values(by='lag', ascending=True)
    
    # calculate ccf_min and ccf_max
    ccf_min_value = min(ccf_all_data['ccf_coeff'])
    ccf_max_value = max(ccf_all_data['ccf_coeff'])
    
    # extract indices of the minimal and maximal values
    max_indices = np.argsort(ccf_all_data['ccf_coeff'])[-top_n:][::-1]
    min_indices = np.argsort(ccf_all_data['ccf_coeff'])[:top_n]
    
    # Filter for max ccf_coeff within the bounds
    max_data = ccf_all_data.loc[max_indices]
    valid_max_indices_unsignificant = max_data[
        (max_data['ccf_coeff'] >= max_data['ci_lower']) &
        (max_data['ccf_coeff'] <= max_data['ci_upper'])].index
    
    # Filter for max ccf_coeff outside the bounds
    valid_max_indices_significant = max_data[
        (max_data['ccf_coeff'] < max_data['ci_lower']) |
        (max_data['ccf_coeff'] > max_data['ci_upper'])].index

    # Filter for min ccf_coeff within the bounds
    min_data = ccf_all_data.loc[min_indices]
    valid_min_indices_unsignificant = min_data[
        (min_data['ccf_coeff'] >= min_data['ci_lower']) &
        (min_data['ccf_coeff'] <= min_data['ci_upper'])].index

    # Filter for min ccf_coeff outside the bounds
    valid_min_indices_significant = min_data[
        (min_data['ccf_coeff'] < min_data['ci_lower']) |
        (min_data['ccf_coeff'] > min_data['ci_upper'])].index
    
    # ----------------------------------------------
    
    # create figure
    fig, ax = plt.subplots(figsize=figsize)

    # set the style to a more scientific look
    plt.style.use('seaborn-whitegrid')
    
    # calculate and set dynamic ylim with an additional margin
    ylim_margin = 0.3 * (ccf_max_value - ccf_min_value)
    plt.ylim([ccf_min_value - ylim_margin, ccf_max_value + ylim_margin])
    
    # set xlim values
    if lag_limits:
        plt.xlim(lag_limits)

    # plot ccf by selected type
    if(plot_type == 'scatter'):
        plt.scatter(ccf_all_data['lag'], ccf_all_data['ccf_coeff'], s=plotsize, label='Crosscorrelation coefficient')
    elif(plot_type == 'stem'):
        stem = plt.stem(ccf_all_data['lag'], ccf_all_data['ccf_coeff'], basefmt=" ", markerfmt='o', label='Crosscorrelation coefficient')
        stem.markerline.set_markersize(plotsize)
        stem.stemlines.set_linewidth(plotwidth)
    else:
        line = plt.plot(ccf_all_data['lag'], ccf_all_data['ccf_coeff'], linewidth=plotwidth, label='Crosscorrelation coefficient')
 
    # highlight the non-significant top max values with red circles
    plt.scatter(
        ccf_all_data['lag'][valid_max_indices_unsignificant], ccf_all_data['ccf_coeff'][valid_max_indices_unsignificant], 
        color='red', s=plotsize*20, edgecolors='red', zorder=10, facecolors='#dddddd',label=f'Top n={top_n} max values (non-significant)')

    # highlight the significant top max values with red dots
    plt.scatter(
        ccf_all_data['lag'][valid_max_indices_significant], ccf_all_data['ccf_coeff'][valid_max_indices_significant], 
        color='red', zorder=10, s=plotsize*20, label=f'Top n={top_n} max values (significant)')

    # highlight the non-significant top min values with blue circles
    plt.scatter(
        ccf_all_data['lag'][valid_min_indices_unsignificant], ccf_all_data['ccf_coeff'][valid_min_indices_unsignificant], 
        color='blue', s=plotsize*20, edgecolors='blue', zorder=10, facecolors='#dddddd',label=f'Top n={top_n} min values (non-significant)')

    # highlight the significant top min values with blue dots
    plt.scatter(
        ccf_all_data['lag'][valid_min_indices_significant], ccf_all_data['ccf_coeff'][valid_min_indices_significant], 
        color='blue', zorder=10, s=plotsize*20, label=f'Top n={top_n} min values (significant)')
    
    # plot confidence intervall
    plt.fill_between(ccf_all_data['lag'], ccf_all_data['ci_lower'], ccf_all_data['ci_upper'], color='gray', alpha=0.2, label='95% confidence interval')

    # plot center line
    plt.axvline(0, color='black', linestyle='--', label='0 lag')

    # plot 0-axis
    plt.axhline(0, color='darkgray', linestyle='--', label='0 ccf coefficient')
    
    # plot lag corresponding to the max CCF value
    ccf_max_lag = lags[np.argmax(ccf_all_data['ccf_coeff'])]
    plt.axvline(ccf_max_lag, color='red', linestyle='--', label='Global max ccf coefficient')
    
    # plot lag corresponding to the min CCF value
    ccf_min_lag = lags[np.argmin(ccf_all_data['ccf_coeff'])]
    plt.axvline(ccf_min_lag, color='blue', linestyle='--', label='Global min ccf coefficient')
    
    # annotate the global min/max values
    if lag_limits:
        if lag_limits[0] <= ccf_max_lag <= lag_limits[1]:
            plt.text(ccf_max_lag, ccf_max_value * 1.20, f'   global max lag = {ccf_max_lag} ({ccf_max_value:.5f})', color='red', ha='left', va='bottom', fontsize=10)
        if lag_limits[0] <= ccf_min_lag <= lag_limits[1]:
            plt.text(ccf_min_lag, ccf_min_value * 1.25, f'   global min lag = {ccf_min_lag} ({ccf_min_value:.5f})', color='blue', ha='left', va='top', fontsize=10)
    else:
        plt.text(ccf_max_lag, ccf_max_value * 1.20, f'   global max lag = {ccf_max_lag} ({ccf_max_value:.5f})', color='red', ha='left', va='bottom', fontsize=10)
        plt.text(ccf_min_lag, ccf_min_value * 1.25, f'   global min lag = {ccf_min_lag} ({ccf_min_value:.5f})', color='blue', ha='left', va='top', fontsize=10)
    
    # show infos
    plt.title(title, fontsize=16)
    plt.xlabel('for negative values: ts2 leads <----    Lag (Shift)    ----> for positive values: ts1 leads', fontsize=14)
    plt.ylabel('CCF coefficient', fontsize=14)
    
    # move the legend below the figure
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=legend_cols)
    
    # remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # save figure to disk
    figurename = f"50_CCF_({ts1_id}_{ts2_id})_({period_name})_({lag_limits}).png".replace(" ", "_").replace(",", "_").replace("__", "_").replace("((", "(").replace("))", ")")
    plt.savefig(os.path.join(figure_save_dir, figurename), format='png', dpi=150, bbox_inches='tight')
    logging.info(f"{script_name}: {figurename} to './{figure_save_dir}'.")
    
    # show plot
    plt.show()
    
    # display the dataframe
    if show_min_max_table:
        print("CCF min-max values for all lags")
        display(ccf_all_data_min_max.style.hide(axis="index"))
        
    # get infos for fft_highpower_lags
    if show_fft_highpower_lags_analysis:
        print("FFT Highpower Lags Analysis (global)")
        fft_highpower_lags_rows_global = ccf_all_data[ccf_all_data['lag'].isin(fft_highpower_lags)]
        display(fft_highpower_lags_rows_global.style.hide(axis="index"))

        print("FFT Highpower Lags Analysis (local)")
        fft_highpower_lags_rows_local = ccf_all_data_min_max[ccf_all_data_min_max['lag'].isin(fft_highpower_lags)]
        display(fft_highpower_lags_rows_local.style.hide(axis="index"))
        
    # end of procedure
    print("\n\n")
        
# ----------------------------------------------------------------------------------------------------------------------------

def analyze_and_plot_fft(ts_data, num_n, identification='', xlimitspd=None, xlimitsfd=None):
    """
    Analyze and plot the Fast Fourier Transform (FFT) of a time series to identify and visualize prominent frequencies.

    Parameters:
        ts_data (pd.Series): The time series data to be transformed using FFT.
        num_n (int): The number of most prominent frequencies to identify and highlight in the plot.
        identification (str, optional): An identifier for the time series, used in plot titles. Default is ''.
        xlimitspd (tuple, optional): Limits for the x-axis in the period domain plot, specified as (min_period, max_period).
        xlimitsfd (tuple, optional): Limits for the x-axis in the frequency domain plot, specified as (min_freq, max_freq).

    Returns:
        None: Displays FFT plots for the frequency and period domains.

    Notes:
        The function computes FFT and visualizes both the frequency domain and period domain representation.
        Key frequencies are highlighted and printed with their corresponding periods and amplitudes.
        This docstring was generated with the help of AI and proofread by the author.
    """
    
    # Prepare your data (values and number N)
    y = ts_data.values
    N = len(y)

    # Compute the FFT
    Y_fft = np.fft.fft(y)

    # Construct the frequency axis in cycles/interval
    freqs = np.fft.fftfreq(N, d=1)

    # Extract positive frequencies and magnitudes
    positive_freqs = freqs[1 : N // 2]
    fft_magnitude = np.abs(Y_fft)[1 : N // 2]

    # Identify and retrieve most prominent frequencies
    peak_indices = np.argpartition(fft_magnitude, -num_n)[-num_n:]
    peak_indices = peak_indices[np.argsort(fft_magnitude[peak_indices])]
    prominent_freqs = positive_freqs[peak_indices]
    prominent_amps = fft_magnitude[peak_indices]

    # Show them sorted by T=1/f
    freq_amp_pairs = list(zip(prominent_freqs, prominent_amps))
    sorted_pairs = sorted(freq_amp_pairs, key=lambda x: 1/x[0])
    print("\nMost prominent frequencies (cycles/interval):")
    for f, amp in sorted_pairs:
        print(f"freq = {f:.6f}, T = {1/f:.0f}, amp = {amp:.0f}")

    # Plot the FFT spectrum in the frquency domain (fd)
    plt.figure(figsize=(16, 6))
    markerline, stemlines, baseline = plt.stem(positive_freqs, fft_magnitude, use_line_collection=True, label='FFT')
    plt.setp(markerline, marker='o', markersize=3, markeredgecolor='#4444cc', markerfacecolor='#4444cc')
    plt.setp(stemlines, linewidth=0.5, color='#4444cc')
    plt.setp(baseline, color='gray', linewidth=1)
    plt.scatter(prominent_freqs, prominent_amps, color='red', s=40, zorder=10, label='Top frequencies')
    plt.xlabel('Frequency (cycles/interval)')
    if(xlimitsfd):
        plt.xlim(xlimitsfd)
    plt.ylabel('FFT Magnitude')
    plt.title(f'FFT of {identification} for the Frequency Domain')
    plt.legend()
    plt.grid(True)
    plt.minorticks_on()
    plt.tick_params(axis='both', which='minor', length=4, color='black')
    plt.tick_params(axis='both', which='major', length=8, color='black')
    plt.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, color='#333333')
    plt.grid(True, which='minor', axis='x', linestyle=':', linewidth=0.3, color='#333333')
    plt.show()
    
    # plot of the FFT spectrum in the period domain (pd)
    plt.figure(figsize=(16, 6))
    markerline, stemlines, baseline = plt.stem(1/positive_freqs, fft_magnitude, use_line_collection=True, label='FFT')
    plt.setp(markerline, marker='o', markersize=3, markeredgecolor='#4444cc', markerfacecolor='#4444cc')
    plt.setp(stemlines, linewidth=0.5, color='#4444cc')
    plt.setp(baseline, color='gray', linewidth=0.5)
    plt.scatter(1/prominent_freqs, prominent_amps, color='red', s=40, zorder=10, label='Top periods')
    plt.xscale('log')
    if(xlimitspd):
        plt.xlim(xlimitspd)
    plt.xlabel('Period (interval)')
    plt.ylabel('FFT Magnitude')
    plt.title(f'FFT of {identification} for the Period Domain')
    plt.grid(True)
    plt.minorticks_on()
    plt.tick_params(axis='both', which='minor', length=4, color='black')
    plt.tick_params(axis='both', which='major', length=8, color='black')
    plt.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, color='#333333')
    plt.grid(True, which='minor', axis='x', linestyle=':', linewidth=0.3, color='#333333')
    plt.legend()
    plt.show()
    
# ----------------------------------------------------------------------------------------------------------------------------

def perform_heteroskedasticity_tests(ts):
    """
    Perform heteroskedasticity tests on a time series dataset.

    This function conducts three different tests to detect heteroskedasticity in a given time series:
    - ARCH (Autoregressive Conditional Heteroskedasticity) test
    - Breusch-Pagan test
    - White test

    Each test evaluates the null hypothesis of homoscedasticity (constant variance of residuals).

    Parameters:
        ts (pandas.Series): The time series data to test for heteroskedasticity.

    Returns:
        pandas.DataFrame: A DataFrame summarizing the results of the three tests, with the following columns:
            - "Statistic": Test statistic for each test.
            - "p-value": p-value corresponding to the test statistic.
            - "Hypothesis": Outcome of hypothesis testing (e.g., "Reject H0" or "FAIL to reject H0").
            - "Heteroskedasticity": Indicator ("yes" or "no") of whether heteroskedasticity is detected.

    The tests' results are provided in rows labeled "ARCH", "Breusch-Pagan", and "White".

    Exceptions:
        If any test fails due to an error (e.g., invalid data or model fitting issues), the respective results
        will have `NaN` values for the statistic and p-value, and "failed" for hypothesis and heteroskedasticity columns.
    
    Notes:
        This docstring was generated with the help of AI and proofread by the author.
    """
    
    # Perform ARCH test
    # p < 0.05 rejects H0, suggesting presence of ARCH effects (heteroscedasticity).
    try:
        arch_test_result = het_arch(ts.values)
        arch_stat, arch_p_value, arch_df = arch_test_result[:3]
        arch_hypothesis = "Reject H0" if arch_p_value < 0.05 else "FAIL to reject H0"
        arch_conclusion = "yes" if arch_p_value < 0.05 else "no"
    except:
        arch_stat = np.nan
        arch_p_value = np.nan
        arch_hypothesis = "failed"
        arch_conclusion = "failed"
    
    # Perform Breusch-Pagan test
    # p < 0.05 rejects null hypothesis of homoscedasticity, suggesting presence of heteroscedasticity
    try:
        model = AutoReg(ts, lags=3).fit()
        bp_test = het_breuschpagan(model.resid, sm.add_constant(model.fittedvalues))
        bp_stat = bp_test[0]
        bp_pvalue = bp_test[1]
        bp_hypothesis = "Reject H0" if bp_pvalue < 0.05 else "FAIL to reject H0"
        bp_conclusion = "yes" if bp_pvalue < 0.05 else "no"
    except:
        bp_stat = np.nan
        bp_pvalue = np.nan
        bp_hypothesis = "failed"
        bp_conclusion = "failed"

    # White test
    # p < 0.05 rejects null hypothesis of homoscedasticity, suggesting presence of heteroscedasticity
    try:
        model = AutoReg(ts, lags=3).fit()
        w_test = het_white(model.resid, sm.add_constant(model.fittedvalues))
        w_stat = w_test[0]
        w_pvalue = w_test[1]
        w_hypothesis = "Reject H0" if w_pvalue < 0.05 else "FAIL to reject H0"
        w_conclusion = "yes" if w_pvalue < 0.05 else "no"
    except:
        w_stat = np.nan
        w_pvalue = np.nan
        w_hypothesis = "failed"
        w_conclusion = "failed"
        
    # Combine results into a dictionary
    results = {
        "ARCH": [arch_stat, f'{arch_p_value:.6f}', arch_hypothesis, arch_conclusion],
        "Breusch-Pagan": [bp_stat, f'{bp_pvalue:.6f}', bp_hypothesis, bp_conclusion],
        "White": [w_stat, f'{w_pvalue:.6f}', w_hypothesis, w_conclusion]
    }

    # Create DataFrame and transpose it
    results_df = pd.DataFrame(
        results,
        index=["Statistic", "p-value", "Hypothesis", "Heteroskedasticity"]
    ).T

    return results_df

# ----------------------------------------------------------------------------------------------------------------------------

def plot_stft(y, fs=1, window_size=128, noverlap=64, title=None):
    """
    Plot the Short-Time Fourier Transform (STFT) of a signal.

    Parameters:
        y (array-like): The input signal to analyze with STFT.
        fs (float, optional): The sampling frequency of the signal. Default is 1.
        window_size (int, optional): The number of samples per segment. Default is 128.
        noverlap (int, optional): The number of samples to overlap between segments. Default is 64.
        title (str, optional): Title for the plot to provide context or identification.

    Returns:
        None: Displays a plot of the STFT magnitude on a logarithmic scale.

    Notes:
        The STFT is computed using a Hamming window and displayed with a logarithmic color scale
        to accentuate variations in magnitude. The function uses `plt.pcolormesh` for a smooth
        shading of the time-frequency representation.
        This docstring was generated with the help of AI and proofread by the author.
    """

    # Calculate Short-Time Fourier Transform (STFT)
    f, t, Zxx = stft(y, fs=fs, nperseg=window_size, noverlap=noverlap, window='hamming')
    magnitude = np.abs(Zxx)

    # Plot STFT
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, magnitude, cmap='inferno', 
                   norm=LogNorm(vmin=magnitude.min() + 1e-3, vmax=magnitude.max()), shading='gouraud')
    plt.title(f'Short-Time Fourier Transform (STFT)\n{title}')
    plt.ylabel('Frequency (cycles/period)')
    plt.xlabel('Time (periods)')
    plt.colorbar(label='STFT Magnitude (log scale)')
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------------

def save_notebook_as_html(script_name, output_dir):
    """
    Save a Jupyter Notebook as an HTML file.

    Parameters:
        script_name (str): The filename of the Jupyter Notebook to be converted, including its path.
        output_dir (str): The directory where the resulting HTML file will be saved.

    Returns:
        None: The function creates and saves an HTML file version of the notebook in the specified directory.

    Notes:
        - The function reads a notebook, converts it to HTML format, and saves it in the specified output directory.
        - The output HTML file will have the same base name as the input notebook, but with an '.html' extension.
        - The function ensures the output directory exists before writing the HTML file.
        This docstring was generated with the help of AI and proofread by the author.
    """
    
    # Load the notebook
    with open(script_name, 'r', encoding='utf-8') as f:
        notebook_content = nbformat.read(f, as_version=4)

    # Convert to HTML
    html_exporter = HTMLExporter()
    html_body, resources = html_exporter.from_notebook_node(notebook_content)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct the output HTML filename
    html_filename = os.path.join(output_dir, os.path.basename(script_name).replace('.ipynb', '.html'))

    # Save the HTML
    with open(html_filename, 'w', encoding='utf-8') as f:
        f.write(html_body)

    print(f"Notebook saved as HTML: {html_filename}")
    
# ----------------------------------------------------------------------------------------------------------------------------

def print_loaded_parameters(parameters):
    """
    Print the loaded parameters from a given object or module, excluding built-in attributes.

    Parameters:
        parameters (object): An object or module containing parameters as attributes.

    Returns:
        None: The function prints the names and values of the parameters in the format 'parameters.name = value'.

    Notes:
        This function iterates over the attributes of the `parameters` object, ignoring those that 
        start with double underscores, and prints each attribute name along with its value.
        This docstring was generated with the help of AI and proofread by the author.
    """
    
    print("Loaded Parameters:\n")
    for key in dir(parameters):
        if not key.startswith("__"):
            value = getattr(parameters, key)
            print(f"parameters.{key} = {value}")
            
# ----------------------------------------------------------------------------------------------------------------------------

def list_subdirectories(directory, prefix="cluster_"):
    """
    List subdirectories within a specified directory that begin with a given prefix.

    Parameters:
        directory (str): The path to the directory in which to search for subdirectories.
        prefix (str, optional): The prefix that each subdirectory name must start with. Default is "cluster_".

    Returns:
        list: A list of names of subdirectories that start with the given prefix.

    Notes:
        The function checks each item in the specified directory to determine if it is a subdirectory
        and if its name starts with the provided prefix, returning a list of those names.
        This docstring was generated with the help of AI and proofread by the author.
    """
    
    # Filter subdirectories with the given prefix
    return [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name)) and name.startswith(prefix)]

# ----------------------------------------------------------------------------------------------------------------------------

def choose_subdirectory(directory):
    """
    Present a list of subdirectories prefixed with 'cluster_' within a specified directory 
    and prompt the user to choose one.

    Parameters:
        directory (str): The path to the directory where the subdirectories are located.

    Returns:
        str or None: The name of the chosen subdirectory, or None if no suitable subdirectories are found.

    Notes:
        The function lists subdirectories that start with 'cluster_' and allows the user to select one by its number.
        If no such subdirectories exist, it notifies the user and returns None. User input is validated to ensure 
        a valid numerical choice is made.
        This docstring was generated with the help of AI and proofread by the author.
    """
    
    subdirs = list_subdirectories(directory)
    
    if not subdirs:
        print(f"No subdirectories starting with 'cluster_' were found in {directory}.")
        return None

    # Print the list of subdirectories with indices
    print("Available subdirectories:")
    for i, subdir in enumerate(subdirs):
        print(f"{i + 1}: {subdir}")
    
    # Prompt the user to select a subdirectory
    while True:
        try:
            choice = int(input("Enter the number of the subdirectory you want to choose: "))
            if 1 <= choice <= len(subdirs):
                return subdirs[choice - 1]
            else:
                print("Invalid choice. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# ----------------------------------------------------------------------------------------------------------------------------

# function to calculate degree buffer approximation for a given latitude
def meters_to_degrees(buffer_distance_meters, latitude):
    """
    Converts a buffer distance in meters to an approximate degree buffer at a given latitude.

    This function computes the degree buffer for both latitude and longitude based on a given 
    distance in meters and the latitude. It accounts for the variation in longitude degree length
    due to the Earth's curvature.

    Parameters:
        buffer_distance_meters (float): The buffer distance in meters to be converted.
        latitude (float): The latitude at which the conversion is calculated (in degrees).

    Returns:
        tuple: A tuple containing:
            - degree_buffer_lat (float): The degree buffer approximation for latitude.
            - degree_buffer_lon (float): The degree buffer approximation for longitude.

    Raises:
        ValueError: If `buffer_distance_meters` is not positive or if `latitude` is not within 
        the range [-90, 90].

    Notes:
        This docstring was generated with the help of AI and proofread by the author.
    """
    
    # Approximation constants
    meters_per_degree_lat = 40075/360*1000
    meters_per_degree_lon = meters_per_degree_lat * np.cos(np.radians(latitude))
    degree_buffer_lat = buffer_distance_meters / meters_per_degree_lat
    degree_buffer_lon = buffer_distance_meters / meters_per_degree_lon
    
    return degree_buffer_lat, degree_buffer_lon

# ----------------------------------------------------------------------------------------------------------------------------

def plot_clusters(df, cluster_code):
    """
    Plots clusters on a 2D scatter plot.

    Parameters:
        df: pandas.DataFrame, the dataframe containing the data.
        cluster_col: str, column name for cluster labels.
        cluster_code: str, name or ID of the clustering method/identifier.

    Returns:
        None
        
    Notes:
        This docstring was generated with the help of AI and proofread by the author.
    """
    
    # get unique cluster labels
    unique_clusters = df[cluster_code].unique()
    
    # Extract columns
    geo_x = df['geo_x']
    geo_y = df['geo_y']
    cluster_labels = df[cluster_code]

    # Create scatter plot
    scatter = plt.scatter(
        geo_x, 
        geo_y, 
        c=cluster_labels, 
        cmap='gist_rainbow'
    )

    # Create legend handles for unique clusters
    unique_clusters = np.unique(cluster_labels)
    handles = [
        mpatches.Patch(color=scatter.cmap(scatter.norm(value)), label=f'Cluster {value}')
        for value in unique_clusters
    ]

    # Add legend
    plt.legend(handles=handles, title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add axis labels and title
    plt.xlabel('geo_x')
    plt.ylabel('geo_y')
    plt.title(f'Clusters for cluster_id = {cluster_code}')

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------------

def evaluate_silhouette_score(data, cluster_labels):
    """
    Computes and interprets the silhouette score for a clustering result.

    Parameters:
        data (array-like): The dataset used for clustering (e.g., coordinates).
        cluster_labels (array-like): Cluster labels assigned to each data point.

    Returns:
        score (float): The silhouette score.
        interpretation (str): Interpretation of the clustering quality.
        
    Notes:
        This docstring was generated with the help of AI and proofread by the author.
    """
    try:
        score = silhouette_score(data, cluster_labels)
    except:
        score = 0

    if score > 0.7:
        interpretation = "The clusters are well-defined and separated."
    elif score > 0.5:
        interpretation = "The clusters are fairly well-separated."
    elif score > 0.25:
        interpretation = "The clusters are weakly separated and may overlap."
    elif score == 0:
        interpretation = "Unable to determine the metrics."
    else:
        interpretation = "The clustering structure is poor."

    return score, interpretation

# ----------------------------------------------------------------------------------------------------------------------------

def print_silhouette_metrics(df, cluster_code):
    """
    Prints the silhouette score and its interpretation for a given clustering result.

    Parameters:
        df: pandas.DataFrame, the data used for clustering.
        cluster_labels: array-like or pandas.Series, the cluster labels for the data.
        cluster_code: str, the identifier for the clustering result.
        evaluate_silhouette_score: function, a function that calculates the silhouette score and its interpretation.

    Returns:
        None
        
    Notes:
        This docstring was generated with the help of AI and proofread by the author.
    """
    # Calculate silhouette score and interpretation
    score, interpretation = evaluate_silhouette_score(df, cluster_code)

    # Print the results
    print(f"Silhouette Score for {cluster_code}: {score:.4f}, {interpretation}")
    
# ----------------------------------------------------------------------------------------------------------------------------

def display_cluster_counts(df, cluster_code):
    """
    Generates and displays an ASCII table with cluster counts and totals.

    Parameters:
        df: pandas.DataFrame, the data containing the clustering results.
        cluster_code: str, the column name for cluster labels.

    Returns:
        None
    """
    print("\n")

    # Generate the ASCII table and counts
    table, count, count_of_items = create_ascii_table(df, cluster_code)

    # Display the results
    print(table)
    print("\nTotal count:", count)
    print("Count of items:", count_of_items)

# ----------------------------------------------------------------------------------------------------------------------------

def calculate_cluster_metrics(df_features, df_results, cluster_code):
    """
    Calculate various clustering evaluation metrics for a given dataset.

    This function computes metrics to evaluate the quality of clustering, including:
    - Silhouette Index (SI)
    - Calinski-Harabasz Index (CHI)
    - Davies-Bouldin Index (DBI)
    - Dunn Index (DI)

    The metrics are returned as a dictionary for further analysis.

    Args:
        df_features (pd.DataFrame): A DataFrame containing the feature set used for clustering.
        df_results (pd.DataFrame): A DataFrame containing clustering results, including the labels for clusters.
        cluster_code (str): The column name in `df_results` representing the cluster labels.

    Returns:
        dict: A dictionary containing the following clustering evaluation metrics:
            - "(SI) Silhouette Index [max=1]": Silhouette Index value (None if fewer than 2 clusters exist).
            - "(CHI) Calinski-Harabasz Index [max]": Calinski-Harabasz Index value.
            - "(DBI) Davies-Bouldin Index [min=0]": Davies-Bouldin Index value.
            - "(DI) Dunn Index [max]": Dunn Index value.
            Values may be NaN if the computation fails or is not applicable.

    Raises:
        None: The function catches all exceptions internally, and NaN is returned for any failed metric calculation.

    Notes:
        This docstring was generated with the help of AI and proofread by the author.
    """

    # get features and labels
    labels = df_results[cluster_code].values
    features = df_features
    
    # Calculate Silhouette Index (requires at least 2 clusters)
    try:
        silhouette = silhouette_score(features, labels) if len(np.unique(labels)) > 1 else None
    except:
        silhouette = np.nan

    # Calculate Calinski-Harabasz Index
    try:
        calinski_harabasz = calinski_harabasz_score(features, labels)
    except:
        calinski_harabasz = np.nan

    # Calculate Davies-Bouldin Index
    try:
        davies_bouldin = davies_bouldin_score(features, labels)
    except:
        davies_bouldin = np.nan

    # Custom function to calculate Dunn Index
    def dunn_index(features, labels):
        try:
            unique_labels = np.unique(labels)
            inter_cluster_distances = []
            intra_cluster_distances = []

            for i in unique_labels:
                cluster_i = features[labels == i]
                intra_cluster_distances.append(np.max(cdist(cluster_i, cluster_i)))

                for j in unique_labels:
                    if i != j:
                        cluster_j = features[labels == j]
                        inter_cluster_distances.append(np.min(cdist(cluster_i, cluster_j)))
                        
            # Dunn Index: min(inter-cluster distance) / max(intra-cluster distance)
            if intra_cluster_distances:
                return np.min(inter_cluster_distances) / np.max(intra_cluster_distances)
            else:
                return np.nan
                        
        except:
                return np.nan
    dunn = dunn_index(features, labels)

    # Return metrics in a dictionary
    metrics = {
        "(SI) Silhouette Index [max=1]": silhouette,
        "(CHI) Calinski-Harabasz Index [max]": calinski_harabasz,
        "(DBI) Davies-Bouldin Index [min=0]": davies_bouldin,
        "(DI) Dunn Index [max]": dunn,
    }

    return metrics

# ----------------------------------------------------------------------------------------------------------------------------
