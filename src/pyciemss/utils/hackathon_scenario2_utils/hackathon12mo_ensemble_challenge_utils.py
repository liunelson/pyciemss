import pandas as pd
import numpy as np

url = 'https://github.com/reichlab/covid19-forecast-hub/blob/master/data-truth/truth-Incident%20Cases.csv'
raw_cases = pd.read_csv(url)
url = 'https://media.githubusercontent.com/media/reichlab/covid19-forecast-hub/master/data-truth/truth-Incident%20Hospitalizations.csv'
raw_hosp = pd.read_csv(url)
url = 'https://media.githubusercontent.com/media/reichlab/covid19-forecast-hub/master/data-truth/truth-Cumulative%20Deaths.csv'
raw_deaths = pd.read_csv(url)

def get_state_data(state_fips):

    state_data = ABC

    return state_data