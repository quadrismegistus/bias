import multiprocessing as mp
# mp.set_start_method('spawn')
mp.set_start_method('fork')
import warnings
warnings.filterwarnings('ignore')


import os
import gensim
from yapmap import *
import pandas as pd
import json
from scipy.spatial import distance
import plotnine as p9
import random
import statsmodels.api as sm






# Reorganizing 
PATH_MODELS=os.path.join(os.path.expanduser('~'), 'DH', 'data', 'models')
YEARBIN=None
YMIN=None
YMAX=None
PATH_FIELDS=os.path.join(os.path.expanduser('~'), 'DH', 'data', 'fields', 'data.fields.json')
PATH_WORDS=os.path.join(os.path.expanduser('~'), 'DH', 'data', 'fields', 'data.words.byu.tsv')
PATH_WORDS_JOBS=os.path.join(os.path.expanduser('~'), 'DH', 'data', 'fields', 'data.occupations.txt')
PATH_WORDS_ANIMALS=os.path.join(os.path.expanduser('~'), 'DH', 'data', 'fields', 'data.animals.txt')
PATH_HERE = os.path.abspath(os.path.dirname(__file__))
PATH_REPO = os.path.abspath(os.path.join(PATH_HERE,'..'))
PATH_DATA = os.path.abspath(os.path.join(PATH_HERE,'..','data'))
PATH_FIGS = os.path.abspath(os.path.join(PATH_HERE,'..','figures'))
PATH_WM_VEC_SCORES = os.path.join(PATH_DATA,'data.vecscores.woman-man.pkl')
KEY='Woman-Man.KW'


from .bias import *
from .words import *
from .models import *
from .plots import *