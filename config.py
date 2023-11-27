from os import sep
from sys import path

sep = '/'
RTF = (sep).join( __file__.split(sep)[:-1] )
RSR = '/opt/anaconda3'
RSR = 'c:/Users/you.zerouali/Documents/Code/classif-factures'

PATHS = {
    'ROOT_RSRC' : RSR,
    'ROOT_CODE' : RTF,
    'ROOT_DATA' : (sep).join([RSR, "data"])
}


# --- TEXT CLEANING CONFIG
LS_VARS = ['STOPWORDS', 'LOCATIONS', 'DATES', 'NAMES', 'STEMS']
LS_FILES = ["stopwords-fr.txt", "stoplocations-fr.txt", "stopdates-fr.txt", "stopnames.txt", "wordstems.txt"]
TEXT_CLEANING = {}
for i_,j_ in zip(LS_VARS, LS_FILES):
    with open( (sep).join([PATHS["ROOT_DATA"], "Auxiliaire", j_]) ) as f:
        TEXT_CLEANING[i_] = [k_.strip() for k_ in f.readlines()[0].replace(' ', '').split(',')]

