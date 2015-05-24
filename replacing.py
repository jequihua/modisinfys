
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import cross_validation
from sklearn import datasets
from sklearn.metrics import r2_score

# import training data as data frame
datain = pd.read_csv("D:/Julian/64_ie_maps/julian_tables_2/training_final.csv")
dataout = pd.read_csv("D:/Julian/64_ie_maps/julian_tables_2/datos_entrenamiento_altprom2.csv")

