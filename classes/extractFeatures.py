import math
from scipy.stats import skew, kurtosis
from statsmodels.tsa import stattools
import csv
import warnings
warnings.filterwarnings('ignore')

class extractFeatures:

  def __init__(self,
               data,
               path):
      self.__data = data   # preProcessing result
      self.__path = path   # save new fetures

  def window(self,axis,dx=100):
      start = 0;
      size = axis.count();

      while (start < size):
          end = start + dx
          yield start,end
          start = start+int (dx/2)

  def window_summary(self,axis, start, end):
      acf = stattools.acf(axis[start:end])
      acv = stattools.acovf(axis[start:end])
      sqd_error = (axis[start:end] - axis[start:end].mean()) ** 2
      return [
          axis[start:end].mean(),
          axis[start:end].std(),
          axis[start:end].var(),
          axis[start:end].min(),
          axis[start:end].max(),
          acf.mean(), # mean auto correlation
          acf.std(), # standard deviation auto correlation
          acv.mean(), # mean auto covariance
          acv.std(), # standard deviation auto covariance
          skew(axis[start:end]),
          kurtosis(axis[start:end]),
          math.sqrt(sqd_error.mean())
      ]

  def adding_features(self,user_id):
      for (start, end) in self.window(user_id['timestamp']):
          features = []
          for axis in ['x-axis', 'y-axis', 'z-axis', 'magnitude','avg_step_speed']:
              features += self.window_summary(user_id[axis], start, end)
          yield features

  def save_features(self):
    with open(self.__path+"Features.csv", 'w') as out:
      rows = csv.writer(out)
      for i in range(0, len(self.__data)):
          for f in self.adding_features(self.__data[i]):
              rows.writerow([i]+f)