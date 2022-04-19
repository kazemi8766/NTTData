import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class preProcessing:

    def __init__(self,
                 data):

        self.__data = data  # dataset originale

    def __magnitude(self, user_id):
        x2 = user_id['x-axis'] * user_id['x-axis']
        y2 = user_id['y-axis'] * user_id['y-axis']
        z2 = user_id['z-axis'] * user_id['z-axis']
        m2 = x2 + y2 + z2
        m = m2.apply(lambda x: math.sqrt(x))
        return m

    def __calc_avg_speed(self, user_id):
        sp = user_id.index / user_id.timestamp
        sp[np.isnan(sp)] = 0
        return sp

    def calc_magnitudes_speed(self):
        for i in range(1, 23):
            self.__data[i - 1]['magnitude'] = self.__magnitude(self.__data[i - 1])
            self.__data[i - 1]['avg_step_speed'] = self.__calc_avg_speed(self.__data[i - 1])
        return self.__data

    def scale_data(self, column: str):
        for i in range(1, 23):
            mms = MinMaxScaler()
            scaled_col = mms.fit_transform(self.__data[i - 1][[column]])
        return scaled_col