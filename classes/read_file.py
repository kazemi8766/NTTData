import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class read_files:

    def __init__(self,
                 path):
        self.__path = path  # dataset originale

    def load_data(self):
        list_string = list(map(str, list(range(1, 23))))
        user_list = []
        # append datasets into teh list
        for i in range(len(list_string)):
            temp_df = pd.read_csv(self.__path + list_string[i] + ".csv",
                                  names=["timestamp", 'x-axis', 'y-axis', 'z-axis'])
            temp_df['person'] = i
            user_list.append(temp_df)
        return user_list