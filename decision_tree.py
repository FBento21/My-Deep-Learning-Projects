import numpy as np
import pandas as pd

def entropy(pp):
    i = 0
    for p in pp:
        i -= p * np.log2(p)

    return i


def decision_tree(df, attr_col):
    y = list(df.iloc[:, attr_col])
    attributes = list(df.iloc[:, :attr_col].columns)
    # Compute impurity of each attribute

    impurities = []
    size = df.shape[0]
    for attr in attributes:
        vars = dict(df[attr].value_counts())
        imp = 0
        for obs, n_obs in vars.items():
            print(obs)
            imp += (n_obs/size)
            print(df[(df[attr] == obs) & (df['Play'] == 'No')])
        break

if __name__ == '__main__':

    # Create Dataset
    d = {'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast'],
         'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal'],
         'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong'],
         'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes']}

    df = pd.DataFrame(data=d)
    print(df)

    decision_tree(df, 3)

