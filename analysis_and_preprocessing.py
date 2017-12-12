import numpy as np
import pandas as pd
import os

def create_df(data_path):
    # input: the path of the csv file
    # output: data frame
    csv_path = os.path.join(data_path, "housing.csv")
    pd.read_csv(csv_path).head()
    return pd.read_csv(csv_path)

def nan_columns(df):
    # input: data frame
    # output: a list of names of columns that contain nan values in the data frame
    nancolumns = df.isnull().any()

    null_column_header = []
    for i in range(len(nancolumns)):
        if nancolumns[i] == True:
            null_column_header.append(nancolumns.keys()[i])

    return null_column_header

def categorical_columns(df):
    # input: data frame
    # output: a list of column names that contain categorical values in the data frame
    cols = df.columns
    num_cols = df._get_numeric_data().columns
    catcolumns = list(set(cols) - set(num_cols))
    return catcolumns

def replace_missing_features(df, nancolumns):
    # input: data frame, list of column names that contain nan values
    # output: data frame
    df2 = df.copy(deep=True)
    for i in nancolumns:
        median = df2[i].median()
        df2[i] = df2[i].fillna(median)
    return df2

def cat_to_num(new_df1, catcolumns):
    # input: data frame, list of categorical feature column names
    # output: data frame
    new_df2 = new_df1.copy(deep=True)
    for i in catcolumns:
        new_df2 = pd.get_dummies(new_df2, columns=[i])
    return new_df2

def standardization(new_df2, labelcol):
    # input: data frame and name of the label column
    # output: scaled data frame
    from sklearn.preprocessing import StandardScaler
    df2 = new_df2.copy(deep=True)
    col_index = df2.columns.get_loc(labelcol)
    label_col = df2[df2.columns[col_index]]
    label_col_nparray = np.array(label_col, dtype=pd.Series)

    df2 = df2.drop([labelcol], axis=1)  # drop the label column

    scaler = StandardScaler()
    df2 = scaler.fit_transform(df2)

    df3 = np.concatenate((df2[:, :col_index], label_col_nparray[:, None], df2[:, col_index:]), axis=1)
    new_df3 = pd.DataFrame(data=df3, index=range(len(df3)), columns=new_df2.columns)

    return new_df3


def my_train_test_split(new_df3, labelcol, test_ratio):
    # input: data frame, name of the label column and test data percentage
    # output: X_train, X_test, y_train, y_test
    from sklearn.model_selection import train_test_split
    np.random.seed(0)

    col_index = new_df3.columns.get_loc(labelcol)

    # Change order of label column to make split easier
    cols = new_df3.columns.tolist()
    cols = cols[col_index:col_index + 1] + cols[0:col_index] + cols[col_index + 1:]
    df4 = new_df3[cols]

    # Split label column and data
    X = df4.iloc[:, 1:].values
    y = df4.iloc[:, 0].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=0)
    return X_train, X_test, y_train, y_test

def main(dataPath, testRatio, labelColumn):
    # input: the path of the csv file, test data percentage and name of the label column
    # output: X_train, X_test, y_train, y_test as numpy arrays
    df = create_df(dataPath)
    nancols = nan_columns(df)
    catcols = categorical_columns(df)
    df = replace_missing_features(df, nancols)
    df = cat_to_num(df, catcols)
    df = standardization(df, labelColumn)
    X_train, X_test, y_train, y_test = my_train_test_split(df, labelColumn, testRatio)
    return X_train, X_test, y_train, y_test

#X_train, X_test, y_train, y_test = main(os.getcwd(), 0.3, "median_house_value")