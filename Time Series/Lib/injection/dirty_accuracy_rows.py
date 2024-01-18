import numpy as np
import pandas as pd
import random


def out_of_range(minimum, maximum):
    foo = ["up", "down"]
    f = random.choice(foo)
    dist = maximum - minimum

    if f == "up":
        number = random.uniform(maximum, maximum + dist * 5)
    else:
        number = random.uniform(minimum - dist * 5, minimum)

    return number


def check_datatypes(df, name_class):
    non_numeric_columns = df.select_dtypes(exclude=[int, float]).columns.tolist()
    non_numeric_columns += name_class
    # Get the indexes of the non-numeric columns
    non_numeric_indexes = [df.columns.get_loc(col) for col in non_numeric_columns]

    return non_numeric_indexes


def injection(df_pandas, seed, name, name_class):
    np.random.seed(seed)

    df_list = []
    outliers_indices = []

    # percentuale di errori
    perc = [0.1, 0.2, 0.3, 0.4, 0.5]
    for p in perc:
        df_dirt = df_pandas.copy()
        excluded_columns = check_datatypes(df_dirt, name_class)
        rows, mask = create_matrix_mask(df_dirt.shape[0], df_dirt.shape[1], p, excluded_columns)

        for j in range(len(df_dirt.columns)):
            cols = df_dirt.columns
            if j not in excluded_columns:
                minimum = float(df_dirt[cols[j]].min())
                maximum = float(df_dirt[cols[j]].max())
                for i in range(mask.shape[0]):
                    if mask[i,1] == j:
                        mask[i,2] = out_of_range(minimum, maximum)


        for row, col, value in mask:
            df_dirt.iat[int(row), int(col)] = value

        rows = np.nonzero(rows)[0].tolist()

        df_list.append(df_dirt)
        outliers_indices.append(rows)
        print("saved {}-accuracy {}%".format(name, round((1 - p) * 100)))

    return df_list, outliers_indices


def create_matrix_mask(rows, cols, p, not_acceptable):
    mask_rows = np.random.choice([True, False], rows, p=[p, 1 - p])
    matrix_mask = np.full((rows, cols), False, dtype=bool)
    for row_idx, is_true in enumerate(mask_rows):
        if is_true:
            bool_array = np.full(cols, False, dtype=bool)
            num_values = np.random.choice([1, 2, 3], size=1, p=[0.5, 0.3, 0.2])
            true_indices = np.random.choice([x for x in range(cols) if x not in not_acceptable], num_values,
                                            replace=False)
            bool_array[true_indices] = True
            matrix_mask[row_idx] = bool_array

    true_indices = np.argwhere(matrix_mask)
    outliers = np.append(true_indices, np.empty((true_indices.shape[0],1)),axis=1)

    return mask_rows, outliers
