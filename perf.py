### Pandas vs Cudf Performance Test

#!hostname;date

"""!pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu12==24.6.* cuml-cu12==24.6.* \
    cugraph-cu12==24.6.*"""

import os
import time
import timeit
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cudf

np.random.seed(0)

### Count (300 million)

num_rows = 300_000_000
pdf = pd.DataFrame(
    {
        "numbers": np.random.randint(-1000, 1000, num_rows, dtype="int64"),
        "business": np.random.choice(
            ["McD", "Buckees", "Walmart", "Costco"], size=num_rows
        ),
    }
)

len(pdf)

pdf.head()

gdf = cudf.from_pandas(pdf)

gdf.head()

def timeit_pandas_cudf(pd_obj, gd_obj, func, **kwargs):
    """
    A utility function to measure execution time of an
    API(`func`) in pandas & cudf.

    Parameters
    ----------
    pd_obj : Pandas object
    gd_obj : cuDF object
    func : callable
    """
    pandas_time = timeit.timeit(lambda: func(pd_obj), **kwargs)
    cudf_time = timeit.timeit(lambda: func(gd_obj), **kwargs)
    return pandas_time, cudf_time

pandas_value_counts, cudf_value_counts = timeit_pandas_cudf(
    pdf, gdf, lambda df: df.value_counts(), number=30
)

print("Time taken by Pandas: ", pandas_value_counts)

print("Time taken by cudf: ", cudf_value_counts)

### Concat Example (100 million)

pdf = pdf.head(100_000_000)
gdf = gdf.head(100_000_000)

pandas_concat = timeit.timeit(lambda: pd.concat([pdf, pdf, pdf]), number=30)

print("Time taken by pandas for concatination: ", pandas_concat)

cudf_concat = timeit.timeit(lambda: cudf.concat([gdf, gdf, gdf]), number=30)

print("Time taken by cudf for concatination: ", cudf_concat)

### Group by (100 million)

pandas_groupby, cudf_groupby = timeit_pandas_cudf(
    pdf,
    gdf,
    lambda df: df.groupby("business").agg(["min", "max", "mean"]),
    number=30,
)

print("Time taken by pandas for group by: ", pandas_groupby)

print("Time taken by cudf for group by: ", cudf_groupby)

### Merge (1 Million)

num_rows = 1_000_000
pdf = pd.DataFrame(
    {
        "numbers": np.random.randint(-1000, 1000, num_rows, dtype="int64"),
        "business": np.random.choice(
            ["McD", "Buckees", "Walmart", "Costco"], size=num_rows
        ),
    }
)
gdf = cudf.from_pandas(pdf)

pandas_merge, cudf_merge = timeit_pandas_cudf(
    pdf, gdf, lambda df: df.merge(df), number=30
)

print("Time taken by pandas for merge: ", pandas_merge)

print("Time taken by cudf for merge: ", cudf_merge)

### Performance Matrix

performance_df = pd.DataFrame(
    {
        "cudf speedup vs. pandas": [
            pandas_value_counts / cudf_value_counts,
            pandas_concat / cudf_concat,
            pandas_groupby / cudf_groupby,
            pandas_merge / cudf_merge,
        ],
    },
    index=["value_counts", "concat", "groupby", "merge"],
)

performance_df

import matplotlib.pyplot as plt

# Assuming performance_df is your DataFrame
# performance_df = ...

fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the size as needed

ax = performance_df.plot.bar(
    ax=ax,
    color="#7400ff",
    ylim=(1, 400),
    rot=0,
    xlabel="Operation",
    ylabel="Speedup factor",
)
ax.bar_label(ax.containers[0], fmt="%.0f")

plt.show()

performance_df_new = pd.DataFrame(
    {
        "pandas": [pandas_value_counts, pandas_concat, pandas_groupby, pandas_merge],
        "cudf": [cudf_value_counts, cudf_concat, cudf_groupby, cudf_merge],
        "cudf speedup vs. pandas": [
            pandas_value_counts / cudf_value_counts,
            pandas_concat / cudf_concat,
            pandas_groupby / cudf_groupby,
            pandas_merge / cudf_merge,
        ],
    },
    index=["value_counts", "concat", "groupby", "merge"],
)

performance_df_new

import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame for the performance comparison
performance_df = pd.DataFrame(
    {
        "Operation": ["value_counts", "concat", "groupby", "merge"],
        "pandas": [pandas_value_counts, pandas_concat, pandas_groupby, pandas_merge],
        "cudf": [cudf_value_counts, cudf_concat, cudf_groupby, cudf_merge],
        "cudf speedup vs. pandas": [
            pandas_value_counts / cudf_value_counts,
            pandas_concat / cudf_concat,
            pandas_groupby / cudf_groupby,
            pandas_merge / cudf_merge,
        ],
    }
)

# Plot the speedup values using a horizontal bar chart
fig, ax = plt.subplots(figsize=(8, 6))

ax.barh(performance_df["Operation"], performance_df["cudf speedup vs. pandas"], color="skyblue")
ax.set_xlabel("Speedup Factor (cuDF vs. pandas)")
ax.set_title("cuDF Speedup vs. pandas for Different Operations")

plt.show()
