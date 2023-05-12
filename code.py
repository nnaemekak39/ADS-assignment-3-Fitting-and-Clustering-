# -*- coding: utf-8 -*-
"""
Created on Fri May 12 05:48:16 2023

@author: Kingsley Ezeja
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from sklearn.metrics import silhouette_score
import scipy.optimize as opt
import errors as err


def fileread(a):
    """
    Reads and imports files from comma seperated values, to a python DataFrame

    Arguments:
    a: string, The name of the csv file which is to be read

    Returns:
    data: A pandas dataframe with all values from the excel file
    transposed_data: The transposed pandas dataframe
    """
    data = pd.read_csv(a, skiprows=4)
    data = data.drop(['Country Code', 'Indicator Code'], axis=1)
    transposed_data = data.set_index(
        data['Country Name']).T.reset_index().rename(columns={'index': 'Year'})
    transposed_data = transposed_data.set_index('Year').dropna(axis=1)
    transposed_data = transposed_data.drop(['Country Name'])
    return data, transposed_data


for_inv = fileread("API_BX.KLT.DINV.WD.GD.ZS_DS2_en_csv_v2_5454977.csv")
for_inv = for_inv[0]
# print(for_inv.head())
gdp_grw = fileread("API_NY.GDP.MKTP.KD.ZG_DS2_en_csv_v2_5358346.csv")
gdp_grw = gdp_grw[0]

# ------------------------------------------------------------------------
# Extracting the Desired Column
foreign_invest = for_inv.iloc[:, [0, 61]]
gdp_growth_ann = gdp_grw.iloc[:, [0, 61]]
foreign_invest = foreign_invest.round(3)

# Combine the dataframes using concat
first_df = pd.concat([gdp_growth_ann, foreign_invest["2019"]], axis=1)

# List of countries/regions not needed
regions = ['Arab World',
           'Caribbean small states',
           'Central Europe and the Baltics',
           'Early-demographic dividend',
           'East Asia & Pacific (excluding high income)',
           'Euro area',
           'Europe & Central Asia (excluding high income)',
           'European Union',
           'Fragile and conflict affected situations',
           'Heavily indebted poor countries (HIPC)',
           'High income',
           'Latin America & Caribbean (excluding high income)',
           'Latin America & the Caribbean (IDA & IBRD countries)',
           'Least developed countries: UN classification',
           'Low & middle income',
           'Low income',
           'Lower middle income',
           'Middle East & North Africa (excluding high income)',
           'Middle income',
           'North America',
           'OECD members',
           'Other small states',
           'Pacific island small states',
           'Small states',
           'South Asia (IDA & IBRD)',
           'Sub-Saharan Africa (excluding high income)',
           'Sub-Saharan Africa (IDA & IBRD countries)',
           'Upper middle income',
           'World']

# Remove the countries from the DataFrame
first_df = first_df[~first_df.index.isin(regions)]
print(first_df)


def data_filter(first_df, a, b):
    """
    This function takes a dataframe as input and performs several
    cleaning tasks on it:
    1. Sets the index to 'Country Name'
    2. Renames the columns to 'Foreign Investment' and 'GDP Annual Growth'
    3. Prints information about the dataframe
    4. Drops rows with missing values

    Parameters:
    first_df (pandas dataframe): the dataframe to be cleaned

    Returns:
    pandas dataframe: the cleaned dataframe
    """
    # set index to 'Country Name'
    first_df = first_df.set_index('Country Name')

    # rename columns
    first_df.columns.values[0] = a
    first_df.columns.values[1] = b

    # print information about the dataframe
    # print(data.info())

    # drop rows with missing values
    first_df = first_df.dropna(axis=0)

    # return cleaned data
    return first_df


data_df = data_filter(first_df, "Foreign Investment", "GDP Annual Growth")
print(data_df)


# Define function to plot a scatter plot
def plot_scatter(data, first_column, second_column, title, xlabel, ylabel):
    """
    This function takes a dataframe and plots a scatter plot with the
    specified columns for x and y values, as well as a title, x-axis label,
    and y-axis label.

    Parameters:
    data (pandas dataframe): the dataframe to be plotted
    first_column (int or string):  index of x-axis column.
    second_column (int or string):  index of y-axis column.
    title (string): the title of the plot
    xlabel (string): the label for the x-axis
    ylabel (string): the label for the y-axis
    """
    # extract x and y values from the dataframe
    x = data.iloc[:, first_column]
    y = data.iloc[:, second_column]

    # create scatter plot
    plt.scatter(x, y)

    # add title, legend, x-axis label, and y-axis label
    plt.title(title, fontweight='bold')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # save plot figure
    plt.savefig('Actual Scatter Plot(kizo).png')

    # show plot
    plt.tight_layout()
    plt.rcParams["figure.dpi"] = 300
    plt.show()


# Activate scatter plot function
plot_scatter(data_df, 0, 1,
             "Scatter Plot of Foreign Investment vs GDP Annual Growth",
             "Foreign Investment", "GDP Annual Growth")

print(data_df['GDP Annual Growth'].describe())

# Filter the dataframe to extract the desired countries
df_b1 = data_df[data_df.iloc[:, 1] <= 4.564]  # countries below world average
print(df_b1['GDP Annual Growth'].mean())
df_b2 = data_df[(data_df.iloc[:, 1] > 4.564) & (data_df.iloc[:, 1] < 50)]
print(df_b2)
df_b3 = data_df[data_df.iloc[:, 1] >= 50]
print(df_b3)

"""
Plot a scatter plot of countries with a GDP Annual Growth
below the world average, i.e < 4.186
"""

# Activate scatter plot function
plot_scatter(df_b1, 0, 1,
             "Countries with GDP Annual Growth Below 4.186(World Average)",
             "Foreign Investment", "GDP Annual Growth")

# Normalise the data
scaler = preprocessing.MinMaxScaler()
df_b1_norm = scaler.fit_transform(df_b1)
# print(df_b1_norm)


# Define a function to determine the number of effective clusters for KMeans
def opt_knum(data, max_k):
    """
    A function to determine the optimal number of clusters for k-means
    clustering on a given dataset. The function plots the relationship between
    the number of clusters and the inertia, and displays the plot.

    Parameters:
    - data (array-like): the dataset to be used for clustering
    - max_k (int): the maximum number of clusters to test for

    Returns: None
    """

    means = []
    inertias = []

    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k, init='k-means++',
                        max_iter=1000)
        kmeans.fit(data)

        means.append(k)
        inertias.append(kmeans.inertia_)

    # Generate the elbow plot
    fig = plt.subplots(figsize=(8, 5))
    plt.plot(means, inertias, "r-")
    plt.xlabel("Number of Clusters", fontsize=12)
    plt.ylabel("Inertia", fontsize=12)
    plt.title("Elbow Method Showing Optimal Number of K", fontsize=16,
              fontweight='bold')
    plt.grid(True)
    plt.rcParams["figure.dpi"] = 300
    plt.savefig("K-Means Elbow Plot(kizo).png")
    plt.show()

    return fig


# Activate the optimum k function to get the number of effective clusters
opt_knum(df_b1_norm, 10)

# Create a function to run the KMeans model on the dataset


def kmeans_func(data, n_clusters):
    """
    Applies K-Means clustering on the data and returns the cluster labels.

    Parameters:
        data (numpy array or pandas dataframe) : The data to be clustered
        n_clusters (int) : The number of clusters to form.

    Returns:
        numpy array : The cluster labels for each data point
        numpy array : The cluster centers
        float : The inertia of the model
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100,
                    random_state=0)
    kmeans.fit(data)
    clusters = kmeans.predict(data)
    centroids = kmeans.cluster_centers_
    inertia = kmeans.inertia_

    return clusters, centroids, inertia


# Activate KMeans clustering function
clusters, centroids, inertia = kmeans_func(df_b1, 4)
print("Clusters: ", clusters)
print("Centroids: ", centroids)
print("Inertia: ", inertia)

# Calculate the silhouette score for the number of clusters
sil_0 = silhouette_score(df_b1, clusters)
print("Silhouette Score:" + str(sil_0))

df_b1['Clusters'] = clusters
print(df_b1)


# Define a function that plots the clusters
def plot_clusters(df, cluster, centroids):
    """
    Plot the clusters formed from a clustering algorithm.

    Parameters:
    df: DataFrame containing the data that was clustered.
    cluster: Array or Series containing the cluster labels for each
    point in the data.
    centroids: Array or DataFrame containing the coordinates of the
    cluster centroids.
    """

    df.iloc[:, 1]
    df.iloc[:, 0]
    cent1 = centroids[:, 1]
    cent2 = centroids[:, 0]
    plt.scatter(df.iloc[cluster == 0, 1], df.iloc[cluster == 0, 0], s=50,
                c='cyan', label='Cluster 0')
    plt.scatter(df.iloc[cluster == 1, 1], df.iloc[cluster == 1, 0], s=50,
                c='yellow', label='Cluster 1')
    plt.scatter(df.iloc[cluster == 2, 1], df.iloc[cluster == 2, 0], s=50,
                c='blue', label='Cluster 2')
    plt.scatter(df.iloc[cluster == 3, 1], df.iloc[cluster == 3, 0], s=50,
                c='green', label='Cluster 3')
    # Centroid plot
    plt.scatter(cent1, cent2, c='red', s=100, label='Centroid')
    plt.title('Cluster of the Countries',
              fontweight='bold')
    plt.ylabel('Foreign Investment', fontsize=12)
    plt.xlabel('GDP Annual Growth', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.rcParams["figure.dpi"] = 300
    plt.savefig("Clusters(kizo).png")
    plt.show()


# Activate the function to plot the clusters
plot_clusters(df_b1, clusters, centroids)

# Carry out cluster analysis by plotting a bar chart showing the country
# distribution in each cluster
sns.countplot(x='Clusters', data=df_b1)
plt.savefig('Cluster distribution(kizo).png')
plt.title('Cluster Distribution of Countries with GDP Annual Growth Below 4.56'
          ' (World Average)', fontweight='bold')
plt.show()


# Define a polynomial function to plot a curve fit curve
def fit_polynomial(x, a, b, c, d):
    """
    Fit a polynomial of degree 3 to a given set of data points.

    Parameters:
    x: x-coordinates of the data points.
    a,b,c,d: function coefficients.

    Returns: Optimal values for the coefficients of the polynomial.
    """
    # popt, pcov = curve_fit(fit_polynomial, x, y)
    return a*x**3 + b*x**2 + c*x + d


# Initialise variables
x_axis = df_b1.values[:, 1]
y_axis = df_b1.values[:, 0]

# Instantiate the curvefit function
popt, pcov = opt.curve_fit(fit_polynomial, x_axis, y_axis)
a, b, c, d = popt
print('y = %.5f * x^3 + %.5f * x^2 + %.5f * x + %.5f' % (a, b, c, d))
print(pcov)

# Generate the curvefit line variables
d_arr = df_b1.values[:, 1]  # convert data to an array
x_line = np.arange(min(d_arr), max(d_arr)+1, 1)  # a random range of points
y_line = fit_polynomial(x_line, a, b, c, d)  # generate y-axis variables
plt.scatter(x_axis, y_axis, label="Countries")  # scatterplot
# plot the curvefit line
plt.plot(x_line, y_line, '-', color='red', linewidth=2, label="Curvefit")
plt.title('Cluster of Countries showing Prediction Line (Curvefit)',
          fontweight='bold')
plt.ylabel('Foreign Investment', fontsize=12)
plt.xlabel('GDP Annual Growth (%)', fontsize=12)
plt.legend(loc='lower right')
plt.annotate('y = 0.00671x + 58.308', (3000, 55), fontweight='bold')
plt.savefig("Scatterplot Prediction Line(kizo).png")
plt.rcParams["figure.dpi"] = 300
plt.show()

# Generate the confidence interval and error range
sigma = np.sqrt(np.diag(pcov))
low, up = err.err_ranges(d_arr, fit_polynomial, popt, sigma)
print(low, up)

ci = 1.95 * np.std(y_axis)/np.sqrt(len(x_axis))
lower = y_line - ci
upper = y_line + ci
print(f'Confidence Interval, ci = {ci}')

# Plot showing best fitting function and the error range
plt.scatter(x_axis, y_axis, label="Countries")
plt.plot(x_line, y_line, '-', color='black', linewidth=2,
         label="Curvefit")
plt.fill_between(x_line, lower, upper, alpha=0.3, color='red',
                 label="Error range")
plt.title('Cluster Showing Prediction Line (Curvefit) & Error Range',
          fontweight='bold')
plt.ylabel('Foreign Investment', fontsize=12)
plt.xlabel('GDP Annual Growth (%)', fontsize=12)
plt.annotate(f'C.I = {ci.round(3)}', (7800, 60), fontweight='bold')
plt.legend(loc='upper right')
plt.rcParams["figure.dpi"] = 300
plt.savefig("Scatterplot Error Line(kizo).png")
plt.show()
