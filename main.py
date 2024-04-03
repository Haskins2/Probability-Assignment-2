import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, weibull_min, expon
from scipy.optimize import curve_fit


location1 = pd.DataFrame()
location2 = pd.DataFrame()
location3 = pd.DataFrame()

#Location 1 - main gate
#Location 2 - Nassau
#location 3 - Pearse

location_to_graph = location3
plot_title = "Pearse Street Sound Levels"
plot_x_axis = "Average Sound Level Recorded"
plot_y_axis = "Frequency"
zoom_x_axis = 3


# Assuming 'data.csv' is your CSV file
df = pd.read_csv('GROUP2_DATA.csv')

# Drop any columns that are entirely blank
df = df.dropna(axis=1, how='all')

# Remove leading and trailing white spaces from column names
df.columns = df.columns.str.strip()

# Remove leading and trailing white spaces from values in all columns
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# This will remove the time of day column
columns_to_remove = list(range(1, len(df.columns), 4))  # Get the list of column indices to remove
df = df.drop(df.columns[columns_to_remove], axis=1)     # Remove every fourth column starting from the second

# Initialize an empty list to store the generated elements
result_list = []

# Outer loop for locations 1_1 to 1_12
for location_index in range(1, 13):
        for day_index in range(1, 4):
            # Append elements to the result list following the specified pattern
            result_list.append(f"Location{location_index}_{day_index}")
            result_list.append(f"Day{location_index}_{day_index}")
            result_list.append(f"Data{location_index}_{day_index}")

# Print the generated list
# print(result_list)
df.columns = result_list


# now we have 12 locations with 3 datasets (location, day, sound level) on 4 different days
# lets find the mean of the data per location per day, ie: we find the avg of datax_1, datax_2, and datax_3


day_names = ["Tuesday", "Wednesday", "Saturday", "Sunday"]

for i in range(1, 13):
    data_columns = [col for col in df.columns if col.startswith(f'Data{i}_')]
    # Find mean of days Locations Data
    if(i <= 4):
        location1 ["Sound levels Location1"] = df[data_columns].mean(axis=1)
    elif(4 < i <= 8):
        location2 ["Sound levels Location2"] = df[data_columns].mean(axis=1)
    elif(i >= 9):
        location3 ["Sound levels Location3"] = df[data_columns].mean(axis=1)


#
# print(location1)
# print(location2)
# print(location3)

# Plot the histogram of the data
# location1.hist(bins=30, density=True, alpha=0.6, color='b', label='Histogram')
# Plot the histogram of the data
data = location_to_graph.values.flatten()
x_min = np.min(data) - zoom_x_axis
x_max = np.max(data) + zoom_x_axis

plt.hist(data, bins=30, density=True, alpha=0.6, color='g', label='Data')
plt.xlabel(plot_x_axis)
plt.ylabel(plot_y_axis)
plt.title(plot_title)

# Fit a log-normal distribution
shape_lognorm, loc_lognorm, scale_lognorm = lognorm.fit(data, floc=0)
x = np.linspace(x_min, x_max, 1000)
pdf_lognorm = lognorm.pdf(x, shape_lognorm, loc=loc_lognorm, scale=scale_lognorm)
plt.plot(x, pdf_lognorm, 'b-', linewidth=2, label='Fitted Log-Normal')

# Fit a Weibull distribution
params_weibull = weibull_min.fit(data)
pdf_weibull = weibull_min.pdf(x, *params_weibull)
plt.plot(x, pdf_weibull, 'r-', linewidth=2, label='Fitted Weibull')

# Fit an exponential distribution
params_expon = expon.fit(data)
pdf_expon = expon.pdf(x, *params_expon)
plt.plot(x, pdf_expon, 'y-', linewidth=2, label='Fitted Exponential')

# Set x-axis limits
plt.xlim(x_min, x_max)

# Display legend
plt.legend()

# Show plot
plt.show()

# # Save the  data to a CSV files (debugging)
# location1.to_csv('location1.csv', index=False)
# df.to_csv("test.csv", index=False)