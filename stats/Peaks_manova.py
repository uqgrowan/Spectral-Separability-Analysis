# Extract spectral features for MANOVA
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from statsmodels.multivariate.manova import MANOVA
import pingouin

# Define data path
path = r"data\MEL_NZ_TAS_spectra.csv"

# Import data and drop unused columns
df = pd.read_csv(path)
sites = df["site"]
labels = df["Class"]

# Drop and format data
df = df.drop(["Class", "site", "setup"], axis = 1)
df.columns = df.columns.astype("int64")

# Import peaks definitions
peaks = pd.read_csv("peak_group_definitions.csv")
peaks = peaks.iloc[:20, :]  # drop peaks beyond 925 for noise
peaks = [460, 461, 464, 465, 466, 469, 471, 472, 473, 474, 475, 477, 478,
       479, 480, 485, 486, 487, 488, 489, 490, 492, 493, 494, 496, 497,
       498, 500, 505, 507, 523, 527, 536, 551, 556, 560, 563, 565, 568,
       573, 574, 575, 576, 577, 578, 579, 580, 581, 583, 584, 585, 586,
       587, 588, 589, 592, 596, 598, 599, 600, 601, 603, 607, 615, 619,
       620, 622, 623, 625, 626, 627, 628, 629, 630, 632, 633, 634, 635,
       636, 637, 638, 640, 641, 643, 644, 645, 646, 647, 648, 649, 651,
       653, 658, 665, 668, 669, 670, 671, 672, 673, 674, 676, 677, 678,
       685, 686, 688, 705, 707, 713, 715, 728, 729, 731, 736, 737, 744,
       745, 746, 747, 748, 753]

# Standardize or normalize data.
#scaler = StandardScaler().fit(df[75:600])
#scaler = StandardScaler().fit(df)
scaler = RobustScaler().fit(df)
df_std = pd.DataFrame(scaler.transform(df))
df_std.columns = df_std.columns +325

# Make results dataframe labeled by peak center (samples X peaks)
peak_maxima = pd.DataFrame(index= df_std.index, columns = peaks["center"])


# Fill peak_maxima dataframe with max values
for row in peaks.index:
    search_range = df_std.loc[:, range(peaks.loc[row, "shortest"], peaks.loc[row, "longest"]+1)]
    peak_maxima.loc[:, peaks.iloc[row, 0]] = np.max(search_range, axis = 1)

# Make sure values are array and numeric
peak_maxima = pd.DataFrame(peak_maxima)
peak_maxima_array = peak_maxima.values.astype(float)
peak_maxima_labelled = pd.concat([labels, peak_maxima], axis = 1)
peak_maxima_labelled.shape
#peak_maxima_labelled.to_csv("Peak_maxima_for_permanova.csv", index= False)

# Make data ingestible for MANOVA and ANOVA
labels_df = pd.DataFrame(labels, columns= ["Class"])
labels_array = labels.astype('category').cat.codes.astype(float)
input = pd.concat([labels_df, peak_maxima], axis = 1)
for col in input.iloc[:, 1:]:
    input[col] = pd.to_numeric(input[col])

# MANOVA
maov = MANOVA(peak_maxima_array, labels_array) 
print(maov.mv_test())

#Access all digits in MANOVA results
maov_df = pd.DataFrame(maov.mv_test().results["x0"]["stat"])
maov_df.loc["Wilks' lambda", "Pr > F"]
for index in maov_df.index:
    print(maov_df.loc[index, "Pr > F"])

# Pairwise ANOVAs
pairwise = {}
for col in peak_maxima.columns:
    
    # Run ANOVA if MANOVA found differences in all features
    aov = pingouin.anova(data = input, dv = col, between = "Class")
    if (aov["p-unc"] < 0.1).any():

        # Run pairwise test if ANOVA found differences in single feature
        pair = pingouin.pairwise_tests(data = input, dv = col, between = "Class", correction = "auto" , padjust = "bonf", alpha = 0.1 )
        pairwise[col] = pair


# Make pairwise dict into DataFrame
pairwise_df = pd.concat(pairwise, axis = 0)
pairwise_df.head()
pairwise_df.reset_index(level = 0, names = ["peak"], inplace= True)


# Plotting

# Make a size list
p_values = pairwise_df["p-corr"]
sizes_list = []
for p_value in p_values:
    if ((p_value >= 0.01) and (p_value <= 0.1)):
        sizes_list.append(1)
    elif ((p_value >= 0.001) and (p_value <= 0.01)):
        sizes_list.append(2)
    elif ((p_value >= 0.0001) and (p_value <= 0.001)):
        sizes_list.append(3)
    elif ((p_value >= 0.000001) and (p_value <= 0.0001)):
        sizes_list.append(4)
    elif ((p_value >= 0) and (p_value <= 0.000001)):
        sizes_list.append(5)
    else:
        sizes_list.append(0.5)

#Add plotting size to dataframe
pairwise_df["size"] = sizes_list
pairwise_df["peak_num"] = pairwise_df["peak"].astype('category').cat.codes.astype(float)

#Plot a histogram of P-values         
sns.histplot(data = pairwise_df, x = "p-corr"),
plt.show()

#Plot a scatterplot of p-values with sized points
sns.scatterplot(data = pairwise_df, x = "peak_num", y = "p-corr", size = "size", alpha = 0.1 )
plt.ylim(bottom = 0.0000000001, top = 0.1)
plt.yscale("log")
plt.show()

def create_violin_plot(data_path):
    # Read CSV file
    df = data_path
    #df = pd.read_csv(data_path)
    
    # Convert p-values to -log10 scale for better visualization
    df['-log10(p)'] = -np.log10(df['p-corr'])
    df['log(inverse-p)'] = np.log10(1/df['p-corr'])
    df['inverse-p'] = 1/df['p-corr']
    
    # Create custom color palette for peaks
    colors = ['#87CEEB', '#0066CC', '#90EE90', '#32CD32', '#FF6B6B', '#CD5C5C', '#FFA500']
    
    # Set figure size and style
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Create violin plot
    sns_plot = sns.violinplot(data=df, x='peak_num', y='-log10(p)',
                             color='white',  # Set base color to white
                             cut=0,  # Don't extend beyond data range
                             scale='width')  # Scale violins to same width
    
    # Customize the plot
    plt.title('Distribution of -log10(p-values) by Peak Number', pad=20, fontsize=12)
    plt.xlabel('Peak Number (m/z)', fontsize=10)
    plt.ylabel('-log10(p-value)', fontsize=10)
    plt.ylim(bottom = 1, top = 10)
    plt.axhline(y = (-np.log(0.05)), linestyle = "-", color = "red")
    plt.axhline(y = (-np.log(0.01)), linestyle = "-", color = 'black')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Set color for each violin
    for i, violin in enumerate(sns_plot.collections):
        violin.set_facecolor(colors[i % len(colors)])
        violin.set_alpha(0.7)  # Set transparency
    
    # Add grid lines
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    return plt.gcf()

# Example usage:
fig = create_violin_plot(pairwise_df)
plt.show()

#Examine specific results
pairwise_df["B"].unique()
check = pairwise_df[pairwise_df["A"] == "rock"]
check = check[check["peak"] == "peak_446"]
check

#Export the pairwise t-test results
pairwise_df.to_csv("pairwise_sigs_400-900.csv")