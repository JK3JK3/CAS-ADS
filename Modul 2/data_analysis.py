"""scale and prepare df for plotting"""

# Load the needed python libraries by executing this python code (press ctrl enter)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import seaborn as sns
import networkx as nx
from scipy import stats
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import mannwhitneyu


############ read file into pandas df ##########################

df = pd.read_csv("CombineROSINA_RowAB_new.log", sep="\t")
#print(df.head()) # Print the first five rows
print(df.describe())


# Open the tab-separated file and read the first row (header)
with open("CombineROSINA_RowAB_new.log", newline='', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')  # use tab as delimiter
    header = next(reader)  # get the first line
    header_slice = header[13:38] #use header_slice as set
    #print(header_slice)

# create empty dataframe and fill it with scaled columns
df_empty = pd.DataFrame()
for i in header_slice:
    scaled_column = df[i] * df["DistCGSC [km]"]**2 * (100000)**2 # *r^2*km->cm-rescale-factor
    df_empty[i] = scaled_column   # assign column directly/fill df_empty

#### get rid of 0 -> maintain original 0.00001 and log-scale to emphasize small values
df_scaled = np.log10(df_empty.replace(0, 1))



#### CHECK: first value of water (H2O) (dist=845km=84500000cm, density(H2O)=627'000ccm)
#density scaled (log10 and *r^2) = 21.65
#print(df_scaled.head())

# timestamp join
df_scaled["time_stamp"] = pd.to_datetime(df["AcquisitionTime"])
#df_scaled["time_stamp"].dt.strftime("%d-%m-%Y, %r")
#print(df_scaled.tail())



"""
##### PLOT 0: density versus time for main species
fig, ax1 = plt.subplots(figsize=(12,6))

# --- left y-axis (densities) ---
ax1.scatter(df_scaled["time_stamp"][::1000], df_scaled['nH2O [cm^-3]'][::1000],
            c='b', label='H2O', alpha=0.5)
ax1.scatter(df_scaled["time_stamp"][::1000], df_scaled['nCO2 [cm^-3]'][::1000],
            c='g', label='CO2', alpha=0.5)
ax1.scatter(df_scaled["time_stamp"][::1000], df_scaled['nCO [cm^-3]'][::1000],
            c='r', label='CO', alpha=0.5)

ax1.set_xlabel("Time [UTC]")
ax1.set_ylabel("Scaled density [cm^-3]")

# --- right y-axis (heliocentric distance r) ---
ax2 = ax1.twinx()
ax2.plot(pd.to_datetime(df["AcquisitionTime"]), df["DistCGSUN [au]"],
         color="black", linewidth=1.5, label="r [au]")
ax2.set_ylabel("Heliocentric distance r [au]")

# --- x-ticks formatting ---
plt.xticks(rotation=90, ha="right")

# --- combine legends from both axes ---
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc="lower left", framealpha=1, facecolor="white")
plt.savefig("density-time-series.png", dpi=300, bbox_inches="tight")
plt.show()
"""




### set time-stamp as index of df_scaled
df_scaled.set_index(["time_stamp"], drop=True, inplace=True)



"""
###### PLOT 1: heat map

sns.set_theme()
plt.figure(figsize=(15,8))
#sns.heatmap(df_scaled.select_dtypes(include="number").head(1000).transpose(),cmap="coolwarm")
sns.heatmap(df_scaled.select_dtypes(include="number").transpose(),cmap="viridis")

plt.xticks(ticks=range(0, len(df_scaled), 25000), 
           labels=df_scaled.index[::25000].strftime('%d.%m.%Y'), 
           rotation=45, ha="right")  # Rotieren der x-Labels
plt.tight_layout()  # ensures labels fit inside canvas
plt.savefig("heatmap_all.png", dpi=300)  # save as PNG, high resolution
plt.show()


#### DESCRIPTION: Pearson correlation matrix from df_scaled
df_scaled_species = df_scaled.reset_index(drop=True)
# df_scaled_species = df_scaled.drop(["time_stamp"], axis=1) # use if time-stamp not index
# print(df_scaled.head())
# print(df_scaled_species.head())

# Pearson correlation
corr_matrix = df_scaled_species.corr(method='pearson')
# print(corr_matrix)


###### PLOT 2: pearson correlation heat map
# Create a custom colormap: blue -> white -> red
#cmap = LinearSegmentedColormap.from_list("blue_white_red", ["blue", "white", "red"])

# Mask the diagonal to white
mask = np.eye(len(corr_matrix), dtype=bool)

# Clean axis labels from df_scaled: remove unit in brackets and leading 'n'
clean_labels = [col.split(' ')[0] for col in df_scaled.columns]  # remove unit
clean_labels = [label[1:] if label.startswith('n') else label for label in clean_labels]  # remove 'n'

plt.figure(figsize=(12,10))
ax = sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", mask=mask,
            cbar_kws={"label": "Pearson correlation"}, square=True)

# Get the tick positions automatically
xticks = ax.get_xticks()
yticks = ax.get_yticks()

# Make sure the number of labels matches number of ticks
ax.set_xticklabels(clean_labels[:len(xticks)], rotation=90)
ax.set_yticklabels(clean_labels[:len(yticks)], rotation=0)

plt.savefig("corr_matrix.png", dpi=300)
plt.title("Pearson correlation matrix of 26 species")
plt.tight_layout()
plt.show()




###### PLOT 3: pearson correlation matrix as network
G = nx.Graph()

# Prepare cleaned labels
clean_labels = [col.split(' ')[0] for col in df_scaled.columns]  # remove unit
clean_labels = [label[1:] if label.startswith('n') else label for label in clean_labels]  # remove 'n'

# Create a mapping from original columns to cleaned labels
label_map = dict(zip(corr_matrix.columns, clean_labels))

# Add nodes using cleaned labels
for col in corr_matrix.columns:
    G.add_node(label_map[col])

# Add edges (upper triangle, threshold)
threshold = 0.5
for i in range(len(corr_matrix)):
    for j in range(i+1, len(corr_matrix)):
        corr = corr_matrix.iloc[i, j]
        if abs(corr) > threshold:
            G.add_edge(label_map[corr_matrix.index[i]], 
                       label_map[corr_matrix.columns[j]], 
                       weight=corr)

# Layout and draw
pos = nx.spring_layout(G, k=0.8)
edges = G.edges(data=True)
weights = [d['weight'] for (u,v,d) in edges]

plt.figure(figsize=(12,10))
nx.draw_networkx_nodes(G, pos, node_size=200, node_color='lightblue')
nx.draw_networkx_labels(G, pos, font_size=10)
nx.draw_networkx_edges(G, pos, width=[abs(w)*3 for w in weights])
plt.axis('off')
plt.savefig("network-from-corr.png", dpi=300)
plt.show()
"""



# equinoxes:
# inbound equinox (Tubiana, Rinaldi, Güttler, Snodgrass, Shi, etc., 2019): 27 April 2015, at 1.76 AU
# according to Luspay-Kuti et al. A&A 630, A30 (2019)
# inbound equinox (10 May 2015)
# outbound equinox (21 March 2016)

# make sub-dfs and slice plus-minus 14d before and after equinoxes

# user input date
date_inbound = pd.to_datetime("2015-05-10")
date_outbound = pd.to_datetime("2016-03-21")

# interval in days
delta = pd.Timedelta(days=30)

# filter the DataFrame
df_inbound = df_scaled.loc[(df_scaled.index >= date_inbound - delta) & (df_scaled.index <= date_inbound + delta)]
df_outbound = df_scaled.loc[(df_scaled.index >= date_outbound - delta) & (df_scaled.index <= date_outbound + delta)]
# print(df_inbound.describe()) # count  16962
# print(df_outbound.describe()) # count  12531

# df_inbound_means = df_inbound.mean(axis=0)
# df_outbound_means = df_outbound.mean(axis=0)
# print(df_inbound_means.head())



##### PLOT 4: Q-Q plots for major species, colored by time

# --- Define major species ---
major_species = ["nH2O [cm^-3]", "nCO [cm^-3]", "nCO2 [cm^-3]"]

# --- Define intervals ---
intervals = {"inbound": df_inbound, "outbound": df_outbound}

# --- Compute numeric time for each interval ---
time_intervals = {}
for name, df_interval in intervals.items():
    timestamps = pd.to_datetime(df_interval[df_interval.columns[0]])  # first column = timestamp
    time_numeric = (timestamps - timestamps.min()).dt.total_seconds()  # seconds from start
    time_intervals[name] = time_numeric

# --- Loop over intervals and species ---
for interval_name, df_interval in intervals.items():
    times = time_intervals[interval_name]  # numeric timestamps aligned with subset

    for species in major_species:
        vals = df_interval[species]
        
        # Sort data to align with theoretical quantiles
        sorted_vals = np.sort(vals)
        sorted_time = times[vals.index[np.argsort(vals)]]  # align times with sorted data
        
        # Compute theoretical quantiles and fitted line
        osm = stats.probplot(sorted_vals, dist="norm")
        theoretical_q = osm[0][0]
        observed_q = osm[0][1]
        slope, intercept = osm[1][0], osm[1][1]  # fitted line: y = slope * x + intercept
        
        # Plot Q-Q
        plt.figure(figsize=(6,6))
        sc = plt.scatter(theoretical_q, observed_q,
                         c=sorted_time, cmap='viridis',
                         alpha=0.2, edgecolor='k')
        
        # Correct reference line using fitted slope and intercept
        plt.plot(theoretical_q, intercept + slope * theoretical_q, 'r--', lw=2)
        
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Observed Quantiles")
        plt.title(f"Q-Q Plot {interval_name.capitalize()} {species} (time color)")
        plt.colorbar(sc, label="Time (seconds)")
        plt.tight_layout()
        
        # Save figure
        filename = f"qqplot_{interval_name}_{species.replace(' ','_').replace('[','').replace(']','')}_timecolor.png"
        plt.savefig(filename, dpi=300)
        plt.show()


# HYPOTHESIS TESTING
# unpaired 2-sample t-test (independent) is not suitable because samples not normally distributed
# Mann-Whitney U test (non-parametric) is suitable for non-normal distributions
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html


print("Mann-Whitney U test (Inbound vs Outbound):")
for species in major_species:
    inbound_vals = df_inbound[species].values
    outbound_vals = df_outbound[species].values
    
    # Perform two-sided Mann–Whitney U test
    stat, p_value = mannwhitneyu(inbound_vals, outbound_vals, alternative='two-sided')
    
    print(f"{species}: U-statistic={stat:.2f}, p-value={p_value:.3e}")


"""
Interpretation:
Null hypothesis in Mann-Whitney U: The two samples (here, inbound and outbound densities) come from the same distribution. Very strong evidence that the distributions are different in location. Not just random variation — inbound and outbound densities are significantly different. Practical meaning for your comet data: The densities of that species inbound and outbound are statistically distinguishable, which could reflect seasonal, heliocentric, or cometary activity changes. The comet has changed during the perihelion passage.
"""
