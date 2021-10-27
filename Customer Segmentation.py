#!/usr/bin/env python
# coding: utf-8

# # Creating Customer Segments
# 
# ### Unsupervised Learning

# ## Getting Started
# 
# In this project, you will analyze a dataset containing data on various customers' annual spending amounts (reported in *monetary units*) of diverse product categories for internal structure. One goal of this project is to best describe the variation in the different types of customers that a wholesale distributor interacts with. Doing so would equip the distributor with insight into how to best structure their delivery service to meet the needs of each customer.
# 
# The dataset for this project can be found on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers). For the purposes of this project, the features `'Channel'` and `'Region'` will be excluded in the analysis — with focus instead on the six product categories recorded for customers.
# 
# **Description of Categories**
# - FRESH: annual spending (m.u.) on fresh products (Continuous)
# - MILK: annual spending (m.u.) on milk products (Continuous)
# - GROCERY: annual spending (m.u.) on grocery products (Continuous)
# - FROZEN: annual spending (m.u.)on frozen products (Continuous) 
# - DETERGENTS_PAPER: annual spending (m.u.) on detergents and paper products (Continuous) 
# - DELICATESSEN: annual spending (m.u.) on and delicatessen products (Continuous)
#     - "A store selling cold cuts, cheeses, and a variety of salads, as well as a selection of unusual or foreign prepared foods."
# 

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import renders as rs
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()
pd.set_option('display.max_columns', None)
pd.set_option("max_rows", None)


# #### **Task 1: Import Dataset and create a copy of that dataset**

# In[2]:


data = pd.read_csv('customers.csv')
df = data.copy()


# In[3]:


data.head()


# **Task 2: Drop Region and Channel column**

# In[4]:


df.drop(['Region', 'Channel'], axis = 1, inplace = True)


# **Task 3: Display first five rows** 

# In[5]:


df.head()


# #### **Task 4: Display last five rows** 

# In[6]:


df.tail()


# #### **Task 5: Check the number of rows and columns**

# In[7]:


df.shape


# #### **Task 6: Check data types of all columns**

# In[8]:


df.dtypes


# **Task 7: Check for missing values and fill missing values if required.**

# In[9]:


df.isnull().sum()


# ## Data Exploration

# #### **Task 8: Checking summary statistics and store the resultant DataFrame in a new variable named *stats***

# In[10]:


stats = df.describe()
stats


# **Question: Explain the summary statistics for the above data set**

# **Answer:** All 6 columns are numeric and positive skewed which means, they have outliers.Furthermore, there are no missing/null values in the dataset.

# ### Implementation: Selecting Samples
# To get a better understanding of the customers and how their data will transform through the analysis, it would be best to select a few sample data points and explore them in more detail. In the code block below, add **three** indices of your choice to the `indices` list which will represent the customers to track. It is suggested to try different sets of samples until you obtain customers that vary significantly from one another.

# **Logic in selecting the 3 samples: Quartiles**
# - As you can previously (in the object "stats"), we've the data showing the first and third quartiles.
# - We can filter samples that are starkly different based on the quartiles.
#     - This way we've two establishments that belong in the first and third quartiles respectively in, for example, the Frozen category.

# **Task 9: Select any random sample and assign the list to given variable**

# In[11]:


#Filtering data that occures in first and third quartile (based on Fresh column)
low, high = df.Fresh.quantile([0.25,0.75]) # finding boundry values of 1st and 3rd quartiles of Fresh column
fst_3rd_df = df.query('{low}<Fresh<{high}'.format(low=low,high=high)) #selecting dataframe (based on Fresh column) from 1st and 3rd quartiles


# In[12]:


# Write code here
indices = [0,270,434]
indices


# These samples will be separated into another dataframe for finding out the details the type of customer each of the selected respresents

# **Task 10: Make a dataframe of selected indices**

# In[13]:


# Write code here
samples = fst_3rd_df.loc[indices]
samples


# In[14]:


# Write code here
samples.describe()


# The selected sample values should be ranked amongst the whole of the data values to check their ranks and get a better understanding of spending of each sample/customer in each category

# In[15]:


percentiles = df.rank(pct=True)
percentiles = 100*percentiles.round(decimals=3)
percentiles = percentiles.iloc[indices]
percentiles


# **Task 11: Draw a heatmap to show the above results achieved in** `percentile` **to have a better understanding.**

# In[16]:


#Write code here
sns.heatmap(percentiles)


# #### Question: What type of customers can you identify by looking into the heatmap?

# #### Answer: Most people are buying Milk, Grocery, Frozen, Detergents_Paper and Delicatessen from the second quartile.

# **Task 12: Find the corelation among all the variables of whole dataframe and describe the findings you infer from the heatmapt.**

# In[17]:


# Write the code here
sns.heatmap(df.corr())


# **Answer:** 
# - When Fresh is sold, it is highly probable that customer will buy Frozen or Delicatessen. <br>
# - When Milk is sold, it is highly probable that Customer will buy Grocery or Detergents_Paper.<br>
# - When Grocery is sold, it is highly probable that customer will buy Milk or DetergentsPpaper. <br>
# - When Frozen is sold, it is moderately possible that customer will buy Fresh or Delicatessen.<br>
# - When Detergents_Paper is sold, it is moderately possible that customer will buy Fresh or Delicatessen.<br>
# - When Delicatessen is sold, it is moderately possible that customer will buy Milk or Frozen.<br>

# ### Pair Plot

# Pairplot is a plot which is used to give and over view of the data in a graphical grid form. The result it shows gives us a picture of variables themselves in a graphical way as well as a relationship of one variable with all the others. For more details you can [click here](https://seaborn.pydata.org/generated/seaborn.pairplot.html)

# **Task 13: Make a pairplot using seaborn.**

# In[18]:


# write code here
sns.pairplot(df);


# **Question: What findings do you get from the above plot? Describe in terms of skewned/normal form by looking at each variable and also look for any outliers that can be visually identified in the plot.**

# **Answer:** 
# - All of the columns are are rightly skewed. <br>
# <br>
# - Delicatessen-Fresh have almost zero correlation with ~2 outliers.
# - Delicatessen-Milk have very weak positive correlation with ~1 outliers.
# - Delicatessen-Grocery have very weak positive correlation with ~1 outliers.
# - Delicatessen-Frozen have very weak positive correlation with ~2 outliers.
# - Delicatessen-Detergents_Paper have very weak positive correlation with ~3 outliers.
# <br> <br>
# - Detergents_Paper-Fresh have strong positive correlation with ~3 outliers.
# - Detergents_Paper-Milk have strong positive correlation with ~5 outliers.
# - Detergents_Paper-Grocery have very strong positive correlation with ~2 outliers.
# - Detergents_Paper-Frozen have ~zero correlation with ~5 outliers.
# - Detergents_Paper-Delicatessen have ~zero correlation with ~3 outliers.
# <br> <br>
# - Same trend of the outliers and skeweness can be observed in the dataset.

# ## Data Preprocessing
# In this section, you will preprocess the data to create a better representation of customers by normalizing it by **removing skewness** and **detecting (and optionally removing) outliers**. 

# ### Implementation: Feature Scaling
# If data is not normally distributed, especially if the mean and median vary significantly (indicating a large skew), it is most [often appropriate](http://econbrowser.com/archives/2014/02/use-of-logarithms-in-economics) to apply a non-linear scaling — particularly for financial data.

# **Task 14: Apply log on data for transforming it from skewed to normalized form. Use function** `np.log()` **and save the result in** `log_data`

# In[19]:


#Write code here
log_data = np.log(df)


# ### Implementation: Outlier Detection
# Detecting outliers in the data is extremely important in the data preprocessing step of any analysis. The presence of outliers can often skew results which take into consideration these data points. There are many "rules of thumb" for what constitutes an outlier in a dataset. Here, we will use [Tukey's Method for identfying outliers](http://datapigtechnologies.com/blog/index.php/highlighting-outliers-in-your-data-with-the-tukey-method/): An *outlier step* is calculated as 1.5 times the interquartile range (IQR). A data point with a feature that is beyond an outlier step outside of the IQR for that feature is considered abnormal.
# 
# In the code block below, you will need to implement the following:
#  - Assign the value of the 25th percentile for the given feature to Q1. Use `np.percentile` for this.
#  - Assign the value of the 75th percentile for the given feature to Q3. Again, use `np.percentile`.
#  - Assign the calculation of an IQR for the given feature.
#  - Query the data to filter out Outliers using IQR
#  - remove data points from the dataset by adding indices to the outliers list
# 
# **NOTE:** If you choose to remove any outliers, ensure that the sample data does not contain any of these points! 
# 
# Once you have performed this implementation, the dataset will be stored in the variable `good_data`.

# In[20]:


outliers=[]
# For each feature find the data points with extreme high or low values
for feature in log_data.keys():
    #Calculating Q1 (25th percentile of the data) for the given feature
    Q1 = log_data[feature].quantile(0.25)
    #Calculating Q3 (75th percentile of the data) for the given feature
    Q3 = log_data[feature].quantile(0.75)
    #Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = (1.5*(Q3-Q1))
    # Display the outliers
    print("Data points considered outliers for the feature '{}':".format(feature))
    out=log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
    display(out)
    outliers=outliers+list(out.index.values)

# Select the indices for data points you wish to remove
outliers = list(set([x for x in outliers if outliers.count(x) > 1]))    
print ("Outliers: {}".format(outliers))

# Remove the outliers, if any were specified
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)


# **Question**<br>
# Are there any data points considered outliers for more than one feature based on the definition above? Should these data points be removed from the dataset? If any data points were added to the `outliers` list to be removed, explain why?

# **Answer:** columns of index 65 and 75 have more than one outliers. To remove such datapoints depends on the condition. For instance the 75th index row has two outliers: Grocery and Detergents_Papers with low values. It is possible that the person is eating outside and need less detergets_papers but he needs to do little grocery, therefore, his grocery expenses are also lesser. In this scenario, this person can be a point of interest. However, we can also remove this datapoint to generalize the model for simpler predictions.

# **Task 15: Make a pairplot to check changes in data after pre-processing and using the** `good_data`

# In[21]:


# Write the code here
sns.pairplot(good_data);


# ## Feature Transformation
# In this section you will use principal component analysis (PCA) to draw conclusions about the underlying structure of the wholesale customer data. Since using PCA on a dataset calculates the dimensions which best maximize variance, we will find which compound combinations of features best describe customers.

# ### Implementation: PCA
# 
# Now that the data has been scaled to a more normal distribution and has had any necessary outliers removed, we can now apply PCA to the `good_data` to discover which dimensions about the data best maximize the variance of features involved. In addition to finding these dimensions, PCA will also report the *explained variance ratio* of each dimension — how much variance within the data is explained by that dimension alone. Note that a component (dimension) from PCA can be considered a new "feature" of the space, however it is a composition of the original features present in the data.
# 
# In the code block below, you will need to implement the following:
#  - Import `sklearn.decomposition.PCA` and 
#  - Apply a PCA transformation of the good data.

# **Task 16: Import PCA Library**

# In[22]:


# Write your code here
from sklearn.decomposition import PCA


# **Task 17: Apply PCA by fitting the good data with the same number of dimensions as features.**

# In[23]:


# Write your code here
pca_ = PCA(n_components=6)


# In[24]:


# Write your code here
pca_.fit(good_data)


# In[25]:


# Generate PCA results plot
pca_results = rs.pca_results(good_data, pca_)
plt.plot(pca_.explained_variance_ratio_)
plt.xlabel('number of components + 1')
plt.ylabel('cumulative explained variance');


# **Task 18: Find cumulative explained variance**

# In[26]:


# Write the code here
cumsum_pca_results= pca_.explained_variance_ratio_
cumsum_pca_results.sum()


# **Question**
# How much variance in the data is explained ***in total*** by the first and second principal component? What about the first four principal components? How many components should be selected for reducing the dimensions? Give your answer along with the reason.

# **Answer:** 100% variance is explained in total, because all features are being used. If we observe the above graph, the first component is explaining ~42% of the total data and second compoent is explaining ~27%. Similarly, 3rd component is explaining 12% and 4th component is explaining 10% of the data. If we add the variance of these 4 components, it will be: 42+27+12+10= 91. It means if we use 4 components. we will be focusing on 91% variance of the total data, which is good to perform analysyi.

# ### Implementation: Dimensionality Reduction
# In the code block below, you will need to implement the following:
#  - Assign the results of fitting PCA in two dimensions with `good_data` to `pca`.
#  - Apply a PCA transformation of `good_data` using `pca.transform`, and assign the results to `reduced_data`.
#  - Apply a PCA transformation of the sample log-data `log_samples` using `pca.transform`, and assign the results to `pca_samples`.

# **Task 19: Apply PCA by fitting the good data with the selected number of components**

# In[27]:


# write your code here
pca = PCA(n_components=2)
pca.fit(good_data)


# **Task 20: Transform the good data using the PCA fit above**

# In[28]:


# write your code here
reduced_data = pca.transform(good_data)


# **Task 21: Create a DataFrame for the reduced data**

# In[29]:


# write your code here
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])


# ## Implementation: Creating Clusters

# In this section, you will choose to use either a K-Means clustering algorithm  and hierarchical clustering to identify the various customer segments hidden in the data. You will then recover specific data points from the clusters to understand their significance by transforming them back into their original dimension and scale. 

# ## Choosing K

# **Before Implementing KMeans and hierarchical clustering, choose the optimal K using the following method**

# - Silhouette Score

# ### Silhouette Score for K-Means

# In[30]:


# Import necessary libraries
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


# **Task 22: Check Silhouette Score for finding Optimal K**

# In[31]:


# write your code here
range_n_clusters = [2, 3, 4, 5, 6, 7,8,9,10,11,12,13,14,15,16]
silhouette_avg = []
for num_clusters in range_n_clusters:
    # initialise kmeans
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(reduced_data)
    cluster_labels = kmeans.labels_
    silhouette_avg.append(silhouette_score(reduced_data, cluster_labels))
silhouette_avg


# **Task 23: Plot a graph representing the Silhouette Score.**

# In[32]:


#add plot
plt.plot(range_n_clusters,silhouette_avg,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Silhouette score') 
plt.title('Silhouette analysis For Optimal k in KMeans')
plt.show()


# ### Silhouette Score for Hierarchical Clustering

# In[33]:


# Import necessary libraries
from sklearn.cluster import AgglomerativeClustering


# **Task 24: Write the code below to calculate silhouette score for each hierarchical clustering**

# In[34]:


# write your code here
range_n_clusters_h = [2, 3, 4, 5, 6, 7,8,9,10,11,12,13,14,15,16]
silhouette_avg_h = []
for num_clusters_h in range_n_clusters_h:
    # initialise AgglomerativeClustering
    hier_clus = AgglomerativeClustering(n_clusters=num_clusters_h)
    hier_clus.fit(reduced_data)
    cluster_labels_h = hier_clus.labels_
    silhouette_avg_h.append(silhouette_score(reduced_data, cluster_labels_h))
silhouette_avg_h


# **Task 25: Write the code below to make a plot for silhouette score**

# In[35]:


#add plot
plt.plot(range_n_clusters_h,silhouette_avg_h,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Silhouette score') 
plt.title('Silhouette analysis For Optimal k in hierarchical clustering')
plt.show()


# **Question: Report the silhouette score for several cluster numbers you tried. Of these, which number of clusters has the best silhouette score?**

# **Answer:**<br>
# KMeans Silhouette Score: 0.42628101546910835, 0.3964019299623205, 0.3316606459254843, 0.34999779752629756 <br>
# hierarchical Silhouette Score: 0.37506864833503123, 0.36013962293428176, 0.27162388334400245, 0.2824756004966861 <br><br>
#  (Sillhout of KMeans > Silhouette of hierarchical) which means that clusters in kmeans are well apart w.r.t clusters in hierarchical clustering. Therefore, KMeans is oerforming better in this scenario.

# ## Implementation of K-Means

# **Task 26: Implement KMeans using your choosen K**

# In[36]:


# write your code here
kmean_f = KMeans(2)
kmean_f.fit(reduced_data)


# In[37]:


# write your code here
preds = kmean_f.fit_predict(reduced_data)


# ## Implementation Hierarchical Clustering

# **Task 27: Implement Hierarchical(agglomerative) clustering using your choosen K**

# In[38]:


# write your code h.ere
a_cluster = AgglomerativeClustering(n_clusters= 2)


# In[39]:


# write your code here
preds_agg = a_cluster.fit_predict(reduced_data)


# ## Best Clustering Algorithm?

# **You will be using** `adjusted rand index` **to select the best clustering algorithm by comparing each of the calculated labels with actual labels found in** `data['Channel]` . Before calculating the score, we need to make sure that the shape of true labels is consistent with the resultant labels.

# In[40]:


true_labels = data['Channel'].drop(data['Channel'].index[outliers]).reset_index(drop = True)


# **Task 28: Find the adjusted rand index for K-Means and Agglomerative Clustering**

# In[41]:


# Import necessary libraries
from sklearn.metrics.cluster import adjusted_rand_score


# In[42]:


ward_ar_score = adjusted_rand_score(true_labels, preds)
ward_ar_score


# In[43]:


aggl_score = adjusted_rand_score(preds_agg, preds)
aggl_score


# **Question: Which has the best score and should be selected?**

# **Answer:** <br>
# adjusted_rand_score of Agglomerative clustering is higher than the adjusted_rand_score of KMeans. It means that Agglomerative clustering is combining similar datapoints in similar clusters. However, the difference is not too large (2% only).

# ## Visualizing the clusters

# **Task 29: Get the centers for KMeans**

# In[44]:


# Write code here
centers = kmean_f.cluster_centers_
centers


# In[45]:


rs.cluster_results(reduced_data, preds, centers)


# # Profiling

# In[46]:


df_pred = df.drop(df.index[outliers]).reset_index(drop = True)
df_pred['pred'] = preds


# **Task 30: Get the average prices for each category from the original data frame for each cluster and then make a profile for each**

# In[47]:


# write the code here
clustered_avg = AgglomerativeClustering(n_clusters = 2, linkage = 'average')
clustered_avg.fit_predict(reduced_data)


# **Task 31: Make a radar chart to show a better profile for each cluster.**

# In[48]:


# Write the code to import the library files for plotly and set your credentials


# In[49]:


# write the code here


# **Task 32: Make the data set for radar chart**

# In[50]:


# Write your code here
radar_data = None


# **Task 33: Set the layout for your radar chart and plot it**

# In[51]:


# Write your code here
radar_layout = None


# In[52]:


# add plot
fig = None


# **Question: What can you infer from the above plot? Explain in detail**

# **Answer:** 
# 

# ## Conclusion

# In this final section, you will investigate ways that you can make use of the clustered data. First, you will consider how the different groups of customers, the ***customer segments***, may be affected differently by a specific delivery scheme. Next, you will consider how giving a label to each customer (which *segment* that customer belongs to) can provide for additional features about the customer data. Finally, you will compare the ***customer segments*** to a hidden variable present in the data, to see whether the clustering identified certain relationships.

# ### Visualizing Underlying Distributions
# 
# At the beginning of this project, it was discussed that the `'Channel'` and `'Region'` features would be excluded from the dataset so that the customer product categories were emphasized in the analysis. By reintroducing the `'Channel'` feature to the dataset, an interesting structure emerges when considering the same PCA dimensionality reduction applied earlier to the original dataset.
# 
# Run the code block below to see how each data point is labeled either `'HoReCa'` (Hotel/Restaurant/Cafe) or `'Retail'` the reduced space. In addition, you will find the sample points are circled in the plot, which will identify their labeling.

# In[53]:


# Display the clustering results based on 'Channel' data
rs.channel_results(reduced_data, outliers)


# **Question:**
# *How well does the clustering algorithm and number of clusters you've chosen compare to this underlying distribution of Hotel/Restaurant/Cafe customers to Retailer customers? Are there customer segments that would be classified as purely 'Retailers' or 'Hotels/Restaurants/Cafes' by this distribution? Would you consider these classifications as consistent with your previous definition of the customer segments?*

# **Answer:** <br>
# There are some intersecting datapoints from -2 to 0 which are from both HoReCa and Retailer, we can call these datapoints as pure 'Retailers' or 'Hotels/Restaurants/Cafes'. This is much simpler representation of data to analyse the customers and has good classification as well.
