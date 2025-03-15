# Enhancing Predictability of Volcanic Eruptions at the Pacific Ring of Fire: Using Localised Cross-Correlated Time Series with Re-Clustered Volcanic Regions

## Repository
Repository for the Final Project Report (Thesis) submitted as the examined coursework of the module DSM500 Final project of MSc Data Science and  Artificial Intelligence (DSAI) at the University of London.

## Abstract
Earthquakes can trigger volcanic eruptions by altering stress in the surrounding crust. This study enhances volcanic eruption forecasting in the Pacific Ring of Fire by analysing seismic-volcanic interactions. Using localized geoscientific data, it prioritizes seismic key features and refines data engineering to integrate historical time series. In a data-driven way, it examines feature combinations, time lags, and unsupervised clustering for reclassifying volcanic regions. Spanning 1970–2019, it employs statistical tests, cross-correlation analysis, and re-clustering to improve volcanic region classification with a robust, validated approach.

The study identified region-specific differences by changepoint detection and autocorrelation shows highly regional variations in seismic and volcanic activity. Clustering analysis using eight methods revealed spatial variations, with WARD ranking highest in internal metrics, offering a cohesive classification, and BIRCH aligning closest to original GVP clusters. Compact methods formed distinct regions, while density-based approaches captured elongated structures. GMM and IDEC showed the most deviation, and MD-DBSCAN exhibited high feature similarity but structural inconsistencies. WARD excelled in spatial clustering but differed from original clusters, emphasizing the impact of clustering methods on seismic-volcanic pattern interpretation. Cross-correlation analysis revealed distinct time lags across clusters, with strong correlations at 3, 4, 20, and 21 years in the complete study area, while regional clustering exposed unique seismic-volcanic relationships. Additionally, three distinct pattern types emerged though feature combinations varied.

Aiming for a global eruption forecasting model, future work should explore advanced forecasting algorithms, additional clustering techniques, and causal inference methods, fostering overall disaster preparedness.

## Directory structure
<pre>
root
|  
+ data
  |
  + scope
    |
    + studyarea_1000
    | + cluster_all
    |
    + gvp_1000
    | + cluster_middle_america_carribbean
    | + cluster_north_amaerica
    | + cluster_northwest_pacific
    | + cluster_south_america
    | + cluster_soutwest_pacific
    | + cluster_sunda_banda
    | + cluster_tonga_kermadec
    | + cluster_western_pacific
    |
    + recluster_1000
      + cluster_Cluster_0
      + cluster_Cluster_1
      + cluster_Cluster_2
      + cluster_Cluster_3
      + cluster_Cluster_4
      + cluster_Cluster_5
      + cluster_Cluster_6
  
</pre>
