# Enhancing Predictability of Volcanic Eruptions at the Pacific Ring of Fire: Using Localised Cross-Correlated Time Series with Re-Clustered Volcanic Regions

## Abstract
Earthquakes can trigger volcanic eruptions by altering stress in the surrounding crust. This study enhances volcanic eruption forecasting in the Pacific Ring of Fire by analysing seismic-volcanic interactions. Using localized geoscientific data, it prioritizes seismic key features and refines data engineering to integrate historical time series. In a data-driven way, it examines feature combinations, time lags, and unsupervised clustering for reclassifying volcanic regions. Spanning 1970–2019, it employs statistical tests, cross-correlation analysis, and re-clustering to improve volcanic region classification with a robust, validated approach.

The study identified region-specific differences by changepoint detection and autocorrelation shows highly regional variations in seismic and volcanic activity. Clustering analysis using eight methods revealed spatial variations, with WARD ranking highest in internal metrics, offering a cohesive classification, and BIRCH aligning closest to original GVP clusters. Compact methods formed distinct regions, while density-based approaches captured elongated structures. GMM and IDEC showed the most deviation, and MD-DBSCAN exhibited high feature similarity but structural inconsistencies. WARD excelled in spatial clustering but differed from original clusters, emphasizing the impact of clustering methods on seismic-volcanic pattern interpretation. Cross-correlation analysis revealed distinct time lags across clusters, with strong correlations at 3, 4, 20, and 21 years in the complete study area, while regional clustering exposed unique seismic-volcanic relationships. Additionally, three distinct pattern types emerged though feature combinations varied.

Aiming for a global eruption forecasting model, future work should explore advanced forecasting algorithms, additional clustering techniques, and causal inference methods, fostering overall disaster preparedness.

## Repository
Repository for the Final Project Report (Thesis) submitted as the examined coursework of the module DSM500 Final project of MSc Data Science and  Artificial Intelligence (DSAI) at the University of London. This repository completely lists the Python scripts together with the scraped raw data and all results to ensure a transparent and reproducible research.

## Directory structure
The directory structure gives an overview about the data and the structure.
<pre>
root
|  
+ log
+ shapefiles  
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

## Scripts
The following table gives a short overview about the used Python scripts. These are ordered to ensure the correct workflow order when executed.

| Script Name                                           | Explanation |
|-------------------------------------------------------|-------------|
| 00_github_push_workflow.ipynb                         | Pushing ANYTHING for backup reasons to a GitHib repository |
| 10_scrape_earthquake_data.ipynb                       | Scrape earthquake data from the USGS |
| 11_scrape_volcanic_eruption_data.ipynb                | Scarpe volcanic and eruption data from the Smithsonian Institution |
| 20_engineer_earthquake_data.ipynb                     | Clean and engineer earthquake datasets |
| 21_engineer_volcanic_eruption_data.ipynb              | Clean and engineer volcanic datasets |
| 251_spatial_operations_studyarea.ipynb                | Cluster operations for the complete studyarea with 1000 km buffer |
| 252_spatial_operations_gvp_1000.ipynb                 | Cluster operations for the original GVP clusters with 1000 km buffer |
| 253_spatial_operations_gvp_reclustering.ipynb         | Cluster operations for the re-clustering with 1000 km buffer |
| 30_create_earthquake_timeseries.ipynb                 | Create earthquake time series |
| 31_create_volcanic_eruption_timeseries.ipynb          | Create volcanic time series |
| 40_eda_earthquakes.ipynb                              | Provide an EDA analysis for earthquqakes |
| 41_eda_volcanic_eruptions.ipynb                       | Provide an EDA analysis for volcanic data |
| 50_timeseries_analysis.ipynb                          | Analysis a time series (both earthquake or volcanic data |
| parameters.py                                         | Flexible parameterisation for all scripts |
| shared_procedures.py                                  | Shared procedures for all scripts |
