# retrospective_bvalue_2021
Work in Progress

Code and Data for "A Retrospective Analysis of b-value Changes Preceding Strong Earthquakes" by Nicolas D. DeSalvio and Maxwell L. Rudolph (In Review)

- ```primary_code_and_data/``` contains all of the data and code to calculate the b-value time series, Traffic Light System (TLS) scores, and uncertainties. There are specific scripts to generate Figures 3, 4 and 5.

- ```post_processing/``` contains scripts that process the output from ```primary_code_and_data/``` to analyze the success of the prediction scheme. There are specific scripts to generate Figures 2 and S1 here.


Within `primary_code_and_data/`, the `b_output/` folder is currently populated with graphs used in the paper, while the images may be redundant, the file structure they are contained in is utilized by the code.
```
├── primary_code_and_data
│   ├── b_output
│   │   ├── Alert       # Figures displaying the percent change in b-value and TLS color for selected events
│   │   ├── FMD         # Figures displaying the frequency-magnitude diagram for selected events
│   │   ├── TS          # Figures displaying the b-value time series for selected events
│   │   ├── alert_unc   # Figures displaying the TLS score and uncertainty range for selected events
├── post_processing
```

In order to run the code, files `usgs_full_catalog_cmt.zip`, `usgs_full_catalog_times_cmt.zip`, and `result_csvs.zip` must be unzipped.

Python Packages Required:
- numpy
- matplotlib
- scipy
- pandas
- tqdm
- tabulate
- libcomcat
- datetime
- 
