# retrospective_bvalue_2021
Code and Data for "A Retrospective Analysis of b-value Changes Preceding Strong Earthquakes" by Nicolas D. DeSalvio and Maxwell L. Rudolph (In Review)

- ```primary_code_and_data/``` contains all of the data and code to calculate the b-value time series, Traffic Light System (TLS) scores, and uncertainties. There are specific scripts to generate Figures 3, 4 and 5.

- ```post_processing/``` contains scripts that process the output from ```primary_code_and_data/``` to analyze the success of the prediction scheme. There are specific scripts to generate Figures 2 and S1 here.


Within `primary_code_and_data/`, the `b_output/` folder is currently populated with graphs used in the paper, while the images may be redundant, the file structure they are contained in is utilized by the code.
```
├── catalog_creation_code
│   ├── download_usgs_catalog_with_cmt.ipynb  # Downloads the entire ComCat catalog and CMT focal mechanisms
│   ├── update_catalog.ipynb                  # Appends the catalog produced by `download_usgs_catalog_with_cmt.ipynb' with new events, this requires the contents of `usgs_full_catalog_cmt.zip' and `usgs_full_catalog_times_cmt.zip'
│   ├── usgs_full_catalog_cmt.zip             # Earthquake catalog from ComCat w/ CMT focal mechanisms
│   ├── usgs_full_catalog_times_cmt.zip       # Earthquake Times
├── primary_code_and_data
│   ├── b_output 
│   │   ├── Alert                             # Figures displaying the percent change in b-value and TLS color for selected events
│   │   ├── FMD                               # Figures displaying the frequency-magnitude diagram for selected events
│   │   ├── TS                                # Figures displaying the b-value time series for selected events
│   │   ├── alert_unc                         # Figures displaying the TLS score and uncertainty range for selected events
│   ├── bvalue_analysis.py                    # Script to run the analysis for a single parameter combination, which can either be specified via the command line or within the script using the `use_command_line' variable. Uploaded version is currently in the command line setting (`use_command_line = True'). This script outputs a csv file containing b-value time series, TLS scores, and uncertainty information for the specified parameter choices.
│   ├── fig_3.py                              # Produces the alerts vs time figure (Figure 3)
│   ├── fig_4a.py                             # Produces to produce the Joshua Tree FMD (Figure 4a)
│   ├── fig_4b.py                             # Produces to produce the 2019 Ridgecrest foreshock FMD (Figure 4b)
│   ├── fig_5a.py                             # Produces to produce the Chalfant Valley foreshock FMD (Figure 5a)
│   ├── fig_5b.py                             # Produces to produce the 1995 Ridgecrest foreshock FMD (Figure 5b)
│   ├── renamed_eq_locations.txt              # List of earthquake names and/or county locations of earthquakes, for easier reading of the output files. The county locations can be generated using `bvalue_analysis.py` but specific names like "Loma Prieta" were altered manually. To generate, set `generate_location_name_file' to True near end of code.
│   ├── usgs_full_catalog_cmt.zip             # Earthquake catalog from ComCat w/ CMT focal mechanisms
│   ├── usgs_full_catalog_times_cmt.zip       # Earthquake Times
├── post_processing
│   ├── Figure_2.ipynb                        # Creates the barchart figure (Figure 2)
│   ├── Figure_S1_probabalistic_analysis.ipynb# Jupyter Notebook to reproduce Figure S3
│   ├── Post-Processing-2011_2021.ipynb       # Applies several statistical measures to judge the sucess of the scheme for the subset of M5+ events between 2011 and 2021.
│   ├── Post-Processing-M5+.ipynb             # Applies several statistical measures to judge the sucess of the scheme for the subset of M5+ events.
│   ├── Post-Processing-M6+.ipynb             # Applies several statistical measures to judge the sucess of the scheme for the subset of M6+ events. Code to create Figure S2 is here.
│   ├── Post-Processing-Normal.ipynb          # Applies several statistical measures to judge the sucess of the scheme for the subset of M5+ normal events.
│   ├── Post-Processing-Reverse.ipynb         # Applies several statistical measures to judge the sucess of the scheme for the subset of M5+ reverse events.
│   ├── Post-Processing_Strike_Slip.ipynb     # Applies several statistical measures to judge the sucess of the scheme for the subset of M5+ strike-slip events.
│   ├── rake_file.txt                         # File containing the rake of the nodal plane selected by  `bvalue_analysis.py' for each earthquake, can be generated by `bvalue_analysis.py' in `distance_from_plane_cut' function (set `produce_rake_list' to True)
│   ├── result_csvs.zip                       # Output files of `bvalue_analysis.py' for every parameter combination
│   ├── usgs_full_catalog_M5.csv              # Earthquake catalog for the M5+ subset of events, can be generated in `bvalue_analysis.py' by setting `save_M5_catalogs' to True in the `pairing' function
│   ├── usgs_full_catalog_M5_info.txt         # Earthquake pairing information for the M5+ subset of events, can be generated in `bvalue_analysis.py' by setting `save_M5_catalogs' to True in the `pairing' function
│   ├── usgs_full_catalog_M5_times.txt        # Earthquake Time information for the M5+ subset of events, can be generated in `bvalue_analysis.py' by setting `save_M5_catalogs' to True in the `pairing' function
│   ├── usgs_full_catalog_times_cmt.zip       # Earthquake Times
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
- obspy
- os
- cartopy
- argparse
- geopy
