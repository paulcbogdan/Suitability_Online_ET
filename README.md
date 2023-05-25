This repository contains the code and data needed to reproduce the online eye-tracking results for the  paper 
"Investigating the Suitability of Online Eye-Tracking..." (Bogdan et al., 2023, _Behavior Research Methods_). 

Along with just the code, the online data itself has been uploaded here. 
The gaze measurements (e.g., proportion of time gazing in the foregrounds)
and behavioral data (EmoRatings) are stored `in_data` as .csv files. Additionally, the raw online eye-tracking data
is in `in_data\eye_timeseries`, where each participant is uploaded as a separate .pkl file, and each .pkl file contains
the series of gaze coordinates recorded for each trial. The .pkl files are named based on participant numbers.

The data based on in-person (infrared) eye-tracking sadly could not be uploaded. Our local IRB informed us that the 
consent forms we used to collect the in-person data years ago did not include language allowing us to share the data.

To perform all the t-tests and multilevel regressions to generate the online results, for both studies 1 and 2,
run `python.final_analysis_all.py`. The data needed are included in this repo already such that the script will work out
of the box.

To perform the meta-analysis aggregating all of the distributions, run `python.aggregate_distributions.py`. 

The final script provided, `python.calculcate_ET.py`, is used for calculating the proportion of time spent in interest
areas based on the online eye-tracking data. This code does not need to be run, as the repo already includes 
spreadsheets including the proportion of time in the FG and BG interest areas for each trial (said spreadsheets are 
used above by `python.final_analysis_all.py`). Nonetheless, you can look at the code to see how these were calculated.

Code for the simulations has not been included as it depends on the in-person data, which we could not upload.