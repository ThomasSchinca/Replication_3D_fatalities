# Replication data – The geometry of conflict : 3D Spatio-temporal patterns in fatalities prediction

This package reproduces the analysis of the paper The geometry of conflict : 3D Spatio-temporal patterns in fatalities prediction

## Structure

```
Scripts/            python scripts
Data/               input and created data
Figure/             figures
requirements.txt   
```

## Installation

Python 3.8.5 with the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

## Overview
The repository includes the necessary scripts and data to replicate the results presented in the paper. The primary script, `3_main_results.py` reproduces the main figures and set of result from pre-loaded matching and validation. Run `1_active_zone_detection_matching.py` and `2_validation.py` to replicate the entire pipeline. 

## Directory Structure
- 1_active_zone_detection_matching.py: Active zone and matching similar historical cases. (runtime: 1h per period)
- 2_validation.py: parameter grid search
- 3_main_results.py: main file to reproduce  
- 4_toy_model_example, 5_appendix_H, 6_appendix_I reproduce some Appendix parts
- download_data.py dowload the benchmark forecast 
- shape.py is needed to reproduce the original 2D model



