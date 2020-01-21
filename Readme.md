This code has been tested in ubuntu 18.04, but it should work in other environments.

## Installation
1. Please ensure you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) installed.
2. Navigate to folder containing this package.
2. Create a new environment with all necessary packages using: `conda env create -f environment.yml`
3. Switch to this new environment called "noiseFilter" using: `conda activate noiseFilter`


## Index
+ `Results/` folder contains the aforementioned results in csv format.
+ `JIRA/` folder contains the 32 datasets used in this study. Original source, [replication package](https://zenodo.org/record/2566774#.XicEiFkzY5k)
+ In `library/` folder:
    + `configs.py` defines the models, cross-validation approach and two evaluation metrics used.
    + `utils.py` contains code for reading dataset and evaluationg models.\
+ `Clean&Noisy.ipynb` computes classifier performance for both noisy and clean labels, `Clean_vs_Noisy - Analysis.ipynb` is used to analyze generated result and produce Figure 1 and Table 2.
+ Code for 9 noise filters are scattered around several files, sometimes with repeatative code.
    + `Built-in Filters.ipynb` computes result for "ENN", "SmoteEnc" and "IHT" ("IHTHRES" in table 3) filters.
    + Code for "NCL", "CLNI", "IPF", "Spider2" and "RFF" ("IHF" in Table 3) is contained in jupyter files with similar name.
    + Smote_IPF is subset of the result computed in "IPF"
    + "NoF" is simply taken from `Clean&Noisy.ipynb`
    + `Filtering-Analysis.ipynb` analyzes the produced result to generate Figure 2,3 and Table 3.
+ `Dataset factors.ipynb` tries to find how several dataset characteristics like noise level, imblanace ratio etc. correlate with final filtered result, and produces Figure 4.
+ `FP vs FN Noise.ipynb` computes the results used to generate "Figure 5"


<p>The primary aim of this  repository is to provide implementation details. However, this is not exactly reproducible, random states were not fixed everywhere necessary during experimentation. We believe the cross validation procedure will ensure that the results will not diverge significantly in any replication attempt.</p> 