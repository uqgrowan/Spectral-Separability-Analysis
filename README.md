# Speca: Spectral separability analysis
Investigate the separability of targets' spectral reflectance profiles. Includes pre-processing, resampling, visualization, and required auxiliary functionalities.

---

## Installation

### Prerequisites

- Python 3.12+
- pvClust (Python implementation available at: <https://github.com/aturanjanin/pvclust>)

### Dependencies

Install all dependencies using
```bash
pip install -r requirements.txt
```

---

## 👌 Logic

The repository contains the workflow to fully preprocess a labelled dataset and analyse spectral separability using Hierarchical Cluster Analysis, Spectral Angle classification and Random Forest classification.

It is broken up into the sections below:

### 🏭 Preprocessing

- Clean the data for missing values and mask bad sensor bands
- Filter, sort, and normalize the cleaned data

### 📈 Visualising

- Plot individual, multiple, or class mean reflectance profiles by search terms
- Show intra-class variability by plotting standard deviation (class means only)

### 🔺Hierarchical Cluster Analysis

- Calculate and plot the dendrogram of best spectral clustering using pvclust [1,2].
- Display the Approximately Unbiased cluster confidence [3].

### 📐Spectral Angle Method

- Implement a modified spectral angle mapper [4] that handles point-source data with sample removal before class-average profile calculation
- Print various accuracy measures of the classification results
- Plot the confusion matrix of the classifcation results.

### 🌳Random Forest Classification

- Extract PCA components
- Evaluate various methods of determining `n_comps`
- Optimize hyperparameters with a grid search
- Aggregate results of many random forests conducted on differing train-test sample divisions
- Plot the confusion matrix of the classification results

### 💥Linear Discriminant Analysis

- Conduct a linear discriminant analysis classification
- Plot the confusion matrix of the classification results, and the standard deviation of results across runs
- Plot the samples using 2 or 3 components

---

## Prerequisites 📰

- Python 3.12 or higher
- pvclust (installed and available in the working directory)

---

## 📋 Usage

### IDE - Jupyter Notebooks

For the most intuitive application of the script, three Jupyter notebooks are provided with step-by-step example implementation.

- `prepare_data.ipynb` includes data cleaning, labelling, denoising, filtering, and standardization options
- `spectral_separability.ipynb` includes visualization and four classifier implementations (hierarchical clustering, spectral angle method, random forest classification, and linear discriminant clssification).
- `spectral_stats.ipynb` includes feature selection and correlation filtering, assumption checks for parametric stats, and stats tests.

### CLI

Most functionality is contained in python scripts structured as classes. Command parsing has not yet been implemented.

---

## 💡 Notes

- **Class hardcoding**: If you are re-using this repository to analyse spectra for different targets, the list of target classes being used in various functions will need to be altered to your specific needs. These include:
`auxiliary.preprocessing.assign_targets_list`, `auxiliary.preprocessing.make_new_labels`, and `auxiliary.tools.sort_classes`

---

## Troubleshooting 🔨

### Common Issues

- Ensure you're using Python 3.12 or higher.
- Ensure pvclust is available
- Check that the input data is formatted as [samples, wavelengths]
- Check that samples are labelled appropriately

---

## References

[1] Suzuki, R., & Shimodaira, H. (2006). Pvclust: An R package for assessing the uncertainty in hierarchical clustering. Bioinformatics, 22(12), 1540–1542. <https://doi.org/10.1093/bioinformatics/btl117>

[2] Turanjanin, A. (2020). Aturanjanin/pvclust [Python]. <https://github.com/aturanjanin/pvclust> (Original work published 2020)

[3] Shimodaira, H. (2002). An Approximately Unbiased Test of Phylogenetic Tree Selection. Syst. Biol., 51(3), 492–508. <https://doi.org/10.1080/10635150290069913>

[4] Kruse, F. A., Lefkoff, A. B., Boardman, J. W., Heidebrecht, K. B., Shapiro, A. T., Barloon, P. J., & Goetz, A. F. H. (1993). The spectral image processing system (SIPS)—Interactive visualization and analysis of imaging spectrometer data. Remote Sensing of Environment, 44(2), 145–163. <https://doi.org/10.1016/0034-4257(93)90013-N>

---

## License

This project is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License: <https://creativecommons.org/licenses/by-sa/4.0/>
