# EEG Signal Processing Fourth Project

## Overview

This project involves analyzing EEG signals for both event-related potentials (ERPs) and steady-state visual evoked potentials (SSVEPs) using various techniques including averaging, frequency content analysis, Canonical Correlation Analysis (CCA), and Common Spatial Pattern (CSP) methods.

## 1. ERP Signal Analysis

### Data
- **File**: `mat.ERP_EEG`
- **Description**: EEG from channel Pz, with 2550 trials, recorded at 240 Hz. The data includes brain response to visual stimulation with 300P potentials.

### Tasks

- **a)** For varying numbers of trials (`100:100:2500`), plot the average response. Compare the plots in one figure, ordered by increasing number of trials.
- **b)** Plot the maximum absolute amplitude of the signal versus the number of averaged trials.
- **c)** Plot the root mean square (RMS) error between the 𝑛-th and the (𝑛-1)-th average patterns as a function of the number of averaged trials (`1:2550`).
- **d)** Determine the minimum number of trials needed to extract the 300P response effectively based on results from parts (a), (b), and (c).
- **e)** Compare the average response from part (d) with responses averaged over:
  - 2550 trials
  - 𝑛 trials (where 𝑛 is the number obtained from part (d))
  - Randomly selected subsets of 2550 responses with 𝑛 trials
  - Random subsets with different 𝑛 values
- **f)** Investigate real-world studies using 300P patterns. Compare the number of repetitions used in practical studies with the results obtained in previous sections. Discuss any discrepancies.

## 2. SSVEP Signal Analysis

### Data
- **File**: `mat.SSVEP`
- **Description**: Contains SSVEP signals from a user, including 6 channels of EEG data (`Pz`, `O1`, `O2`, `P7`, `P8`, `Oz`), stimulus frequencies, and sample times.

### Tasks

#### Frequency Content Analysis

- **a)** Preprocess the data:
  - **a1)** Apply a band-pass filter to remove frequencies below 1 Hz and above 40 Hz.
  - **a2)** Segment each of the 15 trials into 5-second windows.
  - **a3)** Calculate and plot the frequency content for each channel using the `pwelch` function.
  - **a4)** Assess whether all channels show the same frequency content in each trial and explain any differences.
  - **a5)** Determine the dominant frequency in each trial and discuss the reasons for the observed peaks.

#### Canonical Correlation Analysis (CCA)

- **b)** Perform CCA:
  - **b1)** Segment each of the 15 trials into 5-second windows.
  - **b2)** Use CCA to determine dominant frequencies. Implement `canoncorr` and evaluate the classification accuracy.
  - **b3)** Assess whether reducing the number of channels affects the classification performance.
  - **b4)** Investigate if reducing the window length affects classification accuracy.

## 3. CSP for Two-Class Classification

### Data
- **File**: `mat.CSPdata`
- **Description**: Contains EEG data for mental imagery tasks (e.g., foot movement vs. mental subtraction) with training and test datasets.

### Tasks

- **a)** Obtain spatial filters using CSP with the training data and apply them to the training data. Plot the filtered signals for the first and last filters and compare them.
- **b)** Plot the spatial filters using `m.plottopomap` and compare them.
- **c)** Perform 4-fold cross-validation:
  - Split the training data into 4 subsets, using 3 for training and 1 for validation each time.
  - Train the CSP algorithm and extract CSP features. Use simple classifiers (e.g., kNN, linear SVM, LDA) to classify the validation data and compute the average accuracy.
  - Determine the optimal number of CSP filters.
- **d)** Train the best classifier on the full training set and test it on the test data. Store the predicted labels in `TestLabel`.

### Notes

- **EEG Recording Details**:
  - **System**: GAMMAsys tec.g
  - **Electrodes**: 30 channels according to the 10-20 system
  - **Filtering**: Band-pass filter (0.5-100 Hz), notch filter (50 Hz)
  - **Sampling Rate**: 256 Hz
  - **Trial Length**: 1 second (256 time samples)
  - **Total Trials**: 210
  - **Training Trials**: 165 (Labels: 1 - foot movement, 0 - mental subtraction)
  - **Test Trials**: 45

