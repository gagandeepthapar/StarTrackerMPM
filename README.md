# Star Tracker Measurement Process Model (STMPM)
A Measurement Process Model of a Star Tracker in LEO using Monte Carlo Analysis and Effects of the Environment via First Principles Analysis.

## Considerations
Several factors need to be considered to accurately determine the measurement accuracy and precision of a star tracker in orbit. A few of the considerations are listed below

### Hardware
- Focal Length Deviation
- Focal Array Tilt
- Radial Distortion
- Principal Point Deviation
- Baffle (WIP)

### Software
- Centroiding Accuracy
- Noise Minimization (WIP)

### Environment
- Temperature Change/Cycling (WIP)
- Radiation  (WIP)
- Atomic Oxygen (WIP)

## Usage
A few examples of how the tool can be used are listed below

#### **Improved Centroiding Effect**
In *Optical System Error Analysis and Calibration Method of High-Accuracy Star Trackers*, Sun et al., the expected centroiding accuracy used for Monte Carlo Analysis was set to 0.1 pixels (3-sigma). Since publication, several centroiding algorithms and approaches have been developed and have vastly improved. In *A Novel Systematic Error Compensation Algorithm Based on Least Squares Support Vector Regression for Star Sensor Image Centroid Estimation*, Yang et al., a method was proposed bringing the expected centroid error to 6e-5 pixels! Using the STMPM, we can see that the expected star tracker accuracy improvement solely due to the new centroiding algorithm is improved by 1 arcsecond! The results were achieved using Monte Carlo Analysis with the derived model. It should be noted that Sun et al. uses a single incident angle whereas the STMPM uses a uniformly random incident angle within +/- 10 degrees.

![SuperCentroid](media/inAction/improvedCentroidAnalysis.png)

