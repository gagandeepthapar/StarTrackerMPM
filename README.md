# A Statistical Model for Measuring Accuracy and Precision in LEO-based Star Trackers: An Analysis of Disturbance Propagation
This repository contains the work for my thesis in partial requirements of the Master of Science Degree in Aerospace Engineering from California Polytechnic State University - San Luis Obispo.

## Abstract
As the complexity of CubeSat missions continues to escalate, the usage of star trackers as reliable guidance systems has garnered widespread interest.
This increased demand, coupled with prohibitive costs from commercial vendors, has made the development of in-house star trackers a compelling alternative.
However, the high sensitivity of star trackers to various sources of error poses a significant challenge, especially given the lack of existing tools to estimate the accuracy of a given configuration.
This thesis presents a novel approach to address this gap by introducing a statistical model capable of predicting the accuracy of star trackers via replication of the underlying measurement process and error models derived from first principles.
The study examines the associations between star tracker accuracy and sensor-specific factors such as image plane translation, rotation, and centroiding precision and extends to assessing the impact of the thermal and radiation components of the LEO environment on star tracker accuracy throughout its operational lifespan with respect to their effects on additional distortion and image noise, respectively.
It is determined that deviations in image plane translation result in a consistent bias proportional to the magnitude of lateral aberration on accuracy while rotations in the image plane introduce considerable error in attitude estimates.
The influence of centroiding precision and image noise on star vector determination and star detection, respectively, and their relationship with overall accuracy is also examined based on experimental and simulation data.

## Considerations
Several factors need to be considered to accurately determine the measurement accuracy and precision of a star tracker in orbit. A few of the considerations are listed below

- Hardware Aberrations (e.g., Focal Plane tilt, translation)
- Sensor Noise 
- Centroiding Accuracy 
- LEO Thermal Environment (and its effects on hardware aberration and sensor noise)
- LEO Radiation Environment (and its effects on sensor noise)
