# msd_dynamics
Tools used to extract dynamic information from BD trajectories

This script defines several functions used to compute the mean squared displacement of particles over time. 
The function allows averaging over multiple time origins as defined by the user to lower the noise. 
Additionally, bootstrapping is implemented to develop a confidence interval on the computed msd.
References for particular implementation details are provided where applicable.
