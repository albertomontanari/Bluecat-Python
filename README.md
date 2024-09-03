# bluecat
Python Software for producing confidence limits of a generic prediction through comparison with observed data. Confidence limits are estimated by applying the BLUECAT method that transforms any deterministic model in a stochastic model, from which mean stochastic simulation and confidence limits are obtained.
For more details please refer to https:\\www.albertomontanari.it\bluecat, Koutsoyiannis and Montanari (2022) and Montanari and Koutsoyannis (2024, preprint, not yet available).

To run the software in R under the Linux operating system the following commands can be used from the terminal:

python3 bluecat.py

The software comes with two data sets described in Montanari and Koutsoyiannis (2024, preprint). They refer to the prediction of tree ring withds and the multimodel prediction of daily river flow for the Arno River Basin, in Italy (see Montanari and Koutsoyiannis (2024), preprint, not yet available).

To reproduce the case study of the tree ring width simulation (Franke et al., 2022), rename the file settings-TRW.txt to settings.txt. Pay attention to the description of the various options that is given in the file. Additional details for the options are given here below. Once settings are defined, you just have to launch the software with the above command.

To run the case study of the multimodel simulation for the Arno river basin, please rename the file setting-Arno.txt to settings.txt and then launch the code.

If diagnostic is performed, the code will stop when each diagnostic plot is produced. To restart the code, you just need to close the graphic window. Plots will be saved as pdf files on the directory from which you launched the code.

Results will be written in the file results.txt. For each simulation, confidence bands and best estimate for the prediction will be given, along with the percentage of points lying outside the confidence band and the efficiency of determistic and stochastic model (and multimodel simulation if performed).

If you would like to apply the code to other case studies, please pay attention to the option and the format of input data. They need to be given in the same template.

Please contact me if you would like additional help.

## Additional details on the settings

The code takes as input a calibration data set formed by deterministic model predictions and corresponding observed values, for one or more models. By comparing prediction with observation BLUECAT corrects the prediction itself and computes confidence bands.

### Options (to be specified in the settings.txt file)

###Filename calibration data
It is a text file with columns qoss (observed data) and qsim (simulated data) in calibration for each prediction model.

###Filename of model predictions whose uncertainty is to be estimated
In each column a prediction is to be given.

###Nmodels
Number of models in the multimodel simulation.

###unmeas
Uncertainty measure for identifying the best performing model in the multimodel simulation, at each prediction step. Values are 1 for width of the confidence band, 2 for width of the confidence band divided by the stochastic prediction, 3 for the difference between the deterministic and stochastic prediction, 4 for the difference between the deterministic and stochastic prediction divided by the stochastic prediction.

###predsmodel
Character variable to specify if stochastic prediction is generated by the mean (predsmodel="avg") or the median (predsmodel="mdn") of the conditional distribution. Default is predsmodel="avg".}

###empquant
Logical variable to specify if empirical quantiles or robust estimation is to be used for estimating confidence limits. Default is robust estimation, namely, empquant=F.

###siglev
Significance level for confidence limits estimation. Default is siglev=0.2.

###m
Parameter to determine the sample size of river flow neighbours to be used for estimating the probability distribution of true river flow conditioned to the simulated river flow. Default is m=100.

###m1
Number of k-moments used to estimate the PBF distribution on the sample of mean stochastic prediction to make robust quantile estimation. Default is m1=80.

###paramd
Initial parameter values for the PBF distribution fitting the sample of mean stochastic prediction to make robust quantile estimation. Default is paramd=c(0.1,1,10,NA).

###lowparamd
Lower bound for the parameter values of the PBF distribution fitting the sample of mean stochastic prediction to make robust quantile estimation. Default is lowparamd=c(0.001,0.001,0.001,0).

###upparamd
Upper bound for the parameter values of the PBF distribution fitting the sample of mean stochastic prediction to make robust quantile estimation. Default is upparamd=c(1,5,20,NA).

###plotflag

Logical value. Specifies if diagnostic is to be returned. Default is plot=F. If plot=T a scapperplot of observed versus simulated by the stochastic model river flows is returned along with its efficiency. The diagnostic plots are also returned, along with the percentage of observations lying outside the confidence bands.

###cpptresh
Low flow threshold to draw the diagnostic plots and to compute percentage of points lying outside the confidence band. Default value is cpptresh=0, which means that zero values for the simulated river flow are not considered when drawing the plots and computing percentages.

###qoss
Vector of observed data corresponding to the model prediction whose uncertainty is to be estimated if available. It is not necessary for uncertainty estimation. If provided, it must have the same length as the vector of modelprediction dmodelsim.

###Flag for specifying if observed data for the prediction are available to perform the diagnostic.
Options are "Yes" or "no"

###Optimisation method for computing k-moments.
Options are: Nelder-Mead, Powell and others (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)


##References

Franke, J., Evans, M. N., Schurer, A., & Hegerl, G. C. (2022). Climate change detection and attribution using observed and simulated tree-ring width. Climate of the past, 18(12), 2583-2597.

Koutsoyiannis, D., & Montanari, A. (2022). Bluecat: A local uncertainty estimator for deterministic simulations and predictions. Water Resources Research, 58(1), e2021WR031215.

Montanari, A. & Koutsoyiannis, D. (2024). Uncertainty estimation for environmental multimodel simulations: the BLUECAT approach and software. Submitted manuscript.
