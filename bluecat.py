#Python code for applying Bluecat - For more details see README and https:\\www.albertomontanari.it\bluecat
import pandas as pd
import numpy as np
from scipy.special import gamma, loggamma, beta
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#Definition of function to fit the parameters of the PBF distribution

def fitPBF(paramd):
    lambda1k = (1 + (beta(1/paramd[1]/paramd[0] - 1/paramd[1], 1/paramd[1]) / paramd[1]) ** paramd[1]) ** (1/paramd[1]/paramd[0])
    lambda1t = 1 / (1 - (1 + (beta(1/paramd[1]/paramd[0] - 1/paramd[1], 1/paramd[1]) / paramd[1]) ** paramd[1]) ** (-1/paramd[1]/paramd[0]))
    lambdainfk = gamma(1 - paramd[0]) ** (1/paramd[0])
    lambdainft = gamma(1 + 1/paramd[1]) ** (-paramd[1])
    Tfromkk = lambdainfk * ptot + lambda1k - lambdainfk
    Tfromkt = lambdainft * ptot + lambda1t - lambdainft
    Tfromdk = (1 / (1 + paramd[0] * paramd[1] * ((kp - paramd[3]) / paramd[2]) ** paramd[1])) ** (-1/paramd[0]/paramd[1])
    Tfromdt=1 / (1 - (1 + paramd[0] * paramd[1] * ((kptail - paramd[3]) / paramd[2]) ** paramd[1]) ** (-1/paramd[0]/paramd[1]))
    lsquares = np.sum(np.log(Tfromkk / Tfromdk) ** 2) + np.sum(np.log(Tfromkt / Tfromdt) ** 2)
    return lsquares if not np.isnan(lsquares) else 1e20

#Reading data from files - Reading Bluecat input and initial values from file settings.txt
settings = open("settings.txt", "r")
content = settings.readlines()
#Filename calibration data
resultcalib = content[2][:30].strip()
#dmodelsim: simulated data from the D model, one column per each model
dmodelsim = content[3][:30].strip()
#nomodels: number of models in multimodel simulation
nmodels = int(float(content[4][:30].strip()))
#uncmeas: uncertainty measure in multimodel simulatio
uncmeas = int(float(content[5][:30].strip()))
#predsmodel: how to compute stochastic prediction, mean (avg) or median (mdn)
predsmodel = content[6][:30].strip()
#empquant: to compute confidence limits through empirical quantiles or k-moments
empquant = content[7][:30].strip()
empquant = empquant=="True"
#siglev: significance level for confidence limits
siglev = float(content[8][:30].strip())
#m and m1: Bluecat parameters
m = int(float(content[9][:30].strip()))
m1 = int(float(content[10][:30].strip()))
#paramd (along with lower and upeer values lowparamd and upparamd): parameters of the PBF distribution
paramd = np.fromstring(content[11][:30], sep=' ')
lowparamd = np.fromstring(content[12][:30], sep=' ')
upparamd = np.fromstring(content[13][:30], sep=' ')
#plotflag: compute diagnostics or not
plotflag = content[14][:30].strip()
plotflag = plotflag=="True"
#cpptresh: threshold for computing confidence limits
cpptresh = float(content[15][:30].strip())
qosspred = content[16][:30].strip()
if qosspred != 'None':
    qosspred = content[17][:30].strip()
optimethod = content[18][:30].strip()

settings.close()
# If there are any open figures, close them (equivalent to R's graphics.off())
plt.close('all')
#Opening the file for writing the results
f = open("results.txt", "w")
#Reading Bluecat calibration data (from file resultcalib.txt) and modelprediction (from file dmodelsim.txt)
#Beware: resultcalib.txt must be a text file, with columns headers qsim1, qoss1, qsim2, qoss2 and so on.
resultcalib = pd.read_table(resultcalib, sep=" ")
#Beware: dmodelsim.txt must be a text file, with columns header dmodelsim
dmodelsim = pd.read_table(dmodelsim, sep=" ")
#qoss pred are the observed data to be compared with the model prediction.
#These may not be available. If they are not, set qosspred='None' in settings.txt.
#If they are available set qosspred='Yes' and diagnostics for uncertainty estimation can be performed
if qosspred != 'None':
    qosspred = pd.read_table(qosspred, sep=" ")
    qosspred = np.array(qosspred['qosspred'])
else:
    qosspred = None
#nstep is the length of the simulation data set

nstep = len(np.array(dmodelsim.iloc[:,0]))
# nstep1 is the length of the calibration data set. Beware: length of the calibration data set is the same for all models
nstep1 = len(np.array(resultcalib.iloc[:,0]))

#Definition of matrixes
if(nmodels>1):
    detprediction=np.zeros((nmodels+1,nstep))
    stochprediction=np.zeros((nmodels+1,nstep))
    lowlimit=np.zeros((nmodels+1,nstep))
    uplimit=np.zeros((nmodels+1,nstep))
    effsmodel=np.zeros((nmodels+1))
    zeta=np.zeros((nmodels+1,nstep))
else:
    detprediction=np.zeros((nmodels,nstep))
    stochprediction=np.zeros((nmodels,nstep))
    lowlimit=np.zeros((nmodels,nstep))
    uplimit=np.zeros((nmodels,nstep))
    effsmodel=np.zeros((nmodels))
    zeta=np.zeros((nmodels,nstep))

for fmodels in range(nmodels):

    # Beware: resultcalib must be a list, with elements qsimi and qossi, i=1,nmodels, in this same order
    # Beware: dmodelsim must be a list, with elements qsimi, i=1,nmodels, in this same order
    dmodelsim1=np.array(dmodelsim.iloc[:,fmodels])
    nstep = len(np.array(dmodelsim.iloc[:,fmodels]))
    # Variable aux is used to order the simulated data in ascending order and later to put back stochastic predictions in chronological order
    aux = np.argsort(dmodelsim1)
    # sortsim contains the simulated data in ascending order
    sortsim = np.sort(np.array(dmodelsim.iloc[:,(fmodels)]),axis=0)
    # Variable aux2 is used to order the simulated calibration data in ascending order and to order observed calibration data according to ascending order of simulated calibration data
    aux2 = np.argsort(np.array(resultcalib.iloc[:,(fmodels*2+1)]))
    # Ordering simulated calibration data in ascending order
    sortcalibsim = np.sort(np.array(resultcalib.iloc[:,(fmodels*2+1)]),axis=0)
    # Ordering observed calibration data in ascending order of simulated calibration data
    qossc = np.array(resultcalib.iloc[:,(fmodels*2)])[aux2]
    #Find the vector of minimal quantities as computed below. It serves to identify the range of observed data to fit
    vectmin = np.minimum(0.5 + (nstep1 - np.arange(1, nstep1 + 1)) * 0.5 / m / 2, np.ones(nstep1))
    #Find the vector of minimal quantities as computed below. It serves to identify the range of observed data to fit
    vectmin1 = np.floor(np.minimum(np.minimum(m, np.arange(1, nstep1 + 1) - 1), (np.arange(1, nstep1 + 1) - 1)[::-1] / vectmin)).astype(int)
    #Defines the vectors of stochastic prediction and confidence bands
    medpred = np.zeros(nstep)
    infpred = np.zeros(nstep)
    suppred = np.zeros(nstep)

    #Definition of auxiliary variable icount, to be used only to print on screen the progress of computation
    icount = 1
    print("\nBluecat uncertainty estimation for model ", (fmodels+1),". This will take some time")
    f.write("\nBluecat uncertainty estimation for model "+str(fmodels+1)+"\n \n")
    print("Do not worry if you get some RuntimeWarnings - they arise from the iterative procedure for computing k-moments")
    print("Computing the mean stochastic prediction")

    #Routine to show the progress of the computation
    for i in range(nstep):
        if i / nstep * 100 > 10 * icount:
            print(f"{icount * 10}%", end=" ")
            icount += 1
        if i / nstep * 100 > 99 and icount == 10:
            print("100%")
            icount += 1

        #Finding the simulated data in the calibration period closest to the data simulated here. Values are multiplied by one million to avoid small differences leading to multiple matches
        indatasimcal1 = np.argmin(np.abs(sortcalibsim * 1e6 - sortsim[i] * 1e6))
        #Define the end of the range of the observed data to fit
        aux1 = indatasimcal1 - vectmin1[indatasimcal1] + int((1 + vectmin[indatasimcal1]) * vectmin1[indatasimcal1])
        #Puts a limit to upper index of conditioned vector of observed data
        aux1 = min(aux1, nstep1)
        #Define the start of the range of the observed data to fit
        aux2 = indatasimcal1 - vectmin1[indatasimcal1]
        #Puts a limit to lower index of conditioned vector of observed data
        aux2 = max(aux2, 0)
        #Compute mean stochastic prediction
        if predsmodel == "avg":
            medpred[i] = np.mean(qossc[aux2:(aux1+1)])
        elif predsmodel == "mdn":
            medpred[i] = np.median(qossc[aux2:(aux1+1)])
    #Put back medpred in chronological order
    medpred = medpred[np.argsort(aux)]
    #Choose between empirical quantiles and k-moments quantiles
    if not empquant:
        #Estimation of ph and pl - orders of the k-moments for upper and lower tail
        #Fitting of PBF distribution on the sample of mean stochastic prediction
        #Adding a constant to the sample to avoid negative values
        const=min(np.concatenate([medpred,sortcalibsim,dmodelsim1]))
        if const <0: medpred=medpred-const
        paramd[3]=0.5*min(medpred)
        upparamd[3]=0.9*min(medpred)
        #Definition of the number of k-moments to estimate on the sample of mean stochastic prediction to fit the PBF distribution
        m2 = np.arange(0, m1 + 1)
        #Definition of the order p of the k-moments to estimate on the sample of mean stochastic prediction to fit the PBF distribution
        ptot = nstep ** (m2 / m1)
        #Estimation of k-moments for each order. k-moments are in kp and kptail
        Fxarr1 = np.zeros(nstep)
        kp = np.zeros(m1 + 1)
        kptail = np.zeros(m1 + 1)

        for ii in range(m1 + 1):
            p1 = ptot[ii]
            c1 = 0
            for iii in range(nstep):
                if (iii+1) < p1:
                    c1 = 0
                elif (iii+1) < p1 + 1 or abs(c1) < 1e-30:
                    c1 = np.exp(loggamma(nstep - p1 + 1) - loggamma(nstep) + loggamma(iii+1) - loggamma(iii - p1 + 2) + np.log(p1) - np.log(nstep))
                else:
                    c1 *= (iii) / (iii+1 - p1)
                Fxarr1[iii] = c1
            kp[ii] = np.sum(np.sort(medpred) * Fxarr1)
            kptail[ii] = np.sum(np.sort(medpred)[::-1] * Fxarr1)
    #End estimation of k-moments
    #Fitting of PBF distribution by using the Python routine minimize with default parameter bounds.
    #If it fails change the default values for lowparamd and upparamd
        result = minimize(fitPBF, paramd, bounds=list(zip(lowparamd, upparamd)),method=optimethod,tol=0.001)
        paramd = result.x
        #Recomputation of lambda1 and lambdainf with the calibrated parameters
        lambda1k = (1 + (beta(1/paramd[1]/paramd[0] - 1/paramd[1], 1/paramd[1]) / paramd[1]) ** paramd[1]) ** (1/paramd[1]/paramd[0])
        lambda1t = 1 / (1 - (1 + (beta(1/paramd[1]/paramd[0] - 1/paramd[1], 1/paramd[1]) / paramd[1]) ** paramd[1]) ** (-1/paramd[1]/paramd[0]))
        lambdainfk = gamma(1 - paramd[0]) ** (1/paramd[0])
        lambdainft = gamma(1 + 1/paramd[1]) ** (-paramd[1])
        ph = 1 / (lambdainfk * siglev / 2) + 1 - lambda1k / lambdainfk
        pl = 1 / (lambdainft * siglev / 2) + 1 - lambda1t / lambdainft
    #Definition of auxiliary variable icount, to be used only to print on screen the progress of computation
    icount = 1
    #Routine for computing upper and lower confidence bands
    print("Computing prediction confidence bands")
    for i in range(nstep):
    #Finding the simulated data in the calibration period closest to the data simulated here. Values are multiplied by one million to avoid small differences leading to multiple matches
        indatasimcal1 = np.argmin(np.abs(sortcalibsim * 1e6 - sortsim[i] * 1e6))
        #Define the end of the range of the observed data to fit
        aux1 = indatasimcal1 - vectmin1[indatasimcal1] + int((1 + vectmin[indatasimcal1]) * vectmin1[indatasimcal1])
        #Puts a limit to upper index of conditioned vector of observed data
        aux1 = min(aux1, nstep1)
        #Define the start of the range of the observed data to fit
        aux2 = indatasimcal1 - vectmin1[indatasimcal1]
        #Puts a limit to lower index of conditioned vector of observed data
        aux2 = max(aux2, 0)
        #Defines the size of the data sample to fit
        count = aux1 - aux2 + 1
        if not empquant:
            #Routine to show the progress of the computation
            if i / nstep * 100 > 10 * icount:
                print(f"{icount * 10}%", end=" ")
                icount += 1
            if i / nstep * 100 > 99 and icount == 10:
                print("100%")
                icount += 1
            #Estimation of k moments for each observed data sample depending on ph (upper tail) and pl (lower tail)
            Fxpow1 = ph - 1
            Fxpow2 = pl - 1
            Fxarr1 = np.zeros(count)
            Fxarr2 = np.zeros(count)
            for ii in range(count):
                if (ii+1) < ph:
                    c1 = 0
                elif (ii+1) < ph + 1 or abs(c1) < 1e-30:
                    c1 = np.exp(loggamma(count - ph + 1) - loggamma(count) + loggamma(ii + 1) - loggamma(ii - ph + 2) + np.log(ph) - np.log(count))
                else:
                    c1 = c1*(ii) / (ii+1 - ph)
                Fxarr1[ii] = c1
                if (ii+1) < pl:
                    c2 = 0
                elif (ii+1) < pl + 1 or abs(c2) < 1e-30:
                    c2 = np.exp(loggamma(count - pl + 1) - loggamma(count) + loggamma(ii + 1) - loggamma(ii - pl + 2) + np.log(pl) - np.log(count))
                else:
                    c2 = c2* (ii) / (ii - pl+1)
                Fxarr2[ii] = c2
            medpred1 = np.sort(qossc[aux2:(aux1+1)])
            #End of k-moments estimation.
            #Computation of confidence bands
            if(const>=0):
                suppred[i] = np.sum(medpred1 * Fxarr1)
                infpred[i] = np.sum(medpred1[::-1] * Fxarr2)
            else:
                suppred[i] = np.sum((medpred1-const) * Fxarr1)
                infpred[i] = np.sum((medpred1-const)[::-1] * Fxarr2)
                suppred[i] = suppred[i]+const
                infpred[i] = infpred[i]+const
                medpred[i] = medpred[i]+const
            #Do not compute confidence bands with less than 3 data points
            if(count<3):
                suppred[i]=float("nan")
                infpred[i]=float("nan")
        else:
            #Empirical quantile estimation - This is much easier
            #Routine to show the progress of the computation
            if i / nstep * 100 > 10 * icount:
                print(f"{icount * 10}%", end=" ")
                icount += 1
            if i / nstep * 100 > 99 and icount == 10:
                print("100%")
                icount += 1
            count = int(count)
            #Computation of the position (index) of the quantiles in the sample of size count
            count1 = np.ceil(count * siglev / 2).astype(int)
            count2 = np.ceil(count * (1 - siglev / 2)).astype(int)
            #Computation of confidence bands
            infpred[i] = np.sort(qossc[aux2:(aux1+1)])[count1-1]
            suppred[i] = np.sort(qossc[aux2:(aux1+1)])[count2-1]
            #Do not compute confidence bands with less than 3 data points
            if(count<3):
                suppred[i]=float("nan")
                infpred[i]=float("nan")
        #If plotflag=T compute the data to draw the diagnostic plots
        if plotflag and nstep1>20 and qosspred is not None:
            #Defines the z vector
            qossaux=qosspred[aux]
            #Finds the index in sorted vector of data defining the conditional distribution that is closest to the considered observed value
            indataosssim1=np.argmin(np.abs(np.sort(qossc[aux2:(aux1+1)])*1e6-qossaux[i]*1e6))
            #Finds probability with Weibull plotting position, only if there are at least three data points
            if count>=3:
                zeta[fmodels,i]=(indataosssim1+1)/(count+1)
    #Put confidence bands back in chronological order
    infpred=infpred[np.argsort(aux)]
    suppred=suppred[np.argsort(aux)]
    # Diagnostic activated only if plotflag=True, nstep1>20 and qoss is not None
    if plotflag and nstep1 <= 20:
        print("Cannot perform diagnostic if simulation is shorter than 20 time steps")
    if plotflag and qosspred is None:
        print("Cannot perform diagnostic if observed flow corresponding to model simulation is not available")
    if plotflag and nstep1 > 20 and qosspred is not None:
        # Plotting the diagnostic plots and scatterplot
        # Datapoints with observed flow lower than cpptresh are removed
        # Preparing data for D-model plot
        sortdata = np.sort(dmodelsim.iloc[:,fmodels][qosspred > cpptresh])
        sortdata1 = np.sort(medpred[qosspred > cpptresh])
        zQ = np.zeros(101)
        z = np.linspace(0, 1, 101)
        fQ = sortdata[np.ceil(z[1:]*len(sortdata)).astype(int) - 1]
        fQ = np.insert(fQ, 0, 0)
        zQ[0] = 0
        # Preparing data for S-model plot
        zQ1 = np.zeros(101)
        z1 = np.linspace(0, 1, 101)
        fQ1 = sortdata1[np.ceil(z1[1:]*len(sortdata1)).astype(int) - 1]
        fQ1 = np.insert(fQ1, 0, 0)
        zQ1[0] = 0
        qosspred2 = qosspred[qosspred > cpptresh]
        for i in range(1, 101):
            zQ[i] = len(qosspred2[qosspred2 < fQ[i]]) / len(sortdata)
            zQ1[i] = len(qosspred2[qosspred2 < fQ1[i]]) / len(sortdata1)
        # Combining 4 plots in the same window
        plt.figure(figsize=(12, 10))

        # First plot
        plt.subplot(2, 2, 1)
        plt.plot(z, zQ, 'bo-', label='D-model',markersize=4)
        plt.plot(z1, zQ1, 'ro-', label='S-model',markersize=4)
        plt.plot([0, 1], [0, 1], 'k-')
        plt.xlabel('z')
        plt.ylabel(r'$F_q(F_Q^{-1}(z))$')
        plt.legend()
        plt.grid(True)
        plt.title("Combined probability-probability plot")

        # Second plot
        plt.subplot(2, 2, 2)
        plt.plot(np.sort(np.array(zeta[fmodels,])[np.array(zeta[fmodels,]) != 0]), np.linspace(0, 1, len(np.array(zeta[fmodels,])[np.array(zeta[fmodels,]) != 0])), 'bo-',markersize=3)
        plt.plot([0, 1], [0, 1], 'k-')
        plt.xlabel('z')
        plt.ylabel(r'$F_z(z)$')
        plt.grid(True)
        plt.title("Predictive probability-probability plot")
        aux4 = np.argsort(medpred)
        #sortmedpred contains the data simulated by stochastic model in ascending order
        sortmedpred = np.sort(medpred)
        #Ordering observed calibration data and confidence bands in ascending order of simulated data by the stochastic model
        sortmedpredoss = qosspred[aux4]
        sortsuppred = suppred[aux4]
        sortinfpred = infpred[aux4]

        # Third plot
        plt.subplot(2, 2, 3)
        plt.xlim(min(min(np.concatenate([sortmedpred,sortmedpredoss])),0),max(np.concatenate([sortmedpred, sortmedpredoss])*1.1))
        plt.ylim(min(min(np.concatenate([sortmedpred,sortmedpredoss])),0),max(np.concatenate([sortmedpred, sortmedpredoss])*1.1))
        plt.plot(sortmedpred, sortmedpredoss, 'ro', label='Observed data',markersize=2)
        #Adding confidence bands. Not plotted for higher values of flow to avoid discontinuity
        aux5 = nstep - 10
        plt.plot(sortmedpred[:aux5], sortsuppred[:aux5], 'b-', label='Confidence band')
        plt.plot(sortmedpred[:aux5], sortinfpred[:aux5], 'b-')
        plt.plot([min(min(np.concatenate([sortmedpred,sortmedpredoss])),0), max(np.concatenate([sortmedpred, sortmedpredoss]))],
             [min(min(np.concatenate([sortmedpred,sortmedpredoss])),0), max(np.concatenate([sortmedpred, sortmedpredoss]))], 'k-')
        plt.xlabel("Data simulated by stochastic model")
        plt.ylabel("Observed data")
        plt.grid(True)
        #Compute the efficiency of the simulation and put it in the plots
        eff = 1 - np.sum((medpred - qosspred) ** 2) / np.sum((qosspred - np.mean(qosspred)) ** 2)
        plt.legend(loc="upper left", title=f"Efficiency S-model = {np.round(eff, 2)}")
        plt.title("Scatterplot S-model predicted versus observed data")

        # Fourth plot
        plt.subplot(2, 2, 4)
        plt.plot(dmodelsim.iloc[:,fmodels][aux4], sortmedpredoss, 'ro', label='Observed data',markersize=2)
        plt.plot([min(min(np.concatenate([dmodelsim.iloc[:,fmodels][aux4],sortmedpredoss])),0), max(np.concatenate([dmodelsim.iloc[:,fmodels][aux4], sortmedpredoss]))],
             [min(min(np.concatenate([dmodelsim.iloc[:,fmodels][aux4],sortmedpredoss])),0), max(np.concatenate([dmodelsim.iloc[:,fmodels][aux4], sortmedpredoss]))], 'k-')
        plt.xlabel("Data simulated by deterministic model")
        plt.ylabel("Observed data")
        plt.grid(True)
        #Compute the efficiency of the simulation and put it in the plots
        eff1 = 1 - np.sum((dmodelsim1 - qosspred) ** 2) / np.sum((qosspred - np.mean(qosspred)) ** 2)
        plt.legend(loc="upper left", title=f"Efficiency D-model = {np.round(eff1, 2)}")
        plt.title("Scatterplot D-model predicted versus observed data")
        plt.tight_layout()
        plt.savefig("Model"+str(fmodels)+".pdf")
        plt.show(block=True)

    # Computing percentages of points lying outside the confidence bands by focusing on observed data greater than cpptresh
    if plotflag and nstep1 > 20 and qosspred is not None:
        qoss2 = qosspred[qosspred > cpptresh]
        percentage_above_upper = len(qoss2[qoss2 > suppred[qoss2 > cpptresh]]) / len(qoss2[~pd.isna(suppred)]) * 100
        percentage_below_lower = len(qoss2[qoss2 < infpred[qoss2 > cpptresh]]) / len(qoss2[~pd.isna(infpred)]) * 100
        print("Percentage of points lying above the upper confidence limit="+str(round(percentage_above_upper,2))+"%")
        print("Percentage of points lying below the lower confidence limit="+str(round(percentage_below_lower,2))+"%")

    if qosspred is not None:
        eff = 1 - np.sum((medpred - qosspred) ** 2) / np.sum((qosspred - np.mean(qosspred)) ** 2)
        detprediction[fmodels,]=dmodelsim.iloc[:,fmodels]
        stochprediction[fmodels,]=medpred
        lowlimit[fmodels,]=infpred
        uplimit[fmodels,]=suppred
        effsmodel[fmodels]=eff
    else:
        detprediction[fmodels,]=dmodelsim.iloc[:,fmodels]
        stochprediction[fmodels,]=medpred
        lowlimit[fmodels,]=infpred
        uplimit[fmodels,]=suppred
    if plotflag and nstep1 > 20 and qosspred is not None:
        f.write("Efficienty deterministic model: "+str(eff1)+". Efficiency stochastic model: "+str(effsmodel[fmodels])+"\n \n")
        f.write("Percentage of points lying above the upper confidence limit="+str(percentage_above_upper)+"% \n")
        f.write("Percentage of points lying below the lower confidence limit="+str(percentage_below_lower)+"% \n\n")
    f.write("Deterministic prediction     Stochastic prediction    Lower limit    Upper limit \n \n")
    for iiii in range(nstep):
        f.write(str(detprediction[fmodels,iiii])+"  "+str(stochprediction[fmodels,iiii])+"  "+str(lowlimit[fmodels,iiii])+"  "+str(uplimit[fmodels,iiii])+"\n")

# If there is more than one model, multimodel estimation starts here
if nmodels > 1:
    print("\nBluecat uncertainty estimation for multi model. This will take some time")
    f.write("\nBluecat uncertainty estimation for multi model \n\n")
    posminn = np.zeros(nstep, dtype=int)
    newmedpred = np.zeros(nstep)
    newlowlim = np.zeros(nstep)
    newuplim = np.zeros(nstep)
    newdmodelsim = np.zeros(nstep)
    confband = np.zeros((nmodels, nstep))
    # Identification of the best performing model
    for i in range(nstep):
        for ii in range(nmodels):
            if uncmeas == 2:
                confband[ii, i] = abs((uplimit[ii,i] - lowlimit[ii,i]) / stochprediction[ii,i])
            elif uncmeas == 1:
                confband[ii, i] = abs(uplimit[ii,i] - lowlimit[ii,i])
            elif uncmeas == 3:
                confband[ii, i] = abs(detprediction[ii,i] - stochprediction[ii,i])
            else:
                confband[ii, i] = abs((detprediction[ii,i] - stochprediction[ii,i]) / stochprediction[ii,i])
        posmin = np.argmin(np.abs(confband[:, i]))
        if np.isnan(np.max(confband[:,i])):
            posmin = np.argmax(effsmodel)
#        if len(posmin) == 0:
#            posmin = np.argmax(effsmodel)
        posminn[i] = posmin
        newmedpred[i] = stochprediction[posmin,i]
        newlowlim[i] = lowlimit[posmin,i]
        newuplim[i] = uplimit[posmin,i]
        newdmodelsim[i] = detprediction[posmin,i]
    # Diagnostic activated only if plotflag == True, nstep1 > 20, and qosspred is not None

    if plotflag and nstep1 > 20 and qosspred is not None:
        # Plotting the diagnostic plots and scatterplot
        # Datapoints with observed flow lower than cpptresh are removed
        # Preparing data for D-model plot
        sortdata = np.sort(newdmodelsim[qosspred > cpptresh])
        sortdata1 = np.sort(newmedpred[qosspred > cpptresh])
        zQ = np.zeros(101)
        z = np.linspace(0, 1, 101)
        fQ = sortdata[np.ceil(z[1:]*len(sortdata)).astype(int) - 1]
        fQ = np.insert(fQ, 0, 0)
        zQ[0] = 0
        # Preparing data for S-model plot
        zQ1 = np.zeros(101)
        z1 = np.linspace(0, 1, 101)
        fQ1 = sortdata1[np.ceil(z1[1:]*len(sortdata1)).astype(int) - 1]
        fQ1 = np.insert(fQ1, 0, 0)
        zQ1[0] = 0
        qosspred2 = qosspred[qosspred > cpptresh]
        for i in range(1, 101):
            zQ[i] = len(qosspred2[qosspred2 < fQ[i]]) / len(sortdata)
            zQ1[i] = len(qosspred2[qosspred2 < fQ1[i]]) / len(sortdata1)
        # Combining 4 plots in the same window
        plt.figure(figsize=(12, 10))

        # First plot
        plt.subplot(2, 2, 1)
        plt.plot(z, zQ, 'bo-', label='D-model',markersize=4)
        plt.plot(z1, zQ1, 'ro-', label='S-model',markersize=4)
        plt.plot([0, 1], [0, 1], 'k-')
        plt.xlabel('z')
        plt.ylabel(r'$F_q(F_Q^{-1}(z))$')
        plt.legend()
        plt.grid(True)
        plt.title("Combined probability-probability plot")

        # Recomputation of the variable zeta for the multimodel
        zetanew = np.zeros(nstep)
        for zcount in range(1, nmodels):
            posz = np.where(posminn == zcount)
            zetanew[posz] = zeta[zcount][posz]

        # Second plot
        plt.subplot(2, 2, 2)
        plt.plot(np.sort(np.array(zetanew[np.array(zetanew) != 0])), np.linspace(0, 1, len(np.array(zetanew[np.array(zetanew)!=0]))), 'bo-',markersize=3)
        plt.plot([0, 1], [0, 1], 'k-')
        plt.xlabel('z')
        plt.ylabel(r'$F_z(z)$')
        plt.grid(True)
        plt.title("Predictive probability-probability plot")
        aux4 = np.argsort(newmedpred)
        #sortmedpred contains the data simulated by stochastic model in ascending order
        sortmedpred = np.sort(newmedpred)
        #Ordering observed calibration data and confidence bands in ascending order of simulated data by the stochastic model
        sortmedpredoss = qosspred[aux4]
        sortsuppred = newuplim[aux4]
        sortinfpred = newlowlim[aux4]

        # Third plot
        plt.subplot(2, 2, 3)
        plt.xlim(min(min(np.concatenate([sortmedpred,sortmedpredoss])),0),max(np.concatenate([sortmedpred, sortmedpredoss])*1.1))
        plt.ylim(min(min(np.concatenate([sortmedpred,sortmedpredoss])),0),max(np.concatenate([sortmedpred, sortmedpredoss])*1.1))
        plt.plot(sortmedpred, sortmedpredoss, 'ro', label='Observed data',markersize=2)
        #Adding confidence bands. Not plotted for higher values of flow to avoid discontinuity
        aux5 = nstep - 10
        plt.plot(sortmedpred[:aux5], sortsuppred[:aux5], 'b-', label='Confidence band')
        plt.plot(sortmedpred[:aux5], sortinfpred[:aux5], 'b-')
        plt.plot([min(min(np.concatenate([sortmedpred,sortmedpredoss])),0), max(np.concatenate([sortmedpred, sortmedpredoss]))],
             [min(min(np.concatenate([sortmedpred,sortmedpredoss])),0), max(np.concatenate([sortmedpred, sortmedpredoss]))], 'k-')
        plt.xlabel("Data simulated by stochastic model")
        plt.ylabel("Observed data")
        plt.grid(True)
        #Compute the efficiency of the simulation and put it in the plots
        neweff = 1 - np.sum((newmedpred - qosspred) ** 2) / np.sum((qosspred - np.mean(qosspred)) ** 2)
        plt.legend(loc="upper left", title=f"Efficiency S-model = {np.round(neweff, 2)}")
        plt.title("Scatterplot S-model predicted versus observed data")

        # Fourth plot
        plt.subplot(2, 2, 4)
        plt.plot(newdmodelsim[aux4], sortmedpredoss, 'ro', label='Observed data',markersize=2)
        plt.plot([min(min(np.concatenate([newdmodelsim[aux4],sortmedpredoss])),0), max(np.concatenate([newdmodelsim[aux4], sortmedpredoss]))],
             [min(min(np.concatenate([newdmodelsim[aux4],sortmedpredoss])),0), max(np.concatenate([newdmodelsim[aux4], sortmedpredoss]))], 'k-')
        plt.xlabel("Data simulated by deterministic model")
        plt.ylabel("Observed data")
        plt.grid(True)
        #Compute the efficiency of the simulation and put it in the plots
        neweff1 = 1 - np.sum((newdmodelsim - qosspred) ** 2) / np.sum((qosspred - np.mean(qosspred)) ** 2)
        plt.legend(loc="upper left", title=f"Efficiency D-model = {np.round(neweff1, 2)}")
        plt.title("Scatterplot D-model predicted versus observed data")
        plt.tight_layout()
        plt.savefig('Multimodel.pdf')
        plt.show(block=True)

        qosstemp = qoss2[~np.isnan(newuplim)]
        qosstemp1 = qoss2[~np.isnan(newlowlim)]
        suppredtemp = newuplim[~np.isnan(newuplim)]
        infpredtemp = newlowlim[~np.isnan(newlowlim)]

        print(f"Percentage of points lying above the upper confidence limit for multimodel = {np.sum(qosstemp[qosstemp > suppredtemp] > cpptresh) / len(qosstemp) * 100:.2f}%")
        print(f"Percentage of points lying below the lower confidence limit for multimodel = {np.sum(qosstemp1[qosstemp1 < infpredtemp] > cpptresh) / len(qosstemp1) * 100:.2f}%")

    if qosspred is not None:
        neweff = 1 - np.sum((newmedpred - qosspred) ** 2) / np.sum((qosspred - np.mean(qosspred)) ** 2)
        detprediction[nmodels,] = newdmodelsim
        stochprediction[nmodels,] = newmedpred
        lowlimit[nmodels,] = newlowlim
        uplimit[nmodels,] = newuplim
        effsmodel[nmodels] = neweff
    else:
        detprediction[nmodels,] = newdmodelsim
        stochprediction[nmodels,] = newmedpred
        lowlimit[nmodels,] = newlowlim
        uplimit[nmodels,] = newuplim

    if plotflag and nstep1 > 20 and qosspred is not None:
        f.write("Efficienty deterministic multimodel: "+str(neweff1)+"Efficiency stochastic multimodel: "+str(effsmodel[nmodels])+"\n\n")
        f.write("Percentage of points lying above the upper confidence limit="+str(np.sum(qosstemp[qosstemp > suppredtemp] > cpptresh) / len(qosstemp) * 100)+"% \n")
        f.write("Percentage of points lying below the lower confidence limit="+str(np.sum(qosstemp1[qosstemp1 < infpredtemp] > cpptresh) / len(qosstemp1) * 100)+"% \n\n")

    f.write("Deterministic prediction     Stochastic prediction    Lower limit    Upper limit \n \n")
    for iiii in range(nstep):
        f.write(str(detprediction[nmodels,iiii])+"    "+str(stochprediction[nmodels,iiii])+"    "+str(lowlimit[nmodels,iiii])+"    "+str(uplimit[nmodels,iiii])+"\n")

    if plotflag and nstep1 > 20 and qosspred is not None:
        f.write("\nResults of model selection at each time step\n\n")
        f.write(str(posminn))

