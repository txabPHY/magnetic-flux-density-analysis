#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 11:56:31 2023

@author: theobeevers
"""

"""
This is my python code for the Computing 2 Coursework that calculates values for the magnetic flux density,
beta and the time damping constant. It also calculates the relatonship between the B values for different energies.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve

def openfile(filename):
    """Function that opens and reads the file storing the metadata from the file,
    and returning the wanted data in two arrays of times and channels.
    
    Args:
        filename(str):
            The name of the data file to analyse.
    Returns:
        times(arr):
            The times at which the positrons were detected at each detector
        channels(arr):
            The channels at which each positron were detected (1 or 2)
        metadata(dict):
            A dictionary containing the metadata of the file of which the keys
            and elements are located with the "=" string.
        implantation_energies(arr):
            An array of the implantaion energies used in the data.
            
    """
    metadata = dict() # assigning a dictionary to a variable metadata to be added to when the file is read
    try: 
        with open(filename,"r") as myfile:
            for line in myfile:
                line = line.strip() # splitting the file into lines
                if "=" in line:
                    parts = line.split("=")
                    try:
                        metadata[parts[0]]=float(parts[1])# assigning values to the metadata dictionary
                    except ValueError:
                        metadata[parts[0]]=parts[1] # conveting to floats is possible or keeping as strings
                if line == "&END":
                    break #finding the end of the metadata to know when to start reading the file
            else:
                raise RuntimeError("""File didn't have an &END line to indentify the 
                                   end of the metadata and start of the data""") # raising an appropriate error
                                   
            column_names=next(myfile).strip().split("\t")
            
            try:    
                data = np.genfromtxt(myfile, 
                                     delimiter='\t'
                                     ,unpack=False) # getting the data from the file
            
                if np.isnan(data).any() == True:
                    print("""All data points should take float values. The data contained one or more points 
                          which were not float values and these rows were taken out of the data.""")
                    row_to_take_out = np.where(np.isnan(data))[0]
                    data[row_to_take_out,:] = 0 # this code checks wether any data is not a number, notfifying the user
                                                # and taking out the corresponding row.
                if (data<0).any():
                    print("""Negative values were detected in the data and the row was taken out 
                          to avoid possible miscalculation""")
                    row_to_take_out = np.where(data<0)[0]
                    data=np.delete(data, row_to_take_out,axis=0) # this code checks if any values are negative
                                                                 # and takes out the corresponding row
            except ValueError as VE:
                print("""A value error was caught in the data and was unable to be rectified.
                      The code was therefore unable to be analysed""")
                raise VE
    
            times = data[:,0] 
            channels = data[:,1] # assigns the initial columns to the variables times and channels
    
    except TypeError as TE:
        print("""Input file is not of a valid form. Please check the file type""" )
        raise TE
    except IOError as IOE:
        print("Error reading or opening the file:", filename)
        raise IOE # more error checking 
        
    if "Multiple" not in metadata:
        raise KeyError("""The Multiple key was not found in the metadata, so the datatype 
                       is not known the data is not able to be analysed""")
    if "Damping" not in metadata:
        raise KeyError("""The Damping key was not found in the metadata, so the datatype 
                       is not known the data is not able to be analysed""")
                # checks for the appriopriate keys in the metadata
    try:
        implantation_energies=np.array(metadata['Implantation Energy (keV)'].split(',')).astype(float)
    except AttributeError:
        implantation_energies=np.array([metadata['Implantation Energy (keV)']]).astype(float)
    except KeyError:
        raise KeyError("""The Implantation Energy (keV) key was not detected, please check the file that
              this key is in the metadata""")
    # creating an array of implantation energies 
    
    length = len(implantation_energies)
    
    if metadata['Multiple'] == "True":
        for i in range(1,length):
            times = np.column_stack((times,data[:,2*i]))
            channels = np.column_stack((channels,data[:,2*i+1]))
    # for loop that adds the columns to the arrays for times and channels
    return times, channels, metadata, implantation_energies # the returned variables to be used in the rest of the code

def desired_bins(max_time, number_of_bins):
    """Function what calculates the time bins wanted for the histogram from the
    max time in the array and the number of bins wanted. It also returns the midpoints
    of each bin"""
    t_bins = np.linspace(0,max_time, number_of_bins) # array for the bins up to the max time value
    difference = np.diff(t_bins)
    addition = difference / 2
    midpoints = t_bins[:-1] + addition # midpoints of each bin
    
    return t_bins, midpoints

def histogram_func(bins, left_channel, right_channel):
    "Function that collects the histogram data for each energy level"
    
    left_data = np.histogram(left_channel, bins) # histogram data 
    right_data = np.histogram(right_channel, bins)
    
    return left_data, right_data

def histogramplot(bins, left_channel, right_channel):
    """Function that takes in the data for the energy level
    and plots the desired histogram"""
    
    plt.figure()
    plt.hist(left_channel, bins, color="black") # plot for the left histogramn
    plt.xlabel("Time $(\mu s)$")
    plt.ylabel("Counts")
    plt.xlim(0, max(bins))
    plt.title("Left channel detector 10keV (py21txab)")
    
    plt.figure()
    plt.hist(right_channel, bins, color="black") # plot for the right histogram
    plt.xlabel("Time $(\mu s)$")
    plt.ylabel("Counts")
    plt.xlim(0, max(bins))
    plt.title("Right channel detector 10keV (py21txab)")
    
    return

def histogram_errors(histogram_data):
    "Function that calculates the errors in the histogram counts"
    error = np.sqrt(histogram_data) # errors for each histogram
    return error

def asymmetry_func(left_histogram_counts, right_histogram_counts):
    """Function to calculate the asymmetry using the data from the histograms.
    Args:
        left_histogram_counts(arr):
            Nuber of counts in each bin for the left detector.
        right_histogram_counts(arr):
            Nuber of counts in each bin for the right detector.         
    Returns:
        asymmetry(arr):
            The asymmetry of the positrons detected at each detetor.
    """
    
    top_fraction = left_histogram_counts - right_histogram_counts
    bottom_fraction = left_histogram_counts + right_histogram_counts
    asymmetry = top_fraction / bottom_fraction # asymmetry equation for the histogram counts
    
    return asymmetry

def asymmetryerrors(left_data, right_data, left_errors, right_errors):
    "Function to calculate the errors in the asymmetry"
    
    prefactor1 = 1/(np.power((left_data+right_data),4))
    prefactor2 = (np.power(left_data,2) * np.power(right_errors,2)) + (np.power(right_data,2)*np.power(left_errors,2))
    asymmetry_error = 2 * np.sqrt(prefactor1 * prefactor2) 
    
    return asymmetry_error # errors for the asymmetry 

def initial_guess_asymmetry(midpoints, asymmetry):
    """Function that provides variables that are needed to make appropriate guesses
    for B and beta when using the curve_fit function.
    
    Args:
        midpoints(arr):
            The time values which are used to plot the asymmetry .
        asymmetry(arr):
            The asymmetry points that are plotted.         
    Returns:
        time_period(float):
            The time period of the muon spin from precession
        a_max_first(float):
            The value of the first asymmetry point which is the largest
            value of asymmetry with minimal error.
    
    """
    
    asymmetryaccurate=asymmetry[:100] # taking the first 100 values for the asymmetry so the measurments are still accurate
    midpointsaccurate=midpoints[:100] # as the errors increase much more for the later time values
    
    a_sign = np.sign(asymmetryaccurate) # gives the sign of each value (-1,0,1)
    a_sign_diff_abs = abs(np.diff(a_sign)) # finds the difference of these values, and find the magnitude, 
    #this gives values of 0, or 2 if it changed between positive and negative at that point
    index_a_0 = np.where(a_sign_diff_abs==2)[0] # index of these values where asymmetry is close to 0
    index_differences = np.diff(index_a_0) # the differences of these indexes
    length = index_differences.shape[0] # the length of the index array
    
    """Matthew Cross (py21mc) helped me with theory of this code and gave me the idea to take out these values
    with the small index differences when the code couldn't work on certain data files. I have not copied any code 
    of Matthew's, but mentioned because of the specific logic he helped me with."""
    
    for i in range(length): # this loop looks through the array and identifies whether there is a 0 point is too near to another 0 point,
        if index_differences[i] <= 2: # meaning there is an anomaly in the code which would give an incorrect value for the time period.
           index_to_take_out = np.where(index_differences<=2)[0] # if a difference of less than 3 is identified it is taken out of the array
           ones = np.ones_like(index_a_0,dtype=bool)
           ones[index_to_take_out] = False 
           new_a_0 = index_a_0[ones]
           break
        else:
           new_a_0 = index_a_0
    
    time_values_oneless = midpointsaccurate[:-1] 
    time_values_a_is0 = time_values_oneless[new_a_0] # identifying the time values correlating to the a_0 values
    time_period_differences = np.diff(time_values_a_is0)
    time_period = np.average(time_period_differences) * 2 # calculating the time period

    a_max_first = asymmetryaccurate[0] # first asymmetry value
    return time_period, a_max_first

def magnetic_field_guess(time_period):
    """Function that provides an appropriate B guess taking in the time period
    and using the equation B = (2*pi)/(gamma*T) """
    
    gamma = 851.656
    B = (2 * np.pi)/(gamma * time_period) # using the time period to find B
    
    return B

def beta_guess(a_max):
    """Function that provides an appropriate beta guess taking in the first
    (a_max) value and uses the double angle identity to maipulate the asymmetry
    equation and then solve for the equation sin(beta)/3*beta - a_max = 0
    using the fsolve function."""
    
    def f(beta, A):
        return (np.sin(beta)/(3*beta)) - A
    
    beta_guess = fsolve(f, x0=1, args=(a_max,)) # using the first asymmetry value to get beta

    return beta_guess[0]

def asymmetry_equation_no_damping(time, magnetic_flux, beta):
    """Function used to calculate the asymmetry without damping together 
    with the curve_fit function.
    Args:
        time(arr):
            The time values used in the asymmetry equation.
        magnetic_flux(float):
            The B value used in the equation that needs to be optimised.  
        beta(float):
            The beta value used in the equation that needs to be optimised. 
    Returns:
        result(arr):
            The calculted points of asymmetry using the numerical asymmetry equation
            with the optimised values for B and beta.
    """
    gamma = 851.616
    phi = gamma * magnetic_flux * time[:, np.newaxis]
    left = phi - beta
    right = phi + beta
    result = ((-1/3)*(np.sin(left)-np.sin(right))/(2*beta)) # asymmetry equation for curve_fit
    
    return result.flatten()

def asymmetry_equation_damping(time, magnetic_flux, beta, tau):
    """Function used to calculate the asymmetry with damping together 
    with the curve_fit function.
    Args:
        time(arr):
            The time values used in the asymmetry equation.
        magnetic_flux(float):
            The B value used in the equation that needs to be optimised.  
        beta(float):
            The beta value used in the equation that needs to be optimised. 
        tau(float):
            The tau damping constant used in the equation that needs to be optimised.
    Returns:
        result(arr):
            The calculted points of asymmetry using the numerical asymmetry equation
            with the optimised values for B, beta and tau.
    """
    gamma = 851.616
    phi = gamma * magnetic_flux * time[:, np.newaxis]
    left = phi - beta
    right = phi + beta
    expo = np.exp(-time[:, np.newaxis]/tau)
    result = ((-1/3)*(np.sin(left)-np.sin(right))/(2*beta))*expo # asymmetry equation with damping
    
    return result.flatten()

def quadratic(energy, a, b, c):
    """Quadratic function that takes in the array of energies and returns the y
    values used in the curve_fit function with the optimal coefficients"""
    
    y = a*energy**2 + b*energy + c # quadratic equation for curve_fit with multiple energies
    
    return y

def round_func(number, sigfig):
    "function that rounds a given number to an desired number of significant figures"
    # this function was only used for the values in the graphs and not the returned data
    number = '{:.{}g}'.format(number, sigfig)
    rounded_number = float(number)
    
    return rounded_number

def round_sigfig(number, target):
    """function that rounds a number to an amount of significant figures with the
    last significant figure being of the same order of magnitude as another number.
    This code was taken user Marco Sulla on discuss.python.org
    https://discuss.python.org/t/a-function-for-calculating-magnitude-order-of-a-number/18924"""
    # this function was also only used for the graphs not for the returned data
    magnitude = np.floor(np.log10(abs(target)))
    rounded_number = np.round(number, int(-magnitude))
    
    return rounded_number


def ProcessData(filename):
    """The is the main function of the code in which everything operates. All other functions are called
    within this function. It starts by taking the data from the file in the form of arrays for times and channels,
    and proceeds to calculate and plot the histograms for each detector, the asymmetry, and the varying B for each
    energy level if there are multiple energies.
    Args:
        filename(str):
            The name of the file that needs to be analysed.
    Returns:
        results(dict):
            A dictionary of all the calculated results for 10keV for B, beta and tau.
            It also returns the energy coefficents if multiple energies are used.
    """
    B_10keV=None # All variables to be returned in the results dictionary are set to None.
    B_10keV_error=None # This means that if a value is not to be found in the data it will return none,
    beta_10keV=None # or if it is found the varible will change to that value and it will be returned.
    beta_10keV_error=None
    tau_10keV=None
    tau_10keV_error=None
    a=None
    a_error=None
    b=None
    b_error=None
    c=None
    c_error=None
    
    # These arrays are taken from the openfile() function and used for the rest of the data
    times, channels, metadata, implantation_energies = openfile(filename)
    
    B_energies=np.array([]) # Arrays for the B energies and their errors are created
    B_energies_error = np.array([]) # in order to be added to later on.
    
    number_of_bins=400 # number of bins used in the histograms decided here
    multiple = metadata["Multiple"]
    damping = metadata["Damping"] # assigning variabes to the values in the keys Multiple and Damping
    length = len(implantation_energies)
    
    for energies in range(0,length): # a loop over the amount of energies so the data is calculated
                                     # for all energies
            if times.ndim ==1:
                times = times.reshape(-1, 1)
            if channels.ndim==1:
                channels = channels.reshape(-1,1)

            left=times[:,energies][channels[:,energies]==1] # assigning the left and right detectors
            right=times[:,energies][channels[:,energies]==2]
              
            t_max = max(times[:,energies])
            t_bins, midpoints = desired_bins(t_max, number_of_bins) # capturing the bins and the midpoints
            left_data, right_data = histogram_func(t_bins, left, right) # taking the histogram data for each energy
            left_errors = histogram_errors(left_data[0])
            right_errors = histogram_errors(right_data[0]) # taking the histogram errors
            
            asymmetry = asymmetry_func(left_data[0], right_data[0]) # calculating the asymmetry for each energy
            asymmetry_error = asymmetryerrors(left_data[0], right_data[0], left_errors, right_errors) # with errors
            
            time_period, a_max_first = initial_guess_asymmetry(midpoints, asymmetry) # findng the time period and first asymetry value for the guesses
            magnetic_field_g = magnetic_field_guess(time_period)
            beta_g = beta_guess(a_max_first) # obtaining the appropraite guesses from the functions
            
            if damping == "True": # checking if the data involves damping
                 popt, pcov = curve_fit(asymmetry_equation_damping, midpoints, asymmetry, 
                                        p0=(magnetic_field_g, beta_g, 5), sigma=asymmetry_error, 
                                        absolute_sigma=True, maxfev=1000) # using the curve_fit function to obtain optimal values for B, beta and tau
                 perr=np.sqrt(np.diag(pcov)) # obtaining the uncertainties of these values
            elif damping == "False":
                 popt, pcov = curve_fit(asymmetry_equation_no_damping, midpoints, asymmetry, 
                                        p0=(magnetic_field_g, beta_g), sigma=asymmetry_error, 
                                        absolute_sigma=True, maxfev=1000) 
                 perr=np.sqrt(np.diag(pcov)) # other function if damping isn't involved
            else:
                raise KeyError("""Damping type was unable to be identified, please check the type
                               given in the metadata is "True" or "False""")
            
            B_energies = np.append(B_energies, popt[0]) # B for different eenrgy levels added to the array
            B_energies_error = np.append(B_energies_error, perr[0]) 
            
            if implantation_energies[energies]==10: # extracting the values for 10keV for the results
                B_10keV = popt[0]
                B_10keV_error = perr[0]
                beta_10keV = popt[1]
                beta_10keV_error = perr[1]
                asymmetry_10keV = asymmetry
                asymmetry_error_10keV = asymmetry_error
                midpoints_10keV = midpoints
                histogramplot(t_bins, left, right) # plotting the appropriate histogram
                if damping == "True":
                    tau_10keV = popt[2]
                    tau_10keV_error = perr[2]

    B_10keV_error_rd = round_func(B_10keV_error, 1) # taking rounded values to plot on the graph
    B_10keV_rd = round_sigfig(B_10keV, B_10keV_error_rd)
    beta_10keV_error_rd = round_func(beta_10keV_error, 1)
    beta_10keV_rd = round_sigfig(beta_10keV, beta_10keV_error_rd)
    
    if damping == "True":
        
        tau_10keV_error_rd = round_func(tau_10keV_error, 1)
        tau_10keV_rd = round_sigfig(tau_10keV, tau_10keV_error_rd)
        plt.figure()
        plt.errorbar(midpoints_10keV, asymmetry_10keV, yerr=asymmetry_error_10keV, 
                     fmt= 'o',ms=4, capsize=(3),color="black",label="Measured Asymmetry") # plotting the asymmetry with error bars
        plt.plot(midpoints_10keV,asymmetry_equation_damping(midpoints_10keV,B_10keV, 
                     beta_10keV, tau_10keV),'-',lw = '2',color="c",label="Fitted Data") # plotting the optimised fit 
        plt.xlabel(r"Time $(\mu s)$")
        plt.ylabel("Asymmetry ratio")
        plt.title("Measured and fitted asymmetry at 10keV (py21txab)")
        plt.text(0,-0.5,r"""$B={}±{}T$, $\beta={}±{}$rad, 
                 $\tau_{} ={}±{}\mu s$""".format(B_10keV_rd,B_10keV_error_rd,
                 beta_10keV_rd,beta_10keV_error_rd,"{damp}",tau_10keV_rd, tau_10keV_error_rd),) # producing text on the graph
        plt.legend(loc=1)
        tau_10keV=tau_10keV*1e-6 # correcting the units for tau 
        tau_10keV_error=tau_10keV_error*1e-6
    
    elif damping == "False":
        
        plt.figure()
        plt.errorbar(midpoints_10keV, asymmetry_10keV, yerr=asymmetry_error_10keV, 
                     fmt= 'o',ms=4, capsize=(3),color="black",label="Measured Asymmetry") # again checking if dampiing isn't invloved
        plt.plot(midpoints_10keV,asymmetry_equation_no_damping(midpoints_10keV,B_10keV, beta_10keV), # and plotting appropriately
                     '-',lw = '2',color="c",label="Fitted Data")
        plt.xlabel(r"Time $(\mu s)$")
        plt.ylabel("Asymmetry ratio")
        plt.title("Measured and fitted asymmetry at 10keV (py21txab)")
        plt.text(0,-0.4,r"""$B={}±{}T$, $\beta={}±{}rad$""".format(B_10keV_rd,
                        B_10keV_error_rd,beta_10keV_rd,beta_10keV_error_rd))
        plt.legend(loc=1)
    
    if multiple == "True": # checking if multiple energies are used
        
        popt_2, pcov_2 = curve_fit(quadratic, implantation_energies, B_energies, 
                                   sigma = B_energies_error, absolute_sigma=True) # finding the optimal values for the coefficients
        perr_2=np.sqrt(np.diag(pcov_2)) # and their uncertainties
        
        a=popt_2[0] # assigning the values to the variables
        a_error=perr_2[0]
        b=popt_2[1]
        b_error=perr_2[1]
        c=popt_2[2]
        c_error=perr_2[2]
        
        a_error_rd=round_func(perr_2[0], 1) # rounding the values to be used on the graph
        a_rd=round_sigfig(popt_2[0], a_error_rd)
        b_error_rd=round_func(perr_2[1],1)
        b_rd=round_sigfig(popt_2[1], b_error_rd)
        c_error_rd=round_func(perr_2[2],1)
        c_rd=round_sigfig(popt_2[2],c_error_rd)
        
        x_text_2 = (implantation_energies[2]-implantation_energies[1])/2 +implantation_energies[1]
        y_text_2 = ((B_energies[1]-B_energies[0])/2) + B_energies[0]
        
        new_x = np.linspace(implantation_energies[0],implantation_energies[length-1],100) # obtaining more x values so the curve can be plotted
        
        plt.figure()
        plt.errorbar(implantation_energies, B_energies, yerr=B_energies_error,
                     fmt='o',color="black",capsize=(3),label="Measured B-field") # a plot of the B values for each energy with errors
        plt.plot(new_x, quadratic(new_x,popt_2[0], popt_2[1], popt_2[2]),'-', 
                     lw=1, color="c",label="Fitted B field") # a plot of the fitted field 
        plt.xlabel("Implantation energy (keV)")
        plt.ylabel(r"Magnetic Field $\mu_0 H (T)$")
        plt.title("Field profile with Energy (py21txab)")
        plt.text(x_text_2,y_text_2,r'''   $a={}±{} $T/keV$^2$,
        $b={}±{} $T/keV,
        $c={}±{} $T'''.format(a_rd,a_error_rd,b_rd,b_error_rd,c_rd,c_error_rd))
        plt.legend()
    
    elif multiple == "False":
        print("Mutiple energies paramteter was not included in this data.") # a message to say if multiple energies wasnt used
    else:
        print("""Multiple energies parameter is unspecified and it is not known if it 
              needs to be calculated for. Check the metadata if this needs to be included""")
    
    # these are checks to see wether the returned values are sensible and within the expected range when including 3 times their uncertainty.
    if B_10keV > (0.03+3*B_10keV_error):
        print("""Value for 10keV_B was detected to be above the expected range.
              This may mean the result is unreliable and the data should be checked.""") 
    if beta_10keV > (1.5+3*beta_10keV_error):
        print("""Value for 10keV_beta was detected to be above the expected range.
              this may mean the result is unreliable and the data should be checked.""")
    if beta_10keV < (0.5-3*beta_10keV_error):
        print("""Value for 10keV_beta was detected to be below the expected range.
              this may mean the result is unreliable and the data should be checked.""")
    if damping == "True":
        if tau_10keV > ((10e-6)+3*tau_10keV_error):
            print("""Value for 10keV_tau was detected to be above the expected range.
                  this may mean the result is unreliable and the data should be checked.""")
        if tau_10keV < ((2e-6)-3*tau_10keV_error):
            print("""Value for 10keV_tau was detected to be above the expected range.
                  this may mean the result is unreliable and the data should be checked.""")
    
     
    results={"10keV_B": B_10keV, #this would be the magnetic field for 10keV data (T)
             "10keV_B_error": B_10keV_error, # the error in the magnetic field (T)
             "beta": beta_10keV, #Detector angle in radians
             "beta_error": beta_10keV_error, #uncertainity in detector angle (rad)
             "10keV_tau_damp": tau_10keV, #Damping time for 10keV (s)
             "10keV_tau_damp_error": tau_10keV_error, #and error (s)
             "B(Energy)_coeffs":(a,b,c),#(a,b,c), #tuple of a,b,c for quadratic,linear and constant terms
                                                 #for fitting B dependence on energy
                                                   #(T/keV^2,T/keV,T)
             "B(Energy)_coeffs_errors":(a_error,b_error,c_error)#(a_error,b_error,c_error), # Errors in above in same order.
             }
    
    return results

if __name__=="__main__":
    # Put your test code in side this if statement to stop it being run when you import your code
    #Please avoid using raw_input as the testing is going to be done by a computer programme, so
    #can't input things from a keyboard....
    filename="experimental_signal_data.dat"
    test_results=ProcessData(filename)
    print(test_results)
