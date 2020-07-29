###############################
# Multiple Configuration file #
###############################

### SPLINE PARAMETERS ###
knotstep = [35,45,55] #give a list of the parameter you want
preselection_file = '../scripts/config/preset_regdiff.txt' #'config/preset_regdiff.txt'

### RUN PARAMETERS #####
# copies
ncopy = 10 #number of copy per pickle
ncopypkls = 15 #number of pickle

# mock
nsim = 10 #number of copy per pickle
nsimpkls = 20 #number of pickle

### MICROLENSING ####
mlknotsteps = [200, 400]# 0 means no microlensing...

### SIGN OF THE GUESS ###
sign = -1 # +1 with the D3CS & PyCS convention, -1 for the opposite.

### FINAL PLOT DISPLAY ###
# samples
display_silver = True # if you want to display the silver sample
display_gold = True # if you want to display the golden sample

# simulations to exclude for each sample
failed_sim = [] # Write here all the sim that failed
remove_silver_sample = [] # will also automatically remove pair when a precision>40% or a true time delay>100d
remove_golden_sample = [11, 13, 18, 28, 30, 33, 34, 35, 46, 47, 49, 
				 51, 54, 57, 73, 80, 86, 87, 88, 89, 90, 92, 93, 
				 102, 105, 106, 107, 108, 116, 118, 120, 124, 135, 137, 139, 143,
				 151, 155, 157, 164, 169, 172, 177, 178, 179, 183, 186, 190, 193, 197]
				 # remove red and yellow with D3CS

# plot
display_delay = True # if you want to display the summary plot 
display_relative_err = True # if you want to display the relative error plot 
display_precision = True # if you want to display the precision histogram 
display_accuracy = True # if you want to display the accuracy histogram 
display_chi2 = True # if you want to display the chi2 histogram 
display_SNR = True # if you want to display the SNR vs precision/accuracy/chi2
display_ML = True # if you want to display the ML vs precision/accuracy/chi2
display_tdc1 = True  # if you want to plot the summary of the metrics of the tdc1
compare_tdc1 = False # if you want to compare your results with the other submissions of the tdc1



