import numpy as np
import pandas as pd


class ArArWorkSheet:
    def __init__(self, R, R_err, sample_name = 'TEST',
        Calibration = 'BayesCal'):
        # Input unknown R
        self.R = R
        self.R_err = R_err
        self.Calibration = Calibration
        self.sample_name = sample_name

        if self.Calibration == 'BayesCal':
            # Bayesian Decay Calibraiton Values
            self.lam_K40 = 5.50422e-10
            self.lam_K40_err = 0.00543e-10
            self.FCs_age = 28.183e6
            self.FCs_age_err = 0.017e6
            self.corr_FCs_lamtot = 0.009157
            
        if self.Calibration == 'Renne2011':
            # Bayesian Decay Calibraiton Values
            self.lam_K40 = 5.5305e-10
            self.lam_K40_err = 0.0109e-10
            self.FCs_age = 28.294e6
            self.FCs_age_err = 0.036e6
            self.corr_FCs_lamtot = None
            
        if self.Calibration =='Min2000':
            self.lam_K40 = 5.463e-10
            self.lam_K40_err = 0.054e-10
            self.FCs_age = 28.201e6
            self.FCs_age_err = 0.023e6
            self.corr_FCs_lamtot = None
            
        if self.Calibration == 'SJ': # Steiger and Jäger (1977)
            self.lam_K40 = 5.543e-10
            self.lam_K40_err = 0.01e-10
            self.FCs_age = 28.02e6
            self.FCs_age_err = 0.16e6
            self.corr_FCs_lamtot = None
        
    def Age_Calculation_wFCS_age(self):

        Bit = (np.exp(self.lam_K40 * self.FCs_age) - 1)
        age =(1/self.lam_K40) * np.log(Bit * self.R + 1)
        
        return age
        

    def Age_Uncertainties_R(self):
        # This function will return internal uncertainties (only analytical)
        # Just uncertainty in R
        Bit = (np.exp(self.lam_K40 * self.FCs_age) - 1)
        A = Bit * self.R + 1
        dage_dR = Bit / (self.lam_K40 * A)
        
        err_r_2 = (dage_dR)**2 * self.R_err**2
        
        return np.sqrt(err_r_2) / 1e6 # Ma
                
        
    def Age_and_R_FCs_Uncertainty(self):
    
        age = self.Age_Calculation_wFCS_age()
        
        # Need this stuff
        Bit = (np.exp(self.lam_K40 * self.FCs_age) - 1)
        A = Bit * self.R + 1
        
        # R
        dage_dR = Bit / (self.lam_K40 * A)
        
        # FCs
        dage_dfcs = (self.R * np.exp(self.lam_K40 * self.FCs_age))/A
        
        
        jac = np.array([dage_dR, dage_dfcs])
        
        cov = np.zeros((2,2))
        
        cov[0,0] = self.R_err ** 2
        cov[1,1] = self.FCs_age_err ** 2
    
        err2 = jac @ cov @ jac.T
            
            
        return np.sqrt(err2) / 1e6 # Ma
        
        
    def Age_and_Total_Uncertainty(self):
    
        age = self.Age_Calculation_wFCS_age()
        
        # Need this stuff
        Bit = (np.exp(self.lam_K40 * self.FCs_age) - 1)
        A = Bit * self.R + 1
        
        # R
        dage_dR = Bit / (self.lam_K40 * A)
        
        # FCs
        dage_dfcs = (self.R * np.exp(self.lam_K40 * self.FCs_age))/A
        
        #Lambda
        dage_dlambda = - np.log(A)/self.lam_K40**2 + (self.R * self.FCs_age * np.exp(self.lam_K40 * self.FCs_age))/(self.lam_K40 * A)
        
        jac = np.array([dage_dR, dage_dfcs, dage_dlambda])
        
        cov = np.zeros((3,3))
        
        if self.corr_FCs_lamtot is not None:
            cov_xy = self.corr_FCs_lamtot * self.FCs_age_err * self.lam_K40_err
        else:
            cov_xy = 0.0
        
        cov[0,0] = self.R_err ** 2
        cov[1,1] = self.FCs_age_err ** 2
        cov[1,2] = cov_xy
        cov[2,1] = cov[1,2]
        cov[2,2] = self.lam_K40_err ** 2
        
        err2 = jac @ cov @ jac.T
                        
        return age/1e6, np.sqrt(err2)/1e6 # In Ma
        

    def Half_Covariance(self):
        # Helper function
        # Need to determine the covariance between the FCs and total decay constant
        # For uncertainty budget will assume that half the covariance goes to FCs and
        # half to lam tot
        
        age = self.Age_Calculation_wFCS_age()
    
        # Need this stuff
        Bit = (np.exp(self.lam_K40 * self.FCs_age) - 1)
        A = Bit * self.R + 1
        

        # FCs
        dage_dfcs = (self.R * np.exp(self.lam_K40 * self.FCs_age))/A
        
        #Lambda
        dage_dlambda = - np.log(A)/self.lam_K40**2 + (self.R * self.FCs_age * np.exp(self.lam_K40 * self.FCs_age))/(self.lam_K40 * A)
        
        cov_term = 2 * dage_dfcs * dage_dlambda * self.corr_FCs_lamtot * self.lam_K40_err * self.FCs_age_err
        
        return cov_term/2

    def Age_Uncertainty_Pie(self, save = False):
        Age = self.Age_Calculation_wFCS_age()
        
        R_err = self.Age_Uncertainties_R()
        FCs_err = self.Age_Uncertainties_FCs()
        Lam40_err = self.Age_Uncertainties_Decay_Constant()
        
        # Compute combined uncertainties
        just_R = R_err
        R_and_FCS = np.sqrt(R_err**2 + FCs_err**2 + self.Half_Covariance())
        R_FCs_and_Lam40 = np.sqrt(R_err**2 + FCs_err**2 + Lam40_err**2 + 2*self.Half_Covariance())
        
        frac_R = just_R**2/R_FCs_and_Lam40**2
        frac_FCs = FCs_err**2/R_FCs_and_Lam40**2
        frac_lam40 =Lam40_err**2/R_FCs_and_Lam40**2
        
        values = np.array([frac_R, frac_FCs, frac_lam40])
        
        
        
        uncertainties_labels = ['R-value',
                            'Neutron\nFluence\nMonitor',
                            '$\lambda_{^{40}K}$']
        
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize = (5,5))
        
        ax.pie(values,
        startangle = 90)
        ax.text(0.05, 0.9, s= f'Sample: {self.sample_name}',
        transform = ax.transAxes, fontweight = 'bold')
        ax.legend(uncertainties_labels, loc = 'lower right')
        if save is True:
            df = pd.DataFrame({"R": frac_R,
                               "FCs": frac_FCs,
                               "Lambda_40": frac_lam40
                              }, index = [0])
            df.to_excel(f'{self.sample_name}_uncertainty_frac.xlsx')
        #plt.savefig(f'/Users/jackcarter/Documents/ArAr_Uncertainty/{sample_name}_pie.png', dpi = 300)
        
    def Age_Uncertainties_R(self):
        # This function will return internal uncertainties (only analytical)
        # Just uncertainty in R
        Bit = (np.exp(self.lam_K40 * self.FCs_age) - 1)
        A = Bit * self.R + 1
        dage_dR = Bit / (self.lam_K40 * A)
        
        err_r_2 = (dage_dR)**2 * self.R_err**2
        
        return np.sqrt(err_r_2)
        
        
        
    def Age_Uncertainties_FCs(self):
        # This function will return internal uncertainties
        # Analytical and FCS
        # This function will return internal uncertainties (only analytical)
        # Just uncertainty in R
        Bit = (np.exp(self.lam_K40 * self.FCs_age) - 1)
        A = Bit * self.R + 1
        dage_dfcs = (self.R * np.exp(self.lam_K40 * self.FCs_age))/A
        
        err_fcs_2 = (dage_dfcs)**2 * self.FCs_age_err**2 + self.Half_Covariance()
        
        return np.sqrt(err_fcs_2)
        
        
    def Age_Uncertainties_Decay_Constant(self):
        # Analytical, FCs, and Decay constant
        Bit = (np.exp(self.lam_K40 * self.FCs_age) - 1)
        A = Bit * self.R + 1
        dage_dlambda = - np.log(A)/self.lam_K40**2 + (self.R * self.FCs_age * np.exp(self.lam_K40 * self.FCs_age))/(self.lam_K40 * A)
        
        err_fcs_2 = (dage_dlambda)**2 * self.lam_K40_err**2 + self.Half_Covariance()
        
        return np.sqrt(err_fcs_2)
        
        
    def Age_Uncertainties_All(self):
    
        Age = self.Age_Calculation_wFCS_age() / 1e6 # Ma
        
        R_err = self.Age_Uncertainties_R()
        FCs_err = self.Age_Uncertainties_FCs()
        Lam40_err = self.Age_Uncertainties_Decay_Constant()
        
        # Compute combined uncertainties
        just_R = R_err  / 1e6
        R_and_FCS = np.sqrt(R_err**2 + FCs_err**2)  / 1e6
        R_FCs_and_Lam40 = np.sqrt(R_err**2 + FCs_err**2 + Lam40_err**2)  / 1e6
         
        # Create the summary string with formatted output
        summary = (f"Age: {Age:.2f} ± {just_R:.2f} (R only) (Ma), "
                   f"± {R_and_FCS:.2f} (R + FCs) (Ma), "
                   f"± {R_FCs_and_Lam40:.2f} (R + FCs + Lam40) (Ma)")
        
        # Print the summary and explanations
        #print(summary)
        print("\nWhere:")
        print("  - The first uncertainty (X) is computed using only the uncertainty in R (just_R).")
        print("  - The second uncertainty (Y) combines the uncertainties of R and FCs (R_and_FCS).")
        print("  - The third uncertainty (Z) combines the uncertainties of R, FCs, and the decay constant (Lam40).")
        
        return summary
        
        
    def Monte_Carlo(self):
        
        n = 30000
        ages = np.zeros(n)
        for i in range(n):
            lam_40K_mc = np.random.normal(self.lam_K40, self.lam_K40_err)
            FCs_mc = np.random.normal(self.FCs_age, self.FCs_age_err)
            R_mc = np.random.normal(self.R, self.R_err)
            
            Bit = (np.exp(lam_40K_mc * FCs_mc) - 1)
            ages[i]  = (1/lam_40K_mc) * np.log(Bit * R_mc + 1)
         
                
        return ages, ages.mean(axis = 0), ages.std(axis = 0)
        
        
    def Monte_Carlo_Vs_Analytical(self):
    
        MC_samples, _, _  = self.Monte_Carlo()
        
        Age, Age_all_err = self.Age_and_Total_Uncertainty()
        
        return MC_samples, Age, Age_all_err
        
        
    def Monte_Carlo_Vs_Analytical_Hist(self):
        from scipy.stats import norm
        import matplotlib.pyplot as plt
    
        MC_samples, _, _  = self.Monte_Carlo()
        
        Age, Age_all_err = self.Age_and_Total_Uncertainty()
        
        low, high = np.percentile(MC_samples/1e6, [0.05, 99.95], axis = 0)
        
        x = np.linspace(low, high, 1000)
        
        pdf = norm(loc = Age/1e6, scale = Age_all_err/1e6).pdf(x)
        
        fig, ax = plt.subplots(1, 1, figsize = (5,5))
        
        ax.plot(x, pdf, label = 'Linear\nUnceratinty\nPropagation',
        color = 'r',
        lw = 2, zorder = 10)
        ax2 = ax.twinx()
        ax2.hist(MC_samples/1e6, label = 'Monte Carlo\nSamples', bins = 25,
        facecolor = 'dodgerblue', edgecolor ='k', alpha = 0.7, lw = 0.7, zorder = 1)
        ax2.set_yticks([])
        ax2.set_ylim(bottom = 0)
        ax.set_ylim(bottom = 0)
        
        # Bring ax to the front by adjusting zorder and make its patch invisible
        ax.set_zorder(10)
        ax.patch.set_visible(False)
        ax2.set_zorder(5)
        
        # Combine the legends from both axes
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='best')
        
        ax.set_ylabel('Relative Density/Probability')
        ax.set_xlabel('Age (Ma)')
            
