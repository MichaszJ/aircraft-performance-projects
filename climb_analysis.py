import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
from skaero.atmosphere import coesa
from labellines import labelLine, labelLines

class climb_analysis:
    def __init__(self, altitudes):
        # initial properties
        self.altitudes = altitudes
        self.weight = 2400 * 4.44822
        self.drag_coeff_0 = 0.0317
        self.aspect_ratio = 5.71
        self.oswald_eff = 0.6
        self.lift_coeff_max = 1.45
        self.brake_hp = 180

        self.max_velocity = 60.7044

        self.density_SL = 1.225
        self.area = 170 * 0.092903

        self.density = [self.density_SL]
        self.density.extend([coesa.table(alt * 0.3048)[3] for alt in self.altitudes])

        self.density = np.array(self.density)
        self.sigmas = np.array([rho / self.density_SL for rho in self.density])

        # finding lift and velocity ranges
        #self.lift_coeff_min = (2 * self.weight) / (self.density_SL * self.area * np.power(self.max_velocity, 2))
        
        self.lift_coeff_min = 0.2
        self.lift_coeff = np.linspace(self.lift_coeff_min, self.lift_coeff_max, 100)

        #self.stall_velocity = np.sqrt((2 * self.weight) / (self.density_SL * self.area * self.lift_coeff_max))
        #self.velocity = np.linspace(self.stall_velocity, self.max_velocity, 100)

        #self.velocity = np.array([self.velocity * np.sqrt(sigma) for sigma in self.sigmas])
        
        self.velocity = np.array([np.sqrt((2 * self.weight) / (rho * self.area * self.lift_coeff)) for rho in self.density])

        # finding drag
        self.drag_coeff = np.array([self.drag_coeff_0 + np.power(cl, 2) / (np.pi * self.aspect_ratio * self.oswald_eff) for cl in self.lift_coeff])

        self.drag = np.array([
            [0.5 * rho * np.power(self.velocity[i][j], 2) * self.area * self.drag_coeff[j] for j in range(len(self.velocity[i]))] for i, rho in enumerate(self.density)
        ])

    def max_roc(self, metric):
        # finding power required
        self.power_required = self.drag * self.velocity
        
        # finding propeller efficiency
        self.adv_ratio = np.linspace(0.25, 0.95, 15)
        self.prop_eff = np.array([0.45, 0.53, 0.58, 0.63, 0.67, 0.70, 0.73, 0.76, 0.78, 0.80, 0.81, 0.81, 0.80, 0.76, 0.69])
        self.prop_eff_interp_4 = interpolate.UnivariateSpline(self.adv_ratio, self.prop_eff, k=4)

        # finding power available
        self.prop_rot = 2700 * 0.016667
        self.prop_diameter = 73 * 0.0254
        self.advance_ratios = self.velocity / (self.prop_rot * self.prop_diameter)

        self.power_available = np.array([
            [self.prop_eff_interp_4(j) * (self.brake_hp * 745.7) * np.power(self.sigmas[i], 0.7) for j in self.advance_ratios[i]] for i in range(len(self.sigmas))
        ])

        # find rate of climb
        self.climb_rate = (self.power_available - self.power_required) / self.weight

        # find maximum rate of climb
        self.climb_rate_max = np.array([np.max(alt) for alt in self.climb_rate])
        self.climb_rate_max_x = np.array([self.velocity[i][np.argmax(self.climb_rate[i])] for i in range(len(self.velocity))])

        if metric:
            plt.figure(dpi=230, figsize=(7,4))
            plt.style.use(['science', 'no-latex'])

            self.climb_rate_max_interp = interpolate.UnivariateSpline(self.climb_rate_max_x, self.climb_rate_max)
            self.interp_x = np.linspace(np.min(self.climb_rate_max_x), np.max(self.climb_rate_max_x))

            plt.plot(self.interp_x, self.climb_rate_max_interp(self.interp_x), linestyle='--')          

            for i in range(len(self.altitudes) + 1):
                if i == 0:
                    plt.scatter(self.climb_rate_max_x[i], self.climb_rate_max[i], label='Sea Level')
                else:
                    plt.scatter(self.climb_rate_max_x[i], self.climb_rate_max[i], label='{0} ft'.format(self.altitudes[i-1]))

            plt.xlabel('Equivalent Velocity [m/s]')
            plt.ylabel('Maximum Rate of Climb [m/s]')
            plt.legend()

            plt.show()

            print('Spline coefficients: ', self.climb_rate_max_interp.get_coeffs())

        else:
            plt.figure(dpi=230, figsize=(7,4))
            plt.style.use(['science', 'no-latex'])

            self.climb_rate_max_interp = interpolate.UnivariateSpline(self.climb_rate_max_x*1.94384, self.climb_rate_max*1.94384)
            self.interp_x = np.linspace(np.min(self.climb_rate_max_x*1.94384), np.max(self.climb_rate_max_x*1.94384))

            plt.plot(self.interp_x, self.climb_rate_max_interp(self.interp_x), linestyle='--')

            for i in range(len(self.altitudes) + 1):
                if i == 0:
                    plt.scatter(self.climb_rate_max_x[i]*1.94384, self.climb_rate_max[i]*1.94384, label='Sea Level')
                else:
                    plt.scatter(self.climb_rate_max_x[i]*1.94384, self.climb_rate_max[i]*1.94384, label='{0} ft'.format(self.altitudes[i-1]))

            plt.xlabel('Equivalent Velocity [kts]')
            plt.ylabel('Maximum Rate of Climb [kts]')
            plt.legend()

            plt.show()

            print('Spline coefficients: ', self.climb_rate_max_interp.get_coeffs())
        
    def hodograph(self, metric):
        # find horizontal velocity
        self.gamma = self.climb_rate / self.velocity
        self.velocity_hor = self.velocity * np.cos(self.gamma)

        # get only positive rate of climb Values
        self.climb_rate_pos = [[climb_rate for climb_rate in alt if climb_rate > 0] for alt in self.climb_rate]
        self.velocity_hor_pos = [self.velocity_hor[i][0:len(self.climb_rate_pos[i])] for i in range(len(self.climb_rate_pos))]

        if metric:
            plt.figure(dpi=230, figsize=(7,4))
            plt.style.use(['science', 'no-latex'])

            for i in range(len(self.altitudes) + 1):
                if i == 0:
                    plt.plot(self.velocity_hor_pos[i], self.climb_rate_pos[i], label='Sea Level')
                else:
                    plt.plot(self.velocity_hor_pos[i], self.climb_rate_pos[i], label='{0} ft'.format(self.altitudes[i-1]))

            plt.xlabel('Equivalent Horizontal Velocity [m/s]')
            plt.ylabel('Maximum Rate of Climb [m/s]')
            plt.legend()

            plt.show()

        else:
            plt.figure(dpi=230, figsize=(7,4))
            plt.style.use(['science', 'no-latex'])

            for i in range(len(self.altitudes) + 1):
                if i == 0:
                    plt.plot(np.array(self.velocity_hor_pos[i])*1.94384, np.array(self.climb_rate_pos[i])*1.94384, label='Sea Level')
                else:
                    plt.plot(np.array(self.velocity_hor_pos[i])*1.94384, np.array(self.climb_rate_pos[i])*1.94384, label='{0} ft'.format(self.altitudes[i-1]))

            plt.xlabel('Horizontal Horizontal Velocity [kts]')
            plt.ylabel('Rate of Climb [kts]')
            plt.legend()

            plt.show()