# Develop a program to predict the ground-roll distance, velocity, and acceleration as a function of time. 
# Input the parametergiven for a Boeing 767ER and check your results against the appropriate figures that are 
# provided.Please use a numerical integration in time as discussed in classand tutorials.The airspeed for the 
# beginning of transition and lift off is 160 kts.

# Based on your model develop the following:
# i. Ground roll distance and speed with no wind over time in comparison to provided data).

# ii. Speed and altitude versus ground distance during initial climb out until clearing.

# iii. Same as i. and ii., but for 15 kts head wind.

# iv. How do the distances of i. –ii. compare to the results attained using the analytical approximation
#   shown in class? 

# v. Same as i. and ii., but with one engine made inoperable at an airspeed of 80kts. Provide the plot for 
#   distance and airspeed versus time until it reaches the liftoff speed of 160kts. Assume a 10% larger zero-lift 
#   drag coefficient due to the additional drag of the inoperable engine and the need to trim the aircraft for 
#   the asymmetrical thrust condition. Assume zero wind.

# vi. Determine the balanced field length if with one engine made inoperable at an airspeed of 80kts. After engine 
#   failure, wait for 3 seconds before initiating braking. Plot ground roll distance and speed over time. Assume 
#   zero wind. The runway is hardtop

# numerical integration
#   1. Compute a(t=0), V(t=0) = V_wind, S(t=0) = 0
#   2. Compute V(t + dt), S(t + dt)
#   3. Use V(t + dt) to compute thrust, dynamic pressure
#   4. Compute a(t + dt)

import numpy as np
import matplotlib.pyplot as plt

class ground_roll:
    def __init__(self):
        # in metric units
        self.takeoff_weight = 387000 * 4.44822
        self.wing_area = 3084 * 0.092903
        self.wing_span = 156 * 0.3048
        self.ground_cl = 1.0
        self.cd_ratio = 0.4
        self.transition_velocity = 160 * 0.514444
        self.ground_cf = 0.025

        self.ground_cd = 0.0989

        self.d_gamma = 0.0349066 # rad/s

        #self.velocity = np.linspace(0, self.transition_velocity, num=100)
    
    def thrust(self, velocity):
        # returns thrust in N
        return 2 * 4.44822 * (1000*(55.60  - 4.60*((velocity * 3.28084)/100) + 0.357*np.power((velocity/100 * 3.28084), 2)))

    def ground_acceleration(self, velocity):
        # returns ground roll acceleration in m/s^2
        return 9.81*((self.thrust(velocity)/self.takeoff_weight - self.ground_cf) - ((self.ground_cd - self.ground_cf*self.ground_cl)/(self.takeoff_weight/self.wing_span)) * 0.5*1.225*np.power(velocity, 2))

    def lift_coef(self, velocity, gamma):
        return ((self.takeoff_weight/9.81) * velocity * self.d_gamma + self.takeoff_weight * np.cos(gamma))/(0.5 * 1.225 * np.power(velocity, 2) * self.wing_area)
    
    def drag_coeff(self, velocity, gamma):
        return 0.0413 + 0.0576*np.power(self.lift_coef(velocity, gamma), 2)

    def drag(self, velocity, gamma):
        return 0.5 * 1.225 * np.power(velocity, 2) * self.wing_area * self.drag_coeff(velocity, gamma)

    def transition_acceleration(self, thrust, drag, gamma):
        return 9.81*((thrust - drag)/self.takeoff_weight - np.sin(gamma))
    
    def no_wind(self, mode='metric'):
        # numerical integration        
        dt = 0.1
        i = 0

        current_acceleration = self.ground_acceleration(0)
        current_velocity = 0
        current_distance = 0

        self.gr_distance = [current_acceleration]
        self.gr_velocity = [current_velocity]
        self.gr_acceleration = [current_distance]
        self.time = [0]

        # ground roll
        while current_velocity < self.transition_velocity:
            next_velocity = current_velocity + current_acceleration*dt
            next_distance = current_distance + current_velocity*dt + 0.5*current_acceleration*np.power(dt,2)
            next_acceleration = self.ground_acceleration(next_velocity)

            current_acceleration = next_acceleration
            current_velocity = next_velocity
            current_distance = next_distance

            i += 1

            self.gr_acceleration.append(current_acceleration)
            self.gr_velocity.append(current_velocity)
            self.gr_distance.append(current_distance)
            self.time.append(dt*i)

        #self.acceleration = np.array(self.acceleration)
        self.gr_velocity = np.array(self.gr_velocity)
        self.gr_distance = np.array(self.gr_distance)
        #self.time = np.array(self.time)

        plt.style.use(['science', 'no-latex'])

        # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
        fig, ax1 = plt.subplots(dpi=230, figsize=(7,5))
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Distance [ft]')
        line1 = ax1.plot(self.time, self.gr_distance*3.28084, label='Distance')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Velocity [kts]')
        line2 = ax2.plot(self.time, self.gr_velocity*1.94384, linestyle='--', label='Velocity')

        # https://stackoverflow.com/questions/5484922/secondary-axis-with-twinx-how-to-add-to-legend
        lines = line1 + line2
        labs = [line.get_label() for line in lines]
        ax1.legend(lines, labs, loc=0)

        fig.tight_layout()
        plt.show()

        # transition
        current_gamma = 0
        current_acceleration = self.transition_acceleration(self.thrust(current_velocity), self.drag(current_velocity, current_gamma), current_gamma)
        
        # vertical components
        current_v_acceleration = current_velocity*((self.thrust(current_velocity) - self.drag(current_velocity, current_gamma))/self.takeoff_weight - current_acceleration/9.81)
        current_v_velocity = 0
        current_altitude = 0

        self.tr_altitude = [current_altitude]
        self.tr_velocity = [current_velocity]
        #self.tr_acceleration = [current_acceleration]
        self.tr_distance = [current_distance]

        while current_altitude*3.28084 < 35:
            next_gamma = current_gamma + self.d_gamma*dt
            next_velocity = current_velocity + current_acceleration*dt
            next_distance = current_distance + current_velocity*dt + 0.5*current_acceleration*np.power(dt,2)
            next_acceleration = self.transition_acceleration(self.thrust(next_velocity), self.drag(next_velocity, next_gamma), next_gamma)
            
            next_v_velocity = current_v_velocity + current_v_acceleration*dt
            next_altitude = current_altitude + current_v_velocity*dt + 0.5*current_v_acceleration*np.power(dt,2)
            next_v_acceleration = current_velocity*((self.thrust(next_velocity) - self.drag(next_velocity, next_gamma))/self.takeoff_weight - next_acceleration/9.81)

            current_gamma = next_gamma
            current_velocity = next_velocity
            current_distance = next_distance
            current_acceleration = next_acceleration

            current_v_velocity = next_v_velocity
            current_altitude = next_altitude
            current_v_acceleration = next_v_acceleration

            #self.tr_acceleration.append(current_acceleration)
            self.tr_velocity.append(current_velocity)
            self.tr_distance.append(current_distance)
            self.tr_altitude.append(current_altitude)
            
            i += 1
            self.time.append(dt*i)

        self.tr_distance = np.array(self.tr_distance)
        self.tr_altitude = np.array(self.tr_altitude)
        self.tr_velocity = np.array(self.tr_velocity)

        plt.style.use(['science', 'no-latex'])

        fig, ax1 = plt.subplots(dpi=230, figsize=(7,5))
        ax1.set_xlabel('Ground Distance [ft]')
        ax1.set_ylabel('Altitude [ft]')
        line1 = ax1.plot(self.tr_distance*3.28084, self.tr_altitude*3.28084, label='Altitude')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Velocity [kts]')
        line2 = ax2.plot(self.tr_distance*3.28084, self.tr_velocity*1.94384, linestyle='--', label='Velocity')

        lines = line1 + line2
        labs = [line.get_label() for line in lines]
        ax1.legend(lines, labs, loc=0)

        fig.tight_layout()
        plt.show()