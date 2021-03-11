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
import scipy.integrate as integrate

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

        #self.velocity = np.linspace(0, self.transition_velocity, num=100)
        
    def thrust(self, velocity):
        return 2 * 4.44822 * (1000*(55.60  - 4.60*((velocity * 3.28084)/100) + 0.357*np.power((velocity/100 * 3.28084), 2)))

    def accel(self, velocity):
        return 9.81*((self.thrust(velocity)/self.takeoff_weight - self.ground_cf) - ((self.ground_cd - self.ground_cf*self.ground_cl)/(self.takeoff_weight/self.wing_span)) * 0.5*1.225*np.power(velocity, 2))

    def plot_ground_roll(self, mode='metric'):
        # numerical integration        
        dt = 0.1
        i = 0

        current_acceleration = self.accel(0)
        current_velocity = 0
        current_distance = 0

        self.distance = [current_acceleration]
        self.velocity = [current_velocity]
        self.acceleration = [current_distance]
        self.time = [0]

        while current_velocity < self.transition_velocity:
            next_velocity = current_velocity + current_acceleration*dt
            next_distance = current_distance + current_velocity*dt + 0.5*current_acceleration*np.power(dt,2)
            next_acceleration = self.accel(next_velocity)

            current_acceleration = next_acceleration
            current_velocity = next_velocity
            current_distance = next_distance

            i += 1

            #np.append(self.acceleration, current_acceleration, axis=0)
            #np.append(self.velocity, current_velocity, axis=0)
            #np.append(self.distance, current_distance, axis=0)

            self.acceleration.append(current_acceleration)
            self.velocity.append(current_velocity)
            self.distance.append(current_distance)
            self.time.append(dt*i)

        self.acceleration = np.array(self.acceleration)
        self.velocity = np.array(self.velocity)
        self.distance = np.array(self.distance)
        self.time = np.array(self.time)

        plt.style.use(['science', 'no-latex'])

        # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
        fig, ax1 = plt.subplots(dpi=230, figsize=(7,5))
        color = 'tab:red'
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Distance [ft]', color=color)
        ax1.plot(self.time, self.distance*3.28084, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Velocity [kts]', color=color)
        ax2.plot(self.time, self.velocity*1.94384, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()

        plt.show()



    #def plot_climb_out 
    #def plot_head_wind