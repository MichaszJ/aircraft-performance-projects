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
        self.wind_speed = -15 * 0.514444

        self.ground_cd = 0.0989

        self.d_gamma = 0.0349066 # rad/s

        #self.velocity = np.linspace(0, self.transition_velocity, num=100)
    
    def thrust(self, velocity, engine_loss=False):
        # returns thrust in N
        if engine_loss:
            return 4.44822 * (1000*(55.60  - 4.60*((velocity * 3.28084)/100) + 0.357*np.power((velocity/100 * 3.28084), 2)))
        else:
            return 2 * 4.44822 * (1000*(55.60  - 4.60*((velocity * 3.28084)/100) + 0.357*np.power((velocity/100 * 3.28084), 2)))

    def ground_acceleration(self, velocity, engine_loss=False):
        # returns ground roll acceleration in m/s^2
        if engine_loss:
            return 9.81*((self.thrust(velocity, engine_loss)/self.takeoff_weight - self.ground_cf) - ((1.1*self.ground_cd - self.ground_cf*self.ground_cl)/(self.takeoff_weight/self.wing_span)) * 0.5*1.225*np.power(velocity, 2))

        else:
            return 9.81*((self.thrust(velocity, engine_loss)/self.takeoff_weight - self.ground_cf) - ((self.ground_cd - self.ground_cf*self.ground_cl)/(self.takeoff_weight/self.wing_span)) * 0.5*1.225*np.power(velocity, 2))

    def lift_coef(self, velocity, gamma):
        return ((self.takeoff_weight/9.81) * velocity * self.d_gamma + self.takeoff_weight * np.cos(gamma))/(0.5 * 1.225 * np.power(velocity, 2) * self.wing_area)
    
    def drag_coeff(self, velocity, gamma):
        return 0.0413 + 0.0576*np.power(self.lift_coef(velocity, gamma), 2)

    def drag(self, velocity, gamma):
        return 0.5 * 1.225 * np.power(velocity, 2) * self.wing_area * self.drag_coeff(velocity, gamma)

    def transition_acceleration(self, thrust, drag, gamma):
        return 9.81*((thrust - drag)/self.takeoff_weight - np.sin(gamma))
    
    def no_wind(self):
        # numerical integration        
        dt = 0.1
        i = 0

        current_acceleration = self.ground_acceleration(0)
        current_velocity = 0
        current_distance = 0

        self.gr_distance = [current_distance]
        gr_velocity = [current_velocity]
        gr_acceleration = [current_acceleration]
        time = [0]

        # ground roll
        while current_velocity < self.transition_velocity:
            next_velocity = current_velocity + current_acceleration*dt
            next_distance = current_distance + current_velocity*dt + 0.5*current_acceleration*np.power(dt,2)
            next_acceleration = self.ground_acceleration(next_velocity)

            current_acceleration = next_acceleration
            current_velocity = next_velocity
            current_distance = next_distance

            i += 1

            gr_acceleration.append(current_acceleration)
            gr_velocity.append(current_velocity)
            self.gr_distance.append(current_distance)
            time.append(dt*i)

        #self.acceleration = np.array(self.acceleration)
        gr_velocity = np.array(gr_velocity)
        self.gr_distance = np.array(self.gr_distance)
        #self.time = np.array(self.time)

        plt.style.use(['science', 'no-latex'])

        # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
        fig, ax1 = plt.subplots(dpi=230, figsize=(7,5))
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Distance [ft]')
        line1 = ax1.plot(time, self.gr_distance*3.28084, label='Distance')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Velocity [kts]')
        line2 = ax2.plot(time, gr_velocity*1.94384, linestyle='--', label='Velocity')

        # https://stackoverflow.com/questions/5484922/secondary-axis-with-twinx-how-to-add-to-legend
        lines = line1 + line2
        labs = [line.get_label() for line in lines]
        ax1.legend(lines, labs, loc=0)

        fig.tight_layout()
        plt.show()

        print('Final time: ', np.round(time[-1], decimals=2), ' s')
        print('Final distance: ', np.round(self.gr_distance[-1]*3.28084, decimals=2), ' ft')

        # transition
        current_gamma = 0
        current_acceleration = self.transition_acceleration(self.thrust(current_velocity), self.drag(current_velocity, current_gamma), current_gamma)
        
        # vertical components
        current_v_acceleration = current_velocity*((self.thrust(current_velocity) - self.drag(current_velocity, current_gamma))/self.takeoff_weight - current_acceleration/9.81)
        current_v_velocity = 0
        current_altitude = 0

        tr_altitude = [current_altitude]
        tr_velocity = [current_velocity]
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
            tr_velocity.append(current_velocity)
            self.tr_distance.append(current_distance)
            tr_altitude.append(current_altitude)
            
            i += 1
            time.append(dt*i)

        self.tr_distance = np.array(self.tr_distance)
        tr_altitude = np.array(tr_altitude)
        tr_velocity = np.array(tr_velocity)

        plt.style.use(['science', 'no-latex'])

        fig, ax1 = plt.subplots(dpi=230, figsize=(7,5))
        ax1.set_xlabel('Ground Distance [ft]')
        ax1.set_ylabel('Altitude [ft]')
        line1 = ax1.plot(self.tr_distance*3.28084, tr_altitude*3.28084, label='Altitude')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Velocity [kts]')
        line2 = ax2.plot(self.tr_distance*3.28084, tr_velocity*1.94384, linestyle='--', label='Velocity')

        lines = line1 + line2
        labs = [line.get_label() for line in lines]
        ax1.legend(lines, labs, loc=0)

        fig.tight_layout()
        plt.show()

        print('Final velocity: ', np.round(tr_velocity[-1]*1.94384, decimals=2), ' kts')
        print('Final distance: ', np.round(self.tr_distance[-1]*3.28084, decimals=2), ' ft')

    def wind(self):
        # numerical integration        
        dt = 0.1
        i = 0

        current_acceleration = self.ground_acceleration(0)
        current_velocity = self.wind_speed
        current_distance = 0

        self.gr_distance_wind = [current_distance]
        gr_velocity = [current_velocity]
        gr_acceleration = [current_acceleration]
        time = [0]

        # ground roll
        while current_velocity < self.transition_velocity:
            next_velocity = current_velocity + current_acceleration*dt
            next_distance = current_distance + (current_velocity - self.wind_speed)*dt + 0.5*current_acceleration*np.power(dt,2)
            next_acceleration = self.ground_acceleration(next_velocity)

            current_acceleration = next_acceleration
            current_velocity = next_velocity
            current_distance = next_distance

            i += 1

            gr_acceleration.append(current_acceleration)
            gr_velocity.append(current_velocity)
            self.gr_distance_wind.append(current_distance)
            time.append(dt*i)

        gr_velocity = np.array(gr_velocity)
        self.gr_distance_wind = np.array(self.gr_distance_wind)

        plt.style.use(['science', 'no-latex'])

        fig, ax1 = plt.subplots(dpi=230, figsize=(7,5))
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Distance [ft]')
        line1 = ax1.plot(time, self.gr_distance_wind*3.28084, label='Distance')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Velocity [kts]')
        line2 = ax2.plot(time, gr_velocity*1.94384, linestyle='--', label='Velocity')

        lines = line1 + line2
        labs = [line.get_label() for line in lines]
        ax1.legend(lines, labs, loc=0)

        fig.tight_layout()
        plt.show()

        print('Final time: ', np.round(time[-1], decimals=2), ' s')
        print('Final distance: ', np.round(self.gr_distance_wind[-1]*3.28084, decimals=2), ' ft')

        # transition
        current_gamma = 0
        current_acceleration = self.transition_acceleration(self.thrust(current_velocity), self.drag(current_velocity, current_gamma), current_gamma)
        
        # vertical components
        current_v_acceleration = current_velocity*((self.thrust(current_velocity) - self.drag(current_velocity, current_gamma))/self.takeoff_weight - current_acceleration/9.81)
        current_v_velocity = 0
        current_altitude = 0

        tr_altitude = [current_altitude]
        tr_velocity = [current_velocity]
        self.tr_distance_wind = [current_distance]

        while current_altitude*3.28084 < 35:
            next_gamma = current_gamma + self.d_gamma*dt
            next_velocity = current_velocity + current_acceleration*dt
            next_distance = current_distance + (current_velocity - self.wind_speed)*dt + 0.5*current_acceleration*np.power(dt,2)
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

            tr_velocity.append(current_velocity)
            self.tr_distance_wind.append(current_distance)
            tr_altitude.append(current_altitude)
            
            i += 1
            time.append(dt*i)
            
        self.tr_distance_wind = np.array(self.tr_distance_wind)
        tr_altitude = np.array(tr_altitude)
        tr_velocity = np.array(tr_velocity)

        plt.style.use(['science', 'no-latex'])

        fig, ax1 = plt.subplots(dpi=230, figsize=(7,5))
        ax1.set_xlabel('Ground Distance [ft]')
        ax1.set_ylabel('Altitude [ft]')
        line1 = ax1.plot(self.tr_distance_wind*3.28084, tr_altitude*3.28084, label='Altitude')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Velocity [kts]')
        line2 = ax2.plot(self.tr_distance_wind*3.28084, tr_velocity*1.94384, linestyle='--', label='Velocity')

        lines = line1 + line2
        labs = [line.get_label() for line in lines]
        ax1.legend(lines, labs, loc=0)

        fig.tight_layout()
        plt.show()

        print('Final velocity: ', np.round(tr_velocity[-1]*1.94384, decimals=2), ' kts')
        print('Final distance: ', np.round(self.tr_distance_wind[-1]*3.28084, decimals=2), ' ft')

    def analytical_approx(self):
        # no wind
        force_o = self.thrust(0) - self.ground_cf * self.takeoff_weight

        drag = 0.5 * 1.225 * np.power(self.transition_velocity, 2) * self.wing_area * self.ground_cd
        lift = 0.5 * 1.225 * np.power(self.transition_velocity, 2) * self.wing_area * self.ground_cl

        thrust = self.thrust(self.transition_velocity)

        force_lof = thrust - drag - self.ground_cf * (self.takeoff_weight - lift)

        ground_distance = (self.takeoff_weight / (2*9.81)) * (np.power(self.transition_velocity, 2))/(force_o - force_lof) * np.log((force_o / force_lof))

        return ground_distance

    def engine_loss(self):
        # numerical integration        
        dt = 0.1
        i = 0

        current_acceleration = self.ground_acceleration(0)
        current_velocity = 0
        current_distance = 0

        self.gr_distance_engine = [current_distance]
        gr_velocity = [current_velocity]
        gr_acceleration = [current_acceleration]
        time = [0]

        # ground roll
        while current_velocity < self.transition_velocity:
            next_velocity = current_velocity + current_acceleration*dt
            next_distance = current_distance + current_velocity*dt + 0.5*current_acceleration*np.power(dt,2)

            if current_velocity >= 80 * 0.514444:
                next_acceleration = self.ground_acceleration(next_velocity, engine_loss=True)
            else:
                next_acceleration = self.ground_acceleration(next_velocity)

            current_acceleration = next_acceleration
            current_velocity = next_velocity
            current_distance = next_distance

            i += 1

            gr_acceleration.append(current_acceleration)
            gr_velocity.append(current_velocity)
            self.gr_distance_engine.append(current_distance)
            time.append(dt*i)

        #self.acceleration = np.array(self.acceleration)
        gr_velocity = np.array(gr_velocity)
        self.gr_distance_engine = np.array(self.gr_distance_engine)
        #self.time = np.array(self.time)

        plt.style.use(['science', 'no-latex'])

        # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
        fig, ax1 = plt.subplots(dpi=230, figsize=(7,5))
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Distance [ft]')
        line1 = ax1.plot(time, self.gr_distance_engine*3.28084, label='Distance')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Velocity [kts]')
        line2 = ax2.plot(time, gr_velocity*1.94384, linestyle='--', label='Velocity')

        # https://stackoverflow.com/questions/5484922/secondary-axis-with-twinx-how-to-add-to-legend
        lines = line1 + line2
        labs = [line.get_label() for line in lines]
        ax1.legend(lines, labs, loc=0)

        fig.tight_layout()
        plt.show()

        print('Final time: ', np.round(time[-1], decimals=2), ' s')
        print('Final distance: ', np.round(self.gr_distance_engine[-1]*3.28084, decimals=2), ' ft')