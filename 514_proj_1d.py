import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.optimize import fsolve



# system initialization
num_of_ball = 10
######## initial the rest items matrix
R = np.arange(num_of_ball) + 1
R[0] = R[0] - 1
R[-1] = R[-1] + 1
######## initial the transform matrix
M = np.eye(num_of_ball) * -3
M[0][0] += 1
M[-1][-1] += 1
for i in range(num_of_ball):
    if i == 0:
        M[i][i+1] += 1
    elif i == num_of_ball - 1:
        M[i][i-1] += 1
    else:
        M[i][i+1] += 1
        M[i][i-1] += 1
# M *= 3
# R *= 3
#print('R:', R, 'M:', M)

def system_with_electric_field(t, now_loca, now_speed, test):
    slope_loca = now_speed
    slope_speed = np.matmul(M, now_loca) + R + 0.25*np.sin(t)
    slope_test = test
    return slope_loca, slope_speed, slope_test

# the RK-4 loop
def rk4(t, location, speed, test, step_num):
    for j in np.arange(0, step_num):
        k1_loca, k1_speed, k1_test = system_with_electric_field(t[j], location[j], speed[j], test[j])
        k2_loca, k2_speed, k2_test = system_with_electric_field(t[j]+h/2, location[j]+h/2*k1_loca, speed[j]+h/2*k1_speed, test[j]+h/2*k1_test)
        k3_loca, k3_speed, k3_test = system_with_electric_field(t[j]+h/2, location[j]+h/2*k2_loca, speed[j]+h/2*k2_speed, test[j]+h/2*k2_test)
        k4_loca, k4_speed, k4_test = system_with_electric_field(t[j]+h, location[j]+h*k3_loca, speed[j]+h*k3_speed, test[j]+h*k3_test)    
        
        # calculate the final slope
        k_loca = (k1_loca + 2*k2_loca + 2*k3_loca + k4_loca)/6
        k_speed = (k1_speed + 2*k2_speed + 2*k3_speed + k4_speed)/6
        k_test = (k1_test + 2*k2_test + 2*k3_test + k4_test)/6
        # update the value
        location[j+1] = location[j] + h*k_loca
        speed[j+1] = speed[j] + h*k_speed
        test[j+1] = test[j] + h*k_test
        ### check the speed and the location of each ball and the neighbor
        ### if x_i > x_{i+1} and v_i > 0 and v_{i+1} < 0 then we switch the speed immediately
        # for i in range(num_of_ball - 1):
        #     # two balls hit in the reverse direction
        #     if location[j+1][i] > location[j+1][i+1] and speed[j+1][i] > 0 and speed[j+1][i+1] < 0:
        #         speed[j+1][i] *= -1
        #         speed[j+1][i+1] *= -1
        #         continue
        #     # two balls hit in the same direction
        #     if location[j+1][i] > location[j+1][i+1] and ((speed[j+1][i] > 0 and speed[j+1][i+1] > 0) or (speed[j+1][i] < 0 and speed[j+1][i+1] < 0)):
        #         speed[j+1][i+1], speed[j+1][i] = speed[j+1][i], speed[j+1][i+1]
        #         continue
    return t, location, speed, test

def forward_euler(t, location, speed, test, step_num):
    for j in np.arange(0, step_num):
        slope_loca, slope_speed, slope_test = system_with_electric_field(t[j], location[j], speed[j], test[j]) 
        
        # update the value
        location[j+1] = location[j] + h*slope_loca
        speed[j+1] = speed[j] + h*slope_speed
        test[j+1] = test[j] + h*slope_test
        ### check the speed and the location of each ball and the neighbor
        ### if x_i > x_{i+1} and v_i > 0 and v_{i+1} < 0 then we switch the speed immediately
        # for i in range(num_of_ball - 1):
        #     # two balls hit in the reverse direction
        #     if location[j+1][i] > location[j+1][i+1] and speed[j+1][i] > 0 and speed[j+1][i+1] < 0:
        #         speed[j+1][i] *= -1
        #         speed[j+1][i+1] *= -1
        #         continue
        #     # two balls hit in the same direction
        #     if location[j+1][i] > location[j+1][i+1] and ((speed[j+1][i] > 0 and speed[j+1][i+1] > 0) or (speed[j+1][i] < 0 and speed[j+1][i+1] < 0)):
        #         speed[j+1][i+1], speed[j+1][i] = speed[j+1][i], speed[j+1][i+1]
        #         continue
    return t, location, speed, test

def ab2(t, location, speed, test, step_num):
    # using forward euler calculate the first step
    slope_loca, slope_speed, slope_test = system_with_electric_field(t[0], location[0], speed[0], test[0]) 
    location[1] = location[0] + h*slope_loca
    speed[1] = speed[0] + h*slope_speed
    test[1] = test[0] + h*slope_test    
    ### check the speed and the location of each ball and the neighbor
    ### if x_i > x_{i+1} and v_i > 0 and v_{i+1} < 0 then we switch the speed immediately
    # for i in range(num_of_ball - 1):
    #     # two balls hit in the reverse direction
    #     if location[1][i] > location[1][i+1] and speed[1][i] > 0 and speed[1][i+1] < 0:
    #         speed[1][i] *= -1
    #         speed[1][i+1] *= -1
    #         continue
    #     # two balls hit in the same direction
    #     if location[1][i] > location[1][i+1] and ((speed[1][i] > 0 and speed[1][i+1] > 0) or (speed[1][i] < 0 and speed[1][i+1] < 0)):
    #         speed[1][i+1], speed[1][i] = speed[1][i], speed[1][i+1]
    #         continue
        
    # save the old slope
    old_slope_loca, old_slope_speed, old_slope_test = slope_loca, slope_speed, slope_test
    
    for j in np.arange(1, step_num):
        slope_loca, slope_speed, slope_test = system_with_electric_field(t[j], location[j], speed[j], test[j]) 
        # calculate the final slope 
        final_slope_loca = (3/2) * slope_loca - (1/2) * old_slope_loca
        final_slope_speed = (3/2) * slope_speed - (1/2) * old_slope_speed
        final_slope_test = (3/2) * slope_test - (1/2) * old_slope_test
        
        # update the value
        location[j+1] = location[j] + h*final_slope_loca
        speed[j+1] = speed[j] + h*final_slope_speed
        test[j+1] = test[j] + h*final_slope_test
        ### check the speed and the location of each ball and the neighbor
        ### if x_i > x_{i+1} and v_i > 0 and v_{i+1} < 0 then we switch the speed immediately
        # for i in range(num_of_ball - 1):
        #     # two balls hit in the reverse direction
        #     if location[j+1][i] > location[j+1][i+1] and speed[j+1][i] > 0 and speed[j+1][i+1] < 0:
        #         speed[j+1][i] *= -1
        #         speed[j+1][i+1] *= -1
        #         continue
        #     # two balls hit in the same direction
        #     if location[j+1][i] > location[j+1][i+1] and ((speed[j+1][i] > 0 and speed[j+1][i+1] > 0) or (speed[j+1][i] < 0 and speed[j+1][i+1] < 0)):
        #         speed[j+1][i+1], speed[j+1][i] = speed[j+1][i], speed[j+1][i+1]
        #         continue

        old_slope_loca, old_slope_speed, old_slope_test = slope_loca, slope_speed, slope_test   
            
    return t, location, speed, test

def backward_euler_system_with_electric_field(t_j_plus_1, old_loca, old_speed, old_test, num_of_ball, h):
    def innerfun(b):     
        new_loca = b[:num_of_ball]
        new_speed = b[num_of_ball:num_of_ball*2]
        new_test = b[-1]
    
        slope_loca = new_speed
        slope_speed = np.matmul(M, new_loca) + R + 0.25*np.sin(t_j_plus_1)
        slope_test = new_test
        
        # new_loca - old_loca - h* slope_loca = 0
        # new_speed - old_speed - h * slope_speed = 0
        return [(n_l - o_l - h*s_l) for n_l, o_l, s_l in zip(new_loca, old_loca, slope_loca)] + \
                [(n_s - o_s - h*s_s) for n_s, o_s, s_s in zip(new_speed, old_speed, slope_speed)] + \
                [new_test - old_test - h*slope_test]
    return innerfun

def backward_euler(t, location, speed, test, step_num):
    for j in np.arange(0, step_num):
        my_test_fun = backward_euler_system_with_electric_field(t[j+1], location[j], speed[j], test[j], num_of_ball, h)

        temp = fsolve(my_test_fun, np.concatenate([location[j+1], speed[j+1], np.expand_dims(test, axis=-1)[j+1]], axis=0))   
        
        location[j+1] = temp[:num_of_ball]
        speed[j+1] = temp[num_of_ball:num_of_ball*2]
        test[j+1] = test[-1]
        
        ### check the speed and the location of each ball and the neighbor
        ### if x_i > x_{i+1} and v_i > 0 and v_{i+1} < 0 then we switch the speed immediately
        # for i in range(num_of_ball - 1):
        #     # two balls hit in the reverse direction
        #     if location[j+1][i] > location[j+1][i+1] and speed[j+1][i] > 0 and speed[j+1][i+1] < 0:
        #         speed[j+1][i] *= -1
        #         speed[j+1][i+1] *= -1
        #         continue
        #     # two balls hit in the same direction
        #     if location[j+1][i] > location[j+1][i+1] and ((speed[j+1][i] > 0 and speed[j+1][i+1] > 0) or (speed[j+1][i] < 0 and speed[j+1][i+1] < 0)):
        #         speed[j+1][i+1], speed[j+1][i] = speed[j+1][i], speed[j+1][i+1]
        #         continue
    return t, location, speed, test


def bdf2_system_with_electric_field(t_j_plus_1, loca_j, loca_j_minus_1, speed_j, speed_j_minus_1, test_j, test_j_minus_1, num_of_ball, h):
    def innerfun(b):     
        new_loca = b[:num_of_ball]
        new_speed = b[num_of_ball:num_of_ball*2]
        new_test = b[-1]
    
        slope_loca = new_speed
        slope_speed = np.matmul(M, new_loca) + R + 0.25*np.sin(t_j_plus_1)
        slope_test = new_test
        
        # new_loca - old_loca - h* slope_loca = 0
        # new_speed - old_speed - h * slope_speed = 0
        return [(3*n_l - 4*o_l + oo_l - 2*h*s_l) for n_l, o_l, oo_l, s_l in zip(new_loca, loca_j, loca_j_minus_1, slope_loca)] + \
                [(3*n_s - 4*o_s + oo_s - 2*h*s_s) for n_s, o_s, oo_s, s_s in zip(new_speed, speed_j, speed_j_minus_1, slope_speed)] + \
                [3*new_test - 4*test_j + test_j_minus_1 - 2*h*slope_test]
    return innerfun

def bdf2(t, location, speed, test, step_num):
    # using forward euler calculate the first step
    slope_loca, slope_speed, slope_test = system_with_electric_field(t[0], location[0], speed[0], test[0]) 
    location[1] = location[0] + h*slope_loca
    speed[1] = speed[0] + h*slope_speed
    test[1] = test[0] + h*slope_test    
    ### check the speed and the location of each ball and the neighbor
    ### if x_i > x_{i+1} and v_i > 0 and v_{i+1} < 0 then we switch the speed immediately
    # for i in range(num_of_ball - 1):
    #     # two balls hit in the reverse direction
    #     if location[1][i] > location[1][i+1] and speed[1][i] > 0 and speed[1][i+1] < 0:
    #         speed[1][i] *= -1
    #         speed[1][i+1] *= -1
    #         continue
    #     # two balls hit in the same direction
    #     if location[1][i] > location[1][i+1] and ((speed[1][i] > 0 and speed[1][i+1] > 0) or (speed[1][i] < 0 and speed[1][i+1] < 0)):
    #         speed[1][i+1], speed[1][i] = speed[1][i], speed[1][i+1]
    #         continue
    
    for j in np.arange(1, step_num):
        my_test_fun = bdf2_system_with_electric_field(t[j+1], location[j], location[j-1], speed[j], speed[j-1], test[j], test[j-1], num_of_ball, h)

        temp = fsolve(my_test_fun, np.concatenate([location[j+1], speed[j+1], np.expand_dims(test, axis=-1)[j+1]], axis=0))   
        
        location[j+1] = temp[:num_of_ball]
        speed[j+1] = temp[num_of_ball:num_of_ball*2]
        test[j+1] = test[-1]
        
        ### check the speed and the location of each ball and the neighbor
        ### if x_i > x_{i+1} and v_i > 0 and v_{i+1} < 0 then we switch the speed immediately
        # for i in range(num_of_ball - 1):
        #     # two balls hit in the reverse direction
        #     if location[j+1][i] > location[j+1][i+1] and speed[j+1][i] > 0 and speed[j+1][i+1] < 0:
        #         speed[j+1][i] *= -1
        #         speed[j+1][i+1] *= -1
        #         continue
        #     # two balls hit in the same direction
        #     if location[j+1][i] > location[j+1][i+1] and ((speed[j+1][i] > 0 and speed[j+1][i+1] > 0) or (speed[j+1][i] < 0 and speed[j+1][i+1] < 0)):
        #         speed[j+1][i+1], speed[j+1][i] = speed[j+1][i], speed[j+1][i+1]
        #         continue
    return t, location, speed, test


############## approximation initialization
# T=10 # final simulation time
# N=500# number of step
# # T=1 # final simulation time
# # N=100# number of step


# #N_list = [25, 50, 100]
# h=T/N

# ######## initial the vector
# location = np.zeros((N+1,num_of_ball))
# speed = np.zeros((N+1,num_of_ball))
# test = np.zeros(N+1)

# ######## initial the value
# test[0] = 1
# t = np.linspace(0,T,N+1)
# location[0] = np.arange(num_of_ball) + 1
# temp_speed = []
# for i in range(num_of_ball):
#     if i % 2 == 0:
#         temp_speed.append(1)
#     else:
#         temp_speed.append(-1)
# speed[0] = np.array(temp_speed)
# #print('location[0]:', location[0], 'speed[0]:', speed[0])

# #t, location, speed, test = backward_euler(t, location, speed, test, N)
# #t, location, speed, test = bdf2(t, location, speed, test, N)
# #t, location, speed, test = rk4(t, location, speed, test, N)
# #t, location, speed, test = forward_euler(t, location, speed, test, N)
# t, location, speed, test = ab2(t, location, speed, test, N)
# print(location[-1,0], speed[-1,0], test[-1])
# time_axis = np.arange(N)


# # Plot the final tajectory
# plt.xlabel('x')
# plt.ylabel('steps')  
# result_lines = []
# result_points = []
# for j in range(num_of_ball):   
#     line, = plt.plot(location[:, j], np.arange(N+1), lw=1)
#     point, = plt.plot(location[:, j], np.arange(N+1), marker='.')
#     result_lines.append(line)
#     result_points.append(point)

# # Add labels and title
# #plt.title('Trajectory of the ball vs the number of step (Backward Euler)')
# #plt.title('Trajectory of the ball vs the number of step (BDF2)')
# #plt.title('Trajectory of the ball vs the number of step (RK4)')
# #plt.title('Trajectory of the ball vs the number of step (Forward Euler)')
# plt.title('Trajectory of the ball vs the number of step (AB2)')

# # Show the plot
# #plt.savefig("1d_with_electric_field_backeuler_final_trajectory.png")
# #plt.savefig("1d_with_electric_field_BDF2_final_trajectory.png")
# #plt.savefig("1d_with_electric_field_RK4_final_trajectory.png")
# #plt.savefig("1d_with_electric_field_forwardeuler_final_trajectory.png")
# plt.savefig("1d_with_electric_field_AB2_final_trajectory.png")

# ########### create a gif for the trajectory
# fig,ax = plt.subplots()
# def animate(i):
#     ax.clear()
#     ax.set_xlim(0, 12)
#     ax.set_ylim(0, N)
#     ax.set_xlabel('x')
#     ax.set_ylabel('steps')
#     result_lines = []
#     result_points = []
#     for j in range(num_of_ball):     
#         line, = ax.plot(location[:i, j], time_axis[:i], lw=1)
#         point, = ax.plot(location[:i, j], time_axis[:i], marker='.')
#         result_lines.append(line)
#         result_points.append(point)
#     return result_lines + result_points
        
# ani = FuncAnimation(fig, animate, interval=40, blit=False, repeat=False, frames=N+1)    
# #ani.save("1d_with_electric_field_trajectory_backeuler.gif", dpi=300, writer=PillowWriter(fps=25))
# #ani.save("1d_with_electric_field_trajectory_bdf2.gif", dpi=300, writer=PillowWriter(fps=25))
# #ani.save("1d_with_electric_field_trajectory_rk4.gif", dpi=300, writer=PillowWriter(fps=25))
# #ani.save("1d_with_electric_field_trajectory_forwardeuler.gif", dpi=300, writer=PillowWriter(fps=25))
# ani.save("1d_with_electric_field_trajectory_ab2.gif", dpi=300, writer=PillowWriter(fps=25))

# ########### create a gif for the ball movement
# fig,ax = plt.subplots()
# def animate(i):
#     ax.clear()
#     ax.set_xlim(0, 12)
#     ax.set_ylim(-1, 1)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')    
#     result_lines = []
#     result_points = []
#     for j in range(num_of_ball):   
#         line, = ax.plot(location[i, j], 0, lw=1)
#         point, = ax.plot(location[i, j], 0, marker='.')
#         result_lines.append(line)
#         result_points.append(point)
#     return result_lines + result_points
        
# ani = FuncAnimation(fig, animate, interval=40, blit=False, repeat=False, frames=N+1)
# #ani.save("1d_with_electric_field_backeuler.gif", dpi=300, writer=PillowWriter(fps=25))
# #ani.save("1d_with_electric_field_bdf2.gif", dpi=300, writer=PillowWriter(fps=25))
# #ani.save("1d_with_electric_field_rk4.gif", dpi=300, writer=PillowWriter(fps=25))
# #ani.save("1d_with_electric_field_forwardeuler.gif", dpi=300, writer=PillowWriter(fps=25))
# ani.save("1d_with_electric_field_ab2.gif", dpi=300, writer=PillowWriter(fps=25))

############# plot the value with the h
T=1 # final simulation time
N=np.arange(10, 510, 10)
#N_list = [25, 50, 100]

ab2_location_list = []
rk4_location_list = []
fe_location_list = []
backward_euler_location_list = []
bdf2_location_list =[]
for step_num in N:
    h=T/step_num
    ######## initial the vector
    location = np.zeros((step_num+1,num_of_ball))
    speed = np.zeros((step_num+1,num_of_ball))
    test = np.zeros(step_num+1)

    ######## initial the value
    test[0] = 1
    t = np.linspace(0,T,step_num+1)
    location[0] = np.arange(num_of_ball) + 1
    temp_speed = []
    for i in range(num_of_ball):
        if i % 2 == 0:
            temp_speed.append(1)
        else:
            temp_speed.append(-1)
    speed[0] = np.array(temp_speed)
    
    
    t, location, speed, test = bdf2(t, location, speed, test, step_num)
    bdf2_location_list.append(location[-1, 0])
    
    ######## re initial the value
    location = np.zeros((step_num+1,num_of_ball))
    speed = np.zeros((step_num+1,num_of_ball))
    test = np.zeros(step_num+1)
    
    test[0] = 1
    t = np.linspace(0,T,step_num+1)
    location[0] = np.arange(num_of_ball) + 1
    temp_speed = []
    for i in range(num_of_ball):
        if i % 2 == 0:
            temp_speed.append(1)
        else:
            temp_speed.append(-1)
    speed[0] = np.array(temp_speed)

    t, location, speed, test = backward_euler(t, location, speed, test, step_num)
    backward_euler_location_list.append(location[-1, 0])
    t, location, speed, test = ab2(t, location, speed, test, step_num)
    ab2_location_list.append(location[-1, 0])
    t, location, speed, test = rk4(t, location, speed, test, step_num)
    rk4_location_list.append(location[-1, 0])
    t, location, speed, test = forward_euler(t, location, speed, test, step_num)
    fe_location_list.append(location[-1, 0])
    
    
    

######Plot the curve
plt.plot(N, fe_location_list, label='Forward Euler')
plt.plot(N, ab2_location_list, label='AB2')
plt.plot(N, rk4_location_list, label='RK4')
plt.plot(N, backward_euler_location_list, label='Backward Euler')
plt.plot(N, bdf2_location_list, label='BDF2')

# Add labels and title
plt.xlabel('Number of step')
plt.ylabel('The location of the first ball')
#plt.title('Location of the ball given by Forward Euler vs the number of step')
#plt.title('Location of the ball given by RK4 vs the number of step')
#plt.title('Location of the ball given by AB2 vs the number of step')
plt.title('Location of the ball vs the number of step')

# Add a legend
plt.legend()
plt.grid()
# Show the plot
plt.show()
