import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial.distance import cdist
from scipy.optimize import fsolve, newton


#hyper params of the system
num_of_ball = 5
spring_origin_l = 0
spring_origin_k = 10
spring_neighbor_l = 1
spring_neighbor_k = 1
ball_mass = 1
q=10**(-5)
electicK = 9 * (10**9)

def find_all_neighbor(x, y):
    spring_neighbors = []
    addition_electric_neighbors = []
    # corner case 1
    if x == 0 and y == 0:
        spring_neighbors = [[0, 1], [1,0]]
        addition_electric_neighbors = [[1, 1]]
        return spring_neighbors, addition_electric_neighbors
    # corner case 2
    elif x == 0 and y == num_of_ball - 1:
        spring_neighbors = [[0, y-1], [1, y]]
        addition_electric_neighbors = [[1, y-1]]
        return spring_neighbors, addition_electric_neighbors
    # corner case 3
    elif x == num_of_ball - 1 and y == 0:
        spring_neighbors = [[x, 1], [x-1, 0]]
        addition_electric_neighbors = [[x-1, 1]]
        return spring_neighbors, addition_electric_neighbors
    # corner case 4
    elif x == num_of_ball - 1 and x == num_of_ball - 1:
        spring_neighbors = [[x-1, y], [x, y-1]]
        addition_electric_neighbors = [[x-1, y-1]]
        return spring_neighbors, addition_electric_neighbors
    # edge case 1
    elif x == 0 and y != 0:
        spring_neighbors = [[x, y-1], [x, y+1], [x+1, y]]
        addition_electric_neighbors = [[x+1, y+1],[x+1, y-1]]
        return spring_neighbors, addition_electric_neighbors
    # edge case 2
    elif x == num_of_ball - 1 and y != 0:
        spring_neighbors = [[x, y-1], [x, y+1], [x-1, y]]
        addition_electric_neighbors = [[x-1, y+1], [x-1, y-1]]
        return spring_neighbors, addition_electric_neighbors
    # edge case 3 
    elif x != 0 and y == 0:
        spring_neighbors = [[x-1,0], [x+1, 0], [x, 1]]
        addition_electric_neighbors = [[x-1, 1], [x+1, 1]]
        return spring_neighbors, addition_electric_neighbors
    # edge case 4
    elif x != 0 and y == num_of_ball - 1:
        spring_neighbors = [[x-1, y], [x+1, y], [x, y-1]]
        addition_electric_neighbors = [[x-1, y-1], [x+1, y-1]]
        return spring_neighbors, addition_electric_neighbors
    else:
        spring_neighbors = [[x-1, y], [x+1, y], [x, y-1], [x, y+1]]
        addition_electric_neighbors = [[x+1, y+1], [x+1, y-1], [x-1, y+1], [x-1, y-1]]
        return spring_neighbors, addition_electric_neighbors


def system_with_electric_field(t, now_loca_x, now_loca_y, now_speed_x, now_speed_y, test, 
                                  now_loca_x_ref=None, now_loca_y_ref=None, now_speed_x_ref=None, now_speed_y_ref=None):
    #print(t, now_loca_x, now_loca_y)
    slope_loca_x = now_speed_x
    slope_loca_y = now_speed_y
    slope_test = test
    if now_loca_x_ref is not None:
        slope_loca_x_ref = now_speed_x_ref
        slope_loca_y_ref = now_speed_y_ref
    
    # initial the value
    slope_speed_x = np.zeros([num_of_ball, num_of_ball])
    slope_speed_y = np.zeros([num_of_ball, num_of_ball])
    
    # calculate the force at each direction
    for i in range(num_of_ball):
        for j in range(num_of_ball):
            # find all neighbor
            spring_neighbors_idx, addition_electric_neighbors_idx = find_all_neighbor(i, j)
            # get the location base on the index
            currect_loca = np.array([[now_loca_x[i,j], now_loca_y[i,j]]])
            spring_neighbor_loca = np.array([[now_loca_x[tuple(idx_pair)], now_loca_y[tuple(idx_pair)]] for idx_pair in spring_neighbors_idx])
            addition_electric_neighbors_loca = np.array([[now_loca_x[tuple(idx_pair)], now_loca_y[tuple(idx_pair)]] for idx_pair in addition_electric_neighbors_idx])
            current_origin = np.array([[origin_x[i,j], origin_y[i,j]]])
            #print('currect_loca:', currect_loca, 'current_origin:', current_origin)
            
            # calculate the distance to each neighbor and the origin
            distance_to_origin = cdist(currect_loca, current_origin, 'euclidean')
            distance_to_spring_neighbors = cdist(currect_loca, spring_neighbor_loca, 'euclidean')
            distance_to_addition_electric_neighbors = cdist(currect_loca, addition_electric_neighbors_loca, 'euclidean')
            
            # print(i, j, ' distance_to_origin:', distance_to_origin, 'distance_to_spring_neighbors:', 
            #       distance_to_spring_neighbors, 'distance_to_addition_electric_neighbors:', distance_to_addition_electric_neighbors)
            
            # calculate the force spring from origin
            force_spring_origin = -(spring_origin_k / ball_mass) * (distance_to_origin[0][0] - spring_origin_l)
            if distance_to_origin[0][0] == 0:
                force_spring_origin_x = 0
                force_spring_origin_y = 0
            else:
                force_spring_origin_x = force_spring_origin * (currect_loca - current_origin)[0][0] / distance_to_origin[0][0]
                force_spring_origin_y = force_spring_origin * (currect_loca - current_origin)[0][1] / distance_to_origin[0][0]
            #print(currect_loca, current_origin, '(currect_loca - current_origin):', (currect_loca - current_origin), (currect_loca - current_origin)[0][0], (currect_loca - current_origin)[0][1])

            
            # calculate the force from_spring_neighbors (both spring and electric)
            force_spring_neighbors = -(spring_neighbor_k / ball_mass) * (distance_to_spring_neighbors - spring_neighbor_l)
            force_spring_neighbors_x = force_spring_neighbors * (currect_loca - spring_neighbor_loca)[:, 0] / distance_to_spring_neighbors 
            force_spring_neighbors_y = force_spring_neighbors * (currect_loca - spring_neighbor_loca)[:, 1] / distance_to_spring_neighbors 
            # print(currect_loca, spring_neighbor_loca, '(currect_loca - spring_neighbor_loca):', (currect_loca - spring_neighbor_loca), (currect_loca - spring_neighbor_loca)[:, 0], 
            #       (currect_loca - spring_neighbor_loca)[:, 1], distance_to_spring_neighbors, (currect_loca - spring_neighbor_loca)[:, 1] / distance_to_spring_neighbors)
            
            force_electric_neighbors = (electicK * q * q)/ (ball_mass*(distance_to_spring_neighbors)**2)
            force_electric_neighbors_x = force_electric_neighbors * (currect_loca - spring_neighbor_loca)[:, 0] / distance_to_spring_neighbors
            force_electric_neighbors_y = force_electric_neighbors * (currect_loca - spring_neighbor_loca)[:, 1] / distance_to_spring_neighbors 
            # calculate the force from distance_to_addition_electric_neighbors (electric only)  
            force_electric_addition_neighbors = (electicK * q * q) / (ball_mass*(distance_to_addition_electric_neighbors)**2)
            force_electric_addition_neighbors_x = force_electric_addition_neighbors * (currect_loca - addition_electric_neighbors_loca)[:, 0] / distance_to_addition_electric_neighbors
            force_electric_addition_neighbors_y = force_electric_addition_neighbors * (currect_loca - addition_electric_neighbors_loca)[:, 1] / distance_to_addition_electric_neighbors
            
            # calculate the total force 
            total_force_x = force_spring_origin_x + np.sum(force_spring_neighbors_x)
            total_force_y = force_spring_origin_y + np.sum(force_spring_neighbors_y)
            if np.isnan(total_force_x).any():
                print('distance_to_origin', distance_to_origin, distance_to_spring_neighbors, distance_to_addition_electric_neighbors)
            
            total_force_x = force_spring_origin_x + np.sum(force_spring_neighbors_x) + np.sum(force_electric_neighbors_x) + np.sum(force_electric_addition_neighbors_x)
            total_force_y = force_spring_origin_y + np.sum(force_spring_neighbors_y) + np.sum(force_electric_neighbors_y) + np.sum(force_electric_addition_neighbors_y)
            # print(force_spring_origin_x, force_spring_neighbors_x, force_electric_neighbors_x, force_electric_addition_neighbors_x)
            
            # fill the value on the table
            slope_speed_x[i][j] = total_force_x
            slope_speed_y[i][j] = total_force_y
            
    # add the eletric field
    slope_speed_x += np.sin(t)
    slope_speed_y += np.cos(t)
    if now_speed_x_ref is not None:
        slope_speed_x_ref = np.sin(t)
        slope_speed_y_ref = np.cos(t)
    
    #print('slope_speed_x', slope_speed_x, 'slope_speed_y:', slope_speed_y)
    if now_loca_x_ref is not None:
        return slope_loca_x, slope_loca_y, slope_speed_x, slope_speed_y, slope_test, slope_loca_x_ref, slope_loca_y_ref, slope_speed_x_ref, slope_speed_y_ref
    else:
        return slope_loca_x, slope_loca_y, slope_speed_x, slope_speed_y, slope_test


# the RK-4 loop
def rk4(t, loca_x, loca_y, speed_x, speed_y, test, loca_x_ref, loca_y_ref, speed_x_ref, speed_y_ref, step_num, h):
    for j in np.arange(0, step_num):
        if loca_x_ref is not None:
            k1_loca_x, k1_loca_y, k1_speed_x, k1_speed_y, k1_test, k1_loca_x_ref, k1_loca_y_ref, k1_speed_x_ref, k1_speed_y_ref = system_with_electric_field(
                t[j], loca_x[j], loca_y[j], speed_x[j], speed_y[j], test[j], 
                loca_x_ref[j], loca_y_ref[j], speed_x_ref[j], speed_y_ref[j])
            
            k2_loca_x, k2_loca_y, k2_speed_x, k2_speed_y, k2_test, k2_loca_x_ref, k2_loca_y_ref, k2_speed_x_ref, k2_speed_y_ref = system_with_electric_field(
                t[j]+h/2, loca_x[j]+h/2*k1_loca_x, loca_y[j]+h/2*k1_loca_y, speed_x[j]+h/2*k1_speed_x, speed_y[j]+h/2*k1_speed_y, test[j]+h/2*k1_test,
                loca_x_ref[j] + h/2*k1_loca_x_ref, loca_y_ref[j]+h/2*k1_loca_y_ref, speed_x_ref[j] + h/2*k1_speed_x_ref, speed_y_ref[j] + h/2*k1_speed_y_ref)
            
            k3_loca_x, k3_loca_y, k3_speed_x, k3_speed_y, k3_test, k3_loca_x_ref, k3_loca_y_ref, k3_speed_x_ref, k3_speed_y_ref = system_with_electric_field(
                t[j]+h/2, loca_x[j]+h/2*k2_loca_x, loca_y[j]+h/2*k2_loca_y, speed_x[j]+h/2*k2_speed_x, speed_y[j]+h/2*k2_speed_y, test[j]+h/2*k2_test,
                loca_x_ref[j] + h/2*k2_loca_x_ref, loca_y_ref[j]+h/2*k2_loca_y_ref, speed_x_ref[j] + h/2*k2_speed_x_ref, speed_y_ref[j] + h/2*k2_speed_y_ref)
            
            k4_loca_x, k4_loca_y, k4_speed_x, k4_speed_y, k4_test, k4_loca_x_ref, k4_loca_y_ref, k4_speed_x_ref, k4_speed_y_ref = system_with_electric_field(
                t[j]+h, loca_x[j]+h*k3_loca_x, loca_y[j]+h*k3_loca_y, speed_x[j]+h*k3_speed_x, speed_y[j]+h*k3_speed_y, test[j]+h*k3_test,
                loca_x_ref[j] + h*k3_loca_x_ref, loca_y_ref[j]+h*k3_loca_y_ref, speed_x_ref[j] + h*k3_speed_x_ref, speed_y_ref[j] + h*k3_speed_y_ref)    
        else:
            k1_loca_x, k1_loca_y, k1_speed_x, k1_speed_y, k1_test = system_with_electric_field(t[j], loca_x[j], loca_y[j], speed_x[j], speed_y[j], test[j])
            k2_loca_x, k2_loca_y, k2_speed_x, k2_speed_y, k2_test = system_with_electric_field(t[j]+h/2, loca_x[j]+h/2*k1_loca_x, loca_y[j]+h/2*k1_loca_y, 
                                                                                                speed_x[j]+h/2*k1_speed_x, speed_y[j]+h/2*k1_speed_y, test[j]+h/2*k1_test)
            k3_loca_x, k3_loca_y, k3_speed_x, k3_speed_y, k3_test = system_with_electric_field(t[j]+h/2, loca_x[j]+h/2*k2_loca_x, loca_y[j]+h/2*k2_loca_y, 
                                                                                                speed_x[j]+h/2*k2_speed_x, speed_y[j]+h/2*k2_speed_y, test[j]+h/2*k2_test)
            k4_loca_x, k4_loca_y, k4_speed_x, k4_speed_y, k4_test = system_with_electric_field(t[j]+h, loca_x[j]+h*k3_loca_x, loca_y[j]+h*k3_loca_y, 
                                                                                                speed_x[j]+h*k3_speed_x, speed_y[j]+h*k3_speed_y, test[j]+h*k3_test)
        
        # calculate the final slope
        k_loca_x = (k1_loca_x + 2*k2_loca_x + 2*k3_loca_x + k4_loca_x)/6
        k_loca_y = (k1_loca_y + 2*k2_loca_y + 2*k3_loca_y + k4_loca_y)/6
        k_speed_x = (k1_speed_x + 2*k2_speed_x + 2*k3_speed_x + k4_speed_x)/6
        k_speed_y = (k1_speed_y + 2*k2_speed_y + 2*k3_speed_y + k4_speed_y)/6
        k_test = (k1_test + 2*k2_test + 2*k3_test + k4_test)/6
        # update the value
        loca_x[j+1] = loca_x[j] + h*k_loca_x
        loca_y[j+1] = loca_y[j] + h*k_loca_y
        speed_x[j+1] = speed_x[j] + h*k_speed_x
        speed_y[j+1] = speed_y[j] + h*k_speed_y
        test[j+1] = test[j] + h*k_test
        
        # add the result of the reference ball
        if loca_x_ref is not None:
            k_loca_x_ref = (k1_loca_x_ref + 2*k2_loca_x_ref + 2*k3_loca_x_ref + k4_loca_x_ref)/6
            k_loca_y_ref = (k1_loca_y_ref + 2*k2_loca_y_ref + 2*k3_loca_y_ref + k4_loca_y_ref)/6
            k_speed_x_ref = (k1_speed_x_ref + 2*k2_speed_x_ref + 2*k3_speed_x_ref + k4_speed_x_ref)/6
            k_speed_y_ref = (k1_speed_y_ref + 2*k2_speed_y_ref + 2*k3_speed_y_ref + k4_speed_y_ref)/6
            loca_x_ref[j+1] = loca_x_ref[j] + h*k_loca_x_ref
            loca_y_ref[j+1] = loca_y_ref[j] + h*k_loca_y_ref
            speed_x_ref[j+1] = speed_x_ref[j] + h*k_speed_x_ref
            speed_y_ref[j+1] = speed_y_ref[j] + h*k_speed_y_ref
    return loca_x, loca_y, speed_x, speed_y, test, loca_x_ref, loca_y_ref, speed_x_ref, speed_y_ref


# the Forward Euler loop
def forward_euler(t, loca_x, loca_y, speed_x, speed_y, test, loca_x_ref, loca_y_ref, speed_x_ref, speed_y_ref, step_num, h):
    for j in np.arange(0, step_num):
        if loca_x_ref is not None:
            loca_x_slope, loca_y_slope, speed_x_slope, speed_y_slope, test_slope, \
                loca_x_ref_slope, loca_y_ref_slope, speed_x_ref_slope, speed_y_ref_slope = system_with_electric_field(
                t[j], loca_x[j], loca_y[j], speed_x[j], speed_y[j], test[j], 
                loca_x_ref[j], loca_y_ref[j], speed_x_ref[j], speed_y_ref[j])
        else:
            loca_x_slope, loca_y_slope, speed_x_slope, speed_y_slope, test_slope = \
                system_with_electric_field(t[j], loca_x[j], loca_y[j], speed_x[j], speed_y[j], test[j])

        # update the value
        loca_x[j+1] = loca_x[j] + h*loca_x_slope
        loca_y[j+1] = loca_y[j] + h*loca_y_slope
        speed_x[j+1] = speed_x[j] + h*speed_x_slope
        speed_y[j+1] = speed_y[j] + h*speed_y_slope
        test[j+1] = test[j] + h*test_slope
        
        # add the result of the reference ball
        if loca_x_ref is not None:
            loca_x_ref[j+1] = loca_x_ref[j] + h*loca_x_ref_slope
            loca_y_ref[j+1] = loca_y_ref[j] + h*loca_y_ref_slope
            speed_x_ref[j+1] = speed_x_ref[j] + h*speed_x_ref_slope
            speed_y_ref[j+1] = speed_y_ref[j] + h*speed_y_ref_slope
    return loca_x, loca_y, speed_x, speed_y, test, loca_x_ref, loca_y_ref, speed_x_ref, speed_y_ref


# the Forward Euler loop
def ab2(t, loca_x, loca_y, speed_x, speed_y, test, loca_x_ref, loca_y_ref, speed_x_ref, speed_y_ref, step_num, h):
    # use the forward euler for the first step
    if loca_x_ref is not None:
        loca_x_slope, loca_y_slope, speed_x_slope, speed_y_slope, test_slope, \
            loca_x_ref_slope, loca_y_ref_slope, speed_x_ref_slope, speed_y_ref_slope = system_with_electric_field(
            t[0], loca_x[0], loca_y[0], speed_x[0], speed_y[0], test[0], 
            loca_x_ref[0], loca_y_ref[0], speed_x_ref[0], speed_y_ref[0])
    else:
        loca_x_slope, loca_y_slope, speed_x_slope, speed_y_slope, test_slope = \
            system_with_electric_field(t[0], loca_x[0], loca_y[0], speed_x[0], speed_y[0], test[0])
    # update the value
    loca_x[1] = loca_x[0] + h*loca_x_slope
    loca_y[1] = loca_y[0] + h*loca_y_slope
    speed_x[1] = speed_x[0] + h*speed_x_slope
    speed_y[1] = speed_y[0] + h*speed_y_slope
    test[1] = test[0] + h*test_slope
    
    # add the result of the reference ball
    if loca_x_ref is not None:
        loca_x_ref[1] = loca_x_ref[0] + h*loca_x_ref_slope
        loca_y_ref[1] = loca_y_ref[0] + h*loca_y_ref_slope
        speed_x_ref[1] = speed_x_ref[0] + h*speed_x_ref_slope
        speed_y_ref[1] = speed_y_ref[0] + h*speed_y_ref_slope    
        
    # save the t-1 step slope
    old_loca_x_slope, old_loca_y_slope, old_speed_x_slope, old_speed_y_slope, old_test_slope = \
        loca_x_slope, loca_y_slope, speed_x_slope, speed_y_slope, test_slope
    if loca_x_ref is not None:
        old_loca_x_ref_slope, old_loca_y_ref_slope, old_speed_x_ref_slope, old_speed_y_ref_slope = \
            loca_x_ref_slope, loca_y_ref_slope, speed_x_ref_slope, speed_y_ref_slope
    
    # update the follow step with ab2
    for j in np.arange(1, step_num):
        if loca_x_ref is not None:
            loca_x_slope, loca_y_slope, speed_x_slope, speed_y_slope, test_slope, \
                loca_x_ref_slope, loca_y_ref_slope, speed_x_ref_slope, speed_y_ref_slope = system_with_electric_field(
                t[j], loca_x[j], loca_y[j], speed_x[j], speed_y[j], test[j], 
                loca_x_ref[j], loca_y_ref[j], speed_x_ref[j], speed_y_ref[j])
        else:
            loca_x_slope, loca_y_slope, speed_x_slope, speed_y_slope, test_slope = \
                system_with_electric_field(t[j], loca_x[j], loca_y[j], speed_x[j], speed_y[j], test[j])

        # calculate the final slope
        final_loca_x_slope = (3/2) * loca_x_slope - (1/2) * old_loca_x_slope
        final_loca_y_slope = (3/2) * loca_y_slope - (1/2) * old_loca_y_slope
        final_speed_x_slope = (3/2) * speed_x_slope - (1/2) * old_speed_x_slope
        final_speed_y_slope = (3/2) * speed_y_slope - (1/2) * old_speed_y_slope
        final_test_slope = (3/2) * test_slope - (1/2) * old_test_slope
        
        if loca_x_ref is not None:
            final_loca_x_ref_slope = (3/2) * loca_x_ref_slope - (1/2) * old_loca_x_ref_slope
            final_loca_y_ref_slope = (3/2) * loca_y_ref_slope - (1/2) * old_loca_y_ref_slope
            final_speed_x_ref_slope = (3/2) * speed_x_ref_slope - (1/2) * old_speed_x_ref_slope
            final_speed_y_ref_slope = (3/2) * speed_y_ref_slope - (1/2) * old_speed_y_ref_slope
        
        # update the value 
        loca_x[j+1] = loca_x[j] + h*final_loca_x_slope
        loca_y[j+1] = loca_y[j] + h*final_loca_y_slope
        speed_x[j+1] = speed_x[j] + h*final_speed_x_slope
        speed_y[j+1] = speed_y[j] + h*final_speed_y_slope
        test[j+1] = test[j] + h*final_test_slope
        
        # add the result of the reference ball 
        if loca_x_ref is not None:
            loca_x_ref[j+1] = loca_x_ref[j] + h*final_loca_x_ref_slope
            loca_y_ref[j+1] = loca_y_ref[j] + h*final_loca_y_ref_slope
            speed_x_ref[j+1] = speed_x_ref[j] + h*final_speed_x_ref_slope
            speed_y_ref[j+1] = speed_y_ref[j] + h*final_speed_y_ref_slope
            
        # assign the old slope
        old_loca_x_slope, old_loca_y_slope, old_speed_x_slope, old_speed_y_slope, old_test_slope = \
            loca_x_slope, loca_y_slope, speed_x_slope, speed_y_slope, test_slope
        if loca_x_ref is not None:
            old_loca_x_ref_slope, old_loca_y_ref_slope, old_speed_x_ref_slope, old_speed_y_ref_slope = \
                loca_x_ref_slope, loca_y_ref_slope, speed_x_ref_slope, speed_y_ref_slope
            
    return loca_x, loca_y, speed_x, speed_y, test, loca_x_ref, loca_y_ref, speed_x_ref, speed_y_ref


def backward_euler_system_with_electric_field(t_j_plus_1, old_loca_x, old_loca_y,
        old_speed_x, old_speed_y, old_test, num_of_ball, h, old_loca_x_ref = None, old_loca_y_ref = None, 
        old_speed_x_ref = None, old_speed_y_ref = None):
    def innerfun(b):     
        #print(b.shape)
        # TODO double check the number of ball
        
        # split the params and reshape
        new_loca_x = b[:num_of_ball*num_of_ball]
        new_loca_x = new_loca_x.reshape(num_of_ball, num_of_ball)
        
        new_loca_y = b[num_of_ball*num_of_ball : num_of_ball*num_of_ball*2]
        new_loca_y = new_loca_y.reshape(num_of_ball, num_of_ball)
        
        new_speed_x = b[num_of_ball*num_of_ball*2 : num_of_ball*num_of_ball*3]
        new_speed_x = new_speed_x.reshape(num_of_ball, num_of_ball)
        
        new_speed_y = b[num_of_ball*num_of_ball*3 : num_of_ball*num_of_ball*4]
        new_speed_y = new_speed_y.reshape(num_of_ball, num_of_ball)
        
        new_test = b[num_of_ball*num_of_ball*4 : num_of_ball*num_of_ball*4 + 1]
        
        # handle the reference ball
        if old_loca_x_ref is not None:
            new_loca_x_ref = b[num_of_ball*num_of_ball*4 + 1 : num_of_ball*num_of_ball*4 + 2]
            
            new_loca_y_ref = b[num_of_ball*num_of_ball*4 + 2 : num_of_ball*num_of_ball*4 + 3]
            
            new_speed_x_ref = b[num_of_ball*num_of_ball*4 + 3 : num_of_ball*num_of_ball*4 + 4]
            
            new_speed_y_ref = b[num_of_ball*num_of_ball*4 + 4 : num_of_ball*num_of_ball*4 + 5]
        
        # send the params to system_with_electric_field get the slope
        if old_loca_x_ref is not None:
            loca_x_slope, loca_y_slope, speed_x_slope, speed_y_slope, test_slope, \
                    loca_x_ref_slope, loca_y_ref_slope, speed_x_ref_slope, speed_y_ref_slope = \
                    system_with_electric_field(t_j_plus_1, new_loca_x, new_loca_y, new_speed_x, new_speed_y, new_test, 
                                    new_loca_x_ref, new_loca_y_ref, new_speed_x_ref, new_speed_y_ref)
        else:
            loca_x_slope, loca_y_slope, speed_x_slope, speed_y_slope, test_slope = \
                    system_with_electric_field(t_j_plus_1, new_loca_x, new_loca_y, new_speed_x, new_speed_y, new_test)            
        
        # construct and return the equation
        all_results = [(n_l - o_l - h*s_l) for n_l, o_l, s_l in zip(new_loca_x.reshape(-1), old_loca_x.reshape(-1), loca_x_slope.reshape(-1))] + \
            [(n_l - o_l - h*s_l) for n_l, o_l, s_l in zip(new_loca_y.reshape(-1), old_loca_y.reshape(-1), loca_y_slope.reshape(-1))] + \
            [(n_s - o_s - h*s_s) for n_s, o_s, s_s in zip(new_speed_x.reshape(-1), old_speed_x.reshape(-1), speed_x_slope.reshape(-1))] + \
            [(n_s - o_s - h*s_s) for n_s, o_s, s_s in zip(new_speed_y.reshape(-1), old_speed_y.reshape(-1), speed_y_slope.reshape(-1))] + \
            [(new_test - old_test - h*test_slope)[0]]
            
        #print([ele.shape for ele in all_results])
        
        if old_loca_x_ref is not None:
            #print(new_loca_x_ref.shape, old_loca_x_ref.shape, loca_x_ref_slope.shape)
            all_results = all_results + [(new_loca_x_ref - old_loca_x_ref - h*loca_x_ref_slope)[0]] 
            all_results = all_results + [(new_loca_y_ref - old_loca_y_ref - h*loca_y_ref_slope)[0]] 
            all_results = all_results + [(new_speed_x_ref - old_speed_x_ref - h*speed_x_ref_slope)[0]]
            all_results = all_results + [(new_speed_y_ref - old_speed_y_ref - h*speed_y_ref_slope)[0]]
            
        return all_results
    return innerfun


# the Backward Euler Loop
def backward_euler(t, loca_x, loca_y, speed_x, speed_y, test, step_num, h, loca_x_ref = None, loca_y_ref = None, speed_x_ref = None, speed_y_ref = None):
    for j in np.arange(0, step_num):
        # create the closure by sending the old result in
        if loca_x_ref is not None:
            backward_fun = backward_euler_system_with_electric_field(t[j+1], loca_x[j], loca_y[j], 
                speed_x[j], speed_y[j], test[j], num_of_ball, h, loca_x_ref[j], loca_y_ref[j], speed_x_ref[j], speed_y_ref[j])
        else:
            backward_fun = backward_euler_system_with_electric_field(t[j+1], loca_x[j], loca_y[j], 
                speed_x[j], speed_y[j], test[j], num_of_ball, h)            
        
        # fslove the result 
        # print(loca_x[j+1].reshape(-1).shape, loca_y[j+1].reshape(-1).shape, speed_x[j+1].reshape(-1).shape, 
        #         speed_y[j+1].reshape(-1).shape, np.expand_dims(test, axis=-1)[j+1].shape)
        # print(np.concatenate([loca_x[j+1].reshape(-1), loca_y[j+1].reshape(-1), speed_x[j+1].reshape(-1), 
        #         speed_y[j+1].reshape(-1), np.expand_dims(test, axis=-1)[j+1], np.expand_dims(loca_x_ref, axis=-1)[j+1], np.expand_dims(loca_y_ref, axis=-1)[j+1], 
        #         np.expand_dims(speed_x_ref, axis=-1)[j+1], np.expand_dims(speed_y_ref, axis=-1)[j+1]], axis=0).shape)
        
        if loca_x_ref is not None:  
            temp = newton(backward_fun, np.concatenate([loca_x[j].reshape(-1), loca_y[j].reshape(-1), speed_x[j].reshape(-1), 
                    speed_y[j].reshape(-1), np.expand_dims(test, axis=-1)[j], np.expand_dims(loca_x_ref, axis=-1)[j], np.expand_dims(loca_y_ref, axis=-1)[j], 
                    np.expand_dims(speed_x_ref, axis=-1)[j], np.expand_dims(speed_y_ref, axis=-1)[j]], axis=0))
        else:
            temp = newton(backward_fun, np.concatenate([loca_x[j].reshape(-1), loca_y[j].reshape(-1), speed_x[j].reshape(-1), 
                    speed_y[j].reshape(-1), np.expand_dims(test, axis=-1)[j]], axis=0))            
            
        # split the params and reshape
        new_loca_x = temp[:num_of_ball*num_of_ball]
        loca_x[j+1] = new_loca_x.reshape(num_of_ball, num_of_ball)
        
        new_loca_y = temp[num_of_ball*num_of_ball : num_of_ball*num_of_ball*2]
        loca_y[j+1] = new_loca_y.reshape(num_of_ball, num_of_ball)
        
        new_speed_x = temp[num_of_ball*num_of_ball*2 : num_of_ball*num_of_ball*3]
        speed_x[j+1] = new_speed_x.reshape(num_of_ball, num_of_ball)
        
        new_speed_y = temp[num_of_ball*num_of_ball*3 : num_of_ball*num_of_ball*4]
        speed_y[j+1] = new_speed_y.reshape(num_of_ball, num_of_ball)
        
        test[j+1] = temp[num_of_ball*num_of_ball*4 : num_of_ball*num_of_ball*4 + 1]
        
        # handle the reference ball
        if loca_x_ref is not None:
            loca_x_ref[j+1] = temp[num_of_ball*num_of_ball*4 + 1 : num_of_ball*num_of_ball*4 + 2]
            loca_y_ref[j+1] = temp[num_of_ball*num_of_ball*4 + 2 : num_of_ball*num_of_ball*4 + 3]
            speed_x_ref[j+1] = temp[num_of_ball*num_of_ball*4 + 3 : num_of_ball*num_of_ball*4 + 4]
            speed_y_ref[j+1] = temp[num_of_ball*num_of_ball*4 + 4 : num_of_ball*num_of_ball*4 + 5]

    if loca_x_ref is not None:
        return loca_x, loca_y, speed_x, speed_y, test, loca_x_ref, loca_y_ref, speed_x_ref, speed_y_ref
    else:
        return loca_x, loca_y, speed_x, speed_y, test


def bdf2_system_with_electric_field(t_j_plus_1, old_loca_x, old_loca_y, old_speed_x, old_speed_y, old_test, 
                                    old_loca_x_t_minus1, old_loca_y_t_minus1, old_speed_x_t_minus1, old_speed_y_t_minus1, old_test_t_minus1,
                                    num_of_ball, h, 
                                    old_loca_x_ref = None, old_loca_y_ref = None, old_speed_x_ref = None, old_speed_y_ref = None,
                                    old_loca_x_ref_t_minus1 = None, old_loca_y_ref_t_minus1 = None, old_speed_x_ref_t_minus1 = None, old_speed_y_ref_t_minus1 = None):
    def innerfun(b):     
        #print(b.shape)
        # TODO double check the number of ball
        
        # split the params and reshape
        new_loca_x = b[:num_of_ball*num_of_ball]
        new_loca_x = new_loca_x.reshape(num_of_ball, num_of_ball)
        
        new_loca_y = b[num_of_ball*num_of_ball : num_of_ball*num_of_ball*2]
        new_loca_y = new_loca_y.reshape(num_of_ball, num_of_ball)
        
        new_speed_x = b[num_of_ball*num_of_ball*2 : num_of_ball*num_of_ball*3]
        new_speed_x = new_speed_x.reshape(num_of_ball, num_of_ball)
        
        new_speed_y = b[num_of_ball*num_of_ball*3 : num_of_ball*num_of_ball*4]
        new_speed_y = new_speed_y.reshape(num_of_ball, num_of_ball)
        
        new_test = b[num_of_ball*num_of_ball*4 : num_of_ball*num_of_ball*4 + 1]
        
        # handle the reference ball
        if old_loca_x_ref is not None:
            new_loca_x_ref = b[num_of_ball*num_of_ball*4 + 1 : num_of_ball*num_of_ball*4 + 2]
            
            new_loca_y_ref = b[num_of_ball*num_of_ball*4 + 2 : num_of_ball*num_of_ball*4 + 3]
            
            new_speed_x_ref = b[num_of_ball*num_of_ball*4 + 3 : num_of_ball*num_of_ball*4 + 4]
            
            new_speed_y_ref = b[num_of_ball*num_of_ball*4 + 4 : num_of_ball*num_of_ball*4 + 5]
        
        # send the params to system_with_electric_field get the slope
        if old_loca_x_ref is not None:
            loca_x_slope, loca_y_slope, speed_x_slope, speed_y_slope, test_slope, \
                    loca_x_ref_slope, loca_y_ref_slope, speed_x_ref_slope, speed_y_ref_slope = \
                    system_with_electric_field(t_j_plus_1, new_loca_x, new_loca_y, new_speed_x, new_speed_y, new_test, 
                                    new_loca_x_ref, new_loca_y_ref, new_speed_x_ref, new_speed_y_ref)
        else:
            loca_x_slope, loca_y_slope, speed_x_slope, speed_y_slope, test_slope = \
                    system_with_electric_field(t_j_plus_1, new_loca_x, new_loca_y, new_speed_x, new_speed_y, new_test)            
        
        # construct and return the equation
        all_results = [(3*n_l - 4*o_l + oo_l - 2*h*s_l) for n_l, o_l, oo_l, s_l in zip(new_loca_x.reshape(-1), old_loca_x.reshape(-1), old_loca_x_t_minus1.reshape(-1), loca_x_slope.reshape(-1))] + \
            [(3*n_l - 4*o_l + oo_l - 2*h*s_l) for n_l, o_l, oo_l, s_l in zip(new_loca_y.reshape(-1), old_loca_y.reshape(-1), old_loca_y_t_minus1.reshape(-1), loca_y_slope.reshape(-1))] + \
            [(3*n_s - 4*o_s + oo_s - 2*h*s_s) for n_s, o_s, oo_s, s_s in zip(new_speed_x.reshape(-1), old_speed_x.reshape(-1), old_speed_x_t_minus1.reshape(-1), speed_x_slope.reshape(-1))] + \
            [(3*n_s - 4*o_s + oo_s - 2*h*s_s) for n_s, o_s, oo_s, s_s in zip(new_speed_y.reshape(-1), old_speed_y.reshape(-1), old_speed_y_t_minus1.reshape(-1), speed_y_slope.reshape(-1))] + \
            [(3*new_test - 4*old_test + old_test_t_minus1 - 2*h*test_slope)[0]]
            
        #print([ele.shape for ele in all_results])
        
        if old_loca_x_ref is not None:
            #print(new_loca_x_ref.shape, old_loca_x_ref.shape, loca_x_ref_slope.shape)
            all_results = all_results + [(new_loca_x_ref - old_loca_x_ref + old_loca_x_ref_t_minus1 - 2*h*loca_x_ref_slope)[0]] 
            all_results = all_results + [(new_loca_y_ref - old_loca_y_ref + old_loca_y_ref_t_minus1 - 2*h*loca_y_ref_slope)[0]] 
            all_results = all_results + [(new_speed_x_ref - old_speed_x_ref + old_speed_x_ref_t_minus1 - 2*h*speed_x_ref_slope)[0]]
            all_results = all_results + [(new_speed_y_ref - old_speed_y_ref + old_speed_y_ref_t_minus1- 2*h*speed_y_ref_slope)[0]]
            
        return all_results
    return innerfun


# the bdf2 Loop
def bdf2(t, loca_x, loca_y, speed_x, speed_y, test, step_num, h, loca_x_ref = None, loca_y_ref = None, speed_x_ref = None, speed_y_ref = None):
    # use forward euler to calculate the first step
    if loca_x_ref is not None:
        loca_x_slope, loca_y_slope, speed_x_slope, speed_y_slope, test_slope, \
            loca_x_ref_slope, loca_y_ref_slope, speed_x_ref_slope, speed_y_ref_slope = system_with_electric_field(
            t[0], loca_x[0], loca_y[0], speed_x[0], speed_y[0], test[0], 
            loca_x_ref[0], loca_y_ref[0], speed_x_ref[0], speed_y_ref[0])
    else:
        loca_x_slope, loca_y_slope, speed_x_slope, speed_y_slope, test_slope = \
            system_with_electric_field(t[0], loca_x[0], loca_y[0], speed_x[0], speed_y[0], test[0])

    # update the value
    loca_x[1] = loca_x[0] + h*loca_x_slope
    loca_y[1] = loca_y[0] + h*loca_y_slope
    speed_x[1] = speed_x[0] + h*speed_x_slope
    speed_y[1] = speed_y[0] + h*speed_y_slope
    test[1] = test[0] + h*test_slope
    
    # add the result of the reference ball
    if loca_x_ref is not None:
        loca_x_ref[1] = loca_x_ref[0] + h*loca_x_ref_slope
        loca_y_ref[1] = loca_y_ref[0] + h*loca_y_ref_slope
        speed_x_ref[1] = speed_x_ref[0] + h*speed_x_ref_slope
        speed_y_ref[1] = speed_y_ref[0] + h*speed_y_ref_slope    
    
    ########## calculate the following steps
    for j in np.arange(1, step_num):
        # create the closure by sending the old result in
        if loca_x_ref is not None:
            bdf2_fun = bdf2_system_with_electric_field(t[j+1], 
                loca_x[j], loca_y[j], speed_x[j], speed_y[j], test[j],
                loca_x[j-1], loca_y[j-1], speed_x[j-1], speed_y[j-1], test[j-1],
                num_of_ball, h, 
                loca_x_ref[j], loca_y_ref[j], speed_x_ref[j], speed_y_ref[j],
                loca_x_ref[j-1], loca_y_ref[j-1], speed_x_ref[j-1], speed_y_ref[j-1])
        else:
            bdf2_fun = bdf2_system_with_electric_field(t[j+1], 
                loca_x[j], loca_y[j], speed_x[j], speed_y[j], test[j], 
                loca_x[j-1], loca_y[j-1], speed_x[j-1], speed_y[j-1], test[j-1], 
                num_of_ball, h)            
        
        # fslove the result 
        # print(loca_x[j+1].reshape(-1).shape, loca_y[j+1].reshape(-1).shape, speed_x[j+1].reshape(-1).shape, 
        #         speed_y[j+1].reshape(-1).shape, np.expand_dims(test, axis=-1)[j+1].shape)
        # print(np.concatenate([loca_x[j+1].reshape(-1), loca_y[j+1].reshape(-1), speed_x[j+1].reshape(-1), 
        #         speed_y[j+1].reshape(-1), np.expand_dims(test, axis=-1)[j+1], np.expand_dims(loca_x_ref, axis=-1)[j+1], np.expand_dims(loca_y_ref, axis=-1)[j+1], 
        #         np.expand_dims(speed_x_ref, axis=-1)[j+1], np.expand_dims(speed_y_ref, axis=-1)[j+1]], axis=0).shape)
        # initial the value with the previouse step
        loca_x[j+1] = loca_x[j]
        loca_y[j+1] = loca_y[j] 
        speed_x[j+1] = speed_x[j]
        speed_y[j+1] = speed_y[j]
        
        if loca_x_ref is not None:  
            temp = newton(bdf2_fun, np.concatenate([loca_x[j+1].reshape(-1), loca_y[j+1].reshape(-1), speed_x[j+1].reshape(-1), 
                    speed_y[j+1].reshape(-1), np.expand_dims(test, axis=-1)[j+1], np.expand_dims(loca_x_ref, axis=-1)[j+1], np.expand_dims(loca_y_ref, axis=-1)[j+1], 
                    np.expand_dims(speed_x_ref, axis=-1)[j+1], np.expand_dims(speed_y_ref, axis=-1)[j+1]], axis=0))
        else:
            temp = newton(bdf2_fun, np.concatenate([loca_x[j+1].reshape(-1), loca_y[j+1].reshape(-1), speed_x[j+1].reshape(-1), 
                    speed_y[j+1].reshape(-1), np.expand_dims(test, axis=-1)[j+1]], axis=0))            
            
        # split the params and reshape
        new_loca_x = temp[:num_of_ball*num_of_ball]
        loca_x[j+1] = new_loca_x.reshape(num_of_ball, num_of_ball)
        
        new_loca_y = temp[num_of_ball*num_of_ball : num_of_ball*num_of_ball*2]
        loca_y[j+1] = new_loca_y.reshape(num_of_ball, num_of_ball)
        
        new_speed_x = temp[num_of_ball*num_of_ball*2 : num_of_ball*num_of_ball*3]
        speed_x[j+1] = new_speed_x.reshape(num_of_ball, num_of_ball)
        
        new_speed_y = temp[num_of_ball*num_of_ball*3 : num_of_ball*num_of_ball*4]
        speed_y[j+1] = new_speed_y.reshape(num_of_ball, num_of_ball)
        
        test[j+1] = temp[num_of_ball*num_of_ball*4 : num_of_ball*num_of_ball*4 + 1]
        
        # handle the reference ball
        if loca_x_ref is not None:
            loca_x_ref[j+1] = temp[num_of_ball*num_of_ball*4 + 1 : num_of_ball*num_of_ball*4 + 2]
            loca_y_ref[j+1] = temp[num_of_ball*num_of_ball*4 + 2 : num_of_ball*num_of_ball*4 + 3]
            speed_x_ref[j+1] = temp[num_of_ball*num_of_ball*4 + 3 : num_of_ball*num_of_ball*4 + 4]
            speed_y_ref[j+1] = temp[num_of_ball*num_of_ball*4 + 4 : num_of_ball*num_of_ball*4 + 5]

    if loca_x_ref is not None:
        return loca_x, loca_y, speed_x, speed_y, test, loca_x_ref, loca_y_ref, speed_x_ref, speed_y_ref
    else:
        return loca_x, loca_y, speed_x, speed_y, test


# T=10 # final simulation time
# N=500 # number of step
# # T=1 # final simulation time
# # N=100 # number of step
# h=T/N

######## initial the vector
# loca_x = np.zeros((N+1, num_of_ball, num_of_ball))
# loca_y = np.zeros((N+1, num_of_ball, num_of_ball))
# speed_x = np.zeros((N+1, num_of_ball, num_of_ball))
# speed_y = np.zeros((N+1, num_of_ball, num_of_ball))
# origin_x = np.zeros((num_of_ball, num_of_ball))
# origin_y = np.zeros((num_of_ball, num_of_ball))
# test = np.zeros(N+1)
# ####### add reference ball
# loca_x_ref = np.zeros(N+1)
# loca_y_ref = np.zeros(N+1)
# speed_x_ref = np.zeros(N+1)
# speed_y_ref = np.zeros(N+1)


# ######## initial the value
# test[0] = 1
# t = np.linspace(0,T,N+1)

# loca_x[0] = np.arange(num_of_ball) + 1
# loca_y[0] = loca_x[0].T
# origin_x = loca_x[0].copy()
# origin_y = loca_y[0].copy()
# #print('loca_x[0]:', loca_x[0], 'loca_y[0]', loca_y[0], 'origin_x:', origin_x, 'origin_y:', origin_y)

# for i in range(num_of_ball):
#     if i % 2 == 0:
#         speed_x[0][:, i] = 1
#         speed_y[0][i, :] = 1
#     else:
#         speed_x[0][:, i] = -1
#         speed_y[0][i, :] = -1
# #print('loca_x[0]:', loca_x[0], 'loca_y[0]:', loca_y[0])
# #print('speed_x[0]:', speed_x[0], 'speed_y[0]:', speed_y[0])



# # loca_x, loca_y, speed_x, speed_y, test, loca_x_ref, loca_y_ref, speed_x_ref, speed_y_ref = \
# # rk4(t, loca_x, loca_y, speed_x, speed_y, test, loca_x_ref, loca_y_ref, speed_x_ref, speed_y_ref, N, h)

# # loca_x, loca_y, speed_x, speed_y, test, loca_x_ref, loca_y_ref, speed_x_ref, speed_y_ref = \
# # forward_euler(t, loca_x, loca_y, speed_x, speed_y, test, loca_x_ref, loca_y_ref, speed_x_ref, speed_y_ref, N, h)

# # loca_x, loca_y, speed_x, speed_y, test, loca_x_ref, loca_y_ref, speed_x_ref, speed_y_ref = \
# # ab2(t, loca_x, loca_y, speed_x, speed_y, test, loca_x_ref, loca_y_ref, speed_x_ref, speed_y_ref, N, h)

# # loca_x, loca_y, speed_x, speed_y, test, loca_x_ref, loca_y_ref, speed_x_ref, speed_y_ref = \
# # backward_euler(t, loca_x, loca_y, speed_x, speed_y, test, N, h, loca_x_ref, loca_y_ref, speed_x_ref, speed_y_ref)

# loca_x, loca_y, speed_x, speed_y, test, loca_x_ref, loca_y_ref, speed_x_ref, speed_y_ref = \
# bdf2(t, loca_x, loca_y, speed_x, speed_y, test, N, h, loca_x_ref, loca_y_ref, speed_x_ref, speed_y_ref)

# # print('location:', loca_x[-1, 0], test[-1])
# # print('speed:', speed[-1, 0], speed[-1, 1])
# # print('test:', test[-1])


# # Plot the final tajectory
# plt.xlabel('x')
# plt.ylabel('steps')  
# for i in range(num_of_ball):   
#     for j in range(num_of_ball):
#         line, = plt.plot(loca_x[:, j, i], loca_y[:, j, i], lw=1)
#         point, = plt.plot(loca_x[:, j, i], loca_y[:, j, i], marker='.')


# # Add labels and title
# #plt.title('Trajectory of the ball vs the number of step (Backward Euler)')
# plt.title('Trajectory of the ball vs the number of step (BDF2)')
# #plt.title('Trajectory of the ball vs the number of step (RK4)')
# #plt.title('Trajectory of the ball vs the number of step (Forward Euler)')
# #plt.title('Trajectory of the ball vs the number of step (AB2)')

# # Show the plot
# #plt.savefig("2d_with_electric_field_backeuler_final_trajectory.png")
# plt.savefig("2d_with_electric_field_BDF2_final_trajectory.png")
# #plt.savefig("2d_with_electric_field_RK4_final_trajectory.png")
# #plt.savefig("2d_with_electric_field_forwardeuler_final_trajectory.png")
# #plt.savefig("2d_with_electric_field_AB2_final_trajectory.png")


# ########### moving point only
# fig,ax = plt.subplots()
# def animate(i):
#     ax.clear()
#     ax.set_xlim(-5, 10)
#     ax.set_ylim(-5, 10)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')    
#     result_lines = []
#     result_points = []
#     for j in range(num_of_ball):
#         for k in range(num_of_ball):   
#             line, = ax.plot(loca_x[i, j, k], loca_y[i, j, k], lw=1)
#             point, = ax.plot(loca_x[i, j, k], loca_y[i, j, k], marker='.')
#             result_lines.append(line)
#             result_points.append(point)
#     line, = ax.plot(loca_x_ref[i], loca_y_ref[i], lw=1, color='green')
#     point, = ax.plot(loca_x_ref[i], loca_y_ref[i], marker='.', color='green')
#     result_lines.append(line)
#     result_points.append(point)
    
#     return result_lines + result_points
        
# ani = FuncAnimation(fig, animate, interval=40, blit=False, repeat=False, frames=N+1)    
# #ani.save("2d_with_electric_field_sprine10_rk4_with_ref.gif", dpi=300, writer=PillowWriter(fps=25))
# # ani.save("2d_with_electric_field_sprine10_forward_euler_with_ref.gif", dpi=300, writer=PillowWriter(fps=25))
# #ani.save("2d_with_electric_field_sprine10_ab2_with_ref.gif", dpi=300, writer=PillowWriter(fps=25))
# #ani.save("2d_with_electric_field_sprine10_backward_euler_with_ref.gif", dpi=300, writer=PillowWriter(fps=25))
# ani.save("2d_with_electric_field_sprine10_bdf2_with_ref.gif", dpi=300, writer=PillowWriter(fps=25))

# # trajectory
# fig,ax = plt.subplots()
# def animate(i):
#     ax.clear()
#     ax.set_xlim(-5, 10)
#     ax.set_ylim(-5, 10)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')    
#     result_lines = []
#     result_points = []
#     for j in range(num_of_ball):
#         for k in range(num_of_ball):    
#             line, = ax.plot(loca_x[:i, j, k], loca_y[:i, j, k], lw=1)
#             point, = ax.plot(loca_x[:i, j, k], loca_y[:i, j, k], marker='.', markersize=1)
#             result_lines.append(line)
#             result_points.append(point)
#     line, = ax.plot(loca_x_ref[:i], loca_y_ref[:i], lw=1, color='green')
#     point, = ax.plot(loca_x_ref[:i], loca_y_ref[:i], marker='.', markersize=1, color='green')
#     result_lines.append(line)
#     result_points.append(point)    
#     return result_lines + result_points
        
# ani = FuncAnimation(fig, animate, interval=40, blit=False, repeat=False, frames=N+1)    
# #ani.save("2d_with_electric_field_trajectory_sprine10_rk4_with_ref.gif", dpi=300, writer=PillowWriter(fps=25))
# # ani.save("2d_with_electric_field_trajectory_sprine10_forward_euler_with_ref.gif", dpi=300, writer=PillowWriter(fps=25))
# #ani.save("2d_with_electric_field_trajectory_sprine10_ab2_with_ref.gif", dpi=300, writer=PillowWriter(fps=25))
# #ani.save("2d_with_electric_field_trajectory_sprine10_backward_euler_with_ref.gif", dpi=300, writer=PillowWriter(fps=25))
# ani.save("2d_with_electric_field_trajectory_sprine10_bdf2_with_ref.gif", dpi=300, writer=PillowWriter(fps=25))





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
    loca_x = np.zeros((step_num+1, num_of_ball, num_of_ball))
    loca_y = np.zeros((step_num+1, num_of_ball, num_of_ball))
    speed_x = np.zeros((step_num+1, num_of_ball, num_of_ball))
    speed_y = np.zeros((step_num+1, num_of_ball, num_of_ball))
    origin_x = np.zeros((num_of_ball, num_of_ball))
    origin_y = np.zeros((num_of_ball, num_of_ball))
    test = np.zeros(step_num+1)
    ####### add reference ball
    loca_x_ref = np.zeros(step_num+1)
    loca_y_ref = np.zeros(step_num+1)
    speed_x_ref = np.zeros(step_num+1)
    speed_y_ref = np.zeros(step_num+1)
    ######## initial the value
    test[0] = 1
    t = np.linspace(0,T,step_num+1)

    loca_x[0] = np.arange(num_of_ball) + 1
    loca_y[0] = loca_x[0].T
    origin_x = loca_x[0].copy()
    origin_y = loca_y[0].copy()
    
    for i in range(num_of_ball):
        if i % 2 == 0:
            speed_x[0][:, i] = 1
            speed_y[0][i, :] = 1
        else:
            speed_x[0][:, i] = -1
            speed_y[0][i, :] = -1
    
    loca_x, loca_y, speed_x, speed_y, test, loca_x_ref, loca_y_ref, speed_x_ref, speed_y_ref = \
        bdf2(t, loca_x, loca_y, speed_x, speed_y, test, step_num, h, loca_x_ref, loca_y_ref, speed_x_ref, speed_y_ref)
    bdf2_location_list.append(loca_x[-1, 0, 0])
    
    loca_x = np.zeros((step_num+1, num_of_ball, num_of_ball))
    loca_y = np.zeros((step_num+1, num_of_ball, num_of_ball))
    speed_x = np.zeros((step_num+1, num_of_ball, num_of_ball))
    speed_y = np.zeros((step_num+1, num_of_ball, num_of_ball))
    origin_x = np.zeros((num_of_ball, num_of_ball))
    origin_y = np.zeros((num_of_ball, num_of_ball))
    test = np.zeros(step_num+1)
    ####### add reference ball
    loca_x_ref = np.zeros(step_num+1)
    loca_y_ref = np.zeros(step_num+1)
    speed_x_ref = np.zeros(step_num+1)
    speed_y_ref = np.zeros(step_num+1)


    ######## initial the value
    test[0] = 1
    t = np.linspace(0,T,step_num+1)

    loca_x[0] = np.arange(num_of_ball) + 1
    loca_y[0] = loca_x[0].T
    origin_x = loca_x[0].copy()
    origin_y = loca_y[0].copy()
    #print('loca_x[0]:', loca_x[0], 'loca_y[0]', loca_y[0], 'origin_x:', origin_x, 'origin_y:', origin_y)

    for i in range(num_of_ball):
        if i % 2 == 0:
            speed_x[0][:, i] = 1
            speed_y[0][i, :] = 1
        else:
            speed_x[0][:, i] = -1
            speed_y[0][i, :] = -1

    loca_x, loca_y, speed_x, speed_y, test, loca_x_ref, loca_y_ref, speed_x_ref, speed_y_ref = \
        backward_euler(t, loca_x, loca_y, speed_x, speed_y, test, step_num, h, loca_x_ref, loca_y_ref, speed_x_ref, speed_y_ref)
    backward_euler_location_list.append(loca_x[-1, 0, 0])
    
    loca_x, loca_y, speed_x, speed_y, test, loca_x_ref, loca_y_ref, speed_x_ref, speed_y_ref = \
        ab2(t, loca_x, loca_y, speed_x, speed_y, test, loca_x_ref, loca_y_ref, speed_x_ref, speed_y_ref, step_num, h)
    ab2_location_list.append(loca_x[-1, 0, 0])
    
    loca_x, loca_y, speed_x, speed_y, test, loca_x_ref, loca_y_ref, speed_x_ref, speed_y_ref = \
        rk4(t, loca_x, loca_y, speed_x, speed_y, test, loca_x_ref, loca_y_ref, speed_x_ref, speed_y_ref, step_num, h)
    rk4_location_list.append(loca_x[-1, 0, 0])
    
    loca_x, loca_y, speed_x, speed_y, test, loca_x_ref, loca_y_ref, speed_x_ref, speed_y_ref = \
        forward_euler(t, loca_x, loca_y, speed_x, speed_y, test, loca_x_ref, loca_y_ref, speed_x_ref, speed_y_ref, step_num, h)
    fe_location_list.append(loca_x[-1, 0, 0])
    
    
    

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
