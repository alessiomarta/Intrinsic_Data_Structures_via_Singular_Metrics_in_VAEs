#!/usr/bin/env python

import cv2
import math
import numpy as np
import random

def generate_image(n_balls,seed=None,std_intensity=0.1,intensity_dist='gauss',eps=2,fix_int=False, dimension=128):
    x_c_v = dimension/2.
    y_c_v = dimension/2.
    centers=np.zeros((0,2))
    radii=np.zeros(0)
    if seed is None:
        rng=random.Random()
    else:
        rng=random.Random(seed)

    #Generate the balls
    radius_aux = 16 + 16. * (rng.uniform(0,1)) #16-32
    angle = 2*math.pi*rng.uniform(0,1)
    
    x_c1 = x_c_v + radius_aux * np.cos(angle)
    y_c1 = y_c_v + radius_aux * np.sin(angle)

    x_c2 = x_c_v + radius_aux * np.cos(math.pi+angle)
    y_c2 = y_c_v + radius_aux * np.sin(math.pi+angle)
    
    rmax1 = min(x_c1-16,128-x_c1-12,y_c1-12,128-y_c1-12,radius_aux-16)
    rmax2 = min(x_c2-12,128-x_c2-12,y_c2-12,128-y_c2-12,radius_aux-16)

    r1 = 10 + rmax1 * rng.uniform(0,1)
    r2 = 10 + rmax2 * rng.uniform(0,1)
        
    centers=np.array([[x_c1,y_c1],[x_c2,y_c2]])
    radii=np.array([r1,r2])
   
    image = 0.*np.ones((int(dimension),int(dimension),3),np.uint8)
    conds=[]
    colors = [int(255./3. *(1+i)) for i in range(n_balls)]

    for i in range(n_balls):
        if intensity_dist=='gauss':
            cond = rng.gauss(1.5,std_intensity)
        elif intensity_dist=='uniform':
            cond = rng.uniform(1.5-std_intensity,1.5+std_intensity)
        elif intensity_dist=='uniform_rng':
            cond = 8.+i*0.25
        else:
            raise NameError('Unknown intensity distribution!')
        if fix_int and i==n_balls-1:
            cond=1.5*n_balls-np.sum(np.array(conds))
        conds.append(cond)
        a = int(200*(cond-0.2)/(1.8-0.2))
        if intensity_dist=='uniform_rng':
            a = colors[i]
        color = (a,a,a)
        image = cv2.circle(image, (int(centers[i,0]),int(centers[i,1])), int(radii[i]), color, -1)
    
    #Generate the lines
    n_lines = 1
    for i in range(n_lines):
        
        center = (int(dimension//2 + (dimension//4) * (rng.uniform(0,1)-0.5)),int(dimension//2 + (dimension//4) * (rng.uniform(0,1)-0.5)))
        
        delta = 14
        maxl = min(center[0]-delta,center[1]-delta,128-center[0]-delta,128-center[0]-delta)

        length = 4 + maxl * rng.uniform(0,1)
        angle = 2*math.pi*rng.uniform(0,1)
        
        start_point = ( int(center[0] + length * np.cos(angle)), int(center[1] + length * np.sin(angle)))
        end_point = ( int(center[0] + length * np.cos(math.pi+angle)), int(center[1] + length * np.sin(math.pi+angle)))
        
        color = (253+i,253+i,253+i)
        thickness = 3
        image = cv2.line(image, start_point, end_point, color, thickness)

    return np.mean(image,-1)/255.


def gen_paraboloid():

    x = 2*np.random.rand()-1
    y = 2*np.random.rand()-1

    z = 0.5 * (x**2 + y**2)
    return np.array([x,y,z],dtype=np.float32)


def gen_circle():
    
    theta = 2 * np.pi * np.random.rand()
    x = np.cos(theta)
    y = np.sin(theta)
    return np.array([x,y],dtype=np.float32)
