# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 16:22:12 2021

@author: jain
"""

import numpy as np
import cv2 
import matplotlib.pyplot as plt


# # for gif import following
# from PIL import Image
# import glob


###########################################################################################################33
# define maze
maze = np.zeros((300,400))


for i in range(300):
    
    for j in range(400):
    
        # three rectangles
        if (i >= 20) and (i <=70) and (j >= 200) and (j <= 210):
            maze[i][j] = 1
        if (i >= 60) and (i <=70) and (j >= 210) and (j <= 230):
            maze[i][j] = 1
        if (i >= 20) and (i <=30) and (j >= 210) and (j <= 230):
            maze[i][j] = 1 
        
            # circle
        if ((j-90)**2+(i-(-70+300))**2 <= 35**2):
            maze[i][j] = 1
         
            # Ellipse
        if ((j-246)/60)**2 + ((i-155)/30)**2 <= 1 :
             maze[i][j] = 1
            
            #  inclined rectangle
        if (-0.7*j+1*(300-i))>=73.4 and ((300-i)+1.428*j)>=172.55 and (-0.7*j+1*(300-i))<=99.81 and ((300-i)+1.428*j)<=429:
            maze[i][j] = 1
           
            # polygon
        # if ((300-i)+j>=391 and j-(300-i)<=265 and (300-i)+0.8*j<=425.7 and  (300-i)+0.17*j<=200 and 0.9*j -(300-i) >=148.7) or   (13.5*j+(300-i)<=5256.7 and 1.43*j-(300-i)>=369 and (300-i)+0.81*j >=425):    
            # maze[i][j] = 1
###################################################################################################################3          
  
# queues
visited_nodes = []
open_nodes = []


# making all obstacles nodes to be in visited list
maze1 = maze.copy() 
[row,col] = np.where(maze1 == 1)

for b in range(len(row)):
    r = row[b]
    c = col[b]
    visited_nodes.append([r,c])
    
    
# make every zero to be infinity (9999)
[row,col] = np.where(maze1 == 0)
for b in range(len(row)):
    maze1[row[b]][col[b]] = 9999
    
################################################    
################################################    
# initial and goal nodes  
start = [0,0]
goal = [200,150]
################################################
################################################

# making start node to be zero
maze1[start[0]][start[1]] = 0


open_nodes.append([start[0], start[0]])   # putting start position in the open node list


####################################################################

# Movement functions
def N(maze1):
        # make sure robot can move new position
        if (p != 0) and maze1[p-1][q] > maze1[p][q] + 1:  # check weight of new position, if cost to come is lower than previously estimated - change it
            maze1[p-1][q] = maze1[p][q] + 1
            open_nodes.append([p-1,q])  # put node in open node list
    
            return maze1
        else:  # if cost to come for the new position was lower then dont do anything
            return maze1
    

def S(maze1):

       if (p != 299) and maze1[p+1][q] > maze1[p][q] + 1:
           maze1[p+1][q] = maze1[p][q] + 1
           open_nodes.append([p+1,q])  # put node in open node list
                
           return maze1    
       else:
           return maze1
    

def E(maze1):
    
       if (q != 399) and maze1[p][q+1] > maze1[p][q] + 1:
           maze1[p][q+1] = maze1[p][q] + 1
           open_nodes.append([p,q+1])  # put node in open node list  
        
           return maze1
       else:
           return maze1
    

def W(maze1):

       if (q != 0) and maze1[p][q-1] > maze1[p][q] + 1:
           maze1[p][q-1] = maze1[p][q] + 1
           open_nodes.append([p,q-1])

           return maze1
       else:
           return maze1
       

def NW(maze1):
    
    if (maze1[p-1][q-1] > maze1[p][q] + 2**(.5)) and (p != 0) and (q != 0):
        maze1[p-1][q-1] = maze1[p][q] + 2**(.5)
        open_nodes.append([p-1,q-1])
        
        return maze1
    else:
        return maze1
    
    
def NE(maze1):

    if (p != 0) and (q != 399) and (maze1[p-1][q+1] > maze1[p][q] + 2**(.5)):
        maze1[p-1][q+1] = maze1[p][q] + 2**(.5)
        open_nodes.append([p-1,q+1])
        
        return maze1
    else:
        return maze1


def SE(maze1):

    if (p != 299) and (q != 399) and (maze1[p+1][q+1] > maze1[p][q] + 2**(.5)):
        maze1[p+1][q+1] = maze1[p][q] + 2**(.5)
        open_nodes.append([p+1,q+1])
        
        return maze1
    else: 
        return maze1
    


def SW(maze1):
    
    if (p != 299) and (maze1[p+1][q-1] > maze1[p][q] + 2**(.5)) and (q != 0):
        maze1[p+1][q-1] = maze1[p][q] + 2**(.5)
        open_nodes.append([p+1,q-1])

        return maze1  
    else:
        return maze1
    
###################################################################################### 
   
# Check source node surroundings
# modify obstacles to account for clearance
# Identify all edge points of obstacles and then draw clearance circle treating each point as the center to a circle
[row,col] = np.where(maze1 == 1)
r = []
c = []
for b in range(len(row)):
    
      if maze1[row[b]][col[b]+1] != 1 or maze1[row[b]][col[b]-1] != 1 or maze1[row[b]+1][col[b]] != 1 or maze1[row[b]-1][col[b]] != 1:
          r.append(row[b])
          c.append(col[b])
    
for b in range(len(r)):
    for i in range(300):  
        for j in range(400):
                # circle
            if ((j-c[b])**2+(i-r[b])**2 <= 7.5**2):
                maze1[i][j] = 1


# check to see if initial and goal nodes are in obstacle space

if maze1[goal[0]][goal[1]] == 1:
    print(' The goal is in obstacle space, Please provide another goal node')
    
elif maze1[start[0]][start[1]] == 1:
    print(' The start position is in obstacle space, Please provide another goal node')
    
else:
    
    
    # plotting maze with clearance to see what it looks like
    Modified_maze = maze1.copy()
    plt.imshow(Modified_maze),plt.show()
########################################################################################
# main

    # img = maze1.copy()
    vid = []    # list that contains images to later make a video

    [p,q] = open_nodes[0]  # position of robot
     
    maze1 = np.float32(maze1)   # better processing
    
    while open_nodes != []:   # while open node list is not empty
        
        img = maze1.copy()
        vid.append(img)  # frames for video
        
        [p,q] = open_nodes[0]   # move robot to new source
        
        if [p,q] == goal:   # if robot is in goal node then stop the loop
            break
        
        # check if p,q is in visited node list
        if [p,q] not in visited_nodes:
        
            # check and name neighbouring nodes (assign cost to come values)
            maze1 = N(maze1)
            maze1 = S(maze1)
            maze1 = E(maze1)
            maze1 = W(maze1)
            maze1 = NE(maze1)
            maze1 = NW(maze1)
            maze1 = SE(maze1)
            maze1 = SW(maze1)
            # set current node to be in visited node list
            visited_nodes.append([p,q])
            
            # remove current node from the open node list
            if [p,q] in open_nodes:
                
                open_nodes.remove([p,q])
                # remove_items(open_nodes, [p,q])
                
        elif [p,q] in visited_nodes:
            open_nodes.remove([p,q])
    
    
    
    ###################################################################
    # optimal path
    print("optimal_path")
    maze2 = maze1.copy()
    
    # convert all 1 to 9999
    [row,col] = np.where(maze2 == 1)
    for b in range(len(row)):
        maze2[row[b]][col[b]] = 9999
    
    path_nodes = []   # list that depict nodes for the optimal path
    
    path_nodes.append(goal)   # add goal node to the list
    
    [m,n] = goal   # indices
    
    while [m,n] != start:   # loop until path goes from goal to start node
        # choose the smallest neighbor as the next node
        neighbor_values = [maze2[m+1][n], maze2[m-1][n], maze2[m][n+1], maze2[m][n-1],
                            maze2[m+1][n+1], maze2[m-1][n-1], maze2[m-1][n+1], maze2[m+1][n-1]]
        
        min_value = min(neighbor_values)
        min_value_indices = neighbor_values.index(min_value)   # indices for smallet neighbouring value
        
        if min_value_indices == 0:
            [m,n] = [m+1,n]
            
        if min_value_indices == 1:
            [m,n] = [m-1,n]
            
        if min_value_indices == 2:
            [m,n] = [m,n+1]
            
        if min_value_indices == 3:
            [m,n] = [m,n-1]
            
        if min_value_indices == 4:
            [m,n] = [m+1,n+1]
            
        if min_value_indices == 5:
            [m,n] = [m-1,n-1]
            
        if min_value_indices == 6:
            [m,n] = [m-1,n+1]
    
        if min_value_indices == 7:
            [m,n] = [m+1,n-1]
            
        path_nodes.append([m,n])         # add to path
        maze2[m][n] = 7777
        
        
    # Trim number of images if necessary
    while len(vid) > 1200:
       vid = vid[::2]
        
       
    # make the images better
    for i in range(len(vid) - 1):
        
        edit =  vid[i]
        # change untravelled nodes to zero 
        [row,col] = np.where(edit == 9999)
        for b in range(len(row)):
            edit[row[b]][col[b]] = 0
        # change all path nodes to some shade
        [row,col] = np.where(edit > 1)
        for b in range(len(row)):
            edit[row[b]][col[b]] = 100
        #  chenge all obstacles to 255 in to get black in greyscale.
        [row,col] = np.where(edit == 1)
        for b in range(len(row)):
            edit[row[b]][col[b]] = 255
        
            # save the frame as png
        cv2.imwrite(str(i) + '.png', vid[i])
        
    
    
    # get an imaage for optimal path
    img1 = maze2.copy()
    
    #     # change untravelled nodes to zero 
    # [row,col] = np.where(img1 == 9999)
    # for b in range(len(row)):
    #     img1[row[b]][col[b]] = 0
        # change all path nodes to negative
    [row,col] = np.where(img1 == 7777)
    for b in range(len(row)):
        img1[row[b]][col[b]] = -1
        # change all visited nodes except path nodes
    [row,col] = np.where(img1 > 1)
    for b in range(len(row)):
        img1[row[b]][col[b]] = 0
      # give path nodes some intensity for image formation  
    [row,col] = np.where(img1 == -1)
    for b in range(len(row)):
        img1[row[b]][col[b]] = 175
       # make obstacles black 
    [row,col] = np.where(img1 == 1)
    for b in range(len(row)):
        edit[row[b]][col[b]] = 255
      # give start and goal node lighter intensity  
    img1[start[0]][start[1]] = 75
    img1[goal[0]][goal[1]] = 75
    
    # save image
    cv2.imwrite('optimal_path.png', img1)
    
      
        
    #################################################################   
    # video of path finding
    
    
    # gif to get an idea of what video will look like
    # # Create the frames
    # frames = []
    # imgs = glob.glob("*.png")
    # for i in imgs:
    #     new_frame = Image.open(i)
    #     frames.append(new_frame)
     
    # # Save into a GIF file that loops forever
    # frames[0].save('png_to_gif.gif', format='GIF',
    #                append_images=frames[1:],
    #                save_all=True,
    #                duration=3000, loop=0)
    
    # # save all frames as images
    # for i in range(len(vid) - 1):
        
    #     cv2.imwrite(str(i) + '.png', vid[i])
        
    
    img=[]
    for i in range(0,len(vid)):
        img.append(cv2.imread(str(i)+'.png'))
    
    height,width,layers=img[1].shape
    
    img1 = cv2.imread('optimal_path.png')
    img.append(img1)
    video=cv2.VideoWriter('video_Dijkstra.mp4',-1,30,(width,height))
    
    
    
    for j in range(len(vid)+1):
    
        video.write(img[j])
    
    cv2.destroyAllWindows()
    video.release()
    ######################################################################################   
    
    
    # lets make another video for optimal path in original maze
    
    # variable maze is the original maze array
    
    [row,col] = np.where(maze == 1)
    for b in range(len(row)):
        maze[row[b]][col[b]] = 255
    
    frames = []
    
    while path_nodes != []:
        
        [r,c] = path_nodes.pop()
        maze[r][c] = 150
        
        frame = maze.copy()
        frames.append(frame)   
        #cv2.imwrite(str(i) + 'path.png', frame)
        
    for i in range(len(frames)-1):
        cv2.imwrite(str(i) + 'path.png', frames[i])
        
        
    
    # video maker
    imge=[]
    for i in range(0,len(frames)):
        imge.append(cv2.imread(str(i)+'path.png'))
    
    height,width,layers=imge[1].shape
    
    img1 = cv2.imread('optimal_path.png')
    #imge.append(img1)
    video=cv2.VideoWriter('video_optimal_path.mp4',-1,10,(width,height))
    
    
    
    for j in range(len(frames)):
    
        video.write(imge[j])
    
    cv2.destroyAllWindows()
    video.release()
        
 ######################################################################################       
    
    
    
    
    
    



 
    
   










