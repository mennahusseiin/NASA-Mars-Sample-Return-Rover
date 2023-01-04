import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float32)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float32)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    mask=cv2.warpPerspective(np.ones_like(img[:,:,0]),M,(img.shape[1], img.shape[0]))
    return warped , mask
def find_rock(img, yellow_thresh=(110, 110, 50)):
    color_select = np.zeros_like(img[:,:,0])
    rock = (img[:,:,0] > yellow_thresh[0]) & (img[:,:,1] > yellow_thresh[1]) & (img[:,:,2] < yellow_thresh[2])
    color_select[rock] = 1
    return color_select
# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    #save Start position
    #if Rover.start_pos == None:
      #  Rover.start_pos = Rover.pos
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    original_img = Rover.img
    # 1) Define source and destination points for perspective transform

    # to a grid where each 10x10 pixel square represents 1 square meter
    dst_size = 5 
    # Set a bottom offset to account for the fact that the bottom of the image 
    # is not the position of the rover but a bit in front of it
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[original_img.shape[1]/2 - dst_size, original_img.shape[0] - bottom_offset],
                      [original_img.shape[1]/2 + dst_size, original_img.shape[0] - bottom_offset],
                      [original_img.shape[1]/2 + dst_size, original_img.shape[0] - 2*dst_size - bottom_offset], 
                      [original_img.shape[1]/2 - dst_size, original_img.shape[0] - 2*dst_size - bottom_offset],
                      ])


    # 2) Apply perspective transform
    warped_img , mask = perspect_transform(original_img, src = source, dst = destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    
    #terrain
    terrain_img = color_thresh(warped_img, rgb_thresh=(160, 160, 160))
    #rock samples
    rock_img = find_rock(warped_img)
    #obstacles
    obstacles_img = np.absolute(np.float32(terrain_img)-1) * mask # cv2.bitwise_not(terrain_img)-254

    #4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    Rover.vision_image[:, :, 0] = obstacles_img*200
    Rover.vision_image[:, :, 1] = rock_img*200
    Rover.vision_image[:, :, 2] = terrain_img*200
    



    xpix, ypix = rover_coords(terrain_img)
    rock_xpix, rock_ypix = rover_coords(rock_img)
    obs_xpix, obs_ypix = rover_coords(obstacles_img)
    
    # 5) Convert rover-centric pixel values to world coords
    
    world_size = Rover.worldmap.shape[0]
    scale = 3 * dst_size
    yaw = Rover.yaw
    xpos, ypos = Rover.pos
    
    x_world, y_world = pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale)
    rock_x_world, rock_y_world = pix_to_world(rock_xpix, rock_ypix, xpos, ypos, yaw, world_size, scale)
    obs_x_world, obs_y_world = pix_to_world(obs_xpix, obs_ypix, xpos, ypos, yaw, world_size, scale)
    
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1

    #If roll and pitch values are near zero, then the image is valid.
    if ((Rover.roll < 0.5)& (Rover.pitch < 0.5)):
        Rover.worldmap[obs_y_world, obs_x_world, 0] = 255
        
        Rover.worldmap[y_world, x_world, 2] = 255
        nav_pix = Rover.worldmap[:, :, 2] > 0
        Rover.worldmap[nav_pix, 0] = 0
        # clip to avoid overflow
        Rover.worldmap = np.clip(Rover.worldmap, 0, 255)   

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles


    dist, angles = to_polar_coords(xpix, ypix)
    Rover.nav_dists = dist
    Rover.nav_angles = angles
    dist, angles = to_polar_coords(rock_xpix, rock_ypix)
    Rover.samples_dists = dist
    Rover.samples_angles = angles
    rock_dir = np.mean(angles)
    if rock_img.any():
        if ((Rover.roll < 0.5 or Rover.roll > 359.5)  )& ((Rover.pitch < 0.5 or Rover.pitch > 359.5) ):
            Rover.worldmap[rock_y_world, rock_x_world, 1] = 255
            rock_pix = Rover.worldmap[:, :, 1]>0
            Rover.worldmap[rock_pix,0]=0
    else:
        Rover.vision_image[:,:,1] = 0
    
    return Rover
