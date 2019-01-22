
# My Lane Detection Demo
@description: <br>
&nbsp;&nbsp;&nbsp;&nbsp; a simple Straight-lane detection using hough transform<br>
@auth: xrq<br>
@date: 2019-01


```python
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import cv2
```

### Read and display test image 
Image directory: /home/xrq/lane_detection/Lane_detection/test_images/


```python
image = img.imread("real-test.jpg")
print("shape of image",image.shape)
plt.imshow(image)
```

    shape of image (1080, 1920, 3)
    




    <matplotlib.image.AxesImage at 0x9a40b38>




![png](output_3_2.png)


### Processing Functions
Some transforms function definition:
- gray_scale
- gussian_blur
- canny
- ROI -- 65% of the height was selected
- hough transform
- high angle pass filter
- cv2.inRange() for color selection



```python
 
def filter_colors(image):
    """
    Filter the image to include only yellow and white pixels
    """
    # Filter white pixels
    white_threshold = 180
    lower_white = np.array([white_threshold, white_threshold, white_threshold])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(image, lower_white, upper_white)
    white_image = cv2.bitwise_and(image, image, mask=white_mask)
    
    # Filter yellow pixels
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([90,100,100])
    upper_yellow = np.array([110,255,255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)

    # Combine the two above images
    image2 = cv2.addWeighted(white_image, 1., yellow_image, 1., 0.)
    return white_image

def gray_scale(img):
    # change RGB image to gray image
    return cv2.cvtColor(img,cv2.COLOR_RGB2BGRA)

def gussian_blur(img,kernel_size):
    # gussian filter
    return cv2.GaussianBlur(img,(kernel_size,kernel_size),0)

def canny(img,low_threshold,high_threshold):
    # edge detection
    return cv2.Canny(img,low_threshold,high_threshold)

def region_of_interest(img,vertices):
    #define a blank mask to start with
    mask = np.zeros_like(img)
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape)>2:   # 3 channel
        channel_count = img.shape[2]
        ignore_mask_color = (255,)*channel_count
    else: 
        ignore_mask_color = 255  # black
    cv2.fillPoly(mask,vertices,ignore_mask_color)
    masked_image = cv2.bitwise_and(img,mask) #pixel-wise AND operate
    return masked_image 

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    # In case of error, don't draw the line
    draw_right = True
    draw_left = True
    
    # Find slopes of all lines
    # But only care about lines where abs(slope) > slope_threshold
    slope_threshold = 0.25
    slopes = []
    new_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]  # line = [[x1, y1, x2, y2]]
        
        # Calculate slope
        if x2 - x1 == 0.:  # corner case, avoiding division by 0
            slope = 999.  # practically infinite slope
        else:
            slope = (y2 - y1) / (x2 - x1)
            
        # Filter lines based on slope
        if abs(slope) > slope_threshold:
            slopes.append(slope)
            new_lines.append(line)
        
    lines = new_lines
    
    # Split lines into right_lines and left_lines, representing the right and left lane lines
    # Right/left lane lines must have positive/negative slope, and be on the right/left half of the image
    right_lines = []
    left_lines = []
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        img_x_center = img.shape[1] / 2  # x coordinate of center of image
        if slopes[i] > 0 :
            right_lines.append(line)
        elif slopes[i] < 0 and x1 < img_x_center and x2 < img_x_center:
            left_lines.append(line)
            
    # Run linear regression to find best fit line for right and left lane lines
    # Right lane lines
    right_lines_x = []
    right_lines_y = []
    
    for line in right_lines:
        x1, y1, x2, y2 = line[0]
        
        right_lines_x.append(x1)
        right_lines_x.append(x2)
        
        right_lines_y.append(y1)
        right_lines_y.append(y2)
        
    if len(right_lines_x) > 0:
        right_m, right_b = np.polyfit(right_lines_x, right_lines_y, 1)  # y = m*x + b
        print("contain right lines: ",right_lines_x)
    else:
        right_m, right_b = 1, 1
        draw_right = False
        
    # Left lane lines
    left_lines_x = []
    left_lines_y = []
    
    for line in left_lines:
        x1, y1, x2, y2 = line[0]
        
        left_lines_x.append(x1)
        left_lines_x.append(x2)
        
        left_lines_y.append(y1)
        left_lines_y.append(y2)
        
    if len(left_lines_x) > 0:
        left_m, left_b = np.polyfit(left_lines_x, left_lines_y, 1)  # y = m*x + b
        print("contain left lines: ",left_lines_x)
    else:
        left_m, left_b = 1, 1
        draw_left = False
    
    # Find 2 end points for right and left lines, used for drawing the line
    # y = m*x + b --> x = (y - b)/m
    y1 = img.shape[0]
    y2 = img.shape[0] * 0.8
    
    right_x1 = (y1 - right_b) / right_m
    right_x2 = (y2 - right_b) / right_m
    
    left_x1 = (y1 - left_b) / left_m
    left_x2 = (y2 - left_b) / left_m
    
    # Convert calculated end points from float to int
    y1 = int(y1)
    y2 = int(y2)
    right_x1 = int(right_x1)
    right_x2 = int(right_x2)
    left_x1 = int(left_x1)
    left_x2 = int(left_x2)
    
    # Draw the right and left lines on image
    if draw_right:
        cv2.line(img, (right_x1, y1), (right_x2, y2), color, thickness)
    if draw_left:
        cv2.line(img, (left_x1, y1), (left_x2, y2), color, thickness)
        

def hough_lines(img,rho,theta,threshold,min_line_len,max_line_gap):
    # hough transform to detect lines
    lines = cv2.HoughLinesP(img,rho,theta,threshold,np.array([]),minLineLength=min_line_len,maxLineGap = max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)  # 3-channel RGB image
    '''
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_img,(x1,y1),(x2,y2),color=[255,0,0],thickness=2)
     '''
    draw_lines(line_img, lines)
    return line_img

            
def hpass_angle_filter(lines,angle_threshold):
    # high pass angle filter
    if lines.shape!=None:
        filtered_lines =[]
        for line in lines:
            for x1,y1,x2,y2 in line:
                angle = abs(np.arctan((y2-y1)/(x2-x1))*180/np.pi)
                if angle > angle_threshold:
                    filtered_lines.append([[x1,y1,x2,y2]])
        return filtered_lines
    
  
    
            
def show_img(ax,img,cmap,title):
    if cmap=='gray':
        ax.imshow(img,cmap='gray')
    else:
        ax.imshow(img)
    ax.axis('off')
    ax.set_title(title)
    
def weighted_img(img,initial_img,a=0.8,b=1,r=0.0):
    # add the orignal image and line mask together
    return cv2.addWeighted(initial_img,a,img,b,r)



def pipeline(image):
    two_color_image = filter_colors(image)
    gray_img = gray_scale(two_color_image)  # gray scale image
    gussian_img = gussian_blur(gray_img,3)  # gussian filter
    canny_img = canny(gussian_img,10,150)   # edge detection
    # draw the ROI mask
    imshape = image.shape
    point1 = (700,imshape[0]*0.8)  # shape[0] denote width
    point2 = (1000,imshape[0]*0.8)
    point3 = (520,970)
    point4 = (1375,970)  # shape[1] denote height
    vertices = np.array([[point1,point2,point4,point3]],dtype=np.int32)
    roi_img = region_of_interest(canny_img,vertices)
    # *** finfinshed  **** 
    threshold_angle = 20
    # hough line detection and line filter
    hlines_img = hough_lines(roi_img,rho=1,theta=np.pi/180,threshold=25,min_line_len=10,max_line_gap=10)
    #h_lines = hpass_angle_filter(h_lines,threshold_angle) 
    #print ("lines after angle filter",h_lines)
    # draw empty line mask
    
    img_all_lines = weighted_img(hlines_img,image,a=0.7,b=1,r=0.0) # line mask + original image
    # plot image of each process
    
    '''
    _,ax = plt.subplots(2,4,figsize=(20,10))
    
    show_img(ax[0,0],two_color_image,'color mask','Apply color filter')
    show_img(ax[0,1],image,None,'original_img')
    show_img(ax[0,2],gray_img,'gray','Apply grayscale')
    show_img(ax[0,3],gussian_img,'gray','Apply Gaussian Blur')
    show_img(ax[1,0],canny_img,'gray','Apply Canny')
    show_img(ax[1,1],roi_img,'gray','Apply ROI mask')
    show_img(ax[1,2],hlines_img,None,'Apply Hough')
    plt.show()
    # The final result
    plt.imshow(img_all_lines)
    '''
    plt.figure(figsize=(600,10))
    plt.imshow(hlines_img)


```

### The pipeline of the process
gary_scale--> gussian--> canny--> roi_selection--> houghtransform


```python
image_test = img.imread("real-test.jpg")  
pipeline(image_test)

```

    contain right lines:  [914, 963, 916, 965]
    contain left lines:  [712, 740, 721, 744, 712, 738]
    


![png](output_7_1.png)


### Tesing in vedio


```python
import imageio
from moviepy.editor import VideoFileClip
```

#### Define process function of each frame


```python
def process_image(image):
    two_color_image = gray_scale(image)
    gray_img = gray_scale(two_color_image)  # gray scale image
    gussian_img = gussian_blur(gray_img,3)  # gussian filter
    canny_img = canny(gussian_img,10,150)   # edge detection
    # draw the ROI mask
    imshape = image.shape
    point1 = (700,imshape[0]*0.8)  # shape[0] denote width
    point2 = (1000,imshape[0]*0.8)
    point3 = (520,970)
    point4 = (1375,970)  # shape[1] denote height
    vertices = np.array([[point1,point2,point4,point3]],dtype=np.int32)
    roi_img = region_of_interest(canny_img,vertices)
    # *** finfinshed  **** 
    # hough line detection and line filter
    hlines_img = hough_lines(roi_img,rho=1,theta=np.pi/180,threshold=25,min_line_len=10,max_line_gap=10)
    #h_lines = hpass_angle_filter(h_lines,threshold_angle) 
    #print ("lines after angle filter",h_lines)
    # draw empty line mask
    
    img_all_lines = weighted_img(hlines_img,image,a=0.7,b=1,r=0.0) # line mask + original image
    return img_all_lines
```

#### load video and detect


```python
#dir_video = 'test_videos/'
#video_name = ["solidWhiteRight.mp4","challenge.mp4","solidYellowLeft.mp4"]
video_input = "real-test.mp4"
video_output = video_input.split('.')[0]+"_detect.mp4"
clip1 = VideoFileClip(video_input)
white_clip = clip1.fl_image(process_image)
%time white_clip.write_videofile(video_output,audio=False)
```


```python

```
