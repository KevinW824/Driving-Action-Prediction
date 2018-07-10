import cv2
import numpy as np

cap = cv2.VideoCapture('black_car.mp4')

draw_right = True
draw_left = True

# Find slopes of all lines
# But only care about lines where abs(slope) > slope_threshold
slope_threshold = 0.5
slopes = []
new_lines = []

# Region-of-interest vertices
# We want a trapezoid shape, with bottom edge at the bottom of the image
trap_bottom_width = 0.85  # width of bottom edge of trapezoid, expressed as percentage of image width
trap_top_width = 0.07  # ditto for top edge of trapezoid
trap_height = 0.4  # height of the trapezoid expressed as percentage of image height


while cap.isOpened():

	# reads frames from a camera
	ret, frame = cap.read()

	# GRAYSCALE
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	# Gaussian Blur
	blur = cv2.GaussianBlur(gray, (5,5), 0)

	#Threshold
	ret,bt = cv2.threshold(blur,165,255,cv2.THRESH_BINARY)

	# finds edges in the input image image and
	# marks them in the output map edges
	edges = cv2.Canny(bt,50,150)

	mask = np.zeros_like(edges)   
	ignore_mask_color = 255   

	# defining a four sided polygon to mask
	imshape = frame.shape
	vertices = np.array([[(235,imshape[0]),(555, 440), (752, 440), (1270,558), (imshape[1],imshape[0])]], dtype=np.int32)
	poly = cv2.fillPoly(mask, vertices, ignore_mask_color)
	masked_edges = cv2.bitwise_and(edges, mask)

	line_image = np.copy(frame)*0

	# Hough line transform
	x = []
	y = []
	lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 15, np.array([]),40,20)
	for line in lines:
		x1, y1, x2, y2 = line [0]
		x.append(x1)
		y.append(y1)
		# cv2.line(frame, (x1,y1), (x2,y2), (0,0,255), 10)

	# Create a "color" binary image to combine with line image
	color_edges = np.dstack((edges, edges, edges)) 

	# Draw the lines on the edge image
	lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 

	# Draw One Line
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
		img_x_center = frame.shape[1] / 2  # x coordinate of center of image
		if slopes[i] > 0 and x1 > img_x_center and x2 > img_x_center:
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
	else:
		left_m, left_b = 1, 1
		draw_left = False
	
	# Find 2 end points for right and left lines, used for drawing the line
	# y = m*x + b --> x = (y - b)/m
	y1 = frame.shape[0]
	y2 = frame.shape[0] * (1 - trap_height)
	
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
		right = cv2.line(frame, (right_x1, y1), (right_x2, y2), (0,255,0), 3)
	if draw_left:
		left = cv2.line(frame, (left_x1, y1), (left_x2, y2), (0,255,0), 3)


	cv2.imshow('line', frame)
		


	# Wait for key to stop
	if cv2.waitKey(1) == 13:
		break



cap.release()
cv2.destroyAllWindows()