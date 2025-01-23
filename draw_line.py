import numpy as np
import cv2
from scipy.interpolate import splprep, splev

from PIL import Image
from shapely.geometry.polygon import Polygon

from time import sleep

road_width = 9
cell_size = 500
canvas_size_hw = 7000

# Global variables
points = []  # To store clicked points
window_name = "Draw Road"


def interpolate_points(points):
    # Extract x and y coordinates from the points
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    # Perform cubic spline interpolation
    tck, u = splprep([x, y],s=0)
    
    
    # Generate a smooth path (interpolated points)
    u_new = np.linspace(0, 1, 100)
    x_new, y_new = splev(u_new, tck)
    
    # Prepare the interpolated path for the front-end
    interpolated_points = list(zip(x_new.tolist(), y_new.tolist()))

    # Calculate the difference between consecutive points
    vect_diff = np.array(interpolated_points)
    vect_diff = vect_diff[1:] - vect_diff[:-1]

    # Calculate the boundaries of the path
    angle = 90.
    theta = (angle/180.) * np.pi

    rotMatrix = np.array([[np.cos(theta), -np.sin(theta)], 
                            [np.sin(theta),  np.cos(theta)]])
    
    left_boundary = road_width*np.dot(vect_diff, rotMatrix)/np.linalg.norm(vect_diff, axis=1)[:, None] + np.array(interpolated_points[:-1])

    angle = -90.
    theta = (angle/180.) * np.pi

    rotMatrix = np.array([[np.cos(theta), -np.sin(theta)], 
                            [np.sin(theta),  np.cos(theta)]])
    right_boundary = road_width*np.dot(vect_diff, rotMatrix)/np.linalg.norm(vect_diff, axis=1)[:, None] + np.array(interpolated_points[:-1])

    return interpolated_points, left_boundary, right_boundary

def draw_interpolated_road(img, points):
    """
    Generates an interpolated road based on points and draws it on the image.
    """
    global height, width
    if len(points) < 2:
        return img  # Not enough points for interpolation
    global left_boundary, right_boundary
    interpolated_points, left_boundary, right_boundary = interpolate_points(points)

    # Draw the interpolated road and boundary lines
    for i in range(len(interpolated_points)-1):
        cv2.line(img, tuple(map(int, interpolated_points[i])), tuple(map(int, interpolated_points[i+1])), (255, 255, 255), 5)
        if i < len(left_boundary)-1:
            cv2.line(img, tuple(map(int, left_boundary[i])), tuple(map(int, left_boundary[i+1])), (0, 255, 0), 2)
            cv2.line(img, tuple(map(int, right_boundary[i])), tuple(map(int, right_boundary[i+1])), (0, 255, 0), 2)

    img = cv2.resize(img, (height, width))
    return img

def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback to capture points on mouse clicks.
    """
    global points, canvas

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        # Draw the point on the canvas
        cv2.circle(canvas, (x, y), 1, (0, 255, 0), -1)  # Green dot

        # Redraw the road with interpolation
        canvas_copy = canvas.copy()
        canvas_with_road = draw_interpolated_road(canvas_copy, points)
        cv2.imshow(window_name, canvas_with_road)

def process_image(width, height):
    nb_ite_dilate = 0
    file = "./Textures/texture3.png"
    image = Image.open(file)
    mask_file = "./Textures/" + file.split('/')[-1].split('.')[-2] + "_masque.png"
    mask = Image.open(mask_file).convert("L")  # Convert mask to grayscale

    # Create a blank canvas (3000x3000 pixels)
    canvas_size = (canvas_size_hw, canvas_size_hw)
    canvas = Image.new("L", canvas_size, color=0)
    canvas_colored = Image.new("RGB", canvas_size, color=(0, 0, 0))

    # Define grid properties
    x_cells = canvas_size[0] // cell_size
    y_cells = canvas_size[1] // cell_size

    # Resize the uploaded image to fit the grid cell
    resized_image = image.resize((cell_size, cell_size))
    mask = mask.resize((cell_size, cell_size))

    # Define the path polygon
    resize_factor = 6000/ height
    path_polygon = Polygon(np.concatenate([left_boundary*resize_factor, np.flip(right_boundary*resize_factor, axis=0)]))

    # Place the image in the grid
    for y in range(y_cells):
        for x in range(x_cells):
            top_left_x = x * cell_size
            top_left_y = y * cell_size

            cell_polygon = Polygon([(top_left_x-cell_size, top_left_y-cell_size), 
                                    (top_left_x + cell_size*2, top_left_y-cell_size), 
                                    (top_left_x + cell_size*2, top_left_y + cell_size*2), 
                                    (top_left_x-cell_size, top_left_y + cell_size*2)])

            # Check if the cell center is inside the polygon
            if path_polygon.intersects(cell_polygon):
                canvas_colored.paste(resized_image, (top_left_x, top_left_y))
                canvas.paste(mask, (top_left_x, top_left_y))

    # Convert the canvas to a numpy array for OpenCV processing
    canvas_array = np.array(canvas)

    # Invert the mask to make black regions white and white regions black
    inverted_mask = cv2.bitwise_not(canvas_array)

    # Dilate the inverted mask (expand black areas)
    dilated_mask = cv2.dilate(inverted_mask, np.ones((5, 5), np.uint8), iterations=nb_ite_dilate)

    # Re-invert the mask back to its original polarity
    canvas_array = cv2.bitwise_not(dilated_mask)


    # Detect contours in the mask
    contours, _ = cv2.findContours(canvas_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a new blank canvas to draw the filtered contours
    filtered_canvas = np.zeros_like(canvas_array)

    # Filter and draw contours within the path polygon
    for contour in contours:
        # Convert contour to a Shapely Polygon
        contour_points = [tuple(point[0]) for point in contour]
        #contour_polygon = Polygon(contour_points)
        if len(contour_points) >= 3:  # A valid polygon requires at least 3 points
            contour_polygon = Polygon(contour_points)

            # Check if the contour is doesn't intersect the path polygon
            if path_polygon.intersects(contour_polygon):
                # Draw the contour onto the filtered canvas
                cv2.drawContours(
                    filtered_canvas, [np.array(contour_points, dtype=np.int32)], -1, (255, 0, 255), thickness=cv2.FILLED
                )
    # Dilate back the filtered canvas
    filtered_canvas = cv2.dilate(filtered_canvas, np.ones((5, 5), np.uint8), iterations=nb_ite_dilate)


    # Apply the final mask to the colored canvas by multiplying
    canvas_colored_array = np.array(canvas_colored)  # Convert colored canvas to numpy array
    texture_colored_array = cv2.bitwise_and(canvas_colored_array, canvas_colored_array, mask=filtered_canvas)
    
    # show image using cv2 with rgb to bgr conversion and resize it
    #texture_colored_array = cv2.cvtColor(texture_colored_array, cv2.COLOR_RGB2BGR)
    texture_colored_array = cv2.resize(texture_colored_array, (width, height))
    
    return texture_colored_array
    cv2.imshow("Final Image", texture_colored_array)
    cv2.waitKey(0)

    

def draw(h, w,colors):
    global canvas
    global left_boundary, right_boundary
    global width, height
    width, height = h, w

    canvas = (colors*255).reshape((height, width, 3), order="F").astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    # Create the OpenCV window
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, canvas)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("Click to add points. Press 'r' to reset, 'q' to quit.")

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            # Quit the program
            break
        elif key == ord('r'):
            # Reset the canvas and points
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            points.clear()
            print("Canvas reset.")
        elif key == 13:  # Enter key
            # Stop adding points and process the image
            texture_colored_array = process_image(width, height)
            print("Image processing complete.")
            break

    cv2.destroyAllWindows()
    return texture_colored_array