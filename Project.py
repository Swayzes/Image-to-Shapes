# !pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# !pip install git+https://github.com/facebookresearch/segment-anything.git

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from matplotlib.backends.backend_tkagg import FigureCanvasAgg
from matplotlib.patches import Polygon, Circle, Ellipse
import PySimpleGUI as sg
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageGrab
import numpy as np
import os, shutil
import io
import cv2
import torch

from skimage.draw import disk
from skimage.draw import ellipse as ellipsedraw
from skimage.draw import polygon as polydraw
from skimage.measure import regionprops
from skimage import morphology
from skimage import filters

# 
def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# Primary Functions
def draw_figure(element, figure):
    """
    Draws the previously created "figure" in the supplied Image Element

    :param element: an Image Element
    :param figure: a Matplotlib figure
    :return: The figure canvas
    """

    plt.close('all')  # erases previously drawn plots
    
    canv = FigureCanvasAgg(figure)
    buf = io.BytesIO()
    canv.print_figure(buf, format='png')
    
    if buf is not None:
        buf.seek(0)
        element.update(data=buf.read())
        return canv
    else:
        return None

def delete_all_masks():
    """Deletes all of the mask images in the 'masks' folder"""
    folder = os.path.join(os.getcwd(), "masks")
    
    # Checks if a masks folder exists, if it doesn't makes one 
    if (os.path.isdir(folder)):
        pass
    else:
        os.mkdir(folder)
    
    # Deletes the images
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    return

def save_element_as_file(element, filename):
    """
    Saves any element as an image file.  Element needs to have an underlyiong Widget available (almost if not all of them do)
    
    :param element: The element to save
    :param filename: The filename to save to. The extension of the filename determines the format (jpg, png, gif, ?)
    """
    widget = element.Widget
    box = (widget.winfo_rootx(), widget.winfo_rooty(), widget.winfo_rootx() + widget.winfo_width(), widget.winfo_rooty() + widget.winfo_height())
    grab = ImageGrab.grab(bbox=box)
    grab.save(filename)

def show_image(im):
    """
    Shows an image in a pyplot
    
    :param im: An image
    :return fig: The plot to show
    """
    # Sets up the plot
    fig = plt.figure()
    hwc = np.array(im)
    ax1 = fig.add_subplot()
    ax1.set_title("Original Image")
    # Plot formatting
    plt.gca().set_position([0, 0, 1, 1])
    fig.set_size_inches(5, 5)
    ax1.imshow(hwc)
    ax1.axis('off')
    ax1.set_aspect('equal')
    ax1.set_anchor('C')
    return fig

def show_point_fig(im, ilist, labels):
    """
    Shows the points made in an image
    
    param im: The inputted image
    param ilist: The list of points (Coordinates)
    param labels: The types of points (Binary)
    """
    # Plot setup
    hwc = np.array(im)
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.imshow(hwc)

    # Plots the positive and negative points made
    pos_points = ilist[labels==1]
    neg_points = ilist[labels==0]
    ax1.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=200, edgecolor='white', linewidth=1.25)
    ax1.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=200, edgecolor='white', linewidth=1.25)   
    
    # Plot Formatting
    ax1.set_title("Point Locations")
    plt.gca().set_position([0, 0, 1, 1])
    fig.set_size_inches(5, 5)
    ax1.imshow(im)
    ax1.axis('off')
    ax1.set_aspect('equal')
    ax1.set_anchor('C')
    return fig

def view_mask(ilist, img, masks, scores):
    """
    Shows the mask made by the points chosen
    
    param ilist: The list of points (Coordinates)
    param img: The inputted image
    param labels: The types of points (Binary)
    param masks: The masks made (Binary array)
    param scores: The scores of the masks compared to the image (Double)
    """
    fig = plt.figure()
    ax1 = fig.add_subplot()
    best_score = 0
    for i, (mask, score) in enumerate(zip(masks, scores)):
        if score > best_score:
            best_score = score
            plt.imshow(img)
            show_mask(mask, plt.gca())
            ax1.scatter(ilist[:, 0], ilist[:, 1], color='green', marker='*', s=200, edgecolor='white', linewidth=1.25)
            ax1.set_title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.gca().set_position([0, 0, 1, 1])
            fig.set_size_inches(5, 5)
            ax1.axis('off')
            ax1.set_aspect('equal')
            ax1.set_anchor('C')
    return fig

def add_mask(masks, scores):
    """
    Gets the best mask from a list of masks
    
    param masks: The masks made (Binary array)
    param scores: The scores of the masks compared to the image (Double)
    return best_mask: The best mask chosen
    """
    best_score = 0
    best_mask = ''
    for i, (mask, score) in enumerate(zip(masks, scores)):
        if score > best_score:
            best_score = score
            best_mask = mask
    return best_mask

# def sam_processing(im):
#     # Could be used for automatic mask generation

#     return

## Shape Fitting Functions
def calculate_IoU(img, comp):
    """
    Calculates the Intersection over Union of two 'binary' array
    
    param img: inputted image (gets a max of 255 for white)
    param comp: comparison mask (Binary array)
    """
    # img is divided by 255 as white needs to be binary
    overlap = img/255 * comp # Logical AND
    union = img/255 + comp # Logical OR
    IoU_calc = overlap.sum()/float(union.sum())
    return IoU_calc

def get_contours(im):
    """
    Gets the contours and the simple version of an image

    param im: An image
    return contours: A list of points of the contours
    return c: the largest contour in the detected contours
    """
    thresh = cv2.threshold(im, 250, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=3)
    contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    return contours, c

def draw_poly(coords, width, height):
    """
    
    """
    img = np.zeros((height, width), dtype=np.double)
    poly = coords
    # fill polygon
    rr, cc = polydraw(poly[:, :, 1][:,0], poly[:, :, 0][:,0], img.shape)
    img[rr, cc] = 1

    return img

def draw_circle(im, width, height):
    threshold_value = filters.threshold_otsu(im)
    label_im = (im > threshold_value).astype(int)
    props = regionprops(label_im, im)
    y0, x0 = props[0]['centroid']
    r = (props[0]['major_axis_length']) /2
    center = (int(y0), int(x0))
    
    img = np.zeros((height, width), dtype=np.double)
    rr, cc = disk(center, int(r), shape=img.shape)
    img[rr, cc] = 1
    center = (int(x0), int(y0))
    return img, center, int(r)

def draw_ellipse(im, w, h):
    contours, cmax = get_contours(im)
    ellipse = cv2.fitEllipse(contours[0])
    centerx = ellipse[0][0]
    centery = ellipse[0][1]
    ellwidth = ellipse[1][0]/2
    ellheight = ellipse[1][1]/2
    angle = ellipse[2]

    img = np.zeros((h, w), dtype=np.double)
    rr, cc = ellipsedraw(centery, centerx, ellheight, ellwidth, img.shape, rotation=np.deg2rad(angle))
    img[rr, cc] = 1

    return img, ellipse

def draw_rectangle(im, w, h):    
    contours, cmax = get_contours(im)
    rect = cv2.minAreaRect(cmax)
    box = cv2.boxPoints(rect)
    
    img = np.zeros((h, w), dtype=np.double)
    poly = box
    rr, cc = polydraw(poly[:, 1], poly[:, 0], img.shape)
    img[rr, cc] = 1
    
    return img, box

def best_IoU(im, width, height, sides):
    contours, cmax = get_contours(im)
    best_points = 0
    best_poly = 0
    IoU = 0
    for eps in np.linspace(0.001, 0.05, 10):
        # approximate the contour
        peri = cv2.arcLength(cmax, True)
        # approx is used to see predicted coords of epsilon
        approx = cv2.approxPolyDP(cmax, eps * peri, True)
    
        poly = draw_poly(approx, width, height)

        if len(approx) <= sides:
            IoU_calc = calculate_IoU(im.copy(), poly)
            if IoU_calc > IoU:
                best_points = approx
                best_poly = poly
                IoU = IoU_calc
        elif eps == 0.05:
            best_points = np.zeros((1,1))
            best_poly = np.zeros((height, width), dtype=np.double)
    return best_poly, best_points

def mask_colour(count):
    c = np.rint(len(os.listdir("test_masks_convex"))/3)
    # c = np.rint(len(os.listdir("masks"))/3)
    if count > c*2:
        color = 'green'
    elif count <= c*2 and count >= c:
        color = 'red'
    else:
        color = 'blue'
    # print(c)
    return color

def best_primitive():
    fig= plt.figure()
    ax1 = fig.add_subplot()
    count=0

    for file in os.scandir('test_masks_convex'):
        im = cv2.imread("test_masks_convex/" + str(file.name), 0)
    # for file in os.scandir('masks'):
    #     im = cv2.imread("masks/" + str(file.name), 0)
        im = np.array(im)
        h, w = im.shape

        circ, center, r = draw_circle(im, w, h)
        elli, elldata = draw_ellipse(im, w, h)
        triangle, points = best_IoU(im, w, h, 3)
        rect, box = draw_rectangle(im, w, h)
        
        IoU = [calculate_IoU(im, circ), calculate_IoU(im, elli), calculate_IoU(im, triangle), calculate_IoU(im, rect)]
        
        color = mask_colour(count)
        
        # print(IoU)
        # print(max(IoU))
        # print("Circle IoU =" + str(calculate_IoU(im, circ)))
        # print("Ellipse IoU =" + str(calculate_IoU(im, elli)))
        # print("Rect IoU =" + str(calculate_IoU(im, rect)))
        # print("Poly IoU =" + str(calculate_IoU(im, triangle)))
        
        if max(IoU) == IoU[0]:
            circle = Circle(center, r)
            circle.set_edgecolor('0')
            circle.set_facecolor("red")
            ax1.add_patch(circle)
        elif max(IoU) == IoU[1]:
            # print(elldata)
            ellipse = Ellipse((elldata[0][0], elldata[0][1]), elldata[1][0], elldata[1][1], angle=elldata[2])
            ellipse.set_edgecolor('0')
            ellipse.set_facecolor("white")
            ax1.add_patch(ellipse)
        elif max(IoU) == IoU[2]:
            xs = points[:, :, 0][:,0]
            ys = points[:, :, 1][:,0]
            polygon = Polygon([[0, 0], [0, 0]])
            polygon.set_xy(np.column_stack([xs, ys]))
            polygon.set_edgecolor('0')
            polygon.set_facecolor("green")
            ax1.add_patch(polygon)
        else:
            xs = box[:,0]
            ys = box[:,1]
            polygon = Polygon([[0, 0], [0, 0]])
            polygon.set_xy(np.column_stack([xs, ys]))
            polygon.set_edgecolor('0')
            polygon.set_facecolor("blue")
            ax1.add_patch(polygon)

        count = count+1

    plt.gca().set_position([0, 0, 1, 1])
    fig.set_size_inches(5, 5)
    ax1.imshow(im)
    ax1.axis('off')
    ax1.set_aspect('equal')
    ax1.set_anchor('C')
    
    return fig

def best_poly_IoU():
    fig= plt.figure()
    ax1 = fig.add_subplot()
    count = 0
    #Inside Loop
    for file in os.scandir('test_masks_convex'):
        im = cv2.imread("test_masks_convex/" + str(file.name), 0)
    # for file in os.scandir('masks'):
    #     im = cv2.imread("masks/" + str(file.name), 0)
        h, w = im.shape
        
        poly, points = best_IoU(im, w, h, 6)

        xs = points[:, :, 0][:,0]
        ys = points[:, :, 1][:,0]

        color = mask_colour(count)

        polygon = Polygon([[0, 0], [0, 0]])
        polygon.set_xy(np.column_stack([xs, ys]))
        polygon.set_edgecolor('0')
        polygon.set_facecolor(color)
        ax1.add_patch(polygon)

        count = count+1
    
    # Outside loop   
    plt.gca().set_position([0, 0, 1, 1])
    fig.set_size_inches(5, 5)
    ax1.imshow(im)
    ax1.axis('off')
    ax1.set_aspect('equal')
    ax1.set_anchor('C')
    return fig
# End of Primary Functions

# Main
def main():
    ## Consistent Variables
    input_list = []
    label_list = []
    mask_list = []
    mask_point_string = ''
    mask_list_string = ''
    temp_mask = ''
    temp_scores = ''
    input = ''
    maskindex = 1

    # Segment anything (SAM Model) preloading
    sam = sam_model_registry['vit_b'](checkpoint='sam_vit_b_01ec64.pth')
    if torch.cuda.is_available():
        sam.to(device="cuda")
    else:
        sam.to(device="cpu")
    predictor = SamPredictor(sam)
    
    #GUI Code
    mask_col = [[sg.Text('Masks', key='-MASKS-')],
            [sg.Text('', key='-MASK_LIST-')],
            [sg.Button('Clear'), sg.Button('View Mask'), sg.Button('Add Mask')],
            [sg.Button('Reset'), sg.Button('Save')]]
            
    shape_col =[[sg.Text('Shapes', key='-SHAPES-')],
            [sg.Button('Best Poly'), sg.Button('Primitive')],
            [sg.Button('Save Output')]]

    layout =   [[sg.Text('Choose an image to process')],
            [sg.Text('Input'), sg.Input(key='-INPUT-'), sg.FileBrowse()],
            [sg.Button('Confirm')],
            # Uncomment this for testing shapes and comment the one with visible=False
            # [sg.Column(mask_col, key='-MASK_COL-', visible=True), sg.Column(shape_col, key='-SHAPE_COL-', visible=True)],
            [sg.Column(mask_col, key='-MASK_COL-', visible=False), sg.Column(shape_col, key='-SHAPE_COL-', visible=False)],
            [sg.Graph(canvas_size=(0,0), graph_bottom_left=(0,0), graph_top_right=(0,0), change_submits=True, key='-IMAGE-', enable_events=True), sg.Image(key='-OUTPUT-')]]
    
    window = sg.Window("Image to shapes", layout, finalize=True)

    input_element = window['-IMAGE-']
    output_element = window['-OUTPUT-']
    input_element.bind("<Button-3>", "+RIGHT+")

    #Events
    while True:
        event, values = window.read()
 
        if event == sg.WIN_CLOSED:
            break
        elif event=='Confirm':
            window['-SHAPE_COL-'].update(visible=False)
            if values['-INPUT-'] != '':
                input = Image.open(values['-INPUT-'])
                #Image Setup
                input.thumbnail((500,500))
                temp = np.array(input)
                im = Image.fromarray(temp)
                buffer = io.BytesIO()
                im.save(buffer, format='PNG')
                data = buffer.getvalue()

                #Show the image chosen
                input_element.set_size((input.width, input.height))
                input_element.change_coordinates(graph_bottom_left = (0, input.height), graph_top_right = (input.width, 0))
                input_element.draw_image(data=data, location=(0,0))

                #Shows mask column section when complete
                window['-MASK_COL-'].update(visible=True)

                #Temporary Code
                window['-SHAPE_COL-'].update(visible=True)

        #Checks for left click events in the 'Image'
        elif event == '-IMAGE-':
            x, y = values["-IMAGE-"]
            #Makes sure points are in the bounds of image
            if x >= 0 and x<=500 and y >=0 and y<=500:
                #See all of the added points for a mask
                if  len(input_list) == 0 and x!='' and y!='':
                    input_list = [(x, y)]
                    label_list = [1]
                elif x != '' and y != '':
                    input_list.append(tuple([x,y])) 
                    label_list.append(1)

                mask_point_string = f"{mask_point_string}[{x},{y}] "

                temp_array = np.array(input_list)
                temp_label_array= np.array(label_list)
                draw_figure(output_element, show_point_fig(input, temp_array, temp_label_array))
    
        #Checks for right click events in the 'Image'
        elif event == '-IMAGE-+RIGHT+':
            x, y = values["-IMAGE-"]
            #Makes sure points are in the bounds of image
            if x >= 0 and x<=500 and y >=0 and y<=500:
                #See all of the added points for a mask
                if  len(input_list) == 0 and x!='' and y!='':
                    input_list = [(x, y)]
                    label_list = [0]
                elif x != '' and y != '':
                    input_list.append(tuple([x,y])) 
                    label_list.append(0)
                mask_point_string = f"{mask_point_string}[{x},{y}] "
                temp_list = np.array(input_list )
                temp_label_list= np.array(label_list)
                draw_figure(output_element, show_point_fig(input, temp_list, temp_label_list))
    
        #Clears the points made
        elif event == 'Clear':
            #Resets point related variables
            input_list = []
            label_list = []
            mask_point_string = ''
            #Shows clear image
            draw_figure(output_element, show_image(input))
        
        elif event == 'View Mask':
            #Checks if point list isn't empty
            if len(input_list) != 0:
                hwc = np.array(input)
                predictor.set_image(hwc)
                temp_list = np.array(input_list)

                #Creates a mask based on points
                masks, scores, logits = predictor.predict(
                    point_coords = temp_list,
                    point_labels = label_list,
                    multimask_output = False
                )

                #Saves the mask and scores output for future use
                temp_mask = masks
                temp_scores = scores

                draw_figure(output_element, view_mask(temp_list, hwc, masks, scores, logits))
                
        elif event=='Add Mask':
            # Add onto mask list 
            mask_list.append(add_mask(temp_mask, temp_scores))
            mask_list_string = f"{mask_list_string}Mask {maskindex}: {mask_point_string} \n"

            window['-MASK_LIST-'].update(mask_list_string)

            # Draw the figure onto the image element
            draw_figure(output_element, view_mask(temp_list, hwc, masks, scores, logits))
            
            # Resets variables
            input_list = []
            label_list = []
            mask_point_string = ''
            temp_mask = ''
            temp_scores = ''
            maskindex = maskindex + 1

        elif event=='Reset':
            # Reset the whole thing
            input_list = []
            label_list = []
            mask_list = []
            mask_point_string = ''
            temp_mask = ''
            temp_scores = ''
            maskindex = 1

            mask_list_string = ''
            #Update elements to clear masks
            window['-MASK_LIST-'].update(mask_list_string)
            window['-SHAPE_COL-'].update(visible=False)

        elif event=='Save':
            # Save the masks chosen
            if len(mask_list) != 0:

                masks = np.array(mask_list)
                delete_all_masks()
                #Loops through the masks, creates a convex hull mask and saves them in a masks folder
                for index in range(len(masks)):
                    a = morphology.remove_small_objects(masks[index], 10, connectivity=2)
                    chull = morphology.convex_hull_image(a, include_borders = True, tolerance=10)
                    im = Image.fromarray(chull)
                    # im = Image.fromarray(masks[index])
                    i = str(index)
    
                    if len(str(index)) < 2:
                        i= '0'+str(index)
                    im.save("masks/m"+ i +".jpg")
    
                #Shows the shape column
                window['-SHAPE_COL-'].update(visible=True)

        ## Shape Events
        elif event=='Best Poly':
            # Fits Polygons to 
            output = best_poly_IoU()
            draw_figure(output_element, output)

        elif event=='Primitive':
            # Fits primitive shapes to saved masks
            output = best_primitive()
            draw_figure(output_element, output)

        elif event=='Save Output':
            # Save the final output figure
            if output !='':
                filename = sg.popup_get_file('Choose file (PNG, JPG, GIF) to save to', save_as=True)
                save_element_as_file(window['-OUTPUT-'], filename)
            
    window.close()

if __name__ == '__main__':
    os.system("pip install git+https://github.com/facebookresearch/segment-anything.git")
    main()