# !pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# !pip install git+https://github.com/facebookresearch/segment-anything.git

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from matplotlib.backends.backend_tkagg import FigureCanvasAgg
from matplotlib.patches import Polygon, Circle
import PySimpleGUI as sg
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os, shutil
import io
import cv2

from skimage import color, measure, draw, img_as_bool
from skimage.draw import disk
from skimage.draw import polygon as polydraw
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
from skimage import filters

#Testing
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_anns(anns, axes=None):
    if len(anns) == 0:
        return
    if axes:
        ax = axes
    else:
        ax = plt.gca()
        ax.set_autoscale_on(False)
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.5)))

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   

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
    folder = os.path.join(os.getcwd(), "masks")
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

def show_image(im):
    fig = plt.figure()
    hwc = np.array(im)
    ax1 = fig.add_subplot()
    ax1.set_title("Original Image")
    ax1.axis("on")
    ax1.imshow(hwc)
    return fig

def show_point_fig(im, ilist):
    hwc = np.array(im)
    
    fig = plt.figure()
    ax1 = fig.add_subplot()

    ax1.imshow(hwc)
    ax1.scatter(ilist[:, 0], ilist[:, 1], color='green', marker='*', s=200, edgecolor='white', linewidth=1.25)
    ax1.set_position([0, 0, 1, 1])
    ax1.set_title("Point Locations")
    #ax1.axis('on')
    return fig

def view_mask(ilist, img, masks, scores, logits):
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
    return fig

def add_mask(masks, scores, logits):
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
    overlap = img * comp # Logical AND
    union = img + comp # Logical OR
    IoU_calc = overlap.sum()/float(union.sum())
    return IoU_calc

def get_contours(im):
    thresh = cv2.threshold(im, 250, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=3)
    contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    return contours, c

def draw_poly(coords, width, height):
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

def draw_rectangle(im, w, h):    
    contours, cmax = get_contours(im)
    rect = cv2.minAreaRect(cmax)
    box = cv2.boxPoints(rect)
    
    img = np.zeros((h, w), dtype=np.double)
    poly = box
    rr, cc = polydraw(poly[:, 1], poly[:, 0], img.shape)
    img[rr, cc] = 1
    
    return img, box

def best_IoU(im, width, height):
    contours, cmax = get_contours(im)

    count = 0
    best_poly = 0
    IoU = 0
    for eps in np.linspace(0.001, 0.05, 10):
        # approximate the contour
        peri = cv2.arcLength(cmax, True)
        # approx is used to see predicted coords of epsilon
        approx = cv2.approxPolyDP(cmax, eps * peri, True)
    
        poly = draw_poly(approx, width, height)
        #poly=approx
        if len(approx) < 7:
            overlap = im.copy() * poly # Logical AND
            union = im.copy() + poly # Logical OR
            IoU_calc = overlap.sum()/float(union.sum())
            if IoU_calc > IoU:
                best_poly = approx
                IoU = IoU_calc
        elif eps == 0.05:
            best_poly = approx
    return best_poly

def mask_colour(count):
    c = np.rint(len(os.listdir("test_masks"))/3)
    if count > c*2:
        color = 'green'
    elif count < c*2 and count > c:
        color = 'red'
    else:
        color = 'blue'
    return color

def best_primitive(image):
    fig, axes = plt.subplots(1,2, figsize=(10, 8))
    ax = axes.ravel()
    count=0

    for file in os.scandir('test_masks'):
        im = cv2.imread("test_masks/" + str(file.name), 0)
        w = im.shape[1]
        h = im.shape[0]
        rect, box = draw_rectangle(im, w, h)
        circ, center, r = draw_circle(im, w, h)
        #triangle = draw_poly(im, w, h, 4)
        
        color = mask_colour(count)
        
        if calculate_IoU(im, circ) + 0.005  < calculate_IoU(im, rect):
            xs = box[:,0]
            ys = box[:,1]
            polygon = Polygon([[0, 0], [0, 0]])
            polygon.set_xy(np.column_stack([xs, ys]))
            polygon.set_edgecolor('0')
            polygon.set_facecolor(color)
            ax[1].add_patch(polygon)
        else:
            circle = Circle(center, r)
            circle.set_edgecolor('0')
            circle.set_facecolor(color)
            ax[1].add_patch(circle)
        count = count+1

    ax[0].imshow(image)
    ax[0].set_title('Input')
    ax[0].axis('off')
    ax[1].imshow(im)
    ax[1].set_title('Output')
    ax[1].axis('off')
    return fig

def best_poly_IoU(image):
    fig, axes = plt.subplots(1,2, figsize=(10, 8))
    ax = axes.ravel()

    count = 0
    #Inside Loop
    for file in os.scandir('test_masks'):
        im = cv2.imread("test_masks/" + str(file.name), 0)
        w = im.shape[1]
        h = im.shape[0]
        poly = best_IoU(im, w, h)

        xs = poly[:, :, 0][:,0]
        ys = poly[:, :, 1][:,0]

        color = mask_colour(count)

        polygon = Polygon([[0, 0], [0, 0]])
        polygon.set_xy(np.column_stack([xs, ys]))
        polygon.set_edgecolor('0')
        polygon.set_facecolor(color)
        ax[1].add_patch(polygon)

        count = count+1
    
    # Outside loop   
    ax[0].imshow(image)
    ax[0].set_title('Input')
    ax[0].axis('off')
    ax[1].imshow(im)
    ax[1].set_title('Output')
    ax[1].axis('off')
    return fig
# End of Primary Functions

# Main
def main():
    ## Consistent Variables
    input_list = []
    mask_list = []
    mask_point_string = ''
    mask_list_string = ''
    maskindex = 1
    input = ''

    sam = sam_model_registry['vit_b'](checkpoint='sam_vit_b_01ec64.pth')
    sam.to(device="cuda")
    predictor = SamPredictor(sam)
    
    #GUI Code
    mask_col = [[sg.Text('Masks', key='-MASKS-')],
             [sg.Text('', key='-MASK_LIST-')],
             [sg.Text('X='), sg.Input(key='-XCoord-', size=(10,1)), sg.Text('Y='),sg.Input(key='-YCoord-', size=(10,1)), sg.Button('View Point')],
             [sg.Button('Add Point'), sg.Button('View Mask')],
             [sg.Button('Add Mask'), sg.Button('Undo')],
             [sg.Button('Reset'), sg.Button('Save')]]
                
    shape_col = [[sg.Text('Shapes', key='-SHAPES-')],
             [sg.Button('Best Poly'), sg.Button('Primitive')],
             [sg.Button('Save Output')]]

    layout = [[sg.Text('Choose an image to process')],
          [sg.Text('Input'), sg.Input(key='-INPUT-'), sg.FileBrowse()],
          [sg.Button('Confirm')],
          [sg.Column(mask_col, key='-MASK_COL-', visible=False), sg.Column(shape_col, key='-SHAPE_COL-', visible=False)],
          [sg.Image(key='-IMAGE-', enable_events=True)],
          [sg.Text("Mouse Coord:"), sg.Text(size=20, key='-COORD-')]]
    window = sg.Window("Image to shapes", layout, finalize=True)

    input_element = window['-IMAGE-']
    
    # layout = [[sg.Text('Choose an image to process')],
    #       [sg.Text('Input'), sg.Input(key='-INPUT-'), sg.FileBrowse()],
    #       [sg.Button('Confirm')],
    #       [sg.Column(mask_col, key='-MASK_COL-', visible=False), sg.Column(shape_col, key='-SHAPE_COL-', visible=False)],
    #       [sg.Graph(canvas_size=(0,0), graph_bottom_left=(0,0), graph_top_right=(0,0), change_submits=True, key='-IMAGE-', enable_events=True), sg.Image(key='-OUTPUT-')]]
    # window = sg.Window("Image to shapes", layout, finalize=True)

    # input_element = window['-IMAGE-']
    # output_element = window['-OUTPUT-']
    while True:
        event, values = window.read()
 
        if event == sg.WIN_CLOSED:
            break
        elif event=='Confirm':
            window['-SHAPE_COL-'].update(visible=False)
            if values['-INPUT-'] != '':
                input = Image.open(values['-INPUT-'])
                input.thumbnail((500,500))
                # image = sg.EMOJI_BASE64_HAPPY_BIG_SMILE
                # im = cv2.imread(values['-INPUT-'])
                #Show the image chosen
                # temp = np.array(input)
                # im = Image.fromarray(temp)
                # buffer = io.BytesIO()
                # im.save(buffer, format='PNG')
                # data = buffer.getvalue()

                # input_element.set_size((input.width, input.height))
                # input_element.change_coordinates(graph_bottom_left = (0, input.height) ,graph_top_right = (input.width, 0))
                # input_element.draw_image(data=data, location=(0,0))
                draw_figure(input_element, show_image(input))
                # #Attempt at mouse click tracking
                window['-MASK_COL-'].update(visible=True)

                #temporary
                window['-SHAPE_COL-'].update(visible=True)

        elif event == '-IMAGE-':
            try:
                x, y = values["-IMAGE-"]
                #e = window['-IMAGE-'].user_bind_event

                #location = (x-28, y+28) 
                # window['-COORD-'].update(f'({e.x}, {e.y})')
                print(x)
                print(y)
            except:
                print("Err")
                pass

        elif event == 'View Point':
            x = values['-XCoord-']
            y = values['-YCoord-']
            #See the points in the image
            if x != '' and y != '':
                temp_list =  np.array([[int(x), int(y)]])
                draw_figure(input_element, show_point_fig(input, temp_list))

        elif event == 'Add Point':
            x = int(values['-XCoord-'])
            y = int(values['-YCoord-'])
            #See all of the added points for a mask
            if  len(input_list) == 0 and x!='' and y!='':
                input_list = [(x, y)]
            
            elif x != '' and y != '':
                input_list.append(tuple([x,y]))

            mask_point_string = f"{mask_point_string}[{x},{y}] "
            print(mask_point_string)
            temp_list = np.array(input_list)

            draw_figure(input_element, show_point_fig(input, temp_list))
        
        elif event == 'View Mask':
            x = values['-XCoord-']
            y = values['-YCoord-']
            #See the points
            if x != '' and y != '' and len(input_list) != 0:
                input_label = np.ones(len(input_list))
                print(len(input_list))
                print(input_label)

                hwc = np.array(input)
                predictor.set_image(hwc)
                temp_list = np.array(input_list)

                masks, scores, logits = predictor.predict(
                    point_coords = temp_list,
                    point_labels = input_label,
                    multimask_output = False
                )

                draw_figure(input_element, view_mask(temp_list, hwc, masks, scores, logits))
                
        elif event=='Add Mask':
            input_label = np.ones(len(input_list))
            print(input_label)

            hwc = np.array(input)
            predictor.set_image(hwc)
            temp_list = np.array(input_list)

            masks, scores, logits = predictor.predict(
                point_coords = temp_list,
                point_labels = input_label,
                multimask_output = False
            )
            # Update masks fields with its points ('-MASK_LIST-')
            # window['-MASK_LIST-'].update(str(temp_list))

            # Add onto mask list 
            mask_list.append(add_mask(masks, scores, logits))
            mask_list_string = f"{mask_list_string}Mask {maskindex}: {mask_point_string} \n"
            #print(mask_list_string)

            window['-MASK_LIST-'].update(mask_list_string)

            #draw the figure onto the image element
            draw_figure(input_element, view_mask(temp_list, hwc, masks, scores, logits))
            
            #Empty the point list
            input_list=[]
            mask_point_string = ''
            maskindex = maskindex + 1

        elif event=='Undo':
            # Remove 1 mask from the list
            print("UNDO")

        elif event=='Reset':
            # Reset the whole thing
            print("RESET")

        elif event=='Save':
            # Delete & Save the masks chosen
            masks = np.array(mask_list)
            delete_all_masks()
            #Loops through the masks, creates a convex hull mask and saves them in a masks folder
            for index in range(len(masks)):
                #chull = convex_hull_image(masks[index], include_borders = True, tolerance=1000)
                #im = Image.fromarray(chull)
                im = Image.fromarray(masks[index])
                i = str(index)

                if len(str(index)) < 2:
                    i= '0'+str(index)
                im.save("masks/m"+ i +".jpg")

            #Shows the shape column
            window['-SHAPE_COL-'].update(visible=True)

        ## Shape Events
        elif event=='Best Poly':
            # Save the final output figure
            output = best_poly_IoU(input)
            draw_figure(input_element, output)
            print("IoU")

        elif event=='Primitive':
            # Save the final output figure
            output = best_primitive(input)
            draw_figure(input_element, output)
            print("Primitive")

        elif event=='Save Output':
            # Save the final output figure
            print("SAVE OUTPUT")
            
    window.close()

if __name__ == '__main__':
    main()