# Credit: https://github.com/aparande/OCR-Preprocessing
import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_image(name):
    image = cv2.imread(name)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def show_images(images, cols=2, titles=None):
    assert titles is None or len(images) == len(titles)
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    fig.tight_layout()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        a.axis('off')
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

def narrow(image, convert_color = False, binarize = True):
    original = image.copy()
    if convert_color:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)     
    if binarize:            
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if np.mean(image) > 127:
            binary = cv2.bitwise_not(image)    
    box_kernel = np.ones((5, 25), np.uint8)
    dilation = cv2.dilate(image, box_kernel, iterations = 1)
    bounds, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in bounds:
        x, y, w, h = cv2.boundingRect(cnt)
        region = original[y:y + h, x:x + w]
        boxes.append(region)
    boxes = sorted(boxes, key=lambda i: -1 * i.shape[0] * i.shape[1])
    return boxes[:3]

def narrow_images(images, convert_color=False, binarize=False):
    narrowed = []
    titles = []
    for img in images:
        regions = narrow(img, convert_color, binarize)
        for region in regions:
            narrowed.append(region)
    return narrowed

def binarize_images(images, black_on_white=False):
    binarized = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)
        
        binarized.append(binary)
    return binarized

def dilate_images(images, kernel = np.ones((1, 1), np.uint8), iterations=1):
    dilated = []
    for img in images:
        img_dilated = cv2.dilate(img, kernel, iterations=iterations)
        dilated.append(img_dilated)
    return dilated

def blur_images(images, blur_weight=1):
    blurred = []
    for dilated in images:
        img_blurred = cv2.medianBlur(dilated, blur_weight)
        blurred.append(img_blurred)
    return blurred