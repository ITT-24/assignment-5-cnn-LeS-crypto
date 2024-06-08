import cv2
import argparse
import os
import numpy as np
import json
# reshape image to (1920(h)x1080(w)px)
# get normalized points

WINDOW_NAME = "Annot Helper"
FIXED_W = 1080
FIXED_H = 1920
LABELS = {0: "no-gesture", 1: "like", 2: "dislike", 3: "stop", 4: "rock", 5: "peace"}

def parse_cmd_input():
    """
    Read and parse the command line parameters.
    See: https://docs.python.org/3/library/argparse.html
    """

    # init path variables
    source = None
    destination = None
    res = None

    parser = argparse.ArgumentParser( 
        prog="annot-helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Reshape an images to 1920x1080px and select bbox",
        epilog="""----- Keyboard Shortcuts -----\n
        ESC \t - Reset the image 
        s \t - save the resized image
        n \t - go to the next image
        q \t - quit the application
        """
    )

    parser.add_argument('-n', '--name', type=str, metavar='',
                        default="name",
                        required=True,
                        action="store",
                        help="""Your name to create a annotation file""")

    parser.add_argument('-s', '--source', type=str, metavar='',
                        default="img",
                        required=False,
                        action="store",
                        help="""The path to the folder you want to use. 
                            Defaults to the included 'img' in the current folder""")
    
    args = parser.parse_args()

    if args.source:
        source = args.source
        print("image directory:", source)

    if args.name:
        name = args.name
        print("annotating for", name)
        
    return source, name

class Extractor:
    def __init__(self, source, name) -> None:
        self.annotations = {}
        self.bboxes = []
        self.clicks = []
        self.name = name

        self.source_dir = source
        self.image_list = [image_name for image_name in os.listdir(self.source_dir)] # see: https://stackoverflow.com/q/59270464
        self.idx = 0

        self.filepath = f"{self.source_dir}/{self.image_list[self.idx]}"
        self.img = cv2.imread(self.filepath)

        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)

    def show_window(self):
        self.resize_img()
        cv2.imshow(WINDOW_NAME, self.img)

    def resize_img(self, save=False):
        w = FIXED_W // 2
        h = FIXED_H // 2
        if save:
            img =  cv2.resize(self.img, (FIXED_W, FIXED_H))
            filepath = f"{self.source_dir}/r-{self.image_list[self.idx]}"
            cv2.imwrite(filepath, img)
        else:
            self.img = cv2.resize(self.img, (w, h)) # (100, 50) = 100 w & 50 h


    def next_image(self):
        if self.idx < len(self.image_list):
            self.idx += 1
            self.filepath = f"{self.source_dir}/{self.image_list[self.idx]}"
            self.bboxes = []
            self.clicks = []
            print("next image:", self.filepath)
            self.img = cv2.imread(self.filepath)
        else: 
            print("this is the last image")

    def mouse_callback(self, event, x, y, flags, params):
        """Allow 4 mouse clicks before annotation is done"""
        # see: opencv_click.py
        global img
        bbox = None

        if event == cv2.EVENT_LBUTTONDOWN:
            img = cv2.circle(self.img, (x, y), 5, (255, 0, 0), -1)
            cv2.imshow(WINDOW_NAME, img)
            self.clicks.append([x, y])

            if len(self.clicks) == 4:
               bbox = self.get_normalized_bbox()
               self.bboxes.append(bbox)
               

    def get_normalized_bbox(self):
        """BUG: only gets the right bounding box, when the mouseclicks go:
           top-left, top-right, bottom-right, bottom-left. 
           Too lazy to debug...
        """
        print(self.clicks)
        id_tl = np.argmin(np.sum(self.clicks, axis=1)) # get top_left corner
        id_br = np.argmax(np.sum(self.clicks, axis=1)) # got bot_right corner

        top_left = self.clicks[id_tl]
        bot_right = self.clicks[id_br]
        print(top_left)

        # cv2: 0 = height, 1 = width        
        x = top_left[0] / (self.img.shape[1] - 1)
        y = top_left[1] / (self.img.shape[0] - 1)
        width = (bot_right[0] / (self.img.shape[1] - 1) ) - x
        height = (bot_right[1] / (self.img.shape[0] - 1) ) - y

        print("bbox = ", x, y, width, height)
        print("unnormalized", top_left[0], top_left[1], bot_right[0] - top_left[0], bot_right[1] - top_left[1])
        print("image-shape", self.img.shape)

        self.clicks = []

        return [x, y, width, height]

    def add_annotations(self):
        img_id = self.image_list[self.idx]
        labels = []

        print(len(self.bboxes))

        print(f"Input a custom label or use the ids for preexisting labels.")
        for key in LABELS:
            print(f"ID: {key} -> {LABELS[key]}")
        print("Then press ENTER")

        for i in range(0, len(self.bboxes)):
            label = input(f"Label for {i} bounding box: ")
            if LABELS[int(label)]:
                label = LABELS[int(label)]
            labels.append(label)

        anot = {
            "bboxes": self.bboxes,
            "labels": labels,
        }

        print(anot)

        self.annotations[img_id] = anot
        # print(self.annotations)

    def save_annotations(self):
        print("save")
        with open(f"annot-{self.name}.json", 'a') as f:
            json.dump(self.annotations, f, indent=4)

        
"""shape
{
    "0534147c-4548-4ab4-9a8c-f297b43e8ffb": {
    "bboxes": [
        ["top_left_x", "top_left_y", "width", "height"],
        [0.38038597, 0.74085361, 0.08349486, 0.09142549],
        [0.67322755, 0.37933984, 0.06350809, 0.09187757]
    ],
    "labels": [
        "no_gesture",
        "one"
    ]
    }   
}
"""

    

# ----- RUN ----- #

if __name__ == "__main__":

    source, name = parse_cmd_input()
    extractor = Extractor(source, name)

    # PROGRAMM LOOP
    while(True):
        extractor.show_window()
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            extractor.resize_img(True)
        elif key == ord('n'):
            extractor.next_image()
        elif key == ord('a'):
            extractor.add_annotations()
        elif key == ord('s'):
            extractor.save_annotations()
        # elif key == 27: # ESC
        #     extractor.reset_image()
        # close the window with "window-x-button" (mouse)
        elif cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            # see: https://stackoverflow.com/a/63256721
            break
    cv2.destroyAllWindows()