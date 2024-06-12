from pynput.keyboard import Controller, KeyCode, Key
import cv2
import keras
from keras.src.optimizers import Adam
import numpy as np
from time import sleep

"""
- [ ] (2P) three hand poses are tracked and distinguished reliably
- [x] (1P) three media control features are implemented
- [x] (1P) mapping of gestures to media controls works and makes sense
- [x] (1P) low latency between gesture and the system’s reaction
"""

MODEL_PATH = "gesture_recognition_3.keras"
# LABEL_NAMES = ['dislike', 'no-gesture', 'like', 'peace', 'rock', 'stop'] # not good
# LABEL_NAMES = ['like', 'no_gesture', 'stop'] # works good, but too little gestures
# LABEL_NAMES = ['like', 'no_gesture', 'rock', 'stop'] # kinda works
LABEL_NAMES = ['like', 'no_gesture', 'peace', 'stop'] # works a little better than rock

IMG_SIZE = 64
SIZE = (IMG_SIZE, IMG_SIZE)
COLOR_CHANNELS = 3

BBOX_THRESHOLD = 180

VOLUME_TRHESHOLD = 10 # the prevent the volume to get too loud (esp. testing)
GESTURE_HOLD = 5 # to prevent executing actions on gesture change
COOLDOWN = 5

WINDOW_NAME = "CONTROL"
cap = cv2.VideoCapture(0)


class Model:

    def __init__(self):
        self.load_model()
        pass

    def load_model(self):
        # NOTE: current model is kinda bad
        self.model = keras.models.load_model(MODEL_PATH, compile=False) # compile=True returns error
        # kinda like this: https://stackoverflow.com/q/62707558, https://programmerah.com/keras-nightly-import-package-error-cannot-import-name-adam-from-keras-optimizers-29815/
            # https://stackoverflow.com/q/78323015
        print("Using", self.model)


    def detect_hand(self, frame):
        """ via: assignment 04: AR-game.py
        Detect the hand and get its bbox
        """

        # make the contrast bigger to "wash out" lighter areas
        # see: https://docs.opencv.org/4.x/d3/dc1/tutorial_basic_linear_transform.html
        # see: https://stackoverflow.com/a/56909036
        alpha = 3.0 # contrast (1.0 - 3.0)
        beta = -100 # brightness control (0-100)
        contrast = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        # maximize brightness of paper
        lab = cv2.cvtColor(contrast, cv2.COLOR_BGR2LAB)
        l_channel = lab[:,:,0] # maximize brightness of paper
        # changed from: https://stackoverflow.com/a/72264323

        # blur for a smother image
        blur = cv2.GaussianBlur(l_channel,(3,3),0)

        # use automatic thresholding: see https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html (Otsu's Binarization)
        t_min = 40 # 0 # exclude extremes
        t_max = 255
        ret, thresh = cv2.threshold(blur, t_min, t_max, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # ret, thresh = cv2.threshold(blur, t_min, t_max, cv2.THRESH_BINARY)
        thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)
        # use adaptive threshold to then better extract the contours
        
        # MORPH -> to better "connect" the contours
        kernel_size = (7, 7)
        kernel_small = (1, 1)
        kernel = np.ones(kernel_size, dtype=np.float64)

        erotion = cv2.erode(thresh, kernel)
        dilation = cv2.dilate(thresh, kernel)
        opening = cv2.dilate(erotion, kernel_small)

        # detect edges (see: https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)
        c_min = 10
        c_max = 200
        canny = cv2.Canny(erotion, c_min, c_max)

        # detect contours 
        # (see: https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)
        # (see: https://www.geeksforgeeks.org/find-and-draw-contours-using-opencv-python/)
        contours, _ = cv2.findContours(erotion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        c = max(contours, key=cv2.contourArea)
        
        # draw bounding box (see: https://stackoverflow.com/a/23411041)
        rect = cv2.boundingRect(c)
        x,y,w,h = rect

        hand = False
        if h >= BBOX_THRESHOLD:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,'Hand Detected',(x+w+10,y+h),0,0.3,(0,255,0))
            hand = True

        output = frame
        # cv2.drawContours(output, contours,  -1, (0, 255, 0), 2)
        cv2.imshow(WINDOW_NAME, output)

        return hand


    def predict_gesture(self, frame):
        cropped = self.crop(frame)

        found_hand = self.detect_hand(cropped)
        
        label = "no_gesture"
        if found_hand:
            resize = self.resize(cropped)
            prediction = self.model.predict(resize, verbose=0)
            try:
                label = LABEL_NAMES[np.argmax(prediction)]
            except:
                label = "no_gesture"

        return label


    def crop(self, frame):
        height, width, _ = frame.shape 
        # frame = frame[(height//4):(height-(height//4)), (width//3):(width-(width//3))]
        frame = frame[(height//4):(height-(height//4)), (width-(width//3)):width]
        return frame
        # cv2.imshow(WINDOW_NAME, frame)

    def resize(self, frame): # to fit model
        resize = cv2.resize(frame, SIZE)
        reshape = resize.reshape(-1, IMG_SIZE, IMG_SIZE, COLOR_CHANNELS)
        return reshape



class KeyController:
    # VOLUME_TRHESHOLD = 5 # the prevent the volume to get too loud (esp. testing)
    # GESTURE_HOLD = 3 # to prevent executing actions on gesture change

    def __init__(self):
        self.key = Controller()
        self.dislike = 'dislike'
        self.no_gesture = 'no_gesture'
        self.like = 'like'
        self.peace = 'peace'
        self.rock = 'rock'
        self.stop = 'stop'
        self.i = 0 # to threshold volume 
        self.prev_gesture = self.no_gesture
        self.gesture_amount = 0 
        self.is_stopped = True # prevent ambigoutiy btw. pause and play


    def parse_gesture(self, label):
        """Execute the actions corresponding to the gestures.
           With some extra checks against spam execution.
        """

        # print(label)

        # skip executing the same gesture over and over again
        if label == self.no_gesture:
            self.prev_gesture = label # reset
            return
        # if self.prev_gesture == label or label == self.no_gesture:
        #     # print("[HOLD] gesture", self.prev_gesture, "==", label)
        #     return
        elif self.prev_gesture == label:
            return
        else:
            if self.gesture_amount >= GESTURE_HOLD:
                self.prev_gesture = label
                self.gesture_amount = 0
                print("[FOUND]:\t", label)
            else:
                self.gesture_amount += 1
                print("...detecting...")
                return

        match label:
            case self.dislike:
                self.decrease_volume()
            case self.like:
                self.increase_volume()
            case self.rock:
                # self.start_track()
                self.skip_track()
            case self.stop:
                # self.pause_track()
                self.start_stop_track()
            case self.peace:
                self.skip_track()
            case _:
                pass

        sleep(1) # to prevent an instant new (wrong) detection

    def start_stop_track(self):
        print("[CTRL]:\t start/stop")
        self.key.press(Key.media_play_pause)

    def skip_track(self): # peace
        print("[CTRL]:\t skip track")
        self.key.press(Key.media_next)
        pass

    def increase_volume(self): # like
        print("[CTRL]:\t increase volume")
        if self.i < VOLUME_TRHESHOLD: # so the volume doesn't get too loud accidentally
            self.key.press(Key.media_volume_up)
            self.i += 1
        pass

    def decrease_volume(self): # dislike
        print("[CTRL]:\t decrease volume")
        if self.i > 0: 
            self.key.press(Key.media_volume_down)
            self.i -= 1
        pass

    # TODO: check if something is currently playing, so rock is defenetly start
    # using pyaudio see: https://stackoverflow.com/a/70493551

    # def start_track(self): # rock
    #     if self.is_stopped:
    #         self.key.press(Key.media_play_pause)
    #         self.is_stopped = False
    #     pass

    # def pause_track(self): # stop
    #     if not self.is_stopped:
    #         self.key.press(Key.media_play_pause)
    #         self.is_stopped = True
    #     pass



if __name__ == "__main__":

    
    model = Model()
    key_controller = KeyController()

    while(True):
        ret, frame = cap.read()

        # model.detect_hand(frame)
        gesture = model.predict_gesture(frame)
        key_controller.parse_gesture(gesture)

        # cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # cap.release()

    cap.release()
    cv2.destroyAllWindows()