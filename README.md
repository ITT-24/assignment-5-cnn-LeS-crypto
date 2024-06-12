- subset of the HaGRID dataset was placed in the root folder

# 01 - Exploring Hyperparameters 
- Parameter: **data augmentation**
- Logged the model history and confusion matrix as a `.json` in logs-folder 
- Noted down the accuracy_score and inference  in `notes.txt`
- Values, Assumptions, etc. in Section **Exploring**
- Visualized results in Section **Comparison**

# 02 - Gathering a Dataset
- used notebook from [01](#01---exploring-hyperparameters) with original model configs
- Annotation syntax:
  - "Bounding box annotations are proposed in COCO format with normalized relative coordinates"[1]
  - `["top_left_x", "top_left_y", "width", "height"]`
- prediction isn't that good

# 03 - Gesture-based Media Controls
- uses a model created in `gesture-classifier.ipynb` (based on model from notebook of [01](#01---exploring-hyperparameters))
- uses the hand detection from the last assignment to check for hands before predicting the gesture
- confines the webcam frame to a small "detection window" on the left, so the user can just lift their left hand straight up
- if hand detection doesn't work that well, a white, plain background should help
- quit with <kbd>q</kbd>
- requires holding the gesture for a little bit, for less ambiguity 
- Controls:
  - 👍 **volume up** (has to be kinda parallel)
    - volume currently has threshold, so that it doesn't get too loud
  - ✋ **play/pause** (helps when thumb is away from palm and hand is very vertical)
  - ✌️ **skip**
  - "no_gesture" resets the current detection

not implemented: 👎,🤘 

---
[1]: (Kapitanov, A., Kvanchiani, K., Nagaev, A., Kraynov, R., & Makhliarchuk, A. (2024). HaGRID--HAnd Gesture Recognition Image Dataset. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 4572-4581), via: https://arxiv.org/abs/2206.08219 )

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/GaaycKto)
