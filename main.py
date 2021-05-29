import logging
import os
import sys
import time


from modules.cropperandrotater import FaceCropper

def run(input_path = ""):
    """
    This function for running crop and rotate face.

    Args:
        input_path (str, optional): input folder path.
    """
    face_cropper = FaceCropper()

    # 1. Crop and rotate face
    face_crop_folder = "face_crop"
    if not os.path.exists(face_crop_folder):
        os.mkdir(face_crop_folder)

    logging.info("############ Crop and Rotate ###########")
    face_cropper.cropAndRotateFolderFace(input_path, face_crop_folder)


if __name__ == "__main__":
    
    input_folder = "image_folder"
    run(input_folder)
