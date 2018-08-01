import re
import subprocess
import cv2
import subprocess
import shlex
import json
import ntpath
import os


def get_rotation(file_path_with_file_name):
    print(file_path_with_file_name)
    """
    Function to get the rotation of the input video file.
    Adapted from gist.github.com/oldo/dc7ee7f28851922cca09/revisions using the ffprobe comamand by Lord Neckbeard from
    stackoverflow.com/questions/5287603/how-to-extract-orientation-information-from-videos?noredirect=1&lq=1

    Returns a rotation None, 90, 180 or 270
    """

    cmd = 'C:/ffmpeg/bin/ffprobe.exe -loglevel error -select_streams v:0 -show_entries stream_tags=rotate -of default=nw=1:nk=1'
    args = shlex.split(cmd)
    args.append(file_path_with_file_name)
    # run the ffprobe process, decode stdout into utf-8 & convert to JSON
    ffprobe_output = subprocess.check_output(args).decode('utf-8')
    if len(ffprobe_output) > 0:  # Output of cmdis None if it should be 0
        ffprobe_output = json.loads(ffprobe_output)
        rotation = ffprobe_output

    else:
        rotation = 0

    print(rotation)
    return rotation


def extract_frames(input_file, out_dir):
    filename = ntpath.basename(input_file)
    rotation = get_rotation("C:/LifeDetector/temp/1_crop/" + filename)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    vidcap = cv2.VideoCapture(input_file)
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        scale = 1.0
        if rotation == 90:
            M = cv2.getRotationMatrix2D(center, 270, scale)
            image = cv2.warpAffine(image, M, (w, h))
        elif rotation == 270:
            M = cv2.getRotationMatrix2D(center, 90, scale)
            image = cv2.warpAffine(image, M, (w, h))
        elif rotation == 180:
            M = cv2.getRotationMatrix2D(center, 180, scale)
            image = cv2.warpAffine(image, M, (w, h))

        cv2.imwrite(out_dir + str(count).zfill(3) + ".jpg", image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Extracted frames: ' + str(count))
        count += 1
