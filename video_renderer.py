import cv2
import os


def render_video(in_dir, out_dir):
    video_name = 'video.avi'
    images = [img for img in os.listdir(in_dir) if img.endswith(".jpg")]
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    frame = cv2.imread(os.path.join(in_dir, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(out_dir + video_name, fourcc, 30.0, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(out_dir, image)))

    cv2.destroyAllWindows()
    video.release()