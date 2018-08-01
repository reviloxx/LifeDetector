import matlab.engine
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from face_aligner.align_faces import align_faces
from frame_extractor import extract_frames
from video_renderer import render_video

inputFile = 'fraud7.mov'

inputDir = 'video_in/'
inputPath = inputDir + inputFile
eng = matlab.engine.start_matlab()

# 1 - extract 10 second subclip
print('STEP 1: extracting 10 second subclip')
ffmpeg_extract_subclip(inputPath, 0.00, 10.00, targetname='temp/1_crop/' + inputFile)

# 2 - extract frames from subclip
print('STEP 2: extracting frames from subclip')
extract_frames("./temp/1_crop/" + inputFile, './temp/2_align/' + inputFile + '/frames/')

# 3 - crop and align face
print('STEP 3: crop and align face')
path = './temp/2_align/' + inputFile + '/frames/'
out_dir = './temp/2_align/' + inputFile + '/'
align_faces(path, out_dir)

# 4 - generate new video from cropped face frames
print('STEP 4: generate new video from cropped face frames')
render_video('./temp/2_align/' + inputFile + '/', './temp/2_align/' + inputFile + '/')

# 5 - magnify video
print('STEP 5: magnify video')
eng.cd('.\eulerian_video_magnification')
inputPath = '../temp/2_align/' + inputFile + '/video.avi'
eng.amplify_spatial_Gdown_temporal_ideal(inputPath, '../temp/4_magnify/' + inputFile + '/', float(50), float(4),
                                         50 / 60, 60 / 60,
                                         float(30), float(1), nargout=0)

# eng.amplify_spatial_Gdown_temporal_ideal('../video_in/fraud6.mov', '../temp/4_magnify/' + inputFile + '/',
#                                          float(50), float(4), 50 / 60, 60 / 60, float(30), float(1), nargout=0)

# 6 - extract slice
print('STEP 6: extract slice')
eng.cd('..\slicer')
eng.xt_slicer('../temp/4_magnify/' + inputFile + '/video.avi', '../temp/5_slice/' + inputFile + '/', nargout=0)
eng.yt_slicer('../temp/4_magnify/' + inputFile + '/video.avi', '../temp/5_slice/' + inputFile + '/', nargout=0)

# 6 - classify slice
