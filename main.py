import matlab.engine
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from face_aligner.align_faces import align_faces
from frame_extractor import extract_frames
from video_renderer import render_video
from classification.Net import Net


inputFile = 'fraud13.mp4'

inputDir = 'video_in/'
inputPath = inputDir + inputFile
eng = matlab.engine.start_matlab()
net = Net()

net.classify('./temp/5_slice/' + inputFile + '/yt_slice.jpg')
exit()

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


eng.cd('.\eulerian_video_magnification')
inputPath = '../temp/2_align/' + inputFile + '/video.avi'
motAttFile = '../temp/2_align/' + inputFile + '/moAttFile.avi'

# 5 - stabilize video
print('STEP 6: stabilize video')
eng.motionAttenuateFixedPhase(inputPath, motAttFile, nargout=0)

# 6 - magnify video
print('STEP 6: magnify video')
eng.amplify_spatial_Gdown_temporal_ideal(motAttFile, '../temp/4_magnify/' + inputFile + '/', float(50), float(4),
                                         0.83, 1,
                                         float(30), float(1), nargout=0)


# 6 - extract slice
print('STEP 6: extract slice')
eng.cd('..\slicer')
# eng.xt_slicer('../temp/4_magnify/' + inputFile + '/video.avi', '../temp/5_slice/' + inputFile + '/', nargout=0)
eng.yt_slicer('../temp/4_magnify/' + inputFile + '/moAttFile.avi', '../temp/5_slice/' + inputFile + '/', nargout=0)

# 7 - classify slice
print('STEP 7: classify slice')
net.classify('./temp/5_slice/' + inputFile + '/yt_slice.jpg')