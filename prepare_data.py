import os
import sys
import subprocess
import shutil

# sys.path.append("../")

data_path = '/home/profloo/Documents/KTH'
sequences_list = ''
# video_files=os.listdir(data_path + '/data/videos/')

# extract frames from video clips
args=['ffmpeg', '-i']
for video in video_files:
	video_name = video[:-11]	# remove '_uncomp.avi' from name
	frame_name = 'frame_%d.jpg'	# count starts from 1 by default
	os.mkdir(data_path + '/data/frames/'+video_name)
	args.append(data_path + '/data/frames/'+video_name+'/'+frame_name)
	args.append(data_path + '/data/videos/'+video)
	subprocess.call(args)		# execute the system call
	if (video_files.index(video) + 1) % 50 == 0:
		print 'Completed till video : ', (video_files.index(video) + 1)


print '[MESSAGE]	Frames extracted from all videos'

os.mkdir(data_path + '/data/' + 'TRAIN')
os.mkdir(data_path + '/data/' + 'VALIDATION')
os.mkdir(data_path + '/data/' + 'TEST')

train = [11, 12, 13, 14, 15, 16, 17, 18]
validation =[19, 20, 21, 23, 24, 25, 01, 04]
test = [22, 02, 03, 05, 06, 07, 08, 09, 10]

# read file line by line and strip new lines
lines = [line.rstrip('\n').rstrip('\r') for line in open('sequences_list.txt')]
# remove blank entries i.e. empty lines
lines = filter(None, lines)
# split by tabs and remove blank entries
lines = [filter(None, line.split('\t')) for line in lines]

print lines[-1]
for line in lines:
	vid = line[0]
	subsequences = line[-1].split(',')
	person = int(vid[6:8])
	if person in train:
		move_to = 'TRAIN'
	elif person in validation:
		move_to = 'VALIDATION'
	else:
		move_to = 'TEST'
	for seq in subsequences:
		limits=seq.split('-')
		seq_path=data_path + '/data/' + move_to + '/frame_' + str(limits[0]) + '_' + str(limits[1])
		os.mkdir(seq_path)
		for i in xrange(limits[0], limits[1]+1):
			src = data_path + '/data/' + '/frames/' + vid + '/frame_' + str(i) + '.jpg'
			dst = seq_path + '/frame_' + str(i) + '.jpg'
			shutil.copy(src, dst)

	if (lines.index(line) + 1) % 50 == 0:
		print 'Completed till video : ', (lines.index(line) + 1)

print '[MESSAGE]	Data split into train, validation and test'




