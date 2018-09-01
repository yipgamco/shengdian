#-*- coding: utf-8 -*-
import os
import cv2

local_dir = '/home/jim/PycharmProjects/behind'
receive_flag = 0
i=0


while True:
	files = os.listdir(local_dir)
	for f in files:
		if f == 'flag_ok':
			receive_flag = 1
			break

	if receive_flag:
		print('收到了')
		print('做处理')
		os.system('rm flag_ok')
		
		receive_flag = 0

	else:
		
		print(i)
		i += 1
		print('没收到')
