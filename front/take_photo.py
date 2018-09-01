import cv2
import os,paramiko,time

num_to_take = 50				#需要拍100张图发去处理
host_ip = '192.168.12.163'			#我的电脑的ip
port = 22					#ssh端口
username1 = 'jim'				#电脑的用户名
password1 = 'kirovreporting'			#用户的密码
local_dir = '/home/pi/front/storePic'		#本地要传的图的路径
remote_dir = '/home/jim/PycharmProjects/behind/storePic'	#接受图片的路径

cap = cv2.VideoCapture(0)			#打开摄像头

transport = paramiko.Transport(host_ip, port)		#输入目标ip 创建sftp对象
transport.connect(username=username1, password=password1)	#输入用户和密码 建立连接
sftp = paramiko.SFTPClient.from_transport(transport)

files = os.listdir(local_dir)			#遍历存图的文件夹下所有文件的名字，存到files列表里

while True:
	t1 = time.time()
	for i in range(num_to_take):			#拍num_to_take张图
	    _, frame = cap.read()
	    cv2.imshow("frame",frame)
	    cv2.waitKey(1)
	    cv2.imwrite("./storePic/test" + str(i) + ".jpg", frame)
	cv2.destroyWindow("frame")
	os.system('touch ./storePic/flag_ok')

	print('###########################')
	print('Beginning to upload files from %s'%host_ip)
	for f in files:
	    print('Uploading file:'+remote_dir+'/'+f)
	    sftp.put(os.path.join(local_dir,f), remote_dir+'/'+f) 
	print('Upload files success')
	print('Used time: %.2f s'%(time.time() - t1))			#打印用时
	print('###########################')
	
	keyboard == cv2.waitKey(0)
	
	if keyboard == ord('q'):break

transport.close()
