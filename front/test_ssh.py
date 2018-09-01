import paramiko,os,time

host_ip = '192.168.12.163'
port = 22
username1 = 'jim'
password1 = 'kirovreporting'
local_dir = '/home/pi/front/storePic'
remote_dir = '/home/jim/PycharmProjects/behind/storePic'

transport = paramiko.Transport(host_ip, port)
transport.connect(username=username1, password=password1)
sftp = paramiko.SFTPClient.from_transport(transport)

files = os.listdir(local_dir)

t1 = time.time()
print('###########################')
print('Beginning to upload files from %s'%host_ip)
for f in files:
    print('Uploading file:'+remote_dir+'/'+f)
    sftp.put(os.path.join(local_dir,f), remote_dir+'/'+f) 
print('Upload files success')
print('Used time: %.2f s'%(time.time() - t1))
print('###########################')

transport.close()