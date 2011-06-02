import os, time
ctr = 0;
while True:
    ctr += 1
    localtime = time.asctime(time.localtime(time.time()))
    print localtime + ' : ltp_v2.0 create ltp_server for %d time' % ctr
    try:
        os.system('ltp_server.py')
    except Exception, msg:
        continue

