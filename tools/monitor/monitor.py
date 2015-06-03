#!/usr/bin/env python
import subprocess
import sys
import time
import datetime
from smtplib import SMTP
from email.MIMEText import MIMEText
import socket

from config import hosts, remote_exe_path, remote_exe, \
        from_addr, to_addrs, \
        mail_username, mail_password, mail_server, mail_server_port

import logging
logging.basicConfig(
        filename='ltp_monitor.log',
        format='[%(levelname)s] %(asctime)s : %(message)s',
        level=logging.INFO)

try:
    conn = SMTP(mail_server, mail_server_port)
    conn.set_debuglevel(False)
    conn.login(mail_username, mail_password)
except:
    logging.warning("Failed to logging smtp server.")
    conn = None


mail_content_header = """\
Report from LTP monitoring script
=================================

please take care of this message.

"""

mail_content_tail = """this mail is automatically generate from ltp-monitor.py
if you have any questions, please contract yjliu # ir.hit.edu.cn
wish you can solve this bug :)

best regrads
ltp-monitor.py"""

def send_server_unreachable_mail(h):
    # send email to notify that the server is unreachable
    current_time = datetime.datetime.now().strftime("%I:%M %p on %B %d, %Y")
    subject = "{time}, host {host} is unreachable.".format(
            time = current_time, 
            host=h["name"])
    content = mail_content_header
    content += "Report time : {time}\n".format(time = current_time)
    content += "Report host : {host}\n".format(host = h["name"])
    content += "Report content : ssh connection to on {host} failed.\n".format(host = h["name"])
    content += "\n"

    msg = MIMEText(content)
    msg['Subject']= subject
    msg['From'] = from_addr

    try:
        conn.sendmail(from_addr, to_addrs, msg.as_string())
    except:
        logging.warning("Failed to send notification mail")


def send_server_crash_mail(h):
    # send mail to notify that the service is crash
    current_time = datetime.datetime.now().strftime("%I:%M %p on %B %d, %Y")
    subject = "{time}, LTP on host {host} is crash.".format(
            time = current_time,
            host=h["name"])
    content = mail_content_header
    content += "Report time : {time}\n".format(time = current_time)
    content += "Report host : {host}\n".format(host = h["name"])
    content += "Report content : LTP on {host} is crash.\n".format(
            host = h["name"])
    content += "\n"

    if h["case"] is not None:
        detail = "case : \"{case}\"\n".format(case = h["case"])
    else:
        detali = "case : problem case was not detected.\n"

    if h["msg"] is not None:
        detail = "message : {msg}\n".format(msg = h["msg"])
    else:
        detali = "message : no message is left.\n"

    content += detail
    content += "\n"
    content += mail_content_tail

    msg = MIMEText(content)
    msg['Subject']= subject
    msg['From'] = from_addr

    try:
        conn.sendmail(from_addr, to_addrs, msg.as_string())
    except:
        logging.warning("failed to send mail.")


def send_server_status_mail():
    msg = "Server status summary\n"
    msg += "host\tstatus"
    for h in hosts:
        msg += "{host}\t{status}".format(
                host = h["name"],
                status = h["status"])
    conn.sendmail(from_addr, to_addrs, msg)


def epoll(cmd):
    # run command
    p = subprocess.Popen(cmd,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)

    status = 1
    sleeptime = 0
    TIMEOUT=50

    while status != 0 and sleeptime < TIMEOUT:
        status = p.poll()
        time.sleep(0.1)
        sleeptime += 1

    output, stderr = p.communicate()

    if status != 0:
        return None
    else:
        return output

def connect_test(h):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(10)

    try:
        s.connect( (h["ip"], 22) )
    except:
        send_server_unreachable_mail(h)
        return False

    return True

def running_test(h):
    cmd = "ps aux | grep {exe} | grep -v grep".format(
            exe = remote_exe)
    cmd2 = "ssh {ip} \"{cmd}\"".format(
            ip = h["ip"], 
            cmd = cmd)
    info = epoll(cmd2)
    return info

def run():
    while True:
        #test if is running
        for h in hosts:
            if not connect_test(h):
                continue

            # test if this service is running
            info = running_test(h)

            if info is None:
                h["status"] = False
                logging.warning("LTP on {host} is not running".format(
                        exe = remote_exe, 
                        host = h["name"]))

                # detect the failed case
                cmd = "cd {path}; tail -200 {host}.server.log | grep 'CDATA:' | tail -1".format(
                        path = remote_exe_path,
                        host = h["name"])
                cmd2 = "ssh {ip} \"{cmd}\"".format(
                        ip = h["ip"],
                        cmd = cmd)
                ret = epoll(cmd2)
                h["case"] = ret

                # restart the service
                cmd = "cd {path}; nohup {exe} --threads 4 >> {host}.server.log 2>&1 &".format(
                        path = remote_exe_path, 
                        exe = remote_exe, 
                        host = h["name"])
                cmd2 = "ssh {ip} \"{cmd}\"".format(
                        ip = h["ip"], 
                        cmd = cmd)
                ret = epoll(cmd2)

                # test again
                ret = running_test(h)

                if ret is None:
                    h["msg"] = "Failed to restart LTP on {host}.".format(
                            host=h["name"])
                else:
                    h["msg"] = "Success to restart LTP on {host}.".format(
                            host=h["name"])

                send_server_crash_mail(h)
            else:
                logging.info("{exe} on {host} is running".format(
                        exe = remote_exe,
                        host = h["name"]))
                h["status"] = True

        time.sleep(5*60)

def stop(h):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(10)

    # check ssh connection
    if not connect_test(h):
        return

    info = running_test(h)
    if info is None:
        logging.warning("LTP on {host} is not running.".format(host=h["name"]))
        return

    pids = " ".join([line.split()[1] for line in info.strip().split("\n")])
    cmd = "kill -9 {pids}".format(pids = pids)
    cmd2 = "ssh {ip} \"{cmd}\"".format(
            ip = h["ip"], 
            cmd = cmd)
    info = epoll(cmd2)
    logging.info("LTP on {host} is stopped.".format(host=h["name"]))

def start(h):
    if not connect_test(h):
        return

    info = running_test(h)
    if info is not None:
        logging.info("LTP on {host} is already running.".format(host=h["name"]))
        return

    cmd = "cd {path}; nohup {exe} --threads 4 >> {host}.server.log 2>&1 &".format(
            path = remote_exe_path, 
            exe = remote_exe, 
            host = h["name"])
    cmd2 = "ssh {ip} \"{cmd}\"".format(
            ip = h["ip"], 
            cmd = cmd)
    ret = epoll(cmd2)
    logging.info("LTP on {host} is started.".format(host=h["name"]))

def test(h):
    if not connect_test(h):
        return

    info = running_test(h)
    if info is not None:
        logging.info("LTP on {host} is running.".format(host=h["name"]))
    else:
        logging.warning("LTP on {host} is not running.".format(host=h["name"]))

def stopall():
    for h in hosts:
        stop(h)

def startall():
    for h in hosts:
        start(h)

def testall():
    for h in hosts:
        test(h)

if __name__=="__main__":
    usage = "monitor.py {run|startall|stopall|testall|start [host]|stop [host]|test [host]|list}"

    if len(sys.argv) < 2:
        print >> sys.stderr, usage
        sys.exit(1)

    if sys.argv[1] in ["start", "stop", "test"] and len(sys.argv) == 2:
        print >> sys.stderr, usage
        sys.exit(1)

    if sys.argv[1] == "run":
        run()
    elif sys.argv[1] == "startall":
        startall()
    elif sys.argv[1] == "stopall":
        stopall()
    elif sys.argv[1] == "testall":
        testall()
    elif sys.argv[1] == "start":
        for h in hosts:
            if h["name"] == sys.argv[2]:
                start(h)
    elif sys.argv[1] == "stop":
        for h in hosts:
            if h["name"] == sys.argv[2]:
                stop(h)
    elif sys.argv[1] == "test":
        for h in hosts:
            if h["name"] == sys.argv[2]:
                test(h)
    elif sys.argv[1] == "list":
        for h in hosts:
            print >> sys.stderr, "*", h["name"]
    else:
        print >> sys.stderr, usage
