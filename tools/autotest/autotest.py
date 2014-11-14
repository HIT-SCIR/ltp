#!/usr/bin/env python
import os
import sys
import time
import logging
import subprocess
import cStringIO as StringIO
import tempfile
from ConfigParser import ConfigParser
from optparse import OptionParser

FORMAT = "[%(levelname)5s] %(asctime)-15s - %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S")

DUMMY = open(os.devnull, "w")
TMPDIR= tempfile.gettempdir()
SRC_EXTENSIONS = (".h", ".hpp", ".c", ".cpp")
SRC_EXLUDES = ("mongoose.h", "mongoose.c")
FINISHED_JOBS= set([])

def which(program):
    # From http://stackoverflow.com/questions/377017
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None


def static_code_check(rootdir, outputdir, verbose=False):
    '''
    Run the static code check provided by `cppcheck`. This script first generate a
    list of files.

    Parameters
    ----------
    rootdir: str
        The root dir of LTP project.
    outputdir: str
        The detail output dir.
    '''
    if os.name == 'nt':
        logging.info("cppcheck: windows cppcheck is not supported.")
        return

    if not which("cppcheck"):
        logging.warn("cppcheck not installed.")
        logging.info("skip static code checking.")
        return

    sourcedir = os.path.join(opts.rootdir, "src")
    sourcelist_path = os.path.join(TMPDIR, "ltp.autotest.sourcefile.list")
    sourcelist_ofs = open(sourcelist_path, "w")
    for path, subdirs, files in os.walk(sourcedir, topdown=True):
        for filename in files:
            if os.path.splitext(filename)[1] in SRC_EXTENSIONS and filename not in SRC_EXLUDES:
                print >> sourcelist_ofs, os.path.join(path, filename)
    sourcelist_ofs.close()

    command = ["cppcheck", "--enable=all", "--file-list=%s" % sourcelist_path]
    logging.info("cppcheck: static check is running.")

    report_path=os.path.join(outputdir, "cppcheck.report")
    ofs= open(report_path, "w")
    cppcheck = subprocess.call(command, stdout=ofs, stderr=ofs)
    ofs.close()

    ifs= open(report_path, "r")
    nr_style, nr_performance, nr_warning = 0, 0, 0
    for line in ifs:
        if "(style)" in line:
            nr_style += 1
        if "(performance)" in line:
            nr_performance += 1
        if "(warning)" in line:
            nr_warning += 1

    logging.info("cppcheck: static check is done.")
    logging.info("cppcheck: found %d style comments." % nr_style)
    logging.info("cppcheck: found %d performance comments." % nr_performance)
    logging.info("cppcheck: found %d warning comments." % nr_warning)
    ifs.close()
    global FINISHED_JOBS
    FINISHED_JOBS.add("cppcheck")


def executable_check(rootdir, outputdir, input_path, verbose=False):
    ltp_test_exe = os.path.join(rootdir, "bin/ltp_test")
    if os.name == 'nt':
        ltp_test_exe += '.exe'

    if not which(ltp_test_exe):
        logging.error("ltp_test: ltp_test is not executable.")
        logging.info("ltp_test: all dynamic checks are skipped.")
        return False

    original_config_path = os.path.join(os.path.join(rootdir, "conf"), "ltp.cnf")
    cfg_str = '[root]\n' + open(original_config_path, "r").read()
    cfg = ConfigParser()
    cfg.readfp(StringIO.StringIO(cfg_str))

    config_path = os.path.join(TMPDIR, "ltp.autotest.ltp.conf")
    cofs = open(config_path, "w")
    print >> cofs, "target = all"

    def concatenate(name):
        model = cfg.get("root", name)
        if not model.startswith("/"):
            print >> cofs, ("%s = %s" % (name, os.path.join(rootdir, model)))
    concatenate("segmentor-model")
    concatenate("postagger-model")
    concatenate("parser-model")
    concatenate("ner-model")
    concatenate("srl-data")
    cofs.close()

    command = [ltp_test_exe, config_path, "srl", input_path]
    logging.info("ltp_test: dynamically executable check is running.")

    ofs= open(os.path.join(outputdir, "output.txt"), "w")
    subprocess.call(command, stdout=ofs, stderr=DUMMY)
    ofs.close()
    logging.info("ltp_test: dynamically executable check is done.")
    global FINISHED_JOBS
    FINISHED_JOBS.add("ltp_test")
    return True


def memory_leak_check(rootdir, outputdir, input_path, verbose=False):
    if os.name == 'nt':
        logging.info("memcheck: windows memcheck is not supported.")
        return

    if not which("valgrind"):
        logging.error("memcheck: valgrind is not installed")
        logging.info("memcheck: memcheck is skipped.")
        return
    ltp_test_exe = os.path.join(rootdir, "bin/ltp_test")
    config_path = os.path.join(TMPDIR, "ltp.autotest.ltp.conf")
    if not os.path.isfile(config_path):
        logging.error("memcheck: config file is not generated.")
        logging.info("memcheck: memcheck is skipped.")
        return
    command = ["valgrind", "--tool=memcheck", "--leak-check=full",
            ltp_test_exe, config_path, "srl", input_path]

    memcheck_log_path= os.path.join(outputdir, "memcheck.log")
    memcheck_lfs= open(memcheck_log_path, "w")
    logging.info("memcheck: valgrind memory leak check is running.")
    subprocess.call(command, stdout=DUMMY, stderr=memcheck_lfs)
    logging.info("memcheck: valgrind memory check is done.")
    memcheck_lfs.close()
    ifs = open(memcheck_log_path, "r")
    for line in ifs:
        line = line.strip().split("==")[-1].strip()
        if ("definitely lost:"    in line or
                "indirectly lost:"in line or
                "possibly lost:"  in line or
                "still reachable:"in line or
                "suppressed:"     in line):
            line = line.split("==")[-1].strip()
            logging.info("memcheck: %s" % line)
    ifs.close()
    global FINISHED_JOBS
    FINISHED_JOBS.add("memcheck")


def callgrind_check(rootdir, outputdir, input_path, verbose=False):
    if os.name == 'nt':
        logging.info("callgrind: windows calgrind check is not supported.")
        return

    if not which("valgrind"):
        logging.error("callgrind: valgrind is not installed.")
        logging.info("callgrind: callgrind check is skipped.")
        return
    ltp_test_exe = os.path.join(rootdir, "bin/ltp_test")
    config_path = os.path.join(TMPDIR, "ltp.autotest.ltp.conf")
    if not os.path.isfile(config_path):
        logging.error("callgrind: config file is not generated.")
        logging.info("callgrind: memcheck is skipped.")
        return
    callgrind_out_file = os.path.join(TMPDIR, "ltp.autotest.callgrind.out")
    command = ["valgrind", "--tool=callgrind", "--callgrind-out-file=%s" % callgrind_out_file,
            ltp_test_exe, config_path, "srl", input_path]
    callgrind_lfs = open(os.path.join(outputdir, "callgrind.log"), "w")
    logging.info("callgrind: valgrind callgrind check is running.")
    subprocess.call(command, stdout=DUMMY, stderr=callgrind_lfs)
    logging.info("callgrind: valgrind callgrind check is done.")
    callgrind_lfs.close()

    gprof2dot_script = os.path.join(
            os.path.split(os.path.abspath(__file__))[0],
            "gprof2dot.py")

    if not os.path.isfile(gprof2dot_script):
        logging.error("callgrind: gprof2dot.py not found.")
        logging.error("callgrind: callgrind visualization is cancealed.")
        return

    if not which("dot"):
        logging.error("callgrind: graphviz not installed.")
        logging.info("callgrind: callgrind visualization is cancealed.")
        return

    gprof2dot_dot = os.path.join(TMPDIR, "ltp.autotest.dot")
    command = ["python", gprof2dot_script, "--format=callgrind",
            "--output=%s" % gprof2dot_dot, "-w", callgrind_out_file]
    logging.info("callgrind: gprof2dot.py converting callgrind output to dot.")
    subprocess.call(command, stdout=DUMMY, stderr=DUMMY)
    logging.info("callgrind: gprof2dot.py converting callgrind output to dot is done.")

    command = ["dot", "-Tpng", "-o",  os.path.join(outputdir, "graph.png"), gprof2dot_dot]
    logging.info("callgrind: dot converting dot output to PNG.")
    subprocess.call(command, stdout=DUMMY, stderr=DUMMY)
    logging.info("callgrind: dot converting dot output to PNG is done.")

    global FINISHED_JOBS
    FINISHED_JOBS.add("callgrind")


def speed_check(rootdir, outputdir, input_path, verbose=False):
    if os.name == 'nt':
        logging.info("speed: windows speed check is not supported.")
        return

    def build(exe_prefix, model_prefix):
        exe = os.path.join(rootdir, "bin", "examples", ("%s_cmdline" % exe_prefix))
        model = os.path.join(rootdir, "ltp_data", ("%s.model" % model_prefix))
        out = os.path.join(TMPDIR, "ltp.autotest.%s.out" % exe_prefix)
        return (exe, model, out)
    cws_cmdline, cws_model, cws_out = build("cws", "cws")
    pos_cmdline, pos_model, pos_out = build("pos", "pos")
    par_cmdline, par_model, par_out = build("par", "parser")

    if not input_path:
        logging.error("speed: input not specified.")
        logging.info("speed: speed check is canceled.")
        return 

    nr_sz = os.stat(input_path).st_size
    dataset = open(input_path,"r").readlines()
    nr_lines = len(dataset)
    avg_sent_len = float(sum([len(data.decode("utf-8")) for data in dataset]))/nr_lines
    logging.info("speed: average sentence length %f" % avg_sent_len)
    def check(exe):
        if not which(exe):
            logging.error("speed: %s is not found." % exe)
            logging.info("speed: speed check is canceled.")
            return False
        return True

    if not check(cws_cmdline):
        return
    if not check(pos_cmdline):
        return
    if not check(par_cmdline):
        return

    if not os.path.isfile(input_path):
        logging.error("speed: input is not specified.")
        logging.info("speed: speed check is canceled.")
        return

    speed_log = os.path.join(outputdir, "speed.log")
    lfs = open(speed_log, "w")
    def run(exe, model, ifs, ofs):
        subprocess.call([exe, model], stdin=ifs, stdout=ofs, stderr=lfs)
        ifs.close()
        ofs.close()

    run(cws_cmdline, cws_model, open(input_path, "r"), open(cws_out, "w"))
    run(pos_cmdline, pos_model, open(cws_out, "r"), open(pos_out, "w"))
    run(par_cmdline, par_model, open(pos_out, "r"), open(par_out, "w"))
    lfs.close()
    lfs = open(speed_log, "r")

    for line in lfs:
        if "cws-tm-consume" in line:
            wordseg_tm = float(line.strip().split(":")[-1].strip().split()[1])
        if "pos-tm-consume" in line:
            postag_tm = float(line.strip().split(":")[-1].strip().split()[1])
        if "par-tm-consume" in line:
            parser_tm = float(line.strip().split(":")[-1].strip().split()[1])

    logging.info("speed: wordseg speed %f M/s" % (float(nr_sz) / 1024/ 1024/wordseg_tm))
    logging.info("speed: wordseg speed %f sent/s" % (float(nr_lines) / wordseg_tm))
    logging.info("speed: postagger speed %f M/s" % (float(nr_sz) / 1024/ 1024/postag_tm))
    logging.info("speed: postagger speed %f sent/s" % (float(nr_lines) / postag_tm))
    logging.info("speed: parser speed %f M/s" % (float(nr_sz) / 1024/ 1024/ parser_tm))
    logging.info("speed: parser speed %f sent/s" % (float(nr_lines) / parser_tm))
    global FINISHED_JOBS
    FINISHED_JOBS.add("speed")


def multithread_check(rootdir, outputdir, input_path, verbose=False):
    global FINISHED_JOBS
    if "speed" not in FINISHED_JOBS:
        speed_check(rootdir, outputdir, input_path, verbose)

    if os.name == 'nt':
        logging.info("multithread: windows speed check is not supported.")
        return

    def build(exe_prefix, model_prefix):
        exe = os.path.join(rootdir, "bin", "examples", ("multi_%s_cmdline" % exe_prefix))
        model = os.path.join(rootdir, "ltp_data", ("%s.model" % model_prefix))
        out = os.path.join(TMPDIR, "ltp.autotest.multi.%s.out" % exe_prefix)
        return (exe, model, out)
    cws_cmdline, cws_model, cws_out = build("cws", "cws")
    pos_cmdline, pos_model, pos_out = build("pos", "pos")

    if not input_path:
        logging.error("multithread: input not specified.")
        logging.info("multithread: speed check is canceled.")
        return

    nr_sz = os.stat(input_path).st_size
    dataset = open(input_path,"r").readlines()
    nr_lines = len(dataset)
    avg_sent_len = float(sum([len(data.decode("utf-8")) for data in dataset]))/nr_lines
    logging.info("multithread: average sentence length %f" % avg_sent_len)

    def check(exe):
        if not which(exe):
            logging.error("multithread: %s is not found." % exe)
            logging.info("multithread: speed check is canceled.")
            return False
        return True

    if not check(cws_cmdline):
        return
    if not check(pos_cmdline):
        return

    if not os.path.isfile(input_path):
        logging.error("multithread: input is not specified.")
        logging.info("multithread: speed check is canceled.")
        return

    speed_log = os.path.join(outputdir, "multi_speed.log")
    lfs = open(speed_log, "w")
    def run(exe, model, ifs, ofs):
        subprocess.call([exe, model, "2"], stdin=ifs, stdout=ofs, stderr=lfs)
        ifs.close()
        ofs.close()

    run(cws_cmdline, cws_model, open(input_path, "r"), open(cws_out, "w"))
    run(pos_cmdline, pos_model, open(cws_out, "r"), open(pos_out, "w"))
    lfs.close()
    lfs = open(speed_log, "r")

    for line in lfs:
        if "multi-cws-tm-consume" in line:
            multi_wordseg_tm = float(line.strip().split(":")[-1].strip().split()[1])
        if "multi-pos-tm-consume" in line:
            multi_postag_tm = float(line.strip().split(":")[-1].strip().split()[1])

    logging.info("multithread: wordseg speed %f M/s" % (float(nr_sz) / 1024/ 1024/multi_wordseg_tm))
    logging.info("multithread: wordseg speed %f sent/s" % (float(nr_lines) / multi_wordseg_tm))
    logging.info("multithread: postagger speed %f M/s" % (float(nr_sz) / 1024/ 1024/multi_postag_tm))
    logging.info("multithread: postagger speed %f sent/s" % (float(nr_lines) / multi_postag_tm))
    FINISHED_JOBS.add("multithread")


def server_check(rootdir, outputdir, input_path, verbose=False):
    if os.name == 'nt':
        logging.info("memcheck: windows memcheck is not supported.")
        return

    os.chdir(rootdir)
    command = ["./bin/ltp_server"]
    p = subprocess.Popen(command, stderr=DUMMY)
    command2 = ["./tools/autotest/request.py", "--file=%s" % input_path]
    ofs = open(os.path.join(outputdir, "server.return.txt"), "w")
    subprocess.call(command2, stdout=ofs, stderr=DUMMY)
    p.kill()


if __name__=="__main__":
    usage = "automatically test script for LTP project.\n"
    usage += "author: Yijia Liu <yjliu@ir.hit.edu.cn>, 2014"

    default_rootdir = os.path.abspath(__file__)
    default_rootdir = os.path.split(default_rootdir)[0]
    default_rootdir = os.path.split(default_rootdir)[0]
    default_rootdir = os.path.split(default_rootdir)[0]

    default_outputdir = os.path.split(os.path.abspath(__file__))[0]
    default_outputdir = os.path.join(default_outputdir, time.strftime("%Y-%m-%d-%H%M%S"))

    default_inputpath = os.path.join(default_rootdir, "test_data", "test_utf8.txt")

    optparser = OptionParser(usage)
    optparser.add_option("-r", "--root", dest="rootdir", default=default_rootdir,
            help="specify the LTP project root dir [default=%s]" % default_rootdir)
    optparser.add_option("-o", "--output", dest="outputdir", default=default_outputdir,
            help="specify the details output dir [default=%s]" % default_outputdir)
    optparser.add_option("-i", "--input", dest="inputpath", default=default_inputpath,
            help="the input path [default=%s]" % default_inputpath)
    optparser.add_option("-t", "--tasks", dest="tasks", default="all",
            help="the test tasks, tasks are separated by |.")
    opts, args = optparser.parse_args()

    if not os.path.isdir(opts.outputdir):
        os.mkdir(opts.outputdir)

    tasks = opts.tasks.split("|")
    if "all" in tasks or "cppcheck" in tasks:
        static_code_check(opts.rootdir, opts.outputdir)
    if not executable_check(opts.rootdir, opts.outputdir, opts.inputpath):
        sys.exit(1)
    if "all" in tasks or "memcheck" in tasks:
        memory_leak_check(opts.rootdir, opts.outputdir, opts.inputpath)
    if "all" in tasks or "callgrind" in tasks:
        callgrind_check(opts.rootdir, opts.outputdir, opts.inputpath)
    if "all" in tasks or "speed" in tasks:
        speed_check(opts.rootdir, opts.outputdir, opts.inputpath)
    if "all" in tasks or "server" in tasks:
        server_check(opts.rootdir, opts.outputdir, opts.inputpath)
    if "all" in tasks or "multithread" in tasks:
        multithread_check(opts.rootdir, opts.outputdir, opts.inputpath)
