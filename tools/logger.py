import os, time, sys
from datetime import datetime
import os.path as osp
import json, csv
from collections import defaultdict
from contextlib import contextmanager
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod

# ================================================================
# Formatting of different types of logs
# ================================================================

class KVWriter(ABC):
    @abstractmethod
    def writekvs(self, kvs):
        pass

class SeqWriter(ABC):
    @abstractmethod
    def writeseq(self, seq):
        pass

class HumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, filename_or_file):
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, "wt")
            self.own_file = True
        else:
            assert hasattr(filename_or_file, "read"), (
                f"expected file or str, got {filename_or_file}"
            )
            self.file = filename_or_file
            self.own_file = False

    def writekvs(self, kvs):
        # Create strings for printing
        key2str = {}
        for (key, val) in sorted(kvs.items()):
            if hasattr(val, "__float__"):
                valstr =  f"{val:<8.3g}"
            else:
                valstr = str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        # Find max widths
        if len(key2str) == 0:
            print("WARNING: tried to write empty key-value dict")
            return
        else:
            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))

        # Write out the data
        dashes = "-" * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in sorted(key2str.items(), key=lambda kv: kv[0].lower()):
            lines.append(
                f"| {key}{" " * (keywidth - len(key))} | {val}{" " * (valwidth - len(val))} |"
            )
        lines.append(dashes)
        self.file.write("\n".join(lines) + "\n")

        # Flush the output to the file
        self.file.flush()

    def _truncate(self, s):
        maxlen = 30
        return s[: maxlen - 3] + "..." if len(s) > maxlen else s

    def writeseq(self, seq):
        seq = list(seq)
        for (i, elem) in enumerate(seq):
            self.file.write(elem)
            if i < len(seq) - 1:  # add space unless this is the last one
                self.file.write(" ")
        self.file.write("\n")
        self.file.flush()

    def close(self):
        if self.own_file:
            self.file.close()

class JSONOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, "wt")

    def writekvs(self, kvs):
        for k, v in sorted(kvs.items()):    # 对键值对排序（保证一致性）
            if hasattr(v, "dtype"):         # 如果是 numpy 类型，比如 np.float32
                kvs[k] = float(v)           # 转为原生 float，避免 json 序列化失败
        self.file.write(json.dumps(kvs) + "\n")  # 序列化为 json，换行
        self.file.flush()                   # 立即写入硬盘，防止训练中断丢数据

    def close(self):
        self.file.close()

class CSVOutputFormat(KVWriter):
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, mode='w+t', newline='')
        self.writer = None
        self.fieldnames = []

    def writekvs(self, kvs):
        new_keys = set(kvs.keys()) - set(self.fieldnames)  # 处理新 key
        if new_keys:
            self.fieldnames += sorted(new_keys)  # 加入新列名
            self._reopen_with_new_fields()  # 重写文件并回写已有数据

        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
            self.writer.writeheader()

        self.writer.writerow(kvs)  # 写入新的一行数据
        self.file.flush()   # 缓冲区数据写入文件

    def _reopen_with_new_fields(self):
        self.file.seek(0)       # 把文件的 光标 移到开头
        lines = self.file.readlines()   # 读入所有行，变成一个列表
        self.file.seek(0)       # 再回到开头
        self.file.truncate()    # 截断文件：删除原来所有内容

        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        self.writer.writeheader()

        if len(lines) > 1:
            reader = csv.DictReader(lines[1:], fieldnames=self.fieldnames)
            for row in reader:
                self.writer.writerow(row)

    def close(self):
        self.file.close()

class TensorBoardOutputFormat(KVWriter):
    """ 训练用自动可视化图表展示数据的保存类型格式化 """

    def __init__(self, logdir):
        os.makedirs(logdir, exist_ok=True)
        self.writer = SummaryWriter(logdir)
        self.step = 0

    def writekvs(self, kvs):
        for k, v in kvs.items():
            self.writer.add_scalar(k, v, self.step)
        self.step += 1

    def close(self):
        self.writer.close()

def make_output_format(format, ev_dir, log_suffix=""):
    os.makedirs(ev_dir, exist_ok=True)
    if format == "stdout":
        return HumanOutputFormat(sys.stdout)
    elif format == "log":
        return HumanOutputFormat(osp.join(ev_dir, f"log{log_suffix}.txt"))
    elif format == "json":
        return JSONOutputFormat(osp.join(ev_dir, f"progress{log_suffix}.json"))
    elif format == "csv":
        return CSVOutputFormat(osp.join(ev_dir, f"progress{log_suffix}.csv"))
    elif format == "tensorboard":
        return TensorBoardOutputFormat(osp.join(ev_dir, f"tb{log_suffix}"))
    else:
        raise ValueError(f"Unknown format specified: {(format,)}")


# ================================================================
# Logger API
# ================================================================

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
DISABLED = 50

# ------------ 日志所记录的 键值对 的相关函数 ------------
def logkv(key, val):
    get_current().logkv(key, val)

def logkvs(d):
    for (k, v) in d.items():
        logkv(k, v)

def logkv_mean(key, val):
    get_current().logkv_mean(key, val)

def dumpkvs():
    return get_current().dumpkvs()

def getkvs():
    return get_current().name2val

# ------------ 直接输出的信息、及相关等级 ------------
def log(*args, level=INFO):
    get_current().log(*args, level=level)

def debug(*args):
    log(*args, level=DEBUG)

def info(*args):
    log(*args, level=INFO)

def warn(*args):
    log(*args, level=WARN)

def error(*args):
    log(*args, level=ERROR)

def set_level(level):
    get_current().set_level(level)

def get_dir():
    return get_current().get_dir()

# record_tabular = logkv
# dump_tabular = dumpkvs

# ------------ 自定义的装饰器 - 用于计算时间 ------------
@contextmanager
def profile_kv(scopename):
    logkey = "wait_" + scopename
    tstart = time.time()
    try:
        yield
    finally:
        get_current().name2val[logkey] += time.time() - tstart

def profile(n):
    """ 外部调用方式，函数的上一行写 @profile("函数名") """
    def decorator_with_name(func):
        def func_wrapper(*args, **kwargs):
            with profile_kv(n):
                return func(*args, **kwargs)

        return func_wrapper
    return decorator_with_name


# ================================================================
# Logger Management
# ================================================================

def get_current():
    if Logger.CURRENT is None:
        _configure_default_logger()
    return Logger.CURRENT

def _configure_default_logger():
    configure()
    Logger.DEFAULT = Logger.CURRENT

def configure(dir=None, format_strs=None, log_suffix=""):
    """ If comm is provided, average all numerical stats across that comm """
    if dir is None:
        time_now = datetime.now().strftime("idm-%Y-%m-%d-%H-%M-%S-%f")
        dir = osp.join("./output/", time_now)
    
    assert isinstance(dir, str)
    dir = os.path.expanduser(dir)
    os.makedirs(os.path.expanduser(dir), exist_ok=True)

    if format_strs is None:
        format_strs = os.getenv("OPENAI_LOG_FORMAT", "stdout,log,csv").split(",")
    format_strs = filter(None, format_strs)
    output_formats = [make_output_format(f, dir, log_suffix) for f in format_strs]

    Logger.CURRENT = Logger(dir=dir, output_formats=output_formats)
    if output_formats:
        log(f"Logging to {dir}")


class Logger(object):
    DEFAULT = None  # 没有输出文件的日志记录器，直接记录到终端
    CURRENT = None  # 正被上述自由函数使用的当前日志记录器

    def __init__(self, dir, output_formats):
        self.name2val = defaultdict(float)  # values this iteration
        self.name2cnt = defaultdict(int)
        self.level = INFO
        self.dir = dir
        self.output_formats = output_formats

    # ------------ Logging API, forwarded ------------
    def logkv(self, key, val):
        self.name2val[key] = val

    def logkv_mean(self, key, val):
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = oldval * cnt / (cnt + 1) + val / (cnt + 1)
        self.name2cnt[key] = cnt + 1

    def dumpkvs(self):
        out = self.name2val.copy()
        for fmt in self.output_formats:
            if isinstance(fmt, KVWriter):
                fmt.writekvs(out)
        self.name2val.clear()
        self.name2cnt.clear()
        return out

    def log(self, *args, level=INFO):
        if self.level <= level:
            self._do_log(args)

    def _do_log(self, args):
        for fmt in self.output_formats:
            if isinstance(fmt, SeqWriter):
                fmt.writeseq(map(str, args))

    # --------------- Configuration ---------------
    def set_level(self, level):
        self.level = level

    def get_dir(self):
        return self.dir

    def close(self):
        for fmt in self.output_formats:
            fmt.close()    

# def reset():
#     if Logger.CURRENT is not Logger.DEFAULT:
#         Logger.CURRENT.close()
#         Logger.CURRENT = Logger.DEFAULT
#         log("Reset logger")

# @contextmanager
# def scoped_configure(dir=None, format_strs=None, comm=None):
#     prevlogger = Logger.CURRENT
#     configure(dir=dir, format_strs=format_strs, comm=comm)
#     try:
#         yield
#     finally:
#         Logger.CURRENT.close()
#         Logger.CURRENT = prevlogger
