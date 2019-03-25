import datetime
from collections import OrderedDict
import os
import sys
import shutil
import os.path as osp
import json

import dateutil.tz

LOG_OUTPUT_FORMATS = ['stdout', 'log', 'json', 'csv']

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

DISABLED = 50


class OutputFormat(object):
    def writekvs(self, kvs):
        """
        Write key-value pairs
        """
        raise NotImplementedError

    def writeseq(self, args):
        """
        Write a sequence of other data (e.g. a logging message)
        """
        pass

    def close(self):
        return


class HumanOutputFormat(OutputFormat):
    def __init__(self, file):
        self.file = file

    def writekvs(self, kvs):
        # Create strings for printing
        key2str = OrderedDict()
        for (key, val) in kvs.items():
            valstr = '%-8.5g' % (val,) if hasattr(val, '__float__') else val
            key2str[self._truncate(key)] = self._truncate(valstr)

        # Find max widths
        keywidth = max(map(len, key2str.keys()))
        valwidth = max(map(len, key2str.values()))

        # Write out the data
        dashes = '-' * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in key2str.items():
            lines.append('| %s%s | %s%s |' % (
                key,
                ' ' * (keywidth - len(key)),
                val,
                ' ' * (valwidth - len(val)),
            ))
        lines.append(dashes)
        self.file.write('\n'.join(lines) + '\n')

        # Flush the output to the file
        self.file.flush()

    def _truncate(self, s):
        return s[:20] + '...' if len(s) > 23 else s

    def writeseq(self, args):
        for arg in args:
            self.file.write(arg)
        self.file.write('\n')
        self.file.flush()


class JSONOutputFormat(OutputFormat):
    def __init__(self, file):
        self.file = file

    def writekvs(self, kvs):
        for k, v in kvs.items():
            if hasattr(v, 'dtype'):
                v = v.tolist()
                kvs[k] = v
        self.file.write(json.dumps(kvs) + '\n')
        self.file.flush()

    def close(self):
        self.file.close()


class CSVOutputFormat(OutputFormat):
    def __init__(self, file):
        self.file = file
        self.keys = []
        self.sep = ','

    def writekvs(self, kvs):
        # Add our current row to the history
        extra_keys = set(kvs.keys()) - set(self.keys)
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for (i, k) in enumerate(self.keys):
                if i > 0:
                    self.file.write(',')
                self.file.write(k)
            self.file.write('\n')
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write('\n')
        for (i, k) in enumerate(self.keys):
            if i > 0:
                self.file.write(',')
            v = kvs.get(k)
            if v is not None:
                self.file.write(str(v))
        self.file.write('\n')
        self.file.flush()

    def close(self):
        self.file.close()


def make_output_format(format, ev_dir):
    if not os.path.exists(ev_dir):
        os.makedirs(ev_dir)
    if format == 'stdout':
        return HumanOutputFormat(sys.stdout)
    elif format == 'log':
        log_file = open(osp.join(ev_dir, 'log.txt'), 'wt')
        return HumanOutputFormat(log_file)
    elif format == 'json':
        json_file = open(osp.join(ev_dir, 'progress.json'), 'wt')
        return JSONOutputFormat(json_file)
    elif format == 'csv':
        csv_file = open(osp.join(ev_dir, 'progress.csv'), 'w+t')
        return CSVOutputFormat(csv_file)
    else:
        raise ValueError('Unknown format specified: %s' % (format,))


# ================================================================
# API
# ================================================================
def log_params(params):
    assert isinstance(params, dict)
    json_file = open(osp.join(Logger.CURRENT.get_dir(), 'params.json'), 'wt')
    output_format = JSONOutputFormat(json_file)
    output_format.writekvs(params)
    output_format.close()

def logkv(key, val):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    """
    Logger.CURRENT.logkv(key, val)


def dumpkvs():
    """
    Write all of the diagnostics from the current iteration

    level: int. (see old_logger.py docs) If the global logger level is higher than
                the level argument here, don't print to stdout.
    """
    Logger.CURRENT.dumpkvs()


# for backwards compatibility
record_tabular = logkv
dump_tabular = dumpkvs


def log(level, *args):
    """
    Write the sequence of args, with no separators, to the console and output files (if you've configured an output file).
    """
    Logger.CURRENT.log(level, *args)


def debug(*args):
    log(DEBUG, *args)


def info(*args):
    log(INFO, *args)


def warn(*args):
    log(WARN, *args)


def error(*args):
    log(ERROR, *args)


def set_level(level):
    """
    Set logging threshold on current logger.
    """
    Logger.CURRENT.set_level(level)


def get_level():
    """
    Set logging threshold on current logger.
    """
    return Logger.CURRENT.level


def get_dir():
    """
    Get directory that log files are being written to.
    will be None if there is no output directory (i.e., if you didn't call start)
    """
    return Logger.CURRENT.get_dir()


def get_expt_dir():
    sys.stderr.write(
        "get_expt_dir() is Deprecated. Switch to get_dir() [%s]\n" % (get_dir(),))
    return get_dir()


# ================================================================
# Backend
# ================================================================


class Logger(object):
    # A logger with no output files. (See right below class definition)
    DEFAULT = None
    # So that you can still log to the terminal without setting up any output files
    CURRENT = None  # Current logger being used by the free functions above

    def __init__(self, dir, output_formats):
        self.name2val = OrderedDict()  # values this iteration
        self.level = INFO
        self.dir = dir
        self.output_formats = output_formats

    # Logging API, forwarded
    # ----------------------------------------
    def logkv(self, key, val):
        self.name2val[key] = val

    def dumpkvs(self):
        for fmt in self.output_formats:
            fmt.writekvs(self.name2val)
        self.name2val.clear()

    def log(self, level, *args):
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('[%Y-%m-%d %H:%M:%S.%f %Z] ')
        if self.level <= level:
            self._do_log((timestamp,) + args)

    # Configuration
    # ----------------------------------------
    def set_level(self, level):
        self.level = level

    def get_dir(self):
        return self.dir

    def close(self):
        for fmt in self.output_formats:
            fmt.close()

    # Misc
    # ----------------------------------------
    def _do_log(self, args):
        for fmt in self.output_formats:
            fmt.writeseq(args)


# ================================================================

Logger.DEFAULT = Logger(
    output_formats=[HumanOutputFormat(sys.stdout)], dir=None)
Logger.CURRENT = Logger.DEFAULT


class session(object):
    """
    Context manager that sets up the loggers for an experiment.
    """
    def __init__(self, dir, format_strs=LOG_OUTPUT_FORMATS):
        self.dir = dir
        self.format_strs = format_strs

    def __enter__(self):
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        output_formats = [make_output_format(f, self.dir) for f in self.format_strs]
        Logger.CURRENT = Logger(dir=self.dir, output_formats=output_formats)

    def __exit__(self, *args):
        Logger.CURRENT.close()
        Logger.CURRENT = Logger.DEFAULT


# ================================================================


def main():
    info("hi")
    debug("shouldn't appear")
    set_level(DEBUG)
    debug("should appear")
    dir = "./tmp/testlogging"
    if os.path.exists(dir):
        shutil.rmtree(dir)
    with session(dir=dir):
        record_tabular("a", 3)
        record_tabular("b", 2.5)
        dump_tabular()
        record_tabular("b", -2.5)
        record_tabular("a", 5.5)
        dump_tabular()
        info("^^^ should see a = 5.5")

    record_tabular("b", -2.5)
    dump_tabular()

    record_tabular("a", "longasslongasslongasslongasslongasslongassvalue")
    dump_tabular()


if __name__ == "__main__":
    main()