"""One-second process accounting samples via the macOS SDK's libproc ABI.
Sampled maxima do not replace the lifetime maximum RSS from /usr/bin/time -l.
"""
import ctypes,errno,threading,time

class UsageV2(ctypes.Structure):
    _fields_=[('uuid',ctypes.c_uint8*16)]+[(name,ctypes.c_uint64) for name in (
        'user_time','system_time','pkg_idle_wkups','interrupt_wkups','pageins',
        'wired_size','resident_size','phys_footprint','proc_start_abstime',
        'proc_exit_abstime','child_user_time','child_system_time','child_pkg_idle_wkups',
        'child_interrupt_wkups','child_pageins','child_elapsed_abstime',
        'diskio_bytesread','diskio_byteswritten')]
lib=ctypes.CDLL('/usr/lib/libproc.dylib',use_errno=True)
lib.proc_pid_rusage.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p];lib.proc_pid_rusage.restype=ctypes.c_int
lib.proc_listchildpids.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_int];lib.proc_listchildpids.restype=ctypes.c_int
def usage(pid):
    value=UsageV2()
    if lib.proc_pid_rusage(pid,2,ctypes.byref(value))!=0:
        n=ctypes.get_errno();raise OSError(n,'proc_pid_rusage',pid)
    return {name:int(getattr(value,name)) for name in ('wired_size','resident_size','phys_footprint','pageins','diskio_bytesread','diskio_byteswritten','proc_start_abstime')}
def children(pid):
    buffer=(ctypes.c_int*64)()
    n=lib.proc_listchildpids(pid,buffer,ctypes.sizeof(buffer))
    if n<0:raise OSError(ctypes.get_errno(),'proc_listchildpids',pid)
    return [int(x) for x in buffer if x>0]
class MemoryMonitor:
    def __init__(self,pid,follow_time_child=False):
        self.pid=pid;self.follow=follow_time_child;self.samples=[];self.errors=[];self.ended=threading.Event();self.start=time.monotonic()
        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()
    def _sample(self):
        try:
            pids=children(self.pid) if self.follow else [self.pid]
            if len(pids)>1:raise RuntimeError('time wrapper has multiple children; accounting scope is ambiguous')
            for pid in pids:self.samples.append({'seconds':time.monotonic()-self.start,'pid':pid,**usage(pid)})
        except OSError as error:
            # ESRCH is expected if the child exits between enumeration and sampling.
            if error.errno!=errno.ESRCH:self.errors.append(str(error))
        except RuntimeError as error:self.errors.append(str(error))
    def _run(self):
        while not self.ended.is_set():self._sample();self.ended.wait(1.)
    def close(self):
        self.ended.set();self.thread.join()
        return {'interval_s':1.,'scope':'one child of time-l wrapper' if self.follow else 'server process',
                'samples':self.samples,'errors':self.errors,
                'sampled_peak':{key:max((x[key] for x in self.samples),default=None) for key in ('wired_size','resident_size','phys_footprint')},
                'note':'Process footprint, wired and RSS are separate accounting views. Do not add them. Disk counters are OS-accounted process I/O, not proof of cold physical SSD reads.'}
