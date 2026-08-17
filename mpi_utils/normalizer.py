import threading
import numpy as np
from mpi4py import MPI

class normalizer:
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        # some local information
        self.local_sum = np.zeros(self.size, np.float32)
        self.local_sumsq = np.zeros(self.size, np.float32)
        self.local_count = np.zeros(1, np.float32)
        # get the total sum sumsq and sum count
        self.total_sum = np.zeros(self.size, np.float32)
        self.total_sumsq = np.zeros(self.size, np.float32)
        self.total_count = np.ones(1, np.float32)
        # get the mean and std
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)
        # thread locker
        self.lock = threading.Lock()
    
    # update the parameters of the normalizer
    def update(self, v):
        v = v.reshape(-1, self.size)
        # do the computing
        with self.lock:
            self.local_sum += v.sum(axis=0)
            self.local_sumsq += (np.square(v)).sum(axis=0)
            self.local_count[0] += v.shape[0]

    # sync the parameters across the cpus
    def sync(self, local_sum, local_sumsq, local_count):
        local_sum[...] = self._mpi_average(local_sum)
        local_sumsq[...] = self._mpi_average(local_sumsq)
        local_count[...] = self._mpi_average(local_count)
        return local_sum, local_sumsq, local_count

    def recompute_stats(self):
        with self.lock:
            local_count = self.local_count.copy()
            local_sum = self.local_sum.copy()
            local_sumsq = self.local_sumsq.copy()
            # reset
            self.local_count[...] = 0
            self.local_sum[...] = 0
            self.local_sumsq[...] = 0
        # synrc the stats
        sync_sum, sync_sumsq, sync_count = self.sync(local_sum, local_sumsq, local_count)
        # update the total stuff
        self.total_sum += sync_sum
        self.total_sumsq += sync_sumsq
        self.total_count += sync_count
        # calculate the new mean and std
        self.mean = self.total_sum / self.total_count
        self.std = np.sqrt(np.maximum(np.square(self.eps), (self.total_sumsq / self.total_count) - np.square(self.total_sum / self.total_count)))
    
    # average across the cpu's data
    def _mpi_average(self, x):
        buf = np.zeros_like(x)
        MPI.COMM_WORLD.Allreduce(x, buf, op=MPI.SUM)
        buf /= MPI.COMM_WORLD.Get_size()
        return buf

    # normalize the observation
    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        return np.clip((v - self.mean) / (self.std), -clip_range, clip_range)

class RunningMeanStd:
    def __init__(self, og_mean, og_std, og_count):
        self.og_mean = og_mean
        self.og_std = og_std
        self.og_count = og_count
        
        self.cur_mean = og_mean
        self.cur_std = og_std
        self.cur_count = og_count
        
    def update_from_replay(self, obs_buf):
        buf_mean = np.mean(obs_buf, axis=0)
        buf_std = np.std(obs_buf, axis=0)
        buf_count = obs_buf.shape[0]
        self.cur_count = self.og_count + buf_count
        self.cur_mean, self.cur_std = self.update_mean_var_count_from_replay(
            self.og_mean, self.og_std, self.og_count, buf_mean, buf_std, buf_count)

    def update_mean_var_count_from_replay(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count
    
        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
    
        return new_mean, new_var
    
    def combine_rms_across_processes(self):
        # Send rms of 0 to 1
        if MPI.COMM_WORLD.Get_rank() == 0:
            MPI.COMM_WORLD.Send(self.cur_mean, dest=1, tag=13)
            MPI.COMM_WORLD.Send(self.cur_std, dest=1, tag=14)
            MPI.COMM_WORLD.Send(np.array(self.cur_count), dest=1, tag=15)
        if MPI.COMM_WORLD.Get_rank() == 1:
            mean0 = np.zeros_like(self.cur_mean)
            MPI.COMM_WORLD.Recv(mean0, source=0, tag=13)
            std0 = np.zeros_like(self.cur_std)
            MPI.COMM_WORLD.Recv(std0, source=0, tag=14)
            count0 = np.zeros_like(self.cur_count)
            MPI.COMM_WORLD.Recv(count0, source=0, tag=15)
            
            # Calculate overall rms and set in 1
            mean_total, std_total, count_total = self.combine_rms(self.cur_mean, self.cur_std, self.cur_count, mean0, std0, count0)
            self.cur_mean, self.cur_std, self.cur_count = mean_total, std_total, count_total
            
            # Send to 0
            MPI.COMM_WORLD.Send(self.cur_mean, dest=0, tag=23)
            MPI.COMM_WORLD.Send(self.cur_std, dest=0, tag=24)
            MPI.COMM_WORLD.Send(np.array(self.cur_count), dest=0, tag=25)
        if MPI.COMM_WORLD.Get_rank() == 0:
            MPI.COMM_WORLD.Recv(self.cur_mean, source=1, tag=23)
            MPI.COMM_WORLD.Recv(self.cur_std, source=1, tag=24)
            count = np.array(self.cur_count)
            MPI.COMM_WORLD.Recv(count, source=1, tag=25)
            self.cur_count = count
    
    def combine_rms(self, mean1, std1, count1, mean2, std2, count2):
        count_total = count1 + count2
        mean_total = (count1*mean1 + count2*mean2) / (count_total)
        
        d1 = mean1 - mean_total
        d2 = mean2 - mean_total
        
        std_total = np.sqrt( (count1*(std1**2 + d1**2) + count2*(std2**2 + d2**2)) / (count_total) )
        
        return mean_total, std_total, count_total
               