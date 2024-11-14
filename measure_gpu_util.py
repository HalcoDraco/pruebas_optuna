import numpy as np
import subprocess as sp
from threading import Timer
import matplotlib.pyplot as plt
import datetime

class MeasureGPUUtilization:
    def __init__(self):
        self.gpu_util = []
        self.training_starts_list = []
        self.training_stops_list = []
        self.running = True

    def start(self):
        self.save_gpu_util_every_5secs()

    def stop(self):
        self.running = False

    def training_starts(self):
        self.training_starts_list.append(len(self.gpu_util))

    def training_stops(self):
        self.training_stops_list.append(len(self.gpu_util))

    def get_gpu_utilization(self):
        output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
        COMMAND = "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits"
        try:
            utilization_info = output_to_list(sp.check_output(COMMAND.split(), stderr=sp.STDOUT))
        except sp.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

        gpu_utilizations = [int(utilization) for utilization in utilization_info]

        return gpu_utilizations
    
    def save_gpu_util_every_5secs(self):
        """
            This function calls itself every 5 secs and stores the gpu_utilization.
        """  
        if not self.running:
            return
        Timer(5.0, self.save_gpu_util_every_5secs).start()
        self.gpu_util.append(self.get_gpu_utilization())

    def print_results(self):
        gpu_util = np.array(self.gpu_util)
        print('GPU Utilization:\n\tAverage:',gpu_util.mean(axis=0),'\n\tMax:',gpu_util.max(axis=0),'\n\tMin:',gpu_util.min(axis=0), '\n\tStd:',gpu_util.std(axis=0))

    def save_plot(self):
        # Plot a line graph of the GPU utilization for every gpu in the system
        gpu_util = np.array(self.gpu_util)
        time = np.arange(0, len(gpu_util) * 5, 5)  # Create a time array that increments by 5 for each data point
        plt.figure(figsize=(12, 6))
        for i in range(gpu_util.shape[1]):
            plt.plot(time, gpu_util[:,i], label=f'GPU {i}')  # Use the time array as the x-values
        for t in self.training_starts_list:
            plt.axvline(x=t*5, color='g', linestyle='--')
        for t in self.training_stops_list:
            plt.axvline(x=t*5, color='r', linestyle='--')
        plt.xlabel('Time (seconds)')
        plt.ylabel('GPU Utilization (%)')
        plt.title('GPU Utilization over time')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"gpu_utilization_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")


