import os
import torch
import subprocess
import time

class GpuMemoryCalculator():
    def __init__(self, device="cuda:0"):
        self.device = device

    def get_torch_memory_allocated(self):
        m_allocated = torch.cuda.memory_allocated(self.device) / 1024 ** 2  # MB
        m_allocated = round(m_allocated, 2)
        return m_allocated

    def get_torch_memory_reserved(self):
        m_reserved = torch.cuda.memory_reserved(self.device) / 1024 ** 2   # MB
        m_reserved = round(m_reserved, 2)
        return m_reserved

    def get_torch_max_memory_reserved(self):
        m_max_reserved = torch.cuda.max_memory_reserved(self.device) / 1024 ** 2   # MB
        m_max_reserved = round(m_max_reserved, 2)
        return m_max_reserved

    def get_gpu_memory_usage(self):
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used', '--format=csv,nounits,noheader'])

        # 解析输出, 获取每个GPU的显存占用
        gpu_memory_usage = {}
        for line in output.decode().strip().split('\n'):
            gpu_uuid, used_memory = line.split(', ')
            gpu_memory_usage[gpu_uuid] = float(used_memory)
        return gpu_memory_usage

    def get_device_gpu_memory_usage(self):
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used', '--format=csv,nounits,noheader'])

        # 解析输出, 获取每个GPU的显存占用
        gpu_memory_usage = {}
        for line in output.decode().strip().split('\n'):
            gpu_uuid, used_memory = line.split(', ')
            gpu_memory_usage[gpu_uuid] = float(used_memory)

        device_i = int(self.device.split(":")[1])
        device_gpu_memory_usage = gpu_memory_usage[list(gpu_memory_usage.keys())[device_i]]

        return device_gpu_memory_usage

    def get_torch_tensor_size(self, in_tensor):
        element_size = in_tensor.element_size()
        num_elements = in_tensor.nelement()
        in_tensor_size = element_size * num_elements
        in_tensor_size = in_tensor_size / (1024 ** 2)  # MB
        return in_tensor_size

    def get_model_size(self, model):
        torch.save(model, 'tmp_model_debug_size.pth')
        model_size = os.path.getsize('model_debug_size.pth')
        model_size = model_size / (1024 ** 2)  # MB
        return model_size


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


if __name__ == "__main__":
    print("Start")
    '''
    工具类的utils，如显存和时间计算等
    '''
    print("End")