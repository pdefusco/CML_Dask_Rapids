from dask_cuda import LocalCUDACluster
from dask.distributed import Client

cluster = LocalCUDACluster()

cluster = LocalCUDACluster(
    CUDA_VISIBLE_DEVICES="0"
)  # Creates one worker for GPU 0

client = Client(cluster)

def get_gpu_model():
    import pynvml

    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))


result = client.submit(get_gpu_model).result()

print(result)
# b'Tesla V100-SXM2-16GB
