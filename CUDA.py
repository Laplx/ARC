import torch


def check_gpu_memory():
    if torch.cuda.is_available():
        print(f"CUDA可用，设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            allocated_memory = torch.cuda.memory_allocated(i) / 1024**3
            cached_memory = torch.cuda.memory_reserved(i) / 1024**3
            free_memory = total_memory - allocated_memory
            
            print(f"GPU {i} ({torch.cuda.get_device_name(i)}):")
            print(f"  总显存: {total_memory:.2f} GB")
            print(f"  已分配: {allocated_memory:.2f} GB")
            print(f"  缓存: {cached_memory:.2f} GB")
            print(f"  可用: {free_memory:.2f} GB")
    else:
        print("CUDA不可用")


def cleanup_gpu_memory():
    import gc
    
    if torch.cuda.is_available():
        print("清理GPU显存...")
        
        # 释放所有未使用的缓存
        torch.cuda.empty_cache()
        
        # 手动触发垃圾回收
        gc.collect()
        
        # 再次释放缓存
        torch.cuda.empty_cache()
        
        # 检查清理后状态
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i} - 清理后: 已分配 {allocated:.2f} GB, 缓存 {cached:.2f} GB")
    else:
        print("没有GPU可用")


if __name__ == "__main__":
    check_gpu_memory()
    cleanup_gpu_memory()
    
    # export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True