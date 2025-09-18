# cpu-vs-gpu-Guassian-image-filtering-different-sigma

This project demonstrates GPU acceleration in PyTorch by applying multi-scale Gaussian filters to batches of grayscale images.
The pipeline loads real images from multiple categories (cats, chairs, cycles), preprocesses them into tensors, and applies convolution-based filtering both on CPU and GPU.
Average execution times are measured and compared, highlighting the computational advantage of GPU parallelism for convolution operations.
The project includes:

  Custom image dataset loading and batching
  
  On-the-fly Gaussian kernel creation
  
  Parallelized convolutions on GPU
  
  CPU vs GPU performance benchmarking
  
  Visual side-by-side comparisons of original vs filtered images
