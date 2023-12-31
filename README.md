# CUDA_SpMV_CSR
    Introduction to Sparse Matrices:
    A sparse matrix is one in which most of the elements are zero.
    Storing all elements of such a matrix would be inefficient both in terms of memory and computational resources.
    In scientific computing and various engineering applications, matrices tend to be large and sparse.

    Compressed Sparse Row (CSR) Representation:
    The CSR format is a compact way to store sparse matrices. Instead of holding all n^2 elements, CSR only stores non-zero elements and their indices. There are three components to this format:
        An array that contains all non-zero values in the matrix.
        An array of column indices corresponding to each non-zero value.
        An array of row pointers indicating where each row starts in the array of non-zero values.

    SpMV Algorithm:
    The Sparse Matrix-Vector Multiplication algorithm multiplies a sparse matrix with a dense vector.
    The operation y = Ax multiplies matrix A with vector x to yield vector y.
    In a parallel SpMV algorithm, this multiplication is done in a way that exploits the sparsity pattern by skipping calculations involving zero entries of A.

    CUDA Parallelization:
    CUDA allows this operation to be parallelized over thousands of threads on a GPU, with each thread handling a portion of the computation.
    The parallelization strategy often involves assigning rows or sets of rows of A to different threads, thereby dividing the work efficiently.

    Steps in the CUDA-based SpMV Algorithm:
    a. Input Matrix Size and Values: The user inputs the size n and then provides the values of the n x n matrix A along with their positions.
    b. Conversion to CSR: The input matrix is converted to CSR format on the CPU. This pre-processing step is important for efficient parallel computation on the GPU.
    c. Transfer to GPU Memory: The CSR arrays and vector x are transferred to GPU memory, which is optimized for the high-throughput requirements of the multiplication.
    d. Kernel Execution: A CUDA kernel is launched where each thread or block of threads computes one or more outputs of the result vector y based on the division of work.
    Threads read from the CSR data and the vector x, multiplying and summing as needed.
    e. Result Retrieval: After computation, the result vector y is transferred back to the CPU memory.
    f. Output Display: Finally, the results are displayed, showing the product of the sparse matrix-vector multiplication.

By following this process, SpMV can be performed efficiently on large matrices using the powerful parallel processing capabilities of GPUs.
This explanation includes the conceptual understanding as well as the computational flow of the algorithm when implemented with CUDA.

# CSR Matrix-Vector Multiplication using CUDA
Description

This program performs matrix-vector multiplication using the Compressed Sparse Row (CSR) format for matrices. It is designed to run on NVIDIA GPUs using CUDA. The main functionality of the program is to convert a dense matrix into its CSR representation and then multiply it with a vector using CUDA kernels.
Features

    Matrix Conversion: Converts a dense matrix into CSR format.
    Matrix-Vector Multiplication: Performs multiplication of a CSR matrix with a vector using CUDA.
    CUDA Acceleration: Utilizes the parallel processing power of NVIDIA GPUs for efficient computation.

Requirements

    NVIDIA GPU with CUDA support.
    CUDA Toolkit (compatible with your GPU and operating system).

Input Format

    The program prompts for the size of the matrix (n).
    Enter the elements of the matrix row-wise.
    Enter the elements of the right-hand side (RHS) vector.

Output

    The program outputs the CSR representation of the matrix (Av, Aj, Ap).
    The result of the matrix-vector multiplication is displayed.

Code Structure

    __device__ float multiply_row_kernel(...): A device function to multiply a row of the CSR matrix with the vector.
    __global__ void csrmul_kernel(...): A global kernel function for parallel CSR matrix-vector multiplication.
    void convertToCSR(...): Function to convert a dense matrix to CSR format.
    int main(...): The main function to handle user input, matrix conversion, and call CUDA kernels.

Known Issues

    No error handling for invalid inputs.
    Limited to square matrices.

Contributing

Contributions to improve the code or extend its functionality are welcome. Please submit a pull request or open an issue for discussion.

