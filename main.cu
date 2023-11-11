#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__device__ float multiply_row_kernel(unsigned int rowsize,
unsigned int *Aj, // column indices for row
float *Av, //nonzero enntries for row
float *x) //the RHS vector
{
    float sum = 0;

    for(unsigned int column = 0; column < rowsize; column++)
    {
        // Dereference Av to get the value of the nonzero entry
        // and dereference x to get the value from the RHS vector using Aj as the index
        sum += Av[column] * x[Aj[column]]; // Dereferencing occurs here
    }

    return sum;

}

__global__ void csrmul_kernel(unsigned int *Ap, unsigned int *Aj, float *Av, unsigned int num_rows, float *x, float *y)
{
    
    unsigned int row = blockIdx.x*blockDim.x + threadIdx.x;
    if( row<num_rows )
    {
        unsigned int row_begin = Ap[row];
        unsigned int row_end = Ap[row+1];
        y[row] = multiply_row_kernel(row_end-row_begin, Aj+row_begin,
        Av+row_begin, x);
    }
}

void convertToCSR(float *matrix, int n, float **Av, unsigned int **Aj, unsigned int **Ap, int *nnz_out) {
    // Count non-zero elements in the matrix
    int nnz = 0;
    for (int i = 0; i < n * n; ++i) {
        if (matrix[i] != 0) {
            nnz++;
        }
    }
    *nnz_out = nnz;
    // Allocate memory for CSR arrays based on nnz count
    *Av = (float *)malloc(nnz * sizeof(float));
    *Aj = (unsigned int *)malloc(nnz * sizeof(unsigned int));
    *Ap = (unsigned int *)malloc((n + 1) * sizeof(unsigned int));

    // Fill CSR arrays with data
    int k = 0; // Index for non-zero elements
    (*Ap)[0] = 0; // First element of Ap is always 0
    for (int i = 0; i < n; ++i) {
        (*Ap)[i+1] = (*Ap)[i]; // Initialize Ap for this row with the previous value
        for (int j = 0; j < n; ++j) {
            if (matrix[i * n + j] != 0) {
                (*Av)[k] = matrix[i * n + j]; // Assign value to Av
                (*Aj)[k] = j; // Assign column index to Aj
                (*Ap)[i+1]++; // Increment the Ap entry for the next row
                k++;
            }
        }
    }
}

int main(int argc, char *argv[])
{
    int n; // Dimension of the matrix (rows x columns)
    int nnz; // non-zero numbers
    printf("Provide the size 'n' of the matrix A[n][n]:" );
    scanf("%d",&n);
    float A[n][n];
    for (int i = 0; i < n; i++)
    {
        printf("Provide the values for row %d: \n",i);
        for (int j = 0; j < n; j++)
        {
            scanf("%f",&A[i][j]);
        }
        
    }
    // Convert the matrix to 1D array representation for simplicity
    float *A_1D = (float *)A;//Converts a A[n][n] into a A[nxn]
    
    // CSR arrays
    unsigned int *Aj, *Ap;
    float *Av;
    // Function to convert matrix to CSR representation
    convertToCSR(A_1D, n, &Av, &Aj, &Ap, &nnz);

    // Print CSR representation
    printf("Av: ");
    for (int i = 0; i < Ap[n]; ++i) {
        printf("%f ", Av[i]);
    }
    printf("\nAj: ");
    for (int i = 0; i < Ap[n]; ++i) {
        printf("%d ", Aj[i]);
    }
    printf("\nAp: ");
    for (int i = 0; i <= n; ++i) {
        printf("%d ", Ap[i]);
    }
    printf("\n");

    float *d_x;
    cudaMalloc(&d_x, n * sizeof(float));
    float *x = (float*)malloc(n * sizeof(float)); // Example RHS vector values. It has 4 columns since Ap columns indices goes 0-3
    printf("Provide the %d values for RHS vector: \n",n);
    for (int i = 0; i < n; i++)
    {
        scanf("%f",&x[i]);
    }
    
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    float *y = (float*)malloc(n * sizeof(float));// Result vector. It has as many elements as num_rows
    for (int i = 0; i < n; i++)
        {
            y[i] = 0.0f;
        }
    float *d_y;
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    unsigned int *d_Aj, *d_Ap;
    float *d_Av;

    cudaMalloc(&d_Aj, nnz * sizeof(unsigned int));
    cudaMemcpy(d_Aj, Aj, nnz * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_Ap, (n + 1) * sizeof(unsigned int));
    cudaMemcpy(d_Ap, Ap, (n + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_Av, nnz * sizeof(float));
    cudaMemcpy(d_Av, Av, nnz * sizeof(float), cudaMemcpyHostToDevice);

    //csrmul_serial(Ap, Aj, Av, n, x, y);

    int threadsPerBlock = 256;
    int nblocks = (n + threadsPerBlock - 1) / threadsPerBlock; // This ensures rounding up if n is not a multiple of threadsPerBlock
    csrmul_kernel<<<nblocks, 256>>>(d_Ap, d_Aj, d_Av, n, d_x, d_y);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // It is a good practice to synchronize after kernel calls during debugging
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Kernel Sync Error: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpy(y, d_y, nnz * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_Aj);
    cudaFree(d_Ap);
    cudaFree(d_Av);

    for (int i = 0; i < n; i++)
    {
        printf("y[%d] = %f ",i,y[i]);
    }
    printf("\n");
    // Free allocated memory
    free(x);
    free(y);
    free(Av);
    free(Aj);
    free(Ap);

    return 0;
}
