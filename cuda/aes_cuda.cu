#include "aes_cuda.h"

__device__ unsigned char cuda_indexSbox(unsigned char string){ //helper function for lookup
    unsigned char upperHalf = (string & 0xf0) >> 4;
    unsigned char lowerHalf = (string & 0x0f);
    return cuda_s_box[upperHalf][lowerHalf];
}

__device__ unsigned char cuda_indexInvSbox(unsigned char string){  //helper function for inverse lookup
    unsigned char upperHalf = (string & 0xf0) >> 4;
    unsigned char lowerHalf = (string & 0x0f);
    return cuda_inv_s_box[upperHalf][lowerHalf];
}

__device__ void cuda_subBytes(unsigned char * &string, int rows, int cols){ //lookup table for sub bytes
    int i,j;
    for (i = 0; i < rows; i++){
        for (j = 0; j < cols; j++){
            string[i + j*rows] = cuda_indexSbox(string[i + j*rows]);
        }
    }
}

__device__ void cuda_invSubBytes(unsigned char * &string, int rows, int cols){ //lookup table for inv sub bytes
    int i,j;
    for (i = 0; i < rows; i++){
        for (j = 0; j < cols; j++){
            string[i + j*rows] = cuda_indexInvSbox(string[i + j*rows]);
        }
    }
}


__device__ void cuda_initArr(unsigned char ** &strArr, int rows, int cols){ //allocates an array of dim rows x cols
    strArr = new unsigned char*[rows];
    for (int i = 0; i < rows; i++){
        strArr[i] = new unsigned char[cols];
    }
}

__device__ void cuda_killArr(unsigned char ** strArr, int rows, int cols){ //deallocates an array of dim rows x cols
    for (int i = 0; i < rows; i++){
        delete [] strArr[i];
    }
    delete [] strArr;
}

__device__ void printStrCuda(unsigned char * str, int len){ //prints string as hex int
    printf("\n");
    for (int i = 0; i < len; i++){
        printf("%c",str[i]);
    }
    printf("\n");
}

__device__ void cuda_shiftRows(unsigned char * &string, int rows, int cols){
    unsigned char rowValsExt[BLOCKSIZE/NUMROWS];
    int i,j;
    for (i = 1; i < rows; i++){
        int shiftDist = (i == 1)? 1: (i == 2)? 2: (i == 3)? 3 : 0;
        for (j = 0; j < cols; j++){
            int shiftedIdx = (j + shiftDist) % cols; //stores circular shifted values (shift left by shiftDist)
            rowValsExt[j] = string[i + rows*shiftedIdx];
        }
        for (j = 0; j < cols; j++){
            string[i + j*rows] = rowValsExt[j];
        }
    }
}


__device__ void cuda_invShiftRows(unsigned char * &string, int rows, int cols){
    unsigned char rowValsExt[BLOCKSIZE/NUMROWS];
    int i,j;
    for (i = 1; i < rows; i++){
        int shiftDist = (i == 1)? 1: (i == 2)? 2: (i == 3)? 3 : 0;
        for (j = 0; j < cols; j++){
            int shiftedIdx = (cols - shiftDist + j)%cols; //stores circular shifted values (shift right by shiftDist)
            rowValsExt[j] = string[i + rows*shiftedIdx];
        }
        for (j = 0; j < cols; j++){
            string[i + j*rows] = rowValsExt[j];
        }
    }
}

__device__ unsigned char cuda_gfMul(unsigned char c, int factor){
    //GF multiplication. Done with case statements for the cases seen in the algorithm. Creating a general function was out of the scope of this project
    switch (factor){
        case 1:
            return c;
        case 2:
            return (c << 1);
        case 3:
            return ((c << 1) ^ c);
        case 9:
            return ((((c << 1) << 1) << 1) ^ c);
        case 11:
            return (((((c << 1) << 1) ^ c) << 1) ^ c);
        case 13:
            return (((((c << 1) ^ c) << 1) << 1) ^ c);
        case 14:
            return (((((c << 1) ^ c) << 1) ^ c) << 1);
        default:
            return c;
    }
 }

 __device__ void cuda_mixCols(unsigned char * &string, int rows, int cols){
    //mix columns as specified by the algorithm
    for (int j = 0; j < cols; j++){
        unsigned char temp[NUMROWS];
        temp[0] = cuda_gfMul(string[0 + j*rows], 2) ^ cuda_gfMul(string[1 + j*rows], 3) ^ cuda_gfMul(string[2 + j*rows], 1) ^ cuda_gfMul(string[3 + j*rows], 1); //2 3 1 1
        temp[1] = cuda_gfMul(string[0 + j*rows], 1) ^ cuda_gfMul(string[1 + j*rows], 2) ^ cuda_gfMul(string[2 + j*rows], 3) ^ cuda_gfMul(string[3 + j*rows], 1); //1 2 3 1
        temp[2] = cuda_gfMul(string[0 + j*rows], 1) ^ cuda_gfMul(string[1 + j*rows], 1) ^ cuda_gfMul(string[2 + j*rows], 2) ^ cuda_gfMul(string[3 + j*rows], 3); //1 1 2 3
        temp[3] = cuda_gfMul(string[0 + j*rows], 3) ^ cuda_gfMul(string[1 + j*rows], 1) ^ cuda_gfMul(string[2 + j*rows], 1) ^ cuda_gfMul(string[3 + j*rows], 2); //3 1 1 2
        for (int i = 0; i < NUMROWS; i++){
            string[i + j*rows] = temp[i];
        }
    }
}

__device__ void cuda_invMixCols(unsigned char * &string, int rows, int cols){
    //does the inverse of mix columns, as specified by the algorithm
    for (int j = 0; j < cols; j++){
        unsigned char temp[NUMROWS];
        temp[0] = cuda_gfMul(string[0 + j*rows], 14) ^ cuda_gfMul(string[1 + j*rows], 11) ^ cuda_gfMul(string[2 + j*rows], 13) ^ cuda_gfMul(string[3 + j*rows], 9); //14 11 13 9
        temp[1] = cuda_gfMul(string[0 + j*rows], 9) ^  cuda_gfMul(string[1 + j*rows], 14) ^ cuda_gfMul(string[2 + j*rows], 11) ^ cuda_gfMul(string[3 + j*rows], 13); //9 14 11 13
        temp[2] = cuda_gfMul(string[0 + j*rows], 13) ^ cuda_gfMul(string[1 + j*rows], 9) ^  cuda_gfMul(string[2 + j*rows], 14) ^ cuda_gfMul(string[3 + j*rows], 11); //13 9 14 11
        temp[3] = cuda_gfMul(string[0 + j*rows], 11) ^ cuda_gfMul(string[1 + j*rows], 13) ^ cuda_gfMul(string[2 + j*rows], 9) ^  cuda_gfMul(string[3 + j*rows], 14); //11 13 9 14
        for (int i = 0; i < NUMROWS; i++){
            string[i + j*rows] = temp[i];
        }
    }
}

__device__ void cuda_str2arr(unsigned char ** &strArr, int rows, int cols, unsigned char * string){
    //stores string into array
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            strArr[i][j] = string[i + j*rows]; //4 bytes go down the first column before moving to column 2
        }
    }
}

__device__ void cuda_arr2str(unsigned char ** &strArr, int rows, int cols, unsigned char * string){
    //stores back in string
    for (int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            string[i + j*rows] = strArr[i][j];
        }
    }
}

__device__ void cuda_addRoundKey(unsigned char * &string, unsigned char * key, int rows, int cols){
    //just does xor with given key
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            string[i + j*rows] ^= key[i + j*rows]; //4 bytes go down the first column before moving to column 2
        }
    }
}

__device__ void cuda_encryptString(unsigned char * string, unsigned char * key, int len, int keylen, unsigned char * expandedKeys, int numRounds){
    int stringSize = len; //length of string in bytes (must be 16)
    int keySize = keylen; //length of key in bytes (16, 24, 32)
    int numRows = NUMROWS; //constant number of rows
    int numCols = stringSize/numRows; //number of bytes/number of rows should be number of columns in generated array
    int keyCols = keySize/numRows; //same for keys
    int numKeys = numRounds; //number of keys = number of rounds
    int i;
    //round 0
    cuda_addRoundKey(string, &expandedKeys[0], numRows, numCols);
    
    for (i = 1; i < numKeys - 1; ++i){
        //round 1-10
        cuda_subBytes(string, numRows, numCols);
        cuda_shiftRows(string, numRows, numCols);
        cuda_mixCols(string, numRows, numCols);
        cuda_addRoundKey(string, &expandedKeys[i*keySize], numRows, numCols);
    }
    //round 11
    cuda_subBytes(string, numRows, numCols);
    cuda_shiftRows(string, numRows, numCols);
    cuda_addRoundKey(string, &expandedKeys[(numKeys - 1)*keySize], numRows, numCols);

}

__device__ void cuda_decryptString(unsigned char * string, unsigned char * key, int len, int keylen, unsigned char * expandedKeys, int numRounds){
    int stringSize = len; //length of string in bytes (must be 16)
    int keySize = keylen; //length of key in bytes (16, 24, 32)
    const int numRows = NUMROWS; //constant number of rows
    int numCols = stringSize/numRows; //number of bytes/number of rows should be number of columns in generated array
    int keyCols = keySize/numRows; //same for keys
    int numKeys = numRounds; //number of keys = number of rounds
    int i;
    //inv roud 11
    cuda_addRoundKey(string, &expandedKeys[(numKeys - 1)*keySize], numRows, numCols);
    cuda_invShiftRows(string, numRows, numCols);
    cuda_invSubBytes(string, numRows, numCols);

    for (i = numKeys - 2; i > 0; --i){
        //inv round 10 - 1
        cuda_addRoundKey(string, &expandedKeys[i*keySize], numRows, numCols);
        cuda_invMixCols(string, numRows, numCols);
        cuda_invShiftRows(string, numRows, numCols);
        cuda_invSubBytes(string, numRows, numCols);
    }
    //inv round 0
    cuda_addRoundKey(string, &expandedKeys[0], numRows, numCols);

}

__global__ void encryptDecryptNBlocksCuda(int start, int blocks, bool encrypt, unsigned char * dev_expandedKeys, unsigned char * dev_data, int aesBlockSize, unsigned char* dev_key, int keySize, int numRounds){
    int i = start + blockIdx.x*blockDim.x + threadIdx.x;
    //make sure not encrypting data that doesn't exist
    if (i < (start + blocks)){
        if (encrypt){
            cuda_encryptString(&dev_data[i*aesBlockSize], dev_key, aesBlockSize, keySize, dev_expandedKeys, numRounds);
        }
        else{
            cuda_decryptString(&dev_data[i*aesBlockSize], dev_key, aesBlockSize, keySize, dev_expandedKeys, numRounds);
        }
    }
}