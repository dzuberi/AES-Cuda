#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <mutex>
#include "aes_cuda.h"
using namespace std;
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      //if (abort) exit(code);
   }
}

class aesEncryptor {
    private:
        chrono::time_point<chrono::high_resolution_clock> start_time;
        chrono::time_point<chrono::high_resolution_clock> stop_time;
        string ifName; //input filename
        string ofName;
        unsigned char* chunk_data;
        char encryptionKey[32];
        int numRounds = 11;
        int numberOfThreads = 1;
        bool encrypt = 0;
        const int aesBlockSize = BLOCKSIZE;
        int keySize = 32;
        unsigned char * key;
        bool cuda = 1;
        void printDuration();
        void encryptDecryptNBlocks(int start, int blocks, bool encrypt, unsigned char * expandedKeys);
        void encryptDecryptChunkThreads(bool encrypt, int numBytes, unsigned char* &expandedKeys);
        void encryptDecryptChunkCuda(bool encrypt, int numBytes, unsigned char* &expandedKeys);
        void encryptDecryptChunking(bool encrypt);
        void writeOfChunk(unsigned char * &vec, int numBytes);
        bool firstOut = true;
    public:
        aesEncryptor(string ifNameStr, bool encrypt, int numThreads, bool cuda, unsigned char* key, int keyLength);
        aesEncryptor(string ifNameStr, string ofNameStr, bool encrypt, int numThreads,  bool cuda, unsigned char* key, int keyLength);
        ~aesEncryptor();
};


