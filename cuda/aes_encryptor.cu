#include "aes_encryptor.h"
#include <thread>
//#define _VARIADIC_MAX 10
aesEncryptor::aesEncryptor(string ifNameStr, bool encrypt, int numThreads, bool cuda, unsigned char* key, int keyLength) //constructor with no specified output file
:ifName(ifNameStr),ofName(ifNameStr + ".aes"),encrypt(encrypt),numberOfThreads(numThreads),cuda(cuda),key(key),keySize(keyLength){
    numRounds = (keySize == 32) ? 14: 11;
    start_time = chrono::high_resolution_clock::now();
    this -> encryptDecryptChunking(encrypt);
}

aesEncryptor::aesEncryptor(string ifNameStr, string ofNameStr, bool encrypt, int numThreads, bool cuda, unsigned char* key, int keyLength) //construction with specified output file
:ifName(ifNameStr),ofName(ofNameStr),encrypt(encrypt),numberOfThreads(numThreads),cuda(cuda),key(key),keySize(keyLength){
    numRounds = (keySize == 32) ? 14: 11;
    start_time = chrono::high_resolution_clock::now();
    this -> encryptDecryptChunking(encrypt);
}

aesEncryptor::~aesEncryptor(){
}

void aesEncryptor::printDuration(){ //prints the time elapsed since constructor call to stdout
    stop_time = chrono::high_resolution_clock::now();
    chrono::milliseconds d = chrono::duration_cast<chrono::milliseconds>(stop_time - start_time);
    cout << "Time so far: " << d.count() << " milliseconds" << endl;
}

void aesEncryptor::encryptDecryptNBlocks(int start, int blocks, bool encrypt, unsigned char * expandedKeys){ //CPU (no cuda) encryption with specified block range
    int lower = aesBlockSize*start;
    int upper = aesBlockSize*(start + blocks);
    for (int i = lower; i < upper; i += aesBlockSize){
        if (encrypt){
            ::encryptString(&chunk_data[i], key, aesBlockSize, keySize, expandedKeys, numRounds);
        }
        else{
            ::decryptString(&chunk_data[i], key, aesBlockSize, keySize, expandedKeys, numRounds);
        }
    }
}

void aesEncryptor::encryptDecryptChunkThreads(bool encrypt, int numBlocks, unsigned char* &expandedKeys){ //takes a chunk and splits it between threads based on the numberOfThreads variable
    //cout << "not cuda" << endl;
    vector<thread> threads;
    int threadBlocks, threadStart;
    threadStart = 0;
    for(int i = 0; i < numberOfThreads; ++i){
        threadBlocks = numBlocks/numberOfThreads;
        if (i == 0)
            threadBlocks += (numBlocks%numberOfThreads); //in case threadBlocks isn't divisible
        threads.push_back(thread(&aesEncryptor::encryptDecryptNBlocks, this, threadStart, threadBlocks, encrypt, expandedKeys)); //create thread
        threadStart += threadBlocks;
    }
    for(thread &t : threads){
        t.join();//wait here until threads are complete
    }
}

void aesEncryptor::encryptDecryptChunkCuda(bool encrypt, int numBlocks, unsigned char* &expandedKeys){ //takes a chunk and sends it to the GPU
    unsigned char * dev_data;
    unsigned char * dev_expandedKeys;
    unsigned char * dev_key;
    //copy data to dev_data
    gpuErrchk( cudaMalloc((void**)&dev_data,aesBlockSize*numBlocks*sizeof(unsigned char)) );
    gpuErrchk( cudaMemcpyAsync(dev_data,chunk_data,aesBlockSize*numBlocks*sizeof(unsigned char),cudaMemcpyHostToDevice) );
    //copy expandedKeys
    gpuErrchk( cudaMalloc((void**)&dev_expandedKeys,keySize*numRounds*sizeof(unsigned char)) );
    gpuErrchk( cudaMemcpy(dev_expandedKeys,expandedKeys,keySize*numRounds*sizeof(unsigned char),cudaMemcpyHostToDevice) );
    //copy key
    gpuErrchk( cudaMalloc((void**)&dev_key,keySize*sizeof(unsigned char)) );
    gpuErrchk( cudaMemcpy(dev_key,key,keySize*sizeof(unsigned char),cudaMemcpyHostToDevice) );
    gpuErrchk( cudaDeviceSynchronize() ); //this is already implied by Memcpy
    //run kernel
    int threads = 32;
    encryptDecryptNBlocksCuda<<<(numBlocks + threads - 1)/threads,threads>>>(0, numBlocks,encrypt,dev_expandedKeys,dev_data,aesBlockSize,dev_key,keySize,numRounds);
    gpuErrchk( cudaPeekAtLastError() );
    //copy data back
    //gpuErrchk( cudaThreadSynchronize() );
    gpuErrchk( cudaMemcpyAsync(chunk_data,dev_data,aesBlockSize*numBlocks*sizeof(unsigned char),cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaDeviceSynchronize() );
    //free memory
    cudaFree(dev_data);
    cudaFree(dev_expandedKeys);
    cudaFree(dev_key);
}

void aesEncryptor::encryptDecryptChunking(bool encrypt){ //reads input file, decrypts, and writes output file in chunks so massive files are not loaded into memory
    //generate expandedKeys only once
    unsigned char * expandedKeys = new unsigned char[keySize*numRounds];
    ::expandKeys(key, expandedKeys, keySize, numRounds);
    //create file stream
    basic_ifstream<unsigned char> fin(ifName, std::ios_base::binary);
    constexpr size_t bufferSize = 128*1024*1024; //total amount of bytes loaded into memory at once
    unique_ptr<unsigned char []> buffer(new unsigned char[bufferSize]);
    while (!fin.eof()) {
        chunk_data = buffer.get();
        fin.read(chunk_data, bufferSize); //read chunk into buffer
        std::streamsize dataSize = fin.gcount();
        //cout << dataSize << " byte chunk" << endl;
        int dataBytes = dataSize;
        int numBlocks = dataBytes / aesBlockSize;
        if(numBlocks > 0){ //run chunk
            if(cuda) encryptDecryptChunkCuda(encrypt, numBlocks, expandedKeys);
            else encryptDecryptChunkThreads(encrypt, numBlocks, expandedKeys);
        }
        writeOfChunk(chunk_data, dataBytes); //write chunk to output
    }
    fin.close();
    printDuration();
    delete[] expandedKeys;
}

void aesEncryptor::writeOfChunk(unsigned char * &vec, int numBytes){ //writes a chunk into output file. Append if not first time writing
    ofstream fout(ofName, (firstOut) ? ios::out | ios::binary : ios::app | ios::binary);
    fout.write((char*)&vec[0], numBytes * sizeof(unsigned char));
    fout.close();
    cudaGetLastError();
    firstOut = false;
}

//export functions to C#
extern "C" {

    __declspec(dllexport) int __cdecl test(int number){
            return number + 1;
    }

    __declspec(dllexport)  aesEncryptor* __cdecl createNew(char* fileName, bool encrypt, int numThreads, bool cuda, unsigned char* key, int keyLength){
        string fileNameStr(fileName);
        aesEncryptor* objectPtr = new aesEncryptor(fileNameStr, encrypt, numThreads, cuda, key, keyLength);
        return objectPtr;
    }

    __declspec(dllexport)  aesEncryptor* __cdecl createNewWithOF(char* fileName, char* ofName, bool encrypt, int numThreads, bool cuda, unsigned char* key, int keyLength){
        string fileNameStr(fileName);
        string ofNameStr(ofName);
        aesEncryptor* objectPtr = new aesEncryptor(fileNameStr, ofNameStr, encrypt, numThreads, cuda, key, keyLength);
        return objectPtr;
    }

    __declspec(dllexport)  void __cdecl destroy(aesEncryptor* objectPtr){
        if(objectPtr != NULL){
            objectPtr->~aesEncryptor();
            delete objectPtr;
            objectPtr = NULL;
        }
    }

}