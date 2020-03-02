all: aes
aes: aes_cuda.o aes_test.o aes_encryptor.o aes.o
	nvcc aes_cuda.o aes_test.o aes_encryptor.o aes.o -o aes -m64 -lineinfo
aes.o: aes.cpp
	nvcc -c aes.cpp -o aes.o -m64
aes_cuda.o: aes_cuda.cu
	nvcc -c aes_cuda.cu -o aes_cuda.o -m64 
aes_test.o: aes_test.cu
	nvcc -c aes_test.cu -o aes_test.o -m64 
aes_encryptor.o: aes_encryptor.cu
	nvcc -D_VARIADIC_MAX=10 -c aes_encryptor.cu -o aes_encryptor.o -m64 -lineinfo 
clean:
	del *.o
	del *.exe
	del *.dll
dll: aes_cuda.o aes_encryptor.o aes.o
	nvcc -o aes.dll --shared aes_cuda.o aes_encryptor.o aes.o -m64