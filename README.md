# AES CUDA
 AES implementation in C/C++ with C# windows form wrapper. C/C++ code uses parallelism in the forms of traditional CPU multithreading as well as GPU parallelism with Nvidia's CUDA API. The C/C++ code interfaces with the C# with P/Invoke using a DLL to export/import functions.

To compile, first cd to the cuda directory and run

	"nmake -f makefile dll"
	
Then open the Visual Studio solution and build :)
