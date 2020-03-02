#include <string>
#include "aes_encryptor.h"
using namespace std;

int main(int argc, char *argv[]){
    string fileName;
    ifstream inFile;
    bool encrypt = true;

    if (argc > 1){
        fileName = argv[1];
    }
    if (argc > 2){
        if(argv[2][0] == 'd') encrypt = false;
    }

    if (fileName.empty()){
        cout << "No file specified." << endl << "Exiting..." << endl;
        return -1;
    }

    inFile.open(fileName);
    if (!inFile.good()){
        cout << "File does not exist." << endl << "Exiting..." << endl;
        return -1;
    }
    inFile.close();
    aesEncryptor encryptor(fileName,encrypt,8,true);

}