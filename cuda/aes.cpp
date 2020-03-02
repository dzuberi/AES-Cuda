#include "aes.h"


unsigned char indexSbox(unsigned char string){ //helper function for lookup
    unsigned char upperHalf = (string & 0xf0) >> 4;
    unsigned char lowerHalf = (string & 0x0f);
    return s_box[upperHalf][lowerHalf];
}

unsigned char indexInvSbox(unsigned char string){  //helper function for inverse lookup
    unsigned char upperHalf = (string & 0xf0) >> 4;
    unsigned char lowerHalf = (string & 0x0f);
    return inv_s_box[upperHalf][lowerHalf];
}

void subBytes(unsigned char * &string, int rows, int cols){ //lookup table for sub bytes
    int i,j;
    for (i = 0; i < rows; i++){
        for (j = 0; j < cols; j++){
            string[i + j*rows] = indexSbox(string[i + j*rows]);
        }
    }
}

void invSubBytes(unsigned char * &string, int rows, int cols){ //lookup table for inv sub bytes
    int i,j;
    for (i = 0; i < rows; i++){
        for (j = 0; j < cols; j++){
            string[i + j*rows] = indexInvSbox(string[i + j*rows]);
        }
    }
}


void initArr(unsigned char ** &strArr, int rows, int cols){ //allocates an array of dim rows x cols
    strArr = new unsigned char*[rows];
    for (int i = 0; i < rows; i++){
        strArr[i] = new unsigned char[cols];
    }
    //cout << "array initialized" << endl;
}

void killArr(unsigned char ** strArr, int rows, int cols){ //deallocates an array of dim rows x cols
    for (int i = 0; i < rows; i++){
        delete [] strArr[i];
    }
    delete [] strArr;
}

void printArr(unsigned char ** strArr, int rows, int cols){ //prints array as hex int
    cout << endl;
    for (int i = 0; i < rows; i++){
        cout << std::hex;
        for (int j = 0; j < cols; j++){
            cout << (unsigned int) strArr[i][j] << " ";
        }
        cout << endl;
    }
}

void printStr(unsigned char * str, int len){ //prints string as hex int
    cout << endl;
    for (int i = 0; i < len; i++){
        cout << str[i];
    }
    cout << endl;
}

void shiftRows(unsigned char * &string, int rows, int cols){
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


void invShiftRows(unsigned char * &string, int rows, int cols){
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

unsigned char gfMul(unsigned char c, int factor){
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

void mixCols(unsigned char * &string, int rows, int cols){
    //mix columns as specified by the algorithm
    for (int j = 0; j < cols; j++){
        unsigned char temp[NUMROWS];
        temp[0] = gfMul(string[0 + j*rows], 2) ^ gfMul(string[1 + j*rows], 3) ^ gfMul(string[2 + j*rows], 1) ^ gfMul(string[3 + j*rows], 1); //2 3 1 1
        temp[1] = gfMul(string[0 + j*rows], 1) ^ gfMul(string[1 + j*rows], 2) ^ gfMul(string[2 + j*rows], 3) ^ gfMul(string[3 + j*rows], 1); //1 2 3 1
        temp[2] = gfMul(string[0 + j*rows], 1) ^ gfMul(string[1 + j*rows], 1) ^ gfMul(string[2 + j*rows], 2) ^ gfMul(string[3 + j*rows], 3); //1 1 2 3
        temp[3] = gfMul(string[0 + j*rows], 3) ^ gfMul(string[1 + j*rows], 1) ^ gfMul(string[2 + j*rows], 1) ^ gfMul(string[3 + j*rows], 2); //3 1 1 2
        for (int i = 0; i < NUMROWS; i++){
            string[i + j*rows] = temp[i];
        }
    }
}

void invMixCols(unsigned char * &string, int rows, int cols){
    //does the inverse of mix columns, as specified by the algorithm
    for (int j = 0; j < cols; j++){
        unsigned char temp[NUMROWS];
        temp[0] = gfMul(string[0 + j*rows], 14) ^ gfMul(string[1 + j*rows], 11) ^ gfMul(string[2 + j*rows], 13) ^ gfMul(string[3 + j*rows], 9); //14 11 13 9
        temp[1] = gfMul(string[0 + j*rows], 9) ^  gfMul(string[1 + j*rows], 14) ^ gfMul(string[2 + j*rows], 11) ^ gfMul(string[3 + j*rows], 13); //9 14 11 13
        temp[2] = gfMul(string[0 + j*rows], 13) ^ gfMul(string[1 + j*rows], 9) ^  gfMul(string[2 + j*rows], 14) ^ gfMul(string[3 + j*rows], 11); //13 9 14 11
        temp[3] = gfMul(string[0 + j*rows], 11) ^ gfMul(string[1 + j*rows], 13) ^ gfMul(string[2 + j*rows], 9) ^  gfMul(string[3 + j*rows], 14); //11 13 9 14
        for (int i = 0; i < NUMROWS; i++){
            string[i + j*rows] = temp[i];
        }
    }
}

void str2arr(unsigned char ** &strArr, int rows, int cols, unsigned char * string){
    //stores string into array
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            strArr[i][j] = string[i + j*rows]; //4 bytes go down the first column before moving to column 2
        }
    }
}

void arr2str(unsigned char ** &strArr, int rows, int cols, unsigned char * string){
    //stores back in string
    for (int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            string[i + j*rows] = strArr[i][j];
        }
    }
}

void addRoundKey(unsigned char * &string, unsigned char * key, int rows, int cols){
    //just does xor with given key
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            string[i + j*rows] ^= key[i + j*rows]; //4 bytes go down the first column before moving to column 2
        }
    }
}

void keyExpansionWord(unsigned char * in, unsigned char i){
    //circular byte shift
    unsigned char temp;
    temp = in[0];
    in[0] = in[1];
    in[1] = in[2];
    in[2] = in[3];
    in[3] = temp;
    //sbox sub
    for (int i = 0; i < 4; i++){
        in[i] = indexSbox(in[i]);
    }
    //rcon
    in[0] ^= rcon[i];
}

void expandKeys(unsigned char * inputKey, unsigned char * &expandedKeys, int keySize, int numKeys){ //generates expanded keys
    for(int i = 0; i < keySize; i++){
        expandedKeys[i] = inputKey[i];
    }
    int bytesDone = keySize; //key 0 is already done
    int keyNumber = 1; //0 is already done
    unsigned char word[4];

    while (bytesDone < numKeys*keySize){
        for(int i = 0; i < keySize/4; i++){
            word[i] = expandedKeys[i + bytesDone - 4]; //stores the last 4 bytes in word
        }
        if (bytesDone % keySize == 0){
            keyExpansionWord(word, keyNumber); //does the byte shift and rcon
            keyNumber++;
        }
        for (int i = 0; i < 4; i++){
            expandedKeys[bytesDone] = expandedKeys[bytesDone - keySize] ^ word[i]; //xor with word, word is altered if we are at a multiple of keysize
            bytesDone++;
        }
    }
}

void encryptString(unsigned char * string, unsigned char * key, int len, int keylen, unsigned char * expandedKeys, int numRounds){
    int stringSize = len; //length of string in bytes (must be 16)
    int keySize = keylen; //length of key in bytes (16, 24, 32)
    int numRows = NUMROWS; //constant number of rows
    int numCols = stringSize/numRows; //number of bytes/number of rows should be number of columns in generated array
    int keyCols = keySize/numRows; //same for keys

    unsigned char ** strArr; //2D array for the string (to do math on)
    int numKeys = numRounds; //number of keys = number of rounds
    int i,j;
    //round 0
    addRoundKey(string, &expandedKeys[0], numRows, numCols);
    for (i = 1; i < numKeys - 1; ++i){
        //round 1-10
        subBytes(string, numRows, numCols);
        shiftRows(string, numRows, numCols);
        mixCols(string, numRows, numCols);
        addRoundKey(string, &expandedKeys[i*keySize], numRows, numCols);
    }
    //round 11
    subBytes(string, numRows, numCols);
    shiftRows(string, numRows, numCols);
    addRoundKey(string, &expandedKeys[(numKeys - 1)*keySize], numRows, numCols);
}

void decryptString(unsigned char * string, unsigned char * key, int len, int keylen, unsigned char * expandedKeys, int numRounds){
    int stringSize = len; //length of string in bytes (must be 16)
    int keySize = keylen; //length of key in bytes (16, 24, 32)
    const int numRows = NUMROWS; //constant number of rows
    int numCols = stringSize/numRows; //number of bytes/number of rows should be number of columns in generated array
    int keyCols = keySize/numRows; //same for keys
    unsigned char ** strArr; //2D array for the string (to do math on)
    int numKeys = 11; //number of keys = number of rounds
    int i,j;
    //inv round 11
    addRoundKey(string, &expandedKeys[(numKeys - 1)*keySize], numRows, numCols);
    invShiftRows(string, numRows, numCols);
    invSubBytes(string, numRows, numCols);

    for (i = numKeys - 2; i > 0; --i){
        //inv round 10 - 1
        addRoundKey(string, &expandedKeys[i*keySize], numRows, numCols);
        invMixCols(string, numRows, numCols);
        invShiftRows(string, numRows, numCols);
        invSubBytes(string, numRows, numCols);
    }

    //inv round 1
    addRoundKey(string, &expandedKeys[0], numRows, numCols);
}