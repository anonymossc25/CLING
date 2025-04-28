#include <iostream>
using namespace std;

#define ROWS 3
#define COLS 4

int main() {
    int array[ROWS][COLS] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12}
    };
    
    int **ptr = array; // Pointer to an array of COLS integers
    
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            cout << ptr[i][j] << " ";
        }
        cout << endl;
    }

    return 0;
}
