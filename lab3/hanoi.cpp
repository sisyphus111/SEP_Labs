#include <climits>
#include <iostream>
#include <string>

#include "board.h"

using namespace std;

int main() {
    bool initializing=true;
    while (true) {
        bool autoplay=false;
        Board *board;
        while (initializing == true) {
            cout << "How many disks do you want? (1 ~ 5)" << endl;
            string input;
            getline(cin, input);
            if (input == "Q") {
                return 0;
            }
            // TODO
            if (input == "1") {
                initializing = false;
                board = new Board(1);
            } else if (input == "2") {
                initializing = false;
                board = new Board(2);
            } else if (input == "3") {
                initializing = false;
                board = new Board(3);
            } else if (input == "4") {
                initializing = false;
                board = new Board(4);
            } else if (input == "5") {
                initializing = false;
                board = new Board(5);
            }
        }
        board->draw();
        while (autoplay == false) {
            cout << "Move a disk. Format: x y" << endl;
            string input;
            getline(cin, input);
            if (input[0] <= '3' && input[0] >= '1' && input[2] <= '3' && input[2] >= '1') {
                board->move(input[0] - '0', input[2] - '0', true);
                board->draw();
                if (board->win()) { cout << "Congratulations! You win!"<<std::endl;initializing=true;break;}
            }
            else if (input[0] == '0' && input[2] == '0'){autoplay = true;}
            else {board->draw();}
        }
        if (autoplay == true) {
            board->autoplay();
            cout << "Congratulations! You win!"<<std::endl;
            initializing=true;
        }
    }
    return 0;
}