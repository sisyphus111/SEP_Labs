#pragma once

#include <string>

using std::string;

class Line {
public:
    Line * prev;
    Line * next;
    string line;
    Line(Line*,Line*,string);
    Line();
};

class Buffer {
private:
    Line *headLine, *tailLine,*currentLine;

public:
    int currentLineNum;
    int length;
    Buffer();
    ~Buffer();

    void writeToFile(const string &filename) const;
    void swap(int a,int b);
    // When we need `const`?
    const string &moveToLine(int idx) ;//为何有&？
    void showLines(int from, int to) ;
    void delete_pure(int from , int to);//单纯删
    void deleteLines(int from, int to);
    void insertLine(const string &text);
    void appendLine(const string &text);
    Line* find(int n);
    // for test
    void printAddr(int idx) const;
    void loadAddr2(int one, int another) const;
    void testSwap(int one, int another) const;
};
