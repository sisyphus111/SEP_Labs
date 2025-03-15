#include <iostream>
#include "Buffer.h"
#include <fstream>
//TODO: your code here
//implement the functions in Buffer
Line::Line(Line*a,Line*b,std::string c):prev(a),next(b),line(c){};//这里用string、std：：string有区别吗
Line::Line():prev(nullptr),next(nullptr),line(""){}
Buffer::Buffer() {
    headLine=new Line;
    tailLine=new Line;
    headLine->next=tailLine;
    tailLine->prev=headLine;
    length=0;
    currentLineNum=0;
    currentLine=headLine;
}

Buffer::~Buffer() {//其他非动态申请的成员需要也delete掉释放内存吗？
    //将头尾之间的东西全部删光：
    delete_pure(1,length);
    length=0;
    delete tailLine;
    delete headLine;
}
Line* Buffer::find(int n){
    if(n<=0 || n>length){throw std::out_of_range("Line number out of range");}
    else {
        Line *p=headLine;
        for(int i=0;i<n;i++)p=p->next;//p指向第n个
        return p;
    }
}
void Buffer::writeToFile(const string &filename) const {//正确性存疑
    std::ofstream stfile(filename);
    if(stfile){
        int sum=0;
        Line *p=headLine;
        for(int i=1;i<=length;i++) {
            p=p->next;
            stfile<<p->line<<'\n';
            sum+=p->line.size();
            sum++;
        }
        std::cout<<sum<<" byte(s) written"<<std::endl;
    }
    else throw "Filename not specified";
}

void Buffer::showLines(int from, int to)  {
    if(from>to)throw std::range_error("Number range error");
    else if(from<=0 || from>length || to<=0 || to>length)throw std::out_of_range("Line number out of range");
    else {
        Line *p=find(from);
        for(int i=from;i<=to;i++){
            std::cout<<i<<'\t'<<p->line<<std::endl;
            p=p->next;
        }
        currentLineNum=to;
        currentLine=find(to);
    }
}
void Buffer::delete_pure(int from, int to){
    Line *p=headLine,*q;
    for(int i=1;i<=from;i++)p=p->next;//p points at from
    Line*tmp1=p->prev;//tmp1->tmp2
    for(int j=from;j<=to;j++){
        q=p->next;
        delete p;
        p=q;
    }
    Line *tmp2=p;
    tmp1->next=tmp2;
    tmp2->prev=tmp1;
}
void Buffer::deleteLines(int from, int to){//地址应该是对的
    if(from>to)throw std::range_error("Delete range error");
    else if(from<=0 || from>length || to<=0 || to>length)throw std::out_of_range("Line number out of range");
    else if(from==1 && to==length){
        currentLineNum=0;
        currentLine=headLine;
        //除了head和tail外全部删光
        delete_pure(1,length);
        length=0;
    }
    else {
        //先确定currentLineNum的最终值:
        if(from==1)currentLineNum=1;
        else if(to==length)currentLineNum=from-1;
        else currentLineNum=from;
        //再进行删除操作
        delete_pure(from,to);
        // 再更新length,currentLine的值:
        length-=to-from+1;
        currentLine=find(currentLineNum);
    }
}
void Buffer::insertLine(const string &text){
    length++;
    if(currentLineNum==0){
        currentLineNum=1;
        Line *p=headLine->next;//原本的第一项
        currentLine=new Line(headLine,p,text);
        headLine->next=currentLine;
        p->prev=currentLine;
    }
    else{//currentLineNum doesn't require refresh
        Line *p=currentLine->prev;//p,新的，currentLine
        Line *tmp=new Line(p,currentLine,text);
        p->next=tmp;
        currentLine->prev=tmp;
        currentLine=tmp;
        //std::cout<<"currentLineNum="<<currentLineNum;//for test
    }
}

void Buffer::appendLine(const string &text){
    length++;
    if(currentLineNum==0){
        currentLineNum=1;
        Line *p=headLine->next;//原本的第一项
        currentLine=new Line(headLine,p,text);
        headLine->next=currentLine;
        p->prev=currentLine;
    }
    else{
        currentLineNum++;
        Line *p=currentLine->next;//currentLine,新的,p
        Line *tmp=new Line(currentLine,p,text);
        currentLine->next=tmp;
        p->prev=tmp;
        currentLine=tmp;
        //std::cout<<"currentLineNum="<<currentLineNum;//for test
    }
}

const string &Buffer::moveToLine(int idx){
    if(idx<=0 || idx>length)throw std::out_of_range("Line number out of range");
    else{
        currentLineNum=idx;
        currentLine=find(idx);
        return currentLine->line;
    }
}
void Buffer::swap(int a,int b){
    if(a<=0 || a>length || b<=0 || b>length)throw std::out_of_range("Line number out of range");
    else {
        int tmp;
        if(a>b){tmp=a;a=b;b=tmp;}//a小b大
        if(a==b)return;
        else{//a<b
            if(a==b-1){
                Line *tmp2=find(a),*tmp3=find(b);
                Line *tmp1=tmp2->prev,*tmp4=tmp3->next;//tmp1,tmp2,tmp3,tmp4交换中间两个
                tmp1->next=tmp3;
                tmp3->prev=tmp1;
                tmp3->next=tmp2;
                tmp2->prev=tmp3;
                tmp2->next=tmp4;
                tmp4->prev=tmp2;
            }
            else if(a==b-2){
                Line *tmpmid=find(a+1),*tmp2=find(a),*tmp3=find(b);
                Line *tmp1=tmp2->prev,*tmp4=tmp3->next;//tmp1,tmp2,tmpmid,tmp3,tmp4,swap23
                tmp1->next=tmp3;
                tmp3->prev=tmp1;
                tmp3->next=tmpmid;
                tmpmid->prev=tmp3;
                tmpmid->next=tmp2;
                tmp2->prev=tmpmid;
                tmp2->next=tmp4;
                tmp4->prev=tmp2;
            }
            else{
                Line *tmpa=find(a),*tmpb=find(b);
                Line *tmp1=tmpa->prev,*tmp2=tmpa->next,*tmp3=tmpb->prev,*tmp4=tmpb->next;
                //tmp1,tmpa,tmp2...tmp3,tmpb,tmp4
                tmp1->next=tmpb;
                tmpb->prev=tmp1;
                tmpb->next=tmp2;
                tmp2->prev=tmpb;
                tmp3->next=tmpa;
                tmpa->prev=tmp3;
                tmpa->next=tmp4;
                tmp4->prev=tmpa;
            }
        }
        //先交换，然后更新currentLine
        if(currentLineNum==0)currentLine=headLine;
        else currentLine=find(currentLineNum);
    }
}
// for test, Don't modify
static void* aPtr = nullptr;
static void* bPtr = nullptr;

void Buffer::printAddr(int idx) const {

    int curLineNo = 1;
    Line *curLine = headLine;
    while (curLineNo < idx) {
        curLineNo += 1;
        curLine = curLine->next;
    }
    
    std::cout << idx << ":" << curLine << std::endl;
}

void Buffer::loadAddr2(int one, int another) const {
    if (one == another)
        return;

    int first = std::min(one, another);
    int second = std::max(one, another);

    Line *firstLine = nullptr, *secondLine = nullptr;

    int curLineNo = 1;
    Line *curLine = headLine;
    while (curLineNo <= second) {
        if (curLineNo == first)
            firstLine = curLine;
        if (curLineNo == second)
            secondLine = curLine;

        curLineNo += 1;
        curLine = curLine->next;
    }

    if (firstLine && secondLine) {
        aPtr = (void *)firstLine;
        bPtr = (void *)secondLine;
    }
}

void Buffer::testSwap(int one, int another) const {

    if (one == another)
        return;

    int first = std::min(one, another);
    int second = std::max(one, another);

    Line *firstLine = nullptr, *secondLine = nullptr;

    int curLineNo = 1;
    Line *curLine = headLine;
    while (curLineNo <= second) {
        if (curLineNo == first)
            firstLine = curLine;
        if (curLineNo == second)
            secondLine = curLine;

        curLineNo += 1;
        curLine = curLine->next;
    }

    if (firstLine && secondLine) {
        if (aPtr == secondLine && bPtr == firstLine)
            std::cout << "Swap 2 Nodes successfully!" << std::endl;
        else
            std::cout << "Swap 2 Nodes failed!" << std::endl;
        aPtr = nullptr;
        bPtr = nullptr;
    }
}