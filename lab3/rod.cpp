#include "rod.h"
#include <cstddef>
const Disk emptydisk(-1,-1);//表示栈为空
Rod::Rod(const int capacity, const int id) : capacity(capacity),id(id)/* TODO */ { }

bool Rod::push( Disk d) {
    // TODO
    if(d.val<this->top().val || this->empty()){
        stack.push(d);return true;//成功插入

    }
    else return false;
}

const Disk &Rod::top() {
    // TODO
    if(!stack.empty())return stack.top();
    else {return emptydisk;}//如果rod为空，那么应该返回什么？
}

void Rod::pop() {
    // TODO
    if(!stack.empty())stack.pop();
    //若栈为空，则啥也不干
}

size_t Rod::size() const {
    // TODO
    return (stack.size());
}

bool Rod::empty() const {
    // TODO
    return stack.empty();
}
bool Rod::full() const {
    // TODO
    if(stack.size()==capacity*sizeof(const Disk))return true;
    else return false;
}
void Rod::draw(Canvas &canvas) {
    int s_x=15*id-10;
    for(int y=0;y<=10;y++){
        if(canvas.buffer[y][s_x]!='*')canvas.buffer[y][s_x]='|';
    }
}
Rod::~Rod(){
    stack.~Stack();
};