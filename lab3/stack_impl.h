#ifndef STACK_IMPL_H
#define STACK_IMPL_H
#include <cassert>

template <typename T>
void Stack<T>::push(T t) {
    // TODO
    if(sz==0){
        sz++;
        head=MakeUnique<Node<T>>(t);
    }
    else {
        sz++;
        //或：Node<T>*tmp=std::move(head);//先变成右值才能丢进移动构造函数
        /*Node<T>*tmp=head.release();
        head=MakeUnique<Node<T>>(t);
        head->next.reset(tmp);*/
        Node<T>*tmp=head.release();
        head=MakeUnique<Node<T>>(t);
        head->next.reset(tmp);
    }
}

template <typename T>
void Stack<T>::pop() {
    // TODO
    if(sz==0){}//怎么办？
    else {
        sz--;
        /*Node<T>*tmp=head->next.release();
        delete head.get();
        head.reset(tmp);*/
        head.reset(head->next.release());
    }
}

template <typename T>
T &Stack<T>::top() {
    // TODO
    if(sz==0){}//怎么办？
    else {
        return head->val;
    }
}

template <typename T>
bool Stack<T>::empty() const {
    // TODO
    if(/*head==nullptr*/sz==0)return true;
    else return false;
}

template <typename T>
size_t Stack<T>::size() const {
    // TODO
    if(!this->empty())return sizeof(Node<T>)*sz;
    else return 0;
}

#endif  // STACK_IMPL_H