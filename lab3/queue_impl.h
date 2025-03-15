#ifndef QUEUE_IMPL_H
#define QUEUE_IMPL_H

template <typename T>
void Queue<T>::push(T t) {
    // TODO
    if(sz==0){
        sz++;
        /*tail=new Node<T>(t);
        head->next=tail;*/
        head=MakeUnique<Node<T>>(t);
        tail=head.get();
    }
    else {
        sz++;
        tail->next=MakeUnique<Node<T>>(t);
        tail=tail->next.get();
    }
}

template <typename T>
void Queue<T>::pop() {
    // TODO
    if(sz==0){}//怎么办？
    else {
        //head=std::move(head->next);
        sz--;
        if(sz==0){tail=nullptr;head=nullptr;}
        else {
            /*Node<T>*tmp=head->next.release();
            head.reset(nullptr);
            head.reset(tmp);*/
            head.reset(head->next.release());
        }

        //UniquePtr<T>tmp=std::move(head->next);
        //head=std::move(tmp);
    }
}

template <typename T>
T &Queue<T>::front() {
    // TODO
    if(sz==0){}//怎么办？
    else {
        return head->val;
    }
}

template <typename T>
bool Queue<T>::empty() const {
    // TODO
    if(/*head==nullptr*/sz==0)return true;
    else return false;
}

template <typename T>
size_t Queue<T>::size() const {
    // TODO
    return sz*sizeof(Node<T>);
}
#endif  // QUEUE_IMPL_H
