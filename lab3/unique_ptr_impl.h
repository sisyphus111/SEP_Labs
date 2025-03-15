#ifndef UNIQUE_PTR_IMPL_H
#define UNIQUE_PTR_IMPL_H

// You need to implement follow functions, signatures are provided.
// NOTE: DON'T change the function definition

template <typename T>
UniquePtr<T>::UniquePtr(UniquePtr &&other) noexcept : pointer { other.release()/* TODO */ } { }

template <typename T>
UniquePtr<T>::~UniquePtr() {
    // TODO
    delete this->pointer;
}

template <typename T>
UniquePtr<T> &UniquePtr<T>::operator=(UniquePtr &&other) noexcept {
    // TODO
    if(this->get()!=other.get()){
        delete this->pointer;
        this->pointer=other.release();
    }
    else{
        this->pointer=other.release();
    }
    return *this;
}

template <typename T>
UniquePtr<T> &UniquePtr<T>::operator=(std::nullptr_t) noexcept {
    // TODO
    delete this->pointer;
    this->pointer=nullptr;
    return *this;
}

template <typename T>
void UniquePtr<T>::reset(T *ptr) noexcept {
    // TODO
    if(this->pointer!=ptr){delete this->pointer;this->pointer=ptr;}

}

template <typename T>
T *UniquePtr<T>::release() noexcept {
    // TODO
    T* releasedpointer=this->pointer;
    this->pointer=nullptr;
    return releasedpointer;
}

#endif  // UNIQUE_PTR_IMPL_H
