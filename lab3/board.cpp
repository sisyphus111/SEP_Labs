#include <iostream>
#include "board.h"
#include "queue.h"

Board::Board(const int num_disk) : num_disk(num_disk),rods(Rod(num_disk,1),Rod(num_disk,2),Rod(num_disk,3))/* TODO */ {
    // TODO
    disks=new Disk[num_disk];
    for(int i=0;i<=num_disk-1;i++){
        disks[i]=Disk(i+1,2*i+3);
    }
    for(int j=num_disk-1;j>=0;j--){
        rods[0].push(disks[j]);
    }
}

Board::~Board() {
    // TODO
    delete disks;
    history.~Stack();
}

void Board::draw() {
    Canvas canvas {};
    canvas.reset();
    // TODO
    //先画disk，再画rod，再画底板
    //画disk
    unsigned long long int size_of_rod[3]={
        rods[0].size()/sizeof(Node<Disk>),
        rods[1].size()/sizeof(Node<Disk>),
        rods[2].size()/sizeof(Node<Disk>)
    };
    //打印三个rod中的盘子，并利用disks数组作中转，先出栈再进栈


    for (int i0 = 1; i0 <= size_of_rod[0] ; i0++) {
        rods[0].top().draw(canvas,  size_of_rod[0]-i0, 0);
        disks[i0-1] = rods[0].top();
        rods[0].pop();
    }
    for (int j0 = size_of_rod[0] ; j0 >= 1; j0--) {
        rods[0].push(disks[j0-1]);
    }

    for (int i1 = 1; i1 <= size_of_rod[1]; i1++) {
        rods[1].top().draw(canvas, size_of_rod[1]-i1, 1);
        disks[i1-1] = rods[1].top();
        rods[1].pop();
    }
    for (int j1 = size_of_rod[1] ; j1 >= 1; j1--) {
        rods[1].push(disks[j1-1]);
    }

    for (int i2 = 1; i2 <= size_of_rod[2] ; i2++) {
        rods[2].top().draw(canvas, size_of_rod[2]-i2, 2);
        disks[i2-1] = rods[2].top();
        rods[2].pop();
    }
    for (int j2 = size_of_rod[2] ; j2 >= 1; j2--) {
        rods[2].push(disks[j2-1]);
    }

    //画rod
    rods[0].draw(canvas);
    rods[1].draw(canvas);
    rods[2].draw(canvas);
    //画底板
    for(int k=0;k<=40;k++)if(k!=5 && k!=20 && k!=35 )canvas.buffer[10][k]='-';
    canvas.draw();
}

void Board::move(int from, int to, const bool log) {//log=true表示手动
    // TODO
    if(log){//人工
        if(rods[from-1].top().val<rods[to-1].top().val && !rods[from-1].empty() && !rods[to-1].empty() || !rods[from-1].empty() && rods[to-1].empty()) {//输入合法
            history.push(std::pair{from, to});
            rods[to-1].push(rods[from-1].top());//先进到to里去，再from弹出，不知道会不会报错
            rods[from-1].pop();
        }
    }
    else {
        //history.push(std::pair{from, to});
        rods[to-1].push(rods[from-1].top());//先进到to里去，再from弹出，不知道会不会报错
        rods[from-1].pop();
    }//自动
}

bool Board::win() const {
    // TODO
    if(rods[0].empty() && rods[2].empty())return true;
    else return false;
}

void solve(//交给AI做的，正确性还未验证
    const int n,//个数
    const int src,//起始
    const int buf,//辅助
    const int dest,//终点
    Queue<std::pair<int, int>> &solution
) {
    // TODO

    if (n == 1) {
        solution.push(std::pair{src,dest});
        return;
    }
    solve(n - 1, src, dest, buf, solution);
    solution.push(std::pair{src,dest});
    solve(n - 1, buf, src, dest, solution);

}

void Board::autoplay() {
    // TODO
    //首先恢复原状并打印干了什么
    while(!history.empty()){
        int tmp_to=history.top().first;
        int tmp_from=history.top().second;
        move(tmp_from,tmp_to,false);
        history.pop();
        std::cout<<"Auto moving:"<<tmp_from<<"->"<<tmp_to<<std::endl;
        draw();
    }
    //然后先把问题解了，再一步一步开画
    Queue<std::pair<int, int>> solutionauto;
    solve(num_disk,1,3,2,solutionauto);
    while(!solutionauto.empty()){
        std::cout<<"Auto moving:"<<solutionauto.front().first<<"->"<<solutionauto.front().second<<std::endl;
        move(solutionauto.front().first,solutionauto.front().second,false);
        draw();
        solutionauto.pop();
    }
}