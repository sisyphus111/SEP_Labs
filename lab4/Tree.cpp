#include "Tree.h"
#include <algorithm>
#include <cassert>
#include <limits>
#include "Calculator.h"
#include "Comparator.h"
#include "TreeNode.h"
using namespace std;

/*
 * You can use this struct to compare a certain dimension of the tree nodes!
 * For example, if you want to compare the first dimension, you can simply call
 * DimComparator(0), such as DimComparator(0)(node_a, node_b).
 */
struct DimComparator {
  int dim;
  DimComparator(int d) : dim(d) {}
  bool operator()(TreeNode *a, TreeNode *b) const {
    assert(a->dimension() == b->dimension() && a->dimension() > dim);
    return isLessThan((*a)[dim], (*b)[dim]);
  }
};
BinaryDimenTree::BinaryDimenTree(DistanceCalculator *calculator):calculator(calculator),root(nullptr) {}//构造
BinaryDimenTree::~BinaryDimenTree() {
    clearall(this->root);
    root=nullptr;
}//析构
void BinaryDimenTree::clearall(TreeNode*node){
    if(node->left)clearall(node->left);
    if(node->right)clearall(node->right);
    delete node;
}//析构辅助函数
//void BinaryDimenTree::insert(TreeNode*& node, const std::vector<double>& point, int depth) {
//    if (!node) {
//        node = new TreeNode({point[0],point[1]});
//        return;
//    }
//    int cd = depth % 2; // 当前比较维度
//    if (isLessThan(point[cd], node->getCoordinates()[cd])) {
//        insert(node->left, point, depth + 1);
//    } else {
//        insert(node->right, point, depth + 1);
//    }
//}//插入结点，使用时从树根开始插入，故depth的默认值已设为0
TreeNode *BinaryDimenTree::find_nearest_node(TreeNode *target) {
    TreeNode*best=nullptr;
    double best_dist=std::numeric_limits<double>::max();
    this->find_nearest(this->root,target,calculator,0,best,best_dist);
    return best;
}



int partition(std::vector<pair<double,double>>& vec, int low, int high,int dim) {
    double pivot = dim==0?(vec[high].first):(vec[high].second); // 选择最后一个元素作为基准
    int i = (low - 1); // 比基准小的元素的索引

    for (int j = low; j <= high - 1; j++) {
        // 如果当前元素小于或等于基准
        if ((dim==0?(vec[j].first):(vec[j].second)) <= pivot) {
            i++; // 增加比基准小的元素的索引
            std::swap(vec[i], vec[j]);
        }
    }
    std::swap(vec[i + 1], vec[high]);
    return (i + 1);
}


// 快速排序的主函数
void quickSort(std::vector<pair<double,double>>& vec, int low, int high,int dim) {
    if (low < high) {
        // pi是分区索引，vec[pi]现在在正确的位置
        int pi = partition(vec, low, high,dim);
        // 分别递归地对分区前后的元素进行排序
        quickSort(vec, low, pi - 1,dim);
        quickSort(vec, pi + 1, high,dim);
    }
}


void InsertMedianAndDivide(TreeNode*&root,std::vector<pair<double,double>>& vec,int dim) {
    if (vec.empty()) return;
    quickSort(vec, 0, vec.size() - 1,dim);//按照dim快排
    // 插入中位数点
    int mid = vec.size() / 2;
    root=new TreeNode({vec[mid].first, vec[mid].second});
    // 如果只剩一点（已插入），就结束递归
    if (vec.size() == 1) return;

    // 创建两个子序列
    std::vector<pair<double,double>> left(vec.begin(), vec.begin() + mid);
    std::vector<pair<double,double>> right(vec.begin() + mid + 1, vec.end());

    // 递归处理子序列
    InsertMedianAndDivide(root->left,left,(dim+1)%2);
    InsertMedianAndDivide(root->right,right,(dim+1)%2);
}


istream &operator>>(istream &in, BinaryDimenTree &tree) {
    int n; // 结点个数
    in >> n;
    vector<pair<double,double>> points(n);
    for(int i = 0; i < n; i++) {
        double x, y;
        in >> x >> y;
        points[i] = make_pair(x, y); // 使用 make_pair 而不是 emplace_back
    }
    InsertMedianAndDivide(tree.root,points,0);

    return in;
}




void BinaryDimenTree::find_nearest(TreeNode* node,  TreeNode* target,  DistanceCalculator* calculator, int depth ,TreeNode*&best,double&best_dist) {

    if (!node) return;//递归到叶子之下，就返回
    int cd = depth % 2;//进行坐标比较的维度

    double dist = calculator->calculateDistance(*target, *node);//更新当前节点
    if(isLessThan(dist, best_dist)){
        best = node;
        best_dist = dist;
    }
    else if(isEqual(dist,best_dist)){
        if(DimComparator(0)(node,best)){
            best = node;
            best_dist = dist;
        }
        else if(isEqual(node->getCoordinates()[0],best->getCoordinates()[0])   &&    DimComparator(1)(node,best)){
            best = node;
            best_dist = dist;
        }
    }

    TreeNode* good_side = isLessThan(target->getCoordinates()[cd] , node->getCoordinates()[cd] )? node->left : node->right;//先遍历好的那边
    TreeNode* bad_side = isLessThan(target->getCoordinates()[cd] , node->getCoordinates()[cd]) ? node->right : node->left;
    find_nearest(good_side, target, calculator, depth + 1,best,best_dist);
    if (isGreaterThan(best_dist, calculator->verticalDistance(*target,*node,cd))) {//遍历完好的再剪枝
        find_nearest(bad_side, target, calculator, depth + 1,best,best_dist);
    }
}//寻找最近结点，使用时从树根开始，故depth的默认值已设为0