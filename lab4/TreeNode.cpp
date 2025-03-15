#include <stdexcept>

#include "TreeNode.h"

TreeNode::TreeNode(initializer_list<double> coords) :coordinates(coords),right(nullptr),left(nullptr){
}

const double &TreeNode::operator[](int index) const {
  return getCoordinates()[index];
}

int TreeNode::dimension() const {
  return 2;
}

const vector<double> &TreeNode::getCoordinates() const {
    return coordinates;
}

TreeNode::~TreeNode() {} // Even though the implementation might be empty, it's
                         // a good practice to define it.