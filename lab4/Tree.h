#pragma once
#include <iostream>
#include <vector>
#include <limits>
#include "Calculator.h"
#include "Comparator.h"
#include "TreeNode.h"
using namespace std;
class BinaryDimenTree {
  /* DO NOT CHANGE SIGNATURE */
  friend istream &operator>>(istream &in,BinaryDimenTree &tree);
private:
  /* data */
  DistanceCalculator *calculator;

public:
    TreeNode *root;
  /* methods */
  void clearall(TreeNode* node);
  //void insert(TreeNode*& node, const std::vector<double>& point, int depth = 0);
  void find_nearest(TreeNode* node,  TreeNode* target,  DistanceCalculator* calculator, int depth,TreeNode*&best,double &best_dist);
    /* DO NOT CHANGE SIGNATURE */
  BinaryDimenTree(DistanceCalculator *calculator);
  /* DO NOT CHANGE SIGNATURE */
  TreeNode *find_nearest_node(TreeNode *target);
  /* DO NOT CHANGE SIGNATURE */
  ~BinaryDimenTree(); /* DO NOT CHANGE */
};
