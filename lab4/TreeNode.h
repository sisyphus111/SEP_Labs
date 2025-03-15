#pragma once

#include <initializer_list>
#include <vector>

using namespace std;

class TreeNode {
  friend class BinaryDimenTree;

private:
  vector<double> coordinates;


public:
    TreeNode *left;
    TreeNode *right;
  /* DO NOT CHANGE SIGNATURE*/
  TreeNode(initializer_list<double> coords);

  /* DO NOT CHANGE SIGNATURE*/
  const double &operator[](int index) const;

  /* DO NOT CHANGE SIGNATURE*/
  int dimension() const;

  /* DO NOT CHANGE SIGNATURE*/
  const vector<double> &getCoordinates() const;

  /* DO NOT CHANGE SIGNATURE*/
  ~TreeNode(); // Even though empty, defined for completeness.
};