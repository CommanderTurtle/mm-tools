
#ifndef SPARSELU_RELAX_SNODE_H
#define SPARSELU_RELAX_SNODE_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

/**
 * \brief Identify the initial relaxed supernodes
 *
 * This routine is applied to a column elimination tree.
 * It assumes that the matrix has been reordered according to the postorder of the etree
 * \param n  the number of columns
 * \param et elimination tree
 * \param relax_columns Maximum number of columns allowed in a relaxed snode
 * \param descendants Number of descendants of each node in the etree
 * \param relax_end last column in a supernode
 */
template <typename Scalar, typename StorageIndex>
void SparseLUImpl<Scalar, StorageIndex>::relax_snode(const Index n, IndexVector& et, const Index relax_columns,
                                                     IndexVector& descendants, IndexVector& relax_end) {
  // compute the number of descendants of each node in the etree
  Index parent;
  relax_end.setConstant(emptyIdxLU);
  descendants.setZero();
  for (Index j = 0; j < n; j++) {
    parent = et(j);
    if (parent != n)  // not the dummy root
      descendants(parent) += descendants(j) + 1;
  }
  // Identify the relaxed supernodes by postorder traversal of the etree
  Index snode_start;  // beginning of a snode
  for (Index j = 0; j < n;) {
    parent = et(j);
    snode_start = j;
    while (parent != n && descendants(parent) < relax_columns) {
      j = parent;
      parent = et(j);
    }
    // Found a supernode in postordered etree, j is the last column
    relax_end(snode_start) = StorageIndex(j);  // Record last column
    j++;
    // Search for a new leaf
    while (descendants(j) != 0 && j < n) j++;
  }  // End postorder traversal of the etree
}

}  // end namespace internal

}  // end namespace Eigen
#endif
