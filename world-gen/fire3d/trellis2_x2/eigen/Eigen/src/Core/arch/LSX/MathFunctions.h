#ifndef EIGEN_MATH_FUNCTIONS_LSX_H
#define EIGEN_MATH_FUNCTIONS_LSX_H

/* The sin and cos functions of this file are loosely derived from
 * Julien Pommier's sse math library: http://gruntthepeon.free.fr/ssemath/
 */

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

EIGEN_DOUBLE_PACKET_FUNCTION(atanh, Packet2d)
EIGEN_DOUBLE_PACKET_FUNCTION(log, Packet2d)
EIGEN_DOUBLE_PACKET_FUNCTION(log2, Packet2d)
EIGEN_DOUBLE_PACKET_FUNCTION(tanh, Packet2d)

EIGEN_FLOAT_PACKET_FUNCTION(atanh, Packet4f)
EIGEN_FLOAT_PACKET_FUNCTION(log, Packet4f)
EIGEN_FLOAT_PACKET_FUNCTION(log2, Packet4f)
EIGEN_FLOAT_PACKET_FUNCTION(tanh, Packet4f)

EIGEN_GENERIC_PACKET_FUNCTION(atan, Packet2d)
EIGEN_GENERIC_PACKET_FUNCTION(atan, Packet4f)
EIGEN_GENERIC_PACKET_FUNCTION(exp2, Packet2d)
EIGEN_GENERIC_PACKET_FUNCTION(exp2, Packet4f)

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_MATH_FUNCTIONS_LSX_H
