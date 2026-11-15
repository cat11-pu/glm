/// @ref gtc_matrix_inverse
/// @file glm/gtc/matrix_inverse.hpp
///
/// @see core (dependence)
///
/// @defgroup gtc_matrix_inverse GLM_GTC_matrix_inverse
/// @ingroup gtc
///
/// Include <glm/gtc/matrix_inverse.hpp> to use the features of this extension.
///
/// Defines additional matrix inverting functions.

#pragma once

// Dependencies
#include "../detail/setup.hpp"
#include "../matrix.hpp"
#include "../mat2x2.hpp"
#include "../mat3x3.hpp"
#include "../mat4x4.hpp"
#include "../ext/scalar_constants.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_GTC_matrix_inverse extension included")
#endif

namespace glm
{
	/// @addtogroup gtc_matrix_inverse
	/// @{

	/// Fast matrix inverse for affine matrix.
	///
	/// @param m Input matrix to invert.
	/// @tparam genType Squared floating-point matrix: half, float or double. Inverse of matrix based of half-qualifier floating point value is highly inaccurate.
	/// @see gtc_matrix_inverse
	template<typename genType>
	GLM_FUNC_DECL genType affineInverse(genType const& m);

	/// Compute the inverse transpose of a matrix.
	///
	/// @param m Input matrix to invert transpose.
	/// @tparam genType Squared floating-point matrix: half, float or double. Inverse of matrix based of half-qualifier floating point value is highly inaccurate.
	/// @see gtc_matrix_inverse
	template<typename genType>
	GLM_FUNC_DECL genType inverseTranspose(genType const& m);

	/// Compute the singular values of a squared matrix, sorted in descending order.
	///
	/// The decomposition is performed with one-sided Jacobi rotations, which stay
	/// well conditioned even when the matrix is close to singular.
	///
	/// @param m Input matrix.
	/// @tparam N Number of columns and rows.
	/// @tparam T A floating-point scalar type.
	/// @tparam Q A value from qualifier enum.
	/// @see gtc_matrix_inverse
	template<length_t N, typename T, qualifier Q>
	GLM_FUNC_DECL vec<N, T, Q> singularValues(mat<N, N, T, Q> const& m);

	/// Compute the 2-norm condition number of a squared matrix (sigma_max / sigma_min).
	///
	/// Returns positive infinity when the smallest singular value falls below the
	/// numerical-zero floor, i.e. when the matrix is numerically rank deficient.
	///
	/// @param m Input matrix.
	/// @tparam N Number of columns and rows.
	/// @tparam T A floating-point scalar type.
	/// @tparam Q A value from qualifier enum.
	/// @see gtc_matrix_inverse
	template<length_t N, typename T, qualifier Q>
	GLM_FUNC_DECL T matrixCondition(mat<N, N, T, Q> const& m);

	/// Compute the Moore-Penrose pseudoinverse of a squared matrix from its SVD.
	///
	/// Singular values smaller than max(rcond * sigma_max, epsilon * N * sigma_max)
	/// are treated as zero, so the result is always finite, including for rank
	/// deficient inputs.
	///
	/// @param m Input matrix.
	/// @param rcond Relative singular-value cutoff; 0 keeps only the numerical-zero floor.
	/// @tparam N Number of columns and rows.
	/// @tparam T A floating-point scalar type.
	/// @tparam Q A value from qualifier enum.
	/// @see gtc_matrix_inverse
	template<length_t N, typename T, qualifier Q>
	GLM_FUNC_DECL mat<N, N, T, Q> pseudoinverse(mat<N, N, T, Q> const& m, T rcond);

	/// Compute the Moore-Penrose pseudoinverse with the default relative cutoff.
	///
	/// @see gtc_matrix_inverse
	template<length_t N, typename T, qualifier Q>
	GLM_FUNC_DECL mat<N, N, T, Q> pseudoinverse(mat<N, N, T, Q> const& m);

	/// Stable inverse: returns the regular inverse while the condition number is
	/// below conditionThreshold, and falls back to the SVD pseudoinverse otherwise.
	///
	/// @param m Input matrix.
	/// @param rcond Relative singular-value cutoff used by the pseudoinverse fallback.
	/// @param conditionThreshold Condition number above which the fallback is used.
	/// @tparam N Number of columns and rows.
	/// @tparam T A floating-point scalar type.
	/// @tparam Q A value from qualifier enum.
	/// @see gtc_matrix_inverse
	template<length_t N, typename T, qualifier Q>
	GLM_FUNC_DECL mat<N, N, T, Q> stableInverse(mat<N, N, T, Q> const& m, T rcond, T conditionThreshold);

	/// Stable inverse with the default relative cutoff (0) and condition threshold
	/// (1 / epsilon).
	///
	/// @see gtc_matrix_inverse
	template<length_t N, typename T, qualifier Q>
	GLM_FUNC_DECL mat<N, N, T, Q> stableInverse(mat<N, N, T, Q> const& m);

	/// @}
}//namespace glm

#include "matrix_inverse.inl"
