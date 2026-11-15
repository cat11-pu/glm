/// @ref gtc_matrix_inverse

#include <cmath>
#include <limits>

namespace glm
{
	namespace detail
	{
		// One-sided Jacobi SVD of a small squared matrix.
		// On output a = U * Sigma and v holds the right singular vectors, so that
		// m = a * transpose(v_original). Jacobi rotations are individually well
		// conditioned, which keeps the decomposition accurate near singularity.
		template<length_t N, typename T, qualifier Q>
		struct compute_jacobi_svd
		{
			GLM_FUNC_QUALIFIER static void call(mat<N, N, T, Q>& a, mat<N, N, T, Q>& v)
			{
				v = mat<N, N, T, Q>(static_cast<T>(1));

				for(int sweep = 0; sweep < 30; ++sweep)
				{
					T offDiagonal = static_cast<T>(0);
					for(length_t p = 0; p < N - 1; ++p)
						for(length_t q = p + 1; q < N; ++q)
							offDiagonal += abs(dot(a[p], a[q]));
					if(offDiagonal < epsilon<T>())
						return;

					for(length_t p = 0; p < N - 1; ++p)
					{
						for(length_t q = p + 1; q < N; ++q)
						{
							T const alpha = dot(a[p], a[p]);
							T const beta = dot(a[q], a[q]);
							T const gamma = dot(a[p], a[q]);

							if(abs(gamma) <= epsilon<T>())
								continue;

							T cs;
							T sn;
							if(alpha <= epsilon<T>())
							{
								// Column p is numerically zero: a plain 90 degree
								// rotation separates it from an (exactly) collinear
								// column without a 0/0 angle computation.
								cs = static_cast<T>(0);
								sn = static_cast<T>(1);
							}
							else
							{
								// Standard one-sided Jacobi rotation that zeroes the
								// off-diagonal Gram-matrix entry between columns p, q.
								T const zeta = (alpha - beta) / (static_cast<T>(2) * gamma);
								T const t = ((zeta >= static_cast<T>(0))
										? static_cast<T>(1) : static_cast<T>(-1))
									/ (abs(zeta) + sqrt(static_cast<T>(1) + zeta * zeta));
								cs = inversesqrt(static_cast<T>(1) + t * t);
								sn = t * cs;
							}

							vec<N, T, Q> const ap = a[p];
							vec<N, T, Q> const aq = a[q];
							a[p] = cs * ap + sn * aq;
							a[q] = -sn * ap + cs * aq;

							vec<N, T, Q> const vp = v[p];
							vec<N, T, Q> const vq = v[q];
							v[p] = cs * vp + sn * vq;
							v[q] = -sn * vp + cs * vq;
						}
					}
				}
			}
		};
	}//namespace detail

	template<length_t N, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<N, T, Q> singularValues(mat<N, N, T, Q> const& m)
	{
		mat<N, N, T, Q> a = m;
		mat<N, N, T, Q> v;
		detail::compute_jacobi_svd<N, T, Q>::call(a, v);

		vec<N, T, Q> sigma;
		for(length_t i = 0; i < N; ++i)
			sigma[i] = length(a[i]);

		// Sort the singular values in descending order with an insertion sort;
		// N is at most 4, so a simple sort is both sufficient and branch-light.
		for(length_t i = 1; i < N; ++i)
		{
			T const value = sigma[i];
			length_t j = i;
			while(j > 0 && sigma[j - 1] < value)
			{
				sigma[j] = sigma[j - 1];
				--j;
			}
			sigma[j] = value;
		}
		return sigma;
	}

	template<length_t N, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T matrixCondition(mat<N, N, T, Q> const& m)
	{
		vec<N, T, Q> const sigma = singularValues<N, T, Q>(m);
		T const zeroFloor = epsilon<T>() * static_cast<T>(N) * sigma[0];
		if(sigma[N - 1] <= zeroFloor)
			return std::numeric_limits<T>::infinity();
		return sigma[0] / sigma[N - 1];
	}

	template<length_t N, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<N, N, T, Q> pseudoinverse(mat<N, N, T, Q> const& m, T rcond)
	{
		mat<N, N, T, Q> a = m;
		mat<N, N, T, Q> v;
		detail::compute_jacobi_svd<N, T, Q>::call(a, v);

		// Build the index permutation that orders the singular values
		// in descending order; N is at most 4 so insertion sort is enough.
		length_t order[N];
		for(length_t i = 0; i < N; ++i)
			order[i] = i;
		for(length_t i = 1; i < N; ++i)
		{
			length_t const current = order[i];
			T const value = length(a[current]);
			length_t j = i;
			while(j > 0 && length(a[order[j - 1]]) < value)
			{
				order[j] = order[j - 1];
				--j;
			}
			order[j] = current;
		}

		vec<N, T, Q> sigma;
		mat<N, N, T, Q> u(static_cast<T>(0));
		mat<N, N, T, Q> vSorted(static_cast<T>(0));
		for(length_t i = 0; i < N; ++i)
		{
			length_t const source = order[i];
			sigma[i] = length(a[source]);
			vSorted[i] = v[source];
			if(sigma[i] > static_cast<T>(0))
				u[i] = a[source] / sigma[i];
		}

		T const zeroFloor = epsilon<T>() * static_cast<T>(N) * sigma[0];
		T const cutoff = max(rcond * sigma[0], zeroFloor);

		// A+ = V diag(invSigma) transpose(U); singular values at or below the
		// cutoff are inverted to zero instead of to an unbounded quantity.
		mat<N, N, T, Q> result(static_cast<T>(0));
		for(length_t i = 0; i < N; ++i)
		{
			T const invSigma = (sigma[i] > cutoff) ? (static_cast<T>(1) / sigma[i]) : static_cast<T>(0);
			for(length_t r = 0; r < N; ++r)
				for(length_t c = 0; c < N; ++c)
					result[c][r] += vSorted[i][r] * u[i][c] * invSigma;
		}
		return result;
	}

	template<length_t N, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<N, N, T, Q> pseudoinverse(mat<N, N, T, Q> const& m)
	{
		return pseudoinverse<N, T, Q>(m, static_cast<T>(0));
	}

	template<length_t N, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<N, N, T, Q> stableInverse(mat<N, N, T, Q> const& m, T rcond, T conditionThreshold)
	{
		T const condition = matrixCondition<N, T, Q>(m);
		if(condition < conditionThreshold)
			return inverse(m);
		return pseudoinverse<N, T, Q>(m, rcond);
	}

	template<length_t N, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<N, N, T, Q> stableInverse(mat<N, N, T, Q> const& m)
	{
		return stableInverse<N, T, Q>(m, static_cast<T>(0),
			static_cast<T>(1) / epsilon<T>());
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 3, T, Q> affineInverse(mat<3, 3, T, Q> const& m)
	{
		mat<2, 2, T, Q> const Inv(inverse(mat<2, 2, T, Q>(m)));

		return mat<3, 3, T, Q>(
			vec<3, T, Q>(Inv[0], static_cast<T>(0)),
			vec<3, T, Q>(Inv[1], static_cast<T>(0)),
			vec<3, T, Q>(-Inv * vec<2, T, Q>(m[2]), static_cast<T>(1)));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<4, 4, T, Q> affineInverse(mat<4, 4, T, Q> const& m)
	{
		mat<3, 3, T, Q> const Inv(inverse(mat<3, 3, T, Q>(m)));

		return mat<4, 4, T, Q>(
			vec<4, T, Q>(Inv[0], static_cast<T>(0)),
			vec<4, T, Q>(Inv[1], static_cast<T>(0)),
			vec<4, T, Q>(Inv[2], static_cast<T>(0)),
			vec<4, T, Q>(-Inv * vec<3, T, Q>(m[3]), static_cast<T>(1)));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q> inverseTranspose(mat<2, 2, T, Q> const& m)
	{
		T Determinant = m[0][0] * m[1][1] - m[1][0] * m[0][1];

		mat<2, 2, T, Q> Inverse(
			+ m[1][1] / Determinant,
			- m[0][1] / Determinant,
			- m[1][0] / Determinant,
			+ m[0][0] / Determinant);

		return Inverse;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 3, T, Q> inverseTranspose(mat<3, 3, T, Q> const& m)
	{
		T Determinant =
			+ m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
			- m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
			+ m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

		mat<3, 3, T, Q> Inverse;
		Inverse[0][0] = + (m[1][1] * m[2][2] - m[2][1] * m[1][2]);
		Inverse[0][1] = - (m[1][0] * m[2][2] - m[2][0] * m[1][2]);
		Inverse[0][2] = + (m[1][0] * m[2][1] - m[2][0] * m[1][1]);
		Inverse[1][0] = - (m[0][1] * m[2][2] - m[2][1] * m[0][2]);
		Inverse[1][1] = + (m[0][0] * m[2][2] - m[2][0] * m[0][2]);
		Inverse[1][2] = - (m[0][0] * m[2][1] - m[2][0] * m[0][1]);
		Inverse[2][0] = + (m[0][1] * m[1][2] - m[1][1] * m[0][2]);
		Inverse[2][1] = - (m[0][0] * m[1][2] - m[1][0] * m[0][2]);
		Inverse[2][2] = + (m[0][0] * m[1][1] - m[1][0] * m[0][1]);
		Inverse /= Determinant;

		return Inverse;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<4, 4, T, Q> inverseTranspose(mat<4, 4, T, Q> const& m)
	{
		T SubFactor00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
		T SubFactor01 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
		T SubFactor02 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
		T SubFactor03 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
		T SubFactor04 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
		T SubFactor05 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
		T SubFactor06 = m[1][2] * m[3][3] - m[3][2] * m[1][3];
		T SubFactor07 = m[1][1] * m[3][3] - m[3][1] * m[1][3];
		T SubFactor08 = m[1][1] * m[3][2] - m[3][1] * m[1][2];
		T SubFactor09 = m[1][0] * m[3][3] - m[3][0] * m[1][3];
		T SubFactor10 = m[1][0] * m[3][2] - m[3][0] * m[1][2];
		T SubFactor11 = m[1][0] * m[3][1] - m[3][0] * m[1][1];
		T SubFactor12 = m[1][2] * m[2][3] - m[2][2] * m[1][3];
		T SubFactor13 = m[1][1] * m[2][3] - m[2][1] * m[1][3];
		T SubFactor14 = m[1][1] * m[2][2] - m[2][1] * m[1][2];
		T SubFactor15 = m[1][0] * m[2][3] - m[2][0] * m[1][3];
		T SubFactor16 = m[1][0] * m[2][2] - m[2][0] * m[1][2];
		T SubFactor17 = m[1][0] * m[2][1] - m[2][0] * m[1][1];

		mat<4, 4, T, Q> Inverse;
		Inverse[0][0] = + (m[1][1] * SubFactor00 - m[1][2] * SubFactor01 + m[1][3] * SubFactor02);
		Inverse[0][1] = - (m[1][0] * SubFactor00 - m[1][2] * SubFactor03 + m[1][3] * SubFactor04);
		Inverse[0][2] = + (m[1][0] * SubFactor01 - m[1][1] * SubFactor03 + m[1][3] * SubFactor05);
		Inverse[0][3] = - (m[1][0] * SubFactor02 - m[1][1] * SubFactor04 + m[1][2] * SubFactor05);

		Inverse[1][0] = - (m[0][1] * SubFactor00 - m[0][2] * SubFactor01 + m[0][3] * SubFactor02);
		Inverse[1][1] = + (m[0][0] * SubFactor00 - m[0][2] * SubFactor03 + m[0][3] * SubFactor04);
		Inverse[1][2] = - (m[0][0] * SubFactor01 - m[0][1] * SubFactor03 + m[0][3] * SubFactor05);
		Inverse[1][3] = + (m[0][0] * SubFactor02 - m[0][1] * SubFactor04 + m[0][2] * SubFactor05);

		Inverse[2][0] = + (m[0][1] * SubFactor06 - m[0][2] * SubFactor07 + m[0][3] * SubFactor08);
		Inverse[2][1] = - (m[0][0] * SubFactor06 - m[0][2] * SubFactor09 + m[0][3] * SubFactor10);
		Inverse[2][2] = + (m[0][0] * SubFactor07 - m[0][1] * SubFactor09 + m[0][3] * SubFactor11);
		Inverse[2][3] = - (m[0][0] * SubFactor08 - m[0][1] * SubFactor10 + m[0][2] * SubFactor11);

		Inverse[3][0] = - (m[0][1] * SubFactor12 - m[0][2] * SubFactor13 + m[0][3] * SubFactor14);
		Inverse[3][1] = + (m[0][0] * SubFactor12 - m[0][2] * SubFactor15 + m[0][3] * SubFactor16);
		Inverse[3][2] = - (m[0][0] * SubFactor13 - m[0][1] * SubFactor15 + m[0][3] * SubFactor17);
		Inverse[3][3] = + (m[0][0] * SubFactor14 - m[0][1] * SubFactor16 + m[0][2] * SubFactor17);

		T Determinant =
			+ m[0][0] * Inverse[0][0]
			+ m[0][1] * Inverse[0][1]
			+ m[0][2] * Inverse[0][2]
			+ m[0][3] * Inverse[0][3];

		Inverse /= Determinant;

		return Inverse;
	}
}//namespace glm
