#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/epsilon.hpp>
#include <cmath>
#include <limits>

template<int N>
static bool allFinite(glm::mat<N, N, float, glm::defaultp> const& m)
{
	for(glm::length_t c = 0; c < N; ++c)
		for(glm::length_t r = 0; r < N; ++r)
			if(!std::isfinite(m[c][r]))
				return false;
	return true;
}

template<int N>
static float maxAbsDiff(glm::mat<N, N, float, glm::defaultp> const& a,
	glm::mat<N, N, float, glm::defaultp> const& b)
{
	float error = 0.f;
	for(glm::length_t c = 0; c < N; ++c)
		for(glm::length_t r = 0; r < N; ++r)
			error = glm::max(error, std::abs(a[c][r] - b[c][r]));
	return error;
}

static int test_singular_values()
{
	int Error = 0;

	// Singular values of a diagonal matrix are its diagonal entries, sorted.
	glm::mat3 const diagonal(
		3.f, 0.f, 0.f,
		0.f, 2.f, 0.f,
		0.f, 0.f, 1.f);
	glm::vec3 const sigma = glm::singularValues(diagonal);
	glm::bvec3 const match = glm::epsilonEqual(sigma, glm::vec3(3.f, 2.f, 1.f), 1e-5f);
	Error += (match.x && match.y && match.z) ? 0 : 1;

	return Error;
}

static int test_pseudoinverse()
{
	int Error = 0;

	// On a well-conditioned matrix the pseudoinverse agrees with the inverse.
	{
		glm::mat3 const m(
			2.f, 1.f, 0.f,
			1.f, 3.f, 1.f,
			0.f, 1.f, 2.f);
		glm::mat3 const pinv = glm::pseudoinverse(m);
		Error += maxAbsDiff(pinv, glm::inverse(m)) < 1e-5f ? 0 : 1;
	}

	// Rank-deficient inputs report an infinite condition number and a finite
	// pseudoinverse that satisfies the Moore-Penrose identity A A+ A = A.
	{
		glm::mat3 const m(
			1.f, 2.f, 3.f,
			2.f, 4.f, 6.f,
			0.f, 1.f, 1.f);
		Error += std::isinf(glm::matrixCondition(m)) ? 0 : 1;

		glm::mat3 const pinv = glm::pseudoinverse(m);
		Error += allFinite(pinv) ? 0 : 1;
		Error += maxAbsDiff(m * pinv * m, m) < 1e-4f ? 0 : 1;
	}

	// Determinant underflow: the cofactor inverse is non-finite, the SVD
	// pseudoinverse stays finite and keeps the correct 1e20 magnitude.
	{
		glm::mat3 const m(
			1e-20f, 0.f, 0.f,
			0.f, 1e-20f, 0.f,
			0.f, 0.f, 1e-20f);
		glm::mat3 const pinv = glm::pseudoinverse(m);
		Error += allFinite(pinv) ? 0 : 1;
		Error += !allFinite(glm::inverse(m)) ? 0 : 1;
		Error += std::abs(pinv[0][0] - 1e20f) / 1e20f < 1e-4f ? 0 : 1;
	}

	// Rank deficiency is detected for every supported square size.
	{
		glm::mat2 m2(1.f, 2.f, 2.f, 4.f);
		Error += std::isinf(glm::matrixCondition(m2)) && allFinite(glm::pseudoinverse(m2)) ? 0 : 1;

		glm::mat4 m4(1.f);
		m4[2] = m4[1];
		Error += std::isinf(glm::matrixCondition(m4)) && allFinite(glm::pseudoinverse(m4)) ? 0 : 1;
	}

	return Error;
}

static int test_stable_inverse()
{
	int Error = 0;

	// A well-conditioned matrix takes the regular inverse branch.
	{
		glm::mat3 const m(
			2.f, 0.f, 0.f,
			0.f, 2.f, 0.f,
			0.f, 0.f, 2.f);
		Error += maxAbsDiff(glm::stableInverse(m), glm::inverse(m)) < 1e-5f ? 0 : 1;
	}

	// A near-singular matrix falls back to a finite pseudoinverse.
	{
		float const c = 0.6f;
		float const s = 0.8f;
		glm::mat3 const u(
			c, -s, 0.f,
			s, c, 0.f,
			0.f, 0.f, 1.f);
		glm::mat3 const v(
			1.f, 0.f, 0.f,
			0.f, c, -s,
			0.f, s, c);
		glm::mat3 const sigma(
			1.f, 0.f, 0.f,
			0.f, 1.f, 0.f,
			0.f, 0.f, 1e-20f);
		glm::mat3 const m = u * sigma * glm::transpose(v);

		Error += allFinite(glm::stableInverse(m)) ? 0 : 1;
	}

	return Error;
}

static int test_affine()
{
	int Error = 0;

	{
		glm::mat3 const M(
			2.f, 0.f, 0.f,
			0.f, 2.f, 0.f,
			0.f, 0.f, 1.f);
		glm::mat3 const A = glm::affineInverse(M);
		glm::mat3 const I = glm::inverse(M);
		glm::mat3 const R = glm::affineInverse(A);

		for(glm::length_t i = 0; i < A.length(); ++i)
		{
			Error += glm::all(glm::epsilonEqual(M[i], R[i], 0.01f)) ? 0 : 1;
			Error += glm::all(glm::epsilonEqual(A[i], I[i], 0.01f)) ? 0 : 1;
		}
	}

	{
		glm::mat4 const M(
			2.f, 0.f, 0.f, 0.f,
			0.f, 2.f, 0.f, 0.f,
			0.f, 0.f, 2.f, 0.f,
			0.f, 0.f, 0.f, 1.f);
		glm::mat4 const A = glm::affineInverse(M);
		glm::mat4 const I = glm::inverse(M);
		glm::mat4 const R = glm::affineInverse(A);

		for(glm::length_t i = 0; i < A.length(); ++i)
		{
			Error += glm::all(glm::epsilonEqual(M[i], R[i], 0.01f)) ? 0 : 1;
			Error += glm::all(glm::epsilonEqual(A[i], I[i], 0.01f)) ? 0 : 1;
		}
	}

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_affine();
	Error += test_singular_values();
	Error += test_pseudoinverse();
	Error += test_stable_inverse();

	return Error;
}
