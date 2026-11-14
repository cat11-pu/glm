// Batch mat4 * vec4 transform: per-point reference vs vectorized batch kernel.
//
// The batch kernel keeps bit-exact parity with the per-point GLM reference by
// preserving GLM's exact multiply/add evaluation tree per component:
//   r = (m[0]*v.x + m[1]*v.y) + (m[2]*v.z + m[3]*v.w)
// Explicit _mm256_mul_ps/_mm256_add_ps intrinsics are never FMA-contracted;
// build with -ffp-contract=off so the scalar reference is not contracted either.

#define GLM_FORCE_INLINE
#include <glm/ext/matrix_float4x4.hpp>
#include <glm/ext/vector_float4.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

// Per-point reference: one glm::mat4 * glm::vec4 per point, GLM temporaries included.
static void transform_ref(glm::mat4 const& M, glm::vec4 const* I, glm::vec4* O, std::size_t n)
{
	for (std::size_t i = 0; i < n; ++i)
		O[i] = M * I[i];
}

// Batch kernel: matrix columns hoisted into registers once, two points per
// iteration, single 8-float load/store, no per-point temporary objects.
static void transform_batch(glm::mat4 const& M, glm::vec4 const* I, glm::vec4* O, std::size_t n)
{
#if defined(__AVX2__)
	__m256 const c0 = _mm256_broadcast_ps(reinterpret_cast<__m128 const*>(&M[0]));
	__m256 const c1 = _mm256_broadcast_ps(reinterpret_cast<__m128 const*>(&M[1]));
	__m256 const c2 = _mm256_broadcast_ps(reinterpret_cast<__m128 const*>(&M[2]));
	__m256 const c3 = _mm256_broadcast_ps(reinterpret_cast<__m128 const*>(&M[3]));

	std::size_t i = 0;
	for (; i + 2 <= n; i += 2)
	{
		__m256 const p = _mm256_loadu_ps(reinterpret_cast<float const*>(I + i)); // p0 | p1
		__m256 const x = _mm256_permute_ps(p, 0x00);
		__m256 const y = _mm256_permute_ps(p, 0x55);
		__m256 const z = _mm256_permute_ps(p, 0xAA);
		__m256 const w = _mm256_permute_ps(p, 0xFF);
		__m256 const a0 = _mm256_add_ps(_mm256_mul_ps(c0, x), _mm256_mul_ps(c1, y));
		__m256 const a1 = _mm256_add_ps(_mm256_mul_ps(c2, z), _mm256_mul_ps(c3, w));
		_mm256_storeu_ps(reinterpret_cast<float*>(O + i), _mm256_add_ps(a0, a1));
	}
	for (; i < n; ++i) // odd tail: identical GLM expression as the reference
		O[i] = M * I[i];
#else
	transform_ref(M, I, O, n);
#endif
}

// Bitwise comparison of every output component (uint32 bit patterns).
static std::size_t count_bit_mismatches(glm::vec4 const* A, glm::vec4 const* B, std::size_t n)
{
	std::size_t errors = 0;
	for (std::size_t i = 0; i < n; ++i)
	{
		for (int c = 0; c < 4; ++c)
		{
			std::uint32_t a, b;
			std::memcpy(&a, &A[i][c], sizeof(a));
			std::memcpy(&b, &B[i][c], sizeof(b));
			errors += (a == b) ? 0 : 1;
		}
	}
	return errors;
}

template <typename F>
static double time_best_us(F&& f, int reps)
{
	double best = 1e30;
	for (int r = 0; r < reps; ++r)
	{
		std::chrono::high_resolution_clock::time_point const t1 = std::chrono::high_resolution_clock::now();
		f();
		std::chrono::high_resolution_clock::time_point const t2 = std::chrono::high_resolution_clock::now();
		double const us = std::chrono::duration<double, std::micro>(t2 - t1).count();
		if (us < best)
			best = us;
	}
	return best;
}

static int run_case(std::size_t n)
{
	// Deterministic, non-trivial inputs (LCG -> [-8, 8)).
	std::vector<glm::vec4> I(n);
	std::uint32_t state = 0x9E3779B9u;
	for (std::size_t i = 0; i < n; ++i)
		for (int c = 0; c < 4; ++c)
		{
			state = state * 1664525u + 1013904223u;
			I[i][c] = (static_cast<float>(state >> 8) / 16777216.0f) * 16.0f - 8.0f;
		}

	glm::mat4 const M(
		0.36f, -0.48f, 0.80f, 0.0f,
		0.80f, 0.60f, 0.00f, 0.0f,
		-0.48f, 0.64f, 0.60f, 0.0f,
		1.25f, -2.50f, 3.75f, 1.0f);

	std::vector<glm::vec4> ref(n), bat(n);

	int const reps = static_cast<int>(20000000 / (n < 1 ? 1 : n)) + 5;
	double const t_ref = time_best_us([&] { transform_ref(M, I.data(), ref.data(), n); }, reps);
	double const t_bat = time_best_us([&] { transform_batch(M, I.data(), bat.data(), n); }, reps);

	std::size_t const mismatches = count_bit_mismatches(ref.data(), bat.data(), n);
	double const reduction = (t_ref - t_bat) * 100.0 / t_ref;

	std::printf("points=%6zu  ref=%9.2f us  batch=%9.2f us  reduction=%5.1f%%  bit-mismatches=%zu / %zu  %s\n",
		n, t_ref, t_bat, reduction, mismatches, n * 4, mismatches == 0 ? "BIT-IDENTICAL" : "MISMATCH");

	return mismatches == 0 ? 0 : 1;
}

int main()
{
#if defined(__AVX2__)
	std::printf("kernel: AVX2, 2 points/iteration, columns hoisted, -ffp-contract=off\n");
#else
	std::printf("kernel: scalar fallback (no AVX2 at compile time)\n");
#endif

	int error = 0;
	error += run_case(1024);
	error += run_case(10000);
	error += run_case(100000);
	return error;
}
