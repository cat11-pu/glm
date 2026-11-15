#include <glm/ext/vector_float3.hpp>
#include <glm/ext/quaternion_common.hpp>
#include <glm/ext/quaternion_float.hpp>
#include <glm/ext/quaternion_double.hpp>
#include <glm/ext/quaternion_relational.hpp>
#include <glm/ext/quaternion_trigonometric.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/ext/scalar_relational.hpp>
#include <algorithm>
#include <cmath>

static int test_conjugate()
{
	int Error = 0;

	glm::quat const A(glm::vec3(1, 0, 0), glm::vec3(0, 1, 0));
	glm::quat const C = glm::conjugate(A);
	Error += glm::any(glm::notEqual(A, C, glm::epsilon<float>())) ? 0 : 1;

	glm::quat const B = glm::conjugate(C);
	Error += glm::all(glm::equal(A, B, glm::epsilon<float>())) ? 0 : 1;

	return Error;
}

static int test_mix()
{
	int Error = 0;

	glm::quat const Q1(glm::vec3(1, 0, 0), glm::vec3(1, 0, 0));
	glm::quat const Q2(glm::vec3(1, 0, 0), glm::vec3(0, 1, 0));

	{
		glm::quat const Q3 = glm::mix(Q1, Q2, 0.5f);
		float const F3 = glm::degrees(glm::angle(Q3));
		Error += glm::equal(F3, 45.0f, 0.001f) ? 0 : 1;

		glm::quat const Q4 = glm::mix(Q2, Q1, 0.5f);
		float const F4 = glm::degrees(glm::angle(Q4));
		Error += glm::equal(F4, 45.0f, 0.001f) ? 0 : 1;
	}

	{
		glm::quat const Q3 = glm::slerp(Q1, Q2, 0.5f);
		float const F3 = glm::degrees(glm::angle(Q3));
		Error += glm::equal(F3, 45.0f, 0.001f) ? 0 : 1;

		glm::quat const Q4 = glm::slerp(Q2, Q1, 0.5f);
		float const F4 = glm::degrees(glm::angle(Q4));
		Error += glm::equal(F4, 45.0f, 0.001f) ? 0 : 1;
	}

	return Error;
}

static int test_slerp_robustness()
{
	int Error = 0;

	// Exact antipode at a = 0.5 must be a finite unit quaternion, not 0/0.
	{
		glm::dquat const Q1(1.0, 0.0, 0.0, 0.0);
		glm::dquat const Q2(-1.0, 0.0, 0.0, 0.0);
		glm::dquat const Q3 = glm::slerp(Q1, Q2, 0.5);

		bool finite = std::isfinite(Q3.w) && std::isfinite(Q3.x)
			&& std::isfinite(Q3.y) && std::isfinite(Q3.z);
		Error += finite ? 0 : 1;
		Error += glm::abs(glm::length(Q3) - 1.0) < 1e-12 ? 0 : 1;
	}

	// Near-antipode interpolation stays on the short, continuous great arc:
	// every intermediate orientation stays within the bounding end orientations.
	for(int step = 0; step <= 100; ++step)
	{
		double const halfTheta = glm::half_pi<double>() * step / 100.0;
		glm::dquat const Q1(1.0, 0.0, 0.0, 0.0);
		glm::dquat const Q2(std::cos(halfTheta), 0.0, std::sin(halfTheta), 0.0);

		for(int sample = 0; sample <= 10; ++sample)
		{
			double const a = sample / 10.0;
			glm::dquat const Q3 = glm::slerp(Q1, Q2, a);

			bool finite = std::isfinite(Q3.w) && std::isfinite(Q3.x)
				&& std::isfinite(Q3.y) && std::isfinite(Q3.z);
			Error += finite ? 0 : 1;
			Error += glm::abs(glm::length(Q3) - 1.0) < 1e-12 ? 0 : 1;

			// Angular error against the direct short-arc formula, well below 1e-4.
			double const cosTheta = std::min(1.0, std::abs(glm::dot(Q1, Q2)));
			double const angle = std::acos(cosTheta);
			glm::dquat ref;
			glm::dquat z = (glm::dot(Q1, Q2) < 0.0) ? -Q2 : Q2;
			if(angle < 1e-12)
				ref = glm::normalize(Q1 * (1.0 - a) + z * a);
			else
				ref = (std::sin((1.0 - a) * angle) * Q1 + std::sin(a * angle) * z)
					/ std::sin(angle);

			double d = std::abs(glm::dot(Q3, ref));
			d = std::min(1.0, d);
			double const angularError = 2.0 * std::acos(d);
			Error += angularError < 1e-4 ? 0 : 1;
		}
	}

	// Non-unit inputs still produce a unit quaternion from the normalized fallback.
	{
		glm::quat const Q1(1.0f, 0.0f, 0.0f, 0.0f);
		glm::quat const Q2(2.0f, 0.0f, 0.0f, 0.0f);
		glm::quat const Q3 = glm::slerp(Q1, Q2, 0.5f);
		Error += glm::abs(glm::length(Q3) - 1.0f) < 1e-5f ? 0 : 1;
	}

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_conjugate();
	Error += test_mix();
	Error += test_slerp_robustness();

	return Error;
}
