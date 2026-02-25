#include <cmath>
#include <cstdint>

// This class is responsible to convert Faiss distance value to Lucene similarity score.
struct FaissScoreToLuceneScoreTransform final {
    // Convert Faiss inner product value to Max Inner Product scheme whose range is in [0, +Inf)
    static float ipToMaxIpTransform(const float innerProductValue) noexcept {
        return innerProductValue < 0 ? 1 / (1 - innerProductValue) : (1 + innerProductValue);
    }

    // Convert Faiss inner product values to Max Inner Product scheme whose range is in [0, +Inf)
    // Initially, `scores` have Faiss inner product values, after this transform, it will have Max IP value.
    static void ipToMaxIpTransformBulk(float* scores, const int32_t numScores) noexcept {
        int32_t i = 0;
        for (; (i + 8) <= numScores ; i += 8, scores += 8) {
            scores[0] = scores[0] < 0 ? 1 / (1 - scores[0]) : (1 + scores[0]);
            scores[1] = scores[1] < 0 ? 1 / (1 - scores[1]) : (1 + scores[1]);
            scores[2] = scores[2] < 0 ? 1 / (1 - scores[2]) : (1 + scores[2]);
            scores[3] = scores[3] < 0 ? 1 / (1 - scores[3]) : (1 + scores[3]);
            scores[4] = scores[4] < 0 ? 1 / (1 - scores[4]) : (1 + scores[4]);
            scores[5] = scores[5] < 0 ? 1 / (1 - scores[5]) : (1 + scores[5]);
            scores[6] = scores[6] < 0 ? 1 / (1 - scores[6]) : (1 + scores[6]);
            scores[7] = scores[7] < 0 ? 1 / (1 - scores[7]) : (1 + scores[7]);
        }

        for (; (i + 4) <= numScores ; i += 4, scores += 4) {
            scores[0] = scores[0] < 0 ? 1 / (1 - scores[0]) : (1 + scores[0]);
            scores[1] = scores[1] < 0 ? 1 / (1 - scores[1]) : (1 + scores[1]);
            scores[2] = scores[2] < 0 ? 1 / (1 - scores[2]) : (1 + scores[2]);
            scores[3] = scores[3] < 0 ? 1 / (1 - scores[3]) : (1 + scores[3]);
        }

        while (i < numScores) {
            *scores = ipToMaxIpTransform(*scores);
            ++i;
            ++scores;
        }
    }

    // Transform Faiss L2 distance to be bounded (0, 1]
    static float l2Transform(float l2Distance) noexcept {
        return 1.0f / (1.0f + l2Distance);
    }

    // Transform Faiss L2 distance to be bounded (0, 1].
    // `scores` have Faiss L2 distance, and after the transform, it will have a value in (0, 1].
    static void l2TransformBulk(float* scores, const int32_t numScores) noexcept {
        int32_t i = 0;
        for (; (i + 8) <= numScores ; i += 8, scores += 8) {
            scores[0] = 1.0f / (1.0f + scores[0]);
            scores[1] = 1.0f / (1.0f + scores[1]);
            scores[2] = 1.0f / (1.0f + scores[2]);
            scores[3] = 1.0f / (1.0f + scores[3]);
            scores[4] = 1.0f / (1.0f + scores[4]);
            scores[5] = 1.0f / (1.0f + scores[5]);
            scores[6] = 1.0f / (1.0f + scores[6]);
            scores[7] = 1.0f / (1.0f + scores[7]);
        }

        for (; (i + 4) <= numScores ; i += 4, scores += 4) {
            scores[0] = 1.0f / (1.0f + scores[0]);
            scores[1] = 1.0f / (1.0f + scores[1]);
            scores[2] = 1.0f / (1.0f + scores[2]);
            scores[3] = 1.0f / (1.0f + scores[3]);
        }

        while (i < numScores) {
            *scores = l2Transform(*scores);
            ++i;
            ++scores;
        }
    }

private:
    FaissScoreToLuceneScoreTransform() = delete;
};
