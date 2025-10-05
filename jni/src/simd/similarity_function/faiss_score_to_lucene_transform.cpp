#include <cmath>
#include <cstdint>

struct FaissScoreToLuceneScoreTransform final {
    static void noTransformBulk(float* scores, const int32_t numScores) noexcept {
    }
    static float noTransform(float score) noexcept {
        return score;
    }

    static float ipToMaxIpTransform(const float score) noexcept {
        return score < 0 ? 1 / (1 - score) : (1 + score);
    }

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
            *scores = *scores < 0 ? 1 / (1 - *scores) : (1 + *scores);
            ++i;
            ++scores;
        }
    }

    static float l2Transform(float l2Distance) noexcept {
        return 1.0f / (1.0f + l2Distance);
    }

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
            *scores = 1.0f / (1.0f + *scores);
            ++i;
            ++scores;
        }
    }

private:
    FaissScoreToLuceneScoreTransform() = delete;
};
