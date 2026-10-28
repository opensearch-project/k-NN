/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

// SQ 4-bit (Lucene PACKED_NIBBLE: 4-bit query x 4-bit data) similarity function shared by all platforms.
//
// Must be included AFTER the including translation unit has defined `readDataCorrections`
// (see the SQ 1-bit section of each platform file). Each platform supplies a `Kernel` type that
// computes the raw integer dot product; everything else (batching, correction terms, score transform)
// lives here so the score math is identical to Lucene104ScalarQuantizedVectorScorer on every platform.
//
// Layouts:
//   Query (unpacked)  : `2 * packedBytes` bytes, one 4-bit value [0, 15] per byte. Dimension is
//                       discretized (rounded up to an even number) and zero padded by Lucene.
//   Document (packed) : `packedBytes` bytes followed by the correction factors. Byte i holds
//                       element i in its high nibble and element (packedBytes + i) in its low nibble.
//
// The raw score is Lucene's VectorUtil.int4DotProductSinglePacked:
//   Result = sum_i (doc[i] >> 4) * query[i] + (doc[i] & 0x0F) * query[packedBytes + i]
//

#ifndef KNN_SQ4_SIMILARITY_FUNCTION_CPP
#define KNN_SQ4_SIMILARITY_FUNCTION_CPP

// Packed byte length of a PACKED_NIBBLE document for the given dimension.
// Lucene discretizes the dimension to an even number, then packs two dimensions per byte.
static FORCE_INLINE int32_t sq4PackedBytes(const int32_t dimension) {
    return (dimension + 1) / 2;
}

// Scalar int4DotProductSinglePacked. Used by the default kernel and as the reference for SIMD tails.
static FORCE_INLINE int32_t sq4DotProductScalar(const uint8_t* query, const uint8_t* doc, const int32_t packedBytes) {
    const uint8_t* queryLow = query + packedBytes;
    int32_t total = 0;
    for (int32_t i = 0 ; i < packedBytes ; ++i) {
        total += static_cast<int32_t>(doc[i] >> 4) * query[i];
        total += static_cast<int32_t>(doc[i] & 0x0F) * queryLow[i];
    }
    return total;
}

// Kernel requirements:
//   template <int BATCH_SIZE>
//   static void dotProductBatch(const uint8_t* query, uint8_t** docs, int32_t packedBytes, int32_t* results);
template <SQMetricMode Mode, typename Kernel>
struct SQ4SimilarityFunction final : SimilarityFunction {
    struct QueryCorrections {
        float ay;
        float ly;
        float queryAdditional;
        float y1;
        float centroidDp;
    };

    static FORCE_INLINE QueryCorrections readQueryCorrections(SimdVectorSearchContext* srchContext) {
        // Written by saveSQSearchContext: [lower | upper | additional | componentSum(int) | centroidDp]
        const auto* queryCorrectionPtr = reinterpret_cast<const float*>(srchContext->tmpBuffer.data());
        QueryCorrections corrections;
        corrections.ay = queryCorrectionPtr[0];
        corrections.ly = (queryCorrectionPtr[1] - queryCorrectionPtr[0]) * FOUR_BIT_SCALE;
        corrections.queryAdditional = queryCorrectionPtr[2];
        int32_t y1Raw; std::memcpy(&y1Raw, &queryCorrectionPtr[3], sizeof(int32_t));
        corrections.y1 = static_cast<float>(y1Raw);
        corrections.centroidDp = queryCorrectionPtr[4];
        return corrections;
    }

    // Applies the OSQ correction terms to a raw integer dot product. Returns the score before the
    // Lucene score transform, e.g. inner product or (clamped) squared L2 distance.
    static FORCE_INLINE float correctedScore(const QueryCorrections& q,
                                             const uint8_t* doc,
                                             const int32_t packedBytes,
                                             const int32_t dim,
                                             const int32_t dotProduct) {
        float ax, lx, additional, x1;
        readDataCorrections(doc + packedBytes, ax, lx, additional, x1);
        // Unlike 1-bit documents, 4-bit documents are scaled by 1 / (2^4 - 1) as well.
        lx *= FOUR_BIT_SCALE;

        const float score = ax * q.ay * dim
                          + q.ay * lx * x1
                          + ax * q.ly * q.y1
                          + lx * q.ly * static_cast<float>(dotProduct);

        if constexpr (Mode == SQMetricMode::MAX_IP || Mode == SQMetricMode::COSINE) {
            return score + q.queryAdditional + additional - q.centroidDp;
        } else {
            return std::max(0.0F, q.queryAdditional + additional - 2 * score);
        }
    }

    HOT_SPOT void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                            int32_t* internalVectorIds,
                                            float* scores,
                                            const int32_t numVectors) final {
        const auto* queryPtr = reinterpret_cast<const uint8_t*>(srchContext->queryVectorSimdAligned);
        const int32_t dim = srchContext->dimension;
        const int32_t packedBytes = sq4PackedBytes(dim);
        const QueryCorrections queryCorrections = readQueryCorrections(srchContext);

        int32_t processedCount = 0;
        constexpr int32_t vecBlock = 8;
        constexpr int32_t vecHalfBlock = 4;
        uint8_t* vectors[vecBlock];
        int32_t dotProducts[vecBlock];

        // Batch size 8
        for ( ; (processedCount + vecBlock) <= numVectors ; processedCount += vecBlock) {
            srchContext->getVectorPointersInBulk(vectors, &internalVectorIds[processedCount], vecBlock);
            Kernel::template dotProductBatch<vecBlock>(queryPtr, vectors, packedBytes, dotProducts);
            for (int32_t i = 0 ; i < vecBlock ; ++i) {
                scores[processedCount + i] = correctedScore(queryCorrections, vectors[i], packedBytes, dim, dotProducts[i]);
            }
        }

        // Batch size 4
        for ( ; (processedCount + vecHalfBlock) <= numVectors ; processedCount += vecHalfBlock) {
            srchContext->getVectorPointersInBulk(vectors, &internalVectorIds[processedCount], vecHalfBlock);
            Kernel::template dotProductBatch<vecHalfBlock>(queryPtr, vectors, packedBytes, dotProducts);
            for (int32_t i = 0 ; i < vecHalfBlock ; ++i) {
                scores[processedCount + i] = correctedScore(queryCorrections, vectors[i], packedBytes, dim, dotProducts[i]);
            }
        }

        // Tail: remaining vectors
        for ( ; processedCount < numVectors ; ++processedCount) {
            vectors[0] = srchContext->getVectorPointer(internalVectorIds[processedCount]);
            Kernel::template dotProductBatch<1>(queryPtr, vectors, packedBytes, dotProducts);
            scores[processedCount] = correctedScore(queryCorrections, vectors[0], packedBytes, dim, dotProducts[0]);
        }

        if constexpr (Mode == SQMetricMode::MAX_IP) {
            FaissScoreToLuceneScoreTransform::ipToMaxIpTransformBulk(scores, numVectors);
        } else if constexpr (Mode == SQMetricMode::COSINE) {
            FaissScoreToLuceneScoreTransform::cosineTransformBulk(scores, numVectors);
        } else {
            FaissScoreToLuceneScoreTransform::l2TransformBulk(scores, numVectors);
        }
    }

    float calculateSimilarity(SimdVectorSearchContext* srchContext, const int32_t internalVectorId) final {
        const auto* queryPtr = reinterpret_cast<const uint8_t*>(srchContext->queryVectorSimdAligned);
        const int32_t dim = srchContext->dimension;
        const int32_t packedBytes = sq4PackedBytes(dim);
        const QueryCorrections queryCorrections = readQueryCorrections(srchContext);

        uint8_t* vector = srchContext->getVectorPointer(internalVectorId);
        int32_t dotProduct;
        Kernel::template dotProductBatch<1>(queryPtr, &vector, packedBytes, &dotProduct);
        const float score = correctedScore(queryCorrections, vector, packedBytes, dim, dotProduct);

        if constexpr (Mode == SQMetricMode::MAX_IP) {
            return FaissScoreToLuceneScoreTransform::ipToMaxIpTransform(score);
        } else if constexpr (Mode == SQMetricMode::COSINE) {
            return FaissScoreToLuceneScoreTransform::cosineTransform(score);
        } else {
            return FaissScoreToLuceneScoreTransform::l2Transform(score);
        }
    }
};

// Portable kernel relying on compiler auto-vectorization.
struct DefaultSQ4Kernel final {
    template <int BATCH_SIZE>
    static FORCE_INLINE void dotProductBatch(const uint8_t* query, uint8_t** docs, const int32_t packedBytes, int32_t* results) {
        for (int32_t b = 0 ; b < BATCH_SIZE ; ++b) {
            results[b] = sq4DotProductScalar(query, docs[b], packedBytes);
        }
    }
};

#endif  // KNN_SQ4_SIMILARITY_FUNCTION_CPP
