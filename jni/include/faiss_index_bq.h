/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

#ifndef KNNPLUGIN_JNI_FAISSINDEXBQ_H
#define KNNPLUGIN_JNI_FAISSINDEXBQ_H

#include "faiss/IndexFlatCodes.h"
#include "faiss/Index.h"
#include "faiss/impl/DistanceComputer.h"
#include "faiss/utils/hamming_distance/hamdis-inl.h"
#include "faiss/impl/HNSW.h"
#include <vector>
#include <iostream>
#include <cassert>

// Define macro for more readable table lookups. 
// Converts a two-dimensional lookup (batch index, code value) to a one-dimensional array index
#define ADC_FLAT_LOOKUP_BATCH(table, i, j) ((table)[(i) * NUM_POSSIBILITIES_PER_BATCH + (j)])
namespace knn_jni {
    namespace faiss_wrapper {

        /**
         * ADCFlatCodesDistanceComputer1Bit provides a distance computer to compute distances between a full precision query vector and binary-quantized document vectors.
         * Builds a batched lookup table for each query vector. Distances for all possible byte values (256 possibilities)
         * are precomputed for each 8-bit chunk of the vector. Then the per-chunk distances to each document chunks are loaded
         * during search time.
         *
         */
        struct ADCFlatCodesDistanceComputer1Bit : faiss::FlatCodesDistanceComputer {
            static constexpr int BATCH_SIZE = 8;  // Process 8 bits at a time
            static constexpr int NUM_POSSIBILITIES_PER_BATCH = 1 << BATCH_SIZE;  // 256 possible values for an 8-bit chunk
            const float* query;                 // Pointer to the query vector
            int dimension;                      // Dimensionality of the vectors
            size_t code_size;                   // Size of each code in bytes
            faiss::MetricType metric_type;      // Distance metric type (L2 or inner product)
            std::vector<float> lookup_table;    // Precomputed distance contributions for all possible byte values
            std::vector<float> coord_scores;    // Per-dimension distance contributions
            float correction_amount;            // Correction factor for L2 distance calculation

            ADCFlatCodesDistanceComputer1Bit(const uint8_t* codes, size_t code_size, int d,
                faiss::MetricType metric_type = faiss::METRIC_L2) 
                : FlatCodesDistanceComputer(codes, code_size),
                 query(nullptr),
                 dimension(d),
                 metric_type(metric_type),
                 lookup_table(),
                 coord_scores(),
                 correction_amount(0.0f)
                 {}

            /**
             * Computes the distance between the query vector and a binary-quantized code
             * by using the precomputed lookup table.
             */
            virtual float distance_to_code(const uint8_t* code) override final {
                return distance_to_code_batched_unrolled(code);
            };

            virtual void distances_batch_4(
                const faiss::HNSW::storage_idx_t idx0,
                const faiss::HNSW::storage_idx_t idx1,
                const faiss::HNSW::storage_idx_t idx2,
                const faiss::HNSW::storage_idx_t idx3,
                float& result_dis0,
                float& result_dis1,
                float& result_dis2,
                float& result_dis3) final {
                const auto code0 = this->codes + idx0 * code_size;
                const auto code1 = this->codes + idx1 * code_size;
                const auto code2 = this->codes + idx2 * code_size;
                const auto code3 = this->codes + idx3 * code_size;

                float dist0 = 0.0f;
                float dist1 = 0.0f;
                float dist2 = 0.0f;
                float dist3 = 0.0f;
                for (int i = 0 ; i < dimension / 8; ++i) {
                    dist0 += ADC_FLAT_LOOKUP_BATCH(this->lookup_table, i, *(code0 + i));
                    dist1 += ADC_FLAT_LOOKUP_BATCH(this->lookup_table, i, *(code1 + i));
                    dist2 += ADC_FLAT_LOOKUP_BATCH(this->lookup_table, i, *(code2 + i));
                    dist3 += ADC_FLAT_LOOKUP_BATCH(this->lookup_table, i, *(code3 + i));
                }

                result_dis0 = dist0;
                result_dis1 = dist1;
                result_dis2 = dist2;
                result_dis3 = dist3;
            }

            /**
             * Fast distance computation using loop unrolling and batched lookups
             *
             * This method:
             * 1. Processes 4 bytes at a time for better instruction pipelining
             * 2. For each byte, looks up precomputed distance contribution from the table
             * 3. Accumulates partial distances and applies the distance correction
             *
             * The compiler hints direct aggressive optimization of floating point operations
             */
            float distance_to_code_batched_unrolled(const uint8_t * code) {
                float dist0 = 0.0f, dist1 = 0.0f, dist2 = 0.0f, dist3 = 0.0f;
                int i = 0;
                const int limit = dimension / 8;
                // applies autovectorization and loop unrolling depending on compiler. Used in faiss distances_simd.cpp
                FAISS_PRAGMA_IMPRECISE_LOOP
                for (; i + 3 < limit; i += 4) {
                    dist0 += ADC_FLAT_LOOKUP_BATCH(this->lookup_table, i, code[i]);
                    dist1 += ADC_FLAT_LOOKUP_BATCH(this->lookup_table, i+1, code[i+1]);
                    dist2 += ADC_FLAT_LOOKUP_BATCH(this->lookup_table, i+2, code[i+2]);
                    dist3 += ADC_FLAT_LOOKUP_BATCH(this->lookup_table, i+3, code[i+3]);
                }

                // Handle any remaining iterations
                for (; i < limit; ++i) {
                    dist0 += ADC_FLAT_LOOKUP_BATCH(this->lookup_table, i, code[i]);
                }

                // Combine all accumulators and apply correction
                return dist0 + dist1 + dist2 + dist3 + correction_amount;
            }

            /**
             * Compute per-dimension distance contributions based on the query vector
             * and selected distance metric
             */
            void compute_cord_scores() {
                this->coord_scores = std::vector<float>(this->dimension, 0.0f);
                if (this->metric_type == faiss::METRIC_L2) {
                    compute_cord_scores_l2();
                } else if (this->metric_type == faiss::METRIC_INNER_PRODUCT) {
                    compute_cord_scores_inner_product();
                } else {
                    throw std::runtime_error(
                        ("ADC distance computer called with unsupported metric: " + std::to_string(this->metric_type))
                    );
                }
            }

            /**
             * Compute per-dimension contributions for L2 distance
             *
             * For L2 distance with 1-bit quantization, each bit contributes:
             * - If bit=0: query[i] ** 2
             * - If bit=1: (1-query[i]) ** 2
             *
             * This simplifies to: bit_contribution = (1 - 2*query[i])*bit + query[i] ** 2
             * We store (1 - 2*query[i]) as the coefficient and accumulate query[i] ** 2 as correction
             */
            void compute_cord_scores_l2() {
                assert(query != nullptr);
                FAISS_PRAGMA_IMPRECISE_LOOP
                for (int i = 0 ; i < this->dimension; ++i) {
                    float x = query[i];
                    this->coord_scores[i] = 1 - 2 * x;
                    correction_amount += x * x;
                }
            }

            /**
             * Compute per-dimension contributions for inner product distance
             *
             * For inner product with 1-bit quantization, each bit directly
             * contributes the query value when the bit is set
             */
            void compute_cord_scores_inner_product() {
                std::copy(query, query + dimension, coord_scores.begin());
            }

            /**
             * Set query vector and precompute distance table and lookup table.
             */
            virtual void set_query(const float* x) override {
                this->correction_amount = 0.0f;
                this->query = x;
                compute_cord_scores();
                create_batched_lookup_table();
            };

            void create_batched_lookup_table() {
                // number of batches per vector
                const unsigned int num_batches =this->dimension/BATCH_SIZE;

                lookup_table = std::vector<float>(num_batches*NUM_POSSIBILITIES_PER_BATCH, 0.0f);
                // each batch stores all of the 2^8 possible values an 8-bit chunk of the query vector can take at that position.
                for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
                    for (int i = 0 ; i < BATCH_SIZE; ++i) {
                        const unsigned int bit_masked = 1 << i;
                        // for instance for batch_idx 1, this looks starting at position 7 and then scans from right to left.
                        // The scanning pattern must conform to the bit packing strategy in BitPacker.java.
                        const float bit_value = this->coord_scores[ batch_idx * BATCH_SIZE + (7 - i)];

                        for (unsigned int suffix = 0; suffix < bit_masked; ++suffix) {
                            // DP to build batch values one-by-one using previously computed values.
                            // for each batch_idx: batch[bit_masked | suffix] = batch[suffix] + bit_value;
                            ADC_FLAT_LOOKUP_BATCH(this->lookup_table, batch_idx, bit_masked | suffix) =
                                ADC_FLAT_LOOKUP_BATCH(this->lookup_table, batch_idx, suffix) + bit_value;
                        }
                    }
                }
            }
            /**
             * ADCFlatCodesDistanceComputer1Bit::symmetric_dis is not implemented.
             * The FlatCodesDistanceComputer::symmetric_dis function is used for index building. However the k-NN plugin
             * only loads an altered index with this distance computer for search.
             */
            virtual float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override {
                throw std::runtime_error("ADC computer is only implemented for search time, not indexing.");
            };
        };

        /**
         * FaissIndexBQ stores vectors as binary-quantized codes and uses ADCFlatCodesDistanceComputer1Bit
         * to calculate the distance from full-precision query vectors to quantized codes.
        */
        struct FaissIndexBQ : faiss::IndexFlatCodes {
            std::vector<uint8_t> codes_vector; // Storage for binary-quantized vector codes, owned by an instance of FaissIndexBQ.

            /**
             * Populate an empty IndexFlatCodes with the metric and dimensionality, and store the binary codes.
             *
             * @param d Dimensionality of original vectors
             * @param codes_vector Binary codes for all indexed vectors
             * @param metric Distance metric type (L2 or inner product)
             */
            FaissIndexBQ(faiss::idx_t d, std::vector<uint8_t> codes_vector, faiss::MetricType metric=faiss::METRIC_L2)
            : IndexFlatCodes(d/8, d, metric), codes_vector(std::move(codes_vector)) {}

            /**
             * Initialize the index and sync total vector count with parent indexes
             */
            void init(faiss::Index * parent, faiss::Index * grand_parent) {
                this->ntotal = codes_vector.size() / (this->d / 8);
                parent->ntotal = this->ntotal;   
                grand_parent->ntotal = this->ntotal;
            }

            /** Return overridden FlatCodesDistanceComputer with ADC distance_to_code method */
            faiss::FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override {
                // dimension must be a multiple of 8.
                if (this->d % 8 != 0) throw std::runtime_error("ADC distance computer only supports d divisible by 8");
                                return new knn_jni::faiss_wrapper::ADCFlatCodesDistanceComputer1Bit(
                    this->codes_vector.data(), 
                    this->d/8, 
                    this->d,
                    this->metric_type
                );
            };
        };
    }
}
#endif //KNNPLUGIN_JNI_FAISSINDEXBQ_H
