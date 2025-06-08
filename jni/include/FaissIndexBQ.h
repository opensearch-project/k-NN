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
#include <vector>
#include <iostream>
#include <cassert>

namespace knn_jni {
    namespace faiss_wrapper {
        struct ADCFlatCodesDistanceComputer1Bit : faiss::FlatCodesDistanceComputer {
            static constexpr int BATCH_SIZE = 8;
            static constexpr int NUM_POSSIBILITIES_PER_BATCH = 1 << BATCH_SIZE;
            const uint8_t* codes;
            const float* query;
            int dimension;
            size_t code_size;
            faiss::MetricType metric_type;
            std::vector<std::vector<float>> lookup_table;
            std::vector<float> coord_scores; // scores for each dimension
            float correction_amount; // used in batched distances

            ADCFlatCodesDistanceComputer1Bit(const uint8_t* codes, size_t code_size, int d,
                faiss::MetricType metric_type = faiss::METRIC_L2) 
                : FlatCodesDistanceComputer(codes, code_size), dimension(d), query(nullptr), metric_type(metric_type) {
                correction_amount = 0.0f;
            }

            float distance_to_code_batched(const uint8_t * code) {
                float dist = 0.0f;
                for (int i = 0 ; i < dimension / 8; ++i) {
                    const unsigned char code_batch = code[i];
                    dist += this->lookup_table[i][code_batch];
                }

                return dist + correction_amount; 
            }

            virtual float distance_to_code(const uint8_t* code) override {
                return distance_to_code_batched(code);
            };

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

            void compute_cord_scores_l2() {
                assert(query != nullptr);
                for (int i = 0 ; i < this->dimension; ++i) {
                    float x = query[i];
                    this->coord_scores[i] = 1 - 2 * x;
                    correction_amount += x * x;
                }
            }

            void compute_cord_scores_inner_product() {
                for (int i = 0 ; i < this->dimension; ++i) {
                    this->coord_scores[i] = query[i];
                }
            }

            virtual void set_query(const float* x) override {
                this->correction_amount = 0.0f;
                this->query = x;
                compute_cord_scores();
                create_batched_lookup_table();
            };

            // compute all possible distance combinations for the batch at batch_idx against our query vector
            void compute_per_batch_lookup(int batch_idx, std::vector<float> & batch) {
                // batch has 256 dimension, and it only looks at one 8-bit/1 byte chunk of the query vector.
                for (int i = 0 ; i < BATCH_SIZE; ++i) {
                    const unsigned int bit_masked = 1 << i;
                    // for instance for batch_idx 1, this looks starting at position 7 and then scans from right to left.
                    // The scanning pattern must conform to the bit packing strategy in BitPacker.java.
                    const float bit_value = this->coord_scores[ batch_idx * BATCH_SIZE + (7 - i)];

                    for (unsigned int suffix = 0; suffix < bit_masked; ++suffix) {
                        // DP to build batch values one-by-one using previously computed values.
                        batch[bit_masked | suffix] = batch[suffix] + bit_value;
                    }
                }
            }

            void create_batched_lookup_table() {
                // number of batches per vector
                const unsigned int num_batches =this->dimension/BATCH_SIZE;
                this->lookup_table = std::vector<std::vector<float>>(num_batches,
                    std::vector<float>(NUM_POSSIBILITIES_PER_BATCH, 0.0f)
                );

                for (int i = 0 ; i < num_batches; ++i) {
                    compute_per_batch_lookup(i, this->lookup_table[i]);
                }
            }

            // FlatCodesDistanceComputer::symmetric_dis is used in faiss only to calculate distance when building graph.
            virtual float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override {
                throw std::runtime_error("ADC computer is only implemented for search time, not indexing.");
            };
        };

        struct FaissIndexBQ : faiss::IndexFlatCodes {
            std::vector<uint8_t> * codes_ptr;
            
            FaissIndexBQ(faiss::idx_t d, std::vector<uint8_t> * codes_ptr, faiss::MetricType metric=faiss::METRIC_L2)
            : IndexFlatCodes(d/8, d, metric){
                this->codes_ptr = codes_ptr;
                this->code_size = (d/8);
            }

            void init(faiss::Index * parent, faiss::Index * grand_parent) {
                this->ntotal = this->codes_ptr->size() / (this->d / 8);
                parent->ntotal = this->ntotal;   
                grand_parent->ntotal = this->ntotal;
            }

            /** Return overridden FlatCodesDistanceComputer with ADC distance_to_code method */
            faiss::FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override {
                // dimension must be a multiple of 8.
                if (this->d % 8 != 0) throw std::runtime_error("ADC distance computer only supports d divisible by 8");
                return new knn_jni::faiss_wrapper::ADCFlatCodesDistanceComputer1Bit(this->codes_ptr->data(), this->d/8, this->d,
            this->metric_type);
            };
        };
    }
}
#endif //KNNPLUGIN_JNI_FAISSINDEXBQ_H
