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

namespace knn_jni {
    namespace faiss_wrapper {
        struct CustomerFlatCodesDistanceComputer : faiss::FlatCodesDistanceComputer {
            const float* query;
            int dimension;
            size_t code_size;

            CustomerFlatCodesDistanceComputer(const uint8_t* codes, size_t code_size, int d) {
                this->codes = codes;
                this->code_size = code_size;
                this->dimension = d;
            }

            virtual float distance_to_code(const uint8_t* code) override {
                // Compute the dot product between the 2
                // TODO: How can we do this better for 2-bit and 4-bit
                // I think we would want to just shift the multiplier of 1. i.e.
                // -1 << 1 *query[i]
                // -1 << 2 *query[i]
                // -1 << 3 *query[i]
                float score = 0.0f;
                for (int i = 0; i < this->dimension; i++) {
                    score += (code[(i / sizeof(uint8_t))] & (1 << (i % sizeof(uint8_t)))) == 0 ? 0 : -1*query[i];
                }
                return score;
            }

            virtual void set_query(const float* x) override {
                this->query = x;
            };

            virtual float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override {
                // Just return hamming distance for now...
                return faiss::hamming<1, float>(&this->codes[i], &this->codes[j]);
            };
        };

        struct FaissIndexBQ : faiss::IndexFlatCodes {

            FaissIndexBQ(faiss::idx_t d, std::vector<uint8_t> codes) {
                this->d = d;
                this->codes = codes;
                this->code_size = 1;
            }

            void init(faiss::Index * parent, faiss::Index * grand_parent) {
                this->ntotal = this->codes.size() / (this->d / 8);
                parent->ntotal = this->ntotal;
                grand_parent->ntotal = this->ntotal;
            }

            /** a FlatCodesDistanceComputer offers a distance_to_code method */
            faiss::FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override {
                return new knn_jni::faiss_wrapper::CustomerFlatCodesDistanceComputer((const uint8_t*) (this->codes.data()), 1, this->d);
            };

            virtual void merge_from(faiss::Index& otherIndex, faiss::idx_t add_id = 0) override {};

            virtual void search(
                    faiss::idx_t n,
                    const float* x,
                    faiss::idx_t k,
                    float* distances,
                    faiss::idx_t* labels,
                    const faiss::SearchParameters* params = nullptr) const override {};
        };
    }
}
#endif //KNNPLUGIN_JNI_FAISSINDEXBQ_H
