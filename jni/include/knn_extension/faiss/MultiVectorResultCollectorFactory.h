/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <faiss/impl/ResultCollectorFactory.h>
#include "knn_extension/faiss/utils/BitSet.h"

namespace os_faiss {
/**
 * Create MultiVectorResultCollector for single query request
 *
 * Creating new collector is required because MultiVectorResultCollector has instance variables
 * which should be isolated for each query.
 */
struct MultiVectorResultCollectorFactory:faiss::ResultCollectorFactory {
    BitSet* parent_bit_set;

    MultiVectorResultCollectorFactory(BitSet* parent_bit_set);
    faiss::ResultCollector* new_collector() override;
    void delete_collector(faiss::ResultCollector* resultCollector) override;
};

}  // namespace os_faiss
