/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#include "MultiVectorResultCollectorFactory.h"
#include "MultiVectorResultCollector.h"

namespace os_faiss {

MultiVectorResultCollectorFactory::MultiVectorResultCollectorFactory(BitSet* parent_bit_set)
     : parent_bit_set(parent_bit_set) {}

// id_map is set in IndexIDMap.cpp of faiss library with custom patch
// https://github.com/opensearch-project/k-NN/blob/feature/multi-vector/jni/patches/faiss/0001-Custom-patch-to-support-multi-vector.patch#L109
faiss::ResultCollector* MultiVectorResultCollectorFactory::new_collector() {
    return new MultiVectorResultCollector(parent_bit_set, id_map);
}

void MultiVectorResultCollectorFactory::delete_collector(faiss::ResultCollector* resultCollector) {
    delete resultCollector;
}

}  // namespace os_faiss
