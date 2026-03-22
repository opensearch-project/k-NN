#ifndef KNNPLUGIN_JNI_FAISS_BBQ_HNSW_H
#define KNNPLUGIN_JNI_FAISS_BBQ_HNSW_H

#include "faiss/IndexBinaryHNSW.h"
#include "faiss_bbq_flat.h"

namespace knn_jni {

    struct FaissBBQHnsw final : faiss::IndexBinaryHNSW {
        FaissBBQFlat* faiss_bbq_flat;

        FaissBBQHnsw(int32_t _m, FaissBBQFlat* _faiss_bbq_flat)
          : faiss::IndexBinaryHNSW(_faiss_bbq_flat, _m),
            faiss_bbq_flat(_faiss_bbq_flat) {
            // This has the ownership of FaissBBQFlat, setting this true to make it free'd when this class is freed
            own_fields = true;
            // Set metric type that was given, not just blindly use default space type.
            metric_type = _faiss_bbq_flat->metric_type;
            // Since HNSW is setting query pointer as `base + query_index * index_hnsw.code_size`
            // we should update code_size in here.
            code_size = _faiss_bbq_flat->code_size;
        }

        faiss::DistanceComputer* get_distance_computer() const final {
            return faiss_bbq_flat->get_distance_computer();
        }
    };

}

#endif // KNNPLUGIN_JNI_FAISS_BBQ_HNSW_H
