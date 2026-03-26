#ifndef KNNPLUGIN_JNI_FAISS_BBQ_HNSW_H
#define KNNPLUGIN_JNI_FAISS_BBQ_HNSW_H

#include "faiss/IndexBinaryHNSW.h"
#include "faiss_sq_flat.h"

namespace knn_jni {

    struct FaissSQHnsw final : faiss::IndexBinaryHNSW {
        FaissSQFlat* faiss_sq_flat;

        FaissSQHnsw(int32_t _m, FaissSQFlat* _faiss_sq_flat)
          : faiss::IndexBinaryHNSW(_faiss_sq_flat, _m),
            faiss_sq_flat(_faiss_sq_flat) {
            // This has the ownership of FaissSQFlat, setting this true to make it free'd when this class is freed
            own_fields = true;
            // Set metric type that was given, not just blindly use default space type.
            metric_type = _faiss_sq_flat->metric_type;
            // Since HNSW is setting query pointer as `base + query_index * index_hnsw.code_size`
            // we should update code_size in here.
            code_size = _faiss_sq_flat->code_size;
        }

        faiss::DistanceComputer* get_distance_computer() const final {
            return faiss_sq_flat->get_distance_computer();
        }
    };

}

#endif // KNNPLUGIN_JNI_FAISS_BBQ_HNSW_H
