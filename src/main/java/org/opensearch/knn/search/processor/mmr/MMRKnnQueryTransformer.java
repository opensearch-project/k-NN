/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor.mmr;

import lombok.NoArgsConstructor;
import org.opensearch.core.action.ActionListener;
import org.opensearch.knn.index.query.KNNQueryBuilder;

import static org.opensearch.knn.search.processor.mmr.MMRUtil.resolveKnnVectorFieldInfo;

/**
 * A transformer to transform the knn query for MMR
 */
@NoArgsConstructor
public class MMRKnnQueryTransformer implements MMRQueryTransformer<KNNQueryBuilder> {

    @Override
    public void transform(KNNQueryBuilder queryBuilder, ActionListener<Void> listener, MMRTransformContext mmrTransformContext) {
        try {
            if (queryBuilder.getMaxDistance() == null && queryBuilder.getMinScore() == null) {
                queryBuilder.setK(mmrTransformContext.getCandidates());
            }

            if (mmrTransformContext.isVectorFieldInfoResolved()) {
                listener.onResponse(null);
                return;
            }

            MMRRerankContext mmrRerankContext = mmrTransformContext.getMmrRerankContext();
            String knnVectorFieldPath = queryBuilder.fieldName();
            if (knnVectorFieldPath == null) {
                throw new IllegalArgumentException(
                    "Failed to transform the knn query for MMR. Field name of the knn query should not be null."
                );
            }
            mmrRerankContext.setVectorFieldPath(knnVectorFieldPath);

            resolveKnnVectorFieldInfo(
                knnVectorFieldPath,
                mmrTransformContext.getUserProvidedSpaceType(),
                mmrTransformContext.getUserProvidedVectorDataType(),
                mmrTransformContext.getLocalIndexMetadataList(),
                mmrTransformContext.getClient(),
                ActionListener.wrap(vectorFieldInfo -> {
                    mmrRerankContext.setVectorDataType(vectorFieldInfo.getVectorDataType());
                    mmrRerankContext.setSpaceType(vectorFieldInfo.getSpaceType());
                    listener.onResponse(null);
                }, listener::onFailure)
            );
        } catch (Exception e) {
            listener.onFailure(e);
        }
    }

    @Override
    public String getQueryName() {
        return KNNQueryBuilder.NAME;
    }
}
