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

package org.opensearch.knn.plugin.transport;

import org.opensearch.core.action.ActionListener;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNSingleNodeTestCase;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.ModelDao;

import java.io.IOException;
import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ExecutionException;

import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.plugin.transport.TrainingJobRouterTransportAction.estimateVectorSetSizeInKB;

public class TrainingModelTransportActionTests extends KNNSingleNodeTestCase {

    public void testDoExecute() throws InterruptedException, ExecutionException, IOException {
        // Ingest training data into the cluster
        String trainingIndexName = "train-index";
        String trainingFieldName = "train-field";
        int dimension = 16;

        createIndex(trainingIndexName);
        createKnnIndexMapping(trainingIndexName, trainingFieldName, dimension);

        int trainingDataCount = 1000;
        for (int i = 0; i < trainingDataCount; i++) {
            Float[] vector = new Float[dimension];
            Arrays.fill(vector, Float.intBitsToFloat(i));
            addKnnDoc(trainingIndexName, Integer.toString(i + 1), trainingFieldName, vector);
        }

        // Create train model request
        String modelId = "test-model-id";
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_IVF)
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.INNER_PRODUCT.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_NLIST, 4)
            .endObject()
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext = KNNMethodContext.parse(in);

        TrainingModelRequest trainingModelRequest = new TrainingModelRequest(
            modelId,
            knnMethodContext,
            dimension,
            trainingIndexName,
            trainingFieldName,
            null,
            "test-detector",
            VectorDataType.DEFAULT
        );
        trainingModelRequest.setTrainingDataSizeInKB(estimateVectorSetSizeInKB(trainingDataCount, dimension, VectorDataType.DEFAULT));

        // Create listener that ensures that the initial model put succeeds
        ActionListener<TrainingModelResponse> listener = ActionListener.wrap(
            response -> assertEquals(modelId, response.getModelId()),
            e -> fail("Failure: " + e.getMessage())
        );

        TrainingModelTransportAction trainingModelTransportAction = node().injector().getInstance(TrainingModelTransportAction.class);

        trainingModelTransportAction.doExecute(null, trainingModelRequest, listener);

        // Wait for model to be created for a max of 30 seconds. If it is not created, fail.
        assertTrainingSucceeds(ModelDao.OpenSearchKNNModelDao.getInstance(), modelId, 30, 1000);
    }

}
