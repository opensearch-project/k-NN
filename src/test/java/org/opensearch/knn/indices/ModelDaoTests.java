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

package org.opensearch.knn.indices;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.opensearch.ExceptionsHelper;
import org.opensearch.ResourceAlreadyExistsException;
import org.opensearch.action.ActionListener;
import org.opensearch.action.DocWriteResponse;
import org.opensearch.action.admin.indices.create.CreateIndexResponse;
import org.opensearch.action.delete.DeleteResponse;
import org.opensearch.action.index.IndexRequest;
import org.opensearch.action.index.IndexResponse;
import org.opensearch.action.support.WriteRequest;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.index.engine.VersionConflictEngineException;
import org.opensearch.knn.KNNSingleNodeTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.plugin.transport.DeleteModelResponse;
import org.opensearch.rest.RestStatus;

import java.io.IOException;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.Base64;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MODEL_BLOB_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.MODEL_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.MODEL_ERROR;
import static org.opensearch.knn.common.KNNConstants.MODEL_INDEX_NAME;
import static org.opensearch.knn.common.KNNConstants.MODEL_STATE;
import static org.opensearch.knn.common.KNNConstants.MODEL_TIMESTAMP;

public class ModelDaoTests extends KNNSingleNodeTestCase {

    private static ExecutorService modelGetterExecutor;

    @BeforeClass
    public static void setup() {
        modelGetterExecutor = Executors.newSingleThreadExecutor();
    }

    @AfterClass
    public static void teardown() {
        modelGetterExecutor.shutdown();
    }

    public void testCreate() throws IOException, InterruptedException {
        int attempts = 3;
        final CountDownLatch inProgressLatch = new CountDownLatch(attempts);

        ModelDao modelDao = ModelDao.OpenSearchKNNModelDao.getInstance();

        ActionListener<CreateIndexResponse> indexCreationListener = ActionListener.wrap(response -> {
            assertTrue(modelDao.isCreated());
            assertTrue(response.isAcknowledged());
            inProgressLatch.countDown();
        }, exception -> {
            if (!(ExceptionsHelper.unwrapCause(exception) instanceof ResourceAlreadyExistsException)) {
                fail("Failed for reason other than ResourceAlreadyExistsException: " + exception);
            }
            inProgressLatch.countDown();
        });

        for (int i = 0; i < attempts; i++) {
            modelDao.create(indexCreationListener);
        }

        assertTrue(inProgressLatch.await(100, TimeUnit.SECONDS));
    }

    public void testExists() {
        ModelDao modelDao = ModelDao.OpenSearchKNNModelDao.getInstance();
        assertFalse(modelDao.isCreated());
        createIndex(MODEL_INDEX_NAME);
        assertTrue(modelDao.isCreated());
    }

    public void testPut_withId() throws InterruptedException, IOException {
        createIndex(MODEL_INDEX_NAME);

        ModelDao modelDao = ModelDao.OpenSearchKNNModelDao.getInstance();
        String modelId = "efbsdhcvbsd"; // User provided model id
        byte [] modelBlob = "hello".getBytes();
        int dimension = 2;

        Model model = new Model(new ModelMetadata(KNNEngine.DEFAULT, SpaceType.DEFAULT, dimension, ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(), "", ""), modelBlob);

        // Listener to confirm that everything was updated as expected
        final CountDownLatch inProgressLatch1 = new CountDownLatch(1);
        ActionListener<IndexResponse> docCreationListener = ActionListener.wrap(response -> {
            assertEquals(modelId, response.getId());

            // We need to use executor service here so master thread does not block
            modelGetterExecutor.submit(() -> {
                try {
                    assertEquals(model, modelDao.get(modelId));
                } catch (ExecutionException | InterruptedException e) {
                    fail(e.getMessage());
                }
                inProgressLatch1.countDown();
            });

        }, exception -> fail("Unable to put the model: " + exception));

        modelDao.put(modelId, model, docCreationListener);

        assertTrue(inProgressLatch1.await(100, TimeUnit.SECONDS));

        // User provided model id that already exists
        final CountDownLatch inProgressLatch2 = new CountDownLatch(1);
        ActionListener<IndexResponse> docCreationListenerDuplicateId = ActionListener.wrap(
                response -> fail("Model already exists, but creation was successful"),
                exception -> {
                    if (!(ExceptionsHelper.unwrapCause(exception) instanceof VersionConflictEngineException)) {
                        fail("Unable to put the model: " + exception);
                    }
                    inProgressLatch2.countDown();
        });

        modelDao.put(modelId, model, docCreationListenerDuplicateId);
        assertTrue(inProgressLatch2.await(100, TimeUnit.SECONDS));
    }

    public void testPut_withoutModel() throws InterruptedException, IOException {
        createIndex(MODEL_INDEX_NAME);

        ModelDao modelDao = ModelDao.OpenSearchKNNModelDao.getInstance();
        String modelId = "efbsdhcvbsd"; // User provided model id
        byte [] modelBlob = null;
        int dimension = 2;

        Model model = new Model(new ModelMetadata(KNNEngine.DEFAULT, SpaceType.DEFAULT, dimension, ModelState.TRAINING,
                ZonedDateTime.now(ZoneOffset.UTC).toString(), "", ""), modelBlob);

        // Listener to confirm that everything was updated as expected
        final CountDownLatch inProgressLatch1 = new CountDownLatch(1);
        ActionListener<IndexResponse> docCreationListener = ActionListener.wrap(response -> {
            assertEquals(modelId, response.getId());

            // We need to use executor service here so master thread does not block
            modelGetterExecutor.submit(() -> {
                try {
                    assertEquals(model, modelDao.get(modelId));
                } catch (ExecutionException | InterruptedException e) {
                    fail(e.getMessage());
                }
                inProgressLatch1.countDown();
            });

            inProgressLatch1.countDown();
        }, exception -> fail("Unable to put the model: " + exception));

        modelDao.put(modelId, model, docCreationListener);

        assertTrue(inProgressLatch1.await(100, TimeUnit.SECONDS));

        // User provided model id that already exists
        final CountDownLatch inProgressLatch2 = new CountDownLatch(1);
        ActionListener<IndexResponse> docCreationListenerDuplicateId = ActionListener.wrap(
                response -> fail("Model already exists, but creation was successful"),
                exception -> {
                    if (!(ExceptionsHelper.unwrapCause(exception) instanceof VersionConflictEngineException)) {
                        fail("Unable to put the model: " + exception);
                    }
                    inProgressLatch2.countDown();
                });

        modelDao.put(modelId, model, docCreationListenerDuplicateId);
        assertTrue(inProgressLatch2.await(100, TimeUnit.SECONDS));
    }

    public void testPut_withoutId() throws InterruptedException, IOException {
        ModelDao modelDao = ModelDao.OpenSearchKNNModelDao.getInstance();
        byte [] modelBlob = "hello".getBytes();
        int dimension = 2;

        createIndex(MODEL_INDEX_NAME);

        // User does not provide model id
        final CountDownLatch inProgressLatch = new CountDownLatch(1);
        ActionListener<IndexResponse> docCreationListenerNoModelId = ActionListener.wrap(response -> {
                    inProgressLatch.countDown();
                },
                exception -> fail("Unable to put the model: " + exception));

        Model model = new Model(new ModelMetadata(KNNEngine.DEFAULT, SpaceType.DEFAULT, dimension, ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(), "", ""), modelBlob);
        modelDao.put(model, docCreationListenerNoModelId);
        assertTrue(inProgressLatch.await(100, TimeUnit.SECONDS));
    }

    public void testPut_invalid_badState() {
        ModelDao modelDao = ModelDao.OpenSearchKNNModelDao.getInstance();
        byte [] modelBlob = null;
        int dimension = 2;

        createIndex(MODEL_INDEX_NAME);

        // Model is in invalid state
        Model model = new Model(new ModelMetadata(KNNEngine.DEFAULT, SpaceType.DEFAULT, dimension, ModelState.TRAINING,
                ZonedDateTime.now(ZoneOffset.UTC).toString(), "", ""), modelBlob);
        model.getModelMetadata().setState(ModelState.CREATED);

        expectThrows(IllegalArgumentException.class, () -> modelDao.put(model, ActionListener.wrap(
                acknowledgedResponse -> fail("Should not get called."),
                exception -> fail("Should not get to this call."))));
    }

    public void testUpdate() throws IOException, InterruptedException {
        createIndex(MODEL_INDEX_NAME);

        ModelDao modelDao = ModelDao.OpenSearchKNNModelDao.getInstance();
        String modelId = "efbsdhcvbsd"; // User provided model id
        byte [] modelBlob = "hello".getBytes();
        int dimension = 2;

        Model model = new Model(new ModelMetadata(KNNEngine.DEFAULT, SpaceType.DEFAULT, dimension, ModelState.TRAINING,
                ZonedDateTime.now(ZoneOffset.UTC).toString(), "", ""), null);

        // Listener to confirm that everything was updated as expected
        final CountDownLatch inProgressLatch1 = new CountDownLatch(1);
        ActionListener<IndexResponse> docCreationListener = ActionListener.wrap(response -> {
            assertEquals(modelId, response.getId());

            // We need to use executor service here so master thread does not block
            modelGetterExecutor.submit(() -> {
                try {
                    assertEquals(model, modelDao.get(modelId));
                } catch (ExecutionException | InterruptedException e) {
                    fail(e.getMessage());
                }
                inProgressLatch1.countDown();
            });

            inProgressLatch1.countDown();
        }, exception -> fail("Unable to put the model: " + exception));

        modelDao.put(modelId, model, docCreationListener);

        assertTrue(inProgressLatch1.await(100, TimeUnit.SECONDS));

        // User provided model id that already exists - should be able to update
        Model updatedModel = new Model(new ModelMetadata(KNNEngine.DEFAULT, SpaceType.DEFAULT, dimension, ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(), "", ""), modelBlob);

        final CountDownLatch inProgressLatch2 = new CountDownLatch(1);
        ActionListener<IndexResponse> updateListener = ActionListener.wrap(response -> {
            assertEquals(modelId, response.getId());

            // We need to use executor service here so master thread does not block
            modelGetterExecutor.submit(() -> {
                try {
                    assertEquals(updatedModel, modelDao.get(modelId));
                } catch (ExecutionException | InterruptedException e) {
                    fail(e.getMessage());
                }
                inProgressLatch1.countDown();
            });

            inProgressLatch2.countDown();
        }, exception -> fail("Unable to put the model: " + exception));

        modelDao.update(modelId, updatedModel, updateListener);
        assertTrue(inProgressLatch2.await(100, TimeUnit.SECONDS));
    }

    public void testGet() throws IOException, InterruptedException, ExecutionException {
        ModelDao modelDao = ModelDao.OpenSearchKNNModelDao.getInstance();
        String modelId = "efbsdhcvbsd";
        byte[] modelBlob = "hello".getBytes();
        int dimension = 2;

        // model index doesnt exist
        expectThrows(ExecutionException.class, () -> modelDao.get(modelId));

        // model id doesnt exist
        createIndex(MODEL_INDEX_NAME);
        expectThrows(Exception.class, () -> modelDao.get(modelId));

        // model id exists
        Model model = new Model(new ModelMetadata(KNNEngine.DEFAULT, SpaceType.DEFAULT, dimension, ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(), "", ""), modelBlob);
        addDoc(modelId, model);
        assertEquals(model, modelDao.get(modelId));

        // Get model during training
        model = new Model(new ModelMetadata(KNNEngine.DEFAULT, SpaceType.DEFAULT, dimension, ModelState.TRAINING,
                ZonedDateTime.now(ZoneOffset.UTC).toString(), "", ""), null);
        addDoc(modelId, model);
        assertEquals(model, modelDao.get(modelId));
    }

    public void testGetMetadata() throws IOException, InterruptedException {
        ModelDao modelDao = ModelDao.OpenSearchKNNModelDao.getInstance();

        String modelId = "test-model";

        // Model Index does not exist
        assertNull(modelDao.getMetadata(modelId));

        createIndex(MODEL_INDEX_NAME);

        // Model id does not exist
        assertNull(modelDao.getMetadata(modelId));

        // Model exists
        byte [] modelBlob = "hello".getBytes();

        KNNEngine knnEngine = KNNEngine.FAISS;
        SpaceType spaceType = SpaceType.INNER_PRODUCT;
        int dimension = 2;
        ModelMetadata modelMetadata = new ModelMetadata(knnEngine, spaceType, dimension, ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(), "", "");

        Model model = new Model(modelMetadata, modelBlob);

        // Listener to confirm that everything was updated as expected
        final CountDownLatch inProgressLatch1 = new CountDownLatch(1);
        ActionListener<IndexResponse> docCreationListener = ActionListener.wrap(response -> {
            assertEquals(modelId, response.getId());

            ModelMetadata modelMetadata1 = modelDao.getMetadata(modelId);
            assertEquals(modelMetadata, modelMetadata1);

            inProgressLatch1.countDown();
        }, exception -> fail("Unable to put the model: " + exception));

        // We use put so that we can confirm cluster metadata gets added
        modelDao.put(modelId, model, docCreationListener);

        assertTrue(inProgressLatch1.await(100, TimeUnit.SECONDS));
    }

    public void testDelete() throws IOException, InterruptedException {
        ModelDao modelDao = ModelDao.OpenSearchKNNModelDao.getInstance();
        String modelId = "testDeleteModelID";
        byte[] modelBlob = "hello".getBytes();
        int dimension = 2;

        final CountDownLatch inProgressLatch = new CountDownLatch(1);
        ActionListener<DeleteModelResponse> deleteModelIndexDoesNotExistListener = ActionListener.wrap(response -> {
            assertEquals("failed", response.getResult());
            inProgressLatch.countDown();
        }, exception -> fail("Unable to delete the model: " + exception));
        // model index doesnt exist
        modelDao.delete(modelId, deleteModelIndexDoesNotExistListener);
        assertTrue(inProgressLatch.await(100, TimeUnit.SECONDS));

        createIndex(MODEL_INDEX_NAME);

        final CountDownLatch inProgressLatch1 = new CountDownLatch(1);
        ActionListener<DeleteModelResponse> deleteModelDoesNotExistListener = ActionListener.wrap(response -> {
            assertEquals(DocWriteResponse.Result.NOT_FOUND.getLowercase(), response.getResult());
            inProgressLatch1.countDown();
        }, exception -> fail("Unable to delete the model: " + exception));

        modelDao.delete(modelId, deleteModelDoesNotExistListener);
        assertTrue(inProgressLatch1.await(100, TimeUnit.SECONDS));

        final CountDownLatch inProgressLatch2 = new CountDownLatch(1);
        ActionListener<DeleteModelResponse> deleteModelExistsListener = ActionListener.wrap(response -> {
            assertEquals(modelId, response.getModelID());
            assertEquals(DocWriteResponse.Result.DELETED.getLowercase(), response.getResult());
            assertNull(response.getErrorMessage());
            inProgressLatch2.countDown();
        }, exception -> fail("Unable to delete model: " + exception));

        // model id exists
        Model model = new Model(new ModelMetadata(KNNEngine.DEFAULT, SpaceType.DEFAULT, dimension, ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(), "", ""), modelBlob);

        ActionListener<IndexResponse> docCreationListener = ActionListener.wrap(response -> {
            assertEquals(modelId, response.getId());
            modelDao.delete(modelId, deleteModelExistsListener);
        }, exception -> fail("Unable to put the model: " + exception));

        // We use put so that we can confirm cluster metadata gets added
        modelDao.put(modelId, model, docCreationListener);

        assertTrue(inProgressLatch2.await(100, TimeUnit.SECONDS));
    }

    public void addDoc(String modelId, Model model) throws IOException, ExecutionException, InterruptedException {
        ModelMetadata modelMetadata = model.getModelMetadata();

        XContentBuilder builder = XContentFactory.jsonBuilder().startObject()
                .field(KNN_ENGINE, modelMetadata.getKnnEngine().getName())
                .field(METHOD_PARAMETER_SPACE_TYPE, modelMetadata.getSpaceType().getValue())
                .field(DIMENSION, modelMetadata.getDimension())
                .field(MODEL_STATE, modelMetadata.getState().getName())
                .field(MODEL_TIMESTAMP, modelMetadata.getTimestamp().toString())
                .field(MODEL_DESCRIPTION, modelMetadata.getDescription())
                .field(MODEL_ERROR, modelMetadata.getError());

        if (model.getModelBlob() != null) {
            builder.field(MODEL_BLOB_PARAMETER, Base64.getEncoder().encodeToString(model.getModelBlob()));
        }

        builder.endObject();

        IndexRequest indexRequest = new IndexRequest()
                .index(MODEL_INDEX_NAME)
                .id(modelId)
                .source(builder)
                .setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);

        IndexResponse response = client().index(indexRequest).get();
        assertTrue(response.status() == RestStatus.CREATED || response.status() == RestStatus.OK);
    }
}
