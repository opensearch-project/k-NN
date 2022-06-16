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
import org.opensearch.action.StepListener;
import org.opensearch.action.admin.indices.create.CreateIndexResponse;
import org.opensearch.action.delete.DeleteAction;
import org.opensearch.action.delete.DeleteRequestBuilder;
import org.opensearch.action.delete.DeleteResponse;
import org.opensearch.action.index.IndexRequest;
import org.opensearch.action.index.IndexResponse;
import org.opensearch.action.support.WriteRequest;
import org.opensearch.action.support.master.AcknowledgedResponse;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.index.IndexNotFoundException;
import org.opensearch.index.engine.VersionConflictEngineException;
import org.opensearch.knn.KNNSingleNodeTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.plugin.transport.DeleteModelResponse;
import org.opensearch.knn.plugin.transport.GetModelResponse;
import org.opensearch.knn.plugin.transport.RemoveModelFromCacheAction;
import org.opensearch.knn.plugin.transport.RemoveModelFromCacheRequest;
import org.opensearch.knn.plugin.transport.RemoveModelFromCacheResponse;
import org.opensearch.knn.plugin.transport.UpdateBlockedModelAction;
import org.opensearch.knn.plugin.transport.UpdateBlockedModelRequest;
import org.opensearch.knn.plugin.transport.UpdateModelMetadataAction;
import org.opensearch.knn.plugin.transport.UpdateModelMetadataRequest;
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
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
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

    public void testModelIndexHealth() throws InterruptedException, ExecutionException, IOException {
        ModelDao modelDao = ModelDao.OpenSearchKNNModelDao.getInstance();

        // model index doesn't exist
        expectThrows(IndexNotFoundException.class, () -> modelDao.getHealthStatus());

        createIndex(MODEL_INDEX_NAME);

        // insert model
        String modelId = "created-1";
        byte[] modelBlob = "hello".getBytes();
        int dimension = 2;

        Model model = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.DEFAULT,
                dimension,
                ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(),
                "",
                ""
            ),
            modelBlob,
            modelId
        );
        addDoc(model);
        assertEquals(model, modelDao.get(modelId));
        assertNotNull(modelDao.getHealthStatus());

        modelId = "failed-2";
        model = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.DEFAULT,
                dimension,
                ModelState.FAILED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(),
                "",
                ""
            ),
            modelBlob,
            modelId
        );
        addDoc(model);
        assertEquals(model, modelDao.get(modelId));
        assertNotNull(modelDao.getHealthStatus());
    }

    public void testPut_withId() throws InterruptedException, IOException {
        createIndex(MODEL_INDEX_NAME);

        ModelDao modelDao = ModelDao.OpenSearchKNNModelDao.getInstance();
        String modelId = "efbsdhcvbsd"; // User provided model id
        byte[] modelBlob = "hello".getBytes();
        int dimension = 2;

        Model model = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.DEFAULT,
                dimension,
                ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(),
                "",
                ""
            ),
            modelBlob,
            modelId
        );

        // Listener to confirm that everything was updated as expected
        final CountDownLatch inProgressLatch1 = new CountDownLatch(1);
        ActionListener<IndexResponse> docCreationListener = ActionListener.wrap(response -> {
            assertEquals(modelId, response.getId());

            // We need to use executor service here so main thread does not block
            modelGetterExecutor.submit(() -> {
                try {
                    assertEquals(model, modelDao.get(modelId));
                } catch (ExecutionException | InterruptedException e) {
                    fail(e.getMessage());
                }
                inProgressLatch1.countDown();
            });

        }, exception -> fail("Unable to put the model: " + exception));

        modelDao.put(model, docCreationListener);

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
            }
        );

        modelDao.put(model, docCreationListenerDuplicateId);
        assertTrue(inProgressLatch2.await(100, TimeUnit.SECONDS));
    }

    public void testPut_withoutModel() throws InterruptedException, IOException {
        createIndex(MODEL_INDEX_NAME);

        ModelDao modelDao = ModelDao.OpenSearchKNNModelDao.getInstance();
        String modelId = "efbsdhcvbsd"; // User provided model id
        byte[] modelBlob = null;
        int dimension = 2;

        Model model = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.DEFAULT,
                dimension,
                ModelState.TRAINING,
                ZonedDateTime.now(ZoneOffset.UTC).toString(),
                "",
                ""
            ),
            modelBlob,
            modelId
        );

        // Listener to confirm that everything was updated as expected
        final CountDownLatch inProgressLatch1 = new CountDownLatch(1);
        ActionListener<IndexResponse> docCreationListener = ActionListener.wrap(response -> {
            assertEquals(modelId, response.getId());

            // We need to use executor service here so main thread does not block
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

        modelDao.put(model, docCreationListener);

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
            }
        );

        modelDao.put(model, docCreationListenerDuplicateId);
        assertTrue(inProgressLatch2.await(100, TimeUnit.SECONDS));
    }

    public void testPut_invalid_badState() {
        ModelDao modelDao = ModelDao.OpenSearchKNNModelDao.getInstance();
        byte[] modelBlob = null;
        int dimension = 2;

        createIndex(MODEL_INDEX_NAME);

        // Model is in invalid state
        Model model = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.DEFAULT,
                dimension,
                ModelState.TRAINING,
                ZonedDateTime.now(ZoneOffset.UTC).toString(),
                "",
                ""
            ),
            modelBlob,
            "any-id"
        );
        model.getModelMetadata().setState(ModelState.CREATED);

        expectThrows(
            IllegalArgumentException.class,
            () -> modelDao.put(
                model,
                ActionListener.wrap(
                    acknowledgedResponse -> fail("Should not get called."),
                    exception -> fail("Should not get to this call.")
                )
            )
        );
    }

    public void testUpdate() throws IOException, InterruptedException {
        createIndex(MODEL_INDEX_NAME);

        ModelDao modelDao = ModelDao.OpenSearchKNNModelDao.getInstance();
        String modelId = "efbsdhcvbsd"; // User provided model id
        byte[] modelBlob = "hello".getBytes();
        int dimension = 2;

        Model model = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.DEFAULT,
                dimension,
                ModelState.TRAINING,
                ZonedDateTime.now(ZoneOffset.UTC).toString(),
                "",
                ""
            ),
            null,
            modelId
        );

        // Listener to confirm that everything was updated as expected
        final CountDownLatch inProgressLatch1 = new CountDownLatch(1);
        ActionListener<IndexResponse> docCreationListener = ActionListener.wrap(response -> {
            assertEquals(modelId, response.getId());

            // We need to use executor service here so main thread does not block
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

        modelDao.put(model, docCreationListener);

        assertTrue(inProgressLatch1.await(100, TimeUnit.SECONDS));

        // User provided model id that already exists - should be able to update
        Model updatedModel = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.DEFAULT,
                dimension,
                ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(),
                "",
                ""
            ),
            modelBlob,
            modelId
        );

        final CountDownLatch inProgressLatch2 = new CountDownLatch(1);
        ActionListener<IndexResponse> updateListener = ActionListener.wrap(response -> {
            assertEquals(modelId, response.getId());

            // We need to use executor service here so main thread does not block
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

        modelDao.update(updatedModel, updateListener);
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
        Model model = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.DEFAULT,
                dimension,
                ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(),
                "",
                ""
            ),
            modelBlob,
            modelId
        );
        addDoc(model);
        assertEquals(model, modelDao.get(modelId));

        // Get model during training
        model = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.DEFAULT,
                dimension,
                ModelState.TRAINING,
                ZonedDateTime.now(ZoneOffset.UTC).toString(),
                "",
                ""
            ),
            null,
            modelId
        );
        addDoc(model);
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
        byte[] modelBlob = "hello".getBytes();

        KNNEngine knnEngine = KNNEngine.FAISS;
        SpaceType spaceType = SpaceType.INNER_PRODUCT;
        int dimension = 2;
        ModelMetadata modelMetadata = new ModelMetadata(
            knnEngine,
            spaceType,
            dimension,
            ModelState.CREATED,
            ZonedDateTime.now(ZoneOffset.UTC).toString(),
            "",
            ""
        );

        Model model = new Model(modelMetadata, modelBlob, modelId);

        // Listener to confirm that everything was updated as expected
        final CountDownLatch inProgressLatch1 = new CountDownLatch(1);
        ActionListener<IndexResponse> docCreationListener = ActionListener.wrap(response -> {
            assertEquals(modelId, response.getId());

            ModelMetadata modelMetadata1 = modelDao.getMetadata(modelId);
            assertEquals(modelMetadata, modelMetadata1);

            inProgressLatch1.countDown();
        }, exception -> fail("Unable to put the model: " + exception));

        // We use put so that we can confirm cluster metadata gets added
        modelDao.put(model, docCreationListener);

        assertTrue(inProgressLatch1.await(100, TimeUnit.SECONDS));
    }

    public void testDelete() throws IOException, InterruptedException {
        ModelDao modelDao = ModelDao.OpenSearchKNNModelDao.getInstance();
        String modelId = "testDeleteModelID";
        String modelId1 = "testDeleteModelID1";
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
            assertEquals(modelId, response.getModelID());
            assertEquals("failed", response.getResult());
            assertNotNull(response.getErrorMessage());
            inProgressLatch1.countDown();
        }, exception -> fail(exception.getMessage()));

        modelDao.delete(modelId, deleteModelDoesNotExistListener);
        assertTrue(inProgressLatch1.await(100, TimeUnit.SECONDS));

        final CountDownLatch inProgressLatch2 = new CountDownLatch(1);
        ActionListener<DeleteModelResponse> deleteModelTrainingListener = ActionListener.wrap(response -> {
            assertEquals(modelId, response.getModelID());
            assertEquals("failed", response.getResult());
            String errorMessage = String.format("Cannot delete model \"%s\". Model is still in training", modelId);
            assertEquals(errorMessage, response.getErrorMessage());
            inProgressLatch2.countDown();
        }, exception -> fail("Unable to delete model: " + exception));

        // model id exists and model is still in Training
        Model model = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.DEFAULT,
                dimension,
                ModelState.TRAINING,
                ZonedDateTime.now(ZoneOffset.UTC).toString(),
                "",
                ""
            ),
            modelBlob,
            modelId
        );

        ActionListener<IndexResponse> docCreationListener = ActionListener.wrap(response -> {
            assertEquals(modelId, response.getId());
            modelDao.delete(modelId, deleteModelTrainingListener);
        }, exception -> fail("Unable to put the model: " + exception));

        modelDao.put(model, docCreationListener);

        assertTrue(inProgressLatch2.await(100, TimeUnit.SECONDS));

        final CountDownLatch inProgressLatch3 = new CountDownLatch(1);
        ActionListener<DeleteModelResponse> deleteModelExistsListener = ActionListener.wrap(response -> {
            assertEquals(modelId1, response.getModelID());
            assertEquals(DocWriteResponse.Result.DELETED.getLowercase(), response.getResult());
            assertNull(response.getErrorMessage());
            inProgressLatch3.countDown();
        }, exception -> fail("Unable to delete model: " + exception));

        // model id exists
        Model model1 = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.DEFAULT,
                dimension,
                ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(),
                "",
                ""
            ),
            modelBlob,
            modelId1
        );

        ActionListener<IndexResponse> docCreationListener1 = ActionListener.wrap(response -> {
            assertEquals(modelId1, response.getId());
            modelDao.delete(modelId1, deleteModelExistsListener);
        }, exception -> fail("Unable to put the model: " + exception));

        // We use put so that we can confirm cluster metadata gets added
        modelDao.put(model1, docCreationListener1);

        assertTrue(inProgressLatch3.await(100, TimeUnit.SECONDS));
    }

    public void testDeleteWithStepListeners() throws IOException, InterruptedException, ExecutionException {
        String modelId = "test-model-id-delete";
        ModelDao modelDao = ModelDao.OpenSearchKNNModelDao.getInstance();
        byte[] modelBlob = "deleteModel".getBytes();
        int dimension = 2;
        createIndex(MODEL_INDEX_NAME);

        Model model = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.DEFAULT,
                dimension,
                ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(),
                "",
                ""
            ),
            modelBlob,
            modelId
        );

        // created model and added it to index
        addDoc(model);

        final CountDownLatch inProgressLatch = new CountDownLatch(1);

        StepListener<GetModelResponse> getModelStep = new StepListener<>();
        StepListener<AcknowledgedResponse> blockModelIdStep = new StepListener<>();
        StepListener<AcknowledgedResponse> clearModelMetadataStep = new StepListener<>();
        StepListener<DeleteResponse> deleteModelFromIndexStep = new StepListener<>();
        StepListener<RemoveModelFromCacheResponse> clearModelFromCacheStep = new StepListener<>();
        StepListener<AcknowledgedResponse> unblockModelIdStep = new StepListener<>();

        modelDao.get(modelId, ActionListener.wrap(getModelStep::onResponse, getModelStep::onFailure));

        // Asserting that model is in CREATED state
        getModelStep.whenComplete(getModelResponse -> {
            assertEquals(model.getModelMetadata().getState(), getModelResponse.getModel().getModelMetadata().getState());
            assertNotEquals(ModelState.TRAINING.getName(), getModelResponse.getModel().getModelMetadata().getState().toString());

            client().execute(
                UpdateBlockedModelAction.INSTANCE,
                new UpdateBlockedModelRequest(modelId, false),
                ActionListener.wrap(blockModelIdStep::onResponse, blockModelIdStep::onFailure)
            );
        }, exception -> fail(exception.getMessage()));

        blockModelIdStep.whenComplete(acknowledgedResponse -> {
            // Asserting that modelId is in blocked list
            assertTrue(modelDao.isModelBlocked(modelId));

            client().execute(
                UpdateModelMetadataAction.INSTANCE,
                new UpdateModelMetadataRequest(modelId, true, null),
                ActionListener.wrap(clearModelMetadataStep::onResponse, clearModelMetadataStep::onFailure)
            );

        }, exception -> fail(exception.getMessage()));

        DeleteRequestBuilder deleteRequestBuilder = new DeleteRequestBuilder(client(), DeleteAction.INSTANCE, MODEL_INDEX_NAME);
        deleteRequestBuilder.setId(modelId);
        deleteRequestBuilder.setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);

        clearModelMetadataStep.whenComplete(acknowledgedResponse -> {
            // Asserting that metadata is cleared
            assertNull(modelDao.getMetadata(modelId));

            deleteRequestBuilder.execute(ActionListener.wrap(deleteModelFromIndexStep::onResponse, deleteModelFromIndexStep::onFailure));

        }, exception -> fail(exception.getMessage()));

        deleteModelFromIndexStep.whenComplete(deleteResponse -> {
            // Asserting that model is deleted from index
            assertEquals(DocWriteResponse.Result.DELETED, deleteResponse.getResult());
            client().execute(
                RemoveModelFromCacheAction.INSTANCE,
                new RemoveModelFromCacheRequest(modelId),
                ActionListener.wrap(clearModelFromCacheStep::onResponse, clearModelFromCacheStep::onFailure)
            );

        }, exception -> fail(exception.getMessage()));

        clearModelFromCacheStep.whenComplete(removeModelFromCacheResponse -> {
            client().execute(
                UpdateBlockedModelAction.INSTANCE,
                new UpdateBlockedModelRequest(modelId, true),
                ActionListener.wrap(unblockModelIdStep::onResponse, unblockModelIdStep::onFailure)
            );
        }, exception -> fail(exception.getMessage()));

        unblockModelIdStep.whenComplete(acknowledgedResponse -> {
            // Asserting that model is unblocked
            assertFalse(modelDao.isModelBlocked(modelId));
            inProgressLatch.countDown();
        }, exception -> fail(exception.getMessage()));

        assertTrue(inProgressLatch.await(100, TimeUnit.SECONDS));
    }

    public void testDeleteWithStepListenersOnFailure() throws IOException, InterruptedException, ExecutionException {
        String modelId = "test-model-id-delete1";
        ModelDao modelDao = ModelDao.OpenSearchKNNModelDao.getInstance();
        byte[] modelBlob = "deleteModel".getBytes();
        int dimension = 2;
        createIndex(MODEL_INDEX_NAME);
        Model model = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.DEFAULT,
                dimension,
                ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(),
                "",
                ""
            ),
            modelBlob,
            modelId
        );

        addDoc(model);

        // We will validate if the modelId gets unblocked when some exception occurs
        // during the process of deletion after adding that modelId to blocked list
        final CountDownLatch inProgressLatch = new CountDownLatch(1);

        StepListener<AcknowledgedResponse> blockModelIdStep = new StepListener<>();
        StepListener<AcknowledgedResponse> clearModelMetadataStep = new StepListener<>();

        // Add modelId to blocked list
        client().execute(
            UpdateBlockedModelAction.INSTANCE,
            new UpdateBlockedModelRequest(modelId, false),
            ActionListener.wrap(blockModelIdStep::onResponse, blockModelIdStep::onFailure)
        );

        // Asserting that the modelId is blocked
        blockModelIdStep.whenComplete(acknowledgedResponse -> {
            assertTrue(modelDao.isModelBlocked(modelId));

            // Sending empty string for modelId to fail the clear model metadata request
            client().execute(
                UpdateModelMetadataAction.INSTANCE,
                new UpdateModelMetadataRequest("", true, null),
                ActionListener.wrap(clearModelMetadataStep::onResponse, exp -> {
                    // Asserting that modelId is still blocked and clearModelMetadata throws an exception
                    assertNotNull(exp.getMessage());
                    assertTrue(modelDao.isModelBlocked(modelId));
                    client().execute(
                        // OnFailure sending request to unblock modelId
                        UpdateBlockedModelAction.INSTANCE,
                        new UpdateBlockedModelRequest(modelId, true),
                        ActionListener.wrap(ackResponse -> {
                            // Asserting that model is unblocked
                            assertFalse(modelDao.isModelBlocked(modelId));
                            assertNotNull(exp.getMessage());
                        }, exception -> fail(exception.getMessage()))
                    );
                })
            );
            inProgressLatch.countDown();
        }, exception -> fail(exception.getMessage()));

        assertTrue(inProgressLatch.await(50, TimeUnit.SECONDS));

        // Some exception occurs during the process of deletion and unblocking model request also fails
        final CountDownLatch inProgressLatch1 = new CountDownLatch(1);

        StepListener<AcknowledgedResponse> blockModelIdStep1 = new StepListener<>();
        StepListener<AcknowledgedResponse> clearModelMetadataStep1 = new StepListener<>();

        // Add modelId to blocked list
        client().execute(
            UpdateBlockedModelAction.INSTANCE,
            new UpdateBlockedModelRequest(modelId, false),
            ActionListener.wrap(blockModelIdStep1::onResponse, blockModelIdStep1::onFailure)
        );

        // Asserting that the modelId is blocked
        blockModelIdStep1.whenComplete(acknowledgedResponse -> {
            assertTrue(modelDao.isModelBlocked(modelId));

            // Sending empty string for modelId to fail the clear model metadata request
            client().execute(
                UpdateModelMetadataAction.INSTANCE,
                new UpdateModelMetadataRequest("", true, null),
                ActionListener.wrap(clearModelMetadataStep1::onResponse, exp -> {
                    assertNotNull(exp.getMessage());
                    assertTrue(modelDao.isModelBlocked(modelId));

                    // Failing unblock modelId request by sending modelId as an empty string
                    client().execute(
                        UpdateBlockedModelAction.INSTANCE,
                        new UpdateBlockedModelRequest("", true),
                        ActionListener.wrap(ackResponse -> {}, unblockingFailedException -> {
                            // Asserting that model is still blocked and returns both exceptions in response
                            assertTrue(modelDao.isModelBlocked(modelId));
                            assertNotNull(exp.getMessage());
                            assertNotNull(unblockingFailedException.getMessage());
                        })
                    );
                })
            );
            inProgressLatch1.countDown();
        }, exception -> fail(exception.getMessage()));

        assertTrue(inProgressLatch1.await(50, TimeUnit.SECONDS));
    }

    public void addDoc(Model model) throws IOException, ExecutionException, InterruptedException {
        ModelMetadata modelMetadata = model.getModelMetadata();

        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .field(MODEL_ID, model.getModelID())
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

        IndexRequest indexRequest = new IndexRequest().index(MODEL_INDEX_NAME)
            .id(model.getModelID())
            .source(builder)
            .setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);

        IndexResponse response = client().index(indexRequest).get();
        assertTrue(response.status() == RestStatus.CREATED || response.status() == RestStatus.OK);
    }

}
