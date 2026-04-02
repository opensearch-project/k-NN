/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.processor.muvera;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.ingest.AbstractProcessor;
import org.opensearch.ingest.ConfigurationUtils;
import org.opensearch.ingest.IngestDocument;
import org.opensearch.ingest.Processor;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Ingest processor that converts multi-vector embeddings (e.g. ColBERT token vectors)
 * into a single Fixed Dimensional Encoding (FDE) vector using the MUVERA algorithm.
 *
 * The FDE vector is stored in the target knn_vector field for ANN search.
 * The original multi-vectors remain in the source field for optional reranking.
 *
 * Configuration:
 * <pre>
 * {
 *   "muvera": {
 *     "source_field": "colbert_vectors",   (required)
 *     "target_field": "muvera_fde",        (required)
 *     "dim": 128,                          (required - input vector dimension)
 *     "k_sim": 4,                          (default: 4)
 *     "dim_proj": 8,                       (default: 8)
 *     "r_reps": 20,                        (default: 20)
 *     "seed": 42,                          (default: 42)
 *     "fde_dimension": 2560                (optional - validates against computed value)
 *   }
 * }
 * </pre>
 *
 * The FDE output dimension = r_reps * 2^k_sim * dim_proj.
 * With defaults (r_reps=20, k_sim=4, dim_proj=8): 20 * 16 * 8 = 2560.
 * Use this value as the "dimension" in your knn_vector field mapping.
 *
 * The computed FDE dimension is logged at pipeline creation time and included
 * in the processor description visible via GET _ingest/pipeline.
 */
public class MuveraIngestProcessor extends AbstractProcessor {

    public static final String TYPE = "muvera";
    private static final Logger logger = LogManager.getLogger(MuveraIngestProcessor.class);

    private final String sourceField;
    private final String targetField;
    private final MuveraEncoder encoder;
    private final int dim;
    private final int fdeDimension;
    private final boolean ignoreMissing;

    MuveraIngestProcessor(
        String tag,
        String description,
        String sourceField,
        String targetField,
        MuveraEncoder encoder,
        int dim,
        int fdeDimension,
        boolean ignoreMissing
    ) {
        super(tag, description);
        this.sourceField = sourceField;
        this.targetField = targetField;
        this.encoder = encoder;
        this.dim = dim;
        this.fdeDimension = fdeDimension;
        this.ignoreMissing = ignoreMissing;
    }

    @Override
    public IngestDocument execute(IngestDocument document) throws Exception {
        if (document.hasField(sourceField) == false) {
            if (ignoreMissing) {
                return document;
            }
            throw new IllegalArgumentException("field [" + sourceField + "] not present, cannot generate MUVERA encoding");
        }

        Object sourceValue = document.getFieldValue(sourceField, Object.class);
        if (sourceValue == null) {
            if (ignoreMissing) {
                return document;
            }
            throw new IllegalArgumentException("field [" + sourceField + "] is null, cannot generate MUVERA encoding");
        }

        double[][] multiVectors = parseMultiVectors(sourceValue);
        if (multiVectors.length == 0) {
            throw new IllegalArgumentException("field [" + sourceField + "] contains empty multi-vector array");
        }

        // Validate each vector's dimension matches the configured dim
        for (int i = 0; i < multiVectors.length; i++) {
            if (multiVectors[i].length != dim) {
                throw new IllegalArgumentException(
                    "vector at ["
                        + sourceField
                        + "]["
                        + i
                        + "] has dimension ["
                        + multiVectors[i].length
                        + "], expected ["
                        + dim
                        + "]. Check the 'dim' parameter in your MUVERA ingest processor configuration."
                );
            }
        }

        float[] fde = encoder.processDocument(multiVectors);

        // Sanity check: FDE output dimension must match what we computed at pipeline creation
        if (fde.length != fdeDimension) {
            throw new IllegalStateException(
                "MUVERA encoder produced FDE of dimension ["
                    + fde.length
                    + "] but expected ["
                    + fdeDimension
                    + "]. This should not happen — please report this as a bug."
            );
        }

        // Convert float[] to List<Float> for IngestDocument compatibility
        List<Float> fdeList = new ArrayList<>(fde.length);
        for (float v : fde) {
            fdeList.add(v);
        }
        document.setFieldValue(targetField, fdeList);
        return document;
    }

    @SuppressWarnings("unchecked")
    private double[][] parseMultiVectors(Object value) {
        if (value instanceof List == false) {
            throw new IllegalArgumentException(
                "field [" + sourceField + "] must be a list of vectors (list of list of numbers), got: " + value.getClass().getSimpleName()
            );
        }
        List<?> outerList = (List<?>) value;
        double[][] result = new double[outerList.size()][];
        for (int i = 0; i < outerList.size(); i++) {
            Object vecObj = outerList.get(i);
            if (vecObj instanceof List == false) {
                throw new IllegalArgumentException(
                    "each vector in ["
                        + sourceField
                        + "] must be a list of numbers, element "
                        + i
                        + " is: "
                        + (vecObj == null ? "null" : vecObj.getClass().getSimpleName())
                );
            }
            List<?> vecList = (List<?>) vecObj;
            result[i] = new double[vecList.size()];
            for (int j = 0; j < vecList.size(); j++) {
                Object numObj = vecList.get(j);
                if (numObj instanceof Number == false) {
                    throw new IllegalArgumentException("vector element at [" + sourceField + "][" + i + "][" + j + "] is not a number");
                }
                result[i][j] = ((Number) numObj).doubleValue();
            }
        }
        return result;
    }

    @Override
    public String getType() {
        return TYPE;
    }

    public static final class Factory implements Processor.Factory {

        @Override
        public MuveraIngestProcessor create(
            Map<String, Processor.Factory> registry,
            String tag,
            String description,
            Map<String, Object> config
        ) throws Exception {
            String sourceField = ConfigurationUtils.readStringProperty(TYPE, tag, config, "source_field");
            String targetField = ConfigurationUtils.readStringProperty(TYPE, tag, config, "target_field");

            // dim is required — no sensible default since it depends on the embedding model
            Integer dimValue = ConfigurationUtils.readIntProperty(TYPE, tag, config, "dim", null);
            if (dimValue == null) {
                throw ConfigurationUtils.newConfigurationException(
                    TYPE,
                    tag,
                    "dim",
                    "required property is missing. Set this to your embedding model's vector dimension (e.g. 128)."
                );
            }
            int dim = dimValue;

            int kSim = ConfigurationUtils.readIntProperty(TYPE, tag, config, "k_sim", 4);
            int dimProj = ConfigurationUtils.readIntProperty(TYPE, tag, config, "dim_proj", 8);
            int rReps = ConfigurationUtils.readIntProperty(TYPE, tag, config, "r_reps", 20);
            long seed = MuveraProcessorUtils.readLongProperty(TYPE, tag, config, "seed", 42L);

            MuveraEncoder encoder = new MuveraEncoder(dim, kSim, dimProj, rReps, seed);
            int computedDimension = encoder.getEmbeddingSize();

            // Validate fde_dimension if provided
            Integer userDimension = ConfigurationUtils.readIntProperty(TYPE, tag, config, "fde_dimension", null);
            if (userDimension != null && userDimension != computedDimension) {
                throw new IllegalArgumentException(
                    "["
                        + TYPE
                        + "] processor ["
                        + tag
                        + "] fde_dimension ["
                        + userDimension
                        + "] does not match computed dimension ["
                        + computedDimension
                        + "] (r_reps="
                        + rReps
                        + " * 2^k_sim="
                        + (1 << kSim)
                        + " * dim_proj="
                        + dimProj
                        + ")"
                );
            }

            // Log the FDE dimension so the user knows what to set in their knn_vector mapping
            logger.info(
                "[{}] processor [{}]: computed FDE dimension = {} (r_reps={} * 2^k_sim={} * dim_proj={}). "
                    + "Use this as the 'dimension' in your knn_vector field mapping for [{}].",
                TYPE,
                tag,
                computedDimension,
                rReps,
                (1 << kSim),
                dimProj,
                targetField
            );

            // Auto-generate description if not provided, so it shows in GET _ingest/pipeline
            if (description == null || description.isEmpty()) {
                description = "MUVERA FDE encoder: dim="
                    + dim
                    + ", fde_dimension="
                    + computedDimension
                    + " (r_reps="
                    + rReps
                    + ", k_sim="
                    + kSim
                    + ", dim_proj="
                    + dimProj
                    + ", seed="
                    + seed
                    + ")";
            }

            boolean ignoreMissing = ConfigurationUtils.readBooleanProperty(TYPE, tag, config, "ignore_missing", false);

            return new MuveraIngestProcessor(
                tag, description, sourceField, targetField, encoder, dim, computedDimension, ignoreMissing
            );
        }
}
