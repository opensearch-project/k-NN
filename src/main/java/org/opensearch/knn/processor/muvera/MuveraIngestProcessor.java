/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.processor.muvera;

import lombok.extern.log4j.Log4j2;
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
 * <p>The FDE vector is stored in the configured target {@code knn_vector} field for ANN
 * search. The original multi-vectors remain in the source field for optional reranking
 * via the {@code lateInteractionScore} script.
 *
 * <p>Configuration:
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
 * <p>The FDE output dimension equals {@code r_reps * 2^k_sim * dim_proj}. With defaults
 * (r_reps=20, k_sim=4, dim_proj=8) this is 2560. This same value must be set as the
 * {@code dimension} of the target {@code knn_vector} field mapping; the computed value
 * is also recorded in the processor description so it shows up under
 * {@code GET _ingest/pipeline}.
 */
@Log4j2
public class MuveraIngestProcessor extends AbstractProcessor {

    /** Processor type identifier registered with the ingest pipeline framework. */
    public static final String TYPE = "muvera";

    /** Configuration property keys. */
    static final String CONFIG_SOURCE_FIELD = "source_field";
    static final String CONFIG_TARGET_FIELD = "target_field";
    static final String CONFIG_DIM = "dim";
    static final String CONFIG_K_SIM = "k_sim";
    static final String CONFIG_DIM_PROJ = "dim_proj";
    static final String CONFIG_R_REPS = "r_reps";
    static final String CONFIG_SEED = "seed";
    static final String CONFIG_FDE_DIMENSION = "fde_dimension";
    static final String CONFIG_IGNORE_MISSING = "ignore_missing";

    /** Default values for optional configuration properties. */
    static final int DEFAULT_K_SIM = 4;
    static final int DEFAULT_DIM_PROJ = 8;
    static final int DEFAULT_R_REPS = 20;
    static final long DEFAULT_SEED = 42L;
    static final boolean DEFAULT_IGNORE_MISSING = false;

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

        float[][] multiVectors = parseMultiVectors(sourceValue);
        if (multiVectors.length == 0) {
            throw new IllegalArgumentException("field [" + sourceField + "] contains empty multi-vector array");
        }

        // Validate each vector's dimension matches the configured dim.
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

        // Sanity check: encoder output must match the dimension we computed at pipeline creation.
        if (fde.length != fdeDimension) {
            throw new IllegalStateException(
                "MUVERA encoder produced FDE of dimension ["
                    + fde.length
                    + "] but expected ["
                    + fdeDimension
                    + "]. This should not happen — please report this as a bug."
            );
        }

        // IngestDocument.deepCopy supports only Map/List/primitives — float[] would fail.
        // Convert to List<Float> here; the boxing cost is amortized over the document
        // ingest path which already materializes _source as a Map/List structure.
        List<Float> fdeList = new ArrayList<>(fde.length);
        for (float v : fde) {
            fdeList.add(v);
        }
        document.setFieldValue(targetField, fdeList);
        return document;
    }

    /**
     * Parses a {@code List<List<Number>>}-shaped value from {@code _source} into a flat
     * {@code float[][]} for the encoder.
     */
    private float[][] parseMultiVectors(Object value) {
        if (value instanceof List == false) {
            throw new IllegalArgumentException(
                "field [" + sourceField + "] must be a list of vectors (list of list of numbers), got: " + value.getClass().getSimpleName()
            );
        }
        List<?> outerList = (List<?>) value;
        float[][] result = new float[outerList.size()][];
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
            float[] tokenVec = new float[vecList.size()];
            for (int j = 0; j < vecList.size(); j++) {
                Object numObj = vecList.get(j);
                if (numObj instanceof Number == false) {
                    throw new IllegalArgumentException("vector element at [" + sourceField + "][" + i + "][" + j + "] is not a number");
                }
                tokenVec[j] = ((Number) numObj).floatValue();
            }
            result[i] = tokenVec;
        }
        return result;
    }

    @Override
    public String getType() {
        return TYPE;
    }

    /** Factory used by the ingest pipeline framework to instantiate the processor. */
    public static final class Factory implements Processor.Factory {

        @Override
        public MuveraIngestProcessor create(
            Map<String, Processor.Factory> registry,
            String tag,
            String description,
            Map<String, Object> config
        ) throws Exception {
            String sourceField = ConfigurationUtils.readStringProperty(TYPE, tag, config, CONFIG_SOURCE_FIELD);
            String targetField = ConfigurationUtils.readStringProperty(TYPE, tag, config, CONFIG_TARGET_FIELD);

            // dim is required — no sensible default since it depends on the embedding model.
            Integer dimValue = ConfigurationUtils.readIntProperty(TYPE, tag, config, CONFIG_DIM, null);
            if (dimValue == null) {
                throw ConfigurationUtils.newConfigurationException(
                    TYPE,
                    tag,
                    CONFIG_DIM,
                    "required property is missing. Set this to your embedding model's vector dimension (e.g. 128)."
                );
            }
            int dim = dimValue;

            int kSim = ConfigurationUtils.readIntProperty(TYPE, tag, config, CONFIG_K_SIM, DEFAULT_K_SIM);
            int dimProj = ConfigurationUtils.readIntProperty(TYPE, tag, config, CONFIG_DIM_PROJ, DEFAULT_DIM_PROJ);
            int rReps = ConfigurationUtils.readIntProperty(TYPE, tag, config, CONFIG_R_REPS, DEFAULT_R_REPS);
            long seed = MuveraProcessorUtils.readLongProperty(TYPE, tag, config, CONFIG_SEED, DEFAULT_SEED);

            MuveraEncoder encoder = new MuveraEncoder(dim, kSim, dimProj, rReps, seed);
            int computedDimension = encoder.getEmbeddingSize();

            // Validate fde_dimension if provided so users catch mapping mismatches at
            // pipeline creation rather than at indexing time.
            Integer userDimension = ConfigurationUtils.readIntProperty(TYPE, tag, config, CONFIG_FDE_DIMENSION, null);
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

            // Auto-generate the description (visible via GET _ingest/pipeline) if the user
            // didn't supply one. This is how end users discover the computed FDE dimension —
            // we no longer log it at the cluster level since cluster logs aren't user visible.
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

            boolean ignoreMissing = ConfigurationUtils.readBooleanProperty(
                TYPE,
                tag,
                config,
                CONFIG_IGNORE_MISSING,
                DEFAULT_IGNORE_MISSING
            );

            return new MuveraIngestProcessor(tag, description, sourceField, targetField, encoder, dim, computedDimension, ignoreMissing);
        }
    }
}
