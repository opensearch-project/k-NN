/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.indices;

import lombok.AllArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.opensearch.Version;
import org.opensearch.cluster.Diff;
import org.opensearch.cluster.NamedDiff;
import org.opensearch.cluster.metadata.Metadata;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentParser;

import java.io.IOException;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;
import com.google.common.collect.Sets;

/**
 * This class implements Metadata.Custom Interface to store a set of modelIds in the cluster metadata.
 * The modelIds of the models that are under deletion are added to this set and later removed from this set after deletion.
 * Also, this class implements the methods to perform operations on this set (like add, remove, contains)
 */

@AllArgsConstructor
@Log4j2
public class ModelGraveyard implements Metadata.Custom {
    public static final String TYPE = "opensearch-knn-blocked-models";
    private final Set<String> modelIds;

    /**
     * Default Constructor to initialize object when it is null
     */
    public ModelGraveyard() {
        this.modelIds = new HashSet<>();
    }

    /**
     * @param in input stream
     * @throws IOException if read from stream fails
     */
    public ModelGraveyard(StreamInput in) throws IOException {
        this.modelIds = new HashSet<>(in.readStringList());
    }

    @Override
    public EnumSet<Metadata.XContentContext> context() {
        return Metadata.ALL_CONTEXTS;
    }

    /**
     * @return WriteableName for ModelGraveyard
     */
    @Override
    public String getWriteableName() {
        return TYPE;
    }

    @Override
    public Version getMinimalSupportedVersion() {
        return Version.CURRENT.minimumCompatibilityVersion();
    }

    /**
     * @param out output stream
     * @throws IOException if write to stream fails
     */
    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeStringCollection(modelIds);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        return builder;
    }

    /**
     * @param modelId id of the model that needs to be removed from modelIds set
     */
    public void remove(String modelId) {
        modelIds.remove(modelId);
    }

    /**
     * @param modelId id of the model that needs to be added to modelIds set
     */
    public void add(String modelId) {
        modelIds.add(modelId);
    }

    /**
     * @return Set of modelIds in modelGraveyard
     */
    public Set<String> getModelIds() {
        return modelIds;
    }

    /**
     * @return number of modelIds in modelGraveyard
     */
    public int size() {
        return modelIds.size();
    }

    /**
     * @param modelId to check if the id of given model is there in modelIds set
     * @return true if the modelId is in the modelIds set, otherwise false
     */
    public boolean contains(String modelId) {
        return modelIds.contains(modelId);
    }

    /**
     * @param before The previous custom metadata object
     * @return the diff between the current updated object and the previous object
     */
    @Override
    public Diff<Metadata.Custom> diff(Metadata.Custom before) {
        return new ModelGraveyardDiff((ModelGraveyard) before, this);
    }

    /**
     * @param streamInput input stream
     * @return ModelGraveyardDiff
     * @throws IOException if read from stream fails
     */
    public static NamedDiff readDiffFrom(StreamInput streamInput) throws IOException {
        return new ModelGraveyardDiff(streamInput);
    }

    /**
     * @param xContentParser
     * @return ModelGraveyard
     * @throws IOException
     */
    public static ModelGraveyard fromXContent(XContentParser xContentParser) throws IOException {
        return new ModelGraveyard(xContentParser.list().stream().map(Object::toString).collect(Collectors.toSet()));
    }

    /**
     * The ModelGraveyardDiff class compares the previous modelGraveyard object with the current updated modelGraveyard object
     * and returns only the diff of those 2 objects. So that, whenever there is a change in cluster state, clusterManager node only
     * sends the diff to all the data nodes instead of the full cluster state
     */
    public static class ModelGraveyardDiff implements NamedDiff<Metadata.Custom> {
        private final Set<String> added;
        private final Set<String> removed;

        /**
         * @param inp input stream
         * @throws IOException if read from stream fails
         */
        public ModelGraveyardDiff(StreamInput inp) throws IOException {
            added = Set.copyOf(inp.readStringList());
            removed = Set.copyOf(inp.readStringList());
        }

        /**
         * @param previous previous ModelGraveyard object
         * @param current current updated ModelGraveyard object
         *
         * Constructor which compares both the objects to find the entries that are newly added in current object,
         * entries that are deleted from previous object and the deleted entries count
         */
        public ModelGraveyardDiff(ModelGraveyard previous, ModelGraveyard current) {
            final Set<String> previousModelIdsSet = previous.modelIds;
            final Set<String> currentModelIdsSet = current.modelIds;
            final Set<String> added, removed;
            if (previousModelIdsSet.isEmpty()) {
                // nothing will have been removed in previous object, and all entries in current object are new
                added = new HashSet<>(currentModelIdsSet);
                removed = new HashSet<>();
            } else if (currentModelIdsSet.isEmpty()) {
                // nothing will have been added to current object, and all entries in previous object are removed
                added = new HashSet<>();
                removed = new HashSet<>(previousModelIdsSet);
            } else {
                // some entries in previous object are removed and few entries are added to current object
                removed = Sets.difference(previousModelIdsSet, currentModelIdsSet);
                added = Sets.difference(currentModelIdsSet, previousModelIdsSet);
            }
            this.added = Collections.unmodifiableSet(added);
            this.removed = Collections.unmodifiableSet(removed);
        }

        /**
         * @param previous Previous custom metadata object
         * @return ModelGraveyard object after calculating the diff
         */
        @Override
        public ModelGraveyard apply(Metadata.Custom previous) {
            final ModelGraveyard old = (ModelGraveyard) previous;
            int removedCount = removed.size();
            if (removedCount > old.size()) {
                throw new IllegalStateException(
                    "ModelGraveyardDiff cannot remove [" + removedCount + "] entries from [" + old.size() + "] modelIds."
                );
            }
            Set<String> updatedOldGraveyardSet = Sets.difference(old.modelIds, removed);
            Set<String> modelGraveyardDiffSet = new HashSet<>();
            modelGraveyardDiffSet.addAll(added);
            modelGraveyardDiffSet.addAll(updatedOldGraveyardSet);
            return new ModelGraveyard(modelGraveyardDiffSet);
        }

        public Set<String> getAdded() {
            return added;
        }

        public Set<String> getRemoved() {
            return removed;
        }

        @Override
        public String getWriteableName() {
            return TYPE;
        }

        /**
         * @param out output stream
         * @throws IOException if write to stream fails
         */
        @Override
        public void writeTo(StreamOutput out) throws IOException {
            out.writeStringCollection(added);
            out.writeStringCollection(removed);
        }
    }
}
