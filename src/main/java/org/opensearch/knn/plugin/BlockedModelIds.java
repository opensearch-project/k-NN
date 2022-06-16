/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin;

import org.apache.logging.log4j.LogManager;
import org.opensearch.Version;
import org.opensearch.cluster.Diff;
import org.opensearch.cluster.NamedDiff;
import org.opensearch.cluster.metadata.Metadata;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentParser;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.stream.Collectors;

public class BlockedModelIds implements Metadata.Custom {

    public static Logger logger = LogManager.getLogger(BlockedModelIds.class);
    public static final String TYPE = "opensearch-knn-blocked-models";

    List<String> blockedModelIds;

    public BlockedModelIds(List<String> blockedModelIds) {
        this.blockedModelIds = blockedModelIds;
    }

    @Override
    public EnumSet<Metadata.XContentContext> context() {
        return Metadata.ALL_CONTEXTS;
    }

    @Override
    public Diff<Metadata.Custom> diff(Metadata.Custom custom) {
        return null;
    }

    @Override
    public String getWriteableName() {
        return TYPE;
    }

    @Override
    public Version getMinimalSupportedVersion() {
        return Version.CURRENT.minimumCompatibilityVersion();
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeVInt(blockedModelIds.size());
        for (String modelId : blockedModelIds) {
            out.writeString(modelId);
        }
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        return null;
    }

    public BlockedModelIds(StreamInput in) throws IOException {
        int size = in.readVInt();
        List<String> modelIds = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            String modelId = in.readString();
            modelIds.add(modelId);
        }
        this.blockedModelIds = modelIds;
    }

    public List<String> getBlockedModelIds() {
        return blockedModelIds;
    }

    public void remove(String modelId) {
        if (blockedModelIds.contains(modelId)) {
            blockedModelIds.remove(modelId);
        }
    }

    public void add(String modelId) {
        blockedModelIds.add(modelId);
    }

    public int size() {
        return blockedModelIds.size();
    }

    public static NamedDiff readDiffFrom(StreamInput streamInput) throws IOException {
        return new BlockedModelIdsDiff(streamInput);
    }

    public static BlockedModelIds fromXContent(XContentParser xContentParser) throws IOException {
        List<String> modelIds = xContentParser.list().stream().map(obj -> obj.toString()).collect(Collectors.toList());
        return new BlockedModelIds(modelIds);
    }

    public boolean contains(String modelId) {
        return blockedModelIds.contains(modelId);
    }

    static class BlockedModelIdsDiff implements NamedDiff<Metadata.Custom> {
        private List<String> added;
        private int removedCount;

        public BlockedModelIdsDiff(StreamInput inp) throws IOException {
            added = inp.readList((streamInput -> streamInput.toString()));
            removedCount = inp.readVInt();
        }

        @Override
        public Metadata.Custom apply(Metadata.Custom custom) {
            return null;
        }

        @Override
        public String getWriteableName() {
            return TYPE;
        }

        @Override
        public void writeTo(StreamOutput out) throws IOException {
            out.writeVInt(removedCount);
            for (String modelId : added) {
                out.writeString(modelId);
            }
        }
    }
}
