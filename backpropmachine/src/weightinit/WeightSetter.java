package weightinit;

import feedforward.Layer;

public interface WeightSetter {
    void randomizeWeightMatrix();
    void setLayer(Layer layer);
}