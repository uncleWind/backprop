package weightinit;

import feedforward.Layer;
import validation.Validator;
import matrixutil.*;

public class RangedRandom implements WeightSetter {
 
    private double minBound;
    private double maxBound;
    private Layer layer;
     
    @Override
    public void setLayer(Layer layer)
    {
        this.layer = layer;
    }
     
    public void setBounds(double minBound, double maxBound)
    {
        Validator.validateBounds(minBound, maxBound);
        this.minBound = minBound;
        this.maxBound = maxBound;
    }
     
    @Override
    public void randomizeWeightMatrix() 
    {
        Matrix weightMatrix = layer.getWeightMatrix();
         
        int rows = weightMatrix.getRows();
        int columns = weightMatrix.getColumns();
         
        for(int i = 0; i < rows; ++i)
        {
            for(int j = 0; j < columns; ++j)
            {
                double value = generateValue();
                weightMatrix.setCell(i, j, value);
            }
         
        }
    }
     
    private double generateValue()
    {
        double result = minBound + Math.random() * (maxBound - minBound);
        return result;
    }
    
    /**
     * This method should be used -only- for XML network representation
     * purposes.
     */
    public String getStringifiedMinBound() {
    	return String.valueOf(minBound);
    }
    
    /**
     * This method should be used -only- for XML network representation
     * purposes.
     */
    public String getStringifiedMaxBound() {
    	return String.valueOf(maxBound);
    }
}