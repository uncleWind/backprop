package weightinit;

import feedforward.Layer;
import feedforward.LayerType;
import matrixutil.*;

public class NguyenWidrow implements WeightSetter
{
    private double minBound = -0.5f;
    private double maxBound = 0.5f;
    private Layer layer;
 
    @Override
    public void setLayer(Layer layer)
    {
        this.layer = layer;
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
         
        if(layer.getLayerType() == LayerType.INPUT)
        {
            processNguyenWidrow();
        }
    }
     
    private double generateValue()
    {
        double result = minBound + Math.random() * (maxBound - minBound);
        return result;
    }
     
    private void processNguyenWidrow()
    {
        double inputLayerSize = layer.size();
        double hiddenLayerSize = layer.getNextLayer().size();
        Matrix weightMatrix = layer.getWeightMatrix();
         
        double betaFactor = Math.pow(0.7f * hiddenLayerSize, 1 / inputLayerSize);
         
        int rows = weightMatrix.getRows();
        int columns = weightMatrix.getColumns();
         
        for(int j = 0; j < columns; ++j)
        {
            double euclideanNorm = 0.0f;
             
            for(int i = 0; i < rows; ++i)
            {
                euclideanNorm += Math.pow(weightMatrix.getCell(i, j), 2);
            }
            euclideanNorm = Math.sqrt(euclideanNorm);
             
            for(int i = 0; i < rows; ++i)
            {
                double value = betaFactor * weightMatrix.getCell(i, j)
                        / euclideanNorm;
                weightMatrix.setCell(i, j, value);
            }
        }
    }
     
}