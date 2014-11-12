package training;

import matrixutil.*;
import feedforward.*;
import validation.*;

import java.util.*;

public class Gradient 
{
    private Network trained;
    private double learningRate;
     
    public void setNetwork(Network network) {
        trained = network;
    }
     
    public void setLearningRate(double rate) {
        Validator.validateLearningRateValue(rate);
        learningRate = rate;
    }
     
    public Gradient(Network network, double rate) {
        trained = network;
        learningRate = rate;
    }
 
    public ArrayList<Matrix> getGradients(double[] input, double[] ideal) {
        double[] result = getOutputFromInputLearningPattern(input);
        double[] errors = getError(ideal, result);
        ArrayList<double[]> layerDeltas = getLayerDeltas(errors);
        ArrayList<Matrix> setGradients = getSetGradients(layerDeltas);
         
        return setGradients;
    }
     
    private double[] getOutputFromInputLearningPattern(double[] input) { 	
    	trained.presentPattern(input);
        return trained.recieveOutput();
    }
     
    private double[] getError(double[] ideal, double[] actual) {
        Validator.validateSizeEqualityOf(ideal, actual);
         
        int length = ideal.length;
        double[] errors = new double[length];
         
        for(int i = 0; i < length; ++i) {
            errors[i] = actual[i] - ideal[i];
        }
        
        return errors;
    }
     
    private ArrayList<double[]> getLayerDeltas(double[] errors) {
        int processedLayersCount = trained.size() - 1;
        ArrayList<double[]> layerDeltas = new ArrayList<>(processedLayersCount);
         
        int processedLayerIndex = processedLayersCount;
        for(; processedLayerIndex >= 1; --processedLayerIndex) {
            Layer processedLayer = trained.getLayer(processedLayerIndex);
            double[] processedLayerDeltas;
            
            if(processedLayer.getLayerType() == LayerType.OUTPUT) {
                processedLayerDeltas = getOutputLayerDelta(processedLayer, errors);
                layerDeltas.add(0, processedLayerDeltas);
            }
            else if(processedLayer.getLayerType() == LayerType.HIDDEN) {
                double[] previousDeltas = layerDeltas.get(0);
                processedLayerDeltas = getHiddenLayerDeltas(processedLayer,
                		previousDeltas);
                layerDeltas.add(0, processedLayerDeltas);
            }
        }
         
        return layerDeltas;
    }
     
    private double[] getOutputLayerDelta(Layer layer, double[] error) {
        int length = layer.size();
        double[] resultDeltas = new double[length];
        double[] layerDerivatives = layer.getDerivativeValueNeurons();
         
        for(int i = 0; i < length; ++i) {
            //resultDeltas[i] = -error[i] * layer.getDerivativeNeuronValueAt(i);
        	resultDeltas[i] = -error[i] * layerDerivatives[i];
        }
         
        return resultDeltas;
    }
     
    private double[] getHiddenLayerDeltas(Layer layer, double[] previousDeltas) {
        int length = layer.size();
        
        double[] resultDeltas = new double[length];
        double[] layerDerivatives = layer.getDerivativeValueNeurons();
         
        for(int i = 0; i < length; ++i) {
            //double sumDerivative = layer.getDerivativeNeuronValueAt(i);
        	double sumDerivative = layerDerivatives[i];
             
            double[] inputWeights = layer.getWeightMatrix().getRowAsMatrix(i)
                    .toPackedArray();
            double previousDeltasFactor = 0.0f;
            for(int j = 0; j < inputWeights.length; ++j) {
                previousDeltasFactor += previousDeltas[j] * inputWeights[j];
            }
             
            double delta = sumDerivative * previousDeltasFactor;
            resultDeltas[i] = delta;
        }
         
        return resultDeltas;
    }
     
    private ArrayList<Matrix> getSetGradients(ArrayList<double[]> layerDeltas) {
        int processedLayersCount = trained.size() - 1;
        ArrayList<Matrix> result = new ArrayList<>(processedLayersCount);
         
        int processedLayerIndex = processedLayersCount;
        for(; processedLayerIndex >= 1; --processedLayerIndex)
        {
            Layer processedLayer = trained.getLayer(processedLayerIndex);
            double[] processedLayerDeltas = 
                    layerDeltas.get(processedLayerIndex - 1);
            Matrix layerGradients = getIndividualGradients(processedLayer, 
                    processedLayerDeltas);
             
            result.add(0, layerGradients);
        }
         
        return result;
    }
     
    private Matrix getIndividualGradients(Layer layer, 
             double[] layerDeltas) {
        Layer previousLayer = layer.getPreviousLayer();
         
        int rows = layer.size();
        int columns = previousLayer.absoluteSize();
        double[][] gradients = new double[rows][columns];
         
        for(int i = 0; i < rows; ++i) {
            for(int j = 0; j < columns; ++j) {
                double previousNeuronValue = previousLayer.getLayerType()
                		.equals(LayerType.INPUT)
                		? previousLayer.getRawNeuronValueAt(j)
                		: previousLayer.getOutputNeuronValueAt(j);
            	
            	gradients[i][j] = layerDeltas[i] * previousNeuronValue
                        * learningRate;
            }
        }
         
        Matrix result = new Matrix(gradients);
         
        /*Transposing the Matrix is done due to the "transposed" nature of
         *weight matrix in the corresponding previous layer.
         */
        result = MatrixMath.transpose(result);
        return result;
    }
}