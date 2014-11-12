package feedforward;

import java.util.*;

import error.*;
import validation.*;
import weightinit.*;
import activation.*;
import training.*;
import logger.*;
import matrixutil.Matrix;

public class Network 
{
    private ArrayList<Layer> layers;
    private Trainer trainer;
    private ErrorCalculation errorCalc;
    private int epochLimit;
    private int epochIncrement;
    private int learningEpoch;
    private double targetError;
     
    public int size()
    {
        return layers.size();
    }
     
    public Layer getLayer(int index)
    {
        Validator.validateIndex(this, index);
        return layers.get(index);
    }
    
    public int getPresentLearningEpoch() {
    	return this.learningEpoch;
    }
    
    public void setOneTimeEpochLimit(int epochs) {
    	if(Integer.compare(epochs, 0) > 0) {
    		this.epochIncrement = epochs;
    	}
    }
    
    public int getHiddenLayerCount() {
    	int count = 0;
    	
    	for(int i = 0; i < this.size(); ++i) {
    		LayerType layerType = this.getLayer(i).getLayerType();
    		if(layerType.equals(LayerType.HIDDEN)) {
    			++count;
    		}
    	}
    	
    	return count;
    }
    
    public Network(Trainer trainer, double learningRate, double momentum)
    {
        layers = new ArrayList<>();
        this.trainer = trainer;
        
        setTrainerParameters(learningRate, momentum);
        
        this.epochLimit = 0;
        this.learningEpoch = 0;
    }
     
    public Network(Trainer trainer, double learningRate, double momentum,
    		ErrorCalculation errorCalc, double targetError) {
    	layers = new ArrayList<>();
    	
    	this.trainer = trainer;
    	setTrainerParameters(learningRate, momentum);
    	
    	this.errorCalc = errorCalc;
    	this.targetError = targetError;
    	
    	this.epochLimit = 0;
    	this.learningEpoch = 0;
    }
    
    private void setTrainerParameters(double learningRate, double momentum) {
        trainer.setLearningRate(learningRate);
        trainer.setMomentum(momentum);
        trainer.setNetwork(this);
    }
    
    /**
     * Ensures that network have randomized values in span off all
     * weight matrices. Also resets values for epoch limit and actual
     * learning epoch number.
     */
    public void resetNetwork()
    {
        for(int i = 0; i < layers.size() - 1; ++i)
        {
            layers.get(i).reset();
        }
        this.epochLimit = 0;
        this.learningEpoch = 0;
    }
    
    //TODO: extracting lower abstraction levels
    public void trainWith(ArrayList<double[]> inputs, ArrayList<double[]> outputs)
    {	
    	Logger.writeComment("Weight matrices before learning: ");
    	this.logWeightMatrices();
    	trainer.resetToFirstEpoch();
    	
    	epochLimit += epochIncrement;
    	++learningEpoch;
    	
    	for(; learningEpoch <= epochLimit; ++learningEpoch) {
    		Logger.addIndent();
    		Logger.registerEpochInfo(learningEpoch);
    		
    		trainer.learnWith(inputs, outputs);
    		
    		this.logWeightMatrices();
    		for(int i = 0; i < inputs.size(); ++i) {
    			this.presentPattern(inputs.get(i));
    			double[] outputPattern = this.recieveOutput();
    			Logger.writeDoubleArray(outputPattern, "Output for input "
    				+ "pattern " + i + ".");
    		}
    		
        	errorCalc.reset();
        	for(int i = 0; i < inputs.size(); ++i) {
        		this.presentPattern(inputs.get(i));
        		double[] output = this.recieveOutput();
        		errorCalc.update(output, outputs.get(i));
        	}
        	double epochError = errorCalc.get();
        	Logger.writeErrorInfo(epochError, this.targetError);
        	
        	if(epochError <= targetError) {
        		Logger.writeComment("Target error has been reached.");
        		break;
        	}
    	}
    	
    	if(learningEpoch != epochLimit) {
    		epochLimit = learningEpoch;
    	}
    }
    
    private void logWeightMatrices() {
    	for(int i = 0; i < this.size() - 1; ++i) {
    		Logger.writeMatrixAll(this.getLayer(i).getWeightMatrix(),
    			"Weight matrix at layer " + i, MatrixLoggerType.LAYER_WEIGHTS, i);
    	}
    }
    
    public double receiveErrorFor(ArrayList<double[]> inputs, 
    		ArrayList<double[]> outputs) {
    	double result = 0.0f;
    	
    	errorCalc.reset();
    	for(int i = 0; i < inputs.size(); ++i) {
    		this.presentPattern(inputs.get(i));
    		double[] output = this.recieveOutput();
    		errorCalc.update(output, outputs.get(i));
    	}
    	
    	result = errorCalc.get();
    	return result;
    }
    
    public void presentPattern(double[] input)
    {
        Layer inputLayer = layers.get(0);
         
        Validator.validateInputPatternSize(input, inputLayer);
        Validator.validateValuesOf(input);
         
        int patternLength = input.length;
         
        for(int i = 0; i < patternLength; ++i)
        {
            double inputNeuronValue = input[i];
            inputLayer.setNeuronValueAt(i, inputNeuronValue);
        }
         
        int eligibleLayersCount = layers.size() - 1;
         
        for(int i = 1; i <= eligibleLayersCount; ++i)
        {
            Layer processedLayer = layers.get(i);
            processedLayer.calculateOutputs();
        }
    }
    
    /**
     * This method returns output layer neuron values, multiplying them
     * if needed (as in if multipliers for selected neurons were set during
     * network creation). Compare with {@link #recieveOutput()}.
     */
    public double[] recieveDecoratedOutput()
    {
        Layer outputLayer = this.getOutputLayer();
        double[] output = outputLayer.getMultipliedOutputValueNeurons();
         
        return output;
    }
    
    /**
     * This method returns output layer neuron values in standard manner,
     * as in processed by a standard feed forward algorithm, without any
     * modification on activation function level. Compare with {@link
     * #recieveDecoratedOutput()}.
     */
    public double[] recieveOutput() {
    	Layer outputLayer = this.getOutputLayer();
        double[] output = outputLayer.getOutputValueNeurons();
         
        return output;
    }
    
    public void addLayer(int size, WeightSetter setter, 
            ActivationFunction function, boolean asBias)
    {
        Layer newLayer = new Layer(size, setter, function, asBias);
        newLayer.setNetwork(this);
        newLayer.getWeightSetter().setLayer(newLayer);
         
        adjustNetworkNewLayer(newLayer);
    }
        
    /**
     * Resizes layer at given position, including resizing and resetting
     * (compare to {@link #resetNetwork()}, but on selected layer level)
     * weight matrices for this and previous layers.
     * <p>This method is to be used only with hidden layers.
     * <p>Primary objective for using is method is to supplement pruning algorithm
     * with needed operation. It is <em>strongly discouraged</em> to use this
     * method, as there isn't any known reason to adjust layer size besides
     * network pruning. If it is found that it's necessary for whatever reason,
     * one has to be wary that it can have negative impact on network performance.
     */
    public void changeLayerSize(int index, int size) {
    	Layer editedLayer = this.getLayer(index);
    	Validator.validateOperationLegalOnlyForHiddenLayer(editedLayer);
    	
    	WeightSetter weightSetter = editedLayer.getWeightSetter();
    	ActivationFunction activationFunc = editedLayer.getActivationFunction();

    	layers.remove(index);
    	addLayer(size, weightSetter, activationFunc, true);
    	
    	resetNetwork();
    }
     
    private void adjustNetworkNewLayer(Layer newLayer)
    {
        if(layers.isEmpty())
        {
            setInputLayer(newLayer);
        }
        else if(layers.size() == 1)
        {
            setOutputLayer(newLayer);
        }
        else if(layers.size() > 1)
        {
            setHiddenLayer(newLayer);
        }
    }
     
    private void setInputLayer(Layer newLayer)
    {   
        layers.add(newLayer);
        newLayer.setLayerType(LayerType.INPUT);
         
        newLayer.setPreviousLayer(null);
        newLayer.setNextLayer(null);
    }
     
    private void setOutputLayer(Layer newLayer)
    {
        layers.add(newLayer);
        newLayer.setLayerType(LayerType.OUTPUT);
         
        Layer inputLayer = this.getInputLayer();
         
        newLayer.setPreviousLayer(inputLayer);
        inputLayer.setNextLayer(newLayer);
    }
     
    private void setHiddenLayer(Layer newLayer)
    {
        int indexOfNewLayer = layers.size() - 1;
        layers.add(indexOfNewLayer, newLayer);
        newLayer.setLayerType(LayerType.HIDDEN);
         
        int indexOfPreviousLayer = layers.size() - 3;
         
        Layer previousLayer = layers.get(indexOfPreviousLayer);
        Layer nextLayer = this.getOutputLayer();
         
        newLayer.setPreviousLayer(previousLayer);
        newLayer.setNextLayer(nextLayer);
         
        previousLayer.setNextLayer(newLayer);
        nextLayer.setPreviousLayer(newLayer);
         
        previousLayer.setNewWeightMatrixSize();
        newLayer.setNewWeightMatrixSize();
    }
    
    public void setParsedWeightMatricesSet(ArrayList<Matrix> weightMatrices) {
    	Validator.validateNetworkMinimalLayerRequirement(this);
    	
    	if(weightMatrices.size() != this.size() - 1) {
    		System.out.println("Remainder: there is place for "
    				+ (this.size() - 1) + "weight matrices in network when "
    				+ "there are " + weightMatrices.size() + "entries.");
    		System.out.println("Overriding aborted.");
    	}
    	else {
    		for(int i = 0; i < weightMatrices.size(); ++i) {
    			Matrix replacmentMatrix = weightMatrices.get(i);
    			this.getLayer(i).setWeightMatrix(replacmentMatrix);
    		}
    	}
    }
    
    public Layer getInputLayer()
    {
        return layers.get(0);
    }
     
    public Layer getOutputLayer()
    {
        return layers.get(layers.size() - 1);
    }
    
    /**
     * This method should be used <em>only</em> for XML network representation
     * purposes.
     */
    public String getStringifiedNetworkType() {
		if(trainer instanceof Batch) {
			return "Batch";
		}
		else if(trainer instanceof Online) {
			return "Online";
		}
	
		throw new IllegalAccessError("Trying to access network with "
				+ "uninitialized or incompatible type of Trainer.");
    }
    
    /**
     * This method should be used <em>only</em> for XML network representation
     * purposes.
     */
    public String getStringifiedLearningRate() {
    	return this.trainer.getStringifiedLearningRateValue();
    }
    
    /**
     * This method should be used <em>only</em> for XML network representation
     * purposes.
     */
    public String getStringifiedMomentum() {
    	return this.trainer.getStringifiedMomentumValue();
    }
    
    /**
     * This method should be used <em>only</em> for XML network representation
     * purposes.
     */
    public String getStringifiedErrorCalcMethod() {
    	if(errorCalc instanceof MeanSquare) {
			return "MeanSquare";
		}
		else if(errorCalc instanceof RootMeanSquare) {
			return "RootMeanSquare";
		}
		else if(errorCalc instanceof SumOfSquares) {
			return "SumOfSquares";
		}
	
		throw new IllegalAccessError("Trying to access network with "
				+ "uninitialized or incompatible type of Trainer.");
    }
    
    /**
     * This method should be used <em>only</em> for XML network representation
     * purposes.
     */
    public String getStringifiedTargetError() {
    	return String.valueOf(this.targetError);
    }
    
    /**
     * This method should be used <em>only</em> for XML network representation
     * purposes.
     */
    public String getStringifiedBiasInfo() {
    	boolean isBiased = this.getInputLayer().absoluteSize()
    					   != this.getInputLayer().size();
    	return isBiased
    			? "True"
    			: "False";
    }
}