package training;

import validation.Validator;
import logger.Logger;
import logger.MatrixLoggerType;
import matrixutil.*;

import java.util.ArrayList;
import java.util.Collection;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ListMultimap;

public class Online extends Trainer {
    private ArrayList<Matrix> gradients;
	ListMultimap<String, Matrix> previousWeightChanges;
	ListMultimap<String, String> indexedInputs;
	
	public Online() {
    	previousWeightChanges = ArrayListMultimap.create();
    	indexedInputs = ArrayListMultimap.create();
    }
    
    @Override
    public void learnWith(ArrayList<double[]> inputs, ArrayList<double[]> outputs) {
        Validator.validateLearningSetCorrectness(inputs, outputs);
         
        if(isFirstEpoch) {
        	multimapInputs(inputs);
        }
        checkPreviousWeightChangesMultimap();
        
        gradients = new ArrayList<>(trained.size() - 1);
        int epochLength = inputs.size();
         
        Gradient gradientGetter = new Gradient(trained, learningRate);
        for(int i = 0; i < epochLength; ++i)
        {
            gradients = gradientGetter.getGradients(inputs.get(i), outputs.get(i));
            
            Logger.writeComment("Gradient values: ");
            for(int j = 0; j < gradients.size(); ++j) {
            	Matrix gradientMatrix = gradients.get(j);
            	Logger.writeMatrixAll(gradientMatrix, 
            		"Gradient matrix at layer " + j, MatrixLoggerType.LAYER_GRADIENTS, i);
            }
            
            applyChangesToNetwork(inputs.get(i));
        }
        if(isFirstEpoch) {
        	isFirstEpoch = false;
        }
    }
    
    private void multimapInputs(ArrayList<double[]> inputs) {
    	for(int i = 0; i < inputs.size(); ++i) {
    		String stringifiedInput = stringifyInput(inputs.get(i));
    		indexedInputs.put(stringifiedInput, Integer.toString(i));
    	}
    	for(String key : indexedInputs.keySet()) {
    		Collection<String> value = indexedInputs.get(key);
    		System.out.println("Indexes: " + key + ", " + value);
    	}
    }
    
    private String stringifyInput(double[] input) {
    	StringBuffer stringifiedInput = new StringBuffer();
	   	for(int i = 0; i < input.length; ++i) {
	   		stringifiedInput.append(input[i] + ";");
	   	}
	   	
	   	return stringifiedInput.toString();
    }
    
    private void checkPreviousWeightChangesMultimap() {
    	if(isFirstEpoch && !previousWeightChanges.isEmpty()) {
    		previousWeightChanges.clear();
    	}
    	if(previousWeightChanges.isEmpty()) {
    		previousWeightChanges = ArrayListMultimap.create();
    	}
    }
    
    private void applyChangesToNetwork(double[] inputs) {
    	int size = gradients.size();
        
        for(int i = 0; i < size; ++i) {
            Matrix weightMatrix = trained.getLayer(i).getWeightMatrix();
            Matrix layerGradients = gradients.get(i);
            
            String multimapIndex = stringifyInput(inputs);
            String prevChangesIndex = indexedInputs.get(multimapIndex).get(0);
            
            if(isFirstEpoch) {
            	previousWeightChanges.put(prevChangesIndex, new Matrix(weightMatrix
            			.getRows(), weightMatrix.getColumns()));
            }
            Matrix layersPreviousWeightChanges = previousWeightChanges
            		.get(prevChangesIndex).get(i);
            
            int rows = weightMatrix.getRows();
            int columns = weightMatrix.getColumns();
            for(int j = 0; j < rows; ++j) {
                for(int k = 0; k < columns; ++k) {
                    double weightChange = layerGradients.getCell(j, k);
                    if(!isFirstEpoch) {
                    	weightChange += layersPreviousWeightChanges.getCell(j, k)
                    			* momentum;
                    }
                    layersPreviousWeightChanges.setCell(j, k, weightChange);
                    
                	double newWeightValue = weightMatrix.getCell(j, k) 
                			+ weightChange;
                    
                    weightMatrix.setCell(j, k, newWeightValue);
                }
            }
            Logger.writeMatrixAll(layersPreviousWeightChanges, 
            		"Weight changes at layer " + i, 
            		MatrixLoggerType.LAYER_WEIGHT_CHANGES, i);
        }
    }
}