package training;

import validation.Validator;
import logger.Logger;
import logger.MatrixLoggerType;
import matrixutil.*;

import java.util.ArrayList;
/* TODO: Implement threaded version of batch training.
 */
public class Batch extends Trainer { 
    private ArrayList<Matrix> gradients;
    private ArrayList<Matrix> previousWeightChanges;
	
	public Batch() {
		previousWeightChanges = new ArrayList<Matrix>();
    }
    
    @Override
    public void learnWith(ArrayList<double[]> inputs, ArrayList<double[]> outputs) {
        Validator.validateLearningSetCorrectness(inputs, outputs);
        
        int networksInterlayerConnections = this.trained.size() - 1;
        checkPreviousWeightChangesArray(networksInterlayerConnections);
        
        gradients = calculateGradients(inputs, outputs);
        
        Logger.writeComment("Gradient values: ");
        for(int i = 0; i < gradients.size(); ++i) {
        	Matrix gradientMatrix = gradients.get(i);
        	Logger.writeMatrixAll(gradientMatrix, 
        		"Gradient matrix at layer " + i, 
        		MatrixLoggerType.LAYER_GRADIENTS, i);
        }
         
        //applyGradientsToNetwork(gradients);
        applyChangesToNetwork();
    }
    
    private void checkPreviousWeightChangesArray
    		(int networksInterlayerConnections) {
    	if(isFirstEpoch && !previousWeightChanges.isEmpty()) {
    		previousWeightChanges.clear();
    	}
    	if(previousWeightChanges.isEmpty()) {
    		previousWeightChanges = new ArrayList<>(networksInterlayerConnections);
    	}
    }
    
    private ArrayList<Matrix> calculateGradients(ArrayList<double[]> inputs,
    		ArrayList<double[]> outputs) {
    	ArrayList<Matrix> gradients = new ArrayList<>(trained.size() - 1);
        int epochLength = inputs.size();
         
        Gradient gradientGetter = new Gradient(trained, learningRate);
        for(int i = 0; i < epochLength; ++i)
        {
            if(i == 0) {
                gradients = gradientGetter.getGradients(inputs.get(i), outputs.get(i));
            }
            else {
                ArrayList<Matrix> newGradients = gradientGetter.getGradients(inputs.get(i), 
                        outputs.get(i));
                applyNewGradientsToTotal(newGradients, gradients);
            }
        }
        
        return gradients;
    }
     
    private void applyNewGradientsToTotal(ArrayList<Matrix> newGradients, 
            ArrayList<Matrix> batchedGradients)
    {
        int setSize = newGradients.size();
        for(int i = 0; i < setSize; ++i) {
            Matrix newGradientValues = MatrixMath.add(newGradients.get(i),
                    batchedGradients.get(i));
            batchedGradients.set(i, newGradientValues);
        }
    }
    
    private void applyChangesToNetwork() {
    	int size = gradients.size();
        
        for(int i = 0; i < size; ++i) {
            Matrix weightMatrix = trained.getLayer(i).getWeightMatrix();
            Matrix layerGradients = gradients.get(i);
             
            if(isFirstEpoch) {
            	previousWeightChanges.add(new Matrix(weightMatrix.getRows(),
            			weightMatrix.getColumns()));
            }
            Matrix layersPreviousWeightChanges = previousWeightChanges.get(i);
            
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
        if(isFirstEpoch) {
        	isFirstEpoch = false;
        }
    }
}
