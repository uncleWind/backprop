package validation; 

import matrixutil.Matrix; 
import feedforward.*; 
import learningset.*; 

import java.util.ArrayList; 

public class Validator { 
  
    public static void validateCoordinatesWith(Matrix matrix, int row, 
    		int column) { 
        validateRowCoordinateWith(matrix, row); 
        validateColumnCoordinateWith(matrix, column); 
    } 
      
    public static void validateRowCoordinateWith(Matrix matrix, int row) { 
        if(row >= matrix.getRows() || row < 0) { 
            throw new IllegalArgumentException("A row index is invalid."); 
        } 
    } 
      
    public static void validateColumnCoordinateWith(Matrix matrix, int column) { 
        if(column >= matrix.getColumns() || column < 0) { 
            throw new IllegalArgumentException("A column index is invalid."); 
        } 
    } 
      
    public static void validateValuesOf(double[] valuesSet) { 
        for(int i = 0; i < valuesSet.length; ++i) { 
            validateValue(valuesSet[i]); 
        } 
    } 
      
    public static void validateValue(double value) { 
        Double checkedValue = value; 
        
        if(checkedValue.isInfinite() || checkedValue.isNaN()) { 
            throw new IllegalArgumentException("Value is not a number or "
                + "it's an infinite."); 
        } 
    } 
      
    public static void validateSize(int size) { 
        if(size <= 0) { 
            throw new IllegalArgumentException("The size can't less or equal "
                + "zero."); 
        } 
    } 
      
    public static<T> void validateSizeOf(ArrayList<T> list) { 
        if(list.isEmpty()) { 
            throw new IllegalArgumentException("The list cannot be empty."); 
        } 
    } 
      
    public static void validateSizeEqualityOf(double[] firstSet,  
    		double[] secondSet) { 
        if(firstSet.length != secondSet.length) { 
            throw new IllegalArgumentException("Array lenghts aren't equal."); 
        } 
    } 
      
    public static<T> void validateSizeEqualityOf(ArrayList<T> firstSet, 
            ArrayList<T> secondSet) { 
        if(firstSet.size() != secondSet.size()) { 
            throw new IllegalArgumentException("List sizes aren't equal."); 
        } 
    }
    
    public static void validateMatrixMultiplicationSizeCondition( 
        Matrix multiplier, Matrix multiplicand) { 
        if(multiplier.getColumns() != multiplicand.getRows()) { 
            throw new IllegalArgumentException("Column number of multiplier "
                + "isn't equal to multiplicands' rows number."); 
        } 
    } 
      
    public static void validatePackedArrayCellsQuantityEqualityWith( 
            Matrix matrix, double[] packed) { 
        if(matrix.getCellsQuantity() != packed.length) { 
            throw new IllegalArgumentException("The quantity of matrix cells "
                + "doesn't equal the length of the array"); 
        } 
    } 
      
    public static void validateDotProductConditions(Matrix firstVector, 
            Matrix secondVector) { 
        if(firstVector.isNotAVector()) { 
            throw new IllegalArgumentException("The first argument is not "
                + "a vector"); 
        } 
        if(secondVector.isNotAVector()) { 
            throw new IllegalArgumentException("The second argument is not "
                + "a vector"); 
        } 
        if(firstVector.getCellsQuantity() != secondVector.getCellsQuantity()) { 
            throw new IllegalArgumentException("Vectors have different sizes"); 
        } 
    } 
      
    public static void validateRowsAndColumnsEquality(Matrix firstMatrix, 
            Matrix secondMatrix) { 
        validateRowsEquality(firstMatrix, secondMatrix); 
        validateColumnsEquality(firstMatrix, secondMatrix); 
    } 
      
    public static void validateRowsEquality(Matrix firstMatrix, 
            Matrix secondMatrix) { 
        if(firstMatrix.getRows() != secondMatrix.getRows()) { 
            throw new IllegalArgumentException("Rows of matrixes aren't "
                    + "equal."); 
        } 
    } 
      
    public static void validateColumnsEquality(Matrix firstMatrix, 
            Matrix secondMatrix) { 
        if(firstMatrix.getColumns() != secondMatrix.getColumns()) { 
            throw new IllegalArgumentException("Columns of matrixes aren't "
                    + "equal."); 
        } 
    } 
      
    public static void validateBounds(double minBound, double maxBound) { 
        if(minBound > maxBound) { 
            throw new IllegalArgumentException("Minimum bound can't have a "
                + "higher value than a maximum bound."); 
        } 
    } 
      
    public static void validateIndex(Layer layer, int index) { 
        if(index < 0) { 
            throw new IllegalArgumentException("Index can't be lower than "
                    + "zero."); 
        } 
        if(index >= layer.size()) { 
            throw new IllegalArgumentException("Index is out of higher bound."); 
        } 
    } 
      
    public static void validateIndex(Network network, int index) { 
        if(index < 0) { 
            throw new IllegalArgumentException("Index can't be lower than "
                    + "zero."); 
        } 
        if(index >= network.size()) { 
            throw new IllegalArgumentException("Index is out of higher bound."); 
        } 
    } 
    
    public static void validateIndex(NeuronArray array, int index) {
    	if(index < 0) {
    		throw new IllegalArgumentException("Index can't be lower than "
    				+ "zero.");
    	}
    	if(index >= array.absoluteSize()) {
    		throw new IllegalArgumentException("Index is out of higher bound.");
    	}
    }
    
    public static void validateIllegalOperationForInputLayer(Layer layer) { 
        if(layer.getLayerType() == LayerType.INPUT) 
        { 
            throw new IllegalArgumentException("Cannot call the method with an "
                + "Layer argument of INPUT type"); 
        } 
    } 
      
    public static void validateIllegalOperationForOutputLayer(Layer layer) { 
        if(layer.getLayerType() == LayerType.OUTPUT) { 
            throw new IllegalArgumentException("Cannot call the method with an "
                + "Layer argument of OUTPUT type"); 
        } 
    } 
    
    public static void validateOperationLegalOnlyForOutputLayer(Layer layer) {
    	if(layer.getLayerType() != LayerType.OUTPUT) {
    		throw new IllegalArgumentException("Cannot call the method with an "
    			+ "Layer argument of type other than OUTPUT.");
    	}
    }
    
    public static void validateOperationLegalOnlyForHiddenLayer(Layer layer) {
    	if(layer.getLayerType() != LayerType.HIDDEN) {
    		throw new IllegalArgumentException("Cannot call th method with an "
    				+ "Layer argument of type other than HIDDEN.");
    	}
    }
    
    public static void validateInputPatternSize(double[] pattern,  
            Layer inputLayer) 
    { 
        if(pattern.length != inputLayer.size()) 
        { 
            throw new IllegalArgumentException("Input pattern length is not " 
                + "the same as input layers' size."); 
        } 
    } 
      
    public static void validateLearningSetCorrectness(ArrayList<double[]> inputs, 
            ArrayList<double[]> outputs) 
    { 
        validateSizeOf(inputs); 
        validateSizeOf(outputs); 
        validateSizeEqualityOf(outputs, outputs); 
    } 
      
    public static void validateLearningRateValue(double learningRate) { 
        Double rate = (Double) learningRate; 
        if(rate.compareTo(0.0) == -1) { 
            throw new IllegalArgumentException("Learning rate cannot be negative."); 
        } 
        if(rate.compareTo((1.0)) == 1) { 
            throw new IllegalArgumentException("Learning rate of above one, "
                + "honestly, doesn't help anybody. It's invalid."); 
        } 
    } 
      
    public static void validateLearningInputSampleSize(double[] sample,  
            Network network) { 
        Integer sampleSize = (Integer) sample.length; 
        Integer inputSize = (Integer) network.getInputLayer().size(); 
        if(sampleSize.compareTo(inputSize) != 0) { 
            throw new IllegalArgumentException("New learning sample haven't the "
                + "size of networks' input layer size."); 
        } 
    } 
      
    public static void validateLearningOutputSampleSize(double[] sample,  
            Network network) { 
        Integer sampleSize = (Integer) sample.length; 
        Integer inputSize = (Integer) network.getOutputLayer().size(); 
        if(sampleSize.compareTo(inputSize) != 0) { 
            throw new IllegalArgumentException("New learning sample haven't the "
                + "size of networks' output layer size."); 
        } 
    } 
      
    public static void validateParsedNetworkCompletion(Network network) { 
        if(network == null) { 
            throw new IllegalArgumentException("Network hasn't been properly "
                + "parsed."); 
        } 
        else { 
            if(network.size() == 0) { 
                throw new IllegalArgumentException("Layers haven't been "
                    + "properly parsed."); 
            } 
        } 
    } 
      
    public static void validateParsedLearningSetCompletion(LearningSet learningSet) { 
        if(learningSet.getLearningInputsSet().isEmpty()) { 
            throw new IllegalArgumentException("Learning set hasn't been "
                + "properly parsed."); 
        } 
        if(learningSet.getLearningInputsSet().size() != 
                learningSet.getLearningOutputsSet().size()) { 
            throw new IllegalArgumentException("Learning set hasn't been "
                + "properly parsed."); 
        } 
    }
    
    public static void validateIllegalOperationForBiasNeuron(Neuron neuron) {
    	if(neuron.isBias()) {
    		throw new IllegalArgumentException("Cannot assign value to a"
    				+ " bias neuron.");
    	}
    }
    
    public static void validateNetworkMinimalLayerRequirement(Network network) {
    	if(network.size() < 3) {
    		throw new IllegalArgumentException("Accessed network doesn't have "
    				+ "at least 3 layers.");
    	}
    }
    
    public static void validateNetworkCopyAbility(Network network, 
    		ArrayList<Matrix> copy) {
    	if(network.size() - 1 != copy.size()) {
    		throw new IllegalArgumentException("Cannot copy stored weight "
    				+ "matrices to network.");
    	}
    }
}