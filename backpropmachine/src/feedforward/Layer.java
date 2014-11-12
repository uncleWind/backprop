package feedforward;
 
import validation.Validator;
import matrixutil.*;
import activation.*;
import weightinit.*;

public class Layer 
{
    private NeuronArray neurons;
    private Matrix weightMatrix;
    private Matrix previousWeightChanges;
     
    private Network network;
    private Layer previousLayer;
    private Layer nextLayer;
     
    private LayerType type;
    private ActivationFunction activation;
    private WeightSetter setter;
     
    public double[] getRawValueNeurons() {
        return neurons.returnAsRawDoubleAbsoluteArray();
    }
    
    public double[] getOutputValueNeurons() {
    	Validator.validateIllegalOperationForInputLayer(this);
    	
    	return neurons.returnAsOutputDoubleAbsoluteArray();
    }
    
    public double[] getMultipliedOutputValueNeurons() {
    	Validator.validateOperationLegalOnlyForOutputLayer(this);
    	
    	return neurons.returnAsMultipliedOutputDoubleArray();
    }
    
    public double[] getDerivativeValueNeurons() {
    	Validator.validateIllegalOperationForInputLayer(this);
    	
    	return neurons.returnAsDerivativeDoubleAbsoluteArray();
    }
     
    public void setNeuronValueAt(int index, double value) {
        neurons.setNeuronValue(index, value);
    }
     
    public double getRawNeuronValueAt(int index) {
        return neurons.getRawNeuronValue(index);
    }
     
    public double getOutputNeuronValueAt(int index) {
    	return neurons.getOutputNeuronValue(index);
    }
    
    public double getDerivativeNeuronValueAt(int index) {
    	return neurons.getDerivativeNeuronValue(index);
    }
    
    public void setNeuronMultiplierValueAt(int index, double multiplier) {
    	neurons.setNeuronMultiplierValue(index, multiplier);
    }
    
    public void removeNeuronAt(int index) {
    	neurons.removeNeuron(index);
    }
    
    public Matrix getWeightMatrix() {
        return weightMatrix;
    }
    
    public void setWeightMatrix(Matrix weightMatrix) {
    	Validator.validateRowsAndColumnsEquality(this.weightMatrix, weightMatrix);
    	this.weightMatrix = weightMatrix;
    }
    
    /**
     * This method forces new weight matrix to be put in the network instance
     * without size check. This method is used only for selective pruning
     * algorithm and should -not be used- for any other purpose.
     */
    public void forceSetWeightMatrix(Matrix weightMatrix) {
    	this.weightMatrix = weightMatrix;
    }
    
    public Matrix getPreviousWeightChangesMatrix() {
    	return previousWeightChanges;
    }
     
    public void setNewWeightMatrixSize() {
        int rows = this.absoluteSize();
        int columns = nextLayer.size();
         
        weightMatrix = new Matrix(rows, columns);
        previousWeightChanges = new Matrix(rows, columns);
    }
     
    public void setNetwork(Network network) {
        this.network = network;
    }
    
    public Network getNetwork() {
    	return this.network;
    }
     
    public Layer getPreviousLayer() {
        return previousLayer;
    }
     
    public void setPreviousLayer(Layer previous) {
        previousLayer = previous;
    }
     
    public Layer getNextLayer() {
        return nextLayer;
    }
     
    public void setNextLayer(Layer next) {
        nextLayer = next;
    }
     
    public LayerType getLayerType() {
        return type;
    }
     
    public void setLayerType(LayerType type) {
        this.type = type;
    }
     
    public void setActivationFunction(ActivationFunction function) {
        activation = function;
    }
     
    public ActivationFunction getActivationFunction() {
        return activation;
    }
     
    public WeightSetter getWeightSetter() {
        return setter;
    }
     
    public void setWeightInitMethod(WeightSetter method) {
        setter = method;
    }
    
    private void setAsLayerWithBiasNeuron() {
    	neurons.setBiasNeuron();
    }
    
    public int absoluteSize() {
    	return neurons.absoluteSize();
    }
    
    public int size() {
        return neurons.size();
    }
     
    public Layer(int size, WeightSetter setter, ActivationFunction function,
    		boolean asBias) {
    	System.out.println("Creating new layer, asBias: " + asBias);
    	resolveLayerSize(size, asBias);
    	this.setter = setter;
        activation = function; 
    }
    
    private void resolveLayerSize(int size, boolean asBias) {
    	int layerSize = asBias ? size + 1 : size;
    	System.out.println("Resolving size; size: " + size
    			+ ", layerSize: " + layerSize);
    	neurons = new NeuronArray(this, layerSize);
        if(asBias) {
        	setAsLayerWithBiasNeuron();
        }
    }
    
    public void reset() {
        Validator.validateIllegalOperationForOutputLayer(this);
        setter.randomizeWeightMatrix();
    }
     
    public void calculateOutputs() {
        Validator.validateIllegalOperationForInputLayer(this);
         
        Matrix previousWeightMatrix = previousLayer.getWeightMatrix();
         
        int rows = previousWeightMatrix.getRows();
        int columns = previousWeightMatrix.getColumns();
  
        for(int j = 0; j < columns; ++j) {
            double sum = 0.0f;
             
            for(int i = 0; i < rows; ++i) {
            	double previousValue = previousLayer.type.equals(LayerType.INPUT)
            			? previousLayer.getRawNeuronValueAt(i)
            			: previousLayer.getOutputNeuronValueAt(i);
            	
            	sum += previousWeightMatrix.getCell(i, j) * previousValue;
            }
            
            this.setNeuronValueAt(j, sum);
        }
    }
    
    /**
     * This method should be used -only- for XML network representation
     * purposes.
     */
    public String getStringifiedSize() {
    	return String.valueOf(this.size());
    }
    
    /**
     * This method should be used -only- for XML network representation
     * purposes.
     */
    public String getStringifiedWeightSetter() {
    	if(setter instanceof NguyenWidrow) {
    		return "NguyenWidrow";
    	}
    	else if(setter instanceof RangedRandom) {
    		return "RangedRandom";
    	}
    	
    	throw new IllegalAccessError("Trying to access layer with "
				+ "uninitialized or incompatible type of WeightSetter.");
    }
    
    /**
     * This method should be used -only- for XML network representation
     * purposes.
     */
    public String getStringifiedWeightSetterMinBound() {
    	if(setter instanceof NguyenWidrow) {
    		return "None";
    	}
    	return ((RangedRandom)setter).getStringifiedMinBound();
    }
    
    /**
     * This method should be used -only- for XML network representation
     * purposes.
     */
    public String getStringifiedWeightSetterMaxBound() {
    	if(setter instanceof NguyenWidrow) {
    		return "None";
    	}
    	return ((RangedRandom)setter).getStringifiedMaxBound();
    }
    
    /**
     * This method should be used -only- for XML network representation
     * purposes.
     */
    public String getStringifiedActivationMethod() {
    	if(activation instanceof HyperbolicTangent) {
    		return "HyperbolicTangent";
    	}
    	else if(activation instanceof Linear) {
    		return "Linear";
    	}
    	else if(activation instanceof Sigmoid) {
    		return "Sigmoid";
    	}
    	
    	throw new IllegalAccessError("Trying to access layer with "
				+ "uninitialized or incompatible type of ActivationFunction.");
    }
    
    /**
     * This method should be used -only- for XML network representation
     * purposes.
     */
    public String getStringifiedWeightMatrixRowSize() {
    	return String.valueOf(this.getWeightMatrix().getRows());
    }
    
    /**
     * This method should be used -only- for XML network representation
     * purposes.
     */
    public String getStringifiedWeightMatrixColumnSize() {
    	return String.valueOf(this.getWeightMatrix().getColumns());
    }
    
    /**
     * This method should be used -only- for XML network representation
     * purposes.
     */
    public String getStringifiedWeightMatrixRowValuesAt(int index) {
    	double[] row = this.getWeightMatrix().getRowAsMatrix(index).toPackedArray();
    	StringBuilder builder = new StringBuilder();
    	for(double value : row) {
    		builder.append(value + ";");
    	}
    	builder.setLength(builder.length() - 1);
    	
    	return builder.toString();
    }
    
    /**
     * This method should be used -only- for XML network representation
     * purposes.
     */
    public String getStringifiedNeuronMultiplierValueAt(int index) {
    	Validator.validateOperationLegalOnlyForOutputLayer(this);
    	
    	return String.valueOf(neurons.getMultiplierValue(index));
    }
}