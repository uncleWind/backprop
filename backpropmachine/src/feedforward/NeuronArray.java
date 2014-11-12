package feedforward;

import validation.Validator;

public class NeuronArray {
	private Neuron[] neurons;
	private Layer layer;
	
	public int size() {
		int lastNeuronIndex = neurons.length - 1;
		return neurons[lastNeuronIndex].isBias()
				? neurons.length - 1
				: neurons.length;
	}
	
	/** This method is to be used with caution, only if knowledge of
	 * absolute size (including bias neurons) is <<essential>> to
	 * implemented algorithm. Usage of size() method instead is highly
	 * advised.
	 */
	public int absoluteSize() {
		return neurons.length;
	}
	
	public Layer getLayer() {
		return layer;
	}
	
	public boolean isRepresentingBiasLayer() {
		int lastNeuronIndex = neurons.length - 1;
		return neurons[lastNeuronIndex].isBias();
	}
	
	public void setNeuronValue(int index, double value) {
		Validator.validateIndex(this, index);
		
		neurons[index].setValue(value);
	}
	
	public void setNeuronMultiplierValue(int index, double multiplier) {
		Validator.validateIndex(this, index);
		
		neurons[index].setMultiplier(multiplier);
	}
	
	public NeuronArray(Layer layer, int size) {
		this.layer = layer;
		this.neurons = new Neuron[size];
		
		for(int i = 0; i < neurons.length; ++i) {
			neurons[i] = new Neuron(this);
		}
	}
	
	public void removeNeuron(int index) {
		Validator.validateIndex(this, index);
		
		Neuron[] newNeuronsInstance = new Neuron[neurons.length - 1];
		for(int i = 0, j = 0; i < neurons.length; ++i) {
			if(i != index) {
				newNeuronsInstance[j] = neurons[i];
				++j;
			}
		}
		
		this.neurons = newNeuronsInstance;
	}
	
	public void setBiasNeuron() {
		int lastNeuronIndex = neurons.length - 1;
		neurons[lastNeuronIndex].setAsBias();
	}
	
	public double getRawNeuronValue(int index) {
		Validator.validateIndex(this, index);
		
		return neurons[index].returnRawValue();
	}
	
	public double getOutputNeuronValue(int index) {
		Validator.validateIndex(this, index);
		
		return neurons[index].returnOutputValue();
	}
	
	public double getDerivativeNeuronValue(int index) {
		Validator.validateIndex(this, index);
		
		return neurons[index].returnDerivativeValue();
	}
	
	public double[] returnAsRawDoubleArray() {
		double[] result = new double[size()];
		for(int i = 0; i < result.length; ++i) {
			result[i] = neurons[i].returnRawValue();
		}
		
		return result;
	}
	
	/** This method is to be used with caution, only if knowledge of
	 * absolute size (including bias neurons) is <<essential>> to
	 * implemented algorithm. Usage of returnAsRawDoubleArray() 
	 * method instead is highly advised.
	 */
	public double[] returnAsRawDoubleAbsoluteArray() {
		double[] result = new double[absoluteSize()];
		for(int i = 0; i < result.length; ++i) {
			result[i] = neurons[i].returnRawValue();
		}
		
		return result;
	}
	
	public double[] returnAsOutputDoubleArray() {
		double[] result = new double[size()];
		for(int i = 0; i < result.length; ++i) {
			result[i] = neurons[i].returnOutputValue();
		}
		
		return result;
	}
	
	public double[] returnAsMultipliedOutputDoubleArray() {
		double[] result = new double[size()];
		for(int i = 0; i < result.length; ++i) {
			result[i] = neurons[i].returnMultipliedOutputValue();
		}
		
		return result;
	}
	
	/** This method is to be used with caution, only if knowledge of
	 * absolute size (including bias neurons) is <<essential>> to
	 * implemented algorithm. Usage of returnAsOutputDoubleArray() 
	 * method instead is highly advised.
	 */
	public double[] returnAsOutputDoubleAbsoluteArray() {
		double[] result = new double[absoluteSize()];
		for(int i = 0; i < result.length; ++i) {
			result[i] = neurons[i].returnOutputValue();
		}
		
		return result;
	}
	
	public double[] returnAsDerivativeDoubleArray() {
		double[] result = new double[size()];
		for(int i = 0; i < result.length; ++i) {
			result[i] = neurons[i].returnDerivativeValue();
		}
		
		return result;
	}
	
	/** This method is to be used with caution, only if knowledge of
	 * absolute size (including bias neurons) is <<essential>> to
	 * implemented algorithm. Usage of returnAsDerivativeDoubleArray() 
	 * method instead is highly advised.
	 */
	public double[] returnAsDerivativeDoubleAbsoluteArray() {
		double[] result = new double[absoluteSize()];
		for(int i = 0; i < result.length; ++i) {
			result[i] = neurons[i].returnDerivativeValue();
		}
		
		return result;
	}
	
	/** This method should be used -only- for XML parsing purposes. Information
	 * about neuron multipliers should be provided only through XML network file.
	 * Though this information doesn't have any impact on network data processing,
	 * changing it could result in false outcome interpretation.
	 */
	public double getMultiplierValue(int index) {
		Validator.validateIndex(this, index);
		
		return neurons[index].returnMultiplierValue();
	}
}
