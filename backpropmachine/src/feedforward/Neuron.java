package feedforward;

import validation.Validator;
import activation.ActivationFunction;

public class Neuron {
	private double value;
	private double multiplier;
	private boolean isBias;
	private NeuronArray array;
	
	public void setValue(double value) {
		Validator.validateValue(value);
		
		this.value = value;
	}
	
	public void setMultiplier(double multiplier) {
		Validator.validateValue(multiplier);
		
		this.multiplier = multiplier;
	}
	
	public void setAsBias() {
		isBias = true;
	}
	
	public boolean isBias() {
		return isBias;
	}
	
	public Neuron(NeuronArray array) {
		this.array = array;
		value = 1.0;
		multiplier = 1.0;
		isBias = false;
	}
	
	public double returnOutputValue() {
		if(isBias) {
			return returnRawValue();
		}
		
		ActivationFunction activation = array.getLayer().getActivationFunction();
		double result = activation.getValue(this.value);
		return result;
	}
	
	public double returnMultipliedOutputValue() {
		double result = returnOutputValue();
		result *= multiplier;
		return result;
	}
	
	public double returnDerivativeValue() {
		if(isBias) {
			return returnRawValue();
		}
		
		ActivationFunction activation = array.getLayer().getActivationFunction();
		double result = activation.getDerivative(this.value);
		return result;
	}
	
	public double returnRawValue() {
		return this.value;
	}
	
	/** This method should be used -only- for XML parsing purposes. Information
	 * about neuron multipliers should be provided only through XML network file.
	 * Though this information doesn't have any impact on network data processing,
	 * changing it could result in false outcome interpretation.
	 */
	public double returnMultiplierValue() {
		return this.multiplier;
	}
}
