package training;

import feedforward.*;

import java.util.ArrayList;

public abstract class Trainer {
	protected double learningRate;
	protected Network trained;
    protected double momentum;
    protected boolean isFirstEpoch;
	
    public abstract void learnWith(ArrayList<double[]> input,
    		ArrayList<double[]> output);
    
	public void setNetwork(Network network) {
		this.trained = network;
    }
	
    public void setLearningRate(double rate) {
    	this.learningRate = rate;
    }
    
    public void setMomentum(double momentum) {
    	this.momentum = momentum;
    }
    
    /**
     * Indicates that new batch of learning tuples are to be fed to network.
     * </br></br>
     * This method should be used only once each time network is to be fed with
     * new batch of learning tuples to make sure that learning algorithm will not
     * take previous weight changes for first epoch (no data had been generated
     * by that time).
     */
    public void resetToFirstEpoch() {
    	this.isFirstEpoch = true;
    }
    
    /**
     * This method should be used -only- for XML network representation
     * purposes.
     */
    public String getStringifiedLearningRateValue() {
    	return String.valueOf(this.learningRate);
    }
    
    /**
     * This method should be used -only- for XML network representation
     * purposes.
     */
    public String getStringifiedMomentumValue() {
    	return String.valueOf(this.momentum);
    }
    
    //protected abstract void applyChangesToNetwork();
}