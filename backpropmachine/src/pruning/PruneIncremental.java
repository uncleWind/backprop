package pruning;

import java.util.ArrayList;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ListMultimap;

import feedforward.Layer;
import feedforward.Network;

public class PruneIncremental extends Pruner {
	private ListMultimap<Integer, Double> rankEfficiency;
	private double efficiency;
	private boolean continuePruning = true;
	private int maxLayerSize;
	
	final private int repeats_ = 5;
	
	public PruneIncremental(ArrayList<double[]> inputs, 
			ArrayList<double[]> outputs, Network network) {
		this.inputs = inputs;
		this.outputs = outputs;
		this.prunedNetwork = network;
		
		this.rankEfficiency = ArrayListMultimap.create();
	}
	
	/**
	 * Takes existing network and changes <em>first</em> hidden layer size until
	 * the most efficient solution is found. Algorithm starts from initial
	 * size of one and tries to find optimal layer size up to value of twice
	 * the size of input layer.
	 * <p>Network is fed with learning set five times (subjective settlement
	 * between efficiency and accuracy) to measure its efficiency.
	 * Efficiency of the network is measured by abstract value, taking average of 
	 * epochs, taking its inversion and multiplying it by 10^6 (for readability).
	 * <p>As a result of that, network is automatically reset and all
	 * progress made within one will be lost when method is first called. 
	 * Successful attempt will set hidden layers' size to optimal value 
	 * and train it once with fed learning set.
	 */
	@Override
	public void prune() {
		getMaximalLayerSize();
		for(int size = 1; size <= maxLayerSize; ++size) {
			findEfficiencyForSize(size);
			checkPruningBreakConditionForSize(size);
			
			if(!continuePruning) {
				break;
			} 
			else {
				rankEfficiency.removeAll(size - 2);
			}
		}
	}
	
	private void getMaximalLayerSize() {
		Layer inputLayer = prunedNetwork.getInputLayer();
		maxLayerSize = inputLayer.size() * 2;
	}
	
	private void findEfficiencyForSize(int size) {
		prunedNetwork.changeLayerSize(1, size);
		efficiency = 0.0f;
		
		for(int i = 0; i < repeats_; ++i) {
			prunedNetwork.trainWith(inputs, outputs);
			int epochs = prunedNetwork.getPresentLearningEpoch();
			efficiency += epochs;
			
			prunedNetwork.resetNetwork();
		}
		
		this.efficiency = (1 / (this.efficiency / 5)) * 1000000;
		rankEfficiency.put(size, efficiency);
	}
	
	private void checkPruningBreakConditionForSize(int size) {
		if(((Integer)rankEfficiency.size()).equals(3)) {
			System.out.println("Checking values *pruning* @ no. " + (size - 2));
			double efficiencyRankNil = rankEfficiency.get(size - 2).get(0);
			double efficiencyRankOne = rankEfficiency.get(size - 1).get(0);
			double efficiencyRankTwo = rankEfficiency.get(size).get(0);
			
			System.out.println((size - 2) + ":" + efficiencyRankNil
					+ "; " + (size - 1) + ":" + efficiencyRankOne
					+ ";" + (size) + ":" + efficiencyRankTwo);
			if(efficiencyRankNil > efficiencyRankOne
					&& efficiencyRankNil > efficiencyRankTwo) {
				prunedNetwork.changeLayerSize(1, size - 2);
				prunedNetwork.trainWith(inputs, outputs);
				continuePruning = false;
			}
		}
	}
}
