package pruning;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;
import java.util.TreeSet;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ListMultimap;

import matrixutil.Matrix;
import matrixutil.MatrixMath;
import feedforward.Layer;
import feedforward.LayerType;
import feedforward.Network;

/*TODO: Changes to weight matrices for target layers' above layer are also needed.
 *Change code to reflect that.
 */

public class PruneBySelection extends Pruner {
	private ListMultimap<Integer, PruneWeightCopy> networkStateCopies;
	private ListMultimap<Integer, Double> rankEfficiency;
	private ListMultimap<Integer, Integer> rankIterator;
	private ListMultimap<Integer, Integer> rankCutNeurons;
	private boolean neuronsFound = true;
	private int rankLevel;
	
	private Layer activeLayer;
	private ErrorBound errorBound;
	
	public PruneBySelection(ArrayList<double[]> inputs, 
			ArrayList<double[]> outputs, Network network) {
		this.inputs = inputs;
		this.outputs = outputs;
		this.prunedNetwork = network;
		this.errorBound = new ErrorBound();
		
		this.networkStateCopies = ArrayListMultimap.create();
		this.rankEfficiency = ArrayListMultimap.create();
		this.rankIterator = ArrayListMultimap.create();
		this.rankCutNeurons = ArrayListMultimap.create();
		this.rankLevel = 0;
	}
	
	//TODO: finish pruning algorithm
	/**
	 * Takes existing <em>trained</em> network and attempts to remove single
	 * neurons from hidden layers. If at least one neuron was found, changes
	 * to NeuronArray and weights matrices for corresponding layers are
	 * applied in terms of shrinking.
	 */
	@Override
	public void prune() {
		storeRankState();
		storeRankEfficiency();
		
		while(neuronsFound) {
			neuronsFound = false;
			
			setFirstHiddenLayerAsActive();
			iterateActiveLayer();
		}
	}
	
	private void storeRankState() {
		PruneWeightCopy weightCopy = new PruneWeightCopy(this.prunedNetwork);
		networkStateCopies.put(this.rankLevel, weightCopy);
	}
	
	private void storeRankEfficiency() {
		double efficiency = prunedNetwork.receiveErrorFor(inputs, outputs);
		rankEfficiency.put(this.rankLevel, efficiency);
	}
	
	private void setFirstHiddenLayerAsActive() {
		Layer newActiveLayer = prunedNetwork.getInputLayer().getNextLayer();
		this.activeLayer = newActiveLayer;
		
		this.rankLevel = 1;
	}
	
	private void iterateActiveLayer() {
		for(int neuronIndex = 0; neuronIndex < activeLayer.size(); ++neuronIndex) {
			updateRankCounter(neuronIndex);
			clearNeuron(neuronIndex);
			storeRankEfficiency();
			resolveStepAdvancement();
		}
		if(rankLevel == 1) {
			pruneNetwork();
			clearAllStates();
		}
		checkIterationRegression();
	}
	
	private void updateRankCounter(int index) {
		if(rankIterator.containsKey(rankLevel)) {
			rankIterator.removeAll(rankLevel);
		}
		rankIterator.put(rankLevel, index);
	}
	
	private void clearNeuron(int index) {
		clearNeuronAtActiveLayer(index);
		clearNeuronAtPreviousLayer(index);
	}
	
	private void clearNeuronAtPreviousLayer(int index) {
		Layer previousLayer = activeLayer.getPreviousLayer();
		Matrix weightMatrix = previousLayer.getWeightMatrix();
		
		for(int i = 0; i < weightMatrix.getRows(); ++i) {
			weightMatrix.setCell(i, index, 0.0f);
		}
		
		previousLayer.setWeightMatrix(weightMatrix);
	}
	
	private void clearNeuronAtActiveLayer(int index) {
		Matrix weightMatrix = activeLayer.getWeightMatrix();
		
		for(int i = 0; i < weightMatrix.getColumns(); ++i) {
			weightMatrix.setCell(index, i, 0.0f);
		}
		
		activeLayer.setWeightMatrix(weightMatrix);
	}
	
	private void resolveStepAdvancement() {
		int activeRankEfficiencySetSize = rankEfficiency.get(rankLevel).size();
		if(activeRankEfficiencySetSize == 1) {
			resolveFirstNeuronCondition();
		}
		else if(activeRankEfficiencySetSize == 2) {
			resolveOtherNeuronCondition();
		}
	}
	
	private void resolveFirstNeuronCondition() {
		double activeEfficiency = rankEfficiency.get(rankLevel).get(0);
		double previousEfficiency = rankEfficiency.get(rankLevel - 1)
				.get(0);
		double targetEfficiency = errorBound.getErrorBound(previousEfficiency);

		if(activeEfficiency < targetEfficiency) {
			neuronsFound = true;
			storeRankState();
			storeRankNeuron();
			
			errorBound.advanceChanges();
			advanceActiveLayer();
		}
		else {
			rankEfficiency.removeAll(rankLevel);
			PruneWeightCopy copy = networkStateCopies.get(rankLevel - 1)
					.get(0);
			copy.restoreWeightsToNetwork(this.prunedNetwork);
		}
	}
	
	private void resolveOtherNeuronCondition() {
		double activeEfficiency = rankEfficiency.get(rankLevel).get(1);
		double previousEfficiency = rankEfficiency.get(rankLevel).get(0);
		double targetEfficiency = errorBound.getErrorBound(previousEfficiency);
		
		if(activeEfficiency < targetEfficiency) {
			neuronsFound = true;
			
			rankEfficiency.remove(rankLevel, previousEfficiency);
			networkStateCopies.removeAll(rankLevel);
			storeRankState();
			storeRankNeuron();
			
			errorBound.advanceChanges();
			advanceActiveLayer();
		}
		else {
			rankEfficiency.remove(rankLevel, activeEfficiency);
			
			PruneWeightCopy copy = networkStateCopies.get(rankLevel)
					.get(0);
			copy.restoreWeightsToNetwork(this.prunedNetwork);
		}
	}
	
	private void storeRankNeuron() {
		int foundNeuronIndex = rankIterator.get(rankLevel).get(0);
		rankCutNeurons.put(rankLevel, foundNeuronIndex);
	}
	
	private void pruneNetwork() {
		if(neuronsFound) {
			pruneWeightMatrices();
			pruneNeuronArrays();
			overrideRankZeroState();
		}
	}
	
	private void pruneWeightMatrices() {
		int hiddenLayerCount = prunedNetwork.getHiddenLayerCount();
		for(int idx = hiddenLayerCount; idx >= 1; --idx) {
			pruneWeightMatricesAtRank(idx);
		}
	}
	
	private void pruneWeightMatricesAtRank(int rank) {
		if(rankCutNeurons.containsKey(rank)) {
			Set<Integer> cutNeuronsSet = new HashSet<>(rankCutNeurons.get(rank));
			TreeSet<Integer> sortedCutNeuronsSet = new TreeSet<>(cutNeuronsSet);
			Layer prunedLayer = prunedNetwork.getLayer(rank);
			Matrix weightMatrix = new Matrix(prunedLayer.getWeightMatrix());
			
			int prunedNeuronsCount = 0;
			for(int neuronIndex : sortedCutNeuronsSet) {
				int prunedIndex = neuronIndex - prunedNeuronsCount;
				weightMatrix = MatrixMath.deleteRow(weightMatrix, prunedIndex);
				++prunedNeuronsCount;
			}
			prunedLayer.forceSetWeightMatrix(weightMatrix);
			
			prunedLayer = prunedLayer.getPreviousLayer();
			weightMatrix = new Matrix(prunedLayer.getWeightMatrix());
			
			prunedNeuronsCount = 0;
			for(int neuronIndex : sortedCutNeuronsSet) {
				int prunedIndex = neuronIndex - prunedNeuronsCount;
				weightMatrix = MatrixMath.deleteColumn(weightMatrix, prunedIndex);
				++prunedNeuronsCount;
			}
			prunedLayer.forceSetWeightMatrix(weightMatrix);
		}
	}
	
	private void pruneNeuronArrays() {
		int hiddenLayerCount = prunedNetwork.getHiddenLayerCount();
		for(int idx = hiddenLayerCount; idx >= 1; --idx) {
			pruneNeuronArrayAtRank(idx);
		}
	}
	
	private void pruneNeuronArrayAtRank(int rank) {
		if(rankCutNeurons.containsKey(rank)) {
			Set<Integer> cutNeuronsSet = new HashSet<>(rankCutNeurons.get(rank));
			TreeSet<Integer> sortedCutNeuronsSet = new TreeSet<>(cutNeuronsSet);
			Layer prunedLayer = prunedNetwork.getLayer(rank);
			
			int prunedNeuronsCount = 0;
			for(int neuronIndex : sortedCutNeuronsSet) {
				int prunedIndex = neuronIndex - prunedNeuronsCount;
				prunedLayer.removeNeuronAt(prunedIndex);
				++prunedNeuronsCount;
			}
		}
	}
	
	private void overrideRankZeroState() {
		PruneWeightCopy copy = new PruneWeightCopy(prunedNetwork);
		double efficiency = rankEfficiency.get(1).get(0);
		
		networkStateCopies.removeAll(0);
		rankEfficiency.removeAll(0);
		
		networkStateCopies.put(0, copy);
		rankEfficiency.put(0, efficiency);
	}
	
	private void clearAllStates() {
		int rankCount = prunedNetwork.getHiddenLayerCount();
		for(int idx = 1; idx <= rankCount; ++idx) {
			clearStateAtRank(idx);
		}
	}
	
	private void clearStateAtRank(int rank) {
		networkStateCopies.removeAll(rank);
		rankEfficiency.removeAll(rank);
		rankIterator.removeAll(rank);
		rankCutNeurons.removeAll(rank);
	}
	
	private void checkIterationRegression() {
		if(rankLevel > 1) {
			regressActiveLayer();
			
			if(rankEfficiency.get(rankLevel + 1).size() > 0) {
				applyPreviousRankState();
			}
		}
		else {
			System.out.println("Omitted regression.");
		}
	}
	
	private void advanceActiveLayer() {
		Layer triggerLayer = activeLayer.getNextLayer();
		if(triggerLayer.getLayerType() == LayerType.HIDDEN) {
			++rankLevel;
			this.activeLayer = triggerLayer;
			
			iterateActiveLayer();
		}
	}
	
	private void regressActiveLayer() {
		--rankLevel;
		this.activeLayer = activeLayer.getPreviousLayer();
	}
	
	private void applyPreviousRankState() {
		double previousRankEfficiency = rankEfficiency.get(rankLevel + 1)
				.get(0);
		double activeRankEfficiency = rankEfficiency.get(rankLevel).get(0);
		double targetEfficiency = errorBound.
				getRegressionErrorBound(activeRankEfficiency);

		if(previousRankEfficiency < targetEfficiency) {
			PruneWeightCopy stateCopy = networkStateCopies.get(rankLevel + 1)
					.get(0);
			
			rankEfficiency.removeAll(rankLevel);
			networkStateCopies.removeAll(rankLevel);
			
			rankEfficiency.put(rankLevel, activeRankEfficiency);
			networkStateCopies.put(rankLevel, stateCopy);
			
			//Testing.
			clearStateAtRank(rankLevel + 1);
			
			errorBound.advanceRegressionChanges();
		}
	}
}