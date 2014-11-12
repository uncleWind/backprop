package pruning;

import java.util.ArrayList;

import feedforward.Network;

public abstract class Pruner {
	protected Network prunedNetwork;
	protected ArrayList<double[]> inputs;
	protected ArrayList<double[]> outputs;
	
	public abstract void prune();
}
