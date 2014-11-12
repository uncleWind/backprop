package pruning;

import java.util.ArrayList;

import validation.Validator;
import feedforward.Layer;
import feedforward.Network;
import matrixutil.Matrix;

class PruneWeightCopy {
	private ArrayList<Matrix> weightMatricesCopies;
	
	public PruneWeightCopy(Network network) {
		this.weightMatricesCopies = new ArrayList<>();
		
		for(int i = 0; i < network.size() - 1; ++i) {
			Layer layer = network.getLayer(i);
			Matrix matrixCopy = new Matrix(layer.getWeightMatrix());
			weightMatricesCopies.add(matrixCopy);
		}
	}
	
	public void restoreWeightsToNetwork(Network network) {
		Validator.validateNetworkCopyAbility(network, weightMatricesCopies);
		
		for(int i = 0; i < network.size() - 1; ++i) {
			Layer layer = network.getLayer(i);
			Matrix restoredMatrix = new Matrix(weightMatricesCopies.get(i));
			
			System.out.println("Restored matrix at " + i + ":");
			printWeightMatrices(restoredMatrix);
			
			layer.forceSetWeightMatrix(restoredMatrix);
		}
	}
	
	//Testing purposes only.
		private void printWeightMatrices(Matrix matrix) {
			for(int i = 0; i < matrix.getRows(); ++i) {
				for(int j = 0; j < matrix.getColumns(); ++j) {
					System.out.print(matrix.getCell(i, j) + " ");
				}
				System.out.print("\n");
			}
		}
}