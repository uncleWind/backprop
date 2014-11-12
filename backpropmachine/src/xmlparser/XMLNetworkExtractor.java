package xmlparser;

import org.dom4j.Document;
import org.dom4j.DocumentHelper;
import org.dom4j.Element;

import feedforward.Layer;
import feedforward.Network;

public class XMLNetworkExtractor {
	private Network network;
	
	public XMLNetworkExtractor(Network network) {
		this.network = network;
	}
	
	public Document getXMLDocument() {
		Document document = DocumentHelper.createDocument();
		
		Element elemNetwork = document.addElement("network");
		appendNetworkParamsToRoot(elemNetwork);
		appendNetworkLayersToRoot(elemNetwork);
		appendOutputLayerParametersToRoot(elemNetwork);
		
		return document;
	}
	
	private void appendNetworkParamsToRoot(Element root) {
		Element elemParams = root.addElement("params");
		elemParams.addElement("training")
				  .addText(network.getStringifiedNetworkType());
		elemParams.addElement("rate")
				  .addText(network.getStringifiedLearningRate());
		elemParams.addElement("momentum")
				  .addText(network.getStringifiedMomentum());
		elemParams.addElement("errormethod")
				  .addText(network.getStringifiedErrorCalcMethod());
		elemParams.addElement("targeterror")
				  .addText(network.getStringifiedTargetError());
		elemParams.addElement("bias")
				  .addText(network.getStringifiedBiasInfo());
	}
	
	private void appendNetworkLayersToRoot(Element root) {
		Element elemLayers = root.addElement("layers");
		
		Layer inputLayer = network.getInputLayer();
		appendLayerElementWithWeights(elemLayers, inputLayer);
		
		Layer outputLayer = network.getOutputLayer();
		appendLayerElement(elemLayers, outputLayer);
		
		int lastHiddenLayerIndex = network.size() - 2;
		for(int i = 1; i <= lastHiddenLayerIndex; ++i) {
			Layer hiddenLayer = network.getLayer(i);
			appendLayerElementWithWeights(elemLayers, hiddenLayer);
		}
	}
	
	private void appendLayerElement(Element elemLayers, Layer layer) {
		Element elemLayer = elemLayers.addElement("layer");
		elemLayer.addElement("neurons")
				 .addText(layer.getStringifiedSize());
		elemLayer.addElement("randomizer")
				 .addText(layer.getStringifiedWeightSetter());
		elemLayer.addElement("randminbound")
				 .addText(layer.getStringifiedWeightSetterMinBound());
		elemLayer.addElement("randmaxbound")
				 .addText(layer.getStringifiedWeightSetterMaxBound());
		elemLayer.addElement("activation")
				 .addText(layer.getStringifiedActivationMethod());
	}
	
	private void appendLayerElementWithWeights(Element elemLayers, Layer layer) {
		appendLayerElement(elemLayers, layer);
		
		Element elemWeightMatrix = elemLayers.addElement("weightmatrix");
		elemWeightMatrix.addElement("rows")
						.addText(layer.getStringifiedWeightMatrixRowSize());
		elemWeightMatrix.addElement("columns")
						.addText(layer.getStringifiedWeightMatrixColumnSize());
		
		Element elemValues = elemWeightMatrix.addElement("values");
		int rows = layer.getWeightMatrix().getRows();
		for(int i = 0; i < rows; ++i) {
			elemValues.addElement("row")
					  .addText(layer.getStringifiedWeightMatrixRowValuesAt(i));
		}
	}
	
	private void appendOutputLayerParametersToRoot(Element root) {
		Element elemOutputLayerParams = root.addElement("outputlayerparams");
		
		Layer outLayer = network.getOutputLayer();
		for(int i = 0; i < outLayer.size(); ++i) {
			Element elemNeuron = elemOutputLayerParams.addElement("neuron");
			elemNeuron.addElement("position")
					  .addText(String.valueOf(i + 1));
			elemNeuron.addElement("multiplier")
					  .addText(outLayer.getStringifiedNeuronMultiplierValueAt(i));
		}
	}
}
