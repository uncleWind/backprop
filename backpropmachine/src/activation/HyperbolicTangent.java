package activation;

public class HyperbolicTangent implements ActivationFunction {
	@Override
    public double getValue(double x) 
    {
        double result = (Math.exp(2 * x) - 1) / (Math.exp(2 * x) + 1);
        return result;
    }
 
    @Override
    public double getDerivative(double x) 
    {
        double result = 1 - Math.pow(getValue(x), 2);
        return result;
    }
}