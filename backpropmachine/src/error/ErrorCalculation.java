package error;

public abstract class ErrorCalculation {
	private double globalError;
    private int patternSize;
     
    public void setGlobalError(double error)
    {
        globalError = error;
    }
     
    public double getGlobalError()
    {
        return globalError;
    }
     
    public void setPatternSize(int size)
    {
        patternSize = size;
    }
     
    public int getPatternSize()
    {
        return patternSize;
    }
     
    abstract public double get();
     
    public void update(double[] actual, double[] ideal) 
    {
    	int length = actual.length;
         
        for(int i = 0; i < length; ++i)
        {
            double delta = ideal[i] - actual[i];
            globalError += Math.pow(delta, 2);
        }
         
        patternSize = length;
    }
     
    public void reset()
    {
        globalError = 0.0f;
        patternSize = 0;
    }
}
