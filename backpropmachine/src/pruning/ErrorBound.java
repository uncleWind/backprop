package pruning;

class ErrorBound {
	private double multiplierValue;
	private double regressionMultiplierValue;
	private double nextChange;
	private double regressionNextChange;
	
	public ErrorBound() {
		multiplierValue = 1.1f;
		regressionMultiplierValue = 1.1f;
		nextChange = 0.05f;
		regressionNextChange = 0.05f;
	}
	
	public double getErrorBound(double errorValue) {
		double errorBound = errorValue * multiplierValue;
		return errorBound;
	}
	
	public double getRegressionErrorBound(double errorValue) {
		double errorBound = errorValue * regressionMultiplierValue;
		return errorBound;
	}
	
	public void advanceChanges() {
		multiplierValue = 1.0f + nextChange;
		
		nextChange /= 2;
	}
	
	public void advanceRegressionChanges() {
		regressionMultiplierValue = 1.0f + regressionNextChange;
		
		regressionNextChange /= 2;
	}
}
