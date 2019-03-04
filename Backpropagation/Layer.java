import java.util.*;
import java.io.*;
// Layer is an abstract class used for neural network
// The first assignment is only one layer
	abstract class Layer {
	// member variable "activation" can be accessed by subclasses, like "LayerLiner" class
	// Both linear and non-linear layer have output vector
	protected Vec activation;
	protected Vec blame;
	
	protected int inputs;
	protected int outputs;
	//public double momentum;
	// Vector weight is used in linear layer, weight is null in tanh
	//public Vec weight;
	//public Vec gradient;
	//protected Matrix M;
	//protected Vec b;
	//protected Matrix gM;
	//protected Vec gb;
	
	
	// Constructor ==> input:
	Layer(int inputs, int outputs)
	{
        if(inputs < 0 || outputs < 0){
            throw new RuntimeException(
                    "Expected size of inputs and outputs to have positive value");
        }
		activation = new Vec(outputs);
		blame = new Vec(outputs);
		this.inputs = inputs;
		this.outputs = outputs;
		//this.weight = new Vec(outputs * inputs + outputs);
		//this.gradient = new Vec(outputs*inputs + outputs);
		//this.M = new Matrix(outputs,inputs);
		//this.b = new Vec(outputs);
		//this.gM = new Matrix(outputs,inputs);
		//this.gb = new Vec(outputs);
		
	}
	// Calculate activate value, backpro and updateGradient
	abstract void activate(Vec weights, Vec x);
	abstract Vec backprop(Vec weights, Vec prevBlame);
	abstract void updateGradient(Vec x, Vec gradient);
	abstract int NumWeights();
	abstract void initWeights(Vec weights, Random random);
	/*
	public void computeBlame(Vec target)
	{
		if(target.len != activation.len)
			throw new RuntimeException("Dimensions don't match.");
		for(int i =0; i <target.len;i++)
		{
			blame.set(i, target.get(i) - activation.get(i));
		}
	}
	*/
	/*
	public void initWeights(Random rand)
	{
		for(int i =0; i < weight.len;i++)
		{
			double w = (Math.max(0.03,1.0/inputs)) * rand.nextGaussian();
			weight.set(i, w);
		}
		gradient.fill(0.0);
	}
	*/
	/*
	public void setRate(double rate)
	{
		//weights += learning_rate * gradient
		weight.addScaled(rate, gradient);
		//System.out.println("In setRate function==> the value of current layer is:");
		//weight.print();
	}
	*/
}
