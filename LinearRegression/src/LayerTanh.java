import java.util.Random;

public class LayerTanh extends Layer {
	LayerTanh(int outputs)
	{
		super(outputs, outputs);
	}

	void activate(Vec weights, Vec x)
	{
		for(int i = 0; i< outputs;i++)
		{
			this.activation.set(i, Math.tanh(x.get(i)));	
		}
		//activation.print();
	}

	Vec backprop(Vec weights, Vec prevBlame) 
	{
		if(blame.size() != activation.size())
		throw new RuntimeException("blame size != activation size");

	    Vec nextBlame = new Vec(prevBlame.size());

	    blame.fill(0.0);
	    blame.add(prevBlame);

	    for(int i = 0; i < inputs; ++i)
	    {
	    	double derivative = prevBlame.get(i) * (1.0 - (activation.get(i) * activation.get(i)));
	    	nextBlame.set(i, derivative);
	    }

	    return nextBlame;
	}
	@Override
	void updateGradient(Vec x, Vec gradient) {
		// Do nothing	
	}

	@Override
	int NumWeights() {
		return 0;
	}
	void initWeights(Vec weights, Random random)
	{
		// do nothing
	} 
}
