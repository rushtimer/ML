// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

import java.util.Iterator;
import java.util.Random;
import java.lang.StringBuilder;

/// Represents a vector of doubles
public class Vec
{
	protected double[] vals;
	protected int start;
	protected int len;

	/// Makes an vector of the specified size
	public Vec(int size)
	{
		if(size == 0)
			vals = null;
		else
			vals = new double[size];
		start = 0;
		len = size;
	}
	public Vec(Vec that) {
		vals = new double[that.size()];
		for(int i = 0; i < that.size(); i++)
			vals[i] = that.get(i);
		start = 0;
		len = that.size();
	}
	/// Wraps the specified array of doubles
	public Vec(double[] data)
	{
		vals = data;
		start = 0;
		len = data.length;
	}

	/// This is NOT a copy constructor. It wraps the same buffer of values as v.
	public Vec(Vec v, int begin, int length)
	{
		vals = v.vals;
		start = v.start + begin;
		len = length;
	}

	/// Unmarshalling constructor
	public Vec(Json n)
	{
		vals = new double[n.size()];
		for(int i = 0; i < n.size(); i++)
			vals[i] = n.getDouble(i);
		start = 0;
		len = n.size();
	}

	public Json marshal()
	{
		Json list = Json.newList();
		for(int i = 0; i < len; i++)
			list.add(vals[start + i]);
		return list;
	}

	public int size()
	{
		return len;
	}

	public double get(int index)
	{
		//System.out.println(vals.length);
		return vals[start + index];
	}

	public void set(int index, double value)
	{
		vals[start + index] = value;
	}

	public void fill(double val)
	{
		for(int i = 0; i < len; i++)
			vals[start + i] = val;
	}

	public String toString()
	{
		StringBuilder sb = new StringBuilder();
		if(len > 0)
		{
			sb.append(Double.toString(vals[start]));
			for(int i = 1; i < len; i++)
			{
				sb.append(",");
				sb.append(Double.toString(vals[start + i]));
			}
		}
		return sb.toString();
	}

	public double squaredMagnitude()
	{
		double d = 0.0;
		for(int i = 0; i < len; i++)
			d += vals[start + i] * vals[start + i];
		return d;
	}

	public void normalize()
	{
		double mag = squaredMagnitude();
		if(mag <= 0.0) {
			fill(0.0);
			vals[0] = 1.0;
		} else {
			double s = 1.0 / Math.sqrt(mag);
			for(int i = 0; i < len; i++)
				vals[i] *= s;
		}
	}

	public void copy(Vec that)
	{
		vals = new double[that.size()];
		for(int i = 0; i < that.size(); i++)
			vals[i] = that.get(i);
		start = 0;
		len = that.size();
	}

	public void add(Vec that)
	{
		if(that.size() != this.size())
			throw new IllegalArgumentException("mismatching sizes");
		for(int i = 0; i < len; i++)
			vals[start + i] += that.get(i);
	}
	
    public Vec plus(Vec that) {
        if (this.size() != that.size())
            throw new IllegalArgumentException("dimensions disagree");
        Vec c = new Vec(len);
        for (int i = 0; i < len; i++)
            c.vals[i] = this.vals[i] + that.vals[i];
        return c;
    }

	public void scale(double scalar)
	{
		for(int i = 0; i < len; i++)
			vals[start + i] *= scalar;
	}

	public void addScaled(double scalar, Vec that)
	{
		if(that.size() != this.size())
			throw new IllegalArgumentException("mismatching sizes");
		for(int i = 0; i < len; i++)
			vals[start + i] += scalar * that.get(i);
	}

	public double dotProduct(Vec that)
	{
		if(that.size() != this.size())
			throw new IllegalArgumentException("mismatching sizes");
		double d = 0.0;
		for(int i = 0; i < len; i++)
			d += get(i) * that.get(i);
		return d;
	}

	public double squaredDistance(Vec that)
	{
		if(that.size() != this.size())
			throw new IllegalArgumentException("mismatching sizes");
		double d = 0.0;
		for(int i = 0; i < len; i++)
		{
			double t = get(i) - that.get(i);
			d += (t * t);
		}
		return d;
	}
	public void print()
	{
		for(int i = 0; i<len; i++)
		{
			System.out.println(vals[i]);
		}
	}
	public Vec fillrand(Vec that)
	{
		Random rand = new Random();
		//generate random Vector weights	
		for(int i = 0; i<that.size();i++)
		{
			that.set(i, rand.nextGaussian());
		}
		return that;
	}
	
	public Vec convertVec(Matrix that)
	{
		Vec Xvector = new Vec(that.rows()*that.cols());
		int count = 0;
		for(int i = 0; i < that.rows();i++)
		{
			for(int j =0; j< that.cols();j++)
			{
				Xvector.set(count++, that.getvalue(i,j));
			}
		}
		return Xvector;
	}
	/*
	public Vec addvalues(int size, double vals[])
	{
		Vec vector = new Vec(size);
		for(int i = 0; i<size ; i++)
		{
			vector.set(i, vals[i]);
		}
		return vector;
	}
	*/
	// Convert Matrix X: catch each instance from matrix X and convert them in vectors
	// the number of vectors equals to the number of instance of X
	public Vec[] convertXtoVec(Matrix that)
	{
		Vec[] arrayX; 
		arrayX =  new Vec[that.rows()];
		for(int i = 0; i < that.rows();i++)
		{
			int index = 0;
			for(int j =0; j< that.cols();j++)
			{
				arrayX[i].set(index++, that.getvalue(i, j));
				arrayX[i].print();
			}
		}	
		return arrayX;
	}
	
	public Matrix convertoMatrix(Vec x)
	{
		Matrix X = new Matrix(x.size(),1);
		int index = 0;
		for(int i = 0; i < x.size(); i++)
		{
			for(int j= 0; j < 1;j++)
			{
				X.setMatrix(i, j, x.get(index++));
			}
		}
		return X;
	}
	
	public Vec computeB(Vec y_cen, Matrix M, Vec x_cen)
	{

		Vec b = new Vec(M.rows());
		// convert Vec x_cen to Matrix X
		Matrix X = new Matrix(x_cen.size(),1);
		X = x_cen.convertoMatrix(x_cen);
		//System.out.println("Matrix X is: ");
		//X.print();
        if (M.cols() != X.rows())
            throw new IllegalArgumentException("dimensions disagree");
		Matrix MX = new Matrix(M.rows(),X.cols());
        MX = M.scaleMatrix(-1, M).times(X);
        //System.out.println("Matrix MX is ");
        //MX.print();
        // convert Matrix MX to Vector mx
		Vec mx = new Vec(M.rows()*X.cols());
		mx = mx.convertVec(MX);
        //System.out.println("Vector mx is ");
        //mx.print();
        if (mx.size() != y_cen.size())
            throw new IllegalArgumentException("demisions of y_cen doesn't match (-M*x_cen)");
        b= y_cen.plus(mx);
        //System.out.println("the value of b is ");
        //b.print();
        /*
        System.out.println("Matrix -M is " ); 
        M.scaleMatrix(-1, M).print();
        System.out.println("x_cen is");
        x_cen.print();
        System.out.println("-MX is " );
        MX.print();
        M.scaleMatrix(-1, M).times(X).print();
        */
		return b;
	}
	
	public Vec concatenate (Vec w, Vec b)
	{
		Vec combo = new Vec(w.size()+b.size());
		int count = 0;
		for(int i = 0; i < w.size();i++)
		{
			combo.set(count++, w.get(i));
		}
		for(int j = 0; j< b.size();j++)
		{
			combo.set(count++, b.get(j));
		}
		//combo.print();
		return combo;
	}
	public Vec setZero(Vec x)
	{	
		int index=0;
		for(int i =0; i< x.size();i++)
		{
			x.set(index++, 0.0);
		}
		return x;
	}


	public Vec mtProductBlame(Matrix mt, Vec blame) {
		Vec result = new Vec(mt.rows());
    	double c = 0.0;
    	//System.out.println("mt size"+ mt.rows()+" x " + mt.cols());
    	//System.out.println("blame size" + blame.len);
    	if(mt.rows() != blame.len)
    		throw new RuntimeException("Error: MT * blame=> dimensions don't match.");
    	for(int i =0; i< mt.rows();i++)
    	{
    		for(int j =0; j < mt.cols();j++)
    		{
    			c = c+ mt.getvalue(i, j) * blame.get(j);
    		}
    		result.set(i, c);
    		c=0; // reused c 
    	}
    	return result;
	}
	
	public void oneHot() {
		double max = this.get(0);
		int maxIndex = 0;

		// Obtain the maximum value
		for(int i = 0; i < this.size(); ++i) {
			if(this.get(i) > max) {
				max = this.get(i);
				maxIndex = i;
			}
		}

		for(int i = 0; i < this.size(); ++i) {
			if(i != maxIndex)
				this.set(i, 0.0);
		}
		this.set(maxIndex, 1.0);
	}
	
	public Vec subtract(Vec target, Vec prediction)
	{
		Vec result = new Vec(target.len);
		if(target.len !=prediction.len)
			throw new RuntimeException("Error: size of target doesn't match prediction.");
		for(int i =0; i < target.len;i++)
		{
			result.set(i, target.get(i)- prediction.get(i));
		}
		return result;
	}
}






/// A tensor class.
class Tensor extends Vec
{
	int[] dims;

	/// General-purpose constructor. Example:
	/// Tensor t(v, {5, 7, 3});
	Tensor(Vec vals, int[] _dims)
	{
		super(vals, 0, vals.size());
		dims = new int[_dims.length];
		int tot = 1;
		for(int i = 0; i < _dims.length; i++)
		{
			dims[i] = _dims[i];
			tot *= _dims[i];
		}
		if(tot != vals.size())
			throw new RuntimeException("Mismatching sizes. Vec has " + Integer.toString(vals.size()) + ", Tensor has " + Integer.toString(tot));
	}

	/// Copy constructor. Copies the dimensions. Wraps the same vector.
	Tensor(Tensor copyMe)
	{
		super((Vec)copyMe, 0, copyMe.size());
		dims = new int[copyMe.dims.length];
		for(int i = 0; i < copyMe.dims.length; i++)
			dims[i] = copyMe.dims[i];
	}

	/// The result is added to the existing contents of out. It does not replace the existing contents of out.
	/// Padding is computed as necessary to fill the the out tensor.
	/// filter is the filter to convolve with in.
	/// If flipFilter is true, then the filter is flipped in all dimensions.
	static void convolve(Tensor in, Tensor filter, Tensor out, boolean flipFilter, int stride)
	{
		// Precompute some values
		int dc = in.dims.length;
		if(dc != filter.dims.length)
			throw new RuntimeException("Expected tensors with the same number of dimensions");
		if(dc != out.dims.length)
			throw new RuntimeException("Expected tensors with the same number of dimensions");
		int[] kinner = new int[dc];
		int[] kouter = new int[dc];
		int[] stepInner = new int[dc];
		int[] stepFilter = new int[dc];
		int[] stepOuter = new int[dc];

		// Compute step sizes
		stepInner[0] = 1;
		stepFilter[0] = 1;
		stepOuter[0] = 1;
		for(int i = 1; i < dc; i++)
		{
			stepInner[i] = stepInner[i - 1] * in.dims[i - 1];
			stepFilter[i] = stepFilter[i - 1] * filter.dims[i - 1];
			stepOuter[i] = stepOuter[i - 1] * out.dims[i - 1];
		}
		int filterTail = stepFilter[dc - 1] * filter.dims[dc - 1] - 1;

		// Do convolution
		int op = 0;
		int ip = 0;
		int fp = 0;
		for(int i = 0; i < dc; i++)
		{
			kouter[i] = 0;
			kinner[i] = 0;
			int padding = (stride * (out.dims[i] - 1) + filter.dims[i] - in.dims[i]) / 2;
			int adj = (padding - Math.min(padding, kouter[i])) - kinner[i];
			kinner[i] += adj;
			fp += adj * stepFilter[i];
		}
		while(true) // kouter
		{
			double val = 0.0;

			// Fix up the initial kinner positions
			for(int i = 0; i < dc; i++)
			{
				int padding = (stride * (out.dims[i] - 1) + filter.dims[i] - in.dims[i]) / 2;
				int adj = (padding - Math.min(padding, (int)kouter[i])) - kinner[i];
				kinner[i] += adj;
				fp += adj * stepFilter[i];
				ip += adj * stepInner[i];
			}
			while(true) // kinner
			{
				val += (in.get(ip) * filter.get(flipFilter ? filterTail - fp : fp));

				// increment the kinner position
				int i;
				for(i = 0; i < dc; i++)
				{
					kinner[i]++;
					ip += stepInner[i];
					fp += stepFilter[i];
					int padding = (stride * (out.dims[i] - 1) + filter.dims[i] - in.dims[i]) / 2;
					if(kinner[i] < filter.dims[i] && kouter[i] + kinner[i] - padding < in.dims[i])
						break;
					int adj = (padding - Math.min(padding, (int)kouter[i])) - kinner[i];
					kinner[i] += adj;
					fp += adj * stepFilter[i];
					ip += adj * stepInner[i];
				}
				if(i >= dc)
					break;
			}
			out.set(op, out.get(op) + val);

			// increment the kouter position
			int i;
			for(i = 0; i < dc; i++)
			{
				kouter[i]++;
				op += stepOuter[i];
				ip += stride * stepInner[i];
				if(kouter[i] < out.dims[i])
					break;
				op -= kouter[i] * stepOuter[i];
				ip -= kouter[i] * stride * stepInner[i];
				kouter[i] = 0;
			}
			if(i >= dc)
				break;
		}
	}

	/// Throws an exception if something is wrong.
	static void test()
	{
		{
			// 1D test
			Vec in = new Vec(new double[]{2,3,1,0,1});
			Tensor tin = new Tensor(in, new int[]{5});

			Vec k = new Vec(new double[]{1, 0, 2});
			Tensor tk = new Tensor(k, new int[]{3});

			Vec out = new Vec(7);
			Tensor tout = new Tensor(out, new int[]{7});

			Tensor.convolve(tin, tk, tout, true, 1);

			//     2 3 1 0 1
			// 2 0 1 --->
			Vec expected = new Vec(new double[]{2, 3, 5, 6, 3, 0, 2});
			if(Math.sqrt(out.squaredDistance(expected)) > 1e-10)
				throw new RuntimeException("wrong");
		}

		{
			// 2D test
			Vec in = new Vec(new double[]
				{
					1, 2, 3,
					4, 5, 6,
					7, 8, 9
				}
			);
			Tensor tin = new Tensor(in, new int[]{3, 3});

			Vec k = new Vec(new double[]
				{
					1,  2,  1,
					0,  0,  0,
					-1, -2, -1
				}
			);
			Tensor tk = new Tensor(k, new int[]{3, 3});

			Vec out = new Vec(9);
			Tensor tout = new Tensor(out, new int[]{3, 3});

			Tensor.convolve(tin, tk, tout, false, 1);
			
			Vec expected = new Vec(new double[]
				{
					-13, -20, -17,
					-18, -24, -18,
					13,  20,  17
				}
			);
			if(Math.sqrt(out.squaredDistance(expected)) > 1e-10)
				throw new RuntimeException("wrong");
		}
	}
}
