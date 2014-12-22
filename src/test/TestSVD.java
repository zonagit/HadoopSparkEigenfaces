package test;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.LinkedList;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.CompressionType;
import org.apache.hadoop.io.compress.DefaultCodec;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.SingularValueDecomposition;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.stochasticsvd.SSVDHelper;
import org.apache.mahout.math.hadoop.stochasticsvd.SSVDSolver;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;
import org.junit.Test;

public class TestSVD 
{
	private static final double s_epsilon = 1.0E-10d;
	public static Random rnd = RandomUtils.getRandom();
	public static String TMP_DIR_PATH = "/opt/hadoop-2.5.1/share/svdtmp/";
	public static String TMP_LOCAL_DIR_PATH = TMP_DIR_PATH;
	@Test
	@SuppressWarnings("deprecation")
	public static void main(String[] args) throws IOException
	{
		int q = 1;
		
		double percent = 5;
		if (args != null)
		{			
			if (args.length>0)
			{
				percent = Double.parseDouble(args[0]);
			}
			if (args.length>1)
			{
				TMP_LOCAL_DIR_PATH = args[1];
				TMP_DIR_PATH = TMP_LOCAL_DIR_PATH;
			}
			if (args.length>2)
			{
				TMP_DIR_PATH = args[2];
			}
		}
		Configuration conf = new Configuration();
		//conf.set("mapred.job.tracker", "local");
		//conf.set("fs.default.name", "file:///");

		Path aLocPath = new Path(TMP_LOCAL_DIR_PATH, "A/A.seq");
		FileSystem fs = FileSystem.get(aLocPath.toUri(), conf);
		fs.delete(aLocPath,true);
		SequenceFile.Writer w = 
				SequenceFile.createWriter(FileSystem.getLocal(conf),
						conf, 
						aLocPath,
						IntWritable.class,
						VectorWritable.class,
						CompressionType.BLOCK,
						new DefaultCodec());
		int n = 100;
		int m = 2000;
		double muAmplitude = 50.0;
		VectorWritable vw = new VectorWritable();
		IntWritable roww = new IntWritable();
		//write matrix to file A.seq
		for (int i = 0; i < m; i++)
		{
			//vector that only stores non zero doubles as a pair of 
			//parallel arrays one int[] and one double[]
			Vector dv = new SequentialAccessSparseVector(n);
			for (int j = 0; j < n * percent/100; j++)
			{
				dv.setQuick(rnd.nextInt(n), muAmplitude*(rnd.nextDouble()-0.5));
			}
			roww.set(i);
			vw.set(dv);
			w.append(roww, vw);
		}
		w.close();
		
	
	    Path tempDirPath = new Path(TMP_DIR_PATH,"svd-proc");
	    Path aPath = new Path(tempDirPath, "A/A.seq");
	    fs.delete(aPath, true);
	    
	    fs.copyFromLocalFile(aLocPath, aPath);
	    
	    Path svdOutPath = new Path(tempDirPath, "SSVD-out");
	    fs.delete(svdOutPath, true);

	    int ablockRows = 867;
	    int p = 60;
	    int k = 40;
	    SSVDSolver ssvd =
	      new SSVDSolver(conf,
	                     new Path[] { aPath },
	                     svdOutPath,
	                     ablockRows,
	                     k,
	                     p,
	                     3);
	    ssvd.setOuterBlockHeight(500);
	    ssvd.setAbtBlockHeight(251);

	    /*
	     * removing V,U jobs from this test to reduce running time. i will keep them
	     * put in the dense test though.
	     */
	    ssvd.setComputeU(false);
	    ssvd.setComputeV(false);

	    ssvd.setOverwrite(true);
	    ssvd.setQ(q);
	    ssvd.setBroadcast(true);
	    ssvd.run();

	    Vector stochasticSValues = ssvd.getSingularValues();
	    System.out.println("--SSVD solver singular values:");
	    dumpSv(stochasticSValues);
	    System.out.println("--Colt SVD solver singular values:");	    
	    Vector svalues2 = singularValueDecomposition(fs, aPath, conf);
	    
	    for (int i = 0; i < k + p; i++) 
	    {
	        assertTrue(Math.abs(svalues2.getQuick(i) - stochasticSValues.getQuick(i)) <= s_epsilon);
	    }
	    
	    //Spark SSVD
	    org.apache.spark.mllib.linalg.Vector ssparkvalues = sparkSSVD(fs, aPath, conf);
	    for (int i =0; i < k + p; i++)
	    {
	    	assertTrue(Math.abs(ssparkvalues.toArray()[i] - stochasticSValues.getQuick(i)) <= s_epsilon);
	    }
	}

	static org.apache.spark.mllib.linalg.Vector sparkSSVD(FileSystem fs, Path aPath, Configuration conf) throws IOException
	{
		DenseMatrix a = SSVDHelper.drmLoadAsDense(fs, aPath, conf);
		double[][] ac = new double[a.rowSize()][a.columnSize()];//new double[a.columnSize()][a.rowSize()];
		
		for (int i =0;i<ac.length;i++)
		{
			for (int j=0;j<ac[0].length;j++)
			{
				//ac[i][j] = a.get(j, i);
				ac[i][j] = a.get(i, j);		
			}
		}
		LinkedList<org.apache.spark.mllib.linalg.Vector> rowsList = new LinkedList<org.apache.spark.mllib.linalg.Vector>();
		SparkConf sconf = new SparkConf().setAppName("Spark SVD Test").setMaster("local[1]");
		JavaSparkContext jsc = new JavaSparkContext(sconf);
		for (int i=0;i< ac.length;i++)
		{
			org.apache.spark.mllib.linalg.Vector currentRow = Vectors.dense(ac[i]);
			rowsList.add(currentRow);
		}
		JavaRDD<org.apache.spark.mllib.linalg.Vector> rows = jsc.parallelize(rowsList);
		
		//Create a RowMatrix from JavaRDD<Vector>
		RowMatrix mat = new RowMatrix(rows.rdd());
		
		//Compute the top singular values and corresponding singular vectors
		org.apache.spark.mllib.linalg.SingularValueDecomposition<RowMatrix,Matrix> svd = mat.computeSVD(100, true, 1.0E-9d);
		//RowMatrix U = svd.U();
		org.apache.spark.mllib.linalg.Vector s = svd.s();
		dumpSparkSV(s);
		//Matrix V = svd.V();
		
		return s;
	}
	
	//Colt (non randomized) SVD algorithm
	static Vector singularValueDecomposition(FileSystem fs, Path aPath,Configuration conf) throws IOException
	{
		DenseMatrix a = SSVDHelper.drmLoadAsDense(fs, aPath, conf);

	    SingularValueDecomposition svd =
	      new SingularValueDecomposition(a);

	    Vector svalues = new DenseVector(svd.getSingularValues());
	    dumpSv(svalues);
	    
	    return svalues;
	}
	
	//print out singular values
	
	static void dumpSv(Vector s)
	{
		System.out.println("svs:");
		for (Vector.Element el : s.all())
		{
			System.out.println(el.get());
		}
		System.out.println();
	}

	static void dumpSparkSV(org.apache.spark.mllib.linalg.Vector s)
	{
		System.out.println("spark svs:");
		double [] sa = s.toArray();
		for (int i=0;i< sa.length;i++)
		{
			System.out.println(sa[i]);
		}
		System.out.println();
	}
	
	static void dump(double[][] matrix)
	{
		for (double[] amatrix : matrix)
		{
			for (double anAMatrix : amatrix)
			{
				System.out.println(anAMatrix);
			}
			System.out.println();
		}
	}
}
