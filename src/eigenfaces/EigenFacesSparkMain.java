package eigenfaces;

import java.io.File;
import java.net.URL;
import java.util.LinkedList;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.SingularValueDecomposition;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;

/**
 * Adaptation of 
 * Algorithm "Generating EingenFaces with Mahout SVD to recognize persons
 * faces"
 * using stochastic svd and Spark
 */
public class EigenFacesSparkMain 
{
	public static void main(String args[]) throws Exception
	{
		int rank = 50;
		if (args != null && args.length>0)
		{
			rank = Integer.parseInt(args[0]);
		}
		URL purl = EigenFacesMain.class.getProtectionDomain().getCodeSource().getLocation();
		File file = new File(purl.toURI());
		String input_location = file.getParentFile().toString();
		if (args!= null && args.length>1)
		{
			input_location = args[1];
		}
		String output_location = input_location;
		if (args!= null && args.length>2)
		{
			output_location = args[2];
		}
		//1. Train the model
		//1.a Read all n images from the training dir
		//1.b Convert each image to greyscale and scale it down
		//1.c Create a matrix M with each column representing an
		//image (so length wxh) and each entry is a value between 0 and 255
		//1.d Average the images pixels to create mean-image.gif
		//1.e Compute the diff matrix DM by subtracting the mean image
		//from M
		//1.f Write the diff matrix to diffmatrix.seq
		//1.g Write the covariance matrix to covariance.seq
		String args1[] = new String[4];
		args1[0] = 80 + "";
		args1[1] = 60 + "";
		args1[2] = input_location + "/training-set";//training set location
		args1[3] = output_location + "/output";//output dir
		Configuration conf = new Configuration();
		Path output = new Path(output_location,"output");
		FileSystem fs = FileSystem.get(output.toUri(), conf);
		fs.delete(output, true);
		
		GenerateCovarianceMatrix gcm = new GenerateCovarianceMatrix();
		gcm.main(args1);
		Path covMatrix = new Path(file.getParentFile().toString(),"output/covariance.seq");
		//2. Compute the eigenvectors of the covariance matrix
		//using Spark SVD
		SparkConf sconf = new SparkConf().setAppName("Spark SVD").setMaster("local[1]");
		JavaSparkContext jsc = new JavaSparkContext(sconf);
		LinkedList<Vector> rowsList = new LinkedList<Vector>();
		double [][] diffMatrix = gcm.getDiffMatrix(args1);
		for (int i=0;i< diffMatrix.length;i++)
		{
			Vector currentRow = Vectors.dense(diffMatrix[i]);
			rowsList.add(currentRow);
		}
		JavaRDD<Vector> rows = jsc.parallelize(rowsList);
		
		//Create a RowMatrix from JavaRDD<Vector>
		RowMatrix mat = new RowMatrix(rows.rdd());
		
		//Compute the top singular values and corresponding singular vectors
		SingularValueDecomposition<RowMatrix,Matrix> svd = mat.computeSVD(rank, true, 1.0E-9d);
		RowMatrix U = svd.U();
		Vector s = svd.s();
		Matrix V = svd.V();
		//Print singular values
		System.out.println("Printing singular values");
		for (int i=0;i<s.toArray().length;i++)
		{
			System.out.println(s.toArray()[i]);
		}
		//3. Compute the eingenfaces from the clean eigenvectors
		ComputeEigenFaces cef = new ComputeEigenFaces();
		String[] args3 = {
				output_location + "/output/V/v-m-00001",
				output_location + "/output/diffmatrix.seq",
				input_location + "/training-set/mean-image.gif",
				80 + "",
				60 + "",
				input_location + "/training-set",
				output_location + "/output",
				input_location + "/images"
		};
		
		cef.main(args3,V);
		//4. Now test the model
		ComputeDistance cd = new ComputeDistance();
		String[] args4 = {
				output_location + "/output/eigenfaces.seq",
				input_location + "/training-set/mean-image.gif",
				output_location + "/output/weights.seq",
				80 + "",
				60 + "",
				input_location + "/training-set",
				input_location + "/testing-set",
				output_location + "/output",
				input_location + "/images"
		};
		cd.main(args4);
	}
}
