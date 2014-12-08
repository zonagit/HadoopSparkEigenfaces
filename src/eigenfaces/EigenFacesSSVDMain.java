package eigenfaces;

import java.io.File;
import java.net.URL;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.math.hadoop.decomposer.DistributedLanczosSolver;
import org.apache.mahout.math.hadoop.stochasticsvd.SSVDCli;
import org.apache.mahout.math.hadoop.stochasticsvd.SSVDSolver;

/**
 * Adaptation of 
 * Algorithm "Generating EingenFaces with Mahout SVD to recognize persons
 * faces"
 * using stochastic SSVD
 */
public class EigenFacesSSVDMain 
{
	public static void main(String args[]) throws Exception
	{
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
		URL purl = EigenFacesMain.class.getProtectionDomain().getCodeSource().getLocation();
		File file = new File(purl.toURI());
		args1[2] = file.getParentFile() + "/training-set";//training set location
		args1[3] = file.getParentFile() + "/output";//output dir
		Configuration conf = new Configuration();
		Path output = new Path(file.getParentFile().toString(),"output");
		FileSystem fs = FileSystem.get(output.toUri(), conf);
		fs.delete(output, true);
		
		GenerateCovarianceMatrix gcm = new GenerateCovarianceMatrix();
		gcm.main(args1);
		Path covMatrix = new Path(file.getParentFile().toString(),"output/covariance.seq");
		//2. Compute the eigenvectors of the covariance matrix
		//using Mahout SVD
		Path tmp = new Path(file.getParentFile().toString() + "/temp" );
		Path tmpCov = new Path(file.getParentFile().toString(),"temp/covariance.seq" );
		fs = FileSystem.get(tmp.toUri(), conf);
		fs.delete(tmp, true);
		fs.copyFromLocalFile(covMatrix, tmpCov);
		String[] args2 = {
		        "--input", tmpCov.toString(),
		        "--output", output.toString(),
		        "--tempDir", tmp.toString(),
		        "--numRows", "151",
		        "--numCols", "151",
		        "--rank", "50",
		        "--symmetric", "false",
		        "--cleansvd", "true"
		};
	//	ToolRunner.run(conf, new DistributedLanczosSolver().new DistributedLanczosSolverJob(), args2);
		//ssvd
		Path difMatrix = new Path(file.getParentFile().toString(),"output/diffmatrix.seq");
		
		Path tmpDif = new Path(file.getParentFile().toString(),"temp/diffmatrix.seq" );
		fs.copyFromLocalFile(covMatrix, tmpDif);
		String[] args22 = {
		        "--input", tmpDif.toString(),
		        "--output", output.toString(),
		        "--tempDir", tmp.toString(),
		        "--computeU", "true",
		        "--computeV", "true",
		        "--rank", "50",
		        "--reduceTasks", "10",
		        "--powerIter", "0"
		};
		SSVDCli ssvdcli = new SSVDCli();
		ssvdcli.main(args22);
		//3. Compute the eingenfaces from the clean eigenvectors
		ComputeEigenFaces cef = new ComputeEigenFaces();
		String[] args3 = {
				file.getParentFile() + "/output/V/v-m-00000",
				file.getParentFile() + "/output/diffmatrix.seq",
				file.getParentFile() + "/output/mean-image.gif",
				80 + "",
				60 + "",
				file.getParentFile() + "/training-set",
				file.getParentFile() + "/output"
		};
		
		cef.main(args3);
		//4. Now test the model
		ComputeDistance cd = new ComputeDistance();
		String[] args4 = {
				file.getParentFile() + "/output/eigenfaces.seq",
				file.getParentFile() + "/output/mean-image.gif",
				file.getParentFile() + "/output/weights.seq",
				80 + "",
				60 + "",
				file.getParentFile() + "/training-set",
				file.getParentFile() + "/testing-set",
				file.getParentFile() + "/output"
		};
		cd.main(args4);
		
	}
}
