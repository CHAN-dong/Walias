import data_structure.*;
import index.WalasTree;

import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.logging.LogManager;

import static data_structure.Parameters.*;
import static data_structure.Parameters.initialParameter;
import static utils.DatasetVisualizer.drawForPartition;
import static utils.FileUtils.readDataset;
import static utils.FileUtils.readQuery;
import static utils.Tools.getSubDataset;
import static utils.Tools.getSubWorkload;

public class Main {

    public static void main(String[] args) throws Exception {


        String datasetPath = "./src/datasets/dataset/TPC-1M.csv";
        String workloadPath = "./src/datasets/workload/UNI_5.0E-5_Query_TCP-100M_test.csv";
        String queryPath = "./src/datasets/workload/qUNI_5.0E-5_Query_TCP-100M_test.csv";

        Test(datasetPath, workloadPath, queryPath);
    }

    static int roundTime = 0;

    public static void Test(String datasetPath, String workloadPath, String testWorkloadPath) throws Exception {
        LogManager.getLogManager().reset();

        DataSpace dataSpace = readDataset(datasetPath);
        List<Query> workload = readQuery(workloadPath);
        List<Query> testWorkload = readQuery(testWorkloadPath);
        long s, e;

        initialParameter(workload, dataSpace, false, false, false);
        System.out.println("Dataset size: " + dataSpace.getDatasetSize() + ", Workload size: " + workload.size());
        s = System.currentTimeMillis();
        WalasTree walasTree = new WalasTree(dataSpace, workload, testWorkload);
        e = System.currentTimeMillis();
        System.out.println("index construct time: " + (e - s) / 1e3 + "s");
        walasTree.printIndexPerformance();
        System.out.println("WalasTree cost : " + walasTree.refiningSpaceCost(walasTree.root));
        walasTree.drawPartition(roundTime++);

//        walasTree.startRefine(K);

        System.out.println("Once Time End!");
        System.out.println("Once Time End!");
    }
}