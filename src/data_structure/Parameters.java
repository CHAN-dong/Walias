package data_structure;

import java.util.List;

public class Parameters {

    public static int D = 2;

    public static int T = 8;

    public static int H = 8;

    public static int featureSize = 6;

    public static int K = 1;

    public static boolean NeedRefine = true;

    public static boolean NeedExtendWorkload = true;

    public static double Po;

    public static int RECORD = 0;

    public static double StopExtendVolume;

    public static int miniPageSize = 1024;

    public static boolean SplitDimensionOpt = false;

    public static double[] metaLength = new double[20];
    public static double[] QueryLength = new double[20];

    public static void initialParameter(List<Query> workload, DataSpace dataset, boolean needExtendWorkload, boolean needRefine, boolean useDimPredict) {
        metaLength = new double[D];
        QueryLength = new double[D];

        NeedRefine = needRefine;

        SplitDimensionOpt = useDimPredict;

        for (int i = 0; i < D; i++) {
            metaLength[i] = dataset.maxBound.data[i] - dataset.minBound.data[i] + 1;
        }

        // miniVolume
        if (workload != null) {
            Query query = workload.get(0);
            StopExtendVolume = query.computeVolume() / 2;
            for (int i = 0; i < D; i++) {
                QueryLength[i] = query.pointMax.data[i] - query.pointMin.data[i] + 1;
            }
        }

        // initialize Po
        Query MBR = new Query(dataset.minBound, dataset.maxBound);
        double v = MBR.computeVolume();
        int querySize = workload.size();
        Po = querySize / v;


        // initialize
        NeedExtendWorkload = needExtendWorkload;
    }
}
