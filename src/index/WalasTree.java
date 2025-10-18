package index;

import data_structure.*;
import utils.DatasetVisualizer;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

import static data_structure.Parameters.*;
import static index.Node.getNodeCost;
import static utils.DatasetVisualizer.drawForPartition;
import static utils.DatasetVisualizer.exportHighResImage;
import static utils.Tools.*;

public class WalasTree {
    static int cores = Runtime.getRuntime().availableProcessors();
    Instances instances; // samples for training

    boolean useModel = false;
    List<Query> workload;

    public List<Query> testWorkload;

    DataSpace dataset;

    RandomForest randomForest;
    int iteration = 0;

    static final double rate = 0.01; // used for DP

    public Node root;

    public double treeCost;

    static List<Query> insertedWorkload = new ArrayList<>();

    static HashMap<Point, Integer> objectTraveledCountMap = new HashMap<>();

    public WalasTree(DataSpace dataset, List<Query> workload, List<Query> testWorkload) throws Exception {
        this.dataset = dataset;
        this.workload = workload;
        this.testWorkload = testWorkload;

        Object[] objects = recursiveBuildTree(dataset, workload);
        root = (Node)objects[1];
    }

//    public void recursiveRefiningByMode () throws Exception {
//        useModel = true;
//        initialObjectTravelMap(workload);
//        for (iteration = 1; iteration <= K; ++iteration) {
//            // set training samples
//            List<ModelSample> samples = refining(workload); // refining!!!
//            instances = buildInstancesBySamples(samples);
//
//            //train model
//            randomForest = new RandomForest();
//            randomForest.setNumExecutionSlots(cores);
//            randomForest.setNumIterations(100);
//            randomForest.buildClassifier(instances);
//
//            // rebuild tree
//            Node preNode = root;
//            root = (Node) recursiveBuildTree(dataset, workload)[1];
//        }
//    }


    public List<ModelSample> refining(List<Query> workload) {
        List<List<ModelSample>> samples = new ArrayList<>();

        int sampleClass = 11;

        for (int i = 0; i < sampleClass; ++i) {
            samples.add(new ArrayList<>());
        }
        treeCost = refiningSpaceCost(root);

        // set node features
        travelTreeForOtherFeatures(root);
        for (Query query : workload) {
            queryTreeForOiQ_QFeature(root, query);
        }

        getSamples(root, samples);

        List<ModelSample> modelSamples = new ArrayList<>();
        for (int i = 0; i < sampleClass; ++i) {
            modelSamples.addAll(getSubSamples(100000, samples.get(i)));
        }

        return modelSamples;
    }

    public HashSet<Query> travelTreeGetPartitions() {
        HashSet<Query> partitions = new HashSet<>();
        Queue<Node> queue = new LinkedList<>();
        queue.add(root);

        while (!queue.isEmpty()) {
            Node node = queue.remove();
            if (!node.isLeafNode()) {
                partitions.addAll(node.MBRs);
                queue.addAll(node.childes);
            }
        }
        return partitions;
    }

    public void drawPartition(int iteration) {
//        HashSet<Query> partitions = travelTreeGetPartitions();
        HashSet<Query> partitions = new HashSet<>();
//        partitions.addAll(root.MBRs);
//        drawForPartition(partitions, new HashSet<>(workload), dataset);

        partitions = travelTreeGetPartitions();
        drawForPartition(partitions, new HashSet<>(workload), dataset, new HashSet<>(insertedWorkload));

//        partitions.addAll(root.MBRs);

        exportHighResImage("./drawTest" + (RECORD++) + "," +iteration + ".pdf", 1024);
    }

    public void printIndexPerformance() {
        // index size
        getIndexSize();

        // query time
        queryTest(testWorkload);
        queryTest(testWorkload);
        queryTest(testWorkload);
        queryTest(testWorkload);
        queryTest(testWorkload);
        queryTest(testWorkload);
        queryTest(testWorkload);
        queryTest(testWorkload);
        queryTest(testWorkload);
        queryTest(testWorkload);
    }


    public void getIndexSize() {
        long size = 0;
        Queue<Node> queue = new LinkedList<>();
        queue.add(root);

        while (!queue.isEmpty()) {
            size += 40; // Node object: Object header 16 + reference field (6 references*8=48) ≈ 40-64 bytes, take 40
            Node remove = queue.remove();
            if (remove.isLeafNode()) {
                size += 24 + remove.dataset.size() * 8L; //each integer object object header 16 bytes
            } else {
                size += 24 + remove.childes.size() * 8L;//  reference array 8 bytes per reference
                size += 24 + remove.MBRs.size() * 2 * (8 + D * 16L); //each double object 16 bytes in the object head
                queue.addAll(remove.childes);
            }
        }

        System.out.println("index size: " + size / 1024.0 / 1024.0 + "MB");
    }


    public void startRefine(int round) throws Exception {
        long s, e;
        int cores = Runtime.getRuntime().availableProcessors();
        Parameters.K = round;
        useModel = true;
        initialObjectTravelMap(workload);
        System.out.println("----------before refining performance: ");
        printIndexPerformance();
        System.out.println();

        drawPartition(iteration);

        if (!NeedRefine) return;
//        Node preNode = root;
        for (iteration = 1; iteration <= K; ++iteration) {
            System.out.println("------------iteration: " + iteration + " Start!!!");

            s = System.currentTimeMillis();
            // set training samples
            List<ModelSample> samples = refining(workload); // refining!!!
            instances = buildInstancesBySamples(samples);
            e = System.currentTimeMillis();
            System.out.println("Get sample complete and get size " + samples.size() + ", consume time: " + (e - s) / 1000.0 + "s");

//            System.out.println("preNode cost: " + preNode.cost + ", nowNode cost: " + root.cost);
            //train model
            root = null;
            s = System.currentTimeMillis();
            randomForest = new RandomForest();
            randomForest.setNumExecutionSlots(cores);
            randomForest.setNumIterations(100);
            randomForest.buildClassifier(instances);
            e = System.currentTimeMillis();
            System.out.println("model training complete, consume time: " + (e - s) / 1000.0 + "s");

//             evaluate model result by train samples
            s = System.currentTimeMillis();
            Evaluation eval = new Evaluation(instances);
            eval.evaluateModel(randomForest, instances);
//            System.out.println(eval.toSummaryString());
            e = System.currentTimeMillis();
//            System.out.println("model evaluate complete, consume time: " + (e - s) / 1000.0 + "s");

            // rebuild tree
            s = System.currentTimeMillis();
//            preNode = root;
            root = (Node) recursiveBuildTree(dataset, workload)[1];
            e = System.currentTimeMillis();
            System.out.println("tree rebuild complete, consume time: " + (e - s) / 1000.0 + "s");

            // query performance test
            printIndexPerformance();

            System.out.println();

            drawPartition(iteration);
        }

        int a = 1;
    }

    public Instances buildInstancesBySamples(List<ModelSample> samples) {
        ArrayList<Attribute> attrs = new ArrayList<>();
        attrs.add(new Attribute("cost")); // predict the goal
        attrs.add(new Attribute("OiQ"));
        attrs.add(new Attribute("querySize"));
        attrs.add(new Attribute("objectSize"));
        attrs.add(new Attribute("perimeter"));
        attrs.add(new Attribute("volume"));

        Instances data = new Instances("ModelSamples", attrs, samples.size());

        for (ModelSample s : samples) {
            data.add(new DenseInstance(s.weight, s.features));
        }

        data.setClassIndex(0);
        return data;
    }


    public double getSamples(Node node, List<List<ModelSample>> samples) { // return tree avg height
        if (node.isLeafNode()) return 1;
        int n = node.childes.size();
        double[] heights = new double[n];

        for (int i = 0; i < n; i++) {
            heights[i] = getSamples(node.childes.get(i), samples);
            double t = 1, sumH = 0;
            if (node.samples != null) {
                for (int j = i; j < n && j - i < T; j++) {
                    sumH += heights[j];
                    node.samples[i][j].setWeight(sumH / t, t);

                    int part = (int) (node.samples[i][j].weight * 10);
                    samples.get(part).add(node.samples[i][j]);
                    t++;
                }
            }
        }
        double height = Arrays.stream(heights).sum() / n;
        return height + 1;
    }

    public int travelTreeForOtherFeatures(Node node) { // return object size
        int objectSize = 0;
        if (node.isLeafNode()) {
            return node.dataset.size();
        }

        int n = node.childes.size();
        int[] childSize = new int[n];
        for (int i = 0; i < n; ++i) {
            childSize[i] = travelTreeForOtherFeatures(node.childes.get(i));
            objectSize += childSize[i];
        }

        if (node.samples != null) {
            int[][] samplesObjectSize = new int[n][n];
            for (int i = 0; i < n; ++i) {
                samplesObjectSize[i][i] = childSize[i];
                node.samples[i][i].setObjectSizeToFeature(samplesObjectSize[i][i]);
                node.samples[i][i].setVolumeToFeature(node.MBRs.get(i));
                node.samples[i][i].setPerimeterToFeature(node.MBRs.get(i));
            }

            for (int i = 0; i < n; ++i) {
                Query mbr = node.MBRs.get(i);
                for (int j = i + 1; j < n && j - i < T; ++j) {
                    samplesObjectSize[i][j] = samplesObjectSize[i][j - 1] + samplesObjectSize[j][j];
                    mbr.union(node.MBRs.get(j));

                    node.samples[i][j].setObjectSizeToFeature(samplesObjectSize[i][j]);
                    node.samples[i][j].setVolumeToFeature(mbr);
                    node.samples[i][j].setPerimeterToFeature(mbr);
                }
            }
        }

        return objectSize;
    }

    public int queryTreeForOiQ_QFeature(Node node, Query query) {
        if (node.isLeafNode()) {
            List<Point> subRes = node.leafNodeQuery(query);
            return subRes.size();
        }
        int OiQ = 0;
        int n = node.childes.size();

        int[] childOiQ = new int[n];
        for (int i = 0; i < n; ++i) {
            if (query.hasIntersection(node.MBRs.get(i))) {
                childOiQ[i] = queryTreeForOiQ_QFeature(node.childes.get(i), query);
                OiQ += childOiQ[i];
            }
        }

        if (node.samples != null) {
            int[][] samplesQiQ = new int[n][n];
            // get child node QiQ
            for (int i = 0; i < n; ++i) {
                if (query.hasIntersection(node.MBRs.get(i))) {
                    samplesQiQ[i][i] = childOiQ[i];

                    node.samples[i][i].addOiQFeature(samplesQiQ[i][i]);
                    node.samples[i][i].addQuerySizeToFeature(1);
                } else {
                    samplesQiQ[i][i] = 0;
                }
            }

            // compute cost for extend samples
            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < n && j - i < T; ++j) {
                    samplesQiQ[i][j] = samplesQiQ[i][j - 1] + samplesQiQ[j][j];
                    node.samples[i][j].addOiQFeature(samplesQiQ[i][j]);
                    node.samples[i][j].addQuerySizeToFeature(1);
                }
            }
        }

        return OiQ;
    }

    public double refiningSpaceCost(Node node) {
        double cost = node.cost;

        if (node.isLeafNode()) return cost;

        int n = node.childes.size();

        if (node.cost == 0) {
            return 0;
        }

        double workloadSize = cost / n;
        node.initSamples();
        // get child node cost
        for (int i = 0; i < n; ++i) {
            double childCost = refiningSpaceCost(node.childes.get(i));
            node.samples[i][i].setCost(childCost);
            cost += childCost;
        }

        // compute cost for extend samples
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n && j - i < T; ++j) {
                double cst = node.samples[i][j - 1].getCost() + node.samples[j][j].getCost() + D * workloadSize * (j - i + 1);
                node.samples[i][j].setCost(cst);
            }
        }

        return cost;
    }



    public void initialObjectTravelMap(List<Query> workload) {
        for (Query query : workload) {
            this.queryInitialObjectTravelMap(query);
        }
    }

    public void queryInitialObjectTravelMap(Query query) {
        Queue<Node> queue = new LinkedList<>();
        queue.add(root);

        while (!queue.isEmpty()) {
            Node node = queue.remove();
            if (node.isLeafNode()) {
                List<Point> points = node.leafNodeQuery(query);

                for (Point point : points) {
                    objectTraveledCountMap.put(point, objectTraveledCountMap.getOrDefault(point, 0) + 1);
                }
            } else {
                List<Node> resNode = new ArrayList<>();
                List<Node> subNodes = node.nonLeafNodeQuery(query, resNode);
                queue.addAll(subNodes);
                queue.addAll(resNode);
            }
        }
    }

    public void queryTest(List<Query> workload) {
        for (int i = 0; i < 2; ++i) {
            for (Query query : workload) {
                this.query(query);
//                this.basicQuery(query);
            }
        }
        long s = System.nanoTime();
        for (Query query : workload) {
            this.query(query);
        }
        long e = System.nanoTime();
        System.out.println("avg query time: " + (e - s) /1e6/ workload.size()  + "ms");
    }

    public List<Point> basicQuery(Query query) {
        List<Point> res = new ArrayList<>();
        Queue<Node> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            Node node = queue.remove();
            if (node.isLeafNode()) {
                List<Point> subRes = node.leafNodeQuery(query);
                res.addAll(subRes);
            } else {
                List<Node> subNodes = node.nonLeafNodeBasicQuery(query);
                queue.addAll(subNodes);
            }
        }
        return res;
    }


    public List<Point> query(Query query) {
        List<Point> res = new ArrayList<>();
        Queue<Node> queue = new LinkedList<>();
        queue.add(root);
        List<Node> resNode = new ArrayList<>();
        while (!queue.isEmpty()) {
            Node node = queue.remove();
            if (node.isLeafNode()) {

                List<Point> subRes = node.leafNodeQuery(query);
                res.addAll(subRes);

            } else {
                List<Node> subNodes = node.nonLeafNodeQuery(query, resNode);
                queue.addAll(subNodes);
            }
        }

        queue = new LinkedList<>(resNode);
        while (!queue.isEmpty()) {
            Node node = queue.remove();
            if (node.isLeafNode()) {res.addAll(node.dataset);}
            else {
                queue.addAll(node.childes);
            }
        }

        return res;
    }

    // Generate uniform queries: Generate uniform and random queries within MBR
    private static Query generateUniformQuery(Random random, Query MBR) {
        double[] pointMin = new double[D];
        double[] pointMax = new double[D];

        // generate a random minimum point
        for (int i = 0; i < D; i++) {
            pointMin[i] = MBR.pointMin.data[i] + random.nextDouble() * (MBR.pointMax.data[i] - MBR.pointMin.data[i]);
        }

        // Generate the maximum point of a random range (make sure it is within the MBR and greater than the minimum point)
        for (int i = 0; i < D; i++) {
            double minVal = pointMin[i];
            double maxRange = MBR.pointMax.data[i] - minVal;
            // random range up to 10% of the mbr boundary
            double range = QueryLength[i];
            pointMax[i] = minVal + range; // at least 1% range

            // make sure you don t exceed the mbr boundary
            if (pointMax[i] > MBR.pointMax.data[i]) {
                pointMax[i] = MBR.pointMax.data[i];
            }
        }

        return new Query(new Point(pointMin), new Point(pointMax));
    }


    // Generate empirical query: Add a Gaussian perturbation to an existing query
    private static Query generateEmpiricalQuery(List<Query> workload, Random random, Query MBR) {
        // A random selection of an existing query is used as a basis
        Query baseQuery = workload.get(random.nextInt(workload.size()));

        // create a new array of point coordinates
        double[] newPointMin = new double[D];
        double[] newPointMax = new double[D];

        // add a perturbation to the min point
        for (int i = 0; i < D; i++) {
            double stdDev = (MBR.pointMax.data[i] - MBR.pointMin.data[i]) * 0.05;
            double perturbedMin = baseQuery.pointMin.data[i] + random.nextGaussian() * stdDev;
            newPointMin[i] = Math.max(MBR.pointMin.data[i], Math.min(MBR.pointMax.data[i], perturbedMin));
        }

        // Add perturbation to max points (keep a reasonable range)
        for (int i = 0; i < D; i++) {
            double stdDev = (MBR.pointMax.data[i] - MBR.pointMin.data[i]) * 0.05;
            double perturbedMax = baseQuery.pointMax.data[i] + random.nextGaussian() * stdDev;
            double range = QueryLength[i];
            newPointMax[i] = newPointMin[i] + range;

            // make sure min max
            if (newPointMin[i] > newPointMax[i]) {
                double temp = newPointMin[i];
                newPointMin[i] = newPointMax[i];
                newPointMax[i] = temp;
            }

            // Make sure the query range is not too small (at least 1% of the MBR range)
            double minRange = (MBR.pointMax.data[i] - MBR.pointMin.data[i]) * 0.01;
            if (newPointMax[i] - newPointMin[i] < minRange) {
                newPointMax[i] = newPointMin[i] + minRange;
                // adjust if it goes beyond the boundary
                if (newPointMax[i] > MBR.pointMax.data[i]) {
                    newPointMax[i] = MBR.pointMax.data[i];
                    newPointMin[i] = MBR.pointMax.data[i] - minRange;
                }
            }
        }

        return new Query(new Point(newPointMin), new Point(newPointMax));
    }

    public List<Query> getScaleWorkload(DataSpace dataset, List<Query> workload) {

        // Calculate the number of queries that need to be replenished
        double spaceVolume = dataset.getMBR().computeVolume();

        if (spaceVolume <= StopExtendVolume)
            return null;

        int scaleSize = (int) Math.floor(Po * spaceVolume - workload.size());

        if (workload.isEmpty() && spaceVolume > StopExtendVolume) {
            scaleSize = Math.max(scaleSize, 1);
        }

        if (scaleSize < 0) {return null;}

        List<Query> newWorkload = new ArrayList<>();
        // calculate the current density ratio
        Random random = new Random();
        double currentDensity = workload.size() / spaceVolume;
        double densityRatio = currentDensity / Po;
        for (int i = 0; i < scaleSize; ++i) {
            Query newQuery;
            // Choose whether to generate empirical queries or uniform queries based on probability
            if (random.nextDouble() < densityRatio && !workload.isEmpty()) {
                newQuery = generateEmpiricalQuery(workload, random, dataset.getMBR());
            } else {
                newQuery = generateUniformQuery(random, dataset.getMBR());
            }
            newWorkload.add(newQuery);
        }

        return newWorkload;
    }




    //return MBR and the Node
    public Object[] recursiveBuildTree(DataSpace dataset, List<Query> workload) throws Exception {

//        System.out.println("once node construct--" + "dataset size: " + dataset.getDatasetSize() + "workload size: " + workload.size());
        List<Query> scaleWorkload = null;
        if (NeedExtendWorkload)
            scaleWorkload = getScaleWorkload(dataset, workload);


        Query MBR = new Query(dataset.minBound, dataset.maxBound);
        // need not scale workload, continue recursive build tree
        if (NeedExtendWorkload && scaleWorkload != null && !scaleWorkload.isEmpty()) {
            // scale workload, and build a new tree as a node
            if (useModel) {
                int a = 1;
            }
            insertedWorkload.addAll(scaleWorkload);
            workload.addAll(scaleWorkload);
            WalasTree subWalasTree = new WalasTree(dataset, workload, workload);
            return new Object[] {MBR, subWalasTree.root};

        } else {
            // partition space, get sub dataset, sub workload
            Object[] objects = workload.isEmpty() ? null : partitionSpace(dataset, workload);

            if (objects == null) { // need not be split, leaf node
//                double cost = D * dataset.getDatasetSize() * workload.size();

                Object[] subSpace = dataset.spaceAverageSplit();

                if (subSpace == null) {
                    double cost = getNodeCost(dataset.getDatasetSize(), workload.size());
                    Node leafNode = new Node(dataset.dataset, cost);
                    return new Object[] {MBR , leafNode};
                } else {

                    List<DataSpace> sub = (List<DataSpace>) subSpace[1];

                    List<Node> chdNodes = new ArrayList<>(sub.size());
                    List<Query> chdMBRs = new ArrayList<>(sub.size());
                    for (int i = 0; i < sub.size(); i++) {

                        if (sub.get(i).getDatasetSize() == 0) {
                            continue;
                        }

                        Object[] chdObject = recursiveBuildTree(sub.get(i), new ArrayList<>());

                        Query chdMBR = (Query)chdObject[0];
                        Node chdNode = (Node) chdObject[1];
                        chdNodes.add(chdNode);
                        chdMBRs.add(chdMBR);
                    }
                    double cost = getNodeCost(chdNodes.size(), workload.size());
                    Node nonLeafNode = new Node((int) subSpace[0], chdMBRs, chdNodes, cost);
                    return new Object[] {MBR, nonLeafNode};

                }



            } else {

                int splitDimension = (int)objects[0];
                List<DataSpace> subDataset = (List<DataSpace>)objects[1];
                List<List<Query>> subWorkload = (List<List<Query>>)objects[2];
                int chdSize = subDataset.size();
                List<Node> chdNodes = new ArrayList<>(chdSize);
                List<Query> chdMBRs = new ArrayList<>(chdSize);

                for (int i = 0; i < chdSize; i++) {

                    if (subDataset.get(i).getDatasetSize() == 0) {
                        continue;
                    }

                    Object[] chdObject = recursiveBuildTree(subDataset.get(i), subWorkload.get(i));

                    Query chdMBR = (Query)chdObject[0];
                    Node chdNode = (Node) chdObject[1];
                    chdNodes.add(chdNode);
                    chdMBRs.add(chdMBR);
                }

                double cost = getNodeCost(chdNodes.size(), workload.size());
//                double cost = D * chdNodes.size() * workload.size();
                Node nonLeafNode = new Node(splitDimension, chdMBRs, chdNodes, cost);

                return new Object[] {MBR, nonLeafNode};
            }
        }

    }

    public Object[] partitionSpace(DataSpace dataset, List<Query> workload) throws Exception {
//        System.out.println("entry once partition!");

//        long s = System.currentTimeMillis();
        // get split dimension, and split points
        Object[] splitDimensionAndSplits = getSplitDimensionAndSplits(dataset, workload);
//        long e = System.currentTimeMillis();
//        System.out.println("split time: " + (e - s) + "ms");

        if (splitDimensionAndSplits == null) {
            return null;
        }

        int splitDimension = (int) splitDimensionAndSplits[0];
        List<Double> splitPoints = (List<Double>) splitDimensionAndSplits[1];

        // get sub dataset, and sub workload
        List<DataSpace> subDataset = getSubDataset(dataset, splitPoints, splitDimension);
        List<List<Query>> subWorkload = getSubWorkload(workload, subDataset, splitDimension);

        List<DataSpace> effectSubDataset = new ArrayList<>();
        List<List<Query>> effectSubWorkload = new ArrayList<>();
        for (int i = 0; i < subDataset.size(); ++i) {
            if (subDataset.get(i).getDatasetSize() != 0) {
                effectSubDataset.add(subDataset.get(i));
                effectSubWorkload.add(subWorkload.get(i));
            }
        }
        if (effectSubDataset.size() <= 1) return null; // partition false

        return new Object[] {splitDimension, effectSubDataset, effectSubWorkload};
    }

    public static List<DataSpace> getSubDataset(DataSpace dataset, List<Double> splitPoints, int dimension) {
        int n = splitPoints.size() - 1;  // number of sub intervals
        List<DataSpace> subDataset = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            subDataset.add(new DataSpace());
        }

        // Assign each point to the corresponding sub-interval
        for (Point point : dataset.dataset) {
            double pointValue = point.data[dimension];

            // Use dichotomies to find the interval where the determination point is located
            int left = 0;
            int right = n - 1;
            int intervalIndex = -1;

            // Binary lookup: Find the first syncopation point position greater than or equal to the point value
            while (left <= right) {
                int mid = left + (right - left) / 2;
                if (pointValue <= splitPoints.get(mid)) {
                    right = mid - 1;
                } else if (pointValue > splitPoints.get(mid + 1)) {
                    left = mid + 1;
                } else {
                    intervalIndex = mid;
                    break;
                }
            }

            // deal with boundary situations
            if (intervalIndex == -1) {
                if (pointValue <= splitPoints.get(0)) {
                    intervalIndex = 0;  // Less than the first syncopation point, put into the first interval
                } else if (pointValue >= splitPoints.get(n)) {
                    intervalIndex = n - 1;  // Greater than the last syncopation point, put into the last interval
                }
            }

            // add points to the corresponding interval
            if (intervalIndex >= 0 && intervalIndex < n) {
                subDataset.get(intervalIndex).add(point);
            }
        }

        return subDataset;
    }

    public static List<List<Query>> getSubWorkload(List<Query> workload, List<DataSpace> dataspaces, int dimension) {
        List<List<Query>> result = new ArrayList<>();
        for (int i = 0; i < dataspaces.size(); i++) {
            result.add(new ArrayList<>()); // each dataspace corresponds to a subworkload
        }

        for (Query q : workload) {
            double qMin = q.pointMin.data[dimension];
            double qMax = q.pointMax.data[dimension];

            int left = findLeft(dataspaces, qMin, dimension);
            int right = findRight(dataspaces, qMax, dimension);
            if (left == -1 || right == -1 || left > right) continue;

            for (int i = left; i <= right; i++) {
                Query ds = new Query(dataspaces.get(i).minBound, dataspaces.get(i).maxBound);

                Query interQ = q.intersect(ds);
                if (interQ != null && !interQ.valueEqual(ds)) {
                    result.get(i).add(interQ);
                }
            }
        }
        return result;
    }

    public int getOptSplitDimension(DataSpace dataset, List<Query> workload) {

        double maxSigma = 0;
        int dimension = 0;

        for (int dim = 0; dim < D; ++dim) {
            List<Double> workloadPoints = getWorkloadPoint(dataset, workload, dim);
            int n = workloadPoints.size() - 1; // interval number
            int[] pointCount = new int[n];
            int[] queryCount = new int[n];

            // Step 1: point counting
            for (Point p : dataset.dataset) {
                double v = p.data[dim];
                int idx = binarySearchInterval(workloadPoints, v);
                if (idx >= 0 && idx < n) {
                    pointCount[idx]++;
                }
            }

            //step 2: query the count
            for (Query q : workload) {
                double qMin = q.pointMin.data[dim];
                double qMax = q.pointMax.data[dim];

                // Find the interval range that intersects [qMin, qMax]
                int left = binarySearchInterval(workloadPoints, qMin);
                int right = binarySearchInterval(workloadPoints, qMax);

                // Because qMax may fall on the right interval boundary
                if (right < n - 1 && workloadPoints.get(right + 1) <= qMax)
                    right++;

                for (int i = left; i <= right && i < n; i++) {
                    queryCount[i]++;
                }
            }

            // Step 3: cost calculation
            double[] cost = new double[n];
            double costSum = 0;
            for (int i = 0; i < n; i++) {
                cost[i] = (double) pointCount[i] * queryCount[i];
                costSum += cost[i];
            }

            double mu = 0;
            for (int i = 0; i < n; i++) {
                cost[i] = cost[i] / costSum;
                mu += cost[i];
            }
            mu = mu / n;

            double sigma2 = 0;
            for (int i = 0; i < n; i++) {
                sigma2 += Math.pow(cost[i] - mu, 2);
            }

            if (sigma2 > maxSigma) {
                maxSigma = sigma2;
                dimension = dim;
            }
        }

        return dimension;
    }

    public Object[] getSplitDimensionAndSplits(DataSpace dataset, List<Query> workload) throws Exception {
        if (Parameters.SplitDimensionOpt) {
            int splitDimension = getOptSplitDimension(dataset, workload);
            Object[] thePartition = datasetPartition(dataset, workload, splitDimension);
            if (thePartition == null) return null;
            List<Integer> splits = (List<Integer>) thePartition[0];
            return new Object[] {splitDimension, splits};
        } else {
            int splitDimension = -1;
            List<Integer> splits = null;
            double minCost = Double.MAX_VALUE;
            for (int dim = 0; dim < D; ++dim) {
                Object[] thePartition = datasetPartition(dataset, workload, dim);
                if (thePartition == null) {continue;}

                List<Integer> theSplits = (List<Integer>) thePartition[0];
                double theCost = (Double) thePartition[1];
                if (minCost > theCost) {
                    minCost = theCost;
                    splitDimension = dim;
                    splits = theSplits;
                }
            }
            if (splitDimension == -1) {return null;}
            return new Object[] {splitDimension, splits};
        }

    }


    public Object[] datasetPartition(DataSpace dataset, List<Query> workload, int dimension) throws Exception {
        long s, e;

        // 1. get boundaries of queries in workload
        List<Double> workloadPoints = getWorkloadPoint(dataset, workload, dimension);

        // 2. computing cost[i][j]
        Object[] costRes = getSubSpaceCost(dataset, workload, workloadPoints, dimension);
        double[][] cost = (double[][]) costRes[0]; // real cost, model is modeled cost, no model is computed cost
        double[][] compareCost = (double[][]) costRes[1];

        // 3. processing DP

        return DPProcessing(workloadPoints, cost, compareCost, workload);
//        System.out.println(splitPoints.toString());
    }

    public Object[] DPProcessing(List<Double> workloadPoints, double[][] cost, double[][] compareCost, List<Query> workload) {
        int n = workloadPoints.size();
        double[][] optCost = new double[n][n];
        int[][] path = new int[n][n];
        double[] levelCost = new double[n];
        for (int i = 0; i < n; i++) {
            Arrays.fill(optCost[i], Double.MAX_VALUE);
            Arrays.fill(path[i], 0);
        }
        for (int i = 0; i < n; i++) {
            optCost[i][0] = cost[0][i];
            path[i][0] = 0;
        }
        levelCost[0] = optCost[n - 1][0];

        int latency = 0; int round = 0;
        for (int j = 1; j < n; j++) {
            for (int i = j; i < n; i++) {  // i starts with j because at least j+1 points are needed to cut the j segment
                // Enumerate the starting point k of the last paragraph
                for (int k = j - 1; k < i; k++) {
                    if (optCost[k][j - 1] == Double.MAX_VALUE) {
                        continue;
                    }
                    double currentCost = 0;

                    currentCost = cost[k][i] + optCost[k][j - 1];

                    if (currentCost < optCost[i][j]) {
                        optCost[i][j] = currentCost;
                        path[i][j] = k;
                    }
                }
            }

            levelCost[j] = optCost[n - 1][j] + getParentNodeCost(workload.size(), j + 1);

            // Early Stop: Add the relative return of the split < rate
            double relativeGain = (levelCost[j - 1] - levelCost[j]) / levelCost[j - 1];
//            if (relativeGain < rate) {
//                break;
//            }

            if (relativeGain < rate && latency > 10) {
                break;
            } else {
                if (relativeGain < rate) {
                    latency++;
                } else {
                    latency = 0;
                }
            }
            round ++;
        }

        double minCost = Double.MAX_VALUE;
        int splitCount = -1;
        for (int j = round; j >= 0; j--) {
            if (j == 0 && useModel) break;
            if (levelCost[j] < minCost) {
                minCost = levelCost[j];
                splitCount = j;
            }
        }

        boolean arriveLeaf = (compareCost != null && compareCost[0][n-1] <= minCost);
        if (splitCount <= 0 || arriveLeaf) {

            if (useModel && minCost != compareCost[0][n - 1]) {
                int a = 1;
            }

            return null; // need not split
        }

        //4. Generate syncopation points backward
        List<Double> splitPoints = new ArrayList<>();
        int currentIndex = n - 1;
        for (int j = splitCount; j >= 0; j--) {
            splitPoints.add(workloadPoints.get(currentIndex));
            currentIndex = path[currentIndex][j];
        }
        if (splitPoints.get(splitPoints.size() - 1) != workloadPoints.get(0)) {
            splitPoints.add(workloadPoints.get(0));  // Added the left boundary
        }
        Collections.reverse(splitPoints);  // the inversion yields an ascending arrangement

        // make sure to include the right boundary
        if (splitPoints.get(splitPoints.size() - 1) != workloadPoints.get(n - 1)) {
            splitPoints.add(workloadPoints.get(n - 1));
        }

        if (compareCost != null && Objects.equals(splitPoints.get(0), workloadPoints.get(0)) && Objects.equals(splitPoints.get(1), workloadPoints.get(workloadPoints.size() - 1))) {
            return DPProcessing(workloadPoints, compareCost, null, workload);
        }

        return new Object[]{splitPoints, minCost};
    }



    public int getParentNodeCost (int workloadSize, int chdSize) {
        if (chdSize == 0) return 0;
        return D * workloadSize * chdSize;
    }

    public Object[] getSubSpaceCost(DataSpace dataset, List<Query> workload, List<Double> workloadPoints, int dimension) throws Exception {
        long s, e;
        int n = workloadPoints.size();
        ModelSample[][] modelSamples = new ModelSample[n][n];
        if (useModel) {
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    modelSamples[i][j] = new ModelSample();
                }
            }
        }

        // 1. construct prefix array sumPoint，sumPoint[i]: >= workloadPoints[i] points
        int[] sumPoint = new int[n];
        int[] sumObjectTravelCount = new int[n];// used for extract model feature

        // create multiple datarectangles in bulk
        Query[] MBRs = new Query[n];
        for (int i = 0; i < MBRs.length; i++) {
            MBRs[i] = new Query(); // create a two dimensional rectangle
        }

        for (Point point : dataset.dataset) {
            double val = point.data[dimension];
            // find label that first >= val
            int k = Collections.binarySearch(workloadPoints, Math.ceil(val));
            if (k < 0) {
                k = - k - 1;  //if not return to the insertion position
            }
            MBRs[k].addPoint(point.data);
            // from position k add 1 to all elements
            for (int i = k; i < n; i++) {
                sumPoint[i]++;
            }
            if (useModel) {
                Integer ct = objectTraveledCountMap.getOrDefault(point, 0);
                for (int i = k; i < n; i++) {
                    sumObjectTravelCount[i] += ct;
                }
            }
        }

//        s = System.currentTimeMillis();
        // 2. Construct pointCount[i][j] to represent the number of data points in the interval [i, j]; MBRGrid[i,j] represents the MBR of the interval [i, j]
        int[][] pointCount = new int[n][n];
        Query[][] MBRGrid = new Query[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                MBRGrid[i][j] = new Query();
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                pointCount[i][j] = sumPoint[j] - sumPoint[i];
                if (j == i + 1) {
                    MBRGrid[i][j] = MBRs[j];
                } else {
                    MBRGrid[i][j] = MBRs[j].merge(MBRGrid[i][j - 1]);
                }
            }
        }
//        e = System.currentTimeMillis();
//        System.out.println("part 2 time: " + (e - s) + "ms");

        // 3. Calculate queryCount[i][j]: How many query intervals are covered [i, j)
        int[][] queryCount = countQueryIntersections(MBRGrid, workload, dimension);

        // 4. compute dp cost，cost[i][j] = pointCount[i][j] * queryCount[i][j]
        double[][] cost = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                cost[i][j] = D * pointCount[i][j] * queryCount[i][j];
            }
        }

        if (!useModel)
            return new Object[]{cost, cost};

        //5. get features and cost for model
        double[][] modelCost = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                modelSamples[i][j].setOiQFeature(sumObjectTravelCount[j] - sumObjectTravelCount[i]);
                modelSamples[i][j].setVolumeToFeature(MBRGrid[i][j]);
                modelSamples[i][j].setPerimeterToFeature(MBRGrid[i][j]);
                modelSamples[i][j].setQuerySizeToFeature(queryCount[i][j]);
                modelSamples[i][j].setObjectSizeToFeature(pointCount[i][j]);
            }
        }

//        s = System.currentTimeMillis();
        // 6. combine model prediction
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                // prediction
                Instance instance = new DenseInstance(1, modelSamples[i][j].features);
                instance.setDataset(instances);
                modelCost[i][j] = randomForest.classifyInstance(instance);
            }
        }
//        modelCost = batchPredictModelCost(modelSamples);
//        e = System.currentTimeMillis();
//        System.out.println("part 6 time: " + (e - s) + "ms" + ", size n: " + n);

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                // combine cost
                modelCost[i][j] = cost[i][j] - ((Math.log(1 + iteration)) / (Math.log(1 + K))) * (cost[i][j] - modelCost[i][j]);
            }
        }

        return new Object[]{modelCost, useModel ? cost : null};
    }

    public double[][] batchPredictModelCost(ModelSample[][] modelSamples) throws Exception {

        int n = modelSamples.length;
        double[][] modelCost = new double[n][n];
        Instances batchData = new Instances(instances);
        // 1. Collect all samples that need to be predicted
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                Instance ins = new DenseInstance(1.0, modelSamples[i][j].features);
                ins.setDataset(batchData);
                batchData.add(ins);
            }
        }

        //2. Batch forecasting
        Evaluation eval = new Evaluation(batchData);
        double[] predictions = eval.evaluateModel(randomForest, batchData);

        // 3. Map the prediction back to modelCost[i][j]
        int idx = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                modelCost[i][j] = predictions[idx++];
            }
        }

        return modelCost;
    }


    public static int[][] countQueryIntersections(Query[][] MBRGrid, List<Query> workload, int dimension) {
        int n = MBRGrid.length;
        int[][] queryCount = new int[n][n];

        // Extract the one-dimensional coordinate sequence in advance to facilitate binary divisions
        double[] starts = new double[n];
        double[] ends = new double[n];
        starts[0] = -Double.MIN_VALUE;
        ends[0] = -Double.MIN_VALUE;
        for (int i = 0; i < n-1; i++) {
            starts[i + 1] = MBRGrid[i][i+1].pointMin.data[dimension];
            if (starts[i + 1] == Double.MAX_VALUE) {
                starts[i + 1] = starts[i];
            }
        }
        for (int j = 1; j < n; j++) {
            ends[j] = MBRGrid[0][j].pointMax.data[dimension];
            if (ends[j] == Double.MAX_VALUE) {
                ends[j] = ends[j - 1];
            }
        }

        // iterate through each query
        for (Query q : workload) {
            double qMin = q.pointMin.data[dimension];
            double qMax = q.pointMax.data[dimension];

            // The left and right boundaries of the two-part positioning
            int left = lowerBound(ends, qMin);     // the first j which may intersect
            int right = upperBound(starts, qMax);  // the last i that may intersect
            if (left == -1 || right == -1) continue;

            for (int i = 0; i <= right && i < n; i++) {
                for (int j = left; j < n; j++) {
                    Query cell = MBRGrid[i][j];
                    if (cell == null) continue;
                    if (q.checkRelation(cell)) {
                        queryCount[i][j]++;
                    }
                }
            }
        }

        return queryCount;
    }

    private static List<Double> getWorkloadPoint(DataSpace dataset, List<Query> querySet, int dimension) {
        Set<Double> pointSet = new HashSet<>();

        // get boundaries
        double min_id = dataset.minBound.data[dimension] - 1;
        pointSet.add(min_id);
        double max_id = dataset.maxBound.data[dimension];
        pointSet.add(max_id);

        // add query boundaries
        for (Query query : querySet) {
            if(max_id > query.pointMin.data[dimension] && query.pointMin.data[dimension] > min_id){
                pointSet.add(query.pointMin.data[dimension] - 1);
            }
            if(min_id < query.pointMax.data[dimension] && query.pointMax.data[dimension] < max_id){
                pointSet.add(query.pointMax.data[dimension]);
            }
        }

        //sort
        List<Double> workloadPoints = new ArrayList<>(pointSet);
        Collections.sort(workloadPoints);
        return workloadPoints;
    }

}
