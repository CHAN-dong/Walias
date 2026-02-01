package index;

import cost_model.ABC;
import data_structure.*;
import utils.CostPlot;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

import static data_structure.Parameters.*;
import static cost_model.ABC.fitAlphaBetaGammaFromLogs;
import static index.Node.getLeafNodeCost;
import static index.Node.getNonLeafNodeCost;
import static utils.DatasetVisualizer.drawForPartition;
import static utils.DatasetVisualizer.exportHighResImage;
import static utils.Tools.*;

public class Walias {
    static int cores = Runtime.getRuntime().availableProcessors();
    Instances instances; // samples for training

    boolean useModel = false;
    List<Query> workload;

    static double[] workloadLen;

    public List<Query> testWorkload;

    DataSpace dataset;

    RandomForest randomForest;
    int iteration = 0;

    public Node root;

    public double treeCost;

    static List<Query> insertedWorkload = new ArrayList<>();

    static HashMap<Point, Integer> objectTraveledCountMap = new HashMap<>();

    public Walias(DataSpace dataset, List<Query> workload, List<Query> testWorkload) throws Exception {
        this.dataset = dataset;
        this.workload = workload;
        this.testWorkload = testWorkload;
        workloadLen = computeWorkloadAvgLen(workload);

        Object[] objects = recursiveBuildTree(dataset, workload, 0);
        root = (Node) objects[1];
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

    public double[] computeWorkloadAvgLen(List<Query> workload) {
        // Returns workloadLen[D], where workloadLen[d] = average (max[d] - min[d]) over queries.
        // Assumes Query.min.data and Query.max.data are double[D].
        if (D <= 0) throw new IllegalArgumentException("D must be > 0");
        double[] sum = new double[D];
        int cnt = 0;

        if (workload == null) {
            return sum; // all zeros
        }

        for (Query q : workload) {
            if (q.pointMin.data.length < D || q.pointMax.data.length < D) {
                throw new IllegalArgumentException("Query dimension < D");
            }
            for (int d = 0; d < D; d++) {
                double a = q.pointMin.data[d];
                double b = q.pointMax.data[d];
                sum[d] += Math.abs(b - a);
            }
            cnt++;
        }

        if (cnt == 0) return new double[D];

        for (int d = 0; d < D; d++) {
            sum[d] /= cnt;
        }
        return sum;
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
        stats = new Stats();
        queryTest(testWorkload);
        stats = new Stats();
        queryTest(testWorkload);
        queryForCost(testWorkload);
        printStats();
    }


    public void printStats () {

        System.out.println("nonLeaf node info-----------------------------");
        for (Long[] tm : stats.nonLeafNode) {
            // time + objects size + candidate size + res size
            System.out.println(Arrays.toString(tm));
        }

        System.out.println("leaf node info-----------------------------");
        for (Long[] tm : stats.leafNode) {
            // time + objects size + candidate size + res size
            System.out.println(Arrays.toString(tm));
        }

        System.out.println("node visited info-----------------------------");
        for (Long[] tm : stats.nodeVisitedInfo) {
            // time + objects size + candidate size + res size
            System.out.println(Arrays.toString(tm));
        }

        ABC abc = fitAlphaBetaGammaFromLogs(stats.leafNode, stats.nonLeafNode, D, true, 1e-3);
        System.out.println("alpha(us): " + abc.alpha);
        System.out.println("beta(us): " + abc.beta);
        System.out.println("gamma(us): " + abc.gamma);

        System.out.println("nonLeaf node info-----------------------------");
//        for (Long[] tm : stats.nonLeafNode) {
//            // time + objects size + candidate size + res size
//            System.out.println("real time " + tm[0] / 1000+ ", --- predicted time: " + (abc.alpha + abc.beta * Math.log(tm[1]) + abc.gamma *  (D - 1) * tm[2]));
//        }
//
//        System.out.println("leaf node info-----------------------------");
//        for (Long[] tm : stats.leafNode) {
//            // time + objects size + candidate size + res size
//            System.out.println(Arrays.toString(tm));
//            System.out.println("real time " + tm[0] / 1000+ ", --- predicted time: " + (abc.alpha + abc.beta * Math.log(tm[1]) + abc.gamma * (D - 1) * tm[2]));
//        }
//
//        System.out.println("node visited info-----------------------------");
//        for (Long[] tm : stats.nodeVisitedInfo) {
//            System.out.println(Arrays.toString(tm));
//            System.out.println("real time " + tm[0] / 1000+ ", --- predicted time: " + (tm[1] * abc.alpha));
//        }


//        reportErrors(stats.leafNode, stats.nonLeafNode, abc.alpha, abc.beta, abc.gamma, D);


    }

    public void getIndexSize() {
        long size = 0;
        Queue<Node> queue = new LinkedList<>();
        queue.add(root);

        while (!queue.isEmpty()) {
            Node remove = queue.remove();

            size += 20;

            if (!remove.isLeafNode()) {
                size += (8L * D + 8) * remove.childes.size();
                queue.addAll(remove.childes);
            }
        }

        System.out.println("index size: " + size / 1024.0 / 1024.0 + "MB");
    }


    public void startRefine(int round) throws Exception {

        List<List<double[]>> count_costs = new ArrayList<>();

        long s, e;
        int cores = Runtime.getRuntime().availableProcessors();
        Parameters.K = round;
        useModel = true;
        initialObjectTravelMap(workload);
        System.out.println("----------before refining performance: ");
        printIndexPerformance();
        System.out.println();

        count_costs.add(Walias.count_cost);
//        CostPlot.plot(WalasTree.count_cost, "Cost Plot");
        Walias.count_cost = new ArrayList<>();
        firstDP = true;

//        drawPartition(iteration);

//        Node preNode = root;
        for (iteration = 1; iteration <= K; ++iteration) {
            System.out.println("------------iteration: " + iteration + " Start!!!");
            queryForCostInit(testWorkload);

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
            root = (Node) recursiveBuildTree(dataset, workload,0)[1];
            e = System.currentTimeMillis();
            System.out.println("tree rebuild complete, consume time: " + (e - s) / 1000.0 + "s");

            // query performance test
            printIndexPerformance();

            System.out.println();

            drawPartition(iteration);

            count_costs.add(Walias.count_cost);
//        CostPlot.plot(WalasTree.count_cost, "Cost Plot");
            firstDP = true;
            Walias.count_cost = new ArrayList<>();
        }

        CostPlot.plot(count_costs, "Cost Plot");
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
            List<Point> subRes = new ArrayList<>();
            node.leafNodeQuery(query, subRes);
            return subRes.size();
        }
        int OiQ = 0;
        int n = node.childes.size();

        int[] childOiQ = new int[n];
        for (int i = 0; i < n; ++i) {
            if (query.hasIntersection(node.MBRs.get(i), node.splitDimension)) {
                childOiQ[i] = queryTreeForOiQ_QFeature(node.childes.get(i), query);
                OiQ += childOiQ[i];
            }
        }

        if (node.samples != null) {
            int[][] samplesQiQ = new int[n][n];
            // get child node QiQ
            for (int i = 0; i < n; ++i) {
                if (query.hasIntersection(node.MBRs.get(i), node.splitDimension)) {
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
                double cst = node.samples[i][j - 1].getCost() + node.samples[j][j].getCost() + SIGMA * (j - i + 1);
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
                List<Point> points = new ArrayList<>();
                node.leafNodeQuery(query, points);

                for (Point point : points) {
                    objectTraveledCountMap.put(point, objectTraveledCountMap.getOrDefault(point, 0) + 1);
                }
            } else {
                node.nonLeafNodeQuery(query, queue);
            }
        }
    }

    public void queryForCost(List<Query> workload) {

        long sum = 0;

        for (Query query : workload) {
            long nodeVisits = 0;
            List<Point> res = new ArrayList<>();
            Queue<Node> queue = new LinkedList<>();
            queue.add(root);
            while (!queue.isEmpty()) {
                nodeVisits++;
                Node node = queue.remove();
                if (node.isLeafNode()) {
                    node.leafNodeQuery(query, res);
                    Long[] longs = stats.leafNode.get(stats.leafNode.size() - 1);
                    sum += BETA * Math.log(node.dataset.size()) + GAMA * longs[1];
                } else {
                    node.nonLeafNodeQuery(query, queue);
                    Long[] longs = stats.nonLeafNode.get(stats.nonLeafNode.size() - 1);
                    sum += BETA * Math.log(node.childes.size()) + GAMA * longs[1];
                }
            }
            sum += nodeVisits * ALFA;
        }

        System.out.println("query cost: " + sum / 1e6/workload.size() + "ms---------");

    }


    public void queryTest(List<Query> workload) {
//        for (int i = 0; i < 2; ++i) {
//            for (Query query : workload) {
//                this.query(query);
////                this.basicQuery(query);
//            }
//        }
        long s = System.nanoTime();
        for (Query query : workload) {
            this.query(query);
//            break;
        }
        long e = System.nanoTime();
        System.out.println("avg query time: " + (e - s) /1e6/ workload.size()  + "ms");
    }

    public void queryForCostInit(List<Query> workload) {
        long s, e;
        for (Query query : workload) {
                List<Point> res = new ArrayList<>();
                Queue<Node> queue = new LinkedList<>();
                queue.add(root);
                while (!queue.isEmpty()) {
                    Node node = queue.remove();
                    if (node.isLeafNode()) {
                        s = System.nanoTime();
                        node.leafNodeQuery(query, res);
                        e = System.nanoTime();
                    } else {
                        s = System.nanoTime();
                        node.nonLeafNodeQuery(query, queue);
                        e = System.nanoTime();
                    }
                    node.cost += e - s;
                }
        }
    }

    static Stats stats = new Stats();

    public static Long TMP = 0L;
    public List<Point> query(Query query) {
        long nodeVisits = 0;
        long s = System.nanoTime();
        List<Point> res = new ArrayList<>();
        Queue<Node> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            nodeVisits++;
            Node node = queue.remove();
            if (node.isLeafNode()) {
                node.leafNodeQuery(query, res);
            } else {
                node.nonLeafNodeQuery(query, queue);
            }
        }

        long e = System.nanoTime();
        stats.nodeVisitedInfo.add(new Long[]{(e - s) - TMP , nodeVisits});
        TMP = 0L;
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

    public static int[] chooseSplits(DataSpace dataset) {
        // Returns splits[D] using the closed-form:
        //   h* = ((alfa * Vq * V) / (beta * D * N * SA))^(1/(D+1))
        //   splits[x] = ceil(Lx / h*)
        //
        // Assumptions:
        // - dataset bounds define MBR: [minBound, maxBound]
        // - points are (approximately) uniformly distributed inside the MBR
        // - queries appear at random positions (not necessarily fully inside MBR)
        // - boundary cells require filtering: alfa + beta*D*|O|
        // - inner cells cost ~ alfa (output cost ignored as constant)

        double[] minB = dataset.minBound.data;
        double[] maxB = dataset.maxBound.data;

        if (minB.length != D || maxB.length != D) {
            throw new IllegalArgumentException("Dimension mismatch: len.length != bounds dimension.");
        }
        long N = (dataset.dataset == null) ? 0L : dataset.dataset.size();
        if (N <= 0) {
            // No objects: trivial splits
            int[] splits = new int[D];
            Arrays.fill(splits, 1);
            return splits;
        }
        if (ALFA <= 0 || BETA <= 0) {
            throw new IllegalArgumentException("alfa and beta must be > 0.");
        }

        // Lx, ell_x
        double[] L = new double[D];
        double[] ell = new double[D];
        double V = 1.0;
        double Vq = 1.0;

        for (int d = 0; d < D; d++) {
            L[d] = Math.max(0.0, maxB[d] - minB[d]);
            // If the MBR length is 0 in this dimension, it contributes nothing (degenerate)
            // We'll handle it later by forcing splits[d]=1
            if (L[d] == 0.0) {
                ell[d] = 0.0;
                continue;
            }
            double w = Math.max(0.0, workloadLen[d]);
            // Expected overlap length inside MBR: ell = L*w/(L+w)
            ell[d] = (w == 0.0) ? 0.0 : (L[d] * w) / (L[d] + w);

            V *= L[d];
            Vq *= Math.max(1e-12, ell[d]); // avoid Vq becoming exactly 0 (still yields very small h*)
        }

        // If V is zero (degenerate MBR), just return 1 splits.
        if (V == 0.0) {
            int[] splits = new int[D];
            Arrays.fill(splits, 1);
            return splits;
        }

        // SA = 2 * sum_i prod_{j!=i} ell_j   (hyper-rectangle "surface area" of overlap region)
        double SA = 0.0;
        for (int i = 0; i < D; i++) {
            double prod = 1.0;
            for (int j = 0; j < D; j++) {
                if (j == i) continue;
                prod *= Math.max(1e-12, ell[j]);
            }
            SA += prod;
        }
        SA *= 2.0;

        // If SA is ~0 (e.g., ell too small in many dims), fallback to a conservative h*
        // Here we set SA to a tiny value to avoid division by zero, which will make h* larger
        // (=> fewer splits), matching the idea that tiny windows shouldn't over-split.
        SA = Math.max(SA, 1e-12);

        // h* = ((alfa * Vq * V) / (beta * D * N * SA))^(1/(D+1))
        double numerator = ALFA * Vq * V;
        double denominator = BETA * D * (double) N * SA;
        double ratio = numerator / Math.max(denominator, 1e-30);

        double hStar = Math.pow(Math.max(ratio, 1e-30), 1.0 / (D + 1.0));

        // Convert to splits
        int[] splits = new int[D];
        for (int d = 0; d < D; d++) {
            if (L[d] == 0.0) {
                splits[d] = 1;
                continue;
            }
            int s = (int) Math.ceil(L[d] / Math.max(hStar, 1e-12));
            splits[d] = Math.max(1, s);
        }

        return splits;
    }

    public static List<Cell> buildUniformCells(DataSpace dataset, int[] splits) {
        double[] minB = dataset.minBound.data;
        double[] maxB = dataset.maxBound.data;

        // stride & cellCount
        int[] stride = new int[D];
        stride[0] = 1;
        int cellCount = 1;
        for (int d = 0; d < D; d++) {
            if (splits[d] <= 0) throw new IllegalArgumentException("splits[d] must be > 0");
            cellCount = Math.multiplyExact(cellCount, splits[d]);
            if (d > 0) stride[d] = Math.multiplyExact(stride[d - 1], splits[d - 1]);
        }

        // step
        double[] step = new double[D];
        for (int d = 0; d < D; d++) {
            double range = maxB[d] - minB[d];
            step[d] = (range == 0.0) ? 0.0 : (range / splits[d]);
        }

        // init cells
        List<Cell> cells = new ArrayList<>(cellCount);
        for (int i = 0; i < cellCount; i++) cells.add(new Cell());

        // assign points
        if (dataset.dataset == null) return cells;

        int[] idx = new int[D];
        for (Point p : dataset.dataset) {
            for (int d = 0; d < D; d++) {
                if (step[d] == 0.0) {
                    idx[d] = 0;
                } else {
                    int id = (int) Math.floor((p.data[d] - minB[d]) / step[d]);
                    if (id < 0) id = 0;
                    if (id >= splits[d]) id = splits[d] - 1; // maxBound 落最后一格
                    idx[d] = id;
                }
            }
            int lin = 0;
            for (int d = 0; d < D; d++) lin += idx[d] * stride[d];

            cells.get(lin).dataset.add(p);
        }

        return cells;
    }

//    public static void queryOnCells(Query query, Query MBR, int[] splits, List<Cell> cells,  List<Point> res) {
//        double[] minB = MBR.pointMin.data;
//        double[] maxB = MBR.pointMax.data;
//
//        // stride & sanity
//        int[] stride = new int[D];
//        stride[0] = 1;
//        int cellCount = 1;
//        for (int d = 0; d < D; d++) {
//            cellCount = Math.multiplyExact(cellCount, splits[d]);
//            if (d > 0)  stride[d] = Math.multiplyExact(stride[d - 1], splits[d - 1]);
//        }
//        if (cells.size() != cellCount) {
//            throw new IllegalArgumentException("cells.size() != product(splits)");
//        }
//
//        // step
//        double[] step = new double[D];
//        for (int d = 0; d < D; d++) {
//            double range = maxB[d] - minB[d];
//            step[d] = (range == 0.0) ? 0.0 : (range / splits[d]);
//        }
//
//        // helper: clamp cell id
//        java.util.function.BiFunction<Double, Integer, Integer> toCellId = (value, d) -> {
//            if (step[d] == 0.0) return 0;
//            int id = (int) Math.floor((value - minB[d]) / step[d]);
//            if (id < 0) id = 0;
//            if (id >= splits[d]) id = splits[d] - 1;
//            return id;
//        };
//
//        // compute [lo, hi] per dimension (cells overlapped by query)
//        int[] lo = new int[D];
//        int[] hi = new int[D];
//        for (int d = 0; d < D; d++) {
//            int l = toCellId.apply(query.pointMin.data[d], d);
//            int r = toCellId.apply(query.pointMax.data[d], d);
//            if (l > r) { int t = l; l = r; r = t; }
//            lo[d] = l;
//            hi[d] = r;
//        }
//
//        // helpers: point in query / query contains cell bounds
//        java.util.function.Predicate<Point> pointInQuery = (p) -> {
//            for (int d = 0; d < D; d++) {
//                double x = p.data[d];
//                if (x < query.pointMin.data[d] || x > query.pointMax.data[d]) return false;
//            }
//            return true;
//        };
//
//        // enumerate all cells in [lo..hi]
//        int[] cur = lo.clone();
//        enumerateCellsRec(0, D, lo, hi, cur, (cellIdx) -> {
//            int lin = 0;
//            for (int d = 0; d < D; d++) lin += cellIdx[d] * stride[d];
//
//            Cell cell = cells.get(lin);
//            if (cell.dataset == null || cell.dataset.isEmpty()) return;
//
//            // compute cell bounds
//            double[] cMin = new double[D];
//            double[] cMax = new double[D];
//            for (int d = 0; d < D; d++) {
//                double mn = minB[d] + cellIdx[d] * step[d];
//                double mx;
//                if (step[d] == 0.0) {
//                    mx = maxB[d];
//                } else {
//                    mx = minB[d] + (cellIdx[d] + 1) * step[d];
//                    if (cellIdx[d] == splits[d] - 1) mx = maxB[d]; // 对齐末端
//                }
//                cMin[d] = mn;
//                cMax[d] = mx;
//            }
//
//            boolean fullCover = true;
//            for (int d = 0; d < D; d++) {
//                if (query.pointMin.data[d] > cMin[d] || query.pointMax.data[d] < cMax[d]) {
//                    fullCover = false;
//                    break;
//                }
//            }
//
//            if (fullCover) {
//                // 中间完全覆盖：直接加入
//                res.addAll(cell.dataset);
//            } else {
//                long rsz = res.size();
//                long s = System.nanoTime();
//                // 外围：线性过滤
//                for (Point p : cell.dataset) {
//                    if (pointInQuery.test(p)) res.add(p);
//                }
//                long e = System.nanoTime();
//                stats.cellNode.add(new Long[]{(e - s), (long)cell.dataset.size(), (long) res.size() - rsz});
//            }
//        });
//    }

    private static void enumerateCellsRec(
            int dim, int D,
            int[] lo, int[] hi, int[] cur,
            java.util.function.Consumer<int[]> visitor
    ) {
        if (dim == D) {
            visitor.accept(cur.clone());
            return;
        }
        for (int v = lo[dim]; v <= hi[dim]; v++) {
            cur[dim] = v;
            enumerateCellsRec(dim + 1, D, lo, hi, cur, visitor);
        }
    }


    private static int chooseSplitCountForNullWorkload(int objN, double Ld, double ld, double Ld_, double ld_, int expectedQueries) {
        // 边界保护
        if (Ld <= 0 || objN <= 1 || ld <= 0 || Ld_ <= 0 || ld_ <= 0) return 0;
        int Qtilde = Math.max(expectedQueries, 1);

        double best = Double.POSITIVE_INFINITY;
        int bestM = 0;

        for (int m = 0; m <= 1000; m++) {
            int C = m + 1;        // number of children
            double a = Math.max(Ld / C, 1);     // avg child length

            // (i) expected accessed children
            double Eacc = 1.0 + (ld * Ld / (ld + Ld) / a);     // = 1 + C*avgQueryLen/L
            if (Eacc > C) Eacc = C;

            // (ii) expected query cost
            double qCostC = Eacc * (ALFA + BETA * Math.log((double) (objN) / C) + GAMA * (objN * ld_ * Ld_) / ((m + 1) * (Ld_ + ld_)));
            double qCostP = ALFA + BETA * Math.log(C) + GAMA * (D - 1) * Eacc;

            double sCost = SIGMA * (8 * D + 28) * m;

            double total = Qtilde * (qCostC + qCostP) + SIGMA * sCost;

            if (total > best) {
                break;
            }
            best = total;
            bestM = m;
        }
        return bestM;
    }




    //return MBR and the Node
    public Object[] recursiveBuildTree(DataSpace dataset, List<Query> workload, int expectedQueries) throws Exception {

        Query MBR = new Query(dataset.minBound, dataset.maxBound);

        if ((workload == null || workload.size() == 0)) { // leaf node

            if (STRATEGY_II) {
                final int dim = getOptSplitDimension(dataset, workload);

                double Ld = dataset.maxBound.data[dim] - dataset.minBound.data[dim];
                double ld = QueryLength[dim];

                int dim_ = (dim + 1) % D;
                double Ld_ = dataset.maxBound.data[dim_] - dataset.minBound.data[dim_];
                double ld_ = QueryLength[dim_];

                int m = chooseSplitCountForNullWorkload(dataset.dataset.size(), Ld, ld, Ld_, ld_, expectedQueries);

                if (m > 0) {
                    List<Double> splitPoints = new ArrayList<>();
                    double step = Ld / (m + 1);
                    for (int i = 0; i <= m + 1; i++) {
                        splitPoints.add(dataset.minBound.data[dim] + i * step);
                    }

                    List<DataSpace> subDataset = getSubDataset(dataset, splitPoints, dim);
                    List<Node> chdNodes = new ArrayList<>();
                    List<Query> chdMBRs = new ArrayList<>();

                    for (DataSpace subData : subDataset) {
                        if (subData.getDatasetSize() == 0) continue;
                        Object[] chdObject = recursiveBuildTree(subData, null, 1);
                        chdMBRs.add((Query) chdObject[0]);
                        chdNodes.add((Node) chdObject[1]);
                    }

//                    double cost = getNonLeafNodeCost(chdNodes.size(), 1, 1, chdNodes.size());
                    Node nonLeafNode = new Node(dim, chdMBRs, chdNodes, 0);
                    return new Object[] {MBR, nonLeafNode};
                }
            }

            final int dim = getOptSplitDimension(dataset, workload);
            dataset.dataset.sort((a, b) -> Double.compare(a.data[dim], b.data[dim]));

//            double cost = getLeafNodeCost(dataset.dataset, workload, dim);

            Node leafNode = new Node(dataset.dataset, 0, dim);
            return new Object[] {MBR , leafNode};
        }


        // partition space, get sub dataset, sub workload
        Object[] objects = partitionSpace(dataset, workload);

        if (objects == null) { // need not be split, leaf node

            int splitDim = getOptSplitDimension(dataset, workload);
            final int dim = splitDim;
            dataset.dataset.sort((a, b) -> Double.compare(a.data[dim], b.data[dim]));

//            double cost = getLeafNodeCost(dataset.dataset, workload, dim);

            Node leafNode = new Node(dataset.dataset, 0, splitDim);
            return new Object[] {MBR , leafNode};
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
                Object[] chdObject = null;
                if (subWorkload.get(i) == null || subWorkload.get(i).size() == 0) {
                    chdObject = recursiveBuildTree(subDataset.get(i), subWorkload.get(i), 1);
                } else {
                    chdObject = recursiveBuildTree(subDataset.get(i), subWorkload.get(i), 0);
                }

                Query chdMBR = (Query)chdObject[0];
                Node chdNode = (Node) chdObject[1];
                chdNodes.add(chdNode);
                chdMBRs.add(chdMBR);
            }

            double cost = getNonLeafNodeCost(chdNodes.size(), workload.size(), 1, chdNodes.size());
//                double cost = D * chdNodes.size() * workload.size();
            Node nonLeafNode = new Node(splitDimension, chdMBRs, chdNodes, cost);

            return new Object[] {MBR, nonLeafNode};
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

        if (workload == null || workload.size() == 0) {
            double minLen = Double.MAX_VALUE;
            for (int dim = 0; dim < D; ++dim) {
                double tmp = dataset.maxBound.data[dimension] - dataset.minBound.data[dimension];
                if (tmp < minLen) {
                    minLen = tmp;
                    dimension = dim;
                }
            }
            return dimension;
        }


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
//        int[][] objectCount = (int[][]) costRes[1];
//        int[][] queryCount = (int[][]) costRes[2];
//        int[][] candidateCount = (int[][]) costRes[3];

        // 3. processing DP
        Object[] objects = DPProcessing(workloadPoints, cost, workload);

//        if (objects == null && dataset.dataset.size() == 1000000) {
//            costRes = getSubSpaceCost(dataset, workload, workloadPoints, dimension);
//            costRes = getSubSpaceCost(dataset, workload, workloadPoints, dimension);
//            costRes = getSubSpaceCost(dataset, workload, workloadPoints, dimension);
//            costRes = getSubSpaceCost(dataset, workload, workloadPoints, dimension);
//            DPProcessing(workloadPoints, cost, workload);
//        }

        return objects;
//        System.out.println(splitPoints.toString());
    }


    public static List<double[]> count_cost = new ArrayList<>();
    public boolean firstDP = true;

    public Object[] DPProcessing(List<Double> workloadPoints, double[][] cost, List<Query> workload) {
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
            for (int i = j; i < n; i++) {
                // i starts with j because at least j+1 points are needed to cut the j segment
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

            if (firstDP) {
                if (j == 180) {
                    int a= 1;
                }
                count_cost.add(new double[]{j, levelCost[j]});
            }

            // Early Stop: Add the relative return of the split < rate
            double relativeGain = (levelCost[j - 1] - levelCost[j]);

            if (relativeGain <= 0 && latency > 10 && !firstDP) {
                break;
            } else {
                if (relativeGain <= 0) {
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

        firstDP = false;
//        boolean arriveLeaf = (compareCost != null && compareCost[0][n-1] <= minCost);
        if (splitCount <= 0) {
            return null; // need not split
        }
//
//        //4. Generate syncopation points backward
        List<Double> splitPoints = new ArrayList<>();
        int currentIndex = n - 1;
        for (int j = splitCount; j >= 0; j--) {
            splitPoints.add(workloadPoints.get(currentIndex));
            currentIndex = path[currentIndex][j];
        }

        Collections.reverse(splitPoints);  // the inversion yields an ascending arrangement


        return new Object[]{splitPoints, minCost};
    }



    public double getParentNodeCost (int workloadSize, int chdSize) {

        return workloadSize * ( ALFA + BETA * Math.log(chdSize) + GAMA * (D-1) * 1) + SIGMA * chdSize * 20;
    }

    public static int averageQueryLength(
            List<Query> workload, int dimension) {

        if (workload == null || workload.isEmpty()) {
            return 0;
        }

        double sum = 0.0;
        int n = 0;

        for (Query q : workload) {
            if (q == null || q.pointMin == null || q.pointMax == null) {
                continue;
            }
            if (dimension < 0
                    || dimension >= q.pointMin.data.length
                    || dimension >= q.pointMax.data.length) {
                continue;
            }

            double len = q.pointMax.data[dimension]
                    - q.pointMin.data[dimension];

            // 防御性处理（可选）
            if (len < 0) len = 0;

            sum += len;
            n++;
        }

        if (n == 0) return 0;

        return (int) Math.round(sum / n);
    }



    public Object[] getSubSpaceCost(
            DataSpace dataset,
            List<Query> workload,
            List<Double> workloadPoints,
            int dimension
    ) throws Exception {

        int dim_ = (dimension + 1) % D;

        final int n = workloadPoints.size();
        final int W = workload.size();

        ModelSample[][] modelSamples = new ModelSample[n][n];
        if (useModel) {
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    modelSamples[i][j] = new ModelSample();
                }
            }
        }


        // ----------------------------
        // ----------------------------
        @SuppressWarnings("unchecked")
        ArrayList<Point>[] bucket = (ArrayList<Point>[]) new ArrayList[n];
        Query[] bucketMBR = new Query[n];
        int[] bucketSize = new int[n];

        for (int i = 0; i < n; i++) {
            bucket[i] = new ArrayList<>();
            bucketMBR[i] = new Query();
        }

        for (Point p : dataset.dataset) {
            double val = p.data[dimension];
            double key = Math.ceil(val);
            int k = lowerBound(workloadPoints, key); //
            if (k < n) {
                bucket[k].add(p);
                bucketSize[k]++;
                bucketMBR[k].addPoint(p.data); //
            }
        }

        // ----------------------------
        // ----------------------------
        int[] qL = new int[W];
        int[] qR = new int[W];
        for (int qi = 0; qi < W; qi++) {
            Query q = workload.get(qi);
            double left = q.pointMin.data[dimension];
            double right = q.pointMax.data[dimension];

            qL[qi] = lowerBound(workloadPoints, left);
            qR[qi] = upperBound(workloadPoints, right);

        }
        int[][] objectCount = new int[n][n];
        int[][] queryCount = new int[n][n];
        int[][] candidateCount = new int[n][n];

        for (int i = 0; i < n; i++) {
            int objCnt = 0;
            Query curMBR = new Query();

            for (int j = i + 1; j < n; j++) {
                if (bucketSize[j] > 0) {
                    objCnt += bucketSize[j];
                    curMBR = curMBR.merge(bucketMBR[j]);
                }
                objectCount[i][j] = objCnt;

                int qc = 0;
                int cc = 0;

                int segL = i + 1;      // inclusive
                int segR = j + 1;      // exclusive

                for (int qi = 0; qi < W; qi++) {

                    int interL = Math.max(segL, qL[qi]);
                    int interR = Math.min(segR, qR[qi]);
                    if (interR < interL) continue;

                    if (curMBR.hasIntersection(workload.get(qi),dimension)) {
                        qc++;
                        double ql = Math.min(workload.get(qi).pointMax.data[dim_], curMBR.pointMax.data[dim_]) -Math.max(workload.get(qi).pointMin.data[dim_],curMBR.pointMin.data[dim_]);
                        cc += Math.max(objCnt * ((ql)/ (curMBR.pointMax.data[dim_] - curMBR.pointMin.data[dim_])), 0);
                    }
                }

                if (useModel) {
                    modelSamples[i][j].setOiQFeature(cc);
                    modelSamples[i][j].setVolumeToFeature(curMBR);
                    modelSamples[i][j].setPerimeterToFeature(curMBR);
                    modelSamples[i][j].setQuerySizeToFeature(queryCount[i][j]);
                    modelSamples[i][j].setObjectSizeToFeature(objCnt);
                }

                queryCount[i][j] = qc;
                candidateCount[i][j] = cc;
            }
        }

        final int smoothR = Math.max(1, (int) Math.ceil(Math.log(n) / Math.log(2.0) / 2.0));

        double[][] cost = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int obj = objectCount[i][j];
                int qc  = queryCount[i][j];
                int cc  = candidateCount[i][j];

                if (STRATEGY_I) {

//                    if (qc == 0) {
                        qc = smoothDiagInt(queryCount, i, j, smoothR);
                        cc = smoothDiagInt(candidateCount, i, j, smoothR);
                        cost[i][j] = (getLeafNodeCost(objectCount[i][j], queryCount[i][j], candidateCount[i][j]) + getLeafNodeCost(obj, qc, cc)) / 2;
//                    }
                } else {
                    cost[i][j] = getLeafNodeCost(obj, qc, cc);
                }

                if (useModel) {
                    Instance instance = new DenseInstance(1, modelSamples[i][j].features);
                    instance.setDataset(instances);
                    cost[i][j] = (2 * randomForest.classifyInstance(instance) + cost[i][j]) / 3;
                }

//                cost[i][j] = (getLeafNodeCost(objectCount[i][j], queryCount[i][j], candidateCount[i][j]) + getLeafNodeCost(obj, qc, cc)) / 2;
            }
        }



        return new Object[]{cost, objectCount, queryCount, candidateCount};
    }


    private static int smoothDiagInt(int[][] A, int i, int j, int r) {
        long wsum = 0;
        long vsum = 0;

        for (int t = -r; t <= r; t++) {
            if (t == 0) continue;
            int ii = i + t;
            int jj = j + t;
            if (ii < 0 || jj < 0 || ii >= A.length || jj >= A.length) continue;
            if (ii >= jj) continue;

            int v = A[ii][jj];
            int w = (r + 1 - Math.abs(t));
            vsum += (long) v * w;
            wsum += w;
        }

        if (wsum == 0) return A[i][j];
        return (int) Math.round((double) vsum / (double) wsum);
    }




//    public Object[] getSubSpaceCost(DataSpace dataset, List<Query> workload, List<Double> workloadPoints, int dimension) throws Exception {
//        long s, e;
//        int n = workloadPoints.size();
//        ModelSample[][] modelSamples = new ModelSample[n][n];
//        if (useModel) {
//            for (int i = 0; i < n; i++) {
//                for (int j = i + 1; j < n; j++) {
//                    modelSamples[i][j] = new ModelSample();
//                }
//            }
//        }
//
//        // 1. construct prefix array sumPoint，sumPoint[i]: >= workloadPoints[i] points
//        int[] sumPoint = new int[n];
//        int[] sumObjectTravelCount = new int[n];// used for extract model feature
//
//        // create multiple datarectangles in bulk
//        Query[] MBRs = new Query[n];
//        for (int i = 0; i < MBRs.length; i++) {
//            MBRs[i] = new Query(); // create a two dimensional rectangle
//        }
//
//        for (Point point : dataset.dataset) {
//            double val = point.data[dimension];
//            // find label that first >= val
//            int k = Collections.binarySearch(workloadPoints, Math.ceil(val));
//            if (k < 0) {
//                k = - k - 1;  //if not return to the insertion position
//            }
//            if (k < n) {
//                MBRs[k].addPoint(point.data);
//                // from position k add 1 to all elements
//                for (int i = k; i < n; i++) {
//                    sumPoint[i]++;
//                }
//
//                if (useModel) {
//                    Integer ct = objectTraveledCountMap.getOrDefault(point, 0);
//                    for (int i = k; i < n; i++) {
//                        sumObjectTravelCount[i] += ct;
//                    }
//                }
//            }
//        }
//
////        s = System.currentTimeMillis();
//        // 2. Construct pointCount[i][j] to represent the number of data points in the interval [i, j]; MBRGrid[i,j] represents the MBR of the interval [i, j]
//        int[][] pointCount = new int[n][n];
//        Query[][] MBRGrid = new Query[n][n];
//        for (int i = 0; i < n; i++) {
//            for (int j = i; j < n; j++) {
//                MBRGrid[i][j] = new Query();
//            }
//        }
//        for (int i = 0; i < n; i++) {
//            for (int j = i + 1; j < n; j++) {
//                pointCount[i][j] = sumPoint[j] - sumPoint[i];
//                if (j == i + 1) {
//                    MBRGrid[i][j] = MBRs[j];
//                } else {
//                    MBRGrid[i][j] = MBRs[j].merge(MBRGrid[i][j - 1]);
//                }
//            }
//        }
////        e = System.currentTimeMillis();
////        System.out.println("part 2 time: " + (e - s) + "ms");
//
//        // 3. Calculate queryCount[i][j] and candidates[i][j]
//        Object[] queryInfo = countQueryIntersectionsAndCandidates(MBRGrid, workload, dataset, dimension);
//        int[][] queryCount = (int[][]) queryInfo[0];
//        int[][] candidates = (int[][]) queryInfo[1];
//
//        // 4. compute dp cost，cost[i][j] = pointCount[i][j] * queryCount[i][j]
//        double[][] cost = new double[n][n];
//        for (int i = 0; i < n; i++) {
//            for (int j = i + 1; j < n; j++) {
//                cost[i][j] = getNodeCost(pointCount[i][j], queryCount[i][j]);
//            }
//        }
//
//        if (!useModel)
//            return new Object[]{cost, cost};
//
//        //5. get features and cost for model
//        double[][] modelCost = new double[n][n];
//        for (int i = 0; i < n; i++) {
//            for (int j = i + 1; j < n; j++) {
//                modelSamples[i][j].setOiQFeature(sumObjectTravelCount[j] - sumObjectTravelCount[i]);
//                modelSamples[i][j].setVolumeToFeature(MBRGrid[i][j]);
//                modelSamples[i][j].setPerimeterToFeature(MBRGrid[i][j]);
//                modelSamples[i][j].setQuerySizeToFeature(queryCount[i][j]);
//                modelSamples[i][j].setObjectSizeToFeature(pointCount[i][j]);
//                // 如果 ModelSample 支持 candidate 特征，可以在这里设置
//                // modelSamples[i][j].setCandidateFeature(candidates[i][j]);
//            }
//        }
//
////        s = System.currentTimeMillis();
//        // 6. combine model prediction
//        for (int i = 0; i < n; i++) {
//            for (int j = i + 1; j < n; j++) {
//                // prediction
//                Instance instance = new DenseInstance(1, modelSamples[i][j].features);
//                instance.setDataset(instances);
//                modelCost[i][j] = randomForest.classifyInstance(instance);
//            }
//        }
////        modelCost = batchPredictModelCost(modelSamples);
////        e = System.currentTimeMillis();
////        System.out.println("part 6 time: " + (e - s) + "ms" + ", size n: " + n);
//
//        for (int i = 0; i < n; i++) {
//            for (int j = i + 1; j < n; j++) {
//                // combine cost
//                modelCost[i][j] = cost[i][j] - ((Math.log(1 + iteration)) / (Math.log(1 + K))) * (cost[i][j] - modelCost[i][j]);
//            }
//        }
//
//        return new Object[]{modelCost, useModel ? cost : null};
//    }


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


//    public static int[][] countQueryIntersections(Query[][] MBRGrid, List<Query> workload, int dimension) {
//        int n = MBRGrid.length;
//        int[][] queryCount = new int[n][n];
//
//        // Extract the one-dimensional coordinate sequence in advance to facilitate binary divisions
//        double[] starts = new double[n];
//        double[] ends = new double[n];
//        starts[0] = -Double.MIN_VALUE;
//        ends[0] = -Double.MIN_VALUE;
//        for (int i = 0; i < n-1; i++) {
//            starts[i + 1] = MBRGrid[i][i+1].pointMin.data[dimension];
//            if (starts[i + 1] == Double.MAX_VALUE) {
//                starts[i + 1] = starts[i];
//            }
//        }
//        for (int j = 1; j < n; j++) {
//            ends[j] = MBRGrid[0][j].pointMax.data[dimension];
//            if (ends[j] == Double.MAX_VALUE) {
//                ends[j] = ends[j - 1];
//            }
//        }
//
//        // iterate through each query
//        for (Query q : workload) {
//            double qMin = q.pointMin.data[dimension];
//            double qMax = q.pointMax.data[dimension];
//
//            // The left and right boundaries of the two-part positioning
//            int left = lowerBound(ends, qMin);     // the first j which may intersect
//            int right = upperBound(starts, qMax);  // the last i that may intersect
//            if (left == -1 || right == -1) continue;
//
//            for (int i = 0; i <= right && i < n; i++) {
//                for (int j = left; j < n; j++) {
//                    Query cell = MBRGrid[i][j];
//                    if (cell == null) continue;
//                    if (q.checkRelation(cell)) {
//                        queryCount[i][j]++;
//                    }
//                }
//            }
//        }
//
//        return queryCount;
//    }

    private static List<Double> getWorkloadPoint(DataSpace dataset, List<Query> querySet, int dimension) {

        TreeSet<Double> pointSet = new TreeSet<>();

        double min = dataset.minBound.data[dimension] - 1;
        double max = dataset.maxBound.data[dimension];
        pointSet.add(min);
        pointSet.add(max);

        for (Query query : querySet) {
            double ql = query.pointMin.data[dimension];
            double qr = query.pointMax.data[dimension];

            if (max > ql && ql > min) pointSet.add(ql - 1);
            if (min < qr && qr < max) pointSet.add(qr);
        }

        List<Double> base = new ArrayList<>(pointSet);
        if (!STRATEGY_I || base.size() < 3) return base;

        int S = base.size();
        int w = (int) Math.ceil((Math.log(S) / Math.log(2.0)) / 2.0);
        w = Math.max(w, 1);

        double[] gaps = new double[S - 1];
        for (int i = 0; i < S - 1; i++) {
            gaps[i] = base.get(i + 1) - base.get(i);
        }

        ArrayList<Double> toInsert = new ArrayList<>();

        for (int i = 0; i < gaps.length; i++) {
            double li = gaps[i];
            double ldelta = neighborhoodAvgGap(gaps, i, w);

            if (ldelta <= 0) continue;

            if (li > 2.0 * ldelta) {
                int k = (int) Math.ceil(li / ldelta) - 1;
                if (k <= 0) continue;

                double left = base.get(i);
                double right = base.get(i + 1);

                for (int t = 1; t <= k; t++) {
                    double p = left + li * ((double) t / (k + 1));
                    if (p > left && p < right) toInsert.add(p);
                }
            }
        }

        pointSet.addAll(toInsert);
        return new ArrayList<>(pointSet);
    }

    private static double neighborhoodAvgGap(double[] gaps, int i, int w) {
        double sum = 0.0;
        int cnt = 0;
        for (int t = 1; t <= w; t++) {
            int L = i - t;
            int R = i + t;
            if (L >= 0) { sum += gaps[L]; cnt++; }
            if (R < gaps.length) { sum += gaps[R]; cnt++; }
        }
        return cnt == 0 ? 0.0 : (sum / cnt);
    }

}
