package index;
import data_structure.ModelSample;
import data_structure.Point;
import data_structure.Query;

import java.util.ArrayList;
import java.util.List;

import static data_structure.Parameters.D;
import static data_structure.Parameters.T;

public class Node {

    ModelSample[][] samples;

    double cost;
    List<Point> dataset;
    List<Query> MBRs;
    List<Node> childes;

    int splitDimension;

    public void initSamples() {
        if (this.isLeafNode()) return;
        int n = childes.size();
        samples = new ModelSample[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n && j - i < T; j++) {
                samples[i][j] = new ModelSample();
            }
        }

    }

    public Node(List<Point> dataset, double cost) {
        this.dataset = dataset;
        this.cost = cost;
    }

    public Node(int splitDimension, List<Query> MBRs, List<Node> childes, double cost) {
        this.MBRs = MBRs;
        this.childes = childes;
        this.splitDimension = splitDimension;
        this.cost = cost;
    }

    public boolean isLeafNode() {
        return dataset != null;
    }

    public List<Point> leafNodeQuery(Query query) {
        List<Point> res = new ArrayList<Point>();
        if (dataset == null) {
            return res;
        }

        for (Point p : dataset) {
            if (query.hasIntersection(p)) {
                res.add(p);
            }
        }

        return res;
    }

    public List<Node> nonLeafNodeQuery(Query query, List<Node> resNode) {
        List<Node> res = new ArrayList<Node>();
        if (childes == null) {
            return res;
        }

        for (int i = 0; i < childes.size(); i++) {
            if (query.hasIntersection(MBRs.get(i))) {
                if (query.contains(MBRs.get(i))) {
                    resNode.add(childes.get(i));
                } else {
                    res.add(childes.get(i));
                }
            }
        }

        return res;
    }


    public List<Node> nonLeafNodeBasicQuery(Query query) {
        List<Node> res = new ArrayList<Node>();
        if (childes == null) {
            return res;
        }

        for (int i = 0; i < childes.size(); i++) {
            if (query.hasIntersection(MBRs.get(i))) {
                res.add(childes.get(i));
            }
        }

        return res;
    }


    public static double getNodeCost(int datasetSize, int workloadSize) {
        return D * datasetSize * workloadSize;
    }

}
