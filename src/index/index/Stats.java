package index;

import java.util.ArrayList;
import java.util.List;

public class Stats {
    // time + access size
    List<Long[]> nodeVisitedInfo = new ArrayList<>();
    // time + objects size + candidate size + res size
    List<Long[]> leafNode = new ArrayList<>();
    // time + objects size + candidate size + res size
    List<Long[]> nonLeafNode = new ArrayList<>();
}
