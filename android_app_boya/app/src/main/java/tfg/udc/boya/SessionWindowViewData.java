package tfg.udc.boya;

public class SessionWindowViewData {
    public final int index;
    public final int total;
    public final double startSec;
    public final double endSec;
    public final OmbResult metrics;

    public SessionWindowViewData(
            int index,
            int total,
            double startSec,
            double endSec,
            OmbResult metrics
    ) {
        this.index = index;
        this.total = total;
        this.startSec = startSec;
        this.endSec = endSec;
        this.metrics = metrics;
    }
}
