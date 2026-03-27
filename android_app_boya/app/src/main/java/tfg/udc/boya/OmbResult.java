package tfg.udc.boya;

public class OmbResult {
    public final String modeLabel;
    public final double processingFsHz;
    public final int inputN;
    public final double inputDurationSec;
    public final int uniformN;
    public final int segments;
    public final double hs;
    public final double tzSec;
    public final double tcSec;
    public final double tpSec;
    public final double[] spectrumFreqHz;
    public final double[] spectrumElevation;
    public final double[] previewTimeSec;
    public final double[] previewAzDyn;

    public OmbResult(
            String modeLabel,
            double processingFsHz,
            int inputN,
            double inputDurationSec,
            int uniformN,
            int segments,
            double hs,
            double tzSec,
            double tcSec,
            double tpSec,
            double[] spectrumFreqHz,
            double[] spectrumElevation,
            double[] previewTimeSec,
            double[] previewAzDyn
    ) {
        this.modeLabel = modeLabel;
        this.processingFsHz = processingFsHz;
        this.inputN = inputN;
        this.inputDurationSec = inputDurationSec;
        this.uniformN = uniformN;
        this.segments = segments;
        this.hs = hs;
        this.tzSec = tzSec;
        this.tcSec = tcSec;
        this.tpSec = tpSec;
        this.spectrumFreqHz = spectrumFreqHz;
        this.spectrumElevation = spectrumElevation;
        this.previewTimeSec = previewTimeSec;
        this.previewAzDyn = previewAzDyn;
    }
}
