package tfg.udc.boya;

import android.graphics.Color;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.CheckBox;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.github.mikephil.charting.charts.LineChart;
import com.github.mikephil.charting.components.Description;
import com.github.mikephil.charting.data.Entry;
import com.github.mikephil.charting.data.LineData;
import com.github.mikephil.charting.data.LineDataSet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public class SessionWindowPagerAdapter extends RecyclerView.Adapter<SessionWindowPagerAdapter.ViewHolder> {

    private final List<SessionWindowViewData> items = new ArrayList<>();
    private final Map<Integer, boolean[]> selectionByWindow = new HashMap<>();

    public void setItems(List<SessionWindowViewData> data) {
        items.clear();
        items.addAll(data);
        notifyDataSetChanged();
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.item_session_window, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        SessionWindowViewData item = items.get(position);

        holder.tvWindowTitle.setText(String.format(
                Locale.US,
                "Ventana %d/%d (%.1f - %.1f min)",
                item.index,
                item.total,
                item.startSec / 60.0,
                item.endSec / 60.0
        ));

        holder.tvMetrics.setText(String.format(
                Locale.US,
                "Hs=%.4f m | Tz=%.2f s | Tc=%.2f s | Tp=%.2f s\n" +
                        "Hs: altura significativa (tercio mas alto)\n" +
                        "Tz: periodo medio por cruces por cero\n" +
                        "Tc: periodo medio de crestas\n" +
                        "Tp: periodo de pico espectral\n" +
                        "%s\n" +
                        "fs=%.2f Hz | N=%d | seg=%d",
                item.metrics.hs,
                item.metrics.tzSec,
                item.metrics.tcSec,
                item.metrics.tpSec,
                buildQuickInterpretation(item.metrics.hs),
                item.metrics.processingFsHz,
                item.metrics.uniformN,
                item.metrics.segments
        ));

        WaveProfile[] waves = buildWaveProfiles(item);
        holder.tvWaveMean.setText(String.format(
                Locale.US,
                "Ola media: H=%.3f m | T=%.2f s",
                waves[0].heightM, waves[0].periodSec
        ));
        holder.tvWaveSig.setText(String.format(
                Locale.US,
                "Ola significativa (Hs): H=%.3f m | T=%.2f s",
                waves[1].heightM, waves[1].periodSec
        ));
        holder.tvWaveH10.setText(String.format(
                Locale.US,
                "Ola alta (H1/10): H=%.3f m | T=%.2f s",
                waves[2].heightM, waves[2].periodSec
        ));
        holder.tvWaveHmax.setText(String.format(
                Locale.US,
                "Ola maxima esperada: H=%.3f m | T=%.2f s",
                waves[3].heightM, waves[3].periodSec
        ));

        boolean[] selected = selectionByWindow.get(item.index);
        if (selected == null) {
            selected = new boolean[]{false, true, false, false};
            selectionByWindow.put(item.index, selected);
        }

        bindToggle(holder.cbWaveMean, selected, 0, holder.chartWave, waves);
        bindToggle(holder.cbWaveSig, selected, 1, holder.chartWave, waves);
        bindToggle(holder.cbWaveH10, selected, 2, holder.chartWave, waves);
        bindToggle(holder.cbWaveHmax, selected, 3, holder.chartWave, waves);

        renderWaveChart(holder.chartWave, waves, selected);
    }

    @Override
    public int getItemCount() {
        return items.size();
    }

    private void bindToggle(CheckBox checkBox, boolean[] selected, int idx, LineChart chart, WaveProfile[] waves) {
        checkBox.setOnCheckedChangeListener(null);
        checkBox.setChecked(selected[idx]);
        checkBox.setOnCheckedChangeListener((buttonView, isChecked) -> {
            selected[idx] = isChecked;
            renderWaveChart(chart, waves, selected);
        });
    }

    private void renderWaveChart(LineChart chart, WaveProfile[] waves, boolean[] selected) {
        LineData data = new LineData();
        double maxAbs = 0.05;

        for (int wi = 0; wi < waves.length; wi++) {
            if (!selected[wi]) {
                continue;
            }
            WaveProfile wave = waves[wi];
            List<Entry> entries = new ArrayList<>();
            for (int i = 0; i < wave.timeSec.length; i++) {
                entries.add(new Entry((float) wave.timeSec[i], (float) wave.heightSeriesM[i]));
                maxAbs = Math.max(maxAbs, Math.abs(wave.heightSeriesM[i]));
            }
            LineDataSet set = new LineDataSet(entries, wave.label);
            set.setColor(wave.color);
            set.setLineWidth(2.2f);
            set.setDrawCircles(false);
            set.setDrawValues(false);
            set.setMode(LineDataSet.Mode.CUBIC_BEZIER);
            set.setCubicIntensity(0.22f);
            data.addDataSet(set);
        }

        if (data.getDataSetCount() == 0) {
            WaveProfile fallback = waves[1];
            List<Entry> entries = new ArrayList<>();
            for (int i = 0; i < fallback.timeSec.length; i++) {
                entries.add(new Entry((float) fallback.timeSec[i], (float) fallback.heightSeriesM[i]));
                maxAbs = Math.max(maxAbs, Math.abs(fallback.heightSeriesM[i]));
            }
            LineDataSet set = new LineDataSet(entries, "Significativa");
            set.setColor(Color.parseColor("#0D47A1"));
            set.setLineWidth(2.2f);
            set.setDrawCircles(false);
            set.setDrawValues(false);
            set.setMode(LineDataSet.Mode.CUBIC_BEZIER);
            set.setCubicIntensity(0.22f);
            data.addDataSet(set);
        }

        chart.setData(data);
        chart.getAxisRight().setEnabled(false);
        chart.getXAxis().setTextColor(Color.DKGRAY);
        chart.getAxisLeft().setTextColor(Color.DKGRAY);
        chart.getLegend().setTextColor(Color.DKGRAY);
        chart.getAxisLeft().setAxisMinimum((float) (-1.4 * maxAbs));
        chart.getAxisLeft().setAxisMaximum((float) (1.4 * maxAbs));

        Description d = new Description();
        d.setText("Olas tipo (1 periodo visible)");
        chart.setDescription(d);
        chart.invalidate();
    }

    private WaveProfile[] buildWaveProfiles(SessionWindowViewData item) {
        double hs = Math.max(0.0, item.metrics.hs);
        double period = pickRepresentativePeriod(item.metrics);
        double durationSec = item.endSec - item.startSec;
        double tz = (Double.isFinite(item.metrics.tzSec) && item.metrics.tzSec > 0.0) ? item.metrics.tzSec : period;
        double nWaves = Math.max(1.0, durationSec / Math.max(tz, 1e-6));

        double hMean = 0.625 * hs;
        double hSig = hs;
        double h10 = 1.27 * hs;
        double hMax = hs * Math.sqrt(Math.max(1.0, 0.5 * Math.log(Math.max(2.0, nWaves))));

        return new WaveProfile[]{
                makeProfile("Media", hMean, period, Color.parseColor("#455A64")),
                makeProfile("Significativa", hSig, period, Color.parseColor("#0D47A1")),
                makeProfile("H1/10", h10, period, Color.parseColor("#2E7D32")),
                makeProfile("Max esperada", hMax, period, Color.parseColor("#EF6C00"))
        };
    }

    private WaveProfile makeProfile(String label, double heightM, double periodSec, int color) {
        double p = (Double.isFinite(periodSec) && periodSec > 0.0) ? periodSec : 8.0;
        double durationSec = p; // Mostrar exactamente una ola (un periodo).
        double fs = 20.0;
        int n = Math.max(64, (int) Math.round(durationSec * fs) + 1);

        double[] t = new double[n];
        double[] h = new double[n];
        double a = Math.max(0.0, heightM) * 0.5; // H = 2A

        for (int i = 0; i < n; i++) {
            double ti = i / fs;
            t[i] = ti;
            h[i] = a * Math.sin(2.0 * Math.PI * ti / p);
        }
        return new WaveProfile(label, heightM, p, color, t, h);
    }

    private double pickRepresentativePeriod(OmbResult metrics) {
        double tz = (Double.isFinite(metrics.tzSec) && metrics.tzSec > 0.0) ? metrics.tzSec : Double.NaN;
        double tp = (Double.isFinite(metrics.tpSec) && metrics.tpSec > 0.0) ? metrics.tpSec : Double.NaN;
        double tc = (Double.isFinite(metrics.tcSec) && metrics.tcSec > 0.0) ? metrics.tcSec : Double.NaN;

        double chosen;
        if (Double.isFinite(tz)) {
            chosen = tz;
            if (Double.isFinite(tp) && tp >= 0.6 * tz && tp <= 1.8 * tz) {
                chosen = tp;
            }
        } else if (Double.isFinite(tp)) {
            chosen = tp;
        } else if (Double.isFinite(tc)) {
            chosen = tc;
        } else {
            chosen = 8.0;
        }

        // Evita periodos extremos por ruido que deforman la representacion visual.
        if (chosen < 2.5) {
            chosen = 2.5;
        } else if (chosen > 30.0) {
            chosen = 30.0;
        }
        return chosen;
    }

    private String buildQuickInterpretation(double hs) {
        if (!Double.isFinite(hs)) {
            return "Interpretacion: no disponible";
        }
        if (hs < 0.10) {
            return "Interpretacion: mar muy suave / baja energia";
        }
        if (hs < 0.50) {
            return "Interpretacion: mar suave";
        }
        if (hs < 1.25) {
            return "Interpretacion: mar moderado";
        }
        return "Interpretacion: mar energetico";
    }

    static class ViewHolder extends RecyclerView.ViewHolder {
        final TextView tvWindowTitle;
        final TextView tvMetrics;
        final TextView tvWaveMean;
        final TextView tvWaveSig;
        final TextView tvWaveH10;
        final TextView tvWaveHmax;
        final CheckBox cbWaveMean;
        final CheckBox cbWaveSig;
        final CheckBox cbWaveH10;
        final CheckBox cbWaveHmax;
        final LineChart chartWave;

        ViewHolder(@NonNull View itemView) {
            super(itemView);
            tvWindowTitle = itemView.findViewById(R.id.tvWindowTitle);
            tvMetrics = itemView.findViewById(R.id.tvWindowMetrics);
            tvWaveMean = itemView.findViewById(R.id.tvWaveMean);
            tvWaveSig = itemView.findViewById(R.id.tvWaveSig);
            tvWaveH10 = itemView.findViewById(R.id.tvWaveH10);
            tvWaveHmax = itemView.findViewById(R.id.tvWaveHmax);
            cbWaveMean = itemView.findViewById(R.id.cbWaveMean);
            cbWaveSig = itemView.findViewById(R.id.cbWaveSig);
            cbWaveH10 = itemView.findViewById(R.id.cbWaveH10);
            cbWaveHmax = itemView.findViewById(R.id.cbWaveHmax);
            chartWave = itemView.findViewById(R.id.chartWindowWave);
        }
    }

    private static class WaveProfile {
        final String label;
        final double heightM;
        final double periodSec;
        final int color;
        final double[] timeSec;
        final double[] heightSeriesM;

        WaveProfile(String label, double heightM, double periodSec, int color, double[] timeSec, double[] heightSeriesM) {
            this.label = label;
            this.heightM = heightM;
            this.periodSec = periodSec;
            this.color = color;
            this.timeSec = Arrays.copyOf(timeSec, timeSec.length);
            this.heightSeriesM = Arrays.copyOf(heightSeriesM, heightSeriesM.length);
        }
    }
}
