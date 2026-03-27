package tfg.udc.boya;

import android.os.Bundle;
import android.view.View;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;
import androidx.viewpager2.widget.ViewPager2;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class SessionDetailActivity extends AppCompatActivity {

    public static final String EXTRA_SESSION_PATH = "extra_session_path";
    public static final String EXTRA_PROCESS_MODE = "extra_process_mode";

    private static final double WINDOW_SEC_DEFAULT = 20.0 * 60.0;
    private static final double MIN_WINDOW_SEC = 120.0;

    private final SessionRepository sessionRepository = new SessionRepository();
    private final OmbProcessor ombProcessor = new OmbProcessor();

    private TextView tvTitle;
    private TextView tvSummary;
    private ProgressBar progress;
    private ViewPager2 pager;
    private SessionWindowPagerAdapter pagerAdapter;
    private Button btnPrevWindow;
    private Button btnNextWindow;
    private TextView tvWindowPageIndicator;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_session_detail);

        tvTitle = findViewById(R.id.tvSessionDetailTitle);
        tvSummary = findViewById(R.id.tvSessionDetailSummary);
        progress = findViewById(R.id.progressSessionDetail);
        pager = findViewById(R.id.vpSessionWindows);
        btnPrevWindow = findViewById(R.id.btnPrevWindow);
        btnNextWindow = findViewById(R.id.btnNextWindow);
        tvWindowPageIndicator = findViewById(R.id.tvWindowPageIndicator);

        pagerAdapter = new SessionWindowPagerAdapter();
        pager.setAdapter(pagerAdapter);
        pager.setUserInputEnabled(false);

        btnPrevWindow.setOnClickListener(v -> moveToPage(-1));
        btnNextWindow.setOnClickListener(v -> moveToPage(1));
        pager.registerOnPageChangeCallback(new ViewPager2.OnPageChangeCallback() {
            @Override
            public void onPageSelected(int position) {
                updatePagerUi();
            }
        });

        String sessionPath = getIntent().getStringExtra(EXTRA_SESSION_PATH);
        String modeRaw = getIntent().getStringExtra(EXTRA_PROCESS_MODE);

        final OmbProcessor.Mode mode;
        try {
            mode = modeRaw == null ? OmbProcessor.Mode.OMB_10HZ : OmbProcessor.Mode.valueOf(modeRaw);
        } catch (IllegalArgumentException e) {
            Toast.makeText(this, "Modo invalido, se usa OMB 10Hz", Toast.LENGTH_SHORT).show();
            loadSession(sessionPath, OmbProcessor.Mode.OMB_10HZ);
            return;
        }

        loadSession(sessionPath, mode);
    }

    private void loadSession(String sessionPath, OmbProcessor.Mode mode) {
        if (sessionPath == null || sessionPath.trim().isEmpty()) {
            Toast.makeText(this, "No se recibio la sesion", Toast.LENGTH_LONG).show();
            finish();
            return;
        }

        File sessionFile = new File(sessionPath);
        if (!sessionFile.exists()) {
            Toast.makeText(this, "Archivo no encontrado", Toast.LENGTH_LONG).show();
            finish();
            return;
        }

        tvTitle.setText(sessionFile.getName());
        tvSummary.setText("Procesando ventanas...");
        progress.setVisibility(View.VISIBLE);

        new Thread(() -> {
            try {
                SessionData full = sessionRepository.readSession(sessionFile);
                List<double[]> windows = computeWindowRanges(full.durationSec());
                List<SessionWindowViewData> out = new ArrayList<>();

                for (int i = 0; i < windows.size(); i++) {
                    double startSec = windows.get(i)[0];
                    double endSec = windows.get(i)[1];
                    SessionData slice = sliceWindow(full, startSec, endSec);
                    OmbResult result = ombProcessor.process(slice, mode);
                    out.add(new SessionWindowViewData(
                            i + 1,
                            windows.size(),
                            startSec,
                            endSec,
                            result
                    ));
                }

                runOnUiThread(() -> {
                    progress.setVisibility(View.GONE);
                    pagerAdapter.setItems(out);
                    pager.setCurrentItem(0, false);
                    updatePagerUi();
                    tvSummary.setText(String.format(
                            Locale.US,
                            "Duracion total: %.1f min | Ventanas: %d | Usa Anterior/Siguiente",
                            full.durationSec() / 60.0,
                            out.size()
                    ));
                });
            } catch (Exception e) {
                runOnUiThread(() -> {
                    progress.setVisibility(View.GONE);
                    tvSummary.setText("Error: " + e.getMessage());
                    Toast.makeText(this, "No se pudo procesar la sesion", Toast.LENGTH_LONG).show();
                });
            }
        }).start();
    }

    private void moveToPage(int delta) {
        int total = pagerAdapter.getItemCount();
        if (total <= 0) {
            return;
        }
        int current = pager.getCurrentItem();
        int target = current + delta;
        if (target < 0) {
            target = 0;
        } else if (target >= total) {
            target = total - 1;
        }
        if (target != current) {
            pager.setCurrentItem(target, true);
        }
        updatePagerUi();
    }

    private void updatePagerUi() {
        int total = pagerAdapter.getItemCount();
        int current = Math.min(Math.max(pager.getCurrentItem(), 0), Math.max(0, total - 1));
        tvWindowPageIndicator.setText(String.format(Locale.US, "%d/%d", total == 0 ? 0 : (current + 1), total));
        btnPrevWindow.setEnabled(total > 0 && current > 0);
        btnNextWindow.setEnabled(total > 0 && current < total - 1);
    }

    private List<double[]> computeWindowRanges(double durationSec) {
        List<double[]> ranges = new ArrayList<>();
        if (durationSec <= WINDOW_SEC_DEFAULT) {
            ranges.add(new double[]{0.0, durationSec});
            return ranges;
        }

        double start = 0.0;
        while (start + WINDOW_SEC_DEFAULT <= durationSec) {
            ranges.add(new double[]{start, start + WINDOW_SEC_DEFAULT});
            start += WINDOW_SEC_DEFAULT;
        }

        double remaining = durationSec - start;
        if (remaining >= MIN_WINDOW_SEC) {
            ranges.add(new double[]{start, durationSec});
        } else if (!ranges.isEmpty()) {
            double[] last = ranges.get(ranges.size() - 1);
            last[1] = durationSec;
        }

        return ranges;
    }

    private SessionData sliceWindow(SessionData data, double startSec, double endSec) {
        List<Double> tList = new ArrayList<>();
        List<Double> azList = new ArrayList<>();

        for (int i = 0; i < data.tSec.length; i++) {
            double t = data.tSec[i];
            if (t < startSec) {
                continue;
            }
            if (t > endSec) {
                break;
            }
            tList.add(t);
            azList.add(data.az[i]);
        }

        if (tList.size() < 100) {
            throw new IllegalArgumentException("Ventana sin suficientes muestras");
        }

        double t0 = tList.get(0);
        double[] tOut = new double[tList.size()];
        double[] azOut = new double[tList.size()];
        for (int i = 0; i < tList.size(); i++) {
            tOut[i] = tList.get(i) - t0;
            azOut[i] = azList.get(i);
        }

        return new SessionData(tOut, azOut);
    }
}
