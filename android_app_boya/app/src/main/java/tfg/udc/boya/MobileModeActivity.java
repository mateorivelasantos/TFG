package tfg.udc.boya;

import android.app.AlertDialog;
import android.content.Intent;
import android.graphics.Color;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.text.InputType;
import android.view.Menu;
import android.view.MotionEvent;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ListView;
import android.widget.PopupMenu;
import android.widget.RadioButton;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.github.mikephil.charting.charts.LineChart;
import com.github.mikephil.charting.components.Description;
import com.github.mikephil.charting.data.Entry;
import com.github.mikephil.charting.data.LineData;
import com.github.mikephil.charting.data.LineDataSet;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class MobileModeActivity extends AppCompatActivity implements SensorEventListener {

    private static final int DEFAULT_DURATION_SEC = 30;
    private static final int UI_UPDATE_INTERVAL_MS = 150;

    private static final int ACTION_PROCESS = 1;
    private static final int ACTION_INFO = 2;
    private static final int ACTION_RENAME = 3;
    private static final int ACTION_DELETE = 4;
    private static final int ACTION_OPEN_DETAIL = 5;

    private SensorManager sensorManager;
    private Sensor accel;
    private Sensor gyro;

    private TextView tvStatus;
    private TextView tvSelectedSession;
    private LineChart chartSpectrum;
    private LineChart chartSeries;
    private EditText etDurationSec;
    private EditText etSessionName;
    private RadioButton rbModeOmb10;
    private RadioButton rbModeNative;
    private Button btnStart;
    private Button btnStop;
    private Button btnRefreshSessions;
    private ListView lvSessions;

    private boolean capturing = false;
    private long t0ElapsedNs = 0L;
    private long captureDurationMs = DEFAULT_DURATION_SEC * 1000L;
    private long lastUiUpdateMs = 0L;

    private float ax;
    private float ay;
    private float az;
    private float gx;
    private float gy;
    private float gz;

    private final List<ImuSample> currentSamples = new ArrayList<>();
    private final List<File> listedSessionFiles = new ArrayList<>();
    private ArrayAdapter<String> sessionsAdapter;

    private final Handler handler = new Handler(Looper.getMainLooper());
    private Runnable autoStopRunnable;

    private final SessionRepository sessionRepository = new SessionRepository();
    private final OmbProcessor ombProcessor = new OmbProcessor();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        tvStatus = findViewById(R.id.tvStatus);
        tvSelectedSession = findViewById(R.id.tvSelectedSession);
        chartSpectrum = findViewById(R.id.chartSpectrum);
        chartSeries = findViewById(R.id.chartSeries);
        etDurationSec = findViewById(R.id.etDurationSec);
        etSessionName = findViewById(R.id.etSessionName);
        rbModeOmb10 = findViewById(R.id.rbModeOmb10);
        rbModeNative = findViewById(R.id.rbModeNative);
        btnStart = findViewById(R.id.btnStart);
        btnStop = findViewById(R.id.btnStop);
        btnRefreshSessions = findViewById(R.id.btnRefreshSessions);
        lvSessions = findViewById(R.id.lvSessions);

        sessionsAdapter = new ArrayAdapter<>(this, android.R.layout.simple_list_item_1, new ArrayList<>());
        lvSessions.setAdapter(sessionsAdapter);

        sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        accel = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        gyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);

        if (accel == null || gyro == null) {
            tvStatus.setText("ERROR: el movil no tiene acelerometro o giroscopio.");
            btnStart.setEnabled(false);
            btnStop.setEnabled(false);
            return;
        }

        btnStart.setOnClickListener(v -> startCapture());
        btnStop.setOnClickListener(v -> stopCapture(true));
        btnRefreshSessions.setOnClickListener(v -> refreshSessionList());

        lvSessions.setOnItemClickListener((parent, view, position, id) -> {
            if (position >= 0 && position < listedSessionFiles.size()) {
                processAndShowSession(listedSessionFiles.get(position));
            }
        });

        lvSessions.setOnItemLongClickListener((parent, view, position, id) -> {
            if (position >= 0 && position < listedSessionFiles.size()) {
                showSessionOptions(view, listedSessionFiles.get(position));
                return true;
            }
            return false;
        });

        // Evita que el ScrollView padre robe el gesto cuando se desplaza la lista de sesiones.
        lvSessions.setOnTouchListener((v, event) -> {
            switch (event.getActionMasked()) {
                case MotionEvent.ACTION_DOWN:
                case MotionEvent.ACTION_MOVE:
                    v.getParent().requestDisallowInterceptTouchEvent(true);
                    break;
                case MotionEvent.ACTION_UP:
                case MotionEvent.ACTION_CANCEL:
                    v.getParent().requestDisallowInterceptTouchEvent(false);
                    break;
                default:
                    break;
            }
            return false;
        });

        setupCharts();
        refreshSessionList();
        tvStatus.setText("Listo para capturar IMU.");
    }

    private int readDurationSeconds() {
        String raw = etDurationSec.getText().toString().trim();
        if (raw.isEmpty()) {
            return DEFAULT_DURATION_SEC;
        }
        try {
            int value = Integer.parseInt(raw);
            return value > 0 ? value : DEFAULT_DURATION_SEC;
        } catch (NumberFormatException e) {
            return DEFAULT_DURATION_SEC;
        }
    }

    private void startCapture() {
        if (capturing) {
            return;
        }

        int durationSec = readDurationSeconds();
        captureDurationMs = durationSec * 1000L;
        etDurationSec.setText(String.valueOf(durationSec));

        currentSamples.clear();
        ax = ay = az = 0f;
        gx = gy = gz = 0f;

        capturing = true;
        t0ElapsedNs = SystemClock.elapsedRealtimeNanos();
        lastUiUpdateMs = 0L;

        btnStart.setEnabled(false);
        btnStop.setEnabled(true);

        tvStatus.setText(String.format(Locale.US, "Capturando IMU (%d s)...", durationSec));

        sensorManager.registerListener(this, accel, SensorManager.SENSOR_DELAY_GAME);
        sensorManager.registerListener(this, gyro, SensorManager.SENSOR_DELAY_GAME);

        autoStopRunnable = () -> stopCapture(false);
        handler.postDelayed(autoStopRunnable, captureDurationMs);
    }

    private void stopCapture(boolean manualStop) {
        if (!capturing) {
            return;
        }

        capturing = false;
        sensorManager.unregisterListener(this);
        if (autoStopRunnable != null) {
            handler.removeCallbacks(autoStopRunnable);
            autoStopRunnable = null;
        }

        btnStart.setEnabled(true);
        btnStop.setEnabled(false);

        File savedFile;
        try {
            savedFile = sessionRepository.saveSession(
                    this,
                    currentSamples,
                    captureDurationMs,
                    etSessionName.getText().toString()
            );
        } catch (IOException e) {
            Toast.makeText(this, "Error guardando sesion: " + e.getMessage(), Toast.LENGTH_LONG).show();
            tvStatus.setText("Captura detenida sin datos validos.");
            return;
        }

        if (savedFile == null) {
            tvStatus.setText("Captura detenida sin datos validos.");
            return;
        }

        String stopReason = manualStop ? "detenida manualmente" : "finalizada por tiempo";
        tvStatus.setText(String.format(
                Locale.US,
                "Captura %s. Guardada: %s (%d muestras)",
                stopReason,
                savedFile.getName(),
                currentSamples.size()
        ));
        etSessionName.setText("");

        refreshSessionList();
        processAndShowSession(savedFile);
    }

    private void refreshSessionList() {
        listedSessionFiles.clear();
        sessionsAdapter.clear();

        List<File> files = sessionRepository.listSessions(this);
        listedSessionFiles.addAll(files);

        if (files.isEmpty()) {
            tvSelectedSession.setText("Sesion seleccionada: ninguna");
            sessionsAdapter.notifyDataSetChanged();
            return;
        }

        for (File file : files) {
            sessionsAdapter.add(formatSessionRow(file));
        }

        sessionsAdapter.notifyDataSetChanged();
    }

    private String formatSessionRow(File file) {
        return file.getName() + " (" + formatFileSize(file.length()) + ")";
    }

    private String formatFileSize(long bytes) {
        if (bytes < 1024) {
            return bytes + " B";
        }
        double kb = bytes / 1024.0;
        if (kb < 1024) {
            return String.format(Locale.US, "%.1f KB", kb);
        }
        double mb = kb / 1024.0;
        return String.format(Locale.US, "%.2f MB", mb);
    }

    private void showSessionOptions(View anchor, File file) {
        PopupMenu popup = new PopupMenu(this, anchor);
        Menu menu = popup.getMenu();
        menu.add(0, ACTION_PROCESS, 0, "Procesar");
        menu.add(0, ACTION_OPEN_DETAIL, 1, "Abrir detalle");
        menu.add(0, ACTION_INFO, 2, "Ver info");
        menu.add(0, ACTION_RENAME, 3, "Renombrar");
        menu.add(0, ACTION_DELETE, 4, "Eliminar");

        popup.setOnMenuItemClickListener(item -> {
            int id = item.getItemId();
            if (id == ACTION_PROCESS) {
                processAndShowSession(file);
                return true;
            }
            if (id == ACTION_OPEN_DETAIL) {
                openSessionDetail(file);
                return true;
            }
            if (id == ACTION_INFO) {
                showSessionInfo(file);
                return true;
            }
            if (id == ACTION_RENAME) {
                showRenameDialog(file);
                return true;
            }
            if (id == ACTION_DELETE) {
                confirmDelete(file);
                return true;
            }
            return false;
        });

        popup.show();
    }

    private void showSessionInfo(File file) {
        String info = String.format(
                Locale.US,
                "Nombre: %s\nTamano: %s\nUltima modificacion: %s\nRuta: %s",
                file.getName(),
                formatFileSize(file.length()),
                new Date(file.lastModified()).toString(),
                file.getAbsolutePath()
        );

        new AlertDialog.Builder(this)
                .setTitle("Info de sesion")
                .setMessage(info)
                .setPositiveButton("OK", null)
                .show();
    }

    private void showRenameDialog(File file) {
        EditText input = new EditText(this);
        input.setInputType(InputType.TYPE_CLASS_TEXT);
        String current = file.getName().toLowerCase(Locale.US).endsWith(".csv")
                ? file.getName().substring(0, file.getName().length() - 4)
                : file.getName();
        input.setText(current);
        input.setSelection(input.getText().length());

        new AlertDialog.Builder(this)
                .setTitle("Renombrar sesion")
                .setView(input)
                .setPositiveButton("Guardar", (dialog, which) -> {
                    try {
                        sessionRepository.renameSession(file, input.getText().toString());
                        refreshSessionList();
                    } catch (IOException e) {
                        Toast.makeText(this, e.getMessage(), Toast.LENGTH_LONG).show();
                    }
                })
                .setNegativeButton("Cancelar", null)
                .show();
    }

    private void confirmDelete(File file) {
        new AlertDialog.Builder(this)
                .setTitle("Eliminar sesion")
                .setMessage("Seguro que quieres eliminar:\n" + file.getName() + "?")
                .setPositiveButton("Eliminar", (dialog, which) -> {
                    try {
                        sessionRepository.deleteSession(file);
                        refreshSessionList();
                        tvSelectedSession.setText("Sesion eliminada: " + file.getName());
                        chartSpectrum.clear();
                        chartSeries.clear();
                    } catch (IOException e) {
                        Toast.makeText(this, e.getMessage(), Toast.LENGTH_LONG).show();
                    }
                })
                .setNegativeButton("Cancelar", null)
                .show();
    }

    private void setupCharts() {
        Description d1 = new Description();
        d1.setText("S_eta(f)");
        chartSpectrum.setDescription(d1);
        chartSpectrum.setNoDataText("Sin espectro");
        chartSpectrum.getAxisRight().setEnabled(false);
        chartSpectrum.getXAxis().setTextColor(Color.DKGRAY);
        chartSpectrum.getAxisLeft().setTextColor(Color.DKGRAY);
        chartSpectrum.getLegend().setTextColor(Color.DKGRAY);

        Description d2 = new Description();
        d2.setText("az dinamica (preview)");
        chartSeries.setDescription(d2);
        chartSeries.setNoDataText("Sin serie temporal");
        chartSeries.getAxisRight().setEnabled(false);
        chartSeries.getXAxis().setTextColor(Color.DKGRAY);
        chartSeries.getAxisLeft().setTextColor(Color.DKGRAY);
        chartSeries.getLegend().setTextColor(Color.DKGRAY);
    }

    private void renderCharts(OmbResult result) {
        List<Entry> spectrumEntries = new ArrayList<>();
        for (int i = 0; i < result.spectrumFreqHz.length; i++) {
            spectrumEntries.add(new Entry((float) result.spectrumFreqHz[i], (float) result.spectrumElevation[i]));
        }

        LineDataSet spectrumSet = new LineDataSet(spectrumEntries, "S_eta(f) [m^2/Hz]");
        spectrumSet.setColor(Color.parseColor("#1565C0"));
        spectrumSet.setLineWidth(2f);
        spectrumSet.setDrawCircles(false);
        spectrumSet.setDrawValues(false);
        chartSpectrum.setData(new LineData(spectrumSet));
        chartSpectrum.invalidate();

        List<Entry> seriesEntries = new ArrayList<>();
        for (int i = 0; i < result.previewTimeSec.length; i++) {
            seriesEntries.add(new Entry((float) result.previewTimeSec[i], (float) result.previewAzDyn[i]));
        }

        LineDataSet seriesSet = new LineDataSet(seriesEntries, "az_dyn [m/s^2]");
        seriesSet.setColor(Color.parseColor("#2E7D32"));
        seriesSet.setLineWidth(1.8f);
        seriesSet.setDrawCircles(false);
        seriesSet.setDrawValues(false);
        chartSeries.setData(new LineData(seriesSet));
        chartSeries.invalidate();
    }

    private void processAndShowSession(File file) {
        tvSelectedSession.setText("Procesando sesion...\n" + file.getName());

        final OmbProcessor.Mode mode = rbModeNative.isChecked() && !rbModeOmb10.isChecked()
                ? OmbProcessor.Mode.NATIVE_FS
                : OmbProcessor.Mode.OMB_10HZ;

        new Thread(() -> {
            try {
                SessionData data = sessionRepository.readSession(file);
                OmbResult result = ombProcessor.process(data, mode);

                String text = String.format(
                        Locale.US,
                        "Sesion: %s\n" +
                                "Muestras: %d | Duracion: %.2f s\n" +
                                "Modo: %s\n" +
                                "Hs: %.5f m\n" +
                                "Tz: %.5f s\n" +
                                "Tc: %.5f s\n" +
                                "Tp: %.5f s\n" +
                                "fs proc: %.3f Hz | N uniforme: %d | segmentos: %d",
                        file.getName(),
                        result.inputN,
                        result.inputDurationSec,
                        result.modeLabel,
                        result.hs,
                        result.tzSec,
                        result.tcSec,
                        result.tpSec,
                        result.processingFsHz,
                        result.uniformN,
                        result.segments
                );

                runOnUiThread(() -> {
                    tvSelectedSession.setText(text);
                    renderCharts(result);
                });
            } catch (Exception e) {
                runOnUiThread(() -> {
                    tvSelectedSession.setText("Error procesando sesion: " + file.getName() + "\n" + e.getMessage());
                    chartSpectrum.clear();
                    chartSeries.clear();
                });
            }
        }).start();
    }

    private OmbProcessor.Mode getSelectedMode() {
        return rbModeNative.isChecked() && !rbModeOmb10.isChecked()
                ? OmbProcessor.Mode.NATIVE_FS
                : OmbProcessor.Mode.OMB_10HZ;
    }

    private void openSessionDetail(File file) {
        Intent intent = new Intent(this, SessionDetailActivity.class);
        intent.putExtra(SessionDetailActivity.EXTRA_SESSION_PATH, file.getAbsolutePath());
        intent.putExtra(SessionDetailActivity.EXTRA_PROCESS_MODE, getSelectedMode().name());
        startActivity(intent);
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (!capturing) {
            return;
        }

        int type = event.sensor.getType();
        if (type == Sensor.TYPE_ACCELEROMETER) {
            ax = event.values[0];
            ay = event.values[1];
            az = event.values[2];
        } else if (type == Sensor.TYPE_GYROSCOPE) {
            gx = event.values[0];
            gy = event.values[1];
            gz = event.values[2];
        } else {
            return;
        }

        long tMs = (SystemClock.elapsedRealtimeNanos() - t0ElapsedNs) / 1_000_000L;
        currentSamples.add(new ImuSample(tMs, ax, ay, az, gx, gy, gz));

        if (tMs - lastUiUpdateMs >= UI_UPDATE_INTERVAL_MS) {
            long remainingMs = Math.max(0L, captureDurationMs - tMs);
            tvStatus.setText(String.format(
                    Locale.US,
                    "Capturando IMU | t=%d ms | restantes=%.1f s | muestras=%d\n" +
                            "a=[%.3f, %.3f, %.3f] m/s^2 | g=[%.3f, %.3f, %.3f] rad/s",
                    tMs,
                    remainingMs / 1000.0,
                    currentSamples.size(),
                    ax, ay, az,
                    gx, gy, gz
            ));
            lastUiUpdateMs = tMs;
        }

        if (tMs >= captureDurationMs) {
            stopCapture(false);
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        // No se usa por ahora.
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (capturing) {
            stopCapture(true);
        }
    }
}
