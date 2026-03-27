package tfg.udc.boya;

import android.content.Intent;
import android.os.Bundle;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

public class MenuActivity extends AppCompatActivity {

    private TextView tvEsp32State;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_menu);

        Button btnMobileMode = findViewById(R.id.btnMobileMode);
        Button btnEsp32Mode = findViewById(R.id.btnEsp32Mode);
        tvEsp32State = findViewById(R.id.tvEsp32State);

        btnMobileMode.setOnClickListener(v -> {
            Intent i = new Intent(MenuActivity.this, MobileModeActivity.class);
            startActivity(i);
        });

        // Abrimos siempre la pantalla ESP32; ahi ya se gestiona conectar/verificar.
        btnEsp32Mode.setOnClickListener(v -> openEsp32Mode());
    }

    @Override
    protected void onResume() {
        super.onResume();
        updateEsp32State();
    }

    private void updateEsp32State() {
        String state = Esp32ConnectionManager.getStateText(this);
        tvEsp32State.setText("Estado ESP32: " + state);
    }

    private void openEsp32Mode() {
        Intent i = new Intent(MenuActivity.this, Esp32ModeActivity.class);
        startActivity(i);
    }
}
