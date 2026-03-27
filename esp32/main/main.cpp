#include <cstring>
#include <cstdio>
#include <cmath>
#include <cstdlib>

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "driver/i2c.h"
#include "esp_event.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "nvs_flash.h"

static const char *TAG = "BOYA_ESP32";
static constexpr const char *WIFI_AP_SSID = "BOYA_ESP32";
static constexpr const char *WIFI_AP_PASSWORD = "boya1234";

// Ajusta estos pines a tu placa si son distintos.
static constexpr i2c_port_t I2C_OLED_PORT = I2C_NUM_0;
static constexpr i2c_port_t I2C_MCU_PORT = I2C_NUM_1;
// Heltec WiFi LoRa 32 V3 (OLED interna): SDA=17, SCL=18.
static constexpr gpio_num_t I2C_OLED_SDA_PIN = GPIO_NUM_17;
static constexpr gpio_num_t I2C_OLED_SCL_PIN = GPIO_NUM_18;
// Bus dedicado MCU externo (separado de OLED).
static constexpr gpio_num_t I2C_MCU_SDA_PIN = GPIO_NUM_41;
static constexpr gpio_num_t I2C_MCU_SCL_PIN = GPIO_NUM_42;
static constexpr uint32_t I2C_FREQ_HZ = 400000;
static constexpr gpio_num_t OLED_RST_PIN = GPIO_NUM_21;
// En Heltec V3, Vext suele activarse en nivel bajo por GPIO36.
static constexpr gpio_num_t VEXT_CTRL_PIN = GPIO_NUM_36;
static constexpr int VEXT_ACTIVE_LEVEL = 0;

static constexpr uint8_t MPU6050_ADDR_0 = 0x68;
static constexpr uint8_t MPU6050_ADDR_1 = 0x69;
static constexpr uint8_t MPU6050_REG_WHO_AM_I = 0x75;
static constexpr uint8_t MPU6050_WHO_AM_I_EXPECTED = 0x68;
static constexpr float GRAVITY = 9.81f;
static constexpr float PI_F = 3.14159265358979323846f;
static constexpr uint8_t OLED_ADDR_PRIMARY = 0x3C;
static constexpr uint8_t OLED_ADDR_SECONDARY = 0x3D;
static constexpr int OLED_WIDTH = 128;
static constexpr int OLED_HEIGHT = 64;
static constexpr int OLED_PAGES = OLED_HEIGHT / 8;

static bool g_i2c_oled_ready = false;
static bool g_i2c_mcu_ready = false;
static bool g_imu_detected = false;
static uint8_t g_imu_whoami = 0x00;
static uint8_t g_imu_addr = 0x00;
static SemaphoreHandle_t g_i2c_oled_mutex = nullptr;
static SemaphoreHandle_t g_i2c_mcu_mutex = nullptr;
static bool g_oled_ready = false;
static uint8_t g_oled_addr = OLED_ADDR_PRIMARY;
static uint8_t g_oled_fb[OLED_WIDTH * OLED_PAGES] = {0};
static int g_heartbeat_count = 0;
static char g_last_action[32] = "BOOT";
static int g_ap_clients = 0;
static bool g_capture_active = false;
static int64_t g_capture_start_us = 0;
static bool g_download_active = false;
static int g_download_progress_pct = 0;

struct SimCaptureState {
    bool configured = false;
    int duration_sec = 30;
    char session_name[64] = "session";
};

static SimCaptureState g_sim_capture;

static void board_power_prepare()
{
    gpio_config_t io_conf{};
    io_conf.pin_bit_mask = (1ULL << VEXT_CTRL_PIN) | (1ULL << OLED_RST_PIN);
    io_conf.mode = GPIO_MODE_OUTPUT;
    io_conf.pull_up_en = GPIO_PULLUP_DISABLE;
    io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
    io_conf.intr_type = GPIO_INTR_DISABLE;
    ESP_ERROR_CHECK(gpio_config(&io_conf));

    // Habilita Vext (activo en bajo en Heltec V3).
    ESP_ERROR_CHECK(gpio_set_level(VEXT_CTRL_PIN, VEXT_ACTIVE_LEVEL));
    vTaskDelay(pdMS_TO_TICKS(20));

    // Reset hardware de OLED.
    ESP_ERROR_CHECK(gpio_set_level(OLED_RST_PIN, 0));
    vTaskDelay(pdMS_TO_TICKS(20));
    ESP_ERROR_CHECK(gpio_set_level(OLED_RST_PIN, 1));
    vTaskDelay(pdMS_TO_TICKS(20));
}

// 5x7 font. Cada caracter son 5 columnas (LSB arriba).
static constexpr uint8_t FONT5X7[][5] = {
        {0x00,0x00,0x00,0x00,0x00}, // ' '
        {0x00,0x00,0x5F,0x00,0x00}, // '!'
        {0x00,0x07,0x00,0x07,0x00}, // '"'
        {0x14,0x7F,0x14,0x7F,0x14}, // '#'
        {0x24,0x2A,0x7F,0x2A,0x12}, // '$'
        {0x23,0x13,0x08,0x64,0x62}, // '%'
        {0x36,0x49,0x55,0x22,0x50}, // '&'
        {0x00,0x05,0x03,0x00,0x00}, // '''
        {0x00,0x1C,0x22,0x41,0x00}, // '('
        {0x00,0x41,0x22,0x1C,0x00}, // ')'
        {0x14,0x08,0x3E,0x08,0x14}, // '*'
        {0x08,0x08,0x3E,0x08,0x08}, // '+'
        {0x00,0x50,0x30,0x00,0x00}, // ','
        {0x08,0x08,0x08,0x08,0x08}, // '-'
        {0x00,0x60,0x60,0x00,0x00}, // '.'
        {0x20,0x10,0x08,0x04,0x02}, // '/'
        {0x3E,0x51,0x49,0x45,0x3E}, // '0'
        {0x00,0x42,0x7F,0x40,0x00}, // '1'
        {0x42,0x61,0x51,0x49,0x46}, // '2'
        {0x21,0x41,0x45,0x4B,0x31}, // '3'
        {0x18,0x14,0x12,0x7F,0x10}, // '4'
        {0x27,0x45,0x45,0x45,0x39}, // '5'
        {0x3C,0x4A,0x49,0x49,0x30}, // '6'
        {0x01,0x71,0x09,0x05,0x03}, // '7'
        {0x36,0x49,0x49,0x49,0x36}, // '8'
        {0x06,0x49,0x49,0x29,0x1E}, // '9'
        {0x00,0x36,0x36,0x00,0x00}, // ':'
        {0x00,0x56,0x36,0x00,0x00}, // ';'
        {0x08,0x14,0x22,0x41,0x00}, // '<'
        {0x14,0x14,0x14,0x14,0x14}, // '='
        {0x00,0x41,0x22,0x14,0x08}, // '>'
        {0x02,0x01,0x51,0x09,0x06}, // '?'
        {0x32,0x49,0x79,0x41,0x3E}, // '@'
        {0x7E,0x11,0x11,0x11,0x7E}, // 'A'
        {0x7F,0x49,0x49,0x49,0x36}, // 'B'
        {0x3E,0x41,0x41,0x41,0x22}, // 'C'
        {0x7F,0x41,0x41,0x22,0x1C}, // 'D'
        {0x7F,0x49,0x49,0x49,0x41}, // 'E'
        {0x7F,0x09,0x09,0x09,0x01}, // 'F'
        {0x3E,0x41,0x49,0x49,0x7A}, // 'G'
        {0x7F,0x08,0x08,0x08,0x7F}, // 'H'
        {0x00,0x41,0x7F,0x41,0x00}, // 'I'
        {0x20,0x40,0x41,0x3F,0x01}, // 'J'
        {0x7F,0x08,0x14,0x22,0x41}, // 'K'
        {0x7F,0x40,0x40,0x40,0x40}, // 'L'
        {0x7F,0x02,0x0C,0x02,0x7F}, // 'M'
        {0x7F,0x04,0x08,0x10,0x7F}, // 'N'
        {0x3E,0x41,0x41,0x41,0x3E}, // 'O'
        {0x7F,0x09,0x09,0x09,0x06}, // 'P'
        {0x3E,0x41,0x51,0x21,0x5E}, // 'Q'
        {0x7F,0x09,0x19,0x29,0x46}, // 'R'
        {0x46,0x49,0x49,0x49,0x31}, // 'S'
        {0x01,0x01,0x7F,0x01,0x01}, // 'T'
        {0x3F,0x40,0x40,0x40,0x3F}, // 'U'
        {0x1F,0x20,0x40,0x20,0x1F}, // 'V'
        {0x3F,0x40,0x38,0x40,0x3F}, // 'W'
        {0x63,0x14,0x08,0x14,0x63}, // 'X'
        {0x07,0x08,0x70,0x08,0x07}, // 'Y'
        {0x61,0x51,0x49,0x45,0x43}, // 'Z'
        {0x00,0x7F,0x41,0x41,0x00}, // '['
        {0x02,0x04,0x08,0x10,0x20}, // '\'
        {0x00,0x41,0x41,0x7F,0x00}, // ']'
        {0x04,0x02,0x01,0x02,0x04}, // '^'
        {0x40,0x40,0x40,0x40,0x40}, // '_'
        {0x00,0x03,0x05,0x00,0x00}, // '`'
        {0x20,0x54,0x54,0x54,0x78}, // 'a'
        {0x7F,0x48,0x44,0x44,0x38}, // 'b'
        {0x38,0x44,0x44,0x44,0x20}, // 'c'
        {0x38,0x44,0x44,0x48,0x7F}, // 'd'
        {0x38,0x54,0x54,0x54,0x18}, // 'e'
        {0x08,0x7E,0x09,0x01,0x02}, // 'f'
        {0x0C,0x52,0x52,0x52,0x3E}, // 'g'
        {0x7F,0x08,0x04,0x04,0x78}, // 'h'
        {0x00,0x44,0x7D,0x40,0x00}, // 'i'
        {0x20,0x40,0x44,0x3D,0x00}, // 'j'
        {0x7F,0x10,0x28,0x44,0x00}, // 'k'
        {0x00,0x41,0x7F,0x40,0x00}, // 'l'
        {0x7C,0x04,0x18,0x04,0x78}, // 'm'
        {0x7C,0x08,0x04,0x04,0x78}, // 'n'
        {0x38,0x44,0x44,0x44,0x38}, // 'o'
        {0x7C,0x14,0x14,0x14,0x08}, // 'p'
        {0x08,0x14,0x14,0x18,0x7C}, // 'q'
        {0x7C,0x08,0x04,0x04,0x08}, // 'r'
        {0x48,0x54,0x54,0x54,0x20}, // 's'
        {0x04,0x3F,0x44,0x40,0x20}, // 't'
        {0x3C,0x40,0x40,0x20,0x7C}, // 'u'
        {0x1C,0x20,0x40,0x20,0x1C}, // 'v'
        {0x3C,0x40,0x30,0x40,0x3C}, // 'w'
        {0x44,0x28,0x10,0x28,0x44}, // 'x'
        {0x0C,0x50,0x50,0x50,0x3C}, // 'y'
        {0x44,0x64,0x54,0x4C,0x44}, // 'z'
        {0x00,0x08,0x36,0x41,0x00}, // '{'
        {0x00,0x00,0x7F,0x00,0x00}, // '|'
        {0x00,0x41,0x36,0x08,0x00}, // '}'
        {0x08,0x04,0x08,0x10,0x08}  // '~'
};

static bool i2c_lock(SemaphoreHandle_t mutex, TickType_t timeout)
{
    if (mutex == nullptr) {
        return false;
    }
    return xSemaphoreTake(mutex, timeout) == pdTRUE;
}

static void i2c_unlock(SemaphoreHandle_t mutex)
{
    if (mutex != nullptr) {
        xSemaphoreGive(mutex);
    }
}

static esp_err_t oled_write_cmd(uint8_t addr, uint8_t cmd)
{
    uint8_t payload[2] = {0x00, cmd};
    return i2c_master_write_to_device(I2C_OLED_PORT, addr, payload, sizeof(payload), pdMS_TO_TICKS(150));
}

static esp_err_t oled_write_data(uint8_t addr, const uint8_t *data, size_t len)
{
    if (data == nullptr || len == 0) {
        return ESP_OK;
    }

    const size_t max_chunk = 16;
    uint8_t buffer[max_chunk + 1];
    buffer[0] = 0x40;

    size_t offset = 0;
    while (offset < len) {
        size_t n = (len - offset > max_chunk) ? max_chunk : (len - offset);
        std::memcpy(buffer + 1, data + offset, n);
        esp_err_t err = i2c_master_write_to_device(I2C_OLED_PORT, addr, buffer, n + 1, pdMS_TO_TICKS(150));
        if (err != ESP_OK) {
            return err;
        }
        offset += n;
    }
    return ESP_OK;
}

static void oled_clear_fb()
{
    std::memset(g_oled_fb, 0, sizeof(g_oled_fb));
}

static void oled_draw_char(int x, int page, char c)
{
    if (page < 0 || page >= OLED_PAGES) {
        return;
    }
    if (x < 0 || x + 5 >= OLED_WIDTH) {
        return;
    }

    char ch = c;
    if (ch < 32 || ch > 126) {
        ch = '?';
    }
    const uint8_t *glyph = FONT5X7[ch - 32];
    int base = page * OLED_WIDTH + x;
    for (int i = 0; i < 5; ++i) {
        g_oled_fb[base + i] = glyph[i];
    }
    g_oled_fb[base + 5] = 0x00;
}

static void oled_draw_text(int x, int page, const char *text)
{
    if (text == nullptr) {
        return;
    }
    int cx = x;
    while (*text != '\0' && cx + 5 < OLED_WIDTH) {
        oled_draw_char(cx, page, *text++);
        cx += 6;
    }
}

static esp_err_t oled_flush_locked()
{
    for (int page = 0; page < OLED_PAGES; ++page) {
        esp_err_t err = oled_write_cmd(g_oled_addr, static_cast<uint8_t>(0xB0 + page));
        if (err != ESP_OK) return err;
        err = oled_write_cmd(g_oled_addr, 0x00);
        if (err != ESP_OK) return err;
        err = oled_write_cmd(g_oled_addr, 0x10);
        if (err != ESP_OK) return err;
        const uint8_t *line = &g_oled_fb[page * OLED_WIDTH];
        err = oled_write_data(g_oled_addr, line, OLED_WIDTH);
        if (err != ESP_OK) return err;
    }
    return ESP_OK;
}

static void oled_set_last_action(const char *action)
{
    if (action == nullptr || action[0] == '\0') {
        return;
    }
    std::snprintf(g_last_action, sizeof(g_last_action), "%s", action);
}

static int get_capture_elapsed_sec()
{
    if (!g_capture_active || g_capture_start_us <= 0) {
        return 0;
    }
    int64_t now = esp_timer_get_time();
    if (now <= g_capture_start_us) {
        return 0;
    }
    return static_cast<int>((now - g_capture_start_us) / 1000000LL);
}

static void update_capture_runtime_state()
{
    if (!g_capture_active) {
        return;
    }
    int elapsed = get_capture_elapsed_sec();
    if (elapsed >= g_sim_capture.duration_sec) {
        g_capture_active = false;
        oled_set_last_action("CAP DONE");
    }
}

static void make_progress_bar(char *out, size_t out_len, int pct)
{
    if (out == nullptr || out_len < 3) {
        return;
    }
    int p = pct;
    if (p < 0) p = 0;
    if (p > 100) p = 100;
    const int bars = 10;
    int fill = (p * bars) / 100;
    std::snprintf(out, out_len, "[..........]");
    for (int i = 0; i < fill && i < bars; ++i) {
        out[1 + i] = '#';
    }
}

static void oled_render_status()
{
    if (!g_oled_ready) {
        return;
    }

    update_capture_runtime_state();
    char line0[32], line1[32], line2[32], line3[32], line4[32], line5[32], line6[32], line7[32];
    char bar[16];
    make_progress_bar(bar, sizeof(bar), g_download_progress_pct);
    int cap_elapsed = get_capture_elapsed_sec();
    if (cap_elapsed > g_sim_capture.duration_sec) {
        cap_elapsed = g_sim_capture.duration_sec;
    }

    std::snprintf(line0, sizeof(line0), "BOYA ESP32");
    std::snprintf(line1, sizeof(line1), "AP:%s C:%d", WIFI_AP_SSID, g_ap_clients);
    std::snprintf(line2, sizeof(line2), "IMU:%s A%02X W%02X", g_imu_detected ? "OK" : "NO", g_imu_addr, g_imu_whoami);
    std::snprintf(
            line3,
            sizeof(line3),
            "CAP:%s %4d/%ds",
            g_capture_active ? "ON" : "OFF",
            cap_elapsed,
            g_sim_capture.duration_sec);
    std::snprintf(line4, sizeof(line4), "NAME:%.14s", g_sim_capture.session_name);
    std::snprintf(line5, sizeof(line5), "LAST:%.14s", g_last_action);
    std::snprintf(line6, sizeof(line6), "DL:%3d%% %s", g_download_progress_pct, bar);
    std::snprintf(line7, sizeof(line7), "IP:192.168.4.1 H:%d", g_heartbeat_count);

    oled_clear_fb();
    oled_draw_text(0, 0, line0);
    oled_draw_text(0, 1, line1);
    oled_draw_text(0, 2, line2);
    oled_draw_text(0, 3, line3);
    oled_draw_text(0, 4, line4);
    oled_draw_text(0, 5, line5);
    oled_draw_text(0, 6, line6);
    oled_draw_text(0, 7, line7);

    if (!i2c_lock(g_i2c_oled_mutex, pdMS_TO_TICKS(250))) {
        return;
    }
    esp_err_t err = oled_flush_locked();
    i2c_unlock(g_i2c_oled_mutex);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "Fallo escribiendo OLED: %s", esp_err_to_name(err));
        g_oled_ready = false;
    }
}

static bool oled_init_try_addr(uint8_t addr)
{
    const uint8_t init_seq[] = {
            0xAE,       // display off
            0xD5, 0x80, // clock
            0xA8, 0x3F, // multiplex 64
            0xD3, 0x00, // offset
            0x40,       // start line
            0x8D, 0x14, // charge pump on
            0x20, 0x00, // horizontal addressing
            0xA1,       // segment remap
            0xC8,       // COM scan dec
            0xDA, 0x12, // COM pins
            0x81, 0x8F, // contrast
            0xD9, 0xF1, // pre-charge
            0xDB, 0x40, // vcomh
            0xA4,       // display all on resume
            0xA6,       // normal display
            0x2E,       // deactivate scroll
            0xAF        // display on
    };

    for (size_t i = 0; i < sizeof(init_seq); ++i) {
        esp_err_t err = oled_write_cmd(addr, init_seq[i]);
        if (err != ESP_OK) {
            return false;
        }
    }
    g_oled_addr = addr;
    return true;
}

static bool init_oled_display()
{
    if (!g_i2c_oled_ready) {
        return false;
    }
    if (!i2c_lock(g_i2c_oled_mutex, pdMS_TO_TICKS(300))) {
        return false;
    }

    bool ok = oled_init_try_addr(OLED_ADDR_PRIMARY) || oled_init_try_addr(OLED_ADDR_SECONDARY);
    i2c_unlock(g_i2c_oled_mutex);
    if (!ok) {
        ESP_LOGW(TAG, "OLED no detectada en 0x%02X ni 0x%02X", OLED_ADDR_PRIMARY, OLED_ADDR_SECONDARY);
        return false;
    }

    ESP_LOGI(TAG, "OLED detectada en 0x%02X", g_oled_addr);
    oled_set_last_action("OLED READY");
    return true;
}

static bool init_i2c_bus(i2c_port_t port, gpio_num_t sda_pin, gpio_num_t scl_pin, SemaphoreHandle_t *mutex_out, const char *label)
{
    if (mutex_out == nullptr) {
        return false;
    }
    if (*mutex_out == nullptr) {
        *mutex_out = xSemaphoreCreateMutex();
        if (*mutex_out == nullptr) {
            ESP_LOGW(TAG, "No se pudo crear mutex I2C");
            return false;
        }
    }

    i2c_config_t conf{};
    conf.mode = I2C_MODE_MASTER;
    conf.sda_io_num = sda_pin;
    conf.scl_io_num = scl_pin;
    conf.sda_pullup_en = GPIO_PULLUP_ENABLE;
    conf.scl_pullup_en = GPIO_PULLUP_ENABLE;
    conf.master.clk_speed = I2C_FREQ_HZ;

    esp_err_t err = i2c_param_config(port, &conf);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "I2C param config fallo: %s", esp_err_to_name(err));
        return false;
    }

    err = i2c_driver_install(port, I2C_MODE_MASTER, 0, 0, 0);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "I2C driver install fallo: %s", esp_err_to_name(err));
        return false;
    }

    ESP_LOGI(TAG, "I2C %s inicializado (port=%d SDA=%d, SCL=%d)", label, static_cast<int>(port), sda_pin, scl_pin);
    return true;
}

static void i2c_scan_bus(i2c_port_t port, SemaphoreHandle_t mutex, const char *label)
{
    if (mutex == nullptr) {
        return;
    }
    if (!i2c_lock(mutex, pdMS_TO_TICKS(400))) {
        ESP_LOGW(TAG, "No se pudo tomar mutex I2C para escaneo");
        return;
    }

    int found = 0;
    for (uint8_t addr = 0x03; addr <= 0x77; ++addr) {
        i2c_cmd_handle_t cmd = i2c_cmd_link_create();
        if (cmd == nullptr) {
            continue;
        }
        i2c_master_start(cmd);
        i2c_master_write_byte(cmd, static_cast<uint8_t>((addr << 1) | I2C_MASTER_WRITE), true);
        i2c_master_stop(cmd);
        esp_err_t err = i2c_master_cmd_begin(port, cmd, pdMS_TO_TICKS(30));
        i2c_cmd_link_delete(cmd);
        if (err == ESP_OK) {
            ESP_LOGI(TAG, "I2C %s detectado: 0x%02X", label, addr);
            found++;
        }
    }
    i2c_unlock(mutex);

    if (found == 0) {
        ESP_LOGW(TAG, "Escaneo I2C %s: no se detecto ningun dispositivo", label);
    } else {
        ESP_LOGI(TAG, "Escaneo I2C %s completado. Dispositivos encontrados: %d", label, found);
    }
}

static bool mpu6050_read_whoami(uint8_t addr, uint8_t *out)
{
    uint8_t reg = MPU6050_REG_WHO_AM_I;
    uint8_t value = 0x00;
    if (!i2c_lock(g_i2c_mcu_mutex, pdMS_TO_TICKS(250))) {
        return false;
    }
    esp_err_t err = i2c_master_write_read_device(
            I2C_MCU_PORT,
            addr,
            &reg,
            1,
            &value,
            1,
            pdMS_TO_TICKS(200));
    i2c_unlock(g_i2c_mcu_mutex);

    if (err != ESP_OK) {
        ESP_LOGW(TAG, "No se pudo leer WHO_AM_I (addr=0x%02X): %s", addr, esp_err_to_name(err));
        return false;
    }

    *out = value;
    return true;
}

static void wifi_event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data)
{
    if (event_base != WIFI_EVENT) {
        return;
    }

    if (event_id == WIFI_EVENT_AP_STACONNECTED) {
        g_ap_clients++;
        oled_set_last_action("STA JOIN");
        ESP_LOGI(TAG, "Cliente conectado al AP. total=%d", g_ap_clients);
        oled_render_status();
    } else if (event_id == WIFI_EVENT_AP_STADISCONNECTED) {
        if (g_ap_clients > 0) {
            g_ap_clients--;
        }
        oled_set_last_action("STA LEFT");
        ESP_LOGI(TAG, "Cliente desconectado del AP. total=%d", g_ap_clients);
        oled_render_status();
    }
}

static void probe_imu_once()
{
    uint8_t whoami = 0x00;
    uint8_t found_addr = 0x00;

    if (mpu6050_read_whoami(MPU6050_ADDR_0, &whoami)) {
        found_addr = MPU6050_ADDR_0;
    } else if (mpu6050_read_whoami(MPU6050_ADDR_1, &whoami)) {
        found_addr = MPU6050_ADDR_1;
    } else {
        g_imu_detected = false;
        g_imu_addr = 0x00;
        g_imu_whoami = 0x00;
        oled_set_last_action("IMU MISS");
        oled_render_status();
        return;
    }

    g_imu_addr = found_addr;
    g_imu_whoami = whoami;
    g_imu_detected = (whoami == MPU6050_WHO_AM_I_EXPECTED);
    if (g_imu_detected) {
        ESP_LOGI(TAG, "IMU detectado (addr=0x%02X WHO_AM_I=0x%02X)", g_imu_addr, whoami);
        oled_set_last_action("IMU OK");
    } else {
        ESP_LOGW(TAG, "IMU respondio (addr=0x%02X) pero WHO_AM_I inesperado: 0x%02X", g_imu_addr, whoami);
        oled_set_last_action("IMU WHOAMI");
    }
    oled_render_status();
}

static esp_err_t handle_ping(httpd_req_t *req)
{
    const char *resp = "{\"ok\":true,\"msg\":\"pong\"}";
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_send(req, resp, HTTPD_RESP_USE_STRLEN);
}

static uint32_t hash_name(const char *s)
{
    uint32_t h = 2166136261u;
    if (s == nullptr) {
        return h;
    }
    while (*s != '\0') {
        h ^= static_cast<uint8_t>(*s++);
        h *= 16777619u;
    }
    return h;
}

static int clamp_int(int v, int lo, int hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static void parse_capture_query(httpd_req_t *req, int *duration_sec_out, char *name_out, size_t name_out_len)
{
    *duration_sec_out = 30;
    if (name_out_len > 0) {
        std::snprintf(name_out, name_out_len, "session");
    }

    size_t qlen = httpd_req_get_url_query_len(req) + 1;
    if (qlen <= 1) {
        return;
    }

    char *query = static_cast<char *>(std::malloc(qlen));
    if (query == nullptr) {
        return;
    }
    query[0] = '\0';

    if (httpd_req_get_url_query_str(req, query, qlen) == ESP_OK) {
        char duration_buf[16] = {0};
        char name_buf[64] = {0};

        if (httpd_query_key_value(query, "duration", duration_buf, sizeof(duration_buf)) == ESP_OK) {
            int v = std::atoi(duration_buf);
            *duration_sec_out = clamp_int(v, 1, 60 * 60);
        }
        if (httpd_query_key_value(query, "name", name_buf, sizeof(name_buf)) == ESP_OK && name_buf[0] != '\0') {
            std::snprintf(name_out, name_out_len, "%s", name_buf);
        }
    }

    std::free(query);
}

static esp_err_t handle_capture_start(httpd_req_t *req)
{
    int duration_sec = 30;
    char session_name[64] = {0};
    parse_capture_query(req, &duration_sec, session_name, sizeof(session_name));

    g_sim_capture.configured = true;
    g_sim_capture.duration_sec = duration_sec;
    std::snprintf(g_sim_capture.session_name, sizeof(g_sim_capture.session_name), "%s", session_name);
    g_capture_active = true;
    g_capture_start_us = esp_timer_get_time();
    g_download_active = false;
    g_download_progress_pct = 0;

    char resp[192];
    std::snprintf(
            resp,
            sizeof(resp),
            "{\"ok\":true,\"capture\":\"started\",\"duration_sec\":%d,\"name\":\"%s\"}",
            g_sim_capture.duration_sec,
            g_sim_capture.session_name);

    ESP_LOGI(TAG, "Captura simulada start: duration=%ds name=%s", g_sim_capture.duration_sec, g_sim_capture.session_name);
    oled_set_last_action("CAP START");
    oled_render_status();
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_send(req, resp, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t handle_capture_stop(httpd_req_t *req)
{
    if (!g_sim_capture.configured) {
        g_sim_capture.configured = true;
    }
    g_capture_active = false;

    char resp[192];
    std::snprintf(
            resp,
            sizeof(resp),
            "{\"ok\":true,\"capture\":\"stopped\",\"duration_sec\":%d,\"name\":\"%s\"}",
            g_sim_capture.duration_sec,
            g_sim_capture.session_name);

    ESP_LOGI(TAG, "Captura simulada stop: duration=%ds name=%s", g_sim_capture.duration_sec, g_sim_capture.session_name);
    oled_set_last_action("CAP STOP");
    oled_render_status();
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_send(req, resp, HTTPD_RESP_USE_STRLEN);
}

static void synthesize_sample(int i, float fs, uint32_t seed, float *ax, float *ay, float *az, float *gx, float *gy, float *gz)
{
    float t = static_cast<float>(i) / fs;
    float f1 = 0.06f + static_cast<float>(seed % 7) * 0.003f;
    float f2 = 0.11f + static_cast<float>((seed / 7) % 9) * 0.002f;
    float ph2 = static_cast<float>(seed % 314) / 100.0f;

    float w1 = 2.0f * PI_F * f1;
    float w2 = 2.0f * PI_F * f2;

    float a1 = 0.020f; // amplitud de elevacion [m]
    float a2 = 0.010f;

    // aceleracion vertical dinamica aproximada de elevacion sintetica.
    float az_dyn = -a1 * w1 * w1 * std::sin(w1 * t) - a2 * w2 * w2 * std::sin(w2 * t + ph2);
    float n = 0.01f * std::sin(0.73f * t + static_cast<float>(seed % 11));

    *az = GRAVITY + az_dyn + n;
    *ax = 0.03f * std::sin(0.40f * t) + 0.005f * n;
    *ay = 0.03f * std::cos(0.33f * t) - 0.005f * n;
    *gx = 0.02f * std::sin(0.21f * t);
    *gy = 0.02f * std::cos(0.27f * t);
    *gz = 0.01f * std::sin(0.17f * t);
}

static esp_err_t handle_capture_download(httpd_req_t *req)
{
    if (!g_sim_capture.configured) {
        g_sim_capture.configured = true;
        g_sim_capture.duration_sec = 30;
        std::snprintf(g_sim_capture.session_name, sizeof(g_sim_capture.session_name), "session");
    }

    const int fs = 10;
    const int total = g_sim_capture.duration_sec * fs;
    uint32_t seed = hash_name(g_sim_capture.session_name);
    g_download_active = true;
    g_download_progress_pct = 0;
    oled_set_last_action("DL START");
    oled_render_status();

    httpd_resp_set_type(req, "text/csv");
    httpd_resp_set_hdr(req, "Content-Disposition", "attachment; filename=\"esp32_capture.csv\"");

    esp_err_t err = httpd_resp_send_chunk(req, "t_ms,ax,ay,az,gx,gy,gz\n", HTTPD_RESP_USE_STRLEN);
    if (err != ESP_OK) {
        return err;
    }

    char line[160];
    int last_reported_pct = -1;
    for (int i = 0; i < total; ++i) {
        float ax, ay, az, gx, gy, gz;
        synthesize_sample(i, static_cast<float>(fs), seed, &ax, &ay, &az, &gx, &gy, &gz);
        int t_ms = i * 1000 / fs;

        int n = std::snprintf(
                line,
                sizeof(line),
                "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                t_ms,
                ax, ay, az, gx, gy, gz);
        if (n <= 0) {
            continue;
        }
        err = httpd_resp_send_chunk(req, line, HTTPD_RESP_USE_STRLEN);
        if (err != ESP_OK) {
            g_download_active = false;
            g_download_progress_pct = 0;
            return err;
        }

        int pct = ((i + 1) * 100) / total;
        if (pct != last_reported_pct && (pct % 5 == 0 || pct == 100)) {
            g_download_progress_pct = pct;
            last_reported_pct = pct;
            oled_render_status();
        }
    }

    ESP_LOGI(TAG, "CSV sintetico servido: duration=%ds samples=%d name=%s", g_sim_capture.duration_sec, total, g_sim_capture.session_name);
    g_download_active = false;
    g_download_progress_pct = 100;
    g_capture_active = false;
    oled_set_last_action("CAP DOWN");
    oled_render_status();
    return httpd_resp_send_chunk(req, nullptr, 0);
}

static esp_err_t handle_status(httpd_req_t *req)
{
    update_capture_runtime_state();
    char resp[256];
    std::snprintf(
            resp,
            sizeof(resp),
            "{\"ok\":true,\"mode\":\"ap\",\"imu\":\"%s\",\"imu_addr\":\"0x%02X\",\"whoami\":\"0x%02X\",\"clients\":%d,\"capture_active\":%s,\"download_active\":%s,\"download_pct\":%d}",
            g_imu_detected ? "detected" : "not_detected",
            g_imu_addr,
            g_imu_whoami,
            g_ap_clients,
            g_capture_active ? "true" : "false",
            g_download_active ? "true" : "false",
            g_download_progress_pct);

    httpd_resp_set_type(req, "application/json");
    return httpd_resp_send(req, resp, HTTPD_RESP_USE_STRLEN);
}

static httpd_handle_t start_http_server()
{
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 80;

    httpd_handle_t server = nullptr;
    ESP_ERROR_CHECK(httpd_start(&server, &config));

    httpd_uri_t ping_uri{};
    ping_uri.uri = "/ping";
    ping_uri.method = HTTP_GET;
    ping_uri.handler = handle_ping;
    ESP_ERROR_CHECK(httpd_register_uri_handler(server, &ping_uri));

    httpd_uri_t status_uri{};
    status_uri.uri = "/status";
    status_uri.method = HTTP_GET;
    status_uri.handler = handle_status;
    ESP_ERROR_CHECK(httpd_register_uri_handler(server, &status_uri));

    httpd_uri_t capture_start_uri{};
    capture_start_uri.uri = "/capture/start";
    capture_start_uri.method = HTTP_GET;
    capture_start_uri.handler = handle_capture_start;
    ESP_ERROR_CHECK(httpd_register_uri_handler(server, &capture_start_uri));

    httpd_uri_t capture_stop_uri{};
    capture_stop_uri.uri = "/capture/stop";
    capture_stop_uri.method = HTTP_GET;
    capture_stop_uri.handler = handle_capture_stop;
    ESP_ERROR_CHECK(httpd_register_uri_handler(server, &capture_stop_uri));

    httpd_uri_t capture_download_uri{};
    capture_download_uri.uri = "/capture/download";
    capture_download_uri.method = HTTP_GET;
    capture_download_uri.handler = handle_capture_download;
    ESP_ERROR_CHECK(httpd_register_uri_handler(server, &capture_download_uri));

    ESP_LOGI(TAG, "HTTP listo: /ping /status /capture/start /capture/stop /capture/download");
    oled_set_last_action("HTTP READY");
    oled_render_status();
    return server;
}

static void init_wifi_ap()
{
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_ap();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, nullptr));

    wifi_config_t wifi_config{};
    std::strncpy(reinterpret_cast<char *>(wifi_config.ap.ssid), WIFI_AP_SSID, sizeof(wifi_config.ap.ssid) - 1);
    std::strncpy(reinterpret_cast<char *>(wifi_config.ap.password), WIFI_AP_PASSWORD, sizeof(wifi_config.ap.password) - 1);
    wifi_config.ap.ssid_len = std::strlen(WIFI_AP_SSID);
    wifi_config.ap.channel = 1;
    wifi_config.ap.max_connection = 4;
    wifi_config.ap.authmode = WIFI_AUTH_WPA2_PSK;

    if (std::strlen(WIFI_AP_PASSWORD) < 8) {
        wifi_config.ap.authmode = WIFI_AUTH_OPEN;
    }

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "WiFi AP activo. SSID=%s PASS=%s", WIFI_AP_SSID, WIFI_AP_PASSWORD);
    ESP_LOGI(TAG, "Conecta el movil y abre: http://192.168.4.1/ping");
    oled_set_last_action("AP READY");
    oled_render_status();
}

extern "C" void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_LOGI(TAG, "Inicio firmware ESP32-S3 - BOYA IMU");
    board_power_prepare();
    init_wifi_ap();
    start_http_server();

    g_i2c_oled_ready = init_i2c_bus(
            I2C_OLED_PORT,
            I2C_OLED_SDA_PIN,
            I2C_OLED_SCL_PIN,
            &g_i2c_oled_mutex,
            "OLED");
    if (g_i2c_oled_ready) {
        i2c_scan_bus(I2C_OLED_PORT, g_i2c_oled_mutex, "OLED");
        g_oled_ready = init_oled_display();
        if (!g_oled_ready) {
            ESP_LOGW(TAG, "Sin OLED. Continuamos sin feedback en pantalla.");
        }
        oled_render_status();
    } else {
        ESP_LOGW(TAG, "I2C OLED no listo. Continuamos sin feedback en pantalla.");
    }

    g_i2c_mcu_ready = init_i2c_bus(
            I2C_MCU_PORT,
            I2C_MCU_SDA_PIN,
            I2C_MCU_SCL_PIN,
            &g_i2c_mcu_mutex,
            "MCU");
    if (g_i2c_mcu_ready) {
        i2c_scan_bus(I2C_MCU_PORT, g_i2c_mcu_mutex, "MCU");
        probe_imu_once();
    } else {
        ESP_LOGW(TAG, "I2C MCU no listo. Seguimos sin IMU para no bloquear firmware.");
    }

    while (true) {
        // Reintento cada 15 s si no hay IMU.
        if (g_i2c_mcu_ready && !g_imu_detected && (g_heartbeat_count % 15 == 0)) {
            // Reintento periodico por si el IMU se conecta despues.
            probe_imu_once();
        }

        if (g_heartbeat_count % 5 == 0) {
            ESP_LOGI(
                    TAG,
                    "Heartbeat (AP+HTTP activos, clients=%d, imu=%s, whoami=0x%02X, cap=%s, dl=%d%%)",
                    g_ap_clients,
                    g_imu_detected ? "detected" : "not_detected",
                    g_imu_whoami,
                    g_capture_active ? "on" : "off",
                    g_download_progress_pct);
        }
        g_heartbeat_count++;
        if (!g_download_active && !g_capture_active) {
            oled_set_last_action("RUNNING");
        }
        oled_render_status();
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
