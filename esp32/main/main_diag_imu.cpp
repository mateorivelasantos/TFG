#include <cstdio>
#include <cstring>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "driver/i2c.h"
#include "esp_log.h"
#include "nvs_flash.h"

static const char *TAG = "IMU_DIAG";

static constexpr i2c_port_t I2C_OLED_PORT = I2C_NUM_0;
static constexpr i2c_port_t I2C_MCU_PORT = I2C_NUM_1;

static constexpr gpio_num_t I2C_OLED_SDA_PIN = GPIO_NUM_17;
static constexpr gpio_num_t I2C_OLED_SCL_PIN = GPIO_NUM_18;
static constexpr gpio_num_t I2C_MCU_SDA_PIN = GPIO_NUM_41;
static constexpr gpio_num_t I2C_MCU_SCL_PIN = GPIO_NUM_42;
static constexpr uint32_t I2C_FREQ_HZ = 100000;

static constexpr gpio_num_t OLED_RST_PIN = GPIO_NUM_21;
static constexpr gpio_num_t VEXT_CTRL_PIN = GPIO_NUM_36;
static constexpr int VEXT_ACTIVE_LEVEL = 0;

static constexpr gpio_num_t LED_PIN = GPIO_NUM_35;

static constexpr uint8_t OLED_ADDR_PRIMARY = 0x3C;
static constexpr uint8_t OLED_ADDR_SECONDARY = 0x3D;
static constexpr uint8_t MPU_ADDR_0 = 0x68;
static constexpr uint8_t MPU_ADDR_1 = 0x69;
static constexpr uint8_t MPU_WHOAMI_REG = 0x75;

static constexpr int OLED_WIDTH = 128;
static constexpr int OLED_PAGES = 8;
static uint8_t g_oled_fb[OLED_WIDTH * OLED_PAGES] = {0};
static uint8_t g_oled_addr = OLED_ADDR_PRIMARY;
static bool g_oled_ready = false;

// 5x7 font, solo ASCII visible.
static constexpr uint8_t FONT5X7[][5] = {
        {0x00,0x00,0x00,0x00,0x00},{0x00,0x00,0x5F,0x00,0x00},{0x00,0x07,0x00,0x07,0x00},{0x14,0x7F,0x14,0x7F,0x14},
        {0x24,0x2A,0x7F,0x2A,0x12},{0x23,0x13,0x08,0x64,0x62},{0x36,0x49,0x55,0x22,0x50},{0x00,0x05,0x03,0x00,0x00},
        {0x00,0x1C,0x22,0x41,0x00},{0x00,0x41,0x22,0x1C,0x00},{0x14,0x08,0x3E,0x08,0x14},{0x08,0x08,0x3E,0x08,0x08},
        {0x00,0x50,0x30,0x00,0x00},{0x08,0x08,0x08,0x08,0x08},{0x00,0x60,0x60,0x00,0x00},{0x20,0x10,0x08,0x04,0x02},
        {0x3E,0x51,0x49,0x45,0x3E},{0x00,0x42,0x7F,0x40,0x00},{0x42,0x61,0x51,0x49,0x46},{0x21,0x41,0x45,0x4B,0x31},
        {0x18,0x14,0x12,0x7F,0x10},{0x27,0x45,0x45,0x45,0x39},{0x3C,0x4A,0x49,0x49,0x30},{0x01,0x71,0x09,0x05,0x03},
        {0x36,0x49,0x49,0x49,0x36},{0x06,0x49,0x49,0x29,0x1E},{0x00,0x36,0x36,0x00,0x00},{0x00,0x56,0x36,0x00,0x00},
        {0x08,0x14,0x22,0x41,0x00},{0x14,0x14,0x14,0x14,0x14},{0x00,0x41,0x22,0x14,0x08},{0x02,0x01,0x51,0x09,0x06},
        {0x32,0x49,0x79,0x41,0x3E},{0x7E,0x11,0x11,0x11,0x7E},{0x7F,0x49,0x49,0x49,0x36},{0x3E,0x41,0x41,0x41,0x22},
        {0x7F,0x41,0x41,0x22,0x1C},{0x7F,0x49,0x49,0x49,0x41},{0x7F,0x09,0x09,0x09,0x01},{0x3E,0x41,0x49,0x49,0x7A},
        {0x7F,0x08,0x08,0x08,0x7F},{0x00,0x41,0x7F,0x41,0x00},{0x20,0x40,0x41,0x3F,0x01},{0x7F,0x08,0x14,0x22,0x41},
        {0x7F,0x40,0x40,0x40,0x40},{0x7F,0x02,0x0C,0x02,0x7F},{0x7F,0x04,0x08,0x10,0x7F},{0x3E,0x41,0x41,0x41,0x3E},
        {0x7F,0x09,0x09,0x09,0x06},{0x3E,0x41,0x51,0x21,0x5E},{0x7F,0x09,0x19,0x29,0x46},{0x46,0x49,0x49,0x49,0x31},
        {0x01,0x01,0x7F,0x01,0x01},{0x3F,0x40,0x40,0x40,0x3F},{0x1F,0x20,0x40,0x20,0x1F},{0x3F,0x40,0x38,0x40,0x3F},
        {0x63,0x14,0x08,0x14,0x63},{0x07,0x08,0x70,0x08,0x07},{0x61,0x51,0x49,0x45,0x43},{0x00,0x7F,0x41,0x41,0x00},
        {0x02,0x04,0x08,0x10,0x20},{0x00,0x41,0x41,0x7F,0x00},{0x04,0x02,0x01,0x02,0x04},{0x40,0x40,0x40,0x40,0x40},
        {0x00,0x03,0x05,0x00,0x00},{0x20,0x54,0x54,0x54,0x78},{0x7F,0x48,0x44,0x44,0x38},{0x38,0x44,0x44,0x44,0x20},
        {0x38,0x44,0x44,0x48,0x7F},{0x38,0x54,0x54,0x54,0x18},{0x08,0x7E,0x09,0x01,0x02},{0x0C,0x52,0x52,0x52,0x3E},
        {0x7F,0x08,0x04,0x04,0x78},{0x00,0x44,0x7D,0x40,0x00},{0x20,0x40,0x44,0x3D,0x00},{0x7F,0x10,0x28,0x44,0x00},
        {0x00,0x41,0x7F,0x40,0x00},{0x7C,0x04,0x18,0x04,0x78},{0x7C,0x08,0x04,0x04,0x78},{0x38,0x44,0x44,0x44,0x38},
        {0x7C,0x14,0x14,0x14,0x08},{0x08,0x14,0x14,0x18,0x7C},{0x7C,0x08,0x04,0x04,0x08},{0x48,0x54,0x54,0x54,0x20},
        {0x04,0x3F,0x44,0x40,0x20},{0x3C,0x40,0x40,0x20,0x7C},{0x1C,0x20,0x40,0x20,0x1C},{0x3C,0x40,0x30,0x40,0x3C},
        {0x44,0x28,0x10,0x28,0x44},{0x0C,0x50,0x50,0x50,0x3C},{0x44,0x64,0x54,0x4C,0x44},{0x00,0x08,0x36,0x41,0x00},
        {0x00,0x00,0x7F,0x00,0x00},{0x00,0x41,0x36,0x08,0x00},{0x08,0x04,0x08,0x10,0x08}
};

static void board_power_prepare() {
    gpio_config_t io_conf{};
    io_conf.pin_bit_mask = (1ULL << VEXT_CTRL_PIN) | (1ULL << OLED_RST_PIN) | (1ULL << LED_PIN);
    io_conf.mode = GPIO_MODE_OUTPUT;
    ESP_ERROR_CHECK(gpio_config(&io_conf));
    ESP_ERROR_CHECK(gpio_set_level(VEXT_CTRL_PIN, VEXT_ACTIVE_LEVEL));
    ESP_ERROR_CHECK(gpio_set_level(LED_PIN, 0));
    vTaskDelay(pdMS_TO_TICKS(20));
    ESP_ERROR_CHECK(gpio_set_level(OLED_RST_PIN, 0));
    vTaskDelay(pdMS_TO_TICKS(20));
    ESP_ERROR_CHECK(gpio_set_level(OLED_RST_PIN, 1));
    vTaskDelay(pdMS_TO_TICKS(20));
}

static bool init_i2c(i2c_port_t port, gpio_num_t sda, gpio_num_t scl, const char *label) {
    i2c_config_t conf{};
    conf.mode = I2C_MODE_MASTER;
    conf.sda_io_num = sda;
    conf.scl_io_num = scl;
    conf.sda_pullup_en = GPIO_PULLUP_ENABLE;
    conf.scl_pullup_en = GPIO_PULLUP_ENABLE;
    conf.master.clk_speed = I2C_FREQ_HZ;

    esp_err_t err = i2c_param_config(port, &conf);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "I2C %s config fallo: %s", label, esp_err_to_name(err));
        return false;
    }
    err = i2c_driver_install(port, I2C_MODE_MASTER, 0, 0, 0);
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "I2C %s install fallo: %s", label, esp_err_to_name(err));
        return false;
    }
    ESP_LOGI(TAG, "I2C %s listo (SDA=%d SCL=%d)", label, sda, scl);
    return true;
}

static bool probe_addr(i2c_port_t port, uint8_t addr) {
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    if (cmd == nullptr) return false;
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, static_cast<uint8_t>((addr << 1) | I2C_MASTER_WRITE), true);
    i2c_master_stop(cmd);
    esp_err_t err = i2c_master_cmd_begin(port, cmd, pdMS_TO_TICKS(50));
    i2c_cmd_link_delete(cmd);
    return err == ESP_OK;
}

static bool read_whoami(uint8_t addr, uint8_t *whoami) {
    uint8_t reg = MPU_WHOAMI_REG;
    return i2c_master_write_read_device(I2C_MCU_PORT, addr, &reg, 1, whoami, 1, pdMS_TO_TICKS(200)) == ESP_OK;
}

static void oled_clear() {
    std::memset(g_oled_fb, 0, sizeof(g_oled_fb));
}

static void oled_char(int x, int page, char c) {
    if (page < 0 || page >= OLED_PAGES || x < 0 || x + 5 >= OLED_WIDTH) return;
    if (c < 32 || c > 126) c = '?';
    const uint8_t *glyph = FONT5X7[c - 32];
    int base = page * OLED_WIDTH + x;
    for (int i = 0; i < 5; ++i) g_oled_fb[base + i] = glyph[i];
    g_oled_fb[base + 5] = 0x00;
}

static void oled_text(int x, int page, const char *text) {
    int cx = x;
    while (text && *text && cx + 5 < OLED_WIDTH) {
        oled_char(cx, page, *text++);
        cx += 6;
    }
}

static esp_err_t oled_cmd(uint8_t addr, uint8_t cmd) {
    uint8_t payload[2] = {0x00, cmd};
    return i2c_master_write_to_device(I2C_OLED_PORT, addr, payload, sizeof(payload), pdMS_TO_TICKS(100));
}

static esp_err_t oled_data(uint8_t addr, const uint8_t *data, size_t len) {
    uint8_t buffer[17];
    buffer[0] = 0x40;
    size_t offset = 0;
    while (offset < len) {
        size_t n = (len - offset > 16) ? 16 : (len - offset);
        std::memcpy(buffer + 1, data + offset, n);
        esp_err_t err = i2c_master_write_to_device(I2C_OLED_PORT, addr, buffer, n + 1, pdMS_TO_TICKS(100));
        if (err != ESP_OK) return err;
        offset += n;
    }
    return ESP_OK;
}

static bool init_oled() {
    const uint8_t seq[] = {0xAE,0xD5,0x80,0xA8,0x3F,0xD3,0x00,0x40,0x8D,0x14,0x20,0x00,0xA1,0xC8,0xDA,0x12,0x81,0x8F,0xD9,0xF1,0xDB,0x40,0xA4,0xA6,0x2E,0xAF};
    const uint8_t addrs[] = {OLED_ADDR_PRIMARY, OLED_ADDR_SECONDARY};
    for (uint8_t addr : addrs) {
        bool ok = true;
        for (uint8_t c : seq) {
            if (oled_cmd(addr, c) != ESP_OK) {
                ok = false;
                break;
            }
        }
        if (ok) {
            g_oled_addr = addr;
            g_oled_ready = true;
            ESP_LOGI(TAG, "OLED detectada en 0x%02X", addr);
            return true;
        }
    }
    ESP_LOGW(TAG, "OLED no detectada");
    return false;
}

static void oled_flush() {
    if (!g_oled_ready) return;
    for (int page = 0; page < OLED_PAGES; ++page) {
        oled_cmd(g_oled_addr, static_cast<uint8_t>(0xB0 + page));
        oled_cmd(g_oled_addr, 0x00);
        oled_cmd(g_oled_addr, 0x10);
        oled_data(g_oled_addr, &g_oled_fb[page * OLED_WIDTH], OLED_WIDTH);
    }
}

static void render_status(const char *l0, const char *l1, const char *l2, const char *l3) {
    oled_clear();
    oled_text(0, 0, "IMU DIAG");
    oled_text(0, 2, l0);
    oled_text(0, 3, l1);
    oled_text(0, 4, l2);
    oled_text(0, 5, l3);
    oled_flush();
}

extern "C" void app_main(void) {
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    board_power_prepare();
    init_i2c(I2C_OLED_PORT, I2C_OLED_SDA_PIN, I2C_OLED_SCL_PIN, "OLED");
    init_oled();
    init_i2c(I2C_MCU_PORT, I2C_MCU_SDA_PIN, I2C_MCU_SCL_PIN, "MCU");

    ESP_LOGI(TAG, "Diag IMU. Cableado esperado: VCC->3V3 GND->GND SDA->GPIO41 SCL->GPIO42");

    while (true) {
        bool seen68 = probe_addr(I2C_MCU_PORT, MPU_ADDR_0);
        bool seen69 = probe_addr(I2C_MCU_PORT, MPU_ADDR_1);
        uint8_t whoami = 0x00;
        bool who68 = seen68 && read_whoami(MPU_ADDR_0, &whoami);
        bool who69 = (!who68) && seen69 && read_whoami(MPU_ADDR_1, &whoami);

        char l0[24], l1[24], l2[24], l3[24];
        std::snprintf(l0, sizeof(l0), "BUS 68:%s 69:%s", seen68 ? "Y" : "N", seen69 ? "Y" : "N");
        if (who68) {
            std::snprintf(l1, sizeof(l1), "WHO 68:0x%02X", whoami);
            std::snprintf(l2, sizeof(l2), "IMU OK EN 0x68");
            std::snprintf(l3, sizeof(l3), "REVISA APP DESPUES");
            gpio_set_level(LED_PIN, 1);
            ESP_LOGI(TAG, "IMU detectado en 0x68, WHO_AM_I=0x%02X", whoami);
        } else if (who69) {
            std::snprintf(l1, sizeof(l1), "WHO 69:0x%02X", whoami);
            std::snprintf(l2, sizeof(l2), "IMU OK EN 0x69");
            std::snprintf(l3, sizeof(l3), "REVISA AD0");
            gpio_set_level(LED_PIN, 1);
            ESP_LOGI(TAG, "IMU detectado en 0x69, WHO_AM_I=0x%02X", whoami);
        } else {
            std::snprintf(l1, sizeof(l1), "WHO: SIN RESP");
            std::snprintf(l2, sizeof(l2), "REVISA CABLEADO");
            std::snprintf(l3, sizeof(l3), "SDA41 SCL42");
            gpio_set_level(LED_PIN, 0);
            ESP_LOGW(TAG, "Sin respuesta valida del IMU. addr68=%s addr69=%s", seen68 ? "si" : "no", seen69 ? "si" : "no");
        }

        render_status(l0, l1, l2, l3);
        vTaskDelay(pdMS_TO_TICKS(2000));
    }
}
