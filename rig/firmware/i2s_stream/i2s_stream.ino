// Stream raw INMP441 I2S samples to the host.
//
// Each block is an 8-byte header (magic, sequence) followed by 256 int32 samples,
// so the host can detect a dropped block instead of silently losing alignment.
// Samples are sent unmodified: the INMP441's 24 valid bits sit in the top of each
// 32-bit slot, and the shift to int16 is done on the host so no headroom decision
// is baked into the recording.
//
// Throughput is 16000 * 4 = 64 kB/s. On a plain ESP32 that goes over the USB-UART
// bridge at 2 Mbaud (~200 kB/s), which leaves ~3x headroom; if your bridge is an
// older CP2102 limited to 1 Mbaud, it still fits but watch capture.py's drop count.
// On an ESP32-S3 the same sketch uses the native USB CDC and the rate is ignored.
//
// Wiring (INMP441 -> ESP32):
//   VDD -> 3V3      GND -> GND      L/R -> GND (left slot)
//   SCK -> GPIO 26  WS -> GPIO 25   SD  -> GPIO 33
//
// Arduino-ESP32 core 3.x (ESP-IDF 5.x). Board: "ESP32 Dev Module".

#include <driver/i2s_std.h>

#if CONFIG_IDF_TARGET_ESP32S3
#define PIN_BCLK 4
#define PIN_WS 5
#define PIN_DIN 6
#else  // plain ESP32: any GPIO works via the matrix; 33 is input-capable
#define PIN_BCLK 26
#define PIN_WS 25
#define PIN_DIN 33
#endif

#define BAUD 2000000
#define BLOCK 256
#define SAMPLE_RATE 16000

static const uint32_t MAGIC = 0xA5A5A5A5;

static i2s_chan_handle_t rx;
static int32_t samples[BLOCK];
static struct __attribute__((packed)) {
  uint32_t magic;
  uint32_t seq;
} hdr;

void setup() {
  Serial.begin(BAUD);
  while (!Serial) {
  }

  i2s_chan_config_t chan = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  chan.dma_desc_num = 8;
  chan.dma_frame_num = BLOCK;
  ESP_ERROR_CHECK(i2s_new_channel(&chan, NULL, &rx));

  i2s_std_config_t std = {
      .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
      .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
      .gpio_cfg = {
          .mclk = I2S_GPIO_UNUSED,
          .bclk = (gpio_num_t)PIN_BCLK,
          .ws = (gpio_num_t)PIN_WS,
          .dout = I2S_GPIO_UNUSED,
          .din = (gpio_num_t)PIN_DIN,
          .invert_flags = {false, false, false},
      },
  };
  std.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;  // L/R tied to GND
  ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx, &std));
  ESP_ERROR_CHECK(i2s_channel_enable(rx));

  hdr.magic = MAGIC;
  hdr.seq = 0;
}

void loop() {
  size_t got = 0;
  if (i2s_channel_read(rx, samples, sizeof(samples), &got, portMAX_DELAY) != ESP_OK) return;
  if (got != sizeof(samples)) return;
  Serial.write((uint8_t *)&hdr, sizeof(hdr));
  Serial.write((uint8_t *)samples, sizeof(samples));
  hdr.seq++;
}
