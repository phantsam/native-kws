// On-chip test-set evaluation: stream log-mel features in over UART, send predictions back.
//
// The point of this firmware is to get a real accuracy number measured on the chip
// rather than on four embedded clips. The host sends one 40x101 float32 feature at a
// time; the device quantizes it with its own input exponent -- exactly the path
// app_main.cpp already uses for tv_input, so no scale is duplicated on the host --
// runs the model, and replies with the 12 int8 logits and the inference time.
//
// Drop this in beside app_main.cpp and point main/CMakeLists.txt at it instead
// (SRCS app_eval.cpp), keeping the same models/model.espdl embed.
//
// Protocol, little-endian:
//   host -> device : 0xA5A5A5A5 (u32), index (u32), 4040 x float32   [16168 B]
//   device -> host : "R <index> <pred> <us> <l0> ... <l11>\n"
// A request whose magic does not match is skipped a byte at a time until it does, so
// a desynchronised stream recovers instead of returning silent garbage.

#include <cstdio>
#include <cstring>

#include "dl_model_base.hpp"
#include "driver/uart.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"

#define MELS 40
#define FRAMES 101
#define CLASSES 12
#define FEAT (MELS * FRAMES)
#define MAGIC 0xA5A5A5A5u
#define UART UART_NUM_0
#define BAUD 2000000

extern const uint8_t model_espdl[] asm("_binary_model_espdl_start");
static const char *TAG = "kwseval";
static float feat[FEAT];

static void read_exact(uint8_t *dst, size_t n)
{
    size_t got = 0;
    while (got < n) {
        int r = uart_read_bytes(UART, dst + got, n - got, portMAX_DELAY);
        if (r > 0) got += r;
    }
}

extern "C" void app_main(void)
{
    uart_config_t cfg = {};
    cfg.baud_rate = BAUD;
    cfg.data_bits = UART_DATA_8_BITS;
    cfg.parity = UART_PARITY_DISABLE;
    cfg.stop_bits = UART_STOP_BITS_1;
    cfg.flow_ctrl = UART_HW_FLOWCTRL_DISABLE;
    cfg.source_clk = UART_SCLK_DEFAULT;
    ESP_ERROR_CHECK(uart_driver_install(UART, 4096, 4096, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(UART, &cfg));

    dl::Model *model = new dl::Model((const char *)model_espdl, fbs::MODEL_LOCATION_IN_FLASH_RODATA, 0,
                                     dl::MEMORY_MANAGER_GREEDY, nullptr, false);
    dl::TensorBase *in = model->get_inputs().begin()->second;
    dl::TensorBase *out = model->get_outputs().begin()->second;
    if (!in || !out || !in->data || !out->data) {
        ESP_LOGE(TAG, "activation arena did not fit; stopping");
        return;
    }
    if (in->get_size() != FEAT || out->get_size() != CLASSES) {
        ESP_LOGE(TAG, "tensor sizes: in %d (want %d) out %d (want %d)", in->get_size(), FEAT,
                 out->get_size(), CLASSES);
        return;
    }
    const float in_scale = DL_RESCALE(in->get_exponent());

    // Printed once, before the loop, so the host can log what it is talking to.
    // No logging inside the loop: it shares UART0 with the replies.
    ESP_LOGI(TAG, "ready: input exp %d, output exp %d, free internal %u B", in->get_exponent(),
             out->get_exponent(), (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));

    char line[256];
    for (;;) {
        uint32_t magic = 0, index = 0;
        uint8_t b;
        // resynchronise byte by byte until the magic word lands
        while (magic != MAGIC) {
            read_exact(&b, 1);
            magic = (magic >> 8) | ((uint32_t)b << 24);
        }
        read_exact((uint8_t *)&index, 4);
        read_exact((uint8_t *)feat, sizeof(feat));

        int8_t *dst = (int8_t *)in->data;
        for (int i = 0; i < FEAT; i++) dst[i] = dl::quantize<int8_t>(feat[i], in_scale);

        int64_t t0 = esp_timer_get_time();
        model->run();
        int64_t dt = esp_timer_get_time() - t0;

        const int8_t *q = (const int8_t *)out->data;
        int best = 0;
        for (int c = 1; c < CLASSES; c++)
            if (q[c] > q[best]) best = c;

        int n = snprintf(line, sizeof(line), "R %u %d %lld", (unsigned)index, best, (long long)dt);
        for (int c = 0; c < CLASSES; c++)
            n += snprintf(line + n, sizeof(line) - n, " %d", (int)q[c]);
        n += snprintf(line + n, sizeof(line) - n, "\n");
        uart_write_bytes(UART, line, n);
    }
}
